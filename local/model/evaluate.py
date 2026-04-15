#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py — Evaluation script for AIVECTOR_ConformerVEmbedding_SD_JOINT (AVSD model).

Evaluation dataset structure (one directory per session):
    <eval_dir>/
        <session_id>/
            audio_features.npy         # (T, 40)   float32
            diarization_labels.npy     # (T, N)    int or float binary
            nframes.txt                # single integer T
            video_features/
                spk_0.npy              # (T, 96, 96) float32
                spk_1.npy
                ...

Model was trained with output_speaker=4.  Sessions with a different number of
speakers are handled as follows:
  - N <  MODEL_SPEAKERS : padded with zero video frames; extra heads are masked.
  - N == MODEL_SPEAKERS : standard inference.
  - N >  MODEL_SPEAKERS : two-pass strategy — each group of MODEL_SPEAKERS is
                          run independently and predictions are concatenated.

Metrics reported per session and globally:
  - BCE loss (same criterion as training)
  - Frame-level binary accuracy per speaker
  - Precision / Recall / F1 per speaker
  - Diarization Error Rate (DER) — permutation-invariant, via Hungarian matching:
        DER = (FA + MISS + CONF) / total_reference_speech_frames × 100
  - JER (Jaccard Error Rate) as an auxiliary metric
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# ─── path setup ─────────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from avsd_net import AIVECTOR_ConformerVEmbedding_SD_JOINT  # noqa: E402

# ════════════════════════════════════════════════════════════════════════════
# DEFAULT CONFIGURATION  (all overridable via CLI args)
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_CONFIG = {
    "input_dim": 40,
    "average_pooling": 3,
    "speaker_embedding_dim": 100,
    "output_speaker": 4,          # must match the trained checkpoint
    "audio_output_dim": 256,
    "video_embedding_dim": 256,
    "num_attention_heads": 4,
    "decoder_hidden_dim": 256,
    "dropout": 0.0,               # disable dropout at eval time
}

CHUNK_FRAMES   = 200   # 8 s @ 25 fps  — same as training
STRIDE_FRAMES  = 200   # non-overlapping chunks for eval (no data leakage between chunks)
THRESHOLD      = 0.5   # sigmoid threshold for binary activity decision
FPS            = 25


# ════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════

def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Configure root logger to write to both console and a file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "eval.log")

    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    logger = logging.getLogger("avsd_eval")
    logger.info("Logging initialised — writing to %s", log_path)
    return logger


# ════════════════════════════════════════════════════════════════════════════
# DATASET / SESSION UTILITIES
# ════════════════════════════════════════════════════════════════════════════

class EvalSession:
    """Loads a single evaluation session from disk and yields fixed-length chunks."""

    def __init__(self, session_dir: Path, model_speakers: int):
        self.session_id  = session_dir.name
        self.session_dir = session_dir
        self.model_speakers = model_speakers

        # ── mandatory files ──────────────────────────────────────────────
        audio_path  = session_dir / "audio_features.npy"
        labels_path = session_dir / "diarization_labels.npy"
        nframes_path = session_dir / "nframes.txt"

        if not (audio_path.exists() and labels_path.exists() and nframes_path.exists()):
            raise FileNotFoundError(
                f"Session {self.session_id}: missing audio_features.npy / "
                "diarization_labels.npy / nframes.txt"
            )

        self.audio_features      = np.load(audio_path)           # (T, 40)
        self.diarization_labels  = np.load(labels_path)          # (T, N_true)
        with open(nframes_path) as f:
            self.total_frames = int(f.read().strip())

        # ── video features ───────────────────────────────────────────────
        vfeat_dir = session_dir / "video_features"
        spk_files = sorted(vfeat_dir.glob("spk_*.npy")) if vfeat_dir.exists() else []

        self.speaker_ids     = [p.stem for p in spk_files]   # e.g. ['spk_0','spk_1',...]
        self.n_true_speakers = len(self.speaker_ids)

        # Store per-speaker video arrays.
        # NOTE: video arrays may have FEWER rows than self.total_frames if the
        # face detector dropped frames or if audio/video were extracted at
        # different rates.  We record each speaker's actual length and clamp
        # iteration to the minimum across all modalities so the model always
        # receives tensors with the same temporal dimension.
        self.video_features: List[np.ndarray] = [np.load(p) for p in spk_files]

        # Detect per-speaker video lengths and log any mismatch.
        video_lengths = [len(vf) for vf in self.video_features]
        self.video_min_frames = min(video_lengths) if video_lengths else self.total_frames
        if self.video_min_frames != self.total_frames:
            import warnings
            warnings.warn(
                f"Session {self.session_id}: audio has {self.total_frames} frames but "
                f"video speaker lengths are {video_lengths}. "
                f"Clamping to min={self.video_min_frames} frames."
            )

        # effective_frames: the safe upper bound across ALL modalities
        self.effective_frames = min(self.total_frames, self.video_min_frames)

        # Infer spatial dimensions from the first video array (may not be 96x96)
        if self.video_features:
            vf0 = self.video_features[0]
            self.video_h = vf0.shape[1] if vf0.ndim >= 2 else 96
            self.video_w = vf0.shape[2] if vf0.ndim >= 3 else self.video_h
        else:
            self.video_h = self.video_w = 96

        # Ensure label matrix has enough columns
        if self.diarization_labels.ndim == 1:
            self.diarization_labels = self.diarization_labels[:, None]
        while self.diarization_labels.shape[1] < self.n_true_speakers:
            self.diarization_labels = np.concatenate(
                [self.diarization_labels,
                 np.zeros((self.total_frames, 1), dtype=self.diarization_labels.dtype)],
                axis=1
            )

    # ── chunk generator ───────────────────────────────────────────────────

    def iter_chunks(
        self,
        chunk_frames: int = CHUNK_FRAMES,
        stride_frames: int = STRIDE_FRAMES,
    ):
        """
        Yield dicts with tensors for one chunk:
            audio   : (40, chunk_frames)                — right-zero-padded
            video   : (n_true_speakers, nframes, H, W)  — NOT padded (real frames only)
            labels  : (chunk_frames, n_true_speakers)   — right-zero-padded
            nframes : int  — number of REAL frames in this chunk (may be < chunk_frames
                             for the last chunk or when video is shorter than audio)
            start_f : int  — global start frame (for result assembly)

        IMPORTANT: `nframes` reflects the minimum real frames across audio AND video
        modalities.  The video array in the yielded dict already has exactly `nframes`
        rows.  The audio tensor is padded to `chunk_frames` but the caller should slice
        it to `nframes` before feeding to the model.
        """
        H, W = self.video_h, self.video_w
        start = 0
        # Iterate only up to effective_frames (min of audio and all video lengths)
        while start < self.effective_frames:
            # End bounded by both audio length and video length
            end    = min(start + chunk_frames, self.effective_frames)
            actual = end - start  # real frames this chunk

            # ── audio: pad to chunk_frames so downstream batching is uniform ─
            audio_chunk = self.audio_features[start:end].T          # (40, actual)
            if actual < chunk_frames:
                pad = chunk_frames - actual
                audio_chunk = np.concatenate(
                    [audio_chunk, np.zeros((40, pad), dtype=audio_chunk.dtype)], axis=1
                )

            # ── video: yield ONLY real frames (no padding) ───────────────
            # The model always receives video sliced to `nframes`; padding zeros
            # would bias the visual encoder.  _run_inference_pass will slice
            # audio to nframes as well to guarantee matching temporal sizes.
            video_chunks = []
            for vf in self.video_features:
                # Guard against individual speaker arrays shorter than `end`
                spk_end = min(end, len(vf))
                actual  = min(actual, spk_end - start)   # tighten actual if needed

            for vf in self.video_features:
                spk_end = min(end, len(vf))
                vc = vf[start:spk_end]
                if len(vc) < actual:
                    # One speaker's video is shorter than the tightened actual;
                    # pad with zeros so all speakers have the same length.
                    pad_rows = actual - len(vc)
                    vc = np.concatenate(
                        [vc, np.zeros((pad_rows, H, W), dtype=vc.dtype)], axis=0
                    )
                video_chunks.append(vc[:actual])           # ensure exact length

            # ── labels ───────────────────────────────────────────────────
            lbl_chunk = self.diarization_labels[start : start + actual]   # (actual, N_true)
            if actual < chunk_frames:
                pad = chunk_frames - actual
                lbl_chunk = np.concatenate(
                    [lbl_chunk, np.zeros((pad, self.n_true_speakers), dtype=lbl_chunk.dtype)],
                    axis=0
                )

            yield {
                "audio":   torch.FloatTensor(audio_chunk),           # (40, chunk_frames)
                "video":   np.array(video_chunks),                   # (N_true, actual, H, W)
                "labels":  torch.FloatTensor(lbl_chunk),             # (chunk_frames, N_true)
                "nframes": actual,
                "start_f": start,
            }
            start += stride_frames


def discover_sessions(eval_dir: str) -> List[Path]:
    """Return sorted list of session subdirectories."""
    root = Path(eval_dir)
    sessions = sorted(
        [d for d in root.iterdir() if d.is_dir()],
        key=lambda p: p.name,
    )
    return sessions


# ════════════════════════════════════════════════════════════════════════════
# MODEL UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def load_model(
    checkpoint_path: str,
    config: Dict,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[AIVECTOR_ConformerVEmbedding_SD_JOINT, Dict]:
    """Load model from a .pth checkpoint (complete or state-dict only)."""
    logger.info("Loading checkpoint: %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Support both full checkpoint dicts and plain state dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        saved_epoch = checkpoint.get("epoch", "?")
        saved_loss  = checkpoint.get("loss",  float("nan"))
        saved_config = checkpoint.get("config", {})
        logger.info(
            "Checkpoint metadata — epoch: %s  train_loss: %.4f",
            saved_epoch, saved_loss,
        )
        if saved_config:
            logger.info("Checkpoint config: %s", saved_config)
            # Merge saved config (saved values take priority for architecture keys)
            for k in ("input_dim", "output_speaker", "audio_output_dim",
                      "video_embedding_dim", "num_attention_heads", "decoder_hidden_dim"):
                if k in saved_config:
                    config[k] = saved_config[k]
    else:
        state_dict = checkpoint
        logger.info("Checkpoint is a plain state dict.")

    model = AIVECTOR_ConformerVEmbedding_SD_JOINT(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d):  %s …", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d):  %s …", len(unexpected), unexpected[:5])

    model.to(device)
    model.eval()
    logger.info(
        "Model loaded on %s  |  params: %s",
        device,
        f"{sum(p.numel() for p in model.parameters()):,}",
    )
    return model, config


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE — handles speaker-count mismatches via a two-pass strategy
# ════════════════════════════════════════════════════════════════════════════

def _run_inference_pass(
    model: nn.Module,
    audio_tensor: torch.Tensor,         # (1, 40, T)
    video_np: np.ndarray,               # (K, T, 96, 96)  K <= model_speakers
    nframes: int,
    model_speakers: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run one forward pass with K speakers (K <= model_speakers).
    Pads video to model_speakers if K < model_speakers.

    Returns:
        logits : (T, K)  raw logits for the K real speakers
    """
    K = video_np.shape[0]
    assert K <= model_speakers, "Internal error: K > model_speakers in _run_inference_pass"

    # Slice audio to nframes so it matches the video temporal dimension.
    # audio_tensor may be padded to chunk_frames (e.g. 200) but video_np has
    # exactly nframes rows — the model's CASA_Module derives T from the audio
    # shape, so a mismatch causes a reshape RuntimeError.
    audio_sliced = audio_tensor[:, :, :nframes]   # (1, 40, nframes)

    # Pad video speakers to model_speakers
    nframes_vid = video_np.shape[1]   # should equal nframes after iter_chunks fix
    H = video_np.shape[2] if video_np.ndim >= 3 else 96
    W = video_np.shape[3] if video_np.ndim >= 4 else H
    if K < model_speakers:
        pad = np.zeros(
            (model_speakers - K, nframes_vid, H, W), dtype=video_np.dtype
        )
        video_np = np.concatenate([video_np, pad], axis=0)

    video_tensor = torch.FloatTensor(video_np).unsqueeze(0).to(device)  # (1, model_speakers, nframes, H, W)
    audio_embed  = torch.zeros(1, model_speakers, 100, device=device)

    # CRITICAL: patch num_speakers in the visual encoder wrapper so that the
    # reshape inside ASD_Wrapper_Visual_Encoder.forward works correctly.
    v_embedding_wrapper = model.v_embedding
    original_ns = getattr(v_embedding_wrapper, "num_speakers", None)
    setattr(v_embedding_wrapper, "num_speakers", model_speakers)

    with torch.no_grad():
        outputs = model(audio_sliced, audio_embed, video_tensor, [nframes])

    if original_ns is not None:
        setattr(v_embedding_wrapper, "num_speakers", original_ns)  # restore

    # outputs is a list of len model_speakers, each element shape: (sum_valid_frames_across_batch, 1)
    # For batch_size=1, each element has shape (nframes, 1)
    # We only care about the first K speakers (real speakers)
    logit_list = []
    for spk_idx in range(K):
        spk_logits = outputs[spk_idx]           # (nframes, 1)
        if spk_logits is not None and len(spk_logits) > 0:
            logit_list.append(spk_logits.squeeze(-1).cpu().numpy())  # (nframes,)
        else:
            logit_list.append(np.full(nframes, -10.0, dtype=np.float32))

    return np.stack(logit_list, axis=1)          # (nframes, K)


def infer_session_chunk(
    model: nn.Module,
    audio_tensor: torch.Tensor,     # (1, 40, T)
    video_np: np.ndarray,           # (N_true, T, 96, 96)
    nframes: int,
    model_speakers: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run inference for a chunk with N_true speakers.

    If N_true > model_speakers, speakers are processed in groups of model_speakers
    and results concatenated across the speaker dimension.

    Returns:
        logits : (nframes, N_true)  raw logits
    """
    N_true = video_np.shape[0]
    all_logits = []

    for group_start in range(0, N_true, model_speakers):
        group_end  = min(group_start + model_speakers, N_true)
        group_video = video_np[group_start:group_end]                # (G, T, 96, 96)
        G = group_end - group_start

        group_logits = _run_inference_pass(
            model, audio_tensor, group_video, nframes, model_speakers, device
        )                                                             # (nframes, G)
        all_logits.append(group_logits)

    return np.concatenate(all_logits, axis=1)                       # (nframes, N_true)


# ════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════

def compute_bce_loss(
    logits: np.ndarray,   # (T, N)
    labels: np.ndarray,   # (T, N)
) -> float:
    """Frame-level average BCE loss (same criterion as training)."""
    logits_t = torch.FloatTensor(logits)
    labels_t = torch.FloatTensor(labels)
    loss = nn.BCEWithLogitsLoss()(logits_t, labels_t)
    return loss.item()


def compute_per_speaker_metrics(
    predictions: np.ndarray,   # (T, N) binary
    labels: np.ndarray,        # (T, N) binary
) -> List[Dict]:
    """Compute precision, recall, F1, accuracy for each speaker independently."""
    N = predictions.shape[1]
    results = []
    for i in range(N):
        tp = float(np.sum((predictions[:, i] == 1) & (labels[:, i] == 1)))
        fp = float(np.sum((predictions[:, i] == 1) & (labels[:, i] == 0)))
        fn = float(np.sum((predictions[:, i] == 0) & (labels[:, i] == 1)))
        tn = float(np.sum((predictions[:, i] == 0) & (labels[:, i] == 0)))
        total = tp + fp + fn + tn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        accuracy  = (tp + tn) / total if total > 0 else 0.0

        results.append({
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": f1, "accuracy": accuracy,
        })
    return results


def compute_der(
    predictions: np.ndarray,   # (T, N) binary
    labels: np.ndarray,        # (T, N) binary
    use_hungarian: bool = True,
) -> Dict:
    """
    Compute Diarization Error Rate (DER) using the NIST per-speaker-per-frame standard.

    With use_hungarian=True the optimal permutation of predicted speaker columns
    to reference speaker columns is found via the Hungarian algorithm, which is the
    permutation-invariant formulation standard in speaker diarization.

    NIST DER formula (per-speaker-per-frame):
        At each frame t:
            FA(t)   = max(0, n_hyp(t) − n_ref(t))
            MISS(t) = max(0, n_ref(t) − n_hyp(t))
            CONF(t) = min(n_ref(t), n_hyp(t)) − n_correct(t)

        DER = (ΣFA + ΣMISS + ΣCONF) / Σn_ref(t) × 100

    Denominator is total reference SPEAKER-FRAMES (not just speech frames),
    which is the correct NIST standard and avoids artificial 100% DER in
    multi-speaker sessions where any one column mismatch per frame would
    otherwise inflate CONF to equal the entire denominator.

    Also returns JER (Jaccard Error Rate):
        JER = mean over speakers of  1 − IoU_k
    """
    T, N_ref  = labels.shape
    _,  N_hyp = predictions.shape

    # ── permutation matching ─────────────────────────────────────────────
    if use_hungarian and N_ref > 0 and N_hyp > 0:
        # Cost matrix: 1 − frame overlap for each (ref, hyp) speaker pair
        cost = np.zeros((N_ref, N_hyp), dtype=np.float64)
        for r in range(N_ref):
            for h in range(N_hyp):
                cost[r, h] = -float(np.sum(labels[:, r] * predictions[:, h]))
        row_ind, col_ind = linear_sum_assignment(cost)

        # Build aligned prediction matrix (same shape as labels)
        aligned_preds = np.zeros_like(labels)
        for r, h in zip(row_ind, col_ind):
            aligned_preds[:, r] = predictions[:, h]
    else:
        min_n = min(N_ref, N_hyp)
        aligned_preds = np.zeros_like(labels)
        aligned_preds[:, :min_n] = predictions[:, :min_n]

    # ── FA, MISS, CONF  (NIST per-speaker-per-frame) ────────────────────
    # Count active speakers per frame for both ref and hyp.
    n_ref_t  = labels.sum(axis=1).astype(np.float64)        # (T,) ref speaker count
    n_hyp_t  = aligned_preds.sum(axis=1).astype(np.float64) # (T,) hyp speaker count
    # Correctly matched speakers per frame (after permutation alignment)
    n_correct_t = (labels * aligned_preds).sum(axis=1).astype(np.float64)  # (T,)

    # Per-frame NIST DER components:
    #   FA(t)   = excess hypothesis speakers above reference
    #   MISS(t) = excess reference speakers above hypothesis
    #   CONF(t) = speaker-slot confusions (both have speech but wrong identity)
    fa_t   = np.maximum(0.0, n_hyp_t - n_ref_t)
    miss_t = np.maximum(0.0, n_ref_t - n_hyp_t)
    conf_t = np.minimum(n_ref_t, n_hyp_t) - n_correct_t
    conf_t = np.maximum(0.0, conf_t)   # guard against float rounding below zero

    fa   = float(fa_t.sum())
    miss = float(miss_t.sum())
    conf = float(conf_t.sum())

    # Denominator: total reference SPEAKER-FRAMES (standard NIST DER)
    total_ref    = float(n_ref_t.sum())
    total_frames = float(T)

    der = (fa + miss + conf) / total_ref * 100.0 if total_ref > 0 else float("nan")

    # ── JER ──────────────────────────────────────────────────────────────
    jer_values = []
    for r in range(N_ref):
        tp = float(np.sum(labels[:, r] * aligned_preds[:, r]))
        fp = float(np.sum((1 - labels[:, r]) * aligned_preds[:, r]))
        fn = float(np.sum(labels[:, r] * (1 - aligned_preds[:, r])))
        denom = tp + fp + fn
        jer_values.append(1.0 - tp / denom if denom > 0 else 1.0)
    jer = float(np.mean(jer_values)) * 100.0 if jer_values else float("nan")

    return {
        "DER":        round(der, 4),
        "JER":        round(jer, 4),
        "FA":         fa,
        "MISS":       miss,
        "CONF":       conf,
        "total_ref_speech_frames": total_ref,
        "total_frames": total_frames,
    }


# ════════════════════════════════════════════════════════════════════════════
# SESSION-LEVEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def evaluate_session(
    session: EvalSession,
    model: nn.Module,
    model_speakers: int,
    device: torch.device,
    logger: logging.Logger,
    chunk_frames: int = CHUNK_FRAMES,
    stride_frames: int = STRIDE_FRAMES,
    threshold: float = THRESHOLD,
    thresholds: Optional[List[float]] = None,
) -> Dict:
    """
    Run inference over all chunks of a session, aggregate, compute metrics.

    Returns a dict with all metrics for the session.
    """
    logger.info(
        "  Session %-30s | audio=%d  video_min=%d  effective=%d frames | %d speakers",
        session.session_id,
        session.total_frames,
        session.video_min_frames,
        session.effective_frames,
        session.n_true_speakers,
    )

    # eff_frames: safe upper bound across audio and all video tracks
    N_true     = session.n_true_speakers
    eff_frames = session.effective_frames

    all_logits = np.zeros((eff_frames, N_true), dtype=np.float32)  # accumulator; uncovered frames default to 0
    all_labels = np.zeros((eff_frames, N_true), dtype=np.float32)
    frame_hits = np.zeros(eff_frames, dtype=np.int32)

    chunk_losses = []
    chunk_count  = 0

    for chunk in session.iter_chunks(chunk_frames, stride_frames):
        # audio is padded to chunk_frames; video already has exactly nframes rows
        audio_np  = chunk["audio"].unsqueeze(0).to(device)   # (1, 40, chunk_frames)
        video_np  = chunk["video"]                            # (N_true, nframes, H, W)
        labels_np = chunk["labels"].numpy()                   # (chunk_frames, N_true)
        nframes   = chunk["nframes"]
        start_f   = chunk["start_f"]
        end_f     = start_f + nframes

        # ── forward pass ──────────────────────────────────────────────
        # Pass video_np directly — iter_chunks guarantees it has nframes rows.
        # _run_inference_pass slices audio to nframes internally.
        logits_np = infer_session_chunk(
            model, audio_np, video_np, nframes, model_speakers, device,
        )                                                       # (nframes, N_true)

        # ── accumulate (overlapping chunks averaged via hit counts) ───
        all_logits[start_f:end_f] += logits_np
        all_labels[start_f:end_f]  = labels_np[:nframes]
        frame_hits[start_f:end_f] += 1

        # ── per-chunk BCE loss ────────────────────────────────────────
        chunk_loss = compute_bce_loss(logits_np, labels_np[:nframes])
        chunk_losses.append(chunk_loss)
        chunk_count += 1

    # Average accumulated logits where frames were covered by multiple chunks
    mask = frame_hits > 0
    all_logits[mask] /= frame_hits[mask, None]

    logits = all_logits[:eff_frames]
    labels = all_labels[:eff_frames]

    session_loss      = float(np.mean(chunk_losses)) if chunk_losses else float("nan")

    thresholds_to_eval = thresholds if thresholds is not None else [threshold]
    thresholds_to_eval = [float(t) for t in thresholds_to_eval]
    threshold_results: Dict[str, Dict] = {}

    probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

    for thr in thresholds_to_eval:
        preds = (probs >= thr).astype(np.float32)
        per_spk_metrics = compute_per_speaker_metrics(preds, labels)
        der_metrics = compute_der(preds, labels, use_hungarian=True)

        mean_f1 = float(np.mean([m["f1"] for m in per_spk_metrics]))
        mean_acc = float(np.mean([m["accuracy"] for m in per_spk_metrics]))

        threshold_key = f"{thr:.6f}".rstrip("0").rstrip(".")
        threshold_results[threshold_key] = {
            "threshold": thr,
            "mean_f1": round(mean_f1, 4),
            "mean_accuracy": round(mean_acc, 4),
            "per_speaker": per_spk_metrics,
            **der_metrics,
        }

    primary_threshold_key = f"{thresholds_to_eval[0]:.6f}".rstrip("0").rstrip(".")
    primary_metrics = threshold_results[primary_threshold_key]

    result = {
        "session_id":        session.session_id,
        "total_frames":      session.total_frames,
        "effective_frames":  eff_frames,
        "video_min_frames":  session.video_min_frames,
        "n_true_speakers":   N_true,
        "n_chunks":          chunk_count,
        "bce_loss":          round(session_loss, 6),
        "threshold":         primary_metrics["threshold"],
        "threshold_metrics": threshold_results,
        "mean_f1":           primary_metrics["mean_f1"],
        "mean_accuracy":     primary_metrics["mean_accuracy"],
        "per_speaker":       primary_metrics["per_speaker"],
        "DER":               primary_metrics["DER"],
        "JER":               primary_metrics["JER"],
        "FA":                primary_metrics["FA"],
        "MISS":              primary_metrics["MISS"],
        "CONF":              primary_metrics["CONF"],
        "total_ref_speech_frames": primary_metrics["total_ref_speech_frames"],
        "total_frames_eval":  primary_metrics["total_frames"],
    }

    if len(thresholds_to_eval) == 1:
        logger.info(
            "    loss=%.4f  threshold=%.3f  DER=%.2f%%  JER=%.2f%%  F1=%.4f  "
            "FA=%d  MISS=%d  CONF=%d",
            session_loss,
            primary_metrics["threshold"],
            primary_metrics["DER"] if not np.isnan(primary_metrics["DER"]) else -1,
            primary_metrics["JER"] if not np.isnan(primary_metrics["JER"]) else -1,
            primary_metrics["mean_f1"],
            int(primary_metrics["FA"]),
            int(primary_metrics["MISS"]),
            int(primary_metrics["CONF"]),
        )
    else:
        logger.info(
            "    loss=%.4f  thresholds=%d  primary=%.3f  DER=%.2f%%  JER=%.2f%%  F1=%.4f",
            session_loss,
            len(thresholds_to_eval),
            primary_metrics["threshold"],
            primary_metrics["DER"] if not np.isnan(primary_metrics["DER"]) else -1,
            primary_metrics["JER"] if not np.isnan(primary_metrics["JER"]) else -1,
            primary_metrics["mean_f1"],
        )

    return result


# ════════════════════════════════════════════════════════════════════════════
# CHECKPOINT / RESULT SAVING
# ════════════════════════════════════════════════════════════════════════════

def save_session_checkpoint(result: Dict, output_dir: str):
    """Save per-session result JSON to <output_dir>/sessions/<session_id>.json."""
    session_dir = os.path.join(output_dir, "sessions")
    os.makedirs(session_dir, exist_ok=True)
    path = os.path.join(session_dir, f"{result['session_id']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=float)


def save_global_metrics(metrics: Dict, output_dir: str, logger: logging.Logger):
    """Save aggregated metrics to <output_dir>/global_metrics.json."""
    path = os.path.join(output_dir, "global_metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    logger.info("Global metrics saved to %s", path)


# ════════════════════════════════════════════════════════════════════════════
# GLOBAL METRIC AGGREGATION
# ════════════════════════════════════════════════════════════════════════════

def aggregate_global_metrics(session_results: List[Dict], threshold_key: Optional[str] = None) -> Dict:
    """
    Aggregate per-session results into global metrics.

    DER / JER are computed globally (weighted by total reference speech frames),
    loss and F1 are macro-averaged over sessions.
    """
    if not session_results:
        return {}

    if threshold_key is None:
        total_fa   = sum(r["FA"]   for r in session_results)
        total_miss = sum(r["MISS"] for r in session_results)
        total_conf = sum(r["CONF"] for r in session_results)
        total_ref  = sum(r["total_ref_speech_frames"] for r in session_results)

        global_der = (
            (total_fa + total_miss + total_conf) / total_ref * 100.0
            if total_ref > 0
            else float("nan")
        )

        jer_values = [
            r["JER"] for r in session_results if not np.isnan(r["JER"])
        ]
        global_jer = float(np.mean(jer_values)) if jer_values else float("nan")

        losses  = [r["bce_loss"] for r in session_results if not np.isnan(r["bce_loss"])]
        f1s     = [r["mean_f1"]  for r in session_results]
        accs    = [r["mean_accuracy"] for r in session_results]

        return {
            "n_sessions":          len(session_results),
            "global_DER":          round(global_der, 4),
            "global_JER":          round(global_jer, 4),
            "macro_bce_loss":      round(float(np.mean(losses)), 6) if losses else float("nan"),
            "macro_mean_f1":       round(float(np.mean(f1s)),    4),
            "macro_mean_accuracy": round(float(np.mean(accs)),   4),
            "total_FA":            total_fa,
            "total_MISS":          total_miss,
            "total_CONF":          total_conf,
            "total_ref_speech_frames": total_ref,
            "per_session_DER": {
                r["session_id"]: r["DER"] for r in session_results
            },
            "per_session_loss": {
                r["session_id"]: r["bce_loss"] for r in session_results
            },
        }

    total_fa   = 0.0
    total_miss = 0.0
    total_conf = 0.0
    total_ref  = 0.0
    jer_values = []
    losses     = []
    f1s        = []
    accs       = []
    per_session_der = {}
    per_session_loss = {}

    for r in session_results:
        threshold_results = r.get("threshold_metrics", {})
        if threshold_key not in threshold_results:
            continue
        metrics = threshold_results[threshold_key]

        total_fa += metrics["FA"]
        total_miss += metrics["MISS"]
        total_conf += metrics["CONF"]
        total_ref += metrics["total_ref_speech_frames"]

        if not np.isnan(metrics["JER"]):
            jer_values.append(metrics["JER"])
        if not np.isnan(r.get("bce_loss", float("nan"))):
            losses.append(r["bce_loss"])
        f1s.append(metrics["mean_f1"])
        accs.append(metrics["mean_accuracy"])
        per_session_der[r["session_id"]] = metrics["DER"]
        per_session_loss[r["session_id"]] = r["bce_loss"]

    global_der = (
        (total_fa + total_miss + total_conf) / total_ref * 100.0
        if total_ref > 0
        else float("nan")
    )
    global_jer = float(np.mean(jer_values)) if jer_values else float("nan")

    return {
        "threshold_key":       threshold_key,
        "n_sessions":          len(per_session_der),
        "global_DER":          round(global_der, 4),
        "global_JER":          round(global_jer, 4),
        "macro_bce_loss":      round(float(np.mean(losses)), 6) if losses else float("nan"),
        "macro_mean_f1":       round(float(np.mean(f1s)),    4),
        "macro_mean_accuracy": round(float(np.mean(accs)),   4),
        "total_FA":            total_fa,
        "total_MISS":          total_miss,
        "total_CONF":          total_conf,
        "total_ref_speech_frames": total_ref,
        "per_session_DER":     per_session_der,
        "per_session_loss":    per_session_loss,
    }


def aggregate_threshold_sweep(session_results: List[Dict], thresholds: List[float]) -> Dict:
    """Aggregate metrics for each threshold from cached session results."""
    sweep = {}
    best_threshold_key = None
    best_der = float("inf")

    for threshold in thresholds:
        threshold_key = f"{float(threshold):.6f}".rstrip("0").rstrip(".")
        metrics = aggregate_global_metrics(session_results, threshold_key=threshold_key)
        sweep[threshold_key] = metrics

        der_value = metrics.get("global_DER", float("nan"))
        if not np.isnan(der_value) and der_value < best_der:
            best_der = der_value
            best_threshold_key = threshold_key

    return {
        "thresholds": thresholds,
        "metrics_by_threshold": sweep,
        "best_threshold": best_threshold_key,
        "best_metrics": sweep.get(best_threshold_key, {}),
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate AVSD Model — AIVECTOR_ConformerVEmbedding_SD_JOINT"
    )
    p.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to the .pth checkpoint file (epoch-20 model).",
    )
    p.add_argument(
        "--eval_dir", "-e", required=True,
        help="Root directory containing eval session subdirectories.",
    )
    p.add_argument(
        "--output_dir", "-o", default="eval_results",
        help="Directory for logs, per-session JSONs, and global metrics. Default: ./eval_results",
    )
    p.add_argument(
        "--output_speaker", type=int, default=DEFAULT_CONFIG["output_speaker"],
        help=f"Number of speakers the model was trained with. Default: {DEFAULT_CONFIG['output_speaker']}",
    )
    p.add_argument(
        "--chunk_frames", type=int, default=CHUNK_FRAMES,
        help=f"Frames per inference chunk. Default: {CHUNK_FRAMES}",
    )
    p.add_argument(
        "--stride_frames", type=int, default=STRIDE_FRAMES,
        help=f"Frame stride between chunks. Default: {STRIDE_FRAMES} (non-overlapping).",
    )
    p.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"Sigmoid threshold for binary activity prediction. Default: {THRESHOLD}",
    )
    p.add_argument(
        "--thresholds", nargs="*", type=float, default=None,
        help="Optional list of thresholds to evaluate in one inference pass. Overrides --threshold.",
    )
    p.add_argument(
        "--device", default="auto",
        help='Device: "auto" (GPU if available), "cpu", "cuda", "cuda:0", etc.',
    )
    p.add_argument(
        "--session_filter", nargs="*", default=None,
        help="Only evaluate these session IDs (space-separated). Default: all.",
    )
    p.add_argument(
        "--log_level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip sessions that already have a saved JSON checkpoint in output_dir/sessions/.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── output dir & logging ───────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.log_level)

    logger.info("=" * 70)
    logger.info("AVSD Evaluation  |  checkpoint: %s", args.checkpoint)
    logger.info("eval_dir    : %s",  args.eval_dir)
    logger.info("output_dir  : %s",  args.output_dir)
    thresholds = args.thresholds if args.thresholds else [args.threshold]
    logger.info("chunk_frames: %d   stride_frames: %d   thresholds: %s",
                args.chunk_frames, args.stride_frames, ", ".join(f"{t:.3f}" for t in thresholds))
    logger.info("=" * 70)

    # ── device ────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # ── model config ──────────────────────────────────────────────────────
    config = dict(DEFAULT_CONFIG)
    config["output_speaker"] = args.output_speaker
    config["dropout"] = 0.0   # disable dropout for eval

    # ── load model ────────────────────────────────────────────────────────
    model, config = load_model(args.checkpoint, config, device, logger)
    model_speakers = config["output_speaker"]
    logger.info("Model output_speaker (trained): %d", model_speakers)

    # ── discover sessions ─────────────────────────────────────────────────
    session_paths = discover_sessions(args.eval_dir)
    if args.session_filter:
        session_paths = [p for p in session_paths if p.name in args.session_filter]
    logger.info("Sessions to evaluate: %d", len(session_paths))

    # ── evaluation loop ───────────────────────────────────────────────────
    session_results: List[Dict] = []
    failed_sessions: List[str]  = []

    t_start = time.time()
    for idx, session_path in enumerate(tqdm(session_paths, desc="Evaluating", unit="session")):
        session_id = session_path.name

        # Resume support: skip if already evaluated
        if args.resume:
            ckpt_path = os.path.join(args.output_dir, "sessions", f"{session_id}.json")
            if os.path.exists(ckpt_path):
                logger.info("  [SKIP] %s — checkpoint exists", session_id)
                with open(ckpt_path) as f:
                    session_results.append(json.load(f))
                continue

        try:
            session = EvalSession(session_path, model_speakers=model_speakers)
            result  = evaluate_session(
                session, model, model_speakers, device, logger,
                chunk_frames=args.chunk_frames,
                stride_frames=args.stride_frames,
                threshold=args.threshold,
                thresholds=thresholds,
            )
            session_results.append(result)
            save_session_checkpoint(result, args.output_dir)

            # ── periodic global summary every 10 sessions ─────────────
            if (idx + 1) % 10 == 0:
                interim = aggregate_global_metrics(session_results)
                logger.info(
                    "  [%d/%d]  Running DER=%.2f%%  loss=%.4f  F1=%.4f",
                    idx + 1, len(session_paths),
                    interim.get("global_DER", float("nan")),
                    interim.get("macro_bce_loss", float("nan")),
                    interim.get("macro_mean_f1", float("nan")),
                )

        except FileNotFoundError as e:
            logger.warning("SKIP %s — %s", session_id, e)
            failed_sessions.append(session_id)
        except Exception as e:  # noqa: BLE001
            logger.error("FAILED %s — %s", session_id, e, exc_info=True)
            failed_sessions.append(session_id)

    elapsed = time.time() - t_start
    logger.info("Inference finished in %.1f s (%.2f s/session)", elapsed,
                elapsed / max(len(session_results), 1))

    # ── global aggregation ────────────────────────────────────────────────
    primary_threshold_key = f"{float(thresholds[0]):.6f}".rstrip("0").rstrip(".")
    global_metrics = aggregate_global_metrics(session_results, threshold_key=primary_threshold_key)
    global_metrics["elapsed_seconds"] = round(elapsed, 2)
    global_metrics["failed_sessions"] = failed_sessions
    global_metrics["checkpoint"]      = args.checkpoint
    global_metrics["eval_dir"]        = args.eval_dir
    global_metrics["threshold"]       = thresholds[0]

    sweep_metrics = aggregate_threshold_sweep(session_results, thresholds)
    sweep_metrics["elapsed_seconds"] = round(elapsed, 2)
    sweep_metrics["failed_sessions"] = failed_sessions
    sweep_metrics["checkpoint"]      = args.checkpoint
    sweep_metrics["eval_dir"]        = args.eval_dir

    save_global_metrics(global_metrics, args.output_dir, logger)
    sweep_path = os.path.join(args.output_dir, "threshold_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump(sweep_metrics, f, indent=2, default=float)
    logger.info("Threshold sweep saved to %s", sweep_path)

    # ── final summary ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("  Sessions evaluated : %d", len(session_results))
    logger.info("  Sessions failed    : %d  %s",
                len(failed_sessions), failed_sessions or "")
    logger.info("  Global DER         : %.2f %%", global_metrics.get("global_DER", float("nan")))
    logger.info("  Global JER         : %.2f %%", global_metrics.get("global_JER", float("nan")))
    logger.info("  Macro BCE Loss     : %.6f",    global_metrics.get("macro_bce_loss", float("nan")))
    logger.info("  Macro Mean F1      : %.4f",    global_metrics.get("macro_mean_f1", float("nan")))
    logger.info("  Macro Mean Acc     : %.4f",    global_metrics.get("macro_mean_accuracy", float("nan")))
    if len(thresholds) > 1:
        logger.info("  Best threshold     : %s", sweep_metrics.get("best_threshold", "nan"))
    logger.info("  FA / MISS / CONF   : %d / %d / %d",
                int(global_metrics.get("total_FA",   0)),
                int(global_metrics.get("total_MISS", 0)),
                int(global_metrics.get("total_CONF", 0)))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
