#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-session evaluation and per-frame probability dump for manual inspection.

This script intentionally does NOT modify or replace evaluate.py. It reuses the
existing model/session utilities from evaluate.py and adds:
1) single-session inference
2) per-frame probability export [T, N]
3) compact summary json for quick review

Typical usage (cluster):
    python eval.py \
      --checkpoint /home/speech-audio-research/22b3965/AVSD/local/model/checkpoints/model_epoch_100.pth \
      --eval_root /home/speech-audio-research/22b3965/evaulation-bin/modified-bin \
      --session_id session_05 \
      --output_dir /home/speech-audio-research/22b3965/evaluation-results/manual_review \
      --threshold 0.20
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from evaluate import (
    CHUNK_FRAMES,
    DEFAULT_CONFIG,
    STRIDE_FRAMES,
    THRESHOLD,
    FPS,
    EvalSession,
    compute_bce_loss,
    compute_der,
    compute_per_speaker_metrics,
    infer_session_chunk,
    load_model,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one AVSD session and export per-frame probabilities"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.pth)",
    )
    parser.add_argument(
        "--eval_root",
        default="/home/speech-audio-research/22b3965/evaulation-bin/modified-bin",
        help="Root directory containing prepared session folders",
    )
    parser.add_argument(
        "--session_id",
        default="session_05",
        help="Session ID under eval_root. Ignored if --session_dir is provided",
    )
    parser.add_argument(
        "--session_dir",
        default="",
        help="Full path to one prepared session directory (overrides eval_root/session_id)",
    )
    parser.add_argument(
        "--output_dir",
        default="eval_manual",
        help="Directory to write dump and summary files",
    )
    parser.add_argument(
        "--output_speaker",
        type=int,
        default=DEFAULT_CONFIG["output_speaker"],
        help="Number of speakers used by trained model (default from checkpoint config)",
    )
    parser.add_argument(
        "--chunk_frames",
        type=int,
        default=CHUNK_FRAMES,
        help=f"Chunk size in frames (default: {CHUNK_FRAMES})",
    )
    parser.add_argument(
        "--stride_frames",
        type=int,
        default=STRIDE_FRAMES,
        help=f"Chunk stride in frames (default: {STRIDE_FRAMES})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Sigmoid threshold for binary predictions (default: {THRESHOLD})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device: "auto", "cpu", "cuda", "cuda:0", etc.',
    )
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="Also save per-frame logits in dump file",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def resolve_session_path(args: argparse.Namespace) -> Path:
    if args.session_dir:
        session_path = Path(args.session_dir)
    else:
        session_path = Path(args.eval_root) / args.session_id

    if not session_path.exists():
        raise FileNotFoundError(f"Session path not found: {session_path}")
    if not session_path.is_dir():
        raise NotADirectoryError(f"Session path is not a directory: {session_path}")

    return session_path


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def run_single_session(
    session: EvalSession,
    model: torch.nn.Module,
    model_speakers: int,
    device: torch.device,
    logger: logging.Logger,
    chunk_frames: int,
    stride_frames: int,
    threshold: float,
) -> Tuple[Dict, Dict]:
    """
    Run single-session inference and return:
      1) summary metrics dict
      2) frame-level payload dict (for npz dump)
    """
    logger.info(
        "Session %s | audio=%d video_min=%d effective=%d | n_true_speakers=%d",
        session.session_id,
        session.total_frames,
        session.video_min_frames,
        session.effective_frames,
        session.n_true_speakers,
    )

    n_true = session.n_true_speakers
    eff_frames = session.effective_frames

    all_logits = np.zeros((eff_frames, n_true), dtype=np.float32)
    all_labels = np.zeros((eff_frames, n_true), dtype=np.float32)
    frame_hits = np.zeros(eff_frames, dtype=np.int32)

    chunk_losses = []
    chunk_count = 0

    for chunk in session.iter_chunks(chunk_frames=chunk_frames, stride_frames=stride_frames):
        audio_np = chunk["audio"].unsqueeze(0).to(device)
        video_np = chunk["video"]
        labels_np = chunk["labels"].numpy()
        nframes = chunk["nframes"]
        start_f = chunk["start_f"]
        end_f = start_f + nframes

        logits_np = infer_session_chunk(
            model=model,
            audio_tensor=audio_np,
            video_np=video_np,
            nframes=nframes,
            model_speakers=model_speakers,
            device=device,
        )

        all_logits[start_f:end_f] += logits_np
        all_labels[start_f:end_f] = labels_np[:nframes]
        frame_hits[start_f:end_f] += 1

        chunk_losses.append(compute_bce_loss(logits_np, labels_np[:nframes]))
        chunk_count += 1

    covered = frame_hits > 0
    all_logits[covered] /= frame_hits[covered, None]

    logits = all_logits
    labels = all_labels

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.float32)

    session_loss = float(np.mean(chunk_losses)) if chunk_losses else float("nan")
    per_spk_metrics = compute_per_speaker_metrics(preds, labels)
    der_metrics = compute_der(preds, labels, use_hungarian=True)

    mean_f1 = float(np.mean([m["f1"] for m in per_spk_metrics])) if per_spk_metrics else float("nan")
    mean_acc = (
        float(np.mean([m["accuracy"] for m in per_spk_metrics])) if per_spk_metrics else float("nan")
    )

    summary = {
        "session_id": session.session_id,
        "session_path": str(session.session_dir),
        "total_frames": session.total_frames,
        "video_min_frames": session.video_min_frames,
        "effective_frames": eff_frames,
        "n_true_speakers": n_true,
        "speaker_ids": session.speaker_ids,
        "n_chunks": chunk_count,
        "threshold": float(threshold),
        "fps": int(FPS),
        "bce_loss": round(session_loss, 6),
        "mean_f1": round(mean_f1, 4),
        "mean_accuracy": round(mean_acc, 4),
        **der_metrics,
    }

    payload = {
        "session_id": session.session_id,
        "speaker_ids": np.array(session.speaker_ids, dtype=object),
        "probs": probs.astype(np.float32),
        "preds": preds.astype(np.uint8),
        "labels": labels.astype(np.float32),
        "logits": logits.astype(np.float32),
        "frame_hits": frame_hits.astype(np.int32),
        "effective_frames": np.int32(eff_frames),
        "total_frames": np.int32(session.total_frames),
        "video_min_frames": np.int32(session.video_min_frames),
        "threshold": np.float32(threshold),
        "fps": np.int32(FPS),
    }

    return summary, payload


def save_outputs(
    summary: Dict,
    payload: Dict,
    output_dir: Path,
    save_logits: bool,
    logger: logging.Logger,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    session_id = summary["session_id"]
    npz_path = output_dir / f"{session_id}_frame_probs.npz"
    json_path = output_dir / f"{session_id}_summary.json"

    npz_payload = {
        "session_id": np.array([payload["session_id"]], dtype=object),
        "speaker_ids": payload["speaker_ids"],
        "probs": payload["probs"],
        "preds": payload["preds"],
        "labels": payload["labels"],
        "frame_hits": payload["frame_hits"],
        "effective_frames": payload["effective_frames"],
        "total_frames": payload["total_frames"],
        "video_min_frames": payload["video_min_frames"],
        "threshold": payload["threshold"],
        "fps": payload["fps"],
    }
    if save_logits:
        npz_payload["logits"] = payload["logits"]

    np.savez_compressed(npz_path, **npz_payload)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)

    logger.info("Saved probabilities dump: %s", npz_path)
    logger.info("Saved summary json      : %s", json_path)
    return npz_path, json_path


def main() -> None:
    args = parse_args()

    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be in [0, 1] for valid probability cutoff")

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.log_level)

    session_path = resolve_session_path(args)
    device = resolve_device(args.device)

    logger.info("=" * 70)
    logger.info("Single-session AVSD evaluation")
    logger.info("checkpoint : %s", args.checkpoint)
    logger.info("session    : %s", session_path)
    logger.info("output_dir : %s", args.output_dir)
    logger.info("threshold  : %.3f", args.threshold)
    logger.info("device     : %s", device)
    logger.info("=" * 70)

    config = dict(DEFAULT_CONFIG)
    config["output_speaker"] = args.output_speaker
    config["dropout"] = 0.0

    model, config = load_model(args.checkpoint, config, device, logger)
    model_speakers = int(config["output_speaker"])

    session = EvalSession(session_path, model_speakers=model_speakers)

    t0 = time.time()
    summary, payload = run_single_session(
        session=session,
        model=model,
        model_speakers=model_speakers,
        device=device,
        logger=logger,
        chunk_frames=args.chunk_frames,
        stride_frames=args.stride_frames,
        threshold=args.threshold,
    )
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = round(elapsed, 3)

    save_outputs(
        summary=summary,
        payload=payload,
        output_dir=Path(args.output_dir),
        save_logits=args.save_logits,
        logger=logger,
    )

    logger.info("=" * 70)
    logger.info("Done")
    logger.info("DER      : %.4f", summary.get("DER", float("nan")))
    logger.info("JER      : %.4f", summary.get("JER", float("nan")))
    logger.info("Mean F1  : %.4f", summary.get("mean_f1", float("nan")))
    logger.info("BCE loss : %.6f", summary.get("bce_loss", float("nan")))
    logger.info("FA/MISS/CONF: %.0f / %.0f / %.0f", summary.get("FA", 0.0), summary.get("MISS", 0.0), summary.get("CONF", 0.0))
    logger.info("Elapsed  : %.2f s", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
