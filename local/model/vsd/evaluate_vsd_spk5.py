#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vsd_net import VSDNet


DEFAULT_CHUNK_FRAMES = 200
DEFAULT_STRIDE_FRAMES = 200
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_SESSION_SPEAKERS = 5


def setup_logging(output_dir: str, log_level: str = "INFO") -> logging.Logger:
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
    logger = logging.getLogger("vsd_eval")
    logger.info("Logging initialized | file=%s", log_path)
    return logger


def discover_sessions(eval_dir: str) -> List[Path]:
    root = Path(eval_dir)
    return sorted([d for d in root.iterdir() if d.is_dir()], key=lambda p: p.name)


def _spk_index(path: Path) -> int:
    match = re.search(r"spk_(\d+)", path.stem)
    if not match:
        return 10**9
    return int(match.group(1))


class VSDEvalSession:
    def __init__(self, session_dir: Path, max_session_speakers: int = DEFAULT_MAX_SESSION_SPEAKERS):
        self.session_id = session_dir.name
        self.session_dir = session_dir
        self.max_session_speakers = max_session_speakers

        labels_path = session_dir / "diarization_labels.npy"
        nframes_path = session_dir / "nframes.txt"
        vfeat_dir = session_dir / "video_features"

        if not (labels_path.exists() and nframes_path.exists() and vfeat_dir.exists()):
            raise FileNotFoundError(
                f"Session {self.session_id}: missing diarization_labels.npy / nframes.txt / video_features"
            )

        self.labels_raw = np.load(labels_path)
        with open(nframes_path, "r", encoding="utf-8") as f:
            self.nframes_txt = int(f.read().strip())

        spk_files = sorted(vfeat_dir.glob("spk_*.npy"), key=_spk_index)
        self.spk_files = spk_files
        self.n_true_speakers = len(spk_files)

        self.skip_due_to_speaker_cap = self.n_true_speakers > self.max_session_speakers
        self.skip_reason = (
            f"speakers={self.n_true_speakers} > max_session_speakers={self.max_session_speakers}"
            if self.skip_due_to_speaker_cap
            else ""
        )

        self.video_arrays = [np.load(p) for p in self.spk_files]
        self.video_lengths = [len(v) for v in self.video_arrays]
        self.video_min_frames = min(self.video_lengths) if self.video_lengths else 0

        if self.labels_raw.ndim == 1:
            self.labels_raw = self.labels_raw[:, None]

        self.labels_frames = int(self.labels_raw.shape[0])

        self.effective_frames = min(self.nframes_txt, self.labels_frames, self.video_min_frames)
        if self.effective_frames < 1:
            raise ValueError(f"Session {self.session_id}: effective_frames={self.effective_frames}")

        if self.video_arrays:
            h = int(self.video_arrays[0].shape[1])
            w = int(self.video_arrays[0].shape[2])
        else:
            h, w = 96, 96
        self.video_h = h
        self.video_w = w

    def get_labels_matrix(self) -> np.ndarray:
        labels = np.zeros((self.effective_frames, self.n_true_speakers), dtype=np.float32)
        # IMPORTANT:
        # Training CSV stores speaker tracks in slot order (spk0, spk1, ...), where each
        # slot points to a possibly non-contiguous file name (e.g., spk_1.npy, spk_3.npy).
        # Labels are aligned to slot order, not necessarily to the numeric suffix in filename.
        # To match training behavior, evaluation also aligns label columns by local slot index.
        for col_idx, _spk_path in enumerate(self.spk_files):
            if self.labels_raw.shape[1] > col_idx:
                labels[:, col_idx] = self.labels_raw[: self.effective_frames, col_idx]
            else:
                labels[:, col_idx] = 0.0
        return labels


class VSDChunkInferencer:
    def __init__(self, model: nn.Module, device: torch.device, chunk_frames: int, stride_frames: int):
        self.model = model
        self.device = device
        self.chunk_frames = chunk_frames
        self.stride_frames = stride_frames

    def infer_session_logits(self, session: VSDEvalSession) -> np.ndarray:
        n_frames = session.effective_frames
        n_speakers = session.n_true_speakers

        logits_sum = np.zeros((n_frames, n_speakers), dtype=np.float32)
        frame_hits = np.zeros(n_frames, dtype=np.int32)

        for start in range(0, n_frames, self.stride_frames):
            end = min(start + self.chunk_frames, n_frames)
            if end <= start:
                continue

            for spk_col, video_array in enumerate(session.video_arrays):
                chunk = video_array[start:end]
                video_tensor = torch.as_tensor(chunk, dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    logits = self.model(video_tensor)

                logits_np = logits.squeeze(0).detach().cpu().numpy()
                logits_sum[start:end, spk_col] += logits_np

            frame_hits[start:end] += 1

        covered = frame_hits > 0
        logits_sum[covered] /= frame_hits[covered, None]

        return logits_sum


def compute_bce_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float32)
    labels_t = torch.as_tensor(labels, dtype=torch.float32)
    return float(nn.BCEWithLogitsLoss()(logits_t, labels_t).item())


def compute_per_speaker_metrics(predictions: np.ndarray, labels: np.ndarray) -> List[Dict]:
    n_spk = predictions.shape[1]
    results: List[Dict] = []

    for i in range(n_spk):
        tp = float(np.sum((predictions[:, i] == 1) & (labels[:, i] == 1)))
        fp = float(np.sum((predictions[:, i] == 1) & (labels[:, i] == 0)))
        fn = float(np.sum((predictions[:, i] == 0) & (labels[:, i] == 1)))
        tn = float(np.sum((predictions[:, i] == 0) & (labels[:, i] == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total = tp + fp + fn + tn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        results.append(
            {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }
        )

    return results


def compute_der(predictions: np.ndarray, labels: np.ndarray, use_hungarian: bool = True) -> Dict:
    t_frames, n_ref = labels.shape
    _, n_hyp = predictions.shape

    if use_hungarian and n_ref > 0 and n_hyp > 0:
        cost = np.zeros((n_ref, n_hyp), dtype=np.float64)
        for r in range(n_ref):
            for h in range(n_hyp):
                cost[r, h] = -float(np.sum(labels[:, r] * predictions[:, h]))

        row_ind, col_ind = linear_sum_assignment(cost)
        aligned_preds = np.zeros_like(labels)
        for r, h in zip(row_ind, col_ind):
            aligned_preds[:, r] = predictions[:, h]
    else:
        aligned_preds = np.zeros_like(labels)
        min_n = min(n_ref, n_hyp)
        aligned_preds[:, :min_n] = predictions[:, :min_n]

    n_ref_t = labels.sum(axis=1).astype(np.float64)
    n_hyp_t = aligned_preds.sum(axis=1).astype(np.float64)
    n_correct_t = (labels * aligned_preds).sum(axis=1).astype(np.float64)

    fa_t = np.maximum(0.0, n_hyp_t - n_ref_t)
    miss_t = np.maximum(0.0, n_ref_t - n_hyp_t)
    conf_t = np.maximum(0.0, np.minimum(n_ref_t, n_hyp_t) - n_correct_t)

    fa = float(fa_t.sum())
    miss = float(miss_t.sum())
    conf = float(conf_t.sum())

    total_ref = float(n_ref_t.sum())
    der = ((fa + miss + conf) / total_ref * 100.0) if total_ref > 0 else float("nan")

    jer_vals = []
    for r in range(n_ref):
        tp = float(np.sum(labels[:, r] * aligned_preds[:, r]))
        fp = float(np.sum((1 - labels[:, r]) * aligned_preds[:, r]))
        fn = float(np.sum(labels[:, r] * (1 - aligned_preds[:, r])))
        denom = tp + fp + fn
        jer_vals.append(1.0 - tp / denom if denom > 0 else 1.0)
    jer = float(np.mean(jer_vals) * 100.0) if jer_vals else float("nan")

    return {
        "DER": round(der, 4),
        "JER": round(jer, 4),
        "FA": fa,
        "MISS": miss,
        "CONF": conf,
        "total_ref_speech_frames": total_ref,
        "total_frames": float(t_frames),
    }


def evaluate_session_from_logits(
    session: VSDEvalSession,
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: List[float],
) -> Dict:
    threshold_metrics: Dict[str, Dict] = {}

    probs = 1.0 / (1.0 + np.exp(-logits))
    session_loss = compute_bce_loss(logits, labels)

    for thr in thresholds:
        preds = (probs >= thr).astype(np.float32)
        per_spk = compute_per_speaker_metrics(preds, labels)
        der_metrics = compute_der(preds, labels, use_hungarian=True)

        mean_f1 = float(np.mean([m["f1"] for m in per_spk])) if per_spk else float("nan")
        mean_acc = float(np.mean([m["accuracy"] for m in per_spk])) if per_spk else float("nan")

        thr_key = f"{thr:.6f}".rstrip("0").rstrip(".")
        threshold_metrics[thr_key] = {
            "threshold": thr,
            "mean_f1": round(mean_f1, 4),
            "mean_accuracy": round(mean_acc, 4),
            "per_speaker": per_spk,
            **der_metrics,
        }

    primary_key = f"{thresholds[0]:.6f}".rstrip("0").rstrip(".")
    primary = threshold_metrics[primary_key]

    return {
        "session_id": session.session_id,
        "n_true_speakers": session.n_true_speakers,
        "total_frames": session.nframes_txt,
        "effective_frames": session.effective_frames,
        "bce_loss": round(session_loss, 6),
        "threshold": primary["threshold"],
        "threshold_metrics": threshold_metrics,
        "mean_f1": primary["mean_f1"],
        "mean_accuracy": primary["mean_accuracy"],
        "per_speaker": primary["per_speaker"],
        "DER": primary["DER"],
        "JER": primary["JER"],
        "FA": primary["FA"],
        "MISS": primary["MISS"],
        "CONF": primary["CONF"],
        "total_ref_speech_frames": primary["total_ref_speech_frames"],
    }


def aggregate_global_metrics(session_results: List[Dict], threshold_key: Optional[str] = None) -> Dict:
    if not session_results:
        return {}

    if threshold_key is None:
        total_fa = sum(r["FA"] for r in session_results)
        total_miss = sum(r["MISS"] for r in session_results)
        total_conf = sum(r["CONF"] for r in session_results)
        total_ref = sum(r["total_ref_speech_frames"] for r in session_results)

        global_der = ((total_fa + total_miss + total_conf) / total_ref * 100.0) if total_ref > 0 else float("nan")

        jer_vals = [r["JER"] for r in session_results if not np.isnan(r["JER"])]
        global_jer = float(np.mean(jer_vals)) if jer_vals else float("nan")

        losses = [r["bce_loss"] for r in session_results if not np.isnan(r["bce_loss"])]
        f1s = [r["mean_f1"] for r in session_results]
        accs = [r["mean_accuracy"] for r in session_results]

        return {
            "n_sessions": len(session_results),
            "global_DER": round(global_der, 4),
            "global_JER": round(global_jer, 4),
            "macro_bce_loss": round(float(np.mean(losses)), 6) if losses else float("nan"),
            "macro_mean_f1": round(float(np.mean(f1s)), 4),
            "macro_mean_accuracy": round(float(np.mean(accs)), 4),
            "total_FA": total_fa,
            "total_MISS": total_miss,
            "total_CONF": total_conf,
            "total_ref_speech_frames": total_ref,
            "per_session_DER": {r["session_id"]: r["DER"] for r in session_results},
            "per_session_loss": {r["session_id"]: r["bce_loss"] for r in session_results},
        }

    total_fa = 0.0
    total_miss = 0.0
    total_conf = 0.0
    total_ref = 0.0
    jer_vals = []
    losses = []
    f1s = []
    accs = []
    per_session_der = {}
    per_session_loss = {}

    for result in session_results:
        thr_map = result.get("threshold_metrics", {})
        if threshold_key not in thr_map:
            continue
        metrics = thr_map[threshold_key]

        total_fa += metrics["FA"]
        total_miss += metrics["MISS"]
        total_conf += metrics["CONF"]
        total_ref += metrics["total_ref_speech_frames"]

        if not np.isnan(metrics["JER"]):
            jer_vals.append(metrics["JER"])
        if not np.isnan(result.get("bce_loss", float("nan"))):
            losses.append(result["bce_loss"])
        f1s.append(metrics["mean_f1"])
        accs.append(metrics["mean_accuracy"])
        per_session_der[result["session_id"]] = metrics["DER"]
        per_session_loss[result["session_id"]] = result["bce_loss"]

    global_der = ((total_fa + total_miss + total_conf) / total_ref * 100.0) if total_ref > 0 else float("nan")
    global_jer = float(np.mean(jer_vals)) if jer_vals else float("nan")

    return {
        "threshold_key": threshold_key,
        "n_sessions": len(per_session_der),
        "global_DER": round(global_der, 4),
        "global_JER": round(global_jer, 4),
        "macro_bce_loss": round(float(np.mean(losses)), 6) if losses else float("nan"),
        "macro_mean_f1": round(float(np.mean(f1s)), 4),
        "macro_mean_accuracy": round(float(np.mean(accs)), 4),
        "total_FA": total_fa,
        "total_MISS": total_miss,
        "total_CONF": total_conf,
        "total_ref_speech_frames": total_ref,
        "per_session_DER": per_session_der,
        "per_session_loss": per_session_loss,
    }


def aggregate_threshold_sweep(session_results: List[Dict], thresholds: List[float]) -> Dict:
    sweep = {}
    best_key = None
    best_der = float("inf")

    for thr in thresholds:
        key = f"{thr:.6f}".rstrip("0").rstrip(".")
        metrics = aggregate_global_metrics(session_results, threshold_key=key)
        sweep[key] = metrics

        der = metrics.get("global_DER", float("nan"))
        if not np.isnan(der) and der < best_der:
            best_der = der
            best_key = key

    return {
        "thresholds": thresholds,
        "metrics_by_threshold": sweep,
        "best_threshold": best_key,
        "best_metrics": sweep.get(best_key, {}),
    }


def load_model(checkpoint_path: str, device: torch.device, logger: logging.Logger) -> VSDNet:
    model = VSDNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info(
            "Checkpoint metadata | epoch=%s avg_loss=%s",
            checkpoint.get("epoch", "?"),
            checkpoint.get("avg_loss", "nan"),
        )
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.warning("Unexpected keys (%d): %s", len(unexpected), unexpected[:5])

    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded | params=%s | device=%s", f"{total_params:,}", device)
    return model


def save_session_result(result: Dict, output_dir: str):
    session_dir = os.path.join(output_dir, "sessions")
    os.makedirs(session_dir, exist_ok=True)
    out_path = os.path.join(session_dir, f"{result['session_id']}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=float)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VSD model for sessions with <=5 speakers")
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to VSD checkpoint .pth")
    parser.add_argument("--eval_dir", "-e", required=True, help="Root directory with session folders")
    parser.add_argument("--output_dir", "-o", default="eval_results_vsd", help="Output directory")

    parser.add_argument("--chunk_frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    parser.add_argument("--stride_frames", type=int, default=DEFAULT_STRIDE_FRAMES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--thresholds",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of thresholds for one-pass sweep. Overrides --threshold.",
    )

    parser.add_argument("--max_session_speakers", type=int, default=DEFAULT_MAX_SESSION_SPEAKERS)
    parser.add_argument("--device", default="auto", help='auto/cpu/cuda/cuda:0')
    parser.add_argument("--session_filter", nargs="*", default=None)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", action="store_true", help="Skip sessions that already have result JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir, args.log_level)

    thresholds = args.thresholds if args.thresholds else [args.threshold]
    thresholds = [float(t) for t in thresholds]

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("=" * 70)
    logger.info("VSD evaluation start")
    logger.info("checkpoint=%s", args.checkpoint)
    logger.info("eval_dir=%s", args.eval_dir)
    logger.info("output_dir=%s", args.output_dir)
    logger.info("chunk_frames=%d stride_frames=%d thresholds=%s", args.chunk_frames, args.stride_frames, thresholds)
    logger.info("max_session_speakers=%d", args.max_session_speakers)
    logger.info("device=%s", device)
    logger.info("=" * 70)

    model = load_model(args.checkpoint, device, logger)
    inferencer = VSDChunkInferencer(model=model, device=device, chunk_frames=args.chunk_frames, stride_frames=args.stride_frames)

    all_sessions = discover_sessions(args.eval_dir)
    if args.session_filter:
        all_sessions = [s for s in all_sessions if s.name in args.session_filter]

    logger.info("Discovered sessions=%d", len(all_sessions))

    session_results: List[Dict] = []
    failed_sessions: List[str] = []
    skipped_speaker_limit: List[str] = []

    t0 = time.time()

    for idx, session_path in enumerate(tqdm(all_sessions, desc="Evaluating", unit="session"), start=1):
        session_id = session_path.name
        try:
            if args.resume:
                done_path = os.path.join(args.output_dir, "sessions", f"{session_id}.json")
                if os.path.exists(done_path):
                    with open(done_path, "r", encoding="utf-8") as f:
                        session_results.append(json.load(f))
                    logger.info("[SKIP RESUME] %s", session_id)
                    continue

            session = VSDEvalSession(session_path, max_session_speakers=args.max_session_speakers)
            if session.skip_due_to_speaker_cap:
                skipped_speaker_limit.append(session_id)
                logger.info("[SKIP SPEAKER CAP] %s | %s", session_id, session.skip_reason)
                continue

            labels = session.get_labels_matrix()
            logits = inferencer.infer_session_logits(session)
            result = evaluate_session_from_logits(session, logits, labels, thresholds)
            session_results.append(result)
            save_session_result(result, args.output_dir)

            primary_key = f"{thresholds[0]:.6f}".rstrip("0").rstrip(".")
            p = result["threshold_metrics"][primary_key]
            logger.info(
                "Session %s | speakers=%d frames=%d | loss=%.4f DER=%.2f JER=%.2f F1=%.4f",
                session_id,
                session.n_true_speakers,
                session.effective_frames,
                result["bce_loss"],
                p["DER"],
                p["JER"],
                p["mean_f1"],
            )

            if idx % 10 == 0 and session_results:
                interim = aggregate_global_metrics(session_results, threshold_key=primary_key)
                logger.info(
                    "Interim %d/%d | DER=%.2f loss=%.4f F1=%.4f",
                    idx,
                    len(all_sessions),
                    interim.get("global_DER", float("nan")),
                    interim.get("macro_bce_loss", float("nan")),
                    interim.get("macro_mean_f1", float("nan")),
                )

        except Exception as exc:  # pylint: disable=broad-except
            failed_sessions.append(session_id)
            logger.error("[FAILED] %s | %s", session_id, exc, exc_info=True)

    elapsed = time.time() - t0

    primary_key = f"{thresholds[0]:.6f}".rstrip("0").rstrip(".")
    global_metrics = aggregate_global_metrics(session_results, threshold_key=primary_key)
    global_metrics["elapsed_seconds"] = round(elapsed, 2)
    global_metrics["failed_sessions"] = failed_sessions
    global_metrics["skipped_sessions_speaker_cap"] = skipped_speaker_limit
    global_metrics["checkpoint"] = args.checkpoint
    global_metrics["eval_dir"] = args.eval_dir
    global_metrics["threshold"] = thresholds[0]

    sweep_metrics = aggregate_threshold_sweep(session_results, thresholds)
    sweep_metrics["elapsed_seconds"] = round(elapsed, 2)
    sweep_metrics["failed_sessions"] = failed_sessions
    sweep_metrics["skipped_sessions_speaker_cap"] = skipped_speaker_limit
    sweep_metrics["checkpoint"] = args.checkpoint
    sweep_metrics["eval_dir"] = args.eval_dir

    with open(os.path.join(args.output_dir, "global_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(global_metrics, f, indent=2, default=float)

    with open(os.path.join(args.output_dir, "threshold_sweep.json"), "w", encoding="utf-8") as f:
        json.dump(sweep_metrics, f, indent=2, default=float)

    logger.info("=" * 70)
    logger.info("VSD evaluation complete")
    logger.info("sessions_evaluated=%d", len(session_results))
    logger.info("sessions_failed=%d", len(failed_sessions))
    logger.info("sessions_skipped_speaker_cap=%d", len(skipped_speaker_limit))
    logger.info("global_DER=%.2f", global_metrics.get("global_DER", float("nan")))
    logger.info("global_JER=%.2f", global_metrics.get("global_JER", float("nan")))
    logger.info("macro_bce_loss=%.6f", global_metrics.get("macro_bce_loss", float("nan")))
    logger.info("macro_mean_f1=%.4f", global_metrics.get("macro_mean_f1", float("nan")))
    logger.info("best_threshold=%s", sweep_metrics.get("best_threshold", "nan"))
    logger.info("elapsed_seconds=%.2f", elapsed)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
