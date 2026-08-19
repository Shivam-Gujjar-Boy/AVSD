import argparse
import json
import logging
import os
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from dataloader import get_dataloader
from vsd_net import VSDNet, count_parameters


DEFAULT_CSV_PATH = "/home/speech-audio-research/22b3965/AVSD/local/training/training_chunks.csv"
DEFAULT_VAL_CSV_PATH = "/home/speech-audio-research/22b3965/AVSD/local/training/val_chunks.csv"
DEFAULT_CHECKPOINT_DIR = "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Train VSD model for speaker<=5 sessions")
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH, help="Path to training CSV")
    parser.add_argument("--val-csv-path", type=str, default=DEFAULT_VAL_CSV_PATH, help="Path to validation CSV")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Mini-batch size (reduced to prevent OOM)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr-step-size", type=int, default=15, help="StepLR step size")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="StepLR gamma")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Grad clip max norm")
    
    parser.add_argument(
        "--loss-type",
        type=str,
        default="focal",
        choices=["bce", "focal"],
        help="Frame-wise loss: weighted BCE or focal BCE.",
    )
    parser.add_argument("--pos-weight", type=float, default=2.0, help="Positive class weight")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal gamma")
    parser.add_argument("--focal-alpha", type=float, default=0.75, help="Focal alpha")

    parser.add_argument("--max-speakers", type=int, default=8)
    parser.add_argument("--max-session-speakers", type=int, default=5)

    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("vsd_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def masked_binary_loss(logits, labels, mask, args):
    pos_weight = torch.tensor(args.pos_weight, device=logits.device, dtype=logits.dtype)

    if args.loss_type == "bce":
        per_frame = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=pos_weight
        )
    else:
        bce = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=pos_weight
        )
        prob = torch.sigmoid(logits)
        p_t = prob * labels + (1.0 - prob) * (1.0 - labels)
        focal_factor = (1.0 - p_t).pow(args.focal_gamma)
        alpha_t = args.focal_alpha * labels + (1.0 - args.focal_alpha) * (1.0 - labels)
        per_frame = alpha_t * focal_factor * bce

    denom = mask.sum().clamp_min(1.0)
    return (per_frame * mask).sum() / denom


@torch.no_grad()
def evaluate(model, val_loader, device, args):
    model.eval()
    val_loss = 0.0
    valid_steps = 0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for batch in val_loader:
        video = batch["video"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        logits = model(video)
        loss = masked_binary_loss(logits, labels, mask, args)
        
        val_loss += float(loss.item())
        valid_steps += 1

        # Frame-level metrics with mask
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        active_mask = (mask == 1.0)
        tp = ((preds == 1.0) & (labels == 1.0) & active_mask).sum().item()
        fp = ((preds == 1.0) & (labels == 0.0) & active_mask).sum().item()
        fn = ((preds == 0.0) & (labels == 1.0) & active_mask).sum().item()

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall = total_tp / (total_tp + total_fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    avg_loss = val_loss / max(valid_steps, 1)

    return {
        "val_loss": round(avg_loss, 6),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }


def train(args):
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = setup_logger(args.checkpoint_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Setup | device=%s | checkpoint_dir=%s", device, args.checkpoint_dir)

    model = VSDNet().to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info("Model params | total=%d | trainable=%d", total_params, trainable_params)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    use_amp = (device.type == "cuda") and (not args.disable_amp)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    train_loader = get_dataloader(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_speakers=args.max_speakers,
        max_session_speakers=args.max_session_speakers,
    )

    val_loader = None
    if os.path.isfile(args.val_csv_path):
        val_loader = get_dataloader(
            csv_path=args.val_csv_path,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            max_speakers=args.max_speakers,
            max_session_speakers=args.max_session_speakers,
        )
        logger.info("Data | train_batches=%d | val_batches=%d", len(train_loader), len(val_loader))
    else:
        logger.warning("Validation CSV path not found: %s. Skipping validation step.", args.val_csv_path)

    start_epoch = 1
    best_f1 = 0.0

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = checkpoint.get("best_f1", 0.0)
        logger.info("Resume complete | start_epoch=%d | best_f1=%.4f", start_epoch, best_f1)

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        valid_steps = 0
        skipped_oom = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch_idx, batch in enumerate(progress, start=1):
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            amp_context = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype) if device.type == "cuda" else nullcontext()

            try:
                with amp_context:
                    logits = model(video)
                    loss = masked_binary_loss(logits, labels, mask, args)

                if not torch.isfinite(loss):
                    logger.warning("Batch skipped | reason=non_finite_loss | batch=%d", batch_idx)
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaled_loss = loss / args.grad_accum_steps
                if use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if (batch_idx % args.grad_accum_steps == 0) or (batch_idx == len(train_loader)):
                    if args.grad_clip_norm > 0:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                loss_value = float(loss.detach().item())
                epoch_loss += loss_value
                valid_steps += 1

                avg_loss = epoch_loss / max(valid_steps, 1)
                progress.set_postfix({"loss": f"{loss_value:.4f}", "avg": f"{avg_loss:.4f}"})

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.type == "cuda":
                    skipped_oom += 1
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    logger.warning("Batch skipped | reason=oom | batch=%d", batch_idx)
                    continue
                raise

        scheduler.step()

        avg_epoch_loss = epoch_loss / max(valid_steps, 1)
        logger.info("Epoch %d train loss: %.6f (OOMs: %d)", epoch, avg_epoch_loss, skipped_oom)

        # Validation Step
        val_metrics = {}
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device, args)
            logger.info("Epoch %d val summary: %s", epoch, json.dumps(val_metrics))

            # Save best model
            if val_metrics["f1_score"] > best_f1:
                best_f1 = val_metrics["f1_score"]
                best_path = os.path.join(args.checkpoint_dir, "best_model.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_f1": best_f1,
                        "val_metrics": val_metrics,
                    },
                    best_path,
                )
                logger.info("--> Saved new best checkpoint with F1: %.4f", best_f1)

        # Save regular epoch checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "avg_loss": avg_epoch_loss,
                "val_metrics": val_metrics,
                "best_f1": best_f1,
            },
            checkpoint_path,
        )

    logger.info("Training complete. Best F1-Score: %.4f", best_f1)


if __name__ == "__main__":
    train(parse_args())