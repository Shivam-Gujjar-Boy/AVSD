import argparse
import json
import logging
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from dataloader import get_dataloader
from vsd_net import VSDNet, count_parameters


DEFAULT_CSV_PATH = "/home/speech-audio-research/22b3965/AVSD/local/training/training_chunks.csv"
DEFAULT_CHECKPOINT_DIR = "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(description="Train VSD model for speaker<=5 sessions")
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV_PATH, help="Path to chunk-level CSV")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=100, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
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
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=4.0,
        help="Positive class weight for BCEWithLogits; >1 emphasizes missed speech frames.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal gamma (hard-example emphasis). Used when --loss-type focal.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.75,
        help="Focal alpha for positive class balancing. Used when --loss-type focal.",
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Highest speaker index range to scan in CSV columns (spk0..spkN)",
    )
    parser.add_argument(
        "--max-session-speakers",
        type=int,
        default=5,
        help="Only sessions/chunks with <= this number of speakers are used",
    )

    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint folder")
    parser.add_argument("--log-interval", type=int, default=20, help="Steps between detailed logs")
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
            logits,
            labels,
            reduction="none",
            pos_weight=pos_weight,
        )
    else:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
            pos_weight=pos_weight,
        )
        prob = torch.sigmoid(logits)
        p_t = prob * labels + (1.0 - prob) * (1.0 - labels)
        focal_factor = (1.0 - p_t).pow(args.focal_gamma)
        alpha_t = args.focal_alpha * labels + (1.0 - args.focal_alpha) * (1.0 - labels)
        per_frame = alpha_t * focal_factor * bce

    denom = mask.sum().clamp_min(1.0)
    return (per_frame * mask).sum() / denom


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
    amp_mod = getattr(torch, "amp", None)
    amp_autocast_ctor = getattr(amp_mod, "autocast", None) if amp_mod is not None else None
    amp_grad_scaler_ctor = getattr(amp_mod, "GradScaler", None) if amp_mod is not None else None
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = (
        amp_grad_scaler_ctor("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        if amp_grad_scaler_ctor is not None
        else torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))
    )

    logger.info(
        "Optim | lr=%g weight_decay=%g step_size=%d gamma=%g amp=%s dtype=%s",
        args.lr,
        args.weight_decay,
        args.lr_step_size,
        args.lr_gamma,
        use_amp,
        "bf16" if amp_dtype == torch.bfloat16 else "fp16",
    )
    logger.info(
        "Loss | type=%s pos_weight=%.3f focal_gamma=%.3f focal_alpha=%.3f",
        args.loss_type,
        args.pos_weight,
        args.focal_gamma,
        args.focal_alpha,
    )

    train_loader = get_dataloader(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_speakers=args.max_speakers,
        max_session_speakers=args.max_session_speakers,
    )
    logger.info("Data | batches_per_epoch=%d", len(train_loader))

    start_epoch = 1
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        logger.info("Resume | checkpoint=%s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        logger.info("Resume complete | start_epoch=%d", start_epoch)

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

            if device.type == "cuda":
                if amp_autocast_ctor is not None:
                    amp_context = amp_autocast_ctor(device_type="cuda", enabled=use_amp, dtype=amp_dtype)
                else:
                    amp_context = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
            else:
                amp_context = nullcontext()

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

                if batch_idx % args.log_interval == 0:
                    logger.info(
                        "Batch | epoch=%d step=%d/%d loss=%.6f lr=%.3e",
                        epoch,
                        batch_idx,
                        len(train_loader),
                        loss_value,
                        current_lr(optimizer),
                    )

                avg_loss = epoch_loss / max(valid_steps, 1)
                progress.set_postfix({"loss": f"{loss_value:.4f}", "avg": f"{avg_loss:.4f}"})

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.type == "cuda":
                    skipped_oom += 1
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    logger.warning("Batch skipped | reason=oom | batch=%d", batch_idx)
                    continue
                logger.exception("Batch failed | epoch=%d batch=%d", epoch, batch_idx)
                raise

        scheduler.step()

        avg_epoch_loss = epoch_loss / max(valid_steps, 1)
        epoch_summary = {
            "epoch": epoch,
            "avg_loss": round(avg_epoch_loss, 6),
            "valid_steps": valid_steps,
            "oom_skipped": skipped_oom,
            "lr": current_lr(optimizer),
        }
        logger.info("Epoch summary | %s", json.dumps(epoch_summary))

        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "avg_loss": avg_epoch_loss,
                "args": vars(args),
            },
            checkpoint_path,
        )
        logger.info("Checkpoint saved | path=%s", checkpoint_path)

    logger.info("Training complete")


if __name__ == "__main__":
    train(parse_args())
