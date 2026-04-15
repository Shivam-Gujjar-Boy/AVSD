import argparse
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from dataloader_spk6 import get_dataloader
from avsd_net_spk6 import AIVECTOR_ConformerVEmbedding_SD_JOINT


SPEAKER_CAP = 6
DEFAULT_CHECKPOINT_DIR = "checkpoints_for_6_tuned"


def parse_args():
    parser = argparse.ArgumentParser(description="Train AVSD model (speaker cap 6, hyperparameter tuned)")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=120, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--grad-accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--disable-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--train-visual-encoder", action="store_true", help="Train visual encoder (default: frozen)")

    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw"], help="Optimizer type")
    parser.add_argument("--lr", type=float, default=7e-4, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip-norm", type=float, default=5.0, help="Gradient clip max-norm")

    parser.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=["plateau", "cosine", "step", "none"],
        help="LR scheduler type",
    )
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear warmup epochs")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate")

    parser.add_argument("--lr-step-size", type=int, default=15, help="StepLR step size")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="StepLR gamma")

    parser.add_argument("--lr-patience", type=int, default=4, help="ReduceLROnPlateau patience")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="ReduceLROnPlateau factor")

    parser.add_argument("--target-smoothing", type=float, default=0.02, help="Label smoothing for BCE targets")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")

    return parser.parse_args()


def set_optimizer_lr(optimizer, lr_value):
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def get_current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def build_optimizer(model, args):
    if args.optimizer == "adamw":
        return AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def build_scheduler(optimizer, args):
    if args.scheduler == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    if args.scheduler == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=args.min_lr,
        )
    if args.scheduler == "step":
        return lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    return None


def train_avsd(args):
    config = {
        "input_dim": 40,
        "average_pooling": 3,
        "speaker_embedding_dim": 100,
        "output_speaker": SPEAKER_CAP,
        "audio_output_dim": 256,
        "video_embedding_dim": 256,
        "num_attention_heads": 4,
        "decoder_hidden_dim": 256,
        "dropout": 0.1,
    }

    csv_path = "/home/speech-audio-research/22b3965/AVSD/local/training/training_chunks.csv"

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if not (0.0 <= args.target_smoothing < 0.5):
        raise ValueError("--target-smoothing must be in [0.0, 0.5)")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AIVECTOR_ConformerVEmbedding_SD_JOINT(
        config,
        freeze_visual_encoder=(not args.train_visual_encoder),
    ).to(device)

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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args)

    print(
        f"Optimizer={args.optimizer}, base_lr={args.lr}, weight_decay={args.weight_decay}, "
        f"scheduler={args.scheduler}, warmup_epochs={args.warmup_epochs}, min_lr={args.min_lr}"
    )

    start_epoch = 1
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"Resume successful. Starting from epoch {start_epoch}/{args.epochs}")

    train_loader = get_dataloader(
        csv_path=csv_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        max_speakers=config["output_speaker"],
        max_session_speakers=config["output_speaker"],
    )

    print(f"Training with {len(train_loader)} batches/epoch")

    for epoch in range(start_epoch - 1, args.epochs):
        model.train()
        if not args.train_visual_encoder:
            model.v_embedding.eval()

        if epoch < args.warmup_epochs:
            warmup_ratio = float(epoch + 1) / float(max(1, args.warmup_epochs))
            warmup_lr = max(args.min_lr, args.lr * warmup_ratio)
            set_optimizer_lr(optimizer, warmup_lr)

        epoch_loss = 0.0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            batch_loss_value = None

            audio = batch["audio"].to(device, non_blocking=True)
            video = batch["video"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            nframes = batch["nframes"].tolist()

            if args.target_smoothing > 0.0:
                eps = args.target_smoothing
                labels = labels * (1.0 - 2.0 * eps) + eps

            audio_embed = torch.zeros(audio.size(0), config["output_speaker"], 100, device=device)

            if device.type == "cuda":
                if amp_autocast_ctor is not None:
                    amp_context = amp_autocast_ctor(device_type="cuda", enabled=use_amp, dtype=amp_dtype)
                else:
                    amp_context = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
            else:
                amp_context = nullcontext()

            try:
                with amp_context:
                    outputs = model(audio, audio_embed, video, nframes)

                batch_loss = torch.zeros((), device=device)
                valid_speakers = 0

                for i, output in enumerate(outputs):
                    if output is not None and len(output) > 0:
                        gt_labels = labels[:, :, i].reshape(-1, 1)
                        min_len = min(len(output), len(gt_labels))
                        if min_len > 0:
                            speaker_loss = criterion(output[:min_len], gt_labels[:min_len])
                            batch_loss += speaker_loss
                            valid_speakers += 1

                if valid_speakers > 0:
                    batch_loss = batch_loss / valid_speakers
                    if not torch.isfinite(batch_loss):
                        print(f"Non-finite loss at batch {batch_idx + 1}, skipping")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    batch_loss_value = float(batch_loss.detach().item())
                    scaled_loss = batch_loss / args.grad_accum_steps

                    if use_amp:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    if ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                        if args.grad_clip_norm > 0.0:
                            if use_amp:
                                scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    epoch_loss += batch_loss_value
                    batch_count += 1

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and device.type == "cuda":
                    print(f"OOM at batch {batch_idx + 1}, skipping and clearing cache")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            progress_bar.set_postfix(
                {
                    "Loss": f"{batch_loss_value:.4f}" if batch_loss_value is not None else "N/A",
                    "Avg Loss": f"{(epoch_loss / batch_count):.4f}" if batch_count > 0 else "N/A",
                    "LR": f"{get_current_lr(optimizer):.2e}",
                }
            )

        avg_epoch_loss = (epoch_loss / batch_count) if batch_count > 0 else 0.0

        if scheduler is not None and epoch >= args.warmup_epochs:
            if args.scheduler == "plateau":
                scheduler.step(avg_epoch_loss)
            else:
                scheduler.step()

        current_lr = get_current_lr(optimizer)
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.2e}")

        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "loss": avg_epoch_loss,
                "config": config,
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training completed")


if __name__ == "__main__":
    parsed_args = parse_args()
    train_avsd(parsed_args)
