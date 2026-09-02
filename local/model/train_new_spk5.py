import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import os
import argparse
from contextlib import nullcontext
from dataloader_spk5 import get_dataloader
from avsd_net_spk5 import AIVECTOR_ConformerVEmbedding_SD_JOINT

SPEAKER_CAP = 5
CHECKPOINT_DIR = "new_checkpoints_for_5"


def parse_args():
    parser = argparse.ArgumentParser(description="Train AVSD model (speaker cap 5)")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint path to resume from (e.g., new_checkpoints_for_5/model_epoch_20.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total number of epochs to run (overrides default 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Mini-batch size. Reduce if CUDA OOM happens.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps to simulate larger batches with lower memory.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable automatic mixed precision training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (eta). Default 5e-4. Increase for faster convergence, decrease for stability.",
    )
    parser.add_argument(
        "--lr-step-size",
        type=int,
        default=15,
        help="LR scheduler step size (how many epochs before decay). Default 15. Lower = faster decay.",
    )
    parser.add_argument(
        "--lr-gamma",
        type=float,
        default=0.1,
        help="LR scheduler decay factor (lr *= gamma every step-size epochs). Default 0.1.",
    )
    parser.add_argument(
        "--train-visual-encoder",
        action="store_true",
        help="Enable gradients for visual encoder. By default, it is frozen for memory efficiency.",
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default="focal",
        choices=["bce", "focal"],
        help="Loss type for diarization heads. Use focal for stronger class-imbalance handling.",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=4.0,
        help="Positive class weight for BCEWithLogits. >1 emphasizes missed speech frames.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (hard-example emphasis). Used when --loss-type focal.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.75,
        help="Focal alpha for positive class balancing. Used when --loss-type focal.",
    )
    return parser.parse_args()


def compute_speaker_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, args) -> torch.Tensor:
    """Class-imbalance-aware binary loss for one speaker stream with masking"""
    pos_weight = torch.tensor(args.pos_weight, device=logits.device, dtype=logits.dtype)

    if args.loss_type == "bce":
        per_frame = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    else:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=pos_weight,
            reduction="none",
        )
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(args.focal_gamma)

        alpha_pos = args.focal_alpha
        alpha_t = alpha_pos * targets + (1.0 - alpha_pos) * (1.0 - targets)
        per_frame = alpha_t * focal_factor * bce

    # Apply mask to only count valid frames
    denom = mask.sum().clamp_min(1.0)
    return (per_frame * mask).sum() / denom


def train_avsd(args):
    CONFIG = {
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

    CSV_PATH = "/home/speech-audio-research/22b3965/AVSD/local/training/training_chunks.csv"
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs if args.epochs is not None else 100
    LEARNING_RATE = args.lr

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    print("🔄 Initializing model...")
    model = AIVECTOR_ConformerVEmbedding_SD_JOINT(
        CONFIG,
        freeze_visual_encoder=(not args.train_visual_encoder),
    )
    model = model.to(device)

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
    print(f"🧠 AMP enabled: {use_amp}, dtype: {'bf16' if amp_dtype == torch.bfloat16 else 'fp16'}")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    print(f"📚 Learning rate: {LEARNING_RATE}, scheduler: StepLR(step_size={args.lr_step_size}, gamma={args.lr_gamma})")
    print(
        f"🧮 Loss config: type={args.loss_type}, pos_weight={args.pos_weight}, "
        f"focal_gamma={args.focal_gamma}, focal_alpha={args.focal_alpha}"
    )

    start_epoch = 1
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        print(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        print(f"✅ Resume successful. Starting from epoch {start_epoch}/{NUM_EPOCHS}")

    print("🔄 Loading data...")
    train_loader = get_dataloader(
        csv_path=CSV_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        max_speakers=CONFIG["output_speaker"],
        max_session_speakers=CONFIG["output_speaker"],
    )

    print(f"📊 Starting training with {len(train_loader)} batches per epoch")

    for epoch in range(start_epoch - 1, NUM_EPOCHS):
        model.train()
        if not args.train_visual_encoder:
            model.v_embedding.eval()
        epoch_loss = 0
        batch_count = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

        for batch_idx, batch in enumerate(progress_bar):
            batch_loss_value = None
            audio = batch['audio'].to(device, non_blocking=True)
            video = batch['video'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            nframes = batch['nframes'].tolist()

            audio_embed = torch.zeros(
                audio.size(0),
                CONFIG["output_speaker"],
                100,
                device=device,
            )

            if device.type == "cuda":
                if amp_autocast_ctor is not None:
                    amp_context = amp_autocast_ctor(device_type="cuda", enabled=use_amp, dtype=amp_dtype)
                else:
                    amp_context = torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype)
            else:
                amp_context = nullcontext()

            try:
                with amp_context:
                    outputs = model(audio, audio_embed, video, nframes, mask=mask)
                    # outputs: [B, T, num_speakers] logits

                # Reshape for loss computation: [B, T, num_speakers] -> [B*T, num_speakers]
                batch_size, seq_len, num_speakers = outputs.shape
                outputs_flat = outputs.reshape(batch_size * seq_len, num_speakers)
                labels_flat = labels.reshape(batch_size * seq_len, num_speakers)
                mask_flat = mask.reshape(batch_size * seq_len, 1)

                batch_loss = torch.zeros((), device=device)
                valid_speakers = 0

                # Compute loss for each speaker
                for speaker_idx in range(num_speakers):
                    speaker_logits = outputs_flat[:, speaker_idx]  # [B*T]
                    speaker_labels = labels_flat[:, speaker_idx]   # [B*T]
                    speaker_mask = mask_flat[:, 0]                 # [B*T]
                    
                    speaker_loss = compute_speaker_loss(
                        speaker_logits,
                        speaker_labels,
                        speaker_mask,
                        args
                    )
                    batch_loss += speaker_loss
                    valid_speakers += 1

                if valid_speakers > 0:
                    batch_loss = batch_loss / valid_speakers
                    if not torch.isfinite(batch_loss):
                        print(f"⚠️ Non-finite loss at batch {batch_idx + 1}, skipping optimizer step")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    batch_loss_value = float(batch_loss.detach().item())
                    scaled_loss = batch_loss / args.grad_accum_steps

                    if use_amp:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    if ((batch_idx + 1) % args.grad_accum_steps == 0) or ((batch_idx + 1) == len(train_loader)):
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                    epoch_loss += batch_loss_value
                    batch_count += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device.type == "cuda":
                    print(f"⚠️ OOM at batch {batch_idx + 1}, skipping batch and clearing cache")
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                raise

            progress_bar.set_postfix({
                'Loss': f'{batch_loss_value:.4f}' if batch_loss_value is not None else 'N/A',
                'Avg Loss': f'{(epoch_loss/batch_count):.4f}' if batch_count > 0 else 'N/A'
            })

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f'📈 Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_epoch_loss:.4f}')

        scheduler.step()

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss,
            'config': CONFIG
        }, checkpoint_path)
        print(f'💾 Checkpoint saved: {checkpoint_path}')

    print("✅ Training completed!")


if __name__ == "__main__":
    args = parse_args()
    train_avsd(args)
