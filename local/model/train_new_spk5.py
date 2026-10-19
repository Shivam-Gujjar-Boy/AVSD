import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import os
import argparse
import logging
import time
from contextlib import nullcontext
from datetime import datetime
from dataloader_spk5 import get_dataloader
from avsd_net_spk5 import AIVECTOR_ConformerVEmbedding_SD_JOINT

SPEAKER_CAP = 5
CHECKPOINT_DIR = "checkpoints_spk5_sem9_eqweights"
LOG_DIR = "logs_spk5"


def configure_logger() -> tuple[logging.Logger, str]:
    os.makedirs(LOG_DIR, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"training_{run_id}.log")

    logger = logging.getLogger("avsd.train_spk5")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train AVSD model (speaker cap 5)")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Checkpoint path to resume from (e.g., checkpoints_spk5_sem9/model_epoch_20.pth)",
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
    logger, log_path = configure_logger()
    run_start_time = time.perf_counter()

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
    logger.info("Training run started; log_file=%s", os.path.abspath(log_path))
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("CUDA device: %s", torch.cuda.get_device_name(device))
        logger.info("CUDA memory: total=%.2f GiB", torch.cuda.get_device_properties(device).total_memory / 2**30)

    logger.info("Initializing model")
    model = AIVECTOR_ConformerVEmbedding_SD_JOINT(
        CONFIG,
        freeze_visual_encoder=(not args.train_visual_encoder),
    )
    model = model.to(device)
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "Model initialized; total_parameters=%d trainable_parameters=%d visual_encoder_frozen=%s",
        total_parameters,
        trainable_parameters,
        not args.train_visual_encoder,
    )

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
        "AMP enabled=%s dtype=%s",
        use_amp,
        "bf16" if amp_dtype == torch.bfloat16 else "fp16",
    )

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
	mode='min',
	factor=args.lr_gamma,
	patience=args.lr_step_size
    )
    logger.info(
        "Optimizer=Adam learning_rate=%s scheduler=ReduceLROnPlateau(patience=%s, gamma=%s)",
        LEARNING_RATE,
        args.lr_step_size,
        args.lr_gamma,
    )
    logger.info(
        "Loss configuration: type=%s pos_weight=%s focal_gamma=%s focal_alpha=%s",
        args.loss_type,
        args.pos_weight,
        args.focal_gamma,
        args.focal_alpha,
    )
    logger.info(
        "Run configuration: epochs=%d batch_size=%d num_workers=%d grad_accum_steps=%d "
        "train_visual_encoder=%s disable_amp=%s csv_path=%s",
        NUM_EPOCHS,
        BATCH_SIZE,
        args.num_workers,
        args.grad_accum_steps,
        args.train_visual_encoder,
        args.disable_amp,
        CSV_PATH,
    )

    start_epoch = 1
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        logger.info("Resuming from checkpoint: %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        logger.info("Resume successful; starting_epoch=%d total_epochs=%d", start_epoch, NUM_EPOCHS)

    logger.info("Loading data")
    train_loader = get_dataloader(
        csv_path=CSV_PATH,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.num_workers,
        max_speakers=CONFIG["output_speaker"],
        max_session_speakers=CONFIG["output_speaker"],
    )

    logger.info("Data loaded; batches_per_epoch=%d", len(train_loader))

    for epoch in range(start_epoch - 1, NUM_EPOCHS):
        epoch_start_time = time.perf_counter()
        model.train()
        if not args.train_visual_encoder:
            model.v_embedding.eval()
        epoch_loss = 0
        batch_count = 0
        skipped_batches = 0
        optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        logger.info("Epoch %d/%d started; learning_rate=%s", epoch + 1, NUM_EPOCHS, optimizer.param_groups[0]["lr"])

        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.perf_counter()
            batch_loss_value = None
            audio = batch['audio'].to(device, non_blocking=True)
            video = batch['video'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            speaker_mask = batch['speaker_mask'].to(device, non_blocking=True)
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
                    outputs = model(audio, audio_embed, video, nframes, mask=mask, speaker_mask=speaker_mask)
                    # outputs: [B, T, num_speakers] logits

                # Reshape for loss computation: [B, T, num_speakers] -> [B*T, num_speakers]
                batch_size, seq_len, num_speakers = outputs.shape
                outputs_flat = outputs.reshape(batch_size * seq_len, num_speakers)
                labels_flat = labels.reshape(batch_size * seq_len, num_speakers)
                mask_flat = mask.reshape(batch_size * seq_len, 1)
                speaker_mask_flat = speaker_mask[:, None, :].expand(
                    batch_size, seq_len, num_speakers
                ).reshape(batch_size * seq_len, num_speakers)

                batch_loss = torch.zeros((), device=device)
                valid_speakers = 0

                # Compute loss for each speaker
                for speaker_idx in range(num_speakers):
                    speaker_logits = outputs_flat[:, speaker_idx]  # [B*T]
                    speaker_labels = labels_flat[:, speaker_idx]   # [B*T]
                    loss_mask = mask_flat[:, 0] * speaker_mask_flat[:, speaker_idx]

                    if loss_mask.sum() > 0:
                        speaker_loss = compute_speaker_loss(
                            speaker_logits,
                            speaker_labels,
                            loss_mask,
                            args
                        )
                        batch_loss += speaker_loss
                        valid_speakers += 1
                    
                    # speaker_loss = compute_speaker_loss(
                    #     speaker_logits,
                    #     speaker_labels,
                    #     loss_mask,
                    #     args
                    # )
                    # batch_loss += speaker_loss
                    # valid_speakers += 1

                if valid_speakers > 0:
                    batch_loss = batch_loss / valid_speakers
                    if not torch.isfinite(batch_loss):
                        skipped_batches += 1
                        logger.warning(
                            "Non-finite loss; epoch=%d batch=%d/%d skipping optimizer step",
                            epoch + 1,
                            batch_idx + 1,
                            len(train_loader),
                        )
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
                        optimizer_steps += 1

                    epoch_loss += batch_loss_value
                    batch_count += 1
                    logger.debug(
                        "Batch completed; epoch=%d batch=%d/%d loss=%.6f running_avg=%.6f "
                        "valid_speakers=%d optimizer_steps=%d duration_seconds=%.3f",
                        epoch + 1,
                        batch_idx + 1,
                        len(train_loader),
                        batch_loss_value,
                        epoch_loss / batch_count,
                        valid_speakers,
                        optimizer_steps,
                        time.perf_counter() - batch_start_time,
                    )
                else:
                    skipped_batches += 1
                    logger.warning(
                        "No valid speakers; epoch=%d batch=%d/%d skipped",
                        epoch + 1,
                        batch_idx + 1,
                        len(train_loader),
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device.type == "cuda":
                    skipped_batches += 1
                    logger.warning(
                        "CUDA OOM; epoch=%d batch=%d/%d skipping batch and clearing cache",
                        epoch + 1,
                        batch_idx + 1,
                        len(train_loader),
                    )
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    continue
                logger.exception("RuntimeError at epoch=%d batch=%d", epoch + 1, batch_idx + 1)
                raise

            progress_bar.set_postfix({
                'Loss': f'{batch_loss_value:.4f}' if batch_loss_value is not None else 'N/A',
                'Avg Loss': f'{(epoch_loss/batch_count):.4f}' if batch_count > 0 else 'N/A'
            })

        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
        learning_rate_before_scheduler = optimizer.param_groups[0]["lr"]

        scheduler.step(avg_epoch_loss)
        learning_rate_after_scheduler = optimizer.param_groups[0]["lr"]
        epoch_duration = time.perf_counter() - epoch_start_time
        logger.info(
            "Epoch %d/%d completed; average_loss=%.6f processed_batches=%d skipped_batches=%d "
            "optimizer_steps=%d learning_rate_before=%s learning_rate_after=%s duration_seconds=%.2f",
            epoch + 1,
            NUM_EPOCHS,
            avg_epoch_loss,
            batch_count,
            skipped_batches,
            optimizer_steps,
            learning_rate_before_scheduler,
            learning_rate_after_scheduler,
            epoch_duration,
        )

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_epoch_loss,
            'config': CONFIG
        }, checkpoint_path)
        logger.info("Checkpoint saved: %s", os.path.abspath(checkpoint_path))

    logger.info("Training completed successfully; total_duration_seconds=%.2f", time.perf_counter() - run_start_time)


if __name__ == "__main__":
    args = parse_args()
    train_avsd(args)
