"""
Train the AVSD model (speaker cap 5) with evaluation after every epoch.

Per epoch it logs and writes to <run_dir>/metrics.csv:
  train loss, train frame-DER (at 0.5, with dropout/augmentation on),
  eval loss (same objective), eval plain BCE,
  eval frame-DER with miss / false alarm / confusion at threshold 0.5,
  and the best threshold from a sweep with its DER.
A plot of the curves (metrics.png) is refreshed every epoch.

Frame-level DER here uses the fixed face<->label mapping (no permutation search
and no collar), so it is a training-time indicator, not the official md-eval DER.
"""
import argparse
import csv
import logging
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from avsd_net_spk5 import AIVECTOR_ConformerVEmbedding_SD_JOINT
from dataloader_spk5 import get_dataloader

SPEAKER_CAP = 5
DEFAULT_TRAIN_CSV = "/home/speech-audio-research/22b3965/AVSD/local/training/training_chunks.csv"
DEFAULT_EVAL_DIR = "/home/speech-audio-processing/22b3965/evaulation-bin/modified-bin/"
THRESHOLDS = [round(float(t), 2) for t in np.arange(0.2, 0.81, 0.05)]


# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AVSD model (speaker cap 5) with per-epoch evaluation")
    # data
    p.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR, help="Folder of evaluation sessions (same layout as training)")
    p.add_argument("--chunk-frames", type=int, default=200, help="Eval chunk length (8 s at 25 fps)")
    p.add_argument("--no-augment", action="store_true", help="Disable training augmentation")
    # run
    p.add_argument("--run-dir", default="", help="Output folder (default: runs_spk5/<timestamp>)")
    p.add_argument("--resume", default="", help="Checkpoint to resume from (e.g. <run_dir>/last.pth)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=0, help="Default: same as --batch-size")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=2)
    p.add_argument("--disable-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=5, help="Also keep epoch_N.pth every N epochs (0 = never)")
    p.add_argument("--early-stop-patience", type=int, default=15, help="Stop after N epochs without eval-DER improvement (0 = off)")
    # optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--min-lr-ratio", type=float, default=0.02, help="Cosine floor as a fraction of --lr")
    p.add_argument("--scheduler", choices=["cosine", "plateau"], default="cosine")
    p.add_argument("--plateau-patience", type=int, default=4)
    p.add_argument("--plateau-factor", type=float, default=0.5)
    p.add_argument("--grad-clip", type=float, default=5.0, help="Max grad norm (0 = off)")
    # loss
    p.add_argument("--loss-type", choices=["bce", "focal"], default="bce")
    p.add_argument("--pos-weight", type=float, default=1.0, help="BCE positive-class weight (see logged suggestion)")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--focal-alpha", type=float, default=0.5)
    # model
    p.add_argument("--freeze-visual-encoder", action="store_true", help="Only valid with --visual-pretrained")
    p.add_argument("--visual-pretrained", default="", help="Checkpoint with v_embedding.* weights")
    p.add_argument("--visual-grad-checkpoint", action="store_true", help="Trade compute for GPU memory")
    p.add_argument("--train-visual-encoder", action="store_true", help="(Deprecated: the encoder now trains by default)")
    return p.parse_args()


def configure_logger(run_dir):
    logger = logging.getLogger("avsd.train_spk5")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_seed_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ─────────────────────────────────────────────────────────────────────────────
# Loss and metrics
# ─────────────────────────────────────────────────────────────────────────────

def masked_loss(logits, targets, valid, args):
    """logits/targets/valid: [B, T, N]. Mean over valid (frame, present-speaker) cells."""
    logits = logits.float()
    pos_weight = torch.tensor(args.pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    if args.loss_type == "focal":
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = args.focal_alpha * targets + (1 - args.focal_alpha) * (1 - targets)
        bce = alpha_t * (1 - p_t).pow(args.focal_gamma) * bce
    return (bce * valid).sum() / valid.sum().clamp_min(1.0)


def plain_bce(logits, targets, valid):
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="none")
    return (bce * valid).sum() / valid.sum().clamp_min(1.0)


class DERCounter:
    """Frame-level DER components with the fixed face<->label mapping, for several thresholds."""

    def __init__(self, thresholds):
        self.thresholds = list(thresholds)
        self.counts = {t: np.zeros(4) for t in self.thresholds}  # miss, fa, conf, ref_speech

    @torch.no_grad()
    def update(self, logits, labels, valid):
        probs = torch.sigmoid(logits.float())
        ref = (labels > 0.5).float() * valid
        n_ref = ref.sum(-1)                                    # [B, T]
        frame_valid = valid.amax(-1)                           # [B, T]
        for t in self.thresholds:
            hyp = (probs >= t).float() * valid
            n_hyp = hyp.sum(-1)
            n_cor = (ref * hyp).sum(-1)
            miss = (n_ref - n_hyp).clamp_min(0) * frame_valid
            fa = (n_hyp - n_ref).clamp_min(0) * frame_valid
            conf = (torch.minimum(n_ref, n_hyp) - n_cor) * frame_valid
            self.counts[t] += np.array([miss.sum().item(), fa.sum().item(), conf.sum().item(), n_ref.sum().item()])

    def result(self, t):
        miss, fa, conf, ref = self.counts[t]
        ref = max(ref, 1.0)
        return {"der": (miss + fa + conf) / ref, "miss": miss / ref, "fa": fa / ref, "conf": conf / ref}

    def best(self):
        t = min(self.thresholds, key=lambda x: self.result(x)["der"])
        return t, self.result(t)


# ─────────────────────────────────────────────────────────────────────────────
# Train / eval loops
# ─────────────────────────────────────────────────────────────────────────────

def unpack(batch, device):
    audio = batch["audio"].to(device, non_blocking=True)
    video = batch["video"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    speaker_mask = batch["speaker_mask"].to(device, non_blocking=True)
    valid = mask[:, :, None] * speaker_mask[:, None, :]       # [B, T, N]
    return audio, video, labels, mask, speaker_mask, valid, batch["nframes"].tolist()


def amp_context(device, use_amp, amp_dtype):
    if device.type != "cuda" or not use_amp:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


@torch.no_grad()
def evaluate(model, loader, device, args, use_amp, amp_dtype):
    model.eval()
    loss_sum = bce_sum = weight = 0.0
    der = DERCounter(THRESHOLDS)
    for batch in tqdm(loader, desc="  eval", leave=False):
        audio, video, labels, mask, speaker_mask, valid, nframes = unpack(batch, device)
        if valid.sum() == 0:
            continue
        with amp_context(device, use_amp, amp_dtype):
            logits = model(audio, None, video, nframes, mask=mask, speaker_mask=speaker_mask)
        n = valid.sum().item()
        loss_sum += masked_loss(logits, labels, valid, args).item() * n
        bce_sum += plain_bce(logits, labels, valid).item() * n
        weight += n
        der.update(logits, labels, valid)
    weight = max(weight, 1.0)
    best_t, best = der.best()
    return {
        "loss": loss_sum / weight,
        "bce": bce_sum / weight,
        "at05": der.result(0.5),
        "best_threshold": best_t,
        "best": best,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, config, metrics, best_der):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "loss": metrics.get("train_loss"),
            "metrics": metrics,
            "best_eval_der": float(best_der),
        },
        path,
    )


def write_metrics(run_dir, history):
    path = os.path.join(run_dir, "metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ep = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
        axes[0].plot(ep, [h["eval_loss"] for h in history], label="eval loss")
        axes[0].set_title("Loss (training objective)")
        axes[1].plot(ep, [100 * h["train_der@0.5"] for h in history], label="train DER@0.5")
        axes[1].plot(ep, [100 * h["eval_der@0.5"] for h in history], label="eval DER@0.5")
        axes[1].plot(ep, [100 * h["eval_der_best"] for h in history], "--", label="eval DER (best thr)")
        axes[1].set_title("Frame DER (%)")
        for key, name in [("eval_miss@0.5", "miss"), ("eval_fa@0.5", "false alarm"), ("eval_conf@0.5", "confusion")]:
            axes[2].plot(ep, [100 * h[key] for h in history], label=name)
        axes[2].set_title("Eval DER components @0.5 (%)")
        for ax in axes:
            ax.set_xlabel("epoch")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "metrics.png"), dpi=110)
        plt.close(fig)
    except ImportError:
        pass


def train_avsd(args):
    set_seed(args.seed)
    run_dir = args.run_dir or os.path.join("runs_spk5", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    logger = configure_logger(run_dir)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if args.train_visual_encoder:
        logger.info("--train-visual-encoder is no longer needed: the visual encoder trains by default.")
    if args.freeze_visual_encoder and not args.visual_pretrained:
        raise SystemExit("--freeze-visual-encoder needs --visual-pretrained (a frozen random encoder outputs noise).")

    CONFIG = {
        "input_dim": 40,
        "speaker_embedding_dim": 100,
        "output_speaker": SPEAKER_CAP,
        "audio_output_dim": 256,
        "video_embedding_dim": 256,
        "fusion_dim": 256,
        "num_attention_heads": 4,
        "num_cross_layers": 2,
        "num_self_layers": 2,
        "decoder_hidden_dim": 256,
        "dropout": 0.1,
        "visual_grad_checkpoint": args.visual_grad_checkpoint,
        "visual_pretrained_path": args.visual_pretrained or None,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Run dir: %s | device: %s%s", os.path.abspath(run_dir), device,
                f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else "")
    logger.info("Args: %s", vars(args))

    # ── data ────────────────────────────────────────────────────────────────
    loader_kwargs = dict(num_workers=args.num_workers, worker_init_fn=worker_seed_init, max_speakers=SPEAKER_CAP, max_session_speakers=SPEAKER_CAP)
    train_loader = get_dataloader(csv_path=args.train_csv, batch_size=args.batch_size, shuffle=True,
                                  augment=not args.no_augment, drop_last=True, **loader_kwargs)
    eval_loader = get_dataloader(data_dir=args.eval_dir, batch_size=args.eval_batch_size or args.batch_size,
                                 shuffle=False, chunk_frames=args.chunk_frames, stride_frames=args.chunk_frames,
                                 include_tail=True, augment=False, **loader_kwargs)

    stats = train_loader.dataset.label_stats()
    logger.info("Train chunks=%d (batches/epoch=%d) | eval chunks=%d (%d sessions)",
                len(train_loader.dataset), len(train_loader), len(eval_loader.dataset),
                eval_loader.dataset.df["session_id"].nunique())
    logger.info("Train speech ratio per present speaker=%.3f -> balanced BCE pos_weight would be %.2f (using %.2f)",
                stats["speech_ratio"], stats["suggested_pos_weight"], args.pos_weight)

    # ── model / optimiser ───────────────────────────────────────────────────
    model = AIVECTOR_ConformerVEmbedding_SD_JOINT(CONFIG, freeze_visual_encoder=args.freeze_visual_encoder).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: total=%.2fM trainable=%.2fM", n_total / 1e6, n_train / 1e6)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if param.ndim < 2 or "norm" in name.lower() else decay).append(param)
    optimizer = AdamW([{"params": decay, "weight_decay": args.weight_decay},
                       {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)
    if args.scheduler == "cosine":
        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / max(warmup_steps, 1)
            progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
            return args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.plateau_factor, patience=args.plateau_patience)

    use_amp = device.type == "cuda" and not args.disable_amp
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    logger.info("AMP=%s (%s) | scheduler=%s warmup_steps=%d total_steps=%d", use_amp,
                "bf16" if amp_dtype == torch.bfloat16 else "fp16", args.scheduler, warmup_steps, total_steps)

    start_epoch, best_der, epochs_no_improve, history = 1, float("inf"), 0, []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_der = ckpt.get("best_eval_der", best_der)
        metrics_path = os.path.join(run_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                history = [{k: (int(float(v)) if k == "epoch" else float(v)) for k, v in row.items()} for row in csv.DictReader(f)
                           if int(float(row["epoch"])) < start_epoch]
        logger.info("Resumed from %s at epoch %d (best eval DER so far %.2f%%)", args.resume, start_epoch, 100 * best_der)

    # ── epochs ──────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, n_batches, skipped, grad_norms = 0.0, 0, 0, []
        train_der = DERCounter([0.5])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(bar):
            audio, video, labels, mask, speaker_mask, valid, nframes = unpack(batch, device)
            if valid.sum() == 0:
                skipped += 1
                continue
            try:
                with amp_context(device, use_amp, amp_dtype):
                    logits = model(audio, None, video, nframes, mask=mask, speaker_mask=speaker_mask)
                loss = masked_loss(logits, labels, valid, args)
                if not torch.isfinite(loss):
                    logger.warning("Non-finite loss at epoch %d batch %d; skipping", epoch, batch_idx + 1)
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                scaler.scale(loss / args.grad_accum_steps).backward()
            except torch.cuda.OutOfMemoryError:
                logger.warning("CUDA OOM at epoch %d batch %d; skipping (try --visual-grad-checkpoint or smaller batch)",
                               epoch, batch_idx + 1)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                skipped += 1
                continue

            is_step = (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader)
            if is_step:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"))
                grad_norms.append(float(total_norm))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if args.scheduler == "cosine":
                    scheduler.step()

            loss_sum += loss.item()
            n_batches += 1
            train_der.update(logits.detach(), labels, valid)
            bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{loss_sum / n_batches:.4f}",
                            lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = loss_sum / max(n_batches, 1)
        train_res = train_der.result(0.5)
        ev = evaluate(model, eval_loader, device, args, use_amp, amp_dtype)
        if args.scheduler == "plateau":
            scheduler.step(ev["loss"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_loss,
            "train_der@0.5": train_res["der"],
            "eval_loss": ev["loss"],
            "eval_bce": ev["bce"],
            "eval_der@0.5": ev["at05"]["der"],
            "eval_miss@0.5": ev["at05"]["miss"],
            "eval_fa@0.5": ev["at05"]["fa"],
            "eval_conf@0.5": ev["at05"]["conf"],
            "eval_best_threshold": ev["best_threshold"],
            "eval_der_best": ev["best"]["der"],
            "grad_norm_median": float(np.median(grad_norms)) if grad_norms else float("nan"),
            "epoch_minutes": (time.perf_counter() - t0) / 60,
        }
        row = {k: (int(v) if k == "epoch" else float(v)) for k, v in row.items()}  # plain types: safe torch.load
        history.append(row)
        write_metrics(run_dir, history)

        peak_mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
        logger.info(
            "Epoch %d | train loss %.4f DER@0.5 %.2f%% | eval loss %.4f bce %.4f | "
            "eval DER@0.5 %.2f%% (miss %.2f fa %.2f conf %.2f) | best thr %.2f -> %.2f%% | "
            "lr %.2e | grad-norm med %.2f | skipped %d | peak mem %.1f GiB | %.1f min",
            epoch, train_loss, 100 * train_res["der"], ev["loss"], ev["bce"],
            100 * ev["at05"]["der"], 100 * ev["at05"]["miss"], 100 * ev["at05"]["fa"], 100 * ev["at05"]["conf"],
            ev["best_threshold"], 100 * ev["best"]["der"], row["lr"], row["grad_norm_median"], skipped,
            peak_mem, row["epoch_minutes"],
        )

        improved = ev["best"]["der"] < best_der
        if improved:
            best_der, epochs_no_improve = ev["best"]["der"], 0
        else:
            epochs_no_improve += 1

        save_checkpoint(os.path.join(run_dir, "last.pth"), model, optimizer, scheduler, scaler, epoch, CONFIG, row, best_der)
        if improved:
            save_checkpoint(os.path.join(run_dir, "best.pth"), model, optimizer, scheduler, scaler, epoch, CONFIG, row, best_der)
            logger.info("New best eval DER %.2f%% (threshold %.2f) -> best.pth", 100 * best_der, ev["best_threshold"])
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(os.path.join(run_dir, f"epoch_{epoch}.pth"), model, optimizer, scheduler, scaler, epoch, CONFIG, row, best_der)

        if args.early_stop_patience and epochs_no_improve >= args.early_stop_patience:
            logger.info("Early stopping: no eval-DER improvement for %d epochs", epochs_no_improve)
            break

    logger.info("Done. Best eval frame-DER %.2f%%. Outputs in %s", 100 * best_der, os.path.abspath(run_dir))


if __name__ == "__main__":
    train_avsd(parse_args())