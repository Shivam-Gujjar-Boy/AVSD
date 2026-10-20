"""
AVSD dataloader (speaker cap 5), v2.

One sample = one chunk of session audio + the lip videos of every speaker in it.

Changes vs. the previous version:
  * Can build chunks directly from a dataset directory (used for the
    evaluation set), as well as from the training CSV.
  * Video scaling to [0, 1] is decided once per FILE (uint8 or max > 1), not
    per chunk, so every chunk of a file is scaled the same way.
  * Sessions whose label columns don't match the number of speaker videos are
    reported at start-up (a mismatch silently trains faces on wrong labels).
  * Chunk ends are clamped to the shortest of audio / labels / videos, and
    video chunks are length-checked, so short video files can't crash a batch.
  * Optional training augmentation: per-speaker horizontal flip, small
    translation, brightness/contrast jitter, and a frequency mask on audio.
  * label_stats(): speech ratio of present speakers, useful for pos_weight.

Expected session layout (training and evaluation):
    <session>/audio_features.npy          (T, 40)
    <session>/diarization_labels.npy      (T, N)   column i <-> i-th sorted spk_*.npy
    <session>/nframes.txt                 T
    <session>/video_features/spk_*.npy    (T, 96, 96)
"""
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_MAX_SPEAKERS = 8


def _is_valid_path(path_value) -> bool:
    """Check if a path value is valid (not None, not 'MISSING', etc.)"""
    if path_value is None:
        return False
    if not isinstance(path_value, str):
        path_value = str(path_value)
    return path_value not in ("", "MISSING", "nan", "None")


# ─────────────────────────────────────────────────────────────────────────────
# Chunk table construction (same columns as create_training_chunks.py output)
# ─────────────────────────────────────────────────────────────────────────────

def _sorted_speaker_videos(session_dir: Path) -> List[Path]:
    vdir = session_dir / "video_features"
    return sorted(vdir.glob("spk_*.npy")) if vdir.exists() else []


def build_chunk_table(
    data_dir: str,
    chunk_frames: int = 200,
    stride_frames: int = 100,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    include_tail: bool = False,
) -> pd.DataFrame:
    """Scan session folders and return a chunk table identical in format to the training CSV.

    include_tail=True also adds a final shorter chunk so every frame is covered
    (useful for evaluation; the collate function pads it).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    rows: List[Dict] = []
    for session_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        audio_path = session_dir / "audio_features.npy"
        labels_path = session_dir / "diarization_labels.npy"
        nframes_path = session_dir / "nframes.txt"
        if not (audio_path.exists() and labels_path.exists() and nframes_path.exists()):
            warnings.warn(f"Skipping {session_dir.name}: missing audio/labels/nframes")
            continue

        videos = _sorted_speaker_videos(session_dir)
        if not videos:
            warnings.warn(f"Skipping {session_dir.name}: no speaker videos")
            continue
        if len(videos) > max_speakers:
            warnings.warn(f"{session_dir.name}: {len(videos)} speakers > max_speakers={max_speakers}; extra columns dropped")

        with open(nframes_path) as f:
            total = int(f.read().strip())
        lengths = [total, len(np.load(audio_path, mmap_mode="r")), len(np.load(labels_path, mmap_mode="r"))]
        lengths += [len(np.load(v, mmap_mode="r")) for v in videos]
        total = min(lengths)

        starts = list(range(0, max(total - chunk_frames, -1) + 1, stride_frames))
        if include_tail:
            if not starts:
                starts = [0]
            elif starts[-1] + chunk_frames < total:
                starts.append(starts[-1] + stride_frames)

        for start in starts:
            end = min(start + chunk_frames, total)
            if end <= start:
                continue
            row = {
                "session_id": session_dir.name,
                "start_frame": start,
                "end_frame": end,
                "audio_path": str(audio_path),
                "num_speakers": len(videos),
            }
            for i in range(max_speakers):
                row[f"spk{i}_video_path"] = str(videos[i]) if i < len(videos) else ""
                row[f"spk{i}_label_path"] = str(labels_path) if i < len(videos) else ""
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No chunks could be built from {data_dir}")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AVDiarizationDataset(Dataset):
    """Audio-Visual Diarization Dataset - one sample is one (audio_chunk, all_speaker_videos)"""

    def __init__(
        self,
        csv_path: Optional[str] = None,
        max_speakers: int = DEFAULT_MAX_SPEAKERS,
        max_session_speakers: Optional[int] = None,
        data_dir: Optional[str] = None,
        chunk_frames: int = 200,
        stride_frames: int = 100,
        include_tail: bool = False,
        augment: bool = False,
    ):
        if (csv_path is None) == (data_dir is None):
            raise ValueError("Give exactly one of csv_path or data_dir")

        self.max_speakers = max_speakers
        self.max_session_speakers = max_session_speakers
        self.augment = augment
        self._video_scale_cache: Dict[str, float] = {}

        if csv_path is not None:
            self.source = csv_path
            self.df = pd.read_csv(csv_path)
        else:
            self.source = data_dir
            self.df = build_chunk_table(
                data_dir, chunk_frames, stride_frames,
                max_speakers=max(max_speakers, max_session_speakers or 0, DEFAULT_MAX_SPEAKERS),
                include_tail=include_tail,
            )

        total_chunks = len(self.df)
        self._video_cols = sorted(
            (c for c in self.df.columns if c.startswith("spk") and c.endswith("_video_path")),
            key=lambda c: int(c[3:].split("_")[0]),
        )
        self.df["_n_spk"] = [self._count_speakers(row) for _, row in self.df.iterrows()]
        if self.max_session_speakers is not None:
            self.df = self.df[self.df["_n_spk"] <= self.max_session_speakers].reset_index(drop=True)
        kept = len(self.df)
        print(
            f"✅ Loaded {kept}/{total_chunks} chunks from {self.source} "
            + (f"(filtered {total_chunks - kept} with >{self.max_session_speakers} speakers) "
               if self.max_session_speakers is not None else "")
            + f"sessions: {self.df['session_id'].nunique() if 'session_id' in self.df else '?'}, "
            f"max_speakers: {max_speakers}"
        )
        self._check_label_columns()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _count_speakers(self, row) -> int:
        return sum(_is_valid_path(row.get(c, "MISSING")) for c in self._video_cols)

    def _check_label_columns(self):
        """Warn if a session's label matrix doesn't have one column per speaker video."""
        seen = set()
        for _, row in self.df.iterrows():
            label_path = row.get("spk0_label_path", "")
            if not _is_valid_path(label_path) or label_path in seen:
                continue
            seen.add(label_path)
            labels = np.load(label_path, mmap_mode="r")
            n_cols = 1 if labels.ndim == 1 else labels.shape[1]
            if n_cols != row["_n_spk"]:
                warnings.warn(
                    f"{label_path}: {n_cols} label columns but {row['_n_spk']} speaker videos. "
                    "Speakers without a column get all-zero labels; check the column order too."
                )

    def _video_scale(self, path: str, frames) -> float:
        """1/255 for 0-255 data, 1.0 for data already in [0, 1]; decided once per file."""
        if path not in self._video_scale_cache:
            if frames.dtype == np.uint8:
                scale = 1.0 / 255.0
            else:
                probe = np.asarray(frames[: min(len(frames), 500)])
                scale = 1.0 / 255.0 if probe.size and probe.max() > 1.0 else 1.0
            self._video_scale_cache[path] = scale
        return self._video_scale_cache[path]

    def label_stats(self) -> Dict[str, float]:
        """Fraction of frames in which a present speaker is speaking (over unique sessions)."""
        active = total = 0.0
        seen = set()
        for _, row in self.df.iterrows():
            label_path = row.get("spk0_label_path", "")
            if not _is_valid_path(label_path) or label_path in seen:
                continue
            seen.add(label_path)
            labels = np.asarray(np.load(label_path, mmap_mode="r"), dtype=np.float32)
            labels = labels[:, None] if labels.ndim == 1 else labels[:, : row["_n_spk"]]
            active += float((labels > 0.5).sum())
            total += float(labels.size)
        ratio = active / max(total, 1.0)
        return {"speech_ratio": ratio, "suggested_pos_weight": (1.0 - ratio) / max(ratio, 1e-6)}

    def _augment_video(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: [T, H, W] in [0, 1]; same transform for all frames of one speaker."""
        if np.random.rand() < 0.5:
            chunk = chunk[:, :, ::-1]
        max_shift = 4
        dy, dx = np.random.randint(-max_shift, max_shift + 1, size=2)
        if dy or dx:
            padded = np.pad(chunk, ((0, 0), (max_shift, max_shift), (max_shift, max_shift)), mode="edge")
            H, W = chunk.shape[1:]
            chunk = padded[:, max_shift + dy: max_shift + dy + H, max_shift + dx: max_shift + dx + W]
        contrast = np.random.uniform(0.8, 1.2)
        brightness = np.random.uniform(-0.08, 0.08)
        mean = chunk.mean()
        chunk = np.clip((chunk - mean) * contrast + mean + brightness, 0.0, 1.0)
        return np.ascontiguousarray(chunk, dtype=np.float32)

    def _augment_audio(self, chunk: np.ndarray) -> np.ndarray:
        """chunk: [T, F]; one random frequency mask (SpecAugment-style)."""
        chunk = chunk.copy()
        F = chunk.shape[1]
        width = np.random.randint(0, 6)
        if width:
            f0 = np.random.randint(0, F - width + 1)
            chunk[:, f0: f0 + width] = chunk.mean()
        return chunk

    # ── Dataset API ─────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        start_f, end_f = int(row["start_frame"]), int(row["end_frame"])
        T = end_f - start_f

        audio_features = np.load(row["audio_path"], mmap_mode="r")
        audio_chunk = np.array(audio_features[start_f:end_f], dtype=np.float32)
        if len(audio_chunk) != T:
            raise ValueError(f"{row['audio_path']}: expected {T} frames from {start_f}, got {len(audio_chunk)}")
        if self.augment:
            audio_chunk = self._augment_audio(audio_chunk)

        video_features, diarization_labels, speaker_mask = [], [], []
        H = W = 96

        for i in range(self.max_speakers):
            video_path = row.get(f"spk{i}_video_path", "MISSING")
            label_path = row.get(f"spk{i}_label_path", "")

            if _is_valid_path(video_path) and _is_valid_path(label_path):
                video_frames = np.load(video_path, mmap_mode="r")
                video_chunk = np.array(video_frames[start_f:end_f], dtype=np.float32)
                video_chunk *= self._video_scale(video_path, video_frames)
                if len(video_chunk) < T:  # short video file: pad with the last frame
                    pad = np.repeat(video_chunk[-1:] if len(video_chunk) else np.zeros((1, H, W), np.float32),
                                    T - len(video_chunk), axis=0)
                    video_chunk = np.concatenate([video_chunk, pad], axis=0)
                H, W = video_chunk.shape[1:]
                if self.augment:
                    video_chunk = self._augment_video(video_chunk)
                video_features.append(video_chunk[:, None])          # [T, 1, H, W]
                speaker_mask.append(1.0)

                labels = np.load(label_path, mmap_mode="r")
                if labels.ndim == 1:
                    label_chunk = np.array(labels[start_f:end_f], dtype=np.float32)
                elif labels.shape[1] > i:
                    label_chunk = np.array(labels[start_f:end_f, i], dtype=np.float32)
                else:
                    label_chunk = np.zeros(T, dtype=np.float32)
                if len(label_chunk) < T:
                    label_chunk = np.pad(label_chunk, (0, T - len(label_chunk)))
                diarization_labels.append(label_chunk)
            else:
                video_features.append(None)
                diarization_labels.append(np.zeros(T, dtype=np.float32))
                speaker_mask.append(0.0)

        # Dummy (absent) speakers: zeros with the real frame size. The model skips them.
        video_features = [v if v is not None else np.zeros((T, 1, H, W), np.float32) for v in video_features]

        return {
            "audio": torch.from_numpy(audio_chunk).permute(1, 0).contiguous(),   # [40, T]
            "video": torch.from_numpy(np.stack(video_features)),                # [S, T, 1, H, W]
            "labels": torch.from_numpy(np.stack(diarization_labels, axis=1)),   # [T, S]
            "speaker_mask": torch.tensor(speaker_mask, dtype=torch.float32),
            "nframes": torch.tensor(T, dtype=torch.long),
        }


def avsd_collate_fn(batch):
    """Custom collate function for AVSD batches with dynamic padding and masking"""
    lengths = [int(item["nframes"].item()) for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    audio_channels = int(batch[0]["audio"].shape[0])
    num_speakers = int(batch[0]["video"].shape[0])
    c, h, w = (int(s) for s in batch[0]["video"].shape[2:])

    audios = torch.zeros((batch_size, audio_channels, max_len), dtype=torch.float32)
    videos = torch.zeros((batch_size, num_speakers, max_len, c, h, w), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_len, num_speakers), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    speaker_masks = torch.zeros((batch_size, num_speakers), dtype=torch.float32)

    for i, item in enumerate(batch):
        cur_len = lengths[i]
        audios[i, :, :cur_len] = item["audio"]
        videos[i, :, :cur_len] = item["video"]
        labels[i, :cur_len, :] = item["labels"]
        mask[i, :cur_len] = 1.0
        speaker_masks[i] = item["speaker_mask"]

    return {
        "audio": audios,
        "video": videos,
        "labels": labels,
        "mask": mask,
        "speaker_mask": speaker_masks,
        "nframes": torch.tensor(lengths, dtype=torch.long),
    }


def get_dataloader(
    csv_path=None,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    max_speakers=DEFAULT_MAX_SPEAKERS,
    max_session_speakers=None,
    data_dir=None,
    chunk_frames=200,
    stride_frames=100,
    include_tail=False,
    augment=False,
    drop_last=False,
    worker_init_fn=None,
):
    """Create a DataLoader from a chunk CSV (training) or a session directory (e.g. evaluation)."""
    dataset = AVDiarizationDataset(
        csv_path=csv_path,
        max_speakers=max_speakers,
        max_session_speakers=max_session_speakers,
        data_dir=data_dir,
        chunk_frames=chunk_frames,
        stride_frames=stride_frames,
        include_tail=include_tail,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn,
        collate_fn=avsd_collate_fn,
    )