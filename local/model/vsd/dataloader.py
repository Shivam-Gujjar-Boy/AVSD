import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _is_valid_path(path_value) -> bool:
    if path_value is None:
        return False
    if not isinstance(path_value, str):
        path_value = str(path_value)
    return path_value not in ("", "MISSING", "nan", "None")


class VSDSpeakerChunkDataset(Dataset):
    """One sample is one (chunk, speaker-track) pair."""

    def __init__(
        self,
        csv_path: str,
        max_speakers: int = 8,
        max_session_speakers: int = 5,
    ):
        self.csv_path = csv_path
        self.max_speakers = max_speakers
        self.max_session_speakers = max_session_speakers

        self.df = pd.read_csv(csv_path)
        self.df = self._filter_rows(self.df)
        self.samples = self._build_samples(self.df)

        print(
            f"Loaded VSD dataset: rows={len(self.df)}, samples={len(self.samples)}, "
            f"max_speakers={self.max_speakers}, max_session_speakers={self.max_session_speakers}"
        )

    def _infer_row_speaker_count(self, row: pd.Series) -> int:
        if "num_speakers" in row and pd.notna(row["num_speakers"]):
            try:
                return int(row["num_speakers"])
            except (TypeError, ValueError):
                pass

        count = 0
        for i in range(self.max_speakers):
            video_path = row.get(f"spk{i}_video_path", "MISSING")
            if _is_valid_path(video_path):
                count += 1
        return count

    def _filter_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        keep_mask = []
        for _, row in df.iterrows():
            spk_count = self._infer_row_speaker_count(row)
            keep_mask.append(spk_count <= self.max_session_speakers)

        filtered = df[np.array(keep_mask, dtype=bool)].reset_index(drop=True)
        removed = len(df) - len(filtered)
        print(
            f"Filtered rows for <= {self.max_session_speakers} speakers: kept={len(filtered)}, removed={removed}"
        )
        return filtered

    def _build_samples(self, df: pd.DataFrame) -> List[Dict]:
        samples: List[Dict] = []
        for row_idx, row in df.iterrows():
            start_f, end_f = int(row["start_frame"]), int(row["end_frame"])
            if end_f <= start_f:
                continue

            for spk_idx in range(self.max_speakers):
                video_path = row.get(f"spk{spk_idx}_video_path", "MISSING")
                label_path = row.get(f"spk{spk_idx}_label_path", "")

                if not (_is_valid_path(video_path) and _is_valid_path(label_path)):
                    continue
                if not (os.path.isfile(video_path) and os.path.isfile(label_path)):
                    continue

                samples.append(
                    {
                        "row_idx": row_idx,
                        "spk_idx": spk_idx,
                        "video_path": video_path,
                        "label_path": label_path,
                        "start_frame": start_f,
                        "end_frame": end_f,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        spk_idx = sample["spk_idx"]
        start_f = sample["start_frame"]
        end_f = sample["end_frame"]

        video = np.load(sample["video_path"])
        labels = np.load(sample["label_path"])

        video_chunk = video[start_f:end_f]

        if labels.ndim == 1:
            label_chunk = labels[start_f:end_f]
        elif labels.shape[1] > spk_idx:
            label_chunk = labels[start_f:end_f, spk_idx]
        else:
            label_chunk = np.zeros(end_f - start_f, dtype=np.float32)

        valid_len = min(len(video_chunk), len(label_chunk))
        if valid_len <= 0:
            raise ValueError(
                f"Empty chunk after slicing for sample idx={idx}, path={sample['video_path']}"
            )

        video_chunk = video_chunk[:valid_len]
        label_chunk = label_chunk[:valid_len]

        return {
            "video": torch.as_tensor(video_chunk, dtype=torch.float32),
            "labels": torch.as_tensor(label_chunk, dtype=torch.float32),
            "nframes": torch.tensor(valid_len, dtype=torch.long),
        }


def vsd_collate_fn(batch):
    lengths = [int(item["nframes"].item()) for item in batch]
    max_len = max(lengths)

    batch_size = len(batch)
    h = int(batch[0]["video"].shape[1])
    w = int(batch[0]["video"].shape[2])

    videos = torch.zeros((batch_size, max_len, h, w), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_len), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for i, item in enumerate(batch):
        cur_len = lengths[i]
        videos[i, :cur_len] = item["video"]
        labels[i, :cur_len] = item["labels"]
        mask[i, :cur_len] = 1.0

    return {
        "video": videos,
        "labels": labels,
        "mask": mask,
        "nframes": torch.tensor(lengths, dtype=torch.long),
    }


def get_dataloader(
    csv_path: str,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    max_speakers: int = 8,
    max_session_speakers: int = 5,
):
    dataset = VSDSpeakerChunkDataset(
        csv_path=csv_path,
        max_speakers=max_speakers,
        max_session_speakers=max_session_speakers,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=vsd_collate_fn,
    )
