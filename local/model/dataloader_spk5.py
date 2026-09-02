import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def _is_valid_path(path_value) -> bool:
    """Check if a path value is valid (not None, not 'MISSING', etc.)"""
    if path_value is None:
        return False
    if not isinstance(path_value, str):
        path_value = str(path_value)
    return path_value not in ("", "MISSING", "nan", "None")


class AVDiarizationDataset(Dataset):
    """Audio-Visual Diarization Dataset - one sample is one (audio_chunk, all_speaker_videos)"""

    def __init__(self, csv_path, max_speakers=8, max_session_speakers=None):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.max_speakers = max_speakers
        self.max_session_speakers = max_session_speakers

        total_chunks = len(self.df)
        if self.max_session_speakers is not None:
            if 'num_speakers' in self.df.columns:
                speaker_counts = pd.to_numeric(self.df['num_speakers'], errors='coerce').fillna(0)
                self.df = self.df[speaker_counts <= self.max_session_speakers].reset_index(drop=True)
            else:
                keep_rows = []
                for _, row in self.df.iterrows():
                    count = 0
                    for i in range(8):
                        video_path = row.get(f'spk{i}_video_path', 'MISSING')
                        if _is_valid_path(video_path):
                            count += 1
                    keep_rows.append(count <= self.max_session_speakers)
                self.df = self.df[np.array(keep_rows, dtype=bool)].reset_index(drop=True)

        kept_chunks = len(self.df)
        removed_chunks = total_chunks - kept_chunks
        if self.max_session_speakers is not None:
            print(
                f"✅ Loaded dataset with {kept_chunks}/{total_chunks} chunks "
                f"(filtered out {removed_chunks} chunks with >{self.max_session_speakers} speakers), "
                f"max_speakers: {max_speakers}"
            )
        else:
            print(f"✅ Loaded dataset with {kept_chunks} chunks, max_speakers: {max_speakers}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load audio features
        audio_features = np.load(row['audio_path'], mmap_mode='r')
        start_f, end_f = int(row['start_frame']), int(row['end_frame'])
        audio_chunk = np.array(audio_features[start_f:end_f]).astype(np.float32)

        video_features = []
        diarization_labels = []

        for i in range(self.max_speakers):
            video_path = row.get(f'spk{i}_video_path', 'MISSING')
            label_path = row.get(f'spk{i}_label_path', '')

            if _is_valid_path(video_path) and _is_valid_path(label_path):
                # Load video frames with mmap for memory efficiency
                video_frames = np.load(video_path, mmap_mode='r')
                video_chunk = np.array(video_frames[start_f:end_f]).astype(np.float32)
                
                # Normalize video to [0, 1]
                if video_chunk.max() > 1.0:
                    video_chunk = video_chunk / 255.0

                # Add channel dimension: [T, H, W] -> [T, 1, H, W]
                video_chunk = np.expand_dims(video_chunk, axis=1)
                video_features.append(video_chunk)

                # Load labels
                labels = np.load(label_path, mmap_mode='r')
                if labels.ndim == 1:
                    label_chunk = np.array(labels[start_f:end_f]).astype(np.float32)
                elif labels.shape[1] > i:
                    label_chunk = np.array(labels[start_f:end_f, i]).astype(np.float32)
                else:
                    label_chunk = np.zeros(end_f - start_f, dtype=np.float32)
                diarization_labels.append(label_chunk)
            else:
                # Create dummy data for missing speakers
                dummy_frames = np.zeros((end_f - start_f, 1, 96, 96), dtype=np.float32)
                video_features.append(dummy_frames)
                diarization_labels.append(np.zeros(end_f - start_f, dtype=np.float32))

        # Convert to tensors
        # audio: [T, 40]
        audio_tensor = torch.as_tensor(audio_chunk, dtype=torch.float32).permute(1, 0)  # [40, T]
        
        # video: [max_speakers, T, 1, 96, 96]
        video_tensor = torch.as_tensor(np.array(video_features), dtype=torch.float32)
        
        # labels: [T, max_speakers]
        labels_tensor = torch.as_tensor(np.array(diarization_labels).T, dtype=torch.float32)

        return {
            'audio': audio_tensor,
            'video': video_tensor,
            'labels': labels_tensor,
            'nframes': torch.tensor(end_f - start_f, dtype=torch.long)
        }


def avsd_collate_fn(batch):
    """Custom collate function for AVSD batches with dynamic padding and masking"""
    # Extract batch elements
    lengths = [int(item['nframes'].item()) for item in batch]
    max_len = max(lengths)

    batch_size = len(batch)
    
    # Get dimensions from first sample
    audio_channels = int(batch[0]['audio'].shape[0])
    num_speakers = int(batch[0]['video'].shape[0])
    c = int(batch[0]['video'].shape[2])
    h = int(batch[0]['video'].shape[3])
    w = int(batch[0]['video'].shape[4])

    # Initialize tensors with padding
    audios = torch.zeros((batch_size, audio_channels, max_len), dtype=torch.float32)
    videos = torch.zeros((batch_size, num_speakers, max_len, c, h, w), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_len, num_speakers), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

    # Fill tensors
    for i, item in enumerate(batch):
        cur_len = lengths[i]
        audios[i, :, :cur_len] = item['audio']
        videos[i, :, :cur_len, :, :, :] = item['video']
        labels[i, :cur_len, :] = item['labels']
        mask[i, :cur_len] = 1.0

    return {
        'audio': audios,
        'video': videos,
        'labels': labels,
        'mask': mask,
        'nframes': torch.tensor(lengths, dtype=torch.long),
    }


def get_dataloader(csv_path, batch_size=8, shuffle=True, num_workers=4, max_speakers=8, max_session_speakers=None):
    """Create DataLoader for AVSD training"""
    dataset = AVDiarizationDataset(
        csv_path,
        max_speakers=max_speakers,
        max_session_speakers=max_session_speakers,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=avsd_collate_fn,
    )
