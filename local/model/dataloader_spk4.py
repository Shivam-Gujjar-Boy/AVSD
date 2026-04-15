import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class AVDiarizationDataset(Dataset):
    def __init__(self, csv_path, max_speakers=8, max_session_speakers=None):
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
                        if video_path and pd.notna(video_path) and video_path != '' and video_path != 'MISSING':
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

        audio_features = np.load(row['audio_path'])
        start_f, end_f = row['start_frame'], row['end_frame']
        audio_chunk = audio_features[start_f:end_f]

        video_features = []
        diarization_labels = []

        for i in range(self.max_speakers):
            video_path = row.get(f'spk{i}_video_path', 'MISSING')
            label_path = row.get(f'spk{i}_label_path', '')

            if video_path and pd.notna(video_path) and video_path != '' and video_path != 'MISSING':
                video_frames = np.load(video_path)
                video_chunk = video_frames[start_f:end_f]
                video_features.append(video_chunk)

                labels = np.load(label_path)
                if labels.ndim == 1:
                    label_chunk = labels[start_f:end_f]
                elif labels.shape[1] > i:
                    label_chunk = labels[start_f:end_f, i]
                else:
                    label_chunk = np.zeros(end_f - start_f)
                diarization_labels.append(label_chunk)
            else:
                dummy_frames = np.zeros((end_f - start_f, 96, 96))
                video_features.append(dummy_frames)
                diarization_labels.append(np.zeros(end_f - start_f))

        audio_tensor = torch.FloatTensor(audio_chunk).permute(1, 0)
        video_tensor = torch.FloatTensor(np.array(video_features))
        labels_tensor = torch.FloatTensor(np.array(diarization_labels)).T

        return {
            'audio': audio_tensor,
            'video': video_tensor,
            'labels': labels_tensor,
            'nframes': torch.tensor(end_f - start_f)
        }


def get_dataloader(csv_path, batch_size=8, shuffle=True, num_workers=4, max_speakers=8, max_session_speakers=None):
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
        pin_memory=True
    )
