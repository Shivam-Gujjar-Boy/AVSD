import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os

class AVDiarizationDataset(Dataset):
    def __init__(self, csv_path, max_speakers=8):
        """
        Dataset for Audio-Visual Speaker Diarization
        
        Args:
            csv_path: Path to training_chunks.csv
            max_speakers: Maximum number of speakers across all sessions
        """
        self.df = pd.read_csv(csv_path)
        self.max_speakers = max_speakers
        print(f"✅ Loaded dataset with {len(self.df)} chunks, max_speakers: {max_speakers}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio features
        audio_features = np.load(row['audio_path'])
        start_f, end_f = row['start_frame'], row['end_frame']
        audio_chunk = audio_features[start_f:end_f]  # (chunk_frames, 40)
        
        # Load video features and labels for each speaker
        video_features = []
        diarization_labels = []
        
        for i in range(self.max_speakers):
            video_path = row.get(f'spk{i}_video_path', 'MISSING')
            label_path = row.get(f'spk{i}_label_path', '')
            
            if video_path and pd.notna(video_path) and video_path != '' and video_path != 'MISSING':
                # Load video frames
                video_frames = np.load(video_path)
                video_chunk = video_frames[start_f:end_f]  # (chunk_frames, 96, 96)
                video_features.append(video_chunk)
                
                # Load labels
                labels = np.load(label_path)
                if labels.ndim == 1:
                    label_chunk = labels[start_f:end_f]
                elif labels.shape[1] > i:
                    label_chunk = labels[start_f:end_f, i]  # Binary labels for this speaker
                else:
                    label_chunk = np.zeros(end_f - start_f)
                diarization_labels.append(label_chunk)
            else:
                # Create dummy data for missing speakers
                dummy_frames = np.zeros((end_f-start_f, 96, 96))
                video_features.append(dummy_frames)
                diarization_labels.append(np.zeros(end_f-start_f))
        
        # Convert to tensors
        audio_tensor = torch.FloatTensor(audio_chunk).permute(1, 0)  # (40, chunk_frames)
        video_tensor = torch.FloatTensor(np.array(video_features))   # (max_speakers, chunk_frames, 96, 96)
        labels_tensor = torch.FloatTensor(np.array(diarization_labels)).T  # (chunk_frames, max_speakers)
        
        return {
            'audio': audio_tensor,
            'video': video_tensor, 
            'labels': labels_tensor,
            'nframes': torch.tensor(end_f - start_f)
        }

def get_dataloader(csv_path, batch_size=8, shuffle=True, num_workers=4, max_speakers=8):
    """
    Create DataLoader for training
    
    Args:
        csv_path: Path to training_chunks.csv
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        num_workers: Number of parallel data loading workers
        max_speakers: Maximum speakers (should match your model config)
    """
    dataset = AVDiarizationDataset(csv_path, max_speakers=max_speakers)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader

# Quick test function
def test_dataloader():
    """Test the dataloader"""
    CSV_PATH = "/home/speech-audio-research/22b3965/AVSD/local/training/training_chunks.csv"
    
    print("🧪 Testing DataLoader...")
    dataloader = get_dataloader(CSV_PATH, batch_size=2, shuffle=False)
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"  Audio: {batch['audio'].shape}")    # (2, 40, 200)
        print(f"  Video: {batch['video'].shape}")    # (2, 4, 200, 96, 96)
        print(f"  Labels: {batch['labels'].shape}")  # (2, 200, 4)
        print(f"  Nframes: {batch['nframes']}")      # [200, 200]
        
        if i == 1:  # Just check first 2 batches
            break
    
    print("✅ DataLoader test completed!")

if __name__ == "__main__":
    test_dataloader()
