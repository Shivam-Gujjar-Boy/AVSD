# -*- coding: utf-8 -*-
import torch
import numpy as np
import pandas as pd
import os
from torch.nn.utils.rnn import pad_sequence

class CSVAudioVideoDataReader(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_speakers=6):
        self.df = pd.read_csv(csv_path)
        self.max_speakers = max_speakers
        print(f"Loaded {len(self.df)} training chunks from {csv_path}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load audio chunk
        audio_features = np.load(row['audio_path'])
        start_f, end_f = int(row['start_frame']), int(row['end_frame'])
        audio_chunk = audio_features[start_f:end_f]  # [T, F]
        
        # Load video chunks and labels for each speaker
        video_features = []
        labels = []
        
        num_speakers = min(int(row['num_speakers']), self.max_speakers)
        
        for i in range(num_speakers):
            video_path = row[f'spk{i}_video_path']
            label_path = row[f'spk{i}_label_path']
            
            if video_path != "MISSING" and os.path.exists(video_path):
                video_frames = np.load(video_path)[start_f:end_f]  # [T, H, W]
                speaker_labels = np.load(label_path)[start_f:end_f]  # [T] - binary labels
                
                video_features.append(video_frames)
                labels.append(speaker_labels)
        
        # Convert to numpy arrays
        if video_features:
            video_features = np.stack(video_features)  # [num_speakers, T, H, W]
            labels = np.stack(labels)  # [num_speakers, T]
        else:
            # Handle case with no speakers
            video_features = np.zeros((0, end_f-start_f, 96, 96))
            labels = np.zeros((0, end_f-start_f))
        
        # Pad to max_speakers if needed
        if video_features.shape[0] < self.max_speakers:
            pad_count = self.max_speakers - video_features.shape[0]
            video_pad = np.zeros((pad_count, video_features.shape[1], video_features.shape[2], video_features.shape[3]))
            label_pad = np.zeros((pad_count, labels.shape[1]))
            
            video_features = np.concatenate([video_features, video_pad], axis=0)
            labels = np.concatenate([labels, label_pad], axis=0)
        
        # Create dummy speaker embeddings (zeros)
        speaker_embeddings = np.zeros((self.max_speakers, 256))
        
        return audio_chunk, speaker_embeddings, video_features, labels

def csv_collate_fn(batch):
    """Collate function for the CSV data reader"""
    audio_feas, audio_embeddings, video_embeddings, mask_labels = [], [], [], []
    
    for audio, emb, video, label in batch:
        audio_feas.append(torch.FloatTensor(audio))
        audio_embeddings.append(torch.FloatTensor(emb))
        video_embeddings.append(torch.FloatTensor(video))
        mask_labels.append(torch.FloatTensor(label))
    
    # Pad sequences
    audio_feas = pad_sequence(audio_feas, batch_first=True).transpose(1, 2)
    audio_embeddings = torch.stack(audio_embeddings)
    video_embeddings = pad_sequence(video_embeddings, batch_first=True)
    
    # Reshape mask_labels to match model expectations
    mask_labels_reshaped = []
    for label in mask_labels:
        # [num_speakers, T] -> [T, num_speakers] for loss function
        mask_labels_reshaped.append(label.permute(1, 0))
    mask_labels = torch.cat(mask_labels_reshaped, dim=0)  # [Total_T, num_speakers]
    
    nframes = [audio.shape[1] for audio in audio_feas.transpose(1, 2)]
    
    return audio_feas, audio_embeddings, video_embeddings, mask_labels, nframes