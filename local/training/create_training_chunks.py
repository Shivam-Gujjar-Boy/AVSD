import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import List, Dict

class TrainingCSVCreator:
    def __init__(self, dataset_dir: str, output_csv: str, chunk_duration: float = 8.0, stride: float = 4.0, fps: int = 25):
        self.dataset_dir = Path(dataset_dir)
        self.output_csv = output_csv
        self.chunk_duration = chunk_duration
        self.stride = stride
        self.fps = fps
        
        self.chunk_frames = int(chunk_duration * fps)
        self.stride_frames = int(stride * fps)
        
    def find_max_speakers(self) -> int:
        """Find maximum number of speakers across all sessions"""
        max_speakers = 0
        for session_dir in self.dataset_dir.iterdir():
            if session_dir.is_dir():
                video_features_dir = session_dir / "video_features"
                if video_features_dir.exists():
                    speaker_files = list(video_features_dir.glob("spk_*.npy"))
                    max_speakers = max(max_speakers, len(speaker_files))
        print(f"Maximum speakers found: {max_speakers}")
        return max_speakers
    
    def get_session_speakers(self, session_dir: Path) -> List[str]:
        """Get list of speaker IDs for a session"""
        video_features_dir = session_dir / "video_features"
        if not video_features_dir.exists():
            return []
        
        speaker_files = list(video_features_dir.glob("spk_*.npy"))
        speaker_ids = [f.stem for f in speaker_files]  # Get filenames without extension
        return sorted(speaker_ids)
    
    def create_chunks_for_session(self, session_dir: Path, max_speakers: int) -> List[Dict]:
        """Create overlapping chunks for a single session"""
        session_id = session_dir.name
        
        # Check if required files exist
        audio_path = session_dir / "audio_features.npy"
        labels_path = session_dir / "diarization_labels.npy"
        nframes_path = session_dir / "nframes.txt"
        
        if not (audio_path.exists() and labels_path.exists() and nframes_path.exists()):
            print(f"⚠ Skipping {session_id}: Missing required files")
            return []
        
        # Load data
        try:
            with open(nframes_path, 'r') as f:
                total_frames = int(f.read().strip())
            
            labels = np.load(labels_path)
            speaker_ids = self.get_session_speakers(session_dir)
            num_speakers = len(speaker_ids)
            
            print(f"Processing {session_id}: {total_frames} frames, {num_speakers} speakers")
            
        except Exception as e:
            print(f"❌ Error loading data for {session_id}: {e}")
            return []
        
        chunks = []
        start_frame = 0
        
        while start_frame + self.chunk_frames <= total_frames:
            end_frame = start_frame + self.chunk_frames
            
            # Calculate times in seconds
            start_time = start_frame / self.fps
            end_time = end_frame / self.fps
            
            # Create chunk entry
            chunk_data = {
                'session_id': session_id,
                'start_time': start_time,
                'end_time': end_time,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'audio_path': str(audio_path),
                'num_speakers': num_speakers
            }
            
            # Add speaker-specific paths and labels
            for i, speaker_id in enumerate(speaker_ids):
                video_path = session_dir / "video_features" / f"{speaker_id}.npy"
                
                chunk_data[f'spk{i}_video_path'] = str(video_path)
                chunk_data[f'spk{i}_label_path'] = str(labels_path)
                chunk_data[f'spk{i}_start_frame'] = start_frame
                chunk_data[f'spk{i}_end_frame'] = end_frame
            
            # Fill remaining speaker columns with empty values
            for i in range(len(speaker_ids), max_speakers):
                chunk_data[f'spk{i}_video_path'] = ''
                chunk_data[f'spk{i}_label_path'] = ''  
                chunk_data[f'spk{i}_start_frame'] = -1
                chunk_data[f'spk{i}_end_frame'] = -1
            
            chunks.append(chunk_data)
            start_frame += self.stride_frames
        
        print(f"  Created {len(chunks)} chunks for {session_id}")
        return chunks
    
    def create_training_csv(self):
        """Create the complete training CSV file"""
        print("Creating training CSV with overlapping chunks...")
        print(f"Chunk duration: {self.chunk_duration}s, Stride: {self.stride}s, FPS: {self.fps}")
        
        # Find maximum number of speakers
        max_speakers = self.find_max_speakers()
        if max_speakers == 0:
            print("❌ No sessions found!")
            return
        
        all_chunks = []
        
        # Process each session
        session_dirs = sorted([d for d in self.dataset_dir.iterdir() if d.is_dir()])
        print(f"Found {len(session_dirs)} sessions")
        
        for session_dir in session_dirs:
            chunks = self.create_chunks_for_session(session_dir, max_speakers)
            all_chunks.extend(chunks)
        
        # Create DataFrame
        if not all_chunks:
            print("❌ No chunks created!")
            return
        
        df = pd.DataFrame(all_chunks)
        
        # Reorder columns for better readability
        base_columns = ['session_id', 'start_time', 'end_time', 'start_frame', 'end_frame', 
                       'audio_path', 'num_speakers']
        
        speaker_columns = []
        for i in range(max_speakers):
            speaker_columns.extend([f'spk{i}_video_path', f'spk{i}_label_path', 
                                  f'spk{i}_start_frame', f'spk{i}_end_frame'])
        
        final_columns = base_columns + speaker_columns
        df = df[final_columns]
        
        # Save CSV
        df.to_csv(self.output_csv, index=False)
        print(f"✅ Successfully created {self.output_csv}")
        print(f"📊 Total chunks: {len(df)}")
        print(f"👥 Maximum speakers: {max_speakers}")
        print(f"📋 Columns: {len(df.columns)}")
        
        # Show sample
        print("\n📄 Sample of the CSV:")
        print(df.head(3).to_string(max_cols=15))  # Show first few columns only
        
        return df

# Usage
if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "/home/speech-audio-research/22b3965/train-bin/tran-bin/modified-train"  # Your converted dataset path
    OUTPUT_CSV = "/home/speech-audio-research/22b3965/AVSD/local/training/training_chunks.csv"
    
    # Create CSV creator
    creator = TrainingCSVCreator(
        dataset_dir=DATASET_DIR,
        output_csv=OUTPUT_CSV,
        chunk_duration=8.0,  # 8-second chunks
        stride=4.0,          # 4-second stride (50% overlap)
        fps=25               # 25 FPS
    )
    
    # Generate the CSV
    df = creator.create_training_csv()
