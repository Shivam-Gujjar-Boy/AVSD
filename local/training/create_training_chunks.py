# create_training_chunks.py
import pandas as pd
import numpy as np
import os
import glob

def create_training_chunks(session_dirs, chunk_duration=8.0, frame_rate=25.0, overlap=0.5):
    """
    Create training chunks CSV from your dataset structure
    
    Args:
        session_dirs: List of paths to session directories
        chunk_duration: Duration of each chunk in seconds (default: 8.0)
        frame_rate: Video frame rate (default: 25.0 fps)
        overlap: Overlap between consecutive chunks (0.0 to 1.0, default: 0.5)
    
    Returns:
        pandas.DataFrame: DataFrame containing all training chunks
    """
    chunks = []
    
    for session_dir in session_dirs:
        print(f"Processing {session_dir}...")
        
        # Check if session directory exists
        if not os.path.isdir(session_dir):
            print(f"Warning: {session_dir} not found, skipping...")
            continue
            
        # Calculate total duration from audio features
        audio_path = os.path.join(session_dir, 'audio_features.npy')
        if not os.path.exists(audio_path):
            print(f"Warning: {audio_path} not found, skipping session...")
            continue
            
        try:
            audio_features = np.load(audio_path)
            total_frames = audio_features.shape[0]
            total_duration = total_frames / frame_rate
        except Exception as e:
            print(f"Error loading {audio_path}: {e}, skipping session...")
            continue
        
        # Get speaker files
        speaker_dir = os.path.join(session_dir, 'speakers')
        label_dir = os.path.join(session_dir, 'labels')
        
        if not os.path.exists(speaker_dir) or not os.path.exists(label_dir):
            print(f"Warning: speaker or label directory not found in {session_dir}, skipping...")
            continue
            
        speaker_files = sorted([f for f in os.listdir(speaker_dir) if f.endswith('_frames.npy')])
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('_labels.npy')])
        
        if len(speaker_files) != len(label_files):
            print(f"Warning: Mismatch between speaker files ({len(speaker_files)}) and label files ({len(label_files)}) in {session_dir}")
            continue
            
        num_speakers = len(speaker_files)
        
        # Verify all speaker files have corresponding label files
        for spk_file in speaker_files:
            spk_id = spk_file.replace('_frames.npy', '')
            expected_label_file = f"{spk_id}_labels.npy"
            if expected_label_file not in label_files:
                print(f"Warning: Missing label file {expected_label_file} for speaker {spk_file}")
        
        # Create overlapping chunks
        chunk_frames = int(chunk_duration * frame_rate)
        step_frames = int(chunk_frames * (1 - overlap))
        
        if chunk_frames > total_frames:
            print(f"Warning: Chunk duration {chunk_duration}s exceeds total duration {total_duration:.2f}s for {session_dir}")
            continue
            
        for start_frame in range(0, total_frames - chunk_frames + 1, step_frames):
            end_frame = start_frame + chunk_frames
            start_time = start_frame / frame_rate
            end_time = end_frame / frame_rate
            
            chunk_data = {
                'session_id': os.path.basename(session_dir),
                'start_time': start_time,
                'end_time': end_time,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'audio_path': os.path.abspath(audio_path),
                'num_speakers': num_speakers
            }
            
            # Add speaker-specific paths
            for i, spk_file in enumerate(speaker_files):
                spk_id = spk_file.replace('_frames.npy', '')
                chunk_data[f'spk{i}_video_path'] = os.path.abspath(os.path.join(speaker_dir, spk_file))
                
                # Find corresponding label file
                label_file = f"{spk_id}_labels.npy"
                label_path = os.path.join(label_dir, label_file)
                if os.path.exists(label_path):
                    chunk_data[f'spk{i}_label_path'] = os.path.abspath(label_path)
                else:
                    print(f"Warning: Label file {label_path} not found")
                    chunk_data[f'spk{i}_label_path'] = "MISSING"
            
            chunks.append(chunk_data)
        
        print(f"  Created {len([c for c in chunks if c['session_id'] == os.path.basename(session_dir)])} chunks for {session_dir}")
    
    return pd.DataFrame(chunks)

def main():
    # Path to your dataset relative to this script
    dataset_base_path = "../../../data-bin/data-bin/modified_dev/dev/"
    
    # Get all session directories (session_1 to session_25)
    session_dirs = []
    for i in range(40, 45):
        session_path = os.path.join(dataset_base_path, f"session_{i}")
        session_dirs.append(session_path)

    for i in range(48, 58):
        session_path = os.path.join(dataset_base_path, f"session_{i}")
        session_dirs.append(session_path)

    for i in range(132, 142):
        session_path = os.path.join(dataset_base_path, f"session_{i}")
        session_dirs.append(session_path)
    
    # Alternative: Auto-detect all session folders
    # session_dirs = glob.glob(os.path.join(dataset_base_path, "session_*"))
    # session_dirs = [d for d in session_dirs if os.path.isdir(d)]
    
    print(f"Found {len(session_dirs)} session directories")
    
    if len(session_dirs) == 0:
        print("No session directories found! Please check the path:")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for: {dataset_base_path}")
        abs_path = os.path.abspath(dataset_base_path)
        print(f"Absolute path: {abs_path}")
        print(f"Exists: {os.path.exists(abs_path)}")
        return
    
    # Create training chunks
    print("Creating training chunks...")
    df = create_training_chunks(
        session_dirs, 
        chunk_duration=8.0,    # 8-second chunks
        frame_rate=25.0,       # 25 fps
        overlap=0.5            # 50% overlap
    )
    
    # Save to CSV
    output_csv = "training_chunks.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Successfully created {output_csv}")
    print(f"📊 Total chunks created: {len(df)}")
    print(f"🎯 Chunk duration: 8.0 seconds")
    print(f"🔄 Overlap: 50%")
    print(f"📁 Sessions processed: {len(session_dirs)}")
    
    # Show some statistics
    if len(df) > 0:
        print(f"\n📈 Statistics:")
        print(f"   Average chunks per session: {len(df) / len(session_dirs):.1f}")
        print(f"   Total speakers across all chunks: {df['num_speakers'].sum()}")
        print(f"   Max speakers in a chunk: {df['num_speakers'].max()}")
        print(f"   Min speakers in a chunk: {df['num_speakers'].min()}")
        
        # Show first few chunks
        print(f"\n📋 First 3 chunks:")
        print(df[['session_id', 'start_time', 'end_time', 'num_speakers']].head(3).to_string(index=False))

if __name__ == "__main__":
    main()