#!/usr/bin/env python3
"""
Visualize VSD results by creating separate videos for each speaker.
Each video shows:
  - Speaker video frames
  - Speaker activity bar (green if speaking, orange if silent)
  - Combined audio track (from session)
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from vsd_net import VSDNet


def load_vsd_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    model = VSDNet()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def spk_sort_key(p: Path) -> int:
    match = re.search(r"spk_(\d+)", p.name)
    return int(match.group(1)) if match else 10**9


def get_speaker_tracks(spk_dir: Path) -> List[Path]:
    central_crops_dir = spk_dir / "central_crops"
    mp4_files = sorted(central_crops_dir.glob("track_*.mp4"))
    return [p for p in mp4_files if not p.name.endswith("_lip.av.mp4")]


def extract_audio_from_session(session_dir: Path) -> Optional[str]:
    """
    Extract combined audio from session directory.
    Looks for audio files in session_dir. Returns path to audio file.
    """
    # Try to find a standalone audio file in the session directory.
    for ext in ["wav", "mp3", "flac", "m4a"]:
        audio_files = list(session_dir.glob(f"*.{ext}"))
        if audio_files:
            return str(audio_files[0])

    # Every speaker track contains the same combined audio, so use the first one.
    speakers_dir = session_dir / "speakers"
    mp4_files = sorted(speakers_dir.glob("spk_*/central_crops/track_*.mp4"))
    mp4_files = [path for path in mp4_files if not path.name.endswith("_lip.av.mp4")]
    if mp4_files:
        video_path = str(mp4_files[0])
        audio_path = str(session_dir / "extracted_audio.wav")
        try:
            cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-y", audio_path]
            subprocess.run(cmd, check=True, capture_output=True)
            return audio_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    return None


def run_speaker_inference(spk_dir: Path, model: nn.Module, device: torch.device, target_size: Tuple[int, int] = (96, 96)) -> np.ndarray:
    """Infer speaker activity probabilities from video frames."""
    track_paths = get_speaker_tracks(spk_dir)
    gray_frames = []

    for video_path in track_paths:
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            gray_resized = cv2.resize(gray, target_size)
            gray_frames.append(gray_resized)
        cap.release()

    if not gray_frames:
        return np.array([], dtype=np.float32)

    video_array = np.array(gray_frames, dtype=np.float32)
    if video_array.ndim == 4:
        video_array = video_array.squeeze(-1)

    if video_array.max() > 1.0:
        video_array /= 255.0

    T = len(video_array)
    chunk_size = 200
    logits_sum = np.zeros(T, dtype=np.float32)
    hits = np.zeros(T, dtype=np.int32)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        if end <= start:
            continue

        chunk = video_array[start:end]
        T_chunk = len(chunk)
        
        tensor = torch.from_numpy(chunk).to(device)
        tensor = tensor.unsqueeze(1)
        tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                try:
                    logits = model(tensor)
                except RuntimeError:
                    tensor_alt = tensor.transpose(1, 2)
                    logits = model(tensor_alt)

        logits_np = logits.squeeze(0).detach().cpu().numpy()
        logits_sum[start:end] += logits_np
        hits[start:end] += 1

    hits = np.maximum(hits, 1)
    avg_logits = logits_sum / hits
    probs = 1.0 / (1.0 + np.exp(-avg_logits))
    return probs


class SpeakerStreamReader:
    """Streams video frames from speaker track files."""
    def __init__(self, spk_dir: Path):
        self.track_paths = get_speaker_tracks(spk_dir)
        self.current_track_idx = 0
        self.cap: Optional[cv2.VideoCapture] = None
        self._open_next_track()

    def _open_next_track(self):
        if self.cap is not None:
            self.cap.release()
        if self.current_track_idx < len(self.track_paths):
            self.cap = cv2.VideoCapture(str(self.track_paths[self.current_track_idx]))
            self.current_track_idx += 1
        else:
            self.cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None

        ret, frame = self.cap.read()
        if not ret:
            self._open_next_track()
            if self.cap is None:
                return None
            ret, frame = self.cap.read()
            if not ret:
                return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()


def draw_activity_bar(frame: np.ndarray, prob: float, height: int = 40) -> np.ndarray:
    """
    Draw activity bar at bottom of frame.
    Green if speaking (prob >= 0.5), orange if silent.
    """
    h, w, _ = frame.shape
    
    # Create bar region
    bar_h = height
    y_bar = h - bar_h
    x_bar = 0
    bar_w = w

    # Draw bar background
    cv2.rectangle(frame, (x_bar, y_bar), (x_bar + bar_w, h), (30, 30, 30), -1)

    # Draw fill based on probability
    fill_w = int(bar_w * prob)
    color = (0, 230, 0) if prob >= 0.5 else (0, 140, 255)  # Green if speaking, orange if silent
    cv2.rectangle(frame, (x_bar, y_bar), (x_bar + fill_w, h), color, -1)
    cv2.rectangle(frame, (x_bar, y_bar), (x_bar + bar_w, h), (200, 200, 200), 1)

    # Draw probability text
    text = f"Activity: {prob:.2f}"
    cv2.putText(frame, text, (10, y_bar + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def render_speaker_video(spk_dir: Path, probs: np.ndarray, output_video_path: str, fps: int = 25):
    """
    Render video for a single speaker with activity bar.
    Output is saved without audio (audio will be added separately).
    """
    reader = SpeakerStreamReader(spk_dir)
    
    # Get first frame to determine output size
    first_frame = reader.read_frame()
    if first_frame is None:
        print(f"Warning: Could not read frames from {spk_dir}")
        return
    
    frame_h, frame_w, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        reader.release()
        raise RuntimeError(f"Could not open video writer: {output_video_path}")
    
    print(f"Rendering video for {spk_dir.name}: {len(probs)} frames -> {output_video_path}")
    
    for t in range(len(probs)):
        # The first frame was already read to determine the output dimensions.
        frame = first_frame if t == 0 else reader.read_frame()
        if frame is None:
            # Pad with black frame if video ends before probs
            frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        
        # Ensure RGB
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Draw activity bar
        prob = probs[t]
        frame = draw_activity_bar(frame.copy(), prob, height=40)
        
        # Write frame
        writer.write(frame)
    
    reader.release()
    writer.release()
    print(f"  ✓ Video saved (without audio): {output_video_path}")


def add_audio_to_video(video_path: str, audio_path: str, output_path: str) -> bool:
    """
    Add audio to video using ffmpeg.
    Returns True if successful, False otherwise.
    """
    if audio_path is None or not os.path.exists(audio_path):
        print(f"  ⚠ Audio file not found: {audio_path}, skipping audio addition")
        return False
    
    try:
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        print(f"  ✓ Audio added: {output_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  ✗ Error adding audio: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Create separate videos for each speaker with activity bars and combined audio."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to VSD checkpoint")
    parser.add_argument("--session_dir", required=True, help="Session directory with speaker folders")
    parser.add_argument("--output_dir", default="speaker_videos", help="Output directory for videos")
    parser.add_argument("--fps", type=int, default=25, help="Video FPS")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load model
    print(f"📦 Loading model from {args.checkpoint}")
    model = load_vsd_model(args.checkpoint, device)

    # Setup output directory
    session_path = Path(args.session_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Speaker directories live below session_dir/speakers.
    speakers_dir = session_path / "speakers"
    if not speakers_dir.is_dir():
        raise FileNotFoundError(f"Speaker directory not found: {speakers_dir}")

    spk_dirs = sorted(
        [d for d in speakers_dir.glob("spk_*") if d.is_dir()],
        key=spk_sort_key
    )
    if not spk_dirs:
        raise FileNotFoundError(f"No speaker directories found in {speakers_dir}")
    print(f"📋 Found {len(spk_dirs)} speakers")

    # Extract audio
    print("🔊 Extracting audio from session...")
    audio_path = extract_audio_from_session(session_path)
    if audio_path is None:
        print("  ⚠ Warning: Could not find audio file")
    else:
        print(f"  ✓ Audio found: {audio_path}")

    # Process each speaker
    for spk_dir in spk_dirs:
        print(f"\n🎬 Processing {spk_dir.name}...")
        
        # Run inference
        print(f"  Running inference...")
        probs = run_speaker_inference(spk_dir, model, device)
        
        if len(probs) == 0:
            print(f"  ⚠ Skipping {spk_dir.name}: no frames")
            continue
        
        # Check activity
        max_prob = probs.max()
        mean_prob = probs.mean()
        print(f"  Activity stats: max={max_prob:.3f}, mean={mean_prob:.3f}")
        
        # Render video
        temp_video = str(output_dir / f"{spk_dir.name}_temp.mp4")
        final_video = str(output_dir / f"{spk_dir.name}.mp4")
        
        render_speaker_video(spk_dir, probs, temp_video, fps=args.fps)
        
        # Add audio
        if audio_path and os.path.exists(audio_path):
            success = add_audio_to_video(temp_video, audio_path, final_video)
            if success:
                os.remove(temp_video)  # Clean up temp video
            else:
                os.replace(temp_video, final_video)
                print(f"  Audio muxing failed; saved video without audio: {final_video}")
        else:
            os.replace(temp_video, final_video)
            print(f"  No audio available; saved video without audio: {final_video}")

    print("\n✅ All videos created successfully!")
    print(f"📂 Output directory: {output_dir}")


if __name__ == "__main__":
    main()
