#!/usr/bin/env python3
import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

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


def run_speaker_inference(spk_dir: Path, model: nn.Module, device: torch.device, target_size: Tuple[int, int] = (96, 96)) -> np.ndarray:
    """Reads frames specifically for model inference, ensuring strict [B, T, C, H, W] tensor layout."""
    track_paths = get_speaker_tracks(spk_dir)
    gray_frames = []

    for video_path in track_paths:
        cap = cv2.VideoCapture(str(video_path))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Ensure frame is 2D grayscale [96, 96]
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            gray_resized = cv2.resize(gray, target_size)
            gray_frames.append(gray_resized)
        cap.release()

    if not gray_frames:
        return np.array([], dtype=np.float32)

    # Convert to array [T, 96, 96]
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

        chunk = video_array[start:end] # [T_chunk, 96, 96]
        T_chunk = len(chunk)
        
        # Convert to Tensor [T_chunk, 1, 96, 96] then add Batch dim -> [1, T_chunk, 1, 96, 96]
        tensor = torch.from_numpy(chunk).to(device)
        tensor = tensor.unsqueeze(1) # [T_chunk, C=1, H=96, W=96]
        tensor = tensor.unsqueeze(0) # [B=1, T_chunk, C=1, H=96, W=96]

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=device.type == "cuda"):
                try:
                    # Try [B, T, C, H, W] shape first
                    logits = model(tensor)
                except RuntimeError:
                    # Fallback to [B, C, T, H, W] if model expects standard 3D Conv shape
                    tensor_alt = tensor.transpose(1, 2) # [B=1, C=1, T_chunk, H=96, W=96]
                    logits = model(tensor_alt)

        logits_np = logits.squeeze(0).detach().cpu().numpy()
        logits_sum[start:end] += logits_np
        hits[start:end] += 1

    hits = np.maximum(hits, 1)
    avg_logits = logits_sum / hits
    probs = 1.0 / (1.0 + np.exp(-avg_logits))
    return probs


class SpeakerStreamReader:
    """Streams video frames sequentially from a set of track files with minimal RAM impact."""
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


def draw_speaker_card(color_frame: Optional[np.ndarray], prob: float, speaker_name: str, card_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    if color_frame is None:
        card = np.zeros((card_size[1], card_size[0], 3), dtype=np.uint8)
        cv2.putText(card, f"{speaker_name}: END", (10, card_size[1] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return card

    card = cv2.resize(color_frame, card_size)
    h, w, _ = card.shape

    bar_h = 16
    y_bar = h - bar_h - 10
    x_bar = 10
    bar_w = w - 20

    cv2.rectangle(card, (x_bar, y_bar), (x_bar + bar_w, y_bar + bar_h), (30, 30, 30), -1)

    fill_w = int(bar_w * prob)
    color = (0, 230, 0) if prob >= 0.5 else (0, 140, 255)
    cv2.rectangle(card, (x_bar, y_bar), (x_bar + fill_w, y_bar + bar_h), color, -1)
    cv2.rectangle(card, (x_bar, y_bar), (x_bar + bar_w, y_bar + bar_h), (200, 200, 200), 1)

    cv2.putText(card, f"{speaker_name}: {prob:.2f}", (x_bar, y_bar - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return card


def render_grid_video(spk_dirs: List[Path], probs_list: List[np.ndarray], output_path: str, fps: int = 25):
    max_frames = max((len(p) for p in probs_list), default=0)
    num_speakers = len(spk_dirs)
    
    cols = min(3, num_speakers)
    rows = math.ceil(num_speakers / cols)
    card_w, card_h = 224, 224

    out_w = cols * card_w
    out_h = rows * card_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    readers = [SpeakerStreamReader(d) for d in spk_dirs]

    print(f"Rendering grid video frame-by-frame: {max_frames} frames -> {output_path}")

    for t in range(max_frames):
        grid_rows = []
        for r in range(rows):
            row_cards = []
            for c in range(cols):
                spk_idx = r * cols + c
                if spk_idx < num_speakers:
                    color_frame = readers[spk_idx].read_frame()
                    p = probs_list[spk_idx][t] if t < len(probs_list[spk_idx]) else 0.0
                    card = draw_speaker_card(color_frame, p, spk_dirs[spk_idx].name, (card_w, card_h))
                else:
                    card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
                row_cards.append(card)
            grid_rows.append(np.hstack(row_cards))

        full_frame = np.vstack(grid_rows)
        out.write(full_frame)

    for r in readers:
        r.release()

    out.release()
    print("Video saved successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--session_dir", required=True)
    parser.add_argument("--output", default="session_visualized.mp4")
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vsd_model(args.checkpoint, device)

    session_path = Path(args.session_dir)
    spk_dirs = sorted([d for d in session_path.glob("spk_*") if d.is_dir()], key=spk_sort_key)

    probs_list = []
    for spk_dir in spk_dirs:
        print(f"Processing {spk_dir.name}...")
        probs = run_speaker_inference(spk_dir, model, device)
        probs_list.append(probs)

    render_grid_video(spk_dirs, probs_list, args.output, fps=args.fps)


if __name__ == "__main__":
    main()