#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manual review viewer for AVSD per-frame probability dumps.

Features:
- Opens selected speaker track videos (track_00.mp4, track_01.mp4, ...) sequentially.
- Uses track metadata json (track_00.json, ...) for global-frame alignment.
- Shows two synchronized windows:
  1) Speaker video window
  2) LED panel window
     - ANY speech LED: on if any speaker is active at current global frame
     - SPK speech LED: on if selected speaker is active at current global frame

Controls:
- q or ESC: quit
- space: pause/resume
- n: step one frame when paused
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual AVSD output viewer")
    parser.add_argument(
        "--dump_file",
        required=True,
        help="Path to dump created by eval.py (e.g., session_05_frame_probs.npz)",
    )
    parser.add_argument(
        "--raw_session_dir",
        required=True,
        help="Path to raw session directory with metadata.json and speakers/",
    )
    parser.add_argument(
        "--speaker_idx",
        type=int,
        default=0,
        help="Speaker index to review (0-based)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override threshold for LED logic. Default: threshold stored in dump",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Playback fps target",
    )
    parser.add_argument(
        "--window_scale",
        type=float,
        default=1.0,
        help="Scale factor for video window",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable GUI windows and render an annotated review video instead",
    )
    parser.add_argument(
        "--save_video",
        default="",
        help="Output path for annotated mp4. Recommended in --headless mode",
    )
    return parser.parse_args()


def load_dump(dump_file: Path) -> Dict:
    data = np.load(dump_file, allow_pickle=True)
    required = ["probs"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in dump file: {dump_file}")

    probs = data["probs"].astype(np.float32)
    threshold = float(data["threshold"]) if "threshold" in data else 0.5
    session_id = (
        str(data["session_id"][0]) if "session_id" in data and len(data["session_id"]) > 0 else "unknown"
    )
    speaker_ids = data["speaker_ids"].tolist() if "speaker_ids" in data else []

    return {
        "probs": probs,
        "threshold": threshold,
        "session_id": session_id,
        "speaker_ids": speaker_ids,
    }


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_track_json_info(track_json: Path) -> Tuple[Optional[int], Optional[int]]:
    """Return (frame_start, frame_end) from track metadata json if present."""
    try:
        meta = read_json(track_json)
    except Exception:
        return None, None

    frame_start = meta.get("frame_start")
    frame_end = meta.get("frame_end")
    if frame_start is None or frame_end is None:
        return None, None

    return int(frame_start), int(frame_end)


def discover_speaker_tracks(raw_session_dir: Path, speaker_idx: int) -> List[Dict]:
    spk_dir = raw_session_dir / "speakers" / f"spk_{speaker_idx}" / "central_crops"
    if not spk_dir.exists():
        raise FileNotFoundError(f"Speaker crops directory not found: {spk_dir}")

    mp4_files = sorted(
        [
            p for p in spk_dir.glob("track_*.mp4")
            if not p.name.endswith("_lip.av.mp4")
        ],
        key=lambda p: p.name,
    )

    if not mp4_files:
        raise FileNotFoundError(f"No track_XX.mp4 files found in {spk_dir}")

    tracks: List[Dict] = []
    fallback_start = 0

    for mp4 in mp4_files:
        base = mp4.stem  # track_00
        track_json = mp4.with_name(base + ".json")
        frame_start, frame_end = (None, None)
        if track_json.exists():
            frame_start, frame_end = get_track_json_info(track_json)

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {mp4}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if frame_start is None:
            frame_start = fallback_start
        if frame_end is None:
            frame_end = frame_start + max(frame_count - 1, 0)

        fallback_start = frame_end + 1

        tracks.append(
            {
                "mp4": mp4,
                "track_json": track_json if track_json.exists() else None,
                "frame_start": int(frame_start),
                "frame_end": int(frame_end),
                "frame_count": frame_count,
            }
        )

    return tracks


def draw_led_panel(
    width: int,
    height: int,
    any_on: bool,
    spk_on: bool,
    speaker_idx: int,
    global_frame: int,
    prob_spk: float,
    threshold: float,
    track_name: str,
    local_frame: int,
) -> np.ndarray:
    panel = np.full((height, width, 3), 20, dtype=np.uint8)

    def draw_led(y: int, label: str, on: bool):
        color = (0, 220, 0) if on else (60, 60, 60)
        cv2.circle(panel, (40, y), 18, color, -1)
        cv2.circle(panel, (40, y), 18, (200, 200, 200), 1)
        cv2.putText(panel, label, (75, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2)

    draw_led(50, "ANY speech", any_on)
    draw_led(100, f"SPK {speaker_idx} speech", spk_on)

    cv2.putText(panel, f"global_frame: {global_frame}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, f"track: {track_name}", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, f"local_frame: {local_frame}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, f"spk_prob: {prob_spk:.4f}", (20, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, f"threshold: {threshold:.3f}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(panel, "keys: q/ESC quit | space pause | n step", (20, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    return panel


def clamp_frame_index(global_frame: int, total_frames: int) -> int:
    if total_frames <= 0:
        return 0
    if global_frame < 0:
        return 0
    if global_frame >= total_frames:
        return total_frames - 1
    return global_frame


def render_annotated_frame(
    frame: np.ndarray,
    panel: np.ndarray,
    track_name: str,
    local_frame: int,
    global_frame: int,
    aligned_frame: int,
    speaker_idx: int,
    spk_prob: float,
    threshold: float,
) -> np.ndarray:
    """Overlay text on video frame and compose side-by-side with LED panel."""
    cv2.putText(
        frame,
        f"track={track_name} local={local_frame} global={global_frame} aligned={aligned_frame}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"spk={speaker_idx} prob={spk_prob:.4f} thr={threshold:.3f}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    h_v, w_v = frame.shape[:2]
    panel_resized = cv2.resize(panel, (panel.shape[1], h_v), interpolation=cv2.INTER_LINEAR)
    composed = np.concatenate([frame, panel_resized], axis=1)
    return composed


def main() -> None:
    args = parse_args()

    dump_file = Path(args.dump_file)
    raw_session_dir = Path(args.raw_session_dir)

    if not dump_file.exists():
        raise FileNotFoundError(f"Dump file not found: {dump_file}")
    if not raw_session_dir.exists():
        raise FileNotFoundError(f"Raw session directory not found: {raw_session_dir}")

    dump = load_dump(dump_file)
    probs = dump["probs"]  # [T, N]
    total_frames, n_speakers = probs.shape

    if args.speaker_idx < 0 or args.speaker_idx >= n_speakers:
        raise ValueError(f"speaker_idx must be in [0, {n_speakers - 1}], got {args.speaker_idx}")

    threshold = float(args.threshold) if args.threshold is not None else float(dump["threshold"])

    tracks = discover_speaker_tracks(raw_session_dir, args.speaker_idx)

    print("=" * 70)
    print("Manual Review Viewer")
    print(f"dump_file      : {dump_file}")
    print(f"session_id     : {dump['session_id']}")
    print(f"raw_session    : {raw_session_dir}")
    print(f"speaker_idx    : {args.speaker_idx}")
    print(f"model_frames   : {total_frames}")
    print(f"num_speakers   : {n_speakers}")
    print(f"threshold      : {threshold:.3f}")
    print(f"tracks_found   : {len(tracks)}")
    for idx, t in enumerate(tracks):
        print(
            f"  [{idx}] {t['mp4'].name} | frame_start={t['frame_start']} frame_end={t['frame_end']} "
            f"video_frames={t['frame_count']}"
        )
    print("=" * 70)

    video_win = "Speaker Video"
    led_win = "Speech LEDs"

    gui_enabled = not args.headless
    if gui_enabled:
        try:
            cv2.namedWindow(video_win, cv2.WINDOW_NORMAL)
            cv2.namedWindow(led_win, cv2.WINDOW_NORMAL)
        except cv2.error as e:
            print("[WARN] OpenCV GUI is not available on this node.")
            print(f"[WARN] Falling back to headless mode. Reason: {e}")
            gui_enabled = False

    delay_ms = max(1, int(round(1000.0 / max(args.fps, 1e-6))))

    # If no GUI, write annotated output so it can be downloaded and reviewed locally.
    save_video_path = Path(args.save_video) if args.save_video else None
    if not gui_enabled and save_video_path is None:
        auto_name = f"{dump['session_id']}_spk{args.speaker_idx}_review.mp4"
        save_video_path = dump_file.parent / auto_name
        print(f"[INFO] Headless mode enabled. Auto output video: {save_video_path}")

    writer = None
    writer_fps = float(args.fps)

    paused = False
    track_idx = 0
    cap = None
    local_frame = 0

    try:
        while track_idx < len(tracks):
            track = tracks[track_idx]

            if cap is None:
                cap = cv2.VideoCapture(str(track["mp4"]))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video: {track['mp4']}")
                local_frame = 0

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    cap = None
                    track_idx += 1
                    continue

                global_frame = track["frame_start"] + local_frame
                aligned_frame = clamp_frame_index(global_frame, total_frames)

                frame_probs = probs[aligned_frame]
                any_on = bool(np.any(frame_probs >= threshold))
                spk_prob = float(frame_probs[args.speaker_idx])
                spk_on = bool(spk_prob >= threshold)

                if args.window_scale != 1.0:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(
                        frame,
                        (int(w * args.window_scale), int(h * args.window_scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                panel = draw_led_panel(
                    width=520,
                    height=320,
                    any_on=any_on,
                    spk_on=spk_on,
                    speaker_idx=args.speaker_idx,
                    global_frame=global_frame,
                    prob_spk=spk_prob,
                    threshold=threshold,
                    track_name=track["mp4"].name,
                    local_frame=local_frame,
                )

                composed = render_annotated_frame(
                    frame=frame,
                    panel=panel,
                    track_name=track["mp4"].name,
                    local_frame=local_frame,
                    global_frame=global_frame,
                    aligned_frame=aligned_frame,
                    speaker_idx=args.speaker_idx,
                    spk_prob=spk_prob,
                    threshold=threshold,
                )

                if gui_enabled:
                    cv2.imshow(video_win, frame)
                    cv2.imshow(led_win, panel)

                if save_video_path is not None:
                    if writer is None:
                        save_video_path.parent.mkdir(parents=True, exist_ok=True)
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(save_video_path),
                            fourcc,
                            writer_fps,
                            (composed.shape[1], composed.shape[0]),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Failed to open video writer: {save_video_path}")
                    writer.write(composed)

                local_frame += 1

            key = (cv2.waitKey(delay_ms if not paused else 30) & 0xFF) if gui_enabled else 255
            if key in (27, ord("q")):
                break
            if key == ord(" ") and gui_enabled:
                paused = not paused
            if key == ord("n") and paused and gui_enabled:
                # Step one frame while paused
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    cap = None
                    track_idx += 1
                    continue

                global_frame = track["frame_start"] + local_frame
                aligned_frame = clamp_frame_index(global_frame, total_frames)
                frame_probs = probs[aligned_frame]
                any_on = bool(np.any(frame_probs >= threshold))
                spk_prob = float(frame_probs[args.speaker_idx])
                spk_on = bool(spk_prob >= threshold)

                if args.window_scale != 1.0:
                    h, w = frame.shape[:2]
                    frame = cv2.resize(
                        frame,
                        (int(w * args.window_scale), int(h * args.window_scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )

                panel = draw_led_panel(
                    width=520,
                    height=320,
                    any_on=any_on,
                    spk_on=spk_on,
                    speaker_idx=args.speaker_idx,
                    global_frame=global_frame,
                    prob_spk=spk_prob,
                    threshold=threshold,
                    track_name=track["mp4"].name,
                    local_frame=local_frame,
                )

                composed = render_annotated_frame(
                    frame=frame,
                    panel=panel,
                    track_name=track["mp4"].name,
                    local_frame=local_frame,
                    global_frame=global_frame,
                    aligned_frame=aligned_frame,
                    speaker_idx=args.speaker_idx,
                    spk_prob=spk_prob,
                    threshold=threshold,
                )

                if gui_enabled:
                    cv2.imshow(video_win, frame)
                    cv2.imshow(led_win, panel)
                if save_video_path is not None and writer is not None:
                    writer.write(composed)

                local_frame += 1

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
            print(f"[INFO] Saved annotated review video: {save_video_path}")
        if gui_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
