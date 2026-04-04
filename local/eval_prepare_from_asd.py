import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
import librosa
from tqdm import tqdm


def load_asd_scores(asd_path: Path) -> Dict[int, float]:
    """Load ASD JSON as a dict[int, float] of frame_index -> score."""
    with open(asd_path, "r") as f:
        data = json.load(f)
    # Keys are stringified frame indices
    return {int(k): float(v) for k, v in data.items()}


def load_track_metadata(track_meta_path: Path) -> Dict[str, int]:
    """Load track metadata JSON (frame_start, frame_end, etc.)."""
    with open(track_meta_path, "r") as f:
        meta = json.load(f)
    return meta


def read_lip_video_frames(video_path: Path, target_size=(96, 96)) -> np.ndarray:
    """
    Read all frames from a lip crop video and convert to grayscale 96x96.

    Returns:
        np.ndarray with shape [num_frames, H, W]
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape[:2] != target_size:
            gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_LINEAR)
        frames.append(gray.astype(np.float32))

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from video: {video_path}")

    return np.stack(frames, axis=0)  # [T, H, W]


def ensure_audio_features(
    session_dir: Path,
    out_session_dir: Path,
    total_frames: int,
    global_min_frame: int,
    ref_track_meta: Dict[str, float],
    fps_default: int = 25,
    sample_rate: int = 16000,
    n_mels: int = 40,
) -> None:
    """
    Create audio_features.npy for a session in the output directory
    if it does not already exist, aligned to the global frame timeline.

    We use a reference track's (frame_start, frame_end, start_time, end_time)
    to estimate the frame rate and the absolute time of `global_min_frame`,
    then crop the audio so frame 0 in features corresponds to global_min_frame.
    """
    out_session_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_session_dir / "audio_features.npy"
    if out_path.exists():
        return

    # Locate an audio source video.
    # Prefer the central video from metadata if present and existing.
    metadata_path = session_dir / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    central_video_path = None
    any_spk_key = next(iter(metadata.keys()))
    central_info = metadata.get(any_spk_key, {}).get("central", {})
    central_video_rel = central_info.get("video")
    if central_video_rel:
        candidate = session_dir / central_video_rel
        if candidate.exists():
            central_video_path = candidate

    # Fallback: search for any lip/track video under speakers/*/central_crops
    if central_video_path is None:
        speakers_dir = session_dir / "speakers"
        if not speakers_dir.exists():
            raise FileNotFoundError(f"No 'speakers' directory found in {session_dir}")

        lip_candidates = []
        track_candidates = []
        for spk_dir in sorted(speakers_dir.iterdir()):
            cc_dir = spk_dir / "central_crops"
            if not cc_dir.exists():
                continue
            for mp4 in sorted(cc_dir.glob("*.mp4")):
                name = mp4.name
                if name.endswith("_lip.av.mp4"):
                    lip_candidates.append(mp4)
                elif name.startswith("track_") and name.endswith(".mp4"):
                    track_candidates.append(mp4)

        if lip_candidates:
            central_video_path = lip_candidates[0]
        elif track_candidates:
            central_video_path = track_candidates[0]

    if central_video_path is None or not central_video_path.exists():
        raise FileNotFoundError(
            f"Could not locate any suitable audio source video in {session_dir}"
        )

    # Extract temporary mono WAV using ffmpeg
    tmp_wav = out_session_dir / "central_audio_tmp.wav"
    if not tmp_wav.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(central_video_path),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            str(tmp_wav),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Compute approximate fps and time of global_min_frame using reference track
    frame_start = int(ref_track_meta["frame_start"])
    frame_end = int(ref_track_meta["frame_end"])
    start_time = float(ref_track_meta["start_time"])
    end_time = float(ref_track_meta["end_time"])

    span_frames = max(1, frame_end - frame_start)
    time_span = max(1e-6, end_time - start_time)
    time_per_frame = time_span / span_frames
    fps = 1.0 / time_per_frame if time_per_frame > 0 else float(fps_default)

    # Time corresponding to global_min_frame
    t_global_min = start_time + (global_min_frame - frame_start) * time_per_frame
    if t_global_min < 0:
        t_global_min = 0.0

    # Compute log-mel features, cropping audio so t=0 aligns to global_min_frame
    y, sr = librosa.load(str(tmp_wav), sr=sample_rate)
    sample_start = int(t_global_min * sr)
    if sample_start >= len(y):
        # Edge case: if computed start is past audio, fall back to start of file
        sample_start = 0
    y = y[sample_start:]

    hop_length = int(sr / fps)
    if hop_length <= 0:
        hop_length = int(sr / fps_default)

    n_fft = 1024
    melspec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)  # [n_mels, T_audio]
    feats = log_melspec.T.astype(np.float32)  # [T_audio, n_mels]

    # Align to total_frames (video frame count)
    T_audio = feats.shape[0]
    if T_audio >= total_frames:
        feats = feats[:total_frames]
    else:
        pad = np.zeros((total_frames - T_audio, n_mels), dtype=np.float32)
        feats = np.concatenate([feats, pad], axis=0)

    np.save(out_path, feats)


def prepare_single_session(
    session_dir: Path,
    out_root: Path,
    vad_threshold: float = -1.0,
) -> None:
    """
    From a raw eval session with:
        metadata.json
        speakers/spk_X/central_crops/track_YY_*.{json,mp4}

    Create:
        audio_features.npy  (assumed to be prepared separately if needed)
        diarization_labels.npy   [T, num_speakers] with 0/1 labels
        nframes.txt              total number of frames T
        video_features/spk_i.npy [T, 96, 96] per visible speaker index

    The frame index T is derived from ASD JSON keys / track metadata
    and matches the global frame indices used by ASD.
    """
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {session_dir}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    speaker_ids = sorted([k for k in metadata.keys() if k.startswith("spk_")])
    if not speaker_ids:
        print(f"[WARN] No speakers found in metadata for {session_dir.name}, skipping.")
        return

    # Discover global frame range across all speakers and tracks
    global_min_frame = None
    global_max_frame = None

    track_infos: Dict[str, List[Dict]] = {}
    ref_track_meta_for_audio: Dict[str, float] = {}

    for spk_id in speaker_ids:
        spk_info = metadata[spk_id]["central"]
        crops = spk_info.get("crops", [])
        if not crops:
            continue

        track_infos[spk_id] = []
        for crop in crops:
            track_meta_path = session_dir / crop["crop_metadata"]
            meta = load_track_metadata(track_meta_path)
            frame_start = int(meta["frame_start"])
            frame_end = int(meta["frame_end"])

            if global_min_frame is None or frame_start < global_min_frame:
                global_min_frame = frame_start
            if global_max_frame is None or frame_end > global_max_frame:
                global_max_frame = frame_end

            track_infos[spk_id].append(
                {
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "asd_path": session_dir / crop["asd"],
                    "lip_path": session_dir / crop["lip"],
                    "meta": meta,
                }
            )

            # Use the very first track metadata we see as reference for audio alignment
            if not ref_track_meta_for_audio:
                # Expect meta to contain frame_start, frame_end, start_time, end_time
                ref_track_meta_for_audio = {
                    "frame_start": meta["frame_start"],
                    "frame_end": meta["frame_end"],
                    "start_time": meta["start_time"],
                    "end_time": meta["end_time"],
                }

    if global_min_frame is None or global_max_frame is None:
        print(f"[WARN] Could not determine frame range for {session_dir.name}, skipping.")
        return

    # We align everything to a 0-based global frame index for this session
    total_frames = global_max_frame - global_min_frame + 1
    num_speakers = len(speaker_ids)

    # Output session directory (separate from raw input)
    out_session_dir = out_root / session_dir.name
    out_session_dir.mkdir(parents=True, exist_ok=True)

    # Optionally create audio_features.npy aligned to total_frames and global_min_frame
    if ref_track_meta_for_audio:
        ensure_audio_features(
            session_dir=session_dir,
            out_session_dir=out_session_dir,
            total_frames=total_frames,
            global_min_frame=global_min_frame,
            ref_track_meta=ref_track_meta_for_audio,
        )

    # Prepare label matrix [T, num_speakers]
    labels = np.zeros((total_frames, num_speakers), dtype=np.float32)

    # Prepare video arrays per speaker [T, 96, 96]
    video_dir = out_session_dir / "video_features"
    video_dir.mkdir(exist_ok=True)
    speaker_videos = {
        spk_id: np.zeros((total_frames, 96, 96), dtype=np.float32)
        for spk_id in speaker_ids
    }

    # Fill labels and video features
    for spk_idx, spk_id in enumerate(speaker_ids):
        tracks = track_infos.get(spk_id, [])
        if not tracks:
            continue

        for track in tracks:
            frame_start = track["frame_start"]
            frame_end = track["frame_end"]
            asd_scores = load_asd_scores(track["asd_path"])

            # Read lip video frames for this track
            lip_frames = read_lip_video_frames(track["lip_path"])  # [L, 96, 96]
            L = lip_frames.shape[0]

            # Map ASD/global frame indices -> session index (0-based) and video frame idx
            for local_idx, global_frame in enumerate(range(frame_start, frame_end + 1)):
                session_idx = global_frame - global_min_frame
                if session_idx < 0 or session_idx >= total_frames:
                    continue
                if local_idx >= L:
                    break

                # Assign video frame
                speaker_videos[spk_id][session_idx] = lip_frames[local_idx]

                # Assign label from ASD score if available
                if global_frame in asd_scores:
                    score = asd_scores[global_frame]
                    if score >= vad_threshold:
                        labels[session_idx, spk_idx] = 1.0

        # Save per-speaker video .npy
        out_video_path = video_dir / f"{spk_id}.npy"
        np.save(out_video_path, speaker_videos[spk_id])

    # Save labels and nframes
    np.save(out_session_dir / "diarization_labels.npy", labels)
    with open(out_session_dir / "nframes.txt", "w") as f:
        f.write(str(total_frames))

    print(
        f"[OK] {session_dir.name}: frames={total_frames}, speakers={num_speakers} "
        f"-> diarization_labels.npy, nframes.txt, video_features/"
    )


def prepare_eval_root(eval_root: Path, out_root: Path, vad_threshold: float = -1.0) -> None:
    """Run preparation for all session_* folders under eval_root into out_root."""
    sessions = sorted(
        [d for d in eval_root.iterdir() if d.is_dir() and d.name.startswith("session_")]
    )
    if not sessions:
        print(f"No session_* directories found under {eval_root}")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    for session_dir in tqdm(sessions, desc="Preparing eval sessions"):
        try:
            prepare_single_session(session_dir, out_root, vad_threshold=vad_threshold)
        except Exception as e:
            print(f"[ERROR] Failed to process {session_dir.name}: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Convert raw eval sessions with ASD JSON + lip crops into "
            "diarization_labels.npy, nframes.txt, and video_features/spk_*.npy "
            "compatible with the AVSD model."
        )
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        required=True,
        help="Path to eval root containing session_* directories "
             "(e.g., /home/.../Evaluation-Set/eval-bin/eval)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root where converted session_* directories will be created",
    )
    parser.add_argument(
        "--vad_threshold",
        type=float,
        default=-1.0,
        help="ASD logit threshold for speech activity (default: -1.0)",
    )

    args = parser.parse_args()
    prepare_eval_root(Path(args.eval_root), Path(args.out_root), vad_threshold=args.vad_threshold)

