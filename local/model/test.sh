#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./test.sh <speaker_idx> [threshold] [session_id]
# Example:
#   ./test.sh 0
#   ./test.sh 3 0.20 session_05

SPK_IDX="${1:-0}"
THRESHOLD="${2:-0.20}"
SESSION_ID="${3:-session_05}"

# Local paths (edit once if needed)
RAW_ROOT="/home/gujjar/Documents/Evaluation-Set/eval-bin/eval"
DUMP_ROOT="/home/gujjar/Downloads/DDP/npz"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_SESSION_DIR="${RAW_ROOT}/${SESSION_ID}"
DUMP_FILE="${DUMP_ROOT}/${SESSION_ID}_frame_probs.npz"

if [[ ! -d "${RAW_SESSION_DIR}" ]]; then
  echo "ERROR: raw session directory not found: ${RAW_SESSION_DIR}"
  exit 1
fi

if [[ ! -f "${DUMP_FILE}" ]]; then
  echo "ERROR: dump file not found: ${DUMP_FILE}"
  exit 1
fi

TRACK_FILE="$(find "${RAW_SESSION_DIR}/speakers/spk_${SPK_IDX}/central_crops" -maxdepth 1 -type f -name 'track_*.mp4' ! -name '*_lip.av.mp4' | sort | head -n1)"

if [[ -z "${TRACK_FILE}" ]]; then
  echo "ERROR: no track_*.mp4 found for speaker ${SPK_IDX}"
  exit 1
fi

ANN_VIDEO="${DUMP_ROOT}/${SESSION_ID}_spk${SPK_IDX}_annot.mp4"
OUT_VIDEO="${DUMP_ROOT}/${SESSION_ID}_spk${SPK_IDX}_annot_av.mp4"

echo "Session      : ${SESSION_ID}"
echo "Speaker      : ${SPK_IDX}"
echo "Threshold    : ${THRESHOLD}"
echo "Dump         : ${DUMP_FILE}"
echo "Track audio  : ${TRACK_FILE}"
echo "Render video : ${ANN_VIDEO}"
echo "Final A/V    : ${OUT_VIDEO}"

python3 "${SCRIPT_DIR}/review_viewer.py" \
  --dump_file "${DUMP_FILE}" \
  --raw_session_dir "${RAW_SESSION_DIR}" \
  --speaker_idx "${SPK_IDX}" \
  --threshold "${THRESHOLD}" \
  --headless \
  --save_video "${ANN_VIDEO}"

ffmpeg -y \
  -i "${ANN_VIDEO}" \
  -i "${TRACK_FILE}" \
  -map 0:v:0 \
  -map 1:a:0 \
  -c:v copy \
  -c:a aac \
  -shortest \
  "${OUT_VIDEO}"

ffplay "${OUT_VIDEO}"
