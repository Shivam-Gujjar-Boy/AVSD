#!/bin/bash
#SBATCH -J vsd5-eval
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --output=/home/speech-audio-research/22b3965/job_files/vsd5-eval-%j.out
#SBATCH --error=/home/speech-audio-research/22b3965/job_files/vsd5-eval-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 <checkpoint_path> [eval_dir]"
  exit 2
fi

CHECKPOINT="$1"
EVAL_DIR="${2:-/home/speech-audio-research/22b3965/evaulation-bin/modified-bin}"

REPO_ROOT="/home/speech-audio-research/22b3965"
VSD_DIR="${REPO_ROOT}/AVSD/local/model/vsd"

if [[ ! -d "${EVAL_DIR}" ]]; then
  ALT_EVAL_DIR="${REPO_ROOT}/evaluation-bin/modified-bin"
  if [[ -d "${ALT_EVAL_DIR}" ]]; then
    EVAL_DIR="${ALT_EVAL_DIR}"
  fi
fi

RUN_TAG="vsd_epoch_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_ROOT}/evaluation-results-vsd/${RUN_TAG}"
SUMMARY_TSV="${OUTPUT_DIR}/summary.tsv"

THRESHOLDS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07)
CHUNK_FRAMES=200
STRIDE_FRAMES=200
MAX_SESSION_SPEAKERS=5
DEVICE="cuda"
LOG_LEVEL="INFO"

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "VSD Evaluation Job"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "Date        : $(date)"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Eval dir    : ${EVAL_DIR}"
echo "Output dir  : ${OUTPUT_DIR}"
echo "Thresholds  : ${THRESHOLDS[*]}"
echo "Max speakers: ${MAX_SESSION_SPEAKERS}"
echo "========================================================"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}"
  exit 2
fi

if [[ ! -d "${EVAL_DIR}" ]]; then
  echo "ERROR: eval directory not found: ${EVAL_DIR}"
  exit 2
fi

if [[ ! -f "${VSD_DIR}/evaluate_vsd_spk5.py" ]]; then
  echo "ERROR: evaluator script missing: ${VSD_DIR}/evaluate_vsd_spk5.py"
  exit 2
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate avsd

python -u "${VSD_DIR}/evaluate_vsd_spk5.py" \
  --checkpoint "${CHECKPOINT}" \
  --eval_dir "${EVAL_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --chunk_frames "${CHUNK_FRAMES}" \
  --stride_frames "${STRIDE_FRAMES}" \
  --thresholds "${THRESHOLDS[@]}" \
  --max_session_speakers "${MAX_SESSION_SPEAKERS}" \
  --device "${DEVICE}" \
  --log_level "${LOG_LEVEL}"

SWEEP_JSON="${OUTPUT_DIR}/threshold_sweep.json"
if [[ ! -f "${SWEEP_JSON}" ]]; then
  echo "ERROR: threshold_sweep.json not found: ${SWEEP_JSON}"
  exit 2
fi

python - <<PY
import csv
import json
from pathlib import Path

sweep_path = Path("${SWEEP_JSON}")
summary_path = Path("${SUMMARY_TSV}")

with sweep_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for threshold_key, metrics in data.get("metrics_by_threshold", {}).items():
    rows.append((
        float(metrics.get("threshold", threshold_key)),
        metrics,
    ))

rows.sort(key=lambda x: x[0])

with summary_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow([
        "threshold",
        "global_DER",
        "global_JER",
        "macro_F1",
        "macro_BCE",
        "total_MISS",
        "total_FA",
        "total_CONF",
        "n_sessions",
        "output_dir",
    ])
    for _, m in rows:
        w.writerow([
            m.get("threshold", "nan"),
            m.get("global_DER", "nan"),
            m.get("global_JER", "nan"),
            m.get("macro_mean_f1", "nan"),
            m.get("macro_bce_loss", "nan"),
            m.get("total_MISS", "nan"),
            m.get("total_FA", "nan"),
            m.get("total_CONF", "nan"),
            m.get("n_sessions", "nan"),
            "${OUTPUT_DIR}",
        ])

best_threshold = data.get("best_threshold")
best_metrics = data.get("best_metrics", {})
print("BEST threshold=", best_threshold, "DER=", best_metrics.get("global_DER", "nan"), "JER=", best_metrics.get("global_JER", "nan"), "F1=", best_metrics.get("macro_mean_f1", "nan"))
PY

echo ""
echo "========================================================"
echo "Threshold sweep summary"
echo "Summary file: ${SUMMARY_TSV}"
echo "========================================================"
column -t -s $'\t' "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"

echo "========================================================"
echo "Done"
echo "Results: ${OUTPUT_DIR}"
echo "========================================================"
