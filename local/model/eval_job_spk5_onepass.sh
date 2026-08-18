#!/bin/bash
#SBATCH --job-name=avsd_eval_spk5
#SBATCH --output=logs/avsd_eval_spk5_%j.out
#SBATCH --error=logs/avsd_eval_spk5_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=dgx
#SBATCH --qos=dgx

set -euo pipefail

REPO_ROOT="/home/speech-audio-research/22b3965"
MODEL_DIR="${REPO_ROOT}/AVSD/local/model"

# ── user-configurable paths ───────────────────────────────────────────────
CHECKPOINT="${REPO_ROOT}/AVSD/local/model/checkpoints_for_5/model_epoch_100.pth"

# Prefer correctly spelled path; fallback to legacy misspelled path.
EVAL_DIR="${REPO_ROOT}/evaluation-bin/modified-bin"
if [[ ! -d "${EVAL_DIR}" ]]; then
    LEGACY_EVAL_DIR="${REPO_ROOT}/evaulation-bin/modified-bin"
    if [[ -d "${LEGACY_EVAL_DIR}" ]]; then
        EVAL_DIR="${LEGACY_EVAL_DIR}"
    fi
fi

RUN_TAG="epoch_100_$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${REPO_ROOT}/evaluation-results-spk5/${RUN_TAG}"

# ── evaluation hyper-parameters ───────────────────────────────────────────
OUTPUT_SPEAKER=5
MAX_TRUE_SPEAKERS=5
CHUNK_FRAMES=200
STRIDE_FRAMES=200
THRESHOLDS=(0.01 0.02 0.03 0.04 0.05 0.06 0.07)
DEVICE="cuda"
LOG_LEVEL="INFO"

# ── reproducibility ───────────────────────────────────────────────────────
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "========================================================"
echo "AVSD Evaluation Job"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "Date        : $(date)"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Eval dir    : ${EVAL_DIR}"
echo "Output base : ${OUTPUT_BASE}"
echo "Thresholds  : ${THRESHOLDS[*]}"
echo "Max speakers: ${MAX_TRUE_SPEAKERS}"
echo "Mode        : ONE PASS"
echo "========================================================"

mkdir -p logs "${OUTPUT_BASE}"

if [[ ! -f "${MODEL_DIR}/evaluate.py" ]]; then
    echo "ERROR: evaluate.py not found at ${MODEL_DIR}/evaluate.py"
    exit 2
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    exit 2
fi

if [[ ! -d "${EVAL_DIR}" ]]; then
    echo "ERROR: eval directory not found: ${EVAL_DIR}"
    echo "       Tried both:"
    echo "       - ${REPO_ROOT}/evaluation-bin/modified-bin"
    echo "       - ${REPO_ROOT}/evaulation-bin/modified-bin"
    exit 2
fi

RUN_LOG="${OUTPUT_BASE}/eval.log"
SUMMARY_TSV="${OUTPUT_BASE}/summary.tsv"

echo "Running a single inference pass and caching metrics for all thresholds..."

stdbuf -oL -eL python "${MODEL_DIR}/evaluate.py" \
    --checkpoint "${CHECKPOINT}" \
    --eval_dir "${EVAL_DIR}" \
    --output_dir "${OUTPUT_BASE}" \
    --output_speaker "${OUTPUT_SPEAKER}" \
    --chunk_frames "${CHUNK_FRAMES}" \
    --stride_frames "${STRIDE_FRAMES}" \
    --thresholds "${THRESHOLDS[@]}" \
    --max_true_speakers "${MAX_TRUE_SPEAKERS}" \
    --device "${DEVICE}" \
    --log_level "${LOG_LEVEL}" \
    2>&1 | tee -a "${RUN_LOG}"

THRESHOLD_SWEEP_JSON="${OUTPUT_BASE}/threshold_sweep.json"
if [[ ! -f "${THRESHOLD_SWEEP_JSON}" ]]; then
    echo "ERROR: threshold_sweep.json not found: ${THRESHOLD_SWEEP_JSON}"
    exit 2
fi

python - <<PY
import csv
import json
from pathlib import Path

sweep_path = Path("${THRESHOLD_SWEEP_JSON}")
summary_path = Path("${SUMMARY_TSV}")

with sweep_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

metrics_by_threshold = data.get("metrics_by_threshold", {})
rows = []
for threshold_key, metrics in metrics_by_threshold.items():
    rows.append((
        float(metrics.get("threshold", threshold_key)),
        threshold_key,
        metrics,
    ))

rows.sort(key=lambda item: item[0])

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([
        "threshold",
        "global_DER",
        "global_JER",
        "macro_F1",
        "macro_BCE",
        "total_MISS",
        "total_FA",
        "total_CONF",
        "output_dir",
    ])
    for _, threshold_key, metrics in rows:
        writer.writerow([
            metrics.get("threshold", threshold_key),
            metrics.get("global_DER", "nan"),
            metrics.get("global_JER", "nan"),
            metrics.get("macro_mean_f1", "nan"),
            metrics.get("macro_bce_loss", "nan"),
            metrics.get("total_MISS", "nan"),
            metrics.get("total_FA", "nan"),
            metrics.get("total_CONF", "nan"),
            "${OUTPUT_BASE}",
        ])

best_threshold = data.get("best_threshold")
best_metrics = data.get("best_metrics", {})

print("\n========================================================")
print("Threshold sweep summary")
print(f"Summary file: {summary_path}")
print("========================================================")
print(summary_path.read_text())

if best_threshold is not None:
    print(
        f"BEST threshold={best_threshold} DER={best_metrics.get('global_DER', 'nan')} "
        f"JER={best_metrics.get('global_JER', 'nan')} "
        f"F1={best_metrics.get('macro_mean_f1', 'nan')}"
    )
PY

echo ""
echo "========================================================"
echo "Threshold sweep table"
echo "========================================================"
column -t -s $'\t' "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"

echo "========================================================"
echo "Job finished successfully"
echo "Results in : ${OUTPUT_BASE}"
echo "========================================================"