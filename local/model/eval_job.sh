#!/bin/bash
#SBATCH --job-name=avsd_eval
#SBATCH --output=logs/avsd_eval_%j.out
#SBATCH --error=logs/avsd_eval_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --qos=a40

set -euo pipefail

REPO_ROOT="/home/speech-audio-research/22b3965"   # one level up from local/
MODEL_DIR="${REPO_ROOT}/AVSD/local/model"

# ── user-configurable paths ───────────────────────────────────────────────
CHECKPOINT="${REPO_ROOT}/AVSD/local/model/checkpoints/model_epoch_20.pth"
# Prefer correctly spelled path; fallback to legacy misspelled path.
EVAL_DIR="${REPO_ROOT}/evaluation-bin/modified-bin"
if [[ ! -d "${EVAL_DIR}" ]]; then
    LEGACY_EVAL_DIR="${REPO_ROOT}/evaulation-bin/modified-bin"
    if [[ -d "${LEGACY_EVAL_DIR}" ]]; then
        EVAL_DIR="${LEGACY_EVAL_DIR}"
    fi
fi
RUN_TAG="epoch_20_$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${REPO_ROOT}/evaluation-results/${RUN_TAG}"

# ── evaluation hyper-parameters ───────────────────────────────────────────
OUTPUT_SPEAKER=4        # must match the trained model (change if you retrained for more)
CHUNK_FRAMES=200        # 8 s @ 25 fps
STRIDE_FRAMES=200       # non-overlapping (set smaller for overlapping eval)
# Sweep thresholds (edit this list as needed)
THRESHOLDS=(0.15 0.20 0.25 0.30 0.35 0.40 0.50)
DEVICE="cuda"           # force GPU for faster evaluation
LOG_LEVEL="INFO"

# ── reproducibility ───────────────────────────────────────────────────────
export PYTHONHASHSEED=42
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ── job info ─────────────────────────────────────────────────────────────
echo "========================================================"
echo "AVSD Evaluation Job"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "Date        : $(date)"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Eval dir    : ${EVAL_DIR}"
echo "Output base : ${OUTPUT_BASE}"
echo "Thresholds  : ${THRESHOLDS[*]}"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "Mode        : ARRAY (task ${SLURM_ARRAY_TASK_ID})"
else
    echo "Mode        : SEQUENTIAL"
    echo "Tip         : run in parallel with: sbatch --array=0-$((${#THRESHOLDS[@]}-1)) $0"
fi
echo "========================================================"

mkdir -p logs "${OUTPUT_BASE}"

# ── early sanity checks (fail once, fast) ─────────────────────────────────
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

# ── run evaluation sweep ───────────────────────────────────────────────────
SUMMARY_TSV="${OUTPUT_BASE}/summary.tsv"
echo -e "threshold\tstatus\tglobal_DER\tglobal_JER\tmacro_F1\ttotal_MISS\ttotal_FA\ttotal_CONF\toutput_dir" > "${SUMMARY_TSV}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#THRESHOLDS[@]} )); then
        echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}; valid range is 0..$((${#THRESHOLDS[@]}-1))"
        exit 2
    fi
    THRESHOLDS_TO_RUN=("${THRESHOLDS[SLURM_ARRAY_TASK_ID]}")
else
    THRESHOLDS_TO_RUN=("${THRESHOLDS[@]}")
fi

FAILED=0
for THRESHOLD in "${THRESHOLDS_TO_RUN[@]}"; do
    THR_TAG="$(printf "%.2f" "${THRESHOLD}" | tr '.' 'p')"
    OUT_DIR="${OUTPUT_BASE}/thr_${THR_TAG}"
    RUN_LOG="${OUTPUT_BASE}/eval_thr_${THR_TAG}.log"

    echo ""
    echo "--------------------------------------------------------"
    echo "Starting evaluation for threshold=${THRESHOLD}"
    echo "Output dir : ${OUT_DIR}"
    echo "Live log   : ${RUN_LOG}"
    echo "--------------------------------------------------------"

    if stdbuf -oL -eL python "${MODEL_DIR}/evaluate.py" \
        --checkpoint    "${CHECKPOINT}" \
        --eval_dir      "${EVAL_DIR}" \
        --output_dir    "${OUT_DIR}" \
        --output_speaker "${OUTPUT_SPEAKER}" \
        --chunk_frames  "${CHUNK_FRAMES}" \
        --stride_frames "${STRIDE_FRAMES}" \
        --threshold     "${THRESHOLD}" \
        --device        "${DEVICE}" \
        --log_level     "${LOG_LEVEL}" \
        2>&1 | tee -a "${RUN_LOG}"; then

        METRICS_JSON="${OUT_DIR}/global_metrics.json"
        if [[ -f "${METRICS_JSON}" ]]; then
            read -r DER JER F1 MISS FA CONF < <(python - <<PY
import json
with open("${METRICS_JSON}", "r") as f:
    m = json.load(f)
print(
    m.get("global_DER", "nan"),
    m.get("global_JER", "nan"),
    m.get("macro_mean_f1", "nan"),
    m.get("total_MISS", "nan"),
    m.get("total_FA", "nan"),
    m.get("total_CONF", "nan"),
)
PY
)
            echo -e "${THRESHOLD}\tOK\t${DER}\t${JER}\t${F1}\t${MISS}\t${FA}\t${CONF}\t${OUT_DIR}" >> "${SUMMARY_TSV}"
            echo "Completed threshold=${THRESHOLD} | DER=${DER} JER=${JER} F1=${F1}"
        else
            FAILED=1
            echo -e "${THRESHOLD}\tFAILED_NO_METRICS\tNA\tNA\tNA\tNA\tNA\tNA\t${OUT_DIR}" >> "${SUMMARY_TSV}"
            echo "Threshold=${THRESHOLD} finished but global_metrics.json missing"
        fi
    else
        FAILED=1
        echo -e "${THRESHOLD}\tFAILED\tNA\tNA\tNA\tNA\tNA\tNA\t${OUT_DIR}" >> "${SUMMARY_TSV}"
        echo "Threshold=${THRESHOLD} failed (see ${RUN_LOG})"
    fi
done

# ── final summary ─────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "Threshold sweep summary"
echo "Summary file: ${SUMMARY_TSV}"
echo "========================================================"

column -t -s $'\t' "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"

BEST_LINE=$(python - <<PY
import csv
from math import inf
summary = "${SUMMARY_TSV}"
best = None
best_der = inf
with open(summary, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["status"] != "OK":
            continue
        try:
            der = float(row["global_DER"])
        except Exception:
            continue
        if der < best_der:
            best_der = der
            best = row
if best is None:
    print("NO_VALID_RESULT")
else:
    print(
        f"BEST threshold={best['threshold']} DER={best['global_DER']} "
        f"JER={best['global_JER']} F1={best['macro_F1']} output={best['output_dir']}"
    )
PY
)

echo "${BEST_LINE}"

if [[ ${FAILED} -eq 0 ]]; then
    EXIT_CODE=0
else
    EXIT_CODE=1
fi

echo "========================================================"
echo "Job finished with exit code: ${EXIT_CODE}"
echo "Results in : ${OUTPUT_BASE}"
echo "========================================================"

exit "${EXIT_CODE}"
