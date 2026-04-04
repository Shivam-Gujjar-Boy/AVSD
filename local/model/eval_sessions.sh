#!/bin/bash
#SBATCH --job-name=avsd_eval_multi
#SBATCH --output=logs/avsd_eval_multi_%j.out
#SBATCH --error=logs/avsd_eval_multi_%j.err
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=a40
#SBATCH --qos=a40

set -euo pipefail

# Usage examples:
#   sbatch eval_sessions.sh 05 06 07
#   sbatch eval_sessions.sh session_05 session_06
# If no args provided, defaults to session_06.

source ~/miniconda3/etc/profile.d/conda.sh
conda activate avsd

REPO_ROOT="/home/speech-audio-research/22b3965"
MODEL_DIR="${REPO_ROOT}/AVSD/local/model"

CHECKPOINT="${REPO_ROOT}/AVSD/local/model/checkpoints/model_epoch_100.pth"
EVAL_ROOT="${REPO_ROOT}/evaulation-bin/modified-bin"
THRESHOLD="0.20"

RUN_TAG="batch_$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${REPO_ROOT}/evaluation-results/manual_review/${RUN_TAG}"
SUMMARY_TSV="${OUTPUT_BASE}/summary.tsv"

mkdir -p logs "${OUTPUT_BASE}"

echo "========================================================"
echo "Multi-session eval.py job"
echo "Job ID      : ${SLURM_JOB_ID:-local}"
echo "Node        : $(hostname)"
echo "Date        : $(date)"
echo "Checkpoint  : ${CHECKPOINT}"
echo "Eval root   : ${EVAL_ROOT}"
echo "Output base : ${OUTPUT_BASE}"
echo "========================================================"

if [[ $# -eq 0 ]]; then
  SESSIONS=("session_06")
else
  SESSIONS=()
  for s in "$@"; do
    if [[ "$s" =~ ^session_[0-9]+$ ]]; then
      SESSIONS+=("$s")
    elif [[ "$s" =~ ^[0-9]+$ ]]; then
      # Force base-10 so values like 08/09 are not treated as octal.
      s_dec=$((10#$s))
      SESSIONS+=("session_$(printf "%02d" "${s_dec}")")
    else
      echo "ERROR: invalid session token '$s'. Use 06 or session_06 format."
      exit 1
    fi
  done
fi

echo "Sessions     : ${SESSIONS[*]}"
echo "Threshold    : ${THRESHOLD}"
echo "========================================================"

echo -e "session\tstatus\tDER\tJER\tF1\tBCE\tMISS\tFA\tCONF\toutput_dir" > "${SUMMARY_TSV}"

cd "${MODEL_DIR}"

FAILED=0
for SESSION_ID in "${SESSIONS[@]}"; do
  SESSION_OUT="${OUTPUT_BASE}/${SESSION_ID}"
  SESSION_LOG="${SESSION_OUT}/run.log"

  mkdir -p "${SESSION_OUT}"

  echo ""
  echo "--------------------------------------------------------"
  echo "Running ${SESSION_ID}"
  echo "Output dir : ${SESSION_OUT}"
  echo "--------------------------------------------------------"

  if stdbuf -oL -eL python eval.py \
    --checkpoint "${CHECKPOINT}" \
    --eval_root "${EVAL_ROOT}" \
    --session_id "${SESSION_ID}" \
    --output_dir "${SESSION_OUT}" \
    --threshold "${THRESHOLD}" \
    --device "cuda" \
    --log_level "INFO" 2>&1 | tee "${SESSION_LOG}"; then

    METRICS_JSON="${SESSION_OUT}/${SESSION_ID}_summary.json"
    if [[ -f "${METRICS_JSON}" ]]; then
      read -r DER JER F1 BCE MISS FA CONF < <(python - <<PY
import json
with open("${METRICS_JSON}", "r", encoding="utf-8") as f:
    m = json.load(f)
print(
    m.get("DER", "nan"),
    m.get("JER", "nan"),
    m.get("mean_f1", "nan"),
    m.get("bce_loss", "nan"),
    m.get("MISS", "nan"),
    m.get("FA", "nan"),
    m.get("CONF", "nan"),
)
PY
)
      echo -e "${SESSION_ID}\tOK\t${DER}\t${JER}\t${F1}\t${BCE}\t${MISS}\t${FA}\t${CONF}\t${SESSION_OUT}" >> "${SUMMARY_TSV}"
      echo "Completed ${SESSION_ID} | DER=${DER} JER=${JER} F1=${F1}"
    else
      FAILED=1
      echo -e "${SESSION_ID}\tFAILED_NO_SUMMARY\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t${SESSION_OUT}" >> "${SUMMARY_TSV}"
      echo "${SESSION_ID} finished but summary json not found"
    fi
  else
    FAILED=1
    echo -e "${SESSION_ID}\tFAILED\tNA\tNA\tNA\tNA\tNA\tNA\tNA\t${SESSION_OUT}" >> "${SUMMARY_TSV}"
    echo "${SESSION_ID} failed. Check log: ${SESSION_LOG}"
  fi
done

echo ""
echo "========================================================"
echo "Batch complete"
echo "Summary file: ${SUMMARY_TSV}"
echo "========================================================"
column -t -s $'\t' "${SUMMARY_TSV}" || cat "${SUMMARY_TSV}"

if [[ ${FAILED} -eq 0 ]]; then
  EXIT_CODE=0
else
  EXIT_CODE=1
fi

echo "========================================================"
echo "Job finished with exit code: ${EXIT_CODE}"
echo "Results base: ${OUTPUT_BASE}"
echo "========================================================"

exit "${EXIT_CODE}"
