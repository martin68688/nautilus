#!/bin/bash
# =====================================================================
# beta3fullmlebench — MLE-bench Lite sanity driver (5 diverse tasks)
# ---------------------------------------------------------------------
# Goal: verify the multi-task harness end-to-end on 5 different task
# types BEFORE scaling to the full Lite-22. Vanilla baseline config:
#   - coldstart model templates ON (models_guidance_classified.json)
#   - experience_kb / methodology OFF  (config.yaml: methodology_kb_path "")
#   - grading server OFF              (config.yaml: use_grading_server False)
#     -> mlebench package NOT needed in the pod; the run skips format
#        validation cleanly (quality_check.py marks is_valid=True).
#
# The per-task metric is auto-parsed from the agent's stdout by
# result_parse_agent and recorded in the journal (good enough for
# "standard run records"). No separate scoring script needed for sanity.
#
# Usage:
#   DATASET_DIR=/workspace/data GPU=0 bash run_lite_sanity.sh
# Env knobs:
#   DATASET_DIR  mle-bench data root (default ./data)  — <slug>/prepared/{public,private}
#   GPU          CUDA_VISIBLE_DEVICES (default 0)
#   TIME_LIMIT_SECS  per-task wall budget (default 3600 = 1h, sanity)
#   STEPS        agent.steps override (default 8)
#   TASKS        optional space-separated subset to override the default 5
# =====================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Solver needs DEEPSEEK_* (in-pod mlevolve/.env). Source if present.
if [ -f .env ]; then set -a; . ./.env; set +a; fi

DATASET_DIR="${DATASET_DIR:-./data}"
GPU="${GPU:-0}"             # CUDA_VISIBLE_DEVICES mask
NUM_GPUS="${NUM_GPUS:-1}"   # MUST equal pod GPU count -> drives num_gpus + parallel_search_num
                            # (config defaults 3/6 would stack 6 workers on 1 GPU -> OOM)
CPUS="${CPUS:-8}"           # MUST equal pod CPU limit
TIME_LIMIT_SECS="${TIME_LIMIT_SECS:-3600}"
STEPS="${STEPS:-8}"

# Default sanity set: 5 different problem types (authoritative Lite-22 slugs).
# 1 NLP text-clf | 1 image-clf | 1 image-regression | 1 tabular-clf | 1 tabular-regression
DEFAULT_TASKS=(
  "spooky-author-identification"        # NLP text classification (logloss, lower)   — data already on PVC
  "aerial-cactus-identification"        # image classification (AUC, higher)
  "denoising-dirty-documents"           # image-to-image regression (RMSE, lower)
  "leaf-classification"                 # tabular multiclass from pre-extracted features (logloss, lower)
  "new-york-city-taxi-fare-prediction"  # tabular regression (RMSE, lower)           — biggest (~5GB)
)
if [ -n "${TASKS:-}" ]; then read -r -a TASK_ARR <<< "$TASKS"; else TASK_ARR=("${DEFAULT_TASKS[@]}"); fi

fmt_time() { local t=$1; echo "$((t/3600))h $(((t%3600)/60))m $((t%60))s"; }

echo "################################################################"
echo "# beta3 Lite sanity sweep"
echo "#   dataset_dir = ${DATASET_DIR}"
echo "#   GPU=${GPU}  num_gpus=${NUM_GPUS}  cpus=${CPUS}  budget=$(fmt_time ${TIME_LIMIT_SECS})  steps=${STEPS}"
echo "#   tasks (${#TASK_ARR[@]}): ${TASK_ARR[*]}"
echo "################################################################"

declare -a STATUS
i=0
for EXP_ID in "${TASK_ARR[@]}"; do
  i=$((i+1))
  DATA_DIR="${DATASET_DIR}/${EXP_ID}/prepared/public"
  DESC="${DATA_DIR}/description.md"
  echo ""
  echo "================= [$i/${#TASK_ARR[@]}] ${EXP_ID} ================="
  if [ ! -f "$DESC" ]; then
    echo "SKIP — ${DESC} not found."
    echo "      Pull prepared data first, e.g.:"
    echo "      huggingface-cli download TIGER-Lab/mle-bench --repo-type dataset \\"
    echo "        --include \"${EXP_ID}/*\" --local-dir \"${DATASET_DIR}\""
    STATUS+=("${EXP_ID}:SKIP(no-data)")
    continue
  fi
  # exp_name gets a timestamp prefix inside prep_cfg(); result_parse_agent
  # recovers the slug via split('_')[2] for the grading-server header (moot
  # here since the server is off, but keep the invariant).
  CUDA_VISIBLE_DEVICES="$GPU" \
  timeout --foreground --signal=TERM --kill-after=10s "${TIME_LIMIT_SECS}s" \
    python run.py \
      exp_id="${EXP_ID}" \
      dataset_dir="${DATASET_DIR}" \
      data_dir="${DATA_DIR}" \
      desc_file="${DESC}" \
      exp_name="${EXP_ID}" \
      agent.search.num_gpus="${NUM_GPUS}" \
      agent.search.parallel_search_num="${NUM_GPUS}" \
      agent.time_limit="${TIME_LIMIT_SECS}" \
      agent.steps="${STEPS}" \
      start_cpu_id=0 \
      cpu_number="${CPUS}"
  rc=$?
  case $rc in
    0)  STATUS+=("${EXP_ID}:ok") ;;
    124) STATUS+=("${EXP_ID}:TIMEOUT(budget reached — ok for sanity)") ;;
    *)  STATUS+=("${EXP_ID}:FAIL(exit=${rc})") ;;
  esac
done

echo ""
echo "################################################################"
echo "# Sanity summary"
echo "################################################################"
for s in "${STATUS[@]}"; do echo "  - $s"; done
echo ""
echo "Per-task journal + best solution: mlevolve/runs/<timestamp>_<slug>/  (auto-metric recorded)."
