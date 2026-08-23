#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 TASK_ID DATA_ROOT DESCRIPTION OUTPUT_ROOT EXP_NAME" >&2
  exit 2
fi

TASK_ID="$1"
DATA_ROOT="$2"
DESCRIPTION="$3"
OUTPUT_ROOT="$4"
EXP_NAME="$5"

RUNTIME=/workspace/nautilus-exp-end2end-agent-v162-iterative-memory-curve
CONFIG=${RUNTIME}/experiments/end2end_memory_systems_20260804/systems_v162_iterative_memory_curve_phase1_nomemory_3h_luna_low/no_memory_phase1.yaml
ATTEMPT_ROOT=${OUTPUT_ROOT}/runs/phase-1-no-memory/attempt-000

test -n "${OPENAI_API_KEY:-}"
test "${OPENAI_BASE_URL:-}" = "http://cliproxyapi-haoming.ecepxie.svc.cluster.local:8317/v1"
test "${OPENAI_MODEL:-}" = "gpt-5.6-luna"
test "${OPENAI_REASONING_EFFORT:-}" = "low"
test -d "${RUNTIME}/mlevolve"
test -f "${CONFIG}"
test -d "${DATA_ROOT}"
test -f "${DESCRIPTION}"
test ! -e "${OUTPUT_ROOT}"

mkdir -p "${ATTEMPT_ROOT}" "${OUTPUT_ROOT}/runtime-cache/huggingface" "${OUTPUT_ROOT}/runtime-cache/torch" "${OUTPUT_ROOT}/runtime-cache/xdg"
export HF_HOME="${OUTPUT_ROOT}/runtime-cache/huggingface"
export TORCH_HOME="${OUTPUT_ROOT}/runtime-cache/torch"
export XDG_CACHE_HOME="${OUTPUT_ROOT}/runtime-cache/xdg"
export PYTHONPATH="${RUNTIME}/mlevolve:${RUNTIME}"
export MLEVOLVE_CONFIG="${CONFIG}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

cd "${RUNTIME}/mlevolve"
exec python run.py \
  exp_id="${TASK_ID}" \
  dataset_dir="${DATA_ROOT}" \
  data_dir="${DATA_ROOT}" \
  desc_file="${DESCRIPTION}" \
  exp_name="${EXP_NAME}" \
  log_dir="${ATTEMPT_ROOT}" \
  workspace_dir="${ATTEMPT_ROOT}" \
  agent.steps=2147483647 \
  agent.time_limit=10800 \
  exec.timeout=10800 \
  agent.search.num_gpus=1 \
  agent.search.parallel_search_num=1 \
  cpu_number=16
