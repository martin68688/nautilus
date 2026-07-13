#!/usr/bin/env bash
set -euo pipefail

work="${RUNFOREST_DEV_WORKDIR:?RUNFOREST_DEV_WORKDIR is required}"
tag="${RUNFOREST_DEV_TAG:?RUNFOREST_DEV_TAG is required}"
output_runs="${RUNFOREST_DEV_OUTPUT_RUNS:-/workspace/nautilus/mlevolve/runs}"

cd "${work}/mlevolve"
set -a
. ./.env
set +a
export DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL
export MLEVOLVE_CONFIG="${work}/mlevolve/config/config_run_forest_stage_hybrid.yaml"
export RUNFOREST_DEV_EXECUTION_ROLE=memory_reproduction

exec python run.py \
  exp_id=spooky-author-identification \
  dataset_dir=./data \
  data_dir=./data/spooky-author-identification/prepared/public \
  desc_file=./data/spooky-author-identification/prepared/public/description.md \
  exp_name="${tag}" \
  log_dir="${output_runs}" \
  workspace_dir="${output_runs}" \
  agent.steps=18 \
  agent.search.num_gpus=1 \
  agent.search.parallel_search_num=1 \
  cpu_number=8
