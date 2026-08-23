#!/usr/bin/env bash
set -euo pipefail

ATTEMPT_ID="${1:?usage: launch_intent_feedback_attempt.sh ATTEMPT_ID}"
if [[ ! "$ATTEMPT_ID" =~ ^attempt-[0-9]{3}$ ]]; then
  echo "invalid attempt identity: $ATTEMPT_ID" >&2
  exit 2
fi

SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SYSTEM_DIR/../../.." && pwd)"
MLEVOLVE_DIR="$REPO_ROOT/mlevolve"
export MLEVOLVE_CONFIG="$SYSTEM_DIR/intent_feedback.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiment-end2end-plant-seedlings-v167-intent-feedback-6h-sol-low-a100-r1}"
DATA_ROOT="${DATA_ROOT:-/workspace/experiment-end2end-plant-seedlings-v157-dynamic-no-resolver-6h-luna-low/data/public}"
DESC_FILE="${DESC_FILE:-/workspace/nautilus-exp-end2end-agent-v164-plant-seedlings-transfer-only-a100-luna-low-r1/experiments/end2end_memory_systems_20260804/systems_v158_plant_seedlings_component_slots_6h_luna_low/dataset_description.md}"
ATTEMPT_ROOT="$OUTPUT_ROOT/runs/$ATTEMPT_ID"
RECEIPT="$OUTPUT_ROOT/preflight-$ATTEMPT_ID-public-pretrained.json"
LAUNCH_LOG="$OUTPUT_ROOT/launcher-$ATTEMPT_ID.log"
PID_FILE="$OUTPUT_ROOT/main-$ATTEMPT_ID.pid"

if [[ -e "$ATTEMPT_ROOT" || -e "$RECEIPT" || -e "$PID_FILE" ]]; then
  echo "refusing to overwrite existing attempt identity: $ATTEMPT_ID" >&2
  exit 3
fi
test -n "${OPENAI_API_KEY:-}"
test "${OPENAI_BASE_URL:-}" = "http://cliproxyapi-haoming.ecepxie.svc.cluster.local:8317/v1"
test "${OPENAI_MODEL:-}" = "gpt-5.6-sol"
test -d "$DATA_ROOT/train"
test -d "$DATA_ROOT/test"
test -f "$DESC_FILE"
test -f "$MLEVOLVE_CONFIG"

mkdir -p "$OUTPUT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python "$SYSTEM_DIR/../systems_v165_plant_seedlings_transfer_attributes_a100_6h_luna_low/preflight_public_pretrained.py" \
  --repair-cache \
  --receipt "$RECEIPT"

cd "$MLEVOLVE_DIR"
python - <<'PY'
import os
from pathlib import Path
from config import _load_config_tree

cfg = _load_config_tree(Path(os.environ["MLEVOLVE_CONFIG"]))
roles = list(cfg.agent.draft_role_policy.roles)
assert bool(cfg.external_skill_memory.enable)
assert "memory_transfer" in roles
assert bool(cfg.external_skill_memory.memory_strategy_plan_alignment_enabled)
assert bool(cfg.external_skill_memory.memory_strategy_recoverable_rejection_enabled)
assert int(cfg.external_skill_memory.memory_strategy_post_review_repair_attempts) == 1
assert cfg.external_skill_memory.memory_strategy_host_required_backbone_family == "efficientnet_b3"
assert bool(cfg.external_skill_memory.memory_strategy_host_require_public_pretrained)
print(f"intent feedback config preflight passed: roles={roles}")
PY

nohup python run.py \
  exp_id=plant-seedlings-classification \
  dataset_dir="$DATA_ROOT" \
  data_dir="$DATA_ROOT" \
  desc_file="$DESC_FILE" \
  exp_name="e2e-plant-seedlings-v167-intent-feedback-6h-sol-low-a100-r1-$ATTEMPT_ID" \
  log_dir="$ATTEMPT_ROOT" \
  workspace_dir="$ATTEMPT_ROOT" \
  agent.search.num_gpus=1 \
  agent.search.parallel_search_num=1 \
  cpu_number=16 \
  >"$LAUNCH_LOG" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
echo "started $ATTEMPT_ID pid=$PID receipt=$RECEIPT log=$LAUNCH_LOG"
