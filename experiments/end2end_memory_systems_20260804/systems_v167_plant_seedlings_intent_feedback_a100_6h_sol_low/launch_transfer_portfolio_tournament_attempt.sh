#!/usr/bin/env bash
set -euo pipefail

ATTEMPT_ID="${1:?usage: launch_transfer_portfolio_tournament_attempt.sh ATTEMPT_ID}"
if [[ ! "$ATTEMPT_ID" =~ ^attempt-[0-9]{3}$ ]]; then
  echo "invalid attempt identity: $ATTEMPT_ID" >&2
  exit 2
fi

SYSTEM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SYSTEM_DIR/../../.." && pwd)"
MLEVOLVE_DIR="$REPO_ROOT/mlevolve"
export MLEVOLVE_CONFIG="$SYSTEM_DIR/intent_feedback_tournament.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/experiment-end2end-plant-seedlings-v167-transfer-tournament-6h-sol-low-r1}"
DATA_ROOT="${DATA_ROOT:-/workspace/experiment-end2end-plant-seedlings-v157-dynamic-no-resolver-6h-luna-low/data/public}"
DESC_FILE="${DESC_FILE:-/workspace/nautilus-exp-end2end-agent-v164-plant-seedlings-transfer-only-a100-luna-low-r1/experiments/end2end_memory_systems_20260804/systems_v158_plant_seedlings_component_slots_6h_luna_low/dataset_description.md}"
ATTEMPT_ROOT="$OUTPUT_ROOT/runs/$ATTEMPT_ID"
RECEIPT="$OUTPUT_ROOT/preflight-$ATTEMPT_ID-public-pretrained.json"
CANDIDATE_RECEIPT="$OUTPUT_ROOT/preflight-$ATTEMPT_ID-candidate-public-pretrained.json"
CANDIDATE_CACHE_ROOT="$OUTPUT_ROOT/preflight-$ATTEMPT_ID-candidate-runtime-cache"
LAUNCH_LOG="$OUTPUT_ROOT/launcher-$ATTEMPT_ID.log"
PID_FILE="$OUTPUT_ROOT/main-$ATTEMPT_ID.pid"
PORTFOLIO_MANIFEST_PATH="${MLEVOLVE_TRANSFER_PORTFOLIO_MANIFEST_PATH:-}"
PORTFOLIO_MANIFEST_SHA256="${MLEVOLVE_TRANSFER_PORTFOLIO_MANIFEST_SHA256:-}"

for target in \
  "$ATTEMPT_ROOT" \
  "$RECEIPT" \
  "$CANDIDATE_RECEIPT" \
  "$CANDIDATE_CACHE_ROOT" \
  "$LAUNCH_LOG" \
  "$PID_FILE"; do
  if [[ -e "$target" ]]; then
    echo "refusing to overwrite existing attempt artifact: $target" >&2
    exit 3
  fi
done

test -n "${OPENAI_API_KEY:-}"
test "${OPENAI_BASE_URL:-}" = "http://cliproxyapi-haoming.ecepxie.svc.cluster.local:8317/v1"
test "${OPENAI_MODEL:-}" = "gpt-5.6-sol"
test -z "${RUNFOREST_DEV_EXECUTION_ROLE:-}"
test -d "$DATA_ROOT/train"
test -d "$DATA_ROOT/test"
test -f "$DESC_FILE"
test -f "$MLEVOLVE_CONFIG"

if [[ -z "$PORTFOLIO_MANIFEST_PATH" || -z "$PORTFOLIO_MANIFEST_SHA256" ]]; then
  echo "immutable four-seat portfolio path and SHA-256 are required" >&2
  exit 4
fi
test -f "$PORTFOLIO_MANIFEST_PATH"
ACTUAL_PORTFOLIO_SHA256="$(sha256sum "$PORTFOLIO_MANIFEST_PATH" | awk '{print $1}')"
if [[ "$ACTUAL_PORTFOLIO_SHA256" != "$PORTFOLIO_MANIFEST_SHA256" ]]; then
  echo "portfolio manifest SHA-256 mismatch" >&2
  exit 5
fi

python - "$PORTFOLIO_MANIFEST_PATH" <<'PY'
import json
import sys

expected = [
    "comprehensive_best",
    "backbone_diversity_1",
    "backbone_diversity_2",
    "descriptor_fusion_diversity",
]
pack = json.load(open(sys.argv[1], encoding="utf-8"))
portfolio = pack.get("transfer_portfolio") or {}
candidates = portfolio.get("candidates") or []
seats = [str(row.get("seat") or "") for row in candidates]
if seats != expected:
    raise RuntimeError(f"unexpected frozen portfolio seats: {seats}")
if int(portfolio.get("actual_size", 0) or 0) != 4:
    raise RuntimeError("frozen portfolio is not exactly four candidates")
if len({str(row.get("candidate_id") or "") for row in candidates}) != 4:
    raise RuntimeError("frozen portfolio candidate identities are not unique")
print(f"frozen tournament portfolio preflight passed: seats={seats}")
PY

mkdir -p "$OUTPUT_ROOT"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MLEVOLVE_CANDIDATE_HF_HUB_SEED="${MLEVOLVE_CANDIDATE_HF_HUB_SEED:-/workspace/experiment-end2end-plant-seedlings-v167-intent-feedback-6h-sol-low-a100-r1/preflight-attempt-013-host-runtime-cache/huggingface/hub}"
export MLEVOLVE_CANDIDATE_HF_REPOSITORIES="${MLEVOLVE_CANDIDATE_HF_REPOSITORIES:-timm/efficientnet_b3.ra2_in1k}"
export MLEVOLVE_CANDIDATE_TORCH_HUB_SEED="${MLEVOLVE_CANDIDATE_TORCH_HUB_SEED:-$SYSTEM_DIR/public_pretrained_assets/torch_hub_checkpoints}"
export MLEVOLVE_CANDIDATE_TORCH_CHECKPOINTS="${MLEVOLVE_CANDIDATE_TORCH_CHECKPOINTS:-efficientnet_b0_rwightman-7f5810bc.pth,efficientnet_b3_rwightman-b3899882.pth}"
test -f "$MLEVOLVE_CANDIDATE_TORCH_HUB_SEED/efficientnet_b0_rwightman-7f5810bc.pth"
test -f "$MLEVOLVE_CANDIDATE_TORCH_HUB_SEED/efficientnet_b3_rwightman-b3899882.pth"

HF_HUB_CACHE="$MLEVOLVE_CANDIDATE_HF_HUB_SEED" \
HUGGINGFACE_HUB_CACHE="$MLEVOLVE_CANDIDATE_HF_HUB_SEED" \
python "$SYSTEM_DIR/../systems_v165_plant_seedlings_transfer_attributes_a100_6h_luna_low/preflight_public_pretrained.py" \
  --hub-cache "$MLEVOLVE_CANDIDATE_HF_HUB_SEED" \
  --receipt "$RECEIPT"
PYTHONPATH="$MLEVOLVE_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python "$SYSTEM_DIR/preflight_candidate_public_pretrained.py" \
  --runtime-cache-root "$CANDIDATE_CACHE_ROOT" \
  --receipt "$CANDIDATE_RECEIPT"

cd "$MLEVOLVE_DIR"
python - <<'PY'
import os
from pathlib import Path
from config import _load_config_tree
from engine.draft_roles import TRANSFER_PORTFOLIO_DRAFT_ROLES

cfg = _load_config_tree(Path(os.environ["MLEVOLVE_CONFIG"]))
policy = cfg.agent.draft_role_policy
roles = list(policy.roles)
assert roles == list(TRANSFER_PORTFOLIO_DRAFT_ROLES)
assert cfg.agent.initial_drafts == 4
assert cfg.agent.search.num_drafts == 4
assert cfg.agent.max_fusion_drafts == 0
assert bool(policy.ensure_valid_candidate_per_role)
assert bool(policy.transfer_portfolio_tournament_enabled)
assert int(policy.role_balance_min_valid_candidates) == 1
assert int(policy.transfer_portfolio_round1_valid_successors_per_seat) == 1
assert int(policy.transfer_portfolio_survivor_count) == 2
assert int(policy.transfer_portfolio_finalist_valid_successors_per_seat) == 2
assert bool(cfg.external_skill_memory.cross_task_transfer_portfolio_enabled)
assert bool(cfg.external_skill_memory.memory_strategy_plan_alignment_enabled)
assert bool(cfg.external_skill_memory.memory_strategy_recoverable_rejection_enabled)
assert cfg.external_skill_memory.memory_strategy_host_required_backbone_family == ""
assert not bool(cfg.external_skill_memory.memory_strategy_host_require_public_pretrained)
assert cfg.external_skill_memory.memory_strategy_host_required_pretrained_loader == ""
print(f"four-seat tournament config preflight passed: roles={roles}")
PY

CPU_NUMBER="${MLEVOLVE_CPU_NUMBER:-16}"
if [[ ! "$CPU_NUMBER" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid MLEVOLVE_CPU_NUMBER=$CPU_NUMBER; expected a positive integer" >&2
  exit 2
fi

RUN_ARGS=(
  run.py
  exp_id=plant-seedlings-classification
  dataset_dir="$DATA_ROOT"
  data_dir="$DATA_ROOT"
  desc_file="$DESC_FILE"
  exp_name="e2e-plant-seedlings-v167-transfer-tournament-6h-sol-low-r1-$ATTEMPT_ID"
  log_dir="$ATTEMPT_ROOT"
  workspace_dir="$ATTEMPT_ROOT"
  agent.steps=2147483647
  agent.time_limit=21600
  exec.timeout=21600
  finalize_reserve_seconds=0
  agent.search.num_gpus=1
  agent.search.parallel_search_num=1
  "cpu_number=$CPU_NUMBER"
  "external_skill_memory.cross_task_transfer_portfolio_manifest_path=$PORTFOLIO_MANIFEST_PATH"
  "external_skill_memory.cross_task_transfer_portfolio_manifest_sha256=$PORTFOLIO_MANIFEST_SHA256"
)

if [[ "${MLEVOLVE_FOREGROUND:-0}" == "1" ]]; then
  printf '%s\n' "$$" >"$PID_FILE"
  echo "starting $ATTEMPT_ID in foreground pid=$$ portfolio_manifest=$PORTFOLIO_MANIFEST_PATH portfolio_sha256=$PORTFOLIO_MANIFEST_SHA256 log=$LAUNCH_LOG"
  exec python "${RUN_ARGS[@]}" >"$LAUNCH_LOG" 2>&1
fi

nohup python "${RUN_ARGS[@]}" >"$LAUNCH_LOG" 2>&1 &
PID=$!
printf '%s\n' "$PID" >"$PID_FILE"
echo "started $ATTEMPT_ID pid=$PID portfolio_manifest=$PORTFOLIO_MANIFEST_PATH portfolio_sha256=$PORTFOLIO_MANIFEST_SHA256 log=$LAUNCH_LOG"
