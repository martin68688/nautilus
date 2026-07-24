#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC="${WP8_FORMAL_SOURCE_ROOT:-/opt/nautilus}"
TASK_ROOT="${WP8_FORMAL_TASK_ROOT:-/task}"
MEMORY_ROOT="${WP8_FORMAL_MEMORY_ROOT:-/memory}"
CONTRACT_ROOT="${WP8_FORMAL_CONTRACT_ROOT:-/contract}"
OUTPUT="${WP8_FORMAL_OUTPUT_ROOT:-/output}"
TEMPLATE="$CONTRACT_ROOT/BLOCK_TEMPLATE.json"
CONTENT_MANIFEST="$CONTRACT_ROOT/STAGING_CONTENT_MANIFEST.json"
EXPECTED_CONTENT_HASH="${WP8_FORMAL_STAGING_CONTENT_HASH:?required}"
EXPECTED_GATE_HASH="${WP8_FORMAL_STAGING_GATE_HASH:?required}"
ACTUAL_IMAGE_ID="${WP8_FORMAL_ACTUAL_IMAGE_ID:?required}"
POD_NAME="${POD_NAME:?required}"
POD_NAMESPACE="${POD_NAMESPACE:?required}"
POD_UID="${POD_UID:?required}"

test -d "$SRC/mlevolve"
test -f "$TEMPLATE"
test -f "$CONTENT_MANIFEST"
test -d "$TASK_ROOT/input"
test -f "$TASK_ROOT/fixed_holdout_manifest.json"
test -f "$MEMORY_ROOT/CURRENT.json"
test -f /secrets/mlevolve.env
test -d "$OUTPUT"
test -z "$(find "$OUTPUT" -mindepth 1 -print -quit)"
test ! -e /workspace
test ! -e /fixed/evaluator_view

exec > >(tee -a "$OUTPUT/TRAINING_LAUNCHER.log") 2>&1
printf 'training_preflight\n' > "$OUTPUT/STATE"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT/TRAINING_STARTED_AT"

finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$OUTPUT/TRAINING_LAUNCHER_EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then
    printf 'training_launcher_failed\n' > "$OUTPUT/STATE"
  fi
  exit "$rc"
}
terminate() {
  trap - TERM
  printf 'TERM\n' > "$OUTPUT/TRAINING_SIGNAL"
  exit 143
}
trap finish EXIT
trap terminate TERM

export PYTHONPATH="$SRC/mlevolve:$SRC"
export MLEVOLVE_CONFIG="$SRC/mlevolve/config/config_authority_formal_enforce.yaml"
export HF_HOME=/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME=/cache/torch
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" /work/runtime

python - "$TEMPLATE" "$CONTENT_MANIFEST" "$OUTPUT/BLOCK_CONTRACT.json" \
  "$EXPECTED_CONTENT_HASH" "$EXPECTED_GATE_HASH" "$POD_NAME" \
  "$POD_NAMESPACE" "$POD_UID" <<'PY'
import json
import sys
from pathlib import Path

from fixed_holdout.formal_runtime import (
    FORMAL_STAGING_CONTENT_SCHEMAS,
    build_runtime_block_contract,
    payload_hash,
    read_object,
)

template_path, content_path, output_path = map(Path, sys.argv[1:4])
expected_content, expected_gate, pod_name, namespace, uid = sys.argv[4:]
content = read_object(content_path)
assert content.get("schema") in FORMAL_STAGING_CONTENT_SCHEMAS, content.get("schema")
assert content.get("manifest_hash") == payload_hash(content, "manifest_hash")
assert content["manifest_hash"] == expected_content
template = read_object(template_path)
block_id = template.get("block_id")
entry = (content.get("blocks_by_id") or {}).get(block_id) or {}
assert entry.get("block_template_hash") == template.get("template_hash")
runtime = build_runtime_block_contract(
    template,
    staging_content_hash=expected_content,
    staging_gate_hash=expected_gate,
    pod_name=pod_name,
    pod_namespace=namespace,
    pod_uid=uid,
)
output_path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n")
PY

mapfile -t CONTRACT_VALUES < <(python - "$OUTPUT/BLOCK_CONTRACT.json" <<'PY'
import json, sys
from pathlib import Path
p=json.loads(Path(sys.argv[1]).read_text())
c=p["candidate_execution_contract"]
values=(
 p["block_id"],p["task_id"],p["protocol_ref"],str(p["agent_seed"]),
 str(p["steps_per_condition"]),str(p["initial_drafts_per_condition"]),
 str(p["agent_time_limit_seconds"]),str(p["condition_launcher_timeout_seconds"]),
 p["bundle_id"],p["bundle_manifest_sha256"],c["contract_id"],
 str(c["max_execution_seconds"]),str(c["max_epochs"]),str(c["max_cv_folds"]),
 str(c["max_trainable_models"]),",".join(c["allowed_import_roots"]),
 c["contract_hash"],p["contract_hash"],p["container_image_digest"],
)
print("\n".join(values))
PY
)
BLOCK_ID="${CONTRACT_VALUES[0]}"
TASK="${CONTRACT_VALUES[1]}"
PROTOCOL_REF="${CONTRACT_VALUES[2]}"
SEED="${CONTRACT_VALUES[3]}"
STEPS="${CONTRACT_VALUES[4]}"
INITIAL_DRAFTS="${CONTRACT_VALUES[5]}"
TIME_LIMIT="${CONTRACT_VALUES[6]}"
LAUNCHER_TIMEOUT="${CONTRACT_VALUES[7]}"
BUNDLE_ID="${CONTRACT_VALUES[8]}"
BUNDLE_MANIFEST_SHA256="${CONTRACT_VALUES[9]}"
CANDIDATE_CONTRACT_ID="${CONTRACT_VALUES[10]}"
CANDIDATE_TIMEOUT="${CONTRACT_VALUES[11]}"
CANDIDATE_MAX_EPOCHS="${CONTRACT_VALUES[12]}"
CANDIDATE_MAX_CV_FOLDS="${CONTRACT_VALUES[13]}"
CANDIDATE_MAX_MODELS="${CONTRACT_VALUES[14]}"
CANDIDATE_IMPORTS_CSV="${CONTRACT_VALUES[15]}"
CANDIDATE_CONTRACT_HASH="${CONTRACT_VALUES[16]}"
RUNTIME_CONTRACT_HASH="${CONTRACT_VALUES[17]}"
EXPECTED_IMAGE_DIGEST="${CONTRACT_VALUES[18]}"
test "$ACTUAL_IMAGE_ID" = "$EXPECTED_IMAGE_DIGEST"
PROTOCOL_ID="${PROTOCOL_REF%%@*}"
PROTOCOL_VERSION_HASH="${PROTOCOL_REF#*@}"
PROTOCOL_VERSION="${PROTOCOL_VERSION_HASH%%#*}"
CANDIDATE_IMPORTS="[$CANDIDATE_IMPORTS_CSV]"

python - "$SRC" "$TASK_ROOT" "$MEMORY_ROOT" "$OUTPUT" \
  "$OUTPUT/BLOCK_CONTRACT.json" "$ACTUAL_IMAGE_ID" <<'PY'
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from authority.memory_snapshot import ImmutableBaseBundle
from fixed_holdout.common import sha256_file
from fixed_holdout.formal_runtime import (
    TRAINING_ISOLATION_SCHEMA,
    mount_points,
    payload_hash,
    read_object,
    verify_source_snapshot,
)
from fixed_holdout.validation import validate_train_view

source, task, memory, output, contract_path = map(Path, sys.argv[1:6])
actual_image_id = sys.argv[6]
contract = read_object(contract_path)
source_report = verify_source_snapshot(
    source,
    expected_source_sha256=contract["source_snapshot_sha256"],
    expected_manifest_file_sha256=contract["source_manifest_file_sha256"],
)
(output / "SOURCE_PREFLIGHT.json").write_text(
    json.dumps(source_report, indent=2, sort_keys=True) + "\n"
)
train_path = task / "fixed_holdout_manifest.json"
assert sha256_file(train_path) == contract["train_manifest_sha256"]
train = validate_train_view(train_path, task / "input")
for key in ("task_id", "split_id", "metric", "maximize", "protocol_ref"):
    assert train.get(key) == contract.get(key), (key, train.get(key), contract.get(key))
assert train["hidden_labels_present"] is False
assert not list(task.rglob("labels.csv"))

current_path = memory / "CURRENT.json"
assert sha256_file(current_path) == contract["bundle_current_file_sha256"]
current = read_object(current_path)
bundle_path = memory / current["bundle_path"]
bundle = ImmutableBaseBundle.load(bundle_path, verify_artifacts=True)
assert bundle.bundle_id == current["bundle_id"] == contract["bundle_id"]
assert bundle.manifest_sha256 == current["manifest_sha256"] == contract["bundle_manifest_sha256"]
assert sha256_file(bundle_path / "manifest.json") == contract["bundle_manifest_file_sha256"]

points = mount_points()
required = {"/opt/nautilus", "/task", "/memory", "/contract", "/output", "/secrets/mlevolve.env"}
assert required <= points, (required, sorted(points))
assert "/workspace" not in points and "/fixed/evaluator_view" not in points
assert not Path("/workspace").exists()
assert not Path("/fixed/evaluator_view").exists()
for root in (source, task, memory):
    probe = root / ".formal_write_probe"
    try:
        probe.write_text("forbidden")
    except OSError:
        pass
    else:
        probe.unlink(missing_ok=True)
        raise AssertionError(f"read-only mount is writable: {root}")

gpu_lines = [line for line in subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines() if line.strip()]
assert len(gpu_lines) == 1, gpu_lines
cpu_count = len(os.sched_getaffinity(0))
assert cpu_count == 8, cpu_count
imports = {}
for name in contract["candidate_execution_contract"]["allowed_import_roots"]:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        imports[name] = {"importable": False, "error_type": type(error).__name__, "error": str(error)}
    else:
        imports[name] = {"importable": True, "version": str(getattr(module, "__version__", "unknown"))}
assert all(row["importable"] for row in imports.values()), imports
assert actual_image_id == contract["container_image_digest"]
receipt = {
    "schema": TRAINING_ISOLATION_SCHEMA,
    "block_id": contract["block_id"],
    "runtime_contract_hash": contract["contract_hash"],
    "training_pod_identity": contract["training_pod_identity"],
    "source_snapshot_sha256": contract["source_snapshot_sha256"],
    "source_manifest_file_sha256": contract["source_manifest_file_sha256"],
    "train_manifest_sha256": contract["train_manifest_sha256"],
    "bundle_id": contract["bundle_id"],
    "bundle_manifest_sha256": contract["bundle_manifest_sha256"],
    "bundle_current_file_sha256": contract["bundle_current_file_sha256"],
    "container_image_digest": contract["container_image_digest"],
    "gpu_visible": True,
    "gpu_count": 1,
    "gpu_identity": gpu_lines,
    "cpu_count": cpu_count,
    "source_read_only": True,
    "train_view_read_only": True,
    "bundle_read_only": True,
    "solver_secret_single_file_mount": True,
    "whole_workspace_absent": True,
    "evaluator_view_absent": True,
    "terminal_labels_absent": True,
    "candidate_imports": imports,
    "all_candidate_import_roots_importable": True,
    "mount_points": sorted(points),
    "receipt_hash": "",
}
receipt["receipt_hash"] = payload_hash(receipt, "receipt_hash")
(output / "TRAINING_ISOLATION.json").write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n"
)
PY

set -a
. /secrets/mlevolve.env
set +a
export DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL
test -n "${DEEPSEEK_API_KEY:-}"
test -n "${DEEPSEEK_BASE_URL:-}"
test -n "${DEEPSEEK_MODEL:-}"
python - "$OUTPUT/PROVIDER_ATTESTATION.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
from urllib.parse import urlparse
base=os.environ["DEEPSEEK_BASE_URL"]
host=(urlparse(base).hostname or "").lower()
assert host not in {"", "localhost", "127.0.0.1"}
p={"schema":"decision_admissibility_wp8_tier2_provider_attestation_v1","provider":"deepseek","model":os.environ["DEEPSEEK_MODEL"],"base_url_sha256":hashlib.sha256(base.encode()).hexdigest(),"api_key_present":True,"non_local_endpoint":True,"receipt_hash":""}
u={k:v for k,v in p.items() if k!="receipt_hash"}
p["receipt_hash"]=hashlib.sha256(json.dumps(u,sort_keys=True,separators=(",",":")).encode()).hexdigest()
Path(sys.argv[1]).write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY

run_condition() {
  local condition="$1"
  local position="$2"
  local root="$OUTPUT/conditions/$condition"
  local memory_enabled=true
  if [[ "$condition" == "no_memory" ]]; then memory_enabled=false; fi
  test ! -e "$root"
  mkdir -p "$root/runs" "$root/workspace"
  printf 'running\n' > "$root/STATE"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$root/STARTED_AT"
  cd "$SRC/mlevolve"
  set +e
  timeout --foreground --signal=TERM --kill-after=30s "${LAUNCHER_TIMEOUT}s" \
    python -u run.py \
      exp_id="$TASK" \
      exp_name="${BLOCK_ID}-${condition}" \
      dataset_dir="$TASK_ROOT/input" \
      data_dir="$TASK_ROOT/input" \
      desc_file="$TASK_ROOT/input/description.md" \
      log_dir="$root/runs" \
      workspace_dir="$root/workspace" \
      fixed_holdout.enabled=true \
      fixed_holdout.evaluation_mode=terminal_only \
      fixed_holdout.train_manifest_path="$TASK_ROOT/fixed_holdout_manifest.json" \
      fixed_holdout.bypass_protocol_gates=true \
      fixed_holdout.internal_metric_disposition=search_only \
      coldstart.use_coldstart=false \
      agent.check_data_leakage=false \
      agent.protocol_repair.enabled=false \
      evaluation_authority.mode=enforce \
      evaluation_authority.active_protocol_id="$PROTOCOL_ID" \
      evaluation_authority.active_protocol_version="$PROTOCOL_VERSION" \
      evaluation_authority.rollout_id="${BLOCK_ID}-${condition}" \
      evaluation_authority.collector_version=2 \
      evaluation_authority.require_bound_bundle=true \
      evaluation_authority.expected_bundle_id="$BUNDLE_ID" \
      evaluation_authority.expected_bundle_manifest_sha256="$BUNDLE_MANIFEST_SHA256" \
      external_skill_memory.enable=true \
      external_skill_memory.bundle_root="$MEMORY_ROOT" \
      external_skill_memory.current_pointer_path=CURRENT.json \
      external_skill_memory.retrieval_control="$condition" \
      agent.draft_role_policy.enabled=true \
      agent.draft_role_policy.roles='[coldstart_baseline,memory_transfer,novel_exploration]' \
      agent.candidate_execution_contract.enabled=true \
      agent.candidate_execution_contract.contract_id="$CANDIDATE_CONTRACT_ID" \
      agent.candidate_execution_contract.max_execution_seconds="$CANDIDATE_TIMEOUT" \
      agent.candidate_execution_contract.max_epochs="$CANDIDATE_MAX_EPOCHS" \
      agent.candidate_execution_contract.max_cv_folds="$CANDIDATE_MAX_CV_FOLDS" \
      agent.candidate_execution_contract.max_trainable_models="$CANDIDATE_MAX_MODELS" \
      agent.candidate_execution_contract.allowed_import_roots="$CANDIDATE_IMPORTS" \
      agent.candidate_execution_contract.allow_remote_assets=false \
      agent.candidate_execution_contract.allow_unverified_local_assets=false \
      agent.candidate_execution_contract.allow_dataset_wide_per_sample_precompute=true \
      agent.candidate_execution_contract.allow_source_score_inheritance=false \
      run_identity.experiment_group="${BLOCK_ID}-${condition}" \
      run_identity.baseline_reference_group="${BLOCK_ID}-no_memory" \
      run_identity.memory_enabled="$memory_enabled" \
      run_identity.memory_system="$condition" \
      run_identity.memory_version="$BUNDLE_MANIFEST_SHA256" \
      agent.initial_drafts="$INITIAL_DRAFTS" \
      agent.steps="$STEPS" \
      agent.time_limit="$TIME_LIMIT" \
      agent.seed="$SEED" \
      agent.code.temp=1.0 \
      agent.feedback.temp=1.0 \
      agent.search.num_gpus=1 \
      agent.search.parallel_search_num=1 \
      cpu_number=8 \
      exec.timeout="$CANDIDATE_TIMEOUT" \
      2>&1 | tee "$root/run_stdout.log"
  local rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$root/RUN_EXIT_CODE"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$root/FINISHED_AT"
  if [[ "$rc" -eq 0 ]]; then
    printf 'training_complete_unscored\n' > "$root/STATE"
  else
    printf 'pre_terminal_failure\n' > "$root/STATE"
  fi
  python - "$OUTPUT/BLOCK_CONTRACT.json" "$root" "$condition" "$position" "$rc" <<'PY'
import hashlib, json, sys
from pathlib import Path
from fixed_holdout.formal_runtime import CONDITION_RECEIPT_SCHEMA, payload_hash
contract=json.loads(Path(sys.argv[1]).read_text()); root=Path(sys.argv[2]); condition=sys.argv[3]; position=int(sys.argv[4]); rc=int(sys.argv[5])
scores=list(root.rglob("fixed_holdout_scores*.json"))
stdout=root/"run_stdout.log"
p={"schema":CONDITION_RECEIPT_SCHEMA,"block_id":contract["block_id"],"runtime_contract_hash":contract["contract_hash"],"condition":condition,"position":position,"retrieval_control":condition,"memory_enabled":condition!="no_memory","memory_system":condition,"task_id":contract["task_id"],"agent_seed":contract["agent_seed"],"run_exit_code":rc,"steps":contract["steps_per_condition"],"initial_drafts":contract["initial_drafts_per_condition"],"agent_time_limit_seconds":contract["agent_time_limit_seconds"],"launcher_timeout_seconds":contract["condition_launcher_timeout_seconds"],"candidate_execution_contract_hash":contract["candidate_execution_contract_hash"],"training_pod_identity":contract["training_pod_identity"],"only_experimental_variable":"external_skill_memory.retrieval_control","terminal_metric_observed":False,"pre_evaluator_score_file_count":len(scores),"run_stdout_sha256":hashlib.sha256(stdout.read_bytes()).hexdigest() if stdout.is_file() else "","receipt_hash":""}
p["receipt_hash"]=payload_hash(p,"receipt_hash")
(root/"CONDITION_RUNTIME_RECEIPT.json").write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
assert not scores, scores
PY
}

mapfile -t CONDITIONS < <(python - "$OUTPUT/BLOCK_CONTRACT.json" <<'PY'
import json,sys
from pathlib import Path
print("\n".join(json.loads(Path(sys.argv[1]).read_text())["condition_order"]))
PY
)
for position in "${!CONDITIONS[@]}"; do
  run_condition "${CONDITIONS[$position]}" "$position"
done

python -m fixed_holdout.formal_block_training \
  --output-root "$OUTPUT" \
  --block-contract "$OUTPUT/BLOCK_CONTRACT.json" \
  --bundle-root "$MEMORY_ROOT" \
  > "$OUTPUT/TRAINING_FINALIZER_STDOUT.json"

python - "$SRC" "$OUTPUT/BLOCK_CONTRACT.json" "$OUTPUT/SOURCE_POSTRUN.json" <<'PY'
import json,sys
from pathlib import Path
from fixed_holdout.formal_runtime import verify_source_snapshot
source, contract_path, output = map(Path, sys.argv[1:])
contract=json.loads(contract_path.read_text())
r=verify_source_snapshot(source,expected_source_sha256=contract["source_snapshot_sha256"],expected_manifest_file_sha256=contract["source_manifest_file_sha256"])
output.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
PY

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT/TRAINING_FINISHED_AT"
printf 'training_complete_unscored\n' > "$OUTPUT/STATE"
touch "$OUTPUT/TRAINING_COMPLETE"
trap - EXIT
printf '0\n' > "$OUTPUT/TRAINING_LAUNCHER_EXIT_CODE"
