#!/usr/bin/env bash
set -euo pipefail

source_root="${SOURCE_ROOT:-/tmp/leaf-authority-fix-source}"
release_root="${RELEASE_ROOT:-/release}"
runtime_release="${RUNTIME_RELEASE:-/workspace/prevalence-audit-20260729-r3}"
output_root="${OUTPUT_ROOT:-/tmp/leaf-authority-fix-smoke-seed11}"
steps="${STEPS:-6}"
time_limit="${TIME_LIMIT:-3600}"
task=leaf-classification
seed=11

binding="${runtime_release}/bindings/${task}/HOST_PROTOCOL_BINDING.json"
memory_root="${release_root}/memory/natural"
memory_manifest="${memory_root}/MEMORY_MANIFEST.json"
graph="${memory_root}/run_forest_graph.json"
index="${memory_root}/run_forest_index.npz"
replay_targets="${memory_root}/replay_targets.json"
public_data="${release_root}/data/${task}/public"

test -d "${source_root}/mlevolve"
test -s "${binding}"
test -s "${memory_manifest}"
test -s "${graph}"
test -s "${index}"
test -s "${replay_targets}"
test -s "${public_data}/description.md"
test -s /run/host-key-source/collector.ed25519
test ! -e "${output_root}"
test -n "${DEEPSEEK_API_KEY:-}"
test -n "${DEEPSEEK_BASE_URL:-}"
test -n "${DEEPSEEK_MODEL:-}"

python - "${replay_targets}" "${memory_root}/replay_sources" \
  "${source_root}/mlevolve/runs" "${task}" <<'PY'
import json
import pathlib
import shutil
import sys

targets_path = pathlib.Path(sys.argv[1])
replay_root = pathlib.Path(sys.argv[2])
destination = pathlib.Path(sys.argv[3])
task_id = sys.argv[4]
payload = json.loads(targets_path.read_text(encoding="utf-8"))
matches = [
    row for row in payload.get("targets", []) if row.get("task_id") == task_id
]
if len(matches) != 1:
    raise ValueError(f"expected one replay target for {task_id}, found {len(matches)}")
target = matches[0]
if target.get("audit_status") != "verified_clean" or target.get("known_issue_codes"):
    raise ValueError("Leaf replay target is not verified clean")
run_id = str(target["run_id"])
source = replay_root / run_id
target_root = destination / run_id
if target_root.exists():
    shutil.rmtree(target_root)
shutil.copytree(source, target_root)
PY

read -r memory_version memory_source_count < <(
  python - "${memory_manifest}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("profile") != "natural":
    raise ValueError("Leaf smoke requires the natural memory profile")
if payload.get("controlled_candidate_ids"):
    raise ValueError("Positive-control candidates leaked into natural memory")
print(payload["artifact_version"], int(payload["source_count"]))
PY
)

export MLEVOLVE_CONFIG="${source_root}/mlevolve/config/config_prevalence_audit_20260729_host_enforce.yaml"
export MLEVOLVE_HOST_PROTOCOL_BINDING="${binding}"
export MLEVOLVE_HOST_COLLECTOR_KEY_FILE=/run/host-key/collector.ed25519
export MLEVOLVE_CODE_REVISION=devpod-leaf-authority-collision-fix
export MLEVOLVE_CODE_WORKTREE_SHA256="$(
  python - "${source_root}" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
digest = hashlib.sha256()
for path in sorted(root.rglob("*")):
    if not path.is_file() or "__pycache__" in path.parts or "runs" in path.parts:
        continue
    if path.suffix not in {".py", ".yaml", ".json", ".sh"}:
        continue
    digest.update(path.relative_to(root).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)"
export HF_HOME=/tmp/huggingface-leaf-authority-fix
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export MPLCONFIGDIR=/tmp/matplotlib-leaf-authority-fix
export TOKENIZERS_PARALLELISM=false
export DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL

install -m 0400 /run/host-key-source/collector.ed25519 \
  /run/host-key/collector.ed25519
mkdir -p "${output_root}"
cd "${source_root}/mlevolve"
python -u run.py \
  exp_id="${task}" \
  exp_name="devpod-leaf-authority-collision-fix-seed-${seed}" \
  dataset_dir="${release_root}/data/${task}" \
  data_dir="${public_data}" \
  desc_file="${public_data}/description.md" \
  log_dir="${output_root}/logs" \
  workspace_dir="${output_root}/workspace" \
  agent.seed="${seed}" \
  agent.steps="${steps}" \
  agent.time_limit="${time_limit}" \
  exec.timeout=300 \
  finalize_reserve_seconds=300 \
  agent.search.num_gpus=1 \
  agent.search.parallel_search_num=1 \
  cpu_number=8 \
  external_skill_memory.graph_path="${graph}" \
  external_skill_memory.index_path="${index}" \
  agent.draft_role_policy.replay_targets_path="${replay_targets}" \
  run_identity.memory_version="${memory_version}" \
  run_identity.memory_source_count="${memory_source_count}"

python "${source_root}/experiments/prevalence_audit_20260729/validate_run_packet.py" \
  --run-root "${output_root}" \
  --task-id "${task}" \
  --agent-seed "${seed}" \
  --memory-manifest "${memory_manifest}"

python - "${output_root}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
event_paths = sorted((root / "logs").glob("*/authority_events.jsonl"))
ledger_paths = sorted((root / "logs").glob("*/prospective_decision_ledger.jsonl"))
if len(event_paths) != 1 or len(ledger_paths) != 1:
    raise ValueError("Leaf smoke did not produce one authoritative log directory")
events = [
    json.loads(line)
    for line in event_paths[0].read_text(encoding="utf-8").splitlines()
]
ledger = [
    json.loads(line)
    for line in ledger_paths[0].read_text(encoding="utf-8").splitlines()
]
gate = json.loads((root / "ONLINE_PACKET_GATE.json").read_text(encoding="utf-8"))
report = {
    "schema": "mlevolve_leaf_authority_collision_smoke_v1",
    "status": "pass",
    "authority_internal_error_count": sum(
        event.get("event_type") == "authority_internal_error" for event in events
    ),
    "pending_counterfactual_count": sum(
        row.get("counterfactual_status") == "pending" for row in ledger
    ),
    "packet_status": gate.get("status"),
    "decision_count": gate.get("decision_count"),
    "raw_logging_coverage": gate.get("raw_logging_coverage"),
}
if report["authority_internal_error_count"] != 0:
    raise ValueError("Leaf smoke still contains Authority internal errors")
if report["pending_counterfactual_count"] != 0:
    raise ValueError("Leaf smoke contains pending counterfactuals")
if report["packet_status"] != "verified":
    raise ValueError("Leaf smoke packet is not verified")
(root / "LEAF_AUTHORITY_COLLISION_SMOKE.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY
