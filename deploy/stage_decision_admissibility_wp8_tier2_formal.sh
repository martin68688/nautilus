#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD="${1:-decision-admissibility-wp8-tier2-formal-stager-cpu-r11}"
NAMESPACE="${WP8_FORMAL_NAMESPACE:-ecepxie}"
EXPECTED_HEAD="${WP8_FORMAL_EXPECTED_HEAD:-b47dab63b7861f3ea0871094d6dd07b77e6b81a4}"
SOURCE_ROOT="${WP8_FORMAL_SOURCE_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-source-r10}"
CONTROL_ROOT="${WP8_FORMAL_CONTROL_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-control-r10}"
ARTIFACT_ROOT="${WP8_FORMAL_ARTIFACT_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r2}"
STAGING_ROOT="${WP8_FORMAL_STAGING_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r12}"
OUTPUT_ROOT="${WP8_FORMAL_OUTPUT_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r10}"
GATE_ROOT="${WP8_FORMAL_GATE_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r12-stop-gate-r1}"
PIPELINE_ROOT="${WP8_FORMAL_PIPELINE_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r12-pipeline-r1}"
LOCAL_GATE_ROOT="${WP8_FORMAL_LOCAL_GATE_ROOT:-$ROOT/coordination/decision_admissibility_wp8_tier2_formal_staging_stop_gate_20260723_r10}"
SUPERSEDED_STAGING_ABORT="${WP8_FORMAL_SUPERSEDED_STAGING_ABORT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r3-abort-report.json}"
SUPERSEDED_COMPATIBILITY_ABORT="${WP8_FORMAL_SUPERSEDED_COMPATIBILITY_ABORT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r2/blocks/wp8-tier2-formal-aerial-seed-104729-r1/FORMAL_BLOCK_INFRASTRUCTURE_ABORT.json}"
SUPERSEDED_NODE_EVICTION_ABORT="${WP8_FORMAL_SUPERSEDED_NODE_EVICTION_ABORT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r3/blocks/wp8-tier2-formal-aerial-seed-104729-r1/FORMAL_BLOCK_INFRASTRUCTURE_ABORT.json}"
SUPERSEDED_RANKING_DIAGNOSTIC="${WP8_FORMAL_SUPERSEDED_RANKING_DIAGNOSTIC:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r3/FORMAL_PRETERMINAL_RANKING_DIAGNOSTIC.json}"
SUPERSEDED_PRELAUNCH_ABORT="${WP8_FORMAL_SUPERSEDED_PRELAUNCH_ABORT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r4/FORMAL_PRELAUNCH_INFRASTRUCTURE_ABORT.json}"
SUPERSEDED_STAGING_TRANSFER_ABORT="${WP8_FORMAL_SUPERSEDED_STAGING_TRANSFER_ABORT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r7-abort-report.json}"
SUPERSEDED_SECOND_PRELAUNCH_ABORT="${WP8_FORMAL_SUPERSEDED_SECOND_PRELAUNCH_ABORT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r6/FORMAL_PRELAUNCH_INFRASTRUCTURE_ABORT.json}"
SUPERSEDED_PHASE_POLLUTION_ABORT="${WP8_FORMAL_SUPERSEDED_PHASE_POLLUTION_ABORT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r7/blocks/wp8-tier2-formal-aerial-seed-104729-r1/FORMAL_BLOCK_INFRASTRUCTURE_ABORT.json}"
SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC="${WP8_FORMAL_SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r7/FORMAL_HOST_PHASE_POLLUTION_DIAGNOSTIC.json}"
SUPERSEDED_R2_SCHEDULING_ABORT="${WP8_FORMAL_SUPERSEDED_R2_SCHEDULING_ABORT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r9/FORMAL_PRELAUNCH_INFRASTRUCTURE_ABORT.json}"

cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{u}')" = "$EXPECTED_HEAD"
test ! -e "$LOCAL_GATE_ROOT"

kubectl_clean() {
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u NO_PROXY \
      -u https_proxy -u http_proxy -u all_proxy -u no_proxy \
      kubectl "$@"
}

kubectl_retry() {
  local attempt output error_output combined rc error_file
  for attempt in 1 2 3 4 5; do
    error_file="$(mktemp)"
    if output="$(kubectl_clean "$@" 2>"$error_file")"; then
      error_output="$(cat "$error_file")"
      rm -f "$error_file"
      [[ -z "$error_output" ]] || printf '%s\n' "$error_output" >&2
      printf '%s' "$output"
      return 0
    else
      rc=$?
    fi
    error_output="$(cat "$error_file")"
    rm -f "$error_file"
    combined="$output"$'\n'"$error_output"
    if [[ "$combined" == *"TLS handshake timeout"* \
       || "$combined" == *"context deadline exceeded"* \
       || "$combined" == *"unexpected EOF"* \
       || "$combined" == *"websocket: close 1006"* \
       || "$combined" == *"error from a previous attempt: EOF"* ]]; then
      sleep $((attempt * 2))
      continue
    fi
    printf '%s\n' "$combined" >&2
    return "$rc"
  done
  printf '%s\n' "$combined" >&2
  return "$rc"
}

upload_base_source() {
  local attempt rc
  for attempt in 1 2 3 4 5; do
    set +e
    git archive --format=tar.gz "$EXPECTED_HEAD" -- \
      mlevolve/__init__.py mlevolve/run.py \
      mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
      mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils \
      ':(exclude)**/__pycache__' | \
      kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
        tar -xzf - -C "$SOURCE_ROOT"
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] && return 0
    sleep $((attempt * 2))
  done
  return "$rc"
}

upload_list_archive() {
  local list_path="$1" target_root="$2" attempt rc
  for attempt in 1 2 3 4 5; do
    set +e
    COPYFILE_DISABLE=1 tar --no-xattrs -czf - -T "$list_path" | \
      kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
        tar -xzf - -C "$target_root"
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] && return 0
    sleep $((attempt * 2))
  done
  return "$rc"
}

kubectl_cp_retry() {
  local attempt rc
  for attempt in 1 2 3 4 5; do
    set +e
    kubectl_clean cp "$@"
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] && return 0
    sleep $((attempt * 2))
  done
  return "$rc"
}

test "$(kubectl_retry get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}')" = "Running"
test "$POD" != "jupyter-a10-d48dfd589-pqfkb"
kubectl_retry exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; test -f '$SUPERSEDED_STAGING_ABORT'; test -f '$SUPERSEDED_COMPATIBILITY_ABORT'; test -f '$SUPERSEDED_NODE_EVICTION_ABORT'; test -f '$SUPERSEDED_RANKING_DIAGNOSTIC'; test -f '$SUPERSEDED_PRELAUNCH_ABORT'; test -f '$SUPERSEDED_STAGING_TRANSFER_ABORT'; test -f '$SUPERSEDED_SECOND_PRELAUNCH_ABORT'; test -f '$SUPERSEDED_PHASE_POLLUTION_ABORT'; test -f '$SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC'; test -f '$SUPERSEDED_R2_SCHEDULING_ABORT'; test ! -e '$SOURCE_ROOT'; test ! -e '$CONTROL_ROOT'; test ! -e '$STAGING_ROOT'; test ! -e '$OUTPUT_ROOT'; test ! -e '$GATE_ROOT'; test ! -e '$PIPELINE_ROOT'"
kubectl_retry exec "$POD" -n "$NAMESPACE" -- \
  mkdir -p "$SOURCE_ROOT" "$CONTROL_ROOT"

upload_base_source

runtime_list="$(mktemp)"
control_list="$(mktemp)"
trap 'rm -f "$runtime_list" "$control_list"' EXIT
{
  git diff HEAD --name-only --diff-filter=ACMRTUXB -- \
    mlevolve/__init__.py mlevolve/run.py \
    mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
    mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils
  git ls-files --others --exclude-standard -- \
    mlevolve/__init__.py mlevolve/run.py \
    mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
    mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils
  printf '%s\n' \
    deploy/run_decision_admissibility_wp8_tier2_formal_training_devpod.sh \
    deploy/run_decision_admissibility_wp8_tier2_formal_evaluator_devpod.sh
} | LC_ALL=C sort -u > "$runtime_list"
test -s "$runtime_list"
while IFS= read -r path; do test -f "$path"; done < "$runtime_list"
upload_list_archive "$runtime_list" "$SOURCE_ROOT"
kubectl_cp_retry "$runtime_list" \
  "$NAMESPACE/$POD:$SOURCE_ROOT/WP8_TIER2_OVERLAY_PATHS.txt"

kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE_ROOT/mlevolve:$SOURCE_ROOT" \
  python - "$SOURCE_ROOT" "$EXPECTED_HEAD" <<'PY'
import hashlib,json,sys
from pathlib import Path
from fixed_holdout.formal_runtime import SOURCE_SCHEMA, payload_hash
root=Path(sys.argv[1]).resolve(); base_commit=sys.argv[2]
excluded={"WP8_TIER2_SOURCE_MANIFEST.json"}
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded}
overlay=[line.strip() for line in (root/"WP8_TIER2_OVERLAY_PATHS.txt").read_text().splitlines() if line.strip()]
p={"schema":SOURCE_SCHEMA,"base_commit":base_commit,"purpose":"wp8_tier2_formal_3protocol_5online_system_9block_runtime","overlay_paths":overlay,"file_count":len(files),"file_hashes":files,"excluded_user_assets":["mlevolve/data","mlevolve/runs","mlevolve/inference","outputs","coordination","papers"],"source_sha256":""}
p["source_sha256"]=payload_hash(p,"source_sha256")
(root/"WP8_TIER2_SOURCE_MANIFEST.json").write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
print(json.dumps({"file_count":len(files),"source_sha256":p["source_sha256"]},sort_keys=True))
PY

kubectl_retry exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; cp -a '$SOURCE_ROOT/.' '$CONTROL_ROOT/'"

{
  find paper-skills/memory_bundle -maxdepth 1 -type f -name '*.py' -print
  find paper-skills/distillation -type f -name '*.py' -print
  find paper-skills/hyper_memory -maxdepth 1 -type f -name '*.py' -print
  find tests -type f -name '*.py' -print
  find coordination -maxdepth 1 -type f \
    \( -name 'decision_admissibility_wp8_tier2_formal_*' \
       -o -name 'decision_admissibility_complete_execution_plan_20260719.md' \) -print
  printf '%s\n' \
    paper-skills/eval_skill_memory/clean_run_allowlist.json \
    paper-skills/eval_skill_memory/run_identity_registry_v1.json \
    deploy/stage_decision_admissibility_wp8_tier2_formal.sh \
    deploy/run_decision_admissibility_wp8_tier2_formal_block.sh \
    deploy/run_decision_admissibility_wp8_tier2_formal_staging_pipeline.sh
} | LC_ALL=C sort -u > "$control_list"
test -s "$control_list"
while IFS= read -r path; do test -f "$path"; done < "$control_list"
upload_list_archive "$control_list" "$CONTROL_ROOT"

kubectl_retry exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; chmod -R a-w '$SOURCE_ROOT' '$CONTROL_ROOT'; test -z \"\$(find '$SOURCE_ROOT' -type f -perm /222 -print -quit)\"; test -z \"\$(find '$CONTROL_ROOT' -type f -perm /222 -print -quit)\""

kubectl_retry exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='$CONTROL_ROOT/mlevolve:$CONTROL_ROOT'; cd '$CONTROL_ROOT'; python -m pytest -q -p no:cacheprovider tests/authority/test_mlevolve_adapter.py tests/authority/test_tier2_formal_launcher_static.py tests/test_fixed_holdout_terminal_writeback.py tests/test_fixed_holdout_protocol_payload_closure.py tests/test_tier2_formal_block_finalizers.py tests/test_tier2_formal_preregistration.py tests/test_tier2_formal_preregistration_amendment.py tests/test_tier2_formal_claim_authority_amendment.py tests/test_tier2_formal_postfailure_amendment.py tests/test_tier2_formal_holdout_verifier.py tests/test_tier2_formal_child_bundle_publisher.py"

kubectl_clean exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; mkdir '$PIPELINE_ROOT'; nohup env WP8_FORMAL_CONTROL_ROOT='$CONTROL_ROOT' WP8_FORMAL_SOURCE_ROOT='$SOURCE_ROOT' WP8_FORMAL_ARTIFACT_ROOT='$ARTIFACT_ROOT' WP8_FORMAL_STAGING_ROOT='$STAGING_ROOT' WP8_FORMAL_OUTPUT_ROOT='$OUTPUT_ROOT' WP8_FORMAL_GATE_ROOT='$GATE_ROOT' WP8_FORMAL_PIPELINE_ROOT='$PIPELINE_ROOT' WP8_FORMAL_SUPERSEDED_STAGING_ABORT='$SUPERSEDED_STAGING_ABORT' WP8_FORMAL_SUPERSEDED_COMPATIBILITY_ABORT='$SUPERSEDED_COMPATIBILITY_ABORT' WP8_FORMAL_SUPERSEDED_NODE_EVICTION_ABORT='$SUPERSEDED_NODE_EVICTION_ABORT' WP8_FORMAL_SUPERSEDED_RANKING_DIAGNOSTIC='$SUPERSEDED_RANKING_DIAGNOSTIC' WP8_FORMAL_SUPERSEDED_PRELAUNCH_ABORT='$SUPERSEDED_PRELAUNCH_ABORT' WP8_FORMAL_SUPERSEDED_STAGING_TRANSFER_ABORT='$SUPERSEDED_STAGING_TRANSFER_ABORT' WP8_FORMAL_SUPERSEDED_SECOND_PRELAUNCH_ABORT='$SUPERSEDED_SECOND_PRELAUNCH_ABORT' WP8_FORMAL_SUPERSEDED_PHASE_POLLUTION_ABORT='$SUPERSEDED_PHASE_POLLUTION_ABORT' WP8_FORMAL_SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC='$SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC' WP8_FORMAL_SUPERSEDED_R2_SCHEDULING_ABORT='$SUPERSEDED_R2_SCHEDULING_ABORT' bash '$CONTROL_ROOT/deploy/run_decision_admissibility_wp8_tier2_formal_staging_pipeline.sh' > /dev/null 2>&1 < /dev/null &"

while ! kubectl_retry exec "$POD" -n "$NAMESPACE" -- test -f "$PIPELINE_ROOT/EXIT_CODE"; do
  test "$(kubectl_retry get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}')" = "Running"
  sleep 20
done
PIPELINE_RC="$(kubectl_retry exec "$POD" -n "$NAMESPACE" -- cat "$PIPELINE_ROOT/EXIT_CODE")"
if [[ "$PIPELINE_RC" != "0" ]]; then
  kubectl_retry exec "$POD" -n "$NAMESPACE" -- tail -n 200 "$PIPELINE_ROOT/PIPELINE.log" || true
  exit 1
fi

mkdir "$LOCAL_GATE_ROOT"
kubectl_cp_retry \
  "$NAMESPACE/$POD:$GATE_ROOT/STAGING_STOP_GATE.json" \
  "$LOCAL_GATE_ROOT/STAGING_STOP_GATE.json"
kubectl_cp_retry \
  "$NAMESPACE/$POD:$STAGING_ROOT/STAGING_CONTENT_MANIFEST.json" \
  "$LOCAL_GATE_ROOT/STAGING_CONTENT_MANIFEST.json"
kubectl_cp_retry \
  "$NAMESPACE/$POD:$STAGING_ROOT/STAGING_BUILD_REPORT.json" \
  "$LOCAL_GATE_ROOT/STAGING_BUILD_REPORT.json"
kubectl_cp_retry \
  "$NAMESPACE/$POD:$STAGING_ROOT/pods/formal-controller-cpu-r3.yaml" \
  "$LOCAL_GATE_ROOT/formal-controller-cpu-r3.yaml"

python - "$LOCAL_GATE_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1])
gate=json.loads((root/"STAGING_STOP_GATE.json").read_text())
assert gate["formal_training_authorized"] is True, gate["errors"]
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()}
p={"schema":"decision_admissibility_wp8_tier2_formal_local_gate_packet_v1","file_hashes":files,"gate_hash":gate["gate_hash"],"packet_hash":""}
u={k:v for k,v in p.items() if k!="packet_hash"}
p["packet_hash"]=hashlib.sha256(json.dumps(u,sort_keys=True,separators=(",",":")).encode()).hexdigest()
(root/"PACKET_MANIFEST.json").write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY

printf 'Formal staging Stop Gate passed; formal GPU execution remains not started.\n'
