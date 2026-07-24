#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD="${1:-decision-admissibility-wp8-tier2-formal-stager-cpu-r13}"
NAMESPACE="${WP8_FORMAL_NAMESPACE:-ecepxie}"
EXPECTED_HEAD="${WP8_FORMAL_EXPECTED_HEAD:-b47dab63b7861f3ea0871094d6dd07b77e6b81a4}"
SOURCE_ROOT="${WP8_FORMAL_SOURCE_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-source-r11}"
CONTROL_ROOT="${WP8_FORMAL_CONTROL_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-control-r11}"
ARTIFACT_ROOT="${WP8_FORMAL_ARTIFACT_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r2}"
PARENT_STAGING_ROOT="${WP8_FORMAL_PARENT_STAGING_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r12}"
PARENT_GATE="${WP8_FORMAL_PARENT_GATE:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r12-stop-gate-r1/STAGING_STOP_GATE.json}"
COMPLETED_OUTPUT_ROOT="${WP8_FORMAL_COMPLETED_OUTPUT_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r10}"
STAGING_ROOT="${WP8_FORMAL_STAGING_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r13}"
OUTPUT_ROOT="${WP8_FORMAL_OUTPUT_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-runs-r11}"
GATE_ROOT="${WP8_FORMAL_GATE_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r13-stop-gate-r1}"
PIPELINE_ROOT="${WP8_FORMAL_PIPELINE_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r13-pipeline-r1}"
LOCAL_GATE_ROOT="${WP8_FORMAL_LOCAL_GATE_ROOT:-$ROOT/coordination/decision_admissibility_wp8_tier2_formal_continuation_stop_gate_20260723_r1}"

cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{u}')" = "$EXPECTED_HEAD"
test ! -e "$LOCAL_GATE_ROOT"
test "$POD" != "jupyter-a10-d48dfd589-pqfkb"

kubectl_clean() {
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u NO_PROXY \
      -u https_proxy -u http_proxy -u all_proxy -u no_proxy \
      kubectl "$@"
}

kubectl_retry() {
  local attempt rc
  for attempt in 1 2 3 4 5; do
    set +e
    kubectl_clean "$@"
    rc=$?
    set -e
    [[ "$rc" -eq 0 ]] && return 0
    sleep $((attempt * 2))
  done
  return "$rc"
}

upload_archive() {
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

test "$(kubectl_clean get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}')" = "Running"
kubectl_clean exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; test -d '$ARTIFACT_ROOT'; test -d '$PARENT_STAGING_ROOT'; test -f '$PARENT_GATE'; test -d '$COMPLETED_OUTPUT_ROOT'; test ! -e '$SOURCE_ROOT'; test ! -e '$CONTROL_ROOT'; test ! -e '$STAGING_ROOT'; test ! -e '$OUTPUT_ROOT'; test ! -e '$GATE_ROOT'; test ! -e '$PIPELINE_ROOT'; mkdir -p '$SOURCE_ROOT' '$CONTROL_ROOT'"

git archive --format=tar.gz "$EXPECTED_HEAD" -- \
  mlevolve/__init__.py mlevolve/run.py \
  mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
  mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils \
  ':(exclude)**/__pycache__' | \
  kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
    tar -xzf - -C "$SOURCE_ROOT"

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
upload_archive "$runtime_list" "$SOURCE_ROOT"
kubectl_cp_retry "$runtime_list" \
  "$NAMESPACE/$POD:$SOURCE_ROOT/WP8_TIER2_OVERLAY_PATHS.txt"

kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$SOURCE_ROOT/mlevolve:$SOURCE_ROOT" \
  python - "$SOURCE_ROOT" "$EXPECTED_HEAD" <<'PY'
import hashlib,json,sys
from pathlib import Path
from fixed_holdout.formal_runtime import SOURCE_SCHEMA,payload_hash
root=Path(sys.argv[1]).resolve();base_commit=sys.argv[2]
excluded={"WP8_TIER2_SOURCE_MANIFEST.json"}
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file() and p.relative_to(root).as_posix() not in excluded}
overlay=[line.strip() for line in (root/"WP8_TIER2_OVERLAY_PATHS.txt").read_text().splitlines() if line.strip()]
payload={"schema":SOURCE_SCHEMA,"base_commit":base_commit,"purpose":"wp8_tier2_formal_remaining_5_blocks_r4","overlay_paths":overlay,"file_count":len(files),"file_hashes":files,"excluded_user_assets":["mlevolve/data","mlevolve/runs","mlevolve/inference","outputs","papers"],"source_sha256":""}
payload["source_sha256"]=payload_hash(payload,"source_sha256")
(root/"WP8_TIER2_SOURCE_MANIFEST.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

kubectl_clean exec "$POD" -n "$NAMESPACE" -- bash -lc \
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
    deploy/stage_decision_admissibility_wp8_tier2_formal_continuation.sh \
    deploy/run_decision_admissibility_wp8_tier2_formal_continuation_staging_pipeline.sh \
    deploy/run_decision_admissibility_wp8_tier2_formal_block.sh
} | LC_ALL=C sort -u > "$control_list"
while IFS= read -r path; do test -f "$path"; done < "$control_list"
upload_archive "$control_list" "$CONTROL_ROOT"

kubectl_clean exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='$CONTROL_ROOT/mlevolve:$CONTROL_ROOT'; cd '$CONTROL_ROOT'; python -m pytest -q -p no:cacheprovider tests/test_tier2_formal_preterminal_recovery.py tests/test_tier2_formal_preterminal_recovery_amendment.py tests/test_tier2_formal_continuation_amendment.py tests/test_tier2_formal_block_finalizers.py tests/authority/test_tier2_formal_launcher_static.py"

kubectl_clean exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; chmod -R a-w '$SOURCE_ROOT' '$CONTROL_ROOT'; mkdir '$PIPELINE_ROOT'; nohup env CONTROL_ROOT='$CONTROL_ROOT' SOURCE_ROOT='$SOURCE_ROOT' ARTIFACT_ROOT='$ARTIFACT_ROOT' PARENT_STAGING_ROOT='$PARENT_STAGING_ROOT' PARENT_GATE='$PARENT_GATE' COMPLETED_OUTPUT_ROOT='$COMPLETED_OUTPUT_ROOT' STAGING_ROOT='$STAGING_ROOT' OUTPUT_ROOT='$OUTPUT_ROOT' GATE_ROOT='$GATE_ROOT' PIPELINE_ROOT='$PIPELINE_ROOT' bash '$CONTROL_ROOT/deploy/run_decision_admissibility_wp8_tier2_formal_continuation_staging_pipeline.sh' > /dev/null 2>&1 < /dev/null &"

while ! kubectl_clean exec "$POD" -n "$NAMESPACE" -- test -f "$PIPELINE_ROOT/EXIT_CODE"; do
  test "$(kubectl_clean get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}')" = "Running"
  sleep 20
done
test "$(kubectl_clean exec "$POD" -n "$NAMESPACE" -- cat "$PIPELINE_ROOT/EXIT_CODE")" = "0"

mkdir "$LOCAL_GATE_ROOT"
for file in STAGING_STOP_GATE.json; do
  kubectl_cp_retry "$NAMESPACE/$POD:$GATE_ROOT/$file" "$LOCAL_GATE_ROOT/$file"
done
for file in STAGING_CONTENT_MANIFEST.json STAGING_BUILD_REPORT.json; do
  kubectl_cp_retry "$NAMESPACE/$POD:$STAGING_ROOT/$file" "$LOCAL_GATE_ROOT/$file"
done
kubectl_cp_retry "$NAMESPACE/$POD:$STAGING_ROOT/pods/formal-controller-cpu-r4.yaml" \
  "$LOCAL_GATE_ROOT/formal-controller-cpu-r4.yaml"

python - "$LOCAL_GATE_ROOT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);gate=json.loads((root/"STAGING_STOP_GATE.json").read_text())
assert gate["formal_training_authorized"] is True and gate["errors"]==[]
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()}
payload={"schema":"decision_admissibility_wp8_tier2_formal_continuation_local_gate_packet_v1","file_hashes":files,"gate_hash":gate["gate_hash"],"packet_hash":""}
unsigned={k:v for k,v in payload.items() if k!="packet_hash"};payload["packet_hash"]=hashlib.sha256(json.dumps(unsigned,sort_keys=True,separators=(",",":")).encode()).hexdigest()
(root/"PACKET_MANIFEST.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

printf 'Formal continuation staging Stop Gate passed; five remaining blocks are authorized.\n'
