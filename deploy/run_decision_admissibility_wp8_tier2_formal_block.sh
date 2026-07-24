#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLOCK_ID="${1:?usage: $0 BLOCK_ID [CONTROLLER_POD]}"
CONTROLLER="${2:-da-wp8-f-controller-cpu-r3}"
NAMESPACE="${WP8_FORMAL_NAMESPACE:-ecepxie}"
GATE_ROOT="${WP8_FORMAL_LOCAL_GATE_ROOT:-$ROOT/coordination/decision_admissibility_wp8_tier2_formal_staging_stop_gate_20260723_r10}"
GATE="$GATE_ROOT/STAGING_STOP_GATE.json"
CONTENT="$GATE_ROOT/STAGING_CONTENT_MANIFEST.json"
BUILD="$GATE_ROOT/STAGING_BUILD_REPORT.json"
CONTROLLER_YAML="${WP8_FORMAL_CONTROLLER_YAML:-$GATE_ROOT/formal-controller-cpu-r3.yaml}"

test -f "$GATE"
test -f "$CONTENT"
test -f "$BUILD"
test -f "$CONTROLLER_YAML"
test "$CONTROLLER" != "jupyter-a10-d48dfd589-pqfkb"

kubectl_clean() {
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u NO_PROXY \
      -u https_proxy -u http_proxy -u all_proxy -u no_proxy \
      kubectl "$@"
}

kubectl_get() {
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

IFS=$'\t' read -r GATE_HASH CONTENT_HASH IMAGE_DIGEST TRAINING_POD \
  EVALUATOR_POD TRAINING_YAML_SHA EVALUATOR_YAML_SHA <<< "$(python - "$GATE" "$CONTENT" "$BUILD" "$BLOCK_ID" "$CONTROLLER_YAML" "$0" <<'PY'
import hashlib,json,sys
from pathlib import Path
gate_path,content_path,build_path=map(Path,sys.argv[1:4]);block_id=sys.argv[4];controller_path=Path(sys.argv[5]);launcher_path=Path(sys.argv[6]).resolve()
gate=json.loads(gate_path.read_text());content=json.loads(content_path.read_text());build=json.loads(build_path.read_text())
def h(p,field):
    unsigned={k:v for k,v in p.items() if k!=field}
    return hashlib.sha256(json.dumps(unsigned,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
assert gate["gate_hash"]==h(gate,"gate_hash") and gate["formal_training_authorized"] is True
assert content["manifest_hash"]==h(content,"manifest_hash")==gate["staging_content_manifest_hash"]
assert build["build_hash"]==h(build,"build_hash")
assert block_id in content["blocks_by_id"]
block=content["blocks_by_id"][block_id]
assert hashlib.sha256(controller_path.read_bytes()).hexdigest()==build["pod_yamls"]["formal-controller"]["sha256"]
assert hashlib.sha256(launcher_path.read_bytes()).hexdigest()==content["control_source_files"]["deploy/run_decision_admissibility_wp8_tier2_formal_block.sh"]
values=(gate["gate_hash"],content["manifest_hash"],content["container_image_digest"],block["training_pod_name"],block["evaluator_pod_name"],build["pod_yamls"][f"{block_id}:training"]["sha256"],build["pod_yamls"][f"{block_id}:evaluator"]["sha256"])
print("\t".join(values))
PY
)"
test "$TRAINING_POD" != "jupyter-a10-d48dfd589-pqfkb"
test "$EVALUATOR_POD" != "jupyter-a10-d48dfd589-pqfkb"

if ! kubectl_get get pod "$CONTROLLER" -n "$NAMESPACE" >/dev/null 2>&1; then
  kubectl_clean create -f "$CONTROLLER_YAML"
fi
kubectl_clean wait -n "$NAMESPACE" --for=condition=Ready "pod/$CONTROLLER" --timeout=15m
test "$(kubectl_get get pod "$CONTROLLER" -n "$NAMESPACE" -o jsonpath='{.status.phase}')" = "Running"
test "$(kubectl_get get pod "$CONTROLLER" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].imageID}')" = "$IMAGE_DIGEST"
kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; test -f /formal/staging/STAGING_CONTENT_MANIFEST.json; test -d '/formal/outputs/blocks/$BLOCK_ID'; test -z \"\$(find '/formal/outputs/blocks/$BLOCK_ID' -mindepth 1 -print -quit)\""

create_remote_pod() {
  local role="$1" pod="$2" expected_sha="$3" phase ready attempt
  local path="/formal/staging/pods/$BLOCK_ID-$role.yaml"
  CREATED_POD_UID=""
  CREATED_POD_FAILURE_PHASE=""
  test -z "$(kubectl_get get pod "$pod" -n "$NAMESPACE" --ignore-not-found -o name)"
  test "$(kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- sha256sum "$path" | awk '{print $1}')" = "$expected_sha"
  kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- cat "$path" | kubectl_clean create -f -
  CREATED_POD_UID="$(kubectl_get get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.metadata.uid}')"
  for attempt in $(seq 1 120); do
    IFS=$'\t' read -r phase ready <<< "$(kubectl_get get pod "$pod" -n "$NAMESPACE" --ignore-not-found -o jsonpath='{.status.phase}{"\t"}{.status.containerStatuses[0].ready}')"
    if [[ -z "$phase" ]]; then
      CREATED_POD_FAILURE_PHASE="NotFound"
      return 42
    fi
    if [[ "$phase" == "Running" && "$ready" == "true" ]]; then
      break
    fi
    if [[ "$phase" == "Failed" || "$phase" == "Succeeded" ]]; then
      CREATED_POD_FAILURE_PHASE="$phase"
      return 43
    fi
    sleep 10
  done
  if [[ "$phase" != "Running" || "$ready" != "true" ]]; then
    CREATED_POD_FAILURE_PHASE="${phase:-NotFound}"
    return 44
  fi
  test "$(kubectl_get get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.phase}')" = "Running"
  test "$(kubectl_get get pod "$pod" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].imageID}')" = "$IMAGE_DIGEST"
}

poll_launcher() {
  local pod="$1" marker="$2" phase
  POLL_FAILURE_PHASE=""
  while ! kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- test -f "/formal/outputs/blocks/$BLOCK_ID/$marker"; do
    if ! phase="$(kubectl_get get pod "$pod" -n "$NAMESPACE" --ignore-not-found -o jsonpath='{.status.phase}')"; then
      POLL_FAILURE_PHASE="kubernetes_query_failed"
      return 70
    fi
    if [[ -z "$phase" ]]; then
      POLL_FAILURE_PHASE="NotFound"
      return 42
    fi
    if [[ "$phase" != "Running" ]]; then
      POLL_FAILURE_PHASE="$phase"
      return 43
    fi
    sleep 20
  done
}

verify_delete_not_found() {
  local pod="$1" output rc
  set +e
  output="$(kubectl_get get pod "$pod" -n "$NAMESPACE" 2>&1)"
  rc=$?
  set -e
  test "$rc" -ne 0
  [[ "$output" == *"NotFound"* ]]
  printf '%s' "$output" | shasum -a 256 | awk '{print $1}'
}

record_training_prelaunch_loss() {
  local phase="$1" detected_at status_json status_sha event_json event_sha
  local event_reasons_json scheduled_node start_reason start_exit failure_message
  local not_found_sha not_found_at evaluator_not_found_sha
  detected_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  status_json="$(kubectl_get get pod "$TRAINING_POD" -n "$NAMESPACE" --ignore-not-found -o json)"
  status_sha="$(printf '%s' "$status_json" | shasum -a 256 | awk '{print $1}')"
  if [[ -n "$status_json" ]]; then
    scheduled_node="$(printf '%s' "$status_json" | python -c 'import json,sys; print(json.load(sys.stdin).get("spec",{}).get("nodeName","") or "")')"
    start_reason="$(printf '%s' "$status_json" | python -c 'import json,sys; p=json.load(sys.stdin); s=(p.get("status",{}).get("containerStatuses") or [{}])[0].get("state",{}).get("terminated",{}); print(s.get("reason","") or "")')"
    start_exit="$(printf '%s' "$status_json" | python -c 'import json,sys; p=json.load(sys.stdin); s=(p.get("status",{}).get("containerStatuses") or [{}])[0].get("state",{}).get("terminated",{}); print(int(s.get("exitCode",-1)))')"
    failure_message="$(printf '%s' "$status_json" | python -c 'import json,sys; p=json.load(sys.stdin); s=(p.get("status",{}).get("containerStatuses") or [{}])[0].get("state",{}).get("terminated",{}); print(s.get("message","") or "container did not become Ready")')"
  else
    scheduled_node=""
    start_reason="$phase"
    start_exit=-1
    failure_message="training Pod disappeared before becoming Ready"
  fi
  event_json="$(kubectl_get get events -n "$NAMESPACE" \
    --field-selector "involvedObject.uid=$CREATED_POD_UID" -o json)"
  event_sha="$(printf '%s' "$event_json" | shasum -a 256 | awk '{print $1}')"
  event_reasons_json="$(printf '%s' "$event_json" | python -c 'import json,sys; payload=json.load(sys.stdin); print(json.dumps(sorted({str(row.get("reason")) for row in payload.get("items", []) if row.get("reason")}),separators=(",",":")))')"
  if [[ -n "$status_json" ]]; then
    kubectl_clean delete pod "$TRAINING_POD" -n "$NAMESPACE" --wait=true --timeout=10m
  fi
  not_found_sha="$(verify_delete_not_found "$TRAINING_POD")"
  not_found_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  evaluator_not_found_sha="$(verify_delete_not_found "$EVALUATOR_POD")"
  kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- \
    env PYTHONPATH=/opt/nautilus/mlevolve:/opt/nautilus \
    python -m fixed_holdout.formal_host_receipts \
    training-prelaunch-abort \
    --output-root /formal/outputs \
    --contract-root "/formal/staging/blocks/$BLOCK_ID" \
    --namespace "$NAMESPACE" --pod-name "$TRAINING_POD" \
    --pod-uid "$CREATED_POD_UID" --staging-gate-hash "$GATE_HASH" \
    --staging-content-manifest-hash "$CONTENT_HASH" \
    --detected-at "$detected_at" --event-snapshot-sha256 "$event_sha" \
    --event-reasons-json "$event_reasons_json" \
    --pod-status-snapshot-sha256 "$status_sha" \
    --scheduled-node "$scheduled_node" \
    --container-start-reason "$start_reason" \
    --container-start-exit-code "$start_exit" \
    --failure-message "$failure_message" \
    --not-found-verified-at "$not_found_at" \
    --not-found-probe-sha256 "$not_found_sha" \
    --evaluator-not-found-probe-sha256 "$evaluator_not_found_sha"
  kubectl_clean delete pod "$CONTROLLER" -n "$NAMESPACE" \
    --wait=true --timeout=10m || true
  printf 'Formal block %s was sealed as a prelaunch infrastructure abort (phase=%s).\n' \
    "$BLOCK_ID" "$phase" >&2
}

record_training_pod_loss() {
  local phase="$1" detected_at event_json event_sha event_reasons_json
  local pod_status_json="" pod_status_sha="" not_found_sha not_found_at
  local evaluator_not_found_sha
  case "$phase" in
    NotFound|Failed|Succeeded) ;;
    *)
      printf 'Refusing destructive Pod-loss handling for non-terminal phase value: %q\n' \
        "$phase" >&2
      return 74
      ;;
  esac
  detected_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  event_json="$(kubectl_get get events -n "$NAMESPACE" \
    --field-selector "involvedObject.uid=$TRAINING_UID" -o json)"
  event_sha="$(printf '%s' "$event_json" | shasum -a 256 | awk '{print $1}')"
  event_reasons_json="$(printf '%s' "$event_json" | python -c 'import json,sys; payload=json.load(sys.stdin); print(json.dumps(sorted({str(row.get("reason")) for row in payload.get("items", []) if row.get("reason")} ),separators=(",",":")))')"
  if [[ "$phase" != "NotFound" ]]; then
    pod_status_json="$(kubectl_get get pod "$TRAINING_POD" -n "$NAMESPACE" -o json)"
    pod_status_sha="$(printf '%s' "$pod_status_json" | shasum -a 256 | awk '{print $1}')"
    kubectl_clean delete pod "$TRAINING_POD" -n "$NAMESPACE" --wait=true --timeout=10m
  fi
  not_found_sha="$(verify_delete_not_found "$TRAINING_POD")"
  not_found_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  evaluator_not_found_sha="$(verify_delete_not_found "$EVALUATOR_POD")"
  kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- \
    env PYTHONPATH=/opt/nautilus/mlevolve:/opt/nautilus \
    python -m fixed_holdout.formal_host_receipts \
    training-infrastructure-abort \
    --block-root "/formal/outputs/blocks/$BLOCK_ID" \
    --namespace "$NAMESPACE" --pod-name "$TRAINING_POD" \
    --pod-uid "$TRAINING_UID" --staging-gate-hash "$GATE_HASH" \
    --detected-at "$detected_at" --not-found-verified-at "$not_found_at" \
    --not-found-probe-sha256 "$not_found_sha" \
    --evaluator-not-found-probe-sha256 "$evaluator_not_found_sha" \
    --event-snapshot-sha256 "$event_sha" \
    --event-reasons-json "$event_reasons_json" \
    --failure-phase "$phase" \
    --pod-status-snapshot-sha256 "$pod_status_sha"
  kubectl_clean delete pod "$CONTROLLER" -n "$NAMESPACE" \
    --wait=true --timeout=10m || true
  printf 'Formal block %s was sealed as a pre-terminal infrastructure abort (training phase=%s).\n' \
    "$BLOCK_ID" "$phase" >&2
}

record_training_launcher_failure() {
  local detected_at status_json status_sha scheduled_node event_json event_sha
  local event_reasons_json not_found_sha not_found_at evaluator_not_found_sha
  detected_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  status_json="$(kubectl_get get pod "$TRAINING_POD" -n "$NAMESPACE" -o json)"
  status_sha="$(printf '%s' "$status_json" | shasum -a 256 | awk '{print $1}')"
  scheduled_node="$(printf '%s' "$status_json" | python -c 'import json,sys; print(json.load(sys.stdin).get("spec",{}).get("nodeName","") or "")')"
  event_json="$(kubectl_get get events -n "$NAMESPACE" \
    --field-selector "involvedObject.uid=$TRAINING_UID" -o json)"
  event_sha="$(printf '%s' "$event_json" | shasum -a 256 | awk '{print $1}')"
  event_reasons_json="$(printf '%s' "$event_json" | python -c 'import json,sys; payload=json.load(sys.stdin); print(json.dumps(sorted({str(row.get("reason")) for row in payload.get("items", []) if row.get("reason")}),separators=(",",":")))')"
  kubectl_clean delete pod "$TRAINING_POD" -n "$NAMESPACE" \
    --wait=true --timeout=10m
  not_found_sha="$(verify_delete_not_found "$TRAINING_POD")"
  not_found_at="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  evaluator_not_found_sha="$(verify_delete_not_found "$EVALUATOR_POD")"
  if kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- \
    test -f "/formal/outputs/blocks/$BLOCK_ID/BLOCK_CONTRACT.json"; then
    kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- \
      env PYTHONPATH=/opt/nautilus/mlevolve:/opt/nautilus \
      python -m fixed_holdout.formal_host_receipts \
      training-infrastructure-abort \
      --block-root "/formal/outputs/blocks/$BLOCK_ID" \
      --namespace "$NAMESPACE" --pod-name "$TRAINING_POD" \
      --pod-uid "$TRAINING_UID" --staging-gate-hash "$GATE_HASH" \
      --detected-at "$detected_at" --not-found-verified-at "$not_found_at" \
      --not-found-probe-sha256 "$not_found_sha" \
      --evaluator-not-found-probe-sha256 "$evaluator_not_found_sha" \
      --event-snapshot-sha256 "$event_sha" \
      --event-reasons-json "$event_reasons_json" \
      --failure-phase training_launcher_nonzero \
      --pod-status-snapshot-sha256 "$status_sha"
  else
    kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- \
      env PYTHONPATH=/opt/nautilus/mlevolve:/opt/nautilus \
      python -m fixed_holdout.formal_host_receipts \
      training-precontract-abort \
      --output-root "/formal/outputs/blocks/$BLOCK_ID" \
      --contract-root "/formal/staging/blocks/$BLOCK_ID" \
      --namespace "$NAMESPACE" --pod-name "$TRAINING_POD" \
      --pod-uid "$TRAINING_UID" --staging-gate-hash "$GATE_HASH" \
      --staging-content-manifest-hash "$CONTENT_HASH" \
      --detected-at "$detected_at" --event-snapshot-sha256 "$event_sha" \
      --event-reasons-json "$event_reasons_json" \
      --pod-status-snapshot-sha256 "$status_sha" \
      --scheduled-node "$scheduled_node" \
      --not-found-verified-at "$not_found_at" \
      --not-found-probe-sha256 "$not_found_sha" \
      --evaluator-not-found-probe-sha256 "$evaluator_not_found_sha"
  fi
  kubectl_clean delete pod "$CONTROLLER" -n "$NAMESPACE" \
    --wait=true --timeout=10m || true
  printf 'Formal block %s was sealed after a nonzero training launcher exit.\n' \
    "$BLOCK_ID" >&2
}

set +e
create_remote_pod training "$TRAINING_POD" "$TRAINING_YAML_SHA"
TRAINING_CREATE_RC=$?
set -e
if [[ "$TRAINING_CREATE_RC" -ne 0 ]]; then
  if [[ "$TRAINING_CREATE_RC" -eq 42 \
     || "$TRAINING_CREATE_RC" -eq 43 \
     || "$TRAINING_CREATE_RC" -eq 44 ]]; then
    record_training_prelaunch_loss "$CREATED_POD_FAILURE_PHASE"
    exit 75
  fi
  exit "$TRAINING_CREATE_RC"
fi
TRAINING_UID="$CREATED_POD_UID"
TRAINING_IMAGE_ID="$(kubectl_get get pod "$TRAINING_POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].imageID}')"
kubectl_clean exec "$TRAINING_POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; nohup env WP8_FORMAL_STAGING_GATE_HASH='$GATE_HASH' WP8_FORMAL_ACTUAL_IMAGE_ID='$TRAINING_IMAGE_ID' bash /opt/nautilus/deploy/run_decision_admissibility_wp8_tier2_formal_training_devpod.sh > /work/formal-training-host.log 2>&1 < /dev/null & echo \$! > /work/formal-training-host.pid"
set +e
poll_launcher "$TRAINING_POD" TRAINING_LAUNCHER_EXIT_CODE
TRAINING_POLL_RC=$?
set -e
if [[ "$TRAINING_POLL_RC" -ne 0 ]]; then
  if [[ "$TRAINING_POLL_RC" -eq 42 || "$TRAINING_POLL_RC" -eq 43 ]]; then
    record_training_pod_loss "$POLL_FAILURE_PHASE"
    exit 75
  fi
  printf 'Unable to determine formal training Pod state after retries; preserving live state.\n' >&2
  exit "$TRAINING_POLL_RC"
fi
TRAINING_RC="$(kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- cat "/formal/outputs/blocks/$BLOCK_ID/TRAINING_LAUNCHER_EXIT_CODE")"
if [[ "$TRAINING_RC" != "0" ]]; then
  record_training_launcher_failure
  exit 75
fi

DELETE_REQUESTED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
kubectl_clean delete pod "$TRAINING_POD" -n "$NAMESPACE" --wait=true --timeout=10m
NOT_FOUND_SHA="$(verify_delete_not_found "$TRAINING_POD")"
NOT_FOUND_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- env PYTHONPATH=/opt/nautilus/mlevolve:/opt/nautilus \
  python -m fixed_holdout.formal_host_receipts training-deletion \
  --block-root "/formal/outputs/blocks/$BLOCK_ID" --namespace "$NAMESPACE" \
  --pod-name "$TRAINING_POD" --pod-uid "$TRAINING_UID" \
  --delete-requested-at "$DELETE_REQUESTED_AT" \
  --not-found-verified-at "$NOT_FOUND_AT" \
  --not-found-probe-sha256 "$NOT_FOUND_SHA" --staging-gate-hash "$GATE_HASH"

sleep 2
create_remote_pod evaluator "$EVALUATOR_POD" "$EVALUATOR_YAML_SHA"
EVALUATOR_UID="$(kubectl_get get pod "$EVALUATOR_POD" -n "$NAMESPACE" -o jsonpath='{.metadata.uid}')"
EVALUATOR_CREATED="$(kubectl_get get pod "$EVALUATOR_POD" -n "$NAMESPACE" -o jsonpath='{.metadata.creationTimestamp}')"
EVALUATOR_IMAGE_ID="$(kubectl_get get pod "$EVALUATOR_POD" -n "$NAMESPACE" -o jsonpath='{.status.containerStatuses[0].imageID}')"
kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- env PYTHONPATH=/opt/nautilus/mlevolve:/opt/nautilus \
  python -m fixed_holdout.formal_host_receipts evaluator-creation \
  --block-root "/formal/outputs/blocks/$BLOCK_ID" --namespace "$NAMESPACE" \
  --pod-name "$EVALUATOR_POD" --pod-uid "$EVALUATOR_UID" \
  --kubernetes-creation-timestamp "$EVALUATOR_CREATED" \
  --container-image-id "$EVALUATOR_IMAGE_ID" --staging-gate-hash "$GATE_HASH"

kubectl_clean exec "$EVALUATOR_POD" -n "$NAMESPACE" -- bash -lc \
  "set -euo pipefail; nohup env WP8_FORMAL_STAGING_GATE_HASH='$GATE_HASH' WP8_FORMAL_ACTUAL_IMAGE_ID='$EVALUATOR_IMAGE_ID' bash /opt/nautilus/deploy/run_decision_admissibility_wp8_tier2_formal_evaluator_devpod.sh > /work/formal-evaluator-host.log 2>&1 < /dev/null & echo \$! > /work/formal-evaluator-host.pid"
poll_launcher "$EVALUATOR_POD" EVALUATOR_LAUNCHER_EXIT_CODE
EVALUATOR_RC="$(kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- cat "/formal/outputs/blocks/$BLOCK_ID/EVALUATOR_LAUNCHER_EXIT_CODE")"
if [[ "$EVALUATOR_RC" != "0" ]]; then
  EVAL_FAILURE_DETECTED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  EVAL_FAILURE_DELETE_REQUESTED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  kubectl_clean delete pod "$EVALUATOR_POD" -n "$NAMESPACE" \
    --wait=true --timeout=10m
  EVAL_FAILURE_NOT_FOUND_SHA="$(verify_delete_not_found "$EVALUATOR_POD")"
  EVAL_FAILURE_NOT_FOUND_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- \
    env PYTHONPATH=/opt/nautilus/mlevolve:/opt/nautilus \
    python -m fixed_holdout.formal_host_receipts evaluator-failure \
    --block-root "/formal/outputs/blocks/$BLOCK_ID" \
    --namespace "$NAMESPACE" --pod-name "$EVALUATOR_POD" \
    --pod-uid "$EVALUATOR_UID" \
    --failure-detected-at "$EVAL_FAILURE_DETECTED_AT" \
    --delete-requested-at "$EVAL_FAILURE_DELETE_REQUESTED_AT" \
    --not-found-verified-at "$EVAL_FAILURE_NOT_FOUND_AT" \
    --not-found-probe-sha256 "$EVAL_FAILURE_NOT_FOUND_SHA" \
    --staging-gate-hash "$GATE_HASH"
  printf 'Formal evaluator failed for %s with rc=%s; the failed root was hash-sealed after verified Pod deletion.\n' "$BLOCK_ID" "$EVALUATOR_RC" >&2
  exit 1
fi

EVAL_DELETE_REQUESTED_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
kubectl_clean delete pod "$EVALUATOR_POD" -n "$NAMESPACE" --wait=true --timeout=10m
EVAL_NOT_FOUND_SHA="$(verify_delete_not_found "$EVALUATOR_POD")"
EVAL_NOT_FOUND_AT="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
kubectl_clean exec "$CONTROLLER" -n "$NAMESPACE" -- env PYTHONPATH=/opt/nautilus/mlevolve:/opt/nautilus \
  python -m fixed_holdout.formal_host_receipts evaluator-deletion \
  --block-root "/formal/outputs/blocks/$BLOCK_ID" --namespace "$NAMESPACE" \
  --pod-name "$EVALUATOR_POD" --pod-uid "$EVALUATOR_UID" \
  --delete-requested-at "$EVAL_DELETE_REQUESTED_AT" \
  --not-found-verified-at "$EVAL_NOT_FOUND_AT" \
  --not-found-probe-sha256 "$EVAL_NOT_FOUND_SHA" --staging-gate-hash "$GATE_HASH"

printf 'Formal block %s completed; both experiment Pods were verified NotFound.\n' "$BLOCK_ID"
