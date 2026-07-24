#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC="${WP8_FORMAL_SOURCE_ROOT:-/opt/nautilus}"
FINALIZER_SRC="${WP8_FORMAL_FINALIZER_SOURCE_ROOT:-$SRC}"
TRAIN_VIEW="${WP8_FORMAL_TRAIN_VIEW_ROOT:-/fixed/train_view}"
EVALUATOR_VIEW="${WP8_FORMAL_EVALUATOR_VIEW_ROOT:-/fixed/evaluator_view}"
CONTRACT_ROOT="${WP8_FORMAL_CONTRACT_ROOT:-/contract}"
OUTPUT="${WP8_FORMAL_OUTPUT_ROOT:-/output}"
TEMPLATE="$CONTRACT_ROOT/BLOCK_TEMPLATE.json"
CONTENT_MANIFEST="$CONTRACT_ROOT/STAGING_CONTENT_MANIFEST.json"
DELETION_ATTESTATION="$OUTPUT/TRAINING_POD_DELETION_ATTESTATION.json"
CREATION_ATTESTATION="$OUTPUT/EVALUATOR_POD_CREATION_ATTESTATION.json"
RECOVERY_DELETION_ATTESTATION="$OUTPUT/RECOVERY_POD_DELETION_ATTESTATION.json"
EXPECTED_CONTENT_HASH="${WP8_FORMAL_STAGING_CONTENT_HASH:?required}"
EXPECTED_GATE_HASH="${WP8_FORMAL_STAGING_GATE_HASH:?required}"
ACTUAL_IMAGE_ID="${WP8_FORMAL_ACTUAL_IMAGE_ID:?required}"
POD_NAME="${POD_NAME:?required}"
POD_NAMESPACE="${POD_NAMESPACE:?required}"
POD_UID="${POD_UID:?required}"

test -d "$SRC/mlevolve"
test -d "$FINALIZER_SRC/mlevolve"
test -d "$TRAIN_VIEW/input"
test -d "$EVALUATOR_VIEW"
test -f "$TEMPLATE"
test -f "$CONTENT_MANIFEST"
test -f "$OUTPUT/BLOCK_CONTRACT.json"
test -f "$OUTPUT/TRAINING_MANIFEST.json"
test -f "$OUTPUT/TRAINING_COMPLETE"
test -f "$DELETION_ATTESTATION"
test -f "$CREATION_ATTESTATION"
RECOVERY_REQUIRED=false
if [[ "$FINALIZER_SRC" != "$SRC" ]]; then
  RECOVERY_REQUIRED=true
  test -f "$RECOVERY_DELETION_ATTESTATION"
fi
test ! -e "$OUTPUT/EVALUATION_SUMMARY.json"
test ! -e /workspace
test ! -e /memory
test ! -e /secrets/mlevolve.env

exec > >(tee -a "$OUTPUT/EVALUATOR_LAUNCHER.log") 2>&1
printf 'evaluator_preflight\n' > "$OUTPUT/STATE"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT/EVALUATION_STARTED_AT"

finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$OUTPUT/EVALUATOR_LAUNCHER_EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then
    printf 'evaluator_failed\n' > "$OUTPUT/STATE"
  fi
  exit "$rc"
}
terminate() {
  trap - TERM
  printf 'TERM\n' > "$OUTPUT/EVALUATOR_SIGNAL"
  exit 143
}
trap finish EXIT
trap terminate TERM

export PYTHONPATH="$FINALIZER_SRC/mlevolve:$FINALIZER_SRC:$SRC/mlevolve:$SRC"

python - "$SRC" "$FINALIZER_SRC" "$TRAIN_VIEW" "$EVALUATOR_VIEW" "$TEMPLATE" \
  "$CONTENT_MANIFEST" "$OUTPUT/BLOCK_CONTRACT.json" \
  "$OUTPUT/TRAINING_MANIFEST.json" "$DELETION_ATTESTATION" \
  "$CREATION_ATTESTATION" "$RECOVERY_DELETION_ATTESTATION" \
  "$OUTPUT/EVALUATOR_ISOLATION.json" \
  "$EXPECTED_CONTENT_HASH" "$EXPECTED_GATE_HASH" "$ACTUAL_IMAGE_ID" \
  "$RECOVERY_REQUIRED" "$POD_NAME" "$POD_NAMESPACE" "$POD_UID" <<'PY'
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from fixed_holdout.common import read_manifest, sha256_file
from fixed_holdout.formal_block_evaluate import DELETION_ATTESTATION_SCHEMA
from fixed_holdout.formal_runtime import (
    EVALUATOR_CREATION_SCHEMA,
    EVALUATOR_ISOLATION_SCHEMA,
    FORMAL_STAGING_CONTENT_SCHEMAS,
    environment_has_solver_secret,
    mount_points,
    payload_hash,
    read_object,
    validate_block_template,
    verify_source_snapshot,
)

(
    source, finalizer_source, train_root, evaluator_root, template_path,
    content_path, contract_path, training_path, deletion_path, creation_path,
    recovery_deletion_path, output_path,
) = map(Path, sys.argv[1:13])
(
    expected_content, expected_gate, image_id, recovery_required_text,
    pod_name, namespace, uid,
) = sys.argv[13:]
recovery_required = recovery_required_text == "true"
content=read_object(content_path)
assert content.get("schema") in FORMAL_STAGING_CONTENT_SCHEMAS
assert content.get("manifest_hash") == payload_hash(content,"manifest_hash") == expected_content
template=validate_block_template(read_object(template_path))
entry=(content.get("blocks_by_id") or {}).get(template["block_id"]) or {}
assert entry.get("block_template_hash") == template["template_hash"]
contract=read_object(contract_path)
training=read_object(training_path)
assert contract["contract_hash"] == payload_hash(contract,"contract_hash")
assert contract["block_template_hash"] == template["template_hash"]
assert contract["staging_manifest_hash"] == expected_content
assert contract["staging_gate_hash"] == expected_gate
assert contract["container_image_digest"] == image_id == template["container_image_digest"]
assert training["manifest_hash"] == payload_hash(training,"manifest_hash")
assert training["block_contract_hash"] == contract["contract_hash"]
assert training["block_contract_sha256"] == sha256_file(contract_path)

source_report=verify_source_snapshot(
    source,
    expected_source_sha256=contract["source_snapshot_sha256"],
    expected_manifest_file_sha256=contract["source_manifest_file_sha256"],
)
(output_path.parent/"SOURCE_EVALUATOR_PREFLIGHT.json").write_text(
    json.dumps(source_report,indent=2,sort_keys=True)+"\n"
)
train_path=train_root/"fixed_holdout_manifest.json"
evaluator_path=evaluator_root/"fixed_holdout_manifest.json"
assert sha256_file(train_path) == contract["train_manifest_sha256"]
assert sha256_file(evaluator_path) == contract["evaluator_manifest_sha256"]
train=read_manifest(train_path,expected_role="train_view")
evaluator=read_manifest(evaluator_path,expected_role="evaluator_view")
for key in ("task_id","split_id","metric","maximize","protocol_ref"):
    assert train.get(key) == evaluator.get(key) == contract.get(key), key
assert train.get("hidden_labels_present") is False

deletion=read_object(deletion_path)
assert deletion.get("schema") == DELETION_ATTESTATION_SCHEMA
assert deletion.get("attestation_hash") == payload_hash(deletion,"attestation_hash")
assert deletion["block_id"] == contract["block_id"]
assert deletion["training_manifest_hash"] == training["manifest_hash"]
creation=read_object(creation_path)
assert creation.get("schema") == EVALUATOR_CREATION_SCHEMA
assert creation.get("attestation_hash") == payload_hash(creation,"attestation_hash")
assert creation["block_id"] == contract["block_id"]
assert creation["training_pod_deletion_attestation_hash"] == deletion["attestation_hash"]
assert creation["verified_by"] == "host_launcher"
expected_identity={"execution_kind":"devpod","namespace":namespace,"pod_name":pod_name,"pod_uid":uid}
assert creation["evaluator_pod_identity"] == expected_identity
assert pod_name == template["expected_evaluator_pod_name"]
assert namespace == template["expected_evaluator_pod_namespace"]

def timestamp(value):
    return dt.datetime.fromisoformat(str(value).replace("Z","+00:00"))
assert timestamp(creation["kubernetes_creation_timestamp"]) > timestamp(deletion["not_found_verified_at"])
recovery_deletion_hash=""
if recovery_required:
    recovery_deletion=read_object(recovery_deletion_path)
    assert recovery_deletion.get("schema") == (
        "decision_admissibility_wp8_tier2_formal_"
        "recovery_pod_deletion_attestation_v1"
    )
    assert recovery_deletion.get("attestation_hash") == payload_hash(
        recovery_deletion,"attestation_hash"
    )
    assert recovery_deletion["block_id"] == contract["block_id"]
    assert recovery_deletion["training_manifest_hash"] == training["manifest_hash"]
    assert recovery_deletion["not_found_verified"] is True
    assert recovery_deletion["kubernetes_reason"] == "NotFound"
    assert recovery_deletion["verified_by"] == "host_launcher"
    assert recovery_deletion["terminal_metric_observed_before_not_found"] is False
    assert timestamp(creation["kubernetes_creation_timestamp"]) > timestamp(
        recovery_deletion["not_found_verified_at"]
    )
    recovery_deletion_hash=recovery_deletion["attestation_hash"]
else:
    assert finalizer_source.resolve() == source.resolve()

points=mount_points()
required={"/opt/nautilus","/fixed/train_view","/fixed/evaluator_view","/contract","/output"}
assert required <= points, (required,sorted(points))
if recovery_required:
    assert "/recovery" in points
for forbidden in ("/workspace","/memory","/secrets/mlevolve.env"):
    assert forbidden not in points and not Path(forbidden).exists(), forbidden
assert not environment_has_solver_secret()
assert not any(Path("/dev").glob("nvidia*"))
if shutil.which("nvidia-smi"):
    gpu_probe=subprocess.run(["nvidia-smi","-L"],capture_output=True,text=True)
    assert gpu_probe.returncode != 0 or not gpu_probe.stdout.strip()
roots=(source,train_root,evaluator_root)
if recovery_required:
    roots=(*roots,finalizer_source)
for root in roots:
    probe=root/".formal_write_probe"
    try:
        probe.write_text("forbidden")
    except OSError:
        pass
    else:
        probe.unlink(missing_ok=True)
        raise AssertionError(f"read-only mount is writable: {root}")
receipt={
    "schema":EVALUATOR_ISOLATION_SCHEMA,
    "block_id":contract["block_id"],
    "training_manifest_hash":training["manifest_hash"],
    "training_pod_deletion_attestation_hash":deletion["attestation_hash"],
    "evaluator_manifest_sha256":contract["evaluator_manifest_sha256"],
    "train_manifest_sha256":contract["train_manifest_sha256"],
    "source_snapshot_sha256":contract["source_snapshot_sha256"],
    "container_image_digest":contract["container_image_digest"],
    "evaluator_pod_identity":expected_identity,
    "evaluator_creation_attestation_hash":creation["attestation_hash"],
    "cpu_only":True,
    "cpu_count":len(os.sched_getaffinity(0)),
    "memory_bundle_absent":True,
    "solver_secret_absent":True,
    "solver_environment_absent":True,
    "whole_workspace_absent":True,
    "source_read_only":True,
    "train_view_read_only":True,
    "evaluator_view_read_only":True,
    "created_after_training_pod_not_found":True,
    "recovery_finalizer_overlay_used":recovery_required,
    "recovery_pod_deletion_attestation_hash":recovery_deletion_hash,
    "mount_points":sorted(points),
    "receipt_hash":"",
}
receipt["receipt_hash"]=payload_hash(receipt,"receipt_hash")
output_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
PY

python -m fixed_holdout.formal_block_evaluate \
  --output-root "$OUTPUT" \
  --evaluator-manifest "$EVALUATOR_VIEW/fixed_holdout_manifest.json" \
  --training-pod-deletion-attestation "$DELETION_ATTESTATION" \
  --evaluator-isolation "$OUTPUT/EVALUATOR_ISOLATION.json" \
  > "$OUTPUT/EVALUATION_FINALIZER_STDOUT.json"

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT/EVALUATION_FINISHED_AT"
printf 'evaluation_complete\n' > "$OUTPUT/STATE"
touch "$OUTPUT/EVALUATION_COMPLETE"
printf '0\n' > "$OUTPUT/EVALUATOR_LAUNCHER_EXIT_CODE"

python - "$OUTPUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve()
summary=json.loads((root/"EVALUATION_SUMMARY.json").read_text())
assert summary["formal_tier2_evidence"] is True
assert summary["effect_claim_authorized"] is False
assert not list(root.rglob("labels.csv"))
assert not list(root.rglob("*.env"))
files={
    path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(root.rglob("*"))
    if path.is_file()
    and path.name not in {
        "BLOCK_EVALUATION_FILE_MANIFEST.json",
        "EVALUATOR_LAUNCHER.log",
    }
}
p={"schema":"decision_admissibility_wp8_tier2_formal_block_evaluation_files_v1","block_id":summary["block_id"],"file_count":len(files),"file_hashes":files,"hidden_labels_included":False,"solver_secrets_included":False,"manifest_hash":""}
u={k:v for k,v in p.items() if k!="manifest_hash"}
p["manifest_hash"]=hashlib.sha256(json.dumps(u,sort_keys=True,separators=(",",":")).encode()).hexdigest()
(root/"BLOCK_EVALUATION_FILE_MANIFEST.json").write_text(json.dumps(p,indent=2,sort_keys=True)+"\n")
PY
trap - EXIT
