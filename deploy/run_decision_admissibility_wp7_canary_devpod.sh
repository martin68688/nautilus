#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC="${WP7_SOURCE_ROOT:-/workspace/decision-admissibility-wp7-certified-canary-r14-source}"
PVC_ROOT="${WP7_PVC_ROOT:-/workspace}"
CANARY_LABEL="${WP7_CANARY_LABEL:-wp7-canary-certified-aerial-r25}"
RUN_ROOT="${WP7_RUN_ROOT:-/workspace/decision-admissibility-wp7-canary-certified-aerial-r25}"
WORKSPACE_ROOT="${WP7_WORKSPACE_ROOT:-/work/decision-admissibility-wp7-canary-certified-aerial-r25-workspaces}"
SHADOW_ROOT="${WP7_SHADOW_ROOT:-$PVC_ROOT/decision-admissibility-wp7-shadow-certified-aerial-r20}"
STAGING_ABORT="$PVC_ROOT/decision-admissibility-wp7-canary-r15-staging-abort.json"
PREVIOUS_CANARY_ABORT="${WP7_PREVIOUS_CANARY_EVIDENCE:-$PVC_ROOT/decision-admissibility-wp7-canary-certified-aerial-r24/CANARY_ABORT_REPORT.json}"
FORMAL_BUNDLE_REL="${WP7_FORMAL_BUNDLE_REL:-decision-admissibility-wp7-certified-image-bundle-r3/bundles/certified-image-task-heldout-v4}"
FORMAL_BUNDLE="$PVC_ROOT/$FORMAL_BUNDLE_REL"
CURRENT_REL="${WP7_CURRENT_REL:-decision-admissibility-wp7-canary-certified-aerial-r25/CURRENT.json}"
EXPECTED_HEAD="${WP7_EXPECTED_HEAD:-b47dab63b7861f3ea0871094d6dd07b77e6b81a4}"
EXPECTED_BUNDLE_ID="${WP7_EXPECTED_BUNDLE_ID:-mlevolve-be034ec-image-aerial-task-heldout-certified-replay-v4}"
EXPECTED_BUNDLE_SHA256="${WP7_EXPECTED_BUNDLE_MANIFEST_SHA256:?WP7_EXPECTED_BUNDLE_MANIFEST_SHA256 is required}"
EXPECTED_SOURCE_SHA256="${WP7_EXPECTED_SOURCE_SHA256:?WP7_EXPECTED_SOURCE_SHA256 is required}"
EXPECTED_PROTOCOL_REF="${WP7_EXPECTED_PROTOCOL_REF:-mlevolve-default@2#4e54d9e6e3c44af8d92f578ef25b4be489b602e62ccc2ac88fa2113768f7eff2}"
export EXPECTED_PROTOCOL_REF
EXPECTED_REPLAY_CLAUSE_ID="${WP7_EXPECTED_REPLAY_CLAUSE_ID:?WP7_EXPECTED_REPLAY_CLAUSE_ID is required}"
export EXPECTED_REPLAY_CLAUSE_ID
EXPECTED_REPLAY_SOURCE_TASK_ID="${WP7_EXPECTED_REPLAY_SOURCE_TASK_ID:-leaf-classification}"
export EXPECTED_REPLAY_SOURCE_TASK_ID
EXPECTED_SHADOW_PACKET_SHA256="${WP7_EXPECTED_SHADOW_REVIEWED_PACKET_SHA256:?WP7_EXPECTED_SHADOW_REVIEWED_PACKET_SHA256 is required}"
EXPECTED_SHADOW_REPORT_SHA256="${WP7_EXPECTED_SHADOW_REVIEW_REPORT_SHA256:?WP7_EXPECTED_SHADOW_REVIEW_REPORT_SHA256 is required}"
EXPECTED_SHADOW_ROOT_POST_REVIEW_SHA256="${WP7_EXPECTED_SHADOW_ROOT_POST_REVIEW_SHA256:?WP7_EXPECTED_SHADOW_ROOT_POST_REVIEW_SHA256 is required}"
TASK="${WP7_TASK:-aerial-cactus-identification}"
ROLLOUT_ID="${WP7_ROLLOUT_ID:-wp7-canary-certified-image-task-heldout-aerial-r25}"
EXPERIMENT_GROUP="${WP7_EXPERIMENT_GROUP:-wp7_canary_certified_image_task_heldout_aerial_r25}"
BASELINE_GROUP="${WP7_BASELINE_GROUP:-wp7_shadow_certified_image_task_heldout_aerial_r20}"
export TASK CANARY_LABEL

export PYTHONPATH="$SRC:$SRC/mlevolve"
cd "$SRC/mlevolve"

test ! -e "$RUN_ROOT"
test ! -e "$WORKSPACE_ROOT"
test "$(cat "$SHADOW_ROOT/STATE")" = "shadow_complete_review_pending"
test -s "$SHADOW_ROOT/shadow_review_packet_reviewed.json"
test -s "$SHADOW_ROOT/shadow_review_report.json"
test -s "$SHADOW_ROOT/ROOT_PRE_REVIEW_VERIFICATION.json"
test -s "$SHADOW_ROOT/ROOT_POST_REVIEW_VERIFICATION.json"
test -s "$STAGING_ABORT"
test -s "$PREVIOUS_CANARY_ABORT"
mkdir -p "$RUN_ROOT/runs" "$WORKSPACE_ROOT" "$PVC_ROOT/cache/huggingface"
exec > >(tee -a "$RUN_ROOT/launcher.log") 2>&1

printf 'wp7_canary_start=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
printf 'source_root=%s\n' "$SRC"
printf 'run_root=%s\n' "$RUN_ROOT"
printf 'reviewed_shadow_root=%s\n' "$SHADOW_ROOT"
printf 'formal_bundle=%s\n' "$FORMAL_BUNDLE"

verify_source_snapshot() {
  local report_path="${1:-$RUN_ROOT/SOURCE_POSTRUN_VERIFICATION.json}"
  python - "$SRC" "$EXPECTED_HEAD" "$EXPECTED_SOURCE_SHA256" \
    "$report_path" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_head = sys.argv[2]
expected_source_sha256 = sys.argv[3]
report_path = Path(sys.argv[4])
manifest = json.loads((root / "WP7_SOURCE_MANIFEST.json").read_text())
assert manifest["base_commit"] == expected_head
assert manifest["source_sha256"] == expected_source_sha256
files = {}
writable_paths = (
    ["WP7_SOURCE_MANIFEST.json"]
    if (root / "WP7_SOURCE_MANIFEST.json").stat().st_mode & 0o222
    else []
)
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "WP7_SOURCE_MANIFEST.json":
        continue
    relative = str(path.relative_to(root))
    files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_mode & 0o222:
        writable_paths.append(relative)
unsigned = {key: value for key, value in manifest.items() if key != "source_sha256"}
actual_source_sha256 = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
added = sorted(set(files) - set(manifest["file_hashes"]))
removed = sorted(set(manifest["file_hashes"]) - set(files))
changed = sorted(
    path
    for path in set(files) & set(manifest["file_hashes"])
    if files[path] != manifest["file_hashes"][path]
)
unchanged = bool(
    files == manifest["file_hashes"]
    and actual_source_sha256 == expected_source_sha256
    and not writable_paths
)
payload = {
    "schema": "decision_admissibility_wp7_source_postrun_verification_v1",
    "source_sha256": expected_source_sha256,
    "unchanged": unchanged,
    "added_paths": added,
    "removed_paths": removed,
    "changed_paths": changed,
    "pycache_paths": sorted(
        path for path in files if "__pycache__" in Path(path).parts
    ),
    "writable_paths": writable_paths,
}
report_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
assert unchanged, payload
print(json.dumps(payload, sort_keys=True))
PY
}

finalize_launcher() {
  local launcher_rc=$?
  trap - EXIT
  set +e
  printf '%s\n' "$launcher_rc" > "$RUN_ROOT/LAUNCHER_EXIT_CODE"
  if [[ "$launcher_rc" -ne 0 ]]; then
    local current_state=""
    current_state="$(cat "$RUN_ROOT/STATE" 2>/dev/null || true)"
    case "$current_state" in
      failed|failed_*) ;;
      *) printf 'failed\n' > "$RUN_ROOT/STATE" ;;
    esac
    verify_source_snapshot "$RUN_ROOT/SOURCE_POSTRUN_VERIFICATION.json" || true
  fi
  exit "$launcher_rc"
}

handle_launcher_term() {
  trap - TERM
  printf 'TERM\n' > "$RUN_ROOT/LAUNCHER_SIGNAL"
  printf 'failed_signal_term\n' > "$RUN_ROOT/STATE"
  exit 143
}

trap finalize_launcher EXIT
trap handle_launcher_term TERM

verify_source_snapshot "$RUN_ROOT/SOURCE_PREFLIGHT_VERIFICATION.json"

python - "$SHADOW_ROOT" "$RUN_ROOT" "$SRC" \
  "$EXPECTED_SHADOW_PACKET_SHA256" "$EXPECTED_SHADOW_REPORT_SHA256" \
  "$EXPECTED_SHADOW_ROOT_POST_REVIEW_SHA256" "$STAGING_ABORT" \
  "$PREVIOUS_CANARY_ABORT" "$EXPECTED_PROTOCOL_REF" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from authority.rollout import (
    load_shadow_records_from_ledger,
    verify_shadow_review_packet,
)
from authority.protocol_registry import ProtocolRegistry

shadow_root = Path(sys.argv[1])
run_root = Path(sys.argv[2])
source_root = Path(sys.argv[3])
expected_packet_sha = sys.argv[4]
expected_report_sha = sys.argv[5]
expected_root_post_review_sha = sys.argv[6]
staging_abort_path = Path(sys.argv[7])
previous_canary_abort_path = Path(sys.argv[8])
expected_protocol_ref = sys.argv[9]

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

packet_path = shadow_root / "shadow_review_packet_reviewed.json"
report_path = shadow_root / "shadow_review_report.json"
assert sha256(packet_path) == expected_packet_sha
assert sha256(report_path) == expected_report_sha
shadow_summary = json.loads((shadow_root / "SHADOW_RUN_SUMMARY.json").read_text())
ledger = Path(shadow_summary["run_dir"]) / "authority_events.jsonl"
report = verify_shadow_review_packet(
    json.loads(packet_path.read_text()),
    load_shadow_records_from_ledger(ledger),
)
assert report == json.loads(report_path.read_text())
assert report["verified"] is True
assert report["reviewed_sample_count"] == report["population_count"]
assert report["population_count"] == shadow_summary["review_population_count"]
for forbidden in (
    "confirmed_authority_false_denial",
    "confirmed_legacy_false_denial",
    "requires_fix",
):
    assert report["disposition_counts"].get(forbidden, 0) == 0, report
root_verification = json.loads(
    (shadow_root / "ROOT_PRE_REVIEW_VERIFICATION.json").read_text()
)
assert root_verification["verified"] is True
assert root_verification["review_population_count"] == report["population_count"]
assert root_verification["clean_observation_count"] >= 2
assert root_verification["trusted_scope_receipt_count"] > 0
assert root_verification["trusted_scope_receipt_count"] == (
    shadow_summary["trusted_scope_receipt_count"]
)
assert sha256(shadow_root / "ROOT_POST_REVIEW_VERIFICATION.json") == (
    expected_root_post_review_sha
)
root_post_review = json.loads(
    (shadow_root / "ROOT_POST_REVIEW_VERIFICATION.json").read_text()
)
assert root_post_review["verified"] is True
assert root_post_review["eligible_for_synthetic_enforce_and_canary"] is True
assert root_post_review["authority_false_allow_count"] == 0
assert root_post_review["authority_false_denial_count"] == 0
assert root_post_review["clean_observation_count"] == (
    root_verification["clean_observation_count"]
)
assert root_post_review["trusted_scope_receipt_count"] == (
    root_verification["trusted_scope_receipt_count"]
)
source_manifest = json.loads(
    (source_root / "WP7_SOURCE_MANIFEST.json").read_text()
)
previous_canary_abort = json.loads(previous_canary_abort_path.read_text())
if previous_canary_abort.get("schema") == (
    "decision_admissibility_wp7_canary_provenance_v2"
):
    # A corrected canary may use the last completed canary only as its
    # immutable parent-source checkpoint; it must still use a fresh source and
    # fresh output roots.  This branch is selected by r7+ and never reuses r25
    # as new success evidence.
    assert previous_canary_abort["canary_source_sha256"] == (
        source_manifest["parent_source_sha256"]
    )
    assert previous_canary_abort["canary_source_sha256"] != (
        source_manifest["source_sha256"]
    )
    assert previous_canary_abort["canary_source_sha256"] == (
        "22ba164efe6bf4513e0c31c255e248c8d68d9e86c968d8fdabc25ca0e10e7725"
    )
else:
    assert previous_canary_abort["status"] == (
        "aborted_not_usable_as_canary_evidence"
    )
    assert previous_canary_abort["usable_as_wp7_success_evidence"] is False
    assert previous_canary_abort["output_path_must_not_be_reused"] is True
    assert previous_canary_abort["replacement_run_id"] == (
        "wp7-canary-certified-image-task-heldout-aerial-r25"
    )
    assert source_manifest["parent_source_sha256"] == (
        previous_canary_abort["remediation_parent_source_sha256"]
    )
    assert source_manifest["source_sha256"] == (
        previous_canary_abort["remediation_source_sha256"]
    )
    assert previous_canary_abort["failed_source_sha256"] == (
        "36e8cf831cf4d4a1952608308f977c687b9662394c2de1d8d41b3141f2942852"
    )
registry = ProtocolRegistry(source_root / "mlevolve" / "config" / "protocols")
active_protocol = registry.get("mlevolve-default", "2").ref().key()
assert active_protocol == expected_protocol_ref
staging_abort = json.loads(staging_abort_path.read_text())
assert staging_abort["classification"] == (
    "kubelet_exec_transport_timeout_before_source_manifest"
)
assert staging_abort["canary_run_root_created"] is False
assert staging_abort["online_canary_started"] is False
assert staging_abort["usable_as_wp7_evidence"] is False
preflight_exit_code = int(
    (shadow_root / "PREFLIGHT_EXIT_CODE").read_text().strip()
)
assert preflight_exit_code == 0
payload = {
    "schema": "decision_admissibility_wp7_canary_provenance_v2",
    "reviewed_shadow_root": str(shadow_root),
    "reviewed_shadow_packet_sha256": expected_packet_sha,
    "reviewed_shadow_report_sha256": expected_report_sha,
    "reviewed_shadow_report_hash": report["report_hash"],
    "reviewers": report["reviewers"],
    "review_disposition_counts": report["disposition_counts"],
    "shadow_authority_false_allow_or_false_denial_count": 0,
    "shadow_root_verification_sha256": sha256(
        shadow_root / "ROOT_PRE_REVIEW_VERIFICATION.json"
    ),
    "shadow_root_post_review_verification_sha256": (
        expected_root_post_review_sha
    ),
    "canary_source_sha256": source_manifest["source_sha256"],
    "canary_source_parent_sha256": source_manifest[
        "parent_source_sha256"
    ],
    "canary_source_changed_paths": source_manifest[
        "changed_paths_from_parent"
    ],
    "canary_source_remediation_reason": source_manifest[
        "remediation_reason"
    ],
    "active_protocol_ref": active_protocol,
    "previous_staging_abort_sha256": sha256(staging_abort_path),
    "previous_staging_abort_not_experiment_evidence": True,
    "previous_canary_abort_sha256": sha256(previous_canary_abort_path),
    "previous_canary_abort_not_success_evidence": True,
    "shadow_preflight_exit_code": preflight_exit_code,
    "shadow_preflight_tests_sha256": sha256(
        shadow_root / "PREFLIGHT_TESTS.log"
    ),
    "shadow_preflight_deselection_sha256": sha256(
        shadow_root / "PREFLIGHT_DESELECTION.json"
    ),
    "shadow_clean_observation_count": root_verification[
        "clean_observation_count"
    ],
    "shadow_trusted_scope_receipt_count": root_verification[
        "trusted_scope_receipt_count"
    ],
    "transfer_design": "same_domain_different_task_task_heldout",
    "canary_oracle_must_be_independently_reviewed": True,
}
(run_root / "CANARY_PROVENANCE.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n"
)
print(json.dumps(payload, sort_keys=True))
PY

test -s "$FORMAL_BUNDLE/manifest.json"
test -s "$PVC_ROOT/nautilus/mlevolve/.env"
test -s "$PVC_ROOT/nautilus/mlevolve/data/$TASK/prepared/public/description.md"

python - "$PVC_ROOT" "$FORMAL_BUNDLE_REL" "$CURRENT_REL" \
  "$EXPECTED_BUNDLE_ID" "$EXPECTED_BUNDLE_SHA256" "$RUN_ROOT" <<'PY'
import json
import os
import sys
from pathlib import Path

from authority.memory_snapshot import (
    ImmutableBaseBundle,
    make_current_pointer,
    write_json_atomic,
)

root = Path(sys.argv[1]).resolve()
bundle = ImmutableBaseBundle.load(root / sys.argv[2], verify_artifacts=True)
current_path = root / sys.argv[3]
assert bundle.bundle_id == sys.argv[4]
assert bundle.manifest_sha256 == sys.argv[5]
split = json.loads((bundle.path / "splits" / "active.json").read_text())
assert split["split_kind"] == "same-domain-task-heldout"
assert split["heldout_task_ids"] == [os.environ.get("TASK", "aerial-cactus-identification")]
assert split["validation"]["cross_domain_source_run_count"] == 0
pointer = make_current_pointer(
    bundle_path=str(bundle.path.relative_to(root)),
    manifest=bundle.manifest,
    parent_bundle=bundle.manifest.get("parent_bundle"),
)
write_json_atomic(current_path, pointer)
write_json_atomic(
    Path(sys.argv[6]) / "BUNDLE_PREFLIGHT.json",
    {
        "schema": "decision_admissibility_wp7_canary_bundle_preflight_v1",
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "current_pointer_sha256": pointer["pointer_sha256"],
        "split_kind": split["split_kind"],
        "target_task_id": os.environ.get("TASK", "aerial-cactus-identification"),
        "target_domain": "image",
        "cross_domain_source_run_count": 0,
    },
)
PY

export HF_HOME="$PVC_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export MLEVOLVE_CONFIG="$SRC/mlevolve/config/config_authority_canary_enforce.yaml"
export MLEVOLVE_CODE_REVISION="$EXPECTED_HEAD"
export MLEVOLVE_CODE_WORKTREE_SHA256="$EXPECTED_SOURCE_SHA256"
set -a
. "$PVC_ROOT/nautilus/mlevolve/.env"
set +a
export DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL

cd "$SRC"
printf 'synthetic_enforce_preflight_running\n' > "$RUN_ROOT/STATE"
set +e
python -m pytest -q -p no:cacheprovider \
  tests/authority/test_enforce_rollout.py \
  tests/authority/test_canary_gate.py \
  tests/authority/test_high_risk_fail_closed.py \
  tests/authority/test_runtime_protocol_observer.py \
  tests/authority/test_visibility_pre_prompt.py \
  tests/authority/test_visibility_projection_bypass.py \
  tests/authority/test_canary_launcher_static.py \
  tests/test_protocol_repair.py \
  2>&1 | tee "$RUN_ROOT/SYNTHETIC_ENFORCE_TESTS.log"
preflight_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$preflight_rc" > "$RUN_ROOT/SYNTHETIC_ENFORCE_EXIT_CODE"
if [[ "$preflight_rc" -ne 0 ]]; then
  printf 'failed_synthetic_enforce\n' > "$RUN_ROOT/STATE"
  verify_source_snapshot
  exit "$preflight_rc"
fi

cd "$SRC/mlevolve"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  | tee "$RUN_ROOT/GPU_IDENTITY.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_ROOT/STARTED_AT"
printf 'running\n' > "$RUN_ROOT/STATE"
set +e
timeout --foreground --signal=TERM --kill-after=30s 6500s \
  python -u run.py \
    exp_id="$TASK" \
    exp_name="$CANARY_LABEL" \
    dataset_dir="$PVC_ROOT/nautilus/mlevolve/data" \
    data_dir="$PVC_ROOT/nautilus/mlevolve/data/$TASK/prepared/public" \
    desc_file="$PVC_ROOT/nautilus/mlevolve/data/$TASK/prepared/public/description.md" \
    log_dir="$RUN_ROOT/runs" \
    workspace_dir="$WORKSPACE_ROOT" \
    evaluation_authority.mode=enforce \
    evaluation_authority.active_protocol_id=mlevolve-default \
    evaluation_authority.active_protocol_version=2 \
    evaluation_authority.rollout_id="$ROLLOUT_ID" \
    evaluation_authority.collector_version=2 \
    evaluation_authority.require_bound_bundle=true \
    evaluation_authority.expected_bundle_id="$EXPECTED_BUNDLE_ID" \
    evaluation_authority.expected_bundle_manifest_sha256="$EXPECTED_BUNDLE_SHA256" \
    external_skill_memory.bundle_root="$PVC_ROOT" \
    external_skill_memory.current_pointer_path="$CURRENT_REL" \
    external_skill_memory.retrieval_control=stage_hybrid \
    agent.draft_role_policy.roles='[coldstart_baseline,memory_transfer,novel_exploration]' \
    run_identity.experiment_group="$EXPERIMENT_GROUP" \
    run_identity.baseline_reference_group="$BASELINE_GROUP" \
    agent.steps=10 \
    agent.time_limit=6000 \
    agent.seed=271828 \
    agent.search.num_gpus=1 \
    agent.search.parallel_search_num=1 \
    cpu_number=8 \
    exec.timeout=600
run_rc=$?
set -e
printf '%s\n' "$run_rc" > "$RUN_ROOT/EXIT_CODE"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_ROOT/FINISHED_AT"
if [[ "$run_rc" -ne 0 ]]; then
  printf 'failed\n' > "$RUN_ROOT/STATE"
  verify_source_snapshot
  exit "$run_rc"
fi

latest_run="$(find "$RUN_ROOT/runs" -mindepth 1 -maxdepth 1 -type d \
  -name "*_${CANARY_LABEL}" | LC_ALL=C sort | tail -n 1)"
test -n "$latest_run"
logs="$latest_run"
test -s "$logs/authority_events.jsonl"
test -s "$logs/authority_rollout_report.json"

cd "$SRC"
python paper-skills/memory_bundle/build_canary_oracle_packet.py \
  --ledger "$logs/authority_events.jsonl" \
  --packet "$RUN_ROOT/canary_oracle_packet.json"

python - "$logs" "$RUN_ROOT" "$FORMAL_BUNDLE" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from authority.domain_scope import audit_same_domain_task_heldout_exposures
from authority.ledger import AuthorityLedger
from authority.memory_snapshot import ImmutableBaseBundle, write_json_atomic
from authority.rollout import load_shadow_records_from_ledger

logs = Path(sys.argv[1])
run_root = Path(sys.argv[2])
bundle = ImmutableBaseBundle.load(Path(sys.argv[3]), verify_artifacts=True)
ledger = AuthorityLedger(logs / "authority_events.jsonl")
assert ledger.verify()
records = load_shadow_records_from_ledger(ledger.path)
enforced = [record for record in records if record.enforced]
assert enforced
rollout = json.loads((logs / "authority_rollout_report.json").read_text())
assert rollout["mode"] == "enforce"
assert rollout["record_count"] == len(records)
assert rollout["enforced_record_count"] == len(enforced)
assert len(enforced) >= 20
assert rollout["rollout_versions"]["bundle_id"] == bundle.bundle_id
assert rollout["rollout_versions"]["bundle_manifest_sha256"] == bundle.manifest_sha256
assert rollout["rollout_versions"]["protocol_ref"] == os.environ[
    "EXPECTED_PROTOCOL_REF"
]
internal_error_count = sum(
    count
    for taxonomy, count in rollout["taxonomy_counts"].items()
    if "internal_error" in taxonomy
)
assert internal_error_count == 0
runtime_exception_lines = [
    line
    for line in (logs / "MLEvolve.log").read_text(errors="replace").splitlines()
    if "Exception during task execution:" in line
]
runtime_exception_audit = {
    "schema": "decision_admissibility_wp7_runtime_exception_audit_v1",
    "valid": not runtime_exception_lines,
    "runtime_exception_count": len(runtime_exception_lines),
    "runtime_exception_lines": runtime_exception_lines,
    "authority_internal_error_taxonomy_count": internal_error_count,
}
write_json_atomic(
    run_root / "RUNTIME_EXCEPTION_AUDIT.json", runtime_exception_audit
)
assert runtime_exception_audit["valid"] is True, runtime_exception_audit
events = [json.loads(line) for line in ledger.path.read_text().splitlines()]
exposures = [
    event["payload"]
    for event in events
    if event.get("event_type") == "experience_exposed"
]
exposure_audit = audit_same_domain_task_heldout_exposures(
    exposures,
    bundle.read_jsonl("sop/clauses.jsonl"),
    target_task_id=os.environ.get("TASK", "aerial-cactus-identification"),
    target_domain="image",
    certified_clause_id=os.environ["EXPECTED_REPLAY_CLAUSE_ID"],
    certified_source_task_id=os.environ["EXPECTED_REPLAY_SOURCE_TASK_ID"],
)
write_json_atomic(run_root / "EXPOSURE_SCOPE_AUDIT.json", exposure_audit)
assert exposure_audit["valid"] is True, exposure_audit
assert exposure_audit["invalid_exposure_count"] == 0
assert exposure_audit["classified_exposure_count"] == len(exposures)
assert exposure_audit["certified_method_exposure_count"] > 0
packet = json.loads((run_root / "canary_oracle_packet.json").read_text())
assert packet["decision_count"] == len(enforced)
bundle.assert_unchanged()

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "schema": "decision_admissibility_wp7_online_canary_summary_v3",
    "status": "canary_complete_oracle_review_pending",
    "run_dir": str(logs),
    "rollout_id": rollout["rollout_versions"]["rollout_id"],
    "rollout_version_hash": rollout["rollout_versions"]["version_hash"],
    "active_protocol_ref": rollout["rollout_versions"]["protocol_ref"],
    "record_count": len(records),
    "enforced_record_count": len(enforced),
    "minimum_gate_decision_count": 20,
    "minimum_gate_decision_count_met": len(enforced) >= 20,
    "taxonomy_counts": rollout["taxonomy_counts"],
    "internal_error_count": 0,
    "runtime_exception_count": 0,
    "exposure_event_schema": "experience_exposure_event_v2",
    "exposure_count": exposure_audit["exposure_event_count"],
    "classified_exposure_count": exposure_audit[
        "classified_exposure_count"
    ],
    "certified_method_exposure_count": exposure_audit[
        "certified_method_exposure_count"
    ],
    "diagnostic_debug_exposure_count": exposure_audit[
        "diagnostic_debug_exposure_count"
    ],
    "exposed_clause_ids": exposure_audit["exposed_clause_ids"],
    "exposed_source_task_ids": exposure_audit[
        "exposed_source_task_ids"
    ],
    "invalid_exposure_count": 0,
    "cross_domain_or_unscoped_exposure_count": 0,
    "bundle_id": bundle.bundle_id,
    "bundle_manifest_sha256": bundle.manifest_sha256,
    "authority_ledger_sha256": sha256(logs / "authority_events.jsonl"),
    "rollout_report_sha256": sha256(logs / "authority_rollout_report.json"),
    "oracle_packet_sha256": sha256(run_root / "canary_oracle_packet.json"),
    "oracle_evidence_hash": packet["evidence_hash"],
    "exposure_scope_audit_sha256": sha256(
        run_root / "EXPOSURE_SCOPE_AUDIT.json"
    ),
    "runtime_exception_audit_sha256": sha256(
        run_root / "RUNTIME_EXCEPTION_AUDIT.json"
    ),
    "canary_provenance_sha256": sha256(run_root / "CANARY_PROVENANCE.json"),
    "formal_bundle_verified_unchanged": True,
}
write_json_atomic(run_root / "CANARY_RUN_SUMMARY.json", summary)
print(json.dumps(summary, sort_keys=True))
PY

verify_source_snapshot
printf 'canary_complete_oracle_review_pending\n' > "$RUN_ROOT/STATE"
touch "$RUN_ROOT/CANARY_COMPLETE"
printf 'wp7_canary_finish=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
