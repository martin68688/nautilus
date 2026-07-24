#!/usr/bin/env bash
set -euo pipefail

# Set before the first Python process.  The source snapshot is hash-bound and
# must remain byte-for-byte unchanged throughout the run.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC="${WP7_SOURCE_ROOT:-/work/decision-admissibility-wp7-certified-canary-r5-source}"
PVC_ROOT="${WP7_PVC_ROOT:-/workspace}"
RUN_ROOT="${WP7_RUN_ROOT:-/workspace/decision-admissibility-wp7-shadow-certified-aerial-r20}"
WORKSPACE_ROOT="${WP7_WORKSPACE_ROOT:-/work/decision-admissibility-wp7-shadow-certified-aerial-r20-workspaces}"
FORMAL_BUNDLE_REL="${WP7_FORMAL_BUNDLE_REL:-decision-admissibility-wp7-certified-image-bundle-r2/bundles/certified-image-task-heldout-v3}"
FORMAL_BUNDLE="$PVC_ROOT/$FORMAL_BUNDLE_REL"
CURRENT_REL="${WP7_CURRENT_REL:-decision-admissibility-wp7-shadow-certified-aerial-r20/CURRENT.json}"
CERTIFIED_PROVENANCE="${WP7_CERTIFIED_PROVENANCE:-$PVC_ROOT/decision-admissibility-wp7-certified-image-bundle-r3/publication_provenance.json}"
PREVIOUS_SHADOW_RUN="$PVC_ROOT/decision-admissibility-wp7-shadow-aerial-r12"
PRELAUNCH_ABORT="$PVC_ROOT/decision-admissibility-wp7-r13-prelaunch-abort.json"
PREVIOUS_DIAGNOSTIC_RUN="$PVC_ROOT/decision-admissibility-wp7-shadow-aerial-r13"
EXPECTED_DIAGNOSTIC_ABORT_SHA256="0346dd9cf789b3bb87ce662be6f79bf79426def9134931c866c22df140f10bf8"
EXPECTED_DIAGNOSTIC_SOURCE_CLASSIFICATION_SHA256="11b09d7e35d0418b46630678a2c2710fff2c3b3a7f6b9c133ab68794ecdd63fa"
EXPECTED_PREVIOUS_PACKET_SHA256="f6e0d5172afaeb3d8d017134b6f04b8407840690f6b2d45b06a5e7cb81d1a2ab"
EXPECTED_PREVIOUS_REPORT_SHA256="a3564745869ea9dff83fafad8c8ad487afdd862ce250a23dd2bc3a8db3662820"
EXPECTED_PREVIOUS_REPORT_HASH="678f24f888648cc08ac91d1a0bf62c02871020a474bddf23f8b85e89c493a139"
EXPECTED_HEAD="b47dab63b7861f3ea0871094d6dd07b77e6b81a4"
EXPECTED_BUNDLE_ID="${WP7_EXPECTED_BUNDLE_ID:-mlevolve-be034ec-image-aerial-task-heldout-certified-replay-v3}"
EXPECTED_MANIFEST_SHA256="${WP7_EXPECTED_BUNDLE_MANIFEST_SHA256:?WP7_EXPECTED_BUNDLE_MANIFEST_SHA256 is required}"
EXPECTED_SOURCE_SHA256="${WP7_EXPECTED_SOURCE_SHA256:?WP7_EXPECTED_SOURCE_SHA256 is required}"
EXPECTED_CERTIFIED_PROVENANCE_SHA256="${WP7_EXPECTED_CERTIFIED_PROVENANCE_SHA256:?WP7_EXPECTED_CERTIFIED_PROVENANCE_SHA256 is required}"
EXPECTED_REPLAY_CLAUSE_ID="${WP7_EXPECTED_REPLAY_CLAUSE_ID:?WP7_EXPECTED_REPLAY_CLAUSE_ID is required}"
export EXPECTED_REPLAY_CLAUSE_ID
TASK="aerial-cactus-identification"
TARGET_DOMAIN="image"
ROLLOUT_ID="wp7-shadow-certified-image-task-heldout-aerial-r20"

# Bind every repository import to the immutable staged source before the first
# Python process that imports project modules.
export PYTHONPATH="$SRC:$SRC/mlevolve"
cd "$SRC/mlevolve"

test ! -e "$RUN_ROOT"
test ! -e "$WORKSPACE_ROOT"
test -s "$PREVIOUS_SHADOW_RUN/STATE"
test "$(cat "$PREVIOUS_SHADOW_RUN/STATE")" = "shadow_complete_review_pending"
test -s "$PREVIOUS_SHADOW_RUN/shadow_review_packet_reviewed.json"
test -s "$PREVIOUS_SHADOW_RUN/shadow_review_report.json"
test -s "$PRELAUNCH_ABORT"
test -s "$CERTIFIED_PROVENANCE"
test "$(cat "$PREVIOUS_DIAGNOSTIC_RUN/STATE")" = "failed"
test "$(cat "$PREVIOUS_DIAGNOSTIC_RUN/EXIT_CODE")" = "143"
test -s "$PREVIOUS_DIAGNOSTIC_RUN/CONTROLLED_ABORT_REQUEST.json"
test -s "$PREVIOUS_DIAGNOSTIC_RUN/POSTRUN_SOURCE_CONTAMINATION_CLASSIFICATION.json"
mkdir -p "$RUN_ROOT/runs" "$WORKSPACE_ROOT" "$PVC_ROOT/cache/huggingface"
exec > >(tee -a "$RUN_ROOT/launcher.log") 2>&1

printf 'wp7_shadow_start=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
printf 'source_root=%s\n' "$SRC"
printf 'formal_bundle=%s\n' "$FORMAL_BUNDLE"
printf 'run_root=%s\n' "$RUN_ROOT"
printf 'workspace_root=%s\n' "$WORKSPACE_ROOT"
printf 'previous_shadow_run=%s\n' "$PREVIOUS_SHADOW_RUN"
printf 'previous_diagnostic_run=%s\n' "$PREVIOUS_DIAGNOSTIC_RUN"
printf 'task=%s\n' "$TASK"
printf 'target_domain=%s\n' "$TARGET_DOMAIN"
printf 'expected_bundle_id=%s\n' "$EXPECTED_BUNDLE_ID"
printf 'expected_bundle_manifest_sha256=%s\n' "$EXPECTED_MANIFEST_SHA256"
printf 'expected_source_sha256=%s\n' "$EXPECTED_SOURCE_SHA256"
printf 'certified_provenance=%s\n' "$CERTIFIED_PROVENANCE"
printf 'expected_replay_clause_id=%s\n' "$EXPECTED_REPLAY_CLAUSE_ID"
printf 'resources=1xA100,8CPU,32Gi\n'

python - "$PREVIOUS_SHADOW_RUN" "$RUN_ROOT" \
  "$EXPECTED_PREVIOUS_PACKET_SHA256" "$EXPECTED_PREVIOUS_REPORT_SHA256" \
  "$EXPECTED_PREVIOUS_REPORT_HASH" "$PRELAUNCH_ABORT" \
  "$PREVIOUS_DIAGNOSTIC_RUN" "$EXPECTED_DIAGNOSTIC_ABORT_SHA256" \
  "$EXPECTED_DIAGNOSTIC_SOURCE_CLASSIFICATION_SHA256" "$SRC" \
  "$CERTIFIED_PROVENANCE" "$EXPECTED_CERTIFIED_PROVENANCE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

previous = Path(sys.argv[1])
run_root = Path(sys.argv[2])
expected_packet_sha = sys.argv[3]
expected_report_sha = sys.argv[4]
expected_report_hash = sys.argv[5]
prelaunch_abort_path = Path(sys.argv[6])
diagnostic_run = Path(sys.argv[7])
expected_diagnostic_abort_sha = sys.argv[8]
expected_diagnostic_source_classification_sha = sys.argv[9]
source_root = Path(sys.argv[10])
certified_provenance_path = Path(sys.argv[11])
expected_certified_provenance_sha = sys.argv[12]

from authority.rollout import (
    load_shadow_records_from_ledger,
    verify_shadow_review_packet,
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

packet_path = previous / "shadow_review_packet_reviewed.json"
report_path = previous / "shadow_review_report.json"
assert sha256(packet_path) == expected_packet_sha
assert sha256(report_path) == expected_report_sha
stored_report = json.loads(report_path.read_text())
summary = json.loads((previous / "SHADOW_RUN_SUMMARY.json").read_text())
ledger = Path(summary["run_dir"]) / "authority_events.jsonl"
reverified = verify_shadow_review_packet(
    json.loads(packet_path.read_text()),
    load_shadow_records_from_ledger(ledger),
)
assert reverified == stored_report
assert stored_report["verified"] is True
assert stored_report["report_hash"] == expected_report_hash
assert stored_report["disposition_counts"] == {
    "confirmed_authority_false_denial": 11,
    "confirmed_legacy_false_allow": 9,
}
assert stored_report["reviewers"] == [
    "codex-wp7-r12-independent-reviewer-no-claude"
]
source_manifest = json.loads(
    (source_root / "WP7_SOURCE_MANIFEST.json").read_text()
)
prelaunch_abort = json.loads(prelaunch_abort_path.read_text())
assert prelaunch_abort["classification"] == "prelaunch_assertion_abort"
assert prelaunch_abort["training_started"] is False
assert prelaunch_abort["usable_as_wp7_shadow_evidence"] is False
assert prelaunch_abort["run_root_created"] is False
assert prelaunch_abort["workspace_root_created"] is False
diagnostic_abort_path = diagnostic_run / "CONTROLLED_ABORT_REQUEST.json"
diagnostic_source_path = (
    diagnostic_run / "POSTRUN_SOURCE_CONTAMINATION_CLASSIFICATION.json"
)
assert sha256(diagnostic_abort_path) == expected_diagnostic_abort_sha
assert sha256(diagnostic_source_path) == (
    expected_diagnostic_source_classification_sha
)
diagnostic_abort = json.loads(diagnostic_abort_path.read_text())
diagnostic_source = json.loads(diagnostic_source_path.read_text())
certified_provenance = json.loads(certified_provenance_path.read_text())
assert sha256(certified_provenance_path) == expected_certified_provenance_sha
assert certified_provenance["status"] == "sealed"
assert certified_provenance["source_parent_unchanged"] is True
assert certified_provenance["target_task_id"] == "aerial-cactus-identification"
assert certified_provenance["source_domain"] == "image"
assert certified_provenance["target_domain"] == "image"
assert certified_provenance["transfer_scope"] == "same_domain"
assert diagnostic_abort["classification"] == (
    "known_runtime_observer_coverage_gap"
)
assert diagnostic_abort["all_completed_candidates_use_grad_scaler_step"] is True
assert diagnostic_abort["usable_as_wp7_go_evidence"] is False
assert diagnostic_source["source_code_or_config_files_changed"] is False
assert diagnostic_source["all_added_paths_are_pycache"] is True
assert diagnostic_source["source_snapshot_usable_as_wp7_go_evidence"] is False

payload = {
    "schema": "decision_admissibility_wp7_false_denial_remediation_v1",
    "previous_shadow_run": str(previous),
    "previous_shadow_state": (previous / "STATE").read_text().strip(),
    "previous_shadow_summary_sha256": sha256(
        previous / "SHADOW_RUN_SUMMARY.json"
    ),
    "previous_reviewed_packet_sha256": expected_packet_sha,
    "previous_review_report_sha256": expected_report_sha,
    "previous_review_report_hash": expected_report_hash,
    "previous_review_reverified": True,
    "previous_disposition_counts": stored_report["disposition_counts"],
    "previous_reviewer": stored_report["reviewers"][0],
    "previous_reviewer_is_claude": False,
    "prelaunch_abort_path": str(prelaunch_abort_path),
    "prelaunch_abort_sha256": sha256(prelaunch_abort_path),
    "prelaunch_abort_not_experiment_evidence": True,
    "previous_diagnostic_run": str(diagnostic_run),
    "previous_diagnostic_state": (diagnostic_run / "STATE").read_text().strip(),
    "previous_diagnostic_exit_code": int(
        (diagnostic_run / "EXIT_CODE").read_text().strip()
    ),
    "previous_diagnostic_abort_sha256": expected_diagnostic_abort_sha,
    "previous_diagnostic_source_classification_sha256": (
        expected_diagnostic_source_classification_sha
    ),
    "previous_diagnostic_usable_as_go_evidence": False,
    "root_cause": (
        "ordinary clean successful nodes had only legacy_static_only protocol "
        "receipts because trusted runtime protocol collection was limited to "
        "protocol-repair runtime_provenance"
    ),
    "secondary_runtime_coverage_root_cause": diagnostic_abort["root_cause"],
    "remediation": {
        "observer_schema": "mlevolve_host_protocol_observation_v2",
        "host_parent_attestation_required": True,
        "runtime_argument_result_scope_hashes_required": True,
        "grad_scaler_step_fit_scope_supported": True,
        "staged_source_write_bits_removed": True,
        "deterministic_static_audit_still_required": True,
        "clean_rank_select_expected": "allow",
        "unactuated_promote_expected": "quarantine",
    },
    "new_source_root": str(source_root),
    "new_source_sha256": source_manifest["source_sha256"],
    "formal_transfer_design_unchanged": (
        "same_domain_different_task_task_heldout"
    ),
    "formal_bundle_unchanged": False,
    "certified_child_bundle_id": certified_provenance["child_bundle_id"],
    "certified_child_manifest_sha256": certified_provenance[
        "child_manifest_sha256"
    ],
    "certified_replay_clause_id": certified_provenance["replay_clause_id"],
    "certified_publication_provenance_sha256": (
        expected_certified_provenance_sha
    ),
    "fresh_shadow_required": True,
    "reuse_previous_decisions_for_go_no_go": False,
}
(run_root / "REMEDIATION_PROVENANCE.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

test -s "$SRC/WP7_SOURCE_MANIFEST.json"
test -s "$SRC/WP7_OVERLAY_PATHS.txt"
test -s "$FORMAL_BUNDLE/manifest.json"
test -s "$PVC_ROOT/nautilus/mlevolve/.env"
test -s "$PVC_ROOT/nautilus/mlevolve/data/$TASK/prepared/public/description.md"

python - "$SRC" "$EXPECTED_HEAD" "$EXPECTED_SOURCE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected_head = sys.argv[2]
expected_source_sha256 = sys.argv[3]
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
    if (
        not path.is_file()
        or path.name == "WP7_SOURCE_MANIFEST.json"
    ):
        continue
    files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_mode & 0o222:
        writable_paths.append(str(path.relative_to(root)))
assert files == manifest["file_hashes"]
assert writable_paths == []
unsigned = {key: value for key, value in manifest.items() if key != "source_sha256"}
actual = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
assert actual == manifest["source_sha256"]
print(json.dumps({
    "source_snapshot_verified": True,
    "source_write_bits_absent": True,
    "file_count": manifest["file_count"],
    "source_sha256": actual,
}, sort_keys=True))
PY

verify_source_snapshot() {
  python - "$SRC" "$RUN_ROOT/SOURCE_POSTRUN_VERIFICATION.json" \
    "$EXPECTED_SOURCE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report_path = Path(sys.argv[2])
expected_source_sha256 = sys.argv[3]
manifest = json.loads((root / "WP7_SOURCE_MANIFEST.json").read_text())
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
    files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_mode & 0o222:
        writable_paths.append(str(path.relative_to(root)))
unchanged = files == manifest["file_hashes"] and not writable_paths
added = sorted(set(files) - set(manifest["file_hashes"]))
removed = sorted(set(manifest["file_hashes"]) - set(files))
changed = sorted(
    path
    for path in set(files) & set(manifest["file_hashes"])
    if files[path] != manifest["file_hashes"][path]
)
payload = {
    "schema": "decision_admissibility_wp7_source_postrun_verification_v1",
    "source_sha256": manifest["source_sha256"],
    "unchanged": unchanged,
    "added_paths": added,
    "removed_paths": removed,
    "changed_paths": changed,
    "pycache_paths": sorted(
        path for path in files if "__pycache__" in Path(path).parts
    ),
    "writable_paths": writable_paths,
}
report_path.write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
assert unchanged, payload
print(json.dumps(payload, sort_keys=True))
PY
}

python - "$PVC_ROOT" "$FORMAL_BUNDLE_REL" "$CURRENT_REL" \
  "$EXPECTED_BUNDLE_ID" "$EXPECTED_MANIFEST_SHA256" "$RUN_ROOT" \
  "$SRC" <<'PY'
import inspect
import importlib.util
import json
import os
import sys
from pathlib import Path

from authority.memory_snapshot import (
    ImmutableBaseBundle,
    make_current_pointer,
    write_json_atomic,
)
from agents.memory.sop_visibility_gateway import SOPVisibilityGateway

root = Path(sys.argv[1]).resolve()
bundle = ImmutableBaseBundle.load(root / sys.argv[2], verify_artifacts=True)
current_path = root / sys.argv[3]
expected_id = sys.argv[4]
expected_manifest = sys.argv[5]
report_path = Path(sys.argv[6]) / "BUNDLE_PREFLIGHT.json"
source_root = Path(sys.argv[7]).resolve()
gateway_implementation_path = Path(
    inspect.getfile(SOPVisibilityGateway)
).resolve()
gateway_implementation_path.relative_to(source_root)
memory_snapshot_implementation_path = Path(
    sys.modules["authority.memory_snapshot"].__file__
).resolve()
memory_snapshot_implementation_path.relative_to(source_root)
module_origins = {}
for module_name in (
    "agents",
    "agents.memory.sop_visibility_gateway",
    "agents.memory.stage_aware_hybrid_memory",
    "engine",
    "engine.agent_search",
    "authority",
    "authority.adapters.mlevolve.runtime",
    "config",
):
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, module_name
    if spec.origin in {None, "namespace"}:
        raise AssertionError(
            f"Project module is not a regular staged package: {module_name}"
        )
    origin = Path(spec.origin).resolve()
    origin.relative_to(source_root)
    module_origins[module_name] = str(origin)
assert bundle.bundle_id == expected_id
assert bundle.manifest_sha256 == expected_manifest
assert bundle.manifest["certification_level"] == "certified"
split = json.loads((bundle.path / "splits" / "active.json").read_text())
graph = json.loads((bundle.path / "runforest" / "graph.json").read_text())
build_report = json.loads(
    (bundle.path / "runforest" / "build_report.json").read_text()
)
expected_sources = {
    "aptos2019-blindness-detection",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "leaf-classification",
    "plant-pathology-2020-fgvc7",
}
assert split["split_kind"] == "same-domain-task-heldout"
assert set(split["source_task_ids"]) == expected_sources
assert split["heldout_task_ids"] == ["aerial-cactus-identification"]
assert split["validation"]["cross_domain_source_run_count"] == 0
assert graph["meta"]["domain_scope_required"] is True
assert graph["meta"]["target_task_id"] == "aerial-cactus-identification"
assert graph["meta"]["target_domain"] == "image"
assert graph["meta"]["source_domains"] == ["image"]
assert build_report["cross_domain_included_clause_ids"] == []
assert build_report["all_included_clauses_have_domain_lineage"] is True
node_index = {
    str(node["id"]): node for node in graph.get("nodes") or []
}
gateway = SOPVisibilityGateway(node_index, mode="shadow")
assert len(gateway.clauses) == build_report["included_clause_count"]
assert not [
    clause_id
    for clause_id in gateway.clauses
    if clause_id.startswith("legacy::")
]
assert all(
    clause.source_run_ids
    and clause.source_task_ids
    and clause.source_domains == ("image",)
    and clause.transfer_scope == "same_domain"
    for clause in gateway.clauses.values()
)
certified_replay_clause_ids = sorted(
    clause_id
    for clause_id, clause in gateway.clauses.items()
    if clause.publication_class == "certified"
    and clause.legacy_status == "clean_replay_certified_v1"
)
assert certified_replay_clause_ids == [
    os.environ["EXPECTED_REPLAY_CLAUSE_ID"]
]
certified_clause = gateway.clauses[certified_replay_clause_ids[0]]
assert certified_clause.permitted_operations == ("generate_candidate",)
assert set(certified_clause.permitted_generation_stages) == {
    "draft",
    "model_design",
    "improve",
    "evolution",
    "fusion",
}
pointer = make_current_pointer(
    bundle_path=str(bundle.path.relative_to(root)),
    manifest=bundle.manifest,
    parent_bundle=bundle.manifest.get("parent_bundle"),
)
write_json_atomic(current_path, pointer)
report = {
    "schema": "decision_admissibility_wp7_bundle_preflight_v1",
    "bundle_id": bundle.bundle_id,
    "bundle_manifest_sha256": bundle.manifest_sha256,
    "bundle_path": str(bundle.path),
    "current_path": str(current_path),
    "current_pointer_sha256": pointer["pointer_sha256"],
    "artifact_count": len(bundle.manifest.get("artifact_hashes") or {}),
    "split_id": split["split_id"],
    "split_kind": split["split_kind"],
    "transfer_design": split["allocation"]["transfer_design"],
    "target_task_id": split["allocation"]["target_task_id"],
    "target_domain": split["allocation"]["target_domain"],
    "source_task_ids": sorted(expected_sources),
    "source_domains": graph["meta"]["source_domains"],
    "cross_domain_included_clause_count": len(
        build_report["cross_domain_included_clause_ids"]
    ),
    "materialized_gateway_clause_count": len(gateway.clauses),
    "materialized_legacy_fallback_clause_count": 0,
    "all_materialized_clauses_have_domain_lineage": True,
    "gateway_implementation_path": str(gateway_implementation_path),
    "memory_snapshot_implementation_path": str(
        memory_snapshot_implementation_path
    ),
    "all_repo_imports_bound_to_staged_source": True,
    "module_origins": dict(sorted(module_origins.items())),
}
write_json_atomic(report_path, report)
print(json.dumps(report, sort_keys=True))
PY

export HF_HOME="$PVC_ROOT/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export MLEVOLVE_CONFIG="$SRC/mlevolve/config/config_run_forest_stage_hybrid.yaml"
export MLEVOLVE_CODE_REVISION="$EXPECTED_HEAD"
MLEVOLVE_CODE_WORKTREE_SHA256="$(python - "$SRC/WP7_SOURCE_MANIFEST.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["source_sha256"])
PY
)"
export MLEVOLVE_CODE_WORKTREE_SHA256

# Load solver credentials into the process environment without adding a secret
# symlink to the hash-bound source snapshot.
set -a
. "$PVC_ROOT/nautilus/mlevolve/.env"
set +a
export DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL
test -n "${DEEPSEEK_API_KEY:-}"
test -n "${DEEPSEEK_BASE_URL:-}"
test -n "${DEEPSEEK_MODEL:-}"

cd "$SRC"
python - "$RUN_ROOT" <<'PY'
import importlib.util
import json
import sys
from pathlib import Path

assert importlib.util.find_spec("zstandard") is None
payload = {
    "schema": "decision_admissibility_wp7_pod_preflight_deselection_v1",
    "deselected_test": (
        "tests/test_memory_bundle_validation.py::"
        "test_tar_zst_export_is_deterministic_and_contains_manifest"
    ),
    "reason": "optional zstandard package absent from dev Pod image",
    "scope": "deterministic tar.zst export only",
    "bundle_validation_still_required": True,
    "domain_split_and_lineage_tests_still_required": True,
    "local_authority_suite_evidence": "151 passed in 21.06s",
    "local_plan_baseline_suite_evidence": "349 passed in 99.07s",
    "local_full_suite_evidence": "485 passed in 161.32s",
}
(Path(sys.argv[1]) / "PREFLIGHT_DESELECTION.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY
printf 'preflight_running\n' > "$RUN_ROOT/STATE"
set +e
python -m pytest -q -p no:cacheprovider \
  --deselect=tests/test_memory_bundle_validation.py::test_tar_zst_export_is_deterministic_and_contains_manifest \
  tests/authority \
  tests/test_corpus_split_isolation.py \
  tests/test_run_forest_bundle_v2.py \
  tests/test_memory_bundle_validation.py \
  tests/test_memory_snapshot_overlay.py \
  tests/test_run_identity.py \
  tests/test_stage_aware_hybrid_memory.py::test_rendered_sop_only_prompt_exposures_are_returned_as_refs \
  tests/test_sleep_time_bundle_publication.py \
  tests/test_bundle_publication_crash_safety.py \
  2>&1 | tee "$RUN_ROOT/PREFLIGHT_TESTS.log"
preflight_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$preflight_rc" > "$RUN_ROOT/PREFLIGHT_EXIT_CODE"
if [[ "$preflight_rc" -ne 0 ]]; then
  printf 'failed_preflight\n' > "$RUN_ROOT/STATE"
  verify_source_snapshot
  exit "$preflight_rc"
fi
printf 'preflight_passed\n' > "$RUN_ROOT/STATE"

cd "$SRC/mlevolve"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  | tee "$RUN_ROOT/GPU_IDENTITY.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$RUN_ROOT/STARTED_AT"
printf 'running\n' > "$RUN_ROOT/STATE"

set +e
timeout --foreground --signal=TERM --kill-after=30s 6500s \
  python -u run.py \
    exp_id="$TASK" \
    exp_name="wp7-shadow-certified-aerial-r20" \
    dataset_dir="$PVC_ROOT/nautilus/mlevolve/data" \
    data_dir="$PVC_ROOT/nautilus/mlevolve/data/$TASK/prepared/public" \
    desc_file="$PVC_ROOT/nautilus/mlevolve/data/$TASK/prepared/public/description.md" \
    log_dir="$RUN_ROOT/runs" \
    workspace_dir="$WORKSPACE_ROOT" \
    evaluation_authority.mode=shadow \
    evaluation_authority.rollout_id="$ROLLOUT_ID" \
    evaluation_authority.collector_version=2 \
    evaluation_authority.require_bound_bundle=true \
    evaluation_authority.expected_bundle_id="$EXPECTED_BUNDLE_ID" \
    evaluation_authority.expected_bundle_manifest_sha256="$EXPECTED_MANIFEST_SHA256" \
    external_skill_memory.bundle_root="$PVC_ROOT" \
    external_skill_memory.current_pointer_path="$CURRENT_REL" \
    external_skill_memory.retrieval_control=stage_hybrid \
    agent.draft_role_policy.roles='[coldstart_baseline,memory_transfer,novel_exploration]' \
    run_identity.experiment_group=wp7_shadow_certified_image_task_heldout_aerial_r20 \
    run_identity.baseline_reference_group=baseline_no_external_memory \
    agent.steps=12 \
    agent.time_limit=6000 \
    agent.seed=314159 \
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
  -name '*_wp7-shadow-certified-aerial-r20' | LC_ALL=C sort | tail -n 1)"
test -n "$latest_run"
logs="$latest_run"
test -s "$logs/authority_events.jsonl"
test -s "$logs/authority_rollout_report.json"
test -s "$logs/authority_snapshot.json"
test -s "$logs/run_identity.json"
test -s "$logs/config.yaml"

cd "$SRC"
python paper-skills/memory_bundle/build_shadow_review_packet.py \
  --ledger "$logs/authority_events.jsonl" \
  --packet "$RUN_ROOT/shadow_review_packet.json" \
  --max-records 50

python - "$logs" "$RUN_ROOT" "$FORMAL_BUNDLE" \
  "$EXPECTED_BUNDLE_ID" "$EXPECTED_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from authority.ledger import AuthorityLedger
from authority.memory_snapshot import ImmutableBaseBundle, write_json_atomic
from authority.rollout import load_shadow_records_from_ledger

logs = Path(sys.argv[1])
run_root = Path(sys.argv[2])
formal_bundle = Path(sys.argv[3])
expected_id = sys.argv[4]
expected_manifest = sys.argv[5]
ledger = AuthorityLedger(logs / "authority_events.jsonl")
assert ledger.verify()
records = load_shadow_records_from_ledger(ledger.path)
assert records
rollout = json.loads((logs / "authority_rollout_report.json").read_text())
identity = json.loads((logs / "run_identity.json").read_text())
packet = json.loads((run_root / "shadow_review_packet.json").read_text())
versions = rollout["rollout_versions"]
assert rollout["mode"] == "shadow"
assert rollout["record_count"] == len(records)
assert rollout["enforced_record_count"] == 0
assert versions["bundle_id"] == expected_id
assert versions["bundle_manifest_sha256"] == expected_manifest
assert identity["memory_snapshot_sha256"] == expected_manifest
ImmutableBaseBundle.load(formal_bundle, verify_artifacts=True).assert_unchanged()

expected_source_tasks = {
    "aptos2019-blindness-detection",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "leaf-classification",
    "plant-pathology-2020-fgvc7",
}
exposure_events = []
for line in (logs / "authority_events.jsonl").read_text().splitlines():
    event = json.loads(line)
    if event.get("event_type") == "experience_exposed":
        exposure_events.append(event["payload"])
assert exposure_events
invalid_exposures = []
for payload in exposure_events:
    source_tasks = set(payload.get("source_task_ids") or [])
    source_domains = set(payload.get("source_domains") or [])
    target_scope = payload.get("target_scope") or {}
    reasons = []
    if not payload.get("clause_id"):
        reasons.append("missing_clause_id")
    if not payload.get("sop_id"):
        reasons.append("missing_sop_id")
    if not payload.get("source_refs"):
        reasons.append("missing_source_refs")
    if not payload.get("source_run_ids"):
        reasons.append("missing_source_run_ids")
    if not source_tasks or not source_tasks <= expected_source_tasks:
        reasons.append("source_task_outside_reviewed_image_allowlist")
    if payload.get("transfer_scope") == "same_domain":
        if source_domains != {"image"} or target_scope.get("domain") != "image":
            reasons.append("same_domain_mismatch")
    elif payload.get("transfer_scope") != "domain_general":
        reasons.append("invalid_transfer_scope")
    if target_scope.get("task_id") != "aerial-cactus-identification":
        reasons.append("wrong_target_task")
    if reasons:
        invalid_exposures.append(
            {
                "contract_id": payload.get("contract_id"),
                "clause_id": payload.get("clause_id"),
                "reasons": reasons,
            }
        )
assert not invalid_exposures, invalid_exposures
exposure_audit = {
    "schema": "decision_admissibility_wp7_exposure_domain_audit_v1",
    "target_task_id": "aerial-cactus-identification",
    "target_domain": "image",
    "transfer_design": "same_domain_different_task_task_heldout",
    "reviewed_source_task_ids": sorted(expected_source_tasks),
    "exposure_event_count": len(exposure_events),
    "unique_contract_count": len(
        {payload["contract_id"] for payload in exposure_events}
    ),
    "exposed_clause_ids": sorted(
        {payload["clause_id"] for payload in exposure_events}
    ),
    "exposed_sop_ids": sorted(
        {payload["sop_id"] for payload in exposure_events}
    ),
    "exposed_source_task_ids": sorted(
        {
            task
            for payload in exposure_events
            for task in payload.get("source_task_ids") or []
        }
    ),
    "exposed_source_domains": sorted(
        {
            domain
            for payload in exposure_events
            for domain in payload.get("source_domains") or []
        }
    ),
    "same_domain_exposure_count": sum(
        payload.get("transfer_scope") == "same_domain"
        for payload in exposure_events
    ),
    "domain_general_exposure_count": sum(
        payload.get("transfer_scope") == "domain_general"
        for payload in exposure_events
    ),
    "cross_domain_or_unscoped_exposure_count": len(invalid_exposures),
    "invalid_exposures": invalid_exposures,
}
write_json_atomic(run_root / "EXPOSURE_DOMAIN_AUDIT.json", exposure_audit)

internal_error_count = sum(
    count
    for taxonomy, count in rollout["taxonomy_counts"].items()
    if "internal_error" in taxonomy
)
assert internal_error_count == 0, rollout["taxonomy_counts"]

expected_scope_receipt_types = {
    "split_lineage",
    "fit_scope",
    "prediction_scope",
    "evaluator",
    "selection_freeze",
}
scope_receipts_by_artifact = {}
scope_receipt_count = 0
for line in (logs / "authority_events.jsonl").read_text().splitlines():
    event = json.loads(line)
    if event.get("event_type") != "receipt_written":
        continue
    receipt = event.get("payload") or {}
    receipt_type = str(receipt.get("receipt_type") or "")
    if (
        receipt.get("trust_status") != "trusted_host"
        or receipt_type not in expected_scope_receipt_types
    ):
        continue
    assert receipt.get("collector_version") == "2", receipt
    assert receipt.get("collector_id", "").startswith("host."), receipt
    evidence = (receipt.get("payload") or {}).get("protocol_evidence") or {}
    assert evidence.get("schema") == (
        "mlevolve_host_protocol_evidence_binding_v2"
    ), receipt
    assert evidence.get("evidence_level") == (
        "host_runtime_argument_result_scope_trace_plus_"
        "deterministic_static_audit"
    ), receipt
    assert len(str(evidence.get("scope_binding_sha256") or "")) == 64
    scope_receipts_by_artifact.setdefault(
        str(receipt.get("artifact_id")), set()
    ).add(receipt_type)
    scope_receipt_count += 1
complete_scope_artifacts = sorted(
    artifact_id
    for artifact_id, receipt_types in scope_receipts_by_artifact.items()
    if receipt_types == expected_scope_receipt_types
)
assert complete_scope_artifacts, scope_receipts_by_artifact
scope_audit = {
    "schema": "decision_admissibility_wp7_runtime_scope_receipt_audit_v1",
    "observer_schema": "mlevolve_host_protocol_observation_v2",
    "collector_version": "2",
    "trusted_scope_receipt_count": scope_receipt_count,
    "trusted_scope_artifact_count": len(scope_receipts_by_artifact),
    "complete_scope_artifact_ids": complete_scope_artifacts,
    "required_receipt_types": sorted(expected_scope_receipt_types),
    "all_trusted_scope_receipts_bind_runtime_arguments_or_results": True,
}
write_json_atomic(run_root / "RUNTIME_SCOPE_RECEIPT_AUDIT.json", scope_audit)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

remediation_provenance = json.loads(
    (run_root / "REMEDIATION_PROVENANCE.json").read_text()
)
summary = {
    "schema": "decision_admissibility_wp7_shadow_run_summary_v1",
    "status": "shadow_complete_review_pending",
    "run_dir": str(logs),
    "rollout_id": versions["rollout_id"],
    "rollout_version_hash": versions["version_hash"],
    "bundle_id": versions["bundle_id"],
    "bundle_manifest_sha256": versions["bundle_manifest_sha256"],
    "transfer_design": "same_domain_different_task_task_heldout",
    "target_task_id": "aerial-cactus-identification",
    "target_domain": "image",
    "source_task_ids": sorted(expected_source_tasks),
    "exposure_event_count": exposure_audit["exposure_event_count"],
    "unique_exposure_contract_count": exposure_audit[
        "unique_contract_count"
    ],
    "exposed_source_task_ids": exposure_audit[
        "exposed_source_task_ids"
    ],
    "exposed_source_domains": exposure_audit["exposed_source_domains"],
    "cross_domain_or_unscoped_exposure_count": exposure_audit[
        "cross_domain_or_unscoped_exposure_count"
    ],
    "internal_error_count": internal_error_count,
    "trusted_scope_receipt_count": scope_receipt_count,
    "complete_scope_artifact_count": len(complete_scope_artifacts),
    "record_count": len(records),
    "taxonomy_counts": rollout["taxonomy_counts"],
    "disagreement_count": rollout["disagreement_count"],
    "review_population_count": packet["population_count"],
    "review_sample_count": packet["sample_count"],
    "review_evidence_hash": packet["evidence_hash"],
    "authority_ledger_sha256": sha256(logs / "authority_events.jsonl"),
    "rollout_report_sha256": sha256(logs / "authority_rollout_report.json"),
    "review_packet_sha256": sha256(run_root / "shadow_review_packet.json"),
    "exposure_domain_audit_sha256": sha256(
        run_root / "EXPOSURE_DOMAIN_AUDIT.json"
    ),
    "runtime_scope_receipt_audit_sha256": sha256(
        run_root / "RUNTIME_SCOPE_RECEIPT_AUDIT.json"
    ),
    "formal_bundle_verified_unchanged": True,
    "source_sha256": remediation_provenance["new_source_sha256"],
    "previous_shadow_run": str(
        run_root.parent / "decision-admissibility-wp7-shadow-aerial-r12"
    ),
    "previous_diagnostic_run": str(
        run_root.parent / "decision-admissibility-wp7-shadow-aerial-r13"
    ),
    "remediation_provenance_sha256": sha256(
        run_root / "REMEDIATION_PROVENANCE.json"
    ),
}
write_json_atomic(run_root / "SHADOW_RUN_SUMMARY.json", summary)
print(json.dumps(summary, sort_keys=True))
PY

verify_source_snapshot
printf 'shadow_complete_review_pending\n' > "$RUN_ROOT/STATE"
touch "$RUN_ROOT/SHADOW_COMPLETE"
printf 'wp7_shadow_finish=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
