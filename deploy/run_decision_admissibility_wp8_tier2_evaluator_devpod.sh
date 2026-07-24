#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC="${WP8_TIER2_SOURCE_ROOT:-/opt/nautilus}"
OUTPUT="${WP8_TIER2_OUTPUT_ROOT:-/output}"
EXPECTED_SOURCE_SHA256="${WP8_TIER2_EXPECTED_SOURCE_SHA256:?WP8_TIER2_EXPECTED_SOURCE_SHA256 is required}"
EXPECTED_TRAIN_MANIFEST_SHA256="${WP8_TIER2_EXPECTED_TRAIN_MANIFEST_SHA256:-a7968c2bd3ee254f6e70d6b360bc9362aafc4449b5985078d508ecb3b8de64ea}"
EXPECTED_EVALUATOR_MANIFEST_SHA256="${WP8_TIER2_EXPECTED_EVALUATOR_MANIFEST_SHA256:-dac23ecfeb625e7a2cd650a710012f0028f83cf93a873d227356622fc4c1cf55}"
EXPECTED_LABELS_SHA256="${WP8_TIER2_EXPECTED_LABELS_SHA256:-49b835b26ac6bae926fbd92ed508e8d9eb87989366b2cc76342b67ccde38b676}"
DELETION_ATTESTATION="$OUTPUT/TRAINING_POD_DELETION_ATTESTATION.json"

test -f "$OUTPUT/TRAINING_COMPLETE"
test "$(cat "$OUTPUT/STATE")" = "training_complete_unscored"
test -s "$OUTPUT/TRAINING_MANIFEST.json"
test -s "$DELETION_ATTESTATION"
test -f /fixed/train_view/fixed_holdout_manifest.json
test -f /fixed/evaluator_view/fixed_holdout_manifest.json
test -f /fixed/evaluator_view/labels.csv
test ! -e /secrets/mlevolve.env
test ! -e /memory/CURRENT.json
test ! -e "$OUTPUT/EVALUATION_COMPLETE"

exec > >(tee -a "$OUTPUT/evaluator_launcher.log") 2>&1
printf 'evaluation_running\n' > "$OUTPUT/STATE"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT/EVALUATION_STARTED_AT"

finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$OUTPUT/EVALUATOR_LAUNCHER_EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then
    printf 'evaluation_failed\n' > "$OUTPUT/STATE"
  fi
  exit "$rc"
}
term() {
  trap - TERM
  printf 'TERM\n' > "$OUTPUT/EVALUATOR_SIGNAL"
  exit 143
}
trap finish EXIT
trap term TERM

export PYTHONPATH="$SRC:$SRC/mlevolve"

python - "$SRC" "$OUTPUT" "$EXPECTED_SOURCE_SHA256" \
  "$EXPECTED_TRAIN_MANIFEST_SHA256" "$EXPECTED_EVALUATOR_MANIFEST_SHA256" \
  "$EXPECTED_LABELS_SHA256" "$DELETION_ATTESTATION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from fixed_holdout.common import read_manifest

source, output = Path(sys.argv[1]), Path(sys.argv[2])
expected_source, expected_train, expected_eval, expected_labels = sys.argv[3:7]
deletion_path = Path(sys.argv[7])
source_manifest = json.loads((source / "WP8_TIER2_SOURCE_MANIFEST.json").read_text())
unsigned = {
    key: value for key, value in source_manifest.items() if key != "source_sha256"
}
assert source_manifest["source_sha256"] == expected_source
assert hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest() == expected_source
train_path = Path("/fixed/train_view/fixed_holdout_manifest.json")
eval_path = Path("/fixed/evaluator_view/fixed_holdout_manifest.json")
labels_path = Path("/fixed/evaluator_view/labels.csv")
assert hashlib.sha256(train_path.read_bytes()).hexdigest() == expected_train
assert hashlib.sha256(eval_path.read_bytes()).hexdigest() == expected_eval
assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == expected_labels
train = read_manifest(train_path, expected_role="train_view")
evaluator = read_manifest(eval_path, expected_role="evaluator_view")
for key in ("task_id", "split_id", "public_tree_sha256", "holdout_id_sha256", "selection_policy"):
    assert train[key] == evaluator[key]
assert train["hidden_labels_present"] is False
deletion = json.loads(deletion_path.read_text())
assert deletion.get("schema") == "decision_admissibility_wp8_tier2_training_pod_deletion_v1"
assert deletion.get("training_pod_absent_before_evaluation") is True
assert deletion.get("training_process_complete_before_deletion") is True

mount_points = sorted({
    line.split(" - ", 1)[0].split()[4]
    for line in Path("/proc/self/mountinfo").read_text().splitlines()
})
assert {"/opt/nautilus", "/fixed/train_view", "/fixed/evaluator_view", "/output"} <= set(mount_points)
assert "/workspace" not in mount_points
assert "/memory" not in mount_points
report = {
    "schema": "decision_admissibility_wp8_tier2_evaluator_isolation_v1",
    "independent_cpu_evaluator": True,
    "training_pod_absent_before_evaluation": True,
    "whole_workspace_mounted": False,
    "solver_secret_mounted": False,
    "memory_bundle_mounted": False,
    "train_manifest_sha256": expected_train,
    "evaluator_manifest_sha256": expected_eval,
    "labels_sha256": expected_labels,
    "source_sha256": expected_source,
    "mount_points": mount_points,
}
(output / "EVALUATOR_ISOLATION.json").write_text(
    json.dumps(report, sort_keys=True, indent=2) + "\n"
)
PY

python - "$OUTPUT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "TRAINING_MANIFEST.json").read_text())
unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
assert manifest["manifest_hash"] == hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
assert manifest["status"] == "training_complete_unscored"
assert manifest["terminal_scores_visible_during_search"] is False
assert manifest["formal_tier2_evidence"] is False
assert manifest["steps_per_condition"] == 6
assert manifest["initial_drafts_per_condition"] == 3
assert manifest["repair_steps_budget_per_condition"] == 3
assert manifest["same_candidate_execution_contract"] is True
assert manifest["candidate_execution_contract_host_enforced"] is True
contract = manifest["candidate_execution_contract"]
assert contract == json.loads((root / "CANDIDATE_EXECUTION_CONTRACT.json").read_text())
assert contract["enabled"] is True
assert contract["max_execution_seconds"] == 600
assert contract["max_epochs"] == 8
assert contract["max_cv_folds"] == 1
assert contract["max_trainable_models"] == 1
assert contract["allow_remote_assets"] is False
assert contract["allow_unverified_local_assets"] is False
assert contract["allow_dataset_wide_per_sample_precompute"] is False
assert contract["allow_source_score_inheritance"] is False
environment = json.loads((root / "CANDIDATE_EXECUTION_ENVIRONMENT.json").read_text())
assert environment["all_allowed_import_roots_importable"] is True
assert environment["allowed_import_roots"] == contract["allowed_import_roots"]
for condition in ("nm", "full"):
    row = manifest["conditions"][condition]
    assert row["pre_evaluator_score_file_count"] == 0
    assert row["candidate_execution_contract_hash"] == contract["contract_hash"]
    assert row["candidate_execution_contract_role_binding_valid"] is True
    assert row["candidate_execution_audit_count"] == manifest["steps_per_condition"]
    assert row["candidate_execution_audits_integrity_valid"] is True
    assert row["candidate_execution_denials_enforced"] is True
    assert (
        row["candidate_execution_admitted_count"]
        + row["candidate_execution_denied_count"]
        == manifest["steps_per_condition"]
    )
    assert row["candidate_execution_block_receipt_count"] == row[
        "candidate_execution_denied_count"
    ]
    assert row["candidate_execution_submitted_admitted_count"] > 0
    assert row["candidate_execution_submitted_admitted_count"] == row[
        "submission_count"
    ]
    admitted = set(row["candidate_execution_admitted_node_ids"])
    denied = set(row["candidate_execution_denied_node_ids"])
    submitted = set(row["candidate_execution_submitted_node_ids"])
    assert admitted.isdisjoint(denied)
    assert admitted | denied
    assert submitted and submitted <= admitted
    assert submitted.isdisjoint(denied)
    assert not list(Path(row["run_dir"]).glob("fixed_holdout_scores*.json"))
    for relative, digest in row["file_hashes"].items():
        path = root / relative
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
PY

score_condition() {
  local condition="$1"
  local request journal submissions run_dir
  request="$(python - "$OUTPUT/TRAINING_MANIFEST.json" "$condition" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["conditions"][sys.argv[2]]["evaluation_request_path"])
PY
)"
  journal="$(python - "$OUTPUT/TRAINING_MANIFEST.json" "$condition" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["conditions"][sys.argv[2]]["journal_path"])
PY
)"
  submissions="$(python - "$OUTPUT/TRAINING_MANIFEST.json" "$condition" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["conditions"][sys.argv[2]]["submission_dir"])
PY
)"
  run_dir="$(dirname "$request")"
  test ! -e "$run_dir/fixed_holdout_scores.json"
  cd "$SRC/mlevolve"
  python -m fixed_holdout.score_run \
    --manifest /fixed/evaluator_view/fixed_holdout_manifest.json \
    --submission-dir "$submissions" \
    --journal "$journal" \
    --evaluation-request "$request" \
    --output "$run_dir/fixed_holdout_scores.json" \
    --finalize-writeback \
    > "$run_dir/fixed_holdout_evaluator_stdout.log"
}

score_condition nm
score_condition full

python - "$SRC" "$OUTPUT" <<'PY'
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

from authority.ledger import AuthorityLedger
from authority.memory_snapshot import SessionOverlay

source, root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
training = json.loads((root / "TRAINING_MANIFEST.json").read_text())
conditions = {}
for condition in ("nm", "full"):
    train_row = training["conditions"][condition]
    run_dir = Path(train_row["run_dir"])
    request = json.loads(Path(train_row["evaluation_request_path"]).read_text())
    report_path = run_dir / "fixed_holdout_scores.json"
    status_path = run_dir / "fixed_holdout_writeback_status.json"
    report = json.loads(report_path.read_text())
    status = json.loads(status_path.read_text())
    assert report["report_schema"] == "fixed_holdout_terminal_score_report_v2"
    assert report["terminal_score_sealed"] is True
    assert report["candidate_set_frozen_before_scoring"] is True
    assert report["scores_were_visible_during_search"] is False
    assert report["selection_policy"] == "terminal_only"
    assert report["metric"] == "binary_roc_auc"
    assert report["best_node_id"]
    assert isinstance(report["best_score"], (int, float)) and math.isfinite(report["best_score"])
    assert 0.0 <= report["best_score"] <= 1.0
    result_node_ids = {str(row.get("node_id")) for row in report["results"]}
    assert result_node_ids <= set(
        train_row["candidate_execution_submitted_node_ids"]
    )
    assert not result_node_ids & set(
        train_row["candidate_execution_denied_node_ids"]
    )
    assert status["schema"] == "fixed_holdout_terminal_writeback_status_v1"
    assert status["status"] == "complete"
    assert status["completion"] in {"finalized", "already_finalized"}
    descriptor = request["authority_writeback"]
    ledger = AuthorityLedger(descriptor["authority_ledger_path"])
    assert ledger.verify()
    overlay = SessionOverlay(
        descriptor["session_overlay_path"],
        overlay_id=descriptor["session_overlay_id"],
    )
    result_events = [
        event for event in overlay.events()
        if event.event_type == "memory_claim"
        and event.payload.get("publication_class") == "result_fact"
    ]
    assert len(result_events) == 1, (condition, result_events)
    result_payload = result_events[0].payload
    assert result_payload.get("derived_from_refs") == []
    assert result_payload.get("artifact_id") == report["best_node_id"]
    conditions[condition] = {
        "condition": condition,
        "metric": report["metric"],
        "best_score": report["best_score"],
        "best_node_id": report["best_node_id"],
        "scored_candidate_count": sum(row["status"] == "scored" for row in report["results"]),
        "rejected_candidate_count": sum(row["status"] == "rejected" for row in report["results"]),
        "score_report_hash": report["report_hash"],
        "score_report_file_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "writeback_status_hash": status["status_hash"],
        "writeback_status_file_sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
        "result_fact_count": 1,
        "result_fact_derived_from_refs": [],
        "authority_ledger_valid_after_writeback": True,
        "experience_exposure_count": train_row["experience_exposure_count"],
        "candidate_execution_admitted_count": train_row[
            "candidate_execution_admitted_count"
        ],
        "candidate_execution_denied_count": train_row[
            "candidate_execution_denied_count"
        ],
        "candidate_execution_runtime_failed_admitted_count": train_row[
            "candidate_execution_runtime_failed_admitted_count"
        ],
        "candidate_execution_submitted_admitted_count": train_row[
            "candidate_execution_submitted_admitted_count"
        ],
        "exposure_audit_path": train_row["exposure_audit_path"],
    }

summary = {
    "schema": "decision_admissibility_wp8_tier2_canary_evaluation_v1",
    "status": "canary_evaluation_complete",
    "task_id": training["task_id"],
    "protocol_ref": training["protocol_ref"],
    "fixed_holdout_metric": training["fixed_holdout_metric"],
    "conditions": conditions,
    "training_condition_order": training["condition_order"],
    "same_host_seed": training["same_host_seed"],
    "steps_per_condition": training["steps_per_condition"],
    "initial_drafts_per_condition": training["initial_drafts_per_condition"],
    "repair_steps_budget_per_condition": training[
        "repair_steps_budget_per_condition"
    ],
    "host_owned_terminal_evaluator": True,
    "training_pod_absent_before_evaluation": True,
    "scores_used_for_further_search": False,
    "effect_claim_authorized": False,
    "full_superiority_claim_authorized": False,
    "large_scale_tier2_authorized": False,
    "formal_tier2_evidence": False,
    "summary_hash": "",
}
unsigned = {key: value for key, value in summary.items() if key != "summary_hash"}
summary["summary_hash"] = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
(root / "CANARY_EVALUATION_SUMMARY.json").write_text(
    json.dumps(summary, sort_keys=True, indent=2) + "\n"
)

packet = root / "evidence_packet"
assert not packet.exists()
packet.mkdir()
copies = {
    source / "WP8_TIER2_SOURCE_MANIFEST.json": "source_snapshot_manifest.json",
    root / "SOURCE_PREFLIGHT.json": "source_preflight.json",
    root / "SOURCE_POSTRUN.json": "source_postrun.json",
    root / "TRAINING_ISOLATION.json": "training_isolation.json",
    root / "EVALUATOR_ISOLATION.json": "evaluator_isolation.json",
    root / "PROVIDER_ATTESTATION.json": "provider_attestation.json",
    root / "CANDIDATE_EXECUTION_ENVIRONMENT.json": "candidate_execution_environment.json",
    root / "CANDIDATE_EXECUTION_CONTRACT.json": "candidate_execution_contract.json",
    root / "GPU_IDENTITY.txt": "gpu_identity.txt",
    root / "TRAINING_MANIFEST.json": "training_manifest.json",
    root / "CANARY_EVALUATION_SUMMARY.json": "canary_evaluation_summary.json",
    root / "TRAINING_POD_DELETION_ATTESTATION.json": "training_pod_deletion_attestation.json",
    Path("/fixed/train_view/fixed_holdout_manifest.json"): "train_manifest.json",
    Path("/fixed/evaluator_view/fixed_holdout_manifest.json"): "evaluator_manifest.json",
}
for condition in ("nm", "full"):
    row = training["conditions"][condition]
    run_dir = Path(row["run_dir"])
    copies.update({
        Path(row["journal_path"]): f"{condition}/journal.json",
        Path(row["evaluation_request_path"]): f"{condition}/evaluation_request.json",
        Path(row["exposure_audit_path"]): f"{condition}/exposure_audit.json",
        run_dir / "authority_rollout_report.json": f"{condition}/authority_rollout_report.json",
        run_dir / "authority_events.jsonl": f"{condition}/authority_events.jsonl",
        run_dir / "fixed_holdout_scores.json": f"{condition}/fixed_holdout_scores.json",
        run_dir / "fixed_holdout_writeback_status.json": f"{condition}/fixed_holdout_writeback_status.json",
    })
    descriptor = json.loads(Path(row["evaluation_request_path"]).read_text())["authority_writeback"]
    overlay_root = Path(descriptor["session_overlay_path"])
    for name in ("manifest.json", "events.jsonl"):
        candidate = overlay_root / name
        if candidate.is_file():
            copies[candidate] = f"{condition}/session_overlay_{name}"
    for candidate in row["candidate_execution_audit_paths"]:
        audit_path = Path(candidate)
        copies[audit_path] = (
            f"{condition}/candidate_execution_audits/{audit_path.name}"
        )
    for candidate in row["candidate_execution_block_receipt_paths"]:
        block_path = Path(candidate)
        copies[block_path] = (
            f"{condition}/candidate_execution_block_receipts/{block_path.name}"
        )

for source_path, relative in copies.items():
    assert source_path.is_file(), source_path
    target = packet / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target)

assert not list(packet.rglob("labels.csv"))
assert not list(packet.rglob("*.env"))
file_hashes = {
    str(path.relative_to(packet)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted(packet.rglob("*")) if path.is_file()
}
manifest = {
    "schema": "decision_admissibility_wp8_tier2_canary_evidence_packet_v1",
    "file_count": len(file_hashes),
    "file_hashes": file_hashes,
    "hidden_labels_included": False,
    "solver_secrets_included": False,
    "packet_hash": "",
}
unsigned = {key: value for key, value in manifest.items() if key != "packet_hash"}
manifest["packet_hash"] = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
(packet / "EVIDENCE_PACKET_MANIFEST.json").write_text(
    json.dumps(manifest, sort_keys=True, indent=2) + "\n"
)
for path in packet.rglob("*"):
    if path.is_file():
        path.chmod(path.stat().st_mode & ~0o222)
for path in sorted(packet.rglob("*"), reverse=True):
    if path.is_dir():
        path.chmod(path.stat().st_mode & ~0o222)
packet.chmod(packet.stat().st_mode & ~0o222)
PY

date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT/EVALUATION_FINISHED_AT"
printf 'evaluation_complete\n' > "$OUTPUT/STATE"
touch "$OUTPUT/EVALUATION_COMPLETE"
printf '0\n' > "$OUTPUT/EVALUATOR_LAUNCHER_EXIT_CODE"
trap - EXIT
