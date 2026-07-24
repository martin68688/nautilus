from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = ROOT / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_tier2_formal_joint_inventory import (  # noqa: E402
    ANALYSIS_SPEC_SCHEMA,
    COMPLETED_FREEZE_SCHEMA,
    COMPLETED_GATE_SCHEMA,
    COMPLETED_PRECREATED_EMPTY_BLOCK_IDS,
    COMPLETED_ROOT_NAME,
    CONDITIONS,
    CONTINUATION_GATE_SCHEMA,
    CONTINUATION_ROOT_NAME,
    EXPECTED_BLOCKS,
    RECOVERY_DIAGNOSTIC_SCHEMA,
    RECOVERY_GATE_SCHEMA,
    REQUIRED_BLOCK_ARTIFACTS,
    STRUCTURE_AUDIT_SCHEMA,
    build_joint_inventory,
    compute_manifest,
)
from verify_tier2_formal_joint_inventory import (  # noqa: E402
    verify_joint_inventory,
)


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_hashed(
    path: Path, payload: dict[str, Any], field: str
) -> dict[str, Any]:
    payload[field] = ""
    payload[field] = _payload_hash(payload, field)
    _write_json(path, payload)
    return payload


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _block_root(
    completed_root: Path, continuation_root: Path, block_id: str
) -> Path:
    role = EXPECTED_BLOCKS[block_id]["root_role"]
    root = completed_root if role == "completed_r10" else continuation_root
    return root / "blocks" / block_id


def _write_opaque_block(root: Path, block_id: str) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=False)
    for filename in REQUIRED_BLOCK_ARTIFACTS:
        path = root / filename
        if filename == "EVALUATION_SUMMARY.json":
            # Deliberately invalid JSON. A compliant result-blind inventory can
            # hash it, but cannot deserialize or inspect the sentinel values.
            path.write_text(
                '{"selected_score": 991337.125, "best_score": 882244.5, '
                '"opaque": THIS_IS_NOT_JSON}',
                encoding="utf-8",
            )
        elif filename == "HOST_ORACLE.json":
            path.write_text(
                '{"best_score": 773355.75, "opaque": THIS_IS_NOT_JSON}',
                encoding="utf-8",
            )
        elif filename == "EVALUATION_COMPLETE":
            path.write_bytes(b"")
        else:
            path.write_bytes(f"opaque::{block_id}::{filename}\n".encode("utf-8"))
    # Exercise a recursive tree rather than only root-level files.
    nested = root / "conditions" / "no_memory" / "opaque.bin"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"not parsed\x00only hashed")
    return {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _condition_structure(successful: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for index, condition in enumerate(CONDITIONS):
        if index < successful:
            rows[condition] = {
                "status": "scored_selected_result",
                "result_fact_count": 1,
                "result_fact_derived_from_refs": [],
                "terminal_value_omitted": True,
            }
        else:
            rows[condition] = {
                "status": "pre_terminal_failure",
                "result_fact_count": 0,
                "failure_classification": "synthetic_retained_failure",
                "terminal_metric_observed": False,
            }
    return rows


def _make_structure_audit(
    path: Path,
    *,
    block_id: str,
    successful: int,
    file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    expected = EXPECTED_BLOCKS[block_id]
    artifact_bindings = {
        filename: {
            "file_sha256": file_hashes[filename],
            "internal_hash": _digest(f"internal::{block_id}::{filename}"),
        }
        for filename in (
            "TRAINING_MANIFEST.json",
            "TRAINING_POD_DELETION_ATTESTATION.json",
            "EVALUATION_SUMMARY.json",
            "BLOCK_EVALUATION_FILE_MANIFEST.json",
            "EVALUATOR_POD_DELETION_ATTESTATION.json",
        )
    }
    payload: dict[str, Any] = {
        "schema": STRUCTURE_AUDIT_SCHEMA,
        "status": "passed",
        "audited_at_utc": "2026-07-23T00:00:00Z",
        "block_id": block_id,
        "task_id": expected["task_id"],
        "agent_seed": expected["agent_seed"],
        "online_condition_count": 5,
        "successful_selected_result_count": successful,
        "failed_online_condition_count": 5 - successful,
        "result_fact_count": successful,
        "condition_structure": _condition_structure(successful),
        "oracle_structure": {
            "normal_result_fact_published": False,
            "score_values_omitted": True,
            "candidate_union_count": successful,
        },
        "lifecycle": {
            "training_pod_not_found": True,
            "evaluator_created_after_training_not_found_attestation": True,
            "evaluator_pod_not_found": True,
        },
        "structural_invariants": {
            "all_five_conditions_retained": True,
            "successful_plus_failed_equals_five": True,
            "successful_conditions_have_exactly_one_independent_result_fact": True,
            "failed_conditions_have_no_terminal_result_or_result_fact": True,
            "target_history_not_used": True,
            "terminal_values_remain_blinded": True,
        },
        "artifact_bindings": artifact_bindings,
        "score_values_included": False,
        "score_values_inspected": False,
        "audit_hash": "",
    }
    return _write_hashed(path, payload, "audit_hash")


def _fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    completed_root = tmp_path / COMPLETED_ROOT_NAME
    continuation_root = tmp_path / CONTINUATION_ROOT_NAME
    completed_root.mkdir()
    continuation_root.mkdir()
    (completed_root / "blocks").mkdir()
    (continuation_root / "blocks").mkdir()

    block_files: dict[str, dict[str, str]] = {}
    for block_id in EXPECTED_BLOCKS:
        block_files[block_id] = _write_opaque_block(
            _block_root(completed_root, continuation_root, block_id), block_id
        )
    for block_id in COMPLETED_PRECREATED_EMPTY_BLOCK_IDS:
        (completed_root / "blocks" / block_id).mkdir()

    spec_path = tmp_path / "analysis_spec.json"
    spec = {
        "schema": ANALYSIS_SPEC_SCHEMA,
        "status": "result_blind_frozen",
        "expected_blocks": [
            {"block_id": block_id, **EXPECTED_BLOCKS[block_id]}
            for block_id in sorted(EXPECTED_BLOCKS)
        ],
        "expected_online_condition_count": 45,
        "expected_oracle_disposition_count": 9,
        "score_values_may_be_inspected": False,
        "spec_hash": "",
    }
    _write_hashed(spec_path, spec, "spec_hash")

    completed_successes = {
        "wp8-tier2-formal-aerial-seed-104729-r3": 5,
        "wp8-tier2-formal-aerial-seed-130363-r3": 5,
        "wp8-tier2-formal-aerial-seed-155921-r3": 5,
        "wp8-tier2-formal-birds-seed-104729-r3": 3,
    }
    frozen_blocks: dict[str, dict[str, Any]] = {}
    for block_id, successful in completed_successes.items():
        files = block_files[block_id]
        expected = EXPECTED_BLOCKS[block_id]
        frozen_blocks[block_id] = {
            "task_id": expected["task_id"],
            "agent_seed": expected["agent_seed"],
            "evaluation_summary_hash": _digest(f"summary::{block_id}"),
            "evaluation_summary_file_sha256": files["EVALUATION_SUMMARY.json"],
            "successful_selected_result_count": successful,
            "failed_online_condition_count": 5 - successful,
            "result_fact_count": successful,
            "evaluator_pod_deletion_attestation_hash": _digest(
                f"delete::{block_id}"
            ),
            "evaluator_pod_deletion_file_sha256": files[
                "EVALUATOR_POD_DELETION_ATTESTATION.json"
            ],
            "block_evaluation_file_manifest_sha256": files[
                "BLOCK_EVALUATION_FILE_MANIFEST.json"
            ],
            "may_rerun": False,
        }
    completed_freeze_path = tmp_path / "completed_freeze.json"
    completed_freeze = {
        "schema": COMPLETED_FREEZE_SCHEMA,
        "status": "four_completed_blocks_frozen_before_continuation_staging",
        "frozen_at_utc": "2026-07-23T00:00:00Z",
        "completed_block_count": 4,
        "completed_online_condition_count": 20,
        "completed_oracle_count": 4,
        "successful_selected_result_count": sum(completed_successes.values()),
        "failed_online_condition_count": 20 - sum(completed_successes.values()),
        "result_fact_count": sum(completed_successes.values()),
        "blocks": frozen_blocks,
        "remaining_blocks": [],
        "continuation_requirements": {
            "remaining_block_count": 5,
            "remaining_online_condition_count": 25,
            "remaining_oracle_count": 5,
            "total_online_condition_count_after_completion": 45,
            "total_oracle_count_after_completion": 9,
        },
        "score_values_included": False,
        "score_values_inspected": False,
        "inventory_hash": "",
    }
    _write_hashed(completed_freeze_path, completed_freeze, "inventory_hash")

    completed_source = "a" * 64
    continuation_source = "b" * 64
    completed_gate_path = tmp_path / "completed_gate.json"
    completed_gate = {
        "schema": COMPLETED_GATE_SCHEMA,
        "status": "passed",
        "terminal_metric_observed": False,
        "evidence": {
            "source_verification": {
                "source_sha256": completed_source,
            }
        },
        "gate_hash": "",
    }
    _write_hashed(completed_gate_path, completed_gate, "gate_hash")

    continuation_gate_path = tmp_path / "continuation_gate.json"
    continuation_gate = {
        "schema": CONTINUATION_GATE_SCHEMA,
        "status": "passed",
        "terminal_score_values_inspected": False,
        "source_snapshot_sha256": continuation_source,
        "gate_hash": "",
    }
    _write_hashed(continuation_gate_path, continuation_gate, "gate_hash")

    recovery_gate_path = tmp_path / "recovery_gate.json"
    recovery_gate = {
        "schema": RECOVERY_GATE_SCHEMA,
        "status": "passed",
        "terminal_score_values_inspected": False,
        "source_snapshot_sha256": completed_source,
        "gate_hash": "",
    }
    _write_hashed(recovery_gate_path, recovery_gate, "gate_hash")

    recovery_diagnostic_path = tmp_path / "recovery_diagnostic.json"
    recovery_diagnostic = {
        "schema": RECOVERY_DIAGNOSTIC_SCHEMA,
        "status": "preserved_preterminal_finalizer_failure_pending_result_blind_recovery",
        "block_id": "wp8-tier2-formal-birds-seed-104729-r3",
        "task_id": "mlsp-2013-birds",
        "agent_seed": 104729,
        "frozen_execution": {
            "source_snapshot_sha256": completed_source,
            "staging_gate_hash": completed_gate["gate_hash"],
        },
        "preserved_output": {
            "terminal_score_file_count": 0,
            "terminal_metric_observed": False,
            "terminal_score_values_inspected": False,
        },
        "diagnostic_hash": "",
    }
    _write_hashed(recovery_diagnostic_path, recovery_diagnostic, "diagnostic_hash")

    continuation_successes = {
        "wp8-tier2-formal-birds-seed-130363-r5": 0,
        "wp8-tier2-formal-birds-seed-155921-r5": 3,
        "wp8-tier2-formal-taxi-seed-104729-r5": 0,
        "wp8-tier2-formal-taxi-seed-130363-r5": 0,
        "wp8-tier2-formal-taxi-seed-155921-r5": 1,
    }
    audit_paths: list[Path] = []
    # One completed audit checks completed-freeze/audit cross-binding. The five
    # continuation audits are mandatory semantic evidence.
    audited = {
        "wp8-tier2-formal-birds-seed-104729-r3": 3,
        **continuation_successes,
    }
    for block_id, successful in audited.items():
        path = tmp_path / f"audit-{block_id}.json"
        _make_structure_audit(
            path,
            block_id=block_id,
            successful=successful,
            file_hashes=block_files[block_id],
        )
        audit_paths.append(path)

    kwargs = {
        "completed_root": completed_root,
        "continuation_root": continuation_root,
        "analysis_spec_path": spec_path,
        "completed_freeze_path": completed_freeze_path,
        "completed_staging_gate_path": completed_gate_path,
        "continuation_staging_gate_path": continuation_gate_path,
        "recovery_gate_path": recovery_gate_path,
        "recovery_diagnostic_path": recovery_diagnostic_path,
        "structure_audit_paths": audit_paths,
        "created_at": "2026-07-23T08:00:00Z",
    }
    return {
        "kwargs": kwargs,
        "completed_root": completed_root,
        "continuation_root": continuation_root,
        "audit_paths": audit_paths,
    }


def test_joint_inventory_is_hash_only_score_free_and_recomputable(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "joint-inventory"
    report = build_joint_inventory(output_root=output_root, **fixture["kwargs"])

    assert report["status"] == "passed"
    assert report["score_policy"] == "hash_only"
    assert report["score_values_included"] is False
    assert report["score_values_inspected"] is False
    assert report["formal_block_json_parsed"] is False
    assert report["score_bearing_artifacts_parsed"] is False
    assert report["totals"]["block_count"] == 9
    assert report["totals"]["online_condition_count"] == 45
    assert (
        report["totals"]["successful_selected_result_count"]
        + report["totals"]["failed_online_condition_count"]
        == 45
    )
    assert report["totals"]["result_fact_count"] == report["totals"][
        "successful_selected_result_count"
    ]
    assert report["totals"]["oracle_disposition_count"] == 9
    assert all(report["invariants"].values())

    inventory_text = (output_root / "joint_inventory.json").read_text(
        encoding="utf-8"
    )
    manifest_text = (output_root / "manifest.json").read_text(encoding="utf-8")
    for forbidden in (
        "selected_score",
        "best_score",
        "991337.125",
        "882244.5",
        "773355.75",
    ):
        assert forbidden not in inventory_text
        assert forbidden not in manifest_text
    for filename in ("joint_inventory.json", "manifest.json"):
        assert not (output_root / filename).stat().st_mode & stat.S_IWUSR

    verification = verify_joint_inventory(
        inventory_root=output_root,
        **{
            key: value
            for key, value in fixture["kwargs"].items()
            if key != "created_at"
        },
    )
    assert verification["verified"] is True
    assert verification["errors"] == []
    assert verification["source_binding_hash"] == report["source_bindings"][
        "source_binding_hash"
    ]


def test_joint_inventory_accepts_only_the_semantically_checked_legacy_failure_alias(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    audit_path = next(
        path
        for path in fixture["audit_paths"]
        if "birds-seed-104729-r3" in path.name
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    invariants = audit["structural_invariants"]
    assert invariants.pop(
        "failed_conditions_have_no_terminal_result_or_result_fact"
    ) is True
    invariants["failed_conditions_have_no_result_fact"] = True
    audit.pop("lifecycle")
    audit["recovery_structure"] = {
        "recovery_status": "deterministic_preterminal_finalizer_recovered",
        "agent_reexecuted": False,
        "candidate_code_reexecuted": False,
        "full_condition_reexecuted": False,
        "terminal_metric_observed_during_recovery": False,
        "recovery_pod_not_found": True,
        "training_pod_not_found": True,
        "evaluator_pod_not_found": True,
        "evaluator_created_after_both_required_not_found_attestations": True,
    }
    _write_hashed(audit_path, audit, "audit_hash")

    report = build_joint_inventory(
        output_root=tmp_path / "legacy-alias-output", **fixture["kwargs"]
    )
    assert report["status"] == "passed"

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["recovery_structure"]["recovery_pod_not_found"] = False
    _write_hashed(audit_path, audit, "audit_hash")
    with pytest.raises(ValueError, match="recovery lifecycle failed"):
        build_joint_inventory(
            output_root=tmp_path / "legacy-alias-denied", **fixture["kwargs"]
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["recovery_structure"]["recovery_pod_not_found"] = True
    audit["structural_invariants"][
        "failed_conditions_have_no_result_fact"
    ] = False
    _write_hashed(audit_path, audit, "audit_hash")
    with pytest.raises(ValueError, match="failed_conditions_have_no_terminal"):
        build_joint_inventory(
            output_root=tmp_path / "legacy-alias-false", **fixture["kwargs"]
        )


def test_joint_inventory_allows_only_freeze_declared_legacy_manifest_absence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    block_id = "wp8-tier2-formal-aerial-seed-104729-r3"
    manifest_path = (
        fixture["completed_root"]
        / "blocks"
        / block_id
        / "BLOCK_EVALUATION_FILE_MANIFEST.json"
    )
    original = manifest_path.read_bytes()
    manifest_path.unlink()
    freeze_path = fixture["kwargs"]["completed_freeze_path"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze["blocks"][block_id][
        "block_evaluation_file_manifest_sha256"
    ] = None
    _write_hashed(freeze_path, freeze, "inventory_hash")

    report = build_joint_inventory(
        output_root=tmp_path / "legacy-manifest-output", **fixture["kwargs"]
    )
    row = next(item for item in report["blocks"] if item["block_id"] == block_id)
    assert row["block_evaluation_file_manifest_hashed"] is False
    assert row["block_evaluation_file_manifest_legacy_absence_frozen"] is True
    assert report["totals"][
        "legacy_block_evaluation_file_manifest_absence_count"
    ] == 1

    manifest_path.write_bytes(original)
    with pytest.raises(ValueError, match="declared artifact absence"):
        build_joint_inventory(
            output_root=tmp_path / "legacy-manifest-unexpected-file",
            **fixture["kwargs"],
        )


def test_joint_inventory_rejects_nonoutcome_roots_and_r4_blocks(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    failed_root = tmp_path / "decision-admissibility-wp8-tier2-formal-runs-r12"
    failed_root.mkdir()
    with pytest.raises(ValueError, match="root identity mismatch|failed r8/r12"):
        build_joint_inventory(
            output_root=tmp_path / "bad-root-output",
            **{**fixture["kwargs"], "continuation_root": failed_root},
        )

    precreated = fixture["completed_root"] / "blocks" / sorted(
        COMPLETED_PRECREATED_EMPTY_BLOCK_IDS
    )[0]
    unexpected_file = precreated / "unexpected-artifact"
    unexpected_file.write_text("must remain empty", encoding="utf-8")
    with pytest.raises(ValueError, match="Precreated non-outcome block root is not empty"):
        build_joint_inventory(
            output_root=tmp_path / "nonempty-precreated-output",
            **fixture["kwargs"],
        )
    unexpected_file.unlink()

    expected = fixture["continuation_root"] / "blocks" / (
        "wp8-tier2-formal-taxi-seed-155921-r5"
    )
    unexpected = fixture["continuation_root"] / "blocks" / (
        "wp8-tier2-formal-taxi-seed-155921-r4"
    )
    expected.rename(unexpected)
    with pytest.raises(ValueError, match="block set mismatch|non-outcome r4"):
        build_joint_inventory(
            output_root=tmp_path / "bad-block-output", **fixture["kwargs"]
        )


def test_joint_inventory_rejects_score_bearing_structure_audit(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["audit_paths"][0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["condition_structure"]["no_memory"]["selected_score"] = 0.5
    payload["audit_hash"] = _payload_hash(payload, "audit_hash")
    _write_json(path, payload)

    with pytest.raises(ValueError, match="score field is forbidden|Numeric score"):
        build_joint_inventory(
            output_root=tmp_path / "score-bearing-output", **fixture["kwargs"]
        )


def test_joint_inventory_verifier_detects_postbuild_tree_mutation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "joint-inventory"
    build_joint_inventory(output_root=output_root, **fixture["kwargs"])
    block_id = "wp8-tier2-formal-aerial-seed-104729-r3"
    mutated = (
        fixture["completed_root"]
        / "blocks"
        / block_id
        / "HOST_ORACLE.json"
    )
    mutated.write_bytes(mutated.read_bytes() + b"\nmutated")

    verification = verify_joint_inventory(
        inventory_root=output_root,
        **{
            key: value
            for key, value in fixture["kwargs"].items()
            if key != "created_at"
        },
    )
    assert verification["verified"] is False
    assert "inventory_recompute_mismatch" in verification["errors"]


def test_joint_inventory_rejects_rehashed_structural_drift_and_r8_reference(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    audit_path = next(
        path
        for path in fixture["audit_paths"]
        if "birds-seed-130363-r5" in path.name
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["successful_selected_result_count"] = 1
    audit["failed_online_condition_count"] = 4
    audit["result_fact_count"] = 1
    audit["audit_hash"] = _payload_hash(audit, "audit_hash")
    _write_json(audit_path, audit)
    with pytest.raises(ValueError, match="detail/aggregate mismatch"):
        build_joint_inventory(
            output_root=tmp_path / "rehashed-audit-output", **fixture["kwargs"]
        )

    fixture = _fixture(tmp_path / "second")
    spec_path = fixture["kwargs"]["analysis_spec_path"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["formal_outcome_root"] = (
        "/workspace/decision-admissibility-wp8-tier2-formal-runs-r8"
    )
    spec["spec_hash"] = _payload_hash(spec, "spec_hash")
    _write_json(spec_path, spec)
    with pytest.raises(ValueError, match="Failed formal root is forbidden"):
        build_joint_inventory(
            output_root=tmp_path / "r8-reference-output", **fixture["kwargs"]
        )


def test_joint_inventory_hashes_readlink_without_following_and_rejects_reuse(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    block_id = "wp8-tier2-formal-aerial-seed-104729-r3"
    opaque = (
        fixture["completed_root"]
        / "blocks"
        / block_id
        / "conditions"
        / "no_memory"
        / "opaque.bin"
    )
    opaque.unlink()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    opaque.symlink_to(outside)
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    outside_child = outside_dir / "must-not-be-followed.bin"
    outside_child.write_bytes(b"outside directory content")
    directory_link = opaque.parent / "linked-directory"
    directory_link.symlink_to(outside_dir, target_is_directory=True)
    output_root = tmp_path / "symlink-output"
    build_joint_inventory(output_root=output_root, **fixture["kwargs"])

    # Content behind the link is never read and therefore cannot perturb the
    # byte-tree binding.  Changing the readlink target itself must be detected.
    outside.write_bytes(b"changed but never followed")
    outside_child.write_bytes(b"directory target changed but never followed")
    verification = verify_joint_inventory(
        inventory_root=output_root,
        **{
            key: value
            for key, value in fixture["kwargs"].items()
            if key != "created_at"
        },
    )
    assert verification["verified"] is True
    opaque.unlink()
    opaque.symlink_to(tmp_path / "different-target.bin")
    verification = verify_joint_inventory(
        inventory_root=output_root,
        **{
            key: value
            for key, value in fixture["kwargs"].items()
            if key != "created_at"
        },
    )
    assert verification["verified"] is False
    assert "inventory_recompute_mismatch" in verification["errors"]

    existing = tmp_path / "already-exists"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        build_joint_inventory(output_root=existing, **fixture["kwargs"])


def test_joint_inventory_verifier_rejects_self_rehashed_report_mutation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output_root = tmp_path / "joint-inventory"
    build_joint_inventory(output_root=output_root, **fixture["kwargs"])
    report_path = output_root / "joint_inventory.json"
    manifest_path = output_root / "manifest.json"
    report_path.chmod(0o644)
    manifest_path.chmod(0o644)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["totals"]["block_count"] = 8
    report["report_hash"] = _payload_hash(report, "report_hash")
    _write_json(report_path, report)
    manifest = compute_manifest(report, inventory_file_sha256=_file_hash(report_path))
    _write_json(manifest_path, manifest)

    verification = verify_joint_inventory(
        inventory_root=output_root,
        **{
            key: value
            for key, value in fixture["kwargs"].items()
            if key != "created_at"
        },
    )
    assert verification["verified"] is False
    assert "inventory_recompute_mismatch" in verification["errors"]
    assert "block_count" in verification["errors"]
