"""Independently recompute and verify a result-blind Tier-2 joint inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_tier2_formal_joint_inventory import (
    MANIFEST_SCHEMA,
    REPORT_SCHEMA,
    _assert_no_numeric_score_values,
    _json_text,
    _payload_hash,
    _sha256_file,
    compute_joint_inventory,
    compute_manifest,
)


VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_joint_inventory_verification_v1"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    _assert_no_numeric_score_values(value, label=str(path))
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_json_text(payload))
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def verify_joint_inventory(
    *,
    inventory_root: str | Path,
    completed_root: str | Path,
    continuation_root: str | Path,
    analysis_spec_path: str | Path,
    completed_freeze_path: str | Path,
    completed_staging_gate_path: str | Path,
    continuation_staging_gate_path: str | Path,
    recovery_gate_path: str | Path,
    recovery_diagnostic_path: str | Path,
    structure_audit_paths: Sequence[str | Path],
) -> dict[str, Any]:
    inventory_root = Path(inventory_root).resolve()
    report_path = inventory_root / "joint_inventory.json"
    manifest_path = inventory_root / "manifest.json"
    errors: list[str] = []

    try:
        report = _read_object(report_path)
    except Exception as error:
        report = {}
        errors.append(f"inventory_read:{type(error).__name__}")
    try:
        manifest = _read_object(manifest_path)
    except Exception as error:
        manifest = {}
        errors.append(f"manifest_read:{type(error).__name__}")

    if report.get("schema") != REPORT_SCHEMA:
        errors.append("inventory_schema")
    if report.get("report_hash") != _payload_hash(report, "report_hash"):
        errors.append("inventory_internal_hash")
    if report.get("score_values_included") is not False or report.get(
        "score_values_inspected"
    ) is not False:
        errors.append("inventory_result_blindness")
    if report.get("score_policy") != "hash_only":
        errors.append("inventory_score_policy")
    if report.get("formal_block_json_parsed") is not False:
        errors.append("formal_block_json_parse_flag")
    if report.get("score_bearing_artifacts_parsed") is not False:
        errors.append("score_bearing_artifact_parse_flag")

    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("manifest_schema")
    if manifest.get("manifest_hash") != _payload_hash(manifest, "manifest_hash"):
        errors.append("manifest_internal_hash")
    if report_path.is_file() and manifest.get(
        "joint_inventory_file_sha256"
    ) != _sha256_file(report_path):
        errors.append("manifest_inventory_file_hash")
    if manifest.get("joint_inventory_hash") != report.get("report_hash"):
        errors.append("manifest_inventory_internal_binding")

    builder_path = Path(__file__).resolve().with_name(
        "build_tier2_formal_joint_inventory.py"
    )
    builder_source_sha256 = _sha256_file(builder_path)
    if report.get("builder_source_sha256") != builder_source_sha256:
        errors.append("builder_source_hash")
    if manifest.get("builder_source_sha256") != builder_source_sha256:
        errors.append("manifest_builder_source_hash")

    try:
        expected_report = compute_joint_inventory(
            completed_root=completed_root,
            continuation_root=continuation_root,
            analysis_spec_path=analysis_spec_path,
            completed_freeze_path=completed_freeze_path,
            completed_staging_gate_path=completed_staging_gate_path,
            continuation_staging_gate_path=continuation_staging_gate_path,
            recovery_gate_path=recovery_gate_path,
            recovery_diagnostic_path=recovery_diagnostic_path,
            structure_audit_paths=structure_audit_paths,
            created_at=str(report.get("created_at") or ""),
        )
    except Exception as error:
        expected_report = None
        errors.append(f"inventory_recompute:{type(error).__name__}")
    if expected_report is not None and expected_report != report:
        errors.append("inventory_recompute_mismatch")

    expected_manifest: dict[str, Any] | None = None
    if expected_report is not None:
        expected_report_sha256 = hashlib.sha256(
            _json_text(expected_report).encode("utf-8")
        ).hexdigest()
        expected_manifest = compute_manifest(
            expected_report,
            inventory_file_sha256=expected_report_sha256,
        )
        if expected_manifest != manifest:
            errors.append("manifest_recompute_mismatch")

    totals = report.get("totals") or {}
    invariants = report.get("invariants") or {}
    if totals.get("block_count") != 9:
        errors.append("block_count")
    if totals.get("online_condition_count") != 45:
        errors.append("online_condition_count")
    if int(totals.get("successful_selected_result_count", -1)) + int(
        totals.get("failed_online_condition_count", -1)
    ) != 45:
        errors.append("success_failure_partition")
    if totals.get("result_fact_count") != totals.get(
        "successful_selected_result_count"
    ):
        errors.append("result_fact_count")
    if totals.get("oracle_disposition_count") != 9:
        errors.append("oracle_disposition_count")
    if not invariants or not all(value is True for value in invariants.values()):
        errors.append("inventory_invariants")

    source_bindings = report.get("source_bindings") or {}
    source_binding_hash = str(source_bindings.get("source_binding_hash") or "")
    if manifest.get("source_binding_hash") != source_binding_hash:
        errors.append("manifest_source_binding_hash")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed" if not errors else "failed",
        "verified": not errors,
        "errors": sorted(set(errors)),
        "score_policy": "hash_only",
        "score_values_included": False,
        "score_values_inspected": False,
        "formal_block_json_parsed": False,
        "score_bearing_artifacts_parsed": False,
        "block_count": totals.get("block_count"),
        "online_condition_count": totals.get("online_condition_count"),
        "oracle_disposition_count": totals.get("oracle_disposition_count"),
        "joint_inventory_hash": report.get("report_hash", ""),
        "joint_inventory_file_sha256": (
            _sha256_file(report_path) if report_path.is_file() else ""
        ),
        "manifest_hash": manifest.get("manifest_hash", ""),
        "source_binding_hash": source_binding_hash,
        "builder_source_sha256": builder_source_sha256,
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    _assert_no_numeric_score_values(
        verification, label="joint_inventory_verification"
    )
    verification["verification_hash"] = _payload_hash(
        verification, "verification_hash"
    )
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-root", required=True, type=Path)
    parser.add_argument("--completed-root", required=True, type=Path)
    parser.add_argument("--continuation-root", required=True, type=Path)
    parser.add_argument("--analysis-spec", required=True, type=Path)
    parser.add_argument("--completed-freeze", required=True, type=Path)
    parser.add_argument("--completed-staging-gate", required=True, type=Path)
    parser.add_argument("--continuation-staging-gate", required=True, type=Path)
    parser.add_argument("--recovery-gate", required=True, type=Path)
    parser.add_argument("--recovery-diagnostic", required=True, type=Path)
    parser.add_argument(
        "--structure-audit", action="append", default=[], type=Path
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    verification = verify_joint_inventory(
        inventory_root=args.inventory_root,
        completed_root=args.completed_root,
        continuation_root=args.continuation_root,
        analysis_spec_path=args.analysis_spec,
        completed_freeze_path=args.completed_freeze,
        completed_staging_gate_path=args.completed_staging_gate,
        continuation_staging_gate_path=args.continuation_staging_gate,
        recovery_gate_path=args.recovery_gate,
        recovery_diagnostic_path=args.recovery_diagnostic,
        structure_audit_paths=args.structure_audit,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), verification)
    print(_json_text(verification), end="")
    if not verification["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["VERIFICATION_SCHEMA", "verify_joint_inventory"]
