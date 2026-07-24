"""Independently recompute and verify the WP8 final engineering Stop Gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from build_wp8_final_regression_receipt import payload_hash, sha256_file
from build_wp8_final_stop_gate import (
    MANIFEST_SCHEMA,
    STOP_GATE_SCHEMA,
    _render_markdown,
    compute_stop_gate,
)


VERIFICATION_SCHEMA = "decision_admissibility_wp8_final_stop_gate_verification_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_stop_gate(
    *,
    stop_gate_root: str | Path,
    repo_root: str | Path,
    final_regression_receipt_path: str | Path,
    final_test_root: str | Path,
    host_test_receipt_root: str | Path | None = None,
) -> dict[str, Any]:
    raw_stop_gate_root = Path(stop_gate_root)
    errors: list[str] = []
    if raw_stop_gate_root.is_symlink():
        errors.append("stop_gate_root_symlink")
    stop_gate_root = raw_stop_gate_root.resolve()
    repo_root = Path(repo_root).resolve()
    report_path = stop_gate_root / "stop_gate_report.json"
    markdown_path = stop_gate_root / "stop_gate_report.md"
    manifest_path = stop_gate_root / "manifest.json"
    if not stop_gate_root.is_dir():
        errors.append("stop_gate_root_missing")
    elif stop_gate_root.stat().st_mode & 0o222:
        errors.append("stop_gate_root_writable")
    expected_entries = {
        "manifest.json",
        "stop_gate_report.json",
        "stop_gate_report.md",
    }
    observed_entries = (
        {path.name for path in stop_gate_root.iterdir()}
        if stop_gate_root.is_dir()
        else set()
    )
    if observed_entries != expected_entries:
        errors.append("stop_gate_entry_set")
    for path in stop_gate_root.iterdir() if stop_gate_root.is_dir() else ():
        if path.is_symlink() or not path.is_file():
            errors.append(f"stop_gate_non_regular:{path.name}")
        elif path.stat().st_mode & 0o222:
            errors.append(f"stop_gate_file_writable:{path.name}")
    try:
        report = _read_object(report_path)
        manifest = _read_object(manifest_path)
        markdown = markdown_path.read_text(encoding="utf-8")
    except Exception as error:
        report = {}
        manifest = {}
        markdown = ""
        errors.append(f"gate_read:{type(error).__name__}")
    if report.get("schema") != STOP_GATE_SCHEMA:
        errors.append("report_schema")
    if report.get("report_hash") != payload_hash(report, "report_hash"):
        errors.append("report_hash")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "complete":
        errors.append("manifest_schema_or_status")
    if manifest.get("manifest_hash") != payload_hash(manifest, "manifest_hash"):
        errors.append("manifest_hash")
    for filename, path in (
        ("stop_gate_report.json", report_path),
        ("stop_gate_report.md", markdown_path),
    ):
        if not path.is_file() or (manifest.get("files") or {}).get(filename) != sha256_file(path):
            errors.append(f"file_hash:{filename}")
    if manifest.get("report_hash") != report.get("report_hash"):
        errors.append("manifest_report_binding")
    builder_path = Path(__file__).resolve().with_name("build_wp8_final_stop_gate.py")
    if report.get("builder_source_sha256") != sha256_file(builder_path):
        errors.append("builder_source_hash")
    if report.get("verifier_source_sha256") != sha256_file(Path(__file__).resolve()):
        errors.append("verifier_source_hash")
    if manifest.get("builder_source_sha256") != sha256_file(builder_path):
        errors.append("manifest_builder_source_hash")
    if manifest.get("verifier_source_sha256") != sha256_file(Path(__file__).resolve()):
        errors.append("manifest_verifier_source_hash")
    if markdown != _render_markdown(report):
        errors.append("markdown_recompute")
    if report.get("wp8_engineering_complete") is not True:
        errors.append("engineering_completion")
    if report.get("wp8_stop_gate_passed") is not True:
        errors.append("stop_gate_passed")
    if report.get("effect_claim_authorized") is not False:
        errors.append("effect_claim_boundary")
    if report.get("next_authorized_phase") != "Independent Claude audit":
        errors.append("next_phase")
    if report.get("independent_claude_audit_required") is not True:
        errors.append("independent_audit_boundary")
    if report.get("goal_completion_authorized") is not False:
        errors.append("goal_completion_boundary")
    if not all((report.get("prerequisite_checks") or {}).values()):
        errors.append("prerequisite_check_failure")
    if not all((report.get("kill_gates") or {}).values()):
        errors.append("kill_gate_failure")
    if not all(
        value
        for group in (report.get("acceptance_checks") or {}).values()
        for value in group.values()
    ):
        errors.append("acceptance_check_failure")
    if not all((report.get("formal_integrity_checks") or {}).values()):
        errors.append("formal_integrity_check_failure")
    effect_gate = report.get("formal_effect_claim_gate") or {}
    if effect_gate.get("effect_claim_authorized") is not False:
        errors.append("formal_effect_gate")
    claim_status = report.get("claim_status") or {}
    if claim_status.get("WP8-C3-FULL-SUPERIORITY") != "rejected":
        errors.append("superiority_claim_status")
    if claim_status.get("WP8-C4-CONDITIONAL-UTILITY") != "diagnostic":
        errors.append("conditional_claim_status")
    if claim_status.get("WP8-C6-EXPERIENCE-CAUSALITY") != "pending":
        errors.append("causal_claim_status")
    boundary = report.get("post_result_change_boundary") or {}
    if not (
        boundary.get("formal_outcomes_bind_frozen_execution_sources") is True
        and boundary.get("post_result_safety_fixes_not_used_to_rewrite_outcomes") is True
        and boundary.get("post_result_safety_fixes_not_evaluated_for_formal_effect") is True
        and boundary.get("rerun_or_seed_selection_performed") is False
    ):
        errors.append("post_result_change_boundary")
    try:
        expected = compute_stop_gate(
            repo_root=repo_root,
            final_regression_receipt_path=final_regression_receipt_path,
            final_test_root=final_test_root,
            host_test_receipt_root=host_test_receipt_root,
            created_at=str(report.get("created_at") or ""),
        )
    except Exception as error:
        expected = None
        errors.append(f"gate_recompute:{type(error).__name__}")
    if expected is not None and expected != report:
        errors.append("gate_recompute_mismatch")
    try:
        root_name = str(stop_gate_root.relative_to(repo_root))
    except ValueError:
        root_name = str(stop_gate_root)
    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed" if not errors else "failed",
        "verified": not errors,
        "errors": sorted(set(errors)),
        "stop_gate_root": root_name,
        "wp8_engineering_complete": report.get("wp8_engineering_complete"),
        "effect_claim_authorized": report.get("effect_claim_authorized"),
        "goal_completion_authorized": report.get("goal_completion_authorized"),
        "required_prerequisite_count": len(report.get("prerequisite_checks") or {}),
        "kill_gate_count": len(report.get("kill_gates") or {}),
        "acceptance_check_count": sum(
            len(group) for group in (report.get("acceptance_checks") or {}).values()
        ),
        "report_hash": report.get("report_hash", ""),
        "report_file_sha256": sha256_file(report_path) if report_path.is_file() else "",
        "manifest_hash": manifest.get("manifest_hash", ""),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    verification["verification_hash"] = payload_hash(
        verification, "verification_hash"
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-gate-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--final-regression-receipt", required=True, type=Path)
    parser.add_argument("--final-test-root", required=True, type=Path)
    parser.add_argument("--host-test-receipt-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_stop_gate(
        stop_gate_root=args.stop_gate_root,
        repo_root=args.repo_root,
        final_regression_receipt_path=args.final_regression_receipt,
        final_test_root=args.final_test_root,
        host_test_receipt_root=args.host_test_receipt_root,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
