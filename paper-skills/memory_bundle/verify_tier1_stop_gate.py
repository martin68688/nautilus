from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from build_tier1_stop_gate import (
    STOP_GATE_SCHEMA,
    _render_markdown,
    compute_stop_gate,
)
from schema import sha256_json


VERIFICATION_SCHEMA = "decision_admissibility_wp8_tier1_stop_gate_verification_v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) == sha256_json(
        {key: value for key, value in payload.items() if key != field}
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2)
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_stop_gate(
    *,
    stop_gate_root: str | Path,
    plan_path: str | Path,
    work_root: str | Path,
    prevalence_root: str | Path,
    packet_root: str | Path,
    generation_root: str | Path,
    evaluation_root: str | Path,
    statistics_root: str | Path,
    regression_receipt_path: str | Path,
) -> dict[str, Any]:
    stop_gate_root = Path(stop_gate_root).resolve()
    report_path = stop_gate_root / "stop_gate_report.json"
    markdown_path = stop_gate_root / "stop_gate_report.md"
    report = _read_json(report_path)
    errors: list[str] = []
    if report.get("schema") != STOP_GATE_SCHEMA:
        errors.append("report_schema")
    if not _valid_hash(report, "report_hash"):
        errors.append("report_hash")
    builder_path = Path(__file__).resolve().with_name("build_tier1_stop_gate.py")
    if report.get("builder_source_sha256") != _sha256_file(builder_path):
        errors.append("builder_source_hash")
    try:
        recomputed = compute_stop_gate(
            plan_path=plan_path,
            work_root=work_root,
            prevalence_root=prevalence_root,
            packet_root=packet_root,
            generation_root=generation_root,
            evaluation_root=evaluation_root,
            statistics_root=statistics_root,
            regression_receipt_path=regression_receipt_path,
            created_at=str(report.get("created_at") or ""),
        )
    except Exception as error:  # fail closed at the evidence boundary
        errors.append(f"stop_gate_recompute_exception:{type(error).__name__}")
        recomputed = None
    if recomputed is not None and recomputed != report:
        errors.append("stop_gate_report_recompute")
    try:
        observed_markdown = markdown_path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as error:
        errors.append(f"markdown_read:{type(error).__name__}")
        observed_markdown = ""
    if observed_markdown != _render_markdown(report):
        errors.append("markdown_recompute")
    if report.get("passed") is not True or report.get("status") != "pass":
        errors.append("stop_gate_not_passed")
    if report.get("next_authorized_phase") != "WP8 Multi-generation":
        errors.append("next_phase")
    if report.get("large_scale_tier2_authorized") is not False:
        errors.append("tier2_authority_boundary")
    if report.get("wp8_complete") is not False:
        errors.append("wp8_completion_boundary")
    if not all((report.get("stop_gate_checks") or {}).values()):
        errors.append("stop_gate_check_failure")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "stop_gate_root_name": stop_gate_root.name,
        "status": report.get("status"),
        "required_check_count": len(report.get("stop_gate_checks") or {}),
        "passed_check_count": sum(
            value is True for value in (report.get("stop_gate_checks") or {}).values()
        ),
        "next_authorized_phase": report.get("next_authorized_phase"),
        "large_scale_tier2_authorized": report.get(
            "large_scale_tier2_authorized"
        ),
        "verified": not errors,
        "errors": sorted(set(errors)),
        "stop_gate_report_hash": report.get("report_hash", ""),
        "stop_gate_report_file_sha256": _sha256_file(report_path),
        "stop_gate_markdown_file_sha256": _sha256_file(markdown_path),
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    verification["verification_hash"] = sha256_json(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_hash"
        }
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the WP8 Tier-1 Stop Gate.")
    parser.add_argument("--stop-gate-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--prevalence-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--regression-receipt", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verification = verify_stop_gate(
        stop_gate_root=args.stop_gate_root,
        plan_path=args.plan,
        work_root=args.work_root,
        prevalence_root=args.prevalence_root,
        packet_root=args.packet_root,
        generation_root=args.generation_root,
        evaluation_root=args.evaluation_root,
        statistics_root=args.statistics_root,
        regression_receipt_path=args.regression_receipt,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False, indent=2))
    if not verification["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
