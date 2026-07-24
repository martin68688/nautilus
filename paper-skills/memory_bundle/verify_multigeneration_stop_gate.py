from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from build_multigeneration_stop_gate import (
    STOP_GATE_SCHEMA,
    _render_markdown,
    compute_stop_gate,
)
from schema import sha256_json


VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_multigeneration_stop_gate_verification_v1"
)


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
    prior_tier1_stop_gate_root: str | Path,
    packet_root: str | Path,
    run_root: str | Path,
    evaluation_root: str | Path,
    superseded_evaluation_root: str | Path,
    statistics_root: str | Path,
    regression_receipt_path: str | Path,
) -> dict[str, Any]:
    stop_gate_root = Path(stop_gate_root).resolve()
    report_path = stop_gate_root / "stop_gate_report.json"
    markdown_path = stop_gate_root / "stop_gate_report.md"
    report = _read_json(report_path)
    markdown = markdown_path.read_text(encoding="utf-8")
    errors: list[str] = []

    if report.get("schema") != STOP_GATE_SCHEMA:
        errors.append("report_schema")
    if not _valid_hash(report, "report_hash"):
        errors.append("report_hash")
    builder_path = Path(__file__).resolve().with_name(
        "build_multigeneration_stop_gate.py"
    )
    if report.get("builder_source_sha256") != _sha256_file(builder_path):
        errors.append("builder_source_hash")
    try:
        recomputed = compute_stop_gate(
            plan_path=plan_path,
            work_root=work_root,
            prior_tier1_stop_gate_root=prior_tier1_stop_gate_root,
            packet_root=packet_root,
            run_root=run_root,
            evaluation_root=evaluation_root,
            superseded_evaluation_root=superseded_evaluation_root,
            statistics_root=statistics_root,
            regression_receipt_path=regression_receipt_path,
            created_at=str(report.get("created_at") or ""),
        )
    except Exception as error:
        errors.append(f"stop_gate_recompute_exception:{type(error).__name__}")
        recomputed = None
    if recomputed is not None and recomputed != report:
        errors.append("stop_gate_report_recompute")
    if _render_markdown(report) != markdown:
        errors.append("stop_gate_markdown_recompute")
    if report.get("passed") is not True or report.get("status") != "pass":
        errors.append("stop_gate_not_passed")
    if not all((report.get("stop_gate_checks") or {}).values()):
        errors.append("stop_gate_check_failure")
    if report.get("next_authorized_phase") != "WP8 Tier-2 canary":
        errors.append("next_phase_boundary")
    if report.get("tier2_canary_authorized") is not True:
        errors.append("tier2_canary_boundary")
    if report.get("large_scale_tier2_authorized") is not False:
        errors.append("large_scale_tier2_boundary")
    if report.get("wp8_complete") is not False:
        errors.append("wp8_completion_boundary")
    if (
        "does_not_support_full_superiority_over_lineage_only"
        not in set(report.get("claim_boundaries") or [])
    ):
        errors.append("lineage_only_claim_boundary")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "stop_gate_root_name": stop_gate_root.name,
        "verified": not errors,
        "errors": sorted(set(errors)),
        "required_check_count": len(report.get("stop_gate_checks") or {}),
        "passed_check_count": sum(
            value is True for value in (report.get("stop_gate_checks") or {}).values()
        ),
        "gate_5_passed": (report.get("kill_gates") or {})
        .get("gate_5", {})
        .get("passed"),
        "next_authorized_phase": report.get("next_authorized_phase"),
        "large_scale_tier2_authorized": report.get(
            "large_scale_tier2_authorized"
        ),
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
    parser = argparse.ArgumentParser(
        description="Verify the WP8 Multi-generation Gate-5 Stop Gate."
    )
    parser.add_argument("--stop-gate-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--prior-tier1-stop-gate-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--superseded-evaluation-root", required=True, type=Path)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--regression-receipt", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verification = verify_stop_gate(
        stop_gate_root=args.stop_gate_root,
        plan_path=args.plan,
        work_root=args.work_root,
        prior_tier1_stop_gate_root=args.prior_tier1_stop_gate_root,
        packet_root=args.packet_root,
        run_root=args.run_root,
        evaluation_root=args.evaluation_root,
        superseded_evaluation_root=args.superseded_evaluation_root,
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
