from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from build_tier2_canary_stop_gate import (
    STOP_GATE_SCHEMA,
    compute_stop_gate,
    render_markdown,
    sha256_file,
    sha256_json,
    valid_hash,
)


VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_canary_stop_gate_verification_v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def verify_stop_gate(
    *,
    stop_gate_root: str | Path,
    plan_path: str | Path,
    prior_gate5_root: str | Path,
    evidence_packet_root: str | Path,
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
    if not valid_hash(report, "report_hash"):
        errors.append("report_hash")
    builder_path = Path(__file__).resolve().with_name(
        "build_tier2_canary_stop_gate.py"
    )
    if report.get("builder_source_sha256") != sha256_file(builder_path):
        errors.append("builder_source_hash")
    try:
        recomputed = compute_stop_gate(
            plan_path=plan_path,
            prior_gate5_root=prior_gate5_root,
            evidence_packet_root=evidence_packet_root,
            regression_receipt_path=regression_receipt_path,
            created_at=str(report.get("created_at") or ""),
        )
    except Exception as error:
        errors.append(f"recompute_exception:{type(error).__name__}")
        recomputed = None
    if recomputed is not None and recomputed != report:
        errors.append("report_recompute")
    if render_markdown(report) != markdown:
        errors.append("markdown_recompute")
    if report.get("passed") is not True or report.get("status") != "pass":
        errors.append("gate_not_passed")
    if not all((report.get("stop_gate_checks") or {}).values()):
        errors.append("required_check_failure")
    if report.get("next_authorized_phase") != (
        "WP8 Tier-2 formal experiment staging"
    ):
        errors.append("next_phase_boundary")
    if report.get("formal_tier2_authorized") is not True:
        errors.append("formal_tier2_boundary")
    if report.get("large_scale_effect_claim_authorized") is not False:
        errors.append("large_scale_effect_claim_boundary")
    if report.get("paper_effect_claim_authorized") is not False:
        errors.append("paper_effect_claim_boundary")
    if report.get("wp8_complete") is not False:
        errors.append("wp8_completion_boundary")
    required_boundaries = {
        "canary_not_formal_effect_evidence",
        "no_full_superiority_claim",
        "terminal_result_fact_is_not_adoption_or_causality",
        "source_scores_not_inherited",
    }
    if not required_boundaries <= set(report.get("claim_boundaries") or []):
        errors.append("claim_boundaries")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "verified": not errors,
        "errors": sorted(set(errors)),
        "required_check_count": len(report.get("stop_gate_checks") or {}),
        "passed_check_count": sum(
            value is True
            for value in (report.get("stop_gate_checks") or {}).values()
        ),
        "next_authorized_phase": report.get("next_authorized_phase"),
        "formal_tier2_authorized": report.get("formal_tier2_authorized"),
        "paper_effect_claim_authorized": report.get(
            "paper_effect_claim_authorized"
        ),
        "stop_gate_report_hash": report.get("report_hash", ""),
        "stop_gate_report_file_sha256": sha256_file(report_path),
        "stop_gate_markdown_file_sha256": sha256_file(markdown_path),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
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


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-gate-root", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prior-gate5-root", required=True, type=Path)
    parser.add_argument("--evidence-packet-root", required=True, type=Path)
    parser.add_argument("--regression-receipt", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verification = verify_stop_gate(
        stop_gate_root=args.stop_gate_root,
        plan_path=args.plan,
        prior_gate5_root=args.prior_gate5_root,
        evidence_packet_root=args.evidence_packet_root,
        regression_receipt_path=args.regression_receipt,
    )
    if args.output is not None:
        _write_exclusive(args.output.resolve(), verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False, indent=2))
    if not verification["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
