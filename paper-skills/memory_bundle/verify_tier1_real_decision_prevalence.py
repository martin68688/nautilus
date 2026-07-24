from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from audit_tier1_real_decision_prevalence import (
    GATE_THRESHOLDS,
    RECEIPT_SCHEMA,
    REPORT_SCHEMA,
    _summarize_rows,
    load_real_decisions,
)
from schema import sha256_json


VERIFICATION_SCHEMA = "decision_admissibility_real_prevalence_verification_v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) == sha256_json(
        {key: value for key, value in payload.items() if key != field}
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_prevalence(
    work_root: str | Path,
    prevalence_root: str | Path,
) -> dict[str, Any]:
    work_root = Path(work_root).resolve()
    prevalence_root = Path(prevalence_root).resolve()
    errors: list[str] = []
    report_path = prevalence_root / "prevalence_report.json"
    report = _read_json(report_path)
    receipts_path = prevalence_root / str(report["retrieval_receipts_file"])
    receipts = _read_jsonl(receipts_path)
    corpus_manifest = _read_json(work_root / "corpus_manifest.json")
    trace_manifest = _read_json(work_root / "traces" / "trace_manifest.json")
    audit_report_path = work_root / "audit_report.json"
    clauses_path = work_root / "binder" / "clauses.jsonl"

    if report.get("schema") != REPORT_SCHEMA:
        errors.append("report_schema")
    if not _valid_hash(report, "report_hash"):
        errors.append("report_hash")
    if report.get("corpus_manifest_hash") != corpus_manifest.get("manifest_sha256"):
        errors.append("corpus_manifest_binding")
    if report.get("trace_manifest_hash") != trace_manifest.get("manifest_sha256"):
        errors.append("trace_manifest_binding")
    if report.get("clauses_file_sha256") != _sha256_file(clauses_path):
        errors.append("clauses_file_hash")
    if report.get("audit_report_sha256") != _sha256_file(audit_report_path):
        errors.append("audit_report_hash")
    if report.get("retrieval_receipts_file_sha256") != _sha256_file(receipts_path):
        errors.append("retrieval_receipts_file_hash")
    if len(receipts) != report.get("eligible_decision_count"):
        errors.append("eligible_decision_count")
    if len({row.get("node_ref") for row in receipts}) != len(receipts):
        errors.append("duplicate_node_ref")

    for row in receipts:
        node_ref = str(row.get("node_ref") or "")
        if row.get("schema") != RECEIPT_SCHEMA:
            errors.append(f"receipt_schema:{node_ref}")
        if not _valid_hash(row, "receipt_hash"):
            errors.append(f"receipt_hash:{node_ref}")
        candidates = row.get("top_k") or []
        if not candidates:
            errors.append(f"empty_top_k:{node_ref}")
            continue
        if any(
            candidate.get("same_domain") is not True
            or candidate.get("different_task") is not True
            or candidate.get("current_run_excluded") is not True
            for candidate in candidates
        ):
            errors.append(f"transfer_scope:{node_ref}")
        if row.get("target_history_exposure_count") != 0:
            errors.append(f"target_history_exposure:{node_ref}")
        if row.get("cross_domain_exposure_count") != 0:
            errors.append(f"cross_domain_exposure:{node_ref}")
        expected_any = any(candidate.get("mismatch") is True for candidate in candidates)
        expected_top1 = candidates[0].get("mismatch") is True
        expected_stage = any(
            candidate.get("stage_match") is not True for candidate in candidates
        )
        expected_authority = any(
            candidate.get("authority_valid") is not True for candidate in candidates
        )
        expected_valid = any(
            candidate.get("stage_match") is True
            and candidate.get("authority_valid") is True
            for candidate in candidates
        )
        if row.get("top_k_any_mismatch") is not expected_any:
            errors.append(f"any_mismatch_flag:{node_ref}")
        if row.get("top1_mismatch") is not expected_top1:
            errors.append(f"top1_mismatch_flag:{node_ref}")
        if row.get("top_k_stage_mismatch") is not expected_stage:
            errors.append(f"stage_mismatch_flag:{node_ref}")
        if row.get("top_k_authority_invalid") is not expected_authority:
            errors.append(f"authority_invalid_flag:{node_ref}")
        if row.get("top_k_full_valid_available") is not expected_valid:
            errors.append(f"valid_available_flag:{node_ref}")
        forbidden_text_fields = {"query_text", "clause_text", "retrieval_text", "text"}
        if forbidden_text_fields & set(row):
            errors.append(f"raw_query_text_embedded:{node_ref}")

    overall = _summarize_rows(receipts)
    if overall != report.get("overall"):
        errors.append("overall_recompute")
    by_stage = {
        key: _summarize_rows([row for row in receipts if row["stage"] == key])
        for key in sorted({row["stage"] for row in receipts})
    }
    by_domain = {
        key: _summarize_rows(
            [row for row in receipts if row["target_domain"] == key]
        )
        for key in sorted({row["target_domain"] for row in receipts})
    }
    if by_stage != report.get("by_stage"):
        errors.append("by_stage_recompute")
    if by_domain != report.get("by_domain"):
        errors.append("by_domain_recompute")
    if report.get("covered_stage_count") != len(by_stage):
        errors.append("covered_stage_count")
    if report.get("covered_domain_count") != len(by_domain):
        errors.append("covered_domain_count")

    all_nodes, trace_hashes = load_real_decisions(
        work_root / "traces" / "trace_manifest.json",
        work_root / "traces",
    )
    if report.get("real_run_node_count") != len(all_nodes):
        errors.append("real_run_node_count")
    if report.get("non_code_root_count") != sum(
        row["stage"] == "root" for row in all_nodes
    ):
        errors.append("non_code_root_count")
    if report.get("real_code_node_count") != sum(
        row["stage"] != "root" for row in all_nodes
    ):
        errors.append("real_code_node_count")
    if report.get("trace_file_hashes_hash") != sha256_json(trace_hashes):
        errors.append("trace_file_hashes_hash")

    gate = report.get("gate_1") or {}
    thresholds = gate.get("thresholds") or {}
    if thresholds != GATE_THRESHOLDS:
        errors.append("gate_thresholds")
    expected_checks = {
        "eligible_decisions": len(receipts)
        >= int(thresholds.get("minimum_eligible_decisions", 10**12)),
        "covered_domains": len(by_domain)
        >= int(thresholds.get("minimum_covered_domains", 10**12)),
        "covered_stages": len(by_stage)
        >= int(thresholds.get("minimum_covered_stages", 10**12)),
        "top5_any_mismatch_prevalence": bool(
            overall["top_k_any_mismatch_wilson_lower_95"] is not None
            and overall["top_k_any_mismatch_wilson_lower_95"]
            >= float(
                thresholds.get(
                    "minimum_top5_any_mismatch_wilson_lower_95", 1.1
                )
            )
        ),
        "top1_mismatch_prevalence": bool(
            overall["top1_mismatch_wilson_lower_95"] is not None
            and overall["top1_mismatch_wilson_lower_95"]
            >= float(thresholds.get("minimum_top1_mismatch_wilson_lower_95", 1.1))
        ),
    }
    if gate.get("checks") != expected_checks:
        errors.append("gate_checks")
    if gate.get("passed") is not all(expected_checks.values()):
        errors.append("gate_result")
    if gate.get("thresholds_fixed_before_audit") is not True:
        errors.append("threshold_preregistration_flag")
    if report.get("causal_or_downstream_performance_claimed") is not False:
        errors.append("causal_claim_boundary")
    auditor_path = Path(__file__).resolve().with_name(
        "audit_tier1_real_decision_prevalence.py"
    )
    if report.get("auditor_source_sha256") != _sha256_file(auditor_path):
        errors.append("auditor_source_hash")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "prevalence_root_name": prevalence_root.name,
        "real_run_node_count": len(all_nodes),
        "real_code_node_count": sum(row["stage"] != "root" for row in all_nodes),
        "eligible_decision_count": len(receipts),
        "candidate_exposure_count": sum(len(row["top_k"]) for row in receipts),
        "target_history_exposure_count": sum(
            row["target_history_exposure_count"] for row in receipts
        ),
        "cross_domain_exposure_count": sum(
            row["cross_domain_exposure_count"] for row in receipts
        ),
        "gate_1_passed": gate.get("passed") is True,
        "verified": not errors,
        "errors": sorted(set(errors)),
        "prevalence_report_hash": report.get("report_hash", ""),
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
        description="Verify WP8 real decision prevalence evidence."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--prevalence-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_prevalence(args.work_root, args.prevalence_root)
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
