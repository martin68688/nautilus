#!/usr/bin/env python3
"""Evaluate deterministic replay safety gates on the frozen defect matrix.

This runner never executes a candidate.  Optional repaired candidates can be
supplied in JSONL form for preservation and static-audit evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from core import EPISODES, REPORTS, read_jsonl, sha256_file, write_json, write_jsonl


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "mlevolve") not in sys.path:
    sys.path.insert(0, str(REPO / "mlevolve"))

from agents.leakage_audit import (  # noqa: E402
    audit_code,
    audit_repair_preservation,
    build_repair_preservation_contract,
    structural_sha256,
)


def _issue_codes(audit: dict[str, Any]) -> list[str]:
    return sorted({str(row.get("issue_code")) for row in audit.get("issues", []) if row.get("issue_code")})


def run(candidate_path: Path | None = None, semantic_review_path: Path | None = None) -> dict[str, Any]:
    cases = read_jsonl(EPISODES / "replay_defects_v1.jsonl")
    candidates = read_jsonl(candidate_path) if candidate_path else []
    candidate_by_case = {str(row["case_id"]): row for row in candidates}
    reviews = read_jsonl(semantic_review_path) if semantic_review_path else []
    review_by_case = {str(row["case_id"]): row for row in reviews}
    receipts: list[dict[str, Any]] = []
    for case in cases:
        source_audit = audit_code(case["code"])
        codes = _issue_codes(source_audit)
        expected = str(case["issue_code"])
        candidate = candidate_by_case.get(case["case_id"])
        review = review_by_case.get(case["case_id"])
        semantic_codes = sorted({str(value) for value in (review or {}).get("issue_codes", [])})
        semantic_blocked = str((review or {}).get("classification") or "") in {
            "hard_leakage", "transductive_contamination", "selection_bias", "protocol_violation"
        }
        receipt: dict[str, Any] = {
            "schema": "runforest_composite_replay_receipt_v1",
            "case_id": case["case_id"],
            "task_family": case["task_family"],
            "defect": case["defect"],
            "stage": case["stage"],
            "source_code_sha256": case["code_sha256"],
            "source_structural_sha256": structural_sha256(case["code"]),
            "expected_issue_code": expected,
            "detected_issue_codes": codes,
            "semantic_review_supplied": review is not None,
            "semantic_issue_codes": semantic_codes,
            "expected_issue_detected": expected in codes or expected in semantic_codes,
            "source_blocked_before_execution": source_audit.get("execution_disposition") == "block" or semantic_blocked,
            "source_rank_eligible": False,
            "source_metric": None,
            "candidate_supplied": candidate is not None,
            "candidate_execution_attempted": False,
            "candidate_rank_eligible": False,
            "positive_memory_write": False,
        }
        if candidate is not None:
            repaired_code = str(candidate.get("code") or "")
            contract = build_repair_preservation_contract(case["code"])
            static = audit_code(repaired_code)
            preservation = audit_repair_preservation(repaired_code, contract)
            static_clean = static.get("status") == "clean" and static.get("execution_disposition") == "allow"
            preservation_clean = preservation.get("status") == "clean" and preservation.get("execution_disposition") == "allow"
            history = candidate.get("stage_history") if isinstance(candidate.get("stage_history"), list) else []
            expected_stages = ["data_scope", "validation_provenance", "cross_fit", "selection_freeze", "final_holdout"]
            stage_order_clean = [str(row.get("stage")) for row in history] == expected_stages
            intermediate_clean = all(
                row.get("status") == "clean"
                and row.get("executed") is False
                and row.get("rank_eligible") is False
                and row.get("positive_memory_write") is False
                for row in history[:-1]
            ) if stage_order_clean else False
            runtime = candidate.get("runtime_provenance") if isinstance(candidate.get("runtime_provenance"), dict) else {}
            runtime_clean = runtime.get("status") == "clean" and runtime.get("holdout_evaluation_count") == 1
            five_stage_clean = stage_order_clean and intermediate_clean and runtime_clean
            receipt.update(
                {
                    "candidate_code_sha256": static.get("code_sha256"),
                    "candidate_static_status": static.get("status"),
                    "candidate_static_issue_codes": _issue_codes(static),
                    "candidate_preservation_status": preservation.get("status"),
                    "candidate_preservation_issue_codes": _issue_codes(preservation),
                    "candidate_static_clean": static_clean,
                    "candidate_preservation_clean": preservation_clean,
                    "candidate_ready_for_runtime_validation": static_clean and preservation_clean,
                    "stage_order_clean": stage_order_clean,
                    "intermediate_journal_only_clean": intermediate_clean,
                    "runtime_provenance_clean": runtime_clean,
                    "five_stage_protocol_clean": five_stage_clean,
                }
            )
            final_clean = static_clean and preservation_clean and five_stage_clean
            receipt["candidate_rank_eligible"] = final_clean
            receipt["positive_memory_write"] = final_clean and candidate.get("positive_memory_write") is True
        receipts.append(receipt)

    path = REPORTS / "replay_static_receipts_v1.jsonl"
    write_jsonl(path, receipts)
    detected = sum(int(row["expected_issue_detected"]) for row in receipts)
    blocked = sum(int(row["source_blocked_before_execution"]) for row in receipts)
    candidate_rows = [row for row in receipts if row["candidate_supplied"]]
    five_stage = [row for row in candidate_rows if row.get("five_stage_protocol_clean") is True and row.get("candidate_rank_eligible") is True]
    report = {
        "schema": "runforest_composite_replay_report_v1",
        "case_count": len(receipts),
        "structural_template_count": len({row["source_structural_sha256"] for row in receipts}),
        "task_context_count": len({row["task_family"] for row in receipts}),
        "expected_issue_recall": detected / len(receipts) if receipts else None,
        "blocked_before_execution_rate": blocked / len(receipts) if receipts else None,
        "source_execution_count": 0,
        "source_ranked_count": 0,
        "candidate_count": len(candidate_rows),
        "candidate_runtime_validated_count": sum(row.get("runtime_provenance_clean") is True for row in candidate_rows),
        "semantic_review_count": len(reviews),
        "full_five_stage_repair_count": len(five_stage),
        "replay_safety_claim_allowed": bool(receipts) and blocked == len(receipts),
        "replay_repair_success_claim_allowed": len(five_stage) >= 48,
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
        "limitations": [
            "Static detector coverage is diagnostic and does not replace runtime provenance.",
            "No candidate is executed by this runner; final_holdout claims require a separate isolated evaluator.",
        ],
    }
    write_json(REPORTS / "replay_static_report_v1.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-repairs", type=Path)
    parser.add_argument("--semantic-reviews", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.candidate_repairs, args.semantic_reviews), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
