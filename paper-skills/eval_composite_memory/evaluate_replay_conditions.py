#!/usr/bin/env python3
"""Evaluate R0-R3 repair candidates without trusting Agent self-reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from core import EPISODES, REPORTS, read_jsonl, sha256_file, write_json, write_jsonl


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "mlevolve") not in sys.path:
    sys.path.insert(0, str(REPO / "mlevolve"))

from agents.leakage_audit import audit_code, audit_repair_preservation, build_repair_preservation_contract  # noqa: E402


STAGES = ["data_scope", "validation_provenance", "cross_fit", "selection_freeze", "final_holdout"]


def _codes(audit: dict[str, Any]) -> list[str]:
    return sorted({str(row.get("issue_code")) for row in audit.get("issues", []) if row.get("issue_code")})


def evaluate(candidate_paths: list[Path] | None) -> dict[str, Any]:
    cases = read_jsonl(EPISODES / "replay_defects_v1.jsonl")
    candidates = [
        row
        for candidate_path in (candidate_paths or [])
        if candidate_path.exists()
        for row in read_jsonl(candidate_path)
    ]
    candidate_by_key = {(str(row.get("case_id")), str(row.get("condition"))): row for row in candidates}
    receipts: list[dict[str, Any]] = []
    for case in cases:
        source_audit = audit_code(case["code"])
        for condition in ("R0", "R1", "R2", "R3"):
            candidate = candidate_by_key.get((case["case_id"], condition))
            receipt: dict[str, Any] = {
                "schema": "runforest_composite_replay_condition_receipt_v1",
                "case_id": case["case_id"],
                "condition": condition,
                "defect": case["defect"],
                "task_family": case["task_family"],
                "source_blocked": source_audit.get("execution_disposition") == "block",
                "source_executed": False,
                "source_rank_eligible": False,
                "source_metric": None,
                "candidate_supplied": candidate is not None,
                "candidate_generated": bool(candidate and candidate.get("status") == "completed" and candidate.get("code")),
                "candidate_executed": False,
                "clean_repair": False,
                "static_clean": False,
                "preservation_clean": False,
                "stage_protocol_clean": False,
                "runtime_provenance_declared_clean": False,
                "runtime_provenance_verified_clean": False,
                "rank_eligible": False,
                "positive_memory_write": False,
            }
            if candidate and candidate.get("code"):
                static = audit_code(str(candidate["code"]))
                preservation = audit_repair_preservation(
                    str(candidate["code"]), build_repair_preservation_contract(case["code"])
                )
                history = candidate.get("stage_history") if isinstance(candidate.get("stage_history"), list) else []
                stage_names = [str(row.get("stage")) for row in history]
                intermediates = history[:-1] if stage_names == STAGES else []
                staged_clean = stage_names == STAGES and all(
                    row.get("status") == "clean"
                    and row.get("executed") is False
                    and row.get("rank_eligible") is False
                    and row.get("positive_memory_write") is False
                    for row in intermediates
                )
                runtime = candidate.get("runtime_provenance") if isinstance(candidate.get("runtime_provenance"), dict) else {}
                runtime_declared_clean = (
                    runtime.get("status") == "clean" and runtime.get("holdout_evaluation_count") == 1
                )
                static_clean = static.get("status") == "clean" and static.get("execution_disposition") == "allow"
                preservation_clean = preservation.get("status") == "clean" and preservation.get("execution_disposition") == "allow"
                condition_protocol_clean = condition == "R1" or (condition in {"R2", "R3"} and staged_clean)
                condition_preservation_clean = condition != "R3" or preservation_clean
                ready = static_clean and condition_protocol_clean and condition_preservation_clean
                receipt.update(
                    {
                        "candidate_static_issue_codes": _codes(static),
                        "candidate_preservation_issue_codes": _codes(preservation),
                        "static_clean": static_clean,
                        "preservation_clean": preservation_clean,
                        "stage_protocol_clean": staged_clean,
                        "runtime_provenance_declared_clean": runtime_declared_clean,
                        "runtime_provenance_verified_clean": False,
                        "ready_for_isolated_runtime": ready,
                        "clean_repair": False,
                    }
                )
            receipts.append(receipt)

    path = REPORTS / "replay_condition_receipts_v1.jsonl"
    write_jsonl(path, receipts)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipts:
        grouped[row["condition"]].append(row)
    by_condition = {
        condition: {
            "case_count": len(rows),
            "candidate_count": sum(row["candidate_supplied"] for row in rows),
            "generation_success_rate": sum(row["candidate_generated"] for row in rows) / len(rows),
            "static_clean_rate": sum(row["static_clean"] for row in rows) / len(rows),
            "preservation_clean_rate": sum(row["preservation_clean"] for row in rows) / len(rows),
            "stage_protocol_clean_rate": sum(row["stage_protocol_clean"] for row in rows) / len(rows),
            "runtime_provenance_declared_clean_rate": sum(
                row["runtime_provenance_declared_clean"] for row in rows
            ) / len(rows),
            "runtime_provenance_verified_clean_rate": sum(
                row["runtime_provenance_verified_clean"] for row in rows
            ) / len(rows),
            "clean_repair_rate": sum(row["clean_repair"] for row in rows) / len(rows),
            "invalid_rank_count": sum(row["rank_eligible"] and not row["clean_repair"] for row in rows),
            "positive_write_violation_count": sum(row["positive_memory_write"] and not row["clean_repair"] for row in rows),
        }
        for condition, rows in grouped.items()
    }
    report = {
        "schema": "runforest_composite_replay_condition_report_v1",
        "by_condition": by_condition,
        "candidate_count": len(candidates),
        "all_sources_blocked": all(row["source_blocked"] for row in receipts),
        "source_execution_count": 0,
        "source_rank_count": 0,
        "repair_claim_allowed": (
            len(candidates) == len(cases) * 3
            and by_condition["R3"]["clean_repair_rate"] > by_condition["R1"]["clean_repair_rate"]
            and by_condition["R3"]["preservation_clean_rate"] >= by_condition["R1"]["preservation_clean_rate"]
            and by_condition["R3"]["invalid_rank_count"] == 0
            and by_condition["R3"]["positive_write_violation_count"] == 0
        ),
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
    }
    write_json(REPORTS / "replay_condition_report_v1.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.candidates), ensure_ascii=False, indent=2))
