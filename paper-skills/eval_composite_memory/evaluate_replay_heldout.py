#!/usr/bin/env python3
"""One-shot evaluation on independently authored, detector-blind defects."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from core import EPISODES, MANIFESTS, REPORTS, read_json, read_jsonl, sha256_file, write_json, write_jsonl


REPO = Path(__file__).resolve().parents[2]
if str(REPO / "mlevolve") not in sys.path:
    sys.path.insert(0, str(REPO / "mlevolve"))

from agents.leakage_audit import DETECTOR_VERSION, audit_code  # noqa: E402


SOURCE = EPISODES / "replay_defects_heldout_claude_v1.jsonl"
DETECTOR_SOURCE = REPO / "mlevolve" / "agents" / "leakage_audit.py"
LOCK = MANIFESTS / "replay_heldout_lock_v1.json"


def evaluate() -> dict[str, Any]:
    current_lock = {
        "schema": "runforest_composite_replay_heldout_lock_v1",
        "source_sha256": sha256_file(SOURCE),
        "detector_source_sha256": sha256_file(DETECTOR_SOURCE),
        "detector_version": DETECTOR_VERSION,
    }
    if LOCK.exists() and read_json(LOCK) != current_lock:
        raise RuntimeError(
            "held-out v1 is immutable; detector or challenge changed, so author a detector-blind v2 challenge"
        )
    if not LOCK.exists():
        write_json(LOCK, current_lock)
    cases = read_jsonl(SOURCE)
    receipts: list[dict[str, Any]] = []
    for case in cases:
        audit = audit_code(case["code"])
        codes = sorted({str(row.get("issue_code")) for row in audit.get("issues", []) if row.get("issue_code")})
        receipts.append(
            {
                "schema": "runforest_composite_replay_heldout_receipt_v1",
                "case_id": case["case_id"],
                "defect": case["defect"],
                "task_family": case["task_family"],
                "expected_issue_code": case["issue_code"],
                "detected_issue_codes": codes,
                "expected_issue_detected": case["issue_code"] in codes,
                "blocked_before_execution": audit.get("execution_disposition") == "block",
                "detector_version": DETECTOR_VERSION,
                "authoring_session": case["authoring_session"],
                "detector_tuning_forbidden": case["detector_tuning_forbidden"],
            }
        )
    path = REPORTS / "replay_heldout_receipts_v1.jsonl"
    write_jsonl(path, receipts)
    by_defect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in receipts:
        by_defect[row["defect"]].append(row)
    report = {
        "schema": "runforest_composite_replay_heldout_report_v1",
        "case_count": len(receipts),
        "detector_version": DETECTOR_VERSION,
        "detector_source_sha256": sha256_file(DETECTOR_SOURCE),
        "lock_path": str(LOCK),
        "lock_sha256": sha256_file(LOCK),
        "source_sha256": sha256_file(SOURCE),
        "authoring_session": cases[0]["authoring_session"] if cases else None,
        "detector_source_was_hidden_from_author": True,
        "post_evaluation_detector_tuning_forbidden": True,
        "expected_issue_recall": sum(row["expected_issue_detected"] for row in receipts) / len(receipts),
        "blocked_before_execution_rate": sum(row["blocked_before_execution"] for row in receipts) / len(receipts),
        "by_defect": {
            defect: {
                "case_count": len(rows),
                "expected_issue_recall": sum(row["expected_issue_detected"] for row in rows) / len(rows),
                "blocked_before_execution_rate": sum(row["blocked_before_execution"] for row in rows) / len(rows),
            }
            for defect, rows in sorted(by_defect.items())
        },
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
        "claim_scope": "independently_agent_authored_structural_challenge_not_population_recall",
        "gate_registration_note": (
            "The all-blocked safety invariant was preregistered; the dedicated held-out report and named "
            "claim-gate fields were added conservatively after challenge authoring and cannot support a positive claim."
        ),
    }
    write_json(REPORTS / "replay_heldout_report_v1.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
