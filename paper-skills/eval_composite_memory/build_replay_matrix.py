#!/usr/bin/env python3
"""Freeze the preregistered R0-R3 replay-repair comparison matrix."""

from __future__ import annotations

import json

from core import EPISODES, MANIFESTS, read_jsonl, sha256_file, write_json


CONDITIONS = {
    "R0": {
        "label": "reject_drop",
        "agent_generation": False,
        "staged_protocol": False,
        "preservation_contract_exposed": False,
        "runtime_provenance_required": False,
    },
    "R1": {
        "label": "ordinary_debug",
        "agent_generation": True,
        "staged_protocol": False,
        "preservation_contract_exposed": False,
        "runtime_provenance_required": False,
    },
    "R2": {
        "label": "staged_without_preservation",
        "agent_generation": True,
        "staged_protocol": True,
        "preservation_contract_exposed": False,
        "runtime_provenance_required": True,
    },
    "R3": {
        "label": "staged_with_preservation_and_runtime",
        "agent_generation": True,
        "staged_protocol": True,
        "preservation_contract_exposed": True,
        "runtime_provenance_required": True,
    },
}


def build() -> dict:
    cases = read_jsonl(EPISODES / "replay_defects_v1.jsonl")
    runs = [
        {
            "run_id": f"{case['case_id']}::{condition}",
            "case_id": case["case_id"],
            "condition": condition,
            "task_family": case["task_family"],
            "defect": case["defect"],
            "expected_issue_code": case["issue_code"],
        }
        for case in cases
        for condition in CONDITIONS
    ]
    path = MANIFESTS / "replay_matrix_v1.json"
    payload = {
        "schema": "runforest_composite_replay_matrix_v1",
        "conditions": CONDITIONS,
        "case_count": len(cases),
        "run_count": len(runs),
        "runs": runs,
        "source_path": str(EPISODES / "replay_defects_v1.jsonl"),
        "source_sha256": sha256_file(EPISODES / "replay_defects_v1.jsonl"),
    }
    write_json(path, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
