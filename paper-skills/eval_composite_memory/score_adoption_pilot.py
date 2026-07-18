#!/usr/bin/env python3
"""Consolidate bounded T2 pilot attempts without hiding first-pass failures."""

from __future__ import annotations

import json
from collections import defaultdict

from core import REPORTS, read_jsonl, write_json, write_jsonl
from run_agent_adoption import run as score_adoption


ATTEMPT_FILES = [
    REPORTS / "adoption_candidates_v1.jsonl",
    REPORTS / "adoption_candidates_pilot_f00_retry_v1.jsonl",
    REPORTS / "adoption_candidates_pilot_f01_retry_v1.jsonl",
    REPORTS / "adoption_candidates_pilot_f10_retry_v1.jsonl",
]


def score() -> dict:
    attempts = [row for path in ATTEMPT_FILES if path.exists() for row in read_jsonl(path)]
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in attempts:
        by_key[(row["episode_id"], row["condition"])].append(row)
    selected = []
    never_successful = []
    for rows in by_key.values():
        successful = [row for row in rows if row.get("status") == "completed" and row.get("code")]
        if successful:
            selected.append(successful[0])
        else:
            never_successful.append({
                "episode_id": rows[0]["episode_id"],
                "condition": rows[0]["condition"],
                "attempt_count": len(rows),
                "failure_reasons": [str(row.get("error") or "unknown_failure") for row in rows],
            })
    consolidated = REPORTS / "adoption_candidates_pilot_complete_v1.jsonl"
    write_jsonl(consolidated, selected)
    adoption = score_adoption(consolidated, persist=False)
    initial = read_jsonl(REPORTS / "adoption_candidates_v1.jsonl")
    report = {
        "schema": "runforest_composite_adoption_pilot_report_v1",
        "episode_count": len({row["episode_id"] for row in selected}),
        "condition_count": len({row["condition"] for row in selected}),
        "first_pass_attempt_count": len(initial),
        "first_pass_completed_count": sum(row.get("status") == "completed" for row in initial),
        "all_attempt_count": len(attempts),
        "completed_condition_count_after_retry": len(selected),
        "never_successful_count": len(never_successful),
        "never_successful_keys": never_successful,
        "attempts_by_condition": {
            condition: len([row for row in attempts if row["condition"] == condition])
            for condition in ("F00", "F01", "F10", "F11")
        },
        "family_alignment_rate": adoption["family_alignment_rate"],
        "family_alignment_n": len(selected),
        "provenance_complete_rate": adoption["provenance_complete_rate"],
        "claim_allowed": False,
        "test_split_consumed_by_pilot": True,
        "claim_blockers": [
            "pilot_has_only_one_episode",
            "human_adjudication_missing",
            "test_episode_consumed_before_frozen_full_run",
        ],
        "candidate_path": str(consolidated),
    }
    write_json(REPORTS / "adoption_pilot_report_v1.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(score(), ensure_ascii=False, indent=2))
