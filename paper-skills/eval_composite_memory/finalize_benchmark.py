#!/usr/bin/env python3
"""Create the terminal, fail-closed benchmark ledger after applying stop rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core import MANIFESTS, REPORTS, read_json, sha256_file, write_json


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
    }


def finalize() -> dict[str, Any]:
    build = read_json(REPORTS / "build_report_v1.json")
    coverage = read_json(REPORTS / "coverage_audit_v1.json")
    offline = read_json(REPORTS / "offline_test_report_v1.json")
    claims = read_json(REPORTS / "claim_gate_report_v1.json")
    heldout = read_json(REPORTS / "replay_heldout_report_v1.json")
    replay = read_json(REPORTS / "replay_condition_report_v1.json")
    adoption = read_json(REPORTS / "adoption_pilot_report_v1.json")
    micro = read_json(REPORTS / "micro_execution_report_v1.json")

    hard_safety_failure = not (
        heldout.get("expected_issue_recall") == 1.0
        and heldout.get("blocked_before_execution_rate") == 1.0
    )
    t4_started = int(micro.get("completed_count", 0)) > 0
    if not hard_safety_failure:
        raise ValueError("terminal stop ledger is only valid after the preregistered safety stop fires")
    if t4_started:
        raise ValueError("claim-bearing T4 must not run after the safety stop")

    report = {
        "schema": "runforest_composite_benchmark_terminal_v1",
        "execution_status": "completed_stopped_fail_closed",
        "scientific_claim_status": "all_primary_claims_closed",
        "stop_rule_triggered": "independent_replay_safety_challenge_failed",
        "phase_status": {
            "phase_A_protocol_and_manifests": "completed",
            "T0_integrity": "completed",
            "T1_offline_retrieval": "completed_diagnostic_only",
            "phase_0_gate": "failed",
            "T2_agent_adoption": "bounded_pilot_only_then_stopped",
            "T3_replay": "bounded_generation_pilot_and_independent_safety_challenge_completed",
            "T4_micro_execution": "not_started_by_preregistered_stop_rule",
        },
        "key_results": {
            "test_episode_count": build["test_episode_count"],
            "offline_receipt_count": sum(
                int(row.get("episode_count", 0)) for row in offline["conditions"].values()
            ),
            "coverage_gap_count": coverage["coverage_gap_count"],
            "coverage_complete_rate": coverage["coverage_complete_rate"],
            "heldout_replay_expected_issue_recall": heldout.get("expected_issue_recall"),
            "heldout_replay_block_rate": heldout.get("blocked_before_execution_rate"),
            "adoption_first_pass_success_count": adoption.get("first_pass_completed_count"),
            "adoption_first_pass_total_count": adoption.get("first_pass_attempt_count"),
            "replay_clean_repair_count": sum(
                int(row.get("case_count", 0)) * float(row.get("clean_repair_rate", 0.0))
                for row in replay.get("by_condition", {}).values()
            ),
            "T4_completed_count": micro.get("completed_count", 0),
        },
        "claim_gates": {
            "phase0_pass": claims.get("phase0_pass") is True,
            "mechanism_claim_allowed": claims.get("mechanism_claim_allowed") is True,
            "adoption_claim_allowed": claims.get("adoption_claim_allowed") is True,
            "replay_repair_success_claim_allowed": replay.get("repair_claim_allowed") is True,
            "downstream_claim_allowed": claims.get("downstream_claim_allowed") is True,
            "blockers": claims.get("claim_blockers", []),
        },
        "interpretation": (
            "The benchmark run is terminal because the preregistered independent safety gate failed. "
            "Stopping before claim-bearing T4 is a completed negative experimental outcome, not evidence "
            "that T4 or universal safety succeeded."
        ),
        "next_version_requirements": [
            "Add genuinely new clean L1/L2 evidence for uncovered task-family/stage cells and freeze snapshot v2.",
            "Use two independent blind annotators plus adjudication before relevance claims.",
            "Improve the detector without tuning on the frozen held-out challenge, then preregister a new challenge v2.",
            "Run full T2/T3 only after the new safety gate passes; run T4 only after all mechanism gates pass.",
        ],
        "artifacts": [
            _artifact(REPORTS / "build_report_v1.json"),
            _artifact(REPORTS / "coverage_audit_v1.json"),
            _artifact(REPORTS / "offline_test_report_v1.json"),
            _artifact(REPORTS / "claim_gate_report_v1.json"),
            _artifact(REPORTS / "replay_heldout_report_v1.json"),
            _artifact(REPORTS / "adoption_pilot_report_v1.json"),
            _artifact(REPORTS / "replay_condition_report_v1.json"),
            _artifact(REPORTS / "micro_execution_report_v1.json"),
            _artifact(MANIFESTS / "replay_heldout_lock_v1.json"),
        ],
    }
    write_json(REPORTS / "benchmark_terminal_report_v1.json", report)
    _write_markdown(report)
    return report


def _write_markdown(report: dict[str, Any]) -> None:
    key = report["key_results"]
    lines = [
        "# RunForest Composite Benchmark: Terminal Report",
        "",
        "## Verdict",
        "",
        "**COMPLETED, STOPPED FAIL-CLOSED. All primary scientific claims remain closed.**",
        "",
        "The preregistered independent replay-safety gate failed, so claim-bearing T4 was not started.",
        "This is the required stopping behavior, not a missing positive result.",
        "",
        "## Evidence",
        "",
        f"- Test decision episodes: `{key['test_episode_count']}`.",
        f"- Offline receipts: `{key['offline_receipt_count']}`.",
        f"- Coverage gaps: `{key['coverage_gap_count']}`; complete rate `{key['coverage_complete_rate']:.3f}`.",
        f"- Independent replay issue recall: `{key['heldout_replay_expected_issue_recall']:.4f}`.",
        f"- Independent replay pre-execution block rate: `{key['heldout_replay_block_rate']:.4f}`.",
        f"- T4 completed runs: `{key['T4_completed_count']}`.",
        "",
        "## Phase Status",
        "",
    ]
    lines.extend(f"- `{name}`: `{status}`" for name, status in report["phase_status"].items())
    lines += ["", "## Required Before v2", ""]
    lines.extend(f"- {item}" for item in report["next_version_requirements"])
    lines += ["", "No superiority, universal safety, or downstream claim is licensed by this run.", ""]
    (REPORTS / "benchmark_terminal_report_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(json.dumps(finalize(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
