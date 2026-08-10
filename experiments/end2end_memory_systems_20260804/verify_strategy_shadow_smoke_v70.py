#!/usr/bin/env python3
"""Verify the immutable v70 Strategy shadow replay and atomic smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DISCOVERY = "spooky-before-frozen-multibackbone-fivefold-xgb"
REPAIR = "spooky-after-multibackbone-before-cleanup-repair"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def _strategy_checks(
    result: Mapping[str, Any],
    *,
    expected_current: int,
    expected_failure: int,
    expected_history: int,
) -> dict[str, bool]:
    strategy = dict(result.get("strategy_trace") or {})
    evaluation = dict(result.get("evaluation") or {})
    attempts = list(strategy.get("contract_attempts") or [])
    selection = dict(strategy.get("strategy_evidence_selection") or {})
    current_ids = list(selection.get("current_branch_frontier_ids") or [])
    failure_ids = list(selection.get("causal_failure_ids") or [])
    history_ids = list(selection.get("historical_diverse_frontier_ids") or [])
    selected_ids = list(selection.get("selected_memory_ids") or [])
    signatures = list(selection.get("historical_lineage_signatures") or [])
    rejections = dict(selection.get("historical_rejections") or {})
    memory_count = int(strategy.get("memory_card_count") or 0)
    normalizations = [
        dict(attempt.get("json_normalization") or {}) for attempt in attempts
    ]
    return {
        "trace_v2": strategy.get("schema") == "mlevolve_memory_strategy_shadow_trace_v2",
        "completed": strategy.get("status") == "completed",
        "model_is_v4_pro": strategy.get("model") == "deepseek-v4-pro",
        "thinking_enabled": strategy.get("thinking_enabled") is True,
        "contract_valid": bool(attempts and attempts[-1].get("valid")),
        "json_boundary_audited": bool(normalizations)
        and all(
            not row.get("used")
            or (
                row.get("authority") == "serialization_only"
                and row.get("thinking_enabled") is False
            )
            for row in normalizations
        ),
        "noninterference": strategy.get("production_prompt_modified") is False,
        "hidden_future_isolated": not result.get("hidden_future_leakage_ids"),
        "all_citations_visible": evaluation.get("all_citations_visible") is True,
        "at_least_three_compositions": int(
            evaluation.get("composition_count") or 0
        )
        >= 3,
        "evidence_count_is_eight": memory_count == 8,
        "selection_count_matches": int(selection.get("selected_count") or 0)
        == memory_count,
        "selection_ids_unique": len(selected_ids) == len(set(selected_ids)) == 8,
        "current_frontier_count": len(current_ids) == expected_current,
        "causal_failure_count": len(failure_ids) == expected_failure,
        "historical_frontier_count": len(history_ids) == expected_history,
        "partition_is_complete": (
            len(current_ids) + len(failure_ids) + len(history_ids) == memory_count
        ),
        "historical_lineages_unique": (
            len(signatures) == len(set(signatures)) == expected_history
            and all(signatures)
        ),
        "current_history_duplicates_filtered": int(
            rejections.get("current_node_duplicate") or 0
        )
        >= expected_current,
        "protocol_comparison_policy_recorded": bool(
            selection.get("historical_metric_comparison_policy")
        ),
    }


def verify(*, replay_path: Path, cases_path: Path) -> dict[str, Any]:
    replay = _load(replay_path)
    cases_packet = _load(cases_path)
    cases = {case["case_id"]: case for case in cases_packet["cases"]}
    results = {result["case_id"]: result for result in replay["results"]}
    discovery = results[DISCOVERY]
    repair = results[REPAIR]

    discovery_strategy = dict(discovery.get("strategy_trace") or {})
    discovery_eval = dict(discovery.get("evaluation") or {})
    discovery_validation = dict(discovery_strategy.get("validation") or {})
    required_opportunities = set(
        discovery_validation.get("required_opportunity_ids") or []
    )
    addressed_opportunities = set(
        discovery_validation.get("addressed_opportunity_ids") or []
    )

    repair_eval = dict(repair.get("evaluation") or {})
    atomic = dict(repair.get("atomic_actuation") or {})
    planner = dict(atomic.get("planner") or {})
    coder = dict(atomic.get("coder") or {})
    verdict = dict(coder.get("plan_diff_verdict") or {})
    planner_validation = dict(planner.get("validation") or {})
    selected = str(planner_validation.get("selected_hypothesis_id") or "")
    exact_repair_ids = set(repair_eval.get("future_strategy_exact_hit_ids") or [])
    planner_attempts = list(planner.get("contract_attempts") or [])
    parent_code = str(cases[REPAIR]["parent"]["code"] or "")
    candidate_code = str(coder.get("candidate_code") or "")
    cleanup_prefix = "del models, tokenizers"

    checks: dict[str, bool] = {}
    checks.update(
        {
            f"discovery_{key}": value
            for key, value in _strategy_checks(
                discovery,
                expected_current=3,
                expected_failure=0,
                expected_history=5,
            ).items()
        }
    )
    checks.update(
        {
            f"repair_{key}": value
            for key, value in _strategy_checks(
                repair,
                expected_current=3,
                expected_failure=1,
                expected_history=4,
            ).items()
        }
    )
    checks.update(
        {
            "all_diversity_opportunities_addressed": (
                bool(required_opportunities)
                and required_opportunities == addressed_opportunities
            ),
            "discovery_structural_future_hit": (
                discovery_eval.get("future_strategy_structural_hit") is True
            ),
            "repair_structural_future_hit": (
                repair_eval.get("future_strategy_structural_hit") is True
            ),
            "repair_exact_future_hit": (
                repair_eval.get("future_strategy_exact_hit") is True
            ),
            "atomic_pipeline_accepted": atomic.get("status") == "accepted",
            "atomic_planner_accepted": planner.get("status") == "accepted",
            "planner_contract_valid": bool(
                planner_attempts and planner_attempts[-1].get("valid")
            ),
            "planner_selected_exact_repair": bool(
                selected and selected in exact_repair_ids
            ),
            "atomic_coder_accepted": coder.get("status") == "accepted",
            "plan_diff_valid": verdict.get("valid") is True,
            "cleanup_nameerror_removed": bool(
                candidate_code
                and candidate_code.count(cleanup_prefix)
                < parent_code.count(cleanup_prefix)
            ),
        }
    )
    receipt = {
        "schema": "mlevolve_memory_strategy_shadow_smoke_receipt_v6",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "required_opportunity_ids": sorted(required_opportunities),
        "addressed_opportunity_ids": sorted(addressed_opportunities),
        "discovery_structural_hit_ids": discovery_eval.get(
            "future_strategy_structural_hit_ids", []
        ),
        "discovery_exact_hit_ids": discovery_eval.get(
            "future_strategy_exact_hit_ids", []
        ),
        "repair_structural_hit_ids": repair_eval.get(
            "future_strategy_structural_hit_ids", []
        ),
        "repair_exact_hit_ids": repair_eval.get(
            "future_strategy_exact_hit_ids", []
        ),
        "selected_hypothesis_id": selected,
        "changed_symbols": verdict.get("changed_symbols", []),
        "replay_sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(replay_path=args.replay, cases_path=args.cases)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
