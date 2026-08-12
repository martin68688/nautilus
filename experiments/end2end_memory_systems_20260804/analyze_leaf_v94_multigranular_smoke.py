#!/usr/bin/env python3
"""Summarize a Leaf v94 Journal without mutating run artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


GRANULARITIES = (
    "l1_recipe",
    "l2_tactic",
    "l3_repair",
    "runforest_run",
    "runforest_transition",
)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def metric_value(node: Mapping[str, Any]) -> float | None:
    metric = node.get("metric")
    value = metric.get("value") if isinstance(metric, Mapping) else metric
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def host_grep_totals(trace: list[dict[str, Any]]) -> dict[str, int]:
    totals = Counter()
    for round_record in trace:
        for query in round_record.get("queries") or []:
            grep = query.get("host_grep") or {}
            totals["query_count"] += 1
            totals["matched"] += int(grep.get("matched_candidate_count") or 0)
            totals["returned"] += int(grep.get("returned_candidate_count") or 0)
            totals["discarded_by_query_limit"] += int(
                grep.get("discarded_by_query_limit") or 0
            )
    return dict(totals)


def summarize_multigranular(node: Mapping[str, Any]) -> dict[str, Any] | None:
    routing = node.get("memory_routing_trace") or {}
    retrieval = routing.get("retrieval_agent") or {}
    search = retrieval.get("multigranular_search") or {}
    judge = retrieval.get("retrieval_judge") or {}
    if not search and not judge:
        return None
    trace = list(search.get("trace") or [])
    searched_rounds = [row for row in trace if row.get("status") == "searched"]
    first_queries = {
        str(query.get("granularity") or "")
        for query in (searched_rounds[0].get("queries") or [])
    } if searched_rounds else set()
    allocations = [
        {
            "round": int(row.get("round") or 0),
            "allocation": dict(row.get("allocation") or {}),
            "accumulated_counts": dict(row.get("accumulated_counts") or {}),
            "new_candidate_count": int(row.get("new_candidate_count") or 0),
        }
        for row in searched_rounds
    ]
    return {
        "node_id": str(node.get("id") or ""),
        "stage": str(node.get("stage") or ""),
        "mode": str(retrieval.get("mode") or ""),
        "search_agent_calls": int(
            retrieval.get("multigranular_search_agent_calls") or 0
        ),
        "independent_judge_calls": int(
            retrieval.get("independent_retrieval_judge_calls") or 0
        ),
        "main_retrieval_agent_calls": int(
            retrieval.get("main_retrieval_agent_calls") or 0
        ),
        "authorized_counts": dict(search.get("authorized_counts") or {}),
        "accumulated_counts": dict(search.get("accumulated_counts") or {}),
        "initial_all_granularities_queried": first_queries == set(GRANULARITIES),
        "initial_queried_granularities": sorted(first_queries),
        "rounds": allocations,
        "host_grep_totals": host_grep_totals(trace),
        "judge_status": str(judge.get("status") or ""),
        "judge_decision": str(judge.get("decision") or ""),
        "judge_candidate_count": int(judge.get("candidate_count") or 0),
        "judge_selected_ids": list(judge.get("selected_ids") or []),
        "final_selection_authority": str(
            retrieval.get("final_selection_authority") or ""
        ),
        "effective_selected_ids": list(
            retrieval.get("effective_selected_ids") or []
        ),
        "final_prompt_candidate_ids": list(
            routing.get("final_prompt_candidate_ids") or []
        ),
        "fallback_used": bool(retrieval.get("fallback_used")),
        "prompt_truncated": bool(routing.get("prompt_truncated")),
    }


def summarize_debug(node: Mapping[str, Any]) -> dict[str, Any] | None:
    routing = node.get("memory_routing_trace") or {}
    retrieval = routing.get("retrieval_agent") or {}
    grep_calls = int(retrieval.get("grep_search_agent_calls") or 0)
    root_calls = int(retrieval.get("root_cause_agent_calls") or 0)
    if not grep_calls and not root_calls:
        return None
    return {
        "node_id": str(node.get("id") or ""),
        "grep_search_agent_calls": grep_calls,
        "l3_root_cause_judge_calls": root_calls,
        "main_retrieval_agent_calls": int(
            retrieval.get("main_retrieval_agent_calls") or 0
        ),
        "agent_selected_ids": list(retrieval.get("agent_selected_ids") or []),
        "effective_selected_ids": list(
            retrieval.get("effective_selected_ids") or []
        ),
        "final_prompt_candidate_ids": list(
            routing.get("final_prompt_candidate_ids") or []
        ),
        "final_selection_authority": str(
            retrieval.get("final_selection_authority") or ""
        ),
        "fallback_used": bool(retrieval.get("fallback_used")),
    }


def summarize_actuation(node: Mapping[str, Any]) -> dict[str, Any] | None:
    strategy = node.get("memory_strategy_trace") or {}
    atomic = node.get("atomic_actuation_trace") or {}
    if not strategy and not atomic:
        return None
    evidence = strategy.get("strategy_evidence_selection") or {}
    planner = atomic.get("planner") or {}
    coder = atomic.get("coder") or {}
    validation = strategy.get("validation") or {}
    return {
        "node_id": str(node.get("id") or ""),
        "stage": str(node.get("stage") or ""),
        "strategy_status": str(strategy.get("status") or ""),
        "strategy_valid": validation.get("valid"),
        "strategy_evidence_count": int(
            evidence.get("selected_count")
            or len(strategy.get("memory_card_ids") or [])
        ),
        "strategy_evidence_limit": int(evidence.get("max_items") or 0),
        "atomic_status": str(atomic.get("status") or ""),
        "planner_status": str(planner.get("status") or ""),
        "coder_status": str(coder.get("status") or ""),
        "plan_diff_valid": (coder.get("plan_diff_verdict") or {}).get("valid"),
        "verification_mode": str(
            (coder.get("plan_diff_verdict") or {}).get("verification_mode") or ""
        ),
    }


def build_summary(journal: Mapping[str, Any], *, expected_steps: int) -> dict[str, Any]:
    nodes = list(journal.get("nodes") or [])
    work_nodes = [node for node in nodes if str(node.get("stage") or "") != "root"]
    eligible_metrics = [
        (metric_value(node), str(node.get("id") or ""), str(node.get("stage") or ""))
        for node in work_nodes
        if node.get("is_buggy") is False
        and node.get("is_valid") is True
        and metric_value(node) is not None
    ]
    eligible_metrics = [row for row in eligible_metrics if row[0] is not None]
    best = min(eligible_metrics, default=None)
    multigranular = [
        summary
        for node in work_nodes
        if (summary := summarize_multigranular(node)) is not None
    ]
    debug = [
        summary
        for node in work_nodes
        if (summary := summarize_debug(node)) is not None
    ]
    actuation = [
        summary
        for node in work_nodes
        if (summary := summarize_actuation(node)) is not None
    ]
    return {
        "schema": "leaf_v94_multigranular_smoke_summary_v1",
        "expected_steps": int(expected_steps),
        "completed_work_nodes": len(work_nodes),
        "completion_fraction": (
            len(work_nodes) / expected_steps if expected_steps > 0 else None
        ),
        "stage_counts": dict(Counter(str(node.get("stage") or "") for node in work_nodes)),
        "valid_count": sum(node.get("is_valid") is True for node in work_nodes),
        "buggy_count": sum(node.get("is_buggy") is True for node in work_nodes),
        "best_internal_metric": (
            {"value": best[0], "node_id": best[1], "stage": best[2]}
            if best is not None
            else None
        ),
        "multigranular_nodes": multigranular,
        "multigranular_checks": {
            "node_count": len(multigranular),
            "all_initial_rounds_cover_five_granularities": bool(multigranular)
            and all(row["initial_all_granularities_queried"] for row in multigranular),
            "all_use_independent_judge": bool(multigranular)
            and all(
                row["independent_judge_calls"] > 0
                and row["main_retrieval_agent_calls"] == 0
                for row in multigranular
            ),
            "fallback_count": sum(row["fallback_used"] for row in multigranular),
            "prompt_truncation_count": sum(
                row["prompt_truncated"] for row in multigranular
            ),
        },
        "debug_l3_nodes": debug,
        "debug_l3_checks": {
            "node_count": len(debug),
            "grep_call_count": sum(row["grep_search_agent_calls"] for row in debug),
            "root_cause_judge_call_count": sum(
                row["l3_root_cause_judge_calls"] for row in debug
            ),
            "fallback_count": sum(row["fallback_used"] for row in debug),
        },
        "strategy_atomic_nodes": actuation,
        "strategy_atomic_checks": {
            "node_count": len(actuation),
            "strategy_completed_count": sum(
                row["strategy_status"] == "completed" for row in actuation
            ),
            "atomic_accepted_count": sum(
                row["atomic_status"] == "accepted" for row in actuation
            ),
            "mechanical_only_valid_count": sum(
                row["verification_mode"] == "mechanical_only"
                and row["plan_diff_valid"] is True
                for row in actuation
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=16)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = build_summary(
        read_object(args.journal), expected_steps=int(args.expected_steps)
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
