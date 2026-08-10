#!/usr/bin/env python3
"""Point-in-time historical replay for Memory Strategy shadow mode.

The hidden future outcome and evaluator patterns are never passed to the
Strategy Agent.  They are consulted only after the memo is sealed.  Optional
actuation runs the separate Atomic Planner/Coder path; it never executes or
publishes a search candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.atomic_actuation import run_atomic_actuation_pipeline
from agents.memory_strategy_agent import run_memory_strategy_shadow
from config import _load_cfg
from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _visible(event: Mapping[str, Any], cutoff: Mapping[str, Any]) -> bool:
    if cutoff.get("timestamp") is not None and event.get("available_at") is not None:
        return float(event["available_at"]) <= float(cutoff["timestamp"])
    if cutoff.get("order") is not None and event.get("available_order") is not None:
        return int(event["available_order"]) <= int(cutoff["order"])
    return bool(event.get("visible", True))


def _build_router_pack(case: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cutoff = dict(case.get("cutoff") or {})
    visible = [
        copy.deepcopy(dict(event))
        for event in (case.get("memory_events") or [])
        if isinstance(event, Mapping) and _visible(event, cutoff)
    ]
    cards = []
    for event in visible:
        event.pop("available_at", None)
        event.pop("available_order", None)
        event.pop("visible", None)
        if "candidate_id" not in event:
            event["candidate_id"] = str(event.get("memory_id") or event.get("id") or "")
        cards.append(event)
    # The historical harness exposes six as Router-visible and the remainder
    # as wider pre-gate evidence.  Strategy still receives both sets.
    final = cards[:6]
    pack = {
        "schema": "experiment_r_memory_pack_v1",
        "stage_route": {"stage": str(case.get("stage") or "improve")},
        "target_task_id": str(case.get("task_id") or ""),
        "final_prompt_candidate_ids": [
            str(item.get("candidate_id") or "") for item in final
        ],
        "final_prompt_candidates": final,
        "selected_candidates": final,
        "pre_gate_raw_candidates": cards[6:],
        "candidate_pool": {"historical_slice_candidates": cards},
        "retrieval_agent": {
            "enabled": True,
            "mode": "historical_time_slice_replay",
            "agent_abstained": not bool(final),
        },
        "router_activation": {
            "status": "selected" if final else "abstain",
            "candidate_pool_nonempty": bool(cards),
        },
    }
    return pack, [str(item.get("candidate_id") or "") for item in cards]


def _node_from_payload(payload: Mapping[str, Any], *, stage: str) -> SearchNode:
    node = SearchNode(
        code=str(payload.get("code") or ""),
        plan=str(payload.get("plan") or ""),
        stage=stage,
        id=str(payload.get("node_id") or payload.get("id") or "historical-parent"),
        branch_id=payload.get("branch_id"),
        draft_role=str(payload.get("draft_role") or "") or None,
    )
    metric = payload.get("metric")
    if metric is not None:
        node.metric = MetricValue(float(metric), maximize=bool(payload.get("maximize", False)))
    node.is_buggy = bool(payload.get("is_buggy", False))
    node.is_valid = (
        not node.is_buggy
        if payload.get("is_valid") is None
        else bool(payload.get("is_valid"))
    )
    node.code_summary = str(payload.get("code_summary") or "")
    node._term_out = [str(payload.get("execution_output") or "")]
    node.official_submission_receipt = copy.deepcopy(
        dict(payload.get("official_submission_receipt") or {})
    )
    node.leakage_audit = copy.deepcopy(dict(payload.get("leakage_audit") or {}))
    node.metric_protocol = str(
        payload.get("metric_protocol") or payload.get("validation_protocol") or ""
    )
    return node


def _build_agent(case: Mapping[str, Any], config_path: Path):
    cfg = _load_cfg(config_path, use_cli_args=False)
    cfg.exp_id = str(case.get("task_id") or "historical-strategy-replay")
    ext = cfg.external_skill_memory
    ext.memory_strategy_shadow_enabled = True
    ext.memory_strategy_shadow_stages = [str(case.get("stage") or "improve")]
    ext.memory_strategy_evidence_limit = int(case.get("evidence_limit") or 8)
    ext.memory_strategy_current_frontier_slots = int(
        case.get("current_frontier_slots") or 3
    )
    ext.memory_strategy_causal_failure_slots = int(
        case.get("causal_failure_slots") or 1
    )
    ext.memory_strategy_candidate_pool_limit = int(
        case.get("candidate_pool_limit") or 48
    )
    ext.memory_strategy_max_cards = int(case.get("max_cards") or 24)
    ext.memory_strategy_max_input_chars = int(case.get("max_input_chars") or 0)
    ext.memory_strategy_max_output_tokens = int(case.get("max_output_tokens") or 6000)
    ext.memory_strategy_debug_trigger = str(
        case.get("memory_strategy_debug_trigger")
        or getattr(ext, "memory_strategy_debug_trigger", "causal_gap_or_repeated_failure")
    )
    agent = SimpleNamespace(
        cfg=cfg,
        acfg=cfg.agent,
        task_desc=str(case.get("task_description") or ""),
        data_preview=str(case.get("data_preview") or ""),
        start_time=time.time()
        - max(0.0, float((case.get("budget") or {}).get("elapsed_search_seconds") or 0.0)),
        journal=Journal(),
        branch_all_nodes={},
        branch_successful_nodes={},
        metric_maximize=bool((case.get("parent") or {}).get("maximize", False)),
    )
    for raw_node in case.get("current_branch_nodes") or []:
        if not isinstance(raw_node, Mapping):
            continue
        node = _node_from_payload(
            raw_node,
            stage=str(raw_node.get("stage") or "improve"),
        )
        branch_id = node.branch_id
        if branch_id is None:
            continue
        agent.branch_all_nodes.setdefault(branch_id, []).append(node)
        if (
            node.metric is not None
            and node.is_buggy is not True
            and node.is_valid is not False
            and (node.leakage_audit or {}).get("rank_eligible") is not False
        ):
            agent.branch_successful_nodes.setdefault(branch_id, []).append(node)
    for attempt in case.get("attempt_history") or []:
        if not isinstance(attempt, Mapping):
            continue
        node = _node_from_payload(attempt, stage=str(attempt.get("stage") or "improve"))
        agent.journal.append(node)
    return agent, cfg


def _matches_all_groups(text: str, groups: list[list[str]]) -> bool:
    return all(
        any(re.search(pattern, text, re.IGNORECASE) for pattern in group)
        for group in groups
    )


def _positive_composition_text(composition: Mapping[str, Any]) -> str:
    """Serialize asserted experiment content, excluding negated/conflict text."""

    return json.dumps(
        {
            "hypothesis_id": composition.get("hypothesis_id"),
            "hypothesis": composition.get("hypothesis"),
            "minimal_change_set": composition.get("minimal_change_set"),
            "expected_mechanism": composition.get("expected_mechanism"),
            "novelty_kind": composition.get("novelty_kind"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def evaluate_memo(
    case: Mapping[str, Any],
    *,
    memo: Mapping[str, Any],
    visible_memory_ids: list[str],
) -> dict[str, Any]:
    compositions = [
        dict(item)
        for item in (memo.get("candidate_compositions") or [])
        if isinstance(item, Mapping)
    ]
    structural_groups = [
        [str(pattern) for pattern in group]
        for group in (
            case.get("expected_structural_pattern_groups")
            or case.get("expected_future_pattern_groups")
            or []
        )
    ]
    exact_groups = [
        [str(pattern) for pattern in group]
        for group in (case.get("expected_exact_future_pattern_groups") or [])
    ]
    structural_hits = []
    exact_hits = []
    over_budget = []
    duplicate = []
    invalid_combinations = []
    remaining = (case.get("budget") or {}).get("remaining_search_seconds")
    attempted = []
    for raw_signature in case.get("attempted_pattern_signatures") or []:
        if isinstance(raw_signature, Mapping):
            required_groups = raw_signature.get("required_groups") or []
            novelty_groups = raw_signature.get("novelty_exclusion_groups") or []
        else:
            required_groups = raw_signature
            novelty_groups = []
        attempted.append(
            {
                "required_groups": [
                    [str(pattern) for pattern in group]
                    for group in required_groups
                ],
                "novelty_exclusion_groups": [
                    [str(pattern) for pattern in group]
                    for group in novelty_groups
                ],
            }
        )
    incompatibilities = list(case.get("known_incompatibilities") or [])
    for composition in compositions:
        text = _positive_composition_text(composition)
        hypothesis_id = str(composition.get("hypothesis_id") or "")
        if structural_groups and _matches_all_groups(text, structural_groups):
            structural_hits.append(hypothesis_id)
        if exact_groups and _matches_all_groups(text, exact_groups):
            exact_hits.append(hypothesis_id)
        if remaining is not None and int(composition.get("estimated_compute_seconds") or 0) > int(remaining):
            over_budget.append(hypothesis_id)
        if any(
            _matches_all_groups(text, signature["required_groups"])
            and not any(
                _matches_all_groups(text, [group])
                for group in signature["novelty_exclusion_groups"]
            )
            for signature in attempted
        ):
            duplicate.append(hypothesis_id)
        for rule in incompatibilities:
            groups = [
                [str(pattern) for pattern in group]
                for group in (rule.get("pattern_groups") or [])
            ]
            if groups and _matches_all_groups(text, groups):
                resolution = str(rule.get("resolution_pattern") or "")
                checks = " ".join(
                    str(value) for value in (composition.get("compatibility_checks") or [])
                )
                if not resolution or not re.search(resolution, checks, re.IGNORECASE):
                    invalid_combinations.append(
                        {
                            "hypothesis_id": hypothesis_id,
                            "rule": str(rule.get("reason") or "known incompatibility"),
                        }
                    )
    cited = {
        str(value)
        for composition in compositions
        for value in (composition.get("source_memory_ids") or [])
    }
    unsupported_ids = sorted(cited - set(visible_memory_ids))
    total = len(compositions)
    return {
        "schema": "mlevolve_memory_strategy_replay_evaluation_v2",
        "case_id": str(case.get("case_id") or ""),
        "composition_count": total,
        "future_strategy_structural_hit": bool(structural_hits),
        "future_strategy_structural_hit_ids": structural_hits,
        "future_strategy_exact_hit": bool(exact_hits),
        "future_strategy_exact_hit_ids": exact_hits,
        # Compatibility alias: when an exact future criterion exists, a merely
        # structural resemblance is no longer reported as a full future hit.
        "future_strategy_hit": bool(exact_hits if exact_groups else structural_hits),
        "future_strategy_hit_ids": exact_hits if exact_groups else structural_hits,
        "over_budget_count": len(over_budget),
        "over_budget_ratio": len(over_budget) / total if total else 0.0,
        "over_budget_ids": over_budget,
        "duplicate_count": len(duplicate),
        "duplicate_ratio": len(duplicate) / total if total else 0.0,
        "duplicate_ids": duplicate,
        "invalid_combination_count": len(invalid_combinations),
        "invalid_combination_ratio": len(invalid_combinations) / total if total else 0.0,
        "invalid_combinations": invalid_combinations,
        "unsupported_memory_ids": unsupported_ids,
        "all_citations_visible": not unsupported_ids,
    }


def run_case(
    case: Mapping[str, Any],
    *,
    config_path: Path,
    actuate: bool,
) -> dict[str, Any]:
    pack, visible_ids = _build_router_pack(case)
    hidden_ids = {
        str(value)
        for value in (case.get("hidden_future") or {}).get("memory_ids", [])
    }
    leakage_ids = sorted(hidden_ids & set(visible_ids))
    if leakage_ids:
        raise RuntimeError(f"hidden future evidence leaked before inference: {leakage_ids}")
    agent, cfg = _build_agent(case, config_path)
    parent = _node_from_payload(
        dict(case.get("parent") or {}),
        stage=str((case.get("parent") or {}).get("stage") or "draft"),
    )
    agent.journal.append(parent)
    trace = run_memory_strategy_shadow(
        agent,
        parent,
        stage=str(case.get("stage") or "improve"),
        router_pack=pack,
        branch_best_metric=(case.get("metrics") or {}).get("branch_best_metric"),
        production_prompt_sha256=str(case.get("production_prompt_sha256") or "historical-replay"),
    )
    memo = dict(trace.get("memo") or {})
    evaluation = evaluate_memo(
        case,
        memo=memo,
        visible_memory_ids=visible_ids,
    )
    atomic = {}
    if (
        actuate
        and bool(case.get("atomic_actuation_enabled", True))
        and trace.get("status")
        in {"completed", "completed_with_contract_violations"}
    ):
        atomic = run_atomic_actuation_pipeline(
            agent,
            strategy_memo=memo,
            parent_code=parent.code,
            task_description=str(case.get("task_description") or ""),
            execution_output=parent.full_term_out,
            budget=dict(case.get("budget") or {}),
        )
    return {
        "schema": "mlevolve_memory_strategy_historical_replay_result_v2",
        "case_id": str(case.get("case_id") or ""),
        "task_id": str(case.get("task_id") or ""),
        "cutoff": copy.deepcopy(dict(case.get("cutoff") or {})),
        "hidden_future_id": str((case.get("hidden_future") or {}).get("id") or ""),
        "hidden_future_leakage_ids": leakage_ids,
        "visible_memory_ids": visible_ids,
        "model": str(cfg.agent.code.model),
        "strategy_trace": trace,
        "evaluation": evaluation,
        "atomic_actuation": atomic,
    }


def summarize(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    evaluations = [dict(item.get("evaluation") or {}) for item in results]
    total_compositions = sum(int(item.get("composition_count") or 0) for item in evaluations)
    return {
        "schema": "mlevolve_memory_strategy_historical_replay_summary_v2",
        "case_count": len(results),
        "future_strategy_structural_case_hits": sum(
            bool(item.get("future_strategy_structural_hit")) for item in evaluations
        ),
        "future_strategy_exact_case_hits": sum(
            bool(item.get("future_strategy_exact_hit")) for item in evaluations
        ),
        "future_strategy_case_hits": sum(bool(item.get("future_strategy_hit")) for item in evaluations),
        "future_strategy_case_hit_rate": (
            sum(bool(item.get("future_strategy_hit")) for item in evaluations) / len(results)
            if results
            else 0.0
        ),
        "composition_count": total_compositions,
        "over_budget_count": sum(int(item.get("over_budget_count") or 0) for item in evaluations),
        "duplicate_count": sum(int(item.get("duplicate_count") or 0) for item in evaluations),
        "invalid_combination_count": sum(
            int(item.get("invalid_combination_count") or 0) for item in evaluations
        ),
        "unsupported_citation_case_count": sum(
            not bool(item.get("all_citations_visible")) for item in evaluations
        ),
        "atomic_accepted_count": sum(
            str((item.get("atomic_actuation") or {}).get("status") or "") == "accepted"
            for item in results
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "mlevolve" / "config" / "config.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--actuate", action="store_true")
    args = parser.parse_args()

    packet = _load(args.cases)
    cases = list(packet.get("cases") or []) if isinstance(packet, Mapping) else list(packet)
    selected = set(args.case_id or [])
    if selected:
        cases = [case for case in cases if str(case.get("case_id") or "") in selected]
    if not cases:
        raise SystemExit("no replay cases selected")
    results = [
        run_case(case, config_path=args.config, actuate=args.actuate)
        for case in cases
    ]
    output = {
        "schema": "mlevolve_memory_strategy_historical_replay_packet_v1",
        "source_cases": str(args.cases.resolve()),
        "actuation_enabled": bool(args.actuate),
        "results": results,
        "summary": summarize(results),
    }
    _write(args.output, output)
    print(json.dumps(output["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
