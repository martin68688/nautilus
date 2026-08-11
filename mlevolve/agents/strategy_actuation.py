"""Production bridge from Memory Strategy analysis to candidate code.

The Strategy model never edits the legacy Improve/Debug prompt.  When active
mode is enabled, this module instead requires a valid Strategy Memo and
converts one hypothesis into an Atomic Actuation Contract.  Strict mode keeps
the deterministic symbol/import/diff verifier; mechanical-only mode accepts
parseable Planner JSON and checks only patch application, syntax, and count.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any, Mapping

from agents.atomic_actuation import run_atomic_actuation_pipeline
from agents.memory_strategy_agent import run_memory_strategy_active


ACTIVE_STRATEGY_SCHEMA = "mlevolve_memory_strategy_active_actuation_v1"


class MemoryStrategyActuationRejected(RuntimeError):
    """Fail-closed rejection of a required Strategy actuation transaction."""

    def __init__(self, trace: Mapping[str, Any]):
        self.trace = copy.deepcopy(dict(trace))
        reason = str(self.trace.get("reason") or self.trace.get("status") or "rejected")
        super().__init__(f"required Memory Strategy actuation rejected: {reason}")


def active_strategy_enabled(agent: Any, stage: str) -> bool:
    ext_cfg = getattr(getattr(agent, "cfg", None), "external_skill_memory", None)
    if not bool(getattr(ext_cfg, "memory_strategy_active_enabled", False)):
        return False
    stages = {
        str(value)
        for value in (getattr(ext_cfg, "memory_strategy_active_stages", []) or [])
    }
    return str(stage) in stages


def active_strategy_required(agent: Any) -> bool:
    ext_cfg = getattr(getattr(agent, "cfg", None), "external_skill_memory", None)
    return bool(getattr(ext_cfg, "memory_strategy_active_required", True))


def _budget_snapshot(agent: Any) -> dict[str, Any]:
    total = max(0.0, float(getattr(getattr(agent, "acfg", None), "time_limit", 0) or 0))
    started = getattr(agent, "search_start_time", None)
    elapsed = max(0.0, time.time() - float(started)) if started else 0.0
    remaining = max(0.0, total - elapsed) if total else 0.0
    return {
        "total_search_seconds": round(total, 6),
        "elapsed_search_seconds": round(elapsed, 6),
        "remaining_search_seconds": round(remaining, 6),
    }


def run_active_strategy_actuation(
    agent: Any,
    parent_node: Any,
    *,
    stage: str,
    router_pack: Mapping[str, Any] | None,
    branch_best_metric: float | None,
    production_prompt_sha256: str,
) -> dict[str, Any]:
    """Run the required Strategy -> Atomic Planner -> Coder transaction."""

    strategy = run_memory_strategy_active(
        agent,
        parent_node,
        stage=stage,
        router_pack=router_pack,
        branch_best_metric=branch_best_metric,
        production_prompt_sha256=production_prompt_sha256,
    )
    trace: dict[str, Any] = {
        "schema": ACTIVE_STRATEGY_SCHEMA,
        "stage": str(stage),
        "required": active_strategy_required(agent),
        "status": "rejected",
        "strategy": strategy,
        "atomic": {},
        "reason": "",
    }
    validation = dict(strategy.get("validation") or {})
    if strategy.get("status") != "completed" or validation.get("valid") is not True:
        trace["reason"] = "strategy_contract_not_completed"
        return trace
    memo = dict(strategy.get("memo") or {})
    if memo.get("decision") != "propose":
        trace["reason"] = "strategy_did_not_propose"
        return trace

    atomic = run_atomic_actuation_pipeline(
        agent,
        strategy_memo=memo,
        parent_code=str(getattr(parent_node, "code", "") or ""),
        task_description=str(getattr(agent, "task_desc", "") or ""),
        execution_output=str(getattr(parent_node, "term_out", "") or ""),
        budget=_budget_snapshot(agent),
        stage=stage,
    )
    trace["atomic"] = atomic
    if atomic.get("status") != "accepted":
        trace["reason"] = "atomic_pipeline_not_accepted"
        return trace

    planner = dict(atomic.get("planner") or {})
    coder = dict(atomic.get("coder") or {})
    plan = dict(planner.get("plan") or {})
    candidate_code = str(coder.get("candidate_code") or "")
    verdict = dict(coder.get("plan_diff_verdict") or {})
    if not candidate_code or verdict.get("valid") is not True:
        trace["reason"] = "atomic_candidate_missing_or_unverified"
        return trace

    trace.update(
        {
            "status": "accepted",
            "reason": "",
            "candidate_code": candidate_code,
            "plan": plan,
            "plan_text": json.dumps(plan, ensure_ascii=False, sort_keys=True),
            "plan_diff_verdict": verdict,
            "source_memory_ids": list(plan.get("source_memory_ids") or []),
            "prompt_record": {
                "mode": "memory_strategy_active_atomic",
                "stage": str(stage),
                "strategy_memo": memo,
                "atomic_actuation_plan": plan,
            },
        }
    )
    return trace


__all__ = [
    "ACTIVE_STRATEGY_SCHEMA",
    "MemoryStrategyActuationRejected",
    "active_strategy_enabled",
    "active_strategy_required",
    "run_active_strategy_actuation",
]
