"""Bounded Strategy-owner review of an Atomic Planner handoff.

The Atomic Planner is allowed to compile a Strategy hypothesis into executable
phases, but it is not allowed to silently change the hypothesis.  This module
adds the missing return channel: after planning, a read-only Strategy Intent
Reviewer either accepts the handoff, returns typed revision requests, or
declares the selected hypothesis non-actuatable.  It never writes code and it
cannot widen Host limits.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from typing import Any, Mapping

from llm import generate


PLAN_ALIGNMENT_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["aligned", "revise_plan", "abandon_hypothesis"],
        },
        "reason": {"type": "string"},
        "preserved_hypothesis_id": {"type": "string"},
        "preserved_source_memory_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "mismatches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": [
                            "objective",
                            "causal_mechanism",
                            "minimal_change_set",
                            "forbidden_changes",
                            "compatibility",
                            "compute_budget",
                            "phase_dependencies",
                            "fallback",
                        ],
                    },
                    "expected": {"type": "string"},
                    "actual": {"type": "string"},
                    "required_revision": {"type": "string"},
                },
                "required": [
                    "field",
                    "expected",
                    "actual",
                    "required_revision",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "verdict",
        "reason",
        "preserved_hypothesis_id",
        "preserved_source_memory_ids",
        "mismatches",
    ],
    "additionalProperties": False,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def merge_host_representation_requirements(
    contracts: Mapping[str, Any] | None,
    *,
    ext_cfg: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind environment-owned representation requirements to a candidate.

    Retrieval owns which memory cards are useful, but it cannot weaken a
    launch-time method invariant. Host requirements therefore win when they
    conflict with a selected card. The override remains explicit in a receipt
    so Planner and Strategy can reconcile it instead of silently changing the
    experiment.
    """

    merged = copy.deepcopy(dict(contracts or {}))
    required_backbone = str(
        getattr(
            ext_cfg,
            "memory_strategy_host_required_backbone_family",
            "",
        )
        or ""
    ).strip()
    require_public_pretrained = bool(
        getattr(
            ext_cfg,
            "memory_strategy_host_require_public_pretrained",
            False,
        )
    )
    required_pretrained_loader = str(
        getattr(
            ext_cfg,
            "memory_strategy_host_required_pretrained_loader",
            "",
        )
        or ""
    ).strip().lower()
    overrides: list[dict[str, str]] = []

    def bind(slot: str, required_value: str) -> None:
        previous = merged.get(slot)
        previous_value = (
            str(previous.get("required_value") or "")
            if isinstance(previous, Mapping)
            else ""
        )
        if previous_value and previous_value != required_value:
            overrides.append(
                {
                    "slot": slot,
                    "selected_value": previous_value,
                    "host_required_value": required_value,
                }
            )
        merged[slot] = {
            "required_value": required_value,
            "authority": "host_runtime_contract",
        }

    if required_backbone:
        bind("backbone_family", required_backbone)
    if require_public_pretrained:
        bind("initialization_source", "public_pretrained")
    if required_pretrained_loader:
        bind("pretrained_loader", required_pretrained_loader)

    requirements = {
        slot: str(value.get("required_value") or "")
        for slot, value in merged.items()
        if isinstance(value, Mapping)
        and value.get("authority") == "host_runtime_contract"
    }
    receipt = {
        "schema": "mlevolve_host_representation_contract_v1",
        "enabled": bool(requirements),
        "requirements": requirements,
        "selection_overrides": overrides,
    }
    return merged, receipt


def _parse_json_object(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return copy.deepcopy(dict(response))
    text = str(response or "").strip()
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Strategy alignment response is not a JSON object")
    return parsed


def _selected_composition(
    strategy_memo: Mapping[str, Any], plan: Mapping[str, Any]
) -> dict[str, Any] | None:
    hypothesis_id = str(plan.get("hypothesis_id") or "")
    for item in strategy_memo.get("candidate_compositions") or []:
        if (
            isinstance(item, Mapping)
            and str(item.get("hypothesis_id") or "") == hypothesis_id
        ):
            return copy.deepcopy(dict(item))
    return None


def _validate_review(
    review: Mapping[str, Any],
    *,
    selected_composition: Mapping[str, Any],
) -> list[str]:
    violations: list[str] = []
    verdict = str(review.get("verdict") or "")
    if verdict not in {"aligned", "revise_plan", "abandon_hypothesis"}:
        violations.append("invalid alignment verdict")
    expected_hypothesis_id = str(selected_composition.get("hypothesis_id") or "")
    if str(review.get("preserved_hypothesis_id") or "") != expected_hypothesis_id:
        violations.append("intent reviewer changed hypothesis_id")
    expected_sources = {
        str(value) for value in selected_composition.get("source_memory_ids") or []
    }
    reviewed_sources = {
        str(value) for value in review.get("preserved_source_memory_ids") or []
    }
    if reviewed_sources != expected_sources:
        violations.append("intent reviewer changed source_memory_ids")
    mismatches = list(review.get("mismatches") or [])
    if verdict == "aligned" and mismatches:
        violations.append("aligned verdict must not contain mismatches")
    if verdict == "revise_plan" and not mismatches:
        violations.append("revise_plan verdict requires typed mismatches")
    if verdict == "abandon_hypothesis" and not str(review.get("reason") or "").strip():
        violations.append("abandon_hypothesis verdict requires a reason")
    for index, mismatch in enumerate(mismatches):
        if not isinstance(mismatch, Mapping):
            violations.append(f"mismatches[{index}] is not an object")
            continue
        missing = [
            key
            for key in ("field", "expected", "actual", "required_revision")
            if not str(mismatch.get(key) or "").strip()
        ]
        if missing:
            violations.append(f"mismatches[{index}] has empty fields: {missing}")
    return violations


def _alignment_prompt(
    *,
    selected_composition: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage: str,
) -> dict[str, str]:
    system = (
        "You are the read-only Strategy Intent Reviewer. The Strategy Agent owns the "
        "hypothesis; the Atomic Planner only compiles it into an executable plan. Compare "
        "the plan with the selected Strategy composition and decide whether the handoff "
        "preserves the same objective, causal mechanism, indispensable minimal changes, "
        "forbidden changes, compatibility requirements, compute budget, ordered interfaces, "
        "and falsification fallback. A plan may defer work only when the current slice remains "
        "independently executable and tests the same causal mechanism. Return revise_plan for "
        "repairable drift. Return abandon_hypothesis only when no complete independently "
        "testable slice can fit the supplied plan. You cannot invent a new hypothesis, edit "
        "code, relax Host limits, add source memories, or approve a plan merely because it is "
        "well formed. Output exactly one JSON object matching RESPONSE_SCHEMA.\n\n"
        f"RESPONSE_SCHEMA:\n{_canonical_json(PLAN_ALIGNMENT_REVIEW_SCHEMA)}"
    )
    user = _canonical_json(
        {
            "stage": str(stage),
            "selected_strategy_composition": dict(selected_composition),
            "planner_handoff": dict(plan),
        }
    )
    return {"system": system, "user": user, "assistant": "{"}


def run_strategy_plan_alignment_review(
    agent: Any,
    *,
    strategy_memo: Mapping[str, Any],
    plan: Mapping[str, Any],
    stage: str,
    review_attempt: int = 0,
) -> dict[str, Any]:
    """Return an auditable owner verdict for one Planner handoff."""

    started = time.monotonic()
    selected = _selected_composition(strategy_memo, plan)
    base = {
        "schema": "mlevolve_strategy_plan_alignment_trace_v1",
        "strategy_memo_sha256": hashlib.sha256(
            _canonical_json(strategy_memo).encode("utf-8")
        ).hexdigest(),
        "plan_sha256": hashlib.sha256(
            _canonical_json(plan).encode("utf-8")
        ).hexdigest(),
        "stage": str(stage),
        "review_attempt": int(review_attempt),
    }
    if selected is None:
        return {
            **base,
            "status": "revise",
            "verdict": "revise_plan",
            "reason": "planner selected a hypothesis absent from the Strategy Memo",
            "review": {},
            "validation": {
                "valid": True,
                "violations": [],
            },
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }

    prompt = _alignment_prompt(
        selected_composition=selected,
        plan=plan,
        stage=stage,
    )
    try:
        query_fn = getattr(agent, "_strategy_plan_alignment_query_fn", None)
        if callable(query_fn):
            response = query_fn(
                prompt=copy.deepcopy(prompt),
                selected_composition=copy.deepcopy(selected),
                plan=copy.deepcopy(dict(plan)),
                json_schema=copy.deepcopy(PLAN_ALIGNMENT_REVIEW_SCHEMA),
                review_attempt=int(review_attempt),
            )
        else:
            ext_cfg = getattr(agent.cfg, "external_skill_memory", None)
            inherited_model = str(getattr(agent.acfg.code, "model", "") or "")
            strategy_model = str(
                getattr(ext_cfg, "memory_strategy_model", inherited_model)
                or inherited_model
            )
            review_cfg = copy.deepcopy(agent.cfg)
            review_cfg.agent.code.model = strategy_model
            response = generate(
                prompt=prompt,
                cfg=review_cfg,
                temperature=0.0,
                max_tokens=int(
                    getattr(
                        ext_cfg,
                        "memory_strategy_plan_alignment_max_tokens",
                        4000,
                    )
                    or 4000
                ),
                json_schema=PLAN_ALIGNMENT_REVIEW_SCHEMA,
                max_retries=2,
            )
        review = _parse_json_object(response)
        violations = _validate_review(
            review,
            selected_composition=selected,
        )
        verdict = str(review.get("verdict") or "")
        status = (
            "accepted"
            if not violations and verdict == "aligned"
            else "revise"
            if not violations and verdict == "revise_plan"
            else "rejected"
            if not violations and verdict == "abandon_hypothesis"
            else "failed"
        )
        return {
            **base,
            "status": status,
            "verdict": verdict,
            "reason": str(review.get("reason") or ""),
            "review": review,
            "validation": {
                "schema": "mlevolve_strategy_plan_alignment_validation_v1",
                "valid": not violations,
                "violations": violations,
            },
            "response_sha256": hashlib.sha256(
                _canonical_json(response).encode("utf-8")
            ).hexdigest(),
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }
    except Exception as exc:
        return {
            **base,
            "status": "failed",
            "verdict": "",
            "reason": f"{type(exc).__name__}: {exc}",
            "review": {},
            "validation": {
                "schema": "mlevolve_strategy_plan_alignment_validation_v1",
                "valid": False,
                "violations": [f"alignment review failed: {type(exc).__name__}: {exc}"],
            },
            "elapsed_seconds": round(time.monotonic() - started, 6),
        }


__all__ = [
    "PLAN_ALIGNMENT_REVIEW_SCHEMA",
    "merge_host_representation_requirements",
    "run_strategy_plan_alignment_review",
]
