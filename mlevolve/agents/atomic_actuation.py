"""Isolated Strategy Memo -> Atomic Planner -> bounded Coder pipeline.

This module is intentionally not called by the live Improve/Debug path during
the Strategy Agent's shadow phase.  It exists so historical slices can test
whether a useful global hypothesis survives conversion into one executable,
machine-bounded code edit.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import re
import time
from typing import Any, Iterable, Mapping

from llm import generate
from agents.coder.diff_coder.patcher import SearchReplacePatcher
from agents.coder.diff_coder.prompts import (
    DIFF_SYS_FORMAT,
    build_base_diff_instructions,
    build_diff_format_suffix,
)
from agents.memory_strategy_agent import payload_sha256


logger = logging.getLogger("MLEvolve")


ATOMIC_ACTUATION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis_id": {"type": "string"},
        "objective": {"type": "string"},
        "source_memory_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "allowed_modules": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "data_processing_and_feature_engineering",
                    "model_design",
                    "training_evaluation",
                ],
            },
        },
        "allowed_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "change_id": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": ["modify", "add", "delete"],
                    },
                    "target_symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": {"type": "string"},
                },
                "required": [
                    "change_id",
                    "operation",
                    "target_symbols",
                    "description",
                ],
                "additionalProperties": True,
            },
        },
        "allowed_new_imports": {
            "type": "array",
            "items": {"type": "string"},
        },
        "forbidden_symbols": {
            "type": "array",
            "items": {"type": "string"},
        },
        "forbidden_code_patterns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "preserve_invariants": {
            "type": "array",
            "items": {"type": "string"},
        },
        "compatibility_checks": {
            "type": "array",
            "items": {"type": "string"},
        },
        "estimated_compute_seconds": {"type": "integer"},
        "max_patches": {"type": "integer"},
        "expected_mechanism": {"type": "string"},
        "falsification_condition": {"type": "string"},
    },
    "required": [
        "hypothesis_id",
        "objective",
        "source_memory_ids",
        "allowed_modules",
        "allowed_changes",
        "allowed_new_imports",
        "forbidden_symbols",
        "forbidden_code_patterns",
        "preserve_invariants",
        "compatibility_checks",
        "estimated_compute_seconds",
        "max_patches",
        "expected_mechanism",
        "falsification_condition",
    ],
    "additionalProperties": True,
}


ATOMIC_STAGED_ACTUATION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "roadmap_id": {"type": "string"},
        "hypothesis_id": {"type": "string"},
        "objective": {"type": "string"},
        "source_memory_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "roadmap_complete": {"type": "boolean"},
        "phases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    **copy.deepcopy(ATOMIC_ACTUATION_PLAN_SCHEMA["properties"]),
                    "phase_id": {"type": "string"},
                    "phase_index": {"type": "integer"},
                    "depends_on_phase_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    *ATOMIC_ACTUATION_PLAN_SCHEMA["required"],
                    "phase_id",
                    "phase_index",
                    "depends_on_phase_ids",
                ],
                "additionalProperties": True,
            },
        },
    },
    "required": [
        "roadmap_id",
        "hypothesis_id",
        "objective",
        "source_memory_ids",
        "roadmap_complete",
        "phases",
    ],
    "additionalProperties": True,
}


_ATOMIC_REQUIRED_KEYS = tuple(ATOMIC_ACTUATION_PLAN_SCHEMA["required"])
_STAGED_REQUIRED_KEYS = tuple(ATOMIC_STAGED_ACTUATION_PLAN_SCHEMA["required"])
_ALLOWED_MODULES = {
    "data_processing_and_feature_engineering",
    "model_design",
    "training_evaluation",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


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
        raise ValueError("atomic planner response is not a JSON object")
    return parsed


def _composition_by_id(strategy_memo: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("hypothesis_id")): dict(item)
        for item in (strategy_memo.get("candidate_compositions") or [])
        if isinstance(item, Mapping) and item.get("hypothesis_id")
    }


def validate_atomic_plan(
    plan: Mapping[str, Any],
    *,
    strategy_memo: Mapping[str, Any],
    max_modules: int,
    max_changes: int,
    max_patches: int,
    max_symbols: int | None = None,
    parent_code: str | None = None,
    stage: str = "",
    debug_targeted_repair_only: bool = False,
) -> dict[str, Any]:
    violations: list[str] = []
    missing_keys = [key for key in _ATOMIC_REQUIRED_KEYS if key not in plan]
    if missing_keys:
        violations.append(f"missing required top-level keys: {missing_keys}")
    compositions = _composition_by_id(strategy_memo)
    hypothesis_id = str(plan.get("hypothesis_id") or "")
    if hypothesis_id not in compositions:
        violations.append("hypothesis_id is not present in the Strategy Memo")
        source_ids: set[str] = set()
    else:
        selected_composition = compositions[hypothesis_id]
        source_ids = {
            str(value)
            for value in (selected_composition.get("source_memory_ids") or [])
        }
        if (
            str(stage) == "debug"
            and debug_targeted_repair_only
            and str(selected_composition.get("novelty_kind") or "")
            not in {"targeted_repair", "single_memory_actuation"}
        ):
            violations.append(
                "Debug actuation must select a targeted_repair or bounded "
                "single_memory_actuation hypothesis"
            )
    cited = {str(value) for value in (plan.get("source_memory_ids") or [])}
    if not cited:
        violations.append("source_memory_ids must not be empty")
    if not cited <= source_ids:
        violations.append(
            f"atomic plan cites IDs outside selected hypothesis: {sorted(cited - source_ids)}"
        )

    modules = list(plan.get("allowed_modules") or [])
    if not 1 <= len(modules) <= int(max_modules):
        violations.append(f"allowed_modules must contain 1..{int(max_modules)} modules")
    invalid_modules = sorted(set(str(value) for value in modules) - _ALLOWED_MODULES)
    if invalid_modules:
        violations.append(f"invalid allowed_modules: {invalid_modules}")
    changes = list(plan.get("allowed_changes") or [])
    if not 1 <= len(changes) <= int(max_changes):
        violations.append(f"allowed_changes must contain 1..{int(max_changes)} changes")
    seen_change_ids: set[str] = set()
    seen_target_symbols: set[str] = set()
    available_symbols: set[str] = set()
    if parent_code is not None:
        try:
            available_symbols = set(_top_level_units(parent_code)[0])
        except SyntaxError as exc:
            violations.append(f"parent code is not parseable: {exc}")
    for index, change in enumerate(changes):
        if not isinstance(change, Mapping):
            violations.append(f"allowed_changes[{index}] is not an object")
            continue
        change_id = str(change.get("change_id") or "")
        if not change_id or change_id in seen_change_ids:
            violations.append(
                f"allowed_changes[{index}] has missing/duplicate change_id"
            )
        seen_change_ids.add(change_id)
        targets = [str(value) for value in (change.get("target_symbols") or [])]
        if not targets:
            violations.append(
                f"{change_id or index} must name at least one target symbol"
            )
        duplicate_targets = sorted(set(targets) & seen_target_symbols)
        if duplicate_targets:
            violations.append(
                f"{change_id or index} repeats target symbols already assigned to another "
                f"change: {duplicate_targets}"
            )
        seen_target_symbols.update(targets)
        if str(change.get("operation") or "") not in {"modify", "add", "delete"}:
            violations.append(f"{change_id or index} has invalid operation")
        if parent_code is not None and str(change.get("operation") or "") in {
            "modify",
            "delete",
        }:
            unknown_targets = sorted(set(targets) - available_symbols)
            if unknown_targets:
                violations.append(
                    f"{change_id or index} targets non-top-level symbols: "
                    f"{unknown_targets}; available symbols are {sorted(available_symbols)}"
                )
    if max_symbols is not None and int(max_symbols) > 0:
        if len(seen_target_symbols) > int(max_symbols):
            violations.append(
                f"target_symbols must contain at most {int(max_symbols)} distinct "
                f"symbols per phase; found {len(seen_target_symbols)}"
            )
    try:
        requested_patches = int(plan.get("max_patches", 0))
    except (TypeError, ValueError):
        requested_patches = 0
    if not 1 <= requested_patches <= int(max_patches):
        violations.append(f"max_patches must be in [1, {int(max_patches)}]")
    try:
        if int(plan.get("estimated_compute_seconds", -1)) < 0:
            violations.append("estimated_compute_seconds must be non-negative")
    except (TypeError, ValueError):
        violations.append("estimated_compute_seconds must be an integer")
    if not list(plan.get("compatibility_checks") or []):
        violations.append("compatibility_checks must not be empty")
    if not list(plan.get("preserve_invariants") or []):
        violations.append("preserve_invariants must not be empty")
    if not str(plan.get("objective") or "").strip():
        violations.append("objective must not be empty")
    if not str(plan.get("expected_mechanism") or "").strip():
        violations.append("expected_mechanism must not be empty")
    if not str(plan.get("falsification_condition") or "").strip():
        violations.append("falsification_condition must not be empty")
    allowed_imports = [str(value) for value in (plan.get("allowed_new_imports") or [])]
    forbidden_patterns = [
        str(value) for value in (plan.get("forbidden_code_patterns") or [])
    ]
    conflicting_import_rules: list[dict[str, str]] = []
    for allowed_import in allowed_imports:
        parts = allowed_import.split(".")
        module_name = ".".join(parts[:-1])
        imported_name = parts[-1]
        for forbidden_pattern in forbidden_patterns:
            if allowed_import in forbidden_pattern or (
                module_name
                and module_name in forbidden_pattern
                and imported_name in forbidden_pattern
            ):
                conflicting_import_rules.append(
                    {
                        "allowed_import": allowed_import,
                        "forbidden_pattern": forbidden_pattern,
                    }
                )
    if conflicting_import_rules:
        violations.append(
            f"allowed_new_imports conflict with forbidden_code_patterns: "
            f"{conflicting_import_rules}"
        )
    return {
        "schema": "mlevolve_atomic_actuation_plan_validation_v1",
        "valid": not violations,
        "violations": violations,
        "selected_hypothesis_id": hypothesis_id,
        "available_hypothesis_ids": sorted(compositions),
        "available_top_level_symbols": sorted(available_symbols),
    }


def validate_staged_atomic_plan(
    roadmap: Mapping[str, Any],
    *,
    strategy_memo: Mapping[str, Any],
    max_modules: int,
    max_changes: int,
    max_patches: int,
    max_phases: int,
    max_symbols: int | None = None,
    parent_code: str | None = None,
    stage: str = "",
    debug_targeted_repair_only: bool = False,
    require_complete_roadmap: bool = True,
) -> dict[str, Any]:
    """Validate a complete roadmap made of independently bounded Coder phases."""

    violations: list[str] = []
    missing_keys = [key for key in _STAGED_REQUIRED_KEYS if key not in roadmap]
    if missing_keys:
        violations.append(f"missing required roadmap keys: {missing_keys}")
    roadmap_id = str(roadmap.get("roadmap_id") or "").strip()
    if not roadmap_id:
        violations.append("roadmap_id must not be empty")
    if not str(roadmap.get("objective") or "").strip():
        violations.append("roadmap objective must not be empty")
    if require_complete_roadmap and roadmap.get("roadmap_complete") is not True:
        violations.append(
            "roadmap_complete must be true; split oversized work into phases instead of "
            "returning a partial candidate"
        )

    phases = list(roadmap.get("phases") or [])
    if not 1 <= len(phases) <= int(max_phases):
        violations.append(f"phases must contain 1..{int(max_phases)} bounded phases")
    roadmap_hypothesis = str(roadmap.get("hypothesis_id") or "")
    roadmap_sources = {str(value) for value in (roadmap.get("source_memory_ids") or [])}
    seen_phase_ids: set[str] = set()
    seen_change_ids: set[str] = set()
    phase_validations: list[dict[str, Any]] = []
    for index, raw_phase in enumerate(phases):
        if not isinstance(raw_phase, Mapping):
            violations.append(f"phases[{index}] is not an object")
            continue
        phase = dict(raw_phase)
        phase_id = str(phase.get("phase_id") or "")
        if not phase_id or phase_id in seen_phase_ids:
            violations.append(f"phases[{index}] has missing/duplicate phase_id")
        previous_phase = phases[index - 1] if index else None
        previous_phase_id = (
            str(previous_phase.get("phase_id") or "")
            if isinstance(previous_phase, Mapping)
            else ""
        )
        expected_dependencies = [] if index == 0 else [previous_phase_id]
        dependencies = [
            str(value) for value in (phase.get("depends_on_phase_ids") or [])
        ]
        if dependencies != expected_dependencies:
            violations.append(
                f"phases[{index}].depends_on_phase_ids must be exactly "
                f"{expected_dependencies}"
            )
        try:
            phase_index = int(phase.get("phase_index"))
        except (TypeError, ValueError):
            phase_index = -1
        if phase_index != index + 1:
            violations.append(f"phases[{index}].phase_index must be {index + 1}")
        if str(phase.get("hypothesis_id") or "") != roadmap_hypothesis:
            violations.append(f"phases[{index}] must keep roadmap hypothesis_id")
        if {
            str(value) for value in (phase.get("source_memory_ids") or [])
        } != roadmap_sources:
            violations.append(
                f"phases[{index}] must keep the exact roadmap source_memory_ids"
            )
        phase_validation = validate_atomic_plan(
            phase,
            strategy_memo=strategy_memo,
            max_modules=max_modules,
            max_changes=max_changes,
            max_patches=max_patches,
            max_symbols=max_symbols,
            parent_code=parent_code,
            stage=stage,
            debug_targeted_repair_only=debug_targeted_repair_only,
        )
        phase_validations.append(phase_validation)
        violations.extend(
            f"phases[{index}]: {value}"
            for value in (phase_validation.get("violations") or [])
        )
        for change in phase.get("allowed_changes") or []:
            if not isinstance(change, Mapping):
                continue
            change_id = str(change.get("change_id") or "")
            if change_id and change_id in seen_change_ids:
                violations.append(
                    f"phases[{index}] repeats change_id from an earlier phase: {change_id}"
                )
            seen_change_ids.add(change_id)
        seen_phase_ids.add(phase_id)
    return {
        "schema": "mlevolve_staged_atomic_plan_validation_v1",
        "valid": not violations,
        "violations": violations,
        "phase_count": len(phases),
        "max_phases": int(max_phases),
        "max_symbols_per_phase": int(max_symbols or 0),
        "phase_validations": phase_validations,
    }


def _atomic_limits(agent: Any, stage: str) -> dict[str, int]:
    ext_cfg = getattr(agent.cfg, "external_skill_memory", None)
    limits = {
        "max_modules": int(
            getattr(ext_cfg, "memory_strategy_atomic_max_modules", 2) or 2
        ),
        "max_changes": int(
            getattr(ext_cfg, "memory_strategy_atomic_max_changes", 3) or 3
        ),
        "max_patches": int(
            getattr(ext_cfg, "memory_strategy_atomic_max_patches", 6) or 6
        ),
        "max_symbols": int(
            getattr(ext_cfg, "memory_strategy_atomic_max_symbols_per_phase", 64) or 64
        ),
    }
    if str(stage) == "improve":
        limits["max_modules"] = min(
            limits["max_modules"],
            int(getattr(ext_cfg, "experiment_r_improve_max_modules", 2) or 2),
        )
        limits["max_patches"] = min(
            limits["max_patches"],
            int(getattr(ext_cfg, "experiment_r_improve_max_patches", 6) or 6),
        )
    elif str(stage) == "debug":
        limits["max_modules"] = min(
            limits["max_modules"],
            int(getattr(ext_cfg, "memory_strategy_atomic_debug_max_modules", 1) or 1),
        )
        limits["max_changes"] = min(
            limits["max_changes"],
            int(getattr(ext_cfg, "memory_strategy_atomic_debug_max_changes", 2) or 2),
        )
        limits["max_symbols"] = min(
            limits["max_symbols"],
            int(
                getattr(
                    ext_cfg,
                    "memory_strategy_atomic_debug_max_symbols_per_phase",
                    4,
                )
                or 4
            ),
        )
        limits["max_patches"] = min(
            limits["max_patches"],
            int(getattr(ext_cfg, "experiment_r_debug_max_patches", 3) or 3),
        )
    return limits


def _planner_prompt(
    strategy_memo: Mapping[str, Any],
    *,
    parent_code: str,
    budget: Mapping[str, Any] | None,
    limits: Mapping[str, int],
    stage: str = "",
    previous_plan: Mapping[str, Any] | None = None,
    contract_violations: Iterable[str] = (),
    coder_replan_feedback: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    stage_contract = ""
    if str(stage) == "debug":
        stage_contract = (
            " This is a Debug transaction: select only a targeted_repair that fixes the "
            "narrowest demonstrated root cause. The first executable phase must only make "
            "the failed parent runnable; defer OOF, calibration, ensembling, feature expansion, "
            "and performance tuning to later nodes unless one is itself the demonstrated root "
            "cause. Preserve the parent model family, feature families, split, ensemble, loss, "
            "optimizer, and submission variant."
        )
    system = (
        "You are the Atomic Actuation Planner. Select exactly one hypothesis from the "
        "Strategy Memo and convert its smallest independently executable phase into one "
        "falsifiable experiment. A Strategy hypothesis is a roadmap: you may defer later "
        "steps from its minimal_change_set when the current phase can be executed and evaluated "
        "on its own. Record deferred work in optional additional fields, but never bundle it "
        "into the current allowed_changes. You do not invent a "
        "new strategy and you do not write code. Name exact Python top-level target symbols: "
        "function/class names, __imports__, or __module__ for guarded top-level execution. "
        "Keep the allowed set narrow enough for a host verifier. The Coder will be rejected if "
        "it changes anything outside this set. Preserve the evaluation and submission protocol. "
        "Prefer recommended_hypothesis_id when it fits the hard limits; otherwise select another "
        "existing hypothesis that can be executed atomically. Staging must preserve the selected "
        "hypothesis's mechanism and citations; it must not turn it into a different experiment. "
        "Use the exact field names in "
        "RESPONSE_SCHEMA; fields such as experiment, modules, changes, or memory_ids are invalid. "
        f"Output one JSON object only.{stage_contract}\n\nRESPONSE_SCHEMA:\n"
        + _canonical_json(ATOMIC_ACTUATION_PLAN_SCHEMA)
    )
    payload: dict[str, Any] = {
        "strategy_memo": strategy_memo,
        "budget": dict(budget or {}),
        "hard_limits": dict(limits),
        "stage": str(stage),
        "parent_code": parent_code,
    }
    violations = [str(value) for value in contract_violations]
    if previous_plan is not None or violations:
        payload["contract_repair_required"] = {
            "violations": violations,
            "previous_response": dict(previous_plan or {}),
            "instruction": (
                "Return a corrected complete Atomic Actuation Contract for the same selected "
                "Strategy hypothesis. Keep only the smallest independently executable phase "
                "needed now and defer the remainder instead of changing the mechanism."
            ),
        }
    if coder_replan_feedback:
        allow_hypothesis_switch = bool(
            (coder_replan_feedback or {}).get("allow_hypothesis_switch", False)
        )
        replan_mode = str(
            (coder_replan_feedback or {}).get("replan_mode") or "decompose"
        )
        rejected_hypothesis_ids = list(
            (coder_replan_feedback or {}).get("rejected_hypothesis_ids") or []
        )
        payload["coder_replan_required"] = {
            "previous_plan": dict(
                (coder_replan_feedback or {}).get("previous_plan") or {}
            ),
            "coder_verdict": dict(
                (coder_replan_feedback or {}).get("coder_verdict") or {}
            ),
            "allow_hypothesis_switch": allow_hypothesis_switch,
            "replan_mode": replan_mode,
            "rejected_hypothesis_ids": rejected_hypothesis_ids,
            "instruction": (
                "The selected roadmap remained non-actuatable after bounded decomposition. "
                "Select a different existing Strategy hypothesis whose smallest complete test "
                "has the fewest modules, symbols, imports, and patches. Do not select any "
                "rejected_hypothesis_id and do not invent a new hypothesis."
                if allow_hypothesis_switch
                else (
                    "The prior Coder stayed inside the patch cap but touched undeclared "
                    "top-level symbols. Reconcile only the machine-observed boundary: keep "
                    "the same hypothesis, objective, and modules; add only exact "
                    "unauthorized_changed_symbols or new_imports from the Coder verdict. "
                    "Do not authorize any unobserved symbol or widen the experiment."
                    if replan_mode == "scope_reconciliation"
                    else "The Coder could not implement the previous plan inside its verified boundary. "
                    "Keep the same hypothesis_id and source_memory_ids, but decompose the roadmap: "
                    "return a strictly smaller first phase that independently runs and tests one "
                    "mechanism. Remove or defer every change, symbol, and import not required for "
                    "that phase. The host will reject a same-sized plan: allowed modules, changes, "
                    "target symbols, new imports, and max_patches may not increase, and at least one "
                    "of those boundaries must strictly decrease. For Debug, the phase must only "
                    "repair the observed exception."
                )
            ),
        }
    user = _canonical_json(payload)
    return {"system": system, "user": user, "assistant": "{"}


def run_atomic_actuation_planner(
    agent: Any,
    *,
    strategy_memo: Mapping[str, Any],
    parent_code: str,
    budget: Mapping[str, Any] | None = None,
    stage: str = "",
    coder_replan_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    ext_cfg = getattr(agent.cfg, "external_skill_memory", None)
    limits = _atomic_limits(agent, stage)
    started = time.monotonic()
    if not _composition_by_id(strategy_memo):
        return {
            "schema": "mlevolve_atomic_actuation_planner_trace_v1",
            "status": "rejected",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "strategy_memo_sha256": payload_sha256(strategy_memo),
            "parent_code_sha256": hashlib.sha256(
                parent_code.encode("utf-8")
            ).hexdigest(),
            "reason": "Strategy Memo has no candidate_compositions",
            "limits": limits,
            "contract_attempts": [],
        }
    try:
        contract_retries = int(
            getattr(
                ext_cfg,
                "memory_strategy_atomic_planner_contract_retries",
                2,
            )
            or 0
        )
        plan: dict[str, Any] = {}
        validation: dict[str, Any] = {
            "valid": False,
            "violations": ["planner did not produce a plan"],
        }
        contract_attempts: list[dict[str, Any]] = []
        for contract_attempt in range(contract_retries + 1):
            prompt = _planner_prompt(
                strategy_memo,
                parent_code=parent_code,
                budget=budget,
                limits=limits,
                stage=stage,
                previous_plan=plan if contract_attempt else None,
                contract_violations=(validation.get("violations") or [])
                if contract_attempt
                else (),
                coder_replan_feedback=coder_replan_feedback,
            )
            query_fn = getattr(agent, "_atomic_planner_query_fn", None)
            if callable(query_fn):
                response = query_fn(
                    prompt=copy.deepcopy(prompt),
                    strategy_memo=copy.deepcopy(dict(strategy_memo)),
                    json_schema=copy.deepcopy(ATOMIC_ACTUATION_PLAN_SCHEMA),
                    contract_attempt=contract_attempt,
                )
            else:
                response = generate(
                    prompt=prompt,
                    cfg=agent.cfg,
                    temperature=0.0,
                    max_tokens=4000,
                    json_schema=ATOMIC_ACTUATION_PLAN_SCHEMA,
                    max_retries=2,
                )
            try:
                plan = _parse_json_object(response)
                validation = validate_atomic_plan(
                    plan,
                    strategy_memo=strategy_memo,
                    max_modules=limits["max_modules"],
                    max_changes=limits["max_changes"],
                    max_patches=limits["max_patches"],
                    max_symbols=limits["max_symbols"],
                    parent_code=parent_code,
                    stage=stage,
                    debug_targeted_repair_only=bool(
                        getattr(
                            ext_cfg,
                            "memory_strategy_atomic_debug_targeted_repair_only",
                            True,
                        )
                    ),
                )
                if validation.get("valid") and coder_replan_feedback:
                    if bool(
                        (coder_replan_feedback or {}).get(
                            "allow_hypothesis_switch", False
                        )
                    ):
                        rejected_ids = {
                            str(value)
                            for value in (
                                (coder_replan_feedback or {}).get(
                                    "rejected_hypothesis_ids"
                                )
                                or []
                            )
                        }
                        decomposition_violations = (
                            [
                                "alternate atomic plan must select a different "
                                "non-rejected Strategy hypothesis"
                            ]
                            if str(plan.get("hypothesis_id") or "") in rejected_ids
                            else []
                        )
                    elif (
                        str(
                            (coder_replan_feedback or {}).get("replan_mode")
                            or "decompose"
                        )
                        == "scope_reconciliation"
                    ):
                        decomposition_violations = validate_scope_reconciliation(
                            plan,
                            previous_plan=dict(
                                (coder_replan_feedback or {}).get("previous_plan") or {}
                            ),
                            coder_verdict=dict(
                                (coder_replan_feedback or {}).get("coder_verdict") or {}
                            ),
                        )
                    else:
                        decomposition_violations = validate_decomposed_replan(
                            plan,
                            previous_plan=dict(
                                (coder_replan_feedback or {}).get("previous_plan") or {}
                            ),
                        )
                    if decomposition_violations:
                        validation = copy.deepcopy(validation)
                        validation["valid"] = False
                        validation["violations"] = [
                            *list(validation.get("violations") or []),
                            *decomposition_violations,
                        ]
            except Exception as exc:
                plan = {}
                validation = {
                    "schema": "mlevolve_atomic_actuation_plan_validation_v1",
                    "valid": False,
                    "violations": [
                        f"response parse failed: {type(exc).__name__}: {exc}"
                    ],
                    "selected_hypothesis_id": "",
                    "available_hypothesis_ids": sorted(
                        _composition_by_id(strategy_memo)
                    ),
                }
            contract_attempts.append(
                {
                    "attempt": contract_attempt + 1,
                    "response_sha256": hashlib.sha256(
                        _canonical_json(response).encode("utf-8")
                    ).hexdigest(),
                    "valid": bool(validation.get("valid")),
                    "violations": list(validation.get("violations") or []),
                }
            )
            if validation.get("valid"):
                break
        return {
            "schema": "mlevolve_atomic_actuation_planner_trace_v1",
            "status": "accepted" if validation["valid"] else "rejected",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "strategy_memo_sha256": payload_sha256(strategy_memo),
            "parent_code_sha256": hashlib.sha256(
                parent_code.encode("utf-8")
            ).hexdigest(),
            "plan": plan,
            "plan_sha256": payload_sha256(plan),
            "validation": validation,
            "limits": limits,
            "contract_attempts": contract_attempts,
        }
    except Exception as exc:
        return {
            "schema": "mlevolve_atomic_actuation_planner_trace_v1",
            "status": "failed",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "strategy_memo_sha256": payload_sha256(strategy_memo),
            "parent_code_sha256": hashlib.sha256(
                parent_code.encode("utf-8")
            ).hexdigest(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "limits": limits,
        }


def _staged_planner_prompt(
    strategy_memo: Mapping[str, Any],
    *,
    parent_code: str,
    budget: Mapping[str, Any] | None,
    limits: Mapping[str, int],
    max_phases: int,
    stage: str,
    previous_roadmap: Mapping[str, Any] | None = None,
    contract_violations: Iterable[str] = (),
) -> dict[str, str]:
    debug_contract = ""
    if str(stage) == "debug":
        debug_contract = (
            " Debug phases must implement only the narrow causal repair for the observed "
            "failure. Do not include model upgrades, OOF, calibration, ensembling, or feature "
            "expansion unless it is itself required to remove that exact exception. A single "
            "logical repair may name several coupled top-level symbols inside one allowed_change; "
            "do not drop a required symbol merely to reduce the allowed_changes count."
        )
    system = (
        "You are the Staged Atomic Actuation Planner. Select exactly one existing Strategy "
        "hypothesis and compile its COMPLETE executable modification into an ordered roadmap. "
        "Every phase is a strict machine contract for a Coder. If the modification is too large "
        "for one phase, split it across sequential phases; never omit required work, silently "
        "widen a phase, or return an incomplete candidate. A monolithic top-level symbol may "
        "appear in more than one phase only when each phase performs a distinct necessary step. "
        "Phase 2 will receive Phase 1's verified code, and so on. Every phase "
        "must keep the same hypothesis_id and exact source_memory_ids as the roadmap. Use "
        "depends_on_phase_ids to form one linear chain. The first phase has no dependency; each "
        "later phase depends only on the immediately preceding phase. Per-phase hard limits are "
        f"{_canonical_json(limits)} and the roadmap may contain at most {int(max_phases)} phases. "
        "Planner target_symbols are exact requirements, not suggestions: the Coder must perform "
        "the declared add/modify/delete operation on every target and is forbidden to touch any "
        "other symbol. Set roadmap_complete=true only after listing every phase needed to realize "
        "the selected hypothesis. Preserve evaluation and submission invariants. Output one JSON "
        f"object matching RESPONSE_SCHEMA and no prose.{debug_contract}\n\nRESPONSE_SCHEMA:\n"
        + _canonical_json(ATOMIC_STAGED_ACTUATION_PLAN_SCHEMA)
    )
    payload: dict[str, Any] = {
        "strategy_memo": strategy_memo,
        "budget": dict(budget or {}),
        "stage": str(stage),
        "per_phase_hard_limits": dict(limits),
        "max_phases": int(max_phases),
        "parent_code": parent_code,
    }
    violations = [str(value) for value in contract_violations]
    if previous_roadmap is not None or violations:
        payload["contract_repair_required"] = {
            "previous_response": dict(previous_roadmap or {}),
            "violations": violations,
            "instruction": (
                "Return a corrected complete roadmap for the same Strategy hypothesis. If one "
                "phase exceeded a module/change/patch boundary, split that phase into two smaller "
                "ordered phases rather than deleting any required change. If a logical Debug "
                "repair spans several coupled symbols, keep them together as target_symbols under "
                "one allowed_change. Do not change citations or introduce a new experiment."
            ),
        }
    return {"system": system, "user": _canonical_json(payload), "assistant": "{"}


def run_atomic_staged_actuation_planner(
    agent: Any,
    *,
    strategy_memo: Mapping[str, Any],
    parent_code: str,
    budget: Mapping[str, Any] | None = None,
    stage: str = "",
) -> dict[str, Any]:
    ext_cfg = getattr(agent.cfg, "external_skill_memory", None)
    limits = _atomic_limits(agent, stage)
    max_phases = max(
        1,
        int(getattr(ext_cfg, "memory_strategy_atomic_max_phases", 3) or 3),
    )
    require_complete = bool(
        getattr(
            ext_cfg,
            "memory_strategy_atomic_require_complete_roadmap",
            True,
        )
    )
    started = time.monotonic()
    if not _composition_by_id(strategy_memo):
        return {
            "schema": "mlevolve_staged_atomic_planner_trace_v1",
            "status": "rejected",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "reason": "Strategy Memo has no candidate_compositions",
            "limits": limits,
            "max_phases": max_phases,
            "contract_attempts": [],
        }
    try:
        contract_retries = max(
            0,
            int(
                getattr(
                    ext_cfg,
                    "memory_strategy_atomic_planner_contract_retries",
                    2,
                )
                or 0
            ),
        )
        roadmap: dict[str, Any] = {}
        validation: dict[str, Any] = {
            "valid": False,
            "violations": ["planner did not produce a staged roadmap"],
        }
        contract_attempts: list[dict[str, Any]] = []
        query_fn = getattr(agent, "_atomic_planner_query_fn", None)
        for contract_attempt in range(contract_retries + 1):
            prompt = _staged_planner_prompt(
                strategy_memo,
                parent_code=parent_code,
                budget=budget,
                limits=limits,
                max_phases=max_phases,
                stage=stage,
                previous_roadmap=roadmap if contract_attempt else None,
                contract_violations=(validation.get("violations") or [])
                if contract_attempt
                else (),
            )
            if callable(query_fn):
                response = query_fn(
                    prompt=copy.deepcopy(prompt),
                    strategy_memo=copy.deepcopy(dict(strategy_memo)),
                    json_schema=copy.deepcopy(ATOMIC_STAGED_ACTUATION_PLAN_SCHEMA),
                    contract_attempt=contract_attempt,
                )
            else:
                response = generate(
                    prompt=prompt,
                    cfg=agent.cfg,
                    temperature=0.0,
                    max_tokens=8000,
                    json_schema=ATOMIC_STAGED_ACTUATION_PLAN_SCHEMA,
                    max_retries=2,
                )
            try:
                roadmap = _parse_json_object(response)
                validation = validate_staged_atomic_plan(
                    roadmap,
                    strategy_memo=strategy_memo,
                    max_modules=limits["max_modules"],
                    max_changes=limits["max_changes"],
                    max_patches=limits["max_patches"],
                    max_phases=max_phases,
                    max_symbols=limits["max_symbols"],
                    parent_code=parent_code,
                    stage=stage,
                    debug_targeted_repair_only=bool(
                        getattr(
                            ext_cfg,
                            "memory_strategy_atomic_debug_targeted_repair_only",
                            True,
                        )
                    ),
                    require_complete_roadmap=require_complete,
                )
            except Exception as exc:
                roadmap = {}
                validation = {
                    "schema": "mlevolve_staged_atomic_plan_validation_v1",
                    "valid": False,
                    "violations": [
                        f"response parse failed: {type(exc).__name__}: {exc}"
                    ],
                }
            contract_attempts.append(
                {
                    "attempt": contract_attempt + 1,
                    "response_sha256": hashlib.sha256(
                        _canonical_json(response).encode("utf-8")
                    ).hexdigest(),
                    "valid": bool(validation.get("valid")),
                    "violations": list(validation.get("violations") or []),
                }
            )
            if validation.get("valid"):
                break
        return {
            "schema": "mlevolve_staged_atomic_planner_trace_v1",
            "status": "accepted" if validation.get("valid") else "rejected",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "strategy_memo_sha256": payload_sha256(strategy_memo),
            "parent_code_sha256": hashlib.sha256(
                parent_code.encode("utf-8")
            ).hexdigest(),
            "plan": roadmap,
            "plan_sha256": payload_sha256(roadmap),
            "validation": validation,
            "limits": limits,
            "max_phases": max_phases,
            "contract_attempts": contract_attempts,
        }
    except Exception as exc:
        logger.exception("Staged Atomic Planner failed")
        return {
            "schema": "mlevolve_staged_atomic_planner_trace_v1",
            "status": "failed",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "strategy_memo_sha256": payload_sha256(strategy_memo),
            "parent_code_sha256": hashlib.sha256(
                parent_code.encode("utf-8")
            ).hexdigest(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "limits": limits,
            "max_phases": max_phases,
        }


def _top_level_units(code: str) -> tuple[dict[str, str], set[str]]:
    tree = ast.parse(code)
    lines = code.splitlines(keepends=True)
    units: dict[str, list[str]] = {}
    imports: set[str] = set()

    def segment(node: ast.AST) -> str:
        start = max(0, int(getattr(node, "lineno", 1)) - 1)
        end = int(getattr(node, "end_lineno", start + 1))
        return "".join(lines[start:end])

    def assigned_names(target: ast.AST) -> set[str]:
        if isinstance(target, ast.Name):
            return {target.id}
        if isinstance(target, (ast.Tuple, ast.List)):
            return {name for child in target.elts for name in assigned_names(child)}
        return set()

    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
            units.setdefault("__imports__", []).append(segment(node))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.update(f"{node.module}.{alias.name}" for alias in node.names)
            units.setdefault("__imports__", []).append(segment(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            units.setdefault(node.name, []).append(segment(node))
        elif isinstance(node, ast.Assign):
            names = {name for target in node.targets for name in assigned_names(target)}
            for name in sorted(names):
                units.setdefault(name, []).append(segment(node))
            if not names:
                units.setdefault("__module__", []).append(segment(node))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            names = assigned_names(node.target)
            for name in sorted(names):
                units.setdefault(name, []).append(segment(node))
            if not names:
                units.setdefault("__module__", []).append(segment(node))
        else:
            units.setdefault("__module__", []).append(segment(node))
    return (
        {
            name: hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
            for name, parts in units.items()
        },
        imports,
    )


def _allowed_target_symbols(plan: Mapping[str, Any]) -> set[str]:
    symbols: set[str] = set()
    for change in plan.get("allowed_changes") or []:
        if isinstance(change, Mapping):
            symbols.update(str(value) for value in (change.get("target_symbols") or []))
    return symbols


def build_atomic_coder_allowlist(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Compile Planner prose into the exact packet enforced by the host verifier."""

    operations: dict[str, str] = {}
    descriptions: dict[str, str] = {}
    for change in plan.get("allowed_changes") or []:
        if not isinstance(change, Mapping):
            continue
        operation = str(change.get("operation") or "")
        description = str(change.get("description") or "")
        for symbol in change.get("target_symbols") or []:
            operations[str(symbol)] = operation
            descriptions[str(symbol)] = description
    return {
        "schema": "mlevolve_atomic_coder_allowlist_v1",
        "phase_id": str(plan.get("phase_id") or "single"),
        "objective": str(plan.get("objective") or ""),
        "required_symbol_operations": operations,
        "required_symbol_descriptions": descriptions,
        "allowed_new_imports": sorted(
            str(value) for value in (plan.get("allowed_new_imports") or [])
        ),
        "forbidden_symbols": sorted(
            str(value) for value in (plan.get("forbidden_symbols") or [])
        ),
        "forbidden_code_patterns": [
            str(value) for value in (plan.get("forbidden_code_patterns") or [])
        ],
        "max_patches": int(plan.get("max_patches") or 0),
        "authority_rule": (
            "Every required symbol operation must be implemented exactly once in this phase; "
            "all undeclared top-level changes and undeclared imports are forbidden."
        ),
    }


def validate_decomposed_replan(
    plan: Mapping[str, Any],
    *,
    previous_plan: Mapping[str, Any],
) -> list[str]:
    """Require a Coder-feedback replan to make a real, bounded decomposition."""

    violations: list[str] = []
    if str(plan.get("hypothesis_id") or "") != str(
        previous_plan.get("hypothesis_id") or ""
    ):
        violations.append("decomposed replan must keep the same hypothesis_id")
    if {str(value) for value in (plan.get("source_memory_ids") or [])} != {
        str(value) for value in (previous_plan.get("source_memory_ids") or [])
    }:
        violations.append("decomposed replan must keep the same source_memory_ids")

    current_modules = {str(value) for value in (plan.get("allowed_modules") or [])}
    previous_modules = {
        str(value) for value in (previous_plan.get("allowed_modules") or [])
    }
    current_symbols = _allowed_target_symbols(plan)
    previous_symbols = _allowed_target_symbols(previous_plan)
    current_imports = {str(value) for value in (plan.get("allowed_new_imports") or [])}
    previous_imports = {
        str(value) for value in (previous_plan.get("allowed_new_imports") or [])
    }
    current_changes = len(list(plan.get("allowed_changes") or []))
    previous_changes = len(list(previous_plan.get("allowed_changes") or []))
    try:
        current_patches = int(plan.get("max_patches") or 0)
    except (TypeError, ValueError):
        current_patches = 0
    try:
        previous_patches = int(previous_plan.get("max_patches") or 0)
    except (TypeError, ValueError):
        previous_patches = 0

    boundaries = (
        ("allowed_modules", current_modules, previous_modules),
        ("target_symbols", current_symbols, previous_symbols),
        ("allowed_new_imports", current_imports, previous_imports),
    )
    for label, current, previous in boundaries:
        if not current.issubset(previous):
            violations.append(
                f"decomposed replan {label} must be a subset of the rejected plan"
            )
    if current_changes > previous_changes:
        violations.append(
            "decomposed replan allowed_changes may not exceed the rejected plan"
        )
    if current_patches > previous_patches:
        violations.append(
            "decomposed replan max_patches may not exceed the rejected plan"
        )
    strictly_smaller = bool(
        len(current_modules) < len(previous_modules)
        or current_changes < previous_changes
        or len(current_symbols) < len(previous_symbols)
        or len(current_imports) < len(previous_imports)
        or current_patches < previous_patches
    )
    if not strictly_smaller:
        violations.append(
            "decomposed replan must strictly reduce at least one verified boundary"
        )
    return violations


def _coder_replan_mode(
    plan: Mapping[str, Any],
    coder_verdict: Mapping[str, Any],
) -> str:
    """Reconcile scope only when a bounded diff missed declarations."""

    violations = [str(value) for value in (coder_verdict.get("violations") or [])]
    allowed_prefixes = (
        "changed symbols outside allowed set:",
        "imports changed without __imports__ permission",
        "unauthorized new imports:",
    )
    try:
        patch_count = int(coder_verdict.get("patch_count") or 0)
        max_patches = int(plan.get("max_patches") or 0)
    except (TypeError, ValueError):
        return "decompose"
    unauthorized = {
        str(value)
        for value in (coder_verdict.get("unauthorized_changed_symbols") or [])
    }
    if (
        violations
        and unauthorized
        and 1 <= patch_count <= max_patches
        and all(value.startswith(allowed_prefixes) for value in violations)
    ):
        return "scope_reconciliation"
    return "decompose"


def validate_scope_reconciliation(
    plan: Mapping[str, Any],
    *,
    previous_plan: Mapping[str, Any],
    coder_verdict: Mapping[str, Any],
) -> list[str]:
    """Permit only exact missing symbols observed by the deterministic verifier."""

    violations: list[str] = []
    if str(plan.get("hypothesis_id") or "") != str(
        previous_plan.get("hypothesis_id") or ""
    ):
        violations.append("scope reconciliation must keep the same hypothesis_id")
    if {str(value) for value in (plan.get("source_memory_ids") or [])} != {
        str(value) for value in (previous_plan.get("source_memory_ids") or [])
    }:
        violations.append("scope reconciliation must keep the same source_memory_ids")
    current_modules = {str(value) for value in (plan.get("allowed_modules") or [])}
    previous_modules = {
        str(value) for value in (previous_plan.get("allowed_modules") or [])
    }
    if not current_modules.issubset(previous_modules):
        violations.append("scope reconciliation may not add allowed_modules")

    current_symbols = _allowed_target_symbols(plan)
    previous_symbols = _allowed_target_symbols(previous_plan)
    observed_symbols = {
        str(value)
        for value in (coder_verdict.get("unauthorized_changed_symbols") or [])
    }
    added_symbols = current_symbols - previous_symbols
    if not added_symbols:
        violations.append(
            "scope reconciliation must authorize an observed missing symbol"
        )
    if not added_symbols.issubset(observed_symbols):
        violations.append(
            "scope reconciliation added symbols not observed by the Coder verifier"
        )
    if not observed_symbols.issubset(current_symbols):
        violations.append(
            "scope reconciliation must account for every observed unauthorized symbol"
        )

    previous_imports = {
        str(value) for value in (previous_plan.get("allowed_new_imports") or [])
    }
    current_imports = {str(value) for value in (plan.get("allowed_new_imports") or [])}
    observed_imports = {
        str(value) for value in (coder_verdict.get("new_imports") or [])
    }
    if not (current_imports - previous_imports).issubset(observed_imports):
        violations.append(
            "scope reconciliation added imports not observed by the Coder verifier"
        )
    try:
        if int(plan.get("max_patches") or 0) > int(
            previous_plan.get("max_patches") or 0
        ):
            violations.append("scope reconciliation may not increase max_patches")
    except (TypeError, ValueError):
        violations.append("scope reconciliation has invalid max_patches")
    return violations


def verify_atomic_code_change(
    *,
    original_code: str,
    candidate_code: str,
    atomic_plan: Mapping[str, Any],
    patch_count: int,
    require_all_planned_changes: bool = False,
) -> dict[str, Any]:
    violations: list[str] = []
    try:
        before_units, before_imports = _top_level_units(original_code)
        after_units, after_imports = _top_level_units(candidate_code)
        syntax_valid = True
        syntax_error = ""
    except SyntaxError as exc:
        before_units, before_imports = _top_level_units(original_code)
        after_units, after_imports = {}, set()
        syntax_valid = False
        syntax_error = str(exc)
        violations.append(f"candidate syntax error: {exc}")

    changed_symbols = sorted(
        name
        for name in set(before_units) | set(after_units)
        if before_units.get(name) != after_units.get(name)
    )
    allowed_symbols = _allowed_target_symbols(atomic_plan)
    allowlist = build_atomic_coder_allowlist(atomic_plan)
    required_operations = dict(allowlist["required_symbol_operations"])
    new_imports = sorted(after_imports - before_imports)
    removed_imports = sorted(before_imports - after_imports)
    allowed_new_imports = {
        str(value) for value in (atomic_plan.get("allowed_new_imports") or [])
    }

    def import_is_allowed(import_name: str) -> bool:
        return any(
            import_name == allowed
            or allowed.startswith(import_name + ".")
            or import_name.startswith(allowed + ".")
            for allowed in allowed_new_imports
        )

    symbol_changes_to_check = set(changed_symbols)
    if "__imports__" in symbol_changes_to_check:
        if (
            new_imports
            and all(import_is_allowed(value) for value in new_imports)
            and not removed_imports
        ):
            symbol_changes_to_check.remove("__imports__")
        elif "__imports__" not in allowed_symbols:
            violations.append(
                "imports changed without __imports__ permission or allowed_new_imports"
            )
    unauthorized = sorted(symbol_changes_to_check - allowed_symbols)
    if unauthorized:
        violations.append(f"changed symbols outside allowed set: {unauthorized}")
    forbidden_symbols = {
        str(value) for value in (atomic_plan.get("forbidden_symbols") or [])
    }
    forbidden_touched = sorted(set(changed_symbols) & forbidden_symbols)
    if forbidden_touched:
        violations.append(f"forbidden symbols changed: {forbidden_touched}")
    unauthorized_imports = sorted(
        value for value in new_imports if not import_is_allowed(value)
    )
    if unauthorized_imports:
        violations.append(f"unauthorized new imports: {unauthorized_imports}")

    missing_required_symbols: list[str] = []
    operation_violations: list[str] = []
    if require_all_planned_changes and syntax_valid:
        missing_required_symbols = sorted(allowed_symbols - set(changed_symbols))
        if missing_required_symbols:
            violations.append(
                f"planned target symbols were not changed: {missing_required_symbols}"
            )
        for symbol, operation in sorted(required_operations.items()):
            existed_before = symbol in before_units
            exists_after = symbol in after_units
            if operation == "add" and (existed_before or not exists_after):
                operation_violations.append(
                    f"{symbol}: add requires absent-before and present-after"
                )
            elif operation == "modify" and (
                not existed_before or not exists_after or symbol not in changed_symbols
            ):
                operation_violations.append(
                    f"{symbol}: modify requires present-before, present-after, and changed"
                )
            elif operation == "delete" and (not existed_before or exists_after):
                operation_violations.append(
                    f"{symbol}: delete requires present-before and absent-after"
                )
        if operation_violations:
            violations.append(
                f"planned symbol operations were not followed: {operation_violations}"
            )

    newly_introduced_patterns: list[str] = []
    for pattern in atomic_plan.get("forbidden_code_patterns") or []:
        literal = str(pattern)
        if literal and candidate_code.count(literal) > original_code.count(literal):
            newly_introduced_patterns.append(literal)
    if newly_introduced_patterns:
        violations.append(
            f"forbidden code patterns introduced: {newly_introduced_patterns}"
        )
    try:
        max_patches = int(atomic_plan.get("max_patches", 0))
    except (TypeError, ValueError):
        max_patches = 0
    if patch_count < 1 or patch_count > max_patches:
        violations.append(
            f"patch count {patch_count} is outside atomic limit 1..{max_patches}"
        )
    if original_code == candidate_code:
        violations.append("candidate code is unchanged")
    return {
        "schema": "mlevolve_plan_diff_verdict_v1",
        "valid": not violations,
        "violations": violations,
        "syntax_valid": syntax_valid,
        "syntax_error": syntax_error,
        "patch_count": int(patch_count),
        "max_patches": max_patches,
        "allowed_symbols": sorted(allowed_symbols),
        "changed_symbols": changed_symbols,
        "unauthorized_changed_symbols": unauthorized,
        "missing_required_symbols": missing_required_symbols,
        "operation_violations": operation_violations,
        "require_all_planned_changes": bool(require_all_planned_changes),
        "coder_allowlist": allowlist,
        "new_imports": new_imports,
        "removed_imports": removed_imports,
        "allowed_new_imports": sorted(allowed_new_imports),
        "forbidden_symbols_touched": forbidden_touched,
        "forbidden_code_patterns_introduced": newly_introduced_patterns,
        "original_code_sha256": hashlib.sha256(
            original_code.encode("utf-8")
        ).hexdigest(),
        "candidate_code_sha256": hashlib.sha256(
            candidate_code.encode("utf-8")
        ).hexdigest(),
    }


def apply_atomic_diff_response(
    *,
    response: str,
    original_code: str,
    atomic_plan: Mapping[str, Any],
    require_all_planned_changes: bool = False,
) -> tuple[str | None, dict[str, Any]]:
    response = str(response or "")
    patcher = SearchReplacePatcher()
    blocks = list(patcher.PATCH_PATTERN.finditer(response))
    search_markers = response.count("<<<<<<< SEARCH")
    replace_markers = response.count(">>>>>>> REPLACE")
    if not blocks or search_markers != replace_markers or len(blocks) != search_markers:
        verdict = {
            "schema": "mlevolve_plan_diff_verdict_v1",
            "valid": False,
            "violations": [
                "diff contains missing, incomplete, or unparseable SEARCH/REPLACE blocks"
            ],
            "patch_count": len(blocks),
            "search_markers": search_markers,
            "replace_markers": replace_markers,
        }
        return None, verdict
    try:
        candidate_code, count = patcher.apply_patch(
            response, original_code, strict=True
        )
    except Exception as exc:
        return None, {
            "schema": "mlevolve_plan_diff_verdict_v1",
            "valid": False,
            "violations": [f"diff application failed: {exc}"],
            "patch_count": 0,
            "error_type": type(exc).__name__,
        }
    verdict = verify_atomic_code_change(
        original_code=original_code,
        candidate_code=candidate_code,
        atomic_plan=atomic_plan,
        patch_count=count,
        require_all_planned_changes=require_all_planned_changes,
    )
    return (candidate_code if verdict["valid"] else None), verdict


def _coder_prompt(
    *,
    atomic_plan: Mapping[str, Any],
    parent_code: str,
    task_description: str,
    execution_output: str,
    previous_response: str = "",
    previous_verdict: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    contract = _canonical_json(atomic_plan)
    allowlist = _canonical_json(build_atomic_coder_allowlist(atomic_plan))
    system = (
        "You are the Atomic Coder. Implement exactly the supplied Atomic Actuation Contract. "
        "Do not improve, reinterpret, compress, or replace the hypothesis. Do not touch a "
        "top-level symbol absent from allowed_changes.target_symbols. Do not add imports absent "
        "from allowed_new_imports. Every declared target is REQUIRED: perform its exact "
        "add/modify/delete operation and do not silently omit part of the Planner's phase. "
        "If the contract is impossible, return no patch rather than a different experiment. "
        "Output complete SEARCH/REPLACE blocks only."
    )
    instructions = build_base_diff_instructions(
        "The machine-verifiable Atomic Actuation Contract is the complete authority boundary."
    )
    user = (
        f"# Task\n{task_description}\n\n"
        f"# Atomic Actuation Contract\n{contract}\n\n"
        f"# Host-enforced exact allowlist\n{allowlist}\n\n"
        f"# Parent execution output\n{execution_output}\n\n"
        f"{instructions}\n\n{build_diff_format_suffix()}\n\n"
        f"Response format: {DIFF_SYS_FORMAT}"
    )
    if previous_verdict is not None:
        user += (
            "\n\n# Contract-preserving repair\n"
            "The prior diff was rejected. Repair only its mechanical contract violations; "
            "the Atomic Actuation Contract and hypothesis are unchanged. Do not reduce or "
            "replace the experiment. Remove every unauthorized_changed_symbol, implement every "
            "missing_required_symbol, and correct every operation_violation named below. Never "
            "ask to widen the allowlist. Every SEARCH block must match the current Parent code "
            "shown in this prompt, never the previously proposed candidate or an imagined "
            "intermediate state.\n"
            f"Prior verdict: {_canonical_json(previous_verdict)}\n"
            f"Prior response:\n{previous_response}"
        )
    assistant = (
        "I will implement only the allowed symbols in the contract. The current code is:\n"
        f"```python\n{parent_code}\n```\n"
        "Now I will output only complete SEARCH/REPLACE blocks."
    )
    return {"system": system, "user": user, "assistant": assistant}


def run_atomic_coder(
    agent: Any,
    *,
    planner_trace: Mapping[str, Any],
    parent_code: str,
    task_description: str,
    execution_output: str = "",
) -> dict[str, Any]:
    if str(planner_trace.get("status")) != "accepted":
        return {
            "schema": "mlevolve_atomic_coder_trace_v1",
            "status": "not_run",
            "reason": "atomic planner was not accepted",
        }
    plan = dict(planner_trace.get("plan") or {})
    started = time.monotonic()
    query_fn = getattr(agent, "_atomic_coder_query_fn", None)
    try:
        ext_cfg = getattr(agent.cfg, "external_skill_memory", None)
        contract_retries = int(
            getattr(
                ext_cfg,
                "memory_strategy_atomic_coder_contract_retries",
                1,
            )
            or 0
        )
        strict_coder = bool(
            getattr(
                ext_cfg,
                "memory_strategy_atomic_strict_coder_enabled",
                False,
            )
        )
        candidate_code: str | None = None
        response: Any = ""
        verdict: dict[str, Any] = {
            "schema": "mlevolve_plan_diff_verdict_v1",
            "valid": False,
            "violations": ["coder did not produce a diff"],
        }
        contract_attempts: list[dict[str, Any]] = []
        for contract_attempt in range(contract_retries + 1):
            prompt = _coder_prompt(
                atomic_plan=plan,
                parent_code=parent_code,
                task_description=task_description,
                execution_output=execution_output,
                previous_response=str(response or "") if contract_attempt else "",
                previous_verdict=verdict if contract_attempt else None,
            )
            if callable(query_fn):
                response = query_fn(
                    prompt=copy.deepcopy(prompt),
                    atomic_plan=copy.deepcopy(plan),
                    parent_code=parent_code,
                    contract_attempt=contract_attempt,
                )
            else:
                response = generate(
                    prompt=prompt,
                    cfg=agent.cfg,
                    temperature=getattr(agent.acfg.code, "temp", 0.0),
                    max_tokens=12000,
                    max_retries=2,
                )
            candidate_code, verdict = apply_atomic_diff_response(
                response=str(response or ""),
                original_code=parent_code,
                atomic_plan=plan,
                require_all_planned_changes=strict_coder,
            )
            contract_attempts.append(
                {
                    "attempt": contract_attempt + 1,
                    "response_sha256": hashlib.sha256(
                        str(response or "").encode("utf-8")
                    ).hexdigest(),
                    "valid": bool(verdict.get("valid")),
                    "violations": list(verdict.get("violations") or []),
                }
            )
            if candidate_code is not None:
                break
        return {
            "schema": "mlevolve_atomic_coder_trace_v1",
            "status": "accepted" if candidate_code is not None else "rejected",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "atomic_plan_sha256": payload_sha256(plan),
            "response_sha256": hashlib.sha256(
                str(response or "").encode("utf-8")
            ).hexdigest(),
            "candidate_code": candidate_code or "",
            "candidate_code_sha256": (
                hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
                if candidate_code is not None
                else ""
            ),
            "plan_diff_verdict": verdict,
            "strict_coder_enabled": strict_coder,
            "coder_allowlist": build_atomic_coder_allowlist(plan),
            "contract_attempts": contract_attempts,
        }
    except Exception as exc:
        logger.exception("Atomic Coder failed")
        return {
            "schema": "mlevolve_atomic_coder_trace_v1",
            "status": "failed",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "atomic_plan_sha256": payload_sha256(plan),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _run_staged_atomic_actuation_pipeline(
    agent: Any,
    *,
    strategy_memo: Mapping[str, Any],
    parent_code: str,
    task_description: str,
    execution_output: str,
    budget: Mapping[str, Any] | None,
    stage: str,
) -> dict[str, Any]:
    """Apply every verified phase cumulatively and expose only a complete roadmap."""

    planner_trace = run_atomic_staged_actuation_planner(
        agent,
        strategy_memo=strategy_memo,
        parent_code=parent_code,
        budget=budget,
        stage=stage,
    )
    roadmap = dict(planner_trace.get("plan") or {})
    phases = list(roadmap.get("phases") or [])
    phase_traces: list[dict[str, Any]] = []
    cumulative_code = parent_code
    final_coder: dict[str, Any] = {
        "schema": "mlevolve_atomic_coder_trace_v1",
        "status": "not_run",
        "reason": "staged planner was not accepted",
        "candidate_code": "",
        "plan_diff_verdict": {"valid": False},
    }
    if planner_trace.get("status") == "accepted":
        for index, raw_phase in enumerate(phases):
            phase = copy.deepcopy(dict(raw_phase))
            before_sha = hashlib.sha256(cumulative_code.encode("utf-8")).hexdigest()
            phase_planner_trace = {
                "schema": "mlevolve_atomic_phase_planner_trace_v1",
                "status": "accepted",
                "roadmap_id": str(roadmap.get("roadmap_id") or ""),
                "roadmap_sha256": payload_sha256(roadmap),
                "phase_index": index + 1,
                "phase_count": len(phases),
                "parent_code_sha256": before_sha,
                "plan": phase,
                "plan_sha256": payload_sha256(phase),
            }
            final_coder = run_atomic_coder(
                agent,
                planner_trace=phase_planner_trace,
                parent_code=cumulative_code,
                task_description=task_description,
                execution_output=execution_output,
            )
            accepted = final_coder.get("status") == "accepted"
            if accepted:
                cumulative_code = str(final_coder.get("candidate_code") or "")
            after_sha = hashlib.sha256(cumulative_code.encode("utf-8")).hexdigest()
            phase_traces.append(
                {
                    "phase_index": index + 1,
                    "phase_id": str(phase.get("phase_id") or ""),
                    "status": "accepted" if accepted else "rejected",
                    "hypothesis_id": str(phase.get("hypothesis_id") or ""),
                    "source_memory_ids": list(phase.get("source_memory_ids") or []),
                    "input_code_sha256": before_sha,
                    "output_code_sha256": after_sha if accepted else "",
                    "planner": phase_planner_trace,
                    "coder": copy.deepcopy(final_coder),
                }
            )
            if not accepted:
                break
    full_roadmap_applied = bool(
        planner_trace.get("status") == "accepted"
        and roadmap.get("roadmap_complete") is True
        and len(phase_traces) == len(phases)
        and phases
        and all(item.get("status") == "accepted" for item in phase_traces)
    )
    if full_roadmap_applied:
        final_coder = copy.deepcopy(final_coder)
        final_coder["candidate_code"] = cumulative_code
        final_coder["candidate_code_sha256"] = hashlib.sha256(
            cumulative_code.encode("utf-8")
        ).hexdigest()
        final_coder["cumulative_phase_count"] = len(phases)
        final_coder["phase_plan_diff_verdicts"] = [
            copy.deepcopy((item.get("coder") or {}).get("plan_diff_verdict") or {})
            for item in phase_traces
        ]
    else:
        # Never leak a partially applied roadmap to Improve/Debug execution.
        final_coder = copy.deepcopy(final_coder)
        final_coder["candidate_code"] = ""
        final_coder["candidate_code_sha256"] = ""
    return {
        "schema": "mlevolve_atomic_actuation_pipeline_v2",
        "status": "accepted" if full_roadmap_applied else "rejected",
        "strategy_memo_sha256": payload_sha256(strategy_memo),
        "stage": str(stage),
        "planner": planner_trace,
        "coder": final_coder,
        "staged_actuation_enabled": True,
        "strict_coder_enabled": bool(
            getattr(
                getattr(agent.cfg, "external_skill_memory", None),
                "memory_strategy_atomic_strict_coder_enabled",
                False,
            )
        ),
        "decomposition_used": len(phases) > 1,
        "full_roadmap_applied": full_roadmap_applied,
        "completed_phase_count": sum(
            item.get("status") == "accepted" for item in phase_traces
        ),
        "phase_count": len(phases),
        "phase_traces": phase_traces,
        "actuation_attempts": phase_traces,
        "alternate_hypothesis_used": False,
        "scope_reconciliation_used": False,
    }


def run_atomic_actuation_pipeline(
    agent: Any,
    *,
    strategy_memo: Mapping[str, Any],
    parent_code: str,
    task_description: str,
    execution_output: str = "",
    budget: Mapping[str, Any] | None = None,
    stage: str = "",
) -> dict[str, Any]:
    ext_cfg = getattr(agent.cfg, "external_skill_memory", None)
    if bool(
        getattr(
            ext_cfg,
            "memory_strategy_atomic_staged_enabled",
            False,
        )
    ):
        return _run_staged_atomic_actuation_pipeline(
            agent,
            strategy_memo=strategy_memo,
            parent_code=parent_code,
            task_description=task_description,
            execution_output=execution_output,
            budget=budget,
            stage=stage,
        )
    replan_attempts = max(
        0,
        int(
            getattr(
                ext_cfg,
                "memory_strategy_atomic_coder_replan_attempts",
                1,
            )
            or 0
        ),
    )
    alternate_hypothesis_attempts = max(
        0,
        int(
            getattr(
                ext_cfg,
                "memory_strategy_atomic_alternate_hypothesis_attempts",
                0,
            )
            or 0
        ),
    )
    alternate_replan_attempts = max(
        0,
        int(
            getattr(
                ext_cfg,
                "memory_strategy_atomic_alternate_replan_attempts",
                0,
            )
            or 0
        ),
    )
    actuation_attempts: list[dict[str, Any]] = []
    coder_replan_feedback: dict[str, Any] | None = None
    planner_trace: dict[str, Any] = {}
    coder_trace: dict[str, Any] = {}
    for actuation_attempt in range(replan_attempts + 1):
        planner_trace = run_atomic_actuation_planner(
            agent,
            strategy_memo=strategy_memo,
            parent_code=parent_code,
            budget=budget,
            stage=stage,
            coder_replan_feedback=coder_replan_feedback,
        )
        coder_trace = run_atomic_coder(
            agent,
            planner_trace=planner_trace,
            parent_code=parent_code,
            task_description=task_description,
            execution_output=execution_output,
        )
        attempt_trace = {
            "attempt": actuation_attempt + 1,
            "kind": "initial" if actuation_attempt == 0 else "decomposed_replan",
            "replan_mode": str((coder_replan_feedback or {}).get("replan_mode") or ""),
            "planner": planner_trace,
            "coder": coder_trace,
        }
        actuation_attempts.append(attempt_trace)
        if (
            planner_trace.get("status") == "accepted"
            and coder_trace.get("status") == "accepted"
        ):
            break
        if planner_trace.get("status") != "accepted":
            break
        if actuation_attempt >= replan_attempts:
            break
        coder_replan_feedback = {
            "previous_plan": copy.deepcopy(dict(planner_trace.get("plan") or {})),
            "coder_verdict": copy.deepcopy(
                dict(coder_trace.get("plan_diff_verdict") or {})
            ),
            "replan_mode": _coder_replan_mode(
                dict(planner_trace.get("plan") or {}),
                dict(coder_trace.get("plan_diff_verdict") or {}),
            ),
        }
    rejected_hypothesis_ids = {
        str((attempt.get("planner") or {}).get("plan", {}).get("hypothesis_id") or "")
        for attempt in actuation_attempts
        if str(
            (attempt.get("planner") or {}).get("plan", {}).get("hypothesis_id") or ""
        )
    }
    alternate_hypothesis_used = False
    if coder_trace.get("status") != "accepted":
        for alternate_attempt in range(alternate_hypothesis_attempts):
            coder_replan_feedback = {
                "previous_plan": copy.deepcopy(dict(planner_trace.get("plan") or {})),
                "coder_verdict": copy.deepcopy(
                    dict(coder_trace.get("plan_diff_verdict") or {})
                ),
                "allow_hypothesis_switch": True,
                "rejected_hypothesis_ids": sorted(rejected_hypothesis_ids),
            }
            planner_trace = run_atomic_actuation_planner(
                agent,
                strategy_memo=strategy_memo,
                parent_code=parent_code,
                budget=budget,
                stage=stage,
                coder_replan_feedback=coder_replan_feedback,
            )
            coder_trace = run_atomic_coder(
                agent,
                planner_trace=planner_trace,
                parent_code=parent_code,
                task_description=task_description,
                execution_output=execution_output,
            )
            actuation_attempts.append(
                {
                    "attempt": len(actuation_attempts) + 1,
                    "kind": "alternate_hypothesis",
                    "replan_mode": "alternate_hypothesis",
                    "planner": planner_trace,
                    "coder": coder_trace,
                }
            )
            selected_id = str(
                (planner_trace.get("plan") or {}).get("hypothesis_id") or ""
            )
            if selected_id:
                rejected_hypothesis_ids.add(selected_id)
            if (
                planner_trace.get("status") == "accepted"
                and coder_trace.get("status") == "accepted"
            ):
                alternate_hypothesis_used = True
                break
            for _ in range(alternate_replan_attempts):
                if planner_trace.get("status") != "accepted":
                    break
                replan_mode = _coder_replan_mode(
                    dict(planner_trace.get("plan") or {}),
                    dict(coder_trace.get("plan_diff_verdict") or {}),
                )
                coder_replan_feedback = {
                    "previous_plan": copy.deepcopy(
                        dict(planner_trace.get("plan") or {})
                    ),
                    "coder_verdict": copy.deepcopy(
                        dict(coder_trace.get("plan_diff_verdict") or {})
                    ),
                    "replan_mode": replan_mode,
                }
                planner_trace = run_atomic_actuation_planner(
                    agent,
                    strategy_memo=strategy_memo,
                    parent_code=parent_code,
                    budget=budget,
                    stage=stage,
                    coder_replan_feedback=coder_replan_feedback,
                )
                coder_trace = run_atomic_coder(
                    agent,
                    planner_trace=planner_trace,
                    parent_code=parent_code,
                    task_description=task_description,
                    execution_output=execution_output,
                )
                actuation_attempts.append(
                    {
                        "attempt": len(actuation_attempts) + 1,
                        "kind": "alternate_replan",
                        "replan_mode": replan_mode,
                        "planner": planner_trace,
                        "coder": coder_trace,
                    }
                )
                if (
                    planner_trace.get("status") == "accepted"
                    and coder_trace.get("status") == "accepted"
                ):
                    alternate_hypothesis_used = True
                    break
            if alternate_hypothesis_used:
                break
    return {
        "schema": "mlevolve_atomic_actuation_pipeline_v1",
        "status": (
            "accepted"
            if planner_trace.get("status") == "accepted"
            and coder_trace.get("status") == "accepted"
            else "rejected"
        ),
        "strategy_memo_sha256": payload_sha256(strategy_memo),
        "stage": str(stage),
        "planner": planner_trace,
        "coder": coder_trace,
        "decomposition_used": len(actuation_attempts) > 1,
        "alternate_hypothesis_used": alternate_hypothesis_used,
        "scope_reconciliation_used": any(
            attempt.get("replan_mode") == "scope_reconciliation"
            for attempt in actuation_attempts
        ),
        "actuation_attempts": actuation_attempts,
    }


__all__ = [
    "ATOMIC_ACTUATION_PLAN_SCHEMA",
    "ATOMIC_STAGED_ACTUATION_PLAN_SCHEMA",
    "apply_atomic_diff_response",
    "build_atomic_coder_allowlist",
    "run_atomic_actuation_pipeline",
    "run_atomic_actuation_planner",
    "run_atomic_staged_actuation_planner",
    "run_atomic_coder",
    "validate_atomic_plan",
    "validate_decomposed_replan",
    "validate_scope_reconciliation",
    "validate_staged_atomic_plan",
    "verify_atomic_code_change",
]
