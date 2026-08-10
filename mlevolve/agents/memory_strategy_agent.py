"""Read-only task-level synthesis over Dynamic Router memory.

The Retrieval Agent answers "which memories should be visible now?".  This
module answers a different question: "what important combination or coverage
gap is visible across the evidence?".  During the first rollout it is strictly
shadow-only: its memo is journaled, but it is never appended to the production
Planner/Coder prompt.
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


logger = logging.getLogger("MLEvolve")


STRATEGY_MEMO_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["propose", "abstain"],
        },
        "abstention_reason": {"type": "string"},
        "current_system_map": {"type": "object"},
        "evidence_portfolio": {"type": "object"},
        "coverage_gaps": {
            "type": "array",
            "items": {"type": "string"},
        },
        "candidate_compositions": {
            "type": "array",
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "source_memory_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "compatibility_checks": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "known_conflicts": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "estimated_compute_seconds": {"type": "integer"},
                    "minimal_change_set": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "forbidden_changes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "expected_mechanism": {"type": "string"},
                    "falsification_condition": {"type": "string"},
                    "novelty_kind": {
                        "type": "string",
                        "enum": [
                            "new_composition",
                            "missing_ablation",
                            "targeted_repair",
                            "single_memory_actuation",
                        ],
                    },
                },
                "required": [
                    "hypothesis_id",
                    "hypothesis",
                    "source_memory_ids",
                    "compatibility_checks",
                    "known_conflicts",
                    "estimated_compute_seconds",
                    "minimal_change_set",
                    "forbidden_changes",
                    "expected_mechanism",
                    "falsification_condition",
                    "novelty_kind",
                ],
                "additionalProperties": True,
            },
        },
        "recommended_hypothesis_id": {"type": "string"},
        "recommendation_reason": {"type": "string"},
        "declined_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["hypothesis", "reason"],
                "additionalProperties": True,
            },
        },
    },
    "required": [
        "decision",
        "abstention_reason",
        "current_system_map",
        "evidence_portfolio",
        "coverage_gaps",
        "candidate_compositions",
        "recommended_hypothesis_id",
        "recommendation_reason",
        "declined_hypotheses",
    ],
    "additionalProperties": True,
}


_STRATEGY_REQUIRED_KEYS = tuple(STRATEGY_MEMO_SCHEMA["required"])
_COMPOSITION_REQUIRED_KEYS = tuple(
    STRATEGY_MEMO_SCHEMA["properties"]["candidate_compositions"]["items"][
        "required"
    ]
)


_PORTFOLIO_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "representation": {
        "modernbert": (r"\bmodernbert\b",),
        "deberta": (r"\bdeberta\b",),
        "roberta": (r"\broberta\b",),
        "distilbert": (r"\bdistilbert\b",),
        "bert": (r"(?<![a-z])bert(?![a-z])",),
        "efficientnet": (r"\befficientnet\b",),
        "siglip": (r"\bsiglip\b",),
        "dinov2": (r"\bdinov2\b",),
        "vit": (r"\bvit\b", r"vision transformer"),
    },
    "adaptation_mode": {
        "frozen_embedding": (
            r"frozen.{0,40}(embedding|feature|backbone)",
            r"(embedding|backbone).{0,40}frozen",
            r"feature extractor",
        ),
        "fine_tuning": (r"fine[- ]?tun", r"unfrozen", r"classification head"),
    },
    "downstream_estimator": {
        "xgboost": (r"\bxgboost\b", r"\bxgb\b"),
        "lightgbm": (r"\blightgbm\b", r"\blgbm\b"),
        "catboost": (r"\bcatboost\b",),
        "linear_model": (r"logistic regression", r"linear classifier", r"\bsvm\b"),
        "neural_head": (r"classification head", r"\bmlp\b", r"neural network"),
    },
    "validation": {
        "five_fold": (r"(?:\b5\b|five)[- ]?fold", r"stratifiedkfold"),
        "oof": (r"\boof\b", r"out[- ]of[- ]fold"),
        "holdout": (r"holdout", r"train.{0,12}validation split", r"\b85/15\b", r"\b80/20\b"),
        "temporal": (r"temporal split", r"chronological split", r"time[- ]based split"),
    },
    "feature_family": {
        "embedding": (r"\bembedding", r"\bcls token\b", r"mean pooling"),
        "tfidf": (r"tf[- ]?idf", r"n[- ]?gram"),
        "stylometric": (r"stylometric", r"punctuation density", r"readability"),
        "tabular": (r"tabular", r"numeric descriptor"),
        "image": (r"\bimage", r"convolution", r"cnn"),
        "geospatial": (r"geospatial", r"haversine", r"airport", r"manhattan"),
    },
    "calibration_or_ensemble": {
        "temperature_scaling": (r"temperature scal",),
        "probability_blend": (r"\bblend", r"weighted average", r"ensemble average"),
    },
}


_CANDIDATE_ID_KEYS = (
    "candidate_id",
    "id",
    "node_id",
    "sop_id",
    "transition_id",
)
_CANDIDATE_SIGNAL_KEYS = {
    "candidate_id",
    "source",
    "source_stage",
    "source_task_id",
    "prompt_text",
    "plan",
    "title",
    "text",
    "metric",
    "metric_improvement",
    "outcome",
    "method_family",
    "abstraction_level",
}
_CARD_FIELDS = (
    "source",
    "type",
    "source_stage",
    "stage",
    "source_task_id",
    "task_id",
    "task",
    "title",
    "method_family",
    "abstraction_level",
    "outcome",
    "metric",
    "metric_improvement",
    "historical_metric",
    "rank_eligible",
    "eligibility_reason",
    "same_task_priority",
    "selection_reason",
    "plan",
    "text",
    "prompt_text",
    "analysis",
    "failure_signature",
    "repair_action",
    "resource_profile",
    "compute_profile",
    "risk_warnings",
    "supporting_transition_ids",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clean_text(value: Any, max_chars: int) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if max_chars > 0 and len(text) > max_chars:
            return text[: max_chars // 2] + "\n...[card evidence truncated]...\n" + text[-max_chars // 2 :]
        return text
    if isinstance(value, Mapping):
        return {
            str(key): _clean_text(item, max_chars)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clean_text(item, max_chars) for item in value]
    return value


def _candidate_id(row: Mapping[str, Any]) -> str:
    for key in _CANDIDATE_ID_KEYS:
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _walk_candidate_rows(value: Any, *, remaining: int) -> Iterable[dict[str, Any]]:
    if remaining <= 0:
        return
    if isinstance(value, Mapping):
        row = dict(value)
        if _candidate_id(row) and _CANDIDATE_SIGNAL_KEYS.intersection(row):
            yield row
            remaining -= 1
            if remaining <= 0:
                return
        for nested in row.values():
            for candidate in _walk_candidate_rows(nested, remaining=remaining):
                yield candidate
                remaining -= 1
                if remaining <= 0:
                    return
    elif isinstance(value, (list, tuple)):
        for nested in value:
            for candidate in _walk_candidate_rows(nested, remaining=remaining):
                yield candidate
                remaining -= 1
                if remaining <= 0:
                    return


def build_memory_cards(
    router_pack: Mapping[str, Any] | None,
    *,
    max_cards: int = 24,
    card_max_chars: int = 6000,
) -> list[dict[str, Any]]:
    """Build a bounded wide view, retaining selected and rejected evidence.

    Ordering is evidence-aware: prompt-visible rows first, then Agent-selected
    rows, pre-gate near misses, and finally the full deterministic pool.  IDs
    are deduplicated, so one memory cannot consume several card slots merely
    because it appears in several Router trace sections.
    """

    pack = dict(router_pack or {})
    sources = [
        ("prompt_visible", pack.get("final_prompt_candidates")),
        ("agent_selected", pack.get("selected_candidates") or pack.get("selected_items")),
        ("pre_gate", pack.get("pre_gate_raw_candidates")),
        ("candidate_pool", pack.get("candidate_pool")),
        ("raw_candidates", pack.get("raw_candidates")),
    ]
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for visibility, payload in sources:
        if payload is None:
            continue
        for row in _walk_candidate_rows(payload, remaining=max_cards * 8):
            candidate_id = _candidate_id(row)
            if not candidate_id or candidate_id in seen:
                continue
            compact = {
                key: _clean_text(copy.deepcopy(row[key]), card_max_chars)
                for key in _CARD_FIELDS
                if key in row and row[key] not in (None, "", [], {})
            }
            compact.update(
                {
                    "memory_id": candidate_id,
                    "router_visibility": visibility,
                    "router_rank": len(cards) + 1,
                }
            )
            cards.append(compact)
            seen.add(candidate_id)
            if len(cards) >= max_cards:
                return cards
    return cards


def build_component_portfolio(
    cards: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Index independently observed components without inferring causality.

    A whole pipeline's metric is not a component ablation.  The index therefore
    exposes only deterministic text matches and their supporting memory IDs; the
    Strategy Agent must decide whether a cross-memory composition is justified.
    """

    axes: dict[str, dict[str, list[str]]] = {}
    card_components: list[dict[str, Any]] = []
    for card in cards:
        memory_id = str(card.get("memory_id") or "")
        if not memory_id:
            continue
        evidence_text = _canonical_json(
            {
                key: card.get(key)
                for key in (
                    "title",
                    "method_family",
                    "plan",
                    "text",
                    "prompt_text",
                    "analysis",
                    "failure_signature",
                    "repair_action",
                )
                if card.get(key) not in (None, "", [], {})
            }
        )
        matched: dict[str, list[str]] = {}
        for axis, components in _PORTFOLIO_PATTERNS.items():
            for component, patterns in components.items():
                if not any(
                    re.search(pattern, evidence_text, re.IGNORECASE)
                    for pattern in patterns
                ):
                    continue
                axes.setdefault(axis, {}).setdefault(component, []).append(memory_id)
                matched.setdefault(axis, []).append(component)
        card_components.append(
            {
                "memory_id": memory_id,
                "metric": card.get("metric"),
                "outcome": card.get("outcome", ""),
                "components": matched,
            }
        )
    return {
        "schema": "mlevolve_memory_component_portfolio_v1",
        "interpretation_rule": (
            "A pipeline metric applies to the full memory card, not to each matched "
            "component. A weak pipeline may still contain a useful component if its "
            "failure arose at a different interface."
        ),
        "component_axes": axes,
        "card_components": card_components,
    }


def code_component_fingerprint(code: str) -> dict[str, Any]:
    """Return a compact structural view without pretending to understand code."""

    text = str(code or "")
    imports: set[str] = set()
    symbols: list[dict[str, Any]] = []
    try:
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(str(node.module or ""))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(
                    {
                        "name": node.name,
                        "kind": type(node).__name__,
                        "line": int(node.lineno),
                    }
                )
    except SyntaxError as exc:
        return {
            "code_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "parse_status": "syntax_error",
            "syntax_error": str(exc),
            "line_count": len(text.splitlines()),
        }

    lowered = text.lower()
    detectors = {
        "model_families": (
            "lightgbm", "xgboost", "catboost", "randomforest", "resnet",
            "efficientnet", "convnext", "vit", "dinov2", "siglip", "deberta",
            "modernbert", "distilbert", "roberta", "tfidf",
            "logisticregression", "svm", "knn",
        ),
        "validation": (
            "stratifiedkfold", "kfold", "groupkfold", "train_test_split",
            "cross_val", "oof",
        ),
        "composition": (
            "blend", "ensemble", "stacking", "weighted", "average", "voting",
        ),
        "calibration": (
            "temperature", "calibration", "isotonic", "platt", "softmax",
        ),
        "features": (
            "pca", "svd", "statistical", "metadata", "embedding", "augmentation",
        ),
    }
    detected = {
        family: [token for token in tokens if token in lowered]
        for family, tokens in detectors.items()
    }
    return {
        "code_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "parse_status": "ok",
        "line_count": len(text.splitlines()),
        "imports": sorted(value for value in imports if value)[:80],
        "top_level_symbols": symbols[:120],
        "detected_components": detected,
    }


def _metric_payload(node: Any) -> dict[str, Any]:
    metric = getattr(node, "metric", None)
    receipt = copy.deepcopy(getattr(node, "official_submission_receipt", {}) or {})
    return {
        "value": getattr(metric, "value", None),
        "maximize": getattr(metric, "maximize", None),
        "submission_variant": receipt.get("submission_variant")
        or receipt.get("variant")
        or "",
        "submission_sha256": receipt.get("submission_sha256")
        or receipt.get("prediction_sha256")
        or "",
        "submission_aligned_receipt_present": bool(receipt),
    }


def _attempt_history(agent: Any, *, limit: int) -> list[dict[str, Any]]:
    journal = getattr(agent, "journal", None)
    nodes = list(getattr(journal, "nodes", []) or [])[-max(0, int(limit)) :]
    attempts = []
    for node in nodes:
        if str(getattr(node, "stage", "")) == "root":
            continue
        attempts.append(
            {
                "node_id": str(getattr(node, "id", "")),
                "parent_node_id": str(getattr(getattr(node, "parent", None), "id", "")),
                "stage": str(getattr(node, "stage", "")),
                "plan": _clean_text(str(getattr(node, "plan", "") or ""), 4000),
                "metric": _metric_payload(node),
                "is_buggy": getattr(node, "is_buggy", None),
                "is_valid": getattr(node, "is_valid", None),
                "execution_seconds": getattr(node, "exec_time", None),
                "error_type": str(getattr(node, "exc_type", "") or ""),
                "error_summary": _clean_text(str(getattr(node, "term_out", "") or ""), 2000),
                "memory_ids": list(
                    (getattr(node, "memory_routing_trace", {}) or {}).get(
                        "final_prompt_candidate_ids", []
                    )
                ),
            }
        )
    return attempts


def build_strategy_context(
    agent: Any,
    parent_node: Any,
    *,
    stage: str,
    router_pack: Mapping[str, Any] | None,
    branch_best_metric: float | None = None,
) -> dict[str, Any]:
    ext_cfg = getattr(getattr(agent, "cfg", None), "external_skill_memory", None)
    max_cards = int(getattr(ext_cfg, "memory_strategy_max_cards", 24) or 24)
    card_max_chars = int(
        getattr(ext_cfg, "memory_strategy_card_max_chars", 6000) or 6000
    )
    history_limit = int(
        getattr(ext_cfg, "memory_strategy_history_limit", 16) or 16
    )
    cards = build_memory_cards(
        router_pack,
        max_cards=max_cards,
        card_max_chars=card_max_chars,
    )
    total_seconds = int(getattr(getattr(agent, "acfg", None), "time_limit", 0) or 0)
    started = float(getattr(agent, "start_time", time.time()) or time.time())
    elapsed = max(0.0, time.time() - started)
    remaining = max(0.0, total_seconds - elapsed) if total_seconds else None
    current_code = str(getattr(parent_node, "code", "") or "")
    context = {
        "schema": "mlevolve_memory_strategy_context_v1",
        "mode": "shadow_read_only",
        "stage": str(stage),
        "task": {
            "task_id": str(getattr(getattr(agent, "cfg", None), "exp_id", "") or ""),
            "description": str(getattr(agent, "task_desc", "") or ""),
            "data_preview": _clean_text(str(getattr(agent, "data_preview", "") or ""), 12000),
        },
        "current_solution": {
            "node_id": str(getattr(parent_node, "id", "")),
            "stage": str(getattr(parent_node, "stage", "")),
            "draft_origin": str(getattr(parent_node, "draft_role", "") or ""),
            "plan": _clean_text(str(getattr(parent_node, "plan", "") or ""), 8000),
            "code_summary": _clean_text(
                str(getattr(parent_node, "code_summary", "") or ""), 12000
            ),
            "component_fingerprint": code_component_fingerprint(current_code),
            "source_code": current_code,
            "execution_output": _clean_text(
                str(getattr(parent_node, "full_term_out", "") or ""), 16000
            ),
        },
        "metrics": {
            "parent_submission_metric": _metric_payload(parent_node),
            "branch_best_metric": branch_best_metric,
            "branch_best_is_parent": (
                branch_best_metric is not None
                and _metric_payload(parent_node).get("value") == branch_best_metric
            ),
        },
        "budget": {
            "total_search_seconds": total_seconds,
            "elapsed_search_seconds": round(elapsed, 3),
            "remaining_search_seconds": round(remaining, 3) if remaining is not None else None,
            "execution_timeout_seconds": int(
                getattr(getattr(getattr(agent, "cfg", None), "exec", None), "timeout", 0)
                or 0
            ),
            "gpu_count": int(
                getattr(getattr(getattr(agent, "acfg", None), "search", None), "num_gpus", 0)
                or 0
            ),
        },
        "attempt_history": _attempt_history(agent, limit=history_limit),
        "router": {
            "pack_schema": str((router_pack or {}).get("schema") or ""),
            "stage_route": copy.deepcopy((router_pack or {}).get("stage_route") or {}),
            "final_prompt_candidate_ids": list(
                (router_pack or {}).get("final_prompt_candidate_ids") or []
            ),
            "retrieval_agent": copy.deepcopy(
                (router_pack or {}).get("retrieval_agent") or {}
            ),
            "router_activation": copy.deepcopy(
                (router_pack or {}).get("router_activation") or {}
            ),
        },
        "memory_cards": cards,
        "component_portfolio": build_component_portfolio(cards),
    }
    max_input_chars = int(
        getattr(ext_cfg, "memory_strategy_max_input_chars", 0) or 0
    )
    if max_input_chars > 0:
        serialized = _canonical_json(context)
        if len(serialized) > max_input_chars:
            # Source code is the only unbounded field.  Keep all evidence IDs
            # and cards; reduce source text before touching structured memory.
            overflow = len(serialized) - max_input_chars
            source = context["current_solution"]["source_code"]
            keep = max(0, len(source) - overflow - 128)
            context["current_solution"]["source_code"] = (
                _clean_text(source, keep) if keep > 0 else ""
            )
            context["input_truncation"] = {
                "applied": True,
                "limit_chars": max_input_chars,
                "policy": "source_code_first_v1",
            }
    return context


def _strategy_prompt(
    context: Mapping[str, Any],
    *,
    previous_memo: Mapping[str, Any] | None = None,
    contract_violations: Iterable[str] = (),
) -> dict[str, str]:
    exact_contract = _canonical_json(STRATEGY_MEMO_SCHEMA)
    system = (
        "You are the Memory Strategy Agent, a read-only task-level research strategist. "
        "Do not write code and do not merely repeat the highest-ranked memory. Decompose each "
        "memory into representation, adaptation mode, downstream estimator, validation, feature, "
        "calibration, compute, and failure components. A full-pipeline score is not an ablation: "
        "do not discard one component solely because the pipeline containing it scored poorly. "
        "Instead ask whether the failure came from another component or an incompatible interface. "
        "Build a map of the current system, identify missing crossings in the component portfolio, "
        "and propose globally coherent hypotheses. If evidence is sufficient, decision must be "
        "'propose' and candidate_compositions must contain 3 to 5 distinct hypotheses: include "
        "at least one conservative transfer, at least one cross-lineage composition citing two or "
        "more memory IDs, and at least one discriminating ablation or frontier alternative. At "
        "least one composition must explicitly examine a component combination not yet jointly "
        "tested. You may combine ideas only when interfaces and compute budgets are compatible. "
        "If evidence is genuinely insufficient, decision may be 'abstain', candidate_compositions "
        "must be empty, and abstention_reason must be specific. Every proposal must "
        "cite only memory IDs present in memory_cards, distinguish the parent's real "
        "submission-aligned metric from branch_best_metric, estimate full training cost, name "
        "conflicts, preserve a minimal change set, and state a falsification condition. Reject "
        "duplicate, already-tried, over-budget, or unsupported combinations. Output one JSON "
        "object only using the exact field names in RESPONSE_SCHEMA; do not substitute fields such "
        "as proposed_experiment, experiment, changes, or memory_ids. This is SHADOW MODE: your "
        "memo has no authority to change the live plan.\n\nRESPONSE_SCHEMA:\n" + exact_contract
    )
    user = (
        "Analyze this frozen point-in-time context. A natural composition is useful only if "
        "the supplied evidence supports its parts or supplies a concrete failure-mode argument "
        "for transferring them, and the combined experiment fits the remaining budget. Use the "
        "component_portfolio to look for axes that have each been observed but never crossed. "
        "Prefer one executable next experiment, while preserving other defensible alternatives "
        "for evaluation.\n\nCONTEXT_JSON:\n"
        + _canonical_json(context)
    )
    violations = [str(value) for value in contract_violations]
    if previous_memo is not None or violations:
        user += (
            "\n\nCONTRACT_REPAIR_REQUIRED:\n"
            + _canonical_json(
                {
                    "violations": violations,
                    "previous_response": dict(previous_memo or {}),
                    "instruction": (
                        "Return a corrected complete object. Preserve useful analysis, but use "
                        "the exact schema and satisfy every listed violation."
                    ),
                }
            )
        )
    return {"system": system, "user": user, "assistant": "{"}


def _parse_json_object(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return copy.deepcopy(dict(response))
    text = str(response or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("strategy response is not a JSON object")
    return value


def validate_strategy_memo(
    memo: Mapping[str, Any],
    *,
    available_memory_ids: Iterable[str],
    min_candidate_compositions: int = 1,
    max_candidate_compositions: int = 5,
) -> dict[str, Any]:
    violations: list[str] = []
    missing_keys = [key for key in _STRATEGY_REQUIRED_KEYS if key not in memo]
    if missing_keys:
        violations.append(f"missing required top-level keys: {missing_keys}")
    available = {str(value) for value in available_memory_ids}
    compositions = list(memo.get("candidate_compositions") or [])
    decision = str(memo.get("decision") or "")
    if decision not in {"propose", "abstain"}:
        violations.append("decision must be propose or abstain")
    if not isinstance(memo.get("current_system_map"), Mapping):
        violations.append("current_system_map must be an object")
    if not isinstance(memo.get("evidence_portfolio"), Mapping):
        violations.append("evidence_portfolio must be an object")
    if decision == "propose" and not (
        int(min_candidate_compositions)
        <= len(compositions)
        <= int(max_candidate_compositions)
    ):
        violations.append(
            "propose requires "
            f"{int(min_candidate_compositions)}..{int(max_candidate_compositions)} "
            "candidate_compositions"
        )
    if decision == "abstain":
        if compositions:
            violations.append("abstain requires empty candidate_compositions")
        if not str(memo.get("abstention_reason") or "").strip():
            violations.append("abstain requires a specific abstention_reason")
    hypothesis_ids: list[str] = []
    for index, item in enumerate(compositions):
        if not isinstance(item, Mapping):
            violations.append(f"composition[{index}] is not an object")
            continue
        missing_composition_keys = [
            key for key in _COMPOSITION_REQUIRED_KEYS if key not in item
        ]
        if missing_composition_keys:
            violations.append(
                f"composition[{index}] missing required keys: {missing_composition_keys}"
            )
        hypothesis_id = str(item.get("hypothesis_id") or "")
        if not hypothesis_id:
            violations.append(f"composition[{index}] has no hypothesis_id")
        elif hypothesis_id in hypothesis_ids:
            violations.append(f"duplicate hypothesis_id: {hypothesis_id}")
        hypothesis_ids.append(hypothesis_id)
        unknown = sorted(
            set(str(value) for value in (item.get("source_memory_ids") or []))
            - available
        )
        if unknown:
            violations.append(
                f"{hypothesis_id or index} cites unavailable memory IDs: {unknown}"
            )
        cited = [str(value) for value in (item.get("source_memory_ids") or [])]
        if not cited:
            violations.append(f"{hypothesis_id or index} has no source_memory_ids")
        if (
            str(item.get("novelty_kind") or "") == "new_composition"
            and len(set(cited)) < 2
        ):
            violations.append(
                f"{hypothesis_id or index} new_composition requires at least two memory IDs"
            )
        if not list(item.get("compatibility_checks") or []):
            violations.append(f"{hypothesis_id or index} has no compatibility_checks")
        if not list(item.get("minimal_change_set") or []):
            violations.append(f"{hypothesis_id or index} has an empty minimal_change_set")
        if not str(item.get("expected_mechanism") or "").strip():
            violations.append(f"{hypothesis_id or index} has no expected_mechanism")
        if not str(item.get("falsification_condition") or "").strip():
            violations.append(f"{hypothesis_id or index} has no falsification_condition")
        if str(item.get("novelty_kind") or "") not in {
            "new_composition",
            "missing_ablation",
            "targeted_repair",
            "single_memory_actuation",
        }:
            violations.append(f"{hypothesis_id or index} has invalid novelty_kind")
        try:
            if int(item.get("estimated_compute_seconds", -1)) < 0:
                violations.append(f"{hypothesis_id or index} has invalid compute estimate")
        except (TypeError, ValueError):
            violations.append(f"{hypothesis_id or index} has invalid compute estimate")
    recommended = str(memo.get("recommended_hypothesis_id") or "")
    if recommended and recommended not in hypothesis_ids:
        violations.append("recommended_hypothesis_id is not in candidate_compositions")
    if compositions and not recommended:
        violations.append("nonempty compositions require one recommendation")
    if decision == "abstain" and recommended:
        violations.append("abstain requires empty recommended_hypothesis_id")
    return {
        "schema": "mlevolve_memory_strategy_validation_v1",
        "valid": not violations,
        "violations": violations,
        "available_memory_ids": sorted(available),
        "hypothesis_ids": hypothesis_ids,
        "decision": decision,
        "composition_count": len(compositions),
    }


def should_run_memory_strategy_shadow(
    agent: Any,
    *,
    stage: str,
    parent_node: Any,
    router_pack: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    ext_cfg = getattr(getattr(agent, "cfg", None), "external_skill_memory", None)
    if not bool(getattr(ext_cfg, "memory_strategy_shadow_enabled", False)):
        return False, "disabled"
    stages = {
        str(value)
        for value in (getattr(ext_cfg, "memory_strategy_shadow_stages", []) or [])
    }
    if str(stage) not in stages:
        return False, "stage_not_enabled"
    if str(stage) != "debug":
        return True, "stage_enabled"
    trigger = str(
        getattr(
            ext_cfg,
            "memory_strategy_debug_trigger",
            "causal_gap_or_repeated_failure",
        )
        or "causal_gap_or_repeated_failure"
    )
    if trigger == "always":
        return True, "debug_always"
    retrieval = (router_pack or {}).get("retrieval_agent") or {}
    no_causal_memory = not list(
        (router_pack or {}).get("final_prompt_candidate_ids") or []
    ) or bool(retrieval.get("agent_abstained"))
    threshold = int(
        getattr(ext_cfg, "memory_strategy_debug_failure_threshold", 2) or 2
    )
    repeated = int(getattr(parent_node, "debug_depth", 0) or 0) >= threshold
    if trigger == "causal_gap":
        return no_causal_memory, "debug_causal_gap" if no_causal_memory else "debug_has_causal_memory"
    if no_causal_memory or repeated:
        return True, "debug_causal_gap" if no_causal_memory else "debug_repeated_failure"
    return False, "debug_trigger_not_met"


def run_memory_strategy_shadow(
    agent: Any,
    parent_node: Any,
    *,
    stage: str,
    router_pack: Mapping[str, Any] | None,
    branch_best_metric: float | None = None,
    production_prompt_sha256: str = "",
) -> dict[str, Any]:
    """Run the Strategy Agent and return a Journal-ready read-only trace."""

    enabled, trigger_reason = should_run_memory_strategy_shadow(
        agent,
        stage=stage,
        parent_node=parent_node,
        router_pack=router_pack,
    )
    base_trace = {
        "schema": "mlevolve_memory_strategy_shadow_trace_v1",
        "mode": "shadow_read_only",
        "stage": str(stage),
        "parent_node_id": str(getattr(parent_node, "id", "")),
        "enabled": enabled,
        "trigger_reason": trigger_reason,
        "actuation_authority": "none",
        "production_prompt_modified": False,
        "production_prompt_sha256_before": str(production_prompt_sha256 or ""),
    }
    if not enabled:
        return {**base_trace, "status": "not_run"}

    context = build_strategy_context(
        agent,
        parent_node,
        stage=stage,
        router_pack=router_pack,
        branch_best_metric=branch_best_metric,
    )
    cards = list(context.get("memory_cards") or [])
    ext_cfg = getattr(agent.cfg, "external_skill_memory", None)
    started = time.monotonic()
    try:
        contract_retries = int(
            getattr(ext_cfg, "memory_strategy_contract_retries", 2) or 0
        )
        min_compositions = int(
            getattr(ext_cfg, "memory_strategy_min_candidate_compositions", 3)
            or 1
        )
        contract_attempts: list[dict[str, Any]] = []
        memo: dict[str, Any] = {}
        validation: dict[str, Any] = {
            "valid": False,
            "violations": ["strategy call did not produce a memo"],
        }
        for contract_attempt in range(contract_retries + 1):
            prompt = _strategy_prompt(
                context,
                previous_memo=memo if contract_attempt else None,
                contract_violations=(validation.get("violations") or [])
                if contract_attempt
                else (),
            )
            query_fn = getattr(agent, "_memory_strategy_query_fn", None)
            if callable(query_fn):
                response = query_fn(
                    prompt=copy.deepcopy(prompt),
                    context=copy.deepcopy(context),
                    json_schema=copy.deepcopy(STRATEGY_MEMO_SCHEMA),
                    contract_attempt=contract_attempt,
                )
            else:
                response = generate(
                    prompt=prompt,
                    cfg=agent.cfg,
                    temperature=float(
                        getattr(ext_cfg, "memory_strategy_temperature", 0.0) or 0.0
                    ),
                    max_tokens=int(
                        getattr(ext_cfg, "memory_strategy_max_output_tokens", 6000)
                        or 6000
                    ),
                    json_schema=STRATEGY_MEMO_SCHEMA,
                    max_retries=int(
                        getattr(ext_cfg, "memory_strategy_max_retries", 2) or 2
                    ),
                )
            try:
                memo = _parse_json_object(response)
                validation = validate_strategy_memo(
                    memo,
                    available_memory_ids=[card["memory_id"] for card in cards],
                    min_candidate_compositions=min_compositions,
                )
            except Exception as exc:
                memo = {}
                validation = {
                    "schema": "mlevolve_memory_strategy_validation_v1",
                    "valid": False,
                    "violations": [
                        f"response parse failed: {type(exc).__name__}: {exc}"
                    ],
                    "available_memory_ids": sorted(
                        card["memory_id"] for card in cards
                    ),
                    "hypothesis_ids": [],
                    "decision": "",
                    "composition_count": 0,
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
        status = "completed" if validation["valid"] else "completed_with_contract_violations"
        return {
            **base_trace,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "model": str(getattr(agent.acfg.code, "model", "") or ""),
            "context_sha256": payload_sha256(context),
            "context_char_count": len(_canonical_json(context)),
            "memory_card_ids": [card["memory_id"] for card in cards],
            "memory_card_count": len(cards),
            "router_final_prompt_candidate_ids": list(
                (router_pack or {}).get("final_prompt_candidate_ids") or []
            ),
            "memo": memo,
            "memo_sha256": payload_sha256(memo),
            "validation": validation,
            "contract_attempts": contract_attempts,
        }
    except Exception as exc:
        logger.exception("Memory Strategy shadow call failed")
        return {
            **base_trace,
            "status": "failed",
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "context_sha256": payload_sha256(context),
            "context_char_count": len(_canonical_json(context)),
            "memory_card_ids": [card["memory_id"] for card in cards],
            "memory_card_count": len(cards),
        }


__all__ = [
    "STRATEGY_MEMO_SCHEMA",
    "build_component_portfolio",
    "build_memory_cards",
    "build_strategy_context",
    "code_component_fingerprint",
    "payload_sha256",
    "run_memory_strategy_shadow",
    "should_run_memory_strategy_shadow",
    "validate_strategy_memo",
]
