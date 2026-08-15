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
import math
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
        "addressed_opportunities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "opportunity_id": {"type": "string"},
                    "disposition": {
                        "type": "string",
                        "enum": ["proposed", "declined"],
                    },
                    "hypothesis_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "opportunity_id",
                    "disposition",
                    "hypothesis_id",
                    "reason",
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
        "addressed_opportunities",
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
    "candidate_source",
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
    "candidate_source",
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
    "run_id",
    "branch_id",
    "draft_role",
    "code_summary",
    "official_submission_receipt",
    "submission_aligned",
    "metric_protocol",
    "validation_protocol",
    "validation_protocol_evidence",
    "evidence_tier",
    "metric_provenance",
    "metric_disposition",
    "method_fingerprint",
    "code_sha256",
    "original_node_id",
    "resolved_transition_id",
    "resolution_path",
    "evidence_class",
    "parent_node_id",
    "child_node_id",
    "before_code_sha256",
    "after_code_sha256",
    "canonical_diff",
    "before_code",
    "after_code",
    "source_journal",
    "source_journal_sha256",
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
        ("resolved_evidence", pack.get("resolved_evidence")),
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
        # Implementation summaries describe what actually ran.  Planning text
        # often mentions rejected or replaced alternatives, so mixing it into
        # the component detector creates false combinations (for example a
        # DeBERTa implementation whose plan merely discusses ModernBERT).
        if card.get("text") not in (None, "", [], {}):
            evidence_basis = "text"
            evidence_payload = {"text": card.get("text")}
        elif card.get("prompt_text") not in (None, "", [], {}):
            evidence_basis = "prompt_text"
            evidence_payload = {"prompt_text": card.get("prompt_text")}
        else:
            evidence_basis = "plan_fallback"
            evidence_payload = {
                key: card.get(key)
                for key in (
                    "title",
                    "method_family",
                    "plan",
                    "analysis",
                    "failure_signature",
                    "repair_action",
                )
                if card.get(key) not in (None, "", [], {})
            }
        evidence_text = _canonical_json(evidence_payload)
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
                "evidence_basis": evidence_basis,
                "components": matched,
            }
        )
    opportunities: list[dict[str, Any]] = []
    for axis, components in sorted(axes.items()):
        if len(components) < 3:
            continue
        ordered_components = sorted(components)
        support_sets = [set(components[name]) for name in ordered_components]
        # If one historical card already contains every alternative, this is
        # not a missing within-axis composition.
        jointly_observed = set.intersection(*support_sets) if support_sets else set()
        if jointly_observed:
            continue
        interface_sets: list[set[str]] = []
        alternative_evidence = []
        for component, supporting_ids in zip(ordered_components, support_sets):
            interfaces: set[str] = set()
            for row in card_components:
                if row["memory_id"] not in supporting_ids:
                    continue
                for other_axis, values in row["components"].items():
                    if other_axis == axis:
                        continue
                    interfaces.update(f"{other_axis}:{value}" for value in values)
            interface_sets.append(interfaces)
            alternative_evidence.append(
                {
                    "component": component,
                    "memory_ids": sorted(supporting_ids),
                    "observed_interfaces": sorted(interfaces),
                }
            )
        common_interfaces = (
            sorted(set.intersection(*interface_sets)) if interface_sets else []
        )
        if not common_interfaces:
            continue
        opportunity_id = "within_axis::" + axis + "::" + "+".join(
            ordered_components
        )
        opportunities.append(
            {
                "opportunity_id": opportunity_id,
                "axis": axis,
                "alternatives": ordered_components,
                "alternative_evidence": alternative_evidence,
                "common_interfaces": common_interfaces,
                "already_jointly_observed": False,
                "analysis_question": (
                    "Would one budget-compatible experiment combining these alternatives "
                    "at a shared interface reduce representation/model blind spots? Assess "
                    "component compatibility separately from each source pipeline's score."
                ),
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
        "within_axis_diversity_opportunities": opportunities,
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


def _metric_number(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _strategy_metric_maximize(agent: Any, parent_node: Any) -> bool:
    explicit = getattr(agent, "metric_maximize", None)
    if explicit is not None:
        return bool(explicit)
    metric = getattr(parent_node, "metric", None)
    return bool(getattr(metric, "maximize", False))


def _node_strategy_eligible(node: Any) -> bool:
    metric = getattr(node, "metric", None)
    leakage_audit = getattr(node, "leakage_audit", None) or {}
    strategy_alignment = getattr(node, "strategy_alignment", None) or {}
    return bool(
        _metric_number(getattr(metric, "value", None)) is not None
        and getattr(node, "is_buggy", None) is not True
        and getattr(node, "is_valid", None) is not False
        and leakage_audit.get("rank_eligible") is not False
        and strategy_alignment.get("rank_eligible") is not False
    )


def _node_metric_protocol(node: Any) -> str:
    for value in (
        getattr(node, "metric_protocol", None),
        getattr(node, "validation_protocol", None),
    ):
        if value:
            return str(value)
    receipt = getattr(node, "official_submission_receipt", None) or {}
    for key in ("metric_protocol", "validation_protocol"):
        if receipt.get(key):
            return str(receipt[key])
    observation = getattr(node, "protocol_observation", None) or {}
    for key in ("metric_protocol", "validation_protocol"):
        if observation.get(key):
            return str(observation[key])
    if receipt:
        return "submission_aligned_internal_unspecified"
    return ""


def _node_evidence_card(
    agent: Any,
    node: Any,
    *,
    visibility: str,
    selection_reason: str,
    branch_id: Any = None,
) -> dict[str, Any]:
    node_id = str(getattr(node, "id", "") or "")
    plan = str(getattr(node, "plan", "") or "")
    summary = str(getattr(node, "code_summary", "") or "")
    receipt = copy.deepcopy(
        getattr(node, "official_submission_receipt", {}) or {}
    )
    return {
        "memory_id": f"current::{node_id}",
        "source": "current_run",
        "type": "SearchNode",
        "source_stage": str(getattr(node, "stage", "") or ""),
        "source_task_id": str(getattr(getattr(agent, "cfg", None), "exp_id", "") or ""),
        "branch_id": (
            getattr(node, "branch_id", None)
            if getattr(node, "branch_id", None) is not None
            else branch_id
        ),
        "draft_role": str(getattr(node, "draft_role", "") or ""),
        "metric": _metric_number(getattr(getattr(node, "metric", None), "value", None)),
        "submission_metric": _metric_payload(node),
        "official_submission_receipt": receipt,
        "submission_aligned": bool(receipt),
        "metric_protocol": _node_metric_protocol(node),
        "outcome": (
            "execution_failed"
            if getattr(node, "is_buggy", None) is True
            or getattr(node, "is_valid", None) is False
            else "valid"
        ),
        "rank_eligible": _node_strategy_eligible(node),
        "plan": _clean_text(plan, 4000),
        "text": _clean_text(summary or plan, 6000),
        "failure_signature": _clean_text(
            str(getattr(node, "analysis", "") or getattr(node, "term_out", "") or ""),
            3000,
        ),
        "execution_seconds": getattr(node, "exec_time", None),
        "memory_ids": list(
            (getattr(node, "memory_routing_trace", {}) or {}).get(
                "final_prompt_candidate_ids", []
            )
        ),
        "router_visibility": visibility,
        "selection_reason": selection_reason,
    }


def _configured_draft_roles(agent: Any) -> list[str]:
    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    if not bool(getattr(policy, "enabled", False)):
        return []
    return [str(value) for value in (getattr(policy, "roles", []) or []) if value]


def _select_current_branch_frontier(
    agent: Any,
    parent_node: Any,
    *,
    limit: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    maximize = _strategy_metric_maximize(agent, parent_node)
    branch_map = dict(getattr(agent, "branch_successful_nodes", {}) or {})
    branch_best: list[tuple[Any, Any]] = []
    for branch_id, raw_nodes in sorted(
        branch_map.items(), key=lambda item: str(item[0])
    ):
        nodes = [node for node in (raw_nodes or []) if _node_strategy_eligible(node)]
        if not nodes:
            continue
        best = sorted(
            nodes,
            key=lambda node: _metric_number(
                getattr(getattr(node, "metric", None), "value", None)
            ),
            reverse=maximize,
        )[0]
        branch_best.append((branch_id, best))

    # Historical replay fixtures and very early live calls may not have the
    # branch registry populated yet.  The current parent remains useful, but it
    # is explicitly labelled as a fallback rather than pretending to represent
    # all branches.
    if not branch_best and _node_strategy_eligible(parent_node):
        branch_best = [(getattr(parent_node, "branch_id", None), parent_node)]

    configured_roles = _configured_draft_roles(agent)
    selected_nodes: list[tuple[Any, Any]] = []
    selected_ids: set[str] = set()
    for role in configured_roles:
        candidates = [
            (branch_id, node)
            for branch_id, node in branch_best
            if str(getattr(node, "draft_role", "") or "") == role
        ]
        if not candidates:
            continue
        best = sorted(
            candidates,
            key=lambda item: _metric_number(
                getattr(getattr(item[1], "metric", None), "value", None)
            ),
            reverse=maximize,
        )[0]
        selected_nodes.append(best)
        selected_ids.add(str(getattr(best[1], "id", "") or ""))
        if len(selected_nodes) >= limit:
            break

    remaining = [
        (branch_id, node)
        for branch_id, node in branch_best
        if str(getattr(node, "id", "") or "") not in selected_ids
    ]
    remaining.sort(
        key=lambda item: _metric_number(
            getattr(getattr(item[1], "metric", None), "value", None)
        ),
        reverse=maximize,
    )
    selected_nodes.extend(remaining[: max(0, limit - len(selected_nodes))])
    missing_roles = [
        role
        for role in configured_roles
        if not any(
            str(getattr(node, "draft_role", "") or "") == role
            for _, node in selected_nodes
        )
    ]
    cards = [
        _node_evidence_card(
            agent,
            node,
            visibility="current_branch_frontier",
            selection_reason=(
                "best_rank_eligible_node_for_draft_role"
                if str(getattr(node, "draft_role", "") or "")
                else "best_available_current_parent_fallback"
            ),
            branch_id=branch_id,
        )
        for branch_id, node in selected_nodes[:limit]
    ]
    return cards, missing_roles


def _select_causal_failure_cards(
    agent: Any,
    parent_node: Any,
    *,
    exclude_node_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    journal = getattr(agent, "journal", None)
    nodes = [parent_node] + list(reversed(list(getattr(journal, "nodes", []) or [])))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes:
        node_id = str(getattr(node, "id", "") or "")
        if not node_id or node_id in seen or node_id in exclude_node_ids:
            continue
        seen.add(node_id)
        failed = bool(
            getattr(node, "is_buggy", None) is True
            or getattr(node, "is_valid", None) is False
            or str(getattr(node, "exc_type", "") or "")
        )
        if not failed:
            continue
        selected.append(
            _node_evidence_card(
                agent,
                node,
                visibility="causal_failure_evidence",
                selection_reason="most_recent_causal_execution_failure",
            )
        )
        if len(selected) >= limit:
            break
    return selected


def _historical_card_signature(card: Mapping[str, Any]) -> str:
    method_fingerprint = str(card.get("method_fingerprint") or "").strip()
    if method_fingerprint:
        return f"fingerprint={method_fingerprint}"
    portfolio = build_component_portfolio([card])
    rows = list(portfolio.get("card_components") or [])
    components = (rows[0].get("components") or {}) if rows else {}
    normalized = [
        f"{axis}={'+'.join(sorted(str(value) for value in values))}"
        for axis, values in sorted(components.items())
        if values
    ]
    method_family = str(card.get("method_family") or "").strip().lower()
    if method_family:
        normalized.append(f"method={method_family}")
    if normalized:
        return "|".join(normalized)
    title = re.sub(r"[^a-z0-9]+", " ", str(card.get("title") or "").lower()).strip()
    if title:
        return "title=" + " ".join(title.split()[:12])
    return "id=" + str(card.get("memory_id") or "")


def _historical_metric_protocol(card: Mapping[str, Any]) -> str:
    for key in ("metric_protocol", "validation_protocol"):
        if card.get(key):
            return str(card[key])
    receipt = card.get("official_submission_receipt") or {}
    if isinstance(receipt, Mapping):
        for key in ("metric_protocol", "validation_protocol"):
            if receipt.get(key):
                return str(receipt[key])
    return ""


def _node_identity_tokens(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        tokens.add(text.removeprefix("current::"))
        if "::node::" in text:
            tokens.add(text.rsplit("::node::", 1)[-1])
    return tokens


def _select_historical_frontier(
    agent: Any,
    parent_node: Any,
    cards: Iterable[Mapping[str, Any]],
    *,
    limit: int,
    exclude_node_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    maximize = _strategy_metric_maximize(agent, parent_node)
    task_id = str(getattr(getattr(agent, "cfg", None), "exp_id", "") or "")
    current_protocol = _node_metric_protocol(parent_node)
    resolved_evidence: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    method_only: list[dict[str, Any]] = []
    rejected = {
        "task_mismatch": 0,
        "rank_ineligible": 0,
        "missing_metric": 0,
        "missing_metric_retained_as_method_evidence": 0,
        "current_node_duplicate": 0,
        "duplicate_lineage": 0,
    }
    excluded_tokens = _node_identity_tokens(*(exclude_node_ids or set()))
    for raw in cards:
        card = copy.deepcopy(dict(raw))
        card_tokens = _node_identity_tokens(
            card.get("memory_id"),
            card.get("node_id"),
            card.get("original_node_id"),
        )
        if card_tokens & excluded_tokens:
            rejected["current_node_duplicate"] += 1
            continue
        source_task = str(
            card.get("source_task_id") or card.get("task_id") or card.get("task") or ""
        )
        if source_task and task_id and source_task != task_id:
            rejected["task_mismatch"] += 1
            continue
        if card.get("rank_eligible") is False:
            rejected["rank_ineligible"] += 1
            continue
        if str(card.get("router_visibility") or "") == "resolved_evidence":
            # This is the deterministic post-Judge executable view.  It must
            # reach Strategy ahead of score-ranked historical summaries; the
            # latter can otherwise exhaust the bounded evidence frontier and
            # silently undo Judge -> Resolver -> Strategy.
            card["metric"] = None
            card["submission_aligned_receipt_present"] = False
            card["metric_protocol"] = _historical_metric_protocol(card)
            card["metric_comparable_to_current"] = False
            card["metric_claim_status"] = "resolved_transition_evidence"
            resolved_evidence.append(card)
            continue
        metric_value = card.get("metric")
        if metric_value is None:
            metric_value = card.get("historical_metric")
        metric = _metric_number(metric_value)
        if metric is None:
            # The Router frequently exposes a complete method/repair card but
            # intentionally strips a non-comparable historical score.  That
            # makes the card unsafe for numeric ranking, not useless for
            # Strategy synthesis.  Retain it after metric-bearing cards and
            # label it explicitly as method-only evidence.
            rejected["missing_metric_retained_as_method_evidence"] += 1
            card["metric"] = None
            card["submission_aligned_receipt_present"] = False
            card["metric_protocol"] = _historical_metric_protocol(card)
            card["metric_comparable_to_current"] = False
            card["metric_claim_status"] = "same_task_method_evidence_only"
            method_only.append(card)
            continue
        card["metric"] = metric
        receipt = card.get("official_submission_receipt") or {}
        card["submission_aligned_receipt_present"] = bool(
            receipt or card.get("submission_aligned") is True
        )
        protocol = _historical_metric_protocol(card)
        card["metric_protocol"] = protocol
        card["metric_comparable_to_current"] = bool(
            protocol and current_protocol and protocol == current_protocol
        )
        if card["metric_comparable_to_current"]:
            card["metric_claim_status"] = "same_task_same_protocol_comparable"
        elif protocol:
            card["metric_claim_status"] = "same_task_within_protocol_only"
        else:
            card["metric_claim_status"] = "unverified_method_evidence_only"
        eligible.append(card)

    # Never use raw scores to order different validation protocols.  Protocol
    # groups are ordered by evidence confidence; scores only rank cards inside
    # one same-task, same-protocol group.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for card in eligible:
        protocol = str(card.get("metric_protocol") or "")
        group_key = protocol or "__unverified_protocol__"
        grouped.setdefault(group_key, []).append(card)
    for group in grouped.values():
        group.sort(
            key=lambda card: (
                -float(card["metric"]) if maximize else float(card["metric"]),
                int(card.get("router_rank") or 10**9),
            )
        )

    def group_key(item: tuple[str, list[dict[str, Any]]]) -> tuple[Any, ...]:
        protocol, group = item
        same_as_current = bool(current_protocol and protocol == current_protocol)
        has_receipt = any(
            bool(card.get("submission_aligned_receipt_present")) for card in group
        )
        known_protocol = protocol != "__unverified_protocol__"
        return (
            not same_as_current,
            not has_receipt,
            not known_protocol,
            protocol,
        )

    ordered_eligible = [
        card
        for _, group in sorted(grouped.items(), key=group_key)
        for card in group
    ]
    resolved_evidence.sort(
        key=lambda card: (
            int(card.get("router_rank") or 10**9),
            str(card.get("memory_id") or ""),
        )
    )
    visibility_priority = {
        "resolved_evidence": 0,
        "prompt_visible": 1,
        "agent_selected": 2,
        "pre_gate": 3,
        "candidate_pool": 4,
        "raw_candidates": 5,
    }
    method_only.sort(
        key=lambda card: (
            visibility_priority.get(str(card.get("router_visibility") or ""), 9),
            int(card.get("router_rank") or 10**9),
            str(card.get("memory_id") or ""),
        )
    )
    ordered_eligible = resolved_evidence + ordered_eligible + method_only
    selected: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for card in ordered_eligible:
        signature = _historical_card_signature(card)
        if signature in signatures:
            rejected["duplicate_lineage"] += 1
            continue
        signatures.add(signature)
        is_resolved = str(card.get("router_visibility") or "") == "resolved_evidence"
        card["strategy_selection_bucket"] = "historical_diverse_frontier"
        if not is_resolved:
            card["router_visibility"] = "historical_diverse_frontier"
        card["selection_reason"] = (
            "judge_selected_post_resolution_executable_evidence"
            if is_resolved
            else
            "same_task_router_ranked_method_evidence_then_component_diversity"
            if card.get("metric") is None
            else "same_task_same_protocol_metric_then_component_diversity"
            if card.get("metric_comparable_to_current")
            else "same_task_within_protocol_metric_then_component_diversity"
            if card.get("metric_protocol")
            else "same_task_unverified_metric_as_method_evidence_then_component_diversity"
        )
        card["component_lineage_signature"] = signature
        selected.append(card)
        if len(selected) >= limit:
            break
    return selected, rejected


def build_strategy_evidence_view(
    agent: Any,
    parent_node: Any,
    *,
    stage: str,
    router_pack: Mapping[str, Any] | None,
    max_items: int = 8,
    current_frontier_slots: int = 3,
    causal_failure_slots: int = 1,
    candidate_pool_limit: int = 48,
    card_max_chars: int = 6000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one auditable, role-balanced Strategy evidence set.

    The eight-item cap applies to everything the Strategy Agent may cite:
    current branch frontiers, one causal failure, and historical memory.  Raw
    Router candidates remain in the Router trace but do not silently bypass
    this attention budget.
    """

    max_items = max(1, int(max_items))
    current_cards, missing_roles = _select_current_branch_frontier(
        agent,
        parent_node,
        limit=min(max_items, max(0, int(current_frontier_slots))),
    )
    selected_node_ids = {
        str(card.get("memory_id") or "").removeprefix("current::")
        for card in current_cards
    }
    failure_cards = _select_causal_failure_cards(
        agent,
        parent_node,
        exclude_node_ids=selected_node_ids,
        limit=min(
            max(0, max_items - len(current_cards)),
            max(0, int(causal_failure_slots)),
        ),
    )
    raw_historical = build_memory_cards(
        router_pack,
        max_cards=max(max_items, int(candidate_pool_limit)),
        card_max_chars=card_max_chars,
    )
    historical_limit = max(0, max_items - len(current_cards) - len(failure_cards))
    historical_cards, rejected = _select_historical_frontier(
        agent,
        parent_node,
        raw_historical,
        limit=historical_limit,
        exclude_node_ids=selected_node_ids,
    )
    cards = current_cards + failure_cards + historical_cards
    for rank, card in enumerate(cards, start=1):
        card["strategy_evidence_rank"] = rank
    selection = {
        "schema": "mlevolve_memory_strategy_evidence_selection_v1",
        "stage": str(stage),
        "max_items": max_items,
        "selected_count": len(cards),
        "selected_memory_ids": [str(card.get("memory_id") or "") for card in cards],
        "current_branch_frontier_ids": [
            str(card.get("memory_id") or "") for card in current_cards
        ],
        "causal_failure_ids": [
            str(card.get("memory_id") or "") for card in failure_cards
        ],
        "historical_diverse_frontier_ids": [
            str(card.get("memory_id") or "") for card in historical_cards
        ],
        "historical_lineage_signatures": [
            str(card.get("component_lineage_signature") or "")
            for card in historical_cards
        ],
        "historical_metric_claim_statuses": {
            str(card.get("memory_id") or ""): str(
                card.get("metric_claim_status") or ""
            )
            for card in historical_cards
        },
        "missing_configured_draft_roles": missing_roles,
        "historical_candidate_count": len(raw_historical),
        "historical_rejections": rejected,
        "current_metric_protocol": _node_metric_protocol(parent_node),
        "historical_metric_comparison_policy": (
            "scores rank only inside the same task and validation protocol; "
            "cross-protocol, protocol-unknown, and metric-free cards are method evidence only"
        ),
        "policy": (
            "one best valid node per configured draft role, then one recent causal "
            "failure, then same-task metric-ranked historical nodes followed by "
            "Router-ranked metric-free method evidence, all deduplicated by component "
            "lineage; no cross-protocol numeric blending"
        ),
    }
    return cards, selection


def build_strategy_context(
    agent: Any,
    parent_node: Any,
    *,
    stage: str,
    router_pack: Mapping[str, Any] | None,
    branch_best_metric: float | None = None,
    mode: str = "shadow_read_only",
) -> dict[str, Any]:
    ext_cfg = getattr(getattr(agent, "cfg", None), "external_skill_memory", None)
    evidence_limit = int(
        getattr(ext_cfg, "memory_strategy_evidence_limit", 8) or 8
    )
    current_frontier_slots = int(
        getattr(ext_cfg, "memory_strategy_current_frontier_slots", 3) or 3
    )
    causal_failure_slots = int(
        getattr(ext_cfg, "memory_strategy_causal_failure_slots", 1) or 1
    )
    candidate_pool_limit = int(
        getattr(
            ext_cfg,
            "memory_strategy_candidate_pool_limit",
            getattr(ext_cfg, "memory_strategy_max_cards", 48),
        )
        or 48
    )
    card_max_chars = int(
        getattr(ext_cfg, "memory_strategy_card_max_chars", 6000) or 6000
    )
    cards, evidence_selection = build_strategy_evidence_view(
        agent,
        parent_node,
        stage=stage,
        router_pack=router_pack,
        max_items=evidence_limit,
        current_frontier_slots=current_frontier_slots,
        causal_failure_slots=causal_failure_slots,
        candidate_pool_limit=candidate_pool_limit,
        card_max_chars=card_max_chars,
    )
    total_seconds = int(getattr(getattr(agent, "acfg", None), "time_limit", 0) or 0)
    started = float(getattr(agent, "start_time", time.time()) or time.time())
    elapsed = max(0.0, time.time() - started)
    remaining = max(0.0, total_seconds - elapsed) if total_seconds else None
    current_code = str(getattr(parent_node, "code", "") or "")
    context = {
        "schema": "mlevolve_memory_strategy_context_v2",
        "mode": str(mode),
        "stage": str(stage),
        "strategy_contract": {
            "active": str(mode) == "active_atomic",
            "abstention_allowed": bool(
                getattr(ext_cfg, "memory_strategy_active_allow_abstention", False)
            )
            if str(mode) == "active_atomic"
            else True,
            "min_candidate_compositions": int(
                getattr(
                    ext_cfg,
                    "memory_strategy_debug_min_candidate_compositions",
                    1,
                )
                or 1
            )
            if str(stage) == "debug"
            else int(
                getattr(
                    ext_cfg,
                    "memory_strategy_min_candidate_compositions",
                    3,
                )
                or 1
            ),
        },
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
        "strategy_evidence_selection": evidence_selection,
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
    active_mode = str(context.get("mode") or "") == "active_atomic"
    strategy_contract = dict(context.get("strategy_contract") or {})
    min_compositions = int(
        strategy_contract.get("min_candidate_compositions") or 1
    )
    if active_mode:
        mode_contract = (
            "This is ACTIVE ATOMIC MODE: an accepted memo is production input to a "
            "bounded Planner and Coder. Treat every hypothesis as a staged roadmap whose "
            "smallest independently executable phase may be selected now. "
        )
        if bool(strategy_contract.get("abstention_allowed", False)):
            mode_contract += (
                "Abstention is allowed only when the complete citable evidence set is "
                "genuinely insufficient."
            )
        else:
            mode_contract += (
                "Abstention is forbidden for this required transaction; use current and "
                "historical method evidence to propose a falsifiable next step."
            )
        if str(context.get("stage") or "") == "debug":
            mode_contract += (
                " For Debug, a prompt-visible Router L3 repair is the primary causal "
                "evidence: the recommended hypothesis must cite it and carry out its "
                "exact before/after repair unless the current code proves a concrete API "
                "contradiction, which must be named explicitly. Do not stop at the first "
                "exception line when the repair changes or substitutes a model/library "
                "interface. Audit the complete coupled interface in the current source: "
                "artifact/provider and model identifier, input preprocessing and resolution, "
                "invocation API, output type/key/token selection, feature width, and cache "
                "format. Put every necessary compatibility edit for that selected repair in "
                "the same staged roadmap so the Coder does not discover predictable interface "
                "errors one execution at a time."
            )
    else:
        mode_contract = (
            "This is SHADOW MODE: the memo has no authority to change the live plan."
        )
    system = (
        "You are the Memory Strategy Agent, a read-only task-level research strategist. "
        "Do not write code and do not merely repeat the highest-ranked memory. Decompose each "
        "memory into representation, adaptation mode, downstream estimator, validation, feature, "
        "calibration, compute, and failure components. A full-pipeline score is not an ablation: "
        "The memory_cards list is the complete citable evidence set and is capped across all "
        "sources. Treat current_branch_frontier, causal_failure_evidence, and "
        "historical_diverse_frontier as distinct evidence classes. Compare numeric metrics only "
        "when their task and validation protocol are compatible; an unverified historical score "
        "is evidence about a method, not proof that it beats a current branch. "
        "do not discard one component solely because the pipeline containing it scored poorly. "
        "Instead ask whether the failure came from another component or an incompatible interface. "
        "Build a map of the current system, identify missing crossings in the component portfolio, "
        "and propose globally coherent hypotheses. If evidence is sufficient, decision must be "
        "'propose' and candidate_compositions must contain 3 to 5 distinct hypotheses: include "
        "at least one conservative transfer, at least one cross-lineage composition citing two or "
        "more memory IDs, and at least one discriminating ablation or frontier alternative. At "
        "least one composition must explicitly examine a component combination not yet jointly "
        "tested. If an axis contains three or more independently observed alternatives and they "
        "share a budget-compatible interface (for example frozen features or prediction vectors), "
        "include at least one diversity/set-cover hypothesis that combines alternatives within "
        "that axis. The deterministic component_portfolio may list these as "
        "within_axis_diversity_opportunities. Address every listed opportunity exactly once in "
        "addressed_opportunities. Use disposition='proposed' with the corresponding hypothesis_id, "
        "or disposition='declined' with a concrete compatibility/budget reason. A weak source "
        "pipeline score alone is not a valid decline reason because it does not isolate the shared "
        "component. When disposition='proposed', the referenced hypothesis must explicitly name "
        "every alternative listed by that opportunity and the shared interface used to combine "
        "them; a two-item subset does not satisfy a three-alternative opportunity. Do not make "
        "every candidate a descendant of the current best lineage: when "
        "the portfolio permits it, at least two candidates must use a different base lineage. "
        "You may combine ideas only when interfaces and compute budgets are compatible. "
        "If evidence is genuinely insufficient, decision may be 'abstain', candidate_compositions "
        "must be empty, and abstention_reason must be specific. Every proposal must "
        "cite only memory IDs present in memory_cards, distinguish the parent's real "
        "submission-aligned metric from branch_best_metric, estimate full training cost, name "
        "conflicts, preserve a minimal change set, and state a falsification condition. Reject "
        "duplicate, already-tried, over-budget, or unsupported combinations. Output one JSON "
        "object only using the exact field names in RESPONSE_SCHEMA; do not substitute fields such "
        "as proposed_experiment, experiment, changes, or memory_ids. "
        + mode_contract
        + "\n\nRESPONSE_SCHEMA:\n"
        + exact_contract
    )
    if str(context.get("stage") or "") == "debug":
        system += (
            "\n\nDEBUG_STAGE_RULE: Treat the execution output as causal evidence. Propose "
            f"{min_compositions} to 3 hypotheses rather than the general 3-to-5 Improve "
            "portfolio. The recommended hypothesis must be a targeted_repair whose first "
            "phase only fixes the narrowest demonstrated exception while explicitly preserving "
            "model, data, validation, calibration, and submission behavior. Put optional OOF, "
            "calibration, ensembling, feature expansion, or performance work into later roadmap "
            "phases; never include it in the first repair unless it is the demonstrated root "
            "cause. Do not turn a path, cache, import, or cleanup exception into an architecture "
            "rewrite."
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
                    "repair_checklist": {
                        "decision_propose_composition_count": (
                            f"{min_compositions}..3"
                            if str(context.get("stage") or "") == "debug"
                            else f"{min_compositions}..5"
                        ),
                        "required_composition_fields": list(
                            _COMPOSITION_REQUIRED_KEYS
                        ),
                        "required_opportunity_ids": [
                            str(item.get("opportunity_id") or "")
                            for item in (
                                (context.get("component_portfolio") or {}).get(
                                    "within_axis_diversity_opportunities", []
                                )
                            )
                            if isinstance(item, Mapping)
                            and item.get("opportunity_id")
                        ],
                        "recommendation_required_when_compositions_nonempty": True,
                    },
                    "instruction": (
                        "Return a corrected complete object. Preserve useful analysis, but use "
                        "the exact schema and satisfy every listed violation. Before returning, "
                        "audit every composition against required_composition_fields, ensure "
                        "recommended_hypothesis_id names an existing complete composition, and "
                        "address every required opportunity exactly once. Drop an incomplete "
                        "candidate only if at least three complete candidates remain."
                    ),
                }
            )
        )
    return {"system": system, "user": user, "assistant": "{"}


def _parse_json_object(response: Any) -> dict[str, Any]:
    if isinstance(response, Mapping):
        return copy.deepcopy(dict(response))
    text = str(response or "").strip()
    if (
        not text.startswith("{")
        and text.endswith("}")
        and re.match(r'^"[^"\\]+"\s*:', text)
    ):
        # The assistant prefill supplied the opening brace, so some providers
        # stream only the continuation beginning with the first JSON key.
        text = "{" + text
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end >= start:
            text = text[start : end + 1]
        elif start < 0 and end >= 0 and ":" in text:
            # Chat assistant prefill may contain the opening brace while the
            # provider returns only the continuation. Reattach that one known
            # framing character; all substantive JSON still comes from the
            # model and remains subject to the full contract validator.
            text = "{" + text[: end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("strategy response is not a JSON object")
    return value


def _strategy_json_normalization_prompt(
    response: Any,
) -> dict[str, str]:
    """Ask a non-thinking pass to transcribe reasoning into strict JSON.

    This pass has no strategy authority: it may repair serialization and fill
    required container fields, but it may not invent hypotheses, citations,
    metrics, or recommendations absent from the thinking response.
    """

    return {
        "system": (
            "You are a lossless JSON transcriber, not a strategy agent. Convert "
            "the supplied Strategy Agent response into one syntactically valid "
            "JSON object matching the supplied schema. Preserve every substantive "
            "claim, hypothesis, source_memory_id, metric, estimate, and decision. "
            "Do not add new hypotheses, citations, evidence, recommendations, or "
            "numbers. You may only repair JSON syntax and add missing empty "
            "containers required by the schema. Return JSON only."
        ),
        "user": (
            "STRATEGY_SCHEMA_JSON:\n"
            + _canonical_json(STRATEGY_MEMO_SCHEMA)
            + "\n\nMALFORMED_STRATEGY_RESPONSE:\n"
            + str(response or "")
        ),
    }


def _normalize_strategy_json(
    agent: Any,
    *,
    response: Any,
    strategy_cfg: Any,
    strategy_model: str,
    ext_cfg: Any,
) -> tuple[Any, dict[str, Any]]:
    normalizer_model = str(
        getattr(ext_cfg, "memory_strategy_json_normalization_model", "")
        or strategy_model
    )
    normalizer_cfg = copy.deepcopy(strategy_cfg)
    normalizer_cfg.agent.code.model = normalizer_model
    prompt = _strategy_json_normalization_prompt(response)
    query_fn = getattr(agent, "_memory_strategy_json_normalizer_fn", None)
    if callable(query_fn):
        normalized = query_fn(
            prompt=copy.deepcopy(prompt),
            response=response,
            json_schema=copy.deepcopy(STRATEGY_MEMO_SCHEMA),
            model=normalizer_model,
            thinking_enabled=False,
        )
    else:
        normalized = generate(
            prompt=prompt,
            cfg=normalizer_cfg,
            temperature=0.0,
            max_tokens=int(
                getattr(
                    ext_cfg,
                    "memory_strategy_json_normalization_max_tokens",
                    12000,
                )
                or 12000
            ),
            json_schema=STRATEGY_MEMO_SCHEMA,
            max_retries=int(
                getattr(
                    ext_cfg,
                    "memory_strategy_json_normalization_max_retries",
                    2,
                )
                or 2
            ),
        )
    return normalized, {
        "used": True,
        "authority": "serialization_only",
        "model": normalizer_model,
        "thinking_enabled": False,
        "response_sha256": hashlib.sha256(
            _canonical_json(normalized).encode("utf-8")
        ).hexdigest(),
    }


def validate_strategy_memo(
    memo: Mapping[str, Any],
    *,
    available_memory_ids: Iterable[str],
    min_candidate_compositions: int = 1,
    max_candidate_compositions: int = 5,
    abstention_allowed: bool = True,
    required_opportunity_ids: Iterable[str] = (),
    required_opportunities: Iterable[Mapping[str, Any]] = (),
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
    if decision == "abstain" and not abstention_allowed:
        violations.append("abstention is disabled for required active Strategy actuation")
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
    opportunity_details = {
        str(row.get("opportunity_id") or ""): dict(row)
        for row in required_opportunities
        if isinstance(row, Mapping) and row.get("opportunity_id")
    }
    required_opportunity_set = {
        str(value) for value in required_opportunity_ids
    } | set(opportunity_details)
    composition_by_id = {
        str(item.get("hypothesis_id") or ""): item
        for item in compositions
        if isinstance(item, Mapping) and item.get("hypothesis_id")
    }
    addressed_rows = list(memo.get("addressed_opportunities") or [])
    addressed_ids: list[str] = []
    for index, row in enumerate(addressed_rows):
        if not isinstance(row, Mapping):
            violations.append(f"addressed_opportunities[{index}] is not an object")
            continue
        opportunity_id = str(row.get("opportunity_id") or "")
        disposition = str(row.get("disposition") or "")
        hypothesis_id = str(row.get("hypothesis_id") or "")
        if not opportunity_id:
            violations.append(f"addressed_opportunities[{index}] has no opportunity_id")
        elif opportunity_id in addressed_ids:
            violations.append(f"duplicate addressed opportunity: {opportunity_id}")
        addressed_ids.append(opportunity_id)
        if disposition not in {"proposed", "declined"}:
            violations.append(
                f"{opportunity_id or index} disposition must be proposed or declined"
            )
        if disposition == "proposed" and hypothesis_id not in hypothesis_ids:
            violations.append(
                f"{opportunity_id or index} proposed disposition requires a valid hypothesis_id"
            )
        if disposition == "proposed" and hypothesis_id in composition_by_id:
            opportunity = opportunity_details.get(opportunity_id) or {}
            alternatives = [
                str(value)
                for value in (opportunity.get("alternatives") or [])
                if value
            ]
            composition_text = _canonical_json(
                {
                    key: composition_by_id[hypothesis_id].get(key)
                    for key in (
                        "hypothesis",
                        "minimal_change_set",
                        "expected_mechanism",
                    )
                }
            )
            missing_alternatives = [
                alternative
                for alternative in alternatives
                if not re.search(
                    re.escape(alternative).replace(r"\_", r"[- _]?"),
                    composition_text,
                    re.IGNORECASE,
                )
            ]
            if missing_alternatives:
                violations.append(
                    f"{opportunity_id} proposed hypothesis {hypothesis_id} does not "
                    f"explicitly cover alternatives: {missing_alternatives}"
                )
        if disposition == "declined" and not str(row.get("reason") or "").strip():
            violations.append(
                f"{opportunity_id or index} declined disposition requires a reason"
            )
    missing_opportunities = sorted(
        required_opportunity_set - set(addressed_ids)
    )
    if missing_opportunities:
        violations.append(
            f"unaddressed within-axis diversity opportunities: {missing_opportunities}"
        )
    unknown_opportunities = sorted(
        set(addressed_ids) - required_opportunity_set
    )
    if unknown_opportunities:
        violations.append(
            f"addressed unknown diversity opportunities: {unknown_opportunities}"
        )
    return {
        "schema": "mlevolve_memory_strategy_validation_v1",
        "valid": not violations,
        "violations": violations,
        "available_memory_ids": sorted(available),
        "hypothesis_ids": hypothesis_ids,
        "decision": decision,
        "composition_count": len(compositions),
        "required_opportunity_ids": sorted(required_opportunity_set),
        "addressed_opportunity_ids": addressed_ids,
    }


def should_run_memory_strategy_shadow(
    agent: Any,
    *,
    stage: str,
    parent_node: Any,
    router_pack: Mapping[str, Any] | None,
    force_run: bool = False,
) -> tuple[bool, str]:
    if force_run:
        return True, "active_required"
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
    _mode: str = "shadow_read_only",
    _actuation_authority: str = "none",
    _force_run: bool = False,
) -> dict[str, Any]:
    """Run the Strategy Agent and return a Journal-ready trace.

    The public defaults preserve the original read-only shadow contract.  The
    active wrapper below uses the same evidence/contract machinery but marks
    the trace as production-authorized; code generation remains a separate,
    machine-bounded Atomic Planner/Coder transaction.
    """

    enabled, trigger_reason = should_run_memory_strategy_shadow(
        agent,
        stage=stage,
        parent_node=parent_node,
        router_pack=router_pack,
        force_run=_force_run,
    )
    base_trace = {
        "schema": "mlevolve_memory_strategy_shadow_trace_v2",
        "mode": str(_mode),
        "stage": str(stage),
        "parent_node_id": str(getattr(parent_node, "id", "")),
        "enabled": enabled,
        "trigger_reason": trigger_reason,
        "actuation_authority": str(_actuation_authority),
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
        mode=_mode,
    )
    cards = list(context.get("memory_cards") or [])
    evidence_selection = copy.deepcopy(
        context.get("strategy_evidence_selection") or {}
    )
    evidence_limit = int(evidence_selection.get("max_items") or 8)
    if len(cards) > evidence_limit:
        return {
            **base_trace,
            "status": "failed",
            "error_type": "StrategyEvidenceLimitViolation",
            "error": (
                f"strategy evidence count {len(cards)} exceeds configured limit "
                f"{evidence_limit}"
            ),
            "strategy_evidence_selection": evidence_selection,
            "memory_card_ids": [card.get("memory_id", "") for card in cards],
            "memory_card_count": len(cards),
        }
    required_opportunities = [
        copy.deepcopy(dict(item))
        for item in (
            (context.get("component_portfolio") or {}).get(
                "within_axis_diversity_opportunities", []
            )
        )
        if isinstance(item, Mapping) and item.get("opportunity_id")
    ]
    required_opportunity_ids = [
        str(item.get("opportunity_id") or "")
        for item in required_opportunities
    ]
    ext_cfg = getattr(agent.cfg, "external_skill_memory", None)
    inherited_model = str(getattr(agent.acfg.code, "model", "") or "")
    strategy_model = str(
        getattr(ext_cfg, "memory_strategy_model", inherited_model)
        or inherited_model
    )
    thinking_enabled = bool(
        getattr(ext_cfg, "memory_strategy_thinking_enabled", True)
    )
    strategy_cfg = copy.deepcopy(agent.cfg)
    strategy_cfg.agent.code.model = strategy_model
    started = time.monotonic()
    try:
        contract_retries = int(
            getattr(ext_cfg, "memory_strategy_contract_retries", 2) or 0
        )
        min_compositions = (
            int(
                getattr(
                    ext_cfg,
                    "memory_strategy_debug_min_candidate_compositions",
                    1,
                )
                or 1
            )
            if str(stage) == "debug"
            else int(
                getattr(ext_cfg, "memory_strategy_min_candidate_compositions", 3)
                or 1
            )
        )
        abstention_allowed = not _force_run or bool(
            getattr(ext_cfg, "memory_strategy_active_allow_abstention", False)
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
                    model=strategy_model,
                    thinking_enabled=thinking_enabled,
                )
            else:
                response = generate(
                    prompt=prompt,
                    cfg=strategy_cfg,
                    temperature=float(
                        getattr(ext_cfg, "memory_strategy_temperature", 0.0) or 0.0
                    ),
                    max_tokens=int(
                        getattr(ext_cfg, "memory_strategy_max_output_tokens", 12000)
                        or 12000
                    ),
                    # DeepSeek thinking and response_format=json_object are separate
                    # modes.  In thinking mode the explicit prompt contract plus the
                    # host validator/retry loop enforce structure after reasoning.
                    json_schema=None if thinking_enabled else STRATEGY_MEMO_SCHEMA,
                    max_retries=int(
                        getattr(ext_cfg, "memory_strategy_max_retries", 2) or 2
                    ),
                )
            normalization: dict[str, Any] = {
                "used": False,
                "authority": "serialization_only",
            }
            try:
                try:
                    memo = _parse_json_object(response)
                except Exception as initial_parse_error:
                    normalization["initial_parse_error"] = (
                        f"{type(initial_parse_error).__name__}: "
                        f"{initial_parse_error}"
                    )
                    normalization_enabled = bool(
                        getattr(
                            ext_cfg,
                            "memory_strategy_json_normalization_enabled",
                            True,
                        )
                    )
                    if not normalization_enabled or not str(response or "").strip():
                        raise
                    normalized_response, normalized_trace = _normalize_strategy_json(
                        agent,
                        response=response,
                        strategy_cfg=strategy_cfg,
                        strategy_model=strategy_model,
                        ext_cfg=ext_cfg,
                    )
                    normalization.update(normalized_trace)
                    try:
                        memo = _parse_json_object(normalized_response)
                    except Exception as normalized_parse_error:
                        normalization["normalized_parse_error"] = (
                            f"{type(normalized_parse_error).__name__}: "
                            f"{normalized_parse_error}"
                        )
                        raise ValueError(
                            "strategy response remained invalid after "
                            "serialization-only normalization"
                        ) from normalized_parse_error
                validation = validate_strategy_memo(
                    memo,
                    available_memory_ids=[card["memory_id"] for card in cards],
                    min_candidate_compositions=min_compositions,
                    abstention_allowed=abstention_allowed,
                    required_opportunity_ids=required_opportunity_ids,
                    required_opportunities=required_opportunities,
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
                    "json_normalization": normalization,
                }
            )
            if validation.get("valid"):
                break
        status = "completed" if validation["valid"] else "completed_with_contract_violations"
        return {
            **base_trace,
            "status": status,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "model": strategy_model,
            "inherited_model": inherited_model,
            "thinking_enabled": thinking_enabled,
            "context_sha256": payload_sha256(context),
            "context_char_count": len(_canonical_json(context)),
            "memory_card_ids": [card["memory_id"] for card in cards],
            "memory_card_count": len(cards),
            "memory_cards": copy.deepcopy(cards),
            "strategy_evidence_selection": evidence_selection,
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
            "model": strategy_model,
            "inherited_model": inherited_model,
            "thinking_enabled": thinking_enabled,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "context_sha256": payload_sha256(context),
            "context_char_count": len(_canonical_json(context)),
            "memory_card_ids": [card["memory_id"] for card in cards],
            "memory_card_count": len(cards),
            "memory_cards": copy.deepcopy(cards),
            "strategy_evidence_selection": evidence_selection,
        }


def run_memory_strategy_active(
    agent: Any,
    parent_node: Any,
    *,
    stage: str,
    router_pack: Mapping[str, Any] | None,
    branch_best_metric: float | None = None,
    production_prompt_sha256: str = "",
) -> dict[str, Any]:
    """Run the required production Strategy analysis for atomic actuation."""

    return run_memory_strategy_shadow(
        agent,
        parent_node,
        stage=stage,
        router_pack=router_pack,
        branch_best_metric=branch_best_metric,
        production_prompt_sha256=production_prompt_sha256,
        _mode="active_atomic",
        _actuation_authority="atomic_planner_coder",
        _force_run=True,
    )


__all__ = [
    "STRATEGY_MEMO_SCHEMA",
    "build_component_portfolio",
    "build_memory_cards",
    "build_strategy_context",
    "build_strategy_evidence_view",
    "code_component_fingerprint",
    "payload_sha256",
    "run_memory_strategy_active",
    "run_memory_strategy_shadow",
    "should_run_memory_strategy_shadow",
    "validate_strategy_memo",
]
