"""Typed, claim-level memory primitives for failure repair retrieval.

The historical RunForest stores whole programs.  A whole program can be unsafe
for positive reuse while still containing a narrowly verified runtime fact
(for example, a backbone's required input resolution).  This module keeps the
runtime fact separate from the program, metric, and method claims and provides
the deterministic signature used by the Debug ranker.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any


ATOMIC_CLAIM_SCHEMA = "mlevolve_atomic_memory_claim_v1"
ATOMIC_CLAIM_BUNDLE_SCHEMA = "mlevolve_atomic_memory_claim_bundle_v1"
AUTHORIZED_DEBUG_STATUS = "authorized_debug_only"

_EXCEPTION_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*(?:Error|Exception))\b"
)
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?:[xX]\d+(?:\.\d+)?)?(?![A-Za-z0-9_])")
_SHAPE_RE = re.compile(
    r"(?:torch\.Size\s*\(\s*)?[\[(]\s*\d+(?:\s*[,xX]\s*\d+)+\s*[\])]?"
)
_PATH_RE = re.compile(
    r"(?:\.{0,2}/|/)[A-Za-z0-9_.~+@%=-]+(?:/[A-Za-z0-9_.~+@%=-]+)+/?"
)
_QUOTED_RE = re.compile(r"[`'\"]([A-Za-z_][A-Za-z0-9_.+:/-]{2,})[`'\"]")
_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_MODEL_RE = re.compile(
    r"\b(?:dinov\d*|vit(?:amin)?|siglip\d*|deberta|distilbert|modernbert|"
    r"resnet|efficientnet|convnext|swin|lightgbm|lgbm|xgboost|xgb|catboost|torch(?:vision)?|"
    r"timm)[A-Za-z0-9_.+:/-]*\b",
    flags=re.I,
)

_GENERIC_SYMBOLS = {
    "assertionerror",
    "batch",
    "candidate",
    "current",
    "dataframe",
    "debug",
    "error",
    "failed",
    "failure",
    "feature",
    "file",
    "height",
    "image",
    "input",
    "line",
    "match",
    "model",
    "output",
    "repair",
    "runtime",
    "shape",
    "size",
    "tensor",
    "traceback",
    "training",
    "validation",
}


def canonical_task_id(value: object) -> str:
    task = str(value or "").strip()
    while task.startswith("full-"):
        task = task[len("full-") :]
    return task


def _normalized_literal(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "").strip().lower())


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def extract_debug_signature(text: object) -> dict[str, list[str]]:
    """Extract high-information runtime anchors without discarding numbers.

    The old tokenizer required a leading letter and silently removed values
    such as ``224`` and ``518``.  Debug retrieval needs those operands, exact
    exception names, model/API identifiers, symbols, paths, and tensor shapes.
    """

    raw = str(text or "")
    lowered = raw.lower()
    quoted = [_normalized_literal(value) for value in _QUOTED_RE.findall(raw)]
    models = [_normalized_literal(value) for value in _MODEL_RE.findall(raw)]
    symbols = [
        value.lower()
        for value in _SYMBOL_RE.findall(raw)
        if (
            "_" in value
            or value.isupper()
            or any(character.isdigit() for character in value)
        )
        and value.lower() not in _GENERIC_SYMBOLS
    ]
    keywords = [
        marker
        for marker in (
            "broadcast",
            "dimension mismatch",
            "doesn't match",
            "expected",
            "input height",
            "input size",
            "missing",
            "nan",
            "not found",
            "out of memory",
            "required",
            "unexpected keyword",
            "unsupported",
        )
        if marker in lowered
    ]
    return {
        "exception_names": _ordered_unique(
            [value.lower() for value in _EXCEPTION_RE.findall(raw)]
        ),
        "model_api_ids": _ordered_unique(models),
        "quoted_identifiers": _ordered_unique(quoted),
        "numeric_literals": _ordered_unique(
            [_normalized_literal(value) for value in _NUMBER_RE.findall(raw)]
        ),
        "shape_literals": _ordered_unique(
            [_normalized_literal(value) for value in _SHAPE_RE.findall(raw)]
        ),
        "path_literals": _ordered_unique(
            [_normalized_literal(value) for value in _PATH_RE.findall(raw)]
        ),
        "symbol_names": _ordered_unique(symbols + quoted),
        "failure_keywords": keywords,
    }


def extract_before_after(text: object) -> list[dict[str, str]]:
    """Extract explicit, bounded before/after parameter changes."""

    raw = str(text or "")
    rows: list[dict[str, str]] = []
    patterns = (
        re.compile(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\b.{0,60}?\bfrom\s+([0-9.]+)\s+(?:to|->|→)\s+([0-9.]+)",
            flags=re.I | re.S,
        ),
        re.compile(
            r"\b(?:change|set|replace)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|:)?\s*([0-9.]+)?\s*(?:to|->|→|=)\s*([0-9.]+)",
            flags=re.I,
        ),
    )
    for pattern in patterns:
        for symbol, before, after in pattern.findall(raw):
            if not after:
                continue
            rows.append(
                {
                    "symbol": str(symbol),
                    "before": str(before or ""),
                    "after": str(after),
                }
            )
    # Runtime assertions often state actual and expected operands even when
    # the repair plan uses prose rather than an assignment.
    for actual, expected in re.findall(
        r"input\s+(?:height|size)\s*\(?\s*([0-9.]+)\s*\)?.{0,80}?"
        r"(?:doesn['’]?t\s+match|expected|required).{0,30}?\(?\s*([0-9.]+)\s*\)?",
        raw,
        flags=re.I | re.S,
    ):
        rows.append({"symbol": "input_size", "before": actual, "after": expected})
    unique: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["symbol"].lower(), row["before"], row["after"])
        unique[key] = row
    return list(unique.values())


def _set_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _model_family_tokens(values: set[str]) -> set[str]:
    families = set(values)
    known = (
        "dinov2",
        "dinov3",
        "siglip2",
        "siglip",
        "deberta",
        "distilbert",
        "modernbert",
        "resnet",
        "efficientnet",
        "lightgbm",
        "lgbm",
        "xgboost",
        "xgb",
        "catboost",
        "torch",
        "timm",
    )
    for value in values:
        compact = re.sub(r"[^a-z0-9]+", "", value.lower())
        families.update(token for token in known if token in compact)
    return families


def structured_debug_relevance(
    query_text: object,
    candidate_failure: object,
    candidate_repair: object = "",
    claim: Mapping[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Rank Debug evidence by causal anchors, not document-length overlap."""

    query = extract_debug_signature(query_text)
    candidate = extract_debug_signature(
        f"{candidate_failure or ''}\n{candidate_repair or ''}"
    )
    claim = claim if isinstance(claim, Mapping) else {}
    claim_signature = claim.get("failure_signature")
    if isinstance(claim_signature, Mapping):
        for key in query:
            values = claim_signature.get(key)
            if isinstance(values, list):
                candidate[key] = _ordered_unique(
                    [*candidate.get(key, []), *(str(value).lower() for value in values)]
                )
    before_after = claim.get("before_after")
    expected_values = {
        str(row.get("after") or "").lower()
        for row in before_after or []
        if isinstance(row, Mapping) and row.get("after") not in (None, "")
    }
    actual_values = {
        str(row.get("before") or "").lower()
        for row in before_after or []
        if isinstance(row, Mapping) and row.get("before") not in (None, "")
    }

    query_sets = {key: set(values) for key, values in query.items()}
    candidate_sets = {key: set(values) for key, values in candidate.items()}
    shared = {
        key: sorted(query_sets[key] & candidate_sets[key])
        for key in query_sets
    }
    exception_score = _set_score(
        query_sets["exception_names"], candidate_sets["exception_names"]
    )
    model_score = _set_score(
        _model_family_tokens(query_sets["model_api_ids"]),
        _model_family_tokens(candidate_sets["model_api_ids"]),
    )
    numeric_shared = query_sets["numeric_literals"] & candidate_sets["numeric_literals"]
    expected_shared = query_sets["numeric_literals"] & expected_values
    operand_score = min(
        1.0,
        (0.75 if expected_shared else 0.0)
        + (0.20 if numeric_shared else 0.0)
        + (0.05 if query_sets["numeric_literals"] & actual_values else 0.0),
    )
    shape_score = _set_score(
        query_sets["shape_literals"], candidate_sets["shape_literals"]
    )
    symbol_score = _set_score(
        query_sets["symbol_names"], candidate_sets["symbol_names"]
    )
    path_score = _set_score(
        query_sets["path_literals"], candidate_sets["path_literals"]
    )
    keyword_score = _set_score(
        query_sets["failure_keywords"], candidate_sets["failure_keywords"]
    )
    components = {
        "exception": exception_score,
        "model_api": model_score,
        "operand": operand_score,
        "shape": shape_score,
        "symbol": symbol_score,
        "path": path_score,
        "failure_keyword": keyword_score,
    }
    score = (
        0.27 * exception_score
        + 0.23 * model_score
        + 0.25 * operand_score
        + 0.08 * shape_score
        + 0.08 * symbol_score
        + 0.04 * path_score
        + 0.05 * keyword_score
    )
    # Exact exception plus a concrete model/API and expected operand is the
    # strongest portable compatibility signature.
    exact_compatibility = bool(exception_score and model_score and expected_shared)
    if exact_compatibility:
        score = max(score, 0.92)
    return min(1.0, score), {
        "schema": "mlevolve_structured_debug_rank_receipt_v1",
        "components": components,
        "shared_anchors": shared,
        "expected_values": sorted(expected_values),
        "shared_expected_values": sorted(expected_shared),
        "exact_compatibility_match": exact_compatibility,
    }


def verified_atomic_debug_claim(
    transition: Mapping[str, Any],
) -> tuple[bool, str]:
    """Validate the independent claim gate used by the runtime memory layer."""

    claim = transition.get("atomic_repair_claim")
    if not isinstance(claim, Mapping):
        return False, "missing_atomic_repair_claim"
    if claim.get("schema") != ATOMIC_CLAIM_SCHEMA:
        return False, "unsupported_atomic_claim_schema"
    if claim.get("claim_status") != AUTHORIZED_DEBUG_STATUS:
        return False, "atomic_claim_not_authorized"
    if claim.get("claim_type") not in {
        "repair_claim",
        "compatibility_claim",
        "resource_claim",
        "implementation_claim",
    }:
        return False, "atomic_claim_type_not_debug_actionable"
    visibility = claim.get("operation_visibility")
    allowed = set(visibility.get("allowed_operations") or []) if isinstance(visibility, Mapping) else set()
    if not {"debug_hypothesis", "debug_repair"} <= allowed:
        return False, "atomic_claim_missing_debug_visibility"
    verification = claim.get("verification")
    if not isinstance(verification, Mapping):
        return False, "atomic_claim_missing_verification"
    if not all(
        verification.get(key) is True
        for key in (
            "observed_parent_failure",
            "observed_child_execution_success",
            "repair_action_bound_to_transition",
            "claim_scope_independently_audited",
        )
    ):
        return False, "atomic_claim_verification_incomplete"
    taint = claim.get("taint")
    if not isinstance(taint, Mapping) or taint.get("claim") != "clean":
        return False, "atomic_claim_scope_tainted"
    if claim.get("metric_authorized") is not False:
        return False, "atomic_claim_must_not_authorize_metric"
    if str(transition.get("task") or "") != str(claim.get("task_id") or ""):
        return False, "atomic_claim_task_mismatch"
    if not str(claim.get("failure_text") or "").strip():
        return False, "atomic_claim_missing_failure_text"
    if not str(claim.get("repair_action") or "").strip():
        return False, "atomic_claim_missing_repair_action"
    return True, "verified_atomic_debug_claim"
