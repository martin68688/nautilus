"""Host-owned verification that final code still implements Planner intent.

Patch-shape verification cannot prove runtime semantics.  In particular, a
reviewed candidate can be valid Python and remain within its patch budget while
silently retaining an old training loop that contradicts an explicit roadmap
to reuse frozen checkpoints without retraining.  A reviewer can also delete a
predeclared absolute metric-selection gate while leaving valid Python inside
the same edit budget.  This module checks those explicit, mechanically
verifiable invariants on the final executable source, after all reviewer edits.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Iterator, Mapping


_NO_RETRAIN_PATTERNS = (
    re.compile(r"\bwithout\s+(?:any\s+)?re[- ]?train(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\bno\s+re[- ]?train(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\b(?:do\s+not|never)\s+re[- ]?(?:train|fit)\b", re.IGNORECASE),
)
_CHECKPOINT_REUSE_PATTERN = re.compile(
    r"\bexisting\b[^.\n]{0,80}\bfrozen\b[^.\n]{0,80}\bcheckpoints?\b|"
    r"\b(?:reuse|replay|load|use)\w*\b[^.\n]{0,100}\bfrozen\b"
    r"[^.\n]{0,80}\bcheckpoints?\b|"
    r"\bfrozen\b[^.\n]{0,80}\bcheckpoints?\b[^.\n]{0,100}"
    r"\b(?:reuse|replay|load|use)\w*\b|"
    r"\bcheckpoints?\b[^.\n]{0,80}\b(?:remain|keep|kept)\w*\s+frozen\b|"
    r"\b(?:same\s+)?exact\s+retained\b[^.\n]{0,100}\bcheckpoints?\b|"
    r"\b(?:reuse|replay|load|use)\w*\b[^.\n]{0,100}"
    r"\bexact\s+retained\b[^.\n]{0,80}\bcheckpoints?\b",
    re.IGNORECASE,
)
_STRICT_ABSOLUTE_THRESHOLD_PATTERNS = (
    re.compile(
        r"\bstrictly\s+(?:exceeds?|greater\s+than)\b[^.\n]{0,180}?"
        r"\b(0\.\d{4,})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\babsolute\b[^.\n]{0,80}?\bthreshold\b[^.\n]{0,80}?"
        r"\b(0\.\d{4,})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(0\.\d{4,})\b[^.\n]{0,80}?\babsolute\b"
        r"[^.\n]{0,40}?\bthreshold\b",
        re.IGNORECASE,
    ),
)


def _plan_strings(value: Any, path: str = "plan") -> Iterator[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _plan_strings(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _plan_strings(item, f"{path}[{index}]")
    elif isinstance(value, str) and value.strip():
        yield path, value.strip()


def _no_retraining_evidence(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for path, text in _plan_strings(plan):
        direct_match = any(pattern.search(text) for pattern in _NO_RETRAIN_PATTERNS)
        frozen_reuse_match = bool(_CHECKPOINT_REUSE_PATTERN.search(text))
        if direct_match or frozen_reuse_match:
            evidence.append(
                {
                    "path": path,
                    "text": text[:500],
                    "detector": (
                        "explicit_no_retraining"
                        if direct_match
                        else "frozen_checkpoint_reuse"
                    ),
                }
            )
    return evidence[:8]


def _absolute_metric_threshold_evidence(
    plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract only explicit, high-precision absolute selection thresholds."""

    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for path, text in _plan_strings(plan):
        for pattern in _STRICT_ABSOLUTE_THRESHOLD_PATTERNS:
            for match in pattern.finditer(text):
                value = float(match.group(1))
                key = (path, value)
                if key in seen:
                    continue
                seen.add(key)
                evidence.append(
                    {
                        "path": path,
                        "text": text[:500],
                        "detector": "explicit_absolute_metric_threshold",
                        "value": value,
                    }
                )
    return evidence[:8]


def _call_label(call: ast.Call) -> str:
    target = call.func
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return ".".join(reversed(parts))


def _gradient_training_loops(tree: ast.AST) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for loop in ast.walk(tree):
        if not isinstance(loop, (ast.For, ast.AsyncFor, ast.While)):
            continue
        calls = [
            call
            for statement in loop.body
            for call in ast.walk(statement)
            if isinstance(call, ast.Call)
        ]
        backward_calls = [
            call
            for call in calls
            if _call_label(call).split(".")[-1:] == ["backward"]
        ]
        step_calls = [
            call
            for call in calls
            if _call_label(call).split(".")[-1:] == ["step"]
        ]
        if not backward_calls or not step_calls:
            continue
        observations.append(
            {
                "loop_line": int(getattr(loop, "lineno", 0) or 0),
                "loop_kind": type(loop).__name__,
                "backward_calls": [
                    {
                        "line": int(getattr(call, "lineno", 0) or 0),
                        "call": _call_label(call),
                    }
                    for call in backward_calls[:4]
                ],
                "step_calls": [
                    {
                        "line": int(getattr(call, "lineno", 0) or 0),
                        "call": _call_label(call),
                    }
                    for call in step_calls[:4]
                ],
            }
        )
    return observations[:8]


def _numeric_constant(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
    ):
        value = _numeric_constant(node.operand)
        if value is not None:
            return -value if isinstance(node.op, ast.USub) else value
    return None


def _threshold_comparisons(
    tree: ast.AST,
    thresholds: set[float],
) -> list[dict[str, Any]]:
    """Find comparisons that consume a literal or bound required threshold."""

    bindings: dict[str, float] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value_node = node.value
            numeric = _numeric_constant(value_node) if value_node is not None else None
            if numeric is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = numeric

    observations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        matched_values: set[float] = set()
        matched_names: set[str] = set()
        for operand in operands:
            numeric = _numeric_constant(operand)
            if numeric is not None and any(
                abs(numeric - threshold) <= 1e-12 for threshold in thresholds
            ):
                matched_values.add(numeric)
            if isinstance(operand, ast.Name):
                bound = bindings.get(operand.id)
                if bound is not None and any(
                    abs(bound - threshold) <= 1e-12 for threshold in thresholds
                ):
                    matched_values.add(bound)
                    matched_names.add(operand.id)
        if matched_values:
            observations.append(
                {
                    "line": int(getattr(node, "lineno", 0) or 0),
                    "values": sorted(matched_values),
                    "bound_names": sorted(matched_names),
                }
            )
    return observations[:8]


def verify_host_plan_execution(
    plan: Mapping[str, Any] | None,
    code: str,
) -> dict[str, Any]:
    """Verify explicit mechanically checkable roadmap semantics.

    This intentionally does not infer a no-retraining requirement from generic
    preservation language such as "do not alter the training loop" or
    "unchanged checkpoint selection remains executable".  The gate activates
    only when the Planner explicitly promises no retraining or reuse of
    existing frozen checkpoints.
    """

    plan_object = dict(plan or {})
    no_retraining_evidence = _no_retraining_evidence(plan_object)
    threshold_evidence = _absolute_metric_threshold_evidence(plan_object)
    requirements: list[str] = []
    if no_retraining_evidence:
        requirements.append("frozen_checkpoint_reuse_without_retraining")
    if threshold_evidence:
        requirements.append("absolute_metric_selection_threshold")
    requirement_evidence = [*no_retraining_evidence, *threshold_evidence]
    base: dict[str, Any] = {
        "schema": "mlevolve_host_plan_execution_gate_v1",
        "enabled": bool(requirements),
        "requirements": requirements,
        "requirement_evidence": requirement_evidence,
        "observations": {
            "active_gradient_training_loops": [],
            "absolute_metric_threshold_comparisons": [],
        },
        "valid": True,
        "violations": [],
    }
    if not requirements:
        return base

    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError as error:
        violation = (
            "plan_runtime_verification_unavailable: final executable source has "
            f"a syntax error at line {int(error.lineno or 0)}"
        )
        base["observations"]["syntax_error"] = str(error)
        base["valid"] = False
        base["violations"] = [violation]
        return base

    violations: list[str] = []
    training_loops = _gradient_training_loops(tree)
    base["observations"]["active_gradient_training_loops"] = training_loops
    if no_retraining_evidence and training_loops:
        lines = ", ".join(str(row["loop_line"]) for row in training_loops)
        violations.append(
            "plan_runtime_retraining_conflict: roadmap requires reuse of frozen "
            "checkpoints without retraining, but final executable code contains "
            f"gradient-training loop(s) with backward/step at line(s) {lines}"
        )

    required_thresholds = {
        float(row["value"])
        for row in threshold_evidence
        if row.get("value") is not None
    }
    threshold_comparisons = _threshold_comparisons(tree, required_thresholds)
    base["observations"]["absolute_metric_threshold_comparisons"] = (
        threshold_comparisons
    )
    observed_thresholds = {
        float(value)
        for row in threshold_comparisons
        for value in row.get("values") or []
    }
    missing_thresholds = sorted(
        threshold
        for threshold in required_thresholds
        if not any(abs(threshold - seen) <= 1e-12 for seen in observed_thresholds)
    )
    if missing_thresholds:
        violations.append(
            "plan_runtime_absolute_metric_threshold_missing: roadmap requires "
            "selection to enforce absolute metric threshold(s) "
            f"{missing_thresholds}, but final executable code does not compare "
            "against them"
        )

    base["valid"] = not violations
    base["violations"] = violations
    return base


__all__ = ["verify_host_plan_execution"]
