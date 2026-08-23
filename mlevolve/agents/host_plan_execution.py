"""Host-owned verification that final code still implements Planner intent.

Patch-shape verification cannot prove runtime semantics.  In particular, a
reviewed candidate can be valid Python and remain within its patch budget while
silently retaining an old training loop that contradicts an explicit roadmap
to reuse frozen checkpoints without retraining.  This module checks that narrow
invariant on the final executable source, after all reviewer edits.
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
    r"\bcheckpoints?\b[^.\n]{0,80}\b(?:remain|keep|kept)\w*\s+frozen\b",
    re.IGNORECASE,
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


def verify_host_plan_execution(
    plan: Mapping[str, Any] | None,
    code: str,
) -> dict[str, Any]:
    """Verify explicit frozen-checkpoint/no-retraining roadmap semantics.

    This intentionally does not infer a no-retraining requirement from generic
    preservation language such as "do not alter the training loop" or
    "unchanged checkpoint selection remains executable".  The gate activates
    only when the Planner explicitly promises no retraining or reuse of
    existing frozen checkpoints.
    """

    plan_object = dict(plan or {})
    requirement_evidence = _no_retraining_evidence(plan_object)
    base: dict[str, Any] = {
        "schema": "mlevolve_host_plan_execution_gate_v1",
        "enabled": bool(requirement_evidence),
        "requirements": (
            ["frozen_checkpoint_reuse_without_retraining"]
            if requirement_evidence
            else []
        ),
        "requirement_evidence": requirement_evidence,
        "observations": {"active_gradient_training_loops": []},
        "valid": True,
        "violations": [],
    }
    if not requirement_evidence:
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

    training_loops = _gradient_training_loops(tree)
    base["observations"]["active_gradient_training_loops"] = training_loops
    if training_loops:
        lines = ", ".join(str(row["loop_line"]) for row in training_loops)
        base["valid"] = False
        base["violations"] = [
            "plan_runtime_retraining_conflict: roadmap requires reuse of frozen "
            "checkpoints without retraining, but final executable code contains "
            f"gradient-training loop(s) with backward/step at line(s) {lines}"
        ]
    return base


__all__ = ["verify_host_plan_execution"]
