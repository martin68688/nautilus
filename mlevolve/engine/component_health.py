"""Component-level health checks for candidates that completed successfully.

A node is marked buggy only when the whole program fails.  Candidates often
wrap one component (an extra model, a custom loss) in a fallback, so that
component can diverge, collapse or be silently dropped while the node still
passes.  This module reads the execution log for such task-agnostic
symptoms, records them on the node, and lets the search schedule one targeted
repair child.  The repair child is judged by whether the symptoms disappear,
not by whether the overall score improves immediately.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Iterable

logger = logging.getLogger("MLEvolve")

_NAME = re.compile(r"^\s*([A-Za-z][\w.\-]*)")
_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_LOSS_BAD = re.compile(r"\bloss\b\s*[=:]\s*(nan|-?inf)\b", re.IGNORECASE)
_LOSS_VALUE = re.compile(r"\bloss\b\s*[=:]\s*" + _NUM, re.IGNORECASE)
_EPOCH = re.compile(r"\bepoch\b", re.IGNORECASE)
_SCORE = re.compile(
    r"\b(val[_ ]?corr|val[_ ]?score|validation[_ ]?(?:corr|score)|best[_ ]?score)\s*[=:]\s*" + _NUM,
    re.IGNORECASE,
)
_FALLBACK = re.compile(
    r"(training failed|failed to train|falling back|fall back to|skipping (?:model|component))",
    re.IGNORECASE,
)
# Aggregates and post-processing steps are judged by the node metric itself.
_AGGREGATE_NAMES = {
    "final", "best", "ensemble", "blend", "blended", "neutralize", "neutralization",
    "validation", "overall", "total", "refined", "coordinate", "stage", "info",
    "warning", "error", "traceback",
}


def _component(line: str) -> str | None:
    match = _NAME.match(line)
    if not match:
        return None
    name = match.group(1).rstrip(".:")
    return None if name.lower() in _AGGREGATE_NAMES else name


def _issue(component: str, kind: str, detail: str, evidence: Iterable[str]) -> dict:
    return {
        "component": component,
        "kind": kind,
        "detail": detail,
        "evidence": [line.strip()[:200] for line in list(evidence)[:3]],
    }


def detect_component_issues(term_out: str, *, trivial_score: float | None = None) -> list[dict]:
    """Return component-level failure symptoms found in an execution log."""
    lines = [line for line in str(term_out or "").splitlines() if line.strip()]
    issues: dict[tuple[str, str], dict] = {}
    losses: dict[str, list[tuple[float, str]]] = {}
    scores: dict[str, tuple[float, str]] = {}

    for line in lines:
        if line.lstrip().lower().startswith("final validation score"):
            continue
        name = _component(line)
        if name is None:
            continue
        if _LOSS_BAD.search(line):
            issues.setdefault((name, "non_finite_loss"), _issue(
                name, "non_finite_loss", "training loss is NaN or infinite", [line]))
        elif _EPOCH.search(line):
            value = _LOSS_VALUE.search(line)
            if value:
                losses.setdefault(name, []).append((float(value.group(1)), line))
        if _FALLBACK.search(line):
            issues.setdefault((name, "silent_fallback"), _issue(
                name, "silent_fallback", "component failed and the program fell back without it", [line]))
        score = _SCORE.search(line)
        if score:
            scores[name] = (float(score.group(2)), line)

    for name, values in losses.items():
        if len(values) < 3 or (name, "non_finite_loss") in issues:
            continue
        numbers = [v for v, _ in values]
        spread = max(numbers) - min(numbers)
        if spread <= 1e-6 + 1e-6 * max(1.0, abs(sum(numbers) / len(numbers))):
            issues[(name, "constant_loss")] = _issue(
                name, "constant_loss",
                f"training loss did not change over {len(numbers)} epochs ({numbers[0]:g})",
                [line for _, line in values])

    if trivial_score is not None:
        for name, (value, line) in scores.items():
            if math.isfinite(value) and value < trivial_score:
                issues[(name, "below_trivial_score")] = _issue(
                    name, "below_trivial_score",
                    f"component validation score {value:g} is below the trivial level {trivial_score:g}",
                    [line])
    return list(issues.values())


def _cfg(agent: Any):
    return getattr(getattr(agent, "cfg", None), "component_health", None)


def enabled(agent: Any) -> bool:
    return bool(getattr(_cfg(agent), "enabled", False))


def record(agent: Any, node: Any) -> None:
    """Attach detected issues to a successful node and judge repair children."""
    if not enabled(agent) or node.is_buggy:
        if node.component_repair_target and node.is_buggy:
            node.component_repair_status = "failed"
        return
    node.component_issues = detect_component_issues(
        node.term_out, trivial_score=getattr(_cfg(agent), "trivial_score", None)
    )
    if node.component_issues:
        logger.warning(
            "Node %s completed but has component issues: %s",
            node.id,
            ", ".join(f"{i['component']}:{i['kind']}" for i in node.component_issues),
        )
    target = node.component_repair_target or {}
    if target:
        wanted = {(i["component"], i["kind"]) for i in target.get("issues", [])}
        remaining = {(i["component"], i["kind"]) for i in node.component_issues}
        node.component_repair_status = "unresolved" if wanted & remaining else "resolved"
        logger.info("Component repair node %s: %s", node.id, node.component_repair_status)


def _repair_counts(agent: Any, branch_id: Any) -> tuple[int, int]:
    total = branch = 0
    for node in getattr(agent, "journal", []) or []:
        if getattr(node, "component_repair_target", None):
            total += 1
            branch += int(node.branch_id == branch_id)
    return total, branch


def needs_repair(agent: Any, node: Any) -> bool:
    """True when a successful node has unrepaired issues and budget remains."""
    if not enabled(agent) or node.is_buggy is not False or not node.component_issues:
        return False
    if node.component_repair_status in {"dispatched", "unresolved", "failed"}:
        return False
    cfg = _cfg(agent)
    total, branch = _repair_counts(agent, node.branch_id)
    return (
        total < int(getattr(cfg, "max_repairs_per_run", 6))
        and branch < int(getattr(cfg, "max_repairs_per_branch", 2))
    )


def repair_instructions(issues: list[dict]) -> list[str]:
    """Task-agnostic repair brief: what the Host observed, not how to fix it."""
    lines = [
        "The previous program ran end-to-end and produced a valid submission, but the Host "
        "detected that one or more components failed inside it:",
    ]
    for issue in issues:
        lines.append(f"- `{issue['component']}`: {issue['detail']}")
        lines.extend(f"    log: {line}" for line in issue.get("evidence", []))
    lines += [
        "Repair the failing component so it trains and contributes as intended. Find the root "
        "cause in its own code (data, scaling, loss, optimisation) instead of removing it, "
        "wrapping it in another fallback, or changing unrelated components.",
        "Keep every working component, the validation split and the submission format unchanged.",
        "Print the repaired component's per-epoch loss and validation score so the Host can "
        "verify the repair.",
    ]
    return lines
