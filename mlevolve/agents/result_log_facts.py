"""Objective completion and metric signals from unabridged executor output."""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any


_RESULT_SIGNAL_RE = re.compile(
    r"(?i)\b(final|best|oof|out[- ]of[- ]fold|validation|submission|saved|"
    r"completed|finished|metric|score|auc|rmse|mae|mse|log[ _-]?loss)\b"
)
_HIGH_CONFIDENCE_METRIC_PATTERNS = (
    re.compile(
        r"(?im)^\s*(?:final\s+)?(?:oof|out[- ]of[- ]fold)\s+"
        r"(?:validation\s+)?(?:log\s*loss|roc[- ]?auc|auc|rmse|mae|mse|"
        r"score|metric)\s*[:=]\s*"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\b"
    ),
    re.compile(
        r"(?im)^\s*(?:final|best)\s+(?:internal\s+)?"
        r"(?:validation|valid|cv)?\s*"
        r"(?:log\s*loss|roc[- ]?auc|auc|rmse|mae|mse|score|metric)"
        r"\s*[:=]\s*"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\b"
    ),
    re.compile(
        r"(?im)^\s*internal\s+validation\s+"
        r"(?:log\s*loss|roc[- ]?auc|auc|rmse|mae|mse|score|metric)"
        r"\s*[:=]\s*"
        r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\b"
    ),
)

_SUBMISSION_ALIGNED_METRIC_RE = re.compile(
    r"(?im)^\s*Final\s+Submission[- ]Aligned\s+Validation\s+Score\s*:\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\|\s*"
    r"variant\s*=\s*([^\r\n|]+?)\s*$"
)


def extract_submission_aligned_metric(
    output: str,
) -> tuple[float | None, str, str]:
    matches: list[tuple[int, float, str, str]] = []
    for match in _SUBMISSION_ALIGNED_METRIC_RE.finditer(output or ""):
        try:
            value = float(match.group(1))
        except (TypeError, ValueError):
            continue
        variant = str(match.group(2) or "").strip()
        if math.isfinite(value) and variant:
            matches.append((match.start(), value, variant, match.group(0).strip()))
    if not matches:
        return None, "", ""
    _position, value, variant, line = max(matches, key=lambda item: item[0])
    return value, variant, line


def extract_high_confidence_metric(output: str) -> tuple[float | None, str]:
    matches: list[tuple[int, float, str]] = []
    for pattern in _HIGH_CONFIDENCE_METRIC_PATTERNS:
        for match in pattern.finditer(output or ""):
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                matches.append((match.start(), value, match.group(0).strip()))
    if not matches:
        return None, ""
    _position, value, line = max(matches, key=lambda item: item[0])
    return value, line


def result_parser_output_view(node: Any, max_chars: int = 16000) -> str:
    """Keep completion/metric lines even when noisy stderr follows stdout."""

    output = str(getattr(node, "full_term_out", "") or "")
    if len(output) <= max_chars:
        return output
    signal_lines = [
        line.rstrip("\n")
        for line in output.splitlines()
        if _RESULT_SIGNAL_RE.search(line)
    ]
    salient = "\n".join(signal_lines[-120:])
    return "\n".join(
        [
            output[:3500],
            "[... non-salient output omitted from the Agent view ...]",
            "[salient completion/metric/submission lines from full output]",
            salient,
            "[tail of full output]",
            output[-5500:],
        ]
    )


def result_parser_facts(
    node: Any, has_csv_submission: bool, parser_view: str
) -> dict[str, object]:
    output = str(getattr(node, "full_term_out", "") or "")
    metric, metric_line = extract_high_confidence_metric(output)
    aligned_metric, submission_variant, aligned_line = (
        extract_submission_aligned_metric(output)
    )
    return {
        "process_exited_normally": getattr(node, "exc_type", None) is None,
        "exception_type": getattr(node, "exc_type", None),
        "submission_file_exists": bool(has_csv_submission),
        "full_output_char_count": len(output),
        "agent_output_view_char_count": len(parser_view),
        "agent_output_view_compacted": len(parser_view) < len(output),
        "high_confidence_self_reported_metric": metric,
        "high_confidence_metric_line": metric_line,
        "submission_aligned_metric": aligned_metric,
        "submission_variant": submission_variant,
        "submission_aligned_metric_line": aligned_line,
    }


def result_parser_conflict(response: Mapping, facts: Mapping) -> str:
    if not facts.get("process_exited_normally"):
        return ""
    if not facts.get("submission_file_exists"):
        return ""
    aligned = facts.get("submission_aligned_metric")
    if aligned is not None and isinstance(response.get("metric"), (int, float)):
        if not math.isclose(
            float(response["metric"]), float(aligned), rel_tol=1e-9, abs_tol=1e-12
        ):
            return "metric_differs_from_submission_aligned_marker"
    if (
        response.get("metric") is None
        and facts.get("high_confidence_self_reported_metric") is not None
    ):
        return "metric_missing_despite_full_log_metric_and_submission"
    summary = str(response.get("summary") or "").lower()
    false_failure_terms = (
        "timeout",
        "timed out",
        "truncated",
        "interrupted",
        "did not complete",
        "failed to finish",
    )
    if response.get("is_bug") and any(term in summary for term in false_failure_terms):
        return "agent_failure_claim_conflicts_with_clean_process_exit"
    return ""
