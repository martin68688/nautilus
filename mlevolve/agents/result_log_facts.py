"""Objective completion and metric signals from unabridged executor output."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
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

_NAMED_OOF_METRIC_RE = re.compile(
    r"(?im)^\s*(?P<label>[^\r\n:=]{0,80}?)\b"
    r"(?:oof|out[- ]of[- ]fold)\s+"
    r"(?:validation\s+)?(?:log\s*loss|roc[- ]?auc|auc|rmse|mae|mse|"
    r"score|metric)\s*[:=]\s*"
    r"(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\b"
)

_SUBMISSION_ALIGNED_METRIC_RE = re.compile(
    r"(?im)^\s*Final\s+Submission[- ]Aligned\s+Validation\s+Score\s*:\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\|\s*"
    r"variant\s*=\s*([^\r\n|]+?)\s*$"
)

_IMMUTABLE_EXACT_REPLAY_STATUSES = frozenset(
    {
        "exact_source_loaded",
        "exact_source_loaded_fixed_holdout",
        "historical_exact_anchor_loaded",
        "historical_exact_research_loaded",
    }
)


def is_immutable_exact_replay(node: Any) -> bool:
    """Return whether the current artifact is still the byte-exact replay seed."""

    replay_source = getattr(node, "replay_source", None)
    if (
        getattr(node, "draft_role", None) != "memory_reproduction"
        or not isinstance(replay_source, dict)
        or not replay_source
        or getattr(node, "replay_status", None)
        not in _IMMUTABLE_EXACT_REPLAY_STATUSES
    ):
        return False
    if replay_source.get("exact_source_match") is False:
        return False
    source_hash = str(
        replay_source.get("code_sha256")
        or replay_source.get("source_code_sha256")
        or ""
    )
    current_hash = str(replay_source.get("current_code_sha256") or "")
    code = str(getattr(node, "code", None) or "")
    if not current_hash and code:
        current_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    return not (source_hash and current_hash and source_hash != current_hash)


def modified_replay_alignment_is_blocking(
    node: Any,
    *,
    submission_alignment_required: bool,
    aligned_metric: object,
) -> bool:
    """Fail closed when a modified replay ranks a different submission variant."""

    return bool(
        submission_alignment_required
        and aligned_metric is None
        and getattr(node, "draft_role", None) == "memory_reproduction"
        and getattr(node, "replay_source", None)
        and not is_immutable_exact_replay(node)
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


def extract_high_confidence_metric_candidates(
    output: str,
) -> list[dict[str, object]]:
    """Return every explicit final/OOF metric without guessing from epochs."""

    matches: dict[tuple[int, str], tuple[int, float, str]] = {}
    for pattern in _HIGH_CONFIDENCE_METRIC_PATTERNS:
        for match in pattern.finditer(output or ""):
            try:
                value = float(match.group(1))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                line = match.group(0).strip()
                matches[(match.start(), line)] = (match.start(), value, line)
    for match in _NAMED_OOF_METRIC_RE.finditer(output or ""):
        try:
            value = float(match.group("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            line = match.group(0).strip()
            matches[(match.start(), line)] = (match.start(), value, line)

    ordered = sorted(matches.values(), key=lambda item: item[0])
    return [
        {"position": position, "metric": value, "line": line}
        for position, value, line in ordered
    ]


def extract_high_confidence_metric(output: str) -> tuple[float | None, str]:
    candidates = extract_high_confidence_metric_candidates(output)
    if not candidates:
        return None, ""
    selected = candidates[-1]
    return float(selected["metric"]), str(selected["line"])


def reconcile_missing_submission_alignment(
    facts: Mapping, response_metric: object
) -> tuple[float | None, str]:
    """Resolve missing text metadata without making a clean run retrain."""

    raw_candidates = facts.get("high_confidence_metric_candidates") or []
    values: list[float] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping):
            continue
        value = candidate.get("metric")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            numeric = float(value)
            if not any(
                math.isclose(numeric, existing, rel_tol=1e-9, abs_tol=1e-12)
                for existing in values
            ):
                values.append(numeric)

    if len(values) == 1:
        return values[0], "inferred_single_metric"

    response_value: float | None = None
    if isinstance(response_metric, (int, float)):
        numeric = float(response_metric)
        if math.isfinite(numeric):
            response_value = numeric

    if len(values) > 1:
        if response_value is not None:
            for value in values:
                if math.isclose(
                    response_value, value, rel_tol=1e-6, abs_tol=1e-9
                ):
                    return response_value, "agent_reconciled_multiple_metrics"
        return values[-1], "deterministic_last_metric_fallback"

    if response_value is not None:
        return response_value, "agent_reconciled_metric_only"
    return None, "metric_missing_unresolved"


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
    metric_candidates = extract_high_confidence_metric_candidates(output)
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
        "high_confidence_metric_candidates": metric_candidates,
        "high_confidence_metric_ambiguous": len(
            {
                float(candidate["metric"])
                for candidate in metric_candidates
                if isinstance(candidate.get("metric"), (int, float))
            }
        )
        > 1,
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
