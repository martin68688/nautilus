from __future__ import annotations

import hashlib
import json
import re
from typing import Any


METHOD_CLAIM_PURITY_SCHEMA = "method_claim_semantic_purity_v1"

_NUMBER = r"(?:[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?|[-+]?\.\d+)%?"
_METRIC = (
    r"(?:validation\s+|val(?:idation)?[-_ ]?)?"
    r"(?:roc[-_ ]?auc|pr[-_ ]?auc|auc|macro[-_ ]?f1|micro[-_ ]?f1|"
    r"f1(?:[-_ ]?score)?|log[-_ ]?loss|cross[-_ ]?entropy(?:\s+loss)?|"
    r"loss|accuracy|(?<!mixed\s)(?<!mixed-)precision|recall|rmse|mse|mae|r2|r\^2|r²|"
    r"perplexity|bleu|rouge(?:[-_ ]?[12l])?|wer|cer|ndcg|map|"
    r"dice|iou|score|metric)"
)
_PAST_RESULT_VERB = (
    r"(?:achieved|achieving|attained|attaining|reached|reaching|"
    r"obtained|obtaining|yielded|yielding|reported|recorded|scored|"
    r"improved|decreased|dropped|fell|rose|increased|peaked)"
)

_METRIC_TERM_PATTERN = re.compile(rf"(?i)\b{_METRIC}\b")
_VIOLATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "metric_value_assertion",
        re.compile(
            rf"(?i)\b{_METRIC}\b\s*"
            rf"(?:of|was|is|were|are|=|:|@|at|to|from|below|above|under|over|"
            rf"less\s+than|greater\s+than|reached|achieved|attained|obtained|"
            rf"yielded|reported|recorded|improved\s+(?:from|to|by)|"
            rf"decreased\s+(?:from|to|by)|dropped\s+(?:from|to|by))?\s*"
            rf"{_NUMBER}\b"
        ),
    ),
    (
        "numeric_metric_assertion",
        re.compile(rf"(?i)(?<![\w.]){_NUMBER}\s+{_METRIC}\b"),
    ),
    (
        "metric_outcome_assertion",
        re.compile(
            rf"(?i)(?:\b{_PAST_RESULT_VERB}\b[^.;!?\n]{{0,96}}\b{_METRIC}\b|"
            rf"\b{_METRIC}\b[^.;!?\n]{{0,64}}\b{_PAST_RESULT_VERB}\b)"
        ),
    ),
    (
        "untyped_outcome_value_assertion",
        re.compile(
            rf"(?i)(?:"
            rf"\b(?:result|outcome|performance)\b\s*"
            rf"(?:of|was|is|were|are|=|:|at|reached)?\s*{_NUMBER}\b|"
            rf"\b(?:achieved|achieving|attained|attaining|reported|recorded|"
            rf"scored)\b[^.;!?\n]{{0,64}}(?<![\w.]){_NUMBER}\b|"
            rf"(?<![\w.]){_NUMBER}\s+(?:was|were)\s+"
            rf"(?:achieved|attained|reported|recorded)\b"
            rf")"
        ),
    ),
    (
        "comparative_performance_assertion",
        re.compile(
            r"(?i)\b(?:outperformed|underperformed|beat(?:s|ing)?\s+(?:the\s+)?"
            r"baseline|state[- ]of[- ]the[- ]art|best\s+(?:validation\s+)?"
            r"(?:performance|result|score|metric)|worst\s+(?:validation\s+)?"
            r"(?:performance|result|score|metric)|superior\s+performance)\b"
        ),
    ),
    (
        "redacted_outcome_placeholder",
        re.compile(
            r"(?i)\[(?:historical\s+)?(?:score|metric|outcome|result)\s+redacted\]"
        ),
    ),
    (
        "runtime_outcome_assertion",
        re.compile(
            r"(?i)\b(?:"
            r"(?:training|code|pipeline|run|model)\s+(?:ran|executed|completed|"
            r"finished)\s+successfully|"
            r"(?:without|with\s+no)\s+(?:runtime\s+)?errors?|"
            r"early\s+stopping\s+(?:was\s+)?triggered|"
            r"trained\s+for\s+\d+\s+epochs?|"
            r"(?:training|model)\s+ran\s+for\s+\d+\s+epochs?|"
            r"completed\s+\d+\s+epochs?"
            r")\b"
        ),
    ),
)


class MethodClaimSemanticPurityError(ValueError):
    """Raised when a reusable Method Claim contains source-run outcomes."""


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_method_claim_text(raw: str) -> str:
    return re.sub(r"\s+", " ", str(raw or "")).strip()


def audit_method_claim_semantic_purity(raw: str) -> dict[str, Any]:
    """Return a content-addressed, outcome-free audit report.

    The report intentionally stores only the statement hash and violation
    categories.  A rejected source score must not be copied into a new Bundle
    merely because it appeared in a validator report.
    """

    text = normalize_method_claim_text(raw)
    violation_codes = sorted(
        {
            code
            for code, pattern in _VIOLATION_PATTERNS
            if pattern.search(text)
        }
    )
    metric_terms = sorted(
        {
            re.sub(r"\s+", " ", match.group(0).lower()).strip()
            for match in _METRIC_TERM_PATTERN.finditer(text)
        }
    )
    report: dict[str, Any] = {
        "schema": METHOD_CLAIM_PURITY_SCHEMA,
        "passed": bool(text) and not violation_codes,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_character_count": len(text),
        "metric_terms_present": metric_terms,
        "violation_codes": violation_codes or (["empty_method_claim"] if not text else []),
        "source_outcome_assertion_count": len(violation_codes) + int(not text),
        "raw_text_embedded": False,
        "report_hash": "",
    }
    report["report_hash"] = _stable_hash(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


def require_method_claim_semantic_purity(raw: str) -> dict[str, Any]:
    report = audit_method_claim_semantic_purity(raw)
    if not report["passed"]:
        reasons = ",".join(report["violation_codes"])
        raise MethodClaimSemanticPurityError(
            f"method_claim_semantic_purity_failed:{reasons}"
        )
    return report
