from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from method_claim_purity import (  # noqa: E402
    MethodClaimSemanticPurityError,
    audit_method_claim_semantic_purity,
    require_method_claim_semantic_purity,
)
from publish_certified_replay_bundle import (  # noqa: E402
    SOURCE_SNAPSHOT_SCHEMA,
    _method_text,
    _publication_method_text,
    _verify_source_snapshot,
    audit_visible_method_clauses,
)


@pytest.mark.parametrize(
    "statement",
    (
        (
            "Use 5-fold stratified cross-validation, train for 28 epochs with "
            "learning rate 1e-4 and dropout 0.2, and select checkpoints by "
            "validation log loss."
        ),
        (
            "Optimize cross-entropy loss with label smoothing 0.1 and monitor "
            "ROC-AUC for early stopping."
        ),
        (
            "Resize images to 300x300, use an 80/20 split, and set patience to 7."
        ),
    ),
)
def test_method_purity_preserves_method_hyperparameters(statement: str) -> None:
    report = require_method_claim_semantic_purity(statement)

    assert report["passed"] is True
    assert report["source_outcome_assertion_count"] == 0
    assert report["violation_codes"] == []
    assert report["raw_text_embedded"] is False


@pytest.mark.parametrize(
    ("statement", "reason"),
    (
        (
            "Training ran for 28 epochs, achieving a final validation log loss of 0.1458.",
            "metric_value_assertion",
        ),
        ("The replay achieved ROC-AUC 0.9968.", "metric_value_assertion"),
        ("Validation accuracy was 93%.", "metric_value_assertion"),
        ("F1 improved from 0.71 to 0.80.", "metric_value_assertion"),
        ("The replay achieved 0.9968.", "untyped_outcome_value_assertion"),
        ("The final result was 0.1458.", "untyped_outcome_value_assertion"),
        ("Use the same architecture [score redacted].", "redacted_outcome_placeholder"),
        ("This method outperformed the baseline.", "comparative_performance_assertion"),
        ("Training completed successfully without errors.", "runtime_outcome_assertion"),
        ("The model trained for 28 epochs before early stopping was triggered.", "runtime_outcome_assertion"),
    ),
)
def test_method_purity_rejects_source_outcome_assertions(
    statement: str,
    reason: str,
) -> None:
    report = audit_method_claim_semantic_purity(statement)

    assert report["passed"] is False
    assert reason in report["violation_codes"]
    with pytest.raises(MethodClaimSemanticPurityError, match=reason):
        require_method_claim_semantic_purity(statement)


def test_rejected_report_does_not_copy_historical_metric_or_text() -> None:
    statement = "The final validation log loss reached 0.1458."
    report = audit_method_claim_semantic_purity(statement)
    encoded = json.dumps(report, sort_keys=True)

    assert report["passed"] is False
    assert "0.1458" not in encoded
    assert statement not in encoded
    assert report["raw_text_embedded"] is False


def test_certified_publication_rejects_predecessor_metric_not_equal_to_replay_metric() -> None:
    with pytest.raises(
        MethodClaimSemanticPurityError,
        match="metric_value_assertion",
    ):
        _method_text(
            "Train with early stopping; final validation log loss was 0.1458.",
            0.1300,
        )


def test_certified_publication_returns_hash_bound_method_only_report() -> None:
    text, report = _method_text(
        "Use EfficientNet-B3 at 300x300 with 28 epochs and early stopping based "
        "on validation log loss.",
        0.1300,
    )

    assert text.startswith("Use EfficientNet-B3")
    assert report["passed"] is True
    assert report["text_sha256"]
    assert report["report_hash"]


def test_replay_metric_equal_to_a_method_hyperparameter_is_not_a_false_positive() -> None:
    text, report = _method_text(
        "Use label smoothing 0.1 and dropout 0.1 during 5-fold training.",
        0.1,
    )

    assert "label smoothing 0.1" in text
    assert report["passed"] is True


def test_mixed_precision_is_not_misclassified_as_a_metric_term() -> None:
    report = require_method_claim_semantic_purity(
        "Use mixed precision training and mixed-precision inference."
    )

    assert report["metric_terms_present"] == []


def test_default_publication_uses_method_only_retrieval_projection() -> None:
    text, report, source = _publication_method_text(
        explicit_statement=None,
        source_clause={
            "text": "The model achieved validation log loss of 0.1458.",
            "retrieval_text": (
                "Use a ResNet-18 image encoder, 5-fold cross-validation, label "
                "smoothing, mixed precision, and early stopping."
            ),
        },
        metric_value=0.1300,
    )

    assert "0.1458" not in text
    assert source == "source_clause.retrieval_text"
    assert report["passed"] is True


def test_default_publication_fails_closed_without_retrieval_projection() -> None:
    with pytest.raises(ValueError, match="lacks a method-only retrieval projection"):
        _publication_method_text(
            explicit_statement=None,
            source_clause={"text": "A source outcome only."},
            metric_value=0.1300,
        )


def test_full_visible_pack_purity_audit_checks_display_and_retrieval_text() -> None:
    report = audit_visible_method_clauses(
        {
            "clause::clean": {
                "text": "Use 5-fold training with label smoothing 0.1.",
                "retrieval_text": "Use 5-fold training with label smoothing 0.1.",
            }
        },
        {"clause::clean"},
    )

    assert report["status"] == "passed"
    assert report["visible_clause_count"] == 1
    assert report["checked_field_count"] == 2
    assert report["violation_count"] == 0
    assert report["raw_text_embedded"] is False


def test_full_visible_pack_purity_audit_finds_non_primary_clause_leak() -> None:
    report = audit_visible_method_clauses(
        {
            "clause::primary": {
                "text": "Use a ResNet-18 image encoder.",
                "retrieval_text": "Use a ResNet-18 image encoder.",
            },
            "clause::other": {
                "text": "This source run achieved AUC of 0.98.",
                "retrieval_text": "Use ConvNeXt-Tiny [score redacted].",
            },
        },
        {"clause::primary", "clause::other"},
    )
    encoded = json.dumps(report, sort_keys=True)

    assert report["status"] == "failed"
    assert report["violation_count"] == 2
    assert {row["field"] for row in report["violations"]} == {
        "text",
        "retrieval_text",
    }
    assert "0.98" not in encoded
    assert "score redacted" not in encoded


def test_source_snapshot_binds_every_executed_source_file(tmp_path: Path) -> None:
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    payload = {
        "schema": SOURCE_SNAPSHOT_SCHEMA,
        "base_commit": "a" * 40,
        "overlay_paths": ["module.py"],
        "file_count": 1,
        "file_hashes": {
            "module.py": hashlib.sha256(source.read_bytes()).hexdigest()
        },
    }
    payload["source_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest = tmp_path / "WP7_SOURCE_MANIFEST.json"
    manifest.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    verified = _verify_source_snapshot(manifest, expected_root=tmp_path)
    assert verified["source_sha256"] == payload["source_sha256"]

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source snapshot file changed"):
        _verify_source_snapshot(manifest, expected_root=tmp_path)
