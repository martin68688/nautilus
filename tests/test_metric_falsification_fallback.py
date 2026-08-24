from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.code_review_agent import (  # noqa: E402
    _deterministic_contract_audit,
    _metric_falsification_fallback_audit,
)


def test_metric_improvement_assert_is_rejected() -> None:
    report = _metric_falsification_fallback_audit(
        "assert final_blend_logloss < final_oof_logloss, 'blend did not improve'\n"
    )

    assert report["valid"] is False
    assert report["violations"] == [
        "metric_improvement_assert:line=1: empirical non-improvement must "
        "select baseline predictions and continue to submission"
    ]


def test_metric_gated_raise_is_rejected() -> None:
    report = _metric_falsification_fallback_audit(
        "if candidate_score >= baseline_score:\n"
        "    raise RuntimeError('candidate did not improve')\n"
    )

    assert report["valid"] is False
    assert report["violations"][0].startswith("metric_gated_termination:line=1:")


def test_structural_assertions_and_metric_fallback_are_allowed() -> None:
    code = """
assert predictions.shape == expected_shape
assert np.isfinite(predictions).all()
if candidate_logloss < baseline_logloss:
    final_predictions = candidate_predictions
    final_score = candidate_logloss
else:
    final_predictions = baseline_predictions
    final_score = baseline_logloss
write_submission(final_predictions)
"""
    report = _metric_falsification_fallback_audit(code)

    assert report["valid"] is True
    assert report["violations"] == []


def test_checkpoint_integrity_raise_inside_best_score_branch_is_allowed() -> None:
    code = """
if epoch_score > best_score:
    save_checkpoint(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint persistence failed")
else:
    stale_epochs += 1
    if stale_epochs >= patience:
        break
"""
    report = _metric_falsification_fallback_audit(code)

    assert report["valid"] is True
    assert report["violations"] == []


def test_host_audit_includes_exact_source_excerpt_for_repair() -> None:
    agent = SimpleNamespace(
        cfg=SimpleNamespace(agent=SimpleNamespace()),
        acfg=SimpleNamespace(protocol_preflight=SimpleNamespace(enabled=False)),
    )
    audit = _deterministic_contract_audit(
        agent,
        "candidate_score = 0.4\n"
        "baseline_score = 0.5\n"
        "if candidate_score >= baseline_score:\n"
        "    raise RuntimeError('candidate did not improve')\n",
    )

    assert audit is not None
    assert audit["source_excerpts"][0]["line"] == 3
    assert "3: if candidate_score >= baseline_score:" in audit[
        "source_excerpts"
    ][0]["source"]


def test_code_review_host_audit_runs_without_optional_protocol_contract() -> None:
    agent = SimpleNamespace(
        cfg=SimpleNamespace(agent=SimpleNamespace()),
        acfg=SimpleNamespace(protocol_preflight=SimpleNamespace(enabled=False)),
    )
    audit = _deterministic_contract_audit(
        agent,
        "assert proposed_auc > baseline_auc\n",
    )

    assert audit is not None
    assert audit["valid"] is False
    assert audit["metric_falsification_fallback_audit"]["valid"] is False
