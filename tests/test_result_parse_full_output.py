import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))


def test_parser_preserves_metric_and_submission_signals_from_log_middle():
    from agents.result_log_facts import (
        extract_high_confidence_metric,
        result_parser_conflict,
        result_parser_facts,
        result_parser_output_view,
    )
    from engine.search_node import SearchNode

    output = (
        "training started\n"
        + ("epoch noise without useful signal\n" * 500)
        + "Final OOF Log Loss: 0.098610\n"
        + "Submission saved: (99, 100)\n"
        + ("FutureWarning from a dependency\n" * 500)
    )
    node = SearchNode(
        code="print('run')",
        stage="draft",
        _term_out=[output],
        exc_type=None,
    )
    assert "Final OOF Log Loss" not in node.term_out
    assert "Final OOF Log Loss" in node.full_term_out

    view = result_parser_output_view(node)
    assert len(view) < len(output)
    assert "Final OOF Log Loss: 0.098610" in view
    assert "Submission saved: (99, 100)" in view
    metric, line = extract_high_confidence_metric(node.full_term_out)
    assert metric == 0.098610
    assert line == "Final OOF Log Loss: 0.098610"

    facts = result_parser_facts(node, True, view)
    assert facts["process_exited_normally"] is True
    assert facts["submission_file_exists"] is True
    assert facts["high_confidence_self_reported_metric"] == 0.098610
    conflict = result_parser_conflict(
        {
            "is_bug": True,
            "metric": None,
            "summary": "The run was truncated before completion.",
        },
        facts,
    )
    assert conflict == "metric_missing_despite_full_log_metric_and_submission"


def test_parser_does_not_invent_metric_from_ordinary_epoch_lines():
    from agents.result_log_facts import extract_high_confidence_metric

    metric, line = extract_high_confidence_metric(
        "Epoch 1 | Validation log loss: 0.8\nEpoch 2 | Validation log loss: 0.7\n"
    )
    assert metric is None
    assert line == ""


def test_submission_aligned_metric_wins_over_better_component_metric():
    from agents.result_log_facts import (
        extract_submission_aligned_metric,
        result_parser_facts,
    )
    from engine.search_node import SearchNode

    output = (
        "Final OOF Log Loss: 0.005086\n"
        "Blend OOF Log Loss: 0.010162\n"
        "Submission saved: (594, 100)\n"
        "Final Submission-Aligned Validation Score: 0.010162 | variant=nn_lgbm_blend\n"
    )
    node = SearchNode(
        code="print('run')", stage="improve", _term_out=[output], exc_type=None
    )
    metric, variant, line = extract_submission_aligned_metric(output)
    assert metric == 0.010162
    assert variant == "nn_lgbm_blend"
    assert line.endswith("variant=nn_lgbm_blend")
    facts = result_parser_facts(node, True, output)
    assert facts["submission_aligned_metric"] == 0.010162
    assert facts["submission_variant"] == "nn_lgbm_blend"
