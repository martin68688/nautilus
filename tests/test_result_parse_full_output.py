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


def test_missing_marker_with_single_metric_is_nonblocking():
    from agents.result_log_facts import (
        reconcile_missing_submission_alignment,
        result_parser_facts,
    )
    from engine.search_node import SearchNode

    output = (
        "=== OOF Log Loss: 0.043141 ===\n"
        "Submission saved: (99, 100)\n"
        "Final Validation Score: 0.043140515499590075\n"
    )
    node = SearchNode(
        code="print('run')", stage="improve", _term_out=[output], exc_type=None
    )
    facts = result_parser_facts(node, True, output)
    metric, status = reconcile_missing_submission_alignment(
        facts, 0.043140515499590075
    )
    assert metric == 0.043140515499590075
    assert status == "inferred_single_metric"
    assert facts["high_confidence_metric_ambiguous"] is False


def test_missing_marker_with_multiple_metrics_uses_agent_reconciliation():
    from agents.result_log_facts import (
        reconcile_missing_submission_alignment,
        result_parser_facts,
    )
    from engine.search_node import SearchNode

    output = (
        "Final OOF Log Loss: 0.005086\n"
        "Blend OOF Log Loss: 0.010162\n"
        "Submission saved: (594, 100)\n"
    )
    node = SearchNode(
        code="print('run')", stage="improve", _term_out=[output], exc_type=None
    )
    facts = result_parser_facts(node, True, output)
    assert facts["high_confidence_metric_ambiguous"] is True
    assert [
        candidate["metric"]
        for candidate in facts["high_confidence_metric_candidates"]
    ] == [0.005086, 0.010162]
    metric, status = reconcile_missing_submission_alignment(facts, 0.010162)
    assert metric == 0.010162
    assert status == "agent_reconciled_multiple_metrics"


def test_modified_replay_without_submission_aligned_marker_blocks_ranking():
    from agents.result_log_facts import (
        is_immutable_exact_replay,
        modified_replay_alignment_is_blocking,
    )
    from engine.search_node import SearchNode

    source_code = "print('exact source')\n"
    source_hash = __import__("hashlib").sha256(source_code.encode()).hexdigest()
    inherited_source = {
        "code_sha256": source_hash,
        "current_code_sha256": __import__("hashlib").sha256(
            b"print('modified blend')\n"
        ).hexdigest(),
        "exact_replay_execution": True,
        "exact_source_match": False,
    }
    modified = SearchNode(
        code="print('modified blend')\n",
        stage="improve",
        draft_role="novel_exploration",
        replay_source=inherited_source,
        replay_status="replay_derived_novel_candidate",
    )

    assert is_immutable_exact_replay(modified) is False
    assert modified_replay_alignment_is_blocking(
        modified,
        submission_alignment_required=True,
        aligned_metric=None,
    ) is True
    assert modified_replay_alignment_is_blocking(
        modified,
        submission_alignment_required=True,
        aligned_metric=0.011,
    ) is False


def test_replay_alignment_repair_guidance_preserves_exact_variant_and_marker():
    from agents.debug_agent import replay_alignment_repair_guidance
    from engine.search_node import SearchNode

    node = SearchNode(
        code="print('adapted')\n",
        stage="improve",
        draft_role="novel_exploration",
        replay_source={"code_sha256": "a" * 64},
        protocol_observation={
            "submission_metric_alignment": {
                "blocking": True,
                "reexecution_required": True,
            }
        },
    )

    guidance = "\n".join(replay_alignment_repair_guidance(node))
    assert "not a model redesign" in guidance
    assert "Preserve the parent model" in guidance
    assert "Final Submission-Aligned Validation Score" in guidance
    assert "| variant=" in guidance


def test_immutable_exact_replay_keeps_missing_marker_exemption():
    from agents.result_log_facts import (
        is_immutable_exact_replay,
        modified_replay_alignment_is_blocking,
    )
    from engine.search_node import SearchNode

    source_code = "print('exact source')\n"
    source_hash = __import__("hashlib").sha256(source_code.encode()).hexdigest()
    exact = SearchNode(
        code=source_code,
        stage="improve",
        draft_role="memory_reproduction",
        replay_source={
            "code_sha256": source_hash,
            "current_code_sha256": source_hash,
            "exact_replay_execution": True,
            "exact_source_match": True,
        },
        replay_status="historical_exact_research_loaded",
    )

    assert is_immutable_exact_replay(exact) is True
    assert modified_replay_alignment_is_blocking(
        exact,
        submission_alignment_required=True,
        aligned_metric=None,
    ) is False
