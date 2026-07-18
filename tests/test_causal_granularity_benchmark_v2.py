from __future__ import annotations

import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "paper-skills" / "eval_composite_memory"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
if str(REPO / "mlevolve") not in sys.path:
    sys.path.insert(0, str(REPO / "mlevolve"))

from core import GRAPH, INDEX, read_json, read_jsonl  # noqa: E402
from run_causal_granularity_benchmark_v2 import (  # noqa: E402
    GOLD_PATH,
    MANIFEST_PATH,
    QUERY_PATH,
    REPORT_PATH,
    _layer,
    build,
    evaluate,
)


def _ensure_artifacts() -> None:
    if not QUERY_PATH.exists() or not GOLD_PATH.exists():
        build(GRAPH, INDEX)
    if not REPORT_PATH.exists():
        evaluate(GRAPH, INDEX)


def test_v2_uses_real_clean_causal_transitions_and_cross_run_gold() -> None:
    build(GRAPH, INDEX)
    layer = _layer(GRAPH, INDEX)
    queries = {row["episode_id"]: row for row in read_jsonl(QUERY_PATH)}
    gold_rows = read_jsonl(GOLD_PATH)
    assert len(queries) >= 30
    assert any(row["expected_route"] == "causal_tree" for row in gold_rows)
    assert any(row["expected_route"] == "sop_only_fallback" for row in gold_rows)
    for gold in gold_rows:
        query = queries[gold["episode_id"]]
        assert gold["source_transition_id"] not in query["candidate_transition_ids"]
        assert gold["gold_is_cross_run"] is True
        assert gold["gold_is_clean_causal_transition"] is True
        for transition_id in gold["gold_transition_ids"]:
            eligible, _reason = layer._positive_transition(transition_id)
            assert eligible is True
            transition = layer.nodes[transition_id]
            assert transition["outcome"] == "debug_fixed"
            assert transition["parent_buggy"] is True
            assert transition["child_buggy"] is False


def test_v2_manifest_freezes_snapshot_and_nonblind_limitations() -> None:
    _ensure_artifacts()
    manifest = read_json(MANIFEST_PATH)
    assert manifest["episode_count"] >= 30
    assert manifest["covered_episode_count"] > 0
    assert manifest["coverage_gap_episode_count"] > 0
    assert manifest["retrospective"] is True
    assert manifest["blind_test"] is False


def test_v2_dynamic_debug_enforces_source_run_and_safety_conformance() -> None:
    report = evaluate(GRAPH, INDEX)
    method = report["tracks"]["causal_debug_transfer"]["causal_tree_dynamic"]
    assert method["unsafe_count"] == 0
    assert method["source_run_escape_count"] == 0
    assert 0.0 <= method["route_accuracy"] <= 1.0
    assert 0.0 <= method["transition_mrr"] <= 1.0


def test_v2_reports_granularity_and_refuses_downstream_claim() -> None:
    _ensure_artifacts()
    report = read_json(REPORT_PATH)
    assert set(report["tracks"]) == {"stage_granularity", "causal_debug_transfer"}
    assert report["claims"]["granularity_gate_diagnostic_allowed"] is True
    assert report["claims"]["cross_run_causal_retrieval_diagnostic_allowed"] is True
    assert report["claims"]["blind_generalization_claim_allowed"] is False
    assert report["claims"]["downstream_agent_success_claim_allowed"] is False


def test_v2_dynamic_method_beats_legacy_on_combined_test_decision() -> None:
    _ensure_artifacts()
    report = read_json(REPORT_PATH)
    granularity = report["tracks"]["stage_granularity"]
    assert granularity["stage_hybrid_dynamic"]["granularity_precision_at_5"] == 1.0
    assert granularity["stage_hybrid_dynamic"]["detail_intrusion_at_5"] == 0.0
    debug = report["tracks"]["causal_debug_transfer"]
    dynamic = debug["causal_tree_dynamic"]["by_split"]["test"]
    legacy = debug["legacy_success_tree"]["by_split"]["test"]
    random = debug["random_transition"]["by_split"]["test"]
    assert dynamic["selective_decision_accuracy_at_1"] > legacy["selective_decision_accuracy_at_1"]
    assert dynamic["transition_mrr"] > legacy["transition_mrr"]
    assert dynamic["transition_mrr"] > random["transition_mrr"]
