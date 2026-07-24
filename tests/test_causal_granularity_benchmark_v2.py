from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "paper-skills" / "eval_composite_memory"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
if str(REPO / "mlevolve") not in sys.path:
    sys.path.insert(0, str(REPO / "mlevolve"))

from core import GRAPH, INDEX, read_json, read_jsonl  # noqa: E402
import run_causal_granularity_benchmark_v2 as benchmark  # noqa: E402


@pytest.fixture(scope="module")
def artifact_paths(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Path]]:
    """Keep benchmark regeneration out of the source and user-asset trees."""

    tmp_path = tmp_path_factory.mktemp("causal-granularity-v2")
    paths = {
        "query": tmp_path / "episodes" / "causal_debug_leave_one_run_out_v2.jsonl",
        "gold": tmp_path / "episodes" / "causal_debug_leave_one_run_out_gold_v2.jsonl",
        "receipt": tmp_path / "reports" / "causal_granularity_receipts_v2.jsonl",
        "report": tmp_path / "reports" / "causal_granularity_report_v2.json",
        "manifest": tmp_path / "manifests" / "causal_granularity_manifest_v2.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    attribute_paths = {
        "QUERY_PATH": paths["query"],
        "GOLD_PATH": paths["gold"],
        "RECEIPT_PATH": paths["receipt"],
        "REPORT_PATH": paths["report"],
        "MANIFEST_PATH": paths["manifest"],
    }
    previous = {name: getattr(benchmark, name) for name in attribute_paths}
    for name, path in attribute_paths.items():
        setattr(benchmark, name, path)
    try:
        benchmark.build(GRAPH, INDEX)
        benchmark.evaluate(GRAPH, INDEX)
        yield paths
    finally:
        for name, path in previous.items():
            setattr(benchmark, name, path)


def _ensure_artifacts(paths: dict[str, Path]) -> None:
    assert paths["query"].exists()
    assert paths["gold"].exists()
    assert paths["manifest"].exists()
    assert paths["receipt"].exists()
    assert paths["report"].exists()


def test_v2_uses_real_clean_causal_transitions_and_cross_run_gold(
    artifact_paths: dict[str, Path],
) -> None:
    _ensure_artifacts(artifact_paths)
    layer = benchmark._layer(GRAPH, INDEX)
    queries = {
        row["episode_id"]: row for row in read_jsonl(artifact_paths["query"])
    }
    gold_rows = read_jsonl(artifact_paths["gold"])
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


def test_v2_manifest_freezes_snapshot_and_nonblind_limitations(
    artifact_paths: dict[str, Path],
) -> None:
    _ensure_artifacts(artifact_paths)
    manifest = read_json(artifact_paths["manifest"])
    assert manifest["episode_count"] >= 30
    assert manifest["covered_episode_count"] > 0
    assert manifest["coverage_gap_episode_count"] > 0
    assert manifest["retrospective"] is True
    assert manifest["blind_test"] is False


def test_v2_dynamic_debug_enforces_source_run_and_safety_conformance(
    artifact_paths: dict[str, Path],
) -> None:
    _ensure_artifacts(artifact_paths)
    report = read_json(artifact_paths["report"])
    method = report["tracks"]["causal_debug_transfer"]["causal_tree_dynamic"]
    assert method["unsafe_count"] == 0
    assert method["source_run_escape_count"] == 0
    assert 0.0 <= method["route_accuracy"] <= 1.0
    assert 0.0 <= method["transition_mrr"] <= 1.0


def test_v2_reports_granularity_and_refuses_downstream_claim(
    artifact_paths: dict[str, Path],
) -> None:
    _ensure_artifacts(artifact_paths)
    report = read_json(artifact_paths["report"])
    assert set(report["tracks"]) == {"stage_granularity", "causal_debug_transfer"}
    assert report["claims"]["granularity_gate_diagnostic_allowed"] is True
    assert report["claims"]["cross_run_causal_retrieval_diagnostic_allowed"] is True
    assert report["claims"]["blind_generalization_claim_allowed"] is False
    assert report["claims"]["downstream_agent_success_claim_allowed"] is False


def test_v2_dynamic_method_beats_legacy_on_combined_test_decision(
    artifact_paths: dict[str, Path],
) -> None:
    _ensure_artifacts(artifact_paths)
    report = read_json(artifact_paths["report"])
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
