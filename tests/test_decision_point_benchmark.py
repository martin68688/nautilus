import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO / "paper-skills" / "eval_skill_memory" / "build_decision_point_benchmark.py"
EVALUATOR_PATH = REPO / "paper-skills" / "eval_skill_memory" / "evaluate_decision_point_benchmark.py"
GRAPH_PATH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX_PATH = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load(BUILDER_PATH, "decision_point_builder")
evaluator = _load(EVALUATOR_PATH, "decision_point_evaluator")


def test_decision_points_fail_closed_when_latest_clean_graph_is_underpowered():
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    queries, gold, packets = builder.build_records(graph)
    report = builder.validate_records(graph, queries, gold)
    assert report["valid"] is False
    assert report["query_count"] < 25
    assert any(error.startswith("underpowered_query_count") for error in report["errors"])
    assert any(error.startswith("task_family_coverage") for error in report["errors"])
    assert report["gold_sets_unique_globally"] is True
    assert report["cross_task_gold_allowed"] is False
    assert report["blocked_distractors_task_matched"] is True
    assert report["historical_trajectory_gold_used"] is False
    assert report["paper_claim_ready"] is False
    assert len(packets) == len(queries)
    for query, row, packet in zip(queries, gold, packets):
        assert 50 <= query["candidate_count"] <= 400
        assert query["historical_coordinate_free"] is True
        assert "run::" not in query["query_text"]
        assert "transition::" not in query["query_text"]
        assert len(row["labels"]) >= 3
        assert all(label["clean_supporting_transition_count"] >= 1 for label in row["labels"])
        blocked = [candidate for candidate in packet["candidates"] if candidate["candidate_type"] == "RunNode"]
        assert len(blocked) == 5
        assert all(candidate["relevance_0_to_3"] is None for candidate in packet["candidates"])


def test_validator_rejects_historical_child_gold_and_duplicate_decision_text():
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    queries, gold, _packets = builder.build_records(graph, candidate_pool_size=80)
    queries[0]["query_text"] += " run::historical-child"
    queries[1]["query_text"] = queries[0]["query_text"]
    report = builder.validate_records(graph, queries, gold)
    assert report["valid"] is False
    assert any(error.startswith("duplicate_query_text") for error in report["errors"])
    assert any(error.startswith("historical_coordinate_leak") for error in report["errors"])


def test_graded_metrics_and_holm_adjustment_are_well_formed():
    relevance = {"a": 3, "b": 2, "c": 1}
    assert evaluator.graded_ndcg(["a", "b", "c"], relevance, 3) == 1.0
    assert evaluator.average_precision(["a", "x", "b"], relevance, 3) > 0.0
    better = evaluator.paired_inference([1.0] * 20, [0.0] * 20, samples=500)
    assert better["delta"] == 1.0
    assert better["bootstrap_ci95"][0] > 0.0
    assert better["sign_flip_p_value_two_sided"] < 0.05
    adjusted = evaluator.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.20})
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_evaluator_compares_controls_but_keeps_silver_claim_gate_closed(monkeypatch):
    # This contract test exercises method wiring and claim gates, not the optional
    # MiniLM native runtime.  The local macOS torch stack can segfault in BERT
    # embedding kernels, so make the optional baseline take its supported
    # unavailable path deterministically.
    import sentence_transformers

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("disabled in unit contract test")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", unavailable)
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    queries, gold, _packets = builder.build_records(graph, candidate_pool_size=60)
    sample_queries = queries[:2]
    sample_ids = {row["query_id"] for row in sample_queries}
    sample_gold = [row for row in gold if row["query_id"] in sample_ids]
    report = evaluator.evaluate(
        sample_queries,
        sample_gold,
        graph_path=GRAPH_PATH,
        index_path=INDEX_PATH,
        top_k=5,
        bootstrap_samples=100,
    )
    assert set(report["methods"]) == set(evaluator.METHODS)
    assert report["methods"]["oracle_upper"]["graded_ndcg_at_10"] == 1.0
    assert report["methods"]["oracle_upper"]["non_admissible_rate_at_10"] == 0.0
    assert "tfidf_unfiltered" in report["methods"]
    assert "tfidf_safety_filtered" in report["methods"]
    assert "taxonomy_tfidf" not in report["methods"]
    assert report["claim_gates"]["historical_child_as_gold"] is False
    assert report["claim_gates"]["offline_retrieval_claim_allowed"] is False
    assert report["claim_gates"]["online_downstream_claim_allowed"] is False
