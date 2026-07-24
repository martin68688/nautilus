import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO / "paper-skills" / "eval_skill_memory" / "build_stage_hybrid_benchmark.py"
EVALUATOR_PATH = REPO / "paper-skills" / "hyper_memory" / "evaluate_stage_hybrid_retrieval.py"
MATRIX_PATH = REPO / "paper-skills" / "hyper_memory" / "run_runforest_online_matrix.py"
PREFLIGHT_PATH = REPO / "paper-skills" / "hyper_memory" / "run_stage_hybrid_preflight.py"
GRAPH_PATH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX_PATH = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load(BUILDER_PATH, "stage_hybrid_benchmark_builder")
evaluator = _load(EVALUATOR_PATH, "stage_hybrid_evaluator")
matrix = _load(MATRIX_PATH, "stage_hybrid_online_matrix")
preflight = _load(PREFLIGHT_PATH, "stage_hybrid_preflight")


def test_benchmark_is_natural_language_and_grouped_by_run():
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    queries, gold = builder.build_records(graph, max_queries=80)
    report = builder.validate_records(queries, gold)
    assert report["valid"] is True
    assert report["grouped_by_run"] is True
    assert report["natural_language_no_gold_coordinates"] is True
    assert report["query_count"] == 80
    splits_by_run = {}
    for query in queries:
        splits_by_run.setdefault(query["run_id"], set()).add(query["split"])
        assert "Task:" in query["query_text"]
        assert "Agent stage:" in query["query_text"]
    assert all(len(splits) == 1 for splits in splits_by_run.values())


def test_validator_rejects_run_split_leakage():
    queries = [
        {"query_id": "q1", "run_id": "r", "split": "dev", "stage": "draft", "query_text": "Task: x"},
        {"query_id": "q2", "run_id": "r", "split": "test", "stage": "draft", "query_text": "Task: y"},
    ]
    gold = [
        {"query_id": "q1", "run_id": "r", "split": "dev", "gold_sop_ids": ["s"], "gold_transition_ids": ["t"], "gold_execution_ids": ["n"], "gold_evidence_ids": []},
        {"query_id": "q2", "run_id": "r", "split": "test", "gold_sop_ids": ["s"], "gold_transition_ids": ["t"], "gold_execution_ids": ["n"], "gold_evidence_ids": []},
    ]
    report = builder.validate_records(queries, gold)
    assert report["valid"] is False
    assert any(error.startswith("run_split_leakage") for error in report["errors"])


def test_paired_bootstrap_detects_direction():
    better = evaluator.paired_bootstrap([1.0] * 20, [0.0] * 20, samples=500)
    worse = evaluator.paired_bootstrap([0.0] * 20, [1.0] * 20, samples=500)
    assert better["delta"] == 1.0 and better["p_value"] < 0.05
    assert worse["delta"] == -1.0 and worse["p_value"] > 0.95


def test_evaluator_has_all_controls_and_stage_level_gates():
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    queries, gold = builder.build_records(graph, max_queries=80)
    split = max(("train", "dev", "test"), key=lambda value: sum(q["split"] == value for q in queries))
    report = evaluator.evaluate(queries, gold, graph=GRAPH_PATH, index=INDEX_PATH, split=split)
    assert set(report["controls"]) == set(evaluator.CONTROL_NAMES)
    assert report["query_count"] > 0
    assert report["claim_gates"]["by_stage"]
    assert report["claim_gates"]["online_downstream_claim_allowed"] is False
    assert report["claim_gates"]["adoption_precision_available"] is False
    assert report["claim_gates"]["downstream_metric_available"] is False
    assert report["controls"]["stage_hybrid"]["blocked_positive_count"] == 0.0
    for stage, gate in report["claim_gates"]["by_stage"].items():
        assert gate["best_single_channel"] in {"sop_only", "tree_only"}
        assert gate["query_count"] > 0


def test_online_matrix_exposes_the_same_seven_controls():
    assert set(evaluator.CONTROL_NAMES) <= set(matrix.CONDITIONS)
    assert "layered_strategy" in matrix.CONDITIONS
    assert "external_skill_memory.enable=False" in matrix.CONDITIONS["no_memory"]
    assert "agent.draft_role_policy.enabled=False" in matrix.CONDITIONS["no_memory"]
    for control in ("sop_only", "tree_only", "naive_concat", "stage_hybrid"):
        assert f"external_skill_memory.retrieval_control={control}" in matrix.CONDITIONS[control]
    assert "external_skill_memory.retrieval_control=layered_strategy" in matrix.CONDITIONS["layered_strategy"]
    assert "external_skill_memory.scoring_mode=flat_twin" in matrix.CONDITIONS["flat_twin_hybrid"]
    assert "external_skill_memory.scoring_mode=euclidean" in matrix.CONDITIONS["independent_euclidean"]


def test_no_gpu_preflight_covers_config_provenance_routes_and_benchmark():
    report = preflight.run_preflight(evaluate_offline=False)
    assert report["ok"] is True
    assert report["online_training_started"] is False
    assert report["checks"]["coldstart_template"]["sha256"] == preflight.COLDSTART_SHA256
    cases = report["checks"]["runtime_routes"]["cases"]
    assert len(cases) == (
        len(preflight.RETRIEVAL_CONTROLS - {"layered_strategy"})
        * len(preflight.TASKS)
        * 5
    )
    assert {row["task"] for row in cases} == set(preflight.TASKS)
    assert all(row["blocked_positive_count"] == 0 for row in cases)
    assert any(row["historical_source_runs"] for row in cases)
    no_memory = [row for row in cases if row["control"] == "no_memory"]
    assert no_memory
    assert all(row["ref_count"] == 0 for row in no_memory)
    assert all(row["historical_source_runs"] == [] for row in no_memory)
    assert all(
        row["expected_algorithm"] == "formal_flat_relevance_v1"
        for row in cases
        if row["control"]
        in {
            "flat_relevance_memory",
            "global_validity_bit",
            "authority_only",
        }
    )
    assert report["checks"]["sparse_task_memory_fallback"]["ok"] is True
    assert report["checks"]["exact_replay_coverage"]["ok"] is True
    assert report["checks"]["exact_replay_coverage"]["memory_transfer_tasks"] == ["mlsp-2013-birds"]
    assert all(case["blocked_positive_count"] == 0 for case in report["checks"]["runtime_routes"]["cases"])
    assert report["checks"]["layered_three_role"]["status"] == "passed"
    assert len(report["checks"]["layered_three_role"]["strategy_routes"]) == 3
