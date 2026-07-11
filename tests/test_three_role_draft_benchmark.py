import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO / "paper-skills" / "eval_skill_memory" / "build_three_role_draft_benchmark.py"
EVALUATOR_PATH = REPO / "paper-skills" / "hyper_memory" / "evaluate_three_role_draft_retrieval.py"
GRAPH_PATH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
REPLAY_PATH = REPO / "paper-skills" / "eval_skill_memory" / "clean_replay_targets.json"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


builder = _load(BUILDER_PATH, "three_role_draft_builder")
evaluator = _load(EVALUATOR_PATH, "three_role_draft_evaluator")


def _records():
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    replay = json.loads(REPLAY_PATH.read_text(encoding="utf-8"))
    return builder.build_three_role_records(graph, replay)


def test_three_role_benchmark_uses_one_multi_gold_query_per_root():
    queries, gold = _records()
    report = builder.validate(queries, gold)
    assert report["valid"] is True
    assert report["one_query_per_root_run"] is True
    assert len({query["run_id"] for query in queries}) == len(queries)
    assert all(row["gold_policy"] == "multi_gold_method_family_not_single_child" for row in gold)
    assert all(row["relevant_sop_ids"] for row in gold)


def test_three_role_protocol_changes_only_novel_slot():
    queries, _ = _records()
    for query in queries:
        roles = query["three_role_protocol"]
        assert list(roles) == ["coldstart_baseline", "memory_reproduction", "novel_exploration"]
        assert roles["coldstart_baseline"] == {
            "memory_mode": "none",
            "fixed_across_conditions": True,
        }
        assert roles["memory_reproduction"]["fixed_across_conditions"] is True
        assert roles["novel_exploration"]["compared_conditions"] == ["tree_only", "stage_hybrid"]


def test_evaluation_excludes_every_run_in_held_out_split():
    queries, gold = _records()
    report = evaluator.evaluate(queries, gold, split="test")
    excluded = set(report["role_protocol"]["held_out_memory_run_ids"])
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = {str(node["id"]): node for node in graph["nodes"]}
    assert report["statistical_adequacy"]["enough_queries"] is False
    assert report["claim_allowed"] is False
    for rows in report["per_query"].values():
        for row in rows:
            for node_id in row["ranking"]:
                node = nodes[node_id]
                run_id = str(node.get("run_short_id") or node.get("run_id") or "")
                assert run_id not in excluded
            assert row["blocked_positive_count"] == 0


def test_method_family_similarity_is_not_sop_identity():
    left = {"deberta", "tfidf", "xgboost", "ensemble"}
    equivalent = {"deberta", "tfidf", "xgboost", "blend"}
    unrelated = {"resnet", "image", "augmentation"}
    assert evaluator._method_similarity(left, equivalent) >= evaluator.METHOD_SIMILARITY_THRESHOLD
    assert evaluator._method_similarity(left, unrelated) == 0.0
