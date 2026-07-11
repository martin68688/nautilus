import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "paper-skills" / "hyper_memory" / "evaluate_layered_novel_strategy.py"
BENCHMARK = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "three_role_draft_benchmark.jsonl"


def _module():
    spec = importlib.util.spec_from_file_location("layered_strategy_eval", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_layered_strategy_benchmark_reports_actual_three_route_schemes():
    module = _module()
    report = module.evaluate(module._read_jsonl(BENCHMARK), split="test")
    assert report["schema"] == "layered_novel_strategy_evaluation_v1"
    assert set(report["controls"]) == {"tree_only", "stage_hybrid", "layered_strategy"}
    layered = report["controls"]["layered_strategy"]
    assert layered["strategy_precision_at_3"] == 1.0
    assert layered["mean_distinct_method_families_at_3"] == 3.0
    assert layered["detail_intrusion_at_3"] == 0.0
    assert layered["clean_expansion_precision_at_3"] == 1.0
    assert layered["excluded_family_violation_count"] == 0
    assert layered["error_count"] == 0
    assert report["claim_allowed"] is False
    assert all(len(row["schemes"]) == 3 for row in report["per_query"]["layered_strategy"])
