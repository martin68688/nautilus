"""Tune geometric scoring parameters on dev split and evaluate on heldout test.

This script optimizes Agentic Poincare retrieval on the dev split. It reports
two held-out controls separately:
  * Flat-Twin: same Poincare coordinates, Euclidean distance only.
  * Agentic Euclidean Memory: independent flat coordinates, Euclidean distance.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_INDEX = REPO / "paper-skills" / "hyper_memory" / "hyper_index.npz"
DEFAULT_TEXT_MODEL = REPO / "paper-skills" / "hyper_memory" / "hyper_text_model.joblib"
DEFAULT_BENCH = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "hyperbolic_sop_benchmark.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "hyperbolic_sop_gold.jsonl"
DEFAULT_REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "poincare_tuning_report.json"
DEFAULT_RESULTS = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_retrieval_results_tuned.jsonl"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("run_hyperbolic_retrieval_benchmark", REPO / "paper-skills" / "eval_skill_memory" / "run_hyperbolic_retrieval_benchmark.py")
ablation = load_module("evaluate_hyperbolic_ablation", REPO / "paper-skills" / "hyper_memory" / "evaluate_hyperbolic_ablation.py")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def param_grid() -> list[dict[str, Any]]:
    grid = []
    for norm, distance_weight, semantic_weight, constraint_weight, radius_quantile in itertools.product(
        ["none", "minmax", "zscore"],
        [0.35, 0.6, 1.0],
        [0.0, 0.2, 0.5, 0.8],
        [0.05, 0.2, 0.5, 1.0],
        [0.35, 0.5, 0.65],
    ):
        grid.append(
            {
                "geometry_distance_norm": norm,
                "geometry_distance_weight": distance_weight,
                "geometry_semantic_weight": semantic_weight,
                "geometry_constraint_weight": constraint_weight,
                "geometry_query_radius_quantile": radius_quantile,
            }
        )
    return grid


def filter_rows(rows: list[dict[str, Any]], query_ids: set[str]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("query_id")) in query_ids]


def score_report(report: dict[str, Any]) -> float:
    p = report["systems"].get("agentic_poincare", {})
    # Dev objective favors discriminative ranking metrics on the hard benchmark,
    # while keeping rare recall and condition precision in the loop.
    return (
        1.2 * float(p.get("mrr", 0.0))
        + 1.0 * float(p.get("exact_recall_at_1", 0.0))
        + 0.8 * float(p.get("ndcg_at_5", 0.0))
        + 0.8 * float(p.get("rare_recall_at_5", 0.0))
        + 0.7 * float(p.get("condition_precision", 0.0))
        + 0.3 * float(p.get("conflict_warning_precision", 0.0))
        - 0.3 * float(p.get("distractor_rate_at_5", 0.0))
    )


def compact_system_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "systems": report.get("systems", {}),
        "systems_by_query_kind": report.get("systems_by_query_kind", {}),
        "comparisons": report.get("comparisons", {}),
        "rare_recall_at_5": report.get("rare_recall_at_5", {}),
        "condition_precision": report.get("condition_precision", {}),
        "ranking_diagnostics": report.get("ranking_diagnostics", {}),
    }


def tuning_diagnostics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    best_objective = trials[0]["objective"] if trials else 0.0
    near_best = [t for t in trials if abs(float(t["objective"]) - float(best_objective)) < 1e-12]
    by_norm: dict[str, list[float]] = {}
    by_quantile: dict[str, list[float]] = {}
    for trial in trials:
        params = trial["params"]
        by_norm.setdefault(str(params["geometry_distance_norm"]), []).append(float(trial["objective"]))
        by_quantile.setdefault(str(params["geometry_query_radius_quantile"]), []).append(float(trial["objective"]))
    return {
        "grid_size": len(trials),
        "near_best_trial_count": len(near_best),
        "near_best_params_sample": [trial["params"] for trial in near_best[:5]],
        "objective_by_distance_norm": {
            key: {"best": max(vals), "mean": sum(vals) / len(vals)}
            for key, vals in sorted(by_norm.items())
        },
        "objective_by_radius_quantile": {
            key: {"best": max(vals), "mean": sum(vals) / len(vals)}
            for key, vals in sorted(by_quantile.items(), key=lambda item: float(item[0]))
        },
    }


def evaluate_with_params(
    *,
    params: dict[str, Any],
    graph: dict[str, Any],
    graph_path: Path,
    index_path: Path,
    text_model_path: Path,
    benchmark_path: Path,
    gold_by_query: dict[str, list[dict[str, Any]]],
    query_ids: set[str],
    top_k: int,
) -> dict[str, Any]:
    tmp = DEFAULT_RESULTS.with_suffix(".tmp.jsonl")
    run_summary = runner.run(
        graph_path=graph_path,
        index_path=index_path,
        text_model_path=text_model_path,
        benchmark_path=benchmark_path,
        output_path=tmp,
        systems=["agentic_poincare", "agentic_flat_twin"],
        top_k=top_k,
        geometry_params=params,
    )
    rows = filter_rows(read_jsonl(tmp), query_ids)
    tmp.unlink(missing_ok=True)
    report = ablation.evaluate_runner_results(
        result_rows=rows,
        gold_by_query=gold_by_query,
        graph=graph,
        n_resamples=1000,
        seed=42,
    )
    report["run_summary"] = run_summary
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Poincare scoring parameters on dev split.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--text-model", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    benchmark = read_jsonl(args.benchmark)
    dev_ids = {row["query_id"] for row in benchmark if row.get("split") == "dev"}
    test_ids = {row["query_id"] for row in benchmark if row.get("split") == "test"}
    if not dev_ids or not test_ids:
        raise SystemExit("benchmark must contain non-empty dev and test splits")
    gold_by_query = ablation.load_gold(args.gold)
    graph = json.loads(args.graph.read_text(encoding="utf-8"))

    trials = []
    for params in param_grid():
        report = evaluate_with_params(
            params=params,
            graph=graph,
            graph_path=args.graph,
            index_path=args.index,
            text_model_path=args.text_model,
            benchmark_path=args.benchmark,
            gold_by_query=gold_by_query,
            query_ids=dev_ids,
            top_k=args.top_k,
        )
        trials.append(
            {
                "params": params,
                "objective": score_report(report),
                "dev_status": report["status"],
                "dev_poincare": report["systems"].get("agentic_poincare", {}),
                "dev_flat_twin": report["systems"].get("agentic_flat_twin", {}),
                "dev_rr_diff": report["rare_recall_at_5"]["observed_mean_diff"],
                "dev_cp_diff": report["condition_precision"]["mean_diff"],
            }
        )

    trials.sort(
        key=lambda row: (
            -row["objective"],
            -float(row["dev_poincare"].get("rare_recall_at_5", 0.0)),
            -float(row["dev_poincare"].get("condition_precision", 0.0)),
            row["params"]["geometry_distance_norm"],
            float(row["params"]["geometry_query_radius_quantile"]),
        )
    )
    best = trials[0]
    best_params = best["params"]

    full_summary = runner.run(
        graph_path=args.graph,
        index_path=args.index,
        text_model_path=args.text_model,
        benchmark_path=args.benchmark,
        output_path=args.results_output,
        systems=["skillgraph_c_lexical", "agentic_lexical", "agentic_euclidean", "agentic_poincare", "agentic_flat_twin"],
        top_k=args.top_k,
        geometry_params=best_params,
    )
    full_rows = read_jsonl(args.results_output)
    dev_report = ablation.evaluate_runner_results(
        result_rows=filter_rows(full_rows, dev_ids),
        gold_by_query=gold_by_query,
        graph=graph,
        n_resamples=5000,
        seed=42,
    )
    test_report = ablation.evaluate_runner_results(
        result_rows=filter_rows(full_rows, test_ids),
        gold_by_query=gold_by_query,
        graph=graph,
        n_resamples=5000,
        seed=42,
    )
    full_report = ablation.evaluate_runner_results(
        result_rows=full_rows,
        gold_by_query=gold_by_query,
        graph=graph,
        n_resamples=10000,
        seed=42,
    )
    out = {
        "status": "completed",
        "selection_policy": "optimize Agentic Poincare on dev split; report heldout test separately",
        "best_params": best_params,
        "best_dev_trial": best,
        "top_trials": trials[:10],
        "tuning_diagnostics": tuning_diagnostics(trials),
        "dev_query_count": len(dev_ids),
        "test_query_count": len(test_ids),
        "full_run_summary": full_summary,
        "dev_report": compact_system_metrics(dev_report),
        "test_report": compact_system_metrics(test_report),
        "full_report": compact_system_metrics(full_report),
        "note": "Do not claim geometry from dev tuning alone. Use heldout test report for claims.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": out["status"],
        "best_params": best_params,
        "dev_poincare": dev_report["systems"].get("agentic_poincare", {}),
        "test_poincare": test_report["systems"].get("agentic_poincare", {}),
        "test_euclidean": test_report["systems"].get("agentic_euclidean", {}),
        "test_flat_twin": test_report["systems"].get("agentic_flat_twin", {}),
        "test_rr_diff": test_report["rare_recall_at_5"]["observed_mean_diff"],
        "test_p": test_report["rare_recall_at_5"]["p_value"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
