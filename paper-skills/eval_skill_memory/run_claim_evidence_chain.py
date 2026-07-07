"""Run the offline evidence chain for Agentic Hyperbolic SOP Memory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EVAL = REPO / "paper-skills" / "eval_skill_memory"
HYPER = REPO / "paper-skills" / "hyper_memory"


def run_cmd(cmd: list[str]) -> None:
    print("+ " + " ".join(str(x) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run certified graph build, benchmark, retrieval, ablation, and readiness report.")
    parser.add_argument("--skip-rebuild", action="store_true", help="Use existing hyper_graph/hyper_index artifacts.")
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    certified_graph = EVAL / "artifacts" / "graph_skillgraph_c_trace_prereq_certified.json"
    bench = EVAL / "benchmarks" / "hyperbolic_sop_benchmark.jsonl"
    gold = EVAL / "gold" / "hyperbolic_sop_gold.jsonl"
    validation = EVAL / "reports" / "benchmark_validation_report.json"
    results = EVAL / "reports" / "hyperbolic_retrieval_results.jsonl"
    ablation = EVAL / "reports" / "hyperbolic_ablation_report.json"

    if not args.skip_rebuild:
        run_cmd([sys.executable, str(EVAL / "certify_skillgraph_provenance.py"), "--output", str(certified_graph)])
        run_cmd([
            sys.executable,
            str(HYPER / "build_hyperbolic_memory.py"),
            "--input",
            str(certified_graph),
            "--output-dir",
            str(HYPER),
            "--require-clean-provenance",
        ])

    run_cmd([sys.executable, str(EVAL / "build_hyperbolic_benchmark.py"), "--graph", str(HYPER / "hyper_graph.json"), "--benchmark", str(bench), "--gold", str(gold)])
    validation_proc = subprocess.run(
        [
            sys.executable,
            str(EVAL / "validate_hyperbolic_benchmark.py"),
            "--graph",
            str(HYPER / "hyper_graph.json"),
            "--benchmark",
            str(bench),
            "--gold",
            str(gold),
            "--require-certified-graph",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    validation.write_text(validation_proc.stdout, encoding="utf-8")
    print(validation_proc.stdout, end="")

    run_cmd([
        sys.executable,
        str(EVAL / "run_hyperbolic_retrieval_benchmark.py"),
        "--graph",
        str(HYPER / "hyper_graph.json"),
        "--index",
        str(HYPER / "hyper_index.npz"),
        "--text-model",
        str(HYPER / "hyper_text_model.joblib"),
        "--benchmark",
        str(bench),
        "--output",
        str(results),
        "--top-k",
        str(args.top_k),
    ])
    run_cmd([
        sys.executable,
        str(HYPER / "evaluate_hyperbolic_ablation.py"),
        "--results",
        str(results),
        "--gold",
        str(gold),
        "--graph",
        str(HYPER / "hyper_graph.json"),
        "--graph-builder-report",
        str(HYPER / "graph_builder_report.json"),
        "--quality-report",
        str(HYPER / "coordinate_quality_report.json"),
        "--output",
        str(ablation),
        "--n-resamples",
        str(args.n_resamples),
    ])
    run_cmd([sys.executable, str(EVAL / "make_claim_readiness_report.py")])

    summary = {
        "status": "completed",
        "certified_graph": str(certified_graph),
        "benchmark": str(bench),
        "gold": str(gold),
        "validation": str(validation),
        "results": str(results),
        "ablation": str(ablation),
        "readiness": str(REPO / "coordination" / "hyperbolic_claim_readiness.md"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
