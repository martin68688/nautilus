"""Run deterministic offline retrieval for Agentic Hyperbolic SOP Memory.

The runner intentionally avoids LLM tool-choice in the first pass. Each
agentic system performs the same three read-only map steps:
inspect_map -> navigate(condition/failure filters) -> check_conflicts.
This isolates the scorer while preserving the MemoryNavigator shape.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mlevolve"))

from agents.memory.external_skill_memory import ExternalSkillMemoryLayer  # noqa: E402


DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_INDEX = REPO / "paper-skills" / "hyper_memory" / "hyper_index.npz"
DEFAULT_TEXT_MODEL = REPO / "paper-skills" / "hyper_memory" / "hyper_text_model.joblib"
DEFAULT_BENCH = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "hyperbolic_sop_benchmark.jsonl"
DEFAULT_OUTPUT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_retrieval_results.jsonl"

SYSTEMS = {
    "skillgraph_c_lexical": {"mode": "skillgraph", "scoring_mode": "lexical", "enable_agentic": False},
    "agentic_lexical": {"mode": "skillgraph", "scoring_mode": "lexical", "enable_agentic": True},
    "agentic_poincare": {"mode": "agentic_hyperbolic", "scoring_mode": "poincare", "enable_agentic": True},
    "agentic_flat_twin": {"mode": "flat_twin_agentic", "scoring_mode": "flat_twin", "enable_agentic": True},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def query_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("context", "")),
            "Conditions: " + "; ".join(row.get("condition") or []),
            "Failure modes: " + "; ".join(row.get("failure_mode") or []),
        ]
    ).strip()


def make_layer(
    *,
    system_name: str,
    graph_path: Path,
    index_path: Path,
    text_model_path: Path,
    top_k: int,
) -> ExternalSkillMemoryLayer:
    cfg = SYSTEMS[system_name]
    return ExternalSkillMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        text_model_path=str(text_model_path),
        source_name=system_name,
        mode=cfg["mode"],
        scoring_mode=cfg["scoring_mode"],
        enable_agentic=cfg["enable_agentic"],
        navigator_max_steps=3,
        navigator_reference_budget=1200,
        top_k=top_k,
        cfg=None,
    )


def run_one(layer: ExternalSkillMemoryLayer, system_name: str, row: dict[str, Any], top_k: int) -> dict[str, Any]:
    started = time.time()
    task_type = str(row.get("task_type", ""))
    text = query_text(row)
    if system_name == "skillgraph_c_lexical":
        selected = layer._retrieve_ids(task_type=task_type, query_text=text)[:top_k]
        trace = ["skillgraph_c_lexical_retrieve_ids"]
        risk_warnings: list[str] = []
        rejected_sops: list[dict[str, str]] = []
    else:
        map_obs = layer.inspect_map(task_type=task_type, query_text=text, context=str(row.get("context", "")))
        nav = layer.navigate(
            task_type=task_type,
            query_text=text,
            condition=" ".join(row.get("condition") or []),
            failure_mode=" ".join(row.get("failure_mode") or []),
            top_k=top_k,
        )
        selected = [s["id"] for s in nav.get("sops", []) if s.get("id") in layer.nodes][:top_k]
        conflicts = layer.check_conflicts(sop_ids=selected[:4], context=text)
        trace = [
            "inspect_map(context)",
            "navigate(condition=query.condition, failure_mode=query.failure_mode)",
            "check_conflicts(selected_sops)",
        ]
        risk_warnings = conflicts.get("risk_warnings", [])
        rejected_sops = []
        _ = map_obs

    pack = {
        "selected_sops": selected,
        "risk_warnings": risk_warnings,
        "navigation_trace": trace,
        "rejected_sops": rejected_sops,
    }
    injected_tokens = len(json.dumps(pack, ensure_ascii=False).split())
    elapsed = round(time.time() - started, 6)
    return {
        "query_id": row["query_id"],
        "query_kind": row.get("query_kind", ""),
        "task_type": task_type,
        "system": system_name,
        "scoring_mode": layer.scoring_mode,
        "selected_sops": selected,
        "scores": [
            {"sop_id": sid, "rank": rank, "score": round(1.0 / rank, 6)}
            for rank, sid in enumerate(selected, 1)
        ],
        "navigation_trace": trace,
        "risk_warnings": risk_warnings,
        "rejected_sops": rejected_sops,
        "opened_references": [],
        "latency_sec": elapsed,
        "injected_tokens": injected_tokens,
    }


def run(
    *,
    graph_path: Path,
    index_path: Path,
    text_model_path: Path,
    benchmark_path: Path,
    output_path: Path,
    systems: list[str],
    top_k: int = 5,
) -> dict[str, Any]:
    queries = read_jsonl(benchmark_path)
    layers = {
        name: make_layer(
            system_name=name,
            graph_path=graph_path,
            index_path=index_path,
            text_model_path=text_model_path,
            top_k=top_k,
        )
        for name in systems
    }
    rows = []
    for query in queries:
        for name in systems:
            rows.append(run_one(layers[name], name, query, top_k=top_k))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return {
        "status": "completed",
        "output": str(output_path),
        "queries": len(queries),
        "systems": systems,
        "rows": len(rows),
        "top_k": top_k,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic hyperbolic SOP retrieval benchmark.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--text-model", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    unknown = sorted(set(args.systems) - set(SYSTEMS))
    if unknown:
        raise SystemExit(f"Unknown systems: {unknown}")
    print(json.dumps(run(
        graph_path=args.graph,
        index_path=args.index,
        text_model_path=args.text_model,
        benchmark_path=args.benchmark,
        output_path=args.output,
        systems=args.systems,
        top_k=args.top_k,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
