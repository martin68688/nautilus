#!/usr/bin/env python3
"""Evaluate stage-aware SOP/Tree routing on held-out natural-language queries."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer


DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
DEFAULT_INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
DEFAULT_BENCHMARK = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "stage_hybrid_runforest_benchmark.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "stage_hybrid_runforest_gold.jsonl"
DEFAULT_REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "stage_hybrid_retrieval_evaluation.json"
DEFAULT_MD = REPO / "coordination" / "stage_aware_sop_gateway_runforest_readiness.md"

CONTROL_NAMES = (
    "no_memory",
    "sop_only",
    "tree_only",
    "naive_concat",
    "stage_hybrid",
    "flat_twin_hybrid",
    "independent_euclidean",
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def reciprocal_rank(ranking: list[str], gold: set[str]) -> float:
    for index, node_id in enumerate(ranking, 1):
        if node_id in gold:
            return 1.0 / index
    return 0.0


def recall_at(ranking: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 1.0
    return len(set(ranking[:k]) & gold) / len(gold)


def ndcg_at(ranking: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 1.0
    dcg = sum((1.0 / math.log2(index + 2)) for index, node_id in enumerate(ranking[:k]) if node_id in gold)
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(gold))))
    return dcg / ideal if ideal else 0.0


def paired_bootstrap(left: list[float], right: list[float], *, samples: int = 2000, seed: int = 17) -> dict[str, float]:
    if len(left) != len(right) or not left:
        return {"delta": 0.0, "p_value": 1.0}
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = float(delta[rng.integers(0, len(delta), len(delta))].mean())
    return {
        "delta": float(delta.mean()),
        "p_value": float((np.count_nonzero(means <= 0.0) + 1) / (samples + 1)),
    }


def _layer(graph: Path, index: Path, scoring: str) -> StageAwareHybridMemoryLayer:
    return StageAwareHybridMemoryLayer(
        graph_path=str(graph),
        index_path=str(index),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        scoring_mode=scoring,
        enable_agentic=False,
        top_k=10,
        max_chars=0,
    )


def _run_control(
    layer: StageAwareHybridMemoryLayer,
    control: str,
    query: dict,
) -> dict[str, Any]:
    stage = query["stage"]
    text = query["query_text"]
    quotas = layer.stage_quotas[stage]
    started = time.perf_counter()
    if control == "no_memory":
        pack = {
            "selected_sop_gateways": [], "gateway_transitions": {}, "tree_candidates": [],
            "fused_execution_candidates": [], "evidence_refs": [], "risk_warnings": [],
        }
        prompt_text = ""
    elif control == "tree_only":
        tree_ids = layer._rank_tree(
            stage=stage, query_text=text, task_id=query["task"], task_desc=query["task"],
            limit=quotas["tree_candidates"],
        )
        pack = {
            "selected_sop_gateways": [], "gateway_transitions": {}, "tree_candidates": tree_ids,
            "fused_execution_candidates": [{"id": node_id} for node_id in tree_ids],
            "evidence_refs": [], "risk_warnings": [],
        }
        prompt_text = json.dumps(tree_ids)
    else:
        sop_candidates = layer._rank_sops(text, stage, quotas["sop_candidates"])
        selected, selection_meta = layer._select_gateways(
            sop_candidates, stage=stage, query_text=text, limit=quotas["sop_gateways"]
        )
        sop_execution, gateway_transitions, evidence, failures, trace = layer._expand_gateways(selected)
        if control == "sop_only":
            execution = sop_execution
            tree_ids = []
        elif control == "naive_concat":
            tree_ids = layer._rank_tree(
                stage=stage, query_text=text, task_id=query["task"], task_desc=query["task"],
                limit=quotas["tree_candidates"],
            )
            execution = list(dict.fromkeys(sop_execution + tree_ids))
        else:
            pack = layer._hybrid_pack(
                stage=stage, task_id=query["task"], task_desc=query["task"], query_text=text
            )
            prompt_text = layer._format_hybrid_pack(pack)
            elapsed = time.perf_counter() - started
            return {"pack": pack, "latency_sec": elapsed, "token_estimate": len(prompt_text) / 4.0}
        pack = {
            "selected_sop_gateways": selected,
            "gateway_transitions": gateway_transitions,
            "tree_candidates": tree_ids,
            "fused_execution_candidates": [{"id": node_id} for node_id in execution],
            "evidence_refs": evidence,
            "failure_patterns": failures,
            "risk_warnings": layer._risk_warnings(sop_candidates),
            "gateway_selection": selection_meta,
            "navigation_trace": trace,
        }
        prompt_text = json.dumps(pack, ensure_ascii=False)
    elapsed = time.perf_counter() - started
    return {"pack": pack, "latency_sec": elapsed, "token_estimate": len(prompt_text) / 4.0}


def _positive_ids(pack: dict) -> set[str]:
    ids = {item["id"] for item in pack.get("fused_execution_candidates", [])}
    ids.update(item["id"] for item in pack.get("selected_sop_gateways", []))
    return ids


def evaluate(
    queries: list[dict],
    gold_rows: list[dict],
    *,
    graph: Path,
    index: Path,
    split: str = "test",
) -> dict[str, Any]:
    gold_by_id = {row["query_id"]: row for row in gold_rows}
    selected_queries = [query for query in queries if query["split"] == split]
    layers = {
        "default": _layer(graph, index, "poincare"),
        "flat_twin_hybrid": _layer(graph, index, "flat_twin"),
        "independent_euclidean": _layer(graph, index, "euclidean"),
    }
    per_control: dict[str, list[dict]] = {name: [] for name in CONTROL_NAMES}
    for query in selected_queries:
        gold = gold_by_id[query["query_id"]]
        for control in CONTROL_NAMES:
            layer = layers.get(control, layers["default"])
            effective = "stage_hybrid" if control in {"flat_twin_hybrid", "independent_euclidean"} else control
            result = _run_control(layer, effective, query)
            pack = result["pack"]
            gateway_ids = [item["id"] for item in pack.get("selected_sop_gateways", [])]
            transition_ids = [
                transition_id
                for values in pack.get("gateway_transitions", {}).values()
                for transition_id in values
            ]
            execution_ids = [item["id"] for item in pack.get("fused_execution_candidates", [])]
            evidence_ids = pack.get("evidence_refs", [])
            gold_sops = set(gold["gold_sop_ids"])
            gold_transitions = set(gold["gold_transition_ids"])
            gold_execution = set(gold["gold_execution_ids"])
            gold_evidence = set(gold["gold_evidence_ids"])
            positive = _positive_ids(pack)
            blocked_positive = 0
            for node_id in positive:
                node = layer.nodes.get(node_id, {})
                if node.get("type") == "Transition":
                    eligible, _reason = layer._positive_transition(node_id)
                elif node.get("type") == "RunNode":
                    eligible = layer._positive_memory_eligible(node)
                else:
                    eligible = bool(node.get("type") == "SOP" and any(
                        layer._positive_transition(tid)[0] for tid in layer._transitions_by_sop.get(node_id, [])
                    ))
                blocked_positive += int(not eligible)
            evidence_precision = (
                len(set(evidence_ids) & gold_evidence) / len(set(evidence_ids))
                if evidence_ids else 0.0
            )
            per_control[control].append(
                {
                    "query_id": query["query_id"],
                    "stage": query["stage"],
                    "gateway_recall_at_5": recall_at(gateway_ids, gold_sops, 5),
                    "gateway_mrr": reciprocal_rank(gateway_ids, gold_sops),
                    "transition_recall_at_5": recall_at(transition_ids, gold_transitions, 5),
                    "execution_recall_at_5": recall_at(execution_ids, gold_execution, 5),
                    "execution_mrr": reciprocal_rank(execution_ids, gold_execution),
                    "execution_ndcg_at_5": ndcg_at(execution_ids, gold_execution, 5),
                    "evidence_precision": evidence_precision,
                    "blocked_positive_count": blocked_positive,
                    "risk_warning_count": len(pack.get("risk_warnings", [])),
                    "latency_sec": result["latency_sec"],
                    "token_estimate": result["token_estimate"],
                }
            )

    aggregate = {}
    metric_names = (
        "gateway_recall_at_5", "gateway_mrr", "transition_recall_at_5", "execution_recall_at_5",
        "execution_mrr", "execution_ndcg_at_5", "evidence_precision", "blocked_positive_count",
        "latency_sec", "token_estimate",
    )
    for control, rows in per_control.items():
        aggregate[control] = {
            "query_count": len(rows),
            **{metric: float(np.mean([row[metric] for row in rows])) if rows else 0.0 for metric in metric_names},
            "by_stage": {},
        }
        for stage in sorted({row["stage"] for row in rows}):
            stage_rows = [row for row in rows if row["stage"] == stage]
            aggregate[control]["by_stage"][stage] = {
                metric: float(np.mean([row[metric] for row in stage_rows])) for metric in metric_names
            }

    comparisons = {}
    for control in ("sop_only", "tree_only", "naive_concat", "flat_twin_hybrid", "independent_euclidean"):
        comparisons[control] = paired_bootstrap(
            [row["execution_mrr"] for row in per_control["stage_hybrid"]],
            [row["execution_mrr"] for row in per_control[control]],
        )
    stages = sorted({row["stage"] for row in per_control["stage_hybrid"]})
    stage_claim_gates = {}
    for stage in stages:
        single = max(
            ("sop_only", "tree_only"),
            key=lambda name: aggregate[name]["by_stage"][stage]["execution_mrr"],
        )
        hybrid_rows = [row for row in per_control["stage_hybrid"] if row["stage"] == stage]
        single_rows = [row for row in per_control[single] if row["stage"] == stage]
        comparison = paired_bootstrap(
            [row["execution_mrr"] for row in hybrid_rows],
            [row["execution_mrr"] for row in single_rows],
        )
        allowed = (
            comparison["delta"] >= 0
            and comparison["p_value"] < 0.05
            and aggregate["stage_hybrid"]["by_stage"][stage]["blocked_positive_count"] == 0
            and aggregate["stage_hybrid"]["by_stage"][stage]["evidence_precision"]
            >= aggregate[single]["by_stage"][stage]["evidence_precision"]
        )
        stage_claim_gates[stage] = {
            "best_single_channel": single,
            "query_count": len(hybrid_rows),
            "execution_mrr_comparison": comparison,
            "allowed": allowed,
        }
    best_single = max(("sop_only", "tree_only"), key=lambda name: aggregate[name]["execution_mrr"])
    retrieval_gate = bool(stage_claim_gates) and all(item["allowed"] for item in stage_claim_gates.values())
    geometry_gate = (
        comparisons["flat_twin_hybrid"]["delta"] > 0
        and comparisons["flat_twin_hybrid"]["p_value"] < 0.05
        and comparisons["independent_euclidean"]["delta"] > 0
        and comparisons["independent_euclidean"]["p_value"] < 0.05
    )
    return {
        "schema": "stage_hybrid_retrieval_evaluation_v1",
        "split": split,
        "query_count": len(selected_queries),
        "controls": aggregate,
        "comparisons_vs_stage_hybrid": comparisons,
        "claim_gates": {
            "best_single_channel": best_single,
            "by_stage": stage_claim_gates,
            "offline_retrieval_claim_allowed": retrieval_gate,
            "hyperbolic_geometry_claim_allowed": geometry_gate,
            "online_downstream_claim_allowed": False,
            "online_gate_reason": "No concurrent online downstream experiment is part of this offline evaluator.",
            "adoption_precision_available": False,
            "downstream_metric_available": False,
        },
        "per_query": per_control,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage-Aware SOP Gateway + RunForest Readiness",
        "",
        f"Held-out split: `{report['split']}`; queries: {report['query_count']}.",
        "",
        "| Control | Gateway MRR | Transition R@5 | Execution MRR | Evidence precision | Blocked positive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in CONTROL_NAMES:
        row = report["controls"][name]
        lines.append(
            f"| {name} | {row['gateway_mrr']:.4f} | {row['transition_recall_at_5']:.4f} | "
            f"{row['execution_mrr']:.4f} | {row['evidence_precision']:.4f} | {row['blocked_positive_count']:.2f} |"
        )
    lines += [
        "",
        "## Stage Gates",
        "",
        "| Stage | Queries | Best single channel | Hybrid MRR delta | p-value | Allowed |",
        "|---|---:|---|---:|---:|---|",
    ]
    for stage, gate in report["claim_gates"]["by_stage"].items():
        comparison = gate["execution_mrr_comparison"]
        lines.append(
            f"| {stage} | {gate['query_count']} | {gate['best_single_channel']} | "
            f"{comparison['delta']:.4f} | {comparison['p_value']:.4f} | {gate['allowed']} |"
        )
    lines += ["", "## Overall Claim Gates", ""]
    for key in (
        "offline_retrieval_claim_allowed",
        "hyperbolic_geometry_claim_allowed",
        "online_downstream_claim_allowed",
        "adoption_precision_available",
        "downstream_metric_available",
    ):
        lines.append(f"- `{key}`: `{report['claim_gates'][key]}`")
    lines += ["", "## Geometry Comparisons", ""]
    for control in ("flat_twin_hybrid", "independent_euclidean"):
        comparison = report["comparisons_vs_stage_hybrid"][control]
        lines.append(
            f"- Stage Hybrid minus `{control}` Execution MRR: "
            f"`{comparison['delta']:.4f}` (`p={comparison['p_value']:.4f}`)."
        )
    lines += [
        "",
        "Offline retrieval results cannot establish adoption precision or downstream task improvement.",
        "A paper-grade geometry claim additionally requires concurrent online controls and both geometry comparisons to pass.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"))
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    report = evaluate(
        _read_jsonl(args.benchmark),
        _read_jsonl(args.gold),
        graph=args.graph,
        index=args.index,
        split=args.split,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"query_count": report["query_count"], "claim_gates": report["claim_gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
