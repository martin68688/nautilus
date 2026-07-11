#!/usr/bin/env python3
"""Compare Tree-only and Stage Hybrid in the novel slot of a fixed three-role Draft protocol."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer, _tokenize


GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
BENCHMARK = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "three_role_draft_benchmark.jsonl"
GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "three_role_draft_gold.jsonl"
REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "three_role_draft_retrieval_evaluation.json"
MARKDOWN = REPO / "coordination" / "three_role_draft_tree_vs_hybrid_report.md"
METHOD_SIMILARITY_THRESHOLD = 0.20
MIN_CLAIM_QUERIES = 20
METHOD_STOPWORDS = {
    "use", "using", "with", "when", "for", "and", "the", "from", "model", "training",
    "data", "task", "classification", "approach", "improve", "avoid", "apply", "ensure",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_bootstrap():
    path = REPO / "paper-skills" / "hyper_memory" / "evaluate_stage_hybrid_retrieval.py"
    spec = importlib.util.spec_from_file_location("stage_hybrid_shared_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.paired_bootstrap


paired_bootstrap = _load_bootstrap()


def _execution_sops(layer: StageAwareHybridMemoryLayer, node_id: str) -> set[str]:
    node = layer.nodes.get(node_id, {})
    transition_ids = []
    if node.get("type") == "Transition":
        transition_ids.append(node_id)
    elif node.get("type") == "RunNode":
        transition_ids.extend(layer._transitions_by_child.get(node_id, []))
        transition_ids.extend(layer._transitions_by_parent.get(node_id, []))
    result = set()
    for transition_id in transition_ids:
        transition = layer.nodes.get(transition_id, {})
        result.update(str(value) for value in transition.get("attached_sop_ids") or [])
    return result


def _candidate_task(layer: StageAwareHybridMemoryLayer, node_id: str) -> str:
    node = layer.nodes.get(node_id, {})
    if node.get("type") == "Transition":
        child = layer.nodes.get(str(node.get("child_node_id") or ""), {})
        return str(child.get("task") or node.get("task") or "")
    return str(node.get("task") or "")


def _method_tokens(layer: StageAwareHybridMemoryLayer, sop_id: str) -> set[str]:
    node = layer.nodes.get(sop_id, {})
    text = " ".join(
        [
            str(node.get("title") or ""),
            str(node.get("action") or ""),
            " ".join(str(value) for value in node.get("applies_when") or []),
            " ".join(str(value) for value in node.get("prevents") or []),
        ]
    )
    return _tokenize(text) - METHOD_STOPWORDS


def _method_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _replay_graph_node_id(layer: StageAwareHybridMemoryLayer, role: dict) -> str | None:
    wanted_run = str(role.get("run_id") or "")
    wanted_original = str(role.get("original_node_id") or "")
    for node_id, node in layer.nodes.items():
        if node.get("type") != "RunNode":
            continue
        if str(node.get("run_id") or "") == wanted_run and str(node.get("original_node_id") or "") == wanted_original:
            return node_id
    return None


def _score_pack(layer: StageAwareHybridMemoryLayer, pack: dict, query: dict, gold: dict) -> dict[str, Any]:
    ranking = [item["id"] for item in pack.get("fused_execution_candidates", [])]
    relevant_sops = set(gold["relevant_sop_ids"])
    candidate_sops = [_execution_sops(layer, node_id) for node_id in ranking]
    gold_method_tokens = {sop_id: _method_tokens(layer, sop_id) for sop_id in relevant_sops}
    candidate_method_tokens = [
        {sop_id: _method_tokens(layer, sop_id) for sop_id in methods}
        for methods in candidate_sops
    ]
    semantic_hits = []
    for methods in candidate_method_tokens:
        best = max(
            (
                _method_similarity(candidate_tokens, gold_tokens)
                for candidate_tokens in methods.values()
                for gold_tokens in gold_method_tokens.values()
            ),
            default=0.0,
        )
        semantic_hits.append(best >= METHOD_SIMILARITY_THRESHOLD)
    first_relevant_rank = next(
        (index for index, hit in enumerate(semantic_hits, 1) if hit),
        None,
    )
    exact_covered_at_5 = set().union(*(candidate_sops[:5] or [set()])) & relevant_sops
    semantic_covered_at_5 = set()
    for gold_sop_id, gold_tokens in gold_method_tokens.items():
        if any(
            _method_similarity(candidate_tokens, gold_tokens) >= METHOD_SIMILARITY_THRESHOLD
            for methods in candidate_method_tokens[:5]
            for candidate_tokens in methods.values()
        ):
            semantic_covered_at_5.add(gold_sop_id)
    tasks_at_5 = [_candidate_task(layer, node_id) for node_id in ranking[:5]]
    task_precision = (
        sum(task == query["task"] for task in tasks_at_5) / len(tasks_at_5)
        if tasks_at_5 else 0.0
    )
    replay_role = query["three_role_protocol"]["memory_reproduction"]
    replay_node_id = _replay_graph_node_id(layer, replay_role)
    return {
        "query_id": query["query_id"],
        "run_id": query["run_id"],
        "method_mrr": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
        "method_recall_at_5": len(semantic_covered_at_5) / len(relevant_sops) if relevant_sops else 0.0,
        "exact_sop_recall_at_5": len(exact_covered_at_5) / len(relevant_sops) if relevant_sops else 0.0,
        "task_precision_at_5": task_precision,
        "cross_task_contamination_at_5": 1.0 - task_precision,
        "replay_overlap_at_5": float(bool(replay_node_id and replay_node_id in ranking[:5])),
        "blocked_positive_count": sum(
            1
            for node_id in ranking
            if (
                layer.nodes.get(node_id, {}).get("type") == "Transition"
                and not layer._positive_transition(node_id)[0]
            )
            or (
                layer.nodes.get(node_id, {}).get("type") == "RunNode"
                and not layer._positive_memory_eligible(layer.nodes[node_id])
            )
        ),
        "ranking": ranking,
        "semantically_covered_gold_sop_ids_at_5": sorted(semantic_covered_at_5),
        "exactly_covered_sop_ids_at_5": sorted(exact_covered_at_5),
        "gold_sop_count": len(relevant_sops),
    }


def evaluate(queries: list[dict], gold_rows: list[dict], *, split: str = "test") -> dict[str, Any]:
    gold_by_id = {row["query_id"]: row for row in gold_rows}
    split_queries = [query for query in queries if query["split"] == split]
    excluded_runs = sorted(query["run_id"] for query in split_queries)
    results = {"tree_only": [], "stage_hybrid": []}
    latencies = {"tree_only": [], "stage_hybrid": []}
    for condition in results:
        layer = StageAwareHybridMemoryLayer(
            graph_path=str(GRAPH),
            index_path=str(INDEX),
            source_name="run_forest_stage_hybrid_memory",
            mode="run_forest_stage_hybrid",
            scoring_mode="poincare",
            retrieval_control=condition,
            excluded_run_ids=excluded_runs,
            enable_agentic=False,
            top_k=10,
            max_chars=0,
        )
        for query in split_queries:
            started = time.perf_counter()
            layer.retrieve_for_node(
                stage="draft",
                task_id=query["task"],
                task_desc=query["query_text"],
                query_parts=[],
            )
            latencies[condition].append(time.perf_counter() - started)
            results[condition].append(
                _score_pack(layer, layer.current_navigation_pack(), query, gold_by_id[query["query_id"]])
            )

    metrics = (
        "method_mrr", "method_recall_at_5", "exact_sop_recall_at_5", "task_precision_at_5",
        "cross_task_contamination_at_5", "replay_overlap_at_5", "blocked_positive_count",
    )
    aggregate = {
        condition: {
            "query_count": len(rows),
            **{metric: float(np.mean([row[metric] for row in rows])) if rows else 0.0 for metric in metrics},
            "latency_sec": float(np.mean(latencies[condition])) if latencies[condition] else 0.0,
        }
        for condition, rows in results.items()
    }
    comparisons = {
        metric: paired_bootstrap(
            [row[metric] for row in results["stage_hybrid"]],
            [row[metric] for row in results["tree_only"]],
        )
        for metric in ("method_mrr", "method_recall_at_5", "task_precision_at_5")
    }
    claim_allowed = (
        len(split_queries) >= MIN_CLAIM_QUERIES
        and comparisons["method_mrr"]["delta"] > 0
        and comparisons["method_mrr"]["p_value"] < 0.05
        and comparisons["method_recall_at_5"]["delta"] >= 0
        and aggregate["stage_hybrid"]["blocked_positive_count"] == 0
    )
    role_protocol = {
        "order": ["coldstart_baseline", "memory_reproduction", "novel_exploration"],
        "coldstart_fixed_across_conditions": True,
        "replay_fixed_across_conditions": True,
        "only_changed_role": "novel_exploration",
        "novel_conditions": ["tree_only", "stage_hybrid"],
        "held_out_memory_run_ids": excluded_runs,
    }
    return {
        "schema": "three_role_draft_retrieval_evaluation_v1",
        "split": split,
        "query_count": len(split_queries),
        "role_protocol": role_protocol,
        "method_similarity": {
            "metric": "Jaccard over SOP title/action/condition/prevention tokens",
            "threshold": METHOD_SIMILARITY_THRESHOLD,
            "threshold_tuned_on_test": False,
        },
        "statistical_adequacy": {
            "minimum_claim_queries": MIN_CLAIM_QUERIES,
            "enough_queries": len(split_queries) >= MIN_CLAIM_QUERIES,
        },
        "controls": aggregate,
        "comparisons_stage_hybrid_minus_tree_only": comparisons,
        "claim_allowed": claim_allowed,
        "claim_limitations": [
            "Cold-start and replay roles are fixed protocol slots; this offline test scores only the novel slot retrieval.",
            "Gold is a multi-SOP method family from all clean root Draft children, not one historical child node.",
            "All runs in the evaluated split are excluded from positive retrieval.",
            f"A superiority claim requires at least {MIN_CLAIM_QUERIES} held-out queries; this split has {len(split_queries)}.",
            "Draft root context does not identify one uniquely correct method, so semantic method-family recall is primary and exact SOP-ID recall is diagnostic only.",
            "Final generated code, training metric, and downstream adoption require an online three-role run.",
        ],
        "per_query": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    tree = report["controls"]["tree_only"]
    hybrid = report["controls"]["stage_hybrid"]
    lines = [
        "# Three-Role Draft: Tree-Only vs Stage Hybrid",
        "",
        f"Split: `{report['split']}`; root episodes: {report['query_count']}.",
        "",
        "The cold-start and replay slots are fixed. Only the novel_exploration retrieval condition changes.",
        "All runs in the evaluated split are excluded from retrieval memory.",
        "",
        "| Novel slot | Method MRR | Semantic method Recall@5 | Exact SOP Recall@5 | Task precision@5 | Cross-task contamination | Replay overlap@5 | Blocked positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Tree-only | {tree['method_mrr']:.4f} | {tree['method_recall_at_5']:.4f} | {tree['exact_sop_recall_at_5']:.4f} | {tree['task_precision_at_5']:.4f} | {tree['cross_task_contamination_at_5']:.4f} | {tree['replay_overlap_at_5']:.4f} | {tree['blocked_positive_count']:.2f} |",
        f"| Stage Hybrid | {hybrid['method_mrr']:.4f} | {hybrid['method_recall_at_5']:.4f} | {hybrid['exact_sop_recall_at_5']:.4f} | {hybrid['task_precision_at_5']:.4f} | {hybrid['cross_task_contamination_at_5']:.4f} | {hybrid['replay_overlap_at_5']:.4f} | {hybrid['blocked_positive_count']:.2f} |",
        "",
        "## Paired Comparisons",
        "",
    ]
    for metric, comparison in report["comparisons_stage_hybrid_minus_tree_only"].items():
        lines.append(f"- `{metric}`: delta `{comparison['delta']:.4f}`, p-value `{comparison['p_value']:.4f}`.")
    development = report.get("development_diagnostic")
    if development:
        dev_tree = development["controls"]["tree_only"]
        dev_hybrid = development["controls"]["stage_hybrid"]
        lines += [
            "",
            "## Development Diagnostic",
            "",
            f"Dev has {development['query_count']} episodes and is diagnostic only.",
            "",
            "| Novel slot | Method MRR | Semantic method Recall@5 | Task precision@5 |",
            "|---|---:|---:|---:|",
            f"| Tree-only | {dev_tree['method_mrr']:.4f} | {dev_tree['method_recall_at_5']:.4f} | {dev_tree['task_precision_at_5']:.4f} |",
            f"| Stage Hybrid | {dev_hybrid['method_mrr']:.4f} | {dev_hybrid['method_recall_at_5']:.4f} | {dev_hybrid['task_precision_at_5']:.4f} |",
        ]
    lines += ["", f"Claim allowed: `{report['claim_allowed']}`", "", "## Limitations", ""]
    lines.extend(f"- {item}" for item in report["claim_limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--report-out", type=Path, default=REPORT)
    parser.add_argument("--markdown-out", type=Path, default=MARKDOWN)
    args = parser.parse_args()
    queries = _read_jsonl(args.benchmark)
    gold = _read_jsonl(args.gold)
    report = evaluate(queries, gold, split=args.split)
    if args.split == "test":
        development = evaluate(queries, gold, split="dev")
        report["development_diagnostic"] = {
            "query_count": development["query_count"],
            "controls": development["controls"],
            "comparisons_stage_hybrid_minus_tree_only": development[
                "comparisons_stage_hybrid_minus_tree_only"
            ],
        }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"query_count": report["query_count"], "controls": report["controls"], "claim_allowed": report["claim_allowed"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
