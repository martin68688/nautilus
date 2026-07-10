"""Tree-shaped SOP slice experiment for hyperbolic advantage diagnosis.

The previous synthetic-tree check shows that Poincare distance helps when the
data are truly tree-like. This script asks a closer question for our project:
if we take the current certified edge SOPs and force a clean tree slice

    root -> task -> edge_reason -> SOP leaf

do Poincare distances begin to separate from same-coordinate Euclidean
Flat-Twin distances?

This is a diagnostic, not a paper-grade benchmark. The tree labels are derived
from the auto-seeded edge benchmark and should be human-audited before claims.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_EDGE_BENCH = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "hyperbolic_sop_benchmark_edge.jsonl"
DEFAULT_EDGE_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "hyperbolic_sop_gold_edge.jsonl"
DEFAULT_OUTPUT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "sop_tree_slice_experiment.json"
DEFAULT_REPORT = REPO / "coordination" / "sop_tree_slice_hyperbolic_experiment.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_edge_gold(benchmark_path: Path, gold_path: Path) -> list[dict[str, Any]]:
    bench_by_qid = {str(row["query_id"]): row for row in read_jsonl(benchmark_path)}
    rows: list[dict[str, Any]] = []
    for gold_row in read_jsonl(gold_path):
        qid = str(gold_row["query_id"])
        bench = bench_by_qid.get(qid, {})
        for item in gold_row.get("gold_sops", []) or []:
            if item.get("relevance") not in {"required", "helpful", "risk_warning"}:
                continue
            rows.append({
                "query_id": qid,
                "sop_id": str(item["sop_id"]),
                "task": str(bench.get("task_type", "unknown")),
                "edge_reason": str(item.get("edge_reason") or bench.get("edge_reason") or "edge"),
                "query_kind": str(bench.get("query_kind", "")),
                "split": str(bench.get("split", "")),
            })
    return rows


def circular_tree_layout(paths: list[tuple[str, str, str]]) -> dict[str, Any]:
    """Assign Poincare-ball 2D coordinates to root/task/reason/SOP nodes."""
    by_task: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for task, reason, sop_id in sorted(paths):
        if sop_id not in by_task[task][reason]:
            by_task[task][reason].append(sop_id)
    for reason_map in by_task.values():
        for sops in reason_map.values():
            sops.sort()

    tree_nodes: dict[str, dict[str, Any]] = {
        "root": {"id": "root", "type": "root", "parent": "", "level": 0, "children": []}
    }
    leaf_counts: dict[str, int] = {"root": len({s for _t, _r, s in paths})}
    for task, reason_map in sorted(by_task.items()):
        task_id = f"task::{task}"
        tree_nodes[task_id] = {"id": task_id, "type": "task", "label": task, "parent": "root", "level": 1, "children": []}
        tree_nodes["root"]["children"].append(task_id)
        leaf_counts[task_id] = sum(len(sops) for sops in reason_map.values())
        for reason, sops in sorted(reason_map.items()):
            reason_id = f"reason::{task}::{reason}"
            tree_nodes[reason_id] = {
                "id": reason_id,
                "type": "reason",
                "label": reason,
                "task": task,
                "parent": task_id,
                "level": 2,
                "children": [],
            }
            tree_nodes[task_id]["children"].append(reason_id)
            leaf_counts[reason_id] = len(sops)
            for sop_id in sops:
                leaf_id = f"sop::{sop_id}"
                tree_nodes[leaf_id] = {
                    "id": leaf_id,
                    "type": "SOP",
                    "sop_id": sop_id,
                    "task": task,
                    "edge_reason": reason,
                    "parent": reason_id,
                    "level": 3,
                    "children": [],
                }
                tree_nodes[reason_id]["children"].append(leaf_id)
                leaf_counts[leaf_id] = 1

    coords: dict[str, np.ndarray] = {}
    angle_spans: dict[str, tuple[float, float]] = {}

    def assign(node_id: str, lo: float, hi: float) -> None:
        node = tree_nodes[node_id]
        angle_spans[node_id] = (lo, hi)
        theta = (lo + hi) / 2.0
        level = int(node["level"])
        # Edge leaves sit close to the boundary, where hyperbolic angular
        # separation matters most.
        hyperbolic_radius = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.4}.get(level, float(level))
        radius = math.tanh(hyperbolic_radius / 2.0)
        coords[node_id] = np.asarray([radius * math.cos(theta), radius * math.sin(theta)], dtype=np.float64)
        children = list(node.get("children", []))
        if not children:
            return
        total = sum(leaf_counts[c] for c in children)
        cur = lo
        for child in children:
            width = (hi - lo) * leaf_counts[child] / max(1, total)
            assign(child, cur, cur + width)
            cur += width

    assign("root", 0.0, 2.0 * math.pi)
    return {
        "nodes": tree_nodes,
        "coords": coords,
        "angle_spans": angle_spans,
        "leaf_counts": leaf_counts,
        "by_task": {task: dict(reason_map) for task, reason_map in by_task.items()},
    }


def poincare_distance(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.dot(u, u))
    nv = float(np.dot(v, v))
    d2 = float(np.dot(u - v, u - v))
    arg = 1.0 + 2.0 * d2 / max((1.0 - nu) * (1.0 - nv), 1e-12)
    return math.acosh(max(1.0, arg))


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))


def tree_distance(nodes: dict[str, dict[str, Any]], a: str, b: str) -> int:
    def ancestors(x: str) -> list[str]:
        out = []
        while x:
            out.append(x)
            x = str(nodes[x].get("parent", ""))
        return out

    aa = ancestors(a)
    pos = {node: idx for idx, node in enumerate(aa)}
    for j, node in enumerate(ancestors(b)):
        if node in pos:
            return pos[node] + j
    return len(aa) + len(ancestors(b))


def distance_preservation(layout: dict[str, Any]) -> dict[str, Any]:
    nodes = layout["nodes"]
    coords = layout["coords"]
    ids = sorted(nodes)
    pairs: list[tuple[float, float, float]] = []
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            t = float(tree_distance(nodes, left, right))
            pairs.append((t, poincare_distance(coords[left], coords[right]), euclidean_distance(coords[left], coords[right])))
    arr = np.asarray(pairs, dtype=np.float64)

    def metrics(col: int) -> dict[str, float]:
        target = arr[:, 0]
        pred = arr[:, col]
        design = np.vstack([pred, np.ones_like(pred)]).T
        scale, bias = np.linalg.lstsq(design, target, rcond=None)[0]
        fitted = scale * pred + bias
        return {
            "corr_with_tree_distance": float(np.corrcoef(pred, target)[0, 1]),
            "relative_stress": float(np.sqrt(np.mean((fitted - target) ** 2)) / np.mean(target)),
            "scale": float(scale),
            "bias": float(bias),
            "pair_count": int(arr.shape[0]),
        }

    return {
        "poincare": metrics(1),
        "flat_twin_euclidean": metrics(2),
    }


def branch_retrieval(layout: dict[str, Any], edge_rows: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = layout["nodes"]
    coords = layout["coords"]
    sop_leaf_ids = [nid for nid, n in nodes.items() if n.get("type") == "SOP"]
    sop_leaf_ids.sort()

    def evaluate(mode: str, query_node: str, gold_sop_id: str, k: int = 5) -> tuple[float, float, int]:
        q = coords[query_node]
        if mode == "poincare":
            ranked = sorted(sop_leaf_ids, key=lambda nid: (poincare_distance(q, coords[nid]), nid))
        else:
            ranked = sorted(sop_leaf_ids, key=lambda nid: (euclidean_distance(q, coords[nid]), nid))
        gold_leaf = f"sop::{gold_sop_id}"
        rank = ranked.index(gold_leaf) + 1 if gold_leaf in ranked else 10**9
        recall = 1.0 if rank <= k else 0.0
        rr = 1.0 / rank if rank < 10**9 else 0.0
        return recall, rr, rank

    rows = []
    for item in edge_rows:
        reason_node = f"reason::{item['task']}::{item['edge_reason']}"
        task_node = f"task::{item['task']}"
        if reason_node not in nodes:
            continue
        for query_level, query_node in [("reason_parent", reason_node), ("task_parent", task_node)]:
            p_recall, p_rr, p_rank = evaluate("poincare", query_node, item["sop_id"])
            e_recall, e_rr, e_rank = evaluate("flat", query_node, item["sop_id"])
            rows.append({
                "query_id": item["query_id"],
                "sop_id": item["sop_id"],
                "query_level": query_level,
                "task": item["task"],
                "edge_reason": item["edge_reason"],
                "poincare_recall_at_5": p_recall,
                "flat_recall_at_5": e_recall,
                "poincare_rr": p_rr,
                "flat_rr": e_rr,
                "poincare_rank": p_rank,
                "flat_rank": e_rank,
            })

    def aggregate(query_level: str) -> dict[str, Any]:
        subset = [r for r in rows if r["query_level"] == query_level]
        if not subset:
            return {}
        return {
            "queries": len(subset),
            "poincare_recall_at_5": float(np.mean([r["poincare_recall_at_5"] for r in subset])),
            "flat_recall_at_5": float(np.mean([r["flat_recall_at_5"] for r in subset])),
            "recall_diff": float(np.mean([r["poincare_recall_at_5"] - r["flat_recall_at_5"] for r in subset])),
            "poincare_mrr": float(np.mean([r["poincare_rr"] for r in subset])),
            "flat_mrr": float(np.mean([r["flat_rr"] for r in subset])),
            "mrr_diff": float(np.mean([r["poincare_rr"] - r["flat_rr"] for r in subset])),
            "poincare_mean_rank": float(np.mean([r["poincare_rank"] for r in subset])),
            "flat_mean_rank": float(np.mean([r["flat_rank"] for r in subset])),
        }

    return {
        "by_query_level": {
            "reason_parent": aggregate("reason_parent"),
            "task_parent": aggregate("task_parent"),
        },
        "rows": rows,
    }


def render_report(result: dict[str, Any]) -> str:
    dist = result["distance_preservation"]
    branch = result["branch_retrieval"]["by_query_level"]
    lines = [
        "# SOP Tree Slice Hyperbolic Experiment",
        "",
        "## Setup",
        "",
        "This diagnostic forces the current edge SOPs into a clean tree slice: `root -> task -> edge_reason -> SOP`.",
        "It checks whether Poincare begins to separate from same-coordinate Euclidean distance when the SOP memory is made more tree-like.",
        "",
        f"- Unique edge SOP leaves: {result['tree_summary']['sop_leaves']}",
        f"- Tasks: {result['tree_summary']['tasks']}",
        f"- Reason nodes: {result['tree_summary']['reason_nodes']}",
        f"- Total tree nodes: {result['tree_summary']['tree_nodes']}",
        "",
        "## Distance Preservation",
        "",
        "| Metric | Poincare | Flat-Twin Euclidean |",
        "|---|---:|---:|",
        f"| Corr with tree distance | {dist['poincare']['corr_with_tree_distance']:.4f} | {dist['flat_twin_euclidean']['corr_with_tree_distance']:.4f} |",
        f"| Relative stress | {dist['poincare']['relative_stress']:.4f} | {dist['flat_twin_euclidean']['relative_stress']:.4f} |",
        "",
        "## Branch Retrieval",
        "",
        "| Query Level | Poincare R@5 | Flat R@5 | Diff | Poincare MRR | Flat MRR | Diff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for level, metrics in branch.items():
        if not metrics:
            continue
        lines.append(
            f"| {level} | {metrics['poincare_recall_at_5']:.4f} | {metrics['flat_recall_at_5']:.4f} | {metrics['recall_diff']:.4f} "
            f"| {metrics['poincare_mrr']:.4f} | {metrics['flat_mrr']:.4f} | {metrics['mrr_diff']:.4f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "If Poincare wins distance preservation here but does not win branch retrieval, the geometry can encode the tree but the query/gold task is under-specified or saturated.",
        "If Poincare does not even win distance preservation on this tree slice, the constructed SOP hierarchy is too small/unbalanced to reproduce the literature advantage.",
        "",
    ]
    return "\n".join(lines)


def run(*, graph_path: Path, benchmark_path: Path, gold_path: Path, output_path: Path, report_path: Path) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = {str(n["id"]): n for n in graph.get("nodes", [])}
    edge_rows_raw = load_edge_gold(benchmark_path, gold_path)
    # Unique SOP path labels are enough for the tree, but keep all query rows for
    # branch retrieval so variants still count as benchmark pressure.
    paths = sorted({
        (row["task"], row["edge_reason"], row["sop_id"])
        for row in edge_rows_raw
        if row["sop_id"] in nodes
    })
    layout = circular_tree_layout(paths)
    preservation = distance_preservation(layout)
    retrieval = branch_retrieval(layout, [r for r in edge_rows_raw if r["sop_id"] in nodes])
    tree_nodes = layout["nodes"]
    result = {
        "status": "completed",
        "inputs": {
            "graph": str(graph_path),
            "benchmark": str(benchmark_path),
            "gold": str(gold_path),
        },
        "tree_summary": {
            "sop_leaves": sum(1 for n in tree_nodes.values() if n.get("type") == "SOP"),
            "tasks": sum(1 for n in tree_nodes.values() if n.get("type") == "task"),
            "reason_nodes": sum(1 for n in tree_nodes.values() if n.get("type") == "reason"),
            "tree_nodes": len(tree_nodes),
            "paths": len(paths),
            "queries": len(edge_rows_raw),
        },
        "distance_preservation": preservation,
        "branch_retrieval": retrieval,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(result) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SOP tree slice hyperbolic diagnostic.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_EDGE_BENCH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_EDGE_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = run(
        graph_path=args.graph,
        benchmark_path=args.benchmark,
        gold_path=args.gold,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "report": str(args.report),
        "tree_summary": result["tree_summary"],
        "distance_preservation": result["distance_preservation"],
        "branch_retrieval": result["branch_retrieval"]["by_query_level"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
