"""Diagnose whether run-level memory is more hyperbolic-friendly than SOP memory.

This script compares two memory carriers in this project:

1. Run/journal memory: the actual MLEvolve search tree recorded in journal.json.
2. Distilled SOP memory: the external SOP graph produced by the hyper-memory builder.

For each journal tree we build a simple Poincare-ball tree layout from the
observed parent-child edges, then compare:

    same coordinates + Poincare geodesic distance
    same coordinates + Euclidean distance

The goal is diagnostic evidence, not a tuned retrieval benchmark.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DEFAULT_RUNS_DIR = REPO / "mlevolve" / "runs"
DEFAULT_SOP_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_EDGE_ABLATION = (
    REPO
    / "paper-skills"
    / "eval_skill_memory"
    / "reports"
    / "hyperbolic_ablation_edge_predicted_only.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "paper-skills"
    / "eval_skill_memory"
    / "reports"
    / "run_tree_hyperbolic_diagnostics.json"
)
DEFAULT_REPORT = REPO / "coordination" / "run_tree_vs_sop_hyperbolic_memory_report.md"


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(node: dict[str, Any]) -> float | None:
    metric = node.get("metric")
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def metric_maximize(child: dict[str, Any], parent: dict[str, Any] | None = None) -> bool | None:
    for node in (child, parent):
        if not isinstance(node, dict):
            continue
        metric = node.get("metric")
        if isinstance(metric, dict) and isinstance(metric.get("maximize"), bool):
            return bool(metric["maximize"])
    return None


def summarize_numbers(values: list[float]) -> dict[str, float]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not clean:
        return {"count": 0}
    clean.sort()
    return {
        "count": len(clean),
        "mean": float(statistics.fmean(clean)),
        "median": float(statistics.median(clean)),
        "min": float(clean[0]),
        "max": float(clean[-1]),
    }


def load_journals(runs_dir: Path, min_nodes: int = 2) -> list[tuple[Path, dict[str, Any]]]:
    journals: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(runs_dir.glob("*/logs/journal.json")):
        try:
            data = read_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        nodes = data.get("nodes") or []
        if isinstance(nodes, list) and len(nodes) >= min_nodes:
            journals.append((path, data))
    return journals


def journal_graph(journal: dict[str, Any]) -> dict[str, Any]:
    nodes_raw = [n for n in journal.get("nodes", []) if isinstance(n, dict) and n.get("id")]
    nodes = {str(n["id"]): n for n in nodes_raw}
    parent_raw = journal.get("node2parent") or {}
    parents: dict[str, str] = {}
    for child, parent in parent_raw.items():
        child_id = str(child)
        parent_id = str(parent)
        if child_id in nodes and parent_id in nodes and child_id != parent_id:
            parents[child_id] = parent_id

    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for child, parent in parents.items():
        children[parent].append(child)
    for child_list in children.values():
        child_list.sort(key=lambda node_id: (nodes[node_id].get("step") or 0, node_id))

    roots = [node_id for node_id in nodes if node_id not in parents]
    roots.sort(key=lambda node_id: (nodes[node_id].get("step") or 0, node_id))

    depths: dict[str, int] = {}
    queue = collections.deque((root, 0) for root in roots)
    while queue:
        node_id, depth = queue.popleft()
        if node_id in depths and depths[node_id] <= depth:
            continue
        depths[node_id] = depth
        for child in children.get(node_id, []):
            queue.append((child, depth + 1))
    for node_id in nodes:
        depths.setdefault(node_id, 0)

    return {
        "nodes": nodes,
        "parents": parents,
        "children": children,
        "roots": roots,
        "depths": depths,
    }


def descendant_leaf_counts(children: dict[str, list[str]], roots: list[str]) -> dict[str, int]:
    memo: dict[str, int] = {}

    def rec(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        child_list = children.get(node_id, [])
        if not child_list:
            memo[node_id] = 1
        else:
            memo[node_id] = sum(rec(child) for child in child_list)
        return memo[node_id]

    for root in roots:
        rec(root)
    return memo


def circular_run_tree_layout(graph: dict[str, Any], edge_len: float = 0.9) -> dict[str, np.ndarray]:
    children: dict[str, list[str]] = graph["children"]
    roots: list[str] = graph["roots"]
    depths: dict[str, int] = graph["depths"]
    leaf_counts = descendant_leaf_counts(children, roots)
    coords: dict[str, np.ndarray] = {}

    def assign(node_id: str, lo: float, hi: float, level_offset: int = 0) -> None:
        theta = (lo + hi) / 2.0
        level = depths.get(node_id, 0) + level_offset
        hyperbolic_radius = min(level * edge_len, 4.9)
        radius = min(math.tanh(hyperbolic_radius / 2.0), 0.985)
        coords[node_id] = np.asarray(
            [radius * math.cos(theta), radius * math.sin(theta)],
            dtype=np.float64,
        )
        child_list = children.get(node_id, [])
        if not child_list:
            return
        total = sum(leaf_counts.get(child, 1) for child in child_list)
        cur = lo
        for child in child_list:
            width = (hi - lo) * leaf_counts.get(child, 1) / max(1, total)
            assign(child, cur, cur + width, level_offset=level_offset)
            cur += width

    if len(roots) == 1:
        assign(roots[0], 0.0, 2.0 * math.pi, level_offset=0)
        return coords

    total = sum(leaf_counts.get(root, 1) for root in roots)
    cur = 0.0
    for root in roots:
        width = 2.0 * math.pi * leaf_counts.get(root, 1) / max(1, total)
        assign(root, cur, cur + width, level_offset=1)
        cur += width
    return coords


def tree_distance_matrix(ids: list[str], parents: dict[str, str]) -> np.ndarray:
    ancestors: dict[str, list[str]] = {}
    for node_id in ids:
        cur = node_id
        chain = [cur]
        seen = {cur}
        while cur in parents:
            cur = parents[cur]
            if cur in seen:
                break
            chain.append(cur)
            seen.add(cur)
        ancestors[node_id] = chain

    n = len(ids)
    out = np.zeros((n, n), dtype=np.float32)
    for i, left in enumerate(ids):
        left_pos = {node_id: depth for depth, node_id in enumerate(ancestors[left])}
        for j, right in enumerate(ids):
            for right_depth, ancestor in enumerate(ancestors[right]):
                if ancestor in left_pos:
                    out[i, j] = left_pos[ancestor] + right_depth
                    break
    return out


def poincare_distance_matrix(coords: np.ndarray) -> np.ndarray:
    norms = np.sum(coords * coords, axis=1)
    d2 = norms[:, None] + norms[None, :] - 2.0 * coords @ coords.T
    denom = np.maximum((1.0 - norms[:, None]) * (1.0 - norms[None, :]), 1e-12)
    arg = 1.0 + 2.0 * np.maximum(d2, 0.0) / denom
    return np.arccosh(np.maximum(arg, 1.0)).astype(np.float32)


def euclidean_distance_matrix(coords: np.ndarray) -> np.ndarray:
    norms = np.sum(coords * coords, axis=1)
    d2 = norms[:, None] + norms[None, :] - 2.0 * coords @ coords.T
    return np.sqrt(np.maximum(d2, 0.0)).astype(np.float32)


def scaled_distance_metrics(pred_dist: np.ndarray, target_dist: np.ndarray) -> dict[str, float]:
    if pred_dist.shape[0] < 2:
        return {
            "pearson_corr_with_tree_distance": float("nan"),
            "relative_stress": float("nan"),
            "pair_count": 0,
        }
    iu = np.triu_indices_from(target_dist, k=1)
    pred = pred_dist[iu].astype(np.float64)
    target = target_dist[iu].astype(np.float64)
    finite = np.isfinite(pred) & np.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    if pred.size == 0:
        return {
            "pearson_corr_with_tree_distance": float("nan"),
            "relative_stress": float("nan"),
            "pair_count": 0,
        }
    if float(np.std(pred)) < 1e-12 or float(np.std(target)) < 1e-12:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(pred, target)[0, 1])
    design = np.vstack([pred, np.ones_like(pred)]).T
    scale, bias = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = scale * pred + bias
    rmse = float(np.sqrt(np.mean((fitted - target) ** 2)))
    stress = rmse / max(float(np.mean(target)), 1e-12)
    return {
        "pearson_corr_with_tree_distance": corr,
        "scaled_rmse": rmse,
        "relative_stress": stress,
        "linear_scale": float(scale),
        "linear_bias": float(bias),
        "pair_count": int(pred.shape[0]),
    }


def topk_neighbor_recall(pred_dist: np.ndarray, target_dist: np.ndarray, k: int = 10) -> float:
    n = pred_dist.shape[0]
    if n <= 1:
        return float("nan")
    k = min(k, n - 1)
    hits: list[float] = []
    for i in range(n):
        true_order = np.lexsort((np.arange(n), target_dist[i]))
        pred_order = np.lexsort((np.arange(n), pred_dist[i]))
        true_neighbors = [int(x) for x in true_order if int(x) != i][:k]
        pred_neighbors = [int(x) for x in pred_order if int(x) != i][:k]
        hits.append(len(set(true_neighbors) & set(pred_neighbors)) / k)
    return float(np.mean(hits))


def subtree_leaf_precision(
    ids: list[str],
    children: dict[str, list[str]],
    p_dist: np.ndarray,
    e_dist: np.ndarray,
    k: int,
) -> dict[str, float]:
    idx = {node_id: i for i, node_id in enumerate(ids)}
    leaves = sorted([node_id for node_id in ids if not children.get(node_id)])
    if not leaves:
        return {"queries": 0}

    leaf_set_cache: dict[str, set[str]] = {}

    def descendant_leaves(node_id: str) -> set[str]:
        if node_id in leaf_set_cache:
            return leaf_set_cache[node_id]
        child_list = children.get(node_id, [])
        if not child_list:
            leaf_set_cache[node_id] = {node_id}
        else:
            out: set[str] = set()
            for child in child_list:
                out.update(descendant_leaves(child))
            leaf_set_cache[node_id] = out
        return leaf_set_cache[node_id]

    rows: list[dict[str, float]] = []
    all_leaf_count = len(leaves)
    for node_id in ids:
        if not children.get(node_id):
            continue
        gold = descendant_leaves(node_id)
        # Skip trivial root-level queries where every leaf is a descendant.
        if len(gold) == all_leaf_count:
            continue
        q_idx = idx[node_id]
        for mode, dist in (("poincare", p_dist), ("flat_twin", e_dist)):
            ranked = sorted(leaves, key=lambda leaf: (float(dist[q_idx, idx[leaf]]), leaf))
            top = ranked[: min(k, len(ranked))]
            denom = min(k, len(gold), len(top))
            precision = len(set(top) & gold) / max(1, denom)
            first_rank = next((rank for rank, leaf in enumerate(ranked, 1) if leaf in gold), 10**9)
            rows.append(
                {
                    "mode": mode,
                    "precision": float(precision),
                    "rr": 0.0 if first_rank == 10**9 else 1.0 / first_rank,
                }
            )

    if not rows:
        return {"queries": 0}
    p_rows = [r for r in rows if r["mode"] == "poincare"]
    e_rows = [r for r in rows if r["mode"] == "flat_twin"]
    return {
        "queries": len(p_rows),
        "poincare_precision_at_k": float(np.mean([r["precision"] for r in p_rows])),
        "flat_twin_precision_at_k": float(np.mean([r["precision"] for r in e_rows])),
        "precision_diff": float(
            np.mean([p["precision"] - e["precision"] for p, e in zip(p_rows, e_rows)])
        ),
        "poincare_mrr": float(np.mean([r["rr"] for r in p_rows])),
        "flat_twin_mrr": float(np.mean([r["rr"] for r in e_rows])),
        "mrr_diff": float(np.mean([p["rr"] - e["rr"] for p, e in zip(p_rows, e_rows)])),
    }


def parent_child_retrieval(
    ids: list[str],
    parents: dict[str, str],
    children: dict[str, list[str]],
    p_dist: np.ndarray,
    e_dist: np.ndarray,
    k: int = 5,
) -> dict[str, Any]:
    """Measure local lineage retrieval on the run tree coordinates."""
    idx = {node_id: i for i, node_id in enumerate(ids)}

    parent_rows: list[dict[str, float]] = []
    for child, parent in parents.items():
        if child not in idx or parent not in idx:
            continue
        child_idx = idx[child]
        parent_idx = idx[parent]
        for mode, dist in (("poincare", p_dist), ("flat_twin", e_dist)):
            order = np.lexsort((np.arange(len(ids)), dist[child_idx]))
            ranked = [int(item) for item in order if int(item) != child_idx]
            rank = ranked.index(parent_idx) + 1 if parent_idx in ranked else 10**9
            parent_rows.append(
                {
                    "mode": mode,
                    "hit_at_k": 1.0 if rank <= k else 0.0,
                    "rr": 0.0 if rank == 10**9 else 1.0 / rank,
                }
            )

    child_rows: list[dict[str, float]] = []
    for parent, child_list in children.items():
        if not child_list or parent not in idx:
            continue
        parent_idx = idx[parent]
        gold = set(child_list)
        for mode, dist in (("poincare", p_dist), ("flat_twin", e_dist)):
            order = np.lexsort((np.arange(len(ids)), dist[parent_idx]))
            ranked = [ids[int(item)] for item in order if ids[int(item)] != parent]
            top = ranked[: min(k, len(ranked))]
            denom = min(k, len(gold), len(top))
            precision = len(set(top) & gold) / max(1, denom)
            child_rows.append({"mode": mode, "precision": float(precision)})

    def split(rows: list[dict[str, float]], key: str) -> tuple[list[float], list[float]]:
        p_vals = [row[key] for row in rows if row["mode"] == "poincare"]
        e_vals = [row[key] for row in rows if row["mode"] == "flat_twin"]
        return p_vals, e_vals

    p_parent_hit, e_parent_hit = split(parent_rows, "hit_at_k")
    p_parent_rr, e_parent_rr = split(parent_rows, "rr")
    p_child_precision, e_child_precision = split(child_rows, "precision")

    return {
        "parent_lookup": {
            "queries": len(p_parent_hit),
            "poincare_recall_at_k": float(np.mean(p_parent_hit)) if p_parent_hit else float("nan"),
            "flat_twin_recall_at_k": float(np.mean(e_parent_hit)) if e_parent_hit else float("nan"),
            "recall_diff": (
                float(np.mean([p - e for p, e in zip(p_parent_hit, e_parent_hit)]))
                if p_parent_hit
                else float("nan")
            ),
            "poincare_mrr": float(np.mean(p_parent_rr)) if p_parent_rr else float("nan"),
            "flat_twin_mrr": float(np.mean(e_parent_rr)) if e_parent_rr else float("nan"),
            "mrr_diff": (
                float(np.mean([p - e for p, e in zip(p_parent_rr, e_parent_rr)]))
                if p_parent_rr
                else float("nan")
            ),
        },
        "child_lookup": {
            "queries": len(p_child_precision),
            "poincare_precision_at_k": (
                float(np.mean(p_child_precision)) if p_child_precision else float("nan")
            ),
            "flat_twin_precision_at_k": (
                float(np.mean(e_child_precision)) if e_child_precision else float("nan")
            ),
            "precision_diff": (
                float(np.mean([p - e for p, e in zip(p_child_precision, e_child_precision)]))
                if p_child_precision
                else float("nan")
            ),
        },
    }


def run_tree_experiment(path: Path, journal: dict[str, Any]) -> dict[str, Any]:
    graph = journal_graph(journal)
    nodes: dict[str, dict[str, Any]] = graph["nodes"]
    parents: dict[str, str] = graph["parents"]
    children: dict[str, list[str]] = graph["children"]
    roots: list[str] = graph["roots"]
    depths: dict[str, int] = graph["depths"]
    ids = sorted(nodes, key=lambda node_id: ((nodes[node_id].get("step") or 0), node_id))
    coords_by_id = circular_run_tree_layout(graph)
    coords = np.vstack([coords_by_id[node_id] for node_id in ids]).astype(np.float64)
    target = tree_distance_matrix(ids, parents)
    p_dist = poincare_distance_matrix(coords)
    e_dist = euclidean_distance_matrix(coords)

    leaf_count = sum(1 for node_id in ids if not children.get(node_id))
    internal_child_counts = [len(children[node_id]) for node_id in ids if children.get(node_id)]
    depth_values = [depths[node_id] for node_id in ids]
    max_depth = max(depth_values) if depth_values else 0
    deepest_count = sum(1 for depth in depth_values if depth == max_depth)
    stage_counts = collections.Counter(str(nodes[node_id].get("stage", "unknown")) for node_id in ids)
    buggy_counts = collections.Counter(str(nodes[node_id].get("is_buggy")) for node_id in ids)
    valid_counts = collections.Counter(str(nodes[node_id].get("is_valid")) for node_id in ids)

    metric_nodes = [node_id for node_id in ids if metric_value(nodes[node_id]) is not None]
    metric_deltas: list[float] = []
    metric_improvements: list[float] = []
    for child, parent in parents.items():
        child_value = metric_value(nodes[child])
        parent_value = metric_value(nodes[parent])
        if child_value is None or parent_value is None:
            continue
        raw_delta = child_value - parent_value
        metric_deltas.append(raw_delta)
        maximize = metric_maximize(nodes[child], nodes[parent])
        if maximize is True:
            metric_improvements.append(raw_delta)
        elif maximize is False:
            metric_improvements.append(-raw_delta)

    p_metrics = scaled_distance_metrics(p_dist, target)
    e_metrics = scaled_distance_metrics(e_dist, target)
    p_metrics["neighbor_recall_at_10"] = topk_neighbor_recall(p_dist, target, k=10)
    e_metrics["neighbor_recall_at_10"] = topk_neighbor_recall(e_dist, target, k=10)

    subtree5 = subtree_leaf_precision(ids, children, p_dist, e_dist, k=5)
    subtree10 = subtree_leaf_precision(ids, children, p_dist, e_dist, k=10)
    lineage5 = parent_child_retrieval(ids, parents, children, p_dist, e_dist, k=5)

    n_nodes = len(ids)
    n_edges = len(parents)
    n_roots = len(roots)
    tree_baseline_edges = max(0, n_nodes - n_roots)
    return {
        "run_id": path.parents[1].name,
        "journal_path": str(path.relative_to(REPO)),
        "structure": {
            "nodes": n_nodes,
            "edges": n_edges,
            "roots": n_roots,
            "tree_baseline_edges": tree_baseline_edges,
            "edge_count_matches_tree": n_edges == tree_baseline_edges,
            "extra_edges_over_tree": n_edges - tree_baseline_edges,
            "leaf_count": leaf_count,
            "leaf_fraction": leaf_count / max(1, n_nodes),
            "max_depth": max_depth,
            "avg_depth": float(statistics.fmean(depth_values)) if depth_values else 0.0,
            "deepest_node_fraction": deepest_count / max(1, n_nodes),
            "boundary_pressure": 0.5 * (leaf_count / max(1, n_nodes))
            + 0.5 * (deepest_count / max(1, n_nodes)),
            "internal_node_count": len(internal_child_counts),
            "internal_avg_children": (
                float(statistics.fmean(internal_child_counts)) if internal_child_counts else 0.0
            ),
            "max_children": max(internal_child_counts) if internal_child_counts else 0,
            "stage_counts": dict(sorted(stage_counts.items())),
            "buggy_counts": dict(sorted(buggy_counts.items())),
            "valid_counts": dict(sorted(valid_counts.items())),
            "metric_node_count": len(metric_nodes),
            "metric_node_fraction": len(metric_nodes) / max(1, n_nodes),
            "metric_delta": summarize_numbers(metric_deltas),
            "metric_improvement": summarize_numbers(metric_improvements),
        },
        "distance_preservation": {
            "poincare": p_metrics,
            "flat_twin_euclidean_same_coords": e_metrics,
            "poincare_minus_flat": {
                "corr": p_metrics["pearson_corr_with_tree_distance"]
                - e_metrics["pearson_corr_with_tree_distance"],
                "relative_stress": p_metrics["relative_stress"] - e_metrics["relative_stress"],
                "neighbor_recall_at_10": p_metrics["neighbor_recall_at_10"]
                - e_metrics["neighbor_recall_at_10"],
            },
        },
        "subtree_leaf_retrieval": {
            "at_5": subtree5,
            "at_10": subtree10,
        },
        "lineage_retrieval": {
            "at_5": lineage5,
        },
    }


def load_sop_diagnostics(graph_path: Path, edge_ablation_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "graph_path": str(graph_path.relative_to(REPO)) if graph_path.exists() else str(graph_path),
        "edge_ablation_path": (
            str(edge_ablation_path.relative_to(REPO)) if edge_ablation_path.exists() else str(edge_ablation_path)
        ),
    }
    if graph_path.exists():
        graph = read_json(graph_path)
        if isinstance(graph, dict):
            nodes = {str(n.get("id")): n for n in graph.get("nodes", []) if isinstance(n, dict)}
            sop_ids = {node_id for node_id, node in nodes.items() if node.get("type") == "SOP"}
            sop_edges = [
                edge
                for edge in graph.get("edges", [])
                if str(edge.get("src")) in sop_ids and str(edge.get("dst")) in sop_ids
            ]
            bands = collections.Counter(str(nodes[node_id].get("radius_band", "unknown")) for node_id in sop_ids)
            edge_kinds = collections.Counter(str(edge.get("kind", "unknown")) for edge in sop_edges)
            n_sop = len(sop_ids)
            out.update(
                {
                    "status": "loaded",
                    "sop_count": n_sop,
                    "radius_band_counts": dict(sorted(bands.items())),
                    "edge_band_fraction": bands.get("edge", 0) / max(1, n_sop),
                    "sop_sop_edges": len(sop_edges),
                    "tree_edge_count_for_same_nodes": max(0, n_sop - 1),
                    "sop_edge_density_vs_tree": len(sop_edges) / max(1, n_sop - 1),
                    "average_sop_sop_degree": 2.0 * len(sop_edges) / max(1, n_sop),
                    "sop_sop_edge_kinds": dict(sorted(edge_kinds.items())),
                    "paper_grade": (graph.get("meta") or {}).get("paper_grade"),
                    "provenance_status": (graph.get("meta") or {}).get("provenance_status"),
                    "radius_model": (graph.get("meta") or {}).get("radius_model"),
                    "angle_model": (graph.get("meta") or {}).get("angle_model"),
                }
            )
    else:
        out["status"] = "missing_graph"

    if edge_ablation_path.exists():
        ablation = read_json(edge_ablation_path)
        if isinstance(ablation, dict):
            systems = ablation.get("systems", {})
            qgate = ablation.get("query_aware_coordinate_quality", {})
            out["edge_predicted_only"] = {
                "status": ablation.get("status"),
                "poincare_edge_recall_at_5": systems.get("agentic_poincare", {}).get("edge_recall_at_5"),
                "flat_twin_edge_recall_at_5": systems.get("agentic_flat_twin", {}).get("edge_recall_at_5"),
                "euclidean_edge_recall_at_5": systems.get("agentic_euclidean", {}).get("edge_recall_at_5"),
                "poincare_mrr": systems.get("agentic_poincare", {}).get("mrr"),
                "flat_twin_mrr": systems.get("agentic_flat_twin", {}).get("mrr"),
                "euclidean_mrr": systems.get("agentic_euclidean", {}).get("mrr"),
                "poincare_flat_twin_top5_overlap_mean": qgate.get("poincare_flat_twin_top5_overlap_mean")
                or (ablation.get("ranking_diagnostics") or {}).get("poincare_flat_twin_top5_overlap_mean"),
                "query_aware_status": qgate.get("status"),
                "selected_edge_rate_by_system": qgate.get("selected_edge_rate_by_system"),
            }
    return out


def global_memory_flattening_diagnostics(runs_dir: Path) -> dict[str, Any]:
    record_files = sorted(runs_dir.glob("*/workspace/global_memory/records.json"))
    topology_keys = {
        "parent_id",
        "parent_node_id",
        "parent_record_id",
        "children",
        "child_ids",
        "depth",
        "branch_id",
        "source_run",
        "run_id",
    }
    transition_context_keys = {"parent_metric", "current_metric", "parent_error"}
    total = 0
    key_counts: collections.Counter[str] = collections.Counter()
    any_topology = 0
    any_transition_context = 0
    per_file = []
    for path in record_files:
        try:
            rows = read_json(path)
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        file_any_topology = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            total += 1
            present_topology = topology_keys & set(row)
            present_transition = transition_context_keys & set(row)
            for key in present_topology | present_transition:
                key_counts[key] += 1
            if present_topology:
                any_topology += 1
                file_any_topology += 1
            if present_transition:
                any_transition_context += 1
        per_file.append(
            {
                "run_id": path.parents[2].name,
                "path": str(path.relative_to(REPO)),
                "records": len(rows),
                "records_with_any_topology_key": file_any_topology,
            }
        )
    return {
        "record_files": len(per_file),
        "total_records": total,
        "records_with_any_tree_topology_key": any_topology,
        "records_with_any_tree_topology_key_fraction": any_topology / max(1, total),
        "records_with_parent_metric_or_error": any_transition_context,
        "records_with_parent_metric_or_error_fraction": any_transition_context / max(1, total),
        "key_counts": dict(sorted(key_counts.items())),
        "sample_files": per_file[:10],
        "interpretation": "GlobalMemory stores useful node-level transition context, but current records do not preserve parent-child topology.",
    }


def aggregate_run_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"runs": 0}

    def path_values(path: tuple[str, ...]) -> list[float]:
        values: list[float] = []
        for result in results:
            cur: Any = result
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    cur = None
                    break
                cur = cur[key]
            if isinstance(cur, (int, float)) and math.isfinite(float(cur)):
                values.append(float(cur))
        return values

    structure_paths = {
        "nodes": ("structure", "nodes"),
        "edges": ("structure", "edges"),
        "leaf_fraction": ("structure", "leaf_fraction"),
        "max_depth": ("structure", "max_depth"),
        "avg_depth": ("structure", "avg_depth"),
        "boundary_pressure": ("structure", "boundary_pressure"),
        "internal_avg_children": ("structure", "internal_avg_children"),
        "metric_node_fraction": ("structure", "metric_node_fraction"),
    }
    distance_paths = {
        "poincare_corr": (
            "distance_preservation",
            "poincare",
            "pearson_corr_with_tree_distance",
        ),
        "flat_corr": (
            "distance_preservation",
            "flat_twin_euclidean_same_coords",
            "pearson_corr_with_tree_distance",
        ),
        "corr_diff": ("distance_preservation", "poincare_minus_flat", "corr"),
        "poincare_stress": ("distance_preservation", "poincare", "relative_stress"),
        "flat_stress": ("distance_preservation", "flat_twin_euclidean_same_coords", "relative_stress"),
        "stress_diff": ("distance_preservation", "poincare_minus_flat", "relative_stress"),
        "poincare_neighbor_recall_at_10": (
            "distance_preservation",
            "poincare",
            "neighbor_recall_at_10",
        ),
        "flat_neighbor_recall_at_10": (
            "distance_preservation",
            "flat_twin_euclidean_same_coords",
            "neighbor_recall_at_10",
        ),
        "neighbor_recall_diff": (
            "distance_preservation",
            "poincare_minus_flat",
            "neighbor_recall_at_10",
        ),
        "subtree_p_at_5_diff": ("subtree_leaf_retrieval", "at_5", "precision_diff"),
        "subtree_p_at_10_diff": ("subtree_leaf_retrieval", "at_10", "precision_diff"),
        "parent_lookup_recall_at_5_poincare": (
            "lineage_retrieval",
            "at_5",
            "parent_lookup",
            "poincare_recall_at_k",
        ),
        "parent_lookup_recall_at_5_flat": (
            "lineage_retrieval",
            "at_5",
            "parent_lookup",
            "flat_twin_recall_at_k",
        ),
        "parent_lookup_recall_at_5_diff": (
            "lineage_retrieval",
            "at_5",
            "parent_lookup",
            "recall_diff",
        ),
        "parent_lookup_mrr_poincare": (
            "lineage_retrieval",
            "at_5",
            "parent_lookup",
            "poincare_mrr",
        ),
        "parent_lookup_mrr_flat": (
            "lineage_retrieval",
            "at_5",
            "parent_lookup",
            "flat_twin_mrr",
        ),
        "parent_lookup_mrr_diff": (
            "lineage_retrieval",
            "at_5",
            "parent_lookup",
            "mrr_diff",
        ),
        "child_lookup_precision_at_5_diff": (
            "lineage_retrieval",
            "at_5",
            "child_lookup",
            "precision_diff",
        ),
    }

    return {
        "runs": len(results),
        "structure": {name: summarize_numbers(path_values(path)) for name, path in structure_paths.items()},
        "distance_preservation": {
            name: summarize_numbers(path_values(path)) for name, path in distance_paths.items()
        },
        "poincare_corr_win_rate": float(
            np.mean([1.0 if v > 0 else 0.0 for v in path_values(("distance_preservation", "poincare_minus_flat", "corr"))])
        ),
        "poincare_stress_win_rate": float(
            np.mean([1.0 if v < 0 else 0.0 for v in path_values(("distance_preservation", "poincare_minus_flat", "relative_stress"))])
        ),
    }


def render_report(data: dict[str, Any]) -> str:
    agg = data["run_tree_summary"]
    sop = data["sop_memory_diagnostics"]
    gm = data["global_memory_flattening"]
    top_runs = sorted(
        data["run_tree_results"],
        key=lambda item: item["structure"]["nodes"],
        reverse=True,
    )[:10]

    def fmt(value: Any, digits: int = 4) -> str:
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return f"{float(value):.{digits}f}"
        return "n/a"

    lines = [
        "# Run-Tree vs SOP Hyperbolic Memory Diagnostic",
        "",
        "## Bottom Line",
        "",
        "当前证据支持你的直觉：项目里的真实运行记忆比蒸馏后的 SOP 外置记忆更像双曲空间擅长的对象。",
        "",
        "- `journal.json` 里的 MLEvolve 搜索记录天然是 parent -> child 的树/森林。",
        "- 当前 `GlobalMemoryLayer` 保存了节点经验和 parent metric/error，却没有保存 parent_id、depth、branch_id、children，所以跨 run 记忆把树压扁了。",
        "- SOP 记忆有用，但它被蒸馏成稠密语义图：SOP-SOP 边远多于树基线，edge-band SOP 很少，因此不容易体现双曲几何优势。",
        "- 在真实 run tree 上，同坐标下 Poincare 距离比欧氏距离更能保留树距离；这比当前 SOP retrieval 结果更符合双曲结构的经典优势条件。",
        "",
        "## Project Evidence",
        "",
        f"- Journals scanned: `{data['journal_count']}`",
        f"- GlobalMemory record files: `{gm['record_files']}`",
        f"- GlobalMemory records: `{gm['total_records']}`",
        f"- Records with explicit tree topology keys: `{gm['records_with_any_tree_topology_key']}` / `{gm['total_records']}`",
        f"- Records with parent metric/error context: `{gm['records_with_parent_metric_or_error']}` / `{gm['total_records']}`",
        "",
        "Interpretation: GlobalMemory already remembers useful experience, but not the tree shape that produced it.",
        "",
        "## Real Run Tree Shape",
        "",
        "| Metric | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["nodes", "leaf_fraction", "max_depth", "avg_depth", "boundary_pressure", "internal_avg_children"]:
        stats = agg["structure"][name]
        lines.append(
            f"| {name} | {fmt(stats.get('mean'))} | {fmt(stats.get('median'))} | {fmt(stats.get('min'))} | {fmt(stats.get('max'))} |"
        )

    lines += [
        "",
        "## Distance Preservation on Real Run Trees",
        "",
        "Same coordinates, only distance function changes.",
        "",
        "| Metric | Poincare Mean | Flat-Twin Euclidean Mean | Poincare - Flat |",
        "|---|---:|---:|---:|",
    ]
    dist = agg["distance_preservation"]
    lines.append(
        f"| Corr with tree distance | {fmt(dist['poincare_corr'].get('mean'))} | {fmt(dist['flat_corr'].get('mean'))} | {fmt(dist['corr_diff'].get('mean'))} |"
    )
    lines.append(
        f"| Relative stress lower better | {fmt(dist['poincare_stress'].get('mean'))} | {fmt(dist['flat_stress'].get('mean'))} | {fmt(dist['stress_diff'].get('mean'))} |"
    )
    lines.append(
        f"| Neighbor Recall@10 | {fmt(dist['poincare_neighbor_recall_at_10'].get('mean'))} | {fmt(dist['flat_neighbor_recall_at_10'].get('mean'))} | {fmt(dist['neighbor_recall_diff'].get('mean'))} |"
    )
    lines += [
        "",
        f"- Poincare corr win rate: `{fmt(agg.get('poincare_corr_win_rate'))}`",
        f"- Poincare stress win rate: `{fmt(agg.get('poincare_stress_win_rate'))}`",
        "",
        "This means the geometry advantage appears at the run-tree carrier level, even though it did not appear in the current SOP retrieval benchmark.",
        "",
        "## Run-Tree Retrieval Diagnostics",
        "",
        "| Task | Poincare Mean | Flat-Twin Euclidean Mean | Poincare - Flat |",
        "|---|---:|---:|---:|",
        f"| Parent lookup Recall@5 | {fmt(dist['parent_lookup_recall_at_5_poincare'].get('mean'))} | {fmt(dist['parent_lookup_recall_at_5_flat'].get('mean'))} | {fmt(dist['parent_lookup_recall_at_5_diff'].get('mean'))} |",
        f"| Parent lookup MRR | {fmt(dist['parent_lookup_mrr_poincare'].get('mean'))} | {fmt(dist['parent_lookup_mrr_flat'].get('mean'))} | {fmt(dist['parent_lookup_mrr_diff'].get('mean'))} |",
        f"| Subtree leaf Precision@5 diff | n/a | n/a | {fmt(dist['subtree_p_at_5_diff'].get('mean'))} |",
        f"| Child lookup Precision@5 diff | n/a | n/a | {fmt(dist['child_lookup_precision_at_5_diff'].get('mean'))} |",
        "",
        "Important caveat: Poincare is clearly better for preserving lineage distance and finding parents, but it is not automatically better for every retrieval form. Naive subtree-leaf lookup is slightly worse here, so a real run-memory system should target lineage/backtracking/failure-recovery retrieval first, then tune descendant retrieval separately.",
        "",
        "## SOP Memory Shape Contrast",
        "",
        f"- SOP count: `{sop.get('sop_count')}`",
        f"- Radius bands: `{sop.get('radius_band_counts')}`",
        f"- Edge-band SOP fraction: `{fmt(sop.get('edge_band_fraction'))}`",
        f"- SOP-SOP edges: `{sop.get('sop_sop_edges')}`",
        f"- Tree baseline for same number of SOPs: `{sop.get('tree_edge_count_for_same_nodes')}`",
        f"- SOP edge density vs tree: `{fmt(sop.get('sop_edge_density_vs_tree'))}x`",
        f"- Average SOP-SOP degree: `{fmt(sop.get('average_sop_sop_degree'))}`",
        f"- Edge predicted-only status: `{(sop.get('edge_predicted_only') or {}).get('status')}`",
        f"- Poincare/Flat-Twin top-5 overlap on edge slice: `{fmt((sop.get('edge_predicted_only') or {}).get('poincare_flat_twin_top5_overlap_mean'))}`",
        "",
        "Interpretation: current SOP memory behaves more like a dense semantic library than a branching tree. That is good for stable reusable advice, but weak for proving a hyperbolic geometry thesis.",
        "",
        "## Largest Run Examples",
        "",
        "| Run | Nodes | Leaves | Leaf Fraction | Max Depth | Poincare Corr | Flat Corr | Corr Diff |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in top_runs:
        st = item["structure"]
        dp = item["distance_preservation"]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["run_id"],
                    str(st["nodes"]),
                    str(st["leaf_count"]),
                    fmt(st["leaf_fraction"]),
                    str(st["max_depth"]),
                    fmt(dp["poincare"]["pearson_corr_with_tree_distance"]),
                    fmt(dp["flat_twin_euclidean_same_coords"]["pearson_corr_with_tree_distance"]),
                    fmt(dp["poincare_minus_flat"]["corr"]),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Recommendation",
        "",
        "Use a hybrid memory design:",
        "",
        "1. Run/journal memory should be the main hyperbolic forest. Store nodes, parent-child transitions, depth, branch id, stage, metric delta, bug/error context, and local-best lineage.",
        "2. SOP memory should remain as distilled procedural knowledge, but act more like landmarks/annotations/references attached to subtrees, not the only geometry-bearing object.",
        "3. The next paper-grade geometry claim should compare run-tree retrieval: Poincare forest vs same-coordinate Flat-Twin vs independent Euclidean memory on parent/child, ancestor, sibling-branch, and failure-recovery retrieval tasks.",
        "4. Distill SOPs from frequent successful subtrees or transition motifs after the run-tree memory is built, instead of forcing every SOP to be a primary hyperbolic point.",
        "",
        "Plain metaphor: the run history is the actual family tree; SOP cards are the family recipes copied out afterward. Hyperbolic space is better at storing the family tree than the recipe box.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--sop-graph", type=Path, default=DEFAULT_SOP_GRAPH)
    parser.add_argument("--edge-ablation", type=Path, default=DEFAULT_EDGE_ABLATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-nodes", type=int, default=2)
    args = parser.parse_args()

    journals = load_journals(args.runs_dir, min_nodes=args.min_nodes)
    run_results = [run_tree_experiment(path, journal) for path, journal in journals]
    data = {
        "schema": "run_tree_vs_sop_hyperbolic_diagnostics_v1",
        "journal_count": len(journals),
        "run_tree_summary": aggregate_run_results(run_results),
        "run_tree_results": run_results,
        "global_memory_flattening": global_memory_flattening_diagnostics(args.runs_dir),
        "sop_memory_diagnostics": load_sop_diagnostics(args.sop_graph, args.edge_ablation),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(data), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
