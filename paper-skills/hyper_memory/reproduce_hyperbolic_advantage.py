"""Reproduce the canonical setting where hyperbolic geometry should help.

This diagnostic is deliberately separate from the SOP retrieval runtime. It asks:

1. On a clean tree-like hierarchy, does Poincare distance preserve graph/tree
   distance better than Euclidean distance on the same Poincare-ball coordinates?
2. How does a learned Euclidean layout behave as dimensionality increases?
3. Which of those "hyperbolic-friendly" conditions are missing from the current
   SOP memory artifact?

The goal is not to tune the SOP system. It is a sanity check against the
literature claim: hyperbolic space is most useful for low-dimensional,
tree-like, exponentially branching hierarchies.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.manifold import MDS


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_EDGE_ABLATION = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_ablation_edge_predicted_only.json"
DEFAULT_OUTPUT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_advantage_reproduction.json"
DEFAULT_REPORT = REPO / "coordination" / "hyperbolic_advantage_diagnosis.md"


def build_tree(branching: int, depth: int, edge_len: float = 1.0) -> dict[str, Any]:
    ids: list[str] = []
    parents: list[int] = []
    levels: list[int] = []
    coords: list[list[float]] = []

    def rec(prefix: str, level: int, lo: float, hi: float, parent: int) -> None:
        idx = len(ids)
        ids.append(prefix or "root")
        parents.append(parent)
        levels.append(level)
        theta = (lo + hi) / 2.0
        hyperbolic_radius = level * edge_len
        poincare_norm = math.tanh(hyperbolic_radius / 2.0)
        coords.append([poincare_norm * math.cos(theta), poincare_norm * math.sin(theta)])
        if level >= depth:
            return
        width = (hi - lo) / branching
        for child in range(branching):
            rec(f"{prefix}{child}", level + 1, lo + child * width, lo + (child + 1) * width, idx)

    rec("", 0, 0.0, 2.0 * math.pi, -1)
    return {
        "ids": ids,
        "parents": np.asarray(parents, dtype=np.int32),
        "levels": np.asarray(levels, dtype=np.int32),
        "poincare": np.asarray(coords, dtype=np.float64),
        "branching": branching,
        "depth": depth,
        "edge_len": edge_len,
    }


def tree_distance_matrix(parents: np.ndarray) -> np.ndarray:
    n = int(parents.shape[0])
    ancestors: list[list[int]] = []
    for i in range(n):
        cur: list[int] = []
        j = i
        while j >= 0:
            cur.append(j)
            j = int(parents[j])
        ancestors.append(cur)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        lookup = {node: depth for depth, node in enumerate(ancestors[i])}
        for j in range(n):
            for depth_j, node in enumerate(ancestors[j]):
                if node in lookup:
                    dist[i, j] = lookup[node] + depth_j
                    break
    return dist


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
    iu = np.triu_indices_from(target_dist, k=1)
    pred = pred_dist[iu].astype(np.float64)
    target = target_dist[iu].astype(np.float64)
    finite = np.isfinite(pred) & np.isfinite(target)
    pred = pred[finite]
    target = target[finite]
    design = np.vstack([pred, np.ones_like(pred)]).T
    scale, bias = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = scale * pred + bias
    rmse = float(np.sqrt(np.mean((fitted - target) ** 2)))
    stress = rmse / float(np.mean(target))
    corr = float(np.corrcoef(pred, target)[0, 1])
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
    hits = []
    for i in range(n):
        true_order = np.lexsort((np.arange(n), target_dist[i]))
        pred_order = np.lexsort((np.arange(n), pred_dist[i]))
        true_neighbors = [int(x) for x in true_order if int(x) != i][:k]
        pred_neighbors = [int(x) for x in pred_order if int(x) != i][:k]
        hits.append(len(set(true_neighbors) & set(pred_neighbors)) / k)
    return float(np.mean(hits))


def evaluate_tree_layout(branching: int, depth: int, edge_len: float = 1.0) -> dict[str, Any]:
    tree = build_tree(branching=branching, depth=depth, edge_len=edge_len)
    target = tree_distance_matrix(tree["parents"])
    p_dist = poincare_distance_matrix(tree["poincare"])
    e_dist = euclidean_distance_matrix(tree["poincare"])
    n = int(target.shape[0])
    leaves = int(np.sum(tree["levels"] == depth))
    return {
        "branching": branching,
        "depth": depth,
        "nodes": n,
        "leaves": leaves,
        "leaf_fraction": leaves / n,
        "systems": {
            "poincare_2d_tree_layout": {
                **scaled_distance_metrics(p_dist, target),
                "neighbor_recall_at_10": topk_neighbor_recall(p_dist, target, k=10),
            },
            "flat_twin_euclidean_same_coords": {
                **scaled_distance_metrics(e_dist, target),
                "neighbor_recall_at_10": topk_neighbor_recall(e_dist, target, k=10),
            },
        },
    }


def evaluate_euclidean_mds(branching: int, depth: int, dims: list[int], edge_len: float = 1.0) -> dict[str, Any]:
    tree = build_tree(branching=branching, depth=depth, edge_len=edge_len)
    target = tree_distance_matrix(tree["parents"])
    p_dist = poincare_distance_matrix(tree["poincare"])
    e_same = euclidean_distance_matrix(tree["poincare"])
    systems: dict[str, Any] = {
        "poincare_2d_tree_layout": {
            **scaled_distance_metrics(p_dist, target),
            "neighbor_recall_at_10": topk_neighbor_recall(p_dist, target, k=10),
        },
        "flat_twin_euclidean_same_coords": {
            **scaled_distance_metrics(e_same, target),
            "neighbor_recall_at_10": topk_neighbor_recall(e_same, target, k=10),
        },
    }
    for dim in dims:
        model = MDS(
            n_components=dim,
            dissimilarity="precomputed",
            random_state=42,
            n_init=1,
            max_iter=160,
            normalized_stress="auto",
        )
        emb = model.fit_transform(target)
        dist = euclidean_distance_matrix(emb)
        systems[f"euclidean_mds_{dim}d"] = {
            **scaled_distance_metrics(dist, target),
            "neighbor_recall_at_10": topk_neighbor_recall(dist, target, k=10),
            "mds_stress": float(getattr(model, "stress_", float("nan"))),
        }
    return {
        "branching": branching,
        "depth": depth,
        "nodes": int(target.shape[0]),
        "leaves": int(np.sum(tree["levels"] == depth)),
        "leaf_fraction": float(np.mean(tree["levels"] == depth)),
        "systems": systems,
    }


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def sop_graph_diagnostics(graph_path: Path, edge_ablation_path: Path) -> dict[str, Any]:
    graph = load_json(graph_path)
    ablation = load_json(edge_ablation_path)
    if not graph:
        return {"status": "missing_graph", "graph": str(graph_path)}
    nodes = {str(n["id"]): n for n in graph.get("nodes", [])}
    sop_ids = {nid for nid, n in nodes.items() if n.get("type") == "SOP"}
    sop_edges = [
        e for e in graph.get("edges", [])
        if str(e.get("src")) in sop_ids and str(e.get("dst")) in sop_ids
    ]
    bands: dict[str, int] = {}
    for sid in sop_ids:
        band = str(nodes[sid].get("radius_band", "unknown"))
        bands[band] = bands.get(band, 0) + 1
    n_sop = len(sop_ids)
    tree_edge_count = max(0, n_sop - 1)
    diag: dict[str, Any] = {
        "status": "loaded",
        "sop_count": n_sop,
        "radius_band_counts": dict(sorted(bands.items())),
        "edge_band_fraction": bands.get("edge", 0) / max(1, n_sop),
        "sop_sop_edges": len(sop_edges),
        "tree_edge_count_for_same_nodes": tree_edge_count,
        "sop_edge_density_vs_tree": len(sop_edges) / max(1, tree_edge_count),
        "average_sop_sop_degree": 2.0 * len(sop_edges) / max(1, n_sop),
        "sop_sop_edge_kinds": {},
    }
    for edge in sop_edges:
        kind = str(edge.get("kind", "unknown"))
        diag["sop_sop_edge_kinds"][kind] = diag["sop_sop_edge_kinds"].get(kind, 0) + 1
    diag["sop_sop_edge_kinds"] = dict(sorted(diag["sop_sop_edge_kinds"].items()))
    if ablation:
        q = ablation.get("query_aware_coordinate_quality", {})
        diag["edge_predicted_only"] = {
            "query_aware_status": q.get("status"),
            "poincare_flat_twin_top5_overlap": q.get("poincare_flat_twin_edge_top5_overlap_mean"),
            "gold_edge_pressure_mean": q.get("gold_edge_pressure_mean"),
            "selected_edge_rate_by_system": q.get("selected_edge_rate_by_system"),
            "edge_claim": ablation.get("comparisons", {}).get("poincare_vs_flat_twin_edge_claim", {}),
        }
    return diag


def render_report(result: dict[str, Any]) -> str:
    large = result["synthetic_tree_large"]
    mds = result["synthetic_tree_mds"]
    sop = result["sop_graph_diagnostics"]

    def row(name: str, metrics: dict[str, Any]) -> str:
        return (
            f"| {name} | {metrics.get('pearson_corr_with_tree_distance', 'n/a'):.4f} "
            f"| {metrics.get('relative_stress', 'n/a'):.4f} "
            f"| {metrics.get('neighbor_recall_at_10', 'n/a'):.4f} |"
        )

    lines = [
        "# Hyperbolic Advantage Diagnosis",
        "",
        "## Question",
        "",
        "Why does the current SOP memory not show a Poincare advantage, and is the hyperbolic structure useless?",
        "",
        "Short answer: hyperbolic geometry is not useless, but it helps mainly when the stored metric is tree-like, low-dimensional, and has many boundary leaves. The current SOP graph is middle-heavy, dense, TF-IDF-directed, and Poincare/Flat-Twin top-k overlap is very high.",
        "",
        "## Synthetic Tree Reproduction",
        "",
        f"Large tree: branching={large['branching']}, depth={large['depth']}, nodes={large['nodes']}, leaf_fraction={large['leaf_fraction']:.3f}.",
        "",
        "| System | Corr with Tree Distance | Relative Stress | Neighbor Recall@10 |",
        "|---|---:|---:|---:|",
        row("Poincare 2D tree layout", large["systems"]["poincare_2d_tree_layout"]),
        row("Euclidean same coordinates", large["systems"]["flat_twin_euclidean_same_coords"]),
        "",
        "This reproduces the canonical advantage condition: the same radial/angular coordinates preserve tree distances much better when measured with Poincare distance than with Euclidean distance.",
        "",
        "## Euclidean Dimensionality Check",
        "",
        f"Smaller tree for MDS: branching={mds['branching']}, depth={mds['depth']}, nodes={mds['nodes']}, leaf_fraction={mds['leaf_fraction']:.3f}.",
        "",
        "| System | Corr with Tree Distance | Relative Stress | Neighbor Recall@10 |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in mds["systems"].items():
        lines.append(row(name, metrics))
    lines += [
        "",
        "Interpretation: hyperbolic 2D beats Euclidean 2D/same-coordinate on tree geometry, but sufficiently high-dimensional Euclidean MDS can catch up or exceed it. This is why our 16D SOP setup should not be expected to show a free win unless the graph is strongly hierarchical and the query actually uses that hierarchy.",
        "",
        "## Current SOP Graph Mismatch",
        "",
        f"- SOP count: {sop.get('sop_count')}",
        f"- Radius bands: {sop.get('radius_band_counts')}",
        f"- Edge-band SOP fraction: {sop.get('edge_band_fraction'):.3f}",
        f"- SOP-SOP edges: {sop.get('sop_sop_edges')} vs tree baseline {sop.get('tree_edge_count_for_same_nodes')} edges",
        f"- SOP-SOP edge density relative to a tree: {sop.get('sop_edge_density_vs_tree'):.2f}x",
        f"- Average SOP-SOP degree: {sop.get('average_sop_sop_degree'):.2f}",
        f"- SOP-SOP edge kinds: {sop.get('sop_sop_edge_kinds')}",
        "",
    ]
    edge = sop.get("edge_predicted_only", {})
    if edge:
        rates = edge.get("selected_edge_rate_by_system", {})
        claim = edge.get("edge_claim", {})
        rr = claim.get("edge_recall_at_5", {})
        lines += [
            "## Current Edge Retrieval Diagnostics",
            "",
            f"- Query-aware status: {edge.get('query_aware_status')}",
            f"- Poincare/Flat-Twin edge top-5 overlap: {edge.get('poincare_flat_twin_top5_overlap')}",
            f"- Gold edge pressure: {edge.get('gold_edge_pressure_mean')}",
            f"- Selected edge rate by system: {rates}",
            f"- Edge Recall@5 diff Poincare-FlatTwin: {rr.get('observed_mean_diff')} with p={rr.get('p_value')}",
            "",
        ]
    lines += [
        "## Diagnosis",
        "",
        "1. The literature advantage is a geometry-match advantage, not a magic retrieval bonus. It appears when the data metric looks like an exponentially branching tree.",
        "2. Our current SOP graph is not tree-like enough: it has only a small edge band and many SOP-SOP co-occur/enhance/prereq/refines edges, so it is much denser than a tree.",
        "3. Query routing still does not push hard edge queries far enough outward; on edge-only gold, Poincare selected edge SOPs at a much lower rate than the gold pressure.",
        "4. The direction model is TF-IDF-SVD fallback, not sentence embedding or contrastive projection; short abstract failure clues therefore do not reliably point into the right angular sector.",
        "5. Because Poincare and Flat-Twin retrieve almost the same top-5, the experiment is geometry-null: the distance function is not being given a structurally different candidate frontier.",
        "",
        "## Next Reproduction Target",
        "",
        "To make the SOP experiment resemble the successful literature setting, build a small claim-grade slice with explicit Skill -> family -> condition -> edge SOP tree labels, train/derive angular sectors from those labels or sentence embeddings, force edge queries to evaluate boundary retrieval, and then rerun Poincare vs Flat-Twin. If Poincare still ties there, the thesis is in real trouble; if it wins only there, the paper claim must be scoped to tree-like procedural memory.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce hyperbolic advantage conditions and diagnose SOP mismatch.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--edge-ablation", type=Path, default=DEFAULT_EDGE_ABLATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    result = {
        "status": "completed",
        "synthetic_tree_large": evaluate_tree_layout(branching=4, depth=5, edge_len=1.0),
        "synthetic_tree_mds": evaluate_euclidean_mds(branching=3, depth=5, dims=[2, 8, 16], edge_len=1.0),
        "sop_graph_diagnostics": sop_graph_diagnostics(args.graph, args.edge_ablation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(result) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "report": str(args.report),
        "large_tree_nodes": result["synthetic_tree_large"]["nodes"],
        "sop_edge_band_fraction": result["sop_graph_diagnostics"].get("edge_band_fraction"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
