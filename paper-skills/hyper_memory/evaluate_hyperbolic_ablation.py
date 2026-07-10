"""Evaluate Agentic Poincare vs same-coordinate Flat-Twin results.

Two input modes are supported:
1. Legacy JSON list:
   [
     {
       "query_id": "...",
       "poincare": {"rare_recall_at_5": 0.4, "condition_precision": 0.8},
       "flat_twin": {"rare_recall_at_5": 0.2, "condition_precision": 0.8}
     }
   ]
2. Runner JSONL from paper-skills/eval_skill_memory/run_hyperbolic_retrieval_benchmark.py
   plus --gold and --graph.

The geometry claim passes only if:
  1. Rare Recall@5 improves by at least 5 percentage points,
  2. paired bootstrap p < 0.05,
  3. Condition Precision does not decrease,
  4. provenance and coordinate quality are claim-grade.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")
STOP = {"when", "with", "that", "this", "into", "from", "using", "and", "the", "data", "model", "training"}
RELEVANCE_GAIN = {"required": 3.0, "helpful": 2.0, "risk_warning": 2.0}


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """One-sided paired bootstrap for mean(left - right) > 0."""
    if left.shape != right.shape:
        raise ValueError("paired bootstrap arrays must have the same shape")
    if left.ndim != 1 or left.size == 0:
        raise ValueError("paired bootstrap needs a non-empty 1D array")
    diffs = left.astype(float) - right.astype(float)
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    samples = diffs[idx].mean(axis=1)
    p_value = float((np.sum(samples <= 0.0) + 1) / (n_resamples + 1))
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return {
        "observed_mean_diff": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
        "n_pairs": int(diffs.size),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def evaluate_pair(
    results: list[dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
    left_label: str,
    right_label: str,
    status_prefix: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate already-aggregated paired rows for left > right."""
    if not results:
        raise ValueError("empty results")
    left_rr = np.asarray([float(x[left_key]["rare_recall_at_5"]) for x in results], dtype=float)
    right_rr = np.asarray([float(x[right_key]["rare_recall_at_5"]) for x in results], dtype=float)
    left_cp = np.asarray([float(x[left_key]["condition_precision"]) for x in results], dtype=float)
    right_cp = np.asarray([float(x[right_key]["condition_precision"]) for x in results], dtype=float)

    rr_mask = np.isfinite(left_rr) & np.isfinite(right_rr)
    if not np.any(rr_mask):
        raise ValueError("no paired queries with rare gold labels for Rare Recall@5")
    rr_bootstrap = paired_bootstrap(left_rr[rr_mask], right_rr[rr_mask], n_resamples=n_resamples, seed=seed)
    rr_diff = rr_bootstrap["observed_mean_diff"]
    cp_diff = float(np.mean(left_cp - right_cp))
    passed = rr_diff >= 0.05 and rr_bootstrap["p_value"] < 0.05 and cp_diff >= 0.0
    rank_metrics: dict[str, Any] = {}
    for metric in ["exact_recall_at_1", "exact_recall_at_3", "exact_recall_at_5", "mrr", "ndcg_at_5"]:
        if all(metric in x[left_key] and metric in x[right_key] for x in results):
            left_vals = np.asarray([float(x[left_key][metric]) for x in results], dtype=float)
            right_vals = np.asarray([float(x[right_key][metric]) for x in results], dtype=float)
            mask = np.isfinite(left_vals) & np.isfinite(right_vals)
            if np.any(mask):
                boot = paired_bootstrap(left_vals[mask], right_vals[mask], n_resamples=n_resamples, seed=seed)
                rank_metrics[metric] = {
                    **boot,
                    f"{left_key}_mean": float(np.mean(left_vals[mask])),
                    f"{right_key}_mean": float(np.mean(right_vals[mask])),
                }
    rank_not_lower = True
    for metric in ("mrr", "ndcg_at_5"):
        if metric in rank_metrics:
            rank_not_lower = rank_not_lower and rank_metrics[metric]["observed_mean_diff"] >= 0.0
    passed = passed and rank_not_lower
    if all("distractor_rate_at_5" in x[left_key] and "distractor_rate_at_5" in x[right_key] for x in results):
        left_dr = np.asarray([float(x[left_key]["distractor_rate_at_5"]) for x in results], dtype=float)
        right_dr = np.asarray([float(x[right_key]["distractor_rate_at_5"]) for x in results], dtype=float)
        mask = np.isfinite(left_dr) & np.isfinite(right_dr)
        if np.any(mask):
            rank_metrics["distractor_rate_at_5"] = {
                f"{left_key}_mean": float(np.mean(left_dr[mask])),
                f"{right_key}_mean": float(np.mean(right_dr[mask])),
                "mean_diff": float(np.mean(left_dr[mask] - right_dr[mask])),
                "lower_is_better": True,
                "n_pairs": int(np.sum(mask)),
            }
    return {
        "status": f"{status_prefix}_passed" if passed else f"{status_prefix}_not_supported",
        "passed": passed,
        "left_system": left_label,
        "right_system": right_label,
        "thresholds": {
            "rare_recall_at_5_min_diff": 0.05,
            "paired_bootstrap_p_max": 0.05,
            "condition_precision_min_diff": 0.0,
            "mrr_min_diff": 0.0,
            "ndcg_at_5_min_diff": 0.0,
        },
        "rare_recall_at_5": rr_bootstrap,
        "condition_precision": {
            f"{left_key}_mean": float(np.mean(left_cp)),
            f"{right_key}_mean": float(np.mean(right_cp)),
            "mean_diff": cp_diff,
        },
        "rank_metrics": rank_metrics,
    }


def evaluate_edge_pair(
    results: list[dict[str, Any]],
    *,
    left_key: str,
    right_key: str,
    left_label: str,
    right_label: str,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    if not results:
        return {"status": "edge_geometry_claim_no_pairs", "passed": False}
    left_edge = np.asarray([float(x[left_key].get("edge_recall_at_5", float("nan"))) for x in results], dtype=float)
    right_edge = np.asarray([float(x[right_key].get("edge_recall_at_5", float("nan"))) for x in results], dtype=float)
    mask = np.isfinite(left_edge) & np.isfinite(right_edge)
    if not np.any(mask):
        return {
            "status": "edge_geometry_claim_no_edge_gold",
            "passed": False,
            "left_system": left_label,
            "right_system": right_label,
        }
    edge_bootstrap = paired_bootstrap(left_edge[mask], right_edge[mask], n_resamples=n_resamples, seed=seed)
    left_cp = np.asarray([float(x[left_key].get("edge_condition_precision", float("nan"))) for x in results], dtype=float)
    right_cp = np.asarray([float(x[right_key].get("edge_condition_precision", float("nan"))) for x in results], dtype=float)
    cp_mask = np.isfinite(left_cp) & np.isfinite(right_cp)
    cp_diff = float(np.mean(left_cp[cp_mask] - right_cp[cp_mask])) if np.any(cp_mask) else float("nan")
    rank_metrics: dict[str, Any] = {}
    rank_not_lower = True
    for metric in ("edge_mrr", "edge_ndcg_at_5"):
        left_vals = np.asarray([float(x[left_key].get(metric, float("nan"))) for x in results], dtype=float)
        right_vals = np.asarray([float(x[right_key].get(metric, float("nan"))) for x in results], dtype=float)
        metric_mask = np.isfinite(left_vals) & np.isfinite(right_vals)
        if not np.any(metric_mask):
            continue
        boot = paired_bootstrap(left_vals[metric_mask], right_vals[metric_mask], n_resamples=n_resamples, seed=seed)
        rank_metrics[metric] = {
            **boot,
            f"{left_key}_mean": float(np.mean(left_vals[metric_mask])),
            f"{right_key}_mean": float(np.mean(right_vals[metric_mask])),
        }
        rank_not_lower = rank_not_lower and boot["observed_mean_diff"] >= 0.0
    passed = (
        edge_bootstrap["observed_mean_diff"] >= 0.05
        and edge_bootstrap["p_value"] < 0.05
        and (math.isfinite(cp_diff) and cp_diff >= 0.0)
        and rank_not_lower
    )
    return {
        "status": "edge_geometry_claim_passed" if passed else "edge_geometry_claim_not_supported",
        "passed": passed,
        "left_system": left_label,
        "right_system": right_label,
        "thresholds": {
            "edge_recall_at_5_min_diff": 0.05,
            "paired_bootstrap_p_max": 0.05,
            "edge_condition_precision_min_diff": 0.0,
            "edge_mrr_min_diff": 0.0,
            "edge_ndcg_at_5_min_diff": 0.0,
        },
        "edge_recall_at_5": edge_bootstrap,
        "edge_condition_precision": {
            f"{left_key}_mean": float(np.mean(left_cp[cp_mask])) if np.any(cp_mask) else None,
            f"{right_key}_mean": float(np.mean(right_cp[cp_mask])) if np.any(cp_mask) else None,
            "mean_diff": cp_diff if math.isfinite(cp_diff) else None,
        },
        "rank_metrics": rank_metrics,
    }


def evaluate(results: list[dict[str, Any]], *, n_resamples: int = 10_000, seed: int = 42) -> dict[str, Any]:
    """Evaluate already-aggregated paired Poincare/Flat-Twin rows."""
    return evaluate_pair(
        results,
        left_key="poincare",
        right_key="flat_twin",
        left_label="agentic_poincare",
        right_label="agentic_flat_twin",
        status_prefix="hyperbolic_geometry_claim",
        n_resamples=n_resamples,
        seed=seed,
    )


def read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] in "[{":
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def tokenize(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP}


def load_gold(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = read_json_or_jsonl(path)
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    out = {}
    for row in rows:
        out[str(row["query_id"])] = list(row.get("gold_sops", []))
    return out


def node_text(node: dict[str, Any]) -> str:
    parts = [
        str(node.get("title", "")),
        str(node.get("action", "")),
        str(node.get("principle", "")),
        str(node.get("condition", "")),
        " ".join(str(x) for x in node.get("applies_when", []) or []),
    ]
    return " ".join(parts)


def condition_tokens_from_gold(gold_items: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]) -> set[str]:
    toks: set[str] = set()
    for item in gold_items:
        node = nodes.get(str(item.get("sop_id")), {})
        toks |= tokenize(str(node.get("condition", "")))
    return toks


def selected_condition_precision(selected: list[str], gold_items: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], k: int = 5) -> float:
    chosen = selected[:k]
    if not chosen:
        return 0.0
    exact_gold = {
        str(item.get("sop_id"))
        for item in gold_items
        if item.get("condition_match") is True
    }
    q_cond_tokens = condition_tokens_from_gold(gold_items, nodes)
    hits = 0
    for sid in chosen:
        node = nodes.get(sid, {})
        if sid in exact_gold:
            hits += 1
        elif q_cond_tokens and q_cond_tokens & tokenize(node_text(node)):
            hits += 1
    return hits / len(chosen)


def rare_recall_at_k(selected: list[str], gold_items: list[dict[str, Any]], k: int = 5) -> float:
    rare_gold = {
        str(item.get("sop_id"))
        for item in gold_items
        if item.get("is_rare") is True and item.get("relevance") in {"required", "helpful", "risk_warning"}
    }
    if not rare_gold:
        return float("nan")
    return len(rare_gold & set(selected[:k])) / len(rare_gold)


def relevant_gold_gains(gold_items: list[dict[str, Any]]) -> dict[str, float]:
    gains: dict[str, float] = {}
    for item in gold_items:
        relevance = str(item.get("relevance", ""))
        gain = RELEVANCE_GAIN.get(relevance, 0.0)
        if gain <= 0:
            continue
        sid = str(item.get("sop_id"))
        gains[sid] = max(gains.get(sid, 0.0), gain)
    return gains


def exact_recall_at_k(selected: list[str], gold_items: list[dict[str, Any]], k: int) -> float:
    gold = set(relevant_gold_gains(gold_items))
    if not gold:
        return float("nan")
    return len(gold & set(selected[:k])) / len(gold)


def reciprocal_rank(selected: list[str], gold_items: list[dict[str, Any]]) -> float:
    gold = set(relevant_gold_gains(gold_items))
    if not gold:
        return float("nan")
    for rank, sid in enumerate(selected, 1):
        if sid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(selected: list[str], gold_items: list[dict[str, Any]], k: int = 5) -> float:
    gains = relevant_gold_gains(gold_items)
    if not gains:
        return float("nan")
    dcg = 0.0
    for rank, sid in enumerate(selected[:k], 1):
        dcg += gains.get(sid, 0.0) / math.log2(rank + 1)
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    return dcg / idcg if idcg > 0 else float("nan")


def distractor_rate_at_k(row: dict[str, Any], k: int = 5) -> float:
    distractors = {str(x) for x in row.get("distractor_sops", []) or []}
    selected = [str(x) for x in row.get("selected_sops", [])[:k]]
    if not selected:
        return 0.0
    return len(distractors & set(selected)) / len(selected)


def radius_band(node: dict[str, Any]) -> str:
    if node.get("radius_band"):
        return str(node.get("radius_band"))
    try:
        radius = float(node.get("radius"))
        if radius <= 0.35:
            return "core"
        if radius <= 0.60:
            return "middle"
        return "edge"
    except Exception:
        return "unknown"


def edge_band_recall_at_k(selected: list[str], gold_items: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], k: int = 5) -> float:
    edge_gold = {
        str(item.get("sop_id"))
        for item in gold_items
        if item.get("relevance") in {"required", "helpful", "risk_warning"}
        and radius_band(nodes.get(str(item.get("sop_id")), {})) == "edge"
    }
    if not edge_gold:
        return float("nan")
    return len(edge_gold & set(selected[:k])) / len(edge_gold)


def edge_gold_items(gold_items: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in gold_items:
        if item.get("relevance") not in {"required", "helpful", "risk_warning"}:
            continue
        sid = str(item.get("sop_id"))
        if item.get("edge_gold") is True or item.get("gold_radius_band") == "edge" or radius_band(nodes.get(sid, {})) == "edge":
            out.append(item)
    return out


def edge_condition_precision(selected: list[str], gold_items: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], k: int = 5) -> float:
    items = edge_gold_items(gold_items, nodes)
    if not items:
        return float("nan")
    return selected_condition_precision(selected, items, nodes, k=k)


def evidence_coverage(selected: list[str], nodes: dict[str, dict[str, Any]], k: int = 5) -> float:
    chosen = selected[:k]
    if not chosen:
        return 0.0
    hits = 0
    for sid in chosen:
        node = nodes.get(sid, {})
        if (
            node.get("source_branches")
            or node.get("evidence_turns")
            or node.get("reference_ids")
            or node.get("evidence_ids")
            or node.get("metric")
        ):
            hits += 1
    return hits / len(chosen)


def redundancy_rate(selected: list[str], nodes: dict[str, dict[str, Any]], k: int = 5) -> float:
    chosen = selected[:k]
    if len(chosen) < 2:
        return 0.0
    total = 0
    redundant = 0
    for i, left in enumerate(chosen):
        lt = tokenize(node_text(nodes.get(left, {})))
        for right in chosen[i + 1:]:
            rt = tokenize(node_text(nodes.get(right, {})))
            total += 1
            if lt and rt and len(lt & rt) / len(lt | rt) >= 0.85:
                redundant += 1
    return redundant / max(1, total)


def conflict_warning_precision(row: dict[str, Any], gold_items: list[dict[str, Any]]) -> float:
    risk_gold = [item for item in gold_items if item.get("relevance") == "risk_warning"]
    if not risk_gold:
        return 1.0
    selected = set(row.get("selected_sops", [])[:5])
    risk_ids = {str(item.get("sop_id")) for item in risk_gold}
    warnings = row.get("risk_warnings", []) or []
    return 1.0 if warnings and selected & risk_ids else 0.0


def metrics_for_row(row: dict[str, Any], gold_items: list[dict[str, Any]], nodes: dict[str, dict[str, Any]], k: int = 5) -> dict[str, float]:
    selected = [str(x) for x in row.get("selected_sops", [])]
    edge_items = edge_gold_items(gold_items, nodes)
    return {
        "rare_recall_at_5": rare_recall_at_k(selected, gold_items, k=k),
        "exact_recall_at_1": exact_recall_at_k(selected, gold_items, k=1),
        "exact_recall_at_3": exact_recall_at_k(selected, gold_items, k=3),
        "exact_recall_at_5": exact_recall_at_k(selected, gold_items, k=5),
        "mrr": reciprocal_rank(selected, gold_items),
        "ndcg_at_5": ndcg_at_k(selected, gold_items, k=k),
        "condition_precision": selected_condition_precision(selected, gold_items, nodes, k=k),
        "evidence_coverage": evidence_coverage(selected, nodes, k=k),
        "redundancy_rate": redundancy_rate(selected, nodes, k=k),
        "distractor_rate_at_5": distractor_rate_at_k(row, k=k),
        "edge_band_recall_at_5": edge_band_recall_at_k(selected, gold_items, nodes, k=k),
        "edge_recall_at_5": exact_recall_at_k(selected, edge_items, k=5) if edge_items else float("nan"),
        "edge_mrr": reciprocal_rank(selected, edge_items) if edge_items else float("nan"),
        "edge_ndcg_at_5": ndcg_at_k(selected, edge_items, k=k) if edge_items else float("nan"),
        "edge_condition_precision": edge_condition_precision(selected, gold_items, nodes, k=k),
        "conflict_warning_precision": conflict_warning_precision(row, gold_items),
        "navigation_tool_calls": float(len(row.get("navigation_trace", []) or [])),
        "injected_tokens": float(row.get("injected_tokens", 0.0) or 0.0),
        "latency_sec": float(row.get("latency_sec", 0.0) or 0.0),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row if isinstance(row.get(k), (int, float))})
    out = {}
    for key in keys:
        vals = np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size:
            out[key] = float(np.mean(finite))
    return out


def json_metric_value(value: float) -> float | None:
    return None if isinstance(value, float) and not math.isfinite(value) else value


def claim_grade_status(graph: dict[str, Any], graph_report: dict[str, Any] | None, quality_report: dict[str, Any] | None) -> dict[str, Any]:
    graph_meta = graph.get("meta", {}) or {}
    provenance_ok = bool(graph_meta.get("paper_grade"))
    if graph_report:
        provenance_ok = provenance_ok and bool(graph_report.get("provenance", {}).get("paper_grade"))
    quality_ok = True if quality_report is None else bool(quality_report.get("passed"))
    missing = []
    if not provenance_ok:
        missing.append("paper_grade_provenance")
    if not quality_ok:
        missing.append("coordinate_quality")
    return {
        "claim_grade": not missing,
        "provenance_ok": provenance_ok,
        "coordinate_quality_ok": quality_ok,
        "missing": missing,
    }


def pair_rows_for_systems(
    *,
    rows_by_pair: dict[tuple[str, str], dict[str, Any]],
    gold_by_query: dict[str, list[dict[str, Any]]],
    nodes: dict[str, dict[str, Any]],
    left_system: str,
    right_system: str,
    left_key: str,
    right_key: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    pair_rows = []
    overlaps = []
    query_ids = sorted({qid for qid, system in rows_by_pair if system == left_system})
    for qid in query_ids:
        left = rows_by_pair.get((qid, left_system))
        right = rows_by_pair.get((qid, right_system))
        if not left or not right:
            continue
        left_metrics = metrics_for_row(left, gold_by_query.get(qid, []), nodes)
        right_metrics = metrics_for_row(right, gold_by_query.get(qid, []), nodes)
        pair_rows.append({"query_id": qid, left_key: left_metrics, right_key: right_metrics})
        left_top = set(left.get("selected_sops", [])[:5])
        right_top = set(right.get("selected_sops", [])[:5])
        overlaps.append(len(left_top & right_top) / max(1, len(left_top | right_top)))
    return pair_rows, overlaps


def evaluate_runner_results(
    *,
    result_rows: list[dict[str, Any]],
    gold_by_query: dict[str, list[dict[str, Any]]],
    graph: dict[str, Any],
    graph_report: dict[str, Any] | None = None,
    quality_report: dict[str, Any] | None = None,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    per_query = []
    by_system: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    by_system_kind: dict[str, dict[str, list[dict[str, float]]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    rows_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in result_rows:
        qid = str(row["query_id"])
        system = str(row["system"])
        query_kind = str(row.get("query_kind", "unknown") or "unknown")
        metrics = metrics_for_row(row, gold_by_query.get(qid, []), nodes)
        per_query.append({
            "query_id": qid,
            "query_kind": query_kind,
            "query_style": row.get("query_style", ""),
            "query_specificity": row.get("query_specificity", ""),
            "system": system,
            **{key: json_metric_value(value) for key, value in metrics.items()},
        })
        by_system[system].append(metrics)
        by_system_kind[system][query_kind].append(metrics)
        rows_by_pair[(qid, system)] = row

    system_summary = {system: aggregate(rows) for system, rows in sorted(by_system.items())}
    system_kind_summary = {
        system: {kind: aggregate(rows) for kind, rows in sorted(kind_rows.items())}
        for system, kind_rows in sorted(by_system_kind.items())
    }
    pair_rows, overlaps = pair_rows_for_systems(
        rows_by_pair=rows_by_pair,
        gold_by_query=gold_by_query,
        nodes=nodes,
        left_system="agentic_poincare",
        right_system="agentic_flat_twin",
        left_key="poincare",
        right_key="flat_twin",
    )

    if not pair_rows:
        raise ValueError("runner results do not contain paired agentic_poincare and agentic_flat_twin rows")
    ablation = evaluate(pair_rows, n_resamples=n_resamples, seed=seed)
    comparisons = {
        "poincare_vs_flat_twin_same_coordinate": {
            **ablation,
            "ranking_diagnostics": {
                "top5_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
                "paired_queries": len(pair_rows),
            },
        }
    }
    comparisons["poincare_vs_flat_twin_edge_claim"] = evaluate_edge_pair(
        pair_rows,
        left_key="poincare",
        right_key="flat_twin",
        left_label="agentic_poincare",
        right_label="agentic_flat_twin",
        n_resamples=n_resamples,
        seed=seed,
    )
    euclidean_rows, euclidean_overlaps = pair_rows_for_systems(
        rows_by_pair=rows_by_pair,
        gold_by_query=gold_by_query,
        nodes=nodes,
        left_system="agentic_poincare",
        right_system="agentic_euclidean",
        left_key="poincare",
        right_key="euclidean",
    )
    if euclidean_rows:
        comparisons["poincare_vs_euclidean_independent_coordinates"] = {
            **evaluate_pair(
                euclidean_rows,
                left_key="poincare",
                right_key="euclidean",
                left_label="agentic_poincare",
                right_label="agentic_euclidean",
                status_prefix="hyperbolic_vs_euclidean_memory_claim",
                n_resamples=n_resamples,
                seed=seed,
            ),
            "ranking_diagnostics": {
                "top5_overlap_mean": float(np.mean(euclidean_overlaps)) if euclidean_overlaps else 0.0,
                "paired_queries": len(euclidean_rows),
            },
        }
    readiness = claim_grade_status(graph, graph_report, quality_report)
    edge_system_metrics: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
    for row in result_rows:
        qid = str(row["query_id"])
        if edge_gold_items(gold_by_query.get(qid, []), nodes):
            edge_system_metrics[str(row["system"])].append(metrics_for_row(row, gold_by_query.get(qid, []), nodes))
    edge_system_summary = {system: aggregate(rows) for system, rows in sorted(edge_system_metrics.items())}

    selected_band_counts: dict[str, dict[str, int]] = {}
    selected_edge_rates: dict[str, float] = {}
    for system, rows in sorted(by_system.items()):
        counter: collections.Counter[str] = collections.Counter()
        edge_total = 0
        edge_selected = 0
        for row in result_rows:
            if str(row.get("system")) != system:
                continue
            for sid in row.get("selected_sops", [])[:5]:
                band = radius_band(nodes.get(str(sid), {}))
                counter[band] += 1
                qid = str(row.get("query_id", ""))
                if edge_gold_items(gold_by_query.get(qid, []), nodes):
                    edge_total += 1
                    edge_selected += int(band == "edge")
        selected_band_counts[system] = {k: int(v) for k, v in sorted(counter.items())}
        selected_edge_rates[system] = edge_selected / edge_total if edge_total else float("nan")
    gold_band_counter: collections.Counter[str] = collections.Counter()
    edge_pressure_values = []
    for items in gold_by_query.values():
        relevant_count = 0
        edge_count = 0
        for item in items:
            if item.get("relevance") in {"required", "helpful", "risk_warning"}:
                relevant_count += 1
                band = radius_band(nodes.get(str(item.get("sop_id")), {}))
                gold_band_counter[band] += 1
                edge_count += int(band == "edge" or item.get("edge_gold") is True or item.get("gold_radius_band") == "edge")
        if relevant_count:
            edge_pressure_values.append(edge_count / relevant_count)
    edge_query_ids = [
        row["query_id"]
        for row in pair_rows
        if edge_gold_items(gold_by_query.get(str(row["query_id"]), []), nodes)
    ]
    edge_overlaps = []
    for qid in edge_query_ids:
        left = rows_by_pair.get((str(qid), "agentic_poincare"))
        right = rows_by_pair.get((str(qid), "agentic_flat_twin"))
        if not left or not right:
            continue
        left_top = set(left.get("selected_sops", [])[:5])
        right_top = set(right.get("selected_sops", [])[:5])
        edge_overlaps.append(len(left_top & right_top) / max(1, len(left_top | right_top)))
    gold_edge_pressure = float(np.mean(edge_pressure_values)) if edge_pressure_values else 0.0
    edge_overlap_mean = float(np.mean(edge_overlaps)) if edge_overlaps else float("nan")
    poincare_edge_rate = selected_edge_rates.get("agentic_poincare", float("nan"))
    available_gate_checks = {
        "edge_top5_overlap_le_0_90": bool(math.isfinite(edge_overlap_mean) and edge_overlap_mean <= 0.90),
        "poincare_selected_edge_rate_within_20pp_of_gold_pressure": bool(
            math.isfinite(poincare_edge_rate) and poincare_edge_rate >= max(0.0, gold_edge_pressure - 0.20)
        ),
    }
    missing_static_gates = [
        "dev_query_gold_angular_rank_at_10_vs_lexical",
        "edge_gold_median_angular_percentile",
    ]
    if edge_query_ids and not all(available_gate_checks.values()):
        query_quality_status = "coordinate_quality_null"
    elif edge_query_ids:
        query_quality_status = "partial_passed_missing_static_angular_gates"
    else:
        query_quality_status = "diagnostic_only_no_edge_gold"
    query_aware_quality = {
        "status": query_quality_status,
        "passed": query_quality_status == "partial_passed_missing_static_angular_gates",
        "gold_radius_band_distribution": {k: int(v) for k, v in sorted(gold_band_counter.items())},
        "selected_radius_band_distribution_by_system": selected_band_counts,
        "selected_edge_rate_by_system": {
            system: (None if not math.isfinite(rate) else rate)
            for system, rate in sorted(selected_edge_rates.items())
        },
        "gold_edge_pressure_mean": gold_edge_pressure,
        "poincare_flat_twin_top5_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "poincare_flat_twin_edge_top5_overlap_mean": None if not math.isfinite(edge_overlap_mean) else edge_overlap_mean,
        "available_gate_checks": available_gate_checks,
        "missing_static_gates": missing_static_gates,
        "edge_paired_queries": len(edge_query_ids),
        "note": "If this is coordinate_quality_null, interpret failures as map/query-quality issues before claiming hyperbolic geometry failed.",
    }
    if query_quality_status == "coordinate_quality_null":
        comparisons["poincare_vs_flat_twin_edge_claim"]["status"] = "coordinate_quality_null"
        comparisons["poincare_vs_flat_twin_edge_claim"]["passed"] = False
    if not readiness["claim_grade"]:
        ablation["status"] = "not_claim_grade"
        ablation["passed"] = False
        ablation["claim_blockers"] = readiness["missing"]
        for comparison in comparisons.values():
            comparison["status"] = "not_claim_grade"
            comparison["passed"] = False
            comparison["claim_blockers"] = readiness["missing"]

    return {
        **ablation,
        "run_metadata": {
            "radius_hint_modes": sorted({str(row.get("radius_hint_mode", "")) for row in result_rows if row.get("radius_hint_mode")}),
            "navigator_modes": sorted({str(row.get("navigator_mode", "")) for row in result_rows if row.get("navigator_mode")}),
            "query_kinds": sorted({str(row.get("query_kind", "")) for row in result_rows if row.get("query_kind")}),
            "splits": sorted({str(row.get("split", "")) for row in result_rows if row.get("split")}),
        },
        "claim_grade": readiness,
        "comparisons": comparisons,
        "systems": system_summary,
        "systems_edge_gold_only": edge_system_summary,
        "systems_by_query_kind": system_kind_summary,
        "navigation_efficiency": {
            system: {
                "avg_tool_calls": metrics.get("navigation_tool_calls", 0.0),
                "avg_injected_tokens": metrics.get("injected_tokens", 0.0),
                "avg_latency_sec": metrics.get("latency_sec", 0.0),
            }
            for system, metrics in system_summary.items()
        },
        "ranking_diagnostics": {
            "poincare_flat_twin_top5_overlap_mean": float(np.mean(overlaps)) if overlaps else 0.0,
            "paired_queries": len(pair_rows),
        },
        "query_aware_coordinate_quality": query_aware_quality,
        "per_query": per_query,
    }


def maybe_load(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Poincare vs same-coordinate Flat-Twin ablation.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--graph", type=Path, default=None)
    parser.add_argument("--graph-builder-report", type=Path, default=None)
    parser.add_argument("--quality-report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loaded = read_json_or_jsonl(args.results)
    if args.gold:
        if not args.graph:
            raise SystemExit("--graph is required when evaluating runner JSONL")
        report = evaluate_runner_results(
            result_rows=list(loaded),
            gold_by_query=load_gold(args.gold),
            graph=json.loads(args.graph.read_text(encoding="utf-8")),
            graph_report=maybe_load(args.graph_builder_report),
            quality_report=maybe_load(args.quality_report),
            n_resamples=args.n_resamples,
            seed=args.seed,
        )
    else:
        report = evaluate(list(loaded), n_resamples=args.n_resamples, seed=args.seed)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
