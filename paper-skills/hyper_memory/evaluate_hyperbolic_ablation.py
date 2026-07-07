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


def evaluate(results: list[dict[str, Any]], *, n_resamples: int = 10_000, seed: int = 42) -> dict[str, Any]:
    """Evaluate already-aggregated paired Poincare/Flat-Twin rows."""
    if not results:
        raise ValueError("empty results")
    poincare_rr = np.asarray([float(x["poincare"]["rare_recall_at_5"]) for x in results], dtype=float)
    flat_rr = np.asarray([float(x["flat_twin"]["rare_recall_at_5"]) for x in results], dtype=float)
    poincare_cp = np.asarray([float(x["poincare"]["condition_precision"]) for x in results], dtype=float)
    flat_cp = np.asarray([float(x["flat_twin"]["condition_precision"]) for x in results], dtype=float)

    rr_bootstrap = paired_bootstrap(poincare_rr, flat_rr, n_resamples=n_resamples, seed=seed)
    rr_diff = rr_bootstrap["observed_mean_diff"]
    cp_diff = float(np.mean(poincare_cp - flat_cp))
    passed = rr_diff >= 0.05 and rr_bootstrap["p_value"] < 0.05 and cp_diff >= 0.0
    return {
        "status": "hyperbolic_geometry_claim_passed" if passed else "hyperbolic_geometry_claim_not_supported",
        "passed": passed,
        "thresholds": {
            "rare_recall_at_5_min_diff": 0.05,
            "paired_bootstrap_p_max": 0.05,
            "condition_precision_min_diff": 0.0,
        },
        "rare_recall_at_5": rr_bootstrap,
        "condition_precision": {
            "poincare_mean": float(np.mean(poincare_cp)),
            "flat_twin_mean": float(np.mean(flat_cp)),
            "mean_diff": cp_diff,
        },
    }


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
        return 0.0
    return len(rare_gold & set(selected[:k])) / len(rare_gold)


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
    return {
        "rare_recall_at_5": rare_recall_at_k(selected, gold_items, k=k),
        "condition_precision": selected_condition_precision(selected, gold_items, nodes, k=k),
        "evidence_coverage": evidence_coverage(selected, nodes, k=k),
        "redundancy_rate": redundancy_rate(selected, nodes, k=k),
        "conflict_warning_precision": conflict_warning_precision(row, gold_items),
        "navigation_tool_calls": float(len(row.get("navigation_trace", []) or [])),
        "injected_tokens": float(row.get("injected_tokens", 0.0) or 0.0),
        "latency_sec": float(row.get("latency_sec", 0.0) or 0.0),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for row in rows for k in row if isinstance(row.get(k), (int, float))})
    return {key: float(np.mean([float(row.get(key, 0.0)) for row in rows])) for key in keys}


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
    rows_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in result_rows:
        qid = str(row["query_id"])
        system = str(row["system"])
        metrics = metrics_for_row(row, gold_by_query.get(qid, []), nodes)
        per_query.append({"query_id": qid, "system": system, **metrics})
        by_system[system].append(metrics)
        rows_by_pair[(qid, system)] = row

    system_summary = {system: aggregate(rows) for system, rows in sorted(by_system.items())}
    pair_rows = []
    overlaps = []
    query_ids = sorted({qid for qid, system in rows_by_pair if system == "agentic_poincare"})
    for qid in query_ids:
        p = rows_by_pair.get((qid, "agentic_poincare"))
        f = rows_by_pair.get((qid, "agentic_flat_twin"))
        if not p or not f:
            continue
        pg = metrics_for_row(p, gold_by_query.get(qid, []), nodes)
        fg = metrics_for_row(f, gold_by_query.get(qid, []), nodes)
        pair_rows.append({"query_id": qid, "poincare": pg, "flat_twin": fg})
        p_top = set(p.get("selected_sops", [])[:5])
        f_top = set(f.get("selected_sops", [])[:5])
        overlaps.append(len(p_top & f_top) / max(1, len(p_top | f_top)))

    if not pair_rows:
        raise ValueError("runner results do not contain paired agentic_poincare and agentic_flat_twin rows")
    ablation = evaluate(pair_rows, n_resamples=n_resamples, seed=seed)
    readiness = claim_grade_status(graph, graph_report, quality_report)
    if not readiness["claim_grade"]:
        ablation["status"] = "not_claim_grade"
        ablation["passed"] = False
        ablation["claim_blockers"] = readiness["missing"]

    return {
        **ablation,
        "claim_grade": readiness,
        "systems": system_summary,
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
