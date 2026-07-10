"""Validate the hyperbolic SOP benchmark and gold labels."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

from certify_skillgraph_provenance import allowed_run_ids


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_BENCH = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "hyperbolic_sop_benchmark.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "hyperbolic_sop_gold.jsonl"
DEFAULT_ALLOWLIST = REPO / "paper-skills" / "eval_skill_memory" / "clean_run_allowlist.json"
DEFAULT_BASELINE_VALIDATION = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "benchmark_validation_report.json"

STOP = {
    "when", "with", "that", "this", "into", "from", "using", "use", "uses", "for", "and",
    "the", "are", "was", "were", "has", "have", "data", "model", "training", "task",
    "avoid", "ensure", "correct", "prevent", "prevents", "check", "set",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSONL: {exc}") from exc
    return rows


def tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", (text or "").lower())
        if t not in STOP
    }


def source_runs_for_node(node: dict[str, Any]) -> set[str]:
    runs = set()
    for item in node.get("source_branches", []) or []:
        if isinstance(item, (list, tuple)) and item:
            runs.add(str(item[0]))
        elif isinstance(item, str) and item:
            runs.add(item.split(":", 1)[0])
    return runs


def condition_failure_signature(query: dict[str, Any]) -> tuple[str, ...]:
    text = " ".join([*(query.get("condition") or []), *(query.get("failure_mode") or [])])
    return tuple(sorted(tokens(text))[:8])


def title_overlap(query: dict[str, Any], gold_items: list[dict[str, Any]], nodes: dict[str, dict[str, Any]]) -> float:
    if not gold_items:
        return 0.0
    title_toks: set[str] = set()
    for item in gold_items:
        title_toks |= tokens(str(nodes.get(str(item.get("sop_id")), {}).get("title", "")))
    if not title_toks:
        return 0.0
    query_text = " ".join(
        [
            str(query.get("context", "")),
            " ".join(query.get("condition") or []),
            " ".join(query.get("failure_mode") or []),
        ]
    )
    return len(tokens(query_text) & title_toks) / len(title_toks)


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


def maybe_baseline_overlap(path: Path | None) -> float | None:
    if not path or not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        value = report.get("title_token_overlap_mean")
        return float(value) if value is not None else None
    except Exception:
        return None


def validate(
    *,
    graph_path: Path,
    benchmark_path: Path,
    gold_path: Path,
    allowlist_path: Path,
    require_certified_graph: bool = False,
    edge_profile: bool | None = None,
    baseline_validation_path: Path | None = DEFAULT_BASELINE_VALIDATION,
) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    sops = {nid: n for nid, n in nodes.items() if n.get("type") == "SOP"}
    benchmark = read_jsonl(benchmark_path)
    gold = read_jsonl(gold_path)
    gold_by_query = {row.get("query_id"): row for row in gold}
    allowlist = json.loads(allowlist_path.read_text(encoding="utf-8"))
    allowed = allowed_run_ids(allowlist)

    errors: list[str] = []
    warnings: list[str] = []
    if require_certified_graph and not graph.get("meta", {}).get("paper_grade"):
        errors.append("graph meta paper_grade is false")

    query_ids = [row.get("query_id") for row in benchmark]
    if len(query_ids) != len(set(query_ids)):
        errors.append("benchmark query_id values are not unique")
    missing_gold = sorted(set(query_ids) - set(gold_by_query))
    if missing_gold:
        errors.append(f"queries missing gold labels: {missing_gold[:10]}")

    signature_counts = collections.Counter(condition_failure_signature(q) for q in benchmark)
    overlap_values = []
    distractor_counts = []
    sop_splits: dict[str, set[str]] = collections.defaultdict(set)
    if edge_profile is None:
        edge_profile = bool(benchmark) and all(str(row.get("query_kind", "")).startswith("edge_") for row in benchmark)
    for query in benchmark:
        qid = query.get("query_id")
        row = gold_by_query.get(qid, {})
        gold_sops = row.get("gold_sops", [])
        overlap_values.append(title_overlap(query, gold_sops, nodes))
        distractor_counts.append(int(query.get("distractor_count", 0) or 0))
        if query.get("split") not in {"dev", "test"}:
            errors.append(f"{qid}: split must be dev or test")
        if not query.get("gold_title_hidden", False):
            warnings.append(f"{qid}: gold_title_hidden is false")
        if "radius_band_hint" not in query:
            warnings.append(f"{qid}: missing radius_band_hint")
        if "query_style" not in query:
            warnings.append(f"{qid}: missing query_style")
        if "query_specificity" not in query:
            warnings.append(f"{qid}: missing query_specificity")
        for did in query.get("distractor_sops", []) or []:
            if did not in sops:
                errors.append(f"{qid}: distractor_sop does not exist or is not SOP: {did}")
        if not any(g.get("relevance") in {"required", "risk_warning"} for g in gold_sops):
            errors.append(f"{qid}: needs at least one required or risk_warning gold SOP")
        for item in gold_sops:
            sid = item.get("sop_id")
            if sid not in sops:
                errors.append(f"{qid}: gold sop_id does not exist or is not SOP: {sid}")
                continue
            node = sops[sid]
            sop_splits[str(sid)].add(str(query.get("split", "")))
            node_runs = source_runs_for_node(node)
            if not node_runs:
                errors.append(f"{qid}: gold SOP lacks source_branches: {sid}")
            bad = sorted(node_runs - allowed)
            if bad:
                errors.append(f"{qid}: gold SOP {sid} uses disallowed runs: {bad}")
            if item.get("is_rare") and not (edge_profile and item.get("edge_gold") is True):
                sig = condition_failure_signature(query)
                if not sig:
                    errors.append(f"{qid}: rare gold has empty condition/failure signature")
                if signature_counts[sig] > 2 and int(item.get("rarity_count", 0) or 0) > 2:
                    errors.append(f"{qid}: rare gold is not condition/failure-specific rare")
            if edge_profile:
                if radius_band(node) != "edge":
                    errors.append(f"{qid}: edge benchmark gold SOP is not edge-band: {sid}")
                if item.get("gold_radius_band") != "edge" or item.get("edge_gold") is not True:
                    errors.append(f"{qid}: edge gold lacks edge diagnostic fields: {sid}")
                if not item.get("edge_reason"):
                    errors.append(f"{qid}: edge gold lacks edge_reason: {sid}")
        if edge_profile and int(query.get("distractor_count", 0) or 0) < 20:
            errors.append(f"{qid}: edge benchmark requires at least 20 same-task distractors")

    if len(benchmark) < 40:
        warnings.append(f"expected at least 40 benchmark queries, found {len(benchmark)}")
    by_kind = collections.Counter(row.get("query_kind", "unknown") for row in benchmark)
    by_split = collections.Counter(row.get("split", "unspecified") for row in benchmark)
    mean_overlap = sum(overlap_values) / max(1, len(overlap_values))
    max_overlap = max(overlap_values or [0.0])
    mean_distractors = sum(distractor_counts) / max(1, len(distractor_counts))
    if mean_overlap > 0.55:
        warnings.append(f"high mean title-token overlap: {mean_overlap:.3f}")
    baseline_overlap = maybe_baseline_overlap(baseline_validation_path)
    if edge_profile:
        for sid, splits in sop_splits.items():
            if len(splits) > 1:
                errors.append(f"edge SOP variants cross dev/test splits: {sid} -> {sorted(splits)}")
        if baseline_overlap is not None and mean_overlap >= baseline_overlap:
            errors.append(f"edge mean title-token overlap {mean_overlap:.3f} is not lower than baseline hard benchmark {baseline_overlap:.3f}")
        elif baseline_overlap is None and mean_overlap > 0.05:
            warnings.append(f"edge mean title-token overlap has no baseline comparison and is {mean_overlap:.3f}")
    if max_overlap > 0.80:
        warnings.append(f"very high max title-token overlap: {max_overlap:.3f}")
    if mean_distractors < 3.0:
        warnings.append(f"low mean distractor count: {mean_distractors:.3f}")
    if "dev" in by_split or "test" in by_split:
        for kind in by_kind:
            dev = sum(1 for row in benchmark if row.get("query_kind") == kind and row.get("split") == "dev")
            test = sum(1 for row in benchmark if row.get("query_kind") == kind and row.get("split") == "test")
            if dev != test:
                warnings.append(f"uneven dev/test split for {kind}: dev={dev} test={test}")
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "queries": len(benchmark),
        "gold_rows": len(gold),
        "by_kind": dict(by_kind),
        "by_split": dict(by_split),
        "by_query_style": dict(collections.Counter(row.get("query_style", "unknown") for row in benchmark)),
        "by_query_specificity": dict(collections.Counter(row.get("query_specificity", "unknown") for row in benchmark)),
        "title_token_overlap_mean": mean_overlap,
        "title_token_overlap_max": max_overlap,
        "title_leakage_levels": dict(collections.Counter(row.get("title_leakage_level", "unknown") for row in benchmark)),
        "distractor_count_mean": mean_distractors,
        "distractor_count_min": min(distractor_counts or [0]),
        "edge_profile": bool(edge_profile),
        "baseline_title_token_overlap_mean": baseline_overlap,
        "require_certified_graph": require_certified_graph,
        "graph_paper_grade": bool(graph.get("meta", {}).get("paper_grade")),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hyperbolic SOP benchmark and gold labels.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--edge-profile", action="store_true")
    parser.add_argument("--baseline-validation", type=Path, default=DEFAULT_BASELINE_VALIDATION)
    parser.add_argument("--require-certified-graph", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = validate(
        graph_path=args.graph,
        benchmark_path=args.benchmark,
        gold_path=args.gold,
        allowlist_path=args.allowlist,
        require_certified_graph=args.require_certified_graph,
        edge_profile=True if args.edge_profile else None,
        baseline_validation_path=args.baseline_validation,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
