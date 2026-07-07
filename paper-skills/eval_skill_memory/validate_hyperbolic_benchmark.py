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

STOP = {"when", "with", "that", "this", "into", "from", "using", "and", "the", "data", "model", "training"}


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


def validate(
    *,
    graph_path: Path,
    benchmark_path: Path,
    gold_path: Path,
    allowlist_path: Path,
    require_certified_graph: bool = False,
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
    for query in benchmark:
        qid = query.get("query_id")
        row = gold_by_query.get(qid, {})
        gold_sops = row.get("gold_sops", [])
        if not any(g.get("relevance") in {"required", "risk_warning"} for g in gold_sops):
            errors.append(f"{qid}: needs at least one required or risk_warning gold SOP")
        for item in gold_sops:
            sid = item.get("sop_id")
            if sid not in sops:
                errors.append(f"{qid}: gold sop_id does not exist or is not SOP: {sid}")
                continue
            node = sops[sid]
            node_runs = source_runs_for_node(node)
            if not node_runs:
                errors.append(f"{qid}: gold SOP lacks source_branches: {sid}")
            bad = sorted(node_runs - allowed)
            if bad:
                errors.append(f"{qid}: gold SOP {sid} uses disallowed runs: {bad}")
            if item.get("is_rare"):
                sig = condition_failure_signature(query)
                if not sig:
                    errors.append(f"{qid}: rare gold has empty condition/failure signature")
                if signature_counts[sig] > 2 and int(item.get("rarity_count", 0) or 0) > 2:
                    errors.append(f"{qid}: rare gold is not condition/failure-specific rare")

    if len(benchmark) != 40:
        warnings.append(f"expected 40 benchmark queries, found {len(benchmark)}")
    by_kind = collections.Counter(row.get("query_kind", "unknown") for row in benchmark)
    return {
        "status": "passed" if not errors else "failed",
        "passed": not errors,
        "queries": len(benchmark),
        "gold_rows": len(gold),
        "by_kind": dict(by_kind),
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
    parser.add_argument("--require-certified-graph", action="store_true")
    args = parser.parse_args()
    report = validate(
        graph_path=args.graph,
        benchmark_path=args.benchmark,
        gold_path=args.gold,
        allowlist_path=args.allowlist,
        require_certified_graph=args.require_certified_graph,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
