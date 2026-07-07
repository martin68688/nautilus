"""Build an initial offline benchmark for Agentic Hyperbolic SOP Memory.

This creates a deterministic 40-query seed benchmark from the certified
hyper_graph. It is meant to be a machine-checkable starting point, not a final
human-audited gold set. The validator and readiness report keep that boundary
visible.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_BENCH = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "hyperbolic_sop_benchmark.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "hyperbolic_sop_gold.jsonl"

TASKS = [
    "spooky-author-identification",
    "leaf-classification",
    "aerial-cactus-identification",
    "denoising-dirty-documents",
    "new-york-city-taxi-fare-prediction",
]

STOP = {
    "when", "with", "that", "this", "into", "from", "using", "use", "uses", "for", "and",
    "the", "are", "was", "were", "has", "have", "data", "model", "training", "task",
}


def load_graph(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    sops = [n for n in graph.get("nodes", []) if n.get("type") == "SOP"]
    failures: dict[str, list[str]] = collections.defaultdict(list)
    for edge in graph.get("edges", []):
        if edge.get("kind") != "prevents":
            continue
        src, dst = edge.get("src"), edge.get("dst")
        if src in nodes and dst in nodes and nodes[dst].get("type") == "FailureMode":
            failures[src].append(str(nodes[dst].get("title", "")))
    return nodes, sops, failures


def tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", (text or "").lower())
        if t not in STOP
    }


def condition_signature(node: dict[str, Any], failure_labels: list[str]) -> tuple[str, ...]:
    text = " ".join([str(node.get("condition", "")), " ".join(failure_labels)])
    return tuple(sorted(tokens(text))[:8])


def round_robin(nodes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for node in nodes:
        by_task[str(node.get("category"))].append(node)
    for group in by_task.values():
        group.sort(key=lambda n: (n.get("radius_band") != "edge", n.get("title", ""), n.get("id", "")))
    picked: list[dict[str, Any]] = []
    while len(picked) < limit:
        changed = False
        for task in TASKS:
            if by_task.get(task):
                picked.append(by_task[task].pop(0))
                changed = True
                if len(picked) >= limit:
                    break
        if not changed:
            break
    return picked


def query_row(query_id: str, node: dict[str, Any], kind: str, failures: list[str]) -> dict[str, Any]:
    condition = str(node.get("condition", "")).strip()
    failure = failures[:2] or ["method failure"]
    task = str(node.get("category", ""))
    stage = {
        "rare_condition": "improve",
        "debug_failure": "debug",
        "conflict_risk": "debug",
        "method_set": "draft",
    }[kind]
    if kind == "rare_condition":
        context = f"{task}: under a narrow condition, {condition}. Which low-frequency SOP should guide the next change?"
    elif kind == "debug_failure":
        context = f"{task}: the current node shows {', '.join(failure)}. What SOP prevents this failure?"
    elif kind == "conflict_risk":
        context = f"{task}: I am considering {node.get('title')}. Check conflicts, risks, and condition branches before adopting it."
    else:
        context = f"{task}: build a coherent method set around {node.get('title')} while respecting the condition: {condition}."
    return {
        "query_id": query_id,
        "task_type": task,
        "stage": stage,
        "context": context,
        "condition": [condition] if condition else [],
        "failure_mode": failure,
        "source_trace": "",
        "query_kind": kind,
    }


def gold_row(query_id: str, node: dict[str, Any], relevance: str, is_rare: bool, rarity_count: int) -> dict[str, Any]:
    return {
        "query_id": query_id,
        "gold_sops": [
            {
                "sop_id": node["id"],
                "relevance": relevance,
                "condition_match": True,
                "is_rare": bool(is_rare),
                "rarity_basis": "condition_failure_signature",
                "rarity_count": int(rarity_count),
                "rationale": f"Auto-seeded from certified SOP '{node.get('title', node['id'])}'. Requires human audit before final paper tables.",
            }
        ],
    }


def build(graph_path: Path, benchmark_path: Path, gold_path: Path) -> dict[str, Any]:
    _nodes, sops, failure_by_sop = load_graph(graph_path)
    signatures = {n["id"]: condition_signature(n, failure_by_sop.get(n["id"], [])) for n in sops}
    counts = collections.Counter(sig for sig in signatures.values() if sig)

    task_sops = [n for n in sops if n.get("category") in TASKS]
    rare_candidates = [
        n for n in task_sops
        if n.get("radius_band") == "edge" and counts.get(signatures.get(n["id"], ()), 999) <= 2
    ]
    debug_candidates = [
        n for n in task_sops
        if any(f for f in failure_by_sop.get(n["id"], []) if f not in {"general execution failure"})
    ]
    conflict_candidates = [
        n for n in task_sops
        if any(word in (n.get("title", "") + " " + n.get("principle", "")).lower() for word in ("avoid", "disable", "prevent", "check", "correct"))
    ]
    method_candidates = [
        n for n in task_sops
        if n.get("radius_band") in {"core", "middle"} and n not in conflict_candidates
    ]

    groups = [
        ("rare_condition", round_robin(rare_candidates, 10), "required"),
        ("debug_failure", round_robin(debug_candidates, 10), "required"),
        ("conflict_risk", round_robin(conflict_candidates, 10), "risk_warning"),
        ("method_set", round_robin(method_candidates, 10), "required"),
    ]
    rows = []
    gold = []
    for kind, selected, relevance in groups:
        if len(selected) < 10:
            raise ValueError(f"Not enough {kind} candidates: {len(selected)}")
        for idx, node in enumerate(selected[:10], 1):
            query_id = f"{kind}_{idx:02d}_{node['id']}"
            failures = failure_by_sop.get(node["id"], [])
            sig = signatures.get(node["id"], ())
            rarity_count = counts.get(sig, 0)
            is_rare = kind == "rare_condition" or rarity_count <= 2
            rows.append(query_row(query_id, node, kind, failures))
            gold.append(gold_row(query_id, node, relevance, is_rare, rarity_count))

    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    gold_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gold) + "\n", encoding="utf-8")
    return {
        "status": "auto_seeded_needs_human_audit",
        "benchmark": str(benchmark_path),
        "gold": str(gold_path),
        "queries": len(rows),
        "by_kind": dict(collections.Counter(r["query_kind"] for r in rows)),
        "note": "Use for deterministic harness development. Human audit is required before final paper claim tables.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 40-query hyperbolic SOP benchmark seed set.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    args = parser.parse_args()
    print(json.dumps(build(args.graph, args.benchmark, args.gold), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
