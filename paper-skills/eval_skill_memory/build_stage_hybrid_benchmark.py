#!/usr/bin/env python3
"""Build a run-grouped natural-language benchmark for stage-hybrid retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
DEFAULT_BENCHMARK = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "stage_hybrid_runforest_benchmark.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "stage_hybrid_runforest_gold.jsonl"
DEFAULT_REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "stage_hybrid_benchmark_validation.json"


def split_for_run(run_id: str) -> str:
    bucket = int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else "dev" if bucket < 8 else "test"


def _clean(node: dict[str, Any]) -> bool:
    audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
    return (
        audit.get("status") == "clean"
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
    )


def _stage(value: Any) -> str:
    text = str(value or "draft")
    if text in {"draft", "improve", "debug", "evolution", "fusion"}:
        return text
    if "debug" in text:
        return "debug"
    if "evolution" in text:
        return "evolution"
    if "fusion" in text:
        return "fusion"
    return "improve"


def build_records(graph: dict[str, Any], *, max_queries: int = 240) -> tuple[list[dict], list[dict]]:
    nodes = {str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")}
    evidence_by_transition: dict[str, list[str]] = {}
    sops_by_transition: dict[str, list[str]] = {}
    for edge in graph.get("edges", []):
        src, dst = str(edge.get("src", "")), str(edge.get("dst", ""))
        kind = str(edge.get("kind") or edge.get("type") or "")
        if kind == "supported_by" and nodes.get(dst, {}).get("type") == "Evidence":
            evidence_by_transition.setdefault(src, []).append(dst)
        if kind == "distills_to" and nodes.get(dst, {}).get("type") == "SOP":
            sops_by_transition.setdefault(src, []).append(dst)

    candidates = []
    for transition_id, sop_ids in sops_by_transition.items():
        transition = nodes.get(transition_id, {})
        parent = nodes.get(str(transition.get("parent_node_id") or ""), {})
        child = nodes.get(str(transition.get("child_node_id") or ""), {})
        if not parent or not child or not _clean(child):
            continue
        task = str(child.get("task") or parent.get("task") or "")
        stage = _stage(child.get("stage") or transition.get("stage_pair"))
        context_parts = [
            f"Task: {task}",
            f"Agent stage: {stage}",
            f"Current plan: {str(parent.get('plan') or '')[:700]}",
            f"Current code summary: {str(parent.get('code_summary') or '')[:700]}",
        ]
        terminal = str(parent.get("terminal_excerpt") or "")[:700]
        if terminal:
            context_parts.append(f"Observed execution output: {terminal}")
        if stage == "debug":
            context_parts.append("Find a previously verified recovery path for this failure without copying blocked code.")
        elif stage == "improve":
            context_parts.append("Find a verified change that improved a comparable branch and explain its evidence.")
        else:
            context_parts.append("Find a suitable method and verify it against real execution lineage.")
        query_text = "\n".join(context_parts)
        run_id = str(child.get("run_short_id") or child.get("run_id") or "")
        query_id = "stagehybrid::" + hashlib.sha256(
            f"{run_id}|{transition_id}|{stage}".encode("utf-8")
        ).hexdigest()[:16]
        gold_execution = [str(transition.get("child_node_id"))]
        local_best = child.get("local_best_node_id")
        if local_best and str(local_best) not in gold_execution:
            gold_execution.append(str(local_best))
        candidates.append(
            (
                run_id,
                int(child.get("step") or 0),
                {
                    "query_id": query_id,
                    "run_id": run_id,
                    "task": task,
                    "stage": stage,
                    "split": split_for_run(run_id),
                    "query_text": query_text,
                },
                {
                    "query_id": query_id,
                    "run_id": run_id,
                    "split": split_for_run(run_id),
                    "gold_sop_ids": sorted(set(sop_ids)),
                    "gold_transition_ids": [transition_id],
                    "gold_execution_ids": gold_execution,
                    "gold_evidence_ids": sorted(set(evidence_by_transition.get(transition_id, []))),
                    "gold_stage": stage,
                },
            )
        )
    candidates.sort(key=lambda row: (row[0], row[1], row[2]["query_id"]))
    if len(candidates) > max_queries:
        # Deterministic round-robin keeps run diversity instead of taking one long run.
        by_run: dict[str, list[tuple]] = {}
        for row in candidates:
            by_run.setdefault(row[0], []).append(row)
        selected = []
        while len(selected) < max_queries and any(by_run.values()):
            for run_id in sorted(by_run):
                if by_run[run_id] and len(selected) < max_queries:
                    selected.append(by_run[run_id].pop(0))
        candidates = selected
    return [row[2] for row in candidates], [row[3] for row in candidates]


def validate_records(queries: list[dict], gold: list[dict]) -> dict[str, Any]:
    gold_by_id = {row["query_id"]: row for row in gold}
    errors = []
    splits_by_run: dict[str, set[str]] = {}
    for query in queries:
        target = gold_by_id.get(query["query_id"])
        if target is None:
            errors.append(f"missing_gold:{query['query_id']}")
            continue
        splits_by_run.setdefault(query["run_id"], set()).add(query["split"])
        if query["run_id"] != target["run_id"] or query["split"] != target["split"]:
            errors.append(f"query_gold_mismatch:{query['query_id']}")
        if not target["gold_sop_ids"] or not target["gold_transition_ids"]:
            errors.append(f"empty_gold:{query['query_id']}")
        leaked_ids = target["gold_transition_ids"] + target["gold_execution_ids"]
        if any(node_id and node_id in query["query_text"] for node_id in leaked_ids):
            errors.append(f"id_leakage:{query['query_id']}")
    for run_id, splits in splits_by_run.items():
        if len(splits) != 1:
            errors.append(f"run_split_leakage:{run_id}:{sorted(splits)}")
    split_counts = {}
    stage_counts = {}
    for query in queries:
        split_counts[query["split"]] = split_counts.get(query["split"], 0) + 1
        stage_counts[query["stage"]] = stage_counts.get(query["stage"], 0) + 1
    return {
        "schema": "stage_hybrid_benchmark_validation_v1",
        "valid": not errors,
        "errors": errors,
        "query_count": len(queries),
        "run_count": len(splits_by_run),
        "split_counts": split_counts,
        "stage_counts": stage_counts,
        "grouped_by_run": all(len(value) == 1 for value in splits_by_run.values()),
        "natural_language_no_gold_coordinates": True,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--benchmark-out", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--gold-out", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-queries", type=int, default=240)
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    queries, gold = build_records(graph, max_queries=args.max_queries)
    report = validate_records(queries, gold)
    if not report["valid"]:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    _write_jsonl(args.benchmark_out, queries)
    _write_jsonl(args.gold_out, gold)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
