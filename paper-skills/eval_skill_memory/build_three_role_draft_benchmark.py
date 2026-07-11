#!/usr/bin/env python3
"""Build a multi-gold Draft benchmark around the three-role protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
REPLAY_TARGETS = REPO / "paper-skills" / "eval_skill_memory" / "clean_replay_targets.json"
BENCHMARK = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "three_role_draft_benchmark.jsonl"
GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "three_role_draft_gold.jsonl"
REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "three_role_draft_benchmark_validation.json"

TASK_CONTEXT = {
    "spooky-author-identification": "English text authorship classification using train-only features and untouched validation.",
    "aerial-cactus-identification": "Binary image classification for aerial cactus photographs.",
    "leaf-classification": "Multiclass leaf image and tabular-feature classification.",
    "new-york-city-taxi-fare-prediction": "Tabular regression for taxi fares with spatial and temporal features.",
    "denoising-dirty-documents": "Document image denoising and reconstruction.",
}


def _clean(node: dict[str, Any]) -> bool:
    audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
    return (
        audit.get("status") == "clean"
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
    )


def _balanced_splits(run_ids: list[str]) -> dict[str, str]:
    ordered = sorted(run_ids, key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    count = len(ordered)
    train_end = max(1, round(count * 0.60))
    dev_end = min(count - 1, train_end + max(1, round(count * 0.20))) if count >= 3 else count
    return {
        run_id: "train" if index < train_end else "dev" if index < dev_end else "test"
        for index, run_id in enumerate(ordered)
    }


def build_three_role_records(graph: dict, replay_manifest: dict) -> tuple[list[dict], list[dict]]:
    nodes = {str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")}
    replay_by_task = {str(item["task_id"]): item for item in replay_manifest.get("targets", [])}
    sop_tasks: dict[str, set[str]] = {}
    root_rows: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        if node.get("type") != "Transition":
            continue
        parent = nodes.get(str(node.get("parent_node_id") or ""), {})
        child = nodes.get(str(node.get("child_node_id") or ""), {})
        if parent.get("stage") != "root" or child.get("stage") != "draft" or not _clean(child):
            continue
        run_id = str(child.get("run_short_id") or child.get("run_id") or "")
        task = str(child.get("task") or "")
        if task not in replay_by_task:
            # A complete three-role episode requires a declared replay target.
            continue
        row = root_rows.setdefault(
            run_id,
            {
                "run_id": run_id,
                "task": task,
                "draft_transition_ids": [],
                "clean_draft_ids": [],
                "successful_draft_ids": [],
                "sop_ids": [],
            },
        )
        row["draft_transition_ids"].append(node["id"])
        row["clean_draft_ids"].append(child["id"])
        if child.get("is_buggy") is False and child.get("is_valid") is not False:
            row["successful_draft_ids"].append(child["id"])
        for sop_id in node.get("attached_sop_ids") or []:
            row["sop_ids"].append(str(sop_id))
            sop_tasks.setdefault(str(sop_id), set()).add(task)

    split_by_run = _balanced_splits(sorted(root_rows))
    queries, gold = [], []
    for run_id, row in sorted(root_rows.items()):
        task = row["task"]
        replay = replay_by_task[task]
        all_sops = sorted(set(row["sop_ids"]))
        discriminative_sops = [sop_id for sop_id in all_sops if len(sop_tasks.get(sop_id, set())) <= 2]
        gold_sops = discriminative_sops or all_sops
        if not gold_sops:
            continue
        query_id = "three-role-draft::" + hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
        query_text = "\n".join(
            [
                f"Task: {task}",
                "Agent stage: draft",
                f"Task context: {TASK_CONTEXT.get(task, task.replace('-', ' '))}",
                "Generate a distinct initial approach for the novel_exploration slot.",
                "Use only clean historical methods from other runs; do not copy the fixed replay target.",
            ]
        )
        roles = {
            "coldstart_baseline": {"memory_mode": "none", "fixed_across_conditions": True},
            "memory_reproduction": {
                "memory_mode": "blocked_exact_source_repair_seed" if replay.get("audit_status") == "candidate_replay" else "exact_code_replay",
                "fixed_across_conditions": True,
                "run_id": replay["run_id"],
                "original_node_id": replay["original_node_id"],
                "audit_status": replay["audit_status"],
                "method_family": replay.get("method_family"),
                "known_issue_codes": replay.get("known_issue_codes") or [],
            },
            "novel_exploration": {
                "compared_conditions": ["tree_only", "stage_hybrid", "layered_strategy"],
                "must_exclude_query_split_runs": True,
            },
        }
        queries.append(
            {
                "query_id": query_id,
                "run_id": run_id,
                "task": task,
                "stage": "draft",
                "split": split_by_run[run_id],
                "query_text": query_text,
                "three_role_protocol": roles,
            }
        )
        gold.append(
            {
                "query_id": query_id,
                "run_id": run_id,
                "task": task,
                "split": split_by_run[run_id],
                "relevant_sop_ids": gold_sops,
                "acceptable_clean_draft_ids": sorted(set(row["clean_draft_ids"])),
                "successful_draft_ids": sorted(set(row["successful_draft_ids"])),
                "source_transition_ids": sorted(set(row["draft_transition_ids"])),
                "gold_policy": "multi_gold_method_family_not_single_child",
            }
        )
    return queries, gold


def validate(queries: list[dict], gold: list[dict]) -> dict:
    gold_by_id = {row["query_id"]: row for row in gold}
    errors = []
    splits_by_run: dict[str, set[str]] = {}
    for query in queries:
        target = gold_by_id.get(query["query_id"])
        if target is None:
            errors.append(f"missing_gold:{query['query_id']}")
            continue
        splits_by_run.setdefault(query["run_id"], set()).add(query["split"])
        roles = query["three_role_protocol"]
        if list(roles) != ["coldstart_baseline", "memory_reproduction", "novel_exploration"]:
            errors.append(f"role_order:{query['query_id']}")
        if not roles["coldstart_baseline"]["fixed_across_conditions"] or not roles["memory_reproduction"]["fixed_across_conditions"]:
            errors.append(f"fixed_role_violation:{query['query_id']}")
        if target["gold_policy"] != "multi_gold_method_family_not_single_child":
            errors.append(f"single_gold_policy:{query['query_id']}")
        if not target["relevant_sop_ids"] or not target["acceptable_clean_draft_ids"]:
            errors.append(f"empty_multi_gold:{query['query_id']}")
        forbidden = target["acceptable_clean_draft_ids"] + target["source_transition_ids"]
        if any(node_id in query["query_text"] for node_id in forbidden):
            errors.append(f"id_leakage:{query['query_id']}")
    for run_id, splits in splits_by_run.items():
        if len(splits) != 1:
            errors.append(f"run_split_leakage:{run_id}")
    split_counts = {name: sum(query["split"] == name for query in queries) for name in ("train", "dev", "test")}
    return {
        "schema": "three_role_draft_benchmark_validation_v1",
        "valid": not errors,
        "errors": errors,
        "query_count": len(queries),
        "run_count": len(splits_by_run),
        "split_counts": split_counts,
        "one_query_per_root_run": len(queries) == len(splits_by_run),
        "multi_gold": True,
        "three_role_protocol": True,
        "retrieval_memory_must_exclude_query_split_runs": True,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=GRAPH)
    parser.add_argument("--replay-targets", type=Path, default=REPLAY_TARGETS)
    parser.add_argument("--benchmark-out", type=Path, default=BENCHMARK)
    parser.add_argument("--gold-out", type=Path, default=GOLD)
    parser.add_argument("--report-out", type=Path, default=REPORT)
    args = parser.parse_args()
    queries, gold = build_three_role_records(
        json.loads(args.graph.read_text(encoding="utf-8")),
        json.loads(args.replay_targets.read_text(encoding="utf-8")),
    )
    report = validate(queries, gold)
    if not report["valid"]:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    _write_jsonl(args.benchmark_out, queries)
    _write_jsonl(args.gold_out, gold)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
