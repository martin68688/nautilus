#!/usr/bin/env python3
"""Explain frozen benchmark coverage gaps without relaxing the gold policy."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from build_composite_benchmark import _eligible_sops, _family_compatible
from core import (
    ARTIFACTS,
    EPISODES,
    GRAPH,
    MANIFESTS,
    REPORTS,
    clean_support,
    read_json,
    read_jsonl,
    run_id_for_node,
    write_json,
)


def _eligible_rows(
    graph: dict[str, Any],
    support: dict[str, list[str]],
    *,
    family: str,
    stage: str,
    primary_family: str | None,
) -> list[dict[str, Any]]:
    rows = _eligible_sops(graph, support, family, stage)
    if stage == "model_design":
        if not primary_family:
            return []
        rows = [
            row
            for row in rows
            if row.get("abstraction_level") == "L2_tactic"
            and _family_compatible(primary_family, str(row.get("method_family") or "general"))
        ]
    return rows


def audit() -> dict[str, Any]:
    snapshot = read_json(ARTIFACTS / "memory_snapshot_graph_v1.json")
    full_graph = read_json(GRAPH)
    memory_manifest = read_json(MANIFESTS / "memory_snapshot_manifest_v1.json")
    train_runs = set(memory_manifest["run_split"]["memory_train"])
    full_nodes = {str(node["id"]): node for node in full_graph["nodes"] if node.get("id")}
    snapshot_support = clean_support(snapshot)
    full_support = clean_support(full_graph)
    queries = read_jsonl(EPISODES / "decision_test_v1.jsonl")
    gold = {
        row["episode_id"]: row
        for row in read_jsonl(EPISODES / "decision_test_silver_gold_v1.jsonl")
    }

    rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    family_stage_counts: Counter[tuple[str, str]] = Counter()
    for query in queries:
        if query["expected_status"] != "insufficient_strategy_coverage":
            continue
        family = str(query["retrieval_family"])
        stage = str(query["stage"])
        primary_family = query.get("primary_method_family")
        frozen_eligible = _eligible_rows(
            snapshot,
            snapshot_support,
            family=family,
            stage=stage,
            primary_family=primary_family,
        )
        full_eligible = _eligible_rows(
            full_graph,
            full_support,
            family=family,
            stage=stage,
            primary_family=primary_family,
        )
        frozen_count = len(frozen_eligible)
        full_count = len(full_eligible)
        frozen_families = len({str(row.get("method_family") or "general") for row in frozen_eligible})
        full_families = len({str(row.get("method_family") or "general") for row in full_eligible})
        heldout_support_runs = {
            run_id_for_node(full_nodes[transition_id])
            for sop in full_eligible
            for transition_id in full_support.get(str(sop["id"]), [])
            if transition_id in full_nodes and run_id_for_node(full_nodes[transition_id]) not in train_runs
        }
        if full_count < 3:
            reason = "full_graph_has_fewer_than_three_clean_compatible_sops"
        elif frozen_count < 3:
            reason = "source_run_split_removed_required_clean_support"
        else:
            reason = "gold_selection_or_manifest_inconsistency"
        reason_counts[reason] += 1
        family_stage_counts[(str(query["task_family"]), stage)] += 1
        rows.append(
            {
                "episode_id": query["episode_id"],
                "task_family": query["task_family"],
                "retrieval_family": family,
                "stage": stage,
                "primary_method_family": primary_family,
                "silver_label_count": len(gold[query["episode_id"]]["labels"]),
                "frozen_clean_compatible_sop_count": frozen_count,
                "frozen_compatible_method_family_count": frozen_families,
                "full_graph_clean_compatible_sop_count": full_count,
                "full_graph_compatible_method_family_count": full_families,
                "additional_clean_compatible_sops_needed": max(0, 3 - full_count),
                "heldout_support_run_count": len(heldout_support_runs),
                "reason": reason,
            }
        )

    report = {
        "schema": "runforest_composite_coverage_audit_v1",
        "test_episode_count": len(queries),
        "coverage_gap_count": len(rows),
        "coverage_complete_count": len(queries) - len(rows),
        "coverage_complete_rate": (len(queries) - len(rows)) / max(1, len(queries)),
        "reason_counts": dict(sorted(reason_counts.items())),
        "gaps_by_task_family_and_stage": {
            f"{family}|{stage}": count
            for (family, stage), count in sorted(family_stage_counts.items())
        },
        "full_graph_support_removed_by_frozen_split_count": reason_counts.get(
            "source_run_split_removed_required_clean_support", 0
        ),
        "requires_genuinely_new_clean_evidence_count": reason_counts.get(
            "full_graph_has_fewer_than_three_clean_compatible_sops", 0
        ),
        "can_be_fixed_by_resplitting_without_invalidating_frozen_source_separation": not any(
            row["reason"] == "source_run_split_removed_required_clean_support"
            and row["heldout_support_run_count"] > 0
            for row in rows
        ),
        "required_remediation": (
            "Collect and independently audit new clean L1/L2 evidence for the uncovered task-family/stage "
            "cells, then freeze a v2 memory snapshot. Twenty gaps also have support in held-out runs, but moving "
            "that evidence into memory would invalidate the frozen source split. Do not backfill with L3 repairs "
            "or incompatible families."
        ),
        "episodes": rows,
    }
    write_json(REPORTS / "coverage_audit_v1.json", report)
    return report


def main() -> None:
    print(json.dumps(audit(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
