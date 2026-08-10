#!/usr/bin/env python3
"""Verify v74 active Strategy evidence, decomposition, and online outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    journals = list(args.run_root.rglob("journal.json"))
    if len(journals) != 1:
        raise ValueError(f"expected one journal, found {len(journals)}")
    journal_path = journals[0]
    journal = read_object(journal_path)
    nodes = [dict(node) for node in (journal.get("nodes") or [])]
    active_nodes: dict[str, list[dict[str, Any]]] = {"improve": [], "debug": []}
    legacy_stage_nodes: list[str] = []
    rejection_nodes: list[dict[str, str]] = []
    evidence_complete_nodes: list[str] = []
    decomposition_nodes: list[str] = []

    for node in nodes:
        strategy = dict(node.get("memory_strategy_trace") or {})
        atomic = dict(node.get("atomic_actuation_trace") or {})
        # Strategy traces are attached to the parent/produced node.  An Improve
        # transaction may therefore live on a node whose search-tree stage is
        # still ``draft``.  The Strategy/Atomic transaction stage is the
        # authoritative stage for actuation verification; node.stage is only
        # a compatibility fallback for legacy generation.
        stage = str(
            strategy.get("stage")
            or atomic.get("stage")
            or node.get("stage")
            or ""
        )
        observation = dict(node.get("protocol_observation") or {})
        rejection = dict(observation.get("memory_strategy_active_rejection") or {})
        if rejection:
            rejection_nodes.append(
                {
                    "node_id": str(node.get("id") or ""),
                    "stage": str(rejection.get("stage") or stage),
                    "reason": str(rejection.get("reason") or ""),
                }
            )
        if stage not in active_nodes:
            continue
        if strategy.get("mode") != "active_atomic":
            legacy_stage_nodes.append(str(node.get("id") or ""))
            continue
        active_nodes[stage].append(node)
        selection = dict(strategy.get("strategy_evidence_selection") or {})
        historical_ids = list(selection.get("historical_diverse_frontier_ids") or [])
        if historical_ids:
            evidence_complete_nodes.append(str(node.get("id") or ""))
        atomic = dict(node.get("atomic_actuation_trace") or {})
        if atomic.get("decomposition_used") is True:
            decomposition_nodes.append(str(node.get("id") or ""))

    def accepted(node: dict[str, Any], *, require_clean: bool) -> bool:
        strategy = dict(node.get("memory_strategy_trace") or {})
        atomic = dict(node.get("atomic_actuation_trace") or {})
        planner = dict(atomic.get("planner") or {})
        coder = dict(atomic.get("coder") or {})
        initial = dict(coder.get("plan_diff_verdict") or {})
        post_review = dict(atomic.get("post_review_plan_diff_verdict") or {})
        attempts = list(atomic.get("actuation_attempts") or [])
        return bool(
            strategy.get("status") == "completed"
            and strategy.get("actuation_authority") == "atomic_planner_coder"
            and strategy.get("noninterference_verified") is True
            and (strategy.get("validation") or {}).get("valid") is True
            and (strategy.get("memo") or {}).get("decision") == "propose"
            and atomic.get("status") == "accepted"
            and planner.get("status") == "accepted"
            and coder.get("status") == "accepted"
            and initial.get("valid") is True
            and post_review.get("valid") is True
            and attempts
            and node.get("exec_time") is not None
            and (not require_clean or node.get("is_buggy") is False)
            and "memory_strategy_active_atomic" in str(node.get("prompt_input") or "")
        )

    improve_accepted = [
        node for node in active_nodes["improve"] if accepted(node, require_clean=False)
    ]
    debug_accepted = [
        node for node in active_nodes["debug"] if accepted(node, require_clean=False)
    ]
    improve_clean = [
        node for node in active_nodes["improve"] if accepted(node, require_clean=True)
    ]
    debug_clean = [
        node for node in active_nodes["debug"] if accepted(node, require_clean=True)
    ]
    numeric_metrics = []
    for node in nodes:
        metric = node.get("metric")
        value = metric.get("value") if isinstance(metric, dict) else None
        if isinstance(value, (int, float)) and not node.get("is_buggy"):
            numeric_metrics.append(float(value))

    measurements = list(args.run_root.rglob("MEASUREMENT.json"))
    measurement = read_object(measurements[0]) if len(measurements) == 1 else {}
    launches = list(args.run_root.rglob("LAUNCH_RECEIPT.json"))
    launch = read_object(launches[0]) if len(launches) == 1 else {}
    checks = {
        "terminal_measurement_present": len(measurements) == 1,
        "launch_receipt_present": len(launches) == 1,
        "search_budget_is_six_hours": int(
            (launch.get("budget") or {}).get("agent_time_limit_seconds") or 0
        )
        == 21600,
        "journal_has_nodes": len(nodes) > 1,
        "no_legacy_improve_or_debug_generation": not legacy_stage_nodes,
        "historical_method_evidence_reaches_strategy": bool(evidence_complete_nodes),
        "improve_strategy_actuated_and_executed": bool(improve_accepted),
        "debug_strategy_actuated_and_executed": bool(debug_accepted),
        "improve_produces_clean_candidate": bool(improve_clean),
        "debug_produces_clean_candidate": bool(debug_clean),
        "no_required_strategy_rejection": not rejection_nodes,
    }
    receipt = {
        "schema": "mlevolve_leaf_strategy_active_smoke_receipt_v2",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "journal": str(journal_path),
        "journal_sha256": sha256_file(journal_path),
        "node_count": len(nodes),
        "active_improve_node_ids": [node.get("id") for node in active_nodes["improve"]],
        "accepted_improve_node_ids": [node.get("id") for node in improve_accepted],
        "clean_improve_node_ids": [node.get("id") for node in improve_clean],
        "active_debug_node_ids": [node.get("id") for node in active_nodes["debug"]],
        "accepted_debug_node_ids": [node.get("id") for node in debug_accepted],
        "clean_debug_node_ids": [node.get("id") for node in debug_clean],
        "historical_evidence_node_ids": evidence_complete_nodes,
        "decomposition_node_ids": decomposition_nodes,
        "legacy_stage_node_ids": legacy_stage_nodes,
        "rejections": rejection_nodes,
        "best_internal_metric": min(numeric_metrics) if numeric_metrics else None,
        "measurement": str(measurements[0]) if len(measurements) == 1 else "",
        "launch_receipt": str(launches[0]) if len(launches) == 1 else "",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
