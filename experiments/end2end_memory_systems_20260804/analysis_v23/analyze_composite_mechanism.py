#!/usr/bin/env python3
"""Run mechanism analysis only after the composite terminal table is materialized."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "manifests_v23"
TERMINAL_MODULE_PATH = Path(__file__).with_name("analyze_composite_terminal.py")

sys.path.insert(0, str(ROOT))
try:
    import analyze_results as base_analysis
finally:
    sys.path.pop(0)

_spec = importlib.util.spec_from_file_location("composite_terminal", TERMINAL_MODULE_PATH)
terminal = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(terminal)


COUNTERS = (
    "routing_routes",
    "native_routing_trace_routes",
    "reconstructed_routing_trace_routes",
    "raw_candidates",
    "prompt_visible",
    "suppressed",
    "exact_replay_selected",
    "exact_replay_executed",
    "static_adopted",
    "runtime_activated",
    "adopted",
    "partially_adopted",
    "rejected",
    "uncertain",
    "plan_covered",
    "valid_agent_evidence_routes",
    "invalid_agent_evidence_routes",
)


def _empty_counts() -> dict[str, int]:
    return {key: 0 for key in COUNTERS}


def _rates(counts: Mapping[str, int]) -> dict[str, Any]:
    visible = int(counts["prompt_visible"])
    raw = int(counts["raw_candidates"])
    replay = int(counts["exact_replay_selected"])
    covered = int(counts["plan_covered"])
    return {
        "suppression_rate": counts["suppressed"] / raw if raw else None,
        "static_adoption_rate": (
            counts["static_adopted"] / covered if covered else None
        ),
        "runtime_activation_rate": (
            counts["runtime_activated"] / covered if covered else None
        ),
        "agent_adoption_rate": (
            (counts["adopted"] + counts["partially_adopted"]) / covered
            if covered
            else None
        ),
        "exact_replay_execution_rate": (
            counts["exact_replay_executed"] / replay if replay else None
        ),
        "adoption_observable": covered > 0,
        "prompt_visible_without_adoption_plan": max(0, visible - covered),
    }


def aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _empty_counts()
    by_system: dict[str, dict[str, Any]] = {}
    by_task: dict[str, dict[str, Any]] = {}
    by_stage: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in COUNTERS:
            totals[key] += int(row.get(key) or 0)
        for group, field in ((by_system, "system_id"), (by_task, "task_id")):
            name = str(row[field])
            entry = group.setdefault(name, {"runs": 0, **_empty_counts()})
            entry["runs"] += 1
            for key in COUNTERS:
                entry[key] += int(row.get(key) or 0)
        for stage, counts in (row.get("by_stage") or {}).items():
            entry = by_stage.setdefault(str(stage), _empty_counts())
            for key in COUNTERS:
                entry[key] += int((counts or {}).get(key) or 0)
    for group in (by_system, by_task):
        for entry in group.values():
            entry.update(_rates(entry))
    staged = {}
    for stage, counts in by_stage.items():
        staged[stage] = {**counts, **_rates(counts)}
    return {
        "totals": {**totals, **_rates(totals)},
        "by_system": dict(sorted(by_system.items())),
        "by_task": dict(sorted(by_task.items())),
        "by_stage": dict(sorted(staged.items())),
    }


def scoped_outcomes(
    terminal_summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    retained = [
        dict(row)
        for row in terminal_summary.get("cells") or []
        if row.get("status") != "missing"
    ]
    formal = []
    for row in retained:
        scoped = dict(row)
        formal_paths = list(row.get("formal_journal_paths") or [])
        scoped["retained_journal_paths"] = formal_paths
        scoped["journal_path"] = formal_paths[-1] if formal_paths else None
        formal.append(scoped)
    return formal, retained


def build_report(
    terminal_summary: Mapping[str, Any],
    *,
    manifests: Path,
    allow_incomplete: bool,
) -> dict[str, Any]:
    terminal.verify_hash(terminal_summary, "summary_hash", "Terminal summary")
    observed = int(terminal_summary.get("observed_terminal_outcomes") or 0)
    if observed != 40 and not allow_incomplete:
        raise ValueError(
            f"Mechanism analysis requires 40 terminal outcomes; observed={observed}"
        )
    outcomes, retained_outcomes = scoped_outcomes(terminal_summary)
    mechanism = base_analysis.mechanism_summary(outcomes, manifests=manifests)
    rows = list(mechanism["runs"])
    for row in rows:
        row.update(_rates(row))
    retained_mechanism = base_analysis.mechanism_summary(
        retained_outcomes, manifests=manifests
    )
    retained_rows = list(retained_mechanism["runs"])
    for row in retained_rows:
        row.update(_rates(row))
    report = {
        **mechanism,
        "schema": "mlevolve_end2end_composite_mechanism_summary_v1",
        "analysis_order": 2,
        "terminal_summary_hash": terminal_summary["summary_hash"],
        "expected_cells": 40,
        "observed_terminal_outcomes": observed,
        "completed_cells": int(terminal_summary.get("completed_cells") or 0),
        "coverage": {
            "runs_with_journal": sum(
                any(
                    Path(str(path)).is_file()
                    for path in (
                        row.get("retained_journal_paths")
                        or [row.get("journal_path")]
                    )
                    if path
                )
                for row in outcomes
            ),
            "runs_with_routing_trace": sum(
                int(row.get("routing_routes") or 0) > 0 for row in rows
            ),
            "runs_with_runtime_activation": sum(
                int(row.get("runtime_activated") or 0) > 0 for row in rows
            ),
            "runs_with_adoption_observation_plan": sum(
                int(row.get("plan_covered") or 0) > 0 for row in rows
            ),
            "runs_with_prompt_exposure_but_unobserved_adoption": sum(
                int(row.get("prompt_visible") or 0) > 0
                and int(row.get("plan_covered") or 0) == 0
                for row in rows
            ),
        },
        "aggregate": aggregate_runs(rows),
        "retained_operational_analysis": {
            "scope": "all retained attempts including failed, diagnostic, and cancelled",
            "runs": retained_rows,
            "aggregate": aggregate_runs(retained_rows),
        },
        "summary_hash": "",
    }
    report["definitions"].update(
        {
            "adoption_observability": (
                "static adoption and runtime activation rates use only "
                "Prompt-visible items covered by an adoption observation plan"
            ),
            "missing_probe_disposition": (
                "unobserved; never interpreted as zero activation"
            ),
            "formal_mechanism_scope": (
                "routing, suppression, adoption, and activation from formal-result-eligible "
                "attempts only"
            ),
            "retained_operational_scope": (
                "separate diagnostics over every retained failed, adapter, retry, and "
                "operator-cancelled attempt"
            ),
        }
    )
    report["summary_hash"] = terminal.payload_hash(report, "summary_hash")
    return report


def write_outputs(root: Path, report: Mapping[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "mechanism_summary.json").write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = ["logical_run_id", "task_id", "system_id", *COUNTERS]
    with (root / "mechanism_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["runs"]:
            writer.writerow({key: row.get(key) for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-summary", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=MANIFESTS)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    terminal_summary = terminal.read_object(args.terminal_summary)
    report = build_report(
        terminal_summary,
        manifests=args.manifests,
        allow_incomplete=args.allow_incomplete,
    )
    write_outputs(args.analysis_root, report)
    print(
        json.dumps(
            {
                "observed_terminal_outcomes": report["observed_terminal_outcomes"],
                "runs_with_journal": report["coverage"]["runs_with_journal"],
                "routing_routes": report["aggregate"]["totals"]["routing_routes"],
                "runtime_activated": report["aggregate"]["totals"]["runtime_activated"],
                "exploratory_pilot": True,
                "summary_hash": report["summary_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
