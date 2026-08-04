#!/usr/bin/env python3
"""Analyze End2End outcomes first, then complete RunForest mechanism traces.

Seed 1 is always labelled exploratory.  The analyzer does not compute p-values
or confidence intervals and never imputes failed terminal scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parent
MANIFESTS = ROOT / "manifests"
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
STOP = {
    "the", "and", "for", "with", "from", "this", "that", "model",
    "memory", "candidate", "procedure", "stage", "task", "using",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify(payload: Mapping[str, Any], field: str, label: str) -> None:
    if payload_hash(payload, field) != payload.get(field):
        raise ValueError(f"{label} hash mismatch")


def load_measurements(output_root: Path, *, formal_only: bool) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_root.rglob("MEASUREMENT.json")):
        row = read_object(path)
        verify(row, "measurement_hash", str(path))
        if formal_only and row.get("formal_result_eligible") is not True:
            continue
        row["_path"] = str(path)
        rows.append(row)
    return rows


def select_logical_outcomes(
    measurements: Iterable[dict[str, Any]], expected_ids: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for row in measurements:
        by_run.setdefault(str(row["logical_run_id"]), []).append(row)
    unknown = set(by_run) - expected_ids
    if unknown:
        raise ValueError(f"Measurements outside frozen manifest: {sorted(unknown)}")
    outcomes = []
    attempts = []
    for run_id in sorted(expected_ids):
        rows = sorted(by_run.get(run_id, []), key=lambda item: int(item["attempt"]))
        attempts.extend(rows)
        successful = [row for row in rows if row.get("completed") is True]
        if successful:
            outcomes.append(successful[0])
        elif rows:
            outcomes.append(rows[-1])
    return outcomes, attempts


def signed_delta(score: float, baseline: float, direction: str) -> float:
    raw = score - baseline if direction == "maximize" else baseline - score
    return raw / max(abs(baseline), 1e-12)


def _rank_scores(rows: list[dict[str, Any]], direction: str) -> dict[str, float]:
    completed = [row for row in rows if row.get("completed")]
    ordered = sorted(
        completed,
        key=lambda row: float(row["terminal_score"]),
        reverse=direction == "maximize",
    )
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        score = float(ordered[index]["terminal_score"])
        end = index
        while end + 1 < len(ordered) and math.isclose(
            float(ordered[end + 1]["terminal_score"]), score
        ):
            end += 1
        rank = (index + 1 + end + 1) / 2
        for item in ordered[index : end + 1]:
            ranks[item["system_id"]] = rank
        index = end + 1
    for row in rows:
        ranks.setdefault(row["system_id"], float(len(rows) + 1))
    return ranks


def terminal_summary(
    outcomes: list[dict[str, Any]], task_ids: list[str], system_ids: list[str]
) -> dict[str, Any]:
    by_cell = {(row["task_id"], row["system_id"]): row for row in outcomes}
    systems = []
    task_ranks: dict[tuple[str, str], float] = {}
    for task_id in task_ids:
        task_rows = [row for row in outcomes if row["task_id"] == task_id]
        if task_rows:
            direction = str(task_rows[0]["direction"])
            for system_id, rank in _rank_scores(task_rows, direction).items():
                task_ranks[(task_id, system_id)] = rank
    for system_id in system_ids:
        cells = []
        negative_observed = 0
        negative_count = 0
        for task_id in task_ids:
            row = by_cell.get((task_id, system_id))
            baseline = by_cell.get((task_id, "no_memory"))
            delta = None
            negative = None
            if row and baseline and baseline.get("completed"):
                negative_observed += 1
                if row.get("completed"):
                    delta = signed_delta(
                        float(row["terminal_score"]),
                        float(baseline["terminal_score"]),
                        str(row["direction"]),
                    )
                    negative = delta < 0
                else:
                    negative = True
                negative_count += int(negative)
            cells.append(
                {
                    "task_id": task_id,
                    "completed": bool(row and row.get("completed")),
                    "terminal_score": row.get("terminal_score") if row else None,
                    "status": row.get("status") if row else "missing",
                    "normalized_delta_vs_no_memory": delta,
                    "negative_transfer": negative,
                    "time_to_first_valid_seconds": (
                        row.get("time_to_first_valid_seconds") if row else None
                    ),
                    "allocated_gpu_hours": (
                        row.get("allocated_gpu_hours") if row else None
                    ),
                    "llm_token_usage": row.get("llm_token_usage") if row else None,
                    "llm_cost_usd": row.get("llm_cost_usd") if row else None,
                    "task_rank": task_ranks.get((task_id, system_id)),
                }
            )
        completed = sum(cell["completed"] for cell in cells)
        systems.append(
            {
                "system_id": system_id,
                "completed_tasks": completed,
                "completion_rate": completed / len(task_ids),
                "negative_transfer_count": negative_count,
                "negative_transfer_observed_tasks": negative_observed,
                "negative_transfer_rate": (
                    negative_count / negative_observed if negative_observed else None
                ),
                "mean_task_rank": (
                    sum(cell["task_rank"] for cell in cells if cell["task_rank"] is not None)
                    / len([cell for cell in cells if cell["task_rank"] is not None])
                ),
                "allocated_gpu_hours": sum(
                    float(cell["allocated_gpu_hours"] or 0.0) for cell in cells
                ),
                "cells": cells,
            }
        )
    return {
        "schema": "mlevolve_end2end_terminal_summary_v1",
        "analysis_order": 1,
        "exploratory_pilot": True,
        "seed": 1,
        "statistical_significance_claim_allowed": False,
        "missing_scores_imputed": False,
        "systems": systems,
        "summary_hash": "",
    }


def _journal_nodes(payload: object) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("nodes", "journal", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in TOKEN.findall(str(text or "").replace("_", " "))
        if token.lower() not in STOP
    }


def _static_adoption(code: str, memory_text: str) -> bool:
    return len(_tokens(code) & _tokens(memory_text)) >= 2


def mechanism_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for outcome in outcomes:
        journal_path = Path(str(outcome.get("journal_path") or ""))
        counters = {
            "raw_candidates": 0,
            "prompt_visible": 0,
            "suppressed": 0,
            "static_adopted": 0,
            "runtime_activated": 0,
        }
        by_stage: dict[str, dict[str, int]] = {}
        if journal_path.is_file():
            nodes = _journal_nodes(json.loads(journal_path.read_text(encoding="utf-8")))
            for node in nodes:
                route = node.get("memory_routing_trace") or {}
                if route.get("schema") != "mlevolve_memory_routing_trace_v1":
                    continue
                stage = str((route.get("stage_route") or {}).get("stage") or node.get("stage") or "unknown")
                stage_counts = by_stage.setdefault(
                    stage,
                    {"raw_candidates": 0, "prompt_visible": 0, "suppressed": 0,
                     "static_adopted": 0, "runtime_activated": 0},
                )
                selected = {
                    str(item.get("candidate_id")): item
                    for item in route.get("selected_candidates") or []
                    if isinstance(item, dict)
                }
                visible = list(route.get("final_prompt_candidate_ids") or [])
                raw_count = len(route.get("raw_candidates") or [])
                suppressed_count = len(
                    {
                        str(item.get("candidate_id") or "")
                        for item in route.get("suppressed_candidates") or []
                        if isinstance(item, dict)
                    }
                )
                counters["raw_candidates"] += raw_count
                counters["prompt_visible"] += len(visible)
                counters["suppressed"] += suppressed_count
                stage_counts["raw_candidates"] += raw_count
                stage_counts["prompt_visible"] += len(visible)
                stage_counts["suppressed"] += suppressed_count
                code = str(node.get("code") or "")
                runtime_pass = (
                    ((node.get("protocol_observation") or {}).get("host_full_runtime") or {}).get("status")
                    == "pass"
                )
                for candidate_id in visible:
                    candidate = selected.get(str(candidate_id)) or {}
                    adopted = _static_adoption(code, str(candidate.get("prompt_text") or ""))
                    if adopted:
                        counters["static_adopted"] += 1
                        stage_counts["static_adopted"] += 1
                    if adopted and runtime_pass:
                        counters["runtime_activated"] += 1
                        stage_counts["runtime_activated"] += 1
        rows.append(
            {
                "logical_run_id": outcome["logical_run_id"],
                "task_id": outcome["task_id"],
                "system_id": outcome["system_id"],
                **counters,
                "suppression_rate": (
                    counters["suppressed"] / counters["raw_candidates"]
                    if counters["raw_candidates"] else None
                ),
                "static_adoption_rate": (
                    counters["static_adopted"] / counters["prompt_visible"]
                    if counters["prompt_visible"] else None
                ),
                "runtime_activation_rate": (
                    counters["runtime_activated"] / counters["prompt_visible"]
                    if counters["prompt_visible"] else None
                ),
                "by_stage": by_stage,
            }
        )
    return {
        "schema": "mlevolve_end2end_mechanism_summary_v1",
        "analysis_order": 2,
        "exploratory_pilot": True,
        "definitions": {
            "routing": "serialized frozen system route over the common authorized pool",
            "suppression": "raw authorized candidate not visible in the final Prompt",
            "static_adoption": "at least two non-generic memory tokens occur in generated code",
            "runtime_activation": "static adoption on a node with signed Host full-runtime status=pass",
            "causal_attribution": False,
        },
        "runs": rows,
        "summary_hash": "",
    }


def write_json(path: Path, payload: dict[str, Any], field: str) -> None:
    payload[field] = payload_hash(payload, field)
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_terminal_csv(path: Path, summary: Mapping[str, Any]) -> None:
    fields = [
        "system_id", "task_id", "completed", "terminal_score", "status",
        "normalized_delta_vs_no_memory", "negative_transfer",
        "time_to_first_valid_seconds", "allocated_gpu_hours", "llm_token_usage",
        "llm_cost_usd", "task_rank",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for system in summary["systems"]:
            for cell in system["cells"]:
                writer.writerow({"system_id": system["system_id"], **cell})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFESTS / "pilot_manifest.json")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    manifest = read_object(args.manifest)
    verify(manifest, "manifest_hash", "execution manifest")
    expected = {str(row["logical_run_id"]) for row in manifest["runs"]}
    task_ids = list(manifest["task_ids"])
    system_ids = list(manifest["system_ids"])
    measurements = load_measurements(
        args.output_root, formal_only=not args.include_smoke
    )
    outcomes, attempts = select_logical_outcomes(measurements, expected)
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    terminal = terminal_summary(outcomes, task_ids, system_ids)
    write_json(args.analysis_root / "terminal_summary.json", terminal, "summary_hash")
    write_terminal_csv(args.analysis_root / "terminal_cells.csv", terminal)
    attempt_inventory = {
        "schema": "mlevolve_end2end_attempt_inventory_v1",
        "expected_logical_runs": len(expected),
        "observed_logical_runs": len(outcomes),
        "all_attempts_retained": True,
        "attempts": [
            {key: value for key, value in row.items() if key != "_path"}
            for row in attempts
        ],
        "inventory_hash": "",
    }
    write_json(
        args.analysis_root / "attempt_inventory.json",
        attempt_inventory,
        "inventory_hash",
    )
    mechanism = mechanism_summary(outcomes)
    write_json(args.analysis_root / "mechanism_summary.json", mechanism, "summary_hash")
    print(
        json.dumps(
            {
                "expected": len(expected),
                "observed": len(outcomes),
                "completed": sum(row.get("completed") is True for row in outcomes),
                "exploratory_pilot": True,
                "analysis_order": ["terminal", "mechanism"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
