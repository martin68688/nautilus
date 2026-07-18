#!/usr/bin/env python3
"""Score externally evaluated T4 receipts with task-cluster uncertainty."""

from __future__ import annotations

import argparse
import json
from statistics import NormalDist
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from core import REPORTS, holm_adjust, task_cluster_bootstrap, task_cluster_signflip_p, read_jsonl, write_json


COMPARISONS = {"F11-B0": ("F11", "B0"), "F11-F10": ("F11", "F10"), "F11-F01": ("F11", "F01")}


def _normalize(rows: list[dict[str, Any]], reference: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        frozen = reference.get("tasks", {}).get(str(row.get("task_id")), {})
        metric = row.get("metric")
        if not frozen or not isinstance(metric, (int, float)):
            continue
        median = float(frozen["median"])
        scale = float(frozen["iqr"])
        if scale <= 1e-12:
            raise ValueError(f"frozen baseline IQR is non-positive for {row['task_id']}")
        direction = str(row.get("direction") or "minimize")
        normalized = (median - float(metric)) / scale if direction == "minimize" else (float(metric) - median) / scale
        result.append({**row, "task_normalized_final_score": normalized})
    return result


def score(path: Path, baseline_reference_path: Path, claim_gate_path: Path) -> dict[str, Any]:
    raw = read_jsonl(path)
    reference = json.loads(baseline_reference_path.read_text(encoding="utf-8"))
    mechanism_gate = json.loads(claim_gate_path.read_text(encoding="utf-8"))
    if reference.get("split") != "dev" or reference.get("frozen") is not True:
        raise ValueError("baseline reference must be frozen from the dev split")
    trusted = [row for row in raw if row.get("trusted") is True and row.get("rank_eligible") is True]
    analysis_rows = [row for row in raw if row.get("analysis_eligible") is True]
    normalized = _normalize(analysis_rows, reference)
    comparisons = {}
    raw_p_values = {}
    for name, (left, right) in COMPARISONS.items():
        left_rows = [row for row in normalized if row["condition"] == left]
        right_rows = [row for row in normalized if row["condition"] == right]
        comparisons[name] = task_cluster_bootstrap(
            left_rows, right_rows,
            metric="task_normalized_final_score",
        )
        raw_p = task_cluster_signflip_p(left_rows, right_rows, metric="task_normalized_final_score")
        comparisons[name]["raw_signflip_p"] = raw_p
        if raw_p is not None:
            raw_p_values[name] = raw_p
    adjusted = holm_adjust(raw_p_values)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value
    task_count = len({row["task_id"] for row in normalized})
    seeds_per_arm = defaultdict(set)
    for row in normalized:
        seeds_per_arm[(row["task_id"], row["condition"])].add(row["seed"])
    minimum_seeds = min((len(values) for values in seeds_per_arm.values()), default=0)
    planned_task_count = len(reference.get("tasks", {}))
    standardized_mde = (
        (NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)) / np.sqrt(planned_task_count)
        if planned_task_count else None
    )
    success = defaultdict(list)
    for row in raw:
        success[str(row.get("condition"))].append(float(row.get("trusted") is True))
    trusted_success_rates = {condition: float(np.mean(values)) for condition, values in sorted(success.items())}
    noninferiority = (
        trusted_success_rates.get("F11", 0.0) >= trusted_success_rates.get("B0", 0.0) - 0.05
        if "F11" in trusted_success_rates and "B0" in trusted_success_rates else False
    )
    primary = comparisons["F11-B0"]
    lower = primary["cluster_bootstrap_ci95"][0]
    report = {
        "schema": "runforest_composite_downstream_report_v1",
        "raw_receipt_count": len(raw),
        "trusted_receipt_count": len(trusted),
        "analysis_receipt_count": len(analysis_rows),
        "failure_penalty_count": sum(row.get("failure_assigned_worst_valid_metric") is True for row in analysis_rows),
        "task_count": task_count,
        "minimum_seeds_per_observed_arm": minimum_seeds,
        "comparisons": comparisons,
        "standardized_mde": float(standardized_mde) if standardized_mde is not None else None,
        "trusted_success_rates": trusted_success_rates,
        "trusted_success_noninferiority_margin": 0.05,
        "trusted_success_noninferiority_pass": noninferiority,
        "downstream_claim_allowed": bool(task_count >= 10 and minimum_seeds >= 3 and lower is not None and lower > 0),
        "claim_blockers": [],
    }
    if task_count < 10:
        report["claim_blockers"].append("fewer_than_10_tasks")
    if minimum_seeds < 3:
        report["claim_blockers"].append("fewer_than_3_seeds_per_arm")
    if lower is None or lower <= 0:
        report["claim_blockers"].append("F11_vs_B0_cluster_CI_lower_not_positive")
    if standardized_mde is None or standardized_mde > 0.8:
        report["claim_blockers"].append("standardized_MDE_above_0_8_or_unavailable")
    if not noninferiority:
        report["claim_blockers"].append("trusted_success_noninferiority_failed")
    holm_pass = comparisons.get("F11-B0", {}).get("holm_adjusted_p", 1.0) < 0.05
    if not holm_pass:
        report["claim_blockers"].append("F11_vs_B0_Holm_adjusted_p_not_below_0_05")
    if mechanism_gate.get("mechanism_claim_allowed") is not True:
        report["claim_blockers"].append("mechanism_gate_not_passed")
    if not raw or any(row.get("holdout_isolation_mode") != "container_filesystem" for row in raw):
        report["claim_blockers"].append("hidden_holdout_not_filesystem_isolated")
    report["downstream_claim_allowed"] = not report["claim_blockers"]
    write_json(REPORTS / "downstream_report_v1.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    parser.add_argument("--claim-gate", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(score(args.receipts, args.baseline_reference, args.claim_gate), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
