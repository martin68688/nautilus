#!/usr/bin/env python3
"""Validate Phase 0 and generate the single claim-gate report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from core import ANNOTATIONS, EPISODES, MANIFESTS, REPORTS, read_json, read_jsonl, write_json


def _ordinal_alpha(rows: list[dict[str, Any]]) -> float | None:
    by_item: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("episode_id")), str(row.get("candidate_id")))
        if row.get("relevance") is not None:
            by_item[key].append(int(row["relevance"]))
    units = [values for values in by_item.values() if len(values) >= 2]
    if not units:
        return None
    categories = sorted({value for unit in units for value in unit})
    index = {value: position for position, value in enumerate(categories)}
    coincidence = np.zeros((len(categories), len(categories)), dtype=float)
    for unit in units:
        counts = np.asarray([unit.count(category) for category in categories], dtype=float)
        m = counts.sum()
        for left in range(len(categories)):
            for right in range(len(categories)):
                numerator = counts[left] * (counts[right] - (1.0 if left == right else 0.0))
                coincidence[left, right] += numerator / (m - 1.0)
    marginals = coincidence.sum(axis=1)
    n = marginals.sum()
    expected = np.zeros_like(coincidence)
    if n <= 1:
        return None
    for left in range(len(categories)):
        for right in range(len(categories)):
            expected[left, right] = marginals[left] * (marginals[right] - (1.0 if left == right else 0.0)) / (n - 1.0)
    distance = np.zeros_like(coincidence)
    for left in range(len(categories)):
        for right in range(len(categories)):
            lo, hi = sorted((left, right))
            distance[left, right] = (
                marginals[lo : hi + 1].sum() - (marginals[lo] + marginals[hi]) / 2.0
            ) ** 2
    observed_disagreement = float((coincidence * distance).sum())
    expected_disagreement = float((expected * distance).sum())
    return 1.0 - observed_disagreement / expected_disagreement if expected_disagreement > 0 else 1.0


def _top3_difference(receipts: list[dict[str, Any]]) -> float | None:
    by_episode: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in receipts:
        if row["condition"] == "O1" or not row.get("score_applicable"):
            continue
        by_episode[row["episode_id"]].add(tuple(row["ranking"][:3]))
    return sum(len(rankings) > 1 for rankings in by_episode.values()) / len(by_episode) if by_episode else None


def score(split: str = "dev", *, persist: bool = True) -> dict[str, Any]:
    build = read_json(REPORTS / "build_report_v1.json")
    memory = read_json(MANIFESTS / "memory_snapshot_manifest_v1.json")
    offline = read_json(REPORTS / f"offline_{split}_report_v1.json")
    receipts = read_jsonl(REPORTS / f"offline_{split}_receipts_v1.jsonl")
    replay = read_json(REPORTS / "replay_static_report_v1.json") if (REPORTS / "replay_static_report_v1.json").exists() else {}
    replay_heldout = read_json(REPORTS / "replay_heldout_report_v1.json") if (REPORTS / "replay_heldout_report_v1.json").exists() else {}
    adoption = read_json(REPORTS / "adoption_report_v1.json") if (REPORTS / "adoption_report_v1.json").exists() else {}
    annotations_path = ANNOTATIONS / "adjudicated_gold_v1.jsonl"
    annotations = read_jsonl(annotations_path) if annotations_path.exists() else []
    alpha = _ordinal_alpha(annotations)
    split_sets = {name: set(values) for name, values in memory["run_split"].items()}
    overlap = sum(len(split_sets[a] & split_sets[b]) for a, b in (("memory_train", "benchmark_dev"), ("memory_train", "benchmark_test"), ("benchmark_dev", "benchmark_test")))
    scored = {
        key: value for key, value in offline["conditions"].items()
        if value.get("graded_ndcg_at_10") is not None
    }
    oracle = scored.get("O1", {}).get("graded_ndcg_at_10")
    random = scored.get("B1", {}).get("graded_ndcg_at_10")
    oracle_gap = oracle - random if oracle is not None and random is not None else None
    method_difference = _top3_difference(receipts)
    normal_queries = [row for row in read_jsonl(EPISODES / f"decision_{split}_v1.jsonl") if row["expected_status"] != "insufficient_strategy_coverage"]
    gaps = [row for row in read_jsonl(EPISODES / f"decision_{split}_v1.jsonl") if row["expected_status"] == "insufficient_strategy_coverage"]
    forbidden_run_ids = split_sets["benchmark_dev"] | split_sets["benchmark_test"]
    query_payloads = [json.dumps(row, sort_keys=True) for row in read_jsonl(EPISODES / f"decision_{split}_v1.jsonl")]
    checks = {
        "source_run_overlap_zero": overlap == 0,
        "source_run_ids_exposed_zero": (
            all(not row["provenance"]["source_run_ids_exposed"] for row in receipts)
            and all(not any(run_id and run_id in payload for run_id in forbidden_run_ids) for payload in query_payloads)
        ),
        "normal_episode_gold_at_least_3": all(
            len(row["labels"]) >= 3
            for row in read_jsonl(EPISODES / f"decision_{split}_silver_gold_v1.jsonl")
            if row["expected_status"] != "insufficient_strategy_coverage"
        ),
        "coverage_gap_count_zero": len(gaps) == 0,
        "oracle_ndcg_at_least_0_90": oracle is not None and oracle >= 0.90,
        "random_ndcg_at_most_0_50": random is not None and random <= 0.50,
        "oracle_random_gap_at_least_0_30": oracle_gap is not None and oracle_gap >= 0.30,
        "method_top3_difference_at_least_0_30": method_difference is not None and method_difference >= 0.30,
        "unsafe_primary_escape_zero": all(
            row["unsafe_count_at_10"] == 0 for row in receipts if row["condition"] in {"F00", "F01", "F10", "F11"}
        ),
        "two_blind_annotators": bool(annotations) and all(int(row.get("annotator_count", 0)) >= 2 for row in annotations),
        "krippendorff_alpha_at_least_0_67": alpha is not None and alpha >= 0.67,
        "minimum_test_episodes": build["test_episode_count"] >= 120,
        "replay_defect_count_at_least_48": replay.get("case_count", 0) >= 48,
        "replay_all_sources_blocked_before_execution": replay.get("blocked_before_execution_rate") == 1.0,
        "replay_heldout_expected_issue_recall_one": replay_heldout.get("expected_issue_recall") == 1.0,
        "replay_heldout_all_sources_blocked": replay_heldout.get("blocked_before_execution_rate") == 1.0,
    }
    phase0_pass = all(checks.values())
    blockers = [name for name, passed in checks.items() if not passed]
    role_ready = build["role_decomposition_claim_ready"]
    adoption_ready = adoption.get("adoption_claim_allowed") is True
    report = {
        "schema": "runforest_composite_claim_gate_report_v1",
        "split": split,
        "phase0_checks": checks,
        "phase0_pass": phase0_pass,
        "normal_episode_count": len(normal_queries),
        "coverage_gap_episode_count": len(gaps),
        "oracle_ndcg_at_10": oracle,
        "random_ndcg_at_10": random,
        "oracle_random_gap": oracle_gap,
        "method_top3_difference_rate": method_difference,
        "krippendorff_alpha_ordinal": alpha,
        "replay_expected_issue_recall": replay.get("expected_issue_recall"),
        "replay_heldout_expected_issue_recall": replay_heldout.get("expected_issue_recall"),
        "replay_heldout_blocked_before_execution_rate": replay_heldout.get("blocked_before_execution_rate"),
        "adoption_non_mock_count": adoption.get("non_mock_candidate_count", 0),
        "mechanism_claim_allowed": phase0_pass and build["test_episode_count"] >= 120,
        "role_decomposition_claim_allowed": phase0_pass and role_ready,
        "adoption_claim_allowed": phase0_pass and adoption_ready,
        "replay_repair_success_claim_allowed": phase0_pass and replay.get("replay_repair_success_claim_allowed") is True,
        "downstream_claim_allowed": False,
        "claim_blockers": blockers + [
            *([] if role_ready else ["replay_eligible_tasks_below_8"]),
            *([] if adoption_ready else ["non_mock_agent_adoption_below_60"]),
            *([] if replay.get("replay_repair_success_claim_allowed") is True else ["five_stage_replay_repairs_not_completed"]),
            "T4_external_holdout_not_completed",
        ],
        "diagnostic_findings": {
            "stage_hybrid_minus_flat_ndcg": (
                scored.get("F11", {}).get("graded_ndcg_at_10", 0.0)
                - scored.get("F10", {}).get("graded_ndcg_at_10", 0.0)
            ),
            "stage_hybrid_minus_sop_only_ndcg": (
                scored.get("F11", {}).get("graded_ndcg_at_10", 0.0)
                - scored.get("D3", {}).get("graded_ndcg_at_10", 0.0)
            ),
            "poincare_minus_flat_twin_ndcg": (
                scored.get("F11", {}).get("graded_ndcg_at_10", 0.0)
                - scored.get("D6", {}).get("graded_ndcg_at_10", 0.0)
            ),
            "unsafe_ablation_escape_count": scored.get("D7", {}).get("unsafe_escape_count"),
            "portfolio_effect_identifiable_in_T1": False,
        },
        "prohibited_claims": [
            "full_MLE-Bench superiority",
            "universal task-family generalization",
            "Poincare superiority over Flat-Twin",
            "long-budget downstream superiority",
            "legal reproduction of historical replay best score",
        ],
    }
    if persist:
        write_json(REPORTS / "claim_gate_report_v1.json", report)
        _write_markdown(report, scored)
    return report


def _write_markdown(report: dict[str, Any], scored: dict[str, Any]) -> None:
    lines = [
        "# RunForest Composite Benchmark v1",
        "",
        f"- Phase 0: **{'PASS' if report['phase0_pass'] else 'FAIL CLOSED'}**",
        f"- Mechanism claim allowed: **{str(report['mechanism_claim_allowed']).lower()}**",
        f"- Downstream claim allowed: **{str(report['downstream_claim_allowed']).lower()}**",
        f"- Normal episodes: {report['normal_episode_count']}",
        f"- Coverage gaps: {report['coverage_gap_episode_count']}",
        f"- Frozen-fixture replay recall: {report['replay_expected_issue_recall']}",
        f"- Independent held-out replay recall: {report['replay_heldout_expected_issue_recall']}",
        f"- Independent held-out pre-execution block rate: {report['replay_heldout_blocked_before_execution_rate']}",
        "- Evidence status: **DIAGNOSTIC ONLY** (silver labels, incomplete Agent/runtime tiers).",
        "",
        "## Offline retrieval",
        "",
        "F/P portfolio rows are retrieval-identical at T1 because no Agent generation occurs; they must not be used to infer portfolio effects.",
        "",
        "| Condition | nDCG@10 | AP@10 | Unsafe escapes |",
        "|---|---:|---:|---:|",
    ]
    for condition, row in sorted(scored.items()):
        lines.append(f"| {condition} | {row['graded_ndcg_at_10']:.4f} | {row['adoption_ap_at_10']:.4f} | {row['unsafe_escape_count']} |")
    lines += ["", "## Closed gates", ""]
    lines.extend(f"- `{value}`" for value in report["claim_blockers"])
    findings = report["diagnostic_findings"]
    lines += [
        "", "## Diagnostic interpretation", "",
        f"- Stage Hybrid - flat clean nDCG: `{findings['stage_hybrid_minus_flat_ndcg']:.4f}`.",
        f"- Stage Hybrid - SOP-only nDCG: `{findings['stage_hybrid_minus_sop_only_ndcg']:.4f}`.",
        f"- Poincare - Flat-Twin nDCG: `{findings['poincare_minus_flat_twin_ndcg']:.4f}`.",
        f"- Disabling safety admitted `{findings['unsafe_ablation_escape_count']}` unsafe candidates.",
        "- T1 cannot identify portfolio effects because no Agent generation occurs at this tier.",
        "", "## Negative results that block positive interpretation", "",
        (
            "- **Stage Hybrid does not beat SOP-only:** the nDCG difference is "
            f"`{findings['stage_hybrid_minus_sop_only_ndcg']:.4f}`."
        ),
        (
        "- **Poincare does not beat Flat-Twin:** the nDCG difference is "
            f"`{findings['poincare_minus_flat_twin_ndcg']:.4f}`."
        ),
        (
            "- **Static replay safety does not generalize on the independently authored challenge:** "
            f"issue recall is `{report['replay_heldout_expected_issue_recall']:.4f}` and pre-execution block rate is "
            f"`{report['replay_heldout_blocked_before_execution_rate']:.4f}`. The frozen-fixture `1.0` is not an "
            "independent recall estimate."
        ),
        "- These are falsifying diagnostics, not evidence for the composite mechanism.",
        "", "All silver-label and incomplete-runtime results are diagnostic only.", "",
    ]
    (REPORTS / "composite_benchmark_v1.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    print(json.dumps(score(args.split), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
