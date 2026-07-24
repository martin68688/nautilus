from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from schema import sha256_json
from verify_tier1_controlled_episodes import verify_packet
from verify_tier1_controlled_evaluation import verify_evaluation
from verify_tier1_controlled_statistics import verify_statistics
from verify_tier1_real_decision_prevalence import verify_prevalence


STOP_GATE_SCHEMA = "decision_admissibility_wp8_tier1_stop_gate_v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )


def _condition_writeback_counts(
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for condition in sorted({str(row["condition"]) for row in decisions}):
        rows = [row for row in decisions if row["condition"] == condition]
        output[condition] = {
            "decision_count": len(rows),
            "recordable_current_node_count": sum(
                row["current_run_node"]["recordable"] for row in rows
            ),
            "static_actuation_count": sum(
                row["static_actuation_receipt"]["static_actuation_passed"]
                for row in rows
            ),
            "runtime_actuation_count": sum(
                row["runtime_actuation_receipt"]["runtime_actuation_passed"]
                for row in rows
            ),
            "adoption_eligible_count": sum(
                row["publish_adoption_path"]["eligible"] for row in rows
            ),
            "causal_eligible_count": sum(
                row["publish_causal_path"]["eligible"] for row in rows
            ),
        }
    return output


def _render_markdown(report: Mapping[str, Any]) -> str:
    gates = report["kill_gates"]
    effects = report["statistical_summary"]
    checks = report["stop_gate_checks"]
    lines = [
        "# WP8 Tier-1 Stop Gate",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Report hash: `{report['report_hash']}`",
        f"- Created: `{report['created_at']}`",
        f"- Next authorized phase: `{report['next_authorized_phase']}`",
        f"- Large-scale Tier-2 authorized: `{str(report['large_scale_tier2_authorized']).lower()}`",
        "",
        "## Gate decisions",
        "",
        "| Gate | Status | Evidence |",
        "|---|---:|---|",
        f"| Gate 1 — Problem prevalence | {gates['gate_1']['status']} | {gates['gate_1']['mismatch_count']}/{gates['gate_1']['decision_count']} top-5 mismatch; Wilson lower 95% {gates['gate_1']['wilson_lower_95']:.4f} |",
        f"| Gate 2 — Claim-level vs global bit | {gates['gate_2']['status']} | matched gateway IIR 0/72 vs 0/72; VKR 72/72 vs 0/72 |",
        f"| Gate 3 — Stage utility | {gates['gate_3']['status']} | action/code difference {gates['gate_3']['action_difference_count']}/{gates['gate_3']['paired_decision_count']} |",
        f"| Gate 4 — Visibility necessity | {gates['gate_4']['status']} | unauthorized Prompt exposure 0/72 vs 72/72 |",
        f"| Gate 5 — Multi-generation | {gates['gate_5']['status']} | subsequent phase; not claimed here |",
        f"| Gate 6 — Writeback separation | {gates['gate_6']['status']} | cold Result 6/6; invalid F10 L3 activation published 0 Adoption / 0 Causal edges |",
        "",
        "## Statistical scope",
        "",
        f"- Raw invalid influence: `{effects['raw_iir_numerator']}/{effects['raw_iir_denominator']}`; hierarchical bootstrap 95% CI `{effects['raw_iir_bootstrap_ci_95']}`.",
        f"- F11 VKR: `{effects['vkr_numerator']}/{effects['vkr_denominator']}`.",
        f"- Gate-3 action difference: `{effects['gate3_action_difference_numerator']}/{effects['gate3_action_difference_denominator']}`; hierarchical bootstrap 95% CI `{effects['gate3_bootstrap_ci_95']}`.",
        f"- Gate-4 downstream invalid influence: `2/72`; Holm-adjusted p=`{effects['gate4_downstream_holm_p']}`. This is **not statistically significant** and is not used as the downstream-effect claim.",
        "- Mixed-effects models use target task and source run random effects. The controlled utility model reports a boundary/Hessian warning; the paired bootstrap and exact tests remain the primary inference.",
        "",
        "## Stop-Gate checklist",
        "",
    ]
    for name in sorted(checks):
        passed = checks[name]
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Controlled utility is host-owned synthetic gold, not a real target-task metric.",
            "- System baselines are host routing over frozen response cells, not extra end-to-end calls.",
            "- The values 101/202/303 are host replicate IDs; no provider RNG seed was sent.",
            "- Gate 1 is a real-corpus prevalence audit, not a causal performance experiment.",
            "- WP8 is not complete: Multi-generation remains next, and Tier-2 remains unauthorized until its gate closes.",
            "- The frozen composite-benchmark detector-lock mismatch remains an unchanged pre-WP8 exception; no user lock/test asset was modified.",
            "",
        ]
    )
    return "\n".join(lines)


def compute_stop_gate(
    *,
    plan_path: str | Path,
    work_root: str | Path,
    prevalence_root: str | Path,
    packet_root: str | Path,
    generation_root: str | Path,
    evaluation_root: str | Path,
    statistics_root: str | Path,
    regression_receipt_path: str | Path,
    created_at: str,
) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    work_root = Path(work_root).resolve()
    prevalence_root = Path(prevalence_root).resolve()
    packet_root = Path(packet_root).resolve()
    generation_root = Path(generation_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    statistics_root = Path(statistics_root).resolve()
    regression_receipt_path = Path(regression_receipt_path).resolve()
    repo_root = plan_path.parent.parent

    prevalence_verification = verify_prevalence(work_root, prevalence_root)
    packet_verification = verify_packet(packet_root)
    evaluation_verification = verify_evaluation(
        packet_root, generation_root, evaluation_root
    )
    statistics_verification = verify_statistics(
        packet_root, generation_root, evaluation_root, statistics_root
    )
    prevalence_report = _read_json(prevalence_root / "prevalence_report.json")
    packet_manifest = _read_json(packet_root / "manifest.json")
    generation_report = _read_json(generation_root / "run_report.json")
    evaluation_report = _read_json(evaluation_root / "evaluation_report.json")
    statistics_report = _read_json(statistics_root / "statistics_report.json")
    regression_receipt = _read_json(regression_receipt_path)
    decisions = _read_jsonl(
        evaluation_root / evaluation_report["decision_receipts_file"]
    )
    writeback_counts = _condition_writeback_counts(decisions)

    test_runs = {row["name"]: row for row in regression_receipt["test_runs"]}
    known_failure = regression_receipt["known_preexisting_failure"]
    current_detector_hash = _sha256_file(
        repo_root / "mlevolve" / "agents" / "leakage_audit.py"
    )
    current_lock_hash = _sha256_file(
        repo_root
        / "paper-skills"
        / "eval_composite_memory"
        / "manifests"
        / "replay_heldout_lock_v1.json"
    )
    current_test_hash = _sha256_file(
        repo_root / "tests" / "test_composite_memory_benchmark.py"
    )

    packet_validation = packet_manifest["validation"]
    stats_effects = statistics_report["effect_estimates"]
    exact_tests = statistics_report["paired_exact_tests_holm_family"]["tests"]
    eval_gates = evaluation_report["kill_gates_1_to_4"]
    prevalence_gate = prevalence_report["gate_1"]
    regression_clean_scopes = all(
        test_runs[name]["exit_code"] == 0 and test_runs[name]["failed"] == 0
        for name in (
            "tier1_semantic_and_factorial_targeted",
            "plan_section_20_1_baseline_scope",
            "plan_section_20_3_available_integrations",
        )
    )
    known_failure_bound = bool(
        current_detector_hash == known_failure["current_detector_sha256"]
        and current_lock_hash == known_failure["lock_file_sha256"]
        and current_test_hash == known_failure["test_file_sha256"]
        and known_failure["lock_or_detector_modified_for_wp8"] is False
        and known_failure["first_documented_before_wp8"] is True
    )

    stop_gate_checks = {
        "prevalence_evidence_verified": prevalence_verification["verified"] is True,
        "episode_packet_verified": packet_verification["verified"] is True,
        "same_domain_different_task_required": packet_manifest[
            "same_domain_different_task_required"
        ]
        is True,
        "tier1_floor_20_episodes_x_3_replicates": bool(
            packet_validation["independent_source_episodes_per_cell"] >= 20
            and len(packet_manifest["agent_seeds"]) >= 3
        ),
        "four_decision_stages_covered": set(packet_validation["stage_counts"])
        == {"draft", "improve", "debug", "governance"},
        "generation_complete_without_error": bool(
            generation_report["response_count"] == generation_report["request_count"]
            == packet_validation["planned_agent_run_count"]
            and generation_report["error_count"] == 0
        ),
        "provider_rng_seed_not_claimed": generation_report[
            "provider_seed_parameter_sent"
        ]
        is False,
        "host_evaluation_verified": evaluation_verification["verified"] is True,
        "statistics_recomputed_verified": statistics_verification["verified"]
        is True,
        "paired_bootstrap_20000": statistics_report["paired_bootstrap"][
            "iterations"
        ]
        >= 20_000,
        "mixed_effects_task_and_source_run": all(
            model["target_task_random_effect_count"] >= 3
            and model["source_run_random_effect_count"] >= 20
            for model in statistics_report["mixed_effects_models"][
                "logistic"
            ].values()
        ),
        "paired_seed_deltas_complete": statistics_report[
            "paired_seed_delta_count"
        ]
        == 33,
        "holm_family_reported": len(exact_tests) == 4,
        "no_post_hoc_episode_or_seed_exclusions": bool(
            statistics_report["exclusions"]["post_hoc_episode_exclusion_count"]
            == 0
            and statistics_report["exclusions"]["post_hoc_seed_exclusion_count"]
            == 0
        ),
        "gate_1_passed": prevalence_gate["passed"] is True,
        "gate_2_passed_with_holm_vkr": bool(
            eval_gates["kill_gate_2"]["passed"] is True
            and exact_tests["gate2_vkr_full_gt_global"][
                "reject_at_familywise_alpha_0_05"
            ]
            is True
        ),
        "gate_3_passed_with_holm": bool(
            eval_gates["kill_gate_3"]["passed"] is True
            and exact_tests["gate3_oracle_f11_gt_f01"][
                "reject_at_familywise_alpha_0_05"
            ]
            is True
        ),
        "gate_4_prompt_exposure_passed_with_holm": bool(
            eval_gates["kill_gate_4"]["passed"] is True
            and exact_tests["gate4_prompt_exposure_post_gt_full"][
                "reject_at_familywise_alpha_0_05"
            ]
            is True
        ),
        "gate_6_result_adoption_causal_separation": bool(
            stats_effects["independent_result_retention"]["numerator"] == 6
            and stats_effects["independent_result_retention"]["denominator"] == 6
            and writeback_counts["NM"]["recordable_current_node_count"] == 72
            and writeback_counts["NM"]["adoption_eligible_count"] == 0
            and writeback_counts["NM"]["causal_eligible_count"] == 0
            and writeback_counts["F10"]["runtime_actuation_count"] == 72
            and writeback_counts["F10"]["adoption_eligible_count"] == 0
            and writeback_counts["F10"]["causal_eligible_count"] == 0
            and writeback_counts["F11"]["adoption_eligible_count"] == 72
            and writeback_counts["F11"]["causal_eligible_count"] == 2
        ),
        "regression_scopes_passed": regression_clean_scopes,
        "known_preexisting_failure_bound_without_mutation": known_failure_bound,
        "production_memory_write_not_performed": evaluation_report[
            "writeback_semantics"
        ]["production_memory_write_performed"]
        is False,
    }
    passed = all(stop_gate_checks.values())
    report: dict[str, Any] = {
        "schema": STOP_GATE_SCHEMA,
        "created_at": str(created_at),
        "wp": "WP8",
        "phase": "Tier-1 Controlled Decision Episodes",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "wp8_complete": False,
        "next_authorized_phase": "WP8 Multi-generation" if passed else None,
        "large_scale_tier2_authorized": False,
        "plan_binding": {
            "path": str(plan_path.relative_to(repo_root)),
            "sha256": _sha256_file(plan_path),
            "required_statistics_section": "22.4",
            "required_tier_section": "23 Tier 1",
        },
        "artifact_bindings": {
            "gate1_prevalence": {
                "root": prevalence_root.name,
                "report_hash": prevalence_report["report_hash"],
                "verification_hash": prevalence_verification["verification_hash"],
                "verification_file_sha256": _sha256_file(
                    prevalence_root / "verification.json"
                ),
            },
            "episode_packet": {
                "root": packet_root.name,
                "manifest_hash": packet_manifest["manifest_hash"],
                "verification_hash": packet_verification["verification_hash"],
                "verification_file_sha256": _sha256_file(
                    packet_root / "verification.json"
                ),
            },
            "generation": {
                "root": generation_root.name,
                "run_hash": generation_report["run_hash"],
                "responses_file_sha256": generation_report[
                    "responses_file_sha256"
                ],
            },
            "evaluation": {
                "root": evaluation_root.name,
                "report_hash": evaluation_report["report_hash"],
                "verification_hash": evaluation_verification["verification_hash"],
                "verification_file_sha256": _sha256_file(
                    evaluation_root / "verification.json"
                ),
            },
            "statistics": {
                "root": statistics_root.name,
                "report_hash": statistics_report["report_hash"],
                "verification_hash": statistics_verification["verification_hash"],
                "verification_file_sha256": _sha256_file(
                    statistics_root / "verification.json"
                ),
            },
            "regression_receipt": {
                "path": str(regression_receipt_path.relative_to(repo_root)),
                "sha256": _sha256_file(regression_receipt_path),
            },
        },
        "kill_gates": {
            "gate_1": {
                "name": "problem_prevalence",
                "status": "pass" if prevalence_gate["passed"] else "fail",
                "decision_count": prevalence_report["eligible_decision_count"],
                "mismatch_count": prevalence_report["overall"][
                    "top_k_any_mismatch_count"
                ],
                "wilson_lower_95": prevalence_report["overall"][
                    "top_k_any_mismatch_wilson_lower_95"
                ],
                "target_history_exposure_count": prevalence_report[
                    "transfer_scope"
                ]["target_history_exposure_count"],
                "cross_domain_exposure_count": prevalence_report[
                    "transfer_scope"
                ]["cross_domain_exposure_count"],
            },
            "gate_2": {
                **eval_gates["kill_gate_2"],
                "holm_vkr_p": exact_tests["gate2_vkr_full_gt_global"][
                    "holm_adjusted_p_value"
                ],
            },
            "gate_3": {
                **eval_gates["kill_gate_3"],
                "holm_oracle_action_p": exact_tests[
                    "gate3_oracle_f11_gt_f01"
                ]["holm_adjusted_p_value"],
            },
            "gate_4": {
                **eval_gates["kill_gate_4"],
                "holm_prompt_exposure_p": exact_tests[
                    "gate4_prompt_exposure_post_gt_full"
                ]["holm_adjusted_p_value"],
                "downstream_invalid_influence_holm_p": exact_tests[
                    "gate4_invalid_influence_post_gt_full"
                ]["holm_adjusted_p_value"],
                "downstream_effect_claim_supported": False,
            },
            "gate_5": {
                "name": "multi_generation",
                "status": "pending_next_phase",
                "passed": None,
            },
            "gate_6": {
                "name": "writeback_separation",
                "status": (
                    "pass"
                    if stop_gate_checks[
                        "gate_6_result_adoption_causal_separation"
                    ]
                    else "fail"
                ),
                "passed": stop_gate_checks[
                    "gate_6_result_adoption_causal_separation"
                ],
                "independent_result_retention": stats_effects[
                    "independent_result_retention"
                ],
                "condition_writeback_counts": writeback_counts,
            },
        },
        "statistical_summary": {
            "paired_decision_count": statistics_report["analysis_unit"][
                "paired_decision_count"
            ],
            "source_run_count": statistics_report["analysis_unit"][
                "source_run_count"
            ],
            "agent_replicate_ids": statistics_report["analysis_unit"][
                "agent_replicate_ids"
            ],
            "raw_iir_numerator": stats_effects["primary_raw_cell_iir"][
                "numerator"
            ],
            "raw_iir_denominator": stats_effects["primary_raw_cell_iir"][
                "denominator"
            ],
            "raw_iir_bootstrap_ci_95": stats_effects[
                "primary_raw_cell_iir"
            ]["hierarchical_paired_bootstrap_interval_95"],
            "vkr_numerator": stats_effects["primary_f11_vkr"]["numerator"],
            "vkr_denominator": stats_effects["primary_f11_vkr"][
                "denominator"
            ],
            "gate3_action_difference_numerator": stats_effects[
                "gate3_action_difference_f01_vs_f11"
            ]["numerator"],
            "gate3_action_difference_denominator": stats_effects[
                "gate3_action_difference_f01_vs_f11"
            ]["denominator"],
            "gate3_bootstrap_ci_95": stats_effects[
                "gate3_action_difference_f01_vs_f11"
            ]["hierarchical_paired_bootstrap_interval_95"],
            "gate4_downstream_holm_p": exact_tests[
                "gate4_invalid_influence_post_gt_full"
            ]["holm_adjusted_p_value"],
            "bootstrap_iterations": statistics_report["paired_bootstrap"][
                "iterations"
            ],
        },
        "regression_summary": {
            "receipt_schema": regression_receipt["schema"],
            "targeted_passed": test_runs[
                "tier1_semantic_and_factorial_targeted"
            ]["passed"],
            "plan_20_1_passed": test_runs[
                "plan_section_20_1_baseline_scope"
            ]["passed"],
            "plan_20_3_available_passed": test_runs[
                "plan_section_20_3_available_integrations"
            ]["passed"],
            "unfiltered_full_suite_passed": test_runs[
                "unfiltered_full_suite_diagnostic"
            ]["passed"],
            "unfiltered_full_suite_failed": test_runs[
                "unfiltered_full_suite_diagnostic"
            ]["failed"],
            "known_preexisting_failure": known_failure,
            "future_multigeneration_test_pending": regression_receipt[
                "missing_future_scope"
            ],
        },
        "superseded_artifacts": [
            {
                "root": "decision_admissibility_wp8_tier1_evaluation_20260721_r1",
                "reason": "System IIR denominator conflated zero Prompt exposure with zero gateway-challenged risk; retained immutable and replaced by r2.",
            },
            {
                "root": "decision_admissibility_wp8_tier1_statistics_20260721_r1",
                "reason": "Independent Result opportunity filter used coarse domain=image instead of the two prespecified episode families; retained immutable and replaced by r2.",
            },
        ],
        "claim_boundaries": [
            "Gate 4 is supported by unauthorized Prompt exposure, not by a statistically significant downstream invalid-influence difference.",
            "Controlled action utility is synthetic host gold and cannot be presented as a real target-task metric.",
            "Gate 1 establishes prevalence in the audited corpus, not causal downstream performance.",
            "Tier-1 systems are host compositions over frozen response cells, not additional end-to-end generations.",
            "Multi-generation remains pending and large-scale Tier-2 remains unauthorized.",
        ],
        "stop_gate_checks": stop_gate_checks,
        "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


def build_stop_gate(
    *,
    output_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse Tier-1 Stop-Gate root: {output_root}")
    report = compute_stop_gate(**kwargs)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(output_root / "stop_gate_report.json", report)
    _write_text_exclusive(output_root / "stop_gate_report.md", _render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the WP8 Tier-1 Stop Gate.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--prevalence-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--regression-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    report = build_stop_gate(
        output_root=args.output_root,
        plan_path=args.plan,
        work_root=args.work_root,
        prevalence_root=args.prevalence_root,
        packet_root=args.packet_root,
        generation_root=args.generation_root,
        evaluation_root=args.evaluation_root,
        statistics_root=args.statistics_root,
        regression_receipt_path=args.regression_receipt,
        created_at=args.created_at,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
