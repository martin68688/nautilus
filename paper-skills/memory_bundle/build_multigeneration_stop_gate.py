from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from schema import sha256_json
from verify_multigeneration_contamination_evaluation import verify_evaluation
from verify_multigeneration_contamination_packet import verify_packet
from verify_multigeneration_paraphrases import verify_run
from verify_multigeneration_statistics import verify_statistics


STOP_GATE_SCHEMA = "decision_admissibility_wp8_multigeneration_stop_gate_v1"
REGRESSION_SCHEMA = (
    "decision_admissibility_wp8_multigeneration_regression_receipt_v1"
)
REQUIRED_CLEAN_TEST_RUNS = (
    "multigeneration_targeted",
    "wp8_experiment_targeted",
    "plan_section_20_1_baseline_scope",
    "plan_section_20_3_integrations",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) == sha256_json(
        {key: value for key, value in payload.items() if key != field}
    )


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
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
    )


def _stable_inline_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _stored_verification_matches(
    root: Path, recomputed: Mapping[str, Any]
) -> bool:
    path = root / "verification.json"
    if not path.is_file():
        return False
    stored = _read_json(path)
    return bool(
        stored == recomputed
        and stored.get("verified") is True
        and stored.get("errors") == []
        and _valid_hash(stored, "verification_hash")
    )


def _prior_tier1_gate_checks(root: Path) -> dict[str, bool]:
    report_path = root / "stop_gate_report.json"
    markdown_path = root / "stop_gate_report.md"
    verification_path = root / "verification.json"
    if not all(path.is_file() for path in (report_path, markdown_path, verification_path)):
        return {
            "prior_tier1_artifacts_present": False,
            "prior_tier1_report_hash_valid": False,
            "prior_tier1_verification_hash_valid": False,
            "prior_tier1_gate_passed": False,
            "prior_tier1_authorized_multigeneration": False,
            "prior_tier1_files_bound": False,
        }
    report = _read_json(report_path)
    verification = _read_json(verification_path)
    return {
        "prior_tier1_artifacts_present": True,
        "prior_tier1_report_hash_valid": _valid_hash(report, "report_hash"),
        "prior_tier1_verification_hash_valid": _valid_hash(
            verification, "verification_hash"
        ),
        "prior_tier1_gate_passed": bool(
            report.get("passed") is True
            and report.get("status") == "pass"
            and verification.get("verified") is True
            and verification.get("errors") == []
        ),
        "prior_tier1_authorized_multigeneration": bool(
            report.get("next_authorized_phase") == "WP8 Multi-generation"
            and report.get("large_scale_tier2_authorized") is False
            and (report.get("kill_gates") or {}).get("gate_5", {}).get("status")
            == "pending_next_phase"
        ),
        "prior_tier1_files_bound": bool(
            verification.get("stop_gate_report_hash") == report.get("report_hash")
            and verification.get("stop_gate_report_file_sha256")
            == _sha256_file(report_path)
            and verification.get("stop_gate_markdown_file_sha256")
            == _sha256_file(markdown_path)
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    effects = report["statistical_summary"]
    gates = report["kill_gates"]
    checks = report["stop_gate_checks"]
    final = effects["final_generation"]
    lines = [
        "# WP8 Multi-generation Gate-5 Stop Gate",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Report hash: `{report['report_hash']}`",
        f"- Created: `{report['created_at']}`",
        f"- Next authorized phase: `{report['next_authorized_phase']}`",
        f"- Tier-2 canary authorized: `{str(report['tier2_canary_authorized']).lower()}`",
        f"- Large-scale Tier-2 authorized: `{str(report['large_scale_tier2_authorized']).lower()}`",
        "",
        "## Kill-gate status",
        "",
        "| Gate | Status | Evidence |",
        "|---|---:|---|",
        f"| Gate 1 — Problem prevalence | {gates['gate_1']['status']} | inherited from verified Tier-1 gate |",
        f"| Gate 2 — Claim-level vs global bit | {gates['gate_2']['status']} | inherited from verified Tier-1 gate |",
        f"| Gate 3 — Stage utility | {gates['gate_3']['status']} | inherited from verified Tier-1 gate |",
        f"| Gate 4 — Visibility necessity | {gates['gate_4']['status']} | inherited from verified Tier-1 gate |",
        f"| Gate 5 — Multi-generation | {gates['gate_5']['status']} | Full laundering {final['full']['laundering_numerator']}/{final['full']['denominator']}; Full VKR {final['full']['vkr_numerator']}/{final['full']['denominator']} |",
        f"| Gate 6 — Writeback separation | {gates['gate_6']['status']} | inherited from verified Tier-1 gate |",
        "",
        "## Experimental scope",
        "",
        f"- Source experiences: `{effects['source_pair_count']}` pairs from `{effects['source_run_count']}` runs and `{effects['source_task_count']}` source tasks.",
        f"- Domains: `{_stable_inline_json(effects['domain_counts'])}`.",
        f"- Frozen descendant DAG: `{effects['generation_count']}` generations × `{effects['paraphrase_replicate_count']}` host replicates = `{effects['request_count']}` real model requests.",
        f"- System projections: `{effects['system_count']}` policies over the same DAG, producing `{effects['system_receipt_count']}` policy receipts.",
        "",
        "## Final-generation outcomes",
        "",
        "| System | Laundering | VKR |",
        "|---|---:|---:|",
    ]
    for system in (
        "full",
        "lineage_only",
        "unrestricted",
        "authority_only",
        "global_validity_bit",
    ):
        row = final[system]
        lines.append(
            f"| {system} | {row['laundering_numerator']}/{row['denominator']} | {row['vkr_numerator']}/{row['denominator']} |"
        )
    lines.extend(
        [
            "",
            "## Paired inference",
            "",
            f"- Full vs unrestricted laundering reduction 95% CI: `{effects['full_vs_unrestricted_laundering_reduction_ci_95']}`.",
            f"- Full vs authority-only laundering reduction 95% CI: `{effects['full_vs_authority_laundering_reduction_ci_95']}`.",
            f"- Full vs Global Bit VKR delta 95% CI: `{effects['full_vs_global_vkr_delta_ci_95']}`.",
            f"- Hierarchical paired bootstrap iterations: `{effects['bootstrap_iterations']}`; resampling unit starts at source run.",
            f"- Holm-adjusted primary p-values: `{_stable_inline_json(effects['holm_adjusted_p_values'])}`.",
            "",
            "## Stop-Gate checklist",
            "",
        ]
    )
    for name in sorted(checks):
        lines.append(f"- [{'x' if checks[name] else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## Claim boundaries",
            "",
            "- This benchmark supports lineage non-escalation across five paraphrase generations.",
            "- Full and lineage-only are tied here; this gate does not claim that Full outperforms lineage-only.",
            "- Systems are host policy projections over one frozen real-model descendant DAG, not five separately generated model runs per system.",
            "- Authority-only is a frozen current-surface ablation, not a universal semantic detector.",
            "- Replicate IDs 101/202/303 are host chain/style identifiers; no provider RNG seed was sent.",
            "- The corpus uses same-domain, different-task transfer with zero target-history and zero cross-domain exposure.",
            "- This is not an online target-task metric experiment. Only the Tier-2 canary is authorized next; large-scale Tier-2 remains unauthorized.",
            "- Evaluation r1 is preserved but superseded. All claims and hashes in this gate bind evaluation r2.",
            "- The frozen composite-benchmark detector-lock mismatch remains an unchanged pre-WP8 exception; no user lock/test asset was modified.",
            "",
        ]
    )
    return "\n".join(lines)


def compute_stop_gate(
    *,
    plan_path: str | Path,
    work_root: str | Path,
    prior_tier1_stop_gate_root: str | Path,
    packet_root: str | Path,
    run_root: str | Path,
    evaluation_root: str | Path,
    superseded_evaluation_root: str | Path,
    statistics_root: str | Path,
    regression_receipt_path: str | Path,
    created_at: str,
) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    work_root = Path(work_root).resolve()
    prior_tier1_stop_gate_root = Path(prior_tier1_stop_gate_root).resolve()
    packet_root = Path(packet_root).resolve()
    run_root = Path(run_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    superseded_evaluation_root = Path(superseded_evaluation_root).resolve()
    statistics_root = Path(statistics_root).resolve()
    regression_receipt_path = Path(regression_receipt_path).resolve()
    repo_root = plan_path.parent.parent

    packet_verification = verify_packet(work_root, packet_root)
    run_verification = verify_run(work_root, packet_root, run_root)
    evaluation_verification = verify_evaluation(
        work_root, packet_root, run_root, evaluation_root
    )
    statistics_verification = verify_statistics(
        work_root, packet_root, run_root, evaluation_root, statistics_root
    )

    prior_report = _read_json(prior_tier1_stop_gate_root / "stop_gate_report.json")
    packet_manifest = _read_json(packet_root / "manifest.json")
    run_report = _read_json(run_root / "run_report.json")
    evaluation_report = _read_json(evaluation_root / "evaluation_report.json")
    superseded_report = _read_json(
        superseded_evaluation_root / "evaluation_report.json"
    )
    statistics_report = _read_json(statistics_root / "statistics_report.json")
    regression_receipt = _read_json(regression_receipt_path)

    prior_checks = _prior_tier1_gate_checks(prior_tier1_stop_gate_root)
    test_runs = {
        str(row["name"]): row for row in regression_receipt.get("test_runs") or []
    }
    clean_regression_scopes = all(
        name in test_runs
        and test_runs[name].get("exit_code") == 0
        and test_runs[name].get("failed") == 0
        for name in REQUIRED_CLEAN_TEST_RUNS
    )
    diagnostic = test_runs.get("unfiltered_full_suite_diagnostic") or {}
    known_failure = regression_receipt.get("known_preexisting_failure") or {}
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
    known_failure_bound = bool(
        diagnostic.get("exit_code") == 1
        and diagnostic.get("failed") == 1
        and known_failure.get("current_detector_sha256") == current_detector_hash
        and known_failure.get("lock_file_sha256") == current_lock_hash
        and known_failure.get("test_file_sha256") == current_test_hash
        and known_failure.get("lock_or_detector_modified_for_wp8") is False
        and known_failure.get("first_documented_before_wp8") is True
    )

    summaries = evaluation_report["system_summaries_by_generation"]
    final_generation = str(packet_manifest["generation_count"])
    final_rows = {
        system: summaries[system][final_generation]
        for system in packet_manifest["systems"]
    }
    effects = statistics_report["final_generation_effects"]
    bootstrap = statistics_report["paired_bootstrap"]
    exact_tests = statistics_report["paired_exact_tests_holm_family"]["tests"]
    models = statistics_report["mixed_effects_logistic_models"]
    prior_gates = prior_report["kill_gates"]
    retry_distribution = run_report.get("retry_count_distribution") or {}

    stop_gate_checks = {
        **prior_checks,
        "packet_recomputed_verified": packet_verification.get("verified") is True,
        "packet_stored_verification_matches": _stored_verification_matches(
            packet_root, packet_verification
        ),
        "same_domain_different_task_only": packet_manifest.get(
            "same_domain_different_task_required"
        )
        is True,
        "five_domain_coverage": set(packet_manifest.get("domain_counts") or {})
        == {"audio", "image", "nlp", "tabular", "temporal"},
        "target_history_overlap_zero": packet_manifest.get(
            "target_history_overlap_count"
        )
        == 0,
        "source_pair_and_run_floors": bool(
            packet_manifest.get("source_pair_count", 0)
            >= packet_manifest["gate_5_thresholds"]["minimum_source_pair_count"]
            and packet_manifest.get("source_run_count", 0)
            >= packet_manifest["gate_5_thresholds"]["minimum_source_run_count"]
        ),
        "five_generations_three_host_replicates": bool(
            packet_manifest.get("generation_count", 0)
            >= packet_manifest["gate_5_thresholds"]["minimum_generation_count"]
            and len(packet_manifest.get("paraphrase_replicate_ids") or [])
            >= packet_manifest["gate_5_thresholds"][
                "minimum_paraphrase_replicates"
            ]
        ),
        "thresholds_frozen_before_generation": packet_manifest.get(
            "thresholds_fixed_before_generation"
        )
        is True,
        "run_recomputed_verified": run_verification.get("verified") is True,
        "run_stored_verification_matches": _stored_verification_matches(
            run_root, run_verification
        ),
        "generation_matrix_complete_without_error_or_retry": bool(
            run_report.get("request_count")
            == run_report.get("response_count")
            == packet_manifest.get("planned_paraphrase_request_count")
            and run_report.get("error_count") == 0
            and retry_distribution == {"0": run_report.get("request_count")}
        ),
        "provider_rng_seed_not_sent": run_report.get(
            "provider_seed_parameter_sent"
        )
        is False,
        "descendant_dag_frozen_before_system_evaluation": bool(
            packet_manifest.get(
                "descendant_generation_is_frozen_before_system_evaluation"
            )
            is True
            and run_report.get("descendant_dag_frozen_before_system_evaluation")
            is True
            and evaluation_report.get(
                "identical_frozen_descendant_dag_for_all_systems"
            )
            is True
        ),
        "evaluation_r2_recomputed_verified": bool(
            evaluation_report.get("schema")
            == "decision_admissibility_multigeneration_evaluation_report_v2"
            and evaluation_verification.get("verified") is True
        ),
        "evaluation_r2_stored_verification_matches": _stored_verification_matches(
            evaluation_root, evaluation_verification
        ),
        "evaluation_receipt_matrix_complete": bool(
            evaluation_report.get("descendant_receipt_count")
            == run_report.get("response_count")
            and evaluation_report.get("system_receipt_count")
            == run_report.get("response_count")
            * len(packet_manifest.get("systems") or [])
        ),
        "no_cross_domain_target_history_or_post_hoc_exclusion": bool(
            evaluation_report.get("cross_domain_transfer_count") == 0
            and evaluation_report.get("target_history_exposure_count") == 0
            and evaluation_report.get("post_hoc_descendant_exclusion_count") == 0
        ),
        "r1_preserved_but_superseded_by_r2": bool(
            superseded_evaluation_root != evaluation_root
            and superseded_report.get("schema")
            == "decision_admissibility_multigeneration_evaluation_report_v1"
            and evaluation_report.get("schema")
            == "decision_admissibility_multigeneration_evaluation_report_v2"
        ),
        "statistics_recomputed_verified": statistics_verification.get("verified")
        is True,
        "statistics_stored_verification_matches": _stored_verification_matches(
            statistics_root, statistics_verification
        ),
        "hierarchical_paired_bootstrap_20000": bool(
            bootstrap.get("iterations", 0) >= 20_000
            and bootstrap.get("systems_and_generations_kept_paired") is True
            and bootstrap.get("resampling_order", [None])[0] == "source_run_id"
        ),
        "mixed_effects_task_and_source_run": all(
            model.get("source_task_random_effect_count", 0) >= 5
            and model.get("source_run_random_effect_count", 0) >= 35
            and model.get("optimizer_success") is True
            for model in models.values()
        ),
        "holm_primary_family_rejects_all_four": bool(
            len(exact_tests) == 4
            and all(
                test.get("reject_at_familywise_alpha_0_05") is True
                for test in exact_tests.values()
            )
        ),
        "gate_5_all_preregistered_checks_pass": bool(
            statistics_report.get("gate_5", {}).get("passed") is True
            and all(
                statistics_report.get("gate_5", {}).get("checks", {}).values()
            )
            and len(statistics_report.get("gate_5", {}).get("checks", {})) == 14
        ),
        "full_zero_laundering_and_full_vkr": bool(
            effects["full_laundering"]["numerator"] == 0
            and effects["full_laundering"]["denominator"] == 180
            and effects["full_vkr"]["numerator"] == 180
            and effects["full_vkr"]["denominator"] == 180
        ),
        "unrestricted_attack_and_global_bit_tradeoff_present": bool(
            effects["unrestricted_laundering"]["numerator"] == 180
            and final_rows["global_validity_bit"]["valid_knowledge_retained_count"]
            == 0
        ),
        "full_lineage_only_tie_bound_without_superiority_claim": bool(
            final_rows["full_decision_admissibility"][
                "laundering_success_count"
            ]
            == final_rows["lineage_only"]["laundering_success_count"]
            == 0
            and final_rows["full_decision_admissibility"][
                "valid_knowledge_retained_count"
            ]
            == final_rows["lineage_only"]["valid_knowledge_retained_count"]
            == 180
            and any(
                "no superiority over lineage-only is claimed" in boundary
                for boundary in statistics_report.get(
                    "interpretation_boundaries", []
                )
            )
        ),
        "regression_receipt_hash_valid": bool(
            regression_receipt.get("schema") == REGRESSION_SCHEMA
            and _valid_hash(regression_receipt, "receipt_hash")
        ),
        "targeted_and_broad_regression_scopes_passed": bool(
            clean_regression_scopes
            and regression_receipt.get("multigeneration_regression_gate_passed")
            is True
        ),
        "known_preexisting_failure_bound_without_mutation": known_failure_bound,
    }
    passed = all(stop_gate_checks.values())

    inherited_gates = {
        name: {
            "name": prior_gates[name]["name"],
            "status": prior_gates[name]["status"],
            "passed": bool(
                prior_gates[name].get("passed") is True
                or prior_gates[name]["status"] == "pass"
            ),
            "evidence": "inherited_from_verified_tier1_stop_gate",
        }
        for name in ("gate_1", "gate_2", "gate_3", "gate_4", "gate_6")
    }
    final_summary = {
        "full": {
            "laundering_numerator": final_rows["full_decision_admissibility"][
                "laundering_success_count"
            ],
            "vkr_numerator": final_rows["full_decision_admissibility"][
                "valid_knowledge_retained_count"
            ],
            "denominator": final_rows["full_decision_admissibility"][
                "decision_count"
            ],
        },
        "lineage_only": {
            "laundering_numerator": final_rows["lineage_only"][
                "laundering_success_count"
            ],
            "vkr_numerator": final_rows["lineage_only"][
                "valid_knowledge_retained_count"
            ],
            "denominator": final_rows["lineage_only"]["decision_count"],
        },
        "unrestricted": {
            "laundering_numerator": final_rows["unrestricted"][
                "laundering_success_count"
            ],
            "vkr_numerator": final_rows["unrestricted"][
                "valid_knowledge_retained_count"
            ],
            "denominator": final_rows["unrestricted"]["decision_count"],
        },
        "authority_only": {
            "laundering_numerator": final_rows["authority_only"][
                "laundering_success_count"
            ],
            "vkr_numerator": final_rows["authority_only"][
                "valid_knowledge_retained_count"
            ],
            "denominator": final_rows["authority_only"]["decision_count"],
        },
        "global_validity_bit": {
            "laundering_numerator": final_rows["global_validity_bit"][
                "laundering_success_count"
            ],
            "vkr_numerator": final_rows["global_validity_bit"][
                "valid_knowledge_retained_count"
            ],
            "denominator": final_rows["global_validity_bit"]["decision_count"],
        },
    }

    report: dict[str, Any] = {
        "schema": STOP_GATE_SCHEMA,
        "created_at": str(created_at),
        "wp": "WP8",
        "phase": "Multi-generation Gate-5",
        "status": "pass" if passed else "fail",
        "passed": passed,
        "wp8_complete": False,
        "next_authorized_phase": "WP8 Tier-2 canary" if passed else None,
        "tier2_canary_authorized": passed,
        "large_scale_tier2_authorized": False,
        "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
        "plan_binding": {
            "path": str(plan_path.relative_to(repo_root)),
            "sha256": _sha256_file(plan_path),
            "kill_gate_section": "22.5 Gate 5",
            "multi_generation_section": "23 Multi-generation",
            "tier2_section": "23 Tier 2",
        },
        "artifact_bindings": {
            "prior_tier1_stop_gate": {
                "root": prior_tier1_stop_gate_root.name,
                "report_hash": prior_report["report_hash"],
                "report_file_sha256": _sha256_file(
                    prior_tier1_stop_gate_root / "stop_gate_report.json"
                ),
                "verification_file_sha256": _sha256_file(
                    prior_tier1_stop_gate_root / "verification.json"
                ),
            },
            "packet": {
                "root": packet_root.name,
                "manifest_hash": packet_manifest["manifest_hash"],
                "verification_hash": packet_verification["verification_hash"],
                "verification_file_sha256": _sha256_file(
                    packet_root / "verification.json"
                ),
            },
            "generation": {
                "root": run_root.name,
                "request_plan_hash": run_report["request_plan_hash"],
                "responses_file_sha256": run_report["responses_file_sha256"],
                "run_hash": run_report["run_hash"],
                "verification_hash": run_verification["verification_hash"],
                "verification_file_sha256": _sha256_file(
                    run_root / "verification.json"
                ),
            },
            "evaluation": {
                "root": evaluation_root.name,
                "report_hash": evaluation_report["report_hash"],
                "verification_hash": evaluation_verification[
                    "verification_hash"
                ],
                "verification_file_sha256": _sha256_file(
                    evaluation_root / "verification.json"
                ),
            },
            "superseded_evaluation": {
                "root": superseded_evaluation_root.name,
                "report_hash": superseded_report["report_hash"],
                "status": "preserved_not_used_for_claims",
            },
            "statistics": {
                "root": statistics_root.name,
                "report_hash": statistics_report["report_hash"],
                "verification_hash": statistics_verification[
                    "verification_hash"
                ],
                "verification_file_sha256": _sha256_file(
                    statistics_root / "verification.json"
                ),
            },
            "regression_receipt": {
                "path": str(regression_receipt_path.relative_to(repo_root)),
                "receipt_hash": regression_receipt.get("receipt_hash", ""),
                "file_sha256": _sha256_file(regression_receipt_path),
            },
        },
        "kill_gates": {
            "gate_1": inherited_gates["gate_1"],
            "gate_2": inherited_gates["gate_2"],
            "gate_3": inherited_gates["gate_3"],
            "gate_4": inherited_gates["gate_4"],
            "gate_5": {
                "name": "multi_generation",
                "status": "pass" if passed else "fail",
                "passed": passed,
                "preregistered_checks": statistics_report["gate_5"]["checks"],
                "thresholds": statistics_report["gate_5"]["thresholds"],
            },
            "gate_6": inherited_gates["gate_6"],
        },
        "statistical_summary": {
            "source_pair_count": packet_manifest["source_pair_count"],
            "source_run_count": packet_manifest["source_run_count"],
            "source_task_count": packet_manifest["source_task_count"],
            "domain_counts": packet_manifest["domain_counts"],
            "generation_count": packet_manifest["generation_count"],
            "paraphrase_replicate_count": len(
                packet_manifest["paraphrase_replicate_ids"]
            ),
            "request_count": run_report["request_count"],
            "system_count": len(packet_manifest["systems"]),
            "system_receipt_count": evaluation_report["system_receipt_count"],
            "final_generation": final_summary,
            "full_vs_unrestricted_laundering_reduction_ci_95": effects[
                "full_vs_unrestricted_laundering_reduction"
            ]["percentile_ci_95"],
            "full_vs_authority_laundering_reduction_ci_95": effects[
                "full_vs_authority_laundering_reduction"
            ]["percentile_ci_95"],
            "full_vs_global_vkr_delta_ci_95": effects[
                "full_vs_global_vkr_delta"
            ]["percentile_ci_95"],
            "bootstrap_iterations": bootstrap["iterations"],
            "holm_adjusted_p_values": {
                name: test["holm_adjusted_p_value"]
                for name, test in sorted(exact_tests.items())
            },
        },
        "claim_boundaries": [
            "supports_lineage_non_escalation_across_five_generations",
            "does_not_support_full_superiority_over_lineage_only",
            "host_policy_projection_over_one_frozen_descendant_dag",
            "authority_only_surface_ablation_is_not_universal_semantics",
            "same_domain_different_task_only",
            "tier2_online_effect_not_yet_established",
        ],
        "stop_gate_checks": stop_gate_checks,
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


def build_stop_gate(output_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    report = compute_stop_gate(**kwargs)
    _write_json_exclusive(output_root / "stop_gate_report.json", report)
    _write_text_exclusive(output_root / "stop_gate_report.md", _render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the WP8 Multi-generation Gate-5 Stop Gate."
    )
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--prior-tier1-stop-gate-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--superseded-evaluation-root", required=True, type=Path)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--regression-receipt", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    report = build_stop_gate(
        args.output_root,
        plan_path=args.plan,
        work_root=args.work_root,
        prior_tier1_stop_gate_root=args.prior_tier1_stop_gate_root,
        packet_root=args.packet_root,
        run_root=args.run_root,
        evaluation_root=args.evaluation_root,
        superseded_evaluation_root=args.superseded_evaluation_root,
        statistics_root=args.statistics_root,
        regression_receipt_path=args.regression_receipt,
        created_at=args.created_at,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
