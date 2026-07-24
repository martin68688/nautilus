#!/usr/bin/env python3
"""Verify the result-blind Tier-2 formal analysis-policy addendum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "decision_admissibility_wp8_tier2_formal_analysis_policy_addendum_v1"
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_"
    "analysis_policy_addendum_verification_v1"
)
POLICY_ID = "wp8-tier2-formal-analysis-policy-r1"
STATUS = "frozen_after_structural_dispositions_before_score_reveal"

TASKS = (
    "aerial-cactus-identification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
)
SEEDS = (104729, 130363, 155921)
SYSTEMS = (
    "full_decision_admissibility",
    "no_memory",
    "flat_relevance_memory",
    "global_validity_bit",
    "authority_only",
)
SCORED = "scored_selected_result"
FAILED = "pre_terminal_failure"

PREREGISTRATION_BINDINGS = (
    (
        "r1",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260722_r1.json",
        "8e52e09750e84027470624cabdf7b3933055a5dbb50b9c7e431e7b725f3f0806",
    ),
    (
        "r2",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260722_r2.json",
        "1e667eee5647d6a28d9d0441a35e691257b1a56c71e04be22e472317fd958247",
    ),
    (
        "r3",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260722_r3.json",
        "95659800115e75849cdecf7a62f67a4fc4a052821aca8b64af803785b0bf940e",
    ),
    (
        "r4",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260722_r4.json",
        "4594682b3134ec563a17829b9fbd0ac48026357cb695654373e343a20aad01cb",
    ),
    (
        "r5",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260723_r5.json",
        "36bed4960bac6732fb94150e937f1fe8e68826a5d2ccb63655cf8abdd1b2cf5c",
    ),
    (
        "r6",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260723_r6.json",
        "3b1290854630588a5c94f4cc50f8859fd9aef4c9bb63812020c7cc36610e039d",
    ),
    (
        "r7",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260723_r7.json",
        "d19db9e780f6f40f5f03e54f3512d8b86bd29b0b18e5ba6d0a4f18628a8c4ec3",
    ),
    (
        "r8",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260723_r8.json",
        "0af210a13d2462fb1a8bef266d30a8e91dd86416d3259366219cc16d931c7384",
    ),
    (
        "r9",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260723_r9.json",
        "dd41c01ae9b8331566d011f75b1b991cba7ddaf14d400de69cdcb0b3b23d90af",
    ),
)

STRUCTURAL_BINDINGS = (
    (
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "completed_blocks_freeze_20260723_r1.json",
        "74ab746d2cdb3a2bd011a2fe84affc2686d236103b20cbad1c696e7c5ccb2a6b",
    ),
    (
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "birds_seed104729_r3_structure_audit_20260723.json",
        "c36fb2d38d72137e57861af6c707a0a997cc70d79fc931ab4cde9c48ccbdec96",
    ),
    (
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "birds_seed130363_r5_structure_audit_20260723.json",
        "74e0ceb077a561ae169196d79bf0ed289a1bdb2d2b4aa884bdaa15ddd36ac4cf",
    ),
    (
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "birds_seed155921_r5_structure_audit_20260723.json",
        "8679701d0c8e5bba754c0e2bfb38e26bdb631c71bdfe6d3ddb471b45b82e3df4",
    ),
    (
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "taxi_seed104729_r5_structure_audit_20260723.json",
        "703335a21d5338ecc0fe05c490ae13ff40eeef448c78de6dd9b589a69c058db1",
    ),
    (
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "taxi_seed130363_r5_structure_audit_20260723.json",
        "7825c406f806cd6fb21a01418e551390d7d237e7b98432aa6ba524712f5c8b3d",
    ),
    (
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "taxi_seed155921_r5_structure_audit_20260723.json",
        "7091e109466bbccef12d8f57a5a2ff138eb996517143d95df9dcbe616f858f47",
    ),
)

EXPECTED_RESULT_BLIND_FREEZE = {
    "freeze_timing": STATUS,
    "terminal_score_values_included": False,
    "terminal_score_values_inspected_for_this_addendum": False,
    "oracle_score_values_inspected_for_this_addendum": False,
    "source_score_values_used_to_choose_policy": False,
    "policy_selected_from_structural_dispositions_only": True,
    "effect_claim_authorized": False,
}

EXPECTED_DESIGN = {
    "tasks": [
        {
            "task_id": "aerial-cactus-identification",
            "native_metric": "macro_f1",
            "direction": "maximize",
            "standardization": "native_delta_divided_by_1.0",
        },
        {
            "task_id": "mlsp-2013-birds",
            "native_metric": "macro_f1",
            "direction": "maximize",
            "standardization": "native_delta_divided_by_1.0",
        },
        {
            "task_id": "new-york-city-taxi-fare-prediction",
            "native_metric": "rmse",
            "direction": "minimize",
            "standardization": (
                "native_delta_divided_by_abs_contrast_right_reference_score"
            ),
        },
    ],
    "agent_seeds": list(SEEDS),
    "online_systems": list(SYSTEMS),
    "assigned_block_count": 9,
    "assigned_online_outcome_count": 45,
    "assigned_oracle_outcome_count": 9,
    "contrasts": [
        {
            "contrast_id": "full_minus_no_memory",
            "role": "primary",
            "left": "full_decision_admissibility",
            "right": "no_memory",
        },
        {
            "contrast_id": "full_minus_flat_relevance_memory",
            "role": "secondary",
            "left": "full_decision_admissibility",
            "right": "flat_relevance_memory",
        },
        {
            "contrast_id": "full_minus_global_validity_bit",
            "role": "secondary",
            "left": "full_decision_admissibility",
            "right": "global_validity_bit",
        },
        {
            "contrast_id": "full_minus_authority_only",
            "role": "secondary",
            "left": "full_decision_admissibility",
            "right": "authority_only",
        },
    ],
}

EXPECTED_ANALYSIS_POLICY = {
    "itt_population": {
        "estimand": "all_45_assigned_online_system_outcomes",
        "disposition_unit": "task_seed_system_assignment",
        "allowed_final_dispositions": [SCORED, FAILED],
        "all_failures_retained": True,
        "rerun_to_replace_failure_forbidden": True,
        "post_assignment_exclusion_forbidden": True,
        "oracle_or_source_score_imputation_forbidden": True,
        "other_system_score_imputation_forbidden": True,
        "constant_or_model_based_score_imputation_forbidden": True,
    },
    "continuous_native_and_standardized_effects": {
        "pair_eligibility": (
            "left_and_right_both_have_scored_selected_result_in_same_task_seed_block"
        ),
        "unscored_pair_semantics": (
            "continuous_delta_undefined_but_failed_assignment_retained_in_itt_"
            "disposition_and_completion_endpoint"
        ),
        "assigned_denominator_blocks": 9,
        "required_availability_report": {
            "numerator": "n_scored_pairs",
            "denominator": 9,
            "rendering": "n_scored_pairs/9",
            "missing_block_ids_required": True,
            "word_exclusion_for_missing_failures_forbidden": True,
        },
        "native_delta": {
            "maximize": "left_score_minus_right_score",
            "minimize": "right_score_minus_left_score",
            "positive_always_favors_left": True,
        },
        "standardized_delta": {
            "classification_macro_f1": "native_delta_divided_by_1.0",
            "regression_rmse": (
                "native_delta_divided_by_abs_contrast_right_reference_score"
            ),
            "regression_reference_system": "contrast_right",
            "zero_or_nonfinite_reference_rule": (
                "standardized_delta_not_estimable_no_epsilon_substitution"
            ),
        },
        "per_task_native_reporting_required": True,
        "per_task_required_fields": [
            "task_id",
            "native_metric",
            "direction",
            "n_scored_pairs",
            "assigned_pairs",
            "missing_block_ids",
            "paired_native_deltas",
            "mean_native_delta",
        ],
    },
    "cross_task_aggregation": {
        "raw_native_metric_pooling_forbidden": True,
        "raw_native_delta_pooling_across_tasks_forbidden": True,
        "allowed_cross_task_effect_summaries": [
            "task_macro_standardized_delta",
            "win_tie_loss",
        ],
        "task_macro_standardized_delta": (
            "unweighted_mean_of_three_task_mean_standardized_deltas"
        ),
        "all_three_tasks_required_for_task_macro": True,
        "task_without_scored_pair_rule": "task_macro_not_estimable",
        "win_tie_loss_unit": "available_task_seed_standardized_delta",
        "win_rule": "delta_greater_than_zero",
        "tie_rule": "delta_exactly_equal_to_zero_no_tolerance",
        "loss_rule": "delta_less_than_zero",
        "win_tie_loss_reports_n_over_9_and_missing_blocks": True,
    },
    "paired_bootstrap": {
        "seed": 20260723,
        "iterations": 20000,
        "rng": "numpy.random.Generator(numpy.random.PCG64(20260723))",
        "rng_reinitialized_per_contrast": True,
        "resampling_unit": "task_seed_block",
        "resampling_scope": "within_task_available_scored_pairs_with_replacement",
        "sample_size_per_task": "observed_available_scored_pair_count_for_task",
        "replicate_reduction": (
            "mean_within_each_task_then_unweighted_macro_mean_across_three_tasks"
        ),
        "all_three_tasks_require_at_least_one_pair": True,
        "confidence_level": 0.95,
        "interval": "percentile",
        "quantiles": [0.025, 0.975],
        "quantile_method": "linear",
    },
    "exact_sign_flip_and_holm": {
        "role": "inference_only_not_an_additional_cross_task_effect_summary",
        "test_unit": "available_task_seed_standardized_delta",
        "alternative": "mean_standardized_delta_greater_than_zero",
        "enumeration": "all_2_power_n_sign_vectors",
        "raw_p_formula": (
            "count(permuted_mean_greater_than_or_equal_to_observed_mean)" "/2_power_n"
        ),
        "zero_deltas_retained": True,
        "empty_available_set_rule": "not_estimable",
        "family": [
            "full_minus_no_memory",
            "full_minus_flat_relevance_memory",
            "full_minus_global_validity_bit",
            "full_minus_authority_only",
        ],
        "family_size": 4,
        "correction": "Holm_step_down",
        "holm_sort_tie_break": "frozen_family_order",
        "not_estimable_raw_p_for_holm": 1.0,
        "adjusted_p_cap": 1.0,
    },
    "completion_endpoint": {
        "population": "all_9_assigned_task_seed_blocks_per_online_system",
        "completed_value": 1,
        "completed_when": SCORED,
        "failed_value": 0,
        "failed_when": FAILED,
        "system_report": "completion_count_and_rate_with_denominator_9",
        "paired_binary_difference": (
            "mean_over_9_of_left_completion_minus_right_completion"
        ),
        "discordant_table_fields": [
            "both_completed",
            "left_only_completed",
            "right_only_completed",
            "neither_completed",
        ],
        "exact_discordant_sign_test": {
            "alternative": "left_completion_probability_greater_than_right",
            "null_probability": 0.5,
            "p_formula": (
                "binomial_upper_tail_of_left_only_completed_given_all_discordant"
            ),
            "zero_discordant_rule": "p_equals_1.0",
        },
    },
    "mixed_effects_sensitivity": {
        "role": "sensitivity_only_not_used_by_effect_claim_gate",
        "outcome": "available_task_seed_standardized_delta",
        "model": "intercept_only_linear_mixed_model_with_task_random_intercept",
        "minimum_distinct_tasks": 2,
        "minimum_pairs_per_contributing_task": 2,
        "minimum_total_pairs": 4,
        "required_diagnostics": [
            "optimizer_converged",
            "finite_fixed_effect",
            "finite_standard_error",
            "nonsingular_random_effect_covariance",
        ],
        "not_estimable_output": {
            "status": "not_estimable",
            "reason_required": True,
            "allowed_reasons": [
                "fewer_than_two_tasks_with_scored_pairs",
                "fewer_than_two_pairs_in_a_contributing_task",
                "fewer_than_four_total_scored_pairs",
                "zero_within_task_variation",
                "optimizer_nonconvergence",
                "singular_random_effect_covariance",
                "nonfinite_estimate_or_standard_error",
                "software_error",
            ],
        },
    },
}

EXPECTED_EFFECT_GATE = {
    "default_effect_claim_authorized": False,
    "decision_rule": "all_criteria_must_be_true_else_false",
    "not_estimable_or_missing_criterion_value": "criterion_false",
    "criteria": [
        "all_9_blocks_have_complete_five_system_dispositions",
        "all_45_assignments_retained_with_no_imputation_or_post_assignment_exclusion",
        "primary_has_at_least_one_scored_pair_in_each_of_all_3_tasks",
        "each_task_primary_mean_oriented_native_delta_is_strictly_greater_than_zero",
        "primary_task_macro_standardized_bootstrap_95pct_ci_lower_is_strictly_greater_than_zero",
        "primary_holm_adjusted_exact_one_sided_sign_flip_p_is_less_than_or_equal_to_0.05",
        "full_completion_count_is_greater_than_or_equal_to_no_memory_completion_count",
        "no_block_has_full_failed_and_no_memory_scored",
    ],
    "authorized_only_in_post_reveal_statistics_receipt": True,
    "this_addendum_effect_claim_authorized": False,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _binding_rows(
    preregistration: bool,
) -> list[dict[str, str]]:
    if preregistration:
        return [
            {"revision": revision, "path": path, "file_sha256": digest}
            for revision, path, digest in PREREGISTRATION_BINDINGS
        ]
    return [
        {"path": path, "file_sha256": digest} for path, digest in STRUCTURAL_BINDINGS
    ]


def _check_bound_files(
    rows: object,
    expected_rows: list[dict[str, str]],
    *,
    repo: Path,
    prefix: str,
    check: Callable[[str, object], None],
) -> None:
    check(f"{prefix}_bindings_exact", rows == expected_rows)
    if not isinstance(rows, list):
        return
    for index, expected in enumerate(expected_rows):
        row = rows[index] if index < len(rows) and isinstance(rows[index], dict) else {}
        relative = expected["path"]
        path = repo / relative
        check(f"{prefix}_{index}_path_exact", row.get("path") == relative)
        check(f"{prefix}_{index}_exists", path.is_file())
        check(
            f"{prefix}_{index}_file_sha256",
            path.is_file()
            and row.get("file_sha256") == expected["file_sha256"]
            and _file_sha256(path) == expected["file_sha256"],
        )


def _audit_slug(path: str) -> str:
    return Path(path).stem.replace("decision_admissibility_wp8_tier2_formal_", "")


def _derive_structural_matrix(
    repo: Path,
    check: Callable[[str, object], None],
) -> list[dict[str, Any]]:
    matrix: dict[tuple[str, int], dict[str, Any]] = {}
    freeze_relative = STRUCTURAL_BINDINGS[0][0]
    freeze_path = repo / freeze_relative
    freeze = _read(freeze_path) if freeze_path.is_file() else {}
    check(
        "freeze_score_values_included_false",
        freeze.get("score_values_included") is False,
    )
    check(
        "freeze_score_values_inspected_false",
        freeze.get("score_values_inspected") is False,
    )
    check("freeze_completed_block_count", freeze.get("completed_block_count") == 4)
    check(
        "freeze_completed_online_condition_count",
        freeze.get("completed_online_condition_count") == 20,
    )
    freeze_blocks = freeze.get("blocks") or {}
    expected_freeze_blocks = {
        f"wp8-tier2-formal-aerial-seed-{seed}-r3" for seed in SEEDS
    } | {"wp8-tier2-formal-birds-seed-104729-r3"}
    check(
        "freeze_block_ids_exact",
        isinstance(freeze_blocks, dict)
        and set(freeze_blocks) == expected_freeze_blocks,
    )
    if isinstance(freeze_blocks, dict):
        for seed in SEEDS:
            block_id = f"wp8-tier2-formal-aerial-seed-{seed}-r3"
            row = freeze_blocks.get(block_id) or {}
            check(
                f"freeze_aerial_{seed}_all_five_scored",
                row.get("task_id") == TASKS[0]
                and row.get("agent_seed") == seed
                and row.get("successful_selected_result_count") == 5
                and row.get("failed_online_condition_count") == 0
                and row.get("result_fact_count") == 5,
            )
            matrix[(TASKS[0], seed)] = {
                "block_id": block_id,
                "task_id": TASKS[0],
                "agent_seed": seed,
                "dispositions": {system: SCORED for system in SYSTEMS},
            }
        bird_row = freeze_blocks.get("wp8-tier2-formal-birds-seed-104729-r3") or {}
        check(
            "freeze_birds_104729_counts",
            bird_row.get("successful_selected_result_count") == 3
            and bird_row.get("failed_online_condition_count") == 2
            and bird_row.get("result_fact_count") == 3,
        )

    for relative, _digest in STRUCTURAL_BINDINGS[1:]:
        path = repo / relative
        audit = _read(path) if path.is_file() else {}
        slug = _audit_slug(relative)
        check(f"{slug}_status_passed", audit.get("status") == "passed")
        check(
            f"{slug}_score_values_included_false",
            audit.get("score_values_included") is False,
        )
        check(
            f"{slug}_score_values_inspected_false",
            audit.get("score_values_inspected") is False,
        )
        success_count = audit.get("successful_selected_result_count")
        failure_count = audit.get("failed_online_condition_count")
        check(f"{slug}_online_count", audit.get("online_condition_count") == 5)
        check(
            f"{slug}_count_partition",
            isinstance(success_count, int)
            and not isinstance(success_count, bool)
            and isinstance(failure_count, int)
            and not isinstance(failure_count, bool)
            and success_count + failure_count == 5,
        )
        check(
            f"{slug}_result_fact_count", audit.get("result_fact_count") == success_count
        )
        task_id = audit.get("task_id")
        seed = audit.get("agent_seed")
        check(
            f"{slug}_known_task_seed",
            task_id in TASKS and seed in SEEDS,
        )
        conditions = audit.get("condition_structure") or {}
        check(
            f"{slug}_systems_exact",
            isinstance(conditions, dict) and set(conditions) == set(SYSTEMS),
        )
        dispositions: dict[str, str] = {}
        if isinstance(conditions, dict):
            for system in SYSTEMS:
                condition = conditions.get(system) or {}
                disposition = condition.get("status")
                check(
                    f"{slug}_{system}_disposition",
                    disposition in {SCORED, FAILED},
                )
                if disposition == SCORED:
                    check(
                        f"{slug}_{system}_scored_shape",
                        condition.get("result_fact_count") == 1
                        and condition.get("terminal_value_omitted") is True,
                    )
                elif disposition == FAILED:
                    check(
                        f"{slug}_{system}_failure_shape",
                        condition.get("result_fact_count") == 0
                        and condition.get("terminal_metric_observed") is False,
                    )
                dispositions[system] = str(disposition or "")
        check(
            f"{slug}_condition_counts_match_summary",
            list(dispositions.values()).count(SCORED) == success_count
            and list(dispositions.values()).count(FAILED) == failure_count,
        )
        if task_id in TASKS and seed in SEEDS:
            key = (str(task_id), int(seed))
            check(f"{slug}_task_seed_unique", key not in matrix)
            matrix[key] = {
                "block_id": audit.get("block_id"),
                "task_id": task_id,
                "agent_seed": seed,
                "dispositions": dispositions,
            }

    check(
        "derived_task_seed_matrix_complete",
        set(matrix) == {(task, seed) for task in TASKS for seed in SEEDS},
    )
    return [
        matrix.get(
            (task, seed),
            {
                "block_id": "",
                "task_id": task,
                "agent_seed": seed,
                "dispositions": {},
            },
        )
        for task in TASKS
        for seed in SEEDS
    ]


def _derive_counts(matrix: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, dict[str, int]] = {}
    by_system: dict[str, dict[str, int]] = {}
    scored_total = 0
    failed_total = 0
    for task in TASKS:
        task_rows = [row for row in matrix if row.get("task_id") == task]
        scored = sum(
            disposition == SCORED
            for row in task_rows
            for disposition in (row.get("dispositions") or {}).values()
        )
        failed = sum(
            disposition == FAILED
            for row in task_rows
            for disposition in (row.get("dispositions") or {}).values()
        )
        by_task[task] = {
            "block_count": len(task_rows),
            "scored_selected_result_count": scored,
            "failed_online_condition_count": failed,
        }
        scored_total += scored
        failed_total += failed
    for system in SYSTEMS:
        dispositions = [(row.get("dispositions") or {}).get(system) for row in matrix]
        by_system[system] = {
            "assigned_block_count": 9,
            "scored_selected_result_count": dispositions.count(SCORED),
            "failed_online_condition_count": dispositions.count(FAILED),
        }
    return {
        "block_count": 9,
        "assigned_online_outcome_count": 45,
        "scored_selected_result_count": scored_total,
        "failed_online_condition_count": failed_total,
        "by_task": by_task,
        "by_system": by_system,
    }


def _contains_observed_score_payload(value: object) -> bool:
    forbidden = {
        "score",
        "scores",
        "terminal_value",
        "terminal_values",
        "terminal_score",
        "terminal_scores",
        "observed_score",
        "observed_scores",
        "oracle_score",
        "oracle_scores",
        "source_score",
        "source_scores",
    }
    if isinstance(value, dict):
        if any(str(key).lower() in forbidden for key in value):
            return True
        return any(_contains_observed_score_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_observed_score_payload(item) for item in value)
    return False


def verify_analysis_policy_addendum(
    addendum_path: str | Path,
    *,
    repo_root: str | Path = ROOT,
) -> dict[str, Any]:
    path = Path(addendum_path).resolve()
    repo = Path(repo_root).resolve()
    payload = _read(path)
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: object) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(name)

    check("schema", payload.get("schema") == SCHEMA)
    check("policy_id", payload.get("analysis_policy_id") == POLICY_ID)
    check("status", payload.get("status") == STATUS)
    check(
        "top_level_keys_exact",
        set(payload)
        == {
            "schema",
            "analysis_policy_id",
            "status",
            "frozen_at_utc",
            "preregistration_file_bindings",
            "structural_evidence_file_bindings",
            "result_blind_freeze",
            "design",
            "itt_disposition_matrix",
            "known_structural_counts",
            "analysis_policy",
            "effect_claim_gate",
            "analysis_policy_hash",
        },
    )
    check(
        "analysis_policy_hash",
        payload.get("analysis_policy_hash")
        == _payload_hash(payload, "analysis_policy_hash"),
    )
    check(
        "no_observed_score_payload",
        not _contains_observed_score_payload(payload),
    )

    _check_bound_files(
        payload.get("preregistration_file_bindings"),
        _binding_rows(True),
        repo=repo,
        prefix="preregistration",
        check=check,
    )
    _check_bound_files(
        payload.get("structural_evidence_file_bindings"),
        _binding_rows(False),
        repo=repo,
        prefix="structural",
        check=check,
    )

    check(
        "result_blind_freeze_exact",
        payload.get("result_blind_freeze") == EXPECTED_RESULT_BLIND_FREEZE,
    )
    check("design_exact", payload.get("design") == EXPECTED_DESIGN)
    check(
        "analysis_policy_exact",
        payload.get("analysis_policy") == EXPECTED_ANALYSIS_POLICY,
    )
    check(
        "effect_claim_gate_exact",
        payload.get("effect_claim_gate") == EXPECTED_EFFECT_GATE,
    )

    derived_matrix = _derive_structural_matrix(repo, check)
    matrix = payload.get("itt_disposition_matrix")
    check("itt_disposition_matrix_exact", matrix == derived_matrix)
    check("itt_block_count", isinstance(matrix, list) and len(matrix) == 9)
    dispositions = (
        [
            disposition
            for row in matrix
            if isinstance(row, dict)
            for disposition in (
                row.get("dispositions", {}).values()
                if isinstance(row.get("dispositions"), dict)
                else []
            )
        ]
        if isinstance(matrix, list)
        else []
    )
    check("itt_assigned_outcome_count", len(dispositions) == 45)
    check(
        "itt_dispositions_partition",
        len(dispositions) == 45
        and all(disposition in {SCORED, FAILED} for disposition in dispositions),
    )
    expected_counts = _derive_counts(derived_matrix)
    check(
        "known_structural_counts_exact",
        payload.get("known_structural_counts") == expected_counts,
    )
    check(
        "known_structural_count_partition",
        expected_counts["scored_selected_result_count"]
        + expected_counts["failed_online_condition_count"]
        == 45,
    )

    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "analysis_policy_id": payload.get("analysis_policy_id", ""),
        "addendum_file_sha256": _file_sha256(path),
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": sorted(set(errors)),
        "verified": not errors,
        "verifier_source_sha256": _file_sha256(Path(__file__).resolve()),
        "verification_hash": "",
    }
    report["verification_hash"] = _payload_hash(report, "verification_hash")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addendum", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_analysis_policy_addendum(
        args.addendum,
        repo_root=args.repo_root,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["verified"] else 1)


if __name__ == "__main__":
    main()


__all__ = [
    "POLICY_ID",
    "SCHEMA",
    "STATUS",
    "VERIFICATION_SCHEMA",
    "verify_analysis_policy_addendum",
]
