"""Analyze frozen WP8 Tier-2 formal results under the result-blind policy.

This is the first component authorized to deserialize formal terminal summaries.
It requires a passed hash-only joint inventory and its independent verification,
retains every assigned failure, and never imputes a missing terminal score.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OBSERVATION_SCHEMA = "decision_admissibility_wp8_tier2_formal_observation_v1"
ORACLE_SCHEMA = "decision_admissibility_wp8_tier2_formal_oracle_observation_v1"
PAIR_SCHEMA = "decision_admissibility_wp8_tier2_formal_paired_contrast_v1"
ORACLE_GAP_SCHEMA = "decision_admissibility_wp8_tier2_formal_oracle_gap_v1"
REPORT_SCHEMA = "decision_admissibility_wp8_tier2_formal_statistics_v1"
MANIFEST_SCHEMA = "decision_admissibility_wp8_tier2_formal_statistics_manifest_v1"
POLICY_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_analysis_policy_addendum_v1"
)
INVENTORY_SCHEMA = "decision_admissibility_wp8_tier2_formal_joint_inventory_v1"
INVENTORY_MANIFEST_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_joint_inventory_manifest_v1"
)
INVENTORY_VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_joint_inventory_verification_v1"
)
SUMMARY_SCHEMA = "decision_admissibility_wp8_tier2_formal_evaluation_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        _canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value), sort_keys=True, ensure_ascii=False, indent=2
    ) + "\n"


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_exclusive(path, _json_text(value))


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) == _payload_hash(payload, field)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric value: {label}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite value: {label}")
    return result


def _exact_binomial_upper(successes: int, total: int) -> float:
    if total == 0:
        return 1.0
    return sum(math.comb(total, value) for value in range(successes, total + 1)) / (
        2**total
    )


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total <= 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return [(centre - margin) / denominator, (centre + margin) / denominator]


def _exact_sign_flip_p(deltas: Sequence[float]) -> float | None:
    if not deltas:
        return None
    observed = sum(deltas) / len(deltas)
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        permuted = sum(sign * value for sign, value in zip(signs, deltas)) / len(
            deltas
        )
        count += permuted >= observed
        total += 1
    return count / total


def _holm_adjust(
    p_values: Mapping[str, float], family_order: Sequence[str]
) -> dict[str, float]:
    order_index = {name: index for index, name in enumerate(family_order)}
    ordered = sorted(
        p_values.items(), key=lambda item: (float(item[1]), order_index[item[0]])
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return {name: adjusted[name] for name in family_order}


def _bootstrap_task_macro(
    values_by_task: Mapping[str, Sequence[float]],
    *,
    tasks: Sequence[str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if any(not values_by_task.get(task) for task in tasks):
        return {
            "status": "not_estimable",
            "reason": "at_least_one_task_has_no_scored_pair",
            "iterations": iterations,
            "seed": seed,
            "estimate": None,
            "confidence_interval_95": None,
        }
    estimate = float(
        np.mean([np.mean(np.asarray(values_by_task[task])) for task in tasks])
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(iterations, dtype=np.float64)
    arrays = {task: np.asarray(values_by_task[task], dtype=np.float64) for task in tasks}
    for index in range(iterations):
        task_means = []
        for task in tasks:
            values = arrays[task]
            sampled = rng.choice(values, size=len(values), replace=True)
            task_means.append(float(np.mean(sampled)))
        samples[index] = float(np.mean(task_means))
    interval = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "status": "estimated",
        "reason": "",
        "iterations": iterations,
        "seed": seed,
        "estimate": estimate,
        "confidence_interval_95": [float(interval[0]), float(interval[1])],
    }


def _mixed_effects_sensitivity(
    deltas_by_task: Mapping[str, Sequence[float]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    rule = policy["analysis_policy"]["mixed_effects_sensitivity"]
    contributing = {task: list(values) for task, values in deltas_by_task.items() if values}
    if len(contributing) < int(rule["minimum_distinct_tasks"]):
        return {"status": "not_estimable", "reason": "fewer_than_two_tasks_with_scored_pairs"}
    if any(len(values) < int(rule["minimum_pairs_per_contributing_task"]) for values in contributing.values()):
        return {"status": "not_estimable", "reason": "fewer_than_two_pairs_in_a_contributing_task"}
    if sum(map(len, contributing.values())) < int(rule["minimum_total_pairs"]):
        return {"status": "not_estimable", "reason": "fewer_than_four_total_scored_pairs"}
    values = [value for task in sorted(contributing) for value in contributing[task]]
    if all(math.isclose(value, values[0], rel_tol=0.0, abs_tol=0.0) for value in values):
        return {"status": "not_estimable", "reason": "zero_within_task_variation"}
    try:
        import statsmodels.api as sm  # type: ignore

        groups = [task for task in sorted(contributing) for _ in contributing[task]]
        outcome = np.asarray(values, dtype=np.float64)
        exog = np.ones((len(outcome), 1), dtype=np.float64)
        fitted = sm.MixedLM(outcome, exog, groups=groups).fit(
            reml=True, method="lbfgs", disp=False
        )
        estimate = float(fitted.fe_params[0])
        standard_error = float(fitted.bse_fe[0])
        random_variance = float(np.asarray(fitted.cov_re)[0, 0])
        if not bool(fitted.converged):
            return {"status": "not_estimable", "reason": "optimizer_nonconvergence"}
        if not all(map(math.isfinite, (estimate, standard_error, random_variance))):
            return {"status": "not_estimable", "reason": "nonfinite_estimate_or_standard_error"}
        if random_variance <= 0.0:
            return {"status": "not_estimable", "reason": "singular_random_effect_covariance"}
        return {
            "status": "estimated",
            "model": rule["model"],
            "fixed_intercept": estimate,
            "standard_error": standard_error,
            "confidence_interval_95": [
                estimate - 1.959963984540054 * standard_error,
                estimate + 1.959963984540054 * standard_error,
            ],
            "random_intercept_variance": random_variance,
            "optimizer_converged": True,
            "n_pairs": len(values),
            "n_tasks": len(contributing),
        }
    except Exception as error:
        return {
            "status": "not_estimable",
            "reason": "software_error",
            "error_type": type(error).__name__,
        }


def _validate_inputs(
    *,
    analysis_policy_path: Path,
    inventory_root: Path,
    inventory_verification_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = _read_object(analysis_policy_path)
    if policy.get("schema") != POLICY_SCHEMA or policy.get("status") != (
        "frozen_after_structural_dispositions_before_score_reveal"
    ):
        raise ValueError("Analysis policy is not the frozen result-blind addendum")
    if not _valid_hash(policy, "analysis_policy_hash"):
        raise ValueError("Analysis policy hash mismatch")
    if (policy.get("result_blind_freeze") or {}).get(
        "terminal_score_values_inspected_for_this_addendum"
    ) is not False:
        raise ValueError("Analysis policy was not result-blind")

    report_path = inventory_root / "joint_inventory.json"
    manifest_path = inventory_root / "manifest.json"
    inventory = _read_object(report_path)
    manifest = _read_object(manifest_path)
    verification = _read_object(inventory_verification_path)
    if inventory.get("schema") != INVENTORY_SCHEMA or inventory.get("status") != "passed":
        raise ValueError("Joint inventory is not passed")
    if not _valid_hash(inventory, "report_hash"):
        raise ValueError("Joint inventory hash mismatch")
    if manifest.get("schema") != INVENTORY_MANIFEST_SCHEMA or not _valid_hash(
        manifest, "manifest_hash"
    ):
        raise ValueError("Joint inventory manifest mismatch")
    if manifest.get("joint_inventory_file_sha256") != _sha256_file(report_path):
        raise ValueError("Joint inventory file binding mismatch")
    if verification.get("schema") != INVENTORY_VERIFICATION_SCHEMA or verification.get(
        "verified"
    ) is not True or not _valid_hash(verification, "verification_hash"):
        raise ValueError("Joint inventory verification is not passed")
    if verification.get("joint_inventory_hash") != inventory.get("report_hash"):
        raise ValueError("Joint inventory verification binding mismatch")
    return policy, inventory, manifest, verification


def _extract_observations(
    policy: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    completed_root: Path,
    continuation_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roots = {
        "completed_r10": completed_root.resolve(),
        "continuation_r13": continuation_root.resolve(),
    }
    declared_roots = inventory.get("formal_roots") or {}
    for role, root in roots.items():
        if Path(str(declared_roots.get(role) or "")).resolve() != root:
            raise ValueError(f"Formal result root mismatch: {role}")
    disposition_by_block = {
        row["block_id"]: row for row in policy["itt_disposition_matrix"]
    }
    task_design = {row["task_id"]: row for row in policy["design"]["tasks"]}
    observations: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    for block in inventory["blocks"]:
        block_id = str(block["block_id"])
        role = str(block["root_role"])
        summary_path = roots[role] / "blocks" / block_id / "EVALUATION_SUMMARY.json"
        expected_sha = block["required_artifact_hashes"]["EVALUATION_SUMMARY.json"]
        if _sha256_file(summary_path) != expected_sha:
            raise ValueError(f"Formal summary changed after joint inventory: {block_id}")
        summary = _read_object(summary_path)
        if summary.get("schema") != SUMMARY_SCHEMA or not _valid_hash(
            summary, "summary_hash"
        ):
            raise ValueError(f"Formal summary hash mismatch: {block_id}")
        for field in ("block_id", "task_id", "agent_seed"):
            if summary.get(field) != block.get(field):
                raise ValueError(f"Formal summary metadata mismatch: {block_id}:{field}")
        task = task_design[str(block["task_id"])]
        if summary.get("metric") != task["native_metric"]:
            raise ValueError(f"Formal metric mismatch: {block_id}")
        maximize = task["direction"] == "maximize"
        if summary.get("maximize") is not maximize:
            raise ValueError(f"Formal metric direction mismatch: {block_id}")
        online = summary.get("online_conditions") or {}
        expected_dispositions = disposition_by_block[block_id]["dispositions"]
        if set(online) != set(expected_dispositions):
            raise ValueError(f"Formal condition universe mismatch: {block_id}")
        for condition in policy["design"]["online_systems"]:
            source = online[condition]
            status = str(source.get("status") or "")
            if status != expected_dispositions[condition]:
                raise ValueError(f"Formal disposition drift: {block_id}:{condition}")
            score: float | None = None
            failure_classification = ""
            if status == "scored_selected_result":
                score = _finite_number(
                    source.get("selected_score"), label=f"{block_id}:{condition}:score"
                )
                if task["native_metric"] == "macro_f1" and not 0.0 <= score <= 1.0:
                    raise ValueError(f"Macro-F1 is outside [0,1]: {block_id}:{condition}")
                if task["native_metric"] == "rmse" and score < 0.0:
                    raise ValueError(f"RMSE is negative: {block_id}:{condition}")
                if source.get("result_fact_count") != 1 or source.get(
                    "result_fact_derived_from_refs"
                ) != []:
                    raise ValueError(f"Successful Result Fact mismatch: {block_id}:{condition}")
            else:
                if "selected_score" in source or source.get("terminal_metric_observed") is not False:
                    raise ValueError(f"Failed assignment contains terminal score: {block_id}:{condition}")
                if int(source.get("result_fact_count", 0)) != 0:
                    raise ValueError(f"Failed assignment contains Result Fact: {block_id}:{condition}")
                failure_classification = str(
                    source.get("failure_classification") or status
                )
            row: dict[str, Any] = {
                "schema": OBSERVATION_SCHEMA,
                "block_id": block_id,
                "task_id": block["task_id"],
                "agent_seed": block["agent_seed"],
                "system": condition,
                "disposition": status,
                "completed": status == "scored_selected_result",
                "native_metric": task["native_metric"],
                "direction": task["direction"],
                "score": score,
                "failure_classification": failure_classification,
                "result_fact_count": 1 if status == "scored_selected_result" else 0,
                "source_snapshot_sha256": block["source_snapshot_sha256"],
                "staging_gate_hash": block["staging_gate_hash"],
                "evaluation_summary_sha256": expected_sha,
                "evaluation_summary_hash": summary["summary_hash"],
                "imputed": False,
                "excluded": False,
                "row_hash": "",
            }
            row["row_hash"] = _payload_hash(row, "row_hash")
            observations.append(row)

        oracle = summary.get("oracle") or {}
        if oracle.get("normal_result_fact_published") is not False:
            raise ValueError(f"Oracle published normal Result Fact: {block_id}")
        oracle_score = oracle.get("best_score")
        if oracle_score is not None:
            oracle_score = _finite_number(oracle_score, label=f"{block_id}:oracle")
            if task["native_metric"] == "macro_f1" and not 0.0 <= oracle_score <= 1.0:
                raise ValueError(f"Oracle macro-F1 outside [0,1]: {block_id}")
            if task["native_metric"] == "rmse" and oracle_score < 0.0:
                raise ValueError(f"Oracle RMSE is negative: {block_id}")
        oracle_row: dict[str, Any] = {
            "schema": ORACLE_SCHEMA,
            "block_id": block_id,
            "task_id": block["task_id"],
            "agent_seed": block["agent_seed"],
            "native_metric": task["native_metric"],
            "direction": task["direction"],
            "score": oracle_score,
            "candidate_union_count": int(oracle.get("candidate_union_count", 0)),
            "scored_candidate_count": int(oracle.get("scored_candidate_count", 0)),
            "normal_result_fact_published": False,
            "agent_visible": False,
            "row_hash": "",
        }
        oracle_row["row_hash"] = _payload_hash(oracle_row, "row_hash")
        oracle_rows.append(oracle_row)
    observations.sort(key=lambda row: (row["task_id"], row["agent_seed"], row["system"]))
    oracle_rows.sort(key=lambda row: (row["task_id"], row["agent_seed"]))
    if len(observations) != 45 or len(oracle_rows) != 9:
        raise ValueError("Formal observation count mismatch")
    return observations, oracle_rows


def _build_pairs(
    policy: Mapping[str, Any], observations: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    by_block_system = {
        (str(row["block_id"]), str(row["system"])): row for row in observations
    }
    if len(by_block_system) != 45:
        raise ValueError("Duplicate formal system observation")
    blocks = sorted({str(row["block_id"]) for row in observations})
    pairs: list[dict[str, Any]] = []
    for contrast in policy["design"]["contrasts"]:
        contrast_id = str(contrast["contrast_id"])
        left_system = str(contrast["left"])
        right_system = str(contrast["right"])
        for block_id in blocks:
            left = by_block_system[(block_id, left_system)]
            right = by_block_system[(block_id, right_system)]
            eligible = bool(left["completed"] and right["completed"])
            native_delta: float | None = None
            standardized_delta: float | None = None
            standardization_status = "not_eligible"
            if eligible:
                left_score = float(left["score"])
                right_score = float(right["score"])
                if left["direction"] == "maximize":
                    native_delta = left_score - right_score
                else:
                    native_delta = right_score - left_score
                if left["native_metric"] == "macro_f1":
                    standardized_delta = native_delta
                    standardization_status = "estimated"
                elif abs(right_score) > 0.0 and math.isfinite(right_score):
                    standardized_delta = native_delta / abs(right_score)
                    standardization_status = "estimated"
                else:
                    standardization_status = "zero_or_nonfinite_reference"
            pair: dict[str, Any] = {
                "schema": PAIR_SCHEMA,
                "contrast_id": contrast_id,
                "role": contrast["role"],
                "left_system": left_system,
                "right_system": right_system,
                "block_id": block_id,
                "task_id": left["task_id"],
                "agent_seed": left["agent_seed"],
                "native_metric": left["native_metric"],
                "direction": left["direction"],
                "left_disposition": left["disposition"],
                "right_disposition": right["disposition"],
                "left_completed": left["completed"],
                "right_completed": right["completed"],
                "completion_difference": int(left["completed"]) - int(right["completed"]),
                "continuous_pair_eligible": eligible,
                "native_delta": native_delta,
                "standardized_delta": standardized_delta,
                "standardization_status": standardization_status,
                "imputed": False,
                "excluded": False,
                "row_hash": "",
            }
            pair["row_hash"] = _payload_hash(pair, "row_hash")
            pairs.append(pair)
    pairs.sort(key=lambda row: (row["contrast_id"], row["task_id"], row["agent_seed"]))
    if len(pairs) != 36:
        raise ValueError("Formal contrast row count mismatch")
    return pairs


def _summarize(
    policy: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks = [row["task_id"] for row in policy["design"]["tasks"]]
    systems = list(policy["design"]["online_systems"])
    completion_by_system: dict[str, Any] = {}
    for system in systems:
        rows = [row for row in observations if row["system"] == system]
        completed = sum(bool(row["completed"]) for row in rows)
        completion_by_system[system] = {
            "completed": completed,
            "assigned": len(rows),
            "rate": completed / len(rows),
            "wilson_confidence_interval_95": _wilson_interval(completed, len(rows)),
        }

    contrast_summaries: dict[str, Any] = {}
    raw_p_inputs: dict[str, float] = {}
    family_order = [row["contrast_id"] for row in policy["design"]["contrasts"]]
    for contrast_id in family_order:
        rows = [row for row in pairs if row["contrast_id"] == contrast_id]
        left_only = sum(row["left_completed"] and not row["right_completed"] for row in rows)
        right_only = sum(row["right_completed"] and not row["left_completed"] for row in rows)
        both = sum(row["left_completed"] and row["right_completed"] for row in rows)
        neither = len(rows) - left_only - right_only - both
        available = [
            row
            for row in rows
            if row["continuous_pair_eligible"]
            and row["standardized_delta"] is not None
        ]
        standardized = [float(row["standardized_delta"]) for row in available]
        raw_p = _exact_sign_flip_p(standardized)
        raw_p_inputs[contrast_id] = 1.0 if raw_p is None else raw_p
        by_task: dict[str, Any] = {}
        standardized_by_task: dict[str, list[float]] = defaultdict(list)
        for task in tasks:
            task_rows = [row for row in rows if row["task_id"] == task]
            eligible = [row for row in task_rows if row["continuous_pair_eligible"]]
            native = [float(row["native_delta"]) for row in eligible]
            standardized_task = [
                float(row["standardized_delta"])
                for row in eligible
                if row["standardized_delta"] is not None
            ]
            standardized_by_task[task].extend(standardized_task)
            by_task[task] = {
                "task_id": task,
                "native_metric": task_rows[0]["native_metric"],
                "direction": task_rows[0]["direction"],
                "n_scored_pairs": len(eligible),
                "assigned_pairs": len(task_rows),
                "missing_block_ids": [
                    row["block_id"] for row in task_rows if not row["continuous_pair_eligible"]
                ],
                "paired_native_deltas": native,
                "paired_standardized_deltas": standardized_task,
                "mean_native_delta": float(np.mean(native)) if native else None,
                "mean_standardized_delta": (
                    float(np.mean(standardized_task)) if standardized_task else None
                ),
            }
        bootstrap_rule = policy["analysis_policy"]["paired_bootstrap"]
        bootstrap = _bootstrap_task_macro(
            standardized_by_task,
            tasks=tasks,
            iterations=int(bootstrap_rule["iterations"]),
            seed=int(bootstrap_rule["seed"]),
        )
        contrast_summaries[contrast_id] = {
            "role": rows[0]["role"],
            "left_system": rows[0]["left_system"],
            "right_system": rows[0]["right_system"],
            "completion": {
                "both_completed": both,
                "left_only_completed": left_only,
                "right_only_completed": right_only,
                "neither_completed": neither,
                "left_minus_right_mean": sum(int(row["completion_difference"]) for row in rows) / len(rows),
                "exact_discordant_one_sided_p": _exact_binomial_upper(
                    left_only, left_only + right_only
                ),
            },
            "continuous": {
                "n_scored_pairs": len(available),
                "assigned_pairs": len(rows),
                "availability": f"{len(available)}/{len(rows)}",
                "missing_block_ids": [
                    row["block_id"] for row in rows if not row["continuous_pair_eligible"]
                ],
                "win_tie_loss": {
                    "wins": sum(value > 0.0 for value in standardized),
                    "ties": sum(value == 0.0 for value in standardized),
                    "losses": sum(value < 0.0 for value in standardized),
                    "available": len(standardized),
                    "assigned": len(rows),
                },
                "per_task": by_task,
                "task_macro_standardized_bootstrap": bootstrap,
                "exact_one_sided_sign_flip_raw_p": raw_p,
                "mixed_effects_sensitivity": _mixed_effects_sensitivity(
                    standardized_by_task, policy
                ),
            },
        }
    holm = _holm_adjust(raw_p_inputs, family_order)
    for contrast_id in family_order:
        contrast_summaries[contrast_id]["continuous"][
            "holm_adjusted_p"
        ] = holm[contrast_id]
        contrast_summaries[contrast_id]["continuous"][
            "holm_input_for_not_estimable_is_one"
        ] = contrast_summaries[contrast_id]["continuous"][
            "exact_one_sided_sign_flip_raw_p"
        ] is None

    oracle_by_block = {row["block_id"]: row for row in oracle_rows}
    oracle_gaps: list[dict[str, Any]] = []
    for observation in observations:
        oracle = oracle_by_block[observation["block_id"]]
        gap = None
        if observation["score"] is not None and oracle["score"] is not None:
            if observation["direction"] == "maximize":
                gap = float(oracle["score"]) - float(observation["score"])
            else:
                gap = float(observation["score"]) - float(oracle["score"])
        gap_row: dict[str, Any] = {
            "schema": ORACLE_GAP_SCHEMA,
            "block_id": observation["block_id"],
            "task_id": observation["task_id"],
            "agent_seed": observation["agent_seed"],
            "system": observation["system"],
            "system_disposition": observation["disposition"],
            "oracle_score_available": oracle["score"] is not None,
            "system_score_available": observation["score"] is not None,
            "oriented_oracle_minus_system_gap": gap,
            "row_hash": "",
        }
        gap_row["row_hash"] = _payload_hash(gap_row, "row_hash")
        oracle_gaps.append(gap_row)

    primary = contrast_summaries["full_minus_no_memory"]
    primary_tasks = primary["continuous"]["per_task"]
    bootstrap = primary["continuous"]["task_macro_standardized_bootstrap"]
    discordant_blocks = [
        row
        for row in pairs
        if row["contrast_id"] == "full_minus_no_memory"
        and not row["left_completed"]
        and row["right_completed"]
    ]
    criteria = {
        "all_9_blocks_have_complete_five_system_dispositions": len(observations) == 45,
        "all_45_assignments_retained_with_no_imputation_or_post_assignment_exclusion": all(
            row["imputed"] is False and row["excluded"] is False for row in observations
        ),
        "primary_has_at_least_one_scored_pair_in_each_of_all_3_tasks": all(
            primary_tasks[task]["n_scored_pairs"] >= 1 for task in tasks
        ),
        "each_task_primary_mean_oriented_native_delta_is_strictly_greater_than_zero": all(
            primary_tasks[task]["mean_native_delta"] is not None
            and primary_tasks[task]["mean_native_delta"] > 0.0
            for task in tasks
        ),
        "primary_task_macro_standardized_bootstrap_95pct_ci_lower_is_strictly_greater_than_zero": (
            bootstrap["status"] == "estimated"
            and bootstrap["confidence_interval_95"][0] > 0.0
        ),
        "primary_holm_adjusted_exact_one_sided_sign_flip_p_is_less_than_or_equal_to_0.05": primary[
            "continuous"
        ]["holm_adjusted_p"]
        <= 0.05,
        "full_completion_count_is_greater_than_or_equal_to_no_memory_completion_count": completion_by_system[
            "full_decision_admissibility"
        ]["completed"]
        >= completion_by_system["no_memory"]["completed"],
        "no_block_has_full_failed_and_no_memory_scored": len(discordant_blocks) == 0,
    }
    effect_claim_authorized = all(criteria.values())
    return (
        {
            "completion_by_system": completion_by_system,
            "contrasts": contrast_summaries,
            "holm_family_order": family_order,
            "oracle_gap_rows": len(oracle_gaps),
            "effect_claim_gate": {
                "criteria": criteria,
                "effect_claim_authorized": effect_claim_authorized,
                "failed_criteria": sorted(
                    name for name, value in criteria.items() if value is not True
                ),
            },
        },
        oracle_gaps,
    )


def compute_statistics(
    *,
    analysis_policy_path: str | Path,
    inventory_root: str | Path,
    inventory_verification_path: str | Path,
    completed_root: str | Path,
    continuation_root: str | Path,
    created_at: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    analysis_policy_path = Path(analysis_policy_path).resolve()
    inventory_root = Path(inventory_root).resolve()
    inventory_verification_path = Path(inventory_verification_path).resolve()
    policy, inventory, inventory_manifest, inventory_verification = _validate_inputs(
        analysis_policy_path=analysis_policy_path,
        inventory_root=inventory_root,
        inventory_verification_path=inventory_verification_path,
    )
    observations, oracle_rows = _extract_observations(
        policy,
        inventory,
        completed_root=Path(completed_root),
        continuation_root=Path(continuation_root),
    )
    pairs = _build_pairs(policy, observations)
    summary, oracle_gaps = _summarize(policy, observations, oracle_rows, pairs)
    scored = sum(row["completed"] for row in observations)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "created_at": str(created_at),
        "analysis_policy_binding": {
            "path": str(analysis_policy_path),
            "file_sha256": _sha256_file(analysis_policy_path),
            "analysis_policy_hash": policy["analysis_policy_hash"],
        },
        "joint_inventory_binding": {
            "root": str(inventory_root),
            "joint_inventory_hash": inventory["report_hash"],
            "manifest_hash": inventory_manifest["manifest_hash"],
            "verification_hash": inventory_verification["verification_hash"],
        },
        "analysis_population": {
            "assigned_online_outcomes": 45,
            "scored_selected_results": scored,
            "failed_online_conditions": 45 - scored,
            "assigned_oracle_dispositions": 9,
            "imputed_scores": 0,
            "post_assignment_exclusions": 0,
        },
        "native_metric_rule": (
            "macro_f1_reported_directly;rmse_deltas_sign_flipped;"
            "raw_macro_f1_and_rmse_never_pooled"
        ),
        "primary_contrast": "full_minus_no_memory",
        "secondary_contrasts": [
            "full_minus_flat_relevance_memory",
            "full_minus_global_validity_bit",
            "full_minus_authority_only",
        ],
        **summary,
        "observations_file": "formal_observations.jsonl",
        "oracle_observations_file": "oracle_observations.jsonl",
        "paired_contrasts_file": "paired_contrasts.jsonl",
        "oracle_gaps_file": "oracle_gaps.jsonl",
        "analyzer_source_sha256": _sha256_file(Path(__file__).resolve()),
        "report_hash": "",
    }
    report["report_hash"] = _payload_hash(report, "report_hash")
    return report, observations, oracle_rows, pairs, oracle_gaps


def build_statistics(output_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse Tier-2 statistics root: {output_root}")
    report, observations, oracle_rows, pairs, oracle_gaps = compute_statistics(**kwargs)
    output_root.mkdir(parents=True, exist_ok=False)
    payloads = {
        "formal_observations.jsonl": _jsonl_text(observations),
        "oracle_observations.jsonl": _jsonl_text(oracle_rows),
        "paired_contrasts.jsonl": _jsonl_text(pairs),
        "oracle_gaps.jsonl": _jsonl_text(oracle_gaps),
        "statistics_report.json": _json_text(report),
    }
    for filename, text in payloads.items():
        _write_text_exclusive(output_root / filename, text)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "files": {
            filename: _sha256_file(output_root / filename)
            for filename in sorted(payloads)
        },
        "statistics_report_hash": report["report_hash"],
        "analyzer_source_sha256": report["analyzer_source_sha256"],
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    _write_json_exclusive(output_root / "analysis_manifest.json", manifest)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-policy", required=True, type=Path)
    parser.add_argument("--inventory-root", required=True, type=Path)
    parser.add_argument("--inventory-verification", required=True, type=Path)
    parser.add_argument("--completed-root", required=True, type=Path)
    parser.add_argument("--continuation-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build_statistics(
        output_root=args.output_root,
        analysis_policy_path=args.analysis_policy,
        inventory_root=args.inventory_root,
        inventory_verification_path=args.inventory_verification,
        completed_root=args.completed_root,
        continuation_root=args.continuation_root,
        created_at=args.created_at,
    )
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()
