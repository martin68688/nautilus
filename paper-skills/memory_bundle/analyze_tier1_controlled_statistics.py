from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from build_tier1_controlled_episodes import CONDITIONS
from evaluate_tier1_controlled_decisions import SYSTEM_ROUTES
from schema import sha256_json
from verify_tier1_controlled_evaluation import verify_evaluation


STATISTICS_REPORT_SCHEMA = "decision_admissibility_tier1_statistics_report_v1"
SEED_DELTA_SCHEMA = "decision_admissibility_tier1_paired_seed_delta_v1"
DEFAULT_BOOTSTRAP_ITERATIONS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260721
EXPECTED_SYSTEMS = frozenset((*SYSTEM_ROUTES, "oracle"))

METRIC_NAMES = (
    "primary_raw_cell_iir",
    "primary_f11_vkr",
    "primary_stage_mismatch_action_change",
    "gate2_vkr_full_minus_global",
    "gate2_iir_full_minus_global",
    "gate3_oracle_f11_minus_f01",
    "gate3_action_difference_f01_vs_f11",
    "gate4_prompt_exposure_post_minus_full",
    "gate4_invalid_influence_post_minus_full",
    "utility_f11_minus_f01",
    "utility_full_minus_global",
)


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


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> list[float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        (proportion * (1 - proportion) + z * z / (4 * total)) / total
    )
    return [
        (centre - margin) / denominator,
        (centre + margin) / denominator,
    ]


def _holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - index) * float(value))
        running = max(running, candidate)
        adjusted[name] = running
    return {name: adjusted[name] for name in sorted(adjusted)}


def _build_pair_rows(
    decisions: Sequence[Mapping[str, Any]],
    counterfactuals: Sequence[Mapping[str, Any]],
    systems: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decision_groups: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in decisions:
        key = (str(row["episode_id"]), int(row["agent_replicate_id"]))
        condition = str(row["condition"])
        if condition in decision_groups[key]:
            raise ValueError(f"Duplicate condition in paired decision: {key}:{condition}")
        decision_groups[key][condition] = row

    system_groups: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in systems:
        key = (str(row["episode_id"]), int(row["agent_replicate_id"]))
        system = str(row["system"])
        if system in system_groups[key]:
            raise ValueError(f"Duplicate system in paired decision: {key}:{system}")
        system_groups[key][system] = row

    counterfactual_by_request = {
        str(row["memory_request_id"]): row for row in counterfactuals
    }
    if len(counterfactual_by_request) != len(counterfactuals):
        raise ValueError("Duplicate memory request in counterfactual receipts")

    pair_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    all_keys = sorted(set(decision_groups) | set(system_groups))
    for episode_id, replicate in all_keys:
        key = (episode_id, replicate)
        conditions = decision_groups.get(key, {})
        system_rows = system_groups.get(key, {})
        reasons: list[str] = []
        missing_conditions = sorted(set(CONDITIONS) - set(conditions))
        missing_systems = sorted(EXPECTED_SYSTEMS - set(system_rows))
        if missing_conditions:
            reasons.append(f"missing_conditions:{','.join(missing_conditions)}")
        if missing_systems:
            reasons.append(f"missing_systems:{','.join(missing_systems)}")
        counterfactual_by_condition: dict[str, Mapping[str, Any]] = {}
        for condition in CONDITIONS:
            if condition == "NM" or condition not in conditions:
                continue
            request_id = str(conditions[condition]["request_id"])
            counterfactual = counterfactual_by_request.get(request_id)
            if counterfactual is None:
                reasons.append(f"missing_counterfactual:{condition}")
            else:
                counterfactual_by_condition[condition] = counterfactual
        if reasons:
            exclusions.append(
                {
                    "episode_id": episode_id,
                    "agent_replicate_id": replicate,
                    "reasons": sorted(set(reasons)),
                }
            )
            continue

        metadata_fields = (
            "target_task_id",
            "source_task_id",
            "source_run_id",
            "source_episode_id",
            "domain",
            "stage",
        )
        metadata = {
            field: str(conditions["NM"][field]) for field in metadata_fields
        }
        for condition, row in conditions.items():
            for field, expected in metadata.items():
                if str(row[field]) != expected:
                    raise ValueError(
                        f"Paired metadata mismatch: {key}:{condition}:{field}"
                    )

        full = system_rows["full_decision_admissibility"]
        global_bit = system_rows["global_validity_bit"]
        post_tags = system_rows["post_prompt_claim_tags"]
        f00_cf = counterfactual_by_condition["F00"]
        f01_cf = counterfactual_by_condition["F01"]
        f10_cf = counterfactual_by_condition["F10"]
        f01 = conditions["F01"]
        f11 = conditions["F11"]
        binary = {
            "vkr_full": int(bool(full["valid_matched_knowledge_retained"])),
            "vkr_global": int(bool(global_bit["valid_matched_knowledge_retained"])),
            "iir_full": int(bool(full["invalid_claim_changed_action"])),
            "iir_global": int(bool(global_bit["invalid_claim_changed_action"])),
            "oracle_f11": int(bool(f11["oracle_action_selected"])),
            "oracle_f01": int(bool(f01["oracle_action_selected"])),
            "prompt_post": int(bool(post_tags["unauthorized_prompt_exposure"])),
            "prompt_full": int(bool(full["unauthorized_prompt_exposure"])),
            "influence_post": int(bool(post_tags["invalid_claim_changed_action"])),
            "influence_full": int(bool(full["invalid_claim_changed_action"])),
        }
        metrics = {
            "primary_raw_cell_iir": (
                int(bool(f00_cf["action_changed"]))
                + int(bool(f10_cf["action_changed"]))
            )
            / 2.0,
            "primary_f11_vkr": float(
                bool(
                    f11["runtime_actuation_receipt"][
                        "runtime_actuation_passed"
                    ]
                )
            ),
            "primary_stage_mismatch_action_change": float(
                bool(f01_cf["action_changed"])
            ),
            "gate2_vkr_full_minus_global": float(
                binary["vkr_full"] - binary["vkr_global"]
            ),
            "gate2_iir_full_minus_global": float(
                binary["iir_full"] - binary["iir_global"]
            ),
            "gate3_oracle_f11_minus_f01": float(
                binary["oracle_f11"] - binary["oracle_f01"]
            ),
            "gate3_action_difference_f01_vs_f11": float(
                f01["selected_action_id"] != f11["selected_action_id"]
            ),
            "gate4_prompt_exposure_post_minus_full": float(
                binary["prompt_post"] - binary["prompt_full"]
            ),
            "gate4_invalid_influence_post_minus_full": float(
                binary["influence_post"] - binary["influence_full"]
            ),
            "utility_f11_minus_f01": float(f11["controlled_action_utility"])
            - float(f01["controlled_action_utility"]),
            "utility_full_minus_global": float(full["controlled_action_utility"])
            - float(global_bit["controlled_action_utility"]),
        }
        pair_rows.append(
            {
                "episode_id": episode_id,
                "agent_replicate_id": replicate,
                **metadata,
                "metrics": metrics,
                "binary": binary,
                "utility": {
                    "f01": float(f01["controlled_action_utility"]),
                    "f11": float(f11["controlled_action_utility"]),
                    "full": float(full["controlled_action_utility"]),
                    "global": float(global_bit["controlled_action_utility"]),
                },
            }
        )
    return pair_rows, exclusions


def _hierarchical_paired_bootstrap(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    values = np.asarray(
        [
            [float(row["metrics"][metric]) for metric in METRIC_NAMES]
            for row in pair_rows
        ],
        dtype=float,
    )
    by_task_run: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(pair_rows):
        by_task_run[str(row["target_task_id"])][str(row["source_run_id"])].append(
            index
        )
    tasks = sorted(by_task_run)
    if not tasks:
        raise ValueError("No complete paired rows for bootstrap")
    rng = np.random.default_rng(seed)
    draws = np.empty((iterations, len(METRIC_NAMES)), dtype=float)
    for iteration in range(iterations):
        sampled_indices: list[int] = []
        for _ in range(len(tasks)):
            task = tasks[int(rng.integers(0, len(tasks)))]
            runs = sorted(by_task_run[task])
            for _ in range(len(runs)):
                run = runs[int(rng.integers(0, len(runs)))]
                seed_indices = by_task_run[task][run]
                for _ in range(len(seed_indices)):
                    sampled_indices.append(
                        seed_indices[int(rng.integers(0, len(seed_indices)))]
                    )
        draws[iteration, :] = values[sampled_indices, :].mean(axis=0)

    output: dict[str, Any] = {}
    for index, metric in enumerate(METRIC_NAMES):
        output[metric] = {
            "point_estimate": float(values[:, index].mean()),
            "percentile_ci_95": [
                float(np.quantile(draws[:, index], 0.025)),
                float(np.quantile(draws[:, index], 0.975)),
            ],
        }
    return output


def _paired_seed_deltas(
    pair_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replicates = sorted({int(row["agent_replicate_id"]) for row in pair_rows})
    for metric in METRIC_NAMES:
        for replicate in replicates:
            values = [
                float(row["metrics"][metric])
                for row in pair_rows
                if int(row["agent_replicate_id"]) == replicate
            ]
            payload: dict[str, Any] = {
                "schema": SEED_DELTA_SCHEMA,
                "metric": metric,
                "agent_replicate_id": replicate,
                "paired_episode_count": len(values),
                "sum_of_pair_values": sum(values),
                "mean_effect": _ratio(sum(values), len(values)),
                "row_hash": "",
            }
            payload["row_hash"] = sha256_json(
                {key: value for key, value in payload.items() if key != "row_hash"}
            )
            rows.append(payload)
    return rows


def _paired_exact_tests(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from scipy.stats import binomtest

    specs = {
        "gate2_vkr_full_gt_global": ("vkr_full", "vkr_global"),
        "gate3_oracle_f11_gt_f01": ("oracle_f11", "oracle_f01"),
        "gate4_prompt_exposure_post_gt_full": ("prompt_post", "prompt_full"),
        "gate4_invalid_influence_post_gt_full": (
            "influence_post",
            "influence_full",
        ),
    }
    tests: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for name, (left, right) in specs.items():
        positive = sum(
            row["binary"][left] == 1 and row["binary"][right] == 0
            for row in pair_rows
        )
        negative = sum(
            row["binary"][left] == 0 and row["binary"][right] == 1
            for row in pair_rows
        )
        discordant = positive + negative
        p_value = (
            float(
                binomtest(
                    positive,
                    discordant,
                    p=0.5,
                    alternative="greater",
                ).pvalue
            )
            if discordant
            else 1.0
        )
        raw_p_values[name] = p_value
        tests[name] = {
            "method": "exact paired discordance/binomial test",
            "alternative": "left_greater_than_right",
            "paired_decision_count": len(pair_rows),
            "positive_discordant_count": positive,
            "negative_discordant_count": negative,
            "tie_count": len(pair_rows) - discordant,
            "raw_p_value": p_value,
        }
    adjusted = _holm_adjust(raw_p_values)
    for name, value in adjusted.items():
        tests[name]["holm_adjusted_p_value"] = value
        tests[name]["reject_at_familywise_alpha_0_05"] = value < 0.05
    return tests


def _fit_binomial_mixed_model(
    pair_rows: Sequence[Mapping[str, Any]],
    *,
    name: str,
    baseline_key: str,
    treatment_key: str,
    baseline_label: str,
    treatment_label: str,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    observations: list[dict[str, Any]] = []
    for row in pair_rows:
        common = {
            "target_task_id": row["target_task_id"],
            "source_run_id": row["source_run_id"],
        }
        observations.append(
            {"outcome": row["binary"][baseline_key], "treatment": 0, **common}
        )
        observations.append(
            {"outcome": row["binary"][treatment_key], "treatment": 1, **common}
        )
    data = pd.DataFrame(observations)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = BinomialBayesMixedGLM.from_formula(
            "outcome ~ treatment",
            {
                "target_task": "0 + C(target_task_id)",
                "source_run": "0 + C(source_run_id)",
            },
            data,
            vcp_p=1.0,
            fe_p=2.0,
        )
        parameter_count = model.k_fep + model.k_vcp + model.k_vc
        result = model.fit_vb(
            mean=np.zeros(parameter_count, dtype=float),
            sd=np.full(parameter_count, math.exp(-0.5), dtype=float),
            fit_method="BFGS",
            minim_opts={"maxiter": 2_000},
            verbose=False,
        )
    treatment_index = model.exog_names.index("treatment")
    estimate = float(result.fe_mean[treatment_index])
    standard_deviation = float(result.fe_sd[treatment_index])
    lower = estimate - 1.959963984540054 * standard_deviation
    upper = estimate + 1.959963984540054 * standard_deviation
    optimization = dict(result.optim_retvals)
    baseline_values = data.loc[data["treatment"] == 0, "outcome"].to_numpy()
    treatment_values = data.loc[data["treatment"] == 1, "outcome"].to_numpy()
    return {
        "name": name,
        "model": "Bayesian binomial mixed-effects logistic regression",
        "estimation": "mean-field variational Bayes",
        "formula": "outcome ~ treatment + (1|target_task) + (1|source_run)",
        "baseline_label": baseline_label,
        "treatment_label": treatment_label,
        "observation_count": len(data),
        "paired_decision_count": len(pair_rows),
        "target_task_random_effect_count": int(data["target_task_id"].nunique()),
        "source_run_random_effect_count": int(data["source_run_id"].nunique()),
        "fixed_effect_treatment_log_odds": estimate,
        "fixed_effect_treatment_posterior_sd": standard_deviation,
        "fixed_effect_treatment_normal_approx_credible_interval_95": [lower, upper],
        "odds_ratio": math.exp(estimate),
        "odds_ratio_interval_95": [math.exp(lower), math.exp(upper)],
        "baseline_numerator": int(baseline_values.sum()),
        "baseline_denominator": int(len(baseline_values)),
        "treatment_numerator": int(treatment_values.sum()),
        "treatment_denominator": int(len(treatment_values)),
        "complete_separation_present": bool(
            len(np.unique(baseline_values)) == 1
            or len(np.unique(treatment_values)) == 1
        ),
        "prior_stabilized": True,
        "fixed_effect_prior_sd": 2.0,
        "variance_component_log_sd_prior_sd": 1.0,
        "deterministic_initial_mean": 0.0,
        "deterministic_initial_sd": math.exp(-0.5),
        "variance_components": {
            model.vcp_names[index]: {
                "posterior_log_sd_mean": float(result.vcp_mean[index]),
                "posterior_log_sd_sd": float(result.vcp_sd[index]),
                "posterior_sd_point": math.exp(float(result.vcp_mean[index])),
            }
            for index in range(len(model.vcp_names))
        },
        "optimizer_success": bool(optimization.get("success", False)),
        "optimizer_message": str(optimization.get("message", "")),
        "optimizer_iterations": int(optimization.get("nit", 0)),
        "warnings": sorted({str(item.message) for item in caught}),
    }


def _fit_linear_mixed_model(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    from scipy.stats import norm

    observations: list[dict[str, Any]] = []
    deltas: list[float] = []
    for row in pair_rows:
        common = {
            "target_task_id": row["target_task_id"],
            "source_run_id": row["source_run_id"],
        }
        observations.append(
            {"outcome": row["utility"]["f01"], "treatment": 0, **common}
        )
        observations.append(
            {"outcome": row["utility"]["f11"], "treatment": 1, **common}
        )
        deltas.append(float(row["metrics"]["utility_f11_minus_f01"]))
    data = pd.DataFrame(observations)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = smf.mixedlm(
            "outcome ~ treatment",
            data,
            groups=data["target_task_id"],
            re_formula="1",
            vc_formula={"source_run": "0 + C(source_run_id)"},
        )
        result = model.fit(
            reml=False,
            method="powell",
            maxiter=5_000,
            disp=False,
        )
    with warnings.catch_warnings(record=True) as post_fit_warnings:
        warnings.simplefilter("always")
        estimate = float(result.fe_params["treatment"])
        standard_error_raw = float(result.bse_fe["treatment"])
        target_variance_raw = float(result.cov_re.iloc[0, 0])
        source_variance_raw = (
            float(result.vcomp[0]) if len(result.vcomp) else None
        )
    standard_error = (
        standard_error_raw if math.isfinite(standard_error_raw) else None
    )
    z_value = (
        estimate / standard_error
        if standard_error is not None and standard_error > 0
        else None
    )
    p_value = (
        float(2 * norm.sf(abs(z_value))) if z_value is not None else None
    )
    delta_array = np.asarray(deltas, dtype=float)
    delta_sd = float(delta_array.std(ddof=1))
    return {
        "name": "controlled_utility_f11_vs_f01",
        "model": "linear mixed-effects regression",
        "estimation": "maximum likelihood",
        "formula": "controlled_action_utility ~ treatment + (1|target_task) + (1|source_run)",
        "baseline_label": "F01_stage_mismatch_valid",
        "treatment_label": "F11_stage_match_valid",
        "observation_count": len(data),
        "paired_decision_count": len(pair_rows),
        "target_task_random_effect_count": int(data["target_task_id"].nunique()),
        "source_run_random_effect_count": int(data["source_run_id"].nunique()),
        "fixed_effect_treatment_mean_difference": estimate,
        "fixed_effect_treatment_standard_error": standard_error,
        "fixed_effect_treatment_wald_interval_95": (
            [
                estimate - 1.959963984540054 * standard_error,
                estimate + 1.959963984540054 * standard_error,
            ]
            if standard_error is not None
            else None
        ),
        "fixed_effect_treatment_z": z_value,
        "fixed_effect_treatment_two_sided_p_value": p_value,
        "paired_standardized_mean_difference_dz": (
            float(delta_array.mean() / delta_sd) if delta_sd else None
        ),
        "target_task_random_intercept_variance": (
            target_variance_raw if math.isfinite(target_variance_raw) else None
        ),
        "source_run_variance_component": (
            source_variance_raw
            if source_variance_raw is not None
            and math.isfinite(source_variance_raw)
            else None
        ),
        "residual_variance": float(result.scale),
        "optimizer_converged": bool(result.converged),
        "warnings": sorted(
            {
                str(item.message)
                for item in (*caught, *post_fit_warnings)
            }
        ),
    }


def _mixed_effects_models(
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    logistic_specs = (
        (
            "gate2_vkr_full_vs_global",
            "vkr_global",
            "vkr_full",
            "global_validity_bit",
            "full_decision_admissibility",
        ),
        (
            "gate3_oracle_action_f11_vs_f01",
            "oracle_f01",
            "oracle_f11",
            "F01_stage_mismatch_valid",
            "F11_stage_match_valid",
        ),
        (
            "gate4_prompt_exposure_post_vs_full",
            "prompt_full",
            "prompt_post",
            "full_pre_prompt_gateway",
            "post_prompt_claim_tags",
        ),
        (
            "gate4_invalid_influence_post_vs_full",
            "influence_full",
            "influence_post",
            "full_pre_prompt_gateway",
            "post_prompt_claim_tags",
        ),
    )
    logistic = {
        name: _fit_binomial_mixed_model(
            pair_rows,
            name=name,
            baseline_key=baseline_key,
            treatment_key=treatment_key,
            baseline_label=baseline_label,
            treatment_label=treatment_label,
        )
        for name, baseline_key, treatment_key, baseline_label, treatment_label in logistic_specs
    }
    return {
        "logistic": logistic,
        "linear": {
            "controlled_utility_f11_vs_f01": _fit_linear_mixed_model(pair_rows)
        },
    }


def _effect_estimates(
    pair_rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(pair_rows)

    def binary_rate(
        metric: str,
        numerator: int,
        denominator: int,
    ) -> dict[str, Any]:
        return {
            "estimand": "rate",
            "numerator": numerator,
            "denominator": denominator,
            "effect_estimate": _ratio(numerator, denominator),
            "wilson_interval_95": _wilson_interval(numerator, denominator),
            "hierarchical_paired_bootstrap_interval_95": bootstrap[metric][
                "percentile_ci_95"
            ],
            "included_pair_count": count,
            "excluded_pair_count": 0,
        }

    raw_invalid_count = int(
        sum(row["metrics"]["primary_raw_cell_iir"] * 2 for row in pair_rows)
    )
    f11_vkr_count = int(
        sum(row["metrics"]["primary_f11_vkr"] for row in pair_rows)
    )
    stage_change_count = int(
        sum(
            row["metrics"]["primary_stage_mismatch_action_change"]
            for row in pair_rows
        )
    )
    action_difference_count = int(
        sum(
            row["metrics"]["gate3_action_difference_f01_vs_f11"]
            for row in pair_rows
        )
    )

    def paired_difference(
        metric: str,
        left_key: str,
        right_key: str,
        *,
        left_label: str,
        right_label: str,
    ) -> dict[str, Any]:
        left = int(sum(row["binary"][left_key] for row in pair_rows))
        right = int(sum(row["binary"][right_key] for row in pair_rows))
        return {
            "estimand": "paired_risk_difference",
            "left_label": left_label,
            "left_numerator": left,
            "left_denominator": count,
            "right_label": right_label,
            "right_numerator": right,
            "right_denominator": count,
            "effect_estimate": bootstrap[metric]["point_estimate"],
            "hierarchical_paired_bootstrap_interval_95": bootstrap[metric][
                "percentile_ci_95"
            ],
            "included_pair_count": count,
            "excluded_pair_count": 0,
        }

    independent_rows = [
        row
        for row in decisions
        if row["condition"] == "NM"
        and row["stage"] == "governance"
        and row["episode_id"].split("::")[1]
        in {"natural-image", "medical-image"}
    ]
    independent_retained = sum(
        row["current_run_node"]["recordable"]
        and row["static_actuation_receipt"]["static_actuation_passed"] is False
        and row["runtime_actuation_receipt"]["runtime_actuation_passed"] is False
        and row["promote_result_path"]["historical_actuation_required"] is False
        for row in independent_rows
    )
    output = {
        "primary_raw_cell_iir": binary_rate(
            "primary_raw_cell_iir", raw_invalid_count, count * 2
        ),
        "primary_f11_vkr": binary_rate(
            "primary_f11_vkr", f11_vkr_count, count
        ),
        "primary_stage_mismatch_action_change": binary_rate(
            "primary_stage_mismatch_action_change", stage_change_count, count
        ),
        "gate2_vkr_full_minus_global": paired_difference(
            "gate2_vkr_full_minus_global",
            "vkr_full",
            "vkr_global",
            left_label="full_decision_admissibility",
            right_label="global_validity_bit",
        ),
        "gate2_iir_full_minus_global": paired_difference(
            "gate2_iir_full_minus_global",
            "iir_full",
            "iir_global",
            left_label="full_decision_admissibility",
            right_label="global_validity_bit",
        ),
        "gate3_oracle_f11_minus_f01": paired_difference(
            "gate3_oracle_f11_minus_f01",
            "oracle_f11",
            "oracle_f01",
            left_label="F11_stage_match_valid",
            right_label="F01_stage_mismatch_valid",
        ),
        "gate3_action_difference_f01_vs_f11": binary_rate(
            "gate3_action_difference_f01_vs_f11",
            action_difference_count,
            count,
        ),
        "gate4_prompt_exposure_post_minus_full": paired_difference(
            "gate4_prompt_exposure_post_minus_full",
            "prompt_post",
            "prompt_full",
            left_label="post_prompt_claim_tags",
            right_label="full_pre_prompt_gateway",
        ),
        "gate4_invalid_influence_post_minus_full": paired_difference(
            "gate4_invalid_influence_post_minus_full",
            "influence_post",
            "influence_full",
            left_label="post_prompt_claim_tags",
            right_label="full_pre_prompt_gateway",
        ),
        "utility_f11_minus_f01": {
            "estimand": "paired_mean_difference",
            "left_label": "F11_stage_match_valid",
            "left_mean": sum(row["utility"]["f11"] for row in pair_rows) / count,
            "right_label": "F01_stage_mismatch_valid",
            "right_mean": sum(row["utility"]["f01"] for row in pair_rows) / count,
            "effect_estimate": bootstrap["utility_f11_minus_f01"][
                "point_estimate"
            ],
            "hierarchical_paired_bootstrap_interval_95": bootstrap[
                "utility_f11_minus_f01"
            ]["percentile_ci_95"],
            "included_pair_count": count,
            "excluded_pair_count": 0,
        },
        "utility_full_minus_global": {
            "estimand": "paired_mean_difference",
            "left_label": "full_decision_admissibility",
            "left_mean": sum(row["utility"]["full"] for row in pair_rows) / count,
            "right_label": "global_validity_bit",
            "right_mean": sum(row["utility"]["global"] for row in pair_rows)
            / count,
            "effect_estimate": bootstrap["utility_full_minus_global"][
                "point_estimate"
            ],
            "hierarchical_paired_bootstrap_interval_95": bootstrap[
                "utility_full_minus_global"
            ]["percentile_ci_95"],
            "included_pair_count": count,
            "excluded_pair_count": 0,
        },
        "independent_result_retention": {
            "estimand": "rate",
            "numerator": independent_retained,
            "denominator": len(independent_rows),
            "effect_estimate": _ratio(independent_retained, len(independent_rows)),
            "wilson_interval_95": _wilson_interval(
                independent_retained, len(independent_rows)
            ),
            "bootstrap_not_reported_reason": "Only six prespecified governance opportunities across two domains; report exact counts and Wilson interval.",
            "excluded_count": 0,
        },
    }
    return output


def compute_statistics(
    packet_root: str | Path,
    generation_root: str | Path,
    evaluation_root: str | Path,
    *,
    created_at: str,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy
    import pandas
    import scipy
    import statsmodels

    packet_root = Path(packet_root).resolve()
    generation_root = Path(generation_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    verification = verify_evaluation(packet_root, generation_root, evaluation_root)
    if not verification["verified"]:
        raise ValueError(
            f"Tier-1 evaluation verification failed: {verification['errors']}"
        )
    packet_manifest = _read_json(packet_root / "manifest.json")
    generation_report = _read_json(generation_root / "run_report.json")
    evaluation_report_path = evaluation_root / "evaluation_report.json"
    evaluation_report = _read_json(evaluation_report_path)
    decision_path = evaluation_root / evaluation_report["decision_receipts_file"]
    counterfactual_path = (
        evaluation_root / evaluation_report["counterfactual_receipts_file"]
    )
    systems_path = evaluation_root / evaluation_report["system_composition_file"]
    decisions = _read_jsonl(decision_path)
    counterfactuals = _read_jsonl(counterfactual_path)
    systems = _read_jsonl(systems_path)
    pair_rows, exclusions = _build_pair_rows(decisions, counterfactuals, systems)
    if exclusions:
        raise ValueError(f"Incomplete Tier-1 pairs cannot enter statistics: {exclusions}")
    bootstrap = _hierarchical_paired_bootstrap(
        pair_rows,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    seed_rows = _paired_seed_deltas(pair_rows)
    seed_text = _jsonl_text(seed_rows)
    effect_estimates = _effect_estimates(
        pair_rows,
        bootstrap,
        decisions,
    )
    exact_tests = _paired_exact_tests(pair_rows)
    models = _mixed_effects_models(pair_rows)
    stored_evaluation_verification_path = evaluation_root / "verification.json"
    stored_verification_hash = (
        _sha256_file(stored_evaluation_verification_path)
        if stored_evaluation_verification_path.exists()
        else None
    )
    report: dict[str, Any] = {
        "schema": STATISTICS_REPORT_SCHEMA,
        "created_at": str(created_at),
        "input_bindings": {
            "packet_manifest_hash": packet_manifest["manifest_hash"],
            "generation_run_hash": generation_report["run_hash"],
            "evaluation_report_hash": evaluation_report["report_hash"],
            "evaluation_report_file_sha256": _sha256_file(evaluation_report_path),
            "decision_receipts_file_sha256": _sha256_file(decision_path),
            "counterfactual_receipts_file_sha256": _sha256_file(
                counterfactual_path
            ),
            "system_composition_file_sha256": _sha256_file(systems_path),
            "evaluation_recomputed_verification_hash": verification[
                "verification_hash"
            ],
            "stored_evaluation_verification_file_sha256": stored_verification_hash,
        },
        "analysis_unit": {
            "hierarchy": "target task -> source run -> decision episode -> paired replicate -> condition/system",
            "paired_decision_count": len(pair_rows),
            "target_task_count": len(
                {row["target_task_id"] for row in pair_rows}
            ),
            "source_task_count": len(
                {row["source_task_id"] for row in pair_rows}
            ),
            "source_run_count": len({row["source_run_id"] for row in pair_rows}),
            "decision_episode_count": len({row["episode_id"] for row in pair_rows}),
            "agent_replicate_ids": sorted(
                {int(row["agent_replicate_id"]) for row in pair_rows}
            ),
            "condition_count": len(CONDITIONS),
            "system_count": len(EXPECTED_SYSTEMS),
        },
        "exclusions": {
            "incomplete_pair_count": len(exclusions),
            "incomplete_pairs": exclusions,
            "decision_receipt_count": len(decisions),
            "counterfactual_receipt_count": len(counterfactuals),
            "system_composition_count": len(systems),
            "post_hoc_episode_exclusion_count": 0,
            "post_hoc_seed_exclusion_count": 0,
        },
        "paired_bootstrap": {
            "method": "hierarchical paired percentile bootstrap",
            "resampling_order": [
                "target_task_id",
                "source_run_id within sampled target task",
                "agent_replicate_id within sampled source run",
            ],
            "conditions_and_systems_kept_paired": True,
            "iterations": int(bootstrap_iterations),
            "host_rng_seed": int(bootstrap_seed),
            "provider_rng_seed_claimed": False,
            "metrics": bootstrap,
        },
        "effect_estimates": effect_estimates,
        "paired_exact_tests_holm_family": {
            "familywise_alpha": 0.05,
            "correction": "Holm step-down",
            "tests": exact_tests,
        },
        "mixed_effects_models": models,
        "paired_seed_deltas_file": "paired_seed_deltas.jsonl",
        "paired_seed_deltas_file_sha256": _sha256_text(seed_text),
        "paired_seed_delta_count": len(seed_rows),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "interpretation_boundaries": [
            "Controlled action utility is host-owned synthetic gold, not a target-task metric.",
            "System comparisons are host routing over frozen response cells, not extra end-to-end model calls.",
            "The 101/202/303 values are paired host replicate IDs; no provider RNG seed is claimed.",
            "Bayesian logistic mixed models use weak regularizing priors because several controlled contrasts exhibit complete separation.",
            "Mixed-model random effects and hierarchical bootstrap preserve task/source-run clustering; individual nodes are not treated as independent runs.",
            "Gate 4 Prompt exposure is decisive; its rare downstream invalid-influence contrast is reported even if Holm-nonsignificant.",
        ],
        "analyzer_source_sha256": _sha256_file(Path(__file__).resolve()),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report, seed_rows


def analyze_statistics(
    packet_root: str | Path,
    generation_root: str | Path,
    evaluation_root: str | Path,
    output_root: str | Path,
    *,
    created_at: str,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse Tier-1 statistics root: {output_root}")
    report, seed_rows = compute_statistics(
        packet_root,
        generation_root,
        evaluation_root,
        created_at=created_at,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    _write_text_exclusive(
        output_root / report["paired_seed_deltas_file"],
        _jsonl_text(seed_rows),
    )
    _write_json_exclusive(output_root / "statistics_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WP8 Tier-1 paired and mixed-effects statistics."
    )
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=DEFAULT_BOOTSTRAP_ITERATIONS,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    report = analyze_statistics(
        args.packet_root,
        args.generation_root,
        args.evaluation_root,
        args.output_root,
        created_at=args.created_at,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
