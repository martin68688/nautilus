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

from build_multigeneration_contamination_packet import GATE_5_THRESHOLDS, SYSTEMS
from schema import sha256_json
from verify_multigeneration_contamination_evaluation import verify_evaluation


STATISTICS_REPORT_SCHEMA = "decision_admissibility_multigeneration_statistics_report_v1"
SEED_DELTA_SCHEMA = "decision_admissibility_multigeneration_seed_delta_v1"
DEFAULT_BOOTSTRAP_ITERATIONS = 20_000
DEFAULT_BOOTSTRAP_SEED = 20260721


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


def _ratio(numerator: int | float, denominator: int) -> float | None:
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


def _build_chain_rows(
    pairs: Sequence[Mapping[str, Any]],
    system_receipts: Sequence[Mapping[str, Any]],
    *,
    generation_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    grouped: dict[tuple[str, int], dict[int, dict[str, Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in system_receipts:
        key = (str(row["pair_id"]), int(row["paraphrase_replicate_id"]))
        generation = int(row["generation"])
        system = str(row["system"])
        if system in grouped[key][generation]:
            raise ValueError(f"Duplicate system row: {key}:{generation}:{system}")
        grouped[key][generation][system] = row
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for (pair_id, replicate), generations in sorted(grouped.items()):
        reasons: list[str] = []
        for generation in range(1, generation_count + 1):
            missing = sorted(set(SYSTEMS) - set(generations.get(generation, {})))
            if missing:
                reasons.append(f"generation_{generation}_missing:{','.join(missing)}")
        if reasons:
            exclusions.append(
                {
                    "pair_id": pair_id,
                    "paraphrase_replicate_id": replicate,
                    "reasons": reasons,
                }
            )
            continue
        pair = pairs_by_id[pair_id]
        metrics: dict[str, float] = {}
        outcomes: dict[str, int] = {}
        for generation in range(1, generation_count + 1):
            systems = generations[generation]
            for system in SYSTEMS:
                laundering = int(bool(systems[system]["laundering_success"]))
                retained = int(bool(systems[system]["valid_knowledge_retained"]))
                metrics[f"{system}:g{generation}:laundering"] = float(laundering)
                metrics[f"{system}:g{generation}:vkr"] = float(retained)
                outcomes[f"{system}:g{generation}:laundering"] = laundering
                outcomes[f"{system}:g{generation}:vkr"] = retained
        final = generation_count
        metrics["final_laundering_reduction_unrestricted_minus_full"] = (
            metrics[f"unrestricted:g{final}:laundering"]
            - metrics[f"full_decision_admissibility:g{final}:laundering"]
        )
        metrics["final_laundering_reduction_authority_minus_full"] = (
            metrics[f"authority_only:g{final}:laundering"]
            - metrics[f"full_decision_admissibility:g{final}:laundering"]
        )
        metrics["final_vkr_delta_full_minus_global"] = (
            metrics[f"full_decision_admissibility:g{final}:vkr"]
            - metrics[f"global_validity_bit:g{final}:vkr"]
        )
        metrics["final_vkr_delta_full_minus_authority"] = (
            metrics[f"full_decision_admissibility:g{final}:vkr"]
            - metrics[f"authority_only:g{final}:vkr"]
        )
        rows.append(
            {
                "pair_id": pair_id,
                "paraphrase_replicate_id": replicate,
                "source_run_id": pair["source_run_id"],
                "source_task_id": pair["source_task_id"],
                "target_task_id": pair["target_task_id"],
                "domain": pair["domain"],
                "metrics": metrics,
                "outcomes": outcomes,
            }
        )
    return rows, exclusions


def _hierarchical_paired_bootstrap(
    chain_rows: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    if iterations <= 0:
        raise ValueError("Bootstrap iterations must be positive")
    metric_names = sorted(chain_rows[0]["metrics"])
    values = np.asarray(
        [
            [float(row["metrics"][metric]) for metric in metric_names]
            for row in chain_rows
        ],
        dtype=float,
    )
    by_run_pair: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, row in enumerate(chain_rows):
        by_run_pair[str(row["source_run_id"])][str(row["pair_id"])].append(index)
    runs = sorted(by_run_pair)
    rng = np.random.default_rng(seed)
    draws = np.empty((iterations, len(metric_names)), dtype=float)
    for iteration in range(iterations):
        sampled_indices: list[int] = []
        for _ in range(len(runs)):
            run_id = runs[int(rng.integers(0, len(runs)))]
            pairs = sorted(by_run_pair[run_id])
            for _ in range(len(pairs)):
                pair_id = pairs[int(rng.integers(0, len(pairs)))]
                replicates = by_run_pair[run_id][pair_id]
                for _ in range(len(replicates)):
                    sampled_indices.append(
                        replicates[int(rng.integers(0, len(replicates)))]
                    )
        draws[iteration, :] = values[sampled_indices, :].mean(axis=0)
    return {
        metric: {
            "point_estimate": float(values[:, index].mean()),
            "percentile_ci_95": [
                float(np.quantile(draws[:, index], 0.025)),
                float(np.quantile(draws[:, index], 0.975)),
            ],
        }
        for index, metric in enumerate(metric_names)
    }


def _paired_exact_tests(
    chain_rows: Sequence[Mapping[str, Any]],
    *,
    final_generation: int,
) -> dict[str, Any]:
    from scipy.stats import binomtest

    specs = {
        "unrestricted_laundering_gt_full": (
            f"unrestricted:g{final_generation}:laundering",
            f"full_decision_admissibility:g{final_generation}:laundering",
        ),
        "authority_laundering_gt_full": (
            f"authority_only:g{final_generation}:laundering",
            f"full_decision_admissibility:g{final_generation}:laundering",
        ),
        "full_vkr_gt_global": (
            f"full_decision_admissibility:g{final_generation}:vkr",
            f"global_validity_bit:g{final_generation}:vkr",
        ),
        "full_vkr_gt_authority": (
            f"full_decision_admissibility:g{final_generation}:vkr",
            f"authority_only:g{final_generation}:vkr",
        ),
    }
    tests: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for name, (left, right) in specs.items():
        positive = sum(
            row["outcomes"][left] == 1 and row["outcomes"][right] == 0
            for row in chain_rows
        )
        negative = sum(
            row["outcomes"][left] == 0 and row["outcomes"][right] == 1
            for row in chain_rows
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
        raw_p[name] = p_value
        tests[name] = {
            "method": "exact paired discordance/binomial test",
            "alternative": "left_greater_than_right",
            "paired_chain_count": len(chain_rows),
            "positive_discordant_count": positive,
            "negative_discordant_count": negative,
            "tie_count": len(chain_rows) - discordant,
            "raw_p_value": p_value,
        }
    adjusted = _holm_adjust(raw_p)
    for name, value in adjusted.items():
        tests[name]["holm_adjusted_p_value"] = value
        tests[name]["reject_at_familywise_alpha_0_05"] = value < 0.05
    return tests


def _paired_seed_deltas(
    chain_rows: Sequence[Mapping[str, Any]],
    *,
    final_generation: int,
) -> list[dict[str, Any]]:
    metrics = (
        "final_laundering_reduction_unrestricted_minus_full",
        "final_laundering_reduction_authority_minus_full",
        "final_vkr_delta_full_minus_global",
        "final_vkr_delta_full_minus_authority",
        f"full_decision_admissibility:g{final_generation}:laundering",
        f"full_decision_admissibility:g{final_generation}:vkr",
    )
    rows: list[dict[str, Any]] = []
    replicates = sorted(
        {int(row["paraphrase_replicate_id"]) for row in chain_rows}
    )
    for metric in metrics:
        for replicate in replicates:
            values = [
                float(row["metrics"][metric])
                for row in chain_rows
                if int(row["paraphrase_replicate_id"]) == replicate
            ]
            payload: dict[str, Any] = {
                "schema": SEED_DELTA_SCHEMA,
                "metric": metric,
                "paraphrase_replicate_id": replicate,
                "paired_source_pair_count": len(values),
                "sum_of_pair_values": sum(values),
                "mean_effect": _ratio(sum(values), len(values)),
                "row_hash": "",
            }
            payload["row_hash"] = sha256_json(
                {key: value for key, value in payload.items() if key != "row_hash"}
            )
            rows.append(payload)
    return rows


def _fit_binomial_mixed_model(
    chain_rows: Sequence[Mapping[str, Any]],
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
    for row in chain_rows:
        common = {
            "source_task_id": row["source_task_id"],
            "source_run_id": row["source_run_id"],
        }
        observations.append(
            {"outcome": row["outcomes"][baseline_key], "treatment": 0, **common}
        )
        observations.append(
            {"outcome": row["outcomes"][treatment_key], "treatment": 1, **common}
        )
    data = pd.DataFrame(observations)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = BinomialBayesMixedGLM.from_formula(
            "outcome ~ treatment",
            {
                "source_task": "0 + C(source_task_id)",
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
    index = model.exog_names.index("treatment")
    estimate = float(result.fe_mean[index])
    posterior_sd = float(result.fe_sd[index])
    lower = estimate - 1.959963984540054 * posterior_sd
    upper = estimate + 1.959963984540054 * posterior_sd
    optimization = dict(result.optim_retvals)
    baseline = data.loc[data["treatment"] == 0, "outcome"].to_numpy()
    treatment = data.loc[data["treatment"] == 1, "outcome"].to_numpy()
    return {
        "name": name,
        "model": "Bayesian binomial mixed-effects logistic regression",
        "estimation": "mean-field variational Bayes",
        "formula": "outcome ~ treatment + (1|source_task) + (1|source_run)",
        "baseline_label": baseline_label,
        "treatment_label": treatment_label,
        "observation_count": len(data),
        "paired_chain_count": len(chain_rows),
        "source_task_random_effect_count": int(data["source_task_id"].nunique()),
        "source_run_random_effect_count": int(data["source_run_id"].nunique()),
        "baseline_numerator": int(baseline.sum()),
        "baseline_denominator": int(len(baseline)),
        "treatment_numerator": int(treatment.sum()),
        "treatment_denominator": int(len(treatment)),
        "treatment_log_odds": estimate,
        "treatment_posterior_sd": posterior_sd,
        "treatment_normal_approx_credible_interval_95": [lower, upper],
        "odds_ratio": math.exp(estimate),
        "odds_ratio_interval_95": [math.exp(lower), math.exp(upper)],
        "complete_separation_present": bool(
            len(np.unique(baseline)) == 1 or len(np.unique(treatment)) == 1
        ),
        "prior_stabilized": True,
        "deterministic_initial_mean": 0.0,
        "deterministic_initial_sd": math.exp(-0.5),
        "optimizer_success": bool(optimization.get("success", False)),
        "optimizer_message": str(optimization.get("message", "")),
        "optimizer_iterations": int(optimization.get("nit", 0)),
        "warnings": sorted({str(item.message) for item in caught}),
    }


def _mixed_models(
    chain_rows: Sequence[Mapping[str, Any]],
    *,
    final_generation: int,
) -> dict[str, Any]:
    specs = (
        (
            "final_laundering_unrestricted_vs_full",
            f"full_decision_admissibility:g{final_generation}:laundering",
            f"unrestricted:g{final_generation}:laundering",
            "full_decision_admissibility",
            "unrestricted",
        ),
        (
            "final_laundering_authority_vs_full",
            f"full_decision_admissibility:g{final_generation}:laundering",
            f"authority_only:g{final_generation}:laundering",
            "full_decision_admissibility",
            "authority_only",
        ),
        (
            "final_vkr_full_vs_global",
            f"global_validity_bit:g{final_generation}:vkr",
            f"full_decision_admissibility:g{final_generation}:vkr",
            "global_validity_bit",
            "full_decision_admissibility",
        ),
        (
            "final_vkr_full_vs_authority",
            f"authority_only:g{final_generation}:vkr",
            f"full_decision_admissibility:g{final_generation}:vkr",
            "authority_only",
            "full_decision_admissibility",
        ),
    )
    return {
        name: _fit_binomial_mixed_model(
            chain_rows,
            name=name,
            baseline_key=baseline_key,
            treatment_key=treatment_key,
            baseline_label=baseline_label,
            treatment_label=treatment_label,
        )
        for name, baseline_key, treatment_key, baseline_label, treatment_label in specs
    }


def compute_statistics(
    work_root: str | Path,
    packet_root: str | Path,
    run_root: str | Path,
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
    run_root = Path(run_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    verification = verify_evaluation(
        work_root, packet_root, run_root, evaluation_root
    )
    if not verification["verified"]:
        raise ValueError(
            f"Multi-generation evaluation verification failed: {verification['errors']}"
        )
    manifest = _read_json(packet_root / "manifest.json")
    pairs = _read_jsonl(packet_root / str(manifest["pair_file"]))
    run_report = _read_json(run_root / "run_report.json")
    evaluation_report_path = evaluation_root / "evaluation_report.json"
    evaluation_report = _read_json(evaluation_report_path)
    system_path = evaluation_root / evaluation_report["system_receipts_file"]
    system_receipts = _read_jsonl(system_path)
    generation_count = int(manifest["generation_count"])
    chain_rows, exclusions = _build_chain_rows(
        pairs,
        system_receipts,
        generation_count=generation_count,
    )
    if exclusions:
        raise ValueError(f"Incomplete Multi-generation chains: {exclusions}")
    bootstrap = _hierarchical_paired_bootstrap(
        chain_rows,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    exact_tests = _paired_exact_tests(
        chain_rows,
        final_generation=generation_count,
    )
    seed_rows = _paired_seed_deltas(
        chain_rows,
        final_generation=generation_count,
    )
    mixed_models = _mixed_models(
        chain_rows,
        final_generation=generation_count,
    )
    final_key = str(generation_count)
    summaries = evaluation_report["system_summaries_by_generation"]
    final_effects = {
        "full_laundering": {
            "numerator": summaries["full_decision_admissibility"][final_key][
                "laundering_success_count"
            ],
            "denominator": summaries["full_decision_admissibility"][final_key][
                "decision_count"
            ],
            "rate": summaries["full_decision_admissibility"][final_key][
                "laundering_success_rate"
            ],
            "wilson_interval_95": _wilson_interval(
                summaries["full_decision_admissibility"][final_key][
                    "laundering_success_count"
                ],
                summaries["full_decision_admissibility"][final_key][
                    "decision_count"
                ],
            ),
            "hierarchical_bootstrap_interval_95": bootstrap[
                f"full_decision_admissibility:g{generation_count}:laundering"
            ]["percentile_ci_95"],
        },
        "unrestricted_laundering": {
            "numerator": summaries["unrestricted"][final_key][
                "laundering_success_count"
            ],
            "denominator": summaries["unrestricted"][final_key]["decision_count"],
            "rate": summaries["unrestricted"][final_key][
                "laundering_success_rate"
            ],
        },
        "authority_only_laundering": {
            "numerator": summaries["authority_only"][final_key][
                "laundering_success_count"
            ],
            "denominator": summaries["authority_only"][final_key][
                "decision_count"
            ],
            "rate": summaries["authority_only"][final_key][
                "laundering_success_rate"
            ],
        },
        "full_vkr": {
            "numerator": summaries["full_decision_admissibility"][final_key][
                "valid_knowledge_retained_count"
            ],
            "denominator": summaries["full_decision_admissibility"][final_key][
                "decision_count"
            ],
            "rate": summaries["full_decision_admissibility"][final_key][
                "valid_knowledge_retention"
            ],
            "wilson_interval_95": _wilson_interval(
                summaries["full_decision_admissibility"][final_key][
                    "valid_knowledge_retained_count"
                ],
                summaries["full_decision_admissibility"][final_key][
                    "decision_count"
                ],
            ),
            "hierarchical_bootstrap_interval_95": bootstrap[
                f"full_decision_admissibility:g{generation_count}:vkr"
            ]["percentile_ci_95"],
        },
        "full_vs_unrestricted_laundering_reduction": bootstrap[
            "final_laundering_reduction_unrestricted_minus_full"
        ],
        "full_vs_authority_laundering_reduction": bootstrap[
            "final_laundering_reduction_authority_minus_full"
        ],
        "full_vs_global_vkr_delta": bootstrap[
            "final_vkr_delta_full_minus_global"
        ],
        "full_vs_authority_vkr_delta": bootstrap[
            "final_vkr_delta_full_minus_authority"
        ],
    }
    thresholds = manifest["gate_5_thresholds"]
    gate_checks = {
        "source_pair_floor": len(pairs)
        >= int(thresholds["minimum_source_pair_count"]),
        "source_run_floor": manifest["source_run_count"]
        >= int(thresholds["minimum_source_run_count"]),
        "generation_floor": generation_count
        >= int(thresholds["minimum_generation_count"]),
        "replicate_floor": len(manifest["paraphrase_replicate_ids"])
        >= int(thresholds["minimum_paraphrase_replicates"]),
        "full_final_laundering_ceiling": final_effects["full_laundering"]["rate"]
        <= float(thresholds["maximum_full_final_laundering_rate"]),
        "unrestricted_attack_prevalence": final_effects[
            "unrestricted_laundering"
        ]["rate"]
        >= float(thresholds["minimum_unrestricted_final_laundering_rate"]),
        "full_vs_unrestricted_reduction_ci": final_effects[
            "full_vs_unrestricted_laundering_reduction"
        ]["percentile_ci_95"][0]
        >= float(thresholds["minimum_full_vs_unrestricted_reduction_ci_lower"]),
        "full_final_vkr_floor": final_effects["full_vkr"]["rate"]
        >= float(thresholds["minimum_full_final_vkr"]),
        "full_final_vkr_ci_floor": final_effects["full_vkr"][
            "hierarchical_bootstrap_interval_95"
        ][0]
        >= float(thresholds["minimum_full_final_vkr_ci_lower"]),
        "full_vs_global_vkr_delta_ci": final_effects[
            "full_vs_global_vkr_delta"
        ]["percentile_ci_95"][0]
        >= float(thresholds["minimum_full_vs_global_vkr_delta_ci_lower"]),
        "holm_primary_contrasts": all(
            exact_tests[name]["holm_adjusted_p_value"]
            < float(thresholds["familywise_alpha"])
            for name in (
                "unrestricted_laundering_gt_full",
                "authority_laundering_gt_full",
                "full_vkr_gt_global",
            )
        ),
        "no_post_hoc_exclusions": evaluation_report[
            "post_hoc_descendant_exclusion_count"
        ]
        == 0,
        "no_target_history_exposure": evaluation_report[
            "target_history_exposure_count"
        ]
        == 0,
        "no_cross_domain_transfer": evaluation_report["cross_domain_transfer_count"]
        == 0,
    }
    seed_text = _jsonl_text(seed_rows)
    report: dict[str, Any] = {
        "schema": STATISTICS_REPORT_SCHEMA,
        "created_at": str(created_at),
        "input_bindings": {
            "packet_manifest_hash": manifest["manifest_hash"],
            "run_hash": run_report["run_hash"],
            "evaluation_report_hash": evaluation_report["report_hash"],
            "evaluation_report_file_sha256": _sha256_file(evaluation_report_path),
            "system_receipts_file_sha256": _sha256_file(system_path),
            "evaluation_verification_hash": verification["verification_hash"],
            "evaluation_verification_file_sha256": _sha256_file(
                evaluation_root / "verification.json"
            ),
        },
        "analysis_unit": {
            "hierarchy": "source run -> source experience pair -> paraphrase replicate -> generation -> system",
            "source_pair_count": len(pairs),
            "source_run_count": manifest["source_run_count"],
            "source_task_count": manifest["source_task_count"],
            "paired_chain_count": len(chain_rows),
            "generation_count": generation_count,
            "paraphrase_replicate_ids": manifest["paraphrase_replicate_ids"],
            "system_count": len(SYSTEMS),
        },
        "exclusions": {
            "incomplete_chain_count": len(exclusions),
            "incomplete_chains": exclusions,
            "post_hoc_descendant_exclusion_count": evaluation_report[
                "post_hoc_descendant_exclusion_count"
            ],
        },
        "paired_bootstrap": {
            "method": "hierarchical paired percentile bootstrap",
            "resampling_order": [
                "source_run_id",
                "pair_id within sampled source run",
                "paraphrase_replicate_id within sampled pair",
            ],
            "systems_and_generations_kept_paired": True,
            "iterations": int(bootstrap_iterations),
            "host_rng_seed": int(bootstrap_seed),
            "provider_rng_seed_claimed": False,
            "metrics": bootstrap,
        },
        "final_generation_effects": final_effects,
        "trajectory": evaluation_report["system_summaries_by_generation"],
        "paired_exact_tests_holm_family": {
            "familywise_alpha": float(thresholds["familywise_alpha"]),
            "correction": "Holm step-down",
            "tests": exact_tests,
        },
        "mixed_effects_logistic_models": mixed_models,
        "paired_seed_deltas_file": "paired_seed_deltas.jsonl",
        "paired_seed_deltas_file_sha256": _sha256_text(seed_text),
        "paired_seed_delta_count": len(seed_rows),
        "gate_5": {
            "name": "multi_generation",
            "thresholds": thresholds,
            "thresholds_fixed_before_generation": manifest[
                "thresholds_fixed_before_generation"
            ],
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
            "status": "pass" if all(gate_checks.values()) else "fail",
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "interpretation_boundaries": [
            "Lineage-only and Full are tied on this pure non-escalation benchmark; no superiority over lineage-only is claimed.",
            "Authority-only uses a frozen surface-pattern ablation and is not a universal semantic detector.",
            "The descendant DAG is shared across systems; system outcomes are host policy decisions, not separate LLM generations.",
            "Paraphrase replicate IDs are host chain/style identifiers, not provider RNG seeds.",
        ],
        "analyzer_source_sha256": _sha256_file(Path(__file__).resolve()),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report, seed_rows


def analyze(
    work_root: str | Path,
    packet_root: str | Path,
    run_root: str | Path,
    evaluation_root: str | Path,
    output_root: str | Path,
    *,
    created_at: str,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to reuse Multi-generation statistics root: {output_root}"
        )
    report, seed_rows = compute_statistics(
        work_root,
        packet_root,
        run_root,
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
        description="Analyze WP8 Multi-generation Gate-5 statistics."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
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
    report = analyze(
        args.work_root,
        args.packet_root,
        args.run_root,
        args.evaluation_root,
        args.output_root,
        created_at=args.created_at,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["gate_5"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
