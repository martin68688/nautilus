from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from analyze_tier1_controlled_statistics import (  # noqa: E402
    _holm_adjust,
    analyze_statistics,
)
from evaluate_tier1_controlled_decisions import evaluate  # noqa: E402
from tests.test_tier1_controlled_evaluator import (  # noqa: E402
    CREATED_AT,
    _run_fixture_generation,
)
from verify_tier1_controlled_statistics import verify_statistics  # noqa: E402


def test_holm_adjustment_is_step_down_and_name_stable() -> None:
    adjusted = _holm_adjust({"z": 0.04, "a": 0.001, "m": 0.02})

    assert list(adjusted) == ["a", "m", "z"]
    assert adjusted == {"a": 0.003, "m": 0.04, "z": 0.04}


def test_statistics_preserve_pairs_cluster_units_and_recompute(
    tmp_path: Path,
) -> None:
    packet, legacy, generation, _episodes = _run_fixture_generation(
        tmp_path,
        full=True,
    )
    evaluation = tmp_path / "evaluation"
    evaluate(
        packet,
        generation,
        evaluation,
        created_at=CREATED_AT,
        require_full_matrix=True,
        legacy_episode_root=legacy,
    )
    statistics = tmp_path / "statistics"
    report = analyze_statistics(
        packet,
        generation,
        evaluation,
        statistics,
        created_at=CREATED_AT,
        bootstrap_iterations=100,
        bootstrap_seed=7,
    )

    assert report["analysis_unit"]["paired_decision_count"] == 72
    assert report["analysis_unit"]["target_task_count"] == 6
    assert report["analysis_unit"]["source_run_count"] == 24
    assert report["analysis_unit"]["agent_replicate_ids"] == [101, 202, 303]
    assert report["exclusions"]["incomplete_pair_count"] == 0
    assert report["exclusions"]["post_hoc_episode_exclusion_count"] == 0
    assert report["paired_bootstrap"]["conditions_and_systems_kept_paired"] is True
    assert report["paired_bootstrap"]["iterations"] == 100
    assert report["effect_estimates"]["primary_raw_cell_iir"]["numerator"] == 144
    assert report["effect_estimates"]["primary_raw_cell_iir"]["denominator"] == 144
    assert report["effect_estimates"]["primary_f11_vkr"]["numerator"] == 72
    assert report["effect_estimates"]["primary_f11_vkr"]["denominator"] == 72
    assert report["effect_estimates"]["independent_result_retention"][
        "numerator"
    ] == 6
    assert report["effect_estimates"]["independent_result_retention"][
        "denominator"
    ] == 6
    assert report["paired_seed_delta_count"] == 33
    assert all(
        model["target_task_random_effect_count"] == 6
        and model["source_run_random_effect_count"] == 24
        and model["optimizer_success"] is True
        for model in report["mixed_effects_models"]["logistic"].values()
    )
    assert report["mixed_effects_models"]["linear"][
        "controlled_utility_f11_vs_f01"
    ]["optimizer_converged"] is True
    assert stat.S_IMODE((statistics / "statistics_report.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((statistics / "paired_seed_deltas.jsonl").stat().st_mode) == 0o444

    verification = verify_statistics(packet, generation, evaluation, statistics)
    assert verification["verified"] is True
    assert verification["errors"] == []
    assert verification["paired_decision_count"] == 72
    assert verification["paired_seed_delta_count"] == 33

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        analyze_statistics(
            packet,
            generation,
            evaluation,
            statistics,
            created_at=CREATED_AT,
            bootstrap_iterations=10,
            bootstrap_seed=7,
        )
