from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUILD = (
    ROOT
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "build_strategy_shadow_history_slices.py"
)
RUNNER = (
    ROOT
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "run_strategy_shadow_replay.py"
)
SMOKE_JOB = (
    ROOT
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "jobs"
    / "mlevolve-e2e-memory-strategy-shadow-smoke-v64.yaml"
)
SMOKE_JOB_V65 = SMOKE_JOB.with_name(
    "mlevolve-e2e-memory-strategy-shadow-smoke-v65.yaml"
)
SMOKE_JOB_V66 = SMOKE_JOB.with_name(
    "mlevolve-e2e-memory-strategy-shadow-smoke-v66.yaml"
)
SMOKE_JOB_V67 = SMOKE_JOB.with_name(
    "mlevolve-e2e-memory-strategy-shadow-smoke-v67.yaml"
)
SMOKE_JOB_V68 = SMOKE_JOB.with_name(
    "mlevolve-e2e-memory-strategy-shadow-smoke-v68.yaml"
)
SMOKE_JOB_V69 = SMOKE_JOB.with_name(
    "mlevolve-e2e-memory-strategy-shadow-smoke-v69.yaml"
)
SMOKE_JOB_V70 = SMOKE_JOB.with_name(
    "mlevolve-e2e-memory-strategy-shadow-smoke-v70.yaml"
)
SMOKE_JOB_V71 = SMOKE_JOB.with_name(
    "mlevolve-e2e-memory-strategy-shadow-smoke-v71.yaml"
)


def _runner_module():
    spec = importlib.util.spec_from_file_location("strategy_shadow_replay", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_history_slice_builder_hides_future_ids_for_all_three_tasks(tmp_path):
    output = tmp_path / "cases.json"
    subprocess.run(
        [sys.executable, str(BUILD), "--output", str(output)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    packet = json.loads(output.read_text(encoding="utf-8"))

    assert [case["task_id"] for case in packet["cases"]] == [
        "spooky-author-identification",
        "spooky-author-identification",
        "leaf-classification",
        "new-york-city-taxi-fare-prediction",
    ]
    runner = _runner_module()
    for case in packet["cases"]:
        _pack, visible_ids = runner._build_router_pack(case)
        hidden_ids = set(case["hidden_future"]["memory_ids"])
        assert hidden_ids.isdisjoint(visible_ids)
        assert case["hidden_future"]["evaluator_only"] is True
        assert case["parent"]["code"]

    from agents.memory_strategy_agent import (
        build_component_portfolio,
        build_memory_cards,
    )

    spooky = packet["cases"][0]
    spooky_pack, _ = runner._build_router_pack(spooky)
    portfolio = build_component_portfolio(build_memory_cards(spooky_pack))
    axes = portfolio["component_axes"]
    assert {"modernbert", "deberta", "distilbert"} <= set(
        axes["representation"]
    )
    assert "frozen_embedding" in axes["adaptation_mode"]
    assert "xgboost" in axes["downstream_estimator"]
    assert "five_fold" in axes["validation"]
    representation_opportunity = next(
        item
        for item in portfolio["within_axis_diversity_opportunities"]
        if item["axis"] == "representation"
    )
    assert {"modernbert", "deberta", "distilbert"} <= set(
        representation_opportunity["alternatives"]
    )
    assert {
        "adaptation_mode:frozen_embedding",
        "downstream_estimator:xgboost",
        "feature_family:embedding",
    } <= set(representation_opportunity["common_interfaces"])

    repair = packet["cases"][1]
    assert repair["stage"] == "debug"
    assert repair["parent"]["is_buggy"] is True
    assert repair["parent"]["node_id"] == "13675db86b424a579d66e82bf2515d74"
    assert repair["hidden_future"]["id"].endswith(
        "8c5286bb1da2400486bafdd5878e30f9"
    )
    assert repair["hidden_future"]["id"] not in {
        event["candidate_id"] for event in repair["memory_events"]
    }


def test_replay_evaluator_measures_hit_budget_duplicate_and_incompatibility():
    runner = _runner_module()
    case = {
        "case_id": "fixture",
        "budget": {"remaining_search_seconds": 100},
        "expected_future_pattern_groups": [["three model"], ["five fold"]],
        "attempted_pattern_signatures": [[["single model"], ["holdout"]]],
        "known_incompatibilities": [
            {
                "pattern_groups": [["three model"], ["fine tune"]],
                "resolution_pattern": "frozen",
                "reason": "too expensive",
            }
        ],
    }
    memo = {
        "candidate_compositions": [
            {
                "hypothesis_id": "future-hit",
                "hypothesis": "three model five fold fine tune",
                "source_memory_ids": ["m1", "m2"],
                "compatibility_checks": ["class order is stable"],
                "estimated_compute_seconds": 200,
            },
            {
                "hypothesis_id": "duplicate",
                "hypothesis": "single model holdout",
                "source_memory_ids": ["m1"],
                "compatibility_checks": [],
                "estimated_compute_seconds": 50,
            },
        ]
    }

    result = runner.evaluate_memo(
        case,
        memo=memo,
        visible_memory_ids=["m1", "m2"],
    )

    assert result["future_strategy_hit_ids"] == ["future-hit"]
    assert result["future_strategy_structural_hit"] is True
    assert result["future_strategy_exact_hit"] is False
    assert result["over_budget_ids"] == ["future-hit"]
    assert result["duplicate_ids"] == ["duplicate"]
    assert result["invalid_combinations"] == [
        {"hypothesis_id": "future-hit", "rule": "too expensive"}
    ]
    assert result["all_citations_visible"] is True


def test_replay_evaluator_distinguishes_structural_from_exact_future_match():
    runner = _runner_module()
    case = {
        "case_id": "exact-vs-structural",
        "expected_structural_pattern_groups": [
            ["three transformer"],
            ["five fold"],
        ],
        "expected_exact_future_pattern_groups": [
            ["deberta"],
            ["roberta"],
            ["distilbert"],
        ],
    }
    memo = {
        "candidate_compositions": [
            {
                "hypothesis_id": "wrong-three",
                "hypothesis": "three transformer five fold: ModernBERT, DeBERTa, DistilBERT",
                "source_memory_ids": ["m1"],
            },
            {
                "hypothesis_id": "exact-three",
                "hypothesis": "three transformer five fold: DeBERTa, RoBERTa, DistilBERT",
                "source_memory_ids": ["m1"],
            },
        ]
    }

    result = runner.evaluate_memo(case, memo=memo, visible_memory_ids=["m1"])

    assert result["future_strategy_structural_hit_ids"] == [
        "wrong-three",
        "exact-three",
    ]
    assert result["future_strategy_exact_hit_ids"] == ["exact-three"]
    assert result["future_strategy_hit_ids"] == ["exact-three"]


def test_run_case_allows_strategy_current_frontier_citations(monkeypatch):
    runner = _runner_module()
    current_id = "current::branch-best"
    memo = {
        "candidate_compositions": [
            {
                "hypothesis_id": "use-current",
                "hypothesis": "retain the current branch best",
                "source_memory_ids": [current_id],
            }
        ]
    }
    monkeypatch.setattr(
        runner,
        "run_memory_strategy_shadow",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "memory_card_ids": [current_id, "history::one"],
            "memo": memo,
        },
    )
    case = {
        "case_id": "current-citation",
        "task_id": "spooky-author-identification",
        "task_description": "authorship",
        "data_preview": "text",
        "stage": "improve",
        "parent": {
            "node_id": "branch-best",
            "code": "def train():\n    return 1\n",
            "plan": "baseline",
            "metric": 0.3,
            "maximize": False,
            "stage": "draft",
            "is_buggy": False,
        },
        "memory_events": [
            {
                "candidate_id": "history::one",
                "source_task_id": "spooky-author-identification",
                "metric": 0.4,
            }
        ],
        "cutoff": {"order": 1},
        "hidden_future": {
            "id": "future::hidden",
            "memory_ids": ["future::hidden"],
        },
    }

    result = runner.run_case(
        case,
        config_path=ROOT / "mlevolve" / "config" / "config.yaml",
        actuate=False,
    )

    assert result["evaluation"]["all_citations_visible"] is True
    assert result["strategy_visible_memory_ids"] == [current_id, "history::one"]
    assert result["visible_memory_ids"] == ["history::one"]


def test_strategy_smoke_job_is_owned_cpu_only_and_fails_closed():
    for path in (
        SMOKE_JOB,
        SMOKE_JOB_V65,
        SMOKE_JOB_V66,
        SMOKE_JOB_V67,
        SMOKE_JOB_V68,
        SMOKE_JOB_V69,
        SMOKE_JOB_V70,
        SMOKE_JOB_V71,
    ):
        job = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = job["metadata"]["labels"]
        pod_labels = job["spec"]["template"]["metadata"]["labels"]
        container = job["spec"]["template"]["spec"]["containers"][0]
        command = "\n".join(container["command"])

        assert labels["ecepxie.nrp/owner"] == "haoming"
        assert pod_labels["ecepxie.nrp/owner"] == "haoming"
        assert not any("nvidia.com/" in key for key in container["resources"]["limits"])
        assert (
            "future_strategy_hit" in command
            or "future_strategy_structural_hit" in command
            or "verify_strategy_shadow_smoke_v70.py" in command
            or "verify_strategy_shadow_smoke_v71.py" in command
        )
        assert "atomic_actuation" in command or "--actuate" in command
        assert "sleep" not in command
        assert job["spec"]["backoffLimit"] == 0
        if path in {
            SMOKE_JOB_V67,
            SMOKE_JOB_V68,
            SMOKE_JOB_V69,
            SMOKE_JOB_V70,
            SMOKE_JOB_V71,
        }:
            assert (
                "strategy_model_is_v4_pro" in command
                or '"model_is_v4_pro"' in command
                or "verify_strategy_shadow_smoke_v70.py" in command
                or "verify_strategy_shadow_smoke_v71.py" in command
            )
            assert (
                "strategy_thinking_enabled" in command
                or '"thinking_enabled"' in command
                or "verify_strategy_shadow_smoke_v70.py" in command
                or "verify_strategy_shadow_smoke_v71.py" in command
            )
        if path == SMOKE_JOB_V68:
            assert "all_diversity_opportunities_addressed" in command
        if path == SMOKE_JOB_V69:
            assert "future_strategy_structural_hit" in command
            assert "future_strategy_exact_hit" in command
            assert "spooky-after-multibackbone-before-cleanup-repair" in command
        if path in {SMOKE_JOB_V70, SMOKE_JOB_V71}:
            assert "memory_strategy_shadow_history_slices_v3.json" in command
        if path == SMOKE_JOB_V71:
            assert "verify_strategy_shadow_smoke_v71.py" in command
