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
    assert result["over_budget_ids"] == ["future-hit"]
    assert result["duplicate_ids"] == ["duplicate"]
    assert result["invalid_combinations"] == [
        {"hypothesis_id": "future-hit", "rule": "too expensive"}
    ]
    assert result["all_citations_visible"] is True


def test_strategy_smoke_job_is_owned_cpu_only_and_fails_closed():
    for path in (SMOKE_JOB, SMOKE_JOB_V65):
        job = yaml.safe_load(path.read_text(encoding="utf-8"))
        labels = job["metadata"]["labels"]
        pod_labels = job["spec"]["template"]["metadata"]["labels"]
        container = job["spec"]["template"]["spec"]["containers"][0]
        command = "\n".join(container["command"])

        assert labels["ecepxie.nrp/owner"] == "haoming"
        assert pod_labels["ecepxie.nrp/owner"] == "haoming"
        assert not any("nvidia.com/" in key for key in container["resources"]["limits"])
        assert "future_strategy_hit" in command
        assert "atomic_actuation" in command
        assert "sleep" not in command
        assert job["spec"]["backoffLimit"] == 0
