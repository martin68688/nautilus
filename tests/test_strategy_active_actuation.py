from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

import agents.strategy_actuation as strategy_actuation
import agents.atomic_actuation as atomic_actuation
import agents.memory_strategy_agent as memory_strategy_agent
from agents.atomic_actuation import (
    validate_atomic_plan,
    validate_decomposed_replan,
    validate_scope_reconciliation,
    verify_atomic_code_change,
)
from agents.memory_strategy_agent import validate_strategy_memo
from config import Config, _load_cfg


def _agent() -> SimpleNamespace:
    ext = SimpleNamespace(
        memory_strategy_active_enabled=True,
        memory_strategy_active_stages=["improve", "debug"],
        memory_strategy_active_required=True,
    )
    cfg = SimpleNamespace(
        external_skill_memory=ext,
        agent=SimpleNamespace(
            code=SimpleNamespace(model="deepseek-v4-flash", temp=0.0),
            time_limit=21600,
        ),
    )
    return SimpleNamespace(
        cfg=cfg,
        acfg=cfg.agent,
        task_desc="leaf classification",
        search_start_time=time.time() - 10,
    )


def _strategy_trace() -> dict:
    return {
        "schema": "mlevolve_memory_strategy_shadow_trace_v2",
        "status": "completed",
        "mode": "active_atomic",
        "actuation_authority": "atomic_planner_coder",
        "validation": {"valid": True},
        "memo": {
            "decision": "propose",
            "candidate_compositions": [
                {
                    "hypothesis_id": "h1",
                    "novelty_kind": "targeted_repair",
                    "source_memory_ids": ["current::parent"],
                }
            ],
        },
    }


def test_active_strategy_bypasses_shadow_enable_and_marks_authority(monkeypatch):
    agent = _agent()
    agent.cfg.external_skill_memory.memory_strategy_shadow_enabled = False
    captured = {}

    def fake_shadow(*_args, **kwargs):
        captured.update(kwargs)
        return {
            "mode": kwargs["_mode"],
            "actuation_authority": kwargs["_actuation_authority"],
            "trigger_reason": "active_required" if kwargs["_force_run"] else "",
        }

    monkeypatch.setattr(
        memory_strategy_agent, "run_memory_strategy_shadow", fake_shadow
    )
    parent = SimpleNamespace(id="parent", code="x = 1\n", is_buggy=False)
    trace = memory_strategy_agent.run_memory_strategy_active(
        agent,
        parent,
        stage="improve",
        router_pack={},
    )
    assert trace["mode"] == "active_atomic"
    assert trace["actuation_authority"] == "atomic_planner_coder"
    assert trace["trigger_reason"] == "active_required"
    assert captured["_force_run"] is True


def test_active_strategy_actuation_accepts_verified_atomic_candidate(monkeypatch):
    agent = _agent()
    parent = SimpleNamespace(code="value = 1\n", term_out="ok")
    monkeypatch.setattr(
        strategy_actuation,
        "run_memory_strategy_active",
        lambda *_args, **_kwargs: _strategy_trace(),
    )
    monkeypatch.setattr(
        strategy_actuation,
        "run_atomic_actuation_pipeline",
        lambda *_args, **_kwargs: {
            "status": "accepted",
            "planner": {
                "plan": {
                    "hypothesis_id": "h1",
                    "source_memory_ids": ["current::parent"],
                }
            },
            "coder": {
                "candidate_code": "value = 2\n",
                "plan_diff_verdict": {"valid": True, "patch_count": 1},
            },
        },
    )
    trace = strategy_actuation.run_active_strategy_actuation(
        agent,
        parent,
        stage="debug",
        router_pack={},
        branch_best_metric=0.1,
        production_prompt_sha256="a" * 64,
    )
    assert trace["status"] == "accepted"
    assert trace["candidate_code"] == "value = 2\n"
    assert trace["source_memory_ids"] == ["current::parent"]


def test_active_strategy_actuation_fails_closed_before_atomic_on_bad_contract(
    monkeypatch,
):
    agent = _agent()
    parent = SimpleNamespace(code="value = 1\n", term_out="failed")
    bad = _strategy_trace()
    bad["status"] = "completed_with_contract_violations"
    bad["validation"] = {"valid": False, "violations": ["incomplete"]}
    monkeypatch.setattr(
        strategy_actuation,
        "run_memory_strategy_active",
        lambda *_args, **_kwargs: bad,
    )

    def unexpected_atomic(*_args, **_kwargs):
        raise AssertionError("Atomic pipeline must not run for an invalid Strategy memo")

    monkeypatch.setattr(
        strategy_actuation, "run_atomic_actuation_pipeline", unexpected_atomic
    )
    trace = strategy_actuation.run_active_strategy_actuation(
        agent,
        parent,
        stage="improve",
        router_pack={},
        branch_best_metric=None,
        production_prompt_sha256="",
    )
    assert trace["status"] == "rejected"
    assert trace["reason"] == "strategy_contract_not_completed"


def test_debug_atomic_plan_requires_targeted_repair():
    memo = {
        "candidate_compositions": [
            {
                "hypothesis_id": "global-redesign",
                "novelty_kind": "new_composition",
                "source_memory_ids": ["history::1"],
            }
        ]
    }
    plan = {
        "hypothesis_id": "global-redesign",
        "objective": "replace the model",
        "source_memory_ids": ["history::1"],
        "allowed_modules": ["model_design"],
        "allowed_changes": [
            {
                "change_id": "replace",
                "operation": "modify",
                "target_symbols": ["build_model"],
                "description": "replace the model",
            }
        ],
        "allowed_new_imports": [],
        "forbidden_symbols": [],
        "forbidden_code_patterns": [],
        "preserve_invariants": ["submission columns"],
        "compatibility_checks": ["shape matches"],
        "estimated_compute_seconds": 100,
        "max_patches": 1,
        "expected_mechanism": "different model",
        "falsification_condition": "runtime still fails",
    }
    verdict = validate_atomic_plan(
        plan,
        strategy_memo=memo,
        max_modules=1,
        max_changes=2,
        max_patches=3,
        parent_code="def build_model():\n    return None\n",
        stage="debug",
        debug_targeted_repair_only=True,
    )
    assert verdict["valid"] is False
    assert any(
        value.startswith("Debug actuation must select a targeted_repair")
        for value in verdict["violations"]
    )


def test_debug_atomic_plan_accepts_bounded_single_memory_repair():
    memo = {
        "candidate_compositions": [
            {
                "hypothesis_id": "fix-unpack",
                "novelty_kind": "single_memory_actuation",
                "source_memory_ids": ["current::failed"],
            }
        ]
    }
    plan = {
        "hypothesis_id": "fix-unpack",
        "objective": "fix the demonstrated tuple unpack",
        "source_memory_ids": ["current::failed"],
        "allowed_modules": ["data_processing_and_feature_engineering"],
        "allowed_changes": [
            {
                "change_id": "fix",
                "operation": "modify",
                "target_symbols": ["extract_features"],
                "description": "unpack the two values emitted by DataLoader",
            }
        ],
        "allowed_new_imports": [],
        "forbidden_symbols": [],
        "forbidden_code_patterns": [],
        "preserve_invariants": ["model and validation"],
        "compatibility_checks": ["loader emits two values"],
        "estimated_compute_seconds": 60,
        "max_patches": 1,
        "expected_mechanism": "matching tuple arity removes the exception",
        "falsification_condition": "the same unpack error remains",
    }
    verdict = validate_atomic_plan(
        plan,
        strategy_memo=memo,
        max_modules=1,
        max_changes=2,
        max_patches=4,
        parent_code="def extract_features():\n    return None\n",
        stage="debug",
        debug_targeted_repair_only=True,
    )
    assert verdict["valid"] is True


def test_required_active_strategy_rejects_abstention_contract():
    memo = {
        "decision": "abstain",
        "abstention_reason": "only one observed pipeline",
        "current_system_map": {},
        "evidence_portfolio": {},
        "coverage_gaps": [],
        "candidate_compositions": [],
        "addressed_opportunities": [],
        "recommended_hypothesis_id": "",
        "recommendation_reason": "collect more evidence",
        "declined_hypotheses": [],
    }
    verdict = validate_strategy_memo(
        memo,
        available_memory_ids=["history::method"],
        min_candidate_compositions=1,
        abstention_allowed=False,
    )
    assert verdict["valid"] is False
    assert "abstention is disabled for required active Strategy actuation" in verdict[
        "violations"
    ]


def test_atomic_pipeline_replans_smaller_phase_after_coder_rejection(monkeypatch):
    agent = _agent()
    agent.cfg.external_skill_memory.memory_strategy_atomic_coder_replan_attempts = 1
    planner_feedback = []

    def fake_planner(*_args, coder_replan_feedback=None, **_kwargs):
        planner_feedback.append(coder_replan_feedback)
        phase = "broad" if coder_replan_feedback is None else "path-only"
        return {
            "status": "accepted",
            "plan": {
                "hypothesis_id": "h1",
                "source_memory_ids": ["current::parent"],
                "objective": phase,
            },
        }

    def fake_coder(*_args, planner_trace, **_kwargs):
        if planner_trace["plan"]["objective"] == "broad":
            return {
                "status": "rejected",
                "plan_diff_verdict": {
                    "valid": False,
                    "violations": ["patch count 5 is outside atomic limit 1..4"],
                },
            }
        return {
            "status": "accepted",
            "candidate_code": "value = 2\n",
            "plan_diff_verdict": {"valid": True, "patch_count": 1},
        }

    monkeypatch.setattr(
        atomic_actuation,
        "run_atomic_actuation_planner",
        fake_planner,
    )
    monkeypatch.setattr(atomic_actuation, "run_atomic_coder", fake_coder)

    trace = atomic_actuation.run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="value = 1\n",
        task_description="leaf classification",
        execution_output="FileNotFoundError",
        stage="debug",
    )

    assert trace["status"] == "accepted"
    assert trace["decomposition_used"] is True
    assert len(trace["actuation_attempts"]) == 2
    assert planner_feedback[0] is None
    assert planner_feedback[1]["previous_plan"]["objective"] == "broad"
    assert planner_feedback[1]["coder_verdict"]["valid"] is False
    assert trace["planner"]["plan"]["objective"] == "path-only"


def test_atomic_pipeline_tries_smaller_alternate_hypothesis_after_decomposition(
    monkeypatch,
):
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_coder_replan_attempts = 0
    ext.memory_strategy_atomic_alternate_hypothesis_attempts = 1
    planner_feedback = []

    def fake_planner(*_args, coder_replan_feedback=None, **_kwargs):
        planner_feedback.append(coder_replan_feedback)
        hypothesis_id = (
            "h2"
            if (coder_replan_feedback or {}).get("allow_hypothesis_switch")
            else "h1"
        )
        return {
            "status": "accepted",
            "plan": {
                "hypothesis_id": hypothesis_id,
                "source_memory_ids": ["current::parent"],
                "objective": "alternate" if hypothesis_id == "h2" else "broad",
            },
        }

    def fake_coder(*_args, planner_trace, **_kwargs):
        accepted = planner_trace["plan"]["hypothesis_id"] == "h2"
        return {
            "status": "accepted" if accepted else "rejected",
            "candidate_code": "value = 2\n" if accepted else "",
            "plan_diff_verdict": {
                "valid": accepted,
                "violations": [] if accepted else ["scope mismatch"],
            },
        }

    monkeypatch.setattr(atomic_actuation, "run_atomic_actuation_planner", fake_planner)
    monkeypatch.setattr(atomic_actuation, "run_atomic_coder", fake_coder)
    trace = atomic_actuation.run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="value = 1\n",
        task_description="leaf classification",
        stage="improve",
    )

    assert trace["status"] == "accepted"
    assert trace["alternate_hypothesis_used"] is True
    assert [item["kind"] for item in trace["actuation_attempts"]] == [
        "initial",
        "alternate_hypothesis",
    ]
    assert planner_feedback[1]["rejected_hypothesis_ids"] == ["h1"]
    assert trace["planner"]["plan"]["hypothesis_id"] == "h2"


def test_alternate_hypothesis_reconciles_only_observed_scope(monkeypatch):
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_coder_replan_attempts = 0
    ext.memory_strategy_atomic_alternate_hypothesis_attempts = 1
    ext.memory_strategy_atomic_alternate_replan_attempts = 1

    def fake_planner(*_args, coder_replan_feedback=None, **_kwargs):
        feedback = coder_replan_feedback or {}
        hypothesis_id = "h2" if feedback else "h1"
        symbols = ["extract_features"]
        if feedback.get("replan_mode") == "scope_reconciliation":
            symbols.append("__module__")
        return {
            "status": "accepted",
            "plan": {
                "hypothesis_id": hypothesis_id,
                "source_memory_ids": ["current::parent"],
                "allowed_modules": ["data_processing_and_feature_engineering"],
                "allowed_changes": [{"target_symbols": symbols}],
                "allowed_new_imports": [],
                "max_patches": 4,
            },
        }

    def fake_coder(*_args, planner_trace, **_kwargs):
        plan = planner_trace["plan"]
        symbols = {
            value
            for change in plan["allowed_changes"]
            for value in change["target_symbols"]
        }
        if plan["hypothesis_id"] == "h1":
            return {
                "status": "rejected",
                "plan_diff_verdict": {
                    "valid": False,
                    "violations": ["patch count 9 is outside atomic limit 1..4"],
                    "patch_count": 9,
                    "max_patches": 4,
                },
            }
        if "__module__" not in symbols:
            return {
                "status": "rejected",
                "plan_diff_verdict": {
                    "valid": False,
                    "violations": [
                        "changed symbols outside allowed set: ['__module__']"
                    ],
                    "unauthorized_changed_symbols": ["__module__"],
                    "patch_count": 2,
                    "max_patches": 4,
                    "new_imports": [],
                },
            }
        return {
            "status": "accepted",
            "candidate_code": "value = 2\n",
            "plan_diff_verdict": {"valid": True, "patch_count": 2},
        }

    monkeypatch.setattr(atomic_actuation, "run_atomic_actuation_planner", fake_planner)
    monkeypatch.setattr(atomic_actuation, "run_atomic_coder", fake_coder)
    trace = atomic_actuation.run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="value = 1\n",
        task_description="leaf classification",
        stage="improve",
    )
    assert trace["status"] == "accepted"
    assert trace["alternate_hypothesis_used"] is True
    assert trace["scope_reconciliation_used"] is True
    assert [item["kind"] for item in trace["actuation_attempts"]] == [
        "initial",
        "alternate_hypothesis",
        "alternate_replan",
    ]


def test_decomposed_replan_must_strictly_reduce_verified_boundary():
    broad = {
        "hypothesis_id": "h1",
        "source_memory_ids": ["history::1"],
        "allowed_modules": ["data", "model"],
        "allowed_changes": [
            {"target_symbols": ["load_images", "train_transform"]},
            {"target_symbols": ["extract_features"]},
        ],
        "allowed_new_imports": ["torchvision.transforms"],
        "max_patches": 8,
    }
    assert validate_decomposed_replan(broad, previous_plan=broad) == [
        "decomposed replan must strictly reduce at least one verified boundary"
    ]

    first_phase = {
        **broad,
        "allowed_modules": ["data"],
        "allowed_changes": [
            {"target_symbols": ["load_images", "train_transform"]}
        ],
        "max_patches": 4,
    }
    assert validate_decomposed_replan(first_phase, previous_plan=broad) == []


def test_scope_reconciliation_only_authorizes_observed_symbols():
    previous = {
        "hypothesis_id": "h1",
        "source_memory_ids": ["history::1"],
        "allowed_modules": ["data"],
        "allowed_changes": [{"target_symbols": ["extract_features"]}],
        "allowed_new_imports": [],
        "max_patches": 4,
    }
    verdict = {
        "violations": ["changed symbols outside allowed set: ['__module__']"],
        "unauthorized_changed_symbols": ["__module__"],
        "patch_count": 2,
        "new_imports": [],
    }
    assert atomic_actuation._coder_replan_mode(previous, verdict) == (
        "scope_reconciliation"
    )
    reconciled = {
        **previous,
        "allowed_changes": [
            {"target_symbols": ["extract_features", "__module__"]}
        ],
    }
    assert validate_scope_reconciliation(
        reconciled,
        previous_plan=previous,
        coder_verdict=verdict,
    ) == []

    widened = {
        **reconciled,
        "allowed_changes": [
            {"target_symbols": ["extract_features", "__module__", "train_fold"]}
        ],
    }
    assert any(
        "not observed" in value
        for value in validate_scope_reconciliation(
            widened,
            previous_plan=previous,
            coder_verdict=verdict,
        )
    )


def test_atomic_verifier_tracks_top_level_assignments_as_named_symbols():
    original = "def load_images():\n    return []\n"
    candidate = (
        'train_transform = "augment"\n'
        'test_transform = "base"\n'
        "def load_images():\n    return []\n"
    )
    verdict = verify_atomic_code_change(
        original_code=original,
        candidate_code=candidate,
        atomic_plan={
            "allowed_changes": [
                {
                    "target_symbols": ["train_transform", "test_transform"],
                }
            ],
            "allowed_new_imports": [],
            "forbidden_symbols": [],
            "forbidden_code_patterns": [],
            "max_patches": 2,
        },
        patch_count=1,
    )
    assert verdict["valid"] is True
    assert verdict["changed_symbols"] == ["test_transform", "train_transform"]
    assert "__module__" not in verdict["changed_symbols"]


def test_v74_config_enables_required_staged_actuation_with_modest_limits():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v74"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v74-config-test"
    cfg = OmegaConf.merge(
        OmegaConf.structured(Config),
        raw,
    )
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_active_enabled is True
    assert ext.memory_strategy_active_allow_abstention is False
    assert ext.memory_strategy_debug_min_candidate_compositions == 1
    assert ext.memory_strategy_atomic_max_patches == 8
    assert ext.memory_strategy_atomic_debug_max_modules == 1
    assert ext.memory_strategy_atomic_debug_max_changes == 2
    assert ext.experiment_r_debug_max_patches == 4
    assert ext.memory_strategy_atomic_coder_replan_attempts == 2


def test_v75_config_adds_one_bounded_alternate_atomic_hypothesis():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v75"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v75-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_active_required is True
    assert ext.memory_strategy_atomic_coder_replan_attempts == 2
    assert ext.memory_strategy_atomic_alternate_hypothesis_attempts == 1
    assert cfg.agent.draft_role_policy.replay_targets_path.startswith(
        "/workspace/nautilus-exp-end2end-agent-v75/"
    )


def test_v76_config_replans_one_alternate_scope_reconciliation():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v76"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v76-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_active_required is True
    assert ext.memory_strategy_atomic_alternate_hypothesis_attempts == 1
    assert ext.memory_strategy_atomic_alternate_replan_attempts == 1
    assert cfg.agent.draft_role_policy.replay_targets_path.startswith(
        "/workspace/nautilus-exp-end2end-agent-v76/"
    )
