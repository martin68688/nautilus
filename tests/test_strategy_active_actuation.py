from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

import agents.strategy_actuation as strategy_actuation
import agents.memory_strategy_agent as memory_strategy_agent
from agents.atomic_actuation import validate_atomic_plan


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
    assert "Debug actuation must select a targeted_repair hypothesis" in verdict[
        "violations"
    ]
