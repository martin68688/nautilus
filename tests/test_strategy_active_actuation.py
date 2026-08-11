from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

import agents.strategy_actuation as strategy_actuation
import agents.atomic_actuation as atomic_actuation
import agents.memory_strategy_agent as memory_strategy_agent
from agents.atomic_actuation import (
    build_atomic_coder_allowlist,
    validate_atomic_plan,
    validate_decomposed_replan,
    validate_scope_reconciliation,
    validate_staged_atomic_plan,
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


def _atomic_phase(
    phase_id: str,
    phase_index: int,
    *,
    target_symbols: list[str],
    operation: str = "modify",
    depends_on: list[str] | None = None,
) -> dict:
    return {
        "phase_id": phase_id,
        "phase_index": phase_index,
        "depends_on_phase_ids": list(depends_on or []),
        "hypothesis_id": "h1",
        "objective": f"apply {phase_id}",
        "source_memory_ids": ["current::parent"],
        "allowed_modules": ["model_design"],
        "allowed_changes": [
            {
                "change_id": f"change-{phase_id}",
                "operation": operation,
                "target_symbols": target_symbols,
                "description": f"apply {phase_id} exactly",
            }
        ],
        "allowed_new_imports": [],
        "forbidden_symbols": [],
        "forbidden_code_patterns": [],
        "preserve_invariants": ["evaluation and submission protocol"],
        "compatibility_checks": ["module remains parseable"],
        "estimated_compute_seconds": 30,
        "max_patches": 2,
        "expected_mechanism": f"{phase_id} changes the declared component",
        "falsification_condition": f"{phase_id} does not change behavior",
    }


def _staged_roadmap(phases: list[dict]) -> dict:
    return {
        "roadmap_id": "roadmap-h1",
        "hypothesis_id": "h1",
        "objective": "apply the complete h1 modification",
        "source_memory_ids": ["current::parent"],
        "roadmap_complete": True,
        "phases": phases,
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
        raise AssertionError(
            "Atomic pipeline must not run for an invalid Strategy memo"
        )

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
    assert (
        "abstention is disabled for required active Strategy actuation"
        in verdict["violations"]
    )


def test_active_debug_strategy_requires_selected_repair_and_full_interface_audit():
    prompt = memory_strategy_agent._strategy_prompt(
        {
            "mode": "active_atomic",
            "stage": "debug",
            "strategy_contract": {
                "abstention_allowed": False,
                "min_candidate_compositions": 1,
            },
        }
    )
    system = prompt["system"]
    assert "prompt-visible Router L3 repair is the primary causal evidence" in system
    assert "exact before/after repair" in system
    assert "output type/key/token selection" in system
    assert "same staged roadmap" in system


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
        "allowed_changes": [{"target_symbols": ["load_images", "train_transform"]}],
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
        "allowed_changes": [{"target_symbols": ["extract_features", "__module__"]}],
    }
    assert (
        validate_scope_reconciliation(
            reconciled,
            previous_plan=previous,
            coder_verdict=verdict,
        )
        == []
    )

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


def test_atomic_verifier_tracks_bindings_inside_top_level_try_block():
    parent = (
        "try:\n"
        "    dinov3_model = load_backbone()\n"
        "except Exception:\n"
        "    dinov3_model = fallback_backbone()\n"
    )
    phase = _atomic_phase("backbone", 1, target_symbols=["dinov3_model"])
    validation = validate_staged_atomic_plan(
        _staged_roadmap([phase]),
        strategy_memo=_strategy_trace()["memo"],
        max_modules=1,
        max_changes=2,
        max_patches=4,
        max_phases=3,
        max_symbols=4,
        parent_code=parent,
        stage="debug",
        debug_targeted_repair_only=True,
    )
    assert validation["valid"] is True
    candidate = parent.replace("load_backbone()", "load_efficientnet()")
    verdict = verify_atomic_code_change(
        original_code=parent,
        candidate_code=candidate,
        atomic_plan=phase,
        patch_count=1,
        require_all_planned_changes=True,
    )
    assert verdict["valid"] is True
    assert verdict["changed_symbols"] == ["dinov3_model"]


def test_atomic_verifier_attributes_top_level_autocast_header_to_module_only():
    parent = (
        "model = build_model()\n"
        "for epoch in range(2):\n"
        "    train_loss = 0.0\n"
        "    for images, labels in loader:\n"
        '        with torch.cuda.amp.autocast("cuda", enabled=True):\n'
        "            logits = model(images)\n"
        "            loss = criterion(logits, labels)\n"
        "        train_loss += loss.item()\n"
    )
    candidate = parent.replace(
        'torch.cuda.amp.autocast("cuda", enabled=True)',
        "torch.cuda.amp.autocast(enabled=True)",
    )
    phase = _atomic_phase("amp", 1, target_symbols=["__module__"])
    verdict = verify_atomic_code_change(
        original_code=parent,
        candidate_code=candidate,
        atomic_plan=phase,
        patch_count=1,
        require_all_planned_changes=True,
    )
    assert verdict["valid"] is True
    assert verdict["changed_symbols"] == ["__module__"]


def test_atomic_verifier_reattributes_wrong_model_target_from_observed_diff():
    parent = (
        "model = build_model()\n"
        "for images, labels in loader:\n"
        '    with torch.cuda.amp.autocast("cuda", enabled=True):\n'
        "        logits = model(images)\n"
    )
    candidate = parent.replace(
        'torch.cuda.amp.autocast("cuda", enabled=True)',
        "torch.cuda.amp.autocast(enabled=True)",
    )
    wrong_phase = _atomic_phase("amp", 1, target_symbols=["model"])
    verdict = verify_atomic_code_change(
        original_code=parent,
        candidate_code=candidate,
        atomic_plan=wrong_phase,
        patch_count=1,
        require_all_planned_changes=True,
    )
    assert verdict["valid"] is False
    assert verdict["changed_symbols"] == ["__module__"]
    assert verdict["unauthorized_changed_symbols"] == ["__module__"]
    assert verdict["missing_required_symbols"] == ["model"]
    assert atomic_actuation._coder_replan_mode(wrong_phase, verdict) == (
        "scope_reconciliation"
    )

    corrected_phase = _atomic_phase("amp", 1, target_symbols=["__module__"])
    assert (
        validate_scope_reconciliation(
            corrected_phase,
            previous_plan=wrong_phase,
            coder_verdict=verdict,
        )
        == []
    )


def test_mechanical_only_verifier_does_not_reject_semantic_symbol_mismatch():
    parent = (
        "model = build_model()\n"
        "for images, labels in loader:\n"
        '    with torch.cuda.amp.autocast("cuda", enabled=True):\n'
        "        logits = model(images)\n"
    )
    candidate = parent.replace(
        'torch.cuda.amp.autocast("cuda", enabled=True)',
        "torch.cuda.amp.autocast(enabled=True)",
    )
    verdict = verify_atomic_code_change(
        original_code=parent,
        candidate_code=candidate,
        atomic_plan=_atomic_phase("amp", 1, target_symbols=["model"]),
        patch_count=1,
        require_all_planned_changes=True,
        verification_mode="mechanical_only",
    )
    assert verdict["valid"] is True
    assert verdict["verification_mode"] == "mechanical_only"
    assert verdict["changed_symbols"] == []


def test_mechanical_only_staged_pipeline_bypasses_planner_contract_gate():
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_staged_enabled = True
    ext.memory_strategy_atomic_strict_coder_enabled = True
    ext.memory_strategy_atomic_verifier_mode = "mechanical_only"
    ext.memory_strategy_atomic_coder_replan_attempts = 0
    ext.memory_strategy_atomic_planner_contract_retries = 2

    phase = _atomic_phase(
        "wide-interface-change",
        1,
        target_symbols=[
            "Model.forward",
            "local_model",
            "test_features",
            "extra_one",
            "extra_two",
        ],
    )
    roadmap = _staged_roadmap([phase])
    # Mechanical mode must not require the semantic complete-roadmap flag.
    roadmap["roadmap_complete"] = False
    planner_calls = []

    def planner_query(**kwargs):
        planner_calls.append(kwargs)
        return roadmap

    agent._atomic_planner_query_fn = planner_query
    agent._atomic_coder_query_fn = lambda **_kwargs: (
        "<<<<<<< SEARCH\n"
        "    return 1\n"
        "=======\n"
        "    return 2\n"
        ">>>>>>> REPLACE\n"
    )
    result = atomic_actuation.run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="class Model:\n    def forward(self):\n        return 1\n",
        task_description="leaf classification",
        execution_output="runtime failure",
        stage="improve",
    )
    assert result["status"] == "accepted"
    assert result["full_roadmap_applied"] is True
    assert result["planner"]["semantic_contract_bypassed"] is True
    assert result["planner"]["validation"] == {
        "schema": "mlevolve_staged_atomic_plan_validation_v1",
        "verification_mode": "mechanical_only",
        "valid": True,
        "violations": [],
        "semantic_contract_bypassed": True,
        "acceptance_basis": "parseable_planner_json_only",
    }
    assert len(planner_calls) == 1
    assert result["coder"]["verification_mode"] == "mechanical_only"
    assert "return 2" in result["coder"]["candidate_code"]


def test_strict_staged_planner_still_rejects_same_invalid_contract():
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_verifier_mode = "strict"
    ext.memory_strategy_atomic_planner_contract_retries = 0
    phase = _atomic_phase(
        "wide-interface-change",
        1,
        target_symbols=[
            "Model.forward",
            "local_model",
            "test_features",
            "extra_one",
            "extra_two",
        ],
    )
    agent._atomic_planner_query_fn = lambda **_kwargs: _staged_roadmap([phase])
    result = atomic_actuation.run_atomic_staged_actuation_planner(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="class Model:\n    def forward(self):\n        return 1\n",
        stage="improve",
    )
    assert result["status"] == "rejected"
    assert result["semantic_contract_bypassed"] is False
    assert any(
        "targets non-top-level symbols" in violation
        for violation in result["validation"]["violations"]
    )


def test_staged_pipeline_reconciles_observed_scope_and_retries(monkeypatch):
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_staged_enabled = True
    ext.memory_strategy_atomic_strict_coder_enabled = True
    ext.memory_strategy_atomic_coder_replan_attempts = 1
    initial = _atomic_phase("repair", 1, target_symbols=["extract_features"])
    reconciled = _atomic_phase(
        "repair",
        1,
        target_symbols=["extract_features", "helper"],
    )
    planner_calls = []

    def planner_query(**kwargs):
        planner_calls.append(kwargs)
        payload = json.loads(kwargs["prompt"]["user"])
        return _staged_roadmap(
            [reconciled if "coder_replan_required" in payload else initial]
        )

    agent._atomic_planner_query_fn = planner_query

    def coder_query(**kwargs):
        return (
            "<<<<<<< SEARCH\n"
            "helper = 1\n\n"
            "def extract_features():\n"
            "    return helper\n"
            "=======\n"
            "helper = 2\n\n"
            "def extract_features():\n"
            "    return helper + 1\n"
            ">>>>>>> REPLACE\n"
        )

    agent._atomic_coder_query_fn = coder_query
    trace = atomic_actuation.run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="helper = 1\n\ndef extract_features():\n    return helper\n",
        task_description="scope reconciliation test",
        stage="improve",
    )
    assert trace["status"] == "accepted"
    assert trace["replan_used"] is True
    assert trace["scope_reconciliation_used"] is True
    assert trace["coder"]["candidate_code"].count("helper = 2") == 1
    assert len(planner_calls) >= 2


def test_strict_coder_requires_every_planned_symbol_and_forbids_extra_symbol():
    plan = _atomic_phase(
        "p1",
        1,
        target_symbols=["first", "second"],
    )
    original = "first = 1\nsecond = 1\nextra = 1\n"
    candidate = "first = 2\nsecond = 1\nextra = 2\n"
    verdict = verify_atomic_code_change(
        original_code=original,
        candidate_code=candidate,
        atomic_plan=plan,
        patch_count=1,
        require_all_planned_changes=True,
    )
    assert verdict["valid"] is False
    assert verdict["unauthorized_changed_symbols"] == ["extra"]
    assert verdict["missing_required_symbols"] == ["second"]
    packet = build_atomic_coder_allowlist(plan)
    assert packet["required_symbol_operations"] == {
        "first": "modify",
        "second": "modify",
    }


def test_strict_coder_retry_removes_unauthorized_change(monkeypatch):
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_strict_coder_enabled = True
    ext.memory_strategy_atomic_coder_contract_retries = 1
    plan = _atomic_phase("p1", 1, target_symbols=["first"])
    planner_trace = {"status": "accepted", "plan": plan}
    prompts = []

    def coder_query(**kwargs):
        prompts.append(kwargs["prompt"])
        if kwargs["contract_attempt"] == 0:
            return (
                "<<<<<<< SEARCH\nfirst = 1\nsecond = 1\n=======\n"
                "first = 2\nsecond = 2\n>>>>>>> REPLACE\n"
            )
        return "<<<<<<< SEARCH\nfirst = 1\n=======\n" "first = 2\n>>>>>>> REPLACE\n"

    agent._atomic_coder_query_fn = coder_query
    trace = atomic_actuation.run_atomic_coder(
        agent,
        planner_trace=planner_trace,
        parent_code="first = 1\nsecond = 1\n",
        task_description="strict coder test",
    )
    assert trace["status"] == "accepted"
    assert len(trace["contract_attempts"]) == 2
    assert trace["contract_attempts"][0]["valid"] is False
    assert trace["plan_diff_verdict"]["changed_symbols"] == ["first"]
    assert "unauthorized_changed_symbol" in prompts[1]["user"]


def test_staged_planner_retries_oversized_phase_as_complete_roadmap():
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_max_changes = 1
    ext.memory_strategy_atomic_max_phases = 3
    ext.memory_strategy_atomic_planner_contract_retries = 1
    first = _atomic_phase("p1", 1, target_symbols=["first"])
    second = _atomic_phase(
        "p2",
        2,
        target_symbols=["second"],
        depends_on=["p1"],
    )
    oversized = _atomic_phase("too-large", 1, target_symbols=["first"])
    oversized["allowed_changes"].append(
        {
            "change_id": "second-change",
            "operation": "modify",
            "target_symbols": ["second"],
            "description": "second part of the same roadmap",
        }
    )
    calls = []

    def planner_query(**kwargs):
        calls.append(kwargs)
        if kwargs["contract_attempt"] == 0:
            return _staged_roadmap([oversized])
        return _staged_roadmap([first, second])

    agent._atomic_planner_query_fn = planner_query
    trace = atomic_actuation.run_atomic_staged_actuation_planner(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="first = 1\nsecond = 1\n",
        stage="improve",
    )
    assert trace["status"] == "accepted"
    assert trace["validation"]["phase_count"] == 2
    assert trace["contract_attempts"][0]["valid"] is False
    assert "allowed_changes must contain 1..1" in " ".join(
        trace["contract_attempts"][0]["violations"]
    )
    assert "split that phase" in calls[1]["prompt"]["user"]


def test_staged_plan_rejects_one_change_that_hides_too_many_symbols():
    phase = _atomic_phase(
        "oversized-symbol-set",
        1,
        target_symbols=["first", "second", "third", "fourth", "fifth"],
    )
    verdict = validate_staged_atomic_plan(
        _staged_roadmap([phase]),
        strategy_memo=_strategy_trace()["memo"],
        max_modules=2,
        max_changes=4,
        max_patches=8,
        max_phases=3,
        max_symbols=4,
        parent_code=("first = 1\nsecond = 1\nthird = 1\nfourth = 1\nfifth = 1\n"),
        stage="improve",
    )
    assert verdict["valid"] is False
    assert "at most 4 distinct symbols per phase; found 5" in " ".join(
        verdict["violations"]
    )


def test_staged_pipeline_carries_each_phase_code_forward():
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_staged_enabled = True
    ext.memory_strategy_atomic_strict_coder_enabled = True
    first = _atomic_phase("p1", 1, target_symbols=["first"])
    second = _atomic_phase(
        "p2",
        2,
        target_symbols=["second"],
        depends_on=["p1"],
    )
    roadmap = _staged_roadmap([first, second])
    inputs = []
    agent._atomic_planner_query_fn = lambda **_kwargs: roadmap

    def coder_query(**kwargs):
        parent = kwargs["parent_code"]
        inputs.append(parent)
        if kwargs["atomic_plan"]["phase_id"] == "p1":
            return "<<<<<<< SEARCH\nfirst = 1\n=======\n" "first = 2\n>>>>>>> REPLACE\n"
        assert "first = 2" in parent
        return "<<<<<<< SEARCH\nsecond = 1\n=======\n" "second = 2\n>>>>>>> REPLACE\n"

    agent._atomic_coder_query_fn = coder_query
    trace = atomic_actuation.run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="first = 1\nsecond = 1\n",
        task_description="cumulative staging test",
        stage="improve",
    )
    assert trace["status"] == "accepted"
    assert trace["full_roadmap_applied"] is True
    assert trace["completed_phase_count"] == 2
    assert "first = 2" in inputs[1]
    assert "first = 2" in trace["coder"]["candidate_code"]
    assert "second = 2" in trace["coder"]["candidate_code"]
    assert (
        trace["phase_traces"][1]["input_code_sha256"]
        == trace["phase_traces"][0]["output_code_sha256"]
    )


def test_staged_pipeline_can_modify_one_monolithic_symbol_in_two_phases():
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_staged_enabled = True
    ext.memory_strategy_atomic_strict_coder_enabled = True
    first = _atomic_phase("p1", 1, target_symbols=["pipeline"])
    second = _atomic_phase(
        "p2",
        2,
        target_symbols=["pipeline"],
        depends_on=["p1"],
    )
    agent._atomic_planner_query_fn = lambda **_kwargs: _staged_roadmap([first, second])

    def coder_query(**kwargs):
        if kwargs["atomic_plan"]["phase_id"] == "p1":
            return (
                "<<<<<<< SEARCH\npipeline = 1\n=======\n"
                "pipeline = 2\n>>>>>>> REPLACE\n"
            )
        assert "pipeline = 2" in kwargs["parent_code"]
        return (
            "<<<<<<< SEARCH\npipeline = 2\n=======\n" "pipeline = 3\n>>>>>>> REPLACE\n"
        )

    agent._atomic_coder_query_fn = coder_query
    trace = atomic_actuation.run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="pipeline = 1\n",
        task_description="monolithic cumulative staging test",
        stage="improve",
    )
    assert trace["status"] == "accepted"
    assert trace["coder"]["candidate_code"] == "pipeline = 3\n"


def test_staged_pipeline_never_returns_partial_candidate():
    agent = _agent()
    ext = agent.cfg.external_skill_memory
    ext.memory_strategy_atomic_staged_enabled = True
    ext.memory_strategy_atomic_strict_coder_enabled = True
    first = _atomic_phase("p1", 1, target_symbols=["first"])
    second = _atomic_phase(
        "p2",
        2,
        target_symbols=["second"],
        depends_on=["p1"],
    )
    agent._atomic_planner_query_fn = lambda **_kwargs: _staged_roadmap([first, second])

    def coder_query(**kwargs):
        if kwargs["atomic_plan"]["phase_id"] == "p1":
            return "<<<<<<< SEARCH\nfirst = 1\n=======\n" "first = 2\n>>>>>>> REPLACE\n"
        return "no valid patch"

    agent._atomic_coder_query_fn = coder_query
    trace = atomic_actuation.run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_trace()["memo"],
        parent_code="first = 1\nsecond = 1\n",
        task_description="partial staging rejection test",
        stage="improve",
    )
    assert trace["status"] == "rejected"
    assert trace["completed_phase_count"] == 1
    assert trace["full_roadmap_applied"] is False
    assert trace["coder"]["candidate_code"] == ""


def test_debug_coupled_three_symbol_repair_is_one_logical_change():
    phase = _atomic_phase(
        "repair-width",
        1,
        target_symbols=["GROUP_FEAT_DIM", "extract_groups", "LeafClassifier"],
    )
    roadmap = _staged_roadmap([phase])
    verdict = validate_staged_atomic_plan(
        roadmap,
        strategy_memo=_strategy_trace()["memo"],
        max_modules=1,
        max_changes=2,
        max_patches=4,
        max_phases=3,
        parent_code=(
            "GROUP_FEAT_DIM = 30\n"
            "def extract_groups():\n    return []\n"
            "class LeafClassifier:\n    pass\n"
        ),
        stage="debug",
        debug_targeted_repair_only=True,
    )
    assert verdict["valid"] is True


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


def test_v80_config_enables_strict_complete_cumulative_staging():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v80"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v80-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_atomic_staged_enabled is True
    assert ext.memory_strategy_atomic_strict_coder_enabled is True
    assert ext.memory_strategy_atomic_require_complete_roadmap is True
    assert ext.memory_strategy_atomic_max_phases == 3


def test_v81_config_caps_symbols_without_blocking_coupled_debug_repair():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v81"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v81-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_atomic_staged_enabled is True
    assert ext.memory_strategy_atomic_strict_coder_enabled is True
    assert ext.memory_strategy_atomic_max_symbols_per_phase == 4
    assert ext.memory_strategy_atomic_debug_max_symbols_per_phase == 4
    assert cfg.agent.draft_role_policy.replay_targets_path.startswith(
        "/workspace/nautilus-exp-end2end-agent-v81/"
    )


def test_v82_config_keeps_strict_staging_and_enables_bounded_reconciliation():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v82"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v82-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_atomic_staged_enabled is True
    assert ext.memory_strategy_atomic_strict_coder_enabled is True
    assert ext.memory_strategy_atomic_coder_replan_attempts == 2
    assert ext.memory_strategy_atomic_max_symbols_per_phase == 4
    assert cfg.agent.draft_role_policy.replay_targets_path.startswith(
        "/workspace/nautilus-exp-end2end-agent-v82/"
    )


def test_v83_config_disables_semantic_verifier_but_keeps_staged_planner():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v83"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v83-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_atomic_staged_enabled is True
    assert ext.memory_strategy_atomic_strict_coder_enabled is True
    assert ext.memory_strategy_atomic_verifier_mode == "mechanical_only"
    assert cfg.agent.draft_role_policy.replay_targets_path.startswith(
        "/workspace/nautilus-exp-end2end-agent-v83/"
    )


def test_v84_config_preserves_mechanical_mode_with_new_immutable_paths():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v84"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v84-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    assert (
        cfg.external_skill_memory.memory_strategy_atomic_verifier_mode
        == "mechanical_only"
    )
    assert cfg.agent.draft_role_policy.replay_targets_path.startswith(
        "/workspace/nautilus-exp-end2end-agent-v84/"
    )


def test_v85_config_removes_planner_gate_and_bounds_l3_agent_shortlist():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v85"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v85-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_atomic_verifier_mode == "mechanical_only"
    assert ext.experiment_r_l3_agent_match_candidate_limit == 8
    assert cfg.agent.draft_role_policy.replay_targets_path.startswith(
        "/workspace/nautilus-exp-end2end-agent-v85/"
    )


def test_v86_config_enables_authorized_structured_plus_dense_l3_union():
    path = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v86"
        / "dynamic_hybrid.yaml"
    )
    raw = _load_cfg(path, use_cli_args=False)
    raw.exp_name = "leaf-strategy-v86-config-test"
    cfg = OmegaConf.merge(OmegaConf.structured(Config), raw)
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_atomic_verifier_mode == "mechanical_only"
    assert ext.experiment_r_l3_agent_match_candidate_limit == 8
    assert ext.experiment_r_l3_semantic_shortlist_enabled is True
    assert cfg.evaluation_authority.mode == "shadow"
    assert ext.visibility_mode_override == "enforce"
    assert cfg.evaluation_authority.enforce_operations == ["debug_hypothesis"]
    assert cfg.evaluation_authority.enforce_generation_stages == ["debug"]
    assert cfg.evaluation_authority.enforce_governance_stages == ["retrieval"]
    assert cfg.agent.draft_role_policy.replay_targets_path.startswith(
        "/workspace/nautilus-exp-end2end-agent-v86/"
    )
