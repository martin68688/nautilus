from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.atomic_actuation import (  # noqa: E402
    route_atomic_host_feedback,
    run_atomic_actuation_planner,
    run_atomic_staged_actuation_planner,
)
from agents.strategy_actuation import (  # noqa: E402
    apply_strategy_rejection_policy,
    record_recoverable_strategy_rejection,
)
from agents.strategy_plan_alignment import (  # noqa: E402
    merge_host_representation_requirements,
    run_strategy_plan_alignment_review,
)
from agents.host_plan_execution import verify_host_plan_execution  # noqa: E402
from agents.memory.cross_task_dynamic_retrieval import (  # noqa: E402
    verify_representation_adoption,
)
from config import _load_cfg  # noqa: E402
from engine.search_node import SearchNode  # noqa: E402
from engine.run_control import (  # noqa: E402
    draft_execution_lane,
    queue_host_plan_execution_repair,
)


def _memo() -> dict:
    return {
        "decision": "propose",
        "candidate_compositions": [
            {
                "hypothesis_id": "h1",
                "hypothesis": "add target descriptors through one bounded branch",
                "source_memory_ids": ["memory::1"],
                "compatibility_checks": ["descriptor width reaches fusion"],
                "known_conflicts": [],
                "estimated_compute_seconds": 120,
                "minimal_change_set": [
                    "add descriptors",
                    "connect descriptors to the existing fusion interface",
                ],
                "forbidden_changes": ["replace backbone", "change validation split"],
                "expected_mechanism": "target descriptors add complementary shape evidence",
                "falsification_condition": "OOF macro-F1 does not improve",
                "novelty_kind": "single_memory_actuation",
            }
        ],
        "recommended_hypothesis_id": "h1",
    }


def _plan(objective: str = "bounded descriptor branch") -> dict:
    return {
        "hypothesis_id": "h1",
        "objective": objective,
        "source_memory_ids": ["memory::1"],
        "allowed_modules": ["model_design"],
        "allowed_changes": [
            {
                "change_id": "descriptor-fusion",
                "operation": "modify",
                "target_symbols": ["build_model"],
                "description": "connect target descriptors to the existing fusion interface",
            }
        ],
        "allowed_new_imports": [],
        "forbidden_symbols": [],
        "forbidden_code_patterns": [],
        "preserve_invariants": ["backbone and validation split"],
        "compatibility_checks": ["descriptor width reaches fusion"],
        "estimated_compute_seconds": 120,
        "max_patches": 2,
        "expected_mechanism": "target descriptors add complementary shape evidence",
        "falsification_condition": "OOF macro-F1 does not improve",
    }


def _roadmap(objective: str) -> dict:
    phase = {
        **_plan("phase one: bounded descriptor branch"),
        "phase_id": "phase-1",
        "phase_index": 1,
        "depends_on_phase_ids": [],
    }
    return {
        "roadmap_id": "roadmap-h1",
        "hypothesis_id": "h1",
        "objective": objective,
        "source_memory_ids": ["memory::1"],
        "roadmap_complete": True,
        "phases": [phase],
    }


def _agent() -> SimpleNamespace:
    ext = SimpleNamespace(
        memory_strategy_atomic_verifier_mode="mechanical_only",
        memory_strategy_atomic_max_modules=2,
        memory_strategy_atomic_max_changes=3,
        memory_strategy_atomic_max_patches=6,
        memory_strategy_atomic_max_symbols_per_phase=64,
        experiment_r_improve_max_modules=2,
        experiment_r_improve_max_patches=6,
        memory_strategy_atomic_planner_contract_retries=0,
        memory_strategy_plan_alignment_enabled=True,
        memory_strategy_plan_alignment_retries=1,
        memory_strategy_plan_alignment_max_tokens=1000,
        memory_strategy_atomic_max_phases=3,
        memory_strategy_atomic_require_complete_roadmap=True,
        memory_strategy_model="gpt-test",
        memory_strategy_rejection_history_limit=2,
    )
    cfg = SimpleNamespace(
        external_skill_memory=ext,
        agent=SimpleNamespace(code=SimpleNamespace(model="gpt-test", temp=0.0)),
    )
    return SimpleNamespace(cfg=cfg, acfg=cfg.agent)


def test_strategy_owner_can_accept_planner_handoff():
    agent = _agent()
    agent._strategy_plan_alignment_query_fn = lambda **_kwargs: {
        "verdict": "aligned",
        "reason": "same bounded descriptor mechanism",
        "preserved_hypothesis_id": "h1",
        "preserved_source_memory_ids": ["memory::1"],
        "mismatches": [],
    }
    trace = run_strategy_plan_alignment_review(
        agent,
        strategy_memo=_memo(),
        plan=_plan(),
        stage="improve",
    )
    assert trace["status"] == "accepted"
    assert trace["validation"]["valid"] is True


def test_atomic_planner_returns_typed_alignment_feedback_and_revises():
    agent = _agent()
    prompts: list[dict] = []

    def planner_query(*, prompt, contract_attempt, **_kwargs):
        prompts.append(prompt)
        return _plan(
            "replace backbone and descriptors"
            if contract_attempt == 0
            else "bounded descriptor branch"
        )

    reviews = iter(
        [
            {
                "verdict": "revise_plan",
                "reason": "backbone replacement changes the hypothesis",
                "preserved_hypothesis_id": "h1",
                "preserved_source_memory_ids": ["memory::1"],
                "mismatches": [
                    {
                        "field": "forbidden_changes",
                        "expected": "preserve backbone",
                        "actual": "replace backbone",
                        "required_revision": "remove backbone replacement",
                    }
                ],
            },
            {
                "verdict": "aligned",
                "reason": "the revised plan preserves the descriptor hypothesis",
                "preserved_hypothesis_id": "h1",
                "preserved_source_memory_ids": ["memory::1"],
                "mismatches": [],
            },
        ]
    )
    agent._atomic_planner_query_fn = planner_query
    agent._strategy_plan_alignment_query_fn = lambda **_kwargs: next(reviews)

    trace = run_atomic_actuation_planner(
        agent,
        strategy_memo=_memo(),
        parent_code="def build_model():\n    return None\n",
        stage="improve",
    )

    assert trace["status"] == "accepted"
    assert trace["intent_alignment_verified"] is True
    assert [row["status"] for row in trace["intent_alignment_attempts"]] == [
        "revise",
        "accepted",
    ]
    second_payload = json.loads(prompts[1]["user"])
    feedback = second_payload["strategy_intent_revision_required"]
    assert feedback["mismatches"][0]["field"] == "forbidden_changes"
    assert trace["plan"]["objective"] == "bounded descriptor branch"


def test_alignment_abandon_is_not_silently_accepted():
    agent = _agent()
    agent._atomic_planner_query_fn = lambda **_kwargs: _plan()
    agent._strategy_plan_alignment_query_fn = lambda **_kwargs: {
        "verdict": "abandon_hypothesis",
        "reason": "no independently executable slice preserves the interface",
        "preserved_hypothesis_id": "h1",
        "preserved_source_memory_ids": ["memory::1"],
        "mismatches": [],
    }
    trace = run_atomic_actuation_planner(
        agent,
        strategy_memo=_memo(),
        parent_code="def build_model():\n    return None\n",
        stage="improve",
    )
    assert trace["status"] == "rejected"
    assert trace["intent_alignment"]["status"] == "rejected"


def test_staged_planner_also_requires_strategy_owner_alignment():
    agent = _agent()
    prompts: list[dict] = []

    def planner_query(*, prompt, contract_attempt, **_kwargs):
        prompts.append(prompt)
        return _roadmap(
            "cross-module rewrite"
            if contract_attempt == 0
            else "complete bounded descriptor roadmap"
        )

    reviews = iter(
        [
            {
                "verdict": "revise_plan",
                "reason": "the roadmap broadened into an unrelated rewrite",
                "preserved_hypothesis_id": "h1",
                "preserved_source_memory_ids": ["memory::1"],
                "mismatches": [
                    {
                        "field": "objective",
                        "expected": "bounded descriptor mechanism",
                        "actual": "cross-module rewrite",
                        "required_revision": "retain only descriptor integration",
                    }
                ],
            },
            {
                "verdict": "aligned",
                "reason": "ordered phase preserves the complete hypothesis",
                "preserved_hypothesis_id": "h1",
                "preserved_source_memory_ids": ["memory::1"],
                "mismatches": [],
            },
        ]
    )
    agent._atomic_planner_query_fn = planner_query
    agent._strategy_plan_alignment_query_fn = lambda **_kwargs: next(reviews)
    trace = run_atomic_staged_actuation_planner(
        agent,
        strategy_memo=_memo(),
        parent_code="def build_model():\n    return None\n",
        stage="improve",
    )
    assert trace["status"] == "accepted"
    assert trace["intent_alignment_verified"] is True
    assert len(trace["intent_alignment_attempts"]) == 2
    second_payload = json.loads(prompts[1]["user"])
    assert "strategy_intent_revision_required" in second_payload


def test_host_plan_execution_gate_blocks_retraining_after_frozen_checkpoint_plan():
    plan = _roadmap(
        "Reuse existing frozen fold checkpoints and recalibrate without retraining"
    )
    code = """
def main():
    for epoch in range(3):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
"""
    gate = verify_host_plan_execution(plan, code)

    assert gate["enabled"] is True
    assert gate["valid"] is False
    assert gate["observations"]["active_gradient_training_loops"][0][
        "loop_line"
    ] == 3
    assert gate["violations"][0].startswith("plan_runtime_retraining_conflict:")


def test_host_plan_execution_gate_accepts_checkpoint_only_inference():
    plan = _roadmap(
        "Load and reuse existing frozen fold checkpoints without retraining"
    )
    code = """
def main():
    for checkpoint in checkpoints:
        state = torch.load(checkpoint, weights_only=False)
        model.load_state_dict(state)
        predictions.append(model(batch))
"""
    gate = verify_host_plan_execution(plan, code)

    assert gate["enabled"] is True
    assert gate["valid"] is True
    assert gate["observations"]["active_gradient_training_loops"] == []


def test_host_plan_execution_gate_treats_exact_retained_checkpoint_as_frozen():
    plan = _roadmap(
        "Compare original-view and four-view inference on the same exact "
        "retained best single checkpoint"
    )
    code = """
for epoch in range(16):
    loss.backward()
    optimizer.step()
"""

    gate = verify_host_plan_execution(plan, code)

    assert gate["enabled"] is True
    assert gate["valid"] is False
    assert gate["requirement_evidence"][0]["detector"] == (
        "frozen_checkpoint_reuse"
    )
    assert gate["violations"][0].startswith("plan_runtime_retraining_conflict:")


def test_host_plan_execution_gate_does_not_infer_no_retraining_from_preservation():
    plan = _roadmap("Do not alter the existing training loop or optimizer settings")
    plan["phases"][0]["preserve_invariants"] = [
        "The unchanged target-only split, scaler, augmentation, checkpoint "
        "selection, and held-out evaluation remain executable."
    ]
    code = """
def main():
    for epoch in range(3):
        loss.backward()
        optimizer.step()
"""
    gate = verify_host_plan_execution(plan, code)

    assert gate["enabled"] is False
    assert gate["valid"] is True


def test_host_plan_execution_gate_blocks_removed_absolute_metric_threshold():
    plan = _roadmap(
        "Select the seeded average only when it strictly exceeds both the "
        "measured baseline and 0.9478724989519529; otherwise retain baseline"
    )
    code = """
ensemble_score = 0.96
baseline_score = 0.95
exceeds_absolute_threshold = True
select_average = ensemble_score > baseline_score and exceeds_absolute_threshold
"""
    gate = verify_host_plan_execution(plan, code)

    assert gate["enabled"] is True
    assert "absolute_metric_selection_threshold" in gate["requirements"]
    assert gate["valid"] is False
    assert gate["violations"][0].startswith(
        "plan_runtime_absolute_metric_threshold_missing:"
    )


def test_host_plan_execution_gate_accepts_bound_absolute_metric_threshold():
    plan = _roadmap(
        "Select the seeded average only when it strictly exceeds both the "
        "measured baseline and 0.9478724989519529; otherwise retain baseline"
    )
    code = """
ensemble_score = 0.96
baseline_score = 0.95
absolute_score_threshold = 0.9478724989519529
select_average = (
    ensemble_score > baseline_score
    and ensemble_score > absolute_score_threshold
)
"""
    gate = verify_host_plan_execution(plan, code)

    assert gate["enabled"] is True
    assert gate["valid"] is True
    assert gate["observations"]["absolute_metric_threshold_comparisons"]


def test_host_plan_execution_feedback_returns_to_planner_repair_lane():
    gate = verify_host_plan_execution(
        _roadmap("No retraining; all existing checkpoints remain frozen"),
        """
for epoch in range(2):
    loss.backward()
    optimizer.step()
""",
    )
    route = route_atomic_host_feedback(
        gate,
        origin="pre_execution_runtime_invariant",
    )
    node = SimpleNamespace(
        id="plan-runtime-blocked",
        pending_execution=False,
        is_terminal=False,
        host_plan_execution_gate=gate,
    )
    queue_host_plan_execution_repair(node)

    assert route["target"] == "planner"
    assert route["repairable"] is True
    assert node.is_terminal is False
    assert node.audit_repair_required is True
    assert node.leakage_audit["repair_queue_status"] == "pending"
    assert node.leakage_audit["repair_mode"] == "host_plan_execution"
    assert draft_execution_lane(node) == "repair"


def test_recoverable_rejection_preserves_bounded_history():
    agent = _agent()
    parent = SearchNode(code="x = 1\n", stage="improve")
    trace = {
        "stage": "improve",
        "status": "rejected",
        "reason": "atomic_pipeline_not_accepted",
        "atomic": {
            "planner": {
                "status": "rejected",
                "plan_sha256": "a" * 64,
                "plan": {"hypothesis_id": "h1"},
                "intent_alignment": {
                    "status": "revise",
                    "reason": "plan broadened the hypothesis",
                },
            },
            "coder": {"status": "not_run"},
        },
    }
    first = record_recoverable_strategy_rejection(agent, parent, trace)
    record_recoverable_strategy_rejection(agent, parent, trace)
    assert len(parent.memory_strategy_rejection_history) == 1
    assert parent.memory_strategy_rejection_history[0][
        "rejection_fingerprint"
    ] == first["rejection_fingerprint"]


def test_recoverable_rejection_preserves_coder_diagnostic_for_next_strategy():
    agent = _agent()
    parent = SearchNode(code="x = 1\n", stage="debug")
    trace = {
        "stage": "debug",
        "status": "rejected",
        "reason": "atomic_pipeline_not_accepted",
        "atomic": {
            "planner": {
                "status": "accepted",
                "plan_sha256": "c" * 64,
                "plan": {"hypothesis_id": "repair-symbol-closure"},
            },
            "coder": {
                "status": "rejected",
                "response_excerpt": (
                    "No patch: moving model before its class definition would raise NameError."
                ),
                "plan_diff_verdict": {
                    "valid": False,
                    "violations": ["diff application failed: no SEARCH/REPLACE block"],
                },
            },
        },
    }

    receipt = record_recoverable_strategy_rejection(agent, parent, trace)

    assert "NameError" in receipt["coder_diagnostic"]
    assert receipt["coder_violations"] == [
        "diff application failed: no SEARCH/REPLACE block"
    ]


def test_debug_rejection_uses_recoverable_parent_policy():
    agent = _agent()
    agent.cfg.external_skill_memory.memory_strategy_recoverable_rejection_enabled = (
        True
    )
    parent = SearchNode(code="x = 1\n", stage="debug")
    trace = {
        "stage": "debug",
        "status": "rejected",
        "reason": "atomic_pipeline_not_accepted",
        "strategy": {"status": "completed"},
        "atomic": {
            "planner": {
                "status": "rejected",
                "plan_sha256": "b" * 64,
                "plan": {"hypothesis_id": "repair-b3"},
                "intent_alignment": {
                    "status": "revise",
                    "reason": "reconcile compute budget",
                },
            }
        },
    }
    observation = apply_strategy_rejection_policy(
        agent,
        parent,
        trace,
        stage="debug",
    )

    assert parent.is_terminal is False
    assert parent.continue_improve is True
    assert observation["recovery"]["status"] == "deferred_retry"
    assert observation["recovery"]["parent_preserved"] is True
    assert parent.memory_strategy_rejection_history[-1][
        "hypothesis_id"
    ] == "repair-b3"


def test_host_feedback_routes_by_failure_owner():
    post_review = route_atomic_host_feedback(
        {"violations": ["candidate syntax error: invalid syntax"]},
        origin="post_review",
    )
    planner = route_atomic_host_feedback(
        {"violations": ["patch count 8 is outside atomic limit 1..6"]},
        origin="atomic_coder",
    )
    runtime = route_atomic_host_feedback(
        {
            "violations": [
                "selected public_pretrained initialization is absent or weights=None"
            ]
        },
        origin="pre_execution_runtime_invariant",
    )
    assert post_review["target"] == "reviewer"
    assert post_review["repairable"] is True
    assert planner["target"] == "planner"
    assert runtime["target"] == "planner"
    assert runtime["repairable"] is True


def test_host_runtime_contract_overrides_retrieval_and_blocks_random_fallback():
    ext_cfg = SimpleNamespace(
        memory_strategy_host_required_backbone_family="efficientnet_b3",
        memory_strategy_host_require_public_pretrained=True,
        memory_strategy_host_required_pretrained_loader="timm",
    )
    contracts, receipt = merge_host_representation_requirements(
        {
            "backbone_family": {
                "required_value": "efficientnet_b0",
                "candidate_id": "memory::b0",
            }
        },
        ext_cfg=ext_cfg,
    )
    rejected = verify_representation_adoption(
        "from torchvision.models import efficientnet_b0\n"
        "model = efficientnet_b0(weights=None)\n"
        "initialization_mode = 'random'\n",
        contracts,
    )
    accepted = verify_representation_adoption(
        "import timm\n"
        "model = timm.create_model('efficientnet_b3', pretrained=True)\n",
        contracts,
    )
    accepted_alias = verify_representation_adoption(
        "import timm\n"
        "model_identifier = 'efficientnet_b3'\n"
        "model = timm.create_model(model_identifier, pretrained=True)\n",
        contracts,
    )
    rejected_ambiguous_alias = verify_representation_adoption(
        "import timm\n"
        "model_identifier = 'efficientnet_b3'\n"
        "model_identifier = 'efficientnet_b0'\n"
        "model = timm.create_model(model_identifier, pretrained=True)\n",
        contracts,
    )
    accepted_import_alias = verify_representation_adoption(
        "from timm import create_model as build_model\n"
        "model = build_model('efficientnet_b3', pretrained=True)\n",
        contracts,
    )
    rejected_unverified_loader = verify_representation_adoption(
        "from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3\n"
        "model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)\n",
        contracts,
    )

    assert receipt["enabled"] is True
    assert receipt["requirements"] == {
        "backbone_family": "efficientnet_b3",
        "initialization_source": "public_pretrained",
        "pretrained_loader": "timm",
    }
    assert receipt["selection_overrides"] == [
        {
            "slot": "backbone_family",
            "selected_value": "efficientnet_b0",
            "host_required_value": "efficientnet_b3",
        }
    ]
    assert rejected["valid"] is False
    assert accepted["valid"] is True
    assert accepted_alias["valid"] is True
    assert accepted_import_alias["valid"] is True
    assert rejected_ambiguous_alias["valid"] is False
    assert rejected_unverified_loader["valid"] is False
    assert rejected_unverified_loader["violations"] == [
        "public_pretrained initialization does not use the Host-verified loader: timm"
    ]


def test_representation_gate_accepts_active_transformer_encoder_self_attention():
    contracts = {
        "fusion_operator": {"required_value": "self_attention"},
    }
    transformer_encoder = verify_representation_adoption(
        "import torch\n"
        "from torch import nn\n"
        "class FusionNet(nn.Module):\n"
        "    def __init__(self):\n"
        "        layer = nn.TransformerEncoderLayer(d_model=384, nhead=8, "
        "batch_first=True)\n"
        "        self.attn = nn.TransformerEncoder(layer, num_layers=2)\n"
        "    def forward(self, tokens):\n"
        "        attended = self.attn(tokens)\n"
        "        return torch.cat([attended[:, 0], attended[:, 1]], dim=1)\n",
        contracts,
    )
    concat_only = verify_representation_adoption(
        "import torch\n"
        "def forward(vision, descriptors):\n"
        "    return torch.cat([vision, descriptors], dim=1)\n",
        contracts,
    )

    assert transformer_encoder["valid"] is True
    assert concat_only["valid"] is False
    assert concat_only["violations"] == ["selected self-attention fusion is absent"]


def test_config_declares_alignment_feedback_defaults():
    cfg = _load_cfg(
        ROOT / "mlevolve" / "config" / "config.yaml",
        use_cli_args=False,
    )
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_plan_alignment_enabled is False
    assert ext.memory_strategy_plan_alignment_retries == 2
    assert ext.memory_strategy_recoverable_rejection_enabled is False
    assert ext.memory_strategy_post_review_repair_attempts == 0
    assert ext.memory_strategy_host_required_backbone_family == ""
    assert ext.memory_strategy_host_require_public_pretrained is False
    assert ext.memory_strategy_host_required_pretrained_loader == ""


def test_v167_enables_host_runtime_method_floor():
    cfg = _load_cfg(
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v167_plant_seedlings_intent_feedback_a100_6h_sol_low"
        / "intent_feedback.yaml",
        use_cli_args=False,
    )
    ext = cfg.external_skill_memory
    assert ext.memory_strategy_plan_alignment_retries == 3
    assert ext.memory_strategy_host_required_backbone_family == "efficientnet_b3"
    assert ext.memory_strategy_host_require_public_pretrained is True
    assert ext.memory_strategy_host_required_pretrained_loader == "timm"
