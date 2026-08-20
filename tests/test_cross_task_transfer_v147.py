from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


def _policy(**overrides):
    from agents.memory.cross_task_transfer import CrossTaskTransferPolicy

    values = {
        "enabled": True,
        "source_task_id": "leaf-classification",
        "source_task_type": "leaf_descriptor_multiclass",
        "target_task_type": "leaf_descriptor_multiclass",
        "allowed_levels": ("L2_tactic", "L3_repair"),
        "max_items": 6,
    }
    values.update(overrides)
    policy = CrossTaskTransferPolicy(**values)
    policy.validate()
    return policy


def _nodes():
    return {
        "tactic::leaf::001": {
            "id": "tactic::leaf::001",
            "type": "SOP",
            "task_id": "leaf-classification",
            "abstraction_level": "L2_tactic",
            "title": "Complete OOF coverage",
            "instruction": "Write each validation row once and assert full coverage.",
            "when_to_use": "multiclass model selection",
            "teacher_boundary": "fit preprocessing inside each fold",
            "official_metric": 0.00101,
            "official_kaggle_ref": "secret-source-score-reference",
            "implementation_capsule": "print('source code must not transfer')",
        },
        "recipe::leaf::001": {
            "id": "recipe::leaf::001",
            "type": "SOP",
            "task_id": "leaf-classification",
            "abstraction_level": "L1_recipe",
            "title": "Exact source recipe",
            "code": "print('exact replay')",
            "official_metric": 0.0001,
        },
        "tactic::foreign::001": {
            "id": "tactic::foreign::001",
            "type": "SOP",
            "task_id": "spooky-author-identification",
            "abstraction_level": "L2_tactic",
            "title": "Foreign text tactic",
            "instruction": "Use character ngrams.",
        },
    }


def test_host_activates_only_for_different_task_with_same_explicit_type():
    from agents.memory.cross_task_transfer import decide_transfer

    policy = _policy()
    transfer = decide_transfer(
        policy, target_task_id="uci-one-hundred-leaves"
    )
    assert transfer.active is True
    assert transfer.reason == "different_task_same_explicit_type"

    exact = decide_transfer(policy, target_task_id="leaf-classification")
    assert exact.active is False
    assert exact.reason == "exact_task_must_use_existing_replay_path"


def test_mismatched_task_types_fail_at_configuration_time():
    from agents.memory.cross_task_transfer import CrossTaskTransferPolicy

    with pytest.raises(ValueError, match="task types must match"):
        CrossTaskTransferPolicy(
            enabled=True,
            source_task_id="leaf-classification",
            source_task_type="leaf_descriptor_multiclass",
            target_task_type="text_classification",
        ).validate()


def test_projection_exposes_portable_l2_but_never_recipe_code_or_source_score():
    from agents.memory.cross_task_transfer import build_transfer_pack

    pack = build_transfer_pack(
        _nodes(),
        _policy(),
        target_task_id="uci-one-hundred-leaves",
        stage="draft",
        task_description="100-class leaf descriptor classification",
        query_text="stratified folds and complete OOF validation",
    )

    assert pack["memory_transfer"]["activated"] is True
    assert pack["final_prompt_candidate_ids"] == ["tactic::leaf::001"]
    assert "recipe::leaf::001" not in pack["prompt_text"]
    assert "0.00101" not in pack["prompt_text"]
    assert "secret-source-score-reference" not in pack["prompt_text"]
    assert "source code must not transfer" not in pack["prompt_text"]
    assert pack["visibility_safety_gate"]["source_score_fields_exposed"] == 0
    assert pack["visibility_safety_gate"]["source_code_fields_exposed"] == 0
    assert all(
        row["source_score_inherited"] is False
        and row["source_code_exposed"] is False
        for row in pack["candidate_pool"]
    )


def test_transfer_host_gate_precedes_legacy_end2end_controller():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    class ForbiddenEnd2EndController:
        def retrieve(self, *_args, **_kwargs):
            raise AssertionError("legacy End2End retrieval must not run for transfer")

    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer.nodes = _nodes()
    layer.cross_task_transfer_policy = _policy()
    layer.end2end_controller = ForbiddenEnd2EndController()
    layer.prospective_audit_logger = None
    layer._trace_local = threading.local()
    layer._last_agentic_pack = {}

    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="uci-one-hundred-leaves",
        task_desc="100-class leaf descriptors",
        query_parts=["complete OOF validation"],
        draft_role="memory_transfer",
    )

    assert refs == ["tactic::leaf::001"]
    assert "Complete OOF coverage" in text
    pack = layer.current_navigation_pack()
    assert pack["stage_route"]["control"] == "dynamic_cross_task_transfer"
    assert pack["memory_transfer"]["host_decision"]["reason"] == (
        "different_task_same_explicit_type"
    )


def test_adoption_trace_accepts_list_shaped_transfer_candidate_pool():
    from agents.adoption import log_adoption
    from agents.memory.cross_task_transfer import build_transfer_pack

    pack = build_transfer_pack(
        _nodes(),
        _policy(),
        target_task_id="uci-one-hundred-leaves",
        stage="draft",
        task_description="100-class leaf descriptor classification",
        query_text="complete OOF validation",
    )
    layer = SimpleNamespace(
        current_navigation_pack=lambda: pack,
        current_visibility_pack=lambda: None,
        experiment_r_enabled=True,
        memory_snapshot=None,
    )
    node = SimpleNamespace(
        id="node::transfer",
        adoption_log=[],
        memory_navigation_trace=[],
        memory_routing_trace={},
        replay_source={},
    )
    agent = SimpleNamespace(
        external_skill_memory=layer,
        cfg=SimpleNamespace(run_identity=SimpleNamespace()),
        evaluation_authority=None,
        adoption_tracking_enabled=True,
    )

    log_adoption(
        node,
        agent,
        "run_forest_stage_hybrid_memory",
        pack["final_prompt_candidate_ids"],
        "draft",
    )

    trace = node.memory_routing_trace
    assert trace["memory_pack_schema"] == "mlevolve_cross_task_transfer_pack_v1"
    assert isinstance(trace["raw_candidates"], list)
    assert trace["selected_candidates"] == pack["selected_candidates"]
    assert trace["memory_transfer"]["source_score_inheritance_allowed"] is False
    assert trace["memory_transfer"]["source_code_exposure_allowed"] is False


def test_transfer_pair_is_valid_for_protected_coverage_fusion():
    from engine.agent_search import AgentSearch
    from engine.conditions import coverage_synthesis_due

    policy = SimpleNamespace(
        enabled=True,
        roles=["memory_transfer", "novel_exploration"],
        ensure_valid_candidate_per_role=True,
        role_balance_min_valid_candidates=1,
        cross_role_synthesis_after_balance=True,
        cross_role_synthesis_on_coverage=True,
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.acfg = SimpleNamespace(initial_drafts=2, draft_role_policy=policy)
    agent.scfg = SimpleNamespace(num_drafts=2)
    AgentSearch._validate_draft_role_policy(agent)

    due_agent = SimpleNamespace(
        acfg=SimpleNamespace(draft_role_policy=policy),
        fusion_draft_count=0,
        max_fusion_drafts=1,
        role_balance_status=lambda: {
            "enabled": True,
            "active": False,
            "all_slots_reserved": True,
            "deficit_roles": [],
        },
    )
    assert coverage_synthesis_due(due_agent) is True
