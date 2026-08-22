from __future__ import annotations

import json
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
        "architecture_transfer_enabled": False,
        "architecture_max_items": 1,
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
            "title": "Fold ensemble with multimodal fusion",
            "method_family": "stratified_fold_multiview_oof_calibration",
            "teacher_distilled_recipe": (
                "Train a stratified fold ensemble over multimodal leaf features, "
                "derive calibration from OOF predictions, and reproduce the best "
                "official artifact."
            ),
            "pipeline": {
                "data_validation": (
                    "Verify 990 training rows, 594 test rows, 99 classes, and "
                    "CSV bb19d7d42b8e1923825f462dc7b42a033381488e8766f90af50166aee996f6d4."
                ),
                "feature_representation": (
                    "Use descriptor views and a contour/image representation; "
                    "derive every feature width from target arrays."
                ),
                "model_stack": (
                    "Use separate view encoders followed by a probability-level "
                    "fusion interface and a 99-way source head."
                ),
                "training_protocol": (
                    "Train fresh fold models and checkpoint by validation log loss."
                ),
                "oof_protocol": (
                    "Fill row-indexed OOF probabilities exactly once per training row."
                ),
                "ensemble_calibration": (
                    "Learn fusion and scalar temperature only from OOF predictions."
                ),
                "final_refit_inference": (
                    "Average fold test probabilities and require CSV "
                    "bb19d7d42b8e1923825f462dc7b42a033381488e8766f90af50166aee996f6d4."
                ),
            },
            "code": "print('exact replay')",
            "implementation_capsule": "print('recipe capsule')",
            "official_metric": 0.0001,
            "official_kaggle_ref": "55613290",
            "class_mapping": {"Acer": 0},
            "predictions": [[1.0, 0.0]],
        },
        "repair::leaf::001": {
            "id": "repair::leaf::001",
            "type": "SOP",
            "task_id": "leaf-classification",
            "abstraction_level": "L3_repair",
            "title": "Align probability columns",
            "when_to_use": "predict_proba omits or reorders classes",
            "failure_signature": {"exception_names": ["ClassOrderMismatch"]},
            "repair_action": {
                "summary": "Map model classes into the target class order and renormalize."
            },
            "implementation_capsule": "print('source repair code must not transfer')",
            "official_metric": 0.0002,
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
    transfer = decide_transfer(policy, target_task_id="uci-one-hundred-leaves")
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
        row["source_score_inherited"] is False and row["source_code_exposed"] is False
        for row in pack["candidate_pool"]
    )


def test_opt_in_architecture_projection_adds_one_sanitized_l1_blueprint():
    from agents.memory.cross_task_transfer import (
        ARCHITECTURE_TRANSFER_PACK_SCHEMA,
        build_transfer_pack,
    )

    pack = build_transfer_pack(
        _nodes(),
        _policy(architecture_transfer_enabled=True),
        target_task_id="uci-one-hundred-leaves",
        stage="draft",
        task_description=(
            "100-class leaf descriptor and image classification with folds and OOF"
        ),
        query_text="multimodal view encoders, fold ensemble, and calibration",
    )

    assert pack["schema"] == ARCHITECTURE_TRANSFER_PACK_SCHEMA
    assert pack["memory_transfer"]["architecture_transfer_enabled"] is True
    assert pack["memory_transfer"]["architecture_projection_mode"] == (
        "host_component_slots_only_v2"
    )
    assert pack["memory_transfer"]["selected_architecture_ids"] == [
        "recipe::leaf::001::representation_blueprint"
    ]
    assert pack["final_prompt_candidate_ids"] == [
        "recipe::leaf::001::representation_blueprint",
        "recipe::leaf::001::validation_blueprint",
        "recipe::leaf::001::calibration_blueprint",
        "recipe::leaf::001::inference_blueprint",
        "tactic::leaf::001",
    ]
    blueprint = pack["selected_architectures"][0]
    assert blueprint["candidate_kind"] == "representation_blueprint"
    assert blueprint["portable_text"]["pipeline_order"] == [
        "feature_representation",
        "model_stack",
    ]
    assert set(pack["memory_transfer"]["selected_blueprint_slots"]) == {
        "representation_blueprint",
        "validation_blueprint",
        "calibration_blueprint",
        "inference_blueprint",
    }
    prompt = pack["prompt_text"]
    assert "Composable structural blueprints" in prompt
    assert "separate view encoders" in prompt
    assert "a target-task output artifact" in prompt
    assert "source artifact redacted" not in prompt
    assert "0.0001" not in prompt
    assert "55613290" not in prompt
    assert (
        "bb19d7d42b8e1923825f462dc7b42a033381488e8766f90af50166aee996f6d4" not in prompt
    )
    assert "990 training rows" not in prompt
    assert "594 test rows" not in prompt
    assert "99-way" not in prompt
    assert "target-derived capacity" in prompt
    assert "print('exact replay')" not in prompt
    assert "print('recipe capsule')" not in prompt
    assert "Acer" not in prompt
    assert "[[1.0, 0.0]]" not in prompt
    assert blueprint["source_score_inherited"] is False
    assert blueprint["source_code_exposed"] is False
    assert blueprint["source_artifact_exposed"] is False
    assert pack["visibility_safety_gate"]["source_artifact_fields_exposed"] == 0


def test_architecture_projection_is_draft_only_and_debug_remains_l3_only():
    from agents.memory.cross_task_transfer import build_transfer_pack

    pack = build_transfer_pack(
        _nodes(),
        _policy(architecture_transfer_enabled=True),
        target_task_id="uci-one-hundred-leaves",
        stage="debug",
        task_description="100-class leaf descriptor classification",
        query_text="ClassOrderMismatch",
    )

    assert pack["final_prompt_candidate_ids"] == ["repair::leaf::001"]
    assert pack["selected_architectures"] == []
    assert "recipe::leaf::001" not in pack["prompt_text"]

    improve_pack = build_transfer_pack(
        _nodes(),
        _policy(architecture_transfer_enabled=True),
        target_task_id="uci-one-hundred-leaves",
        stage="improve",
        task_description="100-class leaf descriptor classification",
        query_text="improve OOF coverage",
    )
    assert improve_pack["selected_architectures"] == []
    assert improve_pack["final_prompt_candidate_ids"] == ["tactic::leaf::001"]


def test_runtime_normalized_l1_strategy_is_projected_as_l1_recipe():
    from agents.memory.cross_task_transfer import build_transfer_pack

    nodes = _nodes()
    normalized = dict(nodes["recipe::leaf::001"])
    normalized["abstraction_level"] = "L1_strategy"
    normalized["recipe_abstraction_level"] = "L1_recipe"
    nodes["recipe::leaf::001"] = normalized

    pack = build_transfer_pack(
        nodes,
        _policy(architecture_transfer_enabled=True),
        target_task_id="uci-one-hundred-leaves",
        stage="draft",
        task_description="multimodal leaf descriptor classification",
        query_text="fold OOF architecture",
    )

    assert pack["memory_transfer"]["selected_architecture_ids"] == [
        "recipe::leaf::001::representation_blueprint"
    ]
    assert pack["selected_architectures"][0]["abstraction_level"] == ("L1_recipe")


def test_l1_recipe_is_decomposed_and_generic_c11_cannot_take_representation_slot():
    from agents.memory.cross_task_transfer import project_transfer_candidates

    nodes = {
        "recipe::leaf::c11": {
            "id": "recipe::leaf::c11",
            "task_id": "leaf-classification",
            "abstraction_level": "L1_recipe",
            "title": "Fold ensemble with temperature and Sinkhorn",
            "method_family": "fold_oof_temperature_sinkhorn_checkpoint",
            "pipeline": {
                "feature_representation": (
                    "Preserve generic neural feature views and derive widths from target data."
                ),
                "model_stack": (
                    "Use retained fold checkpoints and auxiliary probability components."
                ),
                "training_protocol": "Train fresh fold models and select checkpoints.",
                "oof_protocol": "Fill every OOF row exactly once.",
                "ensemble_calibration": "Fit temperature and optional Sinkhorn on OOF.",
                "final_refit_inference": "Replay target fold checkpoints on test rows.",
            },
        },
        "recipe::leaf::b3": {
            "id": "recipe::leaf::b3",
            "task_id": "leaf-classification",
            "abstraction_level": "L1_recipe",
            "title": "EfficientNet-B3 descriptor fusion",
            "method_family": "efficientnet_b3_descriptor_fusion",
            "pipeline": {
                "feature_representation": (
                    "Fuse an EfficientNet-B3 image embedding with color and shape descriptors."
                ),
                "model_stack": "Use a regularized multimodal fusion classifier.",
                "ensemble_calibration": "Evaluate double temperature on OOF predictions.",
            },
        },
    }
    projection = project_transfer_candidates(
        nodes,
        _policy(architecture_transfer_enabled=True),
        target_task_id="plant-seedlings-classification",
        stage="draft",
        task_description="plant seedling image classification",
        query_text="EfficientNet color shape grouped folds calibration",
        all_safe_levels=True,
    )
    rows = projection["observed_candidates"]
    ids = {row["id"] for row in rows}
    assert "recipe::leaf::c11::representation_blueprint" not in ids
    assert "recipe::leaf::c11::validation_blueprint" in ids
    assert "recipe::leaf::c11::calibration_blueprint" in ids
    assert "recipe::leaf::c11::inference_blueprint" in ids
    assert "recipe::leaf::b3::representation_blueprint" in ids
    b3_representation = next(
        row for row in rows if row["id"] == "recipe::leaf::b3::representation_blueprint"
    )
    assert set(b3_representation["portable_text"]["components"]) == {
        "feature_representation",
        "model_stack",
    }
    assert (
        "double temperature"
        not in json.dumps(
            b3_representation["portable_text"], ensure_ascii=False
        ).lower()
    )


def test_architecture_transfer_never_activates_for_exact_source_task():
    from agents.memory.cross_task_transfer import build_transfer_pack

    pack = build_transfer_pack(
        _nodes(),
        _policy(architecture_transfer_enabled=True),
        target_task_id="leaf-classification",
        stage="draft",
        task_description="source task",
        query_text="fold architecture",
    )

    assert pack["memory_transfer"]["activated"] is False
    assert pack["memory_transfer"]["host_decision"]["reason"] == (
        "exact_task_must_use_existing_replay_path"
    )
    assert pack["selected_items"] == []
    assert pack["prompt_text"] == ""


def test_policy_reads_architecture_channel_without_changing_allowed_sop_levels():
    from agents.memory.cross_task_transfer import CrossTaskTransferPolicy

    config = SimpleNamespace(
        cross_task_transfer_enabled=True,
        cross_task_transfer_source_task_id="leaf-classification",
        cross_task_transfer_source_task_type="leaf_descriptor_multiclass",
        cross_task_transfer_target_task_type="leaf_descriptor_multiclass",
        cross_task_transfer_allowed_levels=["L2_tactic", "L3_repair"],
        cross_task_transfer_max_items=6,
        cross_task_architecture_transfer_enabled=True,
        cross_task_architecture_max_items=1,
    )

    policy = CrossTaskTransferPolicy.from_config(config)
    assert policy.architecture_transfer_enabled is True
    assert policy.architecture_max_items == 1
    assert policy.allowed_levels == ("L2_tactic", "L3_repair")


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


def test_independent_target_novel_never_enters_source_end2end_controller():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    class ForbiddenEnd2EndController:
        def retrieve(self, *_args, **_kwargs):
            raise AssertionError(
                "source End2End retrieval must not run for target Novel"
            )

    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer.nodes = _nodes()
    layer.cross_task_transfer_policy = _policy()
    layer.end2end_controller = ForbiddenEnd2EndController()
    layer.prospective_audit_logger = None
    layer.retrieval_control = "dynamic_agentic"
    layer._trace_local = threading.local()
    layer._last_agentic_pack = {}

    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="uci-one-hundred-leaves",
        task_desc="100-class leaf descriptors",
        query_parts=["independent target hypothesis"],
        draft_role="novel_exploration",
    )

    assert text == ""
    assert refs == []
    pack = layer.current_navigation_pack()
    assert pack["schema"] == "stage_hybrid_role_policy_abstention_v1"
    assert pack["draft_role"] == "novel_exploration"
    assert pack["role_policy_abstention"]["reason"] == (
        "cross_task_transfer_independent_target_no_source_memory"
    )


def test_transfer_debug_uses_score_free_l3_projection_before_legacy_controller():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    class ForbiddenEnd2EndController:
        def retrieve(self, *_args, **_kwargs):
            raise AssertionError(
                "legacy End2End retrieval must not run for transfer Debug"
            )

    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer.nodes = _nodes()
    layer.cross_task_transfer_policy = _policy()
    layer.end2end_controller = ForbiddenEnd2EndController()
    layer.prospective_audit_logger = None
    layer._trace_local = threading.local()
    layer._last_agentic_pack = {}

    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="uci-one-hundred-leaves",
        task_desc="100-class leaf descriptors",
        query_parts=["ClassOrderMismatch"],
        draft_role="memory_transfer",
    )

    assert refs == ["repair::leaf::001"]
    assert "Align probability columns" in text
    assert "0.0002" not in text
    assert "source repair code must not transfer" not in text
    pack = layer.current_navigation_pack()
    assert pack["stage_route"]["stage"] == "debug"
    assert pack["memory_transfer"]["source_score_inheritance_allowed"] is False


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
