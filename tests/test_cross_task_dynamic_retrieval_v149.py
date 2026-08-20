from __future__ import annotations

import json
import threading
from types import SimpleNamespace


def _policy():
    from agents.memory.cross_task_transfer import CrossTaskTransferPolicy

    policy = CrossTaskTransferPolicy(
        enabled=True,
        source_task_id="leaf-classification",
        source_task_type="leaf_descriptor_multiclass",
        target_task_type="leaf_descriptor_multiclass",
        allowed_levels=("L2_tactic", "L3_repair"),
        max_items=6,
        architecture_transfer_enabled=True,
        architecture_max_items=1,
    )
    policy.validate()
    return policy


def _nodes():
    return {
        "recipe::leaf::fold": {
            "id": "recipe::leaf::fold",
            "task_id": "leaf-classification",
            # Exercise the published runtime normalization, not the raw name.
            "abstraction_level": "L1_strategy",
            "method_family": "fold_multiview_calibration",
            "title": "Fold multiview calibrated ensemble",
            "teacher_distilled_recipe": (
                "Train target-derived view encoders, fill OOF predictions, "
                "calibrate, and ensemble fold models."
            ),
            "pipeline": {
                "feature_representation": "Use target-derived descriptor views.",
                "model_stack": "Fuse compatible view encoders.",
                "training_protocol": "Train fresh target fold models.",
                "oof_protocol": "Fill every target OOF row exactly once.",
                "ensemble_calibration": "Calibrate only from target OOF outputs.",
                "final_refit_inference": "Average target-trained fold probabilities.",
            },
            "official_metric": 0.000123,
            "code": "print('raw source code must never reach an LLM')",
            "official_kaggle_ref": "55613290",
            "class_mapping": {"Acer": 0},
        },
        "recipe::leaf::tree": {
            "id": "recipe::leaf::tree",
            "task_id": "leaf-classification",
            "abstraction_level": "L1_recipe",
            "method_family": "tree_descriptor_stack",
            "title": "Tree descriptor stack",
            "teacher_distilled_recipe": "Stack target-trained descriptor trees.",
            "pipeline": {
                "feature_representation": "Use target descriptor blocks.",
                "model_stack": "Stack target-trained tree probabilities.",
            },
            "score": 0.001,
            "implementation_capsule": "secret source implementation",
        },
        "tactic::leaf::oof": {
            "id": "tactic::leaf::oof",
            "task_id": "leaf-classification",
            "abstraction_level": "L2_tactic",
            "method_family": "fold_multiview_calibration",
            "parent_method_families": ["fold_multiview_calibration"],
            "title": "Complete OOF coverage",
            "instruction": "Write each target validation row once.",
            "when_to_use": "fold model selection and calibration",
            "official_metric": 0.000456,
            "implementation_capsule": "source tactic code",
        },
        "tactic::leaf::tree": {
            "id": "tactic::leaf::tree",
            "task_id": "leaf-classification",
            "abstraction_level": "L2_tactic",
            "method_family": "tree_descriptor_stack",
            "parent_method_families": ["tree_descriptor_stack"],
            "title": "Tree-only stacking",
            "instruction": "Stack compatible target-trained tree models.",
            "when_to_use": "tree descriptor architecture",
        },
        "repair::leaf::class-order": {
            "id": "repair::leaf::class-order",
            "task_id": "leaf-classification",
            "abstraction_level": "L3_repair",
            "title": "Align probability columns",
            "when_to_use": "predict_proba omits or reorders classes",
            "failure_signature": {"exception_names": ["ClassOrderMismatch"]},
            "repair_action": {
                "summary": "Map model classes into the target class order."
            },
            "code": "source repair code",
            "metric": 0.002,
        },
        "tactic::foreign": {
            "id": "tactic::foreign",
            "task_id": "spooky-author-identification",
            "abstraction_level": "L2_tactic",
            "title": "Character ngrams",
            "instruction": "Use character features.",
        },
    }


def _search_action():
    queries = {
        "architecture_blueprint": ("fold ensemble", ["fold", "ensemble"]),
        "portable_tactic": ("oof coverage", ["oof", "coverage"]),
        "portable_repair": (
            "class order mismatch",
            ["classordermismatch", "class"],
        ),
    }
    return {
        "action": "search",
        "reason": "broad projected L1 L2 L3 search",
        "information_need": "architecture tactics and observed failures",
        "allocation": {
            "architecture_blueprint": 0.34,
            "portable_tactic": 0.33,
            "portable_repair": 0.33,
        },
        "queries": [
            {
                "granularity": granularity,
                "query": query,
                "terms": terms,
                "top_k": 8,
                "reason": "cover safely projected memory",
            }
            for granularity, (query, terms) in queries.items()
        ],
    }


def _finish_action():
    return {
        "action": "finish",
        "reason": "projected evidence is sufficient",
        "information_need": "none",
        "allocation": {
            "architecture_blueprint": 0.34,
            "portable_tactic": 0.33,
            "portable_repair": 0.33,
        },
        "queries": [],
    }


def _judge_action(cards, selected_granularities):
    selected = [
        row["ref"]
        for row in cards
        if row["granularity"] in set(selected_granularities)
    ]
    return {
        "decision": "select",
        "selected_refs": selected,
        "reason": "target-compatible projected cards",
        "assessments": [
            {
                "ref": row["ref"],
                "applicability": 0.9,
                "target_adaptability": 0.95,
                "coherence": 0.9,
                "contradiction": False,
                "confidence": 0.92,
                "reason": "compatible with target context",
            }
            for row in cards
        ],
    }


def _layer(query_fn, *, rounds=2):
    return SimpleNamespace(
        nodes=_nodes(),
        cfg=None,
        _experiment_r_agentic_query_fn=query_fn,
        cross_task_dynamic_search_rounds=rounds,
        cross_task_dynamic_per_query_limit=8,
        cross_task_dynamic_safe_supplement_per_query=2,
        cross_task_dynamic_max_candidates=32,
        cross_task_dynamic_judge_candidate_limit=18,
        cross_task_dynamic_context_chars=12000,
        cross_task_dynamic_trace_history=6,
        cross_task_dynamic_search_max_tokens=3000,
        cross_task_dynamic_judge_max_tokens=7000,
        cross_task_dynamic_allow_abstention=False,
        cross_task_dynamic_stage_selection_caps={
            "draft": 7,
            "improve": 6,
            "debug": 4,
        },
        cross_task_dynamic_max_selected_architectures=1,
    )


def test_dynamic_search_and_judge_see_only_irreversible_safe_projection():
    from agents.memory.cross_task_dynamic_retrieval import (
        PACK_SCHEMA,
        build_dynamic_transfer_pack,
    )

    search_calls = 0
    judge_calls = 0

    def query_fn(**kwargs):
        nonlocal search_calls, judge_calls
        serialized_prompt = json.dumps(
            kwargs["system_message"], sort_keys=True, ensure_ascii=False
        )
        assert "raw source code must never reach an LLM" not in serialized_prompt
        assert "secret source implementation" not in serialized_prompt
        assert "source tactic code" not in serialized_prompt
        assert "source repair code" not in serialized_prompt
        assert "0.000123" not in serialized_prompt
        assert "0.000456" not in serialized_prompt
        assert "55613290" not in serialized_prompt
        assert "Acer" not in serialized_prompt

        if kwargs["func_spec"].name == "plan_projected_cross_task_memory_search":
            search_calls += 1
            schema = kwargs["func_spec"].json_schema
            assert '"fields"' not in json.dumps(schema)
            if search_calls == 1:
                return _search_action()
            return _finish_action()

        judge_calls += 1
        cards = json.loads(
            kwargs["system_message"]["authorized_projected_cards"]
        )
        assert all("candidate_id" not in row for row in cards)
        assert [row["ref"] for row in cards] == [
            f"C{index:02d}" for index in range(1, len(cards) + 1)
        ]
        return _judge_action(
            cards,
            {"architecture_blueprint", "portable_tactic"},
        )

    pack = build_dynamic_transfer_pack(
        _layer(query_fn),
        _policy(),
        target_task_id="uci-one-hundred-leaves",
        stage="draft",
        task_description="leaf descriptor classification with fold validation",
        query_text="build OOF calibrated ensemble",
    )

    assert pack["schema"] == PACK_SCHEMA
    assert search_calls == 2
    assert judge_calls == 1
    assert pack["stage_route"]["route"] == (
        "projected_multiround_search_judge_resolver"
    )
    assert pack["safety_gate"]["irreversible_projection_before_llm"] is True
    assert pack["safety_gate"]["search_agent_raw_graph_access"] is False
    assert pack["safety_gate"]["judge_raw_graph_access"] is False
    assert pack["dynamic_retrieval"]["search"]["authorized_counts"] == {
        "architecture_blueprint": 2,
        "portable_tactic": 2,
        "portable_repair": 1,
    }
    assert pack["memory_transfer"]["selected_level_counts"] == {
        "architecture_blueprint": 1,
        "portable_tactic": 1,
        "portable_repair": 0,
    }
    assert set(pack["final_prompt_candidate_ids"]) == {
        "recipe::leaf::fold",
        "tactic::leaf::oof",
    }
    assert "tactic::foreign" not in json.dumps(pack)


def test_draft_judge_can_select_only_l2_without_fixed_l1_plus_six():
    from agents.memory.cross_task_dynamic_retrieval import (
        build_dynamic_transfer_pack,
    )

    def query_fn(**kwargs):
        if kwargs["func_spec"].name == "plan_projected_cross_task_memory_search":
            return _search_action()
        cards = json.loads(
            kwargs["system_message"]["authorized_projected_cards"]
        )
        return _judge_action(cards, {"portable_tactic"})

    pack = build_dynamic_transfer_pack(
        _layer(query_fn, rounds=1),
        _policy(),
        target_task_id="uci-one-hundred-leaves",
        stage="draft",
        task_description="leaf descriptor classification",
        query_text="OOF coverage",
    )

    assert pack["memory_transfer"]["selected_level_counts"] == {
        "architecture_blueprint": 0,
        "portable_tactic": 2,
        "portable_repair": 0,
    }
    assert pack["selected_architectures"] == []
    assert len(pack["selected_portable_items"]) == 2


def test_resolver_keeps_one_architecture_and_suppresses_incompatible_tactic():
    from agents.memory.cross_task_dynamic_retrieval import (
        build_dynamic_transfer_pack,
    )

    def query_fn(**kwargs):
        if kwargs["func_spec"].name == "plan_projected_cross_task_memory_search":
            return _search_action()
        cards = json.loads(
            kwargs["system_message"]["authorized_projected_cards"]
        )
        return _judge_action(
            cards,
            {"architecture_blueprint", "portable_tactic"},
        )

    pack = build_dynamic_transfer_pack(
        _layer(query_fn, rounds=1),
        _policy(),
        target_task_id="uci-one-hundred-leaves",
        stage="draft",
        task_description="fold multiview leaf classification",
        query_text="fold ensemble OOF coverage",
    )

    resolver = pack["dynamic_retrieval"]["resolver"]
    assert len(resolver["selected_architecture_ids"]) == 1
    selected_architecture = resolver["selected_architecture_ids"][0]
    if selected_architecture == "recipe::leaf::fold":
        incompatible_tactic = "tactic::leaf::tree"
    else:
        incompatible_tactic = "tactic::leaf::oof"
    assert incompatible_tactic not in pack["final_prompt_candidate_ids"]
    assert {
        row["reason"] for row in resolver["suppressed"]
    } >= {
        "resolver_multiple_architecture_families",
        "resolver_parent_method_family_mismatch",
    }


def test_stage_layer_dispatches_dynamic_route_before_legacy_controller():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    class ForbiddenEnd2EndController:
        def retrieve(self, *_args, **_kwargs):
            raise AssertionError("legacy controller must not run")

    def query_fn(**kwargs):
        if kwargs["func_spec"].name == "plan_projected_cross_task_memory_search":
            return _search_action()
        cards = json.loads(
            kwargs["system_message"]["authorized_projected_cards"]
        )
        return _judge_action(cards, {"portable_repair"})

    dynamic = _layer(query_fn, rounds=1)
    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer.nodes = dynamic.nodes
    layer.cfg = None
    layer.cross_task_transfer_policy = _policy()
    layer.cross_task_dynamic_retrieval_enabled = True
    for key, value in dynamic.__dict__.items():
        if key.startswith("cross_task_dynamic_") or key == (
            "_experiment_r_agentic_query_fn"
        ):
            setattr(layer, key, value)
    layer.end2end_controller = ForbiddenEnd2EndController()
    layer.prospective_audit_logger = None
    layer._trace_local = threading.local()
    layer._last_agentic_pack = {}

    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="uci-one-hundred-leaves",
        task_desc="leaf descriptor classification",
        query_parts=["ClassOrderMismatch"],
        draft_role="memory_transfer",
    )

    assert refs == ["repair::leaf::class-order"]
    assert "Align probability columns" in text
    pack = layer.current_navigation_pack()
    assert pack["schema"] == "mlevolve_cross_task_dynamic_transfer_pack_v1"
    assert pack["stage_route"]["stage"] == "debug"

