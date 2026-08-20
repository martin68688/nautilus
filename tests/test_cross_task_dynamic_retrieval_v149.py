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
        "transition::leaf::fixed-fold-state": {
            "id": "transition::leaf::fixed-fold-state",
            "type": "Transition",
            "task_id": "leaf-classification",
            "outcome": "metric_improved",
            "stage_pair": "improve->improve",
            "text": json.dumps(
                {
                    "reason": (
                        "Preserve fold-local preprocessing state before target "
                        "OOF fusion; source score 0.001234 and "
                        "`secret_transition_call()` must be removed."
                    )
                }
            ),
            "implementation_repair_capsule": {
                "before_code": "print('secret before body')",
                "after_code": "print('secret after body')",
                "unified_diff": "+ secret transition source diff",
            },
        },
        "run-node::leaf::oof-module": {
            "id": "run-node::leaf::oof-module",
            "type": "RunNode",
            "task_id": "leaf-classification",
            "is_buggy": False,
            "is_valid": True,
            "implementation_capsule": {
                "code": (
                    "import numpy as np\n"
                    "from sklearn.model_selection import StratifiedKFold\n\n"
                    "def build_oof_views(train_x, fold_ids, *, calibrate=True):\n"
                    "    secret_module_body = 'source-only-literal'\n"
                    "    return np.asarray(train_x)\n\n"
                    "def align_probabilities(probabilities, class_mapping):\n"
                    "    return probabilities\n\n"
                    "class TargetFusion:\n"
                    "    def __init__(self, view_names):\n"
                    "        self.view_names = view_names\n"
                    "    def fit(self, oof_probabilities, labels):\n"
                    "        return self\n"
                )
            },
        },
        "run-node::leaf::buggy-module": {
            "id": "run-node::leaf::buggy-module",
            "type": "RunNode",
            "task_id": "leaf-classification",
            "is_buggy": True,
            "is_valid": False,
            "implementation_capsule": {
                "code": (
                    "def unsafe_buggy_interface(leaked_source_value):\n"
                    "    return 'buggy-source-body'\n"
                )
            },
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
        "improvement_transition": (
            "fold preprocessing improvement",
            ["fold", "preprocessing"],
        ),
        "module_interface": (
            "oof fusion interface",
            ["oof", "fusion"],
        ),
    }
    return {
        "action": "search",
        "reason": "broad projected five-granularity search",
        "information_need": (
            "architecture tactics failures transitions and module interfaces"
        ),
        "allocation": {granularity: 0.2 for granularity in queries},
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
            "architecture_blueprint": 0.2,
            "portable_tactic": 0.2,
            "portable_repair": 0.2,
            "improvement_transition": 0.2,
            "module_interface": 0.2,
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
        assert "secret before body" not in serialized_prompt
        assert "secret after body" not in serialized_prompt
        assert "secret transition source diff" not in serialized_prompt
        assert "secret_transition_call" not in serialized_prompt
        assert "secret_module_body" not in serialized_prompt
        assert "source-only-literal" not in serialized_prompt
        assert "class_mapping" not in serialized_prompt
        assert "unsafe_buggy_interface" not in serialized_prompt
        assert "buggy-source-body" not in serialized_prompt
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
        "improvement_transition": 1,
        "module_interface": 1,
    }
    assert pack["memory_transfer"]["selected_level_counts"] == {
        "architecture_blueprint": 1,
        "portable_tactic": 1,
        "portable_repair": 0,
        "improvement_transition": 0,
        "module_interface": 0,
    }
    assert set(pack["final_prompt_candidate_ids"]) == {
        "recipe::leaf::fold",
        "tactic::leaf::oof",
    }
    assert [row["id"] for row in pack["final_prompt_candidates"]] == (
        pack["final_prompt_candidate_ids"]
    )
    assert pack["evidence_resolver"]["selected_ids"] == (
        pack["final_prompt_candidate_ids"]
    )
    serialized_pack = json.dumps(pack, sort_keys=True, ensure_ascii=False)
    assert "tactic::foreign" not in serialized_pack
    assert "secret transition source diff" not in serialized_pack
    assert "secret_module_body" not in serialized_pack
    assert "source-only-literal" not in serialized_pack
    assert "unsafe_buggy_interface" not in serialized_pack
    assert "buggy-source-body" not in serialized_pack


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
        "improvement_transition": 0,
        "module_interface": 0,
    }
    assert pack["selected_architectures"] == []
    assert len(pack["selected_portable_items"]) == 2


def test_judge_can_select_transition_reason_and_code_free_module_interface():
    from agents.memory.cross_task_dynamic_retrieval import (
        build_dynamic_transfer_pack,
    )

    def query_fn(**kwargs):
        serialized_prompt = json.dumps(
            kwargs["system_message"], sort_keys=True, ensure_ascii=False
        )
        assert "secret before body" not in serialized_prompt
        assert "secret after body" not in serialized_prompt
        assert "secret transition source diff" not in serialized_prompt
        assert "secret_transition_call" not in serialized_prompt
        assert "secret_module_body" not in serialized_prompt
        assert "source-only-literal" not in serialized_prompt
        assert "class_mapping" not in serialized_prompt
        assert "unsafe_buggy_interface" not in serialized_prompt
        assert "buggy-source-body" not in serialized_prompt
        assert "0.001234" not in serialized_prompt
        if kwargs["func_spec"].name == "plan_projected_cross_task_memory_search":
            return _search_action()
        cards = json.loads(
            kwargs["system_message"]["authorized_projected_cards"]
        )
        return _judge_action(
            cards,
            {"improvement_transition", "module_interface"},
        )

    pack = build_dynamic_transfer_pack(
        _layer(query_fn, rounds=1),
        _policy(),
        target_task_id="uci-one-hundred-leaves",
        stage="improve",
        task_description="improve target OOF fusion implementation",
        query_text="fold preprocessing module interface",
    )

    assert pack["memory_transfer"]["selected_level_counts"] == {
        "architecture_blueprint": 0,
        "portable_tactic": 0,
        "portable_repair": 0,
        "improvement_transition": 1,
        "module_interface": 1,
    }
    prompt = pack["prompt_text"]
    assert "Preserve fold-local preprocessing state" in prompt
    assert "[source numeric redacted]" in prompt
    assert "[implementation detail redacted]" in prompt
    assert "build_oof_views(train_x, fold_ids, *, calibrate)" in prompt
    assert "align_probabilities(probabilities, source_specific_symbol)" in prompt
    assert "TargetFusion" in prompt
    assert "fit(self, oof_probabilities, labels)" in prompt
    assert "secret_module_body" not in prompt
    assert "source-only-literal" not in prompt
    assert "class_mapping" not in prompt


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


def test_adoption_durably_mirrors_dynamic_search_and_resolver_receipts():
    from agents.adoption import log_adoption
    from agents.memory.cross_task_dynamic_retrieval import (
        build_dynamic_transfer_pack,
    )

    def query_fn(**kwargs):
        if kwargs["func_spec"].name == "plan_projected_cross_task_memory_search":
            return _search_action()
        cards = json.loads(
            kwargs["system_message"]["authorized_projected_cards"]
        )
        return _judge_action(cards, {"architecture_blueprint"})

    pack = build_dynamic_transfer_pack(
        _layer(query_fn, rounds=1),
        _policy(),
        target_task_id="uci-one-hundred-leaves",
        stage="draft",
        task_description="leaf descriptor classification",
        query_text="fold architecture",
    )
    layer = SimpleNamespace(
        current_navigation_pack=lambda: pack,
        current_visibility_pack=lambda: None,
        experiment_r_enabled=True,
        memory_snapshot=None,
    )
    node = SimpleNamespace(
        id="node::dynamic-transfer",
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
    assert trace["memory_pack_schema"] == pack["schema"]
    assert trace["final_prompt_candidates"] == pack["final_prompt_candidates"]
    assert trace["evidence_resolver"] == pack["evidence_resolver"]
    assert trace["dynamic_retrieval"] == pack["dynamic_retrieval"]


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
