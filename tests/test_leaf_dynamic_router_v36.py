from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.coder.diff_coder.apply import apply_diff_with_retry
from agents.memory.experiment_r_router import (
    _debug_repair_evidence,
    _portable_debug_anchor_match,
    _task_match_audit,
    _tiered_debug_runforest_rows,
)
from config import Config, _load_cfg
from tests.test_experiment_r_dynamic_routing import _layer


V37_CONFIG = (
    ROOT
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "systems_v37"
    / "dynamic_hybrid.yaml"
)


def _enable_sparse_v36(layer) -> None:
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_agentic_max_steps = 1
    layer.experiment_r_flexible_selection_enabled = True
    layer.experiment_r_allow_agent_abstention = True
    layer.experiment_r_stage_selection_caps = {
        "draft": 6,
        "improve": 3,
        "debug": 2,
    }
    layer.experiment_r_debug_causal_only = True
    layer.experiment_r_same_task_best_pin_stages = {"draft"}


def _enable_tiered_v41(layer) -> None:
    _enable_sparse_v36(layer)
    layer.experiment_r_debug_tiered_retrieval_enabled = True
    layer.experiment_r_debug_portable_runtime_enabled = True
    layer.experiment_r_debug_portable_max_candidates = 2


def _add_debug_transition(
    layer,
    *,
    transition_id: str,
    task: str,
    parent_id: str,
    child_id: str,
    failure: str,
    repair: str = "guard the failing runtime operation",
) -> None:
    layer.nodes[parent_id] = {
        "id": parent_id,
        "type": "RunNode",
        "run_id": f"run::{transition_id}",
        "run_short_id": f"run::{transition_id}",
        "task": task,
        "stage": "debug",
        "is_buggy": True,
        "is_valid": False,
        "analysis": failure,
        "leakage_audit": {
            "status": "clean",
            "memory_disposition": "positive_eligible",
            "paper_grade_eligible": True,
            "rank_eligible": True,
        },
    }
    layer.nodes[child_id] = {
        "id": child_id,
        "type": "RunNode",
        "run_id": f"run::{transition_id}",
        "run_short_id": f"run::{transition_id}",
        "task": task,
        "stage": "debug",
        "metric": 0.25,
        "metric_improvement": 0.10,
        "is_buggy": False,
        "is_valid": True,
        "plan": repair,
        "analysis": "clean validation completed",
        "leakage_audit": {
            "status": "clean",
            "memory_disposition": "positive_eligible",
            "paper_grade_eligible": True,
            "rank_eligible": True,
        },
    }
    before = "value = unsafe_runtime_call(x)\n"
    after = "value = safe_runtime_call(x)\n"
    layer.nodes[transition_id] = {
        "id": transition_id,
        "type": "Transition",
        "run_id": f"run::{transition_id}",
        "run_short_id": f"run::{transition_id}",
        "task": task,
        "parent_node_id": parent_id,
        "child_node_id": child_id,
        "stage_pair": "debug->debug",
        "outcome": "debug_fixed",
        "metric_improvement": 0.10,
        "parent_buggy": True,
        "child_buggy": False,
        "text": repair,
        "implementation_repair_capsule": {
            "before_code": before,
            "after_code": after,
            "before_code_sha256": hashlib.sha256(before.encode()).hexdigest(),
            "after_code_sha256": hashlib.sha256(after.encode()).hexdigest(),
            "unified_diff": "-value = unsafe_runtime_call(x)\n+value = safe_runtime_call(x)\n",
        },
    }
    layer._transitions.append(transition_id)
    layer._node_tokens[parent_id] = set()
    layer._node_tokens[child_id] = set()
    layer._node_tokens[transition_id] = set()


def _add_l3_sop(layer, *, sop_id: str, transition_id: str, task: str) -> None:
    layer.nodes[sop_id] = {
        "id": sop_id,
        "type": "SOP",
        "abstraction_level": "L3_repair",
        "evidence_status": "accepted_clean_repair",
        "task_id": task,
        "task_type": layer._task_type_for_query(task, "text classification"),
        "title": "historical root-cause repair",
        "runtime_stage": "training",
        "method_family": "test_model",
        "when_to_use": "only for the exact historical root cause",
        "failure_signature": {"exception": "IndexError"},
        "repair_action": {"action": "guard the failing index"},
        "supporting_transition_ids": [transition_id],
        "decision_stages": ["debug"],
        "task_families": ["text_classification"],
    }
    layer._sops.append(sop_id)
    layer._node_tokens[sop_id] = set()


def test_v36_config_freezes_sparse_causal_atomic_leaf_policy():
    raw_cfg = _load_cfg(V37_CONFIG, use_cli_args=False)
    raw_cfg.exp_name = "leaf-v37-schema-test"
    cfg = OmegaConf.merge(
        OmegaConf.structured(Config),
        raw_cfg,
    )
    ext = cfg.external_skill_memory
    assert ext.experiment_r_flexible_selection_enabled is True
    assert ext.experiment_r_allow_agent_abstention is True
    assert dict(ext.experiment_r_stage_selection_caps) == {
        "draft": 6,
        "improve": 3,
        "debug": 2,
    }
    assert ext.experiment_r_debug_causal_only is True
    assert list(ext.experiment_r_same_task_best_pin_stages) == ["draft"]
    assert ext.experiment_r_l3_agent_match_enabled is True
    assert ext.experiment_r_atomic_actuation_enabled is True
    assert ext.experiment_r_improve_max_modules == 2
    assert ext.experiment_r_improve_max_patches == 6
    assert ext.experiment_r_debug_max_patches == 3


def test_v36_leaf_task_gate_does_not_treat_taxi_as_same_type(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")

    audit = _task_match_audit(
        layer,
        target_task_id="leaf-classification",
        target_task_desc="multimodal leaf classification",
        source_task_id="new-york-city-taxi-fare-prediction",
    )

    assert audit == {
        "task_match": 0.0,
        "task_scope": "different_task_type",
    }


def test_v36_improve_agent_may_select_one_without_source_quota_or_pin(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_sparse_v36(layer)
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: {
        "action": "finish",
        "reason": "one exact-task execution is sufficient",
        "selected_ids": ["n1"],
    }

    text, refs = layer.retrieve_for_node(
        stage="improve",
        task_id="task",
        task_desc="text classification",
        query_parts=["improve one component"],
        draft_role="memory_reproduction",
    )
    pack = layer.current_navigation_pack()

    assert text and refs == ["n1"]
    assert pack["final_prompt_candidate_ids"] == ["n1"]
    assert pack["stage_route"]["realized_slots"] == {
        "sop": 0,
        "runforest": 1,
    }
    assert pack["retrieval_agent"]["selection_contract"][
        "selection_semantics"
    ] == "agent_variable_cardinality_with_explicit_abstention_v1"
    assert pack["retrieval_agent"]["same_task_best_first"]["enforced"] is False
    assert pack["pre_gate_raw_candidates"]
    assert pack["pre_gate_summary"]["stored_near_miss_count"] > 0
    assert any(
        row["gate_reason"] == "run_node_not_rank_eligible"
        for row in pack["pre_gate_raw_candidates"]
    )


def test_v36_improve_explicit_abstention_does_not_backfill_nonempty_pool(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_sparse_v36(layer)
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: {
        "action": "finish",
        "reason": "none of the observed memories supports an atomic improvement",
        "selected_ids": [],
    }

    text, refs = layer.retrieve_for_node(
        stage="improve",
        task_id="task",
        task_desc="text classification",
        query_parts=["avoid an unsupported rewrite"],
        draft_role="memory_reproduction",
    )
    pack = layer.current_navigation_pack()

    assert text == "" and refs == []
    assert pack["router_activation"]["candidate_pool_nonempty"] is True
    assert pack["router_activation"]["status"] == "abstain"
    assert pack["retrieval_agent"]["agent_abstained"] is True
    assert pack["memory_abstention"]["decision_authority"] == "retrieval_agent"
    assert "none of the observed" in pack["memory_abstention"]["reason"]
    assert pack["final_prompt_candidate_ids"] == []


def test_v36_debug_without_causal_l3_match_abstains_without_generic_backfill(
    tmp_path,
):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_sparse_v36(layer)
    layer.experiment_r_l3_agent_match_enabled = True

    def unexpected_general_agent_call(**_kwargs):
        raise AssertionError("general Retrieval Agent must not fill an empty Debug repair pool")

    layer._experiment_r_agentic_query_fn = unexpected_general_agent_call
    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="task",
        task_desc="text classification",
        query_parts=["IndexError: index 192 is out of bounds for axis 1"],
        draft_role="memory_reproduction",
    )
    pack = layer.current_navigation_pack()

    assert text == "" and refs == []
    assert pack["candidate_pool"]["runforest_candidates"] == []
    assert pack["candidate_pool"]["sop_candidates"] == []
    assert pack["stage_route"]["tree_confidence"] == 0.0
    assert pack["stage_route"]["fallback_reason"] == (
        "no_causally_matched_debug_repair"
    )
    assert pack["retrieval_agent"]["agent_calls"] == 0
    assert pack["retrieval_agent"]["agent_abstained"] is True
    assert pack["retrieval_agent"]["same_task_best_first"]["enforced"] is False
    assert pack["final_prompt_candidate_ids"] == []
    assert pack["pre_gate_raw_candidates"]


def test_v41_l3_abstention_opens_task_local_tier_b_for_main_agent(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_tiered_v41(layer)
    _add_debug_transition(
        layer,
        transition_id="tier_b",
        task="task",
        parent_id="tier_b_parent",
        child_id="tier_b_child",
        failure="IndexError: index 192 is out of bounds for axis 1",
    )
    rows, audit = _tiered_debug_runforest_rows(
        layer,
        query_text="IndexError: index 192 is out of bounds for axis 1",
        task_id="task",
        limit=6,
        l3_agent_match={"decision": "abstain", "trace": []},
    )
    assert rows and rows[0]["id"] == "tier_b"
    assert rows[0]["debug_tier"] == "task_local_clean_transition"
    assert audit["task_local_safe_count"] == 1
    assert audit["main_retrieval_agent_input_count"] == 1
    assert audit["fallback_reason"] == "strict_l3_abstained_safe_backfill"


def test_v41_l3_agent_abstention_still_calls_main_agent_on_tier_b(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_tiered_v41(layer)
    _add_debug_transition(
        layer,
        transition_id="l3_assessed",
        task="task",
        parent_id="l3_assessed_parent",
        child_id="l3_assessed_child",
        failure="IndexError: a historical but different indexing root cause",
    )
    _add_l3_sop(
        layer,
        sop_id="l3_card",
        transition_id="l3_assessed",
        task="task",
    )
    _add_debug_transition(
        layer,
        transition_id="tier_b_after_l3_abstain",
        task="task",
        parent_id="tier_b_after_l3_parent",
        child_id="tier_b_after_l3_child",
        failure="IndexError: index 192 is out of bounds for axis 1",
    )
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_l3_agent_match_enabled = True
    layer.experiment_r_agentic_max_steps = 1
    calls: list[str] = []

    def query_fn(**kwargs):
        prompt = kwargs["system_message"]
        if "authorized_l3_candidates" in prompt:
            calls.append("l3")
            candidates = json.loads(prompt["authorized_l3_candidates"])
            return {
                "decision": "abstain",
                "selected_sop_id": "",
                "selected_transition_id": "",
                "final_confidence": 0.10,
                "reason": "the historical L3 card has a different root cause",
                "assessments": [
                    {
                        "sop_id": row["sop_id"],
                        "keyword_correspondence": 0.40,
                        "root_cause_equivalence": 0.10,
                        "runtime_stage_match": 0.80,
                        "contradiction": True,
                        "confidence": 0.10,
                        "reason": "different indexing contract",
                    }
                    for row in candidates
                ],
            }
        calls.append("main")
        known = json.loads(prompt["known_candidates"])
        assert any(
            row["id"] == "tier_b_after_l3_abstain" for row in known
        )
        return {
            "action": "finish",
            "reason": "the undistilled same-task repair matches",
            "selected_ids": ["tier_b_after_l3_abstain"],
        }

    layer._experiment_r_agentic_query_fn = query_fn
    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="task",
        task_desc="text classification",
        query_parts=["IndexError: index 192 is out of bounds for axis 1"],
        draft_role="memory_reproduction",
    )
    pack = layer.current_navigation_pack()
    assert calls == ["l3", "main"]
    assert refs == ["tier_b_after_l3_abstain"]
    assert "tier_b_after_l3_abstain" in text
    assert pack["l3_agent_match"]["decision"] == "abstain"
    assert pack["retrieval_agent"]["root_cause_agent_calls"] == 1
    assert pack["retrieval_agent"]["main_retrieval_agent_calls"] == 1
    assert pack["retrieval_agent"]["final_selection_authority"] == (
        "retrieval_agent"
    )
    assert pack["stage_route"]["fallback_reason"] is None


def test_v41_l3_abstention_opens_portable_tier_c_with_literal_anchor(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_tiered_v41(layer)
    _add_debug_transition(
        layer,
        transition_id="tier_c",
        task="other-task",
        parent_id="tier_c_parent",
        child_id="tier_c_child",
        failure="RuntimeError: torch.mean(LongTensor) is not implemented",
    )
    rows, audit = _tiered_debug_runforest_rows(
        layer,
        query_text="RuntimeError: torch.mean(LongTensor) is not implemented",
        task_id="task",
        limit=6,
        l3_agent_match={"decision": "abstain", "trace": []},
    )
    assert rows and rows[0]["id"] == "tier_c"
    assert rows[0]["debug_tier"] == "portable_runtime_repair"
    assert rows[0]["portable_runtime_authorized"] is True
    assert rows[0]["portable_anchor_match"]["shared_exception_names"] == [
        "runtimeerror"
    ]
    assert audit["portable_runtime_safe_count"] == 1
    assert _portable_debug_anchor_match(
        "RuntimeError: torch.mean(LongTensor) is not implemented",
        "RuntimeError: unrelated_function(FloatTensor) failed",
    )["authorized"] is False


def test_v41_tiered_agent_failure_uses_fallback_without_l3_confidence_failure(
    tmp_path,
):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_tiered_v41(layer)
    _add_debug_transition(
        layer,
        transition_id="tier_b_fallback",
        task="task",
        parent_id="tier_b_fallback_parent",
        child_id="tier_b_fallback_child",
        failure="IndexError: index 192 is out of bounds for axis 1",
    )
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_l3_agent_match_enabled = False
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("simulated retrieval outage")
    )
    layer.retrieve_for_node(
        stage="debug",
        task_id="task",
        task_desc="text classification",
        query_parts=["IndexError: index 192 is out of bounds for axis 1"],
        draft_role="memory_reproduction",
    )
    pack = layer.current_navigation_pack()
    assert pack["stage_route"]["fallback_reason"] == (
        "tiered_debug_safe_deterministic_fallback"
    )
    assert pack["candidate_pool"]["debug_candidate_tiers"][
        "strict_l3_confidence_below_threshold"
    ] is True
    assert pack["retrieval_agent"]["final_selection_authority"] == (
        "deterministic_fallback"
    )


def test_v41_debug_agent_can_explicitly_abstain_after_nonempty_tiered_pool(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_tiered_v41(layer)
    _add_debug_transition(
        layer,
        transition_id="tier_b_abstain",
        task="task",
        parent_id="tier_b_abstain_parent",
        child_id="tier_b_abstain_child",
        failure="IndexError: index 192 is out of bounds for axis 1",
    )
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_l3_agent_match_enabled = False
    layer.experiment_r_agentic_max_steps = 1
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: {
        "action": "finish",
        "reason": "candidate is not safe for this exact failure",
        "selected_ids": [],
    }
    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="task",
        task_desc="text classification",
        query_parts=["IndexError: index 192 is out of bounds for axis 1"],
        draft_role="memory_reproduction",
    )
    pack = layer.current_navigation_pack()
    assert text == "" and refs == []
    assert pack["router_activation"]["candidate_pool_nonempty"] is True
    assert pack["router_activation"]["status"] == "abstain"
    assert pack["retrieval_agent"]["main_retrieval_agent_calls"] == 1
    assert pack["retrieval_agent"]["final_selection_authority"] == (
        "retrieval_agent"
    )
    assert pack["memory_abstention"]["decision_authority"] == "retrieval_agent"


def test_v41_debug_evidence_distinguishes_full_diff_and_hash_bound_action(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _enable_tiered_v41(layer)
    _add_debug_transition(
        layer,
        transition_id="full_diff",
        task="task",
        parent_id="full_diff_parent",
        child_id="full_diff_child",
        failure="IndexError: index 192 is out of bounds for axis 1",
    )
    full, reason = _debug_repair_evidence(layer, "full_diff")
    assert reason == "safe_hash_bound_debug_repair"
    assert full is not None
    assert full["evidence_mode"] == "full_code_diff"

    _add_debug_transition(
        layer,
        transition_id="hash_bound",
        task="task",
        parent_id="hash_bound_parent",
        child_id="hash_bound_child",
        failure="IndexError: index 192 is out of bounds for axis 1",
    )
    before_hash = "a" * 64
    after_hash = "b" * 64
    layer.nodes["hash_bound"]["implementation_repair_capsule"] = {
        "before_code_sha256": before_hash,
        "after_code_sha256": after_hash,
    }
    layer._recipe_repair_evidence_by_transition["hash_bound"] = {
        "audit_status": "clean",
        "memory_disposition": "positive_eligible",
        "paper_grade_eligible": True,
        "rank_eligible": True,
        "failure_node_code_sha256": before_hash,
        "successful_node_code_sha256": after_hash,
        "failure_text": "the runtime index exceeded the feature width",
        "repair_action_text": "guard the index before feature lookup",
        "successful_execution_summary": "validation completed",
    }
    hash_bound, reason = _debug_repair_evidence(layer, "hash_bound")
    assert reason == "safe_hash_bound_debug_repair"
    assert hash_bound is not None
    assert hash_bound["evidence_mode"] == "hash_bound_repair_action_only"
    assert hash_bound["transition_evidence"]["unified_diff"] == ""

    layer.nodes["hash_bound"]["infrastructure_failure"] = True
    rejected, reason = _debug_repair_evidence(layer, "hash_bound")
    assert rejected is None and reason == "infrastructure_failure"


def test_atomic_diff_cap_retries_with_a_smaller_change():
    original = "x = 1\ny = 1\n"
    search = "<" * 7 + " SEARCH"
    separator = "=" * 7
    replace = ">" * 7 + " REPLACE"
    too_broad = f"""{search}
x = 1
{separator}
x = 2
{replace}
{search}
y = 1
{separator}
y = 2
{replace}
"""
    atomic = f"""{search}
x = 1
{separator}
x = 2
{replace}
"""
    retry_notes: list[str] = []

    def regenerate(_current_code: str, retry_note: str) -> str:
        retry_notes.append(retry_note)
        return atomic

    code, applied, _note = apply_diff_with_retry(
        too_broad,
        original,
        max_retries=2,
        regenerate_fn=regenerate,
        max_total_patches=1,
    )

    assert code == "x = 2\ny = 1\n"
    assert applied == 1
    assert retry_notes and "maximum is 1" in retry_notes[0]
