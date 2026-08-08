from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.coder.diff_coder.apply import apply_diff_with_retry
from agents.memory.experiment_r_router import _task_match_audit
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
