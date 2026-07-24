from __future__ import annotations

from authority.models import GenerationStage, Operation
from tests.authority.sop_visibility_helpers import (
    AUDIT_CLAUSE_ID,
    AUDIT_TEXT,
    FORBIDDEN_SCORE_TEXT,
    MIXED_SOP_ID,
    REPAIR_CLAUSE_ID,
    REPAIR_TEXT,
    SCORE_CLAUSE_ID,
    build_mixed_authority,
    make_stage_layer,
    visibility_request,
)


def test_mixed_value_debug_retains_repair_and_warning_without_score(tmp_path) -> None:
    engine, ref = build_mixed_authority()
    layer = make_stage_layer(tmp_path, engine, ref)
    request = visibility_request(
        ref,
        Operation.DEBUG_HYPOTHESIS,
        generation_stage=GenerationStage.DEBUG,
    )
    pack = layer._hybrid_pack(
        stage="debug",
        task_id="task-1",
        task_desc="tabular OOF alignment failure",
        query_text="repair OOF sample_id alignment",
        visibility_request=request,
    )
    prompt = layer._format_hybrid_pack(pack)
    visibility = layer.current_visibility_pack()

    assert visibility is not None
    assert set(visibility.effective_clause_ids) == {
        REPAIR_CLAUSE_ID,
        AUDIT_CLAUSE_ID,
    }
    oracle_debug_clauses = {REPAIR_CLAUSE_ID, AUDIT_CLAUSE_ID}
    retention = len(oracle_debug_clauses & set(visibility.effective_clause_ids)) / len(
        oracle_debug_clauses
    )
    assert retention == 1.0
    assert visibility.suppressed_clause_refs == [SCORE_CLAUSE_ID]
    assert [warning["clause_id"] for warning in pack["visibility_warnings"]] == [
        AUDIT_CLAUSE_ID
    ]
    assert REPAIR_TEXT in prompt
    assert AUDIT_TEXT in prompt
    assert FORBIDDEN_SCORE_TEXT not in prompt
    assert pack["visibility_safety_gate"]["unauthorized_prompt_exposure"] == 0
    assert pack["visibility_safety_gate"]["unauthorized_activation"] == 0
    assert MIXED_SOP_ID in {
        candidate["id"] for candidate in pack["direct_sop_candidates"]
    }


def test_mixed_value_rank_has_no_embedding_rrf_prompt_or_token_influence(tmp_path) -> None:
    engine, ref = build_mixed_authority()
    layer = make_stage_layer(
        tmp_path,
        engine,
        ref,
        edge_kind="authorized_distills_to",
        edge_outcome="allow",
    )
    request = visibility_request(
        ref,
        Operation.RANK,
        generation_stage=GenerationStage.IMPROVE,
    )
    ranking = layer.rank_sop_hybrid(
        stage="improve",
        task_id="task-1",
        task_desc="tabular model improvement",
        query_text=FORBIDDEN_SCORE_TEXT,
        limit=6,
        visibility_request=request,
    )
    trace = ranking["visibility_trace"]

    assert ranking["direct_sop_candidates"] == []
    assert ranking["tree_projected_sop_ids"] == []
    assert ranking["fused_sop_candidates"] == []
    assert ranking["visible_clause_ids"] == []
    assert SCORE_CLAUSE_ID not in trace["embedding_candidate_clause_ids"]
    assert SCORE_CLAUSE_ID not in trace["rrf_eligible_clause_ids"]
    assert trace["rendered_token_count"] == 0
    assert trace["empty_pack"] is True
    assert FORBIDDEN_SCORE_TEXT not in str(ranking["fused_sop_candidates"])


def test_inspect_shows_all_mixed_clauses_without_adoption_permission(tmp_path) -> None:
    engine, ref = build_mixed_authority()
    layer = make_stage_layer(tmp_path, engine, ref)
    request = visibility_request(
        ref,
        Operation.INSPECT,
        generation_stage=GenerationStage.DEBUG,
    )
    visibility = layer._prepare_visibility(
        stage="debug",
        task_id="task-1",
        task_desc="inspect provenance",
        request=request,
    )

    assert set(visibility.effective_clause_ids) == {
        REPAIR_CLAUSE_ID,
        AUDIT_CLAUSE_ID,
        SCORE_CLAUSE_ID,
    }
    assert visibility.visible_positive_clauses == []
    assert visibility.visible_diagnostic_clauses == []
    assert {clause.clause_id for clause in visibility.warning_clauses} == set(
        visibility.effective_clause_ids
    )
    assert all(
        decision["reason"] == "inspect_navigation_only"
        for decision in visibility.visibility_trace["clause_decisions"].values()
    )
    assert visibility.authority_decision_refs == []


def test_empty_visible_pack_reaches_agent_as_traced_abstention_without_legacy_fallback(
    tmp_path,
) -> None:
    engine, ref = build_mixed_authority()
    layer = make_stage_layer(tmp_path, engine, ref)
    # Isolate the SOP consumer boundary: an empty authorized SOP projection
    # must not be replaced by the legacy container or a Tree-only fallback.
    layer.retrieval_control = "sop_only"
    request = visibility_request(
        ref,
        Operation.RANK,
        generation_stage=GenerationStage.IMPROVE,
    )

    prompt, refs = layer.retrieve_for_node(
        stage="improve",
        task_id="task-1",
        task_desc="rank historical scores",
        query_parts=[FORBIDDEN_SCORE_TEXT],
        visibility_request=request,
    )
    navigation = layer.current_navigation_pack()
    visibility = layer.current_visibility_pack()

    assert visibility is not None
    assert visibility.visibility_trace["empty_pack"] is True
    assert navigation["visible_clause_ids"] == []
    assert navigation["selected_sop_gateways"] == []
    assert navigation["sop_only_candidates"] == []
    assert navigation["fused_execution_candidates"] == []
    assert refs == []
    assert FORBIDDEN_SCORE_TEXT not in prompt
    assert navigation["visibility_abstention"] == {
        "status": "abstain",
        "reason": "empty_visible_pack",
        "legacy_fallback_used": False,
        "warning_preserved": True,
    }
    assert navigation["visibility_trace"]["consumer_disposition"] == navigation[
        "visibility_abstention"
    ]
    assert "No authorized memory clauses are available" in prompt
