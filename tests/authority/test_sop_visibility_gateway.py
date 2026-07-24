from __future__ import annotations

import copy

from agents.memory.sop_visibility_gateway import SOPVisibilityGateway
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
    decision_snapshot,
    mixed_nodes,
    visibility_request,
)


def test_gateway_applies_clause_authority_before_text_materialization() -> None:
    engine, ref = build_mixed_authority()
    gateway = SOPVisibilityGateway(
        mixed_nodes(ref),
        mode="enforce",
        authority_engine=engine,
        decision_lookup=engine.decisions.get,
    )

    debug = gateway.evaluate(
        visibility_request(
            ref,
            Operation.DEBUG_HYPOTHESIS,
            generation_stage=GenerationStage.DEBUG,
        )
    )
    assert [clause.clause_id for clause in debug.visible_diagnostic_clauses] == [
        REPAIR_CLAUSE_ID
    ]
    assert [clause.clause_id for clause in debug.warning_clauses] == [
        AUDIT_CLAUSE_ID
    ]
    assert debug.visible_positive_clauses == []
    assert debug.suppressed_clause_refs == [SCORE_CLAUSE_ID]
    assert debug.effective_clause_ids == [AUDIT_CLAUSE_ID, REPAIR_CLAUSE_ID]
    rendered = debug.rendered_by_sop[MIXED_SOP_ID]
    assert REPAIR_TEXT in rendered["prompt_text"]
    assert AUDIT_TEXT in rendered["prompt_text"]
    assert FORBIDDEN_SCORE_TEXT not in rendered["prompt_text"]
    assert SCORE_CLAUSE_ID not in debug.visibility_trace[
        "embedding_candidate_clause_ids"
    ]
    assert SCORE_CLAUSE_ID not in debug.visibility_trace["rrf_eligible_clause_ids"]

    rank = gateway.evaluate(
        visibility_request(
            ref,
            Operation.RANK,
            generation_stage=GenerationStage.IMPROVE,
        )
    )
    assert rank.effective_clause_ids == []
    assert rank.effective_sop_ids == []
    assert rank.visibility_trace["empty_pack"] is True
    assert rank.visibility_trace["rendered_token_count"] == 0
    assert set(rank.suppressed_clause_refs) == {
        REPAIR_CLAUSE_ID,
        AUDIT_CLAUSE_ID,
        SCORE_CLAUSE_ID,
    }

    inspect = gateway.evaluate(
        visibility_request(
            ref,
            Operation.INSPECT,
            generation_stage=GenerationStage.DEBUG,
        )
    )
    assert inspect.suppressed_clause_refs == []
    assert {clause.clause_id for clause in inspect.warning_clauses} == {
        REPAIR_CLAUSE_ID,
        AUDIT_CLAUSE_ID,
        SCORE_CLAUSE_ID,
    }
    assert inspect.visibility_trace["request"]["operation"] == "inspect"
    assert all(
        text in inspect.rendered_by_sop[MIXED_SOP_ID]["prompt_text"]
        for text in (REPAIR_TEXT, AUDIT_TEXT, FORBIDDEN_SCORE_TEXT)
    )


def test_diagnostic_distillation_is_not_an_uncertified_navigation_bypass() -> None:
    engine, ref = build_mixed_authority()
    gateway = SOPVisibilityGateway(
        mixed_nodes(ref), mode="enforce", authority_engine=engine
    )
    pack = gateway.evaluate(
        visibility_request(
            ref,
            Operation.DISTILL_DIAGNOSTIC,
            generation_stage=GenerationStage.DEBUG,
        )
    )
    assert pack.effective_clause_ids == []
    assert pack.visibility_trace["empty_pack"] is True
    assert pack.visibility_trace["clause_decisions"][AUDIT_CLAUSE_ID]["reason"] == (
        "missing_matching_allow_decision"
    )


def test_frozen_snapshot_binding_and_cache_invalidation_are_exact() -> None:
    _engine, ref = build_mixed_authority()
    nodes = mixed_nodes(ref)
    score_clause = nodes[MIXED_SOP_ID]["clauses"][2]
    score_clause["authority_decision_refs"] = ["decision::snapshot"]
    snapshots = {"decision::snapshot": decision_snapshot(ref)}
    gateway = SOPVisibilityGateway(
        nodes,
        mode="enforce",
        decision_lookup=snapshots.get,
    )
    request = visibility_request(
        ref,
        Operation.RANK,
        generation_stage=GenerationStage.IMPROVE,
    )

    allowed = gateway.evaluate(request)
    assert allowed.effective_clause_ids == [SCORE_CLAUSE_ID]
    assert allowed.visibility_trace["cache_hit"] is False
    cached = gateway.evaluate(request)
    assert cached.effective_clause_ids == [SCORE_CLAUSE_ID]
    assert cached.visibility_trace["cache_hit"] is True

    # Same decision ID, different scope: the cache key must change and the
    # exact Claim type binding must fail closed.
    snapshots["decision::snapshot"] = copy.deepcopy(snapshots["decision::snapshot"])
    snapshots["decision::snapshot"]["permitted_scope"]["claim_types"] = [
        "pairwise_superiority"
    ]
    denied = gateway.evaluate(request)
    assert denied.visibility_trace["cache_hit"] is False
    assert denied.effective_clause_ids == []
    assert SCORE_CLAUSE_ID in denied.suppressed_clause_refs
    assert denied.request_id != allowed.request_id

    snapshots["decision::snapshot"] = decision_snapshot(
        ref, artifact_id="other-artifact"
    )
    wrong_artifact = gateway.evaluate(request)
    assert wrong_artifact.visibility_trace["cache_hit"] is False
    assert wrong_artifact.effective_clause_ids == []
    assert SCORE_CLAUSE_ID in wrong_artifact.suppressed_clause_refs


def test_token_budget_is_applied_after_authority_and_before_rendering() -> None:
    engine, ref = build_mixed_authority()
    gateway = SOPVisibilityGateway(
        mixed_nodes(ref), mode="enforce", authority_engine=engine
    )
    pack = gateway.evaluate(
        visibility_request(
            ref,
            Operation.DEBUG_HYPOTHESIS,
            generation_stage=GenerationStage.DEBUG,
            token_budget=7,
        )
    )
    assert pack.effective_clause_ids == [REPAIR_CLAUSE_ID]
    assert pack.visibility_trace["authority_suppressed_clause_refs"] == [
        SCORE_CLAUSE_ID
    ]
    assert pack.visibility_trace["budget_suppressed_clause_refs"] == [
        AUDIT_CLAUSE_ID
    ]
    assert pack.visibility_trace["rendered_token_count"] <= 7
    assert FORBIDDEN_SCORE_TEXT not in pack.rendered_by_sop[MIXED_SOP_ID][
        "prompt_text"
    ]

    empty = gateway.evaluate(
        visibility_request(
            ref,
            Operation.DEBUG_HYPOTHESIS,
            generation_stage=GenerationStage.DEBUG,
            token_budget=0,
        )
    )
    assert empty.effective_clause_ids == []
    assert empty.visibility_trace["empty_pack"] is True


def test_visibility_internal_error_never_exposes_score_as_debug_repair() -> None:
    engine, ref = build_mixed_authority()

    def fail_authorize(_request):
        raise RuntimeError("broken authority engine")

    engine.authorize = fail_authorize
    gateway = SOPVisibilityGateway(
        mixed_nodes(ref), mode="enforce", authority_engine=engine
    )
    debug = gateway.evaluate(
        visibility_request(
            ref,
            Operation.DEBUG_HYPOTHESIS,
            generation_stage=GenerationStage.DEBUG,
        )
    )
    assert set(debug.effective_clause_ids) == {
        REPAIR_CLAUSE_ID,
        AUDIT_CLAUSE_ID,
    }
    assert SCORE_CLAUSE_ID in debug.suppressed_clause_refs
    assert all(
        decision["reason"].startswith("visibility_internal_error:")
        for clause_id, decision in debug.visibility_trace["clause_decisions"].items()
        if clause_id in {REPAIR_CLAUSE_ID, AUDIT_CLAUSE_ID}
    )

    rank = gateway.evaluate(
        visibility_request(
            ref,
            Operation.RANK,
            generation_stage=GenerationStage.IMPROVE,
        )
    )
    assert rank.effective_clause_ids == []
    assert SCORE_CLAUSE_ID in rank.suppressed_clause_refs


def _domain_bound_mixed_nodes(ref):
    nodes = mixed_nodes(ref)
    for clause in nodes[MIXED_SOP_ID]["clauses"]:
        clause["source_task_ids"] = ["source-tabular-task"]
        clause["source_task_families"] = ["tabular_classification"]
        clause["source_domains"] = ["tabular"]
        clause["transfer_scope"] = "same_domain"
    return nodes


def test_flat_profile_ignores_authority_but_enforces_hard_domain_boundary() -> None:
    engine, ref = build_mixed_authority()
    nodes = _domain_bound_mixed_nodes(ref)
    gateway = SOPVisibilityGateway(
        nodes,
        mode="enforce",
        authority_engine=engine,
        retrieval_profile="flat_relevance_memory",
    )
    request = visibility_request(
        ref,
        Operation.RANK,
        generation_stage=GenerationStage.IMPROVE,
    )
    pack = gateway.evaluate(request)
    assert set(pack.effective_clause_ids) == {
        REPAIR_CLAUSE_ID,
        AUDIT_CLAUSE_ID,
        SCORE_CLAUSE_ID,
    }
    assert FORBIDDEN_SCORE_TEXT in pack.rendered_by_sop[MIXED_SOP_ID][
        "prompt_text"
    ]
    assert pack.visibility_trace["retrieval_profile"] == "flat_relevance_memory"
    assert pack.visibility_trace["intentional_authority_bypass_clause_ids"]

    nodes = _domain_bound_mixed_nodes(ref)
    nodes[MIXED_SOP_ID]["clauses"][2]["source_domains"] = ["nlp"]
    cross_domain = SOPVisibilityGateway(
        nodes,
        mode="enforce",
        authority_engine=engine,
        retrieval_profile="flat_relevance_memory",
    ).evaluate(request)
    assert SCORE_CLAUSE_ID in cross_domain.suppressed_clause_refs
    assert FORBIDDEN_SCORE_TEXT not in cross_domain.rendered_by_sop[
        MIXED_SOP_ID
    ]["prompt_text"]


def test_global_bit_blocks_an_entire_mixed_sop_when_one_clause_is_denied() -> None:
    engine, ref = build_mixed_authority()
    pack = SOPVisibilityGateway(
        _domain_bound_mixed_nodes(ref),
        mode="enforce",
        authority_engine=engine,
        retrieval_profile="global_validity_bit",
    ).evaluate(
        visibility_request(
            ref,
            Operation.DEBUG_HYPOTHESIS,
            generation_stage=GenerationStage.DEBUG,
        )
    )
    assert pack.effective_clause_ids == []
    assert pack.effective_sop_ids == []
    assert pack.visibility_trace["global_invalidated_sop_ids"] == [MIXED_SOP_ID]


def test_authority_only_marginalizes_stage_but_keeps_other_authority_axes() -> None:
    engine, ref = build_mixed_authority()
    request = visibility_request(
        ref,
        Operation.DEBUG_HYPOTHESIS,
        generation_stage=GenerationStage.IMPROVE,
    )
    full = SOPVisibilityGateway(
        _domain_bound_mixed_nodes(ref),
        mode="enforce",
        authority_engine=engine,
    ).evaluate(request)
    assert full.effective_clause_ids == []

    authority_only = SOPVisibilityGateway(
        _domain_bound_mixed_nodes(ref),
        mode="enforce",
        authority_engine=engine,
        retrieval_profile="authority_only",
    ).evaluate(request)
    assert set(authority_only.effective_clause_ids) == {
        REPAIR_CLAUSE_ID,
        AUDIT_CLAUSE_ID,
    }
    assert SCORE_CLAUSE_ID in authority_only.suppressed_clause_refs
    reasons = authority_only.visibility_trace["clause_decisions"]
    assert reasons[REPAIR_CLAUSE_ID]["reason"].startswith(
        "authority_only_stage_marginalized:debug/retrieval"
    )
