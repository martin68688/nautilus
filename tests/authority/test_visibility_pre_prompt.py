from __future__ import annotations

import hashlib
import json

from authority.models import GenerationStage, Operation
from tests.authority.sop_visibility_helpers import (
    AUDIT_CLAUSE_ID,
    FORBIDDEN_SCORE_TEXT,
    MIXED_SOP_ID,
    REPAIR_CLAUSE_ID,
    SCORE_CLAUSE_ID,
    build_mixed_authority,
    make_stage_layer,
    visibility_request,
)


def test_partial_container_uses_only_visible_clause_text_before_ranking(tmp_path) -> None:
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
        Operation.DEBUG_HYPOTHESIS,
        generation_stage=GenerationStage.DEBUG,
    )
    layer._prepare_visibility(
        stage="debug",
        task_id="task-1",
        task_desc="OOF alignment failure",
        request=request,
    )

    # The legacy container cache contains the forbidden score; enforce-mode
    # ranking must use the clause projection instead.
    assert "contaminated" in layer._node_tokens[MIXED_SOP_ID]
    rows = layer._rank_sops(
        FORBIDDEN_SCORE_TEXT,
        "debug",
        6,
        task_id="task-1",
        task_desc="OOF alignment failure",
    )
    assert len(rows) == 1
    row = rows[0]
    assert set(row["visible_clause_ids"]) == {
        REPAIR_CLAUSE_ID,
        AUDIT_CLAUSE_ID,
    }
    assert FORBIDDEN_SCORE_TEXT not in row["visible_text"]
    assert row["geometry_visibility_safe"] is False

    # A bare boolean cannot bless an embedding created from the full mixed SOP.
    layer.nodes[MIXED_SOP_ID]["visibility_safe_container_embedding"] = True
    rows_after_forged_flag = layer._rank_sops(
        FORBIDDEN_SCORE_TEXT,
        "debug",
        6,
        task_id="task-1",
        task_desc="OOF alignment failure",
    )
    assert rows_after_forged_flag[0]["geometry_visibility_safe"] is False
    assert rows_after_forged_flag[0]["hybrid_score_components"]["geometry"] == 0.0

    projection = layer._visibility_projection(MIXED_SOP_ID)
    layer.nodes[MIXED_SOP_ID]["visibility_safe_container_embedding_hash"] = (
        hashlib.sha256(projection["retrieval_text"].encode("utf-8")).hexdigest()
    )
    attested_rows = layer._rank_sops(
        "OOF sample_id alignment",
        "debug",
        6,
        task_id="task-1",
        task_desc="OOF alignment failure",
    )
    assert attested_rows[0]["geometry_visibility_safe"] is True


def test_suppressed_text_is_absent_from_rrf_formatter_and_token_accounting(tmp_path) -> None:
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
        task_desc="OOF alignment failure",
        query_text=FORBIDDEN_SCORE_TEXT,
        visibility_request=request,
    )
    prompt = layer._format_hybrid_pack(pack)
    trace = pack["visibility_trace"]

    assert SCORE_CLAUSE_ID in trace["suppressed_clause_refs"]
    assert SCORE_CLAUSE_ID not in trace["embedding_candidate_clause_ids"]
    assert SCORE_CLAUSE_ID not in trace["rrf_eligible_clause_ids"]
    assert trace["rendered_token_count"] == 14
    assert FORBIDDEN_SCORE_TEXT not in prompt
    assert FORBIDDEN_SCORE_TEXT not in json.dumps(
        pack["navigation_trace"], sort_keys=True
    )
    assert all(
        SCORE_CLAUSE_ID not in candidate.get("visible_clause_ids", [])
        for candidate in pack["direct_sop_candidates"]
    )
