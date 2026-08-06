from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.adoption import log_adoption
from agents.memory.end2end_memory_system import (
    EndToEndMemoryController,
    MemoryCandidate,
    MemorySystemContext,
    SYSTEM_IDS,
    get_memory_system,
)
from engine.search_node import SearchNode


def _candidates(words: int = 8) -> list[MemoryCandidate]:
    candidates = []
    for source in ("sop", "runforest"):
        for index in range(8):
            candidates.append(
                MemoryCandidate(
                    candidate_id=f"{source}-{index}",
                    source=source,
                    relevance=1.0 - index * 0.05 - (
                        0.01 if source == "runforest" else 0.0
                    ),
                    prompt_text=" ".join(
                        [f"{source} procedure {index}"] * words
                    ),
                    source_stage=("draft", "improve", "debug")[index % 3],
                    source_task_id="task",
                    rank=index + 1,
                    metadata={
                        "verified_success": index % 2 == 0,
                        "success_support_count": 3 if index % 2 == 0 else 0,
                        "rejected_support_count": index % 3,
                        "failure_risk": (index % 3) / 3,
                        "execution_feedback": "passed Host execution",
                        "score_delta": 0.1 if index % 2 == 0 else None,
                        "recency": index / 7,
                    },
                )
            )
    return candidates


def _context(stage: str, budget: int = 1536) -> MemorySystemContext:
    return MemorySystemContext(
        stage=stage,
        task_id="task",
        task_description="fixture",
        prompt_token_budget=budget,
    )


def _source_counts(selection) -> dict[str, int]:
    return {
        source: sum(item.source == source for item in selection.selected_candidates)
        for source in ("sop", "runforest")
    }


def test_registry_is_exactly_the_frozen_ten_systems() -> None:
    assert len(SYSTEM_IDS) == 10
    assert len(set(SYSTEM_IDS)) == 10
    assert [get_memory_system(value).system_id for value in SYSTEM_IDS] == list(
        SYSTEM_IDS
    )
    with pytest.raises(ValueError, match="Unknown End2End"):
        get_memory_system("unfrozen-system")


@pytest.mark.parametrize(
    ("system_id", "stage", "expected"),
    [
        ("sop_only", "draft", {"sop": 6, "runforest": 0}),
        ("runforest_only", "draft", {"sop": 0, "runforest": 6}),
        ("static_hybrid", "debug", {"sop": 3, "runforest": 3}),
        ("dynamic_hybrid", "draft", {"sop": 5, "runforest": 1}),
        ("dynamic_hybrid", "improve", {"sop": 3, "runforest": 3}),
        ("dynamic_hybrid", "debug", {"sop": 1, "runforest": 5}),
        ("reversed_router", "draft", {"sop": 2, "runforest": 4}),
        ("reversed_router", "improve", {"sop": 3, "runforest": 3}),
        ("reversed_router", "debug", {"sop": 4, "runforest": 2}),
    ],
)
def test_frozen_source_quotas(system_id, stage, expected) -> None:
    selection = EndToEndMemoryController(system_id).retrieve(
        _candidates(), _context(stage)
    )
    assert _source_counts(selection) == expected
    assert len(selection.prompt_candidate_ids) == 6


def test_no_memory_observes_pool_but_exposes_nothing() -> None:
    selection = EndToEndMemoryController("no_memory").retrieve(
        _candidates(), _context("draft")
    )
    assert len(selection.raw_candidates) == 16
    assert selection.selected_candidates == ()
    assert selection.prompt_candidate_ids == ()
    assert selection.prompt_text == ""
    assert selection.prompt_token_count == 0


def test_flat_retrieval_has_no_source_or_stage_preference() -> None:
    candidates = _candidates()
    candidates.append(
        MemoryCandidate(
            candidate_id="debug-best",
            source="runforest",
            relevance=2.0,
            prompt_text="best global relevance",
            source_stage="debug",
        )
    )
    selection = EndToEndMemoryController("flat_retrieval").retrieve(
        candidates, _context("draft")
    )
    assert selection.selected_candidates[0].candidate_id == "debug-best"


def test_competitor_ports_apply_declared_restricted_semantics() -> None:
    candidates = _candidates()
    gome = EndToEndMemoryController("gome_style_port").retrieve(
        candidates, _context("draft")
    )
    assert gome.selected_candidates
    assert all(
        item.metadata["verified_success"] for item in gome.selected_candidates
    )
    assert "not the full GOME" in gome.prompt_text

    macla = EndToEndMemoryController("macla_style_port").retrieve(
        candidates, _context("improve")
    )
    assert "Expected utility" in macla.prompt_text
    assert "no online contrastive refinement" in macla.prompt_text

    rcr = EndToEndMemoryController("rcr_router_style_port").retrieve(
        candidates, _context("debug")
    )
    assert rcr.route["role"] == "debugger"
    assert "not the original multi-agent QA" in rcr.prompt_text


def test_shared_whitespace_budget_is_deterministic_and_auditable() -> None:
    controller = EndToEndMemoryController("static_hybrid")
    first = controller.retrieve(_candidates(words=30), _context("draft", 45))
    second = controller.retrieve(_candidates(words=30), _context("draft", 45))
    assert first == second
    assert first.prompt_token_count <= 45
    assert first.prompt_truncated is True
    assert len(first.prompt_candidate_ids) < 6
    assert any(
        item["reason"] == "shared_prompt_token_budget"
        for item in first.suppressed_candidates
    )
    assert [item["candidate_id"] for item in first.prompt_candidates] == list(
        first.prompt_candidate_ids
    )
    for item in first.prompt_candidates:
        assert item["prompt_text"] in first.prompt_text
    assert first.prompt_candidates[-1]["prompt_text"] == first.prompt_text.split(
        "\n\n"
    )[-1]


def test_end2end_log_binds_exact_visible_sop_and_runforest_cards() -> None:
    selection = EndToEndMemoryController("static_hybrid").retrieve(
        _candidates(), _context("draft")
    )
    pack = {
        "schema": "mlevolve_end2end_memory_pack_v1",
        "algorithm_version": "end2end_memory_systems_pilot_v1",
        "system_id": "static_hybrid",
        "stage_route": {"stage": "draft"},
        "target_task_id": "task",
        "candidate_pool_hash": "a" * 64,
        "candidate_pool_source": "shared_authority_filtered_sop_runforest",
        "raw_pool_observed": True,
        "candidate_pool": [item.to_dict() for item in selection.raw_candidates],
        "selected_candidates": [
            item.to_dict() for item in selection.selected_candidates
        ],
        "suppressed_candidates": list(selection.suppressed_candidates),
        "final_prompt_candidates": list(selection.prompt_candidates),
        "final_prompt_candidate_ids": list(selection.prompt_candidate_ids),
        "visible_clause_ids": [],
        "prompt_token_count": selection.prompt_token_count,
        "prompt_truncated": selection.prompt_truncated,
        "visibility_safety_gate": {
            "unauthorized_prompt_exposure": 0,
            "unauthorized_activation": 0,
        },
        "unauthorized_prompt_exposure": 0,
        "memory_snapshot_bound_but_not_exposed": False,
        "memory_bundle": {"manifest_sha256": "b" * 64},
        "navigation_trace": [],
    }
    observed = []
    adapter = SimpleNamespace(
        record_memory_candidate_exposure=lambda **kwargs: observed.append(kwargs)
    )
    layer = SimpleNamespace(
        current_navigation_pack=lambda: pack,
        current_visibility_pack=lambda: SimpleNamespace(request_id="request-1"),
    )
    agent = SimpleNamespace(
        external_skill_memory=layer,
        evaluation_authority=adapter,
        adoption_tracking_enabled=True,
    )
    node = SearchNode(
        id="node-visible", code="", plan="", prompt_input=selection.prompt_text,
        stage="draft"
    )

    log_adoption(
        node,
        agent,
        "run_forest_stage_hybrid_memory",
        list(selection.prompt_candidate_ids),
        "draft",
    )

    assert len(observed) == 1
    assert observed[0]["node"] is node
    assert observed[0]["request_id"] == "request-1"
    assert observed[0]["candidates"] == list(selection.prompt_candidates)
    assert {item["source"] for item in observed[0]["candidates"]} == {
        "sop",
        "runforest",
    }
    assert node.memory_routing_trace["final_prompt_candidates"] == list(
        selection.prompt_candidates
    )


def test_no_memory_routing_trace_is_serialized_before_empty_ref_return() -> None:
    pack = {
        "schema": "mlevolve_end2end_memory_pack_v1",
        "algorithm_version": "end2end_memory_systems_pilot_v1",
        "system_id": "no_memory",
        "stage_route": {"stage": "draft"},
        "target_task_id": "task",
        "candidate_pool_hash": "a" * 64,
        "candidate_pool_source": "shared_authority_filtered_sop_runforest",
        "raw_pool_observed": True,
        "selected_candidates": [],
        "suppressed_candidates": [{"candidate_id": "sop-0"}],
        "final_prompt_candidates": [],
        "final_prompt_candidate_ids": [],
        "visible_clause_ids": ["clause-1"],
        "prompt_token_count": 0,
        "prompt_truncated": False,
        "visibility_safety_gate": {
            "unauthorized_prompt_exposure": 0,
            "unauthorized_activation": 0,
        },
        "unauthorized_prompt_exposure": 0,
        "memory_snapshot_bound_but_not_exposed": True,
        "memory_bundle": {"manifest_sha256": "b" * 64},
        "navigation_trace": [{"candidate_id": "sop-0"}],
    }
    layer = SimpleNamespace(current_navigation_pack=lambda: pack)
    agent = SimpleNamespace(
        external_skill_memory=layer,
        adoption_tracking_enabled=True,
    )
    node = SearchNode(id="node", code="", plan="", stage="draft")
    log_adoption(
        node,
        agent,
        "run_forest_stage_hybrid_memory",
        [],
        "draft",
    )
    assert node.adoption_log == []
    assert node.memory_routing_trace["system_id"] == "no_memory"
    assert node.memory_routing_trace["raw_pool_observed"] is True
    assert node.memory_routing_trace[
        "memory_snapshot_bound_but_not_exposed"
    ] is True
    assert node.memory_routing_trace["unauthorized_prompt_exposure"] == 0
    assert node.memory_routing_trace["visibility_safety_gate"] == {
        "unauthorized_prompt_exposure": 0,
        "unauthorized_activation": 0,
    }


def test_exact_replay_trace_distinguishes_direct_code_from_prompt_injection() -> None:
    layer = SimpleNamespace(current_navigation_pack=lambda: {})
    agent = SimpleNamespace(
        external_skill_memory=layer,
        evaluation_authority=None,
        adoption_tracking_enabled=True,
    )
    node = SearchNode(
        id="replay-node",
        code="print('exact replay')",
        plan="execute frozen source",
        stage="draft",
        draft_role="memory_reproduction",
        replay_source={
            "task_id": "leaf-classification",
            "graph_node_id": "postsmoke::leaf-best",
            "source_kind": "recipe_implementation_capsule",
            "historical_metric": 0.08612996973006647,
            "code_sha256": "a" * 64,
        },
    )
    refs = ["postsmoke::leaf-best", "recipe::leaf-classification::003"]

    log_adoption(
        node,
        agent,
        "run_forest_stage_hybrid_memory",
        refs,
        "draft",
        adoption_mode="exact_code_replay",
    )

    route = node.memory_routing_trace
    assert route["direct_code_replay"] is True
    assert route["stage_route"]["control"] == "memory_reproduction"
    assert route["raw_candidates"][0]["candidate_id"] == (
        "postsmoke::leaf-best"
    )
    assert route["selected_candidates"] == route["raw_candidates"]
    assert route["final_prompt_candidate_ids"] == []
    assert route["final_prompt_candidates"] == []
    assert route["direct_replay_source_ref_ids"] == refs
    assert all(
        row["adoption_mode"] == "exact_code_replay"
        for row in node.adoption_log
    )


def test_stage_layer_builds_common_pool_for_no_memory(tmp_path) -> None:
    from tests.test_stage_aware_hybrid_memory import _layer

    layer = _layer(tmp_path, end2end_memory_system="no_memory")
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="task",
        task_desc="transformer validation ensemble",
        query_parts=["build a reliable model"],
    )
    pack = layer.current_navigation_pack()
    assert text == "" and refs == []
    assert pack["schema"] == "mlevolve_end2end_memory_pack_v1"
    assert pack["system_id"] == "no_memory"
    assert pack["raw_pool_observed"] is True
    assert pack["candidate_pool"]
    assert pack["prompt_token_count"] == 0
    assert pack["memory_snapshot_bound_but_not_exposed"] is True
