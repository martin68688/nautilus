from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.memory import multigranular_grep as retrieval  # noqa: E402
from agents.memory import experiment_r_router as router  # noqa: E402


def _candidate(candidate_id: str, granularity: str, source: str, text: str) -> dict:
    row = {
        "id": candidate_id,
        "source": source,
        "score": 0.0,
        "flat_score": 0.0,
    }
    if source == "sop":
        row.update(
            {
                "visible_text": text,
                "clean_supporting_transition_ids": [f"transition::{candidate_id}"],
            }
        )
    return {
        "id": candidate_id,
        "source": source,
        "granularity": granularity,
        "fields": {"identity": text, "task": "leaf classification"},
        "row": row,
        "relations": {},
    }


def _universe() -> list[dict]:
    return [
        _candidate("recipe::dino", "l1_recipe", "sop", "DINO leaf full pipeline"),
        _candidate("tactic::fusion", "l2_tactic", "sop", "DINO morphology fusion"),
        _candidate("repair::shape", "l3_repair", "sop", "DINO feature shape repair"),
        _candidate("run::best", "runforest_run", "runforest", "DINO successful run"),
        _candidate(
            "transition::fusion",
            "runforest_transition",
            "runforest",
            "added DINO morphology fusion metric improved",
        ),
    ]


def _layer(query_fn) -> SimpleNamespace:
    return SimpleNamespace(
        _experiment_r_agentic_query_fn=query_fn,
        cfg=None,
        experiment_r_multigranular_context_chars=12000,
        experiment_r_multigranular_trace_history=6,
        experiment_r_multigranular_search_max_tokens=3000,
        experiment_r_multigranular_judge_max_tokens=7000,
        experiment_r_multigranular_search_rounds=2,
        experiment_r_multigranular_per_query_limit=8,
        experiment_r_multigranular_semantic_per_query=0,
        experiment_r_multigranular_max_candidates=48,
        experiment_r_multigranular_judge_candidate_limit=16,
        experiment_r_stage_selection_caps={"draft": 6, "improve": 3},
        experiment_r_allow_agent_abstention=True,
        experiment_r_candidate_limit=12,
        experiment_r_memory_pool_sha256="a" * 64,
        excluded_run_ids=[],
    )


def test_host_field_grep_records_exact_field_hits_and_discard_count() -> None:
    candidates = [
        _candidate(
            f"tactic::{index:02d}",
            "l2_tactic",
            "sop",
            f"DINO morphology fusion variant {index}",
        )
        for index in range(10)
    ]
    selected, receipt = retrieval._host_grep(
        candidates,
        granularity="l2_tactic",
        terms=["dino", "morphology"],
        fields=["identity"],
        limit=8,
    )
    assert len(selected) == 8
    assert receipt["matched_candidate_count"] == 10
    assert receipt["discarded_by_query_limit"] == 2
    assert receipt["ranking"][0]["all_terms_match"] is True
    assert receipt["ranking"][0]["field_hits"]["identity"] == [
        "dino",
        "morphology",
    ]


def test_initial_search_requires_all_granularities_with_weighted_coverage() -> None:
    action = {
        "action": "search",
        "reason": "broad first pass",
        "information_need": "overall and local implementation evidence",
        "allocation": {
            "l1_recipe": 0.30,
            "l2_tactic": 0.30,
            "l3_repair": 0.05,
            "runforest_run": 0.25,
            "runforest_transition": 0.10,
        },
        "queries": [
            {
                "granularity": granularity,
                "query": "DINO leaf",
                "terms": ["DINO", "leaf"],
                "fields": (
                    ["identity", "task", "authorized_content"]
                    if granularity.startswith("l")
                    else ["identity", "task", "method", "change_and_result"]
                ),
                "top_k": 8,
                "reason": "first-pass coverage",
            }
            for granularity in retrieval.GRANULARITIES
        ],
    }
    validated = retrieval._validate_search_action(action, first_round=True)
    assert set(row["granularity"] for row in validated["queries"]) == set(
        retrieval.GRANULARITIES
    )
    assert validated["allocation"]["l1_recipe"] > validated["allocation"][
        "l3_repair"
    ]
    broken = dict(action)
    broken["queries"] = action["queries"][:-1]
    with pytest.raises(ValueError, match="every granularity"):
        retrieval._validate_search_action(broken, first_round=True)


def test_search_schema_binds_fields_to_each_granularity() -> None:
    schema = retrieval._search_spec().json_schema
    base = {
        "action": "search",
        "reason": "field contract",
        "information_need": "field-scoped search",
        "allocation": {
            granularity: 1.0 / len(retrieval.GRANULARITIES)
            for granularity in retrieval.GRANULARITIES
        },
    }
    valid_queries = []
    for granularity in retrieval.GRANULARITIES:
        valid_queries.append(
            {
                "granularity": granularity,
                "query": "DINO leaf",
                "terms": ["DINO", "leaf"],
                "fields": sorted(retrieval.GRANULARITY_FIELDS[granularity]),
                "top_k": 8,
                "reason": "valid field set",
            }
        )
    jsonschema.Draft7Validator(schema).validate(
        {**base, "queries": valid_queries}
    )

    invalid_sop = json.loads(json.dumps(valid_queries))
    invalid_sop[1]["fields"] = ["method"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(schema).validate(
            {**base, "queries": invalid_sop}
        )

    invalid_runforest = json.loads(json.dumps(valid_queries))
    invalid_runforest[3]["fields"] = ["authorized_content"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft7Validator(schema).validate(
            {**base, "queries": invalid_runforest}
        )


def test_invalid_sop_fields_get_specific_retry_then_search_and_judge(
    monkeypatch,
) -> None:
    universe = _universe()
    search_calls = 0
    judge_calls = 0

    def search_action(*, illegal: bool) -> dict:
        return {
            "action": "search",
            "reason": "broad multi-granularity search",
            "information_need": "DINO leaf evidence",
            "allocation": {
                "l1_recipe": 0.25,
                "l2_tactic": 0.25,
                "l3_repair": 0.10,
                "runforest_run": 0.20,
                "runforest_transition": 0.20,
            },
            "queries": [
                {
                    "granularity": granularity,
                    "query": "DINO leaf",
                    "terms": ["DINO", "leaf"],
                    "fields": (
                        ["method"]
                        if illegal and granularity in {"l2_tactic", "l3_repair"}
                        else ["identity", "task", "authorized_content"]
                        if granularity.startswith("l")
                        else ["identity", "task", "method", "change_and_result"]
                    ),
                    "top_k": 8,
                    "reason": "cover every granularity",
                }
                for granularity in retrieval.GRANULARITIES
            ],
        }

    def query_fn(**kwargs):
        nonlocal search_calls, judge_calls
        if kwargs["func_spec"].name == "plan_multigranular_memory_grep":
            search_calls += 1
            prompt = kwargs["system_message"]
            if search_calls == 1:
                assert prompt["retry_feedback"] == ""
                return search_action(illegal=True)
            assert "l2_tactic" in prompt["retry_feedback"]
            assert "method" in prompt["retry_feedback"]
            assert "authorized_content" in prompt["retry_feedback"]
            field_contract = json.loads(prompt["field_contract"])
            assert field_contract["l2_tactic"] == [
                "authorized_content", "identity", "task"
            ]
            return search_action(illegal=False)

        judge_calls += 1
        candidates = json.loads(kwargs["system_message"]["authorized_candidates"])
        return {
            "decision": "select",
            "selected_ids": ["tactic::fusion", "transition::fusion"],
            "reason": "recovered search reached independent judge",
            "assessments": [
                {
                    "candidate_id": row["candidate_id"],
                    "applicability": 0.95,
                    "stage_fit": 0.95,
                    "implementation_support": 0.95,
                    "contradiction": False,
                    "confidence": 0.95,
                    "reason": "compatible",
                }
                for row in candidates
            ],
        }

    layer = _layer(query_fn)
    layer.experiment_r_multigranular_search_rounds = 1
    monkeypatch.setattr(
        retrieval, "_authorized_candidates", lambda *args, **kwargs: universe
    )
    pool = retrieval.build_multigranular_candidate_pool(
        layer,
        stage="draft",
        task_id="leaf-classification",
        task_desc="multimodal leaf classification",
        query_text="Draft DINO and morphology fusion",
        visible_sop_ids={row["id"] for row in universe if row["source"] == "sop"},
        pre_gate_raw_candidates=[],
        pre_gate_summary={},
    )
    agent = pool["retrieval_agent"]
    assert search_calls == 2
    assert judge_calls == 1
    assert agent["multigranular_search_agent_calls"] == 2
    assert agent["independent_retrieval_judge_calls"] == 1
    assert agent["effective_selected_ids"] == [
        "tactic::fusion", "transition::fusion"
    ]
    first_round = agent["multigranular_search"]["trace"][0]
    assert first_round["attempts"][0]["status"] == "invalid"
    assert first_round["attempts"][1]["status"] == "valid"
    assert first_round["accumulated_counts"] == {
        granularity: 1 for granularity in retrieval.GRANULARITIES
    }
    assert pool["candidate_pool_source"] == "live_multigranular_grep_search"


def test_independent_judge_schema_and_validator_accept_cross_granularity_set() -> None:
    candidates = _universe()
    spec = retrieval._judge_spec(max_candidates=5, max_selected=3)
    action = {
        "decision": "select",
        "selected_ids": ["recipe::dino", "tactic::fusion", "run::best"],
        "reason": "recipe and tactic have successful run support",
        "assessments": [
            {
                "candidate_id": row["id"],
                "applicability": 0.9,
                "stage_fit": 0.9,
                "implementation_support": 0.9,
                "contradiction": False,
                "confidence": 0.9,
                "reason": "compatible evidence",
            }
            for row in candidates
        ],
    }
    jsonschema.Draft7Validator(spec.json_schema).validate(action)
    validated = retrieval._validate_judge(
        action,
        candidates=candidates,
        max_selected=3,
        abstention_allowed=True,
    )
    assert validated["selected_ids"] == action["selected_ids"]


def test_complete_search_then_independent_judge_pool(monkeypatch) -> None:
    universe = _universe()
    calls = []

    def query_fn(**kwargs):
        calls.append(kwargs["func_spec"].name)
        if kwargs["func_spec"].name == "plan_multigranular_memory_grep":
            return {
                "action": "search",
                "reason": "broad multi-granularity search",
                "information_need": "DINO leaf evidence",
                "allocation": {
                    "l1_recipe": 0.25,
                    "l2_tactic": 0.25,
                    "l3_repair": 0.10,
                    "runforest_run": 0.20,
                    "runforest_transition": 0.20,
                },
                "queries": [
                    {
                        "granularity": granularity,
                        "query": "DINO leaf",
                        "terms": ["DINO", "leaf"],
                        "fields": (
                            ["identity", "task", "authorized_content"]
                            if granularity.startswith("l")
                            else [
                                "identity", "task", "method", "change_and_result"
                            ]
                        ),
                        "top_k": 8,
                        "reason": "cover every granularity",
                    }
                    for granularity in retrieval.GRANULARITIES
                ],
            }
        candidates = json.loads(kwargs["system_message"]["authorized_candidates"])
        selected = [
            row["candidate_id"]
            for row in candidates
            if row["candidate_id"] in {"tactic::fusion", "transition::fusion"}
        ]
        return {
            "decision": "select",
            "selected_ids": selected,
            "reason": "tactic plus observed improving transition",
            "assessments": [
                {
                    "candidate_id": row["candidate_id"],
                    "applicability": 0.95,
                    "stage_fit": 0.95,
                    "implementation_support": 0.95,
                    "contradiction": False,
                    "confidence": 0.95,
                    "reason": "compatible",
                }
                for row in candidates
            ],
        }

    layer = _layer(query_fn)
    monkeypatch.setattr(retrieval, "_authorized_candidates", lambda *args, **kwargs: universe)
    pool = retrieval.build_multigranular_candidate_pool(
        layer,
        stage="improve",
        task_id="leaf-classification",
        task_desc="multimodal leaf classification",
        query_text="Improve DINO and morphology fusion",
        visible_sop_ids={row["id"] for row in universe if row["source"] == "sop"},
        pre_gate_raw_candidates=[],
        pre_gate_summary={},
    )
    agent = pool["retrieval_agent"]
    assert calls.count("plan_multigranular_memory_grep") == 2
    assert calls[-1] == "choose_multigranular_retrieval_evidence"
    assert agent["main_retrieval_agent_calls"] == 0
    assert agent["independent_retrieval_judge_calls"] == 1
    assert agent["effective_selected_ids"] == [
        "tactic::fusion",
        "transition::fusion",
    ]
    assert agent["multigranular_search"]["accumulated_counts"] == {
        granularity: 1 for granularity in retrieval.GRANULARITIES
    }
    assert agent["retrieval_judge"]["candidate_count"] == 5
    assert pool["candidate_pool_source"] == "live_multigranular_grep_search"


def test_router_uses_multigranular_path_only_for_enabled_draft_improve(monkeypatch):
    captured = []

    def builder(_layer, **kwargs):
        captured.append(kwargs["stage"])
        return {"schema": "sentinel"}

    monkeypatch.setattr(
        retrieval,
        "build_multigranular_candidate_pool",
        builder,
    )
    monkeypatch.setattr(router, "_agentic_pre_gate_audit", lambda *args, **kwargs: ([], {}))
    layer = SimpleNamespace(
        experiment_r_agentic_per_step_top_k=8,
        experiment_r_agentic_max_observed=48,
        experiment_r_multigranular_grep_enabled=True,
        experiment_r_multigranular_grep_stages={"draft", "improve"},
    )
    result = router._agentic_candidate_pool(
        layer,
        stage="draft",
        task_id="leaf-classification",
        task_desc="leaf",
        query_text="draft",
        visible_sop_ids=set(),
    )
    assert result == {"schema": "sentinel"}
    assert captured == ["draft"]
