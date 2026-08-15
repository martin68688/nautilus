from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jsonschema


ROOT = Path(__file__).resolve().parents[1]
MLEVOLVE_ROOT = ROOT / "mlevolve"
if str(MLEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE_ROOT))

from agents.memory.atomic_claim_memory import extract_debug_signature  # noqa: E402
from agents.memory import experiment_r_router as router  # noqa: E402


def _candidate(
    sop_id: str,
    failure: str,
    repair: str,
) -> dict:
    transition_id = f"transition::{sop_id}"
    return {
        "sop_id": sop_id,
        "transition_id": transition_id,
        "supporting_transition_ids": [transition_id],
        "transition_materialized_in_runforest": True,
        "task_scope": "exact_task",
        "source_task_id": "leaf-classification",
        "task_type": "multimodal",
        "runtime_stage": "preprocessing",
        "method_family": "targeted_runtime_repair",
        "title": sop_id,
        "failure_signature": extract_debug_signature(failure),
        "when_to_use": "Use only for the same causal runtime failure.",
        "repair_action": {"summary": repair, "steps": [repair]},
        "historical_failure": failure,
        "historical_code_change": repair,
        "historical_success_result": "The child executed successfully.",
    }


def _authorized_pool() -> tuple[list[dict], str]:
    exact_id = "repair::pca-component-bound"
    exact = _candidate(
        exact_id,
        "ValueError: PCA n_components=5 must be between 0 and "
        "min(n_samples, n_features)=3 in create_eng_features.",
        "Bound n_components by min(n_samples, n_features).",
    )
    decoys = [
        _candidate(
            f"repair::batchnorm-decoy-{index:02d}",
            "ValueError: ResNet18 BatchNorm received one sample while the "
            f"leaf task has 99 rows; decoy {index}.",
            "Increase the BatchNorm batch size.",
        )
        for index in range(12)
    ]
    numeric = _candidate(
        "repair::numeric-decoy",
        "RuntimeError: a tensor with dimensions 128 and 99 cannot be broadcast.",
        "Align the two tensor shapes.",
    )
    return [*decoys, numeric, exact], exact_id


def test_field_grep_promotes_causal_symbols_over_shared_valueerror() -> None:
    candidates, exact_id = _authorized_pool()
    selected, receipt = router._grep_authorized_l3_candidates(
        candidates,
        axis="symbol",
        terms=["pca", "n_components", "n_samples", "n_features"],
        limit=8,
    )
    assert selected[0]["sop_id"] == exact_id
    assert receipt["authorized_candidate_count"] == len(candidates)
    assert receipt["ranking"][0]["all_terms_match"] is True
    assert receipt["ranking"][0]["primary_hits"] == [
        "pca",
        "n_components",
        "n_samples",
        "n_features",
    ]


def _grep_query_fn(call_names: list[str]):
    used_text = False

    def query_fn(**kwargs):
        nonlocal used_text
        call_names.append(kwargs["func_spec"].name)
        prompt = kwargs["system_message"]
        budget = json.loads(prompt["search_budget"])
        allowed = list(budget["allowed_actions"])
        if allowed == ["finish"]:
            return {"action": "finish", "reason": "candidate search complete"}
        if (
            "grep_exception" in allowed
            and "exception" in budget["required_axes_remaining"]
        ):
            return {
                "action": "grep_exception",
                "reason": "search the exact exception first",
                "terms": ["ValueError"],
                "top_k": 8,
            }
        if "grep_symbol" in allowed and "symbol" in budget["required_axes_remaining"]:
            return {
                "action": "grep_symbol",
                "reason": "search causal PCA operands",
                "terms": ["PCA", "n_components", "n_samples", "n_features"],
                "top_k": 8,
            }
        if "grep_numeric" in allowed and "numeric" in budget["required_axes_remaining"]:
            return {
                "action": "grep_numeric",
                "reason": "search observed dimensions",
                "terms": ["128", "99"],
                "top_k": 8,
            }
        if "grep_text" in allowed and not used_text:
            used_text = True
            return {
                "action": "grep_text",
                "reason": "rewrite the failure as a causal constraint",
                "terms": ["PCA", "components", "samples", "features"],
                "top_k": 8,
            }
        return {
            "action": allowed[0],
            "reason": "repeat the most useful remaining axis",
            "terms": ["PCA", "n_components"],
            "top_k": 8,
        }

    return query_fn


def _grep_layer(query_fn) -> SimpleNamespace:
    return SimpleNamespace(
        _experiment_r_agentic_query_fn=query_fn,
        cfg=None,
        experiment_r_l3_grep_max_steps=7,
        experiment_r_l3_grep_per_query_limit=8,
        experiment_r_l3_grep_min_candidates=8,
        experiment_r_l3_grep_max_candidates=20,
        experiment_r_l3_failure_context_chars=6000,
        experiment_r_l3_grep_trace_history=4,
        experiment_r_l3_grep_max_attempts=2,
        experiment_r_l3_grep_max_tokens=1600,
    )


def test_grep_agent_searches_exception_symbol_numeric_and_rewrites() -> None:
    candidates, exact_id = _authorized_pool()
    calls: list[str] = []
    layer = _grep_layer(_grep_query_fn(calls))
    selected, receipt = router._agentic_l3_grep_search(
        layer,
        task_id="leaf-classification",
        task_desc="multimodal leaf classification",
        query_text=(
            "ValueError: PCA n_components=128 must be between 0 and "
            "min(n_samples, n_features)=99."
        ),
        task_scope="exact_task",
        authorized_candidates=candidates,
    )
    assert receipt["status"] == "completed"
    assert receipt["authorized_candidate_count"] == len(candidates)
    assert receipt["searched_axes"][:3] == ["exception", "symbol", "numeric"]
    assert "text" in receipt["searched_axes"]
    assert 8 <= receipt["candidate_count"] <= 20
    assert exact_id in {row["sop_id"] for row in selected}
    assert all(name == "search_authorized_l3_repairs" for name in calls)


def test_v93_grep_keeps_28_globally_ranked_candidates(monkeypatch) -> None:
    candidates = []
    groups = (
        ("exception", 10, {"exception_names": ["AlphaError"]}),
        ("symbol", 10, {"symbol_names": ["beta_symbol"]}),
        ("numeric", 8, {"numeric_literals": ["314159"]}),
    )
    for group, count, signature in groups:
        for index in range(count):
            row = _candidate(
                f"repair::{group}-{index:02d}",
                "placeholder historical failure",
                "placeholder bounded repair",
            )
            row["failure_signature"] = signature
            candidates.append(row)

    monkeypatch.setattr(
        router,
        "_l3_grep_anchor_suggestions",
        lambda _query: {
            "exception": ["alphaerror"],
            "symbol": ["beta_symbol"],
            "numeric": ["314159"],
            "text": ["alphaerror", "beta_symbol", "314159"],
        },
    )

    def query_fn(**kwargs):
        budget = json.loads(kwargs["system_message"]["search_budget"])
        remaining = budget["required_axes_remaining"]
        if remaining:
            axis = remaining[0]
            return {
                "action": f"grep_{axis}",
                "reason": f"cover {axis}",
                "terms": {
                    "exception": ["AlphaError"],
                    "symbol": ["beta_symbol"],
                    "numeric": ["314159"],
                }[axis],
                "top_k": 12,
            }
        return {"action": "finish", "reason": "all causal axes covered"}

    layer = _grep_layer(query_fn)
    layer.experiment_r_l3_grep_per_query_limit = 12
    layer.experiment_r_l3_grep_max_candidates = 28
    selected, receipt = router._agentic_l3_grep_search(
        layer,
        task_id="leaf-classification",
        task_desc="multimodal leaf classification",
        query_text="AlphaError in beta_symbol with observed value 314159",
        task_scope="exact_task",
        authorized_candidates=candidates,
    )
    assert receipt["status"] == "completed"
    assert receipt["candidate_count"] == 28
    assert len(selected) == 28
    assert {row["sop_id"] for row in selected} == {
        row["sop_id"] for row in candidates
    }


def test_v93_grep_and_l3_prompt_budgets_are_widened() -> None:
    captured = {}

    def query_fn(**kwargs):
        captured[kwargs["func_spec"].name] = kwargs
        if kwargs["func_spec"].name == "search_authorized_l3_repairs":
            return {"action": "finish", "reason": "prompt inspection"}
        return {
            "decision": "abstain",
            "selected_sop_id": "",
            "selected_transition_id": "",
            "final_confidence": 0.0,
            "reason": "prompt inspection",
            "assessments": [],
        }

    layer = _grep_layer(query_fn)
    layer.experiment_r_l3_failure_context_chars = 12000
    layer.experiment_r_l3_grep_trace_history = 6
    layer.experiment_r_l3_agent_match_min_confidence = 0.50
    layer.experiment_r_l3_agent_match_max_tokens = 7000
    failure = "discarded-prefix" + "x" * 12000
    trace = [{"step": index} for index in range(8)]
    router._call_l3_grep_agent(
        layer,
        task_id="leaf-classification",
        task_desc="multimodal leaf classification",
        query_text=failure,
        task_scope="exact_task",
        suggestions={},
        trace=trace,
        accumulated_candidates=[],
        step_index=0,
        max_steps=7,
        required_axes_remaining=[],
        allowed_actions=["finish"],
    )
    grep_call = captured["search_authorized_l3_repairs"]
    assert grep_call["system_message"]["observed_runtime_failure"] == "x" * 12000
    assert json.loads(grep_call["system_message"]["recent_search_trace"]) == trace[-6:]

    candidates = [
        _candidate(
            f"repair::schema-{index:02d}",
            "placeholder failure",
            "placeholder repair",
        )
        for index in range(28)
    ]
    router._call_l3_match_agent(
        layer,
        task_id="leaf-classification",
        task_desc="multimodal leaf classification",
        query_text=failure,
        task_scope="exact_task",
        candidates=candidates,
    )
    l3_call = captured["choose_l3_debug_repair_by_root_cause"]
    assert l3_call["system_message"]["observed_runtime_failure"] == "x" * 12000
    assert l3_call["func_spec"].json_schema["properties"]["assessments"][
        "maxItems"
    ] == 28
    action = {
        "decision": "abstain",
        "selected_sop_id": "",
        "selected_transition_id": "",
        "final_confidence": 0.0,
        "reason": "none is causally equivalent",
        "assessments": [
            {
                "sop_id": row["sop_id"],
                "keyword_correspondence": 0.1,
                "root_cause_equivalence": 0.1,
                "runtime_stage_match": 0.5,
                "contradiction": True,
                "confidence": 0.1,
                "reason": "different root cause",
            }
            for row in candidates
        ],
    }
    jsonschema.Draft7Validator(l3_call["func_spec"].json_schema).validate(action)
    validated = router._validate_l3_match_action(
        action,
        candidates=candidates,
        min_confidence=0.50,
    )
    assert len(validated["assessments"]) == 28


def test_grep_agent_and_independent_l3_judge_form_one_debug_chain(
    monkeypatch,
) -> None:
    candidates, exact_id = _authorized_pool()
    calls: list[str] = []
    grep_query = _grep_query_fn(calls)

    def query_fn(**kwargs):
        if kwargs["func_spec"].name == "search_authorized_l3_repairs":
            return grep_query(**kwargs)
        calls.append(kwargs["func_spec"].name)
        prompt_candidates = json.loads(
            kwargs["system_message"]["authorized_l3_candidates"]
        )
        selected = next(row for row in prompt_candidates if row["sop_id"] == exact_id)
        return {
            "decision": "select",
            "selected_sop_id": exact_id,
            "selected_transition_id": selected["transition_id"],
            "final_confidence": 0.98,
            "reason": "same PCA component-bound root cause",
            "assessments": [
                {
                    "sop_id": row["sop_id"],
                    "keyword_correspondence": 0.98 if row is selected else 0.20,
                    "root_cause_equivalence": 0.99 if row is selected else 0.10,
                    "runtime_stage_match": 1.0 if row is selected else 0.50,
                    "contradiction": row is not selected,
                    "confidence": 0.98 if row is selected else 0.10,
                    "reason": "same root cause" if row is selected else "different",
                }
                for row in prompt_candidates
            ],
        }

    layer = _grep_layer(query_fn)
    layer.experiment_r_l3_agent_match_min_confidence = 0.50
    layer.experiment_r_l3_agent_match_max_attempts = 2
    layer.experiment_r_l3_agent_match_max_tokens = 6000
    layer.experiment_r_l3_agent_match_candidate_limit = 8
    layer.experiment_r_l3_semantic_shortlist_enabled = False
    layer.experiment_r_l3_grep_agent_enabled = True
    layer._visibility_is_enforced = lambda: True
    monkeypatch.setattr(
        router,
        "_hard_gated_l3_candidates",
        lambda _layer, **kwargs: (
            candidates if kwargs["task_scope"] == "exact_task" else []
        ),
    )
    result = router._agentic_l3_debug_match(
        layer,
        task_id="leaf-classification",
        task_desc="multimodal leaf classification",
        query_text=(
            "ValueError: PCA n_components=128 must be between 0 and "
            "min(n_samples, n_features)=99."
        ),
        visible_sop_ids={row["sop_id"] for row in candidates},
    )
    assert result["decision"] == "select"
    assert result["selected_sop_id"] == exact_id
    assert result["algorithm"] == ("authority_gated_grep_then_l3_root_cause_match_v1")
    assert result["grep_agent_calls"] >= 4
    assert result["agent_calls"] == 1
    assert result["trace"][0]["grep_search"]["schema"] == (
        "experiment_r_l3_grep_search_v1"
    )
    assert calls[-1] == "choose_l3_debug_repair_by_root_cause"


def test_empty_enforced_authority_pool_skips_grep_and_judge(monkeypatch) -> None:
    calls: list[str] = []

    def query_fn(**kwargs):
        calls.append(kwargs["func_spec"].name)
        raise AssertionError("an empty Authority pool must not call an Agent")

    layer = _grep_layer(query_fn)
    layer.experiment_r_l3_agent_match_min_confidence = 0.50
    layer.experiment_r_l3_agent_match_max_attempts = 2
    layer.experiment_r_l3_agent_match_candidate_limit = 8
    layer.experiment_r_l3_semantic_shortlist_enabled = False
    layer.experiment_r_l3_grep_agent_enabled = True
    layer._visibility_is_enforced = lambda: True
    monkeypatch.setattr(
        router,
        "_hard_gated_l3_candidates",
        lambda _layer, **_kwargs: [],
    )

    result = router._agentic_l3_debug_match(
        layer,
        task_id="leaf-classification",
        task_desc="multimodal leaf classification",
        query_text="RuntimeError: unseen failure",
        visible_sop_ids=set(),
    )

    assert calls == []
    assert result["decision"] == "authority_pool_empty"
    assert result["grep_agent_calls"] == 0
    assert result["agent_calls"] == 0
    assert [row["grep_search"]["status"] for row in result["trace"]] == [
        "authority_pool_empty",
        "authority_pool_empty",
    ]
