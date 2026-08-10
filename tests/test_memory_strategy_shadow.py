from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.atomic_actuation import (
    apply_atomic_diff_response,
    run_atomic_actuation_pipeline,
)
import agents.memory_strategy_agent as memory_strategy_module
from agents.memory_strategy_agent import (
    build_component_portfolio,
    build_memory_cards,
    build_strategy_context,
    build_strategy_evidence_view,
    payload_sha256,
    run_memory_strategy_shadow,
    should_run_memory_strategy_shadow,
    validate_strategy_memo,
)
from engine.search_node import Journal, SearchNode
from utils.metric import MetricValue


def _config(*, shadow_enabled=True, stages=None):
    ext = SimpleNamespace(
        memory_strategy_shadow_enabled=shadow_enabled,
        memory_strategy_shadow_stages=stages or ["improve"],
        memory_strategy_debug_trigger="causal_gap_or_repeated_failure",
        memory_strategy_debug_failure_threshold=2,
        memory_strategy_evidence_limit=8,
        memory_strategy_current_frontier_slots=3,
        memory_strategy_causal_failure_slots=1,
        memory_strategy_candidate_pool_limit=48,
        memory_strategy_max_cards=24,
        memory_strategy_card_max_chars=6000,
        memory_strategy_max_input_chars=0,
        memory_strategy_max_output_tokens=12000,
        memory_strategy_max_retries=2,
        memory_strategy_contract_retries=2,
        memory_strategy_min_candidate_compositions=1,
        memory_strategy_temperature=0.0,
        memory_strategy_model="deepseek-v4-pro",
        memory_strategy_thinking_enabled=True,
        memory_strategy_history_limit=16,
        memory_strategy_atomic_max_modules=2,
        memory_strategy_atomic_max_changes=3,
        memory_strategy_atomic_max_patches=6,
        memory_strategy_atomic_planner_contract_retries=2,
        memory_strategy_atomic_coder_contract_retries=1,
    )
    code = SimpleNamespace(model="deepseek-v4-flash", temp=0.0)
    search = SimpleNamespace(num_gpus=1)
    acfg = SimpleNamespace(
        code=code,
        search=search,
        time_limit=21600,
        draft_role_policy=SimpleNamespace(enabled=False, roles=[]),
    )
    return SimpleNamespace(
        external_skill_memory=ext,
        agent=acfg,
        exec=SimpleNamespace(timeout=1800),
        exp_id="spooky-author-identification",
    )


def _agent(*, shadow_enabled=True, stages=None):
    cfg = _config(shadow_enabled=shadow_enabled, stages=stages)
    return SimpleNamespace(
        cfg=cfg,
        acfg=cfg.agent,
        task_desc="three-class authorship attribution",
        data_preview="text train rows with author label",
        start_time=time.time() - 300,
        journal=Journal(),
        branch_successful_nodes={},
        branch_all_nodes={},
        metric_maximize=False,
    )


def _parent() -> SearchNode:
    node = SearchNode(
        code=(
            "import numpy as np\n\n"
            "def build_features(texts):\n"
            "    return texts\n\n"
            "def train_models(x, y):\n"
            "    return x\n\n"
            "def write_submission(pred):\n"
            "    return pred\n"
        ),
        plan="TF-IDF word/character features with a three-model probability blend",
        stage="draft",
    )
    node.metric = MetricValue(0.336, maximize=False)
    node.is_buggy = False
    node.is_valid = True
    node._term_out = [
        "Final Submission-Aligned Validation Score: 0.336 | variant=three_model\n"
    ]
    node.official_submission_receipt = {
        "submission_variant": "three_model",
        "submission_sha256": "a" * 64,
    }
    return node


def _router_pack():
    return {
        "schema": "experiment_r_memory_pack_v1",
        "stage_route": {"stage": "improve"},
        "final_prompt_candidate_ids": ["run::three-model"],
        "final_prompt_candidates": [
            {
                "candidate_id": "run::three-model",
                "source": "runforest",
                "source_stage": "improve",
                "source_task_id": "spooky-author-identification",
                "metric": 0.340,
                "prompt_text": "three complementary linear classifiers improve the blend",
            }
        ],
        "selected_candidates": [
            {
                "candidate_id": "run::three-model",
                "source": "runforest",
                "prompt_text": "duplicate selected view",
            }
        ],
        "pre_gate_raw_candidates": [
            {
                "candidate_id": "run::five-fold",
                "source": "runforest",
                "source_stage": "improve",
                "source_task_id": "spooky-author-identification",
                "metric": 0.331,
                "prompt_text": "five-fold OOF averaging improves stability",
            },
            {
                "candidate_id": "run::future-hidden",
                "source": "runforest",
                "prompt_text": "this row is removed by a time-slice harness",
            },
        ],
        "candidate_pool": {
            "runforest_candidates": [
                {
                    "candidate_id": "run::char-tfidf",
                    "source": "runforest",
                    "prompt_text": "character TF-IDF helps author style",
                }
            ]
        },
        "retrieval_agent": {"agent_abstained": False},
        "router_activation": {"status": "selected"},
    }


def _strategy_memo():
    return {
        "decision": "propose",
        "abstention_reason": "",
        "current_system_map": {
            "features": "word+character TF-IDF",
            "models": "three-model blend",
            "validation": "single split",
        },
        "evidence_portfolio": {
            "supported_axes": ["model diversity", "five-fold validation"]
        },
        "coverage_gaps": ["three-model blend and five-fold averaging were not tested together"],
        "candidate_compositions": [
            {
                "hypothesis_id": "h-three-model-five-fold",
                "hypothesis": "cross-fit each of the existing three models over five folds",
                "source_memory_ids": ["run::three-model", "run::five-fold"],
                "compatibility_checks": [
                    "fit TF-IDF inside each fold",
                    "keep class order fixed across models and folds",
                ],
                "known_conflicts": ["15 fits may exceed the remaining budget"],
                "estimated_compute_seconds": 1400,
                "minimal_change_set": [
                    "wrap existing train_models in five-fold OOF/test averaging"
                ],
                "forbidden_changes": ["do not add a fourth model"],
                "expected_mechanism": "reduce split variance without changing model diversity",
                "falsification_condition": "OOF log-loss does not beat the current blend",
                "novelty_kind": "new_composition",
            }
        ],
        "addressed_opportunities": [],
        "recommended_hypothesis_id": "h-three-model-five-fold",
        "recommendation_reason": "both components have independent task-local support",
        "declined_hypotheses": [],
    }


def _successful_node(
    node_id: str,
    *,
    metric: float,
    role: str,
    branch_id: int,
    text: str,
) -> SearchNode:
    node = SearchNode(
        code="def train():\n    return 1\n",
        plan=text,
        stage="improve",
        id=node_id,
        branch_id=branch_id,
        draft_role=role,
        code_summary=text,
    )
    node.metric = MetricValue(metric, maximize=False)
    node.is_buggy = False
    node.is_valid = True
    node.validation_protocol = "full_oof_submission_aligned_internal"
    node.official_submission_receipt = {
        "submission_variant": node_id,
        "submission_sha256": node_id[0] * 64,
        "validation_protocol": "full_oof_submission_aligned_internal",
    }
    return node


def _failed_node(node_id: str) -> SearchNode:
    node = SearchNode(
        code="def train():\n    raise IndexError('shape')\n",
        plan="add one feature block",
        stage="improve",
        id=node_id,
    )
    node.metric = None
    node.is_buggy = True
    node.is_valid = False
    node.exc_type = "IndexError"
    node.analysis = "feature matrix and estimator input dimensions disagree"
    node._term_out = ["IndexError: shape mismatch"]
    return node


def _portfolio_router_pack() -> dict:
    rows = [
        {
            "candidate_id": "history::tfidf-best",
            "source": "runforest",
            "source_task_id": "spooky-author-identification",
            "metric": 0.280,
            "rank_eligible": True,
            "validation_protocol": "full_oof_submission_aligned_internal",
            "submission_aligned": True,
            "text": "word and character TF-IDF with logistic regression",
        },
        {
            "candidate_id": "history::tfidf-duplicate",
            "source": "runforest",
            "source_task_id": "spooky-author-identification",
            "metric": 0.290,
            "rank_eligible": True,
            "validation_protocol": "full_oof_submission_aligned_internal",
            "submission_aligned": True,
            "text": "word and character TF-IDF with logistic regression",
        },
        {
            "candidate_id": "history::xgb",
            "source": "runforest",
            "source_task_id": "spooky-author-identification",
            "metric": 0.310,
            "rank_eligible": True,
            "validation_protocol": "full_oof_submission_aligned_internal",
            "submission_aligned": True,
            "text": "frozen DeBERTa embedding with XGBoost",
        },
        {
            "candidate_id": "history::five-fold-blend",
            "source": "runforest",
            "source_task_id": "spooky-author-identification",
            "metric": 0.320,
            "rank_eligible": True,
            "validation_protocol": "full_oof_submission_aligned_internal",
            "submission_aligned": True,
            "text": "five-fold three-model probability blend",
        },
        {
            "candidate_id": "history::holdout-stylometry",
            "source": "runforest",
            "source_task_id": "spooky-author-identification",
            "metric": 0.100,
            "rank_eligible": True,
            "validation_protocol": "single_holdout_submission_aligned_internal",
            "submission_aligned": True,
            "text": "stylometric punctuation density with a linear classifier",
        },
        {
            "candidate_id": "run::old-spooky::node::replay-best",
            "original_node_id": "replay-best",
            "source": "runforest",
            "source_task_id": "spooky-author-identification",
            "metric": 0.200,
            "rank_eligible": True,
            "validation_protocol": "full_oof_submission_aligned_internal",
            "submission_aligned": True,
            "text": "the same current frozen DeBERTa embedding with XGBoost node",
        },
        {
            "candidate_id": "history::wrong-task",
            "source": "runforest",
            "source_task_id": "leaf-classification",
            "metric": 0.001,
            "rank_eligible": True,
            "text": "EfficientNet image model",
        },
        {
            "candidate_id": "history::rank-blocked",
            "source": "runforest",
            "source_task_id": "spooky-author-identification",
            "metric": 0.002,
            "rank_eligible": False,
            "text": "leaky target encoding",
        },
        {
            "candidate_id": "history::no-metric",
            "source": "runforest",
            "source_task_id": "spooky-author-identification",
            "rank_eligible": True,
            "text": "unscored experiment",
        },
    ]
    return {
        "schema": "experiment_r_memory_pack_v1",
        "stage_route": {"stage": "improve"},
        "final_prompt_candidate_ids": [row["candidate_id"] for row in rows[:3]],
        "final_prompt_candidates": rows[:3],
        "pre_gate_raw_candidates": rows[3:],
        "retrieval_agent": {"agent_abstained": False},
    }


def test_wide_memory_cards_include_unselected_rows_and_deduplicate():
    cards = build_memory_cards(_router_pack(), max_cards=8)
    ids = [card["memory_id"] for card in cards]

    assert ids[0] == "run::three-model"
    assert ids.count("run::three-model") == 1
    assert "run::five-fold" in ids
    assert "run::char-tfidf" in ids
    assert next(card for card in cards if card["memory_id"] == "run::five-fold")[
        "router_visibility"
    ] == "pre_gate"


def test_strategy_evidence_uses_role_frontiers_failure_and_diverse_history():
    agent = _agent()
    roles = [
        "coldstart_baseline",
        "memory_reproduction",
        "novel_exploration",
    ]
    agent.acfg.draft_role_policy = SimpleNamespace(enabled=True, roles=roles)
    cold_best = _successful_node(
        "cold-best",
        metric=0.330,
        role=roles[0],
        branch_id=1,
        text="character TF-IDF linear baseline",
    )
    cold_worse = _successful_node(
        "cold-worse",
        metric=0.410,
        role=roles[0],
        branch_id=1,
        text="word TF-IDF baseline",
    )
    replay_best = _successful_node(
        "replay-best",
        metric=0.300,
        role=roles[1],
        branch_id=2,
        text="frozen DeBERTa embedding with XGBoost",
    )
    novel_best = _successful_node(
        "novel-best",
        metric=0.340,
        role=roles[2],
        branch_id=3,
        text="stylometric feature block",
    )
    cold_best.branch_id = None
    agent.branch_successful_nodes = {
        1: [cold_worse, cold_best],
        2: [replay_best],
        3: [novel_best],
    }
    failure = _failed_node("recent-failure")
    agent.journal.append(cold_worse)
    agent.journal.append(failure)

    cards, selection = build_strategy_evidence_view(
        agent,
        replay_best,
        stage="improve",
        router_pack=_portfolio_router_pack(),
        max_items=8,
        current_frontier_slots=3,
        causal_failure_slots=1,
        candidate_pool_limit=48,
    )

    assert len(cards) == 8
    assert selection["selected_count"] == 8
    assert selection["current_branch_frontier_ids"] == [
        "current::cold-best",
        "current::replay-best",
        "current::novel-best",
    ]
    assert selection["causal_failure_ids"] == ["current::recent-failure"]
    assert "current::cold-worse" not in selection["selected_memory_ids"]
    assert "history::tfidf-best" in selection["historical_diverse_frontier_ids"]
    assert (
        "history::tfidf-duplicate"
        not in selection["historical_diverse_frontier_ids"]
    )
    assert (
        "run::old-spooky::node::replay-best"
        not in selection["historical_diverse_frontier_ids"]
    )
    assert selection["historical_rejections"] == {
        "task_mismatch": 1,
        "rank_ineligible": 1,
        "missing_metric": 1,
        "current_node_duplicate": 1,
        "duplicate_lineage": 1,
    }
    signatures = selection["historical_lineage_signatures"]
    assert len(signatures) == len(set(signatures)) == 4
    # Selecting a frontier must not mutate the live SearchNode.
    assert cold_best.branch_id is None
    cold_card = next(card for card in cards if card["memory_id"] == "current::cold-best")
    assert cold_card["branch_id"] == 1


def test_historical_scores_are_ranked_only_inside_validation_protocol():
    agent = _agent()
    parent = _successful_node(
        "parent-protocol",
        metric=0.330,
        role="memory_reproduction",
        branch_id=2,
        text="three-model blend",
    )
    agent.branch_successful_nodes = {2: [parent]}
    pack = _portfolio_router_pack()
    pack["pre_gate_raw_candidates"] = [
        row
        for row in pack["pre_gate_raw_candidates"]
        if row["candidate_id"] != "run::old-spooky::node::replay-best"
    ]

    cards, selection = build_strategy_evidence_view(
        agent,
        parent,
        stage="improve",
        router_pack=pack,
        max_items=8,
        current_frontier_slots=1,
        causal_failure_slots=0,
        candidate_pool_limit=48,
    )
    historical = [
        card
        for card in cards
        if card["router_visibility"] == "historical_diverse_frontier"
    ]
    ids = [card["memory_id"] for card in historical]

    assert ids.index("history::tfidf-best") < ids.index("history::xgb")
    # The numerically smaller holdout score is not promoted above the matching
    # OOF protocol merely by comparing incompatible raw values.
    assert ids.index("history::holdout-stylometry") > ids.index(
        "history::five-fold-blend"
    )
    holdout = next(
        card for card in historical if card["memory_id"] == "history::holdout-stylometry"
    )
    assert holdout["metric_comparable_to_current"] is False
    assert holdout["metric_claim_status"] == "same_task_within_protocol_only"
    assert "same task and validation protocol" in selection[
        "historical_metric_comparison_policy"
    ]


def test_unverified_historical_score_is_labelled_as_method_evidence():
    agent = _agent()
    parent = _parent()
    agent.branch_successful_nodes = {1: [parent]}
    pack = {
        "final_prompt_candidates": [
            {
                "candidate_id": "history::unverified",
                "source_task_id": "spooky-author-identification",
                "metric": 0.001,
                "rank_eligible": True,
                "text": "character TF-IDF with a linear classifier",
            }
        ]
    }

    cards, _ = build_strategy_evidence_view(
        agent,
        parent,
        stage="improve",
        router_pack=pack,
        max_items=8,
        current_frontier_slots=1,
        causal_failure_slots=0,
    )
    card = next(item for item in cards if item["memory_id"] == "history::unverified")
    assert card["submission_aligned_receipt_present"] is False
    assert card["metric_claim_status"] == "unverified_method_evidence_only"
    assert card["metric_comparable_to_current"] is False


def test_component_portfolio_prefers_executed_summary_over_discussed_plan():
    portfolio = build_component_portfolio(
        [
            {
                "memory_id": "run::deberta-actual",
                "plan": "Replace the failed ModernBERT fine-tuning attempt.",
                "text": "Frozen DeBERTa embeddings feed an XGBoost classifier.",
            }
        ]
    )

    axes = portfolio["component_axes"]
    assert "deberta" in axes["representation"]
    assert "modernbert" not in axes["representation"]
    assert "frozen_embedding" in axes["adaptation_mode"]
    assert "fine_tuning" not in axes["adaptation_mode"]
    assert portfolio["card_components"][0]["evidence_basis"] == "text"


def test_strategy_contract_requires_auditable_opportunity_disposition():
    memo = _strategy_memo()
    missing = validate_strategy_memo(
        memo,
        available_memory_ids=["run::three-model", "run::five-fold"],
        required_opportunity_ids=["within_axis::model::a+b+c"],
    )
    assert missing["valid"] is False
    assert "unaddressed within-axis" in " ".join(missing["violations"])

    memo["addressed_opportunities"] = [
        {
            "opportunity_id": "within_axis::model::a+b+c",
            "disposition": "proposed",
            "hypothesis_id": "h-three-model-five-fold",
            "reason": "all models expose aligned probability vectors",
        }
    ]
    accepted = validate_strategy_memo(
        memo,
        available_memory_ids=["run::three-model", "run::five-fold"],
        required_opportunity_ids=["within_axis::model::a+b+c"],
    )
    assert accepted["valid"] is True


def test_shadow_agent_records_global_composition_without_mutating_router_or_prompt():
    agent = _agent()
    parent = _parent()
    agent.journal.append(parent)
    pack = _router_pack()
    frozen_pack = copy.deepcopy(pack)
    production_prompt = {"Memory": "legacy", "External Skill Memory": "visible only"}
    before = payload_sha256(production_prompt)
    observed = {}

    def query_fn(**kwargs):
        observed["context"] = kwargs["context"]
        return _strategy_memo()

    agent._memory_strategy_query_fn = query_fn
    trace = run_memory_strategy_shadow(
        agent,
        parent,
        stage="improve",
        router_pack=pack,
        branch_best_metric=0.331,
        production_prompt_sha256=before,
    )

    assert trace["status"] == "completed"
    assert trace["schema"] == "mlevolve_memory_strategy_shadow_trace_v2"
    assert trace["actuation_authority"] == "none"
    assert trace["production_prompt_modified"] is False
    assert trace["memo"]["recommended_hypothesis_id"] == "h-three-model-five-fold"
    assert set(trace["memory_card_ids"]) >= {"run::three-model", "run::five-fold"}
    assert trace["memory_card_count"] <= 8
    assert trace["strategy_evidence_selection"]["selected_count"] == trace[
        "memory_card_count"
    ]
    assert observed["context"]["metrics"]["parent_submission_metric"]["value"] == 0.336
    assert observed["context"]["metrics"]["branch_best_metric"] == 0.331
    portfolio = observed["context"]["component_portfolio"]["component_axes"]
    assert "five_fold" in portfolio["validation"]
    assert pack == frozen_pack
    assert payload_sha256(production_prompt) == before


def test_strategy_can_cite_a_current_branch_frontier_node():
    agent = _agent()
    parent = _parent()
    parent.id = "parent-frontier"
    parent.branch_id = 2
    parent.draft_role = "memory_reproduction"
    agent.branch_successful_nodes = {2: [parent]}
    agent.journal.append(parent)
    memo = _strategy_memo()
    memo["candidate_compositions"][0]["source_memory_ids"] = [
        "current::parent-frontier",
        "run::five-fold",
    ]
    agent._memory_strategy_query_fn = lambda **_kwargs: memo

    trace = run_memory_strategy_shadow(
        agent,
        parent,
        stage="improve",
        router_pack=_router_pack(),
    )

    assert trace["status"] == "completed"
    assert "current::parent-frontier" in trace["memory_card_ids"]
    assert trace["validation"]["valid"] is True


def test_strategy_context_keeps_parent_metric_distinct_from_branch_best():
    agent = _agent()
    parent = _parent()
    context = build_strategy_context(
        agent,
        parent,
        stage="improve",
        router_pack=_router_pack(),
        branch_best_metric=0.300,
    )

    assert context["metrics"] == {
        "parent_submission_metric": {
            "value": 0.336,
            "maximize": False,
            "submission_variant": "three_model",
            "submission_sha256": "a" * 64,
            "submission_aligned_receipt_present": True,
        },
        "branch_best_metric": 0.300,
        "branch_best_is_parent": False,
    }


def test_strategy_uses_independent_v4_pro_thinking_without_mutating_coder_cfg(
    monkeypatch,
):
    agent = _agent()
    parent = _parent()
    observed = {}

    def fake_generate(**kwargs):
        observed["model"] = kwargs["cfg"].agent.code.model
        observed["json_schema"] = kwargs["json_schema"]
        observed["max_tokens"] = kwargs["max_tokens"]
        return _strategy_memo()

    monkeypatch.setattr(memory_strategy_module, "generate", fake_generate)
    trace = run_memory_strategy_shadow(
        agent,
        parent,
        stage="improve",
        router_pack=_router_pack(),
    )

    assert trace["status"] == "completed"
    assert trace["model"] == "deepseek-v4-pro"
    assert trace["thinking_enabled"] is True
    assert observed == {
        "model": "deepseek-v4-pro",
        "json_schema": None,
        "max_tokens": 12000,
    }
    assert agent.cfg.agent.code.model == "deepseek-v4-flash"


def test_debug_shadow_runs_only_for_causal_gap_or_repeated_failure():
    agent = _agent(stages=["improve", "debug"])
    parent = _parent()

    enabled, reason = should_run_memory_strategy_shadow(
        agent,
        stage="debug",
        parent_node=parent,
        router_pack={
            "final_prompt_candidate_ids": [],
            "retrieval_agent": {"agent_abstained": True},
        },
    )
    assert enabled is True and reason == "debug_causal_gap"

    enabled, reason = should_run_memory_strategy_shadow(
        agent,
        stage="debug",
        parent_node=parent,
        router_pack={
            "final_prompt_candidate_ids": ["repair::1"],
            "retrieval_agent": {"agent_abstained": False},
        },
    )
    assert enabled is False and reason == "debug_trigger_not_met"


def test_atomic_planner_and_coder_execute_one_strategy_with_verified_symbol_scope():
    agent = _agent()
    parent_code = (
        "import math\n\n"
        "def train_models(x):\n"
        "    return x\n\n"
        "def untouched(x):\n"
        "    return x * 2\n"
    )
    memo = _strategy_memo()
    agent._atomic_planner_query_fn = lambda **_kwargs: {
        "hypothesis_id": "h-three-model-five-fold",
        "objective": "cross-fit the existing model collection",
        "source_memory_ids": ["run::three-model", "run::five-fold"],
        "allowed_modules": ["training_evaluation"],
        "allowed_changes": [
            {
                "change_id": "cross_fit_existing_models",
                "operation": "modify",
                "target_symbols": ["train_models"],
                "description": "loop over five folds and preserve the existing model set",
            }
        ],
        "allowed_new_imports": [],
        "forbidden_symbols": ["untouched"],
        "forbidden_code_patterns": ["FourthModel"],
        "preserve_invariants": ["keep output class order"],
        "compatibility_checks": ["all folds return identical class columns"],
        "estimated_compute_seconds": 1200,
        "max_patches": 2,
        "expected_mechanism": "reduce split variance",
        "falsification_condition": "OOF does not improve",
    }
    agent._atomic_coder_query_fn = lambda **_kwargs: (
        "<<<<<<< SEARCH\n"
        "def train_models(x):\n"
        "    return x\n"
        "=======\n"
        "def train_models(x):\n"
        "    fold_predictions = [x for _ in range(5)]\n"
        "    return sum(fold_predictions) / len(fold_predictions)\n"
        ">>>>>>> REPLACE\n"
    )

    result = run_atomic_actuation_pipeline(
        agent,
        strategy_memo=memo,
        parent_code=parent_code,
        task_description="authorship classification",
        budget={"remaining_search_seconds": 5000},
    )

    assert result["status"] == "accepted"
    assert result["planner"]["plan"]["hypothesis_id"] == "h-three-model-five-fold"
    verdict = result["coder"]["plan_diff_verdict"]
    assert verdict["valid"] is True
    assert verdict["changed_symbols"] == ["train_models"]
    assert "untouched" not in verdict["changed_symbols"]


def test_atomic_diff_rejects_coder_scope_expansion():
    parent_code = (
        "def train_models(x):\n"
        "    return x\n\n"
        "def untouched(x):\n"
        "    return x * 2\n"
    )
    plan = {
        "allowed_changes": [
            {"target_symbols": ["train_models"]},
        ],
        "allowed_new_imports": [],
        "forbidden_symbols": ["untouched"],
        "forbidden_code_patterns": [],
        "max_patches": 2,
    }
    response = (
        "<<<<<<< SEARCH\n"
        "def untouched(x):\n"
        "    return x * 2\n"
        "=======\n"
        "def untouched(x):\n"
        "    return x * 3\n"
        ">>>>>>> REPLACE\n"
    )

    code, verdict = apply_atomic_diff_response(
        response=response,
        original_code=parent_code,
        atomic_plan=plan,
    )

    assert code is None
    assert verdict["valid"] is False
    assert verdict["forbidden_symbols_touched"] == ["untouched"]
    assert any("outside allowed set" in value for value in verdict["violations"])


def test_atomic_diff_allows_explicit_nested_import_when_root_is_already_present():
    parent_code = (
        "import torch\n\n"
        "def train_models(x):\n"
        "    return x\n"
    )
    plan = {
        "allowed_changes": [{"target_symbols": ["train_models"]}],
        "allowed_new_imports": [
            "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
        ],
        "forbidden_symbols": [],
        "forbidden_code_patterns": [],
        "max_patches": 2,
    }
    response = (
        "<<<<<<< SEARCH\n"
        "import torch\n"
        "=======\n"
        "import torch\n"
        "from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "def train_models(x):\n"
        "    return x\n"
        "=======\n"
        "def train_models(x):\n"
        "    scheduler = CosineAnnealingWarmRestarts(x, T_0=2)\n"
        "    return scheduler\n"
        ">>>>>>> REPLACE\n"
    )

    code, verdict = apply_atomic_diff_response(
        response=response,
        original_code=parent_code,
        atomic_plan=plan,
    )

    assert code is not None
    assert verdict["valid"] is True
    assert verdict["new_imports"] == [
        "torch.optim.lr_scheduler.CosineAnnealingWarmRestarts"
    ]
    assert verdict["changed_symbols"] == ["__imports__", "train_models"]


def test_shadow_disabled_does_not_call_model():
    agent = _agent(shadow_enabled=False)
    parent = _parent()
    agent._memory_strategy_query_fn = lambda **_kwargs: (_ for _ in ()).throw(
        AssertionError("must not call model")
    )
    trace = run_memory_strategy_shadow(
        agent,
        parent,
        stage="improve",
        router_pack=_router_pack(),
    )
    assert trace["status"] == "not_run"
    assert trace["trigger_reason"] == "disabled"


def test_strategy_contract_retries_legacy_shape_instead_of_marking_it_valid():
    agent = _agent()
    parent = _parent()
    calls = []

    def query_fn(**kwargs):
        calls.append(kwargs["contract_attempt"])
        if kwargs["contract_attempt"] == 0:
            return {
                "analysis": "use the five-fold protocol",
                "proposed_experiment": {"name": "legacy shape"},
            }
        return _strategy_memo()

    agent._memory_strategy_query_fn = query_fn
    trace = run_memory_strategy_shadow(
        agent,
        parent,
        stage="improve",
        router_pack=_router_pack(),
    )

    assert trace["status"] == "completed"
    assert calls == [0, 1]
    assert trace["contract_attempts"][0]["valid"] is False
    assert "missing required top-level keys" in " ".join(
        trace["contract_attempts"][0]["violations"]
    )
    assert trace["contract_attempts"][1]["valid"] is True


def test_atomic_planner_retries_legacy_shape_without_shrinking_hypothesis():
    agent = _agent()
    parent_code = "def train_models(x):\n    return x\n"
    calls = []

    def planner_query(**kwargs):
        calls.append(kwargs["contract_attempt"])
        if kwargs["contract_attempt"] == 0:
            return {"experiment": {"hypothesis": "legacy planner shape"}}
        return {
            "hypothesis_id": "h-three-model-five-fold",
            "objective": "cross-fit the existing model collection",
            "source_memory_ids": ["run::three-model", "run::five-fold"],
            "allowed_modules": ["training_evaluation"],
            "allowed_changes": [
                {
                    "change_id": "cross_fit_existing_models",
                    "operation": "modify",
                    "target_symbols": ["train_models"],
                    "description": "loop over five folds",
                }
            ],
            "allowed_new_imports": [],
            "forbidden_symbols": [],
            "forbidden_code_patterns": [],
            "preserve_invariants": ["keep output class order"],
            "compatibility_checks": ["fold class columns match"],
            "estimated_compute_seconds": 1200,
            "max_patches": 1,
            "expected_mechanism": "reduce split variance",
            "falsification_condition": "OOF does not improve",
        }

    agent._atomic_planner_query_fn = planner_query
    agent._atomic_coder_query_fn = lambda **_kwargs: (
        "<<<<<<< SEARCH\n"
        "def train_models(x):\n"
        "    return x\n"
        "=======\n"
        "def train_models(x):\n"
        "    return sum([x for _ in range(5)]) / 5\n"
        ">>>>>>> REPLACE\n"
    )
    result = run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_memo(),
        parent_code=parent_code,
        task_description="authorship classification",
    )

    assert result["status"] == "accepted"
    assert calls == [0, 1]
    assert result["planner"]["contract_attempts"][0]["valid"] is False
    assert result["planner"]["plan"]["hypothesis_id"] == "h-three-model-five-fold"


def test_atomic_planner_retries_non_top_level_target_before_coder():
    agent = _agent()
    parent_code = "def train_models(x):\n    criterion = x\n    return criterion\n"

    def plan(target):
        return {
            "hypothesis_id": "h-three-model-five-fold",
            "objective": "cross-fit the existing model collection",
            "source_memory_ids": ["run::three-model", "run::five-fold"],
            "allowed_modules": ["training_evaluation"],
            "allowed_changes": [
                {
                    "change_id": "cross_fit_existing_models",
                    "operation": "modify",
                    "target_symbols": [target],
                    "description": "loop over five folds",
                }
            ],
            "allowed_new_imports": [],
            "forbidden_symbols": [],
            "forbidden_code_patterns": [],
            "preserve_invariants": ["keep output class order"],
            "compatibility_checks": ["fold class columns match"],
            "estimated_compute_seconds": 1200,
            "max_patches": 1,
            "expected_mechanism": "reduce split variance",
            "falsification_condition": "OOF does not improve",
        }

    agent._atomic_planner_query_fn = lambda **kwargs: plan(
        "criterion" if kwargs["contract_attempt"] == 0 else "train_models"
    )
    agent._atomic_coder_query_fn = lambda **_kwargs: (
        "<<<<<<< SEARCH\n"
        "def train_models(x):\n"
        "    criterion = x\n"
        "    return criterion\n"
        "=======\n"
        "def train_models(x):\n"
        "    fold_predictions = [x for _ in range(5)]\n"
        "    return sum(fold_predictions) / len(fold_predictions)\n"
        ">>>>>>> REPLACE\n"
    )

    result = run_atomic_actuation_pipeline(
        agent,
        strategy_memo=_strategy_memo(),
        parent_code=parent_code,
        task_description="authorship classification",
    )

    assert result["status"] == "accepted"
    assert result["planner"]["contract_attempts"][0]["valid"] is False
    assert "non-top-level symbols" in " ".join(
        result["planner"]["contract_attempts"][0]["violations"]
    )
    assert result["planner"]["plan"]["allowed_changes"][0][
        "target_symbols"
    ] == ["train_models"]
