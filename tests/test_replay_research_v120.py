from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

MANIFEST = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "replay_research_v120"
    / "leaf_replay_research_targets.json"
)

V122_SYSTEM = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "systems_v122"
    / "dynamic_hybrid.yaml"
)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_v122_system_uses_fresh_runtime_identity() -> None:
    text = V122_SYSTEM.read_text(encoding="utf-8")
    assert "extends: ../systems_v121/dynamic_hybrid.yaml" in text
    assert "/tmp/nautilus-exp-end2end-agent-v122-runtime/" in text
    assert "manifests_v122/leaf_official_replay_targets.json" in text
    assert "transition_evidence_v122/" in text
    assert "leaf_replay_research_top5_v122_alignment_gate_pod32" in text


def test_v120_portfolio_separates_score_and_diverse_frontiers() -> None:
    from agents.memory.run_forest_replay import _task_replay_portfolio

    portfolio = _task_replay_portfolio(_manifest(), "leaf-classification")
    assert portfolio["anchor_target_id"] == (
        "leaf-official-runforest-efficientnet-b3-00168"
    )
    assert len(portfolio["score_frontier_target_ids"]) == 5
    assert len(portfolio["diverse_frontier_target_ids"]) == 5
    targets = portfolio["targets_by_id"]
    diverse = [targets[target_id] for target_id in portfolio["diverse_frontier_target_ids"]]
    assert len({target["code_sha256"] for target in diverse}) == 5
    assert len({target["architecture_signature"] for target in diverse}) == 5
    assert {
        targets[target_id]["validation_protocol"]
        for target_id in portfolio["score_frontier_target_ids"]
    } == {"official_kaggle_scored_test", "full_oof_internal"}
    assert "never numerically blend" in portfolio["metric_comparison_policy"]


def test_v120_portfolio_fails_closed_on_duplicate_code_or_architecture() -> None:
    from agents.memory.run_forest_replay import _task_replay_portfolio

    duplicate_code = _manifest()
    duplicate_code["targets"][1]["code_sha256"] = duplicate_code["targets"][0][
        "code_sha256"
    ]
    with pytest.raises(ValueError, match="code_sha256"):
        _task_replay_portfolio(duplicate_code, "leaf-classification")

    duplicate_architecture = _manifest()
    by_id = {
        target["target_id"]: target for target in duplicate_architecture["targets"]
    }
    diverse_ids = duplicate_architecture["portfolios"][0][
        "diverse_frontier_target_ids"
    ]
    by_id[diverse_ids[1]]["architecture_signature"] = by_id[diverse_ids[0]][
        "architecture_signature"
    ]
    with pytest.raises(ValueError, match="architecture_signature"):
        _task_replay_portfolio(duplicate_architecture, "leaf-classification")


def test_replay_research_cards_preserve_authority_and_bound_code_attention() -> None:
    from agents.memory.run_forest_replay import replay_research_strategy_cards

    code = "start\n" + ("x = 1\n" * 1000) + "end\n"
    result = {
        "code": code,
        "source_ref_ids": ["run::source", "transition::source"],
        "replay_source": {
            "target_id": "research-a",
            "target_role": "research",
            "method_family": "family-a",
            "architecture_signature": "architecture-a",
            "historical_metric": 0.01,
            "validation_protocol": "full_oof_internal",
            "metric_authority": "strict_internal_observed",
            "code_sha256": "a" * 64,
            "graph_node_id": "run::source",
            "allowed_actions": ["replay_component_transplant"],
            "component_blueprint": {"model": "A"},
            "target_audit_status": "verified_clean",
            "known_issue_codes": [],
            "exact_replay_eligible": True,
            "requires_repair": False,
        },
    }
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            draft_role_policy=SimpleNamespace(
                replay_research_strategy_slots=1,
                replay_research_card_max_chars=1000,
            )
        ),
        _replay_research_results={"research-a": result},
        _replay_research_strategy_target_ids=["research-a"],
    )
    cards = replay_research_strategy_cards(agent)
    assert [card["memory_id"] for card in cards] == [
        "replay-target::research-a"
    ]
    card = cards[0]
    assert card["metric_authority"] == "strict_internal_observed"
    assert card["source_code_truncated"] is True
    assert len(card["source_code"]) < len(code)
    assert "hash-bound source truncated" in card["source_code"]
    assert card["exact_replay_eligible"] is True


def test_exact_research_node_keeps_source_identity_and_forbids_hidden_labels() -> None:
    from agents import replay_research_agent
    from engine.search_node import SearchNode

    anchor_code = "print('anchor')\n"
    target_code = "print('diverse')\n"
    anchor = SearchNode(
        code=anchor_code,
        plan="anchor",
        stage="draft",
        branch_id=1,
        replay_source={
            "target_role": "anchor",
            "exact_replay_execution": True,
            "code_sha256": "anchor-source",
        },
        replay_status="historical_exact_anchor_loaded",
        is_buggy=False,
        is_valid=True,
    )
    replay = {
        "code": target_code,
        "plan": "diverse exact",
        "source_ref_ids": ["run::diverse"],
        "replay_status": "historical_exact_research_loaded",
        "replay_source": {
            "target_id": "diverse-a",
            "target_role": "research",
            "exact_replay_execution": True,
            "exact_replay_eligible": True,
            "code_sha256": __import__("hashlib").sha256(
                target_code.encode("utf-8")
            ).hexdigest(),
            "graph_node_id": "run::diverse",
            "validation_protocol": "full_oof_internal",
            "metric_authority": "strict_internal_observed",
            "architecture_signature": "different-lineage",
        },
    }
    agent = SimpleNamespace(
        _replay_research_results={"diverse-a": replay},
        _replay_research_portfolio_receipt={"schema": "receipt"},
        _serialize_prompt=lambda prompt: json.dumps(prompt, sort_keys=True),
        prospective_audit=None,
        evaluation_authority=None,
        next_branch_id=2,
        branch_all_nodes={1: [anchor]},
        branch_successful_nodes={1: [anchor]},
        adoption_tracking_enabled=False,
    )
    node = replay_research_agent.run(agent, anchor, "diverse-a")
    assert node.code == target_code
    assert node.skip_code_review is True
    assert node.replay_status == "historical_exact_research_loaded"
    assert node.replay_source["research_action"] == "replay_exact_diverse"
    assert node.replay_source["source_target_ids"] == ["diverse-a"]
    assert node.replay_source["hidden_terminal_labels_used"] is False
    assert node.role_contract["source_target_ids"] == ["diverse-a"]
    assert node in agent.branch_all_nodes[1]


def test_replay_research_scheduler_waits_for_successful_anchor_and_obeys_budget() -> None:
    # AgentSearch imports the normal execution stack; this test is also run in
    # the release/container environment where pandas/scipy match NumPy.
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal, SearchNode

    anchor = SearchNode(
        code="print('anchor')",
        plan="anchor",
        stage="draft",
        branch_id=1,
        replay_source={
            "target_role": "anchor",
            "exact_replay_execution": True,
        },
        replay_status="historical_exact_anchor_loaded",
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.acfg = SimpleNamespace(
        draft_role_policy=SimpleNamespace(
            replay_research_enabled=True,
            replay_research_exact_budget=2,
        )
    )
    agent.journal = Journal(nodes=[anchor])
    agent._search_condition = None
    AgentSearch._init_replay_research_scheduler(agent)
    AgentSearch.register_replay_research_portfolio(
        agent,
        {
            "results": {"a": {"code": "a"}, "b": {"code": "b"}, "c": {"code": "c"}},
            "exact_research_target_ids": ["a", "b", "c"],
            "strategy_target_ids": ["a", "b", "c"],
            "receipt": {"schema": "receipt"},
        },
    )
    assert AgentSearch._claim_replay_research_target(agent) is None
    anchor.is_buggy = False
    anchor.is_valid = True
    first = AgentSearch._claim_replay_research_target(agent)
    second = AgentSearch._claim_replay_research_target(agent)
    assert first == (anchor, "a")
    assert second == (anchor, "b")
    assert AgentSearch._claim_replay_research_target(agent) is None


def test_v120_release_and_system_request_six_hours_without_sixteen_step_cap() -> None:
    build_text = (
        REPO
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "build_leaf_replay_research_v120_release.py"
    ).read_text(encoding="utf-8")
    system_text = (
        REPO
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v120"
        / "dynamic_hybrid.yaml"
    ).read_text(encoding="utf-8")
    assert '"agent_steps": 80' in build_text
    assert '"agent_time_limit_seconds": 21600' in build_text
    assert '"execution_timeout_seconds": 3600' in build_text
    assert "replay_research_exact_budget: 2" in system_text
    assert "replay_research_strategy_slots: 4" in system_text
    assert "Compatibility Preflight" not in system_text


def test_v121_reissue_keeps_six_hours_and_matches_standalone_pod_memory_limit() -> None:
    build_text = (
        REPO
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "build_leaf_replay_research_v121_release.py"
    ).read_text(encoding="utf-8")
    system_text = (
        REPO
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v121"
        / "dynamic_hybrid.yaml"
    ).read_text(encoding="utf-8")
    assert '"agent_steps": 80' in build_text
    assert '"agent_time_limit_seconds": 21600' in build_text
    assert '"memory_gib": 32' in build_text
    assert '"--source-manifest-version": "120"' in build_text
    assert "agent-v121-runtime" in system_text
