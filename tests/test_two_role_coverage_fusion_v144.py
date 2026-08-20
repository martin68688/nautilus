from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.search_node import SearchNode


def _metric(value: float):
    return SimpleNamespace(value=value)


def _node(
    node_id: str,
    role: str,
    metric: float,
    *,
    replay_source: dict | None = None,
    role_contract: dict | None = None,
) -> SearchNode:
    return SearchNode(
        id=node_id,
        code=f"print({node_id!r})\n",
        plan=node_id,
        stage="improve",
        draft_role=role,
        metric=_metric(metric),
        is_buggy=False,
        is_valid=True,
        replay_source=replay_source or {},
        role_contract=role_contract or {},
    )


def _two_role_policy(**overrides):
    values = {
        "enabled": True,
        "roles": ["memory_reproduction", "novel_exploration"],
        "ensure_valid_candidate_per_role": True,
        "role_balance_min_valid_candidates": 1,
        "cross_role_synthesis_after_balance": True,
        "cross_role_synthesis_on_coverage": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_two_role_policy_accepts_matching_two_draft_counts():
    from engine.agent_search import AgentSearch

    agent = AgentSearch.__new__(AgentSearch)
    agent.acfg = SimpleNamespace(
        initial_drafts=2,
        draft_role_policy=_two_role_policy(),
    )
    agent.scfg = SimpleNamespace(num_drafts=2)

    AgentSearch._validate_draft_role_policy(agent)
    assert AgentSearch.configured_draft_role(agent, 0) == "memory_reproduction"
    assert AgentSearch.configured_draft_role(agent, 1) == "novel_exploration"

    agent.scfg.num_drafts = 3
    with pytest.raises(ValueError, match=r"num_drafts == len\(roles\) \(2\)"):
        AgentSearch._validate_draft_role_policy(agent)


def test_coverage_mode_rejects_non_two_role_contract():
    from engine.agent_search import AgentSearch

    agent = AgentSearch.__new__(AgentSearch)
    agent.acfg = SimpleNamespace(
        initial_drafts=3,
        draft_role_policy=_two_role_policy(
            roles=[
                "coldstart_baseline",
                "memory_reproduction",
                "novel_exploration",
            ]
        ),
    )
    agent.scfg = SimpleNamespace(num_drafts=3)

    with pytest.raises(ValueError, match="requires exactly the independent"):
        AgentSearch._validate_draft_role_policy(agent)


def test_replay_derived_novel_cannot_satisfy_independent_novel_coverage():
    from engine.role_balance import build_role_balance_status

    replay = _node("replay", "memory_reproduction", 0.001)
    replay_derived = _node(
        "replay-derived",
        "novel_exploration",
        0.0005,
        replay_source={"lineage_kind": "replay_derived_novel"},
        role_contract={"behavioral_role": "replay_derived_novel"},
    )
    agent = SimpleNamespace(
        acfg=SimpleNamespace(draft_role_policy=_two_role_policy()),
        fixed_draft_slots_exhausted=lambda: True,
        branch_all_nodes={1: [replay], 2: [replay_derived]},
        branch_successful_nodes={1: [replay], 2: [replay_derived]},
    )

    status = build_role_balance_status(agent)

    assert status["valid_counts"] == {
        "memory_reproduction": 1,
        "novel_exploration": 0,
    }
    assert status["deficit_roles"] == ["novel_exploration"]

    independent = _node("independent", "novel_exploration", 0.01)
    agent.branch_all_nodes[3] = [independent]
    agent.branch_successful_nodes[3] = [independent]
    status = build_role_balance_status(agent)
    assert status["active"] is False


def test_first_fusion_is_due_on_coverage_then_reverts_to_legacy_conditions():
    from engine.conditions import coverage_synthesis_due, should_trigger_branch_fusion

    agent = SimpleNamespace(
        acfg=SimpleNamespace(draft_role_policy=_two_role_policy()),
        fusion_draft_count=0,
        max_fusion_drafts=2,
        role_balance_status=lambda: {
            "enabled": True,
            "active": False,
            "all_slots_reserved": True,
            "deficit_roles": [],
        },
        search_start_time=None,
    )

    assert coverage_synthesis_due(agent) is True
    assert should_trigger_branch_fusion(agent) is True

    agent.fusion_draft_count = 1
    assert coverage_synthesis_due(agent) is False
    assert should_trigger_branch_fusion(agent) is False


def test_soft_switch_prioritizes_due_coverage_fusion(monkeypatch):
    from engine import node_selection

    sentinel = object()
    agent = SimpleNamespace(
        acfg=SimpleNamespace(draft_role_policy=_two_role_policy()),
        fusion_draft_count=0,
        max_fusion_drafts=2,
        role_balance_status=lambda: {
            "enabled": True,
            "active": False,
            "all_slots_reserved": True,
            "deficit_roles": [],
        },
        virtual_root=object(),
    )
    monkeypatch.setattr(node_selection, "select", lambda _agent, _root: sentinel)

    assert node_selection.select_with_soft_switch(agent) is sentinel


def test_coverage_aggregation_selects_best_exact_replay_and_independent_novel(
    monkeypatch,
):
    from agents import aggregation_agent
    from agents import leakage_audit
    from authority.adapters.mlevolve import ranking_gate

    replay = _node("replay", "memory_reproduction", 0.001)
    replay_worse = _node("replay-worse", "memory_reproduction", 0.01)
    replay_derived = _node(
        "replay-derived",
        "novel_exploration",
        0.0001,
        replay_source={"lineage_kind": "replay_derived_novel"},
    )
    independent = _node("independent", "novel_exploration", 0.008)
    agent = SimpleNamespace(
        acfg=SimpleNamespace(draft_role_policy=_two_role_policy()),
        branch_successful_nodes={
            1: [replay, replay_worse, replay_derived],
            2: [independent],
        },
        metric_maximize=False,
    )
    monkeypatch.setattr(ranking_gate, "filter_ranked_nodes", lambda _a, rows, **_k: rows)
    monkeypatch.setattr(ranking_gate, "authorize_selection", lambda *_a, **_k: True)
    monkeypatch.setattr(leakage_audit, "legacy_rank_eligible", lambda *_a, **_k: True)

    representatives = aggregation_agent._collect_branch_representatives(agent)

    assert [node.id for node in representatives] == ["replay", "independent"]
