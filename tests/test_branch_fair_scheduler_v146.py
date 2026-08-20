from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace

from engine.search_node import SearchNode


def _metric(value: float):
    return SimpleNamespace(value=value, maximize=False)


def _candidate(
    node_id: str,
    parent: SearchNode,
    *,
    buggy: bool | None = False,
    pending: bool = False,
    metric: float | None = 1.0,
) -> SearchNode:
    node = SearchNode(
        id=node_id,
        code="print('candidate')\n",
        plan=node_id,
        parent=parent,
        stage="improve",
        draft_role=parent.draft_role,
        is_buggy=buggy,
        is_valid=buggy is False,
        metric=_metric(metric) if metric is not None else None,
        branch_id=parent.branch_id,
    )
    node.pending_execution = pending
    return node


def _agent():
    virtual_root = SearchNode(
        id="root",
        code="",
        plan="root",
        stage="root",
        step=0,
    )
    replay = SearchNode(
        id="replay-root",
        code="print('replay')\n",
        plan="replay",
        parent=virtual_root,
        stage="draft",
        draft_role="memory_reproduction",
        branch_id=1,
        is_buggy=False,
        is_valid=True,
        metric=_metric(0.0001),
    )
    novel = SearchNode(
        id="novel-root",
        code="print('novel')\n",
        plan="novel",
        parent=virtual_root,
        stage="draft",
        draft_role="novel_exploration",
        branch_id=2,
        is_buggy=False,
        is_valid=True,
        metric=_metric(0.4),
    )
    fusion = SearchNode(
        id="fusion-root",
        code="print('fusion')\n",
        plan="fusion",
        parent=virtual_root,
        stage="fusion_draft",
        draft_role="novel_exploration",
        role_contract={"behavioral_role": "cross_role_synthesis"},
        branch_id=3,
        is_buggy=False,
        is_valid=True,
        metric=_metric(0.2),
    )
    policy = SimpleNamespace(
        enabled=True,
        roles=["memory_reproduction", "novel_exploration"],
        ensure_valid_candidate_per_role=True,
        cross_role_synthesis_after_balance=True,
        cross_role_synthesis_on_coverage=True,
        equal_branch_allocation_after_coverage=True,
        single_coverage_synthesis_only=True,
    )
    agent = SimpleNamespace(
        virtual_root=virtual_root,
        acfg=SimpleNamespace(draft_role_policy=policy),
        fusion_draft_count=1,
        max_fusion_drafts=1,
        branch_all_nodes={1: [replay], 2: [novel], 3: [fusion]},
        branch_successful_nodes={1: [replay], 2: [novel], 3: [fusion]},
        role_balance_status=lambda: {
            "enabled": True,
            "active": False,
            "all_slots_reserved": True,
            "deficit_roles": [],
        },
    )
    return agent, replay, novel, fusion


def test_fairness_counts_failed_debug_and_inflight_attempts_without_scores():
    from engine.role_balance import build_branch_fairness_status

    agent, replay, novel, fusion = _agent()
    agent.branch_all_nodes[1].extend(
        [
            _candidate("replay-failed", replay, buggy=True, metric=None),
            _candidate(
                "replay-inflight",
                replay,
                buggy=None,
                pending=True,
                metric=None,
            ),
        ]
    )
    agent.branch_all_nodes[2].append(
        _candidate("novel-debug", novel, buggy=False, metric=99.0)
    )

    status = build_branch_fairness_status(agent)

    assert status["active"] is True
    assert {
        row["name"]: (
            row["attempted_count"],
            row["completed_count"],
            row["in_flight_count"],
        )
        for row in status["branches"]
    } == {
        "replay": (3, 2, 1),
        "novel": (2, 2, 0),
        "fusion": (1, 1, 0),
    }
    assert status["next_branch_name"] == "fusion"


def test_fair_selector_chooses_least_used_branch_not_global_best(monkeypatch):
    from engine import node_selection
    from engine.role_balance import build_branch_fairness_status

    agent, replay, novel, fusion = _agent()
    # Replay has the best score but already consumed the most attempts.
    agent.branch_all_nodes[1].extend(
        [
            _candidate("replay-2", replay, metric=0.00001),
            _candidate("replay-3", replay, metric=0.000001),
        ]
    )
    agent.branch_all_nodes[2].append(_candidate("novel-2", novel, metric=5.0))
    calls = []

    def _select(_agent, root, branch_id):
        calls.append(branch_id)
        return root

    monkeypatch.setattr(node_selection, "_select_within_branch", _select)

    selected = node_selection.select_branch_fair(
        agent,
        build_branch_fairness_status(agent),
    )

    assert selected is fusion
    assert calls == [3]


def test_temporarily_blocked_branch_is_skipped_without_erasing_its_debt(
    monkeypatch,
):
    from engine import node_selection
    from engine.role_balance import build_branch_fairness_status

    agent, replay, novel, fusion = _agent()
    agent.branch_all_nodes[1].append(_candidate("replay-2", replay))
    agent.branch_all_nodes[2].append(_candidate("novel-2", novel))
    calls = []

    def _select(_agent, root, branch_id):
        calls.append(branch_id)
        return None if branch_id == 3 else root

    monkeypatch.setattr(node_selection, "_select_within_branch", _select)
    before = build_branch_fairness_status(agent)

    selected = node_selection.select_branch_fair(agent, before)

    assert selected is replay
    assert calls == [3, 1]
    after = build_branch_fairness_status(agent)
    assert after["next_branch_name"] == "fusion"


def test_single_coverage_fusion_disables_every_later_cross_branch_entry():
    from engine.conditions import (
        cross_role_synthesis_allowed,
        should_trigger_branch_fusion,
    )

    agent, _, _, _ = _agent()

    assert cross_role_synthesis_allowed(agent, component="test") is False
    assert should_trigger_branch_fusion(agent) is False


def test_mandatory_repair_claim_respects_host_selected_branch(monkeypatch):
    from engine.agent_search import AgentSearch

    agent, replay, novel, _ = _agent()
    replay_repair = _candidate("replay-repair", replay, buggy=True, metric=None)
    novel_repair = _candidate("novel-repair", novel, buggy=True, metric=None)
    search = AgentSearch.__new__(AgentSearch)
    search._mandatory_repair_lock = threading.Lock()
    search._mandatory_repair_queue = deque([replay_repair, novel_repair])
    search._mandatory_repair_queued_ids = {
        replay_repair.id,
        novel_repair.id,
    }
    search._mandatory_repair_inflight_ids = set()
    monkeypatch.setattr(
        AgentSearch,
        "_is_mandatory_repair_parent",
        staticmethod(lambda node: node is not None and node.is_buggy is True),
    )

    selected, duplicate = search._claim_mandatory_repair_parent(
        None,
        required_branch_id=2,
    )

    assert selected is novel_repair
    assert duplicate is False
    assert list(search._mandatory_repair_queue) == [replay_repair]
    assert replay_repair.id in search._mandatory_repair_queued_ids
