from __future__ import annotations

from types import SimpleNamespace

from engine.node_selection import select_role_balance_deficit
from engine.role_balance import build_role_balance_status
from engine.search_node import SearchNode


def _metric(value: float | None):
    return SimpleNamespace(value=value)


def _node(root, role: str, node_id: str, *, metric=None, buggy=False, valid=True):
    return SearchNode(
        id=node_id,
        code="print('candidate')\n",
        plan=node_id,
        parent=root,
        stage="draft",
        draft_role=role,
        metric=_metric(metric),
        is_buggy=buggy,
        is_valid=valid,
    )


def _agent(minimum: int = 2):
    agent = SimpleNamespace()
    roles = [
        "coldstart_baseline",
        "memory_reproduction",
        "novel_exploration",
    ]
    agent.acfg = SimpleNamespace(
        draft_role_policy=SimpleNamespace(
            enabled=True,
            roles=roles,
            ensure_valid_candidate_per_role=True,
            role_balance_min_valid_candidates=minimum,
        )
    )
    agent.scfg = SimpleNamespace(num_improves=3, num_bugs=1, num_drafts=3)
    agent._draft_role_lock = None
    agent._draft_generation_count = 3
    agent.fixed_draft_slots_exhausted = lambda: True
    agent.virtual_root = SearchNode(
        id="root",
        code="",
        plan="root",
        parent=None,
        stage="root",
    )
    agent.is_root = lambda node: node is agent.virtual_root
    roots = {
        role: _node(agent.virtual_root, role, role, metric=None, buggy=None, valid=None)
        for role in roles
    }
    for index, node in enumerate(roots.values(), start=1):
        node.branch_id = index
        node.step = 1
    agent.branch_all_nodes = {
        node.branch_id: [node] for node in roots.values()
    }
    agent.branch_successful_nodes = {
        node.branch_id: [] for node in roots.values()
    }
    return agent, roots


def _add_valid(agent, root, node_id: str, score: float):
    node = SearchNode(
        id=node_id,
        code="print('valid')\n",
        plan=node_id,
        parent=root,
        stage="improve",
        draft_role=root.draft_role,
        metric=_metric(score),
        is_buggy=False,
        is_valid=True,
    )
    node.branch_id = root.branch_id
    node.step = 2
    agent.branch_all_nodes[root.branch_id].append(node)
    agent.branch_successful_nodes[root.branch_id].append(node)
    return node


def test_balance_uses_valid_candidate_counts_not_metric_magnitude():
    agent, roots = _agent(minimum=2)
    _add_valid(agent, roots["coldstart_baseline"], "cold-1", 0.90)
    _add_valid(agent, roots["memory_reproduction"], "replay-1", 0.001)
    _add_valid(agent, roots["memory_reproduction"], "replay-2", 0.002)

    status = build_role_balance_status(agent)

    assert status["active"] is True
    assert status["valid_counts"] == {
        "coldstart_baseline": 1,
        "memory_reproduction": 2,
        "novel_exploration": 0,
    }
    assert status["next_role"] == "novel_exploration"


def test_balance_keeps_low_scoring_role_until_protected_minimum_is_met():
    agent, roots = _agent(minimum=2)
    for role, root in roots.items():
        _add_valid(agent, root, f"{role}-1", 100.0 if role == "novel_exploration" else 0.001)

    first_status = build_role_balance_status(agent)
    assert first_status["next_role"] == "coldstart_baseline"

    _add_valid(agent, roots["coldstart_baseline"], "cold-2", 0.50)
    _add_valid(agent, roots["memory_reproduction"], "replay-2", 0.0005)
    _add_valid(agent, roots["novel_exploration"], "novel-2", 200.0)

    final_status = build_role_balance_status(agent)
    assert final_status["active"] is False
    assert final_status["next_role"] is None


def test_balance_selector_targets_novel_branch_instead_of_best_replay():
    agent, roots = _agent(minimum=1)
    replay = _add_valid(agent, roots["memory_reproduction"], "replay-best", 0.001)
    replay.metric = _metric(0.001)
    roots["novel_exploration"].lock = False

    selected = select_role_balance_deficit(agent, "novel_exploration")

    assert selected is roots["novel_exploration"]
    assert selected.draft_role == "novel_exploration"
    assert selected.lock is True


def test_host_instrumentation_failure_does_not_count_as_completed_role_work():
    agent, roots = _agent(minimum=1)
    novel = roots["novel_exploration"]
    novel.is_buggy = True
    novel.is_valid = False
    novel.exc_type = "HostSourceInstrumentationError"

    status = build_role_balance_status(agent)

    assert status["completed_counts"]["novel_exploration"] == 0
    assert status["host_instrumentation_failures"]["novel_exploration"] == 1
    assert status["next_role"] == "novel_exploration"


def test_cross_role_synthesis_is_blocked_until_role_balance_is_complete():
    from engine.conditions import cross_role_synthesis_allowed, should_trigger_branch_fusion

    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            draft_role_policy=SimpleNamespace(
                enabled=True,
                cross_role_synthesis_after_balance=True,
            )
        ),
        role_balance_status=lambda: {
            "enabled": True,
            "active": True,
            "all_slots_reserved": True,
            "deficit_roles": ["novel_exploration"],
        },
    )

    assert should_trigger_branch_fusion(agent) is False
    assert cross_role_synthesis_allowed(agent, component="test") is False

    agent.role_balance_status = lambda: {
        "enabled": True,
        "active": False,
        "all_slots_reserved": True,
        "deficit_roles": [],
    }
    assert cross_role_synthesis_allowed(agent, component="test") is True


def test_fixed_role_root_can_request_new_fusion_branch_after_balance(monkeypatch):
    import engine.node_selection as node_selection

    agent, _ = _agent(minimum=1)
    agent.acfg.draft_role_policy.cross_role_synthesis_after_balance = True
    agent.acfg.branch_fusion_trigger_prob = 1.0
    agent.role_balance_status = lambda: {
        "enabled": True,
        "active": False,
        "all_slots_reserved": True,
        "deficit_roles": [],
    }
    monkeypatch.setattr(node_selection, "should_trigger_branch_fusion", lambda _: True)
    monkeypatch.setattr(node_selection.random, "random", lambda: 0.0)

    selected = node_selection.select(agent, agent.virtual_root)

    assert selected is agent.virtual_root
    assert agent.virtual_root._aggregation_requested is True


def test_cross_role_synthesis_stays_blocked_before_all_roles_are_reserved():
    from engine.conditions import cross_role_synthesis_allowed

    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            draft_role_policy=SimpleNamespace(
                enabled=True,
                cross_role_synthesis_after_balance=True,
            )
        ),
        role_balance_status=lambda: {
            "enabled": True,
            "active": False,
            "all_slots_reserved": False,
            "deficit_roles": [],
        },
    )

    assert cross_role_synthesis_allowed(agent, component="test") is False
