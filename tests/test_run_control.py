from pathlib import Path
import sys
import threading
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from engine.run_control import (  # noqa: E402
    focused_protocol_status,
    focused_protocol_success_error,
    should_continue_focused_search,
)
from engine.agent_search import AgentSearch  # noqa: E402
from engine.search_node import Journal, SearchNode  # noqa: E402
from utils.metric import WorstMetricValue  # noqa: E402


def _node(*, state, ctime, role="memory_reproduction", clean=False):
    return SimpleNamespace(
        ctime=ctime,
        draft_role=role,
        protocol_repair={"state": state},
        replay_status="staged_protocol_repair_executed_clean" if clean else "staged_protocol_repair",
        leakage_audit={"status": "clean", "rank_eligible": True} if clean else {},
        metric=SimpleNamespace(value=0.2) if clean else None,
    )


def test_focused_protocol_status_uses_only_latest_target_role_node():
    nodes = [
        _node(state="superseded", ctime=1),
        _node(state="completed", ctime=3, role="novel_exploration", clean=True),
        _node(state="final_pending", ctime=2),
    ]
    status = focused_protocol_status(nodes, "memory_reproduction")
    assert status.seen is True
    assert status.state == "final_pending"
    assert status.active is True


def test_active_focused_protocol_outlives_shared_step_budget():
    status = focused_protocol_status(
        [_node(state="final_pending", ctime=1)],
        "memory_reproduction",
    )
    assert should_continue_focused_search(
        completed_steps=10,
        total_steps=10,
        status=status,
        focus_in_flight=False,
    ) is True


def test_inflight_focused_draft_cannot_be_cut_off_before_transaction_exists():
    status = focused_protocol_status([], "memory_reproduction")
    assert should_continue_focused_search(
        completed_steps=10,
        total_steps=10,
        status=status,
        focus_in_flight=True,
    ) is True


def test_focused_job_requires_clean_ranked_metric():
    clean = focused_protocol_status(
        [_node(state="completed", ctime=1, clean=True)],
        "memory_reproduction",
    )
    assert focused_protocol_success_error(clean) is None

    missing_metric_node = _node(state="completed", ctime=2, clean=True)
    missing_metric_node.metric = None
    missing_metric = focused_protocol_status([missing_metric_node], "memory_reproduction")
    assert focused_protocol_success_error(missing_metric) == "the completed transaction has no metric"


def test_exhausted_focused_protocol_fails_job_outcome():
    status = focused_protocol_status(
        [_node(state="exhausted", ctime=1)],
        "memory_reproduction",
    )
    assert "state=exhausted" in focused_protocol_success_error(status)
    assert should_continue_focused_search(
        completed_steps=3,
        total_steps=10,
        status=status,
        focus_in_flight=False,
    ) is False


def _fixed_role_agent():
    agent = AgentSearch.__new__(AgentSearch)
    agent.acfg = SimpleNamespace(
        initial_drafts=3,
        steps=10,
        draft_role_policy=SimpleNamespace(
            enabled=True,
            roles=["coldstart_baseline", "memory_reproduction", "novel_exploration"],
        ),
    )
    agent.scfg = SimpleNamespace(num_drafts=3)
    agent._draft_role_lock = threading.Lock()
    return agent


def test_fixed_root_slots_stop_after_filtered_replay_and_novel_claims():
    agent = _fixed_role_agent()
    agent._draft_generation_count = 1

    assert agent.claim_draft_role("memory_reproduction") == "memory_reproduction"
    assert agent.fixed_draft_slots_exhausted() is False
    assert agent.claim_draft_role() == "novel_exploration"
    assert agent.fixed_draft_slots_exhausted() is True


def test_filtered_replay_does_not_request_illegal_fourth_root_role(monkeypatch):
    from engine import node_selection

    root = SearchNode(code="", plan="root", stage="root", step=0, metric=WorstMetricValue())
    for role in ("memory_reproduction", "novel_exploration"):
        SearchNode(code="print(1)", plan=role, stage="draft", parent=root, draft_role=role, lock=True)
    root.expected_child_count = 2
    agent = _fixed_role_agent()
    agent.virtual_root = root
    agent._draft_generation_count = 3

    monkeypatch.setattr(node_selection, "_compute_exploration_constant", lambda _agent: 1.0)
    assert node_selection.select(agent, root) is None


def test_parallel_repair_lanes_do_not_steal_each_others_roles():
    from engine.search_node import Journal

    root = SearchNode(code="", plan="root", stage="root")
    transaction = {
        "transaction_id": "tx",
        "protocol_plan": {"stages": ["data_scope"]},
        "current_stage_index": 0,
        "state": "pending",
    }
    replay = SearchNode(
        code="replay",
        plan="replay",
        stage="draft",
        parent=root,
        draft_role="memory_reproduction",
        protocol_repair={**transaction, "transaction_id": "replay-tx"},
        leakage_audit={"status": "blocked", "repair_required": True},
        audit_repair_required=True,
    )
    novel = SearchNode(
        code="novel",
        plan="novel",
        stage="draft",
        parent=root,
        draft_role="novel_exploration",
        protocol_repair={**transaction, "transaction_id": "novel-tx"},
        leakage_audit={"status": "blocked", "repair_required": True},
        audit_repair_required=True,
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.journal = Journal(nodes=[root, replay, novel])
    agent._init_mandatory_repair_scheduler()
    agent._mandatory_repair_queue.extend([replay, novel])
    agent._mandatory_repair_queued_ids.update({replay.id, novel.id})

    normal_claim, duplicate = agent._claim_mandatory_repair_parent(
        None,
        excluded_draft_role="memory_reproduction",
    )
    assert duplicate is False
    assert normal_claim is novel

    replay_claim, duplicate = agent._claim_mandatory_repair_parent(
        None,
        required_draft_role="memory_reproduction",
    )
    assert duplicate is False
    assert replay_claim is replay


def test_mandatory_repair_does_not_steal_explicit_runtime_debug(monkeypatch):
    root = SearchNode(code="", plan="root", stage="root")
    runtime_failure = SearchNode(
        code="raise FileNotFoundError('missing model')",
        plan="runtime failure",
        stage="draft",
        parent=root,
        draft_role="memory_reproduction",
        is_buggy=True,
        leakage_audit={"status": "clean", "repair_required": False},
    )
    audit_failure = SearchNode(
        code="fit_before_split()",
        plan="audit failure",
        stage="draft",
        parent=root,
        draft_role="memory_reproduction",
        is_buggy=True,
        is_valid=False,
        audit_repair_required=True,
        leakage_audit={"status": "blocked", "repair_required": True},
    )

    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.journal = Journal(nodes=[root, runtime_failure, audit_failure])
    agent.data_preview = "ready"
    agent.search_start_time = 1.0
    agent.current_step = 0
    agent.branch_all_nodes = {}
    agent.best_node = None
    agent._init_mandatory_repair_scheduler()
    agent._enqueue_mandatory_repair(audit_failure)

    selected = []

    def fake_run_single_step(parent_node, **_kwargs):
        selected.append(parent_node)
        return False, None

    monkeypatch.setattr(agent, "_run_single_step", fake_run_single_step)

    AgentSearch.step(
        agent,
        runtime_failure,
        exec_callback=lambda *_args, **_kwargs: None,
        mandatory_repair_role="memory_reproduction",
    )

    assert selected == [runtime_failure]
    assert list(agent._mandatory_repair_queue) == [audit_failure]
    assert not agent._mandatory_repair_inflight_ids
    assert not agent._is_explicit_runtime_debug_parent(audit_failure)

    AgentSearch.step(
        agent,
        audit_failure,
        exec_callback=lambda *_args, **_kwargs: None,
        mandatory_repair_role="memory_reproduction",
    )
    assert selected == [runtime_failure, audit_failure]
    assert not agent._mandatory_repair_queue
    assert not agent._mandatory_repair_inflight_ids


def test_explicit_runtime_debug_guard_covers_invalid_and_excludes_terminal():
    node = SearchNode(
        code="print('invalid result')",
        plan="invalid runtime result",
        stage="debug",
        is_buggy=False,
        is_valid=False,
        leakage_audit={"status": "clean", "repair_required": False},
    )

    assert AgentSearch._is_explicit_runtime_debug_parent(node) is True

    node.is_terminal = True
    assert AgentSearch._is_explicit_runtime_debug_parent(node) is False
