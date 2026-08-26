from pathlib import Path
from types import SimpleNamespace

from engine.draft_roles import (
    TRANSFER_PORTFOLIO_DRAFT_ROLES,
    canonical_draft_role,
    memory_retrieval_draft_role,
    transfer_portfolio_seat_for_role,
)
from engine.search_node import SearchNode
from engine.transfer_tournament import build_transfer_tournament_status
from engine import node_selection
from utils.metric import MetricValue


def _agent(tmp_path: Path):
    policy = SimpleNamespace(
        transfer_portfolio_tournament_enabled=True,
        roles=list(TRANSFER_PORTFOLIO_DRAFT_ROLES),
        transfer_portfolio_round1_valid_successors_per_seat=1,
        transfer_portfolio_survivor_count=2,
        transfer_portfolio_finalist_valid_successors_per_seat=2,
    )
    virtual_root = SearchNode(code="", stage="root", step=0)
    agent = SimpleNamespace(
        acfg=SimpleNamespace(draft_role_policy=policy),
        cfg=SimpleNamespace(log_dir=str(tmp_path)),
        virtual_root=virtual_root,
        branch_all_nodes={},
        metric_maximize=True,
        scfg=SimpleNamespace(parallel_search_num=1),
    )
    for branch_id, role in enumerate(TRANSFER_PORTFOLIO_DRAFT_ROLES, start=1):
        root = SearchNode(
            code=f"# {role}",
            parent=virtual_root,
            stage="draft",
            step=branch_id,
            branch_id=branch_id,
            draft_role=role,
        )
        agent.branch_all_nodes[branch_id] = [root]
    return agent


def _success(agent, role: str, metric: float, *, stage: str = "improve"):
    root = next(
        node
        for node in agent.virtual_root.children
        if node.draft_role == role
    )
    branch_nodes = agent.branch_all_nodes[root.branch_id]
    parent = branch_nodes[-1]
    node = SearchNode(
        code=parent.code + "\n# successor",
        parent=parent,
        stage=stage,
        step=len(branch_nodes) + 4,
        branch_id=root.branch_id,
        draft_role=role,
        is_buggy=False,
        is_valid=True,
        metric=MetricValue(metric, maximize=True),
    )
    branch_nodes.append(node)
    return node


def _failure(agent, role: str, *, stage: str = "debug"):
    root = next(
        node for node in agent.virtual_root.children if node.draft_role == role
    )
    branch_nodes = agent.branch_all_nodes[root.branch_id]
    parent = branch_nodes[-1]
    node = SearchNode(
        code=parent.code + "\n# failed repair",
        parent=parent,
        stage=stage,
        step=len(branch_nodes) + 4,
        branch_id=root.branch_id,
        draft_role=role,
        is_buggy=True,
        is_valid=False,
        metric=MetricValue(0.0, maximize=True),
    )
    branch_nodes.append(node)
    return node


def test_portfolio_roles_keep_independent_seats_with_transfer_behavior():
    assert [canonical_draft_role(role) for role in TRANSFER_PORTFOLIO_DRAFT_ROLES] == [
        "memory_transfer"
    ] * 4
    assert [
        transfer_portfolio_seat_for_role(role)
        for role in TRANSFER_PORTFOLIO_DRAFT_ROLES
    ] == [
        "comprehensive_best",
        "backbone_diversity_1",
        "backbone_diversity_2",
        "descriptor_fusion_diversity",
    ]


def test_portfolio_roles_keep_unique_identity_for_memory_retrieval():
    assert [
        memory_retrieval_draft_role(role)
        for role in TRANSFER_PORTFOLIO_DRAFT_ROLES
    ] == list(TRANSFER_PORTFOLIO_DRAFT_ROLES)
    assert memory_retrieval_draft_role("memory_transfer") == "memory_transfer"
    assert memory_retrieval_draft_role("novel_exploration_a") == "novel_exploration"


def test_draft_agent_uses_slot_preserving_role_for_memory_retrieval():
    source = Path("mlevolve/agents/draft_agent.py").read_text(encoding="utf-8")
    assert "memory_draft_role = memory_retrieval_draft_role(draft_role)" in source
    assert "memory_draft_role = canonical_draft_role(draft_role)" not in source


def test_tournament_requires_baseline_and_one_successor_for_every_seat(tmp_path):
    agent = _agent(tmp_path)
    status = build_transfer_tournament_status(agent)
    assert status["phase"] == "baseline_qualification"
    assert len(status["allowed_branch_ids"]) == 4

    baselines = [0.90, 0.85, 0.80, 0.87]
    for role, metric in zip(TRANSFER_PORTFOLIO_DRAFT_ROLES, baselines):
        root = next(node for node in agent.virtual_root.children if node.draft_role == role)
        root.is_buggy = False
        root.is_valid = True
        root.metric = MetricValue(metric, maximize=True)

    status = build_transfer_tournament_status(agent)
    assert status["phase"] == "round1_equal_improve"
    assert len(status["allowed_branch_ids"]) == 4

    improvements = [0.91, 0.86, 0.88, 0.89]
    for role, metric in zip(TRANSFER_PORTFOLIO_DRAFT_ROLES, improvements):
        _success(agent, role, metric)

    status = build_transfer_tournament_status(agent)
    assert status["phase"] == "top2_equal_improve"
    assert status["finalist_roles"] == [
        TRANSFER_PORTFOLIO_DRAFT_ROLES[0],
        TRANSFER_PORTFOLIO_DRAFT_ROLES[3],
    ]
    assert set(status["allowed_branch_ids"]) == {1, 4}


def test_debug_failures_do_not_consume_baseline_or_improve_entitlement(tmp_path):
    agent = _agent(tmp_path)
    role = TRANSFER_PORTFOLIO_DRAFT_ROLES[0]
    _failure(agent, role)

    status = build_transfer_tournament_status(agent)
    row = next(row for row in status["seats"] if row["draft_role"] == role)
    assert row["attempted_count"] == 2
    assert row["valid_candidate_count"] == 0
    assert row["post_baseline_valid_count"] == 0
    assert status["phase"] == "baseline_qualification"

    _success(agent, role, 0.90, stage="debug")
    status = build_transfer_tournament_status(agent)
    row = next(row for row in status["seats"] if row["draft_role"] == role)
    assert row["baseline_node_id"] is not None
    assert row["valid_candidate_count"] == 1
    assert row["post_baseline_valid_count"] == 0


def test_unavailable_baseline_rolls_resource_to_next_queued_round1_branch(tmp_path):
    agent = _agent(tmp_path)
    missing_role = TRANSFER_PORTFOLIO_DRAFT_ROLES[2]
    for role in TRANSFER_PORTFOLIO_DRAFT_ROLES:
        if role == missing_role:
            continue
        root = next(
            node for node in agent.virtual_root.children if node.draft_role == role
        )
        root.is_buggy = False
        root.is_valid = True
        root.metric = MetricValue(0.80, maximize=True)

    status = build_transfer_tournament_status(agent)

    assert status["phase"] == "baseline_qualification"
    assert status["target_branch_id"] == 3
    assert status["allowed_branch_ids"] == [3]
    assert status["rollover_branch_ids"] == [1, 2, 4]
    assert status["allocation_branch_ids"] == [3, 1, 2, 4]
    assert [row["allocation_kind"] for row in status["allocation_queue"]] == [
        "phase_debt",
        "early_round1_rollover",
        "early_round1_rollover",
        "early_round1_rollover",
    ]

    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[0], 0.81)
    status = build_transfer_tournament_status(agent)
    assert status["allocation_branch_ids"] == [3, 2, 4]
    assert status["work_conserving_donor_branch_ids"] == [1]


def test_completed_round1_seats_form_separate_donor_queue(tmp_path):
    agent = _agent(tmp_path)
    for role in TRANSFER_PORTFOLIO_DRAFT_ROLES:
        root = next(
            node for node in agent.virtual_root.children if node.draft_role == role
        )
        root.is_buggy = False
        root.is_valid = True
        root.metric = MetricValue(0.80, maximize=True)

    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[0], 0.90)
    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[3], 0.88)
    status = build_transfer_tournament_status(agent)

    assert status["phase"] == "round1_equal_improve"
    assert status["allocation_branch_ids"] == [2, 3]
    assert status["work_conserving_donor_branch_ids"] == [1, 4]
    assert status["target_branch_id"] == 2
    assert [row["allocation_kind"] for row in status["allocation_queue"]] == [
        "phase_debt",
        "phase_debt",
    ]


def test_branch_queue_skips_unexpandable_primary_and_selects_next(
    monkeypatch,
    tmp_path,
):
    calls = []
    agent = _agent(tmp_path)

    def select_branch(_agent, _root, branch_id):
        calls.append(branch_id)
        return None if branch_id == 3 else f"node-{branch_id}"

    monkeypatch.setattr(node_selection, "_select_within_branch", select_branch)

    selected, branch_id = node_selection.select_from_branch_ids(
        agent,
        [3, 1, 2, 4],
    )

    assert selected == "node-1"
    assert branch_id == 1
    assert calls == [3, 1]


def test_continuous_primary_failures_recover_locked_rollover_frontier(
    monkeypatch,
    tmp_path,
):
    agent = _agent(tmp_path)
    primary_role = TRANSFER_PORTFOLIO_DRAFT_ROLES[0]
    for _ in range(3):
        _failure(agent, primary_role)

    for role in TRANSFER_PORTFOLIO_DRAFT_ROLES[1:]:
        root = next(
            node for node in agent.virtual_root.children if node.draft_role == role
        )
        root.is_buggy = False
        root.is_valid = True
        root.metric = MetricValue(0.80, maximize=True)
        root.lock = True

    status = build_transfer_tournament_status(agent)
    assert status["phase"] == "baseline_qualification"
    assert status["allocation_branch_ids"] == [1, 2, 3, 4]

    calls = []

    def select_branch(_agent, root, branch_id):
        calls.append((branch_id, bool(root.lock)))
        if branch_id == 1 or root.lock:
            return None
        return f"node-{branch_id}"

    monkeypatch.setattr(node_selection, "_select_within_branch", select_branch)

    selected, branch_id = node_selection.select_from_branch_ids(
        agent,
        status["allocation_branch_ids"],
    )

    assert selected == "node-2"
    assert branch_id == 2
    assert calls == [
        (1, False),
        (2, True),
        (3, True),
        (4, True),
        (2, False),
    ]
    branch2_root = next(
        node for node in agent.virtual_root.children if node.branch_id == 2
    )
    assert branch2_root.lock is True


def test_locked_rollover_recovery_stays_disabled_with_multiple_workers(
    monkeypatch,
    tmp_path,
):
    agent = _agent(tmp_path)
    agent.scfg.parallel_search_num = 2
    for root in agent.virtual_root.children:
        root.lock = True

    observed_lock_states = []

    def select_branch(_agent, root, branch_id):
        observed_lock_states.append((branch_id, bool(root.lock)))
        return None

    monkeypatch.setattr(node_selection, "_select_within_branch", select_branch)

    selected, branch_id = node_selection.select_from_branch_ids(
        agent,
        [1, 2, 3, 4],
    )

    assert selected is None
    assert branch_id is None
    assert observed_lock_states == [
        (1, True),
        (2, True),
        (3, True),
        (4, True),
    ]


def test_exhausted_debt_queue_uses_donor_without_discharge(
    monkeypatch,
    tmp_path,
):
    agent = _agent(tmp_path)
    for role in TRANSFER_PORTFOLIO_DRAFT_ROLES:
        root = next(
            node for node in agent.virtual_root.children if node.draft_role == role
        )
        root.is_buggy = False
        root.is_valid = True
        root.metric = MetricValue(0.80, maximize=True)
    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[0], 0.90)
    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[3], 0.88)
    status_before = build_transfer_tournament_status(agent)

    calls = []

    def select_branch(_agent, root, branch_id):
        calls.append((branch_id, bool(root.lock)))
        return "node-1" if branch_id == 1 and not root.lock else None

    monkeypatch.setattr(node_selection, "_select_within_branch", select_branch)
    selected, branch_id = node_selection.select_from_branch_ids(
        agent,
        status_before["allocation_branch_ids"],
        status_before["work_conserving_donor_branch_ids"],
    )

    assert selected == "node-1"
    assert branch_id == 1
    assert calls == [(2, False), (3, False), (1, False)]
    status_after = build_transfer_tournament_status(agent)
    assert status_after["allocation_branch_ids"] == [2, 3]
    assert status_after["target_branch_id"] == 2


def test_continuous_failures_roll_over_across_donor_queue(monkeypatch, tmp_path):
    agent = _agent(tmp_path)
    for role in TRANSFER_PORTFOLIO_DRAFT_ROLES:
        root = next(
            node for node in agent.virtual_root.children if node.draft_role == role
        )
        root.is_buggy = False
        root.is_valid = True
        root.metric = MetricValue(0.80, maximize=True)
    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[0], 0.90)
    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[3], 0.88)
    status = build_transfer_tournament_status(agent)

    attempts = []

    def select_branch(_agent, _root, branch_id):
        attempts.append(branch_id)
        return "node-4" if branch_id == 4 else None

    monkeypatch.setattr(node_selection, "_select_within_branch", select_branch)
    selected, branch_id = node_selection.select_from_branch_ids(
        agent,
        status["allocation_branch_ids"],
        status["work_conserving_donor_branch_ids"],
    )

    assert selected == "node-4"
    assert branch_id == 4
    assert attempts == [2, 3, 1, 4]
    still_owed = build_transfer_tournament_status(agent)
    assert still_owed["allocation_branch_ids"] == [2, 3]
    assert still_owed["target_branch_id"] == 2


def test_locked_donor_recovery_stays_single_worker_only(monkeypatch, tmp_path):
    agent = _agent(tmp_path)
    roots = {root.branch_id: root for root in agent.virtual_root.children}
    roots[1].lock = True

    observed = []

    def select_branch(_agent, root, branch_id):
        observed.append((branch_id, bool(root.lock)))
        return "node-1" if branch_id == 1 and not root.lock else None

    monkeypatch.setattr(node_selection, "_select_within_branch", select_branch)
    selected, branch_id = node_selection.select_from_branch_ids(
        agent,
        [2, 3],
        [1, 4],
    )
    assert selected == "node-1"
    assert branch_id == 1
    assert roots[1].lock is True

    agent.scfg.parallel_search_num = 2
    observed.clear()
    selected, branch_id = node_selection.select_from_branch_ids(
        agent,
        [2, 3],
        [1],
    )
    assert selected is None
    assert branch_id is None
    assert observed == [(2, False), (3, False), (1, True)]


def test_entitled_tournament_continuation_is_not_rerouted(tmp_path):
    agent = _agent(tmp_path)
    continuation = _success(
        agent,
        TRANSFER_PORTFOLIO_DRAFT_ROLES[1],
        0.90,
        stage="debug",
    )

    assert node_selection.tournament_continuation_branch_id(
        continuation,
        [1, 2, 3, 4],
    ) == 2
    assert node_selection.tournament_continuation_branch_id(
        continuation,
        [1, 3, 4],
    ) is None

    continuation.is_terminal = True
    assert node_selection.tournament_continuation_branch_id(
        continuation,
        [1, 2, 3, 4],
    ) is None


def test_tournament_keeps_top2_then_continues_only_champion(tmp_path):
    agent = _agent(tmp_path)
    first_round = [0.91, 0.86, 0.88, 0.89]
    for role, metric in zip(TRANSFER_PORTFOLIO_DRAFT_ROLES, first_round):
        root = next(node for node in agent.virtual_root.children if node.draft_role == role)
        root.is_buggy = False
        root.is_valid = True
        root.metric = MetricValue(metric - 0.01, maximize=True)
        _success(agent, role, metric)

    status = build_transfer_tournament_status(agent)
    assert status["phase"] == "top2_equal_improve"
    assert status["allocation_branch_ids"] == [1, 4]
    assert status["work_conserving_donor_branch_ids"] == []

    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[0], 0.93)
    status = build_transfer_tournament_status(agent)
    assert status["phase"] == "top2_equal_improve"
    assert status["allocation_branch_ids"] == [4]
    assert status["work_conserving_donor_branch_ids"] == [1]

    _success(agent, TRANSFER_PORTFOLIO_DRAFT_ROLES[3], 0.90)
    status = build_transfer_tournament_status(agent)

    assert status["phase"] == "champion_continuation"
    assert status["champion_role"] == TRANSFER_PORTFOLIO_DRAFT_ROLES[0]
    assert status["allowed_branch_ids"] == [1]
    assert status["target_branch_id"] == 1


def test_tournament_config_declares_four_real_draft_slots():
    from config import _load_config_tree

    path = Path(
        "experiments/end2end_memory_systems_20260804/"
        "systems_v167_plant_seedlings_intent_feedback_a100_6h_sol_low/"
        "intent_feedback_tournament.yaml"
    )
    cfg = _load_config_tree(path)
    policy = cfg.agent.draft_role_policy

    assert cfg.agent.initial_drafts == 4
    assert cfg.agent.search.num_drafts == 4
    assert list(policy.roles) == list(TRANSFER_PORTFOLIO_DRAFT_ROLES)
    assert policy.transfer_portfolio_tournament_enabled is True
    assert policy.ensure_valid_candidate_per_role is True
    assert policy.transfer_portfolio_survivor_count == 2
