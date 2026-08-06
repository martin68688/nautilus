from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine import solution_manager
from engine.search_node import Journal, SearchNode
from engine.search_resume import (
    archive_candidate_source,
    attach_resumed_active_candidates,
    load_search_resume_checkpoint,
    restore_agent_search_state,
    restore_search_workspace,
    sha256_file,
    write_search_resume_receipt,
)
from utils.metric import MetricValue, WorstMetricValue
from utils.serialize import dump_json, load_json


def _journal() -> Journal:
    journal = Journal()
    root = SearchNode(
        code="",
        plan="root",
        stage="root",
        metric=WorstMetricValue(),
    )
    journal.append(root)
    first = SearchNode(
        code="print('first')",
        plan="first",
        stage="draft",
        parent=root,
        branch_id=1,
        draft_role="coldstart_baseline",
        metric=MetricValue(0.4, maximize=False),
        is_buggy=False,
        is_valid=True,
        leakage_audit={"status": "clean", "rank_eligible": True},
    )
    journal.append(first)
    second = SearchNode(
        code="print('second')",
        plan="second",
        stage="improve",
        parent=first,
        branch_id=1,
        draft_role="coldstart_baseline",
        local_best_node=first,
        metric=MetricValue(0.2, maximize=False),
        is_buggy=False,
        is_valid=True,
        leakage_audit={"status": "clean", "rank_eligible": True},
    )
    journal.append(second)
    return journal


def _checkpoint(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    attempt_root = tmp_path / "attempt-000"
    log_root = attempt_root / "agent" / "logs" / "runtime"
    workspace = attempt_root / "agent" / "workspace" / "runtime"
    log_root.mkdir(parents=True)
    (workspace / "submission").mkdir(parents=True)
    (workspace / "working").mkdir()
    (workspace / "submission" / "submission-node.csv").write_text(
        "id,pred\n1,0.5\n",
        encoding="utf-8",
    )
    journal_path = log_root / "journal.json"
    dump_json(_journal(), journal_path)
    outcome_path = log_root / "RUN_OUTCOME.json"
    outcome_path.write_text(
        json.dumps(
            {
                "schema": "mlevolve_run_outcome_v1",
                "status": "partial",
                "interrupted": True,
                "completed_steps": 2,
                "total_steps": 5,
            }
        ),
        encoding="utf-8",
    )
    env = {
        "MLEVOLVE_RESUME_SOURCE_ATTEMPT_ROOT": str(attempt_root),
        "MLEVOLVE_RESUME_JOURNAL_PATH": str(journal_path),
        "MLEVOLVE_RESUME_JOURNAL_SHA256": sha256_file(journal_path),
        "MLEVOLVE_RESUME_OUTCOME_PATH": str(outcome_path),
        "MLEVOLVE_RESUME_OUTCOME_SHA256": sha256_file(outcome_path),
        "MLEVOLVE_RESUME_WORKSPACE_ROOT": str(workspace),
        "MLEVOLVE_RESUME_PRIOR_WALL_SECONDS": "123.5",
    }
    return attempt_root, env


def test_journal_roundtrip_restores_parents_children_and_local_best(tmp_path) -> None:
    path = tmp_path / "journal.json"
    dump_json(_journal(), path)
    restored = load_json(path, Journal)

    assert len(restored) == 3
    assert restored[1].parent is restored[0]
    assert restored[2].parent is restored[1]
    assert restored[2] in restored[1].children
    assert restored[2].local_best_node is restored[1]
    assert restored[0].child_count_lock is not None
    assert restored[0].child_count_lock is not restored[1].child_count_lock
    before = restored[0].expected_child_count
    restored[0].add_expected_child_count()
    restored[0].sub_expected_child_count()
    assert restored[0].expected_child_count == before


def test_load_and_restore_search_resume_checkpoint(tmp_path) -> None:
    attempt_root, env = _checkpoint(tmp_path)
    checkpoint = load_search_resume_checkpoint(total_steps=5, environ=env)

    assert checkpoint is not None
    assert checkpoint.source_attempt_root == attempt_root.resolve()
    assert checkpoint.completed_steps == 2
    assert checkpoint.prior_agent_wall_seconds == 123.5
    assert len(checkpoint.journal) == 3

    destination = tmp_path / "new-workspace"
    destination.mkdir()
    restored = restore_search_workspace(checkpoint, destination)
    assert restored == ["submission"]
    assert (destination / "submission" / "submission-node.csv").is_file()

    log_dir = tmp_path / "new-logs"
    log_dir.mkdir()
    receipt = write_search_resume_receipt(log_dir, checkpoint, restored)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["completed_steps"] == 2
    assert payload["remaining_steps"] == 3
    assert payload["prior_agent_wall_seconds"] == 123.5
    assert payload["restored_node_count"] == 3


def test_resume_rejects_tampered_journal(tmp_path) -> None:
    _attempt_root, env = _checkpoint(tmp_path)
    Path(env["MLEVOLVE_RESUME_JOURNAL_PATH"]).write_text(
        "{}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Journal SHA-256 mismatch"):
        load_search_resume_checkpoint(total_steps=5, environ=env)


def test_active_candidate_source_is_archived_and_restored(tmp_path) -> None:
    attempt_root, env = _checkpoint(tmp_path)
    log_root = Path(env["MLEVOLVE_RESUME_JOURNAL_PATH"]).parent
    journal = load_json(Path(env["MLEVOLVE_RESUME_JOURNAL_PATH"]), Journal)
    active = SearchNode(
        code="print('active candidate')",
        plan="continue from parent",
        stage="improve",
        parent=journal[2],
        branch_id=1,
        draft_role="coldstart_baseline",
        local_best_node=journal[2],
    )
    agent = SimpleNamespace(cfg=SimpleNamespace(log_dir=log_root))
    archive_candidate_source(agent, active)
    outcome_path = Path(env["MLEVOLVE_RESUME_OUTCOME_PATH"])
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["active_candidate_ids"] = [active.id]
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    env["MLEVOLVE_RESUME_OUTCOME_SHA256"] = sha256_file(outcome_path)

    checkpoint = load_search_resume_checkpoint(total_steps=5, environ=env)

    assert checkpoint is not None
    assert checkpoint.unrestorable_active_candidate_ids == ()
    assert len(checkpoint.active_candidates) == 1
    restored = checkpoint.active_candidates[0]
    assert restored.id == active.id
    assert restored.code == active.code
    assert restored.parent is checkpoint.journal[2]
    assert restored.local_best_node is checkpoint.journal[2]
    assert restored.pending_execution is True
    assert restored.child_count_lock is not None


def test_missing_old_active_candidate_is_reported_without_blocking_resume(
    tmp_path,
) -> None:
    _attempt_root, env = _checkpoint(tmp_path)
    outcome_path = Path(env["MLEVOLVE_RESUME_OUTCOME_PATH"])
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["active_candidate_ids"] = ["legacy-active-without-source"]
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    env["MLEVOLVE_RESUME_OUTCOME_SHA256"] = sha256_file(outcome_path)

    checkpoint = load_search_resume_checkpoint(total_steps=5, environ=env)

    assert checkpoint is not None
    assert checkpoint.active_candidates == ()
    assert checkpoint.unrestorable_active_candidate_ids == (
        "legacy-active-without-source",
    )


def test_agent_rebuilds_branch_and_best_state_from_completed_nodes(
    monkeypatch,
) -> None:
    journal = _journal()
    for node in journal.nodes:
        node.child_count_lock = None
    root = journal[0]
    root.expected_child_count = 99
    root.lock = True
    agent = SimpleNamespace()
    agent.journal = journal
    agent.virtual_root = root
    agent.acfg = SimpleNamespace(
        initial_drafts=3,
        draft_role_policy=SimpleNamespace(
            roles=(
                "coldstart_baseline",
                "memory_reproduction",
                "novel_exploration",
            )
        ),
    )
    agent.metric_maximize = False
    agent.top_k = 6
    agent._enqueue_mandatory_repair = lambda _node: None
    monkeypatch.setattr(solution_manager, "rank_eligible", lambda _a, _n: True)
    monkeypatch.setattr(
        "agents.leakage_audit.rank_eligible",
        lambda _a, _n: True,
    )

    restore_agent_search_state(agent)

    assert agent.current_step == 3
    assert agent.branch_all_nodes == {1: [journal[1], journal[2]]}
    assert agent.branch_successful_nodes == {1: [journal[1], journal[2]]}
    assert agent.next_branch_id == 2
    assert agent.best_node is journal[2]
    assert agent.best_metric == 0.2
    assert agent._draft_generation_count == 1
    assert root.expected_child_count == 1
    assert root.lock is False
    assert all(node.child_count_lock is not None for node in journal.nodes)
    root.add_expected_child_count()
    root.sub_expected_child_count()
    assert root.expected_child_count == 1

    active = SearchNode(
        code="print('pending')",
        plan="pending draft",
        stage="draft",
        parent=root,
        branch_id=2,
        draft_role="memory_reproduction",
    )
    active.pending_execution = True
    attach_resumed_active_candidates(agent, (active,))
    assert agent._draft_generation_count == 2
    assert root.expected_child_count == len(root.children)
