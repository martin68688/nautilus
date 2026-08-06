"""Durable MLEvolve search-step checkpoint loading and workspace restoration."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import copy
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from engine.search_node import Journal, SearchNode
from utils import serialize


RESUME_RECEIPT_SCHEMA = "mlevolve_search_resume_receipt_v1"
logger = logging.getLogger("MLEvolve")
_WORKSPACE_DIRS = (
    "submission",
    "best_solution",
    "best_submission",
    "top_solution",
)
_CANDIDATE_ARCHIVE_DIR = "candidate_checkpoints"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(raw: str, label: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path


def _directory(raw: str, label: str) -> Path:
    path = Path(raw).expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a directory: {path}")
    return path


def _expected_hash(value: str, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{label} must be one SHA-256 digest")
    return digest


@dataclass(frozen=True)
class SearchResumeCheckpoint:
    journal: Journal
    journal_path: Path
    journal_sha256: str
    outcome_path: Path
    outcome_sha256: str
    workspace_root: Path
    source_attempt_root: Path
    source_attempt: int
    completed_steps: int
    total_steps: int
    prior_agent_wall_seconds: float
    active_candidates: tuple[SearchNode, ...]
    unrestorable_active_candidate_ids: tuple[str, ...]

    def receipt(self, restored_workspace_dirs: list[str]) -> dict:
        return {
            "schema": RESUME_RECEIPT_SCHEMA,
            "source_attempt_root": str(self.source_attempt_root),
            "source_attempt": self.source_attempt,
            "journal_path": str(self.journal_path),
            "journal_sha256": self.journal_sha256,
            "outcome_path": str(self.outcome_path),
            "outcome_sha256": self.outcome_sha256,
            "source_workspace_root": str(self.workspace_root),
            "completed_steps": self.completed_steps,
            "total_steps": self.total_steps,
            "remaining_steps": self.total_steps - self.completed_steps,
            "prior_agent_wall_seconds": self.prior_agent_wall_seconds,
            "restored_node_count": len(self.journal),
            "restored_workspace_dirs": list(restored_workspace_dirs),
            "restored_active_candidate_ids": [
                node.id for node in self.active_candidates
            ],
            "unrestorable_active_candidate_ids": list(
                self.unrestorable_active_candidate_ids
            ),
        }


def archive_candidate_source(agent, node: SearchNode) -> Path:
    """Persist the final pre-execution candidate source and routing metadata."""

    archive_root = Path(agent.cfg.log_dir) / _CANDIDATE_ARCHIVE_DIR
    archive_root.mkdir(parents=True, exist_ok=True)
    code_path = archive_root / f"{node.id}.py"
    record_path = archive_root / f"{node.id}.json"
    code = str(node.code or "")
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()

    archived = copy.copy(node)
    parent_id = node.parent.id if node.parent is not None else None
    local_best_id = (
        node.local_best_node.id if node.local_best_node is not None else None
    )
    archived.code = ""
    archived.parent = None
    archived.children = set()
    archived.local_best_node = None
    archived.child_count_lock = None
    payload = {
        "schema": "mlevolve_candidate_checkpoint_v1",
        "candidate_id": node.id,
        "parent_id": parent_id,
        "local_best_node_id": local_best_id,
        "code_path": code_path.name,
        "code_sha256": code_sha256,
        "node": archived.to_dict(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ) + "\n"

    if code_path.exists() or record_path.exists():
        if (
            not code_path.is_file()
            or code_path.is_symlink()
            or not record_path.is_file()
            or record_path.is_symlink()
            or sha256_file(code_path) != code_sha256
        ):
            raise ValueError(f"Candidate checkpoint collision for {node.id}")
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(f"Candidate checkpoint metadata collision for {node.id}")
        return record_path

    with code_path.open("x", encoding="utf-8") as handle:
        handle.write(code)
    try:
        with record_path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
    except Exception:
        code_path.unlink(missing_ok=True)
        raise
    return record_path


def _load_active_candidates(
    *,
    outcome: Mapping[str, object],
    journal: Journal,
    log_root: Path,
) -> tuple[tuple[SearchNode, ...], tuple[str, ...]]:
    active_ids = tuple(
        dict.fromkeys(
            str(value)
            for value in (outcome.get("active_candidate_ids") or [])
            if str(value)
        )
    )
    if not active_ids:
        return (), ()
    id2node = {node.id: node for node in journal.nodes}
    restored: list[SearchNode] = []
    missing: list[str] = []
    archive_root = log_root / _CANDIDATE_ARCHIVE_DIR
    for candidate_id in active_ids:
        record_path = archive_root / f"{candidate_id}.json"
        if not record_path.is_file() or record_path.is_symlink():
            missing.append(candidate_id)
            continue
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "mlevolve_candidate_checkpoint_v1"
            or payload.get("candidate_id") != candidate_id
        ):
            raise ValueError(f"Invalid active candidate checkpoint: {candidate_id}")
        code_path = archive_root / str(payload.get("code_path") or "")
        if (
            code_path.parent != archive_root
            or not code_path.is_file()
            or code_path.is_symlink()
        ):
            raise ValueError(f"Missing active candidate source: {candidate_id}")
        code = code_path.read_text(encoding="utf-8")
        if hashlib.sha256(code.encode("utf-8")).hexdigest() != str(
            payload.get("code_sha256") or ""
        ):
            raise ValueError(f"Active candidate source hash mismatch: {candidate_id}")
        parent_id = str(payload.get("parent_id") or "")
        if parent_id not in id2node:
            raise ValueError(
                f"Active candidate parent is not in resumed Journal: {candidate_id}"
            )
        node = SearchNode.from_dict(dict(payload.get("node") or {}))
        if node.id != candidate_id:
            raise ValueError(f"Active candidate node ID mismatch: {candidate_id}")
        # Active-candidate checkpoints are decoded directly rather than via
        # utils.serialize.loads_json, so rebuild their process-local lock here.
        node.child_count_lock = threading.Lock()
        node.code = code
        node.parent = id2node[parent_id]
        node.__post_init__()
        local_best_id = str(payload.get("local_best_node_id") or "")
        if local_best_id:
            if local_best_id not in id2node:
                raise ValueError(
                    f"Active candidate local best is not in resumed Journal: {candidate_id}"
                )
            node.local_best_node = id2node[local_best_id]
        node.pending_execution = True
        restored.append(node)
    return tuple(restored), tuple(missing)


def load_search_resume_checkpoint(
    *,
    total_steps: int,
    environ: Mapping[str, str] | None = None,
) -> SearchResumeCheckpoint | None:
    """Load one runner-bound checkpoint or return ``None`` for a fresh run."""

    env = os.environ if environ is None else environ
    raw_journal = str(env.get("MLEVOLVE_RESUME_JOURNAL_PATH") or "").strip()
    if not raw_journal:
        return None
    journal_path = _regular_file(raw_journal, "resume journal")
    outcome_path = _regular_file(
        str(env.get("MLEVOLVE_RESUME_OUTCOME_PATH") or ""),
        "resume outcome",
    )
    workspace_root = _directory(
        str(env.get("MLEVOLVE_RESUME_WORKSPACE_ROOT") or ""),
        "resume workspace",
    )
    source_attempt_root = _directory(
        str(env.get("MLEVOLVE_RESUME_SOURCE_ATTEMPT_ROOT") or ""),
        "resume source attempt",
    )
    journal_sha = _expected_hash(
        str(env.get("MLEVOLVE_RESUME_JOURNAL_SHA256") or ""),
        "resume journal hash",
    )
    outcome_sha = _expected_hash(
        str(env.get("MLEVOLVE_RESUME_OUTCOME_SHA256") or ""),
        "resume outcome hash",
    )
    if sha256_file(journal_path) != journal_sha:
        raise ValueError("Resume Journal SHA-256 mismatch")
    if sha256_file(outcome_path) != outcome_sha:
        raise ValueError("Resume RUN_OUTCOME SHA-256 mismatch")

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    if not isinstance(outcome, dict):
        raise ValueError("Resume RUN_OUTCOME must be one JSON object")
    completed_steps = int(outcome.get("completed_steps") or 0)
    checkpoint_total = int(outcome.get("total_steps") or 0)
    if outcome.get("status") != "partial" or outcome.get("interrupted") is not True:
        raise ValueError("Resume source is not an interrupted partial run")
    if checkpoint_total != int(total_steps):
        raise ValueError(
            f"Resume total-step mismatch: checkpoint={checkpoint_total}, "
            f"configured={total_steps}"
        )
    if not 0 < completed_steps < checkpoint_total:
        raise ValueError("Resume checkpoint has no remaining search work")

    try:
        prior_agent_wall_seconds = float(
            str(env.get("MLEVOLVE_RESUME_PRIOR_WALL_SECONDS") or "0")
        )
    except ValueError as error:
        raise ValueError(
            "resume prior wall seconds must be a finite non-negative number"
        ) from error
    if (
        not prior_agent_wall_seconds >= 0.0
        or prior_agent_wall_seconds == float("inf")
    ):
        raise ValueError(
            "resume prior wall seconds must be a finite non-negative number"
        )

    journal = serialize.load_json(journal_path, Journal)
    if len(journal) - 1 != completed_steps:
        raise ValueError(
            "Resume Journal node count does not match RUN_OUTCOME: "
            f"journal={len(journal) - 1}, outcome={completed_steps}"
        )
    source_attempt = int(source_attempt_root.name.removeprefix("attempt-"))
    active_candidates, unrestorable_active_ids = _load_active_candidates(
        outcome=outcome,
        journal=journal,
        log_root=journal_path.parent,
    )
    return SearchResumeCheckpoint(
        journal=journal,
        journal_path=journal_path,
        journal_sha256=journal_sha,
        outcome_path=outcome_path,
        outcome_sha256=outcome_sha,
        workspace_root=workspace_root,
        source_attempt_root=source_attempt_root,
        source_attempt=source_attempt,
        completed_steps=completed_steps,
        total_steps=checkpoint_total,
        prior_agent_wall_seconds=prior_agent_wall_seconds,
        active_candidates=active_candidates,
        unrestorable_active_candidate_ids=unrestorable_active_ids,
    )


def restore_search_workspace(
    checkpoint: SearchResumeCheckpoint,
    destination: Path,
) -> list[str]:
    """Restore only durable candidate/submission artifacts into a fresh workspace."""

    destination = destination.resolve(strict=True)
    restored: list[str] = []
    for name in _WORKSPACE_DIRS:
        source = checkpoint.workspace_root / name
        if not source.is_dir() or source.is_symlink():
            continue
        target = destination / name
        shutil.copytree(source, target, dirs_exist_ok=True, symlinks=False)
        restored.append(name)
    return restored


def write_search_resume_receipt(
    log_dir: Path,
    checkpoint: SearchResumeCheckpoint,
    restored_workspace_dirs: list[str],
) -> Path:
    path = log_dir / "SEARCH_RESUME_RECEIPT.json"
    payload = checkpoint.receipt(restored_workspace_dirs)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    return path


def restore_agent_search_state(agent) -> None:
    """Rebuild process-local indexes on an AgentSearch-like object."""

    from agents.leakage_audit import rank_eligible
    from engine import solution_manager

    journal = agent.journal
    if len(journal) <= 1:
        raise ValueError("A resume checkpoint has no completed search node")
    seen_ids: set[str] = set()
    for position, node in enumerate(journal.nodes):
        if node.id in seen_ids:
            raise ValueError(f"Duplicate node ID in resume Journal: {node.id}")
        seen_ids.add(node.id)
        if node.step != position:
            raise ValueError(
                f"Resume Journal step mismatch for {node.id}: "
                f"stored={node.step}, expected={position}"
            )
        if node is not agent.virtual_root and node.parent is None:
            raise ValueError(f"Resume Journal node has no parent: {node.id}")
        # Be defensive even when the Journal came from an older loader: locks
        # are runtime primitives and must never be trusted from JSON.
        node.child_count_lock = threading.Lock()
        node.expected_child_count = len(node.children)
        node.lock = False

    agent.current_step = len(journal)
    agent.current_node = journal.nodes[-1]
    agent.current_node_list = [agent.current_node]
    agent.all_root = False

    agent.branch_all_nodes = {}
    agent.branch_successful_nodes = {}
    agent.branch_node_count = {}
    for node in journal.nodes[1:]:
        if node.branch_id is None:
            continue
        branch_id = int(node.branch_id)
        agent.branch_all_nodes.setdefault(branch_id, []).append(node)
        agent.branch_successful_nodes.setdefault(branch_id, [])
        agent.branch_node_count[branch_id] = (
            agent.branch_node_count.get(branch_id, 0) + 1
        )
        if (
            node.is_buggy is False
            and node.is_valid is not False
            and node.metric is not None
            and node.metric.value is not None
            and rank_eligible(agent, node)
        ):
            agent.branch_successful_nodes[branch_id].append(node)
    agent.next_branch_id = max(agent.branch_all_nodes, default=0) + 1

    declared_roles = list(
        getattr(getattr(agent.acfg, "draft_role_policy", None), "roles", [])
        or []
    )
    initial_root_nodes = [
        node
        for node in journal.nodes[1:]
        if node.parent is agent.virtual_root
        and node.stage in {"draft", "fusion_draft"}
        and node.draft_role != "replacement_draft"
    ]
    agent._draft_generation_count = min(
        len(initial_root_nodes),
        len(declared_roles) if declared_roles else int(agent.acfg.initial_drafts),
    )
    agent._replacement_draft_count = sum(
        1
        for node in journal.nodes[1:]
        if node.parent is agent.virtual_root
        and node.draft_role == "replacement_draft"
    )
    agent._replacement_draft_inflight = False

    agent.top_candidates = []
    for node in journal.nodes[1:]:
        solution_manager.update_top_candidates(agent, node)
    agent.best_node = agent.top_candidates[0] if agent.top_candidates else None
    agent.best_metric = (
        agent.best_node.metric.value
        if agent.best_node is not None and agent.best_node.metric is not None
        else None
    )

    for node in journal.nodes[1:]:
        agent._enqueue_mandatory_repair(node)
    logger.warning(
        "[resume] restored completed=%s branches=%s top_candidates=%s "
        "best=%s draft_slots=%s",
        len(journal) - 1,
        len(agent.branch_all_nodes),
        len(agent.top_candidates),
        agent.best_node.id if agent.best_node is not None else None,
        agent._draft_generation_count,
    )


def attach_resumed_active_candidates(agent, candidates: tuple[SearchNode, ...]) -> None:
    """Account for archived in-flight nodes before scheduling their re-execution."""

    if not candidates:
        return
    roles = list(
        getattr(getattr(agent.acfg, "draft_role_policy", None), "roles", [])
        or []
    )
    active_root_roles = {
        str(node.draft_role)
        for node in candidates
        if node.parent is agent.virtual_root
        and node.stage in {"draft", "fusion_draft"}
        and node.draft_role not in {None, "replacement_draft"}
    }
    if roles and active_root_roles:
        indexes = [
            roles.index(role) for role in active_root_roles if role in roles
        ]
        if indexes:
            agent._draft_generation_count = max(
                int(agent._draft_generation_count), max(indexes) + 1
            )
    for node in candidates:
        if node.parent is not None:
            node.parent.expected_child_count = len(node.parent.children)
    logger.warning(
        "[resume] queued archived active candidates for boundary replay: %s",
        [node.id for node in candidates],
    )


__all__ = [
    "RESUME_RECEIPT_SCHEMA",
    "SearchResumeCheckpoint",
    "archive_candidate_source",
    "attach_resumed_active_candidates",
    "load_search_resume_checkpoint",
    "restore_agent_search_state",
    "restore_search_workspace",
    "sha256_file",
    "write_search_resume_receipt",
]
