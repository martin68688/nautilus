"""Explicit run-completion semantics for batch launchers."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


RUN_OUTCOME_SCHEMA = "mlevolve_run_outcome_v1"


class PartialRunError(RuntimeError):
    """The search ended before its declared step target but preserved a solution."""


class FailedRunError(RuntimeError):
    """The search ended without completing its target or certifying a solution."""


def _hash_payload(value: Mapping[str, Any]) -> str:
    content = json.dumps(
        {key: item for key, item in value.items() if key != "outcome_hash"},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def classify_run_outcome(
    *,
    completed_steps: int,
    total_steps: int,
    search_exhausted: bool,
    has_certified_solution: bool,
    focused_scope_complete: bool = False,
    termination_reason: str = "",
    active_candidate_ids: list[str] | None = None,
    journal_checkpoint_ref: str = "",
) -> dict[str, Any]:
    if total_steps < 0 or completed_steps < 0:
        raise ValueError("Run step counts cannot be negative")
    if termination_reason:
        status = "partial"
        reason = str(termination_reason)
    elif focused_scope_complete:
        status = "complete"
        reason = "focused_scope_completed"
    elif completed_steps >= total_steps:
        status = "complete"
        reason = "step_target_reached"
    elif search_exhausted and has_certified_solution:
        status = "partial"
        reason = "search_space_exhausted_with_certified_solution"
    elif search_exhausted:
        status = "failed"
        reason = "search_space_exhausted_without_certified_solution"
    else:
        status = "failed"
        reason = "run_stopped_before_step_target"
    payload: dict[str, Any] = {
        "schema": RUN_OUTCOME_SCHEMA,
        "status": status,
        "reason": reason,
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "completion_ratio": (
            1.0 if total_steps == 0 else min(1.0, completed_steps / total_steps)
        ),
        "search_exhausted": bool(search_exhausted),
        "certified_solution_available": bool(has_certified_solution),
        "kubernetes_success_eligible": status == "complete",
        "interrupted": bool(termination_reason),
        "active_candidate_ids": sorted(
            set(str(value) for value in active_candidate_ids or [])
        ),
        "journal_checkpoint_ref": str(journal_checkpoint_ref or ""),
        "outcome_hash": "",
    }
    payload["outcome_hash"] = _hash_payload(payload)
    return payload


def write_run_outcome(log_dir: str | Path, outcome: Mapping[str, Any]) -> Path:
    payload = dict(outcome)
    if payload.get("schema") != RUN_OUTCOME_SCHEMA:
        raise ValueError("Unknown run outcome schema")
    if payload.get("outcome_hash") != _hash_payload(payload):
        raise ValueError("Run outcome hash mismatch")
    directory = Path(log_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "RUN_OUTCOME.json"
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_suffix(
        path.suffix + f".{os.getpid()}.{os.urandom(4).hex()}.tmp"
    )
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Hard-link publication is atomic and refuses to replace an existing
        # immutable outcome. It also avoids exposing a half-written file if a
        # Pod is killed during finalization.
        os.link(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError("Refusing to replace immutable run outcome")
    finally:
        temporary.unlink(missing_ok=True)
    return path


__all__ = [
    "FailedRunError",
    "PartialRunError",
    "RUN_OUTCOME_SCHEMA",
    "classify_run_outcome",
    "write_run_outcome",
]
