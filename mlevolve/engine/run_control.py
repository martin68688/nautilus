"""Run-loop controls for a role-focused protocol repair."""

from dataclasses import dataclass
from typing import Iterable, Optional


ACTIVE_PROTOCOL_STATES = frozenset({"pending", "stage_in_progress", "final_pending"})


@dataclass(frozen=True)
class FocusedProtocolStatus:
    seen: bool
    state: str
    node: Optional[object] = None

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_PROTOCOL_STATES

    @property
    def completed(self) -> bool:
        return self.state == "completed"


def focused_protocol_status(nodes: Iterable[object], draft_role: str) -> FocusedProtocolStatus:
    """Return the latest protocol state belonging to one Draft role."""
    candidates = []
    for node in nodes:
        if getattr(node, "draft_role", None) != draft_role:
            continue
        transaction = getattr(node, "protocol_repair", None) or {}
        if not transaction:
            continue
        candidates.append(node)

    if not candidates:
        return FocusedProtocolStatus(seen=False, state="not_started")

    latest = max(candidates, key=lambda item: float(getattr(item, "ctime", 0.0) or 0.0))
    transaction = getattr(latest, "protocol_repair", None) or {}
    return FocusedProtocolStatus(
        seen=True,
        state=str(transaction.get("state") or "unknown"),
        node=latest,
    )


def should_continue_focused_search(
    *,
    completed_steps: int,
    total_steps: int,
    status: FocusedProtocolStatus,
    focus_in_flight: bool,
) -> bool:
    """Keep a focused replay alive beyond the ordinary shared search budget."""
    if status.completed:
        return False
    if focus_in_flight or status.active:
        return True
    if status.seen:
        return False
    return completed_steps < total_steps


def focused_protocol_success_error(status: FocusedProtocolStatus) -> str | None:
    """Explain why a focused replay cannot be reported as a successful Job."""
    if not status.seen:
        return "the focused role never created a protocol-repair transaction"
    if not status.completed:
        return f"the focused protocol transaction ended with state={status.state}"

    node = status.node
    audit = getattr(node, "leakage_audit", None) or {}
    metric = getattr(getattr(node, "metric", None), "value", None)
    replay_status = str(getattr(node, "replay_status", "") or "")
    if replay_status != "staged_protocol_repair_executed_clean":
        return f"the completed transaction has replay_status={replay_status or 'missing'}"
    if audit.get("status") != "clean":
        return f"the completed transaction has leakage status={audit.get('status', 'missing')}"
    if audit.get("rank_eligible") is not True:
        return "the completed transaction is not rank eligible"
    if metric is None:
        return "the completed transaction has no metric"
    return None
