"""Run-loop controls for a role-focused protocol repair."""

from dataclasses import dataclass
from typing import Iterable, Literal, Optional


ACTIVE_PROTOCOL_STATES = frozenset({"pending", "stage_in_progress", "final_pending"})


def draft_execution_lane(node: object) -> Literal["execute", "repair"]:
    """Return the only legal Phase-2 lane for a generated Draft.

    Pre-execution audit failures are journaled with ``pending_execution=False``
    and ``audit_repair_required=True``.  Treating them as deferred executable
    Drafts silently skips their mandatory repair transaction.
    """
    if bool(getattr(node, "pending_execution", False)):
        return "execute"
    if bool(getattr(node, "audit_repair_required", False)):
        return "repair"
    node_id = getattr(node, "id", "unknown")
    raise RuntimeError(
        f"Generated Draft {node_id} is neither executable nor repair-queued"
    )


@dataclass(frozen=True)
class FocusedProtocolStatus:
    seen: bool
    state: str
    node: Optional[object] = None
    repair_kind: str = ""

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_PROTOCOL_STATES

    @property
    def completed(self) -> bool:
        return self.state == "completed"


def focused_protocol_status(nodes: Iterable[object], draft_role: str) -> FocusedProtocolStatus:
    """Return the latest mandatory-repair state belonging to one Draft role."""
    candidates = []
    for node in nodes:
        if getattr(node, "draft_role", None) != draft_role:
            continue
        transaction = getattr(node, "protocol_repair", None) or {}
        audit = getattr(node, "leakage_audit", None) or {}
        ordinary_repair = bool(
            getattr(node, "audit_repair_required", False)
            or int(getattr(node, "leakage_repair_attempt", 0) or 0) > 0
            or audit.get("repair_queue_status")
            in {"queued", "in_flight", "queued_after_error", "expanded", "exhausted"}
        )
        if not transaction and not ordinary_repair:
            continue
        candidates.append(node)

    if not candidates:
        return FocusedProtocolStatus(seen=False, state="not_started")

    latest = max(candidates, key=lambda item: float(getattr(item, "ctime", 0.0) or 0.0))
    transaction = getattr(latest, "protocol_repair", None) or {}
    if transaction:
        state = str(transaction.get("state") or "unknown")
        repair_kind = "protocol"
    else:
        audit = getattr(latest, "leakage_audit", None) or {}
        if bool(getattr(latest, "is_terminal", False)) or audit.get(
            "repair_queue_status"
        ) == "exhausted":
            state = "exhausted"
        elif bool(getattr(latest, "audit_repair_required", False)):
            state = "pending"
        elif audit.get("status") == "clean":
            state = "completed"
        else:
            state = "unknown"
        repair_kind = "leakage"
    return FocusedProtocolStatus(
        seen=True,
        state=state,
        node=latest,
        repair_kind=repair_kind,
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
        repair_label = (
            "leakage repair"
            if status.repair_kind == "leakage"
            else "protocol transaction"
        )
        return f"the focused {repair_label} ended with state={status.state}"

    node = status.node
    audit = getattr(node, "leakage_audit", None) or {}
    metric = getattr(getattr(node, "metric", None), "value", None)
    replay_status = str(getattr(node, "replay_status", "") or "")
    if status.repair_kind == "protocol":
        if replay_status != "staged_protocol_repair_executed_clean":
            return f"the completed transaction has replay_status={replay_status or 'missing'}"
    elif replay_status not in {
        "mandatory_audit_repair_executed_clean",
        "mandatory_audit_repair_clean_pending_execution",
    }:
        return f"the completed leakage repair has replay_status={replay_status or 'missing'}"
    if audit.get("status") != "clean":
        return f"the completed transaction has leakage status={audit.get('status', 'missing')}"
    if audit.get("rank_eligible") is not True:
        return "the completed transaction is not rank eligible"
    if metric is None:
        return "the completed transaction has no metric"
    return None


def focused_outcome_context(
    draft_role: str,
    status: FocusedProtocolStatus,
    *,
    require_protocol_repair: bool = True,
) -> tuple[bool, str]:
    """Translate focused-role state into durable run-outcome fields.

    A failed focused role must be represented in ``RUN_OUTCOME.json`` instead
    of raising before the immutable outcome is published.
    """
    if not draft_role:
        return False, ""
    if not require_protocol_repair:
        # The role passed pre-execution audit and completed its ordinary
        # execution smoke.  No protocol-repair transaction is expected.
        return False, ""
    error = focused_protocol_success_error(status)
    if error is None:
        return True, ""
    return False, f"focused_role_incomplete: {error}"
