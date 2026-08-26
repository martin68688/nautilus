"""Host-owned scheduling state for the four-seat Transfer tournament.

The portfolio builder decides which four hypotheses exist.  This module does
not alter those hypotheses or their scores; it only prevents the ordinary
single-tree scheduler from collapsing the portfolio before every seat has a
valid baseline and a score-bearing improvement.  After that bounded round it
keeps the best two for one further valid successor each, then spends the
remaining run budget on the resulting champion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.draft_roles import (
    TRANSFER_PORTFOLIO_DRAFT_ROLE_TO_SEAT,
    TRANSFER_PORTFOLIO_DRAFT_ROLES,
)


TOURNAMENT_SCHEMA = "mlevolve_transfer_portfolio_tournament_v1"


def _valid_metric_node(node: Any) -> bool:
    metric = getattr(node, "metric", None)
    return bool(
        getattr(node, "pending_execution", False) is not True
        and getattr(node, "is_buggy", None) is False
        and getattr(node, "is_valid", None) is not False
        and metric is not None
        and getattr(metric, "value", None) is not None
    )


def _rank_rows(rows: list[dict[str, Any]], *, maximize: bool) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        score = row.get("best_metric")
        missing = score is None
        numeric = float(score) if score is not None else 0.0
        return (
            missing,
            -numeric if maximize and not missing else numeric,
            int(row["seat_order"]),
        )

    ranked = sorted(rows, key=key)
    return [dict(row, rank=index + 1) for index, row in enumerate(ranked)]


def _root_by_role(agent: Any) -> dict[str, Any]:
    roots = sorted(
        list(getattr(getattr(agent, "virtual_root", None), "children", set()) or []),
        key=lambda node: (
            float(getattr(node, "ctime", 0.0) or 0.0),
            str(getattr(node, "id", "")),
        ),
    )
    return {
        str(getattr(node, "draft_role", "") or ""): node
        for node in roots
        if getattr(node, "stage", None) == "draft"
        and str(getattr(node, "draft_role", "") or "")
        in TRANSFER_PORTFOLIO_DRAFT_ROLE_TO_SEAT
    }


def build_transfer_tournament_status(agent: Any) -> dict[str, Any]:
    """Return score-transparent tournament scheduling and ranking state."""

    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    enabled = bool(
        policy is not None
        and getattr(policy, "transfer_portfolio_tournament_enabled", False)
    )
    roles = [str(value) for value in list(getattr(policy, "roles", []) or [])]
    if not enabled:
        return {
            "schema": TOURNAMENT_SCHEMA,
            "enabled": False,
            "active": False,
            "phase": "disabled",
            "roles": roles,
            "target_branch_id": None,
            "allowed_branch_ids": [],
        }

    expected_roles = list(TRANSFER_PORTFOLIO_DRAFT_ROLES)
    roots = _root_by_role(agent)
    all_nodes = dict(getattr(agent, "branch_all_nodes", {}) or {})
    seat_rows: list[dict[str, Any]] = []
    for order, role in enumerate(expected_roles):
        root = roots.get(role)
        branch_id = (
            int(getattr(root, "branch_id", 0) or 0) if root is not None else None
        )
        nodes = list(all_nodes.get(branch_id, []) or []) if branch_id is not None else []
        valid_nodes = sorted(
            [node for node in nodes if _valid_metric_node(node)],
            key=lambda node: (
                float(getattr(node, "ctime", 0.0) or 0.0),
                str(getattr(node, "id", "")),
            ),
        )
        scores = [float(node.metric.value) for node in valid_nodes]
        maximize = bool(getattr(agent, "metric_maximize", True))
        best_node = None
        if valid_nodes:
            best_node = (
                max(valid_nodes, key=lambda node: float(node.metric.value))
                if maximize
                else min(valid_nodes, key=lambda node: float(node.metric.value))
            )
        completed = sum(
            1
            for node in nodes
            if getattr(node, "pending_execution", False) is not True
            and getattr(node, "is_buggy", None) is not None
        )
        seat_rows.append(
            {
                "seat_order": order,
                "seat": TRANSFER_PORTFOLIO_DRAFT_ROLE_TO_SEAT[role],
                "draft_role": role,
                "root_node_id": str(getattr(root, "id", "")) if root else None,
                "branch_id": branch_id,
                "attempted_count": len(nodes),
                "completed_count": completed,
                "in_flight_count": max(0, len(nodes) - completed),
                "valid_candidate_count": len(valid_nodes),
                "baseline_node_id": (
                    str(getattr(valid_nodes[0], "id", "")) if valid_nodes else None
                ),
                "post_baseline_valid_count": max(0, len(valid_nodes) - 1),
                "best_node_id": (
                    str(getattr(best_node, "id", "")) if best_node else None
                ),
                "best_metric": (
                    float(best_node.metric.value) if best_node is not None else None
                ),
                "valid_metrics": scores,
            }
        )

    roots_ready = roles == expected_roles and all(
        row["branch_id"] is not None for row in seat_rows
    )
    baseline_ready = roots_ready and all(
        row["valid_candidate_count"] >= 1 for row in seat_rows
    )
    round1_required = max(
        1,
        int(
            getattr(
                policy,
                "transfer_portfolio_round1_valid_successors_per_seat",
                1,
            )
            or 1
        ),
    )
    finalist_required = max(
        round1_required + 1,
        int(
            getattr(
                policy,
                "transfer_portfolio_finalist_valid_successors_per_seat",
                2,
            )
            or 2
        ),
    )
    survivor_count = max(
        1,
        min(
            len(seat_rows),
            int(getattr(policy, "transfer_portfolio_survivor_count", 2) or 2),
        ),
    )
    maximize = bool(getattr(agent, "metric_maximize", True))
    rankings = _rank_rows(seat_rows, maximize=maximize) if baseline_ready else []
    finalists = [row["draft_role"] for row in rankings[:survivor_count]]
    target_row = None
    phase = "waiting_for_roots"
    allowed_rows: list[dict[str, Any]] = []

    if roots_ready and not baseline_ready:
        phase = "baseline_qualification"
        allowed_rows = [row for row in seat_rows if row["valid_candidate_count"] < 1]
    elif baseline_ready:
        round1_deficits = [
            row
            for row in seat_rows
            if row["post_baseline_valid_count"] < round1_required
        ]
        if round1_deficits:
            phase = "round1_equal_improve"
            allowed_rows = round1_deficits
        else:
            finalist_rows = [
                next(row for row in seat_rows if row["draft_role"] == role)
                for role in finalists
            ]
            finalist_deficits = [
                row
                for row in finalist_rows
                if row["post_baseline_valid_count"] < finalist_required
            ]
            if finalist_deficits:
                phase = "top2_equal_improve"
                allowed_rows = finalist_deficits
            else:
                phase = "champion_continuation"
                reranked_finalists = _rank_rows(
                    finalist_rows,
                    maximize=maximize,
                )
                allowed_rows = reranked_finalists[:1]

    def allocation_key(row: dict[str, Any]) -> tuple[int, int, int]:
        return (
            int(row["post_baseline_valid_count"]),
            int(row["attempted_count"]),
            int(row["seat_order"]),
        )

    ordered_allowed_rows = sorted(allowed_rows, key=allocation_key)
    if ordered_allowed_rows:
        target_row = ordered_allowed_rows[0]

    # Keep the tournament work-conserving without silently advancing its
    # qualification phase.  A missing baseline remains the primary debt, but
    # if every node in that branch is temporarily or permanently unavailable,
    # a worker may spend an already-declared round-1 entitlement from another
    # baseline-qualified seat.  Once that one valid successor exists, the seat
    # disappears from this fallback queue.  Later rounds are deliberately not
    # borrowed before all baselines exist because their Top-2 membership is not
    # yet knowable.
    rollover_rows: list[dict[str, Any]] = []
    if phase == "baseline_qualification":
        allowed_roles = {str(row["draft_role"]) for row in ordered_allowed_rows}
        rollover_rows = sorted(
            [
                row
                for row in seat_rows
                if str(row["draft_role"]) not in allowed_roles
                and row["valid_candidate_count"] >= 1
                and row["post_baseline_valid_count"] < round1_required
            ],
            key=allocation_key,
        )
    allocation_rows = ordered_allowed_rows + rollover_rows
    allocation_roles = {str(row["draft_role"]) for row in allocation_rows}

    # ``allocation_rows`` is the rights/debt queue.  Do not add seats that
    # already paid the current phase debt to that queue: doing so would make a
    # donor execution look like it discharged somebody else's entitlement.
    # Instead expose a separate, deterministic single-worker donor queue.  It
    # is consulted only after every rights-bearing branch is unselectable.
    #
    # During baseline and round 1 every baseline-qualified seat is still an
    # active tournament hypothesis.  During Top-2 only finalists may donate;
    # eliminated seats must stay eliminated.  Champion continuation already
    # has an indefinitely entitled branch and therefore needs no donor.
    donor_pool: list[dict[str, Any]] = []
    if phase in {"baseline_qualification", "round1_equal_improve"}:
        donor_pool = [
            row for row in seat_rows if row["valid_candidate_count"] >= 1
        ]
    elif phase == "top2_equal_improve":
        finalist_role_set = set(finalists)
        donor_pool = [
            row
            for row in seat_rows
            if str(row["draft_role"]) in finalist_role_set
        ]
    donor_rows = sorted(
        [
            row
            for row in donor_pool
            if str(row["draft_role"]) not in allocation_roles
        ],
        key=allocation_key,
    )
    allowed_branch_ids = [
        int(row["branch_id"])
        for row in ordered_allowed_rows
        if row.get("branch_id") is not None
    ]
    rollover_branch_ids = [
        int(row["branch_id"])
        for row in rollover_rows
        if row.get("branch_id") is not None
    ]
    allocation_branch_ids = [
        int(row["branch_id"])
        for row in allocation_rows
        if row.get("branch_id") is not None
    ]
    donor_branch_ids = [
        int(row["branch_id"])
        for row in donor_rows
        if row.get("branch_id") is not None
    ]
    champion_role = (
        str(allowed_rows[0]["draft_role"])
        if phase == "champion_continuation" and allowed_rows
        else None
    )
    return {
        "schema": TOURNAMENT_SCHEMA,
        "enabled": True,
        "active": roots_ready,
        "phase": phase,
        "roles": roles,
        "expected_roles": expected_roles,
        "round1_valid_successors_per_seat": round1_required,
        "finalist_valid_successors_per_seat": finalist_required,
        "survivor_count": survivor_count,
        "seats": seat_rows,
        "rankings": rankings,
        "finalist_roles": finalists if rankings else [],
        "champion_role": champion_role,
        "allowed_branch_ids": allowed_branch_ids,
        "rollover_branch_ids": rollover_branch_ids,
        "allocation_branch_ids": allocation_branch_ids,
        "work_conserving_donor_branch_ids": donor_branch_ids,
        "allocation_queue": [
            {
                "branch_id": int(row["branch_id"]),
                "draft_role": str(row["draft_role"]),
                "seat": str(row["seat"]),
                "allocation_kind": (
                    "phase_debt"
                    if row in ordered_allowed_rows
                    else "early_round1_rollover"
                ),
            }
            for row in allocation_rows
            if row.get("branch_id") is not None
        ],
        "work_conserving_donor_queue": [
            {
                "branch_id": int(row["branch_id"]),
                "draft_role": str(row["draft_role"]),
                "seat": str(row["seat"]),
                "allocation_kind": "idle_capacity_donor",
            }
            for row in donor_rows
            if row.get("branch_id") is not None
        ],
        "target_branch_id": (
            int(target_row["branch_id"])
            if target_row is not None and target_row.get("branch_id") is not None
            else None
        ),
        "target_role": str(target_row["draft_role"]) if target_row else None,
        "selection_binding": False,
    }


def persist_transfer_tournament_status(agent: Any, status: dict[str, Any]) -> None:
    """Atomically persist a compact machine-auditable tournament receipt."""

    if not status.get("enabled"):
        return
    log_dir = Path(str(getattr(getattr(agent, "cfg", None), "log_dir", "") or ""))
    if not str(log_dir):
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "TRANSFER_PORTFOLIO_TOURNAMENT.json"
    payload = json.dumps(status, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == payload:
        return
    temp = path.with_suffix(".json.tmp")
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)
