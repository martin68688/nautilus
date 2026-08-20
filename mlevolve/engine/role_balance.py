"""Host-owned startup resource fairness for fixed Draft roles."""

from __future__ import annotations


def is_replay_derived_novel(node) -> bool:
    """Whether a Novel-labeled node originated by modifying Replay code."""

    replay_source = getattr(node, "replay_source", None) or {}
    role_contract = getattr(node, "role_contract", None) or {}
    return bool(
        replay_source.get("lineage_kind") == "replay_derived_novel"
        or role_contract.get("behavioral_role") == "replay_derived_novel"
    )


def candidate_matches_protected_role(node, role: str) -> bool:
    """Match readiness to an independent configured Draft origin.

    Replay adaptations are deliberately relabeled as Novel for honest ranking,
    but they must not satisfy the independent Novel coverage prerequisite used
    by two-role synthesis.
    """

    node_role = str(getattr(node, "draft_role", "") or "")
    required_role = str(role or "")
    if required_role == "novel_exploration":
        from engine.draft_roles import canonical_draft_role

        return bool(
            canonical_draft_role(node_role) == "novel_exploration"
            and not is_replay_derived_novel(node)
        )
    return node_role == required_role


def build_role_balance_status(agent) -> dict:
    """Count valid Candidates per role without comparing their scores."""

    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    enabled = bool(
        policy is not None
        and getattr(policy, "enabled", False)
        and getattr(policy, "ensure_valid_candidate_per_role", False)
    )
    roles = [str(role) for role in list(getattr(policy, "roles", []) or [])]
    minimum = max(
        1,
        int(getattr(policy, "role_balance_min_valid_candidates", 1) or 1),
    )
    slots_exhausted = getattr(agent, "fixed_draft_slots_exhausted", None)
    all_slots_reserved = bool(
        callable(slots_exhausted) and slots_exhausted()
    )
    if not enabled or not roles or not all_slots_reserved:
        return {
            "enabled": enabled,
            "active": False,
            "all_slots_reserved": all_slots_reserved,
            "minimum_valid_candidates": minimum,
            "roles": roles,
            "valid_counts": {},
            "completed_counts": {},
            "deficit_roles": [],
            "next_role": None,
        }

    valid_counts: dict[str, int] = {role: 0 for role in roles}
    completed_counts: dict[str, int] = {role: 0 for role in roles}
    host_instrumentation_failures: dict[str, int] = {role: 0 for role in roles}
    for nodes in dict(getattr(agent, "branch_all_nodes", {}) or {}).values():
        for node in nodes:
            for role in roles:
                if not candidate_matches_protected_role(node, role):
                    continue
                if getattr(node, "exc_type", None) == "HostSourceInstrumentationError":
                    host_instrumentation_failures[role] += 1
                    continue
                if (
                    getattr(node, "pending_execution", False) is not True
                    and getattr(node, "is_buggy", None) is not None
                ):
                    completed_counts[role] += 1

    for nodes in dict(getattr(agent, "branch_successful_nodes", {}) or {}).values():
        for node in nodes:
            metric = getattr(node, "metric", None)
            for role in roles:
                if (
                    candidate_matches_protected_role(node, role)
                    and getattr(node, "is_buggy", None) is False
                    and getattr(node, "is_valid", None) is not False
                    and metric is not None
                    and getattr(metric, "value", None) is not None
                ):
                    valid_counts[role] += 1

    deficit_roles = [role for role in roles if valid_counts[role] < minimum]
    next_role = None
    if deficit_roles:
        role_order = {role: index for index, role in enumerate(roles)}
        next_role = min(
            deficit_roles,
            key=lambda role: (
                # A deterministic Host instrumentation failure is a system
                # defect, not evidence that the role deserves less compute.
                # Repair that role before normal count-based balancing.
                0 if host_instrumentation_failures[role] else 1,
                valid_counts[role],
                completed_counts[role],
                role_order[role],
            ),
        )
    return {
        "enabled": True,
        "active": bool(deficit_roles),
        "all_slots_reserved": all_slots_reserved,
        "minimum_valid_candidates": minimum,
        "roles": roles,
        "valid_counts": valid_counts,
        "completed_counts": completed_counts,
        "host_instrumentation_failures": host_instrumentation_failures,
        "deficit_roles": deficit_roles,
        "next_role": next_role,
    }


def _protected_branch_roots(agent) -> list[tuple[str, object]]:
    """Return the Replay, independent Novel, and first coverage-Fusion roots."""

    roots = sorted(
        list(getattr(getattr(agent, "virtual_root", None), "children", set()) or []),
        key=lambda node: (float(getattr(node, "ctime", 0.0) or 0.0), str(node.id)),
    )
    replay = next(
        (
            node
            for node in roots
            if getattr(node, "stage", None) == "draft"
            and getattr(node, "draft_role", None) == "memory_reproduction"
        ),
        None,
    )
    novel = next(
        (
            node
            for node in roots
            if getattr(node, "stage", None) == "draft"
            and candidate_matches_protected_role(node, "novel_exploration")
        ),
        None,
    )
    fusion = next(
        (
            node
            for node in roots
            if getattr(node, "stage", None) == "fusion_draft"
            and (getattr(node, "role_contract", None) or {}).get(
                "behavioral_role"
            )
            == "cross_role_synthesis"
        ),
        None,
    )
    return [
        (name, node)
        for name, node in (
            ("replay", replay),
            ("novel", novel),
            ("fusion", fusion),
        )
        if node is not None
    ]


def build_branch_fairness_status(agent) -> dict:
    """Build score-independent phase-2 allocation state for three branches.

    A registered candidate is an attempt even when it is buggy, invalid, or
    still in flight. This prevents a failing branch from consuming unlimited
    retries while successful branches are charged for every completed child.
    """

    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    enabled = bool(
        policy is not None
        and getattr(policy, "enabled", False)
        and getattr(policy, "equal_branch_allocation_after_coverage", False)
    )
    fusion_created = int(getattr(agent, "fusion_draft_count", 0) or 0) >= 1
    roots = _protected_branch_roots(agent) if enabled and fusion_created else []
    complete = len(roots) == 3

    branches = []
    all_nodes = dict(getattr(agent, "branch_all_nodes", {}) or {})
    for order, (name, root) in enumerate(roots):
        branch_id = int(getattr(root, "branch_id", 0) or 0)
        nodes = list(all_nodes.get(branch_id, []) or [])
        completed = sum(
            1
            for node in nodes
            if getattr(node, "pending_execution", False) is not True
            and getattr(node, "is_buggy", None) is not None
        )
        branches.append(
            {
                "name": name,
                "order": order,
                "branch_id": branch_id,
                "root_node_id": str(root.id),
                # branch_all_nodes is populated at registration time, before
                # execution, so this includes completed, failed and in-flight
                # candidate attempts without looking at their metric values.
                "attempted_count": len(nodes),
                "completed_count": completed,
                "in_flight_count": max(0, len(nodes) - completed),
            }
        )

    ordered = sorted(
        branches,
        key=lambda row: (
            row["attempted_count"],
            row["order"],
            row["branch_id"],
        ),
    )
    return {
        "enabled": enabled,
        "active": bool(enabled and fusion_created and complete),
        "fusion_created": fusion_created,
        "complete": complete,
        "branches": branches,
        "ordered_branch_ids": [row["branch_id"] for row in ordered],
        "next_branch_id": ordered[0]["branch_id"] if complete else None,
        "next_branch_name": ordered[0]["name"] if complete else None,
    }
