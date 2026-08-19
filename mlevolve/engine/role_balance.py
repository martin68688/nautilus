"""Host-owned startup resource fairness for fixed Draft roles."""

from __future__ import annotations


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
            role = str(getattr(node, "draft_role", "") or "")
            if role not in completed_counts:
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
            role = str(getattr(node, "draft_role", "") or "")
            metric = getattr(node, "metric", None)
            if (
                role in valid_counts
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
        "minimum_valid_candidates": minimum,
        "roles": roles,
        "valid_counts": valid_counts,
        "completed_counts": completed_counts,
        "host_instrumentation_failures": host_instrumentation_failures,
        "deficit_roles": deficit_roles,
        "next_role": next_role,
    }
