"""Search conditions shared by every branch-fusion entry point."""

import logging
import time
from typing import Mapping

from agents.leakage_audit import rank_eligible
logger = logging.getLogger("MLEvolve")


def cross_role_synthesis_enabled(agent) -> bool:
    """Return whether this run opted into balanced cross-role synthesis."""

    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    return bool(
        policy is not None
        and getattr(policy, "enabled", False)
        and getattr(policy, "cross_role_synthesis_after_balance", False)
    )


def cross_role_synthesis_allowed(agent, *, component: str) -> bool:
    """Fail closed while any protected Draft role lacks valid coverage.

    Controls that do not opt into the Dynamic-only contract retain their old
    behavior.  Once enabled, every aggregation/fusion caller uses this same
    Host decision instead of maintaining a separate interpretation of role
    balance.
    """

    if not cross_role_synthesis_enabled(agent):
        return True
    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    if bool(
        getattr(policy, "single_coverage_synthesis_only", False)
        and int(getattr(agent, "fusion_draft_count", 0) or 0) >= 1
    ):
        logger.info(
            "Cross-role synthesis denied in %s: protected coverage Fusion already exists",
            component,
        )
        return False
    status_fn = getattr(agent, "role_balance_status", None)
    if not callable(status_fn):
        logger.error(
            "Cross-role synthesis denied in %s: role_balance_status is unavailable",
            component,
        )
        return False
    balance = status_fn()
    if (
        not isinstance(balance, Mapping)
        or balance.get("enabled") is not True
        or balance.get("all_slots_reserved") is not True
        or balance.get("active") is not False
        or bool(balance.get("deficit_roles"))
    ):
        deficits = (
            balance.get("deficit_roles") or ["roles_not_yet_reserved"]
            if isinstance(balance, Mapping)
            else ["unknown"]
        )
        logger.info(
            "Cross-role synthesis denied in %s; role deficits=%s",
            component,
            deficits,
        )
        return False
    return True


def coverage_synthesis_due(agent) -> bool:
    """Whether the opted-in two-role run owes its first protected Fusion."""

    policy = getattr(getattr(agent, "acfg", None), "draft_role_policy", None)
    if not bool(
        policy is not None
        and getattr(policy, "enabled", False)
        and getattr(policy, "cross_role_synthesis_on_coverage", False)
    ):
        return False
    if int(getattr(agent, "fusion_draft_count", 0) or 0) != 0:
        return False
    if int(getattr(agent, "max_fusion_drafts", 0) or 0) < 1:
        return False
    roles = tuple(str(role) for role in list(getattr(policy, "roles", []) or []))
    if roles not in {
        ("memory_reproduction", "novel_exploration"),
        ("memory_transfer", "novel_exploration"),
    }:
        return False
    return cross_role_synthesis_allowed(
        agent,
        component="engine.conditions.coverage_synthesis_due",
    )


def should_trigger_branch_fusion(agent) -> bool:
    """Whether to trigger multi-branch aggregation: time window, min branches with success, global stagnation, under max attempts."""
    if not cross_role_synthesis_allowed(
        agent,
        component="engine.conditions.should_trigger_branch_fusion",
    ):
        return False
    if agent.fusion_draft_count >= agent.max_fusion_drafts:
        return False

    # The first two-role Fusion is a deterministic coverage milestone. It must
    # not wait for the legacy six-hour/stagnation window; later Fusion attempts
    # continue to use the unchanged legacy conditions below.
    if coverage_synthesis_due(agent):
        logger.info(
            "Two-role coverage complete; first protected cross-role synthesis is due"
        )
        return True

    if not agent.search_start_time:
        return False

    scfg = agent.scfg
    elapsed_time = time.time() - agent.search_start_time
    if elapsed_time < scfg.fusion_min_time_hours * 3600 or elapsed_time > scfg.fusion_max_time_hours * 3600:
        return False

    successful_branches = [
        bid for bid, nodes in agent.branch_successful_nodes.items()
        if len([node for node in nodes if rank_eligible(agent, node)]) >= scfg.fusion_min_successful_nodes
    ]
    if len(successful_branches) < scfg.fusion_min_branches:
        return False

    if not is_globally_stagnant(agent):
        return False

    logger.info(
        f"Branch fusion conditions met at {elapsed_time/3600:.1f}h "
        f"with {len(successful_branches)} successful branches"
    )
    return True


def is_branch_stagnant(agent, branch_id: int, threshold: int = 3) -> bool:
    """True if branch has no improvement over branch best for the last threshold attempts."""
    if branch_id not in agent.branch_successful_nodes:
        return False

    successful_nodes = [
        node for node in agent.branch_successful_nodes[branch_id]
        if rank_eligible(agent, node)
    ]
    if len(successful_nodes) < 1:
        return False

    maximize = agent.metric_maximize if agent.metric_maximize is not None else True

    sorted_nodes = sorted(
        successful_nodes,
        key=lambda n: n.metric.value if n.metric and n.metric.value is not None else (
            float('-inf') if maximize else float('inf')),
        reverse=maximize
    )

    branch_best_metric = sorted_nodes[0].metric.value
    if branch_best_metric is None:
        return False

    consecutive_no_improvement = 0
    max_consecutive = threshold

    recent_nodes = successful_nodes[-max_consecutive:] if len(
        successful_nodes) >= max_consecutive else successful_nodes

    for node in recent_nodes:
        if node.metric and node.metric.value is not None:
            if maximize:
                if node.metric.value >= branch_best_metric:
                    break
            else:
                if node.metric.value <= branch_best_metric:
                    break
            consecutive_no_improvement += 1

    if consecutive_no_improvement >= len(recent_nodes) and len(recent_nodes) >= 2:
        logger.info(
            f"Branch {branch_id} stagnant: {consecutive_no_improvement} consecutive attempts "
            f"didn't exceed branch best {branch_best_metric}")
        return True

    return False


def is_globally_stagnant(agent) -> bool:
    """True if no significant improvement in the last window_size nodes."""
    if not agent.best_node or not agent.best_node.metric:
        return False

    window_size = agent.stagnation_threshold

    if len(agent.journal.nodes) < window_size:
        return False

    recent_nodes = agent.journal.nodes[-window_size:]
    current_best_metric = agent.best_node.metric

    for node in recent_nodes:
        if rank_eligible(agent, node) and node.metric and node.metric.value is not None:
            if agent.metric_maximize:
                improvement = node.metric.value - current_best_metric.value
            else:
                improvement = current_best_metric.value - node.metric.value

            if improvement > agent.scfg.metric_improvement_threshold:
                return False

    logger.info(f"Global stagnation detected: no improvement beyond threshold in last {window_size} nodes")
    return True
