"""Node selection: UCT select, get_exploration_weight, get_top_k_nodes_global, select_from_top_k_weighted, select_with_soft_switch."""

import logging
import random
import time
from typing import List, Optional

from engine.search_node import SearchNode
from engine.conditions import (
    coverage_synthesis_due,
    cross_role_synthesis_enabled,
    should_trigger_branch_fusion,
)
logger = logging.getLogger("MLEvolve")


def _is_repair_only(node: SearchNode) -> bool:
    transaction = getattr(node, "protocol_repair", None) or {}
    protocol_active = transaction.get("state") in {
        "pending", "stage_in_progress", "final_pending"
    }
    return bool(
        protocol_active
        or getattr(node, "audit_repair_required", False)
        or (getattr(node, "leakage_audit", None) or {}).get("search_disposition") == "repair_only"
    )


def _preflight_selectable(node: SearchNode) -> bool:
    report = getattr(node, "protocol_preflight", None) or {}
    return bool(
        not report
        or report.get("status") == "pass"
        or report.get("enforcement_mode") == "shadow"
    )


def _uct_selectable(node: SearchNode) -> bool:
    return (
        not node.is_terminal
        and not _is_repair_only(node)
        and _preflight_selectable(node)
    )


def _piecewise_decay(t, initial_C=1.414, T1=100, T2=200, alpha=0.01, lower_bound=0.7):
    """Piecewise decay: initial_C until T1, linear to lower_bound by T2, then lower_bound."""
    if t < T1:
        return initial_C
    elif T1 <= t <= T2:
        return max(initial_C - alpha * (t - T1), lower_bound)
    else:
        return lower_bound


def _compute_exploration_constant(agent):
    """Compute exploration constant C from search progress (piecewise decay)."""
    dcfg = agent.cfg.agent.decay
    n1 = agent.scfg.num_drafts * (agent.scfg.num_improves ** 2)
    n2 = round(agent.acfg.steps * dcfg.phase_ratios[0])
    t1 = min(n1, n2)
    t2 = round(agent.acfg.steps * dcfg.phase_ratios[1])
    return _piecewise_decay(
        t=agent.current_step,
        initial_C=dcfg.exploration_constant,
        T1=t1,
        T2=t2,
        alpha=dcfg.alpha,
        lower_bound=dcfg.lower_bound,
    )


def select(agent, node: SearchNode) -> Optional[SearchNode]:
    """UCT selection: recurse from node, return node to expand (root lock for drafts)."""
    def _best_child(n: SearchNode) -> Optional[SearchNode]:
        C = _compute_exploration_constant(agent)
        if agent.is_root(n):
            filtered_children = [
                child for child in n.children
                if not child.lock and _uct_selectable(child)
            ]
            if not filtered_children:
                return None
            selected_node = max(filtered_children,
                                key=lambda child: child.uct_value(exploration_constant=C))
            if selected_node.stage in ["draft", "fusion_draft"]:
                selected_node.lock = True
            return selected_node
        else:
            filtered_children = [child for child in n.children if _uct_selectable(child)]
            if not filtered_children:
                return None
            return max(
                filtered_children,
                key=lambda child: child.uct_value(exploration_constant=C),
            )

    while node and not node.is_terminal:
        fixed_slots_exhausted = bool(
            agent.is_root(node)
            and hasattr(agent, "fixed_draft_slots_exhausted")
            and agent.fixed_draft_slots_exhausted()
        )
        if not fixed_slots_exhausted and not node.reached_child_limit(scfg=agent.scfg):
            if node.is_buggy and node.is_debug_success is True:
                node = _best_child(node)
            elif node.continue_improve and len(node.children) > 0:
                node = _best_child(node)
            else:
                logger.info(f"[select] → node {node.id} (method=expand)")
                return node
        else:
            fixed_roles = bool(
                getattr(getattr(agent.acfg, "draft_role_policy", None), "enabled", False)
            )
            fixed_role_synthesis = cross_role_synthesis_enabled(agent)
            if (
                agent.is_root(node)
                and (not fixed_roles or fixed_role_synthesis)
                and should_trigger_branch_fusion(agent)
                and random.random() < agent.acfg.branch_fusion_trigger_prob
            ):
                logger.info(f"Root node {node.id} is fully expanded for regular drafts, aggregation conditions met (including probability), returning root")
                node._aggregation_requested = True
                return node
            next_node = _best_child(node)
            if next_node is None:
                logger.info("[select] → wait (all expandable root children are in flight)")
                return None
            node = next_node
    if node is None:
        logger.info("[select] → wait (no selectable node)")
        return None
    logger.info(f"[select] → node {node.id} (method=uct)")
    return node


def get_exploration_weight(time_elapsed: float, total_time: float,
                           switch_start: float = 0.5,
                           switch_end: float = 0.7,
                           min_weight: float = 0.2) -> float:
    """Exploration weight: 1.0 until switch_start, linear decay to min_weight by switch_end."""
    time_progress = time_elapsed / total_time

    if time_progress < switch_start:
        return 1.0
    elif time_progress < switch_end:
        decay_progress = (time_progress - switch_start) / (switch_end - switch_start)
        return 1.0 - (1.0 - min_weight) * decay_progress
    else:
        return min_weight


def get_top_k_nodes_global(
    agent,
    k: int,
    max_from_same_branch: int,
    branch_ids: set[int] | None = None,
) -> List[dict]:
    """Select top-k nodes globally with branch diversity (recomputed each call). Returns list of {node, branch_id, metric, rank}."""
    all_nodes = []
    for branch_id in agent.branch_all_nodes:
        if branch_ids is not None and branch_id not in branch_ids:
            continue
        for node in agent.branch_all_nodes[branch_id]:
            if (
                not node.is_buggy
                and _preflight_selectable(node)
                and node.metric is not None
                and node.metric.value is not None
            ):
                all_nodes.append(node)

    # Complete mediation: authorize the entire candidate set before metrics are
    # sorted or branch quotas are applied. Partial/denied paths cannot perturb
    # legal candidates' rank positions.
    from authority.adapters.mlevolve.ranking_gate import filter_ranked_nodes
    all_nodes = filter_ranked_nodes(
        agent,
        all_nodes,
        component="engine.node_selection.get_top_k_nodes_global",
    )

    if not all_nodes:
        logger.warning("No valid nodes found for Top-K selection")
        return []

    maximize = agent.metric_maximize
    all_nodes.sort(
        key=lambda n: n.metric.value,
        reverse=maximize
    )

    logger.info(f"Total valid nodes: {len(all_nodes)}, requesting Top-{k}")

    selected = []
    branch_count = {}

    for node in all_nodes:
        if len(selected) >= k:
            break

        branch_id = node.branch_id
        current_count = branch_count.get(branch_id, 0)

        if current_count >= max_from_same_branch:
            logger.debug(f"Branch {branch_id} reached limit ({max_from_same_branch}), skipping node with metric={node.metric.value:.4f}")
            continue

        selected.append({
            'node': node,
            'branch_id': branch_id,
            'metric': node.metric.value,
            'rank': len(selected) + 1
        })
        branch_count[branch_id] = current_count + 1

    if selected:
        branch_distribution = {}
        for item in selected:
            bid = item['branch_id']
            branch_distribution[bid] = branch_distribution.get(bid, 0) + 1

        metrics_str = ", ".join([f"Rank{item['rank']}={item['metric']:.4f}(B{item['branch_id']})" for item in selected])
        logger.info(f"📊 Top-{len(selected)} selected: {metrics_str}")
        logger.info(f"📊 Branch distribution: {branch_distribution}")

    return selected


def select_from_top_k_weighted(
    agent,
    top_k_nodes: List[dict],
    fallback_root: SearchNode | None = None,
) -> Optional[SearchNode]:
    """Weighted random choice from top-k nodes (weight = 1/rank)."""
    if not top_k_nodes:
        return select(agent, fallback_root or agent.virtual_root)

    weights = [1.0 / item['rank'] for item in top_k_nodes]
    total_weight = sum(weights)
    probabilities = [w / total_weight for w in weights]
    selected = random.choices(top_k_nodes, weights=probabilities)[0]

    from agents.leakage_audit import legacy_rank_eligible
    from authority.adapters.mlevolve.ranking_gate import authorize_selection
    if not authorize_selection(
        agent,
        selected["node"],
        legacy_allowed=legacy_rank_eligible(agent, selected["node"]),
        component="engine.node_selection.select_from_top_k_weighted",
    ):
        remaining = [item for item in top_k_nodes if item is not selected]
        return select_from_top_k_weighted(
            agent,
            remaining,
            fallback_root=fallback_root,
        )

    logger.info(f"🎯 Selected: Rank{selected['rank']} (Branch {selected['branch_id']}, "
                f"metric={selected['metric']:.4f}, prob={probabilities[top_k_nodes.index(selected)]:.1%})")

    return selected['node']


def select_with_soft_switch(agent) -> Optional[SearchNode]:
    """Soft switch: exploration (UCT) vs exploitation (Top-K) by time progress."""
    # A completed two-role coverage contract owes exactly one root Fusion.
    # Route through the root selector before random explore/exploit choices can
    # spend another turn on either source branch.
    if coverage_synthesis_due(agent):
        logger.info("[select] prioritizing protected two-role coverage synthesis")
        return select(agent, agent.virtual_root)

    from engine.role_balance import build_branch_fairness_status

    fairness = build_branch_fairness_status(agent)
    if fairness.get("active"):
        return select_branch_fair(agent, fairness)

    if agent.search_start_time is None:
        logger.info("📊 Search not started yet, using standard UCT")
        return select(agent, agent.virtual_root)

    time_elapsed = time.time() - agent.search_start_time
    total_time = agent.acfg.time_limit
    time_progress = time_elapsed / total_time

    scfg = agent.scfg

    exploration_weight = get_exploration_weight(
        time_elapsed, total_time,
        switch_start=scfg.explore_switch_start,
        switch_end=scfg.explore_switch_end,
        min_weight=scfg.min_exploration_weight,
    )

    if random.random() < exploration_weight:
        logger.info(f"📊 Exploration mode (weight={exploration_weight:.2%}, "
                   f"time={time_progress:.1%})")
        return select(agent, agent.virtual_root)

    else:
        # Top-K exploitation
        logger.info(f"🎯 Exploitation mode (weight={1-exploration_weight:.2%}, "
                   f"time={time_progress:.1%})")

        if time_progress < scfg.explore_switch_end:
            k = scfg.topk_early_k
            max_from_same_branch = scfg.topk_early_max_per_branch
            phase = f"early-mid (<{scfg.explore_switch_end:.0%})"
        else:
            k = scfg.topk_late_k
            max_from_same_branch = scfg.topk_late_max_per_branch
            phase = f"late (>={scfg.explore_switch_end:.0%})"

        logger.info(f"📊 Phase: {phase}, requesting Top-{k} (max {max_from_same_branch} per branch)")

        top_k_nodes = get_top_k_nodes_global(
            agent,
            k=k,
            max_from_same_branch=max_from_same_branch
        )

        if not top_k_nodes:
            logger.warning("No valid Top-K nodes found, fallback to standard UCT")
            return select(agent, agent.virtual_root)

        available_nodes = [
            item for item in top_k_nodes
            if not item['node'].reached_child_limit(agent.scfg, for_topk=True)
        ]

        if available_nodes:
            selected_node = select_from_top_k_weighted(agent, available_nodes)
            if selected_node is None:
                return None
            logger.info(f"✅ Selected unexpanded Top-K node {selected_node.id} (from {len(available_nodes)}/{len(top_k_nodes)} available)")
            selected_node._topk_triggered = True
            return selected_node
        else:
            logger.info(f"⚠️ All Top-{len(top_k_nodes)} nodes fully expanded, will apply UCT from selected node")
            selected_node = select_from_top_k_weighted(agent, top_k_nodes)
            if selected_node is None:
                return None
            logger.info(f"Selected fully expanded node {selected_node.id}, applying UCT from it")
            uct_node = select(agent, selected_node)
            if uct_node is None:
                return None
            uct_node._topk_triggered = True
            return uct_node


def _branch_root(agent, root_node_id: str) -> SearchNode | None:
    return next(
        (
            node
            for node in getattr(agent.virtual_root, "children", set())
            if str(node.id) == str(root_node_id)
        ),
        None,
    )


def _select_within_branch(
    agent,
    branch_root: SearchNode,
    branch_id: int,
) -> Optional[SearchNode]:
    """Preserve the legacy explore/exploit policy inside one chosen branch."""

    if branch_root.lock or branch_root.is_terminal or not _uct_selectable(branch_root):
        return None
    if agent.search_start_time is None:
        return select(agent, branch_root)

    time_elapsed = time.time() - agent.search_start_time
    total_time = agent.acfg.time_limit
    time_progress = time_elapsed / total_time
    scfg = agent.scfg
    exploration_weight = get_exploration_weight(
        time_elapsed,
        total_time,
        switch_start=scfg.explore_switch_start,
        switch_end=scfg.explore_switch_end,
        min_weight=scfg.min_exploration_weight,
    )
    if random.random() < exploration_weight:
        logger.info(
            "[branch-fair] branch=%s internal=uct exploration_weight=%.2f%%",
            branch_id,
            exploration_weight * 100,
        )
        return select(agent, branch_root)

    if time_progress < scfg.explore_switch_end:
        k = scfg.topk_early_k
        max_per_branch = scfg.topk_early_max_per_branch
    else:
        k = scfg.topk_late_k
        max_per_branch = scfg.topk_late_max_per_branch
    top_k_nodes = get_top_k_nodes_global(
        agent,
        k=k,
        max_from_same_branch=max(k, max_per_branch),
        branch_ids={branch_id},
    )
    if not top_k_nodes:
        return select(agent, branch_root)
    available = [
        item
        for item in top_k_nodes
        if not item["node"].reached_child_limit(agent.scfg, for_topk=True)
    ]
    selected = select_from_top_k_weighted(
        agent,
        available or top_k_nodes,
        fallback_root=branch_root,
    )
    if selected is None:
        return None
    if not available:
        selected = select(agent, selected)
        if selected is None:
            return None
    selected._topk_triggered = True
    logger.info(
        "[branch-fair] branch=%s internal=topk candidate=%s",
        branch_id,
        selected.id,
    )
    return selected


def select_branch_fair(agent, fairness: dict | None = None) -> Optional[SearchNode]:
    """Choose the least-used protected branch, then select only within it."""

    if fairness is None:
        from engine.role_balance import build_branch_fairness_status

        fairness = build_branch_fairness_status(agent)
    if not fairness.get("active"):
        return None
    rows = {
        int(row["branch_id"]): row for row in fairness.get("branches", [])
    }
    for branch_id in fairness.get("ordered_branch_ids", []):
        row = rows[int(branch_id)]
        root = _branch_root(agent, row["root_node_id"])
        if root is None:
            continue
        selected = _select_within_branch(agent, root, int(branch_id))
        if selected is not None:
            logger.info(
                "[branch-fair] allocated=%s branch=%s attempts=%s completed=%s in_flight=%s",
                row["name"],
                branch_id,
                row["attempted_count"],
                row["completed_count"],
                row["in_flight_count"],
            )
            return selected
        logger.info(
            "[branch-fair] temporarily unavailable=%s branch=%s debt_retained=true",
            row["name"],
            branch_id,
        )
    logger.info("[branch-fair] no protected branch is currently selectable")
    return None


def select_role_balance_deficit(agent, role: str) -> Optional[SearchNode]:
    """Select expandable work only from one Host-designated Draft role.

    This is a startup resource gate, not a new ranking algorithm.  Once every
    role has enough valid Candidates, the caller returns to the unchanged
    UCT/Top-K policy above.
    """

    roots = sorted(
        [
            node
            for node in getattr(agent.virtual_root, "children", set())
            if str(getattr(node, "draft_role", "") or "") == str(role)
        ],
        key=lambda node: (node.ctime, str(node.id)),
    )
    for root in roots:
        if root.lock or root.is_terminal or not _uct_selectable(root):
            continue
        selected = select(agent, root)
        if selected is None:
            continue
        if selected.stage in {"draft", "fusion_draft"}:
            selected.lock = True
        logger.info(
            "[role-balance] selected role=%s node=%s valid-score-independent=true",
            role,
            selected.id,
        )
        return selected
    logger.info("[role-balance] role=%s has no currently selectable node", role)
    return None
