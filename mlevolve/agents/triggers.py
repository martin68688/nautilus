import logging

from agents.leakage_audit import rank_eligible
from engine.search_node import SearchNode

from engine.conditions import should_trigger_branch_fusion

logger = logging.getLogger("MLEvolve")


def should_check_data_leakage(agent, node: SearchNode) -> bool:
    # Every node that successfully produced a validation metric is checked.
    # The old extreme-metric heuristic missed moderate train/validation leaks
    # that still looked plausible, so leakage detection now fails open on all
    # metric-bearing nodes except known-buggy WorstMetricValue nodes.
    if node.metric is None or node.metric.is_worst:
        return False

    logger.info(
        f"Node {node.id} queued for data leakage check "
        f"(metric={node.metric.value}, maximize={agent.metric_maximize})."
    )
    return True


def get_patience_counter(agent, parent_node: SearchNode) -> tuple:
    if not hasattr(parent_node, 'branch_id') or parent_node.branch_id is None:
        return 0, 0, None

    branch_successful_nodes = agent.branch_successful_nodes.get(parent_node.branch_id, [])
    branch_all_nodes = agent.branch_all_nodes.get(parent_node.branch_id, [])

    if len(branch_successful_nodes) == 0:
        return 0, len(branch_all_nodes), None

    valid_nodes = [
        n for n in branch_successful_nodes
        if n.metric and n.metric.value is not None and rank_eligible(agent, n)
    ]
    if not valid_nodes:
        return 0, len(branch_all_nodes), None

    best_node = max(valid_nodes, key=lambda n: n.metric)
    branch_best_score = best_node.metric.value

    try:
        best_idx_success = branch_successful_nodes.index(best_node)
    except ValueError:
        logger.warning(f"Best node {best_node.id} not found in branch_successful_nodes list")
        return 0, len(branch_all_nodes), branch_best_score

    try:
        best_idx_all = branch_all_nodes.index(best_node)
    except ValueError:
        logger.warning(f"Best node {best_node.id} not found in branch_all_nodes list")
        best_idx_all = 0

    success_patience = len(branch_successful_nodes) - best_idx_success - 1
    total_patience = len(branch_all_nodes) - best_idx_all - 1

    logger.info(
        f"🔥 Patience counters: success={success_patience}, total={total_patience} "
        f"(best at pos {best_idx_all+1}/{len(branch_all_nodes)} overall, "
        f"{best_idx_success+1}/{len(branch_successful_nodes)} successful, "
        f"metric={branch_best_score:.4f}, id={best_node.id[:8]})"
    )

    return max(0, success_patience), max(0, total_patience), branch_best_score


def register_node(agent, node: SearchNode, prompt, parent_node=None, new_branch: bool = False):
    import copy
    import time

    node.prompt_input = agent._serialize_prompt(prompt)
    node.created_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    if new_branch:
        node.branch_id = agent.next_branch_id
        agent.next_branch_id += 1
        agent.branch_all_nodes[node.branch_id] = [node]
        agent.branch_successful_nodes[node.branch_id] = []
    else:
        node.branch_id = parent_node.branch_id
        if node.draft_role is None:
            node.draft_role = parent_node.draft_role
        if not node.role_contract:
            node.role_contract = copy.deepcopy(parent_node.role_contract)
        if not node.source_ref_ids:
            node.source_ref_ids = list(parent_node.source_ref_ids)
        if not node.replay_source:
            node.replay_source = copy.deepcopy(parent_node.replay_source)
        if node.replay_status is None:
            node.replay_status = parent_node.replay_status
        parent_audit = parent_node.leakage_audit or {}
        if parent_audit.get("status") not in {None, "clean"}:
            from agents.leakage_audit import build_repair_preservation_contract

            preservation_contract = copy.deepcopy(
                parent_node.leakage_repair_context.get("preservation_contract", {})
            ) or build_repair_preservation_contract(parent_node.code)
            node.leakage_repair_context = {
                "source_node_id": parent_node.id,
                "source_code_sha256": parent_audit.get("code_sha256"),
                "status": parent_audit.get("status"),
                "issues": copy.deepcopy(parent_audit.get("issues") or []),
                "preservation_contract": preservation_contract,
            }
            node.leakage_repair_attempt = parent_node.leakage_repair_attempt + 1
            node.audit_repair_required = True
            if node.replay_source:
                # `requires_repair` describes the replay lineage. Only the
                # immutable source node is a non-executable repair seed.
                node.replay_source["repair_seed_only"] = False
                node.replay_source["repair_parent_node_id"] = parent_node.id
                node.replay_status = "mandatory_audit_repair"
        if node.branch_id in agent.branch_all_nodes:
            agent.branch_all_nodes[node.branch_id].append(node)
