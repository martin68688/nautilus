import logging
import copy
import hashlib

from agents.leakage_audit import rank_eligible
from engine.search_node import SearchNode

from engine.conditions import should_trigger_branch_fusion

logger = logging.getLogger("MLEvolve")


_EXACT_REPLAY_STATUSES = {
    "exact_source_loaded",
    "exact_source_loaded_fixed_holdout",
}


def _sha256_code(code: str) -> str:
    return hashlib.sha256(str(code or "").encode("utf-8")).hexdigest()


def _update_replay_lineage(parent_node: SearchNode, node: SearchNode) -> None:
    """Keep source provenance without laundering a modified child as exact replay.

    ``replay_source`` describes the immutable historical origin and is therefore
    inherited.  ``replay_status`` describes the *current* artifact and must be
    recomputed whenever code changes.
    """

    if not node.replay_source:
        return
    source_hash = str(node.replay_source.get("code_sha256") or "")
    child_hash = _sha256_code(node.code)
    parent_hash = _sha256_code(parent_node.code)
    node.replay_source["current_code_sha256"] = child_hash
    node.replay_source["lineage_parent_node_id"] = str(parent_node.id)
    node.replay_source["lineage_parent_code_sha256"] = parent_hash
    node.replay_source["exact_source_match"] = bool(
        source_hash and child_hash == source_hash
    )

    if source_hash and child_hash == source_hash:
        # A byte-identical child can retain exact status (for example a
        # protocol transaction that journals an immutable seed).
        if node.replay_status is None:
            node.replay_status = parent_node.replay_status
        return

    if node.replay_status in _EXACT_REPLAY_STATUSES or node.replay_status is None:
        node.replay_status = "derived_modified_from_exact_source"
    node.replay_source["lineage_kind"] = "modified_descendant"

    # SearchNode lineage is distinct from Result Fact publication.  Result
    # Facts intentionally keep derived_from_refs=[], while this run-scoped
    # node field lets transition/adoption publication bind the real parents.
    if not node.derived_from_refs:
        parent_claims = list(getattr(parent_node, "claim_refs", []) or [])
        if not parent_claims:
            graph_node_id = str(node.replay_source.get("graph_node_id") or "")
            if graph_node_id:
                parent_claims = [
                    f"replay:{graph_node_id}:method_hypothesis"
                ]
        node.derived_from_refs = list(dict.fromkeys(parent_claims))


def refresh_replay_lineage_after_instrumentation(
    node: SearchNode,
    *,
    original_code: str,
    instrumentation_receipt: dict,
) -> None:
    """Recompute direct-replay lineage after deterministic Host instrumentation."""

    if not node.replay_source:
        return
    source_hash = str(node.replay_source.get("code_sha256") or "")
    original_hash = _sha256_code(original_code)
    current_hash = _sha256_code(node.code)
    recorded_hash = str(node.replay_source.get("current_code_sha256") or "")
    if source_hash and original_hash != source_hash:
        if recorded_hash and recorded_hash != original_hash:
            raise ValueError(
                "Host instrumentation input does not match the recorded derived replay candidate"
            )
    node.replay_source["current_code_sha256"] = current_hash
    node.replay_source["exact_source_match"] = bool(
        source_hash and current_hash == source_hash
    )
    if source_hash and current_hash == source_hash:
        return
    node.replay_status = "derived_modified_from_exact_source"
    node.replay_source["lineage_kind"] = "host_instrumented_descendant"
    node.replay_source["host_entrypoint_instrumentation_receipt"] = copy.deepcopy(
        instrumentation_receipt
    )
    graph_node_id = str(node.replay_source.get("graph_node_id") or "")
    if graph_node_id:
        claim_ref = f"replay:{graph_node_id}:method_hypothesis"
        if claim_ref not in node.derived_from_refs:
            node.derived_from_refs.append(claim_ref)


def refresh_replay_lineage_after_revision(
    node: SearchNode,
    *,
    original_code: str,
    revision_kind: str,
) -> None:
    """Record an LLM/reviewer rewrite as a derived replay candidate."""

    if not node.replay_source:
        return
    original_hash = _sha256_code(original_code)
    current_hash = _sha256_code(node.code)
    if original_hash == current_hash:
        return
    recorded_hash = str(node.replay_source.get("current_code_sha256") or "")
    source_hash = str(node.replay_source.get("code_sha256") or "")
    if recorded_hash and recorded_hash not in {original_hash, source_hash}:
        raise ValueError("Replay revision input does not match its recorded lineage")
    node.replay_source["revision_parent_code_sha256"] = original_hash
    node.replay_source["current_code_sha256"] = current_hash
    node.replay_source["exact_source_match"] = bool(
        source_hash and current_hash == source_hash
    )
    node.replay_source["lineage_kind"] = str(revision_kind)
    node.replay_status = "derived_full_runtime_candidate"
    graph_node_id = str(node.replay_source.get("graph_node_id") or "")
    if graph_node_id:
        claim_ref = f"replay:{graph_node_id}:method_hypothesis"
        if claim_ref not in node.derived_from_refs:
            node.derived_from_refs.append(claim_ref)


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
    prospective = getattr(agent, "prospective_audit", None)
    if prospective is not None:
        prospective.bind_thread_to_node(node)

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
        if not node.task_profile:
            node.task_profile = copy.deepcopy(parent_node.task_profile)
        if not node.strategy_candidates:
            node.strategy_candidates = copy.deepcopy(parent_node.strategy_candidates)
        if not node.selected_strategy:
            node.selected_strategy = copy.deepcopy(parent_node.selected_strategy)
        if not node.excluded_method_families:
            node.excluded_method_families = list(parent_node.excluded_method_families)
        if not node.l2_tactic_refs:
            node.l2_tactic_refs = list(parent_node.l2_tactic_refs)
        if not node.strategy_alignment:
            node.strategy_alignment = copy.deepcopy(parent_node.strategy_alignment)
        if not node.protocol_repair:
            node.protocol_repair = copy.deepcopy(parent_node.protocol_repair)
        if node.protocol_repair:
            # Keep the legacy counter frozen for the whole protocol
            # transaction, including stages whose parent audit was already
            # normalized to protocol_stage_complete.
            node.leakage_repair_attempt = parent_node.leakage_repair_attempt
        if not node.replay_source:
            node.replay_source = copy.deepcopy(parent_node.replay_source)
        if node.replay_status is None:
            node.replay_status = parent_node.replay_status
        _update_replay_lineage(parent_node, node)
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
            if not node.protocol_repair:
                node.leakage_repair_attempt = parent_node.leakage_repair_attempt + 1
            node.audit_repair_required = True
            if node.replay_source:
                # `requires_repair` describes the replay lineage. Only the
                # immutable source node is a non-executable repair seed.
                node.replay_source["repair_seed_only"] = False
                node.replay_source["repair_parent_node_id"] = parent_node.id
                if node.replay_source.get("requires_full_runtime_migration") is True:
                    node.replay_source["execution_seed_only"] = False
                    node.replay_source["migration_parent_node_id"] = parent_node.id
                    node.replay_status = "derived_full_runtime_candidate"
                else:
                    node.replay_status = "mandatory_audit_repair"
        if node.branch_id in agent.branch_all_nodes:
            agent.branch_all_nodes[node.branch_id].append(node)

    if node.replay_source:
        adapter = getattr(agent, "evaluation_authority", None)
        bind_replay = getattr(adapter, "record_replay_exposure", None)
        if callable(bind_replay):
            try:
                bind_replay(node)
            except Exception as error:
                # The node may still execute, but no adoption/derivation edge
                # can be published without a successfully bound contract.
                logger.warning(
                    "Failed to bind replay ExperienceContract for node %s: %s",
                    node.id,
                    type(error).__name__,
                )
