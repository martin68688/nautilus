"""Deterministic exact execution of verified Replay Research targets."""

from __future__ import annotations

import copy
import logging
from typing import Any

from agents.adoption import log_adoption
from agents.triggers import register_node
from engine.search_node import SearchNode


logger = logging.getLogger("MLEvolve")


def run(agent: Any, anchor_node: SearchNode, target_id: str) -> SearchNode:
    """Create one byte-exact research candidate under the historical anchor.

    The queue admits only targets that already passed task, graph, source hash,
    audit, and CODE_SEED Authority checks in ``load_replay_research_portfolio``.
    No model call or code review is allowed in this step.
    """

    replay = copy.deepcopy(
        dict((getattr(agent, "_replay_research_results", {}) or {}).get(target_id) or {})
    )
    if not replay:
        raise ValueError(f"Replay Research target is not loaded: {target_id}")
    source = dict(replay.get("replay_source") or {})
    if source.get("target_role") != "research":
        raise ValueError(f"Replay Research queue cannot execute non-research target {target_id}")
    if source.get("exact_replay_eligible") is not True:
        raise ValueError(f"Replay Research target is reference-only: {target_id}")
    if replay.get("replay_status") != "historical_exact_research_loaded":
        raise ValueError(
            f"Replay Research target is not exact-executable: {target_id} "
            f"({replay.get('replay_status')})"
        )
    if not (
        anchor_node.replay_source.get("target_role") == "anchor"
        and anchor_node.replay_source.get("exact_replay_execution") is True
        and anchor_node.is_buggy is False
        and anchor_node.is_valid is True
    ):
        raise ValueError("Replay Research exact queue requires a successful anchor")

    source["research_action"] = "replay_exact_diverse"
    source["source_target_ids"] = [target_id]
    source["selected_ids_unchanged"] = True
    source["fusion_weights"] = []
    source["hidden_terminal_labels_used"] = False
    receipt = copy.deepcopy(
        getattr(agent, "_replay_research_portfolio_receipt", {}) or {}
    )
    if receipt:
        source["research_portfolio"] = receipt

    role_contract = {
        "role": "memory_reproduction",
        "research_action": "replay_exact_diverse",
        "target_id": target_id,
        "source_target_ids": [target_id],
        "requirement": (
            "Execute this verified historical source byte-for-byte as an independent "
            "reproduction. Do not fuse, transplant, retune, or use hidden terminal labels."
        ),
        "source": source,
    }
    prompt_record = {
        "schema": "mlevolve_replay_research_exact_execution_v1",
        "target_id": target_id,
        "graph_node_id": source.get("graph_node_id"),
        "code_sha256": source.get("code_sha256"),
        "validation_protocol": source.get("validation_protocol"),
        "metric_authority": source.get("metric_authority"),
        "architecture_signature": source.get("architecture_signature"),
        "allowed_action": "replay_exact_diverse",
        "code_generation": "none_byte_exact_source",
    }

    anchor_node.add_expected_child_count()
    node = SearchNode(
        plan=str(replay.get("plan") or "Verified diverse exact replay."),
        code=str(replay.get("code") or ""),
        parent=anchor_node,
        stage="improve",
        local_best_node=anchor_node.local_best_node or anchor_node,
        draft_role="memory_reproduction",
        role_contract=role_contract,
        source_ref_ids=list(replay.get("source_ref_ids") or []),
        replay_source=source,
        replay_status="historical_exact_research_loaded",
        skip_code_review=True,
    )
    register_node(agent, node, prompt_record, parent_node=anchor_node)
    log_adoption(
        node,
        agent,
        "verified_replay_research_portfolio",
        list(replay.get("source_ref_ids") or []),
        "improve",
        adoption_mode="replay_exact_diverse",
    )
    logger.info(
        "[replay-research] exact target=%s node=%s source=%s sha256=%s",
        target_id,
        node.id,
        source.get("graph_node_id"),
        source.get("code_sha256"),
    )
    return node
