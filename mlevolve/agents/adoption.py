"""Adoption tracking helper: record which memory entries were injected into a node's prompt.

Side-channel ONLY:
- appends to node.adoption_log AFTER the prompt is sent and the node is registered
- never touches the prompt string
- no-op when adoption tracking is disabled (agent.adoption_tracking_enabled=False) or ref_ids empty

This lets the post-run analyzer (analysis/adoption_tracker.py) correlate "which memory
entries each node actually saw" with "what the generated code does", without polluting the
LLM prompt (memory ids never appear in any prompt text).
"""
import time
import logging
import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.search_node import SearchNode

logger = logging.getLogger("MLEvolve")


def _hybrid_trace_for_ref(pack, ref_id):
    matches = []
    for item in pack.get("navigation_trace", []):
        if not isinstance(item, dict):
            continue
        if (
            item.get("candidate_id") == ref_id
            or item.get("gateway_sop_id") == ref_id
            or ref_id in (item.get("supporting_transition_ids") or [])
            or ref_id in (item.get("expanded_candidate_ids") or [])
        ):
            matches.append(item)
    if matches:
        return matches[-1]
    if ref_id in (pack.get("evidence_refs") or []):
        return {
            "retrieval_channel": "evidence",
            "candidate_class": None,
            "gateway_sop_id": None,
            "supporting_transition_ids": [],
            "selection_reason": "verified evidence attached to an expanded transition",
            "selection_state": "injected",
        }
    if ref_id in (pack.get("failure_patterns") or []):
        return {
            "retrieval_channel": "risk_warning",
            "candidate_class": None,
            "gateway_sop_id": None,
            "supporting_transition_ids": [],
            "selection_reason": "failure pattern injected as warning-only evidence",
            "selection_state": "injected",
        }
    return {
        "retrieval_channel": "hybrid_pack",
        "candidate_class": None,
        "gateway_sop_id": None,
        "supporting_transition_ids": [],
        "selection_reason": "referenced by the injected hybrid pack",
        "selection_state": "injected",
    }


def log_adoption(
    node: "SearchNode",
    agent,
    source: str,
    ref_ids,
    stage: str,
    adoption_mode: str = "prompt_injection",
) -> None:
    """Append adoption records to node.adoption_log.

    Args:
        node: the SearchNode whose prompt had memory injected.
        agent: AgentSearch instance (reads adoption_tracking_enabled).
        source: "methodology" | "global_memory" | "skillgraph" | custom external-memory source.
        ref_ids: list of memory entry ids injected into this node's prompt.
        stage: "draft" | "improve" | "debug".
    """
    layer = getattr(agent, "external_skill_memory", None)
    ref_ids = [ref_id for ref_id in (ref_ids or []) if ref_id]
    pack = {}
    if source == "run_forest_stage_hybrid_memory" and layer is not None:
        getter = getattr(layer, "current_navigation_pack", None)
        if callable(getter):
            pack = getter()
            node.memory_navigation_trace = copy.deepcopy(
                pack.get("navigation_trace", [])
            )
            if pack.get("schema") == "mlevolve_end2end_memory_pack_v1":
                node.memory_routing_trace = {
                    "schema": "mlevolve_memory_routing_trace_v1",
                    "memory_pack_schema": str(pack.get("schema") or ""),
                    "algorithm_version": str(
                        pack.get("algorithm_version") or ""
                    ),
                    "system_id": str(pack.get("system_id") or ""),
                    "stage_route": copy.deepcopy(pack.get("stage_route") or {}),
                    "target_task_id": str(pack.get("target_task_id") or ""),
                    "candidate_pool_hash": str(
                        pack.get("candidate_pool_hash") or ""
                    ),
                    "candidate_pool_source": str(
                        pack.get("candidate_pool_source") or ""
                    ),
                    "raw_pool_observed": bool(pack.get("raw_pool_observed")),
                    "raw_candidates": copy.deepcopy(
                        pack.get("candidate_pool") or []
                    ),
                    "selected_candidates": copy.deepcopy(
                        pack.get("selected_candidates") or []
                    ),
                    "suppressed_candidates": copy.deepcopy(
                        pack.get("suppressed_candidates") or []
                    ),
                    "final_prompt_candidate_ids": list(
                        pack.get("final_prompt_candidate_ids") or []
                    ),
                    "final_prompt_candidates": copy.deepcopy(
                        pack.get("final_prompt_candidates") or []
                    ),
                    "visible_clause_ids": list(
                        pack.get("visible_clause_ids") or []
                    ),
                    "prompt_token_count": int(
                        pack.get("prompt_token_count") or 0
                    ),
                    "prompt_truncated": bool(pack.get("prompt_truncated")),
                    "visibility_safety_gate": copy.deepcopy(
                        pack.get("visibility_safety_gate") or {}
                    ),
                    "unauthorized_prompt_exposure": int(
                        pack.get("unauthorized_prompt_exposure") or 0
                    ),
                    "memory_snapshot_bound_but_not_exposed": bool(
                        pack.get("memory_snapshot_bound_but_not_exposed")
                    ),
                    "memory_bundle": copy.deepcopy(pack.get("memory_bundle") or {}),
                }
    if not ref_ids:
        return
    visibility_pack = (
        getattr(agent, "methodology_visibility_pack", None)
        if source == "methodology"
        else None
    )
    if visibility_pack is None and layer is not None:
        getter = getattr(layer, "current_visibility_pack", None)
        if callable(getter):
            visibility_pack = getter()
    adapter = getattr(agent, "evaluation_authority", None)
    exposure_recorder = getattr(adapter, "record_prompt_exposure", None)
    candidate_exposure_recorder = getattr(
        adapter, "record_memory_candidate_exposure", None
    )
    if (
        pack.get("schema") == "mlevolve_end2end_memory_pack_v1"
        and callable(candidate_exposure_recorder)
    ):
        try:
            candidate_exposure_recorder(
                node=node,
                candidates=pack.get("final_prompt_candidates") or [],
                request_id=str(
                    getattr(visibility_pack, "request_id", "") or ""
                ),
            )
        except Exception as error:
            logger.warning(
                "Failed to record End2End candidate exposure for node %s: %s",
                getattr(node, "id", "unknown"),
                type(error).__name__,
            )
    elif callable(exposure_recorder) and visibility_pack is not None:
        try:
            exposure_recorder(
                node=node,
                visibility_pack=visibility_pack,
                injected_ref_ids=ref_ids,
            )
        except Exception as error:
            # Exposure bookkeeping must never fabricate adoption or silently
            # alter the already-sent prompt. Fail closed for later writeback.
            logger.warning(
                "Failed to record ExperienceContract exposure for node %s: %s",
                getattr(node, "id", "unknown"),
                type(error).__name__,
            )
    if not getattr(agent, "adoption_tracking_enabled", False):
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    for rid in ref_ids:
        trace = _hybrid_trace_for_ref(pack, rid)
        record = {
            "source": source,
            "ref_id": rid,
            "stage": stage,
            "injected_at": ts,
            "adoption_mode": adoption_mode,
            "adoption_outcome": (
                "rejected_after_inspection"
                if adoption_mode == "strategy_candidate_inspection"
                and trace.get("selection_state") == "rejected"
                else "pending_analysis"
            ),
        }
        for key in (
            "retrieval_channel",
            "candidate_class",
            "gateway_sop_id",
            "supporting_transition_ids",
            "selection_reason",
            "selection_state",
        ):
            record[key] = copy.deepcopy(trace.get(key))
        node.adoption_log.append(record)
