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
    if not getattr(agent, "adoption_tracking_enabled", False):
        return
    if not ref_ids:
        return
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    pack = {}
    layer = getattr(agent, "external_skill_memory", None)
    if source == "run_forest_stage_hybrid_memory" and layer is not None:
        getter = getattr(layer, "current_navigation_pack", None)
        if callable(getter):
            pack = getter()
            node.memory_navigation_trace = copy.deepcopy(pack.get("navigation_trace", []))
    for rid in ref_ids:
        trace = _hybrid_trace_for_ref(pack, rid)
        record = {
            "source": source,
            "ref_id": rid,
            "stage": stage,
            "injected_at": ts,
            "adoption_mode": adoption_mode,
            "adoption_outcome": "pending_analysis",
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
