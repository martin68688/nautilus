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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.search_node import SearchNode

logger = logging.getLogger("MLEvolve")


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
    for rid in ref_ids:
        node.adoption_log.append({
            "source": source,
            "ref_id": rid,
            "stage": stage,
            "injected_at": ts,
            "adoption_mode": adoption_mode,
        })
