from __future__ import annotations

from typing import Any

from ...models import DecisionStage, Operation
from .runtime import get_authority_adapter


def authorize_ranking(agent: Any, node: Any, *, legacy_allowed: bool, component: str) -> bool:
    adapter = get_authority_adapter(agent)
    if adapter is None:
        return legacy_allowed
    return adapter.gate_node(
        node,
        Operation.RANK,
        DecisionStage.BRANCH_SELECTION,
        component,
        legacy_allowed=legacy_allowed,
    )


def authorize_selection(agent: Any, node: Any, *, legacy_allowed: bool, component: str, stage: DecisionStage = DecisionStage.BRANCH_SELECTION) -> bool:
    adapter = get_authority_adapter(agent)
    if adapter is None:
        return legacy_allowed
    return adapter.gate_node(node, Operation.SELECT, stage, component, legacy_allowed=legacy_allowed)


def filter_ranked_nodes(agent: Any, nodes: list[Any], *, component: str) -> list[Any]:
    # Authorization is intentionally performed before sorting so excluded scores
    # cannot affect rank positions or branch quotas.
    from agents.leakage_audit import legacy_rank_eligible

    adapter = get_authority_adapter(agent)
    if adapter is None or adapter.mode == "off":
        return [node for node in nodes if legacy_rank_eligible(agent, node)]
    decisions = adapter.authorize_batch_nodes(
        nodes,
        Operation.RANK,
        DecisionStage.BRANCH_SELECTION,
        component,
    )
    return [
        node
        for node, decision in zip(nodes, decisions)
        if adapter.permits(decision, legacy_allowed=legacy_rank_eligible(agent, node))
    ]
