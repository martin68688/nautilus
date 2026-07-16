from __future__ import annotations

from typing import Any

from ...models import DecisionStage, Operation
from .runtime import get_authority_adapter


def authorize_promotion(agent: Any, node: Any, *, legacy_allowed: bool, component: str) -> bool:
    adapter = get_authority_adapter(agent)
    if adapter is None:
        return legacy_allowed
    return adapter.gate_node(
        node,
        Operation.PROMOTE,
        DecisionStage.MEMORY_WRITEBACK,
        component,
        legacy_allowed=legacy_allowed,
    )
