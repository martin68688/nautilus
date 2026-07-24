from __future__ import annotations

from typing import Any

from ...models import DecisionStage, Operation
from .runtime import get_authority_adapter


def authorize_promotion(agent: Any, node: Any, *, legacy_allowed: bool, component: str) -> bool:
    """Authorize the current executed result as a new memory subject.

    Historical-experience adoption and causal attribution are separate
    operations.  In particular, a clean result does not need actuation
    receipts merely to exist in RunForest/Overlay memory.
    """
    adapter = get_authority_adapter(agent)
    if adapter is None:
        return legacy_allowed
    return adapter.gate_node(
        node,
        Operation.PROMOTE_RESULT,
        DecisionStage.MEMORY_WRITEBACK,
        component,
        legacy_allowed=legacy_allowed,
    )
