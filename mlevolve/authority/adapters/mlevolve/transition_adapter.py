from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, Iterable

from ...claim_decomposer import (
    ClaimBoundaryProposal,
    DecompositionResult,
    decompose_node_claims,
)
from ...models import ProtocolRef


def transition_artifact_id(parent: Any, child: Any) -> str:
    return f"{getattr(parent, 'id', 'root')}->{getattr(child, 'id', 'unknown')}"


def decompose_transition_claims(
    parent: Any,
    child: Any,
    protocol_ref: ProtocolRef,
    task_id: str,
    *,
    proposals: Iterable[ClaimBoundaryProposal] = (),
    legacy_static_only: bool = False,
) -> DecompositionResult:
    transition_id = transition_artifact_id(parent, child)
    code = str(getattr(child, "code", "") or "")
    parent_claims = list(getattr(parent, "claim_refs", []) or [])
    proxy = SimpleNamespace(
        id=transition_id,
        code=code,
        code_summary=getattr(child, "code_summary", ""),
        plan=getattr(child, "plan", ""),
        analysis=getattr(child, "analysis", ""),
        stage=getattr(child, "stage", "unknown"),
        metric=getattr(child, "metric", None),
        exec_time=getattr(child, "exec_time", None),
        leakage_audit=getattr(child, "leakage_audit", None) or {},
        method_fingerprint=str(
            getattr(child, "method_fingerprint", "")
            or hashlib.sha256(code.encode("utf-8")).hexdigest()
        ),
        derived_from_refs=parent_claims,
        claim_refs=[],
    )
    result = decompose_node_claims(
        proxy,
        protocol_ref,
        task_id,
        artifact_kind="transition",
        source_refs=(
            f"node:{getattr(parent, 'id', 'root')}",
            f"node:{getattr(child, 'id', 'unknown')}",
        ),
        proposals=proposals,
        legacy_static_only=legacy_static_only,
    )
    child_refs = getattr(child, "claim_refs", None)
    if isinstance(child_refs, list):
        for claim in result.claims:
            if claim.claim_id not in child_refs:
                child_refs.append(claim.claim_id)
    return result


__all__ = ["decompose_transition_claims", "transition_artifact_id"]
