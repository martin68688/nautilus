from __future__ import annotations

import hashlib
from typing import Any

from ...claim_decomposer import (
    ClaimBoundaryProposal,
    DecompositionResult,
    decompose_node_claims,
)
from ...models import Claim, ClaimType, ProtocolRef
from ...replay_certifier import fingerprint_method


def method_fingerprint(code: str) -> str:
    try:
        return fingerprint_method(code).digest()
    except SyntaxError:
        return hashlib.sha256(code.encode("utf-8")).hexdigest()


def ensure_node_authority_fields(node: Any, protocol_ref: ProtocolRef) -> None:
    defaults = {
        "claim_refs": [],
        "receipt_refs": [],
        "authority_decision_refs": [],
        "derived_from_refs": [],
        "protocol_ref": protocol_ref.key(),
        "method_fingerprint": method_fingerprint(str(getattr(node, "code", "") or "")),
    }
    for name, value in defaults.items():
        current = getattr(node, name, None)
        if current in (None, ""):
            setattr(node, name, value)


def claims_for_node(
    node: Any,
    protocol_ref: ProtocolRef,
    task_id: str,
    *,
    proposals: list[ClaimBoundaryProposal] | None = None,
    legacy_static_only: bool = False,
) -> DecompositionResult:
    ensure_node_authority_fields(node, protocol_ref)
    return decompose_node_claims(
        node,
        protocol_ref,
        task_id,
        proposals=proposals or (),
        legacy_static_only=legacy_static_only,
    )


def score_claim(node: Any, protocol_ref: ProtocolRef, task_id: str) -> Claim:
    result = claims_for_node(node, protocol_ref, task_id)
    matches = result.claims_of_type(ClaimType.SCORE)
    if not matches:
        raise ValueError(f"Node {node.id} has no reported score claim")
    return matches[0]
