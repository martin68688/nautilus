from __future__ import annotations

import hashlib
from typing import Any

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


def score_claim(node: Any, protocol_ref: ProtocolRef, task_id: str) -> Claim:
    ensure_node_authority_fields(node, protocol_ref)
    claim_id = f"node:{node.id}:score"
    metric = getattr(getattr(node, "metric", None), "value", None)
    claim = Claim(
        claim_id=claim_id,
        claim_type=ClaimType.SCORE,
        subject_artifact_id=str(node.id),
        task_scope={"task_id": task_id},
        method_fingerprint=node.method_fingerprint,
        protocol_ref=protocol_ref,
        statement=f"Node {node.id} produced score {metric} under {protocol_ref.key()}",
        parent_claims=list(getattr(node, "derived_from_refs", []) or []),
    )
    if claim_id not in node.claim_refs:
        node.claim_refs.append(claim_id)
    return claim
