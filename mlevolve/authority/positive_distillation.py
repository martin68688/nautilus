from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .authority_engine import AuthorityEngine
from .evidence_graph import EvidencePath
from .models import (
    AuthorityDecision,
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionStage,
    Operation,
    ProtocolRef,
    Receipt,
    ReceiptType,
    SOPClauseV1,
    TaskContext,
)
from .protocol_registry import canonical_json
from .stage_ontology import GenerationStage, GovernanceStage


class PositiveDistillationKind(str, Enum):
    RESULT = "result"
    ADOPTED = "adopted"


@dataclass(frozen=True)
class PositiveDistillationResult:
    kind: PositiveDistillationKind
    decision: AuthorityDecision
    clause: SOPClauseV1 | None
    derived_claim: Claim | None = None
    evidence_path: EvidencePath | None = None


def _stable_clause_id(kind: PositiveDistillationKind, claim: Claim, text: str) -> str:
    digest = hashlib.sha256(
        canonical_json(
            {
                "kind": kind.value,
                "claim_id": claim.claim_id,
                "artifact_id": claim.subject_artifact_id,
                "text": str(text),
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"clause::positive-{kind.value}::{digest[:24]}"


def _stable_derived_claim_id(
    kind: PositiveDistillationKind,
    claim: Claim,
    text: str,
) -> str:
    digest = hashlib.sha256(
        canonical_json(
            {
                "kind": kind.value,
                "source_claim_id": claim.claim_id,
                "artifact_id": claim.subject_artifact_id,
                "text": str(text),
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"claim::positive-{kind.value}-method::{digest[:24]}"


def authorize_positive_distillation(
    engine: AuthorityEngine,
    *,
    kind: PositiveDistillationKind | str,
    claim: Claim,
    receipts: Iterable[Receipt],
    active_protocol: ProtocolRef,
    task_context: TaskContext,
    text: str,
    sop_id: str,
    clause_id: str = "",
    source_run_ids: Iterable[str] = (),
    source_task_ids: Iterable[str] = (),
    source_task_families: Iterable[str] = (),
    source_domains: Iterable[str] = (),
) -> PositiveDistillationResult:
    """Authorize and materialize one explicitly typed positive SOP clause.

    Result clauses use the current node's SCORE path plus a trusted Derivation
    Receipt. Adopted clauses use an EXPERIENCE_ADOPTION or
    CAUSAL_ATTRIBUTION Claim and therefore require the corresponding
    contract-bound actuation Receipts. Neither path falls back to legacy
    ambiguous ``distill_positive`` semantics.
    """

    kind = (
        kind
        if isinstance(kind, PositiveDistillationKind)
        else PositiveDistillationKind(str(kind))
    )
    if kind == PositiveDistillationKind.RESULT:
        compatible = {
            ClaimType.SCORE,
            ClaimType.PAIRWISE_SUPERIORITY,
            ClaimType.GENERALIZATION,
        }
        operation = Operation.DISTILL_POSITIVE_RESULT
        publication_class = "positive_result"
    else:
        compatible = {
            ClaimType.EXPERIENCE_ADOPTION,
            ClaimType.CAUSAL_ATTRIBUTION,
        }
        operation = Operation.DISTILL_POSITIVE_ADOPTED
        publication_class = "positive_adopted"
    if claim.claim_type not in compatible:
        raise ValueError(
            f"{kind.value} positive distillation cannot consume "
            f"{claim.claim_type.value}"
        )
    if claim.protocol_ref.canonical_hash != active_protocol.canonical_hash:
        raise ValueError("Positive distillation Claim protocol mismatch")

    receipt_rows = list(receipts)
    engine.graph.add_claim(claim)
    for receipt in receipt_rows:
        engine.graph.add_receipt(receipt)
    resolved_clause_id = clause_id or _stable_clause_id(kind, claim, text)
    engine.graph.add_path(
        EvidencePath(
            path_id=(
                f"path:positive-distillation:{resolved_clause_id}:"
                f"{active_protocol.canonical_hash[:12]}"
            ),
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipt_rows],
            required_parent_claims=[],
        )
    )
    decision = engine.authorize(
        AuthorityRequest(
            artifact_id=claim.subject_artifact_id,
            claim_id=claim.claim_id,
            operation=operation,
            decision_stage=DecisionStage.DISTILLATION,
            active_protocol=active_protocol,
            task_context=task_context,
            requesting_component="authority.positive_distillation",
            generation_stage=GenerationStage.EVOLUTION,
            governance_stage=GovernanceStage.DISTILLATION,
        )
    )
    if not decision.allowed:
        return PositiveDistillationResult(kind, decision, None)

    derivation_refs = sorted(
        receipt.receipt_id
        for receipt in receipt_rows
        if receipt.receipt_type == ReceiptType.DERIVATION
    )
    derived_claim = Claim(
        claim_id=_stable_derived_claim_id(kind, claim, text),
        claim_type=ClaimType.METHOD_HYPOTHESIS,
        subject_artifact_id=claim.subject_artifact_id,
        task_scope=dict(claim.task_scope),
        method_fingerprint=claim.method_fingerprint,
        protocol_ref=active_protocol,
        statement=str(text),
        parent_claims=[claim.claim_id],
        source_artifact_refs=[claim.subject_artifact_id],
        evidence_refs=sorted(
            {
                claim.claim_id,
                decision.decision_id,
                *derivation_refs,
            }
        ),
        boundary={
            "positive_distillation_kind": kind.value,
            "source_claim_id": claim.claim_id,
            "distillation_decision_id": decision.decision_id,
            "experience_contract_hash": str(
                (claim.boundary or {}).get("experience_contract_hash") or ""
            ),
            "source_claim_type": claim.claim_type.value,
            "scope_widened": False,
        },
        legacy_status="native_positive_distillation_v1",
    )
    engine.graph.add_claim(derived_claim)
    evidence_path = EvidencePath(
        path_id=(
            f"path:positive-derived:{derived_claim.claim_id}:"
            f"{active_protocol.canonical_hash[:12]}"
        ),
        claim_id=derived_claim.claim_id,
        receipt_ids=sorted(
            {receipt.receipt_id for receipt in receipt_rows}
        ),
        required_parent_claims=[claim.claim_id],
    )
    engine.graph.add_path(evidence_path)
    clause = SOPClauseV1(
        clause_id=resolved_clause_id,
        sop_id=str(sop_id),
        text=str(text),
        retrieval_text=str(text),
        claim_refs=(derived_claim.claim_id,),
        claim_types=(ClaimType.METHOD_HYPOTHESIS.value,),
        source_artifact_refs=(claim.subject_artifact_id,),
        source_run_ids=tuple(sorted({str(value) for value in source_run_ids})),
        source_task_ids=tuple(
            sorted({str(value) for value in source_task_ids})
        ),
        source_task_families=tuple(
            sorted({str(value) for value in source_task_families})
        ),
        source_domains=tuple(
            sorted({str(value) for value in source_domains})
        ),
        protocol_scope=(active_protocol.key(),),
        task_scope=dict(claim.task_scope),
        permitted_operations=(
            Operation.INSPECT.value,
            Operation.GENERATE_CANDIDATE.value,
        ),
        permitted_generation_stages=(
            GenerationStage.DRAFT.value,
            GenerationStage.IMPROVE.value,
        ),
        permitted_governance_stages=(GovernanceStage.RETRIEVAL.value,),
        publication_class=publication_class,
        authority_decision_refs=(decision.decision_id,),
        receipt_refs=tuple(
            sorted({receipt.receipt_id for receipt in receipt_rows})
        ),
        derivation_refs=tuple(derivation_refs),
        contract_spec={
            "positive_distillation_kind": kind.value,
            "source_claim_id": claim.claim_id,
            "derived_claim_id": derived_claim.claim_id,
            "target_artifact_id": claim.subject_artifact_id,
            "distillation_decision_id": decision.decision_id,
            "experience_contract_hash": str(
                (claim.boundary or {}).get("experience_contract_hash") or ""
            ),
            "scope_widened": False,
        },
        legacy_status="native_v1",
    )
    return PositiveDistillationResult(
        kind,
        decision,
        clause,
        derived_claim,
        evidence_path,
    )


__all__ = [
    "PositiveDistillationKind",
    "PositiveDistillationResult",
    "authorize_positive_distillation",
]
