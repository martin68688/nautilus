from __future__ import annotations

from dataclasses import dataclass, field

from .models import AuthorityRequest, Claim, ClaimType, Operation, ReceiptType
from .protocol_registry import ProtocolRegistry


@dataclass
class EvidenceObligations:
    required_receipts: set[ReceiptType] = field(default_factory=set)
    minimum_counts: dict[ReceiptType, int] = field(default_factory=dict)
    require_protocol_compatibility: bool = True
    require_clean_ancestry: bool = True
    require_positive_effect: bool = False
    required_payload_flags: dict[ReceiptType, dict[str, object]] = field(default_factory=dict)
    distinct_payload_values: dict[ReceiptType, tuple[str, int]] = field(default_factory=dict)


class ProtocolCompiler:
    HIGH_RISK = {
        Operation.RANK,
        Operation.SELECT,
        Operation.PROMOTE,
        Operation.CODE_SEED,
        Operation.DISTILL,
        Operation.DERIVED_PUBLICATION,
    }

    def __init__(self, registry: ProtocolRegistry):
        self.registry = registry

    def compile(self, claim: Claim, request: AuthorityRequest) -> EvidenceObligations:
        if request.operation in {Operation.INSPECT, Operation.DEBUG_HYPOTHESIS}:
            return EvidenceObligations(require_protocol_compatibility=False, require_clean_ancestry=False)
        if request.operation == Operation.REPAIR_SEED:
            return EvidenceObligations(
                required_receipts={ReceiptType.METHOD_IDENTITY},
                require_protocol_compatibility=False,
                require_clean_ancestry=False,
            )
        required = {ReceiptType.METHOD_IDENTITY}
        payload_flags: dict[ReceiptType, dict[str, object]] = {}
        distinct_values: dict[ReceiptType, tuple[str, int]] = {}
        if claim.claim_type in {ClaimType.EXECUTED, ClaimType.SCORE, ClaimType.PAIRWISE_SUPERIORITY}:
            required.add(ReceiptType.CODE_EXECUTION)
        if claim.claim_type in {ClaimType.SCORE, ClaimType.PAIRWISE_SUPERIORITY, ClaimType.GENERALIZATION}:
            required.update(
                {
                    ReceiptType.SPLIT_LINEAGE,
                    ReceiptType.FIT_SCOPE,
                    ReceiptType.EVALUATOR,
                    ReceiptType.SELECTION_FREEZE,
                }
            )
        minimum: dict[ReceiptType, int] = {}
        if claim.claim_type == ClaimType.PAIRWISE_SUPERIORITY:
            required.update({ReceiptType.SEED_AGGREGATION, ReceiptType.REPLICATION})
            protocol = self.registry.resolve(request.active_protocol)
            minimum[ReceiptType.REPLICATION] = int(protocol.seed_policy.get("pairwise_min_seeds", 3))
            payload_flags[ReceiptType.SEED_AGGREGATION] = {
                "paired": True,
                "preregistered": True,
                "best_seed_selection": False,
            }
            payload_flags[ReceiptType.REPLICATION] = {
                "equal_data_budget": True,
                "equal_compute_budget": True,
                "paired_bootstrap_ci_lower_gt_zero": True,
            }
        if claim.claim_type == ClaimType.GENERALIZATION:
            required.add(ReceiptType.REPLICATION)
            minimum[ReceiptType.REPLICATION] = 2
            distinct_values[ReceiptType.REPLICATION] = ("task_family", 2)
        if claim.claim_type == ClaimType.CAUSAL_ATTRIBUTION or request.operation == Operation.DISTILL:
            required.update({ReceiptType.RUNTIME_ACTUATION, ReceiptType.COUNTERFACTUAL_ACTUATION})
        if request.operation == Operation.DERIVED_PUBLICATION:
            required.add(ReceiptType.DERIVATION)
        return EvidenceObligations(
            required_receipts=required,
            minimum_counts=minimum,
            require_protocol_compatibility=request.operation in self.HIGH_RISK,
            require_clean_ancestry=request.operation in self.HIGH_RISK,
            require_positive_effect=claim.claim_type == ClaimType.CAUSAL_ATTRIBUTION,
            required_payload_flags=payload_flags,
            distinct_payload_values=distinct_values,
        )
