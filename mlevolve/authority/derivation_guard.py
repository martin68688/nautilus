from __future__ import annotations

from dataclasses import dataclass

from .models import (
    AuthorityDecision,
    AuthorityScope,
    DecisionOutcome,
    Operation,
    canonical_operation,
)


@dataclass
class DerivationValidation:
    allowed: bool
    effective_scope: AuthorityScope | None
    reasons: list[str]


@dataclass(frozen=True)
class DerivationOperationDecision:
    outcome: DecisionOutcome
    reasons: list[str]

    @property
    def allowed(self) -> bool:
        return self.outcome in {DecisionOutcome.ALLOW, DecisionOutcome.ALLOW_WITH_WARNING}


def authorize_derivation_operation(
    operation: Operation,
    *,
    parent_claim_refs: list[str],
    clean_ancestry: bool,
    scope_widened: bool = False,
    runtime_actuation_receipts: list[str] | None = None,
    counterfactual_actuation_receipts: list[str] | None = None,
    causal_claim: bool = False,
) -> DerivationOperationDecision:
    """Fail closed for offline distillation/publication mediation.

    Derived publication needs clean, explicit clause parents. Distillation also
    needs positive runtime and counterfactual actuation evidence; clean lineage
    alone never upgrades an attachment into an authoritative SOP.
    """
    operation = canonical_operation(operation)
    if operation not in {
        Operation.DISTILL_DIAGNOSTIC,
        Operation.DISTILL_CANDIDATE,
        Operation.DISTILL_POSITIVE_RESULT,
        Operation.DISTILL_POSITIVE_ADOPTED,
        Operation.DISTILL_POSITIVE,
        Operation.DISTILL,
        Operation.DERIVED_PUBLICATION,
    }:
        raise ValueError(f"unsupported derivation operation: {operation}")
    reasons: list[str] = []
    if not parent_claim_refs:
        reasons.append("missing_parent_claims")
    if operation != Operation.DISTILL_DIAGNOSTIC and not clean_ancestry:
        reasons.append("unclean_ancestry")
    if scope_widened:
        reasons.append("scope_widening")
    if operation in {Operation.DISTILL, Operation.DISTILL_POSITIVE}:
        reasons.append("ambiguous_positive_distillation_semantics")
    if operation == Operation.DISTILL_POSITIVE_ADOPTED:
        if not runtime_actuation_receipts:
            reasons.append("missing_runtime_actuation")
        if causal_claim and not counterfactual_actuation_receipts:
            reasons.append("missing_counterfactual_actuation")
    return DerivationOperationDecision(
        outcome=DecisionOutcome.ALLOW if not reasons else DecisionOutcome.QUARANTINE,
        reasons=reasons,
    )


def _intersection(values: list[list[str]]) -> list[str]:
    if not values:
        return []
    current = set(values[0])
    for group in values[1:]:
        current.intersection_update(group)
    return sorted(current)


def intersect_parent_scopes(decisions: list[AuthorityDecision]) -> AuthorityScope | None:
    scopes = [decision.permitted_scope for decision in decisions if decision.permitted_scope]
    if len(scopes) != len(decisions) or not scopes:
        return None
    return AuthorityScope(
        claim_types=_intersection([scope.claim_types for scope in scopes]),
        operations=_intersection([scope.operations for scope in scopes]),
        stages=_intersection([scope.stages for scope in scopes]),
        protocol_hashes=_intersection([scope.protocol_hashes for scope in scopes]),
        task_ids=_intersection([scope.task_ids for scope in scopes]),
        generation_stages=_intersection([scope.generation_stages for scope in scopes]),
        governance_stages=_intersection([scope.governance_stages for scope in scopes]),
    )


def validate_derivation(
    parent_decisions: list[AuthorityDecision],
    requested_scope: AuthorityScope,
    *,
    has_cycle: bool = False,
) -> DerivationValidation:
    reasons: list[str] = []
    if not parent_decisions:
        reasons.append("missing_parent_claims")
    if has_cycle:
        reasons.append("cyclic_derivation")
    if any(decision.outcome not in {DecisionOutcome.ALLOW, DecisionOutcome.ALLOW_WITH_WARNING} for decision in parent_decisions):
        reasons.append("unauthorized_required_parent")
    effective = intersect_parent_scopes(parent_decisions)
    if effective is None:
        reasons.append("no_parent_scope_intersection")
    else:
        for field in (
            "claim_types",
            "operations",
            "stages",
            "protocol_hashes",
            "task_ids",
            "generation_stages",
            "governance_stages",
        ):
            if not set(getattr(requested_scope, field)).issubset(set(getattr(effective, field))):
                reasons.append(f"scope_widening:{field}")
    return DerivationValidation(not reasons, effective if not reasons else None, reasons)
