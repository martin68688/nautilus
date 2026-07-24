from __future__ import annotations

from .models import DecisionOutcome, Operation, canonical_operation


HIGH_RISK_OPERATIONS = {
    Operation.RANK,
    Operation.SELECT,
    Operation.PROMOTE_RESULT,
    Operation.PUBLISH_ADOPTION,
    Operation.PUBLISH_CAUSAL,
    Operation.PROMOTE,
    Operation.CODE_SEED,
    Operation.DISTILL_DIAGNOSTIC,
    Operation.DISTILL_CANDIDATE,
    Operation.DISTILL_POSITIVE_RESULT,
    Operation.DISTILL_POSITIVE_ADOPTED,
    Operation.DISTILL_POSITIVE,
    Operation.DISTILL,
    Operation.DERIVED_PUBLICATION,
}


def is_high_risk(operation: Operation | str) -> bool:
    return canonical_operation(operation) in HIGH_RISK_OPERATIONS


def failure_outcome(operation: Operation, *, protocol_mismatch: bool, blockers: bool) -> tuple[DecisionOutcome, str]:
    operation = canonical_operation(operation)
    if operation in {
        Operation.INSPECT,
        Operation.DEBUG_HYPOTHESIS,
    }:
        return DecisionOutcome.ALLOW_WITH_WARNING, "retain provenance warning and hide uncertified score"
    if operation == Operation.REPAIR_SEED:
        return DecisionOutcome.ALLOW_WITH_WARNING, "freeze method identity and repair protocol only"
    if protocol_mismatch:
        return DecisionOutcome.REQUIRE_REPLAY, "run a method-preserving clean replay under the active protocol"
    if operation in {
        Operation.PROMOTE_RESULT,
        Operation.PUBLISH_ADOPTION,
        Operation.PUBLISH_CAUSAL,
        Operation.PROMOTE,
        Operation.DERIVED_PUBLICATION,
    }:
        return DecisionOutcome.QUARANTINE, "quarantine the artifact and preserve the audit trail"
    if operation in {
        Operation.DISTILL_DIAGNOSTIC,
        Operation.DISTILL_CANDIDATE,
        Operation.DISTILL_POSITIVE_RESULT,
        Operation.DISTILL_POSITIVE_ADOPTED,
        Operation.DISTILL_POSITIVE,
    }:
        return DecisionOutcome.QUARANTINE, "quarantine the clause until its source obligations are satisfied"
    if blockers:
        return DecisionOutcome.DENY, "resolve contradictory receipts before retrying"
    return DecisionOutcome.DENY, "satisfy the missing evidence obligations"
