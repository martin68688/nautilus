from __future__ import annotations

from .models import DecisionOutcome, Operation


HIGH_RISK_OPERATIONS = {
    Operation.RANK,
    Operation.SELECT,
    Operation.PROMOTE,
    Operation.CODE_SEED,
    Operation.DISTILL,
    Operation.DERIVED_PUBLICATION,
}


def failure_outcome(operation: Operation, *, protocol_mismatch: bool, blockers: bool) -> tuple[DecisionOutcome, str]:
    if operation in {Operation.INSPECT, Operation.DEBUG_HYPOTHESIS}:
        return DecisionOutcome.ALLOW_WITH_WARNING, "retain provenance warning and hide uncertified score"
    if operation == Operation.REPAIR_SEED:
        return DecisionOutcome.ALLOW_WITH_WARNING, "freeze method identity and repair protocol only"
    if protocol_mismatch:
        return DecisionOutcome.REQUIRE_REPLAY, "run a method-preserving clean replay under the active protocol"
    if operation == Operation.PROMOTE or operation == Operation.DERIVED_PUBLICATION:
        return DecisionOutcome.QUARANTINE, "quarantine the artifact and preserve the audit trail"
    if blockers:
        return DecisionOutcome.DENY, "resolve contradictory receipts before retrying"
    return DecisionOutcome.DENY, "satisfy the missing evidence obligations"
