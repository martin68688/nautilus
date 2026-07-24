from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ...models import (
    AuthorityDecision,
    AuthorityRequest,
    ClaimType,
    DecisionOutcome,
    Operation,
    ReceiptType,
    SOPClauseV1,
    TaskContext,
    VisibilityRequest,
)
from ...domain_scope import canonical_domain, transfer_is_compatible
from ...protocol_compiler import ProtocolCompiler


NAVIGATION_OPERATIONS = {
    Operation.INSPECT,
    Operation.DEBUG_HYPOTHESIS,
}


@dataclass(frozen=True)
class ClauseGateDecision:
    clause_id: str
    allowed: bool
    warning: bool
    reason: str
    authority_decision_refs: tuple[str, ...] = ()
    claim_types: tuple[str, ...] = ()


def _as_dict(value: AuthorityDecision | dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    return value if isinstance(value, dict) else None


def _scope_values(scope: dict[str, Any], key: str) -> set[str]:
    return {str(value) for value in scope.get(key) or []}


def _task_ids(scope: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in (scope.get("task_ids") or [scope.get("task_id")])
        if value not in {None, ""}
    }


def _task_families(scope: dict[str, Any]) -> set[str]:
    return {
        str(value)
        for value in scope.get("task_families") or []
        if value not in {None, ""}
    }


def _is_cross_task_transfer(
    clause: SOPClauseV1,
    request: VisibilityRequest,
) -> bool:
    bound_task_ids = _task_ids(clause.task_scope) | {
        str(value)
        for value in clause.source_task_ids
        if value not in {None, ""}
    }
    return request.task_context.task_id not in bound_task_ids


def _source_authorization_context(
    claim: Any,
    clause: SOPClauseV1,
    request: VisibilityRequest,
) -> TaskContext | None:
    """Bind Authority evaluation to the immutable source Claim scope.

    The target task is authorized separately by the clause transfer gate.  It
    must never be substituted into a source Claim merely to make that Claim
    look cross-task.
    """

    claim_scope = getattr(claim, "task_scope", {}) or {}
    claim_task_ids = _task_ids(claim_scope)
    if request.task_context.task_id in claim_task_ids:
        return request.task_context
    clause_source_ids = {
        str(value)
        for value in clause.source_task_ids
        if value not in {None, ""}
    }
    source_candidates = (
        claim_task_ids & clause_source_ids
        if clause_source_ids
        else claim_task_ids
    )
    if not source_candidates:
        return None

    claim_families = _task_families(claim_scope)
    clause_families = {
        str(value)
        for value in clause.source_task_families
        if value not in {None, ""}
    }
    source_domains = {
        canonical_domain(value) for value in clause.source_domains
    }
    source_domains.discard("")
    family_candidates = claim_families or clause_families
    if source_domains:
        matching_families = {
            value
            for value in family_candidates
            if canonical_domain(value) in source_domains
        }
        if matching_families:
            family_candidates = matching_families
    source_family = sorted(family_candidates)[0] if family_candidates else ""
    return TaskContext(
        task_id=sorted(source_candidates)[0],
        task_family=source_family,
    )


def _has_trusted_method_evidence(
    clause: SOPClauseV1,
    compatible_refs: list[str],
    authority_engine: Any | None,
) -> bool:
    """Require an exact trusted Receipt/path binding for cross-task methods."""

    graph = getattr(authority_engine, "graph", None)
    claims = getattr(graph, "claims", {}) if graph is not None else {}
    receipts = getattr(graph, "receipts", {}) if graph is not None else {}
    paths = getattr(graph, "paths", {}) if graph is not None else {}
    claim_paths = getattr(graph, "claim_paths", {}) if graph is not None else {}
    if len(clause.claim_refs) != 1 or len(compatible_refs) != 1:
        return False
    claim_ref = compatible_refs[0]
    claim = claims.get(claim_ref)
    if claim is None or claim.claim_type != ClaimType.METHOD_HYPOTHESIS:
        return False
    declared_receipts = set(clause.receipt_refs)
    if not declared_receipts:
        return False
    receipt_types: set[ReceiptType] = set()
    for receipt_id in declared_receipts:
        receipt = receipts.get(receipt_id)
        if (
            receipt is None
            or getattr(receipt, "trust_status", "") != "trusted_host"
            or getattr(receipt, "artifact_id", "")
            != claim.subject_artifact_id
        ):
            return False
        receipt_types.add(receipt.receipt_type)
    if not {
        ReceiptType.METHOD_IDENTITY,
        ReceiptType.CODE_EXECUTION,
    } <= receipt_types:
        return False
    for path_id in claim_paths.get(claim_ref, []):
        path = paths.get(path_id)
        if path is None or getattr(path, "claim_id", "") != claim_ref:
            continue
        if set(getattr(path, "receipt_ids", []) or []) == declared_receipts:
            return True
    return False


def _snapshot_matches(
    snapshot: dict[str, Any],
    clause: SOPClauseV1,
    request: VisibilityRequest,
    *,
    claim_types_by_ref: dict[str, str],
    claim_artifacts_by_ref: dict[str, str],
    authorized_task_id: str,
) -> bool:
    outcome = str(snapshot.get("outcome") or "")
    if outcome not in {
        DecisionOutcome.ALLOW.value,
        DecisionOutcome.ALLOW_WITH_WARNING.value,
    }:
        return False
    claim_id = str(snapshot.get("claim_id") or "")
    if claim_id not in set(clause.claim_refs):
        return False
    expected_claim_type = claim_types_by_ref.get(claim_id)
    if not expected_claim_type:
        return False
    expected_artifact = claim_artifacts_by_ref.get(claim_id)
    if expected_artifact and str(snapshot.get("artifact_id") or "") != expected_artifact:
        return False
    if str(snapshot.get("operation") or "") != request.operation.value:
        return False
    if str(snapshot.get("generation_stage") or "") != request.generation_stage.value:
        return False
    if str(snapshot.get("governance_stage") or "") != request.governance_stage.value:
        return False
    if str(snapshot.get("policy_version") or "") != request.authority_policy_version:
        return False
    scope = snapshot.get("permitted_scope")
    if not isinstance(scope, dict):
        return False
    scope_claim_types = _scope_values(scope, "claim_types")
    return bool(
        expected_claim_type in scope_claim_types
        and request.operation.value in _scope_values(scope, "operations")
        and request.generation_stage.value in _scope_values(scope, "generation_stages")
        and request.governance_stage.value in _scope_values(scope, "governance_stages")
        and bool(request.active_protocol.canonical_hash)
        and request.active_protocol.canonical_hash in _scope_values(scope, "protocol_hashes")
        and bool(authorized_task_id)
        and authorized_task_id in _scope_values(scope, "task_ids")
    )


def _publication_compatible(clause: SOPClauseV1, operation: Operation) -> bool:
    publication = str(clause.publication_class or "diagnostic")
    if publication == "diagnostic":
        return operation in {
            Operation.INSPECT,
            Operation.DEBUG_HYPOTHESIS,
            Operation.DISTILL_DIAGNOSTIC,
            Operation.REPAIR_SEED,
        }
    if publication in {"candidate", "provisional"}:
        return operation in {
            Operation.INSPECT,
            Operation.DEBUG_HYPOTHESIS,
            Operation.GENERATE_CANDIDATE,
            Operation.REPAIR_SEED,
            Operation.DISTILL_DIAGNOSTIC,
            Operation.DISTILL_CANDIDATE,
        }
    return publication in {
        "certified",
        "positive",
        "positive_result",
        "positive_adopted",
    }


def _declared_scope_compatible(clause: SOPClauseV1, request: VisibilityRequest) -> bool:
    if clause.permitted_operations and request.operation.value not in clause.permitted_operations:
        return False
    if (
        clause.permitted_generation_stages
        and request.generation_stage.value not in clause.permitted_generation_stages
    ):
        return False
    if (
        clause.permitted_governance_stages
        and request.governance_stage.value not in clause.permitted_governance_stages
    ):
        return False
    if clause.protocol_scope and not clause.protocol_agnostic:
        protocol_keys = {
            request.active_protocol.key(),
            request.active_protocol.canonical_hash,
            f"{request.active_protocol.protocol_id}@{request.active_protocol.version}",
        }
        if not protocol_keys.intersection(clause.protocol_scope):
            return False
    task_ids = _task_ids(clause.task_scope)
    source_task_ids = {
        str(value) for value in clause.source_task_ids if value not in {None, ""}
    } or set(task_ids)
    task_families = _task_families(clause.task_scope)
    source_domains = (
        set(clause.source_domains)
        or {
            canonical_domain(value)
            for value in clause.source_task_families
            if canonical_domain(value)
        }
        or {
            canonical_domain(value)
            for value in task_families
            if canonical_domain(value)
        }
    )
    same_task = request.task_context.task_id in (task_ids | source_task_ids)
    if not same_task:
        if not transfer_is_compatible(
            source_domains,
            request.task_context.task_family,
            clause.transfer_scope,
        ):
            return False
    elif task_families and request.task_context.task_family:
        request_domain = canonical_domain(request.task_context.task_family)
        declared_domains = {
            canonical_domain(value) for value in task_families
        }
        declared_domains.discard("")
        if request_domain and request_domain not in declared_domains:
            return False
    return True


def _claim_types(
    clause: SOPClauseV1,
    authority_engine: Any | None,
) -> tuple[str, ...]:
    values = {str(value) for value in clause.claim_types if value}
    graph = getattr(authority_engine, "graph", None)
    claims = getattr(graph, "claims", {}) if graph is not None else {}
    for claim_ref in clause.claim_refs:
        claim = claims.get(claim_ref)
        if claim is not None:
            values.add(str(claim.claim_type.value))
    return tuple(sorted(values))


def _compatible_claim_refs(
    clause: SOPClauseV1,
    request: VisibilityRequest,
    authority_engine: Any | None,
) -> tuple[list[str], tuple[str, ...], dict[str, str], dict[str, str]]:
    graph = getattr(authority_engine, "graph", None)
    claims = getattr(graph, "claims", {}) if graph is not None else {}
    compatible: list[str] = []
    types = _claim_types(clause, authority_engine)
    declared_types = {
        value for value in clause.claim_types if value in {item.value for item in ClaimType}
    }
    claim_types_by_ref: dict[str, str] = {}
    claim_artifacts_by_ref: dict[str, str] = {}
    for claim_ref in clause.claim_refs:
        claim = claims.get(claim_ref)
        if claim is not None:
            actual_type = claim.claim_type.value
            if declared_types and actual_type not in declared_types:
                continue
            if ProtocolCompiler.claim_operation_compatible(
                claim.claim_type, request.operation
            ):
                compatible.append(claim_ref)
                claim_types_by_ref[claim_ref] = actual_type
                claim_artifacts_by_ref[claim_ref] = claim.subject_artifact_id
            continue
        # A frozen decision snapshot can bind an unresolved Claim only when the
        # clause has one unambiguous Claim ref and one declared Claim type.
        if len(clause.claim_refs) != 1 or len(declared_types) != 1:
            continue
        if len(clause.source_artifact_refs) != 1:
            continue
        declared_type = ClaimType(next(iter(declared_types)))
        if ProtocolCompiler.claim_operation_compatible(declared_type, request.operation):
            compatible.append(claim_ref)
            claim_types_by_ref[claim_ref] = declared_type.value
            claim_artifacts_by_ref[claim_ref] = clause.source_artifact_refs[0]
    return (
        list(dict.fromkeys(compatible)),
        types,
        claim_types_by_ref,
        claim_artifacts_by_ref,
    )


def authorize_clause_for_visibility(
    clause: SOPClauseV1,
    request: VisibilityRequest,
    *,
    authority_engine: Any | None = None,
    decision_lookup: Callable[[str], AuthorityDecision | dict[str, Any] | None] | None = None,
) -> ClauseGateDecision:
    (
        compatible_refs,
        claim_types,
        claim_types_by_ref,
        claim_artifacts_by_ref,
    ) = _compatible_claim_refs(
        clause, request, authority_engine
    )
    is_legacy = str(clause.legacy_status).startswith("legacy")

    if request.operation == Operation.INSPECT:
        return ClauseGateDecision(
            clause.clause_id,
            True,
            is_legacy or not compatible_refs,
            "inspect_navigation_only",
            claim_types=claim_types,
        )
    if not _publication_compatible(clause, request.operation):
        return ClauseGateDecision(
            clause.clause_id, False, False, "publication_class_incompatible", claim_types=claim_types
        )
    if not _declared_scope_compatible(clause, request):
        if is_legacy and request.operation in NAVIGATION_OPERATIONS:
            return ClauseGateDecision(
                clause.clause_id,
                True,
                True,
                "legacy_uncertified_navigation",
                claim_types=claim_types,
            )
        return ClauseGateDecision(
            clause.clause_id, False, False, "declared_scope_mismatch", claim_types=claim_types
        )
    if not compatible_refs:
        if is_legacy and request.operation in NAVIGATION_OPERATIONS and not claim_types:
            return ClauseGateDecision(
                clause.clause_id,
                True,
                True,
                "legacy_uncertified_navigation",
                claim_types=claim_types,
            )
        return ClauseGateDecision(
            clause.clause_id, False, False, "claim_operation_incompatible", claim_types=claim_types
        )

    cross_task_transfer = _is_cross_task_transfer(clause, request)
    if (
        cross_task_transfer
        and request.operation == Operation.GENERATE_CANDIDATE
    ):
        if (
            clause.publication_class
            not in {"candidate", "provisional", "certified"}
            or set(claim_types) != {ClaimType.METHOD_HYPOTHESIS.value}
        ):
            return ClauseGateDecision(
                clause.clause_id,
                False,
                False,
                "cross_task_requires_method_candidate_publication",
                claim_types=claim_types,
            )
        if not _has_trusted_method_evidence(
            clause,
            compatible_refs,
            authority_engine,
        ):
            return ClauseGateDecision(
                clause.clause_id,
                False,
                False,
                "cross_task_requires_trusted_method_evidence",
                claim_types=claim_types,
            )

    decisions: list[
        tuple[AuthorityDecision | dict[str, Any], str]
    ] = []
    graph = getattr(authority_engine, "graph", None)
    claims = getattr(graph, "claims", {}) if graph is not None else {}
    authorization_task_ids: dict[str, str] = {}
    if authority_engine is not None:
        for claim_ref in compatible_refs:
            claim = claims.get(claim_ref)
            if claim is None:
                continue
            authorization_context = request.task_context
            if cross_task_transfer:
                authorization_context = _source_authorization_context(
                    claim,
                    clause,
                    request,
                )
                if authorization_context is None:
                    continue
            authorization_task_ids[claim_ref] = authorization_context.task_id
            decisions.append(
                (
                    authority_engine.authorize(
                        AuthorityRequest(
                            artifact_id=claim.subject_artifact_id,
                            claim_id=claim_ref,
                            operation=request.operation,
                            decision_stage=None,
                            active_protocol=request.active_protocol,
                            task_context=authorization_context,
                            requesting_component=request.requesting_component,
                            generation_stage=request.generation_stage,
                            governance_stage=request.governance_stage,
                        )
                    ),
                    authorization_context.task_id,
                )
            )
    if decision_lookup is not None:
        frozen_task_id = request.task_context.task_id
        if cross_task_transfer:
            source_task_ids = set(authorization_task_ids.values())
            frozen_task_id = (
                next(iter(source_task_ids))
                if len(source_task_ids) == 1
                else ""
            )
        for decision_ref in clause.authority_decision_refs:
            decision = decision_lookup(decision_ref)
            if decision is not None:
                decisions.append((decision, frozen_task_id))

    matched: list[dict[str, Any]] = []
    for decision, authorized_task_id in decisions:
        snapshot = _as_dict(decision)
        if snapshot is not None and _snapshot_matches(
            snapshot,
            clause,
            request,
            claim_types_by_ref=claim_types_by_ref,
            claim_artifacts_by_ref=claim_artifacts_by_ref,
            authorized_task_id=authorized_task_id,
        ):
            matched.append(snapshot)
    if matched:
        refs = tuple(
            str(snapshot.get("decision_id"))
            for snapshot in matched
            if snapshot.get("decision_id")
        )
        warning = any(
            str(snapshot.get("outcome")) == DecisionOutcome.ALLOW_WITH_WARNING.value
            for snapshot in matched
        )
        return ClauseGateDecision(
            clause.clause_id,
            True,
            warning,
            "authority_allow_with_warning" if warning else "authority_allow",
            refs,
            claim_types,
        )

    if request.operation in NAVIGATION_OPERATIONS:
        return ClauseGateDecision(
            clause.clause_id,
            True,
            True,
            "uncertified_navigation_only",
            claim_types=claim_types,
        )
    return ClauseGateDecision(
        clause.clause_id,
        False,
        False,
        "missing_matching_allow_decision",
        claim_types=claim_types,
    )


__all__ = [
    "ClauseGateDecision",
    "NAVIGATION_OPERATIONS",
    "authorize_clause_for_visibility",
]
