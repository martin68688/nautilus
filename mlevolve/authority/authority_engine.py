from __future__ import annotations

import dataclasses
import uuid

from .evidence_graph import EvidenceGraph
from .ledger import AuthorityLedger
from .models import (
    AuthorityDecision,
    AuthorityRequest,
    AuthorityScope,
    DecisionOutcome,
)
from .policy import failure_outcome
from .protocol_compiler import ProtocolCompiler
from .protocol_registry import ProtocolRegistry


def _jsonable(value):
    if dataclasses.is_dataclass(value):
        return {key: _jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


class AuthorityEngine:
    def __init__(
        self,
        registry: ProtocolRegistry,
        graph: EvidenceGraph | None = None,
        ledger: AuthorityLedger | None = None,
        policy_version: str = "authority_v1",
    ):
        self.registry = registry
        self.graph = graph or EvidenceGraph()
        self.ledger = ledger
        self.compiler = ProtocolCompiler(registry)
        self.policy_version = policy_version
        self.decisions: dict[str, AuthorityDecision] = {}

    def authorize(self, request: AuthorityRequest) -> AuthorityDecision:
        claim = self.graph.claims.get(request.claim_id)
        missing: list[str] = []
        blockers: list[str] = []
        satisfied: list[str] = []
        protocol_mismatch = False
        if claim is None:
            missing = ["claim_exists"]
        else:
            obligations = self.compiler.compile(claim, request)
            if obligations.require_protocol_compatibility:
                protocol_mismatch = not self.registry.compatible(claim.protocol_ref, request.active_protocol)
                if protocol_mismatch:
                    missing.append("active_protocol_compatibility")
            evaluation = self.graph.evaluate(request.claim_id, obligations)
            satisfied = evaluation.satisfied_paths
            missing.extend(evaluation.missing_obligations)
            blockers = evaluation.blocking_receipts
        if not missing and not blockers:
            outcome = DecisionOutcome.ALLOW
            required_action = None
            permitted_scope = AuthorityScope(
                claim_types=[claim.claim_type.value],
                operations=[request.operation.value],
                stages=[request.decision_stage.value],
                protocol_hashes=[request.active_protocol.canonical_hash],
                task_ids=[request.task_context.task_id],
            )
        else:
            outcome, required_action = failure_outcome(
                request.operation, protocol_mismatch=protocol_mismatch, blockers=bool(blockers)
            )
            permitted_scope = None
        decision = AuthorityDecision(
            decision_id=uuid.uuid4().hex,
            outcome=outcome,
            permitted_scope=permitted_scope,
            satisfied_paths=satisfied,
            missing_obligations=sorted(set(missing)),
            blocking_receipts=sorted(set(blockers)),
            required_action=required_action,
            policy_version=self.policy_version,
            claim_id=request.claim_id,
            artifact_id=request.artifact_id,
            operation=request.operation.value,
            decision_stage=request.decision_stage.value,
        )
        self.decisions[decision.decision_id] = decision
        if self.ledger:
            self.ledger.append(
                "authority_decision",
                {"request": _jsonable(request), "decision": _jsonable(decision)},
            )
        return decision

    def authorize_batch(self, requests: list[AuthorityRequest]) -> list[AuthorityDecision]:
        return [self.authorize(request) for request in requests]

    def snapshot(self) -> dict:
        return {
            "policy_version": self.policy_version,
            "claims": {key: _jsonable(value) for key, value in sorted(self.graph.claims.items())},
            "receipts": {key: _jsonable(value) for key, value in sorted(self.graph.receipts.items())},
            "paths": {key: _jsonable(value) for key, value in sorted(self.graph.paths.items())},
            "decisions": {key: _jsonable(value) for key, value in sorted(self.decisions.items())},
        }
