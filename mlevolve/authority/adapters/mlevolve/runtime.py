from __future__ import annotations

import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

from ...authority_engine import AuthorityEngine
from ...evidence_graph import EvidenceGraph, EvidencePath
from ...ledger import AuthorityLedger
from ...models import (
    AuthorityDecision,
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    Operation,
    TaskContext,
)
from ...protocol_registry import canonical_json
from .node_adapter import ensure_node_authority_fields, score_claim
from .protocol_adapter import build_registry
from .receipt_bridge import receipts_for_node, receipts_for_replay_source

logger = logging.getLogger("MLEvolve")


class MLEvolveAuthorityAdapter:
    def __init__(self, agent: Any):
        self.agent = agent
        self.cfg = agent.cfg
        authority_cfg = getattr(self.cfg, "evaluation_authority", None)
        self.mode = str(getattr(authority_cfg, "mode", "off") or "off").lower()
        if self.mode not in {"off", "shadow", "enforce"}:
            raise ValueError(f"Unsupported evaluation_authority.mode: {self.mode}")
        self.fail_closed_high_risk = bool(getattr(authority_cfg, "fail_closed_high_risk", True))
        self.allow_invalid_debug = bool(getattr(authority_cfg, "allow_invalid_debug", True))
        self.emit_snapshot = bool(getattr(authority_cfg, "emit_snapshot", True))
        registry, active = build_registry(self.cfg)
        self.active_protocol = active.ref()
        ledger_path = Path(self.cfg.log_dir) / "authority_events.jsonl"
        self.ledger = AuthorityLedger(ledger_path)
        self.engine = AuthorityEngine(
            registry,
            graph=EvidenceGraph(),
            ledger=self.ledger,
            policy_version=str(getattr(authority_cfg, "policy_version", "authority_v1")),
        )
        self._registered_receipts: set[str] = set()
        if self.mode != "off":
            self.ledger.append("protocol_registered", dataclasses.asdict(active))

    def _emit_snapshots(self) -> None:
        if not self.emit_snapshot or self.mode == "off":
            return
        log_dir = Path(self.cfg.log_dir)
        snapshot = self.engine.snapshot()
        evidence = {
            "claims": snapshot["claims"],
            "receipts": snapshot["receipts"],
            "paths": snapshot["paths"],
        }
        for filename, payload in (
            ("authority_snapshot.json", snapshot),
            ("evidence_graph.json", evidence),
        ):
            target = log_dir / filename
            temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(target)

    @property
    def task_id(self) -> str:
        return str(getattr(self.cfg, "exp_id", "") or "unknown-task")

    @property
    def run_id(self) -> str:
        return str(getattr(self.cfg, "exp_name", "") or "unknown-run")

    def _register_claim_and_receipts(self, claim: Claim, receipts: Iterable[Any]) -> EvidencePath:
        is_new_claim = claim.claim_id not in self.engine.graph.claims
        self.engine.graph.add_claim(claim)
        if is_new_claim and self.mode != "off":
            self.ledger.append("claim_created", dataclasses.asdict(claim))
        receipt_ids: list[str] = []
        for receipt in receipts:
            self.engine.graph.add_receipt(receipt)
            receipt_ids.append(receipt.receipt_id)
            if receipt.receipt_id not in self._registered_receipts and self.mode != "off":
                self.ledger.append("receipt_written", dataclasses.asdict(receipt))
                self._registered_receipts.add(receipt.receipt_id)
        path = EvidencePath(
            path_id=f"path:{claim.claim_id}:{self.active_protocol.canonical_hash[:12]}",
            claim_id=claim.claim_id,
            receipt_ids=receipt_ids,
            required_parent_claims=list(claim.parent_claims),
        )
        self.engine.graph.add_path(path)
        return path

    def authorize_node(
        self,
        node: Any,
        operation: Operation,
        stage: DecisionStage,
        requesting_component: str,
    ) -> AuthorityDecision:
        ensure_node_authority_fields(node, self.active_protocol)
        claim = score_claim(node, self.active_protocol, self.task_id)
        self._register_claim_and_receipts(
            claim,
            receipts_for_node(node, self.active_protocol, self.run_id),
        )
        decision = self.engine.authorize(
            AuthorityRequest(
                artifact_id=str(node.id),
                claim_id=claim.claim_id,
                operation=operation,
                decision_stage=stage,
                active_protocol=self.active_protocol,
                task_context=TaskContext(task_id=self.task_id),
                requesting_component=requesting_component,
            )
        )
        if decision.decision_id not in node.authority_decision_refs:
            node.authority_decision_refs.append(decision.decision_id)
        for receipt_id in self.engine.graph.paths[
            f"path:{claim.claim_id}:{self.active_protocol.canonical_hash[:12]}"
        ].receipt_ids:
            if receipt_id not in node.receipt_refs:
                node.receipt_refs.append(receipt_id)
        self._emit_snapshots()
        return decision

    def authorize_batch_nodes(
        self,
        nodes: list[Any],
        operation: Operation,
        stage: DecisionStage,
        requesting_component: str,
    ) -> list[AuthorityDecision]:
        return [self.authorize_node(node, operation, stage, requesting_component) for node in nodes]

    def permits(self, decision: AuthorityDecision, legacy_allowed: bool = True) -> bool:
        if self.mode in {"off", "shadow"}:
            return legacy_allowed
        if decision.allowed:
            return True
        if decision.operation in {Operation.INSPECT.value, Operation.DEBUG_HYPOTHESIS.value}:
            return self.allow_invalid_debug
        return not self.fail_closed_high_risk

    def gate_node(
        self,
        node: Any,
        operation: Operation,
        stage: DecisionStage,
        requesting_component: str,
        *,
        legacy_allowed: bool,
    ) -> bool:
        if self.mode == "off":
            return legacy_allowed
        decision = self.authorize_node(node, operation, stage, requesting_component)
        allowed = self.permits(decision, legacy_allowed)
        if not allowed:
            logger.warning(
                "Authority denied %s for node %s at %s: outcome=%s missing=%s blockers=%s",
                operation.value,
                getattr(node, "id", "unknown"),
                stage.value,
                decision.outcome.value,
                decision.missing_obligations,
                decision.blocking_receipts,
            )
        return allowed

    def authorize_replay_source(
        self,
        *,
        artifact_id: str,
        code_sha256: str,
        audit: dict[str, Any],
        source_run_id: str,
        repair_seed: bool = False,
    ) -> AuthorityDecision | None:
        if self.mode == "off":
            return None
        claim = Claim(
            claim_id=f"replay:{artifact_id}:score",
            claim_type=ClaimType.SCORE,
            subject_artifact_id=artifact_id,
            task_scope={"task_id": self.task_id},
            method_fingerprint=code_sha256,
            protocol_ref=self.active_protocol,
            statement=f"Historical artifact {artifact_id} is eligible as an exact replay seed",
        )
        self._register_claim_and_receipts(
            claim,
            receipts_for_replay_source(
                artifact_id, code_sha256, audit, self.active_protocol, source_run_id
            ),
        )
        decision = self.engine.authorize(
            AuthorityRequest(
                artifact_id=artifact_id,
                claim_id=claim.claim_id,
                operation=Operation.REPAIR_SEED if repair_seed else Operation.CODE_SEED,
                decision_stage=DecisionStage.REPLAY,
                active_protocol=self.active_protocol,
                task_context=TaskContext(task_id=self.task_id),
                requesting_component="agents.memory.run_forest_replay",
            )
        )
        self._emit_snapshots()
        return decision


def get_authority_adapter(agent: Any) -> MLEvolveAuthorityAdapter | None:
    return getattr(agent, "evaluation_authority", None)
