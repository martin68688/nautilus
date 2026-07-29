from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

from ...authority_engine import AuthorityEngine
from ...actuation import (
    ActuationLevel,
    ActuationTracker,
    ExperienceContract,
    Predicate,
)
from ...claim_decomposer import select_claim_for_operation
from ...collectors import (
    AdoptionPublicationCollector,
    TrustedCollectorHost,
)
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
    ReceiptType,
    TaskContext,
)
from ...policy import is_high_risk
from ...protocol_registry import canonical_json
from ...rollout import (
    AuthorityRolloutController,
    CanaryThresholds,
    RolloutVersionSet,
    evaluate_canary,
)
from ...stage_ontology import GenerationStage, GovernanceStage, resolve_stage_axes
from .node_adapter import claims_for_node, ensure_node_authority_fields
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
        self.collector_version = str(
            getattr(authority_cfg, "collector_version", "1") or "1"
        )
        self.require_bound_bundle = bool(
            getattr(authority_cfg, "require_bound_bundle", False)
        )
        registry, active = build_registry(self.cfg)
        # Keep the exact frozen spec, not only its ref.  Protocol-only repair
        # audits need the spec-owned allowed-change surface; workflow stage
        # names (for example ``data_scope``) are not repair-surface kinds.
        self.protocol_registry = registry
        self.active_protocol_spec = active
        self.active_protocol = active.ref()
        ledger_path = Path(self.cfg.log_dir) / "authority_events.jsonl"
        self.ledger = AuthorityLedger(ledger_path)
        self.engine = AuthorityEngine(
            registry,
            graph=EvidenceGraph(),
            ledger=self.ledger,
            policy_version=str(getattr(authority_cfg, "policy_version", "authority_v1")),
        )
        expected_bundle_id = str(
            getattr(authority_cfg, "expected_bundle_id", "") or "none"
        )
        expected_bundle_manifest_sha256 = str(
            getattr(authority_cfg, "expected_bundle_manifest_sha256", "") or ""
        ).lower()
        self.rollout = AuthorityRolloutController(
            mode=self.mode,
            versions=RolloutVersionSet(
                rollout_id=str(
                    getattr(authority_cfg, "rollout_id", "")
                    or f"{self.run_id}:{self.mode}"
                ),
                policy_version=self.engine.policy_version,
                protocol_ref=self.active_protocol.key(),
                collector_version=self.collector_version,
                bundle_id=expected_bundle_id,
                bundle_manifest_sha256=expected_bundle_manifest_sha256,
            ),
            ledger=self.ledger if self.mode != "off" else None,
            enforce_operations=list(
                getattr(authority_cfg, "enforce_operations", None) or []
            ),
            enforce_generation_stages=list(
                getattr(authority_cfg, "enforce_generation_stages", None) or []
            ),
            enforce_governance_stages=list(
                getattr(authority_cfg, "enforce_governance_stages", None) or []
            ),
        )
        self.canary_thresholds = CanaryThresholds(
            minimum_decisions=int(
                getattr(authority_cfg, "canary_minimum_decisions", 20)
            ),
            max_unauthorized_authority_allows=int(
                getattr(
                    authority_cfg,
                    "canary_max_unauthorized_authority_allows",
                    0,
                )
            ),
            max_false_denial_rate=float(
                getattr(authority_cfg, "canary_max_false_denial_rate", 0.05)
            ),
        )
        self._registered_receipt_events: set[str] = set()
        self.collector_host = TrustedCollectorHost(
            f"mlevolve:{self.run_id}", collector_version=self.collector_version
        )
        self.actuation_tracker = ActuationTracker(
            collector_host=self.collector_host,
            protocol_ref=self.active_protocol,
            run_id=self.run_id,
            ledger=self.ledger if self.mode != "off" else None,
        )
        self._latest_decisions: dict[tuple[str, str], AuthorityDecision] = {}
        self.memory_snapshot = None
        self._overlay_written_artifacts: set[str] = set()
        self._overlay_written_links: set[tuple[str, str, str]] = set()
        if self.mode != "off":
            self.ledger.append("protocol_registered", dataclasses.asdict(active))

    def seal_rollout_versions(self) -> None:
        if self.require_bound_bundle and self.memory_snapshot is None:
            raise RuntimeError(
                "This rollout requires a hash-verified CURRENT Base Bundle before "
                "any authority or visibility decision"
            )
        if self.rollout.versions.policy_version != self.engine.policy_version:
            raise RuntimeError("Authority policy changed inside a rollout")
        if self.rollout.versions.protocol_ref != self.active_protocol.key():
            raise RuntimeError("Active protocol changed inside a rollout")
        if self.rollout.versions.collector_version != self.collector_host.collector_version:
            raise RuntimeError("Trusted collector version changed inside a rollout")
        self.rollout.freeze()

    def _emit_snapshots(self) -> None:
        if not self.emit_snapshot or self.mode == "off":
            return
        log_dir = Path(self.cfg.log_dir)
        snapshot = self.engine.snapshot()
        actuation_snapshot = self.actuation_tracker.snapshot()
        rollout_snapshot = self.rollout.report()
        snapshot["rollout"] = rollout_snapshot
        evidence = {
            "claims": snapshot["claims"],
            "receipts": snapshot["receipts"],
            "paths": snapshot["paths"],
        }
        for filename, payload in (
            ("authority_snapshot.json", snapshot),
            ("evidence_graph.json", evidence),
            ("actuation_reports.json", actuation_snapshot),
            ("authority_rollout_report.json", rollout_snapshot),
        ):
            target = log_dir / filename
            temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(target)

    def _sealed_authority_snapshot_pointer(self) -> dict[str, str]:
        """Return an immutable, content-addressed Authority snapshot pointer."""

        self._emit_snapshots()
        source = Path(self.cfg.log_dir) / "authority_snapshot.json"
        if not source.is_file():
            return {}
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        directory = source.parent / "authority_snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"authority_snapshot.{digest}.json"
        try:
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            if target.read_bytes() != payload:
                raise ValueError("Content-addressed Authority snapshot collision")
        return {
            "schema": "authority_snapshot_pointer_v1",
            "path": str(target.resolve()),
            "sha256": digest,
        }

    def configure_memory_snapshot(self, snapshot: Any) -> None:
        snapshot.assert_unchanged()
        if snapshot.active_protocol_ref != self.active_protocol.key():
            raise ValueError("MemorySnapshot protocol does not match Authority adapter")
        if snapshot.authority_policy_version != self.engine.policy_version:
            raise ValueError("MemorySnapshot policy does not match Authority adapter")
        self.rollout.bind_bundle(
            bundle_id=snapshot.base_bundle.bundle_id,
            manifest_sha256=snapshot.base_bundle.manifest_sha256,
        )
        from ...bundle_authority import load_snapshot_authority

        authority_load_report = load_snapshot_authority(self.engine, snapshot)
        self.memory_snapshot = snapshot
        if self.mode != "off":
            self.ledger.append("memory_bundle_authority_loaded", authority_load_report)

    def terminal_writeback_descriptor(self) -> dict[str, Any]:
        """Seal the host-only inputs needed by a post-run terminal scorer."""

        self._emit_snapshots()
        snapshot_path = Path(self.cfg.log_dir) / "authority_snapshot.json"
        ledger_events = self.ledger.read()
        overlay_manifest = (
            self.memory_snapshot.session_overlay.manifest
            if self.memory_snapshot is not None
            else {}
        )
        payload: dict[str, Any] = {
            "schema": "fixed_holdout_authority_writeback_descriptor_v1",
            "status": (
                "ready"
                if self.memory_snapshot is not None and snapshot_path.is_file()
                else "writeback_incomplete"
            ),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "authority_mode": self.mode,
            "policy_version": self.engine.policy_version,
            "collector_version": self.collector_version,
            "active_protocol": dataclasses.asdict(self.active_protocol),
            "protocol_registry_path": str(
                self.protocol_registry.registry_dir or ""
            ),
            "authority_snapshot_path": str(snapshot_path.resolve()),
            "authority_snapshot_sha256": (
                hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
                if snapshot_path.is_file()
                else ""
            ),
            "authority_ledger_path": str(self.ledger.path.resolve()),
            "authority_ledger_event_count": len(ledger_events),
            "authority_ledger_last_event_hash": (
                str(ledger_events[-1]["event_hash"]) if ledger_events else ""
            ),
            "session_overlay_path": (
                str(self.memory_snapshot.session_overlay.path)
                if self.memory_snapshot is not None
                else ""
            ),
            "session_overlay_id": (
                self.memory_snapshot.session_overlay.overlay_id
                if self.memory_snapshot is not None
                else ""
            ),
            "session_overlay_manifest_sha256": str(
                overlay_manifest.get("manifest_sha256") or ""
            ),
            "bundle_id": (
                self.memory_snapshot.base_bundle.bundle_id
                if self.memory_snapshot is not None
                else ""
            ),
            "bundle_manifest_sha256": (
                self.memory_snapshot.base_bundle.manifest_sha256
                if self.memory_snapshot is not None
                else ""
            ),
            "rollout_version_hash": self.rollout.versions.version_hash,
            "descriptor_hash": "",
        }
        payload["descriptor_hash"] = hashlib.sha256(
            canonical_json(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "descriptor_hash"
                }
            ).encode("utf-8")
        ).hexdigest()
        return payload

    @property
    def task_id(self) -> str:
        return str(getattr(self.cfg, "exp_id", "") or "unknown-task")

    @property
    def run_id(self) -> str:
        return str(getattr(self.cfg, "exp_name", "") or "unknown-run")

    def _register_claim_and_receipts(
        self,
        claim: Claim,
        receipts: Iterable[Any],
        *,
        required_parent_claims: Iterable[str] | None = None,
    ) -> EvidencePath:
        is_new_claim = claim.claim_id not in self.engine.graph.claims
        self.engine.graph.add_claim(claim)
        if is_new_claim and self.mode != "off":
            self.ledger.append("claim_created", dataclasses.asdict(claim))
        receipt_ids: list[str] = []
        for receipt in receipts:
            self.engine.graph.add_receipt(receipt)
            receipt_ids.append(receipt.receipt_id)
            event_key = receipt.event_hash or receipt.receipt_id
            if event_key not in self._registered_receipt_events and self.mode != "off":
                self.ledger.append("receipt_written", dataclasses.asdict(receipt))
                self._registered_receipt_events.add(event_key)
        path = EvidencePath(
            path_id=f"path:{claim.claim_id}:{self.active_protocol.canonical_hash[:12]}",
            claim_id=claim.claim_id,
            receipt_ids=receipt_ids,
            required_parent_claims=list(
                claim.parent_claims
                if required_parent_claims is None
                else required_parent_claims
            ),
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
        self.seal_rollout_versions()
        try:
            return self._authorize_node(node, operation, stage, requesting_component)
        except Exception as error:
            fallback_axes = resolve_stage_axes(legacy_stage=stage)
            request = AuthorityRequest(
                artifact_id=str(getattr(node, "id", "unknown")),
                claim_id=f"node:{getattr(node, 'id', 'unknown')}:score",
                operation=operation,
                decision_stage=stage,
                active_protocol=self.active_protocol,
                task_context=TaskContext(task_id=self.task_id),
                requesting_component=requesting_component,
                generation_stage=fallback_axes.generation_stage,
                governance_stage=fallback_axes.governance_stage,
            )
            decision = self.engine._internal_error_decision(request, error)
            self._latest_decisions[(request.artifact_id, request.operation.value)] = decision
            refs = getattr(node, "authority_decision_refs", None)
            if isinstance(refs, list) and decision.decision_id not in refs:
                refs.append(decision.decision_id)
            log = logger.error if is_high_risk(operation) else logger.warning
            log(
                "Authority internal error returned a fail-safe decision for %s on node %s: %s: %s",
                operation.value,
                getattr(node, "id", "unknown"),
                type(error).__name__,
                decision.diagnostics.get("error_message", ""),
            )
            self._emit_snapshots()
            return decision

    def _authorize_node(
        self,
        node: Any,
        operation: Operation,
        stage: DecisionStage,
        requesting_component: str,
    ) -> AuthorityDecision:
        ensure_node_authority_fields(node, self.active_protocol)
        decomposition = claims_for_node(
            node,
            self.active_protocol,
            self.task_id,
        )
        receipts = receipts_for_node(
            node,
            self.active_protocol,
            self.run_id,
            collector_host=self.collector_host,
            task_id=self.task_id,
        )
        receipts.extend(
            self.actuation_tracker.receipts_for_artifact(str(node.id))
        )
        for decomposed_claim in decomposition.claims:
            self._register_claim_and_receipts(decomposed_claim, receipts)
        try:
            claim = select_claim_for_operation(decomposition, operation)
        except ValueError:
            # A failed execution can legitimately have no SCORE claim while
            # legacy control flow still asks whether it may be ranked or
            # promoted.  Missing evidence is a normal fail-closed policy
            # result, not an Authority subsystem failure.  Point the request
            # at the deterministic claim that would have existed so the
            # engine returns ``claim_exists`` without inventing evidence.
            claim = None
        missing_claim_id = (
            f"node:{node.id}:score"
            if operation in {
                Operation.RANK,
                Operation.SELECT,
                Operation.PROMOTE_RESULT,
                Operation.PROMOTE,
            }
            else f"node:{node.id}:claim:missing:{operation.value}"
        )
        axes = resolve_stage_axes(runtime_stage=getattr(node, "stage", None), legacy_stage=stage)
        request = AuthorityRequest(
            artifact_id=str(node.id),
            claim_id=claim.claim_id if claim is not None else missing_claim_id,
            operation=operation,
            decision_stage=stage,
            active_protocol=self.active_protocol,
            task_context=TaskContext(task_id=self.task_id),
            requesting_component=requesting_component,
            generation_stage=axes.generation_stage,
            governance_stage=axes.governance_stage,
        )
        decision = self.engine.authorize(request)
        self._latest_decisions[(str(node.id), operation.value)] = decision
        if claim is not None:
            for receipt_id in self.engine.graph.paths[
                f"path:{claim.claim_id}:{self.active_protocol.canonical_hash[:12]}"
            ].receipt_ids:
                if receipt_id not in node.receipt_refs:
                    node.receipt_refs.append(receipt_id)
        self._emit_snapshots()
        if decision.decision_id not in node.authority_decision_refs:
            node.authority_decision_refs.append(decision.decision_id)
        return decision

    def record_prompt_exposure(
        self,
        *,
        node: Any,
        visibility_pack: Any,
        injected_ref_ids: Iterable[str],
    ) -> list[str]:
        """Bind only contracts whose SOP/clause was actually prompt-injected."""

        if self.mode == "off" or visibility_pack is None:
            return []
        refs = {str(value) for value in injected_ref_ids if value}
        contracts = []
        for payload in getattr(visibility_pack, "experience_contracts", []) or []:
            contract = ExperienceContract.from_dict(payload)
            if contract.clause_id in refs or contract.sop_id in refs:
                contracts.append(contract)
        if not contracts:
            return []
        prompt_text = str(getattr(node, "prompt_input", "") or "")
        prompt_sha256 = (
            hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            if prompt_text
            else ""
        )
        contract_ids = self.actuation_tracker.record_exposure(
            artifact_id=str(node.id),
            contracts=contracts,
            request_id=str(getattr(visibility_pack, "request_id", "") or ""),
            prompt_sha256=prompt_sha256,
        )
        refs_field = getattr(node, "experience_contract_refs", None)
        if isinstance(refs_field, list):
            for contract_id in contract_ids:
                if contract_id not in refs_field:
                    refs_field.append(contract_id)
        self._emit_snapshots()
        return contract_ids

    def record_replay_exposure(self, node: Any) -> list[str]:
        """Bind a direct replay/code-seed delivery to a real ExperienceContract.

        This is separate from normal Prompt visibility because an exact replay
        is delivered as executable source rather than prose.  The contract can
        reach L3 only after deterministic lineage checks and trusted execution
        observations; merely inheriting replay metadata never publishes an
        adoption edge.
        """

        if self.mode == "off" or not getattr(node, "replay_source", None):
            return []
        source = dict(node.replay_source)
        graph_node_id = str(source.get("graph_node_id") or "")
        claim_id = f"replay:{graph_node_id}:method_hypothesis"
        if not graph_node_id or claim_id not in self.engine.graph.claims:
            raise ValueError("Replay source Claim is not registered")
        replay_operation = (
            Operation.REPAIR_SEED
            if bool(source.get("requires_repair"))
            else Operation.CODE_SEED
        )
        source_decision = self._latest_decisions.get(
            (graph_node_id, replay_operation.value)
        )
        if source_decision is None or not source_decision.allowed:
            # Shadow-mode legacy control flow may still inspect a denied seed,
            # but a denied/missing source decision cannot mint an adoption
            # contract or later establish certified experience lineage.
            if self.mode != "off":
                self.ledger.append(
                    "replay_experience_contract_suppressed",
                    {
                        "artifact_id": str(node.id),
                        "source_artifact_id": graph_node_id,
                        "operation": replay_operation.value,
                        "decision_id": (
                            source_decision.decision_id
                            if source_decision is not None
                            else ""
                        ),
                        "reason": "source_replay_authority_not_allowed",
                    },
                )
            return []
        axes = resolve_stage_axes(runtime_stage=getattr(node, "stage", None))
        source_hash = str(source.get("code_sha256") or "")
        clause_id = f"replay_clause::{hashlib.sha256(graph_node_id.encode('utf-8')).hexdigest()[:24]}"
        contract = ExperienceContract(
            preconditions=[
                Predicate(
                    "source_replay_claim_authorized",
                    True,
                    "The immutable replay source Claim passed CODE_SEED Authority.",
                )
            ],
            must_preserve=[
                Predicate("active_protocol_ref", self.active_protocol.key()),
                Predicate("task_id", self.task_id),
                Predicate("replay_lineage_preserved", True),
            ],
            must_change=[
                Predicate(
                    f"clause_applied::{clause_id}",
                    True,
                    "The admitted replay method is realized by the target artifact.",
                )
            ],
            must_not_use=[
                Predicate("forbidden_dependency_count", 0),
                Predicate("holdout_used_for_selection", False),
            ],
            expected_runtime_observations=[
                Predicate("target_path_executed", True)
            ],
            clause_id=clause_id,
            sop_id=(str((source.get("sop_ids") or [""])[0]) or f"replay_sop::{graph_node_id}"),
            claim_refs=[claim_id],
            source_artifact_refs=[graph_node_id],
            source_run_ids=[str(source.get("run_id") or "")],
            source_task_ids=[str(source.get("task_id") or self.task_id)],
            task_scope={"task_id": self.task_id},
            active_protocol_ref=self.active_protocol.key(),
            target_task_id=self.task_id,
            operation=replay_operation.value,
            generation_stage=axes.generation_stage.value,
            governance_stage=GovernanceStage.REPLAY.value,
            publication_class="certified",
            minimum_writeback_level=int(ActuationLevel.RUNTIME_CONFORMANT),
            policy_version=self.engine.policy_version,
        ).finalize()
        existing_contracts = {
            value.contract_id
            for value in self.actuation_tracker.contracts_for_artifact(
                str(node.id)
            )
        }
        if contract.contract_id in existing_contracts:
            contract_ids = [contract.contract_id]
            refs_field = getattr(node, "experience_contract_refs", None)
            if not isinstance(refs_field, list):
                refs_field = []
                setattr(node, "experience_contract_refs", refs_field)
            for contract_id in contract_ids:
                if contract_id not in refs_field:
                    refs_field.append(contract_id)
            source["experience_contract_ids"] = contract_ids
            source["source_code_sha256"] = source_hash
            node.replay_source = source
            return contract_ids
        prompt_text = str(getattr(node, "prompt_input", "") or "")
        contract_ids = self.actuation_tracker.record_exposure(
            artifact_id=str(node.id),
            contracts=[contract],
            request_id=f"direct_replay::{node.id}",
            prompt_sha256=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        )
        refs_field = getattr(node, "experience_contract_refs", None)
        if not isinstance(refs_field, list):
            refs_field = []
            setattr(node, "experience_contract_refs", refs_field)
        for contract_id in contract_ids:
            if contract_id not in refs_field:
                refs_field.append(contract_id)
        source["experience_contract_ids"] = list(contract_ids)
        source["source_code_sha256"] = source_hash
        node.replay_source = source
        self._emit_snapshots()
        return contract_ids

    @staticmethod
    def _replay_lineage_preserved(node: Any) -> bool:
        source = dict(getattr(node, "replay_source", None) or {})
        source_hash = str(source.get("code_sha256") or "")
        code = str(getattr(node, "code", "") or "")
        if source_hash and hashlib.sha256(code.encode("utf-8")).hexdigest() == source_hash:
            return True
        instrumentation = dict(
            source.get("host_entrypoint_instrumentation_receipt") or {}
        )
        if instrumentation:
            receipt_hash = str(instrumentation.get("receipt_hash") or "")
            receipt_core = {
                key: value
                for key, value in instrumentation.items()
                if key != "receipt_hash"
            }
            expected_receipt_hash = hashlib.sha256(
                canonical_json(receipt_core).encode("utf-8")
            ).hexdigest()
            if (
                instrumentation.get("schema")
                == "mlevolve_preflight_repair_receipt_v1"
                and instrumentation.get("repair_kind") == "instrumentation"
                and instrumentation.get("method_identity_preserved") is True
                and instrumentation.get("terminal_score_observed") is False
                and instrumentation.get("runtime_receipt_fabricated") is False
                and instrumentation.get("original_code_sha256") == source_hash
                and instrumentation.get("repaired_code_sha256")
                == hashlib.sha256(code.encode("utf-8")).hexdigest()
                and receipt_hash == expected_receipt_hash
            ):
                return True
        if str(getattr(node, "replay_status", "") or "") in {
            "mandatory_audit_repair_executed_clean",
            "staged_protocol_repair_executed_clean",
        }:
            return True
        ancestor = getattr(node, "parent", None)
        while ancestor is not None:
            ancestor_code = str(getattr(ancestor, "code", "") or "")
            if source_hash and hashlib.sha256(ancestor_code.encode("utf-8")).hexdigest() == source_hash:
                try:
                    from agents.leakage_audit import (
                        audit_repair_preservation,
                        build_repair_preservation_contract,
                    )

                    audit = audit_repair_preservation(
                        code,
                        build_repair_preservation_contract(ancestor_code),
                    )
                    return audit.get("status") == "clean"
                except Exception:
                    return False
            ancestor = getattr(ancestor, "parent", None)
        return False

    def finalize_production_actuation(self, node: Any) -> list[dict[str, Any]]:
        """Advance only host-verifiable production adoptions and publish L3 edges."""

        if self.mode == "off":
            return []
        artifact_id = str(node.id)
        contracts = self.actuation_tracker.contracts_for_artifact(artifact_id)
        if not contracts:
            return []

        selected = dict(getattr(node, "selected_strategy", None) or {})
        selected_sop = str(selected.get("sop_id") or "")
        alignment = dict(getattr(node, "strategy_alignment", None) or {})
        replay_preserved = self._replay_lineage_preserved(node)
        host_receipts = receipts_for_node(
            node,
            self.active_protocol,
            self.run_id,
            collector_host=self.collector_host,
            task_id=self.task_id,
        )
        receipt_types = {
            receipt.receipt_type
            for receipt in host_receipts
            if getattr(receipt, "trust_status", "") == "trusted_host"
        }
        protocol_receipts = {
            ReceiptType.SPLIT_LINEAGE,
            ReceiptType.FIT_SCOPE,
            ReceiptType.PREDICTION_SCOPE,
            ReceiptType.EVALUATOR,
            ReceiptType.SELECTION_FREEZE,
        }
        protocol_clean = protocol_receipts <= receipt_types
        execution_observed = bool(
            ReceiptType.CODE_EXECUTION in receipt_types
            and getattr(node, "is_buggy", True) is False
            and getattr(node, "is_valid", True) is not False
        )

        for contract in contracts:
            direct_replay = contract.clause_id.startswith("replay_clause::")
            selected_strategy = bool(
                selected_sop
                and contract.sop_id == selected_sop
                and alignment.get("status") == "verified"
            )
            admitted = replay_preserved if direct_replay else selected_strategy
            if not admitted:
                continue
            report = self.actuation_tracker.report(
                artifact_id=artifact_id,
                contract_id=contract.contract_id,
            )
            if not report.reached(ActuationLevel.CLAIMED_ADOPTION):
                self.actuation_tracker.record_claimed_adoption(
                    artifact_id=artifact_id,
                    contract_id=contract.contract_id,
                )

            supplied = dict(
                (
                    getattr(node, "experience_actuation_observations", None)
                    or {}
                ).get(contract.contract_id)
                or {}
            )
            preconditions = dict(supplied.get("preconditions") or {})
            if direct_replay:
                preconditions["source_replay_claim_authorized"] = True
            static: dict[str, Any] = {
                "active_protocol_ref": self.active_protocol.key(),
                "task_id": self.task_id,
                f"clause_applied::{contract.clause_id}": True,
            }
            static.update(dict(supplied.get("static") or {}))
            if direct_replay:
                static["replay_lineage_preserved"] = replay_preserved
            if protocol_clean:
                static.update(
                    {
                        "forbidden_dependency_count": 0,
                        "holdout_used_for_selection": False,
                    }
                )
            report = self.actuation_tracker.report(
                artifact_id=artifact_id,
                contract_id=contract.contract_id,
            )
            if not report.reached(ActuationLevel.STATIC_CONFORMANT):
                static_receipt = self.actuation_tracker.record_static_observation(
                    artifact_id=artifact_id,
                    contract_id=contract.contract_id,
                    preconditions=preconditions,
                    observations=static,
                    source="host.production_static_actuation",
                )
                if static_receipt is None:
                    continue
            if not execution_observed:
                continue
            runtime = dict(supplied.get("runtime") or {})
            runtime["target_path_executed"] = True
            report = self.actuation_tracker.report(
                artifact_id=artifact_id,
                contract_id=contract.contract_id,
            )
            if not report.reached(ActuationLevel.RUNTIME_CONFORMANT):
                runtime_receipt = self.actuation_tracker.record_runtime_observation(
                    artifact_id=artifact_id,
                    contract_id=contract.contract_id,
                    observations=runtime,
                    source="host.production_runtime_actuation",
                )
                if runtime_receipt is None:
                    continue
            report = self.actuation_tracker.report(
                artifact_id=artifact_id,
                contract_id=contract.contract_id,
            )
            if not report.reached(ActuationLevel.RUNTIME_CONFORMANT):
                continue
            decision = self.authorize_experience_link(
                node,
                contract_id=contract.contract_id,
            )
            if decision.allowed:
                self.append_authorized_experience_link(
                    node,
                    contract_id=contract.contract_id,
                )
                for claim_ref in contract.claim_refs:
                    if claim_ref not in node.derived_from_refs:
                        node.derived_from_refs.append(claim_ref)

        reports = self.actuation_reports_for_node(node)
        return reports

    def claim_experience_adoption(self, node: Any, contract_id: str) -> None:
        self.actuation_tracker.record_claimed_adoption(
            artifact_id=str(node.id), contract_id=str(contract_id)
        )
        self._emit_snapshots()

    def record_static_actuation(
        self,
        node: Any,
        contract_id: str,
        *,
        preconditions: dict[str, Any],
        observations: dict[str, Any],
    ):
        receipt = self.actuation_tracker.record_static_observation(
            artifact_id=str(node.id),
            contract_id=str(contract_id),
            preconditions=preconditions,
            observations=observations,
        )
        self._emit_snapshots()
        return receipt

    def record_runtime_actuation(
        self,
        node: Any,
        contract_id: str,
        *,
        observations: dict[str, Any],
    ):
        receipt = self.actuation_tracker.record_runtime_observation(
            artifact_id=str(node.id),
            contract_id=str(contract_id),
            observations=observations,
        )
        self._emit_snapshots()
        return receipt

    def record_counterfactual_actuation(
        self,
        node: Any,
        contract_id: str,
        *,
        pair_result: Any,
    ):
        receipt = self.actuation_tracker.record_counterfactual(
            artifact_id=str(node.id),
            contract_id=str(contract_id),
            pair_result=pair_result,
        )
        self._emit_snapshots()
        return receipt

    def record_prospective_counterfactual(
        self,
        node: Any,
        *,
        pair_result: dict[str, Any],
    ):
        """Mint a Host-owned observer receipt without executing the other arm."""
        from ...collectors import CounterfactualObservationCollector

        receipt = self.collector_host.collect(
            CounterfactualObservationCollector,
            artifact_id=str(node.id),
            run_id=self.run_id,
            protocol_ref=self.active_protocol,
            source="host.prospective_counterfactual_observer",
            payload=dict(pair_result),
        )
        self.engine.graph.add_receipt(receipt)
        if self.mode != "off":
            self.ledger.append("receipt_written", dataclasses.asdict(receipt))
        self._emit_snapshots()
        return receipt

    def register_clean_replay(
        self,
        *,
        original_claim_id: str,
        verification: Any,
        receipts: Iterable[Any],
        statement: str,
        claim_type: ClaimType | None = None,
        task_scope: dict[str, Any] | None = None,
    ):
        """Ingest a host-verified replay as a new immutable Claim/path."""

        from ...clean_replay import ReplayAuthorityRecovery

        registration = ReplayAuthorityRecovery(
            self.engine.graph, self.engine.registry
        ).register(
            original_claim_id=str(original_claim_id),
            verification=verification,
            receipts=receipts,
            protocol_ref=self.active_protocol,
            statement=str(statement),
            claim_type=claim_type,
            task_scope=task_scope or {"task_id": self.task_id},
        )
        if self.mode != "off":
            self.ledger.append("clean_replay_registered", registration.as_dict())
        self._emit_snapshots()
        return registration

    def actuation_reports_for_node(self, node: Any) -> list[dict[str, Any]]:
        reports = self.actuation_tracker.reports_for_artifact(str(node.id))
        report_refs = getattr(node, "actuation_report_refs", None)
        if isinstance(report_refs, list):
            for report in reports:
                if report.report_id not in report_refs:
                    report_refs.append(report.report_id)
        self._emit_snapshots()
        return [report.as_dict() for report in reports]

    def authorize_experience_link(
        self,
        node: Any,
        *,
        contract_id: str,
        causal: bool = False,
        requesting_component: str = "authority.runtime.experience_link",
    ) -> AuthorityDecision:
        """Authorize one contract-bound adoption or causal lineage edge.

        The link Claim is distinct from the node's SCORE Claim.  Its immutable
        identity binds the source clause claims, target artifact, contract
        hash, and actuation report.  This prevents an L3/L4 Receipt from one
        injected experience from authorizing a different experience edge.
        """

        self.seal_rollout_versions()
        artifact_id = str(node.id)
        operation = (
            Operation.PUBLISH_CAUSAL
            if causal
            else Operation.PUBLISH_ADOPTION
        )
        report = self.actuation_tracker.report(
            artifact_id=artifact_id,
            contract_id=str(contract_id),
        )
        claim_type = (
            ClaimType.CAUSAL_ATTRIBUTION
            if causal
            else ClaimType.EXPERIENCE_ADOPTION
        )
        claim_id = (
            f"node:{artifact_id}:{claim_type.value}:"
            f"{report.contract_hash[:24]}"
        )
        axes = resolve_stage_axes(
            runtime_stage=getattr(node, "stage", None),
            legacy_stage=DecisionStage.MEMORY_WRITEBACK,
        )
        source_claims_present = bool(report.claim_refs) and all(
            claim_ref in self.engine.graph.claims
            for claim_ref in report.claim_refs
        )
        if source_claims_present:
            code = str(getattr(node, "code", "") or "")
            method_fingerprint = str(
                getattr(node, "method_fingerprint", "") or ""
            )
            if len(method_fingerprint) != 64:
                method_fingerprint = hashlib.sha256(
                    code.encode("utf-8")
                ).hexdigest()
            claim = Claim(
                claim_id=claim_id,
                claim_type=claim_type,
                subject_artifact_id=artifact_id,
                task_scope={"task_id": self.task_id},
                method_fingerprint=method_fingerprint,
                protocol_ref=self.active_protocol,
                statement=(
                    "The admitted experience causally changed this artifact."
                    if causal
                    else "The admitted experience was realized by this artifact."
                ),
                parent_claims=sorted(set(report.claim_refs)),
                source_artifact_refs=sorted(
                    set(report.source_artifact_refs)
                ),
                evidence_refs=[report.contract_id],
                boundary={
                    "experience_contract_id": report.contract_id,
                    "experience_contract_hash": report.contract_hash,
                    "required_actuation_level": int(
                        ActuationLevel.CAUSAL_CONFIRMED
                        if causal
                        else ActuationLevel.RUNTIME_CONFORMANT
                    ),
                },
            )
            receipts = receipts_for_node(
                node,
                self.active_protocol,
                self.run_id,
                collector_host=self.collector_host,
                task_id=self.task_id,
            )
            receipts.extend(
                self.actuation_tracker.receipts_for_artifact(artifact_id)
            )
            if causal and self.memory_snapshot is not None:
                adoption_events = [
                    event
                    for event in self.memory_snapshot.session_overlay.events()
                    if event.event_type == "memory_derivation_edge"
                    and event.payload.get("kind") == "adoption"
                    and event.payload.get("target_artifact_id") == artifact_id
                    and event.payload.get("contract_id")
                    == report.contract_id
                    and event.payload.get("contract_hash")
                    == report.contract_hash
                ]
                if len(adoption_events) == 1:
                    adoption_event = adoption_events[0]
                    adoption_decisions = list(
                        adoption_event.payload.get(
                            "authority_decision_refs"
                        )
                        or []
                    )
                    if len(adoption_decisions) == 1:
                        receipts.append(
                            self.collector_host.collect(
                                AdoptionPublicationCollector,
                                artifact_id=artifact_id,
                                run_id=self.run_id,
                                protocol_ref=self.active_protocol,
                                source="host.session_overlay_adoption_edge",
                                payload={
                                    "edge_id": adoption_event.payload[
                                        "edge_id"
                                    ],
                                    "edge_hash": adoption_event.payload[
                                        "edge_hash"
                                    ],
                                    "contract_hash": report.contract_hash,
                                    "adoption_decision_id": (
                                        adoption_decisions[0]
                                    ),
                                },
                            )
                        )
            # Parent Claim existence is checked above.  Do not evaluate the
            # source method under the target link's L3/L4 obligations: those
            # Receipts describe the target adoption edge, not the source
            # Claim's historical execution path.
            self._register_claim_and_receipts(
                claim,
                receipts,
                required_parent_claims=[],
            )
            request_claim_id = claim.claim_id
        else:
            # A link without a source Claim would be untraceable.  Ask the
            # engine about a deliberately absent Claim so the denial is a
            # normal fail-closed decision rather than an internal exception.
            request_claim_id = claim_id
        request = AuthorityRequest(
            artifact_id=artifact_id,
            claim_id=request_claim_id,
            operation=operation,
            decision_stage=DecisionStage.MEMORY_WRITEBACK,
            active_protocol=self.active_protocol,
            task_context=TaskContext(task_id=self.task_id),
            requesting_component=requesting_component,
            generation_stage=axes.generation_stage,
            governance_stage=GovernanceStage.MEMORY_WRITEBACK,
        )
        decision = self.engine.authorize(request)
        key = (artifact_id, f"{operation.value}:{report.contract_id}")
        self._latest_decisions[key] = decision
        refs = getattr(node, "authority_decision_refs", None)
        if isinstance(refs, list) and decision.decision_id not in refs:
            refs.append(decision.decision_id)
        self._emit_snapshots()
        return decision

    def append_authorized_experience_link(
        self,
        node: Any,
        *,
        contract_id: str,
        causal: bool = False,
    ) -> bool:
        """Append an already-authorized adoption/causal edge to the Overlay."""

        if self.memory_snapshot is None:
            return False
        artifact_id = str(node.id)
        operation = (
            Operation.PUBLISH_CAUSAL
            if causal
            else Operation.PUBLISH_ADOPTION
        )
        link_key = (artifact_id, operation.value, str(contract_id))
        if link_key in self._overlay_written_links:
            return False
        decision = self._latest_decisions.get(
            (artifact_id, f"{operation.value}:{contract_id}")
        )
        if decision is None or not decision.allowed:
            return False
        report = self.actuation_tracker.report(
            artifact_id=artifact_id,
            contract_id=str(contract_id),
        )
        required_level = int(
            ActuationLevel.CAUSAL_CONFIRMED
            if causal
            else ActuationLevel.RUNTIME_CONFORMANT
        )
        if report.highest_level is None or report.highest_level < required_level:
            return False
        self.memory_snapshot.assert_unchanged()
        kind = "causal" if causal else "adoption"
        idempotency_key = hashlib.sha256(
            canonical_json(
                {
                    "kind": kind,
                    "target_artifact_id": artifact_id,
                    "contract_hash": report.contract_hash,
                    "protocol_ref": self.active_protocol.key(),
                }
            ).encode("utf-8")
        ).hexdigest()
        existing = [
            event
            for event in self.memory_snapshot.session_overlay.events()
            if event.event_type == "memory_derivation_edge"
            and event.payload.get("idempotency_key") == idempotency_key
        ]
        if existing:
            if len(existing) != 1:
                raise ValueError("Duplicate experience derivation edges")
            self._overlay_written_links.add(link_key)
            return False
        if causal:
            adoption_exists = any(
                event.event_type == "memory_derivation_edge"
                and event.payload.get("kind") == "adoption"
                and event.payload.get("target_artifact_id") == artifact_id
                and event.payload.get("contract_hash")
                == report.contract_hash
                for event in self.memory_snapshot.session_overlay.events()
            )
            if not adoption_exists:
                return False
        contract_receipts = [
            receipt
            for receipt in self.actuation_tracker.receipts_for_artifact(
                artifact_id
            )
            if receipt.payload.get("contract_hash") == report.contract_hash
        ]
        receipt_refs = {
            receipt.receipt_type.value: sorted(
                candidate.receipt_id
                for candidate in contract_receipts
                if candidate.receipt_type == receipt.receipt_type
            )
            for receipt in contract_receipts
        }
        authority_snapshot_pointer = (
            self._sealed_authority_snapshot_pointer()
        )
        edge_payload = {
            "schema": "experience_derivation_edge_v1",
            "kind": kind,
            "idempotency_key": idempotency_key,
            "edge_id": "",
            "edge_hash": "",
            "target_artifact_id": artifact_id,
            "task_id": self.task_id,
            "protocol_ref": self.active_protocol.key(),
            "authority_policy_version": self.engine.policy_version,
            "operation": operation.value,
            "edge_claim_ref": decision.claim_id,
            "contract_id": report.contract_id,
            "contract_hash": report.contract_hash,
            "actuation_report_ref": report.report_id,
            "actuation_report_hash": report.report_hash,
            "actuation_level": report.highest_level,
            "source_claim_refs": sorted(set(report.claim_refs)),
            "source_clause_ref": report.clause_id,
            "source_sop_ref": report.sop_id,
            "source_artifact_refs": sorted(
                set(report.source_artifact_refs)
            ),
            "source_transition_refs": sorted(
                set(report.source_transition_refs)
            ),
            "source_run_ids": sorted(set(report.source_run_ids)),
            "source_task_ids": sorted(set(report.source_task_ids)),
            "static_receipt_refs": receipt_refs.get(
                "static_actuation", []
            ),
            "runtime_receipt_refs": receipt_refs.get(
                "runtime_actuation", []
            ),
            "counterfactual_receipt_refs": receipt_refs.get(
                "counterfactual_actuation", []
            ),
            "authority_decision_refs": [decision.decision_id],
            "authority_snapshot_pointer": authority_snapshot_pointer,
        }
        identity_payload = {
            key: value
            for key, value in edge_payload.items()
            if key not in {"edge_id", "edge_hash"}
        }
        edge_hash = hashlib.sha256(
            canonical_json(identity_payload).encode("utf-8")
        ).hexdigest()
        edge_payload["edge_id"] = f"experience_edge::{edge_hash[:24]}"
        edge_payload["edge_hash"] = edge_hash
        event = self.memory_snapshot.session_overlay.append(
            "memory_derivation_edge",
            edge_payload,
        )
        self._overlay_written_links.add(link_key)
        if self.mode != "off":
            self.ledger.append(
                "experience_link_appended",
                {
                    "artifact_id": artifact_id,
                    "operation": operation.value,
                    "contract_id": report.contract_id,
                    "overlay_event_id": event.event_id,
                    "overlay_event_hash": event.event_hash,
                    "decision_id": decision.decision_id,
                },
            )
        return True

    def append_authorized_memory_overlay(self, node: Any) -> bool:
        """Append a promoted memory event without ever mutating the Base."""

        if self.memory_snapshot is None:
            return False
        artifact_id = str(node.id)
        if artifact_id in self._overlay_written_artifacts:
            return False
        decision = self._latest_decisions.get(
            (artifact_id, Operation.PROMOTE_RESULT.value)
        )
        if decision is None or not decision.allowed:
            return False
        reports = self.actuation_tracker.reports_for_artifact(artifact_id)
        verified_adoption_reports = [
            report
            for report in reports
            if report.highest_level is not None
            and report.highest_level >= 3
        ]
        self.memory_snapshot.assert_unchanged()
        claims = [
            claim
            for claim in self.engine.graph.claims.values()
            if claim.subject_artifact_id == artifact_id
        ]
        authority_snapshot_pointer = (
            self._sealed_authority_snapshot_pointer()
        )
        audit = getattr(node, "leakage_audit", None) or {}
        audited = bool(
            audit.get("status") == "clean"
            and audit.get("metric_disposition") == "accept"
        )
        permitted_operations = list(
            (decision.permitted_scope.operations if decision.permitted_scope else [])
        )
        code_sha256 = str(
            getattr(node, "code_sha256_expected", "")
            or hashlib.sha256(
                str(getattr(node, "code", "") or "").encode("utf-8")
            ).hexdigest()
        )
        idempotency_key = hashlib.sha256(
            canonical_json(
                {
                    "run_id": self.run_id,
                    "artifact_id": artifact_id,
                    "protocol_hash": self.active_protocol.canonical_hash,
                    "publication_class": "result_fact",
                }
            ).encode("utf-8")
        ).hexdigest()
        existing = [
            event
            for event in self.memory_snapshot.session_overlay.events()
            if event.event_type == "memory_claim"
            and event.payload.get("idempotency_key") == idempotency_key
        ]
        if existing:
            if len(existing) != 1:
                raise ValueError("Duplicate Result Fact events")
            self._overlay_written_artifacts.add(artifact_id)
            return False
        event = self.memory_snapshot.session_overlay.append(
            "memory_claim",
            {
                "schema": "runtime_result_fact_v1",
                "idempotency_key": idempotency_key,
                "artifact_id": artifact_id,
                "run_id": self.run_id,
                "task_id": self.task_id,
                "protocol_ref": self.active_protocol.key(),
                "authority_policy_version": self.engine.policy_version,
                "claim_refs": sorted(claim.claim_id for claim in claims),
                "claim_types": sorted(
                    {claim.claim_type.value for claim in claims}
                ),
                "receipt_refs": sorted(
                    {
                        receipt_id
                        for claim in claims
                        for path_id in self.engine.graph.claim_paths.get(
                            claim.claim_id, []
                        )
                        for receipt_id in self.engine.graph.paths[
                            path_id
                        ].receipt_ids
                    }
                ),
                "authority_decision_refs": [decision.decision_id],
                "authority_snapshot_pointer": authority_snapshot_pointer,
                "exposure_report_refs": sorted(
                    report.report_id for report in reports
                ),
                "exposure_report_hashes": sorted(
                    report.report_hash for report in reports
                ),
                "verified_adoption_report_refs": sorted(
                    report.report_id for report in verified_adoption_reports
                ),
                # A result-fact write does not mint a derivation edge.  A
                # separate PUBLISH_ADOPTION decision owns that permission.
                "derived_from_refs": [],
                "adoption_status": (
                    "runtime_verified_not_published"
                    if verified_adoption_reports
                    else "not_runtime_verified"
                    if reports
                    else "not_exposed"
                ),
                "artifact_pointer": {
                    "journal_path": str(
                        (Path(self.cfg.log_dir) / "journal.json").resolve()
                    ),
                    "node_id": artifact_id,
                    "resolution_status": "pending_run_close",
                },
                "code_sha256": code_sha256,
                "permitted_operations": permitted_operations,
                "audited": audited,
                "publication_class": "result_fact",
            },
        )
        self._overlay_written_artifacts.add(artifact_id)
        if self.mode != "off":
            self.ledger.append(
                "session_overlay_appended",
                {
                    "artifact_id": artifact_id,
                    "overlay_event_id": event.event_id,
                    "overlay_event_hash": event.event_hash,
                    "decision_id": decision.decision_id,
                },
            )
        return True

    def authorize_batch_nodes(
        self,
        nodes: list[Any],
        operation: Operation,
        stage: DecisionStage,
        requesting_component: str,
    ) -> list[AuthorityDecision]:
        return [self.authorize_node(node, operation, stage, requesting_component) for node in nodes]

    def permits(self, decision: AuthorityDecision, legacy_allowed: bool = True) -> bool:
        if self.mode == "off":
            return bool(legacy_allowed)
        enforced = self.rollout.should_enforce(decision)
        internal_error = any(
            item.startswith("authority_internal_error:")
            for item in decision.missing_obligations
        )
        if not enforced:
            effective_allowed = bool(legacy_allowed)
        elif internal_error and is_high_risk(decision.operation):
            effective_allowed = False
        elif internal_error:
            navigation_only = decision.operation in {
                Operation.INSPECT.value,
                Operation.DEBUG_HYPOTHESIS.value,
            }
            effective_allowed = bool(
                navigation_only and self.allow_invalid_debug and decision.allowed
            )
            logger.warning(
                "Authority internal error for low-risk %s; effective result is %s",
                decision.operation,
                "navigation-only" if effective_allowed else "abstain",
            )
        elif decision.allowed:
            effective_allowed = True
        elif decision.operation in {
            Operation.INSPECT.value,
            Operation.DEBUG_HYPOTHESIS.value,
        }:
            effective_allowed = self.allow_invalid_debug
        elif is_high_risk(decision.operation):
            effective_allowed = not self.fail_closed_high_risk
        else:
            effective_allowed = False
        self.rollout.record(
            decision,
            legacy_allowed=bool(legacy_allowed),
            effective_allowed=effective_allowed,
            enforced=enforced,
        )
        self._emit_snapshots()
        return effective_allowed

    def rollout_report(
        self,
        *,
        review_dispositions: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self.rollout.report(review_dispositions=review_dispositions)

    def evaluate_canary(
        self,
        oracle_should_allow: dict[str, bool],
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        report = evaluate_canary(
            self.rollout.records(),
            oracle_should_allow=oracle_should_allow,
            thresholds=self.canary_thresholds,
        )
        if self.mode != "off":
            self.ledger.append("authority_canary_evaluated", report)
        target = (
            Path(output_path)
            if output_path is not None
            else Path(self.cfg.log_dir) / "authority_canary_report.json"
        )
        temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(target)
        return report

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
        source_execution_verified: bool = False,
    ) -> AuthorityDecision | None:
        if self.mode == "off":
            return None
        self.seal_rollout_versions()
        claim = Claim(
            claim_id=f"replay:{artifact_id}:method_hypothesis",
            claim_type=ClaimType.METHOD_HYPOTHESIS,
            subject_artifact_id=artifact_id,
            task_scope={"task_id": self.task_id},
            method_fingerprint=code_sha256,
            protocol_ref=self.active_protocol,
            statement=f"Historical artifact {artifact_id} is eligible as an exact replay seed",
            source_artifact_refs=[artifact_id],
            evidence_refs=[f"legacy_source:{source_run_id}:{artifact_id}"],
            boundary={"fact": "historical_replay_candidate"},
            legacy_status="legacy_static_only",
        )
        self._register_claim_and_receipts(
            claim,
            receipts_for_replay_source(
                artifact_id,
                code_sha256,
                audit,
                self.active_protocol,
                source_run_id,
                collector_host=self.collector_host,
                source_execution_verified=source_execution_verified,
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
                generation_stage=GenerationStage.DEBUG,
                governance_stage=GovernanceStage.REPLAY,
            )
        )
        self._latest_decisions[(artifact_id, (
            Operation.REPAIR_SEED.value
            if repair_seed
            else Operation.CODE_SEED.value
        ))] = decision
        self._emit_snapshots()
        return decision


def get_authority_adapter(agent: Any) -> MLEvolveAuthorityAdapter | None:
    return getattr(agent, "evaluation_authority", None)
