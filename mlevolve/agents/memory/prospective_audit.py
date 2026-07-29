"""Prospective dual-observer logging for Claim-use retrieval decisions.

The raw opportunity event is appended before ranking/Prompt filtering.  A
unified decision row is appended only after the generated node has completed
and its runtime Receipt refs are available.  Unfinalized opportunities remain
visible as a coverage failure rather than being silently dropped.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Mapping


DECISION_SCHEMA = "mlevolve_prospective_claim_use_decision_v1"
OPPORTUNITY_SCHEMA = "mlevolve_prospective_decision_opportunity_v1"
RECEIPT_SCHEMA = "mlevolve_prospective_observer_receipt_v1"


logger = logging.getLogger("MLEvolve")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _canonical(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (_canonical(payload) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "ab", closefd=True) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


class ProspectiveAuditLogger:
    """Thread-safe run-scoped writer with thread-local Prompt binding."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.cfg = agent.cfg
        audit_cfg = getattr(self.cfg, "prospective_audit", None)
        self.enabled = bool(getattr(audit_cfg, "enabled", False))
        self.allow_pending_counterfactual = bool(
            getattr(audit_cfg, "allow_pending_counterfactual", False)
        )
        self.counterfactual_timeout_seconds = int(
            getattr(audit_cfg, "counterfactual_timeout_seconds", 300) or 300
        )
        self.counterfactual_generation_attempts = max(
            1,
            int(getattr(audit_cfg, "counterfactual_generation_attempts", 2) or 2),
        )
        self.counterfactual_memory_max_chars = int(
            getattr(audit_cfg, "counterfactual_memory_max_chars", 12000) or 12000
        )
        self.root = Path(self.cfg.log_dir)
        self.opportunity_path = self.root / "decision_opportunities.jsonl"
        self.ledger_path = self.root / "prospective_decision_ledger.jsonl"
        self.receipt_path = self.root / "prospective_observer_receipts.jsonl"
        self._lock = threading.RLock()
        self._local = threading.local()
        self._pending_by_node: dict[str, list[dict[str, Any]]] = {}
        self._counterfactual_prepared: set[str] = set()

    @staticmethod
    def _excerpt(value: Any, limit: int = 1800) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n... [observer excerpt truncated]"

    def _bounded_counterfactual_items(
        self, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        bounded: list[dict[str, Any]] = []
        used = 0
        for item in items:
            normalized = _jsonable(item)
            encoded = _canonical(normalized)
            if used + len(encoded) > self.counterfactual_memory_max_chars:
                remaining = self.counterfactual_memory_max_chars - used
                if remaining <= 256:
                    break
                normalized = {
                    "candidate_id": str(item.get("candidate_id") or ""),
                    "claim_id": str(item.get("claim_id") or ""),
                    "text": self._excerpt(item.get("text"), max(64, remaining - 192)),
                    "truncated": True,
                }
                encoded = _canonical(normalized)
            bounded.append(normalized)
            used += len(encoded)
        return bounded

    def _pending(self) -> list[dict[str, Any]]:
        pending = getattr(self._local, "pending", None)
        if pending is None:
            pending = []
            self._local.pending = pending
        return pending

    def _receipt(
        self,
        *,
        decision_id: str,
        candidate_id: str,
        claim_id: str,
        reason: str,
        authority_refs: list[str],
    ) -> str:
        payload = {
            "schema": RECEIPT_SCHEMA,
            "run_id": self.agent.evaluation_authority.run_id,
            "task_id": self.agent.evaluation_authority.task_id,
            "decision_id": decision_id,
            "candidate_id": candidate_id,
            "claim_id": claim_id,
            "reason": reason,
            "authority_decision_refs": sorted(set(authority_refs)),
        }
        receipt_id = f"prospective-observer::{_sha(payload)}"
        payload["receipt_id"] = receipt_id
        payload["recorded_at_ns"] = time.time_ns()
        _append_jsonl(self.receipt_path, payload)
        return receipt_id

    def _record(self, payload: dict[str, Any]) -> str:
        if not self.enabled:
            return ""
        decision_id = str(payload["decision_id"])
        opportunity = {
            "schema": OPPORTUNITY_SCHEMA,
            "run_id": payload["run_id"],
            "task_id": payload["task_id"],
            "agent_seed": payload["agent_seed"],
            "decision_id": decision_id,
            "decision_stage": payload["decision_stage"],
            "operation": payload["operation"],
            "protocol_ref": payload["protocol_ref"],
            "source": payload["source"],
            "raw_candidate_ids": list(payload["raw_candidate_ids"]),
            "raw_relevance_scores": list(payload["raw_relevance_scores"]),
            "raw_claim_ids": list(payload["raw_claim_ids"]),
            "raw_claim_types": list(payload["raw_claim_types"]),
            "raw_logged_before_filtering": True,
            "raw_logged_at_ns": time.time_ns(),
        }
        opportunity["opportunity_hash"] = _sha(opportunity)
        with self._lock:
            _append_jsonl(self.opportunity_path, opportunity)
            self._pending().append(payload)
        return decision_id

    def record_visibility(self, pack: Any, gateway: Any, *, source: str) -> str:
        """Record exact Claim-use observations emitted by SOP Authority."""

        if not self.enabled or pack is None:
            return ""
        trace = _jsonable(getattr(pack, "visibility_trace", {}) or {})
        observations = trace.get("raw_shadow_authority_decisions") or []
        if not observations:
            return ""
        unique: dict[tuple[str, str], dict[str, Any]] = {}
        for observation in observations:
            if not isinstance(observation, dict):
                continue
            clause_id = str(observation.get("clause_id") or "")
            claim_id = str(observation.get("claim_id") or "")
            if clause_id and claim_id:
                unique[(clause_id, claim_id)] = dict(observation)
        if not unique:
            return ""

        request = trace.get("request") or {}
        decision_id = uuid.uuid4().hex
        suppressed = set(trace.get("suppressed_clause_refs") or [])
        final_ids = set(trace.get("effective_visible_clause_ids") or [])
        decisions = trace.get("clause_decisions") or {}
        raw_candidate_ids: list[str] = []
        raw_claim_ids: list[str] = []
        raw_claim_types: list[str] = []
        shadow: list[dict[str, Any]] = []
        suppression_reasons: dict[str, dict[str, Any]] = {}
        observer_receipts: list[str] = []
        counterfactual_items: list[dict[str, Any]] = []
        clauses = getattr(gateway, "clauses", {}) or {}
        for (clause_id, claim_id), observation in sorted(unique.items()):
            raw_candidate_ids.append(clause_id)
            raw_claim_ids.append(claim_id)
            raw_claim_types.append(str(observation.get("claim_type") or "unknown"))
            normalized_observation = dict(observation)
            # Visibility traces call this identifier ``clause_id`` while the
            # unified Claim-use ledger contract calls it ``candidate_id``.
            # Preserve both so every raw (candidate, Claim) pair joins exactly.
            normalized_observation.setdefault("candidate_id", clause_id)
            shadow.append(normalized_observation)
            if clause_id not in suppressed:
                continue
            clause = clauses.get(clause_id)
            counterfactual_items.append(
                {
                    "candidate_id": clause_id,
                    "claim_id": claim_id,
                    "claim_type": str(observation.get("claim_type") or "unknown"),
                    "text": self._excerpt(
                        getattr(clause, "text", "")
                        or observation.get("text")
                        or observation.get("retrieval_text")
                    ),
                }
            )
            clause_decision = decisions.get(clause_id) or {}
            authority_refs = list(clause_decision.get("authority_decision_refs") or [])
            authority_refs.extend(observation.get("blocking_receipts") or [])
            reason = str(clause_decision.get("reason") or "authority_suppressed")
            receipt_id = self._receipt(
                decision_id=decision_id,
                candidate_id=clause_id,
                claim_id=claim_id,
                reason=reason,
                authority_refs=[str(value) for value in authority_refs],
            )
            observer_receipts.append(receipt_id)
            suppression_reasons[clause_id] = {
                "candidate_id": clause_id,
                "claim_id": claim_id,
                "operation": str(request.get("operation") or observation.get("operation") or ""),
                "decision_stage": str(
                    request.get("generation_stage")
                    or observation.get("decision_stage")
                    or ""
                ),
                "protocol_ref": str(observation.get("protocol_ref") or ""),
                "reason_codes": list(observation.get("reason_codes") or []),
                "gate_reason": reason,
                "authority_decision_refs": sorted(set(map(str, authority_refs))),
                "receipt_refs": [receipt_id],
            }

        protocol_ref = str(next(iter(unique.values())).get("protocol_ref") or "")
        payload = {
            "schema": DECISION_SCHEMA,
            "source": source,
            "run_id": self.agent.evaluation_authority.run_id,
            "task_id": self.agent.evaluation_authority.task_id,
            "agent_seed": int(self.cfg.agent.seed),
            "decision_id": decision_id,
            "decision_stage": str(request.get("generation_stage") or "retrieval"),
            "operation": str(request.get("operation") or "inspect"),
            "protocol_ref": protocol_ref,
            "raw_candidate_ids": raw_candidate_ids,
            "raw_relevance_scores": [0.0] * len(raw_candidate_ids),
            "raw_score_semantics": "unscored_pre_ranking_claim_proposal",
            "raw_claim_ids": raw_claim_ids,
            "raw_claim_types": raw_claim_types,
            "shadow_authority_decisions": shadow,
            "suppressed_candidate_ids": sorted(suppressed & set(raw_candidate_ids)),
            "suppression_reasons": suppression_reasons,
            "final_prompt_candidate_ids": sorted(final_ids & set(raw_candidate_ids)),
            "actual_action_hash": "",
            "actual_code_hash": "",
            "runtime_receipt_refs": sorted(set(observer_receipts)),
            "counterfactual_action_hash": "",
            "counterfactual_code_hash": "",
            "counterfactual_status": "pending" if suppressed else "identity",
            "visibility_request_id": str(getattr(pack, "request_id", "")),
            "_counterfactual_memory_items": self._bounded_counterfactual_items(
                counterfactual_items
            ),
        }
        return self._record(payload)

    @staticmethod
    def _synthetic_claims(candidate: dict[str, Any], node: dict[str, Any]) -> list[tuple[str, str]]:
        candidate_id = str(candidate["candidate_id"])
        claims = [(f"{candidate_id}::claim::method_hypothesis", "method_hypothesis")]
        metric = node.get("metric")
        if metric is not None and metric != {}:
            claims.append((f"{candidate_id}::claim::score", "score"))
        audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
        for issue in audit.get("issues") or []:
            code = str((issue or {}).get("issue_code") or "unknown")
            claims.append((f"{candidate_id}::claim::audit_finding::{code}", "audit_finding"))
        if node.get("is_buggy") is False and node.get("is_valid") is not False:
            claims.append((f"{candidate_id}::claim::executed", "executed"))
        return claims

    def record_run_candidates(self, pack: dict[str, Any], nodes: Mapping[str, Any]) -> str:
        """Record scored RunNode proposals before execution/rank eligibility."""

        if not self.enabled:
            return ""
        candidates = [
            dict(value)
            for value in (pack.get("pre_gate_raw_candidates") or [])
            if isinstance(value, dict) and value.get("candidate_id")
        ]
        if not candidates:
            return ""
        trace = pack.get("visibility_trace") or {}
        request = trace.get("request") or {}
        decision_id = uuid.uuid4().hex
        raw_candidate_ids: list[str] = []
        raw_scores: list[float] = []
        raw_claim_ids: list[str] = []
        raw_claim_types: list[str] = []
        shadow: list[dict[str, Any]] = []
        suppressed: set[str] = set()
        reasons: dict[str, dict[str, Any]] = {}
        observer_receipts: list[str] = []
        counterfactual_items: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            node = nodes.get(candidate_id) or {}
            allowed = bool(candidate.get("operation_authorized"))
            for claim_id, claim_type in self._synthetic_claims(candidate, node):
                raw_candidate_ids.append(candidate_id)
                raw_scores.append(float(candidate.get("score") or 0.0))
                raw_claim_ids.append(claim_id)
                raw_claim_types.append(claim_type)
                shadow.append(
                    {
                        "candidate_id": candidate_id,
                        "claim_id": claim_id,
                        "claim_type": claim_type,
                        "outcome": "allow" if allowed else "deny",
                        "allowed": allowed,
                        "decision_source": "run_node_execution_eligibility_shadow",
                        "gate_reason": str(candidate.get("gate_reason") or ""),
                    }
                )
                if allowed:
                    continue
                suppressed.add(candidate_id)
                if not any(
                    item.get("candidate_id") == candidate_id
                    for item in counterfactual_items
                ):
                    counterfactual_items.append(
                        {
                            "candidate_id": candidate_id,
                            "source_task_id": str(node.get("task") or ""),
                            "source_stage": str(node.get("stage") or ""),
                            "plan": self._excerpt(
                                node.get("plan")
                                or node.get("description")
                                or node.get("code_summary")
                            ),
                            "metric": _jsonable(node.get("metric")),
                            "audit_status": str(
                                (node.get("leakage_audit") or {}).get("status")
                                if isinstance(node.get("leakage_audit"), dict)
                                else node.get("audit_status") or ""
                            ),
                            "gate_reason": str(candidate.get("gate_reason") or ""),
                        }
                    )
                receipt_id = self._receipt(
                    decision_id=decision_id,
                    candidate_id=candidate_id,
                    claim_id=claim_id,
                    reason=str(candidate.get("gate_reason") or "run_node_ineligible"),
                    authority_refs=[],
                )
                observer_receipts.append(receipt_id)
                reasons.setdefault(
                    candidate_id,
                    {
                        "candidate_id": candidate_id,
                        "claim_id": claim_id,
                        "claim_ids": [],
                        "operation": str(request.get("operation") or "generate_candidate"),
                        "decision_stage": str(request.get("generation_stage") or pack.get("stage_route", {}).get("stage") or ""),
                        "protocol_ref": self.agent.evaluation_authority.active_protocol.key(),
                        "reason_codes": [str(candidate.get("gate_reason") or "run_node_ineligible")],
                        "gate_reason": str(candidate.get("gate_reason") or "run_node_ineligible"),
                        "authority_decision_refs": [],
                        "receipt_refs": [],
                    },
                )
                reasons[candidate_id]["claim_ids"].append(claim_id)
                reasons[candidate_id]["receipt_refs"].append(receipt_id)

        final_ids = set(map(str, pack.get("final_prompt_candidate_ids") or []))
        payload = {
            "schema": DECISION_SCHEMA,
            "source": "run_forest_raw_run_nodes",
            "run_id": self.agent.evaluation_authority.run_id,
            "task_id": self.agent.evaluation_authority.task_id,
            "agent_seed": int(self.cfg.agent.seed),
            "decision_id": decision_id,
            "decision_stage": str(request.get("generation_stage") or pack.get("stage_route", {}).get("stage") or "retrieval"),
            "operation": str(request.get("operation") or "generate_candidate"),
            "protocol_ref": self.agent.evaluation_authority.active_protocol.key(),
            "raw_candidate_ids": raw_candidate_ids,
            "raw_relevance_scores": raw_scores,
            "raw_score_semantics": "production_raw_run_node_relevance",
            "raw_claim_ids": raw_claim_ids,
            "raw_claim_types": raw_claim_types,
            "shadow_authority_decisions": shadow,
            "suppressed_candidate_ids": sorted(suppressed),
            "suppression_reasons": reasons,
            "final_prompt_candidate_ids": sorted(final_ids),
            "actual_action_hash": "",
            "actual_code_hash": "",
            "runtime_receipt_refs": sorted(set(observer_receipts)),
            "counterfactual_action_hash": "",
            "counterfactual_code_hash": "",
            "counterfactual_status": "pending" if suppressed else "identity",
            "_counterfactual_memory_items": self._bounded_counterfactual_items(
                counterfactual_items
            ),
        }
        return self._record(payload)

    def bind_thread_to_node(self, node: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            pending = list(self._pending())
            self._local.pending = []
            if pending:
                self._pending_by_node.setdefault(str(node.id), []).extend(pending)

    def prepare_counterfactuals(self, node: Any) -> None:
        """Generate observer-only raw-memory arms before Candidate execution."""
        if not self.enabled:
            return
        node_id = str(node.id)
        with self._lock:
            if node_id in self._counterfactual_prepared:
                return
            self._counterfactual_prepared.add(node_id)
            pending = list(self._pending_by_node.get(node_id, []))

        failures: list[str] = []
        for payload in pending:
            if not payload.get("suppressed_candidate_ids"):
                continue
            items = list(payload.get("_counterfactual_memory_items") or [])
            if not items:
                message = f"{payload['decision_id']}: no frozen suppressed-memory payload"
                payload["_counterfactual_error"] = message
                failures.append(message)
                continue
            actual_prompt = str(getattr(node, "prompt_input", "") or "")
            counterfactual_prompt = (
                actual_prompt
                + "\n\n# Prospective paired observer: raw pre-Authority memory arm\n"
                + "This observer-only replay keeps the same task context, model, "
                + "temperature and decoding policy. For this arm only, treat the "
                + "following frozen pre-Authority Claim-use candidates as visible "
                + "memory. Produce the plan and complete code that you would choose. "
                + "The result will never be submitted to the training Executor.\n"
                + _canonical(items)
            )
            try:
                from agents.coder import plan_and_code_query

                plan = code = ""
                attempt_errors: list[str] = []
                per_attempt_timeout = max(
                    1.0,
                    float(self.counterfactual_timeout_seconds)
                    / self.counterfactual_generation_attempts,
                )
                for attempt in range(1, self.counterfactual_generation_attempts + 1):
                    try:
                        plan, code = plan_and_code_query(
                            self.agent,
                            counterfactual_prompt,
                            request_timeout=per_attempt_timeout,
                        )
                        if str(plan or "").strip() and str(code or "").strip():
                            break
                        raise RuntimeError(
                            "counterfactual generation returned no plan/code"
                        )
                    except Exception as attempt_error:
                        attempt_errors.append(
                            f"attempt {attempt}: {type(attempt_error).__name__}: "
                            f"{attempt_error}"
                        )
                        if attempt < self.counterfactual_generation_attempts:
                            logger.warning(
                                "Retrying prospective counterfactual after %s",
                                attempt_errors[-1],
                            )
                if not str(plan or "").strip() or not str(code or "").strip():
                    raise RuntimeError("; ".join(attempt_errors))
                payload["_counterfactual_plan"] = str(plan)
                payload["_counterfactual_code"] = str(code)
                payload["_counterfactual_prompt_hash"] = hashlib.sha256(
                    counterfactual_prompt.encode("utf-8")
                ).hexdigest()
            except Exception as error:
                message = (
                    f"{payload['decision_id']}: {type(error).__name__}: {error}"
                )
                payload["_counterfactual_error"] = message
                failures.append(message)
                logger.warning("Prospective counterfactual remains pending: %s", message)
        if failures and not self.allow_pending_counterfactual:
            raise RuntimeError(
                "Formal prospective counterfactual generation failed: "
                + "; ".join(failures)
            )

    def finalize_node(self, node: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            pending = self._pending_by_node.pop(str(node.id), [])
            for payload in pending:
                action_hash = _sha(
                    {
                        "stage": str(getattr(node, "stage", "")),
                        "draft_role": str(getattr(node, "draft_role", "") or ""),
                        "plan": str(getattr(node, "plan", "") or ""),
                    }
                )
                code_hash = hashlib.sha256(
                    str(getattr(node, "code", "") or "").encode("utf-8")
                ).hexdigest()
                payload["actual_action_hash"] = action_hash
                payload["actual_code_hash"] = code_hash
                payload["actual_prompt_hash"] = hashlib.sha256(
                    str(getattr(node, "prompt_input", "") or "").encode("utf-8")
                ).hexdigest()
                payload["actual_node_id"] = str(node.id)
                payload["runtime_receipt_refs"] = sorted(
                    set(payload.get("runtime_receipt_refs") or [])
                    | set(map(str, getattr(node, "receipt_refs", []) or []))
                )
                if not payload.get("suppressed_candidate_ids"):
                    payload["counterfactual_action_hash"] = action_hash
                    payload["counterfactual_code_hash"] = code_hash
                    payload["counterfactual_status"] = "identity"
                else:
                    cf_plan = str(payload.pop("_counterfactual_plan", "") or "")
                    cf_code = str(payload.pop("_counterfactual_code", "") or "")
                    items = list(payload.pop("_counterfactual_memory_items", []) or [])
                    cf_prompt_hash = str(
                        payload.pop("_counterfactual_prompt_hash", "") or ""
                    )
                    cf_error = str(payload.pop("_counterfactual_error", "") or "")
                    if cf_plan and cf_code and cf_prompt_hash:
                        cf_action_hash = _sha(
                            {
                                "stage": str(getattr(node, "stage", "")),
                                "draft_role": str(
                                    getattr(node, "draft_role", "") or ""
                                ),
                                "plan": cf_plan,
                            }
                        )
                        cf_code_hash = hashlib.sha256(
                            cf_code.encode("utf-8")
                        ).hexdigest()
                        control_hash = _sha(
                            {
                                "decision_id": payload["decision_id"],
                                "actual_prompt_hash": payload["actual_prompt_hash"],
                                "model": str(self.agent.acfg.code.model),
                                "temperature": float(self.agent.acfg.code.temp),
                                "counterfactual_prompt_hash": cf_prompt_hash,
                            }
                        )
                        memory_payload_hash = _sha(items)
                        pair_id = f"prospective-pair::{control_hash[:24]}"
                        pair_result = {
                            "pair_id": pair_id,
                            "control_hash": control_hash,
                            "memory_payload_hash": memory_payload_hash,
                            "memory_on_action_hash": cf_action_hash,
                            "memory_off_action_hash": action_hash,
                            "memory_on_code_hash": cf_code_hash,
                            "memory_off_code_hash": code_hash,
                            "action_or_code_changed": bool(
                                cf_action_hash != action_hash
                                or cf_code_hash != code_hash
                            ),
                            "never_submitted_to_executor": True,
                        }
                        receipt = self.agent.evaluation_authority.record_prospective_counterfactual(
                            node,
                            pair_result=pair_result,
                        )
                        payload["counterfactual_action_hash"] = cf_action_hash
                        payload["counterfactual_code_hash"] = cf_code_hash
                        payload["counterfactual_status"] = "complete"
                        payload["counterfactual_pair_id"] = pair_id
                        payload["counterfactual_control_hash"] = control_hash
                        payload["counterfactual_memory_payload_hash"] = memory_payload_hash
                        payload["counterfactual_prompt_hash"] = cf_prompt_hash
                        payload["counterfactual_influence_confirmed"] = pair_result[
                            "action_or_code_changed"
                        ]
                        payload["counterfactual_receipt_refs"] = [receipt.receipt_id]
                        payload["runtime_receipt_refs"] = sorted(
                            set(payload["runtime_receipt_refs"])
                            | {receipt.receipt_id}
                        )
                    else:
                        payload["counterfactual_status"] = "pending"
                        payload["counterfactual_error"] = cf_error or "not_generated"
                        if not self.allow_pending_counterfactual:
                            raise RuntimeError(
                                "Formal prospective counterfactual is incomplete: "
                                f"{payload['decision_id']}"
                            )
                payload.pop("_counterfactual_memory_items", None)
                payload["finalized_at_ns"] = time.time_ns()
                payload["decision_row_hash"] = _sha(payload)
                _append_jsonl(self.ledger_path, payload)


__all__ = [
    "DECISION_SCHEMA",
    "OPPORTUNITY_SCHEMA",
    "ProspectiveAuditLogger",
]
