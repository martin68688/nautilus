from __future__ import annotations

import copy
import dataclasses
import hashlib
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from authority.adapters.mlevolve.retrieval_gate import (
    ClauseGateDecision,
    authorize_clause_for_visibility,
)
from authority.actuation import ExperienceContractCompiler
from authority.domain_scope import canonical_domain, transfer_is_compatible
from authority.models import (
    AuthorityDecision,
    AuthorityRequest,
    ClaimType,
    Operation,
    SOPClauseV1,
    VisibilityRequest,
    VisibleSOPPack,
    canonical_operation,
)
from authority.protocol_registry import canonical_json
from authority.stage_ontology import (
    GenerationStage,
    GovernanceStage,
    StageAxes,
    legacy_decision_stage_value,
)


VISIBILITY_MODES = {"off", "shadow", "enforce"}
RETRIEVAL_PROFILES = {
    "full_decision_admissibility",
    "flat_relevance_memory",
    "global_validity_bit",
    "authority_only",
}


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _tuple_values(value: Any) -> tuple[str, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(str(key) for key in value)
    return tuple(
        str(item) for item in value if item is not None and item != ""
    )


def _field_text(node: Mapping[str, Any], field: str) -> str:
    value = node.get(field)
    if isinstance(value, (list, tuple, set)):
        return "; ".join(
            str(item) for item in value if item is not None and item != ""
        )
    return "" if value is None else str(value)


def _clause_from_mapping(
    raw: Mapping[str, Any],
    *,
    sop_id: str,
    default_legacy_status: str,
) -> SOPClauseV1:
    clause_id = str(raw.get("clause_id") or raw.get("id") or "")
    if not clause_id:
        digest = hashlib.sha256(canonical_json(dict(raw)).encode("utf-8")).hexdigest()[:16]
        clause_id = f"{sop_id}::clause::{digest}"
    text = str(raw.get("text") or raw.get("action") or raw.get("title") or "")
    retrieval_text = str(raw.get("retrieval_text") or text)
    task_scope = raw.get("task_scope") if isinstance(raw.get("task_scope"), dict) else {}
    return SOPClauseV1(
        clause_id=clause_id,
        sop_id=sop_id,
        text=text,
        retrieval_text=retrieval_text,
        claim_refs=_tuple_values(raw.get("claim_refs")),
        claim_types=_tuple_values(raw.get("claim_types") or raw.get("claim_type")),
        source_artifact_refs=_tuple_values(
            raw.get("source_artifact_refs") or raw.get("source_refs")
        ),
        source_transition_refs=_tuple_values(raw.get("source_transition_refs")),
        source_run_ids=_tuple_values(raw.get("source_run_ids")),
        source_task_ids=_tuple_values(raw.get("source_task_ids")),
        source_task_families=_tuple_values(raw.get("source_task_families")),
        source_domains=_tuple_values(raw.get("source_domains")),
        transfer_scope=str(raw.get("transfer_scope") or ""),
        protocol_scope=_tuple_values(raw.get("protocol_scope") or raw.get("protocol_ref")),
        task_scope=dict(task_scope),
        permitted_operations=_tuple_values(
            raw.get("permitted_operations") or raw.get("operation_scope")
        ),
        permitted_generation_stages=_tuple_values(raw.get("permitted_generation_stages")),
        permitted_governance_stages=_tuple_values(raw.get("permitted_governance_stages")),
        publication_class=str(raw.get("publication_class") or "diagnostic"),
        authority_decision_refs=_tuple_values(raw.get("authority_decision_refs")),
        receipt_refs=_tuple_values(raw.get("receipt_refs")),
        derivation_refs=_tuple_values(raw.get("derivation_refs")),
        applies_when=_tuple_values(raw.get("applies_when")),
        prevents=_tuple_values(raw.get("prevents")),
        contract_spec=(
            dict(raw.get("contract_spec"))
            if isinstance(raw.get("contract_spec"), Mapping)
            else {}
        ),
        protocol_agnostic=raw.get("protocol_agnostic") is True,
        legacy_status=str(raw.get("legacy_status") or default_legacy_status),
    )


class SOPVisibilityGateway:
    """Clause-level visibility evaluated before any SOP text is ranked."""

    def __init__(
        self,
        nodes: Mapping[str, Mapping[str, Any]],
        *,
        mode: str = "shadow",
        authority_engine: Any | None = None,
        decision_lookup: Callable[
            [str], AuthorityDecision | dict[str, Any] | None
        ] | None = None,
        contract_compiler: ExperienceContractCompiler | None = None,
        retrieval_profile: str = "full_decision_admissibility",
        enforce_operations: Iterable[Operation | str] = (),
        enforce_generation_stages: Iterable[str] = (),
        enforce_governance_stages: Iterable[str] = (),
    ) -> None:
        self.mode = str(mode or "shadow").lower()
        if self.mode not in VISIBILITY_MODES:
            raise ValueError(f"Unsupported SOP visibility mode: {self.mode}")
        self.authority_engine = authority_engine
        self.decision_lookup = decision_lookup
        self.contract_compiler = contract_compiler or ExperienceContractCompiler()
        self.retrieval_profile = str(retrieval_profile)
        if self.retrieval_profile not in RETRIEVAL_PROFILES:
            raise ValueError(
                f"Unsupported SOP retrieval profile: {self.retrieval_profile}"
            )
        self.enforce_operations = {
            canonical_operation(value).value for value in enforce_operations
        }
        self.enforce_generation_stages = {
            str(value) for value in enforce_generation_stages if str(value)
        }
        self.enforce_governance_stages = {
            str(value) for value in enforce_governance_stages if str(value)
        }
        self._lock = threading.RLock()
        self._cache: dict[str, VisibleSOPPack] = {}
        self.clauses: dict[str, SOPClauseV1] = {}
        self.clause_ids_by_sop: dict[str, list[str]] = defaultdict(list)
        self._load_nodes(nodes)

    def should_enforce(self, request: VisibilityRequest) -> bool:
        if self.mode != "enforce":
            return False
        if (
            self.enforce_operations
            and request.operation.value not in self.enforce_operations
        ):
            return False
        if (
            self.enforce_generation_stages
            and request.generation_stage.value not in self.enforce_generation_stages
        ):
            return False
        if (
            self.enforce_governance_stages
            and request.governance_stage.value not in self.enforce_governance_stages
        ):
            return False
        return True

    def _add_clause(self, clause: SOPClauseV1) -> None:
        existing = self.clauses.get(clause.clause_id)
        if existing is not None and existing != clause:
            raise ValueError(f"SOP clause is immutable: {clause.clause_id}")
        self.clauses[clause.clause_id] = clause
        if clause.clause_id not in self.clause_ids_by_sop[clause.sop_id]:
            self.clause_ids_by_sop[clause.sop_id].append(clause.clause_id)

    def _load_nodes(self, nodes: Mapping[str, Mapping[str, Any]]) -> None:
        # Load explicit clause nodes first, independent of graph serialization
        # order.  RunForest v2 stores SOP containers before their SOPClause
        # nodes; synthesizing a legacy fallback during a single pass would
        # duplicate every formal container and create provenance-free Prompt
        # exposures alongside the real clauses.
        for node_id, node in nodes.items():
            node_type = str(node.get("type") or "")
            if node_type == "SOPClause":
                sop_id = str(node.get("sop_id") or "")
                if sop_id:
                    self._add_clause(
                        _clause_from_mapping(
                            node, sop_id=sop_id, default_legacy_status="native_v1"
                        )
                    )
                continue
        for node_id, node in nodes.items():
            node_type = str(node.get("type") or "")
            if node_type != "SOP":
                continue
            sop_id = str(node.get("id") or node_id)
            if "clause_ids" in node:
                declared_clause_ids = [
                    str(value) for value in node.get("clause_ids") or []
                ]
                missing = [
                    clause_id
                    for clause_id in declared_clause_ids
                    if clause_id not in self.clauses
                ]
                if missing:
                    raise ValueError(
                        f"SOP {sop_id} references missing explicit clauses: {missing}"
                    )
                wrong_container = [
                    clause_id
                    for clause_id in declared_clause_ids
                    if self.clauses[clause_id].sop_id != sop_id
                ]
                if wrong_container:
                    raise ValueError(
                        f"SOP {sop_id} references clauses owned by another SOP: "
                        f"{wrong_container}"
                    )
                # An explicit RunForest-v2 container never receives an
                # additional prose-derived legacy clause.
                continue
            if (
                node.get("domain_scope_complete") is True
                or node.get("source_domains")
                or node.get("transfer_scopes")
            ):
                raise ValueError(
                    f"Domain-scoped SOP {sop_id} is missing explicit clause_ids"
                )
            raw_clauses = node.get("clauses")
            if isinstance(raw_clauses, list) and raw_clauses:
                for raw in raw_clauses:
                    if isinstance(raw, Mapping):
                        self._add_clause(
                            _clause_from_mapping(
                                raw,
                                sop_id=sop_id,
                                default_legacy_status=str(
                                    node.get("legacy_status") or "native_v1"
                                ),
                            )
                        )
                continue
            lineage = node.get("clause_lineage")
            if isinstance(lineage, list) and lineage:
                for raw_lineage in lineage:
                    if not isinstance(raw_lineage, Mapping):
                        continue
                    field = str(raw_lineage.get("field") or "")
                    text = _field_text(node, field)
                    if not text:
                        continue
                    self._add_clause(
                        SOPClauseV1(
                            clause_id=str(
                                raw_lineage.get("clause_id")
                                or f"{sop_id}::{field}"
                            ),
                            sop_id=sop_id,
                            text=text,
                            retrieval_text=text,
                            source_artifact_refs=_tuple_values(
                                raw_lineage.get("parent_claim_refs")
                            ),
                            derivation_refs=_tuple_values(
                                raw_lineage.get("derivation_refs")
                            ),
                            publication_class="diagnostic",
                            legacy_status="legacy_uncertified",
                        )
                    )
                continue
            text = "\n".join(
                value
                for value in (
                    _field_text(node, "title"),
                    _field_text(node, "action"),
                    _field_text(node, "applies_when"),
                    _field_text(node, "prevents"),
                    _field_text(node, "text"),
                )
                if value
            )
            self._add_clause(
                SOPClauseV1(
                    clause_id=f"legacy::{sop_id}",
                    sop_id=sop_id,
                    text=text,
                    retrieval_text=text,
                    claim_refs=_tuple_values(node.get("claim_refs")),
                    claim_types=_tuple_values(
                        node.get("claim_types") or node.get("claim_type")
                    ),
                    source_artifact_refs=_tuple_values(
                        node.get("source_artifact_refs")
                        or node.get("source_branches")
                    ),
                    protocol_scope=_tuple_values(
                        node.get("protocol_scope") or node.get("protocol_ref")
                    ),
                    task_scope=(
                        dict(node.get("task_scope"))
                        if isinstance(node.get("task_scope"), dict)
                        else {}
                    ),
                    permitted_operations=_tuple_values(node.get("operation_scope")),
                    publication_class=str(
                        node.get("publication_class") or "diagnostic"
                    ),
                    authority_decision_refs=_tuple_values(
                        node.get("authority_decision_refs")
                    ),
                    receipt_refs=_tuple_values(node.get("receipt_refs")),
                    protocol_agnostic=node.get("protocol_agnostic") is True,
                    legacy_status=str(
                        node.get("legacy_status") or "legacy_uncertified"
                    ),
                )
            )
        for values in self.clause_ids_by_sop.values():
            values.sort()

    def _authority_epoch(self) -> dict[str, Any]:
        graph = getattr(self.authority_engine, "graph", None)
        return {
            "claims": len(getattr(graph, "claims", {}) or {}),
            "receipts": len(getattr(graph, "receipts", {}) or {}),
            "paths": len(getattr(graph, "paths", {}) or {}),
            "policy_version": str(
                getattr(self.authority_engine, "policy_version", "") or ""
            ),
        }

    def _authority_state_hash(self, clauses: Iterable[SOPClauseV1]) -> str:
        clauses = list(clauses)
        claim_refs = sorted(
            {
                claim_ref
                for clause in clauses
                for claim_ref in clause.claim_refs
            }
        )
        decision_refs = sorted(
            {
                decision_ref
                for clause in clauses
                for decision_ref in clause.authority_decision_refs
            }
        )
        graph = getattr(self.authority_engine, "graph", None)
        claims = getattr(graph, "claims", {}) if graph is not None else {}
        paths = getattr(graph, "paths", {}) if graph is not None else {}
        claim_paths = getattr(graph, "claim_paths", {}) if graph is not None else {}
        receipts = getattr(graph, "receipts", {}) if graph is not None else {}
        relevant_path_ids = sorted(
            {
                path_id
                for claim_ref in claim_refs
                for path_id in claim_paths.get(claim_ref, [])
            }
        )
        def receipt_ids_for_path(path_id: str) -> list[str]:
            path = paths.get(path_id)
            if isinstance(path, Mapping):
                return [str(value) for value in path.get("receipt_ids") or []]
            return [str(value) for value in getattr(path, "receipt_ids", [])]

        relevant_receipt_ids = sorted(
            {
                receipt_id
                for path_id in relevant_path_ids
                for receipt_id in receipt_ids_for_path(path_id)
            }
        )
        snapshots: dict[str, Any] = {}
        if self.decision_lookup is not None:
            for decision_ref in decision_refs:
                try:
                    snapshots[decision_ref] = _jsonable(
                        self.decision_lookup(decision_ref)
                    )
                except Exception as error:
                    snapshots[decision_ref] = {
                        "lookup_error": type(error).__name__,
                    }
        payload = {
            "epoch": self._authority_epoch(),
            "claims": {
                claim_ref: _jsonable(claims.get(claim_ref))
                for claim_ref in claim_refs
            },
            "paths": {
                path_id: _jsonable(paths.get(path_id))
                for path_id in relevant_path_ids
            },
            "receipts": {
                receipt_id: _jsonable(receipts.get(receipt_id))
                for receipt_id in relevant_receipt_ids
            },
            "decision_snapshots": snapshots,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def _cache_key(
        self,
        request: VisibilityRequest,
        candidate_sop_ids: Iterable[str],
        candidate_clause_ids: Iterable[str] | None = None,
    ) -> str:
        candidate_sop_ids = sorted(set(candidate_sop_ids))
        normalized_clause_ids = (
            sorted(set(str(value) for value in candidate_clause_ids))
            if candidate_clause_ids is not None
            else None
        )
        clauses = self._candidate_clauses(
            candidate_sop_ids, candidate_clause_ids=normalized_clause_ids
        )
        payload = {
            "mode": self.mode,
            "retrieval_profile": self.retrieval_profile,
            "enforce_operations": sorted(self.enforce_operations),
            "enforce_generation_stages": sorted(
                self.enforce_generation_stages
            ),
            "enforce_governance_stages": sorted(
                self.enforce_governance_stages
            ),
            "operation": request.operation.value,
            "generation_stage": request.generation_stage.value,
            "governance_stage": request.governance_stage.value,
            "protocol_hash": request.active_protocol.canonical_hash,
            "protocol_key": request.active_protocol.key(),
            "task_id": request.task_context.task_id,
            "task_family": request.task_context.task_family,
            "target_domain": canonical_domain(
                request.task_context.task_family
            ),
            "task_attributes": request.task_context.attributes,
            "bundle_version": request.memory_bundle_version,
            "policy_version": request.authority_policy_version,
            "token_budget": request.token_budget,
            "requesting_component": request.requesting_component,
            "candidate_sop_ids": candidate_sop_ids,
            "candidate_clause_ids": normalized_clause_ids,
            "authority_state_hash": self._authority_state_hash(clauses),
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def _candidate_clauses(
        self,
        candidate_sop_ids: Iterable[str],
        *,
        candidate_clause_ids: Iterable[str] | None = None,
    ) -> list[SOPClauseV1]:
        allowed_clause_ids = (
            {str(value) for value in candidate_clause_ids}
            if candidate_clause_ids is not None
            else None
        )
        output: list[SOPClauseV1] = []
        for sop_id in sorted(set(str(value) for value in candidate_sop_ids)):
            output.extend(
                self.clauses[clause_id]
                for clause_id in self.clause_ids_by_sop.get(sop_id, [])
                if allowed_clause_ids is None or clause_id in allowed_clause_ids
            )
        return output

    @staticmethod
    def _category(
        clause: SOPClauseV1,
        decision: ClauseGateDecision,
        request: VisibilityRequest,
    ) -> str:
        types = set(decision.claim_types or clause.claim_types)
        if (
            request.operation == Operation.INSPECT
            or decision.warning
            or ClaimType.AUDIT_FINDING.value in types
        ):
            return "warning"
        if (
            clause.publication_class == "diagnostic"
        ):
            return "diagnostic"
        return "positive"

    @staticmethod
    def _rendered_by_sop(
        clauses: Iterable[SOPClauseV1],
        categories: Mapping[str, str],
    ) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[SOPClauseV1]] = defaultdict(list)
        for clause in clauses:
            grouped[clause.sop_id].append(clause)
        rendered: dict[str, dict[str, Any]] = {}
        for sop_id, values in sorted(grouped.items()):
            values.sort(key=lambda item: item.clause_id)
            retrieval_lines = [item.retrieval_text for item in values if item.retrieval_text]
            prompt_lines = []
            for item in values:
                prefix = "[WARNING] " if categories.get(item.clause_id) == "warning" else ""
                prompt_lines.append(f"{prefix}{item.text}")
            rendered[sop_id] = {
                "clause_ids": [item.clause_id for item in values],
                "retrieval_text": "\n".join(retrieval_lines),
                "prompt_text": "\n".join(prompt_lines),
                "warning_clause_ids": [
                    item.clause_id
                    for item in values
                    if categories.get(item.clause_id) == "warning"
                ],
                "diagnostic_clause_ids": [
                    item.clause_id
                    for item in values
                    if categories.get(item.clause_id) == "diagnostic"
                ],
                "positive_clause_ids": [
                    item.clause_id
                    for item in values
                    if categories.get(item.clause_id) == "positive"
                ],
            }
        return rendered

    @staticmethod
    def _apply_token_budget(
        clauses: Iterable[SOPClauseV1],
        categories: Mapping[str, str],
        token_budget: int,
    ) -> tuple[list[SOPClauseV1], list[str]]:
        selected: list[SOPClauseV1] = []
        suppressed: list[str] = []
        remaining = max(0, int(token_budget))
        priority = {"positive": 0, "diagnostic": 1, "warning": 2}
        ordered = sorted(
            clauses,
            key=lambda clause: (
                priority.get(categories.get(clause.clause_id, "warning"), 3),
                clause.sop_id,
                clause.clause_id,
            ),
        )
        for clause in ordered:
            prefix = "[WARNING] " if categories.get(clause.clause_id) == "warning" else ""
            cost = max(1, len(re.findall(r"\S+", f"{prefix}{clause.text}")))
            if cost > remaining:
                suppressed.append(clause.clause_id)
                continue
            selected.append(clause)
            remaining -= cost
        return selected, suppressed

    def _exact_decision(
        self,
        clause: SOPClauseV1,
        request: VisibilityRequest,
    ) -> ClauseGateDecision:
        try:
            return authorize_clause_for_visibility(
                clause,
                request,
                authority_engine=self.authority_engine,
                decision_lookup=self.decision_lookup,
            )
        except Exception as error:
            claim_types = set(clause.claim_types)
            debug_safe = bool(
                request.operation == Operation.DEBUG_HYPOTHESIS
                and clause.publication_class == "diagnostic"
                and (
                    not claim_types
                    or claim_types
                    <= {
                        ClaimType.DEBUG_REPAIR.value,
                        ClaimType.AUDIT_FINDING.value,
                    }
                )
            )
            navigation_only = bool(
                request.operation == Operation.INSPECT or debug_safe
            )
            return ClauseGateDecision(
                clause.clause_id,
                navigation_only,
                navigation_only,
                f"visibility_internal_error:{type(error).__name__}",
                claim_types=clause.claim_types,
            )

    def _raw_shadow_authority_observations(
        self,
        clause: SOPClauseV1,
        request: VisibilityRequest,
    ) -> list[dict[str, Any]]:
        """Observe every raw Claim-use without changing Prompt visibility.

        Clause publication and declared operation/stage scope are Prompt-gate
        concerns.  The prospective observer must still ask Authority what would
        happen to each raw Claim under the actual request context, including a
        Claim that the Prompt gate subsequently rejects for a scope mismatch.
        """

        graph = getattr(self.authority_engine, "graph", None)
        claims = getattr(graph, "claims", {}) if graph is not None else {}
        observations: list[dict[str, Any]] = []
        for claim_ref in clause.claim_refs:
            claim = claims.get(claim_ref)
            declared_types = sorted(set(clause.claim_types))
            claim_type = (
                str(claim.claim_type.value)
                if claim is not None
                else (declared_types[0] if len(declared_types) == 1 else "")
            )
            artifact_id = (
                str(claim.subject_artifact_id)
                if claim is not None
                else (
                    str(clause.source_artifact_refs[0])
                    if len(clause.source_artifact_refs) == 1
                    else ""
                )
            )
            base = {
                "clause_id": clause.clause_id,
                "claim_id": str(claim_ref),
                "claim_type": claim_type,
                "operation": request.operation.value,
                "decision_stage": legacy_decision_stage_value(
                    StageAxes(
                        request.generation_stage,
                        request.governance_stage,
                    )
                ),
                "generation_stage": request.generation_stage.value,
                "governance_stage": request.governance_stage.value,
                "protocol_ref": request.active_protocol.key(),
                "task_id": request.task_context.task_id,
            }
            if self.authority_engine is None:
                observations.append(
                    {
                        **base,
                        "decision_id": "",
                        "outcome": "observer_unavailable",
                        "allowed": False,
                        "reason_codes": ["authority_engine_unavailable"],
                        "missing_obligations": ["authority_engine"],
                        "blocking_receipts": [],
                    }
                )
                continue
            try:
                decision = self.authority_engine.authorize(
                    AuthorityRequest(
                        artifact_id=artifact_id,
                        claim_id=str(claim_ref),
                        operation=request.operation,
                        decision_stage=None,
                        active_protocol=request.active_protocol,
                        task_context=request.task_context,
                        requesting_component=(
                            f"{request.requesting_component}:raw_shadow_observer"
                        ),
                        generation_stage=request.generation_stage,
                        governance_stage=request.governance_stage,
                    )
                )
                observations.append(
                    {
                        **base,
                        "decision_id": decision.decision_id,
                        "outcome": decision.outcome.value,
                        "allowed": decision.allowed,
                        "reason_codes": list(decision.reason_codes),
                        "missing_obligations": list(decision.missing_obligations),
                        "blocking_receipts": list(decision.blocking_receipts),
                    }
                )
            except Exception as error:
                observations.append(
                    {
                        **base,
                        "decision_id": "",
                        "outcome": "observer_error",
                        "allowed": False,
                        "reason_codes": [
                            f"authority_observer_error:{type(error).__name__}"
                        ],
                        "missing_obligations": ["authority_observer_decision"],
                        "blocking_receipts": [],
                    }
                )
        return observations

    @staticmethod
    def _flat_domain_decision(
        clause: SOPClauseV1,
        request: VisibilityRequest,
    ) -> ClauseGateDecision:
        source_domains = {
            canonical_domain(value)
            for value in (
                clause.source_domains or clause.source_task_families
            )
        }
        source_domains.discard("")
        target_domain = canonical_domain(request.task_context.task_family)
        allowed = transfer_is_compatible(
            source_domains,
            target_domain,
            clause.transfer_scope,
        )
        return ClauseGateDecision(
            clause.clause_id,
            allowed,
            False,
            (
                "flat_same_domain_or_general_universe"
                if allowed
                else "flat_domain_boundary_denied"
            ),
            claim_types=clause.claim_types,
        )

    def _authority_only_decision(
        self,
        clause: SOPClauseV1,
        request: VisibilityRequest,
    ) -> ClauseGateDecision:
        generation_values = tuple(
            GenerationStage(value)
            for value in clause.permitted_generation_stages
        ) or tuple(GenerationStage)
        governance_values = tuple(
            GovernanceStage(value)
            for value in clause.permitted_governance_stages
        ) or tuple(GovernanceStage)
        denials: list[ClauseGateDecision] = []
        for generation_stage in generation_values:
            for governance_stage in governance_values:
                marginal_request = dataclasses.replace(
                    request,
                    generation_stage=generation_stage,
                    governance_stage=governance_stage,
                )
                decision = self._exact_decision(clause, marginal_request)
                if decision.allowed:
                    return ClauseGateDecision(
                        clause.clause_id,
                        True,
                        decision.warning,
                        (
                            "authority_only_stage_marginalized:"
                            f"{generation_stage.value}/{governance_stage.value}:"
                            f"{decision.reason}"
                        ),
                        authority_decision_refs=decision.authority_decision_refs,
                        claim_types=decision.claim_types,
                    )
                denials.append(decision)
        refs = tuple(
            sorted(
                {
                    ref
                    for decision in denials
                    for ref in decision.authority_decision_refs
                }
            )
        )
        return ClauseGateDecision(
            clause.clause_id,
            False,
            False,
            "authority_only_no_authorized_stage",
            authority_decision_refs=refs,
            claim_types=clause.claim_types,
        )

    def _profile_decision(
        self,
        clause: SOPClauseV1,
        request: VisibilityRequest,
    ) -> ClauseGateDecision:
        if self.retrieval_profile == "flat_relevance_memory":
            return self._flat_domain_decision(clause, request)
        if self.retrieval_profile == "authority_only":
            return self._authority_only_decision(clause, request)
        return self._exact_decision(clause, request)

    def evaluate(
        self,
        request: VisibilityRequest,
        *,
        candidate_sop_ids: Iterable[str] | None = None,
        candidate_clause_ids: Iterable[str] | None = None,
    ) -> VisibleSOPPack:
        started = time.perf_counter()
        candidate_sop_ids = list(
            candidate_sop_ids
            if candidate_sop_ids is not None
            else self.clause_ids_by_sop
        )
        candidate_clause_ids = (
            list(candidate_clause_ids)
            if candidate_clause_ids is not None
            else None
        )
        cache_key = self._cache_key(
            request, candidate_sop_ids, candidate_clause_ids
        )
        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            result = copy.deepcopy(cached)
            result.visibility_trace["cache_hit"] = True
            result.visibility_trace["latency_ms"] = (
                time.perf_counter() - started
            ) * 1000.0
            return result

        clauses = self._candidate_clauses(
            candidate_sop_ids,
            candidate_clause_ids=candidate_clause_ids,
        )
        decisions: dict[str, ClauseGateDecision] = {}
        positive: list[SOPClauseV1] = []
        diagnostic: list[SOPClauseV1] = []
        warnings: list[SOPClauseV1] = []
        suppressed: list[str] = []
        categories: dict[str, str] = {}
        decision_refs: list[str] = []
        request_enforced = self.should_enforce(request)
        reference_exact_decisions: dict[str, ClauseGateDecision] = {}
        raw_shadow_authority_decisions: list[dict[str, Any]] = []
        for clause in clauses:
            if self.mode != "off":
                raw_shadow_authority_decisions.extend(
                    self._raw_shadow_authority_observations(clause, request)
                )
            if self.mode == "off":
                decision = ClauseGateDecision(
                    clause.clause_id,
                    True,
                    False,
                    "visibility_off",
                    claim_types=clause.claim_types,
                )
                reference_exact_decisions[clause.clause_id] = decision
            else:
                exact = self._exact_decision(clause, request)
                reference_exact_decisions[clause.clause_id] = exact
                decision = (
                    self._profile_decision(clause, request)
                    if self.retrieval_profile
                    in {"flat_relevance_memory", "authority_only"}
                    else exact
                )
            decisions[clause.clause_id] = decision

        globally_blocked_sops: set[str] = set()
        if self.retrieval_profile == "global_validity_bit":
            globally_blocked_sops = {
                clause.sop_id
                for clause in clauses
                if not decisions[clause.clause_id].allowed
            }
            for clause in clauses:
                if clause.sop_id not in globally_blocked_sops:
                    continue
                original = decisions[clause.clause_id]
                decisions[clause.clause_id] = ClauseGateDecision(
                    clause.clause_id,
                    False,
                    False,
                    f"global_sop_invalidated:{original.reason}",
                    authority_decision_refs=original.authority_decision_refs,
                    claim_types=original.claim_types or clause.claim_types,
                )

        for clause in clauses:
            decision = decisions[clause.clause_id]
            decision_refs.extend(decision.authority_decision_refs)
            if not decision.allowed:
                suppressed.append(clause.clause_id)
                continue
            category = self._category(clause, decision, request)
            categories[clause.clause_id] = category
            if category == "positive":
                positive.append(clause)
            elif category == "diagnostic":
                diagnostic.append(clause)
            else:
                warnings.append(clause)

        policy_visible = [
            clause for clause in clauses if clause.clause_id in categories
        ]
        budget_suppressed: list[str] = []
        if request_enforced:
            effective, budget_suppressed = self._apply_token_budget(
                policy_visible,
                categories,
                request.token_budget,
            )
        else:
            effective = clauses
        effective_ids = {clause.clause_id for clause in effective}
        visible_positive = [
            clause for clause in positive if clause.clause_id in effective_ids
        ]
        visible_diagnostic = [
            clause for clause in diagnostic if clause.clause_id in effective_ids
        ]
        visible_warnings = [
            clause for clause in warnings if clause.clause_id in effective_ids
        ]
        effective_categories = dict(categories)
        if self.mode in {"off", "shadow"}:
            for clause in clauses:
                effective_categories.setdefault(clause.clause_id, "warning")
        rendered = self._rendered_by_sop(effective, effective_categories)
        experience_contracts: list[dict[str, Any]] = []
        contract_errors: dict[str, str] = {}
        for clause in effective if self.mode != "off" else ():
            try:
                contract = self.contract_compiler.compile(clause, request)
            except Exception as error:
                # Contract compilation is an adoption/writeback boundary. A
                # visible warning may remain inspectable, but it receives no
                # contract and therefore cannot advance beyond L0.
                contract_errors[clause.clause_id] = type(error).__name__
                continue
            experience_contracts.append(contract.as_dict())
            rendered.setdefault(clause.sop_id, {}).setdefault(
                "experience_contract_ids", []
            ).append(contract.contract_id)
        rendered_tokens = sum(
            len(re.findall(r"\S+", row["prompt_text"]))
            for row in rendered.values()
        )
        request_payload = {
            "operation": request.operation.value,
            "generation_stage": request.generation_stage.value,
            "governance_stage": request.governance_stage.value,
            "protocol_hash": request.active_protocol.canonical_hash,
            "task_id": request.task_context.task_id,
            "bundle_version": request.memory_bundle_version,
            "policy_version": request.authority_policy_version,
            "requesting_component": request.requesting_component,
        }
        legacy_visible_ids = sorted(clause.clause_id for clause in clauses)
        full_policy_visible_ids = sorted(
            clause.clause_id for clause in policy_visible
        )
        reference_exact_visible_ids = sorted(
            clause_id
            for clause_id, decision in reference_exact_decisions.items()
            if decision.allowed
        )
        intentional_authority_bypass_ids = sorted(
            set(effective_ids) - set(reference_exact_visible_ids)
        )
        full_policy_visible_set = set(full_policy_visible_ids)
        legacy_visible_set = set(legacy_visible_ids)
        legacy_allow_authority_deny = sorted(
            legacy_visible_set - full_policy_visible_set
        )
        legacy_deny_authority_allow = sorted(
            full_policy_visible_set - legacy_visible_set
        )
        agreement_allow = sorted(
            legacy_visible_set & full_policy_visible_set
        )
        trace = {
            "mode": self.mode,
            "retrieval_profile": self.retrieval_profile,
            "effective_mode": (
                "enforce"
                if request_enforced
                else ("shadow" if self.mode == "enforce" else self.mode)
            ),
            "request_enforced": request_enforced,
            "request": request_payload,
            "cache_key": cache_key,
            "cache_hit": False,
            "candidate_sop_count": len(set(candidate_sop_ids)),
            "candidate_clause_count": len(clauses),
            "precompiled_candidate_clause_ids": (
                sorted(set(str(value) for value in candidate_clause_ids))
                if candidate_clause_ids is not None
                else None
            ),
            "legacy_visible_clause_ids": legacy_visible_ids,
            "full_policy_visible_clause_ids": full_policy_visible_ids,
            "policy_visible_clause_ids": full_policy_visible_ids,
            "reference_exact_authority_visible_clause_ids": (
                reference_exact_visible_ids
            ),
            "intentional_authority_bypass_clause_ids": (
                intentional_authority_bypass_ids
            ),
            "global_invalidated_sop_ids": sorted(globally_blocked_sops),
            "effective_visible_clause_ids": sorted(effective_ids),
            "visibility_comparison": {
                "agreement_allow_clause_ids": agreement_allow,
                "agreement_deny_clause_ids": [],
                "legacy_allow_authority_deny_clause_ids": (
                    legacy_allow_authority_deny
                ),
                "legacy_deny_authority_allow_clause_ids": (
                    legacy_deny_authority_allow
                ),
                "agreement_count": len(agreement_allow),
                "legacy_allow_authority_deny_count": len(
                    legacy_allow_authority_deny
                ),
                "legacy_deny_authority_allow_count": len(
                    legacy_deny_authority_allow
                ),
                "retained_count": len(full_policy_visible_ids),
                "suppressed_count": len(legacy_allow_authority_deny),
            },
            "embedding_candidate_clause_ids": sorted(
                clause.clause_id for clause in effective
            ),
            "rrf_eligible_clause_ids": sorted(
                clause.clause_id for clause in effective
            ),
            "authority_suppressed_clause_refs": sorted(suppressed),
            "budget_suppressed_clause_refs": sorted(budget_suppressed),
            "suppressed_clause_refs": sorted(
                set(suppressed) | set(budget_suppressed)
            ),
            "raw_shadow_authority_decisions": sorted(
                raw_shadow_authority_decisions,
                key=lambda item: (
                    str(item.get("clause_id") or ""),
                    str(item.get("claim_id") or ""),
                ),
            ),
            "clause_decisions": {
                clause_id: {
                    "allowed": decision.allowed,
                    "warning": decision.warning,
                    "reason": decision.reason,
                    "claim_types": list(decision.claim_types),
                    "authority_decision_refs": list(
                        decision.authority_decision_refs
                    ),
                    "source_run_ids": list(
                        self.clauses[clause_id].source_run_ids
                    ),
                    "source_task_ids": list(
                        self.clauses[clause_id].source_task_ids
                    ),
                    "source_task_families": list(
                        self.clauses[clause_id].source_task_families
                    ),
                    "source_domains": list(
                        self.clauses[clause_id].source_domains
                    ),
                    "transfer_scope": self.clauses[
                        clause_id
                    ].transfer_scope,
                }
                for clause_id, decision in sorted(decisions.items())
            },
            "rendered_token_count": rendered_tokens,
            "token_budget": request.token_budget,
            "empty_pack": not bool(effective),
            "experience_contract_ids": sorted(
                contract["contract_id"] for contract in experience_contracts
            ),
            "contract_compilation_errors": dict(sorted(contract_errors.items())),
            "legacy_clause_count": sum(
                str(clause.legacy_status).startswith("legacy") for clause in clauses
            ),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        request_id = cache_key
        result = VisibleSOPPack(
            request_id=request_id,
            visible_positive_clauses=visible_positive,
            visible_diagnostic_clauses=visible_diagnostic,
            warning_clauses=visible_warnings,
            suppressed_clause_refs=sorted(
                set(suppressed) | set(budget_suppressed)
            ),
            authority_decision_refs=sorted(set(decision_refs)),
            visibility_trace=trace,
            effective_clause_ids=sorted(clause.clause_id for clause in effective),
            effective_sop_ids=sorted(rendered),
            rendered_by_sop=rendered,
            experience_contracts=sorted(
                experience_contracts, key=lambda item: item["contract_id"]
            ),
        )
        with self._lock:
            self._cache[cache_key] = copy.deepcopy(result)
        return result

    def migration_report(
        self,
        request: VisibilityRequest,
        *,
        candidate_sop_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        pack = self.evaluate(request, candidate_sop_ids=candidate_sop_ids)
        candidates = set(
            str(value)
            for value in (
                candidate_sop_ids
                if candidate_sop_ids is not None
                else self.clause_ids_by_sop
            )
        )
        visible = set(pack.effective_sop_ids)
        legacy_sops = {
            clause.sop_id
            for clause in self.clauses.values()
            if clause.sop_id in candidates
            and str(clause.legacy_status).startswith("legacy")
        }
        return {
            "schema": "sop_visibility_migration_report_v1",
            "mode": self.mode,
            "operation": request.operation.value,
            "sop_count": len(candidates),
            "visible_sop_count": len(visible),
            "suppressed_sop_count": len(candidates - visible),
            "legacy_sop_count": len(legacy_sops),
            "empty_sop_count": sum(
                not self.clause_ids_by_sop.get(sop_id) for sop_id in candidates
            ),
            "visible_clause_count": len(pack.effective_clause_ids),
            "suppressed_clause_count": len(pack.suppressed_clause_refs),
            "request_enforced": pack.visibility_trace["request_enforced"],
            "legacy_visible_clause_count": len(
                pack.visibility_trace["legacy_visible_clause_ids"]
            ),
            "full_policy_visible_clause_count": len(
                pack.visibility_trace["full_policy_visible_clause_ids"]
            ),
            "visibility_comparison": copy.deepcopy(
                pack.visibility_trace["visibility_comparison"]
            ),
            "rendered_token_count": pack.visibility_trace["rendered_token_count"],
            "latency_ms": pack.visibility_trace["latency_ms"],
            "empty_pack": pack.visibility_trace["empty_pack"],
            "cache_key": pack.visibility_trace["cache_key"],
        }


__all__ = ["SOPVisibilityGateway", "VISIBILITY_MODES"]
