from __future__ import annotations

import dataclasses
import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Iterable, Mapping

from .domain_scope import canonical_domain, normalize_transfer_scope
from .models import ProtocolRef, Receipt, SOPClauseV1, VisibilityRequest
from .protocol_registry import canonical_json


ACTUATION_REPORT_SCHEMA = "experience_actuation_report_v1"
EXPERIENCE_CONTRACT_SCHEMA = "experience_contract_v1"
EXPERIENCE_EXPOSURE_EVENT_SCHEMA = "experience_exposure_event_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(_jsonable(value)).encode("utf-8")).hexdigest()


class ActuationLevel(IntEnum):
    EXPOSED = 0
    CLAIMED_ADOPTION = 1
    STATIC_CONFORMANT = 2
    RUNTIME_CONFORMANT = 3
    CAUSAL_CONFIRMED = 4
    EFFECTIVE = 5


ACTUATION_LEVEL_NAMES = {
    ActuationLevel.EXPOSED: "EXPOSED",
    ActuationLevel.CLAIMED_ADOPTION: "CLAIMED_ADOPTION",
    ActuationLevel.STATIC_CONFORMANT: "STATIC_CONFORMANT",
    ActuationLevel.RUNTIME_CONFORMANT: "RUNTIME_CONFORMANT",
    ActuationLevel.CAUSAL_CONFIRMED: "CAUSAL_CONFIRMED",
    ActuationLevel.EFFECTIVE: "EFFECTIVE",
}


@dataclass(frozen=True)
class Predicate:
    """A host-observable equality check used by an ExperienceContract.

    Natural-language guidance is retained in ``description``. It never counts
    as satisfied by itself: the host must provide an observation under
    ``name`` whose value equals ``expected``.
    """

    name: str
    expected: Any = True
    description: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("Predicate name is required")

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_value(
        cls,
        value: Predicate | Mapping[str, Any] | str,
        *,
        prefix: str,
    ) -> Predicate:
        if isinstance(value, Predicate):
            return value
        if isinstance(value, Mapping):
            name = str(value.get("name") or "").strip()
            description = str(value.get("description") or "").strip()
            if not name:
                source = description or canonical_json(_jsonable(value))
                name = f"{prefix}::{hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]}"
            return cls(
                name=name,
                expected=value.get("expected", True),
                description=description,
            )
        description = str(value).strip()
        if not description:
            raise ValueError(f"Empty {prefix} predicate")
        digest = hashlib.sha256(description.encode("utf-8")).hexdigest()[:16]
        return cls(
            name=f"{prefix}::{digest}",
            expected=True,
            description=description,
        )


def _predicates(
    values: Iterable[Predicate | Mapping[str, Any] | str] | None,
    *,
    prefix: str,
) -> list[Predicate]:
    output: list[Predicate] = []
    seen: set[str] = set()
    for value in values or []:
        predicate = Predicate.from_value(value, prefix=prefix)
        if predicate.name in seen:
            continue
        seen.add(predicate.name)
        output.append(predicate)
    return output


@dataclass
class ExperienceContract:
    # Keep these five fields first for compatibility with the original small
    # constructor used by downstream callers.
    preconditions: list[Predicate] = field(default_factory=list)
    must_preserve: list[Predicate] = field(default_factory=list)
    must_change: list[Predicate] = field(default_factory=list)
    must_not_use: list[Predicate] = field(default_factory=list)
    expected_runtime_observations: list[Predicate] = field(default_factory=list)
    clause_id: str = ""
    sop_id: str = ""
    claim_refs: list[str] = field(default_factory=list)
    source_artifact_refs: list[str] = field(default_factory=list)
    source_transition_refs: list[str] = field(default_factory=list)
    source_run_ids: list[str] = field(default_factory=list)
    source_task_ids: list[str] = field(default_factory=list)
    source_task_families: list[str] = field(default_factory=list)
    source_domains: list[str] = field(default_factory=list)
    transfer_scope: str = ""
    active_protocol_ref: str = ""
    task_scope: dict[str, Any] = field(default_factory=dict)
    target_task_id: str = ""
    target_task_family: str = ""
    target_domain: str = ""
    operation: str = ""
    generation_stage: str = ""
    governance_stage: str = ""
    publication_class: str = "diagnostic"
    minimum_writeback_level: int = int(ActuationLevel.RUNTIME_CONFORMANT)
    policy_version: str = "authority_v1"
    compiler_version: str = "experience_contract_compiler_v1"
    contract_id: str = ""
    contract_hash: str = ""
    schema: str = EXPERIENCE_CONTRACT_SCHEMA

    def _normalized_predicates(self) -> None:
        for field_name in (
            "preconditions",
            "must_preserve",
            "must_change",
            "must_not_use",
            "expected_runtime_observations",
        ):
            setattr(
                self,
                field_name,
                _predicates(getattr(self, field_name), prefix=field_name),
            )

    def payload_for_hash(self) -> dict[str, Any]:
        self._normalized_predicates()
        payload = self.as_dict()
        payload.pop("contract_id", None)
        payload.pop("contract_hash", None)
        return payload

    def finalize(self) -> ExperienceContract:
        digest = _sha256_json(self.payload_for_hash())
        self.contract_hash = digest
        self.contract_id = f"experience_contract::{digest[:24]}"
        return self

    def verify(self) -> None:
        if self.schema != EXPERIENCE_CONTRACT_SCHEMA:
            raise ValueError(f"Unsupported ExperienceContract schema: {self.schema}")
        expected_hash = _sha256_json(self.payload_for_hash())
        if self.contract_hash != expected_hash:
            raise ValueError("ExperienceContract hash mismatch")
        if self.contract_id != f"experience_contract::{expected_hash[:24]}":
            raise ValueError("ExperienceContract ID mismatch")

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, verify: bool = True) -> ExperienceContract:
        values = dict(payload)
        for field_name in (
            "preconditions",
            "must_preserve",
            "must_change",
            "must_not_use",
            "expected_runtime_observations",
        ):
            values[field_name] = _predicates(
                values.get(field_name), prefix=field_name
            )
        contract = cls(**values)
        if verify:
            contract.verify()
        return contract


class ExperienceContractCompiler:
    """Compile an admitted SOP clause into host-checkable obligations.

    The compiler never treats clause prose as proof. Descriptions are mapped to
    stable predicate names that remain unsatisfied until host instrumentation
    records matching observations.
    """

    def __init__(
        self,
        *,
        compiler_version: str = "experience_contract_compiler_v1",
    ) -> None:
        self.compiler_version = str(compiler_version)

    @staticmethod
    def _field(clause: SOPClauseV1 | Mapping[str, Any], name: str, default: Any = None) -> Any:
        if isinstance(clause, Mapping):
            return clause.get(name, default)
        return getattr(clause, name, default)

    @staticmethod
    def _explicit_spec(clause: SOPClauseV1 | Mapping[str, Any]) -> Mapping[str, Any]:
        value = ExperienceContractCompiler._field(clause, "contract_spec", {})
        return value if isinstance(value, Mapping) else {}

    def compile(
        self,
        clause: SOPClauseV1 | Mapping[str, Any],
        request: VisibilityRequest,
    ) -> ExperienceContract:
        clause_id = str(self._field(clause, "clause_id", "") or "")
        if not clause_id:
            raise ValueError("Cannot compile a contract without clause_id")
        spec = self._explicit_spec(clause)
        applies_raw = self._field(clause, "applies_when", ()) or ()
        prevents_raw = self._field(clause, "prevents", ()) or ()
        applies_when = (
            [applies_raw] if isinstance(applies_raw, str) else list(applies_raw)
        )
        prevents = (
            [prevents_raw] if isinstance(prevents_raw, str) else list(prevents_raw)
        )

        preconditions = _predicates(
            [*applies_when, *(spec.get("preconditions") or [])],
            prefix="precondition",
        )
        must_preserve = _predicates(
            [
                {
                    "name": "active_protocol_ref",
                    "expected": request.active_protocol.key(),
                    "description": "The active protocol must not change while applying the experience.",
                },
                {
                    "name": "task_id",
                    "expected": request.task_context.task_id,
                    "description": "The experience must remain inside the admitted task scope.",
                },
                *(spec.get("must_preserve") or []),
            ],
            prefix="must_preserve",
        )
        must_change = _predicates(
            [
                {
                    "name": f"clause_applied::{clause_id}",
                    "expected": True,
                    "description": str(self._field(clause, "text", "") or "Apply the admitted clause."),
                },
                *(spec.get("must_change") or []),
            ],
            prefix="must_change",
        )
        must_not_use = _predicates(
            [
                {
                    "name": "forbidden_dependency_count",
                    "expected": 0,
                    "description": "No forbidden data, label, evaluator, or holdout dependency may be introduced.",
                },
                {
                    "name": "holdout_used_for_selection",
                    "expected": False,
                    "description": "Holdout outcomes may not drive candidate or hyperparameter selection.",
                },
                *(spec.get("must_not_use") or []),
            ],
            prefix="must_not_use",
        )
        runtime = _predicates(
            [
                {
                    "name": "target_path_executed",
                    "expected": True,
                    "description": "The host must observe the target implementation path executing.",
                },
                *[
                    {
                        "name": f"prevented::{hashlib.sha256(str(item).encode('utf-8')).hexdigest()[:16]}",
                        "expected": True,
                        "description": str(item),
                    }
                    for item in prevents
                    if str(item).strip()
                ],
                *(spec.get("expected_runtime_observations") or []),
            ],
            prefix="expected_runtime_observation",
        )
        publication_class = str(
            self._field(clause, "publication_class", "diagnostic") or "diagnostic"
        )
        task_scope = dict(self._field(clause, "task_scope", {}) or {})
        source_task_ids = [
            str(value)
            for value in self._field(clause, "source_task_ids", ()) or ()
        ]
        if not source_task_ids:
            source_task_ids = [
                str(value)
                for value in (
                    task_scope.get("task_ids")
                    or [task_scope.get("task_id")]
                )
                if value not in {None, ""}
            ]
        source_task_families = [
            str(value)
            for value in self._field(
                clause, "source_task_families", ()
            )
            or ()
        ]
        if not source_task_families:
            source_task_families = [
                str(value)
                for value in task_scope.get("task_families") or []
                if value not in {None, ""}
            ]
        source_domains = [
            canonical_domain(value)
            for value in self._field(clause, "source_domains", ()) or ()
        ]
        source_domains = sorted({value for value in source_domains if value})
        if not source_domains:
            source_domains = sorted(
                {
                    canonical_domain(value)
                    for value in source_task_families
                    if canonical_domain(value)
                }
            )
        minimum = {
            "diagnostic": int(ActuationLevel.EXPOSED),
            "candidate": int(ActuationLevel.STATIC_CONFORMANT),
            "certified": int(ActuationLevel.RUNTIME_CONFORMANT),
        }.get(publication_class, int(ActuationLevel.RUNTIME_CONFORMANT))
        contract = ExperienceContract(
            preconditions=preconditions,
            must_preserve=must_preserve,
            must_change=must_change,
            must_not_use=must_not_use,
            expected_runtime_observations=runtime,
            clause_id=clause_id,
            sop_id=str(self._field(clause, "sop_id", "") or ""),
            claim_refs=[str(value) for value in self._field(clause, "claim_refs", ()) or ()],
            source_artifact_refs=[
                str(value)
                for value in self._field(clause, "source_artifact_refs", ()) or ()
            ],
            source_transition_refs=[
                str(value)
                for value in self._field(
                    clause, "source_transition_refs", ()
                )
                or ()
            ],
            source_run_ids=[
                str(value)
                for value in self._field(clause, "source_run_ids", ()) or ()
            ],
            source_task_ids=sorted(set(source_task_ids)),
            source_task_families=sorted(set(source_task_families)),
            source_domains=source_domains,
            transfer_scope=normalize_transfer_scope(
                self._field(clause, "transfer_scope", "")
            ),
            active_protocol_ref=request.active_protocol.key(),
            task_scope=task_scope,
            target_task_id=request.task_context.task_id,
            target_task_family=request.task_context.task_family,
            target_domain=canonical_domain(request.task_context.task_family),
            operation=request.operation.value,
            generation_stage=request.generation_stage.value,
            governance_stage=request.governance_stage.value,
            publication_class=publication_class,
            minimum_writeback_level=minimum,
            policy_version=request.authority_policy_version,
            compiler_version=self.compiler_version,
        )
        return contract.finalize()


def evaluate_predicates(
    predicates: Iterable[Predicate], observations: Mapping[str, Any]
) -> list[str]:
    return [
        item.name
        for item in predicates
        if item.name not in observations or observations.get(item.name) != item.expected
    ]


@dataclass(frozen=True)
class ActuationLevelResult:
    level: int
    name: str
    reached: bool
    evidence_refs: tuple[str, ...] = ()
    missing_obligations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass
class ActuationReport:
    artifact_id: str
    contract_id: str
    contract_hash: str
    active_protocol_ref: str
    levels: list[ActuationLevelResult]
    highest_level: int | None
    promotion_eligible: bool
    clause_id: str = ""
    sop_id: str = ""
    claim_refs: list[str] = field(default_factory=list)
    source_artifact_refs: list[str] = field(default_factory=list)
    source_transition_refs: list[str] = field(default_factory=list)
    source_run_ids: list[str] = field(default_factory=list)
    source_task_ids: list[str] = field(default_factory=list)
    source_task_families: list[str] = field(default_factory=list)
    source_domains: list[str] = field(default_factory=list)
    transfer_scope: str = ""
    task_scope: dict[str, Any] = field(default_factory=dict)
    target_scope: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=_utc_now)
    report_id: str = ""
    report_hash: str = ""
    schema: str = ACTUATION_REPORT_SCHEMA

    def finalize(self) -> ActuationReport:
        payload = self.as_dict()
        payload.pop("report_id", None)
        payload.pop("report_hash", None)
        digest = _sha256_json(payload)
        self.report_hash = digest
        self.report_id = f"actuation_report::{digest[:24]}"
        return self

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def reached(self, level: ActuationLevel) -> bool:
        return any(row.level == int(level) and row.reached for row in self.levels)


def _pair_value(pair_result: Mapping[str, Any] | Any | None, key: str, default: Any = None) -> Any:
    if pair_result is None:
        return default
    if isinstance(pair_result, Mapping):
        return pair_result.get(key, default)
    return getattr(pair_result, key, default)


def build_actuation_report(
    contract: ExperienceContract,
    *,
    artifact_id: str,
    exposure_event_hash: str = "",
    claimed: bool = False,
    precondition_observations: Mapping[str, Any] | None = None,
    static_observations: Mapping[str, Any] | None = None,
    runtime_observations: Mapping[str, Any] | None = None,
    pair_result: Mapping[str, Any] | Any | None = None,
    static_receipt_refs: Iterable[str] = (),
    runtime_receipt_refs: Iterable[str] = (),
    counterfactual_receipt_refs: Iterable[str] = (),
    generated_at: str | None = None,
) -> ActuationReport:
    contract.verify()
    level_rows: list[ActuationLevelResult] = []

    l0 = bool(exposure_event_hash)
    level_rows.append(
        ActuationLevelResult(
            int(ActuationLevel.EXPOSED),
            ACTUATION_LEVEL_NAMES[ActuationLevel.EXPOSED],
            l0,
            (exposure_event_hash,) if l0 else (),
            () if l0 else ("prompt_exposure_event",),
        )
    )

    l1 = bool(l0 and claimed)
    level_rows.append(
        ActuationLevelResult(
            int(ActuationLevel.CLAIMED_ADOPTION),
            ACTUATION_LEVEL_NAMES[ActuationLevel.CLAIMED_ADOPTION],
            l1,
            (contract.contract_id,) if l1 else (),
            () if l1 else (("explicit_adoption_claim",) if l0 else ("level:L0",)),
        )
    )

    preconditions = precondition_observations or {}
    static = static_observations or {}
    static_predicates = [
        *contract.must_preserve,
        *contract.must_change,
        *contract.must_not_use,
    ]
    missing_static = [
        *evaluate_predicates(contract.preconditions, preconditions),
        *evaluate_predicates(static_predicates, static),
    ]
    if static_observations is None:
        missing_static.append("host_static_observation")
    if not static_predicates:
        missing_static.append("contract_static_obligations")
    l2 = bool(l1 and not missing_static)
    level_rows.append(
        ActuationLevelResult(
            int(ActuationLevel.STATIC_CONFORMANT),
            ACTUATION_LEVEL_NAMES[ActuationLevel.STATIC_CONFORMANT],
            l2,
            tuple(sorted(set(str(value) for value in static_receipt_refs))) if l2 else (),
            () if l2 else tuple(sorted(set(missing_static or ["level:L1"]))),
        )
    )

    runtime = runtime_observations or {}
    missing_runtime = evaluate_predicates(
        contract.expected_runtime_observations, runtime
    )
    if runtime_observations is None:
        missing_runtime.append("host_runtime_observation")
    if not contract.expected_runtime_observations:
        missing_runtime.append("contract_runtime_obligations")
    l3 = bool(l2 and not missing_runtime)
    level_rows.append(
        ActuationLevelResult(
            int(ActuationLevel.RUNTIME_CONFORMANT),
            ACTUATION_LEVEL_NAMES[ActuationLevel.RUNTIME_CONFORMANT],
            l3,
            tuple(sorted(set(str(value) for value in runtime_receipt_refs))) if l3 else (),
            () if l3 else tuple(sorted(set(missing_runtime or ["level:L2"]))),
        )
    )

    influence = _pair_value(pair_result, "influence_confirmed", False) is True
    l4 = bool(l3 and influence)
    counterfactual_refs = tuple(
        sorted(set(str(value) for value in counterfactual_receipt_refs))
    )
    level_rows.append(
        ActuationLevelResult(
            int(ActuationLevel.CAUSAL_CONFIRMED),
            ACTUATION_LEVEL_NAMES[ActuationLevel.CAUSAL_CONFIRMED],
            l4,
            counterfactual_refs if l4 else (),
            () if l4 else (("paired_influence_counterfactual",) if l3 else ("level:L3",)),
        )
    )

    effective = _pair_value(pair_result, "effective", False) is True
    protocol_legal = _pair_value(pair_result, "protocol_legal", False) is True
    l5 = bool(l4 and effective and protocol_legal)
    missing_l5: tuple[str, ...] = ()
    if not l5:
        if not l4:
            missing_l5 = ("level:L4",)
        else:
            missing_l5 = tuple(
                item
                for item, satisfied in (
                    ("paired_efficacy_counterfactual", effective),
                    ("protocol_legal_outcomes", protocol_legal),
                )
                if not satisfied
            )
    level_rows.append(
        ActuationLevelResult(
            int(ActuationLevel.EFFECTIVE),
            ACTUATION_LEVEL_NAMES[ActuationLevel.EFFECTIVE],
            l5,
            counterfactual_refs if l5 else (),
            missing_l5,
        )
    )

    reached = [row.level for row in level_rows if row.reached]
    report = ActuationReport(
        artifact_id=str(artifact_id),
        contract_id=contract.contract_id,
        contract_hash=contract.contract_hash,
        active_protocol_ref=contract.active_protocol_ref,
        levels=level_rows,
        highest_level=max(reached) if reached else None,
        promotion_eligible=bool(
            int(ActuationLevel.RUNTIME_CONFORMANT) in reached
        ),
        clause_id=contract.clause_id,
        sop_id=contract.sop_id,
        claim_refs=list(contract.claim_refs),
        source_artifact_refs=list(contract.source_artifact_refs),
        source_transition_refs=list(contract.source_transition_refs),
        source_run_ids=list(contract.source_run_ids),
        source_task_ids=list(contract.source_task_ids),
        source_task_families=list(contract.source_task_families),
        source_domains=list(contract.source_domains),
        transfer_scope=contract.transfer_scope,
        task_scope=dict(contract.task_scope),
        target_scope={
            "task_id": contract.target_task_id,
            "task_family": contract.target_task_family,
            "domain": contract.target_domain,
        },
        generated_at=generated_at or _utc_now(),
    )
    return report.finalize()


def classify_actuation(
    contract: ExperienceContract,
    *,
    exposed: bool,
    claimed: bool,
    static_observations: dict[str, Any] | None = None,
    runtime_observations: dict[str, Any] | None = None,
    counterfactual_effect: float | None = None,
) -> ActuationLevel | None:
    """Compatibility wrapper returning the highest rigorously reached level."""

    if not contract.contract_hash:
        contract.finalize()
    pair_result = None
    if counterfactual_effect is not None:
        pair_result = {
            "influence_confirmed": counterfactual_effect != 0,
            "effective": counterfactual_effect > 0,
            "protocol_legal": True,
        }
    report = build_actuation_report(
        contract,
        artifact_id="compatibility-wrapper",
        exposure_event_hash="compatibility-exposure" if exposed else "",
        claimed=claimed,
        precondition_observations=static_observations,
        static_observations=static_observations,
        runtime_observations=runtime_observations,
        pair_result=pair_result,
    )
    return (
        ActuationLevel(report.highest_level)
        if report.highest_level is not None
        else None
    )


@dataclass
class _TrackedContractState:
    contract: ExperienceContract
    exposure_event_hash: str = ""
    claimed: bool = False
    precondition_observations: dict[str, Any] | None = None
    static_observations: dict[str, Any] | None = None
    runtime_observations: dict[str, Any] | None = None
    pair_result: dict[str, Any] | None = None
    receipts: list[Receipt] = field(default_factory=list)
    updated_at: str = ""


class ActuationTracker:
    """Run-scoped, host-owned L0–L5 evidence tracker.

    It records exposure separately from adoption, and mints trusted receipts
    only after the corresponding contract predicates are satisfied.
    """

    def __init__(
        self,
        *,
        collector_host: Any,
        protocol_ref: ProtocolRef,
        run_id: str,
        ledger: Any | None = None,
    ) -> None:
        self.collector_host = collector_host
        self.protocol_ref = protocol_ref
        self.run_id = str(run_id)
        self.ledger = ledger
        self._lock = threading.RLock()
        self._states: dict[tuple[str, str], _TrackedContractState] = {}

    def _append_ledger(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.ledger is not None:
            self.ledger.append(event_type, _jsonable(payload))

    def _state(self, artifact_id: str, contract_id: str) -> _TrackedContractState:
        key = (str(artifact_id), str(contract_id))
        if key not in self._states:
            raise KeyError(f"Unknown tracked contract: {key}")
        return self._states[key]

    def record_exposure(
        self,
        *,
        artifact_id: str,
        contracts: Iterable[ExperienceContract | Mapping[str, Any]],
        request_id: str,
        prompt_sha256: str = "",
    ) -> list[str]:
        output: list[str] = []
        with self._lock:
            for value in contracts:
                contract = (
                    value
                    if isinstance(value, ExperienceContract)
                    else ExperienceContract.from_dict(value)
                )
                contract.verify()
                if contract.active_protocol_ref != self.protocol_ref.key():
                    raise ValueError("Exposure contract protocol mismatch")
                payload = {
                    "schema": EXPERIENCE_EXPOSURE_EVENT_SCHEMA,
                    "artifact_id": str(artifact_id),
                    "contract_id": contract.contract_id,
                    "contract_hash": contract.contract_hash,
                    "clause_id": contract.clause_id,
                    "sop_id": contract.sop_id,
                    "claim_refs": list(contract.claim_refs),
                    "source_refs": sorted(
                        set(contract.source_artifact_refs)
                        | set(contract.source_transition_refs)
                    ),
                    "source_artifact_refs": list(
                        contract.source_artifact_refs
                    ),
                    "source_transition_refs": list(
                        contract.source_transition_refs
                    ),
                    "source_run_ids": list(contract.source_run_ids),
                    "source_task_ids": list(contract.source_task_ids),
                    "source_task_families": list(
                        contract.source_task_families
                    ),
                    "source_domains": list(contract.source_domains),
                    "transfer_scope": contract.transfer_scope,
                    "task_scope": dict(contract.task_scope),
                    "target_scope": {
                        "task_id": contract.target_task_id,
                        "task_family": contract.target_task_family,
                        "domain": contract.target_domain,
                    },
                    "operation": contract.operation,
                    "generation_stage": contract.generation_stage,
                    "governance_stage": contract.governance_stage,
                    "publication_class": contract.publication_class,
                    "minimum_writeback_level": contract.minimum_writeback_level,
                    "policy_version": contract.policy_version,
                    "compiler_version": contract.compiler_version,
                    "request_id": str(request_id),
                    "prompt_sha256": str(prompt_sha256),
                    "protocol_ref": self.protocol_ref.key(),
                }
                event_hash = _sha256_json(payload)
                key = (str(artifact_id), contract.contract_id)
                state = self._states.get(key)
                if state is None:
                    state = _TrackedContractState(contract=contract)
                    self._states[key] = state
                elif state.contract.contract_hash != contract.contract_hash:
                    raise ValueError("Tracked ExperienceContract is immutable")
                state.exposure_event_hash = event_hash
                state.updated_at = _utc_now()
                output.append(contract.contract_id)
                self._append_ledger("experience_exposed", {**payload, "event_hash": event_hash})
        return output

    def record_claimed_adoption(self, *, artifact_id: str, contract_id: str) -> None:
        with self._lock:
            state = self._state(artifact_id, contract_id)
            if not state.exposure_event_hash:
                raise ValueError("Cannot claim adoption before exposure")
            state.claimed = True
            state.updated_at = _utc_now()
            self._append_ledger(
                "experience_adoption_claimed",
                {"artifact_id": str(artifact_id), "contract_id": contract_id},
            )

    def record_static_observation(
        self,
        *,
        artifact_id: str,
        contract_id: str,
        preconditions: Mapping[str, Any],
        observations: Mapping[str, Any],
        source: str = "host.static_actuation",
    ) -> Receipt | None:
        from .collectors import StaticActuationCollector

        with self._lock:
            state = self._state(artifact_id, contract_id)
            state.precondition_observations = dict(preconditions)
            state.static_observations = dict(observations)
            state.updated_at = _utc_now()
            predicates = [
                *state.contract.must_preserve,
                *state.contract.must_change,
                *state.contract.must_not_use,
            ]
            missing = [
                *evaluate_predicates(state.contract.preconditions, preconditions),
                *evaluate_predicates(predicates, observations),
            ]
            if missing or not state.claimed or not predicates:
                self._append_ledger(
                    "static_actuation_rejected",
                    {
                        "artifact_id": str(artifact_id),
                        "contract_id": contract_id,
                        "missing_obligations": sorted(set(missing)),
                    },
                )
                return None
            receipt = self.collector_host.collect(
                StaticActuationCollector,
                artifact_id=str(artifact_id),
                run_id=self.run_id,
                protocol_ref=self.protocol_ref,
                source=source,
                payload={
                    "contract_hash": state.contract.contract_hash,
                    "checks": {predicate.name: True for predicate in predicates},
                },
            )
            state.receipts.append(receipt)
            self._append_ledger("static_actuation_verified", dataclasses.asdict(receipt))
            return receipt

    def record_runtime_observation(
        self,
        *,
        artifact_id: str,
        contract_id: str,
        observations: Mapping[str, Any],
        source: str = "host.runtime_actuation",
    ) -> Receipt | None:
        from .collectors import RuntimeActuationCollector

        with self._lock:
            state = self._state(artifact_id, contract_id)
            state.runtime_observations = dict(observations)
            state.updated_at = _utc_now()
            missing = evaluate_predicates(
                state.contract.expected_runtime_observations, observations
            )
            static_verified = any(
                receipt.receipt_type.value == "static_actuation"
                for receipt in state.receipts
            )
            if missing or not static_verified or not state.contract.expected_runtime_observations:
                self._append_ledger(
                    "runtime_actuation_rejected",
                    {
                        "artifact_id": str(artifact_id),
                        "contract_id": contract_id,
                        "missing_obligations": sorted(set(missing)),
                        "static_verified": static_verified,
                    },
                )
                return None
            observation_hash = _sha256_json(dict(observations))
            receipt = self.collector_host.collect(
                RuntimeActuationCollector,
                artifact_id=str(artifact_id),
                run_id=self.run_id,
                protocol_ref=self.protocol_ref,
                source=source,
                payload={
                    "contract_hash": state.contract.contract_hash,
                    "event_hashes": [observation_hash],
                    "target_path_executed": observations.get("target_path_executed") is True,
                    "observations_hash": observation_hash,
                },
            )
            state.receipts.append(receipt)
            self._append_ledger("runtime_actuation_verified", dataclasses.asdict(receipt))
            return receipt

    def record_counterfactual(
        self,
        *,
        artifact_id: str,
        contract_id: str,
        pair_result: Mapping[str, Any] | Any,
        source: str = "host.counterfactual_actuation",
    ) -> Receipt | None:
        from .collectors import CounterfactualCollector

        with self._lock:
            state = self._state(artifact_id, contract_id)
            result = (
                dict(pair_result)
                if isinstance(pair_result, Mapping)
                else _jsonable(pair_result)
            )
            state.pair_result = dict(result)
            state.updated_at = _utc_now()
            runtime_verified = any(
                receipt.receipt_type.value == "runtime_actuation"
                for receipt in state.receipts
            )
            if not runtime_verified or result.get("influence_confirmed") is not True:
                self._append_ledger(
                    "counterfactual_actuation_rejected",
                    {
                        "artifact_id": str(artifact_id),
                        "contract_id": contract_id,
                        "runtime_verified": runtime_verified,
                        "influence_confirmed": result.get("influence_confirmed") is True,
                    },
                )
                return None
            receipt = self.collector_host.collect(
                CounterfactualCollector,
                artifact_id=str(artifact_id),
                run_id=self.run_id,
                protocol_ref=self.protocol_ref,
                source=source,
                payload={
                    "contract_hash": state.contract.contract_hash,
                    "pair_id": str(result.get("pair_id") or ""),
                    "control_hash": str(result.get("control_hash") or ""),
                    "memory_on_action_hash": str(result.get("memory_on_action_hash") or ""),
                    "memory_off_action_hash": str(result.get("memory_off_action_hash") or ""),
                    "memory_on_code_hash": str(result.get("memory_on_code_hash") or ""),
                    "memory_off_code_hash": str(result.get("memory_off_code_hash") or ""),
                    "action_or_code_changed": True,
                    "protocol_legal": result.get("protocol_legal") is True,
                    "effective": result.get("effective") is True,
                    "outcome_delta": result.get("outcome_delta"),
                    "metric_direction": result.get("metric_direction"),
                },
            )
            state.receipts.append(receipt)
            self._append_ledger("counterfactual_actuation_verified", dataclasses.asdict(receipt))
            return receipt

    def report(
        self,
        *,
        artifact_id: str,
        contract_id: str,
        emit_ledger: bool = True,
    ) -> ActuationReport:
        with self._lock:
            state = self._state(artifact_id, contract_id)
            static_refs = [
                receipt.receipt_id
                for receipt in state.receipts
                if receipt.receipt_type.value == "static_actuation"
            ]
            runtime_refs = [
                receipt.receipt_id
                for receipt in state.receipts
                if receipt.receipt_type.value == "runtime_actuation"
            ]
            counterfactual_refs = [
                receipt.receipt_id
                for receipt in state.receipts
                if receipt.receipt_type.value == "counterfactual_actuation"
            ]
            report = build_actuation_report(
                state.contract,
                artifact_id=str(artifact_id),
                exposure_event_hash=state.exposure_event_hash,
                claimed=state.claimed,
                precondition_observations=state.precondition_observations,
                static_observations=state.static_observations,
                runtime_observations=state.runtime_observations,
                pair_result=state.pair_result,
                static_receipt_refs=static_refs,
                runtime_receipt_refs=runtime_refs,
                counterfactual_receipt_refs=counterfactual_refs,
                generated_at=state.updated_at or _utc_now(),
            )
            if emit_ledger:
                self._append_ledger("actuation_report", report.as_dict())
            return report

    def reports_for_artifact(self, artifact_id: str) -> list[ActuationReport]:
        with self._lock:
            contract_ids = sorted(
                contract_id
                for candidate_artifact, contract_id in self._states
                if candidate_artifact == str(artifact_id)
            )
        return [
            self.report(artifact_id=str(artifact_id), contract_id=contract_id)
            for contract_id in contract_ids
        ]

    def receipts_for_artifact(self, artifact_id: str) -> list[Receipt]:
        with self._lock:
            by_id: dict[str, Receipt] = {}
            for (candidate_artifact, _contract_id), state in self._states.items():
                if candidate_artifact != str(artifact_id):
                    continue
                for receipt in state.receipts:
                    by_id[receipt.receipt_id] = receipt
            return [by_id[key] for key in sorted(by_id)]

    def contracts_for_artifact(self, artifact_id: str) -> list[ExperienceContract]:
        """Return immutable contracts bound to one runtime artifact."""

        with self._lock:
            return [
                self._states[key].contract
                for key in sorted(self._states)
                if key[0] == str(artifact_id)
            ]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            keys = sorted(self._states)
        reports = [
            self.report(
                artifact_id=artifact_id,
                contract_id=contract_id,
                emit_ledger=False,
            ).as_dict()
            for artifact_id, contract_id in keys
        ]
        return {
            "schema": "actuation_tracker_snapshot_v1",
            "run_id": self.run_id,
            "active_protocol_ref": self.protocol_ref.key(),
            "report_count": len(reports),
            "highest_level_counts": {
                str(level): sum(
                    report.get("highest_level") == level for report in reports
                )
                for level in range(6)
            },
            "unexposed_count": sum(
                report.get("highest_level") is None for report in reports
            ),
            "reports": reports,
        }


__all__ = [
    "ACTUATION_LEVEL_NAMES",
    "ACTUATION_REPORT_SCHEMA",
    "ActuationLevel",
    "ActuationLevelResult",
    "ActuationReport",
    "ActuationTracker",
    "ExperienceContract",
    "ExperienceContractCompiler",
    "Predicate",
    "build_actuation_report",
    "classify_actuation",
    "evaluate_predicates",
]
