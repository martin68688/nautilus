from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .stage_ontology import (
    GenerationStage,
    GovernanceStage,
    legacy_decision_stage_value,
    resolve_stage_axes,
)


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ClaimType(StringEnum):
    EXECUTED = "executed"
    SCORE = "score"
    METHOD_HYPOTHESIS = "method_hypothesis"
    DEBUG_REPAIR = "debug_repair"
    AUDIT_FINDING = "audit_finding"
    EXPERIENCE_ADOPTION = "experience_adoption"
    PAIRWISE_SUPERIORITY = "pairwise_superiority"
    CAUSAL_ATTRIBUTION = "causal_attribution"
    GENERALIZATION = "generalization"


class Operation(StringEnum):
    INSPECT = "inspect"
    DEBUG_HYPOTHESIS = "debug_hypothesis"
    GENERATE_CANDIDATE = "generate_candidate"
    RANK = "rank"
    SELECT = "select"
    # Publish the current, protocol-legal executed result as a new memory
    # subject.  This operation says nothing about whether an injected
    # experience influenced the artifact.
    PROMOTE_RESULT = "promote_result"
    # Publish an edge from an exposed historical experience to the current
    # artifact.  This requires host-verified static/runtime actuation.
    PUBLISH_ADOPTION = "publish_adoption"
    # Publish a causal attribution edge.  This additionally requires a
    # counterfactual showing that removing/replacing the experience changed
    # the action or code.
    PUBLISH_CAUSAL = "publish_causal"
    # Legacy positive-memory operation.  It keeps the old, conservative L3
    # semantics for ledger/config compatibility; new result writeback must use
    # PROMOTE_RESULT instead.
    PROMOTE = "promote"
    DISTILL_DIAGNOSTIC = "distill_diagnostic"
    DISTILL_CANDIDATE = "distill_candidate"
    DISTILL_POSITIVE_RESULT = "distill_positive_result"
    DISTILL_POSITIVE_ADOPTED = "distill_positive_adopted"
    # Legacy ambiguous operation retained only for reading old ledgers/configs.
    # New production code must choose RESULT or ADOPTED explicitly.
    DISTILL_POSITIVE = "distill_positive"
    # Legacy ledger/config value. AuthorityRequest maps it fail-closed to the
    # positive-distillation policy while retaining legacy_operation.
    DISTILL = "distill"
    CODE_SEED = "code_seed"
    REPAIR_SEED = "repair_seed"
    DERIVED_PUBLICATION = "derived_publication"


class DecisionStage(StringEnum):
    RETRIEVAL = "retrieval"
    DRAFT = "draft"
    DEBUG = "debug"
    BRANCH_SELECTION = "branch_selection"
    FUSION = "fusion"
    MEMORY_WRITEBACK = "memory_writeback"
    DISTILLATION = "distillation"
    REPLAY = "replay"


class DecisionOutcome(StringEnum):
    ALLOW = "allow"
    ALLOW_WITH_WARNING = "allow_with_warning"
    QUARANTINE = "quarantine"
    DENY = "deny"
    REQUIRE_REPLAY = "require_replay"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


class AuthorityReasonCode(StringEnum):
    PROTOCOL_VIOLATION = "protocol_violation"
    MISSING_EVIDENCE = "missing_evidence"
    UNTRUSTED_EVIDENCE = "untrusted_evidence"
    CONTRACT_MISMATCH = "contract_mismatch"
    COLLECTOR_INTERNAL_ERROR = "collector_internal_error"
    RUNTIME_FAILURE = "runtime_failure"
    TERMINAL_WRITEBACK_ERROR = "terminal_writeback_error"


class ReceiptType(StringEnum):
    CODE_EXECUTION = "code_execution"
    SPLIT_LINEAGE = "split_lineage"
    FIT_SCOPE = "fit_scope"
    PREDICTION_SCOPE = "prediction_scope"
    EVALUATOR = "evaluator"
    SELECTION_FREEZE = "selection_freeze"
    SEED_AGGREGATION = "seed_aggregation"
    REPLICATION = "replication"
    METHOD_IDENTITY = "method_identity"
    DERIVATION = "derivation"
    STATIC_ACTUATION = "static_actuation"
    RUNTIME_ACTUATION = "runtime_actuation"
    ADOPTION_PUBLICATION = "adoption_publication"
    COUNTERFACTUAL_ACTUATION = "counterfactual_actuation"
    COUNTERFACTUAL_OBSERVATION = "counterfactual_observation"


@dataclass(frozen=True)
class ProtocolRef:
    protocol_id: str
    version: str
    canonical_hash: str

    def key(self) -> str:
        return f"{self.protocol_id}@{self.version}#{self.canonical_hash}"


@dataclass
class ProtocolSpec:
    protocol_id: str
    version: str
    parent_version: str | None = None
    task_profile: dict[str, Any] = field(default_factory=dict)
    data_split_policy: dict[str, Any] = field(default_factory=dict)
    preprocessing_policy: dict[str, Any] = field(default_factory=dict)
    evaluator_spec: dict[str, Any] = field(default_factory=dict)
    metric_spec: dict[str, Any] = field(default_factory=dict)
    selection_policy: dict[str, Any] = field(default_factory=dict)
    seed_policy: dict[str, Any] = field(default_factory=dict)
    holdout_policy: dict[str, Any] = field(default_factory=dict)
    promotion_policy: dict[str, Any] = field(default_factory=dict)
    compatibility_rules: dict[str, Any] = field(default_factory=dict)
    canonical_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def ref(self) -> ProtocolRef:
        return ProtocolRef(self.protocol_id, self.version, self.canonical_hash)


@dataclass
class Claim:
    claim_id: str
    claim_type: ClaimType
    subject_artifact_id: str
    task_scope: dict[str, Any]
    method_fingerprint: str
    protocol_ref: ProtocolRef
    statement: str
    parent_claims: list[str] = field(default_factory=list)
    source_artifact_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    boundary: dict[str, Any] = field(default_factory=dict)
    legacy_status: str = "native_v1"


@dataclass
class Receipt:
    receipt_id: str
    receipt_type: ReceiptType
    artifact_id: str
    run_id: str
    protocol_hash: str
    collector_id: str
    collector_version: str
    payload_hash: str
    payload: dict[str, Any]
    timestamp: str
    parent_event_hash: str = ""
    event_hash: str = ""
    trust_status: str = "legacy_static_only"
    observation_id: str = ""
    supports_claim_types: list[str] = field(default_factory=list)
    blocks_claim_types: list[str] = field(default_factory=list)


@dataclass
class TaskContext:
    task_id: str
    task_family: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SOPClauseV1:
    clause_id: str
    sop_id: str
    text: str
    retrieval_text: str
    claim_refs: tuple[str, ...] = ()
    claim_types: tuple[str, ...] = ()
    source_artifact_refs: tuple[str, ...] = ()
    source_transition_refs: tuple[str, ...] = ()
    source_run_ids: tuple[str, ...] = ()
    source_task_ids: tuple[str, ...] = ()
    source_task_families: tuple[str, ...] = ()
    source_domains: tuple[str, ...] = ()
    transfer_scope: str = ""
    protocol_scope: tuple[str, ...] = ()
    task_scope: dict[str, Any] = field(default_factory=dict)
    permitted_operations: tuple[str, ...] = ()
    permitted_generation_stages: tuple[str, ...] = ()
    permitted_governance_stages: tuple[str, ...] = ()
    publication_class: str = "diagnostic"
    authority_decision_refs: tuple[str, ...] = ()
    receipt_refs: tuple[str, ...] = ()
    derivation_refs: tuple[str, ...] = ()
    applies_when: tuple[str, ...] = ()
    prevents: tuple[str, ...] = ()
    contract_spec: dict[str, Any] = field(default_factory=dict)
    protocol_agnostic: bool = False
    legacy_status: str = "native_v1"


@dataclass
class VisibilityRequest:
    operation: Operation
    generation_stage: GenerationStage
    governance_stage: GovernanceStage
    active_protocol: ProtocolRef
    task_context: TaskContext
    memory_bundle_version: str
    token_budget: int
    requesting_component: str
    authority_policy_version: str = "authority_v1"

    def __post_init__(self) -> None:
        self.operation = canonical_operation(self.operation)
        if not isinstance(self.generation_stage, GenerationStage):
            self.generation_stage = GenerationStage(str(self.generation_stage))
        if not isinstance(self.governance_stage, GovernanceStage):
            self.governance_stage = GovernanceStage(str(self.governance_stage))
        self.token_budget = max(0, int(self.token_budget))


@dataclass
class VisibleSOPPack:
    request_id: str
    visible_positive_clauses: list[SOPClauseV1]
    visible_diagnostic_clauses: list[SOPClauseV1]
    warning_clauses: list[SOPClauseV1]
    suppressed_clause_refs: list[str]
    authority_decision_refs: list[str]
    visibility_trace: dict[str, Any]
    effective_clause_ids: list[str] = field(default_factory=list)
    effective_sop_ids: list[str] = field(default_factory=list)
    rendered_by_sop: dict[str, dict[str, Any]] = field(default_factory=dict)
    experience_contracts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AuthorityRequest:
    artifact_id: str
    claim_id: str
    operation: Operation
    decision_stage: DecisionStage | None
    active_protocol: ProtocolRef
    task_context: TaskContext
    requesting_component: str
    generation_stage: GenerationStage | None = None
    governance_stage: GovernanceStage | None = None
    legacy_operation: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.operation, Operation):
            self.operation = Operation(str(self.operation))
        if self.operation == Operation.DISTILL:
            self.legacy_operation = Operation.DISTILL.value
            self.operation = Operation.DISTILL_POSITIVE
        axes = resolve_stage_axes(
            generation_stage=self.generation_stage,
            governance_stage=self.governance_stage,
            legacy_stage=self.decision_stage,
        )
        self.generation_stage = axes.generation_stage
        self.governance_stage = axes.governance_stage
        if self.decision_stage is None:
            self.decision_stage = DecisionStage(legacy_decision_stage_value(axes))


@dataclass
class AuthorityScope:
    claim_types: list[str]
    operations: list[str]
    stages: list[str]
    protocol_hashes: list[str]
    task_ids: list[str]
    generation_stages: list[str] = field(default_factory=list)
    governance_stages: list[str] = field(default_factory=list)


@dataclass
class AuthorityDecision:
    decision_id: str
    outcome: DecisionOutcome
    permitted_scope: AuthorityScope | None
    satisfied_paths: list[str]
    missing_obligations: list[str]
    blocking_receipts: list[str]
    required_action: str | None
    policy_version: str
    claim_id: str = ""
    artifact_id: str = ""
    operation: str = ""
    decision_stage: str = ""
    generation_stage: str = ""
    governance_stage: str = ""
    reason_codes: list[str] = field(default_factory=list)
    responsible_component: str = ""
    repairable: bool = False
    missing_receipts: list[str] = field(default_factory=list)
    missing_payloads: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.outcome in {DecisionOutcome.ALLOW, DecisionOutcome.ALLOW_WITH_WARNING}


def canonical_operation(operation: Operation | str) -> Operation:
    value = operation if isinstance(operation, Operation) else Operation(str(operation))
    return Operation.DISTILL_POSITIVE if value == Operation.DISTILL else value
