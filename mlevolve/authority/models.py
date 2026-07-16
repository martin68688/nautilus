from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ClaimType(StringEnum):
    EXECUTED = "executed"
    SCORE = "score"
    PAIRWISE_SUPERIORITY = "pairwise_superiority"
    CAUSAL_ATTRIBUTION = "causal_attribution"
    GENERALIZATION = "generalization"


class Operation(StringEnum):
    INSPECT = "inspect"
    DEBUG_HYPOTHESIS = "debug_hypothesis"
    GENERATE_CANDIDATE = "generate_candidate"
    RANK = "rank"
    SELECT = "select"
    PROMOTE = "promote"
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
    COUNTERFACTUAL_ACTUATION = "counterfactual_actuation"


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


@dataclass
class TaskContext:
    task_id: str
    task_family: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class AuthorityRequest:
    artifact_id: str
    claim_id: str
    operation: Operation
    decision_stage: DecisionStage
    active_protocol: ProtocolRef
    task_context: TaskContext
    requesting_component: str


@dataclass
class AuthorityScope:
    claim_types: list[str]
    operations: list[str]
    stages: list[str]
    protocol_hashes: list[str]
    task_ids: list[str]


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

    @property
    def allowed(self) -> bool:
        return self.outcome in {DecisionOutcome.ALLOW, DecisionOutcome.ALLOW_WITH_WARNING}
