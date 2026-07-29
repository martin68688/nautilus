"""Host-owned protocol runtime primitives."""

from .data_views import (
    DATA_VIEW_MANIFEST_SCHEMA,
    EVALUATOR_LAUNCH_CONTRACT_SCHEMA,
    TRAINING_MOUNT_CONTRACT_SCHEMA,
    DataViewManifest,
    build_evaluator_launch_contract,
    build_training_mount_contract,
    materialize_data_views,
    read_data_view_manifest,
    verify_data_view_manifest,
)
from .collector import (
    HostCollectorIdentity,
    HostCollectorSidecar,
    verify_collector_artifacts,
)
from .collector_bridge import bridge_signed_journal_to_receipts
from .session import ProtocolSession, activate_session, current_session
from .views import DataViewHandle, ProtocolSplit, build_view_handles
from .reference_adapters import (
    FRAMEWORK_CANDIDATES,
    INVALID_CANDIDATES,
    ProtocolReference,
    protocol_references,
)
from .preflight import (
    PreflightStatus,
    ProtocolPreflightRunner,
    build_bounded_repair_receipt,
    preflight_cache_key,
    static_compatibility_check,
    validate_preflight_admission,
)
from .closure import (
    build_training_evidence_manifest,
    dry_evidence_closure,
    preterminal_evidence_closure,
)
from .rollout import (
    ProtocolRuntimeMode,
    ProtocolRolloutStage,
    aggregate_shadow_reports,
    build_dual_observer_report,
    build_rollback_receipt,
    validate_protocol_runtime_mode,
    validate_rollout_transition,
)

__all__ = [
    "DATA_VIEW_MANIFEST_SCHEMA",
    "EVALUATOR_LAUNCH_CONTRACT_SCHEMA",
    "TRAINING_MOUNT_CONTRACT_SCHEMA",
    "DataViewManifest",
    "build_evaluator_launch_contract",
    "build_training_mount_contract",
    "materialize_data_views",
    "read_data_view_manifest",
    "verify_data_view_manifest",
    "HostCollectorSidecar",
    "HostCollectorIdentity",
    "verify_collector_artifacts",
    "bridge_signed_journal_to_receipts",
    "ProtocolSession",
    "activate_session",
    "current_session",
    "DataViewHandle",
    "ProtocolSplit",
    "ProtocolReference",
    "build_view_handles",
    "FRAMEWORK_CANDIDATES",
    "INVALID_CANDIDATES",
    "protocol_references",
    "PreflightStatus",
    "ProtocolPreflightRunner",
    "build_bounded_repair_receipt",
    "preflight_cache_key",
    "static_compatibility_check",
    "validate_preflight_admission",
    "build_training_evidence_manifest",
    "dry_evidence_closure",
    "preterminal_evidence_closure",
    "ProtocolRuntimeMode",
    "ProtocolRolloutStage",
    "aggregate_shadow_reports",
    "build_dual_observer_report",
    "build_rollback_receipt",
    "validate_protocol_runtime_mode",
    "validate_rollout_transition",
]
