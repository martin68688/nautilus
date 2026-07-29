"""Shadow comparison, rollout gates and rollback for Host Protocol runtime."""

from __future__ import annotations

from collections import Counter
from enum import Enum
import time
from typing import Any, Iterable, Mapping

from authority.runtime_protocol import build_runtime_protocol_plan

from .events import EVENT_ORDER, hash_payload


DUAL_OBSERVER_REPORT_SCHEMA = "mlevolve_protocol_dual_observer_report_v1"
SHADOW_SUMMARY_SCHEMA = "mlevolve_protocol_shadow_summary_v1"
ROLLBACK_RECEIPT_SCHEMA = "mlevolve_protocol_runtime_rollback_receipt_v1"


class ProtocolRuntimeMode(str, Enum):
    LEGACY_AST = "legacy_ast"
    HOST_SDK_SHADOW = "host_sdk_shadow"
    HOST_SDK_ENFORCE = "host_sdk_enforce"


class ProtocolRolloutStage(str, Enum):
    OFF = "off"
    HOST_SDK_SHADOW = "host_sdk_shadow"
    DUAL_OBSERVER_REVIEW = "dual_observer_review"
    CANARY_ENFORCE = "canary_enforce"
    FULL_ENFORCE = "full_enforce"


def validate_rollout_transition(from_stage: str, to_stage: str) -> None:
    source = ProtocolRolloutStage(from_stage)
    target = ProtocolRolloutStage(to_stage)
    order = list(ProtocolRolloutStage)
    if order.index(target) != order.index(source) + 1:
        raise ValueError("Protocol rollout must advance exactly one reviewed stage")


def validate_protocol_runtime_mode(
    mode: str,
    *,
    authority_mode: str,
    preflight_enabled: bool,
) -> ProtocolRuntimeMode:
    try:
        selected = ProtocolRuntimeMode(str(mode))
    except ValueError as error:
        raise ValueError(f"Unknown protocol_runtime_mode: {mode}") from error
    authority = str(authority_mode).lower()
    if selected == ProtocolRuntimeMode.HOST_SDK_SHADOW and authority not in {
        "shadow",
        "enforce",
    }:
        raise ValueError("Host SDK shadow requires Authority shadow/enforce")
    if selected == ProtocolRuntimeMode.HOST_SDK_ENFORCE:
        if authority != "enforce":
            raise ValueError("Host SDK enforce requires Authority enforce")
        if not preflight_enabled:
            raise ValueError("Host SDK enforce requires Protocol Preflight")
    return selected


def build_dual_observer_report(
    source: str,
    host_preflight_report: Mapping[str, Any],
    *,
    task_id: str,
    system_id: str,
    expected_legal: bool,
    full_budget_seconds: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    legacy = build_runtime_protocol_plan(source)
    legacy_duration = time.perf_counter() - started
    legacy_events = sorted(
        {kind for event in legacy.get("events") or [] for kind in event.get("kinds") or []}
    )
    host_pass = host_preflight_report.get("status") == "pass"
    legacy_pass = legacy.get("status") == "ready"
    if host_pass and legacy_pass:
        disagreement = "both_allow"
    elif host_pass:
        disagreement = "host_allow_legacy_block"
    elif legacy_pass:
        disagreement = "host_block_legacy_allow"
    else:
        disagreement = "both_block"
    preflight_seconds = float(
        host_preflight_report.get("preflight_duration_seconds") or 0.0
    )
    collector_seconds = float(
        host_preflight_report.get("collector_overhead_seconds") or 0.0
    )
    report = {
        "schema": DUAL_OBSERVER_REPORT_SCHEMA,
        "task_id": str(task_id),
        "system_id": str(system_id),
        "expected_legal": bool(expected_legal),
        "host_sdk": {
            "status": str(host_preflight_report.get("status") or ""),
            "positive": host_pass,
            "event_coverage": list(EVENT_ORDER) if host_pass else [],
            "report_hash": str(host_preflight_report.get("report_hash") or ""),
        },
        "legacy_ast": {
            "status": str(legacy.get("status") or ""),
            "positive": legacy_pass,
            "event_coverage": legacy_events,
            "missing_events": list(legacy.get("missing_plan_kinds") or []),
            "reason": str(legacy.get("reason") or ""),
            "plan_hash": str(legacy.get("plan_sha256") or ""),
        },
        "disagreement": disagreement,
        "host_false_allow": not expected_legal and host_pass,
        "host_false_deny": expected_legal and not host_pass,
        "legacy_false_allow": not expected_legal and legacy_pass,
        "legacy_false_deny": expected_legal and not legacy_pass,
        "timing": {
            "legacy_plan_seconds": round(legacy_duration, 9),
            "preflight_duration_seconds": preflight_seconds,
            "collector_overhead_seconds": collector_seconds,
            "runtime_overhead_percent": (
                collector_seconds / preflight_seconds * 100.0
                if preflight_seconds > 0
                else 0.0
            ),
            "preflight_budget_percent": (
                preflight_seconds / full_budget_seconds * 100.0
                if full_budget_seconds > 0
                else 0.0
            ),
            "receipt_bytes": int(host_preflight_report.get("receipt_bytes") or 0),
        },
        "report_hash": "",
    }
    report["report_hash"] = hash_payload(report, "report_hash")
    return report


def aggregate_shadow_reports(
    reports: Iterable[Mapping[str, Any]],
    *,
    max_runtime_overhead_percent: float = 5.0,
    max_preflight_budget_percent: float = 10.0,
) -> dict[str, Any]:
    rows = [dict(report) for report in reports]
    if not rows:
        raise ValueError("Shadow summary requires at least one dual-observer report")
    valid = [row for row in rows if row["expected_legal"]]
    invalid = [row for row in rows if not row["expected_legal"]]
    runtime_overheads = [float(row["timing"]["runtime_overhead_percent"]) for row in valid]
    preflight_overheads = [float(row["timing"]["preflight_budget_percent"]) for row in valid]
    receipt_sizes = [int(row["timing"]["receipt_bytes"]) for row in valid]
    host_invalid_blocks = sum(not row["host_sdk"]["positive"] for row in invalid)
    legacy_invalid_blocks = sum(not row["legacy_ast"]["positive"] for row in invalid)
    summary = {
        "schema": SHADOW_SUMMARY_SCHEMA,
        "report_count": len(rows),
        "legal_count": len(valid),
        "invalid_count": len(invalid),
        "event_coverage": {
            "host_sdk_complete": sum(
                row["host_sdk"]["event_coverage"] == list(EVENT_ORDER) for row in valid
            ),
            "legacy_ast_complete": sum(
                row["legacy_ast"]["event_coverage"] == list(EVENT_ORDER) for row in valid
            ),
        },
        "disagreement_distribution": dict(
            sorted(Counter(str(row["disagreement"]) for row in rows).items())
        ),
        "task_distribution": dict(
            sorted(Counter(str(row["task_id"]) for row in rows).items())
        ),
        "system_distribution": dict(
            sorted(Counter(str(row["system_id"]) for row in rows).items())
        ),
        "reason_distribution": dict(
            sorted(
                Counter(
                    str(row["host_sdk"]["status"])
                    if row["host_sdk"]["positive"]
                    else str(row["host_sdk"]["status"] or "blocked")
                    for row in rows
                ).items()
            )
        ),
        "host_false_allow_count": sum(bool(row["host_false_allow"]) for row in rows),
        "host_false_deny_count": sum(bool(row["host_false_deny"]) for row in rows),
        "legacy_false_allow_count": sum(bool(row["legacy_false_allow"]) for row in rows),
        "legacy_false_deny_count": sum(bool(row["legacy_false_deny"]) for row in rows),
        "host_invalid_detection_rate": host_invalid_blocks / len(invalid) if invalid else 1.0,
        "legacy_invalid_detection_rate": legacy_invalid_blocks / len(invalid) if invalid else 1.0,
        "max_runtime_overhead_percent": max(runtime_overheads, default=0.0),
        "max_preflight_budget_percent": max(preflight_overheads, default=0.0),
        "total_receipt_bytes": sum(receipt_sizes),
        "max_receipt_bytes": max(receipt_sizes, default=0),
        "gate_checks": {
            "host_violation_detection_not_lower": host_invalid_blocks >= legacy_invalid_blocks,
            "host_legal_evidence_denial_zero": all(row["host_sdk"]["positive"] for row in valid),
            "runtime_overhead_within_target": max(runtime_overheads, default=0.0)
            <= max_runtime_overhead_percent,
            "preflight_overhead_within_target": max(preflight_overheads, default=0.0)
            <= max_preflight_budget_percent,
        },
        "summary_hash": "",
    }
    summary["summary_hash"] = hash_payload(summary, "summary_hash")
    return summary


def build_rollback_receipt(
    *,
    from_mode: str,
    to_mode: str,
    reason: str,
    artifact_root: str,
) -> dict[str, Any]:
    source = ProtocolRuntimeMode(from_mode)
    target = ProtocolRuntimeMode(to_mode)
    allowed = {
        ProtocolRuntimeMode.HOST_SDK_ENFORCE: {
            ProtocolRuntimeMode.HOST_SDK_SHADOW,
            ProtocolRuntimeMode.LEGACY_AST,
        },
        ProtocolRuntimeMode.HOST_SDK_SHADOW: {ProtocolRuntimeMode.LEGACY_AST},
    }
    if target not in allowed.get(source, set()):
        raise ValueError("Protocol runtime rollback transition is not allowed")
    if not reason or not artifact_root:
        raise ValueError("Rollback requires a reason and preserved artifact root")
    receipt = {
        "schema": ROLLBACK_RECEIPT_SCHEMA,
        "from_mode": source.value,
        "to_mode": target.value,
        "reason": str(reason),
        "artifact_root": str(artifact_root),
        "artifacts_preserved": True,
        "historical_artifacts_deleted": False,
        "current_pointer_changed": False,
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = hash_payload(receipt, "receipt_hash")
    return receipt


__all__ = [
    "DUAL_OBSERVER_REPORT_SCHEMA",
    "ProtocolRuntimeMode",
    "ProtocolRolloutStage",
    "ROLLBACK_RECEIPT_SCHEMA",
    "SHADOW_SUMMARY_SCHEMA",
    "aggregate_shadow_reports",
    "build_dual_observer_report",
    "build_rollback_receipt",
    "validate_protocol_runtime_mode",
    "validate_rollout_transition",
]
