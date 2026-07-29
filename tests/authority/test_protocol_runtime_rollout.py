from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from protocol_runtime.events import hash_payload
from protocol_runtime.rollout import (
    ProtocolRuntimeMode,
    ProtocolRolloutStage,
    aggregate_shadow_reports,
    build_dual_observer_report,
    build_rollback_receipt,
    validate_protocol_runtime_mode,
    validate_rollout_transition,
)


def _host_report(*, status: str, duration: float = 1.0, overhead: float = 0.01):
    value = {
        "schema": "mlevolve_protocol_preflight_report_v1",
        "status": status,
        "preflight_duration_seconds": duration,
        "collector_overhead_seconds": overhead,
        "receipt_bytes": 4096 if status == "pass" else 0,
        "report_hash": "",
    }
    value["report_hash"] = hash_payload(value, "report_hash")
    return value


def test_runtime_modes_fail_closed_and_require_preflight() -> None:
    assert validate_protocol_runtime_mode(
        "legacy_ast", authority_mode="off", preflight_enabled=False
    ) == ProtocolRuntimeMode.LEGACY_AST
    assert validate_protocol_runtime_mode(
        "host_sdk_shadow", authority_mode="shadow", preflight_enabled=True
    ) == ProtocolRuntimeMode.HOST_SDK_SHADOW
    assert validate_protocol_runtime_mode(
        "host_sdk_enforce", authority_mode="enforce", preflight_enabled=True
    ) == ProtocolRuntimeMode.HOST_SDK_ENFORCE
    with pytest.raises(ValueError, match="requires Protocol Preflight"):
        validate_protocol_runtime_mode(
            "host_sdk_enforce", authority_mode="enforce", preflight_enabled=False
        )
    with pytest.raises(ValueError, match="requires Authority enforce"):
        validate_protocol_runtime_mode(
            "host_sdk_enforce", authority_mode="shadow", preflight_enabled=True
        )
    validate_rollout_transition(
        ProtocolRolloutStage.HOST_SDK_SHADOW.value,
        ProtocolRolloutStage.DUAL_OBSERVER_REVIEW.value,
    )
    with pytest.raises(ValueError, match="exactly one"):
        validate_rollout_transition("off", "canary_enforce")


def test_dual_observer_summary_attributes_legacy_false_denial() -> None:
    legal_source = """\
def candidate(session):
    views = session.get_split()
    with session.fit_scope(component='m', data_view=views.train):
        pass
    with session.prediction_scope(component='m', data_view=views.validation):
        pass
    session.evaluate_internal(views.validation, [], label_key='label')
    session.freeze_selection('m', based_on=views.validation)
"""
    invalid_source = """\
from sklearn.model_selection import train_test_split
def candidate(session):
    train_test_split([], shuffle=True)
"""
    legal = build_dual_observer_report(
        legal_source,
        _host_report(status="pass"),
        task_id="taxi",
        system_id="full",
        expected_legal=True,
        full_budget_seconds=30,
    )
    invalid = build_dual_observer_report(
        invalid_source,
        _host_report(status="protocol_violation", duration=0.01, overhead=0.0),
        task_id="taxi",
        system_id="invalid",
        expected_legal=False,
        full_budget_seconds=30,
    )
    assert legal["disagreement"] == "host_allow_legacy_block"
    assert legal["legacy_false_deny"] is True
    assert invalid["host_false_allow"] is False
    summary = aggregate_shadow_reports([legal, invalid])
    assert summary["host_false_allow_count"] == 0
    assert summary["host_false_deny_count"] == 0
    assert summary["host_invalid_detection_rate"] == 1.0
    assert all(summary["gate_checks"].values())


def test_rollback_preserves_artifacts_and_cannot_escalate() -> None:
    receipt = build_rollback_receipt(
        from_mode="host_sdk_enforce",
        to_mode="host_sdk_shadow",
        reason="local rollback drill",
        artifact_root="/new/nonformal/root",
    )
    assert receipt["artifacts_preserved"] is True
    assert receipt["historical_artifacts_deleted"] is False
    assert receipt["current_pointer_changed"] is False
    assert receipt["receipt_hash"] == hash_payload(receipt, "receipt_hash")
    with pytest.raises(ValueError, match="not allowed"):
        build_rollback_receipt(
            from_mode="legacy_ast",
            to_mode="host_sdk_enforce",
            reason="not a rollback",
            artifact_root="/new/nonformal/root",
        )


def test_additive_rollout_profiles_do_not_modify_historical_formal_config() -> None:
    root = Path("mlevolve/config")
    shadow = OmegaConf.load(root / "config_authority_host_protocol_shadow.yaml")
    enforce = OmegaConf.load(root / "config_authority_host_protocol_enforce.yaml")
    formal = OmegaConf.load(root / "config_authority_formal_enforce.yaml")
    assert shadow.evaluation_authority.protocol_runtime_mode == "host_sdk_shadow"
    assert enforce.evaluation_authority.protocol_runtime_mode == "host_sdk_enforce"
    assert shadow.agent.protocol_preflight.enabled is True
    assert enforce.agent.protocol_preflight.enabled is True
    assert "protocol_runtime_mode" not in formal.evaluation_authority
    assert formal.agent.protocol_preflight.enabled is False


def test_shadow_artifacts_are_hash_bound_json(tmp_path: Path) -> None:
    report = build_dual_observer_report(
        "def candidate(session):\n    session.get_split()\n",
        _host_report(status="missing_evidence", duration=0.01, overhead=0.0),
        task_id="cactus",
        system_id="shadow",
        expected_legal=True,
        full_budget_seconds=30,
    )
    path = tmp_path / "dual.json"
    path.write_text(json.dumps(report, sort_keys=True), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["report_hash"] == hash_payload(loaded, "report_hash")
