from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from authority.ledger import AuthorityLedger
from authority.models import (
    AuthorityDecision,
    DecisionOutcome,
    GenerationStage,
    GovernanceStage,
    Operation,
)
from authority.rollout import (
    AuthorityRolloutController,
    CanaryThresholds,
    RolloutVersionSet,
    build_canary_oracle_packet,
    evaluate_canary,
    load_shadow_records_from_ledger,
    verify_canary_oracle_packet,
)


def _controller(
    ledger: AuthorityLedger | None = None,
) -> AuthorityRolloutController:
    return AuthorityRolloutController(
        mode="enforce",
        versions=RolloutVersionSet(
            rollout_id="canary-test",
            policy_version="authority_v1",
            protocol_ref="protocol@1#" + "a" * 64,
            collector_version="1",
        ),
        ledger=ledger,
    )


def _record(
    controller: AuthorityRolloutController,
    decision_id: str,
    *,
    authority_allowed: bool,
    legacy_allowed: bool = True,
    enforced: bool = True,
):
    decision = AuthorityDecision(
        decision_id=decision_id,
        outcome=(
            DecisionOutcome.ALLOW if authority_allowed else DecisionOutcome.DENY
        ),
        permitted_scope=None,
        satisfied_paths=[],
        missing_obligations=[] if authority_allowed else ["receipt:evaluator"],
        blocking_receipts=[],
        required_action=None,
        policy_version="authority_v1",
        claim_id=f"claim-{decision_id}",
        artifact_id=f"artifact-{decision_id}",
        operation=Operation.RANK.value,
        decision_stage="branch_selection",
        generation_stage=GenerationStage.IMPROVE.value,
        governance_stage=GovernanceStage.BRANCH_SELECTION.value,
    )
    return controller.record(
        decision,
        legacy_allowed=legacy_allowed,
        effective_allowed=authority_allowed,
        enforced=enforced,
    )


def test_canary_passes_with_zero_invalid_influence_and_full_retention() -> None:
    controller = _controller()
    _record(controller, "valid-1", authority_allowed=True)
    _record(controller, "valid-2", authority_allowed=True)
    _record(controller, "invalid-1", authority_allowed=False)
    _record(controller, "invalid-2", authority_allowed=False)
    _record(controller, "outside-scope", authority_allowed=False, enforced=False)
    report = evaluate_canary(
        controller.records(),
        oracle_should_allow={
            "valid-1": True,
            "valid-2": True,
            "invalid-1": False,
            "invalid-2": False,
        },
        thresholds=CanaryThresholds(
            minimum_decisions=4,
            max_unauthorized_authority_allows=0,
            max_false_denial_rate=0.0,
        ),
    )

    assert report["passed"] is True
    assert report["observed_record_count"] == 5
    assert report["decision_count"] == 4
    assert report["excluded_unenforced_decision_ids"] == ["outside-scope"]
    assert report["legacy_iir"] == 1.0
    assert report["authority_iir"] == 0.0
    assert report["authority_vkr"] == 1.0
    assert len(report["report_hash"]) == 64


def test_canary_fails_on_any_unauthorized_allow() -> None:
    controller = _controller()
    _record(controller, "valid", authority_allowed=True)
    _record(controller, "invalid", authority_allowed=True)
    report = evaluate_canary(
        controller.records(),
        oracle_should_allow={"valid": True, "invalid": False},
        thresholds=CanaryThresholds(
            minimum_decisions=2,
            max_unauthorized_authority_allows=0,
            max_false_denial_rate=1.0,
        ),
    )
    assert report["passed"] is False
    assert report["unauthorized_authority_allow_count"] == 1


def test_canary_fails_when_false_denial_rate_exceeds_threshold() -> None:
    controller = _controller()
    _record(controller, "valid-allow", authority_allowed=True)
    _record(controller, "valid-deny", authority_allowed=False)
    report = evaluate_canary(
        controller.records(),
        oracle_should_allow={"valid-allow": True, "valid-deny": True},
        thresholds=CanaryThresholds(
            minimum_decisions=2,
            max_unauthorized_authority_allows=0,
            max_false_denial_rate=0.25,
        ),
    )
    assert report["passed"] is False
    assert report["authority_false_denial_rate"] == 0.5


def test_canary_oracle_packet_binds_complete_independent_labels() -> None:
    controller = _controller()
    _record(controller, "valid", authority_allowed=True)
    _record(controller, "invalid", authority_allowed=False)
    _record(controller, "outside-scope", authority_allowed=False, enforced=False)
    packet = build_canary_oracle_packet(controller.records())

    assert packet["decision_count"] == 2
    assert all(
        row["review"]["oracle_should_allow"] is None
        and row["review"]["reviewer"] == ""
        for row in packet["reviewed_records"]
    )
    with pytest.raises(ValueError, match="lacks reviewer"):
        verify_canary_oracle_packet(packet, controller.records())
    for row in packet["reviewed_records"]:
        decision_id = row["record"]["decision_id"]
        row["review"] = {
            "reviewer": "independent-test-reviewer",
            "oracle_should_allow": decision_id == "valid",
            "notes": "fixture oracle",
        }
    verified = verify_canary_oracle_packet(packet, controller.records())

    assert verified["oracle_should_allow"] == {
        "invalid": False,
        "valid": True,
    }
    assert verified["review_report"]["verified"] is True
    assert verified["review_report"]["reviewers"] == [
        "independent-test-reviewer"
    ]
    assert len(verified["review_report"]["report_hash"]) == 64

    tampered = copy.deepcopy(packet)
    tampered["reviewed_records"][0]["record"]["artifact_id"] = "tampered"
    with pytest.raises(ValueError, match="record hash mismatch"):
        verify_canary_oracle_packet(tampered, controller.records())


def test_verified_ledger_can_be_evaluated_offline_by_cli(tmp_path: Path) -> None:
    ledger_path = tmp_path / "authority_events.jsonl"
    controller = _controller(AuthorityLedger(ledger_path))
    _record(controller, "valid", authority_allowed=True)
    _record(controller, "invalid", authority_allowed=False)
    loaded = load_shadow_records_from_ledger(ledger_path)
    assert [record.decision_id for record in loaded] == ["invalid", "valid"]

    packet_path = tmp_path / "canary-oracle-packet.json"
    build_script = (
        Path(__file__).resolve().parents[2]
        / "paper-skills"
        / "memory_bundle"
        / "build_canary_oracle_packet.py"
    )
    built = subprocess.run(
        [
            sys.executable,
            str(build_script),
            "--ledger",
            str(ledger_path),
            "--packet",
            str(packet_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    for row in packet["reviewed_records"]:
        decision_id = row["record"]["decision_id"]
        row["review"] = {
            "reviewer": "independent-cli-test-reviewer",
            "oracle_should_allow": decision_id == "valid",
            "notes": "fixture oracle",
        }
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    oracle = tmp_path / "oracle.json"
    oracle_review = tmp_path / "oracle-review.json"
    verify_script = (
        Path(__file__).resolve().parents[2]
        / "paper-skills"
        / "memory_bundle"
        / "verify_canary_oracle_packet.py"
    )
    verified = subprocess.run(
        [
            sys.executable,
            str(verify_script),
            "--ledger",
            str(ledger_path),
            "--packet",
            str(packet_path),
            "--oracle",
            str(oracle),
            "--report",
            str(oracle_review),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(oracle_review.read_text())["verified"] is True
    report_path = tmp_path / "canary-report.json"
    script = (
        Path(__file__).resolve().parents[2]
        / "paper-skills"
        / "memory_bundle"
        / "evaluate_authority_canary.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--ledger",
            str(ledger_path),
            "--oracle",
            str(oracle),
            "--report",
            str(report_path),
            "--minimum-decisions",
            "2",
            "--max-false-denial-rate",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["gate_metric_basis"] == (
        "effective_allowed_on_enforced_records"
    )
