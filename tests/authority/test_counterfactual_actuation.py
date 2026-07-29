from __future__ import annotations

import pytest

from authority.actuation import ActuationLevel, ActuationTracker
from authority.collectors import (
    CounterfactualObservationCollector,
    TrustedCollectorHost,
)
from authority.paired_replay import PairedReplayRunner
from tests.authority.test_actuation_pipeline import ACTIVE, _contract, _observations


def test_controlled_pair_separates_influence_l4_from_effectiveness_l5() -> None:
    def executor(context, memory_enabled, memory_payload):
        threshold = 0.7 if memory_enabled else 0.5
        return {
            "action": {"threshold": threshold},
            "code": f"threshold = {threshold}",
            "outcome": 0.82 if memory_enabled else 0.75,
            "protocol_legal": True,
            "metadata": {"seed": context["seed"]},
        }

    result = PairedReplayRunner(executor).run(
        context={"seed": 42, "budget": 10},
        memory_payload={"clause_id": "clause-a"},
        metric_direction="maximize",
    )
    assert result.influence_confirmed is True
    assert result.protocol_legal is True
    assert result.outcome_delta == pytest.approx(0.07)
    assert result.effective is True
    assert result.memory_on_action_hash != result.memory_off_action_hash
    assert result.memory_on_code_hash != result.memory_off_code_hash
    assert len(result.result_hash) == 64


def test_no_action_or_code_delta_never_reaches_causal_confirmation() -> None:
    def executor(_context, _memory_enabled, _memory_payload):
        return {
            "action": "same",
            "code": "same = True",
            "outcome": 1.0,
            "protocol_legal": True,
        }

    result = PairedReplayRunner(executor).run(
        context={"seed": 7},
        memory_payload={"clause_id": "clause-a"},
        metric_direction="maximize",
    )
    assert result.influence_confirmed is False
    assert result.outcome_delta == 0
    assert result.effective is False


def test_observer_receipt_accepts_truthful_no_change_pair() -> None:
    host = TrustedCollectorHost("prospective-observer-test")
    receipt = host.collect(
        CounterfactualObservationCollector,
        artifact_id="node-1",
        run_id="run-1",
        protocol_ref=ACTIVE,
        source="host.prospective_counterfactual_observer",
        payload={
            "pair_id": "pair-1",
            "control_hash": "a" * 64,
            "memory_payload_hash": "b" * 64,
            "memory_on_action_hash": "c" * 64,
            "memory_off_action_hash": "c" * 64,
            "memory_on_code_hash": "d" * 64,
            "memory_off_code_hash": "d" * 64,
            "action_or_code_changed": False,
            "never_submitted_to_executor": True,
        },
    )
    assert receipt.payload["counterfactual_observed"] is True
    assert receipt.payload["action_or_code_changed"] is False


def test_protocol_illegal_improvement_is_not_effective() -> None:
    def executor(_context, memory_enabled, _memory_payload):
        return {
            "action": "memory" if memory_enabled else "baseline",
            "code": "x = 2" if memory_enabled else "x = 1",
            "outcome": 0.9 if memory_enabled else 0.7,
            "protocol_legal": not memory_enabled,
        }

    result = PairedReplayRunner(executor).run(
        context={"seed": 7},
        memory_payload={"clause_id": "clause-a"},
        metric_direction="maximize",
    )
    assert result.influence_confirmed is True
    assert result.outcome_delta == pytest.approx(0.2)
    assert result.protocol_legal is False
    assert result.effective is False


def test_minimize_metric_normalizes_positive_improvement_delta() -> None:
    def executor(_context, memory_enabled, _memory_payload):
        return {
            "action": "memory" if memory_enabled else "baseline",
            "code": "loss = 1" if memory_enabled else "loss = 2",
            "outcome": 0.2 if memory_enabled else 0.3,
            "protocol_legal": True,
        }

    result = PairedReplayRunner(executor).run(
        context={"seed": 1},
        memory_payload={"clause_id": "clause-a"},
        metric_direction="minimize",
    )
    assert result.outcome_delta == pytest.approx(0.1)
    assert result.effective is True


def test_executor_cannot_mutate_the_callers_control_context() -> None:
    caller_context = {"seed": 1}

    def executor(_arm_context, memory_enabled, _memory_payload):
        if memory_enabled:
            caller_context["seed"] = 99
        return {
            "action": memory_enabled,
            "code": str(memory_enabled),
            "outcome": float(memory_enabled),
            "protocol_legal": True,
        }

    with pytest.raises(RuntimeError, match="mutated the caller control"):
        PairedReplayRunner(executor).run(
            context=caller_context,
            memory_payload={"clause_id": "clause-a"},
            metric_direction="maximize",
        )


def test_tracker_emits_complete_l0_through_l5_report() -> None:
    contract = _contract()
    tracker = ActuationTracker(
        collector_host=TrustedCollectorHost("l5-host"),
        protocol_ref=ACTIVE,
        run_id="run-l5",
    )
    tracker.record_exposure(
        artifact_id="artifact-l5",
        contracts=[contract],
        request_id="request-l5",
    )
    tracker.record_claimed_adoption(
        artifact_id="artifact-l5", contract_id=contract.contract_id
    )
    preconditions, static, runtime = _observations(contract)
    tracker.record_static_observation(
        artifact_id="artifact-l5",
        contract_id=contract.contract_id,
        preconditions=preconditions,
        observations=static,
    )
    tracker.record_runtime_observation(
        artifact_id="artifact-l5",
        contract_id=contract.contract_id,
        observations=runtime,
    )

    result = PairedReplayRunner(
        lambda _context, enabled, _memory: {
            "action": "adopt" if enabled else "baseline",
            "code": "x = 2" if enabled else "x = 1",
            "outcome": 0.8 if enabled else 0.7,
            "protocol_legal": True,
        }
    ).run(
        context={"seed": 42},
        memory_payload={"contract_id": contract.contract_id},
        metric_direction="maximize",
    )
    receipt = tracker.record_counterfactual(
        artifact_id="artifact-l5",
        contract_id=contract.contract_id,
        pair_result=result,
    )
    assert receipt is not None
    report = tracker.report(
        artifact_id="artifact-l5", contract_id=contract.contract_id
    )
    assert report.highest_level == ActuationLevel.EFFECTIVE
    assert all(row.reached for row in report.levels)
    assert tracker.snapshot()["highest_level_counts"]["5"] == 1
