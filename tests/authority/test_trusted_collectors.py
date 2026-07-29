from __future__ import annotations

import copy
import hashlib
from dataclasses import replace

import pytest

from authority.collectors import (
    CodeExecutionCollector,
    CounterfactualCollector,
    DerivationCollector,
    EvaluatorIntegrityCollector,
    FitScopeCollector,
    MethodIdentityCollector,
    PredictionScopeCollector,
    ReplicationCollector,
    RuntimeActuationCollector,
    SeedAggregationCollector,
    SelectionFreezeCollector,
    SplitLineageCollector,
    StaticActuationCollector,
    TrustedCollectorHost,
    UntrustedObservationError,
)
from authority.models import ProtocolRef
from authority.protocol_registry import canonical_json


DIGEST = "a" * 64
PROTOCOL = ProtocolRef("test", "1", "b" * 64)


def _specifications():
    return (
        (CodeExecutionCollector, {"exit_status": 0, "executed_path": "artifact", "run_hash": DIGEST}),
        (MethodIdentityCollector, {"method_fingerprint": DIGEST, "code_sha256": DIGEST}),
        (SplitLineageCollector, {"partition_hashes": {"train": DIGEST}, "overlap_count": 0}),
        (FitScopeCollector, {"fit_scope_hashes": {"model": DIGEST}, "holdout_fit_count": 0}),
        (PredictionScopeCollector, {"prediction_scope_hashes": {"valid": DIGEST}, "forbidden_overlap_count": 0}),
        (EvaluatorIntegrityCollector, {"evaluator_hash": DIGEST, "inputs_hash": DIGEST, "metric_direction": "maximize", "tampered": False}),
        (SelectionFreezeCollector, {"candidate_set_hash": DIGEST, "frozen_before_holdout": True}),
        (SeedAggregationCollector, {"declared_seeds": [1, 2, 3], "all_results_hash": DIGEST, "aggregation": "mean", "paired": True, "preregistered": True, "best_seed_selection": False}),
        (ReplicationCollector, {"replication_id": "replication-1", "task_family": "tabular", "result_hash": DIGEST, "equal_data_budget": True, "equal_compute_budget": True, "paired_bootstrap_ci_lower_gt_zero": True}),
        (StaticActuationCollector, {"contract_hash": DIGEST, "checks": {"must_change": True}}),
        (RuntimeActuationCollector, {"event_hashes": [DIGEST], "target_path_executed": True}),
        (CounterfactualCollector, {"pair_id": "pair", "memory_on_action_hash": DIGEST, "memory_off_action_hash": "c" * 64, "action_or_code_changed": True}),
        (DerivationCollector, {"parent_claim_refs": ["claim"], "mapping_hash": DIGEST, "scope_widened": False}),
    )


def test_all_trusted_collectors_emit_host_receipts_with_hash_chain() -> None:
    host = TrustedCollectorHost("test-host")
    receipts = [
        host.collect(
            collector,
            artifact_id="artifact",
            run_id="run",
            protocol_ref=PROTOCOL,
            source="tests.host",
            payload=payload,
        )
        for collector, payload in _specifications()
    ]
    assert len({receipt.collector_id for receipt in receipts}) == len(receipts)
    assert all(receipt.trust_status == "trusted_host" for receipt in receipts)
    assert all(receipt.observation_id for receipt in receipts)
    assert receipts[0].parent_event_hash == ""
    for previous, current in zip(receipts, receipts[1:]):
        assert current.parent_event_hash == previous.event_hash
    assert all(len(receipt.event_hash) == 64 for receipt in receipts)


def test_trusted_receipt_id_is_stable_even_as_event_chain_advances() -> None:
    host = TrustedCollectorHost("test-host")
    kwargs = dict(
        artifact_id="artifact",
        run_id="run",
        protocol_ref=PROTOCOL,
        source="tests.host",
        payload={"method_fingerprint": DIGEST, "code_sha256": DIGEST},
    )
    first = host.collect(MethodIdentityCollector, **kwargs)
    second = host.collect(MethodIdentityCollector, **kwargs)
    assert first.receipt_id == second.receipt_id
    assert first.event_hash != second.event_hash


def test_mutated_or_cross_host_observation_is_rejected() -> None:
    first_host = TrustedCollectorHost("first")
    second_host = TrustedCollectorHost("second")
    observation = first_host.observe(
        MethodIdentityCollector.receipt_type,
        artifact_id="artifact",
        run_id="run",
        protocol_ref=PROTOCOL,
        source="tests.host",
        payload={"method_fingerprint": DIGEST, "code_sha256": DIGEST},
    )
    with pytest.raises(UntrustedObservationError, match="not minted"):
        MethodIdentityCollector(second_host).collect(observation)
    observation.payload["code_sha256"] = "c" * 64
    with pytest.raises(UntrustedObservationError, match="changed after capture"):
        MethodIdentityCollector(first_host).collect(observation)


def test_leaked_capability_cannot_rebind_a_host_observation() -> None:
    host = TrustedCollectorHost("real-host")
    observation = host.observe(
        MethodIdentityCollector.receipt_type,
        artifact_id="artifact",
        run_id="run",
        protocol_ref=PROTOCOL,
        source="tests.host",
        payload={"method_fingerprint": DIGEST, "code_sha256": DIGEST},
    )
    forged_payload = {"method_fingerprint": DIGEST, "code_sha256": "c" * 64}
    forged = replace(
        observation,
        payload=forged_payload,
        payload_hash=hashlib.sha256(
            canonical_json(forged_payload).encode("utf-8")
        ).hexdigest(),
    )
    with pytest.raises(UntrustedObservationError, match="metadata changed"):
        MethodIdentityCollector(host).collect(forged)


def test_counterfactual_without_a_real_delta_is_rejected() -> None:
    host = TrustedCollectorHost("test-host")
    with pytest.raises(UntrustedObservationError, match="did not change"):
        host.collect(
            CounterfactualCollector,
            artifact_id="artifact",
            run_id="run",
            protocol_ref=PROTOCOL,
            source="tests.host",
            payload={
                "pair_id": "pair",
                "memory_on_action_hash": DIGEST,
                "memory_off_action_hash": "c" * 64,
                "action_or_code_changed": False,
            },
        )


def test_invalid_runtime_fact_cannot_mint_trusted_receipt() -> None:
    host = TrustedCollectorHost("test-host")
    with pytest.raises(UntrustedObservationError, match="forbidden overlap"):
        host.collect(
            SplitLineageCollector,
            artifact_id="artifact",
            run_id="run",
            protocol_ref=PROTOCOL,
            source="tests.host",
            payload={"partition_hashes": {"train": DIGEST}, "overlap_count": 1},
        )


def test_deterministic_random_split_requires_and_records_host_verification() -> None:
    host = TrustedCollectorHost("deterministic-random-host")
    receipt = host.collect(
        SplitLineageCollector,
        artifact_id="artifact",
        run_id="run",
        protocol_ref=PROTOCOL,
        source="tests.host",
        payload={
            "partition_hashes": {"train": DIGEST, "valid": "c" * 64},
            "overlap_count": 0,
            "split_strategy": "deterministic_random",
            "deterministic_partition_verified": True,
        },
    )
    assert receipt.payload["split_strategy"] == "deterministic_random"
    assert receipt.payload["deterministic_partition_verified"] is True

    with pytest.raises(UntrustedObservationError, match="was not verified"):
        host.collect(
            SplitLineageCollector,
            artifact_id="artifact",
            run_id="run",
            protocol_ref=PROTOCOL,
            source="tests.host",
            payload={
                "partition_hashes": {"train": DIGEST, "valid": "c" * 64},
                "overlap_count": 0,
                "split_strategy": "deterministic_random",
                "deterministic_partition_verified": False,
            },
        )
