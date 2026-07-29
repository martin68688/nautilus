from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from authority.protocol_execution_contract import compile_protocol_execution_contract
from authority.protocol_registry import ProtocolRegistry
from protocol_runtime.collector import HostCollectorIdentity
from protocol_runtime.data_views import materialize_data_views, verify_data_view_manifest
from protocol_runtime.preflight import PreflightStatus, ProtocolPreflightRunner
from protocol_runtime.reference_adapters import (
    FRAMEWORK_CANDIDATES,
    INVALID_CANDIDATES,
    protocol_references,
)


IMAGE = "sha256:pr6-local-test"
SDK = "sdk-pr6-local-test"


def _contract(registry, reference, identity):
    return compile_protocol_execution_contract(
        registry.resolve(reference.protocol_ref),
        task_id=reference.task_id,
        task_family=reference.task_family,
        train_view_ref=f"view://{reference.task_id}/train",
        validation_view_ref=f"view://{reference.task_id}/internal-validation",
        terminal_view_ref=f"evaluator-only://{reference.task_id}/terminal",
        execution_budget={
            "max_epochs": 1,
            "max_folds": 1,
            "max_models": 4,
            "timeout_seconds": 30,
        },
        collector_spec=identity.collector_spec(),
    )


def test_three_reference_protocols_materialize_the_required_host_boundaries(
    tmp_path: Path,
) -> None:
    registry = ProtocolRegistry("mlevolve/config/protocols")
    observations = {}
    for reference in protocol_references():
        identity = HostCollectorIdentity.generate()
        contract = _contract(registry, reference, identity)
        manifest, path = materialize_data_views(
            reference.records,
            tmp_path / reference.task_id,
            contract,
            split_id="pr6-reference",
        )
        verification = verify_data_view_manifest(path, contract=contract)
        observations[reference.task_id] = manifest
        assert manifest.terminal_view_mounted_in_training is False
        assert verification["status"] == "pass"
    assert observations["aerial-cactus-identification"].sample_overlap_count == 0
    assert observations["mlsp-2013-birds"].group_overlap_count == 0
    assert observations["new-york-city-taxi-fare-prediction"].future_to_past_count == 0


@pytest.mark.parametrize("framework", sorted(FRAMEWORK_CANDIDATES))
def test_taxi_all_three_framework_paths_close_preflight(
    tmp_path: Path, framework: str
) -> None:
    reference = protocol_references()[2]
    registry = ProtocolRegistry("mlevolve/config/protocols")
    identity = HostCollectorIdentity.generate()
    contract = _contract(registry, reference, identity)
    _manifest, path = materialize_data_views(
        reference.records,
        tmp_path / framework / "views",
        contract,
        split_id=f"pr6-taxi-{framework}",
    )
    candidate = FRAMEWORK_CANDIDATES[framework]
    report = ProtocolPreflightRunner(registry).run(
        candidate,
        source=inspect.getsource(candidate),
        contract=contract,
        identity=identity,
        data_view_manifest_path=path,
        output_root=tmp_path / framework / "preflight",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.PASS.value
    assert report["terminal_exposure_count"] == 0
    assert report["result_fact_created"] is False
    assert report["preflight_duration_seconds"] > 0
    assert report["collector_overhead_seconds"] >= 0
    assert report["receipt_bytes"] > 0


@pytest.mark.parametrize("reference", protocol_references(), ids=lambda value: value.task_id)
def test_each_protocol_intentional_invalid_control_is_rejected(
    tmp_path: Path, reference
) -> None:
    registry = ProtocolRegistry("mlevolve/config/protocols")
    identity = HostCollectorIdentity.generate()
    contract = _contract(registry, reference, identity)
    _manifest, path = materialize_data_views(
        reference.records,
        tmp_path / reference.task_id / "views",
        contract,
        split_id="pr6-invalid",
    )
    candidate = INVALID_CANDIDATES[reference.task_id]
    report = ProtocolPreflightRunner(registry).run(
        candidate,
        source=inspect.getsource(candidate),
        contract=contract,
        identity=identity,
        data_view_manifest_path=path,
        output_root=tmp_path / reference.task_id / "preflight",
        image_digest=IMAGE,
        sdk_hash=SDK,
    )
    assert report["status"] == PreflightStatus.PROTOCOL_VIOLATION.value
    assert report["terminal_exposure_count"] == 0
