from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest

from authority.models import TaskContext
from authority.protocol_compiler import ProtocolCompiler
from authority.protocol_execution_contract import (
    PROTOCOL_EXECUTION_CONTRACT_SCHEMA,
    ProtocolExecutionContract,
    read_contract_artifact,
    write_contract_artifacts,
)
from authority.protocol_registry import ProtocolRegistry


REGISTRY_PATH = Path("mlevolve/config/protocols")
BUDGET = {
    "max_epochs": 3,
    "max_folds": 2,
    "max_models": 4,
    "timeout_seconds": 300,
}


def _contract(protocol: str = "random-classification@1") -> ProtocolExecutionContract:
    registry = ProtocolRegistry(REGISTRY_PATH)
    compiler = ProtocolCompiler(registry)
    return compiler.compile_execution_contract(
        protocol,
        task_context=TaskContext(task_id="task-a", task_family="image"),
        train_view_ref="view://task-a/split-1/train",
        validation_view_ref="view://task-a/split-1/internal-validation",
        terminal_view_ref="evaluator-only://task-a/split-1/terminal",
        execution_budget=BUDGET,
    )


def test_contract_is_deterministic_protocol_bound_and_immutable() -> None:
    left = _contract()
    right = _contract()
    assert left == right
    assert left.schema == PROTOCOL_EXECUTION_CONTRACT_SCHEMA
    assert left.contract_hash == right.contract_hash
    assert left.protocol_ref.canonical_hash in left.canonical_json()
    assert left.adapter_spec["legacy_ast_positive_proof"] is False
    assert left.adapter_spec["full_runtime_sdk_required"] is True
    assert {"PIL", "cv2", "scipy", "timm", "torchvision", "transformers"} <= set(
        left.allowed_import_roots
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        left.task_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        left.execution_budget["max_epochs"] = 99  # type: ignore[index]


@pytest.mark.parametrize(
    ("protocol", "strategy", "payload_key"),
    [
        ("random-classification@1", "stratified_random", "stratification_verified"),
        ("grouped-classification@1", "grouped", "group_overlap_count"),
        ("chronological-regression@1", "chronological", "future_to_past_count"),
    ],
)
def test_three_formal_protocols_compile(
    protocol: str, strategy: str, payload_key: str
) -> None:
    contract = _contract(protocol)
    assert contract.split_strategy == strategy
    assert payload_key in contract.required_payloads["split_lineage"]
    assert set(contract.required_runtime_events) == {
        "split_lineage",
        "fit_scope",
        "prediction_scope",
        "evaluator",
        "selection_freeze",
    }


def test_round_trip_and_artifact_sidecar(tmp_path: Path) -> None:
    contract = _contract()
    json_path, sha_path = write_contract_artifacts(contract, tmp_path)
    assert json_path.name == "PROTOCOL_EXECUTION_CONTRACT.json"
    assert sha_path.name == "PROTOCOL_EXECUTION_CONTRACT.sha256"
    assert read_contract_artifact(json_path) == contract
    assert json.loads(json_path.read_text()) == contract.as_dict()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("task_id", "tampered-task"),
        ("split_strategy", "random"),
        ("train_view_ref", "view://other/train"),
        ("execution_budget", {"max_epochs": 999}),
        ("required_receipts", ["code_execution"]),
        ("contract_hash", "0" * 64),
    ],
)
def test_tamper_fails_closed(field: str, replacement: object) -> None:
    payload = copy.deepcopy(_contract().as_dict())
    payload[field] = replacement
    with pytest.raises(ValueError):
        ProtocolExecutionContract.from_dict(payload)


def test_unknown_schema_and_extra_field_fail_closed() -> None:
    payload = _contract().as_dict()
    payload["schema"] = "mlevolve_protocol_execution_contract_v999"
    with pytest.raises(ValueError, match="Unsupported"):
        ProtocolExecutionContract.from_dict(payload)
    payload = _contract().as_dict()
    payload["surprise"] = True
    with pytest.raises(ValueError, match="fields do not match"):
        ProtocolExecutionContract.from_dict(payload)
    payload = _contract().as_dict()
    payload["protocol_ref"]["extra"] = "not-in-schema"
    with pytest.raises(ValueError, match="protocol_ref fields"):
        ProtocolExecutionContract.from_dict(payload)


def test_unbound_or_negative_budget_fails_closed() -> None:
    for budget in ({}, {"max_epochs": -1}):
        payload = _contract().as_dict()
        payload["execution_budget"] = budget
        core = {key: value for key, value in payload.items() if key not in {"contract_id", "contract_hash"}}
        # Even a caller that knows how to rebuild canonical IDs may not create
        # an unbounded or nonsensical execution contract.
        import hashlib

        canonical = lambda value: json.dumps(  # noqa: E731
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        payload["contract_id"] = "pec-" + hashlib.sha256(
            canonical(core).encode()
        ).hexdigest()[:24]
        payload["contract_hash"] = hashlib.sha256(
            canonical({**core, "contract_id": payload["contract_id"]}).encode()
        ).hexdigest()
        with pytest.raises(ValueError, match="budget"):
            ProtocolExecutionContract.from_dict(payload)


def test_legacy_protocol_specs_remain_readable() -> None:
    registry = ProtocolRegistry(REGISTRY_PATH)
    assert registry.resolve("mlevolve-default@1").protocol_id == "mlevolve-default"
    assert registry.resolve("mlevolve-default@2").version == "2"


def test_existing_immutable_artifact_cannot_be_replaced(tmp_path: Path) -> None:
    write_contract_artifacts(_contract(), tmp_path)
    path = tmp_path / "PROTOCOL_EXECUTION_CONTRACT.json"
    path.chmod(0o644)
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_contract_artifacts(_contract(), tmp_path)
