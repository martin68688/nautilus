"""Canonical Host-owned execution contracts compiled from immutable protocols."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

from .models import ProtocolRef, ProtocolSpec, ReceiptType


PROTOCOL_EXECUTION_CONTRACT_SCHEMA = "mlevolve_protocol_execution_contract_v1"
_RUNTIME_EVENTS = (
    "split_lineage",
    "fit_scope",
    "prediction_scope",
    "evaluator",
    "selection_freeze",
)
_RUNTIME_RECEIPTS = (
    ReceiptType.CODE_EXECUTION.value,
    ReceiptType.METHOD_IDENTITY.value,
    ReceiptType.SPLIT_LINEAGE.value,
    ReceiptType.FIT_SCOPE.value,
    ReceiptType.PREDICTION_SCOPE.value,
    ReceiptType.EVALUATOR.value,
    ReceiptType.SELECTION_FREEZE.value,
)
_IMPORT_ROOT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, ProtocolRef):
        return {
            "protocol_id": value.protocol_id,
            "version": value.version,
            "canonical_hash": value.canonical_hash,
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Execution contract value is not canonical JSON: {type(value)!r}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Execution contract value cannot be frozen: {type(value)!r}")


def protocol_payload_requirements(spec: ProtocolSpec) -> dict[str, dict[str, Any]]:
    """Compile the score-bearing payload obligations without relaxing v1 rules."""

    payloads: dict[str, dict[str, Any]] = {}
    if spec.promotion_policy.get("enforce_protocol_payloads") is not True:
        return payloads
    split_strategy = str(spec.data_split_policy.get("strategy") or "")
    if split_strategy:
        split_flags: dict[str, Any] = {"split_strategy": split_strategy}
        if split_strategy == "stratified_random":
            split_flags["stratification_verified"] = True
        elif split_strategy == "grouped":
            split_flags["group_overlap_count"] = 0
        elif split_strategy == "chronological":
            split_flags.update(
                {
                    "future_to_past_count": 0,
                    "chronological_order_verified": True,
                }
            )
        elif split_strategy == "deterministic_random":
            split_flags["deterministic_partition_verified"] = True
        else:
            split_flags["unsupported_split_strategy"] = False
        payloads[ReceiptType.SPLIT_LINEAGE.value] = split_flags
    fit_scope = str(spec.preprocessing_policy.get("fit_scope") or "")
    if fit_scope:
        payloads[ReceiptType.FIT_SCOPE.value] = {"fit_scope": fit_scope}
    evaluator: dict[str, Any] = {}
    metric_name = str(spec.metric_spec.get("name") or "")
    metric_direction = str(spec.metric_spec.get("direction") or "")
    if metric_name:
        evaluator["metric_name"] = metric_name
    if metric_direction in {"maximize", "minimize"}:
        evaluator["metric_direction"] = metric_direction
    if evaluator:
        payloads[ReceiptType.EVALUATOR.value] = evaluator
    return payloads


def _core_payload(
    *,
    schema: str,
    protocol_ref: ProtocolRef,
    task_id: str,
    task_family: str,
    split_strategy: str,
    train_view_ref: str,
    validation_view_ref: str,
    terminal_view_ref: str,
    required_runtime_events: tuple[str, ...],
    required_receipts: tuple[str, ...],
    required_payloads: Mapping[str, Any],
    allowed_import_roots: tuple[str, ...],
    execution_budget: Mapping[str, Any],
    evaluator_spec: Mapping[str, Any],
    collector_spec: Mapping[str, Any],
    adapter_spec: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": schema,
        "protocol_ref": _plain(protocol_ref),
        "task_id": task_id,
        "task_family": task_family,
        "split_strategy": split_strategy,
        "train_view_ref": train_view_ref,
        "validation_view_ref": validation_view_ref,
        "terminal_view_ref": terminal_view_ref,
        "required_runtime_events": list(required_runtime_events),
        "required_receipts": list(required_receipts),
        "required_payloads": _plain(required_payloads),
        "allowed_import_roots": list(allowed_import_roots),
        "execution_budget": _plain(execution_budget),
        "evaluator_spec": _plain(evaluator_spec),
        "collector_spec": _plain(collector_spec),
        "adapter_spec": _plain(adapter_spec),
    }


@dataclass(frozen=True)
class ProtocolExecutionContract:
    schema: str
    contract_id: str
    protocol_ref: ProtocolRef
    task_id: str
    task_family: str
    split_strategy: str
    train_view_ref: str
    validation_view_ref: str
    terminal_view_ref: str
    required_runtime_events: tuple[str, ...]
    required_receipts: tuple[str, ...]
    required_payloads: Mapping[str, Any]
    allowed_import_roots: tuple[str, ...]
    execution_budget: Mapping[str, Any]
    evaluator_spec: Mapping[str, Any]
    collector_spec: Mapping[str, Any]
    adapter_spec: Mapping[str, Any]
    contract_hash: str

    def __post_init__(self) -> None:
        if self.schema != PROTOCOL_EXECUTION_CONTRACT_SCHEMA:
            raise ValueError(f"Unsupported execution contract schema: {self.schema!r}")
        for field_name in (
            "task_id",
            "task_family",
            "split_strategy",
            "train_view_ref",
            "validation_view_ref",
            "terminal_view_ref",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Execution contract {field_name} must be non-empty")
        if len(self.protocol_ref.canonical_hash) != 64:
            raise ValueError("Execution contract requires a hash-bound ProtocolRef")
        if tuple(self.required_runtime_events) != tuple(
            sorted(set(self.required_runtime_events))
        ):
            raise ValueError("Runtime events must be sorted and unique")
        if tuple(self.required_receipts) != tuple(sorted(set(self.required_receipts))):
            raise ValueError("Required receipts must be sorted and unique")
        if tuple(self.allowed_import_roots) != tuple(
            sorted(set(self.allowed_import_roots))
        ) or any(not _IMPORT_ROOT.fullmatch(root) for root in self.allowed_import_roots):
            raise ValueError("Allowed import roots must be sorted unique module roots")
        object.__setattr__(self, "required_payloads", _freeze(self.required_payloads))
        object.__setattr__(self, "execution_budget", _freeze(self.execution_budget))
        object.__setattr__(self, "evaluator_spec", _freeze(self.evaluator_spec))
        object.__setattr__(self, "collector_spec", _freeze(self.collector_spec))
        object.__setattr__(self, "adapter_spec", _freeze(self.adapter_spec))
        if not self.execution_budget:
            raise ValueError("Execution contract budget must be explicitly bound")
        for name, value in self.execution_budget.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"Execution budget {name} cannot be negative")
        core = self.core_dict()
        expected_id = f"pec-{_digest(core)[:24]}"
        if self.contract_id != expected_id:
            raise ValueError("Execution contract ID does not match canonical content")
        expected_hash = _digest({**core, "contract_id": self.contract_id})
        if self.contract_hash != expected_hash:
            raise ValueError("Execution contract hash mismatch")

    def core_dict(self) -> dict[str, Any]:
        return _core_payload(
            schema=self.schema,
            protocol_ref=self.protocol_ref,
            task_id=self.task_id,
            task_family=self.task_family,
            split_strategy=self.split_strategy,
            train_view_ref=self.train_view_ref,
            validation_view_ref=self.validation_view_ref,
            terminal_view_ref=self.terminal_view_ref,
            required_runtime_events=tuple(self.required_runtime_events),
            required_receipts=tuple(self.required_receipts),
            required_payloads=self.required_payloads,
            allowed_import_roots=tuple(self.allowed_import_roots),
            execution_budget=self.execution_budget,
            evaluator_spec=self.evaluator_spec,
            collector_spec=self.collector_spec,
            adapter_spec=self.adapter_spec,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.core_dict(),
            "contract_id": self.contract_id,
            "contract_hash": self.contract_hash,
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.as_dict())

    @classmethod
    def create(cls, **values: Any) -> "ProtocolExecutionContract":
        values = dict(values)
        values.setdefault("schema", PROTOCOL_EXECUTION_CONTRACT_SCHEMA)
        values["required_runtime_events"] = tuple(
            sorted(set(map(str, values["required_runtime_events"])))
        )
        values["required_receipts"] = tuple(
            sorted(set(map(str, values["required_receipts"])))
        )
        values["allowed_import_roots"] = tuple(
            sorted(set(map(str, values["allowed_import_roots"])))
        )
        core = _core_payload(
            schema=values["schema"],
            protocol_ref=values["protocol_ref"],
            task_id=str(values["task_id"]),
            task_family=str(values["task_family"]),
            split_strategy=str(values["split_strategy"]),
            train_view_ref=str(values["train_view_ref"]),
            validation_view_ref=str(values["validation_view_ref"]),
            terminal_view_ref=str(values["terminal_view_ref"]),
            required_runtime_events=values["required_runtime_events"],
            required_receipts=values["required_receipts"],
            required_payloads=values["required_payloads"],
            allowed_import_roots=values["allowed_import_roots"],
            execution_budget=values["execution_budget"],
            evaluator_spec=values["evaluator_spec"],
            collector_spec=values["collector_spec"],
            adapter_spec=values["adapter_spec"],
        )
        values["contract_id"] = f"pec-{_digest(core)[:24]}"
        values["contract_hash"] = _digest(
            {**core, "contract_id": values["contract_id"]}
        )
        return cls(**values)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProtocolExecutionContract":
        expected = {
            "schema",
            "contract_id",
            "protocol_ref",
            "task_id",
            "task_family",
            "split_strategy",
            "train_view_ref",
            "validation_view_ref",
            "terminal_view_ref",
            "required_runtime_events",
            "required_receipts",
            "required_payloads",
            "allowed_import_roots",
            "execution_budget",
            "evaluator_spec",
            "collector_spec",
            "adapter_spec",
            "contract_hash",
        }
        if set(payload) != expected:
            raise ValueError(
                "Execution contract fields do not match schema: "
                f"missing={sorted(expected - set(payload))} "
                f"extra={sorted(set(payload) - expected)}"
            )
        protocol = payload["protocol_ref"]
        if not isinstance(protocol, Mapping):
            raise ValueError("Execution contract protocol_ref must be an object")
        protocol_fields = {"protocol_id", "version", "canonical_hash"}
        if set(protocol) != protocol_fields:
            raise ValueError("Execution contract protocol_ref fields do not match schema")
        return cls(
            schema=str(payload["schema"]),
            contract_id=str(payload["contract_id"]),
            protocol_ref=ProtocolRef(
                protocol_id=str(protocol["protocol_id"]),
                version=str(protocol["version"]),
                canonical_hash=str(protocol["canonical_hash"]),
            ),
            task_id=str(payload["task_id"]),
            task_family=str(payload["task_family"]),
            split_strategy=str(payload["split_strategy"]),
            train_view_ref=str(payload["train_view_ref"]),
            validation_view_ref=str(payload["validation_view_ref"]),
            terminal_view_ref=str(payload["terminal_view_ref"]),
            required_runtime_events=tuple(map(str, payload["required_runtime_events"])),
            required_receipts=tuple(map(str, payload["required_receipts"])),
            required_payloads=payload["required_payloads"],
            allowed_import_roots=tuple(map(str, payload["allowed_import_roots"])),
            execution_budget=payload["execution_budget"],
            evaluator_spec=payload["evaluator_spec"],
            collector_spec=payload["collector_spec"],
            adapter_spec=payload["adapter_spec"],
            contract_hash=str(payload["contract_hash"]),
        )


def compile_protocol_execution_contract(
    spec: ProtocolSpec,
    *,
    task_id: str,
    task_family: str,
    train_view_ref: str,
    validation_view_ref: str,
    terminal_view_ref: str,
    execution_budget: Mapping[str, Any],
    allowed_import_roots: tuple[str, ...] = (
        "PIL",
        "cv2",
        "lightgbm",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "timm",
        "torch",
        "torchvision",
        "transformers",
        "xgboost",
    ),
    collector_spec: Mapping[str, Any] | None = None,
    adapter_spec: Mapping[str, Any] | None = None,
) -> ProtocolExecutionContract:
    if not spec.canonical_hash:
        raise ValueError("ProtocolSpec must be registered and hash-bound before compile")
    split_strategy = str(spec.data_split_policy.get("strategy") or "unspecified")
    evaluator_spec = {
        "evaluator": _plain(spec.evaluator_spec),
        "metric": _plain(spec.metric_spec),
        "selection": _plain(spec.selection_policy),
        "holdout": _plain(spec.holdout_policy),
    }
    collector = collector_spec or {
        "schema": "mlevolve_host_collector_v1",
        "transport": "restricted_unix_socket",
        "append_only_hash_chain": True,
        "candidate_has_signing_key": False,
    }
    adapters = adapter_spec or {
        "managed": ["boosting", "sklearn"],
        "scope": ["torch"],
        "legacy_ast_positive_proof": False,
        "full_runtime_sdk_required": True,
    }
    return ProtocolExecutionContract.create(
        protocol_ref=spec.ref(),
        task_id=task_id,
        task_family=task_family,
        split_strategy=split_strategy,
        train_view_ref=train_view_ref,
        validation_view_ref=validation_view_ref,
        terminal_view_ref=terminal_view_ref,
        required_runtime_events=_RUNTIME_EVENTS,
        required_receipts=_RUNTIME_RECEIPTS,
        required_payloads=protocol_payload_requirements(spec),
        allowed_import_roots=allowed_import_roots,
        execution_budget=execution_budget,
        evaluator_spec=evaluator_spec,
        collector_spec=collector,
        adapter_spec=adapters,
    )


def write_contract_artifacts(
    contract: ProtocolExecutionContract,
    directory: str | Path,
) -> tuple[Path, Path]:
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "PROTOCOL_EXECUTION_CONTRACT.json"
    sha_path = root / "PROTOCOL_EXECUTION_CONTRACT.sha256"
    json_bytes = (contract.canonical_json() + "\n").encode("utf-8")
    sha_bytes = (hashlib.sha256(json_bytes).hexdigest() + "\n").encode("ascii")
    for path, content in ((json_path, json_bytes), (sha_path, sha_bytes)):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"Refusing to overwrite immutable contract artifact: {path}")
    return json_path, sha_path


def read_contract_artifact(path: str | Path) -> ProtocolExecutionContract:
    contract_path = Path(path).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    contract = ProtocolExecutionContract.from_dict(payload)
    sidecar = contract_path.with_suffix(".sha256")
    if sidecar.exists():
        expected_file_hash = sidecar.read_text(encoding="ascii").strip()
        actual_file_hash = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        if expected_file_hash != actual_file_hash:
            raise ValueError("Execution contract artifact SHA-256 mismatch")
    return contract


__all__ = [
    "PROTOCOL_EXECUTION_CONTRACT_SCHEMA",
    "ProtocolExecutionContract",
    "compile_protocol_execution_contract",
    "protocol_payload_requirements",
    "read_contract_artifact",
    "write_contract_artifacts",
]
