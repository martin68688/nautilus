"""Materialize and verify Host-owned, terminal-isolated data views."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from authority.models import ProtocolRef
from authority.protocol_execution_contract import ProtocolExecutionContract


DATA_VIEW_MANIFEST_SCHEMA = "mlevolve_data_view_manifest_v1"
TRAINING_MOUNT_CONTRACT_SCHEMA = "mlevolve_training_mount_contract_v1"
EVALUATOR_LAUNCH_CONTRACT_SCHEMA = "mlevolve_evaluator_launch_contract_v1"
_TRAIN_TARGETS = {
    "train": "/data/train_view",
    "internal_validation": "/data/internal_validation_view",
    "inference": "/data/inference_view",
    "manifest": "/data/DATA_VIEW_MANIFEST.json",
}
_TERMINAL_MARKERS = (
    "terminal_holdout",
    "terminal_labels",
    "evaluator_view",
    "labels.csv",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash_payload(value: Mapping[str, Any], hash_field: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {key: item for key, item in value.items() if key != hash_field}
        ).encode("utf-8")
    ).hexdigest()


def _hash_values(values: Iterable[Any]) -> str:
    encoded = [_canonical_json(value) for value in values]
    return hashlib.sha256("\n".join(sorted(encoded)).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _protocol_dict(ref: ProtocolRef) -> dict[str, str]:
    return {
        "protocol_id": ref.protocol_id,
        "version": ref.version,
        "canonical_hash": ref.canonical_hash,
    }


def _validate_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or ".." in path.parts:
        raise ValueError(f"Unsafe data-view relative path: {value!r}")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError(f"Malformed data-view relative path: {value!r}")
    return path


def _confined_regular_file(root: Path, relative: str, *, label: str) -> Path:
    relative_path = _validate_relative_path(relative)
    current = root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing symlink {label}: {current}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes its Host root") from error
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _time_key(value: Any) -> tuple[str, Any]:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a chronological time value")
    if isinstance(value, (int, float)):
        return ("number", float(value))
    if isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            return ("datetime", datetime.fromisoformat(text.replace("Z", "+00:00")))
        except ValueError:
            return ("string", text)
    raise ValueError(f"Unsupported chronological time value: {value!r}")


def _ordered_records(
    records: Sequence[Mapping[str, Any]], sample_id_key: str
) -> list[dict[str, Any]]:
    normalized = [dict(row) for row in records]
    identifiers = [row.get(sample_id_key) for row in normalized]
    if any(value is None or value == "" for value in identifiers):
        raise ValueError(f"Every record must have {sample_id_key}")
    if len({_canonical_json(value) for value in identifiers}) != len(identifiers):
        raise ValueError("Sample IDs must be unique before materialization")
    return normalized


def _stable_order(records: Iterable[dict[str, Any]], sample_id_key: str, seed: str):
    return sorted(
        records,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{_canonical_json(row[sample_id_key])}".encode("utf-8")
        ).hexdigest(),
    )


def _split_stratified(
    records: list[dict[str, Any]],
    *,
    sample_id_key: str,
    label_key: str,
    validation_fraction: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if label_key not in row:
            raise ValueError(f"Stratified materialization requires {label_key}")
        groups.setdefault(_canonical_json(row[label_key]), []).append(row)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for label, rows in sorted(groups.items()):
        ordered = _stable_order(rows, sample_id_key, f"{seed}:{label}")
        validation_count = 0
        if len(ordered) >= 2:
            validation_count = min(
                len(ordered) - 1,
                max(1, int(round(len(ordered) * validation_fraction))),
            )
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:])
    if not train or not validation:
        raise ValueError("Stratified split must produce non-empty train and validation")
    return _stable_order(train, sample_id_key, seed), _stable_order(
        validation, sample_id_key, seed
    )


def _split_grouped(
    records: list[dict[str, Any]],
    *,
    sample_id_key: str,
    group_id_key: str,
    validation_fraction: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        if row.get(group_id_key) in {None, ""}:
            raise ValueError(f"Grouped materialization requires {group_id_key}")
        groups.setdefault(_canonical_json(row[group_id_key]), []).append(row)
    if len(groups) < 2:
        raise ValueError("Grouped split requires at least two distinct groups")
    ordered_groups = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )
    target = max(1, int(round(len(records) * validation_fraction)))
    validation_groups: set[str] = set()
    count = 0
    for group in ordered_groups[:-1]:
        validation_groups.add(group)
        count += len(groups[group])
        if count >= target:
            break
    train = [
        row
        for group, rows in groups.items()
        if group not in validation_groups
        for row in rows
    ]
    validation = [
        row
        for group, rows in groups.items()
        if group in validation_groups
        for row in rows
    ]
    if not train or not validation:
        raise ValueError("Grouped split must produce non-empty train and validation")
    return _stable_order(train, sample_id_key, seed), _stable_order(
        validation, sample_id_key, seed
    )


def _split_chronological(
    records: list[dict[str, Any]],
    *,
    time_key: str,
    validation_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    keyed = []
    kinds: set[str] = set()
    for row in records:
        if time_key not in row:
            raise ValueError(f"Chronological materialization requires {time_key}")
        key = _time_key(row[time_key])
        kinds.add(key[0])
        keyed.append((key, row))
    if len(kinds) != 1:
        raise ValueError("Chronological time values must use one comparable type")
    keyed.sort(key=lambda item: item[0][1])
    target = min(len(keyed) - 1, max(1, int(round(len(keyed) * (1 - validation_fraction)))))
    boundaries = [
        index
        for index in range(1, len(keyed))
        if keyed[index - 1][0][1] < keyed[index][0][1]
    ]
    if not boundaries:
        raise ValueError("Chronological split requires at least two distinct times")
    cut = min(boundaries, key=lambda index: (abs(index - target), index))
    return [item[1] for item in keyed[:cut]], [item[1] for item in keyed[cut:]]


def _split_deterministic_random(
    records: list[dict[str, Any]],
    *,
    sample_id_key: str,
    validation_fraction: float,
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = _stable_order(records, sample_id_key, seed)
    validation_count = min(
        len(ordered) - 1,
        max(1, int(round(len(ordered) * validation_fraction))),
    )
    return ordered[validation_count:], ordered[:validation_count]


def _view_metadata(
    records: Sequence[Mapping[str, Any]],
    *,
    relative_path: str,
    data_sha256: str,
    view_ref: str,
    sample_id_key: str,
    group_id_key: str | None,
    time_key: str | None,
    asset_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample_ids = [row[sample_id_key] for row in records]
    groups = [row[group_id_key] for row in records] if group_id_key else []
    times = [row[time_key] for row in records] if time_key else []
    return {
        "relative_path": relative_path,
        "view_ref": view_ref,
        "sample_count": len(records),
        "sample_id_sha256": _hash_values(sample_ids),
        "group_id_sha256": _hash_values(groups) if groups else "",
        "time_min": min(times, key=lambda value: _time_key(value)[1]) if times else None,
        "time_max": max(times, key=lambda value: _time_key(value)[1]) if times else None,
        "data_sha256": data_sha256,
        **dict(asset_metadata or {}),
    }


def _materialize_assets(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    role: str,
    sample_id_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy declared source assets into the immutable view and hash every file."""

    transformed: list[dict[str, Any]] = []
    entries: list[dict[str, str]] = []
    for row in rows:
        value = dict(row)
        declared = value.pop("_host_assets", None)
        if declared is None:
            transformed.append(value)
            continue
        if not isinstance(declared, Mapping) or not declared:
            raise ValueError("_host_assets must be a non-empty logical-name mapping")
        sample_token = hashlib.sha256(
            _canonical_json(value[sample_id_key]).encode("utf-8")
        ).hexdigest()[:24]
        visible: dict[str, str] = {}
        for logical_name, source_value in sorted(declared.items()):
            logical = str(logical_name)
            if not logical or not logical.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Unsafe Host asset logical name: {logical!r}")
            requested = Path(str(source_value)).expanduser()
            if requested.is_symlink():
                raise ValueError("Refusing symlink Host source asset")
            source = requested.resolve(strict=True)
            if not source.is_file():
                raise ValueError("Host source asset must be a regular file")
            suffix = source.suffix.lower()
            destination = (
                root
                / f"{role}_view"
                / "assets"
                / f"{sample_token}-{logical}{suffix}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError("Host asset destination collision")
            with source.open("rb") as source_handle:
                descriptor = os.open(
                    destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
                )
                with os.fdopen(descriptor, "wb") as destination_handle:
                    shutil.copyfileobj(source_handle, destination_handle)
                    # Closing the file completes the CephFS write.  A per-asset
                    # fsync turns image tasks into tens of thousands of remote
                    # metadata round trips.  The view remains unpublished until
                    # every asset is re-read, hashed, and the fsynced manifest
                    # plus Host binding are written, so a partial/crashed build
                    # is still fail-closed without synchronizing each image.
            relative = destination.relative_to(root).as_posix()
            digest = _sha256_file(destination)
            entries.append(
                {"logical_name": logical, "relative_path": relative, "sha256": digest}
            )
            visible[logical] = str(destination.resolve(strict=True))
        value["assets"] = visible
        transformed.append(value)
    if not entries:
        return transformed, {
            "asset_count": 0,
            "asset_manifest_relative_path": "",
            "asset_manifest_sha256": "",
        }
    manifest_path = root / f"{role}_view" / "ASSET_MANIFEST.json"
    manifest_payload = {
        "schema": "mlevolve_data_view_asset_manifest_v1",
        "role": role,
        "entries": entries,
    }
    _write_exclusive(
        manifest_path,
        (json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    return transformed, {
        "asset_count": len(entries),
        "asset_manifest_relative_path": manifest_path.relative_to(root).as_posix(),
        "asset_manifest_sha256": _sha256_file(manifest_path),
    }


@dataclass(frozen=True)
class DataViewManifest:
    schema: str
    task_id: str
    protocol_ref: Mapping[str, str]
    contract_hash: str
    split_id: str
    split_strategy: str
    strategy_verification: Mapping[str, Any]
    views: Mapping[str, Mapping[str, Any]]
    sample_overlap_count: int
    group_overlap_count: int
    future_to_past_count: int
    terminal_view_mounted_in_training: bool
    terminal_view_ref_sha256: str
    manifest_hash: str

    def __post_init__(self) -> None:
        if self.schema != DATA_VIEW_MANIFEST_SCHEMA:
            raise ValueError(f"Unsupported DataViewManifest schema: {self.schema!r}")
        if set(self.protocol_ref) != {"protocol_id", "version", "canonical_hash"}:
            raise ValueError("DataViewManifest protocol_ref fields do not match schema")
        roles = set(self.views)
        if roles not in (
            {"train", "internal_validation"},
            {"train", "internal_validation", "inference"},
        ):
            raise ValueError(
                "DataViewManifest must contain train/validation and optional inference"
            )
        if self.terminal_view_mounted_in_training:
            raise ValueError("Terminal view may not be mounted in training")
        if min(
            self.sample_overlap_count,
            self.group_overlap_count,
            self.future_to_past_count,
        ) < 0:
            raise ValueError("DataViewManifest counts cannot be negative")
        object.__setattr__(self, "protocol_ref", _freeze(self.protocol_ref))
        object.__setattr__(self, "strategy_verification", _freeze(self.strategy_verification))
        object.__setattr__(self, "views", _freeze(self.views))
        if self.manifest_hash != _hash_payload(self.as_dict(), "manifest_hash"):
            raise ValueError("DataViewManifest hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "protocol_ref": _plain(self.protocol_ref),
            "contract_hash": self.contract_hash,
            "split_id": self.split_id,
            "split_strategy": self.split_strategy,
            "strategy_verification": _plain(self.strategy_verification),
            "views": _plain(self.views),
            "sample_overlap_count": self.sample_overlap_count,
            "group_overlap_count": self.group_overlap_count,
            "future_to_past_count": self.future_to_past_count,
            "terminal_view_mounted_in_training": self.terminal_view_mounted_in_training,
            "terminal_view_ref_sha256": self.terminal_view_ref_sha256,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DataViewManifest":
        expected = {
            "schema",
            "task_id",
            "protocol_ref",
            "contract_hash",
            "split_id",
            "split_strategy",
            "strategy_verification",
            "views",
            "sample_overlap_count",
            "group_overlap_count",
            "future_to_past_count",
            "terminal_view_mounted_in_training",
            "terminal_view_ref_sha256",
            "manifest_hash",
        }
        if set(payload) != expected:
            raise ValueError("DataViewManifest fields do not match schema")
        return cls(**payload)


def materialize_data_views(
    records: Sequence[Mapping[str, Any]],
    output_root: str | Path,
    contract: ProtocolExecutionContract,
    *,
    inference_records: Sequence[Mapping[str, Any]] | None = None,
    inference_view_ref: str = "",
    split_id: str,
    sample_id_key: str = "sample_id",
    label_key: str = "label",
    group_id_key: str = "group_id",
    time_key: str = "timestamp",
    validation_fraction: float = 0.2,
    seed: str = "0",
) -> tuple[DataViewManifest, Path]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    normalized = _ordered_records(records, sample_id_key)
    if len(normalized) < 2:
        raise ValueError("At least two records are required")
    normalized_inference = (
        _ordered_records(inference_records, sample_id_key)
        if inference_records is not None
        else []
    )
    if normalized_inference and not str(inference_view_ref).strip():
        raise ValueError("Inference records require a bound inference_view_ref")
    if any(label_key in row for row in normalized_inference):
        raise ValueError("Inference view must not contain the supervised label")
    if contract.split_strategy == "stratified_random":
        train, validation = _split_stratified(
            normalized,
            sample_id_key=sample_id_key,
            label_key=label_key,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        active_group_key = None
        active_time_key = None
        train_label_counts: dict[str, int] = {}
        validation_label_counts: dict[str, int] = {}
        for row in train:
            key = _canonical_json(row[label_key])
            train_label_counts[key] = train_label_counts.get(key, 0) + 1
        for row in validation:
            key = _canonical_json(row[label_key])
            validation_label_counts[key] = validation_label_counts.get(key, 0) + 1
        strategy_verification = {
            "stratification_verified": True,
            "label_key": label_key,
            "train_label_counts": train_label_counts,
            "validation_label_counts": validation_label_counts,
        }
    elif contract.split_strategy == "grouped":
        train, validation = _split_grouped(
            normalized,
            sample_id_key=sample_id_key,
            group_id_key=group_id_key,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        active_group_key = group_id_key
        active_time_key = None
        strategy_verification = {
            "group_isolation_verified": True,
            "group_id_key": group_id_key,
        }
    elif contract.split_strategy == "chronological":
        train, validation = _split_chronological(
            normalized,
            time_key=time_key,
            validation_fraction=validation_fraction,
        )
        active_group_key = None
        active_time_key = time_key
        strategy_verification = {
            "chronological_order_verified": True,
            "time_key": time_key,
        }
    elif contract.split_strategy == "deterministic_random":
        train, validation = _split_deterministic_random(
            normalized,
            sample_id_key=sample_id_key,
            validation_fraction=validation_fraction,
            seed=seed,
        )
        active_group_key = None
        active_time_key = None
        strategy_verification = {
            "deterministic_partition_verified": True,
            "partition_seed": str(seed),
        }
    else:
        raise ValueError(f"Unsupported Host split strategy: {contract.split_strategy}")

    train_ids = {_canonical_json(row[sample_id_key]) for row in train}
    validation_ids = {_canonical_json(row[sample_id_key]) for row in validation}
    inference_ids = {
        _canonical_json(row[sample_id_key]) for row in normalized_inference
    }
    sample_overlap = len(train_ids & validation_ids)
    if inference_ids & (train_ids | validation_ids):
        raise ValueError("Inference sample IDs overlap train/validation")
    train_groups = (
        {_canonical_json(row[active_group_key]) for row in train}
        if active_group_key
        else set()
    )
    validation_groups = (
        {_canonical_json(row[active_group_key]) for row in validation}
        if active_group_key
        else set()
    )
    group_overlap = len(train_groups & validation_groups)
    future_to_past = 0
    if active_time_key:
        max_train = max(_time_key(row[active_time_key])[1] for row in train)
        min_validation = min(_time_key(row[active_time_key])[1] for row in validation)
        # A strict max(train) < min(validation) boundary proves that the
        # cross-product violation count is zero.  Enumerating every pair is
        # quadratic and becomes intractable for Taxi-scale Host views.
        if not max_train < min_validation:
            raise ValueError("Chronological split violates past-to-future ordering")
    if sample_overlap or group_overlap or future_to_past:
        raise ValueError("Host materializer produced an invalid data boundary")

    requested_root = Path(output_root)
    if requested_root.is_symlink():
        raise ValueError("Refusing symlink data-view output root")
    root = requested_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Refusing to materialize into a non-empty data-view root")
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": root / "train_view" / "data.jsonl",
        "internal_validation": root / "internal_validation_view" / "data.jsonl",
    }
    if normalized_inference:
        paths["inference"] = root / "inference_view" / "data.jsonl"
    materialized_rows: dict[str, list[dict[str, Any]]] = {}
    asset_metadata: dict[str, dict[str, Any]] = {}
    role_rows = [("train", train), ("internal_validation", validation)]
    if normalized_inference:
        role_rows.append(("inference", normalized_inference))
    for role, rows in role_rows:
        visible_rows, visible_assets = _materialize_assets(
            rows, root=root, role=role, sample_id_key=sample_id_key
        )
        materialized_rows[role] = visible_rows
        asset_metadata[role] = visible_assets
        # Search splits use a canonical order.  Submission inference is
        # different: its frozen source order is part of the evaluator
        # contract, so sorting otherwise-valid IDs would make every Host-order
        # submission fail terminal alignment.
        rows_to_write = (
            visible_rows
            if role == "inference"
            else sorted(
                visible_rows,
                key=lambda row: _canonical_json(row[sample_id_key]),
            )
        )
        content = "".join(
            _canonical_json(row) + "\n" for row in rows_to_write
        ).encode("utf-8")
        _write_exclusive(paths[role], content)
    views = {
        "train": _view_metadata(
            materialized_rows["train"],
            relative_path="train_view/data.jsonl",
            data_sha256=_sha256_file(paths["train"]),
            view_ref=contract.train_view_ref,
            sample_id_key=sample_id_key,
            group_id_key=active_group_key,
            time_key=active_time_key,
            asset_metadata=asset_metadata["train"],
        ),
        "internal_validation": _view_metadata(
            materialized_rows["internal_validation"],
            relative_path="internal_validation_view/data.jsonl",
            data_sha256=_sha256_file(paths["internal_validation"]),
            view_ref=contract.validation_view_ref,
            sample_id_key=sample_id_key,
            group_id_key=active_group_key,
            time_key=active_time_key,
            asset_metadata=asset_metadata["internal_validation"],
        ),
    }
    if normalized_inference:
        views["inference"] = _view_metadata(
            materialized_rows["inference"],
            relative_path="inference_view/data.jsonl",
            data_sha256=_sha256_file(paths["inference"]),
            view_ref=str(inference_view_ref),
            sample_id_key=sample_id_key,
            group_id_key=None,
            time_key=None,
            asset_metadata=asset_metadata["inference"],
        )
    payload = {
        "schema": DATA_VIEW_MANIFEST_SCHEMA,
        "task_id": contract.task_id,
        "protocol_ref": _protocol_dict(contract.protocol_ref),
        "contract_hash": contract.contract_hash,
        "split_id": split_id,
        "split_strategy": contract.split_strategy,
        "strategy_verification": strategy_verification,
        "views": views,
        "sample_overlap_count": sample_overlap,
        "group_overlap_count": group_overlap,
        "future_to_past_count": future_to_past,
        "terminal_view_mounted_in_training": False,
        "terminal_view_ref_sha256": hashlib.sha256(
            contract.terminal_view_ref.encode("utf-8")
        ).hexdigest(),
        "manifest_hash": "",
    }
    payload["manifest_hash"] = _hash_payload(payload, "manifest_hash")
    manifest = DataViewManifest.from_dict(payload)
    manifest_path = root / "DATA_VIEW_MANIFEST.json"
    _write_exclusive(
        manifest_path,
        (json.dumps(manifest.as_dict(), indent=2, sort_keys=True) + "\n").encode(),
    )
    build_training_mount_contract(manifest, root)
    return manifest, manifest_path


def read_data_view_manifest(path: str | Path) -> DataViewManifest:
    manifest_path = Path(path)
    if manifest_path.is_symlink():
        raise ValueError("Refusing symlink DataViewManifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DataViewManifest.from_dict(payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("Data-view rows must be JSON objects")
        rows.append(value)
    return rows


def verify_data_view_manifest(
    path: str | Path,
    *,
    contract: ProtocolExecutionContract | None = None,
    sample_id_key: str = "sample_id",
    label_key: str = "label",
    group_id_key: str = "group_id",
    time_key: str = "timestamp",
    verify_asset_contents: bool = True,
) -> dict[str, Any]:
    """Verify a DataView bundle and its split invariants.

    ``verify_asset_contents`` is true for materialization/publication, where
    every copied asset must be audited.  A runtime loading an immutable Host
    binding may set it false and reuse that freeze-time content attestation;
    the manifest, row files, asset manifests, and split invariants are still
    checked.
    """

    requested_manifest_path = Path(path)
    if requested_manifest_path.is_symlink():
        raise ValueError("Refusing symlink DataViewManifest")
    manifest_path = requested_manifest_path.resolve(strict=True)
    manifest = read_data_view_manifest(manifest_path)
    root = manifest_path.parent.resolve(strict=True)
    if contract is not None:
        if manifest.contract_hash != contract.contract_hash:
            raise ValueError("DataViewManifest contract hash mismatch")
        if manifest.protocol_ref != _protocol_dict(contract.protocol_ref):
            raise ValueError("DataViewManifest ProtocolRef mismatch")
    loaded: dict[str, list[dict[str, Any]]] = {}
    for role, metadata in manifest.views.items():
        data_path = _confined_regular_file(
            root, str(metadata["relative_path"]), label=f"{role} data view"
        )
        if _sha256_file(data_path) != metadata["data_sha256"]:
            raise ValueError(f"{role} data view hash mismatch")
        rows = _read_jsonl(data_path)
        if len(rows) != metadata["sample_count"]:
            raise ValueError(f"{role} sample count mismatch")
        if _hash_values(row[sample_id_key] for row in rows) != metadata[
            "sample_id_sha256"
        ]:
            raise ValueError(f"{role} sample ID hash mismatch")
        if manifest.split_strategy == "grouped":
            if _hash_values(row[group_id_key] for row in rows) != metadata[
                "group_id_sha256"
            ]:
                raise ValueError(f"{role} group ID hash mismatch")
        # Chronological ordering is a property of the labeled search split.
        # The unlabeled inference view is terminal-blind and may come from a
        # different timestamp range (or have no timestamp field at all); it
        # remains protected by disjoint IDs, hashes, and post-freeze scope
        # checks below, but must not be compared against train/validation
        # boundaries during bundle verification.
        if manifest.split_strategy == "chronological" and role != "inference":
            times = [row[time_key] for row in rows]
            if min(times, key=lambda value: _time_key(value)[1]) != metadata[
                "time_min"
            ] or max(times, key=lambda value: _time_key(value)[1]) != metadata[
                "time_max"
            ]:
                raise ValueError(f"{role} time boundary mismatch")
        loaded[role] = rows
        asset_count = int(metadata.get("asset_count") or 0)
        asset_manifest_relative = str(
            metadata.get("asset_manifest_relative_path") or ""
        )
        if asset_count:
            asset_manifest_path = _confined_regular_file(
                root,
                asset_manifest_relative,
                label=f"{role} asset manifest",
            )
            if _sha256_file(asset_manifest_path) != metadata.get(
                "asset_manifest_sha256"
            ):
                raise ValueError(f"{role} asset manifest hash mismatch")
            asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
            entries = asset_manifest.get("entries") or []
            if len(entries) != asset_count:
                raise ValueError(f"{role} asset count mismatch")
            if verify_asset_contents:
                for entry in entries:
                    asset_path = _confined_regular_file(
                        root,
                        str(entry.get("relative_path") or ""),
                        label=f"{role} asset",
                    )
                    if _sha256_file(asset_path) != entry.get("sha256"):
                        raise ValueError(f"{role} asset hash mismatch")
        elif asset_manifest_relative or metadata.get("asset_manifest_sha256"):
            raise ValueError(f"{role} empty asset metadata is inconsistent")
    train_ids = {_canonical_json(row[sample_id_key]) for row in loaded["train"]}
    validation_ids = {
        _canonical_json(row[sample_id_key])
        for row in loaded["internal_validation"]
    }
    inference_ids = {
        _canonical_json(row[sample_id_key])
        for row in loaded.get("inference", [])
    }
    if inference_ids & (train_ids | validation_ids):
        raise ValueError("Inference data overlaps train/validation")
    if len(train_ids & validation_ids) != manifest.sample_overlap_count:
        raise ValueError("DataViewManifest sample overlap count mismatch")
    if manifest.split_strategy == "grouped":
        if manifest.strategy_verification.get("group_isolation_verified") is not True:
            raise ValueError("Grouped isolation was not verified")
        if manifest.strategy_verification.get("group_id_key") != group_id_key:
            raise ValueError("Grouped key binding mismatch")
        train_groups = {
            _canonical_json(row[group_id_key]) for row in loaded["train"]
        }
        validation_groups = {
            _canonical_json(row[group_id_key])
            for row in loaded["internal_validation"]
        }
        if len(train_groups & validation_groups) != manifest.group_overlap_count:
            raise ValueError("DataViewManifest group overlap count mismatch")
    if manifest.split_strategy == "chronological":
        if manifest.strategy_verification.get("chronological_order_verified") is not True:
            raise ValueError("Chronological order was not verified")
        if manifest.strategy_verification.get("time_key") != time_key:
            raise ValueError("Chronological time key binding mismatch")
        max_train = max(_time_key(row[time_key])[1] for row in loaded["train"])
        min_validation = min(
            _time_key(row[time_key])[1]
            for row in loaded["internal_validation"]
        )
        if not max_train < min_validation or manifest.future_to_past_count != 0:
            raise ValueError("DataViewManifest chronological boundary mismatch")
    if manifest.split_strategy == "stratified_random":
        verification = manifest.strategy_verification
        if verification.get("stratification_verified") is not True or (
            verification.get("label_key") != label_key
        ):
            raise ValueError("Stratification verification is missing")
        label_field = str(verification["label_key"])
        actual_train: dict[str, int] = {}
        actual_validation: dict[str, int] = {}
        for row in loaded["train"]:
            key = _canonical_json(row[label_field])
            actual_train[key] = actual_train.get(key, 0) + 1
        for row in loaded["internal_validation"]:
            key = _canonical_json(row[label_field])
            actual_validation[key] = actual_validation.get(key, 0) + 1
        if actual_train != dict(verification.get("train_label_counts") or {}) or (
            actual_validation
            != dict(verification.get("validation_label_counts") or {})
        ):
            raise ValueError("Stratification distribution binding mismatch")
    if manifest.split_strategy == "deterministic_random":
        if manifest.strategy_verification.get("deterministic_partition_verified") is not True:
            raise ValueError("Deterministic random partition verification is missing")
    if any(
        value != 0
        for value in (
            manifest.sample_overlap_count,
            manifest.group_overlap_count,
            manifest.future_to_past_count,
        )
    ):
        raise ValueError("DataViewManifest violates zero-overlap invariants")
    return {
        "schema": "mlevolve_data_view_verification_v1",
        "status": "pass",
        "manifest_hash": manifest.manifest_hash,
        "contract_hash": manifest.contract_hash,
        "terminal_exposure_count": 0,
    }


def _validate_mount_source(root: Path, source: Path, role: str) -> None:
    if source.is_symlink():
        raise ValueError(f"Refusing symlink mount source for {role}")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"Mount source for {role} escapes the Host data root") from error
    lowered = resolved.as_posix().lower()
    if any(marker in lowered for marker in _TERMINAL_MARKERS):
        raise ValueError("Terminal data cannot be mounted in training")


def build_training_mount_contract(
    manifest: DataViewManifest,
    data_root: str | Path,
    *,
    extra_mounts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    root = Path(data_root).resolve(strict=True)
    mounts: list[dict[str, Any]] = [
        {
            "role": "train",
            "source": str((root / "train_view").resolve(strict=True)),
            "target": _TRAIN_TARGETS["train"],
            "read_only": True,
        },
        {
            "role": "internal_validation",
            "source": str((root / "internal_validation_view").resolve(strict=True)),
            "target": _TRAIN_TARGETS["internal_validation"],
            "read_only": True,
        },
    ]
    if "inference" in manifest.views:
        mounts.append(
            {
                "role": "inference",
                "source": str((root / "inference_view").resolve(strict=True)),
                "target": _TRAIN_TARGETS["inference"],
                "read_only": True,
            }
        )
    mounts.append(
        {
            "role": "manifest",
            "source": str((root / "DATA_VIEW_MANIFEST.json").resolve(strict=True)),
            "target": _TRAIN_TARGETS["manifest"],
            "read_only": True,
        }
    )
    mounts.extend(dict(value) for value in extra_mounts)
    roles = [str(mount.get("role") or "") for mount in mounts]
    expected_roles = ["train", "internal_validation"]
    if "inference" in manifest.views:
        expected_roles.append("inference")
    expected_roles.append("manifest")
    if roles != expected_roles:
        raise ValueError("Training mount roles must be exactly the Host-owned allowlist")
    for mount in mounts:
        role = str(mount["role"])
        if mount.get("target") != _TRAIN_TARGETS[role]:
            raise ValueError(f"Unexpected training mount target for {role}")
        if mount.get("read_only") is not True:
            raise ValueError("All Host data views must be mounted read-only")
        _validate_mount_source(root, Path(str(mount["source"])), role)
    payload = {
        "schema": TRAINING_MOUNT_CONTRACT_SCHEMA,
        "contract_hash": manifest.contract_hash,
        "data_view_manifest_hash": manifest.manifest_hash,
        "mounts": mounts,
        "terminal_mount_count": 0,
        "mount_contract_hash": "",
    }
    payload["mount_contract_hash"] = _hash_payload(payload, "mount_contract_hash")
    path = root / "TRAINING_MOUNT_CONTRACT.json"
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to replace immutable training mount contract")
    else:
        _write_exclusive(path, content)
    return payload


def build_evaluator_launch_contract(
    manifest: DataViewManifest,
    training_mount_contract: Mapping[str, Any],
    training_deletion_attestation: Mapping[str, Any],
    preterminal_closure_report: Mapping[str, Any],
) -> dict[str, Any]:
    if training_mount_contract.get("schema") != TRAINING_MOUNT_CONTRACT_SCHEMA:
        raise ValueError("Unknown training mount contract schema")
    if training_mount_contract.get("mount_contract_hash") != _hash_payload(
        training_mount_contract, "mount_contract_hash"
    ):
        raise ValueError("Training mount contract hash mismatch")
    mounts = training_mount_contract.get("mounts")
    expected_roles = ["train", "internal_validation"]
    if "inference" in manifest.views:
        expected_roles.append("inference")
    expected_roles.append("manifest")
    if not isinstance(mounts, list) or [
        str(mount.get("role") or "") for mount in mounts
    ] != expected_roles:
        raise ValueError("Training mount contract roles are not the Host allowlist")
    for mount in mounts:
        role = str(mount["role"])
        if mount.get("target") != _TRAIN_TARGETS[role]:
            raise ValueError("Training mount contract target is not allowed")
        if mount.get("read_only") is not True:
            raise ValueError("Training mount contract contains a writable data view")
        source = Path(str(mount.get("source") or ""))
        if source.is_symlink() or not source.exists():
            raise ValueError("Training mount contract source is unavailable or a symlink")
        if any(marker in source.as_posix().lower() for marker in _TERMINAL_MARKERS):
            raise ValueError("Training mount contract exposes terminal data")
    if training_mount_contract.get("contract_hash") != manifest.contract_hash:
        raise ValueError("Training mount contract is not bound to DataViewManifest")
    if training_mount_contract.get("terminal_mount_count") != 0:
        raise ValueError("Training terminal exposure prevents evaluator launch")
    if preterminal_closure_report.get("schema") != (
        "mlevolve_preterminal_evidence_closure_v1"
    ):
        raise ValueError("Pre-terminal Evidence Closure schema mismatch")
    if preterminal_closure_report.get("report_hash") != _hash_payload(
        preterminal_closure_report, "report_hash"
    ):
        raise ValueError("Pre-terminal Evidence Closure hash mismatch")
    if preterminal_closure_report.get("status") != "pass" or (
        preterminal_closure_report.get("evaluator_launch_authorized") is not True
    ):
        raise ValueError("Evaluator requires Pre-terminal Evidence Closure PASS")
    if preterminal_closure_report.get("contract_hash") != manifest.contract_hash or (
        preterminal_closure_report.get("data_view_manifest_hash")
        != manifest.manifest_hash
    ):
        raise ValueError("Pre-terminal Evidence Closure binding mismatch")
    if preterminal_closure_report.get("terminal_exposure_count") != 0:
        raise ValueError("Pre-terminal Evidence Closure contains terminal exposure")
    if preterminal_closure_report.get("terminal_score_observed") is not False:
        raise ValueError("Pre-terminal Closure may not contain a terminal score")
    if training_deletion_attestation.get("not_found_verified") is not True or (
        training_deletion_attestation.get("kubernetes_reason") != "NotFound"
    ):
        raise ValueError("Evaluator requires verified training Pod NotFound")
    if training_deletion_attestation.get("contract_hash") != manifest.contract_hash:
        raise ValueError("Training deletion attestation contract mismatch")
    if training_deletion_attestation.get("data_view_manifest_hash") != (
        manifest.manifest_hash
    ):
        raise ValueError("Training deletion attestation data-view mismatch")
    if training_deletion_attestation.get("preterminal_closure_report_hash") != (
        preterminal_closure_report["report_hash"]
    ):
        raise ValueError("Training deletion attestation closure mismatch")
    if training_deletion_attestation.get("schema") not in {
        "decision_admissibility_wp8_tier2_training_pod_deletion_attestation_v1",
        "mlevolve_training_pod_deletion_attestation_v2",
    }:
        raise ValueError("Unknown training deletion attestation schema")
    if training_deletion_attestation.get("verified_by") != "host_launcher":
        raise ValueError("Training deletion must be verified by the Host launcher")
    if training_deletion_attestation.get("terminal_metric_observed_before_not_found") is not False:
        raise ValueError("Training deletion attestation violates terminal isolation")
    attestation_hash = str(training_deletion_attestation.get("attestation_hash") or "")
    if len(attestation_hash) != 64 or attestation_hash != _hash_payload(
        training_deletion_attestation, "attestation_hash"
    ):
        raise ValueError("Training deletion attestation must be hash-bound")
    payload = {
        "schema": EVALUATOR_LAUNCH_CONTRACT_SCHEMA,
        "contract_hash": manifest.contract_hash,
        "data_view_manifest_hash": manifest.manifest_hash,
        "training_mount_contract_hash": training_mount_contract[
            "mount_contract_hash"
        ],
        "training_deletion_attestation_hash": attestation_hash,
        "preterminal_closure_report_hash": preterminal_closure_report[
            "report_hash"
        ],
        "training_not_found_verified": True,
        "terminal_view_ref_sha256": manifest.terminal_view_ref_sha256,
        "evaluation_system": "existing_fixed_holdout_terminal_evaluator",
        "launch_contract_hash": "",
    }
    payload["launch_contract_hash"] = _hash_payload(payload, "launch_contract_hash")
    return payload


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
]
