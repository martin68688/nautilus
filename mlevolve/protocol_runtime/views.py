"""Immutable Candidate-visible handles for Host materialized views."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

from authority.protocol_execution_contract import ProtocolExecutionContract

from .collector import HostCollectorSidecar
from .data_views import read_data_view_manifest, verify_data_view_manifest
from .errors import InvalidViewHandle


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class DataViewHandle:
    role: str
    view_ref: str
    contract_hash: str
    manifest_hash: str
    data_sha256: str
    _path: Path = field(repr=False, compare=False)
    _capability: str = field(repr=False, compare=False)

    def records(self) -> tuple[dict[str, Any], ...]:
        if self._path.is_symlink():
            raise InvalidViewHandle("Data view became a symlink")
        if _sha256_file(self._path) != self.data_sha256:
            raise InvalidViewHandle("Data view changed after Host materialization")
        rows = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise InvalidViewHandle("Data view row is not an object")
            rows.append(value)
        return tuple(rows)


@dataclass(frozen=True)
class ProtocolSplit:
    train: DataViewHandle
    validation: DataViewHandle
    inference: DataViewHandle | None = None

    @property
    def test(self) -> DataViewHandle | None:
        """Compatibility alias for unlabeled submission inference data."""

        return self.inference


def build_view_handles(
    manifest_path: str | Path,
    contract: ProtocolExecutionContract,
    collector: HostCollectorSidecar,
) -> ProtocolSplit:
    verify_data_view_manifest(manifest_path, contract=contract)
    manifest_file = Path(manifest_path).resolve(strict=True)
    manifest = read_data_view_manifest(manifest_file)
    root = manifest_file.parent
    handles = {}
    event_kinds = {
        "train": ("fit_scope", "split_lineage"),
        "internal_validation": (
            "evaluator",
            "prediction_scope",
            "selection_freeze",
            "split_lineage",
        ),
        # Submission inference happens only after selection freeze and does
        # not mint a new training/evaluation event.
        "inference": (),
    }
    for role, metadata in manifest.views.items():
        path = (root / str(metadata["relative_path"])).resolve(strict=True)
        capability = collector.issue_view_capability(
            role=role,
            view_ref=str(metadata["view_ref"]),
            data_sha256=str(metadata["data_sha256"]),
            event_kinds=event_kinds[role],
        )
        handles[role] = DataViewHandle(
            role=role,
            view_ref=str(metadata["view_ref"]),
            contract_hash=contract.contract_hash,
            manifest_hash=manifest.manifest_hash,
            data_sha256=str(metadata["data_sha256"]),
            _path=path,
            _capability=capability,
        )
    return ProtocolSplit(
        train=handles["train"],
        validation=handles["internal_validation"],
        inference=handles.get("inference"),
    )


__all__ = ["DataViewHandle", "ProtocolSplit", "build_view_handles"]
