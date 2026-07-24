from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CORPUS_SCHEMA = "corpus_manifest_v1"
AUDIT_SIDECAR_SCHEMA = "audit_sidecar_v1"
SPLIT_SCHEMA = "memory_split_manifest_v1"
BUNDLE_SCHEMA = "memory_bundle_manifest_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((jsonable(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(payload: Mapping[str, Any], hash_field: str) -> str:
    clean = dict(payload)
    clean.pop(hash_field, None)
    return sha256_json(clean)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(jsonable(payload), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass
class CorpusRunEntry:
    run_id: str
    task_id: str
    canonical_task_id: str
    task_family: str
    seed: str
    status: str
    journal_path: str | None
    config_path: str | None
    filtered_journal_path: str | None
    best_solution_path: str | None
    artifact_hashes: dict[str, str]
    node_count: int
    code_node_count: int
    metric_node_count: int
    source_relpath: str = ""
    exclusion_reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CorpusRunEntry":
        values = dict(payload)
        values.setdefault("source_relpath", "")
        values.setdefault("exclusion_reason", "")
        values.setdefault("warnings", [])
        return cls(**values)


@dataclass
class CorpusManifestV1:
    corpus_id: str
    created_at: str
    source_repo: str
    source_commit: str
    source_root: str
    exclusion_rules: list[dict[str, Any]]
    runs: list[CorpusRunEntry]
    expected_snapshot: dict[str, Any]
    actual_snapshot: dict[str, Any]
    split_manifests: list[str] = field(default_factory=list)
    manifest_sha256: str = ""
    schema: str = CORPUS_SCHEMA

    def as_dict(self, *, finalize: bool = False) -> dict[str, Any]:
        payload = jsonable(self)
        if finalize:
            payload["manifest_sha256"] = payload_hash(
                payload, "manifest_sha256"
            )
        return payload

    def finalize(self) -> "CorpusManifestV1":
        self.manifest_sha256 = payload_hash(
            self.as_dict(), "manifest_sha256"
        )
        return self

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        verify: bool = True,
    ) -> "CorpusManifestV1":
        values = dict(payload)
        runs = [CorpusRunEntry.from_dict(item) for item in values.pop("runs", [])]
        result = cls(runs=runs, **values)
        if result.schema != CORPUS_SCHEMA:
            raise ValueError(f"Unsupported corpus schema: {result.schema}")
        if verify and result.manifest_sha256 != payload_hash(
            payload, "manifest_sha256"
        ):
            raise ValueError("Corpus manifest hash mismatch")
        return result


@dataclass
class AuditSidecarV1:
    artifact_id: str
    run_id: str
    node_id: str
    code_sha256: str
    detector_schema: str
    detector_version: str
    active_protocol_ref: str
    status: str
    issues: list[dict[str, Any]]
    legacy_receipt_level: str
    generated_at: str
    source_journal_sha256: str = ""
    sidecar_sha256: str = ""
    schema: str = AUDIT_SIDECAR_SCHEMA

    def as_dict(self, *, finalize: bool = False) -> dict[str, Any]:
        payload = jsonable(self)
        if finalize:
            payload["sidecar_sha256"] = payload_hash(
                payload, "sidecar_sha256"
            )
        return payload

    def finalize(self) -> "AuditSidecarV1":
        self.sidecar_sha256 = payload_hash(
            self.as_dict(), "sidecar_sha256"
        )
        return self


@dataclass
class SplitManifestV1:
    split_id: str
    split_kind: str
    split_version: str
    corpus_manifest_hash: str
    created_at: str
    source_run_ids: list[str]
    heldout_run_ids: list[str]
    source_task_ids: list[str]
    heldout_task_ids: list[str]
    source_seed_groups: list[str] = field(default_factory=list)
    heldout_seed_groups: list[str] = field(default_factory=list)
    excluded_run_ids: list[str] = field(default_factory=list)
    allocation: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    manifest_sha256: str = ""
    schema: str = SPLIT_SCHEMA

    def as_dict(self, *, finalize: bool = False) -> dict[str, Any]:
        payload = jsonable(self)
        if finalize:
            payload["manifest_sha256"] = payload_hash(
                payload, "manifest_sha256"
            )
        return payload

    def finalize(self) -> "SplitManifestV1":
        self.manifest_sha256 = payload_hash(
            self.as_dict(), "manifest_sha256"
        )
        return self

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        verify: bool = True,
    ) -> "SplitManifestV1":
        result = cls(**dict(payload))
        if result.schema != SPLIT_SCHEMA:
            raise ValueError(f"Unsupported split schema: {result.schema}")
        if verify and result.manifest_sha256 != payload_hash(
            payload, "manifest_sha256"
        ):
            raise ValueError("Split manifest hash mismatch")
        return result


@dataclass
class MemoryBundleManifestV1:
    bundle_id: str
    bundle_version: str
    parent_bundle: str | None
    corpus_manifest_hash: str
    protocol_registry_hash: str
    authority_policy_version: str
    detector_version: str
    deepseek_model: str
    deepseek_prompt_hash: str
    graph_hashes: dict[str, str]
    index_hashes: dict[str, str]
    lineage_hash: str
    split_id: str
    certification_level: str
    build_report: str
    created_at: str = field(default_factory=utc_now)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    manifest_sha256: str = ""
    schema: str = BUNDLE_SCHEMA

    def as_dict(self, *, finalize: bool = False) -> dict[str, Any]:
        payload = jsonable(self)
        if finalize:
            payload["manifest_sha256"] = payload_hash(
                payload, "manifest_sha256"
            )
        return payload

    def finalize(self) -> "MemoryBundleManifestV1":
        self.manifest_sha256 = payload_hash(
            self.as_dict(), "manifest_sha256"
        )
        return self

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        verify: bool = True,
    ) -> "MemoryBundleManifestV1":
        result = cls(**dict(payload))
        if result.schema != BUNDLE_SCHEMA:
            raise ValueError(f"Unsupported bundle schema: {result.schema}")
        if verify and result.manifest_sha256 != payload_hash(
            payload, "manifest_sha256"
        ):
            raise ValueError("Memory bundle manifest hash mismatch")
        return result
