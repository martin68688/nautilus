from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .domain_scope import canonical_domain, transfer_is_compatible
from .models import Operation, canonical_operation
from .protocol_registry import canonical_json

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


BUNDLE_MANIFEST_SCHEMA = "memory_bundle_manifest_v1"
CURRENT_POINTER_SCHEMA = "memory_bundle_current_v1"
OVERLAY_EVENT_SCHEMA = "session_overlay_event_v1"
OVERLAY_MANIFEST_SCHEMA = "session_overlay_manifest_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(_jsonable(value)).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: Mapping[str, Any], field_name: str) -> str:
    clean = dict(payload)
    clean.pop(field_name, None)
    return sha256_json(clean)


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform-specific fallback
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _safe_relative_path(root: Path, raw: str, *, label: str) -> Path:
    relative = Path(str(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe {label} path: {raw}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"{label} escapes its root: {raw}")
    return resolved


def verify_bundle_directory(
    bundle_path: str | Path,
    *,
    verify_artifacts: bool = True,
    allow_staging: bool = False,
) -> dict[str, Any]:
    bundle_path = Path(bundle_path).resolve()
    if not bundle_path.is_dir():
        raise FileNotFoundError(f"Memory bundle directory does not exist: {bundle_path}")
    if not allow_staging and any(
        part.startswith(".staging-")
        or part.startswith(".failed-")
        or part.startswith(".inputs-")
        for part in bundle_path.parts
    ):
        raise ValueError("Staging/failed/input directories are not loadable bundles")
    manifest_path = bundle_path / "manifest.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("Memory bundle manifest must be an object")
    if manifest.get("schema") != BUNDLE_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported memory bundle schema: {manifest.get('schema')}")
    expected_manifest_hash = _payload_hash(manifest, "manifest_sha256")
    if manifest.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("Memory bundle manifest hash mismatch")
    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("Memory bundle has no artifact hash inventory")
    if verify_artifacts:
        for relative, expected in sorted(artifact_hashes.items()):
            target = _safe_relative_path(
                bundle_path, str(relative), label="bundle artifact"
            )
            if not target.is_file():
                raise FileNotFoundError(f"Missing bundle artifact: {relative}")
            if sha256_file(target) != str(expected):
                raise ValueError(f"Bundle artifact hash mismatch: {relative}")
    return manifest


@dataclass(frozen=True)
class ImmutableBaseBundle:
    path: Path
    manifest: dict[str, Any]
    manifest_file_sha256: str

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        verify_artifacts: bool = True,
    ) -> ImmutableBaseBundle:
        resolved = Path(path).resolve()
        manifest = verify_bundle_directory(
            resolved, verify_artifacts=verify_artifacts
        )
        return cls(
            path=resolved,
            manifest=copy.deepcopy(manifest),
            manifest_file_sha256=sha256_file(resolved / "manifest.json"),
        )

    @property
    def bundle_id(self) -> str:
        return str(self.manifest["bundle_id"])

    @property
    def bundle_version(self) -> str:
        return str(self.manifest["bundle_version"])

    @property
    def manifest_sha256(self) -> str:
        return str(self.manifest["manifest_sha256"])

    def assert_unchanged(self) -> None:
        current = verify_bundle_directory(self.path, verify_artifacts=True)
        if current != self.manifest:
            raise RuntimeError("Published Base Bundle manifest changed after load")
        if sha256_file(self.path / "manifest.json") != self.manifest_file_sha256:
            raise RuntimeError("Published Base Bundle file changed after load")

    def _assert_manifest_unchanged(self) -> None:
        if sha256_file(self.path / "manifest.json") != self.manifest_file_sha256:
            raise RuntimeError("Published Base Bundle manifest changed after load")
        current = _read_json(self.path / "manifest.json")
        if current != self.manifest:
            raise RuntimeError("Published Base Bundle manifest payload changed after load")

    def _verified_artifact(self, relative: str) -> Path:
        self._assert_manifest_unchanged()
        expected = (self.manifest.get("artifact_hashes") or {}).get(relative)
        if expected is None:
            raise PermissionError(f"Base read is not declared by the manifest: {relative}")
        target = _safe_relative_path(self.path, relative, label="base read")
        if not target.is_file():
            raise FileNotFoundError(f"Missing Base artifact: {relative}")
        if sha256_file(target) != str(expected):
            raise RuntimeError(f"Published Base artifact changed after load: {relative}")
        return target

    def _read_json_artifacts(self, relatives: Iterable[str]) -> dict[str, Any]:
        """Read several manifest-bound JSON artifacts with one manifest check."""

        self._assert_manifest_unchanged()
        artifact_hashes = self.manifest.get("artifact_hashes") or {}
        output: dict[str, Any] = {}
        for raw_relative in relatives:
            relative = str(raw_relative)
            expected = artifact_hashes.get(relative)
            if expected is None:
                raise PermissionError(
                    f"Base read is not declared by the manifest: {relative}"
                )
            target = _safe_relative_path(self.path, relative, label="base read")
            if not target.is_file():
                raise FileNotFoundError(f"Missing Base artifact: {relative}")
            if sha256_file(target) != str(expected):
                raise RuntimeError(
                    f"Published Base artifact changed after load: {relative}"
                )
            output[relative] = _read_json(target)
        return output

    def read_json(self, relative: str) -> Any:
        return self._read_json_artifacts([relative])[relative]

    def read_jsonl(self, relative: str) -> list[dict[str, Any]]:
        target = self._verified_artifact(relative)
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Base JSONL row {line_number} is not an object")
            rows.append(value)
        return rows

    def verify_run_identity_provenance(self) -> dict[str, Any]:
        """Verify the hash-bound source-membership and code-audit evidence.

        Legacy standalone graphs carried two booleans in ``graph.meta``.  A
        manifest-driven Bundle instead binds the evidence that establishes
        those facts: corpus/split manifests, drift review, the complete audit
        sidecar index, and both build reports.  Keep that verification here so
        callers never have to mutate a published graph to add legacy flags.
        """

        graph = self.read_json("runforest/graph.json")
        corpus = self.read_json("corpus/manifest.json")
        drift_review = self.read_json("corpus/drift_review.json")
        split = self.read_json("splits/active.json")
        audit_index = self.read_json("audit_sidecars/index.json")
        runforest_report = self.read_json("runforest/build_report.json")
        build_report_path = str(self.manifest.get("build_report") or "")
        if not build_report_path:
            raise ValueError("Bundle provenance has no build report")
        build_report = self.read_json(build_report_path)

        errors: list[str] = []

        def require(condition: bool, label: str) -> None:
            if not condition:
                errors.append(label)

        def count(payload: Mapping[str, Any], field_name: str) -> int:
            try:
                return int(payload.get(field_name, -1))
            except (TypeError, ValueError):
                return -1

        certification = str(self.manifest.get("certification_level") or "")
        # Formal Tier-2 child Bundles preserve the parent's audited graph while
        # narrowing it to one domain and publishing either a certified or a
        # provisional method Claim.  Those are explicit certification states,
        # not arbitrary prefixes; keep unknown formal_domain_* values closed.
        require(
            certification
            in {
                "raw_audited",
                "certified",
                "formal_domain_certified",
                "formal_domain_provisional",
            },
            "certification_level",
        )
        require(corpus.get("schema") == "corpus_manifest_v1", "corpus_schema")
        require(
            split.get("schema") == "memory_split_manifest_v1",
            "split_schema",
        )

        corpus_hash = str(corpus.get("manifest_sha256") or "")
        split_hash = str(split.get("manifest_sha256") or "")
        require(
            corpus_hash == _payload_hash(corpus, "manifest_sha256"),
            "corpus_manifest_hash",
        )
        require(
            corpus_hash == str(self.manifest.get("corpus_manifest_hash") or ""),
            "bundle_corpus_binding",
        )
        require(
            split_hash == _payload_hash(split, "manifest_sha256"),
            "split_manifest_hash",
        )
        require(
            str(split.get("corpus_manifest_hash") or "") == corpus_hash,
            "split_corpus_binding",
        )
        require(
            str(split.get("split_id") or "")
            == str(self.manifest.get("split_id") or ""),
            "bundle_split_binding",
        )

        meta = graph.get("meta") if isinstance(graph, Mapping) else None
        meta = dict(meta) if isinstance(meta, Mapping) else {}
        require(meta.get("bundle_id") == self.bundle_id, "graph_bundle_binding")
        require(
            meta.get("corpus_manifest_hash") == corpus_hash,
            "graph_corpus_binding",
        )
        require(meta.get("split_id") == split.get("split_id"), "graph_split_binding")
        require(
            meta.get("split_manifest_hash") == split_hash,
            "graph_split_hash_binding",
        )
        require(
            meta.get("corpus_id") == corpus.get("corpus_id"),
            "graph_corpus_id_binding",
        )
        graph_certification = str(meta.get("certification_level") or "")
        require(
            graph_certification == "raw_audited"
            or graph_certification == certification,
            "graph_certification_level",
        )
        require(
            meta.get("legacy_artifact_overwritten") is False,
            "graph_legacy_overwrite",
        )

        actual_snapshot = corpus.get("actual_snapshot")
        require(isinstance(actual_snapshot, Mapping), "corpus_actual_snapshot")
        require(
            drift_review.get("schema") == "corpus_drift_review_v1",
            "drift_review_schema",
        )
        require(drift_review.get("reviewed") is True, "drift_reviewed")
        require(
            drift_review.get("excluded_runs_reviewed") is True,
            "drift_exclusions_reviewed",
        )
        require(
            drift_review.get("corpus_manifest_hash") == corpus_hash,
            "drift_corpus_binding",
        )
        if isinstance(actual_snapshot, Mapping):
            require(
                drift_review.get("actual_snapshot_hash")
                == sha256_json(actual_snapshot),
                "drift_snapshot_binding",
            )

        source_run_values = [str(value) for value in split.get("source_run_ids") or []]
        heldout_run_values = [str(value) for value in split.get("heldout_run_ids") or []]
        source_runs = set(source_run_values)
        heldout_runs = set(heldout_run_values)
        require(bool(source_runs), "empty_source_split")
        require(len(source_runs) == len(source_run_values), "duplicate_source_runs")
        require(len(heldout_runs) == len(heldout_run_values), "duplicate_heldout_runs")
        require(not source_runs & heldout_runs, "split_run_overlap")
        validation = split.get("validation")
        validation = dict(validation) if isinstance(validation, Mapping) else {}
        require(count(validation, "run_overlap_count") == 0, "declared_run_overlap")
        require(not (validation.get("run_overlap") or []), "declared_run_overlap_ids")
        split_kind = str(split.get("split_kind") or "")
        if split_kind == "task-heldout":
            require(
                not set(map(str, split.get("source_task_ids") or []))
                & set(map(str, split.get("heldout_task_ids") or [])),
                "task_split_overlap",
            )
            require(
                count(validation, "task_overlap_count") == 0,
                "declared_task_overlap",
            )
        if split_kind == "seed-heldout":
            require(
                not set(map(str, split.get("source_seed_groups") or []))
                & set(map(str, split.get("heldout_seed_groups") or [])),
                "seed_split_overlap",
            )
            require(
                count(validation, "seed_group_overlap_count") == 0,
                "declared_seed_overlap",
            )

        nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
        nodes = list(nodes) if isinstance(nodes, list) else []
        node_ids = [str(node.get("id")) for node in nodes if isinstance(node, Mapping)]
        require(len(node_ids) == len(nodes), "graph_node_shape")
        require(len(set(node_ids)) == len(node_ids), "duplicate_graph_node_ids")
        graph_run_ids = {
            str(node.get("run_id"))
            for node in nodes
            if isinstance(node, Mapping) and node.get("run_id") is not None
        }
        require(graph_run_ids <= source_runs, "graph_run_outside_source_split")
        require(not graph_run_ids & heldout_runs, "heldout_run_in_graph")
        require(
            not any(
                "spooky" in str(node.get("task") or "").lower()
                for node in nodes
                if isinstance(node, Mapping)
            ),
            "spooky_node_in_graph",
        )
        code_nodes = {
            str(node.get("id")): node
            for node in nodes
            if isinstance(node, Mapping)
            and node.get("type") == "RunNode"
            and node.get("code_sha256")
        }

        entries = audit_index.get("entries")
        entries = dict(entries) if isinstance(entries, Mapping) else {}
        require(
            audit_index.get("schema") == "audit_sidecar_index_v1",
            "audit_index_schema",
        )
        require(
            audit_index.get("corpus_manifest_hash") == corpus_hash,
            "audit_corpus_binding",
        )
        require(
            audit_index.get("detector_version")
            == self.manifest.get("detector_version"),
            "audit_detector_binding",
        )
        require(set(entries) == set(code_nodes), "audit_sidecar_coverage")
        require(
            len(set(map(str, entries.values()))) == len(entries),
            "duplicate_audit_sidecar_files",
        )
        artifact_hashes = self.manifest.get("artifact_hashes") or {}
        require(
            all(
                f"audit_sidecars/{filename}" in artifact_hashes
                and not Path(str(filename)).is_absolute()
                and ".." not in Path(str(filename)).parts
                for filename in entries.values()
            ),
            "audit_sidecar_hash_inventory",
        )
        sidecar_relatives = {
            str(artifact_id): f"audit_sidecars/{filename}"
            for artifact_id, filename in entries.items()
        }
        if all(relative in artifact_hashes for relative in sidecar_relatives.values()):
            sidecars = self._read_json_artifacts(sidecar_relatives.values())
            for artifact_id, relative in sidecar_relatives.items():
                sidecar = sidecars[relative]
                node = code_nodes.get(artifact_id) or {}
                require(
                    isinstance(sidecar, Mapping)
                    and sidecar.get("schema") == "audit_sidecar_v1",
                    "audit_sidecar_schema",
                )
                if not isinstance(sidecar, Mapping):
                    continue
                require(
                    sidecar.get("artifact_id") == artifact_id,
                    "audit_sidecar_artifact_binding",
                )
                require(
                    str(sidecar.get("run_id") or "") in source_runs,
                    "audit_sidecar_source_binding",
                )
                require(
                    sidecar.get("code_sha256") == node.get("code_sha256"),
                    "audit_sidecar_code_binding",
                )
                require(
                    sidecar.get("sidecar_sha256")
                    == _payload_hash(sidecar, "sidecar_sha256"),
                    "audit_sidecar_payload_hash",
                )

        expected_source_count = len(source_runs)
        expected_heldout_count = len(heldout_runs)
        require(
            count(meta, "source_run_count") == expected_source_count,
            "graph_source_count",
        )
        require(
            count(meta, "heldout_run_count") == expected_heldout_count,
            "graph_heldout_count",
        )
        require(
            runforest_report.get("bundle_id") == self.bundle_id,
            "runforest_report_bundle_binding",
        )
        require(
            runforest_report.get("split_manifest_hash") == split_hash,
            "runforest_report_split_binding",
        )
        require(
            count(runforest_report, "source_run_count") == expected_source_count,
            "runforest_source_count",
        )
        require(
            count(runforest_report, "heldout_run_count") == expected_heldout_count,
            "runforest_heldout_count",
        )
        require(
            count(runforest_report, "spooky_source_run_count") == 0,
            "runforest_spooky_sources",
        )
        require(
            runforest_report.get("all_code_nodes_have_sidecars") is True,
            "runforest_sidecar_coverage",
        )
        require(
            count(runforest_report, "expected_audited_code_node_count")
            == len(code_nodes),
            "runforest_expected_audit_count",
        )
        require(
            count(runforest_report, "audited_code_node_count") == len(code_nodes),
            "runforest_audit_count",
        )

        require(
            build_report.get("schema") == "memory_bundle_build_report_v1",
            "build_report_schema",
        )
        require(build_report.get("bundle_id") == self.bundle_id, "build_bundle_binding")
        require(build_report.get("split_id") == split.get("split_id"), "build_split_binding")
        require(
            count(build_report, "source_run_count") == expected_source_count,
            "build_source_count",
        )
        require(
            count(build_report, "heldout_run_count") == expected_heldout_count,
            "build_heldout_count",
        )
        require(
            count(build_report, "raw_journal_run_count") == expected_source_count,
            "build_raw_journal_count",
        )
        require(
            count(build_report, "sidecar_count") == len(code_nodes),
            "build_sidecar_count",
        )
        for field_name in (
            "all_code_nodes_have_sidecars",
            "all_clause_sources_resolve",
            "secret_scan_passed",
            "corpus_drift_reviewed",
            "published_atomically",
        ):
            require(build_report.get(field_name) is True, f"build_{field_name}")
        require(
            build_report.get("legacy_artifact_overwritten") is False,
            "build_legacy_overwrite",
        )
        require(
            count(build_report, "spooky_source_run_count") == 0,
            "build_spooky_sources",
        )
        require(
            not (build_report.get("heldout_run_refs_in_graph") or []),
            "build_heldout_graph_refs",
        )

        if errors:
            raise ValueError(
                "Bundle source/leak provenance verification failed: "
                + ",".join(sorted(set(errors)))
            )
        return {
            "source_membership_verified": True,
            "leak_verified": True,
            "source_runs": sorted(source_runs),
            "certification_level": certification,
            "detector_version": str(self.manifest.get("detector_version") or ""),
            "corpus_manifest_hash": corpus_hash,
            "split_manifest_hash": split_hash,
        }

    def write(self, *_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("Published Base Bundle is immutable")


_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    local_lock = _thread_lock(path)
    with local_lock:
        with path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class OverlayEvent:
    sequence: int
    event_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str
    parent_event_hash: str
    event_hash: str
    schema: str = OVERLAY_EVENT_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


class SessionOverlay:
    """Append-only, hash-chained run/session memory overlay."""

    def __init__(self, path: str | Path, *, overlay_id: str = "") -> None:
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.events_path = self.path / "events.jsonl"
        self.manifest_path = self.path / "overlay_manifest.json"
        self.lock_path = self.path / ".overlay.lock"
        self.overlay_id = str(overlay_id or f"overlay::{sha256_json(str(self.path))[:24]}")
        with file_lock(self.lock_path):
            if not self.events_path.exists():
                with self.events_path.open("xb") as handle:
                    handle.flush()
                    os.fsync(handle.fileno())
            if not self.manifest_path.exists():
                if self.events_path.stat().st_size:
                    raise RuntimeError("Overlay events exist without a manifest")
                self._write_manifest_unlocked([], created_at=_utc_now())
            self._validate_unlocked()

    @staticmethod
    def _event_from_dict(payload: Mapping[str, Any]) -> OverlayEvent:
        value = dict(payload)
        if value.get("schema") != OVERLAY_EVENT_SCHEMA:
            raise ValueError("Unsupported overlay event schema")
        expected_hash = _payload_hash(value, "event_hash")
        if value.get("event_hash") != expected_hash:
            raise ValueError("Overlay event hash mismatch")
        identity_payload = dict(value)
        identity_payload.pop("event_id", None)
        identity_payload.pop("event_hash", None)
        expected_id = f"overlay_event::{sha256_json(identity_payload)[:24]}"
        if value.get("event_id") != expected_id:
            raise ValueError("Overlay event ID mismatch")
        return OverlayEvent(**value)

    def _read_events_unlocked(self) -> list[OverlayEvent]:
        events: list[OverlayEvent] = []
        parent = ""
        seen_ids: set[str] = set()
        for line_number, line in enumerate(
            self.events_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Overlay event line {line_number} is not valid JSON"
                ) from error
            if not isinstance(raw, dict):
                raise ValueError(f"Overlay event line {line_number} is not an object")
            event = self._event_from_dict(raw)
            if event.sequence != len(events) + 1:
                raise ValueError("Overlay event sequence is not append-only")
            if event.parent_event_hash != parent:
                raise ValueError("Overlay event hash chain is broken")
            if event.event_id in seen_ids:
                raise ValueError("Duplicate overlay event ID")
            events.append(event)
            parent = event.event_hash
            seen_ids.add(event.event_id)
        return events

    def _manifest_payload(
        self,
        events: list[OverlayEvent],
        *,
        created_at: str,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        payload = {
            "schema": OVERLAY_MANIFEST_SCHEMA,
            "overlay_id": self.overlay_id,
            "created_at": created_at,
            "updated_at": updated_at or created_at,
            "event_count": len(events),
            "last_event_hash": events[-1].event_hash if events else "",
            "events_sha256": sha256_file(self.events_path),
            "manifest_sha256": "",
        }
        payload["manifest_sha256"] = _payload_hash(payload, "manifest_sha256")
        return payload

    def _write_manifest_unlocked(
        self,
        events: list[OverlayEvent],
        *,
        created_at: str,
    ) -> dict[str, Any]:
        manifest = self._manifest_payload(
            events, created_at=created_at, updated_at=_utc_now()
        )
        write_json_atomic(self.manifest_path, manifest)
        return manifest

    def _validate_unlocked(self) -> tuple[list[OverlayEvent], dict[str, Any]]:
        events = self._read_events_unlocked()
        manifest = _read_json(self.manifest_path)
        if not isinstance(manifest, dict):
            raise ValueError("Overlay manifest must be an object")
        if manifest.get("schema") != OVERLAY_MANIFEST_SCHEMA:
            raise ValueError("Unsupported overlay manifest schema")
        if manifest.get("overlay_id") != self.overlay_id:
            raise ValueError("Overlay ID mismatch")
        if manifest.get("manifest_sha256") != _payload_hash(
            manifest, "manifest_sha256"
        ):
            raise ValueError("Overlay manifest hash mismatch")
        expected = self._manifest_payload(
            events,
            created_at=str(manifest.get("created_at") or ""),
            updated_at=str(manifest.get("updated_at") or ""),
        )
        for key in (
            "event_count",
            "last_event_hash",
            "events_sha256",
        ):
            if manifest.get(key) != expected.get(key):
                raise ValueError(f"Overlay manifest does not match events: {key}")
        return events, manifest

    @property
    def manifest(self) -> dict[str, Any]:
        with file_lock(self.lock_path):
            _events, manifest = self._validate_unlocked()
            return copy.deepcopy(manifest)

    def events(self) -> list[OverlayEvent]:
        with file_lock(self.lock_path):
            events, _manifest = self._validate_unlocked()
            return copy.deepcopy(events)

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        created_at: str | None = None,
    ) -> OverlayEvent:
        event_type = str(event_type).strip()
        if not event_type:
            raise ValueError("Overlay event_type is required")
        with file_lock(self.lock_path):
            events, manifest = self._validate_unlocked()
            event_payload = {
                "schema": OVERLAY_EVENT_SCHEMA,
                "sequence": len(events) + 1,
                "event_id": "",
                "event_type": event_type,
                "payload": _jsonable(dict(payload)),
                "created_at": created_at or _utc_now(),
                "parent_event_hash": events[-1].event_hash if events else "",
                "event_hash": "",
            }
            # event_id is derived from the same payload as event_hash. Exclude
            # both self-referential fields from the first digest.
            identity_payload = dict(event_payload)
            identity_payload.pop("event_id")
            identity_payload.pop("event_hash")
            identity_digest = sha256_json(identity_payload)
            event_payload["event_id"] = f"overlay_event::{identity_digest[:24]}"
            event_payload["event_hash"] = _payload_hash(event_payload, "event_hash")
            event = OverlayEvent(**event_payload)
            encoded = (
                json.dumps(
                    event.as_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            ).encode("utf-8")
            with self.events_path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            events.append(event)
            self._write_manifest_unlocked(
                events, created_at=str(manifest["created_at"])
            )
            _fsync_directory(self.path)
            return copy.deepcopy(event)

    def freeze_to(self, destination: str | Path) -> dict[str, Any]:
        destination = Path(destination).resolve()
        if destination.exists():
            raise FileExistsError(f"Overlay freeze destination exists: {destination}")
        with file_lock(self.lock_path):
            events, manifest = self._validate_unlocked()
            destination.mkdir(parents=True)
            target_events = destination / "events.jsonl"
            with self.events_path.open("rb") as source, target_events.open("xb") as target:
                shutil.copyfileobj(source, target)
                target.flush()
                os.fsync(target.fileno())
            write_json_atomic(destination / "overlay_manifest.json", manifest)
            _fsync_directory(destination)
            return {
                "schema": "session_overlay_freeze_report_v1",
                "overlay_id": self.overlay_id,
                "event_count": len(events),
                "last_event_hash": events[-1].event_hash if events else "",
                "events_sha256": sha256_file(target_events),
                "manifest_sha256": manifest["manifest_sha256"],
                "destination": str(destination),
            }

    @staticmethod
    def _event_claim_types(event: OverlayEvent) -> set[str]:
        payload = event.payload
        clause = payload.get("clause")
        clause = clause if isinstance(clause, Mapping) else {}
        values = [
            *(payload.get("claim_types") or [payload.get("claim_type")]),
            *(clause.get("claim_types") or [clause.get("claim_type")]),
        ]
        return {str(value) for value in values if value not in {None, ""}}

    def visible_events(
        self,
        operation: Operation | str,
        *,
        authority_evaluator: Callable[[OverlayEvent, Operation], bool] | None = None,
    ) -> list[OverlayEvent]:
        operation = canonical_operation(operation)
        output: list[OverlayEvent] = []
        for event in self.events():
            payload = event.payload
            clause = payload.get("clause")
            clause = clause if isinstance(clause, Mapping) else {}
            if operation == Operation.INSPECT:
                output.append(event)
                continue
            if "score" in self._event_claim_types(event) and payload.get("audited") is not True:
                continue
            permitted = {
                str(value)
                for value in (
                    payload.get("permitted_operations")
                    or payload.get("allowed_operations")
                    or clause.get("permitted_operations")
                    or clause.get("allowed_operations")
                    or []
                )
            }
            if permitted and operation.value not in permitted:
                continue
            # Overlay authority is always evaluated online. Missing evaluators
            # fail closed for every non-Inspect operation.
            if authority_evaluator is None or not authority_evaluator(event, operation):
                continue
            output.append(event)
        return output


@dataclass(frozen=True)
class MemorySnapshot:
    base_bundle: ImmutableBaseBundle
    session_overlay: SessionOverlay
    active_protocol_ref: str
    authority_policy_version: str
    snapshot_sha256: str

    @property
    def base_bundle_id(self) -> str:
        return self.base_bundle.bundle_id

    @property
    def base_bundle_path(self) -> str:
        return str(self.base_bundle.path)

    @property
    def session_overlay_path(self) -> str:
        return str(self.session_overlay.path)

    def assert_unchanged(self) -> None:
        self.base_bundle.assert_unchanged()

    def base_clauses(
        self,
        operation: Operation | str,
        *,
        task_id: str,
        task_family: str,
        generation_stage: str,
        governance_stage: str,
    ) -> list[dict[str, Any]]:
        operation = canonical_operation(operation)
        generation_stage = str(generation_stage).strip()
        governance_stage = str(governance_stage).strip()
        if not generation_stage or not governance_stage:
            raise ValueError("Base visibility lookup requires both stage axes")
        mask_path = (
            self.base_bundle.path
            / "visibility"
            / "precompiled_masks"
            / "declared_scope_masks.json"
        )
        if not mask_path.is_file():
            # A missing precompiled mask cannot be replaced by post-ranking
            # filtering for any operation that can influence behavior.
            if operation != Operation.INSPECT:
                return []
            declared_ids: set[str] | None = None
        else:
            masks = self.base_bundle.read_json(
                "visibility/precompiled_masks/declared_scope_masks.json"
            )
            if masks.get("schema") != "declared_scope_visibility_masks_v1":
                raise ValueError("Unsupported Base visibility mask schema")
            mask_key = "|".join(
                [
                    self.active_protocol_ref,
                    operation.value,
                    generation_stage,
                    governance_stage,
                ]
            )
            declared_ids = {
                str(value) for value in (masks.get("masks") or {}).get(mask_key, [])
            }
        clauses = self.base_bundle.read_jsonl("sop/clauses.jsonl")
        output = []
        for clause in clauses:
            if declared_ids is not None and str(clause.get("clause_id")) not in declared_ids:
                continue
            permitted = {
                str(value) for value in clause.get("permitted_operations") or []
            }
            if permitted and operation.value not in permitted:
                continue
            protocols = {str(value) for value in clause.get("protocol_scope") or []}
            if protocols and self.active_protocol_ref not in protocols:
                continue
            scope = clause.get("task_scope") or {}
            task_ids = {
                str(value)
                for value in (scope.get("task_ids") or [scope.get("task_id")])
                if value not in {None, ""}
            }
            source_task_ids = {
                str(value)
                for value in clause.get("source_task_ids") or []
                if value not in {None, ""}
            }
            bound_task_ids = task_ids | source_task_ids
            same_task = bool(bound_task_ids and task_id in bound_task_ids)
            task_families = {
                str(value)
                for value in scope.get("task_families") or []
                if value not in {None, ""}
            }
            if same_task:
                if task_families and task_family:
                    target_domain = canonical_domain(task_family)
                    declared_domains = {
                        canonical_domain(value) for value in task_families
                    }
                    declared_domains.discard("")
                    if target_domain and target_domain not in declared_domains:
                        continue
            else:
                source_domains = {
                    canonical_domain(value)
                    for value in clause.get("source_domains") or []
                }
                source_domains.discard("")
                if not source_domains:
                    source_domains = {
                        canonical_domain(value)
                        for value in clause.get("source_task_families") or []
                    }
                    source_domains.discard("")
                if not source_domains:
                    source_domains = {
                        canonical_domain(value) for value in task_families
                    }
                    source_domains.discard("")
                if not transfer_is_compatible(
                    source_domains,
                    task_family,
                    clause.get("transfer_scope"),
                ):
                    continue
            output.append(clause)
        return output

    def overlay_clauses(
        self,
        operation: Operation | str,
        *,
        authority_evaluator: Callable[[OverlayEvent, Operation], bool] | None = None,
    ) -> list[dict[str, Any]]:
        clauses = []
        for event in self.session_overlay.visible_events(
            operation, authority_evaluator=authority_evaluator
        ):
            if event.event_type != "sop_clause":
                continue
            clause = event.payload.get("clause")
            if isinstance(clause, dict):
                clauses.append(copy.deepcopy(clause))
        return clauses


class MemorySnapshotLoader:
    def __init__(self, bundle_root: str | Path) -> None:
        self.bundle_root = Path(bundle_root).resolve()

    def _read_current(self, current_path: str | Path) -> dict[str, Any]:
        current_path = Path(current_path)
        if not current_path.is_absolute():
            current_path = self.bundle_root / current_path
        pointer = _read_json(current_path)
        if not isinstance(pointer, dict):
            raise ValueError("CURRENT pointer must be an object")
        if pointer.get("schema") != CURRENT_POINTER_SCHEMA:
            raise ValueError("Unsupported CURRENT pointer schema")
        if pointer.get("pointer_sha256") != _payload_hash(pointer, "pointer_sha256"):
            raise ValueError("CURRENT pointer hash mismatch")
        return pointer

    def load_base(
        self,
        *,
        current_path: str | Path = "CURRENT.json",
        verify_artifacts: bool = True,
    ) -> ImmutableBaseBundle:
        """Resolve and verify the immutable Base selected by CURRENT."""

        pointer = self._read_current(current_path)
        relative = str(pointer.get("bundle_path") or "")
        if not relative:
            raise ValueError("CURRENT pointer has no bundle_path")
        base_path = _safe_relative_path(
            self.bundle_root, relative, label="CURRENT bundle"
        )
        base = ImmutableBaseBundle.load(
            base_path, verify_artifacts=verify_artifacts
        )
        if pointer.get("manifest_sha256") != base.manifest_sha256:
            raise ValueError("CURRENT pointer does not bind the Base manifest")
        if pointer.get("bundle_id") != base.bundle_id:
            raise ValueError("CURRENT pointer bundle_id mismatch")
        if pointer.get("bundle_version") != base.bundle_version:
            raise ValueError("CURRENT pointer bundle_version mismatch")
        artifact_paths = set(
            map(str, (base.manifest.get("artifact_hashes") or {}).keys())
        )
        provenance_markers = {
            "corpus/manifest.json",
            "splits/active.json",
            "audit_sidecars/index.json",
        }
        corpus_manifest = {}
        if "corpus/manifest.json" in artifact_paths:
            raw_corpus_manifest = _read_json(base.path / "corpus" / "manifest.json")
            if isinstance(raw_corpus_manifest, dict):
                corpus_manifest = raw_corpus_manifest
        modern_manifest_driven_bundle = bool(
            corpus_manifest.get("schema") == "corpus_manifest_v1"
            or artifact_paths
            & {"splits/active.json", "audit_sidecars/index.json"}
        )
        if modern_manifest_driven_bundle:
            required_provenance = {
                *provenance_markers,
                "corpus/drift_review.json",
                "runforest/graph.json",
                "runforest/build_report.json",
            }
            build_report = str(base.manifest.get("build_report") or "")
            if build_report:
                required_provenance.add(build_report)
            missing = sorted(required_provenance - artifact_paths)
            if missing:
                raise ValueError(
                    "Memory Bundle provenance inventory is incomplete: "
                    + ",".join(missing)
                )
            # Hash-valid manifests are not sufficient: a malicious publisher
            # can re-sign an overlapping source/heldout split.  Validate the
            # bound semantic provenance before the Base reaches any Agent.
            base.verify_run_identity_provenance()
        return base

    def load(
        self,
        *,
        session_overlay_path: str | Path,
        active_protocol_ref: str,
        authority_policy_version: str,
        current_path: str | Path = "CURRENT.json",
        verify_artifacts: bool = True,
    ) -> MemorySnapshot:
        base = self.load_base(
            current_path=current_path,
            verify_artifacts=verify_artifacts,
        )
        resolved_overlay_path = Path(session_overlay_path).resolve()
        if resolved_overlay_path.is_relative_to(base.path):
            raise ValueError("Session Overlay cannot be stored inside immutable Base Bundle")
        overlay = SessionOverlay(resolved_overlay_path)
        snapshot_hash = sha256_json(
            {
                "base_bundle_id": base.bundle_id,
                "base_manifest_sha256": base.manifest_sha256,
                "overlay_manifest_sha256": overlay.manifest["manifest_sha256"],
                "active_protocol_ref": str(active_protocol_ref),
                "authority_policy_version": str(authority_policy_version),
            }
        )
        return MemorySnapshot(
            base_bundle=base,
            session_overlay=overlay,
            active_protocol_ref=str(active_protocol_ref),
            authority_policy_version=str(authority_policy_version),
            snapshot_sha256=snapshot_hash,
        )


def make_current_pointer(
    *,
    bundle_path: str,
    manifest: Mapping[str, Any],
    parent_bundle: str | None,
    published_at: str | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": CURRENT_POINTER_SCHEMA,
        "bundle_path": str(bundle_path),
        "bundle_id": str(manifest["bundle_id"]),
        "bundle_version": str(manifest["bundle_version"]),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "parent_bundle": parent_bundle,
        "published_at": published_at or _utc_now(),
        "pointer_sha256": "",
    }
    payload["pointer_sha256"] = _payload_hash(payload, "pointer_sha256")
    return payload


__all__ = [
    "BUNDLE_MANIFEST_SCHEMA",
    "CURRENT_POINTER_SCHEMA",
    "ImmutableBaseBundle",
    "MemorySnapshot",
    "MemorySnapshotLoader",
    "OVERLAY_EVENT_SCHEMA",
    "OVERLAY_MANIFEST_SCHEMA",
    "OverlayEvent",
    "SessionOverlay",
    "file_lock",
    "make_current_pointer",
    "sha256_file",
    "sha256_json",
    "verify_bundle_directory",
    "write_json_atomic",
]
