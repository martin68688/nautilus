from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .evidence_graph import EvidenceGraph, EvidencePath
from .models import Claim, ClaimType, ProtocolRef, Receipt, ReceiptType
from .protocol_registry import canonical_json
from .protocol_registry import ProtocolRegistry
from .replay_certifier import (
    ProtocolRepairSurface,
    ReplayIdentity,
    ReplayVerificationReport,
)


REPLAY_QUEUE_ENTRY_SCHEMA = "clean_replay_queue_entry_v1"
REPLAY_QUEUE_MANIFEST_SCHEMA = "clean_replay_queue_manifest_v1"
REPLAY_REGISTRATION_SCHEMA = "clean_replay_registration_v1"


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


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(_jsonable(value)).encode("utf-8")).hexdigest()


def _is_sha256(value: str) -> bool:
    value = str(value)
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class ReplayCandidate:
    candidate_id: str
    task_id: str
    source_artifact_id: str
    parent_artifact_id: str
    child_artifact_id: str
    original_claim_id: str
    source_clause_id: str
    code_sha256: str
    method_hypothesis: str
    method_family: str
    audit_status: str
    source_refs: tuple[str, ...]
    historical_metric_delta: float | None = None
    method_fatal_issues: tuple[str, ...] = ()
    protocol_issue_codes: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: "ReplayCandidate | Mapping[str, Any]") -> "ReplayCandidate":
        if isinstance(value, ReplayCandidate):
            return value
        payload = dict(value)
        payload["source_refs"] = tuple(str(item) for item in payload.get("source_refs") or ())
        payload["method_fatal_issues"] = tuple(
            str(item) for item in payload.get("method_fatal_issues") or ()
        )
        payload["protocol_issue_codes"] = tuple(
            str(item) for item in payload.get("protocol_issue_codes") or ()
        )
        raw_delta = payload.get("historical_metric_delta")
        payload["historical_metric_delta"] = (
            None if raw_delta is None else float(raw_delta)
        )
        return cls(**payload)

    def rejection_reasons(self) -> list[str]:
        missing = [
            name
            for name in (
                "candidate_id",
                "task_id",
                "source_artifact_id",
                "parent_artifact_id",
                "child_artifact_id",
                "original_claim_id",
                "source_clause_id",
                "method_hypothesis",
                "method_family",
                "audit_status",
            )
            if not str(getattr(self, name)).strip()
        ]
        reasons = [f"missing:{name}" for name in missing]
        if not _is_sha256(self.code_sha256):
            reasons.append("invalid_or_missing_code_sha256")
        required_refs = {
            self.source_artifact_id,
            self.parent_artifact_id,
            self.child_artifact_id,
            self.original_claim_id,
            self.source_clause_id,
        }
        if not required_refs <= set(self.source_refs):
            reasons.append("incomplete_source_parent_child_refs")
        if self.audit_status not in {"clean", "warning", "candidate_replay"}:
            reasons.append("audit_status_not_replay_eligible")
        if self.method_fatal_issues:
            reasons.append("method_fatal_static_audit")
        if self.historical_metric_delta is not None and not math.isfinite(
            self.historical_metric_delta
        ):
            reasons.append("historical_metric_delta_not_finite")
        return reasons


@dataclass
class ReplayQueueEntry:
    candidate_id: str
    task_id: str
    queue_rank: int
    source_artifact_id: str
    parent_artifact_id: str
    child_artifact_id: str
    original_claim_id: str
    source_clause_id: str
    code_sha256: str
    method_hypothesis: str
    method_family: str
    audit_status: str
    source_refs: tuple[str, ...]
    protocol_issue_codes: tuple[str, ...]
    historical_metric_delta: float | None
    selection_basis: tuple[str, ...]
    historical_metric_used_as_evidence: bool = False
    entry_hash: str = ""
    schema: str = REPLAY_QUEUE_ENTRY_SCHEMA

    def finalize(self) -> "ReplayQueueEntry":
        payload = self.as_dict()
        payload.pop("entry_hash", None)
        self.entry_hash = _sha256_json(payload)
        return self

    def verify(self) -> None:
        payload = self.as_dict()
        payload.pop("entry_hash", None)
        if self.schema != REPLAY_QUEUE_ENTRY_SCHEMA:
            raise ValueError("Unsupported replay queue entry schema")
        if self.entry_hash != _sha256_json(payload):
            raise ValueError("Replay queue entry hash mismatch")
        if self.historical_metric_used_as_evidence is not False:
            raise ValueError("Historical metric cannot be replay evidence")

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_value(
        cls, value: "ReplayQueueEntry | Mapping[str, Any]"
    ) -> "ReplayQueueEntry":
        if isinstance(value, cls):
            entry = copy.deepcopy(value)
        else:
            payload = copy.deepcopy(dict(value))
            payload["source_refs"] = tuple(
                str(item) for item in payload.get("source_refs") or ()
            )
            payload["protocol_issue_codes"] = tuple(
                str(item) for item in payload.get("protocol_issue_codes") or ()
            )
            payload["selection_basis"] = tuple(
                str(item) for item in payload.get("selection_basis") or ()
            )
            raw_delta = payload.get("historical_metric_delta")
            payload["historical_metric_delta"] = (
                None if raw_delta is None else float(raw_delta)
            )
            allowed = {field.name for field in dataclasses.fields(cls)}
            unknown = set(payload) - allowed
            if unknown:
                raise ValueError(
                    f"Replay queue entry has unknown fields: {sorted(unknown)}"
                )
            entry = cls(**payload)
        entry.verify()
        return entry


@dataclass
class ReplayQueue:
    entries: list[ReplayQueueEntry]
    rejected: list[dict[str, Any]]
    candidate_count: int
    max_per_task: int
    created_at: str
    entries_sha256: str = ""
    queue_file_sha256: str = ""
    manifest_sha256: str = ""
    schema: str = REPLAY_QUEUE_MANIFEST_SCHEMA

    def _encoded_entries(self) -> bytes:
        return "".join(
            json.dumps(
                entry.as_dict(),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for entry in self.entries
        ).encode("utf-8")

    def finalize(self) -> "ReplayQueue":
        for entry in self.entries:
            entry.finalize()
        self.entries_sha256 = _sha256_json([entry.as_dict() for entry in self.entries])
        self.queue_file_sha256 = hashlib.sha256(self._encoded_entries()).hexdigest()
        payload = self.manifest_dict()
        payload.pop("manifest_sha256", None)
        self.manifest_sha256 = _sha256_json(payload)
        return self

    def manifest_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_count": self.candidate_count,
            "selected_count": len(self.entries),
            "rejected_count": len(self.rejected),
            "max_per_task": self.max_per_task,
            "created_at": self.created_at,
            "entry_hashes": [entry.entry_hash for entry in self.entries],
            "entries_sha256": self.entries_sha256,
            "queue_file_sha256": self.queue_file_sha256,
            "rejected": copy.deepcopy(self.rejected),
            "manifest_sha256": self.manifest_sha256,
        }

    def write(self, queue_path: str | Path, manifest_path: str | Path) -> None:
        self.finalize()
        queue_path = Path(queue_path)
        manifest_path = Path(manifest_path)
        encoded = self._encoded_entries()
        _write_atomic(queue_path, encoded)
        manifest = self.manifest_dict()
        _write_atomic(
            manifest_path,
            (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            ),
        )


def load_replay_queue(
    queue_path: str | Path,
    manifest_path: str | Path,
) -> ReplayQueue:
    """Load and verify both byte-level and semantic queue bindings."""

    queue_path = Path(queue_path)
    manifest_path = Path(manifest_path)
    encoded = queue_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError("Replay queue manifest must be an object")
    if manifest.get("schema") != REPLAY_QUEUE_MANIFEST_SCHEMA:
        raise ValueError("Unsupported replay queue manifest schema")
    rows = [
        json.loads(line)
        for line in encoded.decode("utf-8").splitlines()
        if line.strip()
    ]
    entries = [ReplayQueueEntry.from_value(row) for row in rows]
    if len(entries) != int(manifest.get("selected_count", -1)):
        raise ValueError("Replay queue selected-count mismatch")
    if [entry.entry_hash for entry in entries] != list(
        manifest.get("entry_hashes") or []
    ):
        raise ValueError("Replay queue entry-hash order mismatch")
    entries_sha256 = _sha256_json([entry.as_dict() for entry in entries])
    queue_file_sha256 = hashlib.sha256(encoded).hexdigest()
    if entries_sha256 != str(manifest.get("entries_sha256") or ""):
        raise ValueError("Replay queue semantic hash mismatch")
    if queue_file_sha256 != str(manifest.get("queue_file_sha256") or ""):
        raise ValueError("Replay queue file hash mismatch")
    unsigned = dict(manifest)
    declared_manifest_sha256 = str(unsigned.pop("manifest_sha256", ""))
    if declared_manifest_sha256 != _sha256_json(unsigned):
        raise ValueError("Replay queue manifest hash mismatch")
    rejected = manifest.get("rejected") or []
    if not isinstance(rejected, list):
        raise ValueError("Replay queue rejected population must be a list")
    if len(rejected) != int(manifest.get("rejected_count", -1)):
        raise ValueError("Replay queue rejected-count mismatch")
    queue = ReplayQueue(
        entries=entries,
        rejected=copy.deepcopy(rejected),
        candidate_count=int(manifest.get("candidate_count", -1)),
        max_per_task=int(manifest.get("max_per_task", -1)),
        created_at=str(manifest.get("created_at") or ""),
        entries_sha256=entries_sha256,
        queue_file_sha256=queue_file_sha256,
        manifest_sha256=declared_manifest_sha256,
        schema=str(manifest["schema"]),
    )
    if queue.candidate_count != len(entries) + len(rejected):
        raise ValueError("Replay queue candidate-count mismatch")
    return queue


def build_replay_queue(
    candidates: Iterable[ReplayCandidate | Mapping[str, Any]],
    *,
    max_per_task: int = 3,
    created_at: str | None = None,
) -> ReplayQueue:
    if not 1 <= int(max_per_task) <= 3:
        raise ValueError("Clean Replay permits one to three candidates per task")
    values = [ReplayCandidate.from_value(value) for value in candidates]
    candidate_ids = [value.candidate_id for value in values]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Replay candidate IDs must be unique")
    rejected: list[dict[str, Any]] = []
    eligible: dict[str, list[ReplayCandidate]] = {}
    for candidate in values:
        reasons = candidate.rejection_reasons()
        if reasons:
            rejected.append(
                {"candidate_id": candidate.candidate_id, "reasons": sorted(reasons)}
            )
            continue
        eligible.setdefault(candidate.task_id, []).append(candidate)

    selected: list[ReplayQueueEntry] = []
    for task_id, task_candidates in sorted(eligible.items()):
        ordered = sorted(
            task_candidates,
            key=lambda item: (
                item.historical_metric_delta is None,
                -(item.historical_metric_delta or 0.0),
                item.candidate_id,
            ),
        )
        chosen: list[ReplayCandidate] = []
        seen_families: set[str] = set()
        for candidate in ordered:
            if candidate.method_family in seen_families:
                continue
            chosen.append(candidate)
            seen_families.add(candidate.method_family)
            if len(chosen) == max_per_task:
                break
        if len(chosen) < max_per_task:
            for candidate in ordered:
                if candidate in chosen:
                    continue
                chosen.append(candidate)
                if len(chosen) == max_per_task:
                    break
        for rank, candidate in enumerate(chosen, 1):
            selected.append(
                ReplayQueueEntry(
                    candidate_id=candidate.candidate_id,
                    task_id=task_id,
                    queue_rank=rank,
                    source_artifact_id=candidate.source_artifact_id,
                    parent_artifact_id=candidate.parent_artifact_id,
                    child_artifact_id=candidate.child_artifact_id,
                    original_claim_id=candidate.original_claim_id,
                    source_clause_id=candidate.source_clause_id,
                    code_sha256=candidate.code_sha256,
                    method_hypothesis=candidate.method_hypothesis,
                    method_family=candidate.method_family,
                    audit_status=candidate.audit_status,
                    source_refs=candidate.source_refs,
                    protocol_issue_codes=candidate.protocol_issue_codes,
                    historical_metric_delta=candidate.historical_metric_delta,
                    selection_basis=(
                        "deterministic_complete_artifact_filter",
                        "method_fatal_static_audit_filter",
                        "method_family_diversity_first",
                        "historical_metric_priority_only",
                        "candidate_id_tiebreak",
                    ),
                ).finalize()
            )
        chosen_ids = {candidate.candidate_id for candidate in chosen}
        for candidate in ordered:
            if candidate.candidate_id not in chosen_ids:
                rejected.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "reasons": ["per_task_limit_or_method_family_diversity"],
                    }
                )
    return ReplayQueue(
        entries=selected,
        rejected=sorted(rejected, key=lambda item: item["candidate_id"]),
        candidate_count=len(values),
        max_per_task=int(max_per_task),
        created_at=created_at or _utc_now(),
    ).finalize()


def verify_trusted_receipt_integrity(receipt: Receipt) -> None:
    expected_payload_hash = hashlib.sha256(
        canonical_json(receipt.payload).encode("utf-8")
    ).hexdigest()
    if receipt.payload_hash != expected_payload_hash:
        raise ValueError(f"Replay Receipt payload hash mismatch: {receipt.receipt_id}")
    stable_key = canonical_json(
        {
            "receipt_type": receipt.receipt_type.value,
            "artifact_id": receipt.artifact_id,
            "run_id": receipt.run_id,
            "protocol_hash": receipt.protocol_hash,
            "collector_id": receipt.collector_id,
            "collector_version": receipt.collector_version,
            "payload_hash": receipt.payload_hash,
            "trust_status": receipt.trust_status,
            "supports_claim_types": sorted(set(receipt.supports_claim_types)),
            "blocks_claim_types": sorted(set(receipt.blocks_claim_types)),
        }
    )
    expected_id = uuid.uuid5(uuid.NAMESPACE_URL, stable_key).hex
    if receipt.receipt_id != expected_id:
        raise ValueError("Replay Receipt stable ID mismatch")
    if receipt.trust_status != "trusted_host":
        raise ValueError("Replay recovery requires trusted-host Receipts")
    if not receipt.observation_id or not receipt.event_hash:
        raise ValueError("Replay recovery Receipt lacks a host event binding")
    expected_event_hash = hashlib.sha256(
        canonical_json(
            {
                "receipt_id": receipt.receipt_id,
                "observation_id": receipt.observation_id,
                "parent_event_hash": receipt.parent_event_hash,
                "payload_hash": receipt.payload_hash,
                "trust_status": receipt.trust_status,
            }
        ).encode("utf-8")
    ).hexdigest()
    if receipt.event_hash != expected_event_hash:
        raise ValueError("Replay Receipt event hash mismatch")


class ReplayReceiptIngestor:
    """Validate replay Receipts before any EvidenceGraph mutation."""

    @staticmethod
    def validate(
        receipts: Iterable[Receipt],
        *,
        replay_artifact_id: str,
        protocol_ref: ProtocolRef,
        verification: ReplayVerificationReport,
    ) -> tuple[Receipt, ...]:
        output = tuple(copy.deepcopy(list(receipts)))
        if not output:
            raise ValueError("Replay recovery requires Receipts")
        seen: set[str] = set()
        for receipt in output:
            verify_trusted_receipt_integrity(receipt)
            if receipt.receipt_id in seen:
                raise ValueError("Duplicate replay Receipt ID")
            seen.add(receipt.receipt_id)
            if receipt.artifact_id != str(replay_artifact_id):
                raise ValueError("Historical/source Receipt cannot support replay artifact")
            if receipt.protocol_hash != protocol_ref.canonical_hash:
                raise ValueError("Replay Receipt protocol mismatch")
        method_receipts = [
            receipt
            for receipt in output
            if receipt.receipt_type == ReceiptType.METHOD_IDENTITY
        ]
        if not any(
            receipt.payload.get("method_fingerprint")
            == verification.replay_method_fingerprint
            and receipt.payload.get("code_sha256") == verification.replay_code_sha256
            for receipt in method_receipts
        ):
            raise ValueError("Replay MethodIdentity Receipt does not bind verifier output")
        return output


@dataclass
class ReplayRegistration:
    original_claim_id: str
    replay_claim_id: str
    replay_artifact_id: str
    identity: ReplayIdentity
    path_id: str
    receipt_ids: tuple[str, ...]
    verification_report_hash: str
    old_claim_mutated: bool
    authority_recovered_for_old_claim: bool
    registration_hash: str = ""
    schema: str = REPLAY_REGISTRATION_SCHEMA

    def finalize(self) -> "ReplayRegistration":
        payload = self.as_dict()
        payload.pop("registration_hash", None)
        self.registration_hash = _sha256_json(payload)
        return self

    def verify(self) -> None:
        if self.schema != REPLAY_REGISTRATION_SCHEMA:
            raise ValueError("Unsupported replay registration schema")
        payload = self.as_dict()
        payload.pop("registration_hash", None)
        if self.registration_hash != _sha256_json(payload):
            raise ValueError("Replay registration hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


class ReplayAuthorityRecovery:
    """Register only a new replay/successor Claim and an independent clean path."""

    def __init__(self, graph: EvidenceGraph, registry: ProtocolRegistry):
        self.graph = graph
        self.registry = registry

    def register(
        self,
        *,
        original_claim_id: str,
        verification: ReplayVerificationReport,
        receipts: Iterable[Receipt],
        protocol_ref: ProtocolRef,
        statement: str,
        claim_type: ClaimType | None = None,
        task_scope: Mapping[str, Any] | None = None,
    ) -> ReplayRegistration:
        verification.verify()
        protocol_spec = self.registry.resolve(protocol_ref)
        expected_surface = ProtocolRepairSurface.from_protocol_spec(protocol_spec)
        if verification.repair_protocol_ref != protocol_ref.key():
            raise ValueError("Replay verifier is not bound to the recovery ProtocolRef")
        if verification.repair_surface_hash != expected_surface.surface_hash:
            raise ValueError("Replay verifier did not use the ProtocolSpec repair surface")
        original = self.graph.claims.get(str(original_claim_id))
        if original is None:
            raise KeyError(f"Unknown original Claim: {original_claim_id}")
        original_snapshot = copy.deepcopy(original)
        if verification.source_artifact_id != original.subject_artifact_id:
            raise ValueError("Replay verifier source does not bind original Claim artifact")
        if (
            original.method_fingerprint not in {"", "legacy-unavailable"}
            and original.method_fingerprint
            not in {
                verification.source_method_fingerprint,
                verification.source_protected_surface_hash,
                verification.source_code_sha256,
            }
        ):
            raise ValueError("Replay verifier source method does not bind original Claim")
        if verification.identity == ReplayIdentity.REQUIRE_HUMAN_REVIEW:
            return ReplayRegistration(
                original_claim_id=original.claim_id,
                replay_claim_id="",
                replay_artifact_id=verification.replay_artifact_id,
                identity=verification.identity,
                path_id="",
                receipt_ids=(),
                verification_report_hash=verification.report_hash,
                old_claim_mutated=False,
                authority_recovered_for_old_claim=False,
            ).finalize()
        if not str(statement).strip():
            raise ValueError("Replay Claim requires a new replay-result statement")
        validated = ReplayReceiptIngestor.validate(
            receipts,
            replay_artifact_id=verification.replay_artifact_id,
            protocol_ref=protocol_ref,
            verification=verification,
        )
        resolved_type = claim_type or original.claim_type
        identity_prefix = (
            "replay" if verification.identity == ReplayIdentity.METHOD_PRESERVED else "successor"
        )
        claim_payload = {
            "original_claim_id": original.claim_id,
            "replay_artifact_id": verification.replay_artifact_id,
            "identity": verification.identity.value,
            "verification_report_hash": verification.report_hash,
            "protocol_ref": protocol_ref.key(),
            "claim_type": resolved_type.value,
            "statement": str(statement).strip(),
        }
        replay_claim_id = f"claim::{identity_prefix}::{_sha256_json(claim_payload)[:24]}"
        replay_claim = Claim(
            claim_id=replay_claim_id,
            claim_type=resolved_type,
            subject_artifact_id=verification.replay_artifact_id,
            task_scope=dict(task_scope or original.task_scope),
            method_fingerprint=verification.replay_method_fingerprint,
            protocol_ref=protocol_ref,
            statement=str(statement).strip(),
            # Provenance is recorded below, but the old restricted Claim is not
            # an Authority parent of this independently observed replay result.
            parent_claims=[],
            source_artifact_refs=sorted(
                set(original.source_artifact_refs)
                | {original.subject_artifact_id, verification.replay_artifact_id}
            ),
            evidence_refs=[verification.report_hash],
            boundary={
                "clean_replay": True,
                "replay_identity": verification.identity.value,
                "predecessor_claim_id": original.claim_id,
                "predecessor_protocol_ref": original.protocol_ref.key(),
                "verification_report_hash": verification.report_hash,
                "historical_metric_used_as_evidence": False,
                "old_claim_authority_recovered": False,
            },
            legacy_status="clean_replay_v1",
        )
        path_id = f"path::{identity_prefix}::{_sha256_json({'claim': replay_claim_id, 'receipts': sorted(r.receipt_id for r in validated)})[:24]}"

        # All validation is complete before the graph transaction starts.
        self.graph.add_claim(replay_claim)
        for receipt in validated:
            self.graph.add_receipt(receipt)
        self.graph.add_path(
            EvidencePath(
                path_id=path_id,
                claim_id=replay_claim_id,
                receipt_ids=sorted(receipt.receipt_id for receipt in validated),
                required_parent_claims=[],
            )
        )
        if self.graph.claims[original.claim_id] != original_snapshot:
            raise RuntimeError("Clean Replay mutated the historical Claim")
        return ReplayRegistration(
            original_claim_id=original.claim_id,
            replay_claim_id=replay_claim_id,
            replay_artifact_id=verification.replay_artifact_id,
            identity=verification.identity,
            path_id=path_id,
            receipt_ids=tuple(sorted(receipt.receipt_id for receipt in validated)),
            verification_report_hash=verification.report_hash,
            old_claim_mutated=False,
            authority_recovered_for_old_claim=False,
        ).finalize()


__all__ = [
    "REPLAY_QUEUE_ENTRY_SCHEMA",
    "REPLAY_QUEUE_MANIFEST_SCHEMA",
    "REPLAY_REGISTRATION_SCHEMA",
    "ReplayAuthorityRecovery",
    "ReplayCandidate",
    "ReplayQueue",
    "ReplayQueueEntry",
    "ReplayReceiptIngestor",
    "ReplayRegistration",
    "build_replay_queue",
    "load_replay_queue",
    "verify_trusted_receipt_integrity",
]
