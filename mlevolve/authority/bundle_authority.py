from __future__ import annotations

import copy
from typing import Any, Mapping

from .authority_engine import AuthorityEngine
from .clean_replay import verify_trusted_receipt_integrity
from .evidence_graph import EvidencePath
from .memory_snapshot import MemorySnapshot, sha256_json
from .models import (
    AuthorityDecision,
    AuthorityScope,
    Claim,
    ClaimType,
    DecisionOutcome,
    ProtocolRef,
    Receipt,
    ReceiptType,
)


def _protocol_ref(value: str | Mapping[str, Any], engine: AuthorityEngine) -> ProtocolRef:
    if isinstance(value, Mapping):
        ref = ProtocolRef(
            str(value["protocol_id"]),
            str(value["version"]),
            str(value["canonical_hash"]),
        )
    else:
        prefix, separator, digest = str(value).partition("#")
        protocol_id, version_separator, version = prefix.rpartition("@")
        if not separator or not version_separator or not protocol_id or not version or not digest:
            raise ValueError(f"Invalid bundle ProtocolRef: {value}")
        ref = ProtocolRef(protocol_id, version, digest)
    # Resolution verifies both version registration and immutable canonical hash.
    engine.registry.resolve(ref)
    return ref


def _claim(row: Mapping[str, Any], engine: AuthorityEngine) -> Claim:
    return Claim(
        claim_id=str(row["claim_id"]),
        claim_type=ClaimType(str(row["claim_type"])),
        subject_artifact_id=str(row["subject_artifact_id"]),
        task_scope=copy.deepcopy(dict(row.get("task_scope") or {})),
        method_fingerprint=str(row.get("method_fingerprint") or ""),
        protocol_ref=_protocol_ref(row["protocol_ref"], engine),
        statement=str(row.get("statement") or ""),
        parent_claims=[str(value) for value in row.get("parent_claims") or []],
        source_artifact_refs=[
            str(value) for value in row.get("source_artifact_refs") or []
        ],
        evidence_refs=[str(value) for value in row.get("evidence_refs") or []],
        boundary=copy.deepcopy(dict(row.get("boundary") or {})),
        legacy_status=str(row.get("legacy_status") or "native_v1"),
    )


def _receipt(row: Mapping[str, Any]) -> Receipt:
    values = copy.deepcopy(dict(row))
    values["receipt_type"] = ReceiptType(str(values["receipt_type"]))
    receipt = Receipt(**values)
    if receipt.trust_status == "trusted_host":
        verify_trusted_receipt_integrity(receipt)
    return receipt


def _path(row: Mapping[str, Any]) -> EvidencePath:
    return EvidencePath(
        path_id=str(row["path_id"]),
        claim_id=str(row["claim_id"]),
        receipt_ids=[str(value) for value in row.get("receipt_ids") or []],
        required_parent_claims=[
            str(value) for value in row.get("required_parent_claims") or []
        ],
    )


def _decision(row: Mapping[str, Any]) -> AuthorityDecision:
    values = copy.deepcopy(dict(row))
    values["outcome"] = DecisionOutcome(str(values["outcome"]))
    scope = values.get("permitted_scope")
    values["permitted_scope"] = AuthorityScope(**scope) if isinstance(scope, dict) else None
    return AuthorityDecision(**values)


def load_snapshot_authority(
    engine: AuthorityEngine,
    snapshot: MemorySnapshot,
) -> dict[str, Any]:
    """Load hash-verified Base Authority records into a live engine."""

    snapshot.assert_unchanged()
    declared = set((snapshot.base_bundle.manifest.get("artifact_hashes") or {}))

    def rows(relative: str) -> list[dict[str, Any]]:
        return snapshot.base_bundle.read_jsonl(relative) if relative in declared else []

    claim_rows = rows("authority/claims.jsonl")
    receipt_rows = rows("authority/receipts.jsonl")
    path_rows = [
        *rows("authority/paths.jsonl"),
        *rows("authority/replay_paths.jsonl"),
    ]
    decision_rows = rows("authority/decisions.jsonl")
    for row in claim_rows:
        engine.graph.add_claim(_claim(row, engine))
    for row in receipt_rows:
        engine.graph.add_receipt(_receipt(row))
    for row in path_rows:
        path = _path(row)
        missing_receipts = set(path.receipt_ids) - set(engine.graph.receipts)
        if missing_receipts:
            raise ValueError(
                f"Bundle Authority path references missing Receipts: {sorted(missing_receipts)}"
            )
        current = engine.graph.paths.get(path.path_id)
        if current is not None and current != path:
            raise ValueError(f"Bundle Authority path is immutable: {path.path_id}")
        engine.graph.add_path(path)
    for row in decision_rows:
        decision = _decision(row)
        current = engine.decisions.get(decision.decision_id)
        if current is not None and current != decision:
            raise ValueError(f"Bundle Authority decision is immutable: {decision.decision_id}")
        engine.decisions[decision.decision_id] = decision
    snapshot.assert_unchanged()
    payload = {
        "schema": "memory_bundle_authority_load_report_v1",
        "bundle_id": snapshot.base_bundle.bundle_id,
        "bundle_manifest_sha256": snapshot.base_bundle.manifest_sha256,
        "claim_ids": sorted(str(row["claim_id"]) for row in claim_rows),
        "receipt_ids": sorted(str(row["receipt_id"]) for row in receipt_rows),
        "path_ids": sorted(str(row["path_id"]) for row in path_rows),
        "decision_ids": sorted(str(row["decision_id"]) for row in decision_rows),
    }
    payload["report_hash"] = sha256_json(payload)
    return payload


def restore_engine_snapshot(
    engine: AuthorityEngine,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore a hash-verified runtime Authority snapshot into an engine.

    This is used by host-owned post-run finalizers (for example the isolated
    fixed-holdout scorer).  It restores immutable graph/decision objects but
    never trusts or executes Agent text.
    """

    if str(snapshot.get("policy_version") or "") != engine.policy_version:
        raise ValueError("Authority snapshot policy version mismatch")
    claim_rows = snapshot.get("claims") or {}
    receipt_rows = snapshot.get("receipts") or {}
    path_rows = snapshot.get("paths") or {}
    decision_rows = snapshot.get("decisions") or {}
    if not all(
        isinstance(rows, Mapping)
        for rows in (claim_rows, receipt_rows, path_rows, decision_rows)
    ):
        raise ValueError("Authority snapshot graph sections must be mappings")
    for key, row in sorted(claim_rows.items()):
        claim = _claim(row, engine)
        if claim.claim_id != str(key):
            raise ValueError("Authority snapshot Claim key mismatch")
        engine.graph.add_claim(claim)
    for key, row in sorted(receipt_rows.items()):
        receipt = _receipt(row)
        if receipt.receipt_id != str(key):
            raise ValueError("Authority snapshot Receipt key mismatch")
        engine.graph.add_receipt(receipt)
    for key, row in sorted(path_rows.items()):
        path = _path(row)
        if path.path_id != str(key):
            raise ValueError("Authority snapshot path key mismatch")
        if set(path.receipt_ids) - set(engine.graph.receipts):
            raise ValueError("Authority snapshot path has missing Receipts")
        engine.graph.add_path(path)
    for key, row in sorted(decision_rows.items()):
        decision = _decision(row)
        if decision.decision_id != str(key):
            raise ValueError("Authority snapshot decision key mismatch")
        current = engine.decisions.get(decision.decision_id)
        if current is not None and current != decision:
            raise ValueError("Authority snapshot decision is immutable")
        engine.decisions[decision.decision_id] = decision
    return {
        "schema": "authority_engine_snapshot_restore_report_v1",
        "claim_count": len(claim_rows),
        "receipt_count": len(receipt_rows),
        "path_count": len(path_rows),
        "decision_count": len(decision_rows),
    }


__all__ = ["load_snapshot_authority", "restore_engine_snapshot"]
