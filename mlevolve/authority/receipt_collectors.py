from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from .models import ProtocolRef, Receipt, ReceiptType
from .protocol_registry import canonical_json


def _make_receipt(
    receipt_type: ReceiptType,
    artifact_id: str,
    run_id: str,
    protocol_ref: ProtocolRef,
    collector_id: str,
    payload: dict[str, Any],
    collector_version: str = "1",
    *,
    trust_status: str,
    observation_id: str = "",
    parent_event_hash: str = "",
    supports_claim_types: list[str] | None = None,
    blocks_claim_types: list[str] | None = None,
) -> Receipt:
    supports_claim_types = sorted(set(supports_claim_types or []))
    blocks_claim_types = sorted(set(blocks_claim_types or []))
    payload_hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    stable_key = canonical_json(
        {
            "receipt_type": receipt_type.value,
            "artifact_id": artifact_id,
            "run_id": run_id,
            "protocol_hash": protocol_ref.canonical_hash,
            "collector_id": collector_id,
            "collector_version": collector_version,
            "payload_hash": payload_hash,
            "trust_status": trust_status,
            "supports_claim_types": supports_claim_types,
            "blocks_claim_types": blocks_claim_types,
        }
    )
    receipt_id = uuid.uuid5(uuid.NAMESPACE_URL, stable_key).hex
    event_hash = ""
    if observation_id:
        event_hash = hashlib.sha256(
            canonical_json(
                {
                    "receipt_id": receipt_id,
                    "observation_id": observation_id,
                    "parent_event_hash": parent_event_hash,
                    "payload_hash": payload_hash,
                    "trust_status": trust_status,
                }
            ).encode("utf-8")
        ).hexdigest()
    return Receipt(
        receipt_id=receipt_id,
        receipt_type=receipt_type,
        artifact_id=artifact_id,
        run_id=run_id,
        protocol_hash=protocol_ref.canonical_hash,
        collector_id=collector_id,
        collector_version=collector_version,
        payload_hash=payload_hash,
        payload=payload,
        timestamp=datetime.now(timezone.utc).isoformat(),
        parent_event_hash=parent_event_hash,
        event_hash=event_hash,
        trust_status=trust_status,
        observation_id=observation_id,
        supports_claim_types=supports_claim_types,
        blocks_claim_types=blocks_claim_types,
    )


def make_receipt(
    receipt_type: ReceiptType,
    artifact_id: str,
    run_id: str,
    protocol_ref: ProtocolRef,
    collector_id: str,
    payload: dict[str, Any],
    collector_version: str = "1",
    *,
    supports_claim_types: list[str] | None = None,
    blocks_claim_types: list[str] | None = None,
) -> Receipt:
    """Create a legacy/static receipt that cannot assert host trust."""

    return _make_receipt(
        receipt_type,
        artifact_id,
        run_id,
        protocol_ref,
        collector_id,
        payload,
        collector_version,
        trust_status="legacy_static_only",
        supports_claim_types=supports_claim_types,
        blocks_claim_types=blocks_claim_types,
    )


class ReceiptCollector:
    collector_id = "generic"
    collector_version = "1"

    def collect(self, artifact: Any, protocol_ref: ProtocolRef) -> list[Receipt]:
        raise NotImplementedError
