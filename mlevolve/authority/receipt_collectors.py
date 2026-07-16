from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any

from .models import ProtocolRef, Receipt, ReceiptType
from .protocol_registry import canonical_json


def make_receipt(
    receipt_type: ReceiptType,
    artifact_id: str,
    run_id: str,
    protocol_ref: ProtocolRef,
    collector_id: str,
    payload: dict[str, Any],
    collector_version: str = "1",
) -> Receipt:
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
        }
    )
    receipt_id = uuid.uuid5(uuid.NAMESPACE_URL, stable_key).hex
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
    )


class ReceiptCollector:
    collector_id = "generic"
    collector_version = "1"

    def collect(self, artifact: Any, protocol_ref: ProtocolRef) -> list[Receipt]:
        raise NotImplementedError
