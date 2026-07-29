"""Canonical event and capability helpers shared by SDK and Collector."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any, Mapping


RUNTIME_EVENT_JOURNAL_SCHEMA = "mlevolve_runtime_event_journal_v1"
RUNTIME_EVENT_JOURNAL_MANIFEST_SCHEMA = (
    "mlevolve_runtime_event_journal_manifest_v1"
)
RUNTIME_EVIDENCE_REPORT_SCHEMA = "mlevolve_runtime_evidence_report_v1"
EVENT_ORDER = (
    "split_lineage",
    "fit_scope",
    "prediction_scope",
    "evaluator",
    "selection_freeze",
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_payload(value: Mapping[str, Any], hash_field: str) -> str:
    return sha256_value(
        {key: item for key, item in value.items() if key != hash_field}
    )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def mint_capability(secret: bytes, body: Mapping[str, Any]) -> str:
    encoded = _b64(canonical_json(dict(body)).encode("utf-8"))
    signature = _b64(hmac.new(secret, encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def verify_capability(secret: bytes, token: str) -> dict[str, Any]:
    try:
        encoded, signature = token.split(".", 1)
        encoded_raw = _unb64(encoded)
        signature_raw = _unb64(signature)
        # Reject non-canonical base64url spellings.  Without this check, changing
        # unused trailing bits can produce a different token string that decodes
        # to the same HMAC bytes and intermittently passes tamper tests.
        if _b64(encoded_raw) != encoded or _b64(signature_raw) != signature:
            raise ValueError("capability encoding is not canonical")
        expected = hmac.new(secret, encoded.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, signature_raw):
            raise ValueError("capability signature mismatch")
        value = json.loads(encoded_raw)
    except Exception as error:
        raise ValueError("invalid Collector capability") from error
    if not isinstance(value, dict):
        raise ValueError("Collector capability body must be an object")
    return value


__all__ = [
    "EVENT_ORDER",
    "RUNTIME_EVENT_JOURNAL_MANIFEST_SCHEMA",
    "RUNTIME_EVENT_JOURNAL_SCHEMA",
    "RUNTIME_EVIDENCE_REPORT_SCHEMA",
    "canonical_json",
    "hash_payload",
    "mint_capability",
    "sha256_value",
    "verify_capability",
]
