from __future__ import annotations

import copy
import hashlib
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

from ..models import ClaimType, ProtocolRef, Receipt, ReceiptType
from ..protocol_registry import canonical_json
from ..receipt_collectors import _make_receipt


class UntrustedObservationError(ValueError):
    pass


@dataclass(frozen=True)
class HostObservation:
    observation_id: str
    receipt_type: ReceiptType
    artifact_id: str
    run_id: str
    protocol_ref: ProtocolRef
    source: str
    payload: dict[str, Any]
    payload_hash: str
    observed_at: str
    _capability: object = field(repr=False, compare=False)


CollectorT = TypeVar("CollectorT", bound="TrustedReceiptCollector")


class TrustedCollectorHost:
    """Host-owned capability boundary for receipt creation.

    Agent output can be observed as data, but only a HostObservation minted by
    this object can be consumed by a trusted collector. Payload mutation after
    observation invalidates the observation before receipt creation.
    """

    def __init__(self, host_id: str, *, collector_version: str = "1"):
        self.host_id = str(host_id)
        self.collector_version = str(collector_version)
        self.__capability = object()
        self._lock = threading.RLock()
        self._last_event_hash = ""
        self._minted_observations: dict[str, str] = {}

    @staticmethod
    def _observation_fingerprint(observation: HostObservation) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "observation_id": observation.observation_id,
                    "receipt_type": observation.receipt_type.value,
                    "artifact_id": observation.artifact_id,
                    "run_id": observation.run_id,
                    "protocol_id": observation.protocol_ref.protocol_id,
                    "protocol_version": observation.protocol_ref.version,
                    "protocol_hash": observation.protocol_ref.canonical_hash,
                    "source": observation.source,
                    "payload_hash": observation.payload_hash,
                    "observed_at": observation.observed_at,
                }
            ).encode("utf-8")
        ).hexdigest()

    def observe(
        self,
        receipt_type: ReceiptType,
        *,
        artifact_id: str,
        run_id: str,
        protocol_ref: ProtocolRef,
        source: str,
        payload: dict[str, Any],
    ) -> HostObservation:
        frozen_payload = copy.deepcopy(payload)
        payload_hash = hashlib.sha256(
            canonical_json(frozen_payload).encode("utf-8")
        ).hexdigest()
        observation = HostObservation(
            observation_id=uuid.uuid4().hex,
            receipt_type=receipt_type,
            artifact_id=str(artifact_id),
            run_id=str(run_id),
            protocol_ref=protocol_ref,
            source=str(source),
            payload=frozen_payload,
            payload_hash=payload_hash,
            observed_at=datetime.now(timezone.utc).isoformat(),
            _capability=self.__capability,
        )
        with self._lock:
            self._minted_observations[observation.observation_id] = (
                self._observation_fingerprint(observation)
            )
        return observation

    def collect(
        self,
        collector_type: type[CollectorT],
        *,
        artifact_id: str,
        run_id: str,
        protocol_ref: ProtocolRef,
        source: str,
        payload: dict[str, Any],
    ) -> Receipt:
        observation = self.observe(
            collector_type.receipt_type,
            artifact_id=artifact_id,
            run_id=run_id,
            protocol_ref=protocol_ref,
            source=source,
            payload=payload,
        )
        return collector_type(self).collect(observation)

    def _validate_observation(
        self,
        observation: HostObservation,
        expected_type: ReceiptType,
    ) -> None:
        if not isinstance(observation, HostObservation):
            raise UntrustedObservationError("collector requires a host observation")
        if observation._capability is not self.__capability:
            raise UntrustedObservationError("observation was not minted by this collector host")
        expected_fingerprint = self._minted_observations.get(observation.observation_id)
        if expected_fingerprint is None:
            raise UntrustedObservationError("observation was not minted by this collector host")
        if observation.receipt_type != expected_type:
            raise UntrustedObservationError("observation receipt type does not match collector")
        observed_hash = hashlib.sha256(
            canonical_json(observation.payload).encode("utf-8")
        ).hexdigest()
        if observed_hash != observation.payload_hash:
            raise UntrustedObservationError("host observation payload changed after capture")
        if self._observation_fingerprint(observation) != expected_fingerprint:
            raise UntrustedObservationError("host observation metadata changed after capture")
        if observation.protocol_ref.canonical_hash == "":
            raise UntrustedObservationError("trusted observation requires a canonical protocol")

    def _mint(
        self,
        collector: "TrustedReceiptCollector",
        observation: HostObservation,
    ) -> Receipt:
        with self._lock:
            self._validate_observation(observation, collector.receipt_type)
            payload = collector.validated_payload(copy.deepcopy(observation.payload))
            receipt = _make_receipt(
                collector.receipt_type,
                observation.artifact_id,
                observation.run_id,
                observation.protocol_ref,
                collector.collector_id,
                payload,
                self.collector_version,
                trust_status="trusted_host",
                observation_id=observation.observation_id,
                parent_event_hash=self._last_event_hash,
                supports_claim_types=[item.value for item in collector.supports_claim_types],
                blocks_claim_types=[item.value for item in collector.blocks_claim_types],
            )
            self._last_event_hash = receipt.event_hash
            del self._minted_observations[observation.observation_id]
            return receipt


class TrustedReceiptCollector:
    receipt_type: ReceiptType
    collector_id = "host.generic"
    supports_claim_types: tuple[ClaimType, ...] = tuple(ClaimType)
    blocks_claim_types: tuple[ClaimType, ...] = ()

    def __init__(self, host: TrustedCollectorHost):
        self._host = host

    def collect(self, observation: HostObservation) -> Receipt:
        return self._host._mint(self, observation)

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


def require_nonempty(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value in (None, "", [], {}):
        raise UntrustedObservationError(f"missing trusted observation field: {key}")
    return value


def require_sha256(payload: dict[str, Any], key: str) -> str:
    value = str(require_nonempty(payload, key))
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise UntrustedObservationError(f"{key} must be a SHA256 digest")
    return value.lower()
