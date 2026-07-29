"""Untrusted candidate-side client for requesting Collector observations."""

from __future__ import annotations

import json
import socket
import uuid
from typing import Any, Mapping, Sequence

from .errors import CollectorRejected, CollectorUnavailable


class CollectorClient:
    """A request-only client. It has no receipt or journal-writing API."""

    def __init__(
        self,
        socket_path: str,
        *,
        run_id: str,
        node_id: str,
        code_sha256: str,
        contract_hash: str,
        timeout_seconds: float = 5.0,
    ):
        self.socket_path = str(socket_path)
        self.run_id = str(run_id)
        self.node_id = str(node_id)
        self.code_sha256 = str(code_sha256)
        self.contract_hash = str(contract_hash)
        self.timeout_seconds = float(timeout_seconds)

    def emit(
        self,
        kind: str,
        *,
        capabilities: Sequence[str],
        component: str,
        payload: Mapping[str, Any] | None = None,
        nonce: str | None = None,
    ) -> dict[str, Any]:
        request = {
            "action": "emit",
            "nonce": nonce or uuid.uuid4().hex,
            "kind": str(kind),
            "run_id": self.run_id,
            "node_id": self.node_id,
            "code_sha256": self.code_sha256,
            "contract_hash": self.contract_hash,
            "capabilities": list(capabilities),
            "component": str(component),
            "payload": dict(payload or {}),
        }
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(self.timeout_seconds)
                client.connect(self.socket_path)
                client.sendall(
                    (json.dumps(request, sort_keys=True) + "\n").encode("utf-8")
                )
                buffer = b""
                while not buffer.endswith(b"\n"):
                    chunk = client.recv(65536)
                    if not chunk:
                        break
                    buffer += chunk
        except (OSError, TimeoutError) as error:
            raise CollectorUnavailable(f"Collector is unavailable: {error}") from error
        try:
            response = json.loads(buffer.decode("utf-8"))
        except Exception as error:
            raise CollectorUnavailable("Collector returned an invalid response") from error
        if response.get("status") != "accepted":
            raise CollectorRejected(str(response.get("reason") or "Collector rejected event"))
        return response


__all__ = ["CollectorClient"]
