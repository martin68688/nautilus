"""Process-isolated, signed Host runtime event Collector."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import secrets
import socket
import stat
import time
from typing import TYPE_CHECKING, Any, Mapping
import uuid

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .errors import CollectorRejected, CollectorUnavailable
from .events import (
    EVENT_ORDER,
    RUNTIME_EVENT_JOURNAL_MANIFEST_SCHEMA,
    RUNTIME_EVENT_JOURNAL_SCHEMA,
    RUNTIME_EVIDENCE_REPORT_SCHEMA,
    canonical_json,
    hash_payload,
    mint_capability,
    sha256_value,
    verify_capability,
)

if TYPE_CHECKING:
    from .collector_client import CollectorClient


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"), validate=True)


@dataclass(frozen=True)
class HostCollectorIdentity:
    """Host-launcher signing identity; never place the private bytes in Candidate state."""

    public_key_ed25519: str
    _private_key_raw: bytes = field(repr=False, compare=False)

    @classmethod
    def generate(cls) -> "HostCollectorIdentity":
        private = Ed25519PrivateKey.generate()
        private_raw = private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_raw = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return cls(public_key_ed25519=_b64(public_raw), _private_key_raw=private_raw)

    @classmethod
    def from_private_key_file(cls, path: str | Path) -> "HostCollectorIdentity":
        """Load a Host-only raw Ed25519 key and derive its bound public key."""

        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise ValueError("Refusing symlink Host Collector private key")
        key_path = requested.resolve(strict=True)
        if not key_path.is_file():
            raise ValueError("Host Collector private key must be a regular file")
        if key_path.stat().st_mode & 0o077:
            raise ValueError("Host Collector private key must not be group/world accessible")
        private_raw = key_path.read_bytes()
        if len(private_raw) != 32:
            raise ValueError("Host Collector private key must be 32 raw Ed25519 bytes")
        private = Ed25519PrivateKey.from_private_bytes(private_raw)
        public_raw = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return cls(public_key_ed25519=_b64(public_raw), _private_key_raw=private_raw)

    def write_private_key_file(self, path: str | Path) -> Path:
        """Provision the private half outside Candidate-visible bundle roots."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(self._private_key_raw)
            handle.flush()
            os.fsync(handle.fileno())
        return destination

    def collector_spec(self) -> dict[str, Any]:
        return {
            "schema": "mlevolve_host_collector_v1",
            "transport": "restricted_unix_socket",
            "append_only_hash_chain": True,
            "candidate_has_signing_key": False,
            "signature_algorithm": "ed25519",
            "public_key_ed25519": self.public_key_ed25519,
        }

    def sign_canonical_payload(self, payload: Mapping[str, Any]) -> str:
        """Sign a Host-created canonical JSON payload with the Collector key."""

        private = Ed25519PrivateKey.from_private_bytes(self._private_key_raw)
        return _b64(private.sign(canonical_json(dict(payload)).encode("utf-8")))


def verify_host_canonical_signature(
    payload: Mapping[str, Any],
    *,
    signature_ed25519: str,
    public_key_ed25519: str,
) -> None:
    """Verify a detached Host signature or raise a fail-closed error."""

    try:
        Ed25519PublicKey.from_public_bytes(_unb64(public_key_ed25519)).verify(
            _unb64(signature_ed25519),
            canonical_json(dict(payload)).encode("utf-8"),
        )
    except Exception as error:
        raise ValueError("Host canonical payload signature mismatch") from error


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _recv_line(connection: socket.socket, limit: int = 1024 * 1024) -> bytes:
    buffer = b""
    while not buffer.endswith(b"\n"):
        chunk = connection.recv(min(65536, limit - len(buffer)))
        if not chunk:
            break
        buffer += chunk
        if len(buffer) >= limit:
            raise ValueError("Collector request exceeds size limit")
    return buffer


def _trusted_event_payload(
    kind: str,
    capability_bodies: list[dict[str, Any]],
    contract: Mapping[str, Any],
    candidate_payload: Mapping[str, Any],
) -> dict[str, Any]:
    by_role = {str(body["role"]): body for body in capability_bodies}
    required_payloads = contract.get("required_payloads") or {}
    if kind == "split_lineage":
        payload = {
            "partition_hashes": {
                role: body["data_sha256"] for role, body in sorted(by_role.items())
            },
            "overlap_count": 0,
            "split_strategy": contract["split_strategy"],
            **dict(required_payloads.get("split_lineage") or {}),
        }
    elif kind == "fit_scope":
        train = by_role["train"]
        payload = {
            "fit_scope_hashes": {
                str(candidate_payload.get("component") or "model"): train[
                    "data_sha256"
                ]
            },
            "holdout_fit_count": 0,
            **dict(required_payloads.get("fit_scope") or {}),
        }
    elif kind == "prediction_scope":
        validation = by_role["internal_validation"]
        payload = {
            "prediction_scope_hashes": {
                "internal_validation": validation["data_sha256"]
            },
            "forbidden_overlap_count": 0,
        }
    elif kind == "evaluator":
        metric_value = float(candidate_payload.get("metric_value"))
        if not math.isfinite(metric_value):
            raise ValueError("evaluator metric_value must be finite")
        validation = by_role["internal_validation"]
        metric = dict((contract.get("evaluator_spec") or {}).get("metric") or {})
        payload = {
            "evaluator_hash": sha256_value(contract.get("evaluator_spec") or {}),
            "inputs_hash": validation["data_sha256"],
            "metric_name": str(metric.get("name") or "internal_metric"),
            "metric_direction": str(metric.get("direction") or "maximize"),
            "metric_value": metric_value,
            "tampered": False,
        }
    elif kind == "selection_freeze":
        artifact_hash = str(candidate_payload.get("artifact_hash") or "")
        if len(artifact_hash) != 64 or any(
            character not in "0123456789abcdef" for character in artifact_hash
        ):
            raise ValueError("selection_freeze requires an artifact SHA-256")
        payload = {
            "candidate_set_hash": artifact_hash,
            "frozen_before_holdout": True,
        }
    else:
        raise ValueError(f"Unsupported runtime event kind: {kind}")
    payload["candidate_payload_hash"] = sha256_value(dict(candidate_payload))
    return payload


def _validate_event_request(
    request: Mapping[str, Any],
    *,
    binding: Mapping[str, str],
    contract: Mapping[str, Any],
    secret: bytes,
    used_nonces: set[str],
    observed_kinds: list[str],
    frozen: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if request.get("action") != "emit":
        raise ValueError("Unknown Collector socket action")
    for key in ("run_id", "node_id", "code_sha256", "contract_hash"):
        if request.get(key) != binding[key]:
            raise ValueError(f"Collector binding mismatch: {key}")
    nonce = str(request.get("nonce") or "")
    if len(nonce) < 16 or nonce in used_nonces:
        raise ValueError("Collector nonce is missing or replayed")
    kind = str(request.get("kind") or "")
    if kind not in contract["required_runtime_events"]:
        raise ValueError("Event kind is not required by this Contract")
    if frozen:
        raise ValueError("Selection is frozen; no further runtime event is accepted")
    expected_index = EVENT_ORDER.index(kind)
    if expected_index and not all(
        required in observed_kinds for required in EVENT_ORDER[:expected_index]
    ):
        raise ValueError("Runtime event is out of protocol order")
    capabilities = request.get("capabilities")
    if not isinstance(capabilities, list) or not capabilities:
        raise ValueError("Runtime event requires Host-issued view capabilities")
    bodies = [verify_capability(secret, str(token)) for token in capabilities]
    if len({body["token_id"] for body in bodies}) != len(bodies):
        raise ValueError("Duplicate view capability in one runtime event")
    for body in bodies:
        if body.get("contract_hash") != binding["contract_hash"]:
            raise ValueError("View capability Contract mismatch")
        if kind not in body.get("event_kinds", []):
            raise ValueError("View capability does not permit this event kind")
    roles = {str(body["role"]) for body in bodies}
    expected_roles = {
        "split_lineage": {"train", "internal_validation"},
        "fit_scope": {"train"},
        "prediction_scope": {"internal_validation"},
        "evaluator": {"internal_validation"},
        "selection_freeze": {"internal_validation"},
    }[kind]
    if roles != expected_roles:
        raise ValueError(f"Runtime event {kind} has invalid view roles")
    used_nonces.add(nonce)
    return dict(request), bodies


def _collector_worker(
    control,
    socket_path: str,
    output_dir: str,
    contract: dict[str, Any],
    binding: dict[str, str],
    private_key_raw: bytes,
) -> None:
    server: socket.socket | None = None
    try:
        secret = secrets.token_bytes(32)
        private_key = Ed25519PrivateKey.from_private_bytes(private_key_raw)
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        events: list[dict[str, Any]] = []
        observed_kinds: list[str] = []
        used_nonces: set[str] = set()
        parent_hash = ""
        collector_processing_seconds = 0.0
        sealed = False
        socket_file = Path(socket_path)
        if socket_file.exists():
            socket_file.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        # The Candidate runs under a distinct unprivileged UID.  The socket is
        # request-only and every message still requires Host-issued capabilities;
        # broad socket access does not expose the signing key or journal writer.
        os.chmod(
            socket_path,
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IRGRP
            | stat.S_IWGRP
            | stat.S_IROTH
            | stat.S_IWOTH,
        )
        server.listen(16)
        server.settimeout(0.05)
        control.send({"status": "ready", "public_key": _b64(public_key)})
        running = True
        while running:
            if control.poll():
                command = control.recv()
                action = command.get("action")
                if action == "issue_capability":
                    if sealed:
                        control.send({"status": "error", "reason": "collector sealed"})
                        continue
                    body = {
                        "schema": "mlevolve_view_capability_v1",
                        "token_id": uuid.uuid4().hex,
                        "contract_hash": binding["contract_hash"],
                        "role": str(command["role"]),
                        "view_ref": str(command["view_ref"]),
                        "data_sha256": str(command["data_sha256"]),
                        "event_kinds": sorted(set(map(str, command["event_kinds"]))),
                    }
                    control.send(
                        {"status": "ok", "capability": mint_capability(secret, body)}
                    )
                elif action == "seal":
                    if sealed:
                        control.send({"status": "error", "reason": "already sealed"})
                        continue
                    required = list(contract["required_runtime_events"])
                    missing = sorted(set(required) - set(observed_kinds))
                    execution = {
                        "exit_status": int(command.get("exit_status", -1)),
                        "executed_path": str(command.get("executed_path") or ""),
                        "run_hash": str(command.get("run_hash") or ""),
                    }
                    valid_run_hash = len(execution["run_hash"]) == 64 and all(
                        character in "0123456789abcdef"
                        for character in execution["run_hash"]
                    )
                    coverage_pass = (
                        not missing
                        and execution["exit_status"] == 0
                        and bool(execution["executed_path"])
                        and valid_run_hash
                    )
                    journal_content = "".join(
                        canonical_json(event) + "\n" for event in events
                    ).encode("utf-8")
                    journal_sha = hashlib.sha256(journal_content).hexdigest()
                    manifest_core = {
                        "schema": RUNTIME_EVENT_JOURNAL_MANIFEST_SCHEMA,
                        "collector_id": "host.protocol_runtime.sidecar",
                        **binding,
                        "journal_sha256": journal_sha,
                        "event_count": len(events),
                        "final_event_hash": parent_hash,
                        "required_events": required,
                        "observed_events": sorted(set(observed_kinds)),
                        "collector_processing_seconds": round(
                            collector_processing_seconds, 9
                        ),
                        "execution": execution,
                        "public_key_ed25519": _b64(public_key),
                    }
                    manifest_hash = sha256_value(manifest_core)
                    signed = canonical_json(
                        {**manifest_core, "manifest_hash": manifest_hash}
                    ).encode("utf-8")
                    manifest = {
                        **manifest_core,
                        "manifest_hash": manifest_hash,
                        "signature_ed25519": _b64(private_key.sign(signed)),
                    }
                    report = {
                        "schema": RUNTIME_EVIDENCE_REPORT_SCHEMA,
                        "status": "pass" if coverage_pass else "blocked",
                        "contract_hash": binding["contract_hash"],
                        "manifest_hash": manifest_hash,
                        "missing_events": missing,
                        "event_count": len(events),
                        "collector_processing_seconds": round(
                            collector_processing_seconds, 9
                        ),
                        "terminal_exposure_count": 0,
                        "report_hash": "",
                    }
                    report["report_hash"] = hash_payload(report, "report_hash")
                    root = Path(output_dir)
                    _write_exclusive(root / "RUNTIME_EVENT_JOURNAL.jsonl", journal_content)
                    _write_exclusive(
                        root / "RUNTIME_EVENT_JOURNAL_MANIFEST.json",
                        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
                    )
                    _write_exclusive(
                        root / "RUNTIME_EVIDENCE_REPORT.json",
                        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
                    )
                    sealed = True
                    control.send({"status": "ok", "report": report})
                elif action == "stop":
                    control.send({"status": "ok"})
                    running = False
                else:
                    control.send({"status": "error", "reason": "unknown control action"})
            try:
                connection, _ = server.accept()
            except socket.timeout:
                continue
            with connection:
                request_started = time.perf_counter()
                try:
                    request = json.loads(_recv_line(connection).decode("utf-8"))
                    if sealed:
                        raise ValueError("Collector is sealed")
                    request, bodies = _validate_event_request(
                        request,
                        binding=binding,
                        contract=contract,
                        secret=secret,
                        used_nonces=used_nonces,
                        observed_kinds=observed_kinds,
                        frozen="selection_freeze" in observed_kinds,
                    )
                    trusted_payload = _trusted_event_payload(
                        request["kind"], bodies, contract, request.get("payload") or {}
                    )
                    event = {
                        "schema": RUNTIME_EVENT_JOURNAL_SCHEMA,
                        "event_index": len(events),
                        "event_id": uuid.uuid4().hex,
                        "kind": request["kind"],
                        "run_id": binding["run_id"],
                        "node_id": binding["node_id"],
                        "code_sha256": binding["code_sha256"],
                        "contract_hash": binding["contract_hash"],
                        "component": str(request.get("component") or ""),
                        "nonce": request["nonce"],
                        "view_bindings": [
                            {
                                key: body[key]
                                for key in (
                                    "token_id",
                                    "role",
                                    "view_ref",
                                    "data_sha256",
                                )
                            }
                            for body in sorted(bodies, key=lambda body: body["role"])
                        ],
                        "trusted_payload": trusted_payload,
                        "parent_event_hash": parent_hash,
                        "event_hash": "",
                    }
                    event["event_hash"] = hash_payload(event, "event_hash")
                    parent_hash = event["event_hash"]
                    events.append(event)
                    observed_kinds.append(event["kind"])
                    connection.sendall(
                        (
                            json.dumps(
                                {
                                    "status": "accepted",
                                    "event_id": event["event_id"],
                                    "event_hash": event["event_hash"],
                                }
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                except Exception as error:
                    connection.sendall(
                        (
                            json.dumps(
                                {"status": "rejected", "reason": str(error)}
                            )
                            + "\n"
                        ).encode("utf-8")
                    )
                finally:
                    collector_processing_seconds += time.perf_counter() - request_started
    except BaseException as error:
        try:
            control.send({"status": "fatal", "reason": f"{type(error).__name__}: {error}"})
        except Exception:
            pass
    finally:
        if server is not None:
            server.close()
        try:
            Path(socket_path).unlink(missing_ok=True)
        except Exception:
            pass
        control.close()


class HostCollectorSidecar:
    """Host control plane for a Candidate-inaccessible Collector process."""

    def __init__(
        self,
        output_dir: str | Path,
        contract: Mapping[str, Any],
        *,
        run_id: str,
        node_id: str,
        code_sha256: str,
        identity: HostCollectorIdentity,
        socket_path: str | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.contract = dict(contract)
        self.binding = {
            "run_id": str(run_id),
            "node_id": str(node_id),
            "code_sha256": str(code_sha256),
            "contract_hash": str(contract["contract_hash"]),
        }
        expected_public_key = str(
            (contract.get("collector_spec") or {}).get("public_key_ed25519") or ""
        )
        if expected_public_key != identity.public_key_ed25519:
            raise ValueError("Collector identity is not bound into the Execution Contract")
        self._identity = identity
        for key in ("code_sha256", "contract_hash"):
            value = self.binding[key]
            if len(value) != 64:
                raise ValueError(f"Collector {key} must be SHA-256")
        self.socket_path = socket_path or str(
            Path("/tmp") / f"mlc-{uuid.uuid4().hex[:20]}.sock"
        )
        self._process = None
        self._control = None

    def start(self, timeout_seconds: float = 30.0) -> "HostCollectorSidecar":
        if self._process is not None:
            raise RuntimeError("Collector sidecar already started")
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise ValueError("Collector output directory must be empty")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self._process = context.Process(
            target=_collector_worker,
            args=(
                child,
                self.socket_path,
                str(self.output_dir),
                self.contract,
                self.binding,
                self._identity._private_key_raw,
            ),
            daemon=True,
        )
        self._process.start()
        child.close()
        self._control = parent
        if not parent.poll(timeout_seconds):
            self.terminate()
            raise CollectorUnavailable("Collector did not become ready")
        response = parent.recv()
        if response.get("status") != "ready":
            self.terminate()
            raise CollectorUnavailable(str(response.get("reason") or response))
        return self

    def _command(self, value: dict[str, Any], timeout_seconds: float = 10.0):
        if self._process is None or self._control is None or not self._process.is_alive():
            raise CollectorUnavailable("Collector process is not alive")
        self._control.send(value)
        if not self._control.poll(timeout_seconds):
            raise CollectorUnavailable("Collector control command timed out")
        response = self._control.recv()
        if response.get("status") not in {"ok", "ready"}:
            raise CollectorRejected(str(response.get("reason") or response))
        return response

    def issue_view_capability(
        self,
        *,
        role: str,
        view_ref: str,
        data_sha256: str,
        event_kinds: tuple[str, ...],
    ) -> str:
        response = self._command(
            {
                "action": "issue_capability",
                "role": role,
                "view_ref": view_ref,
                "data_sha256": data_sha256,
                "event_kinds": list(event_kinds),
            }
        )
        return str(response["capability"])

    def client(self) -> "CollectorClient":
        from .collector_client import CollectorClient

        return CollectorClient(self.socket_path, **self.binding)

    def seal(
        self,
        *,
        exit_status: int,
        executed_path: str,
        run_hash: str,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        return self._command(
            {
                "action": "seal",
                "exit_status": exit_status,
                "executed_path": executed_path,
                "run_hash": run_hash,
            },
            timeout_seconds=timeout_seconds,
        )["report"]

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            try:
                self._command({"action": "stop"})
            except Exception:
                pass
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        if self._control is not None:
            self._control.close()
        self._process = None
        self._control = None

    def terminate(self) -> None:
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        if self._control is not None:
            self._control.close()
        self._process = None
        self._control = None
        Path(self.socket_path).unlink(missing_ok=True)

    def __enter__(self) -> "HostCollectorSidecar":
        return self.start()

    def __exit__(self, _type, _value, _traceback) -> None:
        self.stop()


def verify_collector_artifacts(
    root: str | Path, *, expected_public_key_ed25519: str
) -> dict[str, Any]:
    directory = Path(root).resolve(strict=True)
    journal_path = directory / "RUNTIME_EVENT_JOURNAL.jsonl"
    manifest_path = directory / "RUNTIME_EVENT_JOURNAL_MANIFEST.json"
    report_path = directory / "RUNTIME_EVIDENCE_REPORT.json"
    for path in (journal_path, manifest_path, report_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Collector artifact is missing, non-regular, or a symlink")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != RUNTIME_EVENT_JOURNAL_MANIFEST_SCHEMA:
        raise ValueError("Collector manifest schema mismatch")
    if report.get("schema") != RUNTIME_EVIDENCE_REPORT_SCHEMA:
        raise ValueError("Collector evidence report schema mismatch")
    if manifest.get("public_key_ed25519") != expected_public_key_ed25519:
        raise ValueError("Collector signing key is not the Host trust anchor")
    if hashlib.sha256(journal_path.read_bytes()).hexdigest() != manifest.get(
        "journal_sha256"
    ):
        raise ValueError("Collector journal file hash mismatch")
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    parent_hash = ""
    for index, event in enumerate(events):
        if event.get("schema") != RUNTIME_EVENT_JOURNAL_SCHEMA:
            raise ValueError("Collector journal event schema mismatch")
        if event.get("event_index") != index or event.get("parent_event_hash") != parent_hash:
            raise ValueError("Collector journal chain order mismatch")
        if event.get("event_hash") != hash_payload(event, "event_hash"):
            raise ValueError("Collector journal event hash mismatch")
        for key in ("run_id", "node_id", "code_sha256", "contract_hash"):
            if event.get(key) != manifest.get(key):
                raise ValueError(f"Collector event/manifest binding mismatch: {key}")
        parent_hash = event["event_hash"]
    if len(events) != manifest.get("event_count") or parent_hash != manifest.get(
        "final_event_hash"
    ):
        raise ValueError("Collector journal manifest count/final hash mismatch")
    manifest_core = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "signature_ed25519"}
    }
    if manifest.get("manifest_hash") != sha256_value(manifest_core):
        raise ValueError("Collector manifest hash mismatch")
    signed = canonical_json(
        {**manifest_core, "manifest_hash": manifest["manifest_hash"]}
    ).encode("utf-8")
    try:
        Ed25519PublicKey.from_public_bytes(
            _unb64(manifest["public_key_ed25519"])
        ).verify(_unb64(manifest["signature_ed25519"]), signed)
    except Exception as error:
        raise ValueError("Collector manifest signature mismatch") from error
    if report.get("report_hash") != hash_payload(report, "report_hash"):
        raise ValueError("Collector evidence report hash mismatch")
    if report.get("manifest_hash") != manifest.get("manifest_hash"):
        raise ValueError("Collector evidence report manifest mismatch")
    expected_missing = sorted(
        set(manifest["required_events"]) - set(manifest["observed_events"])
    )
    if report.get("missing_events") != expected_missing:
        raise ValueError("Collector evidence coverage mismatch")
    execution = manifest.get("execution") or {}
    valid_run_hash = len(str(execution.get("run_hash") or "")) == 64 and all(
        character in "0123456789abcdef"
        for character in str(execution.get("run_hash") or "")
    )
    expected_status = (
        "pass"
        if not expected_missing
        and execution.get("exit_status") == 0
        and bool(execution.get("executed_path"))
        and valid_run_hash
        else "blocked"
    )
    if report.get("status") != expected_status:
        raise ValueError("Collector evidence status does not match signed execution")
    if report.get("terminal_exposure_count") != 0:
        raise ValueError("Collector evidence reports terminal exposure")
    return {"manifest": manifest, "report": report, "events": events}


__all__ = [
    "HostCollectorIdentity",
    "HostCollectorSidecar",
    "verify_host_canonical_signature",
    "verify_collector_artifacts",
]
