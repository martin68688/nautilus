"""Host bootstrap for Contract-bound lifecycle evidence during full training."""

from __future__ import annotations

import atexit
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from authority.protocol_execution_contract import ProtocolExecutionContract

from .collector import HostCollectorIdentity, HostCollectorSidecar
from .collector_client import CollectorClient
from .data_views import verify_data_view_manifest
from .events import canonical_json
from .session import ProtocolSession, activate_session
from .views import DataViewHandle, ProtocolSplit, build_view_handles


FULL_RUNTIME_BOOTSTRAP_SCHEMA = "mlevolve_full_runtime_bootstrap_v1"
FULL_RUNTIME_EVIDENCE_SCHEMA = "mlevolve_full_runtime_evidence_v1"
_ACTIVE_CONTEXTS: list[Any] = []


def _hash_payload(payload: dict[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_json({key: value for key, value in payload.items() if key != field}).encode(
            "utf-8"
        )
    ).hexdigest()


def _handle_payload(handle: DataViewHandle) -> dict[str, Any]:
    return {
        "role": handle.role,
        "view_ref": handle.view_ref,
        "contract_hash": handle.contract_hash,
        "manifest_hash": handle.manifest_hash,
        "data_sha256": handle.data_sha256,
        "path": str(handle._path),
        "capability": handle._capability,
    }


def _handle_from_payload(payload: dict[str, Any]) -> DataViewHandle:
    expected = {
        "role",
        "view_ref",
        "contract_hash",
        "manifest_hash",
        "data_sha256",
        "path",
        "capability",
    }
    if set(payload) != expected:
        raise ValueError("Full-runtime DataView handle fields do not match schema")
    return DataViewHandle(
        role=str(payload["role"]),
        view_ref=str(payload["view_ref"]),
        contract_hash=str(payload["contract_hash"]),
        manifest_hash=str(payload["manifest_hash"]),
        data_sha256=str(payload["data_sha256"]),
        _path=Path(str(payload["path"])).resolve(strict=True),
        _capability=str(payload["capability"]),
    )


class FullRuntimeEvidenceController:
    """Host-only lifecycle manager for one full Candidate subprocess."""

    def __init__(
        self,
        *,
        contract: ProtocolExecutionContract,
        identity: HostCollectorIdentity,
        data_view_manifest_path: str | Path,
        output_root: str | Path,
        bootstrap_path: str | Path,
        run_id: str,
        node_id: str,
        code_sha256: str,
    ):
        self.contract = contract
        self.identity = identity
        self.data_view_manifest_path = Path(data_view_manifest_path).resolve(strict=True)
        self.output_root = Path(output_root).resolve()
        self.bootstrap_path = Path(bootstrap_path).resolve()
        self.run_id = str(run_id)
        self.node_id = str(node_id)
        self.code_sha256 = str(code_sha256)
        self.sidecar: HostCollectorSidecar | None = None
        self.report: dict[str, Any] = {}

    def start(self) -> "FullRuntimeEvidenceController":
        if self.sidecar is not None:
            raise RuntimeError("Full-runtime evidence controller already started")
        verify_data_view_manifest(
            self.data_view_manifest_path, contract=self.contract
        )
        self.sidecar = HostCollectorSidecar(
            self.output_root / "collector",
            self.contract.as_dict(),
            run_id=self.run_id,
            node_id=self.node_id,
            code_sha256=self.code_sha256,
            identity=self.identity,
        ).start()
        split = build_view_handles(
            self.data_view_manifest_path, self.contract, self.sidecar
        )
        client = self.sidecar.client()
        payload = {
            "schema": FULL_RUNTIME_BOOTSTRAP_SCHEMA,
            "contract": self.contract.as_dict(),
            "split": {
                "train": _handle_payload(split.train),
                "validation": _handle_payload(split.validation),
                **(
                    {"inference": _handle_payload(split.inference)}
                    if split.inference is not None
                    else {}
                ),
            },
            "client": {
                "socket_path": client.socket_path,
                "run_id": client.run_id,
                "node_id": client.node_id,
                "code_sha256": client.code_sha256,
                "contract_hash": client.contract_hash,
            },
            "bootstrap_hash": "",
        }
        payload["bootstrap_hash"] = _hash_payload(payload, "bootstrap_hash")
        self.bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.bootstrap_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return self

    def seal(
        self, *, exit_status: int, executed_path: str, run_hash: str
    ) -> dict[str, Any]:
        if self.sidecar is None:
            raise RuntimeError("Full-runtime evidence controller is not started")
        collector_report = self.sidecar.seal(
            exit_status=exit_status,
            executed_path=executed_path,
            run_hash=run_hash,
        )
        self.report = {
            "schema": FULL_RUNTIME_EVIDENCE_SCHEMA,
            "status": str(collector_report["status"]),
            "run_id": self.run_id,
            "node_id": self.node_id,
            "code_sha256": self.code_sha256,
            "contract_hash": self.contract.contract_hash,
            "contract": self.contract.as_dict(),
            "collector_root": str(self.sidecar.output_dir),
            "collector_report_hash": str(collector_report["report_hash"]),
            "missing_events": list(collector_report.get("missing_events") or []),
            "evidence_hash": "",
        }
        self.report["evidence_hash"] = _hash_payload(self.report, "evidence_hash")
        return dict(self.report)

    def stop(self) -> None:
        if self.sidecar is not None:
            self.sidecar.stop()
            self.sidecar = None


def activate_full_runtime_from_bootstrap(path: str | Path) -> ProtocolSession:
    """Candidate-side bootstrap invoked by Host-injected pre-code."""

    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError("Full-runtime bootstrap is missing or a symlink")
    payload = json.loads(requested.read_text(encoding="utf-8"))
    if payload.get("schema") != FULL_RUNTIME_BOOTSTRAP_SCHEMA:
        raise ValueError("Full-runtime bootstrap schema mismatch")
    if payload.get("bootstrap_hash") != _hash_payload(payload, "bootstrap_hash"):
        raise ValueError("Full-runtime bootstrap hash mismatch")
    if set(payload) != {"schema", "contract", "split", "client", "bootstrap_hash"}:
        raise ValueError("Full-runtime bootstrap fields do not match schema")
    contract = ProtocolExecutionContract.from_dict(payload["contract"])
    split_payload = dict(payload["split"])
    if set(split_payload) not in (
        {"train", "validation"},
        {"train", "validation", "inference"},
    ):
        raise ValueError("Full-runtime bootstrap split fields do not match schema")
    split = ProtocolSplit(
        train=_handle_from_payload(dict(split_payload["train"])),
        validation=_handle_from_payload(dict(split_payload["validation"])),
        inference=(
            _handle_from_payload(dict(split_payload["inference"]))
            if "inference" in split_payload
            else None
        ),
    )
    client_payload = dict(payload["client"])
    if set(client_payload) != {
        "socket_path",
        "run_id",
        "node_id",
        "code_sha256",
        "contract_hash",
    }:
        raise ValueError("Full-runtime Collector client fields do not match schema")
    if client_payload["contract_hash"] != contract.contract_hash:
        raise ValueError("Full-runtime bootstrap Contract binding mismatch")
    session = ProtocolSession(
        contract,
        split,
        CollectorClient(**client_payload),
    )
    context = activate_session(session)
    context.__enter__()
    _ACTIVE_CONTEXTS.append(context)

    def deactivate() -> None:
        if context in _ACTIVE_CONTEXTS:
            _ACTIVE_CONTEXTS.remove(context)
            context.__exit__(None, None, None)

    atexit.register(deactivate)
    return session


def deactivate_full_runtime() -> None:
    """Close active bootstrap contexts; primarily used by in-process Host tests."""

    while _ACTIVE_CONTEXTS:
        context = _ACTIVE_CONTEXTS.pop()
        context.__exit__(None, None, None)


__all__ = [
    "FULL_RUNTIME_BOOTSTRAP_SCHEMA",
    "FULL_RUNTIME_EVIDENCE_SCHEMA",
    "FullRuntimeEvidenceController",
    "activate_full_runtime_from_bootstrap",
    "deactivate_full_runtime",
]
