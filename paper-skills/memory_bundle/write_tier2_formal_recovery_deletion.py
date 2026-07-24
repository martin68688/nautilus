#!/usr/bin/env python3
"""Write the host-observed recovery-devpod deletion attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_" "recovery_pod_deletion_attestation_v1"
)
RECOVERY_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_" "preterminal_finalizer_recovery_v1"
)


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_recovery_deletion(
    block_root: Path,
    *,
    namespace: str,
    pod_name: str,
    pod_uid: str,
    delete_requested_at: str,
    not_found_verified_at: str,
    not_found_probe_sha256: str,
) -> dict[str, Any]:
    block_root = block_root.resolve()
    recovery = _read(block_root / "TRAINING_FINALIZER_RECOVERY.json")
    training = _read(block_root / "TRAINING_MANIFEST.json")
    if recovery.get("schema") != RECOVERY_SCHEMA or recovery.get(
        "receipt_hash"
    ) != _payload_hash(recovery, "receipt_hash"):
        raise ValueError("Training finalizer recovery Receipt is invalid")
    if training.get("manifest_hash") != _payload_hash(training, "manifest_hash"):
        raise ValueError("Recovered training manifest is invalid")
    identity = {
        "execution_kind": "devpod",
        "namespace": namespace,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
    }
    if identity != recovery.get("recovery_pod_identity"):
        raise ValueError("Recovery Pod identity mismatch")
    if recovery.get("training_manifest_hash") != training.get("manifest_hash"):
        raise ValueError("Recovery/training manifest binding mismatch")
    if (
        recovery.get("terminal_metric_observed") is not False
        or recovery.get("candidate_code_reexecuted") is not False
        or recovery.get("full_condition_reexecuted") is not False
    ):
        raise ValueError("Recovery Receipt violates result-blind recovery")
    if len(not_found_probe_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in not_found_probe_sha256
    ):
        raise ValueError("Invalid recovery Pod NotFound probe SHA-256")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "block_id": training["block_id"],
        "training_manifest_hash": training["manifest_hash"],
        "recovery_receipt_hash": recovery["receipt_hash"],
        "recovery_pod_identity": identity,
        "delete_requested": True,
        "delete_requested_at": delete_requested_at,
        "not_found_verified": True,
        "not_found_verified_at": not_found_verified_at,
        "kubernetes_reason": "NotFound",
        "not_found_probe_sha256": not_found_probe_sha256,
        "verified_by": "host_launcher",
        "terminal_metric_observed_before_not_found": False,
        "evaluator_create_allowed_after_verification": True,
        "attestation_hash": "",
    }
    payload["attestation_hash"] = _payload_hash(payload, "attestation_hash")
    output = block_root / "RECOVERY_POD_DELETION_ATTESTATION.json"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-root", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--pod-uid", required=True)
    parser.add_argument("--delete-requested-at", required=True)
    parser.add_argument("--not-found-verified-at", required=True)
    parser.add_argument("--not-found-probe-sha256", required=True)
    args = parser.parse_args()
    result = write_recovery_deletion(
        args.block_root,
        namespace=args.namespace,
        pod_name=args.pod_name,
        pod_uid=args.pod_uid,
        delete_requested_at=args.delete_requested_at,
        not_found_verified_at=args.not_found_verified_at,
        not_found_probe_sha256=args.not_found_probe_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "write_recovery_deletion"]
