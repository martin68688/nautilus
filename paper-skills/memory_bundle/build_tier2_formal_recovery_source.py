#!/usr/bin/env python3
"""Build the immutable source manifest for the r6 finalizer recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "decision_admissibility_wp8_tier2_formal_recovery_source_v1"
ORIGINAL_SOURCE_SCHEMA = "decision_admissibility_wp8_tier2_source_snapshot_v2"
AMENDMENT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_"
    "preterminal_finalizer_recovery_amendment_v1"
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def build_recovery_source_manifest(
    recovery_root: Path,
    *,
    original_source_manifest_path: Path,
    amendment_path: Path,
    amendment_verification_path: Path,
    diagnostic_path: Path,
    overlay_paths_path: Path,
) -> dict[str, Any]:
    recovery_root = recovery_root.resolve()
    output_path = recovery_root / "WP8_TIER2_RECOVERY_SOURCE_MANIFEST.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    original_source_manifest_path = original_source_manifest_path.resolve()
    amendment_path = amendment_path.resolve()
    amendment_verification_path = amendment_verification_path.resolve()
    diagnostic_path = diagnostic_path.resolve()
    overlay_paths_path = overlay_paths_path.resolve()
    for path in (
        original_source_manifest_path,
        amendment_path,
        amendment_verification_path,
        diagnostic_path,
        overlay_paths_path,
    ):
        if not path.is_file():
            raise ValueError(f"Recovery source input is absent: {path}")
    original = _read(original_source_manifest_path)
    if original.get("schema") != ORIGINAL_SOURCE_SCHEMA or original.get(
        "source_sha256"
    ) != _payload_hash(original, "source_sha256"):
        raise ValueError("Original formal source manifest is invalid")
    amendment = _read(amendment_path)
    if amendment.get("schema") != AMENDMENT_SCHEMA or amendment.get(
        "amendment_hash"
    ) != _payload_hash(amendment, "amendment_hash"):
        raise ValueError("Recovery amendment is invalid")
    verification = _read(amendment_verification_path)
    if (
        verification.get("verification_hash")
        != _payload_hash(verification, "verification_hash")
        or verification.get("verified") is not True
        or verification.get("errors") != []
        or verification.get("amendment_file_sha256") != _sha256_file(amendment_path)
    ):
        raise ValueError("Recovery amendment verification is invalid")
    trigger = amendment.get("triggering_failure") or {}
    diagnostic = _read(diagnostic_path)
    if diagnostic.get("diagnostic_hash") != trigger.get(
        "diagnostic_hash"
    ) or _sha256_file(diagnostic_path) != trigger.get("diagnostic_file_sha256"):
        raise ValueError("Recovery diagnostic binding mismatch")
    overlay_paths = [
        line.strip()
        for line in overlay_paths_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not overlay_paths or len(overlay_paths) != len(set(overlay_paths)):
        raise ValueError("Recovery overlay path list is empty or duplicated")
    if overlay_paths != sorted(overlay_paths):
        raise ValueError("Recovery overlay path list is not sorted")
    for relative in overlay_paths:
        relative_path = Path(relative)
        path = recovery_root / relative_path
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or not path.is_file()
            or not path.resolve().is_relative_to(recovery_root)
        ):
            raise ValueError(f"Recovery overlay path is absent: {relative}")

    excluded = {output_path.relative_to(recovery_root).as_posix()}
    files = {
        path.relative_to(recovery_root).as_posix(): _sha256_file(path)
        for path in sorted(recovery_root.rglob("*"))
        if path.is_file() and path.relative_to(recovery_root).as_posix() not in excluded
    }
    overlay_hashes = {relative: files[relative] for relative in overlay_paths}
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "purpose": ("result_blind_host_finalizer_recovery_and_recovered_cpu_evaluator"),
        "original_source_snapshot_sha256": original["source_sha256"],
        "original_source_manifest_sha256": _sha256_file(original_source_manifest_path),
        "amendment_hash": amendment["amendment_hash"],
        "amendment_file_sha256": _sha256_file(amendment_path),
        "amendment_verification_hash": verification["verification_hash"],
        "amendment_verification_file_sha256": _sha256_file(amendment_verification_path),
        "diagnostic_hash": diagnostic["diagnostic_hash"],
        "diagnostic_file_sha256": _sha256_file(diagnostic_path),
        "overlay_paths": overlay_paths,
        "overlay_hashes": overlay_hashes,
        "file_count": len(files),
        "file_hashes": files,
        "contains_training_data": False,
        "contains_terminal_labels": False,
        "contains_solver_secret": False,
        "candidate_or_agent_reexecution_authorized": False,
        "manifest_hash": "",
    }
    payload["manifest_hash"] = _payload_hash(payload, "manifest_hash")
    descriptor = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--original-source-manifest", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-verification", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--overlay-paths", type=Path, required=True)
    args = parser.parse_args()
    result = build_recovery_source_manifest(
        args.recovery_root,
        original_source_manifest_path=args.original_source_manifest,
        amendment_path=args.amendment,
        amendment_verification_path=args.amendment_verification,
        diagnostic_path=args.diagnostic,
        overlay_paths_path=args.overlay_paths,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "build_recovery_source_manifest"]
