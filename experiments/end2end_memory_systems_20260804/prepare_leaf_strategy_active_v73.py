#!/usr/bin/env python3
"""Build an immutable v73 Leaf dev-smoke source from frozen v61 + overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import tarfile
from typing import Any


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: dict[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object in {path}")
    return value


def write_object(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def make_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            raise ValueError(f"frozen source contains symlink: {path}")
        path.chmod(path.stat().st_mode | stat.S_IWUSR)


def extract_overlay(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            target.relative_to(destination.resolve())
            if member.issym() or member.islnk():
                raise ValueError(f"overlay may not contain links: {member.name}")
        handle.extractall(destination)


def freeze(root: Path) -> tuple[int, str]:
    lock_path = root / "SOURCE_FILES.sha256"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != lock_path and not path.is_symlink()
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(root)}" for path in files]
    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    for path in [*root.rglob("*"), root]:
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    return len(files), manifest_sha


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.resolve(strict=True)
    destination = args.destination.resolve()
    overlay = args.overlay.resolve(strict=True)
    if destination.exists():
        raise FileExistsError(f"refusing to replace immutable source: {destination}")
    shutil.copytree(base, destination, symlinks=False)
    make_writable(destination)
    extract_overlay(overlay, destination)

    exp_root = destination / "experiments/end2end_memory_systems_20260804"
    source_manifests = exp_root / "manifests_v61"
    manifests = exp_root / "manifests_v73"
    if manifests.exists():
        raise FileExistsError(f"overlay unexpectedly created {manifests}")
    shutil.copytree(source_manifests, manifests)
    make_writable(manifests)

    config_path = exp_root / "systems_v73/dynamic_hybrid.yaml"
    systems_path = manifests / "systems.json"
    systems = read_object(systems_path)
    systems["experimental_axis"] = (
        "Leaf required Strategy -> Atomic Planner -> Coder online actuation"
    )
    systems["systems"] = [
        {
            "config_path": "systems_v73/dynamic_hybrid.yaml",
            "config_sha256": sha256_file(config_path),
            "description": (
                "Dynamic Router plus required active Strategy actuation for Improve/Debug"
            ),
            "kind": "internal_exploratory",
            "label": "S5-v73-active",
            "limitation": "single exploratory dev-pod smoke",
            "primary_reference": None,
            "system_id": "dynamic_hybrid",
        }
    ]
    systems["system_count"] = 1
    systems["manifest_hash"] = payload_hash(systems, "manifest_hash")
    write_object(systems_path, systems)

    replay_source = source_manifests / "leaf_official_replay_targets.json"
    replay_target = manifests / "leaf_official_replay_targets.json"
    shutil.copy2(replay_source, replay_target)

    source_lock_path = manifests / "source_lock.json"
    source_lock = read_object(source_lock_path)
    source_lock["git_head"] = args.git_head
    source_lock["git_dirty"] = False
    source_lock["files"] = []
    source_lock["complete_runtime_file_hash_lock"] = True
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if path in {source_lock_path, destination / "SOURCE_FILES.sha256"}:
            continue
        source_lock["files"].append(
            {
                "path": str(path.relative_to(destination)),
                "sha256": sha256_file(path),
            }
        )
    source_lock["manifest_hash"] = payload_hash(source_lock, "manifest_hash")
    write_object(source_lock_path, source_lock)

    smoke_source = source_manifests / "leaf_official_smoke_manifest.json"
    smoke_path = manifests / "leaf_strategy_active_smoke_manifest.json"
    smoke = read_object(smoke_source)
    logical_id = (
        "e2e-dev-smoke-leaf-strategy-active-v73__leaf-classification__"
        "dynamic_hybrid__seed-1"
    )
    row = dict(smoke["runs"][0])
    row.update(
        {
            "logical_run_id": logical_id,
            "launch_position": 0,
            "task_launch_position": 0,
            "formal_result_eligible": False,
            "exploratory_pilot": True,
        }
    )
    bindings = dict(smoke["bindings"])
    bindings["systems_manifest_hash"] = systems["manifest_hash"]
    bindings["source_lock_manifest_hash"] = source_lock["manifest_hash"]
    row["bindings"] = dict(bindings)
    row["row_hash"] = payload_hash(row, "row_hash")
    smoke.update(
        {
            "release_id": "end2end-leaf-strategy-active-v73-dev-smoke",
            "kind": "smoke",
            "run_count": 1,
            "runs": [row],
            "system_ids": ["dynamic_hybrid"],
            "task_ids": ["leaf-classification"],
            "first_parallel_batch": ["dynamic_hybrid"],
            "launch_order_randomization": "single active Dynamic Leaf dev smoke",
            "formal_result_eligible": False,
            "exploratory_pilot": True,
            "bindings": bindings,
        }
    )
    smoke["manifest_hash"] = payload_hash(smoke, "manifest_hash")
    write_object(smoke_path, smoke)

    file_count, source_manifest_sha = freeze(destination)
    receipt = {
        "schema": "mlevolve_leaf_strategy_active_source_release_v1",
        "git_head": args.git_head,
        "base_source": str(base),
        "frozen_source": str(destination),
        "source_file_count": file_count,
        "source_manifest_sha256": source_manifest_sha,
        "systems_manifest_sha256": systems["manifest_hash"],
        "source_lock_manifest_sha256": source_lock["manifest_hash"],
        "smoke_manifest_sha256": smoke["manifest_hash"],
        "smoke_manifest": str(smoke_path),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
