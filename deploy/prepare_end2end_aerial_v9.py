#!/usr/bin/env python3
"""Clone the frozen Aerial DataView and restore terminal inference order."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil


SOURCE = Path(
    "/workspace/experiment-r-dev-r1/host-protocol-formal-r2/bindings/"
    "aerial-cactus-identification/data_views"
)
BASE = Path(
    "/workspace/experiment-end2end-host-agent-v9-sparse/"
    "aerial-cactus-identification"
)
TARGET = BASE / "data_views"
SAMPLE_PATH = Path(
    "/workspace/experiment-c-formal-releases-r3/aerial-cactus-identification/"
    "release/fixed-holdout/aerial-cactus-identification/train_view/input/"
    "sample_submission.csv"
)
RESULT_PATH = BASE.parent / "PREPARE_AERIAL_RESULT.json"


def canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def hash_payload(value: dict, field: str) -> str:
    return hashlib.sha256(
        canonical({key: item for key, item in value.items() if key != field}).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    RESULT_PATH.unlink(missing_ok=True)
    if BASE.exists():
        shutil.rmtree(BASE)
    BASE.mkdir(parents=True)
    TARGET.mkdir()
    for filename in ("DATA_VIEW_MANIFEST.json", "TRAINING_MOUNT_CONTRACT.json"):
        shutil.copy2(SOURCE / filename, TARGET / filename)
    for role in ("train_view", "internal_validation_view", "inference_view"):
        destination = TARGET / role
        destination.mkdir()
        shutil.copy2(SOURCE / role / "data.jsonl", destination / "data.jsonl")
        shutil.copy2(
            SOURCE / role / "ASSET_MANIFEST.json",
            destination / "ASSET_MANIFEST.json",
        )

    inference_path = TARGET / "inference_view" / "data.jsonl"
    rows = [
        json.loads(line)
        for line in inference_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["sample_id"]): row for row in rows}
    with SAMPLE_PATH.open(newline="", encoding="utf-8") as handle:
        expected = [str(row["id"]) for row in csv.DictReader(handle)]
    if len(by_id) != len(rows) or set(expected) != set(by_id):
        raise ValueError(
            "Frozen inference IDs do not match terminal sample submission"
        )
    ordered = [by_id[sample_id] for sample_id in expected]
    temporary = inference_path.with_name(".data.jsonl.tmp")
    temporary.write_text(
        "".join(canonical(row) + "\n" for row in ordered),
        encoding="utf-8",
    )
    os.replace(temporary, inference_path)
    inference_path.chmod(0o444)

    manifest_path = TARGET / "DATA_VIEW_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["views"]["inference"]["data_sha256"] = sha256_file(inference_path)
    manifest["manifest_hash"] = hash_payload(manifest, "manifest_hash")
    atomic_json(manifest_path, manifest)
    manifest_path.chmod(0o444)

    mount_path = TARGET / "TRAINING_MOUNT_CONTRACT.json"
    mount = json.loads(mount_path.read_text(encoding="utf-8"))
    mount["data_view_manifest_hash"] = manifest["manifest_hash"]
    for item in mount["mounts"]:
        item["source"] = str(item["source"]).replace(str(SOURCE), str(TARGET))
    mount["mount_contract_hash"] = hash_payload(mount, "mount_contract_hash")
    atomic_json(mount_path, mount)
    mount_path.chmod(0o444)

    for directory in sorted(
        (path for path in TARGET.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    TARGET.chmod(0o555)
    (BASE / "reports").mkdir()
    (BASE / "runtime").mkdir()

    result = {
        "schema": "mlevolve_end2end_dataview_order_repair_v1",
        "status": "pass",
        "source_data_view_manifest_hash": json.loads(
            (SOURCE / "DATA_VIEW_MANIFEST.json").read_text(encoding="utf-8")
        )["manifest_hash"],
        "asset_content_attestation_reused": True,
        "shared_frozen_asset_root": str(SOURCE),
        "data_view_manifest_path": str(manifest_path),
        "data_view_root": str(TARGET),
        "data_view_manifest_hash": manifest["manifest_hash"],
        "data_view_manifest_file_sha256": sha256_file(manifest_path),
        "inference_data_sha256": manifest["views"]["inference"]["data_sha256"],
        "mount_contract_hash": mount["mount_contract_hash"],
        "report_root": str(BASE / "reports"),
        "runtime_artifact_root": str(BASE / "runtime"),
        "row_count": len(ordered),
        "first_ids": expected[:3],
        "target_bytes": sum(
            path.stat().st_size for path in TARGET.rglob("*") if path.is_file()
        ),
    }
    atomic_json(RESULT_PATH, result)
    RESULT_PATH.chmod(0o444)


if __name__ == "__main__":
    main()
