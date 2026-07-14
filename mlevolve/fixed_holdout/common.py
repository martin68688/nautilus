"""Shared manifest and hashing utilities for fixed holdout evaluation."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "mlevolve_fixed_holdout_v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def tree_sha256(root: Path, max_workers: int = 16) -> str:
    if not root.is_dir():
        raise ValueError(f"Expected a directory to hash: {root}")
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    def hash_one(path: Path) -> tuple[str, str]:
        relative = path.relative_to(root).as_posix()
        return relative, sha256_file(path)

    with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(files)))) as executor:
        hashed_files = list(executor.map(hash_one, files))
    for relative, file_digest in hashed_files:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_manifest(path: Path, expected_role: str | None = None) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read fixed-holdout manifest {path}: {exc}") from exc
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported fixed-holdout schema in {path}")
    if expected_role and manifest.get("role") != expected_role:
        raise ValueError(
            f"Manifest {path} has role={manifest.get('role')!r}; "
            f"expected {expected_role!r}"
        )
    return manifest


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
