#!/usr/bin/env python3
"""Run one byte-locked unique candidate from the v123 Top-10 portfolio."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import run_candidate as base
from top10_v123_catalog import CANDIDATES, EXPECTED_CODE_SHA256


base.CANDIDATES = CANDIDATES
base.DATASET_ROOT = Path(
    os.environ.get("LEAF_OFFICIAL_DATASET_ROOT", str(base.DATASET_ROOT))
)
base.DEFAULT_OUTPUT_ROOT = Path(
    "/workspace/experiment-end2end-leaf-official-top10-v123/reproductions-v1"
)
_source_code = base.source_code


def source_code_locked(journal_path: Path, node_id: str) -> str:
    code = _source_code(journal_path, node_id)
    actual = hashlib.sha256(code.encode("utf-8")).hexdigest()
    expected = EXPECTED_CODE_SHA256[node_id]
    if actual != expected:
        raise ValueError(
            f"Frozen source SHA-256 mismatch for {node_id}: {actual} != {expected}"
        )
    return code


base.source_code = source_code_locked


if __name__ == "__main__":
    raise SystemExit(base.main())
