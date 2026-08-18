#!/usr/bin/env python3
"""Read-only v136 PVC/runtime/evaluator and legacy-bundle initialization audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile


SYSTEMS = ("no_memory", "flat_retrieval", "sop_only", "runforest_only")
EXPECTED_TEST_SHA256 = "c70a539ec7a5e900af69307ed3f630ff2650497870170cb2d130b4158e27eeda"
EXPECTED_SAMPLE_SHA256 = "09c16c65e54876a01bafe82b140c0feeefd1b97ca81147f87b33c60879006e1a"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--evaluator", required=True, type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    base_runtime = args.base_runtime.resolve(strict=True)
    evaluator = args.evaluator.resolve(strict=True)

    assert tree_hashes(runtime / "mlevolve") == tree_hashes(base_runtime / "mlevolve")
    manifests = runtime / "experiments/end2end_memory_systems_20260804/manifests_v136"
    source_lock = json.loads((manifests / "source_lock.json").read_text(encoding="utf-8"))
    for row in source_lock["files"]:
        assert sha256_file(runtime / row["path"]) == row["sha256"], row["path"]

    memory = json.loads((manifests / "memory_bundles.json").read_text(encoding="utf-8"))
    bundle_binding = memory["task_bundles"]["leaf-classification"]
    assert bundle_binding["bundle_version"] == "v2"
    bundle_root = Path(bundle_binding["bundle_root"])

    sys.path.insert(0, str(runtime / "mlevolve"))
    from authority.memory_snapshot import MemorySnapshotLoader
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    receipts = {}
    for system_id in SYSTEMS:
        overlay = Path(tempfile.mkdtemp(prefix=f"v136-{system_id}-")) / "overlay"
        snapshot = MemorySnapshotLoader(bundle_root).load(
            session_overlay_path=overlay,
            active_protocol_ref=str(bundle_binding["protocol_ref"]),
            authority_policy_version="authority_v1",
            verify_artifacts=True,
        )
        layer = StageAwareHybridMemoryLayer(
            graph_path=str(snapshot.base_bundle.path / "runforest/graph.json"),
            index_path=str(snapshot.base_bundle.path / "runforest/index.npz"),
            source_name="run_forest_stage_hybrid_memory",
            mode="run_forest_stage_hybrid",
            scoring_mode="flat_twin",
            enable_agentic=False,
            top_k=6,
            retrieval_control="stage_hybrid",
            memory_snapshot=snapshot,
            end2end_memory_system=system_id,
            end2end_prompt_token_budget=1536,
            end2end_candidate_pool_limit=12,
            visibility_mode="off",
            visibility_task_id="leaf-classification",
            visibility_bundle_version=snapshot.base_bundle.bundle_version,
        )
        receipt = dict(layer.base_clause_receipt)
        assert receipt["status"] == "legacy_bundle_without_formal_clauses"
        assert receipt["clause_count"] == 0
        receipts[system_id] = receipt

    release = evaluator / "leaf-classification/release"
    data = release / "dataset/input"
    train = data / "train.csv"
    test = data / "test.csv"
    sample = data / "sample_submission.csv"
    assert csv_rows(train) == 990
    assert csv_rows(test) == 594
    assert csv_rows(sample) == 594
    assert sha256_file(test) == EXPECTED_TEST_SHA256
    assert sha256_file(sample) == EXPECTED_SAMPLE_SHA256
    assert (release / "RUNTIME_SPEC.json").is_file()
    assert (release / "TERMINAL_EVALUATOR_SPEC.json").is_file()

    print(json.dumps({
        "status": "pass",
        "source_locked_files": len(source_lock["files"]),
        "mlevolve_tree_unchanged": True,
        "bundle_id": bundle_binding["bundle_id"],
        "bundle_version": bundle_binding["bundle_version"],
        "base_clause_receipts": receipts,
        "evaluator_rows": {"train": 990, "test": 594, "sample": 594},
        "test_sha256": EXPECTED_TEST_SHA256,
        "sample_sha256": EXPECTED_SAMPLE_SHA256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
