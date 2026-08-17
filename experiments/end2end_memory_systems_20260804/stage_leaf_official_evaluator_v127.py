#!/usr/bin/env python3
"""Stage a fresh label-free Leaf official-test evaluator release on the PVC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(payload: dict, field: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(
            unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-root", required=True, type=Path)
    parser.add_argument("--source-description", required=True, type=Path)
    parser.add_argument("--output-releases-root", required=True, type=Path)
    args = parser.parse_args()

    source_dataset = args.source_dataset_root.resolve(strict=True)
    source_description = args.source_description.resolve(strict=True)
    releases_root = args.output_releases_root.resolve()
    release = releases_root / "leaf-classification" / "release"
    if release.exists():
        raise FileExistsError(f"fresh v127 evaluator release already exists: {release}")
    staging = release.parent / f".release-staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(source_dataset, staging / "dataset", copy_function=shutil.copy2)
        shutil.copy2(source_description, staging / "description.md")
        evaluator = {"kind": "deferred_official_kaggle_v1"}
        write_json(staging / "TERMINAL_EVALUATOR_SPEC.json", evaluator)
        runtime = {
            "schema": "mlevolve_leaf_official_runtime_spec_v1",
            "task_id": "leaf-classification",
            "dataset_dir": "dataset",
            "data_dir": "dataset/input",
            "description": "description.md",
            "terminal_evaluator_spec": "TERMINAL_EVALUATOR_SPEC.json",
            "terminal_evaluator_timeout_seconds": 600,
            "additional_overrides": [
                "fixed_holdout.enabled=false",
                "fixed_holdout.train_manifest_path=",
                "fixed_holdout.bypass_protocol_gates=false",
                "fixed_holdout.preflight_validate_train_view=false",
                "fixed_holdout.internal_metric_disposition=search_only",
                "official_submission.enabled=true",
                "official_submission.provider=kaggle",
                "official_submission.competition=leaf-classification",
                "official_submission.metric=log_loss",
                "official_submission.maximize=false",
                "official_submission.sample_submission_path=sample_submission.csv",
                "official_submission.id_column=id",
                "official_submission.prediction_kind=multiclass_probability",
                "official_submission.probability_row_sum_tolerance=0.0001",
                "official_submission.submission_subdir=submission",
            ],
            "runtime_artifact_sha256": {
                "description": sha256_file(staging / "description.md"),
                "terminal_evaluator_spec": sha256_file(
                    staging / "TERMINAL_EVALUATOR_SPEC.json"
                ),
            },
            "spec_hash": "",
        }
        runtime["spec_hash"] = payload_hash(runtime, "spec_hash")
        write_json(staging / "RUNTIME_SPEC.json", runtime)
        release.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(release)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    input_root = release / "dataset" / "input"
    train_rows = sum(1 for _ in (input_root / "train.csv").open(encoding="utf-8")) - 1
    test_rows = sum(1 for _ in (input_root / "test.csv").open(encoding="utf-8")) - 1
    sample_rows = (
        sum(1 for _ in (input_root / "sample_submission.csv").open(encoding="utf-8"))
        - 1
    )
    if (train_rows, test_rows, sample_rows) != (990, 594, 594):
        raise ValueError(
            "Leaf official release row contract mismatch: "
            f"{train_rows}/{test_rows}/{sample_rows}"
        )
    receipt = {
        "schema": "mlevolve_leaf_official_evaluator_stage_v1",
        "status": "complete",
        "release_root": str(release),
        "train_rows": train_rows,
        "official_test_rows": test_rows,
        "sample_submission_rows": sample_rows,
        "test_csv_sha256": sha256_file(input_root / "test.csv"),
        "sample_submission_sha256": sha256_file(
            input_root / "sample_submission.csv"
        ),
        "runtime_spec_sha256": sha256_file(release / "RUNTIME_SPEC.json"),
        "staged_at_unix": time.time(),
    }
    write_json(releases_root / "STAGING_RECEIPT.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
