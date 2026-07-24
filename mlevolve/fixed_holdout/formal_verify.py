"""Independently verify materialized WP8 Tier-2 formal holdouts.

The builders in :mod:`fixed_holdout.formal_prepare` are intentionally not
trusted merely because they returned successfully.  This module re-reads the
published views, recomputes their hashes, checks physical and semantic label
isolation, and emits a content-addressed verification report suitable for the
formal staging manifest.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from fixed_holdout.common import sha256_file, sha256_lines, tree_sha256
from fixed_holdout.formal_prepare import (
    AERIAL_PROTOCOL_REF,
    AERIAL_SPLIT_VERSION,
    AERIAL_TASK,
    BIRDS_PROTOCOL_REF,
    BIRDS_SPLIT_VERSION_R2,
    BIRDS_TASK,
    TAXI_PROTOCOL_REF,
    TAXI_SPLIT_VERSION,
    TAXI_TASK,
)


REPORT_SCHEMA = "formal_holdout_verification_v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _file_inventory(root: Path) -> tuple[dict[str, str], list[str], list[str]]:
    files: list[tuple[str, Path]] = []
    symlinks: list[str] = []
    multi_link_files: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
            continue
        if not path.is_file():
            continue
        files.append((relative, path))
        if path.stat().st_nlink != 1:
            multi_link_files.append(relative)
    def hash_one(item: tuple[str, Path]) -> tuple[str, str]:
        relative, path = item
        return relative, sha256_file(path)

    with ThreadPoolExecutor(max_workers=min(32, max(1, len(files)))) as pool:
        hashes = dict(pool.map(hash_one, files))
    return hashes, symlinks, multi_link_files


def _inode_set(root: Path) -> set[tuple[int, int]]:
    return {
        (path.stat().st_dev, path.stat().st_ino)
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _receipt_valid(value: object, schema: str) -> bool:
    return bool(
        isinstance(value, Mapping)
        and value.get("schema") == schema
        and value.get("receipt_hash") == _payload_hash(value, "receipt_hash")
    )


def _same_fields(
    values: list[Mapping[str, Any]],
    fields: tuple[str, ...],
) -> bool:
    return all(
        all(value.get(field) == values[0].get(field) for value in values[1:])
        for field in fields
    )


def _counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame[column].value_counts().sort_index().items()
    }


def _verify_aerial(
    input_dir: Path,
    labels: pd.DataFrame,
    split_receipt: Mapping[str, Any],
    require,
) -> dict[str, Any]:
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    sample = pd.read_csv(input_dir / "sample_submission.csv")
    require(list(train.columns) == ["id", "has_cactus"], "aerial_train_schema")
    require(list(test.columns) == ["id"], "aerial_test_schema")
    require(
        list(sample.columns) == ["id", "has_cactus"],
        "aerial_submission_schema",
    )
    train_ids = train["id"].astype(str).tolist()
    test_ids = test["id"].astype(str).tolist()
    label_ids = labels["id"].astype(str).tolist()
    require(len(train_ids) == len(set(train_ids)), "aerial_train_ids_unique")
    require(len(test_ids) == len(set(test_ids)), "aerial_test_ids_unique")
    require(set(train_ids).isdisjoint(test_ids), "aerial_partition_overlap")
    require(test_ids == label_ids, "aerial_label_order")
    require(sample["id"].astype(str).tolist() == test_ids, "aerial_sample_order")
    require(
        np.allclose(sample["has_cactus"].to_numpy(dtype=float), 0.0),
        "aerial_sample_must_not_encode_labels",
    )
    require(set(train["has_cactus"].unique()) == {0, 1}, "aerial_train_classes")
    require(set(labels["has_cactus"].unique()) == {0, 1}, "aerial_holdout_classes")
    train_image_ids = sorted(path.name for path in (input_dir / "train").iterdir())
    test_image_ids = sorted(path.name for path in (input_dir / "test").iterdir())
    require(train_image_ids == sorted(train_ids), "aerial_train_image_membership")
    require(test_image_ids == sorted(test_ids), "aerial_test_image_membership")
    train_counts = _counts(train, "has_cactus")
    holdout_counts = _counts(labels, "has_cactus")
    require(
        split_receipt.get("train_class_counts") == train_counts,
        "aerial_train_class_receipt",
    )
    require(
        split_receipt.get("holdout_class_counts") == holdout_counts,
        "aerial_holdout_class_receipt",
    )
    require(split_receipt.get("overlap_count") == 0, "aerial_overlap_receipt")
    require(
        split_receipt.get("stratification_verified") is True,
        "aerial_stratification_receipt",
    )
    return {
        "train_count": len(train),
        "holdout_count": len(test),
        "train_class_counts": train_counts,
        "holdout_class_counts": holdout_counts,
    }


def _verify_birds(
    root: Path,
    input_dir: Path,
    labels: pd.DataFrame,
    split_receipt: Mapping[str, Any],
    require,
) -> dict[str, Any]:
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    sample = pd.read_csv(input_dir / "sample_submission.csv")
    class_columns = [str(index) for index in range(19)]
    metadata_columns = ["rec_id", "filename", "recording_session_id"]
    require(
        list(train.columns) == [*metadata_columns, *class_columns],
        "birds_train_schema",
    )
    require(list(test.columns) == metadata_columns, "birds_test_schema")
    require(
        list(labels.columns) == ["rec_id", *class_columns],
        "birds_label_schema",
    )
    require(
        list(sample.columns) == ["rec_id", *class_columns],
        "birds_submission_schema",
    )
    train_ids = train["rec_id"].astype(str).tolist()
    test_ids = test["rec_id"].astype(str).tolist()
    label_ids = labels["rec_id"].astype(str).tolist()
    require(len(train_ids) == len(set(train_ids)), "birds_train_ids_unique")
    require(len(test_ids) == len(set(test_ids)), "birds_test_ids_unique")
    require(set(train_ids).isdisjoint(test_ids), "birds_record_overlap")
    require(test_ids == label_ids, "birds_label_order")
    require(sample["rec_id"].astype(str).tolist() == test_ids, "birds_sample_order")
    require(
        np.allclose(sample[class_columns].to_numpy(dtype=float), 0.0),
        "birds_sample_must_not_encode_labels",
    )
    train_groups = set(train["recording_session_id"].astype(str))
    test_groups = set(test["recording_session_id"].astype(str))
    require(train_groups.isdisjoint(test_groups), "birds_group_overlap")
    train_positive = [int(train[column].sum()) for column in class_columns]
    holdout_positive = [int(labels[column].sum()) for column in class_columns]
    require(all(value > 0 for value in train_positive), "birds_train_class_coverage")
    require(
        all(value > 0 for value in holdout_positive),
        "birds_holdout_class_coverage",
    )
    require(
        split_receipt.get("train_positive_counts") == train_positive,
        "birds_train_coverage_receipt",
    )
    require(
        split_receipt.get("holdout_positive_counts") == holdout_positive,
        "birds_holdout_coverage_receipt",
    )
    require(split_receipt.get("record_overlap_count") == 0, "birds_record_receipt")
    require(split_receipt.get("group_overlap_count") == 0, "birds_group_receipt")
    require(
        split_receipt.get("fold1_record_count_in_views") == 0,
        "birds_fold1_receipt",
    )
    train_files = sorted(path.name for path in (input_dir / "train_wavs").iterdir())
    test_files = sorted(path.name for path in (input_dir / "test_wavs").iterdir())
    require(train_files == sorted(train["filename"].astype(str)), "birds_train_wavs")
    require(test_files == sorted(test["filename"].astype(str)), "birds_test_wavs")
    feasibility = _read_json(root / "birds_feasibility_report.json")
    require(
        feasibility.get("report_hash") == _payload_hash(feasibility, "report_hash"),
        "birds_feasibility_hash",
    )
    require(feasibility.get("checks_passed") is True, "birds_feasibility_pass")
    require(
        feasibility.get("report_hash")
        == split_receipt.get("feasibility_report_hash"),
        "birds_feasibility_receipt_binding",
    )
    require(
        feasibility.get("selected_group_ids_sha256")
        == sha256_lines(sorted(test_groups)),
        "birds_selected_group_hash",
    )
    return {
        "train_count": len(train),
        "holdout_count": len(test),
        "train_group_count": len(train_groups),
        "holdout_group_count": len(test_groups),
        "train_positive_counts": train_positive,
        "holdout_positive_counts": holdout_positive,
    }


def _valid_taxi_rows(frame: pd.DataFrame, *, has_target: bool) -> bool:
    numeric = [
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    if has_target:
        numeric.append("fare_amount")
    try:
        values = frame[numeric].to_numpy(dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    if not np.isfinite(values).all():
        return False
    valid = (
        frame["passenger_count"].between(1, 6)
        & frame["pickup_longitude"].between(-75, -72)
        & frame["dropoff_longitude"].between(-75, -72)
        & frame["pickup_latitude"].between(40, 42)
        & frame["dropoff_latitude"].between(40, 42)
    )
    if has_target:
        valid &= frame["fare_amount"].between(2.5, 200)
    return bool(valid.all())


def _verify_taxi(
    input_dir: Path,
    labels: pd.DataFrame,
    split_receipt: Mapping[str, Any],
    require,
) -> dict[str, Any]:
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    sample = pd.read_csv(input_dir / "sample_submission.csv")
    feature_columns = [
        "key",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    require(
        list(train.columns)
        == ["key", "fare_amount", *feature_columns[1:]],
        "taxi_train_schema",
    )
    require(list(test.columns) == feature_columns, "taxi_test_schema")
    require(list(labels.columns) == ["key", "fare_amount"], "taxi_label_schema")
    require(
        list(sample.columns) == ["key", "fare_amount"],
        "taxi_submission_schema",
    )
    train_keys = train["key"].astype(str).tolist()
    test_keys = test["key"].astype(str).tolist()
    label_keys = labels["key"].astype(str).tolist()
    require(len(train_keys) == len(set(train_keys)), "taxi_train_keys_unique")
    require(len(test_keys) == len(set(test_keys)), "taxi_test_keys_unique")
    require(set(train_keys).isdisjoint(test_keys), "taxi_key_overlap")
    require(test_keys == label_keys, "taxi_label_order")
    require(sample["key"].astype(str).tolist() == test_keys, "taxi_sample_order")
    require(
        np.allclose(sample["fare_amount"].to_numpy(dtype=float), 0.0),
        "taxi_sample_must_not_encode_labels",
    )
    require(_valid_taxi_rows(train, has_target=True), "taxi_train_filters")
    test_with_labels = test.merge(labels, on="key", how="left", validate="one_to_one")
    require(
        len(test_with_labels) == len(test) and not test_with_labels["fare_amount"].isna().any(),
        "taxi_evaluator_label_join",
    )
    require(_valid_taxi_rows(test_with_labels, has_target=True), "taxi_holdout_filters")
    train_time = pd.to_datetime(train["pickup_datetime"], utc=True, errors="coerce")
    test_time = pd.to_datetime(test["pickup_datetime"], utc=True, errors="coerce")
    require(not train_time.isna().any(), "taxi_train_time_parse")
    require(not test_time.isna().any(), "taxi_test_time_parse")
    max_train = train_time.max().strftime("%Y-%m-%dT%H:%M:%SZ")
    min_holdout = test_time.min().strftime("%Y-%m-%dT%H:%M:%SZ")
    cutoff = pd.Timestamp("2014-07-01T00:00:00Z")
    require(bool((train_time < cutoff).all()), "taxi_train_before_cutoff")
    require(bool((test_time >= cutoff).all()), "taxi_holdout_after_cutoff")
    require(max_train < min_holdout, "taxi_strict_chronology")
    require(
        all(
            int.from_bytes(
                hashlib.sha256(
                    f"{TAXI_SPLIT_VERSION}\0{key}".encode("utf-8")
                ).digest()[:8],
                "big",
                signed=False,
            )
            % 256
            == 0
            for key in [*train_keys, *test_keys]
        ),
        "taxi_sampling_rule",
    )
    require(len(train) >= 50_000, "taxi_minimum_train_rows")
    require(len(test) >= 10_000, "taxi_minimum_holdout_rows")
    require(
        split_receipt.get("train_holdout_key_overlap_count") == 0,
        "taxi_overlap_receipt",
    )
    require(split_receipt.get("future_to_past_count") == 0, "taxi_future_receipt")
    require(
        split_receipt.get("max_train_pickup_datetime") == max_train,
        "taxi_max_train_receipt",
    )
    require(
        split_receipt.get("min_holdout_pickup_datetime") == min_holdout,
        "taxi_min_holdout_receipt",
    )
    require(split_receipt.get("train_count") == len(train), "taxi_train_count_receipt")
    require(split_receipt.get("holdout_count") == len(test), "taxi_holdout_count_receipt")
    return {
        "train_count": len(train),
        "holdout_count": len(test),
        "max_train_pickup_datetime": max_train,
        "min_holdout_pickup_datetime": min_holdout,
    }


def verify_formal_holdout(
    root: str | Path,
    *,
    verify_source_artifacts: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    errors: list[str] = []
    checks: list[str] = []

    def require(condition: object, code: str) -> None:
        if bool(condition):
            checks.append(code)
        else:
            errors.append(code)

    try:
        split_path = root / "split_manifest.json"
        train_path = root / "train_view" / "fixed_holdout_manifest.json"
        evaluator_path = root / "evaluator_view" / "fixed_holdout_manifest.json"
        split = _read_json(split_path)
        train_manifest = _read_json(train_path)
        evaluator_manifest = _read_json(evaluator_path)
        manifests = [split, train_manifest, evaluator_manifest]
        task_id = str(split.get("task_id") or "")
        expected = {
            AERIAL_TASK: (AERIAL_SPLIT_VERSION, AERIAL_PROTOCOL_REF, "macro_f1", True),
            BIRDS_TASK: (BIRDS_SPLIT_VERSION_R2, BIRDS_PROTOCOL_REF, "macro_f1", True),
            TAXI_TASK: (TAXI_SPLIT_VERSION, TAXI_PROTOCOL_REF, "rmse", False),
        }
        require(task_id in expected, "known_formal_task")
        expected_version, expected_protocol, expected_metric, expected_maximize = expected.get(
            task_id, ("", "", "", False)
        )
        require(root.name == split.get("split_id"), "split_directory_identity")
        require(split.get("manifest_hash") == _payload_hash(split, "manifest_hash"), "split_manifest_hash")
        require(split.get("train_manifest_sha256") == sha256_file(train_path), "train_manifest_hash")
        require(
            split.get("evaluator_manifest_sha256") == sha256_file(evaluator_path),
            "evaluator_manifest_hash",
        )
        require(train_manifest.get("role") == "train_view", "train_role")
        require(evaluator_manifest.get("role") == "evaluator_view", "evaluator_role")
        common_fields = (
            "formal_schema",
            "task_id",
            "split_id",
            "split_version",
            "protocol_ref",
            "metric",
            "maximize",
            "id_column",
            "prediction_columns",
            "row_count",
            "holdout_id_sha256",
            "public_tree_sha256",
            "selection_policy",
            "split_receipt",
            "fit_scope_receipt",
            "metric_spec_receipt",
            "source_artifacts",
        )
        require(_same_fields(manifests, common_fields), "manifest_common_binding")
        require(split.get("split_version") == expected_version, "split_version")
        require(split.get("protocol_ref") == expected_protocol, "protocol_ref")
        require(split.get("metric") == expected_metric, "terminal_metric")
        require(split.get("maximize") is expected_maximize, "metric_direction")
        require(split.get("selection_policy") == "terminal_only", "terminal_only")
        require(train_manifest.get("hidden_labels_present") is False, "hidden_labels_flag")
        require(
            train_manifest.get("final_metric_available_to_training") is False,
            "terminal_metric_hidden_flag",
        )
        require(train_manifest.get("hardlink_count") == 0, "hardlink_manifest_flag")

        input_dir = root / "train_view" / "input"
        labels_path = root / "evaluator_view" / "labels.csv"
        require(tree_sha256(input_dir) == split.get("public_tree_sha256"), "public_tree_hash")
        require(sha256_file(labels_path) == split.get("labels_sha256"), "labels_hash")
        require(
            evaluator_manifest.get("labels_sha256") == split.get("labels_sha256"),
            "evaluator_labels_binding",
        )
        labels = pd.read_csv(labels_path)
        id_column = str(split.get("id_column") or "")
        prediction_columns = [str(value) for value in split.get("prediction_columns") or []]
        require(list(labels.columns) == [id_column, *prediction_columns], "label_schema")
        require(len(labels) == int(split.get("row_count", -1)), "label_row_count")
        require(
            sha256_lines(labels[id_column].astype(str)) == split.get("holdout_id_sha256"),
            "holdout_id_hash",
        )
        require(
            not any(path.name == "labels.csv" for path in (root / "train_view").rglob("*")),
            "no_terminal_label_file_in_train_view",
        )
        require(
            _receipt_valid(split.get("split_receipt"), "formal_split_lineage_receipt_v1"),
            "split_receipt_hash",
        )
        require(
            _receipt_valid(split.get("fit_scope_receipt"), "formal_fit_scope_receipt_v1"),
            "fit_receipt_hash",
        )
        require(
            _receipt_valid(split.get("metric_spec_receipt"), "formal_metric_spec_receipt_v1"),
            "metric_receipt_hash",
        )
        fit_receipt = split.get("fit_scope_receipt") or {}
        metric_receipt = split.get("metric_spec_receipt") or {}
        require(
            (fit_receipt.get("fit_scope_hashes") or {}).get("train_view_input")
            == split.get("public_tree_sha256"),
            "fit_receipt_public_tree_binding",
        )
        require(fit_receipt.get("holdout_fit_count") == 0, "holdout_fit_count")
        require(metric_receipt.get("metric") == expected_metric, "metric_receipt_name")
        require(
            metric_receipt.get("evaluator_module_sha256")
            == sha256_file(Path(__file__).with_name("evaluate.py")),
            "evaluator_module_hash",
        )

        hashes, symlinks, multi_link_files = _file_inventory(root)
        require(not symlinks, "no_symlinks")
        require(not multi_link_files, "no_hardlinks")
        require(
            not (_inode_set(root / "train_view") & _inode_set(root / "evaluator_view")),
            "train_evaluator_inode_isolation",
        )
        source_checks: dict[str, str] = {}
        for name, record in sorted((split.get("source_artifacts") or {}).items()):
            source = Path(str(record.get("path") or ""))
            declared = str(record.get("sha256") or "")
            require(len(declared) == 64, f"source_hash_declared:{name}")
            if verify_source_artifacts:
                require(source.is_file(), f"source_exists:{name}")
                if source.is_file():
                    observed = sha256_file(source)
                    source_checks[str(name)] = observed
                    require(observed == declared, f"source_hash:{name}")
                    if record.get("size_bytes") is not None:
                        require(
                            source.stat().st_size == int(record["size_bytes"]),
                            f"source_size:{name}",
                        )

        split_receipt = split.get("split_receipt") or {}
        details: dict[str, Any]
        if task_id == AERIAL_TASK:
            details = _verify_aerial(input_dir, labels, split_receipt, require)
        elif task_id == BIRDS_TASK:
            details = _verify_birds(root, input_dir, labels, split_receipt, require)
        elif task_id == TAXI_TASK:
            details = _verify_taxi(input_dir, labels, split_receipt, require)
        else:
            details = {}
        require(
            split_receipt.get("terminal_labels_absent_from_train_view") is True,
            "label_isolation_receipt",
        )
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "root": str(root),
            "task_id": task_id,
            "split_id": str(split.get("split_id") or ""),
            "split_manifest_sha256": sha256_file(split_path),
            "train_manifest_sha256": sha256_file(train_path),
            "evaluator_manifest_sha256": sha256_file(evaluator_path),
            "public_tree_sha256": str(split.get("public_tree_sha256") or ""),
            "labels_sha256": str(split.get("labels_sha256") or ""),
            "file_count": len(hashes),
            "file_inventory_sha256": hashlib.sha256(
                _canonical_json(hashes).encode("utf-8")
            ).hexdigest(),
            "source_artifact_hashes_recomputed": source_checks,
            "details": details,
            "checks": sorted(set(checks)),
            "errors": sorted(set(errors)),
            "valid": not errors,
            "report_hash": "",
        }
    except Exception as error:
        report = {
            "schema": REPORT_SCHEMA,
            "root": str(root),
            "task_id": "",
            "split_id": "",
            "checks": sorted(set(checks)),
            "errors": sorted(set([*errors, f"exception:{type(error).__name__}:{error}"])),
            "valid": False,
            "report_hash": "",
        }
    report["report_hash"] = _payload_hash(report, "report_hash")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-source-rehash", action="store_true")
    args = parser.parse_args()
    report = verify_formal_holdout(
        args.root,
        verify_source_artifacts=not args.skip_source_rehash,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"Verification report already exists: {args.output}")
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = ["REPORT_SCHEMA", "verify_formal_holdout"]
