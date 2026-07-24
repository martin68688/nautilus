"""Build the three preregistered WP8 Tier-2 formal holdouts.

The builders are host-owned and deterministic.  They materialize physically
separate label-free training and evaluator views, emit hash-bound split/fit/
metric receipts, and fail closed instead of silently repairing an infeasible
frozen allocation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from fixed_holdout.common import (
    SCHEMA,
    sha256_file,
    sha256_lines,
    tree_sha256,
    write_json,
)


AERIAL_TASK = "aerial-cactus-identification"
BIRDS_TASK = "mlsp-2013-birds"
TAXI_TASK = "new-york-city-taxi-fare-prediction"

AERIAL_SPLIT_VERSION = "wp8-tier2-formal-aerial-stratified-v1"
BIRDS_SPLIT_VERSION_R1 = "wp8-tier2-formal-mlsp-grouped-v1"
BIRDS_SPLIT_VERSION_R2 = "wp8-tier2-formal-mlsp-grouped-v2"
TAXI_SPLIT_VERSION = "wp8-tier2-formal-nyc-chronological-v1"

AERIAL_PROTOCOL_REF = (
    "random-classification@1#"
    "ecf870583bf524f66b11ea6b1e33829351c7c761c1a842d22338a95bd976c9cc"
)
BIRDS_PROTOCOL_REF = (
    "grouped-classification@1#"
    "901703060d3f2dd756cc29339a645583410a1d69ac4f2b6ccd53f8631b411382"
)
TAXI_PROTOCOL_REF = (
    "chronological-regression@1#"
    "bfc61957b422df5cf09dcb37cffe06aae2ccd2b11db4fee0721b90a2bc6dbf04"
)


class FormalSplitInfeasible(ValueError):
    """The frozen deterministic split cannot satisfy its preregistered checks."""


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


def _ordering_digest(split_version: str, value: object) -> str:
    return hashlib.sha256(
        f"{split_version}\0{value}".encode("utf-8")
    ).hexdigest()


def _assert_expected_hash(path: Path, expected: str | None) -> str:
    observed = sha256_file(path)
    if expected and observed != expected:
        raise ValueError(
            f"Source hash mismatch for {path}: expected {expected}, got {observed}"
        )
    return observed


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if os.stat(source).st_ino == os.stat(destination).st_ino:
        raise ValueError("Formal holdout materialization must not use hardlinks")


def _write_description(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _receipt(payload: dict[str, Any], schema: str) -> dict[str, Any]:
    value = {"schema": schema, **payload, "receipt_hash": ""}
    value["receipt_hash"] = _payload_hash(value, "receipt_hash")
    return value


def _finalize_views(
    staging_root: Path,
    *,
    task_id: str,
    split_id: str,
    split_version: str,
    protocol_ref: str,
    metric: str,
    maximize: bool,
    id_column: str,
    prediction_columns: list[str],
    holdout_ids: list[str],
    labels: pd.DataFrame,
    split_receipt_payload: dict[str, Any],
    source_artifacts: dict[str, dict[str, Any]],
    prediction_threshold: float | None = None,
    zero_division: int | None = None,
) -> dict[str, Any]:
    train_view = staging_root / "train_view"
    input_dir = train_view / "input"
    evaluator_view = staging_root / "evaluator_view"
    evaluator_view.mkdir(parents=True, exist_ok=True)
    if list(labels.columns) != [id_column, *prediction_columns]:
        raise ValueError("Evaluator labels do not match the declared prediction schema")
    if labels[id_column].astype(str).tolist() != holdout_ids:
        raise ValueError("Evaluator labels are not aligned to holdout ID order")
    label_path = evaluator_view / "labels.csv"
    labels.to_csv(label_path, index=False)
    public_digest = tree_sha256(input_dir)
    id_digest = sha256_lines(holdout_ids)

    split_receipt = _receipt(
        {
            "task_id": task_id,
            "split_id": split_id,
            "split_version": split_version,
            "protocol_ref": protocol_ref,
            **split_receipt_payload,
        },
        "formal_split_lineage_receipt_v1",
    )
    fit_scope_receipt = _receipt(
        {
            "task_id": task_id,
            "split_id": split_id,
            "protocol_ref": protocol_ref,
            "fit_scope": "train_view_only",
            "fit_scope_hashes": {"train_view_input": public_digest},
            "holdout_fit_count": 0,
            "verified": True,
        },
        "formal_fit_scope_receipt_v1",
    )
    metric_payload: dict[str, Any] = {
        "task_id": task_id,
        "split_id": split_id,
        "protocol_ref": protocol_ref,
        "metric": metric,
        "direction": "maximize" if maximize else "minimize",
        "evaluator_module_sha256": sha256_file(
            Path(__file__).with_name("evaluate.py")
        ),
        "terminal_only": True,
        "verified": True,
    }
    if prediction_threshold is not None:
        metric_payload["prediction_threshold"] = prediction_threshold
    if zero_division is not None:
        metric_payload["zero_division"] = zero_division
    metric_receipt = _receipt(
        metric_payload,
        "formal_metric_spec_receipt_v1",
    )

    common: dict[str, Any] = {
        "schema": SCHEMA,
        "formal_schema": "mlevolve_formal_fixed_holdout_v1",
        "task_id": task_id,
        "split_id": split_id,
        "split_version": split_version,
        "protocol_ref": protocol_ref,
        "metric": metric,
        "maximize": maximize,
        "id_column": id_column,
        "prediction_columns": prediction_columns,
        "normalize_probabilities": False,
        "row_count": len(holdout_ids),
        "holdout_id_sha256": id_digest,
        "public_tree_sha256": public_digest,
        "selection_policy": "terminal_only",
        "split_receipt": split_receipt,
        "fit_scope_receipt": fit_scope_receipt,
        "metric_spec_receipt": metric_receipt,
        "source_artifacts": source_artifacts,
    }
    if prediction_threshold is not None:
        common["prediction_threshold"] = prediction_threshold
    if zero_division is not None:
        common["zero_division"] = zero_division
    train_manifest = {
        **common,
        "role": "train_view",
        "input_subdir": "input",
        "copy_modes": ["copy"],
        "hardlink_count": 0,
        "hidden_labels_present": False,
        "final_metric_available_to_training": False,
    }
    evaluator_manifest = {
        **common,
        "role": "evaluator_view",
        "labels_file": "labels.csv",
        "labels_sha256": sha256_file(label_path),
    }
    write_json(train_view / "fixed_holdout_manifest.json", train_manifest)
    write_json(evaluator_view / "fixed_holdout_manifest.json", evaluator_manifest)
    split_manifest = {
        **common,
        "role": "split_root",
        "train_manifest": "train_view/fixed_holdout_manifest.json",
        "evaluator_manifest": "evaluator_view/fixed_holdout_manifest.json",
        "train_manifest_sha256": sha256_file(
            train_view / "fixed_holdout_manifest.json"
        ),
        "evaluator_manifest_sha256": sha256_file(
            evaluator_view / "fixed_holdout_manifest.json"
        ),
        "labels_sha256": sha256_file(label_path),
        "manifest_hash": "",
    }
    split_manifest["manifest_hash"] = _payload_hash(
        split_manifest, "manifest_hash"
    )
    write_json(staging_root / "split_manifest.json", split_manifest)
    return split_manifest


def _publish_staging(staging: Path, destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(f"Formal split destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.rename(destination)
    return destination


def build_aerial_holdout(
    source_zip: str | Path,
    output_root: str | Path,
    *,
    expected_source_sha256: str | None = None,
) -> Path:
    source_zip = Path(source_zip).resolve()
    output_root = Path(output_root).resolve()
    source_hash = _assert_expected_hash(source_zip, expected_source_sha256)
    with zipfile.ZipFile(source_zip) as outer:
        train = pd.read_csv(outer.open("train.csv"))
        if list(train.columns) != ["id", "has_cactus"]:
            raise ValueError("Unexpected Aerial Cactus training schema")
        if train["id"].isna().any() or train["id"].duplicated().any():
            raise ValueError("Aerial Cactus IDs must be complete and unique")
        if set(train["has_cactus"].unique()) != {0, 1}:
            raise ValueError("Aerial Cactus requires both binary classes")
        holdout_ids: list[str] = []
        for label, rows in train.groupby("has_cactus", sort=True):
            ordered = sorted(
                rows["id"].astype(str),
                key=lambda value: (
                    _ordering_digest(AERIAL_SPLIT_VERSION, value),
                    value,
                ),
            )
            count = max(1, math.floor(len(ordered) * 0.20))
            if count >= len(ordered):
                raise FormalSplitInfeasible(
                    f"Aerial class {label} cannot exist in both partitions"
                )
            holdout_ids.extend(ordered[:count])
        holdout_set = set(holdout_ids)
        train_rows = train[~train["id"].astype(str).isin(holdout_set)].copy()
        holdout_rows = train[train["id"].astype(str).isin(holdout_set)].copy()
        holdout_rows["_order"] = holdout_rows["id"].astype(str).map(
            {value: index for index, value in enumerate(holdout_ids)}
        )
        holdout_rows.sort_values("_order", inplace=True)
        holdout_rows.drop(columns=["_order"], inplace=True)
        holdout_ids = holdout_rows["id"].astype(str).tolist()

        split_id = f"{AERIAL_SPLIT_VERSION}-{source_hash[:12]}"
        destination = output_root / AERIAL_TASK / split_id
        staging_parent = output_root / AERIAL_TASK
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{split_id}.", dir=staging_parent)
        )
        try:
            input_dir = staging / "train_view" / "input"
            (input_dir / "train").mkdir(parents=True)
            (input_dir / "test").mkdir(parents=True)
            train_rows.to_csv(input_dir / "train.csv", index=False)
            pd.DataFrame({"id": holdout_ids}).to_csv(
                input_dir / "test.csv", index=False
            )
            pd.DataFrame(
                {"id": holdout_ids, "has_cactus": [0.0] * len(holdout_ids)}
            ).to_csv(input_dir / "sample_submission.csv", index=False)
            _write_description(
                input_dir / "description.md",
                [
                    "Binary image classification: predict has_cactus for test images.",
                    "Training labels are in train.csv and images in train/.",
                    "Terminal test images are in test/; their labels are sealed.",
                    "Write submission.csv with columns id,has_cactus using probabilities in [0,1].",
                    "The terminal metric is macro-F1 at threshold 0.5.",
                ],
            )
            inner_path = staging / ".train.zip"
            with outer.open("train.zip") as source, inner_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            with zipfile.ZipFile(inner_path) as images:
                entries = {
                    Path(name).name: name
                    for name in images.namelist()
                    if not name.endswith("/")
                }
                expected_ids = set(train["id"].astype(str))
                missing = sorted(expected_ids - set(entries))
                if missing:
                    raise ValueError(
                        f"Aerial training archive is missing {len(missing)} images"
                    )
                for sample_id in sorted(expected_ids):
                    partition = "test" if sample_id in holdout_set else "train"
                    target = input_dir / partition / sample_id
                    with images.open(entries[sample_id]) as source, target.open(
                        "wb"
                    ) as output:
                        shutil.copyfileobj(source, output)
            inner_path.unlink()
            train_counts = {
                str(key): int(value)
                for key, value in train_rows["has_cactus"].value_counts().items()
            }
            holdout_counts = {
                str(key): int(value)
                for key, value in holdout_rows["has_cactus"].value_counts().items()
            }
            _finalize_views(
                staging,
                task_id=AERIAL_TASK,
                split_id=split_id,
                split_version=AERIAL_SPLIT_VERSION,
                protocol_ref=AERIAL_PROTOCOL_REF,
                metric="macro_f1",
                maximize=True,
                id_column="id",
                prediction_columns=["has_cactus"],
                holdout_ids=holdout_ids,
                labels=holdout_rows[["id", "has_cactus"]],
                prediction_threshold=0.5,
                zero_division=0,
                source_artifacts={
                    "source_zip": {
                        "path": str(source_zip),
                        "sha256": source_hash,
                    }
                },
                split_receipt_payload={
                    "strategy": "stratified_random",
                    "train_count": len(train_rows),
                    "holdout_count": len(holdout_rows),
                    "train_class_counts": train_counts,
                    "holdout_class_counts": holdout_counts,
                    "overlap_count": 0,
                    "stratification_verified": set(train_counts)
                    == set(holdout_counts)
                    == {"0", "1"},
                    "terminal_labels_absent_from_train_view": True,
                },
            )
            return _publish_staging(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise


def _read_birds_sources(source_root: Path) -> dict[str, Any]:
    essential = source_root / "essential_data"
    paths = {
        "cvfolds": essential / "CVfolds_2.txt",
        "filename_map": essential / "rec_id2filename.txt",
        "labels": essential / "rec_labels_test_hidden.txt",
        "species": essential / "species_list.txt",
    }
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f"Missing Birds source artifact: {path}")
    folds = {
        int(row["rec_id"]): int(row["fold"])
        for row in csv.DictReader(paths["cvfolds"].open(encoding="utf-8"))
    }
    filenames = {
        int(row["rec_id"]): str(row["filename"])
        for row in csv.DictReader(
            paths["filename_map"].open(encoding="utf-8")
        )
    }
    labels: dict[int, set[int] | None] = {}
    with paths["labels"].open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            rec_id = int(row[0])
            labels[rec_id] = (
                None
                if row[1:] == ["?"]
                else {int(value) for value in row[1:] if value != ""}
            )
    species_rows = list(
        csv.DictReader(paths["species"].open(encoding="utf-8"))
    )
    class_ids = [int(row["class_id"]) for row in species_rows]
    if class_ids != list(range(19)):
        raise ValueError("Birds species list must contain class IDs 0..18")
    if set(folds) != set(filenames) or set(folds) != set(labels):
        raise ValueError("Birds record metadata sources do not have identical IDs")
    fold0 = [rec_id for rec_id in sorted(folds) if folds[rec_id] == 0]
    fold1 = [rec_id for rec_id in sorted(folds) if folds[rec_id] == 1]
    if any(labels[rec_id] is None for rec_id in fold0):
        raise ValueError("Birds fold0 unexpectedly contains hidden labels")
    if any(labels[rec_id] is not None for rec_id in fold1):
        raise ValueError("Birds fold1 unexpectedly exposes labels")
    return {
        "essential": essential,
        "paths": paths,
        "folds": folds,
        "filenames": filenames,
        "labels": labels,
        "class_ids": class_ids,
        "species_rows": species_rows,
        "fold0": fold0,
        "fold1": fold1,
    }


def _birds_group(filename: str) -> str:
    group = re.sub(r"_\d{4}$", "", str(filename))
    if not group or group == filename:
        raise ValueError(f"Birds filename lacks terminal segment ID: {filename}")
    return group


def _birds_group_counts(
    groups: Mapping[str, list[int]],
    labels: Mapping[int, set[int] | None],
    class_ids: list[int],
) -> dict[str, list[int]]:
    return {
        group: [
            sum(class_id in (labels[rec_id] or set()) for rec_id in rec_ids)
            for class_id in class_ids
        ]
        for group, rec_ids in groups.items()
    }


def _select_birds_r1(groups: Iterable[str]) -> list[str]:
    groups = list(groups)
    count = math.ceil(len(groups) * 0.25)
    return sorted(
        groups,
        key=lambda value: (
            _ordering_digest(BIRDS_SPLIT_VERSION_R1, value),
            value,
        ),
    )[:count]


def _select_birds_r2(
    groups: Mapping[str, list[int]],
    group_counts: Mapping[str, list[int]],
    class_ids: list[int],
) -> tuple[list[str], dict[str, Any]]:
    """Deterministic rare-first multilabel group stratification.

    Phase A guarantees at least one positive for every class on the holdout
    side without exhausting that class from train.  Phase B fills the frozen
    25% group budget by minimizing normalized distance to per-class targets.
    Hash order is the final tie-breaker only.
    """

    total = [
        sum(group_counts[group][index] for group in groups)
        for index, _class_id in enumerate(class_ids)
    ]
    if any(value < 2 for value in total):
        raise FormalSplitInfeasible(
            "Every Birds class needs at least two positives for two-sided coverage"
        )
    targets = [
        max(1, min(total[index] - 1, round(total[index] * 0.25)))
        for index, _class_id in enumerate(class_ids)
    ]
    target_group_count = math.ceil(len(groups) * 0.25)
    selected: list[str] = []
    counts = [0] * len(class_ids)

    def valid(group: str) -> bool:
        return all(
            total[index]
            - (counts[index] + group_counts[group][index])
            >= 1
            for index in range(len(class_ids))
        )

    def add(group: str) -> None:
        selected.append(group)
        for index in range(len(class_ids)):
            counts[index] += group_counts[group][index]

    while any(value == 0 for value in counts):
        uncovered = {
            index for index, value in enumerate(counts) if value == 0
        }
        candidates = [
            group
            for group in groups
            if group not in selected
            and valid(group)
            and any(group_counts[group][index] for index in uncovered)
        ]
        if not candidates:
            raise FormalSplitInfeasible(
                "Birds r2 cannot cover every class without emptying train"
            )

        def coverage_key(group: str) -> tuple[Any, ...]:
            prospective = [
                counts[index] + group_counts[group][index]
                for index in range(len(class_ids))
            ]
            newly_covered = sum(
                bool(group_counts[group][index]) for index in uncovered
            )
            overshoot = sum(
                max(0, prospective[index] - targets[index])
                / max(1, targets[index])
                for index in range(len(class_ids))
            )
            distance = sum(
                abs(prospective[index] - targets[index])
                / max(1, targets[index])
                for index in range(len(class_ids))
            )
            return (
                -newly_covered,
                overshoot,
                distance,
                _ordering_digest(BIRDS_SPLIT_VERSION_R2, group),
                group,
            )

        add(min(candidates, key=coverage_key))

    while len(selected) < target_group_count:
        candidates = [
            group
            for group in groups
            if group not in selected and valid(group)
        ]
        if not candidates:
            raise FormalSplitInfeasible(
                "Birds r2 cannot fill the declared group budget"
            )

        def balance_key(group: str) -> tuple[Any, ...]:
            prospective = [
                counts[index] + group_counts[group][index]
                for index in range(len(class_ids))
            ]
            distance = sum(
                abs(prospective[index] - targets[index])
                / max(1, targets[index])
                for index in range(len(class_ids))
            )
            overshoot = sum(
                max(0, prospective[index] - targets[index])
                / max(1, targets[index])
                for index in range(len(class_ids))
            )
            return (
                distance,
                overshoot,
                _ordering_digest(BIRDS_SPLIT_VERSION_R2, group),
                group,
            )

        add(min(candidates, key=balance_key))
    return selected, {
        "class_total_positive_counts": total,
        "class_target_holdout_positive_counts": targets,
        "coverage_phase_group_count": next(
            (
                index
                for index in range(1, len(selected) + 1)
                if all(
                    sum(
                        group_counts[group][class_index]
                        for group in selected[:index]
                    )
                    > 0
                    for class_index in range(len(class_ids))
                )
            ),
            len(selected),
        ),
    }


def birds_split_feasibility(
    source_root: str | Path,
    *,
    split_revision: str,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    source = _read_birds_sources(source_root)
    groups: dict[str, list[int]] = defaultdict(list)
    for rec_id in source["fold0"]:
        groups[_birds_group(source["filenames"][rec_id])].append(rec_id)
    group_counts = _birds_group_counts(
        groups,
        source["labels"],
        source["class_ids"],
    )
    if split_revision == "r1":
        selected = _select_birds_r1(groups)
        algorithm = "hash_only_group_allocation"
        diagnostics: dict[str, Any] = {}
        split_version = BIRDS_SPLIT_VERSION_R1
    elif split_revision == "r2":
        selected, diagnostics = _select_birds_r2(
            groups,
            group_counts,
            source["class_ids"],
        )
        algorithm = "deterministic_rare_first_multilabel_group_stratification"
        split_version = BIRDS_SPLIT_VERSION_R2
    else:
        raise ValueError("Birds split revision must be r1 or r2")
    selected_set = set(selected)
    train_ids = [
        rec_id
        for rec_id in source["fold0"]
        if _birds_group(source["filenames"][rec_id]) not in selected_set
    ]
    holdout_ids = [
        rec_id
        for rec_id in source["fold0"]
        if _birds_group(source["filenames"][rec_id]) in selected_set
    ]

    def coverage(rec_ids: list[int]) -> list[int]:
        return [
            sum(
                class_id in (source["labels"][rec_id] or set())
                for rec_id in rec_ids
            )
            for class_id in source["class_ids"]
        ]

    train_coverage = coverage(train_ids)
    holdout_coverage = coverage(holdout_ids)
    passed = bool(
        set(train_ids).isdisjoint(holdout_ids)
        and set(source["fold0"]) == set(train_ids) | set(holdout_ids)
        and all(train_coverage)
        and all(holdout_coverage)
        and not any(
            source["labels"][rec_id] is None for rec_id in source["fold0"]
        )
    )
    report = {
        "schema": "wp8_tier2_formal_birds_split_feasibility_v1",
        "split_revision": split_revision,
        "split_version": split_version,
        "algorithm": algorithm,
        "fold0_record_count": len(source["fold0"]),
        "fold1_excluded_record_count": len(source["fold1"]),
        "fold0_group_count": len(groups),
        "holdout_group_count": len(selected),
        "train_record_count": len(train_ids),
        "holdout_record_count": len(holdout_ids),
        "train_positive_counts": train_coverage,
        "holdout_positive_counts": holdout_coverage,
        "missing_train_class_ids": [
            class_id
            for class_id, count in zip(source["class_ids"], train_coverage)
            if count == 0
        ],
        "missing_holdout_class_ids": [
            class_id
            for class_id, count in zip(
                source["class_ids"], holdout_coverage
            )
            if count == 0
        ],
        "selected_group_ids_sha256": sha256_lines(sorted(selected)),
        "checks_passed": passed,
        "diagnostics": diagnostics,
        "report_hash": "",
    }
    report["report_hash"] = _payload_hash(report, "report_hash")
    return report


def build_birds_holdout(
    source_root: str | Path,
    output_root: str | Path,
    *,
    split_revision: str = "r2",
    expected_hashes: Mapping[str, str] | None = None,
) -> Path:
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    source = _read_birds_sources(source_root)
    expected_hashes = dict(expected_hashes or {})
    source_artifacts = {
        key: {
            "path": str(path),
            "sha256": _assert_expected_hash(path, expected_hashes.get(key)),
        }
        for key, path in source["paths"].items()
    }
    feasibility = birds_split_feasibility(
        source_root,
        split_revision=split_revision,
    )
    if not feasibility["checks_passed"]:
        raise FormalSplitInfeasible(
            f"Birds {split_revision} allocation fails label coverage: "
            f"{feasibility['missing_holdout_class_ids']}"
        )
    split_version = str(feasibility["split_version"])
    groups: dict[str, list[int]] = defaultdict(list)
    for rec_id in source["fold0"]:
        groups[_birds_group(source["filenames"][rec_id])].append(rec_id)
    group_counts = _birds_group_counts(
        groups,
        source["labels"],
        source["class_ids"],
    )
    if split_revision == "r1":
        selected_groups = _select_birds_r1(groups)
    else:
        selected_groups, _ = _select_birds_r2(
            groups,
            group_counts,
            source["class_ids"],
        )
    holdout_groups = set(selected_groups)
    train_ids = [
        rec_id
        for rec_id in source["fold0"]
        if _birds_group(source["filenames"][rec_id]) not in holdout_groups
    ]
    holdout_rec_ids = [
        rec_id
        for rec_id in source["fold0"]
        if _birds_group(source["filenames"][rec_id]) in holdout_groups
    ]
    holdout_rec_ids.sort()
    class_columns = [str(value) for value in source["class_ids"]]

    def frame(rec_ids: list[int], *, include_labels: bool) -> pd.DataFrame:
        rows = []
        for rec_id in rec_ids:
            filename = source["filenames"][rec_id]
            row: dict[str, Any] = {
                "rec_id": rec_id,
                "filename": f"{filename}.wav",
                "recording_session_id": _birds_group(filename),
            }
            if include_labels:
                row.update(
                    {
                        str(class_id): int(
                            class_id in (source["labels"][rec_id] or set())
                        )
                        for class_id in source["class_ids"]
                    }
                )
            rows.append(row)
        return pd.DataFrame(rows)

    train_frame = frame(train_ids, include_labels=True)
    test_frame = frame(holdout_rec_ids, include_labels=False)
    labels = frame(holdout_rec_ids, include_labels=True)[
        ["rec_id", *class_columns]
    ]
    split_id = f"{split_version}-{feasibility['selected_group_ids_sha256'][:12]}"
    destination = output_root / BIRDS_TASK / split_id
    parent = output_root / BIRDS_TASK
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{split_id}.", dir=parent))
    try:
        input_dir = staging / "train_view" / "input"
        train_wavs = input_dir / "train_wavs"
        test_wavs = input_dir / "test_wavs"
        train_wavs.mkdir(parents=True)
        test_wavs.mkdir(parents=True)
        train_frame.to_csv(input_dir / "train.csv", index=False)
        test_frame.to_csv(input_dir / "test.csv", index=False)
        sample = pd.DataFrame({"rec_id": holdout_rec_ids})
        for column in class_columns:
            sample[column] = 0.0
        sample.to_csv(input_dir / "sample_submission.csv", index=False)
        _write_description(
            input_dir / "description.md",
            [
                "Multilabel audio classification over 19 bird species.",
                "train.csv has rec_id, filename, recording_session_id and binary class columns 0..18.",
                "Train and terminal groups are physically disjoint in train_wavs/ and test_wavs/.",
                "Write submission.csv with rec_id followed by probability columns 0..18.",
                "The terminal metric is macro-F1 over 19 labels at threshold 0.5.",
            ],
        )
        wav_root = source["essential"] / "src_wavs"
        for rec_id in train_ids:
            filename = f"{source['filenames'][rec_id]}.wav"
            _copy_file(wav_root / filename, train_wavs / filename)
        for rec_id in holdout_rec_ids:
            filename = f"{source['filenames'][rec_id]}.wav"
            _copy_file(wav_root / filename, test_wavs / filename)
        train_group_set = {
            _birds_group(source["filenames"][rec_id]) for rec_id in train_ids
        }
        _finalize_views(
            staging,
            task_id=BIRDS_TASK,
            split_id=split_id,
            split_version=split_version,
            protocol_ref=BIRDS_PROTOCOL_REF,
            metric="macro_f1",
            maximize=True,
            id_column="rec_id",
            prediction_columns=class_columns,
            holdout_ids=[str(value) for value in holdout_rec_ids],
            labels=labels,
            prediction_threshold=0.5,
            zero_division=0,
            source_artifacts=source_artifacts,
            split_receipt_payload={
                "strategy": "grouped_multilabel_stratified",
                "split_revision": split_revision,
                "train_count": len(train_ids),
                "holdout_count": len(holdout_rec_ids),
                "train_group_count": len(train_group_set),
                "holdout_group_count": len(holdout_groups),
                "record_overlap_count": 0,
                "group_overlap_count": len(train_group_set & holdout_groups),
                "train_positive_counts": feasibility[
                    "train_positive_counts"
                ],
                "holdout_positive_counts": feasibility[
                    "holdout_positive_counts"
                ],
                "all_19_species_two_sided": True,
                "fold1_record_count_in_views": 0,
                "terminal_labels_absent_from_train_view": True,
                "feasibility_report_hash": feasibility["report_hash"],
            },
        )
        write_json(staging / "birds_feasibility_report.json", feasibility)
        return _publish_staging(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _taxi_row_is_valid(row: Mapping[str, str]) -> bool:
    try:
        fare = float(row["fare_amount"])
        passenger_count = float(row["passenger_count"])
        pickup_longitude = float(row["pickup_longitude"])
        pickup_latitude = float(row["pickup_latitude"])
        dropoff_longitude = float(row["dropoff_longitude"])
        dropoff_latitude = float(row["dropoff_latitude"])
    except (KeyError, TypeError, ValueError):
        return False
    values = (
        fare,
        passenger_count,
        pickup_longitude,
        pickup_latitude,
        dropoff_longitude,
        dropoff_latitude,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    return bool(
        2.5 <= fare <= 200.0
        and 1.0 <= passenger_count <= 6.0
        and -75.0 <= pickup_longitude <= -72.0
        and -75.0 <= dropoff_longitude <= -72.0
        and 40.0 <= pickup_latitude <= 42.0
        and 40.0 <= dropoff_latitude <= 42.0
    )


def _taxi_sampled_rows(labels_csv: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    required = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
    ]
    sampled: list[dict[str, str]] = []
    input_count = 0
    valid_count = 0
    time_window_count = 0
    with labels_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != required:
            raise ValueError(
                f"Unexpected NYC Taxi labels schema: {reader.fieldnames}"
            )
        for row in reader:
            input_count += 1
            if not _taxi_row_is_valid(row):
                continue
            valid_count += 1
            timestamp = str(row["pickup_datetime"])
            if not re.match(
                r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",
                timestamp,
            ):
                continue
            timestamp_prefix = timestamp.strip()[:19].replace(" ", "T") + "Z"
            if not (
                "2009-01-01T00:00:00Z"
                <= timestamp_prefix
                < "2015-07-01T00:00:00Z"
            ):
                continue
            time_window_count += 1
            digest = hashlib.sha256(
                f"{TAXI_SPLIT_VERSION}\0{row['key']}".encode("utf-8")
            ).digest()
            if int.from_bytes(digest[:8], "big", signed=False) % 256 != 0:
                continue
            sampled.append(dict(row))
    return sampled, {
        "source_row_count": input_count,
        "row_filter_valid_count": valid_count,
        "time_window_valid_count": time_window_count,
        "sampled_count": len(sampled),
    }


def _canonical_utc_timestamp(value: str) -> str:
    text = str(value).strip()
    parsed = pd.to_datetime(text, utc=True, errors="raise")
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_taxi_holdout(
    labels_csv: str | Path,
    output_root: str | Path,
    *,
    expected_labels_sha256: str | None = None,
    minimum_train_rows: int = 50_000,
    minimum_holdout_rows: int = 10_000,
) -> Path:
    labels_csv = Path(labels_csv).resolve()
    output_root = Path(output_root).resolve()
    labels_hash = _assert_expected_hash(labels_csv, expected_labels_sha256)
    sampled, sampling_diagnostics = _taxi_sampled_rows(labels_csv)
    keys = [str(row["key"]) for row in sampled]
    if len(keys) != len(set(keys)):
        raise FormalSplitInfeasible(
            "NYC Taxi sampled keys are not unique after filtering"
        )
    for row in sampled:
        row["pickup_datetime"] = _canonical_utc_timestamp(
            row["pickup_datetime"]
        )
    sampled.sort(key=lambda row: (row["pickup_datetime"], row["key"]))
    cutoff = "2014-07-01T00:00:00Z"
    train_rows = [row for row in sampled if row["pickup_datetime"] < cutoff]
    holdout_rows = [
        row for row in sampled if row["pickup_datetime"] >= cutoff
    ]
    if len(train_rows) < minimum_train_rows:
        raise FormalSplitInfeasible(
            f"NYC Taxi train sample has {len(train_rows)} rows; "
            f"requires {minimum_train_rows}"
        )
    if len(holdout_rows) < minimum_holdout_rows:
        raise FormalSplitInfeasible(
            f"NYC Taxi holdout sample has {len(holdout_rows)} rows; "
            f"requires {minimum_holdout_rows}"
        )
    max_train_time = max(row["pickup_datetime"] for row in train_rows)
    min_holdout_time = min(row["pickup_datetime"] for row in holdout_rows)
    if not max_train_time < min_holdout_time:
        raise FormalSplitInfeasible(
            "NYC Taxi chronological boundary is not strictly ordered"
        )
    train_keys = {row["key"] for row in train_rows}
    holdout_keys = {row["key"] for row in holdout_rows}
    if train_keys & holdout_keys:
        raise FormalSplitInfeasible("NYC Taxi partitions overlap")

    split_id = f"{TAXI_SPLIT_VERSION}-{labels_hash[:12]}"
    destination = output_root / TAXI_TASK / split_id
    parent = output_root / TAXI_TASK
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{split_id}.", dir=parent))
    try:
        input_dir = staging / "train_view" / "input"
        input_dir.mkdir(parents=True)
        columns = [
            "key",
            "fare_amount",
            "pickup_datetime",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
        ]
        train_frame = pd.DataFrame(train_rows, columns=columns)
        holdout_frame = pd.DataFrame(holdout_rows, columns=columns)
        train_frame.to_csv(input_dir / "train.csv", index=False)
        test_frame = holdout_frame.drop(columns=["fare_amount"])
        test_frame.to_csv(input_dir / "test.csv", index=False)
        pd.DataFrame(
            {
                "key": holdout_frame["key"].astype(str),
                "fare_amount": [0.0] * len(holdout_frame),
            }
        ).to_csv(input_dir / "sample_submission.csv", index=False)
        _write_description(
            input_dir / "description.md",
            [
                "Chronological tabular regression for NYC taxi fare_amount.",
                "train.csv contains sampled trips strictly before 2014-07-01 UTC.",
                "test.csv contains sampled future trips without fare_amount.",
                "Write submission.csv with columns key,fare_amount.",
                "The sealed terminal metric is RMSE; future labels are never available during search.",
            ],
        )
        evaluator_labels = holdout_frame[["key", "fare_amount"]].copy()
        holdout_ids = holdout_frame["key"].astype(str).tolist()
        _finalize_views(
            staging,
            task_id=TAXI_TASK,
            split_id=split_id,
            split_version=TAXI_SPLIT_VERSION,
            protocol_ref=TAXI_PROTOCOL_REF,
            metric="rmse",
            maximize=False,
            id_column="key",
            prediction_columns=["fare_amount"],
            holdout_ids=holdout_ids,
            labels=evaluator_labels,
            source_artifacts={
                "labels_csv": {
                    "path": str(labels_csv),
                    "sha256": labels_hash,
                    "size_bytes": labels_csv.stat().st_size,
                }
            },
            split_receipt_payload={
                "strategy": "chronological_deterministic_sha256_sample",
                **sampling_diagnostics,
                "train_count": len(train_rows),
                "holdout_count": len(holdout_rows),
                "train_holdout_key_overlap_count": 0,
                "max_train_pickup_datetime": max_train_time,
                "min_holdout_pickup_datetime": min_holdout_time,
                "future_to_past_count": 0,
                "sampling_modulus": 256,
                "sampling_remainder": 0,
                "terminal_cutoff": cutoff,
                "terminal_labels_absent_from_train_view": True,
            },
        )
        return _publish_staging(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    aerial = subparsers.add_parser("aerial")
    aerial.add_argument("--source-zip", type=Path, required=True)
    aerial.add_argument("--output-root", type=Path, required=True)
    aerial.add_argument("--expected-source-sha256")

    birds = subparsers.add_parser("birds")
    birds.add_argument("--source-root", type=Path, required=True)
    birds.add_argument("--output-root", type=Path, required=True)
    birds.add_argument("--split-revision", choices=["r1", "r2"], default="r2")
    birds.add_argument("--expected-cvfolds-sha256")
    birds.add_argument("--expected-filename-map-sha256")
    birds.add_argument("--expected-labels-sha256")
    birds.add_argument("--expected-species-sha256")

    feasibility = subparsers.add_parser("birds-feasibility")
    feasibility.add_argument("--source-root", type=Path, required=True)
    feasibility.add_argument("--split-revision", choices=["r1", "r2"], required=True)
    feasibility.add_argument("--output", type=Path, required=True)

    taxi = subparsers.add_parser("taxi")
    taxi.add_argument("--labels-csv", type=Path, required=True)
    taxi.add_argument("--output-root", type=Path, required=True)
    taxi.add_argument("--expected-labels-sha256")
    taxi.add_argument("--minimum-train-rows", type=int, default=50_000)
    taxi.add_argument("--minimum-holdout-rows", type=int, default=10_000)
    args = parser.parse_args()

    if args.command == "aerial":
        result = build_aerial_holdout(
            args.source_zip,
            args.output_root,
            expected_source_sha256=args.expected_source_sha256,
        )
    elif args.command == "birds":
        result = build_birds_holdout(
            args.source_root,
            args.output_root,
            split_revision=args.split_revision,
            expected_hashes={
                "cvfolds": args.expected_cvfolds_sha256,
                "filename_map": args.expected_filename_map_sha256,
                "labels": args.expected_labels_sha256,
                "species": args.expected_species_sha256,
            },
        )
    elif args.command == "birds-feasibility":
        report = birds_split_feasibility(
            args.source_root,
            split_revision=args.split_revision,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)
        print(args.output)
        return
    else:
        result = build_taxi_holdout(
            args.labels_csv,
            args.output_root,
            expected_labels_sha256=args.expected_labels_sha256,
            minimum_train_rows=args.minimum_train_rows,
            minimum_holdout_rows=args.minimum_holdout_rows,
        )
    print(result)


if __name__ == "__main__":
    main()


__all__ = [
    "FormalSplitInfeasible",
    "birds_split_feasibility",
    "build_aerial_holdout",
    "build_birds_holdout",
    "build_taxi_holdout",
]
