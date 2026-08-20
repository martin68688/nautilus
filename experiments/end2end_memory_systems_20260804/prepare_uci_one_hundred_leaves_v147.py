#!/usr/bin/env python3
"""Prepare a deterministic target-task split for the v147 transfer smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


FEATURE_FILES = (
    ("shape", "data_Sha_64.txt"),
    ("margin", "data_Mar_64.txt"),
    ("texture", "data_Tex_64.txt"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_feature(path: Path) -> tuple[list[str], list[list[float]]]:
    labels: list[str] = []
    features: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 65:
                raise ValueError(f"{path} row has {len(row)} fields, expected 65")
            labels.append(row[0].strip())
            features.append([float(value) for value in row[1:]])
    return labels, features


def image_index(source: Path) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for class_dir in sorted((source / "data").iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.replace("_", " ").lower()
        images = sorted(class_dir.glob("*.jpg"), key=lambda path: path.name.lower())
        if len(images) != 16:
            raise ValueError(f"{class_dir} has {len(images)} images, expected 16")
        for specimen, path in enumerate(images, 1):
            result[(label, specimen)] = path
    return result


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def keyed_feature_rows(
    labels: list[str], features: list[list[float]]
) -> tuple[list[tuple[str, int]], dict[tuple[str, int], list[float]]]:
    if len(labels) != len(features):
        raise ValueError("Feature labels and vectors differ in length")
    occurrences: defaultdict[str, int] = defaultdict(int)
    order: list[tuple[str, int]] = []
    keyed: dict[tuple[str, int], list[float]] = {}
    for label, values in zip(labels, features):
        occurrences[label] += 1
        key = (label, occurrences[label])
        order.append(key)
        keyed[key] = values
    return order, keyed


def prepare(source: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Refusing to reuse target-task release: {output}")

    loaded = [read_feature(source / filename) for _, filename in FEATURE_FILES]
    keyed_views = [keyed_feature_rows(*item) for item in loaded]
    common_keys = set.intersection(*(set(view[1]) for view in keyed_views))
    ordered_keys = [key for key in keyed_views[0][0] if key in common_keys]
    # The published UCI archive has one documented-by-content discrepancy:
    # data_Tex_64.txt contains 15 Acer Campestre vectors while Shape/Margin
    # and the image tree contain 16.  Fail closed for any broader mismatch and
    # drop only that unpaired sample from every view.
    dropped = sorted(set(keyed_views[0][0]) - common_keys)
    if len(ordered_keys) != 1599 or dropped != [("Acer Campestre", 16)]:
        raise ValueError(
            f"Unexpected cross-view sample mismatch: kept={len(ordered_keys)} dropped={dropped}"
        )
    labels = [key[0] for key in ordered_keys]
    counts = Counter(labels)
    if (
        len(counts) != 100
        or sorted(counts.values()).count(15) != 1
        or set(counts.values()) != {15, 16}
    ):
        raise ValueError(f"Unexpected class distribution: {counts}")

    images = image_index(source)
    output.mkdir(parents=True)
    (output / "images").mkdir()
    rows: list[dict[str, object]] = []
    for label, specimen in ordered_keys:
        normalized_label = label.lower()
        source_image = images[(normalized_label, specimen)]
        sample_id = f"{label.replace(' ', '_')}__{specimen:02d}"
        image_name = f"{sample_id}.jpg"
        shutil.copyfile(source_image, output / "images" / image_name)
        row: dict[str, object] = {
            "id": sample_id,
            "species": label,
            "image": f"images/{image_name}",
        }
        for view_index, (prefix, _) in enumerate(FEATURE_FILES):
            for feature_index, value in enumerate(
                keyed_views[view_index][1][(label, specimen)], 1
            ):
                row[f"{prefix}_{feature_index:02d}"] = value
        rows.append(row)

    # Every class contributes its last four available samples to target test;
    # the UCI source discrepancy above therefore yields 1199 train / 400 test.
    # No random state or library version can change this split.
    train = [
        row
        for row in rows
        if int(str(row["id"]).rsplit("__", 1)[1])
        <= counts[str(row["species"])] - 4
    ]
    test_labeled = [row for row in rows if row not in train]
    test = [{key: value for key, value in row.items() if key != "species"} for row in test_labeled]
    feature_columns = [
        f"{prefix}_{index:02d}"
        for prefix, _ in FEATURE_FILES
        for index in range(1, 65)
    ]
    write_csv(
        output / "train.csv",
        ["id", "species", "image", *feature_columns],
        train,
    )
    write_csv(
        output / "test.csv",
        ["id", "image", *feature_columns],
        test,
    )
    classes = sorted(counts)
    sample_rows = [
        {"id": row["id"], **{label: 1.0 / len(classes) for label in classes}}
        for row in test
    ]
    write_csv(output / "sample_submission.csv", ["id", *classes], sample_rows)
    write_csv(
        output / "target_labels_host_only.csv",
        ["id", "species"],
        [{"id": row["id"], "species": row["species"]} for row in test_labeled],
    )
    (output / "description.md").write_text(
        "\n".join(
            [
                "# UCI One Hundred Plant Species Leaves — transfer target",
                "",
                "Classify 400 target rows into 100 leaf species using 1200 labeled training rows.",
                "Each row has three 64-dimensional views: shape, margin, and texture, plus a local binary leaf image.",
                "The target metric is multiclass log loss (lower is better).",
                "Use only train.csv for fitting, validation, calibration, feature selection, and model selection.",
                "test.csv has no labels. Write submission.csv with id followed by the exact 100 class columns from sample_submission.csv.",
                "All assets are local; do not download models or data at runtime.",
                "This task is distinct from Kaggle leaf-classification. Source-task scores, predictions, checkpoints, class mappings, and code are not valid target evidence.",
                "Dataset source: UCI One-hundred plant species leaves data set, DOI 10.24432/C5RG76, CC BY 4.0.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    receipt = {
        "schema": "uci_one_hundred_leaves_transfer_target_v1",
        "task_id": "uci-one-hundred-leaves",
        "source_archive_sha256": "2313a70de450a8a6b81696174f52be1c037090af53b37c6a6313f11245e5fd4c",
        "train_rows": len(train),
        "test_rows": len(test),
        "class_count": len(classes),
        "feature_count": len(feature_columns),
        "image_count": len(list((output / "images").glob("*.jpg"))),
        "files": {
            name: sha256_file(output / name)
            for name in (
                "train.csv",
                "test.csv",
                "sample_submission.csv",
                "target_labels_host_only.csv",
                "description.md",
            )
        },
    }
    (output / "DATASET_RECEIPT.json").write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
