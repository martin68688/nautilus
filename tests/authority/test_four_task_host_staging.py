from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from protocol_runtime.task_staging import TASK_SPECS, stage_task


def test_each_formal_task_binds_the_registered_metric_and_direction() -> None:
    expected = {
        "aerial-cactus-identification": ("roc_auc", "maximize"),
        "leaf-classification": ("log_loss", "minimize"),
        "denoising-dirty-documents": ("rmse", "minimize"),
        "new-york-city-taxi-fare-prediction": ("rmse", "minimize"),
        "spooky-author-identification": ("log_loss", "minimize"),
    }
    assert {
        task: (spec["metric_name"], spec["metric_direction"])
        for task, spec in TASK_SPECS.items()
    } == expected


def _csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aerial(root: Path) -> None:
    images = root / "images"
    images.mkdir(parents=True)
    train = []
    test = []
    for label in (0, 1):
        for index in range(3):
            name = f"train-{label}-{index}.jpg"
            (images / name).write_bytes(name.encode())
            train.append({"id": name, "has_cactus": label})
    for index in range(2):
        name = f"test-{index}.jpg"
        (images / name).write_bytes(name.encode())
        test.append({"id": name, "has_cactus": 0})
    _csv(root / "train.csv", ["id", "has_cactus"], train)
    _csv(root / "sample_submission.csv", ["id", "has_cactus"], test)


def _aerial_zip(root: Path) -> None:
    train = []
    test = []
    with zipfile.ZipFile(root / "train.zip", "w") as archive:
        for label in (0, 1):
            for index in range(3):
                name = f"train-{label}-{index}.jpg"
                archive.writestr(f"train/{name}", name.encode())
                train.append({"id": name, "has_cactus": label})
    with zipfile.ZipFile(root / "test.zip", "w") as archive:
        for index in range(2):
            name = f"test-{index}.jpg"
            archive.writestr(f"test/{name}", name.encode())
            test.append({"id": name, "has_cactus": 0})
    _csv(root / "train.csv", ["id", "has_cactus"], train)
    _csv(root / "sample_submission.csv", ["id", "has_cactus"], test)


def _leaf(root: Path) -> None:
    _csv(
        root / "train.csv",
        ["id", "species", "f1", "f2"],
        [
            {"id": index, "species": f"species-{index % 2}", "f1": index, "f2": index + 1}
            for index in range(6)
        ],
    )
    _csv(
        root / "test.csv",
        ["id", "f1", "f2"],
        [{"id": 10 + index, "f1": index, "f2": index + 1} for index in range(2)],
    )


def _taxi(root: Path) -> None:
    fields = ["key", "fare_amount", "pickup_datetime", "pickup_longitude"]
    _csv(
        root / "train.csv",
        fields,
        [
            {
                "key": f"train-{index}",
                "fare_amount": 5 + index,
                "pickup_datetime": f"2026-01-{index + 1:02d} 00:00:00 UTC",
                "pickup_longitude": -73.0,
            }
            for index in range(6)
        ],
    )
    _csv(
        root / "test.csv",
        ["key", "pickup_datetime", "pickup_longitude"],
        [
            {
                "key": f"test-{index}",
                "pickup_datetime": f"2026-02-{index + 1:02d} 00:00:00 UTC",
                "pickup_longitude": -73.0,
            }
            for index in range(2)
        ],
    )


def _denoising(root: Path) -> None:
    for directory in ("train", "train_cleaned", "test"):
        (root / directory).mkdir(parents=True)
    for index in range(6):
        name = f"train-{index}.png"
        (root / "train" / name).write_bytes(f"dirty-{index}".encode())
        (root / "train_cleaned" / name).write_bytes(f"clean-{index}".encode())
    for index in range(2):
        (root / "test" / f"test-{index}.png").write_bytes(b"dirty-test")


def _spooky(root: Path) -> None:
    _csv(
        root / "train.csv",
        ["id", "text", "author"],
        [
            {"id": f"train-{index}", "text": f"text {index}", "author": "EAP" if index % 2 else "HPL"}
            for index in range(6)
        ],
    )
    _csv(
        root / "test.csv",
        ["id", "text"],
        [{"id": f"test-{index}", "text": f"test text {index}"} for index in range(2)],
    )


@pytest.mark.parametrize(
    ("task_id", "builder"),
    [
        ("aerial-cactus-identification", _aerial),
        ("leaf-classification", _leaf),
        ("denoising-dirty-documents", _denoising),
        ("new-york-city-taxi-fare-prediction", _taxi),
        ("spooky-author-identification", _spooky),
    ],
)
def test_four_task_stagers_freeze_normalized_records(
    tmp_path: Path, task_id: str, builder
) -> None:
    public = tmp_path / "public"
    public.mkdir()
    builder(public)
    payload = stage_task(task_id, public, tmp_path / "staging")
    assert payload["schema"] == "mlevolve_four_task_host_staging_v1"
    assert payload["protocol_ref"] == TASK_SPECS[task_id]["protocol_ref"]
    assert payload["train_record_count"] >= 2
    assert payload["inference_record_count"] >= 1
    assert payload["terminal_labels_in_staging"] is False
    manifest = json.loads(
        (tmp_path / "staging" / "TASK_STAGING_MANIFEST.json").read_text()
    )
    assert manifest["staging_hash"] == payload["staging_hash"]


def test_aerial_zip_assets_are_safely_extracted_and_bound(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    _aerial_zip(public)
    payload = stage_task(
        "aerial-cactus-identification", public, tmp_path / "staging"
    )
    assert payload["train_record_count"] == 6
    records = [
        json.loads(line)
        for line in Path(payload["train_records_path"]).read_text().splitlines()
    ]
    assert all(Path(row["_host_assets"]["image"]).is_file() for row in records)
    assert all("AERIAL_ASSETS/train" in row["_host_assets"]["image"] for row in records)


def test_taxi_accepts_mlebench_labels_csv_alias(tmp_path: Path) -> None:
    public = tmp_path / "public"
    public.mkdir()
    _taxi(public)
    (public / "train.csv").rename(public / "labels.csv")
    payload = stage_task(
        "new-york-city-taxi-fare-prediction", public, tmp_path / "staging"
    )
    assert payload["training_source_name"] == "labels.csv"
    assert payload["training_records_truncated"] is False
