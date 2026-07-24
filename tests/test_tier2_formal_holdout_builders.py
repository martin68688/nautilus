from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from fixed_holdout.formal_prepare import (
    AERIAL_SPLIT_VERSION,
    BIRDS_SPLIT_VERSION_R1,
    FormalSplitInfeasible,
    birds_split_feasibility,
    build_aerial_holdout,
    build_birds_holdout,
    build_taxi_holdout,
)
from fixed_holdout.validation import validate_train_view


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aerial_source(tmp_path: Path) -> Path:
    source = tmp_path / "aerial.zip"
    inner = tmp_path / "train.zip"
    rows = []
    with zipfile.ZipFile(inner, "w") as images:
        for label in (0, 1):
            for index in range(5):
                sample_id = f"class-{label}-{index}.jpg"
                rows.append({"id": sample_id, "has_cactus": label})
                images.writestr(f"train/{sample_id}", f"image-{sample_id}".encode())
    with zipfile.ZipFile(source, "w") as outer:
        outer.writestr("train.csv", pd.DataFrame(rows).to_csv(index=False))
        outer.write(inner, "train.zip")
    return source


def _birds_source(tmp_path: Path) -> Path:
    root = tmp_path / "birds"
    essential = root / "essential_data"
    wavs = essential / "src_wavs"
    wavs.mkdir(parents=True)
    groups = [f"PC{index:02d}_20100101_050000" for index in range(8)]
    selected_r1 = set(
        sorted(
            groups,
            key=lambda value: (
                hashlib.sha256(
                    f"{BIRDS_SPLIT_VERSION_R1}\0{value}".encode()
                ).hexdigest(),
                value,
            ),
        )[:2]
    )
    rare_groups = [group for group in groups if group not in selected_r1][:2]
    with (essential / "CVfolds_2.txt").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rec_id", "fold"])
        for rec_id in range(8):
            writer.writerow([rec_id, 0])
        writer.writerow([8, 1])
    with (essential / "rec_id2filename.txt").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["rec_id", "filename"])
        for rec_id, group in enumerate(groups):
            filename = f"{group}_0010"
            writer.writerow([rec_id, filename])
            (wavs / f"{filename}.wav").write_bytes(f"wav-{rec_id}".encode())
        writer.writerow([8, "PC99_20100101_050000_0010"])
        (wavs / "PC99_20100101_050000_0010.wav").write_bytes(b"hidden")
    with (essential / "rec_labels_test_hidden.txt").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["rec_id", "[labels]"])
        for rec_id, group in enumerate(groups):
            labels = list(range(18))
            if group in rare_groups:
                labels.append(18)
            writer.writerow([rec_id, *labels])
        writer.writerow([8, "?"])
    with (essential / "species_list.txt").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "code", "species"])
        for class_id in range(19):
            writer.writerow([class_id, f"C{class_id}", f"Species {class_id}"])
    return root


def _sampled_taxi_key(prefix: str, index: int) -> str:
    candidate = 0
    while True:
        value = f"{prefix}-{index}-{candidate}"
        digest = hashlib.sha256(
            f"wp8-tier2-formal-nyc-chronological-v1\0{value}".encode()
        ).digest()
        if int.from_bytes(digest[:8], "big") % 256 == 0:
            return value
        candidate += 1


def _taxi_source(tmp_path: Path) -> Path:
    path = tmp_path / "labels.csv"
    rows = []
    for index in range(4):
        rows.append(
            {
                "key": _sampled_taxi_key("train", index),
                "fare_amount": 10.0 + index,
                "pickup_datetime": f"2013-01-{index + 1:02d} 10:00:00 UTC",
                "pickup_longitude": -73.9,
                "pickup_latitude": 40.7,
                "dropoff_longitude": -73.8,
                "dropoff_latitude": 40.8,
                "passenger_count": 1,
            }
        )
    for index in range(3):
        rows.append(
            {
                "key": _sampled_taxi_key("holdout", index),
                "fare_amount": 20.0 + index,
                "pickup_datetime": f"2014-08-{index + 1:02d} 10:00:00 UTC",
                "pickup_longitude": -73.9,
                "pickup_latitude": 40.7,
                "dropoff_longitude": -73.8,
                "dropoff_latitude": 40.8,
                "passenger_count": 2,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_aerial_builder_creates_stratified_label_isolated_views(
    tmp_path: Path,
) -> None:
    source = _aerial_source(tmp_path)
    root = build_aerial_holdout(source, tmp_path / "formal")
    train_view = root / "train_view"
    manifest = validate_train_view(
        train_view / "fixed_holdout_manifest.json",
        train_view / "input",
    )
    assert manifest["split_version"] == AERIAL_SPLIT_VERSION
    assert manifest["metric"] == "macro_f1"
    assert manifest["prediction_threshold"] == 0.5
    assert manifest["split_receipt"]["strategy"] == "stratified_random"
    assert manifest["split_receipt"]["overlap_count"] == 0
    train = pd.read_csv(train_view / "input" / "train.csv")
    test = pd.read_csv(train_view / "input" / "test.csv")
    assert set(train["id"]).isdisjoint(test["id"])
    assert set(train["has_cactus"]) == {0, 1}
    assert len(list((train_view / "input" / "train").iterdir())) == len(train)
    assert len(list((train_view / "input" / "test").iterdir())) == len(test)
    assert not any(
        path.name == "labels.csv" for path in train_view.rglob("*")
    )


def test_birds_r1_failure_is_preserved_and_r2_is_group_disjoint(
    tmp_path: Path,
) -> None:
    source = _birds_source(tmp_path)
    r1 = birds_split_feasibility(source, split_revision="r1")
    assert r1["checks_passed"] is False
    assert r1["missing_holdout_class_ids"] == [18]
    with pytest.raises(FormalSplitInfeasible, match="fails label coverage"):
        build_birds_holdout(
            source,
            tmp_path / "formal-r1",
            split_revision="r1",
        )

    root = build_birds_holdout(
        source,
        tmp_path / "formal-r2",
        split_revision="r2",
    )
    train_view = root / "train_view"
    manifest = validate_train_view(
        train_view / "fixed_holdout_manifest.json",
        train_view / "input",
    )
    receipt = manifest["split_receipt"]
    assert receipt["group_overlap_count"] == 0
    assert receipt["all_19_species_two_sided"] is True
    assert receipt["fold1_record_count_in_views"] == 0
    train = pd.read_csv(train_view / "input" / "train.csv")
    test = pd.read_csv(train_view / "input" / "test.csv")
    assert set(train["recording_session_id"]).isdisjoint(
        test["recording_session_id"]
    )
    assert "18" in train.columns
    assert "18" not in test.columns
    assert all(receipt["train_positive_counts"])
    assert all(receipt["holdout_positive_counts"])
    assert not any("PC99" in path.name for path in train_view.rglob("*.wav"))


def test_taxi_builder_enforces_strict_chronology_and_target_isolation(
    tmp_path: Path,
) -> None:
    source = _taxi_source(tmp_path)
    root = build_taxi_holdout(
        source,
        tmp_path / "formal",
        minimum_train_rows=2,
        minimum_holdout_rows=2,
    )
    train_view = root / "train_view"
    manifest = validate_train_view(
        train_view / "fixed_holdout_manifest.json",
        train_view / "input",
    )
    receipt = manifest["split_receipt"]
    assert receipt["strategy"] == "chronological_deterministic_sha256_sample"
    assert receipt["train_holdout_key_overlap_count"] == 0
    assert receipt["future_to_past_count"] == 0
    assert (
        receipt["max_train_pickup_datetime"]
        < receipt["min_holdout_pickup_datetime"]
    )
    train = pd.read_csv(train_view / "input" / "train.csv")
    test = pd.read_csv(train_view / "input" / "test.csv")
    assert "fare_amount" in train.columns
    assert "fare_amount" not in test.columns
    labels = pd.read_csv(root / "evaluator_view" / "labels.csv")
    assert list(labels.columns) == ["key", "fare_amount"]
    assert test["key"].astype(str).tolist() == labels["key"].astype(str).tolist()


def test_formal_manifest_receipts_are_hash_bound(tmp_path: Path) -> None:
    root = build_aerial_holdout(_aerial_source(tmp_path), tmp_path / "formal")
    manifest = _read(root / "train_view" / "fixed_holdout_manifest.json")
    for name in ("split_receipt", "fit_scope_receipt", "metric_spec_receipt"):
        receipt = manifest[name]
        unsigned = {key: value for key, value in receipt.items() if key != "receipt_hash"}
        assert receipt["receipt_hash"] == hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
