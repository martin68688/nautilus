"""Normalize the five formal tasks into Host-only records before freezing views."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


TASK_STAGING_SCHEMA = "mlevolve_four_task_host_staging_v1"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MAX_ARCHIVE_IMAGE_COUNT = 100_000
MAX_ARCHIVE_IMAGE_BYTES = 20 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 128 * 1024 * 1024
TAXI_HOST_MAX_TRAIN_RECORDS = 250_000
TASK_SPECS: dict[str, dict[str, str]] = {
    "aerial-cactus-identification": {
        "task_family": "image",
        "protocol_ref": "stratified-roc-auc-classification@1",
        "label_key": "label",
        "metric_name": "roc_auc",
        "metric_direction": "maximize",
    },
    "leaf-classification": {
        "task_family": "tabular",
        "protocol_ref": "stratified-log-loss-classification@1",
        "label_key": "label",
        "metric_name": "log_loss",
        "metric_direction": "minimize",
    },
    "denoising-dirty-documents": {
        "task_family": "image",
        "protocol_ref": "deterministic-random-regression@1",
        "label_key": "target",
        "metric_name": "rmse",
        "metric_direction": "minimize",
    },
    "new-york-city-taxi-fare-prediction": {
        "task_family": "tabular",
        "protocol_ref": "chronological-regression@1",
        "label_key": "fare",
        "metric_name": "rmse",
        "metric_direction": "minimize",
    },
    "spooky-author-identification": {
        "task_family": "text",
        "protocol_ref": "stratified-log-loss-classification@1",
        "label_key": "author",
        "metric_name": "log_loss",
        "metric_direction": "minimize",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink {label}: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _unique_named(root: Path, names: Sequence[str], label: str) -> Path:
    lowered = {name.lower() for name in names}
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name.lower() in lowered
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label}; found {len(matches)}")
    return _regular(matches[0], label)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"CSV has no header: {path}")
        return fields, [dict(row) for row in reader]


def _coerce(value: str) -> Any:
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def _asset_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        for key in (path.name, path.stem):
            if key in index and index[key] != path:
                index[key] = Path()
            else:
                index[key] = path
    return index


def _resolve_asset(index: Mapping[str, Path], sample: str) -> Path:
    for key in (sample, Path(sample).name, Path(sample).stem):
        value = index.get(key)
        if value and str(value) != ".":
            return _regular(value, "task asset")
    raise ValueError(f"No unique asset matches sample {sample!r}")


def _extract_image_archive(archive: Path, destination: Path) -> Path:
    """Safely expand only image members from a task-owned ZIP archive."""

    source = _regular(archive, "task image archive")
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    total_size = 0
    with zipfile.ZipFile(source) as handle:
        for member in handle.infolist():
            if member.is_dir():
                continue
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or "\\" in member.filename
            ):
                raise ValueError(f"Unsafe image archive member: {member.filename!r}")
            if relative.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            mode = (member.external_attr >> 16) & 0xFFFF
            if mode and stat.S_ISLNK(mode):
                raise ValueError("Refusing symlink image archive member")
            if member.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError("Image archive member exceeds size limit")
            count += 1
            total_size += member.file_size
            if count > MAX_ARCHIVE_IMAGE_COUNT or total_size > MAX_ARCHIVE_IMAGE_BYTES:
                raise ValueError("Image archive exceeds Host staging limits")
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise ValueError(f"Duplicate image archive destination: {relative}")
            with handle.open(member, "r") as reader, target.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            target.chmod(0o444)
    if not count:
        raise ValueError(f"No images found in archive {source}")
    return destination


def _all_assets_resolve(index: Mapping[str, Path], samples: Iterable[str]) -> bool:
    try:
        for sample in samples:
            _resolve_asset(index, sample)
    except ValueError:
        return False
    return True


def _stage_aerial(
    root: Path,
    staging_assets: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_path = _unique_named(root, ("train.csv",), "aerial train.csv")
    sample_path = _unique_named(
        root, ("sample_submission.csv",), "aerial sample_submission.csv"
    )
    train_fields, train_rows = _read_csv(train_path)
    sample_fields, sample_rows = _read_csv(sample_path)
    label = "has_cactus" if "has_cactus" in train_fields else "label"
    if label not in train_fields:
        raise ValueError("Aerial train.csv requires has_cactus or label")
    id_field = next(
        (name for name in ("id", "image", "filename") if name in train_fields),
        None,
    )
    if id_field is None:
        raise ValueError("Aerial train.csv requires id/image/filename")
    inference_id = next(
        (name for name in (id_field, "id", "image", "filename") if name in sample_fields),
        None,
    )
    if inference_id is None:
        raise ValueError("Aerial sample submission lacks an image identifier")
    assets = _asset_index(root)
    if _all_assets_resolve(assets, (row[id_field] for row in train_rows)) and _all_assets_resolve(
        assets, (row[inference_id] for row in sample_rows)
    ):
        train_assets = inference_assets = assets
    else:
        train_archive = _unique_named(root, ("train.zip",), "aerial train.zip")
        test_archive = _unique_named(root, ("test.zip",), "aerial test.zip")
        train_assets = _asset_index(
            _extract_image_archive(train_archive, staging_assets / "train")
        )
        inference_assets = _asset_index(
            _extract_image_archive(test_archive, staging_assets / "test")
        )
    train = [
        {
            "sample_id": str(row[id_field]),
            "label": int(float(row[label])),
            "_host_assets": {
                "image": str(_resolve_asset(train_assets, row[id_field]))
            },
        }
        for row in train_rows
    ]
    inference = [
        {
            "sample_id": str(row[inference_id]),
            "_host_assets": {
                "image": str(_resolve_asset(inference_assets, row[inference_id]))
            },
        }
        for row in sample_rows
    ]
    return train, inference


def _stage_leaf(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_path = _unique_named(root, ("train.csv",), "leaf train.csv")
    test_path = _unique_named(root, ("test.csv",), "leaf test.csv")
    train_fields, train_rows = _read_csv(train_path)
    test_fields, test_rows = _read_csv(test_path)
    label = "species" if "species" in train_fields else "label"
    if label not in train_fields or "id" not in train_fields or "id" not in test_fields:
        raise ValueError("Leaf CSVs require id and train species/label")

    def normalize(row: Mapping[str, str], *, training: bool) -> dict[str, Any]:
        value = {
            str(key): _coerce(item)
            for key, item in row.items()
            if key not in {label}
        }
        value["sample_id"] = str(row["id"])
        if training:
            value["label"] = str(row[label])
        return value

    return (
        [normalize(row, training=True) for row in train_rows],
        [normalize(row, training=False) for row in test_rows],
    )


def _read_csv_bounded(
    path: Path,
    *,
    max_rows: int,
) -> tuple[list[str], list[dict[str, str]], bool]:
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"CSV has no header: {path}")
        rows: list[dict[str, str]] = []
        truncated = False
        for index, row in enumerate(reader):
            if index >= max_rows:
                truncated = True
                break
            rows.append(dict(row))
    return fields, rows, truncated


def _stage_taxi(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_path = _unique_named(
        root, ("train.csv", "labels.csv"), "taxi train.csv/labels.csv"
    )
    test_path = _unique_named(root, ("test.csv",), "taxi test.csv")
    train_fields, train_rows, truncated = _read_csv_bounded(
        train_path,
        max_rows=TAXI_HOST_MAX_TRAIN_RECORDS,
    )
    test_fields, test_rows = _read_csv(test_path)
    if "fare_amount" not in train_fields or "pickup_datetime" not in train_fields:
        raise ValueError("Taxi train.csv requires fare_amount and pickup_datetime")
    id_field = "key" if "key" in train_fields and "key" in test_fields else None

    def normalize(row: Mapping[str, str], index: int, *, training: bool) -> dict[str, Any]:
        sample_id = str(row[id_field]) if id_field else f"taxi-{index:012d}"
        value = {
            str(key): _coerce(item)
            for key, item in row.items()
            if key != "fare_amount"
        }
        value["sample_id"] = sample_id
        value["timestamp"] = str(row["pickup_datetime"])
        if training:
            value["fare"] = float(row["fare_amount"])
        return value

    return (
        [normalize(row, index, training=True) for index, row in enumerate(train_rows)],
        [normalize(row, index, training=False) for index, row in enumerate(test_rows)],
        {
            "training_source_name": train_path.name,
            "training_record_limit": TAXI_HOST_MAX_TRAIN_RECORDS,
            "training_records_truncated": truncated,
            "training_record_selection": "source_order_prefix",
        },
    )


def _image_dir(root: Path, names: Sequence[str], label: str) -> Path:
    matches = [
        path
        for path in root.rglob("*")
        if path.is_dir() and not path.is_symlink() and path.name.lower() in set(names)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label}; found {len(matches)}")
    return matches[0].resolve(strict=True)


def _images(directory: Path) -> dict[str, Path]:
    values = {
        path.name: _regular(path, "document image")
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    }
    if not values:
        raise ValueError(f"No images found in {directory}")
    return values


def _stage_denoising(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dirty_dir = _image_dir(root, ("train", "train_dirty"), "dirty train directory")
    clean_dir = _image_dir(root, ("train_cleaned", "train_clean"), "clean train directory")
    test_dir = _image_dir(root, ("test", "test_dirty"), "dirty test directory")
    dirty = _images(dirty_dir)
    clean = _images(clean_dir)
    if set(dirty) != set(clean):
        raise ValueError("Denoising dirty/clean training filenames do not match")
    train = [
        {
            "sample_id": name,
            "_host_assets": {"noisy": str(dirty[name]), "target": str(clean[name])},
        }
        for name in sorted(dirty)
    ]
    inference = [
        {"sample_id": name, "_host_assets": {"noisy": str(path)}}
        for name, path in sorted(_images(test_dir).items())
    ]
    return train, inference


def _stage_spooky(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stage the Spooky author text task without exposing test labels."""
    train_path = _unique_named(root, ("train.csv",), "Spooky train.csv")
    test_path = _unique_named(root, ("test.csv",), "Spooky test.csv")
    train_fields, train_rows = _read_csv(train_path)
    test_fields, test_rows = _read_csv(test_path)
    required_train = {"id", "text", "author"}
    required_test = {"id", "text"}
    if not required_train.issubset(train_fields):
        raise ValueError("Spooky train.csv requires id, text and author")
    if not required_test.issubset(test_fields):
        raise ValueError("Spooky test.csv requires id and text")
    train = [
        {
            "sample_id": str(row["id"]),
            "text": str(row["text"]),
            "author": str(row["author"]),
        }
        for row in train_rows
    ]
    inference = [
        {"sample_id": str(row["id"]), "text": str(row["text"])}
        for row in test_rows
    ]
    return train, inference


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = "".join(_canonical_json(dict(row)) + "\n" for row in rows).encode()
    _write_exclusive(path, content)


def stage_task(
    task_id: str,
    public_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    if task_id not in TASK_SPECS:
        raise ValueError(f"Unsupported formal Host adapter: {task_id}")
    requested = Path(public_root).expanduser()
    if requested.is_symlink():
        raise ValueError("Refusing symlink task public root")
    source_root = requested.resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("Task public root must be a directory")
    output = Path(output_root).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Task staging output root must be empty")
    output.mkdir(parents=True, exist_ok=True)
    staging_details: dict[str, Any] = {}
    if task_id == "aerial-cactus-identification":
        train, inference = _stage_aerial(source_root, output / "AERIAL_ASSETS")
    elif task_id == "leaf-classification":
        train, inference = _stage_leaf(source_root)
    elif task_id == "denoising-dirty-documents":
        train, inference = _stage_denoising(source_root)
    elif task_id == "spooky-author-identification":
        train, inference = _stage_spooky(source_root)
    else:
        train, inference, staging_details = _stage_taxi(source_root)
    if len(train) < 2 or not inference:
        raise ValueError("Host task staging requires training and inference records")
    train_path = output / "TRAIN_RECORDS.jsonl"
    inference_path = output / "INFERENCE_RECORDS.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(inference_path, inference)
    spec = TASK_SPECS[task_id]
    payload = {
        "schema": TASK_STAGING_SCHEMA,
        "task_id": task_id,
        **spec,
        "source_public_root": str(source_root),
        "train_records_path": str(train_path),
        "train_record_count": len(train),
        "train_records_sha256": _sha256_file(train_path),
        "inference_records_path": str(inference_path),
        "inference_record_count": len(inference),
        "inference_records_sha256": _sha256_file(inference_path),
        "terminal_labels_in_staging": False,
        **staging_details,
        "staging_hash": "",
    }
    payload["staging_hash"] = hashlib.sha256(
        _canonical_json(
            {key: value for key, value in payload.items() if key != "staging_hash"}
        ).encode()
    ).hexdigest()
    _write_exclusive(
        output / "TASK_STAGING_MANIFEST.json",
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, choices=sorted(TASK_SPECS))
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            stage_task(args.task_id, args.public_root, args.output_root),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TASK_SPECS", "TASK_STAGING_SCHEMA", "stage_task"]
