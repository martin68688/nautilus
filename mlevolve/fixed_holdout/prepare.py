"""Materialize physically separated training and evaluator views."""

from __future__ import annotations

import argparse
import errno
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from fixed_holdout.catalog import task_spec
from fixed_holdout.common import (
    SCHEMA,
    read_manifest,
    sha256_bytes,
    sha256_file,
    sha256_lines,
    tree_sha256,
    write_json,
)


def _copy_file(source: Path, destination: Path, copy_mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode not in {"auto", "hardlink", "copy"}:
        raise ValueError(f"Unsupported copy mode: {copy_mode}")
    if copy_mode in {"auto", "hardlink"}:
        try:
            os.link(source.resolve(strict=True), destination)
            return "hardlink"
        except OSError as exc:
            if copy_mode == "hardlink" or exc.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.EMLINK,
            }:
                raise
    shutil.copy2(source, destination)
    return "copy"


def _copy_tree(source: Path, destination: Path, copy_mode: str) -> set[str]:
    modes: set[str] = set()
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            modes.add(_copy_file(path, target, copy_mode))
    return modes


def _prediction_columns(spec: dict, sample: pd.DataFrame) -> list[str]:
    configured = spec["prediction_columns"]
    if configured == "sample_submission":
        columns = [column for column in sample.columns if column != spec["id_column"]]
    else:
        columns = list(configured)
    if not columns:
        raise ValueError("Fixed-holdout task has no prediction columns")
    return columns


def _validate_ids(sample: pd.DataFrame, labels: pd.DataFrame, id_column: str) -> None:
    for name, frame in (("sample submission", sample), ("private labels", labels)):
        if id_column not in frame.columns:
            raise ValueError(f"{name} is missing ID column {id_column!r}")
        if frame[id_column].isna().any() or frame[id_column].duplicated().any():
            raise ValueError(f"{name} has missing or duplicate IDs")
    sample_ids = sample[id_column].astype(str).tolist()
    label_ids = labels[id_column].astype(str).tolist()
    if sample_ids != label_ids:
        raise ValueError(
            "Sample submission and private labels must contain identical IDs in identical order"
        )


def _write_selection_view(view: Path, source: Path, spec: dict, common: dict) -> None:
    """Host-only labels for the search-time selection set (never in train_view)."""
    columns = [spec["id_column"], *spec["label_columns"], spec["group_column"]]
    labels = pd.read_csv(source)
    missing = [column for column in columns if column not in labels.columns]
    if missing:
        raise ValueError(f"Selection labels are missing columns: {missing}")
    labels = labels[columns]
    if labels[spec["id_column"]].isna().any() or labels[spec["id_column"]].duplicated().any():
        raise ValueError("Selection labels have missing or duplicate IDs")
    view.mkdir(parents=True, exist_ok=True)
    label_path = view / "labels.csv"
    labels.to_csv(label_path, index=False)
    write_json(
        view / "fixed_holdout_manifest.json",
        {
            "schema": SCHEMA,
            "role": "selection_view",
            "task_id": common["task_id"],
            "split_id": common["split_id"],
            "metric": common["metric"],
            "id_column": spec["id_column"],
            "prediction_columns": common["prediction_columns"],
            "label_columns": list(spec["label_columns"]),
            "group_column": spec["group_column"],
            "row_count": int(len(labels)),
            "selection_id_sha256": sha256_lines(labels[spec["id_column"]].astype(str)),
            "labels_file": "labels.csv",
            "labels_sha256": sha256_file(label_path),
            "selection_policy": "host_search_metric",
        },
    )


def prepare_task(
    dataset_root: Path,
    task_id: str,
    output_root: Path,
    *,
    copy_mode: str = "auto",
) -> Path:
    """Create one immutable train/evaluator split and return its root path."""
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    spec = task_spec(task_id)
    task_root = dataset_root / task_id
    public_source = task_root / spec["public_subdir"]
    private_source = task_root / spec["private_labels"]
    sample_source = public_source / spec["sample_submission"]
    for required in (public_source, private_source, sample_source):
        if not required.exists():
            raise ValueError(f"Required fixed-holdout source is missing: {required}")

    sample = pd.read_csv(sample_source)
    private = pd.read_csv(private_source)
    prediction_columns = _prediction_columns(spec, sample)
    # Most tasks name hidden labels after the prediction columns.  Grouped
    # metrics (Numerai) keep differently named labels plus a group column.
    label_columns = list(spec.get("label_columns") or prediction_columns)
    group_column = spec.get("group_column")
    required_private_columns = [
        spec["id_column"],
        *label_columns,
        *([group_column] if group_column else []),
    ]
    missing = [column for column in required_private_columns if column not in private.columns]
    if missing:
        raise ValueError(f"Private labels are missing columns: {missing}")
    _validate_ids(sample, private, spec["id_column"])
    label_frame = private[required_private_columns].copy()
    source_labels_sha256 = sha256_bytes(
        label_frame.to_csv(index=False).encode("utf-8")
    )

    public_digest = tree_sha256(public_source)
    id_digest = sha256_lines(sample[spec["id_column"]].astype(str))
    split_id = f"{task_id}-{public_digest[:12]}-{id_digest[:12]}"
    final_root = output_root / task_id / split_id
    if final_root.exists():
        train_manifest = read_manifest(
            final_root / "train_view" / "fixed_holdout_manifest.json",
            expected_role="train_view",
        )
        evaluator_manifest = read_manifest(
            final_root / "evaluator_view" / "fixed_holdout_manifest.json",
            expected_role="evaluator_view",
        )
        if (
            train_manifest.get("public_tree_sha256") != public_digest
            or train_manifest.get("holdout_id_sha256") != id_digest
            or evaluator_manifest.get("split_id") != split_id
        ):
            raise ValueError(f"Existing split conflicts with current sources: {final_root}")
        if evaluator_manifest.get("labels_sha256") != source_labels_sha256:
            raise ValueError(
                f"Existing split labels do not match current private source: {final_root}"
            )
        label_path = final_root / "evaluator_view" / evaluator_manifest["labels_file"]
        if sha256_file(label_path) != evaluator_manifest.get("labels_sha256"):
            raise ValueError(f"Existing evaluator labels are corrupted: {final_root}")
        return final_root

    output_root.mkdir(parents=True, exist_ok=True)
    staging_parent = output_root / task_id
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=f".{split_id}.", dir=staging_parent))
    try:
        train_view = staging_root / "train_view"
        input_dir = train_view / "input"
        evaluator_view = staging_root / "evaluator_view"
        copy_modes = _copy_tree(public_source, input_dir, copy_mode)

        evaluator_view.mkdir(parents=True, exist_ok=True)
        label_path = evaluator_view / "labels.csv"
        label_frame.to_csv(label_path, index=False)

        common = {
            "schema": SCHEMA,
            "task_id": task_id,
            "split_id": split_id,
            "metric": spec["metric"],
            "maximize": spec["metric"]
            in {"binary_roc_auc", "accuracy", "numerai_era_corr"},
            "id_column": spec["id_column"],
            "prediction_columns": prediction_columns,
            "normalize_probabilities": bool(spec["normalize_probabilities"]),
            "row_count": int(len(sample)),
            "holdout_id_sha256": id_digest,
            "public_tree_sha256": public_digest,
            "selection_policy": "terminal_only",
        }
        if "label_columns" in spec:
            common["label_columns"] = label_columns
        if group_column:
            common["group_column"] = group_column
        train_manifest = {
            **common,
            "role": "train_view",
            "input_subdir": "input",
            "copy_modes": sorted(copy_modes),
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
        selection_source = (
            task_root / spec["selection_labels"] if spec.get("selection_labels") else None
        )
        if selection_source is not None and selection_source.exists():
            _write_selection_view(staging_root / "selection_view", selection_source, spec, common)
        refit_source = task_root / spec["refit_train"] if spec.get("refit_train") else None
        if refit_source is not None and refit_source.exists():
            # Same public files, but train.parquet also holds the Host-labelled
            # embargo and selection eras; used only after the search.
            refit_input = staging_root / "refit_view" / "input"
            _copy_tree(public_source, refit_input, copy_mode)
            (refit_input / "train.parquet").unlink()
            shutil.copy2(refit_source, refit_input / "train.parquet")
            write_json(
                staging_root / "refit_view" / "fixed_holdout_manifest.json",
                {
                    "schema": SCHEMA,
                    "role": "refit_view",
                    "task_id": task_id,
                    "split_id": split_id,
                    "input_subdir": "input",
                    "refit_tree_sha256": tree_sha256(refit_input),
                    "selection_policy": "post_search_refit_only",
                },
            )
        write_json(
            staging_root / "split_manifest.json",
            {
                **common,
                "role": "split_root",
                "train_manifest": "train_view/fixed_holdout_manifest.json",
                "evaluator_manifest": "evaluator_view/fixed_holdout_manifest.json",
            },
        )
        staging_root.rename(final_root)
    except BaseException:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return final_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--copy-mode", choices=["auto", "hardlink", "copy"], default="auto")
    args = parser.parse_args()
    for task_id in args.task:
        split_root = prepare_task(
            args.dataset_root,
            task_id,
            args.output_root,
            copy_mode=args.copy_mode,
        )
        print(split_root)


if __name__ == "__main__":
    main()
