"""Hidden-label evaluator for fixed MLEvolve holdouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fixed_holdout.common import read_manifest, sha256_file, sha256_lines


def _load_aligned_frames(manifest_path: Path, submission_path: Path):
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path, expected_role="evaluator_view")
    label_path = manifest_path.parent / manifest["labels_file"]
    if sha256_file(label_path) != manifest["labels_sha256"]:
        raise ValueError("Hidden labels do not match the evaluator manifest")
    labels = pd.read_csv(label_path)
    predictions = pd.read_csv(submission_path)

    id_column = manifest["id_column"]
    prediction_columns = list(manifest["prediction_columns"])
    expected_columns = [id_column, *prediction_columns]
    if list(predictions.columns) != expected_columns:
        raise ValueError(
            f"Submission columns must be exactly {expected_columns}; "
            f"got {list(predictions.columns)}"
        )
    for name, frame in (("labels", labels), ("submission", predictions)):
        if frame[id_column].isna().any() or frame[id_column].duplicated().any():
            raise ValueError(f"{name} contain missing or duplicate IDs")
    if len(predictions) != int(manifest["row_count"]):
        raise ValueError(
            f"Submission has {len(predictions)} rows; expected {manifest['row_count']}"
        )
    prediction_ids = predictions[id_column].astype(str)
    if sha256_lines(prediction_ids) != manifest["holdout_id_sha256"]:
        raise ValueError("Submission IDs or row order do not match the fixed holdout")
    if not prediction_ids.equals(labels[id_column].astype(str)):
        raise ValueError("Submission IDs do not align with hidden labels")
    y_true = labels[prediction_columns].to_numpy(dtype=float)
    y_pred = predictions[prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain NaN or infinite values")
    return manifest, y_true, y_pred


def _multiclass_log_loss(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool) -> float:
    if y_true.ndim != 2 or y_true.shape != y_pred.shape:
        raise ValueError("Multiclass labels and predictions must have matching matrices")
    if not np.allclose(y_true.sum(axis=1), 1.0):
        raise ValueError("Hidden multiclass labels are not one-hot encoded")
    if (y_pred < 0).any() or (y_pred > 1).any():
        raise ValueError("Multiclass predictions must be in [0, 1]")
    if normalize:
        row_sums = y_pred.sum(axis=1, keepdims=True)
        if (row_sums <= 0).any():
            raise ValueError("Every prediction row must have positive probability mass")
        y_pred = y_pred / row_sums
    clipped = np.clip(y_pred, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(np.sum(y_true * np.log(clipped), axis=1)))


def _binary_targets(y_true: np.ndarray) -> np.ndarray:
    values = y_true.reshape(-1)
    if not set(np.unique(values)).issubset({0.0, 1.0}):
        raise ValueError("Binary labels must contain only 0 and 1")
    return values


def evaluate_submission(manifest_path: Path, submission_path: Path) -> dict:
    manifest, y_true, y_pred = _load_aligned_frames(manifest_path, submission_path)
    metric = manifest["metric"]
    if metric == "multiclass_log_loss":
        score = _multiclass_log_loss(
            y_true,
            y_pred,
            bool(manifest.get("normalize_probabilities", False)),
        )
    elif metric == "binary_roc_auc":
        predictions = y_pred.reshape(-1)
        if (predictions < 0).any() or (predictions > 1).any():
            raise ValueError("Binary probability predictions must be in [0, 1]")
        score = float(roc_auc_score(_binary_targets(y_true), predictions))
    elif metric == "binary_log_loss":
        predictions = np.clip(y_pred.reshape(-1), 1e-15, 1.0 - 1e-15)
        targets = _binary_targets(y_true)
        score = float(-np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions)))
    elif metric == "rmse":
        score = float(math.sqrt(np.mean(np.square(y_pred.reshape(-1) - y_true.reshape(-1)))))
    elif metric == "mae":
        score = float(np.mean(np.abs(y_pred.reshape(-1) - y_true.reshape(-1))))
    elif metric == "accuracy":
        score = float(np.mean(y_pred.reshape(-1) == y_true.reshape(-1)))
    else:
        raise ValueError(f"Unsupported fixed-holdout metric: {metric}")
    return {
        "schema": manifest["schema"],
        "task_id": manifest["task_id"],
        "split_id": manifest["split_id"],
        "metric": metric,
        "maximize": bool(manifest["maximize"]),
        "score": score,
        "row_count": int(manifest["row_count"]),
        "submission_sha256": sha256_file(Path(submission_path)),
        "selection_policy": "terminal_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_submission(args.manifest, args.submission)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
