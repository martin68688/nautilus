"""Label-free validation available inside the training environment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fixed_holdout.common import read_manifest, sha256_lines, tree_sha256


def validate_train_view(manifest_path: Path, data_dir: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    data_dir = Path(data_dir).resolve()
    manifest = read_manifest(manifest_path, expected_role="train_view")
    expected_input = (manifest_path.parent / manifest["input_subdir"]).resolve()
    if data_dir != expected_input:
        raise ValueError(
            f"Fixed-holdout data_dir must be {expected_input}, got {data_dir}"
        )
    if manifest.get("hidden_labels_present") is not False:
        raise ValueError("Train manifest must explicitly declare hidden_labels_present=false")
    if manifest.get("selection_policy") != "terminal_only":
        raise ValueError("Only terminal_only fixed-holdout evaluation is supported")
    observed_digest = tree_sha256(data_dir)
    if observed_digest != manifest.get("public_tree_sha256"):
        raise ValueError("Fixed-holdout train view does not match its immutable manifest")
    return manifest


def validate_submission(manifest_path: Path, submission_path: Path) -> tuple[bool, str]:
    manifest = read_manifest(Path(manifest_path), expected_role="train_view")
    submission_path = Path(submission_path)
    try:
        frame = pd.read_csv(submission_path)
    except Exception as exc:
        return False, f"Cannot read submission CSV: {exc}"

    id_column = manifest["id_column"]
    prediction_columns = list(manifest["prediction_columns"])
    expected_columns = [id_column, *prediction_columns]
    if list(frame.columns) != expected_columns:
        return False, f"Expected columns {expected_columns}, got {list(frame.columns)}"
    if len(frame) != int(manifest["row_count"]):
        return False, f"Expected {manifest['row_count']} rows, got {len(frame)}"
    if frame[id_column].isna().any() or frame[id_column].duplicated().any():
        return False, "Submission has missing or duplicate IDs"
    observed_id_digest = sha256_lines(frame[id_column].astype(str))
    if observed_id_digest != manifest["holdout_id_sha256"]:
        return False, "Submission IDs or row order do not match the fixed holdout"
    try:
        predictions = frame[prediction_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        return False, f"Prediction columns must be numeric: {exc}"
    if not np.isfinite(predictions).all():
        return False, "Predictions contain NaN or infinite values"
    if manifest["metric"] in {
        "multiclass_log_loss",
        "binary_log_loss",
        "binary_roc_auc",
    } and ((predictions < 0).any() or (predictions > 1).any()):
        return False, "Probability predictions must be in [0, 1]"
    if manifest["metric"] == "multiclass_log_loss":
        row_sums = predictions.sum(axis=1)
        if (row_sums <= 0).any():
            return False, "Every multiclass probability row must have positive mass"
    return True, "valid"
