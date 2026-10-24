"""Hidden-label evaluator for fixed MLEvolve holdouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import f1_score, roc_auc_score

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
    label_columns = list(manifest.get("label_columns") or prediction_columns)
    y_true = labels[label_columns].to_numpy(dtype=float)
    y_pred = predictions[prediction_columns].to_numpy(dtype=float)
    if not np.isfinite(y_pred).all():
        raise ValueError("Predictions contain NaN or infinite values")
    group_column = manifest.get("group_column")
    groups = labels[group_column].astype(str).to_numpy() if group_column else None
    benchmark_column = manifest.get("benchmark_column")
    benchmark = labels[benchmark_column].to_numpy(dtype=float) if benchmark_column else None
    return manifest, y_true, y_pred, groups, benchmark


def _numerai_corr(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Numerai's per-era correlation (same transform as ``numerai_tools``).

    Predictions are tie-broken ranked, gaussianized and raised to the 1.5
    power; targets are centered and raised to the 1.5 power; the score is the
    Pearson correlation of the two transformed vectors.
    """
    n = len(predictions)
    order = np.argsort(predictions, kind="stable")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    gaussian = norm.ppf((ranks - 0.5) / n)
    pred_transformed = np.sign(gaussian) * np.abs(gaussian) ** 1.5
    centered = targets - targets.mean()
    target_transformed = np.sign(centered) * np.abs(centered) ** 1.5
    if pred_transformed.std() == 0 or target_transformed.std() == 0:
        return 0.0
    return float(np.corrcoef(pred_transformed, target_transformed)[0, 1])


def _gaussian_rank(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="average").to_numpy()
    return norm.ppf((ranks - 0.5) / len(values))


def _benchmark_contribution(predictions: np.ndarray, benchmark: np.ndarray, targets: np.ndarray) -> float:
    """BMC for one era, as ``numerai_tools.scoring.correlation_contribution``.

    Gaussianized predictions are orthogonalized against the gaussianized
    benchmark predictions; the contribution is their covariance with the
    centered target, with [0, 1] targets first scaled to buckets [-2, 2].
    """
    p = _gaussian_rank(predictions)
    m = _gaussian_rank(benchmark)
    neutral = p - m * (p @ m) / (m @ m)
    if targets.min() >= 0 and targets.max() <= 1:
        targets = targets * 4
    centered = targets - targets.mean()
    return float(centered @ neutral / len(targets))


def _summary(values: np.ndarray, prefix: str) -> dict:
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": std,
        f"{prefix}_sharpe": float(values.mean() / std) if std > 0 else 0.0,
    }


def numerai_era_scores(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, benchmark: np.ndarray | None = None
) -> dict:
    """Score every era separately and summarize the per-era correlations.

    With Host-only ``benchmark`` predictions, also report BMC (benchmark model
    contribution): how much of the prediction's signal is not already in the
    benchmark model.
    """
    if groups is None:
        raise ValueError("numerai_era_corr requires a group_column in the manifest")
    targets = y_true.reshape(-1)
    predictions = y_pred.reshape(-1)
    if not np.isfinite(targets).all():
        raise ValueError("Hidden Numerai targets contain NaN values")
    frame = pd.DataFrame({"era": groups, "pred": predictions, "target": targets})
    per_era = {
        str(era): _numerai_corr(group["pred"].to_numpy(), group["target"].to_numpy())
        for era, group in frame.groupby("era", sort=True)
    }
    values = np.array(list(per_era.values()), dtype=float)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    cumulative = np.cumsum(values)
    drawdown = float(np.max(np.maximum.accumulate(cumulative) - cumulative))
    result = {
        "corr_mean": float(values.mean()),
        "corr_std": std,
        "corr_sharpe": float(values.mean() / std) if std > 0 else 0.0,
        "max_drawdown": drawdown,
        "n_eras": int(len(values)),
        "per_era_corr": per_era,
    }
    if benchmark is not None:
        reference = np.asarray(benchmark, dtype=float).reshape(-1)
        if not np.isfinite(reference).all():
            raise ValueError("Host benchmark predictions contain NaN values")
        frame["benchmark"] = reference
        per_era_bmc = {
            str(era): _benchmark_contribution(
                group["pred"].to_numpy(), group["benchmark"].to_numpy(), group["target"].to_numpy()
            )
            for era, group in frame.groupby("era", sort=True)
        }
        result |= _summary(np.array(list(per_era_bmc.values()), dtype=float), "bmc")
        result["per_era_bmc"] = per_era_bmc
    return result


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


def _macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    threshold: float,
    zero_division: int,
) -> float:
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("Macro-F1 labels and predictions must have matching matrices")
    if not set(np.unique(y_true)).issubset({0.0, 1.0}):
        raise ValueError("Macro-F1 labels must contain only 0 and 1")
    if (y_pred < 0).any() or (y_pred > 1).any():
        raise ValueError("Macro-F1 probability predictions must be in [0, 1]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Macro-F1 threshold must be in [0, 1]")
    true_values = y_true.astype(int)
    predicted_values = (y_pred >= threshold).astype(int)
    if true_values.shape[1] == 1:
        true_values = true_values.reshape(-1)
        predicted_values = predicted_values.reshape(-1)
    return float(
        f1_score(
            true_values,
            predicted_values,
            average="macro",
            zero_division=zero_division,
        )
    )


def evaluate_submission(manifest_path: Path, submission_path: Path) -> dict:
    manifest, y_true, y_pred, groups, benchmark = _load_aligned_frames(manifest_path, submission_path)
    metric = manifest["metric"]
    details = None
    if metric == "numerai_era_corr":
        details = numerai_era_scores(y_true, y_pred, groups, benchmark)
        score = details["corr_mean"]
    elif metric == "multiclass_log_loss":
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
    elif metric == "macro_f1":
        score = _macro_f1(
            y_true,
            y_pred,
            threshold=float(manifest.get("prediction_threshold", 0.5)),
            zero_division=int(manifest.get("zero_division", 0)),
        )
    else:
        raise ValueError(f"Unsupported fixed-holdout metric: {metric}")
    result = {
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
    if details is not None:
        result["metric_details"] = details
    return result


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
