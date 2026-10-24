"""Host-held selection set: the search metric is computed by the Host.

Self-reported validation scores let a candidate tune and report on the same
rows.  With a selection set, labelled public rows are moved out of the train
view; candidates only see their features (``selection.parquet``) and must
write ``./submission/selection_predictions.csv``.  The Host scores those
predictions per group (era), splits the groups into contiguous time windows
and uses the window summary as the node metric.  The fixed-holdout lockbox is
untouched and is still scored only after the search.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fixed_holdout.common import read_manifest, sha256_file
from fixed_holdout.evaluate import numerai_era_scores

SELECTION_FILENAME = "selection_predictions.csv"


def predictions_path(workspace_dir: Path, node_id: str) -> Path:
    return Path(workspace_dir) / "submission" / f"selection_predictions_{node_id}.csv"


def _window_summary(per_group: dict[str, float], windows: int) -> list[float]:
    ordered = [per_group[key] for key in sorted(per_group)]
    chunks = np.array_split(np.asarray(ordered, dtype=float), max(1, min(windows, len(ordered))))
    return [float(chunk.mean()) for chunk in chunks if len(chunk)]


def score_selection(
    manifest_path: Path,
    predictions_file: Path,
    *,
    windows: int = 4,
    objective: str = "mean",
    std_penalty: float = 0.5,
    score_metric: str = "corr",
) -> dict:
    """Score selection predictions against Host-held labels.

    ``score_metric`` picks the per-era series that ranks candidates: ``corr``
    (Numerai CORR) or ``bmc`` (contribution beyond the benchmark model, which
    needs Host-only benchmark predictions in the selection view).
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = read_manifest(manifest_path, expected_role="selection_view")
    if manifest["metric"] != "numerai_era_corr":
        raise ValueError(f"Selection scoring supports grouped numerai_era_corr only, got {manifest['metric']}")
    label_path = manifest_path.parent / manifest["labels_file"]
    if sha256_file(label_path) != manifest["labels_sha256"]:
        raise ValueError("Selection labels do not match the selection manifest")
    labels = pd.read_csv(label_path)
    predictions = pd.read_csv(predictions_file)

    id_column = manifest["id_column"]
    prediction_column = manifest["prediction_columns"][0]
    if list(predictions.columns) != [id_column, prediction_column]:
        raise ValueError(
            f"Selection predictions must have columns {[id_column, prediction_column]}; "
            f"got {list(predictions.columns)}"
        )
    if predictions[id_column].duplicated().any():
        raise ValueError("Selection predictions contain duplicate ids")
    merged = labels.merge(predictions, on=id_column, how="left", validate="one_to_one")
    missing = int(merged[prediction_column].isna().sum())
    if missing:
        raise ValueError(f"Selection predictions are missing {missing} of {len(labels)} ids")
    values = merged[prediction_column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Selection predictions contain NaN or infinite values")

    label_column = manifest["label_columns"][0]
    benchmark_column = manifest.get("benchmark_column")
    details = numerai_era_scores(
        merged[[label_column]].to_numpy(dtype=float),
        values.reshape(-1, 1),
        merged[manifest["group_column"]].astype(str).to_numpy(),
        merged[benchmark_column].to_numpy(dtype=float) if benchmark_column else None,
    )
    series = {"corr": "per_era_corr", "bmc": "per_era_bmc"}.get(score_metric)
    if series is None or series not in details:
        raise ValueError(f"Selection score_metric {score_metric!r} is unavailable for this split")
    window_means = _window_summary(details[series], windows)
    mean = float(np.mean(window_means))
    spread = float(np.std(window_means, ddof=1)) if len(window_means) > 1 else 0.0
    if objective == "mean":
        score = mean
    elif objective == "mean_minus_std":
        score = mean - std_penalty * spread
    else:
        raise ValueError(f"Unsupported selection objective: {objective}")
    return {
        "score": score,
        "score_metric": score_metric,
        "objective": objective,
        "window_means": window_means,
        "window_std": spread,
        "corr_mean": details["corr_mean"],
        "corr_sharpe": details["corr_sharpe"],
        **{key: details[key] for key in ("bmc_mean", "bmc_sharpe") if key in details},
        "n_groups": details["n_eras"],
        "split_id": manifest["split_id"],
    }


def apply_to_node(agent, node, logger) -> None:
    """Replace a successful node's self-reported metric with the Host score."""
    cfg = getattr(agent.cfg, "host_selection", None)
    if not getattr(cfg, "enabled", False) or node.is_buggy:
        return
    from utils.metric import MetricValue, WorstMetricValue

    path = predictions_path(agent.cfg.workspace_dir, node.id)
    reported = node.metric.value if node.metric is not None else None
    try:
        if not path.exists():
            raise ValueError(
                f"missing ./submission/{SELECTION_FILENAME} (predictions for selection.parquet)"
            )
        result = score_selection(
            Path(cfg.manifest_path),
            path,
            windows=int(cfg.windows),
            objective=str(cfg.objective),
            std_penalty=float(cfg.std_penalty),
            score_metric=str(getattr(cfg, "score_metric", "corr")),
        )
    except Exception as error:
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = f"HOST_SELECTION_ERROR: {error}"
        node._term_out.append(f"\n{node.analysis}")
        logger.warning("Node %s failed Host selection scoring: %s", node.id, error)
        return
    result["agent_reported_metric"] = reported
    node.host_selection = result
    node.metric = MetricValue(result["score"], maximize=True)
    logger.info(
        "Node %s Host selection score=%.5f (%s; agent reported %s; windows=%s; corr=%.5f bmc=%s)",
        node.id, result["score"], result["score_metric"], reported,
        [round(v, 4) for v in result["window_means"]], result["corr_mean"],
        f"{result['bmc_mean']:.5f}" if "bmc_mean" in result else "n/a",
    )
