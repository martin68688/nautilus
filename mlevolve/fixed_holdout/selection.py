"""Host-held selection set: the search metric is computed by the Host.

Self-reported validation scores let a candidate tune and report on the same
rows.  With a selection set, labelled public rows are moved out of the train
view; candidates only see their features.  For Numerai they are rows of
``test.parquet`` next to the lockbox rows (manifest ``predictions_source:
submission``), so both get the same code path and the selection rows are read
from the node's submission; older splits use ``selection.parquet`` and
``./submission/selection_predictions.csv``.  The Host scores those
predictions per group (era), splits the groups into contiguous time windows
and uses the window summary as the node metric.  The fixed-holdout lockbox is
untouched and is still scored only after the search.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import re
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_holdout.common import read_manifest, sha256_file
from fixed_holdout.evaluate import numerai_era_scores

_logger = logging.getLogger("MLEvolve")

# ---- verdict-only feedback ---------------------------------------------------
# With ``host_selection.feedback: verdict`` the agent never sees a Host number:
# prompts say only whether a node is better, tied or worse than its parent
# (paired noise band).  As a safety net every prompt is scanned for the Host
# numbers recorded so far and they are replaced before the model sees them.
_VERDICT = {"on": False}
_HIDDEN: set[float] = set()
_HIDDEN_LOCK = threading.Lock()
_FLOAT_RE = re.compile(r"(?<![\w.])[-+]?\d+\.\d{4,}(?:[eE][-+]?\d+)?(?![\w.])")
_HOST_KEYS = ("score", "corr_mean", "corr_sharpe", "bmc_mean", "bmc_sharpe", "benchmark_similarity", "window_std")
VERDICT_TEXT = {
    "better": "better than its parent (beyond the Host's noise band)",
    "tied": "no clear difference from its parent (within the Host's noise band)",
    "worse": "worse than its parent (beyond the Host's noise band)",
    None: "scored by the Host; no scored parent to compare with",
}
HIDDEN_SCORE_NOTE = (
    "not shown: the Host reports only whether a solution is better, tied or worse than its parent"
)


def configure_feedback(cfg) -> bool:
    """Turn verdict-only feedback on when the config asks for it; returns the mode."""
    host = getattr(cfg, "host_selection", None)
    on = bool(getattr(host, "enabled", False)) and str(getattr(host, "feedback", "score") or "score") == "verdict"
    _VERDICT["on"] = on
    return on


def verdict_feedback() -> bool:
    return _VERDICT["on"]


def hide_host_numbers(result: dict) -> None:
    """Record a Host result's numbers so prompts can be scrubbed of them."""
    values = [result.get(key) for key in _HOST_KEYS] + list(result.get("window_means") or [])
    for fold in (result.get("oof") or {}).get("folds", []):
        values += [fold.get(key) for key in ("score", "corr_mean", "bmc_mean", "benchmark_similarity")]
    values.append((result.get("oof") or {}).get("selection_score"))
    with _HIDDEN_LOCK:
        _HIDDEN.update(float(v) for v in values if isinstance(v, (int, float)) and np.isfinite(v) and v != 0)


def _matches_hidden(token: str) -> bool:
    try:
        value = float(token)
    except ValueError:
        return False
    if "e" in token.lower():
        tolerance = abs(value) * 1e-6 + 1e-15
    else:
        tolerance = 0.5 * 10 ** -len(token.split(".", 1)[1]) + 1e-15
    with _HIDDEN_LOCK:
        return any(abs(hidden - value) <= tolerance for hidden in _HIDDEN)


def mask_host_numbers(text: str) -> tuple[str, int]:
    """Replace Host numbers (4+ decimals) in ``text``; returns the text and the count."""
    if not _VERDICT["on"] or not text or not _HIDDEN:
        return text, 0
    count = 0

    def substitute(match: re.Match) -> str:
        nonlocal count
        if _matches_hidden(match.group(0)):
            count += 1
            return "[hidden]"
        return match.group(0)

    return _FLOAT_RE.sub(substitute, text), count


def mask_prompt(prompt):
    """Apply ``mask_host_numbers`` to every string of a prompt (str, dict or list)."""
    if not _VERDICT["on"] or not _HIDDEN:
        return prompt
    total = 0

    def walk(obj):
        nonlocal total
        if isinstance(obj, str):
            text, count = mask_host_numbers(obj)
            total += count
            return text
        if isinstance(obj, dict):
            return {key: walk(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(walk(value) for value in obj)
        return obj

    masked = walk(prompt)
    if total:
        _logger.warning("Verdict feedback: masked %d Host number(s) in a prompt", total)
    return masked


def feedback_text(node) -> str | None:
    """Verdict-only description of a node's Host result; None when numbers may be shown."""
    if not _VERDICT["on"]:
        return None
    host = getattr(node, "host_selection", None) or {}
    if "score" not in host:
        return "not scored by the Host"
    return VERDICT_TEXT.get(host.get("verdict_vs_parent"), VERDICT_TEXT[None])


def _scored_ancestor(node):
    parent = getattr(node, "parent", None)
    while parent is not None:
        host = getattr(parent, "host_selection", None) or {}
        if not getattr(parent, "is_buggy", False) and host.get("per_era_score"):
            return parent
        parent = getattr(parent, "parent", None)
    return None


def predictions_path(workspace_dir: Path, node_id: str, manifest_path: Path | None = None) -> Path:
    """Where a node's selection predictions are: its own file, or its submission
    when the selection rows are part of test.parquet (``predictions_source``)."""
    source = "selection_predictions"
    if manifest_path is not None:
        source = read_manifest(Path(manifest_path), expected_role="selection_view").get(
            "predictions_source", source
        )
    if source == "submission":
        return Path(workspace_dir) / "submission" / f"submission_{node_id}.csv"
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
    corr_weight: float = 0.5,
    bmc_weight: float = 2.0,
) -> dict:
    """Score selection predictions against Host-held labels.

    ``score_metric`` picks the per-era series that ranks candidates: ``corr``
    (Numerai CORR), ``bmc`` (contribution beyond the benchmark model, which
    needs Host-only benchmark predictions in the selection view), or
    ``corr_bmc``: ``corr_weight * CORR + bmc_weight * BMC`` per era (defaults
    follow Numerai's 0.5 * CORR + 2 * MMC payout, with BMC in place of MMC).
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
    if score_metric == "corr_bmc":
        if "per_era_bmc" not in details:
            raise ValueError("Selection score_metric 'corr_bmc' needs Host benchmark predictions")
        per_era = {
            era: corr_weight * details["per_era_corr"][era] + bmc_weight * details["per_era_bmc"][era]
            for era in details["per_era_corr"]
        }
    else:
        series = {"corr": "per_era_corr", "bmc": "per_era_bmc"}.get(score_metric)
        if series is None or series not in details:
            raise ValueError(f"Selection score_metric {score_metric!r} is unavailable for this split")
        per_era = details[series]
    window_means = _window_summary(per_era, windows)
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
        "benchmark_similarity": details.get("benchmark_similarity"),
        "score_metric": score_metric,
        "objective": objective,
        "window_means": window_means,
        "window_std": spread,
        "per_era_score": {era: float(value) for era, value in per_era.items()},
        "corr_mean": details["corr_mean"],
        "corr_sharpe": details["corr_sharpe"],
        **{key: details[key] for key in ("bmc_mean", "bmc_sharpe") if key in details},
        "n_groups": details["n_eras"],
        "split_id": manifest["split_id"],
    }


def paired_noise_band(
    challenger: dict[str, float],
    reference: dict[str, float],
    *,
    z: float = 2.0,
    block: int = 13,
    samples: int = 2000,
) -> tuple[float, float]:
    """Return (mean per-era difference, half-width of its noise band).

    Eras overlap in their 60-day labels, so the difference is resampled in
    contiguous blocks; the fixed seed gives every comparison the same answer.
    """
    eras = sorted(set(challenger) & set(reference))
    if not eras:
        raise ValueError("Noise band needs per-era scores on common eras")
    diff = np.array([challenger[era] - reference[era] for era in eras], dtype=float)
    block = max(1, min(block, len(diff)))
    block_means = np.convolve(diff, np.ones(block) / block, mode="valid")
    count = int(np.ceil(len(diff) / block))
    picks = np.random.default_rng(0).integers(0, len(block_means), size=(samples, count))
    return float(diff.mean()), float(z * block_means[picks].mean(axis=1).std(ddof=1))


def noise_verdict(agent, node, reference) -> str | None:
    """``better``, ``tied`` or ``worse`` than ``reference`` on the Host's series.

    None when either node has no Host series, so the caller keeps its own test.
    """
    cfg = getattr(getattr(agent, "cfg", None), "host_selection", None)
    z = float(getattr(cfg, "noise_band_z", 0.0) or 0.0)
    if not getattr(cfg, "enabled", False) or z <= 0:
        return None
    series = [(getattr(n, "host_selection", None) or {}).get("per_era_score") for n in (node, reference)]
    if not all(series):
        return None
    difference, band = paired_noise_band(
        series[0], series[1], z=z, block=int(cfg.noise_band_block), samples=int(cfg.noise_band_samples)
    )
    if difference > band:
        return "better"
    return "worse" if difference < -band else "tied"


def code_fingerprint(code: str) -> str:
    """Digest of the program itself: comments and layout do not change it."""
    try:
        text = ast.dump(ast.parse(code))
    except SyntaxError:
        text = code
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def close_duplicate_code(agent, node, logger) -> bool:
    """Close a node that repeats an earlier program; return True to skip its run."""
    cfg = getattr(agent.cfg, "host_selection", None)
    if not getattr(cfg, "enabled", False) or not getattr(cfg, "block_duplicate_code", False):
        return False
    if not (node.code or "").strip():
        return False
    fingerprint = code_fingerprint(node.code)
    with agent.journal_lock:
        earlier = [other for other in agent.journal.nodes if other is not node and (other.code or "").strip()]
    match = next((other for other in earlier if code_fingerprint(other.code) == fingerprint), None)
    if match is None:
        return False
    from utils.metric import WorstMetricValue

    node.is_buggy = True
    node.is_valid = False
    node.metric = WorstMetricValue()
    node.is_terminal = True
    node.continue_improve = False
    node.host_selection = {"duplicate_of": match.id}
    node.analysis = f"HOST_DUPLICATE_CODE: same program as node {match.id}; it was not run again"
    node._term_out = [node.analysis]
    node.exec_time = 0.0
    logger.warning("Node %s closed before running: same code as node %s", node.id, match.id)
    return True


def apply_to_node(agent, node, logger) -> None:
    """Replace a successful node's self-reported metric with the Host score."""
    cfg = getattr(agent.cfg, "host_selection", None)
    if not getattr(cfg, "enabled", False) or node.is_buggy:
        return
    from utils.metric import MetricValue, WorstMetricValue

    reported = node.metric.value if node.metric is not None else None
    try:
        path = predictions_path(agent.cfg.workspace_dir, node.id, Path(cfg.manifest_path))
        if not path.exists():
            raise ValueError(f"missing selection predictions ./submission/{path.name}")
        result = score_selection(
            Path(cfg.manifest_path),
            path,
            windows=int(cfg.windows),
            objective=str(cfg.objective),
            std_penalty=float(cfg.std_penalty),
            score_metric=str(getattr(cfg, "score_metric", "corr")),
            corr_weight=float(getattr(cfg, "corr_weight", 0.5)),
            bmc_weight=float(getattr(cfg, "bmc_weight", 2.0)),
        )
    except Exception as error:
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = f"HOST_SELECTION_ERROR: {error}"
        node._term_out.append(f"\n{node.analysis}")
        logger.warning("Node %s failed Host selection scoring: %s", node.id, error)
        return
    copy_threshold = getattr(cfg, "copy_threshold", None)
    similarity = result["benchmark_similarity"]
    is_copy = copy_threshold is not None and similarity is not None and similarity >= copy_threshold
    if getattr(getattr(agent.cfg, "oof", None), "enabled", False) and not is_copy:
        from fixed_holdout.oof import add_folds

        def score_fold(manifest, submission):
            return score_selection(
                manifest, submission, windows=1, objective="mean", std_penalty=0.0,
                score_metric=str(getattr(cfg, "score_metric", "corr")),
                corr_weight=float(getattr(cfg, "corr_weight", 0.5)),
                bmc_weight=float(getattr(cfg, "bmc_weight", 2.0)),
            )

        try:
            add_folds(agent, node, result, score_fold, logger)
        except Exception as error:
            node.is_buggy = True
            node.metric = WorstMetricValue()
            node.analysis = (
                "HOST_OOF_ERROR: the Host re-ran this program unchanged on an earlier period "
                f"and it failed: {error}"
            )
            node._term_out.append(f"\n{node.analysis}")
            logger.warning("Node %s failed the Host OOF re-run: %s", node.id, str(error)[:300])
            return
    result["agent_reported_metric"] = reported
    node.host_selection = result
    node.metric = MetricValue(result["score"], maximize=True)
    if configure_feedback(agent.cfg):
        hide_host_numbers(result)
        reference = _scored_ancestor(node)
        result["verdict_vs_parent"] = noise_verdict(agent, node, reference) if reference is not None else None
    logger.info(
        "Node %s Host selection score=%.5f (%s; agent reported %s; windows=%s; corr=%.5f bmc=%s; "
        "bench_sim=%s)",
        node.id, result["score"], result["score_metric"], reported,
        [round(v, 4) for v in result["window_means"]], result["corr_mean"],
        f"{result['bmc_mean']:.5f}" if "bmc_mean" in result else "n/a",
        f"{result['benchmark_similarity']:.3f}" if result["benchmark_similarity"] is not None else "n/a",
    )
    if is_copy:
        # A copy earns nothing and ends its lineage: check_improvement closes it.
        result["benchmark_copy"] = True
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.is_terminal = True
        node.continue_improve = False
        node.analysis = "HOST_SELECTION_DUPLICATE: predictions duplicate a known reference"
        node._term_out.append(f"\n{node.analysis}")
        logger.warning("Node %s closed as a benchmark copy (similarity %.4f)", node.id, similarity)
