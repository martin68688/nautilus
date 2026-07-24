"""Small helpers for the opt-in fixed-holdout runtime mode."""

from pathlib import Path
from typing import Any


EVALUATION_MODE = "terminal_only"
INTERNAL_METRIC_DISPOSITION = "search_only"


def _config(cfg: Any) -> Any:
    return getattr(cfg, "fixed_holdout", None)


def enabled(cfg: Any) -> bool:
    return bool(getattr(_config(cfg), "enabled", False))


def bypass_protocol_gates(cfg: Any) -> bool:
    fixed_cfg = _config(cfg)
    return enabled(cfg) and bool(
        getattr(fixed_cfg, "bypass_protocol_gates", False)
    )


def search_only_candidate_selection(cfg: Any) -> bool:
    """Return whether target-node ordering is a provisional search operation.

    In fixed-holdout mode the candidate's internal validation metric is never
    terminal evidence.  It may order freshly executed target-task candidates
    before the label-isolated evaluator runs, but it cannot authorize a Result
    Fact, inherit a source score, or rank memory Claims.
    """

    fixed_cfg = _config(cfg)
    return bool(
        bypass_protocol_gates(cfg)
        and getattr(fixed_cfg, "evaluation_mode", EVALUATION_MODE)
        == EVALUATION_MODE
        and getattr(
            fixed_cfg,
            "internal_metric_disposition",
            INTERNAL_METRIC_DISPOSITION,
        )
        == INTERNAL_METRIC_DISPOSITION
    )


def train_manifest_path(cfg: Any) -> Path:
    raw_path = str(getattr(_config(cfg), "train_manifest_path", "") or "")
    if not raw_path:
        raise ValueError("fixed_holdout.train_manifest_path must be configured")
    return Path(raw_path).expanduser().resolve()
