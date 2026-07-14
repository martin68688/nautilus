"""Small helpers for the opt-in fixed-holdout runtime mode."""

from pathlib import Path
from typing import Any


EVALUATION_MODE = "terminal_only"


def _config(cfg: Any) -> Any:
    return getattr(cfg, "fixed_holdout", None)


def enabled(cfg: Any) -> bool:
    return bool(getattr(_config(cfg), "enabled", False))


def bypass_protocol_gates(cfg: Any) -> bool:
    fixed_cfg = _config(cfg)
    return enabled(cfg) and bool(
        getattr(fixed_cfg, "bypass_protocol_gates", False)
    )


def train_manifest_path(cfg: Any) -> Path:
    raw_path = str(getattr(_config(cfg), "train_manifest_path", "") or "")
    if not raw_path:
        raise ValueError("fixed_holdout.train_manifest_path must be configured")
    return Path(raw_path).expanduser().resolve()
