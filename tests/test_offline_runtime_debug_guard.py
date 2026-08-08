from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.debug_agent import (  # noqa: E402
    _offline_runtime_repair_enabled,
    _remote_runtime_asset_markers,
    _requires_offline_model_repair,
    _runtime_recovery_guidance,
)


def test_offline_runtime_repair_guidance_and_marker_guard() -> None:
    agent = SimpleNamespace(
        cfg=SimpleNamespace(
            external_skill_memory=SimpleNamespace(
                experiment_r_offline_runtime_only=True
            )
        )
    )
    parent = SimpleNamespace(
        exc_type="FileNotFoundError",
        term_out="hubconf.py missing from torch.hub.load",
        analysis="local DINO repository is absent",
    )

    assert _offline_runtime_repair_enabled(agent)
    assert _requires_offline_model_repair(parent)
    markers = _remote_runtime_asset_markers(
        'torch.hub.load("facebookresearch/dinov3", "dinov3_vitl16")'
    )
    assert "torch.hub.load(" in markers

    guidance = " ".join(
        _runtime_recovery_guidance(
            parent,
            allow_remote_assets=False,
            offline_runtime_only=True,
        )
    )
    assert "forbidden" in guidance
    assert "torch.hub.load" in guidance

