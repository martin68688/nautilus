from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MLEVOLVE = ROOT / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.code_review_agent import (  # noqa: E402
    _submission_placeholder_safety_guidance,
)


def test_sample_submission_targets_are_not_assumed_to_be_class_registry():
    guidance = " ".join(_submission_placeholder_safety_guidance()).lower()

    assert "placeholders" in guidance
    assert "exhaustive class registry" in guidance
    assert "training-derived target vocabulary" in guidance
    assert "do not add an equality assertion" in guidance
