from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from engine.draft_roles import canonical_draft_role, is_novel_draft_role  # noqa: E402


def test_dual_novel_slots_share_behavior_but_keep_unique_identities() -> None:
    assert canonical_draft_role("novel_exploration_a") == "novel_exploration"
    assert canonical_draft_role("novel_exploration_b") == "novel_exploration"
    assert is_novel_draft_role("novel_exploration_a")
    assert is_novel_draft_role("novel_exploration_b")


def test_dual_novel_slots_have_no_cross_slot_diversity_constraint() -> None:
    source = (ROOT / "mlevolve" / "agents" / "draft_agent.py").read_text()
    assert "dual_novel_diversity" not in source
    assert "novel_slot_diversity" not in source
    assert "different code structure from another exploration slot" in source
