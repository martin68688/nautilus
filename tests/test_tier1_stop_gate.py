from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_tier1_stop_gate import _render_markdown  # noqa: E402


def test_stop_gate_markdown_is_stable_across_sorted_json_roundtrip() -> None:
    report = {
        "status": "pass",
        "report_hash": "a" * 64,
        "created_at": "2026-07-21T00:00:00+08:00",
        "next_authorized_phase": "WP8 Multi-generation",
        "large_scale_tier2_authorized": False,
        "kill_gates": {
            "gate_1": {
                "status": "pass",
                "mismatch_count": 10,
                "decision_count": 20,
                "wilson_lower_95": 0.3,
            },
            "gate_2": {"status": "pass"},
            "gate_3": {
                "status": "pass",
                "action_difference_count": 4,
                "paired_decision_count": 20,
            },
            "gate_4": {"status": "pass"},
            "gate_5": {"status": "pending_next_phase"},
            "gate_6": {"status": "pass"},
        },
        "statistical_summary": {
            "raw_iir_numerator": 3,
            "raw_iir_denominator": 20,
            "raw_iir_bootstrap_ci_95": [0.1, 0.3],
            "vkr_numerator": 20,
            "vkr_denominator": 20,
            "gate3_action_difference_numerator": 4,
            "gate3_action_difference_denominator": 20,
            "gate3_bootstrap_ci_95": [0.1, 0.4],
            "gate4_downstream_holm_p": 0.25,
        },
        # Deliberately use the opposite of lexical order.
        "stop_gate_checks": {"z_last": True, "a_first": True},
    }

    rendered = _render_markdown(report)
    roundtripped = json.loads(json.dumps(report, sort_keys=True))

    assert rendered == _render_markdown(roundtripped)
    assert rendered.index("`a_first`") < rendered.index("`z_last`")
