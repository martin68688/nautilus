from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_multigeneration_stop_gate import _render_markdown  # noqa: E402
from schema import sha256_json  # noqa: E402


def _report() -> dict:
    report = {
        "status": "pass",
        "created_at": "2026-07-21T00:00:00+08:00",
        "next_authorized_phase": "WP8 Tier-2 canary",
        "tier2_canary_authorized": True,
        "large_scale_tier2_authorized": False,
        "kill_gates": {
            name: {"status": "pass"}
            for name in (
                "gate_1",
                "gate_2",
                "gate_3",
                "gate_4",
                "gate_5",
                "gate_6",
            )
        },
        "statistical_summary": {
            "source_pair_count": 60,
            "source_run_count": 41,
            "source_task_count": 13,
            "domain_counts": {
                "audio": 12,
                "image": 13,
                "nlp": 13,
                "tabular": 12,
                "temporal": 10,
            },
            "generation_count": 5,
            "paraphrase_replicate_count": 3,
            "request_count": 900,
            "system_count": 5,
            "system_receipt_count": 4500,
            "final_generation": {
                "full": {
                    "laundering_numerator": 0,
                    "vkr_numerator": 180,
                    "denominator": 180,
                },
                "lineage_only": {
                    "laundering_numerator": 0,
                    "vkr_numerator": 180,
                    "denominator": 180,
                },
                "unrestricted": {
                    "laundering_numerator": 180,
                    "vkr_numerator": 180,
                    "denominator": 180,
                },
                "authority_only": {
                    "laundering_numerator": 70,
                    "vkr_numerator": 110,
                    "denominator": 180,
                },
                "global_validity_bit": {
                    "laundering_numerator": 0,
                    "vkr_numerator": 0,
                    "denominator": 180,
                },
            },
            "full_vs_unrestricted_laundering_reduction_ci_95": [1.0, 1.0],
            "full_vs_authority_laundering_reduction_ci_95": [0.27, 0.51],
            "full_vs_global_vkr_delta_ci_95": [1.0, 1.0],
            "bootstrap_iterations": 20_000,
            "holm_adjusted_p_values": {"z_test": 0.01, "a_test": 0.001},
        },
        "stop_gate_checks": {"z_last": True, "a_first": True},
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


def test_multigeneration_stop_gate_markdown_is_stable_after_json_roundtrip() -> None:
    report = _report()
    rendered = _render_markdown(report)
    roundtripped = json.loads(json.dumps(report, sort_keys=True))

    assert rendered == _render_markdown(roundtripped)
    assert rendered.index("`a_first`") < rendered.index("`z_last`")
    assert "does not claim that Full outperforms lineage-only" in rendered
    assert "Large-scale Tier-2 authorized: `false`" in rendered


def test_multigeneration_stop_gate_report_hash_covers_authorization_boundary() -> None:
    report = _report()
    original_hash = report["report_hash"]
    report["large_scale_tier2_authorized"] = True
    changed_hash = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )

    assert changed_hash != original_hash
