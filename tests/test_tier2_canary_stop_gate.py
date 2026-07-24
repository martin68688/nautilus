from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_tier2_canary_stop_gate import (  # noqa: E402
    _verify_authority_ledger_read_only,
    render_markdown,
    sha256_json,
    valid_hash,
)
from authority.ledger import AuthorityLedger  # noqa: E402


def _report() -> dict:
    report = {
        "status": "pass",
        "report_hash": "",
        "next_authorized_phase": "WP8 Tier-2 formal experiment staging",
        "formal_tier2_authorized": True,
        "paper_effect_claim_authorized": False,
        "condition_summary": {
            "nm": {
                "best_score": 0.9,
                "scored_candidate_count": 3,
                "exposure_event_count": 0,
                "result_fact_count": 1,
            },
            "full": {
                "best_score": 0.91,
                "scored_candidate_count": 3,
                "exposure_event_count": 2,
                "result_fact_count": 1,
            },
        },
        "stop_gate_checks": {"z_last": True, "a_first": True},
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


def test_tier2_canary_markdown_is_stable_and_does_not_claim_effect() -> None:
    report = _report()
    rendered = render_markdown(report)
    roundtripped = json.loads(json.dumps(report, sort_keys=True))

    assert rendered == render_markdown(roundtripped)
    assert rendered.index("`a_first`") < rendered.index("`z_last`")
    assert "without an efficacy or Full-superiority claim" in rendered
    assert "Formal Tier-2 authorized: `true`" in rendered
    assert "Paper effect claim authorized: `false`" in rendered


def test_tier2_canary_report_hash_covers_authorization_boundary() -> None:
    report = _report()
    assert valid_hash(report, "report_hash")
    original = report["report_hash"]
    report["paper_effect_claim_authorized"] = True
    changed = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )

    assert changed != original
    assert not valid_hash(report, "report_hash")


def test_tier2_canary_report_hash_covers_raw_scores_without_interpreting_them() -> None:
    report = _report()
    original = report["report_hash"]
    report["condition_summary"]["full"]["best_score"] = 0.5
    changed = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )

    assert changed != original


def test_read_only_ledger_verification_does_not_create_lock_file(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "authority_events.jsonl"
    ledger = AuthorityLedger(ledger_path)
    ledger.append("test_event", {"value": 1})
    ledger.lock_path.unlink()
    before = sorted(path.name for path in tmp_path.iterdir())

    assert _verify_authority_ledger_read_only(ledger_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == before
    assert not ledger.lock_path.exists()
