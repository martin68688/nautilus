from __future__ import annotations

from pathlib import Path

from authority.models import GenerationStage, Operation
from tests.authority.sop_visibility_helpers import visibility_request
from tests.authority.test_legacy_sop_visibility import LEGACY_PROTOCOL, legacy_layer


ROOT = Path(__file__).resolve().parents[2]
WP3_REPORT = ROOT / "coordination" / (
    "decision_admissibility_wp3_report_20260719.md"
)


def test_visibility_migration_reports_latency_tokens_and_empty_pack_without_gate(
    legacy_layer,
) -> None:
    debug = legacy_layer.visibility_gateway.migration_report(
        visibility_request(
            LEGACY_PROTOCOL,
            Operation.DEBUG_HYPOTHESIS,
            generation_stage=GenerationStage.DEBUG,
            token_budget=10_000_000,
            task_id="legacy-inspection",
            task_family="general",
        )
    )
    rank = legacy_layer.visibility_gateway.migration_report(
        visibility_request(
            LEGACY_PROTOCOL,
            Operation.RANK,
            generation_stage=GenerationStage.IMPROVE,
            token_budget=10_000_000,
            task_id="legacy-inspection",
            task_family="general",
        )
    )

    for report in (debug, rank):
        assert isinstance(report["latency_ms"], float)
        assert report["latency_ms"] >= 0.0
        assert isinstance(report["rendered_token_count"], int)
        assert report["rendered_token_count"] >= 0
        assert isinstance(report["empty_pack"], bool)
    assert debug["rendered_token_count"] > 0
    assert debug["empty_pack"] is False
    assert rank["rendered_token_count"] == 0
    assert rank["empty_pack"] is True

    frozen_report = WP3_REPORT.read_text(encoding="utf-8")
    assert "Rendered tokens | Empty pack | Cold evaluation latency" in frozen_report
    assert "These latency values are a local snapshot, not a post-hoc pilot threshold." in (
        frozen_report
    )
    assert "without choosing a\n      favorable threshold after observation" in frozen_report
