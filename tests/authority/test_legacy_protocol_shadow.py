from __future__ import annotations

from types import SimpleNamespace

from agents.result_parse_agent import _legacy_ast_mode, _shadow_only_audit


def test_host_mode_turns_legacy_audit_into_observation_only() -> None:
    original = {
        "status": "blocked",
        "hard_block": True,
        "repair_required": True,
        "rank_eligible": False,
        "metric_disposition": "reject",
        "issues": [{"issue_code": "DATA_SCOPE_LEAK"}],
    }
    shadow = _shadow_only_audit(original)
    assert shadow["status"] == "clean"
    assert shadow["hard_block"] is False
    assert shadow["repair_required"] is False
    assert shadow["execution_disposition"] == "allow"
    assert shadow["enforcement_mode"] == "shadow"
    assert shadow["legacy_shadow_observation"] == original
    assert original["status"] == "blocked"


def test_legacy_audit_remains_enforcing_without_host_activation() -> None:
    disabled = SimpleNamespace(
        acfg=SimpleNamespace(
            protocol_preflight=SimpleNamespace(enabled=False, legacy_ast_mode="shadow")
        )
    )
    enabled = SimpleNamespace(
        acfg=SimpleNamespace(
            protocol_preflight=SimpleNamespace(enabled=True, legacy_ast_mode="shadow")
        )
    )
    assert _legacy_ast_mode(disabled) == "enforce"
    assert _legacy_ast_mode(enabled) == "shadow"
