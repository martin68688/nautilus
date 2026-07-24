from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verify_tier2_formal_preregistration import (  # noqa: E402
    ONLINE_SYSTEMS,
    verify_preregistration,
)


PREREGISTRATION = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r1.json"
)


def _payload() -> dict:
    return json.loads(PREREGISTRATION.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "preregistration.json"
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def test_formal_preregistration_verifies_against_current_frozen_inputs() -> None:
    report = verify_preregistration(PREREGISTRATION, repo_root=ROOT)
    assert report["verified"] is True
    assert report["errors"] == []
    assert report["check_count"] == report["passed_check_count"]
    assert report["task_ids"] == [
        "aerial-cactus-identification",
        "mlsp-2013-birds",
        "new-york-city-taxi-fare-prediction",
    ]
    assert set(report["online_system_ids"]) == set(ONLINE_SYSTEMS)


def test_preregistration_rejects_noncounterbalanced_duplicate_order(
    tmp_path: Path,
) -> None:
    payload = _payload()
    blocks = payload["condition_order_design"]["blocks"]
    blocks[1]["order"] = copy.deepcopy(blocks[0]["order"])
    report = verify_preregistration(_write(tmp_path, payload), repo_root=ROOT)
    assert report["verified"] is False
    assert "orders_are_unique" in report["errors"]
    assert "position_balance_within_one" in report["errors"]


def test_preregistration_rejects_candidate_contract_drift(tmp_path: Path) -> None:
    payload = _payload()
    payload["shared_candidate_contract"]["max_epochs"] = 9
    report = verify_preregistration(_write(tmp_path, payload), repo_root=ROOT)
    assert report["verified"] is False
    assert any(
        error.startswith("candidate_contract_hash:")
        for error in report["errors"]
    )


def test_preregistration_rejects_terminal_best_as_system_selection(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["selection_and_terminal_evaluation"]["system_metric"] = (
        "Use terminal-best candidate as the system result."
    )
    report = verify_preregistration(_write(tmp_path, payload), repo_root=ROOT)
    assert report["verified"] is False
    assert "system_metric_not_terminal_best" in report["errors"]


def test_preregistration_rejects_direct_raw_bundle_enforce_use(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["memory_bundle_contract"]["formal_child_bundle_requirement"] = (
        "Mount the raw WP4 Bundle directly."
    )
    report = verify_preregistration(_write(tmp_path, payload), repo_root=ROOT)
    assert report["verified"] is False
    assert "raw_wp4_not_direct_enforce_memory" in report["errors"]
