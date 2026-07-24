from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verify_tier2_formal_preregistration_amendment import (  # noqa: E402
    verify_amendment,
)


AMENDMENT = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r2.json"
)


def _payload() -> dict:
    return json.loads(AMENDMENT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    payload["amendment_hash"] = ""
    unsigned = {
        key: value for key, value in payload.items() if key != "amendment_hash"
    }
    import hashlib

    payload["amendment_hash"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path = tmp_path / "amendment.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_r2_amendment_verifies_and_preserves_r1_failure() -> None:
    report = verify_amendment(AMENDMENT, repo_root=ROOT)
    assert report["verified"] is True
    assert report["errors"] == []
    assert report["check_count"] == report["passed_check_count"]


def test_r2_amendment_rejects_manual_group_swapping(tmp_path: Path) -> None:
    payload = _payload()
    payload["overrides"]["tasks_by_id"]["mlsp-2013-birds"][
        "holdout_builder"
    ]["phase_b"] = "Manually swap groups until coverage passes."
    report = verify_amendment(_write(tmp_path, payload), repo_root=ROOT)
    assert report["verified"] is False
    assert "phase_b_frozen" in report["errors"]


def test_r2_amendment_rejects_tampered_feasibility_binding(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["pretraining_evidence"]["r2_feasibility"][
        "selected_group_ids_sha256"
    ] = "0" * 64
    report = verify_amendment(_write(tmp_path, payload), repo_root=ROOT)
    assert report["verified"] is False
    assert "r2_selected_groups_bound" in report["errors"]
