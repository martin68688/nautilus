from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verify_tier2_formal_continuation_amendment import (  # noqa: E402
    verify_continuation_amendment,
)


AMENDMENT = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260723_r7.json"
)


def _payload() -> dict:
    return json.loads(AMENDMENT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    unsigned = {key: value for key, value in payload.items() if key != "amendment_hash"}
    payload["amendment_hash"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "continuation-amendment.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_r7_continuation_amendment_verifies() -> None:
    report = verify_continuation_amendment(AMENDMENT, repo_root=ROOT)

    assert report["verified"] is True
    assert report["errors"] == []
    assert report["check_count"] == report["passed_check_count"]
    assert report["checks"]["remaining_blocks_exact"] is True
    assert report["checks"]["completed_blocks_cannot_rerun"] is True
    assert report["checks"]["primary_contrast_unchanged"] is True


def test_r7_rejects_completed_block_rerun_or_score_inspection(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["continuation_design"]["completed_blocks_may_reexecute"] = True
    payload["scientific_objective"]["terminal_score_values_inspected"] = True
    payload["analysis_integrity"]["no_completed_block_reexecuted"] = False

    report = verify_continuation_amendment(_write(tmp_path, payload), repo_root=ROOT)

    assert report["verified"] is False
    assert "completed_blocks_not_reexecuted" in report["errors"]
    assert "terminal_values_uninspected" in report["errors"]
    assert "integrity_no_completed_block_reexecuted" in report["errors"]


def test_r7_rejects_remaining_pair_or_primary_contrast_change(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["continuation_design"]["remaining_blocks"][0]["agent_seed"] = 999
    payload["scientific_objective"][
        "primary_contrast"
    ] = "full_decision_admissibility minus flat_relevance_memory"

    report = verify_continuation_amendment(_write(tmp_path, payload), repo_root=ROOT)

    assert report["verified"] is False
    assert "remaining_blocks_exact" in report["errors"]
    assert "primary_contrast_unchanged" in report["errors"]


def test_r7_rejects_unbound_continuation_source(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["implementation_correction"][
        "paper-skills/memory_bundle/build_tier2_formal_continuation_staging.py"
    ] = ("0" * 64)

    report = verify_continuation_amendment(_write(tmp_path, payload), repo_root=ROOT)

    assert report["verified"] is False
    assert (
        "source_hash:paper-skills/memory_bundle/"
        "build_tier2_formal_continuation_staging.py" in report["errors"]
    )
