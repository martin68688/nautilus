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

from verify_tier2_formal_preterminal_recovery_amendment import (  # noqa: E402
    verify_preterminal_recovery_amendment,
)


AMENDMENT = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260723_r6.json"
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
    path = tmp_path / "preterminal-recovery-amendment.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_r6_preterminal_recovery_amendment_verifies() -> None:
    report = verify_preterminal_recovery_amendment(AMENDMENT, repo_root=ROOT)

    assert report["verified"] is True
    assert report["errors"] == []
    assert report["check_count"] == report["passed_check_count"]
    assert report["checks"]["diagnostic_terminal_unobserved"] is True
    assert report["checks"]["condition_disposition_exact"] is True
    assert report["checks"]["primary_contrast_unchanged"] is True


def test_r6_rejects_score_inspection_or_candidate_rerun(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["triggering_failure"]["terminal_score_values_inspected"] = True
    payload["preserved_block_recovery"]["candidate_or_agent_reexecution"] = True
    payload["analysis_integrity"]["full_condition_is_not_reexecuted"] = False

    report = verify_preterminal_recovery_amendment(
        _write(tmp_path, payload), repo_root=ROOT
    )

    assert report["verified"] is False
    assert "trigger_scores_uninspected" in report["errors"]
    assert "recovery_candidate_or_agent_reexecution" in report["errors"]
    assert "integrity_full_condition_is_not_reexecuted" in report["errors"]


def test_r6_rejects_promoting_the_denied_full_condition(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["preserved_block_recovery"]["required_condition_disposition"][
        "full_decision_admissibility"
    ] = "training_complete_unscored"

    report = verify_preterminal_recovery_amendment(
        _write(tmp_path, payload), repo_root=ROOT
    )

    assert report["verified"] is False
    assert "condition_disposition_exact" in report["errors"]


def test_r6_rejects_primary_contrast_or_finalizer_hash_change(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["scientific_objective"][
        "primary_contrast"
    ] = "full_decision_admissibility minus flat_relevance_memory"
    payload["implementation_correction"][
        "mlevolve/fixed_holdout/formal_block_training.py"
    ] = ("0" * 64)

    report = verify_preterminal_recovery_amendment(
        _write(tmp_path, payload), repo_root=ROOT
    )

    assert report["verified"] is False
    assert "primary_contrast_unchanged" in report["errors"]
    assert (
        "source_hash:mlevolve/fixed_holdout/formal_block_training.py"
        in report["errors"]
    )
