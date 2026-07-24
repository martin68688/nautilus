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

from verify_tier2_formal_postfailure_amendment import (  # noqa: E402
    verify_postfailure_amendment,
)


AMENDMENT = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260723_r5.json"
)


def _payload() -> dict:
    return json.loads(AMENDMENT.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    unsigned = {
        key: value for key, value in payload.items() if key != "amendment_hash"
    }
    payload["amendment_hash"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "postfailure-amendment.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_r5_postfailure_amendment_verifies_and_binds_r8() -> None:
    report = verify_postfailure_amendment(AMENDMENT, repo_root=ROOT)

    assert report["verified"] is True
    assert report["errors"] == []
    assert report["check_count"] == report["passed_check_count"]
    assert report["checks"]["failure_terminal_metric_observed"] is True
    assert report["checks"]["primary_contrast_unchanged"] is True
    assert report["checks"]["fit_semantics_separated"] is True


def test_r5_rejects_score_inspection_or_r8_reuse(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["failed_formal_attempt"][
        "score_values_inspected_for_this_amendment"
    ] = True
    payload["failed_formal_attempt"]["reuse_for_formal_execution"] = True

    report = verify_postfailure_amendment(
        _write(tmp_path, payload), repo_root=ROOT
    )

    assert report["verified"] is False
    assert "failure_score_values_uninspected" in report["errors"]
    assert "failure_not_reused" in report["errors"]


def test_r5_rejects_post_metric_primary_contrast_change(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["scientific_objective"]["primary_contrast"] = (
        "full_decision_admissibility minus flat_relevance_memory"
    )
    payload["scope"]["primary_contrast_changed"] = True

    report = verify_postfailure_amendment(
        _write(tmp_path, payload), repo_root=ROOT
    )

    assert report["verified"] is False
    assert "primary_contrast_unchanged" in report["errors"]
    assert "scope_primary_contrast_changed" in report["errors"]


def test_r5_rejects_fit_scope_conflation(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["implementation_correction"]["fit_scope_non_equivalence"] = (
        "train_view_only is fold_train_only"
    )

    report = verify_postfailure_amendment(
        _write(tmp_path, payload), repo_root=ROOT
    )

    assert report["verified"] is False
    assert "fit_semantics_separated" in report["errors"]
