from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verify_tier2_formal_precontract_retry_amendment import (  # noqa: E402
    verify_precontract_retry_amendment,
)


AMENDMENT = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260723_r9.json"
)
STAGER = (
    ROOT / "deploy" / "stage_decision_admissibility_wp8_tier2_formal_continuation_r5.sh"
)
PIPELINE = (
    ROOT / "deploy" / "run_decision_admissibility_wp8_tier2_formal_"
    "continuation_r5_staging_pipeline.sh"
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
    path = tmp_path / "precontract-r5-retry-amendment.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_r9_precontract_retry_amendment_verifies() -> None:
    report = verify_precontract_retry_amendment(AMENDMENT, repo_root=ROOT)

    assert report["verified"] is True
    assert report["errors"] == []
    assert report["check_count"] == report["passed_check_count"]
    assert report["checks"]["diagnostic_precontract"] is True
    assert report["checks"]["runtime_changes_exact"] is True
    assert report["checks"]["retry_design_exact"] is True


def test_r9_rejects_r4_reuse_or_scientific_change(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["retry_design"]["formal_execution_revision"] = "r4"
    payload["retry_design"][
        "output_root"
    ] = "/workspace/decision-admissibility-wp8-tier2-formal-runs-r12"
    payload["scientific_objective"]["terminal_score_values_inspected"] = True
    payload["scope"]["search_budgets_changed"] = True

    report = verify_precontract_retry_amendment(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "retry_design_exact" in report["errors"]
    assert "retry_roots_new" in report["errors"]
    assert "scientific_result_blind" in report["errors"]
    assert "scope_search_budgets_changed" in report["errors"]


def test_r5_stager_is_devpod_only_and_uses_fresh_roots() -> None:
    stager = STAGER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    for value in (
        "formal-source-r13",
        "formal-control-r13",
        "formal-staging-r15",
        "formal-runs-r13",
        "formal-staging-r15-stop-gate-r1",
        "formal-staging-r15-pipeline-r1",
        "formal-stager-cpu-r15",
    ):
        assert value in stager
    assert "FORMAL_PRECONTRACT_INFRASTRUCTURE_ABORT.json" in stager
    assert "build_tier2_formal_continuation_r5_staging.py" in pipeline
    assert "verify_tier2_formal_continuation_r5_staging.py" in pipeline
    assert "kind: Job" not in stager
    assert "kubectl create" not in stager
    assert "git add" not in stager
    assert "git commit" not in stager
    assert "git push" not in stager
