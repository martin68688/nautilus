from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
VERIFY_PATH = TOOLS / "verify_tier2_formal_staging_retry_amendment.py"
SPEC = importlib.util.spec_from_file_location("staging_retry_verifier", VERIFY_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
verify_staging_retry_amendment = MODULE.verify_staging_retry_amendment

AMENDMENT = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260723_r8.json"
)
STAGER = (
    ROOT
    / "deploy"
    / "stage_decision_admissibility_wp8_tier2_formal_continuation_retry.sh"
)
PIPELINE = (
    ROOT / "deploy" / "run_decision_admissibility_wp8_tier2_formal_"
    "continuation_retry_staging_pipeline.sh"
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
    path = tmp_path / "staging-retry-amendment.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def test_r8_staging_retry_amendment_verifies() -> None:
    report = verify_staging_retry_amendment(AMENDMENT, repo_root=ROOT)

    assert report["verified"] is True
    assert report["errors"] == []
    assert report["check_count"] == report["passed_check_count"]
    assert report["checks"]["failure_result_blind"] is True
    assert report["checks"]["runtime_source_unchanged"] is True
    assert report["checks"]["retry_roots_fresh"] is True


def test_r8_rejects_score_inspection_runtime_change_or_root_reuse(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["scientific_objective"]["terminal_score_values_inspected"] = True
    payload["control_packaging_correction"]["runtime_source_changed"] = True
    payload["retry_overrides"][
        "source_root"
    ] = "/workspace/decision-admissibility-wp8-tier2-formal-source-r11"

    report = verify_staging_retry_amendment(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "scores_uninspected" in report["errors"]
    assert "runtime_source_unchanged" in report["errors"]
    assert "retry_overrides_exact" in report["errors"]
    assert "retry_roots_fresh" in report["errors"]


def test_retry_stager_uses_fresh_roots_and_complete_control_archive() -> None:
    stager = STAGER.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    for value in (
        "formal-source-r12",
        "formal-control-r12",
        "formal-staging-r14",
        "formal-runs-r12",
        "formal-staging-r14-stop-gate-r1",
        "formal-staging-r14-pipeline-r1",
        "formal-stager-cpu-r14",
    ):
        assert value in stager
    for path in (
        "stage_decision_admissibility_wp8_tier2_formal.sh",
        "run_decision_admissibility_wp8_tier2_formal_staging_pipeline.sh",
        "devpod-decision-admissibility-wp8-tier2-formal-recovery-cpu-r1.yaml",
        "devpod-decision-admissibility-wp8-tier2-formal-recovered-evaluator-cpu-r1.yaml",
    ):
        assert path in stager
    assert "verify_tier2_formal_continuation_retry_staging.py" in pipeline
    assert (
        "decision_admissibility_wp8_tier2_formal_preregistration_20260723_r8.json"
        in pipeline
    )
    assert "kind: Job" not in stager
    assert "kubectl create" not in stager
    assert "git add" not in stager
    assert "git commit" not in stager
    assert "git push" not in stager
