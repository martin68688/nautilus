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

from verify_tier2_formal_claim_authority_amendment import (  # noqa: E402
    verify_claim_authority_amendment,
)


AMENDMENT = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r3.json"
)
AMENDMENT_R4 = (
    ROOT
    / "coordination"
    / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r4.json"
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
    path = tmp_path / "claim-authority-amendment.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _static_report(path: Path) -> dict:
    return verify_claim_authority_amendment(
        path,
        repo_root=ROOT,
        require_source_bundle=False,
    )


def test_r3_claim_authority_amendment_verifies_statically() -> None:
    report = _static_report(AMENDMENT)
    assert report["verified"] is True
    assert report["static_verified"] is True
    assert report["source_bundle_checked"] is False
    assert report["errors"] == []


def test_r4_replacement_verifies_and_binds_failed_r3() -> None:
    report = _static_report(AMENDMENT_R4)
    assert report["verified"] is True
    assert report["static_verified"] is True
    assert report["source_bundle_checked"] is False
    assert report["checks"]["r4_failed_check_exact"] is True
    assert report["checks"]["r4_only_metadata_correction"] is True
    assert report["errors"] == []


def test_r3_rejects_protocol_issue_widening(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    payload["overrides"]["memory_bundle_contract"][
        "birds_claim_specific_publication"
    ]["allowed_source_audit_issue_codes"].append("TEST_LABEL_ACCESS")
    report = _static_report(_write(tmp_path, payload))
    assert report["verified"] is False
    assert "only_protocol_issue_allowed" in report["errors"]


def test_r3_rejects_rank_permission_or_source_score_inheritance(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    birds = payload["overrides"]["memory_bundle_contract"][
        "birds_claim_specific_publication"
    ]
    birds["permitted_operations"].append("rank")
    birds["source_score_inheritance"] = True
    report = _static_report(_write(tmp_path, payload))
    assert report["verified"] is False
    assert "generate_only" in report["errors"]
    assert "source_score_inheritance_false" in report["errors"]


def test_r3_rejects_relabelling_blocked_sidecar_as_clean(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["pretraining_evidence"]["source_audit"]["status"] = "clean"
    payload["pretraining_evidence"]["source_audit"][
        "sidecar_relabelled_clean"
    ] = True
    payload["overrides"]["memory_bundle_contract"][
        "birds_claim_specific_publication"
    ]["source_sidecar_clean"] = True
    report = _static_report(_write(tmp_path, payload))
    assert report["verified"] is False
    assert "source_audit_blocked" in report["errors"]
    assert "source_sidecar_not_relabelled" in report["errors"]
    assert "source_sidecar_not_clean" in report["errors"]


def test_r3_formal_verification_requires_source_bundle() -> None:
    report = verify_claim_authority_amendment(
        AMENDMENT,
        repo_root=ROOT,
        require_source_bundle=True,
    )
    assert report["verified"] is False
    assert "source_bundle_supplied" in report["errors"]
