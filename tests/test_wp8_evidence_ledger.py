from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = ROOT / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_wp8_evidence_ledger import (  # noqa: E402
    _payload_hash,
    _sha256_file,
    build_evidence_ledger,
)
from verify_wp8_evidence_ledger import verify_evidence_ledger  # noqa: E402


POLICY = ROOT / "coordination" / (
    "decision_admissibility_wp8_tier2_formal_"
    "analysis_policy_addendum_20260723_r1.json"
)
INVENTORY = ROOT / "coordination" / (
    "decision_admissibility_wp8_tier2_formal_joint_inventory_20260723_r1"
)
STATISTICS = ROOT / "coordination" / (
    "decision_admissibility_wp8_tier2_formal_statistics_20260723_r1"
)


def _kwargs() -> dict[str, Path | str]:
    return {
        "repo_root": ROOT,
        "analysis_policy_path": POLICY,
        "joint_inventory_root": INVENTORY,
        "statistics_root": STATISTICS,
        "created_at": "2026-07-23T16:00:00Z",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_wp8_evidence_ledger_separates_completion_effect_and_causality(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ledger"
    ledger = build_evidence_ledger(output_root=output, **_kwargs())
    by_id = {row["claim_id"]: row for row in ledger["claims"]}
    assert ledger["headline_effect_claim_authorized"] is False
    assert ledger["paper_claims_binding"]["path"] == (
        "papers/runforest_iclr2025/evidence/claims.md"
    )
    assert by_id["WP8-C1-FORMAL-EXECUTION"]["status"] == "supported"
    assert by_id["WP8-C2-RESULT-WRITEBACK"]["metrics"]["fixed_holdout_orphans"] == 0
    assert by_id["WP8-C3-FULL-SUPERIORITY"]["status"] == "rejected"
    assert by_id["WP8-C4-CONDITIONAL-UTILITY"]["status"] == "diagnostic"
    assert by_id["WP8-C4-CONDITIONAL-UTILITY"]["metrics"]["wins"] == 4
    assert by_id["WP8-C6-EXPERIENCE-CAUSALITY"]["status"] == "pending"
    assert all(
        all(
            {"path", "file_sha256"} <= set(binding)
            for binding in claim["artifact_bindings"]
        )
        for claim in ledger["claims"]
    )
    assert all(
        not (output / filename).stat().st_mode & stat.S_IWUSR
        for filename in ("evidence_ledger.json", "claims_addendum.md", "manifest.json")
    )
    verification = verify_evidence_ledger(
        ledger_root=output,
        repo_root=ROOT,
        analysis_policy_path=POLICY,
        joint_inventory_root=INVENTORY,
        statistics_root=STATISTICS,
    )
    assert verification["verified"] is True
    assert verification["errors"] == []


def test_wp8_evidence_ledger_verifier_rejects_self_rehashed_claim_laundering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ledger"
    build_evidence_ledger(output_root=output, **_kwargs())
    ledger_path = output / "evidence_ledger.json"
    manifest_path = output / "manifest.json"
    ledger_path.chmod(0o644)
    manifest_path.chmod(0o644)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    claim = next(
        row for row in ledger["claims"] if row["claim_id"] == "WP8-C3-FULL-SUPERIORITY"
    )
    claim["status"] = "supported"
    claim["claim_gate"]["effect_claim_authorized"] = True
    claim["claim_hash"] = _payload_hash(claim, "claim_hash")
    ledger["headline_effect_claim_authorized"] = True
    ledger["ledger_hash"] = _payload_hash(ledger, "ledger_hash")
    _write_json(ledger_path, ledger)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["evidence_ledger.json"] = _sha256_file(ledger_path)
    manifest["ledger_hash"] = ledger["ledger_hash"]
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    _write_json(manifest_path, manifest)
    verification = verify_evidence_ledger(
        ledger_root=output,
        repo_root=ROOT,
        analysis_policy_path=POLICY,
        joint_inventory_root=INVENTORY,
        statistics_root=STATISTICS,
    )
    assert verification["verified"] is False
    assert "superiority_status" in verification["errors"]
    assert "headline_claim_gate" in verification["errors"]
    assert "ledger_recompute_mismatch" in verification["errors"]


def test_wp8_evidence_ledger_refuses_output_reuse(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        build_evidence_ledger(output_root=output, **_kwargs())
