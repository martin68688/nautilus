from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from authority.ledger import AuthorityLedger
from authority.rollout import BundleRollbackController
from tests.test_memory_snapshot_overlay import build_tiny_bundle, write_current


def test_bundle_rollback_is_atomic_hash_verified_and_preserves_history(
    tmp_path,
) -> None:
    previous_bundle, previous_manifest = build_tiny_bundle(tmp_path, "v1")
    current_bundle, current_manifest = build_tiny_bundle(tmp_path, "v2")
    write_current(tmp_path, current_bundle, current_manifest)
    ledger = AuthorityLedger(tmp_path / "authority_events.jsonl")
    ledger.append(
        "bundle_publication_committed",
        {
            "bundle_id": current_manifest["bundle_id"],
            "manifest_sha256": current_manifest["manifest_sha256"],
        },
    )
    controller = BundleRollbackController(tmp_path, ledger=ledger)

    with pytest.raises(ValueError, match="CURRENT changed"):
        controller.rollback(
            target_bundle_path=previous_bundle,
            expected_current_manifest_sha256="f" * 64,
        )
    assert json.loads((tmp_path / "CURRENT.json").read_text())["bundle_id"] == (
        current_manifest["bundle_id"]
    )

    report = controller.rollback(
        target_bundle_path=previous_bundle,
        expected_current_manifest_sha256=current_manifest["manifest_sha256"],
    )
    pointer = json.loads((tmp_path / "CURRENT.json").read_text())

    assert pointer["bundle_id"] == previous_manifest["bundle_id"]
    assert pointer["manifest_sha256"] == previous_manifest["manifest_sha256"]
    assert report["from_bundle_id"] == current_manifest["bundle_id"]
    assert report["to_bundle_id"] == previous_manifest["bundle_id"]
    assert previous_bundle.is_dir()
    assert current_bundle.is_dir()
    assert ledger.verify() is True
    assert [event["event_type"] for event in ledger.read()][-2:] == [
        "bundle_rollback_prepared",
        "bundle_rollback_committed",
    ]


def test_bundle_rollback_cli_uses_compare_and_swap(tmp_path) -> None:
    previous_bundle, previous_manifest = build_tiny_bundle(tmp_path, "v1")
    current_bundle, current_manifest = build_tiny_bundle(tmp_path, "v2")
    write_current(tmp_path, current_bundle, current_manifest)
    ledger_path = tmp_path / "authority_events.jsonl"
    AuthorityLedger(ledger_path).append(
        "bundle_publication_committed",
        {
            "bundle_id": current_manifest["bundle_id"],
            "manifest_sha256": current_manifest["manifest_sha256"],
        },
    )
    report_path = tmp_path / "rollback-report.json"
    script = (
        Path(__file__).resolve().parents[2]
        / "paper-skills"
        / "memory_bundle"
        / "rollback_memory_bundle.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bundle-root",
            str(tmp_path),
            "--target-bundle",
            str(previous_bundle),
            "--expected-current-manifest-sha256",
            current_manifest["manifest_sha256"],
            "--ledger",
            str(ledger_path),
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads((tmp_path / "CURRENT.json").read_text())["bundle_id"] == (
        previous_manifest["bundle_id"]
    )
    assert json.loads(report_path.read_text())["to_bundle_id"] == (
        previous_manifest["bundle_id"]
    )
    assert current_bundle.is_dir()
