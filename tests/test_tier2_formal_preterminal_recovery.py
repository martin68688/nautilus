from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

import fixed_holdout.formal_preterminal_recovery as recovery_module
from fixed_holdout.common import sha256_file, write_json
from fixed_holdout.formal_preterminal_recovery import (
    AMENDMENT_SCHEMA,
    DIAGNOSTIC_SCHEMA,
    SOURCE_SCHEMA,
    VERIFICATION_SCHEMA,
    _inventory_hash,
    _tree_inventory,
    recover_preterminal_finalizer,
)
from tests.test_tier2_formal_block_finalizers import (
    _formal_fixture,
    _payload_hash,
)


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from write_tier2_formal_recovery_deletion import (  # noqa: E402
    SCHEMA as RECOVERY_DELETION_SCHEMA,
    write_recovery_deletion,
)


def _write_hashed(path: Path, payload: dict, field: str) -> dict:
    payload[field] = _payload_hash(payload, field)
    write_json(path, payload)
    return payload


def _recovery_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    fixture = _formal_fixture(
        tmp_path,
        enforced_protocol=True,
        blocked_protocol_condition="full_decision_admissibility",
    )
    output = fixture["output_root"]
    authority = output / "conditions" / "authority_only"
    (authority / "RUN_EXIT_CODE").write_text("1\n", encoding="utf-8")
    receipt_path = authority / "CONDITION_RUNTIME_RECEIPT.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["run_exit_code"] = 1
    receipt["receipt_hash"] = _payload_hash(receipt, "receipt_hash")
    write_json(receipt_path, receipt)
    (output / "STATE").write_text("training_launcher_failed\n", encoding="utf-8")
    (output / "TRAINING_LAUNCHER_EXIT_CODE").write_text("1\n", encoding="utf-8")
    (output / "TRAINING_LAUNCHER.log").write_text(
        "pre-terminal finalizer failure\n", encoding="utf-8"
    )

    contract = json.loads(fixture["block_contract_path"].read_text(encoding="utf-8"))
    recovery_root = tmp_path / "recovery-source"
    recovery_root.mkdir()
    source_root = tmp_path / "original-source"
    source_root.mkdir()
    full_run = next(
        (output / "conditions" / "full_decision_admissibility" / "runs").iterdir()
    )
    full_request = json.loads(
        (full_run / "fixed_holdout_evaluation_request.json").read_text(encoding="utf-8")
    )
    full_journal = json.loads((full_run / "journal.json").read_text(encoding="utf-8"))
    full_node = next(
        row
        for row in full_journal["nodes"]
        if str(row.get("id")) == str(full_request["selected_node_id"])
    )
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic = _write_hashed(
        diagnostic_path,
        {
            "schema": DIAGNOSTIC_SCHEMA,
            "block_id": contract["block_id"],
            "frozen_execution": {
                "source_snapshot_sha256": contract["source_snapshot_sha256"]
            },
            "failure": {
                "protocol_observation_reason": full_node["protocol_observation"][
                    "reason"
                ]
            },
            "diagnostic_hash": "",
        },
        "diagnostic_hash",
    )
    pre_inventory = _tree_inventory(output)
    amendment_path = tmp_path / "amendment.json"
    amendment = _write_hashed(
        amendment_path,
        {
            "schema": AMENDMENT_SCHEMA,
            "triggering_failure": {
                "block_id": contract["block_id"],
                "diagnostic_hash": diagnostic["diagnostic_hash"],
                "diagnostic_file_sha256": sha256_file(diagnostic_path),
            },
            "preserved_block_recovery": {
                "required_pre_recovery_file_count": len(pre_inventory),
                "required_pre_recovery_tree_sha256": _inventory_hash(pre_inventory),
                "required_condition_disposition": {
                    "full_decision_admissibility": (
                        "pre_terminal_failure:authority_denial"
                    ),
                    "authority_only": ("pre_terminal_failure:retained_run_failure"),
                    "no_memory": "training_complete_unscored",
                    "global_validity_bit": "training_complete_unscored",
                    "flat_relevance_memory": "training_complete_unscored",
                },
            },
            "amendment_hash": "",
        },
        "amendment_hash",
    )
    verification_path = tmp_path / "verification.json"
    verification = _write_hashed(
        verification_path,
        {
            "schema": VERIFICATION_SCHEMA,
            "verified": True,
            "errors": [],
            "amendment_file_sha256": sha256_file(amendment_path),
            "verification_hash": "",
        },
        "verification_hash",
    )
    source_manifest_path = recovery_root / ("WP8_TIER2_RECOVERY_SOURCE_MANIFEST.json")
    source_manifest = _write_hashed(
        source_manifest_path,
        {
            "schema": SOURCE_SCHEMA,
            "amendment_hash": amendment["amendment_hash"],
            "file_hashes": {},
            "manifest_hash": "",
        },
        "manifest_hash",
    )

    monkeypatch.setattr(
        recovery_module,
        "mount_points",
        lambda: {"/opt/nautilus", "/recovery", "/memory", "/output"},
    )
    monkeypatch.setattr(recovery_module, "environment_has_solver_secret", lambda: False)
    monkeypatch.setattr(recovery_module, "_assert_read_only", lambda root: None)
    monkeypatch.setattr(recovery_module, "_path_exists", lambda path: False)
    monkeypatch.setattr(
        recovery_module,
        "verify_source_snapshot",
        lambda *args, **kwargs: {
            "schema": "synthetic-source-verification-v1",
            "verified": True,
        },
    )
    return {
        **fixture,
        "source_root": source_root,
        "recovery_root": recovery_root,
        "amendment_path": amendment_path,
        "verification_path": verification_path,
        "diagnostic_path": diagnostic_path,
        "source_manifest_path": source_manifest_path,
        "amendment": amendment,
        "verification": verification,
        "source_manifest": source_manifest,
    }


def _recover(fixture: dict) -> dict:
    return recover_preterminal_finalizer(
        output_root=fixture["output_root"],
        block_contract_path=fixture["block_contract_path"],
        bundle_root=fixture["memory_root"],
        source_root=fixture["source_root"],
        recovery_root=fixture["recovery_root"],
        amendment_path=fixture["amendment_path"],
        amendment_verification_path=fixture["verification_path"],
        diagnostic_path=fixture["diagnostic_path"],
        recovery_source_manifest_path=fixture["source_manifest_path"],
        pod_name="synthetic-recovery-cpu",
        pod_namespace="ecepxie",
        pod_uid="synthetic-recovery-uid",
    )


def test_result_blind_recovery_retains_denials_without_reexecution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _recovery_fixture(tmp_path, monkeypatch)

    receipt = _recover(fixture)

    assert receipt["terminal_metric_observed"] is False
    assert receipt["agent_reexecuted"] is False
    assert receipt["candidate_code_reexecuted"] is False
    assert receipt["full_condition_reexecuted"] is False
    assert receipt["successful_condition_count"] == 3
    assert receipt["failed_condition_count"] == 2
    assert (
        receipt["condition_disposition"]["full_decision_admissibility"]
        == "pre_terminal_failure:authority_denial"
    )
    assert (fixture["output_root"] / "TRAINING_COMPLETE").is_file()
    assert (fixture["output_root"] / "RECOVERY_COMPLETE").is_file()

    deletion = write_recovery_deletion(
        fixture["output_root"],
        namespace="ecepxie",
        pod_name="synthetic-recovery-cpu",
        pod_uid="synthetic-recovery-uid",
        delete_requested_at="2026-07-23T02:00:00Z",
        not_found_verified_at="2026-07-23T02:00:01Z",
        not_found_probe_sha256="a" * 64,
    )
    assert deletion["schema"] == RECOVERY_DELETION_SCHEMA
    assert deletion["not_found_verified"] is True
    assert deletion["training_manifest_hash"] == receipt["training_manifest_hash"]


def test_recovery_rejects_any_pre_recovery_tree_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _recovery_fixture(tmp_path, monkeypatch)
    (fixture["output_root"] / "unregistered-change.txt").write_text(
        "changed\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="output tree changed"):
        _recover(fixture)
    assert not (fixture["output_root"] / "TRAINING_MANIFEST.json").exists()
