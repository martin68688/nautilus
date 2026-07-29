from __future__ import annotations

import hashlib
import json
from pathlib import Path

from authority.memory_snapshot import MemorySnapshotLoader, verify_bundle_directory
from authority.protocol_execution_contract import read_contract_artifact
from protocol_runtime.data_views import verify_data_view_manifest
from protocol_runtime.events import hash_payload


ROOT = Path("coordination/protocol_reliability_pr8_staging_20260725")


def _read(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_pr8_local_staging_manifests_are_hash_bound_and_inactive() -> None:
    staging = _read("STAGING_MANIFEST.json")
    preregistration = _read("PREREGISTRATION.json")
    policy = _read("STATISTICAL_POLICY.json")
    blockers = _read("EXECUTION_BLOCKERS.json")
    artifact = _read("artifact_manifest.json")
    assert staging["staging_hash"] == hash_payload(staging, "staging_hash")
    assert preregistration["preregistration_hash"] == hash_payload(
        preregistration, "preregistration_hash"
    )
    assert policy["policy_hash"] == hash_payload(policy, "policy_hash")
    assert blockers["blocker_hash"] == hash_payload(blockers, "blocker_hash")
    assert artifact["manifest_hash"] == hash_payload(artifact, "manifest_hash")
    assert staging["status"] == "prepared_not_activated"
    assert blockers["kubernetes_authorized"] is False
    assert blockers["gpu_authorized"] is False
    assert blockers["terminal_score_observed"] is False


def test_pr8_three_task_contract_view_bundle_and_current_chains_verify(
    tmp_path: Path,
) -> None:
    staging = _read("STAGING_MANIFEST.json")
    assert staging["task_count"] == staging["child_bundle_count"] == 3
    for row in staging["tasks"]:
        task_root = ROOT / "tasks" / row["task_id"]
        contract = read_contract_artifact(
            task_root / "contract" / "PROTOCOL_EXECUTION_CONTRACT.json"
        )
        assert contract.contract_hash == row["contract_hash"]
        verification = verify_data_view_manifest(
            task_root / "data_views" / "DATA_VIEW_MANIFEST.json",
            contract=contract,
        )
        assert verification["status"] == "pass"
        bundle_root = task_root / "bundle_root"
        child = bundle_root / "bundles" / row["child_bundle_id"]
        manifest = verify_bundle_directory(child)
        assert manifest["manifest_sha256"] == row["child_bundle_manifest_hash"]
        base = MemorySnapshotLoader(bundle_root).load_base()
        assert base.bundle_id == row["child_bundle_id"]
        assert base.manifest["parent_bundle"] == row["base_bundle_id"]
        holdout = json.loads(
            (task_root / "TERMINAL_HOLDOUT_COMMITMENT.json").read_text(
                encoding="utf-8"
            )
        )
        assert holdout["commitment_hash"] == hash_payload(
            holdout, "commitment_hash"
        )
        assert holdout["training_mount_count"] == 0
        assert holdout["terminal_score_computed"] is False


def test_pr8_source_snapshot_and_reserved_output_roots_are_immutable_empty() -> None:
    source = _read("SOURCE_SNAPSHOT_MANIFEST.json")
    rows = [
        json.loads(line)
        for line in (ROOT / "SOURCE_SNAPSHOT.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line
    ]
    assert len(rows) == source["file_count"]
    for row in rows:
        frozen = ROOT / "source_snapshot" / "files" / row["path"]
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == row["sha256"]
        assert frozen.stat().st_mode & 0o222 == 0
    reservation = _read("OUTPUT_ROOT_RESERVATION.json")
    assert reservation["both_roots_empty"] is True
    assert not any((ROOT / reservation["reliability_root"]).iterdir())
    assert not any((ROOT / reservation["performance_root"]).iterdir())
    image = _read("CONTAINER_IMAGE_REQUIREMENT.json")
    assert image["status"] == "pending_build_and_authorization"
    assert image["container_built"] is False
    assert image["fake_digest_substituted"] is False
