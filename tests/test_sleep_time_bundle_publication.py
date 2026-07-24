from __future__ import annotations

import json
import shutil
from pathlib import Path

from authority.bundle_publisher import SleepTimePipeline, SleepTimePublisher
from authority.memory_snapshot import (
    MemorySnapshotLoader,
    SessionOverlay,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from tests.test_memory_snapshot_overlay import (
    PROTOCOL,
    build_tiny_bundle,
    write_current,
)


def _rebuild_manifest(candidate: Path, *, version: str, parent_bundle_id: str) -> None:
    manifest_path = candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["bundle_id"] = f"tiny-{version}"
    manifest["bundle_version"] = version
    manifest["parent_bundle"] = parent_bundle_id
    manifest["created_at"] = "2026-07-19T01:00:00Z"
    artifact_hashes = {}
    for path in sorted(candidate.rglob("*")):
        if path.is_file() and path != manifest_path:
            artifact_hashes[str(path.relative_to(candidate))] = sha256_file(path)
    manifest["artifact_hashes"] = artifact_hashes
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    write_json_atomic(manifest_path, manifest)


def passing_pipeline(version: str) -> SleepTimePipeline:
    passed = lambda _context: {"status": "passed"}

    def build(context):
        candidate = context["candidate_dir"]
        shutil.copytree(
            context["parent_bundle_path"], candidate, dirs_exist_ok=True
        )
        shutil.copytree(
            context["overlay_snapshot"],
            candidate / "session_overlay_snapshot",
        )
        _rebuild_manifest(
            candidate,
            version=version,
            parent_bundle_id=context["parent_bundle"].bundle_id,
        )

    return SleepTimePipeline(
        audit=passed,
        claim_decomposition=passed,
        distillation=passed,
        build_candidate=build,
        derivation_validation=passed,
        visibility_validation=passed,
        bundle_validation=lambda context: {
            "valid": (context["candidate_dir"] / "manifest.json").is_file()
        },
    )


def setup_root(tmp_path: Path):
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    overlay = SessionOverlay(tmp_path / "session-overlay")
    overlay.append(
        "diagnostic",
        {"claim_type": "audit_finding", "audited": True},
        created_at="2026-07-19T00:30:00Z",
    )
    return bundle, manifest, overlay


def _tree_hash(root: Path) -> str:
    return sha256_json(
        {
            str(path.relative_to(root)): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }
    )


def test_sleep_time_publication_updates_current_only_after_all_validation(
    tmp_path: Path,
) -> None:
    parent, parent_manifest, overlay = setup_root(tmp_path)
    parent_tree_hash = _tree_hash(parent)
    publisher = SleepTimePublisher(tmp_path)
    report = publisher.publish(
        new_version="v2",
        overlay=overlay,
        pipeline=passing_pipeline("v2"),
        expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
    )

    current = json.loads((tmp_path / "CURRENT.json").read_text(encoding="utf-8"))
    assert current["bundle_version"] == "v2"
    assert current["parent_bundle"] == "tiny-v1"
    assert current["manifest_sha256"] == report.bundle_manifest_sha256
    assert report.parent_manifest_sha256 == parent_manifest["manifest_sha256"]
    assert report.overlay_manifest_sha256 == overlay.manifest["manifest_sha256"]
    assert set(report.pipeline_reports) == {
        "audit",
        "claim_decomposition",
        "distillation",
        "derivation_validation",
        "visibility_validation",
        "bundle_validation",
    }
    assert _tree_hash(parent) == parent_tree_hash
    assert (tmp_path / "bundles" / "v2" / "session_overlay_snapshot" / "events.jsonl").is_file()

    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "next-session-overlay",
        active_protocol_ref=PROTOCOL,
        authority_policy_version="authority_v1",
    )
    assert snapshot.base_bundle.bundle_version == "v2"
    assert snapshot.base_bundle.manifest["parent_bundle"] == "tiny-v1"


def test_old_bundle_remains_available_for_rollback(tmp_path: Path) -> None:
    parent, parent_manifest, overlay = setup_root(tmp_path)
    SleepTimePublisher(tmp_path).publish(
        new_version="v2",
        overlay=overlay,
        pipeline=passing_pipeline("v2"),
        expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
    )
    assert parent.is_dir()
    assert (parent / "manifest.json").is_file()
    assert (tmp_path / "bundles" / "v2" / "manifest.json").is_file()

