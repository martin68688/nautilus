from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from authority.bundle_publisher import (
    PublicationConflictError,
    PublicationValidationError,
    SleepTimePipeline,
    SleepTimePublisher,
)
from authority.memory_snapshot import sha256_file
from tests.test_sleep_time_bundle_publication import (
    _tree_hash,
    passing_pipeline,
    setup_root,
)


def _current_bytes(root: Path) -> bytes:
    return (root / "CURRENT.json").read_bytes()


def test_validator_failure_does_not_change_current(tmp_path: Path) -> None:
    _parent, parent_manifest, overlay = setup_root(tmp_path)
    before = _current_bytes(tmp_path)
    pipeline = passing_pipeline("v2")
    pipeline.visibility_validation = lambda _context: {"status": "failed"}
    with pytest.raises(PublicationValidationError, match="visibility_validation"):
        SleepTimePublisher(tmp_path).publish(
            new_version="v2",
            overlay=overlay,
            pipeline=pipeline,
            expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
        )
    assert _current_bytes(tmp_path) == before
    assert not (tmp_path / "bundles" / "v2").exists()
    assert len(list(tmp_path.glob(".failed-v2-*"))) == 1


def test_crash_before_current_swap_leaves_old_pointer(tmp_path: Path) -> None:
    _parent, parent_manifest, overlay = setup_root(tmp_path)
    before = _current_bytes(tmp_path)

    def inject(point: str) -> None:
        if point == "before_current_swap":
            raise RuntimeError("simulated crash")

    with pytest.raises(RuntimeError, match="simulated crash"):
        SleepTimePublisher(tmp_path).publish(
            new_version="v2",
            overlay=overlay,
            pipeline=passing_pipeline("v2"),
            expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
            fault_injector=inject,
        )
    assert _current_bytes(tmp_path) == before
    # The fully validated orphan is immutable but cannot be loaded through
    # CURRENT until an explicit recovery publication adopts it.
    assert (tmp_path / "bundles" / "v2" / "manifest.json").is_file()


def test_candidate_build_crash_quarantines_partial_staging_without_state_change(
    tmp_path: Path,
) -> None:
    from authority.memory_snapshot import ImmutableBaseBundle

    parent, parent_manifest, overlay = setup_root(tmp_path)
    publisher = SleepTimePublisher(tmp_path)
    current_before = _current_bytes(tmp_path)
    parent_before = _tree_hash(parent)
    ledger_existed_before = publisher.ledger_path.exists()

    def crash_during_build(_parent, _overlay_snapshot, candidate):
        (candidate / "partial").mkdir(parents=True)
        (candidate / "partial" / "written-before-crash.txt").write_text(
            "partial candidate output\n",
            encoding="utf-8",
        )
        raise RuntimeError("simulated candidate build crash")

    with pytest.raises(RuntimeError, match="candidate build crash"):
        publisher.publish(
            new_version="v2",
            overlay=overlay,
            pipeline=crash_during_build,
            expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
        )

    assert _current_bytes(tmp_path) == current_before
    assert _tree_hash(parent) == parent_before
    assert not (tmp_path / "bundles" / "v2").exists()
    assert publisher._ledger_events() == []
    assert publisher.ledger_path.exists() is ledger_existed_before
    assert not list(tmp_path.glob(".staging-v2-*"))
    assert not list(tmp_path.glob(".inputs-v2-*"))
    failed = list(tmp_path.glob(".failed-v2-*"))
    assert len(failed) == 1
    assert (failed[0] / "partial" / "written-before-crash.txt").read_text(
        encoding="utf-8"
    ) == "partial candidate output\n"
    assert (failed[0] / "overlay_snapshot" / "overlay_manifest.json").is_file()
    with pytest.raises(ValueError, match="not loadable"):
        ImmutableBaseBundle.load(failed[0])


def test_concurrent_publishers_use_compare_and_swap_parent(tmp_path: Path) -> None:
    parent, parent_manifest, overlay = setup_root(tmp_path)
    parent_manifest_hash = parent_manifest["manifest_sha256"]
    parent_file_hash = sha256_file(parent / "manifest.json")

    def publish(version: str):
        try:
            report = SleepTimePublisher(tmp_path).publish(
                new_version=version,
                overlay=overlay,
                pipeline=passing_pipeline(version),
                expected_parent_manifest_sha256=parent_manifest_hash,
            )
            return ("published", report.bundle_version)
        except PublicationConflictError:
            return ("conflict", version)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, ("v2", "v3")))
    assert [status for status, _version in results].count("published") == 1
    assert [status for status, _version in results].count("conflict") == 1
    current = json.loads((tmp_path / "CURRENT.json").read_text(encoding="utf-8"))
    assert current["bundle_version"] in {"v2", "v3"}
    assert current["parent_bundle"] == "tiny-v1"
    assert sha256_file(parent / "manifest.json") == parent_file_hash


def test_missing_pipeline_report_fails_closed(tmp_path: Path) -> None:
    _parent, parent_manifest, overlay = setup_root(tmp_path)
    before = _current_bytes(tmp_path)

    def incomplete_pipeline(parent, overlay_snapshot, candidate):
        passing_pipeline("v2")(parent, overlay_snapshot, candidate)
        return {"audit": {"status": "passed"}}

    with pytest.raises(PublicationValidationError, match="Incomplete"):
        SleepTimePublisher(tmp_path).publish(
            new_version="v2",
            overlay=overlay,
            pipeline=incomplete_pipeline,
            expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
        )
    assert _current_bytes(tmp_path) == before


def test_publication_ledger_is_hash_chained(tmp_path: Path) -> None:
    _parent, parent_manifest, overlay = setup_root(tmp_path)
    publisher = SleepTimePublisher(tmp_path)
    report = publisher.publish(
        new_version="v2",
        overlay=overlay,
        pipeline=passing_pipeline("v2"),
        expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
    )

    events = publisher._ledger_events()
    assert [event["sequence"] for event in events] == [1, 2]
    assert [event["event_type"] for event in events] == [
        "publication_prepared",
        "publication_committed",
    ]
    assert events[0]["payload"]["publication_id"] == report.publication_id
    assert events[1]["payload"]["publication_id"] == report.publication_id
    assert events[1]["parent_event_hash"] == events[0]["event_hash"]


def test_corrupt_publication_ledger_fails_before_build(tmp_path: Path) -> None:
    _parent, parent_manifest, overlay = setup_root(tmp_path)
    publisher = SleepTimePublisher(tmp_path)
    report = publisher.publish(
        new_version="v2",
        overlay=overlay,
        pipeline=passing_pipeline("v2"),
        expected_parent_manifest_sha256=parent_manifest["manifest_sha256"],
    )
    before = _current_bytes(tmp_path)
    raw = publisher.ledger_path.read_text(encoding="utf-8")
    publisher.ledger_path.write_text(
        raw.replace(report.publication_id, "publication::tampered", 1),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="event hash mismatch"):
        publisher.publish(
            new_version="v3",
            overlay=overlay,
            pipeline=passing_pipeline("v3"),
            expected_parent_manifest_sha256=report.bundle_manifest_sha256,
        )

    assert _current_bytes(tmp_path) == before
    assert not (tmp_path / "bundles" / "v3").exists()
    assert not list(tmp_path.glob(".inputs-v3-*"))
    assert not list(tmp_path.glob(".staging-v3-*"))
