from __future__ import annotations

import json

import pytest

from tests.memory_bundle_helpers import (
    prepare_audit_and_splits,
    prepare_corpus,
    write_run,
)


def _stats(root):
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_manifest_classifies_complete_partial_invalid_and_excluded_without_writes(
    tmp_path,
) -> None:
    corpus = prepare_corpus(tmp_path)
    manifest = corpus["manifest"]
    snapshot = manifest.actual_snapshot

    assert snapshot["run_directory_count"] == 15
    assert snapshot["status_counts"] == {
        "complete": 12,
        "excluded": 1,
        "invalid_json": 1,
        "partial": 1,
    }
    assert snapshot["complete_non_spooky_task_count"] == 4
    assert snapshot["node_count"] == 24
    assert snapshot["code_node_count"] == 24
    assert snapshot["metric_node_count"] == 24
    assert corpus["inventory"]["drift_detected"] is False
    assert all(
        "spooky" not in run.canonical_task_id
        for run in manifest.runs
        if run.status == "complete"
    )
    spooky = next(
        run for run in manifest.runs if "spooky" in run.canonical_task_id
    )
    assert spooky.status == "excluded"
    assert spooky.exclusion_reason == "excluded_task"


def test_manifest_and_sidecar_audit_preserve_source_journal_stats(tmp_path) -> None:
    corpus = prepare_corpus(tmp_path)
    before = _stats(corpus["runs_root"])
    outputs = prepare_audit_and_splits(tmp_path, corpus)
    after = _stats(corpus["runs_root"])

    assert before == after
    report = outputs["audit_report"]
    assert report["expected_code_node_count"] == 24
    assert report["sidecar_count"] == 24
    assert report["all_code_nodes_have_sidecars"] is True
    assert report["source_journals_modified"] is False
    index = json.loads(
        (outputs["audit_dir"] / "index.json").read_text(encoding="utf-8")
    )
    assert len(index["entries"]) == 24
    assert all(
        json.loads((outputs["audit_dir"] / filename).read_text(encoding="utf-8"))[
            "legacy_receipt_level"
        ]
        == "legacy_static_only"
        for filename in index["entries"].values()
    )


def test_manifest_hash_detects_tampering(tmp_path) -> None:
    from schema import CorpusManifestV1

    corpus = prepare_corpus(tmp_path)
    payload = corpus["manifest"].as_dict()
    payload["actual_snapshot"]["complete_run_count"] = 999
    with pytest.raises(ValueError, match="hash mismatch"):
        CorpusManifestV1.from_dict(payload)


def test_literal_secret_config_is_excluded_without_serializing_the_value(tmp_path) -> None:
    from build_corpus_manifest import build_manifest

    runs = tmp_path / "runs"
    runs.mkdir()
    run_dir = write_run(runs, task_id="task-a", seed=1)
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    with (run_dir / "logs" / "config.yaml").open("a", encoding="utf-8") as handle:
        handle.write(f"api_key: {secret}\n")
    tags = tmp_path / "tags.json"
    tags.write_text('{"task-a":"family-a"}', encoding="utf-8")
    manifest, _report = build_manifest(
        runs,
        source_repo="third_party/MLEvolve",
        source_commit="be034ec",
        task_tags_path=tags,
        created_at="2026-07-19T00:00:00Z",
    )
    assert manifest.runs[0].status == "excluded"
    assert manifest.runs[0].exclusion_reason == "secret_material_detected"
    assert secret not in json.dumps(manifest.as_dict())


def test_omegaconf_python_tags_are_inert_and_metadata_remains_readable(
    tmp_path,
) -> None:
    from build_corpus_manifest import load_config_metadata

    config = tmp_path / "config.yaml"
    config.write_text(
        """data_dir: !!python/object/apply:pathlib.PosixPath
- /
- corpus
exp_id: aerial-cactus-identification
agent:
  seed: 42
""",
        encoding="utf-8",
    )

    task_id, seed, warnings = load_config_metadata(config)

    assert task_id == "aerial-cactus-identification"
    assert seed == "42"
    assert warnings == []


def test_metric_count_ignores_unscored_metric_placeholders(tmp_path) -> None:
    from build_corpus_manifest import build_manifest

    runs = tmp_path / "runs"
    runs.mkdir()
    run_dir = write_run(runs, task_id="task-a", seed=1)
    journal_path = run_dir / "logs" / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["nodes"][0]["metric"] = {"value": None, "maximize": None}
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    tags = tmp_path / "tags.json"
    tags.write_text('{"task-a":"family-a"}', encoding="utf-8")

    manifest, _report = build_manifest(
        runs,
        source_repo="third_party/MLEvolve",
        source_commit="be034ec",
        task_tags_path=tags,
        created_at="2026-07-19T00:00:00Z",
    )

    assert manifest.actual_snapshot["node_count"] == 2
    assert manifest.actual_snapshot["metric_node_count"] == 1
