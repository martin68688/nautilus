from __future__ import annotations

import json
import tarfile

import pytest

from tests.memory_bundle_helpers import (
    prepare_audit_and_splits,
    prepare_corpus,
    prepare_pipeline,
    prepare_runforest_and_bundle,
    prepare_sops,
)


def test_bundle_is_immutable_hash_complete_and_split_isolated(tmp_path) -> None:
    from schema import MemoryBundleManifestV1, read_json, sha256_file
    from validate_memory_bundle import validate_bundle

    result = prepare_pipeline(tmp_path, split_name="task-heldout")
    bundle = result["bundle_dir"]
    validation = validate_bundle(bundle)
    assert validation["valid"] is True
    assert validation["source_run_count"] == 9
    assert validation["heldout_run_count"] == 3
    assert validation["heldout_reference_count"] == 0
    assert validation["spooky_node_count"] == 0
    assert validation["code_node_count"] == validation["sidecar_count"] == 18
    assert validation["all_clause_sources_resolve"] is True

    manifest = MemoryBundleManifestV1.from_dict(read_json(bundle / "manifest.json"))
    assert manifest.certification_level == "raw_audited"
    assert manifest.split_id == "task-heldout-v1"
    assert all(
        sha256_file(bundle / relative) == digest
        for relative, digest in manifest.artifact_hashes.items()
    )
    raw_runs = {path.name for path in (bundle / "raw_journals").iterdir()}
    assert raw_runs == set(result["splits"]["task-heldout"].source_run_ids)
    assert raw_runs.isdisjoint(result["splits"]["task-heldout"].heldout_run_ids)


def test_bundle_validator_detects_tampering_and_publisher_refuses_overwrite(
    tmp_path,
) -> None:
    from build_memory_bundle import build_bundle
    from validate_memory_bundle import validate_bundle

    result = prepare_pipeline(tmp_path, split_name="task-heldout")
    bundle = result["bundle_dir"]
    graph_path = bundle / "runforest" / "graph.json"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["meta"]["bundle_id"] = "tampered"
    graph_path.write_text(json.dumps(payload), encoding="utf-8")
    validation = validate_bundle(bundle)
    assert validation["valid"] is False
    assert any("artifact_hash_mismatch:runforest/graph.json" in error for error in validation["errors"])

    with pytest.raises(FileExistsError, match="already exists"):
        build_bundle(
            result["manifest_path"],
            result["drift_review_path"],
            result["split_paths"]["task-heldout"],
            result["audit_dir"],
            result["runforest_dir"],
            result["protocol_dir"],
            bundle,
            bundle_id="bundle-task-heldout",
            bundle_version="v1",
            authority_policy_version="authority_v1",
            detector_version=result["audit_report"]["detector_version"],
            deepseek_model="deepseek-test",
            deepseek_prompt_hash=result["distillation_report"]["system_prompt_hash"],
        )


def test_bundle_validator_rejects_existing_but_undeclared_core_artifact(
    tmp_path,
) -> None:
    from schema import sha256_file, sha256_json, write_json_atomic
    from validate_memory_bundle import validate_bundle

    result = prepare_pipeline(tmp_path, split_name="task-heldout")
    bundle = result["bundle_dir"]
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (bundle / "corpus" / "manifest.json").is_file()
    manifest["artifact_hashes"].pop("corpus/manifest.json")
    manifest["manifest_sha256"] = sha256_json(
        {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
    )
    write_json_atomic(manifest_path, manifest)
    sums_path = bundle / "SHA256SUMS"
    sums = []
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        _digest, relative = line.split("  ", 1)
        digest = (
            sha256_file(manifest_path)
            if relative == "manifest.json"
            else _digest
        )
        sums.append(f"{digest}  {relative}")
    sums_path.write_text("\n".join(sums) + "\n", encoding="utf-8")

    validation = validate_bundle(bundle)
    assert validation["valid"] is False
    assert "artifact_not_declared:corpus/manifest.json" in validation["errors"]


def test_full_seed_and_task_bundles_all_validate_with_zero_heldout_refs(
    tmp_path,
) -> None:
    from validate_memory_bundle import validate_bundle

    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    sops = prepare_sops(tmp_path, corpus, audit_splits)
    expected_source_counts = {"full": 12, "seed-heldout": 8, "task-heldout": 9}
    for split_name, expected_count in expected_source_counts.items():
        result = prepare_runforest_and_bundle(
            tmp_path,
            corpus,
            audit_splits,
            sops,
            split_name=split_name,
        )
        report = validate_bundle(result["bundle_dir"])
        assert report["valid"] is True
        assert report["source_run_count"] == expected_count
        assert report["heldout_reference_count"] == 0
        assert report["spooky_node_count"] == 0


def test_tar_zst_export_is_deterministic_and_contains_manifest(tmp_path) -> None:
    import zstandard

    from build_memory_bundle import create_tar_zst

    result = prepare_pipeline(tmp_path, split_name="task-heldout")
    first = create_tar_zst(result["bundle_dir"], tmp_path / "first.tar.zst")
    second = create_tar_zst(result["bundle_dir"], tmp_path / "second.tar.zst")
    assert first["sha256"] == second["sha256"]
    extracted_tar = tmp_path / "bundle.tar"
    decompressor = zstandard.ZstdDecompressor()
    with (tmp_path / "first.tar.zst").open("rb") as source, extracted_tar.open(
        "wb"
    ) as destination:
        decompressor.copy_stream(source, destination)
    with tarfile.open(extracted_tar) as archive:
        names = set(archive.getnames())
    assert f"{result['bundle_dir'].name}/manifest.json" in names


def test_secret_scan_rejects_literal_credentials_without_echoing_them(tmp_path) -> None:
    from build_memory_bundle import _secret_scan

    path = tmp_path / "config.yaml"
    path.write_text("api_key: sk-abcdefghijklmnopqrstuvwxyz123456\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Secret-like material detected") as error:
        _secret_scan(path)
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in str(error.value)


def test_bundle_publication_requires_bound_manual_corpus_review(tmp_path) -> None:
    from build_memory_bundle import build_bundle

    result = prepare_pipeline(tmp_path, split_name="task-heldout")
    review = json.loads(
        result["drift_review_path"].read_text(encoding="utf-8")
    )
    review["reviewed"] = False
    bad_review = tmp_path / "bad_review.json"
    bad_review.write_text(json.dumps(review), encoding="utf-8")
    output = tmp_path / "bundle-without-review"
    with pytest.raises(ValueError, match="not approved"):
        build_bundle(
            result["manifest_path"],
            bad_review,
            result["split_paths"]["task-heldout"],
            result["audit_dir"],
            result["runforest_dir"],
            result["protocol_dir"],
            output,
            bundle_id="bundle-without-review",
            bundle_version="v1",
            authority_policy_version="authority_v1",
            detector_version=result["audit_report"]["detector_version"],
            deepseek_model="deepseek-test",
            deepseek_prompt_hash=result["distillation_report"]["system_prompt_hash"],
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".bundle-without-review.staging-*"))


def test_resigned_split_overlap_is_rejected_by_validator_and_snapshot_loader(
    tmp_path,
) -> None:
    from authority.memory_snapshot import (
        MemorySnapshotLoader,
        make_current_pointer,
        sha256_file,
        sha256_json,
        write_json_atomic,
    )
    from validate_memory_bundle import validate_bundle

    result = prepare_pipeline(tmp_path, split_name="task-heldout")
    bundle = result["bundle_dir"]

    # Model an attacker who changes the split and then recomputes every
    # directly affected self/file hash.  A plain hash-mismatch test would not
    # exercise the semantic source/heldout isolation boundary.
    split_path = bundle / "splits" / "active.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    overlapping_run = split["source_run_ids"][0]
    split["heldout_run_ids"][0] = overlapping_run
    assert split["validation"]["run_overlap_count"] == 0
    split["manifest_sha256"] = sha256_json(
        {key: value for key, value in split.items() if key != "manifest_sha256"}
    )
    write_json_atomic(split_path, split)

    graph_path = bundle / "runforest" / "graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["meta"]["split_manifest_hash"] = split["manifest_sha256"]
    write_json_atomic(graph_path, graph)

    runforest_report_path = bundle / "runforest" / "build_report.json"
    runforest_report = json.loads(runforest_report_path.read_text(encoding="utf-8"))
    runforest_report["split_manifest_hash"] = split["manifest_sha256"]
    write_json_atomic(runforest_report_path, runforest_report)

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes"]["splits/active.json"] = sha256_file(split_path)
    manifest["artifact_hashes"]["runforest/graph.json"] = sha256_file(graph_path)
    manifest["artifact_hashes"]["runforest/build_report.json"] = sha256_file(
        runforest_report_path
    )
    manifest["graph_hashes"]["runforest"] = sha256_file(graph_path)
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    write_json_atomic(manifest_path, manifest)

    validation = validate_bundle(bundle)
    assert validation["valid"] is False
    assert "split_run_overlap" in validation["errors"]

    write_json_atomic(
        tmp_path / "CURRENT.json",
        make_current_pointer(
            bundle_path=str(bundle.relative_to(tmp_path)),
            manifest=manifest,
            parent_bundle=None,
        ),
    )
    with pytest.raises(ValueError, match="split_run_overlap"):
        MemorySnapshotLoader(tmp_path).load(
            session_overlay_path=tmp_path / "malicious-overlay",
            active_protocol_ref="test-protocol@1#test-hash",
            authority_policy_version="authority_v1",
        )
