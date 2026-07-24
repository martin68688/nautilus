import hashlib
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from authority.memory_snapshot import sha256_file, sha256_json, write_json_atomic
from config import _populate_run_identity, save_run_identity
from tests.test_memory_snapshot_overlay import build_tiny_bundle, write_current


REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finalize(payload: dict, hash_field: str) -> dict:
    payload[hash_field] = sha256_json(
        {key: value for key, value in payload.items() if key != hash_field}
    )
    return payload


def _build_modern_audited_bundle(
    tmp_path: Path,
    *,
    sidecar_artifact_id: str | None = None,
) -> tuple[Path, dict]:
    bundle, manifest = build_tiny_bundle(tmp_path)
    source_run = "tiny-run"
    source_task = "tiny-task"
    node_id = f"run::{source_run}::node::node-a"
    code_sha256 = "1" * 64
    actual_snapshot = {
        "run_directory_count": 1,
        "complete_run_count": 1,
        "code_node_count": 1,
    }
    corpus = _finalize(
        {
            "schema": "corpus_manifest_v1",
            "corpus_id": "tiny-corpus",
            "created_at": "2026-07-19T00:00:00Z",
            "source_repo": "test",
            "source_root": "/test",
            "source_commit": "test-commit",
            "expected_snapshot": actual_snapshot,
            "actual_snapshot": actual_snapshot,
            "exclusion_rules": [],
            "runs": [],
            "split_manifests": [],
            "manifest_sha256": "",
        },
        "manifest_sha256",
    )
    split = _finalize(
        {
            "schema": "memory_split_manifest_v1",
            "split_id": "tiny-task-heldout-v1",
            "split_kind": "task-heldout",
            "split_version": "v1",
            "corpus_manifest_hash": corpus["manifest_sha256"],
            "created_at": "2026-07-19T00:00:00Z",
            "source_run_ids": [source_run],
            "heldout_run_ids": [],
            "source_task_ids": [source_task],
            "heldout_task_ids": [],
            "source_seed_groups": [],
            "heldout_seed_groups": [],
            "excluded_run_ids": [],
            "allocation": {},
            "validation": {
                "run_overlap": [],
                "run_overlap_count": 0,
                "task_overlap": [],
                "task_overlap_count": 0,
                "seed_group_overlap": [],
                "seed_group_overlap_count": 0,
            },
            "manifest_sha256": "",
        },
        "manifest_sha256",
    )
    graph = {
        "meta": {
            "schema": "hyperbolic_run_forest_memory_v2",
            "bundle_id": manifest["bundle_id"],
            "corpus_id": corpus["corpus_id"],
            "corpus_manifest_hash": corpus["manifest_sha256"],
            "split_id": split["split_id"],
            "split_manifest_hash": split["manifest_sha256"],
            "source_run_count": 1,
            "heldout_run_count": 0,
            "certification_level": "raw_audited",
            "legacy_artifact_overwritten": False,
        },
        "nodes": [
            {
                "id": node_id,
                "type": "RunNode",
                "run_id": source_run,
                "task": source_task,
                "code_sha256": code_sha256,
            }
        ],
        "edges": [],
    }
    sidecar = _finalize(
        {
            "schema": "audit_sidecar_v1",
            "artifact_id": sidecar_artifact_id or node_id,
            "run_id": source_run,
            "code_sha256": code_sha256,
            "status": "clean",
            "issues": [],
            "source_journal_sha256": "2" * 64,
            "sidecar_sha256": "",
        },
        "sidecar_sha256",
    )
    audit_index = {
        "schema": "audit_sidecar_index_v1",
        "corpus_manifest_hash": corpus["manifest_sha256"],
        "detector_version": "deterministic-test-v1",
        "entries": {node_id: "node-a.json"},
    }
    drift_review = {
        "schema": "corpus_drift_review_v1",
        "reviewed": True,
        "excluded_runs_reviewed": True,
        "corpus_manifest_hash": corpus["manifest_sha256"],
        "actual_snapshot_hash": sha256_json(actual_snapshot),
    }
    runforest_report = {
        "schema": "run_forest_builder_report_v2",
        "bundle_id": manifest["bundle_id"],
        "corpus_manifest_hash": corpus["manifest_sha256"],
        "split_manifest_hash": split["manifest_sha256"],
        "source_run_count": 1,
        "heldout_run_count": 0,
        "spooky_source_run_count": 0,
        "expected_audited_code_node_count": 1,
        "audited_code_node_count": 1,
        "all_code_nodes_have_sidecars": True,
    }
    build_report = {
        "schema": "memory_bundle_build_report_v1",
        "bundle_id": manifest["bundle_id"],
        "bundle_version": manifest["bundle_version"],
        "split_id": split["split_id"],
        "source_run_count": 1,
        "heldout_run_count": 0,
        "raw_journal_run_count": 1,
        "sidecar_count": 1,
        "all_code_nodes_have_sidecars": True,
        "all_clause_sources_resolve": True,
        "secret_scan_passed": True,
        "corpus_drift_reviewed": True,
        "published_atomically": True,
        "legacy_artifact_overwritten": False,
        "spooky_source_run_count": 0,
        "heldout_run_refs_in_graph": [],
    }
    json_artifacts = {
        "runforest/graph.json": graph,
        "runforest/build_report.json": runforest_report,
        "corpus/manifest.json": corpus,
        "corpus/drift_review.json": drift_review,
        "splits/active.json": split,
        "audit_sidecars/index.json": audit_index,
        "audit_sidecars/node-a.json": sidecar,
        "reports/build_report.json": build_report,
    }
    for relative, payload in json_artifacts.items():
        write_json_atomic(bundle / relative, payload)

    artifact_hashes = {
        relative: sha256_file(bundle / relative)
        for relative in json_artifacts
    }
    for relative in (
        "runforest/index.npz",
        "sop/clauses.jsonl",
        "visibility/precompiled_masks/declared_scope_masks.json",
    ):
        artifact_hashes[relative] = sha256_file(bundle / relative)
    manifest.update(
        {
            "corpus_manifest_hash": corpus["manifest_sha256"],
            "detector_version": audit_index["detector_version"],
            "graph_hashes": {
                "runforest": artifact_hashes["runforest/graph.json"]
            },
            "index_hashes": {
                "runforest": artifact_hashes["runforest/index.npz"]
            },
            "split_id": split["split_id"],
            "certification_level": "raw_audited",
            "build_report": "reports/build_report.json",
            "artifact_hashes": artifact_hashes,
            "manifest_sha256": "",
        }
    )
    _finalize(manifest, "manifest_sha256")
    write_json_atomic(bundle / "manifest.json", manifest)
    write_current(tmp_path, bundle, manifest)
    return bundle, manifest


def test_memory_run_identity_binds_exact_clean_snapshot(tmp_path, monkeypatch):
    graph = tmp_path / "run_forest_graph.json"
    graph.write_text(
        json.dumps(
            {
                "meta": {
                    "source_membership_verified": True,
                    "leak_verified": True,
                    "source_runs": ["run-a", "run-b"],
                }
            }
        ),
        encoding="utf-8",
    )
    index = tmp_path / "run_forest_index.npz"
    index.write_bytes(b"exact-index-snapshot")
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {
                "enable": True,
                "graph_path": str(graph),
                "index_path": str(index),
            },
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stage_hybrid_v2",
                "memory_snapshot_sha256": "",
                "memory_index_sha256": "",
                "memory_source_count": 0,
                "memory_source_runs": [],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )
    monkeypatch.setenv("MLEVOLVE_CODE_REVISION", "0cb6cdd2")
    monkeypatch.setenv("MLEVOLVE_CODE_WORKTREE_SHA256", "worktree-digest")

    _populate_run_identity(cfg)

    assert cfg.run_identity.memory_snapshot_sha256 == _sha256(graph)
    assert cfg.run_identity.memory_index_sha256 == _sha256(index)
    assert cfg.run_identity.memory_source_runs == ["run-a", "run-b"]
    assert cfg.run_identity.memory_source_count == 2
    assert cfg.run_identity.code_revision == "0cb6cdd2"
    assert cfg.run_identity.code_worktree_sha256 == "worktree-digest"


def test_memory_run_identity_binds_current_base_manifest(tmp_path):
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {
                "enable": True,
                "bundle_root": str(tmp_path),
                "current_pointer_path": "CURRENT.json",
                "graph_path": str(tmp_path / "must-not-be-used.json"),
                "index_path": "",
            },
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stale-version",
                "memory_snapshot_sha256": "",
                "memory_index_sha256": "",
                "memory_source_count": 0,
                "memory_source_runs": [],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )

    _populate_run_identity(cfg)

    assert cfg.run_identity.memory_snapshot_sha256 == manifest["manifest_sha256"]
    assert cfg.run_identity.memory_index_sha256 == _sha256(
        bundle / "runforest" / "index.npz"
    )
    assert cfg.run_identity.memory_version == "v1"
    assert cfg.run_identity.memory_source_runs == ["tiny-run"]


def test_memory_run_identity_accepts_hash_bound_modern_audit_provenance(tmp_path):
    _bundle, manifest = _build_modern_audited_bundle(tmp_path)
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {
                "enable": True,
                "bundle_root": str(tmp_path),
                "current_pointer_path": "CURRENT.json",
                "graph_path": "",
                "index_path": "",
            },
            "evaluation_authority": {
                "require_bound_bundle": True,
                "expected_bundle_id": manifest["bundle_id"],
                "expected_bundle_manifest_sha256": manifest["manifest_sha256"],
                "policy_version": "authority_v1",
            },
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stale-version",
                "memory_snapshot_sha256": "",
                "memory_index_sha256": "",
                "memory_source_count": 0,
                "memory_source_runs": [],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )

    _populate_run_identity(cfg)

    assert cfg.run_identity.memory_snapshot_sha256 == manifest["manifest_sha256"]
    assert cfg.run_identity.memory_source_runs == ["tiny-run"]
    assert cfg.run_identity.memory_source_count == 1


def test_memory_run_identity_rejects_semantically_misbound_modern_sidecar(tmp_path):
    _bundle, manifest = _build_modern_audited_bundle(
        tmp_path,
        sidecar_artifact_id="wrong-artifact",
    )
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {
                "enable": True,
                "bundle_root": str(tmp_path),
                "current_pointer_path": "CURRENT.json",
                "graph_path": "",
                "index_path": "",
            },
            "evaluation_authority": {
                "require_bound_bundle": True,
                "expected_bundle_id": manifest["bundle_id"],
                "expected_bundle_manifest_sha256": manifest["manifest_sha256"],
                "policy_version": "authority_v1",
            },
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stale-version",
                "memory_snapshot_sha256": "",
                "memory_index_sha256": "",
                "memory_source_count": 0,
                "memory_source_runs": [],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )

    with pytest.raises(ValueError, match="provenance verification failed"):
        _populate_run_identity(cfg)


def test_memory_run_identity_requires_explicit_pins_for_bound_bundle(tmp_path):
    _bundle, _manifest = _build_modern_audited_bundle(tmp_path)
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {
                "enable": True,
                "bundle_root": str(tmp_path),
                "current_pointer_path": "CURRENT.json",
                "graph_path": "",
                "index_path": "",
            },
            "evaluation_authority": {
                "require_bound_bundle": True,
                "expected_bundle_id": "",
                "expected_bundle_manifest_sha256": "",
                "policy_version": "authority_v1",
            },
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stale-version",
                "memory_snapshot_sha256": "",
                "memory_index_sha256": "",
                "memory_source_count": 0,
                "memory_source_runs": [],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )

    with pytest.raises(ValueError, match="explicit identity pins"):
        _populate_run_identity(cfg)


def test_run_identity_is_persisted_without_waiting_for_a_journal(tmp_path):
    cfg = OmegaConf.create(
        {
            "log_dir": str(tmp_path / "early-failure" / "logs"),
            "run_identity": {
                "schema": "mlevolve_run_identity_v1",
                "experiment_group": "stage_hybrid_v2_all_clean_history",
                "baseline_reference_group": "baseline_no_external_memory",
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stage_hybrid_v2",
                "memory_snapshot_sha256": "graph-hash",
                "memory_index_sha256": "index-hash",
                "memory_source_count": 29,
                "memory_source_runs": ["run-a"],
                "code_revision": "3ac19fd3",
                "code_worktree_sha256": "source-hash",
                "identity_source": "declared_at_runtime",
            },
        }
    )

    identity_path = save_run_identity(cfg)

    assert identity_path == tmp_path / "early-failure" / "logs" / "run_identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert identity["memory_enabled"] is True
    assert identity["memory_version"] == "stage_hybrid_v2"
    assert identity["experiment_group"] == "stage_hybrid_v2_all_clean_history"


def test_run_persists_identity_before_loading_task_or_generating_drafts():
    source = (REPO / "mlevolve" / "run.py").read_text(encoding="utf-8")
    identity_write = source.index("identity_path = save_run_identity(cfg)")
    task_load = source.index("task_desc = load_task_desc(cfg)")
    journal_create = source.index("journal = Journal()")
    assert identity_write < task_load < journal_create


def test_memory_run_identity_fails_closed_on_unclean_graph(tmp_path):
    graph = tmp_path / "unclean.json"
    graph.write_text(
        json.dumps(
            {
                "meta": {
                    "source_membership_verified": True,
                    "leak_verified": False,
                    "source_runs": ["run-a"],
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {
                "enable": True,
                "graph_path": str(graph),
                "index_path": "",
            },
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "run_forest_stage_hybrid",
                "memory_version": "stage_hybrid_v2",
                "memory_snapshot_sha256": "",
                "memory_index_sha256": "",
                "memory_source_count": 0,
                "memory_source_runs": [],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )

    try:
        _populate_run_identity(cfg)
    except ValueError as exc:
        assert "source-verified and leak-verified" in str(exc)
    else:
        raise AssertionError("unclean memory graph must fail closed")


def test_baseline_identity_cannot_carry_memory_snapshot_fields():
    cfg = OmegaConf.create(
        {
            "external_skill_memory": {"enable": False, "graph_path": "", "index_path": ""},
            "run_identity": {
                "memory_enabled": True,
                "memory_system": "legacy",
                "memory_version": "legacy",
                "memory_snapshot_sha256": "stale",
                "memory_index_sha256": "stale",
                "memory_source_count": 9,
                "memory_source_runs": ["stale"],
                "code_revision": "",
                "code_worktree_sha256": "",
            },
        }
    )

    _populate_run_identity(cfg)

    assert cfg.run_identity.memory_enabled is False
    assert cfg.run_identity.memory_system == "none"
    assert cfg.run_identity.memory_version == "none"
    assert cfg.run_identity.memory_snapshot_sha256 == ""
    assert cfg.run_identity.memory_index_sha256 == ""
    assert cfg.run_identity.memory_source_count == 0
    assert cfg.run_identity.memory_source_runs == []


def test_historical_source_runs_are_explicitly_labeled_as_no_memory_baselines():
    allowlist = json.loads(
        (REPO / "paper-skills/eval_skill_memory/clean_run_allowlist.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (REPO / "paper-skills/eval_skill_memory/run_identity_registry_v1.json").read_text(encoding="utf-8")
    )
    allowed = {row["run_id"] for row in allowlist["entries"] if row.get("allowed")}
    baseline = registry["groups"]["baseline_no_external_memory"]
    assert baseline["memory_enabled"] is False
    assert baseline["memory_system"] == "none"
    assert baseline["memory_version"] == "none"
    assert set(baseline["run_ids"]) == allowed
    memory_group = registry["groups"]["stage_hybrid_v2_all_clean_history"]
    assert memory_group["memory_enabled"] is True
    assert memory_group["memory_version"] == "stage_hybrid_v2"
