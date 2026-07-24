from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

from authority.clean_replay import build_replay_queue, load_replay_queue
from authority.replay_certifier import (
    ProtocolRepairSurface,
    ReplayIdentity,
    verify_protocol_only_patch,
)
from authority.protocol_registry import ProtocolRegistry
from authority.memory_snapshot import sha256_file, sha256_json, write_json_atomic
from tests.authority.clean_replay_helpers import (
    PROTOCOL_REPAIR_CODE,
    SOURCE_CODE,
    build_registry,
)
from tests.test_memory_snapshot_overlay import build_tiny_bundle


REPO = Path(__file__).resolve().parents[2]


def _candidate(index: int, family: str, delta: float, **updates):
    value = {
        "candidate_id": f"candidate-{index}",
        "task_id": "task-a",
        "source_artifact_id": f"source-{index}",
        "parent_artifact_id": f"parent-{index}",
        "child_artifact_id": f"child-{index}",
        "original_claim_id": f"claim-{index}",
        "source_clause_id": f"clause-{index}",
        "code_sha256": f"{index:x}" * 64,
        "method_hypothesis": f"method hypothesis {index}",
        "method_family": family,
        "audit_status": "candidate_replay",
        "source_refs": [
            f"source-{index}",
            f"parent-{index}",
            f"child-{index}",
            f"claim-{index}",
            f"clause-{index}",
        ],
        "historical_metric_delta": delta,
        "method_fatal_issues": [],
        "protocol_issue_codes": ["FIT_SCOPE"],
    }
    value.update(updates)
    return value


def test_protocol_only_patch_preserves_full_method_surface() -> None:
    _registry, ref = build_registry()
    surface = ProtocolRepairSurface.from_protocol_spec(_registry.resolve(ref))
    report = verify_protocol_only_patch(
        SOURCE_CODE,
        PROTOCOL_REPAIR_CODE,
        surface,
        source_artifact_id="historical-artifact",
        replay_artifact_id="clean-replay-artifact",
    )

    report.verify()
    assert report.identity == ReplayIdentity.METHOD_PRESERVED
    assert report.protected_changes == {}
    assert report.source_protected_surface_hash == report.replay_protected_surface_hash
    assert report.source_code_sha256 != report.replay_code_sha256
    assert report.reason == "protected_surface_equal_and_protocol_delta_declared"


def test_repair_surface_is_introduced_by_new_protocol_version() -> None:
    registry = ProtocolRegistry("mlevolve/config/protocols")
    v1 = registry.get("mlevolve-default", "1")
    v2 = registry.get("mlevolve-default", "2")
    assert ProtocolRepairSurface.from_protocol_spec(v1).allowed_change_kinds == ()
    assert set(ProtocolRepairSurface.from_protocol_spec(v2).allowed_change_kinds) == {
        "split_api",
        "preprocessing_scope",
        "evaluator",
        "selection_freeze",
        "seed_aggregation",
        "holdout_access",
        "instrumentation",
    }
    assert v2.parent_version == "1"
    assert v2.canonical_hash != v1.canonical_hash


def test_replay_queue_is_deterministic_diverse_and_capped(tmp_path) -> None:
    candidates = [
        _candidate(1, "linear", 0.9),
        _candidate(2, "linear", 0.8),
        _candidate(3, "tree", 0.7),
        _candidate(4, "neural", 0.6),
        _candidate(5, "kernel", 1.0, method_fatal_issues=["test_label_feature"]),
    ]
    first = build_replay_queue(
        candidates, max_per_task=3, created_at="2026-07-19T00:00:00Z"
    )
    second = build_replay_queue(
        reversed(candidates), max_per_task=3, created_at="2026-07-19T00:00:00Z"
    )

    assert [entry.candidate_id for entry in first.entries] == [
        "candidate-1",
        "candidate-3",
        "candidate-4",
    ]
    assert first.manifest_sha256 == second.manifest_sha256
    assert len(first.entries) == 3
    assert len({entry.method_family for entry in first.entries}) == 3
    assert all(entry.historical_metric_used_as_evidence is False for entry in first.entries)
    assert any(
        item["candidate_id"] == "candidate-5"
        and "method_fatal_static_audit" in item["reasons"]
        for item in first.rejected
    )

    queue_path = tmp_path / "replay_queue.jsonl"
    manifest_path = tmp_path / "replay_queue_manifest.json"
    first.write(queue_path, manifest_path)
    loaded = load_replay_queue(queue_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selected_count"] == 3
    assert manifest["max_per_task"] == 3
    assert len(manifest["queue_file_sha256"]) == 64
    assert all(
        json.loads(line)["historical_metric_used_as_evidence"] is False
        for line in queue_path.read_text(encoding="utf-8").splitlines()
    )
    assert loaded.manifest_sha256 == first.manifest_sha256
    assert [entry.entry_hash for entry in loaded.entries] == [
        entry.entry_hash for entry in first.entries
    ]


def test_replay_verifier_cli_uses_versioned_protocol_surface(tmp_path) -> None:
    source = tmp_path / "source.py"
    replay = tmp_path / "replay.py"
    report = tmp_path / "report.json"
    source.write_text(SOURCE_CODE, encoding="utf-8")
    replay.write_text(PROTOCOL_REPAIR_CODE, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "paper-skills" / "memory_bundle" / "verify_clean_replay.py"),
            "--source-code",
            str(source),
            "--replay-code",
            str(replay),
            "--protocol-registry",
            str(REPO / "mlevolve" / "config" / "protocols"),
            "--protocol-id",
            "mlevolve-default",
            "--protocol-version",
            "2",
            "--source-artifact-id",
            "source",
            "--replay-artifact-id",
            "replay",
            "--report",
            str(report),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["identity"] == ReplayIdentity.METHOD_PRESERVED.value
    assert payload["repair_protocol_ref"].startswith("mlevolve-default@2#")


def test_verified_bundle_candidate_extractor_binds_raw_code_and_lineage(tmp_path) -> None:
    bundle, manifest = build_tiny_bundle(tmp_path / "bundle-root")
    code_hash = hashlib.sha256(SOURCE_CODE.encode("utf-8")).hexdigest()
    write_json_atomic(
        bundle / "raw_journals" / "run-1" / "journal.json",
        {
            "nodes": [
                {"id": "parent", "code": "", "metric": {"value": 0.5, "maximize": True}},
                {
                    "id": "child",
                    "parent_id": "parent",
                    "code": SOURCE_CODE,
                    "metric": {"value": 0.7, "maximize": True},
                },
            ],
            # Production RunForest v2 keeps lineage in the immutable raw
            # journal rather than materializing synthetic Transition nodes.
            "node2parent": {"child": "parent"},
        },
    )
    graph = {
        "schema": "tiny_replay_graph_v1",
        "nodes": [
            {"id": "run::run-1", "type": "Run", "run_id": "run-1"},
            {
                "id": "run::run-1::node::parent",
                "type": "RunNode",
                "run_id": "run-1",
                "raw_node_id": "parent",
                "task": "task-a",
                "code_sha256": "",
            },
            {
                "id": "run::run-1::node::child",
                "type": "RunNode",
                "run_id": "run-1",
                "raw_node_id": "child",
                "task": "task-a",
                "code_sha256": code_hash,
                "leakage_audit": {"status": "clean", "issues": []},
            },
            {
                "id": "clause::method",
                "clause_id": "clause::method",
                "type": "SOPClause",
                "claim_types": ["method_hypothesis"],
                "claim_refs": ["claim::method"],
                "source_artifact_refs": ["run::run-1::node::child"],
                "text": "Use the frozen TF-IDF plus logistic-regression method.",
            },
        ],
        "edges": [],
    }
    write_json_atomic(bundle / "runforest" / "graph.json", graph)
    authority = bundle / "authority"
    authority.mkdir(exist_ok=True)
    (authority / "claims.jsonl").write_text(
        json.dumps(
            {
                "claim_id": "claim::method",
                "claim_type": "method_hypothesis",
                "subject_artifact_id": "run::run-1::node::child",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    for relative in (
        "runforest/graph.json",
        "raw_journals/run-1/journal.json",
        "authority/claims.jsonl",
    ):
        manifest["artifact_hashes"][relative] = sha256_file(bundle / relative)
    manifest["graph_hashes"]["runforest"] = manifest["artifact_hashes"][
        "runforest/graph.json"
    ]
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    write_json_atomic(bundle / "manifest.json", manifest)
    candidates = tmp_path / "candidates.jsonl"
    queue = tmp_path / "replay_queue.jsonl"
    queue_manifest = tmp_path / "replay_queue_manifest.json"
    report = tmp_path / "extraction_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "paper-skills" / "memory_bundle" / "extract_replay_candidates.py"),
            "--bundle",
            str(bundle),
            "--candidates",
            str(candidates),
            "--queue",
            str(queue),
            "--queue-manifest",
            str(queue_manifest),
            "--report",
            str(report),
            "--created-at",
            "2026-07-19T00:00:00Z",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    extraction = json.loads(report.read_text(encoding="utf-8"))
    selected = [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()]
    assert extraction["code_hash_mismatch_count"] == 0
    assert extraction["missing_parent_count"] == 0
    assert extraction["lineage_source_counts"] == {
        "raw_journal_node2parent": 1
    }
    assert extraction["selected_count"] == 1
    assert selected[0]["source_artifact_id"] == "run::run-1"
    assert selected[0]["parent_artifact_id"] == "run::run-1::node::parent"
    assert selected[0]["child_artifact_id"] == "run::run-1::node::child"
    assert selected[0]["original_claim_id"] == "claim::method"
    assert selected[0]["source_clause_id"] == "clause::method"
    assert selected[0]["code_sha256"] == code_hash
    assert selected[0]["historical_metric_used_as_evidence"] is False
