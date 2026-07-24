from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from authority.memory_snapshot import (
    MemorySnapshotLoader,
    SessionOverlay,
    make_current_pointer,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from authority.models import Operation
from authority.models import GenerationStage
from tests.authority.sop_visibility_helpers import (
    FORBIDDEN_SCORE_TEXT,
    MIXED_SOP_ID,
    REPAIR_CLAUSE_ID,
    build_mixed_authority,
    visibility_request,
    write_stage_fixture,
)


PROTOCOL = "test-protocol@1#" + "a" * 64


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def build_tiny_bundle(root: Path, version: str = "v1") -> tuple[Path, dict]:
    bundle = root / "bundles" / version
    clauses = [
        {
            "clause_id": "clause-a",
            "sop_id": "sop-a",
            "text": "Use a clean feature path.",
            "retrieval_text": "Use a clean feature path.",
            "claim_types": ["method_hypothesis"],
            "protocol_scope": [PROTOCOL],
            "task_scope": {"task_ids": ["task-a"]},
            "permitted_operations": ["inspect", "rank"],
            "publication_class": "candidate",
        },
        {
            "clause_id": "clause-other-task",
            "sop_id": "sop-b",
            "text": "Other task only.",
            "retrieval_text": "Other task only.",
            "claim_types": ["method_hypothesis"],
            "source_task_ids": ["task-b"],
            "source_task_families": ["General Image"],
            "source_domains": ["image"],
            "transfer_scope": "same_domain",
            "protocol_scope": [PROTOCOL],
            "task_scope": {"task_ids": ["task-b"]},
            "permitted_operations": ["inspect", "rank"],
            "publication_class": "candidate",
        },
    ]
    _write_jsonl(bundle / "sop" / "clauses.jsonl", clauses)
    write_json_atomic(
        bundle
        / "visibility"
        / "precompiled_masks"
        / "declared_scope_masks.json",
        {
            "schema": "declared_scope_visibility_masks_v1",
            "semantics": "Declared-scope prefilter only; runtime Authority ALLOW is still required.",
            "masks": {
                f"{PROTOCOL}|rank|improve|retrieval": [
                    "clause-a",
                    "clause-other-task",
                ],
                f"{PROTOCOL}|inspect|improve|retrieval": [
                    "clause-a",
                    "clause-other-task",
                ],
            },
        },
    )
    write_json_atomic(
        bundle / "runforest" / "graph.json",
        {
            "schema": "tiny_graph_v1",
            "meta": {
                "source_membership_verified": True,
                "leak_verified": True,
                "source_runs": ["tiny-run"],
            },
            "nodes": [],
            "edges": [],
        },
    )
    (bundle / "runforest" / "index.npz").write_bytes(b"tiny-index")
    artifact_hashes = {
        "runforest/graph.json": sha256_file(bundle / "runforest" / "graph.json"),
        "runforest/index.npz": sha256_file(bundle / "runforest" / "index.npz"),
        "sop/clauses.jsonl": sha256_file(bundle / "sop" / "clauses.jsonl"),
        "visibility/precompiled_masks/declared_scope_masks.json": sha256_file(
            bundle
            / "visibility"
            / "precompiled_masks"
            / "declared_scope_masks.json"
        ),
    }
    manifest = {
        "schema": "memory_bundle_manifest_v1",
        "bundle_id": f"tiny-{version}",
        "bundle_version": version,
        "parent_bundle": None if version == "v1" else "tiny-v1",
        "corpus_manifest_hash": "b" * 64,
        "protocol_registry_hash": "c" * 64,
        "authority_policy_version": "authority_v1",
        "detector_version": "test",
        "deepseek_model": "frozen-test",
        "deepseek_prompt_hash": "d" * 64,
        "graph_hashes": {"runforest": artifact_hashes["runforest/graph.json"]},
        "index_hashes": {},
        "lineage_hash": "e" * 64,
        "split_id": "full-test",
        "certification_level": "raw_audited",
        "build_report": "runforest/graph.json",
        "created_at": "2026-07-19T00:00:00Z",
        "artifact_hashes": artifact_hashes,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    write_json_atomic(bundle / "manifest.json", manifest)
    return bundle, manifest


def write_current(root: Path, bundle: Path, manifest: dict) -> None:
    write_json_atomic(
        root / "CURRENT.json",
        make_current_pointer(
            bundle_path=str(bundle.relative_to(root)),
            manifest=manifest,
            parent_bundle=manifest.get("parent_bundle"),
            published_at="2026-07-19T00:00:00Z",
        ),
    )


def test_snapshot_loads_hash_verified_base_and_filters_before_use(tmp_path: Path) -> None:
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "session-overlay",
        active_protocol_ref=PROTOCOL,
        authority_policy_version="authority_v1",
    )

    assert snapshot.base_bundle_id == "tiny-v1"
    assert snapshot.base_bundle_path == str(bundle.resolve())
    assert len(snapshot.snapshot_sha256) == 64
    assert [row["clause_id"] for row in snapshot.base_clauses(
        Operation.RANK,
        task_id="task-a",
        task_family="tabular",
        generation_stage="improve",
        governance_stage="retrieval",
    )] == ["clause-a"]
    assert snapshot.base_clauses(
        Operation.RANK,
        task_id="task-a",
        task_family="tabular",
        generation_stage="draft",
        governance_stage="retrieval",
    ) == []
    with pytest.raises(PermissionError, match="immutable"):
        snapshot.base_bundle.write("anything")


def test_snapshot_prefilter_keeps_only_same_domain_cross_task_clauses(
    tmp_path: Path,
) -> None:
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "session-overlay",
        active_protocol_ref=PROTOCOL,
        authority_policy_version="authority_v1",
    )

    same_domain = snapshot.base_clauses(
        Operation.RANK,
        task_id="aerial-cactus-identification",
        task_family="image_binary_classification",
        generation_stage="improve",
        governance_stage="retrieval",
    )
    assert [row["clause_id"] for row in same_domain] == [
        "clause-other-task"
    ]
    for target_family in ("text_classification", "audio_classification", ""):
        assert snapshot.base_clauses(
            Operation.RANK,
            task_id="cross-domain-target",
            task_family=target_family,
            generation_stage="improve",
            governance_stage="retrieval",
        ) == []


def test_agent_search_bundle_config_uses_current_base_paths(tmp_path: Path) -> None:
    from engine.agent_search import AgentSearch

    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)

    class FakeAuthority:
        active_protocol = SimpleNamespace(key=lambda: PROTOCOL)
        engine = SimpleNamespace(policy_version="authority_v1")

        def configure_memory_snapshot(self, snapshot) -> None:
            self.snapshot = snapshot

    authority = FakeAuthority()
    ext_cfg = SimpleNamespace(
        bundle_root=str(tmp_path),
        current_pointer_path="CURRENT.json",
        session_overlay_path="overlay",
    )
    snapshot, graph_path, index_path = AgentSearch._load_configured_memory_snapshot(
        ext_cfg,
        log_dir=tmp_path / "run-logs",
        evaluation_authority=authority,
        resolve_memory_path=lambda value: Path(value).resolve(),
    )

    assert authority.snapshot is snapshot
    assert Path(graph_path) == bundle / "runforest" / "graph.json"
    assert Path(index_path) == bundle / "runforest" / "index.npz"
    assert snapshot.session_overlay_path == str((tmp_path / "run-logs" / "overlay").resolve())


def test_loaded_base_detects_any_post_load_mutation(tmp_path: Path) -> None:
    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "session-overlay",
        active_protocol_ref=PROTOCOL,
        authority_policy_version="authority_v1",
    )
    (bundle / "runforest" / "graph.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        snapshot.assert_unchanged()


def test_overlay_is_append_only_and_hash_chained(tmp_path: Path) -> None:
    overlay = SessionOverlay(tmp_path / "overlay", overlay_id="overlay-test")
    first = overlay.append("diagnostic", {"claim_type": "audit_finding"})
    second = overlay.append("sop_clause", {"claim_type": "method_hypothesis"})
    assert first.sequence == 1
    assert second.sequence == 2
    assert second.parent_event_hash == first.event_hash
    assert overlay.manifest["event_count"] == 2

    reloaded = SessionOverlay(tmp_path / "overlay", overlay_id="overlay-test")
    assert [event.event_id for event in reloaded.events()] == [
        first.event_id,
        second.event_id,
    ]


def test_overlay_tampering_is_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "overlay"
    overlay = SessionOverlay(path, overlay_id="overlay-test")
    overlay.append("diagnostic", {"claim_type": "audit_finding"})
    events_path = path / "events.jsonl"
    raw = events_path.read_text(encoding="utf-8")
    events_path.write_text(raw.replace("audit_finding", "score"), encoding="utf-8")
    with pytest.raises(ValueError, match="event hash mismatch"):
        SessionOverlay(path, overlay_id="overlay-test")


def test_unaudited_overlay_score_is_inspect_only(tmp_path: Path) -> None:
    overlay = SessionOverlay(tmp_path / "overlay")
    score = overlay.append(
        "memory_claim",
        {
            "claim_type": "score",
            "audited": False,
            "permitted_operations": ["inspect", "rank"],
        },
    )
    method = overlay.append(
        "memory_claim",
        {
            "claim_type": "method_hypothesis",
            "audited": True,
            "permitted_operations": ["inspect", "rank"],
        },
    )
    nested_score = overlay.append(
        "sop_clause",
        {
            "clause": {
                "clause_id": "nested-score",
                "sop_id": "nested-score-sop",
                "claim_types": ["score"],
                "permitted_operations": ["inspect", "rank"],
            }
        },
    )
    assert {event.event_id for event in overlay.visible_events(Operation.INSPECT)} == {
        score.event_id,
        method.event_id,
        nested_score.event_id,
    }
    ranked = overlay.visible_events(
        Operation.RANK,
        authority_evaluator=lambda _event, _operation: True,
    )
    assert [event.event_id for event in ranked] == [method.event_id]
    assert overlay.visible_events(Operation.RANK) == []


def test_precompiled_mask_filters_clause_before_ranking(tmp_path: Path) -> None:
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    engine, ref = build_mixed_authority()
    graph_path, index_path = write_stage_fixture(tmp_path / "stage", ref)
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        enable_agentic=False,
        top_k=6,
        visibility_mode="enforce",
        visibility_authority_engine=engine,
        visibility_active_protocol=ref,
        visibility_policy_version=engine.policy_version,
        visibility_task_id="task-1",
        visibility_bundle_version="bundle-v1",
        visibility_token_budget=4096,
    )
    calls = []

    def base_clauses(operation, **scope):
        calls.append((operation, scope))
        return [{"clause_id": REPAIR_CLAUSE_ID}]

    layer.memory_snapshot = SimpleNamespace(base_clauses=base_clauses)
    layer._overlay_clause_ids = set()
    request = visibility_request(
        ref,
        Operation.DEBUG_HYPOTHESIS,
        generation_stage=GenerationStage.DEBUG,
    )
    pack = layer._prepare_visibility(
        stage="debug",
        task_id="task-1",
        task_desc="tabular OOF repair",
        request=request,
    )
    ranked = layer._rank_sops(
        FORBIDDEN_SCORE_TEXT,
        "debug",
        6,
        task_id="task-1",
        task_desc="tabular OOF repair",
    )

    assert calls == [
        (
            Operation.DEBUG_HYPOTHESIS,
            {
                "task_id": "task-1",
                "task_family": "tabular",
                "generation_stage": "debug",
                "governance_stage": "retrieval",
            },
        )
    ]
    assert pack.visibility_trace["precompiled_candidate_clause_ids"] == [
        REPAIR_CLAUSE_ID
    ]
    assert pack.visibility_trace["embedding_candidate_clause_ids"] == [
        REPAIR_CLAUSE_ID
    ]
    assert pack.visibility_trace["rrf_eligible_clause_ids"] == [
        REPAIR_CLAUSE_ID
    ]
    assert [row["id"] for row in ranked] == [MIXED_SOP_ID]
    assert FORBIDDEN_SCORE_TEXT not in ranked[0]["visible_text"]


def test_current_pointer_cannot_load_staging_directory(tmp_path: Path) -> None:
    staging = tmp_path / ".staging-v2"
    _bundle, manifest = build_tiny_bundle(staging.parent / staging.name)
    actual_bundle = staging / "bundles" / "v1"
    write_json_atomic(
        tmp_path / "CURRENT.json",
        make_current_pointer(
            bundle_path=str(actual_bundle.relative_to(tmp_path)),
            manifest=manifest,
            parent_bundle=None,
        ),
    )
    with pytest.raises(ValueError, match="not loadable"):
        MemorySnapshotLoader(tmp_path).load(
            session_overlay_path=tmp_path / "overlay",
            active_protocol_ref=PROTOCOL,
            authority_policy_version="authority_v1",
        )


def test_stage_hybrid_materializes_overlay_but_gates_it_before_rank(
    tmp_path: Path,
) -> None:
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    engine, ref = build_mixed_authority()
    bundle_path = tmp_path / "bundle"
    graph_path, index_path = write_stage_fixture(
        bundle_path / "runforest", ref
    )
    overlay = SessionOverlay(tmp_path / "session-overlay")
    overlay.append(
        "sop_clause",
        {
            "claim_type": "score",
            "audited": False,
            "clause": {
                "clause_id": "clause::overlay-score",
                "sop_id": "sop::overlay-score",
                "text": "Unreviewed overlay score 0.99.",
                "retrieval_text": "Unreviewed overlay score 0.99.",
                "claim_refs": ["claim::not-registered"],
                "claim_types": ["score"],
                "source_artifact_refs": ["artifact::overlay"],
                "protocol_scope": [ref.key()],
                "task_scope": {"task_ids": ["task-1"]},
                "permitted_operations": ["inspect", "rank"],
                "permitted_generation_stages": ["improve"],
                "permitted_governance_stages": ["retrieval"],
                "publication_class": "candidate",
            },
        },
    )
    base = SimpleNamespace(path=bundle_path, bundle_version="v2")
    base.assert_unchanged = lambda: None
    snapshot = SimpleNamespace(base_bundle=base, session_overlay=overlay)
    snapshot.assert_unchanged = lambda: None
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        enable_agentic=False,
        top_k=6,
        visibility_mode="enforce",
        visibility_authority_engine=engine,
        visibility_active_protocol=ref,
        visibility_policy_version=engine.policy_version,
        visibility_task_id="task-1",
        visibility_bundle_version="v2",
        visibility_token_budget=4096,
        memory_snapshot=snapshot,
    )
    assert "sop::overlay-score" in layer._sops

    rank_pack = layer.visibility_gateway.evaluate(
        visibility_request(
            ref,
            Operation.RANK,
            generation_stage=GenerationStage.IMPROVE,
        ),
        candidate_sop_ids=["sop::overlay-score"],
    )
    assert rank_pack.effective_clause_ids == []
    assert rank_pack.experience_contracts == []

    inspect_pack = layer.visibility_gateway.evaluate(
        visibility_request(
            ref,
            Operation.INSPECT,
            generation_stage=GenerationStage.IMPROVE,
        ),
        candidate_sop_ids=["sop::overlay-score"],
    )
    assert inspect_pack.effective_clause_ids == ["clause::overlay-score"]
    assert len(inspect_pack.experience_contracts) == 1
