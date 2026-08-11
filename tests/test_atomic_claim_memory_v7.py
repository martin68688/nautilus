from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MLEVOLVE_ROOT = ROOT / "mlevolve"
if str(MLEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE_ROOT))

from agents.memory.atomic_claim_memory import (  # noqa: E402
    ATOMIC_CLAIM_SCHEMA,
    AUTHORIZED_DEBUG_STATUS,
    extract_debug_signature,
    structured_debug_relevance,
    verified_atomic_debug_claim,
)
from agents.memory.external_skill_memory import (  # noqa: E402
    RunForestMemoryLayer,
    _tokenize,
)
from agents.memory.experiment_r_router import (  # noqa: E402
    _debug_repair_evidence,
    _hard_gated_l3_candidates,
    _l3_policy_authorized_sop_ids,
    _shortlist_l3_candidates_for_agent,
)
from agents.memory.stage_aware_hybrid_memory import (  # noqa: E402
    StageAwareHybridMemoryLayer,
)
from authority.memory_snapshot import (  # noqa: E402
    MemorySnapshotLoader,
    make_current_pointer,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from authority.models import Operation, ProtocolRef  # noqa: E402
from authority.recipe_visibility_publication import (  # noqa: E402
    compile_recipe_debug_visibility,
)


ACTIVE_PROTOCOL = (
    "mlevolve-default@1#"
    "cdb8439fa96b3add95788d9c1462811e5319ef73bca1cb4cc606ce07cd7a3a29"
)


def _atomic_claim() -> dict:
    return {
        "schema": ATOMIC_CLAIM_SCHEMA,
        "id": "claim::leaf-classification::compatibility_claim::dino518",
        "task_id": "leaf-classification",
        "claim_type": "compatibility_claim",
        "claim_status": AUTHORIZED_DEBUG_STATUS,
        "failure_text": (
            "AssertionError: Input height (256) doesn't match model (518) "
            "for DINOv2 ViT-S/14."
        ),
        "repair_action": "Change IMG_SIZE from 256 to 518.",
        "failure_signature": extract_debug_signature(
            "AssertionError: Input height (256) doesn't match model (518) "
            "for DINOv2 ViT-S/14; IMG_SIZE."
        ),
        "before_after": [
            {"symbol": "IMG_SIZE", "before": "256", "after": "518"}
        ],
        "metric_authorized": False,
        "taint": {
            "code": "quarantine",
            "metric": "quarantine",
            "claim": "clean",
        },
        "verification": {
            "observed_parent_failure": True,
            "observed_child_execution_success": True,
            "repair_action_bound_to_transition": True,
            "claim_scope_independently_audited": True,
            "before_code_sha256": "a" * 64,
            "after_code_sha256": "b" * 64,
        },
        "operation_visibility": {
            "allowed_operations": ["debug_hypothesis", "debug_repair"],
            "forbidden_operations": [
                "draft_method_selection",
                "improve_method_selection",
                "metric_ranking",
                "exact_replay",
            ],
            "task_scope": "exact_task",
        },
    }


def _atomic_transition() -> dict:
    return {
        "id": "atomic-transition::dino518",
        "type": "Transition",
        "task": "leaf-classification",
        "run_id": "historical-leaf",
        "parent_node_id": "parent",
        "child_node_id": "child",
        "stage_pair": "debug->debug",
        "outcome": "debug_fixed",
        "parent_buggy": True,
        "child_buggy": False,
        "atomic_repair_claim": _atomic_claim(),
    }


def test_debug_signature_retains_exception_model_numbers_and_symbol() -> None:
    signature = extract_debug_signature(
        "AssertionError: Input height (224) doesn't match model (518) in "
        "vit_small_patch14_dinov2.lvd142m; set IMG_SIZE=518."
    )
    assert "assertionerror" in signature["exception_names"]
    assert "224" in signature["numeric_literals"]
    assert "518" in signature["numeric_literals"]
    assert any("dinov2" in value for value in signature["model_api_ids"])
    assert "img_size" in signature["symbol_names"]


def test_structured_rank_matches_v76_224_to_historical_256_to_518() -> None:
    claim = _atomic_claim()
    score, receipt = structured_debug_relevance(
        "AssertionError: Input height (224) doesn't match model (518) "
        "for vit_small_patch14_dinov2.lvd142m",
        claim["failure_text"],
        claim["repair_action"],
        claim,
    )
    generic_score, _ = structured_debug_relevance(
        "AssertionError: Input height (224) doesn't match model (518) "
        "for vit_small_patch14_dinov2.lvd142m",
        "AssertionError while creating a generic tensor",
        "inspect the shape",
        {},
    )
    assert score >= 0.92
    assert score > generic_score
    assert receipt["exact_compatibility_match"] is True
    assert receipt["shared_expected_values"] == ["518"]


def test_l3_agent_shortlist_promotes_exact_repair_before_large_decoy_pool() -> None:
    claim = _atomic_claim()
    exact = {
        "sop_id": "repair-claim::leaf-classification::dino518",
        "transition_id": "atomic-transition::dino518",
        "supporting_transition_ids": ["atomic-transition::dino518"],
        "failure_signature": claim["failure_signature"],
        "repair_action": {
            "summary": claim["repair_action"],
            "before_after": claim["before_after"],
        },
        "historical_failure": claim["failure_text"],
        "historical_code_change": claim["repair_action"],
    }
    decoys = [
        {
            "sop_id": f"repair-claim::leaf-classification::decoy-{index:03d}",
            "transition_id": f"atomic-transition::decoy-{index:03d}",
            "supporting_transition_ids": [f"atomic-transition::decoy-{index:03d}"],
            "failure_signature": extract_debug_signature(
                f"ValueError: unrelated tabular feature failure {index}"
            ),
            "repair_action": {"summary": "change an unrelated table feature"},
            "historical_failure": f"ValueError in tabular feature {index}",
            "historical_code_change": "change an unrelated table feature",
        }
        for index in range(295)
    ]
    selected, receipt = _shortlist_l3_candidates_for_agent(
        "AssertionError: Input image height 256 is not a multiple of patch "
        "height 14 while dinov2_vitl14 extracts frozen features",
        [*decoys, exact],
        limit=8,
    )
    assert receipt["input_candidate_count"] == 296
    assert receipt["output_candidate_count"] == 8
    assert selected[0]["sop_id"] == exact["sop_id"]
    assert selected[0]["agent_shortlist_rank"] == 1
    assert selected[0]["agent_shortlist_score"] > selected[1]["agent_shortlist_score"]


def test_l3_agent_shortlist_unions_structured_and_dense_top_eight() -> None:
    candidates = [
        {
            "sop_id": f"repair::decoy-{index:02d}",
            "transition_id": f"transition::decoy-{index:02d}",
            "supporting_transition_ids": [f"transition::decoy-{index:02d}"],
            "failure_signature": {},
            "repair_action": {"summary": "unrelated parser repair"},
            "historical_failure": "generic parser failure",
            "historical_code_change": "change parser",
        }
        for index in range(12)
    ]
    semantic_target = {
        "sop_id": "repair::semantic-target",
        "transition_id": "transition::semantic-target",
        "supporting_transition_ids": ["transition::semantic-target"],
        "failure_signature": {},
        "repair_action": {"summary": "reduce CUDA batch memory pressure"},
        "historical_failure": "CUDA out of memory while allocating a tensor",
        "historical_code_change": "reduce the batch size",
    }
    candidates.append(semantic_target)

    def semantic_encoder(texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            equivalent_oom = (
                "accelerator workspace depleted" in lowered
                or "cuda out of memory" in lowered
            )
            rows.append([1.0, 0.0] if equivalent_oom else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)

    selected, receipt = _shortlist_l3_candidates_for_agent(
        "RuntimeError: accelerator workspace depleted during tensor allocation",
        candidates,
        limit=8,
        semantic_encode_fn=semantic_encoder,
        semantic_limit=8,
        semantic_model_id="test-semantic-encoder",
    )
    by_id = {row["sop_id"]: row for row in selected}
    assert semantic_target["sop_id"] in by_id
    assert by_id[semantic_target["sop_id"]]["agent_shortlist_semantic_rank"] == 1
    assert by_id[semantic_target["sop_id"]]["agent_shortlist_structured_rank"] is None
    assert receipt["semantic"]["status"] == "ok"
    assert receipt["semantic"]["model_id"] == "test-semantic-encoder"
    assert 8 <= receipt["output_candidate_count"] <= 16
    assert len({row["sop_id"] for row in selected}) == len(selected)


def test_l3_agent_shortlist_records_semantic_degradation_without_losing_structured_route() -> None:
    claim = _atomic_claim()
    candidate = {
        "sop_id": "repair::structured-only",
        "transition_id": "transition::structured-only",
        "supporting_transition_ids": ["transition::structured-only"],
        "failure_signature": claim["failure_signature"],
        "repair_action": {
            "summary": claim["repair_action"],
            "before_after": claim["before_after"],
        },
        "historical_failure": claim["failure_text"],
        "historical_code_change": claim["repair_action"],
    }
    selected, receipt = _shortlist_l3_candidates_for_agent(
        claim["failure_text"],
        [candidate],
        limit=8,
        semantic_limit=8,
    )
    assert [row["sop_id"] for row in selected] == [candidate["sop_id"]]
    assert receipt["semantic"]["status"] == "encoder_unavailable"
    assert selected[0]["agent_shortlist_routes"] == ["structured_causal"]


def test_l3_permission_pool_uses_all_policy_authorized_cards_before_prompt_budget() -> None:
    layer = SimpleNamespace(
        _visibility_is_enforced=lambda: True,
        _trace_local=SimpleNamespace(
            visibility_pack=SimpleNamespace(
                visibility_trace={
                    "full_policy_visible_clause_ids": ["clause::authorized"]
                }
            )
        ),
        visibility_gateway=SimpleNamespace(
            clauses={
                "clause::authorized": SimpleNamespace(
                    sop_id="repair::policy-authorized"
                )
            }
        ),
    )
    result = _l3_policy_authorized_sop_ids(
        layer, {"repair::prompt-budget-survivor"}
    )
    assert result == {"repair::policy-authorized"}


def test_l3_semantic_encoder_loads_lazily_when_global_memory_is_off(monkeypatch) -> None:
    import agents.memory.embedding_models as embedding_models

    observed: dict[str, object] = {}

    class FakeEmbeddingModel:
        def __init__(self, **kwargs):
            observed.update(kwargs)

        def encode(self, texts, show_progress_bar=False):
            observed["texts"] = list(texts)
            observed["show_progress_bar"] = show_progress_bar
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setattr(embedding_models, "EmbeddingModel", FakeEmbeddingModel)
    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer._experiment_r_l3_semantic_model = None
    layer._experiment_r_l3_semantic_model_path = "BAAI/bge-base-en-v1.5"
    layer._experiment_r_l3_semantic_device = "cpu"
    vectors = layer._encode_l3_semantic_texts(["query", "repair"])
    assert vectors.shape == (2, 3)
    assert observed["model_name"] == "BAAI/bge-base-en-v1.5"
    assert observed["device"] == "cpu"
    assert observed["texts"] == ["query", "repair"]
    assert observed["show_progress_bar"] is False


def test_atomic_claim_gate_separates_program_metric_and_local_repair() -> None:
    transition = _atomic_transition()
    eligible, reason = verified_atomic_debug_claim(transition)
    assert eligible is True
    assert reason == "verified_atomic_debug_claim"
    assert transition["atomic_repair_claim"]["taint"]["code"] == "quarantine"
    assert transition["atomic_repair_claim"]["metric_authorized"] is False


def test_runtime_accepts_atomic_repair_but_never_as_replay_candidate() -> None:
    transition = _atomic_transition()
    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer.nodes = {
        transition["id"]: transition,
        "parent": {"id": "parent", "type": "RunNode", "is_buggy": True},
        "child": {"id": "child", "type": "RunNode", "is_buggy": False},
    }
    positive, positive_reason = layer._positive_transition(transition["id"])
    replayable, replay_reason = layer._execution_candidate_eligibility(
        transition["id"]
    )
    evidence, evidence_reason = _debug_repair_evidence(layer, transition["id"])
    assert positive is True
    assert positive_reason == "verified_atomic_debug_claim"
    assert replayable is False
    assert replay_reason == "atomic_claim_debug_only"
    assert evidence is not None
    assert evidence_reason == "safe_verified_atomic_debug_claim"
    assert evidence["evidence_mode"] == "verified_atomic_claim_no_program_or_metric"
    assert evidence["transition_evidence"]["metric_authorized"] is False


def test_general_rank_hard_filters_task_before_anchor() -> None:
    layer = RunForestMemoryLayer.__new__(RunForestMemoryLayer)
    layer.scoring_mode = "euclidean"
    layer.nodes = {
        "leaf-exact": {
            "id": "leaf-exact",
            "task": "leaf-classification",
            "analysis": "DINOv2 AssertionError input height 256 model 518",
        },
        "leaf-generic": {
            "id": "leaf-generic",
            "task": "leaf-classification",
            "analysis": "generic runtime error",
        },
        "other-noisy": {
            "id": "other-noisy",
            "task": "aerial-cactus-identification",
            "analysis": (
                "DINOv2 AssertionError input height 224 model 518 IMG_SIZE "
                "exact traceback"
            ),
        },
    }
    layer._node_tokens = {
        node_id: _tokenize(layer._node_text(node))
        for node_id, node in layer.nodes.items()
    }
    layer._euclidean_coords = {
        "leaf-exact": np.asarray([0.1, 0.1], dtype=np.float32),
        "leaf-generic": np.asarray([0.2, 0.2], dtype=np.float32),
        "other-noisy": np.asarray([0.0, 0.0], dtype=np.float32),
    }
    layer._poincare_coords = dict(layer._euclidean_coords)
    ranked = layer._rank_with_scores(
        query_text="DINOv2 AssertionError input height 224 model 518",
        candidate_ids=["other-noisy", "leaf-generic", "leaf-exact"],
        task_id="leaf-classification",
        task_desc="Leaf classification",
        top_k=3,
    )
    assert [node_id for _score, node_id in ranked] == [
        "leaf-exact",
        "leaf-generic",
    ]


def test_recipe_visibility_compiler_is_claim_bound_and_debug_only() -> None:
    recipe = {
        "schema": "recipe-test-v1",
        "bundle_sha256": "a" * 64,
        "nodes": [
            {
                "id": "repair::task-a::claim-backed",
                "type": "SOP",
                "abstraction_level": "L3_repair",
                "title": "Repair an exact runtime mismatch",
                "task_id": "task-a",
                "task_family": "multimodal_classification",
                "failure_node_ids": ["artifact::failed"],
                "source_claim_ids": ["claim::debug-repair"],
                "supporting_transition_ids": ["transition::repair"],
                "distinct_run_ids": ["run::historical"],
                "failure_signature": {
                    "pattern": "AssertionError: input 256 expected 518"
                },
                "repair_action": {"summary": "Change input size to 518."},
                "when_to_use": "Only for the matching model assertion.",
            },
            {
                "id": "repair::task-a::legacy",
                "type": "SOP",
                "abstraction_level": "L3_repair",
                "task_id": "task-a",
                "failure_signature": {"pattern": "legacy failure"},
                "repair_action": {"summary": "legacy repair"},
            },
            {
                "id": "method::task-a",
                "type": "SOP",
                "abstraction_level": "L1_recipe",
            },
        ],
    }
    publication = compile_recipe_debug_visibility(
        recipe,
        active_protocol_ref=ACTIVE_PROTOCOL,
    )
    assert publication["report"]["l3_recipe_count"] == 2
    assert publication["report"]["published_clause_count"] == 1
    assert publication["report"]["skipped_reason_counts"] == {
        "missing_source_claim_ids": 1
    }
    clause = publication["clauses"][0]
    assert clause["claim_refs"] == ["claim::debug-repair"]
    assert clause["claim_types"] == ["debug_repair"]
    assert clause["permitted_operations"] == ["debug_hypothesis"]
    assert clause["permitted_generation_stages"] == ["debug"]
    assert clause["permitted_governance_stages"] == ["retrieval"]
    assert clause["publication_class"] == "diagnostic"
    assert clause["transfer_scope"] == ""
    mask_ids = next(iter(publication["masks"]["masks"].values()))
    assert mask_ids == [clause["clause_id"]]


def _build_real_v8_visibility_bundle(
    tmp_path: Path,
    release: Path,
) -> tuple[Path, Path, dict]:
    root = tmp_path / "memory-leaf-atomic-v8"
    bundle = root / "bundles" / "v8-test"
    recipe_dir = bundle / "recipe"
    runforest_dir = bundle / "runforest"
    recipe_dir.mkdir(parents=True)
    runforest_dir.mkdir(parents=True)
    for name in (
        "recipe_sops.json",
        "evidence_manifest.json",
        "implementation_capsules.json",
    ):
        shutil.copy2(release / name, recipe_dir / name)
    for name in ("graph.json", "index.npz"):
        shutil.copy2(release / "runforest" / name, runforest_dir / name)

    recipe = json.loads((recipe_dir / "recipe_sops.json").read_text())
    publication = compile_recipe_debug_visibility(
        recipe,
        active_protocol_ref=ACTIVE_PROTOCOL,
    )
    clauses_path = bundle / "sop" / "clauses.jsonl"
    clauses_path.parent.mkdir(parents=True)
    clauses_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in publication["clauses"]
        ),
        encoding="utf-8",
    )
    masks_path = (
        bundle
        / "visibility"
        / "precompiled_masks"
        / "declared_scope_masks.json"
    )
    write_json_atomic(masks_path, publication["masks"])
    artifact_hashes = {
        str(path.relative_to(bundle)): sha256_file(path)
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": "memory_bundle_manifest_v1",
        "bundle_id": "leaf-atomic-v8-test",
        "bundle_version": "v8-test",
        "parent_bundle": "v7-test",
        "authority_policy_version": "claim-level-test-v1",
        "certification_level": "atomic_debug_visibility_exploratory_smoke",
        "artifact_hashes": artifact_hashes,
        "manifest_sha256": "",
    }
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    write_json_atomic(bundle / "manifest.json", manifest)
    write_json_atomic(
        root / "CURRENT.json",
        make_current_pointer(
            bundle_path=str(bundle.relative_to(root)),
            manifest=manifest,
            parent_bundle=manifest["parent_bundle"],
            published_at="2026-08-12T00:00:00Z",
        ),
    )
    return root, bundle, publication


def test_real_bundle_scoped_enforce_reaches_hybrid_l3_shortlist(
    tmp_path: Path,
) -> None:
    release = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "recipe_distillation_v7_leaf_atomic_20260811"
    )
    if not (release / "release_report.json").exists():
        return
    root, bundle, publication = _build_real_v8_visibility_bundle(
        tmp_path,
        release,
    )
    snapshot = MemorySnapshotLoader(root).load(
        session_overlay_path=tmp_path / "session-overlay",
        active_protocol_ref=ACTIVE_PROTOCOL,
        authority_policy_version="claim-level-test-v1",
    )
    recipe = json.loads((bundle / "recipe" / "recipe_sops.json").read_text())
    evidence = json.loads(
        (bundle / "recipe" / "evidence_manifest.json").read_text()
    )
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(bundle / "runforest" / "graph.json"),
        index_path=str(bundle / "runforest" / "index.npz"),
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        retrieval_control="dynamic_hybrid",
        enable_agentic=False,
        top_k=6,
        experiment_r_enabled=True,
        recipe_sop_path=str(bundle / "recipe" / "recipe_sops.json"),
        recipe_sop_file_sha256=sha256_file(
            bundle / "recipe" / "recipe_sops.json"
        ),
        recipe_sop_bundle_sha256=recipe["bundle_sha256"],
        recipe_evidence_path=str(bundle / "recipe" / "evidence_manifest.json"),
        recipe_evidence_file_sha256=sha256_file(
            bundle / "recipe" / "evidence_manifest.json"
        ),
        recipe_evidence_manifest_sha256=evidence["manifest_sha256"],
        recipe_implementation_path=str(
            bundle / "recipe" / "implementation_capsules.json"
        ),
        visibility_mode="enforce",
        visibility_active_protocol=ProtocolRef(
            "mlevolve-default",
            "1",
            ACTIVE_PROTOCOL.split("#", 1)[1],
        ),
        visibility_policy_version="claim-level-test-v1",
        visibility_task_id="leaf-classification",
        visibility_bundle_version="v8-test",
        visibility_enforce_operations=["debug_hypothesis"],
        visibility_enforce_generation_stages=["debug"],
        visibility_enforce_governance_stages=["retrieval"],
        memory_snapshot=snapshot,
    )
    assert layer.base_clause_receipt["status"] == "loaded"
    assert layer.base_clause_receipt["clause_count"] == 296
    assert publication["report"]["published_clause_count"] == 296

    request = layer._visibility_request(
        stage="debug",
        task_id="leaf-classification",
        task_desc="multimodal leaf image classification",
        operation=Operation.DEBUG_HYPOTHESIS,
    )
    pack = layer._prepare_visibility(
        stage="debug",
        task_id="leaf-classification",
        task_desc="multimodal leaf image classification",
        request=request,
    )
    assert pack.visibility_trace["request_enforced"] is True
    assert len(pack.visibility_trace["precompiled_candidate_clause_ids"]) == 296
    assert len(pack.visibility_trace["full_policy_visible_clause_ids"]) == 296
    authorized_sop_ids = _l3_policy_authorized_sop_ids(
        layer,
        set(layer._recipe_sop_ids),
    )
    assert len(authorized_sop_ids) == 296
    authorized = _hard_gated_l3_candidates(
        layer,
        task_id="leaf-classification",
        task_desc="multimodal leaf image classification",
        visible_sop_ids=authorized_sop_ids,
        task_scope="exact_task",
    )
    assert len(authorized) == 296

    def semantic_encoder(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)

    selected, receipt = _shortlist_l3_candidates_for_agent(
        "AssertionError: Input height (256) doesn't match model (518) "
        "for DINOv2 ViT-S/14",
        authorized,
        limit=8,
        semantic_encode_fn=semantic_encoder,
        semantic_limit=8,
        semantic_model_id="BAAI/bge-base-en-v1.5",
    )
    assert receipt["schema"] == "experiment_r_l3_agent_shortlist_v2"
    assert receipt["input_candidate_count"] == 296
    assert receipt["semantic"]["status"] == "ok"
    assert receipt["semantic"]["model_id"] == "BAAI/bge-base-en-v1.5"
    assert 8 <= receipt["output_candidate_count"] <= 16
    assert receipt["structured_top_ids"][0] == (
        "repair-claim::leaf-classification::eb08c01baf682841c035"
    )
    assert selected[0]["sop_id"] == (
        "repair-claim::leaf-classification::eb08c01baf682841c035"
    )


def test_generated_release_covers_v45_claim_when_present() -> None:
    release_dir = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "recipe_distillation_v7_leaf_atomic_20260811"
    )
    if not (release_dir / "release_report.json").exists():
        return
    report = json.loads((release_dir / "release_report.json").read_text())
    bundle = json.loads((release_dir / "atomic_claims.json").read_text())
    assert report["quality_gates"]["all_leaf_transitions_covered"] is True
    matches = [
        claim
        for claim in bundle["claims"]
        if "source-72bdeafd::transition::8e77146c99c9::c18a69664d3c"
        in claim["source_transition_id"]
    ]
    assert len(matches) == 1
    assert matches[0]["claim_status"] == AUTHORIZED_DEBUG_STATUS
    assert matches[0]["taint"]["code"] == "quarantine"
    assert matches[0]["taint"]["claim"] == "clean"


def test_v77_and_v78_configs_bind_atomic_recipe_overlays() -> None:
    systems_root = (
        ROOT / "experiments" / "end2end_memory_systems_20260804"
    )
    for version in (77, 78, 79):
        text = (
            systems_root / f"systems_v{version}" / "dynamic_hybrid.yaml"
        ).read_text(encoding="utf-8")
        memory_root = (
            f"/workspace/experiment-end2end-memory-agent-v{version}/"
            "memory-leaf-atomic-v7/leaf-classification/"
            "bundles/v7-leaf-atomic-20260811/recipe"
        )
        assert f"recipe_sop_path: {memory_root}/recipe_sops.json" in text
        assert f"recipe_evidence_path: {memory_root}/evidence_manifest.json" in text
        assert (
            f"recipe_implementation_path: {memory_root}/implementation_capsules.json"
            in text
        )


def test_generated_release_runtime_rank_when_artifacts_are_present() -> None:
    release = (
        ROOT
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "recipe_distillation_v7_leaf_atomic_20260811"
    )
    if not (release / "runforest" / "graph.json").exists():
        return
    report = json.loads((release / "release_report.json").read_text())
    recipe = json.loads((release / "recipe_sops.json").read_text())
    evidence = json.loads((release / "evidence_manifest.json").read_text())
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(release / "runforest" / "graph.json"),
        index_path=str(release / "runforest" / "index.npz"),
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        retrieval_control="dynamic_hybrid",
        enable_agentic=False,
        top_k=6,
        experiment_r_enabled=True,
        recipe_sop_path=str(release / "recipe_sops.json"),
        recipe_sop_file_sha256=report["files"]["recipe_sops.json"],
        recipe_sop_bundle_sha256=recipe["bundle_sha256"],
        recipe_evidence_path=str(release / "evidence_manifest.json"),
        recipe_evidence_file_sha256=report["files"]["evidence_manifest.json"],
        recipe_evidence_manifest_sha256=evidence["manifest_sha256"],
        recipe_implementation_path=str(release / "implementation_capsules.json"),
    )
    cases = (
        (
            "AssertionError: Input height (224) does not match model (518) "
            "for vit_small_patch14_dinov2.lvd142m",
            ("dinov2", "518"),
        ),
        (
            "FileNotFoundError: ./working/dinov3-main/hubconf.py not found "
            "while torch.hub.load loads DINOv3",
            ("dinov3", "hubconf.py"),
        ),
        (
            "ValueError: operands could not be broadcast together with shapes "
            "(891,31) (891,32) in create_hierarchical_features symmetry_2",
            ("broadcast", "31", "32"),
        ),
        (
            "TypeError: LGBMClassifier.fit() got an unexpected keyword "
            "argument verbose",
            ("lgbmclassifier.fit", "verbose"),
        ),
        (
            "ValueError: Input contains NaN when sklearn.metrics.log_loss "
            "evaluates probabilities",
            ("input contains nan", "log_loss"),
        ),
        (
            "RuntimeError: DataLoader worker exited unexpectedly with a bus "
            "error from insufficient shared memory and num_workers=4",
            ("shared memory", "num_workers=4"),
        ),
        (
            "TypeError: XGBClassifier.fit() got an unexpected keyword "
            "argument early_stopping_rounds",
            ("xgbclassifier.fit", "early_stopping_rounds"),
        ),
    )
    for query, needles in cases:
        rows = layer._rank_debug_transition_rows(
            query_text=query,
            task_id="leaf-classification",
            task_desc="multimodal leaf image classification",
            limit=8,
        )
        assert rows
        claim = layer.nodes[rows[0]["id"]]["atomic_repair_claim"]
        claim_text = (
            f"{claim.get('failure_text', '')}\n{claim.get('repair_action', '')}"
        ).lower()
        assert all(needle in claim_text for needle in needles)
        assert (
            rows[0]["ranking_backend"]
            == "task_first_structured_debug_signature_v3"
        )

    authorized = _hard_gated_l3_candidates(
        layer,
        task_id="leaf-classification",
        task_desc="multimodal leaf image classification",
        visible_sop_ids=None,
        task_scope="exact_task",
    )
    selected, receipt = _shortlist_l3_candidates_for_agent(
        "AssertionError: Input height (224) does not match model (518) "
        "for vit_small_patch14_dinov2.lvd142m",
        authorized,
        limit=8,
    )
    assert len(authorized) == 296
    assert selected[0]["sop_id"] == "repair-claim::leaf-classification::eb08c01baf682841c035"
    assert receipt["structured_top_ids"][0] == selected[0]["sop_id"]

    cleanup_rows = layer._rank_debug_transition_rows(
        query_text="NameError: name cleanup is not defined",
        task_id="leaf-classification",
        task_desc="multimodal leaf image classification",
        limit=8,
    )
    assert not any(
        row.get("ranking_backend")
        == "task_first_structured_debug_signature_v3"
        for row in cleanup_rows
    )
