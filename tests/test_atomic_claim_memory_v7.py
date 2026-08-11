from __future__ import annotations

import json
from pathlib import Path
import sys

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
from agents.memory.experiment_r_router import _debug_repair_evidence  # noqa: E402
from agents.memory.stage_aware_hybrid_memory import (  # noqa: E402
    StageAwareHybridMemoryLayer,
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
