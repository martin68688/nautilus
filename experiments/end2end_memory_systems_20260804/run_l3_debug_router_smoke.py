#!/usr/bin/env python3
"""Exercise the frozen layered L3 router against a real RunForest bundle.

This is a retrieval-only online smoke.  It deliberately covers exact-task,
same-task-type fallback, cross-type blocking, semantic paraphrase, and
infrastructure abstention before a short MLEvolve run is allowed to consume L3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mlevolve"))

from agents.memory.stage_aware_hybrid_memory import (  # noqa: E402
    L3_DYNAMIC_CONFIDENCE_WEIGHTS,
    StageAwareHybridMemoryLayer,
)
from config import _load_cfg  # noqa: E402


DEFAULT_FOURTASK_GRAPH = Path(
    "/workspace/nautilus/experiments/fourtask_graph_v2_replay_20260727/"
    "graph-inputs/run_forest_graph_fourtask_v2.json"
)
DEFAULT_FOURTASK_INDEX = DEFAULT_FOURTASK_GRAPH.with_name(
    "run_forest_index_fourtask_v2.npz"
)
LOCAL_FOURTASK_GRAPH = Path(
    "/Users/haoming/Downloads/nautilus/experiments/"
    "fourtask_graph_v2_replay_20260727/graph-inputs/"
    "run_forest_graph_fourtask_v2.json"
)
LOCAL_FOURTASK_INDEX = LOCAL_FOURTASK_GRAPH.with_name(
    "run_forest_index_fourtask_v2.npz"
)
CONFIG = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "systems"
    / "dynamic_hybrid.yaml"
)
RECIPE_BUNDLE = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "recipe_distillation_v2"
    / "recipe_sops.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_path(explicit: Path | None, primary: Path, local: Path) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if primary.exists():
        return primary
    return local


def compact_case(
    layer: StageAwareHybridMemoryLayer,
    *,
    name: str,
    task_id: str,
    task_desc: str,
    query: str,
) -> dict[str, Any]:
    prompt, refs = layer.retrieve_for_node(
        stage="debug",
        task_id=task_id,
        task_desc=task_desc,
        query_parts=[query],
        context={"runtime_smoke": "l3_debug_router"},
    )
    pack = layer.current_navigation_pack()
    tree = []
    for row in pack.get("tree_candidate_details") or []:
        tree.append(
            {
                "transition_id": row["id"],
                "source_task": row.get("task"),
                "task_scope": row.get("task_scope"),
                "confidence": row.get("confidence"),
                "score_components": row.get("score_components"),
                "dynamic_confidence_weights": row.get(
                    "dynamic_confidence_weights"
                ),
                "causal_attachments": row.get("causal_attachments"),
                "parent_node_id": row.get("parent_node_id"),
                "child_node_id": row.get("child_node_id"),
                "transition_evidence": row.get("transition_evidence"),
            }
        )
    return {
        "name": name,
        "task_id": task_id,
        "task_desc": task_desc,
        "query": query,
        "query_failure_signature": sorted(layer._failure_signature(query)),
        "prompt": prompt,
        "prompt_refs": refs,
        "selected_l3_ids": [
            row["id"] for row in pack.get("selected_sop_gateways") or []
        ],
        "stage_route": pack.get("stage_route"),
        "memory_abstention": pack.get("memory_abstention"),
        "tree_candidates": tree,
        "final_prompt_candidate_ids": pack.get("final_prompt_candidate_ids"),
        "gateway_transitions": pack.get("gateway_transitions"),
        "gateway_selection": pack.get("gateway_selection"),
    }


def require(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    graph = choose_path(args.graph, DEFAULT_FOURTASK_GRAPH, LOCAL_FOURTASK_GRAPH)
    index = choose_path(args.index, DEFAULT_FOURTASK_INDEX, LOCAL_FOURTASK_INDEX)
    if not graph.is_file() or not index.is_file():
        raise FileNotFoundError(f"Missing real RunForest graph/index: {graph} / {index}")

    cfg = _load_cfg(CONFIG, use_cli_args=False)
    cfg.exp_id = "leaf-classification"
    cfg.agent.search.num_gpus = 1
    # This harness tests the deterministic safety/ranking boundary.  The live
    # MLEvolve smoke keeps agentic gateway selection enabled through the same
    # dynamic_hybrid config after these hard assertions pass.
    cfg.external_skill_memory.enable_agentic = False
    cfg.external_skill_memory.experiment_r_l3_agent_match_enabled = False
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(graph),
        index_path=str(index),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        enable_agentic=False,
        top_k=6,
        max_chars=0,
        cfg=cfg,
    )

    cases = {
        "exact_semantic_paraphrase": compact_case(
            layer,
            name="exact_semantic_paraphrase",
            task_id="leaf-classification",
            task_desc=(
                "Leaf image and tabular multimodal multiclass classification "
                "evaluated by log loss."
            ),
            query=(
                "The augmented-label objective is a vector over the batch, so "
                "autograd refuses to backpropagate. Aggregate the per-example "
                "criterion before calling backward."
            ),
        ),
        "same_task_type_fallback": compact_case(
            layer,
            name="same_task_type_fallback",
            task_id="aerial-cactus-identification",
            task_desc="Vision binary image classification evaluated by ROC-AUC.",
            query=(
                "RuntimeError: a U-Net decoder's upsampled spatial lattice "
                "differs from the encoder skip lattice and concatenation fails. "
                "Resize or crop the decoder tensor to the skip height and width."
            ),
        ),
        "cross_task_type_block": compact_case(
            layer,
            name="cross_task_type_block",
            task_id="spooky-author-identification",
            task_desc="NLP multiclass text classification evaluated by log loss.",
            query=(
                "RuntimeError: a U-Net decoder's upsampled spatial lattice "
                "differs from the encoder skip lattice and concatenation fails. "
                "Resize or crop the decoder tensor to the skip height and width."
            ),
        ),
        "infrastructure_abstention": compact_case(
            layer,
            name="infrastructure_abstention",
            task_id="leaf-classification",
            task_desc=(
                "Leaf image and tabular multimodal multiclass classification "
                "evaluated by log loss."
            ),
            query=(
                "Permission denied in a node cache; a temporary file is missing, "
                "the Pod is Pending, and the API timed out."
            ),
        ),
    }

    checks: list[dict[str, Any]] = []
    exact = cases["exact_semantic_paraphrase"]
    require(
        exact["selected_l3_ids"] == ["repair::leaf-classification::002"],
        "semantic paraphrase selects only Leaf non-scalar MixUp repair",
        checks,
    )
    require(
        bool(exact["prompt"]) and exact["tree_candidates"],
        "exact-task L3 is present in the real Debug prompt",
        checks,
    )
    require(
        {row["task_scope"] for row in exact["tree_candidates"]}
        == {"exact_task"},
        "exact-task evidence suppresses all fallback evidence",
        checks,
    )
    require(
        all(
            row["score_components"]["task_match"] == 1.0
            and row["score_components"]["failure_signature_match"] >= 0.50
            for row in exact["tree_candidates"]
        ),
        "exact-task and failure-signature gates meet frozen thresholds",
        checks,
    )

    same_type = cases["same_task_type_fallback"]
    require(
        same_type["selected_l3_ids"]
        == ["repair::denoising-dirty-documents::002"],
        "Vision fallback selects the task-type-compatible Denoising repair",
        checks,
    )
    require(
        {row["task_scope"] for row in same_type["tree_candidates"]}
        == {"same_task_type"},
        "fallback evidence is explicitly labeled same_task_type",
        checks,
    )
    require(
        all(
            row["score_components"]["task_match"] == 0.70
            for row in same_type["tree_candidates"]
        ),
        "same-task-type fallback uses task_match=0.70",
        checks,
    )

    for name in ("cross_task_type_block", "infrastructure_abstention"):
        case = cases[name]
        require(
            case["prompt"] == ""
            and case["prompt_refs"] == []
            and case["selected_l3_ids"] == [],
            f"{name} injects no L3 memory",
            checks,
        )
        require(
            (case["memory_abstention"] or {}).get("status") == "abstain",
            f"{name} records an explicit abstention",
            checks,
        )

    recipe = json.loads(RECIPE_BUNDLE.read_text(encoding="utf-8"))
    l3_nodes = [
        node
        for node in recipe["nodes"]
        if node.get("abstraction_level") == "L3_repair"
    ]
    require(len(l3_nodes) == 35, "frozen bundle contains 35 L3 repairs", checks)
    require(
        {node.get("evidence_status") for node in l3_nodes}
        == {"accepted_clean_repair"},
        "all admitted L3 repairs share one evidence status",
        checks,
    )
    require(
        not any(
            marker in case["prompt"].lower()
            for case in cases.values()
            for marker in ("pod pending", "api timed out", "node cache")
        ),
        "infrastructure incidents are absent from all injected prompts",
        checks,
    )
    require(
        L3_DYNAMIC_CONFIDENCE_WEIGHTS
        == {
            "task_match": 0.40,
            "failure_signature_match": 0.30,
            "runtime_stage_match": 0.12,
            "method_family_match": 0.08,
            "clean_transition_quality": 0.08,
            "successful_repair_frequency": 0.02,
        },
        "dynamic L3 weights match the frozen contract",
        checks,
    )

    report = {
        "schema": "mlevolve_l3_debug_router_smoke_v1",
        "status": "passed",
        "graph_path": str(graph),
        "graph_sha256": sha256_file(graph),
        "index_path": str(index),
        "index_sha256": sha256_file(index),
        "recipe_bundle_path": str(RECIPE_BUNDLE),
        "recipe_bundle_file_sha256": sha256_file(RECIPE_BUNDLE),
        "recipe_bundle_sha256": recipe["bundle_sha256"],
        "l3_count": len(l3_nodes),
        "l3_evidence_tiering_enabled": False,
        "dynamic_confidence_weights": L3_DYNAMIC_CONFIDENCE_WEIGHTS,
        "checks": checks,
        "cases": cases,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
