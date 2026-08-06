#!/usr/bin/env python3
"""Run live DeepSeek L3 root-cause matching before the Dynamic Debug smoke.

This retrieval-only harness uses the same frozen Recipe SOP and repair-evidence
overlays as MLEvolve. It proves exact-task preference, same-task-type fallback,
cross-type blocking, infrastructure abstention, Prompt injection, and the
absence of the maintained synonym matcher on the Dynamic L3 path.
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
    StageAwareHybridMemoryLayer,
)
from config import _load_cfg  # noqa: E402


DEFAULT_GRAPH = Path(
    "/workspace/nautilus/experiments/fourtask_graph_v2_replay_20260727/"
    "graph-inputs/run_forest_graph_fourtask_v2.json"
)
DEFAULT_INDEX = DEFAULT_GRAPH.with_name("run_forest_index_fourtask_v2.npz")
LOCAL_GRAPH = Path(
    "/Users/haoming/Downloads/nautilus/experiments/"
    "fourtask_graph_v2_replay_20260727/graph-inputs/"
    "run_forest_graph_fourtask_v2.json"
)
LOCAL_INDEX = LOCAL_GRAPH.with_name("run_forest_index_fourtask_v2.npz")
CONFIG = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "systems"
    / "dynamic_hybrid.yaml"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_path(explicit: Path | None, primary: Path, local: Path) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return primary if primary.exists() else local


def run_case(
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
        context={"runtime_smoke": "l3_agent_root_cause_match"},
    )
    pack = layer.current_navigation_pack()
    match = pack.get("l3_agent_match") or {}
    return {
        "name": name,
        "task_id": task_id,
        "query": query,
        "algorithm_version": pack.get("algorithm_version"),
        "selected_sop_id": match.get("selected_sop_id"),
        "selected_transition_id": match.get("selected_transition_id"),
        "selected_task_scope": match.get("selected_task_scope"),
        "decision": match.get("decision"),
        "final_confidence": match.get("final_confidence"),
        "manual_synonym_table_used": match.get("manual_synonym_table_used"),
        "literal_anchor_extractor": match.get("literal_anchor_extractor"),
        "agent_calls": match.get("agent_calls"),
        "trace": match.get("trace"),
        "trace_sha256": match.get("trace_sha256"),
        "prompt": prompt,
        "prompt_refs": refs,
        "selected_sop_gateways": [
            row.get("id") for row in pack.get("selected_sop_gateways") or []
        ],
        "tree_candidate_details": pack.get("tree_candidate_details") or [],
        "stage_route": pack.get("stage_route") or {},
        "gateway_selection": pack.get("gateway_selection") or {},
        "memory_abstention": pack.get("memory_abstention") or {},
    }


def require(condition: bool, message: str, checks: list[dict[str, Any]]) -> None:
    checks.append({"check": message, "passed": bool(condition)})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    graph = choose_path(args.graph, DEFAULT_GRAPH, LOCAL_GRAPH)
    index = choose_path(args.index, DEFAULT_INDEX, LOCAL_INDEX)
    if not graph.is_file() or not index.is_file():
        raise FileNotFoundError(f"Missing real RunForest graph/index: {graph} / {index}")

    cfg = _load_cfg(CONFIG, use_cli_args=False)
    cfg.exp_id = "aerial-cactus-identification"
    cfg.agent.search.num_gpus = 1
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
    if not layer.experiment_r_l3_agent_match_enabled:
        raise RuntimeError("Dynamic config did not enable L3 Agent matching")

    cases = {
        "exact_task": run_case(
            layer,
            name="exact_task",
            task_id="aerial-cactus-identification",
            task_desc="Vision binary image classification evaluated by ROC-AUC.",
            query=(
                "RuntimeError: after flattening the custom CNN features, mat1 "
                "has shape [64, 1024], while the classifier's first Linear "
                "layer has a [2048, 512] weight, so matrix multiplication fails."
            ),
        ),
        "nearby_different_root_cause": run_case(
            layer,
            name="nearby_different_root_cause",
            task_id="aerial-cactus-identification",
            task_desc="Vision binary image classification evaluated by ROC-AUC.",
            query=(
                "RuntimeError: ConvNeXt produces a spatial activation "
                "torch.Size([32, 768, 8, 8]), but LayerNorm([768]) in the "
                "classification head receives it before global pooling."
            ),
        ),
        "same_task_type_fallback": run_case(
            layer,
            name="same_task_type_fallback",
            task_id="aerial-cactus-identification",
            task_desc="Vision binary image classification evaluated by ROC-AUC.",
            query=(
                "RuntimeError: a U-Net decoder upsample has height and width "
                "different from its encoder skip tensor, so concatenation "
                "fails; align the decoder lattice to the skip connection."
            ),
        ),
        "cross_task_type_block": run_case(
            layer,
            name="cross_task_type_block",
            task_id="spooky-author-identification",
            task_desc="NLP multiclass text classification evaluated by log loss.",
            query=(
                "RuntimeError: a U-Net decoder upsample has height and width "
                "different from its encoder skip tensor and concatenation fails."
            ),
        ),
        "infrastructure_abstention": run_case(
            layer,
            name="infrastructure_abstention",
            task_id="leaf-classification",
            task_desc="Leaf multimodal classification evaluated by log loss.",
            query=(
                "Permission denied in a temporary node cache; the Pod remained "
                "Pending and the model API timed out before execution."
            ),
        ),
    }

    checks: list[dict[str, Any]] = []
    exact = cases["exact_task"]
    require(
        exact["selected_sop_id"]
        == "repair::aerial-cactus-identification::005",
        "exact-task shape failure selects Aerial L3-005",
        checks,
    )
    require(
        exact["selected_task_scope"] == "exact_task",
        "exact-task tier wins before fallback",
        checks,
    )
    require(
        exact["selected_sop_id"] in exact["prompt_refs"]
        and exact["selected_sop_id"] in exact["prompt"],
        "Agent-selected exact L3 is visible in the Debug Prompt",
        checks,
    )

    nearby = cases["nearby_different_root_cause"]
    require(
        nearby["decision"] == "abstain"
        and not nearby["selected_sop_id"]
        and not nearby["prompt_refs"],
        "nearby LayerNorm-before-pooling failure is not confused with L3-005",
        checks,
    )

    fallback = cases["same_task_type_fallback"]
    require(
        fallback["selected_sop_id"]
        == "repair::denoising-dirty-documents::002",
        "same-type fallback selects Denoising skip-alignment repair",
        checks,
    )
    require(
        fallback["selected_task_scope"] == "same_task_type",
        "same-type fallback occurs only after exact-task abstention",
        checks,
    )

    for name in ("cross_task_type_block", "infrastructure_abstention"):
        case = cases[name]
        require(
            case["decision"] == "abstain"
            and not case["selected_sop_id"]
            and not case["prompt_refs"],
            f"{name} injects no L3 repair",
            checks,
        )

    require(
        all(case["manual_synonym_table_used"] is False for case in cases.values()),
        "all Dynamic L3 decisions record manual_synonym_table_used=false",
        checks,
    )
    require(
        all(
            (case["literal_anchor_extractor"] or {}).get("extractor")
            == "literal_regex_no_synonym_expansion_v1"
            for case in cases.values()
        ),
        "all cases use literal error anchors without synonym expansion",
        checks,
    )
    require(
        all(
            case["algorithm_version"]
            == "stage_hybrid_l3_agent_root_cause_v1"
            for case in cases.values()
        ),
        "all cases traverse the Agent-controlled layered Debug path",
        checks,
    )

    passed = all(bool(row["passed"]) for row in checks)
    report = {
        "schema": "mlevolve_l3_agent_match_smoke_v1",
        "status": "passed" if passed else "failed",
        "graph_path": str(graph),
        "graph_sha256": sha256_file(graph),
        "index_path": str(index),
        "index_sha256": sha256_file(index),
        "config_path": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "checks": checks,
        "cases": cases,
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(args.output),
                    "checks_passed": sum(
                        bool(row["passed"]) for row in checks
                    ),
                    "checks_total": len(checks),
                    "case_summary": {
                        name: {
                            "decision": case["decision"],
                            "selected_sop_id": case["selected_sop_id"],
                            "selected_task_scope": case["selected_task_scope"],
                            "final_confidence": case["final_confidence"],
                            "agent_calls": case["agent_calls"],
                        }
                        for name, case in cases.items()
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        print(encoded)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
