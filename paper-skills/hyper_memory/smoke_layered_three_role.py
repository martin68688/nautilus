#!/usr/bin/env python3
"""No-GPU preflight for fixed three-role layered Novel retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.external_skill_memory import fetch_external_skill_memory
from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer
from config import _load_cfg


CONFIG = REPO / "mlevolve" / "config" / "config_run_forest_stage_hybrid.yaml"
GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
DEFAULT_OUTPUT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "layered_three_role_preflight.json"


def run_smoke() -> dict:
    cfg = _load_cfg(CONFIG, use_cli_args=False)
    cfg.exp_id = "spooky-author-identification"
    cfg.agent.search.num_gpus = 7
    roles = list(cfg.agent.draft_role_policy.roles)
    assert roles == ["coldstart_baseline", "memory_reproduction", "novel_exploration"]
    assert cfg.agent.initial_drafts == cfg.agent.search.num_drafts == 3
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        retrieval_control="layered_strategy",
        enable_agentic=False,
        top_k=10,
        max_chars=0,
        cfg=cfg,
    )
    fake_agent = SimpleNamespace(
        external_skill_memory=layer,
        cfg=cfg,
        task_desc="Small-data text classification evaluated by multiclass log loss.",
    )
    bypass = {}
    for role in roles[:2]:
        text, refs, _source = fetch_external_skill_memory(fake_agent, "draft", draft_role=role)
        bypass[role] = {"text_empty": text == "", "refs_empty": refs == []}
        assert text == "" and refs == []
    layer.retrieve_for_node(
        stage="draft",
        task_id=cfg.exp_id,
        task_desc=fake_agent.task_desc,
        draft_role="novel_exploration",
        context={
            "baseline_model": "ModernBERT",
            "coldstart": "answerdotai/ModernBERT-large",
            "data_preview": "Train shape: (17621, 3)",
            "excluded_method_families": ["modernbert_finetune", "deberta_xgb_lr_ensemble"],
        },
    )
    strategy = layer.current_navigation_pack()
    routes = strategy["strategy_routes"]
    assert len(routes) == 3
    assert len({route["method_family"] for route in routes}) == 3
    _text, l2_refs, l2_pack = layer.retrieve_model_design_tactics(
        task_id=cfg.exp_id,
        task_desc=fake_agent.task_desc,
        strategy_context=strategy,
    )
    assert all(
        layer._family_compatible(l2_pack["method_family"], item["method_family"])
        for item in l2_pack["selected_tactics"]
    )
    abstraction_counts = {}
    for sop_id in layer._sops:
        level = str(layer.nodes[sop_id].get("abstraction_level"))
        abstraction_counts[level] = abstraction_counts.get(level, 0) + 1
    return {
        "schema": "layered_three_role_preflight_v1",
        "status": "passed",
        "roles": roles,
        "root_draft_limit": cfg.agent.search.num_drafts,
        "role_bypass": bypass,
        "strategy_routes": [
            {
                "sop_id": route["sop_id"],
                "method_family": route["method_family"],
                "evidence_node_id": route["best_tree_evidence"]["node_id"],
                "metric": route["best_tree_evidence"]["metric"],
            }
            for route in routes
        ],
        "selected_method_family": strategy["selected_strategy"]["method_family"],
        "l2_tactic_refs": l2_refs,
        "taxonomy_coverage": layer.graph["meta"]["sop_taxonomy_coverage"],
        "taxonomy_reviewed_l1_count": layer.graph["meta"]["sop_taxonomy_reviewed_l1_count"],
        "taxonomy_abstraction_counts": dict(sorted(abstraction_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_smoke()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
