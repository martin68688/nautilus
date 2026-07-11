#!/usr/bin/env python3
"""No-GPU preflight for stage-aware RunForest retrieval."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.stage_aware_hybrid_memory import PACK_SCHEMA, RETRIEVAL_CONTROLS, StageAwareHybridMemoryLayer
from config import Config, _load_cfg


CONFIG = MLEVOLVE / "config" / "config_run_forest_stage_hybrid.yaml"
GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
COLDSTART = MLEVOLVE / "engine" / "coldstart" / "models_guidance_classified.json"
COLDSTART_SHA256 = "5ecbdc00023227e75840f59104c9f5be58ae9efd403beb3d6c5cff894d49b0ff"
REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "stage_hybrid_preflight_report.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_preflight(*, evaluate_offline: bool = True) -> dict:
    checks = {}
    cfg = _load_cfg(CONFIG, use_cli_args=False)
    cfg.exp_name = "stage-hybrid-preflight"
    cfg.exp_id = "stage-hybrid-preflight"
    cfg.data_dir = "./data"
    cfg.goal = "no-gpu preflight"
    cfg.desc_file = None
    merged = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    checks["structured_config"] = {
        "ok": merged.external_skill_memory.mode == "run_forest_stage_hybrid",
        "mode": merged.external_skill_memory.mode,
        "retrieval_control": merged.external_skill_memory.retrieval_control,
        "roles": list(merged.agent.draft_role_policy.roles),
    }

    coldstart_hash = hashlib.sha256(COLDSTART.read_bytes()).hexdigest()
    checks["coldstart_template"] = {
        "ok": coldstart_hash == COLDSTART_SHA256,
        "sha256": coldstart_hash,
    }
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    meta = graph.get("meta") or {}
    checks["clean_graph_provenance"] = {
        "ok": all(
            meta.get(key) is True
            for key in ("source_membership_verified", "leak_audited", "positive_admission_enforced", "paper_grade")
        ) and meta.get("provenance_status") == "source_allowlisted_and_code_audited",
        "provenance_status": meta.get("provenance_status"),
        "source_runs": len(meta.get("source_runs") or []),
    }

    route_checks = []
    for control in sorted(RETRIEVAL_CONTROLS - {"layered_strategy"}):
        layer = StageAwareHybridMemoryLayer(
            graph_path=str(GRAPH),
            index_path=str(INDEX),
            source_name="run_forest_stage_hybrid_memory",
            mode="run_forest_stage_hybrid",
            scoring_mode="poincare",
            retrieval_control=control,
            enable_agentic=False,
            max_chars=6500,
        )
        for stage in ("draft", "improve", "debug", "evolution", "fusion"):
            text, refs = layer.retrieve_for_node(
                stage=stage,
                task_id="spooky-author-identification",
                task_desc="identify the author of a text passage",
                query_parts=["validate a robust model change without leakage"],
            )
            pack = layer.current_navigation_pack()
            positive_ids = [item["id"] for item in pack.get("fused_execution_candidates", [])]
            positive_ids += [item["id"] for item in pack.get("selected_sop_gateways", [])]
            blocked_positive_count = 0
            for node_id in positive_ids:
                node = layer.nodes.get(node_id, {})
                if node.get("type") == "Transition":
                    eligible = layer._positive_transition(node_id)[0]
                elif node.get("type") == "RunNode":
                    eligible = layer._positive_memory_eligible(node)
                elif node.get("type") == "SOP":
                    eligible = any(
                        layer._positive_transition(transition_id)[0]
                        for transition_id in layer._transitions_by_sop.get(node_id, [])
                    )
                else:
                    eligible = False
                blocked_positive_count += int(not eligible)
            route_checks.append(
                {
                    "control": control,
                    "stage": stage,
                    "ok": bool(text) and pack.get("schema") == PACK_SCHEMA and pack.get("stage_route", {}).get("control") == control,
                    "ref_count": len(refs),
                    "blocked_positive_count": blocked_positive_count,
                }
            )
    checks["runtime_routes"] = {
        "ok": all(item["ok"] and item["blocked_positive_count"] == 0 for item in route_checks),
        "cases": route_checks,
    }
    layered_smoke = _load_module(
        REPO / "paper-skills" / "hyper_memory" / "smoke_layered_three_role.py",
        "layered_three_role_preflight",
    ).run_smoke()
    checks["layered_three_role"] = {
        "ok": layered_smoke.get("status") == "passed",
        **layered_smoke,
    }

    builder = _load_module(
        REPO / "paper-skills" / "eval_skill_memory" / "build_stage_hybrid_benchmark.py",
        "stage_hybrid_preflight_builder",
    )
    queries, gold = builder.build_records(graph, max_queries=240)
    validation = builder.validate_records(queries, gold)
    checks["held_out_benchmark"] = {"ok": validation["valid"], **validation}

    if evaluate_offline:
        evaluator = _load_module(
            REPO / "paper-skills" / "hyper_memory" / "evaluate_stage_hybrid_retrieval.py",
            "stage_hybrid_preflight_evaluator",
        )
        evaluation = evaluator.evaluate(queries, gold, graph=GRAPH, index=INDEX, split="test")
        checks["offline_claim_gates"] = {
            "ok": True,
            "claim_gates_computed": True,
            "claim_gates": evaluation["claim_gates"],
        }
    else:
        checks["offline_claim_gates"] = {"ok": True, "claim_gates_computed": False}

    required = (
        "structured_config", "coldstart_template", "clean_graph_provenance", "runtime_routes",
        "layered_three_role", "held_out_benchmark", "offline_claim_gates",
    )
    return {
        "schema": "stage_hybrid_preflight_v1",
        "ok": all(checks[name]["ok"] for name in required),
        "checks": checks,
        "online_training_started": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, default=REPORT)
    parser.add_argument("--skip-offline-evaluation", action="store_true")
    args = parser.parse_args()
    report = run_preflight(evaluate_offline=not args.skip_offline_evaluation)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "checks": {key: value["ok"] for key, value in report["checks"].items()}}, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
