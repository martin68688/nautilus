#!/usr/bin/env python3
"""Evaluate actual Novel Draft strategy routes under fixed three-role controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer
from config import _load_cfg


GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
BENCHMARK = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "three_role_draft_benchmark.jsonl"
CONFIG = REPO / "mlevolve" / "config" / "config_run_forest_stage_hybrid.yaml"
REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "layered_novel_strategy_evaluation.json"
MARKDOWN = REPO / "coordination" / "layered_novel_strategy_routes.md"
CONDITIONS = ("tree_only", "stage_hybrid", "layered_strategy")
EXCLUDED_FAMILIES = ("modernbert_finetune", "deberta_xgb_lr_ensemble")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sops_for_execution(layer: StageAwareHybridMemoryLayer, node_id: str) -> list[str]:
    node = layer.nodes.get(node_id, {})
    transitions = []
    if node.get("type") == "Transition":
        transitions.append(node_id)
    elif node.get("type") == "RunNode":
        transitions.extend(layer._transitions_by_child.get(node_id, []))
        transitions.extend(layer._transitions_by_parent.get(node_id, []))
    output = []
    for transition_id in transitions:
        transition = layer.nodes.get(transition_id, {})
        for sop_id in transition.get("attached_sop_ids") or []:
            if sop_id in layer.nodes and sop_id not in output:
                output.append(sop_id)
    return output


def _tree_scheme_ids(layer: StageAwareHybridMemoryLayer, pack: dict[str, Any]) -> list[tuple[str, str | None]]:
    output = []
    for node_id in pack.get("tree_candidates") or []:
        for sop_id in _sops_for_execution(layer, node_id):
            if sop_id not in {item[0] for item in output}:
                output.append((sop_id, node_id))
            if len(output) >= 3:
                return output
    return output


def _stage_scheme_ids(layer: StageAwareHybridMemoryLayer, pack: dict[str, Any]) -> list[tuple[str, str | None]]:
    output = []
    for gateway in pack.get("selected_sop_gateways") or []:
        sop_id = str(gateway.get("id") or "")
        transitions = gateway.get("clean_supporting_transition_ids") or []
        evidence_node = None
        if transitions:
            evidence_node = layer.nodes.get(transitions[0], {}).get("child_node_id")
        if sop_id and sop_id not in {item[0] for item in output}:
            output.append((sop_id, str(evidence_node) if evidence_node else None))
        if len(output) >= 3:
            break
    return output


def _layered_scheme_ids(pack: dict[str, Any]) -> list[tuple[str, str | None]]:
    return [
        (str(route["sop_id"]), str(route["best_tree_evidence"]["node_id"]))
        for route in (pack.get("strategy_routes") or [])[:3]
    ]


def _scheme_cards(
    layer: StageAwareHybridMemoryLayer,
    condition: str,
    pack: dict[str, Any],
) -> list[dict[str, Any]]:
    if condition == "tree_only":
        pairs = _tree_scheme_ids(layer, pack)
    elif condition == "stage_hybrid":
        pairs = _stage_scheme_ids(layer, pack)
    else:
        pairs = _layered_scheme_ids(pack)
    cards = []
    for sop_id, evidence_node_id in pairs:
        sop = layer.nodes.get(sop_id, {})
        evidence_node = layer.nodes.get(str(evidence_node_id or ""), {})
        audit = evidence_node.get("leakage_audit") if isinstance(evidence_node.get("leakage_audit"), dict) else {}
        cards.append(
            {
                "sop_id": sop_id,
                "title": sop.get("title"),
                "action": sop.get("action"),
                "abstraction_level": sop.get("abstraction_level"),
                "sop_kind": sop.get("sop_kind"),
                "method_family": sop.get("method_family"),
                "evidence_node_id": evidence_node_id,
                "evidence_run_id": evidence_node.get("run_id"),
                "evidence_metric": evidence_node.get("metric"),
                "clean_rank_eligible_evidence": bool(
                    evidence_node
                    and audit.get("status") == "clean"
                    and audit.get("rank_eligible") is True
                    and evidence_node.get("is_buggy") is False
                    and evidence_node.get("is_valid") is True
                    and isinstance(evidence_node.get("metric"), (int, float))
                ),
            }
        )
    return cards


def _quality(cards: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = 3.0
    strategy_count = sum(
        card.get("abstraction_level") == "L1_strategy" and card.get("sop_kind") == "model_strategy"
        for card in cards
    )
    distinct = len({str(card.get("method_family")) for card in cards if card.get("method_family")})
    clean = sum(bool(card.get("clean_rank_eligible_evidence")) for card in cards)
    violations = [
        card["method_family"] for card in cards if card.get("method_family") in EXCLUDED_FAMILIES
    ]
    return {
        "route_count": len(cards),
        "strategy_precision_at_3": strategy_count / denominator,
        "distinct_method_families_at_3": distinct,
        "detail_intrusion_at_3": (len(cards) - strategy_count) / denominator,
        "clean_expansion_precision_at_3": clean / denominator,
        "excluded_family_violations": violations,
        "passes_route_gate": bool(
            len(cards) == 3
            and strategy_count == 3
            and distinct == 3
            and clean == 3
            and not violations
        ),
    }


def evaluate(queries: list[dict[str, Any]], split: str = "test") -> dict[str, Any]:
    selected_queries = [query for query in queries if query.get("split") == split]
    held_out_runs = sorted(str(query["run_id"]) for query in selected_queries)
    rows: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for condition in CONDITIONS:
        for query in selected_queries:
            cfg = _load_cfg(CONFIG, use_cli_args=False)
            cfg.exp_id = query["task"]
            cfg.agent.search.num_gpus = 7
            layer = StageAwareHybridMemoryLayer(
                graph_path=str(GRAPH),
                index_path=str(INDEX),
                source_name="run_forest_stage_hybrid_memory",
                mode="run_forest_stage_hybrid",
                scoring_mode="poincare",
                retrieval_control=condition,
                excluded_run_ids=held_out_runs,
                enable_agentic=False,
                top_k=10,
                max_chars=0,
                cfg=cfg,
            )
            error = None
            try:
                layer.retrieve_for_node(
                    stage="draft",
                    task_id=query["task"],
                    task_desc=query["query_text"],
                    query_parts=[],
                    draft_role="novel_exploration" if condition == "layered_strategy" else None,
                    context={"excluded_method_families": list(EXCLUDED_FAMILIES)},
                )
                pack = layer.current_navigation_pack()
                cards = _scheme_cards(layer, condition, pack)
            except Exception as exc:
                cards = []
                error = f"{type(exc).__name__}: {exc}"
            rows[condition].append(
                {
                    "query_id": query["query_id"],
                    "run_id": query["run_id"],
                    "condition": condition,
                    "schemes": cards,
                    "quality": _quality(cards),
                    "error": error,
                }
            )
    aggregate = {}
    for condition, condition_rows in rows.items():
        aggregate[condition] = {
            "query_count": len(condition_rows),
            "strategy_precision_at_3": sum(row["quality"]["strategy_precision_at_3"] for row in condition_rows) / max(1, len(condition_rows)),
            "mean_distinct_method_families_at_3": sum(row["quality"]["distinct_method_families_at_3"] for row in condition_rows) / max(1, len(condition_rows)),
            "detail_intrusion_at_3": sum(row["quality"]["detail_intrusion_at_3"] for row in condition_rows) / max(1, len(condition_rows)),
            "clean_expansion_precision_at_3": sum(row["quality"]["clean_expansion_precision_at_3"] for row in condition_rows) / max(1, len(condition_rows)),
            "excluded_family_violation_count": sum(len(row["quality"]["excluded_family_violations"]) for row in condition_rows),
            "route_gate_pass_rate": sum(row["quality"]["passes_route_gate"] for row in condition_rows) / max(1, len(condition_rows)),
            "error_count": sum(row["error"] is not None for row in condition_rows),
        }
    return {
        "schema": "layered_novel_strategy_evaluation_v1",
        "split": split,
        "query_count": len(selected_queries),
        "held_out_run_ids": held_out_runs,
        "role_protocol": {
            "fixed_roles": ["coldstart_baseline", "memory_reproduction"],
            "changed_role": "novel_exploration",
            "conditions": list(CONDITIONS),
            "excluded_method_families": list(EXCLUDED_FAMILIES),
        },
        "controls": aggregate,
        "per_query": rows,
        "claim_allowed": bool(
            len(selected_queries) >= 20
            and aggregate["layered_strategy"]["route_gate_pass_rate"] == 1.0
            and aggregate["layered_strategy"]["error_count"] == 0
        ),
        "claim_note": "This benchmark exposes actual method routes. Downstream superiority still requires a concurrent online control.",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Novel Draft Actual Strategy Routes",
        "",
        f"Split: `{report['split']}`; queries: `{report['query_count']}`.",
        "",
        "Baseline and replay roles are fixed. Only the Novel retrieval condition changes.",
        "",
        "| Novel retrieval | Strategy precision@3 | Distinct families@3 | Detail intrusion@3 | Clean expansion@3 | Excluded violations | Gate pass rate | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = report["controls"][condition]
        lines.append(
            f"| {condition} | {row['strategy_precision_at_3']:.4f} | "
            f"{row['mean_distinct_method_families_at_3']:.2f} | {row['detail_intrusion_at_3']:.4f} | "
            f"{row['clean_expansion_precision_at_3']:.4f} | {row['excluded_family_violation_count']} | "
            f"{row['route_gate_pass_rate']:.4f} | {row['error_count']} |"
        )
    for query_index in range(report["query_count"]):
        query_id = report["per_query"]["layered_strategy"][query_index]["query_id"]
        lines += ["", f"## {query_id}", ""]
        for condition in CONDITIONS:
            result = report["per_query"][condition][query_index]
            lines.append(f"### {condition}")
            if result["error"]:
                lines.append(f"- ERROR: {result['error']}")
                continue
            for index, scheme in enumerate(result["schemes"], 1):
                lines.append(
                    f"{index}. `{scheme['method_family']}` via `{scheme['sop_id']}`: {scheme['title']} "
                    f"(level={scheme['abstraction_level']}, clean={scheme['clean_rank_eligible_evidence']}, "
                    f"metric={scheme['evidence_metric']})"
                )
    lines += ["", f"Claim allowed: `{report['claim_allowed']}`", "", report["claim_note"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--report-out", type=Path, default=REPORT)
    parser.add_argument("--markdown-out", type=Path, default=MARKDOWN)
    args = parser.parse_args()
    report = evaluate(_read_jsonl(args.benchmark), split=args.split)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"query_count": report["query_count"], "controls": report["controls"], "claim_allowed": report["claim_allowed"]}, indent=2))


if __name__ == "__main__":
    main()
