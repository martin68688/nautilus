#!/usr/bin/env python3
"""Run the frozen offline decision benchmark and emit per-condition receipts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from core import (
    ARTIFACTS,
    CONDITIONS,
    EPISODES,
    INDEX,
    MANIFESTS,
    REPORTS,
    TAXONOMY,
    average_precision,
    deterministic_order,
    graded_ndcg,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.stage_aware_hybrid_memory import (  # noqa: E402
    StageAwareHybridMemoryLayer,
    weighted_rrf,
)


def _load_legacy_evaluator():
    path = REPO / "paper-skills" / "eval_skill_memory" / "evaluate_decision_point_benchmark.py"
    spec = importlib.util.spec_from_file_location("composite_legacy_retrievers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


legacy = _load_legacy_evaluator()


class CompositeRetriever:
    def __init__(self, graph_path: Path, index_path: Path):
        self.graph_path = graph_path
        self.index_path = index_path
        self.base = legacy.RetrieverSuite(graph_path, index_path)
        self.layers = {
            "stage": self.base.layer,
            "tree": StageAwareHybridMemoryLayer(
                graph_path=str(graph_path), index_path=str(index_path), source_name="composite_tree",
                mode="run_forest_stage_hybrid", scoring_mode="poincare", retrieval_control="tree_only",
                enable_agentic=False, top_k=20, max_chars=0,
            ),
            "sop": StageAwareHybridMemoryLayer(
                graph_path=str(graph_path), index_path=str(index_path), source_name="composite_sop",
                mode="run_forest_stage_hybrid", scoring_mode="poincare", retrieval_control="sop_only",
                enable_agentic=False, top_k=20, max_chars=0,
            ),
            "flat_twin": StageAwareHybridMemoryLayer(
                graph_path=str(graph_path), index_path=str(index_path), source_name="composite_flat_twin",
                mode="run_forest_stage_hybrid", scoring_mode="flat_twin", retrieval_control="stage_hybrid",
                enable_agentic=False, top_k=20, max_chars=0,
            ),
        }
        self.last_navigation: dict[str, Any] = {}

    def clean(self, candidate_id: str) -> bool:
        return self.base.safe(candidate_id)

    def _stage(self, query: dict[str, Any], *, layer_key: str = "stage", task: bool = True, top_k: int = 10) -> list[str]:
        layer = self.layers[layer_key]
        if query["stage"] == "model_design" and layer_key == "stage" and task:
            if not query.get("primary_method_family"):
                self.last_navigation = {
                    "schema": "layered_model_design_tactics_v1",
                    "stage_route": {"stage": "model_design", "status": "insufficient_strategy_coverage"},
                    "selected_tactics": [],
                }
                return []
            selected_id = query.get("selected_l1_sop_id")
            selected = self.base.nodes.get(str(selected_id), {})
            _text, _refs, pack = layer.retrieve_model_design_tactics(
                task_id=query["source_task"],
                task_desc=query["query_text"],
                strategy_context={
                    "selected_strategy": {
                        "sop_id": selected_id,
                        "method_family": query.get("primary_method_family"),
                        "title": selected.get("title"),
                        "action": selected.get("action"),
                        "best_tree_evidence": {},
                    },
                    "task_profile": {"task_family": query["retrieval_family"]},
                },
            )
            self.last_navigation = pack
            allowed = set(query["candidate_ids"])
            return [row["sop_id"] for row in pack["selected_tactics"] if row["sop_id"] in allowed][:top_k]
        effective_stage = "improve" if query["stage"] == "model_design" else query["stage"]
        pack = layer.rank_sop_hybrid(
            stage=effective_stage,
            task_id=query["task_id"] if task else "",
            task_desc=query["retrieval_family"] if task else "",
            query_text=query["query_text"],
            limit=top_k,
            allowed_sop_ids=set(query["candidate_ids"]),
        )
        self.last_navigation = {
            "schema": pack["schema"],
            "stage_route": pack["stage_route"],
            "direct_clean_sop_ids": pack["direct_clean_sop_ids"][:20],
            "tree_projected_sop_ids": pack["tree_projected_sop_ids"][:20],
            "safety_gate": pack["safety_gate"],
        }
        return [row["id"] for row in pack["fused_sop_candidates"]]

    def _unsafe_stage(self, query: dict[str, Any], top_k: int, *, clean_universe: bool = False) -> list[str]:
        effective_stage = "improve" if query["stage"] == "model_design" else query["stage"]
        allowed = set(query["candidate_ids"])
        if clean_universe:
            allowed = {candidate_id for candidate_id in allowed if self.clean(candidate_id)}
        rows = self.base.layer._rank_sops(
            query["query_text"],
            effective_stage,
            len(self.base.layer._sops),
            task_id=query["task_id"],
            task_desc=query["retrieval_family"],
            allowed_sop_ids=allowed,
        )
        compatible = [row for row in rows if row["stage_compatible"] and row["task_compatible"]]
        direct = [row["id"] for row in compatible]
        tree_rows = self.base.layer._rank_tree_rows(
            stage=effective_stage,
            query_text=query["query_text"],
            task_id=query["task_id"],
            task_desc=query["retrieval_family"],
            limit=max(80, top_k * 8),
        )
        projected = self.base.layer._tree_sop_projection([row["id"] for row in tree_rows], allowed)
        weights = self.base.layer.rrf_weights[effective_stage]
        fused = weighted_rrf(direct, projected, sop_weight=weights["sop"], tree_weight=weights["tree"])
        self.last_navigation = {
            "schema": "stage_hybrid_unsafe_ranking_v1",
            "stage_route": {"stage": effective_stage, "rrf": weights, "safety_gate": False},
            "direct_sop_ids": direct[:20],
            "tree_projected_sop_ids": projected[:20],
            "candidate_universe": "preclean" if clean_universe else "mixed",
        }
        return [row["id"] for row in fused[:top_k]]

    def _no_stage_gate(self, query: dict[str, Any], top_k: int) -> list[str]:
        effective_stage = "improve" if query["stage"] == "model_design" else query["stage"]
        rows = self.base.layer._rank_sops(
            query["query_text"],
            effective_stage,
            len(self.base.layer._sops),
            task_id=query["task_id"],
            task_desc=query["retrieval_family"],
            allowed_sop_ids=set(query["candidate_ids"]),
        )
        clean_task = [row for row in rows if row["clean_supporting_transition_ids"] and row["task_compatible"]]
        direct = [row["id"] for row in clean_task]
        tree_rows = self.base.layer._rank_tree_rows(
            stage=effective_stage, query_text=query["query_text"], task_id=query["task_id"],
            task_desc=query["retrieval_family"], limit=max(80, top_k * 8),
        )
        projected = self.base.layer._tree_sop_projection(
            [row["id"] for row in tree_rows], set(query["candidate_ids"])
        )
        compatible = {row["id"] for row in clean_task}
        projected = [value for value in projected if value in compatible]
        weights = self.base.layer.rrf_weights[effective_stage]
        fused = weighted_rrf(direct, projected, sop_weight=weights["sop"], tree_weight=weights["tree"])
        self.last_navigation = {
            "schema": "stage_hybrid_no_stage_gate_v1",
            "stage_route": {"stage": effective_stage, "rrf": weights, "stage_gate": False},
            "direct_sop_ids": direct[:20],
            "tree_projected_sop_ids": projected[:20],
        }
        return [row["id"] for row in fused[:top_k]]

    def rank(self, condition: str, query: dict[str, Any], relevance: dict[str, int], top_k: int) -> tuple[list[str], dict[str, Any]]:
        policy = str(CONDITIONS[condition]["memory"])
        meta: dict[str, Any] = {"policy": policy, "implementation": "composite_v1"}
        if policy in {"none", "greedy_best_valid", "compatible_clean_ensemble"}:
            return [], {**meta, "not_applicable_to_offline_retrieval": True}
        if policy == "random_clean":
            clean = [candidate_id for candidate_id in query["candidate_ids"] if self.clean(candidate_id)]
            return deterministic_order(clean, query["episode_id"])[:top_k], meta
        if policy == "flat_clean":
            ranking = self.base.rank("minilm_dense_safety_filtered", query, relevance, top_k=top_k)
            return ranking, {**meta, "available": self.base.sentence_model is not None, "error": self.base.sentence_model_error}
        if policy == "legacy_gateway":
            legacy_query = {**query, "stage": "improve" if query["stage"] == "model_design" else query["stage"]}
            return self.base.rank("legacy_stage_gateway", legacy_query, relevance, top_k=top_k), meta
        if policy == "tree_only":
            ranking = self._stage(query, layer_key="tree", top_k=top_k)
            return ranking, {**meta, "scoring_mode": "poincare", "navigation_trace": self.last_navigation}
        if policy == "sop_only":
            ranking = self._stage(query, layer_key="sop", top_k=top_k)
            return ranking, {**meta, "scoring_mode": "poincare", "navigation_trace": self.last_navigation}
        if policy == "stage_hybrid_no_stage":
            ranking = self._no_stage_gate(query, top_k)
            return ranking, {**meta, "scoring_mode": "poincare", "navigation_trace": self.last_navigation}
        if policy == "stage_hybrid_no_task":
            ranking = self._stage(query, task=False, top_k=top_k)
            return ranking, {**meta, "scoring_mode": "poincare", "navigation_trace": self.last_navigation}
        if policy == "stage_hybrid_flat_twin":
            ranking = self._stage(query, layer_key="flat_twin", top_k=top_k)
            return ranking, {**meta, "scoring_mode": "flat_twin", "navigation_trace": self.last_navigation}
        if policy == "stage_hybrid_unsafe_offline":
            ranking = self._unsafe_stage(query, top_k)
            return ranking, {**meta, "execution_forbidden": True, "isolated_offline": True, "scoring_mode": "poincare", "navigation_trace": self.last_navigation}
        if policy == "stage_hybrid_clean_universe":
            ranking = self._unsafe_stage(query, top_k, clean_universe=True)
            return ranking, {**meta, "safety_gate_disabled": True, "candidate_universe": "preclean", "scoring_mode": "poincare", "navigation_trace": self.last_navigation}
        if policy == "stage_hybrid_v2":
            ranking = self._stage(query, top_k=top_k)
            return ranking, {**meta, "scoring_mode": "poincare", "navigation_trace": self.last_navigation}
        if policy == "safe_oracle":
            clean = [candidate_id for candidate_id in query["candidate_ids"] if self.clean(candidate_id)]
            return sorted(clean, key=lambda value: (-relevance.get(value, 0), value))[:top_k], {**meta, "oracle": True}
        raise ValueError(f"unsupported condition {condition}: {policy}")


def run(
    *,
    split: str,
    conditions: list[str],
    graph_path: Path,
    index_path: Path,
    top_k: int = 10,
) -> dict[str, Any]:
    memory_manifest = read_json(MANIFESTS / "memory_snapshot_manifest_v1.json")
    if sha256_file(graph_path) != memory_manifest["snapshot_sha256"]:
        raise ValueError("frozen memory snapshot hash mismatch")
    if sha256_file(index_path) != memory_manifest["source_index_sha256"]:
        raise ValueError("frozen memory index hash mismatch")
    queries = read_jsonl(EPISODES / f"decision_{split}_v1.jsonl")
    gold_rows = read_jsonl(EPISODES / f"decision_{split}_silver_gold_v1.jsonl")
    gold_by_id = {row["episode_id"]: row for row in gold_rows}
    retriever = CompositeRetriever(graph_path, index_path)
    try:
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        code_commit = "unavailable"
    receipts: list[dict[str, Any]] = []
    for query in queries:
        gold = gold_by_id[query["episode_id"]]
        relevance = {row["candidate_id"]: int(row["relevance"]) for row in gold["labels"]}
        for condition in conditions:
            started = time.perf_counter()
            ranking, meta = retriever.rank(condition, query, relevance, top_k)
            latency = time.perf_counter() - started
            unsafe = [candidate_id for candidate_id in ranking if not retriever.clean(candidate_id)]
            expected_gap = query["expected_status"] == "insufficient_strategy_coverage"
            score_applicable = not expected_gap and not meta.get("not_applicable_to_offline_retrieval", False)
            receipt = {
                "schema": "runforest_composite_offline_receipt_v1",
                "episode_id": query["episode_id"],
                "task_id": query["task_id"],
                "task_family": query["task_family"],
                "stage": query["stage"],
                "split": split,
                "condition": condition,
                "portfolio": CONDITIONS[condition]["portfolio"],
                "memory_policy": CONDITIONS[condition]["memory"],
                "ranking": ranking,
                "returned_count": len(ranking),
                "graded_ndcg_at_10": graded_ndcg(ranking, relevance, top_k) if relevance and score_applicable else None,
                "adoption_ap_at_10": average_precision(ranking, relevance, top_k) if relevance and score_applicable else None,
                "unsafe_count_at_10": len(unsafe),
                "unsafe_rate_at_10": len(unsafe) / max(1, len(ranking)),
                "expected_coverage_gap": expected_gap,
                "coverage_gap_correct": (len(ranking) < 3) if expected_gap else None,
                "score_applicable": score_applicable,
                "latency_sec": latency,
                "provenance": {
                    "graph_sha256": sha256_file(graph_path),
                    "index_sha256": sha256_file(index_path),
                    "query_candidate_count": query["candidate_count"],
                    "source_run_ids_exposed": query["source_run_ids_exposed"],
                    "code_commit": code_commit,
                    "taxonomy_sha256": sha256_file(TAXONOMY),
                    "query_sha256": sha256_file(EPISODES / f"decision_{split}_v1.jsonl"),
                    "gold_sha256": sha256_file(EPISODES / f"decision_{split}_silver_gold_v1.jsonl"),
                    **meta,
                },
            }
            receipts.append(receipt)
    receipt_path = REPORTS / f"offline_{split}_receipts_v1.jsonl"
    write_jsonl(receipt_path, receipts)
    report = summarize(receipts, conditions, split=split)
    report["receipt_path"] = str(receipt_path)
    report["receipt_sha256"] = sha256_file(receipt_path)
    write_json(REPORTS / f"offline_{split}_report_v1.json", report)
    return report


def summarize(receipts: list[dict[str, Any]], conditions: list[str], *, split: str) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for condition in conditions:
        rows = [row for row in receipts if row["condition"] == condition]
        relevance_rows = [row for row in rows if row["graded_ndcg_at_10"] is not None]
        gap_rows = [row for row in rows if row["expected_coverage_gap"]]
        methods[condition] = {
            "episode_count": len(rows),
            "scored_episode_count": len(relevance_rows),
            "graded_ndcg_at_10": float(np.mean([row["graded_ndcg_at_10"] for row in relevance_rows])) if relevance_rows else None,
            "adoption_ap_at_10": float(np.mean([row["adoption_ap_at_10"] for row in relevance_rows])) if relevance_rows else None,
            "unsafe_escape_count": sum(int(row["unsafe_count_at_10"]) for row in rows),
            "unsafe_rate_at_10": float(np.mean([row["unsafe_rate_at_10"] for row in rows])) if rows else None,
            "coverage_gap_accuracy": float(np.mean([row["coverage_gap_correct"] for row in gap_rows])) if gap_rows else None,
            "latency_sec": float(np.mean([row["latency_sec"] for row in rows])) if rows else None,
            "available": not all(row["provenance"].get("available") is False for row in rows) if rows else False,
            "by_stage": {
                stage: float(np.mean([row["graded_ndcg_at_10"] for row in relevance_rows if row["stage"] == stage]))
                for stage in sorted({row["stage"] for row in relevance_rows})
            },
            "by_task_family": {
                family: float(np.mean([row["graded_ndcg_at_10"] for row in relevance_rows if row["task_family"] == family]))
                for family in sorted({row["task_family"] for row in relevance_rows})
            },
        }
    report = {
        "schema": "runforest_composite_offline_report_v1",
        "split": split,
        "episode_count": len({row["episode_id"] for row in receipts}),
        "conditions": methods,
        "silver_labels": True,
        "blind_annotation_complete": False,
        "offline_relevance_claim_allowed": False,
        "downstream_claim_allowed": False,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--graph", type=Path, default=ARTIFACTS / "memory_snapshot_graph_v1.json")
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    conditions = [value for value in args.conditions.split(",") if value]
    unknown = sorted(set(conditions) - set(CONDITIONS))
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")
    print(json.dumps(run(split=args.split, conditions=conditions, graph_path=args.graph, index_path=args.index, top_k=args.top_k), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
