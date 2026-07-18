#!/usr/bin/env python3
"""Build and evaluate the retrospective RunForest granularity benchmark v2.

The benchmark has two complementary tracks:

1. Stage granularity checks whether Draft/Model Design/Improve/Debug receive
   L1/L2/L2/L3 SOPs respectively.
2. Causal Debug transfer uses real clean parent-failure -> code-change ->
   successful-child transitions.  The query's source run is removed from the
   candidate memory so exact replay cannot earn credit.

This is a retrospective diagnostic benchmark.  It does not execute generated
code and therefore cannot establish downstream task improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer  # noqa: E402
from core import (  # noqa: E402
    EPISODES,
    GRAPH,
    INDEX,
    REPORTS,
    TAXONOMY,
    graded_ndcg,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from run_offline_decisions import CompositeRetriever  # noqa: E402


SCHEMA = "runforest_causal_granularity_benchmark_v2"
QUERY_PATH = EPISODES / "causal_debug_leave_one_run_out_v2.jsonl"
GOLD_PATH = EPISODES / "causal_debug_leave_one_run_out_gold_v2.jsonl"
RECEIPT_PATH = REPORTS / "causal_granularity_receipts_v2.jsonl"
REPORT_PATH = REPORTS / "causal_granularity_report_v2.json"
MANIFEST_PATH = HERE / "manifests" / "causal_granularity_manifest_v2.json"

EXPECTED_LEVEL = {
    "draft": "L1_strategy",
    "model_design": "L2_tactic",
    "improve": "L2_tactic",
    "debug": "L3_repair",
}

# Frozen benchmark-side silver labels use a separate code path from production
# FAILURE_SIGNATURES, but intentionally describe the same failure mechanisms.
# The first matching mechanism is the primary defect.
BENCHMARK_FAILURE_RULES: tuple[tuple[str, str], ...] = (
    ("resource_oom", r"out[- ]of[- ]memory|\boom\b|cuda memory|shared memory|bus error|resource exhausted"),
    (
        "tensor_alignment",
        r"shape mismatch|dimension mismatch|size mismatch|mat1 and mat2|broadcast|tensor shape|"
        r"prediction shape|inhomogeneous|scalar outputs",
    ),
    (
        "dependency_api",
        r"unexpected keyword|no attribute|not supported|modulenotfounderror|importerror|from_pretrained|"
        r"not a valid tree method|has no attribute",
    ),
    (
        "definition_order",
        r"nameerror|not defined|before it was defined|syntax error|syntaxerror|indentationerror|"
        r"parse error|jupyter/ipython",
    ),
    ("path_io", r"filenotfounderror|file not found|wrong path|permission denied|missing file"),
    (
        "numerical_instability",
        r"\bnan\b|infinite|overflow|underflow|division by zero|not finite|float16|fp16|"
        r"cannot be converted to type",
    ),
    (
        "fit_scope_leakage",
        r"data leakage|fit(?:ted)? (?:on|using).*(?:validation|holdout|full data)|"
        r"vectorizer.*(?:train.*validation|combined)|scaler.*(?:train.*validation|combined)|"
        r"imputation.*(?:validation|full)",
    ),
    (
        "gradient_lifecycle",
        r"backward through the graph a second time|gradient checkpoint|loss.*backward",
    ),
    (
        "evaluation_reuse",
        r"early stopping.*(?:report|weight|selection)|ensemble weight|validation reused|"
        r"selection bias|false oof|out.of.fold|stochastic weight averaging|\bswa\b",
    ),
)

DEBUG_METHODS = (
    "sop_only",
    "random_transition",
    "task_only_transition",
    "lexical_transition",
    "legacy_success_tree",
    "causal_tree_fixed_075",
    "causal_tree_dynamic",
    "oracle_transition",
)


def _layer(graph_path: Path, index_path: Path, *, control: str = "stage_hybrid") -> StageAwareHybridMemoryLayer:
    return StageAwareHybridMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        source_name="causal_granularity_v2",
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        retrieval_control=control,
        enable_agentic=False,
        top_k=20,
        max_chars=0,
    )


def primary_failure_mechanism(text: str) -> str:
    normalized = " ".join(str(text or "").lower().split())[:1200]
    for name, pattern in BENCHMARK_FAILURE_RULES:
        if re.search(pattern, normalized, flags=re.DOTALL):
            return name
    return "other"


def _run_id(node: dict[str, Any]) -> str:
    return str(node.get("run_short_id") or node.get("run_id") or "")


def _split_for_run(run_id: str) -> str:
    bucket = int(hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8], 16) % 4
    return "dev" if bucket == 0 else "test"


def _transition_records(layer: StageAwareHybridMemoryLayer) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for transition_id in layer._transitions:
        transition = layer.nodes[transition_id]
        eligible, _reason = layer._positive_transition(transition_id)
        if not eligible:
            continue
        if not (
            str(transition.get("outcome") or "") == "debug_fixed"
            and transition.get("parent_buggy") is True
            and transition.get("child_buggy") is False
            and "debug" in str(transition.get("stage_pair") or "")
        ):
            continue
        task_id = str(transition.get("task") or "")
        task_family = layer._task_family_for_query(task_id, task_id.replace("-", " "))
        attachments = layer._causal_attachment_rows(
            transition,
            stage="debug",
            task_family=task_family,
        )
        if not attachments:
            continue
        parent = layer.nodes.get(str(transition.get("parent_node_id") or ""), {})
        query_text = str(parent.get("analysis") or parent.get("terminal_excerpt") or "").strip()
        if not query_text:
            continue
        evidence = layer._debug_transition_evidence(transition)
        evidence_complete = all(
            bool(evidence.get(key)) for key in ("parent_failure", "code_change", "child_result")
        )
        if not evidence_complete:
            continue
        records.append(
            {
                "transition_id": transition_id,
                "run_id": _run_id(transition),
                "task_id": task_id,
                "task_family": task_family,
                "query_text": query_text,
                "failure_mechanism": primary_failure_mechanism(query_text),
                "sop_ids": [row["sop_id"] for row in attachments],
                "parent_node_id": str(transition.get("parent_node_id") or ""),
                "child_node_id": str(transition.get("child_node_id") or ""),
                "evidence_complete": True,
            }
        )
    return sorted(records, key=lambda row: row["transition_id"])


def build(graph_path: Path = GRAPH, index_path: Path = INDEX) -> dict[str, Any]:
    layer = _layer(graph_path, index_path)
    records = _transition_records(layer)
    by_id = {row["transition_id"]: row for row in records}
    queries: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        allowed = [row for row in records if row["run_id"] != source["run_id"]]
        gold = [
            row
            for row in allowed
            if source["failure_mechanism"] != "other"
            and row["failure_mechanism"] == source["failure_mechanism"]
            and layer._task_families_compatible(source["task_family"], row["task_family"])
        ]
        episode_id = f"causal-debug-v2::{index:03d}"
        expected_route = "causal_tree" if gold else "sop_only_fallback"
        split = _split_for_run(source["run_id"])
        queries.append(
            {
                "schema": SCHEMA,
                "episode_id": episode_id,
                "stage": "debug",
                "task_id": source["task_id"],
                "task_family": source["task_family"],
                "query_text": source["query_text"],
                "candidate_transition_ids": [row["transition_id"] for row in allowed],
                "expected_route": expected_route,
                "split": split,
                "source_run_ids_exposed_to_prompt": False,
                "exact_source_transition_in_candidates": False,
            }
        )
        gold_sops = sorted({sop_id for row in gold for sop_id in row["sop_ids"]})
        gold_rows.append(
            {
                "schema": f"{SCHEMA}_gold",
                "episode_id": episode_id,
                "source_transition_id": source["transition_id"],
                "source_run_id": source["run_id"],
                "failure_mechanism": source["failure_mechanism"],
                "expected_route": expected_route,
                "split": split,
                "gold_transition_ids": [row["transition_id"] for row in gold],
                "gold_sop_ids": gold_sops,
                "gold_is_cross_run": all(row["run_id"] != source["run_id"] for row in gold),
                "gold_is_clean_causal_transition": all(row["transition_id"] in by_id for row in gold),
            }
        )
    write_jsonl(QUERY_PATH, queries)
    write_jsonl(GOLD_PATH, gold_rows)
    manifest = {
        "schema": f"{SCHEMA}_manifest",
        "graph_path": str(graph_path),
        "graph_sha256": sha256_file(graph_path),
        "index_path": str(index_path),
        "index_sha256": sha256_file(index_path),
        "taxonomy_sha256": sha256_file(TAXONOMY),
        "episode_count": len(queries),
        "covered_episode_count": sum(row["expected_route"] == "causal_tree" for row in queries),
        "coverage_gap_episode_count": sum(row["expected_route"] == "sop_only_fallback" for row in queries),
        "failure_mechanisms": dict(sorted(Counter(row["failure_mechanism"] for row in gold_rows).items())),
        "source_run_count": len({row["source_run_id"] for row in gold_rows}),
        "by_split": dict(sorted(Counter(row["split"] for row in gold_rows).items())),
        "query_sha256": sha256_file(QUERY_PATH),
        "gold_sha256": sha256_file(GOLD_PATH),
        "retrospective": True,
        "blind_test": False,
        "failure_label_independence": "separate_code_path_but_semantically_aligned_silver_labels",
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest


def _reciprocal_rank(ranking: list[str], gold: set[str]) -> float:
    for rank, candidate_id in enumerate(ranking, 1):
        if candidate_id in gold:
            return 1.0 / rank
    return 0.0


def _recall(ranking: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    return len(set(ranking[:k]) & gold) / len(gold)


def _legacy_transition_ranking(
    layer: StageAwareHybridMemoryLayer,
    query: dict[str, Any],
    allowed: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    transition_by_child = {
        str(layer.nodes[transition_id].get("child_node_id") or ""): transition_id
        for transition_id in allowed
    }
    tree_rows = layer._rank_tree_rows(
        stage="debug",
        query_text=query["query_text"],
        task_id=query["task_id"],
        task_desc=query["task_family"],
        limit=max(100, limit * 10),
    )
    ranking: list[dict[str, Any]] = []
    for row in tree_rows:
        transition_id = transition_by_child.get(row["id"])
        if transition_id and transition_id not in {item["id"] for item in ranking}:
            transition = layer.nodes[transition_id]
            ranking.append(
                {
                    "id": transition_id,
                    "transition_evidence": layer._debug_transition_evidence(transition),
                    "causal_attachments": layer._causal_attachment_rows(
                        transition,
                        stage="debug",
                        task_family=query["task_family"],
                    ),
                    "confidence": 1.0,
                }
            )
        if len(ranking) >= limit:
            break
    return ranking


def _debug_method(
    method: str,
    *,
    layer: StageAwareHybridMemoryLayer,
    query: dict[str, Any],
    gold: dict[str, Any],
    limit: int,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    allowed = set(query["candidate_transition_ids"])
    if method == "sop_only":
        return [], "sop_only_fallback", {"tree_weight": 0.0, "reason": "tree_disabled"}
    if method in {"random_transition", "task_only_transition", "lexical_transition"}:
        candidates = []
        for transition_id in allowed:
            transition = layer.nodes[transition_id]
            if method == "random_transition":
                score = int(
                    hashlib.sha256(f"{query['episode_id']}::{transition_id}".encode("utf-8")).hexdigest()[:16],
                    16,
                )
            elif method == "task_only_transition":
                score = layer._debug_transition_task_fit(
                    transition,
                    task_id=query["task_id"],
                    task_family=query["task_family"],
                )
            else:
                score = layer._bounded_token_similarity(
                    query["query_text"],
                    layer._debug_parent_failure_text(transition),
                )
            candidates.append((score, transition_id))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        rows = []
        for _score, transition_id in candidates[:limit]:
            transition = layer.nodes[transition_id]
            rows.append(
                {
                    "id": transition_id,
                    "transition_evidence": layer._debug_transition_evidence(transition),
                    "causal_attachments": layer._causal_attachment_rows(
                        transition,
                        stage="debug",
                        task_family=query["task_family"],
                    ),
                    "confidence": None,
                }
            )
        return rows, ("causal_tree" if rows else "sop_only_fallback"), {
            "tree_weight": 1.0,
            "baseline": method,
        }
    if method == "legacy_success_tree":
        rows = _legacy_transition_ranking(layer, query, allowed, limit)
        return rows, ("causal_tree" if rows else "sop_only_fallback"), {"tree_weight": 0.75}
    if method in {"causal_tree_fixed_075", "causal_tree_dynamic"}:
        rows = layer._rank_debug_transition_rows(
            query_text=query["query_text"],
            task_id=query["task_id"],
            task_desc=query["task_family"],
            limit=limit,
            allowed_transition_ids=allowed,
        )
        if method == "causal_tree_fixed_075":
            return rows, ("causal_tree" if rows else "sop_only_fallback"), {"tree_weight": 0.75}
        weights, confidence, fallback_reason = layer._debug_dynamic_weights(rows)
        route = "sop_only_fallback" if fallback_reason else "causal_tree"
        if fallback_reason:
            rows = []
        return rows, route, {
            "tree_weight": weights["tree"],
            "confidence": confidence,
            "fallback_reason": fallback_reason,
        }
    if method == "oracle_transition":
        rows = [
            {
                "id": transition_id,
                "transition_evidence": layer._debug_transition_evidence(layer.nodes[transition_id]),
                "causal_attachments": layer._causal_attachment_rows(
                    layer.nodes[transition_id],
                    stage="debug",
                    task_family=query["task_family"],
                ),
                "confidence": 1.0,
            }
            for transition_id in gold["gold_transition_ids"][:limit]
        ]
        return rows, gold["expected_route"], {"oracle": True, "tree_weight": 1.0 if rows else 0.0}
    raise ValueError(f"unsupported debug method: {method}")


def evaluate_debug(graph_path: Path, index_path: Path, *, limit: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    layer = _layer(graph_path, index_path)
    queries = read_jsonl(QUERY_PATH)
    gold_by_id = {row["episode_id"]: row for row in read_jsonl(GOLD_PATH)}
    receipts: list[dict[str, Any]] = []
    for query in queries:
        gold = gold_by_id[query["episode_id"]]
        gold_ids = set(gold["gold_transition_ids"])
        relevance = {candidate_id: 1 for candidate_id in gold_ids}
        for method in DEBUG_METHODS:
            started = time.perf_counter()
            rows, route, meta = _debug_method(
                method,
                layer=layer,
                query=query,
                gold=gold,
                limit=limit,
            )
            latency = time.perf_counter() - started
            ranking = [row["id"] for row in rows]
            projected_sops = {
                attachment["sop_id"]
                for row in rows
                for attachment in row.get("causal_attachments", [])
            }
            unsafe = [candidate_id for candidate_id in ranking if not layer._positive_transition(candidate_id)[0]]
            source_run_escape = [
                candidate_id
                for candidate_id in ranking
                if _run_id(layer.nodes[candidate_id]) == gold["source_run_id"]
            ]
            packet_complete = bool(rows) and all(
                all(bool(row.get("transition_evidence", {}).get(key)) for key in ("parent_failure", "code_change", "child_result"))
                for row in rows[:1]
            )
            receipts.append(
                {
                    "schema": f"{SCHEMA}_debug_receipt",
                    "track": "causal_debug_transfer",
                    "episode_id": query["episode_id"],
                    "task_family": query["task_family"],
                    "split": query["split"],
                    "failure_mechanism": gold["failure_mechanism"],
                    "method": method,
                    "expected_route": gold["expected_route"],
                    "actual_route": route,
                    "route_correct": route == gold["expected_route"],
                    "ranking": ranking,
                    "gold_transition_count": len(gold_ids),
                    "transition_hit_at_1": bool(set(ranking[:1]) & gold_ids) if gold_ids else None,
                    "transition_recall_at_3": _recall(ranking, gold_ids, 3) if gold_ids else None,
                    "transition_recall_at_5": _recall(ranking, gold_ids, 5) if gold_ids else None,
                    "transition_mrr": _reciprocal_rank(ranking, gold_ids) if gold_ids else None,
                    "transition_ndcg_at_5": graded_ndcg(ranking, relevance, 5) if gold_ids else None,
                    "causal_sop_recall_at_5": (
                        len(projected_sops & set(gold["gold_sop_ids"])) / len(set(gold["gold_sop_ids"]))
                        if gold["gold_sop_ids"] else None
                    ),
                    "top_packet_complete": packet_complete,
                    "selective_transition_success_at_1": (
                        bool(set(ranking[:1]) & gold_ids)
                        if gold_ids
                        else route == "sop_only_fallback"
                    ),
                    "selective_decision_accuracy_at_1": (
                        bool(set(ranking[:1]) & gold_ids)
                        if gold_ids
                        else route == "sop_only_fallback"
                    ),
                    "unsafe_count": len(unsafe),
                    "source_run_escape_count": len(source_run_escape),
                    "latency_sec": latency,
                    "routing": meta,
                }
            )
    def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        covered = [row for row in rows if row["gold_transition_count"]]
        gaps = [row for row in rows if not row["gold_transition_count"]]
        tree_rows = [row for row in rows if row["actual_route"] == "causal_tree"]
        return {
            "episode_count": len(rows),
            "covered_episode_count": len(covered),
            "coverage_gap_episode_count": len(gaps),
            "route_accuracy": sum(row["route_correct"] for row in rows) / len(rows),
            "tree_precision_on_covered": sum(row["actual_route"] == "causal_tree" for row in covered) / max(1, len(covered)),
            "fallback_accuracy_on_gaps": sum(row["actual_route"] == "sop_only_fallback" for row in gaps) / max(1, len(gaps)),
            "transition_hit_at_1": sum(bool(row["transition_hit_at_1"]) for row in covered) / max(1, len(covered)),
            "transition_recall_at_3": sum(float(row["transition_recall_at_3"] or 0.0) for row in covered) / max(1, len(covered)),
            "transition_recall_at_5": sum(float(row["transition_recall_at_5"] or 0.0) for row in covered) / max(1, len(covered)),
            "transition_mrr": sum(float(row["transition_mrr"] or 0.0) for row in covered) / max(1, len(covered)),
            "transition_ndcg_at_5": sum(float(row["transition_ndcg_at_5"] or 0.0) for row in covered) / max(1, len(covered)),
            "causal_sop_recall_at_5": sum(float(row["causal_sop_recall_at_5"] or 0.0) for row in covered) / max(1, len(covered)),
            "selective_transition_success_at_1": sum(row["selective_transition_success_at_1"] for row in rows) / max(1, len(rows)),
            "selective_decision_accuracy_at_1": sum(row["selective_decision_accuracy_at_1"] for row in rows) / max(1, len(rows)),
            "top_packet_complete_rate_when_tree": sum(row["top_packet_complete"] for row in tree_rows) / max(1, len(tree_rows)),
            "unsafe_count": sum(row["unsafe_count"] for row in rows),
            "source_run_escape_count": sum(row["source_run_escape_count"] for row in rows),
            "mean_latency_sec": sum(row["latency_sec"] for row in rows) / max(1, len(rows)),
        }
    summary: dict[str, Any] = {}
    for method in DEBUG_METHODS:
        rows = [row for row in receipts if row["method"] == method]
        summary[method] = summarize_rows(rows)
        summary[method]["by_split"] = {
            split: summarize_rows([row for row in rows if row["split"] == split])
            for split in ("dev", "test")
        }
    return receipts, summary


def _taxonomy_level(entries: dict[str, Any], sop_id: str) -> str | None:
    raw = sop_id.removeprefix("sop::")
    return (entries.get(raw) or entries.get(sop_id) or {}).get("abstraction_level")


def evaluate_granularity(graph_path: Path, index_path: Path, *, limit: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    taxonomy = read_json(TAXONOMY)["entries"]
    queries = read_jsonl(EPISODES / "decision_test_v1.jsonl")
    retriever = CompositeRetriever(graph_path, index_path)
    layer = retriever.layers["flat_twin"]
    methods = ("ungated_flat", "tree_only", "sop_only", "stage_hybrid_dynamic", "oracle_level")
    receipts: list[dict[str, Any]] = []
    for query in queries:
        expected = EXPECTED_LEVEL[query["stage"]]
        allowed = set(query["candidate_ids"])
        for method in methods:
            started = time.perf_counter()
            if method == "tree_only":
                ranking = retriever._stage(query, layer_key="tree", top_k=limit)
            elif method == "sop_only":
                ranking = retriever._stage(query, layer_key="sop", top_k=limit)
            elif method == "stage_hybrid_dynamic":
                ranking = retriever._stage(query, layer_key="flat_twin", top_k=limit)
            elif method == "oracle_level":
                ranking = sorted(
                    allowed,
                    key=lambda sop_id: (_taxonomy_level(taxonomy, sop_id) != expected, sop_id),
                )[:limit]
            else:
                rows = []
                for sop_id in allowed:
                    sop = layer.nodes.get(sop_id, {})
                    has_clean_support = any(
                        layer._positive_transition(transition_id)[0]
                        for transition_id in layer._transitions_by_sop.get(sop_id, [])
                    )
                    if sop.get("type") != "SOP" or not has_clean_support:
                        continue
                    text = " ".join(layer._sop_text_parts(sop).values())
                    rows.append((layer._bounded_token_similarity(query["query_text"], text), sop_id))
                ranking = [sop_id for _score, sop_id in sorted(rows, key=lambda item: (-item[0], item[1]))[:limit]]
            latency = time.perf_counter() - started
            levels = [_taxonomy_level(taxonomy, sop_id) for sop_id in ranking]
            correct = sum(level == expected for level in levels)
            receipts.append(
                {
                    "schema": f"{SCHEMA}_granularity_receipt",
                    "track": "stage_granularity",
                    "episode_id": query["episode_id"],
                    "stage": query["stage"],
                    "task_family": query["task_family"],
                    "method": method,
                    "expected_level": expected,
                    "ranking": ranking,
                    "returned_levels": levels,
                    "granularity_precision_at_5": correct / max(1, len(levels)),
                    "detail_intrusion_at_5": (len(levels) - correct) / max(1, len(levels)),
                    "empty_result": not ranking,
                    "latency_sec": latency,
                }
            )
    summary: dict[str, Any] = {}
    for method in methods:
        rows = [row for row in receipts if row["method"] == method]
        summary[method] = {
            "episode_count": len(rows),
            "granularity_precision_at_5": sum(row["granularity_precision_at_5"] for row in rows) / len(rows),
            "detail_intrusion_at_5": sum(row["detail_intrusion_at_5"] for row in rows) / len(rows),
            "empty_result_rate": sum(row["empty_result"] for row in rows) / len(rows),
            "by_stage": {
                stage: {
                    "granularity_precision_at_5": sum(row["granularity_precision_at_5"] for row in rows if row["stage"] == stage)
                    / max(1, sum(row["stage"] == stage for row in rows)),
                    "detail_intrusion_at_5": sum(row["detail_intrusion_at_5"] for row in rows if row["stage"] == stage)
                    / max(1, sum(row["stage"] == stage for row in rows)),
                }
                for stage in EXPECTED_LEVEL
            },
            "mean_latency_sec": sum(row["latency_sec"] for row in rows) / len(rows),
        }
    return receipts, summary


def evaluate(graph_path: Path = GRAPH, index_path: Path = INDEX) -> dict[str, Any]:
    if not QUERY_PATH.exists() or not GOLD_PATH.exists():
        build(graph_path, index_path)
    manifest = read_json(MANIFEST_PATH)
    if sha256_file(graph_path) != manifest["graph_sha256"] or sha256_file(index_path) != manifest["index_sha256"]:
        raise ValueError("benchmark memory snapshot changed; rebuild v2 before evaluation")
    debug_receipts, debug_summary = evaluate_debug(graph_path, index_path)
    granularity_receipts, granularity_summary = evaluate_granularity(graph_path, index_path)
    receipts = [*granularity_receipts, *debug_receipts]
    write_jsonl(RECEIPT_PATH, receipts)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    report = {
        "schema": f"{SCHEMA}_report",
        "code_commit": commit,
        "memory_manifest": manifest,
        "tracks": {
            "stage_granularity": granularity_summary,
            "causal_debug_transfer": debug_summary,
        },
        "primary_comparison": {
            "method": "causal_tree_dynamic",
            "baseline": "legacy_success_tree",
            "split": "test",
            "route_accuracy_delta": (
                debug_summary["causal_tree_dynamic"]["by_split"]["test"]["route_accuracy"]
                - debug_summary["legacy_success_tree"]["by_split"]["test"]["route_accuracy"]
            ),
            "transition_mrr_delta": (
                debug_summary["causal_tree_dynamic"]["by_split"]["test"]["transition_mrr"]
                - debug_summary["legacy_success_tree"]["by_split"]["test"]["transition_mrr"]
            ),
            "selective_decision_accuracy_at_1_delta": (
                debug_summary["causal_tree_dynamic"]["by_split"]["test"]["selective_decision_accuracy_at_1"]
                - debug_summary["legacy_success_tree"]["by_split"]["test"]["selective_decision_accuracy_at_1"]
            ),
        },
        "fallback_ablation": {
            "method": "causal_tree_dynamic",
            "baseline": "causal_tree_fixed_075",
            "split": "test",
            "route_accuracy_delta": (
                debug_summary["causal_tree_dynamic"]["by_split"]["test"]["route_accuracy"]
                - debug_summary["causal_tree_fixed_075"]["by_split"]["test"]["route_accuracy"]
            ),
            "fallback_accuracy_delta": (
                debug_summary["causal_tree_dynamic"]["by_split"]["test"]["fallback_accuracy_on_gaps"]
                - debug_summary["causal_tree_fixed_075"]["by_split"]["test"]["fallback_accuracy_on_gaps"]
            ),
            "transition_mrr_delta": (
                debug_summary["causal_tree_dynamic"]["by_split"]["test"]["transition_mrr"]
                - debug_summary["causal_tree_fixed_075"]["by_split"]["test"]["transition_mrr"]
            ),
            "selective_decision_accuracy_at_1_delta": (
                debug_summary["causal_tree_dynamic"]["by_split"]["test"]["selective_decision_accuracy_at_1"]
                - debug_summary["causal_tree_fixed_075"]["by_split"]["test"]["selective_decision_accuracy_at_1"]
            ),
        },
        "claims": {
            "granularity_gate_diagnostic_allowed": True,
            "cross_run_causal_retrieval_diagnostic_allowed": True,
            "blind_generalization_claim_allowed": False,
            "downstream_agent_success_claim_allowed": False,
            "reasons": [
                "The benchmark is retrospective and the graph was visible during system development.",
                "Failure labels are deterministic silver labels rather than blinded expert annotations.",
                "Retrieving a historically successful transition is not the same as executing a successful new repair.",
            ],
        },
        "receipt_path": str(RECEIPT_PATH),
        "receipt_sha256": sha256_file(RECEIPT_PATH),
    }
    write_json(REPORT_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--graph", type=Path, default=GRAPH)
    parser.add_argument("--index", type=Path, default=INDEX)
    args = parser.parse_args()
    result = build(args.graph, args.index)
    if not args.build_only:
        result = evaluate(args.graph, args.index)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
