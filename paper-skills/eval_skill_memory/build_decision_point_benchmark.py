#!/usr/bin/env python3
"""Build an independent decision-point benchmark for RunForest memory retrieval.

The benchmark deliberately does not use historical parent/child transitions as
queries or gold labels.  Each record describes a task decision in ordinary
language and is paired with graded, expert-seeded SOP labels.  The labels are
silver until two blind human annotators complete the emitted annotation packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
DEFAULT_BENCHMARK = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "decision_point_benchmark_v1.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "decision_point_silver_gold_v1.jsonl"
DEFAULT_PACKET = REPO / "paper-skills" / "eval_skill_memory" / "annotation" / "decision_point_blind_packet_v1.jsonl"
DEFAULT_REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "decision_point_benchmark_validation_v1.json"


INTENT_TEXT = {
    "efficient_baseline": "Choose a credible, compute-conscious baseline before spending the full training budget.",
    "high_capacity_route": "Choose a higher-capacity primary model route with a clear generalization hypothesis.",
    "hybrid_or_ensemble": "Decide whether complementary representations or model families should be combined.",
    "representation_design": "Improve the representation or feature design without changing the task definition.",
    "validation_design": "Design a validation procedure that supports model selection without optimistic reuse.",
    "regularization": "Improve generalization and probability quality while preserving the selected model family.",
    "fold_local_preprocessing": "Repair a split protocol where learned preprocessing may see validation information.",
    "shape_or_alignment": "Repair a tensor, feature, index, or prediction alignment failure.",
    "resource_failure": "Repair a memory or worker failure without silently changing the scientific method.",
    "api_failure": "Repair a library or model API mismatch while preserving the intended architecture.",
    "augmentation": "Choose an augmentation or inference-time robustness tactic appropriate for the modality.",
    "progressive_training": "Choose a staged training or resizing schedule under a fixed compute budget.",
    "label_loss_failure": "Repair a label, loss, or metric mismatch that invalidates optimization or reporting.",
    "multiscale_features": "Choose how to extract and combine information at multiple visual scales.",
    "multimodal_fusion": "Choose a principled fusion route for image and structured feature groups.",
    "feature_engineering": "Choose structured feature transformations that remain fold-local and reproducible.",
    "missing_categorical": "Repair missing-value and categorical handling without corrupting feature semantics.",
    "calibration": "Improve calibration or confidence quality without tuning on the final evaluation set.",
    "temporal_validation": "Choose a time-aware validation protocol that respects deployment chronology.",
    "categorical_interface": "Repair categorical feature typing or indexing for boosted-tree training.",
    "objective_design": "Choose an objective and target representation appropriate for the regression target.",
    "indexing_failure": "Repair dataframe or array indexing that selects the wrong rows or columns.",
    "patch_reconstruction": "Choose a patch extraction and reconstruction protocol that avoids seam artifacts.",
    "loss_design": "Choose a restoration loss that balances pixel, structure, and edge fidelity.",
    "noise_prior": "Choose an architecture or input prior that exposes document noise structure.",
    "device_dtype": "Repair device or dtype inconsistency in mixed-precision restoration training.",
}


FAMILY_SPECS: list[dict[str, Any]] = [
    {
        "task_family": "text_classification",
        "task": "spooky-author-identification",
        "profile": "Three-class authorship classification; log loss; long-form text; one GPU to multi-GPU budget.",
        "points": [
            ("efficient_baseline", "draft", "L1_strategy", ["sg_0144", "sg_0111", "sg_0175"]),
            ("high_capacity_route", "draft", "L1_strategy", ["sg_0213", "sg_0221", "sg_0088"]),
            ("hybrid_or_ensemble", "draft", "L1_strategy", ["sg_0164", "sg_0202", "sg_0279"]),
            ("representation_design", "improve", "L2_tactic", ["sg_0227", "sg_0262", "sg_0128"]),
            ("validation_design", "improve", "L2_tactic", ["sg_0278", "sg_0226", "sg_0146"]),
            ("regularization", "improve", "L2_tactic", ["sg_0216", "sg_0121", "sg_0228"]),
            ("fold_local_preprocessing", "debug", "L3_repair", ["sg_0211", "sg_0120", "sg_0278"]),
            ("shape_or_alignment", "debug", "L3_repair", ["sg_0101", "sg_0238", "sg_0241"]),
            ("resource_failure", "debug", "L3_repair", ["sg_0094", "sg_0095", "sg_0085"]),
            ("api_failure", "debug", "L3_repair", ["sg_0115", "sg_0148", "sg_0237"]),
        ],
    },
    {
        "task_family": "image_binary_classification",
        "task": "aerial-cactus-identification",
        "profile": "Binary aerial image classification; AUC; limited labelled images; single-GPU budget.",
        "points": [
            ("efficient_baseline", "draft", "L1_strategy", ["sg_0007", "sg_0009", "sg_0011"]),
            ("high_capacity_route", "draft", "L1_strategy", ["sg_0009", "sg_0007", "sg_0011"]),
            ("hybrid_or_ensemble", "draft", "L1_strategy", ["sg_0011", "sg_0007", "sg_0009"]),
            ("representation_design", "improve", "L2_tactic", ["sg_0010", "sg_0005", "sg_0006"]),
            ("augmentation", "improve", "L2_tactic", ["sg_0005", "sg_0007", "sg_0008"]),
            ("progressive_training", "improve", "L2_tactic", ["sg_0006", "sg_0005", "sg_0010"]),
            ("fold_local_preprocessing", "debug", "L3_repair", ["sg_0054", "sg_0063", "sg_0077"]),
            ("label_loss_failure", "debug", "L3_repair", ["sg_0018", "sg_0067", "sg_0126"]),
            ("resource_failure", "debug", "L3_repair", ["sg_0013", "sg_0035", "sg_0096"]),
            ("api_failure", "debug", "L3_repair", ["sg_0016", "sg_0049", "sg_0065"]),
        ],
    },
    {
        "task_family": "image_classification",
        "task": "leaf-classification",
        "profile": "Fine-grained leaf image classification; multiclass log loss; pretrained vision encoders available.",
        "points": [
            ("efficient_baseline", "draft", "L1_strategy", ["sg_0041", "sg_0040", "sg_0044"]),
            ("high_capacity_route", "draft", "L1_strategy", ["sg_0041", "sg_0040", "sg_0044"]),
            ("hybrid_or_ensemble", "draft", "L1_strategy", ["sg_0040", "sg_0044", "sg_0041"]),
            ("multiscale_features", "improve", "L2_tactic", ["sg_0043", "sg_0041", "sg_0058"]),
            ("augmentation", "improve", "L2_tactic", ["sg_0058", "sg_0043", "sg_0041"]),
            ("validation_design", "improve", "L2_tactic", ["sg_0060", "sg_0058", "sg_0124"]),
            ("fold_local_preprocessing", "debug", "L3_repair", ["sg_0054", "sg_0063", "sg_0064"]),
            ("shape_or_alignment", "debug", "L3_repair", ["sg_0050", "sg_0049", "sg_0067"]),
            ("resource_failure", "debug", "L3_repair", ["sg_0048", "sg_0013", "sg_0035"]),
            ("api_failure", "debug", "L3_repair", ["sg_0046", "sg_0065", "sg_0066"]),
        ],
    },
    {
        "task_family": "tabular_multiclass",
        "task": "leaf-classification",
        "profile": "Multiclass structured leaf descriptors; log loss; heterogeneous numeric feature groups.",
        "points": [
            ("efficient_baseline", "draft", "L1_strategy", ["sg_0044", "sg_0040", "sg_0041"]),
            ("feature_engineering", "improve", "L2_tactic", ["sg_0042", "sg_0056", "sg_0057"]),
            ("multimodal_fusion", "improve", "L2_tactic", ["sg_0057", "sg_0040", "sg_0044"]),
            ("hybrid_or_ensemble", "improve", "L2_tactic", ["sg_0044", "sg_0058", "sg_0150"]),
            ("validation_design", "improve", "L2_tactic", ["sg_0060", "sg_0058", "sg_0124"]),
            ("calibration", "improve", "L2_tactic", ["sg_0150", "sg_0121", "sg_0134"]),
            ("fold_local_preprocessing", "debug", "L3_repair", ["sg_0054", "sg_0063", "sg_0064"]),
            ("missing_categorical", "debug", "L3_repair", ["sg_0070", "sg_0075", "sg_0073"]),
            ("shape_or_alignment", "debug", "L3_repair", ["sg_0067", "sg_0101", "sg_0238"]),
            ("api_failure", "debug", "L3_repair", ["sg_0046", "sg_0066", "sg_0047"]),
        ],
    },
    {
        "task_family": "tabular_regression",
        "task": "new-york-city-taxi-fare-prediction",
        "profile": "Taxi fare regression; RMSE; spatial, categorical, and chronological structure.",
        "points": [
            ("efficient_baseline", "improve", "L2_tactic", ["sg_0078", "sg_0082", "sg_0080"]),
            ("high_capacity_route", "improve", "L2_tactic", ["sg_0078", "sg_0082", "sg_0080"]),
            ("feature_engineering", "improve", "L2_tactic", ["sg_0078", "sg_0073", "sg_0075"]),
            ("temporal_validation", "improve", "L2_tactic", ["sg_0079", "sg_0077", "sg_0054"]),
            ("validation_design", "improve", "L2_tactic", ["sg_0079", "sg_0078", "sg_0060"]),
            ("fold_local_preprocessing", "debug", "L3_repair", ["sg_0077", "sg_0079", "sg_0054"]),
            ("categorical_interface", "debug", "L3_repair", ["sg_0072", "sg_0073", "sg_0075"]),
            ("objective_design", "debug", "L3_repair", ["sg_0082", "sg_0080", "sg_0078"]),
            ("indexing_failure", "debug", "L3_repair", ["sg_0075", "sg_0072", "sg_0073"]),
            ("resource_failure", "debug", "L3_repair", ["sg_0083", "sg_0013", "sg_0035"]),
        ],
    },
    {
        "task_family": "image_restoration",
        "task": "denoising-dirty-documents",
        "profile": "Document image denoising; RMSE/SSIM; patch training; high-resolution inference.",
        "points": [
            ("high_capacity_route", "improve", "L2_tactic", ["sg_0028", "sg_0033", "sg_0030"]),
            ("patch_reconstruction", "improve", "L2_tactic", ["sg_0027", "sg_0030", "sg_0038"]),
            ("loss_design", "improve", "L2_tactic", ["sg_0032", "sg_0028", "sg_0036"]),
            ("noise_prior", "improve", "L2_tactic", ["sg_0029", "sg_0031", "sg_0033"]),
            ("multiscale_features", "improve", "L2_tactic", ["sg_0030", "sg_0031", "sg_0033"]),
            ("validation_design", "improve", "L2_tactic", ["sg_0032", "sg_0028", "sg_0060"]),
            ("fold_local_preprocessing", "debug", "L3_repair", ["sg_0054", "sg_0063", "sg_0077"]),
            ("shape_or_alignment", "debug", "L3_repair", ["sg_0034", "sg_0038", "sg_0050"]),
            ("resource_failure", "debug", "L3_repair", ["sg_0035", "sg_0013", "sg_0096"]),
            ("device_dtype", "debug", "L3_repair", ["sg_0036", "sg_0037", "sg_0154"]),
        ],
    },
]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z][a-z0-9_+-]{2,}", text.lower()))


def _candidate_text(node: dict[str, Any]) -> str:
    if node.get("type") == "SOP":
        return " ".join(
            str(node.get(key) or "")
            for key in ("title", "action", "text", "method_family", "sop_kind", "abstraction_level")
        )
    return " ".join(str(node.get(key) or "") for key in ("plan", "code_summary", "analysis", "terminal_excerpt", "text"))


def _positive_child(child: dict[str, Any]) -> bool:
    """Match the runtime memory layer's positive-evidence requirements."""
    audit = child.get("leakage_audit") if isinstance(child.get("leakage_audit"), dict) else {}
    metric = child.get("metric")
    return (
        child.get("strategy_alignment_eligible") is not False
        and audit.get("status") == "clean"
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
        and audit.get("rank_eligible") is True
        and child.get("is_valid") is True
        and child.get("is_buggy") is False
        and child.get("quarantined") is not True
        and child.get("protocol_biased") is not True
        and isinstance(metric, (int, float))
        and not isinstance(metric, bool)
    )


def _clean_support(graph: dict[str, Any]) -> dict[str, int]:
    nodes = {str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")}
    counts: Counter[str] = Counter()
    for edge in graph.get("edges", []):
        if str(edge.get("kind") or edge.get("type") or "") != "distills_to":
            continue
        transition = nodes.get(str(edge.get("src") or ""), {})
        child = nodes.get(str(transition.get("child_node_id") or ""), {})
        clean = (
            transition.get("quarantined") is not True
            and transition.get("protocol_biased") is not True
            and _positive_child(child)
        )
        if clean:
            counts[str(edge.get("dst"))] += 1
    return dict(counts)


def _blocked_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for node in graph.get("nodes", []):
        if node.get("type") != "RunNode" or len(_candidate_text(node)) < 120:
            continue
        audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
        if audit.get("rank_eligible") is False or audit.get("memory_disposition") in {"quarantine", "negative_only"}:
            rows.append(node)
    rows.sort(key=lambda item: hashlib.sha256(str(item["id"]).encode()).hexdigest())
    return rows


def _task_families(node: dict[str, Any]) -> set[str]:
    values = node.get("task_families") or []
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values}


def _gold_is_compatible(nodes: dict[str, dict[str, Any]], task_family: str, gold_ids: list[str]) -> bool:
    return all(task_family in _task_families(nodes[gold_id]) for gold_id in gold_ids)


def _query_id(task_family: str, intent: str) -> str:
    return f"decision::{task_family}::{intent}"


def build_records(graph: dict[str, Any], *, candidate_pool_size: int = 0) -> tuple[list[dict], list[dict], list[dict]]:
    nodes = {str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")}
    sops = [node for node in graph.get("nodes", []) if node.get("type") == "SOP"]
    support = _clean_support(graph)
    blocked = _blocked_nodes(graph)
    queries: list[dict] = []
    gold_rows: list[dict] = []
    packets: list[dict] = []
    seen_gold_sets: set[frozenset[str]] = set()

    for family in FAMILY_SPECS:
        for intent, stage, target_level, gold_short_ids in family["points"]:
            qid = _query_id(family["task_family"], intent)
            query_text = "\n".join(
                [
                    f"Task family: {family['task_family']}",
                    f"Task profile: {family['profile']}",
                    f"Decision stage: {stage}",
                    f"Decision intent: {INTENT_TEXT[intent]}",
                    "Return memories that are scientifically applicable, supported by clean execution evidence, and safe to adopt.",
                ]
            )
            gold_ids = [f"sop::{short_id}" for short_id in gold_short_ids]
            gold_set = frozenset(gold_ids)
            if gold_set in seen_gold_sets or not _gold_is_compatible(nodes, family["task_family"], gold_ids):
                continue
            if any(support.get(gold_id, 0) < 1 for gold_id in gold_ids):
                continue
            seen_gold_sets.add(gold_set)

            task_blocked = [node for node in blocked if str(node.get("task") or "") == family["task"]]
            task_blocked.sort(key=lambda node: hashlib.sha256(f"{qid}|{node['id']}".encode()).hexdigest())
            blocked_ids = [str(node["id"]) for node in task_blocked[:5]]
            if len(blocked_ids) < 5:
                continue
            sop_ids = sorted(str(node["id"]) for node in sops)
            candidate_ids = [*sop_ids, *blocked_ids]
            if candidate_pool_size > 0:
                minimum = len(gold_ids) + len(blocked_ids)
                limit = max(minimum, candidate_pool_size)
                deterministic = sorted(
                    (candidate_id for candidate_id in sop_ids if candidate_id not in gold_set),
                    key=lambda candidate_id: hashlib.sha256(f"pool|{qid}|{candidate_id}".encode()).hexdigest(),
                )
                candidate_ids = list(dict.fromkeys([*gold_ids, *blocked_ids, *deterministic]))[:limit]
            query = {
                "schema": "runforest_decision_point_query_v1",
                "query_id": qid,
                "task_family": family["task_family"],
                "task": family["task"],
                "stage": stage,
                "intent": intent,
                "query_text": query_text,
                "candidate_ids": candidate_ids,
                "candidate_count": len(candidate_ids),
                "split": "test",
                "historical_coordinate_free": True,
            }
            labels = []
            for grade, gold_id in zip((3, 2, 1), gold_ids):
                labels.append(
                    {
                        "candidate_id": gold_id,
                        "relevance": grade,
                        "clean_supporting_transition_count": support.get(gold_id, 0),
                        "rationale": "Expert-seeded acceptable memory; requires two-person blind adjudication before paper use.",
                    }
                )
            gold = {
                "schema": "runforest_decision_point_silver_gold_v1",
                "query_id": qid,
                "labels": labels,
                "annotation_status": "silver_expert_seeded_requires_blind_review",
                "annotator_count": 0,
                "adjudicated": False,
            }
            packet_candidates = []
            for candidate_id in sorted(candidate_ids, key=lambda value: hashlib.sha256(f"blind|{qid}|{value}".encode()).hexdigest()):
                node = nodes[candidate_id]
                packet_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_type": node.get("type"),
                        "text": _candidate_text(node)[:4000],
                        "relevance_0_to_3": None,
                        "safety": None,
                        "annotator_rationale": "",
                    }
                )
            packet = {
                "schema": "runforest_decision_point_blind_annotation_v1",
                "query_id": qid,
                "query_text": query_text,
                "instructions": "Score relevance 0-3 and safety allowed/blocked. Do not inspect run coordinates or silver gold.",
                "candidates": packet_candidates,
            }
            queries.append(query)
            gold_rows.append(gold)
            packets.append(packet)
    return queries, gold_rows, packets


def validate_records(graph: dict[str, Any], queries: list[dict], gold_rows: list[dict]) -> dict[str, Any]:
    nodes = {str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")}
    gold_by_id = {row["query_id"]: row for row in gold_rows}
    errors: list[str] = []
    query_texts: set[str] = set()
    by_family: Counter[str] = Counter()
    by_stage: Counter[str] = Counter()
    gold_sets: set[frozenset[str]] = set()
    for query in queries:
        qid = query["query_id"]
        by_family[query["task_family"]] += 1
        by_stage[query["stage"]] += 1
        if query["query_text"] in query_texts:
            errors.append(f"duplicate_query_text:{qid}")
        query_texts.add(query["query_text"])
        forbidden = ("run_id", "parent_node", "child_node", "local_best", "transition::", "run::")
        if any(token in query["query_text"] for token in forbidden):
            errors.append(f"historical_coordinate_leak:{qid}")
        if not 50 <= query["candidate_count"] <= 400:
            errors.append(f"candidate_pool_out_of_range:{qid}:{query['candidate_count']}")
        if len(query["candidate_ids"]) != len(set(query["candidate_ids"])):
            errors.append(f"duplicate_candidates:{qid}")
        if any(candidate_id not in nodes for candidate_id in query["candidate_ids"]):
            errors.append(f"unknown_candidate:{qid}")
        blocked_candidates = [nodes[candidate_id] for candidate_id in query["candidate_ids"] if nodes[candidate_id].get("type") == "RunNode"]
        if len(blocked_candidates) != 5:
            errors.append(f"blocked_distractor_count:{qid}:{len(blocked_candidates)}")
        for candidate in blocked_candidates:
            if str(candidate.get("task") or "") != query["task"]:
                errors.append(f"task_mismatched_blocked_distractor:{qid}:{candidate['id']}")
        gold = gold_by_id.get(qid)
        if gold is None:
            errors.append(f"missing_gold:{qid}")
            continue
        if len(gold.get("labels") or []) < 3:
            errors.append(f"insufficient_acceptable_gold:{qid}")
        gold_set = frozenset(str(label.get("candidate_id")) for label in gold.get("labels") or [])
        if gold_set in gold_sets:
            errors.append(f"duplicate_gold_set_global:{qid}")
        gold_sets.add(gold_set)
        for label in gold.get("labels") or []:
            cid = label.get("candidate_id")
            if cid not in query["candidate_ids"]:
                errors.append(f"gold_outside_pool:{qid}:{cid}")
            if nodes.get(str(cid), {}).get("type") != "SOP":
                errors.append(f"gold_not_sop:{qid}:{cid}")
            if int(label.get("clean_supporting_transition_count") or 0) < 1:
                errors.append(f"gold_without_clean_execution_evidence:{qid}:{cid}")
            families = _task_families(nodes.get(str(cid), {}))
            if query["task_family"] not in families:
                errors.append(f"cross_task_gold:{qid}:{cid}")
    if len(queries) < 25:
        errors.append(f"underpowered_query_count:{len(queries)}")
    if len(by_family) < 6 or any(count < 3 for count in by_family.values()):
        errors.append(f"task_family_coverage:{dict(by_family)}")
    paper_claim_ready = all(row.get("annotator_count", 0) >= 2 and row.get("adjudicated") is True for row in gold_rows)
    seed_count = sum(len(family["points"]) for family in FAMILY_SPECS)
    return {
        "schema": "runforest_decision_point_benchmark_validation_v1",
        "valid": not errors,
        "errors": errors,
        "query_count": len(queries),
        "seed_query_count": seed_count,
        "strict_retention_rate": len(queries) / seed_count,
        "sampling_design": "deterministic convenience seeds filtered by strict evidence and safety eligibility; not a random task sample",
        "task_family_count": len(by_family),
        "by_task_family": dict(sorted(by_family.items())),
        "by_stage": dict(sorted(by_stage.items())),
        "candidate_pool_min": min((row["candidate_count"] for row in queries), default=0),
        "candidate_pool_max": max((row["candidate_count"] for row in queries), default=0),
        "historical_trajectory_gold_used": False,
        "gold_sets_unique_globally": len(gold_sets) == len(queries),
        "cross_task_gold_allowed": False,
        "general_task_family_escape_hatch_allowed": False,
        "blocked_distractors_per_query": 5,
        "blocked_distractors_task_matched": True,
        "blind_annotator_requirement": 2,
        "paper_claim_ready": paper_claim_ready,
        "claim_gate_reason": "Silver labels are diagnostic only; two blind annotators and adjudication are still required." if not paper_claim_ready else "passed",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--benchmark-out", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--gold-out", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--annotation-packet-out", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--candidate-pool-size", type=int, default=0, help="0 evaluates the full SOP inventory plus five task-matched blocked nodes")
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    queries, gold, packets = build_records(graph, candidate_pool_size=args.candidate_pool_size)
    report = validate_records(graph, queries, gold)
    if not report["valid"]:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    _write_jsonl(args.benchmark_out, queries)
    _write_jsonl(args.gold_out, gold)
    _write_jsonl(args.annotation_packet_out, packets)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
