#!/usr/bin/env python3
"""Build immutable manifests, decision episodes, and replay defect cases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from core import (
    ANNOTATIONS,
    ARTIFACTS,
    CONDITIONS,
    DECISION_TEMPLATES,
    DEFECT_SPECS,
    EPISODES,
    GRAPH,
    INDEX,
    MANIFESTS,
    REPLAY_TARGETS,
    REPORTS,
    REPO,
    TASK_SPECS,
    TAXONOMY,
    clean_support,
    counts_by,
    deterministic_order,
    read_json,
    run_id_for_node,
    sha256_file,
    sha256_value,
    write_json,
    write_jsonl,
)


def _split_runs(graph: dict[str, Any]) -> dict[str, list[str]]:
    runs_by_task: dict[str, list[str]] = defaultdict(list)
    for node in graph.get("nodes", []):
        if node.get("type") == "Run":
            runs_by_task[str(node.get("task") or "unknown")].append(str(node.get("run_short_id") or ""))
    split = {"memory_train": [], "benchmark_dev": [], "benchmark_test": []}
    for task, values in sorted(runs_by_task.items()):
        runs = sorted(set(value for value in values if value))
        if len(runs) == 1:
            split["memory_train"].extend(runs)
            continue
        if len(runs) == 2:
            split["memory_train"].append(runs[0])
            split["benchmark_test"].append(runs[1])
            continue
        train_end = max(1, int(round(len(runs) * 0.60)))
        dev_end = min(len(runs) - 1, train_end + max(1, int(round(len(runs) * 0.20))))
        split["memory_train"].extend(runs[:train_end])
        split["benchmark_dev"].extend(runs[train_end:dev_end])
        split["benchmark_test"].extend(runs[dev_end:])
    return {key: sorted(set(value)) for key, value in split.items()}


def _filter_memory_graph(graph: dict[str, Any], train_runs: set[str]) -> dict[str, Any]:
    nodes = {str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")}
    keep: set[str] = {
        node_id
        for node_id, node in nodes.items()
        if node.get("type") == "SOP"
        or (node.get("type") in {"Run", "RunNode", "Transition"} and run_id_for_node(node) in train_runs)
    }
    for edge in graph.get("edges", []):
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        if src in keep and nodes.get(src, {}).get("type") in {"RunNode", "Transition"}:
            if nodes.get(dst, {}).get("type") in {"Evidence", "FailurePattern"}:
                keep.add(dst)
        if dst in keep and nodes.get(dst, {}).get("type") == "RunNode":
            if nodes.get(src, {}).get("type") == "FailurePattern" and run_id_for_node(nodes[src]) in train_runs:
                keep.add(src)
    filtered_edges = [
        copy.deepcopy(edge)
        for edge in graph.get("edges", [])
        if str(edge.get("src") or "") in keep and str(edge.get("dst") or "") in keep
    ]
    filtered = {
        "meta": {
            **copy.deepcopy(graph.get("meta", {})),
            "benchmark_snapshot_schema": "runforest_composite_memory_snapshot_v1",
            "parent_graph_sha256": sha256_value(graph),
            "allowed_run_short_ids": sorted(train_runs),
            "source_run_split_enforced": True,
        },
        "nodes": [copy.deepcopy(nodes[node_id]) for node_id in sorted(keep)],
        "edges": filtered_edges,
    }
    filtered["meta"]["snapshot_sha256"] = sha256_value(filtered)
    return filtered


def _blocked_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for node in graph.get("nodes", []):
        if node.get("type") != "RunNode":
            continue
        audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
        if audit.get("rank_eligible") is False or audit.get("memory_disposition") in {"quarantine", "negative_only"}:
            rows.append(node)
    return rows


def _eligible_sops(
    graph: dict[str, Any], support: dict[str, list[str]], family: str, stage: str
) -> list[dict[str, Any]]:
    rows = []
    for node in graph.get("nodes", []):
        if node.get("type") != "SOP" or not support.get(str(node["id"])):
            continue
        families = {str(value) for value in node.get("task_families") or []}
        stages = {str(value) for value in node.get("decision_stages") or []}
        if family in families and stage in stages:
            rows.append(node)
    return rows


def _family_compatible(selected_family: str, candidate_family: str) -> bool:
    if candidate_family in {"", "general", selected_family}:
        return True
    stem_rules = {
        "deberta": "deberta_family", "modernbert": "modernbert_family",
        "roberta": "roberta_family", "distilbert": "bert_family",
        "efficientnet": "cnn_vision_family", "convnext": "cnn_vision_family",
        "multi_cnn": "cnn_vision_family", "vision_transformer": "vision_transformer_family",
        "vision_tabular": "vision_transformer_family",
    }
    if any(stem in selected_family and candidate_family == family for stem, family in stem_rules.items()):
        return True
    if "transformer" in selected_family and candidate_family in {
        "deberta_family", "modernbert_family", "roberta_family", "bert_family",
    }:
        return True
    if any(token in selected_family for token in ("xgboost", "classical", "tree")):
        return candidate_family == "boosted_tree_family"
    return False


def _pick_gold(rows: list[dict[str, Any]], episode_id: str) -> list[dict[str, Any]]:
    ordered_ids = deterministic_order((str(row["id"]) for row in rows), episode_id)
    by_id = {str(row["id"]): row for row in rows}
    selected: list[dict[str, Any]] = []
    seen_families: set[str] = set()
    for sop_id in ordered_ids:
        row = by_id[sop_id]
        method_family = str(row.get("method_family") or "general")
        if method_family not in seen_families:
            selected.append(row)
            seen_families.add(method_family)
        if len(selected) == 3:
            return selected
    for sop_id in ordered_ids:
        row = by_id[sop_id]
        if row not in selected:
            selected.append(row)
        if len(selected) == 3:
            break
    return selected


def _query_text(task: dict[str, Any], template: dict[str, str], *, dev_variant: bool = False) -> str:
    prefix = "Development decision" if dev_variant else "Held-out decision"
    return "\n".join(
        [
            f"{prefix} for task: {task['task_id']}",
            f"Task family: {task['family']}",
            f"Task profile: {task['profile']}",
            f"Decision stage: {template['stage']}",
            f"Decision request: {template['prompt']}",
            "Return only task-compatible memories with clean execution evidence. Do not infer a historical child node.",
        ]
    )


def _build_episode(
    task: dict[str, Any],
    template: dict[str, str],
    *,
    split: str,
    graph: dict[str, Any],
    support: dict[str, list[str]],
    blocked: list[dict[str, Any]],
    suffix: str = "",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    retrieval_family = str(task.get("retrieval_family") or task["family"])
    episode_id = f"composite::{split}::{task['task_id']}::{template['name']}{suffix}"
    eligible = _eligible_sops(graph, support, retrieval_family, template["stage"])
    selected_l1: dict[str, Any] | None = None
    if template["stage"] == "model_design":
        l1 = [
            row for row in _eligible_sops(graph, support, retrieval_family, "draft")
            if row.get("abstraction_level") == "L1_strategy"
        ]
        if l1:
            selected_l1_id = deterministic_order((str(row["id"]) for row in l1), f"l1|{episode_id}")[0]
            selected_l1 = next(row for row in l1 if str(row["id"]) == selected_l1_id)
            primary_family = str(selected_l1.get("method_family") or "")
            eligible = [
                row for row in eligible
                if row.get("abstraction_level") == "L2_tactic"
                and _family_compatible(primary_family, str(row.get("method_family") or "general"))
            ]
        else:
            eligible = []
    selected = _pick_gold(eligible, episode_id)
    coverage_gap = len(selected) < 3
    sops = sorted(str(node["id"]) for node in graph.get("nodes", []) if node.get("type") == "SOP")
    task_blocked = [node for node in blocked if str(node.get("task") or "") == task["source_task"]]
    if len(task_blocked) < 5:
        task_blocked = blocked
    blocked_ids = deterministic_order((str(node["id"]) for node in task_blocked), episode_id)[:5]
    candidate_ids = [*sops, *blocked_ids]
    query = {
        "schema": "runforest_composite_decision_v1",
        "episode_id": episode_id,
        "task_id": task["task_id"],
        "task_family": task["family"],
        "retrieval_family": retrieval_family,
        "source_task": task["source_task"],
        "selected_l1_sop_id": str(selected_l1.get("id")) if selected_l1 else None,
        "primary_method_family": str(selected_l1.get("method_family")) if selected_l1 else None,
        "stage": template["stage"],
        "decision_name": template["name"],
        "split": split,
        "query_text": _query_text(task, template, dev_variant=split == "dev"),
        "candidate_ids": candidate_ids,
        "candidate_count": len(candidate_ids),
        "source_run_ids_exposed": False,
        "expected_status": "insufficient_strategy_coverage" if coverage_gap else "rank_candidates",
    }
    labels = [
        {
            "candidate_id": str(row["id"]),
            "relevance": grade,
            "method_family": str(row.get("method_family") or "general"),
            "clean_supporting_transition_ids": support[str(row["id"])],
            "rationale": "Silver task/stage-compatible seed pending two-person blind adjudication.",
        }
        for grade, row in zip((3, 2, 1), selected)
    ]
    gold = {
        "schema": "runforest_composite_silver_gold_v1",
        "episode_id": episode_id,
        "labels": labels,
        "expected_status": query["expected_status"],
        "annotation_status": "silver_requires_two_blind_annotators",
        "annotator_count": 0,
        "adjudicated": False,
    }
    nodes = {str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")}
    packet = {
        "schema": "runforest_composite_blind_packet_v1",
        "episode_id": episode_id,
        "query_text": query["query_text"],
        "candidates": [
            {
                "candidate_id": candidate_id,
                "candidate_type": nodes[candidate_id].get("type"),
                "title": str(nodes[candidate_id].get("title") or ""),
                "action": str(nodes[candidate_id].get("action") or ""),
                "relevance_0_to_3": None,
                "safety_label": None,
            }
            for candidate_id in deterministic_order(candidate_ids, f"blind|{episode_id}")
        ],
    }
    return query, gold, packet


def _defect_code(defect: str, family: str, variant: int) -> str:
    family_sources = {
        "text_classification": "raw_records = text_documents\nentity_key = author_ids\norder_key = publication_time\n",
        "image_classification": "raw_records = image_tensors\nentity_key = capture_session_ids\norder_key = capture_time\n",
        "image_restoration": "raw_records = corrupted_patches\nentity_key = source_document_ids\norder_key = scan_time\n",
        "tabular_multiclass": "raw_records = feature_frame\nentity_key = subject_ids\norder_key = observation_time\n",
        "tabular_regression": "raw_records = regression_frame\nentity_key = trip_ids\norder_key = event_time\n",
        "group_time_aware": "raw_records = longitudinal_frame\nentity_key = patient_ids\norder_key = visit_time\n",
    }
    header = (
        f"# benchmark defect={defect} family={family} variant={variant}\nimport numpy as np\n"
        + family_sources[family]
        + "X_train = raw_records[train_idx]\nX_valid = raw_records[valid_idx]\nX_test = raw_records[test_idx]\n"
    )
    snippets = {
        "transductive_fit": (
            "X_all = np.concatenate([X_train, X_valid, X_test])\nvectorizer.fit(X_all)\nXtr = vectorizer.transform(X_train)\n",
            "leaked_fit_rows = np.vstack((X_train, X_valid))\nscaler.fit_transform(leaked_fit_rows)\nXtr = scaler.transform(X_train)\n",
        ),
        "early_stop_report_reuse": (
            "model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])\nscore = log_loss(y_valid, model.predict_proba(X_valid))\n",
            "history = network.fit(X_train, y_train, validation_data=(X_valid, y_valid))\nscore = accuracy(y_valid, network.predict(X_valid))\n",
        ),
        "false_oof": (
            "oof = model.predict_proba(X_valid)\nreported_oof = oof\n",
            "reported_oof_predictions = classifier.predict(X_valid)\n",
        ),
        "holdout_weight_search": (
            "best_weights = min(weight_grid, key=lambda w: log_loss(y_holdout, blend(holdout_predictions, w)))\n",
            "optimized_weights = minimize(lambda w: brier(y_valid, blend(valid_predictions, w)), initial_weights).x\n",
        ),
        "group_split_leakage": (
            "train_idx, valid_idx = train_test_split(np.arange(len(y)), random_state=42)\n# entity_ids are ignored\n",
            "train_idx, valid_idx = train_test_split(np.arange(len(y)), stratify=y, random_state=7)\n# group_key is available but not passed to the splitter\n",
        ),
        "temporal_order_leakage": (
            "train_idx, valid_idx = train_test_split(np.arange(len(y)), shuffle=True, random_state=42)\n# timestamps are ignored\n",
            "train_idx, valid_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=7)\n# chronological order is available but the default shuffle remains enabled\n",
        ),
        "target_encoding_leakage": (
            "encoder.fit(np.concatenate([X_train, X_valid]), np.concatenate([y_train, y_valid]))\n",
            "target_encoder.fit(X_valid, y_valid)\nencoded_train = target_encoder.transform(X_train)\n",
        ),
        "post_split_dedup": (
            "X_train = deduplicate(X_train)\nX_valid = deduplicate(X_valid)\n",
            "X_train = X_train.drop_duplicates()\nX_valid = X_valid.drop_duplicates()\n",
        ),
    }
    return header + snippets[defect][variant]


def build(graph_path: Path = GRAPH) -> dict[str, Any]:
    graph = read_json(graph_path)
    run_split = _split_runs(graph)
    snapshot = _filter_memory_graph(graph, set(run_split["memory_train"]))
    snapshot_path = ARTIFACTS / "memory_snapshot_graph_v1.json"
    write_json(snapshot_path, snapshot)
    support = clean_support(snapshot)
    blocked = _blocked_nodes(snapshot)

    test_queries: list[dict[str, Any]] = []
    test_gold: list[dict[str, Any]] = []
    test_packets: list[dict[str, Any]] = []
    for task in TASK_SPECS:
        for template in DECISION_TEMPLATES:
            query, gold, packet = _build_episode(
                task, template, split="test", graph=snapshot, support=support, blocked=blocked
            )
            test_queries.append(query)
            test_gold.append(gold)
            test_packets.append(packet)

    dev_queries: list[dict[str, Any]] = []
    dev_gold: list[dict[str, Any]] = []
    dev_packets: list[dict[str, Any]] = []
    for index, task in enumerate(TASK_SPECS[:10]):
        for template in DECISION_TEMPLATES[:2]:
            query, gold, packet = _build_episode(
                task,
                template,
                split="dev",
                graph=snapshot,
                support=support,
                blocked=blocked,
                suffix=f"::pilot{index}",
            )
            dev_queries.append(query)
            dev_gold.append(gold)
            dev_packets.append(packet)

    defects = []
    families = sorted({str(task["family"]) for task in TASK_SPECS})
    for defect_index, spec in enumerate(DEFECT_SPECS):
        for variant in (0, 1):
            family_offset = defect_index + variant * 3
            for family_index in range(3):
                family = families[(family_offset + family_index) % len(families)]
                code = _defect_code(spec["defect"], family, variant)
                defects.append(
                    {
                        "schema": "runforest_composite_replay_defect_v1",
                        "case_id": f"repair::{family}::{spec['defect']}::v{variant + 1}",
                        "task_family": family,
                        "implementation_variant": variant + 1,
                        **spec,
                        "code": code,
                        "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                        "expected_execution": "blocked_before_gpu",
                        "expected_rank_eligible": False,
                        "preservation_contract_required": True,
                    }
                )

    write_jsonl(EPISODES / "decision_test_v1.jsonl", test_queries)
    write_jsonl(EPISODES / "decision_dev_v1.jsonl", dev_queries)
    write_jsonl(EPISODES / "decision_test_silver_gold_v1.jsonl", test_gold)
    write_jsonl(EPISODES / "decision_dev_silver_gold_v1.jsonl", dev_gold)
    write_jsonl(EPISODES / "replay_defects_v1.jsonl", defects)
    write_jsonl(ANNOTATIONS / "blind_packet_a_v1.jsonl", test_packets)
    write_jsonl(
        ANNOTATIONS / "blind_packet_b_v1.jsonl",
        [{**row, "candidates": list(reversed(row["candidates"]))} for row in test_packets],
    )

    replay_targets = read_json(REPLAY_TARGETS)
    replay_task_ids = sorted({str(row.get("task_id") or "") for row in replay_targets.get("targets", [])})
    task_manifest = {
        "schema": "runforest_composite_task_manifest_v1",
        "tasks": list(TASK_SPECS),
        "task_count": len(TASK_SPECS),
        "task_family_count": len({task["family"] for task in TASK_SPECS}),
        "replay_eligible_task_ids": replay_task_ids,
        "replay_eligible_task_count": len(replay_task_ids),
        "minimum_replay_eligible_for_role_claim": 8,
    }
    condition_manifest = {
        "schema": "runforest_composite_condition_manifest_v1",
        "conditions": CONDITIONS,
        "equal_opportunity_budget": {
            "root_slots": 3,
            "same_wall_clock": True,
            "same_gpu_limit": True,
            "same_model_call_limit": True,
            "same_output_token_limit": True,
            "input_tokens_are_measured_cost_not_padded": True,
        },
    }
    claim_manifest = {
        "schema": "runforest_composite_claim_gates_v1",
        "mechanism": {
            "minimum_test_episodes": 120,
            "minimum_task_families": 6,
            "minimum_per_family": 20,
            "minimum_annotators": 2,
            "minimum_krippendorff_alpha": 0.67,
            "role_isolation_violations": 0,
            "unsafe_escape_count": 0,
            "provenance_completeness": 1.0,
        },
        "downstream": {
            "minimum_tasks": 10,
            "minimum_seeds": 3,
            "maximum_standardized_mde": 0.8,
            "primary_comparisons": ["F11-B0", "F11-F10", "F11-F01"],
            "holm_correction": True,
        },
    }
    memory_manifest = {
        "schema": "runforest_composite_memory_snapshot_manifest_v1",
        "source_graph": str(graph_path.relative_to(graph_path.parents[2])),
        "source_graph_sha256": sha256_file(graph_path),
        "source_index_sha256": sha256_file(INDEX),
        "taxonomy_sha256": sha256_file(TAXONOMY),
        "snapshot_path": str(snapshot_path.relative_to(REPO)),
        "snapshot_sha256": sha256_file(snapshot_path),
        "run_split": run_split,
        "split_unit": "source_run_short_id",
    }
    write_json(MANIFESTS / "task_manifest_v1.yaml", task_manifest)
    write_json(MANIFESTS / "condition_manifest_v1.yaml", condition_manifest)
    write_json(MANIFESTS / "claim_gates_v1.yaml", claim_manifest)
    write_json(MANIFESTS / "memory_snapshot_manifest_v1.json", memory_manifest)

    coverage_gaps = [row for row in test_queries if row["expected_status"] == "insufficient_strategy_coverage"]
    report = {
        "schema": "runforest_composite_build_report_v1",
        "valid": True,
        "test_episode_count": len(test_queries),
        "dev_episode_count": len(dev_queries),
        "test_by_family": counts_by(test_queries, "task_family"),
        "test_by_stage": counts_by(test_queries, "stage"),
        "coverage_gap_count": len(coverage_gaps),
        "coverage_gap_episode_ids": [row["episode_id"] for row in coverage_gaps],
        "replay_defect_count": len(defects),
        "replay_eligible_task_count": len(replay_task_ids),
        "role_decomposition_claim_ready": len(replay_task_ids) >= 8,
        "blind_annotation_complete": False,
        "offline_relevance_claim_allowed": False,
        "downstream_claim_allowed": False,
        "artifacts": {
            "task_manifest_sha256": sha256_file(MANIFESTS / "task_manifest_v1.yaml"),
            "condition_manifest_sha256": sha256_file(MANIFESTS / "condition_manifest_v1.yaml"),
            "claim_manifest_sha256": sha256_file(MANIFESTS / "claim_gates_v1.yaml"),
            "test_episode_sha256": sha256_file(EPISODES / "decision_test_v1.jsonl"),
            "test_gold_sha256": sha256_file(EPISODES / "decision_test_silver_gold_v1.jsonl"),
            "replay_defects_sha256": sha256_file(EPISODES / "replay_defects_v1.jsonl"),
        },
    }
    write_json(REPORTS / "build_report_v1.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=GRAPH)
    args = parser.parse_args()
    print(json.dumps(build(args.graph), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
