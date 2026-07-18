#!/usr/bin/env python3
"""Score generated-code adoption without training or hidden-holdout access."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from core import ARTIFACTS, EPISODES, REPORTS, read_json, read_jsonl, sha256_file, write_json, write_jsonl


FAMILY_PATTERNS = {
    "transformer_classical_hybrid": ("transformer", "tfidf", "xgb", "lightgbm", "logisticregression"),
    "transformer_finetuning": ("automodel", "deberta", "bert", "roberta", "trainer"),
    "gradient_boosted_trees": ("xgb", "lightgbm", "gradientboost"),
    "linear_sparse": ("tfidf", "countvectorizer", "logisticregression", "linearsvc"),
    "cnn": ("conv2d", "resnet", "efficientnet", "cnn"),
    "tree_ensemble": ("randomforest", "extratrees", "xgb", "lightgbm"),
    "bert_family": ("bert",),
    "deberta_family": ("deberta",),
    "deberta_finetune": ("deberta",),
    "deberta_multisample_focal_cv": ("deberta", "multisample", "focal"),
    "deberta_xgb_lr_ensemble": ("deberta", "xgb", "logreg"),
    "frozen_transformer_tree": ("frozen_transformer", "xgb", "tree"),
    "modernbert_family": ("modernbert",),
    "modernbert_finetune": ("modernbert",),
    "roberta_family": ("roberta",),
    "tfidf_stylometry_linear": ("tfidf", "logreg", "logisticregression"),
    "transformer_engineered_feature_hybrid": ("engineered_feature", "tfidf", "stylometry"),
    "vision_transformer_family": ("visiontransformer", "vit"),
    "vision_transformer_finetune": ("visiontransformer", "vit"),
}


def extract_code_facts(code: str) -> dict[str, Any]:
    try:
        tree = ast.parse(code)
        calls = sorted({
            (node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
        })
        literals = sorted({
            node.value.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) <= 240
        })
        parse_ok = True
    except SyntaxError:
        calls, literals, parse_ok = [], [], False
    lowered_calls = {value.lower() for value in calls}
    semantic_text = " ".join([*lowered_calls, *literals])
    families = sorted(
        family for family, tokens in FAMILY_PATTERNS.items() if any(token in semantic_text for token in tokens)
    )
    preprocessing = sorted({
        value for value, tokens in {
            "sparse_text": ("tfidfvectorizer", "countvectorizer"),
            "scaling": ("standardscaler", "minmaxscaler", "robustscaler"),
            "dimensionality_reduction": ("pca", "truncatedsvd"),
            "augmentation": ("compose", "albumentations", "randomcrop"),
        }.items() if any(token in semantic_text for token in tokens)
    })
    losses = sorted(value for value in lowered_calls if "loss" in value or value in {"crossentropy", "focalloss"})
    optimizers = sorted(value for value in lowered_calls if value in {"adam", "adamw", "sgd", "rmsprop", "adagrad"})
    schedulers = sorted(value for value in lowered_calls if "scheduler" in value or "warmup" in value)
    protocol_events = sorted(value for value in lowered_calls if value in {
        "register_partition", "record_fit", "record_prediction", "record_global_oof",
        "record_selection", "freeze", "record_final_evaluation", "assert_clean", "emit",
    })
    return {
        "parse_ok": parse_ok,
        "calls": calls,
        "model_families": families,
        "preprocessing_families": preprocessing,
        "loss_calls": losses,
        "optimizer_calls": optimizers,
        "scheduler_calls": schedulers,
        "protocol_events": protocol_events,
        "uses_cross_validation": any(token in semantic_text for token in ("kfold", "cross_val", "oof")),
        "uses_ensemble": any(token in semantic_text for token in ("ensemble", "blend", "average", "weights")),
        "uses_holdout": "holdout" in semantic_text,
        "checkpoint_literals": [value for value in literals if any(token in value for token in ("checkpoint", "pretrained", "/"))],
    }


def run(candidate_path: Path | None, *, persist: bool = True) -> dict[str, Any]:
    candidates = read_jsonl(candidate_path) if candidate_path else []
    graph = read_json(ARTIFACTS / "memory_snapshot_graph_v1.json")
    nodes = {str(row["id"]): row for row in graph.get("nodes", []) if row.get("id")}
    queries = {row["episode_id"]: row for split in ("dev", "test") for row in read_jsonl(EPISODES / f"decision_{split}_v1.jsonl")}
    receipts: list[dict[str, Any]] = []
    for row in candidates:
        episode_id = str(row.get("episode_id") or "")
        if episode_id not in queries:
            raise ValueError(f"unknown episode_id: {episode_id}")
        refs = [str(value) for value in row.get("selected_memory_ids") or []]
        expected_families = sorted({str(nodes[value].get("method_family") or "") for value in refs if value in nodes})
        facts = extract_code_facts(str(row.get("code") or ""))
        overlap = sorted(set(expected_families) & set(facts["model_families"]))
        explicit = str(row.get("adoption_outcome") or "not_reported")
        receipts.append(
            {
                "schema": "runforest_composite_adoption_receipt_v1",
                "episode_id": episode_id,
                "condition": str(row.get("condition") or "unknown"),
                "seed": int(row.get("seed", 0)),
                "selected_memory_ids": refs,
                "expected_method_families": expected_families,
                "code_facts": facts,
                "family_alignment": bool(overlap) if expected_families else None,
                "aligned_families": overlap,
                "adoption_outcome": explicit,
                "generation_status": str(row.get("status") or "unknown"),
                "prompt_sha256": str(row.get("prompt_sha256") or ""),
                "input_tokens": row.get("input_tokens"),
                "output_tokens": row.get("output_tokens"),
                "model": row.get("model"),
                "mock": bool(row.get("mock", False)),
                "human_adjudicated": bool(row.get("human_adjudicated", False)),
            }
        )
    path = REPORTS / "adoption_receipts_v1.jsonl"
    if persist:
        write_jsonl(path, receipts)
    non_mock = [
        row for row in receipts
        if not row["mock"]
        and row["generation_status"] == "completed"
        and row["code_facts"]["parse_ok"]
        and row.get("adoption_outcome") != "generation_failed"
    ]
    aligned = [row for row in non_mock if row["family_alignment"] is not None]
    complete_by_episode: dict[str, set[str]] = {}
    for row in non_mock:
        complete_by_episode.setdefault(row["episode_id"], set()).add(row["condition"])
    complete_episode_count = sum(
        {"F00", "F01", "F10", "F11"}.issubset(conditions)
        for conditions in complete_by_episode.values()
    )
    report = {
        "schema": "runforest_composite_adoption_report_v1",
        "candidate_count": len(receipts),
        "non_mock_candidate_count": len(non_mock),
        "complete_four_condition_episode_count": complete_episode_count,
        "family_alignment_rate": (
            sum(int(row["family_alignment"]) for row in aligned) / len(aligned) if aligned else None
        ),
        "provenance_complete_rate": (
            sum(int(bool(
                row["prompt_sha256"]
                and row["model"] is not None
                and row["input_tokens"] is not None
                and row["output_tokens"] is not None
            )) for row in non_mock) / len(non_mock)
            if non_mock else None
        ),
        "adoption_claim_allowed": (
            complete_episode_count >= 60
            and len(aligned) == len(non_mock)
            and all(
                row["prompt_sha256"]
                and row["model"] is not None
                and row["input_tokens"] is not None
                and row["output_tokens"] is not None
                for row in non_mock
            )
            and all(row["human_adjudicated"] for row in non_mock)
        ),
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path) if persist else None,
    }
    if persist:
        write_json(REPORTS / "adoption_report_v1.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.candidates), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
