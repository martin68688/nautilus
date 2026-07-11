#!/usr/bin/env python3
"""Build the deterministic SOP taxonomy used by layered Novel Draft retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_OVERRIDES = REPO / "paper-skills" / "hyper_memory" / "sop_taxonomy_overrides.json"
DEFAULT_OUTPUT = REPO / "paper-skills" / "hyper_memory" / "sop_taxonomy.json"

SCHEMA = "runforest_sop_taxonomy_v1"
ABSTRACTION_LEVELS = {"L1_strategy", "L2_tactic", "L3_repair"}
SOP_KINDS = {
    "model_strategy",
    "architecture",
    "feature",
    "training_protocol",
    "validation_protocol",
    "debug_fix",
    "infrastructure",
}
COMPUTE_PROFILES = {
    "cpu_light",
    "cpu_or_single_gpu",
    "single_gpu_standard",
    "single_gpu_large",
    "multi_gpu_preferred",
}

TASK_FAMILIES = {
    "spooky-author-identification": ["text_classification"],
    "aerial-cactus-identification": ["image_binary_classification"],
    "denoising-dirty-documents": ["image_restoration"],
    "leaf-classification": ["tabular_multiclass", "image_classification"],
    "new-york-city-taxi-fare-prediction": ["tabular_regression"],
    "general": ["general"],
}

NARROW_TACTIC_PATTERNS = (
    "early stopping",
    "adamw",
    "gradient clipping",
    "focal loss",
    "label smoothing",
    "mean pooling",
    "attention pooling",
    "mixed precision",
    "gradscaler",
    "temperature scaling",
    "cross-validation",
    "cross validation",
    "stratified k-fold",
    "stratified kfold",
    "progressive unfreezing",
    "differential learning rate",
    "scheduler",
    "exponential moving average",
    "multi-sample dropout",
    "augmentation",
    "test-time augmentation",
    "tf-idf n-gram features",
)

REPAIR_PATTERNS = (
    "avoid ",
    "correct ",
    "ensure ",
    "define all",
    "before use",
    "file path",
    "hardcoded",
    "attribute",
    "dtype",
    "tensor dimension",
    "shape matches",
    "nameerror",
    "out of memory",
    "oom",
    "cuda error",
    "merge conflict",
    "stray text",
    "fit on training fold only",
    "prevent data leakage",
    "avoid data leakage",
    "no space left",
    "shared memory",
)

MODEL_PATTERNS = (
    "bert",
    "deberta",
    "roberta",
    "transformer",
    "efficientnet",
    "resnet",
    "siglip",
    "dinov2",
    "vision transformer",
    "vit",
    "unet",
    "u-net",
    "xgboost",
    "lightgbm",
    "logistic regression",
    "mlp",
    "cnn",
    "lstm",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(sop: dict[str, Any]) -> str:
    values = [sop.get("title"), sop.get("action"), sop.get("principle"), sop.get("condition")]
    values.extend(sop.get("applies_when") or [])
    values.extend(sop.get("prevents") or [])
    return " ".join(str(value or "") for value in values).lower()


def _task_families(sop: dict[str, Any]) -> list[str]:
    category = str(sop.get("category") or "general")
    return list(TASK_FAMILIES.get(category, ["general"]))


def _is_repair(sop: dict[str, Any], text: str) -> bool:
    category = str(sop.get("category") or "")
    if category == "api_warning":
        return True
    return any(pattern in text for pattern in REPAIR_PATTERNS)


def _looks_like_strategy(sop: dict[str, Any], text: str) -> bool:
    if str(sop.get("scope") or "") != "task_specific":
        return False
    title = str(sop.get("title") or "").lower()
    if any(pattern in title for pattern in NARROW_TACTIC_PATTERNS):
        return False
    model_count = sum(1 for pattern in MODEL_PATTERNS if pattern in text)
    full_route_signal = any(
        signal in text
        for signal in (
            "fine-tune",
            "finetune",
            "feature extractor",
            "train an xgboost",
            "train a lightgbm",
            "train a multinomial logistic",
            "classifier head",
            "ensemble",
            "combine predictions",
            "combine with",
            "full training",
        )
    )
    return model_count > 0 and full_route_signal


def _sop_kind(level: str, text: str) -> str:
    if level == "L1_strategy":
        return "model_strategy"
    if level == "L3_repair":
        if any(token in text for token in ("path", "attribute", "defined", "merge", "dataloader", "dtype", "shape", "cuda", "oom")):
            return "infrastructure"
        return "debug_fix"
    if any(token in text for token in ("validation", "cross-validation", "k-fold", "early stopping", "calibrat")):
        return "validation_protocol"
    if any(token in text for token in ("feature", "tf-idf", "stylometric", "pooling", "augmentation", "n-gram")):
        return "feature"
    if any(token in text for token in ("optimizer", "loss", "scheduler", "unfreez", "dropout", "mixed precision", "gradscaler", "ema")):
        return "training_protocol"
    return "architecture"


def _method_family(text: str, task_families: list[str], level: str) -> str:
    if level != "L1_strategy":
        if "modernbert" in text:
            return "modernbert_family"
        if "deberta" in text:
            return "deberta_family"
        if "distilroberta" in text or "roberta" in text:
            return "roberta_family"
        if "distilbert" in text or re.search(r"\bbert\b", text):
            return "bert_family"
        if "efficientnet" in text or "convnext" in text or "resnet" in text:
            return "cnn_vision_family"
        if any(token in text for token in ("siglip", "dinov2", "vision transformer", "vit")):
            return "vision_transformer_family"
        if "lightgbm" in text or "xgboost" in text:
            return "boosted_tree_family"
        return "general"
    if "transformer" in text or any(name in text for name in ("bert", "deberta", "roberta")):
        if "frozen" in text and "xgboost" in text:
            return "frozen_transformer_tree"
        if any(name in text for name in ("xgboost", "lightgbm", "logistic regression", "tf-idf")):
            return "transformer_classical_hybrid"
        if any(name in text for name in ("multiple", "two models", "weighted ensemble", "average softmax", "different random seeds")):
            return "multi_transformer_ensemble"
        if any(name in text for name in ("stylometric", "handcrafted", "engineered features", "cross-attention", "gated fusion")):
            return "transformer_feature_fusion"
        for name in ("modernbert", "deberta", "roberta", "distilbert", "bert"):
            if name in text:
                return f"{name}_finetune"
        return "transformer_finetune"
    if "tf-idf" in text and "logistic regression" in text:
        return "tfidf_stylometry_linear"
    if "tf-idf" in text and "mlp" in text:
        return "tfidf_stylometry_mlp"
    if "efficientnet" in text:
        if "frozen" in text:
            return "frozen_cnn_feature_mlp"
        if "handcrafted" in text:
            return "cnn_handcrafted_hybrid"
        return "efficientnet_finetune"
    if any(name in text for name in ("vision transformer", "vit", "siglip", "dinov2")):
        return "vision_transformer_finetune"
    if any(name in text for name in ("unet", "u-net", "denois")):
        return "unet_image_restoration"
    if any(name in text for name in ("lightgbm", "xgboost", "gradient boost")):
        return "gradient_boosted_trees"
    if "ensemble" in text:
        return "heterogeneous_ensemble"
    family = task_families[0] if task_families else "general"
    return f"{family}_pipeline"


def _compute_profile(text: str) -> str:
    if any(token in text for token in ("multiple transformer", "several pretrained transformer", "two roberta", "two models")):
        return "multi_gpu_preferred"
    if any(token in text for token in ("large", "efficientnet-b4", "siglip2", "dinov2", "vit-large")):
        return "single_gpu_large"
    if any(token in text for token in ("transformer", "bert", "cnn", "efficientnet", "resnet", "vit", "unet", "u-net")):
        return "single_gpu_standard"
    if any(token in text for token in ("tf-idf", "xgboost", "lightgbm", "logistic regression")):
        return "cpu_light"
    return "cpu_or_single_gpu"


def classify_sop(sop: dict[str, Any]) -> dict[str, Any]:
    text = _text(sop)
    if _is_repair(sop, text):
        level = "L3_repair"
    elif _looks_like_strategy(sop, text):
        level = "L1_strategy"
    else:
        level = "L2_tactic"
    task_families = _task_families(sop)
    return {
        "abstraction_level": level,
        "sop_kind": _sop_kind(level, text),
        "method_family": _method_family(text, task_families, level),
        "task_families": task_families,
        "decision_stages": (
            ["draft", "evolution"]
            if level == "L1_strategy"
            else ["model_design", "improve"]
            if level == "L2_tactic"
            else ["debug", "repair"]
        ),
        "compute_profile": _compute_profile(text),
        "classification_source": "deterministic_rules_v1",
    }


def validate_entry(sop_id: str, entry: dict[str, Any]) -> None:
    if entry.get("abstraction_level") not in ABSTRACTION_LEVELS:
        raise ValueError(f"{sop_id}: invalid abstraction_level")
    if entry.get("sop_kind") not in SOP_KINDS:
        raise ValueError(f"{sop_id}: invalid sop_kind")
    if entry.get("compute_profile") not in COMPUTE_PROFILES:
        raise ValueError(f"{sop_id}: invalid compute_profile")
    if not re.fullmatch(r"[a-z0-9_]+", str(entry.get("method_family") or "")):
        raise ValueError(f"{sop_id}: invalid method_family")
    for key in ("task_families", "decision_stages"):
        if not isinstance(entry.get(key), list) or not entry[key]:
            raise ValueError(f"{sop_id}: {key} must be a non-empty list")


def build_taxonomy(graph_path: Path, overrides_path: Path) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    overrides_doc = json.loads(overrides_path.read_text(encoding="utf-8"))
    if overrides_doc.get("schema") != "runforest_sop_taxonomy_overrides_v1":
        raise ValueError("Unsupported SOP taxonomy override schema")
    overrides = overrides_doc.get("entries", {})
    reviewed_l1_ids = overrides_doc.get("reviewed_l1_ids")
    if not isinstance(reviewed_l1_ids, list):
        raise ValueError("SOP taxonomy overrides require a reviewed_l1_ids list")
    if len(reviewed_l1_ids) != len(set(reviewed_l1_ids)):
        raise ValueError("SOP taxonomy reviewed_l1_ids contains duplicates")
    sops = [node for node in graph.get("nodes", []) if node.get("type") == "SOP"]
    entries: dict[str, dict[str, Any]] = {}
    for sop in sorted(sops, key=lambda item: str(item.get("id"))):
        sop_id = str(sop.get("id") or "")
        if not sop_id:
            raise ValueError("SOP without id")
        entry = classify_sop(sop)
        if sop_id in overrides:
            entry.update(overrides[sop_id])
            entry["classification_source"] = "explicit_override_v1"
        validate_entry(sop_id, entry)
        entries[sop_id] = entry
    unknown_overrides = sorted(set(overrides) - set(entries))
    if unknown_overrides:
        raise ValueError(f"Overrides reference unknown SOP ids: {unknown_overrides}")
    actual_l1_ids = sorted(
        sop_id
        for sop_id, entry in entries.items()
        if entry["abstraction_level"] == "L1_strategy"
    )
    declared_l1_ids = sorted(str(value) for value in reviewed_l1_ids)
    if actual_l1_ids != declared_l1_ids:
        raise ValueError(
            "Manual L1 review coverage mismatch: "
            f"unreviewed={sorted(set(actual_l1_ids) - set(declared_l1_ids))} "
            f"no_longer_l1={sorted(set(declared_l1_ids) - set(actual_l1_ids))}"
        )
    for sop_id in actual_l1_ids:
        entries[sop_id]["manual_reviewed"] = True
        entries[sop_id]["manual_review_version"] = str(overrides_doc.get("version") or "v1")
    counts: dict[str, int] = {}
    for entry in entries.values():
        level = entry["abstraction_level"]
        counts[level] = counts.get(level, 0) + 1
    return {
        "schema": SCHEMA,
        "source_graph": graph_path.name,
        "source_graph_sha256": _sha256(graph_path),
        "classifier_version": "deterministic_rules_v1",
        "override_version": str(overrides_doc.get("version") or "v1"),
        "reviewed_l1_count": len(actual_l1_ids),
        "reviewed_l1_ids": actual_l1_ids,
        "sop_count": len(entries),
        "coverage": 1.0,
        "abstraction_counts": dict(sorted(counts.items())),
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    taxonomy = build_taxonomy(args.graph.resolve(), args.overrides.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(taxonomy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: taxonomy[key] for key in ("schema", "sop_count", "coverage", "abstraction_counts")}, indent=2))


if __name__ == "__main__":
    main()
