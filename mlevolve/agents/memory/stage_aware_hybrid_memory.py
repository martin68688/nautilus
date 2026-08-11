"""Stage-aware SOP gateway retrieval over the RunForest graph."""

from __future__ import annotations

import collections
import copy
import difflib
import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agents.memory.external_skill_memory import (
    RunForestMemoryLayer,
    _as_list,
    _tokenize,
    bounded_selector_max_tokens,
)
from agents.memory.atomic_claim_memory import (
    structured_debug_relevance,
    verified_atomic_debug_claim,
)
from agents.memory.sop_visibility_gateway import SOPVisibilityGateway
from authority.domain_scope import (
    DOMAIN_GENERAL,
    SAME_DOMAIN,
    canonical_domain,
    normalize_transfer_scope,
    transfer_is_compatible,
)
from authority.models import (
    GenerationStage,
    GovernanceStage,
    Operation,
    ProtocolRef,
    TaskContext,
    VisibilityRequest,
    VisibleSOPPack,
)
from authority.stage_ontology import resolve_stage_axes

logger = logging.getLogger("MLEvolve")

PACK_SCHEMA = "stage_hybrid_memory_pack_v1"
LAYERED_PACK_SCHEMA = "layered_strategy_memory_pack_v1"
RRF_K = 60

STAGE_QUOTAS = {
    "draft": {"sop_candidates": 6, "sop_gateways": 3, "tree_candidates": 2},
    "improve": {"sop_candidates": 4, "sop_gateways": 2, "tree_candidates": 6},
    "debug": {"sop_candidates": 2, "sop_gateways": 1, "tree_candidates": 8},
    "evolution": {"sop_candidates": 6, "sop_gateways": 3, "tree_candidates": 3},
    "fusion": {"sop_candidates": 4, "sop_gateways": 2, "tree_candidates": 4},
}

STAGE_RRF_WEIGHTS = {
    "draft": {"sop": 0.70, "tree": 0.30},
    "improve": {"sop": 0.40, "tree": 0.60},
    "debug": {"sop": 0.25, "tree": 0.75},
    "evolution": {"sop": 0.70, "tree": 0.30},
    "fusion": {"sop": 0.50, "tree": 0.50},
}

STAGE_ROUTE = {
    "draft": "sop_first",
    "improve": "tree_heavy",
    "debug": "tree_first",
    "evolution": "sop_first",
    "fusion": "balanced",
}

STAGE_DECISION_TAGS = {
    # Runtime stages may consume more specific taxonomy decisions. Fusion, for
    # example, reuses model-design/improvement SOPs because taxonomy has no
    # separate fusion label.
    "draft": {"draft"},
    "improve": {"improve", "model_design"},
    "debug": {"debug", "repair"},
    "evolution": {"evolution", "improve"},
    "fusion": {"model_design", "improve"},
}

STAGE_SOP_FIELD_WEIGHTS = {
    "draft": {"semantic": 0.60, "conditions": 0.25, "failures": 0.05, "evidence": 0.10},
    "improve": {"semantic": 0.50, "conditions": 0.25, "failures": 0.10, "evidence": 0.15},
    "debug": {"semantic": 0.35, "conditions": 0.15, "failures": 0.40, "evidence": 0.10},
    "evolution": {"semantic": 0.50, "conditions": 0.25, "failures": 0.10, "evidence": 0.15},
    "fusion": {"semantic": 0.50, "conditions": 0.20, "failures": 0.10, "evidence": 0.20},
}

SOP_HYBRID_SCORE_WEIGHTS = {
    "field_relevance": 0.40,
    "stage_fit": 0.20,
    "task_fit": 0.15,
    "geometry": 0.15,
    "clean_evidence": 0.10,
}

STAGE_ALIASES = {
    "multi_fusion": "fusion",
    "fusion_draft": "fusion",
    "aggregation": "fusion",
}

RETRIEVAL_CONTROLS = {
    "no_memory",
    "stage_hybrid",
    "flat_relevance_memory",
    "global_validity_bit",
    "authority_only",
    "full_decision_admissibility",
    "layered_strategy",
    "sop_only",
    "tree_only",
    "naive_concat",
    "flat_retrieval",
    "runforest_only",
    "static_hybrid",
    "dynamic_hybrid",
    "reversed_router",
}

FORMAL_FLAT_RELEVANCE_CONTROLS = {
    "flat_relevance_memory",
    "global_validity_bit",
    "authority_only",
}

STRATEGY_SCORE_WEIGHTS = {
    "task_fit": 0.30,
    "semantic_fit": 0.20,
    "clean_evidence": 0.25,
    "task_local_improvement": 0.15,
    "compute_fit": 0.10,
}

# These signatures describe failure mechanisms rather than task names or one
# benchmark's wording. Debug Tree retrieval uses them to compare the current
# failure with the parent failure of a previously successful repair.
FAILURE_SIGNATURES = {
    "fit_scope": {
        "fit scope", "fit_scope", "training fold only", "train fold only",
        "validation boundary", "fit on validation", "fit on holdout",
        "data leakage", "leakage", "preprocessing", "vectorizer", "scaler",
        "normalizer", "pca", "target encoding",
    },
    "split_scope": {
        "group split", "group leakage", "temporal split", "time split",
        "chronological", "duplicate leakage", "train test overlap",
        "train validation overlap", "holdout", "outer split",
    },
    "alignment": {
        "alignment", "misaligned", "shape mismatch", "dimension mismatch",
        "size mismatch", "index mismatch", "sample id", "sample_id",
        "broadcast", "tensor shape", "prediction shape", "length mismatch",
    },
    "resource": {
        "out of memory", "oom", "cuda memory", "shared memory", "bus error",
        "no space left", "resource exhausted", "timeout", "timed out",
    },
    "numerical": {
        "nan", "infinite", "overflow", "underflow", "numerical instability",
        "float16", "fp16", "division by zero", "not finite",
    },
    "dependency_api": {
        "importerror", "modulenotfounderror", "unexpected keyword",
        "not supported", "no attribute", "api", "version mismatch",
        "from_pretrained", "missing dependency",
    },
    "syntax_order": {
        "syntaxerror", "nameerror", "not defined", "before it is defined",
        "merge conflict", "script order", "indentationerror", "parse error",
    },
    "path_io": {
        "filenotfounderror", "file not found", "wrong path", "data path",
        "permission denied", "read csv", "missing file",
    },
    "evaluation_reuse": {
        "early stopping", "ensemble weight", "model selection", "oof",
        "out of fold", "validation reused", "selection bias",
    },
}

DEBUG_TREE_CONFIDENCE_THRESHOLD = 0.50
DEBUG_TREE_MAX_WEIGHT = 0.60
CAUSAL_ATTACHMENT_MIN_SCORE = 0.55

TASK_PROFILES = {
    "spooky-author-identification": ("text", "text_classification"),
    "aerial-cactus-identification": ("image", "image_binary_classification"),
    "denoising-dirty-documents": ("image", "image_restoration"),
    "leaf-classification": ("multimodal", "tabular_multiclass"),
    "random-acts-of-pizza": ("text", "text_binary_classification"),
    "mlsp-2013-birds": ("audio", "audio_multilabel_classification"),
    "nomad2018-predict-transparent-conductors": ("tabular", "tabular_multioutput_regression"),
    "new-york-city-taxi-fare-prediction": ("tabular", "tabular_regression"),
}

TASK_TYPES = {
    "spooky-author-identification": "nlp",
    "random-acts-of-pizza": "nlp",
    "aerial-cactus-identification": "vision",
    "denoising-dirty-documents": "vision",
    "leaf-classification": "multimodal",
    "mlsp-2013-birds": "audio",
    "nomad2018-predict-transparent-conductors": "tabular",
    "new-york-city-taxi-fare-prediction": "tabular",
}

L3_DYNAMIC_CONFIDENCE_WEIGHTS = {
    "task_match": 0.40,
    "failure_signature_match": 0.30,
    "runtime_stage_match": 0.12,
    "method_family_match": 0.08,
    "clean_transition_quality": 0.08,
    "successful_repair_frequency": 0.02,
}
L3_FAILURE_SIGNATURE_MIN_MATCH = 0.50
L3_SPECIFIC_SIGNATURE_MIN_OVERLAP = 1.0 / 3.0

# Atomic claims have already passed exact-task, local before/after execution,
# and claim-level taint gates. Their live rank should therefore be dominated
# by the current causal signature rather than by the broad task prior that was
# designed for whole-program RunForest transitions. A 0.30 floor admits
# concrete shape/path/NaN/runtime matches while still rejecting a NameError
# that shares only the exception class (0.27 at most).
ATOMIC_L3_FAILURE_SIGNATURE_MIN_MATCH = 0.30
ATOMIC_L3_DYNAMIC_CONFIDENCE_WEIGHTS = {
    "task_match": 0.10,
    "failure_signature_match": 0.65,
    "runtime_stage_match": 0.08,
    "method_family_match": 0.05,
    "clean_transition_quality": 0.10,
    "successful_repair_frequency": 0.02,
}

# Small deterministic concept groups bridge routine paraphrases before the
# LLM gateway selector sees the already task-gated candidates.  They do not
# create candidates, relax the 0.50 failure gate, or permit cross-task-type
# retrieval; they only normalize semantically equivalent runtime vocabulary.
L3_FAILURE_TOKEN_EQUIVALENCE_GROUPS = (
    {"mixup", "cutmix", "augmentation", "augmented"},
    {
        "scalar",
        "vector",
        "reduce",
        "reduction",
        "aggregate",
        "aggregated",
        "mean",
        "perexample",
        "persample",
    },
    {"backward", "backpropagate", "backpropagation", "autograd", "gradient"},
    {"classifier", "classification", "head"},
    {"dimension", "width", "rank", "shape", "spatial", "vector"},
    {"convolutional", "cnn", "backbone"},
)
L3_GENERIC_SIGNATURE_TERMS = frozenset(
    {
        "api",
        "contract",
        "error",
        "exception",
        "failure",
        "feature",
        "invalid",
        "mismatch",
        "model",
        "runtime",
        "shape",
        "training",
        "unsupported",
    }
)


def canonical_task_id(task_id: str) -> str:
    """Treat orchestration-only ``full-`` run names as the same benchmark."""
    value = str(task_id or "").strip()
    while value.startswith("full-"):
        value = value[len("full-") :]
    return value

FAMILY_CODE_SIGNATURES = {
    "convnext_finetune": [("convnext",)],
    "modernbert_finetune": [("modernbert",)],
    "deberta_finetune": [("deberta",)],
    "roberta_finetune": [("roberta",)],
    "bert_finetune": [("bert",)],
    "transformer_finetune": [("automodel", "transformer", "bert", "roberta", "deberta")],
    "deberta_multisample_focal_cv": [("deberta",), ("focal",), ("dropout",), ("stratifiedkfold", "kfold")],
    "deberta_xgb_lr_ensemble": [("deberta",), ("xgbclassifier", "xgboost"), ("logisticregression",)],
    "multi_frozen_transformer_xgboost": [("xgbclassifier", "xgboost"), ("automodel", "transformer")],
    "frozen_transformer_tree": [("xgbclassifier", "xgboost"), ("automodel", "transformer", "bert", "roberta", "deberta")],
    "multi_transformer_ensemble": [("automodel", "transformer", "bert", "roberta", "deberta"), ("ensemble", "average", "mean")],
    "transformer_engineered_feature_hybrid": [("automodel", "transformer", "deberta", "roberta", "bert"), ("feature", "stylometric", "tfidf")],
    "transformer_feature_fusion": [("automodel", "transformer", "deberta", "roberta", "bert"), ("feature", "stylometric", "attention", "gate")],
    "transformer_classical_hybrid": [("automodel", "transformer", "deberta", "roberta", "bert"), ("xgbclassifier", "lightgbm", "logisticregression", "tfidf")],
    "tfidf_stylometry_linear": [("tfidfvectorizer",), ("logisticregression",)],
    "tfidf_stylometry_mlp": [("tfidfvectorizer",), ("linear", "mlp")],
    "sentence_embedding_mlp": [("sentencetransformer", "sentencebert", "allmpnet"), ("mlp", "sequential", "linear")],
    "textcnn_stylometry": [("textcnn", "conv1d"), ("stylometric", "feature")],
    "stylometry_boosted_tree": [("xgbclassifier", "xgboost", "lightgbm", "lgbm"), ("stylometric", "feature")],
    "efficientnet_finetune": [("efficientnet",)],
    "multi_cnn_ensemble": [("efficientnet",), ("resnet",), ("ensemble", "average", "mean")],
    "frozen_cnn_feature_mlp": [("efficientnet", "resnet", "convnext"), ("linear", "mlp")],
    "cnn_handcrafted_hybrid": [("efficientnet", "resnet", "convnext", "cnn"), ("feature", "histogram", "hog", "lbp")],
    "vision_transformer_finetune": [("vit", "siglip", "dinov2", "visiontransformer")],
    "vision_tabular_fusion": [("vit", "siglip", "dinov2", "visiontransformer"), ("tabular",), ("fusion", "gate", "concat")],
    "heterogeneous_multimodal_ensemble": [("xgbclassifier", "xgboost"), ("logisticregression",), ("mlp", "sequential", "linear")],
    "gradient_boosted_trees": [("lightgbm", "lgbm", "xgbclassifier", "xgboost")],
    # Recipe-first families distilled for the End2End layered router.
    "efficientnet_b0_focal_finetune": [("efficientnet",), ("focal",)],
    "efficientnet_b2_cutmix_differential_lr": [("efficientnet",), ("cutmix",)],
    "convnext_tiny_cutmix_tta": [("convnext",), ("cutmix",), ("tta", "flip")],
    "resnet18_staged_mixup_tta": [("resnet18", "resnet"), ("mixup",)],
    "siglip2_frozen_fivefold_mlp": [("siglip",), ("stratifiedkfold", "kfold"), ("mlp", "sequential", "linear")],
    "compact_cnn_mixup": [("conv2d", "cnn"), ("mixup",)],
    "frozen_resnet_tabular_concat_mlp": [("resnet18", "resnet"), ("tabular", "margin"), ("mlp", "sequential", "linear")],
    "dinov2_bidirectional_cross_attention_fusion": [("dinov2", "dino"), ("multiheadattention", "crossattention", "attention"), ("tabular", "margin")],
    "siglip2_multibranch_self_attention_fusion": [("siglip2", "siglip"), ("multiheadattention", "selfattention", "attention"), ("tabular", "margin")],
    "tabular_residual_se_multibranch_mlp": [("residual", "resblock"), ("squeezeexcitation", "seblock"), ("margin",), ("shape",), ("texture",)],
    "foldwise_lightgbm_logistic_weighted_blend": [("lightgbm", "lgbm"), ("logisticregression",), ("stratifiedkfold", "kfold")],
    "tabular_feature_group_transformer": [("transformerencoder", "multiheadattention", "transformer"), ("margin",), ("shape",), ("texture",)],
    "patch_unet_charbonnier_gaussian_blend": [("unet",), ("charbonnier",), ("gaussian",)],
    "handcrafted_feature_unet_masked_mse": [("unet",), ("masked", "mask"), ("mse",)],
    "lightweight_residual_stochastic_encoder_decoder": [("encoder", "conv2d"), ("residual", "resblock")],
    "nafnet_overlapping_patch_denoiser": [("nafnet",), ("patch",)],
    "se_attention_unet_augmented_patches": [("unet",), ("attention", "seblock", "squeezeexcitation")],
    "temporal_rich_feature_lightgbm": [("lightgbm", "lgbm"), ("datetime", "hour", "weekday")],
    "fare_stratified_lightgbm": [("lightgbm", "lgbm"), ("stratifiedkfold", "kfold")],
    "cluster_zone_lightgbm": [("lightgbm", "lgbm"), ("kmeans", "cluster")],
    "dual_target_lightgbm_blend": [("lightgbm", "lgbm"), ("blend", "average", "mean")],
    "hash_split_regularized_mlp": [("mlp", "sequential", "linear"), ("hash", "md5", "sha256")],
    "geo_nn_xgboost_hybrid": [("xgb", "xgboost"), ("nearestneighbors", "knn")],
    "frozen_transformer_embedding_xgboost": [("automodel", "transformer", "bert"), ("xgb", "xgboost")],
    "deberta_large_ema_holdout": [("deberta",), ("ema", "exponentialmovingaverage")],
    "deberta_base_mean_pool_cv": [("deberta",), ("mean",), ("stratifiedkfold", "kfold")],
    "modernbert_large_raw_text_cv": [("modernbert",), ("stratifiedkfold", "kfold")],
    "roberta_stylometric_lightgbm_hybrid": [("roberta",), ("stylometric", "feature"), ("lightgbm", "lgbm")],
}


def strategy_alignment_for_code(strategy: dict[str, Any], code: str) -> dict[str, Any]:
    """Record whether generated code visibly implements its frozen L1 family."""
    family = str(strategy.get("method_family") or "")
    signatures = FAMILY_CODE_SIGNATURES.get(family, [])
    normalized = re.sub(r"[^a-z0-9]", "", str(code or "").lower())
    checks = []
    for alternatives in signatures:
        matched = next((token for token in alternatives if token in normalized), None)
        checks.append({"alternatives": list(alternatives), "matched": matched})
    if not signatures:
        status = "unverified_no_signature"
    elif all(item["matched"] for item in checks):
        status = "verified"
    elif any(item["matched"] for item in checks):
        status = "partial"
    else:
        status = "mismatch"
    return {
        "schema": "novel_strategy_code_alignment_v1",
        "method_family": family,
        "status": status,
        "checks": checks,
        "rank_eligible": status == "verified",
    }


def _plain_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    try:
        return {str(key): item for key, item in value.items()}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Stage-hybrid configuration must be a mapping") from exc


def _merge_quotas(overrides: Any) -> dict[str, dict[str, int]]:
    merged = {stage: values.copy() for stage, values in STAGE_QUOTAS.items()}
    for stage, raw in _plain_mapping(overrides).items():
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in merged:
            raise ValueError(f"Unknown stage quota: {stage}")
        for key, value in _plain_mapping(raw).items():
            if key not in merged[stage] or isinstance(value, bool) or int(value) <= 0:
                raise ValueError(f"Invalid stage quota {stage}.{key}={value!r}")
            merged[stage][key] = int(value)
    return merged


def _merge_weights(overrides: Any) -> dict[str, dict[str, float]]:
    merged = {stage: values.copy() for stage, values in STAGE_RRF_WEIGHTS.items()}
    for stage, raw in _plain_mapping(overrides).items():
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in merged:
            raise ValueError(f"Unknown RRF stage: {stage}")
        for key, value in _plain_mapping(raw).items():
            if key not in {"sop", "tree"} or isinstance(value, bool):
                raise ValueError(f"Invalid RRF weight {stage}.{key}={value!r}")
            number = float(value)
            if number < 0:
                raise ValueError(f"Invalid RRF weight {stage}.{key}={value!r}")
            merged[stage][key] = number
        if not math.isclose(sum(merged[stage].values()), 1.0, abs_tol=1e-6):
            raise ValueError(f"RRF weights for {stage} must sum to 1")
    return merged


def weighted_rrf(
    sop_ids: list[str],
    tree_ids: list[str],
    *,
    sop_weight: float,
    tree_weight: float,
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """Fuse two rankings over execution IDs with deterministic ties."""
    sop_rank = {node_id: rank for rank, node_id in enumerate(sop_ids, 1)}
    tree_rank = {node_id: rank for rank, node_id in enumerate(tree_ids, 1)}
    rows = []
    for node_id in sorted(set(sop_rank) | set(tree_rank)):
        score = 0.0
        if node_id in sop_rank:
            score += sop_weight / (k + sop_rank[node_id])
        if node_id in tree_rank:
            score += tree_weight / (k + tree_rank[node_id])
        rows.append(
            {
                "id": node_id,
                "rrf_score": score,
                "sop_rank": sop_rank.get(node_id),
                "tree_rank": tree_rank.get(node_id),
                "candidate_class": (
                    "sop_transition_matches" if node_id in sop_rank else "tree_only_candidates"
                ),
            }
        )
    return sorted(rows, key=lambda item: (-item["rrf_score"], item["id"]))


class StageAwareHybridMemoryLayer(RunForestMemoryLayer):
    """Opt-in hybrid layer; the existing RunForest layer remains unchanged."""

    def __init__(
        self,
        *args: Any,
        stage_quotas: Any = None,
        rrf_weights: Any = None,
        blocked_run_prefixes: list[str] | None = None,
        gateway_selector: Callable[..., dict[str, Any]] | None = None,
        strategy_selector: Callable[..., dict[str, Any]] | None = None,
        retrieval_control: str | None = None,
        excluded_run_ids: list[str] | None = None,
        visibility_gateway: SOPVisibilityGateway | None = None,
        visibility_mode: str | None = None,
        visibility_enforce_operations: list[str] | None = None,
        visibility_enforce_generation_stages: list[str] | None = None,
        visibility_enforce_governance_stages: list[str] | None = None,
        visibility_authority_engine: Any | None = None,
        visibility_active_protocol: ProtocolRef | str | None = None,
        visibility_policy_version: str | None = None,
        visibility_task_id: str | None = None,
        visibility_bundle_version: str | None = None,
        visibility_token_budget: int | None = None,
        prospective_audit_logger: Any | None = None,
        memory_snapshot: Any | None = None,
        end2end_memory_system: str | None = None,
        end2end_prompt_token_budget: int | None = None,
        end2end_candidate_pool_limit: int | None = None,
        experiment_r_enabled: bool | None = None,
        experiment_r_candidate_limit: int | None = None,
        experiment_r_top_k: int | None = None,
        experiment_r_prompt_token_budget: int | None = None,
        experiment_r_memory_pool_sha256: str | None = None,
        experiment_r_debug_confidence_threshold: float | None = None,
        experiment_r_agentic_retrieval_enabled: bool | None = None,
        experiment_r_agentic_query_fn: Callable[..., dict[str, Any]] | None = None,
        experiment_r_l3_agent_match_enabled: bool | None = None,
        experiment_r_l3_semantic_encode_fn: Callable[[list[str]], Any] | None = None,
        experiment_r_l3_semantic_model_id: str | None = None,
        recipe_sop_path: str | None = None,
        recipe_sop_file_sha256: str | None = None,
        recipe_sop_bundle_sha256: str | None = None,
        recipe_evidence_path: str | None = None,
        recipe_evidence_file_sha256: str | None = None,
        recipe_evidence_manifest_sha256: str | None = None,
        recipe_implementation_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._trace_local = threading.local()
        self.prospective_audit_logger = prospective_audit_logger
        cfg = kwargs.get("cfg")
        ext_cfg = getattr(cfg, "external_skill_memory", None) if cfg is not None else None
        if stage_quotas is None and ext_cfg is not None:
            stage_quotas = getattr(ext_cfg, "stage_quotas", None)
        if rrf_weights is None and ext_cfg is not None:
            rrf_weights = getattr(ext_cfg, "rrf_weights", None)
        authority_cfg = getattr(cfg, "evaluation_authority", None) if cfg is not None else None
        if visibility_mode is None and authority_cfg is not None:
            visibility_mode = getattr(authority_cfg, "mode", None)
        if visibility_policy_version is None and authority_cfg is not None:
            visibility_policy_version = getattr(authority_cfg, "policy_version", None)
        if visibility_enforce_operations is None and authority_cfg is not None:
            visibility_enforce_operations = list(
                getattr(authority_cfg, "enforce_operations", None) or []
            )
        if (
            visibility_enforce_generation_stages is None
            and authority_cfg is not None
        ):
            visibility_enforce_generation_stages = list(
                getattr(authority_cfg, "enforce_generation_stages", None) or []
            )
        if (
            visibility_enforce_governance_stages is None
            and authority_cfg is not None
        ):
            visibility_enforce_governance_stages = list(
                getattr(authority_cfg, "enforce_governance_stages", None) or []
            )
        if visibility_token_budget is None and ext_cfg is not None:
            visibility_token_budget = getattr(ext_cfg, "visibility_token_budget", None)
        if blocked_run_prefixes is None and ext_cfg is not None:
            configured_prefixes = list(getattr(ext_cfg, "blocked_run_prefixes", None) or [])
            if configured_prefixes:
                blocked_run_prefixes = configured_prefixes
        if retrieval_control is None and ext_cfg is not None:
            retrieval_control = getattr(ext_cfg, "retrieval_control", None)
        self.retrieval_control = str(retrieval_control or "stage_hybrid")
        if self.retrieval_control not in RETRIEVAL_CONTROLS:
            raise ValueError(f"Unsupported stage-hybrid retrieval_control: {self.retrieval_control}")
        if excluded_run_ids is None and ext_cfg is not None:
            excluded_run_ids = list(
                getattr(ext_cfg, "excluded_run_ids", None) or []
            )
        self.excluded_run_ids = {str(value) for value in (excluded_run_ids or [])}
        if experiment_r_enabled is None and ext_cfg is not None:
            experiment_r_enabled = getattr(ext_cfg, "experiment_r_enabled", False)
        self.experiment_r_enabled = bool(experiment_r_enabled)
        self.experiment_r_candidate_limit = int(
            experiment_r_candidate_limit
            if experiment_r_candidate_limit is not None
            else getattr(ext_cfg, "experiment_r_candidate_limit", 12)
            if ext_cfg is not None
            else 12
        )
        self.experiment_r_top_k = int(
            experiment_r_top_k
            if experiment_r_top_k is not None
            else getattr(ext_cfg, "experiment_r_top_k", 6)
            if ext_cfg is not None
            else 6
        )
        self.experiment_r_prompt_token_budget = int(
            experiment_r_prompt_token_budget
            if experiment_r_prompt_token_budget is not None
            else getattr(ext_cfg, "experiment_r_prompt_token_budget", 1536)
            if ext_cfg is not None
            else 1536
        )
        self.experiment_r_memory_pool_sha256 = str(
            experiment_r_memory_pool_sha256
            if experiment_r_memory_pool_sha256 is not None
            else getattr(ext_cfg, "experiment_r_memory_pool_sha256", "")
            if ext_cfg is not None
            else ""
        )
        self.experiment_r_debug_confidence_threshold = float(
            experiment_r_debug_confidence_threshold
            if experiment_r_debug_confidence_threshold is not None
            else getattr(ext_cfg, "experiment_r_debug_confidence_threshold", 0.50)
            if ext_cfg is not None
            else 0.50
        )
        self.experiment_r_agentic_retrieval_enabled = bool(
            experiment_r_agentic_retrieval_enabled
            if experiment_r_agentic_retrieval_enabled is not None
            else getattr(
                ext_cfg, "experiment_r_agentic_retrieval_enabled", False
            )
            if ext_cfg is not None
            else False
        )
        self.experiment_r_agentic_max_steps = int(
            getattr(ext_cfg, "experiment_r_agentic_max_steps", 4)
            if ext_cfg is not None
            else 4
        )
        self.experiment_r_agentic_per_step_top_k = int(
            getattr(ext_cfg, "experiment_r_agentic_per_step_top_k", 8)
            if ext_cfg is not None
            else 8
        )
        self.experiment_r_agentic_max_observed = int(
            getattr(ext_cfg, "experiment_r_agentic_max_observed", 48)
            if ext_cfg is not None
            else 48
        )
        self.experiment_r_agentic_temperature = float(
            getattr(ext_cfg, "experiment_r_agentic_temperature", 0.0)
            if ext_cfg is not None
            else 0.0
        )
        self.experiment_r_agentic_max_tokens = int(
            getattr(ext_cfg, "experiment_r_agentic_max_tokens", 1200)
            if ext_cfg is not None
            else 1200
        )
        self.experiment_r_l3_agent_match_enabled = bool(
            experiment_r_l3_agent_match_enabled
            if experiment_r_l3_agent_match_enabled is not None
            else getattr(ext_cfg, "experiment_r_l3_agent_match_enabled", False)
            if ext_cfg is not None
            else False
        )
        self.experiment_r_l3_agent_match_candidate_limit = int(
            getattr(ext_cfg, "experiment_r_l3_agent_match_candidate_limit", 8)
            if ext_cfg is not None
            else 8
        )
        self.experiment_r_l3_semantic_shortlist_enabled = bool(
            getattr(ext_cfg, "experiment_r_l3_semantic_shortlist_enabled", False)
            if ext_cfg is not None
            else False
        )
        self._experiment_r_l3_semantic_encode_fn = (
            experiment_r_l3_semantic_encode_fn
        )
        agent_cfg = getattr(cfg, "agent", None) if cfg is not None else None
        self._experiment_r_l3_semantic_model_path = str(
            experiment_r_l3_semantic_model_id
            or getattr(agent_cfg, "memory_embedding_model_path", "")
            or ""
        )
        self._experiment_r_l3_semantic_device = str(
            getattr(agent_cfg, "memory_embedding_device", "cpu") or "cpu"
        )
        self._experiment_r_l3_semantic_model = None
        self.experiment_r_l3_semantic_model_id = (
            self._experiment_r_l3_semantic_model_path
        )
        self._experiment_r_l3_semantic_lock = threading.RLock()
        if (
            self._experiment_r_l3_semantic_encode_fn is None
            and self.experiment_r_l3_semantic_shortlist_enabled
            and self._experiment_r_l3_semantic_model_path
        ):
            self._experiment_r_l3_semantic_encode_fn = (
                self._encode_l3_semantic_texts
            )
        self.experiment_r_l3_agent_match_max_attempts = int(
            getattr(ext_cfg, "experiment_r_l3_agent_match_max_attempts", 2)
            if ext_cfg is not None
            else 2
        )
        self.experiment_r_l3_agent_match_min_confidence = float(
            getattr(ext_cfg, "experiment_r_l3_agent_match_min_confidence", 0.50)
            if ext_cfg is not None
            else 0.50
        )
        self.experiment_r_l3_agent_match_max_tokens = int(
            getattr(ext_cfg, "experiment_r_l3_agent_match_max_tokens", 1800)
            if ext_cfg is not None
            else 1800
        )
        self.experiment_r_flexible_selection_enabled = bool(
            getattr(ext_cfg, "experiment_r_flexible_selection_enabled", False)
            if ext_cfg is not None
            else False
        )
        configured_selection_caps = (
            getattr(ext_cfg, "experiment_r_stage_selection_caps", None)
            if ext_cfg is not None
            else None
        )
        default_selection_caps = {
            "draft": self.experiment_r_top_k,
            "improve": self.experiment_r_top_k,
            "debug": self.experiment_r_top_k,
        }
        if configured_selection_caps is not None:
            default_selection_caps.update(
                {
                    stage: int(getattr(configured_selection_caps, stage))
                    for stage in ("draft", "improve", "debug")
                    if getattr(configured_selection_caps, stage, None) is not None
                }
            )
        self.experiment_r_stage_selection_caps = default_selection_caps
        self.experiment_r_allow_agent_abstention = bool(
            getattr(ext_cfg, "experiment_r_allow_agent_abstention", False)
            if ext_cfg is not None
            else False
        )
        self.experiment_r_debug_causal_only = bool(
            getattr(ext_cfg, "experiment_r_debug_causal_only", False)
            if ext_cfg is not None
            else False
        )
        self.experiment_r_debug_tiered_retrieval_enabled = bool(
            getattr(ext_cfg, "experiment_r_debug_tiered_retrieval_enabled", False)
            if ext_cfg is not None
            else False
        )
        self.experiment_r_debug_portable_runtime_enabled = bool(
            getattr(ext_cfg, "experiment_r_debug_portable_runtime_enabled", False)
            if ext_cfg is not None
            else False
        )
        self.experiment_r_debug_portable_max_candidates = int(
            getattr(ext_cfg, "experiment_r_debug_portable_max_candidates", 2)
            if ext_cfg is not None
            else 2
        )
        configured_pin_stages = (
            getattr(ext_cfg, "experiment_r_same_task_best_pin_stages", None)
            if ext_cfg is not None
            else None
        )
        self.experiment_r_same_task_best_pin_stages = {
            str(stage)
            for stage in (
                configured_pin_stages
                if configured_pin_stages is not None
                else ("draft", "improve", "debug")
            )
        }
        self.experiment_r_atomic_actuation_enabled = bool(
            getattr(ext_cfg, "experiment_r_atomic_actuation_enabled", False)
            if ext_cfg is not None
            else False
        )
        self.experiment_r_improve_max_modules = int(
            getattr(ext_cfg, "experiment_r_improve_max_modules", 2)
            if ext_cfg is not None
            else 2
        )
        self.experiment_r_improve_max_patches = int(
            getattr(ext_cfg, "experiment_r_improve_max_patches", 6)
            if ext_cfg is not None
            else 6
        )
        self.experiment_r_debug_max_patches = int(
            getattr(ext_cfg, "experiment_r_debug_max_patches", 3)
            if ext_cfg is not None
            else 3
        )
        self._experiment_r_agentic_query_fn = experiment_r_agentic_query_fn
        if self.experiment_r_enabled:
            from agents.memory.experiment_r_router import ONLINE_CONTROLS

            if self.retrieval_control not in ONLINE_CONTROLS:
                raise ValueError(
                    "Experiment R requires one frozen online routing control; "
                    f"got {self.retrieval_control}"
                )
            if self.experiment_r_candidate_limit < self.experiment_r_top_k:
                raise ValueError("Experiment R candidate limit must be >= Top-K")
            if self.experiment_r_top_k != 6:
                raise ValueError("Experiment R v1 requires the frozen Top-K of 6")
            if self.experiment_r_prompt_token_budget <= 0:
                raise ValueError("Experiment R prompt token budget must be positive")
            if self.experiment_r_agentic_max_steps not in range(1, 9):
                raise ValueError("Agentic retrieval max steps must be in [1, 8]")
            if self.experiment_r_agentic_per_step_top_k not in range(1, 13):
                raise ValueError("Agentic retrieval per-step Top-K must be in [1, 12]")
            if self.experiment_r_agentic_max_observed < self.experiment_r_top_k:
                raise ValueError("Agentic retrieval observation budget is too small")
            if self.experiment_r_l3_agent_match_max_attempts not in range(1, 4):
                raise ValueError("L3 Agent match attempts must be in [1, 3]")
            if self.experiment_r_l3_agent_match_candidate_limit not in range(1, 33):
                raise ValueError("L3 Agent per-route candidate limit must be in [1, 32]")
            if not 0.0 <= self.experiment_r_l3_agent_match_min_confidence <= 1.0:
                raise ValueError("L3 Agent match confidence must be in [0, 1]")
            if self.experiment_r_l3_agent_match_max_tokens not in range(800, 4001):
                raise ValueError("L3 Agent match token budget must be in [800, 4000]")
            for stage, cap in self.experiment_r_stage_selection_caps.items():
                if stage not in {"draft", "improve", "debug"}:
                    raise ValueError(f"Unknown Experiment R selection-cap stage: {stage}")
                if cap not in range(0, self.experiment_r_top_k + 1):
                    raise ValueError(
                        "Experiment R stage selection caps must be between 0 and Top-K"
                    )
            if not self.experiment_r_same_task_best_pin_stages <= {
                "draft",
                "improve",
                "debug",
            }:
                raise ValueError("Experiment R same-task pin stages are invalid")
            if self.experiment_r_improve_max_modules not in range(1, 4):
                raise ValueError("Experiment R Improve module cap must be in [1, 3]")
            if self.experiment_r_improve_max_patches not in range(1, 21):
                raise ValueError("Experiment R Improve patch cap must be in [1, 20]")
            if self.experiment_r_debug_max_patches not in range(1, 11):
                raise ValueError("Experiment R Debug patch cap must be in [1, 10]")
            if self.experiment_r_debug_portable_max_candidates not in range(0, 5):
                raise ValueError(
                    "Experiment R portable Debug candidate cap must be in [0, 4]"
                )
        self.recipe_sop_path = str(
            recipe_sop_path
            if recipe_sop_path is not None
            else getattr(ext_cfg, "recipe_sop_path", "")
            if ext_cfg is not None
            else ""
        ).strip()
        self.recipe_sop_file_sha256 = str(
            recipe_sop_file_sha256
            if recipe_sop_file_sha256 is not None
            else getattr(ext_cfg, "recipe_sop_file_sha256", "")
            if ext_cfg is not None
            else ""
        ).strip()
        self.recipe_sop_bundle_sha256 = str(
            recipe_sop_bundle_sha256
            if recipe_sop_bundle_sha256 is not None
            else getattr(ext_cfg, "recipe_sop_bundle_sha256", "")
            if ext_cfg is not None
            else ""
        ).strip()
        self.recipe_evidence_path = str(
            recipe_evidence_path
            if recipe_evidence_path is not None
            else getattr(ext_cfg, "recipe_evidence_path", "")
            if ext_cfg is not None
            else ""
        ).strip()
        self.recipe_evidence_file_sha256 = str(
            recipe_evidence_file_sha256
            if recipe_evidence_file_sha256 is not None
            else getattr(ext_cfg, "recipe_evidence_file_sha256", "")
            if ext_cfg is not None
            else ""
        ).strip()
        self.recipe_evidence_manifest_sha256 = str(
            recipe_evidence_manifest_sha256
            if recipe_evidence_manifest_sha256 is not None
            else getattr(ext_cfg, "recipe_evidence_manifest_sha256", "")
            if ext_cfg is not None
            else ""
        ).strip()
        self.recipe_implementation_path = str(
            recipe_implementation_path
            if recipe_implementation_path is not None
            else getattr(ext_cfg, "recipe_implementation_path", "")
            if ext_cfg is not None
            else ""
        ).strip()
        if end2end_memory_system is None and ext_cfg is not None:
            end2end_memory_system = getattr(
                ext_cfg, "end2end_memory_system", ""
            )
        self.end2end_memory_system = str(
            end2end_memory_system or ""
        ).strip()
        if end2end_prompt_token_budget is None and ext_cfg is not None:
            end2end_prompt_token_budget = getattr(
                ext_cfg, "end2end_prompt_token_budget", 1536
            )
        self.end2end_prompt_token_budget = int(
            end2end_prompt_token_budget
            if end2end_prompt_token_budget is not None
            else 1536
        )
        if end2end_candidate_pool_limit is None and ext_cfg is not None:
            end2end_candidate_pool_limit = getattr(
                ext_cfg, "end2end_candidate_pool_limit", 12
            )
        self.end2end_candidate_pool_limit = int(
            end2end_candidate_pool_limit
            if end2end_candidate_pool_limit is not None
            else 12
        )
        self.end2end_controller = None
        if self.end2end_memory_system:
            from agents.memory.end2end_memory_system import (
                EndToEndMemoryController,
            )

            self.end2end_controller = EndToEndMemoryController(
                self.end2end_memory_system
            )
            if self.end2end_prompt_token_budget != 1536:
                raise ValueError(
                    "Experiment End2End requires a 1536-whitespace-token prompt budget"
                )
            if self.end2end_candidate_pool_limit != 12:
                raise ValueError(
                    "Experiment End2End requires 12 candidates per source"
                )
        self.stage_quotas = _merge_quotas(stage_quotas)
        self.rrf_weights = _merge_weights(rrf_weights)
        self._injected_gateway_selector = gateway_selector
        self._injected_strategy_selector = strategy_selector
        self._blocked_run_prefixes_override = blocked_run_prefixes
        self.strategy_candidate_limit = int(
            getattr(ext_cfg, "strategy_candidate_limit", 12) if ext_cfg is not None else 12
        )
        self.strategy_route_count = int(
            getattr(ext_cfg, "strategy_route_count", 3) if ext_cfg is not None else 3
        )
        self.l2_tactic_limit = int(
            getattr(ext_cfg, "l2_tactic_limit", 4) if ext_cfg is not None else 4
        )
        if self.strategy_route_count != 3:
            raise ValueError("Layered Novel Draft requires exactly three L1 strategy routes")
        if self.strategy_candidate_limit < self.strategy_route_count:
            raise ValueError("strategy_candidate_limit must be >= strategy_route_count")
        if self.l2_tactic_limit <= 0:
            raise ValueError("l2_tactic_limit must be positive")
        super().__init__(*args, memory_snapshot=memory_snapshot, **kwargs)
        if self.mode != "run_forest_stage_hybrid":
            raise ValueError("StageAwareHybridMemoryLayer requires mode=run_forest_stage_hybrid")
        if self.end2end_controller is not None and self.top_k != 6:
            raise ValueError("Experiment End2End requires top_k=6")
        self.domain_scope_required = (
            (self.graph.get("meta") or {}).get("domain_scope_required") is True
        )
        self.memory_snapshot = memory_snapshot
        self._overlay_clause_ids: set[str] = set()
        if self.memory_snapshot is not None:
            self.memory_snapshot.assert_unchanged()
            expected_graph = (
                self.memory_snapshot.base_bundle.path
                / "runforest"
                / "graph.json"
            ).resolve()
            if self.graph_path.resolve() != expected_graph:
                raise ValueError(
                    "MemorySnapshot Base Bundle does not match the loaded RunForest graph"
                )
            self._load_session_overlay_clauses()
        self._legacy_sop_ids = list(self._sops)
        self._recipe_evidence_ids: list[str] = []
        self._recipe_repair_evidence_by_transition: dict[str, dict[str, Any]] = {}
        self.recipe_evidence_receipt: dict[str, Any] = {}
        if self.recipe_evidence_path:
            self._load_recipe_evidence_overlay()
        self.recipe_implementation_receipt: dict[str, Any] = {}
        if self.recipe_implementation_path:
            self._load_recipe_implementation_capsules()
        self._recipe_sop_ids: list[str] = []
        self.recipe_sop_receipt: dict[str, Any] = {}
        if self.recipe_sop_path:
            self._load_recipe_sop_overlay()
        self.base_clause_receipt: dict[str, Any] = {
            "schema": "base_sop_clause_runtime_load_receipt_v1",
            "status": "not_bound",
            "clause_count": 0,
            "sop_count": 0,
        }
        if self.memory_snapshot is not None:
            self._load_base_bundle_clauses()
        self.visibility_mode = str(visibility_mode or "shadow").lower()
        self.visibility_authority_engine = visibility_authority_engine
        self.visibility_active_protocol = self._coerce_protocol_ref(
            visibility_active_protocol
        )
        self.visibility_policy_version = str(
            visibility_policy_version or "authority_v1"
        )
        self.visibility_task_id = str(visibility_task_id or "")
        self.visibility_bundle_version = str(
            visibility_bundle_version
            or (self.graph.get("meta") or {}).get("bundle_version")
            or (self.graph.get("meta") or {}).get("bundle_id")
            or (self.graph.get("meta") or {}).get("schema")
            or "legacy-unversioned"
        )
        self.visibility_token_budget = max(
            0, int(visibility_token_budget if visibility_token_budget is not None else 4096)
        )
        decision_lookup = None
        if visibility_authority_engine is not None:
            decision_lookup = getattr(visibility_authority_engine, "decisions", {}).get
        self.visibility_gateway = visibility_gateway or SOPVisibilityGateway(
            self.nodes,
            mode=self.visibility_mode,
            authority_engine=visibility_authority_engine,
            decision_lookup=decision_lookup,
            retrieval_profile=(
                self.retrieval_control
                if self.retrieval_control
                in {
                    "flat_relevance_memory",
                    "global_validity_bit",
                    "authority_only",
                    "full_decision_admissibility",
                }
                else "full_decision_admissibility"
            ),
            enforce_operations=visibility_enforce_operations or [],
            enforce_generation_stages=(
                visibility_enforce_generation_stages or []
            ),
            enforce_governance_stages=(
                visibility_enforce_governance_stages or []
            ),
        )
        if visibility_gateway is not None:
            self.visibility_mode = visibility_gateway.mode
        self._build_sop_reverse_index()
        if self.retrieval_control == "layered_strategy":
            self._validate_layered_taxonomy()

    def _load_session_overlay_clauses(self) -> None:
        """Materialize append-only overlay clauses for online Authority gating.

        No overlay event is pre-authorized here. The visibility gateway sees
        the clauses, but every non-Inspect use is still evaluated against the
        live Authority Engine before ranking or prompt materialization.
        """

        by_sop: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for event in self.memory_snapshot.session_overlay.events():
            if event.event_type != "sop_clause":
                continue
            raw = event.payload.get("clause")
            if not isinstance(raw, dict):
                continue
            clause = copy.deepcopy(raw)
            clause_id = str(clause.get("clause_id") or "")
            sop_id = str(clause.get("sop_id") or "")
            if not clause_id or not sop_id:
                raise ValueError("Session Overlay SOP clause lacks stable IDs")
            if clause_id in self.nodes:
                raise ValueError(f"Session Overlay clause collides with Base: {clause_id}")
            clause.update(
                {
                    "id": clause_id,
                    "type": "SOPClause",
                    "origin": "session_overlay",
                    "overlay_event_id": event.event_id,
                }
            )
            self.nodes[clause_id] = clause
            self._overlay_clause_ids.add(clause_id)
            self._node_tokens[clause_id] = _tokenize(self._node_text(clause))
            by_sop[sop_id].append(clause)
        for sop_id, clauses in sorted(by_sop.items()):
            existing = self.nodes.get(sop_id)
            if existing is not None and existing.get("type") != "SOP":
                raise ValueError(f"Session Overlay SOP collides with Base node: {sop_id}")
            if existing is None:
                existing = {
                    "id": sop_id,
                    "type": "SOP",
                    "sop_id": sop_id,
                    "title": "Session Overlay experience",
                    "action": "",
                    "applies_when": [],
                    "prevents": [],
                    "origin": "session_overlay",
                    "clauses": [],
                }
                self.nodes[sop_id] = existing
                self._sops.append(sop_id)
            raw_clauses = list(existing.get("clauses") or [])
            raw_clauses.extend(copy.deepcopy(clauses))
            existing["clauses"] = raw_clauses
            self._node_tokens[sop_id] = _tokenize(self._node_text(existing))
        self._sops = sorted(set(self._sops))

    def _load_base_bundle_clauses(self) -> None:
        """Materialize manifest-bound Base clauses before Gateway creation.

        Recipe SOPs are runtime overlays rather than RunForest graph nodes, so
        their formal clauses cannot be discovered by ``SOPVisibilityGateway``
        until both layers have been loaded.  This method joins the immutable
        Bundle publication to the already verified Recipe containers without
        mutating either artifact on disk.
        """

        base = self.memory_snapshot.base_bundle
        artifact_hashes = {
            str(key): str(value)
            for key, value in (
                (getattr(base, "manifest", {}) or {}).get("artifact_hashes")
                or {}
            ).items()
        }
        clause_artifact = "sop/clauses.jsonl"
        mask_artifact = (
            "visibility/precompiled_masks/declared_scope_masks.json"
        )
        has_clauses = clause_artifact in artifact_hashes
        has_masks = mask_artifact in artifact_hashes
        if not has_clauses and not has_masks:
            self.base_clause_receipt = {
                "schema": "base_sop_clause_runtime_load_receipt_v1",
                "status": "legacy_bundle_without_formal_clauses",
                "clause_count": 0,
                "sop_count": 0,
            }
            return
        if has_clauses != has_masks:
            raise ValueError(
                "Base Bundle must publish SOP clauses and declared-scope masks together"
            )

        rows = base.read_jsonl(clause_artifact)
        if not rows:
            raise ValueError("Base Bundle formal SOP clause artifact is empty")
        seen_clause_ids: set[str] = set()
        clause_ids_by_sop: dict[str, list[str]] = collections.defaultdict(list)
        for raw in rows:
            clause = copy.deepcopy(raw)
            clause_id = str(clause.get("clause_id") or "").strip()
            sop_id = str(clause.get("sop_id") or "").strip()
            if not clause_id or not sop_id:
                raise ValueError("Base Bundle SOP clause lacks stable IDs")
            if clause_id in seen_clause_ids:
                raise ValueError(f"Duplicate Base Bundle SOP clause: {clause_id}")
            seen_clause_ids.add(clause_id)
            if clause_id in self.nodes:
                raise ValueError(
                    f"Base Bundle SOP clause collides with runtime node: {clause_id}"
                )
            container = self.nodes.get(sop_id)
            if container is None or container.get("type") != "SOP":
                raise ValueError(
                    f"Base Bundle SOP clause references missing container: {sop_id}"
                )
            if not clause.get("protocol_scope"):
                raise ValueError(
                    f"Base Bundle SOP clause lacks protocol scope: {clause_id}"
                )
            if not clause.get("permitted_operations"):
                raise ValueError(
                    f"Base Bundle SOP clause lacks operation scope: {clause_id}"
                )
            clause.update(
                {
                    "id": clause_id,
                    "type": "SOPClause",
                    "origin": "immutable_base_bundle",
                    "base_bundle_id": base.bundle_id,
                    "base_bundle_version": base.bundle_version,
                }
            )
            self.nodes[clause_id] = clause
            self._node_tokens[clause_id] = _tokenize(self._node_text(clause))
            clause_ids_by_sop[sop_id].append(clause_id)

        for sop_id, clause_ids in sorted(clause_ids_by_sop.items()):
            container = self.nodes[sop_id]
            existing_ids = [
                str(value) for value in container.get("clause_ids") or []
            ]
            if existing_ids:
                raise ValueError(
                    f"Runtime SOP already declares formal clauses: {sop_id}"
                )
            container["clause_ids"] = sorted(clause_ids)
            self._node_tokens[sop_id] = _tokenize(self._node_text(container))

        self.base_clause_receipt = {
            "schema": "base_sop_clause_runtime_load_receipt_v1",
            "status": "loaded",
            "base_bundle_id": base.bundle_id,
            "base_bundle_version": base.bundle_version,
            "clause_count": len(seen_clause_ids),
            "sop_count": len(clause_ids_by_sop),
            "clause_artifact_sha256": artifact_hashes[clause_artifact],
            "mask_artifact_sha256": artifact_hashes[mask_artifact],
        }

    def _encode_l3_semantic_texts(self, texts: list[str]) -> Any:
        """Lazily load the configured encoder when Global Memory is disabled."""

        if self._experiment_r_l3_semantic_model is None:
            from agents.memory.embedding_models import EmbeddingModel

            self._experiment_r_l3_semantic_model = EmbeddingModel(
                model_type="local",
                model_name=self._experiment_r_l3_semantic_model_path,
                device=self._experiment_r_l3_semantic_device,
            )
        return self._experiment_r_l3_semantic_model.encode(
            texts, show_progress_bar=False
        )

    @staticmethod
    def _coerce_protocol_ref(value: ProtocolRef | str | None) -> ProtocolRef:
        if isinstance(value, ProtocolRef):
            return value
        raw = str(value or "")
        key, separator, digest = raw.partition("#")
        protocol_id, at, version = key.partition("@")
        if separator and at and protocol_id and version and digest:
            return ProtocolRef(protocol_id, version, digest)
        return ProtocolRef("unbound", "0", "")

    @staticmethod
    def _default_visibility_operation(stage: str) -> Operation:
        return (
            Operation.DEBUG_HYPOTHESIS
            if STAGE_ALIASES.get(stage, stage) == "debug"
            else Operation.GENERATE_CANDIDATE
        )

    def _visibility_request(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        operation: Operation | str | None = None,
        active_protocol: ProtocolRef | str | None = None,
        governance_stage: GovernanceStage | str = GovernanceStage.RETRIEVAL,
        requesting_component: str = "agents.memory.stage_aware_hybrid_memory",
    ) -> VisibilityRequest:
        axes = resolve_stage_axes(
            runtime_stage=stage,
            governance_stage=governance_stage,
        )
        return VisibilityRequest(
            operation=operation or self._default_visibility_operation(stage),
            generation_stage=axes.generation_stage,
            governance_stage=axes.governance_stage,
            active_protocol=self._coerce_protocol_ref(active_protocol)
            if active_protocol is not None
            else self.visibility_active_protocol,
            task_context=TaskContext(
                task_id=str(task_id or self.visibility_task_id),
                task_family=self._task_family_for_query(task_id, task_desc),
            ),
            memory_bundle_version=self.visibility_bundle_version,
            token_budget=self.visibility_token_budget,
            requesting_component=requesting_component,
            authority_policy_version=self.visibility_policy_version,
        )

    def _prepare_visibility(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        request: VisibilityRequest | None = None,
        operation: Operation | str | None = None,
        active_protocol: ProtocolRef | str | None = None,
    ) -> VisibleSOPPack:
        request = request or self._visibility_request(
            stage=stage,
            task_id=task_id,
            task_desc=task_desc,
            operation=operation,
            active_protocol=active_protocol,
        )
        pack = self.visibility_gateway.evaluate(
            request,
            candidate_sop_ids=self._sops,
            candidate_clause_ids=self._snapshot_candidate_clause_ids(request),
        )
        self._trace_local.visibility_pack = pack
        if self.prospective_audit_logger is not None:
            self.prospective_audit_logger.record_visibility(
                pack,
                self.visibility_gateway,
                source="run_forest_sop_visibility",
            )
        return pack

    def _snapshot_candidate_clause_ids(
        self, request: VisibilityRequest
    ) -> set[str] | None:
        if (
            self.memory_snapshot is None
            or not self.visibility_gateway.should_enforce(request)
        ):
            return None
        base_clauses = self.memory_snapshot.base_clauses(
            request.operation,
            task_id=request.task_context.task_id,
            task_family=request.task_context.task_family,
            generation_stage=request.generation_stage.value,
            governance_stage=request.governance_stage.value,
        )
        return {
            str(clause["clause_id"])
            for clause in base_clauses
            if clause.get("clause_id")
        } | set(self._overlay_clause_ids)

    def _visibility_is_enforced(self) -> bool:
        pack = getattr(self._trace_local, "visibility_pack", None)
        if pack is None:
            return self.visibility_mode == "enforce"
        return bool(
            pack.visibility_trace.get(
                "request_enforced", self.visibility_mode == "enforce"
            )
        )

    def _effective_visibility_sop_ids(self) -> set[str] | None:
        if not self._visibility_is_enforced():
            return None
        pack = getattr(self._trace_local, "visibility_pack", None)
        return set(pack.effective_sop_ids) if pack is not None else set()

    def _visibility_projection(self, sop_id: str) -> dict[str, Any] | None:
        pack = getattr(self._trace_local, "visibility_pack", None)
        if pack is None:
            return None
        return pack.rendered_by_sop.get(sop_id)

    def _visible_sop_text_parts(
        self, sop_id: str, node: dict[str, Any]
    ) -> dict[str, str]:
        if not self._visibility_is_enforced():
            return self._sop_text_parts(node)
        projection = self._visibility_projection(sop_id)
        text = str((projection or {}).get("retrieval_text") or "")
        return {
            "semantic": text,
            "conditions": "",
            "failures": "",
            "evidence": "",
        }

    def _visible_sop_prompt(self, sop_id: str) -> str:
        if not self._visibility_is_enforced():
            node = self.nodes.get(sop_id, {})
            return "\n".join(
                value
                for value in (
                    str(node.get("title") or ""),
                    str(node.get("action") or ""),
                )
                if value
            )
        projection = self._visibility_projection(sop_id)
        return str((projection or {}).get("prompt_text") or "")

    def _container_embedding_visibility_safe(
        self,
        sop_id: str,
        projection: dict[str, Any] | None,
    ) -> bool:
        projection = projection or {}
        node = self.nodes[sop_id]
        declared_hash = str(
            node.get("visibility_safe_container_embedding_hash") or ""
        )
        expected_hash = hashlib.sha256(
            str(projection.get("retrieval_text") or "").encode("utf-8")
        ).hexdigest()
        return bool(
            projection.get("clause_ids")
            and node.get("visibility_safe_container_embedding") is True
            and declared_hash
            and hmac.compare_digest(declared_hash, expected_hash)
        )

    def _build_sop_reverse_index(self) -> None:
        self._navigation_transitions_by_sop: dict[str, list[str]] = collections.defaultdict(list)
        self._authorized_transitions_by_sop: dict[str, list[str]] = collections.defaultdict(list)
        self._navigation_sops_by_execution: dict[str, list[str]] = collections.defaultdict(list)
        self._authorized_sops_by_execution: dict[str, list[str]] = collections.defaultdict(list)
        self._navigation_sop_links_by_execution: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
        self._authorized_sop_links_by_execution: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
        self._navigation_sops_by_transition: dict[str, list[str]] = collections.defaultdict(list)
        self._authorized_sops_by_transition: dict[str, list[str]] = collections.defaultdict(list)
        self._sop_edge_metadata: dict[tuple[str, str], dict[str, Any]] = {}
        edge_outcomes: collections.Counter[str] = collections.Counter()

        def register(
            transition_id: str,
            sop_id: str,
            transitions_by_sop: dict[str, list[str]],
            sops_by_execution: dict[str, list[str]],
            links_by_execution: dict[str, dict[str, list[str]]],
            sops_by_transition: dict[str, list[str]],
        ) -> None:
            if transition_id not in transitions_by_sop[sop_id]:
                transitions_by_sop[sop_id].append(transition_id)
            if sop_id not in sops_by_transition[transition_id]:
                sops_by_transition[transition_id].append(sop_id)
            transition = self.nodes[transition_id]
            for execution_id in (
                transition_id,
                str(transition.get("parent_node_id") or ""),
                str(transition.get("child_node_id") or ""),
            ):
                if not execution_id:
                    continue
                if sop_id not in sops_by_execution[execution_id]:
                    sops_by_execution[execution_id].append(sop_id)
                links = links_by_execution[execution_id].setdefault(sop_id, [])
                if transition_id not in links:
                    links.append(transition_id)

        for edge in self.graph.get("edges", []):
            kind = str(edge.get("kind") or edge.get("type"))
            if kind not in {
                "distills_to",
                "navigation_attached_to",
                "authorized_distills_to",
            }:
                continue
            transition_id = str(edge.get("src", ""))
            sop_id = str(edge.get("dst", ""))
            if self.nodes.get(transition_id, {}).get("type") != "Transition":
                continue
            if self.nodes.get(sop_id, {}).get("type") != "SOP":
                continue
            self._sop_edge_metadata[(transition_id, sop_id)] = dict(edge)
            outcome = str(edge.get("authority_outcome") or "legacy_missing")
            edge_outcomes[outcome] += 1
            register(
                transition_id,
                sop_id,
                self._navigation_transitions_by_sop,
                self._navigation_sops_by_execution,
                self._navigation_sop_links_by_execution,
                self._navigation_sops_by_transition,
            )
            # A legacy ``distills_to`` edge is navigation-only even when it
            # carries an old allow-like label. Adoption requires the explicit
            # edge kind and a current allow outcome; either signal alone is
            # insufficient.
            authorized = kind == "authorized_distills_to" and outcome in {
                "allow",
                "allow_with_warning",
            }
            if authorized:
                register(
                    transition_id,
                    sop_id,
                    self._authorized_transitions_by_sop,
                    self._authorized_sops_by_execution,
                    self._authorized_sop_links_by_execution,
                    self._authorized_sops_by_transition,
                )
        for mapping in (
            self._navigation_transitions_by_sop,
            self._authorized_transitions_by_sop,
            self._navigation_sops_by_execution,
            self._authorized_sops_by_execution,
            self._navigation_sops_by_transition,
            self._authorized_sops_by_transition,
        ):
            for values in mapping.values():
                values.sort()
        for outer in (
            self._navigation_sop_links_by_execution,
            self._authorized_sop_links_by_execution,
        ):
            for mapping in outer.values():
                for values in mapping.values():
                    values.sort()
        # Compatibility is explicitly navigation-only. Enforced adoption paths
        # use the authorized maps through the helpers below.
        self._transitions_by_sop = self._navigation_transitions_by_sop
        self._sops_by_execution = self._navigation_sops_by_execution
        self._sop_links_by_execution = self._navigation_sop_links_by_execution
        self._sop_edge_migration = {
            "navigation_edge_count": sum(
                len(values) for values in self._navigation_transitions_by_sop.values()
            ),
            "authorized_edge_count": sum(
                len(values) for values in self._authorized_transitions_by_sop.values()
            ),
            "authority_outcomes": dict(sorted(edge_outcomes.items())),
        }
        meta_prefixes = _as_list((self.graph.get("meta") or {}).get("blocked_run_prefixes"))
        override = self._blocked_run_prefixes_override
        self._blocked_run_prefixes = tuple(str(value) for value in (override if override is not None else meta_prefixes))

    def _visibility_navigation_allowed(self) -> bool:
        pack = getattr(self._trace_local, "visibility_pack", None)
        if not self._visibility_is_enforced() or pack is None:
            return True
        operation = str((pack.visibility_trace.get("request") or {}).get("operation") or "")
        return operation in {
            Operation.INSPECT.value,
            Operation.DEBUG_HYPOTHESIS.value,
        }

    def _active_transitions_for_sop(self, sop_id: str) -> list[str]:
        mapping = (
            self._navigation_transitions_by_sop
            if self._visibility_navigation_allowed()
            else self._authorized_transitions_by_sop
        )
        return mapping.get(sop_id, [])

    def _active_sops_for_execution(self, execution_id: str) -> list[str]:
        mapping = (
            self._navigation_sops_by_execution
            if self._visibility_navigation_allowed()
            else self._authorized_sops_by_execution
        )
        return mapping.get(execution_id, [])

    def _active_links_for_execution(self, execution_id: str) -> dict[str, list[str]]:
        mapping = (
            self._navigation_sop_links_by_execution
            if self._visibility_navigation_allowed()
            else self._authorized_sop_links_by_execution
        )
        return mapping.get(execution_id, {})

    def _active_sops_for_transition(self, transition_id: str) -> list[str]:
        mapping = (
            self._navigation_sops_by_transition
            if self._visibility_navigation_allowed()
            else self._authorized_sops_by_transition
        )
        return mapping.get(transition_id, [])

    def _validate_layered_taxonomy(self) -> None:
        meta = self.graph.get("meta") or {}
        if meta.get("sop_taxonomy_schema") != "runforest_sop_taxonomy_v1":
            raise ValueError("Layered strategy retrieval requires runforest_sop_taxonomy_v1")
        if float(meta.get("sop_taxonomy_coverage") or 0.0) != 1.0:
            raise ValueError("Layered strategy retrieval requires 100% SOP taxonomy coverage")
        taxonomy_sop_ids = self._legacy_sop_ids if self._recipe_sop_ids else self._sops
        if int(meta.get("sop_taxonomy_sop_count") or 0) != len(taxonomy_sop_ids):
            raise ValueError("Layered strategy taxonomy count does not match graph SOP count")
        required = {
            "abstraction_level",
            "sop_kind",
            "method_family",
            "task_families",
            "decision_stages",
            "compute_profile",
        }
        missing = [sop_id for sop_id in self._sops if not required <= self.nodes[sop_id].keys()]
        if missing:
            raise ValueError(f"Layered strategy SOP metadata missing for {missing[:5]}")
        allowed_levels = {"L1_strategy", "L2_tactic", "L3_repair"}
        allowed_kinds = {
            "model_strategy",
            "architecture",
            "feature",
            "training_protocol",
            "validation_protocol",
            "debug_fix",
            "infrastructure",
        }
        allowed_compute = {
            "cpu_light",
            "cpu_or_single_gpu",
            "single_gpu_standard",
            "single_gpu_large",
            "multi_gpu_preferred",
        }
        invalid = [
            sop_id
            for sop_id in self._sops
            if self.nodes[sop_id].get("abstraction_level") not in allowed_levels
            or self.nodes[sop_id].get("sop_kind") not in allowed_kinds
            or self.nodes[sop_id].get("compute_profile") not in allowed_compute
            or not self.nodes[sop_id].get("method_family")
            or not self.nodes[sop_id].get("task_families")
            or not self.nodes[sop_id].get("decision_stages")
        ]
        if invalid:
            raise ValueError(f"Layered strategy SOP metadata is invalid for {invalid[:5]}")
        l1_ids = [
            sop_id
            for sop_id in taxonomy_sop_ids
            if self.nodes[sop_id].get("abstraction_level") == "L1_strategy"
        ]
        reviewed_l1_count = meta.get("sop_taxonomy_reviewed_l1_count")
        if reviewed_l1_count is None or int(reviewed_l1_count) != len(l1_ids):
            raise ValueError("Layered strategy taxonomy has incomplete manual L1 review metadata")
        if any(self.nodes[sop_id].get("manual_reviewed") is not True for sop_id in l1_ids):
            raise ValueError("Layered strategy taxonomy contains an unreviewed L1 SOP")
        recipe_l1_ids = [
            sop_id
            for sop_id in self._recipe_sop_ids
            if self.nodes[sop_id].get("abstraction_level") == "L1_strategy"
        ]
        if self._recipe_sop_ids:
            if self.recipe_sop_receipt.get("node_count") != len(self._recipe_sop_ids):
                raise ValueError("Recipe SOP overlay receipt count mismatch")
            if not recipe_l1_ids or any(
                self.nodes[sop_id].get("manual_reviewed") is not True
                for sop_id in recipe_l1_ids
            ):
                raise ValueError("Recipe SOP overlay has incomplete L1 review metadata")

    def _resolve_config_path(self, value: str) -> Path:
        path = Path(str(value or ""))
        if path.is_absolute():
            return path
        mlevolve_root = Path(__file__).resolve().parents[2]
        candidates = [Path.cwd() / path, mlevolve_root / path]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[-1].resolve()

    @staticmethod
    def _recipe_payload_hash(payload: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "bundle_sha256"
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _recipe_compute_profile(node: Mapping[str, Any]) -> str:
        family = str(node.get("method_family") or "").lower()
        if any(token in family for token in ("lightgbm", "xgboost", "logistic")) and not any(
            token in family for token in ("transformer", "resnet", "siglip", "dino")
        ):
            return "cpu_light"
        if any(token in family for token in ("large", "siglip", "dino", "nafnet", "unet")):
            return "single_gpu_large"
        return "single_gpu_standard"

    @staticmethod
    def _format_recipe_action(node: Mapping[str, Any]) -> str:
        level = str(node.get("abstraction_level") or "")
        if level == "L2_tactic":
            return str(node.get("instruction") or "").strip()
        if level == "L3_repair":
            signature = (
                node.get("failure_signature")
                if isinstance(node.get("failure_signature"), Mapping)
                else {}
            )
            repair = (
                node.get("repair_action")
                if isinstance(node.get("repair_action"), Mapping)
                else {}
            )
            steps = [str(value) for value in (repair.get("steps") or []) if str(value)]
            return "\n".join(
                [
                    "Evidence-bound L3 Debug repair:",
                    f"Failure signature: {signature.get('id', '')}",
                    f"Failure pattern: {signature.get('pattern', '')}",
                    f"Root cause: {signature.get('root_cause', '')}",
                    f"Runtime stage: {node.get('runtime_stage', '')}",
                    f"Repair: {repair.get('summary', '')}",
                    *(f"- {value}" for value in steps),
                    f"Evidence admission: {node.get('evidence_status', 'accepted_clean_repair')} "
                    f"({node.get('successful_repair_count', 1)} clean repair success; "
                    "count does not create an evidence tier).",
                ]
            ).strip()
        pipeline = node.get("pipeline") if isinstance(node.get("pipeline"), Mapping) else {}
        labels = (
            ("data_validation", "Data validation"),
            ("split_validation", "Split and validation"),
            ("feature_representation", "Feature representation"),
            ("model_stack", "Model stack"),
            ("training_protocol", "Training protocol"),
            ("oof_protocol", "OOF protocol"),
            ("ensemble_calibration", "Ensemble and calibration"),
            ("final_refit_inference", "Final refit and inference"),
            ("failure_boundaries", "Failure boundaries"),
        )
        lines = ["Complete end-to-end recipe:"]
        for key, label in labels:
            value = str(pipeline.get(key) or "").strip()
            if value:
                lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def _normalize_recipe_sop(self, source: Mapping[str, Any]) -> dict[str, Any]:
        node = copy.deepcopy(dict(source))
        node_id = str(node.get("id") or "")
        raw_level = str(node.get("abstraction_level") or "")
        if not node_id or node.get("type") != "SOP":
            raise ValueError("Recipe SOP entries require stable IDs and type=SOP")
        if raw_level not in {"L1_recipe", "L2_tactic", "L3_repair"}:
            raise ValueError(f"Unsupported Recipe SOP abstraction level for {node_id}")
        task_id = canonical_task_id(str(node.get("task_id") or ""))
        if task_id not in TASK_PROFILES:
            raise ValueError(f"Recipe SOP has unknown task_id: {node_id}")
        family = str(node.get("method_family") or "")
        parents = [str(value) for value in (node.get("parent_method_families") or [])]
        if raw_level == "L1_recipe" and not family:
            raise ValueError(f"Recipe L1 has no method_family: {node_id}")
        if raw_level == "L2_tactic" and not parents:
            raise ValueError(f"Recipe L2 has no parent_method_families: {node_id}")
        if raw_level == "L3_repair":
            signature = node.get("failure_signature")
            repair = node.get("repair_action")
            transitions = [
                str(value) for value in (node.get("supporting_transition_ids") or [])
            ]
            if (
                not isinstance(signature, Mapping)
                or not str(signature.get("id") or "")
                or not isinstance(repair, Mapping)
                or not list(repair.get("steps") or [])
                or not transitions
                or not str(node.get("task_type") or "")
                or not str(node.get("runtime_stage") or "")
            ):
                raise ValueError(f"Recipe L3 has incomplete repair evidence: {node_id}")
            for transition_id in transitions:
                transition = self.nodes.get(transition_id)
                if transition is None:
                    # The frozen overlay spans several task graphs.  A
                    # task-local Base Bundle legitimately lacks transitions
                    # from other tasks; those L3 cards remain inert because
                    # no authorized edge can be materialized locally.
                    continue
                if (
                    not isinstance(transition, Mapping)
                    or transition.get("type") != "Transition"
                    or transition.get("outcome") != "debug_fixed"
                    or transition.get("parent_buggy") is not True
                    or transition.get("child_buggy") is not False
                ):
                    raise ValueError(
                        f"Recipe L3 references an invalid repair transition: {transition_id}"
                    )
            node["available_supporting_transition_ids"] = [
                transition_id
                for transition_id in transitions
                if self.nodes.get(transition_id, {}).get("type") == "Transition"
            ]
        original_kind = str(node.get("sop_kind") or "")
        normalized_kind = {
            "model_strategy_recipe": "model_strategy",
            "model_design": "architecture",
            "ensemble": "training_protocol",
            "debug_fix": "debug_fix",
        }.get(original_kind, original_kind)
        task_domain = str(node.get("task_domain") or "")
        task_family = TASK_PROFILES[task_id][1]
        action = self._format_recipe_action(node)
        if not action:
            raise ValueError(f"Recipe SOP has no prompt-visible method text: {node_id}")
        node.update(
            {
                "task": task_id,
                "task_id": task_id,
                "recipe_task_id": task_id,
                "recipe_abstraction_level": raw_level,
                "abstraction_level": (
                    {
                        "L1_recipe": "L1_strategy",
                        "L2_tactic": "L2_tactic",
                        "L3_repair": "L3_repair",
                    }[raw_level]
                ),
                "recipe_sop_kind": original_kind,
                "sop_kind": normalized_kind,
                "method_family": family or "general",
                "parent_method_families": parents,
                "task_families": list(
                    dict.fromkeys(
                        value
                        for value in (task_family, task_domain)
                        if value
                    )
                ),
                "task_type": str(node.get("task_type") or ""),
                "action": action,
                "text": action,
                "applies_when": [str(node.get("when_to_use") or "")],
                "prevents": [
                    (
                        str((node.get("failure_signature") or {}).get("pattern") or "")
                        if raw_level == "L3_repair"
                        else str((node.get("pipeline") or {}).get("failure_boundaries") or "")
                    )
                ]
                if raw_level == "L3_repair" or isinstance(node.get("pipeline"), Mapping)
                else [],
                "compute_profile": self._recipe_compute_profile(node),
                "manual_reviewed": True,
                "recipe_overlay": True,
                "source_bundle_sha256": self.recipe_sop_bundle_sha256,
            }
        )
        return node

    def _recipe_overlay_route_enabled(self) -> bool:
        """Whether the active router may consume the frozen Recipe overlays.

        Recipe overlays were introduced for ``layered_strategy``.  The full
        Experiment-R route deliberately reuses the same frozen SOP, evidence,
        and implementation capsules, so ``dynamic_hybrid`` must be admitted
        when (and only when) Experiment-R is enabled.
        """

        return self.retrieval_control == "layered_strategy" or (
            self.retrieval_control == "dynamic_hybrid"
            and self.experiment_r_enabled
        )

    def _load_recipe_sop_overlay(self) -> None:
        if not self._recipe_overlay_route_enabled():
            raise ValueError(
                "Recipe SOP overlay requires layered_strategy or the full "
                "Experiment-R dynamic_hybrid router"
            )
        if len(self.recipe_sop_file_sha256) != 64 or len(self.recipe_sop_bundle_sha256) != 64:
            raise ValueError("Recipe SOP overlay requires frozen file and bundle SHA-256 pins")
        path = self._resolve_config_path(self.recipe_sop_path)
        raw = path.read_bytes()
        observed_file_sha = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(observed_file_sha, self.recipe_sop_file_sha256):
            raise ValueError("Recipe SOP file SHA-256 mismatch")
        payload = json.loads(raw.decode("utf-8"))
        declared_bundle_sha = str(payload.get("bundle_sha256") or "")
        observed_bundle_sha = self._recipe_payload_hash(payload)
        if not hmac.compare_digest(declared_bundle_sha, observed_bundle_sha):
            raise ValueError("Recipe SOP canonical payload hash mismatch")
        if not hmac.compare_digest(observed_bundle_sha, self.recipe_sop_bundle_sha256):
            raise ValueError("Recipe SOP bundle SHA-256 does not match the frozen pin")
        sources = payload.get("nodes")
        if not isinstance(sources, list) or not sources:
            raise ValueError("Recipe SOP bundle contains no nodes")
        recipe_ids: list[str] = []
        for source in sources:
            if not isinstance(source, Mapping):
                raise ValueError("Recipe SOP bundle contains a non-object entry")
            node = self._normalize_recipe_sop(source)
            node_id = str(node["id"])
            if node_id in self.nodes:
                raise ValueError(f"Recipe SOP ID collides with RunForest graph: {node_id}")
            self.nodes[node_id] = node
            self._node_tokens[node_id] = _tokenize(self._node_text(node))
            recipe_ids.append(node_id)
            if node.get("abstraction_level") == "L3_repair":
                existing_edges = {
                    (str(edge.get("src") or ""), str(edge.get("dst") or ""))
                    for edge in self.graph.get("edges", [])
                }
                for transition_id in node.get("available_supporting_transition_ids") or []:
                    if (str(transition_id), node_id) in existing_edges:
                        continue
                    self.graph.setdefault("edges", []).append(
                        {
                            "src": str(transition_id),
                            "dst": node_id,
                            "kind": "authorized_distills_to",
                            "authority_outcome": "allow",
                            "quality": "evidence_turn_match",
                            "score": 1.0,
                            "provenance": "frozen_l3_repair_distillation_v2",
                        }
                    )
        self._recipe_sop_ids = sorted(recipe_ids)
        self._sops = sorted(set(self._legacy_sop_ids) | set(self._recipe_sop_ids))
        self.recipe_sop_receipt = {
            "schema": "layered_recipe_sop_overlay_receipt_v1",
            "path": str(path),
            "file_sha256": observed_file_sha,
            "bundle_sha256": observed_bundle_sha,
            "bundle_version": payload.get("bundle_version"),
            "node_count": len(self._recipe_sop_ids),
            "l1_count": sum(
                self.nodes[node_id]["abstraction_level"] == "L1_strategy"
                for node_id in self._recipe_sop_ids
            ),
            "l2_count": sum(
                self.nodes[node_id]["abstraction_level"] == "L2_tactic"
                for node_id in self._recipe_sop_ids
            ),
            "l3_count": sum(
                self.nodes[node_id]["abstraction_level"] == "L3_repair"
                for node_id in self._recipe_sop_ids
            ),
        }

    def _load_recipe_evidence_overlay(self) -> None:
        if not self._recipe_overlay_route_enabled():
            raise ValueError(
                "Recipe evidence overlay requires layered_strategy or the full "
                "Experiment-R dynamic_hybrid router"
            )
        if len(self.recipe_evidence_file_sha256) != 64 or len(
            self.recipe_evidence_manifest_sha256
        ) != 64:
            raise ValueError("Recipe evidence overlay requires frozen file and manifest hashes")
        path = self._resolve_config_path(self.recipe_evidence_path)
        raw = path.read_bytes()
        observed_file_sha = hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(observed_file_sha, self.recipe_evidence_file_sha256):
            raise ValueError("Recipe evidence file SHA-256 mismatch")
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("schema") != "mlevolve_recipe_distillation_evidence_v1":
            raise ValueError("Unsupported Recipe evidence schema")
        observed_manifest_sha = hashlib.sha256(
            json.dumps(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "manifest_sha256"
                },
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        declared_manifest_sha = str(payload.get("manifest_sha256") or "")
        if not hmac.compare_digest(declared_manifest_sha, observed_manifest_sha):
            raise ValueError("Recipe evidence canonical manifest hash mismatch")
        if not hmac.compare_digest(
            observed_manifest_sha,
            self.recipe_evidence_manifest_sha256,
        ):
            raise ValueError("Recipe evidence manifest SHA-256 does not match frozen pin")

        selected = payload.get("selected_evidence")
        if not isinstance(selected, Mapping) or not selected:
            raise ValueError("Recipe evidence manifest has no selected evidence")
        selected_count = 0
        materialized_count = 0
        existing_count = 0
        evidence_ids: list[str] = []
        for selected_task, records in selected.items():
            task_id = canonical_task_id(str(selected_task or ""))
            if task_id not in TASK_PROFILES or not isinstance(records, list):
                raise ValueError(f"Invalid Recipe evidence task group: {selected_task}")
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError("Recipe evidence contains a non-object record")
                selected_count += 1
                node_id = str(record.get("node_id") or "")
                metric = record.get("metric")
                if (
                    not node_id
                    or canonical_task_id(record.get("task_id")) != task_id
                    or not isinstance(metric, (int, float))
                    or isinstance(metric, bool)
                    or not math.isfinite(float(metric))
                    or record.get("audit_status") != "clean"
                    or record.get("memory_disposition") != "positive_eligible"
                    or record.get("paper_grade_eligible") is not True
                    or record.get("rank_eligible") is not True
                    or len(str(record.get("code_sha256") or "")) != 64
                ):
                    raise ValueError(f"Recipe evidence is not strict-clean eligible: {node_id}")
                evidence_ids.append(node_id)
                existing = self.nodes.get(node_id)
                if existing is not None:
                    existing_metric = existing.get("metric")
                    existing_audit = (
                        existing.get("leakage_audit")
                        if isinstance(existing.get("leakage_audit"), Mapping)
                        else {}
                    )
                    if (
                        existing.get("type") != "RunNode"
                        or canonical_task_id(existing.get("task")) != task_id
                        or not isinstance(existing_metric, (int, float))
                        or abs(float(existing_metric) - float(metric)) > 1e-12
                        or str(existing.get("code_sha256") or "")
                        != str(record.get("code_sha256") or "")
                        or existing_audit.get("status") != "clean"
                    ):
                        raise ValueError(
                            f"Recipe evidence conflicts with base RunForest node: {node_id}"
                        )
                    existing_count += 1
                    continue
                source_cohort = str(record.get("source_cohort") or "")
                terminal_evidence = bool(
                    node_id.startswith("postsmoke::")
                    or source_cohort == "post_freeze_leaf_smoke_20260805"
                )
                node = {
                    "id": node_id,
                    "type": "RunNode",
                    "task": task_id,
                    "run_id": str(record.get("run_id") or ""),
                    "run_short_id": str(record.get("run_id") or ""),
                    "stage": str(record.get("stage") or "draft"),
                    "step": record.get("step"),
                    "metric": float(metric),
                    "metric_direction": str(record.get("metric_direction") or "unknown"),
                    "metric_provenance": (
                        "sealed_fixed_holdout_terminal_score"
                        if terminal_evidence
                        else "distilled_historical_search_metric"
                    ),
                    "metric_improvement": record.get("metric_improvement"),
                    "is_buggy": False,
                    "is_valid": True,
                    "plan": str(record.get("plan") or ""),
                    "code_summary": str(record.get("code_summary") or ""),
                    "code_sha256": str(record.get("code_sha256") or ""),
                    "source_cohort": source_cohort,
                    "quarantined": False,
                    "protocol_biased": False,
                    "leakage_audit": {
                        "status": "clean",
                        "memory_disposition": "positive_eligible",
                        "paper_grade_eligible": True,
                        "rank_eligible": True,
                    },
                    "recipe_evidence_overlay": True,
                    "source_manifest_sha256": observed_manifest_sha,
                }
                self.nodes[node_id] = node
                self._node_tokens[node_id] = _tokenize(self._node_text(node))
                materialized_count += 1
        if selected_count != sum(
            int(value) for value in (payload.get("selected_counts_by_task") or {}).values()
        ):
            raise ValueError("Recipe evidence selected-count receipt mismatch")
        selected_repairs = payload.get("selected_repair_evidence")
        if not isinstance(selected_repairs, Mapping):
            raise ValueError("Recipe evidence manifest has no selected repair evidence")
        repair_map: dict[str, dict[str, Any]] = {}
        repair_count = 0
        for selected_task, records in selected_repairs.items():
            task_id = canonical_task_id(str(selected_task or ""))
            if task_id not in TASK_PROFILES or not isinstance(records, list):
                raise ValueError(
                    f"Invalid Recipe repair evidence task group: {selected_task}"
                )
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError("Recipe repair evidence contains a non-object")
                transition_id = str(record.get("transition_id") or "")
                successful_metric = record.get("successful_metric")
                if (
                    not transition_id
                    or transition_id in repair_map
                    or canonical_task_id(record.get("task_id")) != task_id
                    or "debug" not in str(record.get("stage_pair") or "")
                    or not str(record.get("failure_node_id") or "")
                    or not str(record.get("successful_node_id") or "")
                    or not str(record.get("failure_text") or "")
                    or not str(record.get("repair_action_text") or "")
                    or not isinstance(successful_metric, (int, float))
                    or isinstance(successful_metric, bool)
                    or not math.isfinite(float(successful_metric))
                    or record.get("audit_status") != "clean"
                    or record.get("memory_disposition") != "positive_eligible"
                    or record.get("paper_grade_eligible") is not True
                    or record.get("rank_eligible") is not True
                ):
                    raise ValueError(
                        f"Recipe repair evidence is not strict-clean eligible: {transition_id}"
                    )
                repair_map[transition_id] = dict(record)
                repair_count += 1
        if repair_count != sum(
            int(value)
            for value in (payload.get("selected_repair_counts_by_task") or {}).values()
        ):
            raise ValueError("Recipe repair evidence selected-count receipt mismatch")
        materialized_repair_transition_count = 0
        materialized_repair_node_count = 0
        for transition_id, record in sorted(repair_map.items()):
            task_id = canonical_task_id(record.get("task_id"))
            run_id = str(record.get("run_id") or "")
            parent_id = str(record["failure_node_id"])
            child_id = str(record["successful_node_id"])

            parent = self.nodes.get(parent_id)
            if parent is None:
                parent = {
                    "id": parent_id,
                    "type": "RunNode",
                    "task": task_id,
                    "run_id": run_id,
                    "run_short_id": run_id,
                    "stage": "debug",
                    "is_buggy": True,
                    "is_valid": False,
                    "analysis": str(record.get("failure_text") or ""),
                    "terminal_excerpt": str(record.get("failure_text") or "")[-4000:],
                    "code_sha256": str(record.get("failure_node_code_sha256") or ""),
                    "quarantined": False,
                    "protocol_biased": False,
                    "leakage_audit": {
                        "status": "clean_failure_evidence",
                        "memory_disposition": "repair_only",
                        "paper_grade_eligible": False,
                        "rank_eligible": False,
                    },
                    "recipe_repair_evidence_overlay": True,
                    "source_manifest_sha256": observed_manifest_sha,
                }
                self.nodes[parent_id] = parent
                self._node_tokens[parent_id] = _tokenize(self._node_text(parent))
                self._run_nodes.append(parent_id)
                self._run_nodes_by_run[run_id].append(parent_id)
                self.graph.setdefault("nodes", []).append(parent)
                materialized_repair_node_count += 1
            elif (
                parent.get("type") != "RunNode"
                or canonical_task_id(parent.get("task")) != task_id
            ):
                raise ValueError(
                    f"Recipe repair failure node conflicts with graph: {parent_id}"
                )

            child = self.nodes.get(child_id)
            successful_metric = float(record["successful_metric"])
            if child is None:
                child = {
                    "id": child_id,
                    "type": "RunNode",
                    "task": task_id,
                    "run_id": run_id,
                    "run_short_id": run_id,
                    "stage": "debug",
                    "parent_id": parent_id,
                    "metric": successful_metric,
                    "metric_direction": str(
                        record.get("successful_metric_direction") or "unknown"
                    ),
                    "is_buggy": False,
                    "is_valid": True,
                    "plan": str(record.get("repair_action_text") or ""),
                    "code_summary": str(record.get("repair_action_text") or ""),
                    "analysis": str(
                        record.get("successful_execution_summary") or ""
                    ),
                    "code_sha256": str(
                        record.get("successful_node_code_sha256") or ""
                    ),
                    "quarantined": False,
                    "protocol_biased": False,
                    "leakage_audit": {
                        "status": "clean",
                        "memory_disposition": "positive_eligible",
                        "paper_grade_eligible": True,
                        "rank_eligible": True,
                    },
                    "recipe_repair_evidence_overlay": True,
                    "source_manifest_sha256": observed_manifest_sha,
                }
                self.nodes[child_id] = child
                self._node_tokens[child_id] = _tokenize(self._node_text(child))
                self._run_nodes.append(child_id)
                self._run_nodes_by_run[run_id].append(child_id)
                self._children_by_node[parent_id].append(child_id)
                self.graph.setdefault("nodes", []).append(child)
                materialized_repair_node_count += 1
            else:
                child_metric = child.get("metric")
                if (
                    child.get("type") != "RunNode"
                    or canonical_task_id(child.get("task")) != task_id
                    or not isinstance(child_metric, (int, float))
                    or abs(float(child_metric) - successful_metric) > 1e-12
                ):
                    raise ValueError(
                        f"Recipe repair success node conflicts with graph: {child_id}"
                    )
                if child_id not in self._children_by_node[parent_id]:
                    self._children_by_node[parent_id].append(child_id)

            transition = self.nodes.get(transition_id)
            if transition is None:
                transition = {
                    "id": transition_id,
                    "type": "Transition",
                    "task": task_id,
                    "run_id": run_id,
                    "run_short_id": run_id,
                    "parent_node_id": parent_id,
                    "child_node_id": child_id,
                    "stage_pair": str(record.get("stage_pair") or "debug->debug"),
                    "outcome": "debug_fixed",
                    "parent_buggy": True,
                    "child_buggy": False,
                    "child_metric": successful_metric,
                    "metric_improvement": None,
                    "text": str(record.get("repair_action_text") or ""),
                    "quarantined": False,
                    "protocol_biased": False,
                    "recipe_repair_evidence_overlay": True,
                    "source_manifest_sha256": observed_manifest_sha,
                }
                self.nodes[transition_id] = transition
                self._node_tokens[transition_id] = _tokenize(
                    self._node_text(transition)
                )
                self._transitions.append(transition_id)
                self._transitions_by_parent[parent_id].append(transition_id)
                self._transitions_by_child[child_id].append(transition_id)
                self.graph.setdefault("nodes", []).append(transition)
                self.graph.setdefault("edges", []).extend(
                    [
                        {
                            "src": parent_id,
                            "dst": child_id,
                            "kind": "parent_of",
                            "provenance": "frozen_recipe_repair_evidence_v1",
                        },
                        {
                            "src": parent_id,
                            "dst": transition_id,
                            "kind": "has_transition",
                            "provenance": "frozen_recipe_repair_evidence_v1",
                        },
                        {
                            "src": transition_id,
                            "dst": child_id,
                            "kind": "transition_to",
                            "provenance": "frozen_recipe_repair_evidence_v1",
                        },
                    ]
                )
                materialized_repair_transition_count += 1
            elif (
                transition.get("type") != "Transition"
                or str(transition.get("parent_node_id") or "") != parent_id
                or str(transition.get("child_node_id") or "") != child_id
                or transition.get("outcome") != "debug_fixed"
            ):
                raise ValueError(
                    f"Recipe repair transition conflicts with graph: {transition_id}"
                )

        for values in self._run_nodes_by_run.values():
            values.sort(key=lambda node_id: (self.nodes[node_id].get("step") or 0, node_id))
        for values in self._children_by_node.values():
            values.sort(key=lambda node_id: (self.nodes[node_id].get("step") or 0, node_id))
        self._recipe_repair_evidence_by_transition = repair_map
        self._recipe_evidence_ids = sorted(set(evidence_ids))
        self.recipe_evidence_receipt = {
            "schema": "layered_recipe_evidence_overlay_receipt_v1",
            "path": str(path),
            "file_sha256": observed_file_sha,
            "manifest_sha256": observed_manifest_sha,
            "selected_node_count": selected_count,
            "materialized_node_count": materialized_count,
            "existing_node_count": existing_count,
            "selected_repair_transition_count": repair_count,
            "materialized_repair_transition_count": (
                materialized_repair_transition_count
            ),
            "materialized_repair_node_count": materialized_repair_node_count,
            "terminal_node_count": sum(
                self._is_terminal_strategy_evidence(node_id, self.nodes[node_id])
                for node_id in self._recipe_evidence_ids
            ),
        }

    @staticmethod
    def _implementation_unified_diff(
        before_code: str,
        after_code: str,
        parent_id: str,
        child_id: str,
    ) -> str:
        """Return the canonical minimal-context diff stored in a capsule."""

        return "".join(
            difflib.unified_diff(
                str(before_code).splitlines(keepends=True),
                str(after_code).splitlines(keepends=True),
                fromfile=f"before/{parent_id}",
                tofile=f"after/{child_id}",
                n=3,
            )
        )

    def _load_recipe_implementation_capsules(self) -> None:
        """Bind full code and exact repair diffs to frozen RunForest hashes.

        Recipe evidence deliberately keeps the retrieval graph compact.  This
        separate capsule restores executable detail only after a
        strategy or repair has been selected, so code never changes ranking.
        """

        if not self._recipe_overlay_route_enabled():
            raise ValueError(
                "Recipe implementation capsules require layered_strategy or the "
                "full Experiment-R dynamic_hybrid router"
            )
        path = self._resolve_config_path(self.recipe_implementation_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "mlevolve_recipe_implementation_capsules_v1":
            raise ValueError("Unsupported Recipe implementation capsule schema")

        node_rows = payload.get("nodes")
        transition_rows = payload.get("transitions")
        if not isinstance(node_rows, list) or not isinstance(transition_rows, list):
            raise ValueError("Recipe implementation capsule inventory is malformed")
        nodes_by_id: dict[str, Mapping[str, Any]] = {}
        for row in node_rows:
            if not isinstance(row, Mapping):
                raise ValueError("Recipe implementation node is not an object")
            node_id = str(row.get("node_id") or "")
            if not node_id or node_id in nodes_by_id:
                raise ValueError(
                    f"Duplicate or missing implementation node id: {node_id}"
                )
            nodes_by_id[node_id] = row
        transitions_by_id: dict[str, Mapping[str, Any]] = {}
        for row in transition_rows:
            if not isinstance(row, Mapping):
                raise ValueError("Recipe implementation transition is not an object")
            transition_id = str(row.get("transition_id") or "")
            if not transition_id or transition_id in transitions_by_id:
                raise ValueError(
                    "Duplicate or missing implementation transition id: "
                    f"{transition_id}"
                )
            transitions_by_id[transition_id] = row

        required_node_ids = {
            node_id
            for node_id in self._recipe_evidence_ids
            if len(str(self.nodes.get(node_id, {}).get("code_sha256") or "")) == 64
        }
        required_transition_ids: set[str] = set()
        for transition_id, repair in self._recipe_repair_evidence_by_transition.items():
            parent_id = str(repair.get("failure_node_id") or "")
            child_id = str(repair.get("successful_node_id") or "")
            if len(str(repair.get("failure_node_code_sha256") or "")) == 64:
                required_node_ids.add(parent_id)
            if len(str(repair.get("successful_node_code_sha256") or "")) == 64:
                required_node_ids.add(child_id)
            if parent_id in required_node_ids and child_id in required_node_ids:
                required_transition_ids.add(str(transition_id))
        declared_required_nodes = {
            str(value) for value in payload.get("required_node_ids") or []
        }
        declared_required_transitions = {
            str(value) for value in payload.get("required_transition_ids") or []
        }
        if declared_required_nodes != required_node_ids:
            raise ValueError("Recipe implementation required-node coverage mismatch")
        if not set(nodes_by_id).issubset(required_node_ids):
            raise ValueError("Recipe implementation node inventory has unknown nodes")
        if declared_required_transitions != required_transition_ids:
            raise ValueError(
                "Recipe implementation required-transition coverage mismatch"
            )
        if not set(transitions_by_id).issubset(required_transition_ids):
            raise ValueError(
                "Recipe implementation transition inventory has unknown transitions"
            )

        for node_id, row in nodes_by_id.items():
            node = self.nodes.get(node_id)
            if not isinstance(node, dict) or node.get("type") != "RunNode":
                raise ValueError(
                    f"Recipe implementation references a missing RunNode: {node_id}"
                )
            code = row.get("code")
            if not isinstance(code, str) or not code.strip():
                raise ValueError(
                    f"Recipe implementation has no source code: {node_id}"
                )
            code_sha = hashlib.sha256(code.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(
                code_sha, str(row.get("code_sha256") or "")
            ) or not hmac.compare_digest(
                code_sha, str(node.get("code_sha256") or "")
            ):
                raise ValueError(
                    f"Recipe implementation code hash mismatch: {node_id}"
                )
            if not str(row.get("source_journal") or ""):
                raise ValueError(
                    f"Recipe implementation source provenance is incomplete: {node_id}"
                )
            node["implementation_capsule"] = {
                "schema": "mlevolve_run_node_implementation_capsule_v1",
                "node_id": node_id,
                "code": code,
                "code_sha256": code_sha,
                "source_journal": str(row.get("source_journal")),
                "source_raw_node_id": str(row.get("source_raw_node_id") or ""),
            }

        for transition_id, row in transitions_by_id.items():
            transition = self.nodes.get(transition_id)
            if not isinstance(transition, dict) or transition.get("type") != "Transition":
                raise ValueError(
                    "Recipe implementation references a missing Transition: "
                    f"{transition_id}"
                )
            parent_id = str(row.get("parent_node_id") or "")
            child_id = str(row.get("child_node_id") or "")
            if (
                parent_id != str(transition.get("parent_node_id") or "")
                or child_id != str(transition.get("child_node_id") or "")
            ):
                raise ValueError(
                    f"Recipe implementation transition binding mismatch: {transition_id}"
                )
            before = str(nodes_by_id[parent_id]["code"])
            after = str(nodes_by_id[child_id]["code"])
            expected_diff = self._implementation_unified_diff(
                before, after, parent_id, child_id
            )
            transition["implementation_repair_capsule"] = {
                "schema": "mlevolve_repair_implementation_capsule_v1",
                "transition_id": transition_id,
                "parent_node_id": parent_id,
                "child_node_id": child_id,
                "before_code": before,
                "after_code": after,
                "before_code_sha256": str(nodes_by_id[parent_id]["code_sha256"]),
                "after_code_sha256": str(nodes_by_id[child_id]["code_sha256"]),
                "unified_diff": expected_diff,
            }

        self.recipe_implementation_receipt = {
            "schema": "layered_recipe_implementation_receipt_v1",
            "path": str(path),
            "node_count": len(nodes_by_id),
            "transition_count": len(transitions_by_id),
            "required_node_count": len(required_node_ids),
            "required_transition_count": len(required_transition_ids),
            "missing_node_ids": sorted(required_node_ids - set(nodes_by_id)),
            "missing_transition_ids": sorted(
                required_transition_ids - set(transitions_by_id)
            ),
            "complete_recipe_coverage": set(nodes_by_id) == required_node_ids,
        }

    def _model_family_from_text(self, value: str) -> str:
        text = str(value or "").lower()
        if "modernbert" in text:
            return "modernbert_finetune"
        if "deberta" in text and any(token in text for token in ("xgboost", "logistic regression", "tf-idf")):
            return "deberta_xgb_lr_ensemble"
        if "deberta" in text:
            return "deberta_finetune"
        if "roberta" in text:
            return "roberta_finetune"
        if "efficientnet" in text:
            return "efficientnet_finetune"
        # DINO model names are versioned and commonly arrive as checkpoint or
        # repository identifiers (for example ``DINOv3`` or
        # ``facebook/dinov3-vitl16-pretrain``).  Match the model family rather
        # than pinning one release name so a new numeric DINO version cannot
        # abort the strict three-role draft path.
        dino_family = re.search(
            r"(?<![a-z0-9])dino(?:[\s._/-]*v?[\s._/-]*\d+)?(?![a-z0-9])",
            text,
        )
        if "siglip" in text or dino_family or "vision transformer" in text:
            return "vision_transformer_finetune"
        # Audio cold-start templates may use a model repository identifier
        # (for example OpenMuQ/MuQ-large-msd-iter) without an explicit
        # architecture token. Keep audio as its own exclusion family instead
        # of failing the layered retrieval preflight.
        if any(
            token in text
            for token in (
                "muq",
                "audio",
                "librosa",
                "wav2vec",
                "whisper",
                "music information retrieval",
            )
        ):
            return "audio_finetune"
        if any(token in text for token in ("lightgbm", "xgboost")):
            return "gradient_boosted_trees"
        raise ValueError(f"Cannot map model description to method_family: {value!r}")

    def _replay_family(self, task_id: str) -> str:
        if self.cfg is None:
            raise ValueError("Layered strategy retrieval requires cfg for replay-family exclusion")
        policy = getattr(self.cfg.agent, "draft_role_policy", None)
        raw_path = str(getattr(policy, "replay_targets_path", "") or "").strip()
        if not raw_path:
            return ""
        manifest_path = self._resolve_config_path(raw_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Replay target manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        canonical_task = canonical_task_id(task_id)
        target = next(
            (
                item
                for item in manifest.get("targets", [])
                if canonical_task_id(item.get("task_id")) == canonical_task
            ),
            None,
        )
        if target is None:
            # A task with no exact clean source (for example a newly introduced
            # benchmark) may still use cross-task clean memory.  Exact replay
            # remains fail-closed in run_forest_replay.load_exact_replay; this
            # empty value only means there is no replay family to exclude from
            # Novel strategy selection.
            return ""
        family = str(target.get("method_family") or "")
        if not family:
            raise ValueError(f"Replay target {task_id} has no method_family")
        return family

    def _build_task_profile(
        self,
        *,
        task_id: str,
        task_desc: str,
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        context = context or {}
        canonical_task = canonical_task_id(task_id)
        modality, task_family = TASK_PROFILES.get(canonical_task, ("unknown", "general"))
        preview = str(context.get("data_preview") or "")
        row_matches = [int(value) for value in re.findall(r"(?i)train[^\n]{0,80}?(\d{3,})", preview)]
        train_rows = max(row_matches) if row_matches else None
        if train_rows is None:
            size_band = "unknown"
        elif train_rows < 5_000:
            size_band = "tiny"
        elif train_rows < 25_000:
            size_band = "small"
        elif train_rows < 100_000:
            size_band = "medium"
        else:
            size_band = "large"
        description = task_desc.lower()
        if "log loss" in description or "logloss" in description:
            metric_name, metric_direction = "log_loss", "minimize"
        elif "auc" in description:
            metric_name, metric_direction = "auc", "maximize"
        elif "rmse" in description or "root mean squared" in description:
            metric_name, metric_direction = "rmse", "minimize"
        else:
            metric_name, metric_direction = "task_metric", "unknown"
        if "excluded_method_families" in context:
            excluded = [str(value) for value in context.get("excluded_method_families") or []]
            baseline = str(context.get("baseline_model") or "").strip()
            baseline_family = ""
        else:
            baseline = str(context.get("baseline_model") or "").strip()
            baseline_family = (
                self._model_family_from_text(baseline)
                if baseline and baseline.lower() != "none model"
                else ""
            )
            excluded = [baseline_family] if baseline_family else []
            replay_family = self._replay_family(canonical_task)
            if replay_family:
                excluded.append(replay_family)
        gpu_count = 0
        cpu_count: int | str = "unknown"
        if self.cfg is not None:
            gpu_count = int(getattr(self.cfg.agent.search, "num_gpus", 0) or 0)
            raw_cpu = getattr(self.cfg, "cpu_number", "unknown")
            try:
                cpu_count = int(raw_cpu)
            except (TypeError, ValueError):
                cpu_count = "unknown"
        checkpoint_text = str(context.get("coldstart") or context.get("baseline_model") or "")
        checkpoints = list(dict.fromkeys(re.findall(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", checkpoint_text)))[:8]
        return {
            "task_id": canonical_task,
            "modality": modality,
            "task_family": task_family,
            "problem_type": task_family,
            "train_rows": train_rows,
            "train_size_band": size_band,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "coldstart_primary_model_available": bool(baseline_family),
            "coldstart_primary_model_family": baseline_family or None,
            "resource_budget": {
                "gpu_count": gpu_count,
                "cpu_count": cpu_count,
                "ram_gb": context.get("ram_gb", os.environ.get("RUNFOREST_RAM_GB", "unknown")),
            },
            "available_checkpoints": checkpoints,
            "excluded_method_families": list(dict.fromkeys(excluded)),
        }

    def _compute_fit(self, profile: str, gpu_count: int) -> float:
        if profile in {"cpu_light", "cpu_or_single_gpu"}:
            return 1.0
        if profile == "single_gpu_standard":
            return 1.0 if gpu_count >= 1 else 0.0
        if profile == "single_gpu_large":
            return 1.0 if gpu_count >= 1 else 0.0
        if profile == "multi_gpu_preferred":
            return 1.0 if gpu_count >= 2 else 0.35 if gpu_count == 1 else 0.0
        return 0.0

    def _family_compatible(self, selected_family: str, candidate_family: str) -> bool:
        if candidate_family in {"", "general", selected_family}:
            return True
        stem_rules = {
            "deberta": "deberta_family",
            "modernbert": "modernbert_family",
            "roberta": "roberta_family",
            "distilbert": "bert_family",
            "efficientnet": "cnn_vision_family",
            "convnext": "cnn_vision_family",
            "multi_cnn": "cnn_vision_family",
            "vision_transformer": "vision_transformer_family",
            "vision_tabular": "vision_transformer_family",
        }
        for stem, family in stem_rules.items():
            if stem in selected_family and candidate_family == family:
                return True
        if "transformer" in selected_family and candidate_family in {
            "deberta_family",
            "modernbert_family",
            "roberta_family",
            "bert_family",
        }:
            return True
        if any(token in selected_family for token in ("xgboost", "classical", "tree")):
            return candidate_family == "boosted_tree_family"
        return False

    @staticmethod
    def _is_terminal_strategy_evidence(
        node_id: str,
        node: Mapping[str, Any],
    ) -> bool:
        """Whether a support carries a sealed, label-isolated terminal score."""
        provenance = str(node.get("metric_provenance") or "")
        source_cohort = str(node.get("source_cohort") or "")
        return bool(
            provenance == "sealed_fixed_holdout_terminal_score"
            or str(node_id).startswith("postsmoke::")
            or source_cohort == "post_freeze_leaf_smoke_20260805"
        )

    def _strategy_supports(
        self,
        sop_id: str,
        task_id: str,
        *,
        metric_direction: str = "unknown",
    ) -> list[dict[str, Any]]:
        rows = []
        sop = self.nodes.get(sop_id, {})
        for node_id in dict.fromkeys(
            str(value)
            for value in (
                sop.get("clean_supporting_node_ids")
                or sop.get("source_node_ids")
                or []
            )
            if value
        ):
            node = self.nodes.get(node_id, {})
            audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
            metric = node.get("metric")
            run_id = str(node.get("run_short_id") or node.get("run_id") or "")
            strict_clean = bool(
                node.get("type") == "RunNode"
                and canonical_task_id(node.get("task")) == canonical_task_id(task_id)
                and node.get("is_buggy") is False
                and node.get("is_valid") is True
                and isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(float(metric))
                and audit.get("status") == "clean"
                and audit.get("memory_disposition") == "positive_eligible"
                and audit.get("paper_grade_eligible") is True
                and audit.get("rank_eligible") is True
                and node.get("quarantined") is not True
                and node.get("protocol_biased") is not True
                and run_id not in self.excluded_run_ids
                and not any(run_id.startswith(prefix) for prefix in self._blocked_run_prefixes)
            )
            if not strict_clean:
                continue
            improvement = node.get("metric_improvement")
            terminal_evidence = self._is_terminal_strategy_evidence(node_id, node)
            rows.append(
                {
                    "transition_id": "",
                    "evidence_kind": "direct_clean_run_node",
                    "run_id": node.get("run_id"),
                    "run_short_id": node.get("run_short_id"),
                    "node_id": node_id,
                    "stage_pair": node.get("stage"),
                    "outcome": "clean_successful_run_node",
                    "metric": metric,
                    "metric_direction": str(
                        node.get("metric_direction") or metric_direction or "unknown"
                    ),
                    "metric_provenance": (
                        str(node.get("metric_provenance") or "")
                        or (
                            "sealed_fixed_holdout_terminal_score"
                            if terminal_evidence
                            else "historical_search_metric"
                        )
                    ),
                    "terminal_evidence": terminal_evidence,
                    "metric_improvement": (
                        float(improvement)
                        if isinstance(improvement, (int, float))
                        and not isinstance(improvement, bool)
                        and math.isfinite(float(improvement))
                        else 0.0
                    ),
                    "audit_status": audit.get("status"),
                    "rank_eligible": audit.get("rank_eligible"),
                    "code_sha256": node.get("code_sha256"),
                    "implementation_available": bool(
                        node.get("implementation_capsule")
                    ),
                    "eligibility_reason": "direct_strict_clean_recipe_source",
                }
            )
        for transition_id in self._active_transitions_for_sop(sop_id):
            transition = self.nodes[transition_id]
            if canonical_task_id(transition.get("task")) != canonical_task_id(task_id):
                continue
            eligible, reason = self._positive_transition(transition_id)
            if not eligible:
                continue
            child_id = str(transition.get("child_node_id") or "")
            child = self.nodes.get(child_id, {})
            improvement = transition.get("metric_improvement")
            terminal_evidence = self._is_terminal_strategy_evidence(child_id, child)
            rows.append(
                {
                    "transition_id": transition_id,
                    "run_id": transition.get("run_id"),
                    "run_short_id": transition.get("run_short_id"),
                    "node_id": child_id,
                    "stage_pair": transition.get("stage_pair"),
                    "outcome": transition.get("outcome"),
                    "metric": child.get("metric"),
                    "metric_direction": str(
                        child.get("metric_direction") or metric_direction or "unknown"
                    ),
                    "metric_provenance": (
                        str(child.get("metric_provenance") or "")
                        or (
                            "sealed_fixed_holdout_terminal_score"
                            if terminal_evidence
                            else "historical_search_metric"
                        )
                    ),
                    "terminal_evidence": terminal_evidence,
                    "metric_improvement": float(improvement) if isinstance(improvement, (int, float)) else 0.0,
                    "audit_status": (child.get("leakage_audit") or {}).get("status"),
                    "rank_eligible": (child.get("leakage_audit") or {}).get("rank_eligible"),
                    "code_sha256": child.get("code_sha256"),
                    "implementation_available": bool(
                        child.get("implementation_capsule")
                    ),
                    "eligibility_reason": reason,
                }
            )
        def support_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
            direction = str(item.get("metric_direction") or metric_direction or "unknown")
            metric = float(item.get("metric") or 0.0)
            direction_aware_metric = -metric if direction == "maximize" else metric
            if direction not in {"minimize", "maximize"}:
                direction_aware_metric = 0.0
            return (
                0 if item.get("terminal_evidence") is True else 1,
                direction_aware_metric,
                -float(item.get("metric_improvement") or 0.0),
                str(item.get("node_id")),
            )

        return sorted(rows, key=support_key)

    def _rank_strategy_routes(
        self,
        *,
        query_text: str,
        task_profile: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query_text)
        task_family = str(task_profile["task_family"])
        excluded = set(task_profile["excluded_method_families"])
        gpu_count = int(task_profile["resource_budget"].get("gpu_count") or 0)
        rows: list[dict[str, Any]] = []
        visibility_ids = self._effective_visibility_sop_ids()
        strategy_sop_ids = self._recipe_sop_ids or self._sops
        for sop_id in strategy_sop_ids:
            if visibility_ids is not None and sop_id not in visibility_ids:
                continue
            node = self.nodes[sop_id]
            if node.get("abstraction_level") != "L1_strategy" or node.get("sop_kind") != "model_strategy":
                continue
            if "draft" not in (node.get("decision_stages") or []):
                continue
            recipe_task = str(node.get("recipe_task_id") or "")
            if recipe_task and canonical_task_id(recipe_task) != canonical_task_id(
                task_profile["task_id"]
            ):
                continue
            family = str(node.get("method_family") or "")
            if not family or family in excluded:
                continue
            task_families = {str(value) for value in (node.get("task_families") or [])}
            task_fit = 1.0 if task_family in task_families else 0.5 if "general" in task_families else 0.0
            if task_fit == 0.0:
                continue
            supports = self._strategy_supports(
                sop_id,
                str(task_profile["task_id"]),
                metric_direction=str(task_profile.get("metric_direction") or "unknown"),
            )
            if not supports:
                continue
            visible_text = self._visible_sop_prompt(sop_id)
            semantic = self._token_overlap(
                query_tokens,
                _tokenize(visible_text)
                if self._visibility_is_enforced()
                else self._node_tokens.get(sop_id, set()),
            )
            evidence = min(1.0, math.log1p(len(supports)) / math.log(4.0))
            best_improvement = max(float(item.get("metric_improvement") or 0.0) for item in supports)
            rows.append(
                {
                    "sop_id": sop_id,
                    "raw_sop_id": node.get("sop_id"),
                    "title": (
                        node.get("title")
                        if not self._visibility_is_enforced()
                        else "Authorized strategy clauses"
                    ),
                    "action": (
                        node.get("action")
                        if not self._visibility_is_enforced()
                        else visible_text
                    ),
                    "visible_clause_ids": list(
                        (self._visibility_projection(sop_id) or {}).get("clause_ids") or []
                    ),
                    "method_family": family,
                    "task_families": sorted(task_families),
                    "compute_profile": node.get("compute_profile"),
                    "clean_support_count": len(supports),
                    "best_tree_evidence": supports[0],
                    "all_clean_supports": supports[:8],
                    "score_components": {
                        "task_fit": task_fit,
                        "semantic_fit": semantic,
                        "clean_evidence": evidence,
                        "task_local_improvement": best_improvement,
                        "compute_fit": self._compute_fit(str(node.get("compute_profile") or ""), gpu_count),
                    },
                }
            )
        improvement_values = sorted(
            {
                float(row["score_components"]["task_local_improvement"])
                for row in rows
            },
            reverse=True,
        )
        improvement_rank = {
            value: (1.0 if len(improvement_values) == 1 else 1.0 - index / (len(improvement_values) - 1))
            for index, value in enumerate(improvement_values)
        }
        for row in rows:
            raw = float(row["score_components"]["task_local_improvement"])
            row["task_local_improvement_raw"] = raw
            row["score_components"]["task_local_improvement"] = improvement_rank.get(raw, 0.0)
            row["score"] = sum(
                STRATEGY_SCORE_WEIGHTS[key] * float(row["score_components"][key])
                for key in STRATEGY_SCORE_WEIGHTS
            )
        terminal_rows = [
            row
            for row in rows
            if row["best_tree_evidence"].get("terminal_evidence") is True
        ]
        terminal_best_sop_id = ""
        if terminal_rows:
            reverse = str(task_profile.get("metric_direction") or "") == "maximize"
            terminal_best = sorted(
                terminal_rows,
                key=lambda item: (
                    -float(item["best_tree_evidence"]["metric"])
                    if reverse
                    else float(item["best_tree_evidence"]["metric"]),
                    str(item["sop_id"]),
                ),
            )[0]
            terminal_best_sop_id = str(terminal_best["sop_id"])
        for row in rows:
            is_terminal_best = bool(
                terminal_best_sop_id and row["sop_id"] == terminal_best_sop_id
            )
            row["same_task_terminal_best"] = is_terminal_best
            row["selection_priority"] = (
                "mandatory_same_task_terminal_best"
                if is_terminal_best
                else "agent_comparison_candidate"
            )
        rows.sort(key=lambda item: (-float(item["score"]), str(item["sop_id"])))
        if terminal_best_sop_id:
            rows.sort(key=lambda item: 0 if item["same_task_terminal_best"] else 1)
        distinct = []
        seen_families: set[str] = set()
        for row in rows:
            if row["method_family"] in seen_families:
                continue
            seen_families.add(row["method_family"])
            distinct.append(row)
            if len(distinct) >= self.strategy_candidate_limit:
                break
        return distinct

    def _strategy_function_spec(self):
        from llm import FunctionSpec

        return FunctionSpec(
            name="select_novel_strategy_route",
            description="Choose exactly one supplied clean Novel Draft strategy route.",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "strategy_sop_id": {"type": "string", "maxLength": 256},
                    "method_family": {"type": "string", "maxLength": 128},
                    "hypothesis": {"type": "string", "maxLength": 600},
                    "validation_plan": {"type": "string", "maxLength": 600},
                    "model_components": {
                        "type": "array",
                        "maxItems": 12,
                        "items": {"type": "string", "maxLength": 160},
                    },
                    "reason": {"type": "string", "maxLength": 600},
                },
                "required": [
                    "strategy_sop_id",
                    "method_family",
                    "hypothesis",
                    "validation_plan",
                    "model_components",
                    "reason",
                ],
            },
        )

    def _call_strategy_selector(
        self,
        *,
        task_profile: dict[str, Any],
        routes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self._injected_strategy_selector is not None:
            return self._injected_strategy_selector(task_profile=task_profile, routes=routes)
        if self.cfg is None:
            raise RuntimeError("cfg is required for agentic strategy selection")
        from llm import query

        model = getattr(self.cfg.agent.feedback, "model", None) or getattr(self.cfg.agent.code, "model", "")
        compact_routes = [
            {
                key: route.get(key)
                for key in (
                    "sop_id",
                    "method_family",
                    "title",
                    "action",
                    "hypothesis",
                    "model_components",
                    "compute_profile",
                    "score",
                    "same_task_terminal_best",
                    "selection_priority",
                )
            }
            | {
                "best_clean_evidence": {
                    key: route.get("best_tree_evidence", {}).get(key)
                    for key in (
                        "node_id",
                        "metric",
                        "metric_direction",
                        "metric_provenance",
                        "terminal_evidence",
                    )
                }
            }
            for route in routes
        ]
        return query(
            system_message=(
                "Select exactly one supplied strategy. Do not invent a method family or SOP id. "
                "Prefer a task-appropriate, compute-feasible hypothesis that differs from excluded families. "
                "If one route is marked mandatory_same_task_terminal_best and its compute fit is positive, "
                "you must select it: a sealed same-task terminal result outranks speculative reasoning "
                "from historical internal validation scores."
            ),
            user_message=json.dumps(
                {"task_profile": task_profile, "routes": compact_routes},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            model=model,
            temperature=0.0,
            max_tokens=bounded_selector_max_tokens(self.cfg),
            func_spec=self._strategy_function_spec(),
            cfg=self.cfg,
        )

    def _select_strategy(
        self,
        *,
        task_profile: dict[str, Any],
        routes: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        valid = {str(route["sop_id"]): route for route in routes}
        mandatory_terminal = next(
            (
                route
                for route in routes
                if route.get("same_task_terminal_best") is True
                and float(route.get("score_components", {}).get("compute_fit") or 0.0) > 0.0
            ),
            None,
        )
        calls = 0
        last_error = ""
        if self.agentic_enabled:
            for _attempt in range(2):
                calls += 1
                try:
                    result = self._call_strategy_selector(task_profile=task_profile, routes=routes)
                    sop_id = str(result.get("strategy_sop_id") or "")
                    route = valid.get(sop_id)
                    if route is None:
                        raise ValueError(f"unknown strategy_sop_id {sop_id}")
                    if (
                        mandatory_terminal is not None
                        and route["sop_id"] != mandatory_terminal["sop_id"]
                    ):
                        raise ValueError(
                            "selector bypassed mandatory same-task terminal-best route "
                            f"{mandatory_terminal['sop_id']}"
                        )
                    if str(result.get("method_family") or "") != route["method_family"]:
                        raise ValueError("selector method_family does not match supplied route")
                    selected = copy.deepcopy(route)
                    selected["decision"] = {
                        key: result[key]
                        for key in ("hypothesis", "validation_plan", "model_components", "reason")
                    }
                    return selected, {"mode": "llm_validated", "llm_tool_calls": calls}
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning("[LayeredStrategy] strategy selector attempt failed: %s", exc)
        selected = copy.deepcopy(routes[0])
        selected["decision"] = {
            "hypothesis": str(selected.get("action") or selected.get("title") or ""),
            "validation_plan": "Evaluate once on the task's clean validation protocol.",
            "model_components": [str(selected.get("method_family"))],
            "reason": "Highest deterministic eligible strategy score.",
        }
        mode = "deterministic_fallback" if calls else "deterministic"
        return selected, {"mode": mode, "llm_tool_calls": calls, "last_error": last_error}

    def _layered_draft_pack(
        self,
        *,
        task_id: str,
        task_desc: str,
        query_text: str,
        context: dict[str, Any] | None,
        visibility_request: VisibilityRequest | None = None,
        authority_operation: Operation | str | None = None,
        active_protocol: ProtocolRef | str | None = None,
    ) -> dict[str, Any]:
        self._prepare_visibility(
            stage="draft",
            task_id=task_id,
            task_desc=task_desc,
            request=visibility_request,
            operation=authority_operation,
            active_protocol=active_protocol,
        )
        task_profile = self._build_task_profile(task_id=task_id, task_desc=task_desc, context=context)
        candidates = self._rank_strategy_routes(query_text=query_text, task_profile=task_profile)
        routes = candidates[: self.strategy_route_count]
        if len(routes) < self.strategy_route_count:
            reason = (
                "insufficient_strategy_coverage: "
                f"task={task_id} eligible_distinct_families={len(routes)} "
                f"required={self.strategy_route_count}"
            )
            fallback = self._hybrid_pack(
                stage="draft",
                task_id=task_id,
                task_desc=task_desc,
                query_text=query_text,
                visibility_request=visibility_request,
                authority_operation=authority_operation,
                active_protocol=active_protocol,
            )
            fallback["task_profile"] = task_profile
            fallback["layered_strategy_fallback"] = {
                "activated": True,
                "reason": reason,
                "eligible_strategy_routes": routes,
                "fallback_algorithm_version": fallback.get("algorithm_version"),
                "fallback_mode": "stage_hybrid_v2_clean_cross_task",
            }
            fallback["stage_route"]["requested_control"] = "layered_strategy"
            fallback["stage_route"]["fallback_reason"] = reason
            return fallback
        selected, selection = self._select_strategy(task_profile=task_profile, routes=routes)
        trace = []
        for route in routes:
            is_selected = route["sop_id"] == selected["sop_id"]
            trace.append(
                {
                    "retrieval_channel": "l1_strategy",
                    "candidate_class": "strategy_route",
                    "gateway_sop_id": route["sop_id"],
                    "candidate_id": route["sop_id"],
                    "supporting_transition_ids": [
                        route["best_tree_evidence"]["transition_id"]
                    ]
                    if route["best_tree_evidence"].get("transition_id")
                    else [],
                    "selection_reason": (
                        selected["decision"]["reason"] if is_selected else "Rejected after three-route strategy comparison."
                    ),
                    "selection_state": "selected" if is_selected else "rejected",
                    "method_family": route["method_family"],
                    "expanded_candidate_ids": [route["best_tree_evidence"]["node_id"]],
                }
            )
        evidence = selected["best_tree_evidence"]
        trace.append(
            {
                "retrieval_channel": "tree_evidence_expansion",
                "candidate_class": "clean_strategy_evidence",
                "gateway_sop_id": selected["sop_id"],
                "candidate_id": evidence["node_id"],
                "supporting_transition_ids": [evidence["transition_id"]]
                if evidence.get("transition_id")
                else [],
                "selection_reason": "Best clean successful task-local Tree evidence for the selected method family.",
                "selection_state": "injected",
                "method_family": selected["method_family"],
            }
        )
        return {
            "schema": LAYERED_PACK_SCHEMA,
            "stage_route": {"stage": "draft", "route": "l1_strategy_then_tree", "control": "layered_strategy"},
            "task_profile": task_profile,
            "strategy_score_weights": STRATEGY_SCORE_WEIGHTS,
            "strategy_candidates": candidates,
            "strategy_routes": routes,
            "selected_strategy": selected,
            "strategy_selection": selection,
            "mandatory_same_task_terminal_strategy": next(
                (
                    route["sop_id"]
                    for route in routes
                    if route.get("same_task_terminal_best") is True
                ),
                None,
            ),
            "excluded_method_families": task_profile["excluded_method_families"],
            "l2_tactics": [],
            "navigation_trace": trace,
            "risk_warnings": [],
        }

    def _format_selected_strategy(self, pack: dict[str, Any]) -> str:
        selected = pack["selected_strategy"]
        evidence = selected["best_tree_evidence"]
        decision = selected.get("decision") or {
            "hypothesis": selected.get("action") or selected.get("title") or "",
            "validation_plan": "Use the task's clean validation protocol.",
            "model_components": [selected.get("method_family") or ""],
        }
        lines = [
                "## Frozen Novel Strategy Contract",
                "The strategy was selected from three distinct L1 method families with clean Tree evidence.",
                f"Task profile: {json.dumps(pack['task_profile'], ensure_ascii=False)}",
                f"Selected SOP: {selected['sop_id']} - {selected.get('title', '')}",
                f"Primary method family: {selected['method_family']}",
                f"Action: {selected.get('action', '')}",
                f"Hypothesis: {decision.get('hypothesis', '')}",
                f"Validation plan: {decision.get('validation_plan', '')}",
                f"Model components: {', '.join(decision.get('model_components') or [])}",
                f"Clean RunForest evidence: run={evidence.get('run_id')} node={evidence.get('node_id')} "
                f"kind={evidence.get('evidence_kind', 'supporting_transition')} "
                f"transition={evidence.get('transition_id') or 'direct'} metric={evidence.get('metric')} "
                f"direction={evidence.get('metric_direction')} provenance={evidence.get('metric_provenance')} "
                f"terminal={evidence.get('terminal_evidence')} audit={evidence.get('audit_status')} "
                f"code_sha256={evidence.get('code_sha256')}",
                "Do not replace this method family with an excluded baseline/replay family.",
            ]
        implementation = self._format_node_implementation(
            str(evidence.get("node_id") or ""),
            heading="Exact Same-Task RunForest Implementation",
        )
        if implementation:
            lines.extend(
                [
                    "",
                    "Start from this measured implementation. Preserve its data, fold, "
                    "training, and inference mechanics unless the selected SOP explicitly "
                    "requires a named change.",
                    implementation,
                ]
            )
        return "\n".join(lines)

    def _format_node_implementation(self, node_id: str, *, heading: str) -> str:
        node = self.nodes.get(str(node_id), {})
        capsule = node.get("implementation_capsule")
        if not isinstance(capsule, Mapping):
            return ""
        code = str(capsule.get("code") or "")
        if not code.strip():
            return ""
        return "\n".join(
            [
                f"### {heading}",
                f"RunForest node: {node_id}",
                f"Code identity: {capsule.get('code_sha256')}",
                "<implementation_code>",
                code,
                "</implementation_code>",
            ]
        )

    def retrieve_model_design_tactics(
        self,
        *,
        task_id: str,
        task_desc: str,
        strategy_context: dict[str, Any],
    ) -> tuple[str, list[str], dict[str, Any]]:
        self._prepare_visibility(
            stage="model_design",
            task_id=task_id,
            task_desc=task_desc,
        )
        selected = strategy_context.get("selected_strategy") or strategy_context
        task_profile = strategy_context.get("task_profile") or {}
        family = str(selected.get("method_family") or "")
        if not family:
            raise ValueError("L2 retrieval requires a selected L1 method_family")
        query_text = "\n".join(
            [task_desc, str(selected.get("title") or ""), str(selected.get("action") or ""), family]
        )
        query_tokens = _tokenize(query_text)
        task_family = str(task_profile.get("task_family") or TASK_PROFILES.get(task_id, ("", "general"))[1])
        rows = []
        visibility_ids = self._effective_visibility_sop_ids()
        tactic_sop_ids = self._recipe_sop_ids or self._sops
        for sop_id in tactic_sop_ids:
            if visibility_ids is not None and sop_id not in visibility_ids:
                continue
            node = self.nodes[sop_id]
            if node.get("abstraction_level") != "L2_tactic":
                continue
            if "model_design" not in (node.get("decision_stages") or []):
                continue
            if node.get("sop_kind") not in {"architecture", "feature", "training_protocol", "validation_protocol"}:
                continue
            recipe_task = str(node.get("recipe_task_id") or "")
            if recipe_task and canonical_task_id(recipe_task) != canonical_task_id(task_id):
                continue
            node_family = str(node.get("method_family") or "general")
            parent_families = {
                str(value) for value in (node.get("parent_method_families") or [])
            }
            if parent_families:
                if family not in parent_families:
                    continue
            elif not self._family_compatible(family, node_family):
                continue
            families = {str(value) for value in (node.get("task_families") or [])}
            if task_family not in families and "general" not in families:
                continue
            supports = self._strategy_supports(sop_id, task_id)
            if not supports:
                continue
            visible_text = self._visible_sop_prompt(sop_id)
            semantic = self._token_overlap(
                query_tokens,
                _tokenize(visible_text)
                if self._visibility_is_enforced()
                else self._node_tokens.get(sop_id, set()),
            )
            score = 0.55 * semantic + 0.25 * min(1.0, len(supports) / 3.0) + 0.20 * (1.0 if node_family == family else 0.5)
            rows.append(
                {
                    "sop_id": sop_id,
                    "title": (
                        node.get("title")
                        if not self._visibility_is_enforced()
                        else "Authorized tactic clauses"
                    ),
                    "action": (
                        node.get("action")
                        if not self._visibility_is_enforced()
                        else visible_text
                    ),
                    "visible_clause_ids": list(
                        (self._visibility_projection(sop_id) or {}).get("clause_ids") or []
                    ),
                    "sop_kind": node.get("sop_kind"),
                    "method_family": node_family,
                    "score": score,
                    "best_tree_evidence": supports[0],
                }
            )
        rows.sort(key=lambda item: (-float(item["score"]), str(item["sop_id"])))
        selected_tactics = []
        seen_kinds: set[str] = set()
        for row in rows:
            if row["sop_kind"] in seen_kinds and len(selected_tactics) < min(3, self.l2_tactic_limit):
                continue
            selected_tactics.append(row)
            seen_kinds.add(str(row["sop_kind"]))
            if len(selected_tactics) >= self.l2_tactic_limit:
                break
        trace = []
        refs = []
        for tactic in selected_tactics:
            evidence = tactic["best_tree_evidence"]
            refs.extend(
                [
                    tactic["sop_id"],
                    *([evidence["transition_id"]] if evidence.get("transition_id") else []),
                    evidence["node_id"],
                ]
            )
            trace.append(
                {
                    "retrieval_channel": "l2_model_design",
                    "candidate_class": "family_compatible_tactic",
                    "gateway_sop_id": tactic["sop_id"],
                    "candidate_id": tactic["sop_id"],
                    "supporting_transition_ids": [evidence["transition_id"]]
                    if evidence.get("transition_id")
                    else [],
                    "selection_reason": f"L2 {tactic['sop_kind']} compatible with {family}.",
                    "selection_state": "injected",
                    "method_family": family,
                    "expanded_candidate_ids": [evidence["node_id"]],
                }
            )
        pack = {
            "schema": "layered_model_design_tactics_v1",
            "method_family": family,
            "task_profile": task_profile,
            "selected_tactics": selected_tactics,
            "navigation_trace": trace,
        }
        current = self.current_navigation_pack()
        current["l2_tactics"] = selected_tactics
        current["l2_navigation_trace"] = trace
        current["navigation_trace"] = list(current.get("navigation_trace") or []) + trace
        self._trace_local.pack = current
        lines = [self._format_selected_strategy(strategy_context), "", "## L2 Model-Design Tactics"]
        if not selected_tactics:
            lines.append("No clean family-compatible L2 tactic was available; implement only the frozen L1 strategy.")
        for tactic in selected_tactics:
            evidence = tactic["best_tree_evidence"]
            lines.append(f"- {tactic['sop_id']} [{tactic['sop_kind']}]: {tactic.get('title', '')}")
            lines.append(f"  Action: {tactic.get('action', '')}")
            lines.append(
                f"  Clean evidence: {evidence.get('run_id')} / {evidence.get('node_id')} / metric={evidence.get('metric')}"
            )
        return "\n".join(lines), list(dict.fromkeys(refs)), pack

    def _positive_transition(self, transition_id: str) -> tuple[bool, str]:
        transition = self.nodes.get(transition_id, {})
        if transition.get("type") != "Transition":
            return False, "not_transition"
        # Atomic Debug claims have their own local evidence and taint gate.
        # They never authorize the source program or its metric, so evaluating
        # the whole child program here would recreate the all-or-nothing bug
        # that v7 is designed to remove.
        if transition.get("atomic_repair_claim") is not None:
            return verified_atomic_debug_claim(transition)
        run_id = str(transition.get("run_short_id") or transition.get("run_id") or "")
        if run_id in self.excluded_run_ids:
            return False, "held_out_run"
        fast_nonblocking = bool(
            self.memory_snapshot is not None
            and getattr(self.memory_snapshot, "verify_artifacts", True) is False
            and str(getattr(getattr(self.cfg, "evaluation_authority", None), "mode", "") or "").lower()
            == "off"
        )
        child = self.nodes.get(str(transition.get("child_node_id") or ""), {})
        if fast_nonblocking:
            metric = child.get("metric")
            outcome = str(transition.get("outcome") or "")
            if not self._positive_memory_eligible(child):
                audit = (
                    child.get("leakage_audit")
                    if isinstance(child.get("leakage_audit"), dict)
                    else {}
                )
                return False, str(
                    audit.get("memory_disposition")
                    or audit.get("status")
                    or "child_execution_not_successful"
                )
            if not (
                child.get("is_buggy") is False
                and child.get("is_valid") is True
                and isinstance(metric, (int, float))
                and not isinstance(metric, bool)
                and math.isfinite(float(metric))
            ):
                return False, "child_execution_not_successful"
            if outcome in {"buggy", "metric_worsened", "unknown"}:
                return False, f"transition_outcome_{outcome or 'missing'}"
            return True, "experiment_fast_successful_transition"
        if any(run_id.startswith(prefix) for prefix in self._blocked_run_prefixes):
            return False, "blocked_run_prefix"
        if transition.get("quarantined") is True or transition.get("protocol_biased") is True:
            return False, "transition_quarantined_or_protocol_biased"
        if not self._positive_memory_eligible(child):
            audit = child.get("leakage_audit") if isinstance(child.get("leakage_audit"), dict) else {}
            return False, str(audit.get("memory_disposition") or audit.get("status") or "child_not_code_audited_clean")
        if child.get("quarantined") is True or child.get("protocol_biased") is True:
            return False, "child_quarantined_or_protocol_biased"
        audit = child.get("leakage_audit") if isinstance(child.get("leakage_audit"), dict) else {}
        if audit.get("rank_eligible") is not True:
            return False, "child_not_rank_eligible"
        if child.get("is_buggy") is not False or child.get("is_valid") is not True:
            return False, "child_execution_not_successful"
        metric = child.get("metric")
        if isinstance(metric, bool) or not isinstance(metric, (int, float)) or not math.isfinite(float(metric)):
            return False, "child_metric_missing"
        outcome = str(transition.get("outcome") or "")
        if outcome in {"buggy", "metric_worsened", "unknown"}:
            return False, f"transition_outcome_{outcome or 'missing'}"
        return True, "code_audited_clean_success"

    def _successful_run_node(self, node_id: str) -> bool:
        node = self.nodes.get(node_id, {})
        if node.get("type") != "RunNode" or not self._positive_memory_eligible(node):
            return False
        audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
        metric = node.get("metric")
        run_id = str(node.get("run_short_id") or node.get("run_id") or "")
        fast_nonblocking = bool(
            self.memory_snapshot is not None
            and getattr(self.memory_snapshot, "verify_artifacts", True) is False
            and str(getattr(getattr(self.cfg, "evaluation_authority", None), "mode", "") or "").lower()
            == "off"
        )
        return bool(
            (fast_nonblocking or audit.get("rank_eligible") is True)
            and node.get("is_buggy") is False
            and node.get("is_valid") is True
            and isinstance(metric, (int, float))
            and not isinstance(metric, bool)
            and math.isfinite(float(metric))
            and run_id not in self.excluded_run_ids
            and (
                fast_nonblocking
                or not any(run_id.startswith(prefix) for prefix in self._blocked_run_prefixes)
            )
        )

    def _sop_text_parts(self, node: dict[str, Any]) -> dict[str, str]:
        return {
            "semantic": " ".join(str(node.get(key) or "") for key in ("title", "action", "text")),
            "conditions": " ".join(_as_list(node.get("applies_when")) + _as_list(node.get("condition"))),
            "failures": " ".join(_as_list(node.get("prevents")) + _as_list(node.get("failure_modes"))),
            "evidence": " ".join(_as_list(node.get("evidence_turns")) + _as_list(node.get("source_branches"))),
        }

    def _task_family_for_query(self, task_id: str, task_desc: str) -> str:
        task_id = canonical_task_id(task_id)
        configured = TASK_PROFILES.get(task_id)
        if configured:
            return configured[1]
        text = f"{task_id} {task_desc}".lower()
        if any(token in text for token in ("text classification", "nlp", "author", "sentiment")):
            return "text_classification"
        if "restoration" in text or "denois" in text:
            return "image_restoration"
        if "image" in text and any(token in text for token in ("binary", "two class", "2 class")):
            return "image_binary_classification"
        if "image" in text:
            return "image_classification"
        if "regression" in text or "rmse" in text or "mae" in text:
            return "tabular_regression"
        if "tabular" in text or "multiclass" in text:
            return "tabular_multiclass"
        return "general"

    def _sop_stage_fit(self, node: dict[str, Any], stage: str) -> tuple[float, bool]:
        declared = {str(value) for value in (node.get("decision_stages") or [])}
        if not declared:
            return 0.50, True
        compatible = bool(declared & STAGE_DECISION_TAGS[stage])
        return (1.0 if compatible else 0.0), compatible

    def _sop_task_fit(self, node: dict[str, Any], task_family: str) -> float:
        if node.get("abstraction_level") == "L3_repair" and node.get("task_type"):
            declared = {str(value) for value in (node.get("task_families") or [])}
            if task_family in declared:
                return 1.0
            query_type = self._task_type_for_family(task_family)
            node_type = str(node.get("task_type") or "")
            return 0.70 if query_type != "general" and query_type == node_type else 0.0
        if self.domain_scope_required:
            return 1.0 if self._sop_task_compatible(node, task_family) else 0.0
        declared = {str(value) for value in (node.get("task_families") or [])}
        if task_family == "general":
            return 0.50
        if not declared:
            return 0.50
        if task_family in declared:
            return 1.0
        if "general" in declared:
            return 0.60
        query_tokens = _tokenize(task_family.replace("_", " "))
        best = max(
            (min(1.0, self._token_overlap(query_tokens, _tokenize(value.replace("_", " ")))) for value in declared),
            default=0.0,
        )
        return 0.35 * best

    def _sop_task_compatible(self, node: dict[str, Any], task_family: str) -> bool:
        if node.get("abstraction_level") == "L3_repair" and node.get("task_type"):
            declared = {str(value) for value in (node.get("task_families") or [])}
            if task_family in declared:
                return True
            query_type = self._task_type_for_family(task_family)
            node_type = str(node.get("task_type") or "")
            return bool(
                query_type != "general"
                and node_type
                and query_type == node_type
            )
        if self.domain_scope_required:
            scopes = {
                normalize_transfer_scope(value)
                for value in (
                    node.get("transfer_scopes")
                    or [node.get("transfer_scope")]
                )
            }
            scopes.discard("")
            if DOMAIN_GENERAL in scopes:
                return True
            if scopes != {SAME_DOMAIN}:
                return False
            source_domains = {
                canonical_domain(value)
                for value in (
                    node.get("source_domains")
                    or node.get("source_task_families")
                    or []
                )
            }
            source_domains.discard("")
            return transfer_is_compatible(
                source_domains,
                canonical_domain(task_family),
                SAME_DOMAIN,
            )
        declared = {str(value) for value in (node.get("task_families") or [])}
        if task_family == "general" or not declared or "general" in declared or task_family in declared:
            return True
        query_tokens = _tokenize(task_family.replace("_", " "))
        return any(
            self._token_overlap(query_tokens, _tokenize(value.replace("_", " "))) >= 0.50
            for value in declared
        )

    def _clean_sop_support(self, sop_id: str) -> tuple[list[str], list[dict[str, str]]]:
        clean: list[str] = []
        rejected: list[dict[str, str]] = []
        for transition_id in self._active_transitions_for_sop(sop_id):
            eligible, reason = self._positive_transition(transition_id)
            if eligible:
                clean.append(transition_id)
            else:
                rejected.append({"transition_id": transition_id, "reason": reason})
        return clean, rejected

    def _rank_sops(
        self,
        query_text: str,
        stage: str,
        limit: int,
        *,
        allowed_levels: set[str] | None = None,
        method_family: str | None = None,
        task_id: str = "",
        task_desc: str = "",
        allowed_sop_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGE_SOP_FIELD_WEIGHTS:
            raise ValueError(f"Unsupported SOP ranking stage: {stage}")
        query_tokens = _tokenize(query_text)
        task_family = self._task_family_for_query(task_id, task_desc)
        visibility_ids = self._effective_visibility_sop_ids()
        candidate_ids = [
            sop_id
            for sop_id in self._sops
            if (allowed_sop_ids is None or sop_id in allowed_sop_ids)
            and (visibility_ids is None or sop_id in visibility_ids)
        ]
        coords = self._coords()
        geometry_candidate_ids = candidate_ids
        if self._visibility_is_enforced():
            geometry_candidate_ids = []
            for sop_id in candidate_ids:
                projection = self._visibility_projection(sop_id) or {}
                if self._container_embedding_visibility_safe(sop_id, projection):
                    geometry_candidate_ids.append(sop_id)
        anchor = self._query_anchor(query_text, geometry_candidate_ids)
        field_weights = STAGE_SOP_FIELD_WEIGHTS[stage]
        rows = []
        for sop_id in candidate_ids:
            node = self.nodes[sop_id]
            if allowed_levels and str(node.get("abstraction_level") or "") not in allowed_levels:
                continue
            node_family = str(node.get("method_family") or "general")
            if method_family and not self._family_compatible(method_family, node_family):
                continue
            parts = self._visible_sop_text_parts(sop_id, node)
            scores = {
                key: min(1.0, self._token_overlap(query_tokens, _tokenize(text)))
                for key, text in parts.items()
            }
            field_relevance = sum(field_weights[key] * scores[key] for key in field_weights)
            stage_fit, stage_compatible = self._sop_stage_fit(node, stage)
            task_fit = self._sop_task_fit(node, task_family)
            task_compatible = self._sop_task_compatible(node, task_family)
            geometry = 0.0
            projection = self._visibility_projection(sop_id)
            geometry_safe = (
                not self._visibility_is_enforced()
                or self._container_embedding_visibility_safe(sop_id, projection)
            )
            if geometry_safe and anchor is not None and sop_id in coords:
                geometry = 1.0 / (1.0 + self._distance(anchor, coords[sop_id]))
            clean, rejected = self._clean_sop_support(sop_id)
            clean_evidence = min(1.0, len(clean) / 3.0)
            components = {
                "field_relevance": field_relevance,
                "stage_fit": stage_fit,
                "task_fit": task_fit,
                "geometry": geometry,
                "clean_evidence": clean_evidence,
            }
            score = sum(
                SOP_HYBRID_SCORE_WEIGHTS[key] * components[key]
                for key in SOP_HYBRID_SCORE_WEIGHTS
            )
            rows.append(
                {
                    "id": sop_id,
                    "score": score,
                    "score_components": scores,
                    "hybrid_score_components": components,
                    "ranking_backend": "stage_task_geometry_field_hybrid_v2",
                    "abstraction_level": node.get("abstraction_level"),
                    "sop_kind": node.get("sop_kind"),
                    "method_family": node_family,
                    "decision_stages": list(node.get("decision_stages") or []),
                    "task_families": list(node.get("task_families") or []),
                    "stage_compatible": stage_compatible,
                    "task_compatible": task_compatible,
                    "task_family": task_family,
                    "clean_supporting_transition_ids": clean[:8],
                    "clean_supporting_transition_count": len(clean),
                    "rejected_support": rejected[:8],
                    "rejected_support_count": len(rejected),
                    "visible_clause_ids": list((projection or {}).get("clause_ids") or []),
                    "visible_text": self._visible_sop_prompt(sop_id),
                    "geometry_visibility_safe": geometry_safe,
                }
            )
        rows.sort(key=lambda item: (-item["score"], item["id"]))
        return rows[:limit]

    def _gateway_function_spec(self):
        from llm import FunctionSpec

        return FunctionSpec(
            name="select_stage_hybrid_sop_gateways",
            description="Select eligible SOP gateway IDs for the current stage.",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "gateway_ids": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string", "maxLength": 256},
                    },
                    "reasons": {
                        "type": "object",
                        "additionalProperties": {
                            "type": "string",
                            "maxLength": 400,
                        },
                    },
                    "goal": {"type": "string", "maxLength": 500},
                },
                "required": ["gateway_ids", "reasons", "goal"],
            },
        )

    def _call_gateway_selector(self, *, stage: str, query_text: str, eligible: list[dict[str, Any]]) -> dict[str, Any]:
        if self._injected_gateway_selector is not None:
            return self._injected_gateway_selector(stage=stage, query_text=query_text, eligible=eligible)
        if self.cfg is None:
            raise RuntimeError("cfg is required for agentic gateway selection")
        from llm import query

        model = getattr(self.cfg.agent.feedback, "model", None) or getattr(self.cfg.agent.code, "model", "")
        compact_eligible = [
            {
                "id": item.get("id"),
                "score": item.get("score"),
                "method_family": item.get("method_family"),
                "decision_stages": item.get("decision_stages"),
                "task_families": item.get("task_families"),
                "clean_supporting_transition_count": item.get(
                    "clean_supporting_transition_count"
                ),
                "visible_clause_ids": item.get("visible_clause_ids"),
                "visible_text": str(item.get("visible_text") or "")[:800],
            }
            for item in eligible
        ]
        return query(
            system_message="Select only supplied clean SOP gateway IDs. Do not invent IDs.",
            user_message=json.dumps(
                {
                    "stage": stage,
                    "query": query_text[-3000:],
                    "eligible": compact_eligible,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            model=model,
            temperature=0.0,
            max_tokens=bounded_selector_max_tokens(self.cfg),
            func_spec=self._gateway_function_spec(),
            cfg=self.cfg,
        )

    def _select_gateways(
        self, candidates: list[dict[str, Any]], *, stage: str, query_text: str, limit: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        clean_eligible = [item for item in candidates if item["clean_supporting_transition_ids"]]
        eligible = [
            item
            for item in clean_eligible
            if item.get("stage_compatible", True) and item.get("task_compatible", True)
        ]
        fallback_ids = [item["id"] for item in eligible[:limit]]
        reasons = {item["id"]: "deterministic eligible score order" for item in eligible[:limit]}
        mode = "deterministic"
        llm_calls = 0
        selected_ids = fallback_ids
        goal = STAGE_ROUTE[stage]
        if self.agentic_enabled and eligible:
            llm_calls = 1
            try:
                result = self._call_gateway_selector(stage=stage, query_text=query_text, eligible=eligible)
                valid_ids = {item["id"] for item in eligible}
                proposed = [str(value) for value in result.get("gateway_ids", [])]
                if not proposed or any(value not in valid_ids for value in proposed):
                    raise ValueError("gateway selector returned missing or ineligible IDs")
                selected_ids = list(dict.fromkeys(proposed))[:limit]
                raw_reasons = result.get("reasons") if isinstance(result.get("reasons"), dict) else {}
                reasons = {node_id: str(raw_reasons.get(node_id, "LLM-selected eligible gateway")) for node_id in selected_ids}
                goal = str(result.get("goal") or goal)
                mode = "llm_validated"
            except Exception as exc:
                mode = "deterministic_fallback"
                logger.warning("[StageHybrid] gateway selector failed; deterministic fallback: %s", exc)
        by_id = {item["id"]: item for item in eligible}
        selected = []
        for node_id in selected_ids:
            row = dict(by_id[node_id])
            row["selection_reason"] = reasons[node_id]
            selected.append(row)
        return selected, {
            "mode": mode,
            "llm_tool_calls": llm_calls,
            "goal": goal,
            "clean_eligible_count": len(clean_eligible),
            "eligible_count": len(eligible),
            "stage_task_gate_rejected_count": len(clean_eligible) - len(eligible),
        }

    def _append_unique(self, output: list[str], node_id: str) -> None:
        if node_id in self.nodes and node_id not in output:
            output.append(node_id)

    def _execution_candidate_eligibility(self, node_id: str) -> tuple[bool, str]:
        node = self.nodes.get(node_id, {})
        if node.get("type") == "Transition":
            if node.get("atomic_repair_claim") is not None:
                # The claim is a Debug hypothesis/repair source, never a
                # replayable whole-program execution candidate.
                return False, "atomic_claim_debug_only"
            return self._positive_transition(node_id)
        if node.get("type") == "RunNode":
            if self._successful_run_node(node_id):
                return True, "clean_successful_run_node"
            return False, "run_node_not_rank_eligible"
        return False, "not_execution_candidate"

    def _expand_gateways(
        self, selected: list[dict[str, Any]]
    ) -> tuple[
        list[str],
        dict[str, list[str]],
        list[str],
        list[str],
        list[dict[str, Any]],
        dict[str, dict[str, Any]],
    ]:
        execution_ids: list[str] = []
        gateway_transitions: dict[str, list[str]] = {}
        evidence_refs: list[str] = []
        failure_patterns: list[str] = []
        trace: list[dict[str, Any]] = []
        provenance: dict[str, dict[str, Any]] = {}
        for gateway in selected:
            sop_id = gateway["id"]
            transitions = list(gateway["clean_supporting_transition_ids"][:2])
            gateway_transitions[sop_id] = transitions
            for transition_id in transitions:
                transition = self.nodes[transition_id]
                expanded_for_transition: list[str] = []
                parent_id = str(transition.get("parent_node_id") or "")
                child_id = str(transition.get("child_node_id") or "")
                local_best = str(self.nodes.get(child_id, {}).get("local_best_node_id") or "")
                proposed_ids = [
                    transition_id,
                    child_id,
                    parent_id,
                    *self._ancestor_path(child_id, max_hops=5),
                    local_best,
                ]
                for node_id in dict.fromkeys(value for value in proposed_ids if value):
                    eligible, reason = self._execution_candidate_eligibility(node_id)
                    if not eligible:
                        continue
                    self._append_unique(execution_ids, node_id)
                    self._append_unique(expanded_for_transition, node_id)
                    record = provenance.setdefault(
                        node_id,
                        {
                            "candidate_id": node_id,
                            "source_channels": [],
                            "gateway_sop_ids": [],
                            "supporting_transition_ids": [],
                            "safety_status": "clean",
                            "safety_reason": reason,
                        },
                    )
                    if "sop_gateway" not in record["source_channels"]:
                        record["source_channels"].append("sop_gateway")
                    if sop_id not in record["gateway_sop_ids"]:
                        record["gateway_sop_ids"].append(sop_id)
                    if transition_id not in record["supporting_transition_ids"]:
                        record["supporting_transition_ids"].append(transition_id)
                evidence_refs.extend(self._evidence_by_transition.get(transition_id, []))
                parent = self.nodes.get(child_id, {}).get("parent_id")
                for sibling_id in self._children_by_node.get(str(parent), []):
                    if sibling_id == child_id:
                        continue
                    if not self._positive_memory_eligible(self.nodes[sibling_id]):
                        failure_patterns.extend(self._failure_patterns_by_source.get(sibling_id, []))
                trace.append(
                    {
                        "retrieval_channel": "sop_gateway",
                        "candidate_class": "sop_transition_matches",
                        "gateway_sop_id": sop_id,
                        "candidate_id": transition_id,
                        "supporting_transition_ids": [transition_id],
                        "selection_reason": gateway["selection_reason"],
                        "selection_state": "expanded",
                        "expanded_candidate_ids": expanded_for_transition,
                    }
                )
        return (
            execution_ids,
            gateway_transitions,
            list(dict.fromkeys(evidence_refs))[:12],
            list(dict.fromkeys(failure_patterns))[:12],
            trace,
            provenance,
        )

    def _bounded_token_similarity(self, left: str, right: str) -> float:
        """Binary-token cosine that cannot saturate merely because a node is long."""
        left_tokens = _tokenize(left)
        right_tokens = _tokenize(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens))

    def _failure_signature(self, text: str) -> set[str]:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
        padded = f" {normalized} "
        signatures = set()
        for name, phrases in FAILURE_SIGNATURES.items():
            if any(f" {re.sub(r'[^a-z0-9]+', ' ', phrase).strip()} " in padded for phrase in phrases):
                signatures.add(name)
        for exception_name in re.findall(r"\b([a-z][a-z0-9_]*(?:error|exception))\b", normalized):
            signatures.add(f"exception:{exception_name}")
        return signatures

    @staticmethod
    def _semantic_failure_tokens(text: str) -> set[str]:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())
        tokens = _tokenize(normalized)
        compact = normalized.replace(" ", "")
        if "perexample" in compact:
            tokens.add("perexample")
        if "persample" in compact:
            tokens.add("persample")
        for group in L3_FAILURE_TOKEN_EQUIVALENCE_GROUPS:
            if tokens & group:
                tokens.update(group)
        return tokens

    def _specific_failure_signature_overlap(
        self,
        signature_ids: str,
        query_text: str,
    ) -> float:
        signature_tokens = _tokenize(
            re.sub(r"[/_-]+", " ", str(signature_ids or ""))
        ) - L3_GENERIC_SIGNATURE_TERMS
        if not signature_tokens:
            return 0.0
        query_tokens = self._semantic_failure_tokens(
            re.sub(r"[/_-]+", " ", str(query_text or ""))
        )
        return min(
            1.0,
            len(signature_tokens & query_tokens) / min(3, len(signature_tokens)),
        )

    def _task_families_compatible(self, left: str, right: str) -> bool:
        if left == right:
            return True
        related = [
            {"image_classification", "image_binary_classification"},
        ]
        return any(left in group and right in group for group in related)

    @staticmethod
    def _task_type_for_family(task_family: str) -> str:
        value = str(task_family or "").lower()
        if value.startswith("text_") or "text_classification" in value:
            return "nlp"
        if value.startswith("image_") or value == "image_classification":
            return "vision"
        if value.startswith("tabular_"):
            return "tabular"
        if value.startswith("audio_"):
            return "audio"
        if value.startswith("multimodal_"):
            return "multimodal"
        return "general"

    def _task_type_for_query(self, task_id: str, task_desc: str = "") -> str:
        """Prefer the registered task modality over its modeling family.

        Leaf is registered as multimodal even though ``tabular_multiclass`` is
        its useful ranking family.  That ranking label must not authorize Taxi
        repair memories for a Leaf Debug decision.
        """

        canonical = canonical_task_id(task_id)
        return TASK_TYPES.get(canonical) or self._task_type_for_family(
            self._task_family_for_query(canonical, task_desc)
        )

    def _debug_transition_task_fit(
        self,
        transition: dict[str, Any],
        *,
        task_id: str,
        task_family: str,
    ) -> float:
        source_task = canonical_task_id(transition.get("task"))
        task_id = canonical_task_id(task_id)
        if source_task and source_task == task_id:
            return 1.0
        query_type = TASK_TYPES.get(task_id) or self._task_type_for_family(task_family)
        source_type = TASK_TYPES.get(source_task)
        if source_type and query_type and source_type == query_type:
            return 0.70
        return 0.0

    def _task_score(self, node: dict[str, Any], task_id: str, task_desc: str) -> float:
        canonical_node = dict(node)
        canonical_node["task"] = canonical_task_id(node.get("task"))
        return super()._task_score(canonical_node, canonical_task_id(task_id), task_desc)

    def _debug_parent_failure_text(self, transition: dict[str, Any]) -> str:
        claim = transition.get("atomic_repair_claim")
        if isinstance(claim, Mapping):
            return str(claim.get("failure_text") or "")
        parent = self.nodes.get(str(transition.get("parent_node_id") or ""), {})
        return " ".join(
            str(parent.get(key) or "")
            for key in ("analysis", "terminal_excerpt", "plan", "code_summary", "text")
        )

    def _debug_transition_evidence(self, transition: dict[str, Any]) -> dict[str, str]:
        claim = transition.get("atomic_repair_claim")
        if isinstance(claim, Mapping):
            verification = (
                claim.get("verification")
                if isinstance(claim.get("verification"), Mapping)
                else {}
            )
            return {
                "parent_failure": str(claim.get("failure_text") or ""),
                "code_change": str(claim.get("repair_action") or ""),
                "child_result": (
                    "Atomic repair action was followed by an observed successful "
                    "execution; the source program and metric remain separately gated."
                ),
                "before_code": "",
                "after_code": "",
                "before_code_sha256": str(
                    verification.get("before_code_sha256") or ""
                ),
                "after_code_sha256": str(
                    verification.get("after_code_sha256") or ""
                ),
                "unified_diff": "",
            }
        parent = self.nodes.get(str(transition.get("parent_node_id") or ""), {})
        child = self.nodes.get(str(transition.get("child_node_id") or ""), {})
        parent_failure = str(parent.get("analysis") or parent.get("terminal_excerpt") or "").strip()
        code_change = str(child.get("plan") or child.get("code_summary") or transition.get("text") or "").strip()
        child_result = str(child.get("analysis") or child.get("terminal_excerpt") or "").strip()
        repair = transition.get("implementation_repair_capsule")
        repair = repair if isinstance(repair, Mapping) else {}
        return {
            "parent_failure": parent_failure,
            "code_change": code_change,
            "child_result": child_result,
            "before_code": str(repair.get("before_code") or ""),
            "after_code": str(repair.get("after_code") or ""),
            "before_code_sha256": str(repair.get("before_code_sha256") or ""),
            "after_code_sha256": str(repair.get("after_code_sha256") or ""),
            "unified_diff": str(repair.get("unified_diff") or ""),
        }

    @staticmethod
    def _debug_runtime_stage_match(query_text: str, runtime_stage: str) -> float:
        text = str(query_text or "").lower()
        stage = str(runtime_stage or "")
        markers = {
            "checkpoint_averaging": ("checkpoint", "average", "state_dict"),
            "checkpoint_loading": ("checkpoint", "torch.load", "unpickling"),
            "data_loading": ("dataloader", "dataset", "collate", "batch"),
            "import": ("importerror", "modulenotfounderror", "import"),
            "model_loading": ("torch.hub", "from_pretrained", "checkpoint", "weights"),
            "parsing": ("syntaxerror", "indentationerror", "parse"),
            "preprocessing": ("scaler", "pca", "vectorizer", "transform"),
            "split_validation": ("split", "fold", "leakage", "validation"),
            "feature_extraction": ("embedding", "feature", "backbone"),
            "model_forward": ("forward", "shape", "dimension", "tensor"),
            "training": ("backward", "optimizer", "epoch", "loss"),
            "training_metric": ("train auc", "training auc", "mixed label", "metric"),
            "validation": ("validation", "roc_auc", "log loss", "evaluate"),
            "validation_split": ("split", "stratified", "validation size"),
            "oof": ("oof", "out of fold", "cross validation"),
            "inference": ("inference", "predict", "test prediction"),
            "submission": ("submission", "sample_submission"),
        }
        if not stage:
            return 0.70
        values = markers.get(stage, (stage.replace("_", " "),))
        return 1.0 if any(value in text for value in values) else 0.70

    def _debug_method_family_match(self, query_text: str, method_family: str) -> float:
        family = str(method_family or "general")
        if family == "general":
            return 0.80
        tokens = _tokenize(family.replace("_", " "))
        overlap = self._token_overlap(tokens, _tokenize(query_text)) if tokens else 0.0
        return 1.0 if overlap >= 0.50 else 0.60

    def _l3_failure_card_text(self, sop_id: str) -> str:
        node = self.nodes.get(str(sop_id), {})
        signature = (
            node.get("failure_signature")
            if isinstance(node.get("failure_signature"), Mapping)
            else {}
        )
        return " ".join(
            str(value or "")
            for value in (
                signature.get("id"),
                signature.get("pattern"),
                signature.get("root_cause"),
                node.get("title"),
            )
        )

    def _causal_attachment_rows(
        self,
        transition: dict[str, Any],
        *,
        stage: str,
        task_family: str,
        allowed_sop_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        quality_by_sop = {
            str(item.get("sop_id")): item
            for item in (transition.get("attachment_quality") or [])
            if item.get("sop_id")
        }
        rows = []
        transition_id = str(transition.get("id") or "")
        for sop_id in self._active_sops_for_transition(transition_id):
            if allowed_sop_ids is not None and sop_id not in allowed_sop_ids:
                continue
            visibility_ids = self._effective_visibility_sop_ids()
            if visibility_ids is not None and sop_id not in visibility_ids:
                continue
            sop = self.nodes.get(sop_id, {})
            if sop.get("type") != "SOP":
                continue
            _stage_score, stage_compatible = self._sop_stage_fit(sop, stage)
            if not stage_compatible or not self._sop_task_compatible(sop, task_family):
                continue
            edge = self._sop_edge_metadata.get((transition_id, sop_id), {})
            quality = quality_by_sop.get(sop_id, {}) or edge
            quality_kind = str(quality.get("quality") or "")
            quality_score = float(quality.get("score") or 0.0)
            causally_supported = quality_kind == "evidence_turn_match" or quality_score >= CAUSAL_ATTACHMENT_MIN_SCORE
            if not causally_supported:
                continue
            rows.append(
                {
                    "sop_id": sop_id,
                    "quality": quality_kind,
                    "quality_score": 1.0 if quality_kind == "evidence_turn_match" else quality_score,
                }
            )
        return sorted(rows, key=lambda item: (-item["quality_score"], item["sop_id"]))

    def _rank_debug_transition_rows(
        self,
        *,
        query_text: str,
        task_id: str,
        task_desc: str,
        limit: int,
        allowed_sop_ids: set[str] | None = None,
        allowed_transition_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        query_signature = self._failure_signature(query_text)
        task_family = self._task_family_for_query(task_id, task_desc)
        coords = self._coords()
        eligible_transition_ids = [
            transition_id
            for transition_id in self._transitions
            if self._positive_transition(transition_id)[0]
            and (allowed_transition_ids is None or transition_id in allowed_transition_ids)
        ]
        exact_task_transition_ids = [
            transition_id
            for transition_id in eligible_transition_ids
            if canonical_task_id(self.nodes[transition_id].get("task"))
            == canonical_task_id(task_id)
        ]
        anchor = self._query_anchor(
            query_text, exact_task_transition_ids or eligible_transition_ids
        )
        rows: list[dict[str, Any]] = []
        for transition_id in eligible_transition_ids:
            transition = self.nodes[transition_id]
            if (
                str(transition.get("outcome") or "") != "debug_fixed"
                or transition.get("parent_buggy") is not True
                or transition.get("child_buggy") is not False
                or "debug" not in str(transition.get("stage_pair") or "")
            ):
                continue
            task_fit = self._debug_transition_task_fit(
                transition,
                task_id=task_id,
                task_family=task_family,
            )
            if task_fit < 0.70:
                continue
            failure_text = self._debug_parent_failure_text(transition)
            attachments = self._causal_attachment_rows(
                transition,
                stage="debug",
                task_family=task_family,
                allowed_sop_ids=allowed_sop_ids,
            )
            if not attachments:
                continue
            l3_attachments = [
                item
                for item in attachments
                if self.nodes.get(str(item["sop_id"]), {}).get("abstraction_level")
                == "L3_repair"
            ]
            if not l3_attachments:
                continue
            card_text = " ".join(
                self._l3_failure_card_text(item["sop_id"])
                for item in l3_attachments
            )
            signature_ids = " ".join(
                str(
                    (
                        self.nodes.get(str(item["sop_id"]), {}).get(
                            "failure_signature"
                        )
                        or {}
                    ).get("id")
                    or ""
                )
                for item in l3_attachments
            )
            has_explicit_l3_signature = bool(signature_ids.strip())
            specific_signature_overlap = self._specific_failure_signature_overlap(
                signature_ids,
                query_text,
            )
            candidate_signature = self._failure_signature(
                f"{failure_text} {card_text}"
            )
            overlap = len(query_signature & candidate_signature)
            structural_match = (
                overlap / len(query_signature | candidate_signature)
                if overlap
                else 0.0
            )
            card_semantic = self._bounded_token_similarity(query_text, card_text)
            parent_semantic = self._bounded_token_similarity(query_text, failure_text)
            atomic_claim = transition.get("atomic_repair_claim")
            atomic_claim = atomic_claim if isinstance(atomic_claim, Mapping) else {}
            repair_text = str(atomic_claim.get("repair_action") or "")
            structured_match, structured_receipt = structured_debug_relevance(
                query_text,
                failure_text,
                repair_text,
                atomic_claim,
            )
            if atomic_claim:
                # The claim-specific ranker is the causal gate. Do not let a
                # long, semantically broad L3 card rescue a weak signature.
                failure_match = structured_match
                minimum_failure_match = ATOMIC_L3_FAILURE_SIGNATURE_MIN_MATCH
            elif (
                has_explicit_l3_signature
                and specific_signature_overlap
                < L3_SPECIFIC_SIGNATURE_MIN_OVERLAP
                and card_semantic < 0.50
                and structured_match < L3_FAILURE_SIGNATURE_MIN_MATCH
            ):
                continue
            elif has_explicit_l3_signature:
                failure_match = max(
                    structural_match
                    if specific_signature_overlap > 0.0
                    else 0.0,
                    min(1.0, specific_signature_overlap),
                    min(0.90, 1.6 * card_semantic),
                    min(0.75, 1.25 * parent_semantic)
                    if specific_signature_overlap > 0.0
                    else 0.0,
                    structured_match,
                )
                minimum_failure_match = L3_FAILURE_SIGNATURE_MIN_MATCH
            else:
                # Preserve the legacy RunForest L3 path for old bundles.  New
                # evidence-bound L3 cards always take the stricter branch
                # above because they carry a frozen signature ID.
                failure_match = max(
                    structural_match,
                    min(0.90, 1.6 * card_semantic),
                    min(0.75, 1.25 * parent_semantic),
                    structured_match,
                )
                minimum_failure_match = L3_FAILURE_SIGNATURE_MIN_MATCH
            if failure_match < minimum_failure_match:
                continue
            semantic = max(card_semantic, parent_semantic)
            attachment_quality = max(item["quality_score"] for item in attachments)
            primary_l3 = max(
                l3_attachments,
                key=lambda item: float(item.get("quality_score") or 0.0),
            )
            l3_node = self.nodes[str(primary_l3["sop_id"])]
            runtime_match = self._debug_runtime_stage_match(
                query_text, str(l3_node.get("runtime_stage") or "")
            )
            method_match = self._debug_method_family_match(
                query_text, str(l3_node.get("method_family") or "general")
            )
            success_count = max(1, int(l3_node.get("successful_repair_count") or 1))
            success_frequency = min(1.0, math.log1p(success_count) / math.log(5.0))
            confidence_components = {
                "task_match": task_fit,
                "failure_signature_match": failure_match,
                "runtime_stage_match": runtime_match,
                "method_family_match": method_match,
                "clean_transition_quality": min(1.0, attachment_quality),
                "successful_repair_frequency": success_frequency,
            }
            confidence_weights = (
                ATOMIC_L3_DYNAMIC_CONFIDENCE_WEIGHTS
                if atomic_claim
                else L3_DYNAMIC_CONFIDENCE_WEIGHTS
            )
            confidence = sum(
                confidence_weights[key] * value
                for key, value in confidence_components.items()
            )
            score = confidence
            source_task = canonical_task_id(transition.get("task"))
            task_scope = "exact_task" if source_task == canonical_task_id(task_id) else "same_task_type"
            rows.append(
                {
                    "id": transition_id,
                    "score": score,
                    "confidence": min(1.0, confidence),
                    "score_components": {
                        **confidence_components,
                        "failure_signature": failure_match,
                        "task": task_fit,
                        "causal_attachment": attachment_quality,
                        "semantic_diagnostic": semantic,
                        "specific_signature_overlap": specific_signature_overlap,
                        "structured_debug_match": structured_match,
                    },
                    "structured_debug_rank_receipt": structured_receipt,
                    "ranking_backend": (
                        "task_first_structured_debug_signature_v3"
                        if atomic_claim
                        else "stage_task_causal_signature_v2"
                    ),
                    "dynamic_confidence_weights": dict(confidence_weights),
                    "minimum_failure_signature_match": minimum_failure_match,
                    "task_scope": task_scope,
                    "query_failure_signature": sorted(query_signature),
                    "candidate_failure_signature": sorted(candidate_signature),
                    "causal_attachments": attachments,
                    "stage": transition.get("stage_pair"),
                    "task": transition.get("task"),
                    "metric": transition.get("child_metric"),
                    "metric_improvement": transition.get("metric_improvement"),
                    "audit_status": (
                        self.nodes.get(str(transition.get("child_node_id") or ""), {}).get("leakage_audit") or {}
                    ).get("status"),
                    "rank_eligible": True,
                    "eligibility_reason": "clean_causal_debug_transition",
                    "parent_node_id": transition.get("parent_node_id"),
                    "child_node_id": transition.get("child_node_id"),
                    "transition_evidence": self._debug_transition_evidence(transition),
                }
            )
        exact_task_rows = [row for row in rows if row["task_scope"] == "exact_task"]
        if exact_task_rows:
            rows = exact_task_rows
        # Claim-level repair evidence is the primary Debug index. A broad
        # legacy L3 card may have a high historical confidence score while
        # sharing only an exception class with the current failure. Once at
        # least one verified atomic claim passes the causal floor, keep this
        # candidate set pure; legacy rows remain the fallback when no atomic
        # claim matches.
        atomic_rows = [
            row
            for row in rows
            if row.get("ranking_backend")
            == "task_first_structured_debug_signature_v3"
        ]
        if atomic_rows:
            rows = atomic_rows
            rows.sort(
                key=lambda item: (
                    -float(
                        (item.get("score_components") or {}).get(
                            "structured_debug_match"
                        )
                        or 0.0
                    ),
                    -float(item["score"]),
                    str(item["id"]),
                )
            )
        else:
            rows.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return rows[:limit]

    def _project_debug_transitions_to_sops(
        self,
        transition_rows: list[dict[str, Any]],
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        best: dict[str, dict[str, Any]] = {}
        for transition_rank, row in enumerate(transition_rows, 1):
            for attachment in row["causal_attachments"]:
                sop_id = attachment["sop_id"]
                projection_score = float(row["score"]) * float(attachment["quality_score"])
                current = best.get(sop_id)
                if current is None or projection_score > current["projection_score"]:
                    best[sop_id] = {
                        "transition_id": row["id"],
                        "transition_rank": transition_rank,
                        "projection_score": projection_score,
                        "attachment_quality": attachment["quality"],
                        "attachment_quality_score": attachment["quality_score"],
                        "failure_signature": row["candidate_failure_signature"],
                    }
        ordered = sorted(best, key=lambda sop_id: (-best[sop_id]["projection_score"], sop_id))
        return ordered, best

    def _l3_agent_transition_rows(
        self,
        match: Mapping[str, Any],
        *,
        task_family: str,
    ) -> list[dict[str, Any]]:
        """Project one validated Agent root-cause decision into the Debug route."""

        if str(match.get("decision") or "") != "select":
            return []
        sop_id = str(match.get("selected_sop_id") or "")
        transition_id = str(match.get("selected_transition_id") or "")
        transition = self.nodes.get(transition_id, {})
        positive, reason = self._positive_transition(transition_id)
        if not positive:
            raise RuntimeError(
                f"L3 Agent selected a non-positive transition: {transition_id}/{reason}"
            )
        attachments = self._causal_attachment_rows(
            transition,
            stage="debug",
            task_family=task_family,
            allowed_sop_ids={sop_id},
        )
        if not attachments or not any(
            str(item.get("sop_id") or "") == sop_id for item in attachments
        ):
            raise RuntimeError(
                "L3 Agent selection is not bound to its clean repair transition"
            )
        assessment = next(
            (
                row
                for row in match.get("assessments") or []
                if str(row.get("sop_id") or "") == sop_id
            ),
            {},
        )
        confidence = float(match.get("final_confidence") or 0.0)
        node = self.nodes[sop_id]
        audit = (
            self.nodes.get(str(transition.get("child_node_id") or ""), {}).get(
                "leakage_audit"
            )
            or {}
        )
        return [
            {
                "id": transition_id,
                "score": confidence,
                "confidence": confidence,
                "score_components": {
                    "keyword_correspondence": float(
                        assessment.get("keyword_correspondence") or 0.0
                    ),
                    "root_cause_equivalence": float(
                        assessment.get("root_cause_equivalence") or 0.0
                    ),
                    "runtime_stage_match": float(
                        assessment.get("runtime_stage_match") or 0.0
                    ),
                    "agent_confidence": confidence,
                    "task_match": (
                        1.0
                        if match.get("selected_task_scope") == "exact_task"
                        else 0.70
                    ),
                    "manual_synonym_table_used": False,
                },
                "dynamic_confidence_weights": {
                    "agent_keyword_and_root_cause_semantic_match": 1.0
                },
                "task_scope": str(match.get("selected_task_scope") or ""),
                "query_failure_signature": [],
                "candidate_failure_signature": [
                    str((node.get("failure_signature") or {}).get("id") or "")
                ],
                "causal_attachments": attachments,
                "stage": transition.get("stage_pair"),
                "task": transition.get("task"),
                "metric": transition.get("child_metric"),
                "metric_improvement": transition.get("metric_improvement"),
                "audit_status": audit.get("status"),
                "rank_eligible": True,
                "eligibility_reason": "clean_l3_agent_root_cause_match",
                "parent_node_id": transition.get("parent_node_id"),
                "child_node_id": transition.get("child_node_id"),
                "transition_evidence": self._debug_transition_evidence(
                    transition
                ),
                "ranking_backend": (
                    "agent_keyword_and_root_cause_semantic_match_v1"
                ),
                "manual_synonym_table_used": False,
            }
        ]

    def _debug_dynamic_weights(self, transition_rows: list[dict[str, Any]]) -> tuple[dict[str, float], float, str | None]:
        confidence = max((float(row.get("confidence") or 0.0) for row in transition_rows), default=0.0)
        if confidence < DEBUG_TREE_CONFIDENCE_THRESHOLD:
            return {"sop": 1.0, "tree": 0.0}, confidence, "insufficient_causal_tree_confidence"
        configured_tree = float(self.rrf_weights["debug"]["tree"])
        tree_weight = min(configured_tree, DEBUG_TREE_MAX_WEIGHT, DEBUG_TREE_MAX_WEIGHT * confidence)
        return {"sop": 1.0 - tree_weight, "tree": tree_weight}, confidence, None

    def _rank_positive_transition_rows(
        self,
        *,
        stage: str,
        query_text: str,
        task_id: str,
        task_desc: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Rank complete clean positive transitions for layered Improve.

        L1/L2 Recipe slots describe a method before execution.  Once a method
        exists, Improve should consume a concrete parent-to-child change, not
        another detached SOP summary.  The transition remains the injected
        candidate so its parent, child, metric delta, and exact action are all
        recoverable from the RunForest trace.
        """

        task_id = canonical_task_id(task_id)
        query_tokens = _tokenize(query_text)
        candidates: list[str] = []
        for transition_id in self._transitions:
            transition = self.nodes[transition_id]
            if str(transition.get("outcome") or "") != "metric_improved":
                continue
            if not self._positive_transition(transition_id)[0]:
                continue
            candidates.append(transition_id)
        coords = self._coords()
        anchor = self._query_anchor(query_text, candidates)
        improvements = sorted(
            float(self.nodes[transition_id].get("metric_improvement"))
            for transition_id in candidates
            if isinstance(self.nodes[transition_id].get("metric_improvement"), (int, float))
            and not isinstance(self.nodes[transition_id].get("metric_improvement"), bool)
            and math.isfinite(float(self.nodes[transition_id].get("metric_improvement")))
            and float(self.nodes[transition_id].get("metric_improvement")) > 0
        )
        rows: list[dict[str, Any]] = []
        for transition_id in candidates:
            transition = self.nodes[transition_id]
            child = self.nodes.get(str(transition.get("child_node_id") or ""), {})
            source_task = canonical_task_id(
                transition.get("task") or child.get("task") or ""
            )
            task_fit = 1.0 if source_task == task_id else self._task_score(
                child, task_id, task_desc
            )
            if task_fit <= 0.0:
                continue
            action_text = " ".join(
                str(value or "")
                for value in (
                    transition.get("text"),
                    child.get("plan"),
                    child.get("code_summary"),
                    child.get("analysis"),
                )
            )
            lexical = min(
                1.0,
                self._token_overlap(query_tokens, _tokenize(action_text)),
            )
            improvement = transition.get("metric_improvement")
            improvement_quality = 0.0
            if (
                improvements
                and isinstance(improvement, (int, float))
                and not isinstance(improvement, bool)
                and math.isfinite(float(improvement))
            ):
                improvement_quality = sum(
                    value <= float(improvement) for value in improvements
                ) / len(improvements)
            geometry = 0.0
            if anchor is not None and transition_id in coords:
                geometry = 1.0 / (1.0 + self._distance(anchor, coords[transition_id]))
            stage_pair = str(transition.get("stage_pair") or "")
            stage_fit = 1.0 if any(
                token in stage_pair for token in (stage, "improve", "evolution")
            ) else 0.5
            score = (
                0.35 * task_fit
                + 0.25 * lexical
                + 0.20 * improvement_quality
                + 0.10 * stage_fit
                + 0.10 * geometry
            )
            child_audit = child.get("leakage_audit") if isinstance(child.get("leakage_audit"), dict) else {}
            rows.append(
                {
                    "id": transition_id,
                    "score": score,
                    "score_components": {
                        "task": task_fit,
                        "lexical": lexical,
                        "task_local_improvement_percentile": improvement_quality,
                        "stage": stage_fit,
                        "geometry": geometry,
                    },
                    "stage": stage_pair,
                    "task": source_task,
                    "metric": child.get("metric"),
                    "metric_improvement": improvement,
                    "audit_status": child_audit.get("status"),
                    "rank_eligible": child_audit.get("rank_eligible") is True,
                    "eligibility_reason": "clean_positive_improvement_transition",
                    "parent_node_id": transition.get("parent_node_id"),
                    "child_node_id": transition.get("child_node_id"),
                    "transition_evidence": {
                        "proven_action": str(
                            child.get("plan")
                            or child.get("code_summary")
                            or transition.get("text")
                            or ""
                        ),
                        "child_result": str(
                            child.get("analysis")
                            or child.get("terminal_excerpt")
                            or ""
                        ),
                    },
                }
            )
        rows.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return rows[:limit]

    def _rank_tree_rows(
        self,
        *,
        stage: str,
        query_text: str,
        task_id: str,
        task_desc: str,
        limit: int,
        allowed_node_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = [
            node_id
            for node_id in self._run_nodes
            if self._successful_run_node(node_id)
            and (allowed_node_ids is None or node_id in allowed_node_ids)
        ]
        stage_bonus = {
            "draft": {"draft": 0.08},
            "improve": {"improve": 0.10, "evolution": 0.05},
            "debug": {"debug": 0.10, "improve": 0.04},
            "evolution": {"evolution": 0.10, "improve": 0.04},
            "fusion": {"improve": 0.05, "evolution": 0.05},
        }[stage]
        coords = self._coords()
        anchor = self._query_anchor(query_text, candidates)
        query_tokens = _tokenize(query_text)
        positive_improvements_by_task: dict[str, list[float]] = collections.defaultdict(list)
        for candidate_id in candidates:
            candidate = self.nodes[candidate_id]
            improvement = candidate.get("metric_improvement")
            if (
                isinstance(improvement, (int, float))
                and not isinstance(improvement, bool)
                and math.isfinite(float(improvement))
                and improvement > 0
            ):
                positive_improvements_by_task[canonical_task_id(candidate.get("task") or "unknown")].append(
                    float(improvement)
                )
        for values in positive_improvements_by_task.values():
            values.sort()
        rows: list[dict[str, Any]] = []
        for node_id in candidates:
            node = self.nodes[node_id]
            eligible, eligibility_reason = self._execution_candidate_eligibility(node_id)
            if not eligible:
                continue
            audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
            lexical = min(1.0, self._token_overlap(query_tokens, self._node_tokens.get(node_id, set())))
            task = self._task_score(node, task_id, task_desc)
            node_stage = str(node.get("stage") or node.get("stage_pair") or "")
            stage_fit = stage_bonus.get(node_stage, 0.0)
            metric_bonus = 0.0
            metric_quality = 0.0
            improvement = node.get("metric_improvement")
            if (
                isinstance(improvement, (int, float))
                and not isinstance(improvement, bool)
                and math.isfinite(float(improvement))
                and improvement > 0
            ):
                task_improvements = positive_improvements_by_task.get(
                    canonical_task_id(node.get("task") or "unknown"), []
                )
                metric_quality = (
                    sum(value <= float(improvement) for value in task_improvements) / len(task_improvements)
                    if task_improvements else 0.0
                )
                metric_bonus = 0.08 * metric_quality
            geometry = 0.0
            if anchor is not None and node_id in coords:
                geometry = 1.0 / (1.0 + self._distance(anchor, coords[node_id]))
            score = 0.50 * geometry + 0.32 * lexical + task + stage_fit + metric_bonus
            rows.append(
                {
                    "id": node_id,
                    "score": score,
                    "score_components": {
                        "geometry": geometry,
                        "lexical": lexical,
                        "task": task,
                        "stage": stage_fit,
                        "metric_improvement": metric_bonus,
                        "task_local_improvement_percentile": metric_quality,
                    },
                    "stage": node_stage,
                    "task": node.get("task"),
                    "metric": node.get("metric"),
                    "metric_improvement": improvement,
                    "audit_status": audit.get("status"),
                    "rank_eligible": audit.get("rank_eligible") is True,
                    "eligibility_reason": eligibility_reason,
                }
            )
        rows.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return rows[:limit]

    def _rank_tree(self, *, stage: str, query_text: str, task_id: str, task_desc: str, limit: int) -> list[str]:
        return [
            item["id"]
            for item in self._rank_tree_rows(
                stage=stage,
                query_text=query_text,
                task_id=task_id,
                task_desc=task_desc,
                limit=limit,
            )
        ]

    def _tree_sop_projection(self, tree_ids: list[str], allowed_sop_ids: set[str] | None = None) -> list[str]:
        projected: list[str] = []
        visibility_ids = self._effective_visibility_sop_ids()
        for execution_id in tree_ids:
            for sop_id in self._active_sops_for_execution(execution_id):
                if allowed_sop_ids is not None and sop_id not in allowed_sop_ids:
                    continue
                if visibility_ids is not None and sop_id not in visibility_ids:
                    continue
                linked_transitions = self._active_links_for_execution(execution_id).get(sop_id, [])
                linked_clean = any(self._positive_transition(transition_id)[0] for transition_id in linked_transitions)
                if linked_clean and sop_id not in projected:
                    projected.append(sop_id)
        return projected

    def rank_sop_hybrid(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        query_text: str,
        limit: int,
        allowed_sop_ids: set[str] | None = None,
        visibility_request: VisibilityRequest | None = None,
        authority_operation: Operation | str | None = None,
        active_protocol: ProtocolRef | str | None = None,
    ) -> dict[str, Any]:
        """Return the production channel logic projected onto the SOP decision space."""
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGE_QUOTAS:
            raise ValueError(f"Unsupported stage-hybrid stage: {stage}")
        visibility_pack = self._prepare_visibility(
            stage=stage,
            task_id=task_id,
            task_desc=task_desc,
            request=visibility_request,
            operation=authority_operation,
            active_protocol=active_protocol,
        )
        visibility_ids = self._effective_visibility_sop_ids()
        if visibility_ids is not None:
            allowed_sop_ids = (
                visibility_ids
                if allowed_sop_ids is None
                else set(allowed_sop_ids) & visibility_ids
            )
        if stage == "debug" and self.retrieval_control == "layered_strategy":
            frozen_l3_ids = {
                sop_id
                for sop_id in self._recipe_sop_ids
                if self.nodes.get(sop_id, {}).get("abstraction_level")
                == "L3_repair"
            }
            if frozen_l3_ids:
                allowed_sop_ids = (
                    frozen_l3_ids
                    if allowed_sop_ids is None
                    else set(allowed_sop_ids) & frozen_l3_ids
                )
        sop_rows = self._rank_sops(
            query_text,
            stage,
            len(self._sops),
            task_id=task_id,
            task_desc=task_desc,
            allowed_sop_ids=allowed_sop_ids,
        )
        clean_rows = [row for row in sop_rows if row["clean_supporting_transition_ids"]]
        compatible = [
            row for row in clean_rows
            if row["stage_compatible"] and row["task_compatible"]
        ]
        compatible_ids = {row["id"] for row in compatible}
        direct_ids = [row["id"] for row in compatible]
        projection_provenance: dict[str, dict[str, Any]] = {}
        fallback_reason = None
        tree_confidence = None
        if stage == "debug":
            tree_rows = self._rank_debug_transition_rows(
                query_text=query_text,
                task_id=task_id,
                task_desc=task_desc,
                limit=self.stage_quotas[stage]["tree_candidates"],
                allowed_sop_ids=allowed_sop_ids,
            )
            projected_ids, projection_provenance = self._project_debug_transitions_to_sops(tree_rows)
            projected_ids = [sop_id for sop_id in projected_ids if sop_id in compatible_ids]
            weights, tree_confidence, fallback_reason = self._debug_dynamic_weights(tree_rows)
            if fallback_reason and self.retrieval_control == "stage_hybrid":
                projected_ids = []
        else:
            tree_rows = self._rank_tree_rows(
                stage=stage,
                query_text=query_text,
                task_id=task_id,
                task_desc=task_desc,
                limit=max(self.stage_quotas[stage]["tree_candidates"], limit * 8),
            )
            tree_ids = [row["id"] for row in tree_rows]
            projected_ids = [
                sop_id
                for sop_id in self._tree_sop_projection(tree_ids, allowed_sop_ids)
                if sop_id in compatible_ids
            ]
            weights = self.rrf_weights[stage]
        if self.retrieval_control == "sop_only":
            fused = weighted_rrf(direct_ids, [], sop_weight=1.0, tree_weight=0.0)
        elif self.retrieval_control == "tree_only":
            fused = weighted_rrf([], projected_ids, sop_weight=0.0, tree_weight=1.0)
        elif self.retrieval_control == "naive_concat":
            concatenated = [*direct_ids, *(value for value in projected_ids if value not in direct_ids)]
            fused = [
                {
                    "id": sop_id,
                    "rrf_score": 0.0,
                    "sop_rank": direct_ids.index(sop_id) + 1 if sop_id in direct_ids else None,
                    "tree_rank": projected_ids.index(sop_id) + 1 if sop_id in projected_ids else None,
                    "candidate_class": "sop_transition_matches" if sop_id in direct_ids else "tree_only_candidates",
                }
                for sop_id in concatenated
            ]
        else:
            fused = weighted_rrf(
                direct_ids,
                projected_ids,
                sop_weight=weights["sop"],
                tree_weight=weights["tree"],
            )
        final = [
            item
            for item in fused
            if item["id"] in compatible_ids
        ][:limit]
        return {
            "schema": "stage_hybrid_sop_ranking_v2",
            "algorithm_version": "stage_hybrid_v2",
            "stage_route": {
                "stage": stage,
                "route": STAGE_ROUTE[stage],
                "control": self.retrieval_control,
                "rrf": weights,
                "configured_rrf": self.rrf_weights[stage],
                "tree_confidence": tree_confidence,
                "fallback_reason": fallback_reason,
            },
            "direct_sop_candidates": sop_rows,
            "direct_clean_sop_ids": direct_ids,
            "tree_execution_candidates": tree_rows,
            "tree_projected_sop_ids": projected_ids,
            "tree_projection_provenance": projection_provenance,
            "fused_sop_candidates": final,
            "safety_gate": {
                "predicate": "clean_support_and_stage_task_compatibility_required",
                "input_count": len(fused),
                "output_count": len(final),
                "all_outputs_clean": all(item["id"] in compatible_ids for item in final),
                "all_outputs_visible": all(
                    visibility_ids is None or item["id"] in visibility_ids
                    for item in final
                ),
            },
            "visible_clause_ids": visibility_pack.effective_clause_ids,
            "visibility_trace": visibility_pack.visibility_trace,
        }

    def _risk_warnings(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        warnings = []
        for candidate in candidates:
            for rejected in candidate["rejected_support"]:
                warnings.append(
                    {
                        "sop_id": candidate["id"],
                        "transition_id": rejected["transition_id"],
                        "reason": rejected["reason"],
                        "disposition": "warning_or_repair_evidence_only",
                    }
                )
        return warnings[:20]

    def _formal_flat_relevance_pack(
        self,
        *,
        stage: str,
        query_text: str,
        visibility_pack: VisibleSOPPack,
    ) -> dict[str, Any]:
        """Build the three preregistered flat-ranking baseline packs.

        Eligibility is decided by the configured visibility profile. Ranking
        then uses only lexical relevance over the already materialized clause
        text: no Stage, task-fit, geometry, score, Tree, or clean-evidence term
        can change candidate order.
        """

        quotas = self.stage_quotas[stage]
        rows: list[dict[str, Any]] = []
        for sop_id in visibility_pack.effective_sop_ids:
            projection = visibility_pack.rendered_by_sop.get(sop_id) or {}
            retrieval_text = str(projection.get("retrieval_text") or "")
            score = self._bounded_token_similarity(query_text, retrieval_text)
            rows.append(
                {
                    "id": sop_id,
                    "score": score,
                    "score_components": {"flat_text_relevance": score},
                    "hybrid_score_components": {},
                    "ranking_backend": "formal_flat_text_relevance_v1",
                    "abstraction_level": self.nodes.get(sop_id, {}).get(
                        "abstraction_level"
                    ),
                    "sop_kind": self.nodes.get(sop_id, {}).get("sop_kind"),
                    "method_family": self.nodes.get(sop_id, {}).get(
                        "method_family"
                    ),
                    "decision_stages": [],
                    "task_families": [],
                    "stage_compatible": True,
                    "task_compatible": True,
                    "task_family": "marginalized",
                    "clean_supporting_transition_ids": [],
                    "clean_supporting_transition_count": 0,
                    "rejected_support": [],
                    "rejected_support_count": 0,
                    "visible_clause_ids": list(
                        projection.get("clause_ids") or []
                    ),
                    "visible_text": str(projection.get("prompt_text") or ""),
                    "geometry_visibility_safe": False,
                }
            )
        rows.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        candidates = rows[: quotas["sop_candidates"]]
        trace = [
            {
                "retrieval_channel": "formal_flat_sop",
                "candidate_class": "sop_only_candidates",
                "gateway_sop_id": None,
                "supporting_transition_ids": [],
                "selection_reason": (
                    "flat text relevance only; score="
                    f"{float(item['score']):.8f}"
                ),
                "selection_state": "injected",
                "candidate_id": item["id"],
            }
            for item in candidates
        ]
        intentional_bypass = visibility_pack.visibility_trace.get(
            "intentional_authority_bypass_clause_ids", []
        )
        return {
            "schema": PACK_SCHEMA,
            "algorithm_version": "formal_flat_relevance_v1",
            "stage_route": {
                "stage": stage,
                "route": "flat_sop_relevance",
                "control": self.retrieval_control,
                "quotas": quotas,
                "rrf": {"sop": 1.0, "tree": 0.0},
                "configured_rrf": self.rrf_weights[stage],
                "tree_confidence": None,
                "fallback_reason": None,
            },
            "direct_sop_candidates": candidates,
            "selected_sop_gateways": [],
            "gateway_transitions": {},
            "tree_candidates": [],
            "tree_candidate_details": [],
            "sop_transition_matches": [],
            "sop_only_candidates": candidates,
            "tree_only_candidates": [],
            "evidence_refs": [],
            "failure_patterns": [],
            "risk_warnings": [],
            "visibility_warnings": [],
            "navigation_trace": trace,
            "fused_execution_candidates": [],
            "execution_candidate_provenance": {},
            "execution_safety_gate": {
                "predicate": "not_applicable_flat_clause_baseline",
                "rejected": [],
                "all_outputs_clean": True,
            },
            "gateway_selection": {
                "mode": "not_applicable_flat_clause_baseline",
                "llm_tool_calls": 0,
                "goal": "flat clause retrieval",
                "clean_eligible_count": 0,
                "eligible_count": len(candidates),
                "stage_task_gate_rejected_count": 0,
            },
            "visible_clause_ids": visibility_pack.effective_clause_ids,
            "visibility_trace": visibility_pack.visibility_trace,
            "visibility_safety_gate": {
                "mode": self.visibility_mode,
                "pre_ranking": True,
                "intentional_baseline_authority_bypass_count": len(
                    intentional_bypass
                ),
                "unauthorized_prompt_exposure": len(intentional_bypass),
                "unauthorized_activation": 0,
                "all_sop_candidates_visible": all(
                    item["id"] in set(visibility_pack.effective_sop_ids)
                    for item in candidates
                ),
            },
        }

    def _hybrid_pack(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        query_text: str,
        strategy_context: dict[str, Any] | None = None,
        visibility_request: VisibilityRequest | None = None,
        authority_operation: Operation | str | None = None,
        active_protocol: ProtocolRef | str | None = None,
    ) -> dict[str, Any]:
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGE_QUOTAS:
            raise ValueError(f"Unsupported stage-hybrid stage: {stage}")
        if self.experiment_r_enabled:
            # Exp-R deliberately has only three retrieval policies
            # (Draft/Improve/Debug), while the search engine retains finer
            # generation-stage names such as Evolution and Fusion.  Route
            # those aliases through the Improve policy without changing the
            # raw generation stage that retrieve_for_node records on the pack.
            from agents.memory.end2end_memory_system import canonical_stage
            from agents.memory.experiment_r_router import build_experiment_r_pack

            return build_experiment_r_pack(
                self,
                stage=canonical_stage(stage),
                task_id=task_id,
                task_desc=task_desc,
                query_text=query_text,
                visibility_request=visibility_request,
                authority_operation=authority_operation,
                active_protocol=active_protocol,
            )
        visibility_pack = self._prepare_visibility(
            stage=stage,
            task_id=task_id,
            task_desc=task_desc,
            request=visibility_request,
            operation=authority_operation,
            active_protocol=active_protocol,
        )
        visibility_ids = self._effective_visibility_sop_ids()
        raw_ranked = self._rank_with_scores(
            query_text=query_text,
            candidate_ids=[
                node_id
                for node_id in self._run_nodes
                if self.nodes.get(node_id, {}).get("stage") in {
                    "draft", "debug", "improve", "evolution", "fusion"
                }
            ],
            task_id=task_id,
            task_desc=task_desc,
            top_k=max(5, self.top_k),
            stage_bonus={stage: 0.10},
        )
        if self.positive_control_force_raw and self._positive_control_probe_ids:
            probe_ids = list(self._positive_control_probe_ids[: self.top_k])
            raw_ranked = [
                (1.0 - (index * 1e-6), node_id)
                for index, node_id in enumerate(probe_ids)
            ] + [
                row for row in raw_ranked
                if row[1] not in set(probe_ids)
            ]
            raw_ranked = raw_ranked[: max(5, self.top_k)]
        pre_gate_raw_candidates = []
        for raw_rank, (raw_score, node_id) in enumerate(raw_ranked, 1):
            node = self.nodes[node_id]
            eligible, eligibility_reason = self._execution_candidate_eligibility(node_id)
            audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
            pre_gate_raw_candidates.append(
                {
                    "candidate_id": node_id,
                    "rank": raw_rank,
                    "score": raw_score,
                    "source_run_id": node.get("run_id"),
                    "source_task_id": node.get("task"),
                    "source_stage": node.get("stage"),
                    "audit_status": audit.get("status") or node.get("audit_status"),
                    "memory_disposition": audit.get("memory_disposition") or node.get("memory_disposition"),
                    "quarantined": bool(node.get("quarantined")),
                    "operation_authorized": eligible,
                    "gate_reason": eligibility_reason,
                    "controlled_positive_control": node_id in self._positive_control_probe_ids,
                }
            )
        quotas = self.stage_quotas[stage]
        if self.retrieval_control in FORMAL_FLAT_RELEVANCE_CONTROLS:
            return self._formal_flat_relevance_pack(
                stage=stage,
                query_text=query_text,
                visibility_pack=visibility_pack,
            )
        allowed_levels = None
        method_family = None
        layered_debug_sop_ids: set[str] | None = None
        precomputed_debug_rows: list[dict[str, Any]] | None = None
        precomputed_debug_weights: dict[str, float] | None = None
        precomputed_debug_confidence: float | None = None
        precomputed_debug_fallback_reason: str | None = None
        l3_agent_match: dict[str, Any] | None = None
        if self.retrieval_control == "layered_strategy":
            allowed_levels = {"L3_repair"} if stage == "debug" else {"L2_tactic"}
            selected = (strategy_context or {}).get("selected_strategy") or (strategy_context or {})
            method_family = (
                str(selected.get("method_family") or "") or None
                if stage != "debug"
                else None
            )
            if stage == "debug":
                frozen_l3_ids = {
                    sop_id
                    for sop_id in self._recipe_sop_ids
                    if self.nodes.get(sop_id, {}).get("abstraction_level")
                    == "L3_repair"
                }
                if frozen_l3_ids:
                    layered_debug_sop_ids = (
                        frozen_l3_ids
                        if visibility_ids is None
                        else frozen_l3_ids & set(visibility_ids)
                    )
                    if self.experiment_r_l3_agent_match_enabled:
                        from agents.memory.experiment_r_router import (
                            _agentic_l3_debug_match,
                            _l3_policy_authorized_sop_ids,
                        )

                        policy_visible_l3_ids = _l3_policy_authorized_sop_ids(
                            self, visibility_ids
                        )
                        layered_debug_sop_ids = (
                            frozen_l3_ids
                            if policy_visible_l3_ids is None
                            else frozen_l3_ids & set(policy_visible_l3_ids)
                        )
                        l3_agent_match = _agentic_l3_debug_match(
                            self,
                            task_id=task_id,
                            task_desc=task_desc,
                            query_text=query_text,
                            visible_sop_ids=layered_debug_sop_ids,
                        )
                        self._trace_local.l3_agent_match = copy.deepcopy(
                            l3_agent_match
                        )
                        precomputed_debug_rows = self._l3_agent_transition_rows(
                            l3_agent_match,
                            task_family=self._task_family_for_query(
                                task_id, task_desc
                            ),
                        )
                    else:
                        # Legacy controls retain the deterministic lexical
                        # matcher. Dynamic layered_strategy opts into the Agent
                        # path above and never consults its synonym expansion.
                        precomputed_debug_rows = self._rank_debug_transition_rows(
                            query_text=query_text,
                            task_id=task_id,
                            task_desc=task_desc,
                            limit=quotas["tree_candidates"],
                            allowed_sop_ids=layered_debug_sop_ids,
                        )
                    (
                        precomputed_debug_weights,
                        precomputed_debug_confidence,
                        precomputed_debug_fallback_reason,
                    ) = self._debug_dynamic_weights(precomputed_debug_rows)
                    if precomputed_debug_fallback_reason:
                        precomputed_debug_rows = []
                        if l3_agent_match is not None:
                            precomputed_debug_fallback_reason = (
                                "l3_agent_failure_abstain_no_manual_fallback"
                                if l3_agent_match.get("decision")
                                == "agent_failure_abstain"
                                else "l3_agent_abstained_no_manual_fallback"
                            )
                    projected_l3_ids, _projection = (
                        self._project_debug_transitions_to_sops(
                            precomputed_debug_rows
                        )
                    )
                    layered_debug_sop_ids &= set(projected_l3_ids)
        if self.retrieval_control == "layered_strategy" and stage != "debug":
            # Draft/Model Design already consumed Recipe L1/L2.  Improve and
            # Evolution receive complete positive transitions directly, so a
            # second generic SOP gateway cannot displace the proven change.
            ranked_sops = []
            selected = []
            selection_meta = {
                "mode": "layered_transition_only",
                "llm_tool_calls": 0,
                "goal": "retrieve complete positive RunForest transitions",
                "clean_eligible_count": 0,
                "eligible_count": 0,
                "stage_task_gate_rejected_count": 0,
            }
        elif (
            self.retrieval_control == "layered_strategy"
            and stage == "debug"
            and l3_agent_match is not None
        ):
            ranked_sops = self._rank_sops(
                query_text,
                stage,
                len(self._sops),
                allowed_levels=allowed_levels,
                method_family=method_family,
                task_id=task_id,
                task_desc=task_desc,
                allowed_sop_ids=(
                    layered_debug_sop_ids
                    if layered_debug_sop_ids is not None
                    else visibility_ids
                ),
            )
            selected_sop_id = str(
                l3_agent_match.get("selected_sop_id") or ""
            )
            selected = [
                copy.deepcopy(row)
                for row in ranked_sops
                if row["id"] == selected_sop_id
            ][:1]
            if selected:
                selected[0]["score"] = float(
                    l3_agent_match.get("final_confidence") or 0.0
                )
                selected[0]["selection_reason"] = (
                    "specialized L3 Agent root-cause selection"
                )
                selected[0]["l3_agent_selected"] = True
            selection_meta = {
                "mode": "l3_agent_root_cause_match",
                "llm_tool_calls": int(
                    l3_agent_match.get("agent_calls") or 0
                ),
                "goal": str(l3_agent_match.get("reason") or ""),
                "clean_eligible_count": len(ranked_sops),
                "eligible_count": len(layered_debug_sop_ids or []),
                "stage_task_gate_rejected_count": (
                    len(frozen_l3_ids) - len(layered_debug_sop_ids or [])
                ),
                "manual_synonym_table_used": False,
                "selected_sop_id": selected_sop_id,
            }
        else:
            ranked_sops = self._rank_sops(
                query_text,
                stage,
                len(self._sops),
                allowed_levels=allowed_levels,
                method_family=method_family,
                task_id=task_id,
                task_desc=task_desc,
                allowed_sop_ids=(
                    layered_debug_sop_ids
                    if layered_debug_sop_ids is not None
                    else visibility_ids
                ),
            )
            selected, selection_meta = self._select_gateways(
                ranked_sops, stage=stage, query_text=query_text, limit=quotas["sop_gateways"]
            )
        sop_candidates = list(ranked_sops[: quotas["sop_candidates"]])
        candidate_ids = {item["id"] for item in sop_candidates}
        sop_candidates.extend(item for item in selected if item["id"] not in candidate_ids)
        (
            sop_execution,
            gateway_transitions,
            evidence_refs,
            failure_patterns,
            trace,
            execution_provenance,
        ) = self._expand_gateways(selected)
        tree_fallback_reason = None
        tree_confidence = None
        if stage == "debug":
            if precomputed_debug_rows is not None:
                tree_rows = precomputed_debug_rows
                weights = precomputed_debug_weights or {
                    "sop": 1.0,
                    "tree": 0.0,
                }
                tree_confidence = precomputed_debug_confidence
                tree_fallback_reason = precomputed_debug_fallback_reason
            else:
                tree_rows = self._rank_debug_transition_rows(
                    query_text=query_text,
                    task_id=task_id,
                    task_desc=task_desc,
                    limit=quotas["tree_candidates"],
                    allowed_sop_ids=(
                        layered_debug_sop_ids
                        if layered_debug_sop_ids is not None
                        else visibility_ids
                    ),
                )
                weights, tree_confidence, tree_fallback_reason = self._debug_dynamic_weights(tree_rows)
                if tree_fallback_reason and self.retrieval_control in {"stage_hybrid", "layered_strategy"}:
                    tree_rows = []
        elif self.retrieval_control == "layered_strategy" and stage in {
            "improve",
            "evolution",
            "fusion",
        }:
            tree_rows = self._rank_positive_transition_rows(
                stage=stage,
                query_text=query_text,
                task_id=task_id,
                task_desc=task_desc,
                limit=quotas["tree_candidates"],
            )
            weights = self.rrf_weights[stage]
        else:
            tree_rows = self._rank_tree_rows(
                stage=stage,
                query_text=query_text,
                task_id=task_id,
                task_desc=task_desc,
                limit=quotas["tree_candidates"],
            )
            weights = self.rrf_weights[stage]
        tree_ids = [item["id"] for item in tree_rows]
        for item in tree_rows:
            record = execution_provenance.setdefault(
                item["id"],
                {
                    "candidate_id": item["id"],
                    "source_channels": [],
                    "gateway_sop_ids": [],
                    "supporting_transition_ids": [],
                    "safety_status": "clean",
                    "safety_reason": item["eligibility_reason"],
                },
            )
            if "tree_direct" not in record["source_channels"]:
                record["source_channels"].append("tree_direct")
            record["tree_score"] = item["score"]
            record["tree_score_components"] = item["score_components"]
            if self.nodes.get(item["id"], {}).get("type") == "Transition":
                record["supporting_transition_ids"] = [item["id"]]
        if self.retrieval_control == "sop_only":
            tree_ids = []
            tree_rows = []
            execution_provenance = {
                node_id: {
                    **record,
                    "source_channels": ["sop_gateway"],
                }
                for node_id, record in execution_provenance.items()
                if node_id in sop_execution
            }
            fused = [
                {"id": node_id, "rrf_score": 0.0, "sop_rank": rank, "tree_rank": None, "candidate_class": "sop_transition_matches"}
                for rank, node_id in enumerate(sop_execution, 1)
            ]
        elif self.retrieval_control == "tree_only":
            selected = []
            gateway_transitions = {}
            sop_execution = []
            sop_candidates = []
            evidence_refs = []
            failure_patterns = []
            trace = []
            execution_provenance = {
                node_id: {
                    **execution_provenance[node_id],
                    "source_channels": ["tree_direct"],
                    "gateway_sop_ids": [],
                    "supporting_transition_ids": [],
                }
                for node_id in tree_ids
            }
            fused = [
                {"id": node_id, "rrf_score": 0.0, "sop_rank": None, "tree_rank": rank, "candidate_class": "tree_only_candidates"}
                for rank, node_id in enumerate(tree_ids, 1)
            ]
        elif self.retrieval_control == "naive_concat":
            fused = [
                {"id": node_id, "rrf_score": 0.0, "sop_rank": rank, "tree_rank": None, "candidate_class": "sop_transition_matches"}
                for rank, node_id in enumerate(sop_execution, 1)
            ]
            for rank, node_id in enumerate(tree_ids, 1):
                if node_id not in {item["id"] for item in fused}:
                    fused.append(
                        {"id": node_id, "rrf_score": 0.0, "sop_rank": None, "tree_rank": rank, "candidate_class": "tree_only_candidates"}
                    )
        else:
            fused = weighted_rrf(
                sop_execution,
                tree_ids,
                sop_weight=weights["sop"],
                tree_weight=weights["tree"],
            )
        # The raw observer must cover every execution candidate proposed by
        # either retrieval channel before the execution-safety gate.  The
        # broad relevance probe above is intentionally retained to measure
        # naturally ineligible opportunities, but it is not guaranteed to
        # contain SOP-gateway transitions selected by the hybrid route.
        raw_candidate_ids = {
            str(item["candidate_id"]) for item in pre_gate_raw_candidates
        }
        for raw_rank, item in enumerate(fused, len(pre_gate_raw_candidates) + 1):
            node_id = str(item["id"])
            if node_id in raw_candidate_ids:
                continue
            node = self.nodes[node_id]
            eligible, eligibility_reason = self._execution_candidate_eligibility(node_id)
            audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
            pre_gate_raw_candidates.append(
                {
                    "candidate_id": node_id,
                    "rank": raw_rank,
                    "score": float(item.get("rrf_score") or 0.0),
                    "source_run_id": node.get("run_id"),
                    "source_task_id": node.get("task"),
                    "source_stage": node.get("stage"),
                    "audit_status": audit.get("status") or node.get("audit_status"),
                    "memory_disposition": audit.get("memory_disposition") or node.get("memory_disposition"),
                    "quarantined": bool(node.get("quarantined")),
                    "operation_authorized": eligible,
                    "gate_reason": eligibility_reason,
                    "controlled_positive_control": node_id in self._positive_control_probe_ids,
                    "proposal_channel": "hybrid_execution_pre_gate",
                }
            )
            raw_candidate_ids.add(node_id)
        rejected_execution = []
        clean_fused = []
        for item in fused:
            eligible, reason = self._execution_candidate_eligibility(item["id"])
            if eligible:
                clean_fused.append(item)
            else:
                rejected_execution.append({"candidate_id": item["id"], "reason": reason})
        fused = clean_fused
        prompt_execution_limit = self.top_k
        if self.retrieval_control == "layered_strategy" and stage == "debug":
            # Debug has one separately rendered L3 gateway.  Keep the
            # execution half of the Prompt to the configured RunForest quota
            # so a gateway expansion cannot silently turn a frozen 1:5 design
            # into one L3 card plus six (or more) repair transitions.
            prompt_execution_limit = min(
                self.top_k, quotas["tree_candidates"]
            )
            fused = fused[:prompt_execution_limit]
        selected_sop_ids = {item["id"] for item in selected}
        for item in sop_candidates:
            trace.append(
                {
                    "retrieval_channel": "sop_direct",
                    "candidate_class": (
                        "sop_transition_matches" if item["id"] in selected_sop_ids else "sop_only_candidates"
                    ),
                    "gateway_sop_id": item["id"] if item["id"] in selected_sop_ids else None,
                    "candidate_id": item["id"],
                    "supporting_transition_ids": item["clean_supporting_transition_ids"],
                    "selection_reason": next(
                        (value["selection_reason"] for value in selected if value["id"] == item["id"]),
                        "not selected as a formal gateway",
                    ),
                    "selection_state": "selected" if item["id"] in selected_sop_ids else "candidate",
                }
            )
        tree_details_by_id = {item["id"]: item for item in tree_rows}
        for node_id in tree_ids:
            provenance = execution_provenance.get(node_id, {})
            detail = tree_details_by_id[node_id]
            trace.append(
                {
                    "retrieval_channel": "tree_direct",
                    "candidate_class": (
                        "sop_transition_matches" if node_id in sop_execution else "tree_only_candidates"
                    ),
                    "gateway_sop_id": next(iter(provenance.get("gateway_sop_ids") or []), None),
                    "candidate_id": node_id,
                    "supporting_transition_ids": list(provenance.get("supporting_transition_ids") or []),
                    "selection_reason": (
                        f"independent {STAGE_ROUTE[stage]} tree ranking; "
                        f"score={detail['score']:.8f} components={json.dumps(detail['score_components'], sort_keys=True)}"
                    ),
                    "selection_state": "selected",
                }
            )
        for item in fused[:prompt_execution_limit]:
            provenance = execution_provenance.get(item["id"], {})
            trace.append(
                {
                    "retrieval_channel": "hybrid_rrf",
                    "candidate_class": item["candidate_class"],
                    "gateway_sop_id": next(iter(provenance.get("gateway_sop_ids") or []), None),
                    "supporting_transition_ids": list(provenance.get("supporting_transition_ids") or []),
                    "selection_reason": f"weighted RRF score={item['rrf_score']:.8f}",
                    "selection_state": "injected",
                    "candidate_id": item["id"],
                }
            )
        sop_only = [item for item in sop_candidates if item["id"] not in selected_sop_ids]
        visibility_warnings = []
        if self._visibility_is_enforced():
            visibility_warnings = [
                {
                    "clause_id": clause.clause_id,
                    "sop_id": clause.sop_id,
                    "text": clause.text,
                    "disposition": "navigation_warning_only",
                }
                for clause in visibility_pack.warning_clauses
            ]
        final_prompt_candidate_ids = {
            item["id"] for item in fused[:prompt_execution_limit]
        }
        for item in pre_gate_raw_candidates:
            item["final_prompt_visible"] = item["candidate_id"] in final_prompt_candidate_ids
        trace.append(
            {
                "retrieval_channel": "pre_prompt_authority_receipt",
                "candidate_class": "raw_run_node_top_k",
                "gateway_sop_id": "",
                "supporting_transition_ids": [],
                "selection_reason": "raw_candidate_observer_before_authority",
                "selection_state": "candidate",
                "authority_observation_state": "observed_before_gate",
                "target_task_id": task_id,
                "decision_stage": stage,
                "pre_gate_raw_candidates": pre_gate_raw_candidates,
                "final_prompt_candidate_ids": sorted(final_prompt_candidate_ids),
            }
        )
        return {
            "schema": PACK_SCHEMA,
            "algorithm_version": (
                "stage_hybrid_l3_agent_root_cause_v1"
                if l3_agent_match is not None
                else "stage_hybrid_v2"
            ),
            "stage_route": {
                "stage": stage,
                "route": STAGE_ROUTE[stage],
                "control": self.retrieval_control,
                "quotas": quotas,
                "rrf": weights,
                "configured_rrf": self.rrf_weights[stage],
                "tree_confidence": tree_confidence,
                "fallback_reason": tree_fallback_reason,
            },
            "target_task_id": task_id,
            "direct_sop_candidates": sop_candidates,
            "selected_sop_gateways": selected,
            "gateway_transitions": gateway_transitions,
            "tree_candidates": tree_ids,
            "tree_candidate_details": tree_rows,
            "sop_transition_matches": [item for item in fused if item["id"] in sop_execution],
            "sop_only_candidates": sop_only,
            "tree_only_candidates": [item for item in fused if item["id"] not in sop_execution],
            "evidence_refs": evidence_refs,
            "failure_patterns": failure_patterns,
            "risk_warnings": self._risk_warnings(sop_candidates),
            "visibility_warnings": visibility_warnings,
            "navigation_trace": trace,
            "fused_execution_candidates": fused,
            "prompt_execution_limit": prompt_execution_limit,
            "pre_gate_raw_candidates": pre_gate_raw_candidates,
            "final_prompt_candidate_ids": sorted(final_prompt_candidate_ids),
            "execution_candidate_provenance": execution_provenance,
            "execution_safety_gate": {
                "predicate": "positive_transition_or_clean_successful_run_node",
                "rejected": rejected_execution,
                "all_outputs_clean": all(
                    self._execution_candidate_eligibility(item["id"])[0]
                    for item in fused
                ),
            },
            "gateway_selection": selection_meta,
            "l3_agent_match": copy.deepcopy(l3_agent_match or {}),
            "visible_clause_ids": visibility_pack.effective_clause_ids,
            "visibility_trace": visibility_pack.visibility_trace,
            "visibility_safety_gate": {
                "mode": self.visibility_mode,
                "pre_ranking": True,
                "unauthorized_prompt_exposure": 0,
                "unauthorized_activation": 0,
                "all_sop_candidates_visible": all(
                    visibility_ids is None or item["id"] in visibility_ids
                    for item in sop_candidates
                ),
            },
        }

    @staticmethod
    def _mark_empty_visibility_abstention(pack: dict[str, Any]) -> None:
        trace = pack.get("visibility_trace")
        if not isinstance(trace, dict) or trace.get("empty_pack") is not True:
            return
        consumable_keys = (
            "direct_sop_candidates",
            "selected_sop_gateways",
            "fused_execution_candidates",
            "sop_only_candidates",
            "tree_only_candidates",
            "evidence_refs",
            "failure_patterns",
        )
        if any(pack.get(key) for key in consumable_keys):
            return
        disposition = {
            "status": "abstain",
            "reason": "empty_visible_pack",
            "legacy_fallback_used": False,
            "warning_preserved": True,
        }
        pack["visibility_abstention"] = disposition
        trace["consumer_disposition"] = copy.deepcopy(disposition)

    def _format_hybrid_pack(self, pack: dict[str, Any]) -> str:
        if pack.get("schema") == "experiment_r_memory_pack_v1":
            from agents.memory.experiment_r_router import format_experiment_r_pack

            return format_experiment_r_pack(self, pack)
        lines = [
            "## Stage-Aware Hybrid Run-Forest Memory",
            "Candidates are suggestions. Verified execution evidence and risk warnings are separate.",
            "Never present an SOP-only reference as a proven successful recipe.",
            f"Stage route: {json.dumps(pack['stage_route'], ensure_ascii=False)}",
        ]
        if (pack.get("visibility_abstention") or {}).get("status") == "abstain":
            lines += [
                "",
                "### Memory Abstention",
                "- No authorized memory clauses are available; no legacy fallback was used.",
            ]
        if pack["risk_warnings"]:
            lines += ["", "### Risk Warnings (do not adopt as positive recipes)"]
            for warning in pack["risk_warnings"][:6]:
                lines.append(
                    f"- SOP {warning['sop_id']} / transition {warning['transition_id']}: "
                    f"{warning['reason']} [{warning['disposition']}]"
                )
        if pack.get("visibility_warnings"):
            lines += ["", "### Authorized Diagnostic Warnings (navigation only)"]
            for warning in pack["visibility_warnings"][:8]:
                lines.append(
                    f"- {warning['clause_id']}: {warning['text']} "
                    f"[{warning['disposition']}]"
                )
        if pack["selected_sop_gateways"]:
            lines += ["", "### Selected SOP Gateways (clean supporting execution required)"]
            for gateway in pack["selected_sop_gateways"]:
                if self._visibility_is_enforced():
                    lines.append(
                        f"- {gateway['id']} clauses="
                        f"{', '.join(gateway.get('visible_clause_ids') or [])}"
                    )
                    lines.append(f"  Authorized text: {gateway.get('visible_text', '')}")
                else:
                    node = self.nodes[gateway["id"]]
                    lines.append(f"- {gateway['id']}: {node.get('title', '')}")
                    lines.append(f"  Action: {node.get('action', '')}")
                    lines.append(f"  When: {'; '.join(_as_list(node.get('applies_when')))}")
                lines.append(
                    f"  Supporting transitions: {', '.join(pack['gateway_transitions'].get(gateway['id'], []))}"
                )
        if pack["sop_transition_matches"] or pack["tree_only_candidates"]:
            lines += ["", "### Execution Candidates"]
            prompt_execution_limit = int(
                pack.get("prompt_execution_limit", self.top_k)
            )
            for item in pack["fused_execution_candidates"][
                :prompt_execution_limit
            ]:
                node = self.nodes.get(item["id"], {})
                lines.append(
                    f"- [{item['candidate_class']}] {item['id']} "
                    f"stage={node.get('stage') or node.get('stage_pair')} "
                    f"outcome={node.get('outcome')} metric_improvement={node.get('metric_improvement')}"
                )
                detail = next(
                    (row for row in pack.get("tree_candidate_details", []) if row.get("id") == item["id"]),
                    None,
                )
                if detail and node.get("type") == "Transition" and pack["stage_route"]["stage"] == "debug":
                    evidence = detail.get("transition_evidence") or {}
                    lines.append(f"  Parent failure: {str(evidence.get('parent_failure') or '')[:700]}")
                    lines.append(f"  Proven code change: {str(evidence.get('code_change') or '')[:700]}")
                    lines.append(f"  Successful child result: {str(evidence.get('child_result') or '')[:700]}")
                    unified_diff = str(evidence.get("unified_diff") or "")
                    repaired_code = str(evidence.get("after_code") or "")
                    if unified_diff:
                        lines.extend(
                            [
                                "  Exact historical code identities: "
                                f"before={evidence.get('before_code_sha256')} "
                                f"after={evidence.get('after_code_sha256')}",
                                "  <historical_repair_diff>",
                                unified_diff,
                                "  </historical_repair_diff>",
                            ]
                        )
                    if repaired_code:
                        lines.extend(
                            [
                                "  <successful_repaired_code>",
                                repaired_code,
                                "  </successful_repaired_code>",
                            ]
                        )
                    proven_sops = [row["sop_id"] for row in detail.get("causal_attachments", [])]
                    lines.append(f"  Causally supported SOPs only: {', '.join(proven_sops) or 'none'}")
                elif detail and node.get("type") == "Transition":
                    evidence = detail.get("transition_evidence") or {}
                    lines.append(
                        f"  Proven improvement action: {str(evidence.get('proven_action') or '')[:1200]}"
                    )
                    lines.append(
                        f"  Successful child result: {str(evidence.get('child_result') or '')[:700]}"
                    )
                elif (
                    node.get("type") == "RunNode"
                    and canonical_task_id(node.get("task"))
                    == canonical_task_id(pack.get("target_task_id"))
                    and self._is_terminal_strategy_evidence(item["id"], node)
                ):
                    implementation = self._format_node_implementation(
                        item["id"], heading="Exact Same-Task Terminal Implementation"
                    )
                    if implementation:
                        lines.extend(
                            [
                                "  Reproduce this measured implementation before "
                                "making optional improvements.",
                                implementation,
                            ]
                        )
        if pack["sop_only_candidates"]:
            lines += ["", "### SOP-Only Method References (unverified here)"]
            for candidate in pack["sop_only_candidates"][:4]:
                if self._visibility_is_enforced():
                    lines.append(
                        f"- {candidate['id']} clauses="
                        f"{', '.join(candidate.get('visible_clause_ids') or [])}: "
                        f"{candidate.get('visible_text', '')}"
                    )
                else:
                    node = self.nodes[candidate["id"]]
                    lines.append(f"- {candidate['id']}: {node.get('title', '')}; action={node.get('action', '')}")
        if pack["evidence_refs"]:
            lines += ["", "### Verified Evidence Refs"]
            for evidence_id in pack["evidence_refs"][:6]:
                lines.append(f"- {evidence_id}: {str(self.nodes.get(evidence_id, {}).get('text', ''))[:400]}")
        if pack["failure_patterns"]:
            lines += ["", "### Failure Patterns"]
            for pattern_id in pack["failure_patterns"][:6]:
                node = self.nodes.get(pattern_id, {})
                lines.append(f"- {pattern_id}: {node.get('issue_code')} {str(node.get('text', ''))[:300]}")
        return "\n".join(lines)

    def _end2end_candidate_text(
        self,
        candidate_id: str,
        *,
        source: str,
        detail: dict[str, Any],
    ) -> tuple[str, str]:
        node = self.nodes.get(candidate_id, {})
        if source == "sop":
            visible = str(detail.get("visible_text") or "").strip()
            if visible:
                return visible, "authorized SOP clause projection"
            parts = self._sop_text_parts(node)
            text = "\n".join(
                value for value in parts.values() if str(value).strip()
            ).strip()
            return text, "SOP procedure"
        evidence = detail.get("transition_evidence") or {}
        plan = str(
            node.get("plan")
            or node.get("description")
            or node.get("code_summary")
            or node.get("text")
            or ""
        ).strip()
        feedback_parts = [
            str(evidence.get("parent_failure") or "").strip(),
            str(evidence.get("code_change") or "").strip(),
            str(evidence.get("child_result") or "").strip(),
        ]
        feedback = " | ".join(value for value in feedback_parts if value)
        if feedback and feedback not in plan:
            plan = f"{plan}\nStructured execution feedback: {feedback}".strip()
        unified_diff = str(evidence.get("unified_diff") or "")
        repaired_code = str(evidence.get("after_code") or "")
        if unified_diff:
            plan = (
                f"{plan}\n<historical_repair_diff>\n{unified_diff}"
                "\n</historical_repair_diff>"
            ).strip()
        if repaired_code:
            plan = (
                f"{plan}\n<successful_repaired_code>\n{repaired_code}"
                "\n</successful_repaired_code>"
            ).strip()
        return plan, feedback or "verified successful execution"

    def _end2end_common_pool(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        query_text: str,
    ) -> tuple[list[Any], Any]:
        """Return one shared, authorized SOP + RunForest candidate pool."""

        from agents.memory.end2end_memory_system import MemoryCandidate

        visibility_pack = self._prepare_visibility(
            stage=stage,
            task_id=task_id,
            task_desc=task_desc,
        )
        visibility_ids = self._effective_visibility_sop_ids()
        sop_rows = self._rank_sops(
            query_text,
            stage,
            self.end2end_candidate_pool_limit,
            task_id=task_id,
            task_desc=task_desc,
            allowed_sop_ids=visibility_ids,
        )
        if stage == "debug":
            tree_rows = self._rank_debug_transition_rows(
                query_text=query_text,
                task_id=task_id,
                task_desc=task_desc,
                limit=self.end2end_candidate_pool_limit,
                allowed_sop_ids=visibility_ids,
            )
        else:
            tree_rows = self._rank_tree_rows(
                stage=stage,
                query_text=query_text,
                task_id=task_id,
                task_desc=task_desc,
                limit=self.end2end_candidate_pool_limit,
            )

        steps = [
            int(self.nodes.get(row.get("id"), {}).get("step") or 0)
            for row in tree_rows
        ]
        max_step = max(steps, default=0)
        candidates: list[MemoryCandidate] = []
        for source, rows in (("sop", sop_rows), ("runforest", tree_rows)):
            for source_rank, row in enumerate(rows, 1):
                candidate_id = str(row.get("id") or "")
                if not candidate_id:
                    continue
                node = self.nodes.get(candidate_id, {})
                text, feedback = self._end2end_candidate_text(
                    candidate_id,
                    source=source,
                    detail=row,
                )
                if not text:
                    continue
                relevance = self._bounded_token_similarity(query_text, text)
                success_support = int(
                    row.get("clean_supporting_transition_count") or 0
                )
                rejected_support = int(row.get("rejected_support_count") or 0)
                if source == "runforest":
                    if node.get("type") == "Transition":
                        verified_success = self._positive_transition(candidate_id)[0]
                    else:
                        verified_success = self._successful_run_node(candidate_id)
                    success_support = max(success_support, int(verified_success))
                else:
                    verified_success = success_support > 0
                step = int(node.get("step") or 0)
                recency = float(step / max_step) if max_step > 0 else 0.0
                if source == "sop":
                    stage_fit = float(bool(row.get("stage_compatible")))
                else:
                    stage_fit = float(
                        bool(
                            float(
                                (row.get("score_components") or {}).get("stage")
                                or 0.0
                            )
                            > 0
                            or stage == "debug"
                        )
                    )
                candidates.append(
                    MemoryCandidate(
                        candidate_id=candidate_id,
                        source=source,
                        relevance=float(relevance),
                        prompt_text=text,
                        source_stage=str(
                            node.get("stage") or node.get("stage_pair") or ""
                        ),
                        source_task_id=str(node.get("task") or ""),
                        rank=source_rank,
                        metadata={
                            "authorized": True,
                            "source_rank": source_rank,
                            "source_ranking_score": float(row.get("score") or 0.0),
                            "verified_success": bool(verified_success),
                            "success_support_count": success_support,
                            "rejected_support_count": rejected_support,
                            "failure_risk": min(1.0, rejected_support / 3.0),
                            "execution_feedback": feedback,
                            "score_delta": node.get("metric_improvement"),
                            "recency": recency,
                            "stage_fit": stage_fit,
                            "visible_clause_ids": list(
                                row.get("visible_clause_ids") or []
                            ),
                        },
                    )
                )
        return candidates, visibility_pack

    def _retrieve_end2end_for_node(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        query_parts: list[str] | None,
    ) -> tuple[str, list[str]]:
        from agents.memory.end2end_memory_system import MemorySystemContext

        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGE_QUOTAS:
            raise ValueError(f"Unsupported End2End retrieval stage: {stage}")
        if not self.stage_enabled(stage):
            return "", []
        query_text = "\n".join([task_desc or "", *(query_parts or [])])
        if STAGE_ALIASES.get(stage, stage) == "debug" and query_parts:
            # Task identity/type are scored through their dedicated hard gate.
            # Feeding the generic benchmark description into failure matching
            # can create false exact-task hits (for example, merely mentioning
            # ROC-AUC can look like an AUC-label failure).  Debug semantics must
            # therefore come from the observed runtime/error context itself.
            query_text = "\n".join(str(value) for value in query_parts if value)
        candidates, visibility_pack = self._end2end_common_pool(
            stage=stage,
            task_id=task_id,
            task_desc=task_desc,
            query_text=query_text,
        )
        snapshot = getattr(self, "memory_snapshot", None)
        base = getattr(snapshot, "base_bundle", None)
        selection = self.end2end_controller.retrieve(
            candidates,
            MemorySystemContext(
                stage=stage,
                task_id=task_id,
                task_description=task_desc,
                prompt_token_budget=self.end2end_prompt_token_budget,
                top_k=self.top_k,
                memory_bundle_manifest_sha256=str(
                    getattr(base, "manifest_sha256", "") or ""
                ),
            ),
        )
        selection_pack = selection.to_pack()
        raw_rows = [
            {
                "candidate_id": item.candidate_id,
                "rank": rank,
                "score": item.relevance,
                "source_run_id": self.nodes.get(item.candidate_id, {}).get(
                    "run_id"
                ),
                "source_task_id": item.source_task_id,
                "source_stage": item.source_stage,
                "operation_authorized": True,
                "gate_reason": "authorized_common_pool",
                "final_prompt_visible": (
                    item.candidate_id in selection.prompt_candidate_ids
                ),
            }
            for rank, item in enumerate(selection.raw_candidates, 1)
        ]
        suppressed_by_id = {
            str(item.get("candidate_id")): str(item.get("reason") or "")
            for item in selection.suppressed_candidates
        }
        navigation_trace = [
            {
                "retrieval_channel": "end2end_common_pool",
                "candidate_class": item.source,
                "gateway_sop_id": (
                    item.candidate_id if item.source == "sop" else None
                ),
                "candidate_id": item.candidate_id,
                "supporting_transition_ids": list(
                    item.metadata.get("visible_clause_ids") or []
                ),
                "selection_reason": (
                    "selected_by_frozen_system_policy"
                    if item.candidate_id in selection.prompt_candidate_ids
                    else suppressed_by_id.get(
                        item.candidate_id,
                        "not_selected_by_frozen_system_policy",
                    )
                ),
                "selection_state": (
                    "injected"
                    if item.candidate_id in selection.prompt_candidate_ids
                    else "suppressed"
                ),
            }
            for item in selection.raw_candidates
        ]
        pool_payload = [item.to_dict() for item in selection.raw_candidates]
        candidate_pool_hash = hashlib.sha256(
            json.dumps(
                pool_payload,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        visibility_trace = copy.deepcopy(
            getattr(visibility_pack, "visibility_trace", {}) or {}
        )
        pack = {
            "schema": "mlevolve_end2end_memory_pack_v1",
            "algorithm_version": "end2end_memory_systems_pilot_v1",
            "system_id": selection.system_id,
            "stage_route": {
                "stage": stage,
                "control": "end2end_memory_system",
                **selection.route,
            },
            "target_task_id": str(task_id),
            "candidate_pool_source": "shared_authority_filtered_sop_runforest",
            "candidate_pool_hash": candidate_pool_hash,
            "candidate_pool": pool_payload,
            "raw_pool_observed": True,
            "pre_gate_raw_candidates": raw_rows,
            "selected_candidates": selection_pack["selected_candidates"],
            "suppressed_candidates": selection_pack["suppressed_candidates"],
            "final_prompt_candidates": selection_pack["prompt_candidates"],
            "final_prompt_candidate_ids": list(selection.prompt_candidate_ids),
            "prompt_visible_refs": list(selection.prompt_candidate_ids),
            "prompt_text": selection.prompt_text,
            "prompt_token_count": selection.prompt_token_count,
            "prompt_truncated": selection.prompt_truncated,
            "navigation_trace": navigation_trace,
            "visible_clause_ids": list(
                getattr(visibility_pack, "effective_clause_ids", []) or []
            ),
            "visibility_trace": visibility_trace,
            "visibility_safety_gate": {
                "mode": self.visibility_mode,
                "pre_ranking": True,
                "unauthorized_prompt_exposure": 0,
                "unauthorized_activation": 0,
            },
            "unauthorized_prompt_exposure": 0,
            "memory_snapshot_bound_but_not_exposed": (
                selection.system_id == "no_memory"
            ),
            "memory_bundle": {
                "bundle_id": str(getattr(base, "bundle_id", "") or ""),
                "bundle_version": str(
                    getattr(base, "bundle_version", "") or ""
                ),
                "manifest_sha256": str(
                    getattr(base, "manifest_sha256", "") or ""
                ),
                "snapshot_sha256": str(
                    getattr(snapshot, "snapshot_sha256", "") or ""
                ),
            },
            # Compatibility fields consumed by existing audit/adoption code.
            "fused_execution_candidates": [],
            "selected_sop_gateways": [],
            "sop_only_candidates": [],
            "evidence_refs": [],
            "failure_patterns": [],
        }
        self._mark_empty_visibility_abstention(pack)
        self._last_agentic_pack = pack
        self._trace_local.pack = pack
        if self.prospective_audit_logger is not None:
            self.prospective_audit_logger.record_run_candidates(pack, self.nodes)
        return selection.prompt_text, list(selection.prompt_candidate_ids)

    def current_navigation_pack(self) -> dict[str, Any]:
        """Return a defensive copy of this thread's latest retrieval pack."""
        return copy.deepcopy(getattr(self._trace_local, "pack", {}))

    def _begin_navigation_request(self) -> None:
        """Clear request-local state before every retrieval decision.

        A branch that abstains or is role-gated must never inherit a sibling
        node's Router trace.  Clearing both fields up front also makes an
        exception observable as a missing current request rather than stale
        successful navigation.
        """

        self._trace_local.pack = {}
        self._trace_local.visibility_pack = None
        self._last_agentic_pack = {}

    def _record_role_policy_abstention(
        self,
        *,
        stage: str,
        task_id: str,
        draft_role: str | None,
        reason: str,
    ) -> None:
        canonical = STAGE_ALIASES.get(stage, stage)
        pack = {
            "schema": "stage_hybrid_role_policy_abstention_v1",
            "algorithm_version": "draft_origin_only_role_policy_v1",
            "stage_route": {
                "stage": canonical,
                "requested_generation_stage": str(stage),
                "control": self.retrieval_control,
                "route": "role_policy_abstention",
            },
            "target_task_id": str(task_id),
            "draft_role": str(draft_role or ""),
            "role_policy_abstention": {
                "status": "abstain",
                "reason": str(reason),
                "draft_only": canonical == "draft",
            },
            "navigation_trace": [],
            "selected_items": [],
            "selected_sop_gateways": [],
            "fused_execution_candidates": [],
            "sop_only_candidates": [],
            "evidence_refs": [],
            "failure_patterns": [],
            "final_prompt_candidate_ids": [],
            "final_prompt_candidates": [],
            "prompt_text": "",
            "prompt_token_count": 0,
            "prompt_truncated": False,
        }
        self._trace_local.pack = pack
        self._last_agentic_pack = pack

    def current_visibility_pack(self) -> VisibleSOPPack | None:
        """Return the host-only clause visibility result for this thread."""
        pack = getattr(self._trace_local, "visibility_pack", None)
        return copy.deepcopy(pack) if pack is not None else None

    @staticmethod
    def _prompt_visible_refs(text: str, refs: list[str]) -> list[str]:
        """Return exactly the side-channel IDs that survived prompt truncation."""

        return list(
            dict.fromkeys(
                str(ref_id)
                for ref_id in refs
                if ref_id and str(ref_id) in text
            )
        )

    def retrieve_for_node(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        query_parts: list[str] | None = None,
        draft_role: str | None = None,
        context: dict[str, Any] | None = None,
        strategy_context: dict[str, Any] | None = None,
        visibility_request: VisibilityRequest | None = None,
        authority_operation: Operation | str | None = None,
        active_protocol: ProtocolRef | str | None = None,
    ) -> tuple[str, list[str]]:
        self._begin_navigation_request()
        if self.end2end_controller is not None:
            return self._retrieve_end2end_for_node(
                stage=stage,
                task_id=task_id,
                task_desc=task_desc,
                query_parts=query_parts,
            )
        query_text = "\n".join([task_desc or "", *(query_parts or [])])
        if STAGE_ALIASES.get(stage, stage) == "debug" and query_parts:
            # Task identity/type are already handled by the hard task gate;
            # only observed runtime/error context belongs in signature match.
            query_text = "\n".join(str(value) for value in query_parts if value)
        if self.retrieval_control == "no_memory":
            if self.experiment_r_enabled:
                from agents.memory.experiment_r_router import build_no_memory_pack

                self._trace_local.visibility_pack = None
                self._trace_local.pack = build_no_memory_pack(
                    self,
                    stage=STAGE_ALIASES.get(stage, stage),
                    task_id=str(task_id),
                    task_desc=str(task_desc or ""),
                    query_text=query_text,
                    visibility_request=visibility_request,
                    authority_operation=authority_operation,
                    active_protocol=active_protocol,
                )
                return "", []
            self._trace_local.visibility_pack = None
            self._trace_local.pack = {
                "schema": "stage_hybrid_no_memory_pack_v1",
                "stage_route": {
                    "stage": STAGE_ALIASES.get(stage, stage),
                    "control": "no_memory",
                },
                "target_task_id": str(task_id),
                "prompt_text": "",
                "prompt_visible_refs": [],
                "visible_clause_ids": [],
                "fused_execution_candidates": [],
                "selected_sop_gateways": [],
                "sop_only_candidates": [],
                "evidence_refs": [],
                "failure_patterns": [],
                "unauthorized_prompt_exposure": 0,
                "memory_snapshot_bound_but_not_exposed": True,
            }
            return "", []
        canonical_stage = STAGE_ALIASES.get(stage, stage)
        if not self.stage_enabled(stage):
            self._record_role_policy_abstention(
                stage=stage,
                task_id=task_id,
                draft_role=draft_role,
                reason="generation_stage_disabled_by_memory_configuration",
            )
            return "", []
        # Branch roles choose only the initial Draft origin.  Improve, Debug,
        # Evolution and Fusion all regain the same Dynamic Router capability.
        if (
            canonical_stage == "draft"
            and draft_role in {"coldstart_baseline", "memory_reproduction"}
        ):
            self._record_role_policy_abstention(
                stage=stage,
                task_id=task_id,
                draft_role=draft_role,
                reason="draft_origin_policy_uses_no_router_prompt",
            )
            return "", []
        if self.retrieval_control == "layered_strategy" and stage == "draft":
            if draft_role == "memory_transfer":
                pack = self._hybrid_pack(
                    stage=stage,
                    task_id=task_id,
                    task_desc=task_desc,
                    query_text=query_text,
                    visibility_request=visibility_request,
                    authority_operation=authority_operation,
                    active_protocol=active_protocol,
                )
                self._mark_empty_visibility_abstention(pack)
                pack["memory_transfer"] = {
                    "activated": True,
                    "reason": "no_exact_task_replay_target",
                    "mode": "stage_hybrid_v2_clean_cross_task",
                }
                self._last_agentic_pack = pack
                self._trace_local.pack = pack
                if self.prospective_audit_logger is not None:
                    self.prospective_audit_logger.record_run_candidates(pack, self.nodes)
                refs = [item["id"] for item in pack["fused_execution_candidates"][: self.top_k]]
                refs += [item["id"] for item in pack["selected_sop_gateways"]]
                refs += [item["id"] for item in pack["sop_only_candidates"]]
                refs += pack["evidence_refs"] + pack["failure_patterns"]
                text = self._format_hybrid_pack(pack)
                if self.max_chars > 0 and len(text) > self.max_chars:
                    text = text[: self.max_chars].rstrip() + "\n... (memory-transfer pack truncated)"
                return text, self._prompt_visible_refs(text, refs)
            if draft_role != "novel_exploration":
                raise ValueError("Layered L1 Draft retrieval is restricted to novel_exploration")
            pack = self._layered_draft_pack(
                task_id=task_id,
                task_desc=task_desc,
                query_text=query_text,
                context=context,
                visibility_request=visibility_request,
                authority_operation=authority_operation,
                active_protocol=active_protocol,
            )
            self._mark_empty_visibility_abstention(pack)
            self._last_agentic_pack = pack
            self._trace_local.pack = pack
            if self.prospective_audit_logger is not None:
                self.prospective_audit_logger.record_run_candidates(pack, self.nodes)
            selected = pack.get("selected_strategy") or {}
            if selected:
                evidence = selected["best_tree_evidence"]
                refs = [
                    selected["sop_id"],
                    *([evidence["transition_id"]] if evidence.get("transition_id") else []),
                    evidence["node_id"],
                ]
                text = self._format_selected_strategy(pack)
            else:
                refs = [item["id"] for item in pack["fused_execution_candidates"][: self.top_k]]
                refs += [item["id"] for item in pack["selected_sop_gateways"]]
                refs += [item["id"] for item in pack["sop_only_candidates"]]
                refs += pack["evidence_refs"] + pack["failure_patterns"]
                text = self._format_hybrid_pack(pack)
            if self.max_chars > 0 and len(text) > self.max_chars:
                text = text[: self.max_chars].rstrip() + "\n... (layered strategy memory truncated)"
            return text, self._prompt_visible_refs(text, refs)
        pack = self._hybrid_pack(
            stage=stage,
            task_id=task_id,
            task_desc=task_desc,
            query_text=query_text,
            strategy_context=strategy_context,
            visibility_request=visibility_request,
            authority_operation=authority_operation,
            active_protocol=active_protocol,
        )
        if pack.get("schema") == "experiment_r_memory_pack_v1":
            pack["draft_role"] = str(draft_role or "")
            pack["requested_generation_stage"] = str(stage)
            pack.setdefault("stage_route", {})[
                "requested_generation_stage"
            ] = str(stage)
        self._mark_empty_visibility_abstention(pack)
        self._last_agentic_pack = pack
        self._trace_local.pack = pack
        if self.prospective_audit_logger is not None:
            self.prospective_audit_logger.record_run_candidates(pack, self.nodes)
        if (
            self.retrieval_control == "layered_strategy"
            and STAGE_ALIASES.get(stage, stage) == "debug"
            and not pack.get("selected_sop_gateways")
            and not pack.get("fused_execution_candidates")
            and not pack.get("sop_only_candidates")
        ):
            pack["memory_abstention"] = {
                "status": "abstain",
                "reason": (
                    pack.get("stage_route", {}).get("fallback_reason")
                    or "no_failure_signature_matched_clean_l3_repair"
                ),
            }
            return "", []
        text = self._format_hybrid_pack(pack)
        if pack.get("schema") == "experiment_r_memory_pack_v1":
            refs = list(pack.get("final_prompt_candidate_ids") or [])
        else:
            refs = [item["id"] for item in pack["fused_execution_candidates"][: self.top_k]]
            refs += [item["id"] for item in pack["selected_sop_gateways"]]
            refs += [item["id"] for item in pack["sop_only_candidates"]]
            refs += pack["evidence_refs"] + pack["failure_patterns"]
        if self.max_chars > 0 and len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "\n... (stage-hybrid memory truncated)"
        return text, self._prompt_visible_refs(text, refs)
