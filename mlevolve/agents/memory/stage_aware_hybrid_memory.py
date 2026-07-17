"""Stage-aware SOP gateway retrieval over the RunForest graph."""

from __future__ import annotations

import collections
import copy
import json
import logging
import math
import os
import re
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from agents.memory.external_skill_memory import RunForestMemoryLayer, _as_list, _tokenize

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

RETRIEVAL_CONTROLS = {"stage_hybrid", "layered_strategy", "sop_only", "tree_only", "naive_concat"}

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
    "new-york-city-taxi-fare-prediction": ("tabular", "tabular_regression"),
}

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
        **kwargs: Any,
    ) -> None:
        self._trace_local = threading.local()
        cfg = kwargs.get("cfg")
        ext_cfg = getattr(cfg, "external_skill_memory", None) if cfg is not None else None
        if stage_quotas is None and ext_cfg is not None:
            stage_quotas = getattr(ext_cfg, "stage_quotas", None)
        if rrf_weights is None and ext_cfg is not None:
            rrf_weights = getattr(ext_cfg, "rrf_weights", None)
        if blocked_run_prefixes is None and ext_cfg is not None:
            configured_prefixes = list(getattr(ext_cfg, "blocked_run_prefixes", None) or [])
            if configured_prefixes:
                blocked_run_prefixes = configured_prefixes
        if retrieval_control is None and ext_cfg is not None:
            retrieval_control = getattr(ext_cfg, "retrieval_control", None)
        self.retrieval_control = str(retrieval_control or "stage_hybrid")
        if self.retrieval_control not in RETRIEVAL_CONTROLS:
            raise ValueError(f"Unsupported stage-hybrid retrieval_control: {self.retrieval_control}")
        self.excluded_run_ids = {str(value) for value in (excluded_run_ids or [])}
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
        super().__init__(*args, **kwargs)
        if self.mode != "run_forest_stage_hybrid":
            raise ValueError("StageAwareHybridMemoryLayer requires mode=run_forest_stage_hybrid")
        self._build_sop_reverse_index()
        if self.retrieval_control == "layered_strategy":
            self._validate_layered_taxonomy()

    def _build_sop_reverse_index(self) -> None:
        self._transitions_by_sop: dict[str, list[str]] = collections.defaultdict(list)
        self._sops_by_execution: dict[str, list[str]] = collections.defaultdict(list)
        self._sop_links_by_execution: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
        for edge in self.graph.get("edges", []):
            if str(edge.get("kind") or edge.get("type")) != "distills_to":
                continue
            transition_id = str(edge.get("src", ""))
            sop_id = str(edge.get("dst", ""))
            if self.nodes.get(transition_id, {}).get("type") != "Transition":
                continue
            if self.nodes.get(sop_id, {}).get("type") != "SOP":
                continue
            self._transitions_by_sop[sop_id].append(transition_id)
            transition = self.nodes[transition_id]
            for execution_id in (
                transition_id,
                str(transition.get("parent_node_id") or ""),
                str(transition.get("child_node_id") or ""),
            ):
                if execution_id and sop_id not in self._sops_by_execution[execution_id]:
                    self._sops_by_execution[execution_id].append(sop_id)
                if execution_id:
                    links = self._sop_links_by_execution[execution_id].setdefault(sop_id, [])
                    if transition_id not in links:
                        links.append(transition_id)
        for values in self._transitions_by_sop.values():
            values.sort()
        for values in self._sops_by_execution.values():
            values.sort()
        for mapping in self._sop_links_by_execution.values():
            for values in mapping.values():
                values.sort()
        meta_prefixes = _as_list((self.graph.get("meta") or {}).get("blocked_run_prefixes"))
        override = self._blocked_run_prefixes_override
        self._blocked_run_prefixes = tuple(str(value) for value in (override if override is not None else meta_prefixes))

    def _validate_layered_taxonomy(self) -> None:
        meta = self.graph.get("meta") or {}
        if meta.get("sop_taxonomy_schema") != "runforest_sop_taxonomy_v1":
            raise ValueError("Layered strategy retrieval requires runforest_sop_taxonomy_v1")
        if float(meta.get("sop_taxonomy_coverage") or 0.0) != 1.0:
            raise ValueError("Layered strategy retrieval requires 100% SOP taxonomy coverage")
        if int(meta.get("sop_taxonomy_sop_count") or 0) != len(self._sops):
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
            for sop_id in self._sops
            if self.nodes[sop_id].get("abstraction_level") == "L1_strategy"
        ]
        reviewed_l1_count = meta.get("sop_taxonomy_reviewed_l1_count")
        if reviewed_l1_count is None or int(reviewed_l1_count) != len(l1_ids):
            raise ValueError("Layered strategy taxonomy has incomplete manual L1 review metadata")
        if any(self.nodes[sop_id].get("manual_reviewed") is not True for sop_id in l1_ids):
            raise ValueError("Layered strategy taxonomy contains an unreviewed L1 SOP")

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
        if any(token in text for token in ("siglip", "dinov2", "vision transformer")):
            return "vision_transformer_finetune"
        if any(token in text for token in ("lightgbm", "xgboost")):
            return "gradient_boosted_trees"
        raise ValueError(f"Cannot map model description to method_family: {value!r}")

    def _replay_family(self, task_id: str) -> str:
        if self.cfg is None:
            raise ValueError("Layered strategy retrieval requires cfg for replay-family exclusion")
        policy = getattr(self.cfg.agent, "draft_role_policy", None)
        manifest_path = self._resolve_config_path(getattr(policy, "replay_targets_path", ""))
        if not manifest_path.exists():
            raise FileNotFoundError(f"Replay target manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        target = next(
            (item for item in manifest.get("targets", []) if str(item.get("task_id")) == task_id),
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
        modality, task_family = TASK_PROFILES.get(task_id, ("unknown", "general"))
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
        else:
            baseline = str(context.get("baseline_model") or "")
            if not baseline:
                raise ValueError("Layered strategy retrieval requires the cold-start primary model")
            excluded = [self._model_family_from_text(baseline)]
            replay_family = self._replay_family(task_id)
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
            "task_id": task_id,
            "modality": modality,
            "task_family": task_family,
            "problem_type": task_family,
            "train_rows": train_rows,
            "train_size_band": size_band,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
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

    def _strategy_supports(self, sop_id: str, task_id: str) -> list[dict[str, Any]]:
        rows = []
        for transition_id in self._transitions_by_sop.get(sop_id, []):
            transition = self.nodes[transition_id]
            if str(transition.get("task") or "") != task_id:
                continue
            eligible, reason = self._positive_transition(transition_id)
            if not eligible:
                continue
            child_id = str(transition.get("child_node_id") or "")
            child = self.nodes.get(child_id, {})
            improvement = transition.get("metric_improvement")
            rows.append(
                {
                    "transition_id": transition_id,
                    "run_id": transition.get("run_id"),
                    "run_short_id": transition.get("run_short_id"),
                    "node_id": child_id,
                    "stage_pair": transition.get("stage_pair"),
                    "outcome": transition.get("outcome"),
                    "metric": child.get("metric"),
                    "metric_improvement": float(improvement) if isinstance(improvement, (int, float)) else 0.0,
                    "audit_status": (child.get("leakage_audit") or {}).get("status"),
                    "rank_eligible": (child.get("leakage_audit") or {}).get("rank_eligible"),
                    "code_sha256": child.get("code_sha256"),
                    "eligibility_reason": reason,
                }
            )
        return sorted(
            rows,
            key=lambda item: (
                -float(item.get("metric_improvement") or 0.0),
                str(item.get("node_id")),
            ),
        )

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
        for sop_id in self._sops:
            node = self.nodes[sop_id]
            if node.get("abstraction_level") != "L1_strategy" or node.get("sop_kind") != "model_strategy":
                continue
            if "draft" not in (node.get("decision_stages") or []):
                continue
            family = str(node.get("method_family") or "")
            if not family or family in excluded:
                continue
            task_families = {str(value) for value in (node.get("task_families") or [])}
            task_fit = 1.0 if task_family in task_families else 0.5 if "general" in task_families else 0.0
            if task_fit == 0.0:
                continue
            supports = self._strategy_supports(sop_id, str(task_profile["task_id"]))
            if not supports:
                continue
            semantic = self._token_overlap(query_tokens, self._node_tokens.get(sop_id, set()))
            evidence = min(1.0, math.log1p(len(supports)) / math.log(4.0))
            best_improvement = max(float(item.get("metric_improvement") or 0.0) for item in supports)
            rows.append(
                {
                    "sop_id": sop_id,
                    "raw_sop_id": node.get("sop_id"),
                    "title": node.get("title"),
                    "action": node.get("action"),
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
        rows.sort(key=lambda item: (-float(item["score"]), str(item["sop_id"])))
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
                    "strategy_sop_id": {"type": "string"},
                    "method_family": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "validation_plan": {"type": "string"},
                    "model_components": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
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
        return query(
            system_message=(
                "Select exactly one supplied strategy. Do not invent a method family or SOP id. "
                "Prefer a task-appropriate, compute-feasible hypothesis that differs from excluded families."
            ),
            user_message=json.dumps({"task_profile": task_profile, "routes": routes}, ensure_ascii=False),
            model=model,
            temperature=0.0,
            max_tokens=900,
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
    ) -> dict[str, Any]:
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
                    "supporting_transition_ids": [route["best_tree_evidence"]["transition_id"]],
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
                "supporting_transition_ids": [evidence["transition_id"]],
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
        return "\n".join(
            [
                "## Frozen Novel Strategy Contract",
                "The strategy was selected from three distinct L1 method families with clean Tree evidence.",
                f"Task profile: {json.dumps(pack['task_profile'], ensure_ascii=False)}",
                f"Selected SOP: {selected['sop_id']} - {selected.get('title', '')}",
                f"Primary method family: {selected['method_family']}",
                f"Action: {selected.get('action', '')}",
                f"Hypothesis: {decision.get('hypothesis', '')}",
                f"Validation plan: {decision.get('validation_plan', '')}",
                f"Model components: {', '.join(decision.get('model_components') or [])}",
                f"Clean Tree evidence: run={evidence.get('run_id')} node={evidence.get('node_id')} "
                f"transition={evidence.get('transition_id')} metric={evidence.get('metric')} "
                f"audit={evidence.get('audit_status')} code_sha256={evidence.get('code_sha256')}",
                "Do not replace this method family with an excluded baseline/replay family.",
            ]
        )

    def retrieve_model_design_tactics(
        self,
        *,
        task_id: str,
        task_desc: str,
        strategy_context: dict[str, Any],
    ) -> tuple[str, list[str], dict[str, Any]]:
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
        for sop_id in self._sops:
            node = self.nodes[sop_id]
            if node.get("abstraction_level") != "L2_tactic":
                continue
            if "model_design" not in (node.get("decision_stages") or []):
                continue
            if node.get("sop_kind") not in {"architecture", "feature", "training_protocol", "validation_protocol"}:
                continue
            node_family = str(node.get("method_family") or "general")
            if not self._family_compatible(family, node_family):
                continue
            families = {str(value) for value in (node.get("task_families") or [])}
            if task_family not in families and "general" not in families:
                continue
            supports = self._strategy_supports(sop_id, task_id)
            if not supports:
                continue
            semantic = self._token_overlap(query_tokens, self._node_tokens.get(sop_id, set()))
            score = 0.55 * semantic + 0.25 * min(1.0, len(supports) / 3.0) + 0.20 * (1.0 if node_family == family else 0.5)
            rows.append(
                {
                    "sop_id": sop_id,
                    "title": node.get("title"),
                    "action": node.get("action"),
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
            refs.extend([tactic["sop_id"], evidence["transition_id"], evidence["node_id"]])
            trace.append(
                {
                    "retrieval_channel": "l2_model_design",
                    "candidate_class": "family_compatible_tactic",
                    "gateway_sop_id": tactic["sop_id"],
                    "candidate_id": tactic["sop_id"],
                    "supporting_transition_ids": [evidence["transition_id"]],
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
        run_id = str(transition.get("run_short_id") or transition.get("run_id") or "")
        if run_id in self.excluded_run_ids:
            return False, "held_out_run"
        if any(run_id.startswith(prefix) for prefix in self._blocked_run_prefixes):
            return False, "blocked_run_prefix"
        if transition.get("quarantined") is True or transition.get("protocol_biased") is True:
            return False, "transition_quarantined_or_protocol_biased"
        child = self.nodes.get(str(transition.get("child_node_id") or ""), {})
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
        return bool(
            audit.get("rank_eligible") is True
            and node.get("is_buggy") is False
            and node.get("is_valid") is True
            and isinstance(metric, (int, float))
            and not isinstance(metric, bool)
            and math.isfinite(float(metric))
            and run_id not in self.excluded_run_ids
            and not any(run_id.startswith(prefix) for prefix in self._blocked_run_prefixes)
        )

    def _sop_text_parts(self, node: dict[str, Any]) -> dict[str, str]:
        return {
            "semantic": " ".join(str(node.get(key) or "") for key in ("title", "action", "text")),
            "conditions": " ".join(_as_list(node.get("applies_when")) + _as_list(node.get("condition"))),
            "failures": " ".join(_as_list(node.get("prevents")) + _as_list(node.get("failure_modes"))),
            "evidence": " ".join(_as_list(node.get("evidence_turns")) + _as_list(node.get("source_branches"))),
        }

    def _task_family_for_query(self, task_id: str, task_desc: str) -> str:
        configured = TASK_PROFILES.get(str(task_id or ""))
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
        for transition_id in self._transitions_by_sop.get(sop_id, []):
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
        candidate_ids = [
            sop_id
            for sop_id in self._sops
            if allowed_sop_ids is None or sop_id in allowed_sop_ids
        ]
        coords = self._coords()
        anchor = self._query_anchor(query_text, candidate_ids)
        field_weights = STAGE_SOP_FIELD_WEIGHTS[stage]
        rows = []
        for sop_id in candidate_ids:
            node = self.nodes[sop_id]
            if allowed_levels and str(node.get("abstraction_level") or "") not in allowed_levels:
                continue
            node_family = str(node.get("method_family") or "general")
            if method_family and not self._family_compatible(method_family, node_family):
                continue
            parts = self._sop_text_parts(node)
            scores = {
                key: min(1.0, self._token_overlap(query_tokens, _tokenize(text)))
                for key, text in parts.items()
            }
            field_relevance = sum(field_weights[key] * scores[key] for key in field_weights)
            stage_fit, stage_compatible = self._sop_stage_fit(node, stage)
            task_fit = self._sop_task_fit(node, task_family)
            task_compatible = self._sop_task_compatible(node, task_family)
            geometry = 0.0
            if anchor is not None and sop_id in coords:
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
                    "gateway_ids": {"type": "array", "items": {"type": "string"}},
                    "reasons": {"type": "object", "additionalProperties": {"type": "string"}},
                    "goal": {"type": "string"},
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
        return query(
            system_message="Select only supplied clean SOP gateway IDs. Do not invent IDs.",
            user_message=json.dumps({"stage": stage, "query": query_text[-5000:], "eligible": eligible}, ensure_ascii=False),
            model=model,
            temperature=0.0,
            max_tokens=900,
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

    def _task_families_compatible(self, left: str, right: str) -> bool:
        if left == right:
            return True
        related = [
            {"image_classification", "image_binary_classification"},
        ]
        return any(left in group and right in group for group in related)

    def _debug_transition_task_fit(
        self,
        transition: dict[str, Any],
        *,
        task_id: str,
        task_family: str,
    ) -> float:
        source_task = str(transition.get("task") or "")
        if source_task and source_task == task_id:
            return 1.0
        source_family = self._task_family_for_query(source_task, source_task.replace("-", " "))
        if self._task_families_compatible(task_family, source_family):
            return 0.90
        attached_families = {
            str(value)
            for sop_id in (transition.get("attached_sop_ids") or [])
            for value in (self.nodes.get(str(sop_id), {}).get("task_families") or [])
            if str(value) != "general"
        }
        if any(self._task_families_compatible(task_family, value) for value in attached_families):
            return 0.75
        return 0.0

    def _debug_parent_failure_text(self, transition: dict[str, Any]) -> str:
        parent = self.nodes.get(str(transition.get("parent_node_id") or ""), {})
        return " ".join(
            str(parent.get(key) or "")
            for key in ("analysis", "terminal_excerpt", "plan", "code_summary", "text")
        )

    def _debug_transition_evidence(self, transition: dict[str, Any]) -> dict[str, str]:
        parent = self.nodes.get(str(transition.get("parent_node_id") or ""), {})
        child = self.nodes.get(str(transition.get("child_node_id") or ""), {})
        parent_failure = str(parent.get("analysis") or parent.get("terminal_excerpt") or "").strip()
        code_change = str(child.get("plan") or child.get("code_summary") or transition.get("text") or "").strip()
        child_result = str(child.get("analysis") or child.get("terminal_excerpt") or "").strip()
        return {
            "parent_failure": parent_failure,
            "code_change": code_change,
            "child_result": child_result,
        }

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
        for raw_sop_id in transition.get("attached_sop_ids") or []:
            sop_id = str(raw_sop_id)
            if allowed_sop_ids is not None and sop_id not in allowed_sop_ids:
                continue
            sop = self.nodes.get(sop_id, {})
            if sop.get("type") != "SOP":
                continue
            _stage_score, stage_compatible = self._sop_stage_fit(sop, stage)
            if not stage_compatible or not self._sop_task_compatible(sop, task_family):
                continue
            quality = quality_by_sop.get(sop_id, {})
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
        if not query_signature:
            return []
        task_family = self._task_family_for_query(task_id, task_desc)
        coords = self._coords()
        eligible_transition_ids = [
            transition_id
            for transition_id in self._transitions
            if self._positive_transition(transition_id)[0]
            and (allowed_transition_ids is None or transition_id in allowed_transition_ids)
        ]
        anchor = self._query_anchor(query_text, eligible_transition_ids)
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
            if task_fit < 0.75:
                continue
            failure_text = self._debug_parent_failure_text(transition)
            candidate_signature = self._failure_signature(failure_text)
            if query_signature:
                overlap = len(query_signature & candidate_signature)
                failure_match = overlap / len(query_signature | candidate_signature) if overlap else 0.0
                if failure_match == 0.0:
                    continue
            else:
                failure_match = 0.0
            semantic = self._bounded_token_similarity(query_text, failure_text)
            attachments = self._causal_attachment_rows(
                transition,
                stage="debug",
                task_family=task_family,
                allowed_sop_ids=allowed_sop_ids,
            )
            if not attachments:
                continue
            attachment_quality = max(item["quality_score"] for item in attachments)
            geometry = 0.0
            if anchor is not None and transition_id in coords:
                geometry = 1.0 / (1.0 + self._distance(anchor, coords[transition_id]))
            score = (
                0.40 * failure_match
                + 0.20 * semantic
                + 0.20 * task_fit
                + 0.10 * attachment_quality
                + 0.10 * geometry
            )
            confidence = (
                0.55 * failure_match + 0.20 * task_fit + 0.15 * attachment_quality + 0.10 * semantic
                if query_signature
                else 0.40 * semantic + 0.35 * task_fit + 0.25 * attachment_quality
            )
            rows.append(
                {
                    "id": transition_id,
                    "score": score,
                    "confidence": min(1.0, confidence),
                    "score_components": {
                        "failure_signature": failure_match,
                        "semantic": semantic,
                        "task": task_fit,
                        "causal_attachment": attachment_quality,
                        "geometry": geometry,
                    },
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

    def _debug_dynamic_weights(self, transition_rows: list[dict[str, Any]]) -> tuple[dict[str, float], float, str | None]:
        confidence = max((float(row.get("confidence") or 0.0) for row in transition_rows), default=0.0)
        if confidence < DEBUG_TREE_CONFIDENCE_THRESHOLD:
            return {"sop": 1.0, "tree": 0.0}, confidence, "insufficient_causal_tree_confidence"
        configured_tree = float(self.rrf_weights["debug"]["tree"])
        tree_weight = min(configured_tree, DEBUG_TREE_MAX_WEIGHT, DEBUG_TREE_MAX_WEIGHT * confidence)
        return {"sop": 1.0 - tree_weight, "tree": tree_weight}, confidence, None

    def _rank_tree_rows(
        self, *, stage: str, query_text: str, task_id: str, task_desc: str, limit: int
    ) -> list[dict[str, Any]]:
        candidates = [
            node_id
            for node_id in self._run_nodes
            if self._successful_run_node(node_id)
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
                positive_improvements_by_task[str(candidate.get("task") or "unknown")].append(float(improvement))
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
                task_improvements = positive_improvements_by_task.get(str(node.get("task") or "unknown"), [])
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
        for execution_id in tree_ids:
            for sop_id in self._sops_by_execution.get(execution_id, []):
                if allowed_sop_ids is not None and sop_id not in allowed_sop_ids:
                    continue
                linked_transitions = self._sop_links_by_execution.get(execution_id, {}).get(sop_id, [])
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
    ) -> dict[str, Any]:
        """Return the production channel logic projected onto the SOP decision space."""
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGE_QUOTAS:
            raise ValueError(f"Unsupported stage-hybrid stage: {stage}")
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
            },
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

    def _hybrid_pack(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        query_text: str,
        strategy_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        stage = STAGE_ALIASES.get(stage, stage)
        if stage not in STAGE_QUOTAS:
            raise ValueError(f"Unsupported stage-hybrid stage: {stage}")
        quotas = self.stage_quotas[stage]
        allowed_levels = None
        method_family = None
        if self.retrieval_control == "layered_strategy":
            allowed_levels = {"L3_repair"} if stage == "debug" else {"L2_tactic"}
            selected = (strategy_context or {}).get("selected_strategy") or (strategy_context or {})
            method_family = str(selected.get("method_family") or "") or None
        ranked_sops = self._rank_sops(
            query_text,
            stage,
            len(self._sops),
            allowed_levels=allowed_levels,
            method_family=method_family,
            task_id=task_id,
            task_desc=task_desc,
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
            tree_rows = self._rank_debug_transition_rows(
                query_text=query_text,
                task_id=task_id,
                task_desc=task_desc,
                limit=quotas["tree_candidates"],
            )
            weights, tree_confidence, tree_fallback_reason = self._debug_dynamic_weights(tree_rows)
            if tree_fallback_reason and self.retrieval_control in {"stage_hybrid", "layered_strategy"}:
                tree_rows = []
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
        rejected_execution = []
        clean_fused = []
        for item in fused:
            eligible, reason = self._execution_candidate_eligibility(item["id"])
            if eligible:
                clean_fused.append(item)
            else:
                rejected_execution.append({"candidate_id": item["id"], "reason": reason})
        fused = clean_fused
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
        for item in fused[: self.top_k]:
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
        return {
            "schema": PACK_SCHEMA,
            "algorithm_version": "stage_hybrid_v2",
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
            "navigation_trace": trace,
            "fused_execution_candidates": fused,
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
        }

    def _format_hybrid_pack(self, pack: dict[str, Any]) -> str:
        lines = [
            "## Stage-Aware Hybrid Run-Forest Memory",
            "Candidates are suggestions. Verified execution evidence and risk warnings are separate.",
            "Never present an SOP-only reference as a proven successful recipe.",
            f"Stage route: {json.dumps(pack['stage_route'], ensure_ascii=False)}",
        ]
        if pack["risk_warnings"]:
            lines += ["", "### Risk Warnings (do not adopt as positive recipes)"]
            for warning in pack["risk_warnings"][:6]:
                lines.append(
                    f"- SOP {warning['sop_id']} / transition {warning['transition_id']}: "
                    f"{warning['reason']} [{warning['disposition']}]"
                )
        if pack["selected_sop_gateways"]:
            lines += ["", "### Selected SOP Gateways (clean supporting execution required)"]
            for gateway in pack["selected_sop_gateways"]:
                node = self.nodes[gateway["id"]]
                lines.append(f"- {gateway['id']}: {node.get('title', '')}")
                lines.append(f"  Action: {node.get('action', '')}")
                lines.append(f"  When: {'; '.join(_as_list(node.get('applies_when')))}")
                lines.append(
                    f"  Supporting transitions: {', '.join(pack['gateway_transitions'].get(gateway['id'], []))}"
                )
        if pack["sop_transition_matches"] or pack["tree_only_candidates"]:
            lines += ["", "### Execution Candidates"]
            for item in pack["fused_execution_candidates"][: self.top_k]:
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
                    proven_sops = [row["sop_id"] for row in detail.get("causal_attachments", [])]
                    lines.append(f"  Causally supported SOPs only: {', '.join(proven_sops) or 'none'}")
        if pack["sop_only_candidates"]:
            lines += ["", "### SOP-Only Method References (unverified here)"]
            for candidate in pack["sop_only_candidates"][:4]:
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

    def current_navigation_pack(self) -> dict[str, Any]:
        """Return a defensive copy of this thread's latest retrieval pack."""
        return copy.deepcopy(getattr(self._trace_local, "pack", {}))

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
    ) -> tuple[str, list[str]]:
        if not self.stage_enabled(stage):
            return "", []
        query_text = "\n".join([task_desc or "", *(query_parts or [])])
        if draft_role in {"coldstart_baseline", "memory_reproduction"}:
            return "", []
        if self.retrieval_control == "layered_strategy" and stage == "draft":
            if draft_role == "memory_transfer":
                pack = self._hybrid_pack(
                    stage=stage,
                    task_id=task_id,
                    task_desc=task_desc,
                    query_text=query_text,
                )
                pack["memory_transfer"] = {
                    "activated": True,
                    "reason": "no_exact_task_replay_target",
                    "mode": "stage_hybrid_v2_clean_cross_task",
                }
                self._last_agentic_pack = pack
                self._trace_local.pack = pack
                refs = [item["id"] for item in pack["fused_execution_candidates"][: self.top_k]]
                refs += [item["id"] for item in pack["selected_sop_gateways"]]
                refs += pack["evidence_refs"] + pack["failure_patterns"]
                text = self._format_hybrid_pack(pack)
                if self.max_chars > 0 and len(text) > self.max_chars:
                    text = text[: self.max_chars].rstrip() + "\n... (memory-transfer pack truncated)"
                return text, list(dict.fromkeys(refs))
            if draft_role != "novel_exploration":
                raise ValueError("Layered L1 Draft retrieval is restricted to novel_exploration")
            pack = self._layered_draft_pack(
                task_id=task_id,
                task_desc=task_desc,
                query_text=query_text,
                context=context,
            )
            self._last_agentic_pack = pack
            self._trace_local.pack = pack
            selected = pack.get("selected_strategy") or {}
            if selected:
                evidence = selected["best_tree_evidence"]
                refs = [selected["sop_id"], evidence["transition_id"], evidence["node_id"]]
                text = self._format_selected_strategy(pack)
            else:
                refs = [item["id"] for item in pack["fused_execution_candidates"][: self.top_k]]
                refs += [item["id"] for item in pack["selected_sop_gateways"]]
                refs += pack["evidence_refs"] + pack["failure_patterns"]
                text = self._format_hybrid_pack(pack)
            if self.max_chars > 0 and len(text) > self.max_chars:
                text = text[: self.max_chars].rstrip() + "\n... (layered strategy memory truncated)"
            return text, list(dict.fromkeys(refs))
        pack = self._hybrid_pack(
            stage=stage,
            task_id=task_id,
            task_desc=task_desc,
            query_text=query_text,
            strategy_context=strategy_context,
        )
        self._last_agentic_pack = pack
        self._trace_local.pack = pack
        refs = [item["id"] for item in pack["fused_execution_candidates"][: self.top_k]]
        refs += [item["id"] for item in pack["selected_sop_gateways"]]
        refs += pack["evidence_refs"] + pack["failure_patterns"]
        text = self._format_hybrid_pack(pack)
        if self.max_chars > 0 and len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "\n... (stage-hybrid memory truncated)"
        return text, list(dict.fromkeys(refs))
