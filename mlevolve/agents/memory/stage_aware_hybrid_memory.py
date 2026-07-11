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
        for values in self._transitions_by_sop.values():
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
            raise ValueError(f"No replay target exists for task {task_id}")
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
            excluded = [self._model_family_from_text(baseline), self._replay_family(task_id)]
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
            raise ValueError(
                "insufficient_strategy_coverage: "
                f"task={task_id} eligible_distinct_families={len(routes)} required={self.strategy_route_count}"
            )
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

    def _rank_sops(
        self,
        query_text: str,
        stage: str,
        limit: int,
        *,
        allowed_levels: set[str] | None = None,
        method_family: str | None = None,
    ) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query_text)
        rows = []
        for sop_id in self._sops:
            node = self.nodes[sop_id]
            if allowed_levels and str(node.get("abstraction_level") or "") not in allowed_levels:
                continue
            node_family = str(node.get("method_family") or "general")
            if method_family and not self._family_compatible(method_family, node_family):
                continue
            parts = self._sop_text_parts(node)
            scores = {
                key: self._token_overlap(query_tokens, _tokenize(text))
                for key, text in parts.items()
            }
            score = (
                0.50 * scores["semantic"]
                + 0.22 * scores["conditions"]
                + 0.18 * scores["failures"]
                + 0.10 * scores["evidence"]
            )
            if stage == "debug":
                score += 0.12 * scores["failures"]
            clean = []
            rejected = []
            for transition_id in self._transitions_by_sop.get(sop_id, []):
                eligible, reason = self._positive_transition(transition_id)
                (clean if eligible else rejected).append(
                    transition_id if eligible else {"transition_id": transition_id, "reason": reason}
                )
            rows.append(
                {
                    "id": sop_id,
                    "score": score,
                    "score_components": scores,
                    "ranking_backend": "field_aware_lexical",
                    "abstraction_level": node.get("abstraction_level"),
                    "sop_kind": node.get("sop_kind"),
                    "method_family": node_family,
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
        eligible = [item for item in candidates if item["clean_supporting_transition_ids"]]
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
        return selected, {"mode": mode, "llm_tool_calls": llm_calls, "goal": goal, "eligible_count": len(eligible)}

    def _append_unique(self, output: list[str], node_id: str) -> None:
        if node_id in self.nodes and node_id not in output:
            output.append(node_id)

    def _expand_gateways(
        self, selected: list[dict[str, Any]]
    ) -> tuple[list[str], dict[str, list[str]], list[str], list[str], list[dict[str, Any]]]:
        execution_ids: list[str] = []
        gateway_transitions: dict[str, list[str]] = {}
        evidence_refs: list[str] = []
        failure_patterns: list[str] = []
        trace: list[dict[str, Any]] = []
        for gateway in selected:
            sop_id = gateway["id"]
            transitions = list(gateway["clean_supporting_transition_ids"][:2])
            gateway_transitions[sop_id] = transitions
            for transition_id in transitions:
                transition = self.nodes[transition_id]
                expanded_for_transition: list[str] = []
                self._append_unique(execution_ids, transition_id)
                self._append_unique(expanded_for_transition, transition_id)
                parent_id = str(transition.get("parent_node_id") or "")
                child_id = str(transition.get("child_node_id") or "")
                for node_id in (parent_id, child_id):
                    if self._positive_memory_eligible(self.nodes.get(node_id, {})):
                        self._append_unique(execution_ids, node_id)
                        self._append_unique(expanded_for_transition, node_id)
                for node_id in self._ancestor_path(child_id, max_hops=5):
                    if self._positive_memory_eligible(self.nodes.get(node_id, {})):
                        self._append_unique(execution_ids, node_id)
                        self._append_unique(expanded_for_transition, node_id)
                local_best = str(self.nodes.get(child_id, {}).get("local_best_node_id") or "")
                if self._positive_memory_eligible(self.nodes.get(local_best, {})):
                    self._append_unique(execution_ids, local_best)
                    self._append_unique(expanded_for_transition, local_best)
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
        )

    def _rank_tree(self, *, stage: str, query_text: str, task_id: str, task_desc: str, limit: int) -> list[str]:
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
        return self._rank(
            query_text=query_text,
            candidate_ids=candidates,
            task_id=task_id,
            task_desc=task_desc,
            top_k=limit,
            stage_bonus=stage_bonus,
        )

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
        sop_candidates = self._rank_sops(
            query_text,
            stage,
            quotas["sop_candidates"],
            allowed_levels=allowed_levels,
            method_family=method_family,
        )
        selected, selection_meta = self._select_gateways(
            sop_candidates, stage=stage, query_text=query_text, limit=quotas["sop_gateways"]
        )
        sop_execution, gateway_transitions, evidence_refs, failure_patterns, trace = self._expand_gateways(selected)
        tree_ids = self._rank_tree(
            stage=stage,
            query_text=query_text,
            task_id=task_id,
            task_desc=task_desc,
            limit=quotas["tree_candidates"],
        )
        weights = self.rrf_weights[stage]
        if self.retrieval_control == "sop_only":
            tree_ids = []
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
        selected_sop_ids = {item["id"] for item in selected}
        for item in sop_candidates:
            trace.append(
                {
                    "retrieval_channel": "sop_direct",
                    "candidate_class": (
                        "sop_transition_matches" if item["id"] in selected_sop_ids else "sop_only_candidates"
                    ),
                    "gateway_sop_id": item["id"] if item["id"] in selected_sop_ids else None,
                    "supporting_transition_ids": item["clean_supporting_transition_ids"],
                    "selection_reason": next(
                        (value["selection_reason"] for value in selected if value["id"] == item["id"]),
                        "not selected as a formal gateway",
                    ),
                    "selection_state": "selected" if item["id"] in selected_sop_ids else "candidate",
                }
            )
        for node_id in tree_ids:
            trace.append(
                {
                    "retrieval_channel": "tree_direct",
                    "candidate_class": (
                        "sop_transition_matches" if node_id in sop_execution else "tree_only_candidates"
                    ),
                    "gateway_sop_id": None,
                    "supporting_transition_ids": [],
                    "selection_reason": f"independent {STAGE_ROUTE[stage]} tree ranking",
                    "selection_state": "selected",
                }
            )
        for item in fused[: self.top_k]:
            trace.append(
                {
                    "retrieval_channel": "hybrid_rrf",
                    "candidate_class": item["candidate_class"],
                    "gateway_sop_id": None,
                    "supporting_transition_ids": [],
                    "selection_reason": f"weighted RRF score={item['rrf_score']:.8f}",
                    "selection_state": "injected",
                    "candidate_id": item["id"],
                }
            )
        sop_only = [item for item in sop_candidates if item["id"] not in selected_sop_ids]
        return {
            "schema": PACK_SCHEMA,
            "stage_route": {"stage": stage, "route": STAGE_ROUTE[stage], "control": self.retrieval_control, "quotas": quotas, "rrf": weights},
            "direct_sop_candidates": sop_candidates,
            "selected_sop_gateways": selected,
            "gateway_transitions": gateway_transitions,
            "tree_candidates": tree_ids,
            "sop_transition_matches": [item for item in fused if item["id"] in sop_execution],
            "sop_only_candidates": sop_only,
            "tree_only_candidates": [item for item in fused if item["id"] not in sop_execution],
            "evidence_refs": evidence_refs,
            "failure_patterns": failure_patterns,
            "risk_warnings": self._risk_warnings(sop_candidates),
            "navigation_trace": trace,
            "fused_execution_candidates": fused,
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
            selected = pack["selected_strategy"]
            evidence = selected["best_tree_evidence"]
            refs = [selected["sop_id"], evidence["transition_id"], evidence["node_id"]]
            text = self._format_selected_strategy(pack)
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
