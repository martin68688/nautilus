"""
Build a bootstrap hyperbolic SOP memory from the adapted SkillGraph-C artifact.

This is the v0 bridge requested for the hyperbolic structural-memory branch:
  1. Read graph_skillgraph_c_trace_prereq.json.
  2. Treat compact-card SkillGraph nodes as SOP-like nodes.
  3. Compute initial Poincare/Lorentz coordinates from text direction + evidence radius.
  4. Run one GraphBuilderAgent pass to add Condition / FailureMode / Evidence nodes and
     applies_when / prevents / refines / conflicts_with edges.
  5. Validate patches and write hyper_graph.json, hyper_index.npz, graph_builder_report.json.

The GraphBuilderAgent is deterministic by default so the artifact is reproducible offline.
It is deliberately shaped as patch proposal + programmatic validation: future LLM-backed
builders can emit the same patch schema without being allowed to mutate the graph directly.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "paper-skills" / "distillation" / "graph_build" / "graph_skillgraph_c_trace_prereq.json"
DEFAULT_OUT = REPO / "paper-skills" / "hyper_memory"

ALLOWED_EDGE_KINDS = {
    "contains",
    "applies_when",
    "prevents",
    "supported_by",
    "implemented_by",
    "refines",
    "conflicts_with",
    "co_occur",
    "enhance",
    "prereq",
}

FAILURE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("overfitting", ("overfit", "overfitting", "validation loss", "small dataset", "small data", "regularization")),
    ("poor calibration", ("calibration", "overconfident", "label smoothing", "focal loss", "log loss")),
    ("data leakage", ("leak", "leakage", "train fold", "validation fold", "fit on train", "train-only")),
    ("out of memory", ("oom", "out of memory", "memory", "gpu memory", "cuda memory")),
    ("timeout or slow execution", ("timeout", "slow", "speed", "fast", "num_workers", "cache", "runtime")),
    ("training instability", ("unstable", "stabilize", "gradient", "clipping", "nan", "overflow", "amp")),
    ("underfitting or low capacity", ("underfit", "capacity", "smaller model", "small model", "distilbert")),
    ("syntax or generated-code artifact", ("syntax", "merge marker", "artifact", "undefined", "duplicate", "script-order")),
    ("api or checkpoint mismatch", ("attribute", "api", "checkpoint", "load", "path", "file not found", "pretrained")),
    ("submission format mismatch", ("submission", "sample_submission", "column", "csv", "format")),
    ("noisy or dead-code features", ("dead-code", "unused", "noisy", "handcrafted", "feature not connected")),
]

OPPOSING_TERMS: tuple[tuple[str, str], ...] = (
    ("large", "small"),
    ("larger", "smaller"),
    ("complex", "simple"),
    ("full", "partial"),
    ("increase", "reduce"),
    ("enable", "disable"),
    ("freeze", "unfreeze"),
)


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_len] or "empty"


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = clean_text(value)
    return [text] if text else []


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", (text or "").lower()))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def radius_band(radius: float) -> str:
    if radius <= 0.35:
        return "core"
    if radius <= 0.60:
        return "middle"
    return "edge"


def poincare_to_lorentz(points: np.ndarray) -> np.ndarray:
    norms2 = np.sum(points * points, axis=1, keepdims=True)
    norms2 = np.clip(norms2, 0.0, 0.98 ** 2)
    denom = np.maximum(1e-9, 1.0 - norms2)
    time = (1.0 + norms2) / denom
    space = (2.0 * points) / denom
    return np.concatenate([time, space], axis=1)


def direction_to_angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return spherical angles for the first three direction dimensions.

    theta is the azimuth around the x/y plane in [0, 2*pi). phi is the polar
    angle from the positive z axis in [0, pi]. The direction vector remains the
    canonical coordinate carrier; these angles make the "theta/radius" map view
    explicit in JSON and npz artifacts.
    """
    xyz = directions[:, :3]
    if xyz.shape[1] < 3:
        xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])), constant_values=0.0)
    norms = np.maximum(np.linalg.norm(xyz, axis=1), 1e-9)
    theta = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * math.pi)
    phi = np.arccos(np.clip(xyz[:, 2] / norms, -1.0, 1.0))
    return theta.astype(np.float32), phi.astype(np.float32)


def node_text_for_direction(node: dict[str, Any]) -> str:
    structured = [
        clean_text(node.get("title")),
        clean_text(node.get("principle")),
        clean_text(node.get("action")),
        clean_text(node.get("condition")),
        " ".join(clean_text(x) for x in as_list(node.get("applies_when"))),
        " ".join(clean_text(x) for x in as_list(node.get("prevents"))),
        " ".join(clean_text(x) for x in as_list(node.get("failure_modes"))),
        clean_text(node.get("category")),
        clean_text(node.get("scope")),
        " ".join(clean_text(x) for x in as_list(node.get("implementation_ids"))),
        " ".join(clean_text(x) for x in as_list(node.get("reference_ids"))),
    ]
    return " ".join(x for x in structured if x)


def infer_failure_modes(node: dict[str, Any]) -> list[str]:
    text = " ".join(clean_text(node.get(k, "")) for k in ("title", "principle", "condition")).lower()
    modes = [label for label, needles in FAILURE_RULES if any(n in text for n in needles)]
    if not modes:
        if node.get("category") == "general":
            modes = ["general execution failure"]
        else:
            modes = [f"{node.get('category', 'task')} method failure"]
    return list(dict.fromkeys(modes))[:3]


def _normalize_directions(dense: np.ndarray, nodes: list[dict[str, Any]], dims: int) -> np.ndarray:
    if dense.shape[1] < dims:
        dense = np.pad(dense, ((0, 0), (0, dims - dense.shape[1])), constant_values=0.0)
    directions = normalize(dense, norm="l2")
    for i, row in enumerate(directions):
        if not np.isfinite(row).all() or np.linalg.norm(row) < 1e-8:
            seed = int(hashlib.md5(nodes[i]["id"].encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            fallback = rng.normal(size=dims)
            directions[i] = fallback / np.linalg.norm(fallback)
    return directions.astype(np.float32)


def _build_tfidf_directions(
    *,
    nodes: list[dict[str, Any]],
    corpus: list[str],
    dims: int,
    fallback_reason: str = "",
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4096)
    tfidf = vectorizer.fit_transform(corpus)
    n_components = min(dims, max(1, min(tfidf.shape) - 1))
    svd = None
    if n_components >= 1 and tfidf.shape[1] > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = svd.fit_transform(tfidf)
        explained = [float(x) for x in getattr(svd, "explained_variance_ratio_", [])]
    else:
        dense = tfidf.toarray()[:, :1]
        explained = []
    directions = _normalize_directions(dense, nodes, dims)
    meta = {
        "method": "tfidf_truncated_svd",
        "dims": dims,
        "actual_components": int(n_components),
        "n_terms": int(tfidf.shape[1]),
        "explained_variance_ratio": explained,
        "embedding_quality_confidence": "low" if fallback_reason else "medium",
    }
    if fallback_reason:
        meta["fallback_reason"] = fallback_reason
    text_model = {
        "version": "hyper_text_model_v2",
        "method": "tfidf_truncated_svd",
        "fields": ["title", "action/principle", "condition/applies_when", "prevents/failure_modes", "category", "scope", "implementation/reference ids"],
        "dims": dims,
        "vectorizer": vectorizer,
        "svd": svd,
        "fallback_reason": fallback_reason,
    }
    return directions, meta, text_model


def _build_sentence_embedding_directions(
    *,
    nodes: list[dict[str, Any]],
    corpus: list[str],
    dims: int,
    embedding_model: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(embedding_model, local_files_only=True)
        embeddings = np.asarray(
            encoder.encode(corpus, normalize_embeddings=False, show_progress_bar=False),
            dtype=np.float32,
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(nodes) or embeddings.shape[1] < 2:
            raise RuntimeError(f"unexpected sentence embedding shape: {embeddings.shape}")
    except Exception as exc:
        raise RuntimeError(f"sentence embedding backend unavailable locally: {exc}") from exc

    n_components = min(dims, max(1, min(embeddings.shape) - 1))
    projection = TruncatedSVD(n_components=n_components, random_state=42)
    dense = projection.fit_transform(embeddings)
    explained = [float(x) for x in getattr(projection, "explained_variance_ratio_", [])]
    directions = _normalize_directions(dense, nodes, dims)
    meta = {
        "method": "sentence_embedding_svd",
        "embedding_model": embedding_model,
        "dims": dims,
        "actual_components": int(n_components),
        "source_embedding_dims": int(embeddings.shape[1]),
        "explained_variance_ratio": explained,
        "embedding_quality_confidence": "high",
    }
    text_model = {
        "version": "hyper_text_model_v2",
        "method": "sentence_embedding_svd",
        "fields": ["title", "action/principle", "condition/applies_when", "prevents/failure_modes", "category", "scope", "implementation/reference ids"],
        "dims": dims,
        "embedding_model": embedding_model,
        "projection": projection,
    }
    return directions, meta, text_model


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _benchmark_query_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("context", "")),
            "Stage: " + str(row.get("stage", "")),
            "Query kind: " + str(row.get("query_kind", "")),
            "Query specificity: " + str(row.get("query_specificity", "")),
            "Conditions: " + "; ".join(row.get("condition") or []),
            "Failure modes: " + "; ".join(row.get("failure_mode") or []),
            "Task type: " + str(row.get("task_type", "")),
        ]
    ).strip()


def _build_contrastive_projection_directions(
    *,
    nodes: list[dict[str, Any]],
    corpus: list[str],
    dims: int,
    embedding_model: str,
    benchmark_path: Path | None,
    gold_path: Path | None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if benchmark_path is None or gold_path is None:
        raise RuntimeError("contrastive_projection requires --contrastive-benchmark and --contrastive-gold")
    try:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(embedding_model, local_files_only=True)
        sop_embeddings = np.asarray(
            encoder.encode(corpus, normalize_embeddings=False, show_progress_bar=False),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(f"embedding_backend_unavailable: sentence embedding backend unavailable locally: {exc}") from exc
    if sop_embeddings.ndim != 2 or sop_embeddings.shape[0] != len(nodes) or sop_embeddings.shape[1] < 2:
        raise RuntimeError(f"unexpected sentence embedding shape: {sop_embeddings.shape}")

    benchmark = _read_jsonl(benchmark_path)
    gold_by_query = {str(row["query_id"]): row for row in _read_jsonl(gold_path)}
    node_index = {str(node["id"]): idx for idx, node in enumerate(nodes)}
    query_texts: list[str] = []
    gold_indices: list[int] = []
    hard_negative_indices: list[int] = []
    for row in benchmark:
        if str(row.get("split", "")) != "dev":
            continue
        qid = str(row.get("query_id", ""))
        gold = gold_by_query.get(qid, {})
        gold_sid = ""
        for item in gold.get("gold_sops", []) or []:
            if item.get("relevance") in {"required", "helpful", "risk_warning"}:
                gold_sid = str(item.get("sop_id", ""))
                break
        if gold_sid not in node_index:
            continue
        query_texts.append(_benchmark_query_text(row))
        gold_indices.append(node_index[gold_sid])
        for did in row.get("distractor_sops", []) or []:
            if str(did) in node_index:
                hard_negative_indices.append(node_index[str(did)])
    if not query_texts:
        raise RuntimeError("contrastive_projection requires at least one dev query-gold pair")
    query_embeddings = np.asarray(
        encoder.encode(query_texts, normalize_embeddings=False, show_progress_bar=False),
        dtype=np.float32,
    )
    gold_embeddings = sop_embeddings[np.asarray(gold_indices, dtype=int)]
    supervised_parts = [
        sop_embeddings,
        query_embeddings,
        0.5 * (query_embeddings + gold_embeddings),
        gold_embeddings - query_embeddings,
    ]
    if hard_negative_indices:
        negative_embeddings = sop_embeddings[np.asarray(hard_negative_indices[: len(query_embeddings)], dtype=int)]
        repeated_queries = query_embeddings[: len(negative_embeddings)]
        supervised_parts.append(repeated_queries - negative_embeddings)
    projection_input = np.vstack(supervised_parts)
    n_components = min(dims, max(1, min(projection_input.shape) - 1))
    projection = TruncatedSVD(n_components=n_components, random_state=42)
    projection.fit(projection_input)
    dense = projection.transform(sop_embeddings)
    directions = _normalize_directions(dense, nodes, dims)
    explained = [float(x) for x in getattr(projection, "explained_variance_ratio_", [])]
    meta = {
        "method": "contrastive_projection",
        "embedding_model": embedding_model,
        "projection_method": "sentence_embedding_supervised_svd_with_dev_query_gold_pairs",
        "dims": dims,
        "actual_components": int(n_components),
        "source_embedding_dims": int(sop_embeddings.shape[1]),
        "dev_pairs": len(query_texts),
        "hard_negative_vectors": len(hard_negative_indices),
        "explained_variance_ratio": explained,
        "embedding_quality_confidence": "high",
    }
    text_model = {
        "version": "hyper_text_model_v3",
        "method": "contrastive_projection",
        "fields": ["sentence embeddings over structured SOP text, supervised by dev query-gold pairs"],
        "dims": dims,
        "embedding_model": embedding_model,
        "projection": projection,
        "benchmark": str(benchmark_path),
        "gold": str(gold_path),
        "train_split": "dev",
    }
    return directions, meta, text_model


def build_text_directions(
    nodes: list[dict[str, Any]],
    dims: int = 16,
    backend: str = "sentence_embedding",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    allow_embedding_fallback: bool = True,
    contrastive_benchmark_path: Path | None = None,
    contrastive_gold_path: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    corpus = [node_text_for_direction(n) for n in nodes]
    backend = (backend or "sentence_embedding").lower()
    if backend in {"sentence_embedding", "sentence", "bge"}:
        try:
            return _build_sentence_embedding_directions(
                nodes=nodes,
                corpus=corpus,
                dims=dims,
                embedding_model=embedding_model,
            )
        except Exception as exc:
            if not allow_embedding_fallback:
                raise RuntimeError(f"embedding_backend_unavailable: {exc}") from exc
            return _build_tfidf_directions(
                nodes=nodes,
                corpus=corpus,
                dims=dims,
                fallback_reason=str(exc),
            )
    if backend in {"contrastive_projection", "contrastive"}:
        return _build_contrastive_projection_directions(
            nodes=nodes,
            corpus=corpus,
            dims=dims,
            embedding_model=embedding_model,
            benchmark_path=contrastive_benchmark_path,
            gold_path=contrastive_gold_path,
        )
    if backend not in {"tfidf", "tfidf_svd", "tfidf_truncated_svd"}:
        raise ValueError(f"Unsupported direction backend: {backend}")
    return _build_tfidf_directions(nodes=nodes, corpus=corpus, dims=dims)


def coordinate_quality_report(
    *,
    nodes: list[dict[str, Any]],
    directions: np.ndarray,
    theta: np.ndarray,
    n_bins: int = 16,
    rng_seed: int = 42,
) -> dict[str, Any]:
    centered = directions - directions.mean(axis=0, keepdims=True)
    try:
        singular = np.linalg.svd(centered, compute_uv=False)
    except Exception:
        singular = np.zeros((0,), dtype=np.float32)
    if singular.size and float(np.sum(singular ** 2)) > 0:
        probs = (singular ** 2) / np.sum(singular ** 2)
        effective_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    else:
        effective_rank = 0.0

    bins = np.floor((theta % (2.0 * math.pi)) / (2.0 * math.pi) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    bin_counts = collections.Counter(int(b) for b in bins)
    top2_mass = sum(c for _, c in bin_counts.most_common(2)) / max(1, len(nodes))

    labels = []
    for node in nodes:
        category = clean_text(node.get("category"))
        failures = set(infer_failure_modes(node))
        labels.append((category, failures))

    sims = directions @ directions.T
    np.fill_diagonal(sims, -np.inf)
    k = min(10, max(1, len(nodes) - 1))
    nn_hits = []
    for i, (_category, failures) in enumerate(labels):
        nn = np.argsort(-sims[i])[:k]
        hits = 0
        for j in nn:
            same_category = labels[j][0] == labels[i][0] and bool(labels[i][0])
            shared_failure = bool(failures & labels[j][1])
            hits += int(same_category or shared_failure)
        nn_hits.append(hits / k)
    nn_coherence = float(np.mean(nn_hits)) if nn_hits else 0.0

    rng = np.random.default_rng(rng_seed)
    random_hits = []
    all_idx = np.arange(len(nodes))
    for i, (_category, failures) in enumerate(labels):
        pool = np.delete(all_idx, i)
        if pool.size == 0:
            continue
        sample = rng.choice(pool, size=min(k, pool.size), replace=False)
        hits = 0
        for j in sample:
            same_category = labels[j][0] == labels[i][0] and bool(labels[i][0])
            shared_failure = bool(failures & labels[j][1])
            hits += int(same_category or shared_failure)
        random_hits.append(hits / max(1, len(sample)))
    random_coherence = float(np.mean(random_hits)) if random_hits else 0.0
    coherence_lift = nn_coherence - random_coherence

    thresholds = {
        "effective_rank_min": 6.0,
        "top2_bin_mass_max": 0.55,
        "neighbor_coherence_lift_min": 0.10,
    }
    passed = (
        effective_rank >= thresholds["effective_rank_min"]
        and top2_mass <= thresholds["top2_bin_mass_max"]
        and coherence_lift >= thresholds["neighbor_coherence_lift_min"]
    )
    return {
        "status": "passed" if passed else "coordinate_quality_null",
        "passed": passed,
        "thresholds": thresholds,
        "direction_effective_rank": effective_rank,
        "singular_values": [float(x) for x in singular[: min(16, len(singular))]],
        "theta_bin_count": n_bins,
        "theta_bin_counts": {str(k): int(v) for k, v in sorted(bin_counts.items())},
        "theta_top2_bin_mass": top2_mass,
        "neighbor_top_k": k,
        "neighbor_category_or_failure_coherence": nn_coherence,
        "random_category_or_failure_coherence": random_coherence,
        "neighbor_coherence_lift": coherence_lift,
        "query_aware_gate": {
            "status": "not_evaluated",
            "required_inputs": ["benchmark dev queries", "gold SOP labels", "runner output"],
            "note": "Query-aware checks are computed by the offline ablation evaluator because they need query/gold pairs.",
        },
        "note": "Geometry ablation is not claim-grade unless this report passes.",
    }


SPECIFIC_IMPLEMENTATION_TERMS = {
    "api", "checkpoint", "shape", "dimension", "dtype", "cuda", "device", "path",
    "file", "column", "csv", "undefined", "duplicate", "base_estimator",
    "estimator", "argument", "parameter", "attribute", "import", "version",
    "oom", "nan", "exception", "error", "traceback", "sample_submission",
}

GENERIC_FAILURE_LABELS = {
    "general execution failure",
}


def _wilson_lower_bound(successes: float, total: float, fallback: float) -> float:
    if total <= 0:
        return clamp(fallback, 0.0, 1.0)
    p = clamp(successes / total, 0.0, 1.0)
    z = 1.96
    z2 = z * z
    denom = 1.0 + z2 / total
    center = p + z2 / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return clamp((center - margin) / denom, 0.0, 1.0)


def _specificity_features(node: dict[str, Any], *, max_level: float, max_category_count: float, category_counts: collections.Counter[str]) -> dict[str, float]:
    title = clean_text(node.get("title"))
    principle = clean_text(node.get("principle") or node.get("action"))
    condition = clean_text(node.get("condition"))
    text = " ".join([title, principle, condition]).lower()
    cond_tokens = tokenize(condition)
    all_tokens = tokenize(text)
    failure_modes = infer_failure_modes(node)

    condition_specificity = clamp((len(cond_tokens) - 3) / 18.0, 0.0, 1.0)
    if any(tok in cond_tokens for tok in {"missing", "cuda", "shape", "path", "column", "checkpoint", "undefined"}):
        condition_specificity = clamp(condition_specificity + 0.20, 0.0, 1.0)

    non_generic_failures = [
        f for f in failure_modes
        if f not in GENERIC_FAILURE_LABELS and not f.endswith(" method failure")
    ]
    failure_specificity = clamp(0.25 * len(non_generic_failures), 0.0, 0.75)
    if any(any(needle in text for needle in needles) for label, needles in FAILURE_RULES if label in non_generic_failures):
        failure_specificity = clamp(failure_specificity + 0.25, 0.0, 1.0)

    implementation_hits = len(all_tokens & SPECIFIC_IMPLEMENTATION_TERMS)
    code_like_hits = len(re.findall(r"\b[a-zA-Z_]+(?:_[a-zA-Z0-9_]+)+\b|\b[A-Za-z]+Error\b|[/\\.][A-Za-z0-9_/-]+", text))
    implementation_specificity = clamp((implementation_hits + 0.5 * code_like_hits) / 4.0, 0.0, 1.0)

    category = str(node.get("category") or "unknown")
    scope = str(node.get("scope") or "")
    if category == "general" or scope == "universal_general":
        task_coverage_norm = 1.0
    else:
        task_coverage_norm = clamp(float(category_counts.get(category, 1)) / max_category_count, 0.0, 1.0)

    level = float(node.get("level", 0) or 0.0)
    graph_centrality_norm = clamp(1.0 - (level / max_level if max_level > 0 else 0.0), 0.0, 1.0)

    specificity_score = clamp(
        0.30 * condition_specificity
        + 0.25 * failure_specificity
        + 0.20 * implementation_specificity
        + 0.15 * (1.0 - task_coverage_norm)
        + 0.10 * (1.0 - graph_centrality_norm),
        0.0,
        1.0,
    )
    return {
        "condition_specificity": condition_specificity,
        "failure_specificity": failure_specificity,
        "implementation_specificity": implementation_specificity,
        "task_coverage_norm": task_coverage_norm,
        "graph_centrality_norm": graph_centrality_norm,
        "specificity_score": specificity_score,
    }


def compute_radii(nodes: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    max_use = max(float(n.get("n_use", 0) or 0) for n in nodes) or 1.0
    max_level = max(float(n.get("level", 0) or 0) for n in nodes) or 1.0
    category_counts = collections.Counter(str(n.get("category") or "unknown") for n in nodes)
    non_general_counts = [v for k, v in category_counts.items() if k != "general"]
    max_category_count = float(max(non_general_counts or list(category_counts.values()) or [1]))
    radii = []
    features: list[dict[str, Any]] = []
    for n in nodes:
        specificity = _specificity_features(
            n,
            max_level=max_level,
            max_category_count=max_category_count,
            category_counts=category_counts,
        )
        radius = clamp(0.12 + 0.78 * specificity["specificity_score"], 0.12, 0.90)
        n_use = float(n.get("n_use", 0) or 0.0)
        n_succ = float(n.get("n_succ", 0) or 0.0)
        p_hat = clamp(float(n.get("p_hat", 0.0) or 0.0), 0.0, 1.0)
        support_score = math.log1p(n_use) / math.log1p(max_use)
        source_certification = 1.0 if (n.get("source_branches") or n.get("evidence_turns")) else 0.0
        reliability_score = clamp(
            0.65 * _wilson_lower_bound(n_succ, n_use, p_hat)
            + 0.25 * support_score
            + 0.10 * source_certification,
            0.0,
            1.0,
        )
        evidence_confidence = "high" if reliability_score >= 0.67 else ("medium" if reliability_score >= 0.34 else "low")
        features.append({
            **specificity,
            "support_score": support_score,
            "source_certification": source_certification,
            "reliability_score": reliability_score,
            "evidence_confidence": evidence_confidence,
            "wilson_or_p_hat": _wilson_lower_bound(n_succ, n_use, p_hat),
        })
        radii.append(radius)
    meta = {
        "model": "specificity_radius_v2",
        "formula": "radius=clamp(0.12+0.78*(0.30*condition_specificity+0.25*failure_specificity+0.20*implementation_specificity+0.15*(1-task_coverage_norm)+0.10*(1-graph_centrality_norm)))",
        "radius_reliability_decoupled": True,
        "reliability_formula": "reliability_score=0.65*wilson_or_p_hat+0.25*support_score+0.10*source_certification",
        "max_n_use": max_use,
        "max_level": max_level,
        "max_category_count": max_category_count,
        "note": "p_hat/n_use/support_score/reliability_score are not used to place SOPs on the radius; they are stored as scorer features only.",
    }
    return np.asarray(radii, dtype=np.float32), meta, features


@dataclass
class GraphPatch:
    op: str
    kind: str
    src: str | None = None
    dst: str | None = None
    node: dict[str, Any] | None = None
    weight: float = 1.0
    reason: str = ""
    evidence: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)


class GraphBuilderAgent:
    """One-pass patch builder over SOP-like compact cards.

    The class is intentionally named as an agent: it inspects SOP cards, proposes
    graph patches, and never mutates the graph directly. Programmatic validation
    decides which patches are allowed to land.
    """

    def __init__(self, nodes: list[dict[str, Any]], base_edges: list[dict[str, Any]]) -> None:
        self.nodes = nodes
        self.node_by_id = {n["id"]: n for n in nodes}
        self.base_edges = base_edges
        self.condition_for_sop: dict[str, str] = {}
        self.failures_for_sop: dict[str, list[str]] = {}
        self.failure_label_by_id: dict[str, str] = {}

    def propose(self) -> list[GraphPatch]:
        patches: list[GraphPatch] = []
        patches.extend(self._propose_skill_containment())
        patches.extend(self._propose_condition_failure_evidence())
        patches.extend(self._propose_sop_relations())
        return patches

    def _propose_skill_containment(self) -> list[GraphPatch]:
        patches = []
        seen_skills = set()
        for node in self.nodes:
            region = node.get("category") or "unknown"
            scope = node.get("scope") or ""
            skill_key = f"{region}:{scope}" if scope and region == "general" else region
            skill_id = f"skill_{slugify(skill_key, 72)}"
            if skill_id not in seen_skills:
                seen_skills.add(skill_id)
                patches.append(GraphPatch(
                    op="add_node",
                    kind="Skill",
                    node={
                        "id": skill_id,
                        "type": "Skill",
                        "title": skill_key,
                        "category": region,
                        "scope": scope,
                        "description": f"Skill region for {skill_key}",
                    },
                    reason="Skill node anchors SOP-like compact cards into map regions.",
                    evidence=skill_key,
                ))
            patches.append(GraphPatch(
                op="add_edge",
                kind="contains",
                src=skill_id,
                dst=node["id"],
                weight=1.0,
                reason="Every SOP-like card belongs to a Skill region.",
                evidence=f"{node['id']} category={region} scope={scope}",
            ))
        return patches

    def _condition_phrase(self, node: dict[str, Any]) -> str:
        cond = clean_text(node.get("condition", ""))
        if not cond:
            return f"general {node.get('category', 'task')} context"
        return cond[:220]

    def _failure_modes(self, node: dict[str, Any]) -> list[str]:
        return infer_failure_modes(node)

    def _propose_condition_failure_evidence(self) -> list[GraphPatch]:
        patches = []
        for node in self.nodes:
            sid = node["id"]
            cond = self._condition_phrase(node)
            cond_id = stable_id("cond", cond.lower())
            self.condition_for_sop[sid] = cond_id
            patches.append(GraphPatch(
                op="add_node",
                kind="Condition",
                node={"id": cond_id, "type": "Condition", "title": cond, "text": cond},
                reason="Condition extracted from compact-card condition field.",
                evidence=cond,
            ))
            patches.append(GraphPatch(
                op="add_edge",
                kind="applies_when",
                src=sid,
                dst=cond_id,
                weight=0.85,
                reason="SOP applies under its distilled condition.",
                evidence=cond,
            ))

            failure_modes = self._failure_modes(node)
            fail_ids = []
            for failure in failure_modes:
                fail_id = stable_id("fail", failure.lower())
                self.failure_label_by_id[fail_id] = failure
                fail_ids.append(fail_id)
                patches.append(GraphPatch(
                    op="add_node",
                    kind="FailureMode",
                    node={"id": fail_id, "type": "FailureMode", "title": failure, "text": failure},
                    reason="FailureMode inferred from SOP title/principle/condition.",
                    evidence=" | ".join([clean_text(node.get("title")), clean_text(node.get("principle")), clean_text(node.get("condition"))])[:500],
                ))
                patches.append(GraphPatch(
                    op="add_edge",
                    kind="prevents",
                    src=sid,
                    dst=fail_id,
                    weight=0.70,
                    reason=f"SOP is relevant to preventing or mitigating {failure}.",
                    evidence=clean_text(node.get("principle") or node.get("title")),
                ))
            self.failures_for_sop[sid] = fail_ids

            ev_id = f"ev_{sid}"
            patches.append(GraphPatch(
                op="add_node",
                kind="Evidence",
                node={
                    "id": ev_id,
                    "type": "Evidence",
                    "title": f"SkillGraph-C evidence for {sid}",
                    "source": "graph_skillgraph_c_trace_prereq",
                    "n_use": node.get("n_use", 0),
                    "n_succ": node.get("n_succ", 0),
                    "p_hat": node.get("p_hat", 0.0),
                    "level": node.get("level", 0),
                },
                reason="Trace-proxy stats are attached as explicit evidence nodes.",
                evidence=f"n_use={node.get('n_use',0)} n_succ={node.get('n_succ',0)} p_hat={node.get('p_hat',0)}",
            ))
            patches.append(GraphPatch(
                op="add_edge",
                kind="supported_by",
                src=sid,
                dst=ev_id,
                weight=0.75,
                reason="SOP is supported by SkillGraph-C trace-proxy evidence.",
                evidence=f"n_use={node.get('n_use',0)} n_succ={node.get('n_succ',0)} p_hat={node.get('p_hat',0)}",
            ))
        return patches

    def _opposition_terms(self, a: dict[str, Any], b: dict[str, Any]) -> list[str]:
        a_tokens = tokenize(" ".join([clean_text(a.get("title")), clean_text(a.get("principle"))]))
        b_tokens = tokenize(" ".join([clean_text(b.get("title")), clean_text(b.get("principle"))]))
        terms = []
        for x, y in OPPOSING_TERMS:
            if (x in a_tokens and y in b_tokens) or (y in a_tokens and x in b_tokens):
                terms.append(f"{x}/{y}")
        return terms

    def _non_generic_failures(self, failure_ids: set[str]) -> set[str]:
        out = set()
        for fid in failure_ids:
            label = self.failure_label_by_id.get(fid, "")
            if label == "general execution failure":
                continue
            if label.endswith(" method failure"):
                continue
            out.add(fid)
        return out

    def _token_jaccard(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        at = tokenize(" ".join([clean_text(a.get("title")), clean_text(a.get("principle"))]))
        bt = tokenize(" ".join([clean_text(b.get("title")), clean_text(b.get("principle"))]))
        return len(at & bt) / len(at | bt) if (at | bt) else 0.0

    def _propose_sop_relations(self) -> list[GraphPatch]:
        patches = []
        by_category = collections.defaultdict(list)
        for node in self.nodes:
            by_category[node.get("category", "unknown")].append(node)

        refine_out_count: collections.Counter[str] = collections.Counter()
        for category, group in by_category.items():
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    a_fail = set(self.failures_for_sop.get(a["id"], []))
                    b_fail = set(self.failures_for_sop.get(b["id"], []))
                    shared_failures = a_fail & b_fail
                    strong_shared_failures = self._non_generic_failures(shared_failures)
                    same_failure = bool(strong_shared_failures)
                    same_condition = self.condition_for_sop.get(a["id"]) == self.condition_for_sop.get(b["id"])
                    if not (same_failure or same_condition):
                        continue
                    opposition_terms = self._opposition_terms(a, b)
                    if opposition_terms and (same_failure or same_condition):
                        patches.append(GraphPatch(
                            op="add_edge",
                            kind="conflicts_with",
                            src=a["id"],
                            dst=b["id"],
                            weight=0.60,
                            reason="GraphBuilderAgent found same condition/failure context with opposing action words.",
                            evidence=f"{a.get('title')} || {b.get('title')}",
                            attrs={
                                "undirected": True,
                                "category": category,
                                "same_condition": same_condition,
                                "shared_failure": sorted(strong_shared_failures),
                                "opposition_terms": opposition_terms,
                            },
                        ))
                    elif same_failure and not same_condition:
                        # Keep this sparse: only connect nearby evidence levels to avoid a dense clique.
                        if (
                            abs(int(a.get("level", 0) or 0) - int(b.get("level", 0) or 0)) <= 2
                            and self._token_jaccard(a, b) >= 0.12
                        ):
                            src, dst = (a, b) if int(a.get("level", 0) or 0) <= int(b.get("level", 0) or 0) else (b, a)
                            if refine_out_count[src["id"]] >= 3:
                                continue
                            refine_out_count[src["id"]] += 1
                            patches.append(GraphPatch(
                                op="add_edge",
                                kind="refines",
                                src=src["id"],
                                dst=dst["id"],
                                weight=0.45,
                                reason="SOPs share a FailureMode but differ in condition, suggesting a condition-specific refinement.",
                                evidence=f"{src.get('title')} -> {dst.get('title')}",
                                attrs={"shared_failure": sorted(a_fail & b_fail)},
                            ))
        return patches


def build_sop_node(
    node: dict[str, Any],
    radius: float,
    direction: np.ndarray,
    theta: float,
    phi: float,
    poincare: np.ndarray,
    lorentz: np.ndarray,
    geometry_features: dict[str, Any],
) -> dict[str, Any]:
    evidence_confidence = str(geometry_features.get("evidence_confidence") or "low")
    return {
        "id": node["id"],
        "type": "SOP",
        "title": clean_text(node.get("title")),
        "action": clean_text(node.get("principle")),
        "principle": clean_text(node.get("principle")),
        "applies_when": [clean_text(node.get("condition"))] if clean_text(node.get("condition")) else [],
        "condition": clean_text(node.get("condition")),
        "category": node.get("category"),
        "scope": node.get("scope", ""),
        "skill_id": f"skill_{slugify((node.get('category') or 'unknown') + ((':' + node.get('scope')) if node.get('scope') and node.get('category') == 'general' else ''), 72)}",
        "source_node_id": node["id"],
        "source_branches": node.get("source_branches", []),
        "evidence_turns": node.get("evidence_turns", []),
        "reference_ids": node.get("reference_ids", []),
        "evidence_ids": node.get("evidence_ids", []),
        "metric": {
            "p_hat": node.get("p_hat", 0.0),
            "n_use": node.get("n_use", 0),
            "n_succ": node.get("n_succ", 0),
            "level": node.get("level", 0),
            "signal": "skillgraph_trace_proxy",
        },
        "confidence": evidence_confidence,
        "evidence_confidence": evidence_confidence,
        "radius_model": "specificity_radius_v2",
        "specificity_score": round(float(geometry_features.get("specificity_score", 0.0)), 6),
        "condition_specificity": round(float(geometry_features.get("condition_specificity", 0.0)), 6),
        "failure_specificity": round(float(geometry_features.get("failure_specificity", 0.0)), 6),
        "implementation_specificity": round(float(geometry_features.get("implementation_specificity", 0.0)), 6),
        "task_coverage_norm": round(float(geometry_features.get("task_coverage_norm", 0.0)), 6),
        "graph_centrality_norm": round(float(geometry_features.get("graph_centrality_norm", 0.0)), 6),
        "support_score": round(float(geometry_features.get("support_score", 0.0)), 6),
        "source_certification": round(float(geometry_features.get("source_certification", 0.0)), 6),
        "reliability_score": round(float(geometry_features.get("reliability_score", 0.0)), 6),
        "radius": round(float(radius), 6),
        "radius_band": radius_band(float(radius)),
        "angle_theta": round(float(theta), 6),
        "angle_phi": round(float(phi), 6),
        "angle_direction": [round(float(x), 6) for x in direction.tolist()],
        "poincare": [round(float(x), 6) for x in poincare.tolist()],
        "lorentz": [round(float(x), 6) for x in lorentz.tolist()],
    }


def patch_to_edge(patch: GraphPatch) -> dict[str, Any]:
    edge = {
        "src": patch.src,
        "dst": patch.dst,
        "kind": patch.kind,
        "weight": round(float(patch.weight), 6),
        "provenance": "graph_builder_agent",
        "reason": patch.reason,
        "evidence": patch.evidence,
    }
    edge.update(patch.attrs)
    return edge


def validate_and_apply_patches(
    *,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    patches: list[GraphPatch],
) -> dict[str, Any]:
    applied = []
    rejected = []
    skipped = []
    seen_edges = {(e.get("src"), e.get("dst"), e.get("kind")) for e in edges}

    def reject(patch: GraphPatch, reason: str) -> None:
        rejected.append({"op": patch.op, "kind": patch.kind, "src": patch.src, "dst": patch.dst, "reason": reason})

    def skip(patch: GraphPatch, reason: str) -> None:
        skipped.append({"op": patch.op, "kind": patch.kind, "src": patch.src, "dst": patch.dst, "reason": reason})

    for patch in patches:
        if patch.kind not in ALLOWED_EDGE_KINDS and patch.op == "add_edge":
            reject(patch, "illegal edge kind")
            continue
        if not patch.evidence:
            reject(patch, "missing text evidence")
            continue
        if patch.op == "add_node":
            if not patch.node or not patch.node.get("id") or not patch.node.get("type"):
                reject(patch, "invalid node patch")
                continue
            nid = patch.node["id"]
            if nid not in nodes:
                nodes[nid] = patch.node
                applied.append({"op": "add_node", "kind": patch.kind, "id": nid})
            else:
                skip(patch, "duplicate node id")
            continue
        if patch.op != "add_edge":
            reject(patch, "unknown op")
            continue
        if patch.src not in nodes or patch.dst not in nodes:
            reject(patch, "src or dst node does not exist")
            continue
        if patch.kind == "conflicts_with":
            if not patch.reason or not patch.evidence or not (patch.attrs.get("opposition_terms") or "opposing" in patch.reason.lower()):
                reject(patch, "conflicts_with requires explicit opposition evidence")
                continue
            if not (patch.attrs.get("same_condition") or patch.attrs.get("shared_failure")):
                reject(patch, "conflicts_with requires same condition or shared failure context")
                continue
        key = (patch.src, patch.dst, patch.kind)
        if key in seen_edges:
            skip(patch, "duplicate edge")
            continue
        seen_edges.add(key)
        edge = patch_to_edge(patch)
        edges.append(edge)
        applied.append({"op": "add_edge", "kind": patch.kind, "src": patch.src, "dst": patch.dst})

    return {
        "applied": applied,
        "rejected": rejected,
        "skipped": skipped,
        "applied_by_kind": dict(collections.Counter(x["kind"] for x in applied)),
        "rejected_by_reason": dict(collections.Counter(x["reason"] for x in rejected)),
        "skipped_by_reason": dict(collections.Counter(x["reason"] for x in skipped)),
    }


def convert_base_edges(base_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for edge in base_edges:
        kind = edge.get("kind") or edge.get("type")
        if kind not in {"co_occur", "enhance", "prereq"}:
            continue
        converted.append({
            "src": edge.get("src"),
            "dst": edge.get("dst"),
            "kind": kind,
            "weight": float(edge.get("weight", 1.0)),
            "provenance": "skillgraph_c_base",
        })
    return converted


def provenance_report(graph: dict[str, Any], source_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    meta = graph.get("meta", {}) or {}
    required_meta = ["source_runs", "allowlist", "leak_verified"]
    missing_meta = [k for k in required_meta if k not in meta]
    nodes_with_source = [
        n["id"]
        for n in source_nodes
        if n.get("source_branches") or n.get("evidence_turns")
    ]
    clean = not missing_meta and len(nodes_with_source) == len(source_nodes) and bool(meta.get("leak_verified"))
    return {
        "status": "clean_certified" if clean else "uncertified_bootstrap",
        "paper_grade": clean,
        "missing_meta_fields": missing_meta,
        "nodes_with_source_evidence": len(nodes_with_source),
        "nodes_total": len(source_nodes),
        "source_runs": meta.get("source_runs", []),
        "allowlist": meta.get("allowlist", []),
        "leak_verified": meta.get("leak_verified", False),
        "note": "Current artifact is paper-grade only when every SOP has source evidence and the graph meta records clean-run provenance.",
    }


def build(
    input_path: Path,
    output_dir: Path,
    dims: int = 16,
    require_clean_provenance: bool = False,
    direction_backend: str = "sentence_embedding",
    embedding_model: str = "BAAI/bge-base-en-v1.5",
    allow_embedding_fallback: bool = True,
    contrastive_benchmark_path: Path | None = None,
    contrastive_gold_path: Path | None = None,
) -> dict[str, Any]:
    graph = json.loads(input_path.read_text(encoding="utf-8"))
    source_nodes = graph.get("nodes", [])
    source_edges = graph.get("edges", [])
    if not source_nodes:
        raise ValueError(f"No nodes found in {input_path}")

    provenance = provenance_report(graph, source_nodes)
    if require_clean_provenance and not provenance["paper_grade"]:
        raise ValueError(f"Input graph lacks clean provenance: {provenance}")

    directions, direction_meta, text_model = build_text_directions(
        source_nodes,
        dims=dims,
        backend=direction_backend,
        embedding_model=embedding_model,
        allow_embedding_fallback=allow_embedding_fallback,
        contrastive_benchmark_path=contrastive_benchmark_path,
        contrastive_gold_path=contrastive_gold_path,
    )
    radii, radius_meta, geometry_features = compute_radii(source_nodes)
    theta, phi = direction_to_angles(directions)
    quality = coordinate_quality_report(nodes=source_nodes, directions=directions, theta=theta)
    poincare = directions * radii[:, None]
    lorentz = poincare_to_lorentz(poincare)
    flat_twin = poincare.copy()
    euclidean = directions.copy()

    hyper_nodes: dict[str, dict[str, Any]] = {}
    node_ids = []
    for i, node in enumerate(source_nodes):
        sop = build_sop_node(node, radii[i], directions[i], theta[i], phi[i], poincare[i], lorentz[i], geometry_features[i])
        hyper_nodes[sop["id"]] = sop
        node_ids.append(sop["id"])

    hyper_edges = convert_base_edges(source_edges)
    agent = GraphBuilderAgent(source_nodes, source_edges)
    patches = agent.propose()
    patch_result = validate_and_apply_patches(nodes=hyper_nodes, edges=hyper_edges, patches=patches)

    node_type_counts = dict(collections.Counter(n.get("type", "unknown") for n in hyper_nodes.values()))
    edge_kind_counts = dict(collections.Counter(e.get("kind", "unknown") for e in hyper_edges))
    hyper_graph = {
        "meta": {
            "schema": "hyperbolic-sop-memory-v2",
            "source_graph": display_path(input_path),
            "source_schema": graph.get("meta", {}).get("schema"),
            "source_runs": provenance["source_runs"],
            "allowlist": provenance["allowlist"],
            "allowlist_hash": graph.get("meta", {}).get("allowlist_hash"),
            "leak_verified": provenance["leak_verified"],
            "provenance_status": provenance["status"],
            "paper_grade": provenance["paper_grade"],
            "builder": "GraphBuilderAgent(heuristic_patch_v1)+programmatic_validation",
            "coordinate_model": f"Poincare ball + Lorentz hyperboloid, curvature=1, dims={dims}",
            "angle_model": f"theta/phi spherical angles over {direction_meta.get('method')} direction; structured SOP text fields",
            "radius_model": "specificity_radius_v2",
            "radius_reliability_decoupled": True,
            "flat_twin_model": "same coordinates as poincare; Flat-Twin swaps only the distance function at runtime",
            "euclidean_model": "independent flat direction coordinates with L2-normalized direction; no support-radius scaling",
            "node_count": len(hyper_nodes),
            "edge_count": len(hyper_edges),
            "note": "Bootstrap artifact from SkillGraph-C compact cards; metric is p_hat/n_use proxy, not transition-level metric_delta.",
        },
        "nodes": list(hyper_nodes.values()),
        "edges": hyper_edges,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "hyper_graph.json"
    index_path = output_dir / "hyper_index.npz"
    text_model_path = output_dir / "hyper_text_model.joblib"
    report_path = output_dir / "graph_builder_report.json"
    quality_path = output_dir / "coordinate_quality_report.json"

    graph_path.write_text(json.dumps(hyper_graph, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(text_model, text_model_path)
    np.savez_compressed(
        index_path,
        node_ids=np.asarray(node_ids, dtype=str),
        poincare=poincare.astype(np.float32),
        lorentz=lorentz.astype(np.float32),
        flat_twin=flat_twin.astype(np.float32),
        euclidean=euclidean.astype(np.float32),
        radius=radii.astype(np.float32),
        specificity_score=np.asarray([float(x["specificity_score"]) for x in geometry_features], dtype=np.float32),
        support_score=np.asarray([float(x["support_score"]) for x in geometry_features], dtype=np.float32),
        reliability_score=np.asarray([float(x["reliability_score"]) for x in geometry_features], dtype=np.float32),
        theta=theta.astype(np.float32),
        phi=phi.astype(np.float32),
        angle=np.stack([theta, phi], axis=1).astype(np.float32),
        direction=directions.astype(np.float32),
    )

    radius_values = [float(r) for r in radii]
    source_sop_ids = {n["id"] for n in source_nodes}
    all_node_ids = set(hyper_nodes)
    patch_edges = [e for e in hyper_edges if e.get("provenance") == "graph_builder_agent"]
    conflicts = [e for e in hyper_edges if e.get("kind") == "conflicts_with"]
    edge_src_by_kind = collections.defaultdict(set)
    edge_dst_by_kind = collections.defaultdict(set)
    for edge in hyper_edges:
        edge_src_by_kind[edge.get("kind")].add(edge.get("src"))
        edge_dst_by_kind[edge.get("kind")].add(edge.get("dst"))
    report = {
        "input": display_path(input_path),
        "outputs": {
            "hyper_graph": display_path(graph_path),
            "hyper_index": display_path(index_path),
            "hyper_text_model": display_path(text_model_path),
            "graph_builder_report": display_path(report_path),
            "coordinate_quality_report": display_path(quality_path),
        },
        "provenance": provenance,
        "source": {
            "nodes": len(source_nodes),
            "edges": len(source_edges),
            "edge_kinds": dict(collections.Counter(e.get("kind") or e.get("type") for e in source_edges)),
        },
        "hyper_graph": {
            "nodes": len(hyper_nodes),
            "edges": len(hyper_edges),
            "node_type_counts": node_type_counts,
            "edge_kind_counts": edge_kind_counts,
        },
        "coordinates": {
            "direction": direction_meta,
            "angle": {
                "theta": "azimuth atan2(y, x), normalized to [0, 2*pi)",
                "phi": "polar angle arccos(z / ||direction||)",
            },
            "radius": radius_meta,
            "radius_reliability_decoupled": bool(radius_meta.get("radius_reliability_decoupled")),
            "specificity_score_mean": float(np.mean([float(x["specificity_score"]) for x in geometry_features])),
            "reliability_score_mean": float(np.mean([float(x["reliability_score"]) for x in geometry_features])),
            "radius_min": min(radius_values),
            "radius_max": max(radius_values),
            "radius_mean": sum(radius_values) / len(radius_values),
            "radius_bands": dict(collections.Counter(radius_band(r) for r in radius_values)),
            "poincare_max_norm": float(np.linalg.norm(poincare, axis=1).max()),
            "euclidean_norm_min": float(np.linalg.norm(euclidean, axis=1).min()),
            "euclidean_norm_max": float(np.linalg.norm(euclidean, axis=1).max()),
            "euclidean_model": {
                "coordinates": "L2-normalized TF-IDF-SVD directions in Euclidean R^d",
                "distance": "Euclidean L2 distance",
                "independent_from_poincare": bool(not np.array_equal(euclidean.astype(np.float32), poincare.astype(np.float32))),
                "note": "This is the independent flat-coordinate baseline; Flat-Twin remains the same-coordinate distance-function control.",
            },
            "lorentz_minkowski_error_max": float(np.max(np.abs(-(lorentz[:, 0] ** 2) + np.sum(lorentz[:, 1:] ** 2, axis=1) + 1.0))),
            "flat_twin_identity": bool(np.array_equal(flat_twin.astype(np.float32), poincare.astype(np.float32))),
            "quality_report": quality,
        },
        "graph_builder_agent": {
            "patches_proposed": len(patches),
            "patches_applied": len(patch_result["applied"]),
            "patches_rejected": len(patch_result["rejected"]),
            "patches_skipped": len(patch_result["skipped"]),
            "applied_by_kind": patch_result["applied_by_kind"],
            "rejected_by_reason": patch_result["rejected_by_reason"],
            "skipped_by_reason": patch_result["skipped_by_reason"],
            "sample_rejections": patch_result["rejected"][:10],
        },
        "validation": {
            "source_sop_ids_preserved": all(
                sid in hyper_nodes and hyper_nodes[sid].get("type") == "SOP"
                for sid in source_sop_ids
            ),
            "all_node_ids_unique": len(hyper_nodes) == len(all_node_ids),
            "all_edge_endpoints_exist": all(
                e.get("src") in all_node_ids and e.get("dst") in all_node_ids
                for e in hyper_edges
            ),
            "all_base_edge_endpoints_exist": all(e.get("src") in hyper_nodes and e.get("dst") in hyper_nodes for e in convert_base_edges(source_edges)),
            "all_edge_kinds_allowed": all(e.get("kind") in ALLOWED_EDGE_KINDS for e in hyper_edges),
            "patch_edges_have_reason_or_evidence": all(
                e.get("reason") or e.get("evidence")
                for e in patch_edges
            ),
            "every_sop_has_skill_region": source_sop_ids <= edge_dst_by_kind["contains"],
            "every_sop_has_condition_edge": source_sop_ids <= edge_src_by_kind["applies_when"],
            "every_sop_has_failure_edge": source_sop_ids <= edge_src_by_kind["prevents"],
            "every_sop_has_evidence_edge": source_sop_ids <= edge_src_by_kind["supported_by"],
            "conflicts_have_reason_and_evidence": all(
                e.get("kind") != "conflicts_with" or (e.get("reason") and e.get("evidence"))
                for e in hyper_edges
            ),
            "conflicts_have_opposition_terms": all(
                e.get("opposition_terms")
                for e in conflicts
            ),
            "conflicts_have_condition_or_failure_context": all(
                e.get("kind") != "conflicts_with" or (e.get("same_condition") or e.get("shared_failure"))
                for e in hyper_edges
            ),
            "flat_twin_same_coordinates_as_poincare": bool(np.array_equal(flat_twin.astype(np.float32), poincare.astype(np.float32))),
            "euclidean_independent_coordinates": bool(not np.array_equal(euclidean.astype(np.float32), poincare.astype(np.float32))),
            "euclidean_unit_norm_coordinates": bool(np.allclose(np.linalg.norm(euclidean, axis=1), 1.0, atol=1e-5)),
            "coordinate_quality_gate_passed": quality["passed"],
            "radius_reliability_decoupled": bool(radius_meta.get("radius_reliability_decoupled")),
            "paper_grade_provenance": provenance["paper_grade"],
        },
    }
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bootstrap hyperbolic SOP memory from SkillGraph-C.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dims", type=int, default=16)
    parser.add_argument(
        "--direction-backend",
        choices=["sentence_embedding", "tfidf", "tfidf_svd", "tfidf_truncated_svd", "contrastive_projection"],
        default="sentence_embedding",
    )
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--allow-embedding-fallback", action="store_true")
    parser.add_argument("--contrastive-benchmark", type=Path, default=None)
    parser.add_argument("--contrastive-gold", type=Path, default=None)
    parser.add_argument("--require-clean-provenance", action="store_true")
    args = parser.parse_args()

    try:
        report = build(
            args.input,
            args.output_dir,
            dims=args.dims,
            require_clean_provenance=args.require_clean_provenance,
            direction_backend=args.direction_backend,
            embedding_model=args.embedding_model,
            allow_embedding_fallback=args.allow_embedding_fallback,
            contrastive_benchmark_path=args.contrastive_benchmark,
            contrastive_gold_path=args.contrastive_gold,
        )
    except RuntimeError as exc:
        if "embedding_backend_unavailable" in str(exc):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            unavailable = {
                "status": "embedding_backend_unavailable",
                "not_run": True,
                "direction_backend": args.direction_backend,
                "embedding_model": args.embedding_model,
                "error": str(exc),
                "note": "V3 strict embedding mode does not silently fall back to TF-IDF.",
            }
            path = args.output_dir / "embedding_backend_report.json"
            path.write_text(json.dumps(unavailable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(unavailable, ensure_ascii=False, indent=2))
            raise SystemExit(2) from exc
        raise
    print(f"hyper_graph: {report['outputs']['hyper_graph']}")
    print(f"hyper_index: {report['outputs']['hyper_index']}")
    print(f"hyper_text_model: {report['outputs']['hyper_text_model']}")
    print(f"coordinate_quality_report: {report['outputs']['coordinate_quality_report']}")
    print(f"report: {report['outputs']['graph_builder_report']}")
    print(f"nodes: {report['hyper_graph']['nodes']}  edges: {report['hyper_graph']['edges']}")
    print(f"node types: {report['hyper_graph']['node_type_counts']}")
    print(f"edge kinds: {report['hyper_graph']['edge_kind_counts']}")
    print(f"provenance: {report['provenance']['status']}")
    print(f"coordinate quality: {report['coordinates']['quality_report']['status']}")


if __name__ == "__main__":
    main()
