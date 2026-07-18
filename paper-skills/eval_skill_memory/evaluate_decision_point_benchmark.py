#!/usr/bin/env python3
"""Evaluate retrieval systems on the independent RunForest decision benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer  # noqa: E402
from agents.memory.external_skill_memory import _tokenize  # noqa: E402


DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
DEFAULT_INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
DEFAULT_BENCHMARK = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "decision_point_benchmark_v1.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "decision_point_silver_gold_v1.jsonl"
DEFAULT_REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "decision_point_retrieval_evaluation_v1.json"
DEFAULT_MARKDOWN = REPO / "coordination" / "decision_point_benchmark_results.md"
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

METHODS = (
    "random_unfiltered",
    "bm25_unfiltered",
    "bm25_safety_filtered",
    "tfidf_unfiltered",
    "tfidf_safety_filtered",
    "lsa_dense_unfiltered",
    "lsa_dense_safety_filtered",
    "minilm_dense_unfiltered",
    "minilm_dense_safety_filtered",
    "tree_only_mapped_no_task",
    "tree_only_mapped_task_aware",
    "legacy_stage_gateway",
    "stage_hybrid_sop",
    "oracle_upper",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def candidate_text(node: dict[str, Any]) -> str:
    if node.get("type") == "SOP":
        return " ".join(
            str(node.get(key) or "")
            for key in ("title", "action", "text", "method_family", "sop_kind", "abstraction_level")
        )
    return " ".join(str(node.get(key) or "") for key in ("plan", "code_summary", "analysis", "terminal_excerpt", "text"))


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_+-]{2,}", text.lower())


def bm25_scores(query: str, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    tokenized = [tokenize(text) for text in documents]
    query_tokens = tokenize(query)
    lengths = np.asarray([len(row) for row in tokenized], dtype=np.float64)
    avgdl = float(lengths.mean()) if len(lengths) else 1.0
    doc_freq: Counter[str] = Counter()
    term_freq = []
    for row in tokenized:
        counts = Counter(row)
        term_freq.append(counts)
        doc_freq.update(counts.keys())
    scores = np.zeros(len(documents), dtype=np.float64)
    n_docs = max(1, len(documents))
    for token in query_tokens:
        df = doc_freq.get(token, 0)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for index, counts in enumerate(term_freq):
            tf = counts.get(token, 0)
            if not tf:
                continue
            denom = tf + k1 * (1.0 - b + b * lengths[index] / max(avgdl, 1.0))
            scores[index] += idf * tf * (k1 + 1.0) / denom
    return scores


class NumpyTfidfLsaIndex:
    """Small dependency-free TF-IDF and latent semantic index."""

    def __init__(self, candidate_ids: list[str], documents: list[str], *, components: int = 64):
        self.candidate_ids = candidate_ids
        self.row_by_id = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
        tokenized = [tokenize(text) for text in documents]
        df: Counter[str] = Counter()
        for row in tokenized:
            df.update(set(row))
        terms = sorted(df, key=lambda term: (-df[term], term))[:12000]
        self.vocab = {term: index for index, term in enumerate(terms)}
        self.idf = np.asarray(
            [math.log((1.0 + len(documents)) / (1.0 + df[term])) + 1.0 for term in terms],
            dtype=np.float64,
        )
        self.matrix = np.zeros((len(documents), len(terms)), dtype=np.float64)
        for row_index, row in enumerate(tokenized):
            counts = Counter(row)
            for term, count in counts.items():
                column = self.vocab.get(term)
                if column is not None:
                    self.matrix[row_index, column] = (1.0 + math.log(count)) * self.idf[column]
        self.matrix = self._normalize(self.matrix)
        rank = min(components, max(1, min(self.matrix.shape) - 1))
        gram = self.matrix @ self.matrix.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1][:rank]
        singular = np.sqrt(np.maximum(eigenvalues[order], 1e-12))
        self.lsa_docs = self._normalize(eigenvectors[:, order] * singular)
        self.lsa_projection = (eigenvectors[:, order].T @ self.matrix) / singular[:, None]

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def query_vector(self, text: str) -> np.ndarray:
        vector = np.zeros(len(self.vocab), dtype=np.float64)
        counts = Counter(tokenize(text))
        for term, count in counts.items():
            column = self.vocab.get(term)
            if column is not None:
                vector[column] = (1.0 + math.log(count)) * self.idf[column]
        norm = np.linalg.norm(vector)
        return vector / max(norm, 1e-12)

    def tfidf_scores(self, text: str, pool: list[str]) -> np.ndarray:
        query = self.query_vector(text)
        return np.asarray([self.matrix[self.row_by_id[cid]] @ query for cid in pool])

    def lsa_scores(self, text: str, pool: list[str]) -> np.ndarray:
        query = self.query_vector(text) @ self.lsa_projection.T
        query = query / max(np.linalg.norm(query), 1e-12)
        return np.asarray([self.lsa_docs[self.row_by_id[cid]] @ query for cid in pool])


def graded_ndcg(ranking: list[str], relevance: dict[str, int], k: int = 10) -> float:
    gains = [relevance.get(candidate_id, 0) for candidate_id in ranking[:k]]
    dcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def average_precision(ranking: list[str], relevance: dict[str, int], k: int = 10) -> float:
    acceptable = {candidate_id for candidate_id, grade in relevance.items() if grade >= 2}
    if not acceptable:
        return 0.0
    hits = 0
    total = 0.0
    for index, candidate_id in enumerate(ranking[:k], 1):
        if candidate_id in acceptable:
            hits += 1
            total += hits / index
    return total / min(len(acceptable), k)


def paired_inference(left: list[float], right: list[float], *, samples: int = 10000, seed: int = 29) -> dict[str, Any]:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    if not len(delta):
        return {"delta": 0.0, "bootstrap_ci95": [0.0, 0.0], "sign_flip_p_value_two_sided": 1.0}
    bootstrap_rng = np.random.default_rng(seed)
    sign_flip_rng = np.random.default_rng(seed + 1)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = float(delta[bootstrap_rng.integers(0, len(delta), len(delta))].mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    observed = abs(float(delta.mean()))
    null_means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        signs = sign_flip_rng.choice((-1.0, 1.0), size=len(delta))
        null_means[index] = float((delta * signs).mean())
    return {
        "delta": float(delta.mean()),
        "bootstrap_ci95": [float(lower), float(upper)],
        "sign_flip_p_value_two_sided": float((np.count_nonzero(np.abs(null_means) >= observed) + 1) / (samples + 1)),
    }


paired_bootstrap = paired_inference


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


class RetrieverSuite:
    def __init__(self, graph_path: Path, index_path: Path, *, sentence_model: str = DEFAULT_SENTENCE_MODEL):
        self.graph = json.loads(graph_path.read_text(encoding="utf-8"))
        self.nodes = {str(node["id"]): node for node in self.graph.get("nodes", []) if node.get("id")}
        self.layer = StageAwareHybridMemoryLayer(
            graph_path=str(graph_path),
            index_path=str(index_path),
            source_name="decision_point_benchmark",
            mode="run_forest_stage_hybrid",
            scoring_mode="poincare",
            enable_agentic=False,
            top_k=20,
            max_chars=0,
        )
        self.clean_sops = {
            sop_id
            for sop_id in self.layer._sops
            if any(self.layer._positive_transition(tid)[0] for tid in self.layer._transitions_by_sop.get(sop_id, []))
        }
        self.node_to_sops: dict[str, list[str]] = defaultdict(list)
        for node in self.graph.get("nodes", []):
            if node.get("type") != "Transition":
                continue
            sop_ids = [str(value) for value in node.get("attached_sop_ids") or []]
            canonical = [value if value.startswith("sop::") else f"sop::{value}" for value in sop_ids]
            for node_id in (str(node.get("id") or ""), str(node.get("parent_node_id") or ""), str(node.get("child_node_id") or "")):
                self.node_to_sops[node_id].extend(canonical)
        self.sentence_model_name = sentence_model
        self.sentence_model = None
        self.sentence_model_error = ""
        self._sentence_embedding_cache: dict[str, np.ndarray] = {}
        try:
            if int(np.__version__.split(".", 1)[0]) >= 2:
                raise RuntimeError("local sentence-transformer stack is disabled because SciPy was compiled for NumPy 1.x")
            from sentence_transformers import SentenceTransformer

            self.sentence_model = SentenceTransformer(sentence_model, local_files_only=True, device="cpu")
        except Exception as exc:  # optional local baseline
            self.sentence_model_error = f"{type(exc).__name__}: {exc}"
        corpus_ids = [
            node_id for node_id, node in self.nodes.items()
            if node.get("type") == "SOP"
            or (
                node.get("type") == "RunNode"
                and len(candidate_text(node)) >= 120
                and isinstance(node.get("leakage_audit"), dict)
                and (
                    node["leakage_audit"].get("rank_eligible") is False
                    or node["leakage_audit"].get("memory_disposition") in {"quarantine", "negative_only"}
                )
            )
        ]
        # Candidate pools are drawn from SOPs plus the first deterministic blocked nodes.
        sop_ids = sorted(cid for cid in corpus_ids if self.nodes[cid].get("type") == "SOP")
        blocked_ids = sorted(
            (cid for cid in corpus_ids if self.nodes[cid].get("type") == "RunNode"),
            key=lambda cid: hashlib.sha256(cid.encode()).hexdigest(),
        )
        self.text_index = NumpyTfidfLsaIndex(
            [*sop_ids, *blocked_ids],
            [candidate_text(self.nodes[cid]) for cid in [*sop_ids, *blocked_ids]],
        )

    def safe(self, candidate_id: str) -> bool:
        return candidate_id in self.clean_sops

    @staticmethod
    def _order(pool: list[str], scores: np.ndarray, allowed: set[str] | None, top_k: int) -> list[str]:
        return [
            pool[index]
            for index in np.argsort(-scores, kind="stable")
            if allowed is None or pool[index] in allowed
        ][:top_k]

    def rank(self, method: str, query: dict[str, Any], relevance: dict[str, int], *, top_k: int = 10) -> list[str]:
        pool = list(query["candidate_ids"])
        pool_set = set(pool)
        documents = [candidate_text(self.nodes[candidate_id]) for candidate_id in pool]
        text = query["query_text"]
        if method == "oracle_upper":
            safe_pool = [candidate_id for candidate_id in pool if self.safe(candidate_id)]
            return sorted(safe_pool, key=lambda cid: (-relevance.get(cid, 0), cid))[:top_k]
        if method == "random_unfiltered":
            eligible = list(pool)
            eligible.sort(key=lambda cid: hashlib.sha256(f"{query['query_id']}|{cid}".encode()).hexdigest())
            return eligible[:top_k]
        if method in {"bm25_unfiltered", "bm25_safety_filtered"}:
            scores = bm25_scores(text, documents)
            allowed = self.clean_sops if method.endswith("safety_filtered") else None
            return self._order(pool, scores, allowed, top_k)
        if method in {"tfidf_unfiltered", "tfidf_safety_filtered"}:
            scores = self.text_index.tfidf_scores(text, pool)
            allowed = self.clean_sops if method.endswith("safety_filtered") else None
            return self._order(pool, scores, allowed, top_k)
        if method in {"lsa_dense_unfiltered", "lsa_dense_safety_filtered"}:
            scores = self.text_index.lsa_scores(text, pool)
            allowed = self.clean_sops if method.endswith("safety_filtered") else None
            return self._order(pool, scores, allowed, top_k)
        if method in {"minilm_dense_unfiltered", "minilm_dense_safety_filtered"}:
            if self.sentence_model is None:
                return []
            missing = [cid for cid in pool if cid not in self._sentence_embedding_cache]
            if missing:
                encoded = self.sentence_model.encode(
                    [candidate_text(self.nodes[cid]) for cid in missing],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                self._sentence_embedding_cache.update(zip(missing, encoded))
            query_embedding = self.sentence_model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
            embeddings = np.asarray([self._sentence_embedding_cache[cid] for cid in pool])
            scores = embeddings @ query_embedding
            allowed = self.clean_sops if method.endswith("safety_filtered") else None
            return self._order(pool, scores, allowed, top_k)
        if method == "legacy_stage_gateway":
            query_tokens = _tokenize(text)
            scored = []
            for candidate_id in pool:
                node = self.nodes[candidate_id]
                if node.get("type") != "SOP":
                    continue
                parts = self.layer._sop_text_parts(node)
                components = {
                    key: self.layer._token_overlap(query_tokens, _tokenize(value))
                    for key, value in parts.items()
                }
                score = (
                    0.50 * components["semantic"]
                    + 0.22 * components["conditions"]
                    + 0.18 * components["failures"]
                    + 0.10 * components["evidence"]
                )
                if query["stage"] == "debug":
                    score += 0.12 * components["failures"]
                scored.append((score, candidate_id))
            scored.sort(key=lambda item: (-item[0], item[1]))
            return [candidate_id for _score, candidate_id in scored if self.safe(candidate_id)][:top_k]
        if method == "stage_hybrid_sop":
            pack = self.layer.rank_sop_hybrid(
                stage=query["stage"],
                task_id=query["task"],
                task_desc=query["task_family"],
                query_text=text,
                limit=top_k,
                allowed_sop_ids=pool_set,
            )
            return [row["id"] for row in pack["fused_sop_candidates"]]
        if method in {"tree_only_mapped_no_task", "tree_only_mapped_task_aware"}:
            task_aware = method.endswith("task_aware")
            tree_ids = self.layer._rank_tree(
                stage=query["stage"],
                query_text=text,
                task_id=query["task"] if task_aware else "",
                task_desc=query["task_family"] if task_aware else "",
                limit=100,
            )
            mapped = []
            for node_id in tree_ids:
                for sop_id in self.node_to_sops.get(node_id, []):
                    if sop_id in pool_set and sop_id not in mapped:
                        mapped.append(sop_id)
            return mapped[:top_k]
        raise ValueError(f"unknown method: {method}")


def evaluate(
    queries: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    graph_path: Path = DEFAULT_GRAPH,
    index_path: Path = DEFAULT_INDEX,
    top_k: int = 10,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    suite = RetrieverSuite(graph_path, index_path)
    gold_by_id = {row["query_id"]: row for row in gold_rows}
    per_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    availability = {method: True for method in METHODS}
    if suite.sentence_model is None:
        availability["minilm_dense_unfiltered"] = False
        availability["minilm_dense_safety_filtered"] = False

    for query in queries:
        labels = gold_by_id[query["query_id"]]["labels"]
        relevance = {row["candidate_id"]: int(row["relevance"]) for row in labels}
        for method in METHODS:
            started = time.perf_counter()
            ranking = suite.rank(method, query, relevance, top_k=top_k) if availability[method] else []
            latency = time.perf_counter() - started
            blocked_runs = [candidate_id for candidate_id in ranking if suite.nodes[candidate_id].get("type") == "RunNode"]
            unsupported_sops = [
                candidate_id for candidate_id in ranking
                if suite.nodes[candidate_id].get("type") == "SOP" and not suite.safe(candidate_id)
            ]
            non_admissible = [candidate_id for candidate_id in ranking if not suite.safe(candidate_id)]
            families = {
                str(suite.nodes[candidate_id].get("method_family") or "general")
                for candidate_id in ranking
                if suite.safe(candidate_id) and suite.nodes[candidate_id].get("type") == "SOP"
            }
            per_method[method].append(
                {
                    "query_id": query["query_id"],
                    "task_family": query["task_family"],
                    "stage": query["stage"],
                    "ranking": ranking,
                    "graded_ndcg_at_10": graded_ndcg(ranking, relevance, top_k),
                    "adoption_average_precision_at_10": average_precision(ranking, relevance, top_k),
                    "blocked_run_node_count_at_10": len(blocked_runs),
                    "blocked_run_node_rate_at_10": len(blocked_runs) / max(1, len(ranking)),
                    "unsupported_sop_count_at_10": len(unsupported_sops),
                    "unsupported_sop_rate_at_10": len(unsupported_sops) / max(1, len(ranking)),
                    "non_admissible_rate_at_10": len(non_admissible) / max(1, len(ranking)),
                    "method_family_diversity_at_10": len(families),
                    "returned_count": len(ranking),
                    "latency_sec": latency,
                }
            )

    metric_names = (
        "graded_ndcg_at_10",
        "adoption_average_precision_at_10",
        "blocked_run_node_count_at_10",
        "blocked_run_node_rate_at_10",
        "unsupported_sop_count_at_10",
        "unsupported_sop_rate_at_10",
        "non_admissible_rate_at_10",
        "method_family_diversity_at_10",
        "returned_count",
        "latency_sec",
    )
    aggregate: dict[str, Any] = {}
    for method, rows in per_method.items():
        aggregate[method] = {
            "available": availability[method],
            "query_count": len(rows) if availability[method] else 0,
            **{
                metric: float(np.mean([row[metric] for row in rows])) if rows and availability[method] else None
                for metric in metric_names
            },
            "by_task_family": {},
            "by_stage": {},
        }
        if availability[method]:
            for family in sorted({row["task_family"] for row in rows}):
                subset = [row for row in rows if row["task_family"] == family]
                aggregate[method]["by_task_family"][family] = {
                    "query_count": len(subset),
                    "descriptive_only": True,
                    **{metric: float(np.mean([row[metric] for row in subset])) for metric in metric_names},
                }
            for stage in sorted({row["stage"] for row in rows}):
                subset = [row for row in rows if row["stage"] == stage]
                aggregate[method]["by_stage"][stage] = {
                    "query_count": len(subset),
                    "descriptive_only": True,
                    "claim_eligible_sample_size": False,
                    **{metric: float(np.mean([row[metric] for row in subset])) for metric in metric_names},
                }
            family_rows = list(aggregate[method]["by_task_family"].values())
            aggregate[method]["macro_task_family_graded_ndcg_at_10"] = float(
                np.mean([row["graded_ndcg_at_10"] for row in family_rows])
            )
            aggregate[method]["macro_task_family_adoption_average_precision_at_10"] = float(
                np.mean([row["adoption_average_precision_at_10"] for row in family_rows])
            )

    comparison_pairs = {
        "bm25_gate_effect": ("bm25_safety_filtered", "bm25_unfiltered"),
        "tfidf_gate_effect": ("tfidf_safety_filtered", "tfidf_unfiltered"),
        "lsa_gate_effect": ("lsa_dense_safety_filtered", "lsa_dense_unfiltered"),
        "minilm_gate_effect": ("minilm_dense_safety_filtered", "minilm_dense_unfiltered"),
        "task_identity_effect_on_tree": ("tree_only_mapped_task_aware", "tree_only_mapped_no_task"),
        "true_stage_hybrid_vs_legacy_gateway": ("stage_hybrid_sop", "legacy_stage_gateway"),
        "stage_hybrid_vs_tree_no_task": ("stage_hybrid_sop", "tree_only_mapped_no_task"),
        "stage_hybrid_vs_bm25_safety_filtered": ("stage_hybrid_sop", "bm25_safety_filtered"),
        "stage_hybrid_vs_tfidf_safety_filtered": ("stage_hybrid_sop", "tfidf_safety_filtered"),
        "stage_hybrid_vs_lsa_safety_filtered": ("stage_hybrid_sop", "lsa_dense_safety_filtered"),
        "stage_hybrid_vs_minilm_safety_filtered": ("stage_hybrid_sop", "minilm_dense_safety_filtered"),
    }
    comparisons: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for name, (left, right) in comparison_pairs.items():
        if not availability[left] or not availability[right]:
            continue
        comparison = paired_inference(
            [row["graded_ndcg_at_10"] for row in per_method[left]],
            [row["graded_ndcg_at_10"] for row in per_method[right]],
            samples=bootstrap_samples,
        )
        comparison["left"] = left
        comparison["right"] = right
        comparisons[name] = comparison
        p_values[name] = comparison["sign_flip_p_value_two_sided"]
    adjusted = holm_adjust(p_values)
    for name, value in adjusted.items():
        comparisons[name]["holm_adjusted_p"] = value

    all_adjudicated = all(row.get("annotator_count", 0) >= 2 and row.get("adjudicated") is True for row in gold_rows)
    report = {
        "schema": "runforest_decision_point_retrieval_evaluation_v1",
        "query_count": len(queries),
        "benchmark_sampling": {
            "seed_query_count": 60,
            "retained_query_count": len(queries),
            "strict_retention_rate": len(queries) / 60,
            "by_task_family": dict(sorted(Counter(row["task_family"] for row in queries).items())),
            "selection": "deterministic convenience seeds filtered for global gold uniqueness, explicit task compatibility, clean evidence, and task-matched blocked distractors",
            "representative_random_sample": False,
        },
        "methods": aggregate,
        "paired_comparisons": comparisons,
        "per_query": per_method,
        "statistics": {
            "paired_bootstrap_samples": bootstrap_samples,
            "confidence_interval": 0.95,
            "confidence_interval_method": "paired_nonparametric_bootstrap",
            "p_value_method": "paired_random_sign_flip_two_sided",
            "multiple_comparison_correction": "Holm",
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "sentence_transformer_model": suite.sentence_model_name,
            "sentence_transformer_available": suite.sentence_model is not None,
        },
        "dense_baseline": {
            "model": suite.sentence_model_name,
            "available": suite.sentence_model is not None,
            "error": suite.sentence_model_error,
        },
        "claim_gates": {
            "independent_decision_points": True,
            "historical_child_as_gold": False,
            "six_task_families": len({row["task_family"] for row in queries}) >= 6,
            "minimum_25_strict_queries": len(queries) >= 25,
            "unique_gold_sets_globally": len({frozenset(label["candidate_id"] for label in row["labels"]) for row in gold_rows}) == len(gold_rows),
            "task_compatible_gold_only": True,
            "task_matched_blocked_distractors": True,
            "like_for_like_text_scorer_pairs": True,
            "convenience_seed_selection_disclosed": True,
            "two_blind_annotators_and_adjudication": all_adjudicated,
            "offline_retrieval_claim_allowed": all_adjudicated,
            "online_downstream_claim_allowed": False,
            "reason": "Current labels are silver expert seeds; blind human adjudication and concurrent online training remain outstanding.",
        },
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Independent Decision-Point Retrieval Benchmark",
        "",
        f"Queries: `{report['query_count']}`. These are independent decision descriptions, not historical child-recovery queries.",
        "",
        "| Method | Available | graded nDCG@10 | Adoption AP@10 | blocked RunNode@10 | unsupported SOP@10 | non-admissible@10 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]
        if not row["available"]:
            lines.append(f"| {method} | no | - | - | - | - | - |")
            continue
        lines.append(
            f"| {method} | yes | {row['graded_ndcg_at_10']:.4f} | "
            f"{row['adoption_average_precision_at_10']:.4f} | {row['blocked_run_node_rate_at_10']:.4f} | "
            f"{row['unsupported_sop_rate_at_10']:.4f} | {row['non_admissible_rate_at_10']:.4f} |"
        )
    lines += ["", "## Claim Gates", ""]
    for key, value in report["claim_gates"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Paired Comparisons", ""]
    for name, row in report["paired_comparisons"].items():
        low, high = row["bootstrap_ci95"]
        lines.append(
            f"- `{name}`: delta={row['delta']:.4f}, bootstrap 95% CI "
            f"[{low:.4f}, {high:.4f}], sign-flip Holm p={row['holm_adjusted_p']:.4g}."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "This run is a silver-label diagnostic over 29 of 60 deterministic convenience seeds. The retained task-family distribution is uneven, so per-family and per-stage values are descriptive only. It cannot support a paper claim until two blind annotators adjudicate every decision point. Online downstream improvement is not measured here.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args()
    report = evaluate(
        read_jsonl(args.benchmark),
        read_jsonl(args.gold),
        graph_path=args.graph,
        index_path=args.index,
        top_k=args.top_k,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"query_count": report["query_count"], "methods": report["methods"], "claim_gates": report["claim_gates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
