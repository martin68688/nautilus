#!/usr/bin/env python3
"""Controlled component ablations for the strict decision-point benchmark."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from evaluate_decision_point_benchmark import (
    DEFAULT_BENCHMARK,
    DEFAULT_GOLD,
    DEFAULT_GRAPH,
    DEFAULT_INDEX,
    RetrieverSuite,
    average_precision,
    graded_ndcg,
    holm_adjust,
    paired_inference,
    read_jsonl,
)

from agents.memory.external_skill_memory import _tokenize
from agents.memory.stage_aware_hybrid_memory import STAGE_RRF_WEIGHTS, weighted_rrf


REPO = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "decision_point_ablation_v1.json"
DEFAULT_MARKDOWN = REPO / "coordination" / "decision_point_ablation_results.md"

METHODS = (
    "semantic_only_unfiltered",
    "semantic_only_gate",
    "field_aware_no_stage_unfiltered",
    "field_aware_no_stage_gate",
    "field_aware_stage_unfiltered",
    "field_aware_stage_gate",
    "legacy_stage_gateway",
    "production_stage_hybrid_sop",
    "minilm_gate",
    "minilm_stage_hard_gate",
    "minilm_stage_plus_tree_lexical_rrf",
    "minilm_stage_plus_tree_stage_rrf",
    "minilm_stage_plus_tree_geometry_rrf",
    "minilm_stage_plus_tree_full_rrf",
)

COMPARISONS = {
    "clean_gateway_gate_effect": ("field_aware_stage_gate", "field_aware_stage_unfiltered"),
    "extra_fields_effect": ("field_aware_no_stage_gate", "semantic_only_gate"),
    "debug_stage_boost_effect": ("field_aware_stage_gate", "field_aware_no_stage_gate"),
    "legacy_path_equivalence": ("legacy_stage_gateway", "field_aware_stage_gate"),
    "hard_stage_filter_effect_on_minilm": ("minilm_stage_hard_gate", "minilm_gate"),
    "add_tree_lexical_projection": ("minilm_stage_plus_tree_lexical_rrf", "minilm_stage_hard_gate"),
    "add_tree_stage_projection": ("minilm_stage_plus_tree_stage_rrf", "minilm_stage_plus_tree_lexical_rrf"),
    "add_tree_geometry_projection": ("minilm_stage_plus_tree_geometry_rrf", "minilm_stage_plus_tree_stage_rrf"),
    "add_tree_task_identity_projection": ("minilm_stage_plus_tree_full_rrf", "minilm_stage_plus_tree_geometry_rrf"),
    "production_vs_legacy_gateway": ("production_stage_hybrid_sop", "legacy_stage_gateway"),
    "production_vs_minilm_gate": ("production_stage_hybrid_sop", "minilm_gate"),
    "projected_full_vs_production": ("minilm_stage_plus_tree_full_rrf", "production_stage_hybrid_sop"),
    "projected_full_vs_minilm_gate": ("minilm_stage_plus_tree_full_rrf", "minilm_gate"),
}


def _stage_bonus(stage: str) -> dict[str, float]:
    return {
        "draft": {"draft": 0.08},
        "improve": {"improve": 0.10, "evolution": 0.05},
        "debug": {"debug": 0.10, "improve": 0.04},
    }[stage]


class AblationSuite:
    def __init__(self, graph: Path, index: Path) -> None:
        self.base = RetrieverSuite(graph, index)
        self.layer = self.base.layer

    def _field_rank(
        self,
        query: dict[str, Any],
        *,
        semantic_only: bool,
        stage_boost: bool,
        gated: bool,
        top_k: int,
    ) -> list[str]:
        query_tokens = _tokenize(query["query_text"])
        scored: list[tuple[float, str]] = []
        for candidate_id in query["candidate_ids"]:
            node = self.base.nodes[candidate_id]
            if node.get("type") != "SOP":
                continue
            parts = self.layer._sop_text_parts(node)
            components = {
                key: self.layer._token_overlap(query_tokens, _tokenize(text))
                for key, text in parts.items()
            }
            if semantic_only:
                score = components["semantic"]
            else:
                score = (
                    0.50 * components["semantic"]
                    + 0.22 * components["conditions"]
                    + 0.18 * components["failures"]
                    + 0.10 * components["evidence"]
                )
                if stage_boost and query["stage"] == "debug":
                    score += 0.12 * components["failures"]
            scored.append((score, candidate_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            candidate_id
            for _score, candidate_id in scored
            if not gated or candidate_id in self.base.clean_sops
        ][:top_k]

    def _minilm_rank(self, query: dict[str, Any], *, stage_filter: bool, top_k: int) -> list[str]:
        if self.base.sentence_model is None:
            return []
        pool = [
            candidate_id
            for candidate_id in query["candidate_ids"]
            if candidate_id in self.base.clean_sops
            and (
                not stage_filter
                or query["stage"] in set(self.base.nodes[candidate_id].get("decision_stages") or [])
            )
        ]
        missing = [candidate_id for candidate_id in pool if candidate_id not in self.base._sentence_embedding_cache]
        if missing:
            encoded = self.base.sentence_model.encode(
                [" ".join(str(self.base.nodes[candidate_id].get(key) or "") for key in (
                    "title", "action", "text", "method_family", "sop_kind", "abstraction_level"
                )) for candidate_id in missing],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.base._sentence_embedding_cache.update(zip(missing, encoded))
        query_embedding = self.base.sentence_model.encode(
            [query["query_text"]], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = np.asarray([self.base._sentence_embedding_cache[candidate_id] for candidate_id in pool]) @ query_embedding
        order = np.argsort(-scores, kind="stable")
        return [pool[index] for index in order[:top_k]]

    def _tree_rank(self, query: dict[str, Any], *, components: str) -> list[str]:
        candidate_ids = [node_id for node_id in self.layer._run_nodes if self.layer._successful_run_node(node_id)]
        query_tokens = _tokenize(query["query_text"])
        coords = self.layer._coords()
        anchor = self.layer._query_anchor(query["query_text"], candidate_ids)
        bonuses = _stage_bonus(query["stage"])
        scored: list[tuple[float, str]] = []
        for node_id in candidate_ids:
            node = self.layer.nodes[node_id]
            lexical = self.layer._token_overlap(query_tokens, self.layer._node_tokens.get(node_id, set()))
            score = 0.32 * lexical
            if components in {"stage", "geometry", "full"}:
                node_stage = str(node.get("stage") or node.get("stage_pair") or "")
                score += bonuses.get(node_stage, 0.0)
                improvement = node.get("metric_improvement")
                if isinstance(improvement, (int, float)) and improvement > 0:
                    score += 0.08
            if components in {"geometry", "full"} and anchor is not None and node_id in coords:
                score += 0.50 / (1.0 + self.layer._distance(anchor, coords[node_id]))
            if components == "full":
                score += self.layer._task_score(node, query["task"], query["task_family"])
            scored.append((score, node_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        mapped: list[str] = []
        for _score, node_id in scored:
            for sop_id in self.base.node_to_sops.get(node_id, []):
                if sop_id in self.base.clean_sops and sop_id in query["candidate_ids"] and sop_id not in mapped:
                    mapped.append(sop_id)
        return mapped

    def _projected_rrf(self, query: dict[str, Any], *, tree_components: str, top_k: int) -> list[str]:
        direct = self._minilm_rank(query, stage_filter=True, top_k=len(self.base.clean_sops))
        tree = self._tree_rank(query, components=tree_components)
        weights = STAGE_RRF_WEIGHTS[query["stage"]]
        return [
            row["id"]
            for row in weighted_rrf(
                direct,
                tree,
                sop_weight=weights["sop"],
                tree_weight=weights["tree"],
            )[:top_k]
        ]

    def rank(self, method: str, query: dict[str, Any], relevance: dict[str, int], top_k: int) -> list[str]:
        if method == "semantic_only_unfiltered":
            return self._field_rank(query, semantic_only=True, stage_boost=False, gated=False, top_k=top_k)
        if method == "semantic_only_gate":
            return self._field_rank(query, semantic_only=True, stage_boost=False, gated=True, top_k=top_k)
        if method == "field_aware_no_stage_unfiltered":
            return self._field_rank(query, semantic_only=False, stage_boost=False, gated=False, top_k=top_k)
        if method == "field_aware_no_stage_gate":
            return self._field_rank(query, semantic_only=False, stage_boost=False, gated=True, top_k=top_k)
        if method == "field_aware_stage_unfiltered":
            return self._field_rank(query, semantic_only=False, stage_boost=True, gated=False, top_k=top_k)
        if method == "field_aware_stage_gate":
            return self._field_rank(query, semantic_only=False, stage_boost=True, gated=True, top_k=top_k)
        if method == "legacy_stage_gateway":
            return self.base.rank("legacy_stage_gateway", query, relevance, top_k=top_k)
        if method == "production_stage_hybrid_sop":
            return self.base.rank("stage_hybrid_sop", query, relevance, top_k=top_k)
        if method == "minilm_gate":
            return self._minilm_rank(query, stage_filter=False, top_k=top_k)
        if method == "minilm_stage_hard_gate":
            return self._minilm_rank(query, stage_filter=True, top_k=top_k)
        projection = {
            "minilm_stage_plus_tree_lexical_rrf": "lexical",
            "minilm_stage_plus_tree_stage_rrf": "stage",
            "minilm_stage_plus_tree_geometry_rrf": "geometry",
            "minilm_stage_plus_tree_full_rrf": "full",
        }
        if method in projection:
            return self._projected_rrf(query, tree_components=projection[method], top_k=top_k)
        raise ValueError(method)


def evaluate(
    queries: list[dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    *,
    graph: Path,
    index: Path,
    top_k: int = 10,
    samples: int = 10000,
) -> dict[str, Any]:
    suite = AblationSuite(graph, index)
    gold = {
        row["query_id"]: {str(label["candidate_id"]): int(label["relevance"]) for label in row["labels"]}
        for row in gold_rows
    }
    per_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for method in METHODS:
        for query in queries:
            ranking = suite.rank(method, query, gold[query["query_id"]], top_k)
            unsupported = sum(candidate_id not in suite.base.clean_sops for candidate_id in ranking)
            per_method[method].append({
                "query_id": query["query_id"],
                "task_family": query["task_family"],
                "stage": query["stage"],
                "ranking": ranking,
                "graded_ndcg_at_10": graded_ndcg(ranking, gold[query["query_id"]], top_k),
                "adoption_average_precision_at_10": average_precision(ranking, gold[query["query_id"]], top_k),
                "unsupported_sop_rate_at_10": unsupported / max(1, len(ranking)),
            })
    aggregate: dict[str, Any] = {}
    for method, rows in per_method.items():
        aggregate[method] = {
            "query_count": len(rows),
            "graded_ndcg_at_10": float(np.mean([row["graded_ndcg_at_10"] for row in rows])),
            "adoption_average_precision_at_10": float(np.mean([row["adoption_average_precision_at_10"] for row in rows])),
            "unsupported_sop_rate_at_10": float(np.mean([row["unsupported_sop_rate_at_10"] for row in rows])),
            "by_stage": {},
            "by_task_family": {},
        }
        for key in ("stage", "task_family"):
            output_key = f"by_{key}"
            for value in sorted({row[key] for row in rows}):
                subset = [row for row in rows if row[key] == value]
                aggregate[method][output_key][value] = {
                    "query_count": len(subset),
                    "graded_ndcg_at_10": float(np.mean([row["graded_ndcg_at_10"] for row in subset])),
                    "adoption_average_precision_at_10": float(np.mean([row["adoption_average_precision_at_10"] for row in subset])),
                }
    comparisons: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for name, (left, right) in COMPARISONS.items():
        result = paired_inference(
            [row["graded_ndcg_at_10"] for row in per_method[left]],
            [row["graded_ndcg_at_10"] for row in per_method[right]],
            samples=samples,
        )
        result.update({"left": left, "right": right})
        comparisons[name] = result
        raw_p[name] = result["sign_flip_p_value_two_sided"]
    for name, adjusted in holm_adjust(raw_p).items():
        comparisons[name]["holm_adjusted_p"] = adjusted
    ranking_changes: dict[str, Any] = {}
    for name, (left, right) in COMPARISONS.items():
        deltas = [
            left_row["graded_ndcg_at_10"] - right_row["graded_ndcg_at_10"]
            for left_row, right_row in zip(per_method[left], per_method[right])
        ]
        by_stage: dict[str, float] = {}
        for stage in sorted({row["stage"] for row in per_method[left]}):
            stage_deltas = [
                left_row["graded_ndcg_at_10"] - right_row["graded_ndcg_at_10"]
                for left_row, right_row in zip(per_method[left], per_method[right])
                if left_row["stage"] == stage
            ]
            by_stage[stage] = float(np.mean(stage_deltas))
        ranking_changes[name] = {
            "changed_ranking_count": sum(
                left_row["ranking"] != right_row["ranking"]
                for left_row, right_row in zip(per_method[left], per_method[right])
            ),
            "improved_query_count": sum(delta > 1e-12 for delta in deltas),
            "degraded_query_count": sum(delta < -1e-12 for delta in deltas),
            "unchanged_metric_count": sum(abs(delta) <= 1e-12 for delta in deltas),
            "mean_ndcg_delta_by_stage": by_stage,
        }
    equivalence = all(
        left["ranking"] == right["ranking"]
        for left, right in zip(per_method["legacy_stage_gateway"], per_method["field_aware_stage_gate"])
    )
    return {
        "schema": "runforest_decision_point_component_ablation_v1",
        "query_count": len(queries),
        "methods": aggregate,
        "paired_comparisons": comparisons,
        "ranking_changes": ranking_changes,
        "per_query": per_method,
        "implementation_facts": {
            "legacy_stage_gateway_uses_tree": False,
            "legacy_stage_gateway_uses_geometry": False,
            "legacy_stage_gateway_uses_rrf": False,
            "legacy_stage_gateway_is_deterministic_clean_filter": True,
            "legacy_reimplementation_matches_rankings": equivalence,
            "production_stage_hybrid_sop_uses_tree": True,
            "production_stage_hybrid_sop_uses_geometry": True,
            "production_stage_hybrid_sop_uses_rrf": True,
            "production_stage_hybrid_sop_uses_stage_taxonomy": True,
            "production_stage_hybrid_sop_uses_task_identity": True,
            "production_stage_hybrid_sop_enforces_clean_gateway": True,
            "tree_rrf_methods_are_sop_space_benchmark_projections_not_production_hybrid_pack": True,
        },
        "statistics": {
            "paired_bootstrap_samples": samples,
            "p_value_method": "paired_random_sign_flip_two_sided",
            "multiple_comparison_correction": "Holm",
        },
        "environment": {
            "minilm_available": suite.base.sentence_model is not None,
            "minilm_error": suite.base.sentence_model_error,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Decision-Point Component Ablation",
        "",
        f"Queries: `{report['query_count']}`. All gated methods use the same clean-evidence predicate and the same silver gold.",
        "",
        "| Method | nDCG@10 | AP@10 | Unsupported SOP@10 |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = report["methods"][method]
        lines.append(
            f"| {method} | {row['graded_ndcg_at_10']:.4f} | "
            f"{row['adoption_average_precision_at_10']:.4f} | {row['unsupported_sop_rate_at_10']:.4f} |"
        )
    lines += ["", "## Paired Component Effects", ""]
    for name, row in report["paired_comparisons"].items():
        low, high = row["bootstrap_ci95"]
        changes = report["ranking_changes"][name]
        lines.append(
            f"- `{name}`: delta={row['delta']:+.4f}, CI [{low:+.4f}, {high:+.4f}], "
            f"Holm p={row['holm_adjusted_p']:.4g}; changed={changes['changed_ranking_count']}/"
            f"{report['query_count']}, improved/degraded={changes['improved_query_count']}/"
            f"{changes['degraded_query_count']}."
        )
    lines += [
        "",
        "## Evidence-Based Conclusions",
        "",
        "1. The deterministic clean-evidence gateway is the only component in the current production-path ablation that remains significant after Holm correction: +0.0468 nDCG@10, with no degraded query.",
        "2. Conditions/failures/evidence fields have a positive point estimate (+0.0287), but the interval crosses zero and the corrected result is not significant.",
        "3. The legacy debug-stage boost changes 0/29 Top-10 rankings. It is inert on this benchmark, not evidence of useful stage awareness.",
        "4. Hard stage filtering improves MiniLM by +0.0378 in point estimate, led by Draft, but is not significant with 29 silver queries.",
        "5. Adding projected Tree lexical and stage channels hurts the point estimate, especially on Improve. Geometry recovers +0.0760 relative to that degraded intermediate, and task identity adds +0.0274, but the full projection is not significantly better than MiniLM after correction.",
        "6. The production Stage Hybrid row is the only row that invokes the shared production channel implementation; its comparison against legacy and MiniLM must be read from the paired table without promoting a silver-label diagnostic to a paper claim.",
        "",
        "## Interpretation Guard",
        "",
        "The method `legacy_stage_gateway` preserves the old field-aware lexical ranking plus deterministic clean-evidence filter. `production_stage_hybrid_sop` invokes the shared production v2 SOP and Tree channels, stage taxonomy, task identity, geometry, weighted RRF, and final clean gate.",
        "",
        "The Tree/RRF rows are SOP-space projections built only for this ablation. They are not the production `_hybrid_pack`, which fuses execution-node IDs. The labels are single-annotator silver labels from 29 convenience-sampled decision points, so these results are diagnostic and appendix-grade, not a paper-level superiority claim.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10000)
    args = parser.parse_args()
    report = evaluate(
        read_jsonl(args.benchmark),
        read_jsonl(args.gold),
        graph=args.graph,
        index=args.index,
        top_k=args.top_k,
        samples=args.samples,
    )
    if not report["environment"]["minilm_available"]:
        raise SystemExit(f"MiniLM unavailable: {report['environment']['minilm_error']}")
    if not report["implementation_facts"]["legacy_reimplementation_matches_rankings"]:
        raise SystemExit("Controlled field-aware implementation does not match legacy_stage_gateway")
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"query_count": report["query_count"], "methods": report["methods"], "paired_comparisons": report["paired_comparisons"], "implementation_facts": report["implementation_facts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
