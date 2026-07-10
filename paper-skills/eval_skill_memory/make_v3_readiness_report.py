"""Generate the V3 edge/radius/embedding readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_EDGE_VALIDATION = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "benchmark_validation_report_edge.json"
DEFAULT_HARD_USE = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_ablation_hard_use_gold_hint.json"
DEFAULT_HARD_PRED = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_ablation_hard_predicted_only.json"
DEFAULT_EDGE_USE = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_ablation_edge_use_gold_hint.json"
DEFAULT_EDGE_PRED = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_ablation_edge_predicted_only.json"
DEFAULT_HARD_LEARNED = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_ablation_hard_learned_predictor.json"
DEFAULT_EDGE_LEARNED = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_ablation_edge_learned_predictor.json"
DEFAULT_BUILDER = REPO / "paper-skills" / "hyper_memory" / "graph_builder_report.json"
DEFAULT_EMBEDDING_REPORT = REPO / "paper-skills" / "hyper_memory" / "embedding_backend_report.json"
DEFAULT_OUTPUT = REPO / "coordination" / "hyperbolic_geometry_v3_edge_radius_embedding_readiness.md"


def load(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def light(ok: bool | None) -> str:
    if ok is True:
        return "GREEN"
    if ok is False:
        return "RED"
    return "YELLOW"


def edge_claim(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    return report.get("comparisons", {}).get("poincare_vs_flat_twin_edge_claim", {})


def metric_line(label: str, report: dict[str, Any] | None) -> list[str]:
    if not report:
        return [f"- {label}: not run"]
    systems = report.get("systems_edge_gold_only") or report.get("systems") or {}
    p = systems.get("agentic_poincare", {})
    f = systems.get("agentic_flat_twin", {})
    claim = edge_claim(report)
    edge_rr = claim.get("edge_recall_at_5", {})
    cp = claim.get("edge_condition_precision", {})
    rank = claim.get("rank_metrics", {})
    metadata = report.get("run_metadata", {})
    return [
        f"- {label}: status={claim.get('status', report.get('status', 'missing'))}",
        f"  - radius_hint_modes={metadata.get('radius_hint_modes', [])}",
        f"  - Poincare Edge Recall@5 / MRR / NDCG@5: {p.get('edge_recall_at_5', 'n/a')} / {p.get('edge_mrr', 'n/a')} / {p.get('edge_ndcg_at_5', 'n/a')}",
        f"  - Flat-Twin Edge Recall@5 / MRR / NDCG@5: {f.get('edge_recall_at_5', 'n/a')} / {f.get('edge_mrr', 'n/a')} / {f.get('edge_ndcg_at_5', 'n/a')}",
        f"  - Edge Recall diff / p-value: {edge_rr.get('observed_mean_diff', 'n/a')} / {edge_rr.get('p_value', 'n/a')}",
        f"  - Edge Condition Precision diff: {cp.get('mean_diff', 'n/a')}",
        f"  - Edge MRR diff / NDCG diff: {rank.get('edge_mrr', {}).get('observed_mean_diff', 'n/a')} / {rank.get('edge_ndcg_at_5', {}).get('observed_mean_diff', 'n/a')}",
        f"  - Query-aware quality: {report.get('query_aware_coordinate_quality', {}).get('status', 'n/a')}",
    ]


def render(
    *,
    edge_validation: dict[str, Any] | None,
    hard_use: dict[str, Any] | None,
    hard_pred: dict[str, Any] | None,
    edge_use: dict[str, Any] | None,
    edge_pred: dict[str, Any] | None,
    hard_learned: dict[str, Any] | None,
    edge_learned: dict[str, Any] | None,
    builder_report: dict[str, Any] | None,
    embedding_report: dict[str, Any] | None,
) -> str:
    edge_benchmark_ok = bool(edge_validation and edge_validation.get("passed"))
    edge_pred_claim = edge_claim(edge_pred)
    edge_pred_passed = bool(edge_pred_claim.get("passed"))
    provenance_ok = bool(builder_report and builder_report.get("provenance", {}).get("paper_grade"))
    direction = (builder_report or {}).get("coordinates", {}).get("direction", {})
    embedding_unavailable = bool(embedding_report and embedding_report.get("status") == "embedding_backend_unavailable")
    direction_backend = direction.get("method", "missing")
    sentence_actual = direction_backend in {"sentence_embedding_svd", "contrastive_projection"}
    query_quality_ok = bool(edge_pred and edge_pred.get("query_aware_coordinate_quality", {}).get("passed"))
    paper_claim_ok = edge_benchmark_ok and provenance_ok and sentence_actual and query_quality_ok and edge_pred_passed

    lines = [
        "# Hyperbolic Geometry V3 Readiness",
        "",
        "This report focuses on the V3 diagnostic question: does Poincare distance help on edge SOP retrieval when radius hints are not gold-derived?",
        "",
        "## Status Lights",
        "",
        f"- {light(provenance_ok)} provenance: {'paper-grade clean provenance present' if provenance_ok else 'not paper-grade or report missing'}",
        f"- {light(edge_benchmark_ok)} edge benchmark: {'validated' if edge_benchmark_ok else 'failed or not run'}",
        f"- {light(sentence_actual)} sentence/contrastive direction backend: {direction_backend}",
        f"- {light(query_quality_ok)} query-aware coordinate gate: {(edge_pred or {}).get('query_aware_coordinate_quality', {}).get('status', 'not run')}",
        f"- {light(edge_pred_passed)} edge predicted-only geometry claim: {edge_pred_claim.get('status', 'not run')}",
        f"- {light(paper_claim_ok)} paper-grade V3 geometry claim: {'allowed' if paper_claim_ok else 'not allowed'}",
        "",
    ]
    if embedding_unavailable:
        lines += [
            "## Embedding Backend",
            "",
            f"- Strict embedding experiment did not run: {embedding_report.get('error', 'unknown error')}",
            "- This is correct fail-closed behavior; do not label TF-IDF fallback as sentence embedding.",
            "",
        ]
    elif builder_report:
        lines += [
            "## Embedding Backend",
            "",
            f"- Direction backend: {direction_backend}",
            f"- Embedding model: {direction.get('embedding_model', 'n/a')}",
            f"- Projection method: {direction.get('projection_method', direction.get('method', 'n/a'))}",
            f"- Embedding quality confidence: {direction.get('embedding_quality_confidence', 'n/a')}",
            "",
        ]

    lines += [
        "## Edge Benchmark",
        "",
    ]
    if edge_validation:
        lines += [
            f"- Queries: {edge_validation.get('queries', 'n/a')}",
            f"- By kind: {edge_validation.get('by_kind', {})}",
            f"- Split: {edge_validation.get('by_split', {})}",
            f"- Title-token overlap mean/max: {edge_validation.get('title_token_overlap_mean', 'n/a')} / {edge_validation.get('title_token_overlap_max', 'n/a')}",
            f"- Distractor count mean/min: {edge_validation.get('distractor_count_mean', 'n/a')} / {edge_validation.get('distractor_count_min', 'n/a')}",
            f"- Errors: {edge_validation.get('errors', [])[:5]}",
            "",
        ]
    else:
        lines += ["- Edge validation report missing.", ""]

    lines += [
        "## Radius Hint Ablation",
        "",
        *metric_line("hard + use_gold_hint", hard_use),
        *metric_line("hard + predicted_only", hard_pred),
        *metric_line("edge + use_gold_hint", edge_use),
        *metric_line("edge + predicted_only", edge_pred),
        *metric_line("hard + learned_predictor", hard_learned),
        *metric_line("edge + learned_predictor", edge_learned),
        "",
        "## Interpretation Guardrail",
        "",
    ]
    if paper_claim_ok:
        lines.append("V3 passes the cleanest edge predicted-only geometry gate. This supports a hyperbolic-distance-specific claim, subject to human benchmark audit.")
    elif edge_pred:
        lines.append("V3 does not currently allow a paper-grade hyperbolic geometry claim. If Poincare improves over lexical but not Flat-Twin, report agentic memory or coordinate-quality gains only.")
    else:
        lines.append("The V3 edge predicted-only run is missing, so the geometry question is still untested.")
    lines += [
        "",
        "Main claim requires: edge benchmark + predicted_only + sentence/contrastive backend + Poincare Edge Recall@5 >= Flat-Twin + 5pp + paired-bootstrap p < 0.05 + Edge Condition Precision/MRR/NDCG not lower.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write V3 hyperbolic geometry readiness report.")
    parser.add_argument("--edge-validation", type=Path, default=DEFAULT_EDGE_VALIDATION)
    parser.add_argument("--hard-use", type=Path, default=DEFAULT_HARD_USE)
    parser.add_argument("--hard-predicted", type=Path, default=DEFAULT_HARD_PRED)
    parser.add_argument("--edge-use", type=Path, default=DEFAULT_EDGE_USE)
    parser.add_argument("--edge-predicted", type=Path, default=DEFAULT_EDGE_PRED)
    parser.add_argument("--hard-learned", type=Path, default=DEFAULT_HARD_LEARNED)
    parser.add_argument("--edge-learned", type=Path, default=DEFAULT_EDGE_LEARNED)
    parser.add_argument("--builder-report", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--embedding-report", type=Path, default=DEFAULT_EMBEDDING_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    text = render(
        edge_validation=load(args.edge_validation),
        hard_use=load(args.hard_use),
        hard_pred=load(args.hard_predicted),
        edge_use=load(args.edge_use),
        edge_pred=load(args.edge_predicted),
        hard_learned=load(args.hard_learned),
        edge_learned=load(args.edge_learned),
        builder_report=load(args.builder_report),
        embedding_report=load(args.embedding_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
