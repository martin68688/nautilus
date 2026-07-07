"""Generate a human-readable claim readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_BUILDER_REPORT = REPO / "paper-skills" / "hyper_memory" / "graph_builder_report.json"
DEFAULT_QUALITY = REPO / "paper-skills" / "hyper_memory" / "coordinate_quality_report.json"
DEFAULT_BENCH_VALIDATION = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "benchmark_validation_report.json"
DEFAULT_ABLATION = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_ablation_report.json"
DEFAULT_OUTPUT = REPO / "coordination" / "hyperbolic_claim_readiness.md"


def load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def light(ok: bool | None) -> str:
    if ok is True:
        return "GREEN"
    if ok is False:
        return "RED"
    return "YELLOW"


def render(
    *,
    builder_report: dict[str, Any] | None,
    quality_report: dict[str, Any] | None,
    benchmark_report: dict[str, Any] | None,
    ablation_report: dict[str, Any] | None,
) -> str:
    provenance_ok = bool(builder_report and builder_report.get("provenance", {}).get("paper_grade"))
    quality_ok = bool(quality_report and quality_report.get("passed"))
    benchmark_ok = bool(benchmark_report and benchmark_report.get("passed"))
    ablation_claim_grade = bool(ablation_report and ablation_report.get("claim_grade", {}).get("claim_grade"))
    ablation_passed = bool(ablation_report and ablation_report.get("passed"))

    lines = [
        "# Hyperbolic SOP Memory Claim Readiness",
        "",
        "This report separates engineering readiness from paper-grade geometry claims.",
        "",
        "## Status Lights",
        "",
        f"- {light(provenance_ok)} provenance: {'paper-grade clean provenance present' if provenance_ok else 'not paper-grade or report missing'}",
        f"- {light(quality_ok)} coordinate quality: {'gate passed' if quality_ok else 'gate failed or report missing'}",
        f"- {light(benchmark_ok)} benchmark/gold validation: {'passed' if benchmark_ok else 'failed or not run'}",
        f"- {light(ablation_claim_grade)} ablation claim-grade inputs: {'ready' if ablation_claim_grade else 'blocked by provenance/quality/report'}",
        f"- {light(ablation_passed)} hyperbolic geometry claim: {'passed' if ablation_passed else 'not supported yet'}",
        "- YELLOW online pilot: not run by this offline evidence-chain script",
        "",
        "## Current Interpretation",
        "",
    ]
    if ablation_passed:
        lines.append("Agentic Poincare beat same-coordinate Flat-Twin under the pre-registered gate. This supports a geometry claim, subject to benchmark human audit.")
    elif ablation_report and ablation_report.get("status") == "not_claim_grade":
        blockers = ", ".join(ablation_report.get("claim_blockers", [])) or "unknown blockers"
        lines.append(f"The retrieval harness ran, but the result is not claim-grade because: {blockers}. Do not describe this as hyperbolic geometry failing.")
    elif ablation_report:
        lines.append("The ablation did not pass the geometry gate. It may still show agentic memory helps, but not that hyperbolic geometry itself helps.")
    else:
        lines.append("Ablation has not been run yet. The system is not ready for a geometry claim.")

    lines += [
        "",
        "## Key Numbers",
        "",
    ]
    if builder_report:
        prov = builder_report.get("provenance", {})
        hyper = builder_report.get("hyper_graph", {})
        lines += [
            f"- Hyper graph nodes/edges: {hyper.get('nodes', 'n/a')} / {hyper.get('edges', 'n/a')}",
            f"- SOP source evidence: {prov.get('nodes_with_source_evidence', 'n/a')} / {prov.get('nodes_total', 'n/a')}",
            f"- Provenance status: {prov.get('status', 'missing')}",
        ]
    if quality_report:
        lines += [
            f"- Direction effective rank: {quality_report.get('direction_effective_rank', 'n/a')}",
            f"- Theta top-2 bin mass: {quality_report.get('theta_top2_bin_mass', 'n/a')}",
            f"- Neighbor coherence lift: {quality_report.get('neighbor_coherence_lift', 'n/a')}",
        ]
    if benchmark_report:
        lines += [
            f"- Benchmark queries: {benchmark_report.get('queries', 'n/a')}",
            f"- Benchmark by kind: {benchmark_report.get('by_kind', {})}",
        ]
    if ablation_report:
        rr = ablation_report.get("rare_recall_at_5", {})
        cp = ablation_report.get("condition_precision", {})
        lines += [
            f"- Rare Recall@5 mean diff: {rr.get('observed_mean_diff', 'n/a')}",
            f"- Paired bootstrap p-value: {rr.get('p_value', 'n/a')}",
            f"- Condition Precision diff: {cp.get('mean_diff', 'n/a')}",
            f"- Poincare/Flat-Twin top5 overlap: {ablation_report.get('ranking_diagnostics', {}).get('poincare_flat_twin_top5_overlap_mean', 'n/a')}",
        ]

    lines += [
        "",
        "## Guardrail",
        "",
        "If Poincare only beats lexical retrieval but does not beat same-coordinate Flat-Twin, report agentic memory gains only. Do not claim the hyperbolic geometry itself is responsible.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a hyperbolic claim readiness report.")
    parser.add_argument("--builder-report", type=Path, default=DEFAULT_BUILDER_REPORT)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--benchmark-validation", type=Path, default=DEFAULT_BENCH_VALIDATION)
    parser.add_argument("--ablation-report", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    text = render(
        builder_report=load(args.builder_report),
        quality_report=load(args.quality_report),
        benchmark_report=load(args.benchmark_validation),
        ablation_report=load(args.ablation_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
