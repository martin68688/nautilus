#!/usr/bin/env python3
"""Evaluate protocol-controlled SOP distillation policies.

This diagnostic has three parts:

1. Measure the current RunForest's source composition and authority coverage.
2. Run a deterministic mixed-value expressivity test over SOP-level and
   claim-level policies.
3. Re-run the frozen retrospective granularity/debug benchmark on filtered
   graph views.  This last track is optional because it is more expensive.

The script does not mutate the production graph or index.  It is a mechanism
diagnostic, not evidence of downstream online task improvement.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer  # noqa: E402
from core import GRAPH, INDEX, REPORTS  # noqa: E402
from run_causal_granularity_benchmark_v2 import (  # noqa: E402
    evaluate_debug,
    evaluate_granularity,
)


SCHEMA = "sop_protocol_policy_evaluation_v1"
DEFAULT_REPORT = REPORTS / "sop_protocol_policy_evaluation_v1.json"
ALLOW_OUTCOMES = {"allow", "allow_with_warning"}

REQUEST_OPERATIONS = (
    "inspect",
    "debug_hypothesis",
    "repair_seed",
    "rank",
    "select",
    "promote",
    "code_seed",
)
PROTOCOLS = ("v2", "v3")
HIGH_RISK_OPERATIONS = {"rank", "select", "promote", "code_seed"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_layer(graph_path: Path, index_path: Path) -> StageAwareHybridMemoryLayer:
    return StageAwareHybridMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        source_name="sop_protocol_policy_eval",
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        retrieval_control="stage_hybrid",
        enable_agentic=False,
        top_k=20,
        max_chars=0,
    )


def distillation_edges(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        edge
        for edge in graph.get("edges", [])
        if str(edge.get("kind") or edge.get("type") or "") == "distills_to"
    ]


def graph_policy_statistics(
    graph: dict[str, Any],
    layer: StageAwareHybridMemoryLayer,
) -> dict[str, Any]:
    sops = [node for node in graph.get("nodes", []) if node.get("type") == "SOP"]
    runnodes = [node for node in graph.get("nodes", []) if node.get("type") == "RunNode"]
    edges = distillation_edges(graph)

    support: dict[str, list[dict[str, Any]]] = defaultdict(list)
    edge_proxy_counts: Counter[str] = Counter()
    edge_authority_counts: Counter[str] = Counter()
    authority_reasons: Counter[str] = Counter()
    for edge in edges:
        transition_id = str(edge.get("src") or "")
        proxy_allowed, proxy_reason = layer._positive_transition(transition_id)
        outcome = str(edge.get("authority_outcome") or "missing")
        edge_proxy_counts["positive_proxy" if proxy_allowed else "nonpositive_proxy"] += 1
        edge_authority_counts[outcome] += 1
        authority_reasons.update(str(reason) for reason in (edge.get("authority_reasons") or []))
        support[str(edge.get("dst") or "")].append(
            {
                "proxy_allowed": proxy_allowed,
                "proxy_reason": proxy_reason,
                "authority_allowed": outcome in ALLOW_OUTCOMES,
                "authority_outcome": outcome,
            }
        )

    composition: Counter[str] = Counter()
    strict_composition: Counter[str] = Counter()
    for sop in sops:
        rows = support.get(str(sop.get("id") or ""), [])
        proxy = [bool(row["proxy_allowed"]) for row in rows]
        strict = [bool(row["authority_allowed"]) for row in rows]
        if not rows:
            composition["no_distillation_source"] += 1
            strict_composition["no_distillation_source"] += 1
        elif all(proxy):
            composition["positive_only"] += 1
        elif any(proxy):
            composition["mixed_positive_and_nonpositive"] += 1
        else:
            composition["nonpositive_only"] += 1
        if rows:
            if all(strict):
                strict_composition["allow_only"] += 1
            elif any(strict):
                strict_composition["mixed_allow_and_nonallow"] += 1
            else:
                strict_composition["nonallow_only"] += 1

    sop_field_coverage = {
        "protocol_ref": sum(bool(node.get("protocol_ref")) for node in sops),
        "claim_refs": sum(bool(node.get("claim_refs")) for node in sops),
        "receipt_refs": sum(bool(node.get("receipt_refs")) for node in sops),
        "operation_scope": sum(bool(node.get("operation_scope") or node.get("authority_scope")) for node in sops),
        "protocol_agnostic_true": sum(node.get("protocol_agnostic") is True for node in sops),
        "clause_lineage": sum(bool(node.get("clause_lineage")) for node in sops),
        "derived_publication_allow": sum(
            (node.get("derived_publication_authority") or {}).get("outcome") == "allow"
            for node in sops
        ),
    }
    runnode_field_coverage = {
        "leakage_audit": sum(bool(node.get("leakage_audit")) for node in runnodes),
        "protocol_ref": sum(bool(node.get("protocol_ref")) for node in runnodes),
        "claim_refs": sum(bool(node.get("claim_refs")) for node in runnodes),
        "receipt_refs": sum(bool(node.get("receipt_refs")) for node in runnodes),
        "authority_decision_refs": sum(bool(node.get("authority_decision_refs")) for node in runnodes),
    }

    positive_edges = edge_proxy_counts["positive_proxy"]
    strict_edges = sum(
        count for outcome, count in edge_authority_counts.items() if outcome in ALLOW_OUTCOMES
    )
    return {
        "graph_counts": {
            "sop_count": len(sops),
            "runnode_count": len(runnodes),
            "distillation_edge_count": len(edges),
        },
        "sop_field_coverage": sop_field_coverage,
        "runnode_field_coverage": runnode_field_coverage,
        "edge_positive_proxy_counts": dict(sorted(edge_proxy_counts.items())),
        "edge_authority_outcomes": dict(sorted(edge_authority_counts.items())),
        "edge_authority_reasons": dict(sorted(authority_reasons.items())),
        "sop_source_composition_by_node_clean_proxy": dict(sorted(composition.items())),
        "sop_source_composition_by_strict_edge_authority": dict(sorted(strict_composition.items())),
        "policy_capacity": {
            "full_distillation_audit_edges": len(edges),
            "clean_only_positive_edges": positive_edges,
            "strict_current_authorized_edges": strict_edges,
            "dual_view_audit_edges": len(edges),
            "dual_view_positive_edges_using_current_proxy": positive_edges,
            "dual_view_positive_edges_using_strict_current_authority": strict_edges,
        },
    }


def _scope(*operations: str, protocols: tuple[str, ...] = PROTOCOLS) -> dict[str, list[str]]:
    return {protocol: list(operations) for protocol in protocols}


def synthetic_claims() -> list[dict[str, Any]]:
    """Hand-specified mixed-value cases from the protocol threat model."""
    diagnostic = ("inspect", "debug_hypothesis", "repair_seed", "distill_diagnostic")
    universal_procedure = (
        "inspect", "debug_hypothesis", "repair_seed", "promote", "code_seed",
        "distill_positive", "distill_diagnostic",
    )
    return [
        {
            "experience": "oof_fix_plus_test_label_selection",
            "claim": "oof_index_alignment_fix",
            "scope": _scope(*diagnostic),
            "protocol_agnostic": True,
        },
        {
            "experience": "oof_fix_plus_test_label_selection",
            "claim": "score_0_92_superiority",
            "scope": _scope("inspect", "distill_diagnostic"),
            "protocol_agnostic": False,
        },
        {
            "experience": "evaluator_tamper_plus_path_fix",
            "claim": "working_path_and_api_fix",
            "scope": _scope("inspect", "debug_hypothesis", "repair_seed", "code_seed", "distill_positive", "distill_diagnostic"),
            "protocol_agnostic": True,
        },
        {
            "experience": "evaluator_tamper_plus_path_fix",
            "claim": "tampered_metric_0_99",
            "scope": _scope("inspect", "debug_hypothesis", "distill_diagnostic"),
            "protocol_agnostic": False,
        },
        {
            "experience": "best_of_20_seed_selection",
            "claim": "single_seed_score_0_89",
            "scope": _scope("inspect", "debug_hypothesis", "distill_diagnostic"),
            "protocol_agnostic": False,
        },
        {
            "experience": "best_of_20_seed_selection",
            "claim": "aggregate_all_declared_seeds",
            "scope": _scope(*universal_procedure),
            "protocol_agnostic": True,
        },
        {
            "experience": "protocol_v2_to_v3_drift",
            "claim": "architecture_hypothesis",
            "scope": _scope(*universal_procedure),
            "protocol_agnostic": True,
        },
        {
            "experience": "protocol_v2_to_v3_drift",
            "claim": "v2_score_comparison",
            "scope": {
                "v2": ["inspect", "rank", "select", "promote", "distill_positive", "distill_diagnostic"],
                "v3": ["inspect", "debug_hypothesis", "repair_seed", "distill_diagnostic"],
            },
            "protocol_agnostic": False,
        },
        {
            "experience": "method_changing_fake_replay",
            "claim": "old_method_debug_history",
            "scope": _scope(*diagnostic),
            "protocol_agnostic": False,
        },
        {
            "experience": "method_changing_fake_replay",
            "claim": "successor_method_v3_score",
            "scope": {
                "v2": ["inspect", "distill_diagnostic"],
                "v3": ["inspect", "rank", "select", "promote", "code_seed", "distill_positive", "distill_diagnostic"],
            },
            "protocol_agnostic": False,
        },
        {
            "experience": "derived_memory_laundering",
            "claim": "leaked_score_0_95",
            "scope": _scope("inspect", "debug_hypothesis", "distill_diagnostic"),
            "protocol_agnostic": False,
        },
        {
            "experience": "derived_memory_laundering",
            "claim": "never_promote_leakage_derived_score",
            "scope": _scope(*universal_procedure),
            "protocol_agnostic": True,
        },
    ]


def authorized(claim: dict[str, Any], protocol: str, operation: str) -> bool:
    return operation in set((claim.get("scope") or {}).get(protocol, []))


def _policy_views(
    policy: str,
    claims: list[dict[str, Any]],
    protocol: str,
    operation: str,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Return activated, prompt-exposed, positive-stored, audit-stored claim IDs."""
    by_experience: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        by_experience[str(claim["experience"])].append(claim)
    all_ids = {str(claim["claim"]) for claim in claims}

    if policy == "clean_only_run":
        # Global-validity interpretation: a mixed run never reaches SOP memory.
        clean_experiences = {
            experience
            for experience, rows in by_experience.items()
            if all(
                all(authorized(row, p, op) for p in PROTOCOLS for op in HIGH_RISK_OPERATIONS)
                for row in rows
            )
        }
        stored = {
            str(row["claim"])
            for experience in clean_experiences
            for row in by_experience[experience]
        }
        return stored, stored, stored, stored

    if policy == "clean_only_claim_scoped":
        # Only claims allowed into the positive SOP are retained there.  Full
        # source runs remain elsewhere, but no diagnostic SOP view is created.
        stored = {
            str(row["claim"])
            for row in claims
            if authorized(row, protocol, "distill_positive")
        }
        activated = {
            str(row["claim"])
            for row in claims
            if str(row["claim"]) in stored and authorized(row, protocol, operation)
        }
        return activated, activated, stored, stored

    if policy in {"full_sop_union", "full_sop_intersection"}:
        activated: set[str] = set()
        exposed: set[str] = set()
        for rows in by_experience.values():
            votes = [authorized(row, protocol, operation) for row in rows]
            permit = any(votes) if policy == "full_sop_union" else all(votes)
            if permit:
                ids = {str(row["claim"]) for row in rows}
                activated.update(ids)
                exposed.update(ids)
        return activated, exposed, all_ids, all_ids

    if policy == "full_claim_tags_post_prompt":
        activated = {
            str(row["claim"])
            for row in claims
            if authorized(row, protocol, operation)
        }
        exposed: set[str] = set()
        for rows in by_experience.values():
            if any(authorized(row, protocol, operation) for row in rows):
                exposed.update(str(row["claim"]) for row in rows)
        return activated, exposed, all_ids, all_ids

    if policy == "claim_dual_view_pre_prompt":
        activated = {
            str(row["claim"])
            for row in claims
            if authorized(row, protocol, operation)
        }
        positive = {
            str(row["claim"])
            for row in claims
            if authorized(row, protocol, "distill_positive")
        }
        audit = {
            str(row["claim"])
            for row in claims
            if authorized(row, protocol, "distill_diagnostic")
        }
        return activated, activated, positive, audit

    raise ValueError(f"unknown synthetic policy: {policy}")


def mixed_value_policy_test() -> dict[str, Any]:
    claims = synthetic_claims()
    policies = (
        "clean_only_run",
        "clean_only_claim_scoped",
        "full_sop_union",
        "full_sop_intersection",
        "full_claim_tags_post_prompt",
        "claim_dual_view_pre_prompt",
    )
    results: dict[str, Any] = {}
    claim_by_id = {str(claim["claim"]): claim for claim in claims}
    all_ids = set(claim_by_id)
    total_pairs = len(claims) * len(PROTOCOLS) * len(REQUEST_OPERATIONS)
    valid_pairs = sum(
        authorized(claim, protocol, operation)
        for claim in claims
        for protocol in PROTOCOLS
        for operation in REQUEST_OPERATIONS
    )
    invalid_pairs = total_pairs - valid_pairs

    for policy in policies:
        tp = fp = exposed_invalid = 0
        stored_positive_union: set[str] = set()
        stored_audit_union: set[str] = set()
        by_operation: dict[str, Counter[str]] = defaultdict(Counter)
        for protocol in PROTOCOLS:
            for operation in REQUEST_OPERATIONS:
                activated, exposed, positive_stored, audit_stored = _policy_views(
                    policy, claims, protocol, operation
                )
                stored_positive_union.update(positive_stored)
                stored_audit_union.update(audit_stored)
                for claim_id in all_ids:
                    truth = authorized(claim_by_id[claim_id], protocol, operation)
                    predicted = claim_id in activated
                    prompt_seen = claim_id in exposed
                    if truth and predicted:
                        tp += 1
                        by_operation[operation]["tp"] += 1
                    elif not truth and predicted:
                        fp += 1
                        by_operation[operation]["fp"] += 1
                    if not truth and prompt_seen:
                        exposed_invalid += 1
                        by_operation[operation]["prompt_exposed_invalid"] += 1
        activation_count = tp + fp
        results[policy] = {
            "valid_knowledge_retention": tp / max(1, valid_pairs),
            "unauthorized_activation_rate": fp / max(1, invalid_pairs),
            "invalid_fraction_among_activations": fp / max(1, activation_count),
            "unauthorized_prompt_exposure_rate": exposed_invalid / max(1, invalid_pairs),
            "positive_sop_claim_retention": len(stored_positive_union) / len(claims),
            "audit_sop_claim_retention": len(stored_audit_union) / len(claims),
            "true_positive_pairs": tp,
            "false_positive_pairs": fp,
            "valid_pair_count": valid_pairs,
            "invalid_pair_count": invalid_pairs,
            "by_operation": {
                operation: dict(sorted(counts.items()))
                for operation, counts in sorted(by_operation.items())
            },
        }
    return {
        "test_kind": "deterministic_expressivity_sanity_check",
        "claim_count": len(claims),
        "mixed_experience_count": len({claim["experience"] for claim in claims}),
        "protocols": list(PROTOCOLS),
        "request_operations": list(REQUEST_OPERATIONS),
        "request_pair_count": total_pairs,
        "claims": claims,
        "results": results,
        "interpretation_limit": (
            "This proves representational safety/retention differences under the specified ground truth; "
            "it does not measure whether an LLM follows tags or improves an online MLE task."
        ),
    }


def filtered_graph(graph: dict[str, Any], layer: StageAwareHybridMemoryLayer, policy: str) -> dict[str, Any]:
    if policy == "full_current_metadata_only":
        return graph
    kept: list[dict[str, Any]] = []
    allowed_sops_by_transition: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if str(edge.get("kind") or edge.get("type") or "") != "distills_to":
            kept.append(edge)
            continue
        if policy == "clean_only_node_proxy":
            allowed, _reason = layer._positive_transition(str(edge.get("src") or ""))
        elif policy == "strict_current_edge_authority":
            allowed = str(edge.get("authority_outcome") or "") in ALLOW_OUTCOMES
        else:
            raise ValueError(f"unknown graph view policy: {policy}")
        if allowed:
            kept.append(edge)
            allowed_sops_by_transition[str(edge.get("src") or "")].add(str(edge.get("dst") or ""))

    # Transition nodes duplicate the same relation in attached_sop_ids and
    # attachment_quality.  A policy that filters only graph edges is bypassed
    # by the causal Debug path, which reads these embedded fields directly.
    # Keep both representations synchronized in the counterfactual view.
    nodes: list[dict[str, Any]] = []
    for node in graph.get("nodes", []):
        if node.get("type") != "Transition":
            nodes.append(node)
            continue
        transition_id = str(node.get("id") or "")
        allowed_sops = allowed_sops_by_transition.get(transition_id, set())
        nodes.append(
            {
                **node,
                "attached_sop_ids": [
                    str(sop_id)
                    for sop_id in (node.get("attached_sop_ids") or [])
                    if str(sop_id) in allowed_sops
                ],
                "attachment_quality": [
                    item
                    for item in (node.get("attachment_quality") or [])
                    if str(item.get("sop_id") or "") in allowed_sops
                ],
            }
        )
    return {**graph, "nodes": nodes, "edges": kept}


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


def retrospective_views(
    graph: dict[str, Any],
    layer: StageAwareHybridMemoryLayer,
    index_path: Path,
) -> dict[str, Any]:
    policies = (
        "full_current_metadata_only",
        "clean_only_node_proxy",
        "strict_current_edge_authority",
    )
    output: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="sop-protocol-policy-") as temp_dir:
        temp_root = Path(temp_dir)
        for policy in policies:
            view = filtered_graph(graph, layer, policy)
            view_path = temp_root / f"{policy}.json"
            write_json(view_path, view)
            _granularity_receipts, granularity = evaluate_granularity(view_path, index_path)
            _debug_receipts, debug = evaluate_debug(view_path, index_path)
            output[policy] = {
                "distillation_edge_count": len(distillation_edges(view)),
                "stage_granularity": {
                    name: granularity[name]
                    for name in ("sop_only", "stage_hybrid_dynamic")
                },
                "causal_debug_transfer": {
                    name: debug[name]
                    for name in ("legacy_success_tree", "causal_tree_dynamic")
                },
            }
            gc.collect()
    return _finite(output)


def evaluate(
    graph_path: Path,
    index_path: Path,
    *,
    include_retrospective: bool,
) -> dict[str, Any]:
    graph = read_json(graph_path)
    layer = make_layer(graph_path, index_path)
    result = {
        "schema": SCHEMA,
        "source_graph": str(graph_path),
        "source_index": str(index_path),
        "graph_policy_statistics": graph_policy_statistics(graph, layer),
        "mixed_value_policy_test": mixed_value_policy_test(),
        "retrospective_benchmark": (
            retrospective_views(graph, layer, index_path)
            if include_retrospective
            else {"status": "skipped"}
        ),
        "limitations": [
            "RunNode clean status is a node-level leakage proxy, not Claim x Operation x Stage x Protocol authority.",
            "All current distillation edges lack actuation receipts, so strict edge authority has zero positive edges.",
            "The retrospective benchmark measures retrieval/routing, not generated-code adoption or online task score.",
            "The mixed-value test is a deterministic expressivity check and requires online LLM validation.",
        ],
    }
    return _finite(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=GRAPH)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-retrospective", action="store_true")
    args = parser.parse_args()
    result = evaluate(
        args.graph.resolve(),
        args.index.resolve(),
        include_retrospective=not args.skip_retrospective,
    )
    write_json(args.report.resolve(), result)
    print(json.dumps({
        "report": str(args.report.resolve()),
        "graph_policy_statistics": result["graph_policy_statistics"],
        "mixed_value_results": result["mixed_value_policy_test"]["results"],
        "retrospective_policies": list(result["retrospective_benchmark"]),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
