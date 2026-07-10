"""Build an offline benchmark for Agentic Hyperbolic SOP Memory.

The default profile is a harder deterministic benchmark. The earlier 40-query
seed profile is still available with ``--profile seed`` for smoke tests, but it
is too direct for geometry comparisons: full condition/action strings make the
gold SOP rank first for every agentic scorer. The hard profile uses more queries
and lower-specificity probes so ranking quality can differ across retrievers.
It remains auto-seeded and needs human audit before final paper tables.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_BENCH = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "hyperbolic_sop_benchmark.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "hyperbolic_sop_gold.jsonl"
DEFAULT_HARD_PER_KIND = 40
DEFAULT_EDGE_VARIANTS_PER_SOP = 3
DEFAULT_EDGE_MIN_DISTRACTORS = 20
EDGE_QUERY_KINDS = ("edge_api_debug", "edge_shape_path", "edge_version_mismatch", "edge_minimal_failure")

TASKS = [
    "spooky-author-identification",
    "leaf-classification",
    "aerial-cactus-identification",
    "denoising-dirty-documents",
    "new-york-city-taxi-fare-prediction",
]

STOP = {
    "when", "with", "that", "this", "into", "from", "using", "use", "uses", "for", "and",
    "the", "are", "was", "were", "has", "have", "data", "model", "training", "task",
    "avoid", "ensure", "correct", "prevent", "prevents", "check", "use", "set",
    "working", "where", "current", "context", "retrieve", "applicable", "sop", "method",
    "implementation", "solution", "classification", "prediction", "identification",
    "documents", "dirty", "aerial", "cactus", "spooky", "author", "leaf", "taxi",
    "fare", "city", "york", "new", "likely", "relevant",
}

CODE_TOKEN_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_]*|[A-Za-z_]*\d+[A-Za-z0-9_]*|[a-z]+_[a-z0-9_]+|[A-Za-z]+\.[A-Za-z0-9_.]+)\b"
)


def load_graph(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    sops = [n for n in graph.get("nodes", []) if n.get("type") == "SOP"]
    failures: dict[str, list[str]] = collections.defaultdict(list)
    for edge in graph.get("edges", []):
        if edge.get("kind") != "prevents":
            continue
        src, dst = edge.get("src"), edge.get("dst")
        if src in nodes and dst in nodes and nodes[dst].get("type") == "FailureMode":
            failures[src].append(str(nodes[dst].get("title", "")))
    return nodes, sops, failures


def tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", (text or "").lower())
        if t not in STOP
    }


def title_overlap(query: str, title: str) -> float:
    title_tokens = tokens(title)
    if not title_tokens:
        return 0.0
    return len(tokens(query) & title_tokens) / len(title_tokens)


def short_condition(condition: str, *, max_words: int = 18) -> str:
    words = re.findall(r"[A-Za-z0-9_+./=-]+|[^\sA-Za-z0-9_+./=-]+", condition or "")
    clipped = " ".join(words[:max_words]).strip()
    return clipped or "the current node's narrow implementation context"


def abstract_clue(*, node: dict[str, Any], source_text: str, max_terms: int = 2) -> str:
    """Create a low-leakage clue from a SOP while hiding title/code tokens.

    This intentionally removes exact title tokens and obvious API/model tokens.
    The resulting query is still linked to the SOP, but much less like a string
    lookup against the full condition/action card.
    """
    title_tokens = tokens(str(node.get("title", "")))
    source_text = CODE_TOKEN_RE.sub(" ", source_text or "")
    kept: list[str] = []
    for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", source_text.lower()):
        if tok in STOP or tok in title_tokens or tok in kept:
            continue
        kept.append(tok)
        if len(kept) >= max_terms:
            break
    return " ".join(kept) or "validation risk"


def radius_hint(kind: str) -> str:
    if kind in EDGE_QUERY_KINDS:
        return "edge"
    if kind in {"rare_condition", "debug_failure", "conflict_risk", "rare_partial_clue", "abstract_failure", "minimal_context"}:
        return "edge,middle"
    if kind in {"method_set", "hard_method_set"}:
        return "core,middle"
    return ""


def leakage_level(overlap: float) -> str:
    if overlap >= 0.55:
        return "high"
    if overlap >= 0.25:
        return "medium"
    return "low"


def condition_signature(node: dict[str, Any], failure_labels: list[str]) -> tuple[str, ...]:
    text = " ".join([str(node.get("condition", "")), " ".join(failure_labels)])
    return tuple(sorted(tokens(text))[:8])


def distractor_sops(
    *,
    node: dict[str, Any],
    all_sops: list[dict[str, Any]],
    failure_by_sop: dict[str, list[str]],
    query_text: str,
    limit: int = 5,
) -> tuple[list[str], int]:
    """Return same-task near misses for benchmark difficulty diagnostics."""
    gold_id = str(node.get("id", ""))
    task = node.get("category")
    q_tokens = tokens(query_text)
    scored: list[tuple[float, str]] = []
    same_task_count = 0
    for candidate in all_sops:
        cid = str(candidate.get("id", ""))
        if cid == gold_id or candidate.get("category") != task:
            continue
        same_task_count += 1
        c_text = " ".join(
            [
                str(candidate.get("title", "")),
                str(candidate.get("principle", "")),
                str(candidate.get("condition", "")),
                " ".join(failure_by_sop.get(cid, [])),
            ]
        )
        c_tokens = tokens(c_text)
        if not c_tokens:
            continue
        lexical = len(q_tokens & c_tokens) / max(1, len(q_tokens | c_tokens))
        same_band = 0.05 if candidate.get("radius_band") == node.get("radius_band") else 0.0
        same_failure = 0.10 if set(failure_by_sop.get(cid, [])) & set(failure_by_sop.get(gold_id, [])) else 0.0
        score = lexical + same_band + same_failure
        if score > 0:
            scored.append((score, cid))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [cid for _score, cid in scored[:limit]], same_task_count


def edge_reason(node: dict[str, Any], failures: list[str]) -> str:
    text = " ".join([
        str(node.get("title", "")),
        str(node.get("principle", "")),
        str(node.get("condition", "")),
        " ".join(failures),
    ]).lower()
    if any(x in text for x in ("api", "argument", "parameter", "attribute", "keyword", "import")):
        return "api"
    if any(x in text for x in ("shape", "dimension", "dtype", "tensor", "column", "csv")):
        return "shape"
    if any(x in text for x in ("path", "file", "checkpoint", "submission", "sample_submission")):
        return "path"
    if any(x in text for x in ("version", "deprecated", "base_estimator", "estimator", "load")):
        return "version"
    if any(x in text for x in ("syntax", "undefined", "duplicate", "merge", "traceback", "error")):
        return "debug"
    return "rare_failure"


def round_robin(nodes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for node in nodes:
        by_task[str(node.get("category"))].append(node)
    for group in by_task.values():
        group.sort(key=lambda n: (n.get("radius_band") != "edge", n.get("title", ""), n.get("id", "")))
    picked: list[dict[str, Any]] = []
    while len(picked) < limit:
        changed = False
        for task in TASKS:
            if by_task.get(task):
                picked.append(by_task[task].pop(0))
                changed = True
                if len(picked) >= limit:
                    break
        if not changed:
            break
    return picked


def query_row(
    query_id: str,
    node: dict[str, Any],
    kind: str,
    failures: list[str],
    *,
    split: str,
    all_sops: list[dict[str, Any]],
    failure_by_sop: dict[str, list[str]],
    style: str = "seed",
) -> dict[str, Any]:
    condition = str(node.get("condition", "")).strip()
    failure = failures[:2] or ["method failure"]
    task = str(node.get("category", ""))
    condition_view = short_condition(condition)
    failure_view = ", ".join(failure)
    stage = {
        "rare_condition": "improve",
        "debug_failure": "debug",
        "conflict_risk": "debug",
        "method_set": "draft",
        "rare_partial_clue": "improve",
        "abstract_failure": "debug",
        "minimal_context": "improve",
        "hard_method_set": "draft",
        "edge_api_debug": "debug",
        "edge_shape_path": "debug",
        "edge_version_mismatch": "debug",
        "edge_minimal_failure": "debug",
    }[kind]
    query_specificity = "high"
    if style == "partial_clue":
        clue = abstract_clue(
            node=node,
            source_text=" ".join([condition, str(node.get("principle", "")), str(node.get("action", ""))]),
            max_terms=2,
        )
        context = (
            f"{task}: retrieve a SOP from a short anonymized clue rather than an exact title. "
            f"Clue: {clue}. Symptom family: {failure_view}."
        )
        condition_items = [clue]
        query_specificity = "medium_low"
    elif style == "abstract_failure":
        context = (
            f"{task}: a branch is regressing, but the report is abstract and omits the exact library or method name. "
            f"Use the symptom family {failure_view} and the task area to retrieve the most plausible SOP."
        )
        condition_items = []
        query_specificity = "low"
    elif style == "minimal_context":
        band = str(node.get("radius_band", "relevant") or "relevant")
        context = (
            f"{task}: before the next {stage} step, retrieve a {band} SOP. "
            f"Only the broad symptom family is visible: {failure_view}."
        )
        condition_items = []
        query_specificity = "low"
    elif style == "method_partial":
        clue = abstract_clue(
            node=node,
            source_text=" ".join([str(node.get("principle", "")), str(node.get("action", "")), condition]),
            max_terms=2,
        )
        context = (
            f"{task}: assemble a method-set memory pack from a compact procedural clue. "
            f"Clue: {clue}. Prefer the SOP that best fits the hidden implementation choice."
        )
        condition_items = [clue]
        query_specificity = "medium_low"
    elif style == "edge_api_debug":
        clue = abstract_clue(node=node, source_text=" ".join([condition, str(node.get("principle", ""))]), max_terms=1)
        context = (
            f"{task}: a generated node hits a narrow API/debug failure after code changes. "
            f"The exact library call and SOP title are hidden. Weak clue: {clue}. Symptom family: {failure_view}."
        )
        condition_items = [clue]
        query_specificity = "low"
    elif style == "edge_shape_path":
        clue = abstract_clue(node=node, source_text=condition, max_terms=1)
        context = (
            f"{task}: retrieve an edge SOP for a concrete interface issue. "
            f"Only a weak clue is visible: {clue}. Think shape, path, column, file, checkpoint, or submission mismatch when applicable."
        )
        condition_items = [clue]
        query_specificity = "low"
    elif style == "edge_version_mismatch":
        clue = abstract_clue(node=node, source_text=" ".join([str(node.get("principle", "")), condition]), max_terms=1)
        context = (
            f"{task}: a low-frequency implementation detail changed behavior across library/API/checkpoint versions. "
            f"Exact API tokens are masked; weak clue: {clue}. Symptom family: {failure_view}."
        )
        condition_items = [clue]
        query_specificity = "low"
    elif style == "edge_minimal_failure":
        context = (
            f"{task}: minimal context edge lookup. The branch needs a narrow SOP near the map boundary. "
            f"Only broad symptom family is visible: {failure_view}."
        )
        condition_items = []
        query_specificity = "low"
    elif kind == "rare_condition":
        context = (
            f"{task}: a low-support, condition-specific implementation detail is likely relevant. "
            f"Current context: {condition_view}. Symptom family: {failure_view}. Retrieve the most applicable SOP."
        )
        condition_items = [condition_view] if condition_view else []
    elif kind == "debug_failure":
        context = (
            f"{task}: a generated node is failing or regressing. Symptom family: {failure_view}. "
            f"Local context: {condition_view}. Retrieve the SOP that would prevent the failure."
        )
        condition_items = [condition_view] if condition_view else []
    elif kind == "conflict_risk":
        context = (
            f"{task}: before adopting the next implementation branch, inspect risk warnings and condition branches. "
            f"Context: {condition_view}. Main risk family: {failure_view}."
        )
        condition_items = [condition_view] if condition_view else []
    else:
        context = (
            f"{task}: build a coherent method set for this situation. "
            f"Context: {condition_view}. Failure/risk family to manage: {failure_view}."
        )
        condition_items = [condition_view] if condition_view else []
    leakage = title_overlap(" ".join([context, " ".join(condition_items), failure_view]), str(node.get("title", "")))
    distractors, distractor_count = distractor_sops(
        node=node,
        all_sops=all_sops,
        failure_by_sop=failure_by_sop,
        query_text=" ".join([context, " ".join(condition_items), failure_view]),
    )
    return {
        "query_id": query_id,
        "task_type": task,
        "stage": stage,
        "context": context,
        "condition": condition_items,
        "failure_mode": failure,
        "source_trace": "",
        "query_kind": kind,
        "split": split,
        "radius_band_hint": radius_hint(kind),
        "title_token_overlap": round(leakage, 6),
        "title_leakage_level": leakage_level(leakage),
        "gold_title_hidden": True,
        "gold_radius_band": node.get("radius_band", ""),
        "gold_support": int(node.get("n_use", 0) or 0),
        "gold_p_hat": float(node.get("p_hat", 0.0) or 0.0),
        "distractor_sops": distractors,
        "distractor_count": distractor_count,
        "query_style": style,
        "query_specificity": query_specificity,
        "benchmark_source": "auto_seeded_from_certified_hyper_graph",
    }


def gold_row(
    query_id: str,
    node: dict[str, Any],
    relevance: str,
    is_rare: bool,
    rarity_count: int,
    *,
    edge_gold: bool = False,
    edge_reason_value: str = "",
) -> dict[str, Any]:
    item = {
        "sop_id": node["id"],
        "relevance": relevance,
        "condition_match": True,
        "is_rare": bool(is_rare),
        "rarity_basis": "condition_failure_signature",
        "rarity_count": int(rarity_count),
        "rationale": f"Auto-seeded from certified SOP '{node.get('title', node['id'])}'. Requires human audit before final paper tables.",
    }
    if edge_gold:
        item.update({
            "gold_radius_band": "edge",
            "edge_gold": True,
            "edge_reason": edge_reason_value or "rare_failure",
        })
    return {
        "query_id": query_id,
        "gold_sops": [item],
    }


def safe_round_robin(primary: list[dict[str, Any]], fallback: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    picked = round_robin(primary, limit)
    if len(picked) >= limit:
        return picked[:limit]
    seen = {n["id"] for n in picked}
    extra = [n for n in fallback if n["id"] not in seen]
    return (picked + round_robin(extra, limit - len(picked)))[:limit]


def build(
    graph_path: Path,
    benchmark_path: Path,
    gold_path: Path,
    *,
    profile: str = "hard",
    hard_per_kind: int = DEFAULT_HARD_PER_KIND,
    edge_variants_per_sop: int = DEFAULT_EDGE_VARIANTS_PER_SOP,
    edge_min_distractors: int = DEFAULT_EDGE_MIN_DISTRACTORS,
) -> dict[str, Any]:
    _nodes, sops, failure_by_sop = load_graph(graph_path)
    signatures = {n["id"]: condition_signature(n, failure_by_sop.get(n["id"], [])) for n in sops}
    counts = collections.Counter(sig for sig in signatures.values() if sig)

    task_sops = [n for n in sops if n.get("category") in TASKS]
    rare_candidates = [
        n for n in task_sops
        if n.get("radius_band") == "edge" and counts.get(signatures.get(n["id"], ()), 999) <= 2
    ]
    debug_candidates = [
        n for n in task_sops
        if any(f for f in failure_by_sop.get(n["id"], []) if f not in {"general execution failure"})
    ]
    conflict_candidates = [
        n for n in task_sops
        if any(word in (n.get("title", "") + " " + n.get("principle", "")).lower() for word in ("avoid", "disable", "prevent", "check", "correct"))
    ]
    method_candidates = [
        n for n in task_sops
        if n.get("radius_band") in {"core", "middle"} and n not in conflict_candidates
    ]

    if profile == "seed":
        groups = [
            ("rare_condition", safe_round_robin(rare_candidates, task_sops, 10), "required", "seed"),
            ("debug_failure", safe_round_robin(debug_candidates, task_sops, 10), "required", "seed"),
            ("conflict_risk", safe_round_robin(conflict_candidates, task_sops, 10), "risk_warning", "seed"),
            ("method_set", safe_round_robin(method_candidates, task_sops, 10), "required", "seed"),
        ]
    elif profile == "hard":
        per_kind = max(10, int(hard_per_kind or DEFAULT_HARD_PER_KIND))
        edge_or_rare = [n for n in task_sops if n.get("radius_band") == "edge"]
        groups = [
            ("rare_partial_clue", safe_round_robin(rare_candidates or edge_or_rare, task_sops, per_kind), "required", "partial_clue"),
            ("abstract_failure", safe_round_robin(debug_candidates, task_sops, per_kind), "required", "abstract_failure"),
            ("minimal_context", safe_round_robin(edge_or_rare or task_sops, task_sops, per_kind), "required", "minimal_context"),
            ("hard_method_set", safe_round_robin(method_candidates, task_sops, per_kind), "required", "method_partial"),
        ]
    elif profile == "edge":
        task_counts = collections.Counter(n.get("category") for n in task_sops)
        edge_sops = [
            n for n in task_sops
            if n.get("radius_band") == "edge"
            and task_counts.get(n.get("category"), 0) - 1 >= int(edge_min_distractors)
        ]
        if not edge_sops:
            raise ValueError("Edge benchmark requires at least one edge-band SOP with enough same-task distractors")
        edge_sops = round_robin(edge_sops, len(edge_sops))
        groups = []
        for variant_idx in range(max(1, int(edge_variants_per_sop or DEFAULT_EDGE_VARIANTS_PER_SOP))):
            kind = EDGE_QUERY_KINDS[variant_idx % len(EDGE_QUERY_KINDS)]
            groups.append((kind, edge_sops, "required", kind))
    else:
        raise ValueError(f"Unsupported benchmark profile: {profile}")
    rows = []
    gold = []
    split_by_sop: dict[str, str] = {}
    if profile == "edge":
        edge_ids = sorted({node["id"] for _kind, selected, _rel, _style in groups for node in selected})
        cutoff = max(1, len(edge_ids) // 2)
        split_by_sop = {sid: ("dev" if i < cutoff else "test") for i, sid in enumerate(edge_ids)}
    for kind, selected, relevance, style in groups:
        if not selected:
            raise ValueError(f"Not enough {kind} candidates: {len(selected)}")
        split_cutoff = max(1, len(selected) // 2)
        for idx, node in enumerate(selected, 1):
            query_id = f"{kind}_{idx:02d}_{node['id']}"
            failures = failure_by_sop.get(node["id"], [])
            sig = signatures.get(node["id"], ())
            rarity_count = counts.get(sig, 0)
            is_rare = kind in {"rare_condition", "rare_partial_clue"} or profile == "edge"
            split = split_by_sop.get(node["id"]) or ("dev" if idx <= split_cutoff else "test")
            reason = edge_reason(node, failures)
            rows.append(query_row(
                query_id,
                node,
                kind,
                failures,
                split=split,
                all_sops=task_sops,
                failure_by_sop=failure_by_sop,
                style=style,
            ))
            if profile == "edge":
                rows[-1]["benchmark_source"] = "edge_auto_seeded_from_certified_hyper_graph"
                rows[-1]["edge_gold"] = True
                rows[-1]["edge_reason"] = reason
                rows[-1]["gold_radius_band"] = "edge"
            gold.append(gold_row(
                query_id,
                node,
                relevance,
                is_rare,
                rarity_count,
                edge_gold=profile == "edge",
                edge_reason_value=reason,
            ))

    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    gold_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gold) + "\n", encoding="utf-8")
    return {
        "status": (
            "edge_auto_seeded_needs_human_audit" if profile == "edge"
            else ("hard_auto_seeded_needs_human_audit" if profile == "hard" else "auto_seeded_needs_human_audit")
        ),
        "profile": profile,
        "benchmark": str(benchmark_path),
        "gold": str(gold_path),
        "queries": len(rows),
        "by_kind": dict(collections.Counter(r["query_kind"] for r in rows)),
        "by_split": dict(collections.Counter(r["split"] for r in rows)),
        "by_query_style": dict(collections.Counter(r["query_style"] for r in rows)),
        "by_query_specificity": dict(collections.Counter(r["query_specificity"] for r in rows)),
        "title_token_overlap_mean": sum(r["title_token_overlap"] for r in rows) / max(1, len(rows)),
        "title_leakage_levels": dict(collections.Counter(r["title_leakage_level"] for r in rows)),
        "distractor_count_mean": sum(r["distractor_count"] for r in rows) / max(1, len(rows)),
        "distractor_count_min": min((r["distractor_count"] for r in rows), default=0),
        "edge_min_distractors": int(edge_min_distractors),
        "gold_radius_bands": dict(collections.Counter(r.get("gold_radius_band", "") for r in rows)),
        "edge_reasons": dict(collections.Counter(r.get("edge_reason", "") for r in rows if r.get("edge_reason"))),
        "note": "Use for deterministic harness development. Human audit is required before final paper claim tables.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hyperbolic SOP benchmark seed set.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--profile", choices=["hard", "seed", "edge"], default="hard")
    parser.add_argument("--hard-per-kind", type=int, default=DEFAULT_HARD_PER_KIND)
    parser.add_argument("--edge-variants-per-sop", type=int, default=DEFAULT_EDGE_VARIANTS_PER_SOP)
    parser.add_argument("--edge-min-distractors", type=int, default=DEFAULT_EDGE_MIN_DISTRACTORS)
    args = parser.parse_args()
    print(json.dumps(
        build(
            args.graph,
            args.benchmark,
            args.gold,
            profile=args.profile,
            hard_per_kind=args.hard_per_kind,
            edge_variants_per_sop=args.edge_variants_per_sop,
            edge_min_distractors=args.edge_min_distractors,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
