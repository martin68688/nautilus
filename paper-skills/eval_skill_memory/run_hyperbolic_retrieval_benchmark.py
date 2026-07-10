"""Run offline retrieval for Agentic Hyperbolic SOP Memory.

By default the runner avoids LLM tool-choice: each agentic system performs the
same three read-only map steps inspect_map -> navigate -> check_conflicts. Use
``--navigator-mode llm`` for a real MemoryNavigator agent that calls the
configured DeepSeek/OpenAI-compatible model to choose map tools.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "mlevolve"))

from agents.memory.external_skill_memory import ExternalSkillMemoryLayer  # noqa: E402


DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_INDEX = REPO / "paper-skills" / "hyper_memory" / "hyper_index.npz"
DEFAULT_TEXT_MODEL = REPO / "paper-skills" / "hyper_memory" / "hyper_text_model.joblib"
DEFAULT_BENCH = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "hyperbolic_sop_benchmark.jsonl"
DEFAULT_OUTPUT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "hyperbolic_retrieval_results.jsonl"
DEFAULT_LLM_CONFIG = REPO / "mlevolve" / "config" / "config_hyperbolic_agentic.yaml"
DEFAULT_RADIUS_PREDICTOR = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "radius_band_predictor.joblib"

SYSTEMS = {
    "skillgraph_c_lexical": {"mode": "skillgraph", "scoring_mode": "lexical", "enable_agentic": False},
    "agentic_lexical": {"mode": "skillgraph", "scoring_mode": "lexical", "enable_agentic": True},
    "agentic_euclidean": {"mode": "agentic_euclidean", "scoring_mode": "euclidean", "enable_agentic": True},
    "agentic_poincare": {"mode": "agentic_hyperbolic", "scoring_mode": "poincare", "enable_agentic": True},
    "agentic_flat_twin": {"mode": "flat_twin_agentic", "scoring_mode": "flat_twin", "enable_agentic": True},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def query_text(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(row.get("context", "")),
            "Stage: " + str(row.get("stage", "")),
            "Query kind: " + str(row.get("query_kind", "")),
            "Query specificity: " + str(row.get("query_specificity", "")),
            "Conditions: " + "; ".join(row.get("condition") or []),
            "Failure modes: " + "; ".join(row.get("failure_mode") or []),
        ]
    ).strip()


def load_radius_predictor(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"learned radius predictor not found: {path}")
    predictor = joblib.load(path)
    if not isinstance(predictor, dict) or "model" not in predictor or "vectorizer" not in predictor:
        raise ValueError(f"invalid radius predictor artifact: {path}")
    if predictor.get("train_split") and predictor.get("train_split") != "dev":
        raise ValueError(f"radius predictor must be trained on dev split only: {path}")
    return predictor


def learned_radius_hint(predictor: dict[str, Any], row: dict[str, Any]) -> tuple[str, list[dict[str, float | str]]]:
    vectorizer = predictor["vectorizer"]
    model = predictor["model"]
    labels = list(predictor.get("labels") or getattr(model, "classes_", []))
    text = query_text(row)
    features = vectorizer.transform([text])
    distribution: list[dict[str, float | str]] = []
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        labels = list(getattr(model, "classes_", labels))
        pairs = sorted(zip(labels, probs), key=lambda item: (-float(item[1]), str(item[0])))
        total = sum(float(p) for _, p in pairs) or 1.0
        distribution = [
            {"band": str(label), "weight": float(prob) / total, "source": "learned_radius_predictor"}
            for label, prob in pairs
        ]
        top = [str(label) for label, prob in pairs if float(prob) >= 0.20][:2] or [str(pairs[0][0])]
        return ",".join(top), distribution
    label = str(model.predict(features)[0])
    return label, [{"band": label, "weight": 1.0, "source": "learned_radius_predictor"}]


def resolve_radius_hint(
    row: dict[str, Any],
    radius_hint_mode: str,
    *,
    radius_predictor: dict[str, Any] | None = None,
) -> tuple[str, str, list[dict[str, float | str]]]:
    original = str(row.get("radius_band_hint", "") or "")
    if radius_hint_mode == "use_gold_hint":
        return original, original, []
    if radius_hint_mode in {"ignore_hint", "predicted_only"}:
        return original, "", []
    if radius_hint_mode == "learned_predictor":
        if radius_predictor is None:
            raise ValueError("radius_hint_mode=learned_predictor requires a loaded predictor")
        used, distribution = learned_radius_hint(radius_predictor, row)
        return original, used, distribution
    raise ValueError(f"unsupported radius_hint_mode: {radius_hint_mode}")


def make_layer(
    *,
    system_name: str,
    graph_path: Path,
    index_path: Path,
    text_model_path: Path,
    top_k: int,
    geometry_params: dict[str, Any] | None = None,
    cfg: Any | None = None,
) -> ExternalSkillMemoryLayer:
    system_cfg = SYSTEMS[system_name]
    geometry_params = geometry_params or {}
    return ExternalSkillMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        text_model_path=str(text_model_path),
        source_name=system_name,
        mode=system_cfg["mode"],
        scoring_mode=system_cfg["scoring_mode"],
        geometry_distance_weight=float(geometry_params.get("geometry_distance_weight", 0.30)),
        geometry_semantic_weight=float(geometry_params.get("geometry_semantic_weight", 0.20)),
        geometry_constraint_weight=float(geometry_params.get("geometry_constraint_weight", 0.05)),
        geometry_condition_weight=float(geometry_params.get("geometry_condition_weight", 0.18)),
        geometry_failure_weight=float(geometry_params.get("geometry_failure_weight", 0.14)),
        geometry_evidence_weight=float(geometry_params.get("geometry_evidence_weight", 0.08)),
        geometry_reliability_weight=float(geometry_params.get("geometry_reliability_weight", 0.08)),
        geometry_conflict_weight=float(geometry_params.get("geometry_conflict_weight", 0.10)),
        geometry_distance_norm=str(geometry_params.get("geometry_distance_norm", "none")),
        geometry_query_radius_quantile=float(geometry_params.get("geometry_query_radius_quantile", 0.5)),
        geometry_query_radius_mode=str(geometry_params.get("geometry_query_radius_mode", "predicted_distribution")),
        geometry_query_radius_bands=geometry_params.get("geometry_query_radius_bands", ["core", "middle", "edge"]),
        geometry_query_radius_top_bands=int(geometry_params.get("geometry_query_radius_top_bands", 2)),
        geometry_radius_fusion=str(geometry_params.get("geometry_radius_fusion", "weighted_max")),
        enable_agentic=system_cfg["enable_agentic"],
        navigator_max_steps=3,
        navigator_reference_budget=1200,
        top_k=top_k,
        cfg=cfg,
    )


def run_one(
    layer: ExternalSkillMemoryLayer,
    system_name: str,
    row: dict[str, Any],
    top_k: int,
    *,
    navigator_mode: str = "deterministic",
    allow_llm_fallback: bool = True,
    radius_hint_mode: str = "use_gold_hint",
    radius_predictor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    task_type = str(row.get("task_type", ""))
    text = query_text(row)
    original_radius_hint, used_radius_hint, learned_radius_distribution = resolve_radius_hint(
        row,
        radius_hint_mode,
        radius_predictor=radius_predictor,
    )
    navigation_error = ""
    agentic_pack: dict[str, Any] = {}
    if system_name == "skillgraph_c_lexical":
        selected = layer._retrieve_ids(task_type=task_type, query_text=text)[:top_k]
        trace = ["skillgraph_c_lexical_retrieve_ids"]
        risk_warnings: list[str] = []
        rejected_sops: list[dict[str, str]] = []
    elif navigator_mode == "llm":
        try:
            _context, selected = layer._retrieve_agentic(
                task_type=task_type,
                stage=str(row.get("stage", "improve")),
                task_desc=str(row.get("context", "")),
                query_text=text,
            )
            selected = selected[:top_k]
            agentic_pack = dict(getattr(layer, "_last_agentic_pack", {}) or {})
            trace = list(agentic_pack.get("navigation_trace", []) or [])
            risk_warnings = list(agentic_pack.get("risk_warnings", []) or [])
            rejected_sops = list(agentic_pack.get("rejected_sops", []) or [])
        except Exception as exc:
            if not allow_llm_fallback:
                raise
            navigation_error = f"{type(exc).__name__}: {exc}"
            pack, selected = layer._deterministic_agentic_pack(
                task_type=task_type,
                stage=str(row.get("stage", "improve")),
                task_desc=str(row.get("context", "")),
                query_text=text,
            )
            pack["mode"] = "llm_error_fallback"
            pack["navigation_error"] = navigation_error
            agentic_pack = pack
            selected = selected[:top_k]
            trace = list(pack.get("navigation_trace", []) or [])
            risk_warnings = list(pack.get("risk_warnings", []) or [])
            rejected_sops = list(pack.get("rejected_sops", []) or [])
    else:
        map_obs = layer.inspect_map(task_type=task_type, query_text=text, context=str(row.get("context", "")))
        nav = layer.navigate(
            task_type=task_type,
            query_text=text,
            condition=" ".join(row.get("condition") or []),
            failure_mode=" ".join(row.get("failure_mode") or []),
            radius_band=used_radius_hint,
            top_k=top_k,
        )
        selected = [s["id"] for s in nav.get("sops", []) if s.get("id") in layer.nodes][:top_k]
        conflicts = layer.check_conflicts(sop_ids=selected[:4], context=text)
        trace = [
            "inspect_map(context)",
            "navigate(condition=query.condition, failure_mode=query.failure_mode)",
            "check_conflicts(selected_sops)",
        ]
        risk_warnings = conflicts.get("risk_warnings", [])
        rejected_sops = []
        _ = map_obs

    pack = agentic_pack or {
        "selected_sops": selected,
        "risk_warnings": risk_warnings,
        "navigation_trace": trace,
        "rejected_sops": rejected_sops,
        "mode": "deterministic_runner" if system_name != "skillgraph_c_lexical" else "skillgraph_c_lexical",
        "llm_tool_calls": 0,
    }
    if learned_radius_distribution and layer.scoring_mode in {"poincare", "flat_twin", "euclidean"}:
        query_radius_distribution = [
            {
                **item,
                "radius": layer._band_center_radius(str(item.get("band", ""))),
            }
            for item in learned_radius_distribution
        ]
    elif layer.scoring_mode in {"poincare", "flat_twin", "euclidean"}:
        query_radius_distribution = layer._predict_query_radius_distribution(text, used_radius_hint)
    else:
        query_radius_distribution = []

    injected_tokens = len(json.dumps(pack, ensure_ascii=False).split())
    elapsed = round(time.time() - started, 6)
    return {
        "query_id": row["query_id"],
        "query_kind": row.get("query_kind", ""),
        "split": row.get("split", ""),
        "query_style": row.get("query_style", ""),
        "query_specificity": row.get("query_specificity", ""),
        "task_type": task_type,
        "system": system_name,
        "scoring_mode": layer.scoring_mode,
        "navigator_mode": navigator_mode if system_name != "skillgraph_c_lexical" else "none",
        "agentic_pack_mode": pack.get("mode", ""),
        "llm_tool_calls": int(pack.get("llm_tool_calls", 0) or 0),
        "radius_hint_mode": radius_hint_mode,
        "original_radius_band_hint": original_radius_hint,
        "used_radius_band_hint": used_radius_hint,
        "radius_hint_rejected": bool(original_radius_hint and original_radius_hint != used_radius_hint),
        "selected_sops": selected,
        "scores": [
            {"sop_id": sid, "rank": rank, "score": round(1.0 / rank, 6)}
            for rank, sid in enumerate(selected, 1)
        ],
        "selected_radius_bands": [layer._radius_band(layer.nodes.get(sid, {})) for sid in selected],
        "query_radius_distribution": query_radius_distribution,
        "navigation_trace": trace,
        "risk_warnings": risk_warnings,
        "rejected_sops": rejected_sops,
        "distractor_sops": row.get("distractor_sops", []) or [],
        "opened_references": [],
        "navigation_error": navigation_error or str(pack.get("navigation_error", "")),
        "latency_sec": elapsed,
        "injected_tokens": injected_tokens,
    }


def run(
    *,
    graph_path: Path,
    index_path: Path,
    text_model_path: Path,
    benchmark_path: Path,
    output_path: Path,
    systems: list[str],
    top_k: int = 5,
    geometry_params: dict[str, Any] | None = None,
    navigator_mode: str = "deterministic",
    cfg: Any | None = None,
    limit_queries: int | None = None,
    split: str = "",
    query_kinds: set[str] | None = None,
    allow_llm_fallback: bool = True,
    radius_hint_mode: str = "use_gold_hint",
    radius_predictor_path: Path | None = None,
) -> dict[str, Any]:
    queries = read_jsonl(benchmark_path)
    if split:
        queries = [q for q in queries if str(q.get("split", "")) == split]
    if query_kinds:
        queries = [q for q in queries if str(q.get("query_kind", "")) in query_kinds]
    if limit_queries is not None:
        queries = queries[: max(0, int(limit_queries))]
    radius_predictor = load_radius_predictor(radius_predictor_path) if radius_hint_mode == "learned_predictor" else None
    layers = {
        name: make_layer(
            system_name=name,
            graph_path=graph_path,
            index_path=index_path,
            text_model_path=text_model_path,
            top_k=top_k,
            geometry_params=geometry_params,
            cfg=cfg if navigator_mode == "llm" and name != "skillgraph_c_lexical" else None,
        )
        for name in systems
    }
    rows = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for query in queries:
            for name in systems:
                row = run_one(
                    layers[name],
                    name,
                    query,
                    top_k=top_k,
                    navigator_mode=navigator_mode,
                    allow_llm_fallback=allow_llm_fallback,
                    radius_hint_mode=radius_hint_mode,
                    radius_predictor=radius_predictor,
                )
                rows.append(row)
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
    return {
        "status": "completed",
        "output": str(output_path),
        "queries": len(queries),
        "systems": systems,
        "rows": len(rows),
        "top_k": top_k,
        "geometry_params": geometry_params or {},
        "navigator_mode": navigator_mode,
        "split": split,
        "query_kinds": sorted(query_kinds or []),
        "limit_queries": limit_queries,
        "radius_hint_mode": radius_hint_mode,
        "radius_predictor": str(radius_predictor_path) if radius_predictor_path and radius_hint_mode == "learned_predictor" else "",
        "llm_rows": sum(1 for row in rows if row.get("llm_tool_calls", 0) > 0),
        "llm_tool_calls": sum(int(row.get("llm_tool_calls", 0) or 0) for row in rows),
        "fallback_rows": sum(1 for row in rows if "fallback" in str(row.get("agentic_pack_mode", ""))),
        "error_rows": sum(1 for row in rows if row.get("navigation_error")),
    }


def load_llm_cfg(path: Path) -> Any:
    from config import _load_cfg

    return _load_cfg(path, use_cli_args=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hyperbolic SOP retrieval benchmark.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--text-model", type=Path, default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--geometry-params", type=str, default="{}", help="JSON object of geometry scoring parameters.")
    parser.add_argument("--navigator-mode", choices=["deterministic", "llm"], default="deterministic")
    parser.add_argument("--llm-config", type=Path, default=DEFAULT_LLM_CONFIG)
    parser.add_argument("--limit-queries", type=int, default=None)
    parser.add_argument("--split", choices=["", "dev", "test"], default="")
    parser.add_argument("--query-kinds", nargs="*", default=[])
    parser.add_argument("--no-llm-fallback", action="store_true")
    parser.add_argument(
        "--radius-hint-mode",
        choices=["use_gold_hint", "ignore_hint", "predicted_only", "learned_predictor"],
        default="use_gold_hint",
    )
    parser.add_argument("--radius-predictor", type=Path, default=DEFAULT_RADIUS_PREDICTOR)
    args = parser.parse_args()
    unknown = sorted(set(args.systems) - set(SYSTEMS))
    if unknown:
        raise SystemExit(f"Unknown systems: {unknown}")
    geometry_params = json.loads(args.geometry_params)
    cfg = load_llm_cfg(args.llm_config) if args.navigator_mode == "llm" else None
    print(json.dumps(run(
        graph_path=args.graph,
        index_path=args.index,
        text_model_path=args.text_model,
        benchmark_path=args.benchmark,
        output_path=args.output,
        systems=args.systems,
        top_k=args.top_k,
        geometry_params=geometry_params,
        navigator_mode=args.navigator_mode,
        cfg=cfg,
        limit_queries=args.limit_queries,
        split=args.split,
        query_kinds=set(args.query_kinds or []),
        allow_llm_fallback=not args.no_llm_fallback,
        radius_hint_mode=args.radius_hint_mode,
        radius_predictor_path=args.radius_predictor,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
