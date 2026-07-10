"""Train a frozen query-radius-band predictor for V3 diagnostics.

The predictor is intentionally small and deterministic. It only fits dev split
queries and predicts core/middle/edge from benchmark query text. Test split
evaluation must load the saved artifact; this script never fits on test rows.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_BENCH = REPO / "paper-skills" / "eval_skill_memory" / "benchmarks" / "hyperbolic_sop_benchmark.jsonl"
DEFAULT_GOLD = REPO / "paper-skills" / "eval_skill_memory" / "gold" / "hyperbolic_sop_gold.jsonl"
DEFAULT_OUTPUT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "radius_band_predictor.joblib"


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
            "Task type: " + str(row.get("task_type", "")),
        ]
    ).strip()


def radius_band(node: dict[str, Any]) -> str:
    if node.get("radius_band"):
        return str(node.get("radius_band"))
    try:
        radius = float(node.get("radius"))
    except Exception:
        return "middle"
    if radius <= 0.35:
        return "core"
    if radius <= 0.60:
        return "middle"
    return "edge"


def gold_label(gold_row: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> str:
    for item in gold_row.get("gold_sops", []) or []:
        if item.get("relevance") not in {"required", "helpful", "risk_warning"}:
            continue
        if item.get("gold_radius_band"):
            return str(item["gold_radius_band"])
        sid = str(item.get("sop_id", ""))
        if sid in nodes:
            return radius_band(nodes[sid])
    raise ValueError(f"gold row has no usable SOP label: {gold_row.get('query_id')}")


def train(
    *,
    graph_path: Path,
    benchmark_path: Path,
    gold_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = {str(n["id"]): n for n in graph.get("nodes", [])}
    benchmark = read_jsonl(benchmark_path)
    gold_by_query = {str(row["query_id"]): row for row in read_jsonl(gold_path)}

    texts: list[str] = []
    labels: list[str] = []
    query_ids: list[str] = []
    for row in benchmark:
        if str(row.get("split", "")) != "dev":
            continue
        qid = str(row.get("query_id", ""))
        if qid not in gold_by_query:
            continue
        texts.append(query_text(row))
        labels.append(gold_label(gold_by_query[qid], nodes))
        query_ids.append(qid)
    if not texts:
        raise ValueError("no dev split rows available for radius predictor training")

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2048)
    features = vectorizer.fit_transform(texts)
    label_counts = collections.Counter(labels)
    if len(label_counts) == 1:
        model = DummyClassifier(strategy="constant", constant=labels[0])
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(features, labels)

    artifact = {
        "version": "radius_band_predictor_v1",
        "train_split": "dev",
        "model": model,
        "vectorizer": vectorizer,
        "labels": sorted(label_counts),
        "train_query_ids": query_ids,
        "label_counts": dict(sorted(label_counts.items())),
        "benchmark": str(benchmark_path),
        "gold": str(gold_path),
        "graph": str(graph_path),
        "note": "Frozen predictor for test split radius-hint ablation; do not refit on test rows.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return {
        "status": "trained",
        "output": str(output_path),
        "train_split": "dev",
        "train_rows": len(texts),
        "label_counts": dict(sorted(label_counts.items())),
        "model_class": type(model).__name__,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a dev-only radius band predictor.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(train(
        graph_path=args.graph,
        benchmark_path=args.benchmark,
        gold_path=args.gold,
        output_path=args.output,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
