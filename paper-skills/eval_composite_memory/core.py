from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
TAXONOMY = REPO / "paper-skills" / "hyper_memory" / "sop_taxonomy.json"
REPLAY_TARGETS = REPO / "paper-skills" / "eval_skill_memory" / "clean_replay_targets.json"

MANIFESTS = HERE / "manifests"
EPISODES = HERE / "episodes"
ANNOTATIONS = HERE / "annotations"
ARTIFACTS = HERE / "artifacts"
REPORTS = HERE / "reports"


TASK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "spooky-author-identification",
        "family": "text_classification",
        "profile": "Authorship classification with short English prose and multiclass log loss.",
        "metric": "log_loss",
        "direction": "minimize",
        "source_task": "spooky-author-identification",
    },
    {
        "task_id": "news-topic-classification",
        "family": "text_classification",
        "profile": "Topic classification with sparse lexical signals and multiclass log loss.",
        "metric": "log_loss",
        "direction": "minimize",
        "source_task": "elite-peach-mayfly",
    },
    {
        "task_id": "leaf-image-classification",
        "family": "image_classification",
        "profile": "Fine-grained leaf image classification with limited labeled images.",
        "metric": "log_loss",
        "direction": "minimize",
        "source_task": "leaf-classification",
    },
    {
        "task_id": "aerial-cactus-identification",
        "family": "image_classification",
        "profile": "Binary aerial image classification with augmentation-sensitive generalization.",
        "metric": "log_loss",
        "direction": "minimize",
        "source_task": "aerial-cactus-identification",
        "retrieval_family": "image_binary_classification",
    },
    {
        "task_id": "dirty-document-denoising",
        "family": "image_restoration",
        "profile": "Document image restoration with patch training and RMSE evaluation.",
        "metric": "rmse",
        "direction": "minimize",
        "source_task": "denoising-dirty-documents",
    },
    {
        "task_id": "digit-image-denoising",
        "family": "image_restoration",
        "profile": "Small grayscale image denoising under a fixed synthetic corruption process.",
        "metric": "rmse",
        "direction": "minimize",
        "source_task": "denoising-dirty-documents",
    },
    {
        "task_id": "leaf-tabular-multiclass",
        "family": "tabular_multiclass",
        "profile": "Multiclass classification from structured shape and texture descriptors.",
        "metric": "log_loss",
        "direction": "minimize",
        "source_task": "leaf-classification",
    },
    {
        "task_id": "wine-tabular-multiclass",
        "family": "tabular_multiclass",
        "profile": "Small multiclass tabular classification with heterogeneous numeric scales.",
        "metric": "log_loss",
        "direction": "minimize",
        "source_task": "leaf-classification",
    },
    {
        "task_id": "taxi-fare-regression",
        "family": "tabular_regression",
        "profile": "Fare regression with spatial, categorical, and chronological features.",
        "metric": "rmse",
        "direction": "minimize",
        "source_task": "new-york-city-taxi-fare-prediction",
    },
    {
        "task_id": "diabetes-progression-regression",
        "family": "tabular_regression",
        "profile": "Moderate-dimensional numeric regression with limited samples.",
        "metric": "rmse",
        "direction": "minimize",
        "source_task": "new-york-city-taxi-fare-prediction",
    },
    {
        "task_id": "grouped-patient-classification",
        "family": "group_time_aware",
        "retrieval_family": "tabular_multiclass",
        "profile": "Grouped classification where records from one entity must remain in one split.",
        "metric": "log_loss",
        "direction": "minimize",
        "source_task": "leaf-classification",
    },
    {
        "task_id": "temporal-demand-regression",
        "family": "group_time_aware",
        "retrieval_family": "tabular_regression",
        "profile": "Time-ordered regression where future observations cannot inform earlier fits.",
        "metric": "rmse",
        "direction": "minimize",
        "source_task": "new-york-city-taxi-fare-prediction",
    },
)


DECISION_TEMPLATES: tuple[dict[str, str], ...] = (
    {"name": "draft_capacity", "stage": "draft", "prompt": "Choose a high-capacity overall method family with a falsifiable generalization hypothesis."},
    {"name": "draft_efficiency", "stage": "draft", "prompt": "Choose a compute-conscious overall route that remains scientifically credible."},
    {"name": "design_architecture", "stage": "model_design", "prompt": "Specify an architecture tactic compatible with the already selected method family."},
    {"name": "design_features", "stage": "model_design", "prompt": "Choose a feature or representation tactic without changing the task definition."},
    {"name": "improve_regularization", "stage": "improve", "prompt": "Improve generalization while preserving the selected model family."},
    {"name": "improve_validation", "stage": "improve", "prompt": "Improve model selection and validation without optimistic reuse."},
    {"name": "improve_efficiency", "stage": "improve", "prompt": "Reduce cost or instability without silently shrinking the scientific method."},
    {"name": "debug_fit_scope", "stage": "debug", "prompt": "Repair learned preprocessing or fit scope that may cross a validation boundary."},
    {"name": "debug_alignment", "stage": "debug", "prompt": "Repair sample, tensor, feature, or prediction alignment while preserving semantics."},
    {"name": "debug_runtime", "stage": "debug", "prompt": "Repair a deterministic runtime or resource failure without redesigning the model."},
)


CONDITIONS: dict[str, dict[str, Any]] = {
    "F00": {"portfolio": "N+N+N", "memory": "flat_clean", "tier": "primary"},
    "F01": {"portfolio": "N+N+N", "memory": "stage_hybrid_v2", "tier": "primary"},
    "F10": {"portfolio": "B+R+N", "memory": "flat_clean", "tier": "primary"},
    "F11": {"portfolio": "B+R+N", "memory": "stage_hybrid_v2", "tier": "primary"},
    "P0": {"portfolio": "N+N+N", "memory": "stage_hybrid_v2", "tier": "portfolio"},
    "P1": {"portfolio": "B+N+N", "memory": "stage_hybrid_v2", "tier": "portfolio"},
    "P2": {"portfolio": "R+N+N", "memory": "stage_hybrid_v2", "tier": "portfolio"},
    "P3": {"portfolio": "B+R+N", "memory": "stage_hybrid_v2", "tier": "portfolio"},
    "D1": {"portfolio": "B+R+N", "memory": "legacy_gateway", "tier": "diagnostic"},
    "D2": {"portfolio": "B+R+N", "memory": "tree_only", "tier": "diagnostic"},
    "D3": {"portfolio": "B+R+N", "memory": "sop_only", "tier": "diagnostic"},
    "D4": {"portfolio": "B+R+N", "memory": "stage_hybrid_no_stage", "tier": "diagnostic"},
    "D5": {"portfolio": "B+R+N", "memory": "stage_hybrid_no_task", "tier": "diagnostic"},
    "D6": {"portfolio": "B+R+N", "memory": "stage_hybrid_flat_twin", "tier": "diagnostic"},
    "D7": {"portfolio": "B+R+N", "memory": "stage_hybrid_unsafe_offline", "tier": "diagnostic", "execution_forbidden": True},
    "D8": {"portfolio": "B+R+N", "memory": "stage_hybrid_clean_universe", "tier": "diagnostic"},
    "B0": {"portfolio": "B+B+B", "memory": "none", "tier": "baseline"},
    "B1": {"portfolio": "N+N+N", "memory": "random_clean", "tier": "baseline"},
    "B2": {"portfolio": "N+N+N", "memory": "flat_clean", "tier": "baseline"},
    "B3": {"portfolio": "B+B+B", "memory": "greedy_best_valid", "tier": "baseline"},
    "B4": {"portfolio": "B+B+B", "memory": "compatible_clean_ensemble", "tier": "baseline", "conditional": True},
    "O1": {"portfolio": "oracle", "memory": "safe_oracle", "tier": "oracle", "execution_forbidden": True},
}


DEFECT_SPECS: tuple[dict[str, str], ...] = (
    {"defect": "transductive_fit", "stage": "data_scope", "issue_code": "TRANSFORM_FIT_ON_HOLDOUT"},
    {"defect": "early_stop_report_reuse", "stage": "validation_provenance", "issue_code": "VALIDATION_REUSED_FOR_EARLY_STOPPING_AND_REPORT"},
    {"defect": "false_oof", "stage": "cross_fit", "issue_code": "CROSS_FOLD_SUPERVISED_FEATURE_LEAKAGE"},
    {"defect": "holdout_weight_search", "stage": "selection_freeze", "issue_code": "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION"},
    {"defect": "group_split_leakage", "stage": "data_scope", "issue_code": "GROUP_SPLIT_LEAKAGE"},
    {"defect": "temporal_order_leakage", "stage": "data_scope", "issue_code": "TEMPORAL_SPLIT_LEAKAGE"},
    {"defect": "target_encoding_leakage", "stage": "cross_fit", "issue_code": "TARGET_ENCODING_FIT_OUTSIDE_FOLD"},
    {"defect": "post_split_dedup", "stage": "data_scope", "issue_code": "POST_SPLIT_DUPLICATE_LEAKAGE"},
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def run_id_for_node(node: dict[str, Any]) -> str:
    return str(node.get("run_short_id") or node.get("run_id") or "")


def positive_run_node(node: dict[str, Any]) -> bool:
    audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
    metric = node.get("metric")
    return bool(
        audit.get("status") == "clean"
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
        and audit.get("rank_eligible") is True
        and node.get("is_valid") is True
        and node.get("is_buggy") is False
        and node.get("quarantined") is not True
        and node.get("protocol_biased") is not True
        and isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
    )


def clean_support(graph: dict[str, Any]) -> dict[str, list[str]]:
    nodes = {str(node["id"]): node for node in graph.get("nodes", []) if node.get("id")}
    support: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get("edges", []):
        if str(edge.get("kind") or edge.get("type") or "") != "distills_to":
            continue
        transition_id = str(edge.get("src") or "")
        sop_id = str(edge.get("dst") or "")
        transition = nodes.get(transition_id, {})
        child = nodes.get(str(transition.get("child_node_id") or ""), {})
        outcome = str(transition.get("outcome") or "")
        if (
            transition.get("quarantined") is not True
            and transition.get("protocol_biased") is not True
            and outcome not in {"buggy", "metric_worsened", "unknown"}
            and positive_run_node(child)
        ):
            support[sop_id].append(transition_id)
    return {key: sorted(set(value)) for key, value in support.items()}


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


def task_cluster_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    metric: str,
    samples: int = 5000,
    seed: int = 73,
) -> dict[str, Any]:
    right_by_key = {(row["task_id"], row["seed"]): row for row in right}
    pairs = [
        (row["task_id"], float(row[metric]) - float(right_by_key[(row["task_id"], row["seed"])][metric]))
        for row in left
        if (row["task_id"], row["seed"]) in right_by_key
    ]
    by_task: dict[str, list[float]] = defaultdict(list)
    for task_id, delta in pairs:
        by_task[task_id].append(delta)
    task_means = np.asarray([np.mean(values) for values in by_task.values()], dtype=np.float64)
    if not len(task_means):
        return {"task_count": 0, "delta": None, "cluster_bootstrap_ci95": [None, None]}
    rng = np.random.default_rng(seed)
    boot = np.asarray([
        float(task_means[rng.integers(0, len(task_means), len(task_means))].mean())
        for _ in range(samples)
    ])
    return {
        "task_count": len(task_means),
        "delta": float(task_means.mean()),
        "cluster_bootstrap_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
    }


def task_cluster_signflip_p(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, metric: str, samples: int = 20000, seed: int = 91
) -> float | None:
    right_by_key = {(row["task_id"], row["seed"]): row for row in right}
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in left:
        key = (row["task_id"], row["seed"])
        if key in right_by_key:
            by_task[row["task_id"]].append(float(row[metric]) - float(right_by_key[key][metric]))
    effects = np.asarray([np.mean(values) for values in by_task.values()], dtype=float)
    if not len(effects):
        return None
    observed = abs(float(effects.mean()))
    if len(effects) <= 16:
        null = np.asarray([
            abs(float(np.mean(effects * np.asarray([1 if mask & (1 << index) else -1 for index in range(len(effects))]))))
            for mask in range(1 << len(effects))
        ])
    else:
        rng = np.random.default_rng(seed)
        null = np.asarray([
            abs(float(np.mean(effects * rng.choice((-1.0, 1.0), size=len(effects))))) for _ in range(samples)
        ])
    return float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))


def holm_adjust(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    result: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - index) * value))
        result[name] = running
    return result


def deterministic_order(values: Iterable[str], salt: str) -> list[str]:
    return sorted(values, key=lambda value: hashlib.sha256(f"{salt}|{value}".encode()).hexdigest())


def seeded_random_order(values: Iterable[str], seed: int) -> list[str]:
    rows = list(values)
    random.Random(seed).shuffle(rows)
    return rows


def counts_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key)) for row in rows).items()))
