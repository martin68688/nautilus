#!/usr/bin/env python3
"""Build auditable point-in-time Strategy shadow replay cases."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments" / "end2end_memory_systems_20260804"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _metric(node: Mapping[str, Any]) -> float | None:
    value = node.get("metric")
    if isinstance(value, Mapping):
        value = value.get("value")
    return float(value) if value is not None else None


def _journal_node_payload(node: Mapping[str, Any], *, task_id: str) -> dict[str, Any]:
    return {
        "node_id": str(node.get("id") or ""),
        "code": str(node.get("code") or ""),
        "code_summary": str(node.get("code_summary") or ""),
        "plan": str(node.get("plan") or ""),
        "metric": _metric(node),
        "maximize": bool((node.get("metric") or {}).get("maximize", False))
        if isinstance(node.get("metric"), Mapping)
        else False,
        "stage": str(node.get("stage") or "improve"),
        "is_buggy": bool(node.get("is_buggy", False)),
        "is_valid": node.get("is_valid"),
        "branch_id": node.get("branch_id"),
        "draft_role": str(node.get("draft_role") or ""),
        "leakage_audit": dict(node.get("leakage_audit") or {}),
        "official_submission_receipt": dict(
            node.get("official_submission_receipt") or {}
        ),
        "metric_protocol": str(
            node.get("metric_protocol") or node.get("validation_protocol") or ""
        ),
        "execution_output": "".join(node.get("_term_out") or []),
        "task_id": task_id,
    }


def _current_branch_nodes(
    nodes: list[Mapping[str, Any]],
    *,
    task_id: str,
    max_step: int | None = None,
    max_ctime: float | None = None,
) -> list[dict[str, Any]]:
    """Return the best valid point-in-time node for every observed branch."""

    by_branch: dict[int, list[Mapping[str, Any]]] = {}
    for node in nodes:
        branch_id = node.get("branch_id")
        if branch_id is None:
            continue
        if max_step is not None and int(node.get("step") or 0) > max_step:
            continue
        if max_ctime is not None and float(node.get("ctime") or 0.0) > max_ctime:
            continue
        if (
            _metric(node) is None
            or node.get("is_buggy") is True
            or node.get("is_valid") is False
            or (node.get("leakage_audit") or {}).get("rank_eligible") is False
        ):
            continue
        by_branch.setdefault(int(branch_id), []).append(node)
    selected = []
    for branch_id, candidates in sorted(by_branch.items()):
        maximize = bool(
            (candidates[0].get("metric") or {}).get("maximize", False)
        ) if isinstance(candidates[0].get("metric"), Mapping) else False
        best = sorted(
            candidates,
            key=lambda node: float(_metric(node)),
            reverse=maximize,
        )[0]
        payload = _journal_node_payload(best, task_id=task_id)
        payload["branch_id"] = branch_id
        selected.append(payload)
    return selected


def _card_from_graph_node(node: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(node.get("id") or ""),
        "source": "runforest",
        "type": str(node.get("type") or "RunNode"),
        "source_stage": str(node.get("stage") or ""),
        "source_task_id": str(node.get("task") or ""),
        "metric": _metric(node),
        "outcome": str(node.get("outcome") or ""),
        "plan": str(node.get("plan") or ""),
        "text": str(node.get("code_summary") or node.get("text") or ""),
        "rank_eligible": bool(node.get("rank_eligible", True)),
        "run_id": str(node.get("run_id") or ""),
        "branch_id": node.get("branch_id"),
        "draft_role": str(node.get("draft_role") or ""),
        "validation_protocol": str(node.get("validation_protocol") or ""),
        "evidence_tier": str(node.get("evidence_tier") or ""),
        "metric_provenance": str(node.get("metric_provenance") or ""),
        "method_fingerprint": str(node.get("method_fingerprint") or ""),
        "original_node_id": str(node.get("original_node_id") or ""),
        "available_order": int(node.get("step") or 0),
    }


def _spooky_case() -> dict[str, Any]:
    run_id = "20260509_042918_spooky-author-identification"
    graph = _load(ROOT / "paper-skills/hyper_memory/run_forest_graph.json")
    journal = _load(ROOT / f"mlevolve/runs/{run_id}/logs/journal.json")
    graph_nodes = [
        node
        for node in graph["nodes"]
        if node.get("type") == "RunNode"
        and node.get("run_id") == run_id
        and int(node.get("step") or 0) <= 42
    ]
    visible = [
        node
        for node in graph_nodes
        if int(node.get("step") or 0) <= 34
        and _metric(node) is not None
        and node.get("is_buggy") is not True
    ]
    visible.sort(key=lambda node: (_metric(node), -int(node.get("step") or 0)))
    journal_by_id = {str(node["id"]): node for node in journal["nodes"]}
    parent_raw = journal_by_id["bfd0f74b749d4b5cbd9d374a2fdba397"]
    hidden = next(node for node in graph_nodes if int(node.get("step") or 0) == 42)
    return {
        "case_id": "spooky-before-frozen-multibackbone-fivefold-xgb",
        "task_id": "spooky-author-identification",
        "task_description": "Predict one of three authors from short text; metric is multiclass log-loss.",
        "data_preview": "Approximately 19.5k short English passages, three author labels, raw text and ID columns.",
        "stage": "improve",
        "atomic_actuation_enabled": False,
        "cutoff": {"order": 34, "meaning": "same-run journal step <= 34"},
        "parent": _journal_node_payload(parent_raw, task_id="spooky-author-identification"),
        "current_branch_nodes": _current_branch_nodes(
            journal["nodes"],
            task_id="spooky-author-identification",
            max_step=34,
        ),
        "metrics": {"branch_best_metric": 0.30252106930036915},
        "budget": {
            "total_search_seconds": 21600,
            "elapsed_search_seconds": 11500,
            "remaining_search_seconds": 10100,
        },
        "memory_events": [_card_from_graph_node(node) for node in visible],
        "hidden_future": {
            "id": str(hidden["id"]),
            "memory_ids": [str(hidden["id"])],
            "metric": _metric(hidden),
            "method_summary": str(hidden.get("code_summary") or ""),
            "evaluator_only": True,
        },
        # The first score is deliberately structural. RoBERTa-base had not
        # appeared in a successful implementation before this cutoff, so an
        # exact-name match is a harder, separately reported criterion rather
        # than something leaked into the Agent context.
        "expected_structural_pattern_groups": [
            [r"three.*(transformer|backbone|model)", r"multi[- ]?(transformer|backbone)"],
            [r"frozen.*embedding", r"embedding.*frozen", r"feature extractor"],
            [r"(five|5)[- ]fold", r"stratified.*fold"],
            [r"xgboost", r"xgb"],
        ],
        "expected_exact_future_pattern_groups": [
            [r"deberta"],
            [r"(?<!distil)roberta"],
            [r"distilbert"],
            [r"frozen.*embedding", r"embedding.*frozen", r"feature extractor"],
            [r"(five|5)[- ]fold", r"stratified.*fold"],
            [r"xgboost", r"xgb"],
        ],
        "attempted_pattern_signatures": [
            {
                "required_groups": [
                    [r"modernbert"],
                    [r"fine[- ]tun"],
                    [r"(five|5)[- ]fold", r"stratified.*fold"],
                ],
                "novelty_exclusion_groups": [
                    [r"xgboost", r"xgb", r"blend", r"tf[- ]?idf", r"stylometric", r"deberta", r"distilbert"]
                ],
            },
            {
                "required_groups": [[r"deberta"], [r"single.*(split|holdout)"]],
                "novelty_exclusion_groups": [
                    [r"(five|5)[- ]fold", r"xgboost", r"xgb", r"blend"]
                ],
            },
        ],
        "known_incompatibilities": [
            {
                "pattern_groups": [[r"three.*transformer"], [r"fine[- ]tun"]],
                "resolution_pattern": r"frozen|feature extract",
                "reason": "fine-tuning three transformer backbones exceeds the historical budget",
            }
        ],
    }


def _spooky_repair_case() -> dict[str, Any]:
    """Expose the failed pipeline while hiding its successful cleanup repair."""

    run_id = "20260509_042918_spooky-author-identification"
    graph = _load(ROOT / "paper-skills/hyper_memory/run_forest_graph.json")
    journal = _load(ROOT / f"mlevolve/runs/{run_id}/logs/journal.json")
    graph_nodes = [
        node
        for node in graph["nodes"]
        if node.get("type") == "RunNode"
        and node.get("run_id") == run_id
        and int(node.get("step") or 0) <= 42
    ]
    graph_by_step = {int(node.get("step") or 0): node for node in graph_nodes}
    journal_by_step = {
        int(node.get("step") or 0): node for node in journal["nodes"]
    }
    parent_raw = journal_by_step[38]
    hidden = graph_by_step[42]

    # Put the two causal failures first so they cannot be crowded out of the
    # bounded 24-card Strategy context by unrelated successful attempts.
    events: list[dict[str, Any]] = []
    for step in (38, 35):
        graph_node = graph_by_step[step]
        journal_node = journal_by_step[step]
        card = _card_from_graph_node(graph_node)
        card.update(
            {
                "outcome": "execution_failed",
                "rank_eligible": False,
                "failure_signature": str(journal_node.get("analysis") or ""),
                "repair_action": str(journal_node.get("plan") or ""),
                "available_order": step,
            }
        )
        events.append(card)
    for node in graph_nodes:
        step = int(node.get("step") or 0)
        if step > 41 or step in {35, 38}:
            continue
        if _metric(node) is None or node.get("is_buggy") is True:
            continue
        events.append(_card_from_graph_node(node))

    return {
        "case_id": "spooky-after-multibackbone-before-cleanup-repair",
        "task_id": "spooky-author-identification",
        "task_description": "Predict one of three authors from short text; metric is multiclass log-loss.",
        "data_preview": "Approximately 19.5k short English passages, three author labels, raw text and ID columns.",
        "stage": "debug",
        "memory_strategy_debug_trigger": "always",
        "atomic_actuation_enabled": True,
        "cutoff": {"order": 41, "meaning": "same-run journal step <= 41"},
        "parent": _journal_node_payload(
            parent_raw, task_id="spooky-author-identification"
        ),
        "current_branch_nodes": _current_branch_nodes(
            journal["nodes"],
            task_id="spooky-author-identification",
            max_step=41,
        ),
        "metrics": {"branch_best_metric": 0.30252106930036915},
        "budget": {
            "total_search_seconds": 21600,
            "elapsed_search_seconds": 16200,
            "remaining_search_seconds": 5400,
        },
        "memory_events": events,
        "hidden_future": {
            "id": str(hidden["id"]),
            "memory_ids": [str(hidden["id"])],
            "metric": _metric(hidden),
            "method_summary": str(hidden.get("code_summary") or ""),
            "evaluator_only": True,
        },
        "expected_structural_pattern_groups": [
            [r"targeted.*repair", r"cleanup", r"nameerror"],
            [r"preserv", r"no.*model.*change", r"forbid.*model"],
        ],
        "expected_exact_future_pattern_groups": [
            [r"remove", r"delete"],
            [r"redundant", r"already.*deleted", r"second.*cleanup"],
            [r"models"],
            [r"tokenizers"],
        ],
        "attempted_pattern_signatures": [],
        "known_incompatibilities": [
            {
                "pattern_groups": [[r"replace.*(backbone|model)"]],
                "resolution_pattern": r"preserve|no.*model.*change",
                "reason": "the completed training and submission prove that only cleanup needs repair",
            }
        ],
    }


def _leaf_case() -> dict[str, Any]:
    source_root = ROOT / "coordination/kaggle_leaf_validation_20260810/source_journals"
    names = (
        "dynamic_v22_attempt000.json",
        "gome_v21_attempt000.json",
        "macla_v21_attempt001.json",
        "runforest_v21_attempt001.json",
    )
    journals = {name: _load(source_root / name) for name in names}
    runforest = journals["runforest_v21_attempt001.json"]
    target = next(node for node in runforest["nodes"] if int(node.get("step") or 0) == 30)
    cutoff = float(target["ctime"]) - 1e-6
    parent_raw = next(node for node in runforest["nodes"] if int(node.get("step") or 0) == 28)
    events = []
    for name, journal in journals.items():
        for node in journal["nodes"]:
            if float(node.get("ctime") or 0.0) > cutoff:
                continue
            if _metric(node) is None or node.get("is_buggy") is True:
                continue
            events.append(
                {
                    "candidate_id": f"history::{name}::{node.get('id')}",
                    "source": "runforest",
                    "source_stage": str(node.get("stage") or ""),
                    "source_task_id": "leaf-classification",
                    "metric": _metric(node),
                    "plan": str(node.get("plan") or ""),
                    "text": str(node.get("code_summary") or ""),
                    "rank_eligible": True,
                    "available_at": float(node.get("ctime") or 0.0),
                }
            )
    events.sort(key=lambda item: (float(item.get("metric") or 1e9), -float(item["available_at"])))
    return {
        "case_id": "leaf-before-efficientnet-temperature-fivefold",
        "task_id": "leaf-classification",
        "task_description": "Classify 99 leaf species from 192 tabular shape/texture/margin features and associated images; metric is multiclass log-loss.",
        "data_preview": "891 training rows, 99 balanced classes, 192 numeric descriptors plus image files.",
        "stage": "improve",
        "cutoff": {"timestamp": cutoff, "meaning": "global node ctime before RunForest step 30"},
        "parent": _journal_node_payload(parent_raw, task_id="leaf-classification"),
        "current_branch_nodes": _current_branch_nodes(
            runforest["nodes"],
            task_id="leaf-classification",
            max_ctime=cutoff,
        ),
        "metrics": {"branch_best_metric": 0.04335948075990093},
        "budget": {
            "total_search_seconds": 21600,
            "elapsed_search_seconds": 13800,
            "remaining_search_seconds": 7800,
        },
        "memory_events": events,
        "hidden_future": {
            "id": str(target["id"]),
            "memory_ids": [f"history::runforest_v21_attempt001.json::{target['id']}"],
            "metric": _metric(target),
            "method_summary": str(target.get("plan") or ""),
            "evaluator_only": True,
        },
        "expected_future_pattern_groups": [
            [r"efficientnet[- ]?b3", r"stronger.*efficientnet"],
            [r"temperature scal", r"post[- ]hoc calibration"],
            [r"(five|5)[- ]fold", r"across folds"],
        ],
        "attempted_pattern_signatures": [
            [[r"resnet18"], [r"cross[- ]attention"]],
            [[r"lightgbm"], [r"enhanced.*feature"]],
        ],
        "known_incompatibilities": [
            {
                "pattern_groups": [[r"pretrained.*efficientnet"], [r"download|remote"]],
                "resolution_pattern": r"local|offline|cached|available",
                "reason": "the frozen runtime cannot download a new backbone",
            }
        ],
    }


def _run_timestamp(run_id: str) -> float:
    match = run_id[:15]
    return dt.datetime.strptime(match, "%Y%m%d_%H%M%S").replace(
        tzinfo=dt.timezone.utc
    ).timestamp()


def _taxi_case() -> dict[str, Any]:
    manifest = _load(BASE / "recipe_distillation_v4_taxi/evidence_manifest.json")
    evidence = list(manifest["selected_evidence"]["new-york-city-taxi-fare-prediction"])
    hidden = next(
        item
        for item in evidence
        if item["node_id"].endswith("eeb6e2364829449ba6e1ce6c1600fc3d")
    )
    cutoff = _run_timestamp(str(hidden["run_id"])) - 1.0
    visible = [item for item in evidence if _run_timestamp(str(item["run_id"])) <= cutoff]
    visible.sort(key=lambda item: float(item.get("metric") or 1e9))
    parent_evidence = visible[0]
    capsules = _load(BASE / "recipe_distillation_v4_taxi/implementation_capsules.json")
    capsule_by_id = {
        str(item["node_id"]): item for item in capsules.get("nodes", [])
    }
    parent_capsule = capsule_by_id.get(str(parent_evidence["node_id"]), {})
    parent = {
        "node_id": str(parent_evidence["node_id"]),
        "code": str(parent_capsule.get("code") or ""),
        "code_summary": str(parent_evidence.get("code_summary") or ""),
        "plan": str(parent_evidence.get("plan") or ""),
        "metric": float(parent_evidence["metric"]),
        "maximize": False,
        "stage": str(parent_evidence.get("stage") or "draft"),
        "is_buggy": False,
        "execution_output": "",
    }
    events = [
        {
            "candidate_id": str(item["node_id"]),
            "source": "runforest",
            "source_stage": str(item.get("stage") or ""),
            "source_task_id": "new-york-city-taxi-fare-prediction",
            "metric": float(item["metric"]),
            "plan": str(item.get("plan") or ""),
            "text": str(item.get("code_summary") or ""),
            "rank_eligible": bool(item.get("rank_eligible", True)),
            "available_at": _run_timestamp(str(item["run_id"])),
        }
        for item in visible
    ]
    return {
        "case_id": "taxi-before-full-data-temporal-lightgbm",
        "task_id": "new-york-city-taxi-fare-prediction",
        "task_description": "Predict NYC taxi fare from pickup time, coordinates and passenger count; metric is RMSE.",
        "data_preview": "Tens of millions of training trips, a small official test set, temporal and geospatial covariates.",
        "stage": "improve",
        "cutoff": {"timestamp": cutoff, "meaning": "run timestamp before the hidden host-shadow best"},
        "parent": parent,
        "metrics": {"branch_best_metric": float(parent_evidence["metric"])},
        "budget": {
            "total_search_seconds": 21600,
            "elapsed_search_seconds": 9000,
            "remaining_search_seconds": 12600,
        },
        "memory_events": events,
        "hidden_future": {
            "id": str(hidden["node_id"]),
            "memory_ids": [str(hidden["node_id"])],
            "metric": float(hidden["metric"]),
            "method_summary": str(hidden.get("code_summary") or ""),
            "evaluator_only": True,
        },
        "expected_future_pattern_groups": [
            [r"lightgbm", r"lgbm"],
            [r"temporal.*(split|holdout)", r"chronological.*split", r"time[- ]based.*split"],
            [r"full.*(data|training)", r"all.*(rows|training)", r"large.*data"],
            [r"geo", r"haversine|airport|manhattan"],
        ],
        "attempted_pattern_signatures": [
            [[r"lightgbm"], [r"five[- ]fold|5[- ]fold"]],
            [[r"wide.*deep|mlp"], [r"time[- ]based|temporal"]],
        ],
        "known_incompatibilities": [
            {
                "pattern_groups": [[r"full.*data"], [r"five[- ]fold|5[- ]fold"]],
                "resolution_pattern": r"budget|sample|single.*holdout|temporal",
                "reason": "five full-data LightGBM fits are unlikely to finish inside one search child",
            }
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "memory_strategy_shadow_history_slices_v3.json",
    )
    args = parser.parse_args()
    packet = {
        "schema": "mlevolve_memory_strategy_history_slices_v3",
        "policy": {
            "hidden_future_not_model_visible": True,
            "strategy_evidence_limit": 8,
            "current_branch_frontier_slots": 3,
            "causal_failure_slots": 1,
            "historical_lineage_deduplication": True,
            "source": "existing Journal/RunForest/validated recipe evidence",
            "tasks": [
                "leaf-classification",
                "spooky-author-identification",
                "new-york-city-taxi-fare-prediction",
            ],
        },
        "cases": [
            _spooky_case(),
            _spooky_repair_case(),
            _leaf_case(),
            _taxi_case(),
        ],
    }
    _write(args.output, packet)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
