#!/usr/bin/env python3
"""External one-shot evaluator for T4 predictions.

Hidden labels are loaded only in this process.  Input records must contain
``sample_id`` and either ``prediction`` (regression) or ``probabilities``
(classification).  The output can then be passed to score_downstream.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from core import REPORTS, read_jsonl, write_jsonl


def _metric(name: str, labels: list[Any], predictions: list[Any]) -> float:
    if name == "rmse":
        return float(np.sqrt(np.mean((np.asarray(labels, dtype=float) - np.asarray(predictions, dtype=float)) ** 2)))
    if name == "log_loss":
        probs = np.asarray(predictions, dtype=float)
        if probs.ndim != 2:
            raise ValueError("log_loss predictions must be a 2D probability matrix")
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        probs /= probs.sum(axis=1, keepdims=True)
        y = np.asarray(labels, dtype=int)
        return float(-np.mean(np.log(probs[np.arange(len(y)), y])))
    raise ValueError(f"unsupported metric: {name}")


def evaluate(
    receipt_path: Path,
    hidden_manifest: Path,
    *,
    output_path: Path | None = None,
) -> list[dict[str, Any]]:
    receipts = read_jsonl(receipt_path)
    hidden = json.loads(hidden_manifest.read_text(encoding="utf-8"))
    scored = []
    for receipt in receipts:
        key = str(receipt["task_id"])
        task_hidden = hidden["tasks"][key]
        labels = {str(row["sample_id"]): row["label"] for row in read_jsonl(Path(task_hidden["labels_path"]))}
        prediction_path = Path(receipt["prediction_output_path"])
        try:
            rows = read_jsonl(prediction_path)
            ids = [str(row["sample_id"]) for row in rows]
            if len(ids) != len(set(ids)) or set(ids) != set(labels):
                raise ValueError("prediction sample_id coverage is not exactly one-to-one")
            target = [labels[sample_id] for sample_id in ids]
            values = [row["probabilities"] if task_hidden["metric"] == "log_loss" else row["prediction"] for row in rows]
            value = _metric(task_hidden["metric"], target, values)
            trusted = math.isfinite(value) and receipt.get("status") == "completed"
            reason = "" if trusted else "nonfinite_or_incomplete"
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            value = task_hidden.get("worst_valid_metric")
            trusted = False
            reason = f"external_evaluation_failed:{type(exc).__name__}:{exc}"
        analysis_eligible = isinstance(value, (int, float)) and math.isfinite(float(value))
        scored.append(
            {
                **receipt,
                "metric": value,
                "metric_name": task_hidden["metric"],
                "direction": task_hidden.get("direction", "minimize"),
                "trusted": trusted,
                "rank_eligible": trusted,
                "analysis_eligible": analysis_eligible,
                "failure_assigned_worst_valid_metric": not trusted and analysis_eligible,
                "external_evaluation_pending": False,
                "external_evaluation_reason": reason,
                "hidden_holdout_labels_exposed": False,
            }
        )
    path = output_path or REPORTS / "micro_scored_receipts_v1.jsonl"
    write_jsonl(path, scored)
    return scored


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--hidden-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPORTS / "micro_scored_receipts_v1.jsonl")
    args = parser.parse_args()
    rows = evaluate(args.receipts, args.hidden_manifest, output_path=args.output)
    print(json.dumps({"scored_count": len(rows), "trusted_count": sum(row["trusted"] for row in rows)}, ensure_ascii=False, indent=2))
