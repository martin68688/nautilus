#!/usr/bin/env python3
"""Build auditable Recipe evidence from retained End2End Smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mlevolve") not in sys.path:
    sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.leakage_audit import audit_code  # noqa: E402


SCHEMA = "mlevolve_end2end_incremental_recipe_evidence_v1"
METHOD_SIGNALS = (
    "siglip",
    "dinov2",
    "efficientnet",
    "convnext",
    "resnet",
    "xgboost",
    "xgbclassifier",
    "lightgbm",
    "lgbmclassifier",
    "catboost",
    "randomforest",
    "extratrees",
    "logisticregression",
    "standardscaler",
    "pca",
    "stratifiedkfold",
    "kfold",
    "train_test_split",
    "mixup",
    "label_smoothing",
    "early stopping",
    "optimize",
    "ensemble",
    "blend",
    "oof",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _node_metric(node: Mapping[str, Any]) -> float | None:
    metric = node.get("metric")
    value = metric.get("value") if isinstance(metric, Mapping) else metric
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize_code(code: str, plan: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", " ", code.lower())
    signals = [signal for signal in METHOD_SIGNALS if signal.replace(" ", "_") in normalized or signal in code.lower()]
    imports = []
    for line in code.splitlines():
        match = re.match(r"\s*(?:from|import)\s+([a-zA-Z0-9_.]+)", line)
        if match:
            imports.append(match.group(1).split(".")[0])
    plan = re.sub(r"\s+", " ", str(plan or "")).strip()
    return (
        f"Plan: {plan} Implementation signals: {', '.join(dict.fromkeys(signals)) or 'none detected'}. "
        f"Primary libraries: {', '.join(list(dict.fromkeys(imports))[:12]) or 'none detected'}."
    )


def build(
    input_root: Path, *, excluded_logical_run_ids: set[str] | None = None
) -> dict[str, Any]:
    input_root = input_root.resolve(strict=True)
    excluded_logical_run_ids = set(excluded_logical_run_ids or set())
    records: list[dict[str, Any]] = []
    attempts_seen = 0
    terminal_reports_seen = 0
    missing_nodes: list[dict[str, str]] = []
    for report_path in sorted(input_root.rglob("TERMINAL_SCORE_REPORT.json")):
        terminal_reports_seen += 1
        attempt_root = report_path.parent
        measurement_path = attempt_root / "MEASUREMENT.json"
        if not measurement_path.exists():
            continue
        attempts_seen += 1
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        if measurement.get("status") != "scored_terminal_result":
            continue
        if str(measurement.get("logical_run_id") or "") in excluded_logical_run_ids:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        journals = list((attempt_root / "agent" / "logs").glob("*/journal.json"))
        if len(journals) != 1:
            raise ValueError(f"Expected one journal for scored attempt: {attempt_root}")
        journal_path = journals[0]
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        nodes = {
            str(node.get("id")): node
            for node in journal.get("nodes") or []
            if isinstance(node, Mapping) and node.get("id")
        }
        selected_id = str(report.get("selected_node_id") or "")
        for result in report.get("results") or []:
            if not isinstance(result, Mapping) or result.get("status") != "scored":
                continue
            node_id = str(result.get("node_id") or "")
            if node_id != selected_id:
                continue
            node = nodes.get(node_id)
            if node is None:
                missing_nodes.append({"report": str(report_path), "node_id": node_id})
                continue
            code = str(node.get("code") or "")
            if not code.strip():
                missing_nodes.append({"report": str(report_path), "node_id": node_id})
                continue
            audit = audit_code(code)
            terminal_score = result.get("score")
            records.append(
                {
                    "node_id": f"postsmoke::{measurement['logical_run_id']}::{node_id}",
                    "source_node_id": node_id,
                    "task_id": str(measurement["task_id"]),
                    "run_id": f"{measurement['logical_run_id']}::attempt-{attempt_root.name.split('-')[-1]}",
                    "system_id": str(measurement["system_id"]),
                    "stage": str(node.get("stage") or result.get("stage") or "draft"),
                    "step": node.get("step"),
                    "metric": float(terminal_score),
                    "metric_improvement": None,
                    "metric_provenance": "sealed_fixed_holdout_terminal_score",
                    "internal_metric": _node_metric(node),
                    "is_buggy": bool(node.get("is_buggy")),
                    "is_valid": bool(node.get("is_valid")),
                    "plan": str(node.get("plan") or ""),
                    "code_summary": summarize_code(code, str(node.get("plan") or "")),
                    "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
                    "leakage_audit": audit,
                    "selected_for_terminal": node_id == selected_id,
                    "source_cohort": "post_freeze_leaf_smoke_20260805",
                    "source_files": {
                        "measurement": str(measurement_path.relative_to(input_root)),
                        "measurement_sha256": sha256_file(measurement_path),
                        "terminal_report": str(report_path.relative_to(input_root)),
                        "terminal_report_sha256": sha256_file(report_path),
                        "journal": str(journal_path.relative_to(input_root)),
                        "journal_sha256": sha256_file(journal_path),
                    },
                }
            )
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record["leakage_audit"].get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    output: dict[str, Any] = {
        "schema": SCHEMA,
        "input_root": str(input_root),
        "input_archive_sha256": (
            sha256_file(input_root / "end2end-recipe-records.tgz")
            if (input_root / "end2end-recipe-records.tgz").exists()
            else None
        ),
        "terminal_reports_seen": terminal_reports_seen,
        "scored_attempts_seen": attempts_seen,
        "record_count": len(records),
        "candidate_policy": "selected_terminal_program_only",
        "excluded_logical_run_ids": sorted(excluded_logical_run_ids),
        "audit_status_counts": dict(sorted(status_counts.items())),
        "missing_node_count": len(missing_nodes),
        "missing_nodes": missing_nodes,
        "records": records,
        "manifest_sha256": "",
    }
    output["manifest_sha256"] = payload_hash(output, "manifest_sha256")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-logical-run-id", action="append", default=[])
    args = parser.parse_args()
    result = build(
        args.input_root,
        excluded_logical_run_ids=set(args.exclude_logical_run_id),
    )
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "record_count": result["record_count"],
                "audit_status_counts": result["audit_status_counts"],
                "missing_node_count": result["missing_node_count"],
                "manifest_sha256": result["manifest_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
