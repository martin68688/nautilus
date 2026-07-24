"""Independently recompute and verify WP8 Tier-2 formal statistics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from analyze_tier2_formal_statistics import (
    MANIFEST_SCHEMA,
    OBSERVATION_SCHEMA,
    ORACLE_GAP_SCHEMA,
    ORACLE_SCHEMA,
    PAIR_SCHEMA,
    REPORT_SCHEMA,
    _json_text,
    _payload_hash,
    _sha256_file,
    compute_statistics,
)


VERIFICATION_SCHEMA = "decision_admissibility_wp8_tier2_formal_statistics_verification_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in: {path}")
    return rows


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_json_text(payload))
        handle.flush()
        os.fsync(handle.fileno())


def verify_statistics(
    *,
    statistics_root: str | Path,
    analysis_policy_path: str | Path,
    inventory_root: str | Path,
    inventory_verification_path: str | Path,
    completed_root: str | Path,
    continuation_root: str | Path,
) -> dict[str, Any]:
    statistics_root = Path(statistics_root).resolve()
    report_path = statistics_root / "statistics_report.json"
    manifest_path = statistics_root / "analysis_manifest.json"
    file_map = {
        "formal_observations.jsonl": statistics_root / "formal_observations.jsonl",
        "oracle_observations.jsonl": statistics_root / "oracle_observations.jsonl",
        "paired_contrasts.jsonl": statistics_root / "paired_contrasts.jsonl",
        "oracle_gaps.jsonl": statistics_root / "oracle_gaps.jsonl",
        "statistics_report.json": report_path,
    }
    errors: list[str] = []
    try:
        report = _read_object(report_path)
        manifest = _read_object(manifest_path)
        observations = _read_jsonl(file_map["formal_observations.jsonl"])
        oracle_rows = _read_jsonl(file_map["oracle_observations.jsonl"])
        pairs = _read_jsonl(file_map["paired_contrasts.jsonl"])
        oracle_gaps = _read_jsonl(file_map["oracle_gaps.jsonl"])
    except Exception as error:
        report = {}
        manifest = {}
        observations = []
        oracle_rows = []
        pairs = []
        oracle_gaps = []
        errors.append(f"statistics_read:{type(error).__name__}")

    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "complete":
        errors.append("report_schema_or_status")
    if report.get("report_hash") != _payload_hash(report, "report_hash"):
        errors.append("report_hash")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "complete":
        errors.append("manifest_schema_or_status")
    if manifest.get("manifest_hash") != _payload_hash(manifest, "manifest_hash"):
        errors.append("manifest_hash")
    for filename, path in file_map.items():
        if not path.is_file() or (manifest.get("files") or {}).get(filename) != _sha256_file(path):
            errors.append(f"file_hash:{filename}")
    if manifest.get("statistics_report_hash") != report.get("report_hash"):
        errors.append("manifest_report_binding")

    typed_rows = (
        (observations, OBSERVATION_SCHEMA, "observation"),
        (oracle_rows, ORACLE_SCHEMA, "oracle"),
        (pairs, PAIR_SCHEMA, "pair"),
        (oracle_gaps, ORACLE_GAP_SCHEMA, "oracle_gap"),
    )
    for rows, schema, label in typed_rows:
        for index, row in enumerate(rows):
            if row.get("schema") != schema:
                errors.append(f"{label}_schema:{index}")
            if row.get("row_hash") != _payload_hash(row, "row_hash"):
                errors.append(f"{label}_hash:{index}")
    if len(observations) != 45:
        errors.append("observation_count")
    if len(oracle_rows) != 9:
        errors.append("oracle_count")
    if len(pairs) != 36:
        errors.append("pair_count")
    if len(oracle_gaps) != 45:
        errors.append("oracle_gap_count")
    if sum(bool(row.get("completed")) for row in observations) + sum(
        not bool(row.get("completed")) for row in observations
    ) != 45:
        errors.append("disposition_partition")
    if any(row.get("imputed") is not False or row.get("excluded") is not False for row in observations):
        errors.append("imputation_or_exclusion")

    analyzer_path = Path(__file__).resolve().with_name(
        "analyze_tier2_formal_statistics.py"
    )
    analyzer_sha = _sha256_file(analyzer_path)
    if report.get("analyzer_source_sha256") != analyzer_sha:
        errors.append("analyzer_source_hash")
    if manifest.get("analyzer_source_sha256") != analyzer_sha:
        errors.append("manifest_analyzer_source_hash")

    try:
        recomputed = compute_statistics(
            analysis_policy_path=analysis_policy_path,
            inventory_root=inventory_root,
            inventory_verification_path=inventory_verification_path,
            completed_root=completed_root,
            continuation_root=continuation_root,
            created_at=str(report.get("created_at") or ""),
        )
    except Exception as error:
        recomputed = None
        errors.append(f"statistics_recompute:{type(error).__name__}")
    if recomputed is not None:
        expected_report, expected_observations, expected_oracle, expected_pairs, expected_gaps = recomputed
        if expected_report != report:
            errors.append("report_recompute_mismatch")
        if expected_observations != observations:
            errors.append("observations_recompute_mismatch")
        if expected_oracle != oracle_rows:
            errors.append("oracle_recompute_mismatch")
        if expected_pairs != pairs:
            errors.append("pairs_recompute_mismatch")
        if expected_gaps != oracle_gaps:
            errors.append("oracle_gaps_recompute_mismatch")

    population = report.get("analysis_population") or {}
    gate = report.get("effect_claim_gate") or {}
    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed" if not errors else "failed",
        "verified": not errors,
        "errors": sorted(set(errors)),
        "assigned_online_outcomes": population.get("assigned_online_outcomes"),
        "scored_selected_results": population.get("scored_selected_results"),
        "failed_online_conditions": population.get("failed_online_conditions"),
        "assigned_oracle_dispositions": population.get("assigned_oracle_dispositions"),
        "imputed_scores": population.get("imputed_scores"),
        "post_assignment_exclusions": population.get("post_assignment_exclusions"),
        "effect_claim_authorized": gate.get("effect_claim_authorized"),
        "statistics_report_hash": report.get("report_hash", ""),
        "statistics_manifest_hash": manifest.get("manifest_hash", ""),
        "analyzer_source_sha256": analyzer_sha,
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    verification["verification_hash"] = _payload_hash(
        verification, "verification_hash"
    )
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--analysis-policy", required=True, type=Path)
    parser.add_argument("--inventory-root", required=True, type=Path)
    parser.add_argument("--inventory-verification", required=True, type=Path)
    parser.add_argument("--completed-root", required=True, type=Path)
    parser.add_argument("--continuation-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = verify_statistics(
        statistics_root=args.statistics_root,
        analysis_policy_path=args.analysis_policy,
        inventory_root=args.inventory_root,
        inventory_verification_path=args.inventory_verification,
        completed_root=args.completed_root,
        continuation_root=args.continuation_root,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), result)
    print(_json_text(result), end="")
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

