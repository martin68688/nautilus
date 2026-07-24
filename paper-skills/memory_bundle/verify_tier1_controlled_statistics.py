from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from analyze_tier1_controlled_statistics import (
    SEED_DELTA_SCHEMA,
    STATISTICS_REPORT_SCHEMA,
    compute_statistics,
)
from schema import sha256_json


VERIFICATION_SCHEMA = "decision_admissibility_tier1_statistics_verification_v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) == sha256_json(
        {key: value for key, value in payload.items() if key != field}
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2)
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_statistics(
    packet_root: str | Path,
    generation_root: str | Path,
    evaluation_root: str | Path,
    statistics_root: str | Path,
) -> dict[str, Any]:
    statistics_root = Path(statistics_root).resolve()
    report_path = statistics_root / "statistics_report.json"
    report = _read_json(report_path)
    seed_path = statistics_root / str(report.get("paired_seed_deltas_file") or "")
    seed_rows = _read_jsonl(seed_path)
    errors: list[str] = []

    if report.get("schema") != STATISTICS_REPORT_SCHEMA:
        errors.append("report_schema")
    if not _valid_hash(report, "report_hash"):
        errors.append("report_hash")
    analyzer_path = Path(__file__).resolve().with_name(
        "analyze_tier1_controlled_statistics.py"
    )
    if report.get("analyzer_source_sha256") != _sha256_file(analyzer_path):
        errors.append("analyzer_source_hash")
    if report.get("paired_seed_deltas_file_sha256") != _sha256_file(seed_path):
        errors.append("paired_seed_deltas_file_hash")
    if report.get("paired_seed_delta_count") != len(seed_rows):
        errors.append("paired_seed_delta_count")
    if len(
        {
            (row.get("metric"), row.get("agent_replicate_id"))
            for row in seed_rows
        }
    ) != len(seed_rows):
        errors.append("duplicate_seed_delta")
    for row in seed_rows:
        identity = f"{row.get('metric')}:{row.get('agent_replicate_id')}"
        if row.get("schema") != SEED_DELTA_SCHEMA:
            errors.append(f"seed_delta_schema:{identity}")
        if not _valid_hash(row, "row_hash"):
            errors.append(f"seed_delta_hash:{identity}")

    bootstrap = report.get("paired_bootstrap") or {}
    try:
        recomputed_report, recomputed_seed_rows = compute_statistics(
            packet_root,
            generation_root,
            evaluation_root,
            created_at=str(report.get("created_at") or ""),
            bootstrap_iterations=int(bootstrap.get("iterations", 0)),
            bootstrap_seed=int(bootstrap.get("host_rng_seed", 0)),
        )
    except Exception as error:  # fail-closed verification surface
        errors.append(f"statistics_recompute_exception:{type(error).__name__}")
        recomputed_report = None
        recomputed_seed_rows = None
    if recomputed_report is not None and recomputed_report != report:
        errors.append("statistics_report_recompute")
    if recomputed_seed_rows is not None and recomputed_seed_rows != seed_rows:
        errors.append("paired_seed_deltas_recompute")

    effects = report.get("effect_estimates") or {}
    exact_tests = (
        report.get("paired_exact_tests_holm_family", {}).get("tests") or {}
    )
    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "statistics_root_name": statistics_root.name,
        "paired_decision_count": report.get("analysis_unit", {}).get(
            "paired_decision_count"
        ),
        "bootstrap_iterations": bootstrap.get("iterations"),
        "paired_seed_delta_count": len(seed_rows),
        "primary_raw_iir_numerator": effects.get("primary_raw_cell_iir", {}).get(
            "numerator"
        ),
        "primary_raw_iir_denominator": effects.get(
            "primary_raw_cell_iir", {}
        ).get("denominator"),
        "primary_vkr_numerator": effects.get("primary_f11_vkr", {}).get(
            "numerator"
        ),
        "primary_vkr_denominator": effects.get("primary_f11_vkr", {}).get(
            "denominator"
        ),
        "holm_rejected_test_count": sum(
            row.get("reject_at_familywise_alpha_0_05") is True
            for row in exact_tests.values()
        ),
        "verified": not errors,
        "errors": sorted(set(errors)),
        "statistics_report_hash": report.get("report_hash", ""),
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    verification["verification_hash"] = sha256_json(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_hash"
        }
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify WP8 Tier-1 paired and mixed-effects statistics."
    )
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verification = verify_statistics(
        args.packet_root,
        args.generation_root,
        args.evaluation_root,
        args.statistics_root,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False, indent=2))
    if not verification["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
