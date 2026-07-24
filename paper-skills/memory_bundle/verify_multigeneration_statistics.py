from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from analyze_multigeneration_statistics import (
    SEED_DELTA_SCHEMA,
    STATISTICS_REPORT_SCHEMA,
    compute_statistics,
)
from schema import sha256_json


VERIFICATION_SCHEMA = "decision_admissibility_multigeneration_statistics_verification_v1"


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
    work_root: str | Path,
    packet_root: str | Path,
    run_root: str | Path,
    evaluation_root: str | Path,
    statistics_root: str | Path,
) -> dict[str, Any]:
    statistics_root = Path(statistics_root).resolve()
    report = _read_json(statistics_root / "statistics_report.json")
    seed_path = statistics_root / str(report.get("paired_seed_deltas_file") or "")
    seed_rows = _read_jsonl(seed_path)
    errors: list[str] = []
    if report.get("schema") != STATISTICS_REPORT_SCHEMA:
        errors.append("report_schema")
    if not _valid_hash(report, "report_hash"):
        errors.append("report_hash")
    analyzer_path = Path(__file__).resolve().with_name(
        "analyze_multigeneration_statistics.py"
    )
    if report.get("analyzer_source_sha256") != _sha256_file(analyzer_path):
        errors.append("analyzer_source_hash")
    if report.get("paired_seed_deltas_file_sha256") != _sha256_file(seed_path):
        errors.append("seed_delta_file_hash")
    if report.get("paired_seed_delta_count") != len(seed_rows):
        errors.append("seed_delta_count")
    if len(
        {
            (row.get("metric"), row.get("paraphrase_replicate_id"))
            for row in seed_rows
        }
    ) != len(seed_rows):
        errors.append("duplicate_seed_delta")
    for row in seed_rows:
        identity = f"{row.get('metric')}:{row.get('paraphrase_replicate_id')}"
        if row.get("schema") != SEED_DELTA_SCHEMA:
            errors.append(f"seed_delta_schema:{identity}")
        if not _valid_hash(row, "row_hash"):
            errors.append(f"seed_delta_hash:{identity}")
    bootstrap = report.get("paired_bootstrap") or {}
    try:
        recomputed_report, recomputed_seed_rows = compute_statistics(
            work_root,
            packet_root,
            run_root,
            evaluation_root,
            created_at=str(report.get("created_at") or ""),
            bootstrap_iterations=int(bootstrap.get("iterations", 0)),
            bootstrap_seed=int(bootstrap.get("host_rng_seed", 0)),
        )
    except Exception as error:
        errors.append(f"statistics_recompute_exception:{type(error).__name__}")
        recomputed_report = None
        recomputed_seed_rows = None
    if recomputed_report is not None and recomputed_report != report:
        errors.append("statistics_report_recompute")
    if recomputed_seed_rows is not None and recomputed_seed_rows != seed_rows:
        errors.append("seed_deltas_recompute")
    gate = report.get("gate_5") or {}
    if gate.get("passed") is not True or gate.get("status") != "pass":
        errors.append("gate_5_not_passed")
    if not all((gate.get("checks") or {}).values()):
        errors.append("gate_5_check_failure")
    effects = report.get("final_generation_effects") or {}
    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "statistics_root_name": statistics_root.name,
        "source_pair_count": report.get("analysis_unit", {}).get(
            "source_pair_count"
        ),
        "source_run_count": report.get("analysis_unit", {}).get(
            "source_run_count"
        ),
        "paired_chain_count": report.get("analysis_unit", {}).get(
            "paired_chain_count"
        ),
        "bootstrap_iterations": bootstrap.get("iterations"),
        "gate_5_passed": gate.get("passed"),
        "full_final_laundering_numerator": effects.get(
            "full_laundering", {}
        ).get("numerator"),
        "full_final_laundering_denominator": effects.get(
            "full_laundering", {}
        ).get("denominator"),
        "full_final_vkr_numerator": effects.get("full_vkr", {}).get(
            "numerator"
        ),
        "full_final_vkr_denominator": effects.get("full_vkr", {}).get(
            "denominator"
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
        description="Verify WP8 Multi-generation Gate-5 statistics."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verification = verify_statistics(
        args.work_root,
        args.packet_root,
        args.run_root,
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
