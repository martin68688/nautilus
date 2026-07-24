from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from evaluate_tier1_controlled_decisions import (
    DECISION_RECEIPT_SCHEMA,
    EVALUATION_REPORT_SCHEMA,
    _cell_metrics,
    _compose_systems,
    _gate_report,
    _primary_metrics,
)
from schema import sha256_json
from tier1_controlled_runtime import (
    CODE_EXECUTION_RECEIPT_SCHEMA,
    RUNTIME_ACTUATION_RECEIPT_SCHEMA,
    STATIC_ACTUATION_RECEIPT_SCHEMA,
)


VERIFICATION_SCHEMA = "decision_admissibility_tier1_evaluation_verification_v2"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) == sha256_json(
        {key: value for key, value in payload.items() if key != field}
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_evaluation(
    packet_root: str | Path,
    generation_root: str | Path,
    evaluation_root: str | Path,
) -> dict[str, Any]:
    packet_root = Path(packet_root).resolve()
    generation_root = Path(generation_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    errors: list[str] = []
    report = _read_json(evaluation_root / "evaluation_report.json")
    packet_manifest = _read_json(packet_root / "manifest.json")
    generation_report = _read_json(generation_root / "run_report.json")
    episodes = _read_jsonl(packet_root / str(packet_manifest["episode_file"]))
    episodes_by_id = {row["episode_id"]: row for row in episodes}
    decision_path = evaluation_root / str(report["decision_receipts_file"])
    counterfactual_path = evaluation_root / str(
        report["counterfactual_receipts_file"]
    )
    systems_path = evaluation_root / str(report["system_composition_file"])
    decisions = _read_jsonl(decision_path)
    counterfactuals = _read_jsonl(counterfactual_path)
    systems = _read_jsonl(systems_path)

    if report.get("schema") != EVALUATION_REPORT_SCHEMA:
        errors.append("report_schema")
    if not _valid_hash(report, "report_hash"):
        errors.append("report_hash")
    if report.get("packet_manifest_hash") != packet_manifest.get("manifest_hash"):
        errors.append("packet_manifest_binding")
    if report.get("generation_run_hash") != generation_report.get("run_hash"):
        errors.append("generation_run_binding")
    if _sha256_file(decision_path) != report.get("decision_receipts_file_sha256"):
        errors.append("decision_receipts_file_hash")
    if _sha256_file(counterfactual_path) != report.get(
        "counterfactual_receipts_file_sha256"
    ):
        errors.append("counterfactual_receipts_file_hash")
    if _sha256_file(systems_path) != report.get("system_composition_file_sha256"):
        errors.append("system_composition_file_hash")
    if len(decisions) != report.get("evaluated_decision_count"):
        errors.append("decision_count")
    if len({row.get("request_id") for row in decisions}) != len(decisions):
        errors.append("duplicate_decision_request_id")

    code_hashes: dict[str, str] = {}
    counterfactual_by_hash = {
        row.get("receipt_hash"): row for row in counterfactuals
    }
    independent_cold_result_count = 0
    for row in decisions:
        request_id = str(row.get("request_id") or "")
        if row.get("schema") != DECISION_RECEIPT_SCHEMA:
            errors.append(f"decision_schema:{request_id}")
        if not _valid_hash(row, "receipt_hash"):
            errors.append(f"decision_receipt_hash:{request_id}")
        execution = row.get("code_execution_receipt") or {}
        static = row.get("static_actuation_receipt") or {}
        runtime = row.get("runtime_actuation_receipt") or {}
        if execution.get("schema") != CODE_EXECUTION_RECEIPT_SCHEMA:
            errors.append(f"code_execution_schema:{request_id}")
        if static.get("schema") != STATIC_ACTUATION_RECEIPT_SCHEMA:
            errors.append(f"static_actuation_schema:{request_id}")
        if runtime.get("schema") != RUNTIME_ACTUATION_RECEIPT_SCHEMA:
            errors.append(f"runtime_actuation_schema:{request_id}")
        for name, receipt in (
            ("code_execution", execution),
            ("static_actuation", static),
            ("runtime_actuation", runtime),
        ):
            if not _valid_hash(receipt, "receipt_hash"):
                errors.append(f"{name}_receipt_hash:{request_id}")
        if runtime.get("code_execution_receipt_hash") != execution.get("receipt_hash"):
            errors.append(f"runtime_execution_binding:{request_id}")
        if runtime.get("static_receipt_hash") != static.get("receipt_hash"):
            errors.append(f"runtime_static_binding:{request_id}")
        if execution.get("execution_passed") is not True:
            errors.append(f"code_execution_failed:{request_id}")
        if row.get("current_run_node", {}).get("code_execution_receipt_hash") != execution.get(
            "receipt_hash"
        ):
            errors.append(f"current_node_execution_binding:{request_id}")
        promote = row.get("promote_result_path") or {}
        if promote.get("historical_actuation_required") is not False:
            errors.append(f"result_requires_actuation:{request_id}")
        if promote.get("derived_from_refs") != []:
            errors.append(f"result_has_historical_derivation:{request_id}")
        if promote.get("production_result_fact_written") is not False:
            errors.append(f"controlled_result_written_to_production:{request_id}")
        if (
            row.get("condition") == "NM"
            and execution.get("execution_passed") is True
            and static.get("static_actuation_passed") is False
            and runtime.get("runtime_actuation_passed") is False
            and row.get("current_run_node", {}).get("recordable") is True
        ):
            independent_cold_result_count += 1

        adoption = row.get("publish_adoption_path") or {}
        causal = row.get("publish_causal_path") or {}
        if adoption.get("eligible") and not (
            row.get("authority_valid") is True
            and row.get("granularity_match") is True
            and static.get("static_actuation_passed") is True
            and runtime.get("runtime_actuation_passed") is True
        ):
            errors.append(f"adoption_without_authority_or_actuation:{request_id}")
        if causal.get("eligible") and not adoption.get("eligible"):
            errors.append(f"causal_without_adoption:{request_id}")
        counterfactual_hash = str(row.get("counterfactual_receipt_hash") or "")
        if causal.get("eligible"):
            counterfactual = counterfactual_by_hash.get(counterfactual_hash)
            if not counterfactual or not counterfactual.get(
                "counterfactual_actuation_passed"
            ):
                errors.append(f"causal_without_counterfactual:{request_id}")

        code_relative = str(row.get("code_file") or "")
        code_path = evaluation_root / code_relative
        try:
            source = code_path.read_text(encoding="utf-8")
            ast.parse(source, mode="exec")
        except (FileNotFoundError, OSError, SyntaxError) as error:
            errors.append(f"code_read_or_parse:{request_id}:{type(error).__name__}")
            continue
        code_hash = _sha256_file(code_path)
        code_hashes[code_relative] = code_hash
        if code_hash != row.get("code_artifact", {}).get("source_sha256"):
            errors.append(f"code_source_hash:{request_id}")
        if "memory::" in source:
            errors.append(f"memory_metadata_in_code:{request_id}")
        if row.get("code_artifact", {}).get("memory_metadata_embedded") is not False:
            errors.append(f"memory_metadata_flag:{request_id}")

    if len(code_hashes) != report.get("code_file_count"):
        errors.append("code_file_count")
    if sha256_json(code_hashes) != report.get("code_file_hashes_hash"):
        errors.append("code_file_hashes_hash")
    if independent_cold_result_count == 0:
        errors.append("no_independent_cold_result_evidence")
    for row in counterfactuals:
        if not _valid_hash(row, "receipt_hash"):
            errors.append(
                f"counterfactual_receipt_hash:{row.get('memory_request_id')}"
            )

    recomposed, system_summaries = _compose_systems(
        decisions,
        episodes_by_id,
        require_full_matrix=bool(report.get("matrix_complete")),
    )
    if recomposed != systems:
        errors.append("system_composition_recompute")
    cell_metrics = _cell_metrics(decisions, counterfactuals)
    if cell_metrics != report.get("cell_metrics"):
        errors.append("cell_metrics_recompute")
    primary_metrics = _primary_metrics(decisions, counterfactuals)
    if primary_metrics != report.get("primary_metrics"):
        errors.append("primary_metrics_recompute")
    if system_summaries != report.get("system_summaries"):
        errors.append("system_summaries_recompute")
    gates = _gate_report(
        decisions,
        system_summaries,
        matrix_complete=bool(report.get("matrix_complete")),
    )
    if gates != report.get("kill_gates_1_to_4"):
        errors.append("kill_gates_recompute")

    evaluator_path = Path(__file__).resolve().with_name(
        "evaluate_tier1_controlled_decisions.py"
    )
    runtime_path = Path(__file__).resolve().with_name("tier1_controlled_runtime.py")
    if _sha256_file(evaluator_path) != report.get("evaluator_source_sha256"):
        errors.append("evaluator_source_hash")
    if _sha256_file(runtime_path) != report.get("runtime_source_sha256"):
        errors.append("runtime_source_hash")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "evaluation_root_name": evaluation_root.name,
        "matrix_complete": bool(report.get("matrix_complete")),
        "decision_count": len(decisions),
        "counterfactual_count": len(counterfactuals),
        "system_composition_count": len(systems),
        "code_file_count": len(code_hashes),
        "independent_cold_result_evidence_count": independent_cold_result_count,
        "code_execution_schema_distinct_from_static_actuation": (
            CODE_EXECUTION_RECEIPT_SCHEMA != STATIC_ACTUATION_RECEIPT_SCHEMA
        ),
        "code_execution_schema_distinct_from_runtime_actuation": (
            CODE_EXECUTION_RECEIPT_SCHEMA != RUNTIME_ACTUATION_RECEIPT_SCHEMA
        ),
        "verified": not errors,
        "errors": sorted(set(errors)),
        "evaluation_report_hash": report.get("report_hash", ""),
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
        description="Verify host-owned WP8 Tier-1 evaluation artifacts."
    )
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--generation-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_evaluation(
        args.packet_root,
        args.generation_root,
        args.evaluation_root,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
