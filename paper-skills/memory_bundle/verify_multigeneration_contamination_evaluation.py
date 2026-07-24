from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from evaluate_multigeneration_contamination import (
    DESCENDANT_RECEIPT_SCHEMA,
    EVALUATION_REPORT_SCHEMA,
    SYSTEM_RECEIPT_SCHEMA,
    SYSTEMS,
    evaluate_rows,
)
from schema import sha256_json
from verify_multigeneration_paraphrases import verify_run


VERIFICATION_SCHEMA = "decision_admissibility_multigeneration_evaluation_verification_v2"


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


def verify_evaluation(
    work_root: str | Path,
    packet_root: str | Path,
    run_root: str | Path,
    evaluation_root: str | Path,
) -> dict[str, Any]:
    packet_root = Path(packet_root).resolve()
    run_root = Path(run_root).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    errors: list[str] = []
    run_verification = verify_run(work_root, packet_root, run_root)
    manifest = _read_json(packet_root / "manifest.json")
    pairs = _read_jsonl(packet_root / str(manifest["pair_file"]))
    run_report = _read_json(run_root / "run_report.json")
    responses = _read_jsonl(run_root / str(run_report["responses_file"]))
    report = _read_json(evaluation_root / "evaluation_report.json")
    descendant_path = evaluation_root / str(report["descendant_receipts_file"])
    system_path = evaluation_root / str(report["system_receipts_file"])
    descendants = _read_jsonl(descendant_path)
    systems = _read_jsonl(system_path)
    if run_verification.get("verified") is not True:
        errors.append("run_verification")
    if report.get("schema") != EVALUATION_REPORT_SCHEMA:
        errors.append("report_schema")
    if not _valid_hash(report, "report_hash"):
        errors.append("report_hash")
    if report.get("packet_manifest_hash") != manifest.get("manifest_hash"):
        errors.append("packet_binding")
    if report.get("run_hash") != run_report.get("run_hash"):
        errors.append("run_binding")
    if report.get("run_verification_hash") != run_verification.get(
        "verification_hash"
    ):
        errors.append("run_verification_binding")
    if report.get("descendant_receipts_file_sha256") != _sha256_file(
        descendant_path
    ):
        errors.append("descendant_file_hash")
    if report.get("system_receipts_file_sha256") != _sha256_file(system_path):
        errors.append("system_file_hash")
    if len(descendants) != report.get("descendant_receipt_count"):
        errors.append("descendant_count")
    if len(systems) != report.get("system_receipt_count"):
        errors.append("system_count")
    if report.get("systems") != list(SYSTEMS):
        errors.append("system_list")
    if len({row.get("request_id") for row in descendants}) != len(descendants):
        errors.append("duplicate_descendant_request")
    if len(
        {(row.get("request_id"), row.get("system")) for row in systems}
    ) != len(systems):
        errors.append("duplicate_system_request")
    descendant_hashes = {row.get("receipt_hash") for row in descendants}
    for row in descendants:
        request_id = str(row.get("request_id") or "")
        if row.get("schema") != DESCENDANT_RECEIPT_SCHEMA:
            errors.append(f"descendant_schema:{request_id}")
        if not _valid_hash(row, "receipt_hash"):
            errors.append(f"descendant_hash:{request_id}")
        if row.get("target_history_exposed") is not False:
            errors.append(f"target_history:{request_id}")
        if row.get("cross_domain_transfer") is not False:
            errors.append(f"cross_domain:{request_id}")
        if row.get("valid_lineage_decision", {}).get("allowed") is not True:
            errors.append(f"valid_lineage:{request_id}")
        if row.get("invalid_lineage_decision", {}).get("allowed") is not False:
            errors.append(f"invalid_lineage:{request_id}")
    for row in systems:
        identity = f"{row.get('request_id')}:{row.get('system')}"
        if row.get("schema") != SYSTEM_RECEIPT_SCHEMA:
            errors.append(f"system_schema:{identity}")
        if not _valid_hash(row, "receipt_hash"):
            errors.append(f"system_hash:{identity}")
        if row.get("descendant_receipt_hash") not in descendant_hashes:
            errors.append(f"system_descendant_binding:{identity}")
        if row.get("identical_frozen_descendant_dag") is not True:
            errors.append(f"system_dag:{identity}")
        if row.get("laundering_success") != row.get(
            "unauthorized_prompt_exposure"
        ):
            errors.append(f"laundering_exposure_semantics:{identity}")
        if row.get("current_surface_detector_applied") != (
            row.get("system") == "authority_only"
        ):
            errors.append(f"surface_ablation_scope:{identity}")
        if row.get("source_claim_authority_applied") != (
            row.get("system") == "full_decision_admissibility"
        ):
            errors.append(f"source_authority_scope:{identity}")
    recomputed_descendants, recomputed_systems, recomputed_summaries = evaluate_rows(
        pairs, responses
    )
    if recomputed_descendants != descendants:
        errors.append("descendants_recompute")
    if recomputed_systems != systems:
        errors.append("systems_recompute")
    if recomputed_summaries != report.get("system_summaries_by_generation"):
        errors.append("summaries_recompute")
    evaluator_path = Path(__file__).resolve().with_name(
        "evaluate_multigeneration_contamination.py"
    )
    if report.get("evaluator_source_sha256") != _sha256_file(evaluator_path):
        errors.append("evaluator_source_hash")
    if report.get("post_hoc_descendant_exclusion_count") != 0:
        errors.append("post_hoc_exclusions")
    if report.get("target_history_exposure_count") != 0:
        errors.append("target_history_count")
    if report.get("cross_domain_transfer_count") != 0:
        errors.append("cross_domain_count")
    if report.get("identical_frozen_descendant_dag_for_all_systems") is not True:
        errors.append("frozen_dag_boundary")
    if report.get("provider_rng_seed_claimed") is not False:
        errors.append("provider_seed_boundary")

    final_generation = str(manifest["generation_count"])
    summaries = report.get("system_summaries_by_generation") or {}
    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "evaluation_root_name": evaluation_root.name,
        "source_pair_count": len(pairs),
        "source_run_count": manifest.get("source_run_count"),
        "generation_count": manifest.get("generation_count"),
        "descendant_receipt_count": len(descendants),
        "system_receipt_count": len(systems),
        "full_final_laundering_count": summaries.get(
            "full_decision_admissibility", {}
        ).get(final_generation, {}).get("laundering_success_count"),
        "full_final_vkr_count": summaries.get(
            "full_decision_admissibility", {}
        ).get(final_generation, {}).get("valid_knowledge_retained_count"),
        "verified": not errors,
        "errors": sorted(set(errors)),
        "run_verification_hash": run_verification.get("verification_hash", ""),
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
        description="Verify WP8 Multi-generation policy evaluation."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verification = verify_evaluation(
        args.work_root,
        args.packet_root,
        args.run_root,
        args.evaluation_root,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False, indent=2))
    if not verification["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
