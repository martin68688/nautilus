from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from run_multigeneration_paraphrases import (
    ATTEMPT_LOG_SCHEMA,
    RAW_RESPONSE_SCHEMA,
    REQUEST_PLAN_SCHEMA,
    RUN_REPORT_SCHEMA,
    _materialize_request,
    parse_paraphrase_response,
)
from schema import sha256_json
from verify_multigeneration_contamination_packet import verify_packet


VERIFICATION_SCHEMA = "decision_admissibility_multigeneration_run_verification_v1"


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


def verify_run(
    work_root: str | Path,
    packet_root: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    packet_root = Path(packet_root).resolve()
    run_root = Path(run_root).resolve()
    errors: list[str] = []
    packet_verification = verify_packet(work_root, packet_root)
    manifest = _read_json(packet_root / "manifest.json")
    pairs = _read_jsonl(packet_root / str(manifest["pair_file"]))
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    plan_path = run_root / "request_plan.json"
    report_path = run_root / "run_report.json"
    plan = _read_json(plan_path)
    report = _read_json(report_path)
    responses_path = run_root / str(report.get("responses_file") or "")
    responses = _read_jsonl(responses_path)
    if packet_verification.get("verified") is not True:
        errors.append("packet_verification")
    if plan.get("schema") != REQUEST_PLAN_SCHEMA:
        errors.append("request_plan_schema")
    if not _valid_hash(plan, "request_plan_hash"):
        errors.append("request_plan_hash")
    if report.get("schema") != RUN_REPORT_SCHEMA:
        errors.append("run_report_schema")
    if not _valid_hash(report, "run_hash"):
        errors.append("run_hash")
    if plan.get("packet_manifest_hash") != manifest.get("manifest_hash"):
        errors.append("plan_packet_binding")
    if report.get("packet_manifest_hash") != manifest.get("manifest_hash"):
        errors.append("report_packet_binding")
    if report.get("request_plan_hash") != plan.get("request_plan_hash"):
        errors.append("report_plan_binding")
    runner_path = Path(__file__).resolve().with_name(
        "run_multigeneration_paraphrases.py"
    )
    if plan.get("runner_source_sha256") != _sha256_file(runner_path):
        errors.append("plan_runner_source_hash")
    if report.get("runner_source_sha256") != _sha256_file(runner_path):
        errors.append("report_runner_source_hash")
    if report.get("responses_file_sha256") != _sha256_file(responses_path):
        errors.append("responses_file_hash")
    if len(responses) != plan.get("request_count"):
        errors.append("response_count")
    if report.get("response_count") != len(responses):
        errors.append("report_response_count")
    if report.get("error_count") != 0:
        errors.append("run_errors")
    if report.get("provider_seed_parameter_sent") is not False:
        errors.append("provider_seed_boundary")
    if report.get("descendant_dag_frozen_before_system_evaluation") is not True:
        errors.append("descendant_freeze")
    if len({row.get("request_id") for row in responses}) != len(responses):
        errors.append("duplicate_response_id")
    response_by_id = {row["request_id"]: row for row in responses}
    schedule_ids = [row["request_id"] for row in plan.get("request_schedule") or []]
    if set(response_by_id) != set(schedule_ids):
        errors.append("response_schedule_set")

    raw_hashes = {
        path.name: _sha256_file(path)
        for path in sorted((run_root / "raw_responses").glob("*.json"))
    }
    attempt_hashes = {
        path.name: _sha256_file(path)
        for path in sorted((run_root / "attempt_logs").glob("*.json"))
    }
    if raw_hashes != report.get("raw_response_file_hashes"):
        errors.append("raw_response_file_hashes")
    if attempt_hashes != report.get("attempt_log_file_hashes"):
        errors.append("attempt_log_file_hashes")
    if len(raw_hashes) != report.get("raw_response_file_count"):
        errors.append("raw_response_file_count")
    if len(attempt_hashes) != report.get("attempt_log_file_count"):
        errors.append("attempt_log_file_count")

    schedule_by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in plan.get("request_schedule") or []:
        schedule_by_generation[int(row["generation"])].append(dict(row))
    parent_by_chain: dict[tuple[str, int], Mapping[str, Any]] = {}
    generation_counts: Counter[int] = Counter()
    provider_seed_violation_count = 0
    prompt_target_history_count = 0
    for generation in range(1, int(plan.get("generation_count") or 0) + 1):
        for schedule in schedule_by_generation[generation]:
            pair_id = str(schedule["pair_id"])
            replicate = int(schedule["paraphrase_replicate_id"])
            pair = pairs_by_id[pair_id]
            expected_request = _materialize_request(
                schedule,
                pair,
                plan,
                parent_record=parent_by_chain.get((pair_id, replicate)),
            )
            request_id = str(schedule["request_id"])
            record = response_by_id.get(request_id)
            if record is None:
                continue
            raw_path = run_root / "raw_responses" / f"{request_id}.json"
            if _read_json(raw_path) != record:
                errors.append(f"raw_response_binding:{request_id}")
            if record.get("schema") != RAW_RESPONSE_SCHEMA:
                errors.append(f"raw_response_schema:{request_id}")
            if not _valid_hash(record, "record_hash"):
                errors.append(f"raw_response_hash:{request_id}")
            if record.get("identity_hash") != expected_request["identity_hash"]:
                errors.append(f"request_identity:{request_id}")
            for field in (
                "parent_record_hash",
                "valid_parent_ref",
                "invalid_parent_ref",
                "valid_parent_text_sha256",
                "invalid_parent_text_sha256",
            ):
                if record.get(field) != expected_request[field]:
                    errors.append(f"parent_binding:{request_id}:{field}")
            if record.get("system_prompt_sha256") != hashlib.sha256(
                expected_request["system_prompt"].encode("utf-8")
            ).hexdigest():
                errors.append(f"system_prompt_hash:{request_id}")
            if record.get("user_prompt_sha256") != hashlib.sha256(
                expected_request["user_prompt"].encode("utf-8")
            ).hexdigest():
                errors.append(f"user_prompt_hash:{request_id}")
            if pair["target_task_id"].lower() in expected_request[
                "user_prompt"
            ].lower():
                prompt_target_history_count += 1
            parsed = parse_paraphrase_response(record.get("raw_response"))
            if parsed != record.get("parsed_response"):
                errors.append(f"response_parse:{request_id}")
            if record.get("provider_seed_parameter_sent") is not False:
                provider_seed_violation_count += 1
            generation_counts[generation] += 1
            parent_by_chain[(pair_id, replicate)] = record
            attempt_path = run_root / "attempt_logs" / f"{request_id}.json"
            attempt = _read_json(attempt_path)
            if attempt.get("schema") != ATTEMPT_LOG_SCHEMA:
                errors.append(f"attempt_schema:{request_id}")
            if attempt.get("identity_hash") != record.get("identity_hash"):
                errors.append(f"attempt_identity:{request_id}")
            if attempt.get("completed") is not True:
                errors.append(f"attempt_incomplete:{request_id}")
    expected_per_generation = int(plan.get("source_pair_count") or 0) * len(
        plan.get("paraphrase_replicate_ids") or []
    )
    if any(
        generation_counts[generation] != expected_per_generation
        for generation in range(1, int(plan.get("generation_count") or 0) + 1)
    ):
        errors.append("generation_matrix")
    if provider_seed_violation_count:
        errors.append("raw_provider_seed_boundary")
    if prompt_target_history_count:
        errors.append("prompt_target_history")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "run_root_name": run_root.name,
        "source_pair_count": plan.get("source_pair_count"),
        "source_run_count": manifest.get("source_run_count"),
        "generation_count": plan.get("generation_count"),
        "paraphrase_replicate_count": len(
            plan.get("paraphrase_replicate_ids") or []
        ),
        "request_count": len(schedule_ids),
        "response_count": len(responses),
        "generation_response_counts": {
            str(key): value for key, value in sorted(generation_counts.items())
        },
        "provider_seed_violation_count": provider_seed_violation_count,
        "prompt_target_history_count": prompt_target_history_count,
        "verified": not errors,
        "errors": sorted(set(errors)),
        "packet_verification_hash": packet_verification.get(
            "verification_hash", ""
        ),
        "request_plan_hash": plan.get("request_plan_hash", ""),
        "run_hash": report.get("run_hash", ""),
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
        description="Verify the frozen WP8 Multi-generation descendant DAG."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verification = verify_run(args.work_root, args.packet_root, args.run_root)
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False, indent=2))
    if not verification["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
