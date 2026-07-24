from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.clean_replay import load_replay_queue  # noqa: E402
from authority.clean_replay_runner import (  # noqa: E402
    CLEAN_REPLAY_EXECUTION_REPORT_SCHEMA,
    validate_replay_execution_attempt,
    write_json_exclusive,
)
from authority.memory_snapshot import (  # noqa: E402
    ImmutableBaseBundle,
    sha256_file,
    sha256_json,
)
from authority.protocol_registry import ProtocolRegistry  # noqa: E402


VALIDATION_SCHEMA = "clean_replay_execution_validation_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verify_source_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    root = path.parent
    manifest = _read_json(path)
    files = {
        str(candidate.relative_to(root)): sha256_file(candidate)
        for candidate in sorted(root.rglob("*"))
        if candidate.is_file() and candidate.name != path.name
    }
    if files != manifest.get("file_hashes") or len(files) != manifest.get(
        "file_count"
    ):
        raise ValueError(f"Source snapshot file inventory mismatch: {path}")
    expected = sha256_json(
        {key: value for key, value in manifest.items() if key != "source_sha256"}
    )
    if manifest.get("source_sha256") != expected:
        raise ValueError(f"Source snapshot hash mismatch: {path}")
    return root, manifest


def _verify_queue_provenance(path: Path) -> dict[str, Any]:
    provenance = _read_json(path)
    expected = sha256_json(
        {
            key: value
            for key, value in provenance.items()
            if key != "provenance_hash"
        }
    )
    if provenance.get("provenance_hash") != expected:
        raise ValueError("Replay queue provenance hash mismatch")
    for filename, digest in (provenance.get("output_file_hashes") or {}).items():
        if sha256_file(path.parent / filename) != digest:
            raise ValueError(f"Replay queue artifact changed: {filename}")
    return provenance


def _verify_launch(path: Path) -> dict[str, Any]:
    launch = _read_json(path)
    expected = sha256_json(
        {key: value for key, value in launch.items() if key != "launch_hash"}
    )
    if launch.get("launch_hash") != expected:
        raise ValueError("Clean Replay launch manifest hash mismatch")
    return launch


def validate_execution(
    *,
    bundle_path: Path,
    queue_path: Path,
    queue_manifest_path: Path,
    queue_provenance_path: Path,
    execution_source_manifest_path: Path,
    validator_source_manifest_path: Path,
    launch_manifest_path: Path,
    launcher_log_path: Path,
    execution_output: Path,
    report_path: Path,
) -> dict[str, Any]:
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite validation report: {report_path}")
    execution_output = execution_output.resolve()
    if report_path.resolve().parent != execution_output:
        raise ValueError("Execution validation report must live in the output root")

    execution_source, execution_source_manifest = _verify_source_manifest(
        execution_source_manifest_path
    )
    validator_source, validator_source_manifest = _verify_source_manifest(
        validator_source_manifest_path
    )
    queue_provenance = _verify_queue_provenance(queue_provenance_path)
    launch = _verify_launch(launch_manifest_path)
    bundle = ImmutableBaseBundle.load(bundle_path, verify_artifacts=True)
    queue = load_replay_queue(queue_path, queue_manifest_path)
    registry = ProtocolRegistry(bundle.path / "protocol_registry")
    protocol_ref = registry.get("mlevolve-default", "2").ref()

    if str(execution_output) != str(launch.get("output_path") or ""):
        raise ValueError("Launch manifest/output path mismatch")
    if execution_source_manifest.get("source_sha256") != launch.get(
        "source_snapshot_sha256"
    ):
        raise ValueError("Launch manifest/execution source mismatch")
    if sha256_file(execution_source_manifest_path) != launch.get(
        "source_manifest_file_sha256"
    ):
        raise ValueError("Launch execution source manifest file mismatch")
    if execution_source_manifest["file_hashes"].get(
        "mlevolve/authority/clean_replay_runner.py"
    ) != launch.get("runner_sha256"):
        raise ValueError("Launch manifest/runner source mismatch")
    if queue.queue_file_sha256 != launch.get("queue_file_sha256"):
        raise ValueError("Launch manifest/queue file mismatch")
    if queue.manifest_sha256 != launch.get("queue_manifest_sha256"):
        raise ValueError("Launch manifest/queue manifest mismatch")
    if queue_provenance.get("provenance_hash") != launch.get(
        "queue_provenance_hash"
    ):
        raise ValueError("Launch manifest/queue provenance mismatch")
    if bundle.manifest_sha256 != launch.get("bundle_manifest_sha256"):
        raise ValueError("Launch manifest/Bundle mismatch")
    if protocol_ref.key() != launch.get("protocol_ref"):
        raise ValueError("Launch manifest/ProtocolRef mismatch")

    execution_report = _read_json(execution_output / "execution_report.json")
    if execution_report.get("schema") != CLEAN_REPLAY_EXECUTION_REPORT_SCHEMA:
        raise ValueError("Unsupported Clean Replay execution report")
    if execution_report.get("report_hash") != sha256_json(
        {
            key: value
            for key, value in execution_report.items()
            if key != "report_hash"
        }
    ):
        raise ValueError("Clean Replay execution report hash mismatch")
    if (
        execution_report.get("status") != "certification_material_ready"
        or execution_report.get("success_count") != 1
        or execution_report.get("failed_task_ids") != []
        or execution_report.get("historical_metric_used_as_evidence") is not False
    ):
        raise ValueError("Clean Replay execution did not produce one clean success")
    if execution_report.get("bundle_manifest_sha256") != bundle.manifest_sha256:
        raise ValueError("Execution report/Bundle mismatch")
    if execution_report.get("queue_manifest_sha256") != queue.manifest_sha256:
        raise ValueError("Execution report/queue mismatch")

    record_files = sorted(execution_output.rglob("execution_record.json"))
    records_by_hash: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in record_files:
        record = _read_json(path)
        expected = sha256_json(
            {key: value for key, value in record.items() if key != "record_hash"}
        )
        if record.get("record_hash") != expected:
            raise ValueError(f"Execution record hash mismatch: {path}")
        records_by_hash[str(record["record_hash"])] = (path.parent, record)
    if set(records_by_hash) != set(execution_report.get("record_hashes") or []):
        raise ValueError("Execution report/attempt record inventory mismatch")
    records_file = _read_json(execution_output / "execution_records.json")
    if [row.get("record_hash") for row in records_file.get("records") or []] != (
        execution_report.get("record_hashes") or []
    ):
        raise ValueError("Aggregate execution records are out of order or incomplete")
    success_hashes = execution_report.get("success_record_hashes") or []
    if len(success_hashes) != 1 or success_hashes[0] not in records_by_hash:
        raise ValueError("Clean Replay success record is unavailable")
    attempt_dir, record = records_by_hash[success_hashes[0]]
    entry = next(
        (value for value in queue.entries if value.entry_hash == record["entry_hash"]),
        None,
    )
    if entry is None:
        raise ValueError("Clean Replay success is absent from the immutable queue")
    validated = validate_replay_execution_attempt(
        bundle=bundle,
        entry=entry,
        protocol_ref=protocol_ref,
        registry=registry,
        attempt_dir=attempt_dir,
    )

    launch_entries = {
        str(value.get("entry_hash") or ""): value
        for value in launch.get("queue_entries") or []
    }
    launch_entry = launch_entries.get(entry.entry_hash)
    if not launch_entry or launch_entry.get("execution_eligible") is not True:
        raise ValueError("Successful replay was not preregistered as execution-eligible")
    if int(launch_entry.get("queue_rank", 0)) != entry.queue_rank:
        raise ValueError("Launch manifest changed the immutable queue rank")

    input_files = {
        str(path.relative_to(attempt_dir / "input")): sha256_file(path)
        for path in sorted((attempt_dir / "input").rglob("*"))
        if path.is_file()
    }
    if input_files != launch.get("data_file_hashes"):
        raise ValueError("Private replay input does not match the launch data snapshot")
    input_tree_sha256 = sha256_json(input_files)
    if input_tree_sha256 != launch.get("data_tree_sha256"):
        raise ValueError("Private replay input tree hash mismatch")

    launcher_payload = json.loads(launcher_log_path.read_text(encoding="utf-8"))
    if launcher_payload != execution_report:
        raise ValueError("Launcher log/execution report mismatch")
    output_artifact_hashes = {
        str(path.relative_to(execution_output)): sha256_file(path)
        for path in sorted(execution_output.rglob("*"))
        if path.is_file() and path.resolve() != report_path.resolve()
    }
    validation = {
        "schema": VALIDATION_SCHEMA,
        "status": "validated",
        "launch_hash": launch["launch_hash"],
        "launch_manifest_sha256": sha256_file(launch_manifest_path),
        "launcher_log_sha256": sha256_file(launcher_log_path),
        "execution_source_path": str(execution_source),
        "execution_source_sha256": execution_source_manifest["source_sha256"],
        "execution_runner_sha256": launch["runner_sha256"],
        "validator_source_path": str(validator_source),
        "validator_source_sha256": validator_source_manifest["source_sha256"],
        "validator_script_sha256": validator_source_manifest["file_hashes"][
            "paper-skills/memory_bundle/validate_clean_replay_execution.py"
        ],
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "queue_file_sha256": queue.queue_file_sha256,
        "queue_manifest_sha256": queue.manifest_sha256,
        "queue_provenance_hash": queue_provenance["provenance_hash"],
        "task_id": entry.task_id,
        "queue_rank": entry.queue_rank,
        "entry_hash": entry.entry_hash,
        "candidate_id": entry.candidate_id,
        "original_claim_id": entry.original_claim_id,
        "source_clause_id": entry.source_clause_id,
        "replay_artifact_id": record["replay_artifact_id"],
        "record_hash": record["record_hash"],
        "execution_report_hash": execution_report["report_hash"],
        "material_validation": validated.validation,
        "input_file_count": len(input_files),
        "input_tree_sha256": input_tree_sha256,
        "output_artifact_file_count": len(output_artifact_hashes),
        "output_artifact_hashes": output_artifact_hashes,
        "output_artifact_tree_sha256": sha256_json(output_artifact_hashes),
        "historical_metric_used_as_evidence": False,
        "validation_hash": "",
    }
    validation["validation_hash"] = sha256_json(
        {key: value for key, value in validation.items() if key != "validation_hash"}
    )
    bundle.assert_unchanged()
    write_json_exclusive(report_path, validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate persisted Clean Replay certification material."
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--queue-manifest", required=True, type=Path)
    parser.add_argument("--queue-provenance", required=True, type=Path)
    parser.add_argument("--execution-source-manifest", required=True, type=Path)
    parser.add_argument("--validator-source-manifest", required=True, type=Path)
    parser.add_argument("--launch-manifest", required=True, type=Path)
    parser.add_argument("--launcher-log", required=True, type=Path)
    parser.add_argument("--execution-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = validate_execution(
        bundle_path=args.bundle,
        queue_path=args.queue,
        queue_manifest_path=args.queue_manifest,
        queue_provenance_path=args.queue_provenance,
        execution_source_manifest_path=args.execution_source_manifest,
        validator_source_manifest_path=args.validator_source_manifest,
        launch_manifest_path=args.launch_manifest,
        launcher_log_path=args.launcher_log,
        execution_output=args.execution_output,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "validation_hash": result["validation_hash"],
                "record_hash": result["record_hash"],
                "output_artifact_tree_sha256": result[
                    "output_artifact_tree_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
