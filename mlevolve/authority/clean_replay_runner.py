from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

from agents.leakage_audit import audit_code

from .adapters.mlevolve.receipt_bridge import receipts_for_node
from .clean_replay import (
    ReplayQueueEntry,
    ReplayReceiptIngestor,
    load_replay_queue,
)
from .collectors import TrustedCollectorHost
from .memory_snapshot import ImmutableBaseBundle, sha256_file, sha256_json
from .models import ProtocolRef, Receipt, ReceiptType
from .protocol_registry import ProtocolRegistry, canonical_json
from .replay_certifier import (
    ProtocolRepairSurface,
    ReplayIdentity,
    verify_protocol_only_patch,
)
from .runtime_protocol import (
    PROTOCOL_EVIDENCE_LEVEL,
    PROTOCOL_EVIDENCE_SCHEMA,
    verify_persisted_runtime_protocol_observation,
    verify_runtime_protocol_observation,
)


CLEAN_REPLAY_EXECUTION_SCHEMA = "clean_replay_execution_record_v1"
CLEAN_REPLAY_EXECUTION_REPORT_SCHEMA = "clean_replay_execution_report_v1"
_FINAL_METRIC = re.compile(
    r"Final\s+Validation\s+Score\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_REQUIRED_TRUSTED_RECEIPTS = frozenset(
    {
        ReceiptType.METHOD_IDENTITY,
        ReceiptType.CODE_EXECUTION,
        ReceiptType.SPLIT_LINEAGE,
        ReceiptType.FIT_SCOPE,
        ReceiptType.PREDICTION_SCOPE,
        ReceiptType.EVALUATOR,
        ReceiptType.SELECTION_FREEZE,
    }
)
_RUNTIME_PROTOCOL_RECEIPTS = frozenset(
    {
        ReceiptType.SPLIT_LINEAGE,
        ReceiptType.FIT_SCOPE,
        ReceiptType.PREDICTION_SCOPE,
        ReceiptType.EVALUATOR,
        ReceiptType.SELECTION_FREEZE,
    }
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    if not normalized:
        raise ValueError("Replay path component is empty")
    return normalized[:160]


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_exclusive(
        path,
        (
            json.dumps(
                _jsonable(payload),
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )


def parse_final_validation_metric(output: str) -> float:
    matches = _FINAL_METRIC.findall(str(output or ""))
    if not matches:
        raise ValueError("Replay output has no Final Validation Score")
    value = float(matches[-1])
    if not math.isfinite(value):
        raise ValueError("Replay metric is not finite")
    return value


def _secure_extract(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe replay dataset archive member: {member.filename}")
        handle.extractall(destination)


def stage_task_input(source: Path, destination: Path) -> None:
    """Copy source data before executing untrusted historical code."""

    source = source.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Replay input already exists: {destination}")
    if not source.is_dir():
        raise FileNotFoundError(f"Replay task data is unavailable: {source}")
    shutil.copytree(source, destination, symlinks=False)
    for archive in sorted(destination.glob("*.zip")):
        _secure_extract(archive, destination)
    # This is defense in depth. The authoritative source is already protected
    # because candidate code receives only this private copy.
    for path in sorted(destination.rglob("*"), reverse=True):
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )


def validate_submission(input_dir: Path, submission_dir: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    sample_path = input_dir / "sample_submission.csv"
    if not sample_path.is_file():
        raise FileNotFoundError("Replay input has no sample_submission.csv")
    sample = pd.read_csv(sample_path)
    if sample.empty or not list(sample.columns):
        raise ValueError("Replay sample submission is empty")
    valid: list[tuple[Path, Any]] = []
    for path in sorted(submission_dir.rglob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if list(frame.columns) != list(sample.columns) or len(frame) != len(sample):
            continue
        identifier = frame.columns[0]
        if frame[identifier].isna().any() or frame[identifier].duplicated().any():
            continue
        if set(map(str, frame[identifier])) != set(map(str, sample[identifier])):
            continue
        predictions = frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        if predictions.empty or not np.isfinite(predictions.to_numpy()).all():
            continue
        valid.append((path, frame))
    if len(valid) != 1:
        raise ValueError(
            "Replay must produce exactly one schema-valid submission; "
            f"observed={len(valid)}"
        )
    path, frame = valid[0]
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(frame),
        "columns": list(frame.columns),
    }


def _read_bound_material(
    bundle: ImmutableBaseBundle,
    entry: ReplayQueueEntry,
) -> dict[str, Any]:
    graph = bundle.read_json("runforest/graph.json")
    nodes = {
        str(node.get("id") or ""): node
        for node in graph.get("nodes") or []
        if isinstance(node, Mapping) and node.get("id")
    }
    run_node = nodes.get(entry.child_artifact_id)
    if not isinstance(run_node, Mapping) or run_node.get("type") != "RunNode":
        raise ValueError("Replay queue child is not a Bundle RunNode")
    run_id = str(run_node.get("run_id") or "")
    raw_node_id = str(run_node.get("raw_node_id") or "")
    journal = bundle.read_json(f"raw_journals/{run_id}/journal.json")
    raw_nodes = journal.get("nodes") if isinstance(journal, Mapping) else None
    if not isinstance(raw_nodes, list):
        raise ValueError("Replay raw journal has no nodes")
    raw_node = next(
        (
            value
            for index, value in enumerate(raw_nodes)
            if isinstance(value, Mapping)
            and str(value.get("id") or value.get("node_id") or index)
            == raw_node_id
        ),
        None,
    )
    if raw_node is None:
        raise ValueError("Replay queue raw node is missing")
    code = str(raw_node.get("code") or "")
    if _sha256_text(code) != entry.code_sha256:
        raise ValueError("Replay queue code hash does not match the immutable Bundle")

    claims = {
        str(row.get("claim_id") or ""): row
        for row in bundle.read_jsonl("authority/claims.jsonl")
    }
    claim = claims.get(entry.original_claim_id)
    if not isinstance(claim, Mapping):
        raise ValueError("Replay queue original Claim is missing")
    if str(claim.get("subject_artifact_id") or "") != entry.child_artifact_id:
        raise ValueError("Replay queue original Claim subject mismatch")
    if str(claim.get("claim_type") or "") not in {
        "method_hypothesis",
        "debug_repair",
    }:
        raise ValueError("Replay queue original Claim is not replay-transferable")

    clauses = {
        str(row.get("clause_id") or ""): row
        for row in bundle.read_jsonl("sop/clauses.jsonl")
    }
    clause = clauses.get(entry.source_clause_id)
    if not isinstance(clause, Mapping):
        raise ValueError("Replay queue source clause is missing")
    if entry.original_claim_id not in set(map(str, clause.get("claim_refs") or [])):
        raise ValueError("Replay queue source clause/Claim binding mismatch")
    if entry.child_artifact_id not in set(
        map(str, clause.get("source_artifact_refs") or [])
    ):
        raise ValueError("Replay queue source clause/artifact binding mismatch")
    if str(clause.get("text") or "").strip() != entry.method_hypothesis:
        raise ValueError("Replay queue method hypothesis text mismatch")
    return {
        "code": code,
        "run_node": copy.deepcopy(dict(run_node)),
        "original_claim": copy.deepcopy(dict(claim)),
        "source_clause": copy.deepcopy(dict(clause)),
    }


def _interpreter_config(*, cpu_number: int, num_gpus: int) -> Any:
    return SimpleNamespace(
        evaluation_authority=SimpleNamespace(
            mode="enforce",
            runtime_protocol_observer_enabled=True,
        ),
        agent=SimpleNamespace(
            search=SimpleNamespace(
                parallel_search_num=1,
                num_gpus=int(num_gpus),
            )
        ),
        start_cpu_id=0,
        cpu_number=int(cpu_number),
    )


def _metric_direction(run_node: Mapping[str, Any]) -> bool:
    value = run_node.get("metric_maximize")
    if value is None and isinstance(run_node.get("metric"), Mapping):
        value = run_node["metric"].get("maximize")
    return value is not False


def execute_replay_entry(
    *,
    bundle: ImmutableBaseBundle,
    entry: ReplayQueueEntry,
    protocol_ref: ProtocolRef,
    registry: ProtocolRegistry,
    data_dir: Path,
    attempt_dir: Path,
    run_id: str,
    timeout: int,
    cpu_number: int,
    num_gpus: int,
    collector_host: TrustedCollectorHost,
    interpreter_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute one immutable queue entry and return a certification material record."""

    if attempt_dir.exists():
        raise FileExistsError(f"Replay attempt already exists: {attempt_dir}")
    attempt_dir.mkdir(parents=True)
    material = _read_bound_material(bundle, entry)
    code = material["code"]
    code_path = attempt_dir / "replay_code.py"
    _write_bytes_exclusive(code_path, code.encode("utf-8"))
    input_dir = attempt_dir / "input"
    working_dir = attempt_dir / "working"
    submission_dir = attempt_dir / "submission"

    static_audit = audit_code(code)
    write_json_exclusive(attempt_dir / "static_audit.json", static_audit)
    audit_clean = bool(
        static_audit.get("detector_status") == "complete"
        and static_audit.get("status") == "clean"
        and static_audit.get("issues") == []
        and static_audit.get("hard_block") is False
        and static_audit.get("metric_disposition") == "accept"
    )
    replay_artifact_id = (
        f"clean-replay::{entry.candidate_id.rsplit('::', 1)[-1]}::"
        f"{entry.code_sha256[:16]}"
    )
    verification = verify_protocol_only_patch(
        code,
        code,
        ProtocolRepairSurface.from_protocol_spec(registry.resolve(protocol_ref)),
        source_artifact_id=entry.child_artifact_id,
        replay_artifact_id=replay_artifact_id,
    )
    write_json_exclusive(
        attempt_dir / "verification_report.json", verification.as_dict()
    )
    if verification.identity != ReplayIdentity.METHOD_PRESERVED:
        raise ValueError("Exact Clean Replay unexpectedly changed method identity")

    execution_attempted = audit_clean
    if execution_attempted:
        # Candidate code receives only a private dataset copy. Known-blocked
        # code is rejected before that copy and before an Interpreter exists.
        stage_task_input(data_dir, input_dir)
        working_dir.mkdir()
        submission_dir.mkdir()
        if interpreter_factory is None:
            from engine.executor import Interpreter

            interpreter_factory = Interpreter
        interpreter = interpreter_factory(
            working_dir=attempt_dir,
            timeout=int(timeout),
            max_parallel_run=1,
            cfg=_interpreter_config(
                cpu_number=int(cpu_number), num_gpus=int(num_gpus)
            ),
        )
        execution = interpreter.run(
            code,
            id=entry.candidate_id.rsplit("::", 1)[-1],
            working_dir=str(attempt_dir),
        )
    else:
        execution = SimpleNamespace(
            term_out=[],
            protocol_observation={
                "schema": "clean_replay_protocol_observation_absent_v1",
                "status": "not_executed",
                "reason": "deterministic_static_audit_not_clean",
                "source_code_sha256": entry.code_sha256,
            },
            exc_type=None,
            exec_time=0.0,
        )
    output = "".join(execution.term_out or [])
    _write_bytes_exclusive(
        attempt_dir / "execution.log", output.encode("utf-8", errors="replace")
    )
    observation = execution.protocol_observation or {}
    write_json_exclusive(attempt_dir / "protocol_observation.json", observation)
    runtime_observation_verified_in_host = bool(
        execution_attempted and verify_runtime_protocol_observation(observation)
    )
    persisted_observation_integrity_valid = bool(
        execution_attempted
        and verify_persisted_runtime_protocol_observation(observation)
    )

    status = "execution_failed"
    failure_reasons: list[str] = []
    metric_value: float | None = None
    submission: dict[str, Any] | None = None
    if execution_attempted and execution.exc_type:
        failure_reasons.append(f"execution:{execution.exc_type}")
    if not audit_clean:
        failure_reasons.append("deterministic_static_audit_not_clean")
        status = "blocked_static_audit"
    if execution_attempted and not runtime_observation_verified_in_host:
        failure_reasons.append(
            "runtime_protocol_observation:"
            + str(observation.get("reason") or "not_clean")
        )
    if execution_attempted:
        try:
            metric_value = parse_final_validation_metric(output)
        except Exception as error:
            failure_reasons.append(f"terminal_metric:{type(error).__name__}")
        try:
            submission = validate_submission(input_dir, submission_dir)
        except Exception as error:
            failure_reasons.append(f"submission:{type(error).__name__}")

    receipts = []
    if not failure_reasons and metric_value is not None:
        metric = SimpleNamespace(
            value=metric_value,
            maximize=_metric_direction(material["run_node"]),
        )
        node = SimpleNamespace(
            id=replay_artifact_id,
            code=code,
            code_sha256_expected=entry.code_sha256,
            method_fingerprint=verification.replay_method_fingerprint,
            metric=metric,
            exec_time=float(execution.exec_time),
            is_buggy=False,
            is_valid=True,
            leakage_audit=static_audit,
            protocol_observation=observation,
            protocol_repair={},
            draft_role="clean_replay",
            selected_strategy=None,
        )
        receipts = [
            receipt
            for receipt in receipts_for_node(
                node,
                protocol_ref,
                run_id,
                collector_host=collector_host,
            )
            if receipt.trust_status == "trusted_host"
        ]
        observed_types = {receipt.receipt_type for receipt in receipts}
        missing_types = sorted(
            receipt_type.value
            for receipt_type in _REQUIRED_TRUSTED_RECEIPTS - observed_types
        )
        if missing_types:
            failure_reasons.append(
                "missing_trusted_receipts:" + ",".join(missing_types)
            )
        else:
            ReplayReceiptIngestor.validate(
                receipts,
                replay_artifact_id=replay_artifact_id,
                protocol_ref=protocol_ref,
                verification=verification,
            )
    if not failure_reasons:
        status = "certification_material_ready"

    receipt_rows = [_jsonable(receipt) for receipt in receipts]
    write_json_exclusive(
        attempt_dir / "trusted_receipts.json",
        {"receipts": receipt_rows},
    )
    record = {
        "schema": CLEAN_REPLAY_EXECUTION_SCHEMA,
        "status": status,
        "candidate_id": entry.candidate_id,
        "queue_rank": entry.queue_rank,
        "task_id": entry.task_id,
        "entry_hash": entry.entry_hash,
        "source_artifact_id": entry.source_artifact_id,
        "parent_artifact_id": entry.parent_artifact_id,
        "child_artifact_id": entry.child_artifact_id,
        "original_claim_id": entry.original_claim_id,
        "source_clause_id": entry.source_clause_id,
        "replay_artifact_id": replay_artifact_id,
        "source_code_sha256": entry.code_sha256,
        "replay_code_sha256": sha256_file(code_path),
        "verification_report_hash": verification.report_hash,
        "verification_identity": verification.identity.value,
        "protocol_ref": protocol_ref.key(),
        "execution_attempted": execution_attempted,
        "execution_status": (
            "not_started_static_audit_blocked"
            if not execution_attempted
            else "failed"
            if execution.exc_type
            else "completed"
        ),
        "execution_exception_type": execution.exc_type,
        "execution_time_seconds": float(execution.exec_time),
        "runtime_observation_verified_in_execution_host": (
            runtime_observation_verified_in_host
        ),
        "persisted_observation_integrity_valid": (
            persisted_observation_integrity_valid
        ),
        "metric_value": metric_value,
        "metric_maximize": _metric_direction(material["run_node"]),
        "submission": submission,
        "static_audit_sha256": sha256_file(attempt_dir / "static_audit.json"),
        "protocol_observation_sha256": sha256_file(
            attempt_dir / "protocol_observation.json"
        ),
        "execution_log_sha256": sha256_file(attempt_dir / "execution.log"),
        "trusted_receipt_ids": sorted(
            receipt.receipt_id for receipt in receipts
        ),
        "trusted_receipt_types": sorted(
            receipt.receipt_type.value for receipt in receipts
        ),
        "trusted_receipts_sha256": sha256_file(
            attempt_dir / "trusted_receipts.json"
        ),
        "source_domains": sorted(
            set(map(str, material["source_clause"].get("source_domains") or []))
        ),
        "transfer_scope": str(
            material["source_clause"].get("transfer_scope") or ""
        ),
        "failure_reasons": sorted(set(failure_reasons)),
        "record_hash": "",
    }
    record["record_hash"] = sha256_json(
        {key: value for key, value in record.items() if key != "record_hash"}
    )
    write_json_exclusive(attempt_dir / "execution_record.json", record)
    bundle.assert_unchanged()
    return record


@dataclasses.dataclass(frozen=True)
class ValidatedReplayMaterial:
    record: dict[str, Any]
    verification: Any
    receipts: tuple[Receipt, ...]
    validation: dict[str, Any]


def _receipt_from_value(value: Mapping[str, Any]) -> Receipt:
    payload = copy.deepcopy(dict(value))
    payload["receipt_type"] = ReceiptType(str(payload["receipt_type"]))
    return Receipt(**payload)


def validate_replay_execution_attempt(
    *,
    bundle: ImmutableBaseBundle,
    entry: ReplayQueueEntry,
    protocol_ref: ProtocolRef,
    registry: ProtocolRegistry,
    attempt_dir: Path,
) -> ValidatedReplayMaterial:
    """Revalidate persisted certification material in a later host process."""

    record = json.loads(
        (attempt_dir / "execution_record.json").read_text(encoding="utf-8")
    )
    if (
        record.get("schema") != CLEAN_REPLAY_EXECUTION_SCHEMA
        or record.get("status") != "certification_material_ready"
        or record.get("execution_attempted") is not True
        or record.get("execution_status") != "completed"
        or record.get("failure_reasons") != []
    ):
        raise ValueError("Replay attempt is not certification material")
    if record.get("entry_hash") != entry.entry_hash:
        raise ValueError("Replay execution record/queue entry mismatch")
    if record.get("protocol_ref") != protocol_ref.key():
        raise ValueError("Replay execution record ProtocolRef mismatch")
    expected_record_hash = sha256_json(
        {key: value for key, value in record.items() if key != "record_hash"}
    )
    if record.get("record_hash") != expected_record_hash:
        raise ValueError("Replay execution record hash mismatch")

    material = _read_bound_material(bundle, entry)
    code = material["code"]
    code_path = attempt_dir / "replay_code.py"
    if sha256_file(code_path) != entry.code_sha256:
        raise ValueError("Persisted replay code does not match the queue")
    verification = verify_protocol_only_patch(
        code,
        code,
        ProtocolRepairSurface.from_protocol_spec(registry.resolve(protocol_ref)),
        source_artifact_id=entry.child_artifact_id,
        replay_artifact_id=str(record.get("replay_artifact_id") or ""),
    )
    verification_payload = json.loads(
        (attempt_dir / "verification_report.json").read_text(encoding="utf-8")
    )
    if verification.as_dict() != verification_payload:
        raise ValueError("Replay verification report is not reproducible")
    if record.get("verification_report_hash") != verification.report_hash:
        raise ValueError("Replay record/verifier hash mismatch")

    static_audit = json.loads(
        (attempt_dir / "static_audit.json").read_text(encoding="utf-8")
    )
    if not (
        static_audit.get("detector_status") == "complete"
        and static_audit.get("status") == "clean"
        and static_audit.get("issues") == []
        and static_audit.get("hard_block") is False
        and static_audit.get("metric_disposition") == "accept"
        and static_audit.get("code_sha256") == entry.code_sha256
    ):
        raise ValueError("Persisted replay static audit is not clean")
    observation = json.loads(
        (attempt_dir / "protocol_observation.json").read_text(encoding="utf-8")
    )
    if not verify_persisted_runtime_protocol_observation(observation):
        raise ValueError("Persisted replay protocol observation is invalid")

    receipt_rows = json.loads(
        (attempt_dir / "trusted_receipts.json").read_text(encoding="utf-8")
    ).get("receipts")
    if not isinstance(receipt_rows, list):
        raise ValueError("Persisted replay Receipts are unavailable")
    receipts = tuple(_receipt_from_value(row) for row in receipt_rows)
    ReplayReceiptIngestor.validate(
        receipts,
        replay_artifact_id=str(record["replay_artifact_id"]),
        protocol_ref=protocol_ref,
        verification=verification,
    )
    if {receipt.receipt_type for receipt in receipts} != _REQUIRED_TRUSTED_RECEIPTS:
        raise ValueError("Persisted replay does not contain exactly seven Receipts")
    if not receipts or receipts[0].parent_event_hash:
        raise ValueError("Replay Receipt event chain has an invalid root")
    if any(
        current.parent_event_hash != previous.event_hash
        for previous, current in zip(receipts, receipts[1:])
    ):
        raise ValueError("Replay Receipt event chain is discontinuous")
    if sorted(receipt.receipt_id for receipt in receipts) != record.get(
        "trusted_receipt_ids"
    ):
        raise ValueError("Replay record/Receipt ID mismatch")
    if sorted(receipt.receipt_type.value for receipt in receipts) != record.get(
        "trusted_receipt_types"
    ):
        raise ValueError("Replay record/Receipt type mismatch")

    static_audit_binding = _sha256_text(
        canonical_json(
            {
                "schema": static_audit.get("schema"),
                "detector_version": static_audit.get("detector_version"),
                "detector_status": static_audit.get("detector_status"),
                "code_sha256": static_audit.get("code_sha256"),
                "structural_sha256": static_audit.get("structural_sha256"),
                "issues": static_audit.get("issues"),
                "status": static_audit.get("status"),
                "metric_disposition": static_audit.get("metric_disposition"),
            }
        )
    )
    scope_binding = _sha256_text(
        canonical_json(
            {
                "scope_hashes": observation.get("scope_hashes"),
                "scope_input_hashes": observation.get("scope_input_hashes"),
                "scope_output_hashes": observation.get("scope_output_hashes"),
            }
        )
    )
    evidence_binding = {
        "schema": PROTOCOL_EVIDENCE_SCHEMA,
        "evidence_level": PROTOCOL_EVIDENCE_LEVEL,
        "source_code_sha256": entry.code_sha256,
        "executed_source_sha256": observation["executed_source_sha256"],
        "plan_sha256": observation["plan_sha256"],
        "trace_sha256": observation["trace_sha256"],
        "attestation_sha256": observation["attestation_sha256"],
        "static_audit_sha256": static_audit_binding,
        "scope_binding_sha256": scope_binding,
    }
    for receipt in receipts:
        if receipt.receipt_type in _RUNTIME_PROTOCOL_RECEIPTS:
            if receipt.payload.get("protocol_evidence") != evidence_binding:
                raise ValueError("Replay Receipt protocol-evidence binding mismatch")

    output = (attempt_dir / "execution.log").read_text(
        encoding="utf-8", errors="replace"
    )
    metric_value = parse_final_validation_metric(output)
    if metric_value != record.get("metric_value"):
        raise ValueError("Replay terminal metric/record mismatch")
    submission = validate_submission(
        attempt_dir / "input", attempt_dir / "submission"
    )
    if submission != record.get("submission"):
        raise ValueError("Replay submission/record mismatch")
    for filename, field_name in (
        ("static_audit.json", "static_audit_sha256"),
        ("protocol_observation.json", "protocol_observation_sha256"),
        ("execution.log", "execution_log_sha256"),
        ("trusted_receipts.json", "trusted_receipts_sha256"),
    ):
        if sha256_file(attempt_dir / filename) != record.get(field_name):
            raise ValueError(f"Replay persisted artifact hash mismatch: {filename}")

    explicit_host_verification = record.get(
        "runtime_observation_verified_in_execution_host"
    )
    if explicit_host_verification not in {None, True}:
        raise ValueError("Replay record says host runtime verification failed")
    explicit_persisted_integrity = record.get(
        "persisted_observation_integrity_valid"
    )
    if explicit_persisted_integrity not in {None, True}:
        raise ValueError("Replay record says persisted observation was invalid")
    validation = {
        "schema": "validated_replay_material_v1",
        "status": "validated",
        "record_hash": record["record_hash"],
        "verification_report_hash": verification.report_hash,
        "receipt_ids": sorted(receipt.receipt_id for receipt in receipts),
        "receipt_types": sorted(
            receipt.receipt_type.value for receipt in receipts
        ),
        "metric_value": metric_value,
        "metric_maximize": bool(record.get("metric_maximize")),
        "submission_sha256": submission["sha256"],
        "persisted_observation_integrity_valid": True,
        "runtime_host_verification_explicit": explicit_host_verification is True,
        "runtime_host_verification_inferred": explicit_host_verification is None,
        "runtime_host_verification_basis": (
            "execution_record_explicit"
            if explicit_host_verification is True
            else "legacy_success_path_plus_required_trusted_receipts"
        ),
        "validation_hash": "",
    }
    validation["validation_hash"] = sha256_json(
        {key: value for key, value in validation.items() if key != "validation_hash"}
    )
    bundle.assert_unchanged()
    return ValidatedReplayMaterial(
        record=copy.deepcopy(record),
        verification=verification,
        receipts=copy.deepcopy(receipts),
        validation=validation,
    )


def run_replay_queue(
    *,
    bundle_path: Path,
    queue_path: Path,
    queue_manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    task_ids: Iterable[str],
    protocol_id: str,
    protocol_version: str,
    timeout: int,
    cpu_number: int,
    num_gpus: int,
    collector_id: str,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Replay output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    bundle = ImmutableBaseBundle.load(bundle_path, verify_artifacts=True)
    queue = load_replay_queue(queue_path, queue_manifest_path)
    registry = ProtocolRegistry(bundle.path / "protocol_registry")
    protocol_ref = registry.get(protocol_id, protocol_version).ref()
    selected_tasks = sorted({str(value) for value in task_ids if str(value)})
    if not selected_tasks:
        raise ValueError("Clean Replay requires at least one explicit task")
    entries_by_task: dict[str, list[ReplayQueueEntry]] = {}
    for entry in queue.entries:
        if entry.task_id in selected_tasks:
            entries_by_task.setdefault(entry.task_id, []).append(entry)
    missing_tasks = sorted(set(selected_tasks) - set(entries_by_task))
    if missing_tasks:
        raise ValueError(f"Replay queue lacks selected tasks: {missing_tasks}")

    records: list[dict[str, Any]] = []
    collector_host = TrustedCollectorHost(collector_id)
    for task_id in selected_tasks:
        task_data = data_root / task_id / "prepared" / "public"
        success = False
        for entry in sorted(
            entries_by_task[task_id], key=lambda value: value.queue_rank
        ):
            attempt_dir = (
                output_dir
                / "attempts"
                / _safe_component(task_id)
                / f"rank-{entry.queue_rank}-{entry.candidate_id.rsplit('::', 1)[-1]}"
            )
            record = execute_replay_entry(
                bundle=bundle,
                entry=entry,
                protocol_ref=protocol_ref,
                registry=registry,
                data_dir=task_data,
                attempt_dir=attempt_dir,
                run_id=(
                    f"clean-replay::{queue.manifest_sha256[:16]}::"
                    f"{_safe_component(task_id)}"
                ),
                timeout=timeout,
                cpu_number=cpu_number,
                num_gpus=num_gpus,
                collector_host=collector_host,
            )
            records.append(record)
            if record["status"] == "certification_material_ready":
                success = True
                break
        if not success:
            # Failure is explicit in the aggregate report; no Claim or Bundle
            # can be published for this task.
            continue

    successes = [
        record
        for record in records
        if record["status"] == "certification_material_ready"
    ]
    report = {
        "schema": CLEAN_REPLAY_EXECUTION_REPORT_SCHEMA,
        "status": (
            "certification_material_ready"
            if len({record["task_id"] for record in successes})
            == len(selected_tasks)
            else "failed"
        ),
        "execution_policy": "first_clean_success_by_immutable_queue_rank",
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "bundle_certification_level": bundle.manifest.get(
            "certification_level"
        ),
        "queue_manifest_sha256": queue.manifest_sha256,
        "queue_file_sha256": queue.queue_file_sha256,
        "protocol_ref": protocol_ref.key(),
        "collector_id": collector_id,
        "selected_task_ids": selected_tasks,
        "attempt_count": len(records),
        "success_count": len(successes),
        "successful_task_ids": sorted(
            {record["task_id"] for record in successes}
        ),
        "failed_task_ids": sorted(
            set(selected_tasks)
            - {record["task_id"] for record in successes}
        ),
        "record_hashes": [record["record_hash"] for record in records],
        "success_record_hashes": [
            record["record_hash"] for record in successes
        ],
        "historical_metric_used_as_evidence": False,
        "source_bundle_verified_unchanged": True,
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    write_json_exclusive(output_dir / "execution_report.json", report)
    write_json_exclusive(
        output_dir / "execution_records.json", {"records": records}
    )
    bundle.assert_unchanged()
    return report


__all__ = [
    "CLEAN_REPLAY_EXECUTION_REPORT_SCHEMA",
    "CLEAN_REPLAY_EXECUTION_SCHEMA",
    "execute_replay_entry",
    "parse_final_validation_metric",
    "run_replay_queue",
    "stage_task_input",
    "validate_submission",
    "write_json_exclusive",
]
