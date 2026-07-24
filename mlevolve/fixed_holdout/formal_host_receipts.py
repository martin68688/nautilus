"""Write host-observed Kubernetes lifecycle attestations for formal blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from fixed_holdout.formal_runtime import (
    EVALUATOR_CREATION_SCHEMA,
    payload_hash,
    read_object,
    validate_block_template,
)


TRAINING_DELETION_SCHEMA = (
    "decision_admissibility_wp8_tier2_training_pod_deletion_attestation_v1"
)
EVALUATOR_DELETION_SCHEMA = (
    "decision_admissibility_wp8_tier2_evaluator_pod_deletion_attestation_v1"
)
EVALUATOR_FAILURE_SCHEMA = (
    "decision_admissibility_wp8_tier2_evaluator_failure_attestation_v1"
)
TRAINING_POD_IDENTITY_SCHEMA = (
    "decision_admissibility_wp8_tier2_training_pod_identity_v1"
)
INFRASTRUCTURE_ABORT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_infrastructure_abort_v1"
)
PRELAUNCH_ABORT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_prelaunch_infrastructure_abort_v1"
)
PRECONTRACT_ABORT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_precontract_infrastructure_abort_v1"
)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["attestation_hash"] = payload_hash(payload, "attestation_hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _write_report_exclusive(
    path: Path,
    value: Mapping[str, Any],
) -> dict[str, Any]:
    payload = dict(value)
    payload["report_hash"] = payload_hash(payload, "report_hash")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _seal_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink() or not path.exists():
            continue
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    root.chmod(root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def write_training_deletion(
    block_root: Path,
    *,
    namespace: str,
    pod_name: str,
    pod_uid: str,
    delete_requested_at: str,
    not_found_verified_at: str,
    not_found_probe_sha256: str,
    staging_gate_hash: str,
) -> dict[str, Any]:
    training = read_object(block_root / "TRAINING_MANIFEST.json")
    identity = {
        "schema": TRAINING_POD_IDENTITY_SCHEMA,
        "execution_kind": "devpod",
        "namespace": namespace,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
    }
    if identity != training.get("training_pod_identity"):
        raise ValueError("Host training Pod identity does not match the manifest")
    return _write_exclusive(
        block_root / "TRAINING_POD_DELETION_ATTESTATION.json",
        {
            "schema": TRAINING_DELETION_SCHEMA,
            "block_id": training["block_id"],
            "training_manifest_hash": training["manifest_hash"],
            "training_pod_identity": identity,
            "delete_requested": True,
            "delete_requested_at": delete_requested_at,
            "not_found_verified": True,
            "not_found_verified_at": not_found_verified_at,
            "kubernetes_reason": "NotFound",
            "not_found_probe_sha256": not_found_probe_sha256,
            "verified_by": "host_launcher",
            "terminal_metric_observed_before_not_found": False,
            "evaluator_create_allowed_after_verification": True,
            "staging_gate_hash": staging_gate_hash,
            "attestation_hash": "",
        },
    )


def write_evaluator_creation(
    block_root: Path,
    *,
    namespace: str,
    pod_name: str,
    pod_uid: str,
    kubernetes_creation_timestamp: str,
    container_image_id: str,
    staging_gate_hash: str,
) -> dict[str, Any]:
    deletion = read_object(block_root / "TRAINING_POD_DELETION_ATTESTATION.json")
    return _write_exclusive(
        block_root / "EVALUATOR_POD_CREATION_ATTESTATION.json",
        {
            "schema": EVALUATOR_CREATION_SCHEMA,
            "block_id": deletion["block_id"],
            "training_pod_deletion_attestation_hash": deletion["attestation_hash"],
            "evaluator_pod_identity": {
                "execution_kind": "devpod",
                "namespace": namespace,
                "pod_name": pod_name,
                "pod_uid": pod_uid,
            },
            "kubernetes_creation_timestamp": kubernetes_creation_timestamp,
            "container_image_id": container_image_id,
            "verified_by": "host_launcher",
            "staging_gate_hash": staging_gate_hash,
            "attestation_hash": "",
        },
    )


def write_evaluator_deletion(
    block_root: Path,
    *,
    namespace: str,
    pod_name: str,
    pod_uid: str,
    delete_requested_at: str,
    not_found_verified_at: str,
    not_found_probe_sha256: str,
    staging_gate_hash: str,
) -> dict[str, Any]:
    summary = read_object(block_root / "EVALUATION_SUMMARY.json")
    return _write_exclusive(
        block_root / "EVALUATOR_POD_DELETION_ATTESTATION.json",
        {
            "schema": EVALUATOR_DELETION_SCHEMA,
            "block_id": summary["block_id"],
            "evaluation_summary_hash": summary["summary_hash"],
            "evaluator_pod_identity": {
                "execution_kind": "devpod",
                "namespace": namespace,
                "pod_name": pod_name,
                "pod_uid": pod_uid,
            },
            "delete_requested": True,
            "delete_requested_at": delete_requested_at,
            "not_found_verified": True,
            "not_found_verified_at": not_found_verified_at,
            "kubernetes_reason": "NotFound",
            "not_found_probe_sha256": not_found_probe_sha256,
            "verified_by": "host_launcher",
            "staging_gate_hash": staging_gate_hash,
            "attestation_hash": "",
        },
    )


def write_evaluator_failure(
    block_root: Path,
    *,
    namespace: str,
    pod_name: str,
    pod_uid: str,
    failure_detected_at: str,
    delete_requested_at: str,
    not_found_verified_at: str,
    not_found_probe_sha256: str,
    staging_gate_hash: str,
) -> dict[str, Any]:
    """Seal an evaluator failure, including post-metric partial evidence."""

    block_root = block_root.resolve()
    report_path = block_root / "FORMAL_EVALUATOR_FAILURE_ATTESTATION.json"
    training = read_object(block_root / "TRAINING_MANIFEST.json")
    if training.get("manifest_hash") != payload_hash(training, "manifest_hash"):
        raise ValueError("Formal training manifest hash mismatch")
    creation = read_object(block_root / "EVALUATOR_POD_CREATION_ATTESTATION.json")
    if creation.get("attestation_hash") != payload_hash(creation, "attestation_hash"):
        raise ValueError("Evaluator creation attestation hash mismatch")
    identity = {
        "execution_kind": "devpod",
        "namespace": namespace,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
    }
    if identity != creation.get("evaluator_pod_identity"):
        raise ValueError("Failed evaluator Pod identity mismatch")
    if creation.get("block_id") != training.get("block_id"):
        raise ValueError("Evaluator creation/training block mismatch")
    if staging_gate_hash != creation.get("staging_gate_hash") or (
        staging_gate_hash != training.get("staging_gate_hash")
    ):
        raise ValueError("Evaluator failure staging gate mismatch")
    if len(not_found_probe_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in not_found_probe_sha256
    ):
        raise ValueError("Invalid evaluator NotFound probe SHA-256")
    exit_path = block_root / "EVALUATOR_LAUNCHER_EXIT_CODE"
    log_path = block_root / "EVALUATOR_LAUNCHER.log"
    state_path = block_root / "STATE"
    if not exit_path.is_file() or not log_path.is_file():
        raise ValueError("Evaluator failure lacks launcher evidence")
    exit_code = int(exit_path.read_text(encoding="utf-8").strip())
    if exit_code == 0:
        raise ValueError("Evaluator failure attestation received a zero exit code")
    state = state_path.read_text(encoding="utf-8").strip()
    if state != "evaluator_failed":
        raise ValueError("Evaluator failure state marker mismatch")
    if (block_root / "EVALUATION_SUMMARY.json").exists():
        raise ValueError("Evaluator failure cannot coexist with a final summary")

    score_paths = sorted(
        path
        for path in block_root.rglob("*.json")
        if path.name
        in {
            "all_candidate_terminal_scores.json",
            "fixed_holdout_scores.json",
        }
    )
    status_paths = sorted(block_root.rglob("fixed_holdout_writeback_status.json"))
    status_rows = []
    for path in status_paths:
        value = read_object(path)
        if value.get("status_hash") != payload_hash(value, "status_hash"):
            raise ValueError("Terminal writeback failure status hash mismatch")
        status_rows.append(
            {
                "path": path.relative_to(block_root).as_posix(),
                "status": str(value.get("status") or ""),
                "error_type": str(value.get("error_type") or ""),
                "reason": str(value.get("reason") or ""),
                "status_hash": str(value.get("status_hash") or ""),
                "file_sha256": _file_sha256(path),
            }
        )
    reasons = [row["reason"] for row in status_rows if row["reason"]]
    classification = (
        "authority_denial"
        if any("Authority denied" in reason for reason in reasons)
        else "evaluator_process_failure"
    )

    overlay_result_facts = 0
    overlay_event_count = 0
    for path in sorted(block_root.rglob("session_overlay/events.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            overlay_event_count += 1
            payload = event.get("payload") or {}
            if payload.get("publication_class") == "result_fact":
                overlay_result_facts += 1

    inventory: dict[str, str] = {}
    for path in sorted(block_root.rglob("*")):
        if not path.is_file() or path == report_path:
            continue
        inventory[path.relative_to(block_root).as_posix()] = _file_sha256(path)
    inventory_hash = hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    report = _write_report_exclusive(
        report_path,
        {
            "schema": EVALUATOR_FAILURE_SCHEMA,
            "status": "formal_evaluator_failed",
            "classification": classification,
            "block_id": training["block_id"],
            "task_id": training["task_id"],
            "agent_seed": training["agent_seed"],
            "failure_detected_at_utc": failure_detected_at,
            "evaluator_pod_identity": identity,
            "evaluator_creation_attestation_hash": creation["attestation_hash"],
            "evaluator_launcher_exit_code": exit_code,
            "evaluator_launcher_exit_code_sha256": _file_sha256(exit_path),
            "evaluator_launcher_log_sha256": _file_sha256(log_path),
            "delete_requested": True,
            "delete_requested_at": delete_requested_at,
            "not_found_verified": True,
            "not_found_verified_at": not_found_verified_at,
            "kubernetes_reason": "NotFound",
            "not_found_probe_sha256": not_found_probe_sha256,
            "terminal_metric_observed": bool(score_paths),
            "pre_metric_abort": not bool(score_paths),
            "partial_terminal_artifact_hashes": {
                path.relative_to(block_root).as_posix(): _file_sha256(path)
                for path in score_paths
            },
            "terminal_writeback_statuses": status_rows,
            "session_overlay_event_count": overlay_event_count,
            "normal_result_fact_count": overlay_result_facts,
            "training_manifest_hash": training["manifest_hash"],
            "training_manifest_sha256": _file_sha256(
                block_root / "TRAINING_MANIFEST.json"
            ),
            "staging_gate_hash": staging_gate_hash,
            "partial_file_count": len(inventory),
            "partial_file_inventory_hash": inventory_hash,
            "reuse_for_formal_execution": False,
            "silent_retry_permitted": False,
            "retry_requires_post_failure_preregistration_and_new_roots": True,
            "report_hash": "",
        },
    )
    _seal_tree(block_root)
    return report


def write_training_infrastructure_abort(
    block_root: Path,
    *,
    namespace: str,
    pod_name: str,
    pod_uid: str,
    detected_at: str,
    not_found_verified_at: str,
    not_found_probe_sha256: str,
    evaluator_not_found_probe_sha256: str,
    event_snapshot_sha256: str,
    event_reasons: list[str],
    failure_phase: str,
    pod_status_snapshot_sha256: str,
    staging_gate_hash: str,
) -> dict[str, Any]:
    """Seal a block whose training devpod vanished before its completion marker."""

    block_root = block_root.resolve()
    report_path = block_root / "FORMAL_BLOCK_INFRASTRUCTURE_ABORT.json"
    contract = read_object(block_root / "BLOCK_CONTRACT.json")
    if contract.get("contract_hash") != payload_hash(contract, "contract_hash"):
        raise ValueError("Block contract hash mismatch")
    identity = {
        "schema": TRAINING_POD_IDENTITY_SCHEMA,
        "execution_kind": "devpod",
        "namespace": namespace,
        "pod_name": pod_name,
        "pod_uid": pod_uid,
    }
    if identity != contract.get("training_pod_identity"):
        raise ValueError("Lost training Pod identity does not match the block contract")
    if staging_gate_hash != contract.get("staging_gate_hash"):
        raise ValueError(
            "Lost training Pod gate hash does not match the block contract"
        )
    for value, label in (
        (not_found_probe_sha256, "training NotFound probe"),
        (evaluator_not_found_probe_sha256, "evaluator NotFound probe"),
        (event_snapshot_sha256, "event snapshot"),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Invalid {label} SHA-256")
    if pod_status_snapshot_sha256 and (
        len(pod_status_snapshot_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in pod_status_snapshot_sha256
        )
    ):
        raise ValueError("Invalid Pod status snapshot SHA-256")

    terminal_artifacts = [
        path
        for path in block_root.rglob("*")
        if path.is_file()
        and (
            path.name.startswith("fixed_holdout_scores")
            or path.name in {"EVALUATION_SUMMARY.json", "EVALUATION_COMPLETE"}
        )
    ]
    if terminal_artifacts:
        raise ValueError("Cannot classify Pod loss after terminal evaluation")
    result_fact_markers = []
    for path in block_root.rglob("events.jsonl"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "result_fact" in text:
            result_fact_markers.append(path)
    if result_fact_markers:
        raise ValueError("Pre-terminal infrastructure abort contains a Result Fact")

    started_conditions = sorted(
        path.name for path in (block_root / "conditions").glob("*") if path.is_dir()
    )
    condition_receipts = list(block_root.rglob("CONDITION_RECEIPT.json"))
    evaluation_requests = list(
        block_root.rglob("fixed_holdout_evaluation_request.json")
    )
    submissions = list(block_root.rglob("submission_*.csv"))
    training_manifest_path = block_root / "TRAINING_MANIFEST.json"
    inventory: dict[str, str] = {}
    for path in sorted(block_root.rglob("*")):
        if not path.is_file() or path == report_path:
            continue
        inventory[path.relative_to(block_root).as_posix()] = _file_sha256(path)
    inventory_hash = hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    reasons = sorted({str(reason) for reason in event_reasons if str(reason)})
    if "NodeNotReady" in reasons and "TaintManagerEviction" in reasons:
        classification = "kubernetes_node_not_ready_taint_manager_eviction"
    elif failure_phase == "training_launcher_nonzero":
        classification = "training_launcher_process_failure"
    else:
        classification = "unexpected_training_devpod_loss"
    report = _write_report_exclusive(
        report_path,
        {
            "schema": INFRASTRUCTURE_ABORT_SCHEMA,
            "status": "aborted_before_training_completion_marker",
            "classification": classification,
            "block_id": contract["block_id"],
            "task_id": contract["task_id"],
            "agent_seed": contract["agent_seed"],
            "failure_detected_at_utc": detected_at,
            "failure_phase": failure_phase,
            "kubernetes_event_reasons": reasons,
            "event_snapshot_sha256": event_snapshot_sha256,
            "pod_status_snapshot_sha256": pod_status_snapshot_sha256,
            "training_pod_identity": identity,
            "training_pod_not_found_verified": True,
            "training_pod_not_found_verified_at": not_found_verified_at,
            "training_pod_not_found_probe_sha256": not_found_probe_sha256,
            "evaluator_pod_created": False,
            "evaluator_pod_not_found_verified": True,
            "evaluator_pod_not_found_probe_sha256": (evaluator_not_found_probe_sha256),
            "training_launcher_exit_code_written": (
                block_root / "TRAINING_LAUNCHER_EXIT_CODE"
            ).is_file(),
            "training_manifest_written": training_manifest_path.is_file(),
            "training_manifest_sha256": (
                _file_sha256(training_manifest_path)
                if training_manifest_path.is_file()
                else ""
            ),
            "evaluation_summary_written": False,
            "completed_condition_receipt_count": len(condition_receipts),
            "evaluation_request_count": len(evaluation_requests),
            "started_conditions": started_conditions,
            "candidate_submission_count": len(submissions),
            "partial_candidate_execution_only": True,
            "internal_search_metrics_only": True,
            "terminal_label_mount_created": False,
            "terminal_metric_observed": False,
            "formal_effect_observation": False,
            "positive_writeback_observed": False,
            "block_contract_hash": contract["contract_hash"],
            "source_snapshot_sha256": contract["source_snapshot_sha256"],
            "staging_content_manifest_hash": contract["staging_manifest_hash"],
            "staging_gate_hash": contract["staging_gate_hash"],
            "container_image_digest": contract["container_image_digest"],
            "partial_file_count": len(inventory),
            "partial_file_inventory_hash": inventory_hash,
            "reuse_for_formal_execution": False,
            "retry_requires_new_source_staging_and_output_root": True,
            "report_hash": "",
        },
    )
    _seal_tree(block_root)
    return report


def write_training_prelaunch_abort(
    output_root: Path,
    contract_root: Path,
    *,
    namespace: str,
    pod_name: str,
    pod_uid: str,
    detected_at: str,
    event_snapshot_sha256: str,
    event_reasons: list[str],
    pod_status_snapshot_sha256: str,
    scheduled_node: str,
    container_start_reason: str,
    container_start_exit_code: int,
    failure_message: str,
    not_found_verified_at: str,
    not_found_probe_sha256: str,
    evaluator_not_found_probe_sha256: str,
    staging_content_manifest_hash: str,
    staging_gate_hash: str,
) -> dict[str, Any]:
    """Seal an output root when the training container never became Ready."""

    output_root = output_root.resolve()
    contract_root = contract_root.resolve()
    template = validate_block_template(
        read_object(contract_root / "BLOCK_TEMPLATE.json")
    )
    if template["expected_training_pod_name"] != pod_name:
        raise ValueError("Prelaunch Pod name does not match the block template")
    report_path = output_root / "FORMAL_PRELAUNCH_INFRASTRUCTURE_ABORT.json"
    existing_files = [path for path in output_root.rglob("*") if path.is_file()]
    if existing_files:
        raise ValueError(f"Prelaunch output root is not empty: {existing_files}")
    for value, label in (
        (event_snapshot_sha256, "event snapshot"),
        (pod_status_snapshot_sha256, "Pod status snapshot"),
        (not_found_probe_sha256, "training NotFound probe"),
        (evaluator_not_found_probe_sha256, "evaluator NotFound probe"),
    ):
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Invalid {label} SHA-256")
    reasons = sorted({str(reason) for reason in event_reasons if str(reason)})
    lowered = failure_message.lower()
    classification = (
        "gpu_node_nvml_driver_library_version_mismatch"
        if "driver/library version mismatch" in lowered
        else "training_container_failed_before_ready"
    )
    report = _write_report_exclusive(
        report_path,
        {
            "schema": PRELAUNCH_ABORT_SCHEMA,
            "status": "aborted_before_training_container_start",
            "classification": classification,
            "failure_detected_at_utc": detected_at,
            "block_id": template["block_id"],
            "task_id": template["task_id"],
            "agent_seed": template["agent_seed"],
            "training_pod_identity": {
                "execution_kind": "devpod",
                "namespace": namespace,
                "pod_name": pod_name,
                "pod_uid": pod_uid,
            },
            "scheduled_node": scheduled_node,
            "container_started": False,
            "container_start_reason": container_start_reason,
            "container_start_exit_code": container_start_exit_code,
            "failure_message": failure_message,
            "kubernetes_event_reasons": reasons,
            "uid_scoped_event_snapshot_sha256": event_snapshot_sha256,
            "pod_status_snapshot_sha256": pod_status_snapshot_sha256,
            "training_pod_not_found_verified": True,
            "training_pod_not_found_verified_at": not_found_verified_at,
            "training_pod_not_found_probe_sha256": not_found_probe_sha256,
            "evaluator_pod_created": False,
            "evaluator_pod_not_found_verified": True,
            "evaluator_pod_not_found_probe_sha256": (evaluator_not_found_probe_sha256),
            "formal_training_started": False,
            "block_output_empty_at_abort": True,
            "candidate_submission_count": 0,
            "evaluation_request_count": 0,
            "terminal_label_mount_created": False,
            "terminal_metric_observed": False,
            "formal_effect_observation": False,
            "positive_writeback_observed": False,
            "source_snapshot_sha256": template["source_snapshot_sha256"],
            "staging_content_manifest_hash": staging_content_manifest_hash,
            "staging_gate_hash": staging_gate_hash,
            "container_image_digest": template["container_image_digest"],
            "reuse_for_formal_execution": False,
            "retry_requires_new_source_staging_and_output_root": True,
            "report_hash": "",
        },
    )
    _seal_tree(output_root)
    return report


def write_training_precontract_abort(
    output_root: Path,
    contract_root: Path,
    *,
    namespace: str,
    pod_name: str,
    pod_uid: str,
    detected_at: str,
    event_snapshot_sha256: str,
    event_reasons: list[str],
    pod_status_snapshot_sha256: str,
    scheduled_node: str,
    not_found_verified_at: str,
    not_found_probe_sha256: str,
    evaluator_not_found_probe_sha256: str,
    staging_content_manifest_hash: str,
    staging_gate_hash: str,
) -> dict[str, Any]:
    """Seal a launcher failure before its runtime block contract existed."""

    output_root = output_root.resolve()
    contract_root = contract_root.resolve()
    template = validate_block_template(
        read_object(contract_root / "BLOCK_TEMPLATE.json")
    )
    if template["expected_training_pod_name"] != pod_name:
        raise ValueError("Pre-contract Pod name does not match the block template")
    report_path = output_root / "FORMAL_PRECONTRACT_INFRASTRUCTURE_ABORT.json"
    allowed_names = {
        "STATE",
        "TRAINING_LAUNCHER.log",
        "TRAINING_LAUNCHER_EXIT_CODE",
        "TRAINING_STARTED_AT",
    }
    existing = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file()
    )
    if set(existing) != allowed_names:
        raise ValueError(f"Unexpected pre-contract artifacts: {existing}")
    if (output_root / "conditions").exists() or (
        output_root / "BLOCK_CONTRACT.json"
    ).exists():
        raise ValueError("Pre-contract abort contains a contract or condition")
    state_path = output_root / "STATE"
    exit_path = output_root / "TRAINING_LAUNCHER_EXIT_CODE"
    log_path = output_root / "TRAINING_LAUNCHER.log"
    if state_path.read_text(encoding="utf-8").strip() != "training_launcher_failed":
        raise ValueError("Pre-contract launcher state is not failed")
    exit_code = int(exit_path.read_text(encoding="utf-8").strip())
    if exit_code == 0:
        raise ValueError("Pre-contract abort received a zero launcher exit code")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    terminal_artifacts = [
        path
        for path in output_root.rglob("*")
        if path.is_file()
        and (
            path.name.startswith("fixed_holdout_scores")
            or path.name in {"EVALUATION_SUMMARY.json", "EVALUATION_COMPLETE"}
            or path.name.startswith("submission_")
        )
    ]
    if terminal_artifacts:
        raise ValueError("Pre-contract abort contains terminal artifacts")
    for value, label, allow_empty in (
        (event_snapshot_sha256, "event snapshot", False),
        (pod_status_snapshot_sha256, "Pod status snapshot", True),
        (not_found_probe_sha256, "training NotFound probe", False),
        (evaluator_not_found_probe_sha256, "evaluator NotFound probe", False),
    ):
        if allow_empty and not value:
            continue
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"Invalid {label} SHA-256")
    reasons = sorted({str(reason) for reason in event_reasons if str(reason)})
    classification = (
        "staging_schema_compatibility_failure"
        if "AssertionError: decision_admissibility_wp8_tier2_formal_"
        "continuation_staging_content_v1" in log_text
        else "training_launcher_failed_before_block_contract"
    )
    inventory = {
        relative: _file_sha256(output_root / relative) for relative in existing
    }
    inventory_hash = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = _write_report_exclusive(
        report_path,
        {
            "schema": PRECONTRACT_ABORT_SCHEMA,
            "status": "aborted_before_runtime_block_contract",
            "classification": classification,
            "failure_detected_at_utc": detected_at,
            "block_id": template["block_id"],
            "task_id": template["task_id"],
            "agent_seed": template["agent_seed"],
            "training_pod_identity": {
                "execution_kind": "devpod",
                "namespace": namespace,
                "pod_name": pod_name,
                "pod_uid": pod_uid,
            },
            "scheduled_node": scheduled_node,
            "container_started": "Started" in reasons,
            "training_launcher_exit_code": exit_code,
            "training_launcher_exit_code_sha256": _file_sha256(exit_path),
            "training_launcher_log_sha256": _file_sha256(log_path),
            "training_state_sha256": _file_sha256(state_path),
            "kubernetes_event_reasons": reasons,
            "uid_scoped_event_snapshot_sha256": event_snapshot_sha256,
            "pod_status_snapshot_sha256": pod_status_snapshot_sha256,
            "training_pod_not_found_verified": True,
            "training_pod_not_found_verified_at": not_found_verified_at,
            "training_pod_not_found_probe_sha256": not_found_probe_sha256,
            "evaluator_pod_created": False,
            "evaluator_pod_not_found_verified": True,
            "evaluator_pod_not_found_probe_sha256": (evaluator_not_found_probe_sha256),
            "runtime_block_contract_written": False,
            "condition_directory_created": False,
            "agent_generation_started": False,
            "candidate_execution_started": False,
            "candidate_submission_count": 0,
            "evaluation_request_count": 0,
            "terminal_label_mount_created": False,
            "terminal_metric_observed": False,
            "terminal_score_values_inspected": False,
            "formal_effect_observation": False,
            "positive_writeback_observed": False,
            "source_snapshot_sha256": template["source_snapshot_sha256"],
            "staging_content_manifest_hash": staging_content_manifest_hash,
            "staging_gate_hash": staging_gate_hash,
            "container_image_digest": template["container_image_digest"],
            "partial_file_count": len(inventory),
            "partial_file_inventory_hash": inventory_hash,
            "collector_source_sha256": _file_sha256(Path(__file__).resolve()),
            "reuse_for_formal_execution": False,
            "retry_requires_new_source_staging_and_output_root": True,
            "report_hash": "",
        },
    )
    _seal_tree(output_root)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "training-deletion",
        "evaluator-creation",
        "evaluator-deletion",
        "training-infrastructure-abort",
    ):
        sub = subparsers.add_parser(name)
        sub.add_argument("--block-root", type=Path, required=True)
        sub.add_argument("--namespace", required=True)
        sub.add_argument("--pod-name", required=True)
        sub.add_argument("--pod-uid", required=True)
        sub.add_argument("--staging-gate-hash", required=True)
        if name.endswith("deletion"):
            sub.add_argument("--delete-requested-at", required=True)
            sub.add_argument("--not-found-verified-at", required=True)
            sub.add_argument("--not-found-probe-sha256", required=True)
        elif name == "evaluator-creation":
            sub.add_argument("--kubernetes-creation-timestamp", required=True)
            sub.add_argument("--container-image-id", required=True)
        else:
            sub.add_argument("--detected-at", required=True)
            sub.add_argument("--not-found-verified-at", required=True)
            sub.add_argument("--not-found-probe-sha256", required=True)
            sub.add_argument("--evaluator-not-found-probe-sha256", required=True)
            sub.add_argument("--event-snapshot-sha256", required=True)
            sub.add_argument("--event-reasons-json", required=True)
            sub.add_argument("--failure-phase", required=True)
            sub.add_argument("--pod-status-snapshot-sha256", default="")
    evaluator_failure = subparsers.add_parser("evaluator-failure")
    evaluator_failure.add_argument("--block-root", type=Path, required=True)
    evaluator_failure.add_argument("--namespace", required=True)
    evaluator_failure.add_argument("--pod-name", required=True)
    evaluator_failure.add_argument("--pod-uid", required=True)
    evaluator_failure.add_argument("--staging-gate-hash", required=True)
    evaluator_failure.add_argument("--failure-detected-at", required=True)
    evaluator_failure.add_argument("--delete-requested-at", required=True)
    evaluator_failure.add_argument("--not-found-verified-at", required=True)
    evaluator_failure.add_argument("--not-found-probe-sha256", required=True)
    prelaunch = subparsers.add_parser("training-prelaunch-abort")
    prelaunch.add_argument("--output-root", type=Path, required=True)
    prelaunch.add_argument("--contract-root", type=Path, required=True)
    prelaunch.add_argument("--namespace", required=True)
    prelaunch.add_argument("--pod-name", required=True)
    prelaunch.add_argument("--pod-uid", required=True)
    prelaunch.add_argument("--staging-gate-hash", required=True)
    prelaunch.add_argument("--staging-content-manifest-hash", required=True)
    prelaunch.add_argument("--detected-at", required=True)
    prelaunch.add_argument("--event-snapshot-sha256", required=True)
    prelaunch.add_argument("--event-reasons-json", required=True)
    prelaunch.add_argument("--pod-status-snapshot-sha256", required=True)
    prelaunch.add_argument("--scheduled-node", required=True)
    prelaunch.add_argument("--container-start-reason", required=True)
    prelaunch.add_argument("--container-start-exit-code", type=int, required=True)
    prelaunch.add_argument("--failure-message", required=True)
    prelaunch.add_argument("--not-found-verified-at", required=True)
    prelaunch.add_argument("--not-found-probe-sha256", required=True)
    prelaunch.add_argument("--evaluator-not-found-probe-sha256", required=True)
    precontract = subparsers.add_parser("training-precontract-abort")
    precontract.add_argument("--output-root", type=Path, required=True)
    precontract.add_argument("--contract-root", type=Path, required=True)
    precontract.add_argument("--namespace", required=True)
    precontract.add_argument("--pod-name", required=True)
    precontract.add_argument("--pod-uid", required=True)
    precontract.add_argument("--staging-gate-hash", required=True)
    precontract.add_argument("--staging-content-manifest-hash", required=True)
    precontract.add_argument("--detected-at", required=True)
    precontract.add_argument("--event-snapshot-sha256", required=True)
    precontract.add_argument("--event-reasons-json", required=True)
    precontract.add_argument("--pod-status-snapshot-sha256", default="")
    precontract.add_argument("--scheduled-node", required=True)
    precontract.add_argument("--not-found-verified-at", required=True)
    precontract.add_argument("--not-found-probe-sha256", required=True)
    precontract.add_argument("--evaluator-not-found-probe-sha256", required=True)
    args = parser.parse_args()
    common = {
        "namespace": args.namespace,
        "pod_name": args.pod_name,
        "pod_uid": args.pod_uid,
        "staging_gate_hash": args.staging_gate_hash,
    }
    if args.command == "training-precontract-abort":
        event_reasons = json.loads(args.event_reasons_json)
        if not isinstance(event_reasons, list):
            raise ValueError("event-reasons-json must be a JSON list")
        result = write_training_precontract_abort(
            args.output_root,
            args.contract_root,
            namespace=args.namespace,
            pod_name=args.pod_name,
            pod_uid=args.pod_uid,
            detected_at=args.detected_at,
            event_snapshot_sha256=args.event_snapshot_sha256,
            event_reasons=list(map(str, event_reasons)),
            pod_status_snapshot_sha256=args.pod_status_snapshot_sha256,
            scheduled_node=args.scheduled_node,
            not_found_verified_at=args.not_found_verified_at,
            not_found_probe_sha256=args.not_found_probe_sha256,
            evaluator_not_found_probe_sha256=(args.evaluator_not_found_probe_sha256),
            staging_content_manifest_hash=(args.staging_content_manifest_hash),
            staging_gate_hash=args.staging_gate_hash,
        )
    elif args.command == "training-prelaunch-abort":
        event_reasons = json.loads(args.event_reasons_json)
        if not isinstance(event_reasons, list):
            raise ValueError("event-reasons-json must be a JSON list")
        result = write_training_prelaunch_abort(
            args.output_root,
            args.contract_root,
            namespace=args.namespace,
            pod_name=args.pod_name,
            pod_uid=args.pod_uid,
            detected_at=args.detected_at,
            event_snapshot_sha256=args.event_snapshot_sha256,
            event_reasons=list(map(str, event_reasons)),
            pod_status_snapshot_sha256=args.pod_status_snapshot_sha256,
            scheduled_node=args.scheduled_node,
            container_start_reason=args.container_start_reason,
            container_start_exit_code=args.container_start_exit_code,
            failure_message=args.failure_message,
            not_found_verified_at=args.not_found_verified_at,
            not_found_probe_sha256=args.not_found_probe_sha256,
            evaluator_not_found_probe_sha256=(args.evaluator_not_found_probe_sha256),
            staging_content_manifest_hash=(args.staging_content_manifest_hash),
            staging_gate_hash=args.staging_gate_hash,
        )
    elif args.command == "training-deletion":
        result = write_training_deletion(
            args.block_root,
            delete_requested_at=args.delete_requested_at,
            not_found_verified_at=args.not_found_verified_at,
            not_found_probe_sha256=args.not_found_probe_sha256,
            **common,
        )
    elif args.command == "evaluator-creation":
        result = write_evaluator_creation(
            args.block_root,
            kubernetes_creation_timestamp=args.kubernetes_creation_timestamp,
            container_image_id=args.container_image_id,
            **common,
        )
    elif args.command == "evaluator-failure":
        result = write_evaluator_failure(
            args.block_root,
            failure_detected_at=args.failure_detected_at,
            delete_requested_at=args.delete_requested_at,
            not_found_verified_at=args.not_found_verified_at,
            not_found_probe_sha256=args.not_found_probe_sha256,
            **common,
        )
    elif args.command == "evaluator-deletion":
        result = write_evaluator_deletion(
            args.block_root,
            delete_requested_at=args.delete_requested_at,
            not_found_verified_at=args.not_found_verified_at,
            not_found_probe_sha256=args.not_found_probe_sha256,
            **common,
        )
    else:
        event_reasons = json.loads(args.event_reasons_json)
        if not isinstance(event_reasons, list):
            raise ValueError("event-reasons-json must be a JSON list")
        result = write_training_infrastructure_abort(
            args.block_root,
            detected_at=args.detected_at,
            not_found_verified_at=args.not_found_verified_at,
            not_found_probe_sha256=args.not_found_probe_sha256,
            evaluator_not_found_probe_sha256=(args.evaluator_not_found_probe_sha256),
            event_snapshot_sha256=args.event_snapshot_sha256,
            event_reasons=list(map(str, event_reasons)),
            failure_phase=args.failure_phase,
            pod_status_snapshot_sha256=args.pod_status_snapshot_sha256,
            **common,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EVALUATOR_DELETION_SCHEMA",
    "EVALUATOR_FAILURE_SCHEMA",
    "INFRASTRUCTURE_ABORT_SCHEMA",
    "PRECONTRACT_ABORT_SCHEMA",
    "PRELAUNCH_ABORT_SCHEMA",
    "TRAINING_DELETION_SCHEMA",
    "write_evaluator_creation",
    "write_evaluator_deletion",
    "write_evaluator_failure",
    "write_training_deletion",
    "write_training_infrastructure_abort",
    "write_training_precontract_abort",
    "write_training_prelaunch_abort",
]
