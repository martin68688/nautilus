"""Result-blind recovery for a pre-terminal formal training finalizer failure."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

from fixed_holdout.common import sha256_file
from fixed_holdout.formal_block_training import finalize_training_block
from fixed_holdout.formal_runtime import (
    environment_has_solver_secret,
    mount_points,
    payload_hash,
    read_object,
    verify_source_snapshot,
)


SCHEMA = "decision_admissibility_wp8_tier2_formal_" "preterminal_finalizer_recovery_v1"
SOURCE_SCHEMA = "decision_admissibility_wp8_tier2_formal_recovery_source_v1"
AMENDMENT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_"
    "preterminal_finalizer_recovery_amendment_v1"
)
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_"
    "preterminal_finalizer_recovery_amendment_verification_v1"
)
DIAGNOSTIC_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_" "preterminal_finalizer_diagnostic_v1"
)
EXPECTED_DISPOSITION = {
    "full_decision_admissibility": ("pre_terminal_failure:authority_denial"),
    "authority_only": "pre_terminal_failure:retained_run_failure",
    "no_memory": "training_complete_unscored",
    "global_validity_bit": "training_complete_unscored",
    "flat_relevance_memory": "training_complete_unscored",
}
TERMINAL_FILENAMES = {
    "all_candidate_terminal_scores.json",
    "fixed_holdout_scores.json",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _inventory_hash(inventory: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical(dict(inventory))).hexdigest()


def _write_exclusive(
    path: Path,
    payload: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    value = dict(payload)
    value[hash_field] = payload_hash(value, hash_field)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return value


def _write_marker(path: Path, value: str = "") -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _assert_read_only(root: Path) -> None:
    probe = root / ".formal_recovery_write_probe"
    try:
        probe.write_text("forbidden", encoding="utf-8")
    except OSError:
        return
    probe.unlink(missing_ok=True)
    raise ValueError(f"Recovery read-only mount is writable: {root}")


def _path_exists(path: str) -> bool:
    return Path(path).exists()


def _condition_disposition(training: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for condition, row_value in (training.get("conditions") or {}).items():
        row = dict(row_value or {})
        status = str(row.get("status") or "")
        if status == "training_complete_unscored":
            result[str(condition)] = status
        elif row.get("failure_classification") == "authority_denial":
            result[str(condition)] = f"{status}:authority_denial"
        else:
            result[str(condition)] = f"{status}:retained_run_failure"
    return result


def recover_preterminal_finalizer(
    *,
    output_root: Path,
    block_contract_path: Path,
    bundle_root: Path,
    source_root: Path,
    recovery_root: Path,
    amendment_path: Path,
    amendment_verification_path: Path,
    diagnostic_path: Path,
    recovery_source_manifest_path: Path,
    pod_name: str,
    pod_namespace: str,
    pod_uid: str,
) -> dict[str, Any]:
    output_root = output_root.resolve()
    block_contract_path = block_contract_path.resolve()
    bundle_root = bundle_root.resolve()
    source_root = source_root.resolve()
    recovery_root = recovery_root.resolve()
    amendment_path = amendment_path.resolve()
    amendment_verification_path = amendment_verification_path.resolve()
    diagnostic_path = diagnostic_path.resolve()
    recovery_source_manifest_path = recovery_source_manifest_path.resolve()

    for path in (
        output_root,
        bundle_root,
        source_root,
        recovery_root,
    ):
        if not path.is_dir():
            raise ValueError(f"Formal recovery root is absent: {path}")
    for path in (
        block_contract_path,
        amendment_path,
        amendment_verification_path,
        diagnostic_path,
        recovery_source_manifest_path,
    ):
        if not path.is_file():
            raise ValueError(f"Formal recovery input is absent: {path}")

    amendment = read_object(amendment_path)
    if amendment.get("schema") != AMENDMENT_SCHEMA or amendment.get(
        "amendment_hash"
    ) != payload_hash(amendment, "amendment_hash"):
        raise ValueError("Recovery amendment is invalid")
    verification = read_object(amendment_verification_path)
    if (
        verification.get("schema") != VERIFICATION_SCHEMA
        or verification.get("verification_hash")
        != payload_hash(verification, "verification_hash")
        or verification.get("verified") is not True
        or verification.get("errors") != []
        or verification.get("amendment_file_sha256") != sha256_file(amendment_path)
    ):
        raise ValueError("Recovery amendment verification is invalid")
    diagnostic = read_object(diagnostic_path)
    trigger = amendment.get("triggering_failure") or {}
    expected_block_id = str(trigger.get("block_id") or "")
    if not expected_block_id:
        raise ValueError("Recovery amendment lacks the failed block ID")
    if (
        diagnostic.get("schema") != DIAGNOSTIC_SCHEMA
        or diagnostic.get("diagnostic_hash")
        != payload_hash(diagnostic, "diagnostic_hash")
        or diagnostic.get("diagnostic_hash") != trigger.get("diagnostic_hash")
        or sha256_file(diagnostic_path) != trigger.get("diagnostic_file_sha256")
    ):
        raise ValueError("Recovery diagnostic is invalid")

    source_manifest = read_object(recovery_source_manifest_path)
    if source_manifest.get("schema") != SOURCE_SCHEMA or source_manifest.get(
        "manifest_hash"
    ) != payload_hash(source_manifest, "manifest_hash"):
        raise ValueError("Recovery source manifest is invalid")
    for relative, expected in (source_manifest.get("file_hashes") or {}).items():
        path = recovery_root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Recovery source changed: {relative}")
    if source_manifest.get("amendment_hash") != amendment.get("amendment_hash"):
        raise ValueError("Recovery source/amendment binding mismatch")

    contract = read_object(block_contract_path)
    if (
        contract.get("contract_hash") != payload_hash(contract, "contract_hash")
        or contract.get("block_id") != expected_block_id
        or diagnostic.get("block_id") != expected_block_id
    ):
        raise ValueError("Recovery block contract binding mismatch")
    if contract.get("source_snapshot_sha256") != diagnostic.get(
        "frozen_execution", {}
    ).get("source_snapshot_sha256"):
        raise ValueError("Recovery source snapshot binding mismatch")
    verify_source_snapshot(
        source_root,
        expected_source_sha256=str(contract["source_snapshot_sha256"]),
        expected_manifest_file_sha256=str(contract["source_manifest_file_sha256"]),
    )

    points = mount_points()
    required_points = {"/opt/nautilus", "/recovery", "/memory", "/output"}
    if not required_points <= points:
        raise ValueError(f"Recovery mount set is incomplete: {sorted(points)}")
    for forbidden in (
        "/workspace",
        "/task",
        "/fixed/train_view",
        "/fixed/evaluator_view",
        "/secrets/mlevolve.env",
    ):
        if forbidden in points or _path_exists(forbidden):
            raise ValueError(f"Forbidden recovery mount is present: {forbidden}")
    if environment_has_solver_secret():
        raise ValueError("Solver secret is present in recovery environment")
    if list(Path("/dev").glob("nvidia*")):
        raise ValueError("GPU device is visible in recovery environment")
    for root in (source_root, recovery_root, bundle_root):
        _assert_read_only(root)
    output_probe = output_root / ".formal_recovery_output_probe"
    output_probe.write_text("writable", encoding="utf-8")
    output_probe.unlink()

    if (output_root / "STATE").read_text(encoding="utf-8").strip() != (
        "training_launcher_failed"
    ):
        raise ValueError("Recovery input is not the preserved launcher failure")
    if (
        int(
            (output_root / "TRAINING_LAUNCHER_EXIT_CODE")
            .read_text(encoding="utf-8")
            .strip()
        )
        == 0
    ):
        raise ValueError("Recovery input has a successful launcher exit")
    for forbidden_name in (
        "TRAINING_MANIFEST.json",
        "TRAINING_COMPLETE",
        "TRAINING_FINALIZER_RECOVERY.json",
        "EVALUATION_SUMMARY.json",
        "EVALUATION_COMPLETE",
        "EVALUATOR_LAUNCHER_EXIT_CODE",
    ):
        if (output_root / forbidden_name).exists():
            raise ValueError(f"Recovery output already contains {forbidden_name}")
    terminal_paths = [
        path for path in output_root.rglob("*.json") if path.name in TERMINAL_FILENAMES
    ]
    if terminal_paths:
        raise ValueError("Recovery input contains terminal score artifacts")

    pre_inventory = _tree_inventory(output_root)
    recovery_contract = amendment.get("preserved_block_recovery") or {}
    if len(pre_inventory) != recovery_contract.get(
        "required_pre_recovery_file_count"
    ) or _inventory_hash(pre_inventory) != recovery_contract.get(
        "required_pre_recovery_tree_sha256"
    ):
        raise ValueError("Preserved pre-recovery output tree changed")

    training = finalize_training_block(
        output_root,
        block_contract_path,
        bundle_root,
    )
    disposition = _condition_disposition(training)
    if disposition != EXPECTED_DISPOSITION or disposition != recovery_contract.get(
        "required_condition_disposition"
    ):
        raise ValueError("Recovered condition disposition mismatch")
    if (
        training.get("successful_condition_count") != 3
        or training.get("failed_condition_count") != 2
    ):
        raise ValueError("Recovered success/failure count mismatch")
    full = (training.get("conditions") or {}).get("full_decision_admissibility") or {}
    denial = full.get("selected_runtime_protocol_denial") or {}
    expected_denial_reason = str(
        (diagnostic.get("failure") or {}).get("protocol_observation_reason") or ""
    )
    if (
        not expected_denial_reason
        or full.get("terminal_scoring_authorized") is not False
        or full.get("candidate_reexecution_authorized") is not False
        or denial.get("observation_reason") != expected_denial_reason
        or denial.get("retry_as_infrastructure_authorized") is not False
    ):
        raise ValueError("Recovered Full denial semantics mismatch")
    if any(path.name in TERMINAL_FILENAMES for path in output_root.rglob("*.json")):
        raise ValueError("Recovery produced a terminal score artifact")

    source_postrun = verify_source_snapshot(
        source_root,
        expected_source_sha256=str(contract["source_snapshot_sha256"]),
        expected_manifest_file_sha256=str(contract["source_manifest_file_sha256"]),
    )
    source_postrun_path = output_root / "SOURCE_POSTRUN_RECOVERY.json"
    _write_exclusive(
        source_postrun_path,
        source_postrun,
        hash_field="report_hash",
    )

    recovery_identity = {
        "execution_kind": "devpod",
        "namespace": str(pod_namespace),
        "pod_name": str(pod_name),
        "pod_uid": str(pod_uid),
    }
    finished_at = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    receipt = _write_exclusive(
        output_root / "TRAINING_FINALIZER_RECOVERY.json",
        {
            "schema": SCHEMA,
            "status": "deterministic_preterminal_finalizer_recovered",
            "block_id": expected_block_id,
            "recovery_pod_identity": recovery_identity,
            "amendment_hash": amendment["amendment_hash"],
            "amendment_file_sha256": sha256_file(amendment_path),
            "amendment_verification_hash": verification["verification_hash"],
            "amendment_verification_file_sha256": sha256_file(
                amendment_verification_path
            ),
            "diagnostic_hash": diagnostic["diagnostic_hash"],
            "diagnostic_file_sha256": sha256_file(diagnostic_path),
            "recovery_source_manifest_hash": source_manifest["manifest_hash"],
            "recovery_source_manifest_sha256": sha256_file(
                recovery_source_manifest_path
            ),
            "original_source_snapshot_sha256": contract["source_snapshot_sha256"],
            "block_contract_hash": contract["contract_hash"],
            "pre_recovery_file_count": len(pre_inventory),
            "pre_recovery_tree_sha256": _inventory_hash(pre_inventory),
            "training_manifest_hash": training["manifest_hash"],
            "training_manifest_sha256": sha256_file(
                output_root / "TRAINING_MANIFEST.json"
            ),
            "condition_disposition": disposition,
            "successful_condition_count": 3,
            "failed_condition_count": 2,
            "terminal_metric_observed": False,
            "terminal_score_values_inspected": False,
            "agent_reexecuted": False,
            "candidate_code_reexecuted": False,
            "full_condition_reexecuted": False,
            "source_score_inherited": False,
            "target_history_used": False,
            "cpu_only": True,
            "gpu_visible": False,
            "terminal_labels_mounted": False,
            "solver_secret_mounted": False,
            "memory_bundle_read_only": True,
            "mount_points": sorted(points),
            "finished_at_utc": finished_at,
            "receipt_hash": "",
        },
        hash_field="receipt_hash",
    )
    _write_marker(output_root / "RECOVERY_FINISHED_AT", finished_at + "\n")
    _write_marker(output_root / "RECOVERY_COMPLETE")
    _write_marker(output_root / "TRAINING_COMPLETE")
    for path in (
        source_postrun_path,
        output_root / "TRAINING_FINALIZER_RECOVERY.json",
        output_root / "RECOVERY_FINISHED_AT",
        output_root / "RECOVERY_COMPLETE",
        output_root / "TRAINING_COMPLETE",
    ):
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--block-contract", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-verification", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--recovery-source-manifest", type=Path, required=True)
    parser.add_argument("--pod-name", required=True)
    parser.add_argument("--pod-namespace", required=True)
    parser.add_argument("--pod-uid", required=True)
    args = parser.parse_args()
    receipt = recover_preterminal_finalizer(
        output_root=args.output_root,
        block_contract_path=args.block_contract,
        bundle_root=args.bundle_root,
        source_root=args.source_root,
        recovery_root=args.recovery_root,
        amendment_path=args.amendment,
        amendment_verification_path=args.amendment_verification,
        diagnostic_path=args.diagnostic,
        recovery_source_manifest_path=args.recovery_source_manifest,
        pod_name=args.pod_name,
        pod_namespace=args.pod_namespace,
        pod_uid=args.pod_uid,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "SOURCE_SCHEMA", "recover_preterminal_finalizer"]
