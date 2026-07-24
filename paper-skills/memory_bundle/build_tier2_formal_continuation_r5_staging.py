#!/usr/bin/env python3
"""Build r5 continuation staging after sealing the pre-contract r4 abort."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping


import build_tier2_formal_continuation_staging as base
from fixed_holdout.common import sha256_file
from fixed_holdout.formal_host_receipts import PRECONTRACT_ABORT_SCHEMA
from fixed_holdout.formal_runtime import payload_hash, read_object
from verify_tier2_formal_precontract_retry_amendment import (
    verify_precontract_retry_amendment,
)


BINDING_SCHEMA = "decision_admissibility_wp8_tier2_formal_continuation_r5_binding_v1"
REVISION = "r5"
CONTROLLER_NAME = "da-wp8-f-controller-cpu-r5"
CONTROLLER_YAML = "formal-controller-cpu-r5.yaml"


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tree_is_read_only(root: Path) -> bool:
    paths = [root, *root.rglob("*")]
    return all(
        path.is_symlink()
        or not path.exists()
        or not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for path in paths
    )


def _verify_recovery(
    receipt_path: Path,
    *,
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = read_object(receipt_path)
    diagnostic_ref = amendment.get("precontract_failure") or {}
    recovery = amendment.get("recovery_plan") or {}
    collector_hash = (amendment.get("implementation_files") or {}).get(
        "mlevolve/fixed_holdout/formal_host_receipts.py"
    )
    if (
        receipt.get("schema") != PRECONTRACT_ABORT_SCHEMA
        or receipt.get("report_hash") != payload_hash(receipt, "report_hash")
        or receipt.get("block_id") != diagnostic_ref.get("block_id")
        or receipt.get("classification") != "staging_schema_compatibility_failure"
        or receipt.get("runtime_block_contract_written") is not False
        or receipt.get("agent_generation_started") is not False
        or receipt.get("candidate_execution_started") is not False
        or receipt.get("terminal_metric_observed") is not False
        or receipt.get("terminal_score_values_inspected") is not False
        or receipt.get("positive_writeback_observed") is not False
        or receipt.get("reuse_for_formal_execution") is not False
        or receipt.get("collector_source_sha256") != collector_hash
        or str(receipt_path) != recovery.get("receipt_path")
        or not _tree_is_read_only(receipt_path.parent)
    ):
        raise ValueError("Pre-contract recovery receipt is invalid")
    return receipt


def build_r5_staging(
    *,
    source_root: Path,
    artifact_root: Path,
    parent_staging_root: Path,
    parent_gate_path: Path,
    completed_output_root: Path,
    staging_root: Path,
    output_root: Path,
    preregistration_paths: tuple[Path, ...],
    continuation_amendment_path: Path,
    continuation_verification_path: Path,
    completed_freeze_path: Path,
    r9_amendment_path: Path,
    r9_verification_path: Path,
    recovery_receipt_path: Path,
) -> dict[str, Any]:
    r9_amendment_path = r9_amendment_path.resolve()
    r9_verification_path = r9_verification_path.resolve()
    recovery_receipt_path = recovery_receipt_path.resolve()
    amendment = read_object(r9_amendment_path)
    live = verify_precontract_retry_amendment(
        r9_amendment_path,
        repo_root=Path(__file__).resolve().parents[2],
    )
    frozen = read_object(r9_verification_path)
    if (
        live.get("verified") is not True
        or frozen != live
        or frozen.get("verification_hash") != payload_hash(frozen, "verification_hash")
    ):
        raise ValueError("r9 amendment verification failed")
    receipt = _verify_recovery(recovery_receipt_path, amendment=amendment)

    r7_frozen = read_object(continuation_verification_path)
    if (
        r7_frozen.get("verified") is not True
        or r7_frozen.get("errors") != []
        or r7_frozen.get("verification_hash")
        != payload_hash(r7_frozen, "verification_hash")
        or r7_frozen.get("amendment_file_sha256")
        != sha256_file(continuation_amendment_path)
    ):
        raise ValueError("Historical r7 verification is invalid")

    # r7 was verified before the result-blind compatibility correction.  Its
    # frozen verification remains the scientific-design authority; r9 binds
    # the changed control/runtime files and fresh r5 roots independently.
    base.verify_continuation_amendment = lambda *_args, **_kwargs: dict(r7_frozen)
    base.REVISION = REVISION
    base.CONTROLLER_NAME = CONTROLLER_NAME
    base.CONTROLLER_YAML = CONTROLLER_YAML
    result = base.build_continuation_staging(
        source_root=source_root,
        artifact_root=artifact_root,
        parent_staging_root=parent_staging_root,
        parent_gate_path=parent_gate_path,
        completed_output_root=completed_output_root,
        staging_root=staging_root,
        output_root=output_root,
        preregistration_paths=preregistration_paths,
        continuation_amendment_path=continuation_amendment_path,
        continuation_verification_path=continuation_verification_path,
        completed_freeze_path=completed_freeze_path,
    )
    content = read_object(staging_root / "STAGING_CONTENT_MANIFEST.json")
    binding: dict[str, Any] = {
        "schema": BINDING_SCHEMA,
        "status": "r5_content_built_pending_independent_stop_gate",
        "staging_root": str(staging_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_root": str(source_root.resolve()),
        "formal_execution_revision": REVISION,
        "controller_pod": CONTROLLER_NAME,
        "staging_content_manifest_hash": content["manifest_hash"],
        "r9_amendment_hash": amendment["amendment_hash"],
        "r9_amendment_file_sha256": sha256_file(r9_amendment_path),
        "r9_verification_hash": frozen["verification_hash"],
        "r9_verification_file_sha256": sha256_file(r9_verification_path),
        "precontract_recovery_report_hash": receipt["report_hash"],
        "precontract_recovery_file_sha256": sha256_file(recovery_receipt_path),
        "failed_r4_output_reused": False,
        "terminal_score_values_inspected": False,
        "remaining_block_count": 5,
        "binding_hash": "",
    }
    binding["binding_hash"] = payload_hash(binding, "binding_hash")
    _write_exclusive(staging_root / "STAGING_R5_BINDING.json", binding)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-staging-root", type=Path, required=True)
    parser.add_argument("--parent-gate", type=Path, required=True)
    parser.add_argument("--completed-output-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, action="append", required=True)
    parser.add_argument("--continuation-amendment", type=Path, required=True)
    parser.add_argument("--continuation-verification", type=Path, required=True)
    parser.add_argument("--completed-freeze", type=Path, required=True)
    parser.add_argument("--r9-amendment", type=Path, required=True)
    parser.add_argument("--r9-verification", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = build_r5_staging(
        source_root=args.source_root,
        artifact_root=args.artifact_root,
        parent_staging_root=args.parent_staging_root,
        parent_gate_path=args.parent_gate,
        completed_output_root=args.completed_output_root,
        staging_root=args.staging_root,
        output_root=args.output_root,
        preregistration_paths=tuple(args.preregistration),
        continuation_amendment_path=args.continuation_amendment,
        continuation_verification_path=args.continuation_verification,
        completed_freeze_path=args.completed_freeze,
        r9_amendment_path=args.r9_amendment,
        r9_verification_path=args.r9_verification,
        recovery_receipt_path=args.recovery_receipt,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["BINDING_SCHEMA", "CONTROLLER_NAME", "REVISION", "build_r5_staging"]
