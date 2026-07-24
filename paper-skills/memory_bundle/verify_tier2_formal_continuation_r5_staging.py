#!/usr/bin/env python3
"""Independently verify r5 continuation staging and pre-contract recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping


import verify_tier2_formal_continuation_staging as base_verify
from build_tier2_formal_continuation_r5_staging import BINDING_SCHEMA
from fixed_holdout.common import sha256_file
from fixed_holdout.formal_host_receipts import PRECONTRACT_ABORT_SCHEMA
from fixed_holdout.formal_runtime import payload_hash, read_object
from verify_tier2_formal_precontract_retry_amendment import (
    EXPECTED_RUNTIME_CHANGES,
    verify_precontract_retry_amendment,
)


SCHEMA = "decision_admissibility_wp8_tier2_formal_continuation_r5_stop_gate_v1"
REVISION = "r5"
CONTROLLER = "da-wp8-f-controller-cpu-r5"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _source_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    left = dict(before.get("file_hashes") or {})
    right = dict(after.get("file_hashes") or {})
    return sorted(
        path for path in set(left) | set(right) if left.get(path) != right.get(path)
    )


def _tree_is_read_only(root: Path) -> bool:
    return all(
        path.is_symlink()
        or not path.exists()
        or not bool(path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        for path in [root, *root.rglob("*")]
    )


def verify_r5_staging(
    staging_root: Path,
    *,
    repo_root: Path,
    r9_amendment_path: Path,
    r9_verification_path: Path,
    r7_verification_path: Path,
    recovery_receipt_path: Path,
    seal_on_success: bool = True,
) -> dict[str, Any]:
    staging_root = staging_root.resolve()
    repo_root = repo_root.resolve()
    r9_amendment_path = r9_amendment_path.resolve()
    r9_verification_path = r9_verification_path.resolve()
    r7_verification_path = r7_verification_path.resolve()
    recovery_receipt_path = recovery_receipt_path.resolve()
    amendment = read_object(r9_amendment_path)
    frozen_r9 = read_object(r9_verification_path)
    frozen_r7 = read_object(r7_verification_path)
    recovery = read_object(recovery_receipt_path)
    binding = read_object(staging_root / "STAGING_R5_BINDING.json")
    content = read_object(staging_root / "STAGING_CONTENT_MANIFEST.json")
    build = read_object(staging_root / "STAGING_BUILD_REPORT.json")

    base_verify.verify_continuation_amendment = lambda *_args, **_kwargs: dict(
        frozen_r7
    )
    base = base_verify.verify_continuation_staging(
        staging_root,
        repo_root=repo_root,
        seal_on_success=False,
    )
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: object) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(name)

    blocks = content.get("blocks_by_id") or {}
    expected_legacy_errors = {
        "revision",
        "source_diff_exact",
        "controller_name",
        *(f"template_revision:{block_id}" for block_id in blocks),
    }
    check(
        "base_only_expected_revision_errors",
        set(base.get("errors") or []) == expected_legacy_errors,
    )
    for name, passed in (base.get("checks") or {}).items():
        if name in expected_legacy_errors:
            continue
        check(f"base:{name}", passed is True)

    live_r9 = verify_precontract_retry_amendment(
        r9_amendment_path,
        repo_root=repo_root,
    )
    check("r9_live_verified", live_r9.get("verified") is True)
    check(
        "r9_frozen_exact",
        frozen_r9 == live_r9
        and frozen_r9.get("verification_hash")
        == payload_hash(frozen_r9, "verification_hash"),
    )
    check(
        "r7_historical_verification_valid",
        frozen_r7.get("verified") is True
        and frozen_r7.get("errors") == []
        and frozen_r7.get("verification_hash")
        == payload_hash(frozen_r7, "verification_hash"),
    )

    retry = amendment.get("retry_design") or {}
    check("staging_root_exact", str(staging_root) == retry.get("staging_root"))
    check("source_root_exact", content.get("source_root") == retry.get("source_root"))
    check("output_root_exact", content.get("output_root") == retry.get("output_root"))
    check(
        "revision_r5",
        content.get("formal_execution_revision") == REVISION
        and retry.get("formal_execution_revision") == REVISION,
    )
    check("five_remaining_blocks", len(blocks) == 5)
    for block_id, row in blocks.items():
        check(
            f"r5_block:{block_id}",
            block_id.endswith("-r5")
            and str(row.get("training_pod_name") or "").endswith("-r5")
            and str(row.get("evaluator_pod_name") or "").endswith("-r5"),
        )

    current_source = read_object(
        Path(content["source_root"]) / "WP8_TIER2_SOURCE_MANIFEST.json"
    )
    failed_source_root = Path(
        str((amendment.get("recovery_plan") or {}).get("failed_source_root") or "")
    )
    failed_source = read_object(failed_source_root / "WP8_TIER2_SOURCE_MANIFEST.json")
    changed_paths = _source_diff(failed_source, current_source)
    check("source_diff_exact", changed_paths == EXPECTED_RUNTIME_CHANGES)
    check(
        "source_diff_excludes_agent_generation",
        not any(
            path.startswith("mlevolve/agents/")
            or path.startswith("mlevolve/engine/agent_search")
            for path in changed_paths
        ),
    )

    check(
        "recovery_schema_hash",
        recovery.get("schema") == PRECONTRACT_ABORT_SCHEMA
        and recovery.get("report_hash") == payload_hash(recovery, "report_hash"),
    )
    check(
        "recovery_result_blind",
        recovery.get("agent_generation_started") is False
        and recovery.get("candidate_execution_started") is False
        and recovery.get("terminal_metric_observed") is False
        and recovery.get("terminal_score_values_inspected") is False
        and recovery.get("positive_writeback_observed") is False,
    )
    check(
        "recovery_failed_root_sealed",
        _tree_is_read_only(recovery_receipt_path.parent),
    )

    check(
        "binding_schema_hash",
        binding.get("schema") == BINDING_SCHEMA
        and binding.get("binding_hash") == payload_hash(binding, "binding_hash"),
    )
    check(
        "binding_content",
        binding.get("staging_content_manifest_hash") == content.get("manifest_hash")
        and binding.get("formal_execution_revision") == REVISION
        and binding.get("controller_pod") == CONTROLLER
        and binding.get("r9_amendment_hash") == amendment.get("amendment_hash")
        and binding.get("r9_verification_hash") == frozen_r9.get("verification_hash")
        and binding.get("precontract_recovery_report_hash")
        == recovery.get("report_hash")
        and binding.get("terminal_score_values_inspected") is False,
    )

    pod_rows = build.get("pod_yamls") or {}
    controller_row = pod_rows.get("formal-controller") or {}
    controller_path = Path(str(controller_row.get("path") or ""))
    controller_doc = (
        __import__("yaml").safe_load(controller_path.read_text(encoding="utf-8"))
        if controller_path.is_file()
        else {}
    )
    check(
        "controller_r5",
        (controller_doc.get("metadata") or {}).get("name") == CONTROLLER
        and controller_path.name == "formal-controller-cpu-r5.yaml",
    )

    for relative, expected in (amendment.get("implementation_files") or {}).items():
        source = repo_root / relative
        check(f"implementation_exists:{relative}", source.is_file())
        if source.is_file():
            check(f"implementation_hash:{relative}", sha256_file(source) == expected)

    combined_errors = sorted(set(errors))
    gate: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if not combined_errors else "failed",
        "formal_training_authorized": not combined_errors,
        "authorized_block_count": 5 if not combined_errors else 0,
        "authorized_online_condition_count": 25 if not combined_errors else 0,
        "authorized_oracle_count": 5 if not combined_errors else 0,
        "completed_blocks_authorized_to_rerun": False,
        "failed_r4_attempt_authorized_to_reuse": False,
        "terminal_score_values_inspected": False,
        "terminal_metric_observed_for_remaining_blocks": False,
        "effect_claim_authorized": False,
        "staging_content_manifest_hash": content.get("manifest_hash", ""),
        "r9_amendment_hash": amendment.get("amendment_hash", ""),
        "r9_verification_hash": frozen_r9.get("verification_hash", ""),
        "precontract_recovery_report_hash": recovery.get("report_hash", ""),
        "staging_r5_binding_hash": binding.get("binding_hash", ""),
        "completed_freeze_hash": base.get("completed_freeze_hash", ""),
        "source_snapshot_sha256": content.get("source_snapshot_sha256", ""),
        "parent_source_snapshot_sha256": failed_source.get("source_sha256", ""),
        "source_diff_paths": changed_paths,
        "underlying_legacy_gate_hash": base.get("gate_hash", ""),
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": combined_errors,
        "evidence": {
            "source_diff_paths": changed_paths,
            "base_expected_legacy_errors": sorted(expected_legacy_errors),
            "r9_amendment_file_sha256": sha256_file(r9_amendment_path),
            "r9_verification_file_sha256": sha256_file(r9_verification_path),
            "recovery_receipt_file_sha256": sha256_file(recovery_receipt_path),
        },
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "gate_hash": "",
    }
    gate["gate_hash"] = _payload_hash(gate, "gate_hash")
    if not combined_errors and seal_on_success:
        base_verify._seal(staging_root)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--r9-amendment", type=Path, required=True)
    parser.add_argument("--r9-verification", type=Path, required=True)
    parser.add_argument("--r7-verification", type=Path, required=True)
    parser.add_argument("--recovery-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-seal", action="store_true")
    args = parser.parse_args()
    gate = verify_r5_staging(
        args.staging_root,
        repo_root=args.repo_root,
        r9_amendment_path=args.r9_amendment,
        r9_verification_path=args.r9_verification,
        r7_verification_path=args.r7_verification,
        recovery_receipt_path=args.recovery_receipt,
        seal_on_success=not args.no_seal,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    raise SystemExit(0 if gate["formal_training_authorized"] else 1)


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "verify_r5_staging"]
