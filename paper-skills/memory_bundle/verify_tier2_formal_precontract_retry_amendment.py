#!/usr/bin/env python3
"""Verify the result-blind r9 pre-contract recovery and r5 retry amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_"
    "precontract_recovery_and_r5_retry_amendment_v1"
)
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_"
    "precontract_recovery_and_r5_retry_amendment_verification_v1"
)
PARENT_ID = "wp8-tier2-formal-3protocol-6system-r8-control-packaging-retry"
EXPECTED_PRIMARY = (
    "full_decision_admissibility minus no_memory, paired within task and " "agent_seed"
)
EXPECTED_RUNTIME_CHANGES = [
    "deploy/run_decision_admissibility_wp8_tier2_formal_evaluator_devpod.sh",
    "deploy/run_decision_admissibility_wp8_tier2_formal_training_devpod.sh",
    "mlevolve/fixed_holdout/formal_host_receipts.py",
    "mlevolve/fixed_holdout/formal_runtime.py",
]


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_precontract_retry_amendment(
    amendment_path: str | Path,
    *,
    repo_root: str | Path = ROOT,
) -> dict[str, Any]:
    path = Path(amendment_path).resolve()
    repo = Path(repo_root).resolve()
    payload = _read(path)
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: object) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(name)

    check("schema", payload.get("schema") == SCHEMA)
    check(
        "amendment_hash",
        payload.get("amendment_hash") == _payload_hash(payload, "amendment_hash"),
    )
    check(
        "status",
        payload.get("status")
        == "result_blind_precontract_recovery_and_r5_retry_frozen",
    )

    parent_ref = payload.get("parent_amendment") or {}
    parent_path = repo / str(parent_ref.get("path") or "")
    check("parent_id", parent_ref.get("preregistration_id") == PARENT_ID)
    check("parent_exists", parent_path.is_file())
    parent: dict[str, Any] = {}
    if parent_path.is_file():
        parent = _read(parent_path)
        check(
            "parent_file_hash",
            _file_sha256(parent_path) == parent_ref.get("file_sha256"),
        )
        check("parent_payload_id", parent.get("preregistration_id") == PARENT_ID)
        check(
            "parent_internal_hash",
            parent.get("amendment_hash")
            == _payload_hash(parent, "amendment_hash")
            == parent_ref.get("amendment_hash"),
        )
    parent_verification_ref = payload.get("parent_verification") or {}
    parent_verification_path = repo / str(parent_verification_ref.get("path") or "")
    check("parent_verification_exists", parent_verification_path.is_file())
    if parent_verification_path.is_file():
        parent_verification = _read(parent_verification_path)
        check(
            "parent_verification_file_hash",
            _file_sha256(parent_verification_path)
            == parent_verification_ref.get("file_sha256"),
        )
        check(
            "parent_verification_valid",
            parent_verification.get("verified") is True
            and parent_verification.get("errors") == []
            and parent_verification.get("verification_hash")
            == _payload_hash(parent_verification, "verification_hash")
            == parent_verification_ref.get("verification_hash"),
        )

    diagnostic_ref = payload.get("precontract_failure") or {}
    diagnostic_path = repo / str(diagnostic_ref.get("path") or "")
    check("diagnostic_exists", diagnostic_path.is_file())
    diagnostic: dict[str, Any] = {}
    if diagnostic_path.is_file():
        diagnostic = _read(diagnostic_path)
        check(
            "diagnostic_file_hash",
            _file_sha256(diagnostic_path) == diagnostic_ref.get("file_sha256"),
        )
        check(
            "diagnostic_internal_hash",
            diagnostic.get("diagnostic_hash")
            == _payload_hash(diagnostic, "diagnostic_hash")
            == diagnostic_ref.get("diagnostic_hash"),
        )
        partial = diagnostic.get("partial_output") or {}
        integrity = diagnostic.get("analysis_integrity") or {}
        check(
            "diagnostic_precontract",
            diagnostic.get("classification") == "staging_schema_compatibility_failure"
            and partial.get("block_contract_written") is False
            and partial.get("condition_directory_created") is False
            and partial.get("agent_generation_started") is False
            and partial.get("candidate_execution_started") is False,
        )
        check(
            "diagnostic_result_blind",
            partial.get("terminal_score_file_count") == 0
            and integrity.get("terminal_metric_observed") is False
            and integrity.get("terminal_score_values_inspected") is False
            and integrity.get("formal_effect_observation") is False,
        )
        check(
            "diagnostic_retry_requires_fresh_roots",
            integrity.get("retry_requires_amendment_new_source_staging_and_output_root")
            is True
            and integrity.get("silent_retry_permitted") is False,
        )

    scientific = payload.get("scientific_objective") or {}
    parent_scientific = parent.get("scientific_objective") or {}
    check(
        "primary_contrast_unchanged",
        scientific.get("primary_contrast")
        == parent_scientific.get("primary_contrast")
        == EXPECTED_PRIMARY,
    )
    check(
        "primary_hypothesis_unchanged",
        scientific.get("primary_hypothesis")
        == parent_scientific.get("primary_hypothesis"),
    )
    check(
        "scientific_result_blind",
        scientific.get("terminal_score_values_inspected") is False
        and scientific.get("effect_claim_authorized") is False,
    )

    recovery = payload.get("recovery_plan") or {}
    check(
        "recovery_existing_cpu_devpod",
        recovery.get("execution_kind") == "existing_cpu_controller_devpod"
        and recovery.get("controller_pod") == "da-wp8-f-controller-cpu-r4",
    )
    for field in (
        "may_mount_terminal_labels",
        "may_mount_solver_secret",
        "may_mount_memory_bundle",
        "may_execute_agent",
        "may_execute_candidate_code",
        "may_start_evaluator",
        "may_inspect_terminal_scores",
        "may_reuse_failed_output_for_formal_execution",
    ):
        check(f"recovery_{field}", recovery.get(field) is False)
    check(
        "recovery_exact_operation",
        recovery.get("operation")
        == "seal_existing_four_file_precontract_output_with_host_receipt",
    )

    retry = payload.get("retry_design") or {}
    expected_retry = {
        "source_root": "/workspace/decision-admissibility-wp8-tier2-formal-source-r13",
        "control_root": "/workspace/decision-admissibility-wp8-tier2-formal-control-r13",
        "staging_root": "/workspace/decision-admissibility-wp8-tier2-formal-staging-r15",
        "output_root": "/workspace/decision-admissibility-wp8-tier2-formal-runs-r13",
        "gate_root": (
            "/workspace/decision-admissibility-wp8-tier2-formal-"
            "staging-r15-stop-gate-r1"
        ),
        "pipeline_root": (
            "/workspace/decision-admissibility-wp8-tier2-formal-"
            "staging-r15-pipeline-r1"
        ),
        "formal_execution_revision": "r5",
        "block_id_suffix": "r5",
        "controller_pod": "da-wp8-f-controller-cpu-r5",
        "stager_pod": "decision-admissibility-wp8-tier2-formal-stager-cpu-r15",
        "remaining_block_count": 5,
        "formal_training_authorized": False,
    }
    check("retry_design_exact", retry == expected_retry)
    check(
        "retry_roots_new",
        not any(
            str(retry.get(field) or "").endswith(suffix)
            for field, suffix in (
                ("source_root", "formal-source-r12"),
                ("control_root", "formal-control-r12"),
                ("staging_root", "formal-staging-r14"),
                ("output_root", "formal-runs-r12"),
            )
        ),
    )

    correction = payload.get("implementation_correction") or {}
    check(
        "runtime_changes_exact",
        sorted(correction.get("allowed_runtime_source_changes") or [])
        == EXPECTED_RUNTIME_CHANGES,
    )
    for field in (
        "candidate_generation_semantics_changed",
        "memory_retrieval_semantics_changed",
        "condition_failure_semantics_changed",
        "terminal_evaluation_semantics_changed",
        "result_writeback_semantics_changed",
    ):
        check(f"correction_{field}", correction.get(field) is False)
    check(
        "correction_schema_only",
        correction.get("schema_compatibility_fix") is True
        and correction.get("host_precontract_receipt_fix") is True,
    )

    implementation = payload.get("implementation_files") or {}
    paths = sorted(
        key
        for key, value in implementation.items()
        if "/" in str(key) and isinstance(value, str) and len(value) == 64
    )
    check("implementation_paths_present", len(paths) >= 10)
    for relative in paths:
        source = repo / relative
        check(f"implementation_exists:{relative}", source.is_file())
        if source.is_file():
            check(
                f"implementation_hash:{relative}",
                _file_sha256(source) == implementation.get(relative),
            )

    scope = payload.get("scope") or {}
    for field in (
        "primary_contrast_changed",
        "systems_changed",
        "tasks_changed",
        "agent_seeds_changed",
        "condition_orders_changed",
        "search_budgets_changed",
        "candidate_contracts_changed",
        "holdouts_changed",
        "memory_bundles_changed",
        "memory_claim_permissions_changed",
        "oracle_algorithm_changed",
        "statistics_changed",
        "target_history_exclusion_changed",
        "source_score_inheritance_changed",
        "terminal_score_value_used_to_choose_fix",
    ):
        check(f"scope_{field}", scope.get(field) is False)
    for field in (
        "runtime_schema_compatibility_changed",
        "host_failure_receipt_changed",
        "formal_roots_changed",
        "formal_execution_revision_changed",
    ):
        check(f"scope_{field}", scope.get(field) is True)

    integrity = payload.get("analysis_integrity") or {}
    for field in (
        "all_four_completed_blocks_retained",
        "all_five_remaining_blocks_unchanged",
        "failed_r4_attempt_retained_as_non_outcome",
        "failed_r4_output_never_reused",
        "no_completed_block_reexecuted",
        "no_condition_seed_or_task_excluded",
        "failures_remain_outcomes",
        "no_imputation_from_oracle_or_source_score",
    ):
        check(f"integrity_{field}", integrity.get(field) is True)
    check(
        "integrity_effect_not_authorized",
        integrity.get("effect_claim_authorized_before_results") is False,
    )

    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "preregistration_id": payload.get("preregistration_id", ""),
        "amendment_file_sha256": _file_sha256(path),
        "diagnostic_file_sha256": (
            _file_sha256(diagnostic_path) if diagnostic_path.is_file() else ""
        ),
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": sorted(set(errors)),
        "verified": not errors,
        "verifier_source_sha256": _file_sha256(Path(__file__).resolve()),
        "verification_hash": "",
    }
    report["verification_hash"] = _payload_hash(report, "verification_hash")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_precontract_retry_amendment(
        args.amendment,
        repo_root=args.repo_root,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.write_text(encoded, encoding="utf-8")
    raise SystemExit(0 if report["verified"] else 1)


if __name__ == "__main__":
    main()


__all__ = ["EXPECTED_RUNTIME_CHANGES", "verify_precontract_retry_amendment"]
