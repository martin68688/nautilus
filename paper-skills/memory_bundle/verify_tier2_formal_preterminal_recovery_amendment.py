#!/usr/bin/env python3
"""Verify the result-blind r6 pre-terminal finalizer recovery amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tier2_formal_revision_chain import r9_binds_current_source


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_"
    "preterminal_finalizer_recovery_amendment_v1"
)
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_"
    "preterminal_finalizer_recovery_amendment_verification_v1"
)
EXPECTED_PARENT_ID = "wp8-tier2-formal-3protocol-6system-r5-postfailure"
EXPECTED_PRIMARY_CONTRAST = (
    "full_decision_admissibility minus no_memory, paired within task and " "agent_seed"
)
EXPECTED_DISPOSITION = {
    "full_decision_admissibility": ("pre_terminal_failure:authority_denial"),
    "authority_only": "pre_terminal_failure:retained_run_failure",
    "no_memory": "training_complete_unscored",
    "global_validity_bit": "training_complete_unscored",
    "flat_relevance_memory": "training_complete_unscored",
}


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()


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


def verify_preterminal_recovery_amendment(
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
        payload.get("status") == "result_blind_recovery_frozen_pending_verification",
    )

    parent = payload.get("parent_preregistration") or {}
    parent_path = repo / str(parent.get("path") or "")
    check("parent_id", parent.get("preregistration_id") == EXPECTED_PARENT_ID)
    check("parent_exists", parent_path.is_file())
    if parent_path.is_file():
        check(
            "parent_file_hash",
            _file_sha256(parent_path) == parent.get("file_sha256"),
        )
        check(
            "parent_payload_id",
            _read(parent_path).get("preregistration_id") == EXPECTED_PARENT_ID,
        )

    trigger = payload.get("triggering_failure") or {}
    diagnostic_path = repo / str(trigger.get("diagnostic_path") or "")
    check("diagnostic_exists", diagnostic_path.is_file())
    diagnostic: dict[str, Any] = {}
    if diagnostic_path.is_file():
        diagnostic = _read(diagnostic_path)
        check(
            "diagnostic_file_hash",
            _file_sha256(diagnostic_path) == trigger.get("diagnostic_file_sha256"),
        )
        check(
            "diagnostic_internal_hash",
            diagnostic.get("diagnostic_hash")
            == _payload_hash(diagnostic, "diagnostic_hash")
            == trigger.get("diagnostic_hash"),
        )
        check(
            "diagnostic_block_binding",
            diagnostic.get("block_id") == trigger.get("block_id"),
        )
        preserved = diagnostic.get("preserved_output") or {}
        check(
            "diagnostic_terminal_unobserved",
            preserved.get("terminal_metric_observed") is False
            and preserved.get("terminal_score_file_count") == 0,
        )
        check(
            "diagnostic_scores_uninspected",
            preserved.get("terminal_score_values_inspected") is False,
        )
        required = diagnostic.get("required_disposition") or {}
        check(
            "diagnostic_full_failure_retained",
            required.get("full_condition_classification") == "authority_denial"
            and required.get("full_condition_candidate_reexecution_authorized")
            is False,
        )
    check(
        "trigger_terminal_unobserved",
        trigger.get("terminal_metric_observed") is False,
    )
    check(
        "trigger_scores_uninspected",
        trigger.get("terminal_score_values_inspected") is False,
    )
    check("trigger_evaluator_unstarted", trigger.get("evaluator_started") is False)
    check("trigger_training_absent", trigger.get("training_pod_not_found") is True)

    scientific = payload.get("scientific_objective") or {}
    r1_path = (
        repo
        / "coordination"
        / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r1.json"
    )
    r1 = _read(r1_path) if r1_path.is_file() else {}
    check(
        "primary_contrast_unchanged",
        scientific.get("primary_contrast")
        == (r1.get("analysis_plan") or {}).get("primary_online_contrast")
        == EXPECTED_PRIMARY_CONTRAST,
    )
    check(
        "effect_not_authorized",
        scientific.get("effect_claim_authorized") is False,
    )

    semantics = payload.get("semantic_clarification") or {}
    check(
        "blocked_is_authority_denial",
        "Authority-denial" in str(semantics.get("blocked_observation") or "")
        and "no terminal system score"
        in str(semantics.get("blocked_observation") or ""),
    )
    check(
        "tampering_remains_fatal",
        "block-fatal" in str(semantics.get("malformed_or_tampered_observation") or ""),
    )
    check(
        "retry_forbidden",
        "neither" in str(semantics.get("retry_rule") or "").lower()
        and "may rerun" in str(semantics.get("retry_rule") or "").lower(),
    )

    correction = payload.get("implementation_correction") or {}
    source_paths = tuple(
        sorted(
            key
            for key, value in correction.items()
            if "/" in str(key) and isinstance(value, str) and len(value) == 64
        )
    )
    check("implementation_hash_paths_present", len(source_paths) >= 10)
    for relative in source_paths:
        source_path = repo / relative
        check(f"source_exists:{relative}", source_path.is_file())
        if source_path.is_file():
            current_hash = _file_sha256(source_path)
            check(
                f"source_hash:{relative}",
                current_hash == correction.get(relative)
                or r9_binds_current_source(
                    repo,
                    relative,
                    official_amendment_path=path,
                    ancestor_revision="r6",
                ),
            )
    check(
        "positive_path_unchanged",
        correction.get("positive_path_unchanged") is True,
    )
    check(
        "blocked_path_is_failure",
        correction.get("blocked_path_becomes_condition_failure") is True,
    )
    check(
        "tampered_path_is_fatal",
        correction.get("tampered_blocked_path_remains_block_fatal") is True,
    )
    overlay_paths = list(correction.get("recovery_overlay_paths") or [])
    check("overlay_paths_present", len(overlay_paths) >= 20)
    check("overlay_paths_sorted", overlay_paths == sorted(set(overlay_paths)))
    check(
        "overlay_binds_amendment",
        "coordination/decision_admissibility_wp8_tier2_formal_"
        "preregistration_20260723_r6.json" in overlay_paths,
    )
    check(
        "overlay_binds_recovery_and_evaluator",
        "mlevolve/fixed_holdout/formal_preterminal_recovery.py" in overlay_paths
        and "deploy/run_decision_admissibility_wp8_tier2_formal_"
        "evaluator_devpod.sh" in overlay_paths,
    )

    pod_specs = payload.get("recovery_pod_specs") or {}
    for role in ("recovery", "evaluator"):
        pod_path = repo / str(pod_specs.get(f"{role}_pod_yaml") or "")
        check(f"{role}_pod_yaml_exists", pod_path.is_file())
        if pod_path.is_file():
            check(
                f"{role}_pod_yaml_hash",
                _file_sha256(pod_path) == pod_specs.get(f"{role}_pod_yaml_sha256"),
            )
    check(
        "recovery_pod_name",
        pod_specs.get("recovery_pod_name") == "da-wp8-f-birds-s104729-recovery-cpu-r1",
    )
    check(
        "evaluator_pod_name",
        pod_specs.get("evaluator_pod_name") == "da-wp8-f-birds-s104729-cpu-r3",
    )

    recovery = payload.get("preserved_block_recovery") or {}
    preserved = diagnostic.get("preserved_output") or {}
    check(
        "recovery_root_bound",
        recovery.get("input_root") == preserved.get("path"),
    )
    check(
        "recovery_tree_bound",
        recovery.get("required_pre_recovery_tree_sha256")
        == preserved.get("tree_sha256_before_recovery"),
    )
    check(
        "recovery_file_count_bound",
        recovery.get("required_pre_recovery_file_count")
        == preserved.get("file_count_before_recovery"),
    )
    check("recovery_is_devpod", recovery.get("execution_kind") == "cpu_devpod")
    for field in (
        "recovery_pod_must_have_gpu",
        "recovery_pod_may_mount_terminal_labels",
        "recovery_pod_may_mount_solver_secret",
        "recovery_pod_may_mount_target_memory_bundle_read_write",
        "candidate_or_agent_reexecution",
        "full_candidate_reexecution_forbidden",
        "silent_whole_block_retry_forbidden",
    ):
        expected = field in {
            "full_candidate_reexecution_forbidden",
            "silent_whole_block_retry_forbidden",
        }
        check(f"recovery_{field}", recovery.get(field) is expected)
    check(
        "condition_disposition_exact",
        recovery.get("required_condition_disposition") == EXPECTED_DISPOSITION,
    )

    future = payload.get("future_blocks") or {}
    check("future_remaining_count", future.get("remaining_block_count") == 5)
    for field in (
        "new_runtime_source_required",
        "new_immutable_staging_and_stop_gate_required",
        "completed_r10_blocks_may_not_rerun",
        "remaining_blocks_must_use_corrected_finalizer",
        "analysis_must_report_each_block_source_and_gate_hash",
    ):
        check(f"future_{field}", future.get(field) is True)

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
        "terminal_score_value_used_to_choose_correction",
    ):
        check(f"scope_{field}", scope.get(field) is False)
    check(
        "scope_failure_classification_changed",
        scope.get("failure_classification_implementation_changed") is True,
    )
    check(
        "scope_finalizer_source_changed",
        scope.get("host_finalizer_source_changed") is True,
    )

    integrity = payload.get("analysis_integrity") or {}
    for field in (
        "unfavorable_full_outcome_is_retained",
        "full_condition_is_not_reexecuted",
        "no_completed_condition_is_reexecuted",
        "no_condition_seed_or_task_is_excluded",
        "missing_terminal_system_result_is_reported_as_failure",
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
    report = verify_preterminal_recovery_amendment(
        args.amendment, repo_root=args.repo_root
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["verified"] else 1)


if __name__ == "__main__":
    main()


__all__ = [
    "SCHEMA",
    "VERIFICATION_SCHEMA",
    "verify_preterminal_recovery_amendment",
]
