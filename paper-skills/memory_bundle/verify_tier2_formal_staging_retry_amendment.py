#!/usr/bin/env python3
"""Verify the result-blind r8 continuation-staging retry amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "decision_admissibility_wp8_tier2_formal_staging_retry_amendment_v1"
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_staging_retry_" "amendment_verification_v1"
)
PARENT_ID = "wp8-tier2-formal-3protocol-6system-r7-five-block-continuation"
EXPECTED_PRIMARY = (
    "full_decision_admissibility minus no_memory, paired within task and " "agent_seed"
)
EXPECTED_MISSING = [
    "deploy/devpod-decision-admissibility-wp8-tier2-formal-"
    "recovered-evaluator-cpu-r1.yaml",
    "deploy/devpod-decision-admissibility-wp8-tier2-formal-recovery-cpu-r1.yaml",
    "deploy/run_decision_admissibility_wp8_tier2_formal_staging_pipeline.sh",
    "deploy/stage_decision_admissibility_wp8_tier2_formal.sh",
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


def verify_staging_retry_amendment(
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
        == "result_blind_control_packaging_retry_frozen_pending_stop_gate",
    )

    parent_ref = payload.get("parent_continuation") or {}
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

    failure_ref = payload.get("failed_attempt") or {}
    failure_path = repo / str(failure_ref.get("path") or "")
    check("failure_diagnostic_exists", failure_path.is_file())
    diagnostic: dict[str, Any] = {}
    if failure_path.is_file():
        diagnostic = _read(failure_path)
        check(
            "failure_diagnostic_file_hash",
            _file_sha256(failure_path) == failure_ref.get("file_sha256"),
        )
        check(
            "failure_diagnostic_internal_hash",
            diagnostic.get("diagnostic_hash")
            == _payload_hash(diagnostic, "diagnostic_hash")
            == failure_ref.get("diagnostic_hash"),
        )
        integrity = diagnostic.get("integrity") or {}
        check(
            "failure_before_training",
            diagnostic.get("phase") == "control_targeted_regression"
            and integrity.get("formal_training_authorized") is False
            and integrity.get("training_or_evaluator_pod_created") is False
            and integrity.get("remaining_block_started") is False,
        )
        check(
            "failure_result_blind",
            integrity.get("terminal_metric_observed") is False
            and integrity.get("terminal_score_values_inspected") is False,
        )
        check(
            "failure_roots_not_reusable",
            integrity.get("failed_roots_may_be_reused") is False
            and integrity.get("fresh_roots_required") is True,
        )
        check(
            "failure_downstream_absent",
            (diagnostic.get("downstream_absence") or {})
            and not any((diagnostic.get("downstream_absence") or {}).values()),
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
        "scores_uninspected", scientific.get("terminal_score_values_inspected") is False
    )
    check("effect_not_authorized", scientific.get("effect_claim_authorized") is False)

    retry = payload.get("retry_overrides") or {}
    expected_retry = {
        "source_root": "/workspace/decision-admissibility-wp8-tier2-formal-source-r12",
        "control_root": "/workspace/decision-admissibility-wp8-tier2-formal-control-r12",
        "staging_root": "/workspace/decision-admissibility-wp8-tier2-formal-staging-r14",
        "output_root": "/workspace/decision-admissibility-wp8-tier2-formal-runs-r12",
        "gate_root": (
            "/workspace/decision-admissibility-wp8-tier2-formal-"
            "staging-r14-stop-gate-r1"
        ),
        "pipeline_root": (
            "/workspace/decision-admissibility-wp8-tier2-formal-"
            "staging-r14-pipeline-r1"
        ),
        "formal_execution_revision": "r4",
        "block_id_suffix": "r4",
        "controller_pod": "da-wp8-f-controller-cpu-r4",
        "stager_pod": "decision-admissibility-wp8-tier2-formal-stager-cpu-r14",
        "formal_training_authorized": False,
    }
    check("retry_overrides_exact", retry == expected_retry)
    old_retry = parent.get("continuation_overrides") or {}
    check(
        "retry_roots_fresh",
        all(
            retry.get(field) != old_retry.get(field)
            for field in (
                "source_root",
                "control_root",
                "staging_root",
                "output_root",
                "gate_root",
                "pipeline_root",
                "stager_pod",
            )
        ),
    )
    check(
        "execution_revision_unchanged",
        retry.get("formal_execution_revision")
        == old_retry.get("formal_execution_revision")
        == "r4"
        and retry.get("block_id_suffix") == old_retry.get("block_id_suffix") == "r4"
        and retry.get("controller_pod") == old_retry.get("controller_pod"),
    )

    correction = payload.get("control_packaging_correction") or {}
    missing = sorted(correction.get("missing_control_paths") or [])
    added = sorted(correction.get("added_control_paths") or [])
    diagnostic_missing = sorted(
        ((diagnostic.get("failure") or {}).get("missing_control_paths") or [])
    )
    check("missing_paths_bound", missing == diagnostic_missing == EXPECTED_MISSING)
    check("missing_paths_added", added == missing)
    check("missing_path_count", len(missing) == 4)
    check("runtime_source_unchanged", correction.get("runtime_source_changed") is False)
    check(
        "source_snapshot_identity_required",
        correction.get("required_source_sha256")
        == ((diagnostic.get("failed_source_snapshot") or {}).get("source_sha256")),
    )
    check("control_only", correction.get("correction_scope") == "control_archive_only")

    implementation = payload.get("implementation_files") or {}
    implementation_paths = sorted(
        key
        for key, value in implementation.items()
        if "/" in str(key) and isinstance(value, str) and len(value) == 64
    )
    check("implementation_paths_present", len(implementation_paths) >= 6)
    for relative in implementation_paths:
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
        "runtime_source_changed",
        "terminal_score_value_used_to_choose_retry",
    ):
        check(f"scope_{field}", scope.get(field) is False)
    for field in ("formal_roots_changed", "control_packaging_changed"):
        check(f"scope_{field}", scope.get(field) is True)

    integrity = payload.get("analysis_integrity") or {}
    for field in (
        "all_four_completed_blocks_retained",
        "all_five_remaining_blocks_unchanged",
        "no_completed_block_reexecuted",
        "no_remaining_block_started_during_failed_staging",
        "failed_staging_roots_preserved_read_only",
        "fresh_roots_required",
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
        "failure_diagnostic_file_sha256": (
            _file_sha256(failure_path) if failure_path.is_file() else ""
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
    report = verify_staging_retry_amendment(
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


__all__ = ["verify_staging_retry_amendment"]
