#!/usr/bin/env python3
"""Verify the r7 result-blind continuation amendment for five blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tier2_formal_revision_chain import r9_binds_current_source


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "decision_admissibility_wp8_tier2_formal_continuation_amendment_v1"
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_" "continuation_amendment_verification_v1"
)
EXPECTED_PARENT_ID = (
    "wp8-tier2-formal-3protocol-6system-r6-preterminal-finalizer-recovery"
)
EXPECTED_PRIMARY_CONTRAST = (
    "full_decision_admissibility minus no_memory, paired within task and " "agent_seed"
)
EXPECTED_REMAINING = [
    ("mlsp-2013-birds", 130363),
    ("mlsp-2013-birds", 155921),
    ("new-york-city-taxi-fare-prediction", 104729),
    ("new-york-city-taxi-fare-prediction", 130363),
    ("new-york-city-taxi-fare-prediction", 155921),
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


def verify_continuation_amendment(
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
        == "continuation_design_frozen_pending_new_staging_stop_gate",
    )

    parent = payload.get("parent_preregistration") or {}
    parent_path = repo / str(parent.get("path") or "")
    check("parent_id", parent.get("preregistration_id") == EXPECTED_PARENT_ID)
    check("parent_exists", parent_path.is_file())
    parent_payload: dict[str, Any] = {}
    if parent_path.is_file():
        parent_payload = _read(parent_path)
        check(
            "parent_file_hash",
            _file_sha256(parent_path) == parent.get("file_sha256"),
        )
        check(
            "parent_payload_id",
            parent_payload.get("preregistration_id") == EXPECTED_PARENT_ID,
        )

    completed_ref = payload.get("completed_blocks_freeze") or {}
    completed_path = repo / str(completed_ref.get("path") or "")
    check("completed_freeze_exists", completed_path.is_file())
    completed: dict[str, Any] = {}
    if completed_path.is_file():
        completed = _read(completed_path)
        check(
            "completed_freeze_file_hash",
            _file_sha256(completed_path) == completed_ref.get("file_sha256"),
        )
        check(
            "completed_freeze_internal_hash",
            completed.get("inventory_hash")
            == _payload_hash(completed, "inventory_hash")
            == completed_ref.get("inventory_hash"),
        )
        check("completed_block_count", completed.get("completed_block_count") == 4)
        check(
            "completed_condition_count",
            completed.get("completed_online_condition_count") == 20,
        )
        check(
            "completed_score_values_uninspected",
            completed.get("score_values_inspected") is False,
        )
        check(
            "completed_blocks_cannot_rerun",
            all(
                row.get("may_rerun") is False
                for row in (completed.get("blocks") or {}).values()
            ),
        )

    continuation = payload.get("continuation_design") or {}
    remaining = [
        (str(row.get("task_id") or ""), int(row.get("agent_seed") or 0))
        for row in continuation.get("remaining_blocks") or []
    ]
    check("remaining_blocks_exact", remaining == EXPECTED_REMAINING)
    check("remaining_block_count", continuation.get("remaining_block_count") == 5)
    check(
        "remaining_online_count",
        continuation.get("remaining_online_condition_count") == 25,
    )
    check("remaining_oracle_count", continuation.get("remaining_oracle_count") == 5)
    check(
        "completed_blocks_excluded_from_new_outputs",
        continuation.get("completed_blocks_excluded_from_new_output_root") is True,
    )
    check(
        "completed_blocks_not_reexecuted",
        continuation.get("completed_blocks_may_reexecute") is False,
    )

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
    check(
        "terminal_values_uninspected",
        scientific.get("terminal_score_values_inspected") is False,
    )

    overrides = payload.get("continuation_overrides") or {}
    expected_suffixes = {
        "source_root": "formal-source-r11",
        "control_root": "formal-control-r11",
        "staging_root": "formal-staging-r13",
        "output_root": "formal-runs-r11",
        "gate_root": "formal-staging-r13-stop-gate-r1",
        "pipeline_root": "formal-staging-r13-pipeline-r1",
    }
    for field, suffix in expected_suffixes.items():
        check(
            f"override_{field}",
            str(overrides.get(field) or "").endswith(suffix),
        )
    check(
        "override_revision",
        overrides.get("formal_execution_revision") == "r4",
    )
    check("override_block_suffix", overrides.get("block_id_suffix") == "r4")
    check(
        "override_controller",
        overrides.get("controller_pod") == "da-wp8-f-controller-cpu-r4",
    )
    check(
        "training_not_authorized_before_gate",
        overrides.get("formal_training_authorized") is False,
    )

    implementation = payload.get("implementation_correction") or {}
    source_paths = tuple(
        sorted(
            key
            for key, value in implementation.items()
            if "/" in str(key) and isinstance(value, str) and len(value) == 64
        )
    )
    check("implementation_hash_paths_present", len(source_paths) >= 8)
    for relative in source_paths:
        source_path = repo / relative
        check(f"source_exists:{relative}", source_path.is_file())
        if source_path.is_file():
            current_hash = _file_sha256(source_path)
            check(
                f"source_hash:{relative}",
                current_hash == implementation.get(relative)
                or r9_binds_current_source(
                    repo,
                    relative,
                    official_amendment_path=path,
                    ancestor_revision="r7",
                ),
            )
    check(
        "candidate_generation_semantics_unchanged",
        implementation.get("candidate_generation_semantics_changed") is False,
    )
    check(
        "failure_classification_fix_retained",
        implementation.get("blocked_observation_condition_failure_fix") is True,
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
        "terminal_score_value_used_to_choose_continuation",
    ):
        check(f"scope_{field}", scope.get(field) is False)
    check("scope_source_changed", scope.get("runtime_source_changed") is True)
    check("scope_roots_changed", scope.get("formal_roots_changed") is True)

    integrity = payload.get("analysis_integrity") or {}
    for field in (
        "all_four_completed_blocks_retained",
        "all_five_remaining_blocks_run_once",
        "no_completed_block_reexecuted",
        "no_condition_seed_or_task_excluded",
        "failures_remain_outcomes",
        "no_imputation_from_oracle_or_source_score",
        "mixed_source_hashes_reported_per_block",
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
        "completed_freeze_file_sha256": (
            _file_sha256(completed_path) if completed_path.is_file() else ""
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
    report = verify_continuation_amendment(args.amendment, repo_root=args.repo_root)
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
    "verify_continuation_amendment",
]
