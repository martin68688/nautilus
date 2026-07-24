#!/usr/bin/env python3
"""Verify the result-blind r5 amendment after the failed r8 formal block."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "decision_admissibility_wp8_tier2_formal_postfailure_amendment_v1"
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_postfailure_amendment_verification_v1"
)
EXPECTED_PARENT_ID = "wp8-tier2-formal-3protocol-6system-r4"
EXPECTED_PRIMARY_CONTRAST = (
    "full_decision_admissibility minus no_memory, paired within task and "
    "agent_seed"
)
EXPECTED_STRATEGY_MAPPING = {
    "stratified_random": "stratified_random",
    "grouped_multilabel_stratified": "grouped",
    "chronological_deterministic_sha256_sample": "chronological",
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


def verify_postfailure_amendment(
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
        payload.get("amendment_hash")
        == _payload_hash(payload, "amendment_hash"),
    )
    check(
        "status",
        payload.get("status")
        == "postfailure_design_frozen_pending_new_staging_stop_gate",
    )

    parent = payload.get("parent_preregistration") or {}
    parent_path = repo / str(parent.get("path") or "")
    check("parent_id", parent.get("preregistration_id") == EXPECTED_PARENT_ID)
    check("parent_exists", parent_path.is_file())
    if parent_path.is_file():
        check("parent_file_hash", _file_sha256(parent_path) == parent.get("file_sha256"))
        check(
            "parent_payload_id",
            _read(parent_path).get("preregistration_id") == EXPECTED_PARENT_ID,
        )

    failed = payload.get("failed_formal_attempt") or {}
    diagnostic_path = repo / str(failed.get("diagnostic_path") or "")
    check("failure_diagnostic_exists", diagnostic_path.is_file())
    diagnostic: dict[str, Any] = {}
    if diagnostic_path.is_file():
        diagnostic = _read(diagnostic_path)
        check(
            "failure_diagnostic_file_hash",
            _file_sha256(diagnostic_path)
            == failed.get("diagnostic_file_sha256"),
        )
        check(
            "failure_diagnostic_internal_hash",
            diagnostic.get("diagnostic_hash")
            == _payload_hash(diagnostic, "diagnostic_hash")
            == failed.get("diagnostic_hash"),
        )
        check(
            "failure_output_tree_bound",
            diagnostic.get("preserved_output", {}).get("tree_sha256")
            == failed.get("output_tree_sha256"),
        )
        check(
            "failure_block_bound",
            diagnostic.get("block_id") == failed.get("block_id"),
        )
    check("failure_terminal_metric_observed", failed.get("terminal_metric_observed") is True)
    check("failure_not_premetric", diagnostic.get("pre_metric_abort") is False)
    check(
        "failure_score_values_uninspected",
        failed.get("score_values_inspected_for_this_amendment") is False
        and diagnostic.get("score_values_inspected_during_recovery") is False,
    )
    check("failure_not_reused", failed.get("reuse_for_formal_execution") is False)
    check("failure_excluded_from_effect", failed.get("included_in_effect_analysis") is False)
    check("failure_no_result_fact", failed.get("normal_result_fact_published") is False)

    scientific = payload.get("scientific_objective") or {}
    r1_path = (
        repo
        / "coordination"
        / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r1.json"
    )
    r1 = _read(r1_path) if r1_path.is_file() else {}
    frozen_contrast = (r1.get("analysis_plan") or {}).get(
        "primary_online_contrast"
    )
    check(
        "primary_contrast_unchanged",
        scientific.get("primary_contrast")
        == frozen_contrast
        == EXPECTED_PRIMARY_CONTRAST,
    )
    check(
        "primary_restatement_declared",
        scientific.get("status_relative_to_r1")
        == "unchanged_restatement_of_the_frozen_r1_primary_online_contrast",
    )
    check(
        "zero_adoption_not_excluded",
        "cannot support" in str(scientific.get("zero_adoption_rule") or "")
        and "excluded" in str(scientific.get("zero_adoption_rule") or ""),
    )

    scope = payload.get("scope") or {}
    for field in (
        "systems_changed",
        "tasks_changed",
        "agent_seeds_changed",
        "condition_orders_changed",
        "search_budgets_changed",
        "candidate_contracts_changed",
        "holdouts_changed",
        "memory_claim_permissions_changed",
        "oracle_changed",
        "statistics_changed",
        "primary_contrast_changed",
        "target_history_exclusion_changed",
        "source_score_inheritance_changed",
        "terminal_score_value_used_to_choose_correction",
    ):
        check(f"scope_{field}", scope.get(field) is False)
    check("scope_terminal_metric_disclosed", scope.get("terminal_metric_observed_before_revision") is True)
    check("scope_source_changed", scope.get("implementation_source_changed") is True)
    check("scope_roots_changed", scope.get("formal_root_revision_changed") is True)

    correction = payload.get("implementation_correction") or {}
    check(
        "missing_obligations_exact",
        correction.get("failure_obligations")
        == ["payload:evaluator", "payload:fit_scope", "payload:split_lineage"],
    )
    check(
        "strategy_mapping_exact",
        correction.get("strategy_mapping") == EXPECTED_STRATEGY_MAPPING,
    )
    fit_rule = str(correction.get("fit_scope_non_equivalence") or "")
    check(
        "fit_semantics_separated",
        "train_view_only" in fit_rule
        and "never relabelled" in fit_rule
        and "ProtocolSpec" in fit_rule,
    )
    check(
        "dual_chain_declared",
        bool(correction.get("runtime_chain"))
        and bool(correction.get("terminal_chain"))
        and bool(correction.get("join_rule")),
    )
    check(
        "failure_sealing_declared",
        "NotFound" in str(correction.get("evaluator_failure_rule") or "")
        and "sealing" in str(correction.get("evaluator_failure_rule") or ""),
    )

    overrides = payload.get("overrides") or {}
    expected_roots = {
        "source_root": "formal-source-r9",
        "control_root": "formal-control-r9",
        "staging_root": "formal-staging-r11",
        "output_root": "formal-runs-r9",
        "gate_root": "formal-staging-r11-stop-gate-r1",
        "pipeline_root": "formal-staging-r11-pipeline-r1",
    }
    for field, suffix in expected_roots.items():
        value = str(overrides.get(field) or "")
        check(f"fresh_{field}", value.endswith(suffix) and "runs-r8" not in value)
    check("execution_revision", overrides.get("formal_execution_revision") == "r2")
    check("block_revision", overrides.get("block_id_suffix") == "r2")
    check("controller_revision", overrides.get("controller_pod") == "da-wp8-f-controller-cpu-r2")
    check("training_not_authorized", overrides.get("formal_training_authorized") is False)

    integrity = payload.get("analysis_integrity") or {}
    for field in (
        "r8_terminal_values_are_excluded",
        "r8_is_not_completed_post_hoc",
        "r8_is_not_called_pre_metric",
        "new_results_must_use_only_r9_roots",
        "no_condition_seed_or_task_may_be_excluded_after_terminal_metric",
    ):
        check(f"integrity_{field}", integrity.get(field) is True)
    check("effect_not_authorized", integrity.get("effect_claim_authorized_before_results") is False)
    check("gate_not_bypassed", integrity.get("formal_training_authorized_before_new_gate") is False)

    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "preregistration_id": payload.get("preregistration_id", ""),
        "amendment_file_sha256": _file_sha256(path),
        "failure_diagnostic_file_sha256": (
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
    report = verify_postfailure_amendment(
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


__all__ = ["SCHEMA", "VERIFICATION_SCHEMA", "verify_postfailure_amendment"]
