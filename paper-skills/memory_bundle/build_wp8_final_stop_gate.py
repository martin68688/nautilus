"""Build the final WP8 engineering Stop Gate without laundering effect claims."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from build_wp8_final_regression_receipt import (
    parse_junit,
    payload_hash,
    sha256_file,
)
from run_wp8_final_tests import verify_test_receipt as verify_host_test_receipt
from verify_wp8_final_regression_receipt import verify_receipt


STOP_GATE_SCHEMA = "decision_admissibility_wp8_final_stop_gate_v1"
MANIFEST_SCHEMA = "decision_admissibility_wp8_final_stop_gate_manifest_v1"

PLAN = "coordination/decision_admissibility_complete_execution_plan_20260719.md"
WP0_WP7_AUDIT = "coordination/decision_admissibility_wp0_wp7_completion_audit_20260720.md"
WP7_FINAL = "coordination/decision_admissibility_wp7_corrected_canary_report_20260721.md"
WP3_REPORT = "coordination/decision_admissibility_wp3_report_20260719.md"
WP4_REPORT = "coordination/decision_admissibility_wp4_report_20260719.md"
WP5_REPORT = "coordination/decision_admissibility_wp5_report_20260719.md"
WP5_WRITEBACK_ADDENDUM = (
    "coordination/decision_admissibility_wp5_result_writeback_addendum_20260720.md"
)
WP6_REPORT = "coordination/decision_admissibility_wp6_report_20260719.md"
GATE1_ROOT = "coordination/decision_admissibility_wp8_gate1_prevalence_20260721_r1"
TIER0_ROOT = "coordination/decision_admissibility_wp8_tier0_20260721"
TIER1_ROOT = "coordination/decision_admissibility_wp8_tier1_stop_gate_20260721_r2"
MULTIGEN_ROOT = "coordination/decision_admissibility_wp8_multigeneration_stop_gate_20260721_r1"
CANARY_ROOT = "coordination/decision_admissibility_wp8_tier2_canary_stop_gate_20260722_r10_r2"
PURITY_ROOT = "coordination/decision_admissibility_wp8_semantic_purity_20260721"
STAGING_GATE = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "staging_stop_gate_20260723_r10/STAGING_STOP_GATE.json"
)
CONTINUATION_GATE = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "continuation_stop_gate_20260723_r3/STAGING_STOP_GATE.json"
)
RECOVERY_GATE = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "recovery_stop_gate_20260723_r1/RECOVERY_STOP_GATE.json"
)
RECOVERY_DIAGNOSTIC = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "r10_birds_s104729_preterminal_finalizer_diagnostic_20260723.json"
)
JOINT_ROOT = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "joint_inventory_20260723_r1"
)
STATISTICS_ROOT = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "statistics_20260723_r1"
)
LEDGER_ROOT = "coordination/decision_admissibility_wp8_evidence_ledger_20260723_r2"
ANALYSIS_POLICY = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "analysis_policy_addendum_20260723_r1.json"
)
COMPLETED_FREEZE = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "completed_blocks_freeze_20260723_r1.json"
)
STATISTICS_FREEZE = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "statistics_implementation_freeze_20260723_r1.json"
)

PREREG_FILES = (
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r1.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r2.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r3.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r4.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r5.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r6.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r7.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r8.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r9.json",
)
PREREG_VERIFICATIONS = (
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260722_r1.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260723_r6.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260723_r7.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260723_r8.json",
    "coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260723_r9.json",
)

REQUIRED_FULL_SUITE_TESTS = (
    "tests.authority.test_stage_ontology::test_every_runtime_stage_has_one_explicit_dual_axis_mapping",
    "tests.authority.test_claim_decomposition::test_mixed_node_splits_fact_claims_with_stable_ids_and_bindings",
    "tests.authority.test_trusted_collectors::test_all_trusted_collectors_emit_host_receipts_with_hash_chain",
    "tests.authority.test_global_memory_authority_scope::test_matching_outcome_scope_policy_protocol_and_stages_are_required",
    "tests.authority.test_visibility_projection_bypass::test_explicit_allowed_authorized_edge_is_the_only_adoption_path",
    "tests.authority.test_result_adoption_causal_writeback::test_result_adoption_and_causal_objects_are_separately_materialized",
    "tests.test_fixed_holdout_terminal_writeback::test_terminal_scorer_writes_one_idempotent_result_fact",
    "tests.authority.test_legacy_promote_not_used::test_production_call_sites_never_invoke_legacy_promote",
    "tests.test_positive_result_vs_adopted_distillation::test_positive_result_distillation_uses_target_evidence_not_actuation",
    "tests.test_bundle_publication_crash_safety::test_crash_before_current_swap_leaves_old_pointer",
    "tests.authority.test_method_changing_fake_replay::test_unclassified_call_delta_requires_human_review",
    "tests.test_corpus_manifest::test_manifest_classifies_complete_partial_invalid_and_excluded_without_writes",
    "tests.test_corpus_split_isolation::test_full_seed_and_task_splits_are_deterministic_and_disjoint",
    "tests.test_memory_bundle_validation::test_bundle_is_immutable_hash_complete_and_split_isolated",
    "tests.test_multigeneration_contamination::test_invalid_ancestry_cannot_gain_publication_authority_by_paraphrase",
    "tests.test_decision_admissibility_factorial::test_every_factorial_case_emits_the_required_trace_contract",
    "tests.test_tier1_controlled_evaluator::test_host_code_execution_is_distinct_from_historical_actuation",
    "tests.test_certified_replay_semantic_purity::test_default_publication_uses_method_only_retrieval_projection",
    "tests.test_memory_bundle_validation::test_resigned_split_overlap_is_rejected_by_validator_and_snapshot_loader",
    "tests.test_bundle_publication_crash_safety::test_candidate_build_crash_quarantines_partial_staging_without_state_change",
    "tests.authority.test_mixed_value_sop_visibility::test_empty_visible_pack_reaches_agent_as_traced_abstention_without_legacy_fallback",
    "tests.test_decision_admissibility_factorial::test_independent_verifier_exactly_replays_and_rejects_resigned_tampering",
    "tests.authority.test_visibility_overhead_reporting::test_visibility_migration_reports_latency_tokens_and_empty_pack_without_gate",
    "tests.test_wp8_final_test_runner::test_host_receipt_rejects_self_rehashed_executed_command_laundering",
    "tests.test_wp8_final_test_runner::test_host_receipt_rejects_self_rehashed_skipped_scope",
)
STRICT_CLOSEOUT_TESTS = REQUIRED_FULL_SUITE_TESTS[-7:]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return bool(payload.get(field) == payload_hash(payload, field))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _binding(
    repo_root: Path,
    path: str | Path,
    *,
    internal_field: str = "",
) -> dict[str, Any]:
    unresolved = Path(path)
    if not unresolved.is_absolute():
        unresolved = repo_root / unresolved
    _require(not unresolved.is_symlink(), f"Symlink binding forbidden: {unresolved}")
    resolved = unresolved.resolve()
    _require(resolved.is_file() and not resolved.is_symlink(), f"Missing binding: {resolved}")
    result: dict[str, Any] = {
        "path": _display_path(resolved, repo_root),
        "file_sha256": sha256_file(resolved),
    }
    if internal_field:
        payload = _read_object(resolved)
        _require(_valid_hash(payload, internal_field), f"Invalid internal hash: {resolved}")
        result["internal_hash_field"] = internal_field
        result["internal_hash"] = payload[internal_field]
    return result


def _sealed_flat_root(root: Path, expected_names: set[str]) -> bool:
    if root.is_symlink() or not root.is_dir() or root.stat().st_mode & 0o222:
        return False
    entries = list(root.iterdir())
    if {path.name for path in entries} != expected_names:
        return False
    return all(
        not path.is_symlink()
        and path.is_file()
        and not (path.stat().st_mode & 0o222)
        for path in entries
    )


def _sidecar_matches(path: Path) -> bool:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        return False
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0] == sha256_file(path) and fields[-1] == path.name:
            return True
    return False


def _hash_list_matches(path: Path, hash_list: Path) -> bool:
    if not hash_list.is_file() or hash_list.is_symlink():
        return False
    return any(
        len(fields := line.strip().split()) >= 2
        and fields[0] == sha256_file(path)
        and fields[-1] == path.name
        for line in hash_list.read_text(encoding="utf-8").splitlines()
    )


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def gate_state(
    *,
    prerequisite_checks: Mapping[str, bool],
    kill_gates: Mapping[str, bool],
    acceptance_checks: Mapping[str, Mapping[str, bool]],
) -> dict[str, Any]:
    engineering_complete = bool(
        prerequisite_checks
        and kill_gates
        and acceptance_checks
        and all(value is True for value in prerequisite_checks.values())
        and all(value is True for value in kill_gates.values())
        and all(
            value is True
            for group in acceptance_checks.values()
            for value in group.values()
        )
    )
    return {
        "wp8_engineering_complete": engineering_complete,
        "wp8_stop_gate_passed": engineering_complete,
        "effect_claim_authorized": False,
        "next_authorized_phase": "Independent Claude audit" if engineering_complete else None,
        "independent_claude_audit_required": engineering_complete,
        "goal_completion_authorized": False,
    }


def formal_integrity_checks(
    *,
    inventory: Mapping[str, Any],
    statistics: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> dict[str, bool]:
    """Keep engineering closure independent from, but strict about, claims."""

    totals = inventory.get("totals") or {}
    population = statistics.get("analysis_population") or {}
    effect_gate = statistics.get("effect_claim_gate") or {}
    claims = {
        str(row.get("claim_id") or ""): row for row in ledger.get("claims") or []
    }
    result = claims.get("WP8-C2-RESULT-WRITEBACK") or {}
    no_imputation = claims.get("WP8-C5-NO-IMPUTATION") or {}
    causal = claims.get("WP8-C6-EXPERIENCE-CAUSALITY") or {}
    return {
        "exact_9_blocks_45_online_9_oracle": bool(
            totals.get("block_count") == 9
            and totals.get("online_condition_count") == 45
            and totals.get("oracle_disposition_count") == 9
            and population.get("assigned_online_outcomes") == 45
            and population.get("assigned_oracle_dispositions") == 9
        ),
        "exact_22_success_23_retained_failure_partition": bool(
            totals.get("successful_selected_result_count") == 22
            and totals.get("failed_online_condition_count") == 23
            and population.get("scored_selected_results") == 22
            and population.get("failed_online_conditions") == 23
        ),
        "result_fact_closure_has_zero_orphans": bool(
            totals.get("result_fact_count") == 22
            and (result.get("metrics") or {}).get("result_facts") == 22
            and (result.get("metrics") or {}).get("fixed_holdout_orphans") == 0
            and (result.get("metrics") or {}).get(
                "failed_conditions_with_result_fact"
            )
            == 0
        ),
        "no_imputation_and_no_post_assignment_exclusion": bool(
            population.get("imputed_scores") == 0
            and population.get("post_assignment_exclusions") == 0
            and (no_imputation.get("metrics") or {}).get("imputed") == 0
            and (no_imputation.get("metrics") or {}).get(
                "post_assignment_exclusions"
            )
            == 0
        ),
        "full_superiority_rejection_preserved": bool(
            effect_gate.get("effect_claim_authorized") is False
            and ledger.get("headline_effect_claim_authorized") is False
            and (claims.get("WP8-C3-FULL-SUPERIORITY") or {}).get("status")
            == "rejected"
        ),
        "conditional_utility_remains_diagnostic_only": bool(
            (claims.get("WP8-C4-CONDITIONAL-UTILITY") or {}).get("status")
            == "diagnostic"
            and (
                (claims.get("WP8-C4-CONDITIONAL-UTILITY") or {}).get(
                    "claim_gate"
                )
                or {}
            ).get("superiority_authorized")
            is False
        ),
        "experience_causality_remains_pending_without_l4": bool(
            causal.get("status") == "pending"
            and (causal.get("claim_gate") or {}).get("satisfied") is False
            and (causal.get("metrics") or {}).get(
                "required_minimum_actuation_level"
            )
            == "L4"
        ),
    }


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# WP8 Final Stop Gate",
        "",
        f"- Engineering complete: `{str(report['wp8_engineering_complete']).lower()}`",
        f"- Stop Gate passed: `{str(report['wp8_stop_gate_passed']).lower()}`",
        f"- Effect claim authorized: `{str(report['effect_claim_authorized']).lower()}`",
        f"- Next phase: `{report['next_authorized_phase']}`",
        f"- Goal completion authorized: `{str(report['goal_completion_authorized']).lower()}`",
        "",
        "## Kill gates",
        "",
    ]
    for name, value in sorted(report["kill_gates"].items()):
        lines.append(f"- [{'x' if value else ' '}] `{name}`")
    lines.extend(["", "## Acceptance", ""])
    for group, checks in sorted(report["acceptance_checks"].items()):
        lines.append(f"### {group}")
        lines.append("")
        for name, value in sorted(checks.items()):
            lines.append(f"- [{'x' if value else ' '}] `{name}`")
        lines.append("")
    lines.extend(
        [
            "## Claim boundary",
            "",
            "WP8 engineering completion is independent of the formal Full-vs-No-Memory effect claim.",
            "The formal effect gate is false; conditional positive pairs remain diagnostic only.",
            "Post-result safety fixes are covered by regression tests but were not used to rewrite or rerun formal outcomes.",
            "Independent Claude review remains mandatory before the overall goal may be closed.",
            "",
            f"Report hash: `{report['report_hash']}`",
            "",
        ]
    )
    return "\n".join(lines)


def compute_stop_gate(
    *,
    repo_root: str | Path,
    final_regression_receipt_path: str | Path,
    final_test_root: str | Path,
    created_at: str,
    host_test_receipt_root: str | Path | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    final_regression_receipt_path = Path(final_regression_receipt_path).resolve()
    final_test_root = Path(final_test_root).resolve()
    host_test_receipt_root = Path(
        host_test_receipt_root or final_test_root
    ).resolve()

    plan_binding = _binding(repo_root, PLAN)
    wp0_audit_path = repo_root / WP0_WP7_AUDIT
    wp7_path = repo_root / WP7_FINAL
    _require(_sidecar_matches(wp0_audit_path), "WP0-WP7 audit sidecar mismatch")
    _require(_sidecar_matches(wp7_path), "WP7 corrected report sidecar mismatch")
    wp0_audit = wp0_audit_path.read_text(encoding="utf-8")
    wp7_text = wp7_path.read_text(encoding="utf-8")
    wp3_text = (repo_root / WP3_REPORT).read_text(encoding="utf-8")
    wp4_text = (repo_root / WP4_REPORT).read_text(encoding="utf-8")
    wp5_writeback_text = (repo_root / WP5_WRITEBACK_ADDENDUM).read_text(
        encoding="utf-8"
    )
    wp6_text = (repo_root / WP6_REPORT).read_text(encoding="utf-8")

    gate1_report = _read_object(repo_root / GATE1_ROOT / "prevalence_report.json")
    gate1_verification = _read_object(repo_root / GATE1_ROOT / "verification.json")
    _require(_valid_hash(gate1_report, "report_hash"), "Gate-1 report hash")
    _require(_valid_hash(gate1_verification, "verification_hash"), "Gate-1 verification hash")
    gate1_ok = bool(
        gate1_verification.get("verified") is True
        and gate1_verification.get("errors") == []
        and gate1_verification.get("gate_1_passed") is True
        and gate1_report.get("gate_1", {}).get("passed") is True
    )

    tier0_report = _read_object(repo_root / TIER0_ROOT / "tier0_factorial_report_v6.json")
    tier0_verification = _read_object(repo_root / TIER0_ROOT / "tier0_factorial_verification_v6.json")
    _require(_valid_hash(tier0_report, "report_hash"), "Tier-0 report hash")
    _require(_valid_hash(tier0_verification, "verification_hash"), "Tier-0 verification hash")
    tier0_ok = bool(
        tier0_verification.get("valid") is True
        and tier0_verification.get("errors") == []
        and tier0_report.get("all_cases_passed") is True
        and tier0_report.get("case_count") == tier0_report.get("expected_case_count") == 63
    )

    tier1_report = _read_object(repo_root / TIER1_ROOT / "stop_gate_report.json")
    tier1_verification = _read_object(repo_root / TIER1_ROOT / "verification.json")
    _require(_valid_hash(tier1_report, "report_hash"), "Tier-1 report hash")
    _require(_valid_hash(tier1_verification, "verification_hash"), "Tier-1 verification hash")
    tier1_ok = bool(
        tier1_report.get("passed") is True
        and all((tier1_report.get("stop_gate_checks") or {}).values())
        and tier1_verification.get("verified") is True
        and tier1_verification.get("errors") == []
    )

    multigen_report = _read_object(repo_root / MULTIGEN_ROOT / "stop_gate_report.json")
    multigen_verification = _read_object(repo_root / MULTIGEN_ROOT / "verification.json")
    _require(_valid_hash(multigen_report, "report_hash"), "Multi-generation report hash")
    _require(_valid_hash(multigen_verification, "verification_hash"), "Multi-generation verification hash")
    multigen_ok = bool(
        multigen_report.get("passed") is True
        and all((multigen_report.get("stop_gate_checks") or {}).values())
        and multigen_verification.get("verified") is True
        and multigen_verification.get("errors") == []
        and multigen_verification.get("gate_5_passed") is True
    )

    canary_report = _read_object(repo_root / CANARY_ROOT / "stop_gate_report.json")
    canary_verification = _read_object(repo_root / CANARY_ROOT / "verification.json")
    _require(_valid_hash(canary_report, "report_hash"), "Tier-2 canary report hash")
    _require(_valid_hash(canary_verification, "verification_hash"), "Tier-2 canary verification hash")
    canary_ok = bool(
        canary_report.get("passed") is True
        and all((canary_report.get("stop_gate_checks") or {}).values())
        and canary_verification.get("verified") is True
        and canary_verification.get("errors") == []
    )

    purity = _read_object(repo_root / PURITY_ROOT / "method_semantic_purity_report.json")
    purity_formal = _read_object(repo_root / PURITY_ROOT / "formal_validation_report.json")
    purity_independent = _read_object(repo_root / PURITY_ROOT / "independent_validation_report.json")
    _require(_valid_hash(purity, "report_hash"), "Semantic-purity report hash")
    purity_ok = bool(
        purity.get("passed") is True
        and purity.get("source_outcome_assertion_count") == 0
        and purity.get("raw_text_embedded") is False
        and purity_formal.get("valid") is True
        and purity_formal.get("errors") == []
        and purity_independent.get("valid") is True
        and purity_independent.get("errors") == []
    )

    prereg_bindings = []
    for index, relative in enumerate(PREREG_FILES, 1):
        path = repo_root / relative
        _require(_sidecar_matches(path), f"Preregistration r{index} sidecar")
        payload = _read_object(path)
        if index > 1:
            _require(_valid_hash(payload, "amendment_hash"), f"Preregistration r{index} hash")
        prereg_bindings.append(_binding(repo_root, relative))
    prereg_verification_bindings = []
    for relative in PREREG_VERIFICATIONS:
        path = repo_root / relative
        hash_list = (
            repo_root / PREREG_FILES[0]
        ).with_suffix(".sha256") if relative.endswith("20260722_r1.json") else path.with_suffix(".sha256")
        _require(
            _hash_list_matches(path, hash_list),
            f"Preregistration verification sidecar: {path}",
        )
        payload = _read_object(path)
        _require(_valid_hash(payload, "verification_hash"), f"Prereg verification hash: {path}")
        _require(payload.get("verified") is True and payload.get("errors") == [], f"Prereg verification failed: {path}")
        prereg_verification_bindings.append(_binding(repo_root, relative, internal_field="verification_hash"))
    prereg_ok = len(prereg_bindings) == 9 and len(prereg_verification_bindings) == 5

    staging = _read_object(repo_root / STAGING_GATE)
    continuation = _read_object(repo_root / CONTINUATION_GATE)
    recovery = _read_object(repo_root / RECOVERY_GATE)
    for name, payload in (
        ("staging", staging),
        ("continuation", continuation),
        ("recovery", recovery),
    ):
        _require(_valid_hash(payload, "gate_hash"), f"{name} gate hash")
        _require(payload.get("status") == "passed", f"{name} gate status")
        _require(payload.get("errors") == [], f"{name} gate errors")
        _require(all((payload.get("checks") or {}).values()), f"{name} gate checks")
    staging_ok = bool(
        staging.get("formal_training_authorized") is True
        and continuation.get("formal_training_authorized") is True
        and continuation.get("effect_claim_authorized") is False
        and continuation.get("completed_blocks_authorized_to_rerun") is False
        and continuation.get("failed_r4_attempt_authorized_to_reuse") is False
        and recovery.get("terminal_score_values_inspected") is False
    )

    inventory = _read_object(repo_root / JOINT_ROOT / "joint_inventory.json")
    inventory_verification = _read_object(repo_root / JOINT_ROOT / "verification.json")
    _require(_valid_hash(inventory, "report_hash"), "Joint inventory hash")
    _require(_valid_hash(inventory_verification, "verification_hash"), "Joint inventory verification hash")
    inventory_ok = bool(
        inventory_verification.get("verified") is True
        and inventory_verification.get("errors") == []
        and inventory.get("score_policy") == "hash_only"
        and inventory.get("score_values_included") is False
        and inventory.get("score_values_inspected") is False
    )
    input_bindings = inventory.get("input_bindings") or {}
    expected_gate_files = {
        "completed_staging_gate": repo_root / STAGING_GATE,
        "continuation_staging_gate": repo_root / CONTINUATION_GATE,
        "recovery_gate": repo_root / RECOVERY_GATE,
        "recovery_diagnostic": repo_root / RECOVERY_DIAGNOSTIC,
    }
    for key, path in expected_gate_files.items():
        _require(
            (input_bindings.get(key) or {}).get("file_sha256") == sha256_file(path),
            f"Joint inventory gate binding mismatch: {key}",
        )

    statistics = _read_object(repo_root / STATISTICS_ROOT / "statistics_report.json")
    statistics_verification = _read_object(repo_root / STATISTICS_ROOT / "verification.json")
    _require(_valid_hash(statistics, "report_hash"), "Statistics report hash")
    _require(_valid_hash(statistics_verification, "verification_hash"), "Statistics verification hash")
    population = statistics.get("analysis_population") or {}
    effect_gate = statistics.get("effect_claim_gate") or {}
    statistics_ok = bool(
        statistics_verification.get("verified") is True
        and statistics_verification.get("errors") == []
        and statistics_verification.get("effect_claim_authorized") is False
        and effect_gate.get("effect_claim_authorized") is False
        and population
        == {
            "assigned_online_outcomes": 45,
            "scored_selected_results": 22,
            "failed_online_conditions": 23,
            "assigned_oracle_dispositions": 9,
            "imputed_scores": 0,
            "post_assignment_exclusions": 0,
        }
    )

    ledger = _read_object(repo_root / LEDGER_ROOT / "evidence_ledger.json")
    ledger_verification = _read_object(repo_root / LEDGER_ROOT / "verification.json")
    _require(_valid_hash(ledger, "ledger_hash"), "Evidence Ledger hash")
    _require(_valid_hash(ledger_verification, "verification_hash"), "Evidence Ledger verification hash")
    ledger_ok = bool(
        ledger_verification.get("verified") is True
        and ledger_verification.get("errors") == []
        and ledger_verification.get("headline_effect_claim_authorized") is False
        and ledger.get("headline_effect_claim_authorized") is False
    )
    claim_status = {row.get("claim_id"): row.get("status") for row in ledger.get("claims") or []}
    _require(claim_status.get("WP8-C3-FULL-SUPERIORITY") == "rejected", "Ledger superiority status")
    _require(claim_status.get("WP8-C4-CONDITIONAL-UTILITY") == "diagnostic", "Ledger conditional status")
    _require(claim_status.get("WP8-C6-EXPERIENCE-CAUSALITY") == "pending", "Ledger causal status")
    formal_checks = formal_integrity_checks(
        inventory=inventory,
        statistics=statistics,
        ledger=ledger,
    )

    regression_verification = verify_receipt(
        receipt_path=final_regression_receipt_path,
        repo_root=repo_root,
        test_root=final_test_root,
    )
    regression_receipt = _read_object(final_regression_receipt_path)
    regression_package_ok = _sealed_flat_root(
        final_regression_receipt_path.parent,
        {
            final_regression_receipt_path.name,
            final_regression_receipt_path.with_suffix(".sha256").name,
        },
    )
    regression_ok = bool(
        regression_verification.get("verified") is True
        and regression_verification.get("errors") == []
        and regression_receipt.get("final_regression_passed") is True
        and regression_receipt.get("unexpected_failure_count") == 0
        and regression_package_ok
    )
    host_test_verification = verify_host_test_receipt(
        receipt_root=host_test_receipt_root,
        repo_root=repo_root,
    )
    host_test_receipt_ok = bool(
        host_test_verification.get("verified") is True
        and host_test_verification.get("errors") == []
        and host_test_verification.get("full_suite_test_count", 0) >= 735
        and len(
            str(
                host_test_verification.get(
                    "full_suite_testcase_ids_sha256", ""
                )
            )
        )
        == 64
    )
    host_test_receipt = _read_object(
        host_test_receipt_root / "test_receipt.json"
    )
    host_full_suite_name = str(
        host_test_receipt.get("full_suite_run_name") or ""
    )
    _require(host_full_suite_name == "full_suite_r2", "Host full-suite identity")
    host_run_by_name = {
        str(row.get("name") or ""): row
        for row in host_test_receipt.get("test_runs") or []
    }
    host_full_suite_run = host_run_by_name.get(host_full_suite_name) or {}
    host_full_suite_relative = str(host_full_suite_run.get("junit_path") or "")
    _require(
        Path(host_full_suite_relative).name == host_full_suite_relative
        and host_full_suite_relative == "full_suite_r2.xml",
        "Host full-suite JUnit path",
    )
    host_full_suite_path = host_test_receipt_root / host_full_suite_relative
    final_suite = parse_junit(host_full_suite_path, repo_root=repo_root)
    final_ids = set(final_suite.pop("_testcase_ids"))
    required_tests_present = all(
        test_id in final_ids for test_id in REQUIRED_FULL_SUITE_TESTS
    )
    strict_closeout_tests_present = all(
        test_id in final_ids for test_id in STRICT_CLOSEOUT_TESTS
    )

    tier1_gates = tier1_report.get("kill_gates") or {}
    multigen_gates = multigen_report.get("kill_gates") or {}
    kill_gates = {
        "gate_1_problem_prevalence": gate1_ok and tier1_gates.get("gate_1", {}).get("status") == "pass",
        "gate_2_claim_level_vs_global_bit": tier1_gates.get("gate_2", {}).get("passed") is True,
        "gate_3_stage_utility": tier1_gates.get("gate_3", {}).get("passed") is True,
        "gate_4_visibility_necessity": tier1_gates.get("gate_4", {}).get("passed") is True,
        "gate_5_multigeneration": multigen_gates.get("gate_5", {}).get("passed") is True,
        "gate_6_writeback_separation": tier1_gates.get("gate_6", {}).get("passed") is True,
    }

    code_checks = {
        "baseline_and_new_tests_pass": regression_ok
        and host_test_receipt_ok
        and final_suite["failures"] == final_suite["errors"] == final_suite["skipped"] == 0,
        "stage_ontology_unique": REQUIRED_FULL_SUITE_TESTS[0] in final_ids,
        "mixed_claims_split": REQUIRED_FULL_SUITE_TESTS[1] in final_ids,
        "trusted_collectors_host_owned": REQUIRED_FULL_SUITE_TESTS[2] in final_ids,
        "global_memory_scope_checked": REQUIRED_FULL_SUITE_TESTS[3] in final_ids,
        "pre_prompt_visibility_and_bypass_closed": REQUIRED_FULL_SUITE_TESTS[4] in final_ids,
        "result_adoption_causal_paths_separate": REQUIRED_FULL_SUITE_TESTS[5] in final_ids,
        "terminal_writeback_exactly_once": REQUIRED_FULL_SUITE_TESTS[6] in final_ids,
        "legacy_promote_not_used": REQUIRED_FULL_SUITE_TESTS[7] in final_ids,
        "positive_result_and_adopted_sop_separate": REQUIRED_FULL_SUITE_TESTS[8] in final_ids,
        "atomic_publication_and_crash_safety": REQUIRED_FULL_SUITE_TESTS[9] in final_ids,
        "clean_replay_and_successor_separate": REQUIRED_FULL_SUITE_TESTS[10] in final_ids,
        "required_representative_tests_present": required_tests_present,
        "strict_section_20_4_20_5_closeout_present": strict_closeout_tests_present,
        "host_full_suite_current_and_above_floor": (
            host_test_receipt_ok
            and final_suite["tests"] >= 735
            and final_suite["failures"] == 0
            and final_suite["errors"] == 0
            and final_suite["skipped"] == 0
        ),
    }
    corpus_checks = {
        "spooky_zero": "| Spooky formal/source runs | 0 |" in wp4_text,
        "complete_partial_excluded_reasoned": "All 11 partial directories lack core artifacts" in wp4_text,
        "core_artifacts_hashed": "Every code node has a deterministic audit sidecar" in wp4_text,
        "clause_sources_resolve": "Every included clause source resolves" in wp4_text,
        "full_seed_task_bundles_separate": "Full: 79 source runs" in wp4_text and "Seed-heldout" in wp4_text and "Task-heldout" in wp4_text,
        "source_test_zero_overlap": REQUIRED_FULL_SUITE_TESTS[12] in final_ids,
        "old_assets_not_overwritten": "old RunForest still contains exactly 281 SOP containers" in wp4_text,
    }
    safety_checks = {
        "unauthorized_prompt_exposure_zero": tier0_report.get("unauthorized_prompt_exposure_count") == 0,
        "unauthorized_activation_zero": tier0_report.get("invalid_activation_count") == 0,
        "debug_repair_retention_preserved": tier0_report.get("valid_knowledge_retention") == 1.0,
        "historical_score_not_ranked": purity_ok and "0.92" in (repo_root / PLAN).read_text(encoding="utf-8"),
        "unadopted_result_retained_without_edges": "clean unexposed node independently produced a Result Fact" in wp7_text,
        "receipt_types_not_interchangeable": REQUIRED_FULL_SUITE_TESTS[16] in final_ids,
        "descendant_non_escalation": REQUIRED_FULL_SUITE_TESTS[14] in final_ids,
        "shadow_disagreements_reviewed": "Shadow disagreement population independently reviewed before enforce" in wp7_text,
        "visibility_overhead_reported_without_threshold": "not a post-hoc pilot threshold" in wp3_text,
    }
    paper_checks = {
        "fresh_heldout_factorial_episodes": tier1_ok,
        "seed_and_task_heldout": statistics_ok and (inventory.get("totals") or {}).get("block_count") == 9,
        "three_protocol_families_kernel_fixed": len(
            {row.get("task_id") for row in inventory.get("blocks") or []}
        )
        == 3,
        "strong_baselines_and_ablations": tier1_ok and multigen_ok,
        "iir_vkr_pareto_evidence": tier1_gates.get("gate_2", {}).get("passed") is True,
        "l2_l3_l4_adoption_evidence": tier1_gates.get("gate_6", {}).get("passed") is True,
        "multi_task_multi_seed_paired_statistics": population.get("assigned_online_outcomes") == 45,
        "headline_claims_frozen_and_hash_bound": ledger_ok and statistics_ok,
        "effect_claim_rejected_not_laundered": effect_gate.get("effect_claim_authorized") is False,
    }
    acceptance_checks = {
        "code_correctness": code_checks,
        "corpus_and_bundle": corpus_checks,
        "safety_and_utility": safety_checks,
        "paper_evidence": paper_checks,
        "formal_integrity": formal_checks,
    }
    prerequisite_checks = {
        "wp0_wp6_prior_audit_bound": "WP0–WP6 engineering Stop Gates remain **PASSED**" in wp0_audit,
        "wp7_corrected_stop_gate_passed": "**WP7 Stop Gate: PASSED" in wp7_text,
        "gate1_verified": gate1_ok,
        "tier0_verified": tier0_ok,
        "tier1_verified": tier1_ok,
        "multigeneration_verified": multigen_ok,
        "tier2_canary_verified": canary_ok,
        "semantic_purity_verified": purity_ok,
        "preregistration_r1_r9_bound": prereg_ok,
        "staging_and_recovery_gates_bound": staging_ok,
        "joint_inventory_verified": inventory_ok,
        "formal_statistics_verified": statistics_ok,
        "evidence_ledger_verified": ledger_ok,
        "final_regression_verified": regression_ok,
        "final_regression_package_immutable": regression_package_ok,
        "host_test_commands_exit_codes_junit_and_source_verified": host_test_receipt_ok,
        "formal_effect_gate_false": effect_gate.get("effect_claim_authorized") is False,
        "formal_population_result_and_claim_boundaries_exact": all(
            formal_checks.values()
        ),
        "wp5_result_actuation_boundary_documented": (
            "Result Fact" in wp5_writeback_text
            and "Adoption Edge" in wp5_writeback_text
            and "Causal Edge" in wp5_writeback_text
        ),
        "wp6_clean_replay_boundary_documented": "Clean Replay" in wp6_text,
    }
    state = gate_state(
        prerequisite_checks=prerequisite_checks,
        kill_gates=kill_gates,
        acceptance_checks=acceptance_checks,
    )

    artifact_bindings = {
        "plan": plan_binding,
        "wp0_wp7_audit": _binding(repo_root, WP0_WP7_AUDIT),
        "wp7_corrected_report": _binding(repo_root, WP7_FINAL),
        "wp3_report": _binding(repo_root, WP3_REPORT),
        "wp4_report": _binding(repo_root, WP4_REPORT),
        "wp5_report": _binding(repo_root, WP5_REPORT),
        "wp5_writeback_addendum": _binding(repo_root, WP5_WRITEBACK_ADDENDUM),
        "wp6_report": _binding(repo_root, WP6_REPORT),
        "gate1_verification": _binding(repo_root, Path(GATE1_ROOT) / "verification.json", internal_field="verification_hash"),
        "tier0_verification": _binding(repo_root, Path(TIER0_ROOT) / "tier0_factorial_verification_v6.json", internal_field="verification_hash"),
        "tier1_verification": _binding(repo_root, Path(TIER1_ROOT) / "verification.json", internal_field="verification_hash"),
        "multigeneration_verification": _binding(repo_root, Path(MULTIGEN_ROOT) / "verification.json", internal_field="verification_hash"),
        "tier2_canary_verification": _binding(repo_root, Path(CANARY_ROOT) / "verification.json", internal_field="verification_hash"),
        "semantic_purity": _binding(repo_root, Path(PURITY_ROOT) / "method_semantic_purity_report.json", internal_field="report_hash"),
        "analysis_policy": _binding(repo_root, ANALYSIS_POLICY, internal_field="analysis_policy_hash"),
        "completed_blocks_freeze": _binding(repo_root, COMPLETED_FREEZE, internal_field="inventory_hash"),
        "statistics_implementation_freeze": _binding(repo_root, STATISTICS_FREEZE, internal_field="implementation_hash"),
        "staging_gate": _binding(repo_root, STAGING_GATE, internal_field="gate_hash"),
        "continuation_gate": _binding(repo_root, CONTINUATION_GATE, internal_field="gate_hash"),
        "recovery_gate": _binding(repo_root, RECOVERY_GATE, internal_field="gate_hash"),
        "joint_inventory": _binding(repo_root, Path(JOINT_ROOT) / "joint_inventory.json", internal_field="report_hash"),
        "joint_inventory_verification": _binding(repo_root, Path(JOINT_ROOT) / "verification.json", internal_field="verification_hash"),
        "statistics": _binding(repo_root, Path(STATISTICS_ROOT) / "statistics_report.json", internal_field="report_hash"),
        "statistics_verification": _binding(repo_root, Path(STATISTICS_ROOT) / "verification.json", internal_field="verification_hash"),
        "evidence_ledger": _binding(repo_root, Path(LEDGER_ROOT) / "evidence_ledger.json", internal_field="ledger_hash"),
        "evidence_ledger_verification": _binding(repo_root, Path(LEDGER_ROOT) / "verification.json", internal_field="verification_hash"),
        "final_regression_receipt": _binding(repo_root, final_regression_receipt_path, internal_field="receipt_hash"),
        "host_test_receipt": _binding(
            repo_root,
            host_test_receipt_root / "test_receipt.json",
            internal_field="receipt_hash",
        ),
        "host_test_manifest": _binding(
            repo_root,
            host_test_receipt_root / "manifest.json",
            internal_field="manifest_hash",
        ),
    }
    report: dict[str, Any] = {
        "schema": STOP_GATE_SCHEMA,
        "status": (
            "engineering_complete_effect_claim_not_authorized"
            if state["wp8_engineering_complete"]
            else "engineering_incomplete"
        ),
        "created_at": str(created_at),
        **state,
        "prerequisite_checks": prerequisite_checks,
        "kill_gates": kill_gates,
        "acceptance_checks": acceptance_checks,
        "formal_population": population,
        "formal_effect_claim_gate": effect_gate,
        "claim_status": claim_status,
        "formal_integrity_checks": formal_checks,
        "post_result_change_boundary": {
            "formal_outcomes_bind_frozen_execution_sources": True,
            "final_source_inventory_hash": (regression_receipt.get("source_inventory") or {}).get("inventory_hash"),
            "host_test_source_inventory_hash": host_test_verification.get(
                "source_inventory_sha256"
            ),
            "post_result_safety_fixes_not_used_to_rewrite_outcomes": True,
            "post_result_safety_fixes_not_evaluated_for_formal_effect": True,
            "rerun_or_seed_selection_performed": False,
        },
        "artifact_bindings": artifact_bindings,
        "preregistration_bindings": prereg_bindings,
        "preregistration_verification_bindings": prereg_verification_bindings,
        "final_regression_verification": regression_verification,
        "host_test_receipt_verification": host_test_verification,
        "host_test_full_suite": {
            "path": _display_path(host_full_suite_path, repo_root),
            "file_sha256": sha256_file(host_full_suite_path),
            "tests": final_suite["tests"],
            "failures": final_suite["failures"],
            "errors": final_suite["errors"],
            "skipped": final_suite["skipped"],
            "testcase_ids_hash": final_suite["testcase_ids_hash"],
            "required_test_count": len(REQUIRED_FULL_SUITE_TESTS),
            "all_required_tests_present": required_tests_present,
        },
        "independent_review_scope": [
            "preregistration adherence",
            "failure retention and no imputation",
            "statistics and multiplicity",
            "same-domain leakage boundary",
            "claim authorization",
            "Evidence Ledger",
            "final Stop Gate",
            "post-result source-version boundary",
        ],
        "claim_boundaries": [
            "Formal Full superiority over No Memory is not supported.",
            "Positive deltas among the four successful Full/No-Memory pairs are diagnostic only.",
            "Experience causality remains pending because Adoption/Causal evidence was not established for the formal gains.",
            "Engineering completion does not authorize an effect headline.",
            "The independent Claude audit must pass before the overall goal can close.",
        ],
        "builder_source_sha256": sha256_file(Path(__file__).resolve()),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve().with_name("verify_wp8_final_stop_gate.py")),
        "report_hash": "",
    }
    report["report_hash"] = payload_hash(report, "report_hash")
    return report


def build_stop_gate(*, output_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    raw_output_root = Path(output_root)
    if raw_output_root.is_symlink():
        raise FileExistsError(
            f"Refusing symlink WP8 final Stop-Gate root: {raw_output_root}"
        )
    output_root = raw_output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse WP8 final Stop-Gate root: {output_root}")
    report = compute_stop_gate(**kwargs)
    if not report["wp8_stop_gate_passed"]:
        raise ValueError("WP8 final engineering Stop Gate did not pass")
    markdown = _render_markdown(report)
    output_root.mkdir(parents=True, exist_ok=False)
    try:
        _write_text_exclusive(
            output_root / "stop_gate_report.json",
            json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        )
        _write_text_exclusive(output_root / "stop_gate_report.md", markdown)
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "status": "complete",
            "files": {
                "stop_gate_report.json": sha256_file(output_root / "stop_gate_report.json"),
                "stop_gate_report.md": sha256_file(output_root / "stop_gate_report.md"),
            },
            "report_hash": report["report_hash"],
            "builder_source_sha256": report["builder_source_sha256"],
            "verifier_source_sha256": report["verifier_source_sha256"],
            "manifest_hash": "",
        }
        manifest["manifest_hash"] = payload_hash(manifest, "manifest_hash")
        _write_text_exclusive(
            output_root / "manifest.json",
            json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        )
        directory_descriptor = os.open(output_root, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.chmod(output_root, 0o555)
    except Exception:
        # An incomplete exclusive root has no complete manifest and is never reusable.
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--final-regression-receipt", required=True, type=Path)
    parser.add_argument("--final-test-root", required=True, type=Path)
    parser.add_argument("--host-test-receipt-root", type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    report = build_stop_gate(
        output_root=args.output_root,
        repo_root=args.repo_root,
        final_regression_receipt_path=args.final_regression_receipt,
        final_test_root=args.final_test_root,
        host_test_receipt_root=args.host_test_receipt_root,
        created_at=args.created_at,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
