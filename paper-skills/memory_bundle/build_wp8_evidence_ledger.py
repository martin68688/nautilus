"""Build the hash-bound WP8 experiment Evidence Ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


LEDGER_SCHEMA = "decision_admissibility_wp8_evidence_ledger_v1"
MANIFEST_SCHEMA = "decision_admissibility_wp8_evidence_ledger_manifest_v1"

PRIOR_VERIFICATIONS = (
    "coordination/decision_admissibility_wp8_gate1_prevalence_20260721_r1/verification.json",
    "coordination/decision_admissibility_wp8_tier1_stop_gate_20260721_r2/verification.json",
    "coordination/decision_admissibility_wp8_multigeneration_stop_gate_20260721_r1/verification.json",
    "coordination/decision_admissibility_wp8_tier2_canary_stop_gate_20260722_r10_r2/verification.json",
)
PRIOR_REPORTS = (
    "coordination/decision_admissibility_wp8_tier0_report_20260721.md",
    "coordination/decision_admissibility_wp8_semantic_purity_report_20260721.md",
    "coordination/decision_admissibility_wp7_corrected_canary_report_20260721.md",
)
FAILURE_EVIDENCE = (
    "coordination/decision_admissibility_wp8_tier2_formal_r8_authority_failure_diagnostic_20260723.json",
    "coordination/decision_admissibility_wp8_tier2_formal_continuation_staging_failure_20260723_r1.json",
    "coordination/decision_admissibility_wp8_tier2_formal_birds_seed130363_r4_precontract_diagnostic_20260723.json",
    "coordination/decision_admissibility_wp8_tier2_formal_r10_birds_s104729_preterminal_finalizer_diagnostic_20260723.json",
)
PAPER_CLAIMS = "papers/runforest_iclr2025/evidence/claims.md"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        _canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) == _payload_hash(payload, field)


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _binding(repo_root: Path, relative: str, *, internal_field: str = "") -> dict[str, Any]:
    path = (repo_root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result: dict[str, Any] = {
        "path": relative,
        "file_sha256": _sha256_file(path),
    }
    if internal_field:
        payload = _read_object(path)
        if not _valid_hash(payload, internal_field):
            raise ValueError(f"Internal hash mismatch: {relative}:{internal_field}")
        result["internal_hash_field"] = internal_field
        result["internal_hash"] = payload[internal_field]
    return result


def _claim(
    *,
    claim_id: str,
    group: str,
    statement: str,
    status: str,
    condition: str,
    sample_unit: str,
    metrics: Mapping[str, Any],
    artifact_bindings: Sequence[Mapping[str, Any]],
    claim_gate: Mapping[str, Any],
    interpretation: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "claim_id": claim_id,
        "group": group,
        "statement": statement,
        "status": status,
        "condition": condition,
        "sample_unit": sample_unit,
        "metrics": dict(metrics),
        "artifact_bindings": [dict(value) for value in artifact_bindings],
        "claim_gate": dict(claim_gate),
        "interpretation": interpretation,
        "claim_hash": "",
    }
    row["claim_hash"] = _payload_hash(row, "claim_hash")
    return row


def compute_evidence_ledger(
    *,
    repo_root: str | Path,
    analysis_policy_path: str | Path,
    joint_inventory_root: str | Path,
    statistics_root: str | Path,
    created_at: str,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    analysis_policy_path = Path(analysis_policy_path).resolve()
    joint_inventory_root = Path(joint_inventory_root).resolve()
    statistics_root = Path(statistics_root).resolve()
    policy = _read_object(analysis_policy_path)
    inventory = _read_object(joint_inventory_root / "joint_inventory.json")
    inventory_verification = _read_object(joint_inventory_root / "verification.json")
    statistics = _read_object(statistics_root / "statistics_report.json")
    statistics_verification = _read_object(statistics_root / "verification.json")
    if not _valid_hash(policy, "analysis_policy_hash"):
        raise ValueError("Analysis policy hash mismatch")
    if not _valid_hash(inventory, "report_hash"):
        raise ValueError("Joint inventory hash mismatch")
    if inventory_verification.get("verified") is not True or not _valid_hash(
        inventory_verification, "verification_hash"
    ):
        raise ValueError("Joint inventory verification is not passed")
    if not _valid_hash(statistics, "report_hash"):
        raise ValueError("Statistics report hash mismatch")
    if statistics_verification.get("verified") is not True or not _valid_hash(
        statistics_verification, "verification_hash"
    ):
        raise ValueError("Statistics verification is not passed")
    if statistics_verification.get("statistics_report_hash") != statistics.get(
        "report_hash"
    ):
        raise ValueError("Statistics verification binding mismatch")
    population = statistics["analysis_population"]
    if population != {
        "assigned_online_outcomes": 45,
        "scored_selected_results": 22,
        "failed_online_conditions": 23,
        "assigned_oracle_dispositions": 9,
        "imputed_scores": 0,
        "post_assignment_exclusions": 0,
    }:
        raise ValueError("Formal population drift")
    primary = statistics["contrasts"]["full_minus_no_memory"]
    gate = statistics["effect_claim_gate"]
    if gate.get("effect_claim_authorized") is not False:
        raise ValueError("Unexpected Full-superiority authorization")

    core_bindings = {
        "analysis_policy": {
            "path": str(analysis_policy_path.relative_to(repo_root)),
            "file_sha256": _sha256_file(analysis_policy_path),
            "internal_hash_field": "analysis_policy_hash",
            "internal_hash": policy["analysis_policy_hash"],
        },
        "joint_inventory": {
            "path": str((joint_inventory_root / "joint_inventory.json").relative_to(repo_root)),
            "file_sha256": _sha256_file(joint_inventory_root / "joint_inventory.json"),
            "internal_hash_field": "report_hash",
            "internal_hash": inventory["report_hash"],
        },
        "joint_inventory_verification": {
            "path": str((joint_inventory_root / "verification.json").relative_to(repo_root)),
            "file_sha256": _sha256_file(joint_inventory_root / "verification.json"),
            "internal_hash_field": "verification_hash",
            "internal_hash": inventory_verification["verification_hash"],
        },
        "statistics": {
            "path": str((statistics_root / "statistics_report.json").relative_to(repo_root)),
            "file_sha256": _sha256_file(statistics_root / "statistics_report.json"),
            "internal_hash_field": "report_hash",
            "internal_hash": statistics["report_hash"],
        },
        "statistics_verification": {
            "path": str((statistics_root / "verification.json").relative_to(repo_root)),
            "file_sha256": _sha256_file(statistics_root / "verification.json"),
            "internal_hash_field": "verification_hash",
            "internal_hash": statistics_verification["verification_hash"],
        },
    }
    prior_verifications = []
    for relative in PRIOR_VERIFICATIONS:
        payload = _read_object(repo_root / relative)
        if payload.get("verified") is not True or payload.get("errors") not in ([], None):
            raise ValueError(f"Prior verification is not passed: {relative}")
        prior_verifications.append(_binding(repo_root, relative, internal_field="verification_hash"))
    prior_reports = [_binding(repo_root, relative) for relative in PRIOR_REPORTS]
    failure_evidence = [_binding(repo_root, relative) for relative in FAILURE_EVIDENCE]
    paper_claims_text = (repo_root / PAPER_CLAIMS).read_text(encoding="utf-8")
    for marker in (
        "| C28 | Formal Tier-2 retained all 45 assigned online outcomes",
        "| C29 | Full Decision Admissibility improves target-task training performance over No Memory in the formal experiment. | rejected |",
        "| C30 | Conditional on both systems producing a protocol-legal selected result",
        "| C31 | Formal Tier-2 statistics use no Oracle",
        "| C32 | Injected historical experience caused the conditional Full gains. | pending/not established |",
    ):
        if marker not in paper_claims_text:
            raise ValueError(f"Paper Evidence Ledger marker is absent: {marker}")
    paper_claims_binding = _binding(repo_root, PAPER_CLAIMS)

    per_task = primary["continuous"]["per_task"]
    claims = [
        _claim(
            claim_id="WP8-C1-FORMAL-EXECUTION",
            group="Authority",
            statement="Formal Tier-2 completed all assigned task-seed-system dispositions under host-owned terminal evaluation.",
            status="supported",
            condition="five online systems plus host-only Oracle across three tasks and three seeds",
            sample_unit="task_seed_system_assignment",
            metrics={
                "blocks": 9,
                "assigned_online_outcomes": 45,
                "scored_selected_results": 22,
                "retained_failed_outcomes": 23,
                "oracle_dispositions": 9,
            },
            artifact_bindings=[core_bindings["joint_inventory"], core_bindings["joint_inventory_verification"]],
            claim_gate={"formal_integrity_verified": True, "effect_claim_required": False},
            interpretation="This supports execution completeness and traceability, not Full superiority.",
        ),
        _claim(
            claim_id="WP8-C2-RESULT-WRITEBACK",
            group="Actuation",
            statement="Every successful formal online condition published exactly one independent Result Fact and every failed condition published none.",
            status="supported",
            condition="formal Tier-2 selected-node terminal writeback",
            sample_unit="successful_or_failed_online_condition",
            metrics={
                "successful_conditions": 22,
                "result_facts": 22,
                "fixed_holdout_orphans": 0,
                "failed_conditions_with_result_fact": 0,
            },
            artifact_bindings=[core_bindings["joint_inventory"], core_bindings["statistics_verification"]],
            claim_gate={"result_adoption_causal_separation_required": True, "passed": True},
            interpretation="Result retention is supported; this does not establish Adoption or Causal edges.",
        ),
        _claim(
            claim_id="WP8-C3-FULL-SUPERIORITY",
            group="Authority",
            statement="Full Decision Admissibility improves target-task training performance over No Memory.",
            status="rejected",
            condition="Full minus No Memory primary contrast",
            sample_unit="paired_task_seed_block_with_all_9_assignments_retained",
            metrics={
                "full_completed": statistics["completion_by_system"]["full_decision_admissibility"]["completed"],
                "no_memory_completed": statistics["completion_by_system"]["no_memory"]["completed"],
                "scored_pairs": primary["continuous"]["n_scored_pairs"],
                "assigned_pairs": primary["continuous"]["assigned_pairs"],
                "wins": primary["continuous"]["win_tie_loss"]["wins"],
                "ties": primary["continuous"]["win_tie_loss"]["ties"],
                "losses": primary["continuous"]["win_tie_loss"]["losses"],
                "exact_one_sided_raw_p": primary["continuous"]["exact_one_sided_sign_flip_raw_p"],
                "holm_adjusted_p": primary["continuous"]["holm_adjusted_p"],
            },
            artifact_bindings=[core_bindings["statistics"], core_bindings["statistics_verification"]],
            claim_gate={"effect_claim_authorized": False, "failed_criteria": gate["failed_criteria"]},
            interpretation="Conditional scored pairs are favorable, but poorer completion, missing Taxi pairs and corrected uncertainty reject the headline claim.",
        ),
        _claim(
            claim_id="WP8-C4-CONDITIONAL-UTILITY",
            group="Granularity",
            statement="Among blocks where both Full and No Memory produced legal selected results, all observed paired deltas favor Full.",
            status="diagnostic",
            condition="available-pair analysis only; failures remain in the ITT denominator",
            sample_unit="both_scored_task_seed_pair",
            metrics={
                "available_pairs": 4,
                "assigned_pairs": 9,
                "wins": 4,
                "ties": 0,
                "losses": 0,
                "aerial_mean_macro_f1_delta": per_task["aerial-cactus-identification"]["mean_native_delta"],
                "birds_mean_macro_f1_delta": per_task["mlsp-2013-birds"]["mean_native_delta"],
                "taxi_pair_count": per_task["new-york-city-taxi-fare-prediction"]["n_scored_pairs"],
                "holm_adjusted_p": primary["continuous"]["holm_adjusted_p"],
            },
            artifact_bindings=[core_bindings["statistics"], core_bindings["analysis_policy"]],
            claim_gate={"diagnostic_only": True, "superiority_authorized": False},
            interpretation="This identifies potential conditional value but cannot be generalized to all assigned runs.",
        ),
        _claim(
            claim_id="WP8-C5-NO-IMPUTATION",
            group="Authority",
            statement="Formal statistics retain all failures without Oracle, source-score or other-system score imputation.",
            status="supported",
            condition="all formal Tier-2 outcomes",
            sample_unit="task_seed_system_assignment",
            metrics={
                "assigned": 45,
                "imputed": 0,
                "post_assignment_exclusions": 0,
                "retained_failures": 23,
            },
            artifact_bindings=[core_bindings["analysis_policy"], core_bindings["statistics_verification"]],
            claim_gate={"preregistration_missingness_policy_followed": True},
            interpretation="The negative completion evidence is part of the result rather than filtered away.",
        ),
        _claim(
            claim_id="WP8-C6-EXPERIENCE-CAUSALITY",
            group="Actuation",
            statement="Injected historical experience caused the observed conditional Full gains.",
            status="pending",
            condition="experience-level attribution",
            sample_unit="exposed_experience_to_target_code_edge",
            metrics={"required_minimum_actuation_level": "L4", "formal_system_contrast_is_not_l4": True},
            artifact_bindings=[core_bindings["statistics"], core_bindings["analysis_policy"]],
            claim_gate={"static_runtime_and_counterfactual_required": True, "satisfied": False},
            interpretation="The randomized system contrast is not a substitute for adoption or causal receipts.",
        ),
        _claim(
            claim_id="WP8-C7-PRIOR-KILL-GATES",
            group="Recursive",
            statement="WP8 prevalence, deterministic, controlled, multi-generation and canary evidence packets remain hash-verified inputs to the final synthesis.",
            status="supported",
            condition="pre-formal WP8 evidence tracks",
            sample_unit="immutable_evidence_packet",
            metrics={"verified_prior_packets": len(prior_verifications), "bound_reports": len(prior_reports)},
            artifact_bindings=[*prior_verifications, *prior_reports],
            claim_gate={"packet_verifications_passed": True, "formal_superiority_implied": False},
            interpretation="These packets support mechanism and safety gates; they do not override the formal downstream result.",
        ),
    ]
    ledger: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "status": "complete",
        "created_at": str(created_at),
        "core_artifact_bindings": core_bindings,
        "prior_verification_bindings": prior_verifications,
        "prior_report_bindings": prior_reports,
        "failure_and_retry_evidence": failure_evidence,
        "paper_claims_binding": paper_claims_binding,
        "claims": claims,
        "claim_status_counts": {
            status: sum(row["status"] == status for row in claims)
            for status in ("supported", "diagnostic", "rejected", "pending")
        },
        "headline_effect_claim_authorized": False,
        "engineering_evidence_complete_pending_tests_and_final_gate": True,
        "ledger_hash": "",
    }
    ledger["ledger_hash"] = _payload_hash(ledger, "ledger_hash")
    return ledger


def render_claims_addendum(ledger: Mapping[str, Any]) -> str:
    lines = [
        "# WP8 Formal Evidence Ledger Addendum",
        "",
        f"Ledger hash: `{ledger['ledger_hash']}`",
        "",
        "| ID | Group | Status | Claim gate | Evidence |",
        "|---|---|---|---|---|",
    ]
    for claim in ledger["claims"]:
        gate = json.dumps(claim["claim_gate"], sort_keys=True, ensure_ascii=False)
        evidence = "; ".join(
            f"`{row['path']}` (`{row['file_sha256']}`)"
            for row in claim["artifact_bindings"]
        )
        lines.append(
            f"| {claim['claim_id']} | {claim['group']} | {claim['status']} | "
            f"`{gate}` | {evidence} |"
        )
        lines.append("")
        lines.append(f"{claim['statement']} {claim['interpretation']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_evidence_ledger(output_root: str | Path, **kwargs: Any) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse Evidence Ledger root: {output_root}")
    ledger = compute_evidence_ledger(**kwargs)
    output_root.mkdir(parents=True, exist_ok=False)
    ledger_path = output_root / "evidence_ledger.json"
    addendum_path = output_root / "claims_addendum.md"
    _write_text_exclusive(ledger_path, _json_text(ledger))
    _write_text_exclusive(addendum_path, render_claims_addendum(ledger))
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "files": {
            "claims_addendum.md": _sha256_file(addendum_path),
            "evidence_ledger.json": _sha256_file(ledger_path),
        },
        "ledger_hash": ledger["ledger_hash"],
        "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    _write_text_exclusive(output_root / "manifest.json", _json_text(manifest))
    return ledger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--analysis-policy", required=True, type=Path)
    parser.add_argument("--joint-inventory-root", required=True, type=Path)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    ledger = build_evidence_ledger(
        output_root=args.output_root,
        repo_root=args.repo_root,
        analysis_policy_path=args.analysis_policy,
        joint_inventory_root=args.joint_inventory_root,
        statistics_root=args.statistics_root,
        created_at=args.created_at,
    )
    print(_json_text(ledger), end="")


if __name__ == "__main__":
    main()
