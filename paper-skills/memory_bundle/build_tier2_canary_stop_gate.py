from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from authority.ledger import AuthorityLedger
from engine.candidate_execution_contract import (
    valid_candidate_execution_audit,
    valid_candidate_execution_block_receipt,
)


STOP_GATE_SCHEMA = "decision_admissibility_wp8_tier2_canary_stop_gate_v1"
REGRESSION_SCHEMA = "decision_admissibility_wp8_tier2_canary_regression_receipt_v1"
REQUIRED_TEST_RUNS = (
    "tier2_canary_targeted",
    "plan_section_20_1",
    "plan_section_20_3",
)


def sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return bool(
        payload.get(field)
        and payload[field]
        == sha256_json({key: value for key, value in payload.items() if key != field})
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _verify_authority_ledger_read_only(path: Path) -> bool:
    """Verify a frozen ledger without creating its normal process-lock file."""
    events = _jsonl(path)
    return AuthorityLedger(path).verify(events)


def _packet_inventory(packet_root: Path) -> tuple[dict[str, Any], bool]:
    manifest_path = packet_root / "EVIDENCE_PACKET_MANIFEST.json"
    manifest = read_json(manifest_path)
    actual = {
        str(path.relative_to(packet_root)): sha256_file(path)
        for path in sorted(packet_root.rglob("*"))
        if path.is_file() and path != manifest_path
    }
    return manifest, bool(
        manifest.get("schema")
        == "decision_admissibility_wp8_tier2_canary_evidence_packet_v1"
        and valid_hash(manifest, "packet_hash")
        and manifest.get("file_count") == len(actual)
        and manifest.get("file_hashes") == actual
        and manifest.get("hidden_labels_included") is False
        and manifest.get("solver_secrets_included") is False
        and not any(Path(name).name == "labels.csv" for name in actual)
        and not any(Path(name).suffix == ".env" for name in actual)
    )


def _overlay_result_facts(packet_root: Path, condition: str) -> list[dict[str, Any]]:
    path = packet_root / condition / "session_overlay_events.jsonl"
    return [
        event
        for event in _jsonl(path)
        if event.get("event_type") == "memory_claim"
        and (event.get("payload") or {}).get("publication_class") == "result_fact"
    ]


def validate_evidence_packet(packet_root: str | Path) -> dict[str, Any]:
    packet_root = Path(packet_root).resolve()
    packet_manifest, packet_inventory_valid = _packet_inventory(packet_root)
    source = read_json(packet_root / "source_snapshot_manifest.json")
    source_pre = read_json(packet_root / "source_preflight.json")
    source_post = read_json(packet_root / "source_postrun.json")
    training_isolation = read_json(packet_root / "training_isolation.json")
    evaluator_isolation = read_json(packet_root / "evaluator_isolation.json")
    provider = read_json(packet_root / "provider_attestation.json")
    deletion = read_json(packet_root / "training_pod_deletion_attestation.json")
    training = read_json(packet_root / "training_manifest.json")
    evaluation = read_json(packet_root / "canary_evaluation_summary.json")
    train_manifest = read_json(packet_root / "train_manifest.json")
    evaluator_manifest = read_json(packet_root / "evaluator_manifest.json")
    candidate_environment = read_json(
        packet_root / "candidate_execution_environment.json"
    )
    candidate_contract_artifact = read_json(
        packet_root / "candidate_execution_contract.json"
    )
    execution_contract = training.get("candidate_execution_contract") or {}
    execution_contract_hash_valid = bool(
        execution_contract.get("contract_hash")
        and execution_contract["contract_hash"]
        == sha256_json(
            {
                key: value
                for key, value in execution_contract.items()
                if key != "contract_hash"
            }
        )
    )

    common_manifest_keys = (
        "task_id",
        "split_id",
        "public_tree_sha256",
        "holdout_id_sha256",
        "selection_policy",
    )
    checks: dict[str, bool] = {
        "evidence_packet_hash_inventory_valid": packet_inventory_valid,
        "source_snapshot_hash_valid": valid_hash(source, "source_sha256"),
        "source_snapshot_preflight_verified": source_pre.get("verified") is True,
        "source_snapshot_postrun_verified": source_post.get("verified") is True,
        "source_snapshot_same_pre_and_post": bool(
            source_pre.get("source_sha256")
            == source_post.get("source_sha256")
            == source.get("source_sha256")
            and source_pre.get("file_inventory_matches") is True
            and source_post.get("file_inventory_matches") is True
        ),
        "training_manifest_hash_valid": valid_hash(training, "manifest_hash"),
        "evaluation_summary_hash_valid": valid_hash(evaluation, "summary_hash"),
        "training_view_label_isolated": bool(
            training_isolation.get("training_role") == "label_isolated"
            and training_isolation.get("whole_workspace_mounted") is False
            and training_isolation.get("evaluator_view_mounted") is False
            and training_isolation.get("labels_csv_visible") is False
            and training_isolation.get("source_read_only") is True
            and training_isolation.get("train_view_read_only") is True
            and training_isolation.get("bundle_read_only") is True
        ),
        "independent_cpu_evaluator_isolated": bool(
            evaluator_isolation.get("independent_cpu_evaluator") is True
            and evaluator_isolation.get("whole_workspace_mounted") is False
            and evaluator_isolation.get("solver_secret_mounted") is False
            and evaluator_isolation.get("memory_bundle_mounted") is False
        ),
        "training_pod_deleted_before_evaluation": bool(
            deletion.get("schema")
            == "decision_admissibility_wp8_tier2_training_pod_deletion_v1"
            and deletion.get("training_process_complete_before_deletion") is True
            and deletion.get("training_pod_absent_before_evaluation") is True
            and valid_hash(deletion, "attestation_hash")
            and evaluator_isolation.get("training_pod_absent_before_evaluation")
            is True
        ),
        "real_deepseek_provider_attested": bool(
            provider.get("provider") == "deepseek"
            and provider.get("api_key_present") is True
            and provider.get("non_local_endpoint") is True
            and provider.get("real_provider_required") is True
            and len(str(provider.get("base_url_sha256") or "")) == 64
        ),
        "fixed_holdout_train_evaluator_manifests_match": bool(
            train_manifest.get("role") == "train_view"
            and evaluator_manifest.get("role") == "evaluator_view"
            and train_manifest.get("hidden_labels_present") is False
            and all(
                train_manifest.get(key) == evaluator_manifest.get(key)
                for key in common_manifest_keys
            )
            and train_manifest.get("metric")
            == evaluator_manifest.get("metric")
            == "binary_roc_auc"
        ),
        "same_source_seed_steps_and_bundle_binding": bool(
            training.get("status") == "training_complete_unscored"
            and training.get("condition_order") == ["nm", "full"]
            and isinstance(training.get("same_host_seed"), int)
            and training.get("steps_per_condition") == 6
            and training.get("initial_drafts_per_condition") == 3
            and training.get("repair_steps_budget_per_condition") == 3
            and training.get("same_source_snapshot") is True
            and training.get("same_bundle_binding") is True
        ),
        "paired_conditions_disable_legacy_static_coldstart": bool(
            training.get("legacy_static_coldstart_enabled") is False
            and training.get(
                "condition_difference_limited_to_external_memory_retrieval"
            )
            is True
        ),
        "paired_candidate_execution_contract_enforced": bool(
            execution_contract_hash_valid
            and execution_contract == candidate_contract_artifact
            and execution_contract.get("enabled") is True
            and execution_contract.get("max_execution_seconds") == 600
            and execution_contract.get("max_epochs") == 8
            and execution_contract.get("max_cv_folds") == 1
            and execution_contract.get("max_trainable_models") == 1
            and execution_contract.get("allow_remote_assets") is False
            and execution_contract.get("allow_unverified_local_assets") is False
            and execution_contract.get(
                "allow_dataset_wide_per_sample_precompute"
            )
            is False
            and execution_contract.get("allow_source_score_inheritance") is False
            and training.get("same_candidate_execution_contract") is True
            and training.get("candidate_execution_contract_host_enforced") is True
            and candidate_environment.get(
                "all_allowed_import_roots_importable"
            )
            is True
            and candidate_environment.get("allowed_import_roots")
            == execution_contract.get("allowed_import_roots")
            and all(
                (candidate_environment.get("imports") or {})
                .get(root, {})
                .get("importable")
                is True
                for root in execution_contract.get("allowed_import_roots") or []
            )
        ),
        "terminal_scores_hidden_during_search": bool(
            training.get("terminal_scores_visible_during_search") is False
            and all(
                (training.get("conditions") or {}).get(name, {}).get(
                    "pre_evaluator_score_file_count"
                )
                == 0
                for name in ("nm", "full")
            )
            and evaluation.get("scores_used_for_further_search") is False
        ),
        "canary_claim_boundary_preserved": bool(
            training.get("effect_claim_authorized") is False
            and training.get("formal_tier2_evidence") is False
            and evaluation.get("effect_claim_authorized") is False
            and evaluation.get("full_superiority_claim_authorized") is False
            and evaluation.get("formal_tier2_evidence") is False
        ),
    }

    condition_summary: dict[str, Any] = {}
    for condition in ("nm", "full"):
        exposure = read_json(packet_root / condition / "exposure_audit.json")
        score = read_json(packet_root / condition / "fixed_holdout_scores.json")
        writeback = read_json(
            packet_root / condition / "fixed_holdout_writeback_status.json"
        )
        journal = read_json(packet_root / condition / "journal.json")
        request = read_json(packet_root / condition / "evaluation_request.json")
        rollout = read_json(
            packet_root / condition / "authority_rollout_report.json"
        )
        ledger_path = packet_root / condition / "authority_events.jsonl"
        facts = _overlay_result_facts(packet_root, condition)
        best_node_id = str(score.get("best_node_id") or "")
        nodes = {
            str(node.get("id")): node
            for node in journal.get("nodes") or []
            if isinstance(node, dict) and node.get("id")
        }
        score_value = score.get("best_score")
        candidate_nodes = {
            node_id: node
            for node_id, node in nodes.items()
            if node.get("stage") != "root"
        }
        training_row = (training.get("conditions") or {}).get(condition, {})
        audit_paths = sorted(
            (packet_root / condition / "candidate_execution_audits").glob(
                "candidate_execution_contract_audit_*.json"
            )
        )
        audit_payloads = [read_json(path) for path in audit_paths]
        audit_node_ids = [
            path.stem.removeprefix("candidate_execution_contract_audit_")
            for path in audit_paths
        ]
        original_hashes = training_row.get("file_hashes") or {}
        audit_by_node = dict(zip(audit_node_ids, audit_payloads))
        block_paths = sorted(
            (
                packet_root
                / condition
                / "candidate_execution_block_receipts"
            ).glob("candidate_execution_block_receipt_*.json")
        )
        block_node_ids = [
            path.stem.removeprefix("candidate_execution_block_receipt_")
            for path in block_paths
        ]
        block_by_node = {
            node_id: read_json(path)
            for node_id, path in zip(block_node_ids, block_paths)
        }
        admitted_node_ids = {
            node_id
            for node_id, payload in audit_by_node.items()
            if payload.get("valid") is True
        }
        denied_node_ids = set(audit_by_node) - admitted_node_ids
        submitted_node_ids = set(
            training_row.get("candidate_execution_submitted_node_ids") or []
        )
        runtime_failed_admitted_node_ids = {
            node_id
            for node_id in admitted_node_ids
            if candidate_nodes.get(node_id, {}).get("is_buggy") is True
        }

        def packet_file_bound(path: Path) -> bool:
            return sha256_file(path) in {
                digest
                for relative, digest in original_hashes.items()
                if Path(relative).name == path.name
            }

        checks[f"{condition}_candidate_execution_contract_bound"] = bool(
            len(audit_paths)
            == len(candidate_nodes)
            == training.get("steps_per_condition")
            and training_row.get("candidate_execution_audit_count")
            == len(audit_paths)
            and training_row.get("candidate_execution_audits_integrity_valid")
            is True
            and training_row.get("candidate_execution_denials_enforced") is True
            and training_row.get("candidate_execution_contract_role_binding_valid")
            is True
            and training_row.get("candidate_execution_contract_hash")
            == execution_contract.get("contract_hash")
            and set(audit_node_ids) == set(candidate_nodes)
            and all(
                valid_candidate_execution_audit(payload)
                and payload.get("contract_hash")
                == execution_contract.get("contract_hash")
                and payload.get("code_sha256")
                == hashlib.sha256(
                    str(candidate_nodes[node_id].get("code") or "").encode()
                ).hexdigest()
                and packet_file_bound(path)
                for path, payload, node_id in zip(
                    audit_paths, audit_payloads, audit_node_ids
                )
            )
            and all(
                (node.get("role_contract") or {}).get(
                    "candidate_execution_contract"
                )
                == execution_contract
                for node in candidate_nodes.values()
            )
            and set(block_by_node) == denied_node_ids
            and all(
                valid_candidate_execution_block_receipt(block_by_node[node_id])
                and block_by_node[node_id].get("node_id") == node_id
                and block_by_node[node_id].get("contract_hash")
                == execution_contract.get("contract_hash")
                and block_by_node[node_id].get("audit_hash")
                == audit_by_node[node_id].get("audit_hash")
                and block_by_node[node_id].get("code_sha256")
                == audit_by_node[node_id].get("code_sha256")
                and packet_file_bound(path)
                and candidate_nodes[node_id].get("exc_type")
                == "CandidateExecutionContractError"
                and candidate_nodes[node_id].get("is_buggy") is True
                and (candidate_nodes[node_id].get("metric") or {}).get("value")
                is None
                and float(candidate_nodes[node_id].get("exec_time") or 0.0)
                < 30.0
                and Path(
                    (candidate_nodes[node_id].get("exc_info") or {}).get(
                        "block_receipt_path", ""
                    )
                ).name
                == path.name
                and (candidate_nodes[node_id].get("exc_info") or {}).get(
                    "block_receipt_hash"
                )
                == block_by_node[node_id].get("receipt_hash")
                for node_id, path in zip(block_node_ids, block_paths)
            )
            and submitted_node_ids
            and submitted_node_ids <= admitted_node_ids
            and not submitted_node_ids & denied_node_ids
            and training_row.get("candidate_execution_admitted_count")
            == len(admitted_node_ids)
            and training_row.get("candidate_execution_denied_count")
            == len(denied_node_ids)
            and training_row.get("candidate_execution_block_receipt_count")
            == len(block_paths)
            and training_row.get(
                "candidate_execution_runtime_failed_admitted_count"
            )
            == len(runtime_failed_admitted_node_ids)
            and training_row.get("candidate_execution_submitted_admitted_count")
            == len(submitted_node_ids)
            and set(training_row.get("candidate_execution_admitted_node_ids") or [])
            == admitted_node_ids
            and set(training_row.get("candidate_execution_denied_node_ids") or [])
            == denied_node_ids
            and training_row.get("submission_count") == len(submitted_node_ids)
        )
        score_result_node_ids = {
            str(row.get("node_id")) for row in score.get("results") or []
        }
        checks[f"{condition}_score_report_valid"] = bool(
            valid_hash(score, "report_hash")
            and score.get("report_schema")
            == "fixed_holdout_terminal_score_report_v2"
            and score.get("terminal_score_sealed") is True
            and score.get("candidate_set_frozen_before_scoring") is True
            and score.get("scores_were_visible_during_search") is False
            and score.get("selection_policy") == "terminal_only"
            and isinstance(score_value, (int, float))
            and math.isfinite(float(score_value))
            and 0.0 <= float(score_value) <= 1.0
            and best_node_id in nodes
            and bool(nodes[best_node_id].get("code"))
            and nodes[best_node_id].get("exec_time") is not None
            and score_result_node_ids <= submitted_node_ids
            and not score_result_node_ids & denied_node_ids
        )
        checks[f"{condition}_terminal_result_writeback_valid"] = bool(
            valid_hash(writeback, "status_hash")
            and writeback.get("status") == "complete"
            and writeback.get("completion") in {"finalized", "already_finalized"}
            and len(facts) == 1
            and (facts[0].get("payload") or {}).get("artifact_id") == best_node_id
            and (facts[0].get("payload") or {}).get("derived_from_refs") == []
            and writeback.get("overlay_event_id") == facts[0].get("event_id")
            and writeback.get("overlay_event_hash") == facts[0].get("event_hash")
            and _verify_authority_ledger_read_only(ledger_path)
        )
        checks[f"{condition}_bundle_and_protocol_bound"] = bool(
            rollout.get("mode") == "enforce"
            and (rollout.get("rollout_versions") or {}).get("bundle_id")
            == training_isolation.get("bundle_id")
            and (rollout.get("rollout_versions") or {}).get(
                "bundle_manifest_sha256"
            )
            == training_isolation.get("bundle_manifest_sha256")
            and request.get("task_id") == training.get("task_id")
        )
        if condition == "nm":
            checks["no_memory_bundle_bound_zero_exposure"] = bool(
                (training.get("conditions") or {}).get("nm", {}).get(
                    "retrieval_control"
                )
                == "no_memory"
                and exposure.get("valid") is True
                and exposure.get("exposure_event_count") == 0
                and exposure.get("invalid_exposure_count") == 0
                and (training.get("conditions") or {}).get("nm", {}).get(
                    "experience_exposure_count"
                )
                == 0
            )
        else:
            checks["full_same_domain_method_exposure_valid"] = bool(
                (training.get("conditions") or {}).get("full", {}).get(
                    "retrieval_control"
                )
                == "stage_hybrid"
                and exposure.get("valid") is True
                and exposure.get("invalid_exposure_count") == 0
                and exposure.get("certified_method_exposure_count", 0) > 0
                and exposure.get("cross_domain_exposure_count", 0) == 0
                and (training.get("conditions") or {}).get("full", {}).get(
                    "experience_exposure_count",
                    0,
                )
                > 0
            )
        condition_summary[condition] = {
            "best_score": score_value,
            "best_node_id": best_node_id,
            "scored_candidate_count": sum(
                row.get("status") == "scored" for row in score.get("results") or []
            ),
            "exposure_event_count": exposure.get("exposure_event_count", 0),
            "result_fact_count": len(facts),
            "candidate_execution_audit_count": len(audit_paths),
            "candidate_execution_admitted_count": len(admitted_node_ids),
            "candidate_execution_denied_count": len(denied_node_ids),
            "candidate_execution_runtime_failed_admitted_count": len(
                runtime_failed_admitted_node_ids
            ),
            "candidate_execution_submitted_admitted_count": len(
                submitted_node_ids
            ),
            "result_fact_derived_from_refs": (facts[0].get("payload") or {}).get(
                "derived_from_refs"
            )
            if len(facts) == 1
            else None,
        }

    return {
        "checks": checks,
        "packet_hash": packet_manifest.get("packet_hash", ""),
        "source_sha256": source.get("source_sha256", ""),
        "task_id": training.get("task_id"),
        "protocol_ref": training.get("protocol_ref"),
        "fixed_holdout_metric": training.get("fixed_holdout_metric"),
        "same_host_seed": training.get("same_host_seed"),
        "steps_per_condition": training.get("steps_per_condition"),
        "initial_drafts_per_condition": training.get(
            "initial_drafts_per_condition"
        ),
        "repair_steps_budget_per_condition": training.get(
            "repair_steps_budget_per_condition"
        ),
        "condition_summary": condition_summary,
    }


def _prior_gate_checks(root: Path) -> dict[str, bool]:
    report_path = root / "stop_gate_report.json"
    verification_path = root / "verification.json"
    report = read_json(report_path)
    verification = read_json(verification_path)
    return {
        "prior_gate5_report_hash_valid": valid_hash(report, "report_hash"),
        "prior_gate5_verification_hash_valid": valid_hash(
            verification, "verification_hash"
        ),
        "prior_gate5_passed": bool(
            report.get("passed") is True
            and report.get("status") == "pass"
            and verification.get("verified") is True
            and verification.get("errors") == []
        ),
        "prior_gate5_authorized_tier2_canary_only": bool(
            report.get("next_authorized_phase") == "WP8 Tier-2 canary"
            and report.get("tier2_canary_authorized") is True
            and report.get("large_scale_tier2_authorized") is False
        ),
        "prior_gate5_files_bound": bool(
            verification.get("stop_gate_report_hash") == report.get("report_hash")
            and verification.get("stop_gate_report_file_sha256")
            == sha256_file(report_path)
        ),
    }


def _regression_checks(path: Path) -> dict[str, bool]:
    receipt = read_json(path)
    rows = {str(row.get("name")): row for row in receipt.get("test_runs") or []}
    return {
        "regression_receipt_hash_valid": bool(
            receipt.get("schema") == REGRESSION_SCHEMA
            and valid_hash(receipt, "receipt_hash")
        ),
        "required_regression_scopes_clean": all(
            name in rows
            and rows[name].get("exit_code") == 0
            and rows[name].get("failed") == 0
            for name in REQUIRED_TEST_RUNS
        ),
        "protected_preexisting_assets_unchanged": receipt.get(
            "protected_preexisting_assets_unchanged"
        )
        is True,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    rows = report["condition_summary"]
    lines = [
        "# WP8 Tier-2 Canary Stop Gate",
        "",
        f"- Status: **{report['status'].upper()}**",
        f"- Report hash: `{report['report_hash']}`",
        f"- Next authorized phase: `{report['next_authorized_phase']}`",
        f"- Formal Tier-2 authorized: `{str(report['formal_tier2_authorized']).lower()}`",
        f"- Paper effect claim authorized: `{str(report['paper_effect_claim_authorized']).lower()}`",
        "",
        "## Frozen canary outcomes",
        "",
        "| Condition | Terminal AUC | Scored candidates | Prompt exposures | Result Facts |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in ("nm", "full"):
        row = rows[condition]
        lines.append(
            f"| {condition} | {row['best_score']} | {row['scored_candidate_count']} | "
            f"{row['exposure_event_count']} | {row['result_fact_count']} |"
        )
    lines.extend(["", "## Stop-Gate checks", ""])
    for name in sorted(report["stop_gate_checks"]):
        lines.append(
            f"- [{'x' if report['stop_gate_checks'][name] else ' '}] `{name}`"
        )
    lines.extend(
        [
            "",
            "## Claim boundaries",
            "",
            "- This is an infrastructure canary, not formal Tier-2 effect evidence.",
            "- The two terminal AUC values are reported without an efficacy or Full-superiority claim.",
            "- No-Memory still binds the same immutable Bundle so terminal Result writeback remains comparable.",
            "- Result Facts describe each newly executed target node; they do not assert adoption or causality.",
            "- Formal Tier-2 must run the preregistered six-system matrix, at least three protocol families, and at least three agent seeds per task/system.",
            "",
        ]
    )
    return "\n".join(lines)


def compute_stop_gate(
    *,
    plan_path: str | Path,
    prior_gate5_root: str | Path,
    evidence_packet_root: str | Path,
    regression_receipt_path: str | Path,
    created_at: str,
) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    prior_gate5_root = Path(prior_gate5_root).resolve()
    evidence_packet_root = Path(evidence_packet_root).resolve()
    regression_receipt_path = Path(regression_receipt_path).resolve()
    evidence = validate_evidence_packet(evidence_packet_root)
    checks = {
        **_prior_gate_checks(prior_gate5_root),
        **_regression_checks(regression_receipt_path),
        **evidence["checks"],
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": STOP_GATE_SCHEMA,
        "status": "pass" if passed else "fail",
        "passed": passed,
        "created_at": created_at,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "prior_gate5_root": str(prior_gate5_root),
        "evidence_packet_root": str(evidence_packet_root),
        "evidence_packet_hash": evidence["packet_hash"],
        "regression_receipt_path": str(regression_receipt_path),
        "regression_receipt_file_sha256": sha256_file(regression_receipt_path),
        "source_sha256": evidence["source_sha256"],
        "task_id": evidence["task_id"],
        "protocol_ref": evidence["protocol_ref"],
        "fixed_holdout_metric": evidence["fixed_holdout_metric"],
        "same_host_seed": evidence["same_host_seed"],
        "steps_per_condition": evidence["steps_per_condition"],
        "initial_drafts_per_condition": evidence[
            "initial_drafts_per_condition"
        ],
        "repair_steps_budget_per_condition": evidence[
            "repair_steps_budget_per_condition"
        ],
        "condition_summary": evidence["condition_summary"],
        "stop_gate_checks": checks,
        "required_check_count": len(checks),
        "passed_check_count": sum(value is True for value in checks.values()),
        "next_authorized_phase": (
            "WP8 Tier-2 formal experiment staging" if passed else "none"
        ),
        "formal_tier2_authorized": passed,
        "large_scale_effect_claim_authorized": False,
        "paper_effect_claim_authorized": False,
        "wp8_complete": False,
        "formal_tier2_requirements": {
            "minimum_protocol_families": 3,
            "minimum_agent_seeds_per_task_system": 3,
            "required_systems": [
                "no_memory",
                "flat_relevance_memory",
                "global_validity_bit",
                "authority_only",
                "full_decision_admissibility",
                "oracle",
            ],
            "host_owned_terminal_evaluator_only": True,
            "counterbalanced_condition_order": True,
        },
        "claim_boundaries": [
            "canary_not_formal_effect_evidence",
            "no_full_superiority_claim",
            "terminal_result_fact_is_not_adoption_or_causality",
            "source_scores_not_inherited",
        ],
        "builder_source_sha256": sha256_file(Path(__file__).resolve()),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


def _write_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--prior-gate5-root", required=True, type=Path)
    parser.add_argument("--evidence-packet-root", required=True, type=Path)
    parser.add_argument("--regression-receipt", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    report = compute_stop_gate(
        plan_path=args.plan,
        prior_gate5_root=args.prior_gate5_root,
        evidence_packet_root=args.evidence_packet_root,
        regression_receipt_path=args.regression_receipt,
        created_at=args.created_at,
    )
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write_exclusive(
        output / "stop_gate_report.json",
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )
    _write_exclusive(output / "stop_gate_report.md", render_markdown(report))
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
