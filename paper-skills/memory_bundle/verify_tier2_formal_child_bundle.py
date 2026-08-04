from __future__ import annotations

import argparse
import dataclasses
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from authority.adapters.mlevolve.retrieval_gate import (
    authorize_clause_for_visibility,
)
from authority.authority_engine import AuthorityEngine
from authority.bundle_authority import load_snapshot_authority
from authority.clean_replay import verify_trusted_receipt_integrity
from authority.memory_snapshot import MemorySnapshotLoader
from authority.models import (
    GenerationStage,
    GovernanceStage,
    Operation,
    ProtocolRef,
    Receipt,
    ReceiptType,
    SOPClauseV1,
    TaskContext,
    VisibilityRequest,
)
from authority.protocol_registry import ProtocolRegistry

from bind_sop_clauses import read_jsonl
from method_claim_purity import audit_method_claim_semantic_purity
from schema import payload_hash, read_json, sha256_file, sha256_json
from validate_memory_bundle import validate_bundle


SCHEMA = "tier2_formal_child_bundle_independent_verification_v1"
GENERATION_STAGES = {
    GenerationStage.DRAFT.value,
    GenerationStage.MODEL_DESIGN.value,
    GenerationStage.IMPROVE.value,
    GenerationStage.EVOLUTION.value,
    GenerationStage.FUSION.value,
}
LEGACY_PARENT_LINEAGE_ERROR_PREFIXES = {
    "runforest_domain_scope_not_required",
    "runforest_source_task_lineage_mismatch",
    "runforest_source_family_lineage_mismatch",
    "runforest_source_domain_lineage_mismatch",
    "container_domain_scope_incomplete",
    "clause_source_run_lineage",
    "clause_source_task_lineage",
    "clause_source_family_lineage",
    "clause_source_domain_lineage",
    "clause_missing_transfer_scope",
}


def _protocol_hash(value: object) -> str:
    if isinstance(value, Mapping):
        return str(value.get("canonical_hash") or "")
    _prefix, separator, digest = str(value or "").partition("#")
    return digest if separator else ""


def _parse_protocol_ref(value: str) -> ProtocolRef:
    prefix, separator, digest = str(value or "").partition("#")
    protocol_id, version_separator, version = prefix.rpartition("@")
    if not separator or not version_separator or not protocol_id or not version or not digest:
        raise ValueError(f"Invalid ProtocolRef: {value}")
    return ProtocolRef(protocol_id, version, digest)


def _receipt(row: Mapping[str, Any]) -> Receipt:
    values = dict(row)
    values["receipt_type"] = ReceiptType(str(values["receipt_type"]))
    return Receipt(**values)


def _is_read_only(path: Path) -> bool:
    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    return not bool(path.stat().st_mode & writable)


def verify_formal_child_publication(
    publication_root: str | Path,
    *,
    expected_parent_bundle_id: str,
    expected_parent_manifest_sha256: str,
    target_task_id: str,
    target_task_family: str,
    target_domain: str,
    split_mode: str,
    source_clause_id: str,
    source_run_id: str,
    source_node_id: str,
    publication_class: str,
    protocol_ref: str,
    expected_parent_validation_mode: str = "strict_current_schema",
    allowed_protocol_issue_codes: Iterable[str] = (),
    agent_seeds: Iterable[int] = (104729, 130363, 155921),
) -> dict[str, Any]:
    """Independently verify a sealed child without importing its publisher."""

    publication_root = Path(publication_root).resolve()
    expected_issues = sorted(set(map(str, allowed_protocol_issue_codes)))
    expected_seeds = {str(int(value)) for value in agent_seeds}
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: Any) -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(name)

    try:
        check("publication_root_exists", publication_root.is_dir())
        check("publication_root_not_symlink", not publication_root.is_symlink())
        all_paths = [publication_root, *publication_root.rglob("*")]
        check(
            "publication_has_no_symlinks",
            all(not path.is_symlink() for path in all_paths),
        )
        check(
            "publication_tree_read_only",
            all(_is_read_only(path) for path in all_paths),
        )

        current_path = publication_root / "CURRENT.json"
        current = read_json(current_path)
        check(
            "current_internal_hash",
            current.get("pointer_sha256")
            == sha256_json(
                {
                    key: value
                    for key, value in current.items()
                    if key != "pointer_sha256"
                }
            ),
        )
        check("current_parent", current.get("parent_bundle") == expected_parent_bundle_id)

        loader = MemorySnapshotLoader(publication_root)
        base = loader.load_base(verify_artifacts=True)
        check("current_bundle_id", current.get("bundle_id") == base.bundle_id)
        check("current_bundle_version", current.get("bundle_version") == base.bundle_version)
        check("current_manifest", current.get("manifest_sha256") == base.manifest_sha256)
        check(
            "manifest_file_hash_self",
            base.manifest_file_sha256 == sha256_file(base.path / "manifest.json"),
        )
        check(
            "current_bundle_path",
            (publication_root / str(current.get("bundle_path") or "")).resolve()
            == base.path,
        )

        generic = validate_bundle(base.path)
        check("generic_bundle_validation", generic.get("valid") is True)
        check("manifest_parent", base.manifest.get("parent_bundle") == expected_parent_bundle_id)

        publication_path = publication_root / "reports" / "publication_report.json"
        publication = read_json(publication_path)
        check(
            "publication_report_internal_hash",
            publication.get("report_hash")
            == payload_hash(publication, "report_hash"),
        )
        expected_publication_fields = {
            "bundle_id": base.bundle_id,
            "bundle_version": base.bundle_version,
            "bundle_manifest_sha256": base.manifest_sha256,
            "current_pointer_sha256": current.get("pointer_sha256"),
            "parent_bundle_id": expected_parent_bundle_id,
            "parent_manifest_sha256": expected_parent_manifest_sha256,
            "target_task_id": target_task_id,
            "target_task_family": target_task_family,
            "target_domain": target_domain,
            "split_mode": split_mode,
            "source_clause_id": source_clause_id,
            "source_run_id": source_run_id,
            "source_node_id": source_node_id,
            "formal_protocol_ref": protocol_ref,
            "publication_class": publication_class,
        }
        for key, expected in expected_publication_fields.items():
            check(f"publication_field:{key}", publication.get(key) == expected)
        check("publication_source_score_false", publication.get("source_score_inheritance") is False)
        check("publication_historical_metric_false", publication.get("historical_metric_used_as_evidence") is False)
        check("publication_issue_set", publication.get("allowed_protocol_issue_codes") == expected_issues)
        parent_disposition = publication.get("parent_validation_disposition") or {}
        check(
            "parent_disposition_internal_hash",
            parent_disposition.get("disposition_hash")
            == payload_hash(parent_disposition, "disposition_hash"),
        )
        check("parent_disposition_accepted", parent_disposition.get("accepted") is True)
        check("parent_disposition_reasons_empty", parent_disposition.get("reasons") == [])
        check(
            "parent_disposition_mode",
            parent_disposition.get("mode") == expected_parent_validation_mode,
        )
        check(
            "parent_disposition_allowlist_exact",
            set(parent_disposition.get("allowed_legacy_error_prefixes") or [])
            == LEGACY_PARENT_LINEAGE_ERROR_PREFIXES,
        )
        observed_parent_prefixes = set(
            (parent_disposition.get("validator_error_prefix_counts") or {}).keys()
        )
        if expected_parent_validation_mode == "strict_current_schema":
            check("strict_parent_validator_valid", parent_disposition.get("validator_valid") is True)
            check("strict_parent_error_count_zero", parent_disposition.get("validator_error_count") == 0)
        else:
            check("legacy_parent_validator_not_current", parent_disposition.get("validator_valid") is False)
            check("legacy_parent_errors_present", int(parent_disposition.get("validator_error_count") or 0) > 0)
            check("legacy_parent_prefixes_narrow", bool(observed_parent_prefixes) and observed_parent_prefixes <= LEGACY_PARENT_LINEAGE_ERROR_PREFIXES)

        source_view_path = publication_root / "build_inputs" / "parent_source_view"
        source_view_report = read_json(
            publication_root
            / "build_inputs"
            / "parent_source_view_report.json"
        )
        source_view_hashes = source_view_report.get("artifact_hashes") or {}
        check(
            "source_view_hash",
            source_view_report.get("source_view_hash")
            == sha256_json(dict(sorted(source_view_hashes.items())))
            == publication.get("parent_source_view_hash"),
        )
        check(
            "source_view_parent",
            source_view_report.get("parent_bundle_id")
            == expected_parent_bundle_id
            and source_view_report.get("parent_manifest_sha256")
            == expected_parent_manifest_sha256,
        )
        source_view_artifacts_valid = True
        for relative, expected_hash in source_view_hashes.items():
            relative_path = Path(str(relative))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                source_view_artifacts_valid = False
                continue
            artifact = source_view_path / relative_path
            if not artifact.is_file() or sha256_file(artifact) != expected_hash:
                source_view_artifacts_valid = False
        check("source_view_artifacts", source_view_artifacts_valid)

        registry = ProtocolRegistry(base.path / "protocol_registry")
        active_protocol = registry.resolve(_parse_protocol_ref(protocol_ref)).ref()
        check("protocol_exact", active_protocol.key() == protocol_ref)

        split = base.read_json("splits/active.json")
        source_runs = set(map(str, split.get("source_run_ids") or []))
        heldout_runs = set(map(str, split.get("heldout_run_ids") or []))
        check("split_kind", split.get("split_kind") == split_mode)
        check("split_manifest_binding", split.get("manifest_sha256") == publication.get("split_manifest_sha256"))
        check("source_run_present", source_run_id in source_runs)
        check(
            "source_view_run_set",
            set(map(str, source_view_report.get("source_run_ids") or []))
            == source_runs,
        )
        check("run_overlap_zero", not source_runs & heldout_runs)
        source_seed_values = {
            str(value).rsplit("::", 1)[-1]
            for value in split.get("source_seed_groups") or []
        }
        check("formal_agent_seed_overlap_zero", not source_seed_values & expected_seeds)
        if split_mode == "same-domain-task-heldout":
            check("target_history_absent", target_task_id not in set(split.get("source_task_ids") or []))
            check("heldout_target_exact", split.get("heldout_task_ids") == [target_task_id])

        corpus = base.read_json("corpus/manifest.json")
        check(
            "corpus_rebased_to_source_view",
            Path(str(corpus.get("source_root") or "")).resolve()
            == source_view_path.resolve(),
        )
        corpus_runs = {
            str(row.get("run_id") or ""): row
            for row in corpus.get("runs") or []
        }
        core_fields = (
            "journal_path",
            "config_path",
            "filtered_journal_path",
            "best_solution_path",
        )
        check(
            "source_core_paths_materialized",
            all(
                all(
                    not row.get(field)
                    or (source_view_path / str(row[field])).is_file()
                    for field in core_fields
                )
                for run_id, row in corpus_runs.items()
                if run_id in source_runs
            ),
        )
        check(
            "heldout_core_paths_absent",
            all(
                all(not row.get(field) for field in core_fields)
                for run_id, row in corpus_runs.items()
                if run_id in heldout_runs
            ),
        )

        graph = base.read_json("runforest/graph.json")
        run_nodes = [
            row
            for row in graph.get("nodes") or []
            if row.get("type") in {"Run", "RunNode"}
        ]
        check("runforest_nonempty", bool(run_nodes))
        check(
            "runforest_domain_exact",
            all(str(row.get("task_domain") or "") == target_domain for row in run_nodes),
        )

        clauses = base.read_jsonl("sop/clauses.jsonl")
        clause_by_id = {str(row.get("clause_id") or ""): row for row in clauses}
        check("clause_ids_unique", len(clause_by_id) == len(clauses))
        formal_clause_id = str(publication.get("formal_clause_id") or "")
        formal_debug_clause_id = str(
            publication.get("formal_debug_clause_id") or ""
        )
        formal_claim_id = str(publication.get("formal_claim_id") or "")
        formal_path_id = str(publication.get("formal_path_id") or "")
        formal_receipt_ids = set(map(str, publication.get("formal_receipt_ids") or []))
        formal = clause_by_id.get(formal_clause_id) or {}
        formal_debug = clause_by_id.get(formal_debug_clause_id) or {}
        check("formal_clause_present", bool(formal))
        check("formal_clause_claim", formal.get("claim_refs") == [formal_claim_id])
        check("formal_clause_type", formal.get("claim_types") == ["method_hypothesis"])
        check("formal_clause_protocol", formal.get("protocol_scope") == [protocol_ref])
        check("formal_clause_domain", formal.get("source_domains") == [target_domain])
        check("formal_clause_transfer", formal.get("transfer_scope") == "same_domain")
        check("formal_clause_publication", formal.get("publication_class") == publication_class)
        check("formal_clause_operation", formal.get("permitted_operations") == ["generate_candidate"])
        check("formal_clause_generation_stages", set(formal.get("permitted_generation_stages") or []) == GENERATION_STAGES)
        check("formal_clause_governance", formal.get("permitted_governance_stages") == ["retrieval"])
        check("formal_clause_receipts", set(map(str, formal.get("receipt_refs") or [])) == formal_receipt_ids)
        check("formal_debug_clause_present", bool(formal_debug))
        check(
            "formal_debug_clause_claim",
            formal_debug.get("claim_refs") == [formal_claim_id],
        )
        check(
            "formal_debug_clause_operation",
            formal_debug.get("permitted_operations") == ["debug_hypothesis"],
        )
        check(
            "formal_debug_clause_stage",
            formal_debug.get("permitted_generation_stages") == ["debug"],
        )
        check(
            "formal_debug_clause_transition",
            formal_debug.get("source_transition_refs")
            == formal.get("source_transition_refs")
            and len(formal_debug.get("source_transition_refs") or []) == 1,
        )
        contract = formal.get("contract_spec") or {}
        check("formal_source_clause", contract.get("source_clause_id") == source_clause_id)
        check("formal_parent_bundle", contract.get("parent_bundle_id") == expected_parent_bundle_id)
        check("formal_parent_manifest", contract.get("parent_manifest_sha256") == expected_parent_manifest_sha256)
        check("formal_source_score_false", contract.get("source_score_inheritance") is False)
        check("formal_historical_metric_false", contract.get("historical_metric_used_as_evidence") is False)
        purity = audit_method_claim_semantic_purity(str(formal.get("retrieval_text") or ""))
        check("formal_method_pure", purity.get("passed") is True)
        check("formal_method_purity_binding", purity.get("report_hash") == contract.get("method_semantic_purity_report_hash"))
        target_protocol_clauses = [
            row for row in clauses if protocol_ref in set(row.get("protocol_scope") or [])
        ]
        check(
            "target_protocol_clause_pair",
            {str(row.get("clause_id") or "") for row in target_protocol_clauses}
            == {formal_clause_id, formal_debug_clause_id},
        )
        check("flat_raw_population_present", len(clauses) > 1)
        check(
            "all_clause_domains_in_scope",
            all(
                set(map(str, row.get("source_domains") or []))
                in ({target_domain}, set())
                for row in clauses
            ),
        )

        claim_rows = base.read_jsonl("authority/claims.jsonl")
        target_claims = [
            row
            for row in claim_rows
            if _protocol_hash(row.get("protocol_ref")) == active_protocol.canonical_hash
        ]
        check("one_target_protocol_claim", len(target_claims) == 1)
        claim = target_claims[0] if len(target_claims) == 1 else {}
        check("formal_claim_id", claim.get("claim_id") == formal_claim_id)
        check("formal_claim_type", claim.get("claim_type") == "method_hypothesis")
        boundary = claim.get("boundary") or {}
        check("claim_publication_class", boundary.get("publication_class") == publication_class)
        check("claim_same_domain_only", boundary.get("same_domain_only") is True)
        check("claim_source_score_false", boundary.get("source_score_inheritance") is False)
        check("claim_historical_metric_false", boundary.get("historical_metric_used_as_evidence") is False)
        check("claim_issue_set", boundary.get("source_audit_issue_codes") == expected_issues)
        check("claim_allowed_issue_set", boundary.get("allowed_protocol_issue_codes") == expected_issues)

        receipt_rows = {
            str(row.get("receipt_id") or ""): row
            for row in base.read_jsonl("authority/receipts.jsonl")
        }
        check("formal_receipt_count", len(formal_receipt_ids) == 2)
        check("formal_receipts_resolve", formal_receipt_ids <= set(receipt_rows))
        selected_receipts = [
            _receipt(receipt_rows[receipt_id])
            for receipt_id in sorted(formal_receipt_ids)
            if receipt_id in receipt_rows
        ]
        for receipt in selected_receipts:
            try:
                verify_trusted_receipt_integrity(receipt)
                integrity = True
            except Exception:
                integrity = False
            check(f"receipt_integrity:{receipt.receipt_id}", integrity)
            check(
                f"receipt_artifact:{receipt.receipt_id}",
                receipt.artifact_id == claim.get("subject_artifact_id"),
            )
        check(
            "formal_receipt_types_exact",
            {receipt.receipt_type for receipt in selected_receipts}
            == {ReceiptType.METHOD_IDENTITY, ReceiptType.CODE_EXECUTION},
        )
        check(
            "no_actuation_receipt_for_source_method",
            not {
                ReceiptType.STATIC_ACTUATION,
                ReceiptType.RUNTIME_ACTUATION,
                ReceiptType.COUNTERFACTUAL_ACTUATION,
            }
            & {receipt.receipt_type for receipt in selected_receipts},
        )

        paths = [
            *base.read_jsonl("authority/paths.jsonl"),
            *base.read_jsonl("authority/replay_paths.jsonl"),
        ]
        exact_paths = [
            row
            for row in paths
            if row.get("claim_id") == formal_claim_id
            and set(map(str, row.get("receipt_ids") or [])) == formal_receipt_ids
        ]
        check("one_exact_evidence_path", len(exact_paths) == 1)
        check("formal_path_binding", bool(exact_paths) and exact_paths[0].get("path_id") == formal_path_id)

        derivations = [
            row
            for row in base.read_jsonl("authority/derivations.jsonl")
            if row.get("claim_id") == formal_claim_id
        ]
        check("formal_derivation_pair", len(derivations) == 2)
        for derivation in derivations:
            derivation_id = str(derivation.get("derivation_id") or "")
            check(
                f"derivation_internal_hash:{derivation_id}",
                derivation.get("derivation_hash")
                == payload_hash(derivation, "derivation_hash"),
            )
            check(
                f"derivation_source_score_false:{derivation_id}",
                derivation.get("source_score_inheritance") is False,
            )
            check(
                f"derivation_historical_metric_false:{derivation_id}",
                derivation.get("historical_metric_used_as_evidence") is False,
            )
            check(
                f"derivation_issue_set:{derivation_id}",
                derivation.get("allowed_protocol_issue_codes") == expected_issues,
            )

        with tempfile.TemporaryDirectory(prefix="wp8-formal-child-overlay-") as overlay:
            snapshot = loader.load(
                session_overlay_path=overlay,
                active_protocol_ref=protocol_ref,
                authority_policy_version="authority_v1",
                verify_artifacts=True,
            )
            engine = AuthorityEngine(registry, policy_version="authority_v1")
            authority_load = load_snapshot_authority(engine, snapshot)
            check("formal_claim_loaded", formal_claim_id in authority_load.get("claim_ids", []))
            fields = {field.name for field in dataclasses.fields(SOPClauseV1)}
            clause_value = SOPClauseV1(
                **{key: value for key, value in formal.items() if key in fields}
            )
            for stage in sorted(GENERATION_STAGES):
                request = VisibilityRequest(
                    operation=Operation.GENERATE_CANDIDATE,
                    generation_stage=GenerationStage(stage),
                    governance_stage=GovernanceStage.RETRIEVAL,
                    active_protocol=active_protocol,
                    task_context=TaskContext(
                        task_id=target_task_id,
                        task_family=target_task_family,
                    ),
                    memory_bundle_version=base.bundle_version,
                    token_budget=4096,
                    requesting_component="independent_formal_child_verifier",
                )
                decision = authorize_clause_for_visibility(
                    clause_value,
                    request,
                    authority_engine=engine,
                )
                check(f"generate_allowed:{stage}", decision.allowed is True)
                visible = snapshot.base_clauses(
                    Operation.GENERATE_CANDIDATE,
                    task_id=target_task_id,
                    task_family=target_task_family,
                    generation_stage=stage,
                    governance_stage=GovernanceStage.RETRIEVAL.value,
                )
                check(
                    f"visible_exact:{stage}",
                    {str(row.get("clause_id") or "") for row in visible}
                    == {formal_clause_id},
                )
            debug_clause_value = SOPClauseV1(
                **{
                    key: value
                    for key, value in formal_debug.items()
                    if key in fields
                }
            )
            debug_request = VisibilityRequest(
                operation=Operation.DEBUG_HYPOTHESIS,
                generation_stage=GenerationStage.DEBUG,
                governance_stage=GovernanceStage.RETRIEVAL,
                active_protocol=active_protocol,
                task_context=TaskContext(
                    task_id=target_task_id,
                    task_family=target_task_family,
                ),
                memory_bundle_version=base.bundle_version,
                token_budget=4096,
                requesting_component="independent_formal_child_verifier",
            )
            debug_decision = authorize_clause_for_visibility(
                debug_clause_value,
                debug_request,
                authority_engine=engine,
            )
            check("debug_hypothesis_allowed", debug_decision.allowed is True)
            debug_visible = snapshot.base_clauses(
                Operation.DEBUG_HYPOTHESIS,
                task_id=target_task_id,
                task_family=target_task_family,
                generation_stage=GenerationStage.DEBUG.value,
                governance_stage=GovernanceStage.RETRIEVAL.value,
            )
            check(
                "debug_visible_exact",
                {str(row.get("clause_id") or "") for row in debug_visible}
                == {formal_debug_clause_id},
            )

            denied = (
                (Operation.RANK, GovernanceStage.BRANCH_SELECTION),
                (Operation.SELECT, GovernanceStage.BRANCH_SELECTION),
                (Operation.PROMOTE_RESULT, GovernanceStage.MEMORY_WRITEBACK),
                (Operation.PUBLISH_ADOPTION, GovernanceStage.MEMORY_WRITEBACK),
                (Operation.PUBLISH_CAUSAL, GovernanceStage.MEMORY_WRITEBACK),
                (Operation.DISTILL_POSITIVE_RESULT, GovernanceStage.DISTILLATION),
                (Operation.CODE_SEED, GovernanceStage.RETRIEVAL),
            )
            for operation, governance_stage in denied:
                request = VisibilityRequest(
                    operation=operation,
                    generation_stage=GenerationStage.IMPROVE,
                    governance_stage=governance_stage,
                    active_protocol=active_protocol,
                    task_context=TaskContext(
                        task_id=target_task_id,
                        task_family=target_task_family,
                    ),
                    memory_bundle_version=base.bundle_version,
                    token_budget=4096,
                    requesting_component="independent_formal_child_verifier",
                )
                decision = authorize_clause_for_visibility(
                    clause_value,
                    request,
                    authority_engine=engine,
                )
                check(f"operation_denied:{operation.value}", decision.allowed is False)

            wrong_domain_request = VisibilityRequest(
                operation=Operation.GENERATE_CANDIDATE,
                generation_stage=GenerationStage.DRAFT,
                governance_stage=GovernanceStage.RETRIEVAL,
                active_protocol=active_protocol,
                task_context=TaskContext(
                    task_id=f"{target_task_id}-wrong-domain",
                    task_family="nlp_classification",
                ),
                memory_bundle_version=base.bundle_version,
                token_budget=4096,
                requesting_component="independent_formal_child_verifier",
            )
            wrong_domain = authorize_clause_for_visibility(
                clause_value,
                wrong_domain_request,
                authority_engine=engine,
            )
            check("cross_domain_generation_denied", wrong_domain.allowed is False)
            snapshot.assert_unchanged()
    except Exception as error:
        check(f"exception:{type(error).__name__}:{error}", False)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "publication_root": str(publication_root),
        "target_task_id": target_task_id,
        "target_domain": target_domain,
        "protocol_ref": protocol_ref,
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": sorted(set(errors)),
        "verified": not errors,
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    report["verification_hash"] = payload_hash(report, "verification_hash")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--expected-parent-bundle-id", required=True)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--target-task-id", required=True)
    parser.add_argument("--target-task-family", required=True)
    parser.add_argument("--target-domain", required=True)
    parser.add_argument("--split-mode", required=True)
    parser.add_argument("--source-clause-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-node-id", required=True)
    parser.add_argument("--publication-class", required=True)
    parser.add_argument("--protocol-ref", required=True)
    parser.add_argument(
        "--expected-parent-validation-mode",
        choices=["strict_current_schema", "legacy_lineage_migration"],
        default="strict_current_schema",
    )
    parser.add_argument("--allow-source-audit-issue-code", action="append")
    parser.add_argument("--agent-seed", type=int, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_formal_child_publication(
        args.publication_root,
        expected_parent_bundle_id=args.expected_parent_bundle_id,
        expected_parent_manifest_sha256=args.expected_parent_manifest_sha256,
        target_task_id=args.target_task_id,
        target_task_family=args.target_task_family,
        target_domain=args.target_domain,
        split_mode=args.split_mode,
        source_clause_id=args.source_clause_id,
        source_run_id=args.source_run_id,
        source_node_id=args.source_node_id,
        publication_class=args.publication_class,
        protocol_ref=args.protocol_ref,
        expected_parent_validation_mode=args.expected_parent_validation_mode,
        allowed_protocol_issue_codes=args.allow_source_audit_issue_code or (),
        agent_seeds=args.agent_seed or (104729, 130363, 155921),
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


__all__ = ["SCHEMA", "verify_formal_child_publication"]
