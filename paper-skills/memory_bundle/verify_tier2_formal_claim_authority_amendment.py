from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from authority.domain_scope import canonical_domain
from authority.memory_snapshot import ImmutableBaseBundle

from build_corpus_manifest import journal_nodes
from method_claim_purity import audit_method_claim_semantic_purity
from verify_tier2_formal_preregistration_amendment import verify_amendment


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "decision_admissibility_wp8_tier2_formal_claim_authority_amendment_v1"
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_claim_authority_amendment_verification_v1"
)
PREREGISTRATION_SOURCE_FAMILIES = {
    "wp8-tier2-formal-3protocol-6system-r3": "audio_multilabel_classification",
    "wp8-tier2-formal-3protocol-6system-r4": "Audio",
}
PARENT_PREREGISTRATION_ID = "wp8-tier2-formal-3protocol-6system-r2"
SOURCE_CLAUSE_ID = "clause::06c4a3d0455368ac50513172"
SOURCE_CLAIM_ID = "claim::00fa8a166a3234c91f8ae9b7"
SOURCE_RUN_ID = "20260716_070743_mlsp-2013-birds"
SOURCE_NODE_ID = "91ffa7572b3d4774bc33d0c000e418e4"
SOURCE_ARTIFACT_ID = f"run::{SOURCE_RUN_ID}::node::{SOURCE_NODE_ID}"
PROTOCOL_ISSUE = "TEMPORAL_SPLIT_LEAKAGE"
PROTOCOL_REF = (
    "grouped-classification@1#"
    "901703060d3f2dd756cc29339a645583410a1d69ac4f2b6ccd53f8631b411382"
)
GENERATION_STAGES = {
    "draft",
    "model_design",
    "improve",
    "evolution",
    "fusion",
}
UNCHANGED_PROHIBITIONS = {
    "no_source_score_inheritance",
    "no_historical_score_for_rank_or_select",
    "no_source_result_promotion",
    "no_positive_result_distillation_from_the_source",
    "no_cross_domain_method_transfer",
    "no_target_terminal_label_exposure",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: object) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _issue_codes(sidecar: Mapping[str, Any]) -> list[str]:
    return sorted(
        {
            str(issue.get("issue_code") or "")
            for issue in sidecar.get("issues") or []
            if isinstance(issue, Mapping) and str(issue.get("issue_code") or "")
        }
    )


def verify_claim_authority_amendment(
    amendment_path: str | Path,
    *,
    repo_root: str | Path = ROOT,
    source_bundle: str | Path | None = None,
    require_source_bundle: bool = True,
) -> dict[str, Any]:
    """Verify the immutable r3 design and, formally, its exact source Bundle.

    The checker deliberately does not call the formal child publisher or its
    verifier.  It independently re-derives the source Clause, code, execution,
    sidecar, and claim-specific permission boundary from the parent Bundle.
    """

    path = Path(amendment_path).resolve()
    repo_root = Path(repo_root).resolve()
    payload = _read(path)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    source_checks: dict[str, bool] = {}

    def check(name: str, condition: Any, *, source: bool = False) -> None:
        value = bool(condition)
        checks[name] = value
        if source:
            source_checks[name] = value
        if not value:
            errors.append(name)

    check("schema", payload.get("schema") == SCHEMA)
    preregistration_id = str(payload.get("preregistration_id") or "")
    expected_source_task_family = PREREGISTRATION_SOURCE_FAMILIES.get(
        preregistration_id, ""
    )
    check(
        "preregistration_id",
        preregistration_id in PREREGISTRATION_SOURCE_FAMILIES,
    )
    check(
        "status_pending_staging",
        payload.get("status") == "design_frozen_pending_staging_hash_manifest",
    )
    check(
        "amendment_hash",
        payload.get("amendment_hash") == _payload_hash(payload, "amendment_hash"),
    )

    parent = payload.get("parent_preregistration") or {}
    parent_path = repo_root / str(parent.get("path") or "")
    check("parent_exists", parent_path.is_file())
    check(
        "parent_file_hash",
        parent_path.is_file() and parent.get("file_sha256") == _sha256_file(parent_path),
    )
    parent_report = (
        verify_amendment(parent_path, repo_root=repo_root)
        if parent_path.is_file()
        else {"verified": False, "preregistration_id": "", "verification_hash": ""}
    )
    check("parent_verifies", parent_report.get("verified") is True)
    check(
        "parent_id",
        parent.get("preregistration_id") == PARENT_PREREGISTRATION_ID
        == parent_report.get("preregistration_id"),
    )

    evidence = payload.get("pretraining_evidence") or {}
    check(
        "no_formal_training_before_revision",
        evidence.get("formal_training_observed_before_revision") is False,
    )
    check(
        "no_terminal_metric_before_revision",
        evidence.get("terminal_metric_observed_before_revision") is False,
    )
    source_identity = evidence.get("source_method") or {}
    expected_identity = {
        "clause_id": SOURCE_CLAUSE_ID,
        "claim_id": SOURCE_CLAIM_ID,
        "source_run_id": SOURCE_RUN_ID,
        "source_node_id": SOURCE_NODE_ID,
        "source_artifact_id": SOURCE_ARTIFACT_ID,
        "source_task_id": "mlsp-2013-birds",
        "source_task_family": expected_source_task_family,
        "source_domain": "audio",
    }
    for key, expected in expected_identity.items():
        check(f"source_identity:{key}", source_identity.get(key) == expected)
    superseded = payload.get("supersedes_failed_revision") or {}
    if preregistration_id.endswith("-r4"):
        superseded_path = repo_root / str(superseded.get("path") or "")
        check("r4_superseded_r3_exists", superseded_path.is_file())
        check(
            "r4_superseded_r3_hash",
            superseded_path.is_file()
            and superseded.get("file_sha256") == _sha256_file(superseded_path),
        )
        check(
            "r4_superseded_r3_id",
            superseded.get("preregistration_id")
            == "wp8-tier2-formal-3protocol-6system-r3",
        )
        failed = superseded.get("failed_source_verification") or {}
        check(
            "r4_failed_report_path_frozen",
            failed.get("cluster_path")
            == "/workspace/decision-admissibility-wp8-tier2-formal-staging-r2/reports/preregistration-r3-claim-authority-verification-r2.json",
        )
        check(
            "r4_failed_report_file_hash_frozen",
            failed.get("file_sha256")
            == "40233864f1f5e38967bf3a3607d456aa5db6a494ef379ac61f760146bafc99ad",
        )
        check(
            "r4_failed_report_hash_frozen",
            failed.get("verification_hash")
            == "fe306643c543c457c91f1191b11823837ede02ec85df5e2071af7f22ba964f1a",
        )
        check(
            "r4_failed_check_exact",
            failed.get("errors") == ["source_task_family_bound"],
        )
        check(
            "r4_only_metadata_correction",
            superseded.get("correction")
            == {
                "field": "pretraining_evidence.source_method.source_task_family",
                "r3_value": "audio_multilabel_classification",
                "r4_value": "Audio",
            },
        )
    method_text = str(source_identity.get("method_text") or "")
    purity = audit_method_claim_semantic_purity(method_text)
    check("method_text_hash", source_identity.get("method_text_sha256") == _sha256_text(method_text))
    check("method_semantic_purity", purity.get("passed") is True)
    check("method_has_no_outcome_assertion", purity.get("source_outcome_assertion_count") == 0)
    execution_checks = source_identity.get("execution_checks") or {}
    check(
        "declared_execution_checks_complete",
        set(execution_checks)
        == {
            "is_valid",
            "not_buggy",
            "no_exception_type",
            "positive_exec_time",
            "terminal_execution_signal",
            "submission_signal",
        }
        and all(value is True for value in execution_checks.values()),
    )

    source_audit = evidence.get("source_audit") or {}
    check("source_audit_blocked", source_audit.get("status") == "blocked")
    check("source_audit_exact_issue", source_audit.get("issue_codes") == [PROTOCOL_ISSUE])
    check(
        "source_audit_issue_retained",
        source_audit.get("issue_disposition")
        == "retained_as_protocol_and_score_blocker",
    )
    check("source_sidecar_not_relabelled", source_audit.get("sidecar_relabelled_clean") is False)

    scope = payload.get("scope") or {}
    check("inherit_parent_except_overrides", scope.get("inherit_all_parent_fields_except_explicit_overrides") is True)
    check(
        "claim_specific_scope",
        scope.get("changed_contract")
        == "memory_bundle_contract.selected_seed_heldout_provisional_methods[mlsp-2013-birds]",
    )
    check("one_changed_claim", scope.get("changed_source_claims") == [SOURCE_CLAIM_ID])
    check("one_changed_run", scope.get("changed_source_runs") == [SOURCE_RUN_ID])
    for key in (
        "systems_seeds_condition_order_budgets_oracle_statistics_or_holdouts_changed",
        "taxi_provisional_rule_changed",
        "aerial_certified_rule_changed",
        "source_global_audit_status_changed",
        "formal_training_observed_before_revision",
        "terminal_metric_observed_before_revision",
    ):
        check(f"scope_false:{key}", scope.get(key) is False)

    overrides = payload.get("overrides") or {}
    contract = overrides.get("memory_bundle_contract") or {}
    birds = contract.get("birds_claim_specific_publication") or {}
    check("publication_is_provisional", birds.get("publication_class") == "provisional")
    check("target_task", birds.get("target_task_id") == "mlsp-2013-birds")
    check("target_family", birds.get("target_task_family") == "audio_multilabel_classification")
    check("target_domain", birds.get("target_domain") == "audio")
    check("active_protocol", birds.get("active_protocol_ref") == PROTOCOL_REF)
    check("method_claim_only", birds.get("claim_type") == "method_hypothesis")
    check("generate_only", birds.get("permitted_operations") == ["generate_candidate"])
    check("generation_stages_exact", set(birds.get("permitted_generation_stages") or []) == GENERATION_STAGES)
    check("retrieval_only", birds.get("permitted_governance_stages") == ["retrieval"])
    check("only_protocol_issue_allowed", birds.get("allowed_source_audit_issue_codes") == [PROTOCOL_ISSUE])
    check("source_score_inheritance_false", birds.get("source_score_inheritance") is False)
    check("historical_metric_not_evidence", birds.get("historical_metric_used_as_evidence") is False)
    check("no_target_result_inference", birds.get("target_result_fact_inferred_from_source") is False)
    check("source_result_not_certified", birds.get("source_result_certified") is False)
    check("source_sidecar_not_clean", birds.get("source_sidecar_clean") is False)
    check(
        "unchanged_prohibitions_exact",
        set(contract.get("unchanged_prohibitions") or []) == UNCHANGED_PROHIBITIONS,
    )
    check(
        "clean_sidecar_obligation_narrowly_replaced",
        contract.get("superseded_parent_obligation_for_birds_only")
        == "source_node_code_hash_matches_clean_audit_sidecar",
    )
    staging_requirement = str(overrides.get("formal_staging_requirement") or "")
    check(
        "staging_binds_revision_chain",
        all(token in staging_requirement for token in ("r3", "r2", "r1")),
    )
    check(
        "staging_binds_source_and_child",
        all(
            token in staging_requirement
            for token in (
                "source journal",
                "source node code hash",
                "blocked audit sidecar",
                "immutable child Bundle",
                "CURRENT pointer",
            )
        ),
    )

    interpretation = payload.get("interpretation") or {}
    check(
        "method_interpretation",
        interpretation.get("method_claim_status")
        == "provisional_candidate_generation_only",
    )
    check(
        "score_interpretation",
        interpretation.get("historical_score_status")
        == "blocked_by_TEMPORAL_SPLIT_LEAKAGE",
    )
    check("source_result_not_certified_interpretation", interpretation.get("historical_result_status") == "not_certified")
    check("global_sidecar_still_blocked", interpretation.get("global_source_sidecar_status") == "blocked_unchanged")
    check("effect_claim_forbidden", interpretation.get("effect_claim_authorized") is False)
    check("training_still_forbidden", interpretation.get("formal_training_authorized") is False)

    source_bundle_path = Path(source_bundle).resolve() if source_bundle else None
    if source_bundle_path is None:
        check("source_bundle_supplied", not require_source_bundle, source=True)
    else:
        check("source_bundle_supplied", source_bundle_path.is_dir(), source=True)
        try:
            base = ImmutableBaseBundle.load(source_bundle_path, verify_artifacts=True)
            declared_bundle = evidence.get("source_bundle") or {}
            check("bundle_id", base.bundle_id == declared_bundle.get("bundle_id"), source=True)
            check("bundle_manifest", base.manifest_sha256 == declared_bundle.get("manifest_sha256"), source=True)
            check("bundle_manifest_file", base.manifest_file_sha256 == declared_bundle.get("manifest_file_sha256"), source=True)

            split = base.read_json("splits/active.json")
            check("source_run_on_memory_side", SOURCE_RUN_ID in set(split.get("source_run_ids") or []), source=True)
            check("source_run_not_heldout", SOURCE_RUN_ID not in set(split.get("heldout_run_ids") or []), source=True)
            corpus = base.read_json("corpus/manifest.json")
            run_rows = [row for row in corpus.get("runs") or [] if str(row.get("run_id") or "") == SOURCE_RUN_ID]
            check("one_source_run", len(run_rows) == 1, source=True)
            run_row = run_rows[0] if len(run_rows) == 1 else {}
            check("source_task_id_bound", run_row.get("canonical_task_id") == "mlsp-2013-birds", source=True)
            check(
                "source_task_family_bound",
                run_row.get("task_family") == expected_source_task_family,
                source=True,
            )
            check("source_domain_bound", canonical_domain(run_row.get("task_family")) == "audio", source=True)

            clauses = base.read_jsonl("sop/clauses.jsonl")
            clause_rows = [row for row in clauses if str(row.get("clause_id") or "") == SOURCE_CLAUSE_ID]
            check("one_source_clause", len(clause_rows) == 1, source=True)
            clause = clause_rows[0] if len(clause_rows) == 1 else {}
            check("source_claim_binding", clause.get("claim_refs") == [SOURCE_CLAIM_ID], source=True)
            check("source_claim_type", clause.get("claim_types") == ["method_hypothesis"], source=True)
            check("source_artifact_binding", SOURCE_ARTIFACT_ID in set(clause.get("source_artifact_refs") or []), source=True)
            check("source_method_text", clause.get("retrieval_text") == method_text, source=True)
            live_purity = audit_method_claim_semantic_purity(str(clause.get("retrieval_text") or ""))
            check("live_method_semantic_purity", live_purity.get("passed") is True, source=True)
            check("live_method_text_hash", live_purity.get("text_sha256") == source_identity.get("method_text_sha256"), source=True)

            journal_relative = f"raw_journals/{SOURCE_RUN_ID}/journal.json"
            check("journal_manifest_bound", journal_relative in (base.manifest.get("artifact_hashes") or {}), source=True)
            journal_path = base.path / journal_relative
            check("journal_hash", _sha256_file(journal_path) == source_identity.get("source_journal_sha256"), source=True)
            journal = base.read_json(journal_relative)
            nodes = [
                node
                for node in journal_nodes(journal)
                if str(node.get("id") or node.get("node_id") or "") == SOURCE_NODE_ID
            ]
            check("one_source_node", len(nodes) == 1, source=True)
            node = nodes[0] if len(nodes) == 1 else {}
            code = str(node.get("code") or "")
            check("node_code_hash", _sha256_text(code) == source_identity.get("code_sha256"), source=True)
            check("node_is_valid", node.get("is_valid") is True, source=True)
            check("node_not_buggy", node.get("is_buggy") is False, source=True)
            check("node_no_exception", node.get("exc_type") in {None, ""}, source=True)
            exec_time = node.get("exec_time")
            check(
                "node_exec_time",
                isinstance(exec_time, (int, float))
                and not isinstance(exec_time, bool)
                and math.isfinite(float(exec_time))
                and math.isclose(
                    float(exec_time),
                    float(source_identity.get("exec_time_seconds") or -1),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ),
                source=True,
            )
            terminal_output = "".join(
                str(value)
                for value in (
                    node.get("_term_out")
                    if isinstance(node.get("_term_out"), list)
                    else [node.get("_term_out") or ""]
                )
            )
            check("terminal_output_hash", _sha256_text(terminal_output) == source_identity.get("terminal_output_sha256"), source=True)
            check("terminal_execution_signal", "Execution time:" in terminal_output, source=True)
            check("terminal_submission_signal", "Submission saved" in terminal_output, source=True)

            graph = base.read_json("runforest/graph.json")
            graph_nodes = {
                str(row.get("id") or ""): row for row in graph.get("nodes") or []
            }
            graph_node = graph_nodes.get(SOURCE_ARTIFACT_ID) or {}
            check("runforest_node_present", bool(graph_node), source=True)
            check("runforest_code_hash", graph_node.get("code_sha256") == source_identity.get("code_sha256"), source=True)

            audit_index = base.read_json("audit_sidecars/index.json")
            audit_filename = (audit_index.get("entries") or {}).get(SOURCE_ARTIFACT_ID)
            check("audit_index_binding", audit_filename == source_audit.get("index_filename"), source=True)
            audit_relative = f"audit_sidecars/{audit_filename}"
            check("audit_manifest_bound", audit_relative in (base.manifest.get("artifact_hashes") or {}), source=True)
            audit_path = base.path / audit_relative
            check("audit_file_hash", _sha256_file(audit_path) == source_audit.get("file_sha256"), source=True)
            sidecar = base.read_json(audit_relative)
            check("audit_internal_hash", sidecar.get("sidecar_sha256") == _payload_hash(sidecar, "sidecar_sha256") == source_audit.get("sidecar_sha256"), source=True)
            check("audit_code_hash", sidecar.get("code_sha256") == source_identity.get("code_sha256"), source=True)
            check("audit_journal_hash", sidecar.get("source_journal_sha256") == source_identity.get("source_journal_sha256"), source=True)
            check("audit_status_blocked", sidecar.get("status") == "blocked", source=True)
            check("audit_issue_exact", _issue_codes(sidecar) == [PROTOCOL_ISSUE], source=True)
            issues = [issue for issue in sidecar.get("issues") or [] if isinstance(issue, Mapping)]
            check(
                "audit_issue_is_blocking_protocol_issue",
                len(issues) == 1
                and issues[0].get("issue_code") == PROTOCOL_ISSUE
                and issues[0].get("execution_disposition") == "block"
                and issues[0].get("category") == "split_contamination",
                source=True,
            )
            base.assert_unchanged()
        except Exception as error:
            check(
                f"source_bundle_exception:{type(error).__name__}:{error}",
                False,
                source=True,
            )

    source_evidence_verified = (
        source_bundle_path is not None
        and bool(source_checks)
        and all(source_checks.values())
    )
    static_check_names = set(checks) - set(source_checks)
    static_verified = bool(static_check_names) and all(
        checks[name] for name in static_check_names
    )
    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "preregistration_id": payload.get("preregistration_id", ""),
        "amendment_file_sha256": _sha256_file(path),
        "parent_verification_hash": parent_report.get("verification_hash", ""),
        "source_bundle_checked": source_bundle_path is not None,
        "source_bundle_path": str(source_bundle_path) if source_bundle_path else "",
        "static_verified": static_verified,
        "source_evidence_verified": source_evidence_verified,
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": sorted(set(errors)),
        "verified": not errors,
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    report["verification_hash"] = _payload_hash(report, "verification_hash")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_claim_authority_amendment(
        args.amendment,
        repo_root=args.repo_root,
        source_bundle=args.source_bundle,
        require_source_bundle=True,
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
    "verify_claim_authority_amendment",
]
