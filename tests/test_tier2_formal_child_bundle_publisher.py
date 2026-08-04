from __future__ import annotations

import json
import shutil
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from authority.memory_snapshot import ImmutableBaseBundle  # noqa: E402
from authority.protocol_registry import ProtocolRegistry  # noqa: E402
from bind_sop_clauses import read_jsonl, write_jsonl  # noqa: E402
from publish_tier2_formal_child_bundle import (  # noqa: E402
    _parent_validation_disposition,
    publish_formal_child_bundle,
)
from runforest_v2 import journal_parent_links  # noqa: E402
from schema import read_json, sha256_json, write_json_atomic  # noqa: E402
from validate_memory_bundle import validate_bundle  # noqa: E402
from verify_tier2_formal_child_bundle import (  # noqa: E402
    verify_formal_child_publication,
)
from tests import memory_bundle_helpers as helpers  # noqa: E402


CREATED_AT = "2026-07-22T00:00:00Z"
SOURCE_CLAUSE_ID = "clause::formal-provisional-source-test"
SOURCE_SOP_ID = "sop::formal-provisional-source-test"
PROTOCOL_ISSUE = "TEMPORAL_SPLIT_LEAKAGE"


def test_formal_publisher_topology_helper_supports_modern_journals_fail_closed() -> None:
    nodes = [("root", {}), ("child", {"parent_id": "root"})]
    assert journal_parent_links(
        {"node2parent": {"child": "root"}}, nodes
    ) == {"child": "root"}
    with pytest.raises(ValueError, match="disagrees"):
        journal_parent_links(
            {"node2parent": {"child": "other"}},
            [*nodes, ("other", {})],
        )
    with pytest.raises(ValueError, match="cycle"):
        journal_parent_links(
            {"node2parent": {"root": "child", "child": "root"}},
            [("root", {}), ("child", {})],
        )


def _add_execution_evidence(run_dir: Path) -> None:
    for name in ("journal.json", "filtered_journal.json"):
        path = run_dir / "logs" / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for index, node in enumerate(payload.get("nodes") or []):
            node["exec_time"] = 1.25 + index
            node["_term_out"] = [
                f"Execution time: {1.25 + index:.2f} seconds\n",
                "Submission saved to /tmp/submission.csv\n",
            ]
        path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )


def _prepare_parent_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    original_write_run = helpers.write_run

    def write_run_with_execution(*args, **kwargs):
        run_dir = original_write_run(*args, **kwargs)
        _add_execution_evidence(run_dir)
        return run_dir

    monkeypatch.setattr(helpers, "write_run", write_run_with_execution)
    corpus = helpers.prepare_corpus(tmp_path)
    audit_splits = helpers.prepare_audit_and_splits(tmp_path, corpus)
    sops = helpers.prepare_sops(tmp_path, corpus, audit_splits)

    parent_protocol_dir = tmp_path / "protocols"
    parent_protocol_dir.mkdir()
    parent_protocol_payload = {
        "protocol_id": "test-protocol",
        "version": "1",
        "task_profile": {},
        "data_split_policy": {},
        "preprocessing_policy": {},
        "evaluator_spec": {},
        "metric_spec": {},
        "selection_policy": {},
        "seed_policy": {},
        "holdout_policy": {},
        "promotion_policy": {},
        "compatibility_rules": {},
    }
    write_json_atomic(
        parent_protocol_dir / "test-protocol.json",
        parent_protocol_payload,
    )
    parent_protocol_ref = ProtocolRegistry(parent_protocol_dir).resolve(
        "test-protocol@1"
    ).ref().key()

    split = audit_splits["splits"]["seed-heldout"]
    source_run_id = next(
        run_id
        for run_id in split.source_run_ids
        if run_id.startswith("run-task-a-")
    )
    source_node_id = "n0"
    source_artifact_id = f"run::{source_run_id}::node::{source_node_id}"

    audit_index = read_json(audit_splits["audit_dir"] / "index.json")
    sidecar_name = audit_index["entries"][source_artifact_id]
    sidecar_path = audit_splits["audit_dir"] / sidecar_name
    sidecar = read_json(sidecar_path)
    sidecar["status"] = "blocked"
    sidecar["issues"] = [
        {
            "category": "split_contamination",
            "detector": "synthetic-formal-publisher-test",
            "evidence": "Historical source used a randomized split despite order metadata.",
            "execution_disposition": "block",
            "issue_code": PROTOCOL_ISSUE,
            "line": 1,
            "remediation": "Use the frozen grouped target protocol.",
            "severity": "high",
        }
    ]
    sidecar["sidecar_sha256"] = sha256_json(
        {
            key: value
            for key, value in sidecar.items()
            if key != "sidecar_sha256"
        }
    )
    write_json_atomic(sidecar_path, sidecar)

    clauses_path = sops["merge_dir"] / "clauses.jsonl"
    clauses = read_jsonl(clauses_path)
    for clause in clauses:
        clause["protocol_scope"] = [parent_protocol_ref]
    clauses.append(
        {
            "clause_id": SOURCE_CLAUSE_ID,
            "sop_id": SOURCE_SOP_ID,
            "text": "Use a convolutional encoder with global pooling and a linear head.",
            "retrieval_text": "Use a convolutional encoder with global pooling and a linear head.",
            "claim_refs": [],
            "claim_types": ["method_hypothesis"],
            "source_artifact_refs": [source_artifact_id],
            "source_transition_refs": [],
            "task_scope": {"task_ids": ["task-a"]},
            "permitted_operations": [
                "inspect",
                "generate_candidate",
                "distill_candidate",
            ],
            "permitted_generation_stages": [
                "draft",
                "model_design",
                "improve",
                "evolution",
                "fusion",
            ],
            "permitted_governance_stages": ["retrieval"],
            "publication_class": "candidate",
            "authority_decision_refs": [],
            "receipt_refs": [],
            "derivation_refs": [],
            "protocol_agnostic": False,
            "protocol_scope": [parent_protocol_ref],
            "legacy_status": "native_v1",
            "applies_when": ["designing an image classifier"],
            "prevents": ["unnecessary spatial parameters"],
        }
    )
    write_jsonl(
        clauses_path,
        sorted(clauses, key=lambda row: str(row["clause_id"])),
    )

    containers_path = sops["merge_dir"] / "containers.json"
    containers = read_json(containers_path)
    containers["containers"].append(
        {
            "sop_id": SOURCE_SOP_ID,
            "title": "Formal provisional source test",
            "task_id": "task-a",
            "clause_ids": [SOURCE_CLAUSE_ID],
        }
    )
    containers["containers"].sort(key=lambda row: str(row["sop_id"]))
    write_json_atomic(containers_path, containers)

    claims_path = sops["binder_dir"] / "claims.jsonl"
    claims = read_jsonl(claims_path)
    for claim in claims:
        claim["protocol_ref"] = parent_protocol_ref
    write_jsonl(claims_path, claims)

    final = helpers.prepare_runforest_and_bundle(
        tmp_path,
        corpus,
        audit_splits,
        sops,
        split_name="seed-heldout",
    )
    return {
        **corpus,
        **audit_splits,
        **sops,
        **final,
        "source_run_id": source_run_id,
        "source_node_id": source_node_id,
        "source_artifact_id": source_artifact_id,
    }


def _write_target_protocol(tmp_path: Path) -> Path:
    path = tmp_path / "formal-target-protocol-v1.json"
    write_json_atomic(
        path,
        {
            "protocol_id": "formal-target-protocol",
            "version": "1",
            "task_profile": {},
            "data_split_policy": {},
            "preprocessing_policy": {},
            "evaluator_spec": {},
            "metric_spec": {},
            "selection_policy": {},
            "seed_policy": {},
            "holdout_policy": {},
            "promotion_policy": {},
            "compatibility_rules": {},
        },
    )
    return path


def _publish(
    parent: dict,
    publication_root: Path,
    protocol_file: Path,
    *,
    allowed_issues: tuple[str, ...],
    split_mode: str = "seed-heldout",
) -> dict:
    base = ImmutableBaseBundle.load(parent["bundle_dir"], verify_artifacts=True)
    return publish_formal_child_bundle(
        parent_bundle=base.path,
        expected_parent_manifest_sha256=base.manifest_sha256,
        publication_root=publication_root,
        bundle_id="formal-test-child",
        bundle_version="v1",
        target_task_id="task-a",
        target_task_family="family-a",
        target_domain="family_a",
        split_mode=split_mode,
        source_clause_id=SOURCE_CLAUSE_ID,
        source_run_id=parent["source_run_id"],
        source_node_id=parent["source_node_id"],
        protocol_file=protocol_file,
        publication_class="provisional",
        agent_seeds=(104729, 130363, 155921),
        allowed_protocol_issue_codes=allowed_issues,
        created_at=CREATED_AT,
    )


def test_same_domain_seed_child_keeps_target_history_and_other_domain_peers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _prepare_parent_bundle(tmp_path, monkeypatch)
    result = _publish(
        parent,
        tmp_path / "same-domain-seed-child",
        _write_target_protocol(tmp_path),
        allowed_issues=(PROTOCOL_ISSUE,),
        split_mode="same-domain-seed-heldout",
    )
    child = ImmutableBaseBundle.load(
        result["publication"]["bundle_path"], verify_artifacts=True
    )
    split = child.read_json("splits/active.json")
    assert split["split_kind"] == "same-domain-seed-heldout"
    assert set(split["source_task_ids"]) == {"task-a", "task-b"}
    assert split["validation"]["target_history_in_source_count"] > 0
    assert split["validation"]["cross_domain_source_run_count"] == 0
    assert set(split["source_seed_groups"]).isdisjoint(
        split["heldout_seed_groups"]
    )


def test_formal_child_publisher_is_fail_closed_then_publishes_exact_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _prepare_parent_bundle(tmp_path, monkeypatch)
    protocol = _write_target_protocol(tmp_path)
    parent_base_before_removal = ImmutableBaseBundle.load(
        parent["bundle_dir"], verify_artifacts=True
    )
    parent_validation = validate_bundle(parent_base_before_removal.path)
    legacy_validation = dict(parent_validation)
    legacy_validation["valid"] = False
    legacy_validation["errors"] = [
        "runforest_domain_scope_not_required",
        "container_domain_scope_incomplete:sop::legacy",
        "clause_source_domain_lineage:clause::legacy",
        "clause_missing_transfer_scope:clause::legacy",
    ]
    legacy = _parent_validation_disposition(
        parent_base_before_removal, legacy_validation
    )
    assert legacy["accepted"] is True
    assert legacy["mode"] == "legacy_lineage_migration"
    unsafe_validation = dict(legacy_validation)
    unsafe_validation["errors"] = [
        *legacy_validation["errors"],
        "artifact_hash_mismatch:raw_journals/run/journal.json",
    ]
    unsafe = _parent_validation_disposition(
        parent_base_before_removal, unsafe_validation
    )
    assert unsafe["accepted"] is False
    assert unsafe["mode"] == "rejected"
    # A formal child must be derivable from the immutable parent Bundle even
    # after the historical corpus root has disappeared on a later devpod.
    shutil.rmtree(parent["runs_root"])
    assert not parent["runs_root"].exists()

    failed_root = tmp_path / "failed-formal-child"
    with pytest.raises(ValueError, match="non-protocol audit blockers"):
        _publish(parent, failed_root, protocol, allowed_issues=())
    assert (failed_root / "reports" / "publication_failure.json").is_file()
    assert not (failed_root / "CURRENT.json").exists()

    widened_root = tmp_path / "widened-formal-child"
    with pytest.raises(ValueError, match="unsupported audit exceptions"):
        _publish(
            parent,
            widened_root,
            protocol,
            allowed_issues=(PROTOCOL_ISSUE, "TEST_LABEL_ACCESS"),
        )
    assert (widened_root / "reports" / "publication_failure.json").is_file()
    assert not (widened_root / "CURRENT.json").exists()

    publication_root = tmp_path / "published-formal-child"
    result = _publish(
        parent,
        publication_root,
        protocol,
        allowed_issues=(PROTOCOL_ISSUE,),
    )
    assert result["verification"]["valid"] is True
    assert result["verification"]["errors"] == []
    assert result["publication"]["source_score_inheritance"] is False
    assert result["publication"]["historical_metric_used_as_evidence"] is False
    assert result["publication"]["allowed_protocol_issue_codes"] == [
        PROTOCOL_ISSUE
    ]
    assert result["publication"]["parent_validation_disposition"]["mode"] == (
        "strict_current_schema"
    )

    parent_base = ImmutableBaseBundle.load(
        parent["bundle_dir"], verify_artifacts=True
    )
    independent = verify_formal_child_publication(
        publication_root,
        expected_parent_bundle_id=parent_base.bundle_id,
        expected_parent_manifest_sha256=parent_base.manifest_sha256,
        target_task_id="task-a",
        target_task_family="family-a",
        target_domain="family_a",
        split_mode="seed-heldout",
        source_clause_id=SOURCE_CLAUSE_ID,
        source_run_id=parent["source_run_id"],
        source_node_id=parent["source_node_id"],
        publication_class="provisional",
        protocol_ref=result["publication"]["formal_protocol_ref"],
        allowed_protocol_issue_codes=(PROTOCOL_ISSUE,),
        agent_seeds=(104729, 130363, 155921),
    )
    assert independent["verified"] is True, independent["errors"]
    assert independent["check_count"] == independent["passed_check_count"]

    current = read_json(publication_root / "CURRENT.json")
    child_path = publication_root / current["bundle_path"]
    child = ImmutableBaseBundle.load(child_path, verify_artifacts=True)
    validation = validate_bundle(child.path)
    assert validation["valid"] is True
    provenance = child.verify_run_identity_provenance()
    assert provenance["source_membership_verified"] is True
    assert provenance["leak_verified"] is True
    assert provenance["certification_level"] == "formal_domain_provisional"
    assert child.manifest_sha256 == result["publication"][
        "bundle_manifest_sha256"
    ]
    assert not (child.path.stat().st_mode & stat.S_IWUSR)
    assert not ((publication_root / "CURRENT.json").stat().st_mode & stat.S_IWUSR)
    assert not (publication_root.stat().st_mode & stat.S_IWUSR)
    assert not (
        (publication_root / "reports" / "publication_report.json").stat().st_mode
        & stat.S_IWUSR
    )

    formal_clause_id = result["publication"]["formal_clause_id"]
    formal_claim_id = result["publication"]["formal_claim_id"]
    clauses = {
        row["clause_id"]: row for row in child.read_jsonl("sop/clauses.jsonl")
    }
    formal_clause = clauses[formal_clause_id]
    assert formal_clause["claim_refs"] == [formal_claim_id]
    assert formal_clause["permitted_operations"] == ["generate_candidate"]
    assert formal_clause["publication_class"] == "provisional"
    assert formal_clause["contract_spec"]["source_score_inheritance"] is False

    claims = {
        row["claim_id"]: row
        for row in child.read_jsonl("authority/claims.jsonl")
    }
    claim = claims[formal_claim_id]
    assert claim["claim_type"] == "method_hypothesis"
    assert claim["boundary"]["source_audit_status"] == "blocked"
    assert claim["boundary"]["source_audit_issue_codes"] == [PROTOCOL_ISSUE]
    assert claim["boundary"]["allowed_protocol_issue_codes"] == [
        PROTOCOL_ISSUE
    ]

    receipt_ids = set(result["publication"]["formal_receipt_ids"])
    receipts = {
        row["receipt_id"]: row
        for row in child.read_jsonl("authority/receipts.jsonl")
    }
    assert receipt_ids <= set(receipts)
    assert {receipts[value]["receipt_type"] for value in receipt_ids} == {
        "method_identity",
        "code_execution",
    }
    paths = child.read_jsonl("authority/paths.jsonl")
    assert any(
        row["claim_id"] == formal_claim_id
        and set(row["receipt_ids"]) == receipt_ids
        for row in paths
    )

    with pytest.raises(FileExistsError, match="already exists"):
        _publish(
            parent,
            publication_root,
            protocol,
            allowed_issues=(PROTOCOL_ISSUE,),
        )
