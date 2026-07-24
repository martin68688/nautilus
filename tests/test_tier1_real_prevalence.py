from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from audit_tier1_real_decision_prevalence import (  # noqa: E402
    audit_prevalence,
    load_real_decisions,
)
from verify_tier1_real_decision_prevalence import verify_prevalence  # noqa: E402


CREATED_AT = "2026-07-21T00:00:00+08:00"
PROTOCOL = "test-protocol@1#hash"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> Path:
    root = tmp_path / "wp4"
    traces = root / "traces" / "run-target"
    traces.mkdir(parents=True)
    stages = ["draft", "debug", "improve", "draft"]
    refs = []
    blocks = []
    for index, stage in enumerate(stages, start=1):
        node_ref = f"run::run-target::node::node-{index}"
        refs.append(
            {
                "node_ref": node_ref,
                "node_id": f"node-{index}",
                "stage": stage,
                "audit_status": "clean",
                "step": index,
                "transition_ref": None,
                "audit_sidecar_sha256": "0" * 64,
            }
        )
        blocks.append(
            f"## Turn {index}\n"
            f"- node_ref: `{node_ref}`\n"
            "- transition_ref: `root`\n"
            f"- stage: `{stage}`\n"
            "- buggy: `False`\n"
            "- metric: `None`\n"
            "- audit_status: `clean`\n"
            "- audit_issue_refs: `[]`\n"
            f"- plan: shared feature pipeline decision {index}\n"
            "  with multiline planning context\n"
            "- code_summary: shared feature model pipeline\n"
            "- observation: post-decision text must not enter the query\n"
            "- failure: `None`\n"
        )
    trace_path = traces / "branch.md"
    trace_path.write_text("# Trace\n\n" + "\n".join(blocks), encoding="utf-8")
    trace_manifest = {
        "schema": "test",
        "manifest_sha256": "1" * 64,
        "traces": [
            {
                "path": "run-target/branch.md",
                "sha256": _sha(trace_path),
                "run_id": "run-target",
                "task_id": "task-a",
                "branch_id": "1",
                "refs": refs,
            }
        ],
    }
    (root / "traces" / "trace_manifest.json").write_text(
        json.dumps(trace_manifest), encoding="utf-8"
    )
    corpus = {
        "manifest_sha256": "2" * 64,
        "runs": [
            {
                "status": "complete",
                "canonical_task_id": "task-a",
                "task_family": "image",
            },
            {
                "status": "complete",
                "canonical_task_id": "task-b",
                "task_family": "image",
            },
            {
                "status": "complete",
                "canonical_task_id": "task-c",
                "task_family": "nlp",
            },
        ],
    }
    (root / "corpus_manifest.json").write_text(json.dumps(corpus), encoding="utf-8")
    (root / "audit_report.json").write_text(
        json.dumps(
            {
                "expected_code_node_count": 4,
                "active_protocol_ref": PROTOCOL,
            }
        ),
        encoding="utf-8",
    )

    def clause(
        clause_id: str,
        task: str,
        stages: list[str],
        operations: list[str],
    ) -> dict:
        return {
            "clause_id": clause_id,
            "text": "shared feature model pipeline",
            "retrieval_text": "shared feature pipeline method",
            "applies_when": ["shared pipeline"],
            "task_scope": {"task_ids": [task]},
            "source_artifact_refs": [f"run::run-{task}::node::source"],
            "permitted_generation_stages": stages,
            "permitted_operations": operations,
            "protocol_agnostic": False,
            "protocol_scope": [PROTOCOL],
            "publication_class": "candidate",
            "claim_types": ["method_hypothesis"],
        }

    clauses = [
        clause(
            "clause::valid-draft",
            "task-b",
            ["draft"],
            ["generate_candidate"],
        ),
        clause("clause::debug-only", "task-b", ["debug"], ["repair_seed"]),
        clause("clause::invalid-score", "task-b", ["draft", "improve"], ["rank"]),
        clause(
            "clause::target-history",
            "task-a",
            ["draft"],
            ["generate_candidate"],
        ),
        clause(
            "clause::cross-domain",
            "task-c",
            ["draft"],
            ["generate_candidate"],
        ),
    ]
    binder = root / "binder"
    binder.mkdir()
    (binder / "clauses.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in clauses),
        encoding="utf-8",
    )
    return root


def test_trace_parser_uses_plan_and_code_summary_but_not_observation(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    decisions, hashes = load_real_decisions(
        root / "traces" / "trace_manifest.json",
        root / "traces",
    )

    assert len(decisions) == 4
    assert len(hashes) == 1
    assert "multiline planning context" in decisions[0]["query_text"]
    assert "post-decision text" not in decisions[0]["query_text"]
    assert {row["stage"] for row in decisions} == {"draft", "debug", "improve"}


def test_real_prevalence_audit_enforces_same_domain_different_task_scope(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    output = tmp_path / "prevalence"
    report = audit_prevalence(
        root,
        output,
        created_at=CREATED_AT,
        top_k=3,
        gate_thresholds={
            "minimum_eligible_decisions": 1,
            "minimum_covered_domains": 1,
            "minimum_covered_stages": 3,
            "minimum_top5_any_mismatch_wilson_lower_95": 0.0,
            "minimum_top1_mismatch_wilson_lower_95": 0.0,
        },
    )

    assert report["real_code_node_count"] == 4
    assert report["eligible_decision_count"] == 4
    assert report["covered_domain_count"] == 1
    assert report["covered_stage_count"] == 3
    assert report["transfer_scope"]["target_history_exposure_count"] == 0
    assert report["transfer_scope"]["cross_domain_exposure_count"] == 0
    assert report["overall"]["top_k_any_mismatch_count"] == 4
    assert report["gate_1"]["passed"] is True
    receipts = [
        json.loads(line)
        for line in (output / "retrieval_receipts.jsonl").read_text().splitlines()
        if line
    ]
    assert all(
        candidate["source_task_ids"] == ["task-b"]
        and candidate["different_task"] is True
        and candidate["same_domain"] is True
        for row in receipts
        for candidate in row["top_k"]
    )
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        audit_prevalence(root, output, created_at=CREATED_AT)


def test_real_prevalence_verifier_recomputes_accepted_evidence(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    output = tmp_path / "prevalence"
    audit_prevalence(root, output, created_at=CREATED_AT)

    verification = verify_prevalence(root, output)

    assert verification["verified"] is True
    assert verification["errors"] == []
    assert verification["eligible_decision_count"] == 4
    assert verification["candidate_exposure_count"] > 0
    # The fixture is intentionally smaller than the preregistered real gate floor.
    assert verification["gate_1_passed"] is False


def test_real_prevalence_verifier_rejects_tampered_receipts(
    tmp_path: Path,
) -> None:
    root = _fixture(tmp_path)
    output = tmp_path / "prevalence"
    audit_prevalence(root, output, created_at=CREATED_AT)
    receipts_path = output / "retrieval_receipts.jsonl"
    rows = [
        json.loads(line)
        for line in receipts_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows[0]["target_history_exposure_count"] = 1
    receipts_path.chmod(0o644)
    receipts_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    verification = verify_prevalence(root, output)

    assert verification["verified"] is False
    assert "retrieval_receipts_file_hash" in verification["errors"]
    assert any(
        error.startswith("receipt_hash:") for error in verification["errors"]
    )
    assert any(
        error.startswith("target_history_exposure:")
        for error in verification["errors"]
    )
