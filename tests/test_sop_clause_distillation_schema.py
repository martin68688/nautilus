from __future__ import annotations

import json

import pytest

from tests.memory_bundle_helpers import (
    prepare_audit_and_splits,
    prepare_corpus,
    prepare_sops,
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_frozen_distillation_binder_and_container_merge_preserve_clause_authority(
    tmp_path,
) -> None:
    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    sops = prepare_sops(tmp_path, corpus, audit_splits)

    assert sops["distillation_report"]["network_allowed"] is False
    assert sops["distillation_report"]["all_responses_frozen_or_saved"] is True
    assert sops["distillation_report"]["parse_report"]["status"] == "ok"
    assert sops["distillation_report"]["retry_count"] == 0
    assert (sops["distill_dir"] / "frozen_responses.json").is_file()
    assert len(list((sops["distill_dir"] / "raw_responses").glob("*.json"))) == 12
    assert sops["binder_report"]["quarantine_count"] == 0
    assert sops["binder_report"]["all_clause_sources_resolve"] is True
    assert sops["binder_report"]["scope_widened_count"] == 0
    assert sops["binder_report"]["publication_class_upgrade_count"] == 0
    clauses = _read_jsonl(sops["binder_dir"] / "clauses.jsonl")
    assert len(clauses) == 36
    repair_and_audit = [
        clause
        for clause in clauses
        if clause["claim_types"][0] in {"debug_repair", "audit_finding"}
    ]
    score = [clause for clause in clauses if clause["claim_types"] == ["score"]]
    assert all("0.92" not in clause["retrieval_text"] for clause in repair_and_audit)
    assert all(clause["publication_class"] == "diagnostic" for clause in score)
    assert all(
        clause["publication_class_proposal"] == "certified" for clause in score
    )
    assert all(clause["permitted_operations"] == ["inspect"] for clause in score)
    assert all("0.92" in clause["retrieval_text"] for clause in score)
    assert all(clause["source_artifact_refs"] for clause in clauses)
    assert all(clause["source_transition_refs"] for clause in clauses)

    merge_report = sops["merge_report"]
    assert merge_report["clause_authority_changed"] is False
    assert merge_report["clause_payload_sha256_before"] == merge_report[
        "clause_payload_sha256_after"
    ]


def test_distillation_fails_closed_when_frozen_cache_is_incomplete(tmp_path) -> None:
    from distill_sop_clauses import distill

    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    from extract_branches import extract_branches

    traces = tmp_path / "traces"
    extract_branches(
        corpus["manifest_path"],
        traces,
        split_manifest_path=audit_splits["split_paths"]["full"],
        audit_dir=audit_splits["audit_dir"],
        created_at="2026-07-19T00:00:00Z",
    )
    empty_cache = tmp_path / "empty.json"
    empty_cache.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cache is incomplete"):
        distill(
            traces / "trace_manifest.json",
            traces,
            tmp_path / "distill-fail",
            frozen_responses_path=empty_cache,
            model="deepseek-test",
            allow_network=False,
        )


def test_network_distillation_resumes_saved_responses_after_failure(
    tmp_path, monkeypatch
) -> None:
    import distill_sop_clauses as module
    from extract_branches import extract_branches

    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    traces = tmp_path / "traces"
    trace_manifest = extract_branches(
        corpus["manifest_path"],
        traces,
        split_manifest_path=audit_splits["split_paths"]["full"],
        audit_dir=audit_splits["audit_dir"],
        created_at="2026-07-19T00:00:00Z",
    )
    output = tmp_path / "resumable-distillation"
    first_calls = 0

    def response(_request):
        return {
            "sop_containers": [
                {
                    "title": "resumable clause",
                    "clauses": [{"text": "resume safely"}],
                }
            ]
        }, {"total_tokens": 1}

    def fail_after_first(request):
        nonlocal first_calls
        first_calls += 1
        if first_calls > 1:
            raise TimeoutError("synthetic timeout")
        return response(request)

    monkeypatch.setattr(module, "call_deepseek", fail_after_first)
    with pytest.raises(RuntimeError, match="failed after 3 new attempts"):
        module.distill(
            traces / "trace_manifest.json",
            traces,
            output,
            model="deepseek-test",
            allow_network=True,
        )
    saved = list((output / "raw_responses").glob("*.json"))
    assert len(saved) == 1

    resumed_calls = 0

    def succeed(request):
        nonlocal resumed_calls
        resumed_calls += 1
        return response(request)

    monkeypatch.setattr(module, "call_deepseek", succeed)
    report = module.distill(
        traces / "trace_manifest.json",
        traces,
        output,
        model="deepseek-test",
        allow_network=True,
    )

    assert report["proposal_count"] == trace_manifest["trace_count"]
    assert resumed_calls == trace_manifest["trace_count"] - 1
    assert report["request_reports"][0]["source"] == "saved"


def test_binder_normalizes_scalar_conditions_without_widening_diagnostic_method(
    tmp_path,
) -> None:
    from bind_sop_clauses import bind
    from extract_branches import extract_branches

    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    traces = tmp_path / "traces"
    trace_manifest = extract_branches(
        corpus["manifest_path"],
        traces,
        split_manifest_path=audit_splits["split_paths"]["full"],
        audit_dir=audit_splits["audit_dir"],
        created_at="2026-07-19T00:00:00Z",
    )
    trace = trace_manifest["traces"][0]
    ref = trace["refs"][0]
    proposal = {
        "request_id": "scalar-schema-test",
        "run_id": trace["run_id"],
        "branch_id": trace["branch_id"],
        "task_id": trace["task_id"],
        "response": {
            "sop_containers": [
                {
                    "title": "conservative method",
                    "clauses": [
                        {
                            "text": "Inspect a conservative method hypothesis.",
                            "retrieval_text": "Inspect a conservative method hypothesis.",
                            "claim_type_proposal": "method_hypothesis",
                            "source_refs": [ref["node_ref"]],
                            "evidence_refs": [ref["node_ref"]],
                            "applies_when": "during inspection",
                            "prevents": "unreviewed generation",
                            "publication_class_proposal": "diagnostic",
                        }
                    ],
                }
            ]
        },
    }
    proposals = tmp_path / "proposals.jsonl"
    proposals.write_text(json.dumps(proposal) + "\n", encoding="utf-8")
    output = tmp_path / "binder"

    report = bind(
        proposals,
        traces / "trace_manifest.json",
        output,
        active_protocol_ref="test-protocol@1#hash",
        created_at="2026-07-19T00:00:00Z",
    )
    clause = _read_jsonl(output / "clauses.jsonl")[0]

    assert clause["applies_when"] == ["during inspection"]
    assert clause["prevents"] == ["unreviewed generation"]
    assert clause["publication_class"] == "diagnostic"
    assert clause["permitted_operations"] == ["inspect"]
    assert clause["permitted_generation_stages"] == []
    assert report["schema_normalization_counts"] == {
        "applies_when_scalar_to_list": 1,
        "prevents_scalar_to_list": 1,
    }
    assert report["publication_class_upgrade_count"] == 0
    assert report["scope_widened_count"] == 0
