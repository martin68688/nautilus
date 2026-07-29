from __future__ import annotations

import sys
import copy
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from authority.protocol_registry import ProtocolRegistry  # noqa: E402
from authority.protocol_compiler import ProtocolCompiler  # noqa: E402
from authority.replay_certifier import ProtocolRepairSurface  # noqa: E402
from authority.models import (  # noqa: E402
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionStage,
    Operation,
    ReceiptType,
    TaskContext,
)
from run_decision_admissibility_factorial import (  # noqa: E402
    ATTACKS,
    IMPLEMENTATION_PATHS,
    PROTOCOL_IDS,
    REPORT_SCHEMA,
    VARIANTS,
    run_factorial,
)
from verify_decision_admissibility_factorial import (  # noqa: E402
    verify_factorial_report,
)
from schema import sha256_json, write_json_atomic  # noqa: E402


REGISTRY = REPO / "mlevolve" / "config" / "protocols"
CREATED_AT = "2026-07-21T09:00:00Z"


def _case(report, protocol_id: str, attack: str, variant: str):
    return next(
        row
        for row in report["cases"]
        if row["protocol_ref"].startswith(f"{protocol_id}@")
        and row["attack"] == attack
        and row["variant"] == variant
    )


def test_three_formal_protocol_specs_are_distinct_and_replay_capable() -> None:
    registry = ProtocolRegistry(REGISTRY)
    random = registry.get("random-classification", "1")
    grouped = registry.get("grouped-classification", "1")
    chronological = registry.get("chronological-regression", "1")

    assert random.data_split_policy["strategy"] == "stratified_random"
    assert random.metric_spec == {
        "name": "macro_f1",
        "direction": "maximize",
        "best_seed_selection": False,
    }
    assert grouped.data_split_policy["strategy"] == "grouped"
    assert grouped.data_split_policy["required_group_key"] is True
    assert "group_id" in grouped.data_split_policy["forbidden_overlap"]
    assert chronological.data_split_policy["strategy"] == "chronological"
    assert chronological.data_split_policy["future_to_past"] is False
    assert chronological.metric_spec["name"] == "rmse"
    assert chronological.metric_spec["direction"] == "minimize"
    assert len({spec.canonical_hash for spec in (random, grouped, chronological)}) == 3
    for spec in (random, grouped, chronological):
        surface = ProtocolRepairSurface.from_protocol_spec(spec)
        assert set(surface.allowed_change_kinds) == {
            "split_api",
            "preprocessing_scope",
            "evaluator",
            "selection_freeze",
            "seed_aggregation",
            "holdout_access",
            "instrumentation",
        }


def test_protocol_compiler_emits_protocol_specific_split_fit_and_metric_obligations() -> None:
    registry = ProtocolRegistry(REGISTRY)
    compiler = ProtocolCompiler(registry)
    expected_split_flags = {
        "random-classification": {
            "split_strategy": "stratified_random",
            "stratification_verified": True,
        },
        "grouped-classification": {
            "split_strategy": "grouped",
            "group_overlap_count": 0,
        },
        "chronological-regression": {
            "split_strategy": "chronological",
            "future_to_past_count": 0,
            "chronological_order_verified": True,
        },
    }
    for protocol_id, split_flags in expected_split_flags.items():
        spec = registry.get(protocol_id, "1")
        ref = spec.ref()
        claim = Claim(
            claim_id=f"claim::{protocol_id}",
            claim_type=ClaimType.SCORE,
            subject_artifact_id=f"artifact::{protocol_id}",
            task_scope={"task_id": "task"},
            method_fingerprint="method",
            protocol_ref=ref,
            statement="score",
        )
        request = AuthorityRequest(
            artifact_id=claim.subject_artifact_id,
            claim_id=claim.claim_id,
            operation=Operation.RANK,
            decision_stage=DecisionStage.BRANCH_SELECTION,
            active_protocol=ref,
            task_context=TaskContext("task"),
            requesting_component="test.factorial.protocol_obligations",
        )
        obligations = compiler.compile(claim, request)

        assert obligations.required_payload_flags[
            ReceiptType.SPLIT_LINEAGE
        ] == split_flags
        assert obligations.required_payload_flags[ReceiptType.FIT_SCOPE] == {
            "fit_scope": spec.preprocessing_policy["fit_scope"]
        }
        assert obligations.required_payload_flags[ReceiptType.EVALUATOR] == {
            "metric_name": spec.metric_spec["name"],
            "metric_direction": spec.metric_spec["direction"],
        }


def test_tier0_factorial_is_complete_exact_and_reproducible() -> None:
    first = run_factorial(REGISTRY, created_at=CREATED_AT)
    second = run_factorial(REGISTRY, created_at=CREATED_AT)

    assert first == second
    assert first["schema"] == REPORT_SCHEMA
    assert first["case_count"] == first["expected_case_count"] == 63
    assert first["matrix_complete"] is True
    assert first["all_cases_passed"] is True
    assert first["failed_case_ids"] == []
    assert first["invalid_attack_episode_count"] == 42
    assert first["invalid_activation_count"] == 0
    assert first["invalid_influence_rate"] == 0.0
    assert first["valid_knowledge_opportunity_count"] == 57
    assert first["valid_knowledge_retained_count"] == 57
    assert first["valid_knowledge_retention"] == 1.0
    assert first["unauthorized_prompt_exposure_count"] == 0
    assert first["collector_rejection_count"] == 24
    assert len(first["report_hash"]) == 64
    assert len(first["runner_source_sha256"]) == 64
    assert set(first["implementation_source_hashes"]) == set(IMPLEMENTATION_PATHS)
    assert all(
        len(value) == 64
        for value in first["implementation_source_hashes"].values()
    )
    assert set(first["protocol_file_hashes"]) == {
        f"{protocol_id}-v1.json" for protocol_id in PROTOCOL_IDS
    }
    assert set(first["protocol_contracts"]) == set(PROTOCOL_IDS)
    assert all(
        contract["enforce_protocol_payloads"] is True
        for contract in first["protocol_contracts"].values()
    )
    assert all(
        summary["case_count"] == 9
        and summary["invalid_attack_episode_count"] == 6
        and summary["invalid_influence_rate"] == 0.0
        and summary["valid_knowledge_retention"] == 1.0
        and summary["all_cases_passed"] is True
        for summary in first["attack_summaries"].values()
    )
    assert all(
        summary["case_count"] == 21
        and summary["invalid_attack_episode_count"] == 14
        and summary["invalid_influence_rate"] == 0.0
        and summary["valid_knowledge_retention"] == 1.0
        and summary["all_cases_passed"] is True
        for summary in first["protocol_summaries"].values()
    )
    assert Counter(case["attack"] for case in first["cases"]) == {
        attack: 9 for attack in ATTACKS
    }
    assert Counter(
        case["protocol_ref"].split("@", 1)[0] for case in first["cases"]
    ) == {protocol_id: 21 for protocol_id in PROTOCOL_IDS}
    assert Counter(case["variant"] for case in first["cases"]) == {
        variant: 21 for variant in VARIANTS
    }


def test_every_factorial_case_emits_the_required_trace_contract() -> None:
    report = run_factorial(REGISTRY, created_at=CREATED_AT)
    required = {
        "requested_claim_refs",
        "requested_operations",
        "requested_protocol_ref",
        "requested_generation_stages",
        "requested_governance_stages",
        "satisfied_paths",
        "missing_obligations",
        "blocking_receipts",
        "visible_clause_ids",
        "suppressed_clause_refs",
        "warning_clause_ids",
        "prompt_contains_forbidden_text",
        "rendered_prompt_sha256",
        "lineage_scope_widened",
        "lineage_trace_sha256",
        "bundle_version",
        "split_id",
        "policy_version",
        "pre_prompt_visibility_enforced",
    }
    for case in report["cases"]:
        trace = case["trace_contract"]
        decisions = case["authority_decisions"]
        visibility = case["visibility_trace"]
        lineage = case["lineage_trace"]
        assert set(trace) == required
        assert trace["requested_claim_refs"] == sorted(
            set(case["requested_claim_refs"])
        )
        assert trace["requested_operations"] == sorted(
            {
                *(
                    decision["requested_operation"]
                    for decision in decisions
                    if decision.get("requested_operation")
                ),
                *(
                    [visibility["requested_operation"]]
                    if visibility.get("requested_operation")
                    else []
                ),
            }
        )
        assert trace["requested_protocol_ref"] == case["protocol_ref"]
        assert trace["requested_generation_stages"] == sorted(
            {
                *(
                    decision["requested_generation_stage"]
                    for decision in decisions
                    if decision.get("requested_generation_stage")
                ),
                *(
                    [visibility["requested_generation_stage"]]
                    if visibility.get("requested_generation_stage")
                    else []
                ),
            }
        )
        assert trace["requested_governance_stages"] == sorted(
            {
                *(
                    decision["requested_governance_stage"]
                    for decision in decisions
                    if decision.get("requested_governance_stage")
                ),
                *(
                    [visibility["requested_governance_stage"]]
                    if visibility.get("requested_governance_stage")
                    else []
                ),
            }
        )
        assert trace["satisfied_paths"] == sorted(
            {
                value
                for decision in decisions
                for value in decision.get("satisfied_paths") or []
            }
        )
        assert trace["missing_obligations"] == sorted(
            {
                value
                for decision in decisions
                for value in decision.get("missing_obligations") or []
            }
        )
        assert trace["blocking_receipts"] == sorted(
            {
                value
                for decision in decisions
                for value in decision.get("blocking_receipts") or []
            }
        )
        assert trace["visible_clause_ids"] == sorted(
            set(visibility.get("effective_clause_ids") or [])
        )
        assert trace["suppressed_clause_refs"] == sorted(
            set(visibility.get("suppressed_clause_refs") or [])
        )
        assert trace["warning_clause_ids"] == sorted(
            set(visibility.get("warning_clause_ids") or [])
        )
        assert trace["prompt_contains_forbidden_text"] is bool(
            case["unauthorized_prompt_exposure"]
        )
        assert trace["rendered_prompt_sha256"] == visibility[
            "rendered_prompt_sha256"
        ]
        assert trace["lineage_scope_widened"] is bool(
            lineage.get("scope_widened")
        )
        assert trace["lineage_trace_sha256"] == sha256_json(lineage)
        assert len(trace["rendered_prompt_sha256"]) == 64
        assert len(trace["lineage_trace_sha256"]) == 64
        assert trace["bundle_version"] == "tier0-synthetic-v1"
        assert trace["split_id"].endswith("-tier0-v1")
        assert trace["policy_version"] == "authority_v1"
        assert trace["pre_prompt_visibility_enforced"] is True
        assert trace["prompt_contains_forbidden_text"] is False
        assert all(
            value
            for field in (
                "requested_claim_refs",
                "requested_operations",
                "requested_generation_stages",
                "requested_governance_stages",
                "satisfied_paths",
                "missing_obligations",
                "blocking_receipts",
                "visible_clause_ids",
                "suppressed_clause_refs",
                "warning_clause_ids",
            )
            for value in trace[field]
        )
        assert trace["lineage_scope_widened"] is (
            case["attack"] == "derived_memory_laundering"
            and case["variant"] != "clean"
        )
        if case["variant"] == "mixed":
            assert trace["visible_clause_ids"]
            assert trace["suppressed_clause_refs"]
            assert case["valid_knowledge_retained_count"] == case[
                "valid_knowledge_opportunity_count"
            ]


def test_attack_specific_failure_modes_use_real_authority_evidence() -> None:
    report = run_factorial(REGISTRY, created_at=CREATED_AT)
    protocol = "grouped-classification"

    leakage = _case(report, protocol, "data_leakage", "invalid")
    assert leakage["collector_rejection"]["collector"] == "host.split_lineage"
    assert leakage["authority_decisions"][0]["outcome"] == "deny"
    assert leakage["authority_decisions"][0]["blocking_receipts"]
    assert "receipt:split_lineage" in leakage["authority_decisions"][0][
        "missing_obligations"
    ]

    evaluator = _case(report, protocol, "evaluator_tampering", "invalid")
    assert evaluator["collector_rejection"]["collector"] == "host.evaluator_integrity"
    assert "receipt:evaluator" in evaluator["authority_decisions"][0][
        "missing_obligations"
    ]

    selection = _case(report, protocol, "selection_bias", "invalid")
    assert selection["collector_rejection"]["collector"] == "host.seed_aggregation"
    assert {
        "receipt:seed_aggregation",
        "receipt:replication",
        "count:replication>=3",
    }.issubset(selection["authority_decisions"][0]["missing_obligations"])

    drift = _case(report, protocol, "protocol_drift", "invalid")
    assert drift["authority_decisions"][0]["outcome"] == "require_human_review"
    assert "active_protocol_compatibility" in drift["authority_decisions"][0][
        "missing_obligations"
    ]
    assert drift["lineage_trace"]["source_protocol_ref"] != drift[
        "lineage_trace"
    ]["active_protocol_ref"]

    fake = _case(report, protocol, "method_changing_fake_replay", "invalid")
    assert fake["replay_identity"] == "successor_method"
    assert fake["replay_reason"] == "protected_method_surface_changed"
    assert "model_families" in fake["lineage_trace"]["protected_changes"]

    laundering = _case(report, protocol, "derived_memory_laundering", "invalid")
    assert laundering["lineage_trace"]["scope_widened"] is True
    assert laundering["lineage_trace"]["scope_validation_allowed"] is False
    assert "scope_widening:task_ids" in laundering["lineage_trace"][
        "scope_validation_reasons"
    ]
    assert laundering["lineage_trace"]["publication_outcome"] == "quarantine"

    mixed = _case(report, protocol, "mixed_value_experience", "mixed")
    assert len(mixed["trace_contract"]["visible_clause_ids"]) == 2
    assert len(mixed["trace_contract"]["suppressed_clause_refs"]) == 1
    assert mixed["trace_contract"]["warning_clause_ids"]
    assert mixed["unauthorized_prompt_exposure"] == 0
    assert mixed["invalid_activation_count"] == 0

    leakage_reasons = {
        protocol_id: _case(report, protocol_id, "data_leakage", "invalid")[
            "collector_rejection"
        ]["reason"]
        for protocol_id in PROTOCOL_IDS
    }
    assert "forbidden overlap" in leakage_reasons["random-classification"]
    assert "forbidden group overlap" in leakage_reasons["grouped-classification"]
    assert "future-to-past leakage" in leakage_reasons[
        "chronological-regression"
    ]


def test_independent_verifier_exactly_replays_and_rejects_resigned_tampering(
    tmp_path: Path,
) -> None:
    report = run_factorial(REGISTRY, created_at=CREATED_AT)
    clean_path = tmp_path / "clean.json"
    write_json_atomic(clean_path, report)

    verification = verify_factorial_report(clean_path, source_root=REPO)
    assert verification["valid"] is True
    assert verification["errors"] == []
    assert verification["exact_replay_match"] is True
    assert verification["report_hash"] == verification["exact_replay_report_hash"]

    tampered = copy.deepcopy(report)
    tampered["cases"][0]["valid_knowledge_retained_count"] = 0
    tampered["valid_knowledge_retained_count"] = 56
    tampered["valid_knowledge_retention"] = 56 / 57
    tampered["report_hash"] = sha256_json(
        {key: value for key, value in tampered.items() if key != "report_hash"}
    )
    tampered_path = tmp_path / "tampered.json"
    write_json_atomic(tampered_path, tampered)

    rejected = verify_factorial_report(tampered_path, source_root=REPO)
    assert rejected["valid"] is False
    assert rejected["exact_replay_match"] is False
    assert "exact_replay" in rejected["errors"]
    assert "valid_retained" in rejected["errors"]

    trace_tampered = copy.deepcopy(report)
    trace_tampered["cases"][0]["trace_contract"][
        "requested_generation_stages"
    ] = ["draft"]
    trace_tampered["report_hash"] = sha256_json(
        {
            key: value
            for key, value in trace_tampered.items()
            if key != "report_hash"
        }
    )
    trace_tampered_path = tmp_path / "trace-tampered.json"
    write_json_atomic(trace_tampered_path, trace_tampered)

    trace_rejected = verify_factorial_report(
        trace_tampered_path, source_root=REPO
    )
    assert trace_rejected["valid"] is False
    assert trace_rejected["exact_replay_match"] is False
    assert "exact_replay" in trace_rejected["errors"]
