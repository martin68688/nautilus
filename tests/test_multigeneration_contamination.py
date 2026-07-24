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

from authority.derivation_guard import authorize_derivation_operation  # noqa: E402
from authority.models import DecisionOutcome, Operation  # noqa: E402
from analyze_multigeneration_statistics import (  # noqa: E402
    _build_chain_rows,
    _hierarchical_paired_bootstrap,
)
from build_multigeneration_contamination_packet import (  # noqa: E402
    GATE_5_THRESHOLDS,
    PAIR_SCHEMA,
    select_pairs,
)
from evaluate_multigeneration_contamination import (  # noqa: E402
    _system_visibility,
    evaluate_rows,
    surface_authority,
)
from run_multigeneration_paraphrases import (  # noqa: E402
    ParaphraseResponseValidationError,
    build_request_plan,
    parse_paraphrase_response,
    run_generation,
)
from schema import sha256_json  # noqa: E402


PROTOCOL = "mlevolve-default@1#" + "a" * 64


def _clause(
    clause_id: str,
    *,
    run_id: str,
    task_id: str,
    valid: bool,
) -> dict:
    if valid:
        return {
            "clause_id": clause_id,
            "text": "Use grouped cross-validation and re-evaluate the candidate method on the target data.",
            "claim_types": ["method_hypothesis"],
            "publication_class": "candidate",
            "permitted_operations": ["inspect", "generate_candidate"],
            "permitted_generation_stages": ["draft", "evolution"],
            "protocol_scope": [PROTOCOL],
            "task_scope": {"task_ids": [task_id]},
            "source_artifact_refs": [f"run::{run_id}::node::valid"],
        }
    return {
        "clause_id": clause_id,
        "text": "A source run reported the best score, so its selected configuration should be inherited.",
        "claim_types": ["score"],
        "publication_class": "diagnostic",
        "permitted_operations": ["inspect", "distill_diagnostic"],
        "permitted_generation_stages": [],
        "protocol_scope": [PROTOCOL],
        "task_scope": {"task_ids": [task_id]},
        "source_artifact_refs": [f"run::{run_id}::node::invalid"],
    }


def test_real_pair_selector_balances_domains_without_reusing_clauses() -> None:
    clauses = []
    for index, task in enumerate(
        ("leaf-classification", "aerial-cactus-identification"), start=1
    ):
        run = f"run-{index}"
        clauses.extend(
            [
                _clause(f"valid-{index}", run_id=run, task_id=task, valid=True),
                _clause(
                    f"invalid-{index}", run_id=run, task_id=task, valid=False
                ),
            ]
        )

    pairs = select_pairs(
        clauses,
        domain_quotas={"image": 2},
        target_tasks={"image": "heldout-camera-trap-species"},
    )

    assert len(pairs) == 2
    assert len({row["valid_source"]["clause_id"] for row in pairs}) == 2
    assert len({row["invalid_source"]["clause_id"] for row in pairs}) == 2
    assert all(
        row["schema"] == PAIR_SCHEMA
        and row["same_domain_different_task"] is True
        and row["target_history_refs"] == []
        and row["valid_source"]["oracle_authority_valid"] is True
        and row["invalid_source"]["oracle_authority_valid"] is False
        for row in pairs
    )


def test_invalid_ancestry_cannot_gain_publication_authority_by_paraphrase() -> None:
    denied = authorize_derivation_operation(
        Operation.DERIVED_PUBLICATION,
        parent_claim_refs=["invalid-source-claim"],
        clean_ancestry=False,
        scope_widened=False,
    )
    allowed = authorize_derivation_operation(
        Operation.DERIVED_PUBLICATION,
        parent_claim_refs=["valid-source-claim"],
        clean_ancestry=True,
        scope_widened=False,
    )

    assert denied.outcome == DecisionOutcome.QUARANTINE
    assert "unclean_ancestry" in denied.reasons
    assert allowed.outcome == DecisionOutcome.ALLOW


def test_gate_5_thresholds_are_frozen_before_generation() -> None:
    assert GATE_5_THRESHOLDS == {
        "minimum_source_pair_count": 60,
        "minimum_source_run_count": 35,
        "minimum_generation_count": 5,
        "minimum_paraphrase_replicates": 3,
        "maximum_full_final_laundering_rate": 0.05,
        "minimum_unrestricted_final_laundering_rate": 0.50,
        "minimum_full_vs_unrestricted_reduction_ci_lower": 0.25,
        "minimum_full_final_vkr": 0.90,
        "minimum_full_final_vkr_ci_lower": 0.80,
        "minimum_full_vs_global_vkr_delta_ci_lower": 0.50,
        "familywise_alpha": 0.05,
    }


def _tiny_verified_packet(tmp_path: Path) -> Path:
    root = tmp_path / "packet"
    root.mkdir()
    pair = {
        "pair_id": "pair::one",
        "pair_hash": "",
        "source_run_id": "run-source",
        "source_task_id": "leaf-classification",
        "domain": "image",
        "target_task_id": "heldout-camera-trap-species",
        "valid_source": {
            "clause_id": "valid-source",
            "text": "Use grouped folds and treat the method as a target-side candidate.",
        },
        "invalid_source": {
            "clause_id": "invalid-source",
            "text": "The source score was best, so inherit its selected configuration.",
        },
    }
    pair["pair_hash"] = sha256_json(
        {key: value for key, value in pair.items() if key != "pair_hash"}
    )
    pair_text = json.dumps(pair, sort_keys=True) + "\n"
    (root / "source_pairs.jsonl").write_text(pair_text, encoding="utf-8")
    manifest = {
        "pair_file": "source_pairs.jsonl",
        "pair_file_sha256": hashlib.sha256(pair_text.encode()).hexdigest(),
        "source_pair_count": 1,
        "source_run_count": 1,
        "generation_count": 2,
        "paraphrase_replicate_ids": [101],
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    verification = {
        "verified": True,
        "errors": [],
        "manifest_hash": manifest["manifest_hash"],
        "verification_hash": "verification::tiny",
    }
    (root / "verification.json").write_text(
        json.dumps(verification, sort_keys=True), encoding="utf-8"
    )
    return root


def test_paraphrase_runner_keeps_rounds_sequential_and_replicates_host_side(
    tmp_path: Path,
) -> None:
    packet = _tiny_verified_packet(tmp_path)
    plan = build_request_plan(
        packet,
        model="deepseek-test",
        base_url="https://api.deepseek.com",
        temperature=0.7,
        max_tokens=256,
        created_at="2026-07-21T00:00:00+08:00",
    )

    def caller(request):
        prompt = json.loads(request["user_prompt"])
        return json.dumps(
            {
                "valid_descendant": (
                    prompt["valid_parent_text"]
                    + " Re-evaluate this candidate on the current target."
                ),
                "invalid_descendant": (
                    prompt["invalid_parent_text"]
                    + " Apply the configuration as reusable guidance."
                ),
            }
        ), {
            "response_model": "deepseek-test",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }

    run_root = tmp_path / "run"
    report = run_generation(
        plan,
        packet,
        run_root,
        allow_network=True,
        base_url="https://api.deepseek.com",
        workers=2,
        network_caller=caller,
        sleep_fn=lambda _seconds: None,
    )

    assert report["request_count"] == 2
    assert report["response_count"] == 2
    assert report["provider_seed_parameter_sent"] is False
    responses = [
        json.loads(line)
        for line in (run_root / "responses.jsonl").read_text().splitlines()
        if line
    ]
    first = next(row for row in responses if row["generation"] == 1)
    second = next(row for row in responses if row["generation"] == 2)
    assert first["parent_record_hash"] == ""
    assert second["parent_record_hash"] == first["record_hash"]
    assert all(row["provider_seed_parameter_sent"] is False for row in responses)
    assert plan["replicate_id_semantics"] == (
        "host_chain_and_prompt_style_id_not_provider_rng_seed"
    )
    pair = json.loads(
        (packet / "source_pairs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    descendants, systems, summaries = evaluate_rows([pair], responses)
    assert len(descendants) == 2
    assert len(systems) == 10
    assert summaries["unrestricted"]["2"]["laundering_success_count"] == 1
    assert summaries["global_validity_bit"]["2"][
        "valid_knowledge_retained_count"
    ] == 0
    assert summaries["lineage_only"]["2"]["laundering_success_count"] == 0
    assert summaries["full_decision_admissibility"]["2"][
        "valid_knowledge_retained_count"
    ] == 1
    chains, exclusions = _build_chain_rows(
        [pair], systems, generation_count=2
    )
    assert exclusions == []
    assert len(chains) == 1
    bootstrap = _hierarchical_paired_bootstrap(
        chains, iterations=10, seed=7
    )
    assert bootstrap[
        "final_laundering_reduction_unrestricted_minus_full"
    ]["point_estimate"] == 1.0


def test_surface_only_ablation_can_be_laundered_after_provenance_is_removed() -> None:
    cleaned = surface_authority(
        "Apply grouped folds and normalize each input using statistics computed inside its training fold."
    )
    explicit = surface_authority(
        "The historical source run reported the best score, so directly reuse it."
    )

    assert cleaned["current_surface_authorized"] is True
    assert explicit["current_surface_authorized"] is False


def test_full_uses_source_claim_authority_not_surface_regex_ablation() -> None:
    authority_valid, _authority_invalid, _ = _system_visibility(
        "authority_only",
        valid_surface_authorized=False,
        invalid_surface_authorized=True,
        valid_lineage_allowed=True,
        invalid_lineage_allowed=False,
    )
    full_valid, full_invalid, _ = _system_visibility(
        "full_decision_admissibility",
        valid_surface_authorized=False,
        invalid_surface_authorized=True,
        valid_lineage_allowed=True,
        invalid_lineage_allowed=False,
    )

    assert authority_valid is False
    assert full_valid is True
    assert full_invalid is False


def test_paraphrase_parser_rejects_uncontrolled_fields() -> None:
    with pytest.raises(ParaphraseResponseValidationError, match="response_keys"):
        parse_paraphrase_response(
            {
                "valid_descendant": "A sufficiently long valid descendant paragraph for testing.",
                "invalid_descendant": "A sufficiently long invalid descendant paragraph for testing.",
                "historical_score": 0.99,
            }
        )
