from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verify_tier2_formal_analysis_policy_addendum import (  # noqa: E402
    verify_analysis_policy_addendum,
)


ADDENDUM = (
    ROOT / "coordination" / "decision_admissibility_wp8_tier2_formal_"
    "analysis_policy_addendum_20260723_r1.json"
)


def _payload() -> dict:
    return json.loads(ADDENDUM.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    unsigned = {
        key: value for key, value in payload.items() if key != "analysis_policy_hash"
    }
    payload["analysis_policy_hash"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "analysis-policy-addendum.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_analysis_policy_addendum_verifies_result_blind_freeze() -> None:
    report = verify_analysis_policy_addendum(ADDENDUM, repo_root=ROOT)

    assert report["verified"] is True
    assert report["errors"] == []
    assert report["check_count"] == report["passed_check_count"]
    assert report["checks"]["itt_assigned_outcome_count"] is True
    assert report["checks"]["known_structural_count_partition"] is True
    assert report["checks"]["no_observed_score_payload"] is True


def test_rejects_rehashed_preregistration_or_structural_binding_mutation(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["preregistration_file_bindings"][0]["file_sha256"] = "0" * 64
    payload["structural_evidence_file_bindings"][-1]["file_sha256"] = "f" * 64

    report = verify_analysis_policy_addendum(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "preregistration_bindings_exact" in report["errors"]
    assert "preregistration_0_file_sha256" in report["errors"]
    assert "structural_bindings_exact" in report["errors"]
    assert "structural_6_file_sha256" in report["errors"]


def test_rejects_rehashed_disposition_or_failure_count_mutation(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["itt_disposition_matrix"][0]["dispositions"][
        "full_decision_admissibility"
    ] = "pre_terminal_failure"
    payload["known_structural_counts"]["failed_online_condition_count"] += 1
    payload["known_structural_counts"]["scored_selected_result_count"] -= 1

    report = verify_analysis_policy_addendum(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "itt_disposition_matrix_exact" in report["errors"]
    assert "known_structural_counts_exact" in report["errors"]


def test_rejects_failure_exclusion_or_continuous_imputation(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    itt = payload["analysis_policy"]["itt_population"]
    itt["post_assignment_exclusion_forbidden"] = False
    itt["oracle_or_source_score_imputation_forbidden"] = False
    continuous = payload["analysis_policy"][
        "continuous_native_and_standardized_effects"
    ]
    continuous["assigned_denominator_blocks"] = 5
    continuous["required_availability_report"]["missing_block_ids_required"] = False

    report = verify_analysis_policy_addendum(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "analysis_policy_exact" in report["errors"]


def test_rejects_orientation_reference_or_cross_task_raw_pooling(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    continuous = payload["analysis_policy"][
        "continuous_native_and_standardized_effects"
    ]
    continuous["native_delta"]["minimize"] = "left_score_minus_right_score"
    continuous["standardized_delta"]["regression_reference_system"] = "contrast_left"
    cross_task = payload["analysis_policy"]["cross_task_aggregation"]
    cross_task["raw_native_delta_pooling_across_tasks_forbidden"] = False

    report = verify_analysis_policy_addendum(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "analysis_policy_exact" in report["errors"]


def test_rejects_bootstrap_sign_flip_or_holm_mutation(tmp_path: Path) -> None:
    payload = copy.deepcopy(_payload())
    bootstrap = payload["analysis_policy"]["paired_bootstrap"]
    bootstrap["seed"] = 1
    bootstrap["iterations"] = 1000
    inference = payload["analysis_policy"]["exact_sign_flip_and_holm"]
    inference["alternative"] = "two_sided"
    inference["family_size"] = 1

    report = verify_analysis_policy_addendum(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "analysis_policy_exact" in report["errors"]


def test_rejects_relaxed_mixed_model_or_effect_claim_gate(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    mixed = payload["analysis_policy"]["mixed_effects_sensitivity"]
    mixed["minimum_distinct_tasks"] = 1
    mixed["not_estimable_output"]["reason_required"] = False
    payload["effect_claim_gate"]["default_effect_claim_authorized"] = True
    payload["effect_claim_gate"]["criteria"].pop()

    report = verify_analysis_policy_addendum(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "analysis_policy_exact" in report["errors"]
    assert "effect_claim_gate_exact" in report["errors"]


def test_rejects_injected_observed_terminal_score_even_when_rehashed(
    tmp_path: Path,
) -> None:
    payload = copy.deepcopy(_payload())
    payload["observed_scores"] = {"full_decision_admissibility": 0.9}

    report = verify_analysis_policy_addendum(
        _write(tmp_path, payload),
        repo_root=ROOT,
    )

    assert report["verified"] is False
    assert "top_level_keys_exact" in report["errors"]
    assert "no_observed_score_payload" in report["errors"]
