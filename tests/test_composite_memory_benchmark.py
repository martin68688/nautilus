from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "paper-skills" / "eval_composite_memory"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
if str(REPO / "mlevolve") not in sys.path:
    sys.path.insert(0, str(REPO / "mlevolve"))

from core import EPISODES, REPORTS, read_json, read_jsonl, sha256_file  # noqa: E402
from run_agent_adoption import extract_code_facts  # noqa: E402
from run_agent_adoption import run as score_adoption  # noqa: E402
from evaluate_hidden_holdout import evaluate  # noqa: E402
from agents.leakage_audit import audit_code, structural_sha256  # noqa: E402
from score_composite_benchmark import _ordinal_alpha, score  # noqa: E402


def test_builder_freezes_expected_episode_and_defect_counts() -> None:
    report = read_json(REPORTS / "build_report_v1.json")
    assert report["test_episode_count"] == 120
    assert report["dev_episode_count"] == 20
    assert report["replay_defect_count"] == 48
    assert set(report["test_by_family"].values()) == {20}
    assert report["test_by_stage"] == {"debug": 36, "draft": 24, "improve": 36, "model_design": 24}


def test_source_run_split_is_disjoint_and_ids_are_hidden() -> None:
    manifest = read_json(BENCH / "manifests" / "memory_snapshot_manifest_v1.json")
    split = {key: set(value) for key, value in manifest["run_split"].items()}
    assert not split["memory_train"] & split["benchmark_dev"]
    assert not split["memory_train"] & split["benchmark_test"]
    assert not split["benchmark_dev"] & split["benchmark_test"]
    assert all(row["source_run_ids_exposed"] is False for row in read_jsonl(EPISODES / "decision_test_v1.jsonl"))


def test_gold_is_production_clean_or_explicit_coverage_gap() -> None:
    gold = read_jsonl(EPISODES / "decision_test_silver_gold_v1.jsonl")
    assert all(len(row["labels"]) >= 3 or row["expected_status"] == "insufficient_strategy_coverage" for row in gold)
    assert any(row["expected_status"] == "insufficient_strategy_coverage" for row in gold)


def test_replay_static_runner_never_executes_or_ranks_source() -> None:
    report = read_json(REPORTS / "replay_static_report_v1.json")
    receipts = read_jsonl(REPORTS / "replay_static_receipts_v1.jsonl")
    assert report["case_count"] == 48
    assert report["source_execution_count"] == 0
    assert report["source_ranked_count"] == 0
    assert all(row["candidate_execution_attempted"] is False for row in receipts)
    assert all(row["source_rank_eligible"] is False for row in receipts)


def test_all_replay_defect_templates_are_detected_and_blocked() -> None:
    cases = read_jsonl(EPISODES / "replay_defects_v1.jsonl")
    assert len({structural_sha256(case["code"]) for case in cases}) == 16
    for defect in {case["defect"] for case in cases}:
        defect_cases = [case for case in cases if case["defect"] == defect]
        assert {case["implementation_variant"] for case in defect_cases} == {1, 2}
        for variant in (1, 2):
            families = {
                case["task_family"] for case in defect_cases if case["implementation_variant"] == variant
            }
            assert len(families) == 3
    for case in cases:
        audit = audit_code(case["code"])
        codes = {row["issue_code"] for row in audit["issues"]}
        assert case["issue_code"] in codes, case["case_id"]
        assert audit["execution_disposition"] == "block", case["case_id"]


def test_replay_matrix_separates_r0_r1_r2_r3_and_fails_closed() -> None:
    matrix = read_json(BENCH / "manifests" / "replay_matrix_v1.json")
    report = read_json(REPORTS / "replay_condition_report_v1.json")
    assert matrix["case_count"] == 48
    assert matrix["run_count"] == 192
    assert set(matrix["conditions"]) == {"R0", "R1", "R2", "R3"}
    assert matrix["conditions"]["R0"]["agent_generation"] is False
    assert matrix["conditions"]["R2"]["staged_protocol"] is True
    assert matrix["conditions"]["R2"]["preservation_contract_exposed"] is False
    assert matrix["conditions"]["R3"]["preservation_contract_exposed"] is True
    assert report["all_sources_blocked"] is True
    assert report["source_execution_count"] == 0
    assert report["source_rank_count"] == 0
    assert report["repair_claim_allowed"] is False
    assert all(
        values["runtime_provenance_verified_clean_rate"] == 0.0
        for values in report["by_condition"].values()
    )


def test_heldout_replay_set_is_independently_authored_and_frozen() -> None:
    cases = read_jsonl(EPISODES / "replay_defects_heldout_claude_v1.jsonl")
    report = read_json(REPORTS / "replay_heldout_report_v1.json")
    assert len(cases) == 16
    assert len({case["defect"] for case in cases}) == 8
    assert all(case["detector_tuning_forbidden"] is True for case in cases)
    assert len({case["authoring_session"] for case in cases}) == 1
    assert report["case_count"] == 16
    assert report["detector_source_was_hidden_from_author"] is True
    assert report["post_evaluation_detector_tuning_forbidden"] is True
    lock = read_json(BENCH / "manifests" / "replay_heldout_lock_v1.json")
    assert lock["source_sha256"] == sha256_file(EPISODES / "replay_defects_heldout_claude_v1.jsonl")
    assert lock["detector_source_sha256"] == report["detector_source_sha256"]
    addendum = read_json(
        BENCH
        / "manifests"
        / "replay_heldout_detector_provenance_addendum_v1.json"
    )
    unsigned = {
        key: value for key, value in addendum.items() if key != "addendum_hash"
    }
    assert addendum["addendum_hash"] == hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert addendum["heldout_lock"]["file_sha256"] == sha256_file(
        BENCH / "manifests" / "replay_heldout_lock_v1.json"
    )
    assert addendum["heldout_report"]["file_sha256"] == sha256_file(
        REPORTS / "replay_heldout_report_v1.json"
    )
    historical = addendum["historical_detector_source"]
    assert historical["sha256"] == lock["detector_source_sha256"]
    live = addendum["live_detector_at_addendum"]
    assert live["sha256"] == sha256_file(
        REPO / "mlevolve" / "agents" / "leakage_audit.py"
    )
    assert live["detector_version"] == "deterministic_static_v5"
    assert live["used_to_recompute_v4_heldout_results"] is False
    archived = subprocess.run(
        [
            "git",
            "show",
            f"{historical['git_commit']}:{historical['path']}",
        ],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    if archived.returncode == 0:
        assert hashlib.sha256(archived.stdout).hexdigest() == historical[
            "sha256"
        ]


def test_micro_matrix_has_12_tasks_3_seeds_and_5_conditions() -> None:
    matrix = read_json(BENCH / "manifests" / "micro_execution_matrix_v1.json")
    assert matrix["task_count"] == 12
    assert matrix["run_count"] == 180
    assert matrix["seeds"] == [17, 43, 79]
    assert set(matrix["conditions"]) == {"B0", "F00", "F01", "F10", "F11"}


def test_adoption_matrix_uses_all_normal_episodes_without_gold() -> None:
    matrix = read_json(BENCH / "manifests" / "adoption_matrix_v1.json")
    assert matrix["episode_count"] >= 60
    assert matrix["run_count"] == matrix["episode_count"] * 4
    assert matrix["gold_exposed_to_generator"] is False


def test_code_fact_extractor_checks_actual_code() -> None:
    facts = extract_code_facts("vectorizer = TfidfVectorizer()\nmodel = LogisticRegression()\n")
    assert facts["parse_ok"] is True
    assert "linear_sparse" in facts["model_families"]
    exact = extract_code_facts("_modernbert_large_spec()\n_tfidf_logreg_control()\n")
    assert "modernbert_finetune" in exact["model_families"]
    assert "tfidf_stylometry_linear" in exact["model_families"]


def test_failed_agent_generation_is_not_counted_as_adoption(tmp_path: Path) -> None:
    candidates = tmp_path / "failed.jsonl"
    candidates.write_text(json.dumps({
        "episode_id": "composite::test::spooky-author-identification::draft_capacity",
        "condition": "F00",
        "seed": 0,
        "selected_memory_ids": [],
        "prompt_sha256": "x",
        "code": "",
        "adoption_outcome": "generation_failed",
        "status": "failed",
        "mock": False,
    }) + "\n")
    report = score_adoption(candidates, persist=False)
    assert report["candidate_count"] == 1
    assert report["non_mock_candidate_count"] == 0
    assert report["complete_four_condition_episode_count"] == 0


def test_t2_pilot_preserves_first_pass_failures_and_cannot_open_claim() -> None:
    path = REPORTS / "adoption_pilot_report_v1.json"
    if not path.exists():
        return
    report = read_json(path)
    assert report["first_pass_attempt_count"] == 4
    assert report["first_pass_completed_count"] <= report["completed_condition_count_after_retry"]
    assert report["condition_count"] == 4
    assert report["claim_allowed"] is False
    assert report["never_successful_count"] == 0
    assert report["test_split_consumed_by_pilot"] is True
    assert "pilot_has_only_one_episode" in report["claim_blockers"]


def test_hidden_evaluator_requires_exact_sample_coverage(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    receipts = tmp_path / "receipts.jsonl"
    hidden = tmp_path / "hidden.json"
    labels.write_text('{"sample_id":"a","label":1.0}\n{"sample_id":"b","label":3.0}\n')
    predictions.write_text('{"sample_id":"a","prediction":1.0}\n{"sample_id":"b","prediction":2.0}\n')
    receipts.write_text(json.dumps({
        "task_id": "toy", "condition": "B0", "seed": 17, "status": "completed",
        "prediction_output_path": str(predictions),
    }) + "\n")
    hidden.write_text(json.dumps({"tasks": {"toy": {"labels_path": str(labels), "metric": "rmse"}}}))
    scored = evaluate(receipts, hidden, output_path=tmp_path / "scored.jsonl")
    assert scored[0]["trusted"] is True
    assert scored[0]["metric"] == 2 ** -0.5


def test_test_split_claims_fail_closed_without_blind_labels_and_t4() -> None:
    assert (REPORTS / "offline_test_report_v1.json").exists()
    report = score("test", persist=False)
    assert report["split"] == "test"
    assert report["normal_episode_count"] == 70
    assert report["coverage_gap_episode_count"] == 50
    assert report["mechanism_claim_allowed"] is False
    assert report["downstream_claim_allowed"] is False
    assert "coverage_gap_count_zero" in report["claim_blockers"]
    assert "replay_heldout_expected_issue_recall_one" in report["claim_blockers"]
    assert "replay_heldout_all_sources_blocked" in report["claim_blockers"]


def test_current_t1_negative_results_are_not_hidden() -> None:
    report = read_json(REPORTS / "claim_gate_report_v1.json")
    findings = report["diagnostic_findings"]
    assert findings["stage_hybrid_minus_sop_only_ndcg"] < 0.0
    assert findings["poincare_minus_flat_twin_ndcg"] < 0.0


def test_ordinal_krippendorff_alpha_has_standard_endpoints() -> None:
    perfect = [
        {"episode_id": "e", "candidate_id": str(index), "annotator_id": annotator, "relevance": value}
        for index, value in enumerate((0, 1, 2, 3)) for annotator in ("a", "b")
    ]
    assert _ordinal_alpha(perfect) == 1.0
    disagreement = [
        {"episode_id": "e", "candidate_id": str(index), "annotator_id": "a", "relevance": value}
        for index, value in enumerate((0, 1, 2, 3))
    ] + [
        {"episode_id": "e", "candidate_id": str(index), "annotator_id": "b", "relevance": 3 - value}
        for index, value in enumerate((0, 1, 2, 3))
    ]
    assert _ordinal_alpha(disagreement) < 0.0


def test_geometry_and_clean_universe_receipts_are_distinct() -> None:
    path = REPORTS / "offline_test_receipts_v1.jsonl"
    assert path.exists()
    rows = read_jsonl(path)
    f11 = [row for row in rows if row["condition"] == "F11"]
    d6 = [row for row in rows if row["condition"] == "D6"]
    d7 = [row for row in rows if row["condition"] == "D7"]
    d8 = [row for row in rows if row["condition"] == "D8"]
    assert f11 and d6 and d7 and d8
    assert all(row["provenance"]["scoring_mode"] == "poincare" for row in f11)
    assert all(row["provenance"]["scoring_mode"] == "flat_twin" for row in d6)
    assert all(row["provenance"]["candidate_universe"] == "preclean" for row in d8)
    assert all(row["provenance"]["safety_gate_disabled"] is True for row in d8)
    assert sum(row["unsafe_count_at_10"] for row in d7) > 0
    assert all(row["unsafe_count_at_10"] == 0 for row in d8)


def test_coverage_audit_does_not_hide_missing_memory() -> None:
    from audit_memory_coverage import audit

    report = audit()
    assert report["coverage_gap_count"] == 50
    assert report["coverage_complete_count"] == 70
    assert report["requires_genuinely_new_clean_evidence_count"] == 30
    assert report["full_graph_support_removed_by_frozen_split_count"] == 20
    assert report["can_be_fixed_by_resplitting_without_invalidating_frozen_source_separation"] is False


def test_terminal_report_applies_preregistered_stop_rule() -> None:
    from finalize_benchmark import finalize

    report = finalize()
    assert report["execution_status"] == "completed_stopped_fail_closed"
    assert report["stop_rule_triggered"] == "independent_replay_safety_challenge_failed"
    assert report["phase_status"]["T4_micro_execution"] == "not_started_by_preregistered_stop_rule"
    assert report["key_results"]["T4_completed_count"] == 0
    assert not any(
        report["claim_gates"][key]
        for key in (
            "mechanism_claim_allowed",
            "adoption_claim_allowed",
            "replay_repair_success_claim_allowed",
            "downstream_claim_allowed",
        )
    )
