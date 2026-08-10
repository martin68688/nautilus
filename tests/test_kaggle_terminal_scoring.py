from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest

from fixed_holdout.common import sha256_file
from fixed_holdout.kaggle_terminal import (
    score_kaggle_terminal,
    write_evaluator_spec,
    write_official_measurement,
)
from official_submission import (
    validate_candidate_submission,
    write_evaluation_request as write_official_evaluation_request,
)


REPO = Path(__file__).resolve().parents[1]
END2END = REPO / "experiments" / "end2end_memory_systems_20260804"


def _hash(payload: dict, field: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != field},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_hashed(path: Path, payload: dict, field: str) -> None:
    payload[field] = _hash(payload, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _submission(path: Path, first_probability: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "id,A,B\n"
        f"1,{first_probability},{1 - first_probability}\n"
        f"2,{1 - first_probability},{first_probability}\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    sample = tmp_path / "sample_submission.csv"
    _submission(sample, 0.5)
    submission_dir = tmp_path / "run" / "workspace" / "submission"
    _submission(submission_dir / "submission_selected.csv", 0.9)
    _submission(submission_dir / "submission_other.csv", 0.4)
    journal = tmp_path / "run" / "logs" / "journal.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "selected", "metric": {"value": 0.02}},
                    {"id": "other", "metric": {"value": 0.01}},
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    inventory = [
        {
            "node_id": path.stem.removeprefix("submission_"),
            "submission": path.name,
            "submission_sha256": sha256_file(path),
        }
        for path in sorted(submission_dir.glob("submission_*.csv"))
    ]
    selected = next(row for row in inventory if row["node_id"] == "selected")
    request = {
        "request_schema": "fixed_holdout_evaluation_request_v3",
        "task_id": "leaf-classification",
        "split_id": "official-test",
        "metric": "multiclass_log_loss",
        "maximize": False,
        "selection_policy": "terminal_only",
        "scores_were_visible_during_search": False,
        "selection_frozen_before_terminal_evaluation": True,
        "status": "awaiting_external_evaluator",
        "journal_path": str(journal.resolve()),
        "journal_sha256": sha256_file(journal),
        "submission_dir": str(submission_dir.resolve()),
        "candidate_inventory": inventory,
        "candidate_set_hash": _hash({"candidate_inventory": inventory}, "unused"),
        "selected_node_id": "selected",
        "selected_submission": selected["submission"],
        "selected_submission_sha256": selected["submission_sha256"],
        "selection_basis": {
            "type": "solver_internal_search_metric",
            "metric_disposition": "search_only",
            "terminal_metric_observed": False,
        },
        "request_hash": "",
    }
    request_path = tmp_path / "run" / "logs" / "fixed_holdout_evaluation_request.json"
    _write_hashed(request_path, request, "request_hash")
    spec = {
        "schema": "mlevolve_kaggle_terminal_evaluator_v1",
        "task_id": "leaf-classification",
        "competition": "leaf-classification",
        "metric": "multiclass_log_loss",
        "maximize": False,
        "sample_submission": str(sample.resolve()),
        "id_column": "id",
        "prediction_kind": "probability",
        "score_field_preference": ["privateScore", "publicScore"],
        "poll_seconds": 0,
        "poll_timeout_seconds": 5,
        "spec_hash": "",
    }
    spec_path = tmp_path / "kaggle_evaluator.json"
    _write_hashed(spec_path, spec, "spec_hash")
    return {
        "sample": sample,
        "submission_dir": submission_dir,
        "journal": journal,
        "request": request_path,
        "spec": spec_path,
        "report": tmp_path / "OFFICIAL_SCORE_REPORT.json",
        "work": tmp_path / "official-work",
    }


class FakeKaggle:
    def __init__(self, *, existing: bool = False) -> None:
        self.rows: list[dict[str, str]] = []
        self.submit_calls = 0
        self.list_calls = 0
        if existing:
            self._complete(
                "mlevolve_terminal_placeholder.csv",
                "placeholder",
            )

    def _complete(self, filename: str, description: str) -> None:
        self.rows = [
            {
                "fileName": filename,
                "date": "2026-08-10",
                "description": description,
                "status": "SubmissionStatus.COMPLETE",
                "publicScore": "0.01008",
                "privateScore": "0.01111",
                "ref": "12345",
            }
        ]

    def __call__(self, command: Sequence[str]) -> str:
        command = list(command)
        if "submissions" in command:
            self.list_calls += 1
            output = []
            if self.rows:
                fieldnames = list(self.rows[0])
                from io import StringIO

                stream = StringIO()
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
                output.append(stream.getvalue())
            return "".join(output)
        if "submit" in command:
            self.submit_calls += 1
            filename = Path(command[command.index("-f") + 1]).name
            description = command[command.index("-m") + 1]
            self._complete(filename, description)
            return "Successfully submitted"
        raise AssertionError(command)


def test_task_evaluator_spec_pins_dynamic_sample_submission(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "generated-kaggle-spec.json"
    spec = write_evaluator_spec(
        output,
        task_id="leaf-classification",
        competition="leaf-classification",
        metric="multiclass_log_loss",
        maximize=False,
        sample_submission=fixture["sample"],
        id_column="id",
        prediction_kind="multiclass_probability",
    )
    assert spec["sample_submission_sha256"] == sha256_file(fixture["sample"])
    assert spec["credentials_embedded"] is False
    assert (
        write_evaluator_spec(
            output,
            task_id="leaf-classification",
            competition="leaf-classification",
            metric="multiclass_log_loss",
            maximize=False,
            sample_submission=fixture["sample"],
            id_column="id",
            prediction_kind="multiclass_probability",
        )
        == spec
    )


def test_official_scorer_submits_only_preselected_candidate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fake = FakeKaggle()
    report = score_kaggle_terminal(
        fixture["spec"],
        fixture["request"],
        fixture["report"],
        work_dir=fixture["work"],
        kaggle_cli=Path("fake-kaggle"),
        poll_seconds=0,
        poll_timeout_seconds=1,
        command_runner=fake,
        sleep=lambda _: None,
    )

    assert fake.submit_calls == 1
    assert report["selected_node_id"] == "selected"
    assert report["selected_score"] == pytest.approx(0.01111)
    assert report["selected_score_source"] == "privateScore"
    assert report["internal_search_metric"] == pytest.approx(0.02)
    assert report["scores_were_visible_during_search"] is False
    assert report["system_selection_used_official_score"] is False
    assert report["memory_admission"]["official_score_sufficient"] is False
    assert len(list(fixture["work"].glob("*.csv"))) == 1


def test_official_scorer_is_idempotent_without_resubmission(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fake = FakeKaggle()
    first = score_kaggle_terminal(
        fixture["spec"],
        fixture["request"],
        fixture["report"],
        work_dir=fixture["work"],
        command_runner=fake,
        sleep=lambda _: None,
    )

    def forbidden(_: Sequence[str]) -> str:
        raise AssertionError("idempotent replay must not call Kaggle")

    second = score_kaggle_terminal(
        fixture["spec"],
        fixture["request"],
        fixture["report"],
        work_dir=fixture["work"],
        command_runner=forbidden,
    )
    assert second == first
    assert fake.submit_calls == 1


def test_official_scorer_reuses_matching_completed_submission(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selected = Path(fixture["submission_dir"]) / "submission_selected.csv"
    digest = sha256_file(selected)
    fake = FakeKaggle()
    fake._complete(
        f"mlevolve_terminal_{digest[:16]}.csv",
        f"mlevolve-terminal-v1 sha256={digest}",
    )
    report = score_kaggle_terminal(
        fixture["spec"],
        fixture["request"],
        fixture["report"],
        work_dir=fixture["work"],
        command_runner=fake,
        sleep=lambda _: None,
    )
    assert fake.submit_calls == 0
    assert report["official_submission"]["reference"] == "12345"


def test_official_scorer_accepts_native_no_retraining_request(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    submission_dir = fixture["submission_dir"]
    log_dir = fixture["journal"].parent
    cfg = SimpleNamespace(
        exp_id="leaf-classification",
        data_dir=str(fixture["sample"].parent),
        workspace_dir=submission_dir.parent,
        log_dir=log_dir,
        fixed_holdout=SimpleNamespace(enabled=False),
        official_submission=SimpleNamespace(
            enabled=True,
            provider="kaggle",
            competition="leaf-classification",
            metric="multiclass_log_loss",
            maximize=False,
            sample_submission_path=str(fixture["sample"]),
            id_column="id",
            prediction_kind="multiclass_probability",
            probability_row_sum_tolerance=1e-6,
            submission_subdir="submission",
        ),
    )
    node = SimpleNamespace(id="selected", code="print('single training execution')")
    receipt = validate_candidate_submission(cfg, node)
    journal_payload = json.loads(fixture["journal"].read_text(encoding="utf-8"))
    selected = next(row for row in journal_payload["nodes"] if row["id"] == "selected")
    selected["code"] = node.code
    selected["official_submission_receipt"] = receipt
    fixture["journal"].write_text(
        json.dumps(journal_payload, sort_keys=True), encoding="utf-8"
    )
    native_request = write_official_evaluation_request(
        cfg,
        fixture["journal"],
        selected_node_id="selected",
        selection_basis={"type": "solver_internal_search_metric"},
    )
    native_report = tmp_path / "NATIVE_OFFICIAL_SCORE_REPORT.json"
    report = score_kaggle_terminal(
        fixture["spec"],
        native_request,
        native_report,
        work_dir=fixture["work"],
        command_runner=FakeKaggle(),
        sleep=lambda _: None,
    )
    assert report["official_test_inference_during_candidate_execution"] is True
    assert report["selected_model_retrained_after_search"] is False


def test_official_scorer_rejects_post_freeze_submission_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _submission(
        Path(fixture["submission_dir"]) / "submission_selected.csv",
        0.8,
    )
    with pytest.raises(ValueError, match="Candidate inventory changed"):
        score_kaggle_terminal(
            fixture["spec"],
            fixture["request"],
            fixture["report"],
            work_dir=fixture["work"],
            command_runner=FakeKaggle(),
        )


def test_official_measurement_preserves_internal_score_as_diagnostic(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    base_path = tmp_path / "attempt-000" / "MEASUREMENT.json"
    report_path = base_path.with_name("OFFICIAL_SCORE_REPORT.json")
    report = score_kaggle_terminal(
        fixture["spec"],
        fixture["request"],
        report_path,
        work_dir=fixture["work"],
        command_runner=FakeKaggle(),
        sleep=lambda _: None,
    )
    base = {
        "schema": "mlevolve_end2end_condition_measurement_v1",
        "logical_run_id": "leaf-dynamic",
        "attempt": 0,
        "task_id": "leaf-classification",
        "system_id": "dynamic_hybrid",
        "seed": 1,
        "completed": False,
        "status": "awaiting_official_terminal_score",
        "failure_class": "none",
        "formal_result_eligible": True,
        "terminal_score": None,
        "internal_search_metric": 0.02,
        "internal_metric_disposition": "diagnostic_only",
        "terminal_metric": "log_loss",
        "selected_candidate_id": "selected",
        "candidate_set_frozen": True,
        "candidate_set_hash": report["candidate_set_hash"],
        "measurement_hash": "",
    }
    _write_hashed(base_path, base, "measurement_hash")
    official_path = base_path.with_name("OFFICIAL_MEASUREMENT.json")
    overlay = write_official_measurement(base_path, report_path, official_path)
    repeated = write_official_measurement(base_path, report_path, official_path)
    assert repeated == overlay
    assert overlay["primary_score"] == report["selected_score"]
    assert overlay["internal_terminal_score"] == pytest.approx(0.02)
    assert overlay["primary_score_authority"] == "official_kaggle_terminal"

    sys.path.insert(0, str(END2END))
    try:
        import analyze_results

        rows = analyze_results.load_measurements(
            tmp_path, formal_only=True, score_authority="official"
        )
    finally:
        sys.path.pop(0)
    assert rows[0]["terminal_score"] == overlay["primary_score"]
    assert rows[0]["internal_terminal_score"] == pytest.approx(0.02)
    assert rows[0]["score_authority"] == "official_kaggle_terminal"
    assert rows[0]["completed"] is True

    import validate_smoke_gate

    effective = validate_smoke_gate._official_measurement_overlay(
        base,
        base_path=base_path,
        attempt_root=base_path.parent,
    )
    assert effective["completed"] is True
    assert effective["terminal_score"] == report["selected_score"]
    assert effective["_terminal_report_filename"] == "OFFICIAL_SCORE_REPORT.json"
