from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from official_submission import (
    REQUEST_SCHEMA,
    validate_candidate_submission,
    validate_submission,
    write_evaluation_request,
)


REPO = Path(__file__).resolve().parents[1]
END2END = REPO / "experiments" / "end2end_memory_systems_20260804"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _cfg(tmp_path: Path, *, metric: str = "multiclass_log_loss") -> SimpleNamespace:
    sample = _write(
        tmp_path / "data" / "sample_submission.csv",
        "id,A,B\n10,0.5,0.5\n20,0.5,0.5\n30,0.5,0.5\n",
    )
    workspace = tmp_path / "workspace"
    (workspace / "submission").mkdir(parents=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return SimpleNamespace(
        exp_id="task-with-arbitrary-official-test-size",
        data_dir=str(sample.parent),
        workspace_dir=workspace,
        log_dir=log_dir,
        fixed_holdout=SimpleNamespace(enabled=False),
        official_submission=SimpleNamespace(
            enabled=True,
            provider="kaggle",
            competition="task-competition",
            metric=metric,
            maximize=False,
            sample_submission_path=str(sample),
            id_column="id",
            prediction_kind="auto",
            probability_row_sum_tolerance=1e-6,
            submission_subdir="submission",
        ),
    )


def test_native_contract_uses_sample_shape_and_freezes_same_execution(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    node = SimpleNamespace(id="node-a", code="print('trained once')")
    submission = _write(
        cfg.workspace_dir / "submission" / "submission_node-a.csv",
        "id,A,B\n10,0.9,0.1\n20,0.2,0.8\n30,0.4,0.6\n",
    )
    receipt = validate_candidate_submission(cfg, node)
    assert receipt["row_count"] == 3
    assert receipt["requires_post_search_retraining"] is False
    assert receipt["submission_sha256"]

    journal = _write(
        cfg.log_dir / "journal.json",
        json.dumps(
            {
                "nodes": [
                    {
                        "id": node.id,
                        "code": node.code,
                        "metric": {"value": 0.123, "maximize": False},
                        "official_submission_receipt": receipt,
                    }
                ]
            },
            sort_keys=True,
        ),
    )
    request_path = write_evaluation_request(
        cfg,
        journal,
        selected_node_id=node.id,
        selection_basis={"type": "solver_internal_search_metric"},
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["request_schema"] == REQUEST_SCHEMA
    assert request["official_test_row_count"] == 3
    assert request["official_test_inference_during_candidate_execution"] is True
    assert request["selected_model_retrained_after_search"] is False
    assert request["selected_submission_sha256"] == receipt["submission_sha256"]
    assert submission.is_file()


@pytest.mark.parametrize(
    ("submission", "message"),
    [
        ("id,A,B\n10,0.9,0.1\n20,0.2,0.8\n", "row count"),
        ("id,A,B\n20,0.9,0.1\n10,0.2,0.8\n30,0.4,0.6\n", "IDs or order"),
        ("id,A,B\n10,0.9,0.2\n20,0.2,0.8\n30,0.4,0.6\n", "sum to one"),
    ],
)
def test_native_contract_rejects_incomplete_or_misaligned_official_predictions(
    tmp_path: Path,
    submission: str,
    message: str,
) -> None:
    cfg = _cfg(tmp_path)
    path = _write(tmp_path / "bad.csv", submission)
    with pytest.raises(ValueError, match=message):
        validate_submission(
            path,
            Path(cfg.official_submission.sample_submission_path),
            id_column="id",
            prediction_kind="multiclass_probability",
            probability_row_sum_tolerance=1e-6,
        )


def test_numeric_regression_predictions_do_not_require_probability_bounds(
    tmp_path: Path,
) -> None:
    sample = _write(
        tmp_path / "sample_submission.csv",
        "key,fare_amount\na,0\nb,0\n",
    )
    submission = _write(
        tmp_path / "submission.csv",
        "key,fare_amount\na,-3.5\nb,127.25\n",
    )
    result = validate_submission(
        submission,
        sample,
        id_column="key",
        prediction_kind="numeric",
    )
    assert result["row_count"] == 2
    assert result["prediction_columns"] == ["fare_amount"]


def test_end2end_runner_defers_official_score_without_retraining(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    node = SimpleNamespace(id="node-a", code="print('single candidate execution')")
    _write(
        cfg.workspace_dir / "submission" / "submission_node-a.csv",
        "id,A,B\n10,0.9,0.1\n20,0.2,0.8\n30,0.4,0.6\n",
    )
    receipt = validate_candidate_submission(cfg, node)
    journal = _write(
        cfg.log_dir / "journal.json",
        json.dumps(
            {
                "nodes": [
                    {
                        "id": node.id,
                        "code": node.code,
                        "metric": {"value": 0.123, "maximize": False},
                        "official_submission_receipt": receipt,
                    }
                ]
            },
            sort_keys=True,
        ),
    )
    write_evaluation_request(
        cfg,
        journal,
        selected_node_id=node.id,
        selection_basis={"type": "solver_internal_search_metric"},
    )
    evaluator_spec = _write(
        tmp_path / "release" / "evaluator.json",
        json.dumps({"kind": "deferred_official_kaggle_v1"}),
    )
    sys.path.insert(0, str(END2END))
    try:
        import run_assignment

        result = run_assignment.terminal_evaluate(
            evaluator_spec_path=evaluator_spec,
            release_root=evaluator_spec.parent,
            agent_log_root=cfg.log_dir,
            agent_workspace_root=cfg.workspace_dir,
            output_path=tmp_path / "TERMINAL_SCORE_REPORT.json",
            task={},
            timeout_seconds=30,
        )
    finally:
        sys.path.pop(0)
    assert result["deferred_official"] is True
    assert result["internal_search_metric"] == pytest.approx(0.123)
    assert result["selected_candidate_id"] == node.id
    assert result["report"]["official_score"] is None
    assert result["report"]["status"] == "awaiting_official_terminal_score"
