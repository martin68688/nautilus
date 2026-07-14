import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from fixed_holdout.evaluate import evaluate_submission  # noqa: E402
from fixed_holdout.mode import bypass_protocol_gates, enabled  # noqa: E402
from fixed_holdout.prepare import prepare_task  # noqa: E402
from fixed_holdout.score_run import score_run  # noqa: E402
from fixed_holdout.validation import validate_submission, validate_train_view  # noqa: E402
from agents.memory.run_forest_replay import (  # noqa: E402
    requires_protocol_repair,
    validate_candidate_audit,
)
from config import _load_cfg  # noqa: E402


def _source_dataset(tmp_path: Path) -> Path:
    task_root = tmp_path / "datasets" / "spooky-author-identification" / "prepared"
    public = task_root / "public"
    private = task_root / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": ["train-1", "train-2", "train-3"],
            "text": ["one", "two", "three"],
            "author": ["EAP", "HPL", "MWS"],
        }
    ).to_csv(public / "train.csv", index=False)
    pd.DataFrame(
        {"id": ["holdout-1", "holdout-2"], "text": ["alpha", "beta"]}
    ).to_csv(public / "test.csv", index=False)
    pd.DataFrame(
        {
            "id": ["holdout-1", "holdout-2"],
            "EAP": [1 / 3, 1 / 3],
            "HPL": [1 / 3, 1 / 3],
            "MWS": [1 / 3, 1 / 3],
        }
    ).to_csv(public / "sample_submission.csv", index=False)
    (public / "description.md").write_text("Synthetic author task\n", encoding="utf-8")
    pd.DataFrame(
        {
            "id": ["holdout-1", "holdout-2"],
            "EAP": [1, 0],
            "HPL": [0, 1],
            "MWS": [0, 0],
            "private_note": ["must not escape", "must not escape"],
        }
    ).to_csv(private / "test.csv", index=False)
    return tmp_path / "datasets"


def _prepared(tmp_path: Path) -> Path:
    return prepare_task(
        _source_dataset(tmp_path),
        "spooky-author-identification",
        tmp_path / "fixed",
        copy_mode="copy",
    )


def _write_scalar_task_source(
    tmp_path: Path,
    *,
    task_id: str,
    id_column: str,
    prediction_column: str,
    labels: list[float],
) -> Path:
    task_root = tmp_path / "datasets" / task_id / "prepared"
    public = task_root / "public"
    private = task_root / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    ids = [f"holdout-{index}" for index in range(len(labels))]
    pd.DataFrame({id_column: ["train-1", "train-2"], "feature": [0.0, 1.0]}).to_csv(
        public / "train.csv", index=False
    )
    pd.DataFrame({id_column: ids, "feature": list(range(len(ids)))}).to_csv(
        public / "test.csv", index=False
    )
    pd.DataFrame({id_column: ids, prediction_column: [0.0] * len(ids)}).to_csv(
        public / "sample_submission.csv", index=False
    )
    (public / "description.md").write_text("Synthetic scalar task\n", encoding="utf-8")
    pd.DataFrame({id_column: ids, prediction_column: labels}).to_csv(
        private / "test.csv", index=False
    )
    return tmp_path / "datasets"


def test_prepare_physically_separates_train_and_hidden_label_views(tmp_path):
    split_root = _prepared(tmp_path)
    assert _prepared(tmp_path) == split_root
    train_view = split_root / "train_view"
    evaluator_view = split_root / "evaluator_view"
    train_manifest_path = train_view / "fixed_holdout_manifest.json"
    train_manifest = validate_train_view(train_manifest_path, train_view / "input")

    assert train_manifest["hidden_labels_present"] is False
    assert train_manifest["selection_policy"] == "terminal_only"
    assert "labels_file" not in train_manifest
    assert "labels_sha256" not in train_manifest
    assert not any(path.name == "labels.csv" for path in train_view.rglob("*"))
    assert not any(
        "must not escape" in path.read_text(encoding="utf-8", errors="ignore")
        for path in train_view.rglob("*")
        if path.is_file()
    )
    hidden = pd.read_csv(evaluator_view / "labels.csv")
    assert list(hidden.columns) == ["id", "EAP", "HPL", "MWS"]


def test_train_view_hash_is_fail_closed(tmp_path):
    split_root = _prepared(tmp_path)
    train_view = split_root / "train_view"
    manifest_path = train_view / "fixed_holdout_manifest.json"
    (train_view / "input" / "description.md").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable manifest"):
        validate_train_view(manifest_path, train_view / "input")


def test_existing_split_rejects_changed_private_labels(tmp_path):
    dataset_root = _source_dataset(tmp_path)
    output_root = tmp_path / "fixed"
    prepare_task(
        dataset_root,
        "spooky-author-identification",
        output_root,
        copy_mode="copy",
    )
    private = (
        dataset_root
        / "spooky-author-identification"
        / "prepared"
        / "private"
        / "test.csv"
    )
    labels = pd.read_csv(private)
    labels.loc[0, ["EAP", "HPL"]] = [0, 1]
    labels.to_csv(private, index=False)
    with pytest.raises(ValueError, match="do not match current private source"):
        prepare_task(
            dataset_root,
            "spooky-author-identification",
            output_root,
            copy_mode="copy",
        )


def test_label_free_submission_validation_checks_exact_ids_and_columns(tmp_path):
    split_root = _prepared(tmp_path)
    manifest = split_root / "train_view" / "fixed_holdout_manifest.json"
    good = tmp_path / "good.csv"
    pd.DataFrame(
        {
            "id": ["holdout-1", "holdout-2"],
            "EAP": [0.8, 0.1],
            "HPL": [0.1, 0.8],
            "MWS": [0.1, 0.1],
        }
    ).to_csv(good, index=False)
    assert validate_submission(manifest, good) == (True, "valid")

    reordered = tmp_path / "reordered.csv"
    pd.read_csv(good).iloc[::-1].to_csv(reordered, index=False)
    valid, reason = validate_submission(manifest, reordered)
    assert valid is False
    assert "IDs or row order" in reason


def test_hidden_evaluator_computes_multiclass_log_loss(tmp_path):
    split_root = _prepared(tmp_path)
    submission = tmp_path / "submission_node-a.csv"
    pd.DataFrame(
        {
            "id": ["holdout-1", "holdout-2"],
            "EAP": [0.8, 0.1],
            "HPL": [0.1, 0.8],
            "MWS": [0.1, 0.1],
        }
    ).to_csv(submission, index=False)
    result = evaluate_submission(
        split_root / "evaluator_view" / "fixed_holdout_manifest.json",
        submission,
    )
    assert result["score"] == pytest.approx(-math.log(0.8))
    assert result["maximize"] is False
    assert result["selection_policy"] == "terminal_only"


def test_binary_auc_task_uses_the_same_label_isolated_adapter(tmp_path):
    task_id = "aerial-cactus-identification"
    dataset_root = _write_scalar_task_source(
        tmp_path,
        task_id=task_id,
        id_column="id",
        prediction_column="has_cactus",
        labels=[0, 1, 0, 1],
    )
    split_root = prepare_task(dataset_root, task_id, tmp_path / "fixed", copy_mode="copy")
    submission = tmp_path / "submission_binary.csv"
    pd.DataFrame(
        {
            "id": [f"holdout-{index}" for index in range(4)],
            "has_cactus": [0.1, 0.9, 0.2, 0.8],
        }
    ).to_csv(submission, index=False)

    train_manifest = split_root / "train_view" / "fixed_holdout_manifest.json"
    assert validate_submission(train_manifest, submission) == (True, "valid")
    result = evaluate_submission(
        split_root / "evaluator_view" / "fixed_holdout_manifest.json",
        submission,
    )
    assert result["metric"] == "binary_roc_auc"
    assert result["score"] == pytest.approx(1.0)
    assert result["maximize"] is True


def test_regression_task_uses_the_same_label_isolated_adapter(tmp_path):
    task_id = "new-york-city-taxi-fare-prediction"
    dataset_root = _write_scalar_task_source(
        tmp_path,
        task_id=task_id,
        id_column="key",
        prediction_column="fare_amount",
        labels=[10.0, 20.0, 30.0],
    )
    split_root = prepare_task(dataset_root, task_id, tmp_path / "fixed", copy_mode="copy")
    submission = tmp_path / "submission_regression.csv"
    pd.DataFrame(
        {
            "key": [f"holdout-{index}" for index in range(3)],
            "fare_amount": [11.0, 19.0, 31.0],
        }
    ).to_csv(submission, index=False)

    train_manifest = split_root / "train_view" / "fixed_holdout_manifest.json"
    assert validate_submission(train_manifest, submission) == (True, "valid")
    result = evaluate_submission(
        split_root / "evaluator_view" / "fixed_holdout_manifest.json",
        submission,
    )
    assert result["metric"] == "rmse"
    assert result["score"] == pytest.approx(1.0)
    assert result["maximize"] is False


def test_terminal_scorer_ranks_all_submissions_without_mutating_search_metrics(tmp_path):
    split_root = _prepared(tmp_path)
    submissions = tmp_path / "submissions"
    submissions.mkdir()
    for node_id, probability in (("good", 0.9), ("bad", 0.4)):
        pd.DataFrame(
            {
                "id": ["holdout-1", "holdout-2"],
                "EAP": [probability, 1 - probability],
                "HPL": [1 - probability, probability],
                "MWS": [0.0, 0.0],
            }
        ).to_csv(submissions / f"submission_{node_id}.csv", index=False)
    journal = tmp_path / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": "good", "code": "print('good')", "metric": {"value": 9.0}},
                    {"id": "bad", "code": "print('bad')", "metric": {"value": 1.0}},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "fixed_holdout_scores.json"
    report = score_run(
        split_root / "evaluator_view" / "fixed_holdout_manifest.json",
        submissions,
        output,
        journal_path=journal,
    )
    assert report["best_node_id"] == "good"
    assert report["scores_were_visible_during_search"] is False
    assert all(
        item["internal_metric_disposition"] == "search_only"
        for item in report["results"]
    )
    assert (tmp_path / "best_solution_fixed_holdout.py").read_text() == "print('good')"


def test_fixed_mode_explicitly_replaces_internal_protocol_gates():
    cfg = SimpleNamespace(
        fixed_holdout=SimpleNamespace(
            enabled=True,
            bypass_protocol_gates=True,
        )
    )
    assert enabled(cfg) is True
    assert bypass_protocol_gates(cfg) is True
    agent = SimpleNamespace(cfg=cfg)
    assert requires_protocol_repair(agent, "candidate_replay") is False

    ordinary_agent = SimpleNamespace(
        cfg=SimpleNamespace(
            fixed_holdout=SimpleNamespace(
                enabled=False,
                bypass_protocol_gates=False,
            )
        )
    )
    assert requires_protocol_repair(ordinary_agent, "candidate_replay") is True

    clean_audit = {"status": "clean", "repair_required": False, "issues": []}
    validate_candidate_audit(
        {"known_issue_codes": ["historical_issue"]},
        clean_audit,
        external_holdout_mode=True,
    )
    with pytest.raises(ValueError, match="does not reproduce known issues"):
        validate_candidate_audit(
            {"known_issue_codes": ["historical_issue"]},
            clean_audit,
            external_holdout_mode=False,
        )


def test_fixed_holdout_config_resolves_nested_inheritance():
    cfg = _load_cfg(
        REPO / "mlevolve" / "config" / "config_run_forest_fixed_holdout.yaml",
        use_cli_args=False,
    )
    assert "extends" not in cfg
    assert cfg.fixed_holdout.enabled is True
    assert cfg.external_skill_memory.retrieval_control == "layered_strategy"
    assert cfg.agent.draft_role_policy.roles == [
        "coldstart_baseline",
        "memory_reproduction",
        "novel_exploration",
    ]
