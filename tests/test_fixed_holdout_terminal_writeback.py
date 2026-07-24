from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from authority.ledger import AuthorityLedger
from authority.memory_snapshot import MemorySnapshotLoader
from fixed_holdout.handoff import write_evaluation_request
from fixed_holdout.score_run import score_run
from fixed_holdout.writeback import (
    TerminalWritebackError,
    finalize_result_writeback,
)
from tests.authority.test_mlevolve_adapter import fake_agent
from tests.test_fixed_holdout import _prepared
from tests.test_memory_snapshot_overlay import build_tiny_bundle, write_current


def _payload_hash(payload: dict, field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _setup(tmp_path: Path, *, selected_node_id: str = "good"):
    split_root = _prepared(tmp_path)
    memory_root = tmp_path / "memory"
    bundle, manifest = build_tiny_bundle(memory_root)
    write_current(memory_root, bundle, manifest)
    log_dir = tmp_path / "run" / "logs"
    workspace_dir = tmp_path / "run" / "workspace"
    submission_dir = workspace_dir / "submission"
    submission_dir.mkdir(parents=True)
    agent = fake_agent(log_dir, mode="enforce")
    agent.cfg.exp_id = "spooky-author-identification"
    agent.cfg.exp_name = "fixed-holdout-run"
    agent.cfg.workspace_dir = workspace_dir
    agent.cfg.fixed_holdout = type(
        "FixedHoldoutConfig",
        (),
        {
            "enabled": True,
            "evaluation_mode": "terminal_only",
            "bypass_protocol_gates": True,
            "internal_metric_disposition": "search_only",
            "train_manifest_path": str(
                split_root
                / "train_view"
                / "fixed_holdout_manifest.json"
            ),
        },
    )()
    adapter = MLEvolveAuthorityAdapter(agent)
    snapshot = MemorySnapshotLoader(memory_root).load(
        session_overlay_path=log_dir / "session_overlay",
        active_protocol_ref=adapter.active_protocol.key(),
        authority_policy_version=adapter.engine.policy_version,
    )
    adapter.configure_memory_snapshot(snapshot)
    agent.evaluation_authority = adapter
    journal_path = log_dir / "journal.json"
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    journal_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": "good",
                        "stage": "improve",
                        "code": "print('good')",
                        "exec_time": 1.0,
                        "is_buggy": False,
                        "is_valid": True,
                        "metric": {"value": 9.0},
                        "actuation_report_refs": [],
                    },
                    {
                        "id": "bad",
                        "stage": "improve",
                        "code": "print('bad')",
                        "exec_time": 1.0,
                        "is_buggy": False,
                        "is_valid": True,
                        "metric": {"value": 1.0},
                        "actuation_report_refs": [],
                    },
                ]
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    for node_id, probability in (("good", 0.9), ("bad", 0.4)):
        pd.DataFrame(
            {
                "id": ["holdout-1", "holdout-2"],
                "EAP": [probability, 1 - probability],
                "HPL": [1 - probability, probability],
                "MWS": [0.0, 0.0],
            }
        ).to_csv(submission_dir / f"submission_{node_id}.csv", index=False)
    request_path = write_evaluation_request(
        agent.cfg,
        journal_path,
        authority=adapter,
        selected_node_id=selected_node_id,
    )
    assert request_path is not None
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert request["request_schema"] == "fixed_holdout_evaluation_request_v3"
    assert request["selected_node_id"] == selected_node_id
    assert request["selection_frozen_before_terminal_evaluation"] is True
    assert request["selection_basis"]["metric_disposition"] == "search_only"
    assert request["selection_basis"]["terminal_metric_observed"] is False
    assert request["selection_basis"]["formal_rank_claim_authorized"] is False
    assert request["selection_basis"]["source_score_inherited"] is False
    return {
        "split_root": split_root,
        "agent": agent,
        "adapter": adapter,
        "snapshot": snapshot,
        "journal_path": journal_path,
        "submission_dir": submission_dir,
        "request_path": request_path,
        "score_path": log_dir / "fixed_holdout_scores.json",
    }


def test_terminal_scorer_writes_one_idempotent_result_fact(tmp_path: Path) -> None:
    fixture = _setup(tmp_path)
    assert fixture["snapshot"].session_overlay.events() == []

    report = score_run(
        fixture["split_root"]
        / "evaluator_view"
        / "fixed_holdout_manifest.json",
        fixture["submission_dir"],
        fixture["score_path"],
        journal_path=fixture["journal_path"],
        evaluation_request_path=fixture["request_path"],
        finalize_writeback=True,
    )

    assert report["selected_node_id"] == "good"
    assert report["oracle_best_node_id"] == "good"
    assert report["system_selection_used_terminal_labels"] is False
    assert report["terminal_score_sealed"] is True
    events = fixture["snapshot"].session_overlay.events()
    assert len(events) == 1
    payload = events[0].payload
    assert events[0].event_type == "memory_claim"
    assert payload["publication_class"] == "result_fact"
    assert payload["artifact_id"] == "good"
    assert payload["derived_from_refs"] == []
    assert payload["verified_adoption_report_refs"] == []
    assert payload["artifact_pointer"]["node_id"] == "good"
    assert len(payload["artifact_pointer"]["journal_sha256"]) == 64

    repeated = finalize_result_writeback(
        fixture["request_path"],
        fixture["score_path"],
        fixture["split_root"]
        / "evaluator_view"
        / "fixed_holdout_manifest.json",
    )
    assert repeated["status"] == "complete"
    assert repeated["completion"] == "already_finalized"
    assert len(fixture["snapshot"].session_overlay.events()) == 1
    descriptor = json.loads(
        fixture["request_path"].read_text(encoding="utf-8")
    )["authority_writeback"]
    assert AuthorityLedger(descriptor["authority_ledger_path"]).verify()


def test_terminal_oracle_cannot_replace_preselected_system_node(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path, selected_node_id="bad")

    report = score_run(
        fixture["split_root"]
        / "evaluator_view"
        / "fixed_holdout_manifest.json",
        fixture["submission_dir"],
        fixture["score_path"],
        journal_path=fixture["journal_path"],
        evaluation_request_path=fixture["request_path"],
        finalize_writeback=True,
    )

    assert report["selected_node_id"] == "bad"
    assert report["oracle_best_node_id"] == "good"
    assert report["selected_score"] > report["oracle_best_score"]
    assert (
        fixture["score_path"].with_name(
            "selected_solution_fixed_holdout.py"
        ).read_text(encoding="utf-8")
        == "print('bad')"
    )
    events = fixture["snapshot"].session_overlay.events()
    assert len(events) == 1
    payload = events[0].payload
    assert payload["artifact_id"] == "bad"
    assert payload["score"] == report["selected_score"]
    assert payload["derived_from_refs"] == []


def test_terminal_writeback_failure_is_explicit_and_writes_no_result(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    journal = json.loads(fixture["journal_path"].read_text(encoding="utf-8"))
    journal["nodes"][0]["code"] = "print('tampered after handoff')"
    fixture["journal_path"].write_text(
        json.dumps(journal, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(TerminalWritebackError, match="Journal changed"):
        score_run(
            fixture["split_root"]
            / "evaluator_view"
            / "fixed_holdout_manifest.json",
            fixture["submission_dir"],
            fixture["score_path"],
            journal_path=fixture["journal_path"],
            evaluation_request_path=fixture["request_path"],
            finalize_writeback=True,
        )

    status_path = fixture["score_path"].with_name(
        "fixed_holdout_writeback_status.json"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "writeback_incomplete"
    assert "Journal changed" in status["reason"]
    assert fixture["snapshot"].session_overlay.events() == []


def test_terminal_scorer_rejects_rehashed_request_without_selection_freeze(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    request = json.loads(fixture["request_path"].read_text(encoding="utf-8"))
    request["selection_frozen_before_terminal_evaluation"] = False
    request["request_hash"] = _payload_hash(request, "request_hash")
    fixture["request_path"].write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TerminalWritebackError, match="not frozen"):
        score_run(
            fixture["split_root"]
            / "evaluator_view"
            / "fixed_holdout_manifest.json",
            fixture["submission_dir"],
            fixture["score_path"],
            journal_path=fixture["journal_path"],
            evaluation_request_path=fixture["request_path"],
            finalize_writeback=True,
        )

    assert not fixture["score_path"].exists()
    status = json.loads(
        fixture["score_path"]
        .with_name("fixed_holdout_writeback_status.json")
        .read_text(encoding="utf-8")
    )
    assert status["status"] == "writeback_incomplete"
    assert fixture["snapshot"].session_overlay.events() == []


def test_terminal_writeback_rejects_request_replacement_after_scoring(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    score_run(
        fixture["split_root"]
        / "evaluator_view"
        / "fixed_holdout_manifest.json",
        fixture["submission_dir"],
        fixture["score_path"],
        journal_path=fixture["journal_path"],
        evaluation_request_path=fixture["request_path"],
    )
    request = json.loads(fixture["request_path"].read_text(encoding="utf-8"))
    request["selection_basis"] = {"type": "post_hoc_replacement"}
    request["request_hash"] = _payload_hash(request, "request_hash")
    fixture["request_path"].write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TerminalWritebackError, match="not bound"):
        finalize_result_writeback(
            fixture["request_path"],
            fixture["score_path"],
            fixture["split_root"]
            / "evaluator_view"
            / "fixed_holdout_manifest.json",
        )

    assert fixture["snapshot"].session_overlay.events() == []


def test_terminal_scorer_rejects_train_manifest_changed_after_handoff(
    tmp_path: Path,
) -> None:
    fixture = _setup(tmp_path)
    train_manifest_path = (
        fixture["split_root"]
        / "train_view"
        / "fixed_holdout_manifest.json"
    )
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    train_manifest["post_handoff_mutation"] = True
    train_manifest_path.write_text(
        json.dumps(train_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(TerminalWritebackError, match="Train manifest changed"):
        score_run(
            fixture["split_root"]
            / "evaluator_view"
            / "fixed_holdout_manifest.json",
            fixture["submission_dir"],
            fixture["score_path"],
            journal_path=fixture["journal_path"],
            evaluation_request_path=fixture["request_path"],
            finalize_writeback=True,
        )

    assert not fixture["score_path"].exists()
    assert fixture["snapshot"].session_overlay.events() == []
