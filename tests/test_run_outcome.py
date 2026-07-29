from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.run_outcome import classify_run_outcome, write_run_outcome


def test_step_target_is_complete_and_kubernetes_success_eligible() -> None:
    outcome = classify_run_outcome(
        completed_steps=80,
        total_steps=80,
        search_exhausted=False,
        has_certified_solution=True,
    )
    assert outcome["status"] == "complete"
    assert outcome["kubernetes_success_eligible"] is True


def test_exhausted_run_with_best_node_is_partial_not_success() -> None:
    outcome = classify_run_outcome(
        completed_steps=16,
        total_steps=80,
        search_exhausted=True,
        has_certified_solution=True,
    )
    assert outcome["status"] == "partial"
    assert outcome["reason"] == "search_space_exhausted_with_certified_solution"
    assert outcome["kubernetes_success_eligible"] is False


def test_exhausted_run_without_best_node_is_failed() -> None:
    outcome = classify_run_outcome(
        completed_steps=3,
        total_steps=80,
        search_exhausted=True,
        has_certified_solution=False,
    )
    assert outcome["status"] == "failed"
    assert outcome["kubernetes_success_eligible"] is False


def test_run_outcome_is_immutable(tmp_path: Path) -> None:
    outcome = classify_run_outcome(
        completed_steps=16,
        total_steps=80,
        search_exhausted=True,
        has_certified_solution=True,
    )
    path = write_run_outcome(tmp_path, outcome)
    assert json.loads(path.read_text())["status"] == "partial"
    assert write_run_outcome(tmp_path, outcome) == path

    changed = classify_run_outcome(
        completed_steps=80,
        total_steps=80,
        search_exhausted=False,
        has_certified_solution=True,
    )
    with pytest.raises(ValueError, match="immutable"):
        write_run_outcome(tmp_path, changed)
