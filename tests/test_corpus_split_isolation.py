from __future__ import annotations

import pytest

from tests.memory_bundle_helpers import prepare_audit_and_splits, prepare_corpus
from build_split_manifests import build_same_domain_task_heldout_split


def test_full_seed_and_task_splits_are_deterministic_and_disjoint(tmp_path) -> None:
    corpus = prepare_corpus(tmp_path)
    outputs = prepare_audit_and_splits(tmp_path, corpus)
    full = outputs["splits"]["full"]
    seed = outputs["splits"]["seed-heldout"]
    task = outputs["splits"]["task-heldout"]
    same_domain = outputs["splits"]["same-domain-task-heldout"]

    assert len(full.source_run_ids) == 12
    assert full.heldout_run_ids == []
    assert all("spooky" not in run_id for run_id in full.source_run_ids)

    assert len(seed.source_run_ids) == 8
    assert len(seed.heldout_run_ids) == 4
    assert set(seed.source_run_ids).isdisjoint(seed.heldout_run_ids)
    assert set(seed.source_seed_groups).isdisjoint(seed.heldout_seed_groups)
    assert seed.validation["seed_group_overlap_count"] == 0
    assert seed.validation["task_overlap_expected"] is True
    assert len(seed.validation["shared_task_ids"]) == 4

    assert len(task.source_run_ids) == 9
    assert len(task.heldout_run_ids) == 3
    assert len(task.heldout_task_ids) == 1
    assert set(task.source_run_ids).isdisjoint(task.heldout_run_ids)
    assert set(task.source_task_ids).isdisjoint(task.heldout_task_ids)
    assert task.validation["run_overlap_count"] == 0
    assert task.validation["task_overlap_count"] == 0
    assert task.validation["every_family_has_source_task"] is True

    assert same_domain.split_kind == "same-domain-task-heldout"
    assert same_domain.source_task_ids == ["task-b"]
    assert same_domain.heldout_task_ids == ["task-a"]
    assert len(same_domain.source_run_ids) == 3
    assert len(same_domain.heldout_run_ids) == 3
    assert set(same_domain.source_run_ids).isdisjoint(
        same_domain.heldout_run_ids
    )
    assert same_domain.allocation["target_domain"] == "family_a"
    assert same_domain.validation["transfer_design"] == (
        "same_domain_different_task_task_heldout"
    )
    assert same_domain.validation["target_task_fully_heldout"] is True
    assert same_domain.validation["target_task_absent_from_source"] is True
    assert same_domain.validation["all_sources_have_target_domain"] is True
    assert same_domain.validation["cross_domain_source_run_count"] == 0


def test_repeated_runs_with_the_same_seed_never_cross_seed_split(tmp_path) -> None:
    corpus = prepare_corpus(tmp_path)
    outputs = prepare_audit_and_splits(tmp_path, corpus)
    seed = outputs["splits"]["seed-heldout"]
    run_to_group = {}
    by_id = {run.run_id: run for run in corpus["manifest"].runs}
    for run_id in [*seed.source_run_ids, *seed.heldout_run_ids]:
        run = by_id[run_id]
        run_to_group[run_id] = f"{run.canonical_task_id}::{run.seed}"
    assert {
        run_to_group[run_id] for run_id in seed.source_run_ids
    }.isdisjoint({run_to_group[run_id] for run_id in seed.heldout_run_ids})


def test_same_domain_split_rejects_cross_domain_source_allowlist(tmp_path) -> None:
    corpus = prepare_corpus(tmp_path)
    with pytest.raises(ValueError, match="Cross-domain source tasks"):
        build_same_domain_task_heldout_split(
            corpus["manifest"],
            version="v1",
            created_at="2026-07-19T00:00:00Z",
            target_task_id="task-a",
            source_task_ids=["task-c"],
            target_task_family="family-a",
        )
