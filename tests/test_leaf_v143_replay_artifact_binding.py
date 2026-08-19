from pathlib import Path

import pytest

from experiments.end2end_memory_systems_20260804.smoke_load_leaf_llm_redistilled_memory_v10 import (
    resolve_replay_binding,
)


def _bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    (bundle / "reports").mkdir(parents=True)
    (bundle / "reports" / "leaf_official_replay_targets_v139.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (bundle / "run_artifacts").mkdir()
    return bundle


def _config(path: Path, *, targets: Path, runs_root: Path) -> None:
    path.write_text(
        "agent:\n"
        "  draft_role_policy:\n"
        f"    replay_targets_path: {targets}\n"
        f"    replay_runs_root: {runs_root}\n",
        encoding="utf-8",
    )


def test_replay_binding_is_pinned_to_bundle_run_artifacts(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    config = tmp_path / "dynamic.yaml"
    _config(
        config,
        targets=bundle / "reports" / "leaf_official_replay_targets_v139.json",
        runs_root=bundle / "run_artifacts",
    )

    policy = resolve_replay_binding(config, bundle)

    assert Path(policy.replay_targets_path) == (
        bundle / "reports" / "leaf_official_replay_targets_v139.json"
    ).resolve()
    assert Path(policy.replay_runs_root) == (bundle / "run_artifacts").resolve()


def test_replay_binding_rejects_legacy_runs_root(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    legacy = tmp_path / "nautilus" / "mlevolve" / "runs"
    legacy.mkdir(parents=True)
    config = tmp_path / "dynamic.yaml"
    _config(
        config,
        targets=bundle / "reports" / "leaf_official_replay_targets_v139.json",
        runs_root=legacy,
    )

    with pytest.raises(ValueError, match="journal root is not bound"):
        resolve_replay_binding(config, bundle)
