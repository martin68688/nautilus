from __future__ import annotations

from types import SimpleNamespace

from config import _install_task_training_filename_compatibility


def test_taxi_workspace_adds_legacy_training_alias(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    train = input_dir / "train.csv"
    train.write_text("fare_amount\n12.5\n", encoding="utf-8")
    cfg = SimpleNamespace(
        exp_id="new-york-city-taxi-fare-prediction",
        workspace_dir=tmp_path,
    )

    _install_task_training_filename_compatibility(cfg)

    legacy = input_dir / "labels.csv"
    assert legacy.is_symlink()
    assert legacy.read_bytes() == train.read_bytes()


def test_non_taxi_workspace_is_unchanged(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "train.csv").write_text("label\n1\n", encoding="utf-8")
    cfg = SimpleNamespace(exp_id="leaf-classification", workspace_dir=tmp_path)

    _install_task_training_filename_compatibility(cfg)

    assert not (input_dir / "labels.csv").exists()
