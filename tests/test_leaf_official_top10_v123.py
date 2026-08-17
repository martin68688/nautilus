from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = (
    ROOT
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "kaggle_leaf_official"
    / "top10_v123_catalog.py"
)


def _catalog():
    spec = importlib.util.spec_from_file_location("top10_v123_catalog", CATALOG_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_top10_slots_deduplicate_to_seven_locked_sources() -> None:
    catalog = _catalog()
    assert len(catalog.TOP10_SLOTS) == 10
    assert len(set(catalog.TOP10_SLOTS)) == 7
    assert len(catalog.CANDIDATES) == 7
    assert set(catalog.TOP10_SLOTS) == set(catalog.BY_ID)
    assert len(catalog.EXPECTED_CODE_SHA256) == 7
    assert len(set(catalog.EXPECTED_CODE_SHA256.values())) == 7


def test_every_candidate_binds_one_explicit_official_variant() -> None:
    for candidate in _catalog().CANDIDATES:
        variant = candidate["official_submission_variant"]
        assert list(candidate["array_variants"]) == [variant]
        assert len(candidate["expected_code_sha256"]) == 64
        assert candidate["memory_disposition_before_official_score"]


def test_leaf_official_profiles_disable_fixed_holdout() -> None:
    paths = [
        ROOT / "mlevolve" / "config" / "config_leaf_official.yaml",
        (
            ROOT
            / "experiments"
            / "end2end_memory_systems_20260804"
            / "systems_v123"
            / "dynamic_hybrid.yaml"
        ),
    ]
    for path in paths:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert value["fixed_holdout"]["enabled"] is False
        assert value["official_submission"]["enabled"] is True
        assert value["official_submission"]["competition"] == "leaf-classification"
        assert value["official_submission"]["prediction_kind"] == (
            "multiclass_probability"
        )
