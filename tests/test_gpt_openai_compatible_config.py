from pathlib import Path
import sys

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from config import Config, _load_cfg  # noqa: E402
from llm import _provider  # noqa: E402
from llm.model_compat import deepseek_thinking_extra_body, resolve_model_name  # noqa: E402
from llm.model_profiles import get_profile, supports_json_schema  # noqa: E402


def test_gpt56sol_routes_to_openai_compatible_backend():
    assert _provider("gpt-5.6-sol") == "openai"
    resolution = resolve_model_name(
        "gpt-5.6-sol",
        base_url="https://apizh.net/v1",
    )
    assert resolution.effective_name == "gpt-5.6-sol"
    assert resolution.migrated is False
    assert deepseek_thinking_extra_body(resolution, use_thinking=True) == {}
    assert supports_json_schema(resolution.effective_name) is True
    assert get_profile(resolution.effective_name, use_thinking=True) == {
        "temperature": 1.0
    }


def test_primary_config_resolves_all_solver_roles_to_gpt56sol(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-compatible-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    cfg = _load_cfg(
        REPO / "mlevolve" / "config" / "config.yaml",
        use_cli_args=False,
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["agent"]["code"] == {
        "model": "gpt-5.6-sol",
        "temp": 1,
        "base_url": "https://gateway.example.test/v1",
        "api_key": "test-openai-compatible-key",
    }
    assert resolved["agent"]["feedback"] == resolved["agent"]["code"]
    assert resolved["external_skill_memory"]["memory_strategy_model"] == "gpt-5.6-sol"


def test_latest_leaf_config_resolves_every_llm_role_to_gpt56sol(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-compatible-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example.test/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-sol")
    cfg = _load_cfg(
        REPO
        / "experiments"
        / "end2end_memory_systems_20260804"
        / "systems_v129"
        / "dynamic_hybrid.yaml",
        use_cli_args=False,
    )
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert resolved["agent"]["code"]["model"] == "gpt-5.6-sol"
    assert resolved["agent"]["feedback"]["model"] == "gpt-5.6-sol"
    assert resolved["adoption_verifier"]["model"] == "gpt-5.6-sol"
    assert (
        resolved["external_skill_memory"]["memory_strategy_model"]
        == "gpt-5.6-sol"
    )
    assert (
        resolved["external_skill_memory"][
            "memory_strategy_json_normalization_model"
        ]
        == "gpt-5.6-sol"
    )
    assert "deepseek" not in str(resolved).lower()
    assert resolved["agent"]["check_data_leakage"] is False
    assert resolved["fixed_holdout"]["bypass_protocol_gates"] is True


def test_official_evaluator_overrides_merge_into_typed_config(monkeypatch):
    experiment = REPO / "experiments" / "end2end_memory_systems_20260804"
    sys.path.insert(0, str(experiment))
    try:
        from stage_leaf_official_evaluator_v127 import OFFICIAL_OVERRIDES
    finally:
        sys.path.pop(0)

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-compatible-key")
    cfg = _load_cfg(
        experiment / "systems_v129" / "dynamic_hybrid.yaml",
        use_cli_args=False,
    )
    cfg_with_cli = OmegaConf.merge(
        cfg,
        OmegaConf.from_dotlist(["exp_name=typed-merge-smoke"]),
        OmegaConf.from_dotlist(OFFICIAL_OVERRIDES),
    )
    merged = OmegaConf.merge(OmegaConf.structured(Config), cfg_with_cli)

    assert all(not override.endswith("=") for override in OFFICIAL_OVERRIDES)
    assert merged.fixed_holdout.enabled is False
    assert merged.fixed_holdout.train_manifest_path == ""
    assert merged.official_submission.enabled is True
