from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from llm.model_compat import (  # noqa: E402
    UnsupportedDeepSeekModel,
    deepseek_thinking_extra_body,
    resolve_model_name,
)


def test_legacy_chat_alias_migrates_and_preserves_non_thinking_mode():
    resolution = resolve_model_name(
        "deepseek-chat",
        base_url="https://api.deepseek.com",
    )

    assert resolution.effective_name == "deepseek-v4-flash"
    assert resolution.migrated is True
    assert deepseek_thinking_extra_body(resolution, use_thinking=True) == {
        "thinking": {"type": "disabled"}
    }


def test_legacy_reasoner_alias_migrates_and_preserves_thinking_mode():
    resolution = resolve_model_name(
        "deepseek-reasoner",
        base_url="https://api.deepseek.com/v1",
    )

    assert resolution.effective_name == "deepseek-v4-flash"
    assert deepseek_thinking_extra_body(resolution, use_thinking=False) == {
        "thinking": {"type": "enabled"}
    }


@pytest.mark.parametrize("model", ["deepseek-v4-flash", "deepseek-v4-pro"])
def test_current_models_pass_through_and_follow_call_mode(model):
    resolution = resolve_model_name(model, base_url="https://api.deepseek.com")

    assert resolution.effective_name == model
    assert resolution.migrated is False
    assert deepseek_thinking_extra_body(resolution, use_thinking=False) == {
        "thinking": {"type": "disabled"}
    }


def test_unknown_model_fails_before_request_on_official_endpoint():
    with pytest.raises(UnsupportedDeepSeekModel, match="deepseek-v4-typo"):
        resolve_model_name(
            "deepseek-v4-typo",
            base_url="https://api.deepseek.com",
        )


def test_custom_openai_compatible_endpoint_keeps_its_model_namespace():
    resolution = resolve_model_name(
        "private-deepseek-tuned",
        base_url="https://gateway.example.test/v1",
    )

    assert resolution.effective_name == "private-deepseek-tuned"
    assert deepseek_thinking_extra_body(resolution, use_thinking=True) == {}


def test_generate_sends_effective_model_and_legacy_thinking_mode(monkeypatch):
    from llm import openai as backend

    captured = {}

    class FakeCompletions:
        def create(self, **params):
            captured.update(params)
            return [
                SimpleNamespace(
                    choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))]
                )
            ]

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(backend, "OpenAI", FakeClient)
    stage = SimpleNamespace(
        model="deepseek-chat",
        api_key="test-key",
        base_url="https://api.deepseek.com",
    )
    cfg = SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage))

    result = backend.generate("hello", cfg, max_retries=1)

    assert result == "ok"
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}


def test_primary_config_defaults_to_gpt56sol_openai_compatible_endpoint():
    text = (REPO / "mlevolve" / "config" / "config.yaml").read_text(encoding="utf-8")

    assert text.count("${oc.env:OPENAI_MODEL, gpt-5.6-sol}") == 3
    assert text.count("${oc.env:OPENAI_BASE_URL, https://apizh.net/v1}") == 2
    assert text.count("${oc.env:OPENAI_API_KEY}") == 2
    assert "${oc.env:DEEPSEEK_MODEL" not in text
