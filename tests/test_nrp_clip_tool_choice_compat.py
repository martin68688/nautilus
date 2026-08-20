from pathlib import Path
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from llm.gemini import FunctionSpec  # noqa: E402


NRP_BASE_URL = (
    "http://cliproxyapi-haoming.ecepxie.svc.cluster.local:8317/v1"
)


def _spec() -> FunctionSpec:
    return FunctionSpec(
        name="emit_direction",
        description="Emit metric direction",
        json_schema={
            "type": "object",
            "properties": {"lower_is_better": {"type": "boolean"}},
            "required": ["lower_is_better"],
            "additionalProperties": False,
        },
    )


def test_private_nrp_endpoint_uses_openai_standard_named_tool_shape():
    assert _spec().openai_tool_choice_for_base_url(NRP_BASE_URL) == {
        "type": "function",
        "function": {"name": "emit_direction"},
    }


def test_other_gateways_retain_historical_top_level_named_tool_shape():
    assert _spec().openai_tool_choice_for_base_url(
        "https://gateway.example.test/v1"
    ) == {
        "type": "function",
        "name": "emit_direction",
    }


def test_openai_backend_dispatches_nrp_specific_named_tool_shape(monkeypatch):
    from llm import openai as backend

    captured = {}

    class FakeCompletions:
        def create(self, **params):
            captured.update(params)
            message = SimpleNamespace(
                content=None,
                tool_calls=[
                    SimpleNamespace(
                        function=SimpleNamespace(
                            name="emit_direction",
                            arguments='{"lower_is_better": true}',
                        )
                    )
                ],
            )
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=message, finish_reason="tool_calls")
                ],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                model="gpt-5.6-sol",
                created=1,
            )

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(backend, "OpenAI", FakeClient)
    stage = SimpleNamespace(
        model="gpt-5.6-sol",
        api_key="test-key",
        base_url=NRP_BASE_URL,
    )
    cfg = SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage))

    output, *_ = backend.query(
        system_message="judge",
        user_message=None,
        model="gpt-5.6-sol",
        temperature=0.0,
        max_tokens=100,
        func_spec=_spec(),
        cfg=cfg,
    )

    assert output == {"lower_is_better": True}
    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "emit_direction"},
    }
