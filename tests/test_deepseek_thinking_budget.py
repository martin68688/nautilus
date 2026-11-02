from pathlib import Path
import sys
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))


def _fake_client(captured, chunks):
    class FakeCompletions:
        def create(self, **params):
            captured.update(params)
            return iter(chunks)

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    return FakeClient


def _chunk(content, finish_reason=None):
    delta = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)])


def _cfg(model, base_url):
    stage = SimpleNamespace(model=model, api_key="test-key", base_url=base_url)
    return SimpleNamespace(agent=SimpleNamespace(code=stage, feedback=stage))


def test_deepseek_thinking_generate_adds_reasoning_headroom(monkeypatch):
    from llm import openai as backend

    captured = {}
    monkeypatch.setattr(backend, "OpenAI", _fake_client(captured, [_chunk("code")]))
    out = backend.generate(
        "write code", _cfg("deepseek-v4-pro", "https://api.deepseek.com"), max_tokens=12000
    )

    assert out == "code"
    assert captured["extra_body"]["thinking"] == {"type": "enabled"}
    assert captured["max_tokens"] == 12000 + backend.DEEPSEEK_THINKING_HEADROOM_TOKENS


def test_dedicated_deployment_generate_adds_reasoning_headroom(monkeypatch):
    from llm import openai as backend

    captured = {}
    monkeypatch.setattr(backend, "OpenAI", _fake_client(captured, [_chunk("code")]))
    backend.generate(
        "write code",
        _cfg("red-lobster-7636/ds-rl-test", "https://api.together.ai/v1"),
        max_tokens=12000,
    )

    assert captured["model"] == "red-lobster-7636/ds-rl-test"
    assert captured["extra_body"]["thinking"] == {"type": "enabled"}
    assert captured["max_tokens"] == 12000 + backend.DEEPSEEK_THINKING_HEADROOM_TOKENS


def test_non_thinking_generate_keeps_requested_max_tokens(monkeypatch):
    from llm import openai as backend

    captured = {}
    monkeypatch.setattr(backend, "OpenAI", _fake_client(captured, [_chunk("code")]))
    backend.generate(
        "write code", _cfg("gpt-5.6-sol", "https://gateway.example.test/v1"), max_tokens=12000
    )

    assert captured["max_tokens"] == 12000


def test_truncated_generate_is_logged(monkeypatch, caplog):
    from llm import openai as backend

    captured = {}
    monkeypatch.setattr(
        backend, "OpenAI", _fake_client(captured, [_chunk(None, finish_reason="length")])
    )
    with caplog.at_level("WARNING", logger="MLEvolve"):
        backend.generate(
            "write code", _cfg("deepseek-v4-pro", "https://api.deepseek.com"), max_tokens=100
        )

    assert "truncated by max_tokens" in caplog.text


def _client_answering(replies):
    calls = []

    class FakeCompletions:
        def create(self, **params):
            calls.append(params)
            return iter(replies[min(len(calls), len(replies)) - 1])

    class FakeClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    return FakeClient, calls


def test_headroom_follows_the_deployment_setting(monkeypatch):
    from llm import openai as backend

    captured = {}
    monkeypatch.setenv("OPENAI_THINKING_HEADROOM_TOKENS", "24000")
    monkeypatch.setattr(backend, "OpenAI", _fake_client(captured, [_chunk("code")]))
    backend.generate(
        "write code",
        _cfg("red-lobster-7636/ds-rl-test", "https://api.together.ai/v1"),
        max_tokens=16384,
    )

    assert captured["max_tokens"] == 40384


def test_reasoning_that_leaves_no_answer_is_tried_again(monkeypatch):
    from llm import openai as backend

    empty = [_chunk(None, finish_reason="length")]
    client, calls = _client_answering([empty, empty, [_chunk("code")]])
    monkeypatch.setattr(backend, "OpenAI", client)
    out = backend.generate(
        "write code", _cfg("deepseek-v4-pro", "https://api.deepseek.com"), max_tokens=100
    )

    assert out == "code"
    assert len(calls) == 3


def test_empty_answer_tries_stop_at_the_limit(monkeypatch):
    from llm import openai as backend

    client, calls = _client_answering([[_chunk(None, finish_reason="length")]])
    monkeypatch.setattr(backend, "OpenAI", client)
    out = backend.generate(
        "write code", _cfg("deepseek-v4-pro", "https://api.deepseek.com"), max_tokens=100
    )

    assert out == ""
    assert len(calls) == backend.EMPTY_ANSWER_RETRIES + 1


def test_cut_off_answer_with_text_is_returned(monkeypatch):
    from llm import openai as backend

    client, calls = _client_answering([[_chunk("part", finish_reason="length")]])
    monkeypatch.setattr(backend, "OpenAI", client)
    out = backend.generate(
        "write code", _cfg("deepseek-v4-pro", "https://api.deepseek.com"), max_tokens=100
    )

    assert out == "part"
    assert len(calls) == 1


def test_local_embedding_falls_back_to_cpu_without_cuda(monkeypatch):
    import types
    import torch

    moved = []

    class FakeST:
        def __init__(self, name):
            self.name = name

        def to(self, device):
            moved.append(device)
            return self

        def get_sentence_embedding_dimension(self):
            return 8

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from agents.memory.embedding_models import EmbeddingModel

    model = EmbeddingModel(model_type="local", model_name="bge", device="cuda")

    assert model.device == "cpu"
    assert moved == ["cpu"]
