#!/usr/bin/env python3
"""Stage A probe (zero API cost) — prove --max-tokens 8192 reaches the DeepSeek request.

Faithfully reproduces the EXACT request-construction path the evolver uses:
  ParallelSkillEvolver._call_llm builds ModelSettings(max_tokens=self.max_tokens)
  -> OpenAIClient.chat(): config = generation_config.copy(); config.update(settings.to_dict())
  -> _send_request_with_retry -> self._client.chat.completions.create(model=, messages=, **config)

We monkeypatch the OpenAI SDK ``.create`` to capture kwargs (no network), then assert:
  NEW (max_tokens=8192): captured create-kwargs contain max_tokens == 8192   (the fix)
  OLD (max_tokens=None):  captured create-kwargs do NOT contain max_tokens    (reproduces the bug)

Exit 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
T2S = REPO / "third_party" / "Trace2Skill"
sys.path.insert(0, str(T2S))

from src.react_agent.models import OpenAIClient, ModelSettings  # noqa: E402

captured: dict = {}


class _FakeResp:
    """Mimics openai ChatCompletion: .choices[0].message.content / .reasoning_content"""

    def __init__(self, content: str = "ok"):
        self.choices = [
            SimpleNamespace(message=SimpleNamespace(content=content, reasoning_content=""))
        ]


def _fake_create(**kwargs):
    captured.clear()
    captured.update(kwargs)
    return _FakeResp()


def make_client() -> OpenAIClient:
    # Same construction as run_parallel_combined_skill_evolution.py:319
    return OpenAIClient(
        model="deepseek-chat",
        api_key="EMPTY",
        base_url="http://localhost:1",  # never contacted: create() is mocked
        generation_config={"temperature": 0.6, "top_p": 0.95},  # deepseek_chat.json contents
        use_cache=False,
    )


def run_case(max_tokens, label):
    client = make_client()
    client._client.chat.completions.create = _fake_create  # mock the SDK call
    settings = ModelSettings(temperature=0.2, max_tokens=max_tokens)  # what _call_llm builds
    client.chat([SimpleNamespace(role="user", content="ping")], settings)
    has = "max_tokens" in captured
    val = captured.get("max_tokens")
    status = "ABSENT (bug reproduced)" if not has else f"present = {val}"
    print(f"  [{label}] ModelSettings(max_tokens={max_tokens!r:>5}) -> create() {status}")
    return has, val


def main():
    print("Stage A: max_tokens request-shape probe (no network)\n")

    print("1) ModelSettings.to_dict() (the gate at models.py:72 `if self.max_tokens:`):")
    new_d = ModelSettings(temperature=0.2, max_tokens=8192).to_dict()
    old_d = ModelSettings(temperature=0.6, max_tokens=None).to_dict()
    print(f"   NEW (8192): to_dict()['max_tokens'] = {new_d.get('max_tokens', '<<OMITTED>>')}")
    print(f"   OLD (None) : to_dict()['max_tokens'] = {old_d.get('max_tokens', '<<OMITTED — this is the bug>>')}")
    print()

    print("2) Full OpenAIClient.chat -> create(**config) kwargs capture:")
    new_has, new_val = run_case(8192, "NEW fix")
    old_has, _ = run_case(None, "OLD bug")
    print()

    fix_ok = new_has and new_val == 8192
    bug_shown = not old_has
    ok = fix_ok and bug_shown
    print("=" * 64)
    print("PASS ✅  fix lands max_tokens=8192 in the request" if ok else "FAIL ❌")
    print(f"   fix sends max_tokens=8192 to API : {'yes' if fix_ok else 'NO'}")
    print(f"   old config omits max_tokens (bug): {'confirmed' if bug_shown else 'NO (unexpected)'}")
    print("=" * 64)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
