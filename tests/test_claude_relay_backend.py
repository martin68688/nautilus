"""Relay regressions: ignored JSON schema, truncation, and GPT-only parameters."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import jsonschema
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mlevolve'))
from llm import _provider
from llm import claude

SCHEMA = {'type': 'object', 'properties': {'ok': {'type': 'boolean'}},
          'required': ['ok'], 'additionalProperties': False}


def message(name='probe', args='{"ok":true}'):
    return NS(tool_calls=[NS(function=NS(name=name, arguments=args))])


def test_dispatch_preserves_existing_providers():
    assert _provider('claude-sonnet-4-6') == 'claude'
    assert _provider('gpt-5.6-terra') == 'openai'
    assert _provider('glm-5') == 'anthropic'
    assert _provider('gemini-pro') == 'gemini'


@pytest.mark.parametrize('response', [NS(tool_calls=[]), message('wrong'),
    message(args='{"ok":"true"}'), message(args='{"ok":true,"extra":1}'),
    message(args='```json\n{"ok":true}\n```')])
def test_rejects_missing_wrong_or_invalid_tool(response):
    with pytest.raises((ValueError, jsonschema.ValidationError)):
        claude._tool_value(response, 'probe', SCHEMA)


def test_forced_tool_does_not_send_gpt_effort(monkeypatch):
    monkeypatch.setenv('OPENAI_REASONING_EFFORT', 'low')
    response = NS(choices=[NS(finish_reason='tool_calls', message=message())],
                  usage=NS(prompt_tokens=2, completion_tokens=3), model='claude-sonnet-4-6')
    client = Mock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr(claude, '_client', lambda *args: client)
    result = claude._complete([{'role': 'user', 'content': 'ok'}],
                             NS(model='claude-sonnet-4-6'), name='probe', schema=SCHEMA)
    assert result[0] == {'ok': True}
    params = client.chat.completions.create.call_args.kwargs
    assert params['tool_choice'] == {'type': 'function', 'function': {'name': 'probe'}}
    assert not {'reasoning_effort', 'extra_body', 'response_format'} & params.keys()


def test_truncated_tool_is_not_accepted(monkeypatch):
    client = Mock()
    client.chat.completions.create.return_value = NS(choices=[NS(finish_reason='length', message=message())])
    monkeypatch.setattr(claude, '_client', lambda *args: client)
    with pytest.raises(ValueError, match='incomplete'):
        claude._complete([], NS(model='claude-sonnet-4-6'), name='probe', schema=SCHEMA)


def test_stream_disconnect_without_finish_is_rejected(monkeypatch):
    client = Mock()
    stream = Mock()
    stream.__enter__ = Mock(return_value=iter([NS(choices=[NS(delta=NS(content='partial'), finish_reason=None)])]))
    stream.__exit__ = Mock(return_value=False)
    client.chat.completions.create.return_value = stream
    monkeypatch.setattr(claude, '_client', lambda *args: client)
    cfg = NS(agent=NS(code=NS(model='claude-sonnet-4-6')))
    with pytest.raises(ValueError, match='incomplete'):
        claude.generate('test', cfg, max_retries=1)


def test_schema_failures_have_bounded_retries(monkeypatch):
    complete = Mock(side_effect=ValueError('invalid schema response'))
    monkeypatch.setattr(claude, '_complete', complete)
    monkeypatch.setenv('OPENAI_COMPAT_MAX_ATTEMPTS', '2')
    cfg = NS(agent=NS(code=NS(model='claude-sonnet-4-6')))
    with pytest.raises(ValueError):
        claude.generate('test', cfg, json_schema=SCHEMA, retry_delay=0, max_retries=20)
    assert complete.call_count == 2
