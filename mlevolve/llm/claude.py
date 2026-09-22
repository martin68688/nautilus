"""Claude through an OpenAI-compatible relay (not the native Messages API).

Structured responses use a forced named tool because some relays silently
ignore response_format for Claude. The requested schema is checked locally.
No GPT reasoning_effort parameter is sent, and no model fallback is allowed.
"""
import json
import os
import time

import jsonschema
from openai import OpenAI

from .openai import _build_messages, _prompt_to_messages, _stage_config_for_model


def _client(stage, timeout=None):
    return OpenAI(api_key=stage.api_key, base_url=stage.base_url or None,
                  timeout=timeout if timeout is not None else float(os.environ.get('OPENAI_COMPAT_TIMEOUT', '180')),
                  max_retries=0)


def _tool_value(message, name, schema):
    calls = message.tool_calls or []
    if len(calls) != 1 or calls[0].function.name != name:
        raise ValueError('Claude must return exactly the requested tool: ' + name)
    value = json.loads(calls[0].function.arguments)
    jsonschema.Draft7Validator(schema).validate(value)
    return value


def _wire_schema(schema):
    # Relay corrupts nested tool objects into XML fragments. Carry each structured
    # top-level field as JSON text, then validate the original schema after decode.
    import copy
    wire = copy.deepcopy(schema)
    encoded = []
    for key, field in wire.get('properties', {}).items():
        if field.get('type') in ('object', 'array'):
            encoded.append(key)
            wire['properties'][key] = {'type': 'string', 'description':
                'A JSON-encoded value, never XML. Decode must satisfy: ' + json.dumps(field)}
    return wire, encoded


def _complete(messages, stage, *, name=None, schema=None, temperature=None,
              max_tokens=None, timeout=None):
    params = {'model': stage.model, 'messages': messages,
              'max_tokens': max_tokens if max_tokens is not None else 16384}
    if temperature is not None:
        params['temperature'] = temperature
    if schema is not None:
        jsonschema.Draft7Validator.check_schema(schema)
        wire_schema, encoded_fields = _wire_schema(schema)
        if encoded_fields:
            params['messages'] = list(messages) + [{'role': 'user', 'content': 'Transport requirement: tool fields ' + ', '.join(encoded_fields) + ' must be JSON-encoded strings, not nested objects or XML. Preserve the requested values exactly.'}]
        params['tools'] = [{'type': 'function', 'function': {
            'name': name, 'description': 'Return the requested structured result.',
            'parameters': wire_schema}}]
        params['tool_choice'] = {'type': 'function', 'function': {'name': name}}
    start = time.monotonic()
    response = _client(stage, timeout).chat.completions.create(**params)
    if not response.choices:
        raise ValueError('Claude returned no completion')
    choice = response.choices[0]
    if choice.finish_reason in ('length', 'content_filter'):
        raise ValueError('Claude response incomplete: ' + choice.finish_reason)
    output = (_tool_value(choice.message, name, wire_schema) if schema is not None
              else choice.message.content)
    if schema is not None:
        for key in encoded_fields:
            if key in output:
                output[key] = json.loads(output[key])
        jsonschema.Draft7Validator(schema).validate(output)
    if output is None:
        raise ValueError('Claude returned no text')
    usage = response.usage
    return output, time.monotonic() - start, getattr(usage, 'prompt_tokens', 0) or 0, getattr(usage, 'completion_tokens', 0) or 0, {
        'model': response.model, 'requested_model': stage.model,
        'transport': 'openai_compatible_claude', 'reasoning_effort': None}


def query(system_message, user_message, func_spec=None, cfg=None, **model_kwargs):
    if cfg is None:
        raise ValueError('cfg is required for Claude backend')
    stage = _stage_config_for_model(cfg, model_kwargs.get('model', ''))
    messages = _build_messages(system_message, user_message)
    if not messages:
        raise ValueError('a prompt is required')
    return _complete(messages, stage,
                     name=func_spec.name if func_spec else None,
                     schema=func_spec.json_schema if func_spec else None,
                     temperature=model_kwargs.get('temperature'),
                     max_tokens=model_kwargs.get('max_tokens'))


def generate(prompt, cfg, temperature=None, max_tokens=None, stop_tokens=None,
             json_schema=None, max_retries=20, retry_delay=3, request_timeout=None):
    stage = cfg.agent.code
    messages = _prompt_to_messages(prompt, model=stage.model)
    attempts = min(max_retries, max(1, int(os.environ.get('OPENAI_COMPAT_MAX_ATTEMPTS', '2'))))
    if attempts < 1:
        raise ValueError('max_retries must be positive')
    for attempt in range(attempts):
        try:
            if json_schema is not None:
                value = _complete(messages, stage, name='structured_output', schema=json_schema,
                                  temperature=temperature, max_tokens=max_tokens, timeout=request_timeout)[0]
                return json.dumps(value, ensure_ascii=False)
            params = {'model': stage.model, 'messages': messages, 'stream': True,
                      'max_tokens': max_tokens if max_tokens is not None else 16384}
            if temperature is not None:
                params['temperature'] = temperature
            if stop_tokens:
                params['stop'] = stop_tokens
            pieces = []
            finish = None
            with _client(stage, request_timeout).chat.completions.create(**params) as stream:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.delta.content:
                        pieces.append(choice.delta.content)
                    if choice.finish_reason:
                        finish = choice.finish_reason
            if finish != 'stop' or not pieces:
                raise ValueError('Claude stream incomplete: ' + str(finish))
            return ''.join(pieces)
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(retry_delay)
