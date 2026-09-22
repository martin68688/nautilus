"""Explicit CPA Chat Completions adapter. No Claude SDK or filesystem tools."""
import json
import os
import time
import urllib.request
from pathlib import Path
import shlex
import hashlib
import urllib.error

BASE_URL = 'https://apizh-ai.com/v1'


def configure(env_file, expected_base_url):
    file=Path(env_file).resolve(strict=True)
    if str(file).startswith('/workspace/') or file.stat().st_mode & 0o007:
        raise ValueError('endpoint credential file must be restricted on root volume')
    for line in file.read_text().splitlines():
        if line.startswith('OPENAI_API_KEY='):
            os.environ['OPENAI_API_KEY']=shlex.split(line.split('=',1)[1])[0]
    if not os.environ.get('OPENAI_API_KEY'):raise ValueError('no endpoint key')
    os.environ['OPENAI_BASE_URL']=expected_base_url


def complete(messages, deadline, model='gpt-5.6-terra', effort='low', expected_base_url=BASE_URL, raw_response_path=None):
    if os.environ.get('OPENAI_BASE_URL') != expected_base_url:
        raise ValueError('unexpected provider; no relay fallback')
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise ValueError('missing CPA client secret')
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError('end-to-end deadline before generation')
    payload = {'model': model, 'reasoning_effort': effort, 'messages': messages,
               'response_format': {'type': 'json_object'}, 'stream': False}
    request = urllib.request.Request(expected_base_url + '/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
                                                   'Authorization': 'Bearer ' + key})
    start = time.monotonic()
    def persist(body):
        if raw_response_path is not None:
            path=Path(raw_response_path)
            with path.open('xb') as f:
                os.chmod(path,0o600);f.write(body);f.flush();os.fsync(f.fileno())
    try:
        with urllib.request.urlopen(request, timeout=min(180, remaining)) as response:
            body=response.read()
    except urllib.error.HTTPError as error:
        persist(error.read())
        raise
    persist(body)
    raw = json.loads(body)
    choice = raw['choices'][0]
    if choice.get('finish_reason') != 'stop':
        raise ValueError('incomplete model generation')
    content = json.loads(choice['message']['content'])
    return content, {'model_requested': model, 'effort': effort,
                     'raw_response_sha256':hashlib.sha256(body).hexdigest(),
                     'model_returned': raw.get('model'), 'usage': raw.get('usage'),
                     'elapsed': time.monotonic() - start}
