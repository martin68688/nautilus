#!/usr/bin/env python3
"""Minimal OpenAI -> Anthropic proxy for the vendored Trace2Skill baseline.

``third_party/Trace2Skill`` speaks the OpenAI Chat Completions protocol, and we
deliberately do NOT edit its source — it is the faithful, paper-faithful baseline.
This tiny localhost proxy exposes ``POST /v1/chat/completions``, translates each
request to the Anthropic Messages API, and calls GLM via Zhipu's Coding Plan
endpoint (``https://open.bigmodel.cn/api/anthropic``). The response is reshaped
back into the OpenAI ``chat.completion`` JSON that Trace2Skill expects, so the
analysts + skill-evolver run on GLM-5.2 with zero source changes.

Run:
    python glm_proxy.py                       # listens on 127.0.0.1:18211
Then point Trace2Skill at it:
    OPENAI_BASE_URL=http://127.0.0.1:18211/v1  OPENAI_API_KEY=anything

(Non-streaming only — the Trace2Skill analysts + skill-evolver read
``choices[0].message.content``. A ``stream=true`` request is answered with a
single non-streamed completion, which is fine for this pipeline.)

Config (paper-skills/.env): GLM_API_KEY, GLM_BASE_URL, GLM_MODEL, GLM_PROXY_PORT.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import anthropic
from dotenv import load_dotenv

# GLM creds live in the gitignored mlevolve/.env; paper-skills/.env for the rest.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")
load_dotenv(Path(__file__).resolve().parents[2] / "mlevolve" / ".env")

GLM_API_KEY = os.getenv("GLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
GLM_BASE_URL = os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/anthropic")
DEFAULT_MODEL = os.getenv("GLM_MODEL", "glm-5.2")
PORT = int(os.getenv("GLM_PROXY_PORT", "18211"))

_CLIENT = anthropic.Anthropic(api_key=GLM_API_KEY, base_url=GLM_BASE_URL, timeout=1200.0)

_FINISH_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _content_to_text(content) -> str:
    """Flatten an OpenAI message `content` (str | list of parts) to a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def _translate_request(body: dict) -> dict:
    """OpenAI chat.completions request -> Anthropic messages.create params."""
    system_parts: list[str] = []
    out: list[dict] = []
    for m in body.get("messages", []):
        role = m.get("role")
        text = _content_to_text(m.get("content", ""))
        if role == "system":
            system_parts.append(text)
        else:
            out.append({"role": role or "user", "content": text})
    if not out:
        out.append({"role": "user", "content": "(proceed)"})

    params: dict = {
        "model": body.get("model") or DEFAULT_MODEL,
        "messages": out,
        # Default 8192 (not 4096) so callers that omit max_tokens (e.g. the
        # analysts) don't hit the old DeepSeek-style ~4k truncation.
        "max_tokens": body.get("max_tokens") or 8192,
    }
    if system_parts:
        params["system"] = "\n\n".join(system_parts)
    if body.get("temperature") is not None:
        params["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        params["top_p"] = body["top_p"]
    stop = body.get("stop")
    if stop:
        params["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    # OpenAI response_format -> Anthropic has no direct knob; instruct in system.
    rf = body.get("response_format")
    if isinstance(rf, dict):
        instr = "Respond with ONLY valid JSON (no markdown fences, no prose)."
        params["system"] = (params.get("system", "") + "\n\n" + instr).strip() if params.get("system") else instr
    return params


def _call_anthropic(params: dict) -> dict:
    # STREAM on the backend even though Trace2Skill sends non-streaming OpenAI
    # requests: GLM's Anthropic endpoint drops non-streaming connections for long
    # generations (~3 min), but streaming stays alive. We accumulate the stream
    # and return a normal (non-streamed) OpenAI chat.completion response.
    with _CLIENT.messages.stream(**params) as stream:
        text = ""
        for chunk in stream.text_stream:
            text += chunk
        resp = stream.get_final_message()
    if "</think>" in text:
        text = text[text.find("</think>") + 8:]
    finish = _FINISH_MAP.get(getattr(resp, "stop_reason", None), "stop")
    in_tok = getattr(resp.usage, "input_tokens", 0) or 0
    out_tok = getattr(resp.usage, "output_tokens", 0) or 0
    return {
        "id": getattr(resp, "id", "chatcmpl-proxy"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": getattr(resp, "model", params["model"]),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": in_tok,
            "completion_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        },
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.rstrip("/").endswith("/models"):
            self._send(200, {"object": "list", "data": [{"id": DEFAULT_MODEL, "object": "model"}]})
        else:
            self._send(200, {"status": "ok", "model": DEFAULT_MODEL})

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw.decode()) if raw else {}
        except Exception:
            self._send(400, {"error": {"message": "invalid JSON", "type": "invalid_request_error"}})
            return
        if "/chat/completions" not in path:
            self._send(404, {"error": {"message": f"unknown path {path}", "type": "invalid_request_error"}})
            return
        try:
            result = _call_anthropic(_translate_request(body))
            self._send(200, result)
        except anthropic.APIStatusError as e:
            self._send(502, {"error": {"message": str(getattr(e, "message", e)), "type": "api_error"}})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": {"message": repr(e), "type": "internal_error"}})

    def log_message(self, fmt, *args):
        sys.stderr.write("[glm-proxy] " + (fmt % args) + "\n")


def main():
    if not GLM_API_KEY:
        sys.exit("GLM_API_KEY not set (put it in paper-skills/.env)")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(
        f"[glm-proxy] OpenAI->Anthropic proxy on http://127.0.0.1:{PORT}/v1"
        f"  ->  {GLM_BASE_URL}  (model {DEFAULT_MODEL})",
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
