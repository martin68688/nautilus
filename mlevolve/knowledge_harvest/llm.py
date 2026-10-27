"""JSON-mode chat calls with an on-disk cache keyed by the full request."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


class JsonLLM:
    def __init__(self, model: str, cache_dir: Path, max_tokens: int = 8192):
        self.model = model
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ.get("OPENAI_BASE_URL") or None,
            timeout=600.0,
        )

    def __call__(self, system: str, user: str) -> dict[str, Any]:
        key = hashlib.sha256(
            json.dumps([self.model, self.max_tokens, system, user]).encode("utf-8")
        ).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    # DeepSeek V4: plain JSON extraction, no reasoning pass.
                    extra_body={"thinking": {"type": "disabled"}},
                )
                result = json.loads(completion.choices[0].message.content or "")
                break
            except Exception as exc:  # network, rate limit, malformed JSON
                last_error = exc
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"LLM call failed: {last_error}") from last_error
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return result
