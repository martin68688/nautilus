"""Small cached HTTP client: every response is stored on disk, so a rerun
reads the same bytes and the corpus is reproducible."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

USER_AGENT = "mlevolve-knowledge-harvest/1.0 (research; contact via repository owner)"
_PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy")


class CachedHttp:
    def __init__(self, cache_dir: Path, delay: float = 0.4, retries: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.retries = retries
        self._last = 0.0
        # Stale shell proxies break direct HTTPS on this host; go direct.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for name in _PROXY_VARS:
            os.environ.pop(name, None)

    def _path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / digest

    def get_bytes(self, url: str) -> bytes:
        path = self._path(url)
        if path.exists():
            return path.read_bytes()
        last_error: Exception | None = None
        for attempt in range(self.retries):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with self._opener.open(request, timeout=60) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 404:
                    raise
                # Discourse answers 429 when polled too fast; back off.
                time.sleep(min(60.0, 5.0 * (attempt + 1)))
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                time.sleep(min(60.0, 5.0 * (attempt + 1)))
        else:
            raise RuntimeError(f"GET failed after {self.retries} attempts: {url}") from last_error
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(body)
        os.replace(tmp, path)
        return body

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_bytes(url).decode("utf-8"))
