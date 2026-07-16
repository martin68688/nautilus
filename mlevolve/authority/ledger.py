from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .protocol_registry import canonical_json

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None


class AuthorityLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._lock = threading.RLock()

    @contextmanager
    def _process_lock(self):
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    @staticmethod
    def _hash_event(event: dict[str, Any]) -> str:
        unsigned = dict(event)
        unsigned.pop("event_hash", None)
        return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            with self._process_lock():
                events = self.read()
                if events and not self._verify_events(events):
                    raise ValueError(f"Refusing to append to invalid authority ledger: {self.path}")
                parent = events[-1]["event_hash"] if events else ""
                event = {
                    "sequence": len(events),
                    "event_type": event_type,
                    "parent_event_hash": parent,
                    "payload": payload,
                }
                event["event_hash"] = self._hash_event(event)
                fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, (canonical_json(event) + "\n").encode("utf-8"))
                    os.fsync(fd)
                finally:
                    os.close(fd)
                return event

    def _verify_events(self, events: Iterable[dict[str, Any]]) -> bool:
        previous = ""
        for expected_sequence, event in enumerate(events):
            if event.get("sequence") != expected_sequence:
                return False
            if event.get("parent_event_hash") != previous:
                return False
            if event.get("event_hash") != self._hash_event(event):
                return False
            previous = event["event_hash"]
        return True

    def verify(self, events: Iterable[dict[str, Any]] | None = None) -> bool:
        if events is not None:
            return self._verify_events(events)
        with self._lock:
            with self._process_lock():
                return self._verify_events(self.read())
