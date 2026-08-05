from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from agents.result_parse_agent import _emit_first_protocol_valid_candidate


def _payload_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "event_hash"},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def test_first_protocol_valid_event_is_first_writer_wins(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "FIRST_PROTOCOL_VALID_CANDIDATE.json"
    monkeypatch.setenv("MLEVOLVE_FIRST_VALID_EVENT_PATH", str(path))
    monkeypatch.setenv("MLEVOLVE_CONDITION_STARTED_AT_NS", "1000000000")
    times = iter((2500000000, 3500000000))
    monkeypatch.setattr(
        "agents.result_parse_agent.time.time_ns", lambda: next(times)
    )

    _emit_first_protocol_valid_candidate(SimpleNamespace(id="candidate-first"))
    _emit_first_protocol_valid_candidate(SimpleNamespace(id="candidate-later"))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["node_id"] == "candidate-first"
    assert payload["event_time_ns"] == 2500000000
    assert payload["condition_started_at_ns"] == 1000000000
    assert payload["event_hash"] == _payload_hash(payload)


def test_first_protocol_valid_event_is_optional(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MLEVOLVE_FIRST_VALID_EVENT_PATH", raising=False)
    _emit_first_protocol_valid_candidate(SimpleNamespace(id="candidate"))
    assert not list(tmp_path.iterdir())
