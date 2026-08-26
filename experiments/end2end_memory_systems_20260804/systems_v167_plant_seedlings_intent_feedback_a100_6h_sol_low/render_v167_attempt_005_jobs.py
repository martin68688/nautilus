#!/usr/bin/env python3
"""Render the immutable r24/attempt-005 five-queue GPU race manifests."""

from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SOURCES = (
    HERE / "nrp_jobs_gpu_tournament_donorqueue_attempt_004.yaml",
    HERE / "nrp_jobs_gpu_tournament_donorqueue_attempt_004_l40_l40s.yaml",
)

REPLACEMENTS = (
    (
        "full-six-hour-donor-queue-fixed-four-seat-transfer-tournament",
        "full-six-hour-time-budget-terminal-fixed-four-seat-transfer-tournament",
    ),
    (
        "nautilus-exp-end2end-agent-v167-transfer-tournament-donorqueue-sol-low-r23",
        "nautilus-exp-end2end-agent-v167-transfer-tournament-terminalfix-sol-low-r24",
    ),
    (
        "5f8796c77e3416a50b1bfca4b2b8a2610b836a70ffc9cdc99adbd55534e97d3a",
        "1e0c569d7b2a57db0a24100b4c76bc276ea54c66a4382ddcd29b361f97845efa",
    ),
    (
        "3b73302dbb72b0f5a0c04e37f59d6e78839891eb3be2f063e2d8cd61e3b2ed7d",
        "3307ec73ea049e3b11a31d31bcba02fd14637ea98f1525a31f559195d636a9b5",
    ),
    (
        "7a3118d4ad57e56d8bd29ade151a4e07e77cae8d",
        "44d63dc45c45aeca6912a69011cced015b693531",
    ),
    ("mlevolve.ai/source-commit: 7a3118d4", "mlevolve.ai/source-commit: 44d63dc4"),
    ("attempt-004", "attempt-005"),
    ("donorqueue", "terminalfix"),
)

RUN_SHA_CHECK = (
    'test "$(sha256sum "$RUNTIME/mlevolve/run.py" | awk \'{print $1}\')" = '
    '"1e0c569d7b2a57db0a24100b4c76bc276ea54c66a4382ddcd29b361f97845efa"'
)
RUN_OUTCOME_SHA_CHECK = (
    'test "$(sha256sum "$RUNTIME/mlevolve/engine/run_outcome.py" | awk '
    "'{print $1}')\" = \"92e7e769d7260f76c402c0c9a3113a0658d322d1b09f48c18c868e879d643225\""
)
TRANSFER_CLAIM = (
    "printf '%s\\n' 3307ec73ea049e3b11a31d31bcba02fd14637ea98f1525a31f559195d636a9b5 "
    '>"$CLAIM_DIR/transfer_tournament_sha256"'
)
RUN_OUTCOME_CLAIM = (
    "printf '%s\\n' 92e7e769d7260f76c402c0c9a3113a0658d322d1b09f48c18c868e879d643225 "
    '>"$CLAIM_DIR/run_outcome_sha256"'
)


def render(source: Path) -> str:
    payload = source.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        if old not in payload:
            raise RuntimeError(f"expected source token missing in {source.name}: {old}")
        payload = payload.replace(old, new)
    if RUN_SHA_CHECK not in payload:
        raise RuntimeError(f"run SHA guard missing after render: {source.name}")
    payload = payload.replace(
        RUN_SHA_CHECK,
        RUN_SHA_CHECK + "\n                  " + RUN_OUTCOME_SHA_CHECK,
        1,
    )
    if TRANSFER_CLAIM not in payload:
        raise RuntimeError(f"transfer receipt guard missing after render: {source.name}")
    payload = payload.replace(
        TRANSFER_CLAIM,
        TRANSFER_CLAIM + "\n                  " + RUN_OUTCOME_CLAIM,
        1,
    )
    stale_tokens = (
        "attempt-004",
        "donorqueue",
        "sol-low-r23",
        "7a3118d4",
        "5f8796c77e3416a50b1bfca4b2b8a2610b836a70ffc9cdc99adbd55534e97d3a",
        "3b73302dbb72b0f5a0c04e37f59d6e78839891eb3be2f063e2d8cd61e3b2ed7d",
    )
    stale = [token for token in stale_tokens if token in payload]
    if stale:
        raise RuntimeError(f"stale tokens after render in {source.name}: {stale}")
    return payload.rstrip() + "\n"


def main() -> None:
    sys.stdout.write("---\n".join(render(source) for source in SOURCES))


if __name__ == "__main__":
    main()
