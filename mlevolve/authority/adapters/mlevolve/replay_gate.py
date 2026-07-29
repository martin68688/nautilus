from __future__ import annotations

from typing import Any


def authorize_replay_source(
    agent: Any,
    *,
    artifact_id: str,
    code_sha256: str,
    audit: dict,
    source_run_id: str,
    repair_seed: bool = False,
    source_execution_verified: bool = False,
) -> bool:
    adapter = getattr(agent, "evaluation_authority", None)
    if adapter is None or adapter.mode == "off":
        return True
    decision = adapter.authorize_replay_source(
        artifact_id=artifact_id,
        code_sha256=code_sha256,
        audit=audit,
        source_run_id=source_run_id,
        repair_seed=repair_seed,
        source_execution_verified=source_execution_verified,
    )
    return adapter.permits(decision, legacy_allowed=True)
