from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .protocol_registry import canonical_json


PAIRED_REPLAY_SCHEMA = "memory_on_off_paired_replay_v1"


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(_jsonable(value)).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReplayObservation:
    action: Any
    code: str
    outcome: float | None
    protocol_legal: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: ReplayObservation | Mapping[str, Any]) -> ReplayObservation:
        if isinstance(value, ReplayObservation):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("Paired replay executor must return a mapping or ReplayObservation")
        raw_outcome = value.get("outcome")
        outcome = None if raw_outcome is None else float(raw_outcome)
        if outcome is not None and not math.isfinite(outcome):
            raise ValueError("Paired replay outcome must be finite")
        return cls(
            action=copy.deepcopy(value.get("action")),
            code=str(value.get("code") or ""),
            outcome=outcome,
            protocol_legal=value.get("protocol_legal") is True,
            metadata=copy.deepcopy(dict(value.get("metadata") or {})),
        )

    @property
    def action_hash(self) -> str:
        return _sha256_json(self.action)

    @property
    def code_hash(self) -> str:
        return _sha256_text(self.code)


@dataclass
class PairedReplayResult:
    pair_id: str
    control_hash: str
    memory_payload_hash: str
    metric_direction: str
    memory_on_action_hash: str
    memory_off_action_hash: str
    memory_on_code_hash: str
    memory_off_code_hash: str
    memory_on_outcome: float | None
    memory_off_outcome: float | None
    influence_confirmed: bool
    outcome_delta: float | None
    protocol_legal: bool
    effective: bool
    execution_order: tuple[str, str] = ("memory_off", "memory_on")
    result_hash: str = ""
    schema: str = PAIRED_REPLAY_SCHEMA

    def finalize(self) -> PairedReplayResult:
        payload = self.as_dict()
        payload.pop("result_hash", None)
        self.result_hash = _sha256_json(payload)
        return self

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


class PairedReplayRunner:
    """Run a controlled memory-off / memory-on influence and efficacy pair.

    The same deep-copied control context is supplied to each arm. Memory is the
    only runner-controlled difference. Positive outcome deltas always mean
    improvement, independent of metric direction.
    """

    def __init__(
        self,
        executor: Callable[[dict[str, Any], bool, Any], ReplayObservation | Mapping[str, Any]],
    ) -> None:
        self.executor = executor

    def run(
        self,
        *,
        context: Mapping[str, Any],
        memory_payload: Any,
        metric_direction: str,
        pair_id: str = "",
    ) -> PairedReplayResult:
        direction = str(metric_direction).strip().lower()
        if direction not in {"maximize", "minimize"}:
            raise ValueError("metric_direction must be maximize or minimize")
        frozen_context = copy.deepcopy(dict(context))
        original_hash = _sha256_json(frozen_context)
        memory_payload_copy = copy.deepcopy(memory_payload)
        memory_payload_hash = _sha256_json(memory_payload_copy)

        off_context = copy.deepcopy(frozen_context)
        on_context = copy.deepcopy(frozen_context)
        off = ReplayObservation.from_value(
            self.executor(off_context, False, None)
        )
        on = ReplayObservation.from_value(
            self.executor(on_context, True, copy.deepcopy(memory_payload_copy))
        )

        # The executor may mutate its private arm copy, but it cannot mutate the
        # caller's control or change what the other arm received.
        if _sha256_json(dict(context)) != original_hash:
            raise RuntimeError("Paired replay executor mutated the caller control context")
        control_hash = _sha256_json(
            {
                "context": frozen_context,
                "memory_payload_hash": memory_payload_hash,
                "metric_direction": direction,
            }
        )
        resolved_pair_id = str(pair_id).strip() or f"pair::{control_hash[:24]}"
        influence = bool(
            on.action_hash != off.action_hash or on.code_hash != off.code_hash
        )
        protocol_legal = bool(on.protocol_legal and off.protocol_legal)
        delta: float | None = None
        if on.outcome is not None and off.outcome is not None:
            delta = (
                on.outcome - off.outcome
                if direction == "maximize"
                else off.outcome - on.outcome
            )
        effective = bool(
            influence and protocol_legal and delta is not None and delta > 0
        )
        return PairedReplayResult(
            pair_id=resolved_pair_id,
            control_hash=control_hash,
            memory_payload_hash=memory_payload_hash,
            metric_direction=direction,
            memory_on_action_hash=on.action_hash,
            memory_off_action_hash=off.action_hash,
            memory_on_code_hash=on.code_hash,
            memory_off_code_hash=off.code_hash,
            memory_on_outcome=on.outcome,
            memory_off_outcome=off.outcome,
            influence_confirmed=influence,
            outcome_delta=delta,
            protocol_legal=protocol_legal,
            effective=effective,
        ).finalize()


__all__ = [
    "PAIRED_REPLAY_SCHEMA",
    "PairedReplayResult",
    "PairedReplayRunner",
    "ReplayObservation",
]
