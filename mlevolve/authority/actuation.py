from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


class ActuationLevel(IntEnum):
    EXPOSED = 0
    CLAIMED_ADOPTION = 1
    STATIC_CONFORMANT = 2
    RUNTIME_CONFORMANT = 3
    CAUSAL_CONFIRMED = 4
    EFFECTIVE = 5


@dataclass
class Predicate:
    name: str
    expected: Any = True


@dataclass
class ExperienceContract:
    preconditions: list[Predicate] = field(default_factory=list)
    must_preserve: list[Predicate] = field(default_factory=list)
    must_change: list[Predicate] = field(default_factory=list)
    must_not_use: list[Predicate] = field(default_factory=list)
    expected_runtime_observations: list[Predicate] = field(default_factory=list)


def evaluate_predicates(predicates: list[Predicate], observations: dict[str, Any]) -> list[str]:
    return [item.name for item in predicates if observations.get(item.name) != item.expected]


def classify_actuation(
    contract: ExperienceContract,
    *,
    exposed: bool,
    claimed: bool,
    static_observations: dict[str, Any] | None = None,
    runtime_observations: dict[str, Any] | None = None,
    counterfactual_effect: float | None = None,
) -> ActuationLevel:
    if not exposed:
        return ActuationLevel.EXPOSED
    if not claimed:
        return ActuationLevel.EXPOSED
    level = ActuationLevel.CLAIMED_ADOPTION
    static_observations = static_observations or {}
    static_checks = contract.must_preserve + contract.must_change + contract.must_not_use
    if evaluate_predicates(static_checks, static_observations):
        return level
    level = ActuationLevel.STATIC_CONFORMANT
    if evaluate_predicates(contract.expected_runtime_observations, runtime_observations or {}):
        return level
    level = ActuationLevel.RUNTIME_CONFORMANT
    if counterfactual_effect is None:
        return level
    if counterfactual_effect == 0:
        return level
    return ActuationLevel.EFFECTIVE if counterfactual_effect > 0 else ActuationLevel.CAUSAL_CONFIRMED
