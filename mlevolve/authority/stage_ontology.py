from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class _StageEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class GenerationStage(_StageEnum):
    DRAFT = "draft"
    MODEL_DESIGN = "model_design"
    IMPROVE = "improve"
    DEBUG = "debug"
    EVOLUTION = "evolution"
    FUSION = "fusion"


class GovernanceStage(_StageEnum):
    RETRIEVAL = "retrieval"
    BRANCH_SELECTION = "branch_selection"
    MEMORY_WRITEBACK = "memory_writeback"
    DISTILLATION = "distillation"
    REPLAY = "replay"


@dataclass(frozen=True)
class StageAxes:
    generation_stage: GenerationStage
    governance_stage: GovernanceStage


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")


# SearchNode creation sites emit exactly these runtime stages. Each has one
# deterministic default mapping; governance callers can override only the
# governance axis while retaining the node's generation provenance.
_RUNTIME_STAGE_AXES: dict[str, StageAxes] = {
    "root": StageAxes(GenerationStage.DRAFT, GovernanceStage.RETRIEVAL),
    "draft": StageAxes(GenerationStage.DRAFT, GovernanceStage.RETRIEVAL),
    "model_design": StageAxes(GenerationStage.MODEL_DESIGN, GovernanceStage.RETRIEVAL),
    "improve": StageAxes(GenerationStage.IMPROVE, GovernanceStage.RETRIEVAL),
    "debug": StageAxes(GenerationStage.DEBUG, GovernanceStage.RETRIEVAL),
    "evolution": StageAxes(GenerationStage.EVOLUTION, GovernanceStage.RETRIEVAL),
    "fusion": StageAxes(GenerationStage.FUSION, GovernanceStage.BRANCH_SELECTION),
    "fusion_draft": StageAxes(GenerationStage.FUSION, GovernanceStage.RETRIEVAL),
}


# One-cycle compatibility mapping. New policy consumes the two canonical axes,
# never this legacy enum directly.
_LEGACY_STAGE_AXES: dict[str, StageAxes] = {
    "retrieval": StageAxes(GenerationStage.DRAFT, GovernanceStage.RETRIEVAL),
    "draft": StageAxes(GenerationStage.DRAFT, GovernanceStage.RETRIEVAL),
    "debug": StageAxes(GenerationStage.DEBUG, GovernanceStage.RETRIEVAL),
    "branch_selection": StageAxes(GenerationStage.IMPROVE, GovernanceStage.BRANCH_SELECTION),
    "fusion": StageAxes(GenerationStage.FUSION, GovernanceStage.BRANCH_SELECTION),
    "memory_writeback": StageAxes(GenerationStage.EVOLUTION, GovernanceStage.MEMORY_WRITEBACK),
    "distillation": StageAxes(GenerationStage.EVOLUTION, GovernanceStage.DISTILLATION),
    "replay": StageAxes(GenerationStage.DEBUG, GovernanceStage.REPLAY),
}


def runtime_stage_axes(runtime_stage: Any) -> StageAxes:
    key = _enum_value(runtime_stage)
    try:
        return _RUNTIME_STAGE_AXES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown runtime stage: {runtime_stage!r}") from exc


def legacy_stage_axes(legacy_stage: Any) -> StageAxes:
    key = _enum_value(legacy_stage)
    try:
        return _LEGACY_STAGE_AXES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown legacy decision stage: {legacy_stage!r}") from exc


def resolve_stage_axes(
    *,
    generation_stage: GenerationStage | str | None = None,
    governance_stage: GovernanceStage | str | None = None,
    runtime_stage: Any | None = None,
    legacy_stage: Any | None = None,
) -> StageAxes:
    """Resolve the canonical stage pair or fail closed on missing ontology."""

    runtime_axes = runtime_stage_axes(runtime_stage) if runtime_stage is not None else None
    legacy_axes = legacy_stage_axes(legacy_stage) if legacy_stage is not None else None

    if generation_stage is not None:
        generation = GenerationStage(_enum_value(generation_stage))
    elif runtime_axes is not None:
        generation = runtime_axes.generation_stage
    elif legacy_axes is not None:
        generation = legacy_axes.generation_stage
    else:
        raise ValueError("generation_stage is required when no runtime or legacy stage is supplied")

    if governance_stage is not None:
        governance = GovernanceStage(_enum_value(governance_stage))
    elif legacy_axes is not None:
        governance = legacy_axes.governance_stage
    elif runtime_axes is not None:
        governance = runtime_axes.governance_stage
    else:
        raise ValueError("governance_stage is required when no runtime or legacy stage is supplied")

    return StageAxes(generation, governance)


def legacy_decision_stage_value(axes: StageAxes) -> str:
    governance_legacy = {
        GovernanceStage.BRANCH_SELECTION: "branch_selection",
        GovernanceStage.MEMORY_WRITEBACK: "memory_writeback",
        GovernanceStage.DISTILLATION: "distillation",
        GovernanceStage.REPLAY: "replay",
    }
    if axes.governance_stage in governance_legacy:
        return governance_legacy[axes.governance_stage]
    generation_legacy = {
        GenerationStage.DRAFT: "draft",
        GenerationStage.DEBUG: "debug",
        GenerationStage.FUSION: "fusion",
    }
    return generation_legacy.get(axes.generation_stage, "retrieval")


def known_runtime_stage_mappings() -> dict[str, StageAxes]:
    return dict(_RUNTIME_STAGE_AXES)


__all__ = [
    "GenerationStage",
    "GovernanceStage",
    "StageAxes",
    "known_runtime_stage_mappings",
    "legacy_decision_stage_value",
    "legacy_stage_axes",
    "resolve_stage_axes",
    "runtime_stage_axes",
]
