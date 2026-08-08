"""Frozen memory-system policies for the exploratory End2End pilot.

The policies in this module operate on one common, Authority-filtered candidate
pool.  They are deliberately pure and deterministic: retrieval and Authority
remain the responsibility of :mod:`stage_aware_hybrid_memory`, while this
module decides which already-authorized items may enter the prompt.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Iterable, Protocol


SYSTEM_IDS = (
    "no_memory",
    "flat_retrieval",
    "sop_only",
    "runforest_only",
    "static_hybrid",
    "dynamic_hybrid",
    "reversed_router",
    "gome_style_port",
    "macla_style_port",
    "rcr_router_style_port",
)

STAGE_ALIASES = {
    "coldstart": "draft",
    "multi_fusion": "improve",
    "fusion_draft": "improve",
    "aggregation": "improve",
    "fusion": "improve",
    "evolution": "improve",
}


def canonical_stage(stage: str) -> str:
    value = str(stage or "").lower()
    return STAGE_ALIASES.get(value, value)


def whitespace_tokens(text: str) -> list[str]:
    """The preregistered, provider-independent token accounting rule."""

    return str(text or "").split()


@dataclass(frozen=True)
class MemoryCandidate:
    candidate_id: str
    source: str  # sop | runforest
    relevance: float
    prompt_text: str
    source_stage: str = ""
    source_task_id: str = ""
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("MemoryCandidate requires a stable candidate_id")
        if self.source not in {"sop", "runforest"}:
            raise ValueError(f"Unsupported memory candidate source: {self.source}")
        if not math.isfinite(float(self.relevance)):
            raise ValueError("MemoryCandidate relevance must be finite")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemorySystemContext:
    stage: str
    task_id: str
    task_description: str
    prompt_token_budget: int = 1536
    top_k: int = 6
    memory_bundle_manifest_sha256: str = ""

    def __post_init__(self) -> None:
        if canonical_stage(self.stage) not in {"draft", "improve", "debug"}:
            raise ValueError(f"Unsupported End2End stage: {self.stage}")
        if self.prompt_token_budget <= 0:
            raise ValueError("prompt_token_budget must be positive")
        if self.top_k != 6:
            raise ValueError("The frozen End2End comparison requires top_k=6")


@dataclass(frozen=True)
class MemorySelection:
    system_id: str
    raw_candidates: tuple[MemoryCandidate, ...]
    selected_candidates: tuple[MemoryCandidate, ...]
    suppressed_candidates: tuple[dict[str, Any], ...]
    prompt_candidates: tuple[dict[str, str], ...]
    prompt_candidate_ids: tuple[str, ...]
    prompt_text: str
    prompt_token_count: int
    prompt_truncated: bool
    route: dict[str, Any]

    def to_pack(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "raw_candidates": [item.to_dict() for item in self.raw_candidates],
            "selected_candidates": [
                item.to_dict() for item in self.selected_candidates
            ],
            "suppressed_candidates": [dict(item) for item in self.suppressed_candidates],
            "prompt_candidates": [dict(item) for item in self.prompt_candidates],
            "final_prompt_candidate_ids": list(self.prompt_candidate_ids),
            "prompt_text": self.prompt_text,
            "prompt_token_count": self.prompt_token_count,
            "prompt_truncated": self.prompt_truncated,
            "route": dict(self.route),
        }


class MemorySystem(Protocol):
    system_id: str

    def select(
        self,
        candidates: Iterable[MemoryCandidate],
        context: MemorySystemContext,
    ) -> MemorySelection:
        ...


def _ranked(candidates: Iterable[MemoryCandidate]) -> list[MemoryCandidate]:
    return sorted(
        candidates,
        key=lambda item: (-float(item.relevance), str(item.candidate_id)),
    )


def _take_by_source(
    candidates: list[MemoryCandidate], quotas: dict[str, int]
) -> list[MemoryCandidate]:
    # Keep each source quota independent, then restore deterministic global
    # order.  There is intentionally no cross-source backfill.
    selected: list[MemoryCandidate] = []
    for source in ("sop", "runforest"):
        selected.extend(
            [item for item in candidates if item.source == source][
                : int(quotas.get(source, 0))
            ]
        )
    return _ranked(selected)


class _BaseMemorySystem:
    system_id: ClassVar[str]
    prompt_label: ClassVar[str] = "Memory"
    limitation: ClassVar[str] = ""

    def _choose(
        self,
        candidates: list[MemoryCandidate],
        context: MemorySystemContext,
    ) -> tuple[list[MemoryCandidate], dict[str, Any]]:
        raise NotImplementedError

    def _format_candidate(self, candidate: MemoryCandidate) -> str:
        return (
            f"### {candidate.candidate_id} [{candidate.source}]\n"
            f"{candidate.prompt_text.strip()}"
        ).strip()

    def select(
        self,
        candidates: Iterable[MemoryCandidate],
        context: MemorySystemContext,
    ) -> MemorySelection:
        raw = _ranked(candidates)
        chosen, route = self._choose(raw, context)
        unique: dict[str, MemoryCandidate] = {}
        for item in chosen:
            unique.setdefault(item.candidate_id, item)
        chosen = list(unique.values())[: context.top_k]
        selected_ids = {item.candidate_id for item in chosen}
        suppressed = [
            {
                "candidate_id": item.candidate_id,
                "source": item.source,
                "reason": "not_selected_by_frozen_system_policy",
            }
            for item in raw
            if item.candidate_id not in selected_ids
        ]

        header = f"## End2End Memory: {self.prompt_label}"
        if self.limitation:
            header += f"\nPort scope: {self.limitation}"
        parts = [header]
        visible: list[str] = []
        prompt_candidates: list[dict[str, str]] = []
        truncated = False
        for candidate in chosen:
            card = self._format_candidate(candidate)
            prospective = "\n\n".join([*parts, card])
            if len(whitespace_tokens(prospective)) <= context.prompt_token_budget:
                parts.append(card)
                visible.append(candidate.candidate_id)
                prompt_candidates.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "source": candidate.source,
                        "source_stage": candidate.source_stage,
                        "source_task_id": candidate.source_task_id,
                        "prompt_text": card,
                    }
                )
                continue
            remaining = context.prompt_token_budget - len(
                whitespace_tokens("\n\n".join(parts))
            )
            if remaining > 0:
                clipped = whitespace_tokens(card)[:remaining]
                if clipped:
                    clipped_text = " ".join(clipped)
                    parts.append(clipped_text)
                    visible.append(candidate.candidate_id)
                    prompt_candidates.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "source": candidate.source,
                            "source_stage": candidate.source_stage,
                            "source_task_id": candidate.source_task_id,
                            "prompt_text": clipped_text,
                        }
                    )
            truncated = True
            break

        prompt = "\n\n".join(parts) if visible else ""
        visible_ids = set(visible)
        for candidate in chosen:
            if candidate.candidate_id not in visible_ids:
                suppressed.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "source": candidate.source,
                        "reason": "shared_prompt_token_budget",
                    }
                )
        route = {
            **route,
            "canonical_stage": canonical_stage(context.stage),
            "top_k": context.top_k,
            "prompt_token_budget": context.prompt_token_budget,
        }
        return MemorySelection(
            system_id=self.system_id,
            raw_candidates=tuple(raw),
            selected_candidates=tuple(chosen),
            suppressed_candidates=tuple(suppressed),
            prompt_candidates=tuple(prompt_candidates),
            prompt_candidate_ids=tuple(visible),
            prompt_text=prompt,
            prompt_token_count=len(whitespace_tokens(prompt)),
            prompt_truncated=truncated,
            route=route,
        )


class NoMemorySystem(_BaseMemorySystem):
    system_id = "no_memory"
    prompt_label = "No Memory"

    def _choose(self, candidates, context):
        return [], {"policy": "bundle_bound_zero_prompt_exposure"}


class FlatRetrievalSystem(_BaseMemorySystem):
    system_id = "flat_retrieval"
    prompt_label = "Flat Retrieval"

    def _choose(self, candidates, context):
        return candidates[: context.top_k], {
            "policy": "global_common_relevance_without_source_or_stage_preference"
        }


class SOPOnlySystem(_BaseMemorySystem):
    system_id = "sop_only"
    prompt_label = "SOP Only"

    def _choose(self, candidates, context):
        return [item for item in candidates if item.source == "sop"][: context.top_k], {
            "policy": "sop_only"
        }


class RunForestOnlySystem(_BaseMemorySystem):
    system_id = "runforest_only"
    prompt_label = "RunForest Only"

    def _choose(self, candidates, context):
        return [item for item in candidates if item.source == "runforest"][: context.top_k], {
            "policy": "runforest_only"
        }


class StaticHybridSystem(_BaseMemorySystem):
    system_id = "static_hybrid"
    prompt_label = "Static Hybrid"

    def _choose(self, candidates, context):
        quotas = {"sop": 3, "runforest": 3}
        return _take_by_source(candidates, quotas), {"policy": "fixed_quota", "quotas": quotas}


class DynamicHybridSystem(_BaseMemorySystem):
    system_id = "dynamic_hybrid"
    prompt_label = "Dynamic Hybrid"
    QUOTAS = {
        "draft": {"sop": 5, "runforest": 1},
        "improve": {"sop": 3, "runforest": 3},
        "debug": {"sop": 1, "runforest": 5},
    }

    def _choose(self, candidates, context):
        quotas = dict(self.QUOTAS[canonical_stage(context.stage)])
        return _take_by_source(candidates, quotas), {"policy": "stage_quota", "quotas": quotas}


class ReversedRouterSystem(DynamicHybridSystem):
    system_id = "reversed_router"
    prompt_label = "Reversed Router"
    QUOTAS = {
        "draft": {"sop": 2, "runforest": 4},
        "improve": {"sop": 3, "runforest": 3},
        "debug": {"sop": 4, "runforest": 2},
    }


class GOMEStylePortSystem(_BaseMemorySystem):
    system_id = "gome_style_port"
    prompt_label = "GOME-style Port"
    limitation = "success-memory cards only; not the full GOME multi-trace optimizer"

    def _choose(self, candidates, context):
        eligible = [
            item
            for item in candidates
            if item.metadata.get("verified_success") is True
        ]
        return eligible[: context.top_k], {
            "policy": "verified_success_memory",
            "required": "verified_success=true",
        }

    def _format_candidate(self, candidate):
        feedback = str(candidate.metadata.get("execution_feedback") or "")
        delta = candidate.metadata.get("score_delta")
        return (
            f"### {candidate.candidate_id} [success memory]\n"
            f"Hypothesis/plan: {candidate.prompt_text.strip()}\n"
            f"Execution feedback: {feedback or 'verified successful execution'}\n"
            f"Score delta: {delta if delta is not None else 'verified-positive; magnitude unavailable'}"
        )


class MACLAStylePortSystem(_BaseMemorySystem):
    system_id = "macla_style_port"
    prompt_label = "MACLA-style Port"
    limitation = "Beta-reliability utility over frozen memory; no online contrastive refinement"

    @staticmethod
    def _utility(candidate: MemoryCandidate) -> float:
        success = max(0, int(candidate.metadata.get("success_support_count") or 0))
        rejected = max(0, int(candidate.metadata.get("rejected_support_count") or 0))
        alpha = 1.0 + success
        beta = 1.0 + rejected
        posterior_mean = alpha / (alpha + beta)
        failure_probability = beta / (alpha + beta)
        failure_risk = float(candidate.metadata.get("failure_risk") or 0.0)
        information_gain = 1.0 / (alpha + beta)
        return (
            float(candidate.relevance) * posterior_mean
            - failure_risk * failure_probability
            + 0.10 * information_gain
        )

    def _choose(self, candidates, context):
        ranked = sorted(
            candidates,
            key=lambda item: (-self._utility(item), item.candidate_id),
        )
        return ranked[: context.top_k], {
            "policy": "beta_reliability_expected_utility",
            "utility": "relevance*posterior_mean-failure_risk*failure_probability+0.10*information_gain",
        }

    def _format_candidate(self, candidate):
        kind = "meta-procedure" if candidate.source == "sop" else "procedure"
        return (
            f"### {candidate.candidate_id} [{kind}]\n"
            f"Expected utility: {self._utility(candidate):.8f}\n"
            f"{candidate.prompt_text.strip()}"
        )


class RCRRouterStylePortSystem(_BaseMemorySystem):
    system_id = "rcr_router_style_port"
    prompt_label = "RCR-Router-style Port"
    limitation = "MLE stage/role importance and greedy prompt packing; not the original multi-agent QA environment"

    @staticmethod
    def _importance(candidate: MemoryCandidate, stage: str) -> float:
        configured_fit = candidate.metadata.get("stage_fit")
        if configured_fit is None:
            same_stage = canonical_stage(candidate.source_stage) == canonical_stage(stage)
            stage_score = 1.0 if same_stage else 0.0
        else:
            stage_score = max(0.0, min(1.0, float(configured_fit)))
        recency = float(candidate.metadata.get("recency") or 0.0)
        return 0.55 * float(candidate.relevance) + 0.25 * stage_score + 0.20 * recency

    def _choose(self, candidates, context):
        ranked = sorted(
            candidates,
            key=lambda item: (-self._importance(item, context.stage), item.candidate_id),
        )
        return ranked[: context.top_k], {
            "policy": "role_stage_recency_importance_then_greedy_token_pack",
            "role": {"draft": "planner", "improve": "optimizer", "debug": "debugger"}[
                canonical_stage(context.stage)
            ],
        }

    def _format_candidate(self, candidate):
        return (
            f"### {candidate.candidate_id} [{candidate.source}]\n"
            "Priority: frozen MLE role/stage/recency importance.\n"
            f"{candidate.prompt_text.strip()}"
        )


_SYSTEMS: dict[str, type[_BaseMemorySystem]] = {
    cls.system_id: cls
    for cls in (
        NoMemorySystem,
        FlatRetrievalSystem,
        SOPOnlySystem,
        RunForestOnlySystem,
        StaticHybridSystem,
        DynamicHybridSystem,
        ReversedRouterSystem,
        GOMEStylePortSystem,
        MACLAStylePortSystem,
        RCRRouterStylePortSystem,
    )
}


def get_memory_system(system_id: str) -> MemorySystem:
    value = str(system_id or "").strip().lower()
    try:
        return _SYSTEMS[value]()
    except KeyError as error:
        raise ValueError(
            f"Unknown End2End memory system {system_id!r}; expected one of {SYSTEM_IDS}"
        ) from error


class EndToEndMemoryController:
    """Apply exactly one registered system to a shared authorized pool."""

    def __init__(self, system_id: str) -> None:
        self.system = get_memory_system(system_id)

    @property
    def system_id(self) -> str:
        return self.system.system_id

    def retrieve(
        self,
        candidates: Iterable[MemoryCandidate],
        context: MemorySystemContext,
    ) -> MemorySelection:
        return self.system.select(candidates, context)


__all__ = [
    "EndToEndMemoryController",
    "MemoryCandidate",
    "MemorySelection",
    "MemorySystem",
    "MemorySystemContext",
    "SYSTEM_IDS",
    "canonical_stage",
    "get_memory_system",
    "whitespace_tokens",
]
