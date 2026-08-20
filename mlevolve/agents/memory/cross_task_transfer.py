"""Fail-closed, score-free projection for same-type cross-task memory transfer.

This module deliberately sits beside exact Replay.  It never loads historical
source code, predictions, checkpoints, or source-task metrics.  The Host first
decides whether two *different* task IDs share an explicitly configured task
type, then projects only portable SOP text into a transfer prompt.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


TRANSFER_PACK_SCHEMA = "mlevolve_cross_task_transfer_pack_v1"
DEFAULT_LEVELS = ("L2_tactic", "L3_repair")
FORBIDDEN_FIELDS = frozenset(
    {
        "code",
        "code_sha256",
        "implementation_capsule",
        "metric",
        "metric_improvement",
        "official_csv_sha256",
        "official_kaggle_ref",
        "official_metric",
        "official_submission_variant",
        "official_support",
        "prediction",
        "predictions",
        "score",
        "submission",
    }
)


def _canonical_task_id(value: object) -> str:
    task_id = str(value or "").strip()
    while task_id.startswith("full-"):
        task_id = task_id[len("full-") :]
    return task_id


def _tokens(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", str(value or "").lower()))


def _node_task_id(node: Mapping[str, Any]) -> str:
    return _canonical_task_id(
        node.get("task_id") or node.get("recipe_task_id") or node.get("task")
    )


@dataclass(frozen=True)
class CrossTaskTransferPolicy:
    enabled: bool = False
    source_task_id: str = ""
    source_task_type: str = ""
    target_task_type: str = ""
    allowed_levels: tuple[str, ...] = DEFAULT_LEVELS
    max_items: int = 6

    @classmethod
    def from_config(cls, config: Any) -> "CrossTaskTransferPolicy":
        if config is None:
            return cls()
        raw_levels = tuple(
            str(value)
            for value in (
                getattr(config, "cross_task_transfer_allowed_levels", None)
                or DEFAULT_LEVELS
            )
        )
        policy = cls(
            enabled=bool(
                getattr(config, "cross_task_transfer_enabled", False)
            ),
            source_task_id=_canonical_task_id(
                getattr(config, "cross_task_transfer_source_task_id", "")
            ),
            source_task_type=str(
                getattr(config, "cross_task_transfer_source_task_type", "") or ""
            ).strip(),
            target_task_type=str(
                getattr(config, "cross_task_transfer_target_task_type", "") or ""
            ).strip(),
            allowed_levels=raw_levels,
            max_items=int(
                getattr(config, "cross_task_transfer_max_items", 6) or 6
            ),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.source_task_id:
            raise ValueError("cross-task transfer requires source_task_id")
        if not self.source_task_type or not self.target_task_type:
            raise ValueError("cross-task transfer requires both task types")
        if self.source_task_type != self.target_task_type:
            raise ValueError(
                "cross-task transfer source/target task types must match exactly"
            )
        if not self.allowed_levels or not set(self.allowed_levels) <= set(DEFAULT_LEVELS):
            raise ValueError(
                "cross-task transfer may expose only L2_tactic and L3_repair"
            )
        if not 1 <= self.max_items <= 12:
            raise ValueError("cross-task transfer max_items must be in [1, 12]")


@dataclass(frozen=True)
class CrossTaskTransferDecision:
    active: bool
    reason: str
    source_task_id: str
    target_task_id: str
    source_task_type: str
    target_task_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_transfer(
    policy: CrossTaskTransferPolicy,
    *,
    target_task_id: str,
) -> CrossTaskTransferDecision:
    target = _canonical_task_id(target_task_id)
    if not policy.enabled:
        reason = "feature_disabled"
        active = False
    elif target == policy.source_task_id:
        reason = "exact_task_must_use_existing_replay_path"
        active = False
    elif policy.source_task_type != policy.target_task_type:
        reason = "task_type_mismatch"
        active = False
    else:
        reason = "different_task_same_explicit_type"
        active = True
    return CrossTaskTransferDecision(
        active=active,
        reason=reason,
        source_task_id=policy.source_task_id,
        target_task_id=target,
        source_task_type=policy.source_task_type,
        target_task_type=policy.target_task_type,
    )


def _portable_text(node: Mapping[str, Any], *, level: str) -> dict[str, str]:
    if level == "L2_tactic":
        fields = {
            "title": str(node.get("title") or ""),
            "instruction": str(node.get("instruction") or node.get("action") or ""),
            "when_to_use": str(node.get("when_to_use") or ""),
            "boundary": str(node.get("teacher_boundary") or ""),
        }
    else:
        repair = node.get("repair_action") or {}
        if not isinstance(repair, Mapping):
            repair = {}
        signature = node.get("failure_signature") or {}
        if not isinstance(signature, Mapping):
            signature = {}
        fields = {
            "title": str(node.get("title") or ""),
            "when_to_use": str(node.get("when_to_use") or ""),
            "failure_signature": " ".join(
                str(value)
                for value in (signature.get("exception_names") or [])
            ),
            "repair": str(repair.get("summary") or ""),
        }
    return {key: value.strip() for key, value in fields.items() if value.strip()}


def _candidate(
    node_id: str,
    node: Mapping[str, Any],
    *,
    query_tokens: set[str],
) -> dict[str, Any] | None:
    level = str(node.get("abstraction_level") or "")
    text = _portable_text(node, level=level)
    if not text:
        return None
    lexical = _tokens(" ".join(text.values()))
    overlap = len(query_tokens & lexical) / max(1, len(query_tokens))
    return {
        "id": str(node_id),
        "source": "sop",
        "source_task_id": _node_task_id(node),
        "abstraction_level": level,
        "portable_text": text,
        # This is target-query lexical fit only.  It contains no source score.
        "target_relevance": round(float(overlap), 8),
        "source_score_inherited": False,
        "source_code_exposed": False,
    }


def build_transfer_pack(
    nodes: Mapping[str, Mapping[str, Any]],
    policy: CrossTaskTransferPolicy,
    *,
    target_task_id: str,
    stage: str,
    task_description: str,
    query_text: str,
) -> dict[str, Any]:
    decision = decide_transfer(policy, target_task_id=target_task_id)
    canonical_stage = str(stage or "").lower()
    allowed_for_stage = (
        {"L3_repair"} if canonical_stage == "debug" else {"L2_tactic"}
    )
    allowed = allowed_for_stage & set(policy.allowed_levels)
    query_tokens = _tokens(f"{task_description} {query_text}")
    observed: list[dict[str, Any]] = []
    if decision.active:
        for node_id, node in nodes.items():
            if _node_task_id(node) != policy.source_task_id:
                continue
            if str(node.get("abstraction_level") or "") not in allowed:
                continue
            row = _candidate(str(node_id), node, query_tokens=query_tokens)
            if row is not None:
                observed.append(row)
    observed.sort(
        key=lambda row: (
            -float(row["target_relevance"]),
            str(row["id"]),
        )
    )
    selected = observed[: policy.max_items]
    parts = [
        "## Cross-task Transfer Memory (Host-projected)",
        (
            "Adapt these portable tactics to the target task. They are hypotheses, "
            "not source-task answers. Do not copy source code, checkpoints, "
            "predictions, submissions, class mappings, or source scores. Validate "
            "every choice only on target-task training data."
        ),
    ]
    for row in selected:
        lines = [
            f"### {row['id']} [{row['abstraction_level']}]",
            *[
                f"{key}: {value}"
                for key, value in row["portable_text"].items()
            ],
        ]
        parts.append("\n".join(lines))
    prompt = "\n\n".join(parts) if selected else ""
    safe_payload = {
        "decision": decision.to_dict(),
        "stage": canonical_stage,
        "allowed_levels": sorted(allowed),
        "observed_candidates": observed,
        "selected_candidate_ids": [row["id"] for row in selected],
    }
    projection_sha256 = hashlib.sha256(
        json.dumps(
            safe_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": TRANSFER_PACK_SCHEMA,
        "algorithm_version": "host_same_type_score_free_projection_v1",
        "target_task_id": decision.target_task_id,
        "source_task_id": decision.source_task_id,
        "stage_route": {
            "stage": canonical_stage,
            "control": "dynamic_cross_task_transfer",
            "route": "host_transfer_projection",
        },
        "memory_transfer": {
            "activated": decision.active and bool(selected),
            "host_decision": decision.to_dict(),
            "mode": "same_type_cross_task_portable_sop_projection_v1",
            "source_score_inheritance_allowed": False,
            "source_code_exposure_allowed": False,
            "exact_replay_allowed": False,
            "forbidden_fields": sorted(FORBIDDEN_FIELDS),
            "projection_sha256": projection_sha256,
        },
        "pre_gate_raw_candidates": observed,
        "candidate_pool": observed,
        "selected_candidates": selected,
        "suppressed_candidates": [
            {"candidate_id": row["id"], "reason": "outside_transfer_top_k"}
            for row in observed[policy.max_items :]
        ],
        "selected_sop_gateways": selected,
        "sop_only_candidates": selected,
        "fused_execution_candidates": [],
        "evidence_refs": [],
        "failure_patterns": [],
        "final_prompt_candidate_ids": [row["id"] for row in selected],
        "prompt_visible_refs": [row["id"] for row in selected],
        "prompt_text": prompt,
        "prompt_token_count": len(prompt.split()),
        "prompt_truncated": False,
        "navigation_trace": [
            {
                "candidate_id": row["id"],
                "retrieval_channel": "cross_task_transfer_projection",
                "selection_state": (
                    "injected" if row in selected else "suppressed"
                ),
            }
            for row in observed
        ],
        "visible_clause_ids": [],
        "visibility_safety_gate": {
            "pre_ranking": True,
            "unauthorized_prompt_exposure": 0,
            "source_score_fields_exposed": 0,
            "source_code_fields_exposed": 0,
        },
        "unauthorized_prompt_exposure": 0,
    }


__all__ = [
    "CrossTaskTransferDecision",
    "CrossTaskTransferPolicy",
    "TRANSFER_PACK_SCHEMA",
    "build_transfer_pack",
    "decide_transfer",
]
