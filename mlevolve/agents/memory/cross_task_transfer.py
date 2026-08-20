"""Fail-closed, score-free projection for same-type cross-task memory transfer.

This module deliberately sits beside exact Replay.  It never loads historical
source code, predictions, checkpoints, or source-task metrics.  The Host first
decides whether two *different* task IDs share an explicitly configured task
type, then projects portable SOP text into a transfer prompt.  An independent,
opt-in architecture channel may additionally project a structural L1 blueprint
whose source-specific artifact, score, data-shape, and implementation details
have been removed by the Host.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping


TRANSFER_PACK_SCHEMA = "mlevolve_cross_task_transfer_pack_v1"
ARCHITECTURE_TRANSFER_PACK_SCHEMA = "mlevolve_cross_task_transfer_pack_v2"
DEFAULT_LEVELS = ("L2_tactic", "L3_repair")
ARCHITECTURE_LEVEL = "L1_recipe"
ARCHITECTURE_RUNTIME_LEVELS = frozenset(
    {ARCHITECTURE_LEVEL, "L1_strategy"}
)
ARCHITECTURE_PIPELINE_FIELDS = (
    "feature_representation",
    "model_stack",
    "training_protocol",
    "oof_protocol",
    "ensemble_calibration",
    "final_refit_inference",
)
FORBIDDEN_FIELDS = frozenset(
    {
        "class_mapping",
        "class_mappings",
        "code",
        "code_sha256",
        "implementation",
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
        "weights",
    }
)

_SOURCE_HASH_RE = re.compile(r"\b[a-f0-9]{32,}\b", flags=re.IGNORECASE)
_SOURCE_DECIMAL_RE = re.compile(r"(?<![A-Za-z0-9_])\d+\.\d+(?![A-Za-z0-9_])")
_SOURCE_LONG_ID_RE = re.compile(
    r"(?i)(?:(?:kaggle|submission|csv)\s*(?:ref(?:erence)?|id|sha(?:256)?)?\s*[:#=-]?\s*)\d{6,}"
)
_SOURCE_SHAPE_RE = re.compile(
    r"\b\d+\s+(?:training|train|test|validation|rows?|classes?|columns?)\b",
    flags=re.IGNORECASE,
)
_SOURCE_DIMENSION_RE = re.compile(
    r"\b(?:\d+\s*[- ]\s*(?:way|class|classes|component|components|feature|features|"
    r"dimensional|dimensions?|epochs?)|patience\s+\d+)\b",
    flags=re.IGNORECASE,
)
_SOURCE_ARTIFACT_PHRASE_RE = re.compile(
    r"(?:the\s+)?(?:exact\s+)?(?:[A-Za-z0-9_-]+\s+)?CSV\s+"
    r"(?:\[source artifact redacted\]|hash)",
    flags=re.IGNORECASE,
)
_SOURCE_VARIANT_ID_RE = re.compile(
    r"\b[a-z][a-z0-9-]*_exact\b",
    flags=re.IGNORECASE,
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
    architecture_transfer_enabled: bool = False
    architecture_max_items: int = 1

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
            architecture_transfer_enabled=bool(
                getattr(
                    config,
                    "cross_task_architecture_transfer_enabled",
                    False,
                )
            ),
            architecture_max_items=int(
                getattr(config, "cross_task_architecture_max_items", 1) or 1
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
        if not 1 <= self.architecture_max_items <= 3:
            raise ValueError(
                "cross-task architecture_max_items must be in [1, 3]"
            )


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


def _sanitize_architecture_text(value: object, *, source_task_id: str) -> str:
    """Remove source-instance evidence while retaining structural semantics."""

    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if source_task_id:
        text = re.sub(
            re.escape(source_task_id),
            "source task",
            text,
            flags=re.IGNORECASE,
        )
    text = _SOURCE_HASH_RE.sub("[source artifact redacted]", text)
    text = _SOURCE_LONG_ID_RE.sub("[source identifier redacted]", text)
    text = _SOURCE_DECIMAL_RE.sub("[source numeric redacted]", text)
    text = _SOURCE_SHAPE_RE.sub("target-derived data shape", text)
    text = _SOURCE_DIMENSION_RE.sub("target-derived capacity", text)
    text = _SOURCE_ARTIFACT_PHRASE_RE.sub(
        "a target-task output artifact",
        text,
    )
    text = _SOURCE_VARIANT_ID_RE.sub("architecture output", text)
    replacements = {
        "best official artifact": "target-validated artifact",
        "official artifact": "source artifact",
        "official kaggle": "source evaluation",
        "kaggle": "source evaluation",
        "csv sha256": "source artifact identity",
        "exact replay": "source architecture blueprint",
        "exact architecture signature": "architecture interface",
        "scored architecture signature": "architecture interface",
        "exact selected checkpoint": "target-selected checkpoint",
        "exact checkpointed": "structurally corresponding",
        "exact states": "target-trained states",
        "exact validation-selected": "target-validation-selected",
        "exact scored artifact": "target-validated output",
        "exact scored submission": "target-validated output",
        "exact gome test probabilities": "target-derived ensemble probabilities",
        "recorded": "blueprint-level",
        "source-derived": "blueprint-level",
        "source architecture blueprint": "architecture blueprint",
        "do not silently substitute unavailable backbones or change feature widths": (
            "replace unavailable backbones and derive feature widths from target data"
        ),
    }
    for source, target in replacements.items():
        text = re.sub(source, target, text, flags=re.IGNORECASE)
    return text.strip()


def _architecture_blueprint(node: Mapping[str, Any]) -> dict[str, Any]:
    """Project one L1 Recipe into a code/artifact/score-free architecture."""

    source_task_id = _node_task_id(node)
    pipeline = node.get("pipeline") or {}
    if not isinstance(pipeline, Mapping):
        pipeline = {}
    components = {
        key: _sanitize_architecture_text(
            pipeline.get(key), source_task_id=source_task_id
        )
        for key in ARCHITECTURE_PIPELINE_FIELDS
    }
    components = {key: value for key, value in components.items() if value}
    summary = _sanitize_architecture_text(
        node.get("teacher_distilled_recipe")
        or node.get("distilled_recipe")
        or "",
        source_task_id=source_task_id,
    )
    blueprint = {
        "title": _sanitize_architecture_text(
            node.get("title"), source_task_id=source_task_id
        ),
        "method_family": _sanitize_architecture_text(
            node.get("method_family"), source_task_id=source_task_id
        ),
        "architecture_summary": summary,
        "pipeline_order": list(components),
        "components": components,
        "target_adaptation_contract": (
            "Derive feature dimensions, class order, dependencies, folds, model "
            "selection, calibration, and inference state entirely from target-task "
            "training data. Replace or omit every component that fails target "
            "runtime preflight or target validation."
        ),
    }
    return {
        key: value
        for key, value in blueprint.items()
        if value not in ("", [], {})
    }


def _candidate(
    node_id: str,
    node: Mapping[str, Any],
    *,
    query_tokens: set[str],
) -> dict[str, Any] | None:
    raw_level = str(node.get("abstraction_level") or "")
    if raw_level in ARCHITECTURE_RUNTIME_LEVELS:
        level = ARCHITECTURE_LEVEL
        text = _architecture_blueprint(node)
        candidate_kind = "architecture_blueprint"
    else:
        level = raw_level
        source_task_id = _node_task_id(node)
        text = {
            key: _sanitize_architecture_text(
                value,
                source_task_id=source_task_id,
            )
            for key, value in _portable_text(node, level=level).items()
        }
        text = {key: value for key, value in text.items() if value}
        candidate_kind = (
            "portable_tactic" if level == "L2_tactic" else "portable_repair"
        )
    if not text:
        return None
    lexical = _tokens(json.dumps(text, ensure_ascii=False, sort_keys=True))
    overlap = len(query_tokens & lexical) / max(1, len(query_tokens))
    source_task_id = _node_task_id(node)
    method_family = _sanitize_architecture_text(
        node.get("method_family"),
        source_task_id=source_task_id,
    )
    parent_method_families = [
        value
        for value in (
            _sanitize_architecture_text(
                item,
                source_task_id=source_task_id,
            )
            for item in (node.get("parent_method_families") or [])
        )
        if value
    ]
    return {
        "id": str(node_id),
        "source": "sop",
        "source_task_id": source_task_id,
        "abstraction_level": level,
        "candidate_kind": candidate_kind,
        "portable_text": text,
        "method_family": method_family,
        "parent_method_families": parent_method_families,
        # This is target-query lexical fit only.  It contains no source score.
        "target_relevance": round(float(overlap), 8),
        "source_score_inherited": False,
        "source_code_exposed": False,
        "source_artifact_exposed": False,
    }


def project_transfer_candidates(
    nodes: Mapping[str, Mapping[str, Any]],
    policy: CrossTaskTransferPolicy,
    *,
    target_task_id: str,
    stage: str,
    task_description: str,
    query_text: str,
    all_safe_levels: bool = False,
) -> dict[str, Any]:
    """Build the irreversible Host-safe candidate universe.

    ``all_safe_levels`` is reserved for the dynamic Search/Judge/Resolver
    route.  It removes the legacy stage preselection, but it does not widen
    the authority boundary: only sanitized L1 structure and portable L2/L3
    text from the configured same-type source task can enter the universe.
    """

    decision = decide_transfer(policy, target_task_id=target_task_id)
    canonical_stage = str(stage or "").lower()
    if all_safe_levels:
        allowed = set(policy.allowed_levels)
        if policy.architecture_transfer_enabled:
            allowed.update(ARCHITECTURE_RUNTIME_LEVELS)
    else:
        allowed_for_stage = (
            {"L3_repair"} if canonical_stage == "debug" else {"L2_tactic"}
        )
        if policy.architecture_transfer_enabled and canonical_stage == "draft":
            allowed_for_stage.update(ARCHITECTURE_RUNTIME_LEVELS)
        allowed = allowed_for_stage & set(policy.allowed_levels)
        if policy.architecture_transfer_enabled and canonical_stage == "draft":
            allowed.update(ARCHITECTURE_RUNTIME_LEVELS)

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
        key=lambda row: (-float(row["target_relevance"]), str(row["id"]))
    )
    payload = {
        "decision": decision.to_dict(),
        "stage": canonical_stage,
        "all_safe_levels": bool(all_safe_levels),
        "allowed_levels": sorted(allowed),
        "observed_candidates": observed,
    }
    payload["projection_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


def build_transfer_pack(
    nodes: Mapping[str, Mapping[str, Any]],
    policy: CrossTaskTransferPolicy,
    *,
    target_task_id: str,
    stage: str,
    task_description: str,
    query_text: str,
) -> dict[str, Any]:
    projection = project_transfer_candidates(
        nodes,
        policy,
        target_task_id=target_task_id,
        stage=stage,
        task_description=task_description,
        query_text=query_text,
    )
    decision = CrossTaskTransferDecision(**projection["decision"])
    canonical_stage = str(projection["stage"])
    allowed = set(projection["allowed_levels"])
    observed = list(projection["observed_candidates"])
    architecture_candidates = [
        row for row in observed if row["abstraction_level"] == ARCHITECTURE_LEVEL
    ]
    portable_candidates = [
        row for row in observed if row["abstraction_level"] != ARCHITECTURE_LEVEL
    ]
    selected_architectures = architecture_candidates[
        : policy.architecture_max_items
    ]
    selected_portable = portable_candidates[: policy.max_items]
    selected = [*selected_architectures, *selected_portable]
    selected_ids = {row["id"] for row in selected}
    parts = [
        "## Cross-task Transfer Memory (Host-projected)",
        (
            "Adapt the Host-projected architecture blueprint and portable tactics "
            "to the target task. They are hypotheses, not source-task answers. "
            "Do not copy source code, checkpoints, weights, predictions, "
            "submissions, class mappings, source data dimensions, artifact "
            "identities, or source scores. Select one coherent architecture; do "
            "not concatenate incompatible source families. Validate every choice "
            "only on target-task training data."
        ),
    ]
    if selected_architectures:
        parts.append("## Structural architecture blueprint")
    for row in selected_architectures:
        lines = [
            f"### {row['id']} [{row['abstraction_level']}]",
            json.dumps(
                row["portable_text"],
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ),
        ]
        parts.append("\n".join(lines))
    if selected_portable:
        parts.append("## Portable tactics and repairs")
    for row in selected_portable:
        lines = [f"### {row['id']} [{row['abstraction_level']}]"]
        lines.extend(
            f"{key}: {value}" for key, value in row["portable_text"].items()
        )
        parts.append("\n".join(lines))
    prompt = "\n\n".join(parts) if selected else ""
    safe_payload = {
        "decision": decision.to_dict(),
        "stage": canonical_stage,
        "allowed_levels": sorted(allowed),
        "observed_candidates": observed,
        "selected_candidate_ids": [row["id"] for row in selected],
        "selected_architecture_ids": [
            row["id"] for row in selected_architectures
        ],
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
        "schema": (
            ARCHITECTURE_TRANSFER_PACK_SCHEMA
            if policy.architecture_transfer_enabled
            else TRANSFER_PACK_SCHEMA
        ),
        "algorithm_version": (
            "host_same_type_score_code_artifact_free_architecture_projection_v2"
            if policy.architecture_transfer_enabled
            else "host_same_type_score_free_projection_v1"
        ),
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
            "mode": (
                "same_type_cross_task_architecture_and_portable_projection_v2"
                if policy.architecture_transfer_enabled
                else "same_type_cross_task_portable_sop_projection_v1"
            ),
            "architecture_transfer_enabled": (
                policy.architecture_transfer_enabled
            ),
            "architecture_projection_mode": (
                "host_structural_fields_only_v1"
                if policy.architecture_transfer_enabled
                else "disabled"
            ),
            "selected_architecture_ids": [
                row["id"] for row in selected_architectures
            ],
            "source_score_inheritance_allowed": False,
            "source_code_exposure_allowed": False,
            "source_artifact_exposure_allowed": False,
            "exact_replay_allowed": False,
            "forbidden_fields": sorted(FORBIDDEN_FIELDS),
            "projection_sha256": projection_sha256,
        },
        "candidate_pool_source": "host_same_type_score_free_projection",
        "candidate_pool_hash": projection_sha256,
        "memory_pool_sha256": projection_sha256,
        "live_query_used_for_candidate_pool": False,
        "pre_gate_raw_candidates": observed,
        "candidate_pool": observed,
        "selected_candidates": selected,
        "selected_architectures": selected_architectures,
        "selected_portable_items": selected_portable,
        "selected_items": selected,
        "suppressed_candidates": [
            {"candidate_id": row["id"], "reason": "outside_transfer_top_k"}
            for row in observed
            if row["id"] not in selected_ids
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
            "source_artifact_fields_exposed": 0,
        },
        "safety_gate": {
            "pre_ranking": True,
            "unauthorized_prompt_exposure": 0,
            "source_score_fields_exposed": 0,
            "source_code_fields_exposed": 0,
            "source_artifact_fields_exposed": 0,
        },
        "unauthorized_prompt_exposure": 0,
    }


__all__ = [
    "ARCHITECTURE_TRANSFER_PACK_SCHEMA",
    "CrossTaskTransferDecision",
    "CrossTaskTransferPolicy",
    "TRANSFER_PACK_SCHEMA",
    "build_transfer_pack",
    "decide_transfer",
    "project_transfer_candidates",
]
