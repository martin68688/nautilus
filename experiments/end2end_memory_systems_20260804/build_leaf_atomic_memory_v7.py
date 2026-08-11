#!/usr/bin/env python3
"""Build the Leaf v7 full-Forest, claim-level memory release.

This builder deliberately separates four safety decisions:

* the source program may be quarantined;
* its reported metric may be quarantined;
* a minimal runtime repair can still be independently verified;
* visibility of that repair is limited to Debug hypothesis/repair operations.

Every Leaf transition remains represented in ``atomic_claims.json``.  The
teacher only consolidates already-authorized atomic repairs; it cannot promote,
delete, or alter their evidence bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any, Mapping

# Load the pure claim utility without importing ``agents.memory.__init__``;
# that package eagerly imports the training retriever and its GPU-time
# dependencies, none of which are required by this deterministic builder.
REPO_ROOT = Path(__file__).resolve().parents[2]
MLEVOLVE_ROOT = REPO_ROOT / "mlevolve"
_CLAIM_MODULE_PATH = MLEVOLVE_ROOT / "agents" / "memory" / "atomic_claim_memory.py"
_CLAIM_SPEC = importlib.util.spec_from_file_location(
    "mlevolve_atomic_claim_memory", _CLAIM_MODULE_PATH
)
if _CLAIM_SPEC is None or _CLAIM_SPEC.loader is None:
    raise RuntimeError(f"Cannot load atomic claim utility: {_CLAIM_MODULE_PATH}")
_CLAIM_MODULE = importlib.util.module_from_spec(_CLAIM_SPEC)
sys.modules[_CLAIM_SPEC.name] = _CLAIM_MODULE
_CLAIM_SPEC.loader.exec_module(_CLAIM_MODULE)
ATOMIC_CLAIM_BUNDLE_SCHEMA = _CLAIM_MODULE.ATOMIC_CLAIM_BUNDLE_SCHEMA
ATOMIC_CLAIM_SCHEMA = _CLAIM_MODULE.ATOMIC_CLAIM_SCHEMA
AUTHORIZED_DEBUG_STATUS = _CLAIM_MODULE.AUTHORIZED_DEBUG_STATUS
extract_before_after = _CLAIM_MODULE.extract_before_after
extract_debug_signature = _CLAIM_MODULE.extract_debug_signature


TASK_ID = "leaf-classification"
TASK_DOMAIN = "multimodal_multiclass_classification"
TASK_FAMILY = "tabular_multiclass"
TASK_TYPE = "multimodal"
TEACHER_MODEL = "gpt-5.6-sol"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: item for key, item in value.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_replace(source: Path, target: Path) -> None:
    """Idempotently replace a copied read-only release artifact."""

    if target.exists():
        target.chmod(target.stat().st_mode | stat.S_IWUSR)
    shutil.copy2(source, target)


def trim(value: object, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def clean_positive_node(node: Mapping[str, Any]) -> bool:
    audit = node.get("leakage_audit")
    return bool(
        node.get("type") == "RunNode"
        and node.get("is_buggy") is False
        and node.get("is_valid") is True
        and finite_number(node.get("metric"))
        and isinstance(audit, Mapping)
        and audit.get("status") == "clean"
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
        and audit.get("rank_eligible") is True
    )


def node_hash(node: Mapping[str, Any]) -> str:
    audit = node.get("leakage_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    return str(node.get("code_sha256") or audit.get("code_sha256") or "")


def failure_text(node: Mapping[str, Any]) -> str:
    primary = "\n".join(
        str(node.get(key) or "")
        for key in ("analysis", "terminal_excerpt")
        if node.get(key)
    )
    fallback = "\n".join(
        str(node.get(key) or "")
        for key in ("plan", "text")
        if node.get(key)
    )
    return trim(primary or fallback, 5200)


def repair_text(transition: Mapping[str, Any], child: Mapping[str, Any]) -> str:
    # The successful Debug plan is the narrowest account of the actual repair.
    return trim(
        child.get("plan")
        or child.get("code_summary")
        or transition.get("text")
        or "",
        4200,
    )


def runtime_stage(text: str) -> str:
    lowered = text.lower()
    rules = (
        ("data_loading", ("filenotfound", "read_csv", "dataset", "data path")),
        ("preprocessing", ("scaler", "vectorizer", "pca", "transform")),
        ("feature_extraction", ("backbone", "embedding", "dinov", "timm", "torch.hub")),
        ("model_forward", ("input height", "shape", "broadcast", "dimension", "tensor")),
        ("training", ("out of memory", "backward", "optimizer", "batch size")),
        ("validation", ("log_loss", "roc_auc", "validation")),
        ("inference", ("inference", "predict", "test prediction")),
        ("submission", ("submission", "sample_submission")),
    )
    for stage, markers in rules:
        if any(marker in lowered for marker in markers):
            return stage
    return "training"


def claim_type_for_debug(text: str, signature: Mapping[str, list[str]]) -> str:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "out of memory",
            "cuda out of memory",
            "cuda memory exhausted",
            "resourceexhausted",
        )
    ):
        return "resource_claim"
    if (
        signature.get("model_api_ids")
        and (
            signature.get("numeric_literals")
            or signature.get("shape_literals")
            or any(marker in lowered for marker in ("unexpected keyword", "unsupported", "requires"))
        )
    ):
        return "compatibility_claim"
    if any(
        marker in lowered
        for marker in (
            "filenotfound",
            "not found",
            "nameerror",
            "modulenotfound",
            "unexpected keyword",
            "wrong path",
        )
    ):
        return "implementation_claim"
    return "repair_claim"


def source_taint(node: Mapping[str, Any]) -> dict[str, Any]:
    audit = node.get("leakage_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    return {
        "status": str(audit.get("status") or node.get("audit_status") or "unknown"),
        "memory_disposition": str(
            audit.get("memory_disposition") or node.get("memory_disposition") or "unknown"
        ),
        "rank_eligible": audit.get("rank_eligible") is True,
        "issue_codes": [
            str(item.get("issue_code") or "")
            for item in (audit.get("issues") or [])
            if isinstance(item, Mapping) and item.get("issue_code")
        ],
    }


def independently_authorized_debug_claim(
    *,
    transition: Mapping[str, Any],
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    failure: str,
    action: str,
    signature: Mapping[str, list[str]],
    before_after: list[dict[str, str]],
) -> tuple[bool, str]:
    if not (
        transition.get("outcome") == "debug_fixed"
        and transition.get("parent_buggy") is True
        and transition.get("child_buggy") is False
        and parent.get("is_buggy") is True
        and child.get("is_buggy") is False
        and child.get("is_valid") is True
    ):
        return False, "not_observed_failure_to_success"
    if len(node_hash(parent)) != 64 or len(node_hash(child)) != 64:
        return False, "missing_before_after_code_hash"
    if not failure or not action:
        return False, "missing_failure_or_repair_action"
    distinctive = bool(
        before_after
        or (
            signature.get("exception_names")
            and any(
                signature.get(key)
                for key in (
                    "model_api_ids",
                    "numeric_literals",
                    "shape_literals",
                    "path_literals",
                    "symbol_names",
                )
            )
        )
    )
    if not distinctive:
        return False, "no_specific_runtime_operands"
    if len(action) > 4200:
        return False, "repair_action_not_atomic"
    unsafe_action = re.search(
        r"(?:fit(?:_transform)?\s*\([^)]*(?:test|holdout)|"
        r"train\s*\+\s*test|including\s+(?:the\s+)?test\s+(?:set|data)|"
        r"select\s+(?:the\s+)?best.*(?:test|holdout))",
        action,
        flags=re.I | re.S,
    )
    if unsafe_action:
        return False, "repair_action_contains_unsafe_data_scope"
    return True, "independent_local_failure_repair_execution"


def claim_id(transition_id: str, claim_type: str, action: str) -> str:
    digest = hashlib.sha256(
        f"{transition_id}\0{claim_type}\0{action}".encode("utf-8")
    ).hexdigest()[:20]
    return f"claim::{TASK_ID}::{claim_type}::{digest}"


def build_claims(graph: Mapping[str, Any], source_graph_sha256: str) -> list[dict[str, Any]]:
    nodes = graph.get("nodes") if isinstance(graph, Mapping) else None
    if not isinstance(nodes, list):
        raise ValueError("RunForest graph has no node list")
    by_id = {
        str(node.get("id") or ""): node
        for node in nodes
        if isinstance(node, Mapping) and node.get("id")
    }
    claims: list[dict[str, Any]] = []
    for transition in nodes:
        if (
            not isinstance(transition, Mapping)
            or transition.get("type") != "Transition"
            or str(transition.get("task") or "") != TASK_ID
        ):
            continue
        transition_id = str(transition.get("id") or "")
        parent = by_id.get(str(transition.get("parent_node_id") or ""), {})
        child = by_id.get(str(transition.get("child_node_id") or ""), {})
        outcome = str(transition.get("outcome") or "unknown")
        common = {
            "schema": ATOMIC_CLAIM_SCHEMA,
            "task_id": TASK_ID,
            "task_type": TASK_TYPE,
            "task_family": TASK_DOMAIN,
            "source_transition_id": transition_id,
            "source_parent_node_id": str(transition.get("parent_node_id") or ""),
            "source_child_node_id": str(transition.get("child_node_id") or ""),
            "source_run_id": str(transition.get("run_id") or ""),
            "source_graph_sha256": source_graph_sha256,
            "outcome": outcome,
        }
        if outcome == "debug_fixed":
            failure = failure_text(parent)
            action = repair_text(transition, child)
            signature = extract_debug_signature(f"{failure}\n{action}")
            before_after = extract_before_after(f"{failure}\n{action}")
            kind = claim_type_for_debug(f"{failure}\n{action}", signature)
            authorized, reason = independently_authorized_debug_claim(
                transition=transition,
                parent=parent,
                child=child,
                failure=failure,
                action=action,
                signature=signature,
                before_after=before_after,
            )
            cid = claim_id(transition_id, kind, action)
            code_taint = source_taint(child)
            claims.append(
                {
                    **common,
                    "id": cid,
                    "claim_type": kind,
                    "claim_status": (
                        AUTHORIZED_DEBUG_STATUS if authorized else "quarantine"
                    ),
                    "claim_status_reason": reason,
                    "failure_text": failure,
                    "repair_action": action,
                    "failure_signature": signature,
                    "before_after": before_after,
                    "runtime_stage": runtime_stage(f"{failure}\n{action}"),
                    "metric_authorized": False,
                    "metric_value": None,
                    "taint": {
                        "code": (
                            "clean" if clean_positive_node(child) else "quarantine"
                        ),
                        "metric": (
                            "clean" if clean_positive_node(child) else "quarantine"
                        ),
                        "claim": "clean" if authorized else "quarantine",
                        "source_program": code_taint,
                    },
                    "verification": {
                        "observed_parent_failure": parent.get("is_buggy") is True,
                        "observed_child_execution_success": bool(
                            child.get("is_buggy") is False
                            and child.get("is_valid") is True
                        ),
                        "repair_action_bound_to_transition": bool(action),
                        "claim_scope_independently_audited": authorized,
                        "before_code_sha256": node_hash(parent),
                        "after_code_sha256": node_hash(child),
                        "full_program_clean": clean_positive_node(child),
                    },
                    "operation_visibility": {
                        "allowed_operations": [
                            "debug_hypothesis",
                            "debug_repair",
                        ]
                        if authorized
                        else [],
                        "forbidden_operations": [
                            "draft_method_selection",
                            "improve_method_selection",
                            "metric_ranking",
                            "exact_replay",
                        ],
                        "task_scope": "exact_task",
                    },
                }
            )
            continue

        child_clean = clean_positive_node(child)
        if outcome == "metric_improved" and child_clean:
            kind = "method_claim"
            action = trim(child.get("plan") or transition.get("text"), 4200)
            status = "authorized_strategy_only"
            reason = "strict_clean_metric_improved_transition"
        elif outcome in {"buggy", "metric_worsened", "metric_flat", "unknown"}:
            kind = "negative_result_claim"
            action = trim(
                failure_text(child) or transition.get("text") or failure_text(parent),
                4200,
            )
            status = "authorized_warning_only"
            reason = "observed_nonpositive_transition"
        else:
            kind = "implementation_claim"
            action = trim(transition.get("text"), 4200)
            status = "quarantine"
            reason = "unsupported_transition_outcome"
        cid = claim_id(transition_id, kind, action)
        claims.append(
            {
                **common,
                "id": cid,
                "claim_type": kind,
                "claim_status": status,
                "claim_status_reason": reason,
                "claim_text": action,
                "metric_authorized": bool(child_clean and finite_number(child.get("metric"))),
                "metric_value": float(child["metric"])
                if child_clean and finite_number(child.get("metric"))
                else None,
                "taint": {
                    "code": "clean" if child_clean else "quarantine",
                    "metric": "clean" if child_clean else "quarantine",
                    "claim": "clean" if status != "quarantine" else "quarantine",
                    "source_program": source_taint(child),
                },
                "verification": {
                    "observed_transition": True,
                    "strict_clean_child": child_clean,
                    "before_code_sha256": node_hash(parent),
                    "after_code_sha256": node_hash(child),
                },
                "operation_visibility": {
                    "allowed_operations": (
                        ["improve_strategy"]
                        if status == "authorized_strategy_only"
                        else ["inspect_warning"]
                        if status == "authorized_warning_only"
                        else []
                    ),
                    "forbidden_operations": ["exact_replay"]
                    if not child_clean
                    else [],
                    "task_scope": "exact_task",
                },
            }
        )
    return claims


def teacher_schema() -> dict[str, Any]:
    repair = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "signature_id": {"type": "string"},
            "failure_pattern": {"type": "string"},
            "root_cause": {"type": "string"},
            "repair_steps": {"type": "array", "items": {"type": "string"}},
            "when_to_use": {"type": "string"},
            "method_family": {"type": "string"},
            "runtime_stage": {"type": "string"},
            "source_claim_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "title",
            "signature_id",
            "failure_pattern",
            "root_cause",
            "repair_steps",
            "when_to_use",
            "method_family",
            "runtime_stage",
            "source_claim_ids",
        ],
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "generalized_repairs": {"type": "array", "items": repair},
            "synthesis_assessment": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "coverage_risks": {"type": "array", "items": {"type": "string"}},
                    "ranking_recommendations": {"type": "array", "items": {"type": "string"}},
                    "do_not_generalize": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "coverage_risks",
                    "ranking_recommendations",
                    "do_not_generalize",
                ],
            },
        },
        "required": ["generalized_repairs", "synthesis_assessment"],
    }


def teacher_packet(claims: list[dict[str, Any]]) -> dict[str, Any]:
    authorized = [
        claim
        for claim in claims
        if claim.get("claim_status") == AUTHORIZED_DEBUG_STATUS
    ]
    compact = [
        {
            "claim_id": claim["id"],
            "claim_type": claim["claim_type"],
            "runtime_stage": claim["runtime_stage"],
            "failure_text": trim(claim["failure_text"], 1800),
            "repair_action": trim(claim["repair_action"], 1600),
            "failure_signature": claim["failure_signature"],
            "before_after": claim["before_after"],
            "source_transition_id": claim["source_transition_id"],
            "source_program_taint": claim["taint"]["source_program"],
        }
        for claim in authorized
    ]
    return {
        "schema": "mlevolve_atomic_memory_teacher_packet_v1",
        "teacher_model": TEACHER_MODEL,
        "task_id": TASK_ID,
        "instructions": [
            "Consolidate only supplied authorized atomic claims into reusable Debug repair families.",
            "Cite only supplied claim_id values; never invent evidence, metrics, models, paths, or operands.",
            "A quarantined source program does not invalidate a separately verified local repair, but none of its model strategy or metric may be promoted.",
            "Preserve exact exception, model/API, numeric operand, tensor-shape, path, and symbol preconditions.",
            "Do not merge repairs that have different causal mechanisms merely because they share a broad category.",
            "Prefer 12-30 high-value generalized repairs; output fewer if evidence does not support them.",
        ],
        "authorized_claim_count": len(compact),
        "authorized_claims": compact,
    }


def transition_for_claim(claim: Mapping[str, Any]) -> dict[str, Any]:
    suffix = str(claim["id"]).split("::")[-1]
    return {
        "id": f"atomic-transition::{TASK_ID}::{suffix}",
        "type": "Transition",
        "task": TASK_ID,
        "run_id": str(claim.get("source_run_id") or ""),
        "run_short_id": str(claim.get("source_run_id") or ""),
        "parent_node_id": str(claim["source_parent_node_id"]),
        "child_node_id": str(claim["source_child_node_id"]),
        "stage_pair": "debug->debug",
        "outcome": "debug_fixed",
        "parent_buggy": True,
        "child_buggy": False,
        "child_metric": None,
        "metric_improvement": None,
        "text": str(claim["repair_action"]),
        "atomic_repair_claim": dict(claim),
        "atomic_claim_id": str(claim["id"]),
        "metric_authorized": False,
        "quarantined": False,
        "protocol_biased": False,
    }


def deterministic_claim_sop(claim: Mapping[str, Any], transition_id: str) -> dict[str, Any]:
    signature = claim.get("failure_signature") or {}
    exception = next(iter(signature.get("exception_names") or []), "runtime failure")
    model = next(iter(signature.get("model_api_ids") or []), "affected component")
    before_after = claim.get("before_after") or []
    operand = ""
    if before_after:
        row = before_after[0]
        operand = f" {row.get('before') or '?'}→{row.get('after') or '?'}"
    title = trim(f"{exception} / {model}{operand} atomic repair", 180)
    cid_suffix = str(claim["id"]).split("::")[-1]
    return {
        "id": f"repair-claim::{TASK_ID}::{cid_suffix}",
        "type": "SOP",
        "abstraction_level": "L3_repair",
        "sop_kind": "debug_fix",
        "task_id": TASK_ID,
        "task_domain": TASK_DOMAIN,
        "task_family": TASK_DOMAIN,
        "task_type": TASK_TYPE,
        "decision_stages": ["debug"],
        "runtime_stage": str(claim.get("runtime_stage") or "training"),
        "runtime_stages": [str(claim.get("runtime_stage") or "training")],
        "title": title,
        "method_family": "runtime_compatibility"
        if claim.get("claim_type") == "compatibility_claim"
        else "targeted_runtime_repair",
        "failure_signature": {
            "id": f"atomic/{claim.get('claim_type')}/{cid_suffix}",
            "pattern": trim(claim.get("failure_text"), 1200),
            "root_cause": trim(claim.get("repair_action"), 1000),
            **{key: list(value) for key, value in signature.items()},
        },
        "repair_action": {
            "summary": trim(claim.get("repair_action"), 900),
            "steps": [trim(claim.get("repair_action"), 1800)],
            "before_after": list(before_after),
        },
        "when_to_use": "Use only when the current exception and concrete model/API/operand anchors match this claim.",
        "supporting_transition_ids": [transition_id],
        "source_transition_ids": [transition_id],
        "source_claim_ids": [str(claim["id"])],
        "failure_node_ids": [str(claim["source_parent_node_id"])],
        "successful_node_ids": [str(claim["source_child_node_id"])],
        "source_node_ids": [
            str(claim["source_parent_node_id"]),
            str(claim["source_child_node_id"]),
        ],
        "distinct_run_ids": [str(claim.get("source_run_id") or "")],
        "distinct_run_count": 1,
        "successful_repair_count": 1,
        "evidence_status": "accepted_atomic_repair_claim",
        "confidence_prior": 0.70,
        "source_admission": "independent_claim_level_failure_repair_verification",
        "infrastructure_failure": False,
        "one_off_code_failure": False,
        "metric_authorized": False,
        "operation_visibility": dict(claim["operation_visibility"]),
    }


def teacher_sops(
    response: Mapping[str, Any],
    claims_by_id: Mapping[str, Mapping[str, Any]],
    transition_by_claim: Mapping[str, str],
    claim_id_aliases: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    output: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    allowed_stages = {
        "checkpoint_averaging",
        "checkpoint_loading",
        "data_loading",
        "preprocessing",
        "split_validation",
        "feature_extraction",
        "import",
        "model_loading",
        "model_forward",
        "parsing",
        "training",
        "training_metric",
        "validation",
        "validation_split",
        "oof",
        "inference",
        "submission",
    }
    for index, row in enumerate(response.get("generalized_repairs") or [], 1):
        if not isinstance(row, Mapping):
            continue
        aliases = claim_id_aliases or {}
        source_ids = list(
            dict.fromkeys(
                str(aliases.get(str(value), str(value)))
                for value in row.get("source_claim_ids") or []
            )
        )
        reason = ""
        if not source_ids or any(
            value not in claims_by_id
            or claims_by_id[value].get("claim_status") != AUTHORIZED_DEBUG_STATUS
            for value in source_ids
        ):
            reason = "invalid_or_unauthorized_source_claim"
        signature_id = re.sub(
            r"[^a-z0-9/_-]+", "_", str(row.get("signature_id") or "").lower()
        ).strip("_/")
        if not signature_id or signature_id in seen:
            reason = reason or "missing_or_duplicate_signature"
        steps = [trim(value, 1400) for value in row.get("repair_steps") or [] if trim(value, 1400)]
        stage = str(row.get("runtime_stage") or "")
        if not steps or stage not in allowed_stages:
            reason = reason or "invalid_steps_or_runtime_stage"
        if reason:
            rejected.append(
                {
                    "title": trim(row.get("title"), 180),
                    "reason": reason,
                }
            )
            continue
        seen.add(signature_id)
        transitions = [transition_by_claim[value] for value in source_ids]
        failure_nodes = list(
            dict.fromkeys(
                str(claims_by_id[value]["source_parent_node_id"])
                for value in source_ids
            )
        )
        child_nodes = list(
            dict.fromkeys(
                str(claims_by_id[value]["source_child_node_id"])
                for value in source_ids
            )
        )
        run_ids = list(
            dict.fromkeys(
                str(claims_by_id[value].get("source_run_id") or "")
                for value in source_ids
            )
        )
        output.append(
            {
                "id": f"repair-teacher::{TASK_ID}::{index:03d}",
                "type": "SOP",
                "abstraction_level": "L3_repair",
                "sop_kind": "debug_fix",
                "task_id": TASK_ID,
                "task_domain": TASK_DOMAIN,
                "task_family": TASK_DOMAIN,
                "task_type": TASK_TYPE,
                "decision_stages": ["debug"],
                "runtime_stage": stage,
                "runtime_stages": [stage],
                "title": trim(row.get("title"), 180),
                "method_family": re.sub(
                    r"[^a-z0-9]+", "_", str(row.get("method_family") or "general").lower()
                ).strip("_")
                or "general",
                "failure_signature": {
                    "id": signature_id,
                    "pattern": trim(row.get("failure_pattern"), 1200),
                    "root_cause": trim(row.get("root_cause"), 1200),
                },
                "repair_action": {"summary": steps[0], "steps": steps},
                "when_to_use": trim(row.get("when_to_use"), 900),
                "supporting_transition_ids": transitions,
                "source_transition_ids": transitions,
                "source_claim_ids": source_ids,
                "failure_node_ids": failure_nodes,
                "successful_node_ids": child_nodes,
                "source_node_ids": [*failure_nodes, *child_nodes],
                "distinct_run_ids": run_ids,
                "distinct_run_count": len(run_ids),
                "successful_repair_count": len(source_ids),
                "evidence_status": "accepted_atomic_repair_claim",
                "confidence_prior": 0.75,
                "source_admission": "gpt_5_6_sol_consolidation_of_authorized_atomic_claims",
                "infrastructure_failure": False,
                "one_off_code_failure": False,
                "metric_authorized": False,
                "operation_visibility": {
                    "allowed_operations": ["debug_hypothesis", "debug_repair"],
                    "forbidden_operations": [
                        "draft_method_selection",
                        "improve_method_selection",
                        "metric_ranking",
                        "exact_replay",
                    ],
                    "task_scope": "exact_task",
                },
            }
        )
    return output, rejected


def build(args: argparse.Namespace) -> dict[str, Any]:
    source_dir = args.source_dir.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    source_graph = source_dir / "runforest" / "graph.json"
    graph = json.loads(source_graph.read_text(encoding="utf-8"))
    source_graph_sha = sha256_file(source_graph)
    claims = build_claims(graph, source_graph_sha)
    if not claims:
        raise ValueError("No Leaf claims were extracted")
    counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    source_transition_ids = set()
    for claim in claims:
        counts[claim["claim_type"]] = counts.get(claim["claim_type"], 0) + 1
        status_counts[claim["claim_status"]] = status_counts.get(claim["claim_status"], 0) + 1
        source_transition_ids.add(claim["source_transition_id"])
    expected_transitions = {
        str(node.get("id") or "")
        for node in graph.get("nodes") or []
        if isinstance(node, Mapping)
        and node.get("type") == "Transition"
        and node.get("task") == TASK_ID
    }
    if source_transition_ids != expected_transitions:
        raise ValueError("Atomic claim extraction did not cover every Leaf transition")

    bundle: dict[str, Any] = {
        "schema": ATOMIC_CLAIM_BUNDLE_SCHEMA,
        "bundle_version": args.bundle_version,
        "created_at": args.created_at,
        "task_id": TASK_ID,
        "source_graph": str(source_graph),
        "source_graph_sha256": source_graph_sha,
        "source_transition_count": len(expected_transitions),
        "covered_source_transition_count": len(source_transition_ids),
        "claim_count": len(claims),
        "claim_type_counts": dict(sorted(counts.items())),
        "claim_status_counts": dict(sorted(status_counts.items())),
        "taint_contract": {
            "code": "whole-program audit; never inferred from a local repair",
            "metric": "authorized only for strict-clean successful programs",
            "claim": "independent local failure/action/execution verification",
        },
        "claims": claims,
        "bundle_sha256": "",
    }
    bundle["bundle_sha256"] = payload_hash(bundle, "bundle_sha256")
    write_json(output_dir / "atomic_claims.json", bundle)
    write_json(output_dir / "teacher_packet.json", teacher_packet(claims))
    write_json(output_dir / "teacher_output_schema.json", teacher_schema())

    if args.prepare_only:
        return {
            "phase": "prepared",
            "claim_count": len(claims),
            "claim_type_counts": counts,
            "claim_status_counts": status_counts,
            "atomic_claim_bundle_sha256": bundle["bundle_sha256"],
        }

    if args.teacher_response is None:
        raise ValueError("--teacher-response is required unless --prepare-only is used")
    response = json.loads(args.teacher_response.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(response, Mapping):
        raise ValueError("Teacher response must be a JSON object")
    authorized = [
        claim for claim in claims if claim["claim_status"] == AUTHORIZED_DEBUG_STATUS
    ]
    claims_by_id = {str(claim["id"]): claim for claim in authorized}
    claim_id_aliases: dict[str, str] = {}
    claim_alias_receipt: dict[str, Any] = {}
    if args.claim_id_aliases is not None:
        claim_alias_receipt = json.loads(
            args.claim_id_aliases.resolve(strict=True).read_text(encoding="utf-8")
        )
        raw_aliases = claim_alias_receipt.get("aliases")
        if not isinstance(raw_aliases, Mapping):
            raise ValueError("Claim-ID alias receipt has no aliases object")
        claim_id_aliases = {
            str(old): str(new) for old, new in raw_aliases.items()
        }
        if any(new not in claims_by_id for new in claim_id_aliases.values()):
            raise ValueError("Claim-ID alias points outside authorized v7 claims")
    synthetic = [transition_for_claim(claim) for claim in authorized]
    transition_by_claim = {
        str(transition["atomic_claim_id"]): str(transition["id"])
        for transition in synthetic
    }
    graph = json.loads(source_graph.read_text(encoding="utf-8"))
    existing_ids = {str(node.get("id") or "") for node in graph.get("nodes") or []}
    if any(str(node["id"]) in existing_ids for node in synthetic):
        raise ValueError("Synthetic atomic transition ID collision")
    graph.setdefault("nodes", []).extend(synthetic)
    for transition in synthetic:
        graph.setdefault("edges", []).extend(
            [
                {
                    "src": transition["parent_node_id"],
                    "dst": transition["id"],
                    "kind": "has_atomic_claim_transition",
                    "provenance": "leaf_atomic_memory_v7",
                },
                {
                    "src": transition["id"],
                    "dst": transition["child_node_id"],
                    "kind": "atomic_claim_verified_by_execution",
                    "provenance": "leaf_atomic_memory_v7",
                },
            ]
        )
    graph.setdefault("meta", {}).update(
        {
            "artifact_label": "Leaf full-Forest atomic-claim memory v7",
            "artifact_version": args.bundle_version,
            "created_at": args.created_at,
            "source_graph_sha256": source_graph_sha,
            "atomic_claim_bundle_sha256": bundle["bundle_sha256"],
            "atomic_claim_count": len(claims),
            "atomic_debug_authorized_count": len(authorized),
            "atomic_claim_transition_count": len(synthetic),
            "claim_level_taint_separation": True,
            "ranking_contract": "task_first_structured_debug_signature_v3",
        }
    )

    source_recipe_path = source_dir / "recipe_sops.json"
    recipe = json.loads(source_recipe_path.read_text(encoding="utf-8"))
    source_recipe_nodes = recipe.get("nodes")
    if not isinstance(source_recipe_nodes, list):
        raise ValueError("Source Recipe bundle has no nodes")
    deterministic_sops = [
        deterministic_claim_sop(claim, transition_by_claim[str(claim["id"])])
        for claim in authorized
    ]
    generalized_sops, teacher_rejected = teacher_sops(
        response,
        claims_by_id,
        transition_by_claim,
        claim_id_aliases,
    )
    recipe["bundle_version"] = args.bundle_version
    recipe["created_at"] = args.created_at
    recipe["teacher"] = {
        "model_requested": TEACHER_MODEL,
        "execution": "local_codex_exec",
        "response_sha256": sha256_file(args.teacher_response.resolve(strict=True)),
        "atomic_claim_bundle_sha256": bundle["bundle_sha256"],
        "generalized_repair_count": len(generalized_sops),
        "rejected_generalization_count": len(teacher_rejected),
    }
    recipe["routing_contract"] = {
        **dict(recipe.get("routing_contract") or {}),
        "debug": (
            "Repair index first: exact task + structured exception/model/API/operand signature; "
            "claim-level visibility; metric and whole-program taint remain isolated"
        ),
    }
    recipe["nodes"] = [*source_recipe_nodes, *deterministic_sops, *generalized_sops]
    recipe["bundle_sha256"] = ""
    recipe["bundle_sha256"] = payload_hash(recipe, "bundle_sha256")

    runforest_dir = output_dir / "runforest"
    runforest_dir.mkdir(parents=True, exist_ok=True)
    write_json(runforest_dir / "graph.json", graph)
    copy_replace(source_dir / "runforest" / "index.npz", runforest_dir / "index.npz")
    for name in ("TASK_AUDIT_REPORT.json",):
        source = source_dir / "runforest" / name
        if source.exists():
            copy_replace(source, runforest_dir / name)
    for name in ("evidence_manifest.json", "implementation_capsules.json"):
        copy_replace(source_dir / name, output_dir / name)
    write_json(output_dir / "recipe_sops.json", recipe)
    write_json(output_dir / "teacher_response_gpt56sol.json", response)

    release: dict[str, Any] = {
        "schema": "mlevolve_leaf_atomic_memory_release_v1",
        "bundle_version": args.bundle_version,
        "created_at": args.created_at,
        "teacher_model": TEACHER_MODEL,
        "source_graph_sha256": source_graph_sha,
        "atomic_claim_bundle_sha256": bundle["bundle_sha256"],
        "source_transition_count": len(expected_transitions),
        "covered_transition_count": len(source_transition_ids),
        "atomic_debug_authorized_count": len(authorized),
        "deterministic_repair_sop_count": len(deterministic_sops),
        "teacher_generalized_repair_sop_count": len(generalized_sops),
        "teacher_rejected_outputs": teacher_rejected,
        "teacher_claim_id_aliases": claim_id_aliases,
        "teacher_claim_id_alias_receipt": claim_alias_receipt,
        "files": {},
        "quality_gates": {
            "all_leaf_transitions_covered": source_transition_ids == expected_transitions,
            "no_atomic_claim_authorizes_metric": all(
                claim.get("metric_authorized") is False for claim in authorized
            ),
            "all_atomic_repairs_debug_only": all(
                set(claim["operation_visibility"]["allowed_operations"])
                == {"debug_hypothesis", "debug_repair"}
                for claim in authorized
            ),
            "teacher_cites_only_authorized_claims": not teacher_rejected,
            "teacher_claim_id_aliases_resolve_to_authorized_claims": all(
                value in claims_by_id for value in claim_id_aliases.values()
            ),
            "source_program_and_claim_taint_separated": any(
                claim["taint"]["code"] == "quarantine"
                and claim["taint"]["claim"] == "clean"
                for claim in authorized
            ),
        },
        "release_sha256": "",
    }
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "release_report.json":
            release["files"][str(path.relative_to(output_dir))] = sha256_file(path)
    release["release_sha256"] = payload_hash(release, "release_sha256")
    write_json(output_dir / "release_report.json", release)
    return release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-response", type=Path)
    parser.add_argument("--claim-id-aliases", type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--bundle-version", default="leaf-atomic-memory-v7-20260811"
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    report = build(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
