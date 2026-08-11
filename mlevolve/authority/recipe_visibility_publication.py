from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from typing import Any

from .domain_scope import canonical_domain


RECIPE_VISIBILITY_PUBLICATION_SCHEMA = "recipe_visibility_publication_v1"
DECLARED_SCOPE_MASK_SCHEMA = "declared_scope_visibility_masks_v1"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_clause_id(
    *,
    sop_id: str,
    claim_id: str,
    active_protocol_ref: str,
) -> str:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "active_protocol_ref": active_protocol_ref,
                "claim_id": claim_id,
                "sop_id": sop_id,
            }
        ).encode("utf-8")
    ).hexdigest()[:24]
    return f"recipe-debug-clause::{digest}"


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return []
    return sorted(
        {
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        }
    )


def _failure_text(node: Mapping[str, Any]) -> str:
    signature = node.get("failure_signature")
    signature = dict(signature) if isinstance(signature, Mapping) else {}
    return str(
        signature.get("pattern")
        or signature.get("root_cause")
        or node.get("teacher_failure_signature_summary")
        or ""
    ).strip()


def _repair_text(node: Mapping[str, Any]) -> str:
    action = node.get("repair_action")
    if isinstance(action, Mapping):
        summary = str(action.get("summary") or "").strip()
        if summary:
            return summary
        steps = _strings(action.get("steps"))
        if steps:
            return " ".join(steps)
    return str(action or node.get("teacher_repair_action_summary") or "").strip()


def _source_artifact(node: Mapping[str, Any]) -> str:
    for field in (
        "failure_node_ids",
        "source_parent_node_ids",
        "source_node_ids",
    ):
        values = _strings(node.get(field))
        if values:
            return values[0]
    return ""


def compile_recipe_debug_visibility(
    recipe_payload: Mapping[str, Any],
    *,
    active_protocol_ref: str,
) -> dict[str, Any]:
    """Compile claim-backed task-local L3 repairs into formal visibility data.

    The compiler is deliberately task-agnostic and conservative.  It only
    publishes a Recipe entry when the entry is an L3 repair with an explicit
    source Claim and a concrete source artifact.  Method, score, replay, and
    cross-task permissions are never inferred from prose.
    """

    active_protocol_ref = str(active_protocol_ref or "").strip()
    protocol_key, separator, protocol_hash = active_protocol_ref.partition("#")
    if (
        not separator
        or "@" not in protocol_key
        or len(protocol_hash) != 64
        or any(character not in "0123456789abcdef" for character in protocol_hash)
    ):
        raise ValueError("active_protocol_ref must be a hash-bound ProtocolRef")

    nodes = recipe_payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("Recipe publication requires a non-empty nodes list")

    clauses: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    seen_sop_ids: set[str] = set()
    seen_clause_ids: set[str] = set()
    l3_count = 0

    for raw in nodes:
        if not isinstance(raw, Mapping):
            raise ValueError("Recipe contains a non-object node")
        node = dict(raw)
        if str(node.get("abstraction_level") or "") != "L3_repair":
            continue
        l3_count += 1
        sop_id = str(node.get("id") or "").strip()
        if not sop_id:
            raise ValueError("L3 Recipe entry is missing an ID")
        if sop_id in seen_sop_ids:
            raise ValueError(f"Duplicate L3 Recipe SOP ID: {sop_id}")
        seen_sop_ids.add(sop_id)

        claim_ids = _strings(node.get("source_claim_ids"))
        if not claim_ids:
            skipped.append(
                {"sop_id": sop_id, "reason": "missing_source_claim_ids"}
            )
            continue
        source_artifact = _source_artifact(node)
        if not source_artifact:
            raise ValueError(
                f"Claim-backed L3 Recipe entry lacks a source artifact: {sop_id}"
            )
        task_id = str(node.get("task_id") or "").strip()
        if not task_id:
            raise ValueError(f"Claim-backed L3 Recipe entry lacks task_id: {sop_id}")
        task_family = str(
            node.get("task_family")
            or node.get("task_domain")
            or node.get("task_type")
            or ""
        ).strip()
        source_domain = canonical_domain(
            node.get("task_domain") or task_family
        )
        failure = _failure_text(node)
        repair = _repair_text(node)
        when_to_use = str(node.get("when_to_use") or "").strip()
        if not failure or not repair:
            raise ValueError(
                f"Claim-backed L3 Recipe entry lacks failure/repair text: {sop_id}"
            )
        source_transition_refs = _strings(
            node.get("supporting_transition_ids")
            or node.get("source_transition_ids")
        )
        source_run_ids = _strings(
            node.get("distinct_run_ids") or node.get("source_run_ids")
        )
        # Exact task identity is the operative scope.  Keep family/domain as
        # source provenance, but do not let a stale taxonomy alias override an
        # exact task match (for example, older Leaf manifests used a tabular
        # routing family for a multimodal task).
        task_scope: dict[str, Any] = {"task_ids": [task_id]}

        # One SOP represents one repair decision and therefore receives one
        # visibility Clause.  When distillation merged several equivalent
        # source Claims, the first stable Claim is the Authority subject while
        # the complete set remains immutable derivation provenance.  Creating
        # one Clause per Claim would duplicate the same repair in the Router
        # pool and distort Top-K ranking.
        primary_claim_id = claim_ids[0]
        clause_id = _stable_clause_id(
            sop_id=sop_id,
            claim_id=primary_claim_id,
            active_protocol_ref=active_protocol_ref,
        )
        if clause_id in seen_clause_ids:
            raise ValueError(f"Compiled Recipe clause collision: {clause_id}")
        seen_clause_ids.add(clause_id)
        retrieval_lines = [
            str(node.get("title") or "").strip(),
            f"Failure: {failure}",
            f"Repair: {repair}",
            f"Use when: {when_to_use}" if when_to_use else "",
        ]
        retrieval_text = "\n".join(
            line for line in retrieval_lines if line
        )
        clauses.append(
            {
                "schema": "sop_clause_v1",
                "clause_id": clause_id,
                "sop_id": sop_id,
                "text": retrieval_text,
                "retrieval_text": retrieval_text,
                "claim_refs": [primary_claim_id],
                "claim_types": ["debug_repair"],
                "source_artifact_refs": [source_artifact],
                "source_transition_refs": source_transition_refs,
                "source_run_ids": source_run_ids,
                "source_task_ids": [task_id],
                "source_task_families": [task_family] if task_family else [],
                "source_domains": [source_domain] if source_domain else [],
                "transfer_scope": "",
                "protocol_scope": [active_protocol_ref],
                "task_scope": task_scope,
                "permitted_operations": ["debug_hypothesis"],
                "permitted_generation_stages": ["debug"],
                "permitted_governance_stages": ["retrieval"],
                "publication_class": "diagnostic",
                "authority_decision_refs": [],
                "receipt_refs": [],
                "derivation_refs": sorted(
                    set(claim_ids) | set(source_transition_refs)
                ),
                "additional_source_claim_refs": claim_ids[1:],
                "applies_when": [when_to_use] if when_to_use else [failure],
                "prevents": [failure],
                "protocol_agnostic": False,
                "legacy_status": "native_recipe_l3_debug_v1",
                "publication_origin": "recipe_l3_debug_compiler_v1",
            }
        )

    clauses.sort(key=lambda row: (str(row["sop_id"]), str(row["clause_id"])))
    mask_key = "|".join(
        [active_protocol_ref, "debug_hypothesis", "debug", "retrieval"]
    )
    masks = {
        "schema": DECLARED_SCOPE_MASK_SCHEMA,
        "semantics": (
            "Declared-scope prefilter only; runtime Authority evaluation and "
            "the task-local atomic repair hard gate still apply."
        ),
        "active_protocol_refs": [active_protocol_ref],
        "masks": {mask_key: [row["clause_id"] for row in clauses]},
    }
    report = {
        "schema": RECIPE_VISIBILITY_PUBLICATION_SCHEMA,
        "source_recipe_schema": str(recipe_payload.get("schema") or ""),
        "source_recipe_bundle_sha256": str(
            recipe_payload.get("bundle_sha256") or ""
        ),
        "active_protocol_ref": active_protocol_ref,
        "recipe_node_count": len(nodes),
        "l3_recipe_count": l3_count,
        "published_clause_count": len(clauses),
        "published_sop_count": len({row["sop_id"] for row in clauses}),
        "skipped_count": len(skipped),
        "skipped_reason_counts": dict(
            sorted(Counter(row["reason"] for row in skipped).items())
        ),
        "skipped": sorted(skipped, key=lambda row: row["sop_id"]),
        "mask_keys": [mask_key],
        "claim_types": ["debug_repair"],
        "publication_classes": ["diagnostic"],
        "cross_task_transfer_enabled": False,
        "score_or_metric_claims_published": False,
        "replay_claims_published": False,
    }
    return {"clauses": clauses, "masks": masks, "report": report}


__all__ = [
    "DECLARED_SCOPE_MASK_SCHEMA",
    "RECIPE_VISIBILITY_PUBLICATION_SCHEMA",
    "compile_recipe_debug_visibility",
]
