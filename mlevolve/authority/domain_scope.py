from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


SAME_DOMAIN = "same_domain"
DOMAIN_GENERAL = "domain_general"
TRANSFER_SCOPES = frozenset({SAME_DOMAIN, DOMAIN_GENERAL})


def canonical_domain(value: object) -> str:
    """Map task-family labels to a stable experiment domain.

    The mapping is deliberately deterministic and metadata-only.  It never
    infers a domain from SOP prose, because doing so would let generated text
    widen its own transfer scope.  Unknown non-empty labels retain a normalized
    identifier so synthetic and future corpora can still use exact-domain
    matching; missing labels remain missing and therefore fail closed.
    """

    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    tokens = set(normalized.split("_"))
    if "multimodal" in tokens or {"multi", "modal"} <= tokens:
        return "multimodal"
    if tokens & {"image", "images", "vision", "visual"}:
        return "image"
    if tokens & {"nlp", "language", "text", "linguistic"}:
        return "nlp"
    if tokens & {"audio", "speech", "sound", "acoustic"}:
        return "audio"
    if tokens & {"tabular", "structured", "others", "other"}:
        return "tabular"
    return normalized


def normalize_transfer_scope(value: object) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+", "_", str(value or "").strip().lower()
    ).strip("_")
    aliases = {
        "same_modality": SAME_DOMAIN,
        "same_task_family": SAME_DOMAIN,
        "cross_task_same_domain": SAME_DOMAIN,
        "general": DOMAIN_GENERAL,
        "cross_domain_general": DOMAIN_GENERAL,
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in TRANSFER_SCOPES else ""


def transfer_is_compatible(
    source_domains: Iterable[object],
    target_domain: object,
    transfer_scope: object,
) -> bool:
    """Return whether an explicitly scoped clause may cross task boundaries.

    ``domain_general`` is the only cross-domain escape hatch.  A missing or
    unknown scope never broadens access.  ``same_domain`` additionally requires
    every bound source to have the same non-empty domain as the target.
    """

    scope = normalize_transfer_scope(transfer_scope)
    if scope == DOMAIN_GENERAL:
        return True
    if scope != SAME_DOMAIN:
        return False
    target = canonical_domain(target_domain)
    sources = {canonical_domain(value) for value in source_domains}
    sources.discard("")
    return bool(target and sources == {target})


def _string_values(values: object) -> set[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, Iterable):
        return set()
    return {
        str(value)
        for value in values
        if value not in {None, ""}
    }


def audit_same_domain_task_heldout_exposures(
    exposures: Iterable[Mapping[str, Any]],
    bundle_clauses: Iterable[Mapping[str, Any]],
    *,
    target_task_id: str,
    target_domain: str,
    certified_clause_id: str,
    certified_source_task_id: str,
    require_certified_exposure: bool = True,
) -> dict[str, Any]:
    """Audit effective Prompt exposures for a same-domain task-heldout run.

    The positive transfer path is deliberately narrow: cross-task candidate
    generation must use the one certified method clause selected for the
    canary.  The plan separately permits legacy diagnostic knowledge in a
    Debug view, so Bundle-declared diagnostic clauses remain admissible only
    for ``debug_hypothesis`` at the Debug/Retrieval axes.  Every event must be
    the v2 host event that binds these axes into its contract hash.
    """

    target_task_id = str(target_task_id or "").strip()
    target_domain = canonical_domain(target_domain)
    certified_clause_id = str(certified_clause_id or "").strip()
    certified_source_task_id = str(certified_source_task_id or "").strip()
    if not all(
        (
            target_task_id,
            target_domain,
            certified_clause_id,
            certified_source_task_id,
        )
    ):
        raise ValueError("Task-heldout exposure audit requires bound target/source IDs")

    clauses: dict[str, dict[str, Any]] = {}
    duplicate_clause_ids: set[str] = set()
    for raw in bundle_clauses:
        clause = dict(raw)
        clause_id = str(clause.get("clause_id") or "").strip()
        if not clause_id:
            continue
        if clause_id in clauses and clauses[clause_id] != clause:
            duplicate_clause_ids.add(clause_id)
        clauses[clause_id] = clause
    if duplicate_clause_ids:
        raise ValueError(
            f"Conflicting Bundle clause IDs: {sorted(duplicate_clause_ids)}"
        )
    if certified_clause_id not in clauses:
        raise ValueError("Certified canary clause is absent from the Bundle")

    values = [dict(value) for value in exposures]
    invalid: list[dict[str, Any]] = []
    certified_count = 0
    diagnostic_count = 0
    classified_count = 0

    for index, payload in enumerate(values):
        clause_id = str(payload.get("clause_id") or "")
        clause = clauses.get(clause_id)
        reasons: list[str] = []
        if payload.get("schema") != "experience_exposure_event_v2":
            reasons.append("exposure_schema_not_v2")
        if clause is None:
            reasons.append("clause_absent_from_bound_bundle")

        source_tasks = _string_values(payload.get("source_task_ids"))
        source_domains = {
            canonical_domain(value)
            for value in _string_values(payload.get("source_domains"))
        }
        source_domains.discard("")
        target_scope = payload.get("target_scope")
        target_scope = dict(target_scope) if isinstance(target_scope, Mapping) else {}
        operation = str(payload.get("operation") or "")
        generation_stage = str(payload.get("generation_stage") or "")
        governance_stage = str(payload.get("governance_stage") or "")
        publication_class = str(payload.get("publication_class") or "")

        if not source_tasks:
            reasons.append("missing_source_task_ids")
        if target_task_id in source_tasks:
            reasons.append("heldout_target_appears_in_sources")
        if source_domains != {target_domain}:
            reasons.append("source_domain_mismatch")
        if normalize_transfer_scope(payload.get("transfer_scope")) != SAME_DOMAIN:
            reasons.append("transfer_scope_not_same_domain")
        if target_scope.get("task_id") != target_task_id:
            reasons.append("target_task_mismatch")
        if canonical_domain(target_scope.get("domain")) != target_domain:
            reasons.append("target_domain_mismatch")
        if len(str(payload.get("prompt_sha256") or "")) != 64:
            reasons.append("prompt_hash_missing")
        if not operation or not generation_stage or not governance_stage:
            reasons.append("decision_axes_missing")

        if clause is not None:
            declared_scope = clause.get("task_scope")
            declared_scope = (
                dict(declared_scope)
                if isinstance(declared_scope, Mapping)
                else {}
            )
            declared_tasks = _string_values(clause.get("source_task_ids"))
            if not declared_tasks:
                declared_tasks = _string_values(
                    declared_scope.get("task_ids")
                    or [declared_scope.get("task_id")]
                )
            declared_domains = {
                canonical_domain(value)
                for value in _string_values(clause.get("source_domains"))
            }
            declared_domains.discard("")
            if source_tasks != declared_tasks:
                reasons.append("source_task_binding_mismatch")
            if source_domains != declared_domains:
                reasons.append("source_domain_binding_mismatch")
            if normalize_transfer_scope(clause.get("transfer_scope")) != SAME_DOMAIN:
                reasons.append("bundle_clause_not_same_domain")
            if publication_class != str(
                clause.get("publication_class") or "diagnostic"
            ):
                reasons.append("publication_binding_mismatch")
            permitted_operations = _string_values(
                clause.get("permitted_operations")
            )
            permitted_generation = _string_values(
                clause.get("permitted_generation_stages")
            )
            permitted_governance = _string_values(
                clause.get("permitted_governance_stages")
            )
            if permitted_operations and operation not in permitted_operations:
                reasons.append("operation_outside_declared_scope")
            if permitted_generation and generation_stage not in permitted_generation:
                reasons.append("generation_stage_outside_declared_scope")
            if permitted_governance and governance_stage not in permitted_governance:
                reasons.append("governance_stage_outside_declared_scope")

            if clause_id == certified_clause_id:
                certified_count += 1
                if source_tasks != {certified_source_task_id}:
                    reasons.append("certified_source_task_mismatch")
                if publication_class != "certified":
                    reasons.append("certified_publication_required")
                if operation != "generate_candidate":
                    reasons.append("certified_method_operation_mismatch")
                if _string_values(clause.get("claim_types")) != {
                    "method_hypothesis"
                }:
                    reasons.append("certified_method_claim_type_mismatch")
                if not _string_values(clause.get("receipt_refs")):
                    reasons.append("certified_method_receipts_missing")
            elif operation == "debug_hypothesis":
                diagnostic_count += 1
                if publication_class != "diagnostic":
                    reasons.append("debug_requires_diagnostic_publication")
                if generation_stage != "debug":
                    reasons.append("debug_generation_stage_mismatch")
                if governance_stage != "retrieval":
                    reasons.append("debug_governance_stage_mismatch")
            else:
                reasons.append("noncertified_behavior_influencing_exposure")

        if not reasons:
            classified_count += 1
        else:
            invalid.append(
                {
                    "index": index,
                    "contract_id": str(payload.get("contract_id") or ""),
                    "clause_id": clause_id,
                    "operation": operation,
                    "generation_stage": generation_stage,
                    "reasons": sorted(set(reasons)),
                }
            )

    if require_certified_exposure and certified_count == 0:
        invalid.append(
            {
                "index": -1,
                "contract_id": "",
                "clause_id": certified_clause_id,
                "operation": "generate_candidate",
                "generation_stage": "",
                "reasons": ["certified_method_was_never_exposed"],
            }
        )

    return {
        "schema": "same_domain_task_heldout_exposure_audit_v1",
        "valid": not invalid,
        "target_task_id": target_task_id,
        "target_domain": target_domain,
        "certified_clause_id": certified_clause_id,
        "certified_source_task_id": certified_source_task_id,
        "exposure_event_count": len(values),
        "classified_exposure_count": classified_count,
        "certified_method_exposure_count": certified_count,
        "diagnostic_debug_exposure_count": diagnostic_count,
        "unique_contract_count": len(
            {
                str(payload.get("contract_id") or "")
                for payload in values
                if payload.get("contract_id")
            }
        ),
        "exposed_clause_ids": sorted(
            {
                str(payload.get("clause_id") or "")
                for payload in values
                if payload.get("clause_id")
            }
        ),
        "exposed_source_task_ids": sorted(
            {
                task_id
                for payload in values
                for task_id in _string_values(payload.get("source_task_ids"))
            }
        ),
        "invalid_exposure_count": len(invalid),
        "invalid_exposures": invalid,
    }


__all__ = [
    "DOMAIN_GENERAL",
    "SAME_DOMAIN",
    "TRANSFER_SCOPES",
    "canonical_domain",
    "audit_same_domain_task_heldout_exposures",
    "normalize_transfer_scope",
    "transfer_is_compatible",
]
