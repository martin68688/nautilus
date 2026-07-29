"""Claim-level Authority gateway for cold-start methodology references."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from agents.memory.sop_visibility_gateway import SOPVisibilityGateway
from authority.models import (
    ClaimType,
    GenerationStage,
    GovernanceStage,
    Operation,
    TaskContext,
    VisibilityRequest,
    VisibleSOPPack,
)


def _nodes(candidates: Iterable[dict[str, str]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        sop_id = str(candidate["candidate_id"])
        clause_id = f"{sop_id}::claim"
        output[sop_id] = {
            "id": sop_id,
            "type": "SOP",
            "title": str(candidate.get("title") or sop_id),
            "action": str(candidate.get("text") or ""),
            "clauses": [
                {
                    "clause_id": clause_id,
                    "sop_id": sop_id,
                    "text": str(candidate.get("text") or ""),
                    "retrieval_text": str(candidate.get("title") or ""),
                    "claim_refs": [str(candidate["claim_id"])],
                    "claim_types": [ClaimType.METHOD_HYPOTHESIS.value],
                    "source_artifact_refs": [str(candidate["ref_id"])],
                    "source_run_ids": [],
                    "source_task_ids": [],
                    "source_domains": [],
                    "transfer_scope": "general",
                    "protocol_agnostic": True,
                    "permitted_operations": [
                        Operation.INSPECT.value,
                        Operation.GENERATE_CANDIDATE.value,
                    ],
                    "permitted_generation_stages": [
                        GenerationStage.DRAFT.value
                    ],
                    "permitted_governance_stages": [
                        GovernanceStage.RETRIEVAL.value
                    ],
                    "publication_class": "candidate",
                    # Legacy methodology has no trusted Claim/path until a
                    # future bundle explicitly certifies this deterministic ID.
                    "legacy_status": "legacy_uncertified",
                }
            ],
            "methodology_ref_id": str(candidate["ref_id"]),
            "methodology_category": str(candidate.get("category") or ""),
            "content_sha256": str(candidate.get("content_sha256") or ""),
        }
    return output


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def evaluate_methodology_visibility(
    agent: Any,
    candidates: list[dict[str, str]],
) -> tuple[str, list[str], VisibleSOPPack | None]:
    """Record raw proposals, evaluate Authority, then render admitted text."""

    if not candidates:
        return "", [], None
    adapter = agent.evaluation_authority
    nodes = _nodes(candidates)
    raw_payload = {
        "schema": "methodology_raw_candidate_set_v1",
        "run_id": adapter.run_id,
        "task_id": adapter.task_id,
        "decision_stage": GenerationStage.DRAFT.value,
        "operation": Operation.GENERATE_CANDIDATE.value,
        "protocol_ref": adapter.active_protocol.key(),
        "raw_candidates": [dict(item) for item in candidates],
    }
    # This write precedes every admission, ranking, and Prompt decision.
    _atomic_json(
        Path(agent.cfg.log_dir) / "methodology_raw_candidates.json",
        raw_payload,
    )

    rollout = getattr(adapter, "rollout", None)
    gateway = SOPVisibilityGateway(
        nodes,
        mode=adapter.mode,
        authority_engine=adapter.engine,
        decision_lookup=adapter.engine.decisions.get,
        retrieval_profile="full_decision_admissibility",
        enforce_operations=sorted(
            getattr(rollout, "enforce_operations", ()) or ()
        ),
        enforce_generation_stages=sorted(
            getattr(rollout, "enforce_generation_stages", ()) or ()
        ),
        enforce_governance_stages=sorted(
            getattr(rollout, "enforce_governance_stages", ()) or ()
        ),
    )
    family = "general"
    layer = getattr(agent, "external_skill_memory", None)
    resolver = getattr(layer, "_task_family_for_query", None)
    if callable(resolver):
        family = str(resolver(adapter.task_id, agent.task_desc) or "general")
    request = VisibilityRequest(
        operation=Operation.GENERATE_CANDIDATE,
        generation_stage=GenerationStage.DRAFT,
        governance_stage=GovernanceStage.RETRIEVAL,
        active_protocol=adapter.active_protocol,
        task_context=TaskContext(
            task_id=adapter.task_id,
            task_family=family,
        ),
        memory_bundle_version=(
            adapter.memory_snapshot.base_bundle.bundle_version
            if adapter.memory_snapshot is not None
            else "unbound-methodology"
        ),
        token_budget=int(
            getattr(agent.cfg.external_skill_memory, "visibility_token_budget", 4096)
            or 4096
        ),
        requesting_component="engine.coldstart.methodology_agent",
        authority_policy_version=adapter.engine.policy_version,
    )
    pack = gateway.evaluate(request, candidate_sop_ids=nodes)
    prospective = getattr(agent, "prospective_audit", None)
    if prospective is not None:
        prospective.record_visibility(
            pack,
            gateway,
            source="dynamic_methodology_visibility",
        )
    candidate_by_id = {item["candidate_id"]: item for item in candidates}
    sections: list[str] = []
    refs: list[str] = []
    for sop_id in pack.effective_sop_ids:
        rendered = pack.rendered_by_sop.get(sop_id) or {}
        prompt_text = str(rendered.get("prompt_text") or "").strip()
        if not prompt_text:
            continue
        candidate = candidate_by_id[sop_id]
        sections.append(
            f"### [{candidate.get('category')}] {candidate.get('title')}\n\n"
            f"{prompt_text}"
        )
        refs.append(sop_id)
        refs.extend(str(value) for value in rendered.get("clause_ids") or [])
    report = {
        **raw_payload,
        "schema": "methodology_visibility_report_v1",
        "visibility_trace": pack.visibility_trace,
        "effective_sop_ids": list(pack.effective_sop_ids),
        "effective_clause_ids": list(pack.effective_clause_ids),
        "suppressed_clause_ids": list(pack.suppressed_clause_refs),
    }
    _atomic_json(Path(agent.cfg.log_dir) / "methodology_visibility.json", report)
    if not sections:
        return "", [], pack
    text = (
        "\n\n---\n## Authority-Gated Methodology Insights\n"
        "Only the Claim-use entries admitted for this task, operation, stage, "
        "and Protocol are visible below.\n\n"
        + "\n\n---\n\n".join(sections)
    )
    return text, list(dict.fromkeys(refs)), pack


__all__ = ["evaluate_methodology_visibility"]
