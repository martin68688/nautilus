from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .models import Claim, ClaimType, Operation, ProtocolRef, canonical_operation
from .protocol_registry import canonical_json


@dataclass(frozen=True)
class ClaimBoundaryProposal:
    """Untrusted LLM suggestion limited to wording and fact boundaries."""

    claim_type: str
    statement: str
    source_refs: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    boundary: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionResult:
    artifact_id: str
    claims: list[Claim]
    deterministic_facts: dict[str, Any]
    bindings: dict[str, dict[str, list[str]]]
    quarantined_proposals: list[dict[str, Any]]

    def claims_of_type(self, claim_type: ClaimType) -> list[Claim]:
        return [claim for claim in self.claims if claim.claim_type == claim_type]


def _stable_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _code_sha256(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _metric_value(node: Any) -> Any:
    return getattr(getattr(node, "metric", None), "value", None)


def _issue_code(issue: dict[str, Any], index: int) -> str:
    return str(
        issue.get("issue_code")
        or issue.get("code")
        or issue.get("category")
        or f"audit_issue_{index}"
    )


def _short_text(value: Any, limit: int = 360) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _claim_id(
    artifact_kind: str,
    artifact_id: str,
    claim_type: ClaimType,
    boundary: dict[str, Any],
) -> str:
    # Preserve the existing SCORE ref for one migration cycle.
    if artifact_kind == "node" and claim_type == ClaimType.SCORE:
        return f"node:{artifact_id}:score"
    digest = _stable_digest(
        {
            "artifact_kind": artifact_kind,
            "artifact_id": artifact_id,
            "claim_type": claim_type.value,
            "boundary": boundary,
        }
    )[:16]
    return f"{artifact_kind}:{artifact_id}:claim:{claim_type.value}:{digest}"


class ClaimDecomposer:
    def decompose_node(
        self,
        node: Any,
        protocol_ref: ProtocolRef,
        task_id: str,
        *,
        artifact_kind: str = "node",
        source_refs: Iterable[str] = (),
        proposals: Iterable[ClaimBoundaryProposal] = (),
        legacy_static_only: bool = False,
    ) -> DecompositionResult:
        artifact_id = str(node.id)
        code = str(getattr(node, "code", "") or "")
        plan = _short_text(
            getattr(node, "plan", "")
            or getattr(node, "code_summary", "")
            or getattr(node, "analysis", "")
        )
        stage = str(getattr(node, "stage", "") or "unknown").lower()
        metric = _metric_value(node)
        method_fingerprint = str(
            getattr(node, "method_fingerprint", "") or _code_sha256(code)
        )
        audit = getattr(node, "leakage_audit", None) or {}
        issues = [item for item in audit.get("issues", []) if isinstance(item, dict)]
        if audit and audit.get("status") not in {None, "", "clean"} and not issues:
            issues = [{
                "issue_code": f"AUDIT_STATUS_{str(audit.get('status')).upper()}",
                "category": "audit_status",
                "evidence": str(audit.get("status")),
            }]

        source_artifact_refs = sorted(set([
            f"{artifact_kind}:{artifact_id}",
            *(str(value) for value in source_refs),
        ]))
        evidence_catalog: set[str] = set()
        claim_rows: list[tuple[ClaimType, str, list[str], dict[str, Any]]] = []

        execution_observed = getattr(node, "exec_time", None) is not None or metric is not None
        if execution_observed:
            evidence_ref = f"execution:{artifact_kind}:{artifact_id}"
            evidence_catalog.add(evidence_ref)
            claim_rows.append((
                ClaimType.EXECUTED,
                f"Artifact {artifact_id} was observed executing; this does not certify its score.",
                [evidence_ref],
                {"fact": "execution_observed", "stage": stage},
            ))

        if code or plan:
            evidence_ref = f"method:{artifact_kind}:{artifact_id}:{method_fingerprint[:16]}"
            evidence_catalog.add(evidence_ref)
            statement = f"Provisional method hypothesis for {artifact_id}: {plan or 'code-bearing method candidate'}"
            claim_rows.append((
                ClaimType.METHOD_HYPOTHESIS,
                statement,
                [evidence_ref],
                {"fact": "method_candidate", "method_fingerprint": method_fingerprint},
            ))

        repair_text = " ".join((plan, code[:4000])).lower()
        repair_signals = sorted(set(re.findall(
            r"\b(?:oof|out[- ]of[- ]fold|sample_id|index alignment|align(?:ment|ed)?|repair|fix)\b",
            repair_text,
        )))
        if stage == "debug" or repair_signals:
            evidence_ref = f"repair:{artifact_kind}:{artifact_id}"
            evidence_catalog.add(evidence_ref)
            claim_rows.append((
                ClaimType.DEBUG_REPAIR,
                f"Debug/repair candidate for {artifact_id}: {plan or 'localized code repair'}",
                [evidence_ref],
                {"fact": "debug_repair_candidate", "stage": stage, "signals": repair_signals},
            ))

        for index, issue in enumerate(issues):
            code_value = _issue_code(issue, index)
            evidence_ref = f"audit:{artifact_kind}:{artifact_id}:{code_value}"
            evidence_catalog.add(evidence_ref)
            evidence = _short_text(issue.get("evidence") or issue.get("message") or code_value)
            claim_rows.append((
                ClaimType.AUDIT_FINDING,
                f"Audit finding {code_value} for {artifact_id}: {evidence}",
                [evidence_ref],
                {
                    "fact": "audit_finding",
                    "issue_code": code_value,
                    "category": str(issue.get("category") or ""),
                    "severity": str(issue.get("severity") or ""),
                },
            ))

        if metric is not None:
            evidence_ref = f"metric:{artifact_kind}:{artifact_id}"
            evidence_catalog.add(evidence_ref)
            claim_rows.append((
                ClaimType.SCORE,
                f"Artifact {artifact_id} reported score {metric} under {protocol_ref.key()}.",
                [evidence_ref],
                {"fact": "reported_score", "metric": metric},
            ))

        legacy_status = "legacy_static_only" if legacy_static_only else "native_v1"
        parent_claims = list(getattr(node, "derived_from_refs", []) or [])
        claims = [
            Claim(
                claim_id=_claim_id(artifact_kind, artifact_id, claim_type, boundary),
                claim_type=claim_type,
                subject_artifact_id=artifact_id,
                task_scope={"task_id": task_id},
                method_fingerprint=method_fingerprint,
                protocol_ref=protocol_ref,
                statement=statement,
                parent_claims=parent_claims,
                source_artifact_refs=source_artifact_refs,
                evidence_refs=sorted(evidence_refs),
                boundary=boundary,
                legacy_status=legacy_status,
            )
            for claim_type, statement, evidence_refs, boundary in claim_rows
        ]

        claims, quarantined = self._bind_proposals(
            claims,
            proposals,
            source_catalog=set(source_artifact_refs),
            evidence_catalog=evidence_catalog,
        )
        claim_refs = getattr(node, "claim_refs", None)
        if isinstance(claim_refs, list):
            for claim in claims:
                if claim.claim_id not in claim_refs:
                    claim_refs.append(claim.claim_id)

        facts = {
            "artifact_kind": artifact_kind,
            "artifact_id": artifact_id,
            "task_id": task_id,
            "stage": stage,
            "execution_observed": execution_observed,
            "metric_reported": metric is not None,
            "metric": metric,
            "method_fingerprint": method_fingerprint,
            "code_sha256": _code_sha256(code),
            "repair_signals": repair_signals,
            "audit_status": str(audit.get("status") or "unavailable"),
            "audit_issue_codes": [_issue_code(issue, index) for index, issue in enumerate(issues)],
            "legacy_status": legacy_status,
        }
        bindings = {
            claim.claim_id: {
                "source_artifact_refs": claim.source_artifact_refs,
                "evidence_refs": claim.evidence_refs,
            }
            for claim in claims
        }
        return DecompositionResult(artifact_id, claims, facts, bindings, quarantined)

    def _bind_proposals(
        self,
        claims: list[Claim],
        proposals: Iterable[ClaimBoundaryProposal],
        *,
        source_catalog: set[str],
        evidence_catalog: set[str],
    ) -> tuple[list[Claim], list[dict[str, Any]]]:
        bound = list(claims)
        quarantined: list[dict[str, Any]] = []
        for proposal in proposals:
            try:
                proposal_type = ClaimType(str(proposal.claim_type))
            except ValueError:
                quarantined.append({"proposal": proposal.statement, "reason": "unknown_claim_type"})
                continue
            if not set(proposal.source_refs).issubset(source_catalog):
                quarantined.append({"proposal": proposal.statement, "reason": "unbound_source_ref"})
                continue
            if not set(proposal.evidence_refs).issubset(evidence_catalog):
                quarantined.append({"proposal": proposal.statement, "reason": "unbound_evidence_ref"})
                continue
            candidates = [claim for claim in bound if claim.claim_type == proposal_type]
            if proposal.evidence_refs:
                candidates = [
                    claim for claim in candidates
                    if set(proposal.evidence_refs).issubset(set(claim.evidence_refs))
                ]
            if len(candidates) != 1:
                quarantined.append({
                    "proposal": proposal.statement,
                    "reason": "proposal_does_not_match_one_deterministic_fact",
                })
                continue
            target = candidates[0]
            replacement = replace(
                target,
                statement=_short_text(proposal.statement) or target.statement,
                boundary={**target.boundary, "llm_boundary": dict(proposal.boundary)},
            )
            bound[bound.index(target)] = replacement
        return bound, quarantined


def decompose_node_claims(
    node: Any,
    protocol_ref: ProtocolRef,
    task_id: str,
    **kwargs: Any,
) -> DecompositionResult:
    return ClaimDecomposer().decompose_node(node, protocol_ref, task_id, **kwargs)


def select_claim_for_operation(
    result: DecompositionResult,
    operation: Operation | str,
) -> Claim:
    operation = canonical_operation(operation)
    preference = {
        Operation.INSPECT: (
            ClaimType.AUDIT_FINDING,
            ClaimType.DEBUG_REPAIR,
            ClaimType.METHOD_HYPOTHESIS,
            ClaimType.EXECUTED,
            ClaimType.SCORE,
        ),
        Operation.DEBUG_HYPOTHESIS: (ClaimType.DEBUG_REPAIR, ClaimType.AUDIT_FINDING),
        Operation.REPAIR_SEED: (ClaimType.DEBUG_REPAIR, ClaimType.METHOD_HYPOTHESIS),
        Operation.GENERATE_CANDIDATE: (ClaimType.METHOD_HYPOTHESIS, ClaimType.DEBUG_REPAIR),
        Operation.CODE_SEED: (ClaimType.METHOD_HYPOTHESIS,),
        Operation.DISTILL_DIAGNOSTIC: (ClaimType.AUDIT_FINDING, ClaimType.DEBUG_REPAIR),
        Operation.DISTILL_CANDIDATE: (ClaimType.METHOD_HYPOTHESIS,),
        Operation.DISTILL_POSITIVE_RESULT: (
            ClaimType.SCORE,
            ClaimType.PAIRWISE_SUPERIORITY,
            ClaimType.GENERALIZATION,
        ),
        Operation.DISTILL_POSITIVE_ADOPTED: (
            ClaimType.EXPERIENCE_ADOPTION,
            ClaimType.CAUSAL_ATTRIBUTION,
        ),
        Operation.DISTILL_POSITIVE: (ClaimType.METHOD_HYPOTHESIS, ClaimType.SCORE),
        Operation.RANK: (ClaimType.SCORE,),
        Operation.SELECT: (ClaimType.SCORE,),
        Operation.PROMOTE_RESULT: (ClaimType.SCORE, ClaimType.EXECUTED),
        Operation.PUBLISH_ADOPTION: (ClaimType.EXPERIENCE_ADOPTION,),
        Operation.PUBLISH_CAUSAL: (ClaimType.CAUSAL_ATTRIBUTION,),
        Operation.PROMOTE: (ClaimType.SCORE,),
    }.get(operation, (ClaimType.SCORE, ClaimType.METHOD_HYPOTHESIS))
    for claim_type in preference:
        matches = result.claims_of_type(claim_type)
        if matches:
            return matches[0]
    raise ValueError(f"No claim in artifact {result.artifact_id} supports {operation.value}")
