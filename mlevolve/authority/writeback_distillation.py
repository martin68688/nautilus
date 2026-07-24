from __future__ import annotations

import copy
import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .authority_engine import AuthorityEngine
from .bundle_authority import restore_engine_snapshot
from .collectors import DerivationCollector, TrustedCollectorHost
from .domain_scope import canonical_domain
from .memory_snapshot import sha256_file, sha256_json
from .models import ClaimType, ProtocolRef, Receipt, TaskContext
from .positive_distillation import (
    PositiveDistillationKind,
    authorize_positive_distillation,
)
from .protocol_registry import ProtocolRegistry


WRITEBACK_PLAN_SCHEMA = "positive_writeback_distillation_plan_v1"
WRITEBACK_MATERIAL_SCHEMA = "positive_writeback_materialization_v1"
_CAUSAL_LANGUAGE = re.compile(
    r"\b(?:caus(?:e|ed|al)|led\s+to|due\s+to|because\s+of|resulted\s+from)\b",
    re.IGNORECASE,
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _snapshot(pointer: Mapping[str, Any]) -> dict[str, Any]:
    if pointer.get("schema") != "authority_snapshot_pointer_v1":
        raise ValueError("Unsupported Authority snapshot pointer")
    path = Path(str(pointer.get("path") or "")).resolve()
    digest = str(pointer.get("sha256") or "")
    if not path.is_file() or len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError("Authority snapshot pointer is not hash-resolvable")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Authority snapshot must be an object")
    return payload


def _source_claim_ref(
    event: Mapping[str, Any],
    *,
    edge: bool,
    edge_kind: str = "adoption",
) -> str:
    payload = event.get("payload") or {}
    decision_refs = [
        str(value) for value in payload.get("authority_decision_refs") or []
    ]
    if len(decision_refs) != 1:
        raise ValueError("Writeback event must bind exactly one decision")
    snapshot = _snapshot(payload.get("authority_snapshot_pointer") or {})
    decision = (snapshot.get("decisions") or {}).get(decision_refs[0])
    if not isinstance(decision, Mapping):
        raise ValueError("Writeback event decision is absent from its snapshot")
    claim_ref = str(decision.get("claim_id") or "")
    if edge and claim_ref != str(payload.get("edge_claim_ref") or ""):
        raise ValueError("Edge decision/Claim binding mismatch")
    claim = (snapshot.get("claims") or {}).get(claim_ref)
    if not isinstance(claim, Mapping):
        raise ValueError("Writeback source Claim is absent from its snapshot")
    allowed_types = (
        {
            ClaimType.CAUSAL_ATTRIBUTION.value
            if edge_kind == "causal"
            else ClaimType.EXPERIENCE_ADOPTION.value
        }
        if edge
        else {
            ClaimType.SCORE.value,
            ClaimType.PAIRWISE_SUPERIORITY.value,
            ClaimType.GENERALIZATION.value,
        }
    )
    if str(claim.get("claim_type") or "") not in allowed_types:
        raise ValueError("Writeback decision does not select a distillable Claim")
    return claim_ref


def build_positive_writeback_plan(
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Account for every Result/Adoption/Causal event without inferring edges."""

    if inventory.get("schema") != "writeback_event_inventory_v1":
        raise ValueError("Unsupported writeback inventory")
    results = list(inventory.get("result_facts") or [])
    adoptions = list(inventory.get("adoption_edges") or [])
    causals = list(inventory.get("causal_edges") or [])
    result_by_target: dict[tuple[str, str], dict[str, Any]] = {}
    quarantined: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    consumed_event_ids: set[str] = set()

    for event in results:
        payload = event.get("payload") or {}
        key = (
            str(payload.get("artifact_id") or ""),
            str(payload.get("protocol_ref") or ""),
        )
        if key in result_by_target:
            raise ValueError("Ambiguous duplicate Result Facts for one target")
        result_by_target[key] = event
        consumed_event_ids.add(str(event["event_id"]))
        source_claim_ref = _source_claim_ref(event, edge=False)
        identity = {
            "kind": PositiveDistillationKind.RESULT.value,
            "result_event_id": event["event_id"],
            "artifact_id": key[0],
            "protocol_ref": key[1],
            "source_claim_ref": source_claim_ref,
        }
        candidate_id = (
            "positive_writeback::result::" + sha256_json(identity)[:24]
        )
        items.append(
            {
                "candidate_id": candidate_id,
                **identity,
                "task_id": str(payload.get("task_id") or ""),
                "run_id": str(payload.get("run_id") or ""),
                "source_event_ids": [str(event["event_id"])],
                "authority_decision_ref": str(
                    (payload.get("authority_decision_refs") or [""])[0]
                ),
                "authority_snapshot_pointer": copy.deepcopy(
                    payload.get("authority_snapshot_pointer") or {}
                ),
                "status": "eligible",
            }
        )

    causal_by_edge: dict[tuple[str, str, str], dict[str, Any]] = {}
    for event in causals:
        payload = event.get("payload") or {}
        key = (
            str(payload.get("target_artifact_id") or ""),
            str(payload.get("protocol_ref") or ""),
            str(payload.get("contract_hash") or ""),
        )
        if key in causal_by_edge:
            raise ValueError("Duplicate Causal Edge for one contract/target")
        causal_by_edge[key] = event
        consumed_event_ids.add(str(event["event_id"]))

    for event in adoptions:
        payload = event.get("payload") or {}
        target_key = (
            str(payload.get("target_artifact_id") or ""),
            str(payload.get("protocol_ref") or ""),
        )
        edge_key = (*target_key, str(payload.get("contract_hash") or ""))
        consumed_event_ids.add(str(event["event_id"]))
        result_event = result_by_target.get(target_key)
        causal_event = causal_by_edge.get(edge_key)
        source_events = [str(event["event_id"])]
        if causal_event is not None:
            source_events.append(str(causal_event["event_id"]))
        if result_event is None:
            quarantined.append(
                {
                    "event_id": str(event["event_id"]),
                    "kind": PositiveDistillationKind.ADOPTED.value,
                    "reason": "matching_result_fact_missing",
                    "source_event_ids": source_events,
                }
            )
            continue
        source_claim_ref = _source_claim_ref(event, edge=True)
        causal_claim_ref = ""
        causal_decision_ref = ""
        causal_snapshot_pointer: dict[str, Any] = {}
        if causal_event is not None:
            causal_claim_ref = _source_claim_ref(
                causal_event,
                edge=True,
                edge_kind="causal",
            )
            causal_payload = causal_event.get("payload") or {}
            causal_decision_ref = str(
                (causal_payload.get("authority_decision_refs") or [""])[0]
            )
            causal_snapshot_pointer = copy.deepcopy(
                causal_payload.get("authority_snapshot_pointer") or {}
            )
        identity = {
            "kind": PositiveDistillationKind.ADOPTED.value,
            "result_event_id": str(result_event["event_id"]),
            "adoption_event_id": str(event["event_id"]),
            "causal_event_id": (
                str(causal_event["event_id"]) if causal_event else ""
            ),
            "artifact_id": target_key[0],
            "protocol_ref": target_key[1],
            "contract_hash": edge_key[2],
                "source_claim_ref": source_claim_ref,
                "causal_claim_ref": causal_claim_ref,
                "causal_decision_ref": causal_decision_ref,
                "causal_snapshot_pointer": causal_snapshot_pointer,
            }
        candidate_id = (
            "positive_writeback::adopted::" + sha256_json(identity)[:24]
        )
        result_payload = result_event.get("payload") or {}
        items.append(
            {
                "candidate_id": candidate_id,
                **identity,
                "task_id": str(result_payload.get("task_id") or ""),
                "run_id": str(result_payload.get("run_id") or ""),
                "source_event_ids": source_events,
                "authority_decision_ref": str(
                    (payload.get("authority_decision_refs") or [""])[0]
                ),
                "authority_snapshot_pointer": copy.deepcopy(
                    payload.get("authority_snapshot_pointer") or {}
                ),
                "status": "eligible",
            }
        )

    adoption_keys = {
        (
            str((event.get("payload") or {}).get("target_artifact_id") or ""),
            str((event.get("payload") or {}).get("protocol_ref") or ""),
            str((event.get("payload") or {}).get("contract_hash") or ""),
        )
        for event in adoptions
    }
    for key, event in causal_by_edge.items():
        if key not in adoption_keys:
            quarantined.append(
                {
                    "event_id": str(event["event_id"]),
                    "kind": "causal",
                    "reason": "matching_adoption_edge_missing",
                    "source_event_ids": [str(event["event_id"])],
                }
            )

    expected_event_ids = {
        str(event["event_id"])
        for event in [*results, *adoptions, *causals]
    }
    if consumed_event_ids != expected_event_ids:
        raise ValueError("Writeback plan did not account for every typed event")
    items.sort(key=lambda row: row["candidate_id"])
    quarantined.sort(key=lambda row: row["event_id"])
    report = {
        "schema": WRITEBACK_PLAN_SCHEMA,
        "inventory_hash": str(inventory.get("inventory_hash") or ""),
        "result_fact_count": len(results),
        "adoption_edge_count": len(adoptions),
        "causal_edge_count": len(causals),
        "positive_result_candidate_count": sum(
            row["kind"] == PositiveDistillationKind.RESULT.value
            for row in items
        ),
        "positive_adopted_candidate_count": sum(
            row["kind"] == PositiveDistillationKind.ADOPTED.value
            for row in items
        ),
        "items": items,
        "quarantined": quarantined,
        "consumed_event_ids": sorted(consumed_event_ids),
        "plan_hash": "",
    }
    report["plan_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "plan_hash"}
    )
    return report


@dataclass(frozen=True)
class PositiveWritebackMaterialization:
    plan: dict[str, Any]
    clauses: tuple[dict[str, Any], ...]
    containers: tuple[dict[str, Any], ...]
    authority_snapshot: dict[str, Any]
    derivations: tuple[dict[str, Any], ...]
    report: dict[str, Any]


def _source_receipts(
    engine: AuthorityEngine,
    *,
    claim_ref: str,
    decision_ref: str,
) -> list[Receipt]:
    decision = engine.decisions.get(decision_ref)
    if decision is None or decision.claim_id != claim_ref or not decision.allowed:
        raise ValueError("Distillation source decision is unavailable or denied")
    path_ids = sorted(
        path_id
        for path_id in decision.satisfied_paths
        if path_id in engine.graph.paths
        and engine.graph.paths[path_id].claim_id == claim_ref
    )
    if not path_ids:
        raise ValueError("Distillation source decision has no complete path")
    path = engine.graph.paths[path_ids[0]]
    receipts = [
        engine.graph.receipts[receipt_id]
        for receipt_id in path.receipt_ids
        if receipt_id in engine.graph.receipts
    ]
    if len(receipts) != len(path.receipt_ids):
        raise ValueError("Distillation source path has missing Receipts")
    return receipts


def materialize_positive_writeback(
    plan: Mapping[str, Any],
    proposals: Mapping[str, Mapping[str, Any]],
    *,
    registry: ProtocolRegistry,
    policy_version: str,
    collector_version: str = "1",
) -> PositiveWritebackMaterialization:
    """Authorize typed positive clauses from one complete writeback plan."""

    if plan.get("schema") != WRITEBACK_PLAN_SCHEMA:
        raise ValueError("Unsupported positive writeback plan")
    expected_hash = sha256_json(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    if plan.get("plan_hash") != expected_hash:
        raise ValueError("Positive writeback plan hash mismatch")
    engine = AuthorityEngine(registry, policy_version=str(policy_version))
    host = TrustedCollectorHost(
        f"sleep-time-positive:{expected_hash[:24]}",
        collector_version=str(collector_version),
    )
    clauses: list[dict[str, Any]] = []
    containers: list[dict[str, Any]] = []
    derivations: list[dict[str, Any]] = []
    dispositions: list[dict[str, Any]] = []
    for item in plan.get("items") or []:
        candidate_id = str(item["candidate_id"])
        kind = PositiveDistillationKind(str(item["kind"]))
        proposal = proposals.get(candidate_id)
        if not isinstance(proposal, Mapping):
            raise ValueError(
                f"Missing explicit positive distillation proposal: {candidate_id}"
            )
        text = str(proposal.get("text") or "").strip()
        if not text:
            raise ValueError(f"Positive proposal has no text: {candidate_id}")
        assertion_level = str(proposal.get("assertion_level") or "adoption").strip().lower()
        if assertion_level not in {"adoption", "causal"}:
            raise ValueError(
                f"Unsupported positive assertion level: {assertion_level}"
            )
        if kind == PositiveDistillationKind.RESULT:
            if assertion_level != "adoption" or _CAUSAL_LANGUAGE.search(text):
                raise ValueError(
                    "Positive Result text cannot claim historical adoption/causality"
                )
        elif assertion_level == "causal":
            if not item.get("causal_claim_ref"):
                raise ValueError(
                    "Causal Positive Adopted text requires a published Causal Edge"
                )
        elif _CAUSAL_LANGUAGE.search(text):
            raise ValueError(
                "Causal wording requires assertion_level=causal and a Causal Edge"
            )
        source_snapshot = _snapshot(
            (
                item.get("causal_snapshot_pointer")
                if assertion_level == "causal"
                else item.get("authority_snapshot_pointer")
            )
        )
        restore_engine_snapshot(engine, source_snapshot)
        claim_ref = str(
            item.get("causal_claim_ref")
            if assertion_level == "causal"
            else item["source_claim_ref"]
        )
        claim = engine.graph.claims.get(claim_ref)
        if claim is None:
            raise ValueError("Positive distillation source Claim is unavailable")
        protocol_ref: ProtocolRef = claim.protocol_ref
        if protocol_ref.key() != str(item.get("protocol_ref") or ""):
            raise ValueError("Positive proposal ProtocolRef mismatch")
        receipts = _source_receipts(
            engine,
            claim_ref=claim_ref,
            decision_ref=str(
                item.get("causal_decision_ref")
                if assertion_level == "causal"
                else item["authority_decision_ref"]
            ),
        )
        mapping_hash = sha256_json(
            {
                "candidate_id": candidate_id,
                "source_claim_ref": claim_ref,
                "assertion_level": assertion_level,
                "source_event_ids": item.get("source_event_ids") or [],
                "text": text,
                "scope_widened": False,
            }
        )
        derivation_receipt = host.collect(
            DerivationCollector,
            artifact_id=str(item["artifact_id"]),
            run_id=str(item.get("run_id") or "sleep-time"),
            protocol_ref=protocol_ref,
            source="host.sleep_time_positive_binder",
            payload={
                "parent_claim_refs": [claim_ref],
                "mapping_hash": mapping_hash,
                "scope_widened": False,
            },
        )
        sop_id = (
            f"sop::positive-{kind.value}::"
            f"{sha256_json({'candidate_id': candidate_id, 'text': text})[:24]}"
        )
        source_family = str(
            proposal.get("source_task_family") or ""
        ).strip()
        source_domain = str(proposal.get("source_domain") or "").strip()
        if not source_domain and source_family:
            source_domain = canonical_domain(source_family)
        result = authorize_positive_distillation(
            engine,
            kind=kind,
            claim=claim,
            receipts=[*receipts, derivation_receipt],
            active_protocol=protocol_ref,
            task_context=TaskContext(
                task_id=str(item.get("task_id") or ""),
                task_family=source_family,
            ),
            text=text,
            sop_id=sop_id,
            source_run_ids=[str(item.get("run_id") or "")],
            source_task_ids=[str(item.get("task_id") or "")],
            source_task_families=[source_family] if source_family else [],
            source_domains=[source_domain] if source_domain else [],
        )
        if not result.decision.allowed or result.clause is None:
            raise ValueError(
                "Positive distillation Authority denied: "
                + ",".join(result.decision.missing_obligations)
            )
        clause = dataclasses.asdict(result.clause)
        clause["publication_origin"] = "typed_writeback_distillation_v1"
        clause["writeback_candidate_id"] = candidate_id
        clause["source_event_ids"] = list(item.get("source_event_ids") or [])
        clauses.append(_jsonable(clause))
        containers.append(
            {
                "sop_id": result.clause.sop_id,
                "title": str(proposal.get("title") or text),
                "task_id": str(item.get("task_id") or ""),
                "clause_ids": [result.clause.clause_id],
                "source_run_ids": list(result.clause.source_run_ids),
                "source_task_ids": list(result.clause.source_task_ids),
                "source_task_families": list(
                    result.clause.source_task_families
                ),
                "source_domains": list(result.clause.source_domains),
                "transfer_scopes": [result.clause.transfer_scope],
                "domain_scope_complete": bool(
                    result.clause.source_domains
                    and result.clause.source_task_ids
                ),
                "publication_origin": "typed_writeback_distillation_v1",
            }
        )
        derivations.append(
            {
                "derivation_id": mapping_hash,
                "candidate_id": candidate_id,
                "kind": kind.value,
                "source_event_ids": list(item.get("source_event_ids") or []),
                "source_claim_ref": claim_ref,
                "derived_claim_ref": result.derived_claim.claim_id,
                "clause_id": result.clause.clause_id,
                "receipt_id": derivation_receipt.receipt_id,
                "decision_id": result.decision.decision_id,
                "scope_widened": False,
            }
        )
        dispositions.append(
            {
                "candidate_id": candidate_id,
                "kind": kind.value,
                "status": "published",
                "clause_id": result.clause.clause_id,
                "derived_claim_id": result.derived_claim.claim_id,
                "decision_id": result.decision.decision_id,
            }
        )

    snapshot = engine.snapshot()
    clauses.sort(key=lambda row: row["clause_id"])
    containers.sort(key=lambda row: row["sop_id"])
    derivations.sort(key=lambda row: row["derivation_id"])
    dispositions.sort(key=lambda row: row["candidate_id"])
    report = {
        "schema": WRITEBACK_MATERIAL_SCHEMA,
        "status": "passed",
        "plan_hash": expected_hash,
        "candidate_count": len(plan.get("items") or []),
        "published_count": len(clauses),
        "positive_result_count": sum(
            row["kind"] == PositiveDistillationKind.RESULT.value
            for row in dispositions
        ),
        "positive_adopted_count": sum(
            row["kind"] == PositiveDistillationKind.ADOPTED.value
            for row in dispositions
        ),
        "quarantined_input_count": len(plan.get("quarantined") or []),
        "clause_ids": [row["clause_id"] for row in clauses],
        "derived_claim_ids": [
            row["derived_claim_id"] for row in dispositions
        ],
        "source_event_ids": list(plan.get("consumed_event_ids") or []),
        "dispositions": dispositions,
        "materialization_hash": "",
    }
    report["materialization_hash"] = sha256_json(
        {
            "plan_hash": expected_hash,
            "clauses": clauses,
            "containers": containers,
            "authority_snapshot": snapshot,
            "derivations": derivations,
            "report": {
                key: value
                for key, value in report.items()
                if key != "materialization_hash"
            },
        }
    )
    return PositiveWritebackMaterialization(
        plan=copy.deepcopy(dict(plan)),
        clauses=tuple(clauses),
        containers=tuple(containers),
        authority_snapshot=copy.deepcopy(snapshot),
        derivations=tuple(derivations),
        report=report,
    )


__all__ = [
    "PositiveWritebackMaterialization",
    "WRITEBACK_MATERIAL_SCHEMA",
    "WRITEBACK_PLAN_SCHEMA",
    "build_positive_writeback_plan",
    "materialize_positive_writeback",
]
