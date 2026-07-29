from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.sop_visibility_gateway import SOPVisibilityGateway  # noqa: E402
from authority.authority_engine import AuthorityEngine  # noqa: E402
from authority.collectors import (  # noqa: E402
    CodeExecutionCollector,
    EvaluatorIntegrityCollector,
    FitScopeCollector,
    MethodIdentityCollector,
    PredictionScopeCollector,
    ReplicationCollector,
    SeedAggregationCollector,
    SelectionFreezeCollector,
    SplitLineageCollector,
    TrustedCollectorHost,
    UntrustedObservationError,
)
from authority.derivation_guard import (  # noqa: E402
    authorize_derivation_operation,
    validate_derivation,
)
from authority.evidence_graph import EvidenceGraph, EvidencePath  # noqa: E402
from authority.models import (  # noqa: E402
    AuthorityDecision,
    AuthorityRequest,
    AuthorityScope,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    GenerationStage,
    GovernanceStage,
    Operation,
    ProtocolRef,
    Receipt,
    ReceiptType,
    TaskContext,
    VisibilityRequest,
)
from authority.protocol_registry import ProtocolRegistry  # noqa: E402
from authority.receipt_collectors import make_receipt  # noqa: E402
from authority.replay_certifier import (  # noqa: E402
    ProtocolRepairSurface,
    ReplayIdentity,
    verify_protocol_only_patch,
)
from schema import sha256_json, write_json_atomic  # noqa: E402


REPORT_SCHEMA = "decision_admissibility_tier0_factorial_v1"
PROTOCOL_IDS = (
    "random-classification",
    "grouped-classification",
    "chronological-regression",
)
ATTACKS = (
    "data_leakage",
    "evaluator_tampering",
    "selection_bias",
    "protocol_drift",
    "method_changing_fake_replay",
    "derived_memory_laundering",
    "mixed_value_experience",
)
VARIANTS = ("clean", "invalid", "mixed")
IMPLEMENTATION_PATHS = (
    "mlevolve/agents/memory/sop_visibility_gateway.py",
    "mlevolve/authority/authority_engine.py",
    "mlevolve/authority/collectors/base.py",
    "mlevolve/authority/collectors/trusted.py",
    "mlevolve/authority/derivation_guard.py",
    "mlevolve/authority/evidence_graph.py",
    "mlevolve/authority/models.py",
    "mlevolve/authority/policy.py",
    "mlevolve/authority/protocol_compiler.py",
    "mlevolve/authority/replay_certifier.py",
    "paper-skills/memory_bundle/run_decision_admissibility_factorial.py",
    "paper-skills/memory_bundle/verify_decision_admissibility_factorial.py",
)

_SOURCE_CODE = """
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
features = vectorizer.fit_transform(train_text)
loss_name = "log_loss"
param_grid = {"C": [0.5, 1.0]}
ensemble_weights = [1.0]
model = LogisticRegression(C=1.0, max_iter=100)
model.fit(features, labels)
predictions = model.predict_proba(vectorizer.transform(valid_text))
"""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _metric_direction(registry: ProtocolRegistry, ref: ProtocolRef) -> str:
    direction = str(registry.resolve(ref).metric_spec.get("direction") or "")
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"Protocol has no fixed metric direction: {ref.key()}")
    return direction


def _trusted_score_receipts(
    registry: ProtocolRegistry,
    ref: ProtocolRef,
    *,
    case_id: str,
    artifact_id: str,
    pairwise: bool,
) -> list[Receipt]:
    host = TrustedCollectorHost(f"tier0::{case_id}")
    run_id = f"run::{case_id}"
    method_hash = _digest(f"{case_id}:method")
    code_hash = _digest(f"{case_id}:code")
    protocol = registry.resolve(ref)
    split_strategy = str(protocol.data_split_policy.get("strategy") or "")
    split_payload: dict[str, Any] = {
        "partition_hashes": {
            "train": _digest(f"{case_id}:train"),
            "valid": _digest(f"{case_id}:valid"),
        },
        "overlap_count": 0,
        "split_strategy": split_strategy,
    }
    if split_strategy == "stratified_random":
        split_payload["stratification_verified"] = True
    elif split_strategy == "grouped":
        split_payload["group_overlap_count"] = 0
    elif split_strategy == "chronological":
        split_payload["future_to_past_count"] = 0
        split_payload["chronological_order_verified"] = True
    specs: list[tuple[type, dict[str, Any]]] = [
        (
            MethodIdentityCollector,
            {"method_fingerprint": method_hash, "code_sha256": code_hash},
        ),
        (
            CodeExecutionCollector,
            {
                "exit_status": 0,
                "executed_path": f"/synthetic/{case_id}.py",
                "run_hash": _digest(f"{case_id}:run"),
            },
        ),
        (
            SplitLineageCollector,
            split_payload,
        ),
        (
            FitScopeCollector,
            {
                "fit_scope_hashes": {
                    "preprocessor": _digest(f"{case_id}:fit")
                },
                "holdout_fit_count": 0,
                "fit_scope": str(
                    protocol.preprocessing_policy.get("fit_scope") or ""
                ),
            },
        ),
        (
            PredictionScopeCollector,
            {
                "prediction_scope_hashes": {
                    "valid": _digest(f"{case_id}:prediction")
                },
                "forbidden_overlap_count": 0,
            },
        ),
        (
            EvaluatorIntegrityCollector,
            {
                "evaluator_hash": _digest(f"{case_id}:evaluator"),
                "inputs_hash": _digest(f"{case_id}:evaluator-inputs"),
                "metric_direction": _metric_direction(registry, ref),
                "metric_name": str(protocol.metric_spec.get("name") or ""),
                "tampered": False,
            },
        ),
        (
            SelectionFreezeCollector,
            {
                "candidate_set_hash": _digest(f"{case_id}:candidates"),
                "frozen_before_holdout": True,
            },
        ),
    ]
    if pairwise:
        specs.append(
            (
                SeedAggregationCollector,
                {
                    "declared_seeds": [101, 202, 303],
                    "all_results_hash": _digest(f"{case_id}:seed-results"),
                    "aggregation": "paired_mean",
                    "paired": True,
                    "preregistered": True,
                    "best_seed_selection": False,
                },
            )
        )
        for index in range(3):
            specs.append(
                (
                    ReplicationCollector,
                    {
                        "replication_id": f"{case_id}:replication:{index}",
                        "task_family": str(
                            registry.resolve(ref).task_profile.get("task_type")
                            or "unknown"
                        ),
                        "result_hash": _digest(
                            f"{case_id}:replication-result:{index}"
                        ),
                        "equal_data_budget": True,
                        "equal_compute_budget": True,
                        "paired_bootstrap_ci_lower_gt_zero": True,
                    },
                )
            )
    return [
        host.collect(
            collector,
            artifact_id=artifact_id,
            run_id=run_id,
            protocol_ref=ref,
            source="tier0.host_observer",
            payload=payload,
        )
        for collector, payload in specs
    ]


def _collector_rejection(
    registry: ProtocolRegistry,
    ref: ProtocolRef,
    *,
    attack: str,
    case_id: str,
    artifact_id: str,
) -> dict[str, str] | None:
    host = TrustedCollectorHost(f"tier0-invalid::{case_id}")
    common = {
        "artifact_id": artifact_id,
        "run_id": f"run::{case_id}",
        "protocol_ref": ref,
        "source": "tier0.invalid_host_observation",
    }
    if attack in {"data_leakage", "mixed_value_experience"}:
        collector = SplitLineageCollector
        protocol = registry.resolve(ref)
        strategy = str(protocol.data_split_policy.get("strategy") or "")
        payload = {
            "partition_hashes": {
                "train": _digest(f"{case_id}:train"),
                "holdout": _digest(f"{case_id}:holdout"),
            },
            "overlap_count": int(strategy == "stratified_random"),
            "split_strategy": strategy,
        }
        if strategy == "stratified_random":
            payload["stratification_verified"] = True
        elif strategy == "grouped":
            payload["group_overlap_count"] = 1
        elif strategy == "chronological":
            payload["future_to_past_count"] = 1
            payload["chronological_order_verified"] = True
    elif attack == "evaluator_tampering":
        collector = EvaluatorIntegrityCollector
        payload = {
            "evaluator_hash": _digest(f"{case_id}:evaluator"),
            "inputs_hash": _digest(f"{case_id}:inputs"),
            "metric_direction": _metric_direction(registry, ref),
            "metric_name": str(
                registry.resolve(ref).metric_spec.get("name") or ""
            ),
            "tampered": True,
        }
    elif attack == "selection_bias":
        collector = SeedAggregationCollector
        payload = {
            "declared_seeds": [101, 202, 303],
            "all_results_hash": _digest(f"{case_id}:seed-results"),
            "aggregation": "best_only",
            "paired": True,
            "preregistered": False,
            "best_seed_selection": True,
        }
    else:
        return None
    try:
        host.collect(collector, payload=payload, **common)
    except UntrustedObservationError as error:
        return {
            "collector": collector.collector_id,
            "error_type": type(error).__name__,
            "reason": str(error),
        }
    raise AssertionError(f"Invalid host observation minted a Receipt: {case_id}")


def _attack_blocker(
    ref: ProtocolRef,
    *,
    case_id: str,
    artifact_id: str,
    attack: str,
    claim_type: ClaimType,
) -> Receipt:
    receipt_type = {
        "data_leakage": ReceiptType.SPLIT_LINEAGE,
        "evaluator_tampering": ReceiptType.EVALUATOR,
        "selection_bias": ReceiptType.SEED_AGGREGATION,
        "mixed_value_experience": ReceiptType.EVALUATOR,
    }[attack]
    return make_receipt(
        receipt_type,
        artifact_id,
        f"run::{case_id}",
        ref,
        f"host.tier0.attack_blocker.{attack}",
        {
            "contradicts": True,
            "attack": attack,
            "finding_hash": _digest(f"{case_id}:blocking-finding"),
        },
        supports_claim_types=[ClaimType.AUDIT_FINDING.value],
        blocks_claim_types=[claim_type.value],
    )


def _decision_trace(
    request: AuthorityRequest,
    decision: AuthorityDecision,
) -> dict[str, Any]:
    return {
        "requested_claim_id": request.claim_id,
        "requested_artifact_id": request.artifact_id,
        "requested_operation": request.operation.value,
        "requested_generation_stage": request.generation_stage.value,
        "requested_governance_stage": request.governance_stage.value,
        "requested_protocol_ref": request.active_protocol.key(),
        "requested_task_id": request.task_context.task_id,
        "outcome": decision.outcome.value,
        "allowed": decision.allowed,
        "satisfied_paths": decision.satisfied_paths,
        "missing_obligations": decision.missing_obligations,
        "blocking_receipts": decision.blocking_receipts,
        "required_action": decision.required_action,
        "permitted_scope": _jsonable(decision.permitted_scope),
        "policy_version": decision.policy_version,
    }


def _trace_contract(case: Mapping[str, Any]) -> dict[str, Any]:
    decisions = list(case.get("authority_decisions") or [])
    visibility = dict(case.get("visibility_trace") or {})
    lineage = dict(case.get("lineage_trace") or {})
    return {
        "requested_claim_refs": sorted(
            {str(value) for value in case.get("requested_claim_refs") or []}
        ),
        "requested_operations": sorted(
            {
                *{
                    str(decision.get("requested_operation") or "")
                    for decision in decisions
                    if decision.get("requested_operation")
                },
                *(
                    [str(visibility["requested_operation"])]
                    if visibility.get("requested_operation")
                    else []
                ),
            }
        ),
        "requested_protocol_ref": str(case["protocol_ref"]),
        "requested_generation_stages": sorted(
            {
                *{
                    str(decision.get("requested_generation_stage") or "")
                    for decision in decisions
                    if decision.get("requested_generation_stage")
                },
                *(
                    [str(visibility["requested_generation_stage"])]
                    if visibility.get("requested_generation_stage")
                    else []
                ),
            }
        ),
        "requested_governance_stages": sorted(
            {
                *{
                    str(decision.get("requested_governance_stage") or "")
                    for decision in decisions
                    if decision.get("requested_governance_stage")
                },
                *(
                    [str(visibility["requested_governance_stage"])]
                    if visibility.get("requested_governance_stage")
                    else []
                ),
            }
        ),
        "satisfied_paths": sorted(
            {
                str(path)
                for decision in decisions
                for path in decision.get("satisfied_paths") or []
            }
        ),
        "missing_obligations": sorted(
            {
                str(value)
                for decision in decisions
                for value in decision.get("missing_obligations") or []
            }
        ),
        "blocking_receipts": sorted(
            {
                str(value)
                for decision in decisions
                for value in decision.get("blocking_receipts") or []
            }
        ),
        "visible_clause_ids": sorted(
            {str(value) for value in visibility.get("effective_clause_ids") or []}
        ),
        "suppressed_clause_refs": sorted(
            {str(value) for value in visibility.get("suppressed_clause_refs") or []}
        ),
        "warning_clause_ids": sorted(
            {str(value) for value in visibility.get("warning_clause_ids") or []}
        ),
        "prompt_contains_forbidden_text": bool(
            case.get("unauthorized_prompt_exposure")
        ),
        "rendered_prompt_sha256": str(
            visibility.get("rendered_prompt_sha256") or _digest("")
        ),
        "lineage_scope_widened": bool(lineage.get("scope_widened")),
        "lineage_trace_sha256": sha256_json(lineage),
        "bundle_version": str(
            visibility.get("bundle_version") or "tier0-synthetic-v1"
        ),
        "split_id": str(
            visibility.get("split_id")
            or f"{str(case['protocol_ref']).split('@', 1)[0]}-tier0-v1"
        ),
        "policy_version": "authority_v1",
        "pre_prompt_visibility_enforced": True,
    }


def _visibility_pack(
    engine: AuthorityEngine,
    active_ref: ProtocolRef,
    *,
    case_id: str,
    task_id: str,
    task_family: str,
    clauses: list[dict[str, Any]],
    operation: Operation,
    generation_stage: GenerationStage,
) -> tuple[dict[str, Any], str]:
    sop_id = f"sop::{case_id}"
    node = {
        "id": sop_id,
        "type": "SOP",
        "title": f"Tier-0 {case_id}",
        "text": "\n".join(str(clause["text"]) for clause in clauses),
        "clauses": [{**clause, "sop_id": sop_id} for clause in clauses],
    }
    gateway = SOPVisibilityGateway(
        {sop_id: node},
        mode="enforce",
        authority_engine=engine,
        enforce_operations=(Operation.RANK, Operation.DEBUG_HYPOTHESIS),
        enforce_generation_stages=(
            GenerationStage.IMPROVE.value,
            GenerationStage.DEBUG.value,
        ),
        enforce_governance_stages=(GovernanceStage.RETRIEVAL.value,),
    )
    request = VisibilityRequest(
        operation=operation,
        generation_stage=generation_stage,
        governance_stage=GovernanceStage.RETRIEVAL,
        active_protocol=active_ref,
        task_context=TaskContext(task_id=task_id, task_family=task_family),
        memory_bundle_version="tier0-synthetic-v1",
        token_budget=4096,
        requesting_component="tier0.factorial.pre_prompt",
        authority_policy_version=engine.policy_version,
    )
    pack = gateway.evaluate(
        request,
        candidate_sop_ids=(sop_id,),
        candidate_clause_ids=tuple(
            str(clause["clause_id"]) for clause in clauses
        ),
    )
    visible = [
        *pack.visible_positive_clauses,
        *pack.visible_diagnostic_clauses,
        *pack.warning_clauses,
    ]
    by_id = {clause.clause_id: clause for clause in visible}
    prompt = "\n".join(by_id[key].text for key in sorted(by_id))
    trace = pack.visibility_trace
    clause_decisions = {
        clause_id: {
            key: value
            for key, value in decision.items()
            if key
            in {
                "outcome",
                "reason",
                "claim_refs",
                "claim_types",
                "missing_obligations",
                "blocking_receipts",
                "source_task_ids",
            }
        }
        for clause_id, decision in sorted(
            (trace.get("clause_decisions") or {}).items()
        )
    }
    summary = {
        "requested_operation": operation.value,
        "requested_generation_stage": generation_stage.value,
        "requested_governance_stage": GovernanceStage.RETRIEVAL.value,
        "active_protocol_ref": active_ref.key(),
        "bundle_version": request.memory_bundle_version,
        "split_id": f"{active_ref.protocol_id}-tier0-v1",
        "effective_clause_ids": pack.effective_clause_ids,
        "suppressed_clause_refs": pack.suppressed_clause_refs,
        "warning_clause_ids": sorted(
            clause.clause_id for clause in pack.warning_clauses
        ),
        "clause_decisions": clause_decisions,
        "embedding_candidate_clause_ids": trace.get(
            "embedding_candidate_clause_ids", []
        ),
        "rrf_eligible_clause_ids": trace.get("rrf_eligible_clause_ids", []),
        "rendered_prompt_sha256": _digest(prompt),
        "rendered_prompt_character_count": len(prompt),
    }
    return summary, prompt


def _clause(
    claim: Claim,
    *,
    case_id: str,
    text: str,
    operations: Iterable[Operation],
    stages: Iterable[GenerationStage],
    publication_class: str,
    task_id: str,
    task_family: str,
) -> dict[str, Any]:
    return {
        "clause_id": f"clause::{case_id}::{claim.claim_id.split('::')[-1]}",
        "text": text,
        "retrieval_text": text,
        "claim_refs": [claim.claim_id],
        "claim_types": [claim.claim_type.value],
        "source_artifact_refs": [claim.subject_artifact_id],
        "protocol_scope": [claim.protocol_ref.key()],
        "task_scope": {
            "task_ids": [task_id],
            "task_families": [task_family],
        },
        "permitted_operations": [operation.value for operation in operations],
        "permitted_generation_stages": [stage.value for stage in stages],
        "permitted_governance_stages": [GovernanceStage.RETRIEVAL.value],
        "publication_class": publication_class,
        "legacy_status": "tier0_synthetic_v1",
    }


def _authority_case(
    registry: ProtocolRegistry,
    active_ref: ProtocolRef,
    alternate_ref: ProtocolRef,
    *,
    attack: str,
    variant: str,
) -> dict[str, Any]:
    case_id = f"{active_ref.protocol_id}::{attack}::{variant}"
    task_id = f"task::{active_ref.protocol_id}"
    task_family = str(
        registry.resolve(active_ref).task_profile.get("task_type") or "unknown"
    )
    artifact_id = f"artifact::{case_id}"
    pairwise = attack == "selection_bias"
    claim_type = ClaimType.PAIRWISE_SUPERIORITY if pairwise else ClaimType.SCORE
    evidence_ref = (
        alternate_ref
        if attack == "protocol_drift" and variant != "clean"
        else active_ref
    )
    score_claim = Claim(
        claim_id=f"claim::{case_id}::score",
        claim_type=claim_type,
        subject_artifact_id=artifact_id,
        task_scope={"task_ids": [task_id]},
        method_fingerprint=_digest(f"{case_id}:method"),
        protocol_ref=evidence_ref,
        statement=f"Synthetic high-risk score claim for {case_id}",
        source_artifact_refs=[artifact_id],
    )
    graph = EvidenceGraph()
    graph.add_claim(score_claim)
    receipts = _trusted_score_receipts(
        registry,
        evidence_ref,
        case_id=case_id,
        artifact_id=artifact_id,
        pairwise=pairwise,
    )
    rejection = None
    if variant != "clean" and attack != "protocol_drift":
        rejection = _collector_rejection(
            registry,
            evidence_ref,
            attack=attack,
            case_id=case_id,
            artifact_id=artifact_id,
        )
        omitted = {
            "data_leakage": {ReceiptType.SPLIT_LINEAGE},
            "evaluator_tampering": {ReceiptType.EVALUATOR},
            "selection_bias": {
                ReceiptType.SEED_AGGREGATION,
                ReceiptType.REPLICATION,
            },
            "mixed_value_experience": {ReceiptType.EVALUATOR},
        }[attack]
        receipts = [
            receipt for receipt in receipts if receipt.receipt_type not in omitted
        ]
        receipts.append(
            _attack_blocker(
                evidence_ref,
                case_id=case_id,
                artifact_id=artifact_id,
                attack=attack,
                claim_type=claim_type,
            )
        )
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id=f"path::{case_id}::score",
            claim_id=score_claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipts],
        )
    )

    repair_claim = None
    audit_claim = None
    if variant == "mixed":
        repair_claim = Claim(
            claim_id=f"claim::{case_id}::repair",
            claim_type=ClaimType.DEBUG_REPAIR,
            subject_artifact_id=artifact_id,
            task_scope={"task_ids": [task_id]},
            method_fingerprint=_digest(f"{case_id}:method"),
            protocol_ref=active_ref,
            statement=f"Repair the {attack} protocol violation.",
            source_artifact_refs=[artifact_id],
        )
        audit_claim = Claim(
            claim_id=f"claim::{case_id}::audit",
            claim_type=ClaimType.AUDIT_FINDING,
            subject_artifact_id=artifact_id,
            task_scope={"task_ids": [task_id]},
            method_fingerprint=_digest(f"{case_id}:method"),
            protocol_ref=active_ref,
            statement=f"Warning: the historical {attack} outcome is invalid.",
            source_artifact_refs=[artifact_id],
        )
        for claim in (repair_claim, audit_claim):
            graph.add_claim(claim)
            graph.add_path(
                EvidencePath(
                    path_id=f"path::{claim.claim_id}",
                    claim_id=claim.claim_id,
                    receipt_ids=[],
                )
            )

    engine = AuthorityEngine(registry, graph=graph, policy_version="authority_v1")
    rank_request = AuthorityRequest(
        artifact_id=artifact_id,
        claim_id=score_claim.claim_id,
        operation=Operation.RANK,
        decision_stage=DecisionStage.BRANCH_SELECTION,
        active_protocol=active_ref,
        task_context=TaskContext(task_id=task_id, task_family=task_family),
        requesting_component="tier0.factorial.rank",
        generation_stage=GenerationStage.IMPROVE,
        governance_stage=GovernanceStage.BRANCH_SELECTION,
    )
    rank_decision = engine.authorize(rank_request)
    forbidden_text = f"FORBIDDEN_OUTCOME::{case_id}"
    score_clause = _clause(
        score_claim,
        case_id=case_id,
        text=(
            f"Protocol-legal score evidence for {case_id}"
            if variant == "clean"
            else forbidden_text
        ),
        operations=(Operation.RANK,),
        stages=(GenerationStage.IMPROVE,),
        publication_class="certified",
        task_id=task_id,
        task_family=task_family,
    )
    if variant == "mixed":
        assert repair_claim is not None and audit_claim is not None
        clauses = [
            _clause(
                repair_claim,
                case_id=case_id,
                text=repair_claim.statement,
                operations=(Operation.DEBUG_HYPOTHESIS, Operation.REPAIR_SEED),
                stages=(GenerationStage.DEBUG,),
                publication_class="diagnostic",
                task_id=task_id,
                task_family=task_family,
            ),
            _clause(
                audit_claim,
                case_id=case_id,
                text=audit_claim.statement,
                operations=(Operation.DEBUG_HYPOTHESIS, Operation.INSPECT),
                stages=(GenerationStage.DEBUG,),
                publication_class="diagnostic",
                task_id=task_id,
                task_family=task_family,
            ),
            score_clause,
        ]
        visibility, prompt = _visibility_pack(
            engine,
            active_ref,
            case_id=case_id,
            task_id=task_id,
            task_family=task_family,
            clauses=clauses,
            operation=Operation.DEBUG_HYPOTHESIS,
            generation_stage=GenerationStage.DEBUG,
        )
        expected_visible = {clauses[0]["clause_id"], clauses[1]["clause_id"]}
        retained = len(expected_visible & set(visibility["effective_clause_ids"]))
        valid_opportunities = len(expected_visible)
    else:
        visibility, prompt = _visibility_pack(
            engine,
            active_ref,
            case_id=case_id,
            task_id=task_id,
            task_family=task_family,
            clauses=[score_clause],
            operation=Operation.RANK,
            generation_stage=GenerationStage.IMPROVE,
        )
        valid_opportunities = 1 if variant == "clean" else 0
        retained = int(
            variant == "clean"
            and rank_decision.allowed
            and score_clause["clause_id"] in visibility["effective_clause_ids"]
        )

    invalid_present = variant != "clean"
    invalid_activation_count = int(invalid_present and rank_decision.allowed)
    unauthorized_prompt_exposure = int(
        invalid_present and forbidden_text in prompt
    )
    expected_outcome = (
        DecisionOutcome.ALLOW.value
        if variant == "clean"
        else (
            DecisionOutcome.REQUIRE_HUMAN_REVIEW.value
            if attack == "protocol_drift"
            else DecisionOutcome.DENY.value
        )
    )
    passed = (
        rank_decision.outcome.value == expected_outcome
        and invalid_activation_count == 0
        and unauthorized_prompt_exposure == 0
        and retained == valid_opportunities
    )
    return {
        "case_id": case_id,
        "protocol_ref": active_ref.key(),
        "attack": attack,
        "variant": variant,
        "invalid_attack_present": invalid_present,
        "invalid_activation_count": invalid_activation_count,
        "unauthorized_prompt_exposure": unauthorized_prompt_exposure,
        "valid_knowledge_opportunity_count": valid_opportunities,
        "valid_knowledge_retained_count": retained,
        "collector_rejection": rejection,
        "requested_claim_refs": [
            score_claim.claim_id,
            *(
                [repair_claim.claim_id, audit_claim.claim_id]
                if repair_claim is not None and audit_claim is not None
                else []
            ),
        ],
        "authority_decisions": [_decision_trace(rank_request, rank_decision)],
        "visibility_trace": visibility,
        "lineage_trace": {
            "source_claim_id": score_claim.claim_id,
            "source_protocol_ref": evidence_ref.key(),
            "active_protocol_ref": active_ref.key(),
            "scope_widened": False,
        },
        "expected_primary_outcome": expected_outcome,
        "passed": passed,
    }


def _protocol_repair_code(protocol_id: str) -> str:
    splitter = {
        "random-classification": (
            "from sklearn.model_selection import StratifiedKFold\n"
            "folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n"
            "train_idx, valid_idx = next(folds.split(train_text, labels))"
        ),
        "grouped-classification": (
            "from sklearn.model_selection import GroupKFold\n"
            "folds = GroupKFold(n_splits=5)\n"
            "train_idx, valid_idx = next(folds.split(train_text, labels, groups))"
        ),
        "chronological-regression": (
            "from sklearn.model_selection import TimeSeriesSplit\n"
            "folds = TimeSeriesSplit(n_splits=5)\n"
            "train_idx, valid_idx = next(folds.split(train_text))"
        ),
    }[protocol_id]
    return _SOURCE_CODE.replace(
        "from sklearn.linear_model import LogisticRegression",
        "from sklearn.linear_model import LogisticRegression\n"
        "from agents.protocol_repair_runtime import ProtocolProvenanceGuard",
    ).replace(
        "vectorizer = TfidfVectorizer(ngram_range=(1, 2))",
        "guard = ProtocolProvenanceGuard()\n"
        + splitter
        + "\nguard.register_partition('outer_train', train_idx)\n"
        + "guard.register_partition('outer_holdout', valid_idx)\n"
        + "vectorizer = TfidfVectorizer(ngram_range=(1, 2))",
    )


def _fake_replay_case(
    registry: ProtocolRegistry,
    active_ref: ProtocolRef,
    *,
    variant: str,
) -> dict[str, Any]:
    attack = "method_changing_fake_replay"
    case_id = f"{active_ref.protocol_id}::{attack}::{variant}"
    surface = ProtocolRepairSurface.from_protocol_spec(registry.resolve(active_ref))
    replay_code = _protocol_repair_code(active_ref.protocol_id)
    if variant != "clean":
        replay_code = replay_code.replace(
            "LogisticRegression(C=1.0, max_iter=100)",
            "RandomForestClassifier(n_estimators=100)",
        ).replace(
            "from sklearn.linear_model import LogisticRegression",
            "from sklearn.ensemble import RandomForestClassifier",
        )
    verification = verify_protocol_only_patch(
        _SOURCE_CODE,
        replay_code,
        surface,
        source_artifact_id=f"source::{case_id}",
        replay_artifact_id=f"replay::{case_id}",
    )
    invalid_present = variant != "clean"
    invalid_activation_count = int(
        invalid_present and verification.identity == ReplayIdentity.METHOD_PRESERVED
    )
    if variant == "clean":
        valid_opportunities = 1
        retained = int(verification.identity == ReplayIdentity.METHOD_PRESERVED)
        diagnostic = None
    elif variant == "mixed":
        diagnostic = authorize_derivation_operation(
            Operation.DISTILL_DIAGNOSTIC,
            parent_claim_refs=[f"claim::{case_id}::audit"],
            clean_ancestry=False,
        )
        valid_opportunities = 1
        retained = int(diagnostic.allowed)
    else:
        diagnostic = None
        valid_opportunities = 0
        retained = 0
    expected_identity = (
        ReplayIdentity.METHOD_PRESERVED
        if variant == "clean"
        else ReplayIdentity.SUCCESSOR_METHOD
    )
    passed = (
        verification.identity == expected_identity
        and invalid_activation_count == 0
        and retained == valid_opportunities
    )
    return {
        "case_id": case_id,
        "protocol_ref": active_ref.key(),
        "attack": attack,
        "variant": variant,
        "invalid_attack_present": invalid_present,
        "invalid_activation_count": invalid_activation_count,
        "unauthorized_prompt_exposure": 0,
        "valid_knowledge_opportunity_count": valid_opportunities,
        "valid_knowledge_retained_count": retained,
        "collector_rejection": None,
        "requested_claim_refs": [
            f"claim::{case_id}::old-method",
            *([f"claim::{case_id}::audit"] if diagnostic is not None else []),
        ],
        "authority_decisions": [],
        "visibility_trace": {
            "requested_operation": Operation.REPAIR_SEED.value,
            "requested_generation_stage": GenerationStage.DEBUG.value,
            "requested_governance_stage": GovernanceStage.REPLAY.value,
            "active_protocol_ref": active_ref.key(),
            "bundle_version": "tier0-synthetic-v1",
            "split_id": f"{active_ref.protocol_id}-tier0-v1",
            "effective_clause_ids": (
                [f"clause::{case_id}::audit"]
                if diagnostic is not None and diagnostic.allowed
                else []
            ),
            "suppressed_clause_refs": (
                [f"clause::{case_id}::old-method"] if invalid_present else []
            ),
            "warning_clause_ids": (
                [f"clause::{case_id}::audit"] if diagnostic is not None else []
            ),
            "rendered_prompt_sha256": _digest(
                "diagnostic" if diagnostic is not None and diagnostic.allowed else ""
            ),
            "rendered_prompt_character_count": (
                len("diagnostic")
                if diagnostic is not None and diagnostic.allowed
                else 0
            ),
        },
        "lineage_trace": {
            "source_artifact_id": verification.source_artifact_id,
            "replay_artifact_id": verification.replay_artifact_id,
            "source_code_sha256": verification.source_code_sha256,
            "replay_code_sha256": verification.replay_code_sha256,
            "source_protected_surface_hash": verification.source_protected_surface_hash,
            "replay_protected_surface_hash": verification.replay_protected_surface_hash,
            "protected_changes": verification.protected_changes,
            "unclassified_call_deltas": list(verification.unclassified_call_deltas),
            "scope_widened": False,
        },
        "replay_identity": verification.identity.value,
        "replay_reason": verification.reason,
        "expected_replay_identity": expected_identity.value,
        "passed": passed,
    }


def _derivation_case(
    registry: ProtocolRegistry,
    active_ref: ProtocolRef,
    *,
    variant: str,
) -> dict[str, Any]:
    del registry
    attack = "derived_memory_laundering"
    case_id = f"{active_ref.protocol_id}::{attack}::{variant}"
    task_id = f"task::{active_ref.protocol_id}"
    parent_scope = AuthorityScope(
        claim_types=[ClaimType.METHOD_HYPOTHESIS.value],
        operations=[Operation.DERIVED_PUBLICATION.value],
        stages=[DecisionStage.DISTILLATION.value],
        protocol_hashes=[active_ref.canonical_hash],
        task_ids=[task_id],
        generation_stages=[GenerationStage.EVOLUTION.value],
        governance_stages=[GovernanceStage.DISTILLATION.value],
    )
    parent = AuthorityDecision(
        decision_id="excluded-from-report",
        outcome=DecisionOutcome.ALLOW,
        permitted_scope=parent_scope,
        satisfied_paths=[f"path::{case_id}::parent"],
        missing_obligations=[],
        blocking_receipts=[],
        required_action=None,
        policy_version="authority_v1",
        claim_id=f"claim::{case_id}::parent",
        artifact_id=f"artifact::{case_id}",
        operation=Operation.DERIVED_PUBLICATION.value,
        decision_stage=DecisionStage.DISTILLATION.value,
        generation_stage=GenerationStage.EVOLUTION.value,
        governance_stage=GovernanceStage.DISTILLATION.value,
    )
    scope_widened = variant != "clean"
    requested_scope = AuthorityScope(
        **{
            **dataclasses.asdict(parent_scope),
            "task_ids": (
                [task_id, "foreign-task"] if scope_widened else [task_id]
            ),
        }
    )
    scope_validation = validate_derivation([parent], requested_scope)
    publication = authorize_derivation_operation(
        Operation.DERIVED_PUBLICATION,
        parent_claim_refs=[parent.claim_id],
        clean_ancestry=variant == "clean",
        scope_widened=scope_widened,
    )
    invalid_present = variant != "clean"
    invalid_activation_count = int(
        invalid_present and scope_validation.allowed and publication.allowed
    )
    if variant == "clean":
        valid_opportunities = 1
        retained = int(scope_validation.allowed and publication.allowed)
        diagnostic = None
    elif variant == "mixed":
        diagnostic = authorize_derivation_operation(
            Operation.DISTILL_DIAGNOSTIC,
            parent_claim_refs=[f"claim::{case_id}::audit"],
            clean_ancestry=False,
            scope_widened=False,
        )
        valid_opportunities = 1
        retained = int(diagnostic.allowed)
    else:
        diagnostic = None
        valid_opportunities = 0
        retained = 0
    expected_allowed = variant == "clean"
    passed = (
        (scope_validation.allowed and publication.allowed) == expected_allowed
        and invalid_activation_count == 0
        and retained == valid_opportunities
    )
    return {
        "case_id": case_id,
        "protocol_ref": active_ref.key(),
        "attack": attack,
        "variant": variant,
        "invalid_attack_present": invalid_present,
        "invalid_activation_count": invalid_activation_count,
        "unauthorized_prompt_exposure": 0,
        "valid_knowledge_opportunity_count": valid_opportunities,
        "valid_knowledge_retained_count": retained,
        "collector_rejection": None,
        "requested_claim_refs": [
            parent.claim_id,
            *([f"claim::{case_id}::audit"] if diagnostic is not None else []),
        ],
        "authority_decisions": [],
        "visibility_trace": {
            "requested_operation": Operation.DERIVED_PUBLICATION.value,
            "requested_generation_stage": GenerationStage.EVOLUTION.value,
            "requested_governance_stage": GovernanceStage.DISTILLATION.value,
            "active_protocol_ref": active_ref.key(),
            "bundle_version": "tier0-synthetic-v1",
            "split_id": f"{active_ref.protocol_id}-tier0-v1",
            "effective_clause_ids": (
                [f"clause::{case_id}::derived"]
                if expected_allowed
                else (
                    [f"clause::{case_id}::audit"]
                    if diagnostic is not None and diagnostic.allowed
                    else []
                )
            ),
            "suppressed_clause_refs": (
                [f"clause::{case_id}::laundered"] if invalid_present else []
            ),
            "warning_clause_ids": (
                [f"clause::{case_id}::audit"] if diagnostic is not None else []
            ),
            "rendered_prompt_sha256": _digest(
                "diagnostic" if diagnostic is not None and diagnostic.allowed else ""
            ),
            "rendered_prompt_character_count": (
                len("diagnostic")
                if diagnostic is not None and diagnostic.allowed
                else 0
            ),
        },
        "lineage_trace": {
            "parent_claim_refs": [parent.claim_id],
            "parent_scope": _jsonable(parent_scope),
            "requested_scope": _jsonable(requested_scope),
            "scope_widened": scope_widened,
            "scope_validation_allowed": scope_validation.allowed,
            "scope_validation_reasons": scope_validation.reasons,
            "publication_outcome": publication.outcome.value,
            "publication_reasons": publication.reasons,
        },
        "expected_publication_allowed": expected_allowed,
        "passed": passed,
    }


def run_factorial(
    protocol_registry: str | Path,
    *,
    created_at: str,
) -> dict[str, Any]:
    registry = ProtocolRegistry(protocol_registry)
    refs = [registry.get(protocol_id, "1").ref() for protocol_id in PROTOCOL_IDS]
    cases: list[dict[str, Any]] = []
    authority_attacks = {
        "data_leakage",
        "evaluator_tampering",
        "selection_bias",
        "protocol_drift",
        "mixed_value_experience",
    }
    for index, active_ref in enumerate(refs):
        alternate_ref = refs[(index + 1) % len(refs)]
        for attack in ATTACKS:
            for variant in VARIANTS:
                if attack in authority_attacks:
                    case = _authority_case(
                        registry,
                        active_ref,
                        alternate_ref,
                        attack=attack,
                        variant=variant,
                    )
                elif attack == "method_changing_fake_replay":
                    case = _fake_replay_case(
                        registry,
                        active_ref,
                        variant=variant,
                    )
                else:
                    case = _derivation_case(
                        registry,
                        active_ref,
                        variant=variant,
                    )
                case["trace_contract"] = _trace_contract(case)
                cases.append(case)
    cases.sort(key=lambda row: (row["protocol_ref"], row["attack"], row["variant"]))
    invalid_cases = [case for case in cases if case["invalid_attack_present"]]
    invalid_activations = sum(case["invalid_activation_count"] for case in cases)
    valid_opportunities = sum(
        case["valid_knowledge_opportunity_count"] for case in cases
    )
    valid_retained = sum(case["valid_knowledge_retained_count"] for case in cases)
    unauthorized_prompt_exposure = sum(
        case["unauthorized_prompt_exposure"] for case in cases
    )

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        invalid = [case for case in group if case["invalid_attack_present"]]
        invalid_count = sum(case["invalid_activation_count"] for case in group)
        opportunities = sum(
            case["valid_knowledge_opportunity_count"] for case in group
        )
        retained = sum(case["valid_knowledge_retained_count"] for case in group)
        prompt_count = sum(
            case["unauthorized_prompt_exposure"] for case in group
        )
        return {
            "case_count": len(group),
            "invalid_attack_episode_count": len(invalid),
            "invalid_activation_count": invalid_count,
            "invalid_influence_rate": (
                invalid_count / len(invalid) if invalid else 0.0
            ),
            "valid_knowledge_opportunity_count": opportunities,
            "valid_knowledge_retained_count": retained,
            "valid_knowledge_retention": (
                retained / opportunities if opportunities else 0.0
            ),
            "unauthorized_prompt_exposure_count": prompt_count,
            "all_cases_passed": all(case["passed"] for case in group),
        }

    attack_summaries = {
        attack: summarize([case for case in cases if case["attack"] == attack])
        for attack in ATTACKS
    }
    protocol_summaries = {
        protocol_id: summarize(
            [
                case
                for case in cases
                if case["protocol_ref"].startswith(f"{protocol_id}@")
            ]
        )
        for protocol_id in PROTOCOL_IDS
    }
    expected_keys = {
        (protocol_id, attack, variant)
        for protocol_id in PROTOCOL_IDS
        for attack in ATTACKS
        for variant in VARIANTS
    }
    observed_keys = {
        (case["protocol_ref"].split("@", 1)[0], case["attack"], case["variant"])
        for case in cases
    }
    registry_path = Path(protocol_registry).resolve()
    try:
        registry_ref = registry_path.relative_to(REPO).as_posix()
    except ValueError:
        registry_ref = registry_path.name
    protocol_file_hashes = {
        f"{protocol_id}-v1.json": _digest(
            (registry_path / f"{protocol_id}-v1.json").read_text(encoding="utf-8")
        )
        for protocol_id in PROTOCOL_IDS
    }
    protocol_contracts = {
        protocol_id: {
            "protocol_ref": registry.get(protocol_id, "1").ref().key(),
            "split_strategy": registry.get(protocol_id, "1").data_split_policy[
                "strategy"
            ],
            "preprocessing_fit_scope": registry.get(
                protocol_id, "1"
            ).preprocessing_policy["fit_scope"],
            "metric_name": registry.get(protocol_id, "1").metric_spec["name"],
            "metric_direction": registry.get(protocol_id, "1").metric_spec[
                "direction"
            ],
            "enforce_protocol_payloads": registry.get(
                protocol_id, "1"
            ).promotion_policy.get("enforce_protocol_payloads")
            is True,
        }
        for protocol_id in PROTOCOL_IDS
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at": str(created_at),
        "runner_source_sha256": _file_sha256(Path(__file__)),
        "implementation_source_hashes": {
            relative: _file_sha256(REPO / relative)
            for relative in IMPLEMENTATION_PATHS
        },
        "authority_policy_version": "authority_v1",
        "protocol_registry": registry_ref,
        "protocol_file_hashes": protocol_file_hashes,
        "protocol_contracts": protocol_contracts,
        "protocol_refs": [ref.key() for ref in refs],
        "attacks": list(ATTACKS),
        "variants": list(VARIANTS),
        "case_count": len(cases),
        "expected_case_count": len(expected_keys),
        "matrix_complete": observed_keys == expected_keys,
        "all_cases_passed": all(case["passed"] for case in cases),
        "failed_case_ids": [
            case["case_id"] for case in cases if not case["passed"]
        ],
        "invalid_attack_episode_count": len(invalid_cases),
        "invalid_activation_count": invalid_activations,
        "invalid_influence_rate": (
            invalid_activations / len(invalid_cases) if invalid_cases else 0.0
        ),
        "valid_knowledge_opportunity_count": valid_opportunities,
        "valid_knowledge_retained_count": valid_retained,
        "valid_knowledge_retention": (
            valid_retained / valid_opportunities if valid_opportunities else 0.0
        ),
        "unauthorized_prompt_exposure_count": unauthorized_prompt_exposure,
        "collector_rejection_count": sum(
            case["collector_rejection"] is not None for case in cases
        ),
        "attack_summaries": attack_summaries,
        "protocol_summaries": protocol_summaries,
        "cases": cases,
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic WP8 Tier-0 7-attack x 3-protocol x "
            "clean/invalid/mixed Decision Admissibility matrix."
        )
    )
    parser.add_argument(
        "--protocol-registry",
        type=Path,
        default=REPO / "mlevolve" / "config" / "protocols",
    )
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite Tier-0 report: {args.output}")
    report = run_factorial(
        args.protocol_registry,
        created_at=args.created_at,
    )
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "case_count",
                    "matrix_complete",
                    "all_cases_passed",
                    "invalid_influence_rate",
                    "valid_knowledge_retention",
                    "unauthorized_prompt_exposure_count",
                    "report_hash",
                )
            },
            sort_keys=True,
        )
    )
    if not report["matrix_complete"] or not report["all_cases_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
