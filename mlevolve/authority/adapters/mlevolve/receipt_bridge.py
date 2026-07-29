from __future__ import annotations

import hashlib
import math
from typing import Any

from ...collectors import (
    CodeExecutionCollector,
    EvaluatorIntegrityCollector,
    FitScopeCollector,
    MethodIdentityCollector,
    PredictionScopeCollector,
    SelectionFreezeCollector,
    SplitLineageCollector,
    TrustedCollectorHost,
    UntrustedObservationError,
)
from ...models import ClaimType, ProtocolRef, Receipt, ReceiptType
from ...protocol_registry import canonical_json
from ...receipt_collectors import make_receipt
from ...runtime_protocol import (
    OBSERVATION_SCHEMA,
    PROTOCOL_EVIDENCE_LEVEL,
    PROTOCOL_EVIDENCE_SCHEMA,
    verify_runtime_protocol_observation,
)


_SCORE_CLAIMS = [
    ClaimType.SCORE.value,
    ClaimType.PAIRWISE_SUPERIORITY.value,
    ClaimType.CAUSAL_ATTRIBUTION.value,
    ClaimType.GENERALIZATION.value,
]
_SCORE_BLOCKED_BY_AUDIT = [*_SCORE_CLAIMS]


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _legacy_method_and_execution_receipts(
    node: Any,
    protocol_ref: ProtocolRef,
    run_id: str,
) -> list[Receipt]:
    artifact_id = str(node.id)
    receipts = [
        make_receipt(
            ReceiptType.METHOD_IDENTITY,
            artifact_id,
            run_id,
            protocol_ref,
            "legacy.mlevolve.node_adapter",
            {"method_fingerprint": str(getattr(node, "method_fingerprint", ""))},
        )
    ]
    metric = getattr(getattr(node, "metric", None), "value", None)
    if getattr(node, "exec_time", None) is not None or metric is not None:
        receipts.append(
            make_receipt(
                ReceiptType.CODE_EXECUTION,
                artifact_id,
                run_id,
                protocol_ref,
                "legacy.mlevolve.executor",
                {"exec_time": getattr(node, "exec_time", None), "metric": metric},
            )
        )
    return receipts


def _trusted_method_and_execution_receipts(
    node: Any,
    protocol_ref: ProtocolRef,
    run_id: str,
    host: TrustedCollectorHost,
) -> list[Receipt]:
    artifact_id = str(node.id)
    code = str(getattr(node, "code", "") or "")
    code_sha256 = str(
        getattr(node, "code_sha256_expected", "") or _sha256(code)
    )
    method_fingerprint = str(
        getattr(node, "method_fingerprint", "") or code_sha256
    )
    receipts = [
        host.collect(
            MethodIdentityCollector,
            artifact_id=artifact_id,
            run_id=run_id,
            protocol_ref=protocol_ref,
            source="host.node_code_snapshot",
            payload={
                "method_fingerprint": method_fingerprint,
                "code_sha256": code_sha256,
            },
        )
    ]
    metric = getattr(getattr(node, "metric", None), "value", None)
    execution_observed = getattr(node, "exec_time", None) is not None or metric is not None
    execution_success = (
        execution_observed
        and getattr(node, "is_buggy", False) is not True
        and getattr(node, "is_valid", True) is not False
    )
    if execution_success:
        run_hash = _sha256(
            f"{run_id}|{artifact_id}|{code_sha256}|{getattr(node, 'exec_time', None)}"
        )
        receipts.append(
            host.collect(
                CodeExecutionCollector,
                artifact_id=artifact_id,
                run_id=run_id,
                protocol_ref=protocol_ref,
                source="host.executor_result",
                payload={
                    "exit_status": 0,
                    "executed_path": artifact_id,
                    "run_hash": run_hash,
                },
            )
        )
    return receipts


def _trusted_runtime_protocol_receipts(
    node: Any,
    protocol_ref: ProtocolRef,
    run_id: str,
    host: TrustedCollectorHost,
    task_id: str | None = None,
) -> list[Receipt]:
    full_runtime = dict(
        (getattr(node, "protocol_observation", None) or {}).get(
            "host_full_runtime"
        )
        or {}
    )
    if full_runtime:
        try:
            from authority.protocol_execution_contract import (
                ProtocolExecutionContract,
            )
            from protocol_runtime.collector_bridge import (
                bridge_signed_journal_to_receipts,
            )

            evidence_hash = str(full_runtime.get("evidence_hash") or "")
            expected_evidence_hash = _sha256(
                canonical_json(
                    {
                        key: value
                        for key, value in full_runtime.items()
                        if key != "evidence_hash"
                    }
                )
            )
            contract = ProtocolExecutionContract.from_dict(
                dict(full_runtime["contract"])
            )
            expected_code_sha256 = str(
                getattr(node, "code_sha256_expected", "")
                or _sha256(str(getattr(node, "code", "") or ""))
            )
            if (
                full_runtime.get("schema")
                != "mlevolve_full_runtime_evidence_v1"
                or full_runtime.get("status") != "pass"
                or evidence_hash != expected_evidence_hash
                or full_runtime.get("node_id") != str(node.id)
                or full_runtime.get("code_sha256") != expected_code_sha256
                or full_runtime.get("contract_hash") != contract.contract_hash
                or contract.protocol_ref != protocol_ref
                or (task_id is not None and contract.task_id != str(task_id))
                or full_runtime.get("missing_events") != []
            ):
                return []
            receipts = bridge_signed_journal_to_receipts(
                str(full_runtime["collector_root"]), contract=contract
            )
            required = {
                ReceiptType.CODE_EXECUTION,
                ReceiptType.METHOD_IDENTITY,
                ReceiptType.SPLIT_LINEAGE,
                ReceiptType.FIT_SCOPE,
                ReceiptType.PREDICTION_SCOPE,
                ReceiptType.EVALUATOR,
                ReceiptType.SELECTION_FREEZE,
            }
            by_type = {receipt.receipt_type: receipt for receipt in receipts}
            if not required <= set(by_type):
                return []
            metric = getattr(getattr(node, "metric", None), "value", None)
            evaluator_metric = by_type[ReceiptType.EVALUATOR].payload.get(
                "metric_value"
            )
            if metric is None or evaluator_metric is None or not math.isclose(
                float(metric),
                float(evaluator_metric),
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                return []
            return receipts
        except (KeyError, TypeError, ValueError, OSError):
            return []

    observation = getattr(node, "protocol_observation", None) or {}
    audit = getattr(node, "leakage_audit", None) or {}
    expected_code_sha256 = str(
        getattr(node, "code_sha256_expected", "")
        or _sha256(str(getattr(node, "code", "") or ""))
    )
    event_hashes = observation.get("event_hashes") or {}
    scope_hashes = observation.get("scope_hashes") or {}
    scope_input_hashes = observation.get("scope_input_hashes") or {}
    scope_output_hashes = observation.get("scope_output_hashes") or {}
    observation_clean = bool(
        verify_runtime_protocol_observation(observation)
        and observation.get("schema") == OBSERVATION_SCHEMA
        and observation.get("source_code_sha256") == expected_code_sha256
        and observation.get("code_snapshot_frozen_before_execution") is True
        and observation.get("evidence_level") == PROTOCOL_EVIDENCE_LEVEL
        and audit.get("schema") == "mlevolve_leakage_audit_v2"
        and str(audit.get("detector_version") or "").startswith(
            "deterministic_static_"
        )
        and audit.get("detector_status") == "complete"
        and audit.get("code_sha256") == expected_code_sha256
        and audit.get("status") == "clean"
        and audit.get("issues") == []
        and audit.get("hard_block") is False
        and audit.get("metric_disposition") == "accept"
        and all(event_hashes.get(kind) for kind in (
            "split_lineage",
            "fit_scope",
            "prediction_scope",
            "evaluator",
            "selection_freeze",
        ))
        and all(scope_hashes.get(kind) for kind in (
            "split_lineage",
            "fit_scope",
            "prediction_scope",
            "evaluator",
            "selection_freeze",
        ))
    )
    if observation_clean:
        static_audit_sha256 = _sha256(canonical_json({
            "schema": audit.get("schema"),
            "detector_version": audit.get("detector_version"),
            "detector_status": audit.get("detector_status"),
            "code_sha256": audit.get("code_sha256"),
            "structural_sha256": audit.get("structural_sha256"),
            "issues": audit.get("issues"),
            "status": audit.get("status"),
            "metric_disposition": audit.get("metric_disposition"),
        }))
        scope_binding_sha256 = _sha256(canonical_json({
            "scope_hashes": scope_hashes,
            "scope_input_hashes": scope_input_hashes,
            "scope_output_hashes": scope_output_hashes,
        }))
        protocol_evidence = {
            "schema": PROTOCOL_EVIDENCE_SCHEMA,
            "evidence_level": observation["evidence_level"],
            "source_code_sha256": expected_code_sha256,
            "executed_source_sha256": observation["executed_source_sha256"],
            "plan_sha256": observation["plan_sha256"],
            "trace_sha256": observation["trace_sha256"],
            "attestation_sha256": observation["attestation_sha256"],
            "static_audit_sha256": static_audit_sha256,
            "scope_binding_sha256": scope_binding_sha256,
        }
        artifact_id = str(node.id)
        direction = "maximize" if bool(
            getattr(getattr(node, "metric", None), "maximize", True)
        ) else "minimize"
        evaluator_hash = _sha256(canonical_json({
            "protocol_hash": protocol_ref.canonical_hash,
            "direction": direction,
            "callable_refs": (observation.get("callable_refs") or {}).get(
                "evaluator", []
            ),
            "event_hashes": event_hashes["evaluator"],
            "scope_hashes": scope_hashes["evaluator"],
        }))
        evaluator_inputs_hash = _sha256(canonical_json({
            "prediction_scope": scope_hashes["prediction_scope"],
            "evaluator_inputs": (
                scope_input_hashes.get("evaluator")
                or scope_hashes["evaluator"]
            ),
            "code_sha256": expected_code_sha256,
        }))
        candidate_set_hash = _sha256(canonical_json({
            "selection_freeze": scope_hashes["selection_freeze"],
            "code_sha256": expected_code_sha256,
        }))
        specifications = (
            (SplitLineageCollector, {
                "partition_hashes": {
                    f"executed_split_{index}": digest
                    for index, digest in enumerate(
                        scope_hashes["split_lineage"], start=1
                    )
                },
                "overlap_count": 0,
                "protocol_evidence": protocol_evidence,
            }),
            (FitScopeCollector, {
                "fit_scope_hashes": {
                    f"executed_fit_{index}": digest
                    for index, digest in enumerate(
                        scope_hashes["fit_scope"], start=1
                    )
                },
                "holdout_fit_count": 0,
                "protocol_evidence": protocol_evidence,
            }),
            (PredictionScopeCollector, {
                "prediction_scope_hashes": {
                    f"executed_prediction_{index}": digest
                    for index, digest in enumerate(
                        scope_hashes["prediction_scope"], start=1
                    )
                },
                "forbidden_overlap_count": 0,
                "protocol_evidence": protocol_evidence,
            }),
            (EvaluatorIntegrityCollector, {
                "evaluator_hash": evaluator_hash,
                "inputs_hash": evaluator_inputs_hash,
                "metric_direction": direction,
                "tampered": False,
                "protocol_evidence": protocol_evidence,
            }),
            (SelectionFreezeCollector, {
                "candidate_set_hash": candidate_set_hash,
                "frozen_before_holdout": True,
                "protocol_evidence": protocol_evidence,
            }),
        )
        receipts: list[Receipt] = []
        for collector, payload in specifications:
            try:
                receipts.append(
                    host.collect(
                        collector,
                        artifact_id=artifact_id,
                        run_id=run_id,
                        protocol_ref=protocol_ref,
                        source="host.executor_protocol_scope_observer",
                        payload=payload,
                    )
                )
            except UntrustedObservationError:
                return []
        return receipts

    runtime = (getattr(node, "protocol_repair", None) or {}).get(
        "runtime_provenance", {}
    )
    if runtime.get("status") != "clean":
        return []
    digest = str(runtime.get("payload_sha256") or "")
    counts = runtime.get("counts") or {}
    required_counts = ("partitions", "fits", "predictions", "selections", "final_evaluations")
    if len(digest) != 64 or any(int(counts.get(key, 0)) <= 0 for key in required_counts):
        return []
    artifact_id = str(node.id)
    direction = "maximize" if bool(
        getattr(getattr(node, "metric", None), "maximize", True)
    ) else "minimize"
    evaluator_hash = _sha256(
        f"{protocol_ref.canonical_hash}|{direction}|host_terminal_evaluator"
    )
    specifications = (
        (SplitLineageCollector, {
            "partition_hashes": {"runtime_provenance": digest},
            "overlap_count": 0,
        }),
        (FitScopeCollector, {
            "fit_scope_hashes": {"runtime_provenance": digest},
            "holdout_fit_count": 0,
        }),
        (PredictionScopeCollector, {
            "prediction_scope_hashes": {"runtime_provenance": digest},
            "forbidden_overlap_count": 0,
        }),
        (EvaluatorIntegrityCollector, {
            "evaluator_hash": evaluator_hash,
            "inputs_hash": digest,
            "metric_direction": direction,
            "tampered": False,
        }),
        (SelectionFreezeCollector, {
            "candidate_set_hash": digest,
            "frozen_before_holdout": True,
        }),
    )
    receipts: list[Receipt] = []
    for collector, payload in specifications:
        try:
            receipts.append(
                host.collect(
                    collector,
                    artifact_id=artifact_id,
                    run_id=run_id,
                    protocol_ref=protocol_ref,
                    source="host.protocol_runtime_audit",
                    payload=payload,
                )
            )
        except UntrustedObservationError:
            return []
    return receipts


def _legacy_static_audit_receipts(
    node: Any,
    protocol_ref: ProtocolRef,
    run_id: str,
) -> list[Receipt]:
    artifact_id = str(node.id)
    audit = getattr(node, "leakage_audit", None) or {}
    if not audit:
        return []
    expected_hash = str(
        getattr(node, "code_sha256_expected", "")
        or _sha256(str(getattr(node, "code", "") or ""))
    )
    issue_categories = sorted({
        str(item.get("category"))
        for item in audit.get("issues", [])
        if isinstance(item, dict) and item.get("category")
    })
    strategy_clean = True
    if getattr(node, "draft_role", None) == "novel_exploration" and getattr(
        node, "selected_strategy", None
    ):
        alignment = getattr(node, "strategy_alignment", None) or {}
        strategy_clean = bool(
            alignment.get("status") == "verified"
            and alignment.get("rank_eligible") is True
        )
    audit_clean = bool(
        audit.get("schema") == "mlevolve_leakage_audit_v2"
        and audit.get("detector_status") == "complete"
        and audit.get("code_sha256") == expected_hash
        and audit.get("status") == "clean"
        and audit.get("metric_disposition") == "accept"
        and audit.get("paper_grade_eligible") is True
        and strategy_clean
    )
    if audit_clean:
        evidence = {
            ReceiptType.SPLIT_LINEAGE: {
                "partition_lineage_verified": True,
                "observed_issue_categories": issue_categories,
            },
            ReceiptType.FIT_SCOPE: {
                "fit_scope_verified": True,
                "observed_issue_categories": issue_categories,
            },
            ReceiptType.PREDICTION_SCOPE: {
                "prediction_scope_verified": True,
                "observed_issue_categories": issue_categories,
            },
            ReceiptType.EVALUATOR: {
                "evaluator_integrity_verified": True,
                "metric_disposition": audit.get("metric_disposition"),
            },
            ReceiptType.SELECTION_FREEZE: {
                "selection_freeze_verified": True,
                "selection_bias_absent": "selection_bias" not in issue_categories,
            },
        }
        return [
            make_receipt(
                receipt_type,
                artifact_id,
                run_id,
                protocol_ref,
                f"legacy.static_audit.{receipt_type.value}",
                {
                    "audit_schema": audit.get("schema"),
                    "code_sha256": audit.get("code_sha256"),
                    "status": "clean",
                    **payload,
                },
                supports_claim_types=_SCORE_CLAIMS,
            )
            for receipt_type, payload in evidence.items()
        ]
    issue_codes = [
        str(item.get("issue_code"))
        for item in audit.get("issues", [])
        if isinstance(item, dict) and item.get("issue_code")
    ]
    return [
        make_receipt(
            ReceiptType.EVALUATOR,
            artifact_id,
            run_id,
            protocol_ref,
            "legacy.static_audit.blocker",
            {
                "contradicts": True,
                "status": audit.get("status", "unknown"),
                "issue_codes": issue_codes,
            },
            supports_claim_types=[ClaimType.AUDIT_FINDING.value],
            blocks_claim_types=_SCORE_BLOCKED_BY_AUDIT,
        )
    ]


def receipts_for_node(
    node: Any,
    protocol_ref: ProtocolRef,
    run_id: str,
    *,
    collector_host: TrustedCollectorHost | None = None,
    task_id: str | None = None,
) -> list[Receipt]:
    if collector_host is None:
        receipts = _legacy_method_and_execution_receipts(node, protocol_ref, run_id)
    else:
        receipts = _trusted_method_and_execution_receipts(
            node, protocol_ref, run_id, collector_host
        )
        receipts.extend(
            _trusted_runtime_protocol_receipts(
                node,
                protocol_ref,
                run_id,
                collector_host,
                task_id=task_id,
            )
        )
    receipts.extend(_legacy_static_audit_receipts(node, protocol_ref, run_id))
    return receipts


def receipts_for_replay_source(
    artifact_id: str,
    code_sha256: str,
    audit: dict[str, Any],
    protocol_ref: ProtocolRef,
    run_id: str,
    *,
    collector_host: TrustedCollectorHost | None = None,
    source_execution_verified: bool = False,
) -> list[Receipt]:
    proxy = type(
        "ReplayArtifact",
        (),
        {
            "id": artifact_id,
            "code": "",
            "method_fingerprint": code_sha256,
            "code_sha256_expected": code_sha256,
            "metric": None,
            # The loader may assert this only after checking the immutable
            # source journal node, code hash, validity and target manifest.
            "exec_time": 0.0 if source_execution_verified else None,
            "is_buggy": False,
            "is_valid": True,
            "leakage_audit": audit,
            "protocol_repair": {},
        },
    )()
    return receipts_for_node(
        proxy,
        protocol_ref,
        run_id,
        collector_host=collector_host,
    )
