from __future__ import annotations

import hashlib
from typing import Any

from ...models import ProtocolRef, Receipt, ReceiptType
from ...receipt_collectors import make_receipt


def receipts_for_node(node: Any, protocol_ref: ProtocolRef, run_id: str) -> list[Receipt]:
    artifact_id = str(node.id)
    receipts: list[Receipt] = []
    method_payload = {"method_fingerprint": str(getattr(node, "method_fingerprint", ""))}
    receipts.append(
        make_receipt(
            ReceiptType.METHOD_IDENTITY,
            artifact_id,
            run_id,
            protocol_ref,
            "mlevolve.node_adapter",
            method_payload,
        )
    )
    metric = getattr(getattr(node, "metric", None), "value", None)
    if getattr(node, "exec_time", None) is not None or metric is not None:
        receipts.append(
            make_receipt(
                ReceiptType.CODE_EXECUTION,
                artifact_id,
                run_id,
                protocol_ref,
                "mlevolve.executor",
                {"exec_time": getattr(node, "exec_time", None), "metric": metric},
            )
        )
    audit = getattr(node, "leakage_audit", None) or {}
    expected_hash = str(
        getattr(node, "code_sha256_expected", "")
        or hashlib.sha256(str(getattr(node, "code", "") or "").encode("utf-8")).hexdigest()
    )
    strategy_clean = True
    if getattr(node, "draft_role", None) == "novel_exploration" and getattr(node, "selected_strategy", None):
        alignment = getattr(node, "strategy_alignment", None) or {}
        strategy_clean = bool(alignment.get("status") == "verified" and alignment.get("rank_eligible") is True)
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
        issue_categories = sorted(
            {
                str(item.get("category"))
                for item in audit.get("issues", [])
                if isinstance(item, dict) and item.get("category")
            }
        )
        runtime_provenance = (getattr(node, "protocol_repair", None) or {}).get(
            "runtime_provenance", {}
        )
        typed_evidence = {
            ReceiptType.SPLIT_LINEAGE: (
                "mlevolve.leakage_audit.split_lineage",
                {
                    "partition_lineage_verified": True,
                    "forbidden_categories_absent": ["target_leakage", "split_contamination"],
                    "observed_issue_categories": issue_categories,
                    "runtime_partition_events": runtime_provenance.get("partitions", 0),
                },
            ),
            ReceiptType.FIT_SCOPE: (
                "mlevolve.leakage_audit.fit_scope",
                {
                    "fit_scope_verified": True,
                    "forbidden_categories_absent": ["fit_scope", "transductive_contamination"],
                    "observed_issue_categories": issue_categories,
                    "runtime_fit_events": runtime_provenance.get("fits", 0),
                },
            ),
            ReceiptType.PREDICTION_SCOPE: (
                "mlevolve.leakage_audit.prediction_scope",
                {
                    "prediction_scope_verified": True,
                    "observed_issue_categories": issue_categories,
                    "runtime_prediction_events": runtime_provenance.get("predictions", 0),
                },
            ),
            ReceiptType.EVALUATOR: (
                "mlevolve.leakage_audit.evaluator",
                {
                    "evaluator_integrity_verified": True,
                    "metric_disposition": audit.get("metric_disposition"),
                    "detector_status": audit.get("detector_status"),
                },
            ),
            ReceiptType.SELECTION_FREEZE: (
                "mlevolve.leakage_audit.selection_freeze",
                {
                    "selection_freeze_verified": True,
                    "selection_bias_absent": "selection_bias" not in issue_categories,
                    "runtime_selection_events": runtime_provenance.get("selections", 0),
                },
            ),
        }
        for receipt_type, (collector_id, evidence_payload) in typed_evidence.items():
            receipts.append(
                make_receipt(
                    receipt_type,
                    artifact_id,
                    run_id,
                    protocol_ref,
                    collector_id,
                    {
                        "audit_schema": audit.get("schema"),
                        "code_sha256": audit.get("code_sha256"),
                        "status": "clean",
                        **evidence_payload,
                    },
                )
            )
    elif audit:
        receipts.append(
            make_receipt(
                ReceiptType.EVALUATOR,
                artifact_id,
                run_id,
                protocol_ref,
                "mlevolve.leakage_audit",
                {
                    "contradicts": True,
                    "status": audit.get("status", "unknown"),
                    "issue_codes": [item.get("issue_code") for item in audit.get("issues", [])],
                },
            )
        )
    return receipts


def receipts_for_replay_source(
    artifact_id: str,
    code_sha256: str,
    audit: dict[str, Any],
    protocol_ref: ProtocolRef,
    run_id: str,
) -> list[Receipt]:
    proxy = type(
        "ReplayArtifact",
        (),
        {
            "id": artifact_id,
            "method_fingerprint": code_sha256,
            "code_sha256_expected": code_sha256,
            "metric": None,
            "exec_time": 0.0,
            "leakage_audit": audit,
        },
    )()
    return receipts_for_node(proxy, protocol_ref, run_id)
