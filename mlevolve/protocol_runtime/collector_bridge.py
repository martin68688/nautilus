"""Bridge signed Collector journals into existing trusted Receipt collectors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from authority.collectors import (
    CodeExecutionCollector,
    EvaluatorIntegrityCollector,
    FitScopeCollector,
    MethodIdentityCollector,
    PredictionScopeCollector,
    SelectionFreezeCollector,
    SplitLineageCollector,
    TrustedCollectorHost,
)
from authority.models import Receipt
from authority.protocol_execution_contract import ProtocolExecutionContract

from .collector import verify_collector_artifacts


_COLLECTORS = {
    "split_lineage": SplitLineageCollector,
    "fit_scope": FitScopeCollector,
    "prediction_scope": PredictionScopeCollector,
    "evaluator": EvaluatorIntegrityCollector,
    "selection_freeze": SelectionFreezeCollector,
}


def bridge_signed_journal_to_receipts(
    root: str | Path,
    *,
    contract: ProtocolExecutionContract,
    collector_version: str = "host_protocol_runtime_v1",
) -> list[Receipt]:
    expected_public_key = str(
        contract.collector_spec.get("public_key_ed25519") or ""
    )
    if not expected_public_key:
        raise ValueError("Execution Contract lacks a Collector trust anchor")
    verified = verify_collector_artifacts(
        root, expected_public_key_ed25519=expected_public_key
    )
    manifest = verified["manifest"]
    report = verified["report"]
    if report.get("status") != "pass" or report.get("missing_events"):
        raise ValueError("Incomplete runtime journal cannot mint trusted Receipts")
    if manifest.get("contract_hash") == "" or len(manifest["contract_hash"]) != 64:
        raise ValueError("Runtime journal lacks a Contract hash")
    if manifest["contract_hash"] != contract.contract_hash:
        raise ValueError("Runtime journal Execution Contract mismatch")
    protocol_ref = contract.protocol_ref
    execution = manifest.get("execution") or {}
    host = TrustedCollectorHost(
        f"protocol-sidecar:{manifest['run_id']}",
        collector_version=collector_version,
    )
    common = {
        "artifact_id": manifest["node_id"],
        "run_id": manifest["run_id"],
        "protocol_ref": protocol_ref,
        "source": "host.protocol_runtime.sidecar.signed_journal",
    }
    receipts = [
        host.collect(
            CodeExecutionCollector,
            **common,
            payload={
                "exit_status": execution.get("exit_status"),
                "executed_path": execution.get("executed_path"),
                "run_hash": execution.get("run_hash"),
                "execution_contract_hash": contract.contract_hash,
            },
        ),
        host.collect(
            MethodIdentityCollector,
            **common,
            payload={
                "method_fingerprint": manifest["code_sha256"],
                "code_sha256": manifest["code_sha256"],
                "execution_contract_hash": contract.contract_hash,
            },
        ),
    ]
    for event in verified["events"]:
        collector = _COLLECTORS[event["kind"]]
        payload: dict[str, Any] = dict(event["trusted_payload"])
        payload.pop("candidate_payload_hash", None)
        payload["execution_contract_hash"] = contract.contract_hash
        receipts.append(host.collect(collector, **common, payload=payload))
    return receipts


__all__ = ["bridge_signed_journal_to_receipts"]
