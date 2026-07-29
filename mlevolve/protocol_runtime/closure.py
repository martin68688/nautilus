"""Dry and pre-terminal evidence-closure primitives."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from authority.authority_engine import AuthorityEngine
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionStage,
    Operation,
    Receipt,
    TaskContext,
)
from authority.protocol_execution_contract import ProtocolExecutionContract
from authority.protocol_registry import ProtocolRegistry

from .collector import verify_collector_artifacts
from .collector_bridge import bridge_signed_journal_to_receipts
from .data_views import read_data_view_manifest, verify_data_view_manifest
from .events import hash_payload


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def dry_evidence_closure(
    contract: ProtocolExecutionContract,
    registry: ProtocolRegistry,
    receipts: Iterable[Receipt],
    *,
    run_id: str,
    node_id: str,
    code_sha256: str,
) -> dict[str, Any]:
    """Run the production compiler/Authority path without creating a Result Fact."""

    receipt_list = list(receipts)
    claim = Claim(
        claim_id=f"preflight:{run_id}:{node_id}:score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id=node_id,
        task_scope={"task_id": contract.task_id},
        method_fingerprint=code_sha256,
        protocol_ref=contract.protocol_ref,
        statement="Dry-run evidence-closure probe; no terminal score or Result Fact.",
        source_artifact_refs=[node_id],
        evidence_refs=[receipt.receipt_id for receipt in receipt_list],
        boundary={
            "dry_run": True,
            "execution_contract_hash": contract.contract_hash,
            "terminal_score_present": False,
        },
    )
    graph = EvidenceGraph()
    graph.add_claim(claim)
    for receipt in receipt_list:
        graph.add_receipt(receipt)
    path = EvidencePath(
        path_id=f"preflight-path:{run_id}:{node_id}",
        claim_id=claim.claim_id,
        receipt_ids=[receipt.receipt_id for receipt in receipt_list],
    )
    graph.add_path(path)
    engine = AuthorityEngine(registry, graph=graph)
    decision = engine.authorize(
        AuthorityRequest(
            artifact_id=node_id,
            claim_id=claim.claim_id,
            operation=Operation.PROMOTE_RESULT,
            decision_stage=DecisionStage.MEMORY_WRITEBACK,
            active_protocol=contract.protocol_ref,
            task_context=TaskContext(contract.task_id, contract.task_family),
            requesting_component="protocol_runtime.preflight.dry_evidence_closure",
        )
    )
    payload = {
        "schema": "mlevolve_preflight_evidence_closure_v1",
        "run_id": run_id,
        "node_id": node_id,
        "code_sha256": code_sha256,
        "contract_hash": contract.contract_hash,
        "protocol_hash": contract.protocol_ref.canonical_hash,
        "terminal_score_present": False,
        "result_fact_created": False,
        "claim": _jsonable(claim),
        "path": _jsonable(path),
        "receipt_ids": [receipt.receipt_id for receipt in receipt_list],
        "authority_decision": _jsonable(decision),
        "status": "pass" if decision.allowed else "blocked",
        "closure_hash": "",
    }
    payload["closure_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "closure_hash"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    content = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def build_training_evidence_manifest(
    contract: ProtocolExecutionContract,
    *,
    data_view_manifest_path: str | Path,
    collector_root: str | Path,
    candidate_code_path: str | Path,
    frozen_submission_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Seal the non-terminal full-training inputs/outputs before evaluator launch."""

    verification = verify_data_view_manifest(
        data_view_manifest_path, contract=contract
    )
    view_manifest = read_data_view_manifest(data_view_manifest_path)
    expected_key = str(contract.collector_spec.get("public_key_ed25519") or "")
    collector = verify_collector_artifacts(
        collector_root, expected_public_key_ed25519=expected_key
    )
    if collector["report"].get("status") != "pass":
        raise ValueError("Full training Collector evidence is incomplete")
    code_path = Path(candidate_code_path)
    submission_path = Path(frozen_submission_path)
    for path, label in ((code_path, "candidate code"), (submission_path, "submission")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Frozen {label} is missing or a symlink")
        path.chmod(path.stat().st_mode & ~0o222)
    code_sha256 = _sha256_file(code_path)
    if code_sha256 != collector["manifest"].get("code_sha256"):
        raise ValueError("Candidate code is not bound to the runtime Collector journal")
    payload = {
        "schema": "mlevolve_host_training_evidence_manifest_v1",
        "contract_hash": contract.contract_hash,
        "protocol_hash": contract.protocol_ref.canonical_hash,
        "data_view_manifest_hash": view_manifest.manifest_hash,
        "data_view_verification_status": verification["status"],
        "candidate_code_path": str(code_path.resolve()),
        "candidate_code_sha256": code_sha256,
        "frozen_submission_path": str(submission_path.resolve()),
        "frozen_submission_sha256": _sha256_file(submission_path),
        "collector_manifest_hash": collector["manifest"]["manifest_hash"],
        "collector_report_hash": collector["report"]["report_hash"],
        "training_complete": True,
        "selection_frozen": "selection_freeze"
        in collector["manifest"]["observed_events"],
        "terminal_exposure_count": 0,
        "terminal_score_observed": False,
        "manifest_hash": "",
    }
    payload["manifest_hash"] = hash_payload(payload, "manifest_hash")
    _write_exclusive(Path(output_path).resolve(), payload)
    return payload


def preterminal_evidence_closure(
    contract: ProtocolExecutionContract,
    registry: ProtocolRegistry,
    *,
    data_view_manifest_path: str | Path,
    collector_root: str | Path,
    training_evidence_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Close every non-terminal obligation before training Pod deletion/evaluation."""

    training_path = Path(training_evidence_manifest_path)
    if training_path.is_symlink() or not training_path.is_file():
        raise ValueError("Training evidence manifest is missing or a symlink")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("schema") != "mlevolve_host_training_evidence_manifest_v1":
        raise ValueError("Training evidence manifest schema mismatch")
    if training.get("manifest_hash") != hash_payload(training, "manifest_hash"):
        raise ValueError("Training evidence manifest hash mismatch")
    view_verification = verify_data_view_manifest(
        data_view_manifest_path, contract=contract
    )
    view_manifest = read_data_view_manifest(data_view_manifest_path)
    expected_key = str(contract.collector_spec.get("public_key_ed25519") or "")
    collector = verify_collector_artifacts(
        collector_root, expected_public_key_ed25519=expected_key
    )
    code_path = Path(str(training.get("candidate_code_path") or ""))
    submission_path = Path(str(training.get("frozen_submission_path") or ""))
    checks = {
        "contract_valid": training.get("contract_hash") == contract.contract_hash,
        "protocol_valid": training.get("protocol_hash")
        == contract.protocol_ref.canonical_hash,
        "data_views_valid": (
            training.get("data_view_manifest_hash") == view_manifest.manifest_hash
            and view_verification["status"] == "pass"
        ),
        "candidate_code_frozen": (
            code_path.is_file()
            and not code_path.is_symlink()
            and _sha256_file(code_path) == training.get("candidate_code_sha256")
            and code_path.stat().st_mode & 0o222 == 0
        ),
        "submission_frozen": (
            submission_path.is_file()
            and not submission_path.is_symlink()
            and _sha256_file(submission_path)
            == training.get("frozen_submission_sha256")
            and submission_path.stat().st_mode & 0o222 == 0
        ),
        "code_execution_valid": (
            collector["report"].get("status") == "pass"
            and collector["manifest"]["execution"].get("exit_status") == 0
        ),
        "runtime_journal_bound": training.get("collector_manifest_hash")
        == collector["manifest"]["manifest_hash"]
        and training.get("collector_report_hash")
        == collector["report"]["report_hash"],
        "collector_code_bound": training.get("candidate_code_sha256")
        == collector["manifest"]["code_sha256"],
        "split_lineage_complete": "split_lineage"
        in collector["manifest"]["observed_events"],
        "fit_scope_complete": "fit_scope"
        in collector["manifest"]["observed_events"],
        "prediction_scope_complete": "prediction_scope"
        in collector["manifest"]["observed_events"],
        "internal_evaluator_complete": "evaluator"
        in collector["manifest"]["observed_events"],
        "selection_frozen": (
            training.get("selection_frozen") is True
            and "selection_freeze" in collector["manifest"]["observed_events"]
        ),
        "terminal_not_exposed": (
            training.get("terminal_exposure_count") == 0
            and training.get("terminal_score_observed") is False
            and collector["report"].get("terminal_exposure_count") == 0
        ),
        "training_manifest_complete": training.get("training_complete") is True,
    }
    receipts = bridge_signed_journal_to_receipts(collector_root, contract=contract)
    authority_closure = dry_evidence_closure(
        contract,
        registry,
        receipts,
        run_id=str(collector["manifest"]["run_id"]),
        node_id=str(collector["manifest"]["node_id"]),
        code_sha256=str(collector["manifest"]["code_sha256"]),
    )
    checks["authority_obligations_closed"] = authority_closure["status"] == "pass"
    missing = sorted(name for name, value in checks.items() if not value)
    report = {
        "schema": "mlevolve_preterminal_evidence_closure_v1",
        "status": "pass" if not missing else "blocked",
        "contract_hash": contract.contract_hash,
        "protocol_hash": contract.protocol_ref.canonical_hash,
        "data_view_manifest_hash": view_manifest.manifest_hash,
        "training_evidence_manifest_hash": training["manifest_hash"],
        "collector_manifest_hash": collector["manifest"]["manifest_hash"],
        "authority_closure_hash": authority_closure["closure_hash"],
        "checks": checks,
        "missing_obligations": missing,
        "terminal_exposure_count": 0,
        "terminal_score_observed": False,
        "evaluator_launch_authorized": not missing,
        "result_fact_created": False,
        "report_hash": "",
    }
    report["report_hash"] = hash_payload(report, "report_hash")
    _write_exclusive(Path(output_path).resolve(), report)
    return report


__all__ = [
    "build_training_evidence_manifest",
    "dry_evidence_closure",
    "preterminal_evidence_closure",
]
