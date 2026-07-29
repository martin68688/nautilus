from __future__ import annotations

import hashlib
import json
from pathlib import Path

from authority.authority_engine import AuthorityEngine
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    AuthorityReasonCode,
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    Operation,
    ProtocolSpec,
    ReceiptType,
    TaskContext,
)
from authority.protocol_execution_contract import compile_protocol_execution_contract
from authority.protocol_registry import ProtocolRegistry
from authority.receipt_collectors import make_receipt
from fixed_holdout.writeback import record_terminal_writeback_failure
from protocol_runtime.adapters import sklearn as sklearn_adapter
from protocol_runtime.closure import (
    build_training_evidence_manifest,
    preterminal_evidence_closure,
)
from protocol_runtime.collector import HostCollectorIdentity, HostCollectorSidecar
from protocol_runtime.data_views import (
    build_evaluator_launch_contract,
    materialize_data_views,
)
from protocol_runtime.session import ProtocolSession
from protocol_runtime.views import build_view_handles


DIGEST = "a" * 64


def _request(ref, claim_id="claim"):
    return AuthorityRequest(
        artifact_id="artifact",
        claim_id=claim_id,
        operation=Operation.PROMOTE_RESULT,
        decision_stage=DecisionStage.MEMORY_WRITEBACK,
        active_protocol=ref,
        task_context=TaskContext("task"),
        requesting_component="test.reason_codes",
    )


def _claim(ref):
    return Claim(
        claim_id="claim",
        claim_type=ClaimType.SCORE,
        subject_artifact_id="artifact",
        task_scope={"task_id": "task"},
        method_fingerprint=DIGEST,
        protocol_ref=ref,
        statement="score",
    )


def test_missing_untrusted_contract_and_violation_have_distinct_outcomes() -> None:
    registry = ProtocolRegistry()
    first = registry.register(ProtocolSpec(protocol_id="p", version="1"))
    second = registry.register(ProtocolSpec(protocol_id="p", version="2"))

    graph = EvidenceGraph()
    graph.add_claim(_claim(first.ref()))
    graph.add_path(EvidencePath("path", "claim", []))
    missing = AuthorityEngine(registry, graph=graph).authorize(_request(first.ref()))
    assert missing.outcome == DecisionOutcome.REQUIRE_REPLAY
    assert missing.reason_codes == [AuthorityReasonCode.MISSING_EVIDENCE.value]
    assert missing.responsible_component == "evidence_collector"
    assert missing.repairable is True
    assert ReceiptType.CODE_EXECUTION.value in missing.missing_receipts

    untrusted_graph = EvidenceGraph()
    untrusted_graph.add_claim(_claim(first.ref()))
    untrusted = make_receipt(
        ReceiptType.METHOD_IDENTITY,
        "artifact",
        "run",
        first.ref(),
        "candidate.self_report",
        {"verified": True},
    )
    untrusted_graph.add_receipt(untrusted)
    untrusted_graph.add_path(EvidencePath("path", "claim", [untrusted.receipt_id]))
    decision = AuthorityEngine(registry, graph=untrusted_graph).authorize(
        _request(first.ref())
    )
    assert decision.outcome == DecisionOutcome.QUARANTINE
    assert decision.reason_codes == [AuthorityReasonCode.UNTRUSTED_EVIDENCE.value]

    mismatch = AuthorityEngine(registry, graph=graph).authorize(_request(second.ref()))
    assert mismatch.outcome == DecisionOutcome.REQUIRE_HUMAN_REVIEW
    assert mismatch.reason_codes == [AuthorityReasonCode.CONTRACT_MISMATCH.value]

    blocker_graph = EvidenceGraph()
    blocker_graph.add_claim(_claim(first.ref()))
    blocker = make_receipt(
        ReceiptType.CODE_EXECUTION,
        "artifact",
        "run",
        first.ref(),
        "host.blocker",
        {"contradicts": True},
    )
    blocker_graph.add_receipt(blocker)
    blocker_graph.add_path(EvidencePath("path", "claim", [blocker.receipt_id]))
    blocked = AuthorityEngine(registry, graph=blocker_graph).authorize(
        _request(first.ref())
    )
    assert blocked.outcome == DecisionOutcome.DENY
    assert blocked.reason_codes == [AuthorityReasonCode.PROTOCOL_VIOLATION.value]


def test_authority_internal_error_is_quarantine_not_candidate_denial(monkeypatch) -> None:
    registry = ProtocolRegistry()
    spec = registry.register(ProtocolSpec(protocol_id="p", version="1"))
    graph = EvidenceGraph()
    graph.add_claim(_claim(spec.ref()))
    engine = AuthorityEngine(registry, graph=graph)

    def fail(*_args, **_kwargs):
        raise RuntimeError("collector plumbing")

    monkeypatch.setattr(engine.compiler, "compile", fail)
    decision = engine.authorize(_request(spec.ref()))
    assert decision.outcome == DecisionOutcome.QUARANTINE
    assert decision.reason_codes == [
        AuthorityReasonCode.COLLECTOR_INTERNAL_ERROR.value
    ]
    assert decision.responsible_component == "authority_engine"


def _attestation(manifest, closure):
    value = {
        "schema": "mlevolve_training_pod_deletion_attestation_v2",
        "not_found_verified": True,
        "kubernetes_reason": "NotFound",
        "contract_hash": manifest.contract_hash,
        "data_view_manifest_hash": manifest.manifest_hash,
        "verified_by": "host_launcher",
        "terminal_metric_observed_before_not_found": False,
        "preterminal_closure_report_hash": closure["report_hash"],
        "attestation_hash": "",
    }
    value["attestation_hash"] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "attestation_hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return value


def test_preterminal_closure_gates_evaluator_launch_and_binds_frozen_code(
    tmp_path: Path,
) -> None:
    from sklearn.tree import DecisionTreeClassifier

    registry = ProtocolRegistry("mlevolve/config/protocols")
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        registry.resolve("random-classification@1"),
        task_id="preterminal-cactus",
        task_family="image",
        train_view_ref="view://preterminal/train",
        validation_view_ref="view://preterminal/validation",
        terminal_view_ref="evaluator-only://preterminal/terminal",
        execution_budget={"max_epochs": 1, "max_folds": 1, "timeout_seconds": 30},
        collector_spec=identity.collector_spec(),
    )
    records = [
        {"sample_id": f"c-{label}-{index}", "label": label, "x": index}
        for label in (0, 1)
        for index in range(8)
    ]
    manifest, manifest_path = materialize_data_views(
        records, tmp_path / "views", contract, split_id="preterminal-split"
    )
    code = b"# frozen candidate code\n"
    code_path = tmp_path / "candidate.py"
    code_path.write_bytes(code)
    code_hash = hashlib.sha256(code).hexdigest()
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,prediction\n1,0\n", encoding="utf-8")
    sidecar = HostCollectorSidecar(
        tmp_path / "collector",
        contract.as_dict(),
        run_id="full-run",
        node_id="selected-node",
        code_sha256=code_hash,
        identity=identity,
    ).start()
    try:
        views = build_view_handles(manifest_path, contract, sidecar)
        session = ProtocolSession(contract, views, sidecar.client())
        model = sklearn_adapter.fit(
            session,
            DecisionTreeClassifier(max_depth=1, random_state=1),
            views.train,
            feature_keys=("x",),
            label_key="label",
        )
        predictions = sklearn_adapter.predict(
            session, model, views.validation, feature_keys=("x",)
        )
        session.evaluate_internal(views.validation, predictions, label_key="label")
        session.freeze_selection(
            "selected", based_on=views.validation, artifact_hash="b" * 64
        )
        sidecar.seal(
            exit_status=0,
            executed_path="candidate.py",
            run_hash="c" * 64,
        )
    finally:
        sidecar.stop()

    training = build_training_evidence_manifest(
        contract,
        data_view_manifest_path=manifest_path,
        collector_root=tmp_path / "collector",
        candidate_code_path=code_path,
        frozen_submission_path=submission_path,
        output_path=tmp_path / "TRAINING_EVIDENCE_MANIFEST.json",
    )
    closure = preterminal_evidence_closure(
        contract,
        registry,
        data_view_manifest_path=manifest_path,
        collector_root=tmp_path / "collector",
        training_evidence_manifest_path=tmp_path / "TRAINING_EVIDENCE_MANIFEST.json",
        output_path=tmp_path / "PRETERMINAL_EVIDENCE_CLOSURE_REPORT.json",
    )
    assert closure["status"] == "pass"
    assert closure["evaluator_launch_authorized"] is True
    assert all(closure["checks"].values())
    mount = json.loads(
        (tmp_path / "views" / "TRAINING_MOUNT_CONTRACT.json").read_text()
    )
    deletion = _attestation(manifest, closure)
    launch = build_evaluator_launch_contract(manifest, mount, deletion, closure)
    assert launch["preterminal_closure_report_hash"] == closure["report_hash"]

    code_path.chmod(0o644)
    code_path.write_text("# tampered\n", encoding="utf-8")
    blocked = preterminal_evidence_closure(
        contract,
        registry,
        data_view_manifest_path=manifest_path,
        collector_root=tmp_path / "collector",
        training_evidence_manifest_path=tmp_path / "TRAINING_EVIDENCE_MANIFEST.json",
        output_path=tmp_path / "PRETERMINAL_TAMPER_REPORT.json",
    )
    assert blocked["status"] == "blocked"
    assert "candidate_code_frozen" in blocked["missing_obligations"]
    assert training["terminal_score_observed"] is False


def test_terminal_writeback_plumbing_failure_is_infrastructure_quarantine(
    tmp_path: Path,
) -> None:
    status = record_terminal_writeback_failure(
        tmp_path / "writeback_status.json",
        ValueError("collector payload:evaluator schema mismatch"),
    )
    assert status["reason_codes"] == [
        AuthorityReasonCode.COLLECTOR_INTERNAL_ERROR.value
    ]
    assert status["authority_disposition"] == "quarantine"
    assert status["responsible_component"] == "terminal_host_collector"
