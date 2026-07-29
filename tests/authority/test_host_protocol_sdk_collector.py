from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from authority.authority_engine import AuthorityEngine
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    Operation,
    TaskContext,
)
from authority.protocol_execution_contract import compile_protocol_execution_contract
from authority.protocol_registry import ProtocolRegistry
from protocol_runtime.adapters import boosting, sklearn as sklearn_adapter, torch as torch_adapter
from protocol_runtime.collector import (
    HostCollectorIdentity,
    HostCollectorSidecar,
    verify_collector_artifacts,
)
from protocol_runtime.collector_bridge import bridge_signed_journal_to_receipts
from protocol_runtime.data_views import materialize_data_views
from protocol_runtime.errors import CollectorRejected, CollectorUnavailable, ProtocolStateError
from protocol_runtime.session import ProtocolSession, activate_session, current_session
from protocol_runtime.views import build_view_handles


REGISTRY = ProtocolRegistry("mlevolve/config/protocols")
BUDGET = {"max_epochs": 1, "max_folds": 1, "timeout_seconds": 60}
CODE_HASH = hashlib.sha256(b"reference-candidate-code").hexdigest()
RUN_HASH = hashlib.sha256(b"reference-run").hexdigest()


def _contract(protocol: str, task: str, family: str, identity: HostCollectorIdentity):
    return compile_protocol_execution_contract(
        REGISTRY.resolve(protocol),
        task_id=task,
        task_family=family,
        train_view_ref=f"view://{task}/train",
        validation_view_ref=f"view://{task}/internal-validation",
        terminal_view_ref=f"evaluator-only://{task}/terminal",
        execution_budget=BUDGET,
        collector_spec=identity.collector_spec(),
    )


def _cactus():
    return [
        {"sample_id": f"c-{label}-{index}", "label": label, "x": index}
        for label in (0, 1)
        for index in range(8)
    ]


def _birds():
    return [
        {
            "sample_id": f"b-{group}-{index}",
            "group_id": group,
            "label": [index % 2, (index + 1) % 2],
            "x": index,
        }
        for group in ("a", "b", "c", "d")
        for index in range(4)
    ]


def _taxi():
    return [
        {"sample_id": f"t-{index}", "timestamp": index, "fare": index * 2.0, "x": index}
        for index in range(16)
    ]


class FakeBoostingRegressor:
    __module__ = "xgboost.sklearn"

    def fit(self, features, labels):
        self.value = sum(labels) / len(labels)
        return self

    def predict(self, features):
        return [self.value for _ in features]


def _setup(tmp_path: Path, protocol: str, task: str, family: str, records):
    identity = HostCollectorIdentity.generate()
    contract = _contract(protocol, task, family, identity)
    _manifest, manifest_path = materialize_data_views(
        records, tmp_path / "views", contract, split_id=f"{task}-split", seed="13"
    )
    sidecar = HostCollectorSidecar(
        tmp_path / "collector",
        contract.as_dict(),
        run_id=f"run-{task}",
        node_id=f"node-{task}",
        code_sha256=CODE_HASH,
        identity=identity,
    ).start()
    split = build_view_handles(manifest_path, contract, sidecar)
    session = ProtocolSession(contract, split, sidecar.client())
    return contract, sidecar, split, session


def _seal_and_bridge(contract, sidecar):
    report = sidecar.seal(
        exit_status=0, executed_path="candidate.py", run_hash=RUN_HASH
    )
    assert report["status"] == "pass"
    verified = verify_collector_artifacts(
        sidecar.output_dir,
        expected_public_key_ed25519=contract.collector_spec["public_key_ed25519"],
    )
    receipts = bridge_signed_journal_to_receipts(
        sidecar.output_dir, contract=contract
    )
    return verified, receipts


def _authority_allows(contract, receipts, task: str) -> None:
    claim = Claim(
        claim_id=f"claim-{task}",
        claim_type=ClaimType.SCORE,
        subject_artifact_id=f"node-{task}",
        task_scope={"task_id": task},
        method_fingerprint=CODE_HASH,
        protocol_ref=contract.protocol_ref,
        statement="Host SDK reference score",
    )
    graph = EvidenceGraph()
    graph.add_claim(claim)
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath(f"path-{task}", claim.claim_id, [r.receipt_id for r in receipts]))
    decision = AuthorityEngine(REGISTRY, graph=graph).authorize(
        AuthorityRequest(
            artifact_id=claim.subject_artifact_id,
            claim_id=claim.claim_id,
            operation=Operation.PROMOTE_RESULT,
            decision_stage=DecisionStage.MEMORY_WRITEBACK,
            active_protocol=contract.protocol_ref,
            task_context=TaskContext(task),
            requesting_component="test.host_protocol_sdk",
        )
    )
    assert decision.outcome == DecisionOutcome.ALLOW


def test_sklearn_managed_reference_has_complete_trusted_evidence(tmp_path: Path) -> None:
    from sklearn.tree import DecisionTreeClassifier

    contract, sidecar, split, session = _setup(
        tmp_path, "random-classification@1", "cactus", "image", _cactus()
    )
    try:
        with activate_session(session):
            assert current_session() is session
            views = session.get_split()
            model = sklearn_adapter.fit(
                session,
                DecisionTreeClassifier(max_depth=2, random_state=1),
                views.train,
                feature_keys=("x",),
                label_key="label",
            )
            predictions = sklearn_adapter.predict(
                session, model, views.validation, feature_keys=("x",)
            )
            session.evaluate_internal(
                views.validation, predictions, label_key="label"
            )
            session.freeze_selection(
                "sklearn-model", based_on=views.validation, artifact_hash="1" * 64
            )
        verified, receipts = _seal_and_bridge(contract, sidecar)
        assert [event["kind"] for event in verified["events"]] == [
            "split_lineage",
            "fit_scope",
            "prediction_scope",
            "evaluator",
            "selection_freeze",
        ]
        assert len(receipts) == 7
        assert all(receipt.trust_status == "trusted_host" for receipt in receipts)
        assert all(
            receipt.payload["execution_contract_hash"] == contract.contract_hash
            for receipt in receipts
        )
        _authority_allows(contract, receipts, "cactus")
    finally:
        sidecar.stop()


def test_boosting_managed_reference_has_complete_trusted_evidence(tmp_path: Path) -> None:
    contract, sidecar, split, session = _setup(
        tmp_path, "chronological-regression@1", "taxi", "tabular", _taxi()
    )
    try:
        views = session.get_split()
        model = boosting.fit(
            session,
            FakeBoostingRegressor(),
            views.train,
            feature_keys=("x",),
            label_key="fare",
        )
        predictions = boosting.predict(
            session, model, views.validation, feature_keys=("x",)
        )
        session.evaluate_internal(views.validation, predictions, label_key="fare")
        session.freeze_selection(
            "boosting-model", based_on=views.validation, artifact_hash="2" * 64
        )
        _verified, receipts = _seal_and_bridge(contract, sidecar)
        _authority_allows(contract, receipts, "taxi")
    finally:
        sidecar.stop()


def test_torch_custom_loop_scopes_have_complete_trusted_evidence(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    contract, sidecar, split, session = _setup(
        tmp_path, "grouped-classification@1", "birds", "audio", _birds()
    )
    try:
        views = session.get_split()
        with torch_adapter.fit_scope(
            session, component="torch_model", data_view=views.train
        ) as train_rows:
            weight = torch.tensor([float(len(train_rows))], requires_grad=True)
            (weight * 0.0).sum().backward()
        with torch_adapter.prediction_scope(
            session, component="torch_model", data_view=views.validation
        ) as validation_rows:
            predictions = [row["label"] for row in validation_rows]
        session.evaluate_internal(
            views.validation, predictions, label_key="label"
        )
        session.freeze_selection(
            "torch-checkpoint", based_on=views.validation, artifact_hash="3" * 64
        )
        _verified, receipts = _seal_and_bridge(contract, sidecar)
        _authority_allows(contract, receipts, "birds")
    finally:
        sidecar.stop()


def test_forged_capability_wrong_view_and_nonce_replay_are_rejected(tmp_path: Path) -> None:
    contract, sidecar, split, session = _setup(
        tmp_path, "random-classification@1", "cactus", "image", _cactus()
    )
    client = sidecar.client()
    try:
        forged = split.train._capability[:-1] + (
            "A" if split.train._capability[-1] != "A" else "B"
        )
        with pytest.raises(CollectorRejected, match="capability"):
            client.emit(
                "split_lineage",
                capabilities=(forged, split.validation._capability),
                component="forged",
            )
        nonce = "n" * 32
        client.emit(
            "split_lineage",
            capabilities=(split.train._capability, split.validation._capability),
            component="raw-but-bound",
            nonce=nonce,
        )
        with pytest.raises(CollectorRejected, match="replayed"):
            client.emit(
                "split_lineage",
                capabilities=(split.train._capability, split.validation._capability),
                component="replay",
                nonce=nonce,
            )
        with pytest.raises(CollectorRejected, match="does not permit|view roles"):
            client.emit(
                "fit_scope",
                capabilities=(split.validation._capability,),
                component="wrong-view",
                payload={"component": "model"},
            )
        with pytest.raises(CollectorRejected, match="out of protocol order"):
            client.emit(
                "prediction_scope",
                capabilities=(split.validation._capability,),
                component="bypass-fit",
            )
        assert not hasattr(client, "collect")
        assert not hasattr(client, "mint_receipt")
    finally:
        sidecar.stop()


def test_self_signed_collector_identity_is_not_accepted_by_host_contract(
    tmp_path: Path,
) -> None:
    trusted_identity = HostCollectorIdentity.generate()
    attacker_identity = HostCollectorIdentity.generate()
    contract = _contract(
        "random-classification@1", "cactus", "image", trusted_identity
    )
    with pytest.raises(ValueError, match="not bound"):
        HostCollectorSidecar(
            tmp_path / "collector",
            contract.as_dict(),
            run_id="run-cactus",
            node_id="node-cactus",
            code_sha256=CODE_HASH,
            identity=attacker_identity,
        )


def test_scope_failure_does_not_create_positive_fit_event(tmp_path: Path) -> None:
    contract, sidecar, split, session = _setup(
        tmp_path, "random-classification@1", "cactus", "image", _cactus()
    )
    try:
        with pytest.raises(RuntimeError, match="training crashed"):
            with session.fit_scope(component="model", data_view=split.train):
                raise RuntimeError("training crashed")
        report = sidecar.seal(
            exit_status=1, executed_path="candidate.py", run_hash=RUN_HASH
        )
        assert report["status"] == "blocked"
        assert "fit_scope" in report["missing_events"]
        with pytest.raises(ValueError, match="Incomplete"):
            bridge_signed_journal_to_receipts(
                sidecar.output_dir, contract=contract
            )
    finally:
        sidecar.stop()


def test_journal_or_signature_tamper_is_rejected(tmp_path: Path) -> None:
    contract, sidecar, split, session = _setup(
        tmp_path, "random-classification@1", "cactus", "image", _cactus()
    )
    try:
        views = session.get_split()
        from sklearn.tree import DecisionTreeClassifier

        model = sklearn_adapter.fit(
            session,
            DecisionTreeClassifier(max_depth=1),
            views.train,
            feature_keys=("x",),
            label_key="label",
        )
        pred = sklearn_adapter.predict(
            session, model, views.validation, feature_keys=("x",)
        )
        session.evaluate_internal(views.validation, pred, label_key="label")
        session.freeze_selection("m", based_on=views.validation, artifact_hash="4" * 64)
        _seal_and_bridge(contract, sidecar)
    finally:
        sidecar.stop()
    signature_copy = tmp_path / "signature-copy"
    shutil.copytree(sidecar.output_dir, signature_copy)
    manifest_path = signature_copy / "RUNTIME_EVENT_JOURNAL_MANIFEST.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    signature = manifest["signature_ed25519"]
    manifest["signature_ed25519"] = signature[:-1] + (
        "A" if signature[-1] != "A" else "B"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_collector_artifacts(
            signature_copy,
            expected_public_key_ed25519=contract.collector_spec[
                "public_key_ed25519"
            ],
        )
    journal = sidecar.output_dir / "RUNTIME_EVENT_JOURNAL.jsonl"
    journal.chmod(0o644)
    journal.write_text(journal.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="journal file hash"):
        verify_collector_artifacts(
            sidecar.output_dir,
            expected_public_key_ed25519=contract.collector_spec[
                "public_key_ed25519"
            ],
        )


def test_collector_crash_cannot_leave_trusted_artifacts(tmp_path: Path) -> None:
    contract, sidecar, split, session = _setup(
        tmp_path, "random-classification@1", "cactus", "image", _cactus()
    )
    client = sidecar.client()
    sidecar.terminate()
    with pytest.raises(CollectorUnavailable):
        client.emit(
            "split_lineage",
            capabilities=(split.train._capability, split.validation._capability),
            component="after-crash",
        )
    assert list((tmp_path / "collector").iterdir()) == []


def test_selection_freeze_blocks_later_training(tmp_path: Path) -> None:
    contract, sidecar, split, session = _setup(
        tmp_path, "random-classification@1", "cactus", "image", _cactus()
    )
    try:
        from sklearn.tree import DecisionTreeClassifier

        views = session.get_split()
        model = sklearn_adapter.fit(
            session,
            DecisionTreeClassifier(max_depth=1),
            views.train,
            feature_keys=("x",),
            label_key="label",
        )
        pred = sklearn_adapter.predict(
            session, model, views.validation, feature_keys=("x",)
        )
        session.evaluate_internal(views.validation, pred, label_key="label")
        session.freeze_selection("m", based_on=views.validation, artifact_hash="5" * 64)
        with pytest.raises(ProtocolStateError, match="frozen"):
            session.fit(
                model,
                views.train,
                feature_keys=("x",),
                label_key="label",
            )
    finally:
        sidecar.stop()
