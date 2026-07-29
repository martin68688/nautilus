from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

from authority.adapters.mlevolve.receipt_bridge import receipts_for_node
from authority.authority_engine import AuthorityEngine
from authority.collectors import TrustedCollectorHost
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionStage,
    Operation,
    ReceiptType,
    TaskContext,
)
import pytest

from authority.protocol_execution_contract import (
    compile_protocol_execution_contract,
    write_contract_artifacts,
)
from authority.protocol_registry import ProtocolRegistry
from protocol_runtime.collector import HostCollectorIdentity
from protocol_runtime.data_views import materialize_data_views
from protocol_runtime.full_runtime import (
    FullRuntimeEvidenceController,
    activate_full_runtime_from_bootstrap,
    deactivate_full_runtime,
)


def test_host_full_runtime_validation_source_is_parseable_and_complete():
    import ast

    from agents.prompts.impl_guideline import _host_full_runtime_validation_source

    source = _host_full_runtime_validation_source(
        {
            "task_id": "denoising-dirty-documents",
            "label_key": "target",
            "metric_name": "rmse",
            "inference_view_required": True,
        }
    )
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    calls = {
        getattr(node.func, "attr", getattr(node.func, "id", ""))
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
    }
    assert {
        "current_session",
        "get_split",
        "fit_scope",
        "prediction_scope",
        "evaluate_internal",
        "freeze_selection",
        "inference_scope",
    } <= calls


REGISTRY = ProtocolRegistry("mlevolve/config/protocols")


def test_full_training_session_mints_score_bound_trusted_receipts(tmp_path: Path):
    identity = HostCollectorIdentity.generate()
    contract = compile_protocol_execution_contract(
        REGISTRY.resolve("deterministic-random-regression@1"),
        task_id="new-york-city-taxi-fare-prediction",
        task_family="tabular",
        train_view_ref="view://taxi/train",
        validation_view_ref="view://taxi/internal-validation",
        terminal_view_ref="evaluator-only://taxi/terminal",
        execution_budget={
            "max_epochs": 1,
            "max_folds": 1,
            "max_models": 1,
            "timeout_seconds": 60,
        },
        collector_spec=identity.collector_spec(),
    )
    records = [
        {
            "sample_id": f"row-{index}",
            "fare": float(index + 1),
            "x": float(index),
        }
        for index in range(12)
    ]
    _manifest, manifest_path = materialize_data_views(
        records,
        tmp_path / "views",
        contract,
        inference_records=[
            {"sample_id": "test-row", "x": 99.0},
        ],
        inference_view_ref="view://taxi/inference",
        split_id="full-runtime-test",
    )
    source = "def main():\n    pass\n"
    code_sha256 = hashlib.sha256(source.encode()).hexdigest()
    controller = FullRuntimeEvidenceController(
        contract=contract,
        identity=identity,
        data_view_manifest_path=manifest_path,
        output_root=tmp_path / "runtime",
        bootstrap_path=tmp_path / "working" / "bootstrap.json",
        run_id="full-runtime-test-run",
        node_id="node-a",
        code_sha256=code_sha256,
    ).start()
    try:
        session = activate_full_runtime_from_bootstrap(controller.bootstrap_path)
        views = session.get_split()
        with session.fit_scope(
            component="mean_regressor", data_view=views.train
        ) as train_rows:
            fitted = sum(row["fare"] for row in train_rows) / len(train_rows)
        with session.prediction_scope(
            component="mean_regressor", data_view=views.validation
        ) as validation_rows:
            predictions = [fitted for _ in validation_rows]
        score = session.evaluate_internal(
            views.validation, predictions, label_key="fare"
        )
        checkpoint = tmp_path / "mean-regressor.bin"
        checkpoint.write_bytes(b"frozen-mean-regressor")
        session.freeze_selection(
            "mean-regressor",
            based_on=views.validation,
            artifact_hash=str(checkpoint),
        )
        with session.inference_scope(
            component="final_submission", data_view=views.inference
        ) as inference_rows:
            assert [row["sample_id"] for row in inference_rows] == ["test-row"]
        evidence = controller.seal(
            exit_status=0,
            executed_path="candidate.py",
            run_hash="b" * 64,
        )
    finally:
        deactivate_full_runtime()
        controller.stop()

    assert evidence["status"] == "pass"
    node = SimpleNamespace(
        id="node-a",
        code=source,
        code_sha256_expected=code_sha256,
        method_fingerprint=code_sha256,
        metric=SimpleNamespace(value=score, maximize=False),
        exec_time=1.0,
        is_buggy=False,
        is_valid=True,
        protocol_observation={"host_full_runtime": evidence},
        leakage_audit={},
        protocol_repair={},
    )
    receipts = receipts_for_node(
        node,
        contract.protocol_ref,
        "authority-run",
        collector_host=TrustedCollectorHost("test-authority"),
        task_id=contract.task_id,
    )
    receipt_types = {receipt.receipt_type for receipt in receipts}
    assert {
        ReceiptType.CODE_EXECUTION,
        ReceiptType.METHOD_IDENTITY,
        ReceiptType.SPLIT_LINEAGE,
        ReceiptType.FIT_SCOPE,
        ReceiptType.PREDICTION_SCOPE,
        ReceiptType.EVALUATOR,
        ReceiptType.SELECTION_FREEZE,
    } <= receipt_types
    evaluator = next(
        receipt for receipt in receipts if receipt.receipt_type == ReceiptType.EVALUATOR
    )
    assert evaluator.payload["metric_value"] == score
    graph = EvidenceGraph()
    claim = Claim(
        claim_id="node-a:score",
        claim_type=ClaimType.SCORE,
        subject_artifact_id="node-a",
        task_scope={"task_id": contract.task_id},
        method_fingerprint=code_sha256,
        protocol_ref=contract.protocol_ref,
        statement="Host full-runtime internal score",
        source_artifact_refs=["node-a"],
        evidence_refs=[receipt.receipt_id for receipt in receipts],
    )
    graph.add_claim(claim)
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id="node-a:path",
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipts],
        )
    )
    decision = AuthorityEngine(REGISTRY, graph=graph).authorize(
        AuthorityRequest(
            artifact_id="node-a",
            claim_id=claim.claim_id,
            operation=Operation.PROMOTE_RESULT,
            decision_stage=DecisionStage.MEMORY_WRITEBACK,
            active_protocol=contract.protocol_ref,
            task_context=TaskContext(contract.task_id, contract.task_family),
            requesting_component="test.full_runtime",
        )
    )
    assert decision.allowed is True

    node.metric.value = score + 1.0
    mismatched = receipts_for_node(
        node,
        contract.protocol_ref,
        "authority-run",
        collector_host=TrustedCollectorHost("test-authority-mismatch"),
        task_id=contract.task_id,
    )
    assert ReceiptType.EVALUATOR not in {
        receipt.receipt_type for receipt in mismatched
    }


@pytest.mark.skipif(os.geteuid() != 0, reason="enforce UID isolation requires root")
def test_executor_runs_full_training_under_host_session_and_seals_evidence():
    from engine.executor import Interpreter

    root = Path(tempfile.mkdtemp(prefix="mlevolve-full-runtime-"))
    root.chmod(0o755)
    try:
        identity = HostCollectorIdentity.generate()
        key_path = root / "secrets" / "collector.ed25519"
        identity.write_private_key_file(key_path)
        contract = compile_protocol_execution_contract(
            REGISTRY.resolve("deterministic-random-regression@1"),
            task_id="new-york-city-taxi-fare-prediction",
            task_family="tabular",
            train_view_ref="view://taxi/train",
            validation_view_ref="view://taxi/internal-validation",
            terminal_view_ref="evaluator-only://taxi/terminal",
            execution_budget={
                "max_epochs": 1,
                "max_folds": 1,
                "max_models": 1,
                "timeout_seconds": 60,
            },
            collector_spec=identity.collector_spec(),
        )
        contract_path, _sha_path = write_contract_artifacts(
            contract, root / "contract"
        )
        _manifest, manifest_path = materialize_data_views(
            [
                {
                    "sample_id": f"row-{index}",
                    "fare": float(index + 1),
                }
                for index in range(12)
            ],
            root / "views",
            contract,
            split_id="executor-full-runtime",
        )
        source = '''def candidate(session):
    views = session.get_split()
    with session.fit_scope(component="mean", data_view=views.train) as train_rows:
        fitted = sum(row["fare"] for row in train_rows) / len(train_rows)
    with session.prediction_scope(component="mean", data_view=views.validation) as validation_rows:
        predictions = [fitted for _ in validation_rows]
    session.evaluate_internal(views.validation, predictions, label_key="fare")
    session.freeze_selection("dry", based_on=views.validation, artifact_hash="a" * 64)

from protocol_runtime import current_session

def main():
    session = current_session()
    views = session.get_split()
    with session.fit_scope(component="mean", data_view=views.train) as train_rows:
        fitted = sum(row["fare"] for row in train_rows) / len(train_rows)
    with session.prediction_scope(component="mean", data_view=views.validation) as validation_rows:
        predictions = [fitted for _ in validation_rows]
    score = session.evaluate_internal(views.validation, predictions, label_key="fare")
    session.freeze_selection("full", based_on=views.validation, artifact_hash="b" * 64)
    print(f"Final Validation RMSE: {score}")

if __name__ == "__main__":
    main()
'''
        cfg = SimpleNamespace(
            agent=SimpleNamespace(
                search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
                protocol_preflight=SimpleNamespace(
                    enabled=True,
                    report_root=str(root / "reports"),
                    expected_contract_hash=contract.contract_hash,
                    contract_path=str(contract_path),
                    data_view_manifest_path=str(manifest_path),
                    image_digest="sha256:test-image",
                    sdk_hash="c" * 64,
                    collector_private_key_path=str(key_path),
                    candidate_uid=65534,
                    consume_collector_private_key=False,
                ),
            ),
            evaluation_authority=SimpleNamespace(
                mode="enforce",
                protocol_runtime_mode="host_sdk_enforce",
                runtime_protocol_observer_enabled=True,
                protocol_registry="config/protocols",
            ),
            start_cpu_id="0",
            cpu_number="1",
        )
        workspace = root / "workspace"
        workspace.mkdir()
        result = Interpreter(
            workspace, timeout=60, max_parallel_run=1, cfg=cfg
        ).run(source, id="full-runtime-node")
        assert result.exc_type is None, result.term_out
        assert "Final Validation RMSE:" in "".join(result.term_out)
        full = result.protocol_observation["host_full_runtime"]
        assert full["status"] == "pass"
        assert full["missing_events"] == []
        assert Path(full["collector_root"]).is_dir()
    finally:
        shutil.rmtree(root, ignore_errors=True)
