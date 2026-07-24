from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from authority.protocol_registry import ProtocolRegistry
from engine.executor import Interpreter
from fixed_holdout.common import sha256_file, write_json
from fixed_holdout.formal_runtime import (
    build_selected_runtime_protocol_evidence,
)
from fixed_holdout.writeback import _trusted_terminal_receipts


REGISTRY_ROOT = (
    Path(__file__).resolve().parents[1] / "mlevolve" / "config" / "protocols"
)

RUNTIME_CODE = """
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X = np.arange(80, dtype=float).reshape(40, 2)
y = np.array([0, 1] * 20)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)
model = LogisticRegression().fit(X_train, y_train)
pred = model.predict_proba(X_valid)[:, 1]
print(roc_auc_score(y_valid, pred))
"""


def _payload_hash(payload: dict, field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _receipt(schema: str, payload: dict) -> dict:
    value = {"schema": schema, **payload, "receipt_hash": ""}
    value["receipt_hash"] = _payload_hash(value, "receipt_hash")
    return value


def _runtime_node(tmp_path: Path, protocol_ref: str) -> tuple[dict, dict]:
    cfg = SimpleNamespace(
        start_cpu_id=0,
        cpu_number=1,
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1)
        ),
        evaluation_authority=SimpleNamespace(
            mode="enforce", runtime_protocol_observer_enabled=True
        ),
    )
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir(parents=True)
    result = Interpreter(runtime_root, timeout=30, cfg=cfg).run(
        RUNTIME_CODE, "selected"
    )
    assert result.exc_type is None
    node = {
        "id": "selected",
        "stage": "improve",
        "code": RUNTIME_CODE,
        "exec_time": result.exec_time,
        "is_buggy": False,
        "is_valid": True,
        "method_fingerprint": "",
        "protocol_observation": result.protocol_observation,
    }
    evidence = build_selected_runtime_protocol_evidence(
        node, protocol_ref=protocol_ref
    )
    return node, evidence


def _formal_manifests(
    tmp_path: Path,
    *,
    protocol_ref: str,
    formal_strategy: str,
    metric: str,
    direction: str,
) -> tuple[dict, dict, Path, Path]:
    task_id = "target-task"
    split_id = "immutable-split"
    public_hash = "1" * 64
    holdout_hash = "2" * 64
    split_payload = {
        "task_id": task_id,
        "split_id": split_id,
        "split_version": "test-v1",
        "protocol_ref": protocol_ref,
        "strategy": formal_strategy,
        "terminal_labels_absent_from_train_view": True,
    }
    if formal_strategy == "stratified_random":
        split_payload.update(
            {"overlap_count": 0, "stratification_verified": True}
        )
    elif formal_strategy == "grouped_multilabel_stratified":
        split_payload.update(
            {"record_overlap_count": 0, "group_overlap_count": 0}
        )
    elif formal_strategy == "chronological_deterministic_sha256_sample":
        split_payload.update(
            {
                "train_holdout_key_overlap_count": 0,
                "future_to_past_count": 0,
                "max_train_pickup_datetime": "2013-12-31T23:59:59+00:00",
                "min_holdout_pickup_datetime": "2014-07-01T00:00:00+00:00",
            }
        )
    split_receipt = _receipt(
        "formal_split_lineage_receipt_v1", split_payload
    )
    fit_receipt = _receipt(
        "formal_fit_scope_receipt_v1",
        {
            "task_id": task_id,
            "split_id": split_id,
            "protocol_ref": protocol_ref,
            "fit_scope": "train_view_only",
            "fit_scope_hashes": {"train_view_input": public_hash},
            "holdout_fit_count": 0,
            "verified": True,
        },
    )
    from fixed_holdout import evaluate

    metric_receipt = _receipt(
        "formal_metric_spec_receipt_v1",
        {
            "task_id": task_id,
            "split_id": split_id,
            "protocol_ref": protocol_ref,
            "metric": metric,
            "direction": direction,
            "evaluator_module_sha256": sha256_file(Path(evaluate.__file__)),
            "terminal_only": True,
            "verified": True,
        },
    )
    common = {
        "task_id": task_id,
        "split_id": split_id,
        "protocol_ref": protocol_ref,
        "metric": metric,
        "maximize": direction == "maximize",
        "public_tree_sha256": public_hash,
        "holdout_id_sha256": holdout_hash,
        "split_receipt": split_receipt,
        "fit_scope_receipt": fit_receipt,
        "metric_spec_receipt": metric_receipt,
    }
    train = {**common, "role": "train_view", "hidden_labels_present": False}
    evaluator = {
        **common,
        "role": "evaluator_view",
        "labels_sha256": "3" * 64,
    }
    train_path = tmp_path / "train_manifest.json"
    evaluator_path = tmp_path / "evaluator_manifest.json"
    write_json(train_path, train)
    write_json(evaluator_path, evaluator)
    return train, evaluator, train_path, evaluator_path


def _closure_case(
    tmp_path: Path,
    *,
    protocol_id: str,
    formal_strategy: str,
) -> tuple:
    registry = ProtocolRegistry(REGISTRY_ROOT)
    spec = registry.get(protocol_id, "1")
    ref = spec.ref()
    direction = str(spec.metric_spec["direction"])
    metric = str(spec.metric_spec["name"])
    node, runtime_evidence = _runtime_node(tmp_path, ref.key())
    train, evaluator, train_path, evaluator_path = _formal_manifests(
        tmp_path,
        protocol_ref=ref.key(),
        formal_strategy=formal_strategy,
        metric=metric,
        direction=direction,
    )
    report = {
        "report_schema": "fixed_holdout_terminal_score_report_v3",
        "selected_node_id": node["id"],
        "selected_score": 0.5,
        "selected_submission": "submission_selected.csv",
        "selected_submission_sha256": "4" * 64,
        "candidate_set_hash": "5" * 64,
        "report_hash": "6" * 64,
        "metric": metric,
        "maximize": direction == "maximize",
    }
    binding = {
        "condition": "full_decision_admissibility",
        "training_manifest_sha256": "7" * 64,
        "training_manifest_hash": "8" * 64,
        "runtime_evidence_hash": runtime_evidence["evidence_hash"],
        "runtime_observation_sha256": runtime_evidence[
            "observation_sha256"
        ],
    }
    receipts, code_hash = _trusted_terminal_receipts(
        host=TrustedCollectorHost(f"test:{protocol_id}"),
        protocol_ref=ref,
        protocol_spec=spec,
        run_id=f"run:{protocol_id}",
        node=node,
        report=report,
        train_manifest=train,
        evaluator_manifest=evaluator,
        journal_hash="9" * 64,
        train_manifest_path=train_path,
        evaluator_manifest_path=evaluator_path,
        runtime_evidence=runtime_evidence,
        training_binding=binding,
    )
    return registry, spec, node, report, receipts, code_hash


@pytest.mark.parametrize(
    ("protocol_id", "formal_strategy", "expected_split_flag"),
    [
        (
            "random-classification",
            "stratified_random",
            ("stratification_verified", True),
        ),
        (
            "grouped-classification",
            "grouped_multilabel_stratified",
            ("group_overlap_count", 0),
        ),
        (
            "chronological-regression",
            "chronological_deterministic_sha256_sample",
            ("chronological_order_verified", True),
        ),
    ],
)
def test_three_formal_protocols_close_terminal_payload_obligations(
    tmp_path: Path,
    protocol_id: str,
    formal_strategy: str,
    expected_split_flag: tuple[str, object],
) -> None:
    registry, spec, node, _report, receipts, code_hash = _closure_case(
        tmp_path,
        protocol_id=protocol_id,
        formal_strategy=formal_strategy,
    )
    by_type = {receipt.receipt_type: receipt for receipt in receipts}
    split = by_type[ReceiptType.SPLIT_LINEAGE].payload
    fit = by_type[ReceiptType.FIT_SCOPE].payload
    evaluator = by_type[ReceiptType.EVALUATOR].payload
    assert split["split_strategy"] == spec.data_split_policy["strategy"]
    assert split[expected_split_flag[0]] == expected_split_flag[1]
    assert fit["fit_scope"] == spec.preprocessing_policy["fit_scope"]
    assert fit["terminal_fit_scope"] == "train_view_only"
    assert fit["fit_scope"] != fit["terminal_fit_scope"]
    assert evaluator["metric_name"] == spec.metric_spec["name"]
    assert evaluator["metric_direction"] == spec.metric_spec["direction"]
    for receipt_type in (
        ReceiptType.SPLIT_LINEAGE,
        ReceiptType.FIT_SCOPE,
        ReceiptType.PREDICTION_SCOPE,
        ReceiptType.EVALUATOR,
        ReceiptType.SELECTION_FREEZE,
    ):
        evidence = by_type[receipt_type].payload[
            "terminal_protocol_evidence"
        ]
        assert evidence["internal_runtime_verified"] is True
        assert evidence["terminal_manifest_verified"] is True
        assert evidence["receipt_semantics_separated"] is True

    claim = Claim(
        claim_id=f"claim:{protocol_id}",
        claim_type=ClaimType.SCORE,
        subject_artifact_id=node["id"],
        task_scope={"task_id": "target-task"},
        method_fingerprint=code_hash,
        protocol_ref=spec.ref(),
        statement="sealed terminal score",
    )
    graph = EvidenceGraph()
    graph.add_claim(claim)
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(
        EvidencePath(
            path_id=f"path:{protocol_id}",
            claim_id=claim.claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipts],
        )
    )
    decision = AuthorityEngine(registry, graph=graph).authorize(
        AuthorityRequest(
            artifact_id=node["id"],
            claim_id=claim.claim_id,
            operation=Operation.PROMOTE_RESULT,
            decision_stage=DecisionStage.MEMORY_WRITEBACK,
            active_protocol=spec.ref(),
            task_context=TaskContext(task_id="target-task"),
            requesting_component="test.formal_terminal_payload_closure",
        )
    )
    assert decision.allowed is True
    assert decision.missing_obligations == []


def test_enforced_terminal_payloads_fail_closed_without_runtime_chain(
    tmp_path: Path,
) -> None:
    registry = ProtocolRegistry(REGISTRY_ROOT)
    spec = registry.get("random-classification", "1")
    ref = spec.ref()
    node, _runtime_evidence = _runtime_node(tmp_path, ref.key())
    train, evaluator, train_path, evaluator_path = _formal_manifests(
        tmp_path,
        protocol_ref=ref.key(),
        formal_strategy="stratified_random",
        metric="macro_f1",
        direction="maximize",
    )
    report = {
        "report_schema": "fixed_holdout_terminal_score_report_v3",
        "selected_node_id": node["id"],
        "selected_score": 0.5,
        "selected_submission": "submission_selected.csv",
        "selected_submission_sha256": "4" * 64,
        "candidate_set_hash": "5" * 64,
        "report_hash": "6" * 64,
        "metric": "macro_f1",
        "maximize": True,
    }
    with pytest.raises(ValueError, match="lacks formal runtime evidence"):
        _trusted_terminal_receipts(
            host=TrustedCollectorHost("test:missing-runtime"),
            protocol_ref=ref,
            protocol_spec=spec,
            run_id="run:missing-runtime",
            node=node,
            report=report,
            train_manifest=train,
            evaluator_manifest=evaluator,
            journal_hash="9" * 64,
            train_manifest_path=train_path,
            evaluator_manifest_path=evaluator_path,
            runtime_evidence={},
            training_binding={},
        )


def test_terminal_train_view_scope_cannot_be_relabelled_as_internal_scope(
    tmp_path: Path,
) -> None:
    registry = ProtocolRegistry(REGISTRY_ROOT)
    spec = registry.get("random-classification", "1")
    ref = spec.ref()
    node, runtime_evidence = _runtime_node(tmp_path, ref.key())
    train, evaluator, train_path, evaluator_path = _formal_manifests(
        tmp_path,
        protocol_ref=ref.key(),
        formal_strategy="stratified_random",
        metric="macro_f1",
        direction="maximize",
    )
    for manifest in (train, evaluator):
        manifest["fit_scope_receipt"]["fit_scope"] = (
            spec.preprocessing_policy["fit_scope"]
        )
        manifest["fit_scope_receipt"]["receipt_hash"] = _payload_hash(
            manifest["fit_scope_receipt"], "receipt_hash"
        )
    write_json(train_path, train)
    write_json(evaluator_path, evaluator)
    report = {
        "report_schema": "fixed_holdout_terminal_score_report_v3",
        "selected_node_id": node["id"],
        "selected_score": 0.5,
        "selected_submission": "submission_selected.csv",
        "selected_submission_sha256": "4" * 64,
        "candidate_set_hash": "5" * 64,
        "report_hash": "6" * 64,
        "metric": "macro_f1",
        "maximize": True,
    }
    binding = {
        "condition": "full_decision_admissibility",
        "training_manifest_sha256": "7" * 64,
        "training_manifest_hash": "8" * 64,
        "runtime_evidence_hash": runtime_evidence["evidence_hash"],
        "runtime_observation_sha256": runtime_evidence[
            "observation_sha256"
        ],
    }
    with pytest.raises(ValueError, match="terminal fit-scope Receipt"):
        _trusted_terminal_receipts(
            host=TrustedCollectorHost("test:relabelled-fit"),
            protocol_ref=ref,
            protocol_spec=spec,
            run_id="run:relabelled-fit",
            node=node,
            report=report,
            train_manifest=train,
            evaluator_manifest=evaluator,
            journal_hash="9" * 64,
            train_manifest_path=train_path,
            evaluator_manifest_path=evaluator_path,
            runtime_evidence=runtime_evidence,
            training_binding=binding,
        )
