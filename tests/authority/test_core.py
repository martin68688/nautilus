from __future__ import annotations

import copy
import json
import multiprocessing
from pathlib import Path
import threading

from authority.authority_engine import AuthorityEngine
from authority.collectors import (
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
)
from authority.derivation_guard import authorize_derivation_operation
from authority.derivation_guard import validate_derivation
from authority.evidence_graph import EvidenceGraph, EvidencePath
from authority.ledger import AuthorityLedger
from authority.models import (
    AuthorityDecision,
    AuthorityRequest,
    AuthorityScope,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    Operation,
    ProtocolSpec,
    ReceiptType,
    TaskContext,
)
from authority.protocol_registry import ProtocolRegistry, protocol_hash
from authority.receipt_collectors import make_receipt
from authority.replay_certifier import ReplayIdentity, certify_replay


def _append_ledger_events(path: str, worker: int, count: int) -> None:
    ledger = AuthorityLedger(path)
    for index in range(count):
        ledger.append("receipt", {"worker": worker, "index": index})


def protocol(version="1", **updates):
    values = dict(
        protocol_id="test",
        version=version,
        task_profile={"family": "text"},
        data_split_policy={"folds": 5},
        preprocessing_policy={"fit": "fold_train"},
        evaluator_spec={"name": "logloss"},
        metric_spec={"maximize": False},
        selection_policy={"freeze": True},
        seed_policy={"pairwise_min_seeds": 3},
        holdout_policy={"terminal_only": True},
        promotion_policy={"complete_path": True},
        compatibility_rules={},
    )
    values.update(updates)
    return ProtocolSpec(**values)


def claim(ref, claim_id="c1"):
    return Claim(
        claim_id=claim_id,
        claim_type=ClaimType.SCORE,
        subject_artifact_id="a1",
        task_scope={"task_id": "task"},
        method_fingerprint="method",
        protocol_ref=ref,
        statement="artifact has a protocol-scoped score",
    )


def request(ref, operation=Operation.RANK):
    return AuthorityRequest(
        artifact_id="a1",
        claim_id="c1",
        operation=operation,
        decision_stage=DecisionStage.BRANCH_SELECTION,
        active_protocol=ref,
        task_context=TaskContext(task_id="task"),
        requesting_component="test",
    )


def required_receipts(ref, path_suffix=""):
    digest = "a" * 64
    host = TrustedCollectorHost(f"test-host{path_suffix}")
    specifications = (
        (MethodIdentityCollector, {
            "method_fingerprint": digest,
            "code_sha256": digest,
        }),
        (CodeExecutionCollector, {
            "exit_status": 0,
            "executed_path": "a1",
            "run_hash": digest,
        }),
        (SplitLineageCollector, {
            "partition_hashes": {"train": digest, "valid": digest},
            "overlap_count": 0,
        }),
        (FitScopeCollector, {
            "fit_scope_hashes": {"model": digest},
            "holdout_fit_count": 0,
        }),
        (PredictionScopeCollector, {
            "prediction_scope_hashes": {"valid": digest},
            "forbidden_overlap_count": 0,
        }),
        (EvaluatorIntegrityCollector, {
            "evaluator_hash": digest,
            "inputs_hash": digest,
            "metric_direction": "minimize",
            "tampered": False,
        }),
        (SelectionFreezeCollector, {
            "candidate_set_hash": digest,
            "frozen_before_holdout": True,
        }),
    )
    return [
        host.collect(
            collector,
            artifact_id="a1",
            run_id="run",
            protocol_ref=ref,
            source="tests.host",
            payload=payload,
        )
        for collector, payload in specifications
    ]


def test_protocol_hash_is_canonical_and_version_immutable():
    first = protocol(data_split_policy={"folds": 5, "shuffle": True})
    second = protocol(data_split_policy={"shuffle": True, "folds": 5})
    assert protocol_hash(first) == protocol_hash(second)
    registry = ProtocolRegistry()
    registered = registry.register(first)
    registry.register(second)
    changed = protocol(data_split_policy={"folds": 10})
    try:
        registry.register(changed)
    except ValueError as exc:
        assert "immutable" in str(exc)
    else:
        raise AssertionError("changed protocol reused an immutable version")
    assert registered.canonical_hash


def test_protocol_registry_ignores_hidden_appledouble_json(tmp_path: Path):
    source = Path("mlevolve/config/protocols/mlevolve-default-v1.json")
    (tmp_path / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / f"._{source.name}").write_bytes(b"\x00\x05\x16\x07AppleDouble")
    registry = ProtocolRegistry(tmp_path)
    refs = registry.refs()
    assert len(refs) == 1
    assert refs[0].protocol_id == "mlevolve-default"


def test_and_or_paths_cannot_combine_two_incomplete_paths():
    registry = ProtocolRegistry()
    spec = registry.register(protocol())
    graph = EvidenceGraph()
    graph.add_claim(claim(spec.ref()))
    receipts = required_receipts(spec.ref())
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath("left", "c1", [item.receipt_id for item in receipts[:3]]))
    graph.add_path(EvidencePath("right", "c1", [item.receipt_id for item in receipts[3:]]))
    engine = AuthorityEngine(registry, graph=graph)
    decision = engine.authorize(request(spec.ref()))
    assert decision.outcome == DecisionOutcome.REQUIRE_REPLAY
    assert decision.reason_codes == ["missing_evidence"]
    graph.add_path(EvidencePath("complete", "c1", [item.receipt_id for item in receipts]))
    decision = engine.authorize(request(spec.ref()))
    assert decision.outcome == DecisionOutcome.ALLOW
    assert decision.satisfied_paths == ["complete"]
    assert decision.generation_stage == "improve"
    assert decision.governance_stage == "branch_selection"
    assert decision.permitted_scope is not None
    assert decision.permitted_scope.generation_stages == ["improve"]
    assert decision.permitted_scope.governance_stages == ["branch_selection"]


def test_request_artifact_and_task_context_must_match_the_claim() -> None:
    registry = ProtocolRegistry()
    spec = registry.register(protocol())
    graph = EvidenceGraph()
    graph.add_claim(claim(spec.ref()))
    receipts = required_receipts(spec.ref())
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath("complete", "c1", [item.receipt_id for item in receipts]))
    engine = AuthorityEngine(registry, graph=graph)

    wrong_artifact = request(spec.ref())
    wrong_artifact.artifact_id = "other-artifact"
    artifact_decision = engine.authorize(wrong_artifact)
    assert artifact_decision.outcome == DecisionOutcome.DENY
    assert "claim_subject_artifact" in artifact_decision.missing_obligations

    wrong_task = request(spec.ref())
    wrong_task.task_context = TaskContext(task_id="other-task")
    task_decision = engine.authorize(wrong_task)
    assert task_decision.outcome == DecisionOutcome.DENY
    assert "claim_task_scope" in task_decision.missing_obligations


def test_contradictory_receipt_cannot_increase_authority():
    registry = ProtocolRegistry()
    spec = registry.register(protocol())
    graph = EvidenceGraph()
    graph.add_claim(claim(spec.ref()))
    receipts = required_receipts(spec.ref())
    contradiction = make_receipt(
        ReceiptType.EVALUATOR,
        "a1",
        "run",
        spec.ref(),
        "counterevidence",
        {"contradicts": True, "reason": "labels observed"},
    )
    for receipt in [*receipts, contradiction]:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath("contaminated", "c1", [item.receipt_id for item in [*receipts, contradiction]]))
    decision = AuthorityEngine(registry, graph=graph).authorize(request(spec.ref()))
    assert decision.outcome == DecisionOutcome.DENY
    assert contradiction.receipt_id in decision.blocking_receipts


def test_contaminated_alternative_path_does_not_poison_clean_support_path():
    registry = ProtocolRegistry()
    spec = registry.register(protocol())
    graph = EvidenceGraph()
    graph.add_claim(claim(spec.ref()))
    receipts = required_receipts(spec.ref())
    contradiction = make_receipt(
        ReceiptType.EVALUATOR,
        "a1",
        "run",
        spec.ref(),
        "counterevidence",
        {"contradicts": True, "reason": "labels observed"},
    )
    for receipt in [*receipts, contradiction]:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath("clean", "c1", [item.receipt_id for item in receipts]))
    graph.add_path(EvidencePath(
        "contaminated",
        "c1",
        [item.receipt_id for item in [*receipts, contradiction]],
    ))
    decision = AuthorityEngine(registry, graph=graph).authorize(request(spec.ref()))
    assert decision.outcome == DecisionOutcome.ALLOW
    assert decision.satisfied_paths == ["clean"]
    assert decision.blocking_receipts == []


def test_pairwise_claim_can_be_satisfied_by_complete_trusted_evidence():
    registry = ProtocolRegistry()
    spec = registry.register(protocol())
    pairwise = claim(spec.ref())
    pairwise.claim_type = ClaimType.PAIRWISE_SUPERIORITY
    receipts = required_receipts(spec.ref(), "-pairwise")
    host = TrustedCollectorHost("pairwise-host")
    receipts.append(host.collect(
        SeedAggregationCollector,
        artifact_id="a1",
        run_id="run",
        protocol_ref=spec.ref(),
        source="tests.host",
        payload={
            "declared_seeds": [1, 2, 3],
            "all_results_hash": "b" * 64,
            "aggregation": "paired_mean",
            "paired": True,
            "preregistered": True,
            "best_seed_selection": False,
        },
    ))
    for index in range(3):
        receipts.append(host.collect(
            ReplicationCollector,
            artifact_id="a1",
            run_id="run",
            protocol_ref=spec.ref(),
            source="tests.host",
            payload={
                "replication_id": f"seed-{index}",
                "task_family": "text",
                "result_hash": f"{index + 1:x}" * 64,
                "equal_data_budget": True,
                "equal_compute_budget": True,
                "paired_bootstrap_ci_lower_gt_zero": True,
            },
        ))
    graph = EvidenceGraph()
    graph.add_claim(pairwise)
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath("pairwise-complete", "c1", [
        receipt.receipt_id for receipt in receipts
    ]))
    decision = AuthorityEngine(registry, graph=graph).authorize(request(spec.ref()))
    assert decision.outcome == DecisionOutcome.ALLOW


def test_protocol_drift_requires_new_replay_path():
    registry = ProtocolRegistry()
    v1 = registry.register(protocol("1"))
    v2 = registry.register(protocol("2", parent_version="1", data_split_policy={"folds": 10}))
    graph = EvidenceGraph()
    graph.add_claim(claim(v1.ref()))
    receipts = required_receipts(v1.ref())
    for receipt in receipts:
        graph.add_receipt(receipt)
    graph.add_path(EvidencePath("v1-path", "c1", [item.receipt_id for item in receipts]))
    decision = AuthorityEngine(registry, graph=graph).authorize(request(v2.ref()))
    assert decision.outcome == DecisionOutcome.REQUIRE_HUMAN_REVIEW
    assert decision.reason_codes == ["contract_mismatch"]
    assert "active_protocol_compatibility" in decision.missing_obligations


def test_derivation_scope_never_exceeds_parent_intersection():
    parent_scope = AuthorityScope(["score"], ["derived_publication"], ["distillation"], ["p1"], ["task"])
    parents = [
        AuthorityDecision("d1", DecisionOutcome.ALLOW, parent_scope, ["p"], [], [], None, "v1")
    ]
    widened = AuthorityScope(["score"], ["derived_publication", "promote"], ["distillation"], ["p1"], ["task"])
    result = validate_derivation(parents, widened)
    assert result.allowed is False
    assert "scope_widening:operations" in result.reasons


def test_offline_distillation_requires_actuation_but_publication_only_needs_lineage():
    publication = authorize_derivation_operation(
        Operation.DERIVED_PUBLICATION,
        parent_claim_refs=["claim::source"],
        clean_ancestry=True,
    )
    assert publication.outcome == DecisionOutcome.ALLOW

    distillation = authorize_derivation_operation(
        Operation.DISTILL,
        parent_claim_refs=["claim::source"],
        clean_ancestry=True,
    )
    assert distillation.outcome == DecisionOutcome.QUARANTINE
    assert distillation.reasons == [
        "ambiguous_positive_distillation_semantics"
    ]

    actuated = authorize_derivation_operation(
        Operation.DISTILL,
        parent_claim_refs=["claim::source"],
        clean_ancestry=True,
        runtime_actuation_receipts=["receipt::runtime"],
        counterfactual_actuation_receipts=["receipt::counterfactual"],
    )
    assert actuated.outcome == DecisionOutcome.QUARANTINE
    assert actuated.reasons == [
        "ambiguous_positive_distillation_semantics"
    ]

    positive_result = authorize_derivation_operation(
        Operation.DISTILL_POSITIVE_RESULT,
        parent_claim_refs=["claim::source"],
        clean_ancestry=True,
    )
    assert positive_result.outcome == DecisionOutcome.ALLOW

    adopted = authorize_derivation_operation(
        Operation.DISTILL_POSITIVE_ADOPTED,
        parent_claim_refs=["claim::source"],
        clean_ancestry=True,
        runtime_actuation_receipts=["receipt::runtime"],
    )
    assert adopted.outcome == DecisionOutcome.ALLOW

    causal = authorize_derivation_operation(
        Operation.DISTILL_POSITIVE_ADOPTED,
        parent_claim_refs=["claim::source"],
        clean_ancestry=True,
        runtime_actuation_receipts=["receipt::runtime"],
        causal_claim=True,
    )
    assert causal.outcome == DecisionOutcome.QUARANTINE
    assert causal.reasons == ["missing_counterfactual_actuation"]


def test_ledger_detects_tampering_and_orders_concurrent_events(tmp_path):
    ledger = AuthorityLedger(tmp_path / "authority_events.jsonl")
    threads = [threading.Thread(target=ledger.append, args=("receipt", {"i": index})) for index in range(25)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    events = ledger.read()
    assert [event["sequence"] for event in events] == list(range(25))
    assert ledger.verify()
    tampered = copy.deepcopy(events)
    tampered[5]["payload"]["i"] = 999
    assert not ledger.verify(tampered)


def test_ledger_orders_cross_process_events(tmp_path):
    path = str(tmp_path / "authority_events.jsonl")
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_append_ledger_events, args=(path, worker, 8))
        for worker in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    ledger = AuthorityLedger(path)
    assert len(ledger.read()) == 32
    assert ledger.verify()


def test_method_preserving_replay_and_successor_detection():
    source = "from sklearn.linear_model import LogisticRegression\nmodel=LogisticRegression(C=1)\nmodel.fit(X,y)\np=model.predict_proba(T)"
    protocol_only = source + "\n# split protocol repaired"
    changed = source.replace("LogisticRegression", "RandomForestClassifier")
    assert certify_replay(source, protocol_only) == ReplayIdentity.METHOD_PRESERVED
    assert certify_replay(source, changed) == ReplayIdentity.SUCCESSOR_METHOD
    hyperparameter_change = source.replace("C=1", "C=2")
    assert certify_replay(source, hyperparameter_change) == ReplayIdentity.SUCCESSOR_METHOD
    assert certify_replay(source, changed, ["fit_scope"]) == ReplayIdentity.SUCCESSOR_METHOD
    assert certify_replay(source, hyperparameter_change, ["fit_scope"]) == ReplayIdentity.SUCCESSOR_METHOD


def test_protocol_replay_ignores_guard_and_exact_fold_constructor_duplicates():
    source = """
from sklearn.linear_model import LogisticRegression
train_data = LeafDataset(X_train, y_train)
model = LogisticRegression(C=1)
model.fit(X, y)
pred = model.predict_proba(T)
"""
    repaired = """
from sklearn.linear_model import LogisticRegression
from agents.protocol_repair_runtime import ProtocolProvenanceGuard
guard = ProtocolProvenanceGuard()
for fold in folds:
    train_data = LeafDataset(fold.X_train, fold.y_train)
    model = LogisticRegression(C=1)
    model.fit(fold.X, fold.y)
    pred = model.predict_proba(fold.T)
final_model = LogisticRegression(C=1)
"""
    assert certify_replay(source, repaired, ["cross_fit"]) == ReplayIdentity.METHOD_PRESERVED


def test_protocol_replay_allows_fold_scoped_fit_inputs_and_error_guards():
    source = """
model = XGBClassifier(max_depth=6)
pca = PCA(n_components=min(32, data.shape[1]))
model.fit(X_train, y_train)
pred = model.predict_proba(X_valid)
"""
    repaired = """
for fold in folds:
    model = XGBClassifier(max_depth=6)
    pca = PCA(n_components=min(32, fold_train.shape[1]))
    model.fit(fold_X_train, fold_y_train, eval_set=[(fold_X_valid, fold_y_valid)])
    pred = model.predict_proba(fold_X_valid)
    if pred is None:
        raise ValueError("missing OOF prediction")
"""
    assert certify_replay(source, repaired, ["cross_fit"]) == ReplayIdentity.METHOD_PRESERVED
