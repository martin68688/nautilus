import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from agents import protocol_repair
from agents.leakage_audit import audit_code, build_repair_preservation_contract
from agents.protocol_repair_runtime import ProtocolProvenanceGuard
from agents import result_parse_agent
from agents import protocol_repair_agent
from engine.search_node import SearchNode
from engine.agent_search import AgentSearch
from engine.search_node import Journal
from utils.metric import MetricValue


def _agent(tmp_path, task_desc="generic task"):
    return SimpleNamespace(
        task_desc=task_desc,
        acfg=SimpleNamespace(
            check_data_leakage=True,
            protocol_repair=SimpleNamespace(
                enabled=True,
                per_stage_attempt_limit=2,
                require_runtime_provenance=True,
            ),
        ),
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        global_memory=None,
        external_skill_memory=None,
    )


@pytest.mark.parametrize(
    ("task_desc", "code", "modality", "objective", "split_family"),
    [
        ("multiclass author text classification", "TfidfVectorizer()", "text", "classification", "stratified"),
        ("binary image classification", "torchvision.models.resnet18()", "image", "classification", "stratified"),
        ("tabular house price regression RMSE", "XGBRegressor()", "tabular", "regression", "random"),
        ("forecast a time series by timestamp", "TimeSeriesSplit()", "tabular", "regression", "time_ordered"),
        ("patient classification with patient_id groups", "GroupKFold()", "tabular", "classification", "grouped"),
    ],
)
def test_task_profiles_are_generic(task_desc, code, modality, objective, split_family):
    profile = protocol_repair.infer_task_profile(task_desc, code)
    assert profile["modality"] == modality
    assert profile["objective"] == objective
    assert profile["split_family"] == split_family


def test_plan_adds_only_needed_stages_for_single_and_ensemble_models():
    single = protocol_repair.build_protocol_plan(
        "tabular regression",
        "model = RandomForestRegressor()",
    )
    assert single["stages"] == ["data_scope", "final_holdout"]

    ensemble = protocol_repair.build_protocol_plan(
        "text classification",
        """
        model_a = XGBClassifier()
        model_b = LogisticRegression()
        early_stopping_rounds = 20
        best_weights = minimize(objective, x0, args=(y_val, val_preds))
        """,
    )
    assert ensemble["stages"] == [
        "data_scope",
        "validation_provenance",
        "cross_fit",
        "selection_freeze",
        "final_holdout",
    ]
    assert "spooky" not in json.dumps(ensemble).lower()
    assert "deberta" not in json.dumps(ensemble).lower()


def test_only_protocol_scope_failures_enter_staged_repair(tmp_path):
    code = """
X_train, X_val = train_test_split(texts)
all_texts = np.concatenate([X_train, X_val, test_texts])
features = TfidfVectorizer().fit_transform(all_texts)
model = LogisticRegression()
"""
    node = SearchNode(code=code, plan="useful model with bad scope", stage="draft")
    node.leakage_audit = audit_code(code)
    tx = protocol_repair.ensure_transaction(_agent(tmp_path), node)
    assert tx["schema"] == protocol_repair.PROTOCOL_REPAIR_SCHEMA
    assert tx["protocol_plan"]["stages"][0] == "data_scope"
    assert tx["preservation_contract"]["component_calls"]["LogisticRegression"] == 1

    syntax_error = SearchNode(code="model = (", plan="broken", stage="draft")
    syntax_error.leakage_audit = {"status": "blocked", "issues": [{
        "issue_code": "LLM_SELECTION_BIAS", "category": "selection_bias"
    }]}
    assert protocol_repair.ensure_transaction(_agent(tmp_path), syntax_error) == {}


def test_stage_audits_advance_independently_and_exhaust_per_stage():
    plan = protocol_repair.build_protocol_plan("image classification", "model = ResNetModel()")
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": plan,
        "current_stage_index": 0,
        "stage_attempts": {},
        "history": [],
        "state": "pending",
        "max_attempts_per_stage": 2,
    }
    bad = protocol_repair.audit_stage("model = ResNetModel()", tx)
    assert bad["status"] == "blocked"
    tx = protocol_repair.apply_stage_result(tx, bad, "n1")
    assert protocol_repair.current_stage(tx) == "data_scope"
    tx = protocol_repair.apply_stage_result(tx, bad, "n2")
    assert tx["state"] == "exhausted"

    clean_tx = {**tx, "state": "pending", "stage_attempts": {}, "history": []}
    code = """
sample_ids = list(range(len(df)))
outer_train, outer_holdout = train_test_split(sample_ids, stratify=labels)
"""
    passed = protocol_repair.audit_stage(code, clean_tx)
    assert passed["status"] == "clean"
    clean_tx = protocol_repair.apply_stage_result(clean_tx, passed, "n3")
    assert protocol_repair.current_stage(clean_tx) == "final_holdout"


def test_stage_scope_gate_defers_later_issues_but_blocks_current_or_preservation():
    plan = {
        "stages": ["data_scope", "cross_fit", "selection_freeze", "final_holdout"]
    }
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": plan,
        "current_stage_index": 0,
        "state": "pending",
    }
    audit = {"issues": [
        {"issue_code": "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION", "category": "selection_bias"},
    ]}
    gate = protocol_repair.stage_scope_gate(audit, tx)
    assert gate["status"] == "clean"
    assert gate["deferred_issue_codes"] == ["REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION"]

    audit["issues"].append({
        "issue_code": "TRANSFORM_FIT_ON_HOLDOUT",
        "category": "transductive_contamination",
    })
    assert protocol_repair.stage_scope_gate(audit, tx)["status"] == "blocked"


def test_single_model_early_stopping_does_not_require_oof():
    plan = protocol_repair.build_protocol_plan(
        "image classification",
        "model = ResNetModel()\nPATIENCE = 3",
    )
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": plan,
        "current_stage_index": 1,
        "stage_attempts": {},
        "history": [],
        "state": "pending",
        "max_attempts_per_stage": 2,
    }
    code = "inner_train_ids = outer_train_ids[:80]\ninner_valid_ids = outer_train_ids[80:]"
    assert protocol_repair.audit_stage(code, tx)["status"] == "clean"


def test_preexecution_intermediate_stage_is_journal_only(tmp_path):
    code = """
sample_ids = list(range(len(df)))
outer_train, outer_holdout = train_test_split(sample_ids, stratify=labels)
model = LogisticRegression()
"""
    contract = build_repair_preservation_contract(code)
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx",
        "source_node_id": "seed",
        "source_code_sha256": "source",
        "preservation_contract": contract,
        "protocol_plan": protocol_repair.build_protocol_plan("classification", code),
        "current_stage_index": 0,
        "stage_attempts": {},
        "history": [],
        "state": "pending",
        "max_attempts_per_stage": 2,
        "require_runtime_provenance": True,
    }
    node = SearchNode(
        code=code, plan="data stage", stage="debug", protocol_repair=tx,
        leakage_repair_context={"preservation_contract": contract, "issues": []},
    )
    blocked = result_parse_agent.run_pre_execution_leakage_audit(_agent(tmp_path), node)
    assert blocked is True
    assert node.leakage_audit["status"] == "protocol_stage_complete"
    assert node.protocol_repair["current_stage_index"] == 1
    assert node.metric.is_worst
    assert node.replay_status == "staged_protocol_repair_stage_complete"


def test_intermediate_stage_cannot_advance_after_model_simplification(tmp_path):
    source = """
model_a = XGBClassifier(n_estimators=800)
model_b = LogisticRegression(C=2.0)
"""
    child_code = """
sample_ids = list(range(len(df)))
outer_train_ids, outer_holdout_ids = train_test_split(sample_ids, stratify=labels)
model_a = XGBClassifier(n_estimators=800)
"""
    contract = build_repair_preservation_contract(source)
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx-preserve",
        "source_node_id": "seed",
        "source_code_sha256": "source",
        "preservation_contract": contract,
        "protocol_plan": protocol_repair.build_protocol_plan("classification", source),
        "current_stage_index": 0,
        "stage_attempts": {},
        "history": [],
        "state": "pending",
        "max_attempts_per_stage": 2,
        "require_runtime_provenance": True,
    }
    node = SearchNode(
        code=child_code, plan="simplified", stage="debug", protocol_repair=tx,
        leakage_repair_context={"preservation_contract": contract, "issues": []},
    )
    assert result_parse_agent.run_pre_execution_leakage_audit(_agent(tmp_path), node) is True
    assert node.protocol_repair["current_stage_index"] == 0
    assert node.leakage_audit["protocol_preservation_clean"] is False
    assert "REPAIR_MODEL_COMPONENT_REMOVED" in {
        item["issue_code"] for item in node.leakage_audit["issues"]
    }


def test_final_stage_requires_all_static_gates_then_becomes_executable(tmp_path):
    code = """
from agents.protocol_repair_runtime import ProtocolProvenanceGuard
sample_ids = list(range(len(df)))
outer_train_ids, outer_holdout_ids = train_test_split(sample_ids, stratify=labels)
guard = ProtocolProvenanceGuard()
guard.register_partition("outer_train", outer_train_ids)
guard.register_partition("outer_holdout", outer_holdout_ids)
model = LogisticRegression()
guard.record_fit("model", outer_train_ids)
model.fit(X_outer_train, y_outer_train)
guard.record_prediction("model", outer_train_ids, outer_holdout_ids, purpose="final")
guard.record_selection("frozen_model_design", outer_train_ids)
guard.freeze()
guard.record_final_evaluation(outer_holdout_ids)
guard.assert_clean()
guard.emit()
"""
    contract = build_repair_preservation_contract(code)
    plan = protocol_repair.build_protocol_plan("classification", code)
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx-final",
        "source_node_id": "seed",
        "source_code_sha256": "source",
        "preservation_contract": contract,
        "protocol_plan": plan,
        "current_stage_index": len(plan["stages"]) - 1,
        "stage_attempts": {},
        "history": [],
        "state": "final_pending",
        "max_attempts_per_stage": 2,
        "require_runtime_provenance": True,
    }
    node = SearchNode(
        code=code, plan="final", stage="debug", protocol_repair=tx,
        leakage_repair_context={"preservation_contract": contract, "issues": []},
    )
    assert result_parse_agent.run_pre_execution_leakage_audit(_agent(tmp_path), node) is False
    assert node.protocol_repair["state"] == "ready_for_execution"
    assert node.leakage_audit["status"] == "clean"
    assert node.replay_status == "staged_protocol_repair_clean_pending_execution"


def test_runtime_guard_accepts_clean_generic_protocol_and_rejects_overlap():
    guard = ProtocolProvenanceGuard()
    train_ids, holdout_ids = [0, 1, 2, 3], [4, 5]
    guard.register_partition("outer_train", train_ids)
    guard.register_partition("outer_holdout", holdout_ids)
    guard.record_fit("model", train_ids)
    guard.record_prediction("model", [0, 1], [2, 3], purpose="oof")
    guard.record_selection("frozen_model_design", train_ids)
    guard.freeze()
    guard.record_prediction("model", train_ids, holdout_ids, purpose="final")
    guard.record_final_evaluation(holdout_ids)
    guard.assert_clean()
    payload = guard.emit()
    result = protocol_repair.runtime_provenance_audit(
        protocol_repair.RUNTIME_MARKER + json.dumps(payload, sort_keys=True), {}
    )
    assert result["status"] == "clean"

    leaked = ProtocolProvenanceGuard()
    leaked.register_partition("outer_train", [0, 1])
    leaked.register_partition("outer_holdout", [2])
    leaked.record_fit("scaler", [0, 2])
    leaked.record_prediction("model", [0], [1], purpose="oof")
    leaked.record_selection("weights", [0, 1])
    leaked.freeze()
    leaked.record_final_evaluation([2])
    with pytest.raises(RuntimeError, match="outer_holdout"):
        leaked.assert_clean()


def test_final_runtime_failure_returns_to_final_stage_until_budget_exhausted():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["data_scope", "final_holdout"]},
        "current_stage_index": 2,
        "stage_attempts": {"data_scope": 1, "final_holdout": 1},
        "history": [],
        "state": "ready_for_execution",
        "max_attempts_per_stage": 2,
    }
    retried = protocol_repair.rollback_final_runtime_failure(tx, "n1", "missing marker")
    assert retried["state"] == "pending"
    assert protocol_repair.current_stage(retried) == "final_holdout"
    retried["stage_attempts"]["final_holdout"] = 2
    exhausted = protocol_repair.rollback_final_runtime_failure(retried, "n2", "crashed")
    assert exhausted["state"] == "exhausted"


def test_protocol_repair_agent_skips_context_free_code_review(monkeypatch):
    parent = SearchNode(
        code="model = LogisticRegression()", plan="seed", stage="draft",
        leakage_audit={"status": "protocol_biased", "issues": []},
        protocol_repair={
            "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
            "transaction_id": "tx-agent",
            "source_node_id": "seed",
            "source_code_sha256": "source",
            "preservation_contract": build_repair_preservation_contract("model = LogisticRegression()"),
            "protocol_plan": protocol_repair.build_protocol_plan("classification", "model = LogisticRegression()"),
            "current_stage_index": 0,
            "stage_attempts": {},
            "history": [],
            "state": "pending",
            "max_attempts_per_stage": 2,
        },
        is_buggy=True,
        is_valid=False,
    )
    monkeypatch.setattr(protocol_repair_agent, "plan_and_code_query", lambda *_args: ("repair", parent.code))
    monkeypatch.setattr(protocol_repair_agent, "register_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_repair_agent, "log_adoption", lambda *_args, **_kwargs: None)
    agent = SimpleNamespace(
        task_desc="classification",
        acfg=SimpleNamespace(code=SimpleNamespace(model="test")),
    )
    child = protocol_repair_agent.run(agent, parent)
    assert child.skip_code_review is True
    assert child.protocol_repair["transaction_id"] == "tx-agent"


def test_agent_search_routes_active_transaction_only_to_protocol_agent(monkeypatch):
    root = SearchNode(code="", plan="root", stage="root")
    parent = SearchNode(
        code="model = LogisticRegression()", plan="seed", stage="draft",
        parent=root, is_buggy=True, is_valid=False,
        protocol_repair={
            "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
            "protocol_plan": {"stages": ["data_scope", "final_holdout"]},
            "current_stage_index": 0,
            "state": "pending",
        },
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    called = []
    monkeypatch.setattr(
        "engine.agent_search.protocol_repair_agent.run",
        lambda _agent, node: called.append(node) or None,
    )
    executed = []
    is_root, result = AgentSearch._run_single_step(
        agent,
        parent,
        exec_callback=lambda *_args, **_kwargs: executed.append(True),
    )
    assert called == [parent]
    assert executed == []
    assert result is None
    assert is_root is False


def test_releasing_repair_parent_marks_consumed_transaction_superseded():
    parent_tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx-release",
        "protocol_plan": {"stages": ["data_scope", "final_holdout"]},
        "current_stage_index": 0,
        "state": "pending",
    }
    parent = SearchNode(
        code="parent", plan="parent", stage="draft", protocol_repair=parent_tx,
        leakage_audit={"status": "blocked", "repair_required": True},
        audit_repair_required=True,
    )
    child = SearchNode(
        code="child", plan="child", stage="debug", parent=parent,
        protocol_repair={**parent_tx, "state": "exhausted"},
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.journal = Journal(nodes=[parent, child])
    AgentSearch._init_mandatory_repair_scheduler(agent)
    agent._mandatory_repair_inflight_ids.add(parent.id)
    AgentSearch._release_mandatory_repair_parent(agent, parent)
    assert parent.protocol_repair["state"] == "abandoned"
    assert parent.protocol_repair["successor_node_id"] == child.id
    assert parent.is_terminal is True
    assert parent.audit_repair_required is False

    active_parent = SearchNode(
        code="parent2", plan="parent2", stage="draft",
        protocol_repair={**parent_tx, "state": "pending"},
        leakage_audit={"status": "blocked", "repair_required": True},
        audit_repair_required=True,
    )
    active_child = SearchNode(
        code="child2", plan="child2", stage="debug", parent=active_parent,
        protocol_repair={**parent_tx, "state": "pending"},
    )
    agent.journal.nodes.extend([active_parent, active_child])
    agent._mandatory_repair_inflight_ids.add(active_parent.id)
    AgentSearch._release_mandatory_repair_parent(agent, active_parent)
    assert active_parent.protocol_repair["state"] == "superseded"
    assert active_parent.protocol_repair["successor_node_id"] == active_child.id
    assert active_parent.is_terminal is True


def test_postexecution_runtime_provenance_completes_or_rolls_back(monkeypatch, tmp_path):
    clean_payload = {
        "schema": "mlevolve_protocol_provenance_v1",
        "status": "clean",
        "violations": [],
        "counts": {"partitions": 2, "fits": 1, "predictions": 2, "selections": 1, "final_evaluations": 1},
    }
    base_tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["data_scope", "final_holdout"]},
        "current_stage_index": 2,
        "stage_attempts": {"data_scope": 1, "final_holdout": 1},
        "history": [],
        "state": "ready_for_execution",
        "max_attempts_per_stage": 2,
    }
    monkeypatch.setattr(
        result_parse_agent.data_leakage_agent,
        "run",
        lambda *_args, **_kwargs: {"classification": "clean", "confidence": "high", "reason": "clean"},
    )
    agent = _agent(tmp_path)
    agent.metric_maximize = True

    clean = SearchNode(
        code="print('clean')", plan="clean", stage="debug",
        metric=MetricValue(0.8, maximize=True), is_buggy=False, is_valid=True,
        protocol_repair=dict(base_tx),
        _term_out=[protocol_repair.RUNTIME_MARKER + json.dumps(clean_payload, sort_keys=True)],
    )
    clean.leakage_audit = audit_code(clean.code)
    result_parse_agent._check_data_leakage(agent, clean, {"metric": 0.8})
    assert clean.protocol_repair["state"] == "completed"
    assert clean.leakage_audit["status"] == "clean"

    failed = SearchNode(
        code="print('missing marker')", plan="failed", stage="debug",
        metric=MetricValue(0.7, maximize=True), is_buggy=False, is_valid=True,
        protocol_repair=dict(base_tx), _term_out=["no provenance"],
    )
    failed.leakage_audit = audit_code(failed.code)
    result_parse_agent._check_data_leakage(agent, failed, {"metric": 0.7})
    assert failed.protocol_repair["state"] == "pending"
    assert protocol_repair.current_stage(failed.protocol_repair) == "final_holdout"
    assert failed.is_buggy is True


def test_preservation_contract_is_carried_by_transaction(tmp_path):
    code = """
model = XGBClassifier(n_estimators=800)
linear = LogisticRegression(C=2.0)
"""
    node = SearchNode(code=code, plan="ensemble", stage="draft")
    node.leakage_audit = {
        "status": "protocol_biased",
        "issues": [{
            "issue_code": "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION",
            "category": "selection_bias",
        }],
    }
    tx = protocol_repair.ensure_transaction(_agent(tmp_path), node)
    assert tx["preservation_contract"]["component_calls"] == {
        "LogisticRegression": 1,
        "XGBClassifier": 1,
    }


def test_preservation_contract_covers_non_text_and_lowercase_model_factories():
    code = """
backbone = torchvision.models.resnet50(weights="DEFAULT")
model = keras.Sequential([backbone])
checkpoint_path = "vendor/vision-checkpoint-v2"
"""
    contract = build_repair_preservation_contract(code)
    assert contract["component_calls"]["resnet50"] == 1
    assert contract["component_calls"]["Sequential"] == 1
    assert "vendor/vision-checkpoint-v2" in contract["model_literals"]
