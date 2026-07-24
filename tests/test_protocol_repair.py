import ast
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from agents import protocol_repair
from agents.leakage_audit import (
    audit_code,
    audit_repair_preservation,
    build_repair_preservation_contract,
)
from agents.protocol_repair_runtime import ProtocolProvenanceGuard
from agents import result_parse_agent
from agents import protocol_repair_agent
from agents.triggers import register_node
from engine.search_node import SearchNode
from engine.agent_search import AgentSearch
from engine import executor
from engine.search_node import Journal
from utils.metric import MetricValue
from authority.protocol_registry import ProtocolRegistry


def _agent(tmp_path, task_desc="generic task"):
    return SimpleNamespace(
        task_desc=task_desc,
        acfg=SimpleNamespace(
            check_data_leakage=True,
            feedback=SimpleNamespace(model="test-reviewer", temp=0.0),
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

    fixed_ensemble = protocol_repair.build_protocol_plan(
        "text classification",
        "model_a = XGBClassifier(); model_b = LogisticRegression(); prediction = 0.7 * a + 0.3 * b",
    )
    assert "cross_fit" in fixed_ensemble["stages"]
    assert "selection_freeze" not in fixed_ensemble["stages"]


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
    second_bad = protocol_repair.audit_stage("model = AlternateResNetModel()", tx)
    tx = protocol_repair.apply_stage_result(tx, second_bad, "n2")
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


def test_identical_rejection_twice_stays_retriable_with_stronger_feedback():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["selection_freeze"]},
        "current_stage_index": 0,
        "stage_attempts": {},
        "history": [],
        "state": "pending",
        "stage_attempt_limits": {"selection_freeze": 6},
    }
    audit = {
        "status": "blocked",
        "code_sha256": "same-candidate",
        "issues": [{
            "issue_code": "PROTOCOL_STAGE_SELECTION_FREEZE_INCOMPLETE",
            "evidence": "selected state is not causally updated",
            "remediation": "trace the OOF metric into the selected state",
            "line": 0,
        }],
    }
    first = protocol_repair.apply_stage_result(tx, audit, "n1")
    assert first["state"] == "pending"
    second = protocol_repair.apply_stage_result(first, audit, "n2")
    assert second["state"] == "pending"
    assert second["repeated_candidate"] == {
        "stage": "selection_freeze",
        "code_sha256": "same-candidate",
        "issue_codes": ["PROTOCOL_STAGE_SELECTION_FREEZE_INCOMPLETE"],
        "repeated_attempts": [1, 2],
    }
    assert second["history"][-1]["feedback"][-1]["issue_code"] == (
        "PROTOCOL_REPAIR_REPEATED_IDENTICAL_CANDIDATE"
    )
    assert "materially different" in second["history"][-1]["feedback"][-1]["remediation"]
    for attempt in range(3, 6):
        second = protocol_repair.apply_stage_result(second, audit, f"n{attempt}")
        assert second["state"] == "pending"
        assert second["history"][-1]["feedback"][-1]["issue_code"] == (
            "PROTOCOL_REPAIR_REPEATED_IDENTICAL_CANDIDATE"
        )
    exhausted = protocol_repair.apply_stage_result(second, audit, "n6")
    assert exhausted["state"] == "exhausted"
    assert exhausted["terminal_reason"] == "stage_attempts_exhausted:selection_freeze"


def test_empty_rejection_does_not_create_repeated_candidate_metadata():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["cross_fit"]},
        "current_stage_index": 0,
        "stage_attempts": {},
        "history": [],
        "state": "pending",
        "stage_attempt_limits": {"cross_fit": 7},
    }
    audit = {"status": "blocked", "code_sha256": "same", "issues": []}
    first = protocol_repair.apply_stage_result(tx, audit, "n1")
    second = protocol_repair.apply_stage_result(first, audit, "n2")
    assert second["state"] == "pending"
    assert "repeated_candidate" not in second


def test_nonrepeated_failure_clears_stale_repeated_candidate_metadata():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["cross_fit"]},
        "current_stage_index": 0,
        "stage_attempts": {"cross_fit": 2},
        "history": [],
        "state": "pending",
        "stage_attempt_limits": {"cross_fit": 7},
        "repeated_candidate": {"stage": "cross_fit", "code_sha256": "old"},
    }
    audit = {
        "status": "blocked",
        "code_sha256": "new",
        "issues": [{"issue_code": "NEW", "evidence": "new", "remediation": "fix"}],
    }
    updated = protocol_repair.apply_stage_result(tx, audit, "n3")
    assert updated["state"] == "pending"
    assert "repeated_candidate" not in updated


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


def test_method_identity_uses_active_protocol_surface_not_workflow_stages(tmp_path):
    code = """
sample_ids = list(range(len(df)))
outer_train_ids, outer_holdout_ids = train_test_split(sample_ids, stratify=labels)
model = LogisticRegression()
model.fit(X_outer_train, y_outer_train)
"""
    source = SearchNode(code=code, plan="frozen", stage="draft")
    transaction = {
        "source_node_id": source.id,
        # These are workflow stages and deliberately are not valid
        # ProtocolRepairSurface change kinds.
        "protocol_plan": {
            "stages": ["data_scope", "validation_provenance", "final_holdout"]
        },
    }
    node = SearchNode(
        code=code,
        plan="protocol-only",
        stage="debug",
        protocol_repair=transaction,
        leakage_repair_context={"source_node_id": source.id},
    )
    agent = _agent(tmp_path)
    agent.journal = Journal(nodes=[source])
    registry = ProtocolRegistry(REPO / "mlevolve" / "config" / "protocols")
    agent.evaluation_authority = SimpleNamespace(
        active_protocol_spec=registry.get("mlevolve-default", "2")
    )

    report = result_parse_agent._method_identity_audit(agent, node)

    assert report["method_identity"] == "method_preserved"
    assert report["issues"] == []
    assert report["repair_surface"]["allowed_change_kinds"] == [
        "evaluator",
        "holdout_access",
        "instrumentation",
        "preprocessing_scope",
        "seed_aggregation",
        "selection_freeze",
        "split_api",
    ]
    assert len(report["replay_verification_hash"]) == 64


def test_final_stage_requires_all_static_gates_then_becomes_executable(tmp_path, monkeypatch):
    code = """
from agents.protocol_repair_runtime import ProtocolProvenanceGuard
sample_ids = list(range(len(df)))
outer_train_ids, outer_holdout_ids = train_test_split(sample_ids, stratify=labels)
guard = ProtocolProvenanceGuard()
guard.register_partition("outer_train", outer_train_ids)
guard.register_partition("outer_holdout", outer_holdout_ids)
model = LogisticRegression()
guard.record_selection("frozen_model_design", outer_train_ids)
guard.freeze()
guard.record_fit("model", outer_train_ids)
model.fit(X_outer_train, y_outer_train)
guard.record_prediction("model", outer_train_ids, outer_holdout_ids, purpose="final")
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
    monkeypatch.setattr(
        result_parse_agent.data_leakage_agent,
        "run_pre_execution_protocol_review",
        lambda *_args, **_kwargs: {
            "status": "clean",
            "classification": "clean",
            "reason": "final predictions and labels are aligned to outer_holdout",
            "prediction_source": "X_outer_holdout",
            "label_source": "y_outer_holdout",
            "required_fix": "none",
        },
    )
    assert result_parse_agent.run_pre_execution_leakage_audit(_agent(tmp_path), node) is False
    assert node.protocol_repair["state"] == "ready_for_execution"
    assert node.leakage_audit["status"] == "clean"
    assert node.replay_status == "staged_protocol_repair_clean_pending_execution"


def test_final_semantic_review_blocks_misaligned_prediction_source(tmp_path, monkeypatch):
    code = _complete_final_holdout_program()
    contract = build_repair_preservation_contract(code)
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx-semantic-final",
        "source_node_id": "seed",
        "source_code_sha256": "source",
        "preservation_contract": contract,
        "protocol_plan": _final_holdout_transaction()["protocol_plan"],
        "current_stage_index": 3,
        "stage_attempts": {},
        "history": [],
        "state": "final_pending",
        "max_attempts_per_stage": 3,
        "require_runtime_provenance": True,
    }
    node = SearchNode(
        code=code, plan="misaligned final", stage="debug", protocol_repair=tx,
        leakage_repair_context={"preservation_contract": contract, "issues": []},
    )
    monkeypatch.setattr(
        result_parse_agent.data_leakage_agent,
        "run_pre_execution_protocol_review",
        lambda *_args, **_kwargs: {
            "status": "violation",
            "classification": "prediction_label_misalignment",
            "reason": "test predictions are sliced and paired with outer holdout labels",
            "prediction_source": "test_df",
            "label_source": "outer_holdout",
            "required_fix": "predict X_outer_holdout separately before computing the metric",
        },
    )
    assert result_parse_agent.run_pre_execution_leakage_audit(_agent(tmp_path), node) is True
    assert node.protocol_repair["state"] == "pending"
    feedback = node.protocol_repair["history"][-1]["feedback"]
    assert any(item["issue_code"] == "PROTOCOL_FINAL_SEMANTIC_SPLIT_NOT_CLEAN" for item in feedback)
    assert "test predictions are sliced" in feedback[-1]["evidence"]


def test_semantic_protocol_reviewer_traces_real_prediction_and_label_sources(tmp_path, monkeypatch):
    captured = {}

    def fake_query(**kwargs):
        from llm import compile_prompt_to_md

        captured["compiled_prompt"] = compile_prompt_to_md(kwargs["system_message"])
        captured.update(kwargs)
        return {
            "status": "violation",
            "classification": "prediction_label_misalignment",
            "reason": "submission predictions do not correspond to holdout labels",
            "prediction_source": "external test rows",
            "label_source": "outer holdout rows",
            "required_fix": "generate a separate holdout prediction array",
        }

    monkeypatch.setattr(result_parse_agent.data_leakage_agent, "query", fake_query)
    result = result_parse_agent.data_leakage_agent.run_pre_execution_protocol_review(
        _agent(tmp_path),
        SearchNode(code="metric = score(y_outer_holdout, test_predictions[:n])", plan="", stage="debug"),
        {
            "protocol_plan": {
                "stages": ["final_holdout"],
                "capabilities": {"has_ensemble": True, "model_component_count": 3},
            },
            "preservation_contract": {"status": "frozen", "required": True},
        },
    )
    prompt = captured["compiled_prompt"]
    assert result["status"] == "violation"
    assert "Do not trust variable names" in prompt
    assert "actual outer_holdout feature rows" in prompt
    assert "external test/submission predictions" in prompt
    assert '"has_ensemble": true' in prompt
    assert captured["temperature"] == 0.0


def test_semantic_protocol_reviewer_retries_self_contradictory_violation(tmp_path, monkeypatch):
    responses = iter([
        {
            "status": "violation",
            "classification": "selection_bias",
            "reason": "The OOF selection and untouched holdout are correct; status should be clean.",
            "prediction_source": "outer_holdout",
            "label_source": "outer_holdout",
            "required_fix": "No fix needed. The protocol is clean.",
        },
        {
            "status": "clean",
            "classification": "clean",
            "reason": "OOF selection is frozen before the untouched holdout is evaluated.",
            "prediction_source": "outer_holdout",
            "label_source": "outer_holdout",
            "required_fix": "none",
        },
    ])
    prompts = []

    def fake_query(**kwargs):
        prompts.append(kwargs["system_message"])
        return next(responses)

    monkeypatch.setattr(result_parse_agent.data_leakage_agent, "query", fake_query)
    result = result_parse_agent.data_leakage_agent.run_pre_execution_protocol_review(
        _agent(tmp_path),
        SearchNode(code=_complete_final_holdout_program(), plan="", stage="debug"),
        _final_holdout_transaction(),
    )

    assert result["status"] == "clean"
    assert result["classification"] == "clean"
    assert result["review_attempts"] == 2
    assert len(prompts) == 2
    assert "Previous self-contradictory review" in prompts[1]
    assert "Do not call clean behavior a violation" in prompts[1]["Correction required"]


def test_semantic_protocol_reviewer_fails_closed_after_three_contradictions(tmp_path, monkeypatch):
    def fake_query(**_kwargs):
        return {
            "status": "violation",
            "classification": "selection_bias",
            "reason": "The protocol appears clean and status should be clean.",
            "prediction_source": "outer_holdout",
            "label_source": "outer_holdout",
            "required_fix": "No fix needed.",
        }

    monkeypatch.setattr(result_parse_agent.data_leakage_agent, "query", fake_query)
    result = result_parse_agent.data_leakage_agent.run_pre_execution_protocol_review(
        _agent(tmp_path),
        SearchNode(code=_complete_final_holdout_program(), plan="", stage="debug"),
        _final_holdout_transaction(),
    )

    assert result["status"] == "uncertain"
    assert result["classification"] == "audit_unavailable"
    assert result["review_attempts"] == 3
    assert "Do not modify the candidate" in result["required_fix"]


def test_runtime_guard_accepts_clean_generic_protocol_and_rejects_overlap():
    guard = ProtocolProvenanceGuard()
    train_ids, holdout_ids = [0, 1, 2, 3], [4, 5]
    guard.register_partition("outer_train", train_ids)
    guard.register_partition("outer_holdout", holdout_ids)
    guard.check_no_overlap("outer_train", "outer_holdout")
    guard.check_containment("outer_train", train_ids)
    guard.record_fit("model", train_ids)
    guard.record_prediction("model", [0, 1], [2, 3], purpose="oof")
    guard.record_prediction("model", [2, 3], [0, 1], purpose="oof")
    guard.record_global_oof([[0.1], [0.2], [0.3], [0.4]], train_ids)
    guard.record_selection("frozen_model_design", train_ids)
    guard.freeze()
    guard.record_prediction("model", train_ids, holdout_ids, purpose="final")
    guard.record_final_evaluation(holdout_ids)
    guard.assert_clean()
    payload = guard.emit()
    result = protocol_repair.runtime_provenance_audit(
        protocol_repair.RUNTIME_MARKER + json.dumps(payload, sort_keys=True),
        {"protocol_plan": {"stages": ["data_scope", "cross_fit", "selection_freeze", "final_holdout"]}},
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
    assert retried["history"][-1]["feedback"][0]["issue_code"] == "PROTOCOL_FINAL_RUNTIME_FAILED"
    retried["stage_attempts"]["final_holdout"] = 2
    exhausted = protocol_repair.rollback_final_runtime_failure(retried, "n2", "crashed")
    assert exhausted["state"] == "exhausted"


def test_final_runtime_budget_is_independent_from_static_rejections():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["data_scope", "final_holdout"]},
        "current_stage_index": 2,
        "stage_attempts": {"data_scope": 1, "final_holdout": 5},
        "stage_attempt_limits": {"final_holdout": 8},
        "stage_runtime_attempts": {},
        "final_runtime_attempt_limit": 4,
        "history": [],
        "state": "ready_for_execution",
    }

    for runtime_attempt in range(1, 4):
        tx = protocol_repair.rollback_final_runtime_failure(
            tx,
            f"runtime-{runtime_attempt}",
            f"ValueError on runtime attempt {runtime_attempt}",
        )
        assert tx["state"] == "pending"
        assert tx["stage_runtime_attempts"]["final_holdout"] == runtime_attempt
        tx["state"] = "ready_for_execution"

    exhausted = protocol_repair.rollback_final_runtime_failure(
        tx,
        "runtime-4",
        "ValueError on runtime attempt 4",
    )
    assert exhausted["state"] == "exhausted"
    assert exhausted["terminal_reason"] == "runtime_attempts_exhausted:final_holdout"
    assert exhausted["stage_attempts"]["final_holdout"] == 5


def test_configured_complex_stage_budgets_are_recorded_in_transaction(tmp_path):
    agent = _agent(tmp_path)
    agent.acfg.protocol_repair.stage_attempt_limits = {
        "cross_fit": 8,
        "selection_freeze": 7,
        "final_holdout": 12,
    }
    agent.acfg.protocol_repair.stage_generation_attempt_limits = {
        "cross_fit": 8,
        "selection_freeze": 7,
        "final_holdout": 12,
    }
    agent.acfg.protocol_repair.final_runtime_attempt_limit = 9
    code = """
model_a = XGBClassifier()
model_b = LogisticRegression()
early_stopping_rounds = 10
best_weights = minimize(objective, x0, args=(y_val, val_preds))
"""
    node = SearchNode(code=code, plan="ensemble", stage="draft")
    node.leakage_audit = {
        "status": "protocol_biased",
        "issues": [{
            "issue_code": "LLM_SELECTION_BIAS",
            "category": "selection_bias",
        }],
    }

    tx = protocol_repair.ensure_transaction(agent, node)

    assert tx["stage_attempt_limits"]["cross_fit"] == 8
    assert tx["stage_attempt_limits"]["selection_freeze"] == 7
    assert tx["stage_attempt_limits"]["final_holdout"] == 12
    assert tx["stage_generation_attempt_limits"]["final_holdout"] == 12
    assert tx["final_runtime_attempt_limit"] == 9
    assert tx["stage_runtime_attempts"] == {}


def test_runtime_failure_feedback_is_reinjected_into_final_generation():
    tx = {
        "protocol_plan": {"stages": ["final_holdout"]},
        "current_stage_index": 1,
        "stage_attempts": {"final_holdout": 1},
        "stage_attempt_limits": {"final_holdout": 5},
        "history": [],
        "state": "ready_for_execution",
    }
    failed = protocol_repair.rollback_final_runtime_failure(
        tx,
        "runtime-node",
        "AttributeError: ProtocolProvenanceGuard has no attribute check_no_overlap",
    )
    parent = SearchNode(code="", plan="runtime failed", stage="debug", leakage_audit={"issues": []})
    feedback = protocol_repair_agent._rejection_feedback(parent, failed, "final_holdout")
    assert "AttributeError" in feedback
    assert "check_no_overlap" in feedback


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
    monkeypatch.setattr(
        protocol_repair_agent,
        "plan_and_code_query",
        lambda *_args, **_kwargs: ("repair", parent.code),
    )
    monkeypatch.setattr(protocol_repair_agent, "register_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_repair_agent, "log_adoption", lambda *_args, **_kwargs: None)
    agent = SimpleNamespace(
        task_desc="classification",
        acfg=SimpleNamespace(code=SimpleNamespace(model="test")),
    )
    child = protocol_repair_agent.run(agent, parent)
    assert child.skip_code_review is True
    assert child.protocol_repair["transaction_id"] == "tx-agent"


def test_protocol_repair_agent_recovers_valid_code_only_response(monkeypatch):
    parent = SearchNode(
        code="model = LogisticRegression()", plan="seed", stage="draft",
        leakage_audit={"status": "protocol_biased", "issues": []},
        protocol_repair={
            "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
            "transaction_id": "tx-code-only",
            "source_node_id": "seed",
            "source_code_sha256": "source",
            "preservation_contract": {},
            "protocol_plan": {"stages": ["data_scope"]},
            "current_stage_index": 0,
            "stage_attempts": {},
            "stage_generation_attempts": {},
            "history": [],
            "state": "pending",
            "max_attempts_per_stage": 2,
        },
        is_buggy=True,
        is_valid=False,
    )
    monkeypatch.setattr(
        protocol_repair_agent,
        "plan_and_code_query",
        lambda *_args, **_kwargs: ("", "```python\nprint('repaired')\n```"),
    )
    monkeypatch.setattr(protocol_repair_agent, "register_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_repair_agent, "log_adoption", lambda *_args, **_kwargs: None)
    agent = SimpleNamespace(
        task_desc="classification",
        acfg=SimpleNamespace(code=SimpleNamespace(model="test")),
    )

    child = protocol_repair_agent.run(agent, parent)

    assert child.code.strip() == 'print("repaired")'
    assert child.plan.startswith("[staged_protocol_repair:data_scope]")
    assert "frozen `data_scope`" in child.plan


def test_protocol_repair_prompt_pins_exact_protected_constructors(monkeypatch):
    source = (
        "loss_fn = CrossEntropyLoss(label_smoothing=0.1)\n"
        "lr = LogisticRegression(C=2.0, max_iter=1000)\n"
        "checkpoint = './working/best_deberta_model.pt'\n"
    )
    parent = SearchNode(
        code=source, plan="seed", stage="draft",
        leakage_audit={"status": "protocol_biased", "issues": []},
        protocol_repair={
            "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
            "transaction_id": "tx-pinned-calls",
            "source_node_id": "seed",
            "source_code_sha256": "source",
            "preservation_contract": build_repair_preservation_contract(source),
            "protocol_plan": {"stages": ["cross_fit"]},
            "current_stage_index": 0,
            "stage_attempts": {}, "stage_generation_attempts": {}, "history": [],
            "state": "pending", "max_attempts_per_stage": 2,
        },
        is_buggy=True, is_valid=False,
    )
    captured = {}

    def fake_query(_agent, prompt, **_kwargs):
        captured["prompt"] = str(prompt)
        return "repair", source

    monkeypatch.setattr(protocol_repair_agent, "plan_and_code_query", fake_query)
    monkeypatch.setattr(protocol_repair_agent, "register_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_repair_agent, "log_adoption", lambda *_args, **_kwargs: None)
    agent = SimpleNamespace(task_desc="classification", acfg=SimpleNamespace(code=SimpleNamespace(model="test")))

    protocol_repair_agent.run(agent, parent)

    assert "CrossEntropyLoss(label_smoothing=0.1)" in captured["prompt"]
    assert "LogisticRegression(C=2.0, max_iter=1000)" in captured["prompt"]
    assert "./working/best_deberta_model.pt" in captured["prompt"]
    assert "Copy every expression above exactly" in captured["prompt"]
    assert "must still appear verbatim" in captured["prompt"]


def test_protocol_repair_anchors_missing_identity_literals_after_future_imports(monkeypatch):
    source = "checkpoint = './working/best_deberta_model.pt'\nmodel = LogisticRegression()\n"
    generated = '"""module"""\nfrom __future__ import annotations\nmodel = LogisticRegression()\n'
    parent = SearchNode(
        code=source, plan="seed", stage="draft",
        leakage_audit={"status": "protocol_biased", "issues": []},
        protocol_repair={
            "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
            "transaction_id": "tx-anchor-literal", "source_node_id": "seed", "source_code_sha256": "source",
            "preservation_contract": build_repair_preservation_contract(source),
            "protocol_plan": {"stages": ["cross_fit"]}, "current_stage_index": 0,
            "stage_attempts": {}, "stage_generation_attempts": {}, "history": [], "state": "pending",
            "max_attempts_per_stage": 2,
        },
        is_buggy=True, is_valid=False,
    )
    monkeypatch.setattr(protocol_repair_agent, "plan_and_code_query", lambda *_args, **_kwargs: ("repair", generated))
    monkeypatch.setattr(protocol_repair_agent, "register_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_repair_agent, "log_adoption", lambda *_args, **_kwargs: None)
    agent = SimpleNamespace(task_desc="classification", acfg=SimpleNamespace(code=SimpleNamespace(model="test")))

    child = protocol_repair_agent.run(agent, parent)

    compile(child.code, "<repair>", "exec")
    assert "./working/best_deberta_model.pt" in child.code
    assert child.code.index("from __future__ import annotations") < child.code.index("_MLEVOLVE_PRESERVED_MODEL_LITERALS")


def test_protocol_repair_restores_protected_constructor_arguments(monkeypatch):
    source = "model = AutoModel.from_pretrained('microsoft/deberta-v3-large', num_labels=3)\n"
    generated = "model = AutoModel.from_pretrained('microsoft/deberta-v3-large', num_labels=2)\n"
    parent = SearchNode(
        code=source, plan="seed", stage="draft",
        leakage_audit={"status": "protocol_biased", "issues": []},
        protocol_repair={
            "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
            "transaction_id": "tx-restore-call", "source_node_id": "source-node", "source_code_sha256": "source",
            "preservation_contract": build_repair_preservation_contract(source),
            "protocol_plan": {"stages": ["data_scope"]}, "current_stage_index": 0,
            "stage_attempts": {}, "stage_generation_attempts": {}, "history": [], "state": "pending",
            "max_attempts_per_stage": 2,
        },
        is_buggy=True, is_valid=False,
    )
    parent.id = "source-node"
    monkeypatch.setattr(protocol_repair_agent, "plan_and_code_query", lambda *_args, **_kwargs: ("repair", generated))
    monkeypatch.setattr(protocol_repair_agent, "register_node", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(protocol_repair_agent, "log_adoption", lambda *_args, **_kwargs: None)
    agent = SimpleNamespace(task_desc="classification", acfg=SimpleNamespace(code=SimpleNamespace(model="test")))

    child = protocol_repair_agent.run(agent, parent)

    assert "num_labels=3" in child.code
    assert "num_labels=2" not in child.code


def test_constructor_restore_is_independent_when_another_component_has_extra_calls():
    source = """
optimizer = AdamW(params, lr=2e-5, eps=1e-8)
criterion = CrossEntropyLoss(label_smoothing=0.1)
"""
    generated = """
optimizer = AdamW(params, lr=1e-3)
criterion = CrossEntropyLoss()
auxiliary_criterion = CrossEntropyLoss()
"""
    restored = protocol_repair_agent._restore_protected_component_calls(
        generated,
        source,
        build_repair_preservation_contract(source),
    )
    assert "AdamW(params, lr=2e-05, eps=1e-08)" in restored
    assert "CrossEntropyLoss(label_smoothing=0.1)" in restored
    assert restored.count("CrossEntropyLoss") == 2


def test_stage_retry_uses_latest_failed_candidate_program():
    clean = SearchNode(
        code="clean_stage_input = True",
        plan="clean",
        stage="debug",
        protocol_repair={
            "history": [{"stage": "data_scope", "status": "passed"}],
        },
    )
    first_failure = SearchNode(
        code="bad_validation_attempt = 1",
        plan="failed",
        stage="debug",
        parent=clean,
        protocol_repair={
            "history": [
                {"stage": "data_scope", "status": "passed"},
                {"stage": "validation_provenance", "status": "failed"},
            ],
        },
    )
    second_failure = SearchNode(
        code="bad_validation_attempt = 2",
        plan="failed again",
        stage="debug",
        parent=first_failure,
        protocol_repair={
            "history": [
                {"stage": "data_scope", "status": "passed"},
                {"stage": "validation_provenance", "status": "failed"},
                {"stage": "validation_provenance", "status": "failed"},
            ],
        },
    )
    base = protocol_repair_agent._stage_generation_base(
        second_failure,
        "validation_provenance",
    )
    assert base is second_failure
    assert base.code == "bad_validation_attempt = 2"


def test_actionable_cross_fit_contract_names_exact_runtime_calls():
    tx = {
        "history": [{
            "stage": "cross_fit",
            "status": "failed",
            "feedback": [{
                "issue_code": "PROTOCOL_STAGE_CROSS_FIT_INCOMPLETE",
                "evidence": (
                    "global OOF coverage provenance is not recorded; "
                    "fold-local preprocessor word_vectorizer lacks record_fit "
                    "provenance using that exact component label"
                ),
                "remediation": "record complete OOF coverage",
            }],
        }],
    }
    contract = json.loads(
        protocol_repair_agent._actionable_rejection_contract(tx, "cross_fit")
    )
    assert contract["retry_mode"] == "edit_latest_candidate_in_place"
    assert (
        "protocol_guard.record_global_oof(oof_predictions, outer_train_ids)"
        in contract["required_runtime_calls"]
    )
    assert contract["rejections_to_fix"][0]["issue_code"] == (
        "PROTOCOL_STAGE_CROSS_FIT_INCOMPLETE"
    )
    assert contract["required_component_fit_calls"] == [
        'protocol_guard.record_fit("word_vectorizer", inner_train_ids, '
        'purpose="fold_preprocess")'
    ]
    assert any(
        "after complete OOF assignment" in item
        for item in contract["acceptance_checks"]
    )


def test_five_protocol_stages_do_not_consume_legacy_repair_attempts(monkeypatch):
    stages = [
        "data_scope",
        "validation_provenance",
        "cross_fit",
        "selection_freeze",
        "final_holdout",
    ]
    root = SearchNode(code="", plan="root", stage="root")
    parent = SearchNode(
        code="source", plan="seed", stage="draft", parent=root,
        leakage_repair_attempt=1,
        leakage_audit={"status": "protocol_biased", "issues": []},
        protocol_repair={
            "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
            "transaction_id": "tx-five-stage",
            "protocol_plan": {"stages": stages},
            "current_stage_index": 0,
            "stage_attempts": {},
            "stage_generation_attempts": {},
            "state": "pending",
            "max_attempts_per_stage": 2,
            "max_generation_attempts_per_stage": 2,
        },
    )
    agent = SimpleNamespace(
        next_branch_id=1,
        branch_all_nodes={1: [parent]},
        branch_successful_nodes={1: []},
        _serialize_prompt=lambda prompt: str(prompt),
    )

    for index, stage in enumerate(stages):
        generation = protocol_repair.begin_stage_generation(parent.protocol_repair)
        child = SearchNode(
            code=f"stage-{stage}", plan=stage, stage="debug", parent=parent,
            protocol_repair=protocol_repair.finish_stage_generation(generation),
        )
        register_node(agent, child, "prompt", parent_node=parent)
        assert child.leakage_repair_attempt == 1
        assert child.protocol_repair["stage_generation_attempts"][stage] == 1
        child.protocol_repair = protocol_repair.apply_stage_result(
            child.protocol_repair,
            {"status": "clean", "code_sha256": f"sha-{index}", "issues": []},
            child.id,
        )
        parent = child

    assert parent.protocol_repair["state"] == "ready_for_execution"
    assert parent.leakage_repair_attempt == 1
    assert set(parent.protocol_repair["stage_generation_attempts"]) == set(stages)


def test_protocol_generation_failures_have_per_stage_budget():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx-generation-budget",
        "protocol_plan": {"stages": ["data_scope", "cross_fit"]},
        "current_stage_index": 0,
        "stage_attempts": {},
        "stage_generation_attempts": {},
        "stage_generation_failures": {},
        "history": [],
        "state": "pending",
        "max_generation_attempts_per_stage": 2,
    }
    first = protocol_repair.begin_stage_generation(tx)
    first = protocol_repair.record_stage_generation_failure(first, "n1", "timeout")
    assert first["state"] == "pending"
    assert first["stage_generation_attempts"] == {"data_scope": 1}
    assert first["stage_generation_failures"] == {"data_scope": 1}

    second = protocol_repair.begin_stage_generation(first)
    second = protocol_repair.record_stage_generation_failure(second, "n2", "timeout")
    assert second["state"] == "exhausted"
    assert second["stage_generation_attempts"] == {"data_scope": 2}
    assert second["stage_generation_failures"] == {"data_scope": 2}
    assert second["terminal_reason"] == "stage_generation_attempts_exhausted:data_scope"
    assert [entry["status"] for entry in second["history"]] == [
        "generation_failed", "generation_failed"
    ]


def test_stage_specific_budget_allows_more_cross_fit_attempts():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx-stage-specific-budget",
        "protocol_plan": {"stages": ["cross_fit"]},
        "current_stage_index": 0,
        "stage_attempts": {},
        "history": [],
        "state": "pending",
        "max_attempts_per_stage": 2,
        "stage_attempt_limits": {"cross_fit": 4},
    }
    for attempt in range(1, 4):
        tx = protocol_repair.apply_stage_result(
            tx,
            {
                "status": "blocked",
                "code_sha256": f"sha-{attempt}",
                "issues": [{
                    "issue_code": "PROTOCOL_STAGE_CROSS_FIT_INCOMPLETE",
                    "evidence": f"failure {attempt}",
                    "remediation": "repair the OOF scope",
                }],
            },
            f"n{attempt}",
        )
        assert tx["state"] == "pending"
    tx = protocol_repair.apply_stage_result(
        tx,
        {"status": "blocked", "code_sha256": "sha-4", "issues": []},
        "n4",
    )
    assert tx["state"] == "exhausted"
    assert tx["stage_attempts"] == {"cross_fit": 4}


def test_rejection_feedback_is_injected_into_next_stage_prompt():
    tx = {
        "history": [{
            "stage": "cross_fit",
            "status": "failed",
            "feedback": [{
                "issue_code": "OOF_SCOPE",
                "evidence": "outer holdout was transformed before freeze",
                "remediation": "remove all outer holdout use from cross_fit",
            }],
        }],
    }
    parent = SearchNode(
        code="", plan="failed", stage="debug",
        leakage_audit={"status": "blocked", "issues": []},
    )
    feedback = protocol_repair_agent._rejection_feedback(parent, tx, "cross_fit")
    assert "same stage were rejected" in feedback
    assert "outer holdout was transformed before freeze" in feedback
    assert "remove all outer holdout use from cross_fit" in feedback
    assert "already-passed protocol stage" in feedback


def test_rejection_feedback_accumulates_and_deduplicates_same_stage_history():
    duplicate = {
        "issue_code": "FIX_OUTER_IDS",
        "evidence": "use stable outer partition ids",
        "remediation": "keep canonical outer ids",
    }
    tx = {
        "history": [
            {"stage": "cross_fit", "status": "failed", "feedback": [duplicate]},
            {
                "stage": "cross_fit",
                "status": "failed",
                "feedback": [duplicate, {
                    "issue_code": "FIX_OOF",
                    "evidence": "write predictions by validation indices",
                    "remediation": "fill the aligned OOF matrix",
                }],
            },
            {
                "stage": "selection_freeze",
                "status": "failed",
                "feedback": [{"issue_code": "OTHER_STAGE", "evidence": "ignore", "remediation": "ignore"}],
            },
        ],
    }
    parent = SearchNode(code="", plan="failed", stage="debug", leakage_audit={"issues": []})
    feedback = protocol_repair_agent._rejection_feedback(parent, tx, "cross_fit")
    assert feedback.count("use stable outer partition ids") == 1
    assert "write predictions by validation indices" in feedback
    assert "OTHER_STAGE" not in feedback
    assert "Do not regress any previously repaired item" in feedback


def test_successful_candidate_does_not_consume_generation_failure_budget():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx-independent-budgets",
        "protocol_plan": {"stages": ["cross_fit"]},
        "current_stage_index": 0,
        "stage_attempts": {"cross_fit": 1},
        "stage_generation_attempts": {"cross_fit": 1},
        "stage_generation_failures": {},
        "history": [],
        "state": "pending",
        "max_attempts_per_stage": 2,
        "max_generation_attempts_per_stage": 2,
    }
    claimed = protocol_repair.begin_stage_generation(tx)
    retriable = protocol_repair.record_stage_generation_failure(claimed, "n1", "empty")
    assert retriable["state"] == "pending"
    assert retriable["stage_generation_attempts"] == {"cross_fit": 2}
    assert retriable["stage_generation_failures"] == {"cross_fit": 1}


def test_log_loss_logistic_regression_protocol_profile_is_classification():
    profile = protocol_repair.infer_task_profile(
        "A multiclass prediction task uses log loss",
        "model = LogisticRegression(); loss = CrossEntropyLoss()",
    )
    assert profile["objective"] == "classification"
    assert profile["split_family"] == "stratified"


@pytest.mark.parametrize("source_ids", ["train_sample_ids", "all_train_ids"])
def test_data_scope_accepts_stable_task_specific_id_names(source_ids):
    code = f"""
protocol_guard = ProtocolProvenanceGuard()
{source_ids} = train_df["id"].values
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
train_indices, holdout_indices = next(splitter.split({source_ids}, labels))
outer_train_ids = {source_ids}[train_indices]
outer_holdout_ids = {source_ids}[holdout_indices]
protocol_guard.register_partition("outer_train", outer_train_ids)
protocol_guard.register_partition("outer_holdout", outer_holdout_ids)
"""
    tx = {
        "protocol_plan": {
            "stages": ["data_scope"],
            "task_profile": {"objective": "classification", "grouped": False, "temporal": False},
            "capabilities": {},
        },
        "current_stage_index": 0,
    }
    audit = protocol_repair.audit_stage(code, tx)
    assert audit["status"] == "clean", audit["issues"]


def test_cross_fit_instructions_pin_runtime_provenance_api():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {
            "stages": ["cross_fit"],
            "task_profile": {"split_family": "stratified", "modality": "text", "objective": "classification"},
        },
        "current_stage_index": 0,
        "state": "pending",
    }
    instructions = "\n".join(protocol_repair.stage_instructions(tx))
    assert "ProtocolProvenanceGuard" in instructions
    assert "protocol_guard.record_prediction" in instructions
    assert 'purpose="oof"' in instructions
    assert "protocol_guard.record_fit" in instructions
    assert "inner_valid_ids" in instructions
    assert "early_stop_train_ids" in instructions
    assert "early_stop_eval_ids" in instructions
    assert "Do not tune fold-specific ensemble weights" in instructions
    assert "raw per-component OOF" in instructions


def test_selection_and_final_instructions_pin_runtime_calls():
    base = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {
            "stages": ["selection_freeze", "final_holdout"],
            "task_profile": {"split_family": "stratified", "modality": "text", "objective": "classification"},
        },
        "state": "pending",
    }
    selection = "\n".join(protocol_repair.stage_instructions({**base, "current_stage_index": 0}))
    assert "protocol_guard.record_selection" in selection
    assert "frozen_protocol_state" in selection
    assert "protocol_guard.freeze()" in selection
    assert "never substitute historical/fold-local weights or a no-op search loop" in selection
    final = "\n".join(protocol_repair.stage_instructions({**base, "current_stage_index": 1}))
    assert "protocol_guard.record_final_evaluation" in final
    assert "protocol_guard.assert_clean()" in final
    assert "protocol_guard.emit()" in final
    assert "before every outer-holdout feature extraction" in final
    assert "merely delaying the metric is not enough" in final
    final_tx = {**base, "current_stage_index": 1}
    assert protocol_repair._stage_attempt_limit(final_tx, "final_holdout") == 10
    assert protocol_repair._stage_attempt_limit(final_tx, "final_holdout", generation=True) == 10


def test_protocol_plan_is_generic_across_task_and_model_families():
    regression = protocol_repair.build_protocol_plan(
        "Tabular regression evaluated with RMSE",
        "model = RandomForestRegressor(); search = GridSearchCV(model, params)",
    )
    assert regression["task_profile"]["objective"] == "regression"
    assert regression["task_profile"]["split_family"] == "random"
    assert "cross_fit" in regression["stages"]
    assert "selection_freeze" in regression["stages"]

    grouped = protocol_repair.build_protocol_plan(
        "Medical image classification grouped by patient_id",
        "model = ConvNet(); splitter = GroupKFold()",
    )
    assert grouped["task_profile"]["modality"] == "image"
    assert grouped["task_profile"]["split_family"] == "grouped"

    temporal = protocol_repair.build_protocol_plan(
        "Time series forecasting with timestamp and MAE",
        "model = ForecastNet(); splitter = TimeSeriesSplit()",
    )
    assert temporal["task_profile"]["objective"] == "regression"
    assert temporal["task_profile"]["split_family"] == "time_ordered"

    fixed = protocol_repair.build_protocol_plan(
        "Binary classification",
        "model = FixedClassifier()",
    )
    assert "selection_freeze" not in fixed["stages"]
    fixed_tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": fixed,
        "current_stage_index": len(fixed["stages"]) - 1,
        "state": "final_pending",
    }
    instructions = "\n".join(protocol_repair.stage_instructions(fixed_tx))
    assert "fixed_protocol_state" in instructions
    assert 'record_prediction("final_predictor"' in instructions


def test_unknown_protocol_profile_fails_closed():
    with pytest.raises(ValueError, match="unsupported protocol objective"):
        protocol_repair.build_protocol_plan(
            "Discover latent structure without labels",
            "model = KMeans(n_clusters=4)",
        )

    node = SearchNode(code="model = KMeans(n_clusters=4)", plan="seed", stage="draft")
    node.leakage_audit = {
        "status": "protocol_biased",
        "issues": [{"issue_code": "LLM_SELECTION_BIAS", "category": "selection_bias"}],
    }
    agent = SimpleNamespace(
        task_desc="Discover latent structure without labels",
        acfg=SimpleNamespace(protocol_repair=SimpleNamespace(enabled=True)),
    )
    assert protocol_repair.ensure_transaction(agent, node) == {}
    assert node.leakage_audit["status"] == "blocked"
    assert node.leakage_audit["hard_block"] is True
    assert node.leakage_audit["rank_eligible"] is False
    assert node.leakage_audit["memory_disposition"] == "negative_only"
    assert node.leakage_audit["repair_mode"] == "unsupported_protocol"
    assert any(
        issue["issue_code"] == "UNSUPPORTED_PROTOCOL_PROFILE"
        for issue in node.leakage_audit["issues"]
    )


def test_stage_prompt_uses_dynamic_protected_components():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {
            "stages": ["cross_fit"],
            "task_profile": {"split_family": "random", "modality": "tabular", "objective": "regression"},
        },
        "preservation_contract": {"component_calls": {"CatBoostRegressor": 1, "StandardScaler": 1}},
        "current_stage_index": 0,
        "state": "pending",
    }
    instructions = "\n".join(protocol_repair.stage_instructions(tx))
    assert "CatBoostRegressor, StandardScaler" in instructions
    assert "DeBERTa" not in instructions
    assert "XGBoost" not in instructions


def test_selection_stage_accepts_materialized_compound_weight_names():
    code = """
oof_predictions = values
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
selected_ensemble_weights = optimize(oof_predictions)
frozen_ensemble_weights = selected_ensemble_weights.copy()
protocol_guard.record_selection("ensemble_weights", outer_train_ids)
protocol_guard.freeze()
"""
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["selection_freeze"], "task_profile": {}},
        "current_stage_index": 0,
        "state": "pending",
    }
    assert protocol_repair.audit_stage(code, tx)["status"] == "clean"


def test_selection_stage_accepts_generic_frozen_protocol_state():
    code = """
oof_predictions = values
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
selected_protocol_state = optimize(oof_predictions)
frozen_protocol_state = selected_protocol_state.copy()
protocol_guard.record_selection("protocol_state", outer_train_ids)
protocol_guard.freeze()
"""
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["selection_freeze"], "task_profile": {}},
        "current_stage_index": 0,
        "state": "pending",
    }
    assert protocol_repair.audit_stage(code, tx)["status"] == "clean"


def test_cross_fit_rejects_fold_weight_selection_before_global_oof():
    code = """
oof_predictions = np.zeros((len(outer_train_ids), 3))
for inner_train_idx, inner_valid_idx in KFold(5).split(outer_train_ids):
    inner_train_ids = outer_train_ids[inner_train_idx]
    inner_valid_ids = outer_train_ids[inner_valid_idx]
    model.fit(X[inner_train_idx], y[inner_train_idx])
    protocol_guard.record_fit("model", inner_train_ids)
    fold_predictions = model.predict_proba(X[inner_valid_idx])
    best_weights_fold = optimize(fold_predictions, y[inner_valid_idx])
    oof_predictions[inner_valid_idx] = blend(fold_predictions, best_weights_fold)
    protocol_guard.record_prediction("model", inner_train_ids, inner_valid_ids, purpose="oof")
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
"""
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["cross_fit"], "task_profile": {}},
        "current_stage_index": 0,
        "state": "pending",
    }
    audit = protocol_repair.audit_stage(code, tx)
    assert audit["status"] == "blocked"
    assert "before complete global OOF" in audit["issues"][0]["evidence"]


def test_selection_stage_rejects_noop_search_that_never_scores_oof():
    code = """
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
selected_ensemble_weights = {"a": 0.5, "b": 0.5}
for candidate in candidates:
    combined = candidate[0] * historical_weights[0] + candidate[1] * historical_weights[1]
protocol_guard.record_selection("ensemble_weights", outer_train_ids)
protocol_guard.freeze()
"""
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["selection_freeze"], "task_profile": {}},
        "current_stage_index": 0,
        "state": "pending",
    }
    audit = protocol_repair.audit_stage(code, tx)
    assert audit["status"] == "blocked"
    evidence = " ".join(issue["evidence"] for issue in audit["issues"])
    assert "does not compute a metric/search from OOF predictions" in evidence


def test_selection_stage_accepts_metric_on_candidate_derived_from_oof():
    code = """
protocol_guard.record_global_oof(oof_predictions_by_model, outer_train_ids)
best_ensemble_weights = None
best_oof_loss = float("inf")
for weights in candidates:
    candidate_predictions = blend(oof_predictions_by_model, weights)
    candidate_loss = log_loss(y_outer_train, candidate_predictions)
    if candidate_loss < best_oof_loss:
        best_oof_loss = candidate_loss
        best_ensemble_weights = weights
frozen_protocol_state = {"weights": best_ensemble_weights}
protocol_guard.record_selection("ensemble_weights", outer_train_ids)
protocol_guard.freeze()
"""
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["selection_freeze"], "task_profile": {}},
        "current_stage_index": 0,
        "state": "pending",
    }
    assert protocol_repair.audit_stage(code, tx)["status"] == "clean"


def test_selection_stage_accepts_manual_oof_logloss_formula():
    code = """
protocol_guard.record_global_oof(oof_lr_probs, outer_train_ids)
protocol_guard.record_global_oof(oof_xgb_probs, outer_train_ids)
protocol_guard.record_global_oof(oof_deberta_probs, outer_train_ids)
best_logloss = float("inf")
best_weights = None
for weights in candidates:
    ensemble_oof = blend(oof_deberta_probs, oof_xgb_probs, oof_lr_probs, weights)
    logloss = -np.mean(np.log(ensemble_oof[np.arange(len(y_outer_train)), y_outer_train]))
    if logloss < best_logloss:
        best_logloss = logloss
        best_weights = weights
frozen_protocol_state = {"weights": best_weights, "oof_logloss": best_logloss}
protocol_guard.record_selection("protocol_state", outer_train_ids)
protocol_guard.freeze()
"""
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["selection_freeze"], "task_profile": {}},
        "current_stage_index": 0,
        "state": "pending",
    }
    assert protocol_repair.audit_stage(code, tx)["status"] == "clean"


def test_selection_stage_traces_scalar_weight_assignments_from_oof_metric():
    code = """
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
best_avg_logloss = float("inf")
best_w1, best_w2, best_w3 = 0.4, 0.35, 0.25
for w1, w2 in candidates:
    w3 = 1.0 - w1 - w2
    ensemble_oof = w1 * oof_deberta + w2 * oof_xgboost + w3 * oof_logistic
    logloss = -np.mean(np.log(ensemble_oof[np.arange(len(y_outer_train)), y_outer_train]))
    if logloss < best_avg_logloss:
        best_avg_logloss = logloss
        best_w1 = w1
        best_w2 = w2
        best_w3 = w3
frozen_protocol_state = {"w1": best_w1, "w2": best_w2, "w3": best_w3}
protocol_guard.record_selection("protocol_state", outer_train_ids)
protocol_guard.freeze()
"""
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["selection_freeze"], "task_profile": {}},
        "current_stage_index": 0,
        "state": "pending",
    }
    assert protocol_repair.audit_stage(code, tx)["status"] == "clean"


def test_selection_stage_rejects_dummy_oof_metric_with_fixed_weights():
    code = """
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
dummy_oof_loss = log_loss(y_outer_train, oof_predictions)
selected_ensemble_weights = {"a": 0.5, "b": 0.5}
frozen_protocol_state = {"weights": selected_ensemble_weights}
protocol_guard.record_selection("ensemble_weights", outer_train_ids)
protocol_guard.freeze()
"""
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {"stages": ["selection_freeze"], "task_profile": {}},
        "current_stage_index": 0,
        "state": "pending",
    }
    audit = protocol_repair.audit_stage(code, tx)
    assert audit["status"] == "blocked"
    evidence = " ".join(issue["evidence"] for issue in audit["issues"])
    assert "not causally updated by the OOF metric" in evidence


def test_protocol_guard_normalization_removes_generated_shadow_class():
    code = """
class ProtocolProvenanceGuard:
    def freeze(self):
        pass

protocol_guard = ProtocolProvenanceGuard()
"""
    normalized = protocol_repair_agent._normalize_protocol_guard_calls(code)
    assert "class ProtocolProvenanceGuard" not in normalized
    assert "from agents.protocol_repair_runtime import ProtocolProvenanceGuard" in normalized
    compile(normalized, "<normalized-real-guard>", "exec")


def test_protocol_guard_aliases_are_normalized_to_runtime_api():
    code = """
from agents.protocol_repair_runtime import ProtocolProvenanceGuard
protocol_guard = ProtocolProvenanceGuard()
protocol_guard.register_outer_train(outer_train_ids)
protocol_guard.register_outer_holdout(outer_holdout_ids)
protocol_guard.record_outer_train(outer_train_ids)
protocol_guard.record_outer_holdout(outer_holdout_ids)
protocol_guard.register('fold_manual', manual_ids)
for fold, train_ids, valid_ids in folds:
    protocol_guard.register_fold(fold, train_ids, valid_ids)
"""
    normalized = protocol_repair_agent._normalize_protocol_guard_calls(code)
    assert "register_outer_train" not in normalized
    assert "register_outer_holdout" not in normalized
    assert "record_outer_train" not in normalized
    assert "record_outer_holdout" not in normalized
    assert "register_partition('fold_manual', manual_ids)" in normalized
    assert "register_fold" not in normalized
    assert "register_partition('outer_train', outer_train_ids)" in normalized
    assert "register_partition('outer_holdout', outer_holdout_ids)" in normalized
    assert "register_partition(f'fold_{fold}_train', train_ids)" in normalized
    assert "register_partition(f'fold_{fold}_valid', valid_ids)" in normalized

    global_oof_aliases = protocol_repair_agent._normalize_protocol_guard_calls(
        "protocol_guard.register_global_oof(oof_predictions, outer_train_ids)\n"
        "protocol_guard.record_global_oof_coverage(oof_predictions, outer_train_ids)\n"
    )
    assert "register_global_oof" not in global_oof_aliases
    assert "record_global_oof_coverage" not in global_oof_aliases
    assert global_oof_aliases.count("record_global_oof(") == 2
    alias_checks = protocol_repair_agent._normalize_protocol_guard_calls(
        "protocol_guard.verify_no_leak('outer_train', 'outer_holdout')\n"
        "protocol_guard.assert_no_overlap('fold_train', 'fold_valid')\n"
    )
    assert "verify_no_leak" not in alias_checks
    assert "assert_no_overlap" not in alias_checks
    assert "check_no_overlap('outer_train', 'outer_holdout')" in alias_checks
    compile(normalized, "<normalized-protocol-guard>", "exec")

    compact = protocol_repair_agent._normalize_protocol_guard_calls(
        "protocol_guard.register(sample_ids, outer_train_ids, outer_holdout_ids)\n"
    )
    assert "register(" not in compact
    assert "register_partition('outer_train', outer_train_ids)" in compact
    assert "register_partition('outer_holdout', outer_holdout_ids)" in compact

    outer_split = protocol_repair_agent._normalize_protocol_guard_calls(
        "protocol_guard.register_outer_split(sample_ids, train_ids, holdout_ids)\n"
    )
    assert "register_outer_split" not in outer_split
    assert "register_partition('outer_train', train_ids)" in outer_split
    assert "register_partition('outer_holdout', holdout_ids)" in outer_split

    compact_outer_split = protocol_repair_agent._normalize_protocol_guard_calls(
        "guard = ProtocolProvenanceGuard(all_ids)\n"
        "guard.record_outer_split(train_ids, holdout_ids)\n"
    )
    assert "ProtocolProvenanceGuard()" in compact_outer_split
    assert "register_partition('outer_train', train_ids)" in compact_outer_split
    assert "register_partition('outer_holdout', holdout_ids)" in compact_outer_split

    dataset_alias = protocol_repair_agent._normalize_protocol_guard_calls(
        "guard.register_dataset(train_ids, holdout_ids)\n"
    )
    assert "register_dataset" not in dataset_alias
    assert "register_partition('outer_train', train_ids)" in dataset_alias
    assert "register_partition('outer_holdout', holdout_ids)" in dataset_alias

    alternate = protocol_repair_agent._normalize_protocol_guard_calls(
        "protocol_guard.register_outer_partition(train_ids, purpose='train')\n"
        "protocol_guard.register_outer_partition(holdout_ids, purpose='holdout')\n"
        "protocol_guard.register_inner_split(inner_train_ids, inner_valid_ids, fold=fold)\n"
    )
    assert "register_outer_partition" not in alternate
    assert "register_inner_split" not in alternate
    assert "register_partition('outer_train', train_ids)" in alternate
    assert "register_partition('outer_holdout', holdout_ids)" in alternate
    assert "register_partition(f'fold_{fold}_train', inner_train_ids)" in alternate
    assert "register_partition(f'fold_{fold}_valid', inner_valid_ids)" in alternate


def test_protocol_guard_normalizes_reversed_partition_arguments():
    normalized = protocol_repair_agent._normalize_protocol_guard_calls(
        "protocol_guard.register_partition(outer_train_ids, 'outer_train')\n"
        "protocol_guard.register_partition(outer_holdout_ids, 'outer_holdout')\n"
    )
    assert "register_partition('outer_train', outer_train_ids)" in normalized
    assert "register_partition('outer_holdout', outer_holdout_ids)" in normalized


def test_canonical_outer_partitions_allow_dynamic_fold_names():
    tree = ast.parse("""
protocol_guard.register_partition("outer_train", outer_train_ids)
protocol_guard.register_partition("outer_holdout", outer_holdout_ids)
for fold in range(5):
    protocol_guard.register_partition(f"fold_{fold}_train", inner_train_ids)
    protocol_guard.register_partition(f"fold_{fold}_valid", inner_valid_ids)
""")
    assert protocol_repair._canonical_partition_failures(tree) == []


def test_cross_fit_rejects_preprocessor_fitted_on_all_outer_train_rows():
    code = """
protocol_guard.register_partition("outer_train", outer_train_ids)
protocol_guard.register_partition("outer_holdout", outer_holdout_ids)
skf = StratifiedKFold(n_splits=5)
word_vectorizer = TfidfVectorizer()
outer_features = word_vectorizer.fit_transform(X_outer_train)
for fold, (inner_train_idx, inner_valid_idx) in enumerate(skf.split(outer_train_ids, y)):
    inner_train_ids = outer_train_ids[inner_train_idx]
    inner_valid_ids = outer_train_ids[inner_valid_idx]
    model.fit(outer_features[inner_train_idx], y[inner_train_idx])
    oof_predictions[inner_valid_idx] = model.predict_proba(outer_features[inner_valid_idx])
    protocol_guard.record_fit("model", inner_train_ids)
    protocol_guard.record_prediction("model", inner_train_ids, inner_valid_ids, purpose="oof")
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
"""
    tx = {
        "protocol_plan": {
            "stages": ["cross_fit"],
            "task_profile": {},
            "capabilities": {"has_stateful_preprocessing": True},
        },
        "current_stage_index": 0,
    }
    audit = protocol_repair.audit_stage(code, tx)
    assert audit["status"] == "blocked"
    assert "outside the fold loop" in " ".join(
        issue["evidence"] for issue in audit["issues"]
    )


def test_cross_fit_accepts_fold_local_preprocessor_with_provenance():
    code = """
protocol_guard.register_partition("outer_train", outer_train_ids)
protocol_guard.register_partition("outer_holdout", outer_holdout_ids)
skf = StratifiedKFold(n_splits=5)
for fold, (inner_train_idx, inner_valid_idx) in enumerate(skf.split(outer_train_ids, y)):
    inner_train_ids = outer_train_ids[inner_train_idx]
    inner_valid_ids = outer_train_ids[inner_valid_idx]
    word_vectorizer = TfidfVectorizer()
    train_features = word_vectorizer.fit_transform(X_outer_train[inner_train_idx])
    valid_features = word_vectorizer.transform(X_outer_train[inner_valid_idx])
    model.fit(train_features, y[inner_train_idx])
    oof_predictions[inner_valid_idx] = model.predict_proba(valid_features)
    protocol_guard.record_fit("word_vectorizer", inner_train_ids, purpose="fold_preprocess")
    protocol_guard.record_fit("model", inner_train_ids)
    protocol_guard.record_prediction("model", inner_train_ids, inner_valid_ids, purpose="oof")
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
"""
    tx = {
        "protocol_plan": {
            "stages": ["cross_fit"],
            "task_profile": {},
            "capabilities": {"has_stateful_preprocessing": True},
        },
        "current_stage_index": 0,
    }
    assert protocol_repair.audit_stage(code, tx)["status"] == "clean"


def test_cross_fit_rejects_cv_folds_disguised_as_outer_holdouts():
    code = """
protocol_guard = ProtocolProvenanceGuard()
oof_predictions = np.zeros((len(sample_ids), 3))
skf = StratifiedKFold(n_splits=5)
for outer_train_idx, outer_holdout_idx in skf.split(sample_ids, labels):
    outer_train_ids = sample_ids[outer_train_idx]
    outer_holdout_ids = sample_ids[outer_holdout_idx]
    protocol_guard.register_partition(f"fold_{fold}_outer_train", outer_train_ids)
    protocol_guard.register_partition(f"fold_{fold}_outer_holdout", outer_holdout_ids)
    model.fit(X[outer_train_idx], labels[outer_train_idx])
    oof_predictions[outer_holdout_idx] = model.predict_proba(X[outer_holdout_idx])
    protocol_guard.record_fit("model", outer_train_ids)
    protocol_guard.record_prediction("model", outer_train_ids, outer_holdout_ids, purpose="oof")
protocol_guard.record_global_oof(oof_predictions, sample_ids)
"""
    tx = {
        "protocol_plan": {
            "stages": ["data_scope", "cross_fit"],
            "task_profile": {},
            "capabilities": {"has_stateful_preprocessing": False},
        },
        "current_stage_index": 1,
    }
    audit = protocol_repair.audit_stage(code, tx)
    evidence = " ".join(issue["evidence"] for issue in audit["issues"])
    assert audit["status"] == "blocked"
    assert "canonical outer_train and outer_holdout" in evidence
    assert "reassigned inside a fold loop" in evidence
    assert "outer_train_ids" in evidence


def test_cross_fit_rejects_outer_holdout_use_before_final_stage():
    code = """
protocol_guard = ProtocolProvenanceGuard()
outer_train_ids, outer_holdout_ids = train_test_split(sample_ids)
protocol_guard.register_partition("outer_train", outer_train_ids)
protocol_guard.register_partition("outer_holdout", outer_holdout_ids)
oof_predictions = np.zeros((len(outer_train_ids), 3))
skf = KFold(n_splits=5)
for inner_train_idx, inner_valid_idx in skf.split(outer_train_ids):
    inner_train_ids = outer_train_ids[inner_train_idx]
    inner_valid_ids = outer_train_ids[inner_valid_idx]
    model.fit(X[inner_train_idx], y[inner_train_idx])
    oof_predictions[inner_valid_idx] = model.predict_proba(X[inner_valid_idx])
    forbidden = model.predict_proba(X_outer_holdout)
    protocol_guard.record_fit("model", inner_train_ids)
    protocol_guard.record_prediction("model", inner_train_ids, inner_valid_ids, purpose="oof")
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
"""
    tx = {
        "protocol_plan": {
            "stages": ["data_scope", "cross_fit"],
            "task_profile": {},
            "capabilities": {"has_stateful_preprocessing": False},
        },
        "current_stage_index": 1,
    }
    audit = protocol_repair.audit_stage(code, tx)
    assert audit["status"] == "blocked"
    assert "outer_holdout is consumed" in " ".join(
        issue["evidence"] for issue in audit["issues"]
    )


def test_cross_fit_accepts_fixed_outer_split_and_inner_oof_scope():
    code = """
protocol_guard = ProtocolProvenanceGuard()
outer_train_ids, outer_holdout_ids = train_test_split(sample_ids)
protocol_guard.register_partition("outer_train", outer_train_ids)
protocol_guard.register_partition("outer_holdout", outer_holdout_ids)
oof_predictions = np.zeros((len(outer_train_ids), 3))
skf = KFold(n_splits=5)
for inner_train_idx, inner_valid_idx in skf.split(outer_train_ids):
    inner_train_ids = outer_train_ids[inner_train_idx]
    inner_valid_ids = outer_train_ids[inner_valid_idx]
    model.fit(X_outer_train[inner_train_idx], y_outer_train[inner_train_idx])
    oof_predictions[inner_valid_idx] = model.predict_proba(X_outer_train[inner_valid_idx])
    protocol_guard.record_fit("model", inner_train_ids)
    protocol_guard.record_prediction("model", inner_train_ids, inner_valid_ids, purpose="oof")
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
"""
    tx = {
        "protocol_plan": {
            "stages": ["data_scope", "cross_fit"],
            "task_profile": {},
            "capabilities": {"has_stateful_preprocessing": False},
        },
        "current_stage_index": 1,
    }
    assert protocol_repair.audit_stage(code, tx)["status"] == "clean"


def _complete_final_holdout_program(*, premature_holdout_metric: bool = False) -> str:
    premature = (
        "premature_loss = log_loss(y_outer_holdout, leaked_holdout_predictions)"
        if premature_holdout_metric
        else ""
    )
    return f"""
from agents.protocol_repair_runtime import ProtocolProvenanceGuard
protocol_guard = ProtocolProvenanceGuard()
outer_train_ids, outer_holdout_ids = train_test_split(sample_ids)
protocol_guard.register_partition("outer_train", outer_train_ids)
protocol_guard.register_partition("outer_holdout", outer_holdout_ids)
oof_predictions = np.zeros((len(outer_train_ids), 3))
skf = KFold(n_splits=5)
for inner_train_idx, inner_valid_idx in skf.split(outer_train_ids):
    inner_train_ids = outer_train_ids[inner_train_idx]
    inner_valid_ids = outer_train_ids[inner_valid_idx]
    model.fit(X_outer_train[inner_train_idx], y_outer_train[inner_train_idx])
    oof_predictions[inner_valid_idx] = model.predict_proba(X_outer_train[inner_valid_idx])
    protocol_guard.record_fit("model", inner_train_ids)
    protocol_guard.record_prediction("model", inner_train_ids, inner_valid_ids, purpose="oof")
protocol_guard.record_global_oof(oof_predictions, outer_train_ids)
selected_protocol_state = optimize(oof_predictions)
frozen_protocol_state = selected_protocol_state.copy()
protocol_guard.record_selection("protocol_state", outer_train_ids)
{premature}
protocol_guard.freeze()
final_model.fit(X_outer_train, y_outer_train)
protocol_guard.record_fit("final_model", outer_train_ids)
final_predictions = final_model.predict_proba(X_outer_holdout)
protocol_guard.record_prediction("final_predictor", outer_train_ids, outer_holdout_ids, purpose="final")
final_metric = log_loss(y_outer_holdout, final_predictions)
protocol_guard.record_final_evaluation(outer_holdout_ids)
protocol_guard.assert_clean()
protocol_guard.emit()
"""


def _final_holdout_transaction() -> dict:
    return {
        "protocol_plan": {
            "stages": [
                "data_scope", "cross_fit", "selection_freeze", "final_holdout"
            ],
            "task_profile": {},
            "capabilities": {"has_stateful_preprocessing": False},
        },
        "current_stage_index": 3,
    }


def test_final_holdout_allows_one_post_freeze_evaluation():
    audit = protocol_repair.audit_stage(
        _complete_final_holdout_program(), _final_holdout_transaction()
    )
    assert audit["status"] == "clean", audit["issues"]


@pytest.mark.parametrize(
    "split_input",
    ["outer_train_ids", "outer_train_indices", "outer_train_texts", "np.arange(N_outer_train)"],
)
def test_cross_fit_splitter_accepts_equivalent_outer_train_views(split_input):
    code = f"""
skf = StratifiedKFold(n_splits=5)
for text in outer_train_texts:
    words = text.split()
for fold, (train_idx, valid_idx) in enumerate(skf.split({split_input}, y_outer_train)):
    pass
"""
    failures = protocol_repair._cross_fit_scope_failures(ast.parse(code))
    assert not any("cross-fit splitter" in failure for failure in failures)


def test_cross_fit_splitter_rejects_full_dataset_ids():
    code = """
skf = StratifiedKFold(n_splits=5)
for train_idx, valid_idx in skf.split(sample_ids, labels):
    pass
"""
    failures = protocol_repair._cross_fit_scope_failures(ast.parse(code))
    assert "cross-fit splitter consumes rows outside outer_train" in failures


def test_final_holdout_rejects_holdout_metric_before_freeze():
    audit = protocol_repair.audit_stage(
        _complete_final_holdout_program(premature_holdout_metric=True),
        _final_holdout_transaction(),
    )
    assert audit["status"] == "blocked"
    assert "before protocol freeze" in " ".join(
        issue["evidence"] for issue in audit["issues"]
    )


def test_final_holdout_rejects_selection_state_change_after_freeze():
    code = _complete_final_holdout_program().replace(
        "final_model.fit(X_outer_train, y_outer_train)",
        "selected_protocol_state = {'weights': [1.0]}\nfinal_model.fit(X_outer_train, y_outer_train)",
    )
    audit = protocol_repair.audit_stage(code, _final_holdout_transaction())
    assert audit["status"] == "blocked"
    assert "modified after protocol freeze" in " ".join(
        issue["evidence"] for issue in audit["issues"]
    )


def test_model_literal_anchor_does_not_confuse_identifiers_with_literals():
    code = "deberta_predictions = None\nxgboost_predictions = None\n"
    anchored = protocol_repair_agent._anchor_missing_model_literals(
        code,
        {"model_literals": ["deberta", "xgboost"]},
    )
    tree = __import__("ast").parse(anchored)
    literals = {
        node.value for node in __import__("ast").walk(tree)
        if isinstance(node, __import__("ast").Constant) and isinstance(node.value, str)
    }
    assert {"deberta", "xgboost"}.issubset(literals)


def test_isolated_executor_can_import_protocol_runtime(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from agents.protocol_repair_runtime import ProtocolProvenanceGuard; "
            "print(ProtocolProvenanceGuard.schema)",
        ],
        cwd=tmp_path,
        env=executor._execution_environment(),
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "mlevolve_protocol_provenance_v1" in result.stdout


def test_protocol_runtime_records_are_read_only_diagnostics():
    guard = ProtocolProvenanceGuard()
    guard.register_partition("outer_train", [1, 2])
    guard.register_partition("outer_holdout", [3])
    guard.record_fit("model", [1, 2])
    guard.record_prediction("model", [1, 2], [3], purpose="final")
    guard.record_selection("weights", [1, 2])
    guard.freeze()
    guard.record_final_evaluation([3])
    kinds = [record["kind"] for record in guard.records]
    assert kinds == ["fit", "prediction", "selection", "final_evaluation"]


def test_protocol_runtime_global_oof_checks_row_coverage():
    clean = ProtocolProvenanceGuard()
    clean.register_partition("outer_train", ["a", "b"])
    clean.register_partition("outer_holdout", ["c"])
    clean.record_global_oof([[0.2], [0.8]], ["a", "b"])
    assert clean.violations == []

    mismatched = ProtocolProvenanceGuard()
    mismatched.record_global_oof([[0.2]], ["a", "b"])
    assert mismatched.violations == [
        "global OOF prediction coverage does not match sample IDs"
    ]


def test_protocol_runtime_rejects_partial_duplicate_or_unguarded_oof_selection():
    partial = ProtocolProvenanceGuard()
    partial.register_partition("outer_train", ["a", "b", "c"])
    partial.register_partition("outer_holdout", ["d"])
    partial.record_prediction("model", ["b", "c"], ["a"], purpose="oof")
    partial.record_prediction("model", ["a", "c"], ["a"], purpose="oof")
    partial.record_global_oof([[0.1], [0.2]], ["a", "b"])
    partial.record_selection("hyperparameters", ["a", "b"])
    partial.freeze()
    partial.record_prediction("model", ["a", "b", "c"], ["d"], purpose="final")
    partial.record_final_evaluation(["d"])
    with pytest.raises(RuntimeError) as exc:
        partial.assert_clean()
    message = str(exc.value)
    assert "duplicated" in message
    assert "do not exactly cover outer_train" in message
    assert "selection scope is not exactly outer_train" in message
    assert "without complete global OOF evidence" in message


def test_protocol_runtime_allows_fixed_design_without_cross_fit():
    guard = ProtocolProvenanceGuard()
    guard.register_partition("outer_train", [1, 2])
    guard.register_partition("outer_holdout", [3])
    guard.record_fit("fixed_model", [1, 2])
    guard.record_selection("fixed_protocol_state", [1, 2])
    guard.freeze()
    guard.record_prediction("fixed_model", [1, 2], [3], purpose="final")
    guard.record_final_evaluation([3])
    guard.assert_clean()


def test_runtime_audit_requires_global_oof_only_when_plan_cross_fits():
    payload = {
        "schema": "mlevolve_protocol_provenance_v1",
        "status": "clean",
        "violations": [],
        "counts": {
            "partitions": 2,
            "fits": 1,
            "predictions": 1,
            "selections": 1,
            "final_evaluations": 1,
            "global_oof": 0,
        },
    }
    marker = protocol_repair.RUNTIME_MARKER + json.dumps(payload, sort_keys=True)
    fixed = protocol_repair.runtime_provenance_audit(
        marker,
        {"protocol_plan": {"stages": ["data_scope", "final_holdout"]}},
    )
    assert fixed["status"] == "clean"
    cross_fit = protocol_repair.runtime_provenance_audit(
        marker,
        {"protocol_plan": {"stages": ["data_scope", "cross_fit", "final_holdout"]}},
    )
    assert cross_fit["status"] == "blocked"
    assert "global OOF" in cross_fit["reason"]


def test_validation_provenance_instructions_pin_inner_ids():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {
            "stages": ["validation_provenance"],
            "task_profile": {"split_family": "stratified", "modality": "text", "objective": "classification"},
        },
        "current_stage_index": 0,
        "state": "pending",
    }
    instructions = "\n".join(protocol_repair.stage_instructions(tx))
    assert "inner_train_ids" in instructions
    assert "inner_valid_ids" in instructions


def test_validation_stage_defers_oof_requirement_to_cross_fit():
    code = "inner_train_ids = ids[:8]\ninner_valid_ids = ids[8:]\n"
    validation_tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "protocol_plan": {
            "stages": ["validation_provenance", "cross_fit"],
            "capabilities": {"has_early_stopping": True, "has_ensemble": True},
            "task_profile": {},
        },
        "current_stage_index": 0,
        "state": "pending",
    }
    assert protocol_repair.audit_stage(code, validation_tx)["status"] == "clean"
    cross_fit_tx = {**validation_tx, "current_stage_index": 1}
    audit = protocol_repair.audit_stage(code, cross_fit_tx)
    assert audit["status"] == "blocked"
    assert any("OOF" in issue["evidence"] or "prediction provenance" in issue["evidence"] for issue in audit["issues"])


def test_duplicate_inflight_repair_waits_without_falling_back_to_uct(monkeypatch):
    root = SearchNode(code="", plan="root", stage="root", step=0)
    parent = SearchNode(
        code="source", plan="repair", stage="draft", parent=root,
        leakage_audit={"status": "blocked", "repair_required": True},
        audit_repair_required=True,
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.journal = Journal(nodes=[root, parent])
    agent.data_preview = "ready"
    agent.search_start_time = 1.0
    AgentSearch._init_mandatory_repair_scheduler(agent)
    agent._mandatory_repair_inflight_ids.add(parent.id)
    selected = []
    monkeypatch.setattr(
        "engine.agent_search.node_selection.select_with_soft_switch",
        lambda _agent: selected.append(True) or root,
    )

    result = AgentSearch.step(agent, parent, exec_callback=lambda *_args: None)
    assert result is None
    assert selected == []


def test_uct_excludes_repair_only_children():
    from engine import node_selection

    root = SearchNode(code="", plan="root", stage="root", step=0)
    repairs = [
        SearchNode(
            code=f"repair-{index}", plan=f"repair-{index}", stage="draft",
            parent=root, audit_repair_required=True,
            protocol_repair={
                "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
                "protocol_plan": {"stages": ["data_scope"]},
                "current_stage_index": 0,
                "state": "pending",
            },
        )
        for index in range(3)
    ]
    normal = SearchNode(
        code="normal", plan="normal", stage="draft", parent=root,
        is_buggy=False,
    )
    agent = SimpleNamespace(
        cfg=SimpleNamespace(agent=SimpleNamespace(
            decay=SimpleNamespace(
                phase_ratios=[0.2, 0.7], exploration_constant=1.414,
                alpha=0.01, lower_bound=0.7,
            )
        )),
        scfg=SimpleNamespace(num_drafts=2, num_improves=2),
        acfg=SimpleNamespace(
            steps=80,
            draft_role_policy=SimpleNamespace(enabled=True),
        ),
        current_step=1,
        is_root=lambda node: node is root,
    )

    selected = node_selection.select(agent, root)
    assert selected is normal
    assert selected not in repairs


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
        "active_stage": "data_scope",
        "active_generation_attempt": 1,
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
    assert "active_stage" not in parent.protocol_repair
    assert "active_generation_attempt" not in parent.protocol_repair
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
    assert "active_stage" not in active_parent.protocol_repair
    assert "active_generation_attempt" not in active_parent.protocol_repair
    assert active_parent.is_terminal is True


def test_concurrent_parent_release_is_idempotent():
    tx = {
        "schema": protocol_repair.PROTOCOL_REPAIR_SCHEMA,
        "transaction_id": "tx-concurrent-release",
        "protocol_plan": {"stages": ["data_scope", "final_holdout"]},
        "current_stage_index": 0,
        "state": "stage_in_progress",
        "active_stage": "data_scope",
        "active_generation_attempt": 1,
    }
    parent = SearchNode(
        code="parent", plan="parent", stage="draft", protocol_repair=tx,
        leakage_audit={"status": "blocked", "repair_required": True},
        audit_repair_required=True,
    )
    child = SearchNode(
        code="child", plan="child", stage="debug", parent=parent,
        protocol_repair={**tx, "state": "pending"},
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.journal = Journal(nodes=[parent, child])
    AgentSearch._init_mandatory_repair_scheduler(agent)
    agent._mandatory_repair_inflight_ids.add(parent.id)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(
            lambda _index: AgentSearch._release_mandatory_repair_parent(agent, parent),
            range(8),
        ))

    assert parent.protocol_repair["state"] == "superseded"
    assert parent.protocol_repair["successor_node_id"] == child.id
    assert "active_stage" not in parent.protocol_repair
    assert "active_generation_attempt" not in parent.protocol_repair
    assert parent.is_terminal is True
    assert not agent._mandatory_repair_inflight_ids
    assert not agent._mandatory_repair_queue


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


def test_preservation_allows_loading_persisted_protocol_artifacts():
    source = "label_encoder = LabelEncoder()\n"
    contract = build_repair_preservation_contract(source)
    repaired = """
label_encoder = LabelEncoder()
loaded_label_encoder = joblib.load(label_encoder_path)
loaded_model = torch.load(checkpoint_path)
"""
    audit = audit_repair_preservation(repaired, contract)
    assert audit["status"] == "clean", audit["issues"]
    assert "load" not in build_repair_preservation_contract(repaired)["component_calls"]


@pytest.mark.parametrize(
    "stage",
    ["data_scope", "validation_provenance", "cross_fit", "selection_freeze", "final_holdout"],
)
def test_every_protocol_stage_injects_its_latest_rejection(stage):
    tx = {
        "history": [{
            "stage": stage,
            "status": "failed",
            "feedback": [{
                "issue_code": f"FAILED_{stage.upper()}",
                "evidence": f"{stage} concrete rejection",
                "remediation": f"repair only {stage}",
            }],
        }],
    }
    parent = SearchNode(
        code="", plan="failed", stage="debug",
        leakage_audit={"status": "blocked", "issues": []},
    )
    feedback = protocol_repair_agent._rejection_feedback(parent, tx, stage)
    assert f"{stage} concrete rejection" in feedback
    assert f"repair only {stage}" in feedback


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


def test_preservation_rejects_dead_original_plus_new_or_reconfigured_model():
    source = """
checkpoint = "microsoft/deberta-v3-large"
model = LogisticRegression(C=2.0)
"""
    contract = build_repair_preservation_contract(source)
    repaired = source + """
replacement = LogisticRegression(C=0.1)
extra_model = XGBClassifier(n_estimators=10)
extra_checkpoint = "vendor/distilbert-small"
"""
    audit = audit_repair_preservation(repaired, contract)
    codes = {issue["issue_code"] for issue in audit["issues"]}
    assert "REPAIR_MODEL_COMPONENT_ADDED" in codes
    assert "REPAIR_UNAPPROVED_MODEL_CONFIGURATION_ADDED" in codes
    assert "REPAIR_MODEL_IDENTITY_ADDED" in codes


def test_preservation_allows_exact_constructor_duplicates_for_folds_and_refit():
    source = "model = LogisticRegression(C=2.0)\n"
    contract = build_repair_preservation_contract(source)
    repaired = source + "fold_model = LogisticRegression(C=2.0)\n"
    assert audit_repair_preservation(repaired, contract)["status"] == "clean"


def test_preservation_allows_fold_local_checkpoint_artifacts_without_new_model_identity():
    source = """
checkpoint = "microsoft/deberta-v3-large"
checkpoint_path = "./working/best_deberta_model.pt"
model = LogisticRegression(C=2.0)
"""
    contract = build_repair_preservation_contract(source)
    repaired = source + """
fold_checkpoint = "./working/best_deberta_model_fold_2.pt"
refit_checkpoint = "/tmp/working/best_deberta_model_epoch-4.ckpt"
"""
    assert audit_repair_preservation(repaired, contract)["status"] == "clean"


def test_preservation_still_rejects_new_vendor_checkpoint_identity():
    source = """
checkpoint = "microsoft/deberta-v3-large"
model = LogisticRegression(C=2.0)
"""
    contract = build_repair_preservation_contract(source)
    repaired = source + 'replacement_checkpoint = "vendor/modernbert-large"\n'
    audit = audit_repair_preservation(repaired, contract)
    assert "REPAIR_MODEL_IDENTITY_ADDED" in {
        issue["issue_code"] for issue in audit["issues"]
    }

    suffixed = source + 'replacement_checkpoint = "vendor/modernbert-large.pt"\n'
    suffixed_audit = audit_repair_preservation(suffixed, contract)
    assert "REPAIR_MODEL_IDENTITY_ADDED" in {
        issue["issue_code"] for issue in suffixed_audit["issues"]
    }


def test_preservation_ignores_model_names_used_only_in_logs():
    source = '''
checkpoint = "microsoft/deberta-v3-large"
model = LogisticRegression(C=2.0)
'''
    contract = build_repair_preservation_contract(source)
    repaired = source + '''
print(f"  DeBERTa: {best_w1}, XGBoost: {best_w2}")
logger.info("ModernBERT training finished")
'''
    current = build_repair_preservation_contract(repaired)
    assert "  DeBERTa: " not in current["model_literals"]
    assert "ModernBERT training finished" not in current["model_literals"]
    assert audit_repair_preservation(repaired, contract)["status"] == "clean"


def test_preservation_keeps_direct_model_loader_literals():
    source = 'model = AutoModel.from_pretrained("vendor/deberta-large")\n'
    contract = build_repair_preservation_contract(source)
    assert contract["model_literals"] == ["vendor/deberta-large"]
    repaired = 'model = AutoModel.from_pretrained("vendor/modernbert-large")\n'
    audit = audit_repair_preservation(repaired, contract)
    codes = {issue["issue_code"] for issue in audit["issues"]}
    assert "REPAIR_MODEL_IDENTITY_CHANGED" in codes
    assert "REPAIR_MODEL_IDENTITY_ADDED" in codes


def test_preservation_allows_new_runtime_provenance_component_labels():
    source = """
checkpoint = "microsoft/deberta-v3-large"
model = LogisticRegression(C=2.0)
"""
    contract = build_repair_preservation_contract(source)
    repaired = source + """
protocol_guard.record_fit("Deberta", fold_train_ids, purpose="train")
protocol_guard.record_prediction("Deberta", fold_train_ids, fold_valid_ids, purpose="oof")
"""
    current = build_repair_preservation_contract(repaired)
    assert current["protocol_component_labels"] == ["Deberta"]
    assert {"Deberta", "oof", "train"} <= set(current["protocol_metadata_literals"])
    assert audit_repair_preservation(repaired, contract)["status"] == "clean"


def test_preservation_allows_fold_varying_runtime_provenance_labels():
    source = '''
checkpoint = "microsoft/deberta-v3-large"
model = LogisticRegression(C=2.0)
'''
    contract = build_repair_preservation_contract(source)
    repaired = source + '''
for fold in range(5):
    protocol_guard.record_fit(f"deberta_fold{fold}", fold_train_ids, purpose=f"fold_{fold}_training")
    protocol_guard.record_prediction(f"deberta_fold{fold}", fold_train_ids, fold_valid_ids, purpose="oof")
'''
    current = build_repair_preservation_contract(repaired)
    assert {"deberta_fold", "fold_", "_training"} <= set(
        current["protocol_metadata_literals"]
    )
    assert audit_repair_preservation(repaired, contract)["status"] == "clean"


def test_preservation_allows_protocol_purpose_labels_and_local_prediction_artifacts():
    source = """
checkpoint = "microsoft/deberta-v3-large"
model = LogisticRegression(C=2.0)
"""
    contract = build_repair_preservation_contract(source)
    repaired = source + """
protocol_guard.record_fit("AdamW", fold_train_ids, purpose="deberta_optimizer_fold")
protocol_guard.record_global_oof("deberta_fold", outer_train_ids)
np.save("./working/holdout_deberta.npy", holdout_predictions)
"""
    assert audit_repair_preservation(repaired, contract)["status"] == "clean"
