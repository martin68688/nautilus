from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from agents.memory.prospective_audit import ProspectiveAuditLogger


class _Protocol:
    def key(self) -> str:
        return "protocol@1#" + "a" * 64


def _agent(tmp_path: Path):
    authority = SimpleNamespace(
        run_id="run-1",
        task_id="denoising-dirty-documents",
        active_protocol=_Protocol(),
        record_prospective_counterfactual=lambda *_args, **_kwargs: SimpleNamespace(
            receipt_id="host-counterfactual-receipt-1"
        ),
    )
    cfg = SimpleNamespace(
        log_dir=tmp_path,
        agent=SimpleNamespace(seed=11),
        prospective_audit=SimpleNamespace(
            enabled=True,
            allow_pending_counterfactual=True,
        ),
    )
    return SimpleNamespace(
        cfg=cfg,
        evaluation_authority=authority,
        acfg=SimpleNamespace(
            code=SimpleNamespace(model="test-model", temp=0.2)
        ),
    )


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_raw_opportunity_precedes_final_decision_and_binds_runtime_receipts(tmp_path: Path):
    logger = ProspectiveAuditLogger(_agent(tmp_path))
    protocol_ref = "protocol@1#" + "a" * 64
    trace = {
        "request": {
            "operation": "generate_candidate",
            "generation_stage": "draft",
        },
        "raw_shadow_authority_decisions": [
            {
                "clause_id": "clause-valid",
                "claim_id": "claim-valid",
                "claim_type": "method_hypothesis",
                "outcome": "allow",
                "protocol_ref": protocol_ref,
            },
            {
                "clause_id": "clause-invalid",
                "claim_id": "claim-invalid",
                "claim_type": "score",
                "outcome": "deny",
                "protocol_ref": protocol_ref,
                "reason_codes": ["missing_receipt"],
                "blocking_receipts": [],
            },
        ],
        "suppressed_clause_refs": ["clause-invalid"],
        "effective_visible_clause_ids": ["clause-valid"],
        "clause_decisions": {
            "clause-valid": {"allowed": True},
            "clause-invalid": {
                "allowed": False,
                "reason": "missing_receipt",
                "authority_decision_refs": ["authority-deny-1"],
            },
        },
    }
    pack = SimpleNamespace(
        request_id="visibility-request-1",
        visibility_trace=trace,
    )
    decision_id = logger.record_visibility(pack, object(), source="test")
    assert decision_id
    opportunities = _rows(tmp_path / "decision_opportunities.jsonl")
    assert len(opportunities) == 1
    assert opportunities[0]["raw_logged_before_filtering"] is True
    assert not (tmp_path / "prospective_decision_ledger.jsonl").exists()

    node = SimpleNamespace(
        id="node-1",
        stage="draft",
        draft_role="memory_transfer",
        plan="Use a U-Net.",
        code="print('ok')\n",
        prompt_input="safe prompt",
        receipt_refs=["runtime-receipt-1"],
    )
    logger.bind_thread_to_node(node)
    logger.finalize_node(node)
    decisions = _rows(tmp_path / "prospective_decision_ledger.jsonl")
    assert len(decisions) == 1
    row = decisions[0]
    assert row["decision_id"] == decision_id
    assert row["raw_candidate_ids"] == ["clause-invalid", "clause-valid"]
    assert {
        (decision["candidate_id"], decision["claim_id"])
        for decision in row["shadow_authority_decisions"]
    } == {
        ("clause-invalid", "claim-invalid"),
        ("clause-valid", "claim-valid"),
    }
    assert row["suppressed_candidate_ids"] == ["clause-invalid"]
    assert row["final_prompt_candidate_ids"] == ["clause-valid"]
    assert len(row["actual_action_hash"]) == 64
    assert len(row["actual_code_hash"]) == 64
    assert row["counterfactual_status"] == "pending"
    assert row["counterfactual_action_hash"] == ""
    assert "runtime-receipt-1" in row["runtime_receipt_refs"]
    assert row["suppression_reasons"]["clause-invalid"]["receipt_refs"]


def test_no_suppression_uses_identity_counterfactual(tmp_path: Path):
    logger = ProspectiveAuditLogger(_agent(tmp_path))
    pack = {
        "stage_route": {"stage": "debug"},
        "visibility_trace": {
            "request": {"operation": "debug_hypothesis", "generation_stage": "debug"}
        },
        "pre_gate_raw_candidates": [
            {
                "candidate_id": "run-node-1",
                "score": 0.75,
                "operation_authorized": True,
                "gate_reason": "clean_successful_run_node",
            }
        ],
        "final_prompt_candidate_ids": ["run-node-1"],
    }
    nodes = {
        "run-node-1": {
            "metric": 0.1,
            "is_buggy": False,
            "is_valid": True,
            "leakage_audit": {"issues": []},
        }
    }
    logger.record_run_candidates(pack, nodes)
    node = SimpleNamespace(
        id="node-2",
        stage="debug",
        draft_role="memory_transfer",
        plan="Repair.",
        code="print('fixed')\n",
        prompt_input="prompt",
        receipt_refs=[],
    )
    logger.bind_thread_to_node(node)
    logger.finalize_node(node)
    row = _rows(tmp_path / "prospective_decision_ledger.jsonl")[0]
    assert row["counterfactual_status"] == "identity"
    assert row["counterfactual_action_hash"] == row["actual_action_hash"]
    assert row["counterfactual_code_hash"] == row["actual_code_hash"]


def test_suppressed_claims_generate_host_receipted_counterfactual(
    tmp_path: Path, monkeypatch
):
    logger = ProspectiveAuditLogger(_agent(tmp_path))
    protocol_ref = "protocol@1#" + "a" * 64
    clause = SimpleNamespace(text="Use the suppressed raw method.")
    pack = SimpleNamespace(
        request_id="visibility-request-counterfactual",
        visibility_trace={
            "request": {
                "operation": "generate_candidate",
                "generation_stage": "draft",
            },
            "raw_shadow_authority_decisions": [
                {
                    "clause_id": "clause-invalid",
                    "claim_id": "claim-invalid",
                    "claim_type": "method_hypothesis",
                    "outcome": "deny",
                    "protocol_ref": protocol_ref,
                }
            ],
            "suppressed_clause_refs": ["clause-invalid"],
            "effective_visible_clause_ids": [],
            "clause_decisions": {
                "clause-invalid": {
                    "allowed": False,
                    "reason": "missing_receipt",
                }
            },
        },
    )
    gateway = SimpleNamespace(clauses={"clause-invalid": clause})
    logger.record_visibility(pack, gateway, source="test-counterfactual")
    node = SimpleNamespace(
        id="node-counterfactual",
        stage="draft",
        draft_role="memory_transfer",
        plan="Authority-enforced plan.",
        code="actual = True\n",
        prompt_input="frozen actual prompt",
        receipt_refs=[],
    )
    logger.bind_thread_to_node(node)

    def fake_generate(_agent, prompt, **_kwargs):
        assert "Use the suppressed raw method." in prompt
        return "Counterfactual plan.", "counterfactual = True\n"

    monkeypatch.setattr("agents.coder.plan_and_code_query", fake_generate)
    logger.prepare_counterfactuals(node)
    logger.finalize_node(node)
    row = _rows(tmp_path / "prospective_decision_ledger.jsonl")[0]
    assert row["counterfactual_status"] == "complete"
    assert row["counterfactual_pair_id"].startswith("prospective-pair::")
    assert row["counterfactual_influence_confirmed"] is True
    assert row["counterfactual_receipt_refs"] == [
        "host-counterfactual-receipt-1"
    ]
    assert "host-counterfactual-receipt-1" in row["runtime_receipt_refs"]
    assert row["counterfactual_code_hash"] != row["actual_code_hash"]


def test_counterfactual_retries_one_empty_generation(tmp_path: Path, monkeypatch):
    agent = _agent(tmp_path)
    agent.cfg.prospective_audit.counterfactual_generation_attempts = 2
    logger = ProspectiveAuditLogger(agent)
    protocol_ref = "protocol@1#" + "a" * 64
    pack = SimpleNamespace(
        request_id="visibility-request-counterfactual-retry",
        visibility_trace={
            "request": {
                "operation": "generate_candidate",
                "generation_stage": "draft",
            },
            "raw_shadow_authority_decisions": [{
                "clause_id": "clause-invalid",
                "claim_id": "claim-invalid",
                "claim_type": "method_hypothesis",
                "outcome": "deny",
                "protocol_ref": protocol_ref,
            }],
            "suppressed_clause_refs": ["clause-invalid"],
            "effective_visible_clause_ids": [],
            "clause_decisions": {
                "clause-invalid": {"allowed": False, "reason": "missing_receipt"}
            },
        },
    )
    logger.record_visibility(
        pack,
        SimpleNamespace(
            clauses={"clause-invalid": SimpleNamespace(text="suppressed method")}
        ),
        source="test-counterfactual-retry",
    )
    node = SimpleNamespace(
        id="node-counterfactual-retry",
        stage="draft",
        draft_role="memory_transfer",
        plan="actual plan",
        code="actual = True\n",
        prompt_input="actual prompt",
        receipt_refs=[],
    )
    logger.bind_thread_to_node(node)
    calls = []

    def flaky_generate(_agent, _prompt, **kwargs):
        calls.append(kwargs["request_timeout"])
        if len(calls) == 1:
            return "", ""
        return "retry plan", "retry_code = True\n"

    monkeypatch.setattr("agents.coder.plan_and_code_query", flaky_generate)
    logger.prepare_counterfactuals(node)
    logger.finalize_node(node)
    row = _rows(tmp_path / "prospective_decision_ledger.jsonl")[0]
    assert len(calls) == 2
    assert calls == [150.0, 150.0]
    assert row["counterfactual_status"] == "complete"


def test_code_only_counterfactual_is_complete_not_pending(tmp_path: Path, monkeypatch):
    agent = _agent(tmp_path)
    logger = ProspectiveAuditLogger(agent)
    protocol_ref = "protocol@1#" + "a" * 64
    pack = SimpleNamespace(
        request_id="visibility-request-code-only-counterfactual",
        visibility_trace={
            "request": {
                "operation": "generate_candidate",
                "generation_stage": "draft",
            },
            "raw_shadow_authority_decisions": [{
                "clause_id": "clause-invalid",
                "claim_id": "claim-invalid",
                "claim_type": "method_hypothesis",
                "outcome": "deny",
                "protocol_ref": protocol_ref,
            }],
            "suppressed_clause_refs": ["clause-invalid"],
            "effective_visible_clause_ids": [],
            "clause_decisions": {
                "clause-invalid": {"allowed": False, "reason": "missing_receipt"}
            },
        },
    )
    logger.record_visibility(
        pack,
        SimpleNamespace(
            clauses={"clause-invalid": SimpleNamespace(text="suppressed method")}
        ),
        source="test-code-only-counterfactual",
    )
    node = SimpleNamespace(
        id="node-code-only-counterfactual",
        stage="draft",
        draft_role="memory_transfer",
        plan="actual plan",
        code="actual = True\n",
        prompt_input="actual prompt",
        receipt_refs=[],
    )
    logger.bind_thread_to_node(node)
    monkeypatch.setattr(
        "agents.coder.plan_and_code_query",
        lambda *_args, **_kwargs: ("", "```python\ncounterfactual = True\n```"),
    )

    logger.prepare_counterfactuals(node)
    logger.finalize_node(node)

    row = _rows(tmp_path / "prospective_decision_ledger.jsonl")[0]
    assert row["counterfactual_status"] == "complete"
    assert row["counterfactual_action_hash"]
    assert row["counterfactual_code_hash"]


def test_search_replace_counterfactual_is_reconstructed_from_parent(
    tmp_path: Path, monkeypatch
):
    agent = _agent(tmp_path)
    agent.cfg.prospective_audit.allow_pending_counterfactual = False
    logger = ProspectiveAuditLogger(agent)
    protocol_ref = "protocol@1#" + "a" * 64
    pack = SimpleNamespace(
        request_id="visibility-request-diff-counterfactual",
        visibility_trace={
            "request": {
                "operation": "debug_hypothesis",
                "generation_stage": "debug",
            },
            "raw_shadow_authority_decisions": [{
                "clause_id": "clause-invalid",
                "claim_id": "claim-invalid",
                "claim_type": "method_hypothesis",
                "outcome": "deny",
                "protocol_ref": protocol_ref,
            }],
            "suppressed_clause_refs": ["clause-invalid"],
            "effective_visible_clause_ids": [],
            "clause_decisions": {
                "clause-invalid": {"allowed": False, "reason": "missing_receipt"}
            },
        },
    )
    logger.record_visibility(
        pack,
        SimpleNamespace(
            clauses={"clause-invalid": SimpleNamespace(text="suppressed method")}
        ),
        source="test-diff-counterfactual",
    )
    parent_code = "value = 1\nprint(value)\n"
    parent = SimpleNamespace(code=parent_code)
    node = SimpleNamespace(
        id="node-diff-counterfactual",
        stage="debug",
        draft_role="memory_transfer",
        plan="actual repair",
        code="value = 3\nprint(value)\n",
        prompt_input="diff-oriented actual prompt",
        receipt_refs=[],
        parent=parent,
    )
    logger.bind_thread_to_node(node)
    raw_diff = (
        "The suppressed memory suggests changing the value.\n"
        "<<<<<<< SEARCH\n"
        "value = 1\n"
        "=======\n"
        "value = 2\n"
        ">>>>>>> REPLACE\n"
    )
    monkeypatch.setattr(
        "agents.coder.plan_and_code_query",
        lambda *_args, **_kwargs: ("", raw_diff),
    )

    logger.prepare_counterfactuals(node)
    logger.finalize_node(node)

    row = _rows(tmp_path / "prospective_decision_ledger.jsonl")[0]
    expected_code = "value = 2\nprint(value)\n"
    assert row["counterfactual_status"] == "complete"
    assert row["counterfactual_generation_format"] == "search_replace_diff"
    assert row["counterfactual_diff_patch_count"] == 1
    assert row["counterfactual_base_code_hash"] == hashlib.sha256(
        parent_code.encode("utf-8")
    ).hexdigest()
    assert row["counterfactual_raw_completion_hash"] == hashlib.sha256(
        raw_diff.encode("utf-8")
    ).hexdigest()
    assert row["counterfactual_code_hash"] == hashlib.sha256(
        expected_code.encode("utf-8")
    ).hexdigest()
