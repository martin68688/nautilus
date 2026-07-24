from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from authority.adapters.mlevolve import runtime
from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from authority.authority_engine import AuthorityEngine
from authority.evidence_graph import EvidenceGraph
from authority.models import (
    AuthorityRequest,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    Operation,
    ProtocolSpec,
    TaskContext,
)
from authority.protocol_registry import ProtocolRegistry


def _registry():
    registry = ProtocolRegistry()
    spec = registry.register(
        ProtocolSpec(
            protocol_id="test",
            version="1",
            task_profile={},
            data_split_policy={},
            preprocessing_policy={},
            evaluator_spec={},
            metric_spec={},
            selection_policy={},
            seed_policy={},
            holdout_policy={},
            promotion_policy={},
            compatibility_rules={},
        )
    )
    return registry, spec.ref()


def _request(protocol_ref, operation: Operation) -> AuthorityRequest:
    return AuthorityRequest(
        artifact_id="artifact",
        claim_id="claim",
        operation=operation,
        decision_stage=DecisionStage.BRANCH_SELECTION,
        active_protocol=protocol_ref,
        task_context=TaskContext("task"),
        requesting_component="test",
    )


def test_engine_internal_exception_denies_high_risk_operation(monkeypatch) -> None:
    registry, protocol_ref = _registry()
    graph = EvidenceGraph()
    graph.add_claim(
        Claim(
            claim_id="claim",
            claim_type=ClaimType.SCORE,
            subject_artifact_id="artifact",
            task_scope={"task_id": "task"},
            method_fingerprint="method",
            protocol_ref=protocol_ref,
            statement="score",
        )
    )
    engine = AuthorityEngine(registry, graph=graph)

    def broken(*_args, **_kwargs):
        raise RuntimeError("simulated compiler failure")

    monkeypatch.setattr(engine.compiler, "compile", broken)
    decision = engine.authorize(_request(protocol_ref, Operation.RANK))
    assert decision.outcome == DecisionOutcome.DENY
    assert decision.permitted_scope is None
    assert decision.missing_obligations == ["authority_internal_error:RuntimeError"]


def test_low_risk_internal_exception_returns_warning_navigation_only(monkeypatch) -> None:
    registry, protocol_ref = _registry()
    graph = EvidenceGraph()
    graph.add_claim(
        Claim(
            claim_id="claim",
            claim_type=ClaimType.AUDIT_FINDING,
            subject_artifact_id="artifact",
            task_scope={"task_id": "task"},
            method_fingerprint="method",
            protocol_ref=protocol_ref,
            statement="audit",
        )
    )
    engine = AuthorityEngine(registry, graph=graph)
    monkeypatch.setattr(
        engine.compiler,
        "compile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("host-visible")),
    )
    decision = engine.authorize(_request(protocol_ref, Operation.INSPECT))
    assert decision.outcome == DecisionOutcome.ALLOW_WITH_WARNING
    assert decision.allowed is True
    assert decision.permitted_scope is None
    assert decision.missing_obligations == ["authority_internal_error:RuntimeError"]
    assert "navigation only" in str(decision.required_action)
    assert "abstain" in str(decision.required_action)


def _agent(tmp_path: Path):
    cfg = SimpleNamespace(
        exp_id="task",
        exp_name="run",
        eval="metric",
        log_dir=tmp_path,
        fixed_holdout=SimpleNamespace(enabled=False),
        evaluation_authority=SimpleNamespace(
            mode="enforce",
            protocol_registry=str(
                Path(__file__).resolve().parents[2] / "mlevolve" / "config" / "protocols"
            ),
            active_protocol_id="mlevolve-default",
            active_protocol_version="1",
            policy_version="authority_v1",
            fail_closed_high_risk=False,
            allow_invalid_debug=True,
            emit_snapshot=False,
        ),
    )
    return SimpleNamespace(cfg=cfg)


def test_adapter_pre_engine_exception_cannot_use_legacy_allow(monkeypatch, tmp_path) -> None:
    adapter = MLEvolveAuthorityAdapter(_agent(tmp_path))
    node = SimpleNamespace(
        id="node",
        stage="improve",
        code="print('ok')",
        claim_refs=[],
        receipt_refs=[],
        authority_decision_refs=[],
        derived_from_refs=[],
        protocol_ref="",
        method_fingerprint="",
    )

    def broken(*_args, **_kwargs):
        raise RuntimeError("simulated claim adapter failure")

    # WP2 decomposes a node into operation-scoped claims before choosing the
    # claim to authorize.  Inject the failure at that current pre-engine
    # boundary so this regression continues to prove fail-closed behavior.
    monkeypatch.setattr(runtime, "claims_for_node", broken)
    decision = adapter.authorize_node(
        node,
        Operation.RANK,
        DecisionStage.BRANCH_SELECTION,
        "test",
    )
    assert decision.outcome == DecisionOutcome.DENY
    assert adapter.permits(decision, legacy_allowed=True) is False
    assert decision.decision_id in node.authority_decision_refs
