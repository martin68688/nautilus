from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import threading

from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from authority.adapters.mlevolve.protocol_adapter import build_registry
from authority.models import DecisionOutcome, DecisionStage, Operation


class Metric:
    def __init__(self, value):
        self.value = value
        self.maximize = True


def fake_agent(tmp_path: Path, mode="enforce"):
    cfg = SimpleNamespace(
        exp_id="task",
        exp_name="run",
        eval="metric",
        log_dir=tmp_path,
        fixed_holdout=SimpleNamespace(enabled=False),
        evaluation_authority=SimpleNamespace(
            mode=mode,
            protocol_registry=str(Path(__file__).resolve().parents[2] / "mlevolve" / "config" / "protocols"),
            active_protocol_id="mlevolve-default",
            active_protocol_version="1",
            policy_version="authority_v1",
            fail_closed_high_risk=True,
            allow_invalid_debug=True,
            emit_snapshot=True,
        ),
    )
    return SimpleNamespace(cfg=cfg)


def node(node_id="n1", clean=True):
    code = "print('ok')"
    audit = {
        "schema": "mlevolve_leakage_audit_v2",
        "detector_status": "complete",
        "status": "clean" if clean else "blocked",
        "metric_disposition": "accept" if clean else "reject",
        "paper_grade_eligible": clean,
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "issues": [] if clean else [{"issue_code": "TEST_LABEL_ACCESS"}],
    }
    return SimpleNamespace(
        id=node_id,
        code=code,
        metric=Metric(0.8),
        exec_time=1.0,
        leakage_audit=audit,
        derived_from_refs=[],
        claim_refs=[],
        receipt_refs=[],
        authority_decision_refs=[],
        protocol_ref="",
        method_fingerprint="",
        is_buggy=False,
        is_valid=True,
        branch_id=1,
        draft_role="general_draft",
        selected_strategy={},
        strategy_alignment={},
    )


def test_enforce_allows_clean_and_denies_contaminated_node(tmp_path):
    agent = fake_agent(tmp_path)
    agent.evaluation_authority = MLEvolveAuthorityAdapter(agent)
    clean = node("clean", True)
    dirty = node("dirty", False)
    assert agent.evaluation_authority.gate_node(
        clean, Operation.RANK, DecisionStage.BRANCH_SELECTION, "test", legacy_allowed=True
    )
    assert not agent.evaluation_authority.gate_node(
        dirty, Operation.RANK, DecisionStage.BRANCH_SELECTION, "test", legacy_allowed=True
    )
    assert (tmp_path / "authority_events.jsonl").exists()
    assert (tmp_path / "authority_snapshot.json").exists()
    assert (tmp_path / "evidence_graph.json").exists()


def test_shadow_records_same_decision_but_preserves_legacy_behavior(tmp_path):
    agent = fake_agent(tmp_path, mode="shadow")
    agent.evaluation_authority = MLEvolveAuthorityAdapter(agent)
    dirty = node("dirty", False)
    decision = agent.evaluation_authority.authorize_node(
        dirty, Operation.RANK, DecisionStage.BRANCH_SELECTION, "test"
    )
    assert decision.outcome == DecisionOutcome.DENY
    assert agent.evaluation_authority.permits(decision, legacy_allowed=True)
    assert not agent.evaluation_authority.permits(decision, legacy_allowed=False)


def test_global_top_k_authorizes_before_sorting(tmp_path):
    from engine.node_selection import get_top_k_nodes_global

    agent = fake_agent(tmp_path)
    agent.acfg = SimpleNamespace(check_data_leakage=True)
    agent.metric_maximize = True
    legal = node("legal", True)
    legal.metric = Metric(0.5)
    contaminated = node("contaminated", False)
    contaminated.metric = Metric(0.99)
    agent.branch_all_nodes = {1: [contaminated, legal]}
    agent.evaluation_authority = MLEvolveAuthorityAdapter(agent)

    ranked = get_top_k_nodes_global(agent, k=2, max_from_same_branch=2)

    assert [item["node"].id for item in ranked] == ["legal"]
    assert ranked[0]["rank"] == 1


def test_unknown_protocol_fails_closed(tmp_path):
    agent = fake_agent(tmp_path)
    agent.cfg.evaluation_authority.active_protocol_version = "typo"
    try:
        build_registry(agent.cfg)
    except KeyError as exc:
        assert "Unknown protocol" in str(exc)
    else:
        raise AssertionError("unknown protocol was synthesized at runtime")


def test_clean_receipts_have_typed_collectors_and_payloads(tmp_path):
    agent = fake_agent(tmp_path)
    agent.evaluation_authority = MLEvolveAuthorityAdapter(agent)
    candidate = node("typed", True)
    agent.evaluation_authority.authorize_node(
        candidate, Operation.RANK, DecisionStage.BRANCH_SELECTION, "test"
    )
    receipts = list(agent.evaluation_authority.engine.graph.receipts.values())
    split = next(item for item in receipts if item.receipt_type.value == "split_lineage")
    fit = next(item for item in receipts if item.receipt_type.value == "fit_scope")
    assert split.collector_id != fit.collector_id
    assert split.payload["partition_lineage_verified"] is True
    assert fit.payload["fit_scope_verified"] is True


def test_rank_gate_internal_error_fails_closed_in_enforce_mode(monkeypatch):
    from agents.leakage_audit import rank_eligible
    from authority.adapters.mlevolve import ranking_gate

    agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=False),
        evaluation_authority=SimpleNamespace(mode="enforce"),
    )
    candidate = node("broken-gate", True)

    def broken(*_args, **_kwargs):
        raise AttributeError("simulated authority regression")

    monkeypatch.setattr(ranking_gate, "authorize_ranking", broken)
    assert rank_eligible(agent, candidate) is False


def test_filtered_journal_excludes_runtime_authority_agent_with_thread_locks():
    from engine.search_node import Journal, SearchNode, filter_on_path

    root = SearchNode(code="", stage="root")
    journal = Journal([root])
    journal.authority_agent = SimpleNamespace(lock=threading.Lock())
    journal.authority_enforced = True

    filtered = filter_on_path(journal, [root.id])
    persisted_copy = copy.deepcopy(journal)

    assert [item.id for item in filtered.nodes] == [root.id]
    assert not hasattr(filtered, "authority_agent")
    assert not hasattr(filtered, "authority_enforced")
    assert [item.id for item in persisted_copy.nodes] == [root.id]
    assert not hasattr(persisted_copy, "authority_agent")
