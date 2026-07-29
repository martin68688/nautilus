from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from agents.memory.run_forest_replay import full_runtime_migration_report
from agents import result_parse_agent
from agents.triggers import (
    refresh_replay_lineage_after_instrumentation,
    refresh_replay_lineage_after_revision,
    register_node,
)
from authority.protocol_execution_contract import (
    compile_protocol_execution_contract,
    write_contract_artifacts,
)
from authority.protocol_registry import ProtocolRegistry
from engine.search_node import SearchNode
from protocol_runtime.collector import HostCollectorIdentity


def _contract(tmp_path: Path) -> Path:
    identity = HostCollectorIdentity.generate()
    registry = ProtocolRegistry("mlevolve/config/protocols")
    contract = compile_protocol_execution_contract(
        registry.resolve("deterministic-random-regression@1"),
        task_id="denoising-dirty-documents",
        task_family="image",
        train_view_ref="view://denoising/train",
        validation_view_ref="view://denoising/internal-validation",
        terminal_view_ref="evaluator-only://denoising/terminal",
        execution_budget={
            "max_epochs": 0,
            "max_folds": 1,
            "max_models": 4,
            "timeout_seconds": 180,
        },
        collector_spec=identity.collector_spec(),
    )
    path, _ = write_contract_artifacts(contract, tmp_path / "contract")
    return path


def test_legacy_replay_without_host_main_requires_derived_candidate(tmp_path: Path):
    contract_path = _contract(tmp_path)
    agent = SimpleNamespace(
        acfg=SimpleNamespace(
            protocol_preflight=SimpleNamespace(
                enabled=True,
                contract_path=str(contract_path),
            )
        )
    )
    source = "print('historical program without Host lifecycle')\n"
    report = full_runtime_migration_report(agent, source)

    assert report["status"] == "required"
    assert report["source_execution_allowed"] is False
    assert report["derived_candidate_required"] is True
    assert {
        "current_session",
        "split_lineage",
        "fit_scope",
        "prediction_scope",
        "evaluator",
        "selection_freeze",
        "main_guard",
    } <= set(report["missing_full_runtime_coverage"])


def test_migration_seed_never_executes_and_child_has_explicit_derivation(tmp_path: Path):
    source = "model = object()\nprint('legacy execution')\n"
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    parent = SearchNode(
        code=source,
        plan="historical method seed",
        stage="draft",
        branch_id=1,
        draft_role="memory_reproduction",
        replay_source={
            "graph_node_id": "run::old::node::best",
            "code_sha256": source_hash,
            "requires_full_runtime_migration": True,
            "execution_seed_only": True,
            "full_runtime_migration": {
                "missing_full_runtime_coverage": ["fit_scope", "selection_freeze"]
            },
        },
        replay_status="blocked_legacy_full_runtime_seed",
    )
    audit_agent = SimpleNamespace(
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        acfg=SimpleNamespace(check_data_leakage=False),
        global_memory=None,
        external_skill_memory=None,
    )

    assert result_parse_agent.run_pre_execution_leakage_audit(audit_agent, parent)
    assert parent.leakage_audit["pre_execution_gate_reason"] == (
        "legacy_full_runtime_migration_seed"
    )
    assert parent.leakage_audit["rank_eligible"] is False
    assert parent.is_buggy is True
    assert parent.metric.is_worst

    derived_code = source + "\ndef main():\n    pass\n"
    child = SearchNode(
        code=derived_code,
        plan="Host full-runtime migration",
        stage="debug",
        parent=parent,
    )
    branch_agent = SimpleNamespace(
        _serialize_prompt=str,
        next_branch_id=2,
        branch_all_nodes={1: [parent]},
        branch_successful_nodes={1: []},
    )
    register_node(branch_agent, child, "migration", parent_node=parent)

    assert child.replay_source["execution_seed_only"] is False
    assert child.replay_source["migration_parent_node_id"] == parent.id
    assert child.replay_status == "derived_full_runtime_candidate"
    assert child.derived_from_refs == [
        "replay:run::old::node::best:method_hypothesis"
    ]


def test_review_and_host_instrumentation_extend_derived_hash_chain():
    source = "print('source')\n"
    generated = "def main():\n    print('generated')\n"
    reviewed = generated.replace("generated", "reviewed")
    instrumented = "def candidate(session):\n    pass\n\n" + reviewed
    node = SearchNode(
        code=generated,
        plan="derived candidate",
        stage="draft",
        replay_source={
            "graph_node_id": "run::old::node::best",
            "code_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "current_code_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        },
        replay_status="derived_full_runtime_candidate",
    )

    node.code = reviewed
    refresh_replay_lineage_after_revision(
        node,
        original_code=generated,
        revision_kind="code_review_descendant",
    )
    assert node.replay_source["current_code_sha256"] == hashlib.sha256(
        reviewed.encode()
    ).hexdigest()

    node.code = instrumented
    refresh_replay_lineage_after_instrumentation(
        node,
        original_code=reviewed,
        instrumentation_receipt={"receipt_hash": "a" * 64},
    )
    assert node.replay_source["current_code_sha256"] == hashlib.sha256(
        instrumented.encode()
    ).hexdigest()
    assert node.replay_source["lineage_kind"] == "host_instrumented_descendant"
    assert node.derived_from_refs == [
        "replay:run::old::node::best:method_hypothesis"
    ]
