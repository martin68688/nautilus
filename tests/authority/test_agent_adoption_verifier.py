from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from authority.actuation import ActuationLevel, ExperienceContract, Predicate
from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from agents.adoption_verifier_agent import AdoptionVerifierAgent
from engine.executor import Interpreter
from config import _validate_adoption_verifier_config
from authority.adoption_verification import (
    build_final_verdict,
    build_verification_plan,
    verify_plan,
)
from protocol_runtime.adoption_trace import (
    bootstrap_for_prefix,
    seal_runtime_trace,
    verify_sealed_trace,
)
from protocol_runtime.collector import HostCollectorIdentity
from tests.authority.test_mlevolve_adapter import fake_agent, node


DIGEST = "a" * 64


def _contract(contract_id: str = "experience_contract::one") -> dict:
    return {
        "contract_id": contract_id,
        "contract_hash": DIGEST,
        "clause_id": "clause-one",
        "sop_id": "sop-one",
        "preconditions": [],
        "must_preserve": [
            {"name": "task_id", "expected": "task-a"},
        ],
        "must_change": [
            {"name": "clause_applied::clause-one", "expected": True},
        ],
        "must_not_use": [],
        "expected_runtime_observations": [
            {"name": "target_path_executed", "expected": True},
        ],
    }


def _plan(source: str) -> dict:
    return build_verification_plan(
        artifact_id="node-a",
        source=source,
        contracts=[_contract()],
        verifier_model="verifier-test",
        response={
            "contract_results": [
                {
                    "contract_id": "experience_contract::one",
                    "disposition": "implemented",
                    "reasoning": "The memory-specific assignment is executable.",
                    "code_evidence": [
                        {
                            "start_line": 2,
                            "end_line": 2,
                            "description": "memory-specific implementation",
                        }
                    ],
                    "runtime_probes": [
                        {
                            "probe_id": "probe-memory-path",
                            "start_line": 2,
                            "end_line": 2,
                            "description": "observe implementation line",
                        }
                    ],
                    "static_observations": [
                        {
                            "name": "clause_applied::clause-one",
                            "value": True,
                            "reason": "bound source line",
                        }
                    ],
                    "runtime_observations": [
                        {
                            "name": "target_path_executed",
                            "value": True,
                            "reason": "bound runtime probe",
                        }
                    ],
                }
            ]
        },
    )


def test_plan_is_bound_to_exact_code_and_contract() -> None:
    source = "value = 1\nresult = value + 1\nprint(result)\n"
    plan = _plan(source)

    verify_plan(plan, artifact_id="node-a", source=source)
    row = plan["contract_results"][0]
    assert row["disposition"] == "implemented"
    assert row["runtime_probes"][0]["probe_id"] == "probe-memory-path"
    assert row["static_observations"] == [
        {
            "name": "clause_applied::clause-one",
            "value": True,
            "reason": "bound source line",
        }
    ]


def test_missing_positive_evidence_fails_closed_to_uncertain() -> None:
    plan = build_verification_plan(
        artifact_id="node-a",
        source="print('ok')\n",
        contracts=[_contract()],
        verifier_model="verifier-test",
        response={
            "contract_results": [
                {
                    "contract_id": "experience_contract::one",
                    "disposition": "implemented",
                    "reasoning": "unsupported positive",
                    "code_evidence": [],
                    "runtime_probes": [],
                    "static_observations": [],
                    "runtime_observations": [],
                }
            ]
        },
    )

    assert plan["contract_results"][0]["disposition"] == "uncertain"


def test_generic_line_probe_is_host_sealed_and_supports_final_verdict(
    tmp_path: Path,
) -> None:
    source = "value = 1\nresult = value + 1\nprint(result)\n"
    plan = _plan(source)
    raw_trace = tmp_path / "raw-trace.json"
    nonce = "n" * 64
    prefix = ""
    bootstrap = bootstrap_for_prefix(
        plan,
        output_path=raw_trace,
        nonce=nonce,
        prefix=prefix,
    )
    script = tmp_path / "candidate.py"
    script.write_text(prefix + bootstrap + source, encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(raw_trace.read_text())["probe_hits"]["probe-memory-path"]

    identity = HostCollectorIdentity.generate()
    trace = seal_runtime_trace(
        raw_path=raw_trace,
        plan=plan,
        nonce=nonce,
        exit_status=completed.returncode,
        identity=identity,
    )
    verify_sealed_trace(trace, identity)
    assert trace["probe_results"] == [
        {
            "probe_id": "probe-memory-path",
            "kind": "line_range_executed",
            "executed": True,
            "executed_lines": [2],
            "hit_count": 1,
        }
    ]

    verdict = build_final_verdict(
        artifact_id="node-a",
        plan=plan,
        trace=trace,
        verifier_model="verifier-test",
        response={
            "contract_results": [
                {
                    "contract_id": "experience_contract::one",
                    "verdict": "adopted",
                    "reasoning": "The memory-specific source path executed.",
                    "supporting_probe_ids": ["probe-memory-path"],
                }
            ]
        },
    )
    assert verdict["contract_results"][0]["verdict"] == "adopted"
    assert verdict["contract_results"][0]["runtime_evidence_valid"] is True


def test_agent_cannot_claim_adoption_with_an_unexecuted_probe() -> None:
    source = "value = 1\nresult = value + 1\nprint(result)\n"
    plan = _plan(source)
    trace = {
        "artifact_id": "node-a",
        "code_sha256": plan["code_sha256"],
        "plan_hash": plan["plan_hash"],
        "trace_hash": "b" * 64,
        "probe_results": [
            {
                "probe_id": "probe-memory-path",
                "executed": False,
                "executed_lines": [],
                "hit_count": 0,
            }
        ],
    }
    verdict = build_final_verdict(
        artifact_id="node-a",
        plan=plan,
        trace=trace,
        verifier_model="verifier-test",
        response={
            "contract_results": [
                {
                    "contract_id": "experience_contract::one",
                    "verdict": "adopted",
                    "reasoning": "unsupported",
                    "supporting_probe_ids": ["probe-memory-path"],
                }
            ]
        },
    )

    assert verdict["contract_results"][0]["verdict"] == "uncertain"
    assert verdict["contract_results"][0]["runtime_evidence_valid"] is False


def test_agent_enforce_replaces_legacy_strategy_signature_gate(tmp_path: Path) -> None:
    agent = fake_agent(tmp_path, mode="enforce")
    agent.cfg.adoption_verifier = SimpleNamespace(
        enabled=True,
        mode="enforce",
        require_signed_trace=False,
    )
    adapter = MLEvolveAuthorityAdapter(agent)
    agent.evaluation_authority = adapter
    candidate = node("agent-adoption", clean=True)
    candidate.selected_strategy = {"sop_id": "unrelated-legacy-sop"}
    candidate.strategy_alignment = {"status": "mismatch", "rank_eligible": False}
    candidate.experience_actuation_observations = {}
    candidate.experience_contract_refs = []
    candidate.actuation_report_refs = []
    candidate.adoption_verifier_mode = "enforce"

    contract = ExperienceContract(
        must_preserve=[
            Predicate("active_protocol_ref", adapter.active_protocol.key()),
            Predicate("task_id", adapter.task_id),
        ],
        must_change=[Predicate("clause_applied::generic-clause", True)],
        must_not_use=[
            Predicate("forbidden_dependency_count", 0),
            Predicate("holdout_used_for_selection", False),
        ],
        expected_runtime_observations=[Predicate("target_path_executed", True)],
        clause_id="generic-clause",
        sop_id="generic-sop",
        active_protocol_ref=adapter.active_protocol.key(),
        target_task_id=adapter.task_id,
        publication_class="certified",
    ).finalize()
    adapter.actuation_tracker.record_exposure(
        artifact_id=candidate.id,
        contracts=[contract],
        request_id="agent-verifier-test",
    )
    plan = build_verification_plan(
        artifact_id=candidate.id,
        source=candidate.code,
        contracts=[contract],
        verifier_model="verifier-test",
        response={
            "contract_results": [
                {
                    "contract_id": contract.contract_id,
                    "disposition": "implemented",
                    "reasoning": "The specific implementation is present.",
                    "code_evidence": [
                        {"start_line": 1, "end_line": 1, "description": "path"}
                    ],
                    "runtime_probes": [
                        {
                            "probe_id": "generic-probe",
                            "start_line": 1,
                            "end_line": 1,
                            "description": "path executes",
                        }
                    ],
                    "static_observations": [],
                    "runtime_observations": [],
                }
            ]
        },
    )
    trace = {
        "artifact_id": candidate.id,
        "code_sha256": plan["code_sha256"],
        "plan_hash": plan["plan_hash"],
        "trace_hash": "c" * 64,
        "probe_results": [
            {
                "probe_id": "generic-probe",
                "executed": True,
                "executed_lines": [1],
                "hit_count": 1,
            }
        ],
    }
    verdict = build_final_verdict(
        artifact_id=candidate.id,
        plan=plan,
        trace=trace,
        verifier_model="verifier-test",
        response={
            "contract_results": [
                {
                    "contract_id": contract.contract_id,
                    "verdict": "adopted",
                    "reasoning": "The planned memory path executed.",
                    "supporting_probe_ids": ["generic-probe"],
                }
            ]
        },
    )
    candidate.adoption_verification_plan = plan
    candidate.adoption_runtime_trace = trace
    candidate.adoption_verifier_verdict = verdict
    agent.adoption_verifier = SimpleNamespace(
        enabled=True,
        mode="enforce",
        finalize=lambda _node: verdict,
    )

    reports = adapter.finalize_production_actuation(candidate)

    assert reports[0]["highest_level"] == int(ActuationLevel.RUNTIME_CONFORMANT)
    assert reports[0]["promotion_eligible"] is True


def test_interpreter_collects_generic_adoption_trace_without_task_adapter(
    tmp_path: Path,
) -> None:
    source = "value = 1\nresult = value + 1\nprint(result)\n"
    plan = build_verification_plan(
        artifact_id="executor-node",
        source=source,
        contracts=[_contract()],
        verifier_model="verifier-test",
        response={
            "contract_results": [
                {
                    "contract_id": "experience_contract::one",
                    "disposition": "implemented",
                    "reasoning": "implementation",
                    "code_evidence": [
                        {"start_line": 2, "end_line": 2, "description": "path"}
                    ],
                    "runtime_probes": [
                        {
                            "probe_id": "probe-memory-path",
                            "start_line": 2,
                            "end_line": 2,
                            "description": "path",
                        }
                    ],
                    "static_observations": [],
                    "runtime_observations": [],
                }
            ]
        },
    )
    cfg = SimpleNamespace(
        start_cpu_id=0,
        cpu_number=1,
        evaluation_authority=SimpleNamespace(
            mode="shadow",
            protocol_runtime_mode="host_sdk_shadow",
            runtime_protocol_observer_enabled=False,
        ),
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
            protocol_preflight=SimpleNamespace(enabled=False),
            candidate_execution_contract=SimpleNamespace(enabled=False),
        ),
    )
    interpreter = Interpreter(
        tmp_path,
        timeout=30,
        max_parallel_run=1,
        cfg=cfg,
    )

    result = interpreter.run(
        source,
        "executor-node",
        adoption_verification_plan=plan,
    )

    assert result.exc_type is None
    assert result.adoption_trace["schema"] == "agent_adoption_runtime_trace_v1"
    assert result.adoption_trace["probe_results"][0]["executed"] is True
    assert result.adoption_trace["probe_results"][0]["executed_lines"] == [2]


def test_verifier_agent_owns_static_and_final_semantic_decisions(
    tmp_path: Path,
) -> None:
    source = "value = 1\nresult = value + 1\nprint(result)\n"
    contract = _contract()
    responses = [
        {
            "contract_results": [
                {
                    "contract_id": contract["contract_id"],
                    "disposition": "implemented",
                    "reasoning": "specific implementation",
                    "code_evidence": [
                        {"start_line": 2, "end_line": 2, "description": "path"}
                    ],
                    "runtime_probes": [
                        {
                            "probe_id": "agent-probe",
                            "start_line": 2,
                            "end_line": 2,
                            "description": "path",
                        }
                    ],
                    "static_observations": [],
                    "runtime_observations": [],
                }
            ]
        },
        {
            "contract_results": [
                {
                    "contract_id": contract["contract_id"],
                    "verdict": "adopted",
                    "reasoning": "probe executed",
                    "supporting_probe_ids": ["agent-probe"],
                }
            ]
        },
    ]
    calls = []

    def fake_query(**kwargs):
        calls.append(kwargs["func_spec"].name)
        return responses.pop(0)

    cfg = SimpleNamespace(
        log_dir=tmp_path,
        adoption_verifier=SimpleNamespace(
            enabled=True,
            mode="enforce",
            model="verifier-model",
            temperature=0.0,
            max_tokens=2048,
            max_contracts_per_call=8,
            max_code_chars=10000,
        ),
    )
    agent = SimpleNamespace(
        cfg=cfg,
        acfg=SimpleNamespace(feedback=SimpleNamespace(model="fallback-model")),
        task_desc="generic task",
        evaluation_authority=SimpleNamespace(ledger=None),
    )
    candidate = SimpleNamespace(
        id="agent-node",
        code=source,
        adoption_verification_plan={},
        adoption_runtime_trace={},
        adoption_verifier_verdict={},
        adoption_verifier_mode="off",
    )
    verifier = AdoptionVerifierAgent(agent, query_fn=fake_query)

    plan = verifier.prepare(candidate, [contract])
    candidate.adoption_runtime_trace = {
        "artifact_id": candidate.id,
        "code_sha256": plan["code_sha256"],
        "plan_hash": plan["plan_hash"],
        "trace_hash": "d" * 64,
        "probe_results": [
            {
                "probe_id": "agent-probe",
                "executed": True,
                "executed_lines": [2],
                "hit_count": 1,
            }
        ],
    }
    verdict = verifier.finalize(candidate)

    assert calls == [
        "submit_memory_adoption_verification_plan",
        "submit_memory_adoption_verdict",
    ]
    assert candidate.adoption_verifier_mode == "enforce"
    assert verdict["contract_results"][0]["verdict"] == "adopted"
    assert (tmp_path / "adoption_verifier" / "agent-node.plan.json").is_file()
    assert (tmp_path / "adoption_verifier" / "agent-node.verdict.json").is_file()


def test_enforce_config_rejects_legacy_hardcoded_runtime() -> None:
    cfg = SimpleNamespace(
        adoption_verifier=SimpleNamespace(
            enabled=True,
            mode="enforce",
            require_signed_trace=False,
        ),
        evaluation_authority=SimpleNamespace(
            mode="enforce",
            protocol_runtime_mode="legacy_ast",
        ),
        agent=SimpleNamespace(protocol_preflight=SimpleNamespace(enabled=False)),
    )

    try:
        _validate_adoption_verifier_config(cfg)
    except ValueError as error:
        assert "requires host_sdk_shadow or host_sdk_enforce" in str(error)
    else:  # pragma: no cover - explicit failure message is clearer than pytest magic here
        raise AssertionError("legacy hardcoded runtime unexpectedly accepted")


def test_agent_search_imports_replay_anchor_used_by_verifier_guard() -> None:
    """Catch conflict resolutions that retain the call but drop its import."""

    import ast

    path = (
        Path(__file__).resolve().parents[2]
        / "mlevolve"
        / "engine"
        / "agent_search.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "agents.memory.run_forest_replay"
        for alias in node.names
    }
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }

    assert "is_historical_replay_anchor" in called
    assert "is_historical_replay_anchor" in imported


def test_replay_anchor_guard_is_importable_and_classifies_exact_anchor() -> None:
    """Exercise the imported dependency instead of checking syntax alone."""

    from types import SimpleNamespace

    from agents.memory.run_forest_replay import is_historical_replay_anchor

    anchor = SimpleNamespace(
        replay_source={"historical_anchor_only": True},
        replay_status="historical_exact_anchor_loaded",
    )
    derived = SimpleNamespace(
        replay_source={"historical_anchor_only": False},
        replay_status="derived_modified_from_exact_source",
    )

    assert is_historical_replay_anchor(anchor) is True
    assert is_historical_replay_anchor(derived) is False
