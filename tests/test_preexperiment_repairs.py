from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import signal
import threading
import time
from types import SimpleNamespace

import pytest

from agents.memory.external_skill_memory import bounded_selector_max_tokens
from agents.memory.methodology_visibility import evaluate_methodology_visibility
from agents.triggers import register_node
from agents.triggers import refresh_replay_lineage_after_instrumentation
from authority.models import DecisionOutcome
from engine.search_node import SearchNode
from experiment_freeze import (
    FreezeError,
    create_freeze_manifest,
    verify_freeze_manifest,
)
from protocol_runtime.online_enforce_gate import (
    SmokeGateError,
    verify_online_enforce_smoke,
    write_smoke_manifest,
)


def test_base_config_declares_disabled_host_protocol_preflight_schema():
    from config import _load_cfg

    cfg = _load_cfg(use_cli_args=False)
    assert cfg.agent.protocol_preflight.enabled is False
    assert cfg.agent.protocol_preflight.binding_path == ""
    assert cfg.agent.protocol_preflight.consume_collector_private_key is False


def _simple_agent(tmp_path: Path, *, mode: str = "shadow"):
    from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
    from tests.authority.test_mlevolve_adapter import fake_agent

    tmp_path.mkdir(parents=True, exist_ok=True)
    agent = fake_agent(tmp_path, mode=mode)
    agent.task_desc = "small image classification task"
    agent.cfg.external_skill_memory = SimpleNamespace(visibility_token_budget=256)
    agent.cfg.agent = SimpleNamespace(
        feedback=SimpleNamespace(model="test-model"),
        code=SimpleNamespace(model="test-model"),
    )
    agent.evaluation_authority = MLEvolveAuthorityAdapter(agent)
    return agent


def test_modified_replay_descendant_loses_exact_status_but_identical_child_keeps_it():
    parent = SearchNode(
        code="print('source')",
        plan="source",
        stage="draft",
        branch_id=1,
        replay_source={
            "graph_node_id": "run::r::node::n",
            "code_sha256": hashlib.sha256(b"print('source')").hexdigest(),
        },
        replay_status="exact_source_loaded",
        claim_refs=["replay:source:claim"],
    )
    agent = SimpleNamespace(
        _serialize_prompt=str,
        next_branch_id=2,
        branch_all_nodes={1: [parent]},
        branch_successful_nodes={1: []},
    )
    identical = SearchNode(
        code=parent.code, plan="same", stage="improve", parent=parent
    )
    register_node(agent, identical, "same", parent_node=parent)
    assert identical.replay_status == "exact_source_loaded"
    assert identical.replay_source["exact_source_match"] is True

    modified = SearchNode(
        code="print('changed')", plan="changed", stage="improve", parent=parent
    )
    register_node(agent, modified, "changed", parent_node=parent)
    assert modified.replay_status == "derived_modified_from_exact_source"
    assert modified.replay_source["exact_source_match"] is False
    assert modified.derived_from_refs == ["replay:source:claim"]


def test_dynamic_modified_replay_becomes_novel_without_losing_source_provenance():
    source = "print('immutable replay')\n"
    parent = SearchNode(
        code=source,
        plan="source",
        stage="draft",
        branch_id=1,
        draft_role="memory_reproduction",
        role_contract={"role": "memory_reproduction"},
        replay_source={
            "graph_node_id": "run::official::node::best",
            "run_id": "official",
            "code_sha256": hashlib.sha256(source.encode()).hexdigest(),
        },
        replay_status="historical_exact_anchor_loaded",
    )
    agent = SimpleNamespace(
        _serialize_prompt=str,
        next_branch_id=2,
        branch_all_nodes={1: [parent]},
        branch_successful_nodes={1: []},
        acfg=SimpleNamespace(
            draft_role_policy=SimpleNamespace(
                enabled=True,
                replay_adaptation_as_novel=True,
            )
        ),
    )

    identical = SearchNode(
        code=source, plan="same", stage="improve", parent=parent
    )
    register_node(agent, identical, "same", parent_node=parent)
    assert identical.draft_role == "memory_reproduction"
    assert identical.replay_status == "historical_exact_anchor_loaded"

    modified = SearchNode(
        code="print('adapted replay')\n",
        plan="adapt",
        stage="improve",
        parent=parent,
    )
    register_node(agent, modified, "adapt", parent_node=parent)

    assert modified.draft_role == "novel_exploration"
    assert modified.replay_status == "replay_derived_novel_candidate"
    assert modified.replay_source["graph_node_id"] == "run::official::node::best"
    assert modified.replay_source["code_sha256"] == hashlib.sha256(
        source.encode()
    ).hexdigest()
    assert modified.replay_source["origin_draft_role"] == "memory_reproduction"
    assert modified.replay_source["adaptation_parent_node_id"] == parent.id
    assert modified.replay_source["lineage_kind"] == "replay_derived_novel"
    assert modified.role_contract["behavioral_role"] == "replay_derived_novel"


def test_host_instrumented_replay_is_derived_with_hash_bound_lineage_receipt():
    from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
    from protocol_runtime.preflight import (
        PreflightStatus,
        build_bounded_repair_receipt,
    )

    original = "def main():\n    print('source method')\n"
    instrumented = original + "\ndef candidate(session):\n    session.get_split()\n"
    receipt = build_bounded_repair_receipt(
        original,
        instrumented,
        preflight_status=PreflightStatus.MISSING_EVIDENCE.value,
        repair_kind="instrumentation",
        attempt=1,
        max_attempts=1,
    )
    node = SearchNode(
        code=instrumented,
        plan="host instrumented replay",
        stage="draft",
        replay_source={
            "graph_node_id": "run::r::node::n",
            "code_sha256": hashlib.sha256(original.encode()).hexdigest(),
        },
        replay_status="exact_source_loaded",
    )
    refresh_replay_lineage_after_instrumentation(
        node,
        original_code=original,
        instrumentation_receipt=receipt,
    )

    assert node.replay_status == "derived_modified_from_exact_source"
    assert node.replay_source["exact_source_match"] is False
    assert node.derived_from_refs == ["replay:run::r::node::n:method_hypothesis"]
    assert MLEvolveAuthorityAdapter._replay_lineage_preserved(node) is True
    node.replay_source["host_entrypoint_instrumentation_receipt"][
        "repaired_code_sha256"
    ] = "0" * 64
    assert MLEvolveAuthorityAdapter._replay_lineage_preserved(node) is False


def test_methodology_raw_candidates_are_written_before_authority_and_enforce_masks_legacy(
    tmp_path: Path,
):
    candidate = {
        "candidate_id": "methodology::bad",
        "claim_id": "methodology_claim::bad",
        "ref_id": "spooky/bad",
        "category": "spooky",
        "title": "Known invalid method",
        "text": "Use a contaminated terminal score.",
        "content_sha256": "a" * 64,
    }
    shadow = _simple_agent(tmp_path / "shadow", mode="shadow")
    text, refs, pack = evaluate_methodology_visibility(shadow, [candidate])
    raw = json.loads(
        (shadow.cfg.log_dir / "methodology_raw_candidates.json").read_text()
    )
    assert raw["raw_candidates"][0]["candidate_id"] == candidate["candidate_id"]
    assert text and refs
    assert pack is not None

    enforce = _simple_agent(tmp_path / "enforce", mode="enforce")
    text, refs, pack = evaluate_methodology_visibility(enforce, [candidate])
    assert text == ""
    assert refs == []
    assert pack is not None
    assert pack.suppressed_clause_refs


def test_selector_budget_is_clamped_and_payload_is_compact(monkeypatch):
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer
    import llm

    seen = {}

    def fake_query(**kwargs):
        seen.update(kwargs)
        return {"strategy_sop_id": "sop", "method_family": "linear"}

    monkeypatch.setattr(llm, "query", fake_query)
    cfg = SimpleNamespace(
        external_skill_memory=SimpleNamespace(selector_max_tokens=99999),
        agent=SimpleNamespace(
            feedback=SimpleNamespace(model="m"), code=SimpleNamespace(model="m")
        ),
    )
    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer.cfg = cfg
    layer._injected_strategy_selector = None
    layer._call_strategy_selector(
        task_profile={"task": "x", "long": "y" * 5000},
        routes=[
            {
                "sop_id": "sop",
                "method_family": "linear",
                "hypothesis": "h",
                "model_components": ["x"],
                "compute_profile": "cpu",
                "score": 1,
                "huge_unused_field": "z" * 10000,
            }
        ],
    )
    assert seen["max_tokens"] == 2400
    payload = json.loads(seen["user_message"])
    assert "huge_unused_field" not in payload["routes"][0]


def _freeze_fixture(tmp_path: Path):
    categories = (
        "code",
        "protocol",
        "data",
        "evaluator",
        "model",
        "config",
        "memory_bundle",
        "task_seed",
        "environment",
    )
    artifacts = []
    for category in categories:
        path = tmp_path / category
        path.mkdir()
        (path / "payload.txt").write_text(category, encoding="utf-8")
        artifacts.append({"name": category, "category": category, "path": category})
    spec = {
        "root": str(tmp_path),
        "container_image_digest": "registry.example/mlevolve@sha256:" + "1" * 64,
        "protocol_ref": "protocol@1#" + "2" * 64,
        "evaluator": {"id": "terminal-v1", "revision": "3" * 64},
        "model": {"id": "solver-v1", "revision": "4" * 64},
        "task_seeds": [{"task_id": "leaf-classification", "seed": 7}],
        "artifacts": artifacts,
    }
    spec_path = tmp_path / "freeze_spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


def test_freeze_manifest_hashes_all_required_inputs_and_rejects_mutation(tmp_path: Path):
    spec = _freeze_fixture(tmp_path)
    manifest_path = tmp_path / "FREEZE_MANIFEST.json"
    manifest = create_freeze_manifest(spec, manifest_path)
    verified = verify_freeze_manifest(manifest_path)
    assert verified["status"] == "verified"
    assert verified["manifest_hash"] == manifest["manifest_hash"]
    (tmp_path / "config" / "payload.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(FreezeError, match="Frozen artifact changed"):
        verify_freeze_manifest(manifest_path)


def test_freeze_rejects_floating_image_and_job_sync(tmp_path: Path):
    spec = _freeze_fixture(tmp_path)
    value = json.loads(spec.read_text())
    value["container_image_digest"] = "registry.example/mlevolve:latest"
    spec.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(FreezeError, match="immutable image digest"):
        create_freeze_manifest(spec, tmp_path / "bad.json")


def test_replay_contract_advances_l0_to_l3_and_publishes_one_adoption_edge(
    tmp_path: Path,
):
    from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
    from authority.memory_snapshot import MemorySnapshotLoader
    from authority.models import GenerationStage, GovernanceStage, Operation
    from tests.authority.test_enforce_rollout import _configure_scope
    from tests.authority.test_mlevolve_adapter import fake_agent, node
    from tests.test_memory_snapshot_overlay import build_tiny_bundle, write_current

    bundle, manifest = build_tiny_bundle(tmp_path)
    write_current(tmp_path, bundle, manifest)
    agent = fake_agent(tmp_path / "logs", mode="enforce")
    _configure_scope(
        agent,
        operations=[Operation.CODE_SEED.value, Operation.PUBLISH_ADOPTION.value],
        generation_stages=[GenerationStage.DEBUG.value, GenerationStage.DRAFT.value],
        governance_stages=[
            GovernanceStage.REPLAY.value,
            GovernanceStage.MEMORY_WRITEBACK.value,
        ],
    )
    adapter = MLEvolveAuthorityAdapter(agent)
    snapshot = MemorySnapshotLoader(tmp_path).load(
        session_overlay_path=tmp_path / "overlay",
        active_protocol_ref=adapter.active_protocol.key(),
        authority_policy_version=adapter.engine.policy_version,
    )
    adapter.configure_memory_snapshot(snapshot)
    code = "print('verified replay')"
    code_hash = hashlib.sha256(code.encode()).hexdigest()
    audit = {
        "schema": "mlevolve_leakage_audit_v2",
        "detector_version": "deterministic_static_v1",
        "detector_status": "complete",
        "status": "clean",
        "metric_disposition": "accept",
        "paper_grade_eligible": True,
        "hard_block": False,
        "code_sha256": code_hash,
        "issues": [],
    }
    source_id = "run::source::node::verified"
    source_decision = adapter.authorize_replay_source(
        artifact_id=source_id,
        code_sha256=code_hash,
        audit=audit,
        source_run_id="source",
        source_execution_verified=True,
    )
    assert source_decision.allowed

    candidate = node("replay-target", clean=True)
    candidate.code = code
    candidate.stage = "draft"
    candidate.replay_source = {
        "graph_node_id": source_id,
        "code_sha256": code_hash,
        "run_id": "source",
        "task_id": "task",
        "requires_repair": False,
        "sop_ids": ["sop::source"],
    }
    candidate.replay_status = "exact_source_loaded"
    first_ids = adapter.record_replay_exposure(candidate)
    second_ids = adapter.record_replay_exposure(candidate)
    assert first_ids == second_ids
    assert len(adapter.actuation_tracker.contracts_for_artifact(candidate.id)) == 1
    assert adapter.actuation_tracker.reports_for_artifact(candidate.id)[0].highest_level == 0

    reports = adapter.finalize_production_actuation(candidate)
    assert reports[0]["highest_level"] == 3
    edges = [
        event
        for event in snapshot.session_overlay.events()
        if event.event_type == "memory_derivation_edge"
    ]
    assert len(edges) == 1
    assert edges[0].payload["kind"] == "adoption"
    adapter.finalize_production_actuation(candidate)
    assert len(snapshot.session_overlay.events()) == 1


def test_unverified_replay_execution_cannot_bind_experience_contract(tmp_path: Path):
    from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
    from tests.authority.test_mlevolve_adapter import fake_agent, node

    agent = fake_agent(tmp_path, mode="shadow")
    adapter = MLEvolveAuthorityAdapter(agent)
    code = "print('unverified')"
    digest = hashlib.sha256(code.encode()).hexdigest()
    audit = {
        "schema": "mlevolve_leakage_audit_v2",
        "detector_status": "complete",
        "status": "clean",
        "metric_disposition": "accept",
        "paper_grade_eligible": True,
        "code_sha256": digest,
        "issues": [],
    }
    source = "run::unverified::node::1"
    decision = adapter.authorize_replay_source(
        artifact_id=source,
        code_sha256=digest,
        audit=audit,
        source_run_id="unverified",
        source_execution_verified=False,
    )
    assert decision.outcome != DecisionOutcome.ALLOW
    candidate = node("unverified-target", clean=True)
    candidate.code = code
    candidate.replay_source = {
        "graph_node_id": source,
        "code_sha256": digest,
        "requires_repair": False,
    }
    assert adapter.record_replay_exposure(candidate) == []


def test_bounded_adoption_timeout_status_cannot_race_back_to_complete(
    tmp_path: Path, monkeypatch
):
    from analysis import adoption_tracker

    release = threading.Event()

    def blocked(_cfg, _journal):
        release.wait(timeout=2)

    monkeypatch.setattr(adoption_tracker, "run_adoption_analysis", blocked)
    cfg = SimpleNamespace(
        log_dir=tmp_path,
        adoption_tracking=SimpleNamespace(analysis_timeout_seconds=0),
    )
    status = adoption_tracker.run_bounded_adoption_analysis(cfg, object())
    assert status["status"] == "timeout"
    release.set()
    time.sleep(0.02)
    persisted = json.loads((tmp_path / "adoption_analysis_status.json").read_text())
    assert persisted["status"] == "timeout"
    assert persisted["authoritative_for_run_status"] is False


def test_preflight_time_is_charged_to_candidate_deadline(
    tmp_path: Path, monkeypatch
):
    from engine import executor as executor_module
    from engine.executor import Interpreter

    preflight = SimpleNamespace(
        enabled=True,
        contract_path="",
        collector_private_key_path="",
        report_root=str(tmp_path / "preflight"),
        expected_contract_hash="c" * 64,
        candidate_uid=65534,
    )
    cfg = SimpleNamespace(
        agent=SimpleNamespace(
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
            protocol_preflight=preflight,
        ),
        evaluation_authority=SimpleNamespace(
            mode="enforce", protocol_runtime_mode="host_sdk_enforce"
        ),
        start_cpu_id=0,
        cpu_number=1,
    )
    interpreter = Interpreter(tmp_path, timeout=60, cfg=cfg)
    interpreter.set_run_deadline(
        time.monotonic() + 60, finalize_reserve_seconds=1
    )

    def consume_budget(*_args, **_kwargs):
        interpreter.deadline_monotonic = time.monotonic() + 1
        return {"status": "pass", "report_hash": "r" * 64}

    monkeypatch.setattr(
        executor_module, "validate_preflight_admission", consume_budget
    )
    monkeypatch.setattr(
        executor_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("full subprocess must not start"),
    )
    result = interpreter.run("print('never launched')", "deadline-case")
    assert result.exc_type == "RunDeadlineExceeded"
    assert result.exc_info["candidate_subprocess_started"] is False


def test_sigterm_finalizer_writes_partial_outcome_and_restores_handler(
    tmp_path: Path, monkeypatch
):
    run_module = importlib.import_module("run")
    terminated = []
    checkpointed = []
    interpreter = SimpleNamespace(
        terminate_all_subprocesses=lambda: terminated.append(True),
        active_candidate_ids=lambda: ["candidate-a"],
    )
    state = {
        "completed": 2,
        "total_steps": 8,
        "journal": object(),
        "agent": SimpleNamespace(best_node=object()),
        "interpreter": interpreter,
        "terminating": False,
    }
    monkeypatch.setattr(
        run_module, "save_run", lambda _cfg, _journal: checkpointed.append(True)
    )
    logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    finalizer = run_module.SigtermFinalizer(
        SimpleNamespace(log_dir=tmp_path), logger, state
    )
    with pytest.raises(SystemExit) as exit_info:
        finalizer(signal.SIGTERM, None)
    assert exit_info.value.code == 128 + signal.SIGTERM
    assert terminated and checkpointed
    outcome = json.loads((tmp_path / "RUN_OUTCOME.json").read_text())
    assert outcome["status"] == "partial"
    assert outcome["active_candidate_ids"] == ["candidate-a"]

    original = signal.getsignal(signal.SIGTERM)

    def caller_handler(_signum, _frame):
        return None

    signal.signal(signal.SIGTERM, caller_handler)
    monkeypatch.setattr(run_module, "_run_impl", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        with pytest.raises(RuntimeError, match="boom"):
            run_module.run()
        assert signal.getsignal(signal.SIGTERM) is caller_handler
    finally:
        signal.signal(signal.SIGTERM, original)


def test_formal_job_forwards_sigterm_to_active_python_child():
    manifest = Path("deploy/prevalence-audit-20260729-five-a100.yaml").read_text(
        encoding="utf-8"
    )
    assert 'active_pid=""' in manifest
    assert "trap forward_term TERM INT" in manifest
    assert 'kill -TERM "${active_pid}"' in manifest
    assert 'active_pid=$!' in manifest
    assert 'wait "${active_pid}"' in manifest


def _smoke_case(
    case_id: str,
    task_id: str,
    *,
    expected_legal: bool,
    case_class: str,
    raw: list[str],
    suppressed: list[str],
    final: list[str],
):
    reasons = {
        candidate: {
            "claim_id": f"claim::{candidate}",
            "operation": "generate_candidate",
            "decision_stage": "draft",
            "protocol_ref": "protocol@1#" + "2" * 64,
            "receipt_refs": [f"receipt::{candidate}"],
        }
        for candidate in suppressed
    }
    return {
        "case_id": case_id,
        "task_id": task_id,
        "case_class": case_class,
        "expected_legal": expected_legal,
        "decision_stage": "draft",
        "operation": "generate_candidate",
        "raw_candidate_ids": raw,
        "raw_claim_ids": [f"claim::{candidate}" for candidate in raw],
        "suppressed_candidate_ids": suppressed,
        "suppression_reasons": reasons,
        "final_prompt_candidate_ids": final,
        "prompt_visible_invalid_candidate_ids": [],
        "shadow_authority_decisions": [
            {
                "claim_id": f"claim::{candidate}",
                "outcome": "deny" if candidate in suppressed else "allow",
                "decision_ref": f"decision::{candidate}",
            }
            for candidate in raw
        ],
        "host_evidence": {
            "status": "pass" if expected_legal else "protocol_violation",
            "closure_hash": "c" * 64 if expected_legal else "",
            "runtime_receipt_refs": [f"runtime::{case_id}"] if expected_legal else [],
            "terminal_exposure_count": 0,
        },
        "evidence_files": [],
    }


def test_online_enforce_release_gate_requires_legal_invalid_and_mixed_cases(
    tmp_path: Path,
):
    freeze_hash = "a" * 64
    image = "registry.example/mlevolve@sha256:" + "b" * 64
    cases = [
        _smoke_case(
            "denoise-legal",
            "denoising-dirty-documents",
            expected_legal=True,
            case_class="legal",
            raw=["denoise-valid"],
            suppressed=[],
            final=["denoise-valid"],
        ),
        _smoke_case(
            "leaf-legal",
            "leaf-classification",
            expected_legal=True,
            case_class="legal",
            raw=["leaf-valid"],
            suppressed=[],
            final=["leaf-valid"],
        ),
        _smoke_case(
            "spooky-invalid",
            "spooky-author-identification",
            expected_legal=False,
            case_class="known_invalid",
            raw=["spooky-invalid"],
            suppressed=["spooky-invalid"],
            final=[],
        ),
        _smoke_case(
            "mixed",
            "leaf-classification",
            expected_legal=True,
            case_class="mixed_value",
            raw=["mixed-valid", "mixed-invalid"],
            suppressed=["mixed-invalid"],
            final=["mixed-valid"],
        ),
    ]
    smoke_path = tmp_path / "ONLINE_ENFORCE_SMOKE.json"
    write_smoke_manifest(
        smoke_path,
        {
            "freeze_manifest_hash": freeze_hash,
            "container_image_digest": image,
            "runtime": {
                "authority_mode": "enforce",
                "protocol_runtime_mode": "host_sdk_enforce",
                "execution_environment": "online_gpu",
                "gpu_probe_passed": True,
            },
            "cases": cases,
        },
    )
    report = verify_online_enforce_smoke(
        smoke_path,
        required_freeze_hash=freeze_hash,
        required_image_digest=image,
    )
    assert report["status"] == "passed"
    assert report["checks"]["legal_false_denial_zero"] is True

    payload = json.loads(smoke_path.read_text())
    payload["cases"][2]["prompt_visible_invalid_candidate_ids"] = ["spooky-invalid"]
    payload["manifest_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "manifest_hash"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    bad = tmp_path / "bad-smoke.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SmokeGateError, match="release gate is blocked"):
        verify_online_enforce_smoke(bad)
