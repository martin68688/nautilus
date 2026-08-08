from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mlevolve"))

from agents.memory.experiment_r_router import (
    _experiment_r_clean_sop_support,
    _prompt_marker_visible,
    _same_task_best_rows,
    _truncate_prompt,
    count_prompt_tokens,
)
from agents.adoption import log_adoption
from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer
from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from authority.ledger import AuthorityLedger
from authority.models import ProtocolRef, ProtocolSpec
from authority.protocol_execution_contract import ProtocolExecutionContract
from config import _load_cfg
from protocol_runtime.collector import (
    HostCollectorIdentity,
    verify_host_canonical_signature,
)
from experiments.dynamic_memory_routing_injection_20260731.design import (
    HELDOUT_RUN_IDS,
    MAX_INJECTED_ITEMS,
    MEMORY_PROMPT_TOKEN_BUDGET,
    ONLINE_SYSTEMS,
    RAW_CANDIDATES_PER_SOURCE,
    validate_design,
)
from tests.test_stage_aware_hybrid_memory import _write_fixture


CONFIG = (
    ROOT / "mlevolve" / "config" / "config_experiment_r_dynamic_memory_routing.yaml"
)


def _host_contract(tmp_path: Path, identity: HostCollectorIdentity):
    protocol_ref = ProtocolRef(
        protocol_id="experiment-r-test",
        version="1",
        canonical_hash="e" * 64,
    )
    contract = ProtocolExecutionContract.create(
        protocol_ref=protocol_ref,
        task_id="leaf-classification",
        task_family="image",
        split_strategy="deterministic_random",
        train_view_ref="view://leaf/train",
        validation_view_ref="view://leaf/validation",
        terminal_view_ref="evaluator-only://leaf/terminal",
        required_runtime_events=(),
        required_receipts=(),
        required_payloads={},
        allowed_import_roots=(),
        execution_budget={"timeout_seconds": 60},
        evaluator_spec={},
        collector_spec=identity.collector_spec(),
        adapter_spec={},
    )
    contract_path = tmp_path / "PROTOCOL_EXECUTION_CONTRACT.json"
    contract_path.write_text(
        contract.canonical_json() + "\n",
        encoding="utf-8",
    )
    protocol_spec = ProtocolSpec(
        protocol_id=protocol_ref.protocol_id,
        version=protocol_ref.version,
        canonical_hash=protocol_ref.canonical_hash,
    )
    return protocol_spec, contract, contract_path


def _collector_adapter(
    *,
    protocol_spec: ProtocolSpec,
    contract: ProtocolExecutionContract,
    contract_path: Path,
    key_path: Path,
):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            exp_id="leaf-classification",
            agent=SimpleNamespace(
                protocol_preflight=SimpleNamespace(
                    contract_path=str(contract_path),
                    expected_contract_hash=contract.contract_hash,
                    collector_private_key_path=str(key_path),
                )
            ),
        ),
        active_protocol_spec=protocol_spec,
        active_protocol=protocol_spec.ref(),
        task_id="leaf-classification",
    )


def _layer(tmp_path: Path, control: str, *, excluded_run_ids=None):
    graph, index = _write_fixture(tmp_path)
    return StageAwareHybridMemoryLayer(
        graph_path=str(graph),
        index_path=str(index),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        enable_agentic=False,
        top_k=MAX_INJECTED_ITEMS,
        max_chars=0,
        retrieval_control=control,
        visibility_mode="shadow",
        excluded_run_ids=excluded_run_ids or [],
        experiment_r_enabled=True,
        experiment_r_candidate_limit=RAW_CANDIDATES_PER_SOURCE,
        experiment_r_top_k=MAX_INJECTED_ITEMS,
        experiment_r_prompt_token_budget=MEMORY_PROMPT_TOKEN_BUDGET,
        experiment_r_memory_pool_sha256="a" * 64,
    )


def _retrieve(layer, stage="draft", query="transformer validation ensemble"):
    text, refs = layer.retrieve_for_node(
        stage=stage,
        task_id="task",
        task_desc="text classification",
        query_parts=[query],
        draft_role="memory_transfer",
    )
    return text, refs, layer.current_navigation_pack()


def test_frozen_design_and_config_are_structurally_valid():
    validate_design()
    cfg = _load_cfg(CONFIG, use_cli_args=False)
    ext = cfg.external_skill_memory
    assert ext.experiment_r_enabled is True
    assert ext.retrieval_control == "dynamic_hybrid"
    assert ext.top_k == MAX_INJECTED_ITEMS == ext.experiment_r_top_k
    assert ext.experiment_r_candidate_limit == RAW_CANDIDATES_PER_SOURCE
    assert ext.experiment_r_prompt_token_budget == MEMORY_PROMPT_TOKEN_BUDGET
    assert tuple(ext.excluded_run_ids) == HELDOUT_RUN_IDS
    assert cfg.evaluation_authority.mode == "enforce"
    assert cfg.methodology_dynamic is False
    assert cfg.methodology_kb_path == ""


def test_host_candidate_pool_attestation_signature_is_fail_closed():
    identity = HostCollectorIdentity.generate()
    payload = {
        "schema": "mlevolve_experiment_r_host_candidate_pool_attestation_v1",
        "candidate_pool_hash": "a" * 64,
        "final_prompt_candidate_ids": ["sop::1", "run::1"],
    }
    signature = identity.sign_canonical_payload(payload)
    verify_host_canonical_signature(
        payload,
        signature_ed25519=signature,
        public_key_ed25519=identity.public_key_ed25519,
    )
    with pytest.raises(ValueError, match="signature mismatch"):
        verify_host_canonical_signature(
            {**payload, "candidate_pool_hash": "b" * 64},
            signature_ed25519=signature,
            public_key_ed25519=identity.public_key_ed25519,
        )


def test_enforce_runtime_writes_signed_candidate_pool_attestation(
    tmp_path, monkeypatch
):
    identity = HostCollectorIdentity.generate()
    key_path = identity.write_private_key_file(tmp_path / "collector.ed25519")
    monkeypatch.setenv("MLEVOLVE_HOST_COLLECTOR_KEY_FILE", str(key_path))
    pool_identity = {
        "stage": "improve",
        "task_id": "leaf-classification",
        "memory_pool_sha256": "a" * 64,
        "sop_ids": ["sop::1"],
        "runforest_ids": ["run::1"],
    }
    import hashlib
    import json

    pool_hash = hashlib.sha256(
        json.dumps(
            pool_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    trace = {
        "stage_route": {"stage": "improve"},
        "memory_pool_sha256": "a" * 64,
        "memory_snapshot_sha256": "b" * 64,
        "candidate_pool_hash": pool_hash,
        "candidate_pool_identity": pool_identity,
        "final_prompt_candidate_ids": ["sop::1", "run::1"],
        "visible_clause_ids": ["clause::1"],
        "budget_contract": {
            "max_injected_items": 6,
            "memory_prompt_token_budget": 1536,
        },
        "production_binding_sha256": "c" * 64,
        "current_file_sha256": "d" * 64,
    }
    adapter = SimpleNamespace(
        cfg=SimpleNamespace(
            external_skill_memory=SimpleNamespace(experiment_r_enabled=True)
        ),
        mode="enforce",
        task_id="leaf-classification",
        run_id="run::1",
        active_protocol=SimpleNamespace(key=lambda: "protocol@1#" + "e" * 64),
        active_protocol_spec=ProtocolSpec(
            protocol_id="experiment-r-test",
            version="1",
            canonical_hash="e" * 64,
        ),
        ledger=AuthorityLedger(tmp_path / "authority_events.jsonl"),
        _experiment_r_collector_identity=identity,
        _experiment_r_collector_public_key=identity.public_key_ed25519,
    )
    # The enforced Executor consumes this file before Candidate execution. The
    # Host Authority adapter must retain its separately pinned in-memory copy.
    key_path.unlink()
    node = SimpleNamespace(id="node::1", memory_routing_trace=trace)
    result = MLEvolveAuthorityAdapter.attest_experiment_r_candidate_pool(adapter, node)
    assert result["attestation_id"].startswith("candidate-pool::")
    assert trace["host_candidate_pool_attestation_ref"] == result["attestation_id"]
    events = adapter.ledger.read()
    assert events[-1]["event_type"] == "experiment_r_candidate_pool_attested"
    payload = events[-1]["payload"]
    verify_host_canonical_signature(
        payload["attestation"],
        signature_ed25519=payload["signature_ed25519"],
        public_key_ed25519=identity.public_key_ed25519,
    )


def test_experiment_r_collector_identity_is_pinned_before_key_consumption(
    tmp_path,
):
    identity = HostCollectorIdentity.generate()
    key_path = identity.write_private_key_file(tmp_path / "collector.ed25519")
    protocol_spec, contract, contract_path = _host_contract(tmp_path, identity)
    adapter = _collector_adapter(
        protocol_spec=protocol_spec,
        contract=contract,
        contract_path=contract_path,
        key_path=key_path,
    )
    pinned = MLEvolveAuthorityAdapter._load_experiment_r_collector_identity(adapter)
    key_path.unlink()
    assert pinned.public_key_ed25519 == identity.public_key_ed25519
    assert pinned.sign_canonical_payload({"after": "consumption"})


def test_experiment_r_collector_identity_binding_mismatches_fail_closed(tmp_path):
    identity = HostCollectorIdentity.generate()
    key_path = identity.write_private_key_file(tmp_path / "collector.ed25519")
    protocol_spec, contract, contract_path = _host_contract(tmp_path, identity)

    hash_mismatch = _collector_adapter(
        protocol_spec=protocol_spec,
        contract=contract,
        contract_path=contract_path,
        key_path=key_path,
    )
    hash_mismatch.cfg.agent.protocol_preflight.expected_contract_hash = "0" * 64
    with pytest.raises(ValueError, match="contract hash mismatch"):
        MLEvolveAuthorityAdapter._load_experiment_r_collector_identity(hash_mismatch)

    protocol_mismatch = _collector_adapter(
        protocol_spec=protocol_spec,
        contract=contract,
        contract_path=contract_path,
        key_path=key_path,
    )
    protocol_mismatch.active_protocol = ProtocolRef(
        protocol_id="foreign-protocol",
        version="1",
        canonical_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="contract is not active"):
        MLEvolveAuthorityAdapter._load_experiment_r_collector_identity(
            protocol_mismatch
        )

    task_mismatch = _collector_adapter(
        protocol_spec=protocol_spec,
        contract=contract,
        contract_path=contract_path,
        key_path=key_path,
    )
    task_mismatch.task_id = "spooky-author-identification"
    with pytest.raises(ValueError, match="contract task mismatch"):
        MLEvolveAuthorityAdapter._load_experiment_r_collector_identity(task_mismatch)

    unbound_identity = HostCollectorIdentity.generate()
    unbound_key_path = unbound_identity.write_private_key_file(
        tmp_path / "unbound-collector.ed25519"
    )
    key_mismatch = _collector_adapter(
        protocol_spec=protocol_spec,
        contract=contract,
        contract_path=contract_path,
        key_path=unbound_key_path,
    )
    with pytest.raises(ValueError, match="key is not Protocol-bound"):
        MLEvolveAuthorityAdapter._load_experiment_r_collector_identity(key_mismatch)


def test_all_memory_arms_share_one_candidate_pool_and_budget(tmp_path):
    hashes = set()
    identities = set()
    for control in ONLINE_SYSTEMS:
        layer = _layer(tmp_path / control, control)
        text, refs, pack = _retrieve(layer)
        assert pack["schema"] == "experiment_r_memory_pack_v1"
        assert pack["budget_contract"]["max_injected_items"] == MAX_INJECTED_ITEMS
        assert (
            pack["budget_contract"]["memory_prompt_token_budget"]
            == MEMORY_PROMPT_TOKEN_BUDGET
        )
        assert len(refs) <= MAX_INJECTED_ITEMS
        assert count_prompt_tokens(text) <= MEMORY_PROMPT_TOKEN_BUDGET
        assert pack["safety_gate"]["unsafe_candidate_escape_count"] == 0
        if control == "no_memory":
            assert text == "" and refs == []
            assert pack["memory_snapshot_bound_but_not_exposed"] is True
        hashes.add(pack["candidate_pool_hash"])
        assert pack["final_prompt_candidate_ids"] == refs
        assert set(refs) == {row["id"] for row in pack["selected_items"]}
        identity = pack["candidate_pool"]["pool_identity"]
        identities.add((tuple(identity["sop_ids"]), tuple(identity["runforest_ids"])))
    assert len(hashes) == 1
    assert len(identities) == 1


def test_paired_route_uses_hash_bound_qualification_pool_not_live_query(
    tmp_path, monkeypatch
):
    source = _layer(tmp_path / "source", "dynamic_hybrid")
    _text, _refs, source_pack = _retrieve(
        source, query="qualification query that selected the frozen universe"
    )
    source_pool = source_pack["candidate_pool"]
    artifact = tmp_path / "candidate_pool.json"
    artifact.write_text(
        json.dumps(
            {
                "candidate_pool_hash": source_pool["candidate_pool_hash"],
                "candidate_pool_identity": source_pool["pool_identity"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setenv("EXPERIMENT_R_QUALIFICATION_CANDIDATE_POOL", str(artifact))
    monkeypatch.setenv(
        "EXPERIMENT_R_QUALIFICATION_CANDIDATE_POOL_SHA256", artifact_sha256
    )
    monkeypatch.setenv(
        "EXPERIMENT_R_QUALIFICATION_CANDIDATE_POOL_HASH",
        source_pool["candidate_pool_hash"],
    )
    monkeypatch.setenv("EXPERIMENT_R_QUALIFICATION_CHECKPOINT_ID", "checkpoint::frozen")

    paired = _layer(tmp_path / "paired", "reversed_router")
    _text, _refs, paired_pack = _retrieve(
        paired, query="a deliberately different online LLM-cleaned query"
    )
    assert paired_pack["candidate_pool_hash"] == source_pool["candidate_pool_hash"]
    assert (
        paired_pack["candidate_pool"]["pool_identity"] == source_pool["pool_identity"]
    )
    assert paired_pack["candidate_pool_source"] == "qualification_checkpoint_artifact"
    assert paired_pack["qualification_checkpoint_id"] == "checkpoint::frozen"
    assert (
        paired_pack["qualification_candidate_pool_artifact_sha256"] == artifact_sha256
    )
    assert paired_pack["ranking_contract"] == "qualification_frozen_source_rank_v1"
    assert paired_pack["live_query_used_for_candidate_pool"] is False


def test_paired_route_rejects_substituted_qualification_pool(tmp_path, monkeypatch):
    source = _layer(tmp_path / "source", "dynamic_hybrid")
    _text, _refs, source_pack = _retrieve(source)
    source_pool = source_pack["candidate_pool"]
    artifact = tmp_path / "candidate_pool.json"
    artifact.write_text(
        json.dumps(
            {
                "candidate_pool_hash": source_pool["candidate_pool_hash"],
                "candidate_pool_identity": source_pool["pool_identity"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXPERIMENT_R_QUALIFICATION_CANDIDATE_POOL", str(artifact))
    monkeypatch.setenv("EXPERIMENT_R_QUALIFICATION_CANDIDATE_POOL_SHA256", "0" * 64)
    monkeypatch.setenv(
        "EXPERIMENT_R_QUALIFICATION_CANDIDATE_POOL_HASH",
        source_pool["candidate_pool_hash"],
    )
    monkeypatch.setenv("EXPERIMENT_R_QUALIFICATION_CHECKPOINT_ID", "checkpoint::frozen")
    paired = _layer(tmp_path / "paired", "dynamic_hybrid")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        _retrieve(paired, query="different query")


def test_no_memory_persists_a_bound_zero_exposure_routing_trace(tmp_path):
    layer = _layer(tmp_path, "no_memory")
    text, refs, pack = _retrieve(layer)
    assert text == "" and refs == []
    node = SimpleNamespace(
        adoption_log=[],
        memory_navigation_trace=[],
        memory_routing_trace={},
    )
    agent = SimpleNamespace(
        external_skill_memory=layer,
        cfg=SimpleNamespace(
            run_identity=SimpleNamespace(
                memory_bundle_binding_sha256="b" * 64,
                memory_current_sha256="c" * 64,
            )
        ),
    )

    log_adoption(
        node,
        agent,
        "run_forest_stage_hybrid_memory",
        refs,
        "draft",
    )

    trace = node.memory_routing_trace
    assert trace["memory_pack_schema"] == "experiment_r_memory_pack_v1"
    assert trace["stage_route"]["control"] == "no_memory"
    assert trace["memory_pool_sha256"] == "a" * 64
    assert trace["memory_snapshot_bound_but_not_exposed"] is True
    assert trace["final_prompt_candidate_ids"] == []
    assert trace["production_binding_sha256"] == "b" * 64
    assert node.adoption_log == []


def test_prompt_truncation_counts_the_marker_inside_the_budget():
    text, count, truncated = _truncate_prompt(
        " ".join(f"token-{i}" for i in range(20)), 5
    )
    assert truncated is True
    assert count == count_prompt_tokens(text) == 5


def test_prompt_visibility_requires_an_exact_item_marker():
    row = {"source": "runforest", "id": "node::abc"}
    assert _prompt_marker_visible("- [RunForest] node::abc type=RunNode", row)
    assert not _prompt_marker_visible("Evidence mentions node::abc incidentally.", row)


def test_dynamic_and_reversed_router_change_only_source_allocation(tmp_path):
    observed = {}
    for control in ("dynamic_hybrid", "static_hybrid", "reversed_router"):
        layer = _layer(tmp_path / control, control)
        _text, _refs, pack = _retrieve(layer, stage="draft")
        observed[control] = pack
    pool_hashes = {pack["candidate_pool_hash"] for pack in observed.values()}
    assert len(pool_hashes) == 1
    assert observed["dynamic_hybrid"]["stage_route"]["requested_slots"] == {
        "sop": 5,
        "runforest": 1,
    }
    assert observed["static_hybrid"]["stage_route"]["requested_slots"] == {
        "sop": 3,
        "runforest": 3,
    }
    assert observed["reversed_router"]["stage_route"]["requested_slots"] == {
        "sop": 2,
        "runforest": 4,
    }


def test_debug_dynamic_falls_back_when_causal_transition_confidence_is_low(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    _text, _refs, pack = _retrieve(
        layer,
        stage="debug",
        query="an unclassified failure without a known mechanism",
    )
    assert (
        pack["stage_route"]["fallback_reason"] == "insufficient_causal_tree_confidence"
    )
    assert pack["stage_route"]["requested_slots"] == {
        "sop": 1,
        "runforest": 5,
    }


def test_heldout_run_is_removed_from_both_sop_support_and_runforest(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid", excluded_run_ids=["clean_run"])
    _text, refs, pack = _retrieve(layer)
    assert "n0" not in refs and "n1" not in refs and "t1" not in refs
    pool = pack["candidate_pool"]
    assert not pool["pool_identity"]["runforest_ids"]
    assert not pool["pool_identity"]["sop_ids"]


def test_agentic_router_searches_same_task_best_first_without_role_policy(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_agentic_max_steps = 2
    actions = iter(
        [
            {
                "action": "search_runforest",
                "reason": "look for another same-task improvement",
                "query": "task best validation improvement",
                "top_k": 4,
            },
            {
                "action": "finish",
                "reason": "enough clean same-task evidence",
                "selected_ids": ["n1", "s1", "t1", "n0"],
            },
        ]
    )
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: next(actions)

    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="task",
        task_desc="text classification",
        query_parts=["build the best reliable model"],
        draft_role=None,
    )
    pack = layer.current_navigation_pack()
    agent = pack["retrieval_agent"]
    assert text and refs
    assert pack["candidate_pool_source"] == "live_agentic_retrieval"
    assert pack["ranking_contract"] == (
        "authority_tool_agentic_final_selection_v2" "+same_task_best_prompt_pin_v1"
    )
    assert pack["algorithm_version"] == "experiment_r_agentic_final_selection_v2"
    assert agent["fallback_used"] is False
    assert agent["trace"][0]["action"] == "search_same_task_best"
    same_task_ids = agent["trace"][0]["observation"]["candidate_ids"]
    # The fixture has no trustworthy metric direction, so positive
    # metric_improvement is the fail-closed best-history signal. This proves
    # the mandatory first pass is not ordinary prompt-text similarity.
    assert same_task_ids[:3] == ["n1", "t1", "n0"]
    assert agent["same_task_best_first"] == {
        "enforced": True,
        "independent_of_draft_role_policy": True,
        "target_task_id": "task",
        "eligible_history_found": True,
        "observed_candidate_ids": same_task_ids,
        "best_runforest_id": "n1",
        "best_sop_id": "s1",
        "ranking_contract": "same_task_best_history_v2",
        "prompt_pin": {
            "required": True,
            "candidate_id": "n1",
            "source": "runforest",
            "quota_preserving": True,
            "applied": True,
            "prompt_visible": True,
        },
    }
    assert agent["agent_selected_ids"] == ["n1", "s1", "t1", "n0"]
    assert agent["effective_selected_ids"] == ["n1", "s1", "t1", "n0"]
    assert agent["selection_complete"] is True
    assert pack["stage_route"]["route"] == ("dynamic_hybrid_agent_final_selection")
    assert pack["stage_route"]["decision_authority"] == "retrieval_agent"
    assert pack["stage_route"]["deterministic_quota_selection_used"] is False
    assert (
        pack["candidate_pool"]["pool_identity"]["retrieval_agent_trace_sha256"]
        == agent["trace_sha256"]
    )


def test_agentic_router_forces_finish_after_repeated_no_progress_searches(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_agentic_max_steps = 4
    calls = []
    actions = iter(
        [
            {
                "action": "search_runforest",
                "reason": "first rewritten search",
                "query": "task best validation improvement",
                "top_k": 4,
            },
            {
                "action": "search_runforest",
                "reason": "second rewritten search",
                "query": "task strongest clean historical execution",
                "top_k": 4,
            },
            {
                "action": "finish",
                "reason": "mandatory final decision",
                "selected_ids": ["n1", "s1", "t1", "n0"],
            },
        ]
    )

    def query_fn(**kwargs):
        calls.append(kwargs)
        return next(actions)

    layer._experiment_r_agentic_query_fn = query_fn
    _text, _refs, pack = _retrieve(layer)
    agent = pack["retrieval_agent"]

    assert agent["agent_calls"] == 3
    assert agent["forced_finalization_used"] is True
    assert agent["selection_complete"] is True
    assert [row["action"] for row in agent["trace"][-3:]] == [
        "search_runforest",
        "search_runforest",
        "finish",
    ]
    final_prompt = calls[-1]["system_message"]
    decision_budget = json.loads(final_prompt["decision_budget"])
    assert decision_budget["must_finish_now"] is True
    assert decision_budget["consecutive_searches_without_new_candidates"] == 2
    final_schema = calls[-1]["func_spec"].json_schema
    assert final_schema["properties"]["action"]["enum"] == ["finish"]
    assert final_schema["properties"]["selected_ids"]["minItems"] == 4
    assert "selected_ids" in final_schema["required"]


def test_agentic_inspect_returns_known_candidate_content(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_agentic_max_steps = 2
    actions = iter(
        [
            {
                "action": "inspect_candidate",
                "reason": "inspect the strongest same-task execution",
                "candidate_id": "n1",
            },
            {
                "action": "finish",
                "reason": "inspection complete",
                "selected_ids": ["n1", "s1", "t1", "n0"],
            },
        ]
    )
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: next(actions)

    _text, _refs, pack = _retrieve(layer)
    inspection = next(
        row
        for row in pack["retrieval_agent"]["trace"]
        if row["action"] == "inspect_candidate"
    )["observation"]
    assert inspection["candidate_ids"] == ["n1"]
    assert inspection["new_candidate_ids"] == []
    assert inspection["candidates"][0]["id"] == "n1"
    assert "transformer validation ensemble" in inspection["candidates"][0]["summary"]


def test_same_task_best_history_respects_minimize_metric_direction(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.nodes["n0"]["metric"] = 0.42
    layer.nodes["n0"]["maximize"] = False
    layer.nodes["n1"]["metric"] = 0.31
    layer.nodes["n1"]["maximize"] = False
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_agentic_max_steps = 1
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: {
        "action": "finish",
        "reason": "best direction-aware history observed",
        "selected_ids": ["n1", "s1", "t1", "n0"],
    }

    layer.retrieve_for_node(
        stage="draft",
        task_id="task",
        task_desc="text classification with log loss",
        query_parts=["build a reliable model"],
        draft_role=None,
    )
    agent = layer.current_navigation_pack()["retrieval_agent"]
    assert agent["same_task_best_first"]["best_runforest_id"] == "n1"
    first_rows = agent["trace"][0]["observation"]["candidates"]
    assert [row["id"] for row in first_rows[:2]] == ["n1", "n0"]


def test_fast_experiment_accepts_successful_history_without_paper_grade_markers(
    tmp_path,
):
    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.cfg = SimpleNamespace(evaluation_authority=SimpleNamespace(mode="off"))
    layer.memory_snapshot = SimpleNamespace(verify_artifacts=False)
    for node_id in ("n0", "n1"):
        layer.nodes[node_id]["leakage_audit"] = {
            "status": "clean",
            "legacy_receipt_level": "legacy_static_only",
        }
    layer.experiment_r_agentic_retrieval_enabled = True
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: {
        "action": "finish",
        "reason": "use the successful same-task best",
        "selected_ids": ["n1", "s1", "s2", "t1", "n0"],
    }

    _text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="task",
        task_desc="text classification with log loss",
        query_parts=["start from the strongest historical method"],
        draft_role="memory_transfer",
    )
    pack = layer.current_navigation_pack()
    same_task = pack["retrieval_agent"]["same_task_best_first"]
    assert same_task["best_runforest_id"] == "n1"
    assert same_task["prompt_pin"]["prompt_visible"] is True
    assert "n1" in refs
    assert pack["stage_route"]["realized_slots"] == {
        "sop": 2,
        "runforest": 3,
    }


def test_fast_experiment_respects_modern_positive_admission_labels(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.cfg = SimpleNamespace(evaluation_authority=SimpleNamespace(mode="off"))
    layer.memory_snapshot = SimpleNamespace(verify_artifacts=False)
    layer.nodes["n0"]["metric"] = 0.01
    layer.nodes["n0"]["leakage_audit"] = {
        "status": "blocked",
        "memory_disposition": "quarantine",
        "paper_grade_eligible": False,
        "rank_eligible": False,
    }
    layer.nodes["n1"]["metric"] = 0.10
    layer.nodes["n1"]["leakage_audit"] = {
        "status": "clean",
        "memory_disposition": "positive_eligible",
        "paper_grade_eligible": True,
        "rank_eligible": True,
    }
    assert layer._execution_candidate_eligibility("n0") == (
        False,
        "run_node_not_rank_eligible",
    )
    assert layer._execution_candidate_eligibility("n1") == (
        True,
        "clean_successful_run_node",
    )
    rows = _same_task_best_rows(
        layer,
        task_id="task",
        visible_sop_ids=None,
        limit=8,
    )
    runforest_ids = [row["id"] for row in rows if row["source"] == "runforest"]
    assert runforest_ids == ["n1", "t1"]


def test_dynamic_prompt_pins_same_task_best_when_retrieval_agent_declines_it(
    tmp_path,
):
    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.experiment_r_top_k = 3
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_agentic_max_steps = 1
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: {
        "action": "finish",
        "reason": "prefer two other observed executions",
        "selected_ids": ["t1", "n0", "s1"],
    }

    _text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="task",
        task_desc="text classification",
        query_parts=["build a reliable model"],
        draft_role=None,
    )
    pack = layer.current_navigation_pack()
    pin = pack["retrieval_agent"]["same_task_best_first"]["prompt_pin"]
    assert pin == {
        "required": True,
        "candidate_id": "n1",
        "source": "runforest",
        "quota_preserving": True,
        "applied": True,
        "prompt_visible": True,
    }
    assert "n1" in refs
    assert "n1" in pack["final_prompt_candidate_ids"]
    assert pack["retrieval_agent"]["agent_selected_ids"] == ["t1", "n0", "s1"]
    assert pack["retrieval_agent"]["effective_selected_ids"] == [
        "t1",
        "n1",
        "s1",
    ]
    assert pack["retrieval_agent"]["selection_overrides"] == [
        {
            "reason": "mandatory_same_task_best",
            "inserted_id": "n1",
            "replaced_id": "n0",
            "source": "runforest",
            "quota_preserving": True,
        }
    ]
    assert pack["stage_route"]["requested_slots"] == {
        "sop": 5,
        "runforest": 1,
    }
    # The tiny fixture has only one eligible SOP. The Agent chooses the full
    # realizable three-item set, then the mandatory best-history invariant
    # replaces one RunForest choice without changing its source count.
    assert pack["stage_route"]["realized_slots"] == {
        "sop": 1,
        "runforest": 2,
    }


def test_agentic_router_invalid_id_falls_back_and_retains_failure(tmp_path):
    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_agentic_max_steps = 1
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: {
        "action": "finish",
        "reason": "malformed attempt",
        "selected_ids": ["invented::candidate", "n1", "s1", "t1"],
    }
    _text, _refs, pack = _retrieve(layer)
    assert pack["candidate_pool_source"] == "live_retrieval_deterministic_fallback"
    assert pack["ranking_contract"] == (
        "agentic_invalid_deterministic_fallback_v1" "+same_task_best_prompt_pin_v1"
    )
    assert pack["retrieval_agent"]["fallback_used"] is True
    assert "unobserved candidate" in pack["retrieval_agent"]["fallback_reason"]


def test_draft_coldstart_adoption_alias_matches_dynamic_router_pack(tmp_path):
    from engine.search_node import SearchNode

    layer = _layer(tmp_path, "dynamic_hybrid")
    _text, refs, pack = _retrieve(layer, stage="draft")
    assert pack["stage_route"]["stage"] == "draft"
    agent = SimpleNamespace(
        external_skill_memory=layer,
        cfg=SimpleNamespace(exp_id="task", run_identity=SimpleNamespace()),
        adoption_tracking_enabled=True,
        evaluation_authority=None,
    )
    node = SearchNode(code="print('draft')", stage="draft")

    log_adoption(
        node,
        agent,
        layer.source_name,
        refs,
        "coldstart",
    )

    assert node.memory_routing_trace["node_stage"] == "draft"
    assert node.memory_routing_trace["node_stage_raw"] == "coldstart"
    assert node.memory_routing_trace["pack_stage"] == "draft"
    assert node.memory_routing_trace["pack_stage_raw"] == "draft"


def test_coldstart_role_abstention_pack_records_canonical_draft_stage(tmp_path):
    from engine.search_node import SearchNode

    layer = _layer(tmp_path, "dynamic_hybrid")
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="task",
        task_desc="text classification",
        query_parts=["cold start"],
        draft_role="coldstart_baseline",
    )
    pack = layer.current_navigation_pack()
    assert text == "" and refs == []
    assert pack["schema"] == "stage_hybrid_role_policy_abstention_v1"
    assert pack["stage_route"]["stage"] == "draft"
    agent = SimpleNamespace(
        external_skill_memory=layer,
        cfg=SimpleNamespace(exp_id="task", run_identity=SimpleNamespace()),
        adoption_tracking_enabled=True,
        evaluation_authority=None,
    )
    node = SearchNode(
        code="print('coldstart')",
        stage="draft",
        draft_role="coldstart_baseline",
    )

    log_adoption(node, agent, layer.source_name, [], "draft")

    assert node.memory_routing_trace["node_stage"] == "draft"
    assert node.memory_routing_trace["node_stage_raw"] == "draft"
    assert node.memory_routing_trace["pack_stage"] == "draft"
    assert node.memory_routing_trace["pack_stage_raw"] == "draft"


def test_replay_branch_reopens_router_for_improve_and_debug_without_pack_leakage(
    tmp_path,
):
    from agents.memory.external_skill_memory import fetch_external_skill_memory
    from engine.search_node import SearchNode

    layer = _layer(tmp_path, "dynamic_hybrid")
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_agentic_max_steps = 1
    calls = []

    def query_fn(**kwargs):
        prompt = kwargs["system_message"]
        stage = str(prompt["stage"])
        known = json.loads(prompt["known_candidates"])
        contract = json.loads(prompt["final_selection_contract"])
        selected = []
        for source in ("sop", "runforest"):
            required = int(contract["minimum_source_counts"][source])
            selected.extend(
                row["id"]
                for row in known
                if row["source"] == source and row["id"] not in selected
            )
            source_ids = [
                node_id
                for node_id in selected
                if next(row for row in known if row["id"] == node_id)["source"]
                == source
            ]
            for node_id in source_ids[required:]:
                selected.remove(node_id)
        for row in known:
            if len(selected) >= int(contract["exact_selection_count"]):
                break
            if row["id"] not in selected:
                selected.append(row["id"])
        calls.append(
            {
                "stage": stage,
                "selected_ids": list(selected),
                "known_candidates": known,
            }
        )
        return {
            "action": "finish",
            "reason": f"final {stage} selection",
            "selected_ids": selected,
        }

    layer._experiment_r_agentic_query_fn = query_fn
    agent = SimpleNamespace(
        external_skill_memory=layer,
        cfg=SimpleNamespace(
            exp_id="task",
            run_identity=SimpleNamespace(),
        ),
        task_desc="text classification",
        adoption_tracking_enabled=True,
        evaluation_authority=None,
    )

    improve_text, improve_refs, source = fetch_external_skill_memory(
        agent,
        "improve",
        parent_plan="improve the replayed classifier",
        execution_output="validation succeeded",
        draft_role="memory_reproduction",
    )
    improve_pack = layer.current_navigation_pack()
    assert improve_text and improve_refs and source == layer.source_name
    assert improve_pack["stage_route"]["stage"] == "improve"
    assert improve_pack["stage_route"]["decision_authority"].startswith(
        "retrieval_agent"
    )
    assert improve_pack["router_activation"]["status"] == (
        "retrieval_agent_selected"
    )
    assert improve_pack["router_activation"]["prompt_nonempty"] is True
    assert improve_pack["retrieval_agent"]["agent_calls"] == 1
    assert improve_pack["retrieval_agent"]["fallback_used"] is False
    assert improve_pack["retrieval_agent"]["shortlist_rrf_applied"] is True
    assert improve_pack["retrieval_agent"]["shortlist_rrf_weights"] == {
        "sop": 0.40,
        "runforest": 0.60,
    }
    assert any(
        row["rrf_priority_score"] > 0
        for row in calls[0]["known_candidates"]
    )
    same_task_best = improve_pack["retrieval_agent"]["same_task_best_first"][
        "best_runforest_id"
    ]
    assert same_task_best in improve_pack["final_prompt_candidate_ids"]

    node = SearchNode(
        code="print('improve')",
        stage="improve",
        draft_role="memory_reproduction",
    )
    log_adoption(node, agent, source, improve_refs, "improve")
    assert node.memory_routing_trace["node_stage"] == "improve"
    assert node.memory_routing_trace["pack_stage"] == "improve"
    assert node.memory_routing_trace["retrieval_agent"]["agent_calls"] == 1

    debug_text, debug_refs, _source = fetch_external_skill_memory(
        agent,
        "debug",
        execution_output="RuntimeError: classifier shape mismatch",
        error_type="RuntimeError",
        draft_role="memory_reproduction",
    )
    debug_pack = layer.current_navigation_pack()
    assert debug_text and debug_refs
    assert debug_pack["stage_route"]["stage"] == "debug"
    assert debug_pack["requested_generation_stage"] == "debug"
    assert debug_pack["retrieval_agent"]["agent_calls"] == 1
    assert [row["stage"] for row in calls] == ["improve", "debug"]

    gated_text, gated_refs = layer.retrieve_for_node(
        stage="draft",
        task_id="task",
        task_desc="text classification",
        query_parts=["exact replay"],
        draft_role="memory_reproduction",
    )
    gated_pack = layer.current_navigation_pack()
    assert gated_text == "" and gated_refs == []
    assert gated_pack["schema"] == "stage_hybrid_role_policy_abstention_v1"
    assert gated_pack["stage_route"]["stage"] == "draft"
    assert gated_pack["stage_route"]["route"] == "role_policy_abstention"
    assert gated_pack["role_policy_abstention"]["draft_only"] is True
    assert "retrieval_agent" not in gated_pack


def test_formal_visible_clause_closes_exact_navigation_support_only():
    sop_id = "sop::formal"
    transition_id = "transition::clean"
    clause_id = "clause::generate"
    layer = SimpleNamespace(
        nodes={
            clause_id: {
                "id": clause_id,
                "type": "SOPClause",
                "sop_id": sop_id,
                "source_transition_refs": [transition_id],
                "contract_spec": {
                    "supporting_transition": {
                        "transition_ref": transition_id,
                        "checks": {
                            "audit_clean": True,
                            "not_buggy": True,
                            "positive_or_initial_outcome": True,
                            "valid": True,
                        },
                    }
                },
            },
            transition_id: {"id": transition_id, "type": "Transition"},
        },
        _clean_sop_support=lambda _sop_id: ([], []),
        _visibility_is_enforced=lambda: True,
        _visibility_projection=lambda _sop_id: {"clause_ids": [clause_id]},
        _effective_visibility_sop_ids=lambda: {sop_id},
        _navigation_transitions_by_sop={sop_id: [transition_id]},
        _sop_edge_metadata={
            (transition_id, sop_id): {
                "kind": "navigation_attached_to",
                "clause_ids": [clause_id],
            }
        },
        _positive_transition=lambda _transition_id: (True, "clean"),
    )
    clean, rejected = _experiment_r_clean_sop_support(layer, sop_id)
    assert clean == [transition_id]
    assert rejected == []

    layer.nodes[clause_id]["source_transition_refs"] = ["transition::other"]
    layer.nodes[clause_id]["contract_spec"]["supporting_transition"][
        "transition_ref"
    ] = "transition::other"
    clean, rejected = _experiment_r_clean_sop_support(layer, sop_id)
    assert clean == []
    assert rejected == [
        {
            "transition_id": transition_id,
            "reason": "visible_clause_transition_binding_mismatch",
        }
    ]
