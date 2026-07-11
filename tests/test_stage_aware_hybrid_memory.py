import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
HYBRID_CONFIG = REPO / "mlevolve" / "config" / "config_run_forest_stage_hybrid.yaml"


def _clean_audit():
    return {
        "schema": "mlevolve_leakage_audit_v2",
        "status": "clean",
        "memory_disposition": "positive_eligible",
        "paper_grade_eligible": True,
        "issues": [],
    }


def _write_fixture(tmp_path: Path, *, clean=True, blocked=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    child_audit = _clean_audit() if clean else {
        "status": "blocked",
        "memory_disposition": "repair_only",
        "paper_grade_eligible": False,
        "issues": [{"issue_code": "LEAK"}],
    }
    run_id = "blocked_run" if blocked else "clean_run"
    nodes = [
        {"id": "run::1", "type": "Run", "run_id": run_id},
        {"id": "n0", "type": "RunNode", "run_id": run_id, "run_short_id": run_id, "stage": "draft", "step": 0, "text": "text baseline", "leakage_audit": _clean_audit()},
        {"id": "n1", "type": "RunNode", "run_id": run_id, "run_short_id": run_id, "stage": "improve", "step": 1, "parent_id": "n0", "local_best_node_id": "n1", "text": "transformer validation ensemble", "leakage_audit": child_audit},
        {"id": "n_bad", "type": "RunNode", "run_id": run_id, "run_short_id": run_id, "stage": "debug", "step": 2, "parent_id": "n0", "is_buggy": True, "text": "failure", "leakage_audit": {"status": "blocked", "memory_disposition": "warning_only", "paper_grade_eligible": False}},
        {"id": "t1", "type": "Transition", "run_id": run_id, "run_short_id": run_id, "parent_node_id": "n0", "child_node_id": "n1", "stage_pair": "draft->improve", "outcome": "metric_improved", "text": "transformer validation ensemble"},
        {"id": "s1", "type": "SOP", "title": "validation ensemble", "action": "use transformer ensemble", "applies_when": ["text classification"], "prevents": ["overfit"], "evidence_turns": ["B0.T1"], "text": "validation ensemble transformer"},
        {"id": "s2", "type": "SOP", "title": "unattached method", "action": "try another feature", "applies_when": ["draft"], "prevents": [], "text": "unattached feature"},
        {"id": "e1", "type": "Evidence", "transition_id": "t1", "text": "metric improved"},
        {"id": "f1", "type": "FailurePattern", "issue_code": "LEAK", "text": "blocked leakage"},
    ]
    edges = [
        {"src": "n0", "dst": "n1", "kind": "parent_of"},
        {"src": "n0", "dst": "n_bad", "kind": "parent_of"},
        {"src": "n0", "dst": "t1", "kind": "has_transition"},
        {"src": "t1", "dst": "n1", "kind": "transition_to"},
        {"src": "t1", "dst": "s1", "kind": "distills_to"},
        {"src": "t1", "dst": "e1", "kind": "supported_by"},
        {"src": "n_bad", "dst": "f1", "kind": "has_failure_pattern"},
    ]
    graph = {
        "meta": {
            "schema": "hyperbolic_run_forest_memory_v1",
            "leak_verified": True,
            "paper_grade": True,
            "leak_audited": True,
            "positive_admission_enforced": True,
            "blocked_run_prefixes": ["blocked"],
        },
        "nodes": nodes,
        "edges": edges,
    }
    graph_path = tmp_path / "graph.json"
    index_path = tmp_path / "index.npz"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    node_ids = np.asarray([node["id"] for node in nodes], dtype=object)
    poincare = np.zeros((len(nodes), 2), dtype=np.float32)
    for index in range(len(nodes)):
        poincare[index] = [0.01 * index, 0.005 * index]
    np.savez(
        index_path,
        node_ids=node_ids,
        poincare=poincare,
        flat_twin=poincare.copy(),
        euclidean=np.zeros((len(nodes), 16), dtype=np.float32),
    )
    return graph_path, index_path


def _layer(tmp_path, **options):
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    graph, index = _write_fixture(
        tmp_path,
        clean=options.pop("clean", True),
        blocked=options.pop("blocked", False),
    )
    return StageAwareHybridMemoryLayer(
        graph_path=str(graph),
        index_path=str(index),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        enable_agentic=options.pop("enable_agentic", False),
        top_k=6,
        **options,
    )


def test_exact_stage_quotas_and_config_roles():
    from omegaconf import OmegaConf
    from agents.memory.stage_aware_hybrid_memory import STAGE_QUOTAS

    assert [tuple(STAGE_QUOTAS[stage].values()) for stage in ("draft", "improve", "debug", "evolution", "fusion")] == [
        (6, 3, 2), (4, 2, 6), (2, 1, 8), (6, 3, 3), (4, 2, 4)
    ]
    cfg = OmegaConf.load(HYBRID_CONFIG)
    assert cfg.external_skill_memory.mode == "run_forest_stage_hybrid"
    assert list(cfg.agent.draft_role_policy.roles) == [
        "coldstart_baseline", "memory_reproduction", "novel_exploration"
    ]
    assert cfg.agent.draft_role_policy.extra_role == "novel_exploration"


def test_hybrid_config_passes_structured_runtime_schema():
    from omegaconf import OmegaConf
    from config import Config, _load_cfg

    cfg = _load_cfg(HYBRID_CONFIG, use_cli_args=False)
    cfg.exp_name = "smoke"
    cfg.exp_id = "smoke"
    cfg.data_dir = "./data"
    cfg.goal = "smoke"
    cfg.desc_file = None
    merged = OmegaConf.merge(OmegaConf.structured(Config), cfg)
    assert merged.external_skill_memory.mode == "run_forest_stage_hybrid"
    assert merged.external_skill_memory.retrieval_control == "stage_hybrid"


def test_real_graph_reverse_index_uses_distills_to():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH), index_path=str(INDEX), mode="run_forest_stage_hybrid", enable_agentic=False
    )
    assert layer._transitions_by_sop
    assert sum(len(values) for values in layer._transitions_by_sop.values()) > 0
    assert all(layer.nodes[tid]["type"] == "Transition" for values in layer._transitions_by_sop.values() for tid in values)


def test_gateway_requires_code_audited_clean_support(tmp_path):
    clean = _layer(tmp_path)
    candidates = clean._rank_sops("transformer validation ensemble", "draft", 6)
    selected, meta = clean._select_gateways(candidates, stage="draft", query_text="ensemble", limit=3)
    assert [item["id"] for item in selected] == ["s1"]
    assert meta["eligible_count"] == 1

    dirty = _layer(tmp_path / "dirty", clean=False)
    candidates = dirty._rank_sops("transformer validation ensemble", "draft", 6)
    selected, meta = dirty._select_gateways(candidates, stage="draft", query_text="ensemble", limit=3)
    assert selected == []
    assert meta["eligible_count"] == 0


def test_blocked_support_is_warning_only(tmp_path):
    layer = _layer(tmp_path, blocked=True)
    pack = layer._hybrid_pack(stage="draft", task_id="task", task_desc="text", query_text="transformer ensemble")
    assert pack["selected_sop_gateways"] == []
    assert pack["sop_transition_matches"] == []
    assert pack["risk_warnings"]
    assert all(item["disposition"] == "warning_or_repair_evidence_only" for item in pack["risk_warnings"])


def test_llm_gateway_call_is_single_and_ids_are_validated(tmp_path):
    calls = []

    def valid_selector(**kwargs):
        calls.append(kwargs)
        return {"gateway_ids": ["s1"], "reasons": {"s1": "best match"}, "goal": "draft"}

    layer = _layer(tmp_path, enable_agentic=True, gateway_selector=valid_selector)
    pack = layer._hybrid_pack(stage="draft", task_id="task", task_desc="text", query_text="transformer ensemble")
    assert len(calls) == 1
    assert pack["gateway_selection"]["llm_tool_calls"] == 1
    assert pack["gateway_selection"]["mode"] == "llm_validated"

    invalid_calls = []

    def invalid_selector(**kwargs):
        invalid_calls.append(kwargs)
        return {"gateway_ids": ["invented"], "reasons": {}, "goal": "draft"}

    fallback = _layer(tmp_path / "invalid", enable_agentic=True, gateway_selector=invalid_selector)
    pack = fallback._hybrid_pack(stage="draft", task_id="task", task_desc="text", query_text="transformer ensemble")
    assert len(invalid_calls) == 1
    assert pack["gateway_selection"]["mode"] == "deterministic_fallback"
    assert [item["id"] for item in pack["selected_sop_gateways"]] == ["s1"]


def test_common_execution_id_rrf_is_deterministic():
    from agents.memory.stage_aware_hybrid_memory import weighted_rrf

    rows = weighted_rrf(["n1", "t1"], ["n1", "n0"], sop_weight=0.5, tree_weight=0.5)
    assert rows[0]["id"] == "n1"
    assert rows[0]["candidate_class"] == "sop_transition_matches"
    assert next(row for row in rows if row["id"] == "n0")["candidate_class"] == "tree_only_candidates"
    assert all(not row["id"].startswith("s") for row in rows)


def test_pack_schema_classes_trace_and_stage_quota(tmp_path):
    layer = _layer(tmp_path)
    pack = layer._hybrid_pack(stage="draft", task_id="task", task_desc="text", query_text="transformer ensemble")
    required = {
        "stage_route", "direct_sop_candidates", "selected_sop_gateways", "gateway_transitions",
        "tree_candidates", "sop_transition_matches", "sop_only_candidates", "tree_only_candidates",
        "evidence_refs", "failure_patterns", "risk_warnings", "navigation_trace",
    }
    assert pack["schema"] == "stage_hybrid_memory_pack_v1"
    assert required <= pack.keys()
    assert len(pack["direct_sop_candidates"]) <= 6
    assert len(pack["selected_sop_gateways"]) <= 3
    assert len(pack["tree_candidates"]) <= 2
    assert {item["id"] for item in pack["sop_only_candidates"]} == {"s2"}
    for item in pack["navigation_trace"]:
        assert {"retrieval_channel", "candidate_class", "gateway_sop_id", "supporting_transition_ids", "selection_reason", "selection_state"} <= item.keys()
        assert item["selection_state"] in {"candidate", "selected", "expanded", "injected"}


def test_invalid_config_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="quota"):
        _layer(tmp_path, stage_quotas={"draft": {"sop_candidates": 0}})
    with pytest.raises(ValueError, match="sum to 1"):
        _layer(tmp_path / "weights", rrf_weights={"draft": {"sop": 0.9, "tree": 0.9}})


def test_empty_config_does_not_clear_graph_blocked_prefixes(tmp_path):
    cfg = SimpleNamespace(external_skill_memory=SimpleNamespace(blocked_run_prefixes=[]))
    layer = _layer(tmp_path, blocked=True, cfg=cfg)
    assert layer._blocked_run_prefixes == ("blocked",)
    pack = layer._hybrid_pack(stage="draft", task_id="task", task_desc="text", query_text="ensemble")
    assert pack["selected_sop_gateways"] == []


def test_existing_run_forest_mode_remains_separate():
    from agents.memory.external_skill_memory import RunForestMemoryLayer
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    assert not issubclass(RunForestMemoryLayer, StageAwareHybridMemoryLayer)
    source = (REPO / "mlevolve" / "engine" / "agent_search.py").read_text(encoding="utf-8")
    assert 'str(ext_mode).lower() == "run_forest_stage_hybrid"' in source
    assert "Failed to initialize required stage-hybrid memory" in source


def test_hybrid_retrieval_errors_fail_closed():
    from agents.memory.external_skill_memory import fetch_external_skill_memory

    class BrokenHybrid:
        mode = "run_forest_stage_hybrid"
        source_name = "run_forest_stage_hybrid_memory"

        def retrieve_for_node(self, **_kwargs):
            raise ValueError("deterministic defect")

    agent = SimpleNamespace(
        external_skill_memory=BrokenHybrid(),
        cfg=SimpleNamespace(exp_id="task"),
        task_desc="task description",
    )
    with pytest.raises(RuntimeError, match="Stage-hybrid memory retrieval failed"):
        fetch_external_skill_memory(agent, "draft")


def test_baseline_and_reproduction_roles_bypass_hybrid():
    from agents.memory.external_skill_memory import fetch_external_skill_memory

    class CountingHybrid:
        mode = "run_forest_stage_hybrid"
        source_name = "run_forest_stage_hybrid_memory"

        def __init__(self):
            self.calls = 0

        def retrieve_for_node(self, **_kwargs):
            self.calls += 1
            return "memory", ["ref"]

    layer = CountingHybrid()
    agent = SimpleNamespace(
        external_skill_memory=layer,
        cfg=SimpleNamespace(exp_id="task"),
        task_desc="task description",
    )
    for role in ("coldstart_baseline", "memory_reproduction"):
        text, refs, _source = fetch_external_skill_memory(agent, "improve", draft_role=role)
        assert text == "" and refs == []
    assert layer.calls == 0
    text, refs, _source = fetch_external_skill_memory(agent, "improve", draft_role="novel_exploration")
    assert text == "memory" and refs == ["ref"]
    assert layer.calls == 1


def test_hybrid_trace_is_thread_local_and_logged_on_node(tmp_path):
    from agents.adoption import log_adoption

    layer = _layer(tmp_path)

    def retrieve(stage):
        _text, refs = layer.retrieve_for_node(
            stage=stage,
            task_id="task",
            task_desc="text classification",
            query_parts=["transformer validation ensemble"],
        )
        return stage, refs, layer.current_navigation_pack()["stage_route"]["stage"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(retrieve, ["draft", "improve"]))
    assert {(stage, observed) for stage, _refs, observed in rows} == {("draft", "draft"), ("improve", "improve")}

    _text, refs = layer.retrieve_for_node(
        stage="draft", task_id="task", task_desc="text", query_parts=["transformer ensemble"]
    )
    node = SimpleNamespace(adoption_log=[], memory_navigation_trace=[])
    agent = SimpleNamespace(adoption_tracking_enabled=True, external_skill_memory=layer)
    log_adoption(node, agent, layer.source_name, refs, "draft")
    assert node.memory_navigation_trace
    assert node.adoption_log
    assert all(record["adoption_outcome"] == "pending_analysis" for record in node.adoption_log)
    required = {"retrieval_channel", "candidate_class", "gateway_sop_id", "supporting_transition_ids", "selection_reason", "selection_state"}
    assert all(required <= record.keys() for record in node.adoption_log)
    assert all(record["retrieval_channel"] for record in node.adoption_log)


def test_prompt_separates_evidence_sop_only_and_risk(tmp_path):
    clean = _layer(tmp_path)
    text, _refs = clean.retrieve_for_node(
        stage="draft", task_id="task", task_desc="text", query_parts=["transformer ensemble"]
    )
    assert "Selected SOP Gateways (clean supporting execution required)" in text
    assert "SOP-Only Method References (unverified here)" in text
    assert "Verified Evidence Refs" in text

    blocked = _layer(tmp_path / "blocked_prompt", blocked=True)
    text, _refs = blocked.retrieve_for_node(
        stage="draft", task_id="task", task_desc="text", query_parts=["transformer ensemble"]
    )
    assert "Risk Warnings (do not adopt as positive recipes)" in text


@pytest.mark.parametrize(
    ("control", "has_sop", "has_tree"),
    [
        ("stage_hybrid", True, True),
        ("sop_only", True, False),
        ("tree_only", False, True),
        ("naive_concat", True, True),
    ],
)
def test_runtime_retrieval_controls_are_isolated(tmp_path, control, has_sop, has_tree):
    layer = _layer(tmp_path, retrieval_control=control)
    pack = layer._hybrid_pack(stage="draft", task_id="task", task_desc="text", query_text="transformer ensemble")
    assert bool(pack["selected_sop_gateways"]) is has_sop
    assert bool(pack["tree_candidates"]) is has_tree
    assert pack["stage_route"]["control"] == control
    if control == "tree_only":
        assert pack["direct_sop_candidates"] == []
        assert pack["sop_only_candidates"] == []


def test_final_adoption_outcome_taxonomy():
    from analysis.adoption_tracker import classify_adoption_outcome

    assert classify_adoption_outcome({"adoption_mode": "exact_code_replay"}, adopted=False, inspected=False) == "fully_adopted"
    assert classify_adoption_outcome({"adoption_mode": "mandatory_audit_repair"}, adopted=True, inspected=True) == "adopted_with_constraints"
    assert classify_adoption_outcome({"adoption_mode": "prompt_injection"}, adopted=True, inspected=True) == "fully_adopted"
    assert classify_adoption_outcome({"adoption_mode": "prompt_injection"}, adopted=False, inspected=True) == "rejected_after_inspection"
    assert classify_adoption_outcome({"adoption_mode": "prompt_injection"}, adopted=False, inspected=False) == "not_adopted"
    assert classify_adoption_outcome({"adoption_outcome": "partially_adopted"}, adopted=False, inspected=True) == "partially_adopted"


def test_hybrid_prompt_labels_and_diff_wording_are_evidence_aware():
    from agents.memory.external_skill_memory import external_memory_section_intro, external_memory_section_title

    source = "run_forest_stage_hybrid_memory"
    assert external_memory_section_title(source) == "Stage-Aware SOP Gateway and Run-Forest Memory"
    intro = external_memory_section_intro(source, "improvement")
    assert "distinct classes" in intro
    for relative in (
        "mlevolve/agents/improve_agent.py",
        "mlevolve/agents/evolution_agent.py",
        "mlevolve/agents/fusion_agent.py",
    ):
        text = (REPO / relative).read_text(encoding="utf-8")
        assert "do not treat SOP-only references as proven recipes" in text
        assert "persistent SOP memories as constraints" not in text


def test_aggregation_is_an_explicit_novel_exploration_branch():
    source = (REPO / "mlevolve" / "agents" / "aggregation_agent.py").read_text(encoding="utf-8")
    assert 'draft_role="novel_exploration"' in source
    assert 'draft_role="novel_exploration",' in source
