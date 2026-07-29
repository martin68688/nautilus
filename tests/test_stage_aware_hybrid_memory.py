import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))
GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
HYBRID_CONFIG = REPO / "mlevolve" / "config" / "config_run_forest_stage_hybrid.yaml"
TAXONOMY = REPO / "paper-skills" / "hyper_memory" / "sop_taxonomy.json"
RESEARCH_NOTE = REPO / "coordination" / "stage_aware_sop_gateway_runforest_research_note.md"


def _clean_audit():
    return {
        "schema": "mlevolve_leakage_audit_v2",
        "status": "clean",
        "memory_disposition": "positive_eligible",
        "paper_grade_eligible": True,
        "rank_eligible": True,
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
        {"id": "n0", "type": "RunNode", "run_id": run_id, "run_short_id": run_id, "task": "task", "stage": "draft", "step": 0, "text": "text baseline", "metric": 0.5, "metric_improvement": 0.001, "is_buggy": False, "is_valid": True, "leakage_audit": _clean_audit()},
        {"id": "n1", "type": "RunNode", "run_id": run_id, "run_short_id": run_id, "task": "task", "stage": "improve", "step": 1, "parent_id": "n0", "local_best_node_id": "n1", "text": "transformer validation ensemble", "metric": 0.4, "metric_improvement": 0.1, "is_buggy": False, "is_valid": True, "leakage_audit": child_audit},
        {"id": "n_bad", "type": "RunNode", "run_id": run_id, "run_short_id": run_id, "stage": "debug", "step": 2, "parent_id": "n0", "is_buggy": True, "text": "failure", "leakage_audit": {"status": "blocked", "memory_disposition": "warning_only", "paper_grade_eligible": False}},
        {"id": "t1", "type": "Transition", "run_id": run_id, "run_short_id": run_id, "task": "task", "parent_node_id": "n0", "child_node_id": "n1", "stage_pair": "draft->improve", "outcome": "metric_improved", "metric_improvement": 0.1, "text": "transformer validation ensemble"},
        {"id": "s1", "type": "SOP", "title": "validation ensemble", "action": "use transformer ensemble", "applies_when": ["text classification"], "prevents": ["overfit"], "evidence_turns": ["B0.T1"], "text": "validation ensemble transformer", "decision_stages": ["draft"], "task_families": ["text_classification"]},
        {"id": "s2", "type": "SOP", "title": "unattached method", "action": "try another feature", "applies_when": ["draft"], "prevents": [], "text": "unattached feature", "decision_stages": ["debug"], "task_families": ["tabular_regression"]},
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
    assert cfg.external_skill_memory.scoring_mode == "flat_twin"
    assert cfg.external_skill_memory.retrieval_control == "layered_strategy"
    assert list(cfg.agent.draft_role_policy.roles) == [
        "coldstart_baseline", "memory_reproduction", "novel_exploration"
    ]
    assert cfg.agent.initial_drafts == 3
    assert cfg.agent.search.num_drafts == 3


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
    assert merged.external_skill_memory.scoring_mode == "flat_twin"
    assert merged.external_skill_memory.retrieval_control == "layered_strategy"


def test_real_graph_reverse_index_uses_distills_to():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH), index_path=str(INDEX), mode="run_forest_stage_hybrid", enable_agentic=False
    )
    assert layer._transitions_by_sop
    assert sum(len(values) for values in layer._transitions_by_sop.values()) > 0
    assert all(layer.nodes[tid]["type"] == "Transition" for values in layer._transitions_by_sop.values() for tid in values)


@pytest.mark.parametrize(
    ("task_id", "task_desc"),
    [
        ("denoising-dirty-documents", "restore noisy document images and minimize RMSE"),
        ("new-york-city-taxi-fare-prediction", "tabular taxi fare regression and RMSE"),
        ("mlsp-2013-birds", "multiclass bird audio classification"),
    ],
)
def test_layered_novel_draft_uses_explicit_clean_v2_fallback_when_l1_coverage_is_sparse(task_id, task_desc):
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        retrieval_control="layered_strategy",
        enable_agentic=False,
        top_k=6,
    )
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id=task_id,
        task_desc=task_desc,
        query_parts=["choose a robust clean model"],
        draft_role="novel_exploration",
        context={"excluded_method_families": []},
    )
    pack = layer.current_navigation_pack()
    fallback = pack["layered_strategy_fallback"]
    assert text
    assert refs
    assert fallback["activated"] is True
    assert fallback["fallback_mode"] == "stage_hybrid_v2_clean_cross_task"
    assert pack["algorithm_version"] == "stage_hybrid_v2"
    assert pack["execution_safety_gate"]["all_outputs_clean"] is True


def test_memory_transfer_role_is_explicit_and_clean_for_task_without_exact_replay():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        retrieval_control="layered_strategy",
        enable_agentic=False,
        top_k=6,
    )
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="mlsp-2013-birds",
        task_desc="multiclass bird audio classification",
        query_parts=["transfer only clean historical evidence"],
        draft_role="memory_transfer",
    )
    pack = layer.current_navigation_pack()
    assert text and refs
    assert pack["memory_transfer"]["activated"] is True
    assert pack["memory_transfer"]["mode"] == "stage_hybrid_v2_clean_cross_task"
    assert pack["execution_safety_gate"]["all_outputs_clean"] is True


def test_no_memory_binds_bundle_but_has_zero_prompt_refs_and_exposure(tmp_path):
    from agents.adoption import log_adoption

    layer = _layer(tmp_path, retrieval_control="no_memory")
    assert layer.nodes
    assert layer.graph_path.is_file()

    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="aerial-cactus-identification",
        task_desc="Binary aerial image classification.",
        query_parts=["Use the loaded task-heldout bundle."],
        draft_role="memory_transfer",
    )

    pack = layer.current_navigation_pack()
    assert text == ""
    assert refs == []
    assert layer.current_visibility_pack() is None
    assert pack["schema"] == "stage_hybrid_no_memory_pack_v1"
    assert pack["stage_route"]["control"] == "no_memory"
    assert pack["memory_snapshot_bound_but_not_exposed"] is True
    assert pack["prompt_text"] == ""
    assert pack["prompt_visible_refs"] == []
    assert pack["visible_clause_ids"] == []
    assert pack["fused_execution_candidates"] == []
    assert pack["selected_sop_gateways"] == []
    assert pack["sop_only_candidates"] == []
    assert pack["evidence_refs"] == []
    assert pack["failure_patterns"] == []

    recorded_exposures = []
    authority = SimpleNamespace(
        record_prompt_exposure=lambda **payload: recorded_exposures.append(payload)
    )
    node = SimpleNamespace(adoption_log=[], memory_navigation_trace=[])
    agent = SimpleNamespace(
        adoption_tracking_enabled=True,
        evaluation_authority=authority,
        external_skill_memory=layer,
    )
    log_adoption(node, agent, layer.source_name, refs, "draft")
    assert recorded_exposures == []
    assert node.adoption_log == []
    assert node.memory_navigation_trace == []


def test_rendered_sop_only_prompt_exposures_are_returned_as_refs(tmp_path):
    layer = _layer(tmp_path, clean=False, retrieval_control="stage_hybrid")

    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="task",
        task_desc="text classification",
        draft_role="memory_transfer",
    )

    pack = layer.current_navigation_pack()
    exposed = [row["id"] for row in pack["sop_only_candidates"]]
    rendered = [sop_id for sop_id in exposed if sop_id in text]
    truncated = set(exposed) - set(rendered)
    assert rendered
    assert set(rendered) <= set(refs)
    assert not truncated & set(refs)


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


def test_production_hybrid_v2_scores_both_channels_and_gates_every_execution_candidate(tmp_path):
    layer = _layer(tmp_path)
    pack = layer._hybrid_pack(
        stage="draft",
        task_id="task",
        task_desc="text classification",
        query_text="transformer validation ensemble",
    )
    assert pack["algorithm_version"] == "stage_hybrid_v2"
    assert pack["tree_candidate_details"]
    assert all(item["audit_status"] == "clean" for item in pack["tree_candidate_details"])
    assert all(item["rank_eligible"] is True for item in pack["tree_candidate_details"])
    assert all(item["eligibility_reason"] == "clean_successful_run_node" for item in pack["tree_candidate_details"])
    details = {item["id"]: item for item in pack["tree_candidate_details"]}
    assert details["n1"]["score_components"]["task_local_improvement_percentile"] == 1.0
    assert details["n0"]["score_components"]["task_local_improvement_percentile"] == 0.5
    assert details["n1"]["score_components"]["metric_improvement"] > details["n0"]["score_components"]["metric_improvement"]
    assert pack["execution_safety_gate"]["all_outputs_clean"] is True
    assert all(
        layer._execution_candidate_eligibility(item["id"])[0]
        for item in pack["fused_execution_candidates"]
    )
    assert all(
        item["id"] in pack["execution_candidate_provenance"]
        for item in pack["fused_execution_candidates"]
    )
    sop = next(item for item in pack["direct_sop_candidates"] if item["id"] == "s1")
    assert sop["ranking_backend"] == "stage_task_geometry_field_hybrid_v2"
    assert set(sop["hybrid_score_components"]) == {
        "field_relevance", "stage_fit", "task_fit", "geometry", "clean_evidence"
    }
    wrong_task = next(item for item in pack["direct_sop_candidates"] if item["id"] == "s2")
    assert sop["stage_compatible"] is True and sop["task_compatible"] is True
    assert wrong_task["stage_compatible"] is False and wrong_task["task_compatible"] is False


def test_true_sop_hybrid_projects_tree_evidence_and_preserves_stage_weights(tmp_path):
    layer = _layer(tmp_path)
    pack = layer.rank_sop_hybrid(
        stage="draft",
        task_id="task",
        task_desc="text classification",
        query_text="transformer validation ensemble",
        limit=5,
    )
    assert pack["schema"] == "stage_hybrid_sop_ranking_v2"
    assert pack["stage_route"]["rrf"] == {"sop": 0.70, "tree": 0.30}
    assert "s1" in pack["direct_clean_sop_ids"]
    assert "s1" in pack["tree_projected_sop_ids"]
    assert [item["id"] for item in pack["fused_sop_candidates"]] == ["s1"]
    assert pack["safety_gate"]["all_outputs_clean"] is True


def test_debug_tree_ranks_complete_causal_transitions_and_respects_quota():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        enable_agentic=False,
    )
    query = (
        "Text classification debug: repair a deterministic runtime or resource failure "
        "without redesigning the model. CUDA out of memory or incompatible API calls may be involved."
    )
    rows = layer._rank_debug_transition_rows(
        query_text=query,
        task_id="spooky-author-identification",
        task_desc="text classification",
        limit=8,
    )
    assert rows
    assert len(rows) <= 8
    for row in rows:
        transition = layer.nodes[row["id"]]
        assert transition["type"] == "Transition"
        assert transition["outcome"] == "debug_fixed"
        assert transition["parent_buggy"] is True
        assert transition["child_buggy"] is False
        assert row["score_components"]["failure_signature"] > 0.0
        assert row["score_components"]["task"] >= 0.75
        assert row["causal_attachments"]
        assert row["transition_evidence"]["parent_failure"]
        assert row["transition_evidence"]["code_change"]
        assert row["transition_evidence"]["child_result"]
        assert all(item["quality"] == "evidence_turn_match" or item["quality_score"] >= 0.55 for item in row["causal_attachments"])


def test_debug_prompt_expands_the_complete_causal_transition():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        enable_agentic=False,
    )
    text, _refs = layer.retrieve_for_node(
        stage="debug",
        task_id="spooky-author-identification",
        task_desc="text classification",
        query_parts=["Repair a deterministic runtime resource failure such as CUDA out of memory."],
    )
    assert "Parent failure:" in text
    assert "Proven code change:" in text
    assert "Successful child result:" in text
    assert "Causally supported SOPs only:" in text


def test_debug_tree_confidence_controls_weight_and_can_fall_back_to_sop_only():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        enable_agentic=False,
    )
    strong = layer.rank_sop_hybrid(
        stage="debug",
        task_id="spooky-author-identification",
        task_desc="text classification",
        query_text="Repair a deterministic runtime resource failure such as CUDA out of memory.",
        limit=10,
    )
    strong_route = strong["stage_route"]
    assert strong_route["fallback_reason"] is None
    assert 0.0 < strong_route["rrf"]["tree"] <= 0.60
    assert strong_route["rrf"]["tree"] != strong_route["configured_rrf"]["tree"]

    weak = layer.rank_sop_hybrid(
        stage="debug",
        task_id="unknown-audio-task",
        task_desc="audio event classification",
        query_text="Repair sample alignment while preserving semantics.",
        limit=10,
    )
    weak_route = weak["stage_route"]
    assert weak_route["fallback_reason"] == "insufficient_causal_tree_confidence"
    assert weak_route["rrf"] == {"sop": 1.0, "tree": 0.0}
    assert weak["tree_projected_sop_ids"] == []


def test_debug_failure_similarity_does_not_saturate_on_long_generic_history(tmp_path):
    layer = _layer(tmp_path)
    query = "repair validation fit scope leakage"
    unrelated = " ".join(["model train validation error fix"] * 100)
    assert layer._bounded_token_similarity(query, unrelated) < 0.50
    assert layer._failure_signature(query) == {"fit_scope"}
    assert "resource" in layer._failure_signature("CUDA out of memory during validation")
    assert "exception:valueerror" in layer._failure_signature("ValueError: labels are misaligned")


def test_debug_tree_requires_an_explicit_failure_signature(tmp_path):
    layer = _layer(tmp_path)
    assert layer._rank_debug_transition_rows(
        query_text="Improve this implementation in a generally useful way.",
        task_id="task",
        task_desc="text classification",
        limit=8,
    ) == []


def test_stage_taxonomy_is_an_effective_gate_in_real_sop_projection():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        mode="run_forest_stage_hybrid",
        enable_agentic=False,
    )
    for stage in ("draft", "improve", "debug"):
        pack = layer.rank_sop_hybrid(
            stage=stage,
            task_id="spooky-author-identification",
            task_desc="Small-data text classification evaluated by multiclass log loss.",
            query_text="choose a robust model and avoid validation overfitting",
            limit=10,
        )
        first_ten = pack["direct_clean_sop_ids"][:10]
        assert len(first_ten) == 10
        assert all(
            layer._sop_stage_fit(layer.nodes[sop_id], stage)[1]
            for sop_id in first_ten
        )


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


def test_replacement_draft_retrieves_as_novel_exploration():
    from agents.memory.external_skill_memory import fetch_external_skill_memory

    class RoleCapturingHybrid:
        mode = "run_forest_stage_hybrid"
        source_name = "run_forest_stage_hybrid_memory"

        def __init__(self):
            self.roles = []

        def retrieve_for_node(self, **kwargs):
            self.roles.append(kwargs.get("draft_role"))
            return "replacement memory", ["replacement-ref"]

    layer = RoleCapturingHybrid()
    agent = SimpleNamespace(
        external_skill_memory=layer,
        cfg=SimpleNamespace(exp_id="task"),
        task_desc="task description",
    )

    text, refs, _source = fetch_external_skill_memory(
        agent, "draft", draft_role="replacement_draft"
    )

    assert text == "replacement memory"
    assert refs == ["replacement-ref"]
    assert layer.roles == ["novel_exploration"]


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
        ("full_decision_admissibility", True, True),
        ("flat_relevance_memory", False, False),
        ("global_validity_bit", False, False),
        ("authority_only", False, False),
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
        assert all(
            item["source_channels"] == ["tree_direct"]
            for item in pack["execution_candidate_provenance"].values()
        )
        assert pack["sop_transition_matches"] == []
        assert all(
            item["candidate_class"] == "tree_only_candidates"
            for item in pack["fused_execution_candidates"]
        )
    if control == "sop_only":
        assert pack["tree_candidate_details"] == []
        assert all(
            item["source_channels"] == ["sop_gateway"]
            for item in pack["execution_candidate_provenance"].values()
        )
        assert pack["tree_only_candidates"] == []
        assert set(pack["execution_candidate_provenance"]) == {
            item["id"] for item in pack["fused_execution_candidates"]
        }
    if control in {
        "flat_relevance_memory",
        "global_validity_bit",
        "authority_only",
    }:
        assert pack["algorithm_version"] == "formal_flat_relevance_v1"
        assert pack["direct_sop_candidates"]
        assert pack["sop_only_candidates"] == pack["direct_sop_candidates"]
        assert pack["tree_candidate_details"] == []
        assert pack["fused_execution_candidates"] == []
        assert all(
            set(item["score_components"]) == {"flat_text_relevance"}
            for item in pack["direct_sop_candidates"]
        )


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


def _real_layered():
    from config import _load_cfg
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    cfg = _load_cfg(HYBRID_CONFIG, use_cli_args=False)
    cfg.exp_id = "spooky-author-identification"
    cfg.agent.search.num_gpus = 7
    return StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        retrieval_control="layered_strategy",
        enable_agentic=False,
        top_k=10,
        max_chars=0,
        cfg=cfg,
    )


def test_taxonomy_is_complete_and_known_levels_are_pinned():
    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    graph = json.loads((REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json").read_text(encoding="utf-8"))
    sop_ids = {node["id"] for node in graph["nodes"] if node.get("type") == "SOP"}
    assert taxonomy["schema"] == "runforest_sop_taxonomy_v1"
    assert taxonomy["coverage"] == 1.0
    assert taxonomy["sop_count"] == len(sop_ids) == 281
    assert set(taxonomy["entries"]) == sop_ids
    l1_ids = {
        sop_id
        for sop_id, entry in taxonomy["entries"].items()
        if entry["abstraction_level"] == "L1_strategy"
    }
    assert taxonomy["reviewed_l1_count"] == len(l1_ids) == 28
    assert set(taxonomy["reviewed_l1_ids"]) == l1_ids
    assert all(taxonomy["entries"][sop_id]["manual_reviewed"] is True for sop_id in l1_ids)
    from agents.memory.stage_aware_hybrid_memory import FAMILY_CODE_SIGNATURES

    assert {taxonomy["entries"][sop_id]["method_family"] for sop_id in l1_ids} <= set(FAMILY_CODE_SIGNATURES)
    for sop_id in ("sg_0089", "sg_0221", "sg_0164"):
        assert taxonomy["entries"][sop_id]["abstraction_level"] == "L1_strategy"
    assert taxonomy["entries"]["sg_0227"]["abstraction_level"] == "L2_tactic"
    for sop_id in ("sg_0069", "sg_0115"):
        assert taxonomy["entries"][sop_id]["abstraction_level"] == "L3_repair"


def test_research_note_has_current_layered_flow_without_stale_draft_claims():
    text = RESEARCH_NOTE.read_text(encoding="utf-8")
    for stale in ("Status at baseline: design only", "Additional initial drafts default", "run_forest_stage_hybrid is not implemented"):
        assert stale not in text
    assert "L1 Strategy Retriever" in text
    assert "进入 model_design 时才检索 L2 tactic" in text
    assert "initial_drafts = 3" in text
    assert "claim_allowed=false" in text


def test_layered_novel_draft_returns_three_clean_distinct_strategy_families():
    layer = _real_layered()
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="spooky-author-identification",
        task_desc="Small-data text classification evaluated by multiclass log loss.",
        query_parts=["Seven GPUs are available."],
        draft_role="novel_exploration",
        context={
            "baseline_model": "ModernBERT",
            "coldstart": "answerdotai/ModernBERT-large",
            "data_preview": "Train shape: (17621, 3)",
            "excluded_method_families": ["modernbert_finetune", "deberta_xgb_lr_ensemble"],
        },
    )
    pack = layer.current_navigation_pack()
    routes = pack["strategy_routes"]
    assert pack["schema"] == "layered_strategy_memory_pack_v1"
    assert len(routes) == 3
    assert len({route["method_family"] for route in routes}) == 3
    assert not ({route["method_family"] for route in routes} & set(pack["excluded_method_families"]))
    assert all(layer.nodes[route["sop_id"]]["abstraction_level"] == "L1_strategy" for route in routes)
    assert all(route["best_tree_evidence"]["audit_status"] == "clean" for route in routes)
    assert all(route["best_tree_evidence"]["rank_eligible"] is True for route in routes)
    assert not {"sop::sg_0227", "sop::sg_0069", "sop::sg_0115"} & {route["sop_id"] for route in routes}
    assert "Frozen Novel Strategy Contract" in text
    assert len(refs) == 3


@pytest.mark.parametrize(
    "model_name",
    [
        "DINOv2",
        "DINOv3",
        "DINO-v4",
        "DINO v12",
        "facebook/dinov3-vitl16-pretrain",
    ],
)
def test_dino_versions_and_checkpoint_names_map_to_one_model_family(model_name):
    layer = _real_layered()
    assert layer._model_family_from_text(model_name) == "vision_transformer_finetune"


def test_leaf_runtime_context_with_dinov3_reaches_clean_layered_retrieval():
    layer = _real_layered()
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="leaf-classification",
        task_desc="Leaf image classification evaluated by multiclass log loss.",
        query_parts=["One GPU is available."],
        draft_role="novel_exploration",
        context={
            "baseline_model": "DINOv3",
            "coldstart": "facebook/dinov3-vitl16-pretrain",
            "data_preview": "Train shape: (990, 194)",
        },
    )
    pack = layer.current_navigation_pack()
    assert text and refs
    assert "vision_transformer_finetune" in pack["task_profile"]["excluded_method_families"]
    if pack.get("selected_strategy"):
        assert pack["selected_strategy"]["method_family"] != "vision_transformer_finetune"
        assert pack["selected_strategy"]["best_tree_evidence"]["audit_status"] == "clean"
    else:
        assert pack["layered_strategy_fallback"]["activated"] is True
        assert pack["layered_strategy_fallback"]["fallback_mode"] == "stage_hybrid_v2_clean_cross_task"
        assert pack["execution_safety_gate"]["all_outputs_clean"] is True


def test_nomad_without_a_coldstart_template_uses_explicit_clean_fallback():
    layer = _real_layered()
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="nomad2018-predict-transparent-conductors",
        task_desc="Tabular multi-output regression evaluated by RMSLE.",
        query_parts=["One GPU and eight CPUs are available."],
        draft_role="novel_exploration",
        context={"baseline_model": "", "coldstart": "None model"},
    )
    pack = layer.current_navigation_pack()
    profile = pack["task_profile"]
    assert text and refs
    assert profile["modality"] == "tabular"
    assert profile["task_family"] == "tabular_multioutput_regression"
    assert profile["coldstart_primary_model_available"] is False
    assert profile["coldstart_primary_model_family"] is None
    assert pack["layered_strategy_fallback"]["activated"] is True
    assert pack["layered_strategy_fallback"]["fallback_mode"] == "stage_hybrid_v2_clean_cross_task"


def test_full_prefixed_historical_tasks_are_task_local_for_runtime_retrieval():
    layer = _real_layered()
    assert layer._task_score(
        {"task": "full-leaf-classification"},
        "leaf-classification",
        "leaf image classification",
    ) == pytest.approx(0.35)
    assert layer._task_family_for_query(
        "full-aerial-cactus-identification",
        "binary image classification",
    ) == "image_binary_classification"


def test_layered_l1_l2_are_isolated_from_baseline_and_replay():
    layer = _real_layered()
    for role in ("coldstart_baseline", "memory_reproduction"):
        text, refs = layer.retrieve_for_node(
            stage="draft",
            task_id="spooky-author-identification",
            task_desc="Text classification evaluated by log loss.",
            draft_role=role,
        )
        assert text == ""
        assert refs == []


def test_layered_l2_is_family_compatible_and_excludes_repair_sops():
    layer = _real_layered()
    layer.retrieve_for_node(
        stage="draft",
        task_id="spooky-author-identification",
        task_desc="Text classification evaluated by log loss.",
        draft_role="novel_exploration",
        context={
            "excluded_method_families": ["modernbert_finetune", "deberta_xgb_lr_ensemble"],
        },
    )
    strategy = layer.current_navigation_pack()
    text, refs, l2 = layer.retrieve_model_design_tactics(
        task_id="spooky-author-identification",
        task_desc="Text classification evaluated by log loss.",
        strategy_context=strategy,
    )
    assert len(l2["selected_tactics"]) <= 4
    assert all(
        layer._family_compatible(l2["method_family"], item["method_family"])
        for item in l2["selected_tactics"]
    )
    assert all(layer.nodes[item["sop_id"]]["abstraction_level"] == "L2_tactic" for item in l2["selected_tactics"])
    assert not {"sop::sg_0069", "sop::sg_0115"} & {item["sop_id"] for item in l2["selected_tactics"]}
    assert "L2 Model-Design Tactics" in text
    for tactic in l2["selected_tactics"]:
        evidence = tactic["best_tree_evidence"]
        assert tactic["sop_id"] in refs
        assert evidence["transition_id"] in refs
        assert evidence["node_id"] in refs


def test_stepwise_retrieves_l2_only_when_model_design_starts(monkeypatch):
    from agents.coder import stepwise_coder

    seen = []

    class FakeLayer:
        retrieval_control = "layered_strategy"

        def _format_selected_strategy(self, _context):
            return "L1_ONLY"

        def retrieve_model_design_tactics(self, **_kwargs):
            seen.append("retrieve_l2")
            return "L1_PLUS_L2", ["sop::l2"], {"selected_tactics": [{"sop_id": "sop::l2"}]}

    def fake_generate(self, **kwargs):
        seen.append((self.name, kwargs["prompt_base"].get("External Skill Memory")))
        return f"plan-{self.name}", f"code_{self.name} = True"

    def fake_merge(self, **kwargs):
        seen.append(("merge", kwargs["prompt_base"].get("External Skill Memory")))
        return "merged", "print('done')"

    monkeypatch.setattr(stepwise_coder.StepAgent, "generate", fake_generate)
    monkeypatch.setattr(stepwise_coder.MetaAgent, "merge", fake_merge)
    fake_agent = SimpleNamespace(
        external_skill_memory=FakeLayer(),
        cfg=SimpleNamespace(exp_id="spooky-author-identification"),
    )
    _plan, _code, metadata = stepwise_coder.stepwise_plan_and_code_query(
        fake_agent,
        {
            "Task description": "text classification",
            "Instructions": {},
            "External Skill Memory": "must not be reused",
        },
        "preview",
        {
            "stage": "draft",
            "draft_role": "novel_exploration",
            "strategy_context": {
                "selected_strategy": {"method_family": "deberta_finetune"},
                "task_profile": {"task_family": "text_classification"},
            },
        },
    )
    assert seen == [
        ("data_processing_and_feature_engineering", "L1_ONLY"),
        "retrieve_l2",
        ("model_design", "L1_PLUS_L2"),
        ("training_evaluation", "L1_PLUS_L2"),
        ("merge", "L1_PLUS_L2"),
    ]
    assert metadata["l2_ref_ids"] == ["sop::l2"]


def test_selected_strategy_code_alignment_controls_certified_ranking():
    from agents.leakage_audit import audit_code, rank_eligible
    from agents.memory.stage_aware_hybrid_memory import strategy_alignment_for_code
    from engine.search_node import SearchNode

    strategy = {"method_family": "deberta_xgb_lr_ensemble"}
    code = "DebertaModel(); XGBClassifier(); LogisticRegression()"
    node = SearchNode(
        code=code,
        plan="aligned",
        stage="draft",
        draft_role="novel_exploration",
        selected_strategy=strategy,
        is_buggy=False,
        is_valid=True,
    )
    node.leakage_audit = audit_code(code)
    node.strategy_alignment = strategy_alignment_for_code(strategy, code)
    agent = SimpleNamespace(acfg=SimpleNamespace(check_data_leakage=True))
    assert node.strategy_alignment["status"] == "verified"
    assert rank_eligible(agent, node) is True

    node.code = "DebertaModel()"
    node.leakage_audit = audit_code(node.code)
    node.strategy_alignment = strategy_alignment_for_code(strategy, node.code)
    assert node.strategy_alignment["status"] == "partial"
    assert rank_eligible(agent, node) is False
