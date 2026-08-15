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
RECIPE_BUNDLE = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "recipe_distillation_v3"
    / "recipe_sops.json"
)
RECIPE_EVIDENCE = RECIPE_BUNDLE.parent / "evidence_manifest.json"
RECIPE_IMPLEMENTATIONS = RECIPE_BUNDLE.parent / "implementation_capsules.json"
DYNAMIC_CONFIG = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "systems"
    / "dynamic_hybrid.yaml"
)
RECIPE_FILE_SHA256 = "e6db95649c20a642738d6ee35df1aa11ff15287e3613221becb393e28d2a9398"
RECIPE_BUNDLE_SHA256 = "8cce9dd7ee70897e23e5f1dfda08d056cf6ae77ad63e758bd2bbd05376e88749"
RECIPE_EVIDENCE_FILE_SHA256 = "fcb084206cdaa31cfd052c1bce290871b8c075a6376ee698b5c4119636adda04"
RECIPE_EVIDENCE_MANIFEST_SHA256 = "25f6729ece9b1ead76b0d8501aa6aa4026cb163e3eaa7eb05c755b1d72f6160f"


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


def test_v93_l3_prompt_budgets_pass_runtime_validation(tmp_path):
    ext = SimpleNamespace(
        experiment_r_l3_grep_min_candidates=8,
        experiment_r_l3_grep_max_candidates=28,
        experiment_r_l3_failure_context_chars=12000,
        experiment_r_l3_grep_trace_history=6,
    )
    layer = _layer(
        tmp_path,
        cfg=SimpleNamespace(external_skill_memory=ext),
    )
    assert layer.experiment_r_l3_grep_max_candidates == 28
    assert layer.experiment_r_l3_failure_context_chars == 12000
    assert layer.experiment_r_l3_grep_trace_history == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("experiment_r_l3_grep_max_candidates", 33),
        ("experiment_r_l3_failure_context_chars", 16001),
        ("experiment_r_l3_grep_trace_history", 9),
    ],
)
def test_l3_prompt_budget_safety_caps_fail_closed(tmp_path, field, value):
    ext = SimpleNamespace(
        experiment_r_l3_grep_min_candidates=8,
        experiment_r_l3_grep_max_candidates=28,
        experiment_r_l3_failure_context_chars=12000,
        experiment_r_l3_grep_trace_history=6,
    )
    setattr(ext, field, value)
    with pytest.raises(ValueError, match=field):
        _layer(
            tmp_path / field,
            cfg=SimpleNamespace(external_skill_memory=ext),
        )


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


def test_draft_roles_bypass_only_draft_and_reopen_hybrid_afterward():
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
        text, refs, _source = fetch_external_skill_memory(
            agent, "draft", draft_role=role
        )
        assert text == "" and refs == []
    assert layer.calls == 0

    for role in ("coldstart_baseline", "memory_reproduction"):
        text, refs, _source = fetch_external_skill_memory(
            agent, "improve", draft_role=role
        )
        assert text == "memory" and refs == ["ref"]
    assert layer.calls == 2
    text, refs, _source = fetch_external_skill_memory(agent, "improve", draft_role="novel_exploration")
    assert text == "memory" and refs == ["ref"]
    assert layer.calls == 3


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
    assert node.memory_routing_trace["schema"] == (
        "mlevolve_memory_routing_trace_v1"
    )
    assert node.memory_routing_trace["system_id"] == "dynamic_hybrid"
    assert node.memory_routing_trace["raw_pool_observed"] is True
    assert node.memory_routing_trace["final_prompt_candidate_ids"] == list(
        dict.fromkeys(refs)
    )
    assert node.memory_routing_trace["observational_only"] is True
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
    layer = StageAwareHybridMemoryLayer(
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
    # Most legacy-focused tests below exercise the frozen deterministic
    # comparator directly. Agent-path tests opt in explicitly with an injected
    # selector so the test suite never reaches an external model.
    layer.experiment_r_l3_agent_match_enabled = False
    return layer


def _real_recipe_layer():
    from config import _load_cfg
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    cfg = _load_cfg(DYNAMIC_CONFIG, use_cli_args=False)
    cfg.exp_id = "leaf-classification"
    cfg.agent.search.num_gpus = 1
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        enable_agentic=False,
        top_k=6,
        max_chars=0,
        cfg=cfg,
    )
    # Agent-path tests opt in explicitly with an injected selector so the test
    # suite never reaches an external model.
    layer.experiment_r_l3_agent_match_enabled = False
    return layer


def test_enforced_l3_preflight_rejects_empty_formal_debug_projection():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    layer = object.__new__(StageAwareHybridMemoryLayer)
    layer.experiment_r_l3_agent_match_enabled = True
    layer.memory_snapshot = SimpleNamespace(
        base_bundle_id="old-base-without-formal-clauses",
        base_clauses=lambda *_args, **_kwargs: [],
    )
    layer.visibility_gateway = SimpleNamespace(should_enforce=lambda _request: True)
    layer._visibility_request = lambda **_kwargs: SimpleNamespace(
        operation="debug_hypothesis",
        task_context=SimpleNamespace(
            task_id="leaf-classification",
            task_family="image_classification",
        ),
        generation_stage=SimpleNamespace(value="debug"),
        governance_stage=SimpleNamespace(value="retrieval"),
    )
    layer.visibility_task_id = "leaf-classification"
    layer.nodes = {}

    with pytest.raises(ValueError, match="non-empty formal Base Clause projection"):
        layer._validate_enforced_l3_formal_projection()
    assert layer.enforced_l3_formal_visibility_receipt["status"] == (
        "failed_empty_formal_l3_projection"
    )


def test_enforced_l3_preflight_accepts_296_formal_debug_clauses():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    clauses = [
        {"clause_id": f"clause::{index:03d}", "sop_id": f"repair::{index:03d}"}
        for index in range(296)
    ]
    layer = object.__new__(StageAwareHybridMemoryLayer)
    layer.experiment_r_l3_agent_match_enabled = True
    layer.memory_snapshot = SimpleNamespace(
        base_bundle_id="end2end-leaf-atomic-recipe-runforest-v8",
        base_clauses=lambda *_args, **_kwargs: clauses,
    )
    layer.visibility_gateway = SimpleNamespace(should_enforce=lambda _request: True)
    layer._visibility_request = lambda **_kwargs: SimpleNamespace(
        operation="debug_hypothesis",
        task_context=SimpleNamespace(
            task_id="leaf-classification",
            task_family="image_classification",
        ),
        generation_stage=SimpleNamespace(value="debug"),
        governance_stage=SimpleNamespace(value="retrieval"),
    )
    layer.visibility_task_id = "leaf-classification"
    layer.nodes = {
        row["sop_id"]: {"abstraction_level": "L3_repair"} for row in clauses
    }

    layer._validate_enforced_l3_formal_projection()

    assert layer.enforced_l3_formal_visibility_receipt == {
        "schema": "enforced_l3_formal_visibility_preflight_v1",
        "status": "validated",
        "formal_debug_clause_count": 296,
        "authorized_l3_sop_count": 296,
        "base_bundle_id": "end2end-leaf-atomic-recipe-runforest-v8",
    }


def _inject_frozen_recipe_evidence(layer, *task_ids):
    evidence = json.loads(RECIPE_EVIDENCE.read_text(encoding="utf-8"))
    for task_id in task_ids:
        for row in evidence["selected_evidence"][task_id]:
            node_id = row["node_id"]
            if node_id in layer.nodes:
                continue
            node = {
                "id": node_id,
                "type": "RunNode",
                "task": task_id,
                "run_id": row["run_id"],
                "run_short_id": row["run_id"],
                "stage": row["stage"],
                "metric": row["metric"],
                "metric_direction": row.get("metric_direction"),
                "metric_provenance": row.get("metric_provenance"),
                "metric_improvement": row.get("metric_improvement"),
                "is_buggy": False,
                "is_valid": True,
                "plan": row.get("plan"),
                "code_summary": row.get("code_summary"),
                "code_sha256": row.get("code_sha256"),
                "source_cohort": row.get("source_cohort"),
                "leakage_audit": {
                    "status": row["audit_status"],
                    "memory_disposition": row["memory_disposition"],
                    "paper_grade_eligible": row["paper_grade_eligible"],
                    "rank_eligible": row["rank_eligible"],
                },
            }
            layer.nodes[node_id] = node
            layer._node_tokens[node_id] = layer._node_tokens.get(node_id, set())


def test_recipe_overlay_loads_frozen_three_layer_nodes_and_dynamic_uses_layered_router():
    import hashlib
    from config import _load_cfg

    cfg = _load_cfg(DYNAMIC_CONFIG, use_cli_args=False)
    memory = cfg.external_skill_memory
    assert hashlib.sha256(RECIPE_BUNDLE.read_bytes()).hexdigest() == RECIPE_FILE_SHA256
    assert memory.retrieval_control == "layered_strategy"
    assert memory.enable_agentic is True
    assert memory.experiment_r_enabled is False
    assert memory.experiment_r_l3_agent_match_enabled is False
    assert memory.experiment_r_l3_agent_match_max_attempts == 2
    assert memory.experiment_r_l3_agent_match_min_confidence == pytest.approx(0.50)
    assert memory.experiment_r_l3_agent_match_max_tokens == 1800
    assert memory.recipe_sop_file_sha256 == RECIPE_FILE_SHA256
    assert memory.recipe_sop_bundle_sha256 == RECIPE_BUNDLE_SHA256
    assert memory.recipe_evidence_file_sha256 == RECIPE_EVIDENCE_FILE_SHA256
    assert memory.recipe_evidence_manifest_sha256 == RECIPE_EVIDENCE_MANIFEST_SHA256
    layer = _real_recipe_layer()
    assert layer.recipe_sop_receipt == {
        "schema": "layered_recipe_sop_overlay_receipt_v1",
        "path": str(RECIPE_BUNDLE),
        "file_sha256": RECIPE_FILE_SHA256,
        "bundle_sha256": RECIPE_BUNDLE_SHA256,
        "bundle_version": "recipe-sop-v3-20260806",
        "node_count": 89,
        "l1_count": 28,
        "l2_count": 26,
        "l3_count": 35,
    }
    assert len(layer._recipe_sop_ids) == 89
    assert len(layer._legacy_sop_ids) == 281
    assert layer.recipe_evidence_receipt["schema"] == "layered_recipe_evidence_overlay_receipt_v1"
    assert layer.recipe_evidence_receipt["file_sha256"] == RECIPE_EVIDENCE_FILE_SHA256
    assert layer.recipe_evidence_receipt["manifest_sha256"] == RECIPE_EVIDENCE_MANIFEST_SHA256
    assert layer.recipe_evidence_receipt["selected_node_count"] == 152
    assert layer.recipe_evidence_receipt["selected_repair_transition_count"] == 81
    assert layer.recipe_evidence_receipt["materialized_node_count"] > 0
    assert layer.recipe_evidence_receipt["terminal_node_count"] >= 4
    terminal_id = (
        "postsmoke::e2e-smoke-leaf-controls-v14__leaf-classification__"
        "flat_retrieval__seed-1::abb66aed49824ec490ab833abbb5e05b"
    )
    assert layer.nodes[terminal_id]["metric"] == pytest.approx(0.09353439660745823)
    assert layer.nodes[terminal_id]["metric_provenance"] == "sealed_fixed_holdout_terminal_score"
    assert layer.recipe_implementation_receipt["node_count"] == 83
    assert layer.recipe_implementation_receipt["transition_count"] == 24
    assert layer.recipe_implementation_receipt["required_node_count"] == 290
    assert layer.recipe_implementation_receipt["missing_node_ids"]
    assert layer.recipe_implementation_receipt["complete_recipe_coverage"] is False
    best_id = (
        "postsmoke::e2e-smoke-leaf-layered-recipe-v4__leaf-classification__"
        "dynamic_hybrid__seed-1::d2fccc688085447c9ad84356deac9194"
    )
    assert len(layer.nodes[best_id]["implementation_capsule"]["code"]) == 21762


def test_recipe_overlays_load_for_full_experiment_r_dynamic_router():
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer
    from config import _load_cfg

    cfg = _load_cfg(DYNAMIC_CONFIG, use_cli_args=False)
    cfg.exp_id = "leaf-classification"
    cfg.agent.search.num_gpus = 1
    cfg.external_skill_memory.retrieval_control = "dynamic_hybrid"
    cfg.external_skill_memory.experiment_r_enabled = True
    cfg.external_skill_memory.experiment_r_memory_pool_sha256 = "a" * 64
    layer = StageAwareHybridMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        source_name="run_forest_stage_hybrid_memory",
        mode="run_forest_stage_hybrid",
        scoring_mode="flat_twin",
        enable_agentic=False,
        top_k=6,
        max_chars=0,
        cfg=cfg,
    )

    assert layer.retrieval_control == "dynamic_hybrid"
    assert layer.experiment_r_enabled is True
    assert layer.recipe_sop_receipt["node_count"] == 89
    assert layer.recipe_evidence_receipt["selected_node_count"] == 152
    assert layer.recipe_implementation_receipt["node_count"] == 83
    best_id = (
        "postsmoke::e2e-smoke-leaf-layered-recipe-v4__leaf-classification__"
        "dynamic_hybrid__seed-1::d2fccc688085447c9ad84356deac9194"
    )
    assert len(layer.nodes[best_id]["implementation_capsule"]["code"]) == 21762


def test_l3_debug_retrieval_prefers_exact_task_then_same_task_type_and_blocks_cross_type():
    from agents.memory.stage_aware_hybrid_memory import (
        L3_DYNAMIC_CONFIDENCE_WEIGHTS,
    )

    layer = _real_recipe_layer()
    query = (
        "TypeError during model initialization: "
        "ModernBertForSequenceClassification received unexpected keyword argument "
        "hidden_dropout_prob. Configure dropout through the checkpoint config."
    )
    exact = layer._rank_debug_transition_rows(
        query_text=query,
        task_id="spooky-author-identification",
        task_desc="NLP multiclass author classification",
        limit=8,
    )
    assert exact
    assert all(row["task_scope"] == "exact_task" for row in exact)
    assert all(row["score_components"]["task_match"] == 1.0 for row in exact)
    assert any(
        attachment["sop_id"] == "repair::spooky-author-identification::001"
        for row in exact
        for attachment in row["causal_attachments"]
    )
    assert exact[0]["dynamic_confidence_weights"] == L3_DYNAMIC_CONFIDENCE_WEIGHTS
    assert L3_DYNAMIC_CONFIDENCE_WEIGHTS["successful_repair_frequency"] == 0.02

    same_type = layer._rank_debug_transition_rows(
        query_text=query,
        task_id="random-acts-of-pizza",
        task_desc="NLP binary text classification",
        limit=8,
    )
    assert same_type
    assert all(row["task_scope"] == "same_task_type" for row in same_type)
    assert all(row["score_components"]["task_match"] == 0.70 for row in same_type)

    cross_type = layer._rank_debug_transition_rows(
        query_text=query,
        task_id="aerial-cactus-identification",
        task_desc="Vision binary image classification",
        limit=8,
    )
    assert cross_type == []


def test_l3_failure_matching_handles_semantic_paraphrase_without_generic_false_positive():
    layer = _real_recipe_layer()
    query = (
        "The augmented-label objective is a vector over the batch, so autograd "
        "refuses to backpropagate. Aggregate the per-example criterion before "
        "calling backward."
    )
    rows = layer._rank_debug_transition_rows(
        query_text=query,
        task_id="leaf-classification",
        task_desc="Leaf multimodal classification evaluated by log loss.",
        limit=8,
    )
    assert rows
    assert {
        attachment["sop_id"]
        for row in rows
        for attachment in row["causal_attachments"]
        if attachment["sop_id"].startswith("repair::")
    } == {"repair::leaf-classification::002"}
    assert all(
        row["score_components"]["failure_signature_match"] >= 0.50
        for row in rows
    )


def test_l3_classifier_dimension_failure_does_not_match_batch_schema_repair():
    layer = _real_recipe_layer()
    query = (
        "RuntimeError in the first batch: the backbone outputs a spatial feature "
        "map [32, 768, 8, 8], but the classification head expects a one-dimensional "
        "feature vector [*, 768], causing a layer_norm shape mismatch."
    )
    assert layer._specific_failure_signature_overlap(
        "feature_extraction/labeled_unlabeled_batch_schema_mismatch",
        query,
    ) < 0.50
    assert layer._specific_failure_signature_overlap(
        "model_forward/convolutional_feature_classifier_dimension_mismatch",
        query,
    ) == 1.0


def test_dynamic_manual_l3_router_hard_gates_before_gateway_agent():
    from agents.adoption import log_adoption
    from agents.memory.stage_aware_hybrid_memory import (
        L3_FAILURE_TOKEN_EQUIVALENCE_GROUPS,
    )

    layer = _real_recipe_layer()
    assert layer.experiment_r_l3_agent_match_enabled is False
    layer.agentic_enabled = True
    assert any(
        {"classifier", "classification", "head"} <= group
        for group in L3_FAILURE_TOKEN_EQUIVALENCE_GROUPS
    )
    observed = []

    def selector(**kwargs):
        eligible = kwargs["eligible"]
        observed.extend(row["id"] for row in eligible)
        assert eligible
        assert all(
            layer.nodes[row["id"]].get("task_id")
            == "aerial-cactus-identification"
            for row in eligible
        )
        selected = next(
            row
            for row in eligible
            if row["id"] == "repair::aerial-cactus-identification::005"
        )
        return {
            "gateway_ids": [selected["id"]],
            "reasons": {selected["id"]: "same classifier dimension failure"},
            "goal": "inject the exact-task clean repair",
        }

    layer._injected_gateway_selector = selector
    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="aerial-cactus-identification",
        task_desc="Vision binary image classification",
        query_parts=[
            "RuntimeError: the convolutional backbone returns a spatial feature "
            "map [32, 768, 8, 8], while the classification head expects a "
            "flattened vector [*, 768]."
        ],
    )
    pack = layer.current_navigation_pack()
    assert observed
    assert pack["l3_agent_match"] == {}
    assert pack["gateway_selection"]["mode"] == "llm_validated"
    assert pack["selected_sop_gateways"][0]["id"] == (
        "repair::aerial-cactus-identification::005"
    )
    assert pack["stage_route"]["quotas"]["sop_gateways"] == 1
    assert pack["stage_route"]["quotas"]["tree_candidates"] == 5
    assert pack["prompt_execution_limit"] == 5
    assert len(pack["fused_execution_candidates"]) <= 5
    assert set(pack["final_prompt_candidate_ids"]) == {
        row["id"] for row in pack["fused_execution_candidates"]
    }
    assert "repair::aerial-cactus-identification::005" in refs
    assert "repair::aerial-cactus-identification::005" in text

    node = SimpleNamespace(adoption_log=[], memory_navigation_trace=[])
    agent = SimpleNamespace(
        adoption_tracking_enabled=True,
        evaluation_authority=None,
        external_skill_memory=layer,
    )
    log_adoption(node, agent, layer.source_name, refs, "debug")
    route = node.memory_routing_trace
    assert route["memory_pack_schema"] == "stage_hybrid_memory_pack_v1"
    assert route["stage_route"]["quotas"]["sop_gateways"] == 1
    assert route["stage_route"]["quotas"]["tree_candidates"] == 5
    assert route["final_prompt_candidate_ids"] == list(dict.fromkeys(refs))
    assert route["raw_candidates"]
    assert route["navigation_trace"]


def test_dynamic_l3_agent_sees_all_exact_task_cards_and_selects_by_root_cause():
    from agents.memory.experiment_r_router import _agentic_l3_debug_match

    layer = _real_recipe_layer()
    calls = []

    def query_fn(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["system_message"]
        candidates = json.loads(prompt["authorized_l3_candidates"])
        assert prompt["manual_synonym_table_used"] == "false"
        # Literal extraction keeps the observed wording; it does not silently
        # expand ``classification`` into the maintained synonym ``classifier``.
        assert prompt["literal_failure_anchors"].find("classification") >= 0
        assert {row["task_scope"] for row in candidates} == {"exact_task"}
        assert {row["source_task_id"] for row in candidates} == {
            "aerial-cactus-identification"
        }
        assessments = []
        for row in candidates:
            selected = row["sop_id"] == "repair::aerial-cactus-identification::005"
            assessments.append(
                {
                    "sop_id": row["sop_id"],
                    "keyword_correspondence": 0.96 if selected else 0.15,
                    "root_cause_equivalence": 0.99 if selected else 0.10,
                    "runtime_stage_match": 1.0 if selected else 0.40,
                    "contradiction": not selected,
                    "confidence": 0.97 if selected else 0.12,
                    "reason": "classifier input width" if selected else "different root cause",
                }
            )
        selected = next(
            row
            for row in candidates
            if row["sop_id"] == "repair::aerial-cactus-identification::005"
        )
        return {
            "decision": "select",
            "selected_sop_id": selected["sop_id"],
            "selected_transition_id": selected["transition_id"],
            "final_confidence": 0.97,
            "reason": "same convolutional-to-classifier dimension contract",
            "assessments": assessments,
        }

    layer._experiment_r_agentic_query_fn = query_fn
    layer.experiment_r_l3_agent_match_enabled = True
    result = _agentic_l3_debug_match(
        layer,
        task_id="aerial-cactus-identification",
        task_desc="Vision binary image classification",
        query_text=(
            "RuntimeError: a convolutional backbone returns [32, 768, 8, 8] "
            "but the classification head passes it directly into LayerNorm "
            "which expects [*, 768]."
        ),
        visible_sop_ids=None,
    )
    assert len(calls) == 1
    assert result["decision"] == "select"
    assert result["selected_sop_id"] == "repair::aerial-cactus-identification::005"
    assert result["selected_task_scope"] == "exact_task"
    assert result["manual_synonym_table_used"] is False
    assert result["literal_anchor_extractor"]["extractor"] == (
        "literal_regex_no_synonym_expansion_v1"
    )


def test_dynamic_l3_agent_abstention_is_only_condition_for_same_type_fallback():
    from agents.memory.experiment_r_router import _agentic_l3_debug_match

    layer = _real_recipe_layer()
    scopes = []

    def query_fn(**kwargs):
        prompt = kwargs["system_message"]
        scope = prompt["task_scope_already_enforced_by_host"]
        scopes.append(scope)
        candidates = json.loads(prompt["authorized_l3_candidates"])
        assessments = [
            {
                "sop_id": row["sop_id"],
                "keyword_correspondence": 0.10,
                "root_cause_equivalence": 0.10,
                "runtime_stage_match": 0.50,
                "contradiction": True,
                "confidence": 0.10,
                "reason": "different failure",
            }
            for row in candidates
        ]
        return {
            "decision": "abstain",
            "selected_sop_id": "",
            "selected_transition_id": "",
            "final_confidence": 0.10,
            "reason": "none share the same root cause",
            "assessments": assessments,
        }

    layer._experiment_r_agentic_query_fn = query_fn
    layer.experiment_r_l3_agent_match_enabled = True
    result = _agentic_l3_debug_match(
        layer,
        task_id="aerial-cactus-identification",
        task_desc="Vision binary image classification",
        query_text="RuntimeError: a genuinely unseen library failure",
        visible_sop_ids=None,
    )
    assert scopes == ["exact_task", "same_task_type"]
    assert result["decision"] == "abstain"
    assert result["selected_sop_id"] == ""
    assert result["agent_calls"] == 2


def test_dynamic_l3_agent_failure_abstains_without_manual_router_fallback():
    from agents.memory.experiment_r_router import _agentic_l3_debug_match

    layer = _real_recipe_layer()
    layer._experiment_r_agentic_query_fn = lambda **_kwargs: {
        "decision": "select",
        "selected_sop_id": "invented-repair",
        "selected_transition_id": "invented-transition",
        "final_confidence": 1.0,
        "reason": "invalid",
        "assessments": [],
    }
    layer.experiment_r_l3_agent_match_enabled = True
    layer.experiment_r_l3_agent_match_max_attempts = 2
    result = _agentic_l3_debug_match(
        layer,
        task_id="aerial-cactus-identification",
        task_desc="Vision binary image classification",
        query_text="RuntimeError: unknown failure",
        visible_sop_ids=None,
    )
    assert result["decision"] == "agent_failure_abstain"
    assert result["selected_sop_id"] == ""
    assert result["manual_synonym_table_used"] is False
    assert result["agent_calls"] == 2
    assert len(result["trace"][0]["attempts"]) == 2


def test_layered_dynamic_debug_uses_agent_match_before_prompt_injection():
    layer = _real_recipe_layer()
    calls = []

    def query_fn(**kwargs):
        calls.append(kwargs["func_spec"].name)
        candidates = json.loads(
            kwargs["system_message"]["authorized_l3_candidates"]
        )
        selected = next(
            row
            for row in candidates
            if row["sop_id"] == "repair::aerial-cactus-identification::005"
        )
        return {
            "decision": "select",
            "selected_sop_id": selected["sop_id"],
            "selected_transition_id": selected["transition_id"],
            "final_confidence": 0.97,
            "reason": "same convolutional classifier dimension contract",
            "assessments": [
                {
                    "sop_id": row["sop_id"],
                    "keyword_correspondence": 0.96 if row is selected else 0.10,
                    "root_cause_equivalence": 0.99 if row is selected else 0.10,
                    "runtime_stage_match": 1.0 if row is selected else 0.40,
                    "contradiction": row is not selected,
                    "confidence": 0.97 if row is selected else 0.10,
                    "reason": "same root cause" if row is selected else "different",
                }
                for row in candidates
            ],
        }

    layer._experiment_r_agentic_query_fn = query_fn
    layer.experiment_r_l3_agent_match_enabled = True
    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="aerial-cactus-identification",
        task_desc="Vision binary image classification",
        query_parts=[
            "RuntimeError: ConvNeXt returns [32, 768, 8, 8] and the "
            "classification head passes it to LayerNorm([768]) without pooling."
        ],
    )
    pack = layer.current_navigation_pack()
    assert calls == ["choose_l3_debug_repair_by_root_cause"]
    assert pack["algorithm_version"] == "stage_hybrid_l3_agent_root_cause_v1"
    assert pack["selected_sop_gateways"][0]["id"] == (
        "repair::aerial-cactus-identification::005"
    )
    assert "repair::aerial-cactus-identification::005" in refs
    assert "repair::aerial-cactus-identification::005" in text
    assert pack["gateway_selection"]["manual_synonym_table_used"] is False
    assert pack["l3_agent_match"]["manual_synonym_table_used"] is False
    assert pack["tree_candidate_details"][0]["ranking_backend"] == (
        "agent_keyword_and_root_cause_semantic_match_v1"
    )
    assert (
        "failure_signature_match"
        not in pack["tree_candidate_details"][0]["score_components"]
    )


def test_dynamic_experiment_r_prompt_is_pinned_to_agent_selected_l3():
    layer = _real_recipe_layer()

    def query_fn(**kwargs):
        spec_name = kwargs["func_spec"].name
        prompt = kwargs["system_message"]
        if spec_name == "choose_l3_debug_repair_by_root_cause":
            candidates = json.loads(prompt["authorized_l3_candidates"])
            selected = next(
                row
                for row in candidates
                if row["sop_id"] == "repair::aerial-cactus-identification::005"
            )
            return {
                "decision": "select",
                "selected_sop_id": selected["sop_id"],
                "selected_transition_id": selected["transition_id"],
                "final_confidence": 0.97,
                "reason": "same classifier dimension root cause",
                "assessments": [
                    {
                        "sop_id": row["sop_id"],
                        "keyword_correspondence": (
                            0.95 if row is selected else 0.10
                        ),
                        "root_cause_equivalence": (
                            0.99 if row is selected else 0.10
                        ),
                        "runtime_stage_match": 1.0 if row is selected else 0.40,
                        "contradiction": row is not selected,
                        "confidence": 0.97 if row is selected else 0.10,
                        "reason": "selected" if row is selected else "different",
                    }
                    for row in candidates
                ],
            }
        known = json.loads(prompt["known_candidates"])
        contract = json.loads(prompt["final_selection_contract"])
        by_source = {
            source: [row["id"] for row in known if row["source"] == source]
            for source in ("sop", "runforest")
        }
        selected_ids = ["repair::aerial-cactus-identification::005"]
        for source in ("sop", "runforest"):
            required = int(contract["minimum_source_counts"][source])
            current = sum(
                node_id in set(by_source[source]) for node_id in selected_ids
            )
            selected_ids.extend(
                node_id
                for node_id in by_source[source]
                if node_id not in selected_ids
            )
            # Keep only as many newly appended source IDs as its minimum.
            keep = required - current
            if keep < 0:
                keep = 0
            source_selected = [
                node_id for node_id in selected_ids if node_id in by_source[source]
            ]
            for node_id in source_selected[required:]:
                if node_id != "repair::aerial-cactus-identification::005":
                    selected_ids.remove(node_id)
        for row in known:
            if len(selected_ids) >= int(contract["exact_selection_count"]):
                break
            if row["id"] not in selected_ids:
                selected_ids.append(row["id"])
        return {
            "action": "finish",
            "reason": "selected root-cause repair and clean supporting memory",
            "selected_ids": selected_ids,
        }

    layer.experiment_r_enabled = True
    layer.retrieval_control = "dynamic_hybrid"
    layer.experiment_r_agentic_retrieval_enabled = True
    layer.experiment_r_l3_agent_match_enabled = True
    layer.experiment_r_memory_pool_sha256 = "a" * 64
    layer._experiment_r_agentic_query_fn = query_fn
    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="aerial-cactus-identification",
        task_desc="Vision binary image classification",
        query_parts=[
            "RuntimeError: backbone output [32, 768, 8, 8] was passed into "
            "LayerNorm([768]) before spatial pooling."
        ],
    )
    pack = layer.current_navigation_pack()
    assert "repair::aerial-cactus-identification::005" in refs
    assert "repair::aerial-cactus-identification::005" in text
    assert pack["l3_agent_match"]["decision"] == "select"
    assert pack["l3_agent_match"]["manual_synonym_table_used"] is False
    assert pack["stage_route"]["l3_agent_prompt_pin"]["prompt_visible"] is True
    assert pack["candidate_pool"]["ranking_contract"].endswith(
        "+l3_agent_root_cause_match_v1"
    )


def test_layered_debug_prompt_requires_a_causal_l3_match_and_abstains_on_infrastructure():
    layer = _real_recipe_layer()
    exact_query = (
        "TypeError during model initialization: ModernBertForSequenceClassification "
        "received unexpected keyword argument hidden_dropout_prob. Configure "
        "dropout through the checkpoint config."
    )
    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="spooky-author-identification",
        task_desc="NLP multiclass author classification",
        query_parts=[exact_query],
    )
    pack = layer.current_navigation_pack()
    assert "repair::spooky-author-identification::001" in text
    assert "repair::spooky-author-identification::001" in refs
    assert {
        row["task_scope"] for row in pack["tree_candidate_details"]
    } == {"exact_task"}

    text, refs = layer.retrieve_for_node(
        stage="debug",
        task_id="spooky-author-identification",
        task_desc="NLP multiclass author classification",
        query_parts=[
            "Permission denied in a node cache; a temporary file is missing, "
            "the Pod is Pending, and the API timed out."
        ],
    )
    pack = layer.current_navigation_pack()
    assert text == ""
    assert refs == []
    assert pack["selected_sop_gateways"] == []
    assert pack["fused_execution_candidates"] == []
    assert pack["memory_abstention"]["status"] == "abstain"


def test_layered_router_without_replay_manifest_has_no_replay_exclusion(tmp_path):
    layer = _real_recipe_layer()
    layer.cfg.agent.draft_role_policy.replay_targets_path = ""
    assert layer._replay_family("leaf-classification") == ""


def test_recipe_layered_leaf_draft_is_task_local_and_prompt_contains_complete_l1_recipe():
    from agents.adoption import log_adoption

    layer = _real_recipe_layer()
    _inject_frozen_recipe_evidence(
        layer,
        "leaf-classification",
        "spooky-author-identification",
    )
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="leaf-classification",
        task_desc="Leaf multimodal classification evaluated by multiclass log loss.",
        query_parts=["Use one A100 and leakage-safe validation."],
        draft_role="novel_exploration",
        context={
            "excluded_method_families": [],
            "data_preview": "Train shape: (990, 194)",
            "ram_gb": 64,
        },
    )
    pack = layer.current_navigation_pack()
    assert pack["schema"] == "layered_strategy_memory_pack_v1"
    assert len(pack["strategy_routes"]) == 3
    assert all(
        row["sop_id"].startswith("recipe::leaf-classification::")
        for row in pack["strategy_candidates"]
    )
    assert not any("spooky-author-identification" in row["sop_id"] for row in pack["strategy_candidates"])
    assert "Complete end-to-end recipe:" in text
    assert "Data validation:" in text
    assert "Model stack:" in text
    assert "OOF protocol:" in text
    assert pack["selected_strategy"]["sop_id"] in refs
    assert pack["selected_strategy"]["best_tree_evidence"]["node_id"] in refs
    assert pack["selected_strategy"]["best_tree_evidence"]["evidence_kind"] == "direct_clean_run_node"

    node = SimpleNamespace(adoption_log=[], memory_navigation_trace=[])
    agent = SimpleNamespace(
        adoption_tracking_enabled=True,
        evaluation_authority=None,
        external_skill_memory=layer,
    )
    log_adoption(node, agent, layer.source_name, refs, "draft")
    route = node.memory_routing_trace
    assert route["memory_pack_schema"] == "layered_strategy_memory_pack_v1"
    assert route["stage_route"]["stage"] == "draft"
    assert route["final_prompt_candidate_ids"] == list(dict.fromkeys(refs))
    assert len(route["raw_candidates"]) >= 3
    assert len(route["suppressed_candidates"]) >= 2
    assert {row["candidate_id"] for row in route["final_prompt_candidates"]} == set(
        refs
    )
    assert route["prompt_token_count_available"] is False


def test_dynamic_replay_uses_sealed_leaf_terminal_capsule_and_novel_excludes_its_family():
    from agents.memory.run_forest_replay import load_exact_replay

    layer = _real_recipe_layer()
    agent = SimpleNamespace(
        cfg=layer.cfg,
        acfg=layer.cfg.agent,
        external_skill_memory=layer,
        evaluation_authority=None,
    )
    replay = load_exact_replay(agent)
    assert replay["replay_source"]["source_kind"] == "recipe_implementation_capsule"
    assert replay["replay_source"]["historical_metric"] == pytest.approx(
        0.08612996973006647
    )
    assert replay["replay_source"]["graph_node_id"].startswith(
        "postsmoke::e2e-smoke-leaf-layered-recipe-v4"
    )
    assert replay["replay_source"]["sop_ids"] == [
        "recipe::leaf-classification::003"
    ]
    assert replay["replay_source"]["code_sha256"] == (
        "79e202f8e3ce146a8b867461fe9d77100073fa93f86d4fb8dc794e2164002c89"
    )
    assert "SigLIP" in replay["code"]
    assert "MultiheadAttention" in replay["code"]

    profile = layer._build_task_profile(
        task_id="leaf-classification",
        task_desc="Leaf multimodal classification evaluated by multiclass log loss.",
        context={
            "baseline_model": "DINOv3",
            "data_preview": "Train shape: (990, 194)",
            "ram_gb": 64,
        },
    )
    assert "siglip2_multibranch_self_attention_fusion" in profile[
        "excluded_method_families"
    ]
    routes = layer._rank_strategy_routes(
        query_text="try a method different from exact replay",
        task_profile=profile,
    )
    assert len(routes) >= 3
    assert all(
        row["method_family"] != "siglip2_multibranch_self_attention_fusion"
        for row in routes
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash_mismatch", "manifest hash"),
        ("missing_capsule", "no frozen implementation capsule"),
        ("graph_node_mismatch", "capsule node mismatch"),
        ("metric_mismatch", "metric does not match target manifest"),
    ],
)
def test_recipe_capsule_replay_fails_closed_on_binding_mismatch(
    mutation, message
):
    from agents.memory.run_forest_replay import load_exact_replay

    layer = _real_recipe_layer()
    node_id = (
        "postsmoke::e2e-smoke-leaf-layered-recipe-v4__leaf-classification__"
        "dynamic_hybrid__seed-1::d2fccc688085447c9ad84356deac9194"
    )
    node = layer.nodes[node_id]
    if mutation == "hash_mismatch":
        node["implementation_capsule"]["code"] += "\n# changed after freeze\n"
    elif mutation == "missing_capsule":
        node.pop("implementation_capsule")
    elif mutation == "graph_node_mismatch":
        node["implementation_capsule"]["node_id"] = "postsmoke::wrong-node"
    elif mutation == "metric_mismatch":
        node["metric"] = float(node["metric"]) + 0.01

    agent = SimpleNamespace(
        cfg=layer.cfg,
        acfg=layer.cfg.agent,
        external_skill_memory=layer,
        evaluation_authority=None,
    )
    with pytest.raises(ValueError, match=message):
        load_exact_replay(agent)


def test_recipe_layered_leaf_pins_best_clean_terminal_result_before_internal_metrics():
    layer = _real_recipe_layer()
    _inject_frozen_recipe_evidence(layer, "leaf-classification")
    text, refs = layer.retrieve_for_node(
        stage="draft",
        task_id="leaf-classification",
        task_desc="Leaf multimodal classification evaluated by multiclass log loss.",
        query_parts=["Use one A100 and prefer the best clean same-task terminal result."],
        draft_role="novel_exploration",
        context={
            "excluded_method_families": [],
            "data_preview": "Train shape: (990, 194)",
            "ram_gb": 64,
        },
    )
    pack = layer.current_navigation_pack()
    selected = pack["selected_strategy"]
    evidence = selected["best_tree_evidence"]
    assert pack["mandatory_same_task_terminal_strategy"] == "recipe::leaf-classification::003"
    assert pack["strategy_routes"][0]["same_task_terminal_best"] is True
    assert selected["sop_id"] == "recipe::leaf-classification::003"
    assert evidence["node_id"].startswith(
        "postsmoke::e2e-smoke-leaf-layered-recipe-v4__leaf-classification__dynamic_hybrid"
    )
    assert evidence["metric"] == pytest.approx(0.08612996973006647)
    assert evidence["metric_direction"] == "minimize"
    assert evidence["metric_provenance"] == "sealed_fixed_holdout_terminal_score"
    assert evidence["terminal_evidence"] is True
    assert evidence["node_id"] in refs
    assert "terminal=True" in text
    assert "Exact Same-Task RunForest Implementation" in text
    assert "class LeafMultimodalClassifier" in text


def test_recipe_layered_agent_cannot_bypass_compute_feasible_terminal_best_route():
    layer = _real_recipe_layer()
    _inject_frozen_recipe_evidence(layer, "leaf-classification")
    layer.agentic_enabled = True

    def choose_non_terminal(*, task_profile, routes):
        del task_profile
        rejected = next(route for route in routes if not route["same_task_terminal_best"])
        return {
            "strategy_sop_id": rejected["sop_id"],
            "method_family": rejected["method_family"],
            "hypothesis": "Ignore the measured terminal result.",
            "validation_plan": "Run cross-validation.",
            "model_components": [rejected["method_family"]],
            "reason": "Theoretical preference only.",
        }

    layer._injected_strategy_selector = choose_non_terminal
    layer.retrieve_for_node(
        stage="draft",
        task_id="leaf-classification",
        task_desc="Leaf multimodal classification evaluated by multiclass log loss.",
        query_parts=["Use one A100."],
        draft_role="novel_exploration",
        context={
            "excluded_method_families": [],
            "data_preview": "Train shape: (990, 194)",
            "ram_gb": 64,
        },
    )
    pack = layer.current_navigation_pack()
    assert pack["selected_strategy"]["sop_id"] == "recipe::leaf-classification::003"
    assert pack["strategy_selection"]["mode"] == "deterministic_fallback"
    assert pack["strategy_selection"]["llm_tool_calls"] == 2
    assert "bypassed mandatory same-task terminal-best" in pack["strategy_selection"]["last_error"]


def test_recipe_layered_model_design_only_expands_same_task_same_family_l2():
    layer = _real_recipe_layer()
    _inject_frozen_recipe_evidence(layer, "leaf-classification")
    layer.retrieve_for_node(
        stage="draft",
        task_id="leaf-classification",
        task_desc="Leaf multimodal classification evaluated by multiclass log loss.",
        query_parts=["Use one A100."],
        draft_role="novel_exploration",
        context={"excluded_method_families": []},
    )
    strategy = layer.current_navigation_pack()
    text, refs, l2 = layer.retrieve_model_design_tactics(
        task_id="leaf-classification",
        task_desc="Leaf multimodal classification evaluated by multiclass log loss.",
        strategy_context=strategy,
    )
    family = l2["method_family"]
    assert l2["selected_tactics"]
    for tactic in l2["selected_tactics"]:
        node = layer.nodes[tactic["sop_id"]]
        assert tactic["sop_id"].startswith("tactic::leaf-classification::")
        assert family in node["parent_method_families"]
        assert tactic["sop_id"] in refs
        assert tactic["best_tree_evidence"]["node_id"] in refs
    assert "L2 Model-Design Tactics" in text
    assert "tactic::spooky-author-identification" not in text


def test_recipe_layered_improve_injects_positive_transitions_not_generic_sops():
    layer = _real_recipe_layer()
    text, refs = layer.retrieve_for_node(
        stage="improve",
        task_id="leaf-classification",
        task_desc="Leaf multiclass log loss improvement.",
        query_parts=["Improve validation log loss with a proven change."],
        strategy_context={
            "selected_strategy": {
                "method_family": "siglip2_multibranch_self_attention_fusion"
            }
        },
    )
    pack = layer.current_navigation_pack()
    assert pack["gateway_selection"]["mode"] == "layered_transition_only"
    assert pack["selected_sop_gateways"] == []
    assert pack["fused_execution_candidates"]
    for row in pack["fused_execution_candidates"]:
        node = layer.nodes[row["id"]]
        assert node["type"] == "Transition"
        assert node["outcome"] == "metric_improved"
        assert row["id"] in refs
    assert "Proven improvement action:" in text


@pytest.mark.parametrize(
    "family",
    [
        "frozen_resnet_tabular_concat_mlp",
        "dinov2_bidirectional_cross_attention_fusion",
        "siglip2_multibranch_self_attention_fusion",
        "tabular_residual_se_multibranch_mlp",
        "foldwise_lightgbm_logistic_weighted_blend",
        "tabular_feature_group_transformer",
    ],
)
def test_leaf_recipe_families_have_runtime_code_alignment_signatures(family):
    from agents.memory.stage_aware_hybrid_memory import FAMILY_CODE_SIGNATURES

    assert family in FAMILY_CODE_SIGNATURES
    assert FAMILY_CODE_SIGNATURES[family]


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
