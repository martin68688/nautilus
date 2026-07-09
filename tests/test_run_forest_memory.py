import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
GRAPH = REPO / "paper-skills" / "hyper_memory" / "run_forest_graph.json"
INDEX = REPO / "paper-skills" / "hyper_memory" / "run_forest_index.npz"
BUILDER_REPORT = REPO / "paper-skills" / "hyper_memory" / "run_forest_builder_report.json"
EVAL_REPORT = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "run_forest_memory_evaluation.json"
ALLOWLIST = REPO / "paper-skills" / "eval_skill_memory" / "clean_run_allowlist.json"
RUN_FOREST_CONFIG = REPO / "mlevolve" / "config" / "config_run_forest_agentic.yaml"
METHODOLOGY_MAP = REPO / "mlevolve" / "engine" / "coldstart" / "methodology_map.json"


def _short_run_id(value: object) -> str:
    text = str(value or "")
    parts = text.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return "_".join(parts[:2])
    return text


def test_run_forest_artifacts_exist_and_preserve_topology():
    assert GRAPH.exists()
    assert INDEX.exists()
    assert BUILDER_REPORT.exists()

    graph = json.loads(GRAPH.read_text())
    report = json.loads(BUILDER_REPORT.read_text())
    node_types = report["node_type_counts"]
    edge_kinds = report["edge_kind_counts"]

    assert graph["meta"]["schema"] == "hyperbolic_run_forest_memory_v1"
    assert node_types["RunNode"] > 0
    assert node_types["Transition"] > 0
    assert node_types["SOP"] > 0
    assert node_types["Evidence"] > 0
    assert edge_kinds["parent_of"] == node_types["Transition"]
    assert edge_kinds["has_transition"] == node_types["Transition"]
    assert edge_kinds["transition_to"] == node_types["Transition"]
    assert report["run_node_topology_preserved"] is True
    assert report["transitions_with_sop_attachments"] > 0


def test_run_forest_artifacts_are_clean_certified():
    graph = json.loads(GRAPH.read_text())
    report = json.loads(BUILDER_REPORT.read_text())
    allowlist = json.loads(ALLOWLIST.read_text())
    allowed = {entry["run_id"] for entry in allowlist["entries"] if entry.get("allowed")}

    meta = graph["meta"]
    assert meta["provenance_status"] == "clean_certified"
    assert meta["leak_verified"] is True
    assert meta["paper_grade"] is True
    assert report["paper_grade_provenance"] is True
    assert set(meta["source_runs"]) == allowed
    assert not any(str(run_id).startswith("20260512") for run_id in meta["source_runs"])

    run_nodes = [node for node in graph["nodes"] if node.get("type") in {"Run", "RunNode", "Transition", "Evidence"}]
    assert run_nodes
    node_runs = {_short_run_id(node.get("run_short_id") or node.get("run_id")) for node in run_nodes}
    assert node_runs == allowed
    assert not any(run_id.startswith("20260512") for run_id in node_runs)


def test_run_forest_builder_requires_allowlist_for_clean_mode():
    import sys

    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    from build_run_forest_memory import build_artifact

    with pytest.raises(ValueError, match="requires --allowlist"):
        build_artifact(REPO / "mlevolve" / "runs", REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json", require_clean_provenance=True)


def test_run_forest_coordinates_have_clean_controls():
    index = np.load(INDEX, allow_pickle=True)
    poincare = index["poincare"]
    flat_twin = index["flat_twin"]
    euclidean = index["euclidean"]

    assert np.array_equal(poincare, flat_twin)
    assert poincare.shape[0] == euclidean.shape[0]
    assert poincare.shape[1] == 2
    assert euclidean.shape[1] == 16
    assert float(np.linalg.norm(poincare, axis=1).max()) < 1.0


def test_run_forest_online_config_disables_contaminated_methodology():
    cfg_text = RUN_FOREST_CONFIG.read_text(encoding="utf-8")
    mapping = json.loads(METHODOLOGY_MAP.read_text(encoding="utf-8"))

    assert 'methodology_kb_path: ""' in cfg_text
    assert "methodology_dynamic: False" in cfg_text
    spooky_entries = mapping.get("spooky-author-identification", [])
    assert "winning-recipe-nlp-classification" not in spooky_entries
    assert "ensemble-diversity-vs-validation-gap" not in spooky_entries
    assert "small-data-transformer-finetuning" not in spooky_entries


def test_run_forest_evaluation_supports_lineage_claim_but_not_all_tasks():
    assert EVAL_REPORT.exists()
    report = json.loads(EVAL_REPORT.read_text())
    systems = report["systems"]

    p_parent = systems["run_forest_poincare"]["parent_lookup"]["mrr"]
    f_parent = systems["run_forest_flat_twin"]["parent_lookup"]["mrr"]
    e_parent = systems["run_forest_euclidean"]["parent_lookup"]["mrr"]
    assert p_parent > f_parent
    assert p_parent > e_parent

    p_signpost = systems["run_forest_poincare"]["transition_to_sop_signpost"]["mrr"]
    f_signpost = systems["run_forest_flat_twin"]["transition_to_sop_signpost"]["mrr"]
    e_signpost = systems["run_forest_euclidean"]["transition_to_sop_signpost"]["mrr"]
    assert p_signpost > f_signpost
    assert p_signpost > e_signpost

    p_debug = systems["run_forest_poincare"]["debug_recovery_child_lookup"]["mrr"]
    f_debug = systems["run_forest_flat_twin"]["debug_recovery_child_lookup"]["mrr"]
    assert p_debug < f_debug

    gates = report["claim_gates"]
    assert gates["lineage_backtracking"]["passed"] is True
    assert gates["debug_child_graph_expansion"]["passed"] is True
    assert gates["sop_only_geometry"]["status"] == "not_supported"


def test_run_forest_runtime_layer_returns_map_path_pack():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents.memory.external_skill_memory import RunForestMemoryLayer, external_memory_section_title

    layer = RunForestMemoryLayer(
        graph_path="../paper-skills/hyper_memory/run_forest_graph.json",
        index_path="../paper-skills/hyper_memory/run_forest_index.npz",
        source_name="run_forest_agentic_memory",
        mode="run_forest_agentic",
        scoring_mode="poincare",
        enable_agentic=False,
        top_k=4,
        max_chars=5000,
    )
    text, ref_ids = layer.retrieve_for_node(
        stage="debug",
        task_id="leaf-classification",
        task_desc="leaf classification with image features",
        query_parts=["Traceback shape mismatch in feature matrix during improve"],
    )
    assert "Map Path Pack JSON" in text
    assert "matched_run_paths" in text
    assert "selected_transitions" in text
    assert "attached_sops" in text
    assert ref_ids
    assert external_memory_section_title(layer.source_name) == "Agentic Run-Forest Memory Navigation"


def test_run_forest_coldstart_does_not_modify_model_template():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.coldstart import knowledge

    cfg = SimpleNamespace(
        exp_id="spooky-author-identification",
        torch_hub_dir="",
        methodology_kb_path="",
        methodology_dynamic=False,
        coldstart=SimpleNamespace(
            task_json_path=str(REPO / "mlevolve" / "engine" / "coldstart" / "competition_tag_classified.json"),
            model_json_path=str(REPO / "mlevolve" / "engine" / "coldstart" / "models_guidance_classified.json"),
        ),
        external_skill_memory=SimpleNamespace(
            enable=True,
            graph_path=str(GRAPH),
            index_path=str(INDEX),
            source_name="run_forest_memory",
            mode="run_forest",
            scoring_mode="poincare",
            enable_agentic=False,
            navigator_max_steps=3,
            navigator_reference_budget=1200,
            top_k=4,
            max_chars=4500,
        ),
    )
    text = knowledge.build_guidance_description(cfg, task_desc="Spooky author text classification")
    assert "Run-Forest" not in text
    assert "Map Path Pack" not in text
    assert "Run-Forest Cold-Start Map Path Pack" in knowledge._LAST_RUN_FOREST_TEXT
    assert knowledge._LAST_RUN_FOREST_REF_IDS
