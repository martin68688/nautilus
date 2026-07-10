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
REPLAY_TARGETS = REPO / "paper-skills" / "eval_skill_memory" / "clean_replay_targets.json"


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
    assert meta["provenance_status"] == "source_allowlisted_and_code_audited"
    assert meta["source_membership_verified"] is True
    assert meta["leak_audited"] is True
    assert meta["positive_admission_enforced"] is True
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

    run_nodes = [node for node in graph["nodes"] if node.get("type") == "RunNode" and node.get("code_length", 0) > 0]
    assert run_nodes
    assert all(len(node.get("code_sha256", "")) == 64 for node in run_nodes)
    assert all(node.get("leakage_audit", {}).get("schema") == "mlevolve_leakage_audit_v2" for node in run_nodes)
    assert report["failure_pattern_count"] > 0


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


def test_run_forest_config_assigns_explicit_draft_roles():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(RUN_FOREST_CONFIG)
    policy = cfg.agent.draft_role_policy
    assert policy.enabled is True
    assert list(policy.roles) == [
        "coldstart_baseline",
        "memory_reproduction",
        "novel_exploration",
    ]
    assert policy.extra_role == "novel_exploration"
    assert Path(policy.replay_targets_path).name == REPLAY_TARGETS.name


def test_draft_role_policy_validates_capacity_and_extra_role():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch

    policy = SimpleNamespace(
        enabled=True,
        roles=["coldstart_baseline", "memory_reproduction", "novel_exploration"],
        extra_role="novel_exploration",
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.acfg = SimpleNamespace(initial_drafts=3, draft_role_policy=policy)
    agent.scfg = SimpleNamespace(num_drafts=5)
    AgentSearch._validate_draft_role_policy(agent)
    assert AgentSearch.configured_draft_role(agent, 0) == "coldstart_baseline"
    assert AgentSearch.configured_draft_role(agent, 1) == "memory_reproduction"
    assert AgentSearch.configured_draft_role(agent, 2) == "novel_exploration"
    assert AgentSearch.configured_draft_role(agent, 4) == "novel_exploration"

    agent.acfg.initial_drafts = 2
    with pytest.raises(ValueError, match="initial_drafts >= 3"):
        AgentSearch._validate_draft_role_policy(agent)


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


def _replay_agent(targets_path=REPLAY_TARGETS):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents.memory.external_skill_memory import RunForestMemoryLayer

    layer = RunForestMemoryLayer(
        graph_path=str(GRAPH),
        index_path=str(INDEX),
        source_name="run_forest_agentic_memory",
        mode="run_forest_agentic",
        scoring_mode="poincare",
        enable_agentic=False,
        top_k=4,
    )
    return SimpleNamespace(
        cfg=SimpleNamespace(exp_id="spooky-author-identification"),
        acfg=SimpleNamespace(
            draft_role_policy=SimpleNamespace(replay_targets_path=str(targets_path))
        ),
        external_skill_memory=layer,
    )


def test_exact_replay_rejects_historical_three_model_source_with_known_protocol_issues():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents.memory.run_forest_replay import load_exact_replay

    manifest = json.loads(REPLAY_TARGETS.read_text(encoding="utf-8"))
    target = manifest["targets"][0]
    assert target["audit_status"] == "candidate_replay"
    assert set(target["known_issue_codes"]) == {
        "TRANSFORM_FIT_ON_HOLDOUT",
        "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION",
    }
    with pytest.raises(ValueError, match="TRANSFORM_FIT_ON_HOLDOUT"):
        load_exact_replay(_replay_agent())


def test_exact_replay_fails_closed_on_hash_or_provenance_mismatch(tmp_path):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents.memory.run_forest_replay import load_exact_replay

    manifest = json.loads(REPLAY_TARGETS.read_text(encoding="utf-8"))
    manifest["targets"][0]["code_sha256"] = "0" * 64
    bad_manifest = tmp_path / "bad_replay_targets.json"
    bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash"):
        load_exact_replay(_replay_agent(bad_manifest))

    agent = _replay_agent()
    agent.external_skill_memory.graph["meta"]["paper_grade"] = False
    with pytest.raises(ValueError, match="clean-certified"):
        load_exact_replay(agent)


def test_role_metadata_is_inherited_by_debug_and_improve_nodes():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents.triggers import register_node
    from engine.search_node import SearchNode

    parent = SearchNode(
        code="print('source')",
        plan="source",
        stage="draft",
        branch_id=2,
        draft_role="memory_reproduction",
        role_contract={"role": "memory_reproduction"},
        source_ref_ids=["sop::sg_0108"],
        replay_source={"code_sha256": "abc"},
        replay_status="exact_source_loaded",
    )
    from agents.leakage_audit import audit_code
    parent.leakage_audit = audit_code(
        "X_train, X_val = train_test_split(X)\nTfidfVectorizer().fit_transform(X_val)"
    )
    child = SearchNode(code="print('fixed')", plan="fix", stage="debug", parent=parent)
    fake_agent = SimpleNamespace(
        _serialize_prompt=lambda prompt: str(prompt),
        next_branch_id=3,
        branch_all_nodes={2: [parent]},
        branch_successful_nodes={2: []},
    )
    register_node(fake_agent, child, "debug", parent_node=parent)
    assert child.branch_id == 2
    assert child.draft_role == "memory_reproduction"
    assert child.role_contract == parent.role_contract
    assert child.source_ref_ids == parent.source_ref_ids
    assert child.replay_source == parent.replay_source
    assert child.replay_status == "exact_source_loaded"
    assert child.leakage_audit == {}
    assert child.audit_repair_required is True
    assert child.leakage_repair_attempt == 1
    assert child.leakage_repair_context["source_code_sha256"] == parent.leakage_audit["code_sha256"]
    contract = child.leakage_repair_context["preservation_contract"]
    assert contract["status"] == "frozen"
    from agents.leakage_audit import code_sha256
    assert contract["source_code_sha256"] == code_sha256(parent.code)


def test_repair_preservation_contract_cannot_ratchet_down_across_attempts():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents.leakage_audit import audit_code
    from agents.triggers import register_node
    from engine.search_node import SearchNode

    original = SearchNode(
        code="TfidfVectorizer()\nXGBClassifier()\nLogisticRegression()\n",
        plan="source", stage="draft", branch_id=4,
    )
    original.leakage_audit = audit_code(
        "X_train, X_val = train_test_split(X)\nTfidfVectorizer().fit_transform(X_val)"
    )
    fake_agent = SimpleNamespace(
        _serialize_prompt=str, next_branch_id=5,
        branch_all_nodes={4: [original]}, branch_successful_nodes={4: []},
    )
    first = SearchNode(code="TfidfVectorizer()", plan="bad repair", stage="debug", parent=original)
    register_node(fake_agent, first, "first", parent_node=original)
    first.leakage_audit = audit_code(
        "X_train, X_val = train_test_split(X)\nTfidfVectorizer().fit_transform(X_val)"
    )
    second = SearchNode(code="print('simpler')", plan="worse repair", stage="debug", parent=first)
    register_node(fake_agent, second, "second", parent_node=first)
    assert (
        second.leakage_repair_context["preservation_contract"]
        == first.leakage_repair_context["preservation_contract"]
    )
    assert second.leakage_repair_context["preservation_contract"]["component_calls"] == {
        "LogisticRegression": 1,
        "TfidfVectorizer": 1,
        "XGBClassifier": 1,
    }


def test_role_specific_prompt_rules_are_not_global():
    source = (REPO / "mlevolve" / "agents" / "draft_agent.py").read_text(encoding="utf-8")
    assert "This first solution design should be relatively simple" not in source
    assert "Your solution MUST be NOVEL compared to ALL existing attempts" not in source
    assert 'draft_role == "novel_exploration"' in source
    assert 'draft_role == "coldstart_baseline"' in source
    assert 'draft_role != "coldstart_baseline"' in source


def test_repair_contract_is_high_priority_in_debug_and_improve_prompts():
    for relative in ("mlevolve/agents/debug_agent.py", "mlevolve/agents/improve_agent.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        assert "LEAKAGE REPAIR CONTRACT - HIGHEST PRIORITY" in source
        assert "fresh audit" in source


def test_d93_structural_rename_still_matches_failure_patterns():
    import ast
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents.memory.external_skill_memory import RunForestMemoryLayer

    target = json.loads(REPLAY_TARGETS.read_text(encoding="utf-8"))["targets"][0]
    journal_path = REPO / "mlevolve" / "runs" / target["run_id"] / "logs" / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    code = next(
        node["code"] for node in journal["nodes"]
        if node["id"] == target["original_node_id"]
    )
    assigned = {
        node.id for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    class RenameLocals(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id in assigned:
                node.id = f"renamed_{node.id}"
            return node

    renamed = ast.unparse(RenameLocals().visit(ast.parse(code)))
    layer = RunForestMemoryLayer(str(GRAPH), index_path=str(INDEX), top_k=3)
    matches = layer.structural_failure_patterns(renamed)
    assert {
        "TRANSFORM_FIT_ON_HOLDOUT",
        "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION",
    }.issubset({item.get("issue_code") for item in matches})


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
    assert knowledge._LAST_PRIMARY_MODEL_NAME == "ModernBERT"
    assert "answerdotai/ModernBERT-large" in knowledge._LAST_PRIMARY_MODEL_TEXT
    assert "microsoft/deberta-v3-large" not in knowledge._LAST_PRIMARY_MODEL_TEXT
