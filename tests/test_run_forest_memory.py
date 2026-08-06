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
REPLAY_SOURCE_MANIFEST = (
    REPO / "paper-skills" / "eval_skill_memory" / "non_spooky_replay_source_manifest_v1.json"
)


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
    assert graph["meta"]["sop_taxonomy_schema"] == "runforest_sop_taxonomy_v1"
    assert graph["meta"]["sop_taxonomy_coverage"] == 1.0
    assert graph["meta"]["sop_taxonomy_sop_count"] == node_types["SOP"]
    assert graph["meta"]["sop_taxonomy_reviewed_l1_count"] == 28


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


def test_non_spooky_exact_replay_targets_bind_to_clean_graph_and_frozen_source_manifest():
    graph = json.loads(GRAPH.read_text(encoding="utf-8"))
    nodes = {str(node["id"]): node for node in graph["nodes"] if node.get("id")}
    manifest = json.loads(REPLAY_TARGETS.read_text(encoding="utf-8"))
    source_manifest = json.loads(REPLAY_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    assert source_manifest["schema"] == "non_spooky_replay_source_manifest_v1"
    sources = {row["run_id"]: row for row in source_manifest["entries"]}
    targets = {row["task_id"]: row for row in manifest["targets"]}
    expected = {
        "leaf-classification",
        "aerial-cactus-identification",
        "denoising-dirty-documents",
        "new-york-city-taxi-fare-prediction",
    }
    assert expected.issubset(targets)
    assert "mlsp-2013-birds" not in targets
    for task_id in expected:
        target = targets[task_id]
        if target.get("source_kind") == "recipe_implementation_capsule":
            # Post-freeze terminal evidence is intentionally carried by the
            # frozen Recipe overlay rather than mutating the base RunForest.
            # Its full graph/code/hash binding is exercised by the layered
            # memory tests that load that overlay.
            assert task_id == "leaf-classification"
            assert target["graph_node_id"].startswith("postsmoke::")
            assert target["metric_status"] == "sealed_fixed_holdout_terminal_score"
            assert target["maximize"] is False
            assert len(target["code_sha256"]) == 64
            continue
        run_id = target["run_id"]
        node_id = f"run::{run_id}::node::{target['original_node_id']}"
        node = nodes[node_id]
        audit = node["leakage_audit"]
        assert node["task"] == task_id
        assert audit["status"] == "clean"
        assert audit["rank_eligible"] is True
        assert audit["memory_disposition"] == "positive_eligible"
        assert node["code_sha256"] == target["code_sha256"]
        assert node["metric"] == target["historical_metric"]
        run = nodes[f"run::{run_id}"]
        source = sources[run_id]
        assert source["task_id"] == task_id
        assert source["original_node_id"] == target["original_node_id"]
        assert source["journal_path"] == run["journal_path"]
        assert source["journal_bytes"] > 0
        assert len(source["journal_sha256"]) == 64
        assert source["code_length"] > 0
        assert source["code_sha256"] == target["code_sha256"]
        assert all(sop_id in nodes and nodes[sop_id]["type"] == "SOP" for sop_id in target["sop_ids"])


def test_run_forest_builder_requires_allowlist_for_clean_mode():
    import sys

    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    from build_run_forest_memory import build_artifact

    with pytest.raises(ValueError, match="requires --allowlist"):
        build_artifact(REPO / "mlevolve" / "runs", REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json", require_clean_provenance=True)


def test_builder_canonicalizes_full_orchestration_prefix_without_changing_run_id():
    import sys

    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    import build_run_forest_memory as builder

    run_id = "20260717_032623_full-leaf-classification"
    assert builder.run_short_id(run_id) == "20260717_032623"
    assert builder.task_from_run_id(run_id) == "leaf-classification"
    assert builder.canonical_task_id("full-full-aerial-cactus-identification") == "aerial-cactus-identification"


def test_load_journals_rejects_non_allowlisted_runs_before_reading(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    import build_run_forest_memory as builder

    allowed_id = "20260101_000000_allowed-task"
    excluded_id = "20260102_000000_excluded-task"
    allowed_path = tmp_path / allowed_id / "logs" / "journal.json"
    excluded_path = tmp_path / excluded_id / "logs" / "journal.json"
    allowed_path.parent.mkdir(parents=True)
    excluded_path.parent.mkdir(parents=True)
    allowed_path.write_text(json.dumps({"nodes": [{"id": "root"}, {"id": "child"}]}), encoding="utf-8")
    excluded_path.write_text("must not be read", encoding="utf-8")

    original_read_json = builder.read_json
    read_paths = []

    def guarded_read_json(path):
        path = Path(path)
        read_paths.append(path)
        if path == excluded_path:
            raise AssertionError("non-allowlisted journal was read")
        return original_read_json(path)

    monkeypatch.setattr(builder, "read_json", guarded_read_json)
    rows, report = builder.load_journals(
        tmp_path,
        {
            "allowed_run_ids": ["20260101_000000"],
            "blocked_prefixes": [],
        },
    )

    assert [row[0] for row in rows] == [allowed_id]
    assert read_paths == [allowed_path]
    assert report["discovered_journal_count"] == 2
    assert report["included_journal_count"] == 1
    assert report["excluded_by_reason"] == {"not_allowlisted": 1}


def test_sop_clause_publication_requires_lineage_and_quarantines_uncertified():
    import sys

    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    import build_run_forest_memory as builder

    sop = {
        "id": "s1",
        "source_branches": [["run1", "2"]],
        "evidence_turns": ["run::run1::node::n1"],
    }
    allowed = builder.clause_lineage_for_sop(sop, publication_allowed=True)
    assert {item["field"] for item in allowed} == {"title", "action", "applies_when", "prevents"}
    assert all(item["outcome"] == "allow" for item in allowed)
    assert all(item["scope_widened"] is False for item in allowed)

    quarantined = builder.clause_lineage_for_sop({"id": "s2"}, publication_allowed=False)
    assert all(item["outcome"] == "quarantine" for item in quarantined)
    with pytest.raises(ValueError, match="no parent evidence refs"):
        builder.clause_lineage_for_sop({"id": "s3"}, publication_allowed=True)


def test_sop_taxonomy_stale_hash_fails_closed(tmp_path):
    import sys

    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    from build_run_forest_memory import load_sops

    graph_path = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
    taxonomy = json.loads(
        (REPO / "paper-skills" / "hyper_memory" / "sop_taxonomy.json").read_text(encoding="utf-8")
    )
    taxonomy["source_graph_sha256"] = "0" * 64
    stale = tmp_path / "stale_taxonomy.json"
    stale.write_text(json.dumps(taxonomy), encoding="utf-8")
    with pytest.raises(ValueError, match="taxonomy is stale"):
        load_sops(graph_path, taxonomy_path=stale)


def test_sop_taxonomy_illegal_entry_fails_closed(tmp_path):
    import sys

    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    from build_run_forest_memory import load_sops

    graph_path = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
    taxonomy = json.loads(
        (REPO / "paper-skills" / "hyper_memory" / "sop_taxonomy.json").read_text(encoding="utf-8")
    )
    taxonomy["entries"]["sg_0001"]["compute_profile"] = "unbounded_cluster"
    invalid = tmp_path / "invalid_taxonomy.json"
    invalid.write_text(json.dumps(taxonomy), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid compute_profile"):
        load_sops(graph_path, taxonomy_path=invalid)


def test_sop_taxonomy_requires_complete_manual_l1_review(tmp_path):
    import sys

    sys.path.insert(0, str(REPO / "paper-skills" / "hyper_memory"))
    from build_sop_taxonomy import build_taxonomy

    graph_path = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
    source = REPO / "paper-skills" / "hyper_memory" / "sop_taxonomy_overrides.json"
    overrides = json.loads(source.read_text(encoding="utf-8"))
    overrides["reviewed_l1_ids"] = overrides["reviewed_l1_ids"][1:]
    incomplete = tmp_path / "incomplete_l1_review.json"
    incomplete.write_text(json.dumps(overrides), encoding="utf-8")

    with pytest.raises(ValueError, match="Manual L1 review coverage mismatch"):
        build_taxonomy(graph_path, incomplete)


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
    assert cfg.agent.initial_drafts == 3
    assert cfg.agent.search.num_drafts == 3
    assert Path(policy.replay_targets_path).name == REPLAY_TARGETS.name


def test_draft_role_policy_validates_fixed_three_roles():
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
    agent.scfg = SimpleNamespace(num_drafts=3)
    AgentSearch._validate_draft_role_policy(agent)
    assert AgentSearch.configured_draft_role(agent, 0) == "coldstart_baseline"
    assert AgentSearch.configured_draft_role(agent, 1) == "memory_reproduction"
    assert AgentSearch.configured_draft_role(agent, 2) == "novel_exploration"
    with pytest.raises(ValueError, match="exceeds the fixed three-role policy"):
        AgentSearch.configured_draft_role(agent, 3)

    agent.acfg.initial_drafts = 2
    with pytest.raises(ValueError, match="initial_drafts == 3"):
        AgentSearch._validate_draft_role_policy(agent)

    agent.acfg.initial_drafts = 3
    policy.roles = ["coldstart_baseline", "memory_transfer", "novel_exploration"]
    AgentSearch._validate_draft_role_policy(agent)
    assert AgentSearch.configured_draft_role(agent, 1) == "memory_transfer"


def test_seven_workers_atomically_claim_only_three_root_roles():
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch

    policy = SimpleNamespace(
        enabled=True,
        roles=["coldstart_baseline", "memory_reproduction", "novel_exploration"],
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.acfg = SimpleNamespace(initial_drafts=3, draft_role_policy=policy)
    agent.scfg = SimpleNamespace(num_drafts=3)
    agent._draft_role_lock = threading.Lock()
    agent._draft_generation_count = 0

    def claim(_worker):
        try:
            return "claimed", AgentSearch.claim_draft_role(agent)
        except ValueError as exc:
            return "blocked", str(exc)

    with ThreadPoolExecutor(max_workers=7) as pool:
        results = list(pool.map(claim, range(7)))

    claimed = [value for status, value in results if status == "claimed"]
    blocked = [value for status, value in results if status == "blocked"]
    assert set(claimed) == {"coldstart_baseline", "memory_reproduction", "novel_exploration"}
    assert len(claimed) == 3
    assert len(blocked) == 4

    agent._draft_generation_count = 0
    with pytest.raises(ValueError, match="slot 0 requires coldstart_baseline"):
        AgentSearch.claim_draft_role(agent, "novel_exploration")
    assert agent._draft_generation_count == 0


def test_rejected_draft_role_reservation_does_not_steal_child_count(monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch, DraftRoleReservationError
    from engine.search_node import SearchNode
    from utils.metric import WorstMetricValue

    root = SearchNode(code="", plan="root", stage="root", step=0, metric=WorstMetricValue())
    root.expected_child_count = 2
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.scfg = SimpleNamespace(num_drafts=3)
    agent.is_root = lambda node: node is root

    def reject_role(*_args, **_kwargs):
        raise DraftRoleReservationError("full")

    monkeypatch.setattr(
        "engine.agent_search.draft_agent.run",
        reject_role,
    )
    monkeypatch.setattr("engine.agent_search.evaluation.backpropagate", lambda *_args, **_kwargs: None)

    with pytest.raises(DraftRoleReservationError):
        AgentSearch._run_single_step(agent, root, exec_callback=lambda *_args, **_kwargs: None)
    assert root.expected_child_count == 2


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
            draft_role_policy=SimpleNamespace(replay_targets_path=str(targets_path)),
            check_data_leakage=True,
        ),
        external_skill_memory=layer,
    )


def test_exact_replay_loads_historical_three_model_source_as_blocked_repair_seed():
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
    replay = load_exact_replay(_replay_agent())
    assert replay["requires_repair"] is True
    assert replay["replay_status"] == "blocked_exact_source_repair_seed"
    assert replay["adoption_mode"] == "blocked_exact_source_repair_seed"
    assert replay["replay_source"]["target_audit_status"] == "candidate_replay"
    assert replay["replay_source"]["repair_seed_only"] is True
    assert replay["replay_source"]["journal_path"] == (
        "mlevolve/runs/20260509_185008_spooky-author-identification/logs/journal.json"
    )
    assert replay["leakage_audit"]["hard_block"] is True
    assert set(replay["replay_source"]["known_issue_codes"]) == {
        "TRANSFORM_FIT_ON_HOLDOUT",
        "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION",
    }
    for required in ("XGBClassifier", "LogisticRegression", "TfidfVectorizer"):
        assert required in replay["code"]


def test_replay_repair_seed_is_blocked_before_execution_and_freezes_original_design(tmp_path):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents import result_parse_agent
    from agents.triggers import register_node
    from engine.search_node import SearchNode

    loader_agent = _replay_agent()
    replay = __import__(
        "agents.memory.run_forest_replay", fromlist=["load_exact_replay"]
    ).load_exact_replay(loader_agent)
    seed = SearchNode(
        code=replay["code"], plan=replay["plan"], stage="draft", branch_id=1,
        draft_role="memory_reproduction", role_contract=replay["role_contract"],
        source_ref_ids=replay["source_ref_ids"], replay_source=replay["replay_source"],
        replay_status=replay["replay_status"], skip_code_review=True,
    )
    audit_agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=True),
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        global_memory=None,
        external_skill_memory=loader_agent.external_skill_memory,
    )
    assert result_parse_agent.run_pre_execution_leakage_audit(audit_agent, seed) is True
    assert seed.replay_status == "blocked_exact_source_repair_seed"
    assert seed.leakage_audit["repair_seed_execution_blocked"] is True
    assert seed.is_buggy is True
    assert seed.metric.value is None

    child = SearchNode(code=seed.code, plan="repair only", stage="debug", parent=seed)
    branch_agent = SimpleNamespace(
        _serialize_prompt=str, next_branch_id=2,
        branch_all_nodes={1: [seed]}, branch_successful_nodes={1: []},
    )
    register_node(branch_agent, child, "repair", parent_node=seed)
    contract = child.leakage_repair_context["preservation_contract"]
    assert child.audit_repair_required is True
    assert child.leakage_repair_attempt == 1
    assert contract["source_code_sha256"] == replay["replay_source"]["code_sha256"]
    assert contract["component_calls"]["XGBClassifier"] == 1
    assert contract["component_calls"]["LogisticRegression"] == 1
    assert contract["component_calls"]["TfidfVectorizer"] == 4
    assert child.replay_source["repair_seed_only"] is False
    assert child.replay_status == "mandatory_audit_repair"


def test_candidate_replay_fails_closed_when_runtime_audit_is_disabled():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents.memory.run_forest_replay import load_exact_replay

    agent = _replay_agent()
    agent.acfg.check_data_leakage = False
    with pytest.raises(ValueError, match="requires deterministic leakage auditing"):
        load_exact_replay(agent)


def test_clean_repair_child_passes_pre_execution_gate(tmp_path):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents import result_parse_agent
    from agents.leakage_audit import audit_code
    from agents.triggers import register_node
    from engine.search_node import SearchNode

    parent_code = """
X_train, X_val = train_test_split(texts, test_size=0.2)
all_texts = np.concatenate([X_train, X_val, test_texts])
vectorizer = CountVectorizer(analyzer="char")
features = vectorizer.fit_transform(all_texts)
"""
    repaired_code = """
X_train, X_val = train_test_split(texts, test_size=0.2)
vectorizer = CountVectorizer(analyzer="char")
train_features = vectorizer.fit_transform(X_train)
val_features = vectorizer.transform(X_val)
test_features = vectorizer.transform(test_texts)
"""
    seed = SearchNode(
        code=parent_code,
        plan="blocked source",
        stage="draft",
        branch_id=1,
        replay_source={
            "requires_repair": True,
            "repair_seed_only": True,
            "code_sha256": "source-hash",
        },
        replay_status="blocked_exact_source_repair_seed",
    )
    seed.leakage_audit = audit_code(parent_code)
    child = SearchNode(
        code=repaired_code,
        plan="repair",
        stage="debug",
        parent=seed,
    )
    branch_agent = SimpleNamespace(
        _serialize_prompt=str,
        next_branch_id=2,
        branch_all_nodes={1: [seed]},
        branch_successful_nodes={1: []},
    )
    register_node(branch_agent, child, "repair", parent_node=seed)
    assert child.replay_source["repair_seed_only"] is False
    assert child.replay_status == "mandatory_audit_repair"

    audit_agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=True),
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        global_memory=None,
        external_skill_memory=None,
    )
    assert result_parse_agent.run_pre_execution_leakage_audit(audit_agent, child) is False
    assert child.leakage_audit["status"] == "clean"
    assert child.replay_status == "mandatory_audit_repair_clean_pending_execution"
    assert child.resolved_issue_codes == ["TRANSFORM_FIT_ON_HOLDOUT"]


def test_mandatory_repair_scheduler_preserves_initial_roles_and_forces_runtime_repair(monkeypatch):
    import sys
    from types import MethodType

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal, SearchNode
    from utils.metric import WorstMetricValue

    root = SearchNode(
        parent=None, plan="root", code="", metric=WorstMetricValue(), stage="root"
    )
    seed = SearchNode(
        parent=root,
        plan="repair seed",
        code="XGBClassifier()",
        metric=WorstMetricValue(),
        stage="draft",
        branch_id=1,
        draft_role="memory_reproduction",
        is_buggy=True,
        is_valid=False,
        audit_repair_required=True,
        leakage_audit={"status": "blocked", "repair_required": True},
    )
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.journal = Journal(nodes=[root, seed], audit_enforced=True)
    agent.data_preview = "ready"
    agent.search_start_time = 1.0
    agent.current_step = 0
    agent.branch_all_nodes = {1: [seed]}
    agent.best_node = None
    AgentSearch._init_mandatory_repair_scheduler(agent)
    AgentSearch._enqueue_mandatory_repair(agent, seed)

    selected = []

    def fake_run_single_step(self, parent_node, **kwargs):
        selected.append(parent_node)
        return False, None

    agent._run_single_step = MethodType(fake_run_single_step, agent)
    monkeypatch.setattr(
        "engine.node_selection.select_with_soft_switch",
        lambda _agent: root,
    )

    # Sequential draft generation must not consume the repair queue or replace
    # the declared third (novel) role with a repair child.
    AgentSearch.step(
        agent,
        root,
        exec_callback=lambda *_args, **_kwargs: None,
        execute_immediately=False,
        draft_role="novel_exploration",
    )
    assert selected == [root]
    assert list(agent._mandatory_repair_queue) == [seed]

    selected.clear()
    AgentSearch.step(
        agent,
        root,
        exec_callback=lambda *_args, **_kwargs: None,
    )
    assert selected == [seed]
    assert not agent._mandatory_repair_queue
    assert not agent._mandatory_repair_inflight_ids


def test_mandatory_repair_scheduler_prevents_duplicate_parallel_expansion():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch
    from engine.search_node import SearchNode
    from utils.metric import WorstMetricValue

    seed = SearchNode(
        plan="repair seed",
        code="XGBClassifier()",
        metric=WorstMetricValue(),
        stage="draft",
        is_buggy=True,
        is_valid=False,
        audit_repair_required=True,
        leakage_audit={"status": "blocked", "repair_required": True},
    )
    agent = AgentSearch.__new__(AgentSearch)
    AgentSearch._init_mandatory_repair_scheduler(agent)
    AgentSearch._enqueue_mandatory_repair(agent, seed)

    claimed, duplicate = AgentSearch._claim_mandatory_repair_parent(agent, None)
    assert claimed is seed
    assert duplicate is False
    claimed_again, duplicate = AgentSearch._claim_mandatory_repair_parent(agent, seed)
    assert claimed_again is None
    assert duplicate is True

    AgentSearch._release_mandatory_repair_parent(agent, seed, retry=True)
    claimed_after_retry, duplicate = AgentSearch._claim_mandatory_repair_parent(agent, None)
    assert claimed_after_retry is seed
    assert duplicate is False


def test_post_execution_audit_failure_is_requeued_for_mandatory_repair(monkeypatch):
    import sys
    import threading

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal, SearchNode
    from utils.metric import WorstMetricValue

    parent = SearchNode(
        code="print('parent')",
        plan="parent",
        stage="draft",
        branch_id=1,
        is_buggy=False,
        is_valid=True,
    )
    child = SearchNode(
        code="print('executed repair')",
        plan="repair",
        stage="debug",
        parent=parent,
        branch_id=1,
        leakage_repair_attempt=1,
    )
    child.pending_execution = True

    agent = AgentSearch.__new__(AgentSearch)
    agent.journal = Journal(nodes=[parent], audit_enforced=True)
    agent.journal_lock = threading.Lock()
    agent.best_node = None
    AgentSearch._init_mandatory_repair_scheduler(agent)

    monkeypatch.setattr(
        "agents.result_parse_agent.run_pre_execution_leakage_audit",
        lambda *_args, **_kwargs: False,
    )

    def fail_post_execution_audit(_agent, node, exec_result):
        node.metric = WorstMetricValue()
        node.is_buggy = True
        node.is_valid = False
        node.audit_repair_required = True
        node.leakage_audit = {
            "status": "blocked",
            "repair_required": True,
            "hard_block": True,
        }
        return node

    monkeypatch.setattr("agents.result_parse_agent.run", fail_post_execution_audit)
    monkeypatch.setattr("engine.execution.validate_executed_node", lambda *_args: None)
    monkeypatch.setattr("engine.evaluation.check_improvement", lambda *_args: False)
    monkeypatch.setattr("engine.solution_manager.update_best_solution", lambda *_args: None)

    result = AgentSearch.execute_deferred_node(
        agent,
        child,
        exec_callback=lambda *_args, **_kwargs: object(),
    )
    assert result is child
    assert list(agent._mandatory_repair_queue) == [child]
    assert child.leakage_audit["repair_queue_status"] == "queued"


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
    assert child.replay_source["code_sha256"] == parent.replay_source["code_sha256"]
    assert child.replay_source["repair_seed_only"] is False
    assert child.replay_source["repair_parent_node_id"] == parent.id
    assert child.replay_status == "mandatory_audit_repair"
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


def test_three_locked_drafts_return_wait_instead_of_forced_root(monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine import node_selection
    from engine.search_node import SearchNode
    from utils.metric import WorstMetricValue

    root = SearchNode(code="", plan="root", stage="root", step=0, metric=WorstMetricValue())
    for role in ("coldstart_baseline", "memory_reproduction", "novel_exploration"):
        SearchNode(code="print(1)", plan=role, stage="draft", parent=root, draft_role=role, lock=True)
    root.expected_child_count = 3
    agent = SimpleNamespace(
        virtual_root=root,
        scfg=SimpleNamespace(num_drafts=3),
        acfg=SimpleNamespace(
            draft_role_policy=SimpleNamespace(enabled=True),
            branch_fusion_trigger_prob=1.0,
        ),
        is_root=lambda node: node is root,
    )
    monkeypatch.setattr(node_selection, "_compute_exploration_constant", lambda _agent: 1.0)
    assert node_selection.select(agent, root) is None
    assert getattr(root, "_aggregation_requested", False) is False


def test_agent_step_returns_explicit_wait_signal(monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch
    from engine.search_node import SearchNode
    from utils.metric import WorstMetricValue

    class NoWaitCondition:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def wait(self, timeout=None):
            return None

    root = SearchNode(code="", plan="root", stage="root", step=0, metric=WorstMetricValue())
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.journal = SimpleNamespace(nodes=[root])
    agent.data_preview = "ready"
    agent._search_condition = NoWaitCondition()
    agent._active_search_work_lock = __import__("threading").Lock()
    agent._active_search_work = 1
    monkeypatch.setattr("engine.node_selection.select_with_soft_switch", lambda _agent: None)

    result = AgentSearch.step(
        agent,
        root,
        exec_callback=lambda *_args, **_kwargs: None,
        execute_immediately=False,
        draft_role="novel_exploration",
    )
    assert result is None


def test_agent_step_raises_when_search_space_is_permanently_exhausted(monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch, SearchSpaceExhausted
    from engine.search_node import SearchNode
    from utils.metric import WorstMetricValue

    class NoWaitCondition:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def wait(self, timeout=None):
            return None

    root = SearchNode(code="", plan="root", stage="root", step=0, metric=WorstMetricValue())
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.journal = SimpleNamespace(nodes=[root])
    agent.data_preview = "ready"
    agent._search_condition = NoWaitCondition()
    agent._active_search_work_lock = __import__("threading").Lock()
    agent._active_search_work = 0
    monkeypatch.setattr("engine.node_selection.select_with_soft_switch", lambda _agent: None)

    with pytest.raises(SearchSpaceExhausted, match="No expandable node"):
        AgentSearch.step(
            agent,
            root,
            exec_callback=lambda *_args, **_kwargs: None,
            execute_immediately=False,
            draft_role="novel_exploration",
        )


def test_agent_step_ignores_stale_journal_reservations_when_exhausted(monkeypatch):
    import sys
    import threading

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch, SearchSpaceExhausted
    from engine.search_node import SearchNode
    from utils.metric import WorstMetricValue

    class NoWaitCondition:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def wait(self, timeout=None):
            return None

    root = SearchNode(code="", plan="root", stage="root", step=0, metric=WorstMetricValue())
    root.lock = True
    root.expected_child_count = 99
    agent = AgentSearch.__new__(AgentSearch)
    agent.virtual_root = root
    agent.journal = SimpleNamespace(nodes=[root])
    agent.data_preview = "ready"
    agent._search_condition = NoWaitCondition()
    agent._active_search_work_lock = threading.Lock()
    agent._active_search_work = 0
    monkeypatch.setattr("engine.node_selection.select_with_soft_switch", lambda _agent: None)

    with pytest.raises(SearchSpaceExhausted, match="No expandable node"):
        AgentSearch.step(
            agent,
            root,
            exec_callback=lambda *_args, **_kwargs: None,
            execute_immediately=False,
            draft_role="novel_exploration",
        )


def test_replacement_draft_slots_are_bounded_and_single_inflight():
    import sys
    import threading

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.agent_search import AgentSearch

    agent = AgentSearch.__new__(AgentSearch)
    agent.scfg = SimpleNamespace(
        replacement_drafts_enabled=True,
        max_replacement_drafts=2,
    )
    agent.acfg = SimpleNamespace(steps=80)
    agent.journal = [object()]
    agent._replacement_draft_lock = threading.Lock()
    agent._replacement_draft_count = 0
    agent._replacement_draft_inflight = False
    agent._notify_search_state = lambda: None

    assert agent._claim_replacement_draft_slot() is True
    assert agent._claim_replacement_draft_slot() is False
    assert agent._replacement_draft_count == 1

    agent._release_replacement_draft_slot()
    assert agent._claim_replacement_draft_slot() is True
    agent._release_replacement_draft_slot()
    assert agent._claim_replacement_draft_slot() is False
    assert agent._replacement_draft_count == 2


def test_expected_child_count_never_becomes_negative():
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine.search_node import SearchNode

    root = SearchNode(code="", plan="root", stage="root")
    root.sub_expected_child_count()
    assert root.expected_child_count == 0
    root.add_expected_child_count()
    root.sub_expected_child_count()
    root.sub_expected_child_count()
    assert root.expected_child_count == 0


def test_topk_fully_expanded_selection_propagates_wait(monkeypatch):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from engine import node_selection

    selected = SimpleNamespace(
        id="selected",
        reached_child_limit=lambda *_args, **_kwargs: True,
    )
    agent = SimpleNamespace(
        search_start_time=0.0,
        acfg=SimpleNamespace(time_limit=1.0),
        scfg=SimpleNamespace(
            explore_switch_start=0.0,
            explore_switch_end=0.0,
            min_exploration_weight=0.0,
            topk_early_k=1,
            topk_early_max_per_branch=1,
            topk_late_k=1,
            topk_late_max_per_branch=1,
        ),
    )
    top_k = [{"node": selected, "branch_id": 1, "metric": 0.2, "rank": 1}]
    monkeypatch.setattr(node_selection, "get_top_k_nodes_global", lambda *_args, **_kwargs: top_k)
    monkeypatch.setattr(node_selection, "select_from_top_k_weighted", lambda *_args, **_kwargs: selected)
    monkeypatch.setattr(node_selection, "select", lambda *_args, **_kwargs: None)

    assert node_selection.select_with_soft_switch(agent) is None


def test_protocol_biased_preflight_is_blocked_before_execution(tmp_path):
    import sys

    sys.path.insert(0, str(REPO / "mlevolve"))
    from agents import result_parse_agent
    from engine.search_node import SearchNode

    code = """
val_probas = {"a": a_val_probs, "b": b_val_probs}
best_weights = None
best_ll = 99
for w1 in np.arange(0.1, 0.9, 0.1):
    candidate = w1 * val_probas["a"] + (1 - w1) * val_probas["b"]
    ll = log_loss(y_val, candidate)
    if ll < best_ll:
        best_ll = ll
        best_weights = (w1, 1 - w1)
print("optimized ensemble weights", best_weights)
print("validation log loss", best_ll)
"""
    node = SearchNode(code=code, plan="biased repair", stage="debug")
    agent = SimpleNamespace(
        acfg=SimpleNamespace(check_data_leakage=True),
        cfg=SimpleNamespace(workspace_dir=tmp_path),
        global_memory=None,
        external_skill_memory=None,
    )
    assert result_parse_agent.run_pre_execution_leakage_audit(agent, node) is True
    assert node.leakage_audit["status"] == "protocol_biased"
    assert node.leakage_audit["execution_disposition"] == "block"
    assert node.leakage_audit["rank_eligible"] is False
    assert node.metric.value is None


def test_repair_contract_is_high_priority_in_debug_and_improve_prompts():
    for relative in ("mlevolve/agents/debug_agent.py", "mlevolve/agents/improve_agent.py"):
        source = (REPO / relative).read_text(encoding="utf-8")
        assert "LEAKAGE REPAIR CONTRACT - HIGHEST PRIORITY" in source
        assert "fresh audit" in source


def test_debug_runtime_recovery_targets_shm_without_redesigning_branch():
    from agents.debug_agent import _runtime_recovery_guidance
    from engine.search_node import SearchNode

    parent = SearchNode(
        code="loader = DataLoader(dataset, num_workers=4)",
        plan="novel image pipeline",
        stage="debug",
        draft_role="novel_exploration",
        is_buggy=True,
        analysis="DataLoader workers were killed by a bus error from insufficient shared memory",
    )
    guidance = "\n".join(_runtime_recovery_guidance(parent))

    assert "num_workers=0" in guidance
    assert "supersedes the generic num_workers>=2" in guidance
    assert "Do not redesign or simplify" in guidance


def test_debug_runtime_recovery_preserves_missing_torch_hub_model_family():
    from agents.debug_agent import _runtime_recovery_guidance
    from engine.search_node import SearchNode

    parent = SearchNode(
        code="torch.hub.load('./missing', 'dinov3_vitl16', source='local')",
        plan="novel DINOv3 pipeline",
        stage="draft",
        draft_role="novel_exploration",
        is_buggy=True,
        exc_type="FileNotFoundError",
        analysis="hubconf.py was not found while calling torch.hub.load",
    )
    guidance = "\n".join(_runtime_recovery_guidance(parent))

    assert "Do not replace the architecture or model family" in guidance
    assert "online GitHub source" in guidance
    host_guidance = "\n".join(
        _runtime_recovery_guidance(parent, allow_remote_assets=False)
    )
    assert "online GitHub source" not in host_guidance


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
