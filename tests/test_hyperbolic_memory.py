import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module("build_hyperbolic_memory", REPO / "paper-skills" / "hyper_memory" / "build_hyperbolic_memory.py")
ablation = load_module("evaluate_hyperbolic_ablation", REPO / "paper-skills" / "hyper_memory" / "evaluate_hyperbolic_ablation.py")
certifier = load_module("certify_skillgraph_provenance", REPO / "paper-skills" / "eval_skill_memory" / "certify_skillgraph_provenance.py")
validator = load_module("validate_hyperbolic_benchmark", REPO / "paper-skills" / "eval_skill_memory" / "validate_hyperbolic_benchmark.py")
benchmark_builder = load_module("build_hyperbolic_benchmark", REPO / "paper-skills" / "eval_skill_memory" / "build_hyperbolic_benchmark.py")
retrieval_runner = load_module("run_hyperbolic_retrieval_benchmark", REPO / "paper-skills" / "eval_skill_memory" / "run_hyperbolic_retrieval_benchmark.py")
radius_trainer = load_module("train_radius_band_predictor", REPO / "paper-skills" / "eval_skill_memory" / "train_radius_band_predictor.py")
sys.path.insert(0, str(REPO / "mlevolve"))
from agents.memory.external_skill_memory import ExternalSkillMemoryLayer, euclidean_distance, poincare_distance  # noqa: E402


def synthetic_graph(path: Path) -> None:
    nodes = [
        {
            "id": "sg_a",
            "title": "Use label smoothing for small text transformer",
            "principle": "Set label smoothing to improve calibration.",
            "condition": "small text classification dataset with overconfident predictions",
            "category": "spooky-author-identification",
            "scope": "task_specific",
            "level": 1,
            "n_use": 4,
            "n_succ": 3,
            "p_hat": 0.75,
            "source_branches": ["run_clean_a:B1"],
        },
        {
            "id": "sg_b",
            "title": "Use gradient accumulation for memory limited transformer",
            "principle": "Use accumulation to fit larger effective batches.",
            "condition": "CUDA memory pressure during transformer training",
            "category": "spooky-author-identification",
            "scope": "task_specific",
            "level": 2,
            "n_use": 3,
            "n_succ": 2,
            "p_hat": 0.67,
            "source_branches": ["run_clean_b:B2"],
        },
        {
            "id": "sg_c",
            "title": "Fit vectorizers only on training folds",
            "principle": "Fit preprocessing on train fold only to avoid leakage.",
            "condition": "cross validation text preprocessing",
            "category": "general",
            "scope": "universal_general",
            "level": 0,
            "n_use": 5,
            "n_succ": 5,
            "p_hat": 1.0,
            "source_branches": ["run_clean_a:B3", "run_clean_b:B3"],
        },
        {
            "id": "sg_d",
            "title": "Avoid loading unavailable external embeddings",
            "principle": "Prefer bundled embeddings when external files are absent.",
            "condition": "missing GloVe or pretrained embedding files",
            "category": "spooky-author-identification",
            "scope": "task_specific",
            "level": 3,
            "n_use": 1,
            "n_succ": 0,
            "p_hat": 0.0,
            "source_branches": ["run_clean_c:B4"],
        },
    ]
    graph = {
        "meta": {
            "schema": "skillgraph-static-v1",
            "source_runs": ["run_clean_a", "run_clean_b", "run_clean_c"],
            "allowlist": ["run_clean_a", "run_clean_b", "run_clean_c"],
            "leak_verified": True,
        },
        "nodes": nodes,
        "edges": [{"src": "sg_c", "dst": "sg_a", "kind": "enhance", "weight": 0.2}],
    }
    path.write_text(json.dumps(graph), encoding="utf-8")


def edge_hyper_graph(path: Path, n_edge: int = 24) -> None:
    nodes = []
    for idx in range(n_edge):
        nodes.append({
            "id": f"edge_{idx:02d}",
            "type": "SOP",
            "title": f"Narrow interface repair {idx}",
            "principle": "Repair a low-frequency generated-code interface failure.",
            "condition": f"api shape path version failure family {idx}",
            "category": "spooky-author-identification",
            "scope": "task_specific",
            "radius_band": "edge",
            "radius": 0.82,
            "n_use": 1,
            "p_hat": 0.0,
            "source_branches": [["run_clean_edge", str(idx)]],
        })
    graph = {
        "meta": {
            "schema": "hyperbolic-sop-memory-v2",
            "paper_grade": True,
            "source_runs": ["run_clean_edge"],
            "allowlist": ["run_clean_edge"],
            "leak_verified": True,
        },
        "nodes": nodes,
        "edges": [],
    }
    path.write_text(json.dumps(graph), encoding="utf-8")


def test_builder_flat_twin_identity_and_quality_report(tmp_path):
    input_path = tmp_path / "graph.json"
    out_dir = tmp_path / "out"
    synthetic_graph(input_path)
    report = builder.build(input_path, out_dir, dims=3, require_clean_provenance=True)

    idx = np.load(out_dir / "hyper_index.npz")
    assert np.array_equal(idx["flat_twin"], idx["poincare"])
    assert "euclidean" in idx.files
    assert "specificity_score" in idx.files
    assert "reliability_score" in idx.files
    assert "support_score" in idx.files
    assert idx["euclidean"].shape == idx["poincare"].shape
    assert not np.array_equal(idx["euclidean"], idx["poincare"])
    assert np.allclose(np.linalg.norm(idx["euclidean"], axis=1), 1.0, atol=1e-5)
    assert report["validation"]["flat_twin_same_coordinates_as_poincare"] is True
    assert report["validation"]["euclidean_independent_coordinates"] is True
    assert report["validation"]["euclidean_unit_norm_coordinates"] is True
    assert report["provenance"]["paper_grade"] is True
    assert report["validation"]["radius_reliability_decoupled"] is True
    assert report["coordinates"]["radius"]["model"] == "specificity_radius_v2"
    assert (out_dir / "hyper_text_model.joblib").exists()
    assert (out_dir / "coordinate_quality_report.json").exists()
    assert report["coordinates"]["quality_report"]["status"] == "coordinate_quality_null"


def test_specificity_radius_is_decoupled_from_reliability():
    base = {
        "id": "a",
        "title": "Use estimator parameter for calibration API",
        "principle": "Replace deprecated base_estimator with estimator.",
        "condition": "CalibratedClassifierCV raises unexpected keyword argument base_estimator",
        "category": "leaf-classification",
        "scope": "task_specific",
        "level": 2,
        "source_branches": ["run:B1"],
    }
    low = {**base, "id": "low", "n_use": 1, "n_succ": 0, "p_hat": 0.0}
    high = {**base, "id": "high", "n_use": 20, "n_succ": 20, "p_hat": 1.0}
    radii, meta, features = builder.compute_radii([low, high])
    assert meta["radius_reliability_decoupled"] is True
    assert radii[0] == pytest.approx(radii[1])
    assert features[0]["reliability_score"] < features[1]["reliability_score"]


def test_builder_requires_clean_provenance_fails_closed(tmp_path):
    input_path = tmp_path / "graph.json"
    synthetic_graph(input_path)
    graph = json.loads(input_path.read_text(encoding="utf-8"))
    graph["meta"].pop("source_runs")
    graph["nodes"][0].pop("source_branches")
    input_path.write_text(json.dumps(graph), encoding="utf-8")
    with pytest.raises(ValueError, match="clean provenance"):
        builder.build(input_path, tmp_path / "out", dims=3, require_clean_provenance=True)


def test_certifier_attaches_source_provenance(tmp_path):
    graph_path = tmp_path / "compact.json"
    source_path = tmp_path / "source_nodes.json"
    allowlist_path = tmp_path / "allowlist.json"
    output_path = tmp_path / "certified.json"
    graph_path.write_text(json.dumps({
        "meta": {"schema": "skillgraph-static-v1"},
        "nodes": [{"id": "sg_a", "title": "A", "principle": "Do A", "condition": "when A", "category": "task"}],
        "edges": [],
    }), encoding="utf-8")
    source_path.write_text(json.dumps({
        "nodes": [{
            "id": "sg_a",
            "source_branches": [["run_good", "1"]],
            "evidence_turns": ["B1.T1"],
        }]
    }), encoding="utf-8")
    allowlist_path.write_text(json.dumps({
        "entries": [{"run_id": "run_good", "task": "task", "path": "runs/run_good", "audit_status": "clean", "allowed": True, "notes": ""}]
    }), encoding="utf-8")
    report = certifier.certify(
        graph_path=graph_path,
        source_nodes_path=source_path,
        allowlist_path=allowlist_path,
        output_path=output_path,
    )
    certified = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "clean_certified"
    assert certified["meta"]["leak_verified"] is True
    assert certified["nodes"][0]["source_branches"] == [["run_good", "1"]]


def test_geometry_runtime_same_coordinates_two_distances(tmp_path):
    input_path = tmp_path / "graph.json"
    out_dir = tmp_path / "out"
    synthetic_graph(input_path)
    builder.build(input_path, out_dir, dims=8, require_clean_provenance=True)

    layer_p = ExternalSkillMemoryLayer(
        graph_path=str(out_dir / "hyper_graph.json"),
        index_path=str(out_dir / "hyper_index.npz"),
        text_model_path=str(out_dir / "hyper_text_model.joblib"),
        mode="agentic_hyperbolic",
        scoring_mode="poincare",
        geometry_distance_norm="minmax",
        geometry_query_radius_quantile=0.65,
        enable_agentic=True,
        cfg=None,
    )
    layer_f = ExternalSkillMemoryLayer(
        graph_path=str(out_dir / "hyper_graph.json"),
        index_path=str(out_dir / "hyper_index.npz"),
        text_model_path=str(out_dir / "hyper_text_model.joblib"),
        mode="flat_twin_agentic",
        scoring_mode="flat_twin",
        enable_agentic=True,
        cfg=None,
    )
    layer_e = ExternalSkillMemoryLayer(
        graph_path=str(out_dir / "hyper_graph.json"),
        index_path=str(out_dir / "hyper_index.npz"),
        text_model_path=str(out_dir / "hyper_text_model.joblib"),
        mode="agentic_euclidean",
        scoring_mode="euclidean",
        enable_agentic=True,
        cfg=None,
    )
    assert layer_p._poincare_coords.keys() == layer_f._flat_twin_coords.keys()
    assert layer_p._poincare_coords.keys() == layer_e._euclidean_coords.keys()
    first = next(iter(layer_p._poincare_coords))
    assert np.array_equal(layer_p._poincare_coords[first], layer_f._flat_twin_coords[first])
    assert not np.array_equal(layer_p._poincare_coords[first], layer_e._euclidean_coords[first])

    result_p = layer_p.navigate(
        task_type="spooky-author-identification",
        query_text="small text classification overconfident transformer",
        top_k=3,
    )
    result_f = layer_f.navigate(
        task_type="spooky-author-identification",
        query_text="small text classification overconfident transformer",
        top_k=3,
    )
    result_e = layer_e.navigate(
        task_type="spooky-author-identification",
        query_text="small text classification overconfident transformer",
        top_k=3,
    )
    assert result_p["scoring_mode"] == "poincare"
    assert result_f["scoring_mode"] == "flat_twin"
    assert result_e["scoring_mode"] == "euclidean"
    assert result_p["sops"] and result_f["sops"] and result_e["sops"]
    assert layer_p.geometry_query_radius_quantile == pytest.approx(0.65)
    assert result_p["query_radius_distribution"]


def test_query_radius_predictor_edge_biased_for_sparse_debug_context(tmp_path):
    input_path = tmp_path / "graph.json"
    out_dir = tmp_path / "out"
    synthetic_graph(input_path)
    builder.build(input_path, out_dir, dims=8, require_clean_provenance=True)

    layer = ExternalSkillMemoryLayer(
        graph_path=str(out_dir / "hyper_graph.json"),
        index_path=str(out_dir / "hyper_index.npz"),
        text_model_path=str(out_dir / "hyper_text_model.joblib"),
        mode="agentic_hyperbolic",
        scoring_mode="poincare",
        geometry_query_radius_mode="predicted_distribution",
        enable_agentic=True,
        cfg=None,
    )
    dist = layer._predict_query_radius_distribution(
        "Query kind: minimal_context\nStage: debug\nTraceback ValueError shape mismatch in checkpoint path"
    )
    weights = {str(x["band"]): float(x["weight"]) for x in dist}
    assert weights["edge"] > weights.get("middle", 0.0)

    hinted = layer.navigate(
        task_type="spooky-author-identification",
        query_text="missing pretrained embedding file path",
        radius_band="edge",
        top_k=3,
    )
    assert {s["radius_band"] for s in hinted["sops"]} <= {"edge"}


def test_geometry_mode_missing_index_fails_closed(tmp_path):
    input_path = tmp_path / "graph.json"
    out_dir = tmp_path / "out"
    synthetic_graph(input_path)
    builder.build(input_path, out_dir, dims=3, require_clean_provenance=True)
    with pytest.raises(FileNotFoundError):
        ExternalSkillMemoryLayer(
            graph_path=str(out_dir / "hyper_graph.json"),
            index_path=str(out_dir / "missing.npz"),
            text_model_path=str(out_dir / "hyper_text_model.joblib"),
            mode="agentic_hyperbolic",
            scoring_mode="poincare",
            enable_agentic=True,
            cfg=None,
        )


def test_distance_functions_share_coordinates():
    u = np.asarray([0.2, 0.1, 0.0], dtype=np.float32)
    v = np.asarray([0.5, -0.1, 0.2], dtype=np.float32)
    assert poincare_distance(u, u) == pytest.approx(0.0)
    assert poincare_distance(u, v) == pytest.approx(poincare_distance(v, u))
    assert euclidean_distance(u, v) == pytest.approx(euclidean_distance(v, u))
    assert poincare_distance(u, v) > euclidean_distance(u, v)


def test_paired_bootstrap_gate_requires_p_value_and_precision():
    rows = [
        {"poincare": {"rare_recall_at_5": 0.7, "condition_precision": 0.8}, "flat_twin": {"rare_recall_at_5": 0.5, "condition_precision": 0.8}},
        {"poincare": {"rare_recall_at_5": 0.6, "condition_precision": 0.7}, "flat_twin": {"rare_recall_at_5": 0.4, "condition_precision": 0.7}},
        {"poincare": {"rare_recall_at_5": 0.8, "condition_precision": 0.9}, "flat_twin": {"rare_recall_at_5": 0.6, "condition_precision": 0.9}},
        {"poincare": {"rare_recall_at_5": 0.7, "condition_precision": 0.8}, "flat_twin": {"rare_recall_at_5": 0.5, "condition_precision": 0.8}},
    ]
    report = ablation.evaluate(rows, n_resamples=2000, seed=1)
    assert report["rare_recall_at_5"]["observed_mean_diff"] >= 0.05
    assert report["rare_recall_at_5"]["p_value"] < 0.05
    assert report["passed"] is True


def test_runner_evaluator_rare_recall_ignores_nonrare_queries():
    graph = {
        "meta": {"paper_grade": True},
        "nodes": [
            {"id": "sg_rare", "type": "SOP", "title": "Rare", "condition": "rare condition"},
            {"id": "sg_common", "type": "SOP", "title": "Common", "condition": "common condition"},
        ],
    }
    gold = {
        "q_rare": [{"sop_id": "sg_rare", "relevance": "required", "condition_match": True, "is_rare": True}],
        "q_common": [{"sop_id": "sg_common", "relevance": "required", "condition_match": True, "is_rare": False}],
    }
    rows = [
        {"query_id": "q_rare", "query_kind": "rare_condition", "system": "agentic_poincare", "selected_sops": ["sg_rare"], "navigation_trace": ["a"], "risk_warnings": []},
        {"query_id": "q_rare", "query_kind": "rare_condition", "system": "agentic_flat_twin", "selected_sops": [], "navigation_trace": ["a"], "risk_warnings": []},
        {"query_id": "q_rare", "query_kind": "rare_condition", "system": "agentic_euclidean", "selected_sops": [], "navigation_trace": ["a"], "risk_warnings": []},
        {"query_id": "q_common", "query_kind": "method_set", "system": "agentic_poincare", "selected_sops": [], "navigation_trace": ["a"], "risk_warnings": []},
        {"query_id": "q_common", "query_kind": "method_set", "system": "agentic_flat_twin", "selected_sops": ["sg_common"], "navigation_trace": ["a"], "risk_warnings": []},
        {"query_id": "q_common", "query_kind": "method_set", "system": "agentic_euclidean", "selected_sops": ["sg_common"], "navigation_trace": ["a"], "risk_warnings": []},
    ]
    report = ablation.evaluate_runner_results(result_rows=rows, gold_by_query=gold, graph=graph, n_resamples=200, seed=1)
    assert report["systems"]["agentic_poincare"]["rare_recall_at_5"] == pytest.approx(1.0)
    assert report["systems"]["agentic_flat_twin"]["rare_recall_at_5"] == pytest.approx(0.0)
    assert "rare_recall_at_5" not in report["systems_by_query_kind"]["agentic_poincare"]["method_set"]
    euclidean = report["comparisons"]["poincare_vs_euclidean_independent_coordinates"]
    assert euclidean["rare_recall_at_5"]["observed_mean_diff"] == pytest.approx(1.0)


def test_benchmark_validator_accepts_certified_gold(tmp_path):
    input_path = tmp_path / "graph.json"
    out_dir = tmp_path / "out"
    synthetic_graph(input_path)
    builder.build(input_path, out_dir, dims=8, require_clean_provenance=True)
    benchmark = tmp_path / "bench.jsonl"
    gold = tmp_path / "gold.jsonl"
    allowlist = tmp_path / "allowlist.json"
    benchmark.write_text(json.dumps({
        "query_id": "q1",
        "task_type": "spooky-author-identification",
        "stage": "debug",
        "context": "small text classification overconfident transformer",
        "condition": ["small text classification dataset with overconfident predictions"],
        "failure_mode": ["poor calibration"],
        "source_trace": "",
        "query_kind": "rare_condition",
        "split": "dev",
        "radius_band_hint": "edge,middle",
        "title_leakage_level": "low",
        "gold_title_hidden": True,
        "distractor_sops": ["sg_b"],
        "distractor_count": 1,
    }) + "\n", encoding="utf-8")
    gold.write_text(json.dumps({
        "query_id": "q1",
        "gold_sops": [{
            "sop_id": "sg_a",
            "relevance": "required",
            "condition_match": True,
            "is_rare": True,
            "rarity_count": 1,
            "rationale": "fixture",
        }],
    }) + "\n", encoding="utf-8")
    allowlist.write_text(json.dumps({
        "entries": [
            {"run_id": "run_clean_a", "task": "task", "path": "", "audit_status": "clean", "allowed": True, "notes": ""},
            {"run_id": "run_clean_b", "task": "task", "path": "", "audit_status": "clean", "allowed": True, "notes": ""},
            {"run_id": "run_clean_c", "task": "task", "path": "", "audit_status": "clean", "allowed": True, "notes": ""},
        ]
    }), encoding="utf-8")
    report = validator.validate(
        graph_path=out_dir / "hyper_graph.json",
        benchmark_path=benchmark,
        gold_path=gold,
        allowlist_path=allowlist,
        require_certified_graph=True,
    )
    assert report["passed"] is True


def test_edge_benchmark_profile_keeps_edge_gold_and_grouped_split(tmp_path):
    graph = tmp_path / "edge_graph.json"
    bench = tmp_path / "edge_bench.jsonl"
    gold = tmp_path / "edge_gold.jsonl"
    allowlist = tmp_path / "allowlist.json"
    edge_hyper_graph(graph, n_edge=24)
    build_report = benchmark_builder.build(
        graph,
        bench,
        gold,
        profile="edge",
        edge_variants_per_sop=2,
    )
    assert build_report["profile"] == "edge"
    assert build_report["queries"] == 48
    rows = [json.loads(line) for line in bench.read_text(encoding="utf-8").splitlines() if line.strip()]
    gold_rows = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {row["gold_radius_band"] for row in rows} == {"edge"}
    assert all(row["query_kind"].startswith("edge_") for row in rows)
    assert all(item["gold_sops"][0]["edge_gold"] is True for item in gold_rows)
    splits_by_sop = {}
    for item in gold_rows:
        sid = item["gold_sops"][0]["sop_id"]
        qid = item["query_id"]
        split = next(row["split"] for row in rows if row["query_id"] == qid)
        splits_by_sop.setdefault(sid, set()).add(split)
    assert all(len(splits) == 1 for splits in splits_by_sop.values())
    allowlist.write_text(json.dumps({
        "entries": [
            {"run_id": "run_clean_edge", "task": "task", "path": "", "audit_status": "clean", "allowed": True, "notes": ""}
        ]
    }), encoding="utf-8")
    validation = validator.validate(
        graph_path=graph,
        benchmark_path=bench,
        gold_path=gold,
        allowlist_path=allowlist,
        require_certified_graph=True,
        edge_profile=True,
        baseline_validation_path=None,
    )
    assert validation["passed"] is True
    assert validation["edge_profile"] is True


def test_runner_predicted_only_rejects_gold_radius_hint(tmp_path):
    input_path = tmp_path / "graph.json"
    out_dir = tmp_path / "out"
    synthetic_graph(input_path)
    builder.build(input_path, out_dir, dims=8, require_clean_provenance=True)
    bench = tmp_path / "bench.jsonl"
    output = tmp_path / "results.jsonl"
    bench.write_text(json.dumps({
        "query_id": "q1",
        "task_type": "spooky-author-identification",
        "stage": "debug",
        "context": "Traceback ValueError shape mismatch in checkpoint path",
        "condition": ["shape mismatch"],
        "failure_mode": ["api or checkpoint mismatch"],
        "query_kind": "edge_shape_path",
        "split": "test",
        "radius_band_hint": "edge",
        "query_style": "edge_shape_path",
        "query_specificity": "low",
        "distractor_sops": [],
    }) + "\n", encoding="utf-8")
    retrieval_runner.run(
        graph_path=out_dir / "hyper_graph.json",
        index_path=out_dir / "hyper_index.npz",
        text_model_path=out_dir / "hyper_text_model.joblib",
        benchmark_path=bench,
        output_path=output,
        systems=["agentic_poincare"],
        top_k=3,
        radius_hint_mode="predicted_only",
    )
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["original_radius_band_hint"] == "edge"
    assert row["used_radius_band_hint"] == ""
    assert row["radius_hint_rejected"] is True
    assert row["query_radius_distribution"]
    assert {item["source"] for item in row["query_radius_distribution"]} == {"deterministic_query_radius_v1"}


def test_radius_band_predictor_trains_on_dev_only(tmp_path):
    graph = tmp_path / "edge_graph.json"
    bench = tmp_path / "bench.jsonl"
    gold = tmp_path / "gold.jsonl"
    output = tmp_path / "radius.joblib"
    edge_hyper_graph(graph, n_edge=24)
    benchmark_builder.build(graph, bench, gold, profile="edge", edge_variants_per_sop=1)
    report = radius_trainer.train(
        graph_path=graph,
        benchmark_path=bench,
        gold_path=gold,
        output_path=output,
    )
    assert report["status"] == "trained"
    artifact = retrieval_runner.load_radius_predictor(output)
    assert artifact["train_split"] == "dev"
    assert all("test" not in qid for qid in artifact["train_query_ids"])


def test_strict_sentence_embedding_unavailable_fails_closed(tmp_path):
    input_path = tmp_path / "graph.json"
    synthetic_graph(input_path)
    with pytest.raises(RuntimeError, match="embedding_backend_unavailable"):
        builder.build(
            input_path,
            tmp_path / "out",
            dims=8,
            require_clean_provenance=True,
            direction_backend="sentence_embedding",
            embedding_model="__missing_local_sentence_model__",
            allow_embedding_fallback=False,
        )


def test_runner_evaluator_reports_not_claim_grade_when_uncertified():
    graph = {
        "meta": {"paper_grade": False},
        "nodes": [
            {
                "id": "sg_a",
                "type": "SOP",
                "title": "Use label smoothing",
                "condition": "small data overconfident predictions",
                "source_branches": [["run_clean", "1"]],
                "evidence_turns": ["B1.T1"],
            }
        ],
    }
    gold = {"q1": [{"sop_id": "sg_a", "relevance": "required", "condition_match": True, "is_rare": True}]}
    rows = [
        {"query_id": "q1", "system": "agentic_poincare", "selected_sops": ["sg_a"], "navigation_trace": ["a"], "risk_warnings": []},
        {"query_id": "q1", "system": "agentic_flat_twin", "selected_sops": [], "navigation_trace": ["a"], "risk_warnings": []},
    ]
    report = ablation.evaluate_runner_results(result_rows=rows, gold_by_query=gold, graph=graph, n_resamples=200, seed=1)
    assert report["status"] == "not_claim_grade"
    assert report["passed"] is False
    assert "paper_grade_provenance" in report["claim_blockers"]
