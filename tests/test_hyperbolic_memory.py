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


def test_builder_flat_twin_identity_and_quality_report(tmp_path):
    input_path = tmp_path / "graph.json"
    out_dir = tmp_path / "out"
    synthetic_graph(input_path)
    report = builder.build(input_path, out_dir, dims=3, require_clean_provenance=True)

    idx = np.load(out_dir / "hyper_index.npz")
    assert np.array_equal(idx["flat_twin"], idx["poincare"])
    assert report["validation"]["flat_twin_same_coordinates_as_poincare"] is True
    assert report["provenance"]["paper_grade"] is True
    assert (out_dir / "hyper_text_model.joblib").exists()
    assert (out_dir / "coordinate_quality_report.json").exists()
    assert report["coordinates"]["quality_report"]["status"] == "coordinate_quality_null"


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
    assert layer_p._poincare_coords.keys() == layer_f._flat_twin_coords.keys()
    first = next(iter(layer_p._poincare_coords))
    assert np.array_equal(layer_p._poincare_coords[first], layer_f._flat_twin_coords[first])

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
    assert result_p["scoring_mode"] == "poincare"
    assert result_f["scoring_mode"] == "flat_twin"
    assert result_p["sops"] and result_f["sops"]


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
