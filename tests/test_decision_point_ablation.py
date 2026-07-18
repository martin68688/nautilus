import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ABLATION_PATH = REPO / "paper-skills" / "eval_skill_memory" / "evaluate_decision_point_ablation.py"
REPORT_PATH = REPO / "paper-skills" / "eval_skill_memory" / "reports" / "decision_point_ablation_v1.json"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ablation = _load(ABLATION_PATH, "decision_point_ablation")


def test_stage_bonus_contract_is_explicit():
    assert ablation._stage_bonus("draft") == {"draft": 0.08}
    assert ablation._stage_bonus("improve") == {"improve": 0.10, "evolution": 0.05}
    assert ablation._stage_bonus("debug") == {"debug": 0.10, "improve": 0.04}


def test_generated_ablation_keeps_production_and_projection_claims_separate():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    facts = report["implementation_facts"]
    assert facts["legacy_reimplementation_matches_rankings"] is True
    assert facts["legacy_stage_gateway_uses_tree"] is False
    assert facts["legacy_stage_gateway_uses_geometry"] is False
    assert facts["legacy_stage_gateway_uses_rrf"] is False
    assert facts["production_stage_hybrid_sop_uses_tree"] is True
    assert facts["production_stage_hybrid_sop_uses_geometry"] is True
    assert facts["production_stage_hybrid_sop_uses_rrf"] is True
    assert facts["production_stage_hybrid_sop_uses_stage_taxonomy"] is True
    assert facts["production_stage_hybrid_sop_uses_task_identity"] is True
    assert facts["production_stage_hybrid_sop_enforces_clean_gateway"] is True
    assert facts["tree_rrf_methods_are_sop_space_benchmark_projections_not_production_hybrid_pack"] is True


def test_ablation_records_component_effects_without_overclaiming_stage_boost():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    gate = report["paired_comparisons"]["clean_gateway_gate_effect"]
    stage = report["paired_comparisons"]["debug_stage_boost_effect"]
    changes = report["ranking_changes"]["debug_stage_boost_effect"]
    assert gate["delta"] > 0.0
    assert gate["holm_adjusted_p"] < 0.05
    assert stage["delta"] == 0.0
    assert changes["changed_ranking_count"] == 0
    assert changes["improved_query_count"] == 0
    assert changes["degraded_query_count"] == 0
