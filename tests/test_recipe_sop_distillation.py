from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DISTILLER_PATH = REPO / "paper-skills" / "distillation" / "distill_recipe_sops.py"
ARTIFACT_ROOT = (
    REPO
    / "experiments"
    / "end2end_memory_systems_20260804"
    / "recipe_distillation_v2"
)
INCREMENTAL_ROOT = ARTIFACT_ROOT.parent / "recipe_distillation_v1"


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


distiller = _module(DISTILLER_PATH, "distill_recipe_sops_test_module")


def _load(name: str):
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def test_strict_recipe_admission_rejects_nonclean_or_invalid_nodes() -> None:
    clean = {
        "type": "RunNode",
        "is_buggy": False,
        "is_valid": True,
        "metric": 0.1,
        "leakage_audit": {
            "status": "clean",
            "memory_disposition": "positive_eligible",
            "paper_grade_eligible": True,
            "rank_eligible": True,
        },
    }
    assert distiller.strict_recipe_eligible(clean)
    for field, value in (
        ("is_buggy", True),
        ("is_valid", False),
        ("metric", None),
    ):
        rejected = dict(clean)
        rejected[field] = value
        assert not distiller.strict_recipe_eligible(rejected)
    for field, value in (
        ("status", "warning"),
        ("memory_disposition", "warning_only"),
        ("paper_grade_eligible", False),
        ("rank_eligible", False),
    ):
        rejected = dict(clean)
        rejected["leakage_audit"] = dict(clean["leakage_audit"])
        rejected["leakage_audit"][field] = value
        assert not distiller.strict_recipe_eligible(rejected)


def test_l3_admission_requires_real_clean_repair_and_rejects_infrastructure() -> None:
    parent = {
        "id": "parent",
        "type": "RunNode",
        "task": "leaf-classification",
        "is_buggy": True,
        "code_sha256": "1" * 64,
        "analysis": "RuntimeError: indices should be on CPU or on the same device as the indexed tensor",
    }
    child = {
        "id": "child",
        "type": "RunNode",
        "task": "leaf-classification",
        "is_buggy": False,
        "is_valid": True,
        "metric": 0.1,
        "code_sha256": "2" * 64,
        "plan": "Keep the index tensor on CPU before indexing the CPU feature matrix.",
        "leakage_audit": {
            "status": "clean",
            "memory_disposition": "positive_eligible",
            "paper_grade_eligible": True,
            "rank_eligible": True,
        },
    }
    transition = {
        "id": "transition",
        "type": "Transition",
        "task": "leaf-classification",
        "run_id": "run-1",
        "parent_node_id": "parent",
        "child_node_id": "child",
        "stage_pair": "draft->debug",
        "outcome": "debug_fixed",
        "parent_buggy": True,
        "child_buggy": False,
        "text": "Move the index to the same device before indexing.",
    }
    eligible, reason = distiller.repair_transition_eligibility(
        transition,
        nodes_by_id={"parent": parent, "child": child},
        allowed_tasks={"leaf-classification"},
    )
    assert eligible is True
    assert reason == "one_clean_failure_repair_success_transition"

    infrastructure_parent = dict(parent)
    infrastructure_parent["analysis"] = (
        "Permission denied in the node torch hub cache directory"
    )
    eligible, reason = distiller.repair_transition_eligibility(
        transition,
        nodes_by_id={"parent": infrastructure_parent, "child": child},
        allowed_tasks={"leaf-classification"},
    )
    assert eligible is False
    assert reason == "infrastructure_failure"


def test_frozen_recipe_artifact_has_complete_three_layer_memory() -> None:
    report = _load("distillation_report.json")
    bundle = _load("recipe_sops.json")
    assert report["abstraction_counts"] == {
        "L1_recipe": 28,
        "L2_tactic": 26,
        "L3_repair": 35,
    }
    assert report["complete_recipe_ratio"] >= 0.40
    assert report["repair_trivia_in_l1_l2_count"] == 0
    assert report["accepted_clean_repair_l3_count"] == 35
    assert report["l3_evidence_tiering_enabled"] is False
    assert all(report["quality_gates"].values())
    assert len(bundle["nodes"]) == report["node_count"] == 89
    l3 = [
        node for node in bundle["nodes"]
        if node["abstraction_level"] == "L3_repair"
    ]
    assert {node["evidence_status"] for node in l3} == {
        "accepted_clean_repair"
    }
    assert {node["confidence_prior"] for node in l3} == {0.60}
    assert not any(
        distiller.REPAIR_TRIVIA.search(node["title"]) for node in bundle["nodes"]
        if node["abstraction_level"] in {"L1_recipe", "L2_tactic"}
    )


def test_all_recipes_are_complete_and_tactics_bind_task_local_families() -> None:
    bundle = _load("recipe_sops.json")
    recipes = [
        node for node in bundle["nodes"] if node["abstraction_level"] == "L1_recipe"
    ]
    tactics = [
        node for node in bundle["nodes"] if node["abstraction_level"] == "L2_tactic"
    ]
    for recipe in recipes:
        assert recipe["recipe_complete"] is True
        assert set(recipe["pipeline"]) == set(distiller.RECIPE_SECTIONS)
        assert all(recipe["pipeline"][section] for section in distiller.RECIPE_SECTIONS)
        assert recipe["decision_stages"] == ["draft"]
    families_by_task = {}
    for recipe in recipes:
        families_by_task.setdefault(recipe["task_id"], set()).add(recipe["method_family"])
    for tactic in tactics:
        assert tactic["decision_stages"] == ["model_design"]
        assert set(tactic["parent_method_families"]) <= families_by_task[tactic["task_id"]]


def test_every_sop_source_is_in_same_task_selected_evidence() -> None:
    evidence = _load("evidence_manifest.json")
    bundle = _load("recipe_sops.json")
    allowed = {
        task: {row["node_id"] for row in rows}
        for task, rows in evidence["selected_evidence"].items()
    }
    for node in bundle["nodes"]:
        if node["abstraction_level"] == "L3_repair":
            continue
        assert node["source_node_ids"]
        assert set(node["source_node_ids"]) <= allowed[node["task_id"]]


def test_every_l3_has_task_bound_clean_transition_evidence() -> None:
    evidence = _load("evidence_manifest.json")
    bundle = _load("recipe_sops.json")
    allowed = {
        task: {row["transition_id"]: row for row in rows}
        for task, rows in evidence["selected_repair_evidence"].items()
    }
    repairs = [
        node for node in bundle["nodes"] if node["abstraction_level"] == "L3_repair"
    ]
    assert len(repairs) == 35
    for node in repairs:
        rows = allowed[node["task_id"]]
        assert node["task_type"] in {"vision", "multimodal", "tabular", "nlp"}
        assert node["failure_signature"]["id"]
        assert node["repair_action"]["steps"]
        assert node["runtime_stage"] in distiller.L3_RUNTIME_STAGES
        assert node["supporting_transition_ids"]
        assert set(node["supporting_transition_ids"]) <= set(rows)
        assert node["failure_node_ids"] == list(
            dict.fromkeys(rows[value]["failure_node_id"] for value in node["supporting_transition_ids"])
        )
        assert node["successful_node_ids"] == list(
            dict.fromkeys(rows[value]["successful_node_id"] for value in node["supporting_transition_ids"])
        )
        assert node["infrastructure_failure"] is False
        assert node["one_off_code_failure"] is False


def test_incremental_smoke_admission_keeps_failures_out_of_recipe_pool() -> None:
    incremental = json.loads(
        (INCREMENTAL_ROOT / "incremental_evidence.json").read_text(encoding="utf-8")
    )
    assert incremental["candidate_policy"] == "selected_terminal_program_only"
    assert incremental["record_count"] == 9
    assert incremental["audit_status_counts"] == {
        "blocked": 1,
        "clean": 4,
        "protocol_biased": 2,
        "warning": 2,
    }
    admitted = [
        row
        for row in incremental["records"]
        if distiller.strict_recipe_eligible(
            {
                "type": "RunNode",
                "is_buggy": row["is_buggy"],
                "is_valid": row["is_valid"],
                "metric": row["metric"],
                "leakage_audit": row["leakage_audit"],
            }
        )
    ]
    assert {row["system_id"] for row in admitted} == {
        "flat_retrieval",
        "rcr_router_style_port",
        "static_hybrid",
        "dynamic_hybrid",
    }


def test_bundle_and_report_hashes_are_reproducible() -> None:
    bundle = _load("recipe_sops.json")
    report = _load("distillation_report.json")
    assert distiller.payload_hash(bundle, "bundle_sha256") == bundle["bundle_sha256"]
    assert distiller.payload_hash(report, "report_sha256") == report["report_sha256"]
    archive = INCREMENTAL_ROOT / "post_freeze_logs" / "end2end-recipe-records.tgz"
    incremental = json.loads(
        (INCREMENTAL_ROOT / "incremental_evidence.json").read_text(encoding="utf-8")
    )
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == incremental[
        "input_archive_sha256"
    ]
