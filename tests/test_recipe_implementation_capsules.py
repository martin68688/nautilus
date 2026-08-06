import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))


def _sha(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path: Path):
    before = "def score(x):\n    return x / 0\n"
    after = "def score(x):\n    return x / max(1, x)\n"
    parent_id = "run::leaf::node::parent"
    child_id = "run::leaf::node::child"
    transition_id = "run::leaf::transition::parent::child"
    evidence = {
        "schema": "mlevolve_recipe_distillation_evidence_v1",
        "selected_evidence": {
            "leaf-classification": [
                {"node_id": child_id, "code_sha256": _sha(after)}
            ]
        },
        "selected_repair_evidence": {
            "leaf-classification": [
                {
                    "transition_id": transition_id,
                    "failure_node_id": parent_id,
                    "successful_node_id": child_id,
                    "failure_node_code_sha256": _sha(before),
                    "successful_node_code_sha256": _sha(after),
                }
            ]
        },
    }
    evidence_path = tmp_path / "evidence.json"
    _write(evidence_path, evidence)
    _write(
        tmp_path / "runs" / "leaf" / "logs" / "journal.json",
        {
            "nodes": [
                {"id": "parent", "code": before},
                {"id": "child", "code": after},
            ]
        },
    )
    return evidence_path, before, after, parent_id, child_id, transition_id


def test_builder_recovers_full_code_for_every_selected_node_and_repair(tmp_path):
    from experiments.end2end_memory_systems_20260804.build_recipe_implementation_capsules import (
        build,
    )

    evidence_path, before, after, parent_id, child_id, transition_id = _fixture(
        tmp_path
    )
    payload = build(evidence_path, [tmp_path / "runs"])
    assert payload["required_node_ids"] == [child_id, parent_id]
    assert payload["required_transition_ids"] == [transition_id]
    assert payload["node_count"] == 2
    assert payload["transition_count"] == 1
    by_id = {row["node_id"]: row for row in payload["nodes"]}
    assert by_id[parent_id]["code"] == before
    assert by_id[child_id]["code"] == after
    assert by_id[parent_id]["code_sha256"] == _sha(before)


def test_layer_binds_exact_code_and_generates_real_debug_diff(tmp_path):
    from experiments.end2end_memory_systems_20260804.build_recipe_implementation_capsules import (
        build,
    )
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer
    from agents.memory.experiment_r_router import _runforest_lines

    evidence_path, before, after, parent_id, child_id, transition_id = _fixture(
        tmp_path
    )
    payload = build(evidence_path, [tmp_path / "runs"])
    capsule_path = tmp_path / "capsules.json"
    _write(capsule_path, payload)

    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer.retrieval_control = "layered_strategy"
    layer.recipe_implementation_path = str(capsule_path)
    layer._recipe_evidence_ids = [child_id]
    layer._recipe_repair_evidence_by_transition = {
        transition_id: {
            "failure_node_id": parent_id,
            "successful_node_id": child_id,
            "failure_node_code_sha256": _sha(before),
            "successful_node_code_sha256": _sha(after),
        }
    }
    layer.nodes = {
        parent_id: {
            "id": parent_id,
            "type": "RunNode",
            "task": "leaf-classification",
            "code_sha256": _sha(before),
            "analysis": "division by zero",
        },
        child_id: {
            "id": child_id,
            "type": "RunNode",
            "task": "leaf-classification",
            "code_sha256": _sha(after),
            "plan": "guard the denominator",
            "analysis": "successful",
        },
        transition_id: {
            "id": transition_id,
            "type": "Transition",
            "task": "leaf-classification",
            "parent_node_id": parent_id,
            "child_node_id": child_id,
            "stage_pair": "debug->debug",
        },
    }
    layer._load_recipe_implementation_capsules()

    assert layer.nodes[child_id]["implementation_capsule"]["code"] == after
    evidence = layer._debug_transition_evidence(layer.nodes[transition_id])
    assert evidence["before_code"] == before
    assert evidence["after_code"] == after
    assert "-    return x / 0" in evidence["unified_diff"]
    assert "+    return x / max(1, x)" in evidence["unified_diff"]
    formatted = "\n".join(
        _runforest_lines(
            layer,
            {
                "id": transition_id,
                "transition_evidence": evidence,
            },
            "debug",
        )
    )
    assert "<historical_repair_diff>" in formatted
    assert "<successful_repaired_code>" in formatted
    assert after in formatted

    strategy_text = layer._format_selected_strategy(
        {
            "task_profile": {},
            "selected_strategy": {
                "sop_id": "recipe::leaf::001",
                "title": "recipe",
                "method_family": "leaf_model",
                "action": "train",
                "best_tree_evidence": {
                    "node_id": child_id,
                    "run_id": "leaf",
                    "metric": 0.1,
                    "metric_direction": "minimize",
                    "metric_provenance": "terminal",
                    "terminal_evidence": True,
                    "audit_status": "clean",
                    "code_sha256": _sha(after),
                },
            },
        }
    )
    assert "Exact Same-Task RunForest Implementation" in strategy_text
    assert after in strategy_text


def test_layer_rejects_code_that_does_not_match_runforest_identity(tmp_path):
    from experiments.end2end_memory_systems_20260804.build_recipe_implementation_capsules import (
        build,
    )
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

    evidence_path, before, after, parent_id, child_id, transition_id = _fixture(
        tmp_path
    )
    payload = build(evidence_path, [tmp_path / "runs"])
    payload["nodes"][0]["code"] += "# wrong node\n"
    capsule_path = tmp_path / "bad.json"
    _write(capsule_path, payload)
    layer = StageAwareHybridMemoryLayer.__new__(StageAwareHybridMemoryLayer)
    layer.retrieval_control = "layered_strategy"
    layer.recipe_implementation_path = str(capsule_path)
    layer._recipe_evidence_ids = [child_id]
    layer._recipe_repair_evidence_by_transition = {
        transition_id: {
            "failure_node_id": parent_id,
            "successful_node_id": child_id,
            "failure_node_code_sha256": _sha(before),
            "successful_node_code_sha256": _sha(after),
        }
    }
    layer.nodes = {
        parent_id: {"id": parent_id, "type": "RunNode", "code_sha256": _sha(before)},
        child_id: {"id": child_id, "type": "RunNode", "code_sha256": _sha(after)},
        transition_id: {
            "id": transition_id,
            "type": "Transition",
            "parent_node_id": parent_id,
            "child_node_id": child_id,
        },
    }
    with pytest.raises(ValueError, match="code hash mismatch"):
        layer._load_recipe_implementation_capsules()
