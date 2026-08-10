from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "end2end_memory_systems_20260804"
TARGETS = EXPERIMENT / "manifests_v63" / "taxi_dynamic_replay_targets.json"
CAPSULES = EXPERIMENT / "recipe_distillation_v4_taxi" / "implementation_capsules.json"
SYSTEM = EXPERIMENT / "systems_v63" / "dynamic_hybrid.yaml"


def test_taxi_v63_replay_uses_the_frozen_recipe_capsule() -> None:
    manifest = json.loads(TARGETS.read_text(encoding="utf-8"))
    assert manifest["schema"] == "run-forest-replay-targets-v1"
    assert len(manifest["targets"]) == 1
    target = manifest["targets"][0]
    node_id = (
        "run::20260726_022228_new-york-city-taxi-fare-prediction-host-shadow-r7"
        "::node::eeb6e2364829449ba6e1ce6c1600fc3d"
    )

    assert target["task_id"] == "new-york-city-taxi-fare-prediction"
    assert target["source_kind"] == "recipe_implementation_capsule"
    assert target["graph_node_id"] == node_id
    assert target["run_id"] == (
        "20260726_022228_new-york-city-taxi-fare-prediction-host-shadow-r7"
    )
    assert target["original_node_id"] == "eeb6e2364829449ba6e1ce6c1600fc3d"
    assert target["historical_metric"] == 3.108837
    assert target["maximize"] is False
    assert target["audit_status"] == "verified_clean"
    assert target["sop_ids"] == ["sop::sg_0081"]
    assert "sop::sg_0002" not in target["sop_ids"]

    capsules = json.loads(CAPSULES.read_text(encoding="utf-8"))["nodes"]
    capsule = next(row for row in capsules if row["node_id"] == node_id)
    assert capsule["source_raw_node_id"] == target["original_node_id"]
    assert hashlib.sha256(capsule["code"].encode("utf-8")).hexdigest() == target[
        "code_sha256"
    ]


def test_taxi_v63_dynamic_config_rebinds_only_the_replay_manifest() -> None:
    text = SYSTEM.read_text(encoding="utf-8")
    assert "extends: ../systems_v62/dynamic_hybrid.yaml" in text
    assert (
        "replay_targets_path: /workspace/nautilus-exp-end2end-agent-v63/"
        "experiments/end2end_memory_systems_20260804/manifests_v63/"
        "taxi_dynamic_replay_targets.json"
    ) in text
    for stage in ("draft", "improve", "debug"):
        assert f"    - {stage}" in text
