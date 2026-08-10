from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "end2end_memory_systems_20260804"
TARGETS = EXPERIMENT / "manifests_v62" / "taxi_dynamic_replay_targets.json"
CAPSULES = (
    EXPERIMENT
    / "recipe_distillation_v4_taxi"
    / "implementation_capsules.json"
)
SYSTEM = EXPERIMENT / "systems_v62" / "dynamic_hybrid.yaml"


def test_taxi_v62_replay_target_matches_frozen_recipe_capsule() -> None:
    manifest = json.loads(TARGETS.read_text(encoding="utf-8"))
    assert manifest["schema"] == "run-forest-replay-targets-v1"
    assert len(manifest["targets"]) == 1
    target = manifest["targets"][0]
    assert target["task_id"] == "new-york-city-taxi-fare-prediction"
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
    node_id = f"run::{target['run_id']}::node::{target['original_node_id']}"
    capsule = next(row for row in capsules if row["node_id"] == node_id)
    assert hashlib.sha256(capsule["code"].encode("utf-8")).hexdigest() == target[
        "code_sha256"
    ]


def test_taxi_v62_dynamic_config_uses_repaired_target_manifest() -> None:
    text = SYSTEM.read_text(encoding="utf-8")
    assert "extends: ../systems_v54/dynamic_hybrid.yaml" in text
    assert (
        "replay_targets_path: /workspace/nautilus-exp-end2end-agent-v62/"
        "experiments/end2end_memory_systems_20260804/manifests_v62/"
        "taxi_dynamic_replay_targets.json"
    ) in text
    for stage in ("draft", "improve", "debug"):
        assert f"    - {stage}" in text
