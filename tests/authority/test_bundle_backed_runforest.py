from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _write_v2_graph(bundle: Path) -> tuple[Path, Path]:
    runforest = bundle / "runforest"
    runforest.mkdir(parents=True)
    graph_path = runforest / "graph.json"
    index_path = runforest / "index.npz"
    graph_path.write_text(
        json.dumps(
            {
                "meta": {
                    "schema": "hyperbolic_run_forest_memory_v2",
                    "bundle_id": "bound-v2",
                },
                "nodes": [],
                "edges": [],
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        index_path,
        node_ids=np.asarray([], dtype=object),
        node_types=np.asarray([], dtype=object),
        poincare=np.zeros((0, 2), dtype=np.float32),
        flat_twin=np.zeros((0, 2), dtype=np.float32),
        euclidean=np.zeros((0, 64), dtype=np.float32),
    )
    return graph_path, index_path


def test_bundle_backed_coldstart_defers_until_authority_runtime(monkeypatch) -> None:
    from agents.memory import external_skill_memory
    from engine.coldstart import knowledge

    class ForbiddenPreAuthorityLayer:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("pre-Authority graph load")

    monkeypatch.setattr(
        external_skill_memory,
        "RunForestMemoryLayer",
        ForbiddenPreAuthorityLayer,
    )
    cfg = SimpleNamespace(
        external_skill_memory=SimpleNamespace(
            enable=True,
            mode="run_forest_stage_hybrid",
            source_name="run_forest_stage_hybrid_memory",
            graph_path="/forbidden/legacy/run_forest_graph.json",
            bundle_root="/workspace",
        )
    )

    text, ref_ids, source = knowledge._build_run_forest_coldstart_text(
        cfg,
        "held-out task",
    )

    assert text == ""
    assert ref_ids == []
    assert source == "run_forest_stage_hybrid_memory"


def test_runforest_v2_requires_and_accepts_hash_bound_snapshot(tmp_path: Path) -> None:
    from agents.memory.external_skill_memory import RunForestMemoryLayer

    bundle = tmp_path / "bundle"
    graph_path, index_path = _write_v2_graph(bundle)
    with pytest.raises(ValueError, match="hash-verified MemorySnapshot"):
        RunForestMemoryLayer(
            graph_path=str(graph_path),
            index_path=str(index_path),
            scoring_mode="flat_twin",
        )

    calls: list[str] = []
    base = SimpleNamespace(path=bundle, bundle_id="bound-v2")

    def verify_provenance():
        calls.append("verify")
        return {
            "source_membership_verified": True,
            "leak_verified": True,
        }

    base.verify_run_identity_provenance = verify_provenance
    snapshot = SimpleNamespace(base_bundle=base)
    snapshot.assert_unchanged = lambda: calls.append("snapshot")

    layer = RunForestMemoryLayer(
        graph_path=str(graph_path),
        index_path=str(index_path),
        scoring_mode="flat_twin",
        memory_snapshot=snapshot,
    )

    assert layer.graph["meta"]["bundle_id"] == "bound-v2"
    assert calls == ["snapshot", "verify"]
