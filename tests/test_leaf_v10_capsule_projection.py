from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.end2end_memory_systems_20260804.publish_leaf_llm_redistilled_memory_v10 import (
    project_recipe_implementation_capsules,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_fixture(stage: Path) -> tuple[dict, dict[str, dict]]:
    recipe = stage / "recipe"
    recipe.mkdir(parents=True)
    (stage / "reports").mkdir()
    codes = {f"node-{index:02d}": f"print({index})\n" for index in range(41)}
    graph_by_id = {
        node_id: {
            "id": node_id,
            "type": "RunNode",
            "code_sha256": _sha(code),
        }
        for node_id, code in codes.items()
    }
    positive_ids = [f"node-{index:02d}" for index in range(4)]
    repair_pairs = [
        ("node-03", "node-04"),
        ("node-05", "node-06"),
        ("node-07", "node-08"),
        ("node-09", "node-10"),
        ("node-11", "node-12"),
        ("node-13", "node-14"),
    ]
    evidence = {
        "selected_evidence": {
            "leaf-classification": [
                {"node_id": node_id, "code_sha256": _sha(codes[node_id])}
                for node_id in positive_ids
            ]
        },
        "selected_repair_evidence": {
            "leaf-classification": [
                {
                    "transition_id": f"transition-{index:02d}",
                    "failure_node_id": parent_id,
                    "failure_node_code_sha256": _sha(codes[parent_id]),
                    "successful_node_id": child_id,
                    "successful_node_code_sha256": _sha(codes[child_id]),
                }
                for index, (parent_id, child_id) in enumerate(repair_pairs)
            ]
        },
    }
    capsule = {
        "schema": "mlevolve_recipe_implementation_capsules_v1",
        "coverage_policy": "stale parent coverage",
        "required_node_ids": sorted(codes),
        "required_transition_ids": [f"transition-{index:02d}" for index in range(6)],
        "nodes": [
            {
                "node_id": node_id,
                "code": code,
                "code_sha256": _sha(code),
                "source_journal": f"journal/{node_id}.json",
            }
            for node_id, code in sorted(codes.items())
        ],
        "transitions": [
            {
                "transition_id": f"transition-{index:02d}",
                "parent_node_id": parent_id,
                "child_node_id": child_id,
            }
            for index, (parent_id, child_id) in enumerate(repair_pairs)
        ],
        "node_count": 41,
        "transition_count": 6,
        "unique_code_count": 41,
        "missing_node_ids": [],
        "missing_transition_ids": [],
        "complete_recipe_coverage": True,
    }
    (recipe / "implementation_capsules.json").write_text(
        json.dumps(capsule), encoding="utf-8"
    )
    return evidence, graph_by_id


def test_projects_stale_41_node_parent_capsule_to_exact_15_node_authority(
    tmp_path: Path,
) -> None:
    evidence, graph_by_id = _write_fixture(tmp_path)

    report = project_recipe_implementation_capsules(
        stage=tmp_path,
        evidence=evidence,
        graph_by_id=graph_by_id,
    )

    projected = json.loads(
        (tmp_path / "recipe" / "implementation_capsules.json").read_text()
    )
    assert report["required_node_count"] == 15
    assert report["required_transition_count"] == 6
    assert report["dropped_parent_node_count"] == 26
    assert projected["node_count"] == 15
    assert projected["transition_count"] == 6
    assert {row["node_id"] for row in projected["nodes"]} == set(
        projected["required_node_ids"]
    )
    assert {row["transition_id"] for row in projected["transitions"]} == set(
        projected["required_transition_ids"]
    )


def test_projection_rejects_code_that_differs_from_selected_evidence(
    tmp_path: Path,
) -> None:
    evidence, graph_by_id = _write_fixture(tmp_path)
    evidence["selected_evidence"]["leaf-classification"][0][
        "code_sha256"
    ] = "f" * 64

    with pytest.raises(
        ValueError, match="differs from selected evidence"
    ):
        project_recipe_implementation_capsules(
            stage=tmp_path,
            evidence=evidence,
            graph_by_id=graph_by_id,
        )
