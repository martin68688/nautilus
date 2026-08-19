from experiments.end2end_memory_systems_20260804.publish_leaf_replay_target_projection_v10 import (
    project_replay_target_sop_ids,
)


def _recipe() -> dict:
    return {
        "nodes": [
            {
                "id": "recipe::leaf::v140::001",
                "type": "SOP",
                "abstraction_level": "L1_recipe",
                "official_support": [{"candidate_id": "official-flat"}],
            },
            {
                "id": "tactic::leaf::v140::001",
                "type": "SOP",
                "abstraction_level": "L2_tactic",
                "official_support": [{"candidate_id": "official-flat"}],
            },
        ]
    }


def _graph() -> dict:
    return {"nodes": [{"id": "sop::legacy-valid", "type": "SOP"}]}


def test_projection_restores_canonical_l1_recipe_by_official_target() -> None:
    targets = {
        "targets": [
            {"target_id": "official-flat", "sop_ids": ["recipe::leaf::v139::001"]}
        ]
    }

    projected, report = project_replay_target_sop_ids(
        targets=targets, recipe=_recipe(), graph=_graph()
    )

    assert projected["targets"][0]["sop_ids"] == ["recipe::leaf::v140::001"]
    assert report["canonical_l1_binding_count"] == 1
    assert report["invalid_projected_sop_id_count"] == 0


def test_projection_does_not_fabricate_recipe_for_unrepresented_target() -> None:
    targets = {
        "targets": [
            {"target_id": "official-no-memory", "sop_ids": ["missing::recipe"]}
        ]
    }

    projected, report = project_replay_target_sop_ids(
        targets=targets, recipe=_recipe(), graph=_graph()
    )

    assert projected["targets"][0]["sop_ids"] == []
    assert report["targets_without_distilled_l1_recipe_count"] == 1


def test_projection_preserves_existing_valid_graph_sop_without_l1_recipe() -> None:
    targets = {
        "targets": [
            {"target_id": "legacy-target", "sop_ids": ["sop::legacy-valid"]}
        ]
    }

    projected, report = project_replay_target_sop_ids(
        targets=targets, recipe=_recipe(), graph=_graph()
    )

    assert projected["targets"][0]["sop_ids"] == ["sop::legacy-valid"]
    assert report["rows"][0]["disposition"] == "existing_valid_sop_binding_retained"
