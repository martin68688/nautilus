from __future__ import annotations

import pytest

from authority.replay_certifier import (
    ProtocolRepairSurface,
    ReplayIdentity,
    verify_protocol_only_patch,
)
from tests.authority.clean_replay_helpers import SOURCE_CODE


SURFACE = ProtocolRepairSurface.from_allowed_changes(
    ["split_api", "preprocessing_scope", "evaluator", "instrumentation"]
)


@pytest.mark.parametrize(
    ("changed", "field"),
    [
        (
            SOURCE_CODE.replace("LogisticRegression", "RandomForestClassifier"),
            "model_families",
        ),
        (
            SOURCE_CODE.replace("ngram_range=(1, 2)", "ngram_range=(1, 3)"),
            "feature_signatures",
        ),
        (SOURCE_CODE.replace('loss_name = "log_loss"', 'loss_name = "hinge"'), "loss_objective_signatures"),
        (SOURCE_CODE.replace('"C": [0.5, 1.0]', '"C": [0.25, 2.0]'), "search_space_signatures"),
        (SOURCE_CODE.replace("max_iter=100", "max_iter=200"), "model_signatures"),
        (SOURCE_CODE.replace("ensemble_weights = [1.0]", "ensemble_weights = [0.5, 0.5]"), "ensemble_signatures"),
    ],
)
def test_declared_protocol_repair_cannot_change_method_surface(changed, field) -> None:
    report = verify_protocol_only_patch(
        SOURCE_CODE,
        changed,
        SURFACE,
        source_artifact_id="source",
        replay_artifact_id="fake-replay",
    )
    assert report.identity == ReplayIdentity.SUCCESSOR_METHOD
    assert field in report.protected_changes
    assert report.reason == "protected_method_surface_changed"


def test_unclassified_call_delta_requires_human_review() -> None:
    report = verify_protocol_only_patch(
        SOURCE_CODE,
        SOURCE_CODE + "\nsecret_side_effect()\n",
        SURFACE,
    )
    assert report.identity == ReplayIdentity.REQUIRE_HUMAN_REVIEW
    assert report.unclassified_call_deltas == ("secret_side_effect",)
