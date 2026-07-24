from __future__ import annotations

import copy
import hashlib
import json
import stat
import sys
from collections import Counter
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_tier1_controlled_episodes import (  # noqa: E402
    AGENT_SEEDS,
    CELLS,
    CONDITIONS,
    DEFAULT_LEGACY_EPISODE_ROOT,
    HIDDEN_AGENT_KEYS,
    MISMATCH_SOURCE_STAGE,
    STAGES,
    audit_legacy_overlap,
    build,
    build_episodes,
    project_agent_view,
    validate_episodes,
)
from schema import sha256_json  # noqa: E402
from verify_tier1_controlled_episodes import (  # noqa: E402
    verify_packet,
    write_verification_exclusive,
)


CREATED_AT = "2026-07-21T00:00:00Z"


def _by_id(episodes: list[dict], episode_id: str) -> dict:
    return next(row for row in episodes if row["episode_id"] == episode_id)


def _action_by_role(episode: dict, role: str) -> dict:
    return next(
        action for action in episode["action_candidates"] if action["role"] == role
    )


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _write_legacy_fixture(root: Path, rows: list[dict]) -> Path:
    root.mkdir(parents=True)
    path = root / "legacy.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return root


def test_packet_has_fresh_balanced_heldout_factorial_floor() -> None:
    episodes = build_episodes()
    validation = validate_episodes(episodes)

    assert validation["valid"] is True
    assert validation["errors"] == []
    assert validation["episode_count"] == 24
    assert validation["stage_counts"] == {stage: 6 for stage in STAGES}
    assert validation["source_task_count"] == 6
    assert validation["target_task_count"] == 6
    assert validation["source_run_count"] == 24
    assert validation["source_episode_count"] == 24
    assert validation["independent_source_episodes_per_cell"] == 24
    assert validation["planned_agent_run_count"] == 360
    assert Counter(
        cell for episode in episodes for cell in episode["memory_cells"]
    ) == {cell: 24 for cell in CELLS}
    assert all(episode["agent_seeds"] == list(AGENT_SEEDS) for episode in episodes)


def test_every_source_is_same_domain_different_task_without_target_history() -> None:
    episodes = build_episodes()

    for episode in episodes:
        assert episode["same_domain_different_task"] is True
        assert episode["source_task_id"] != episode["target_task_id"]
        assert episode["target_history_refs"] == []
        assert not episode["source_task_id"].endswith(f"-{episode['stage']}")
        for memory in episode["memory_cells"].values():
            assert memory["scope"] == "same_domain_different_task"
            assert memory["source_domain"] == episode["domain"]
            assert memory["source_task_id"] == episode["source_task_id"]
            assert memory["source_run_id"] == episode["source_run_id"]
            assert memory["source_episode_id"] == episode["source_episode_id"]
            assert memory["target_task_history_present"] is False


def test_factor_pairs_change_authority_without_changing_recommended_action() -> None:
    for episode in build_episodes():
        cells = episode["memory_cells"]
        assert cells["F10"]["recommended_action_id"] == cells["F11"][
            "recommended_action_id"
        ]
        assert cells["F00"]["recommended_action_id"] == cells["F01"][
            "recommended_action_id"
        ]
        assert cells["F10"]["granularity_match"] is True
        assert cells["F11"]["granularity_match"] is True
        assert cells["F00"]["granularity_match"] is False
        assert cells["F01"]["granularity_match"] is False
        assert cells["F00"]["authority_valid"] is False
        assert cells["F10"]["authority_valid"] is False
        assert cells["F01"]["authority_valid"] is True
        assert cells["F11"]["authority_valid"] is True
        assert cells["F00"]["semantic_purity_report"]["passed"] is False
        assert cells["F10"]["semantic_purity_report"]["passed"] is False
        assert cells["F01"]["semantic_purity_report"]["passed"] is True
        assert cells["F11"]["semantic_purity_report"]["passed"] is True
        assert "inherit its conclusion" in cells["F10"]["text"]
        assert "inherit no source score" in cells["F11"]["text"]
        assert (
            f"{MISMATCH_SOURCE_STAGE[episode['stage']]}-level"
            in cells["F01"]["text"]
        )


def test_agent_projection_is_the_only_gold_free_prompt_surface() -> None:
    episode = build_episodes()[0]
    views_by_condition = {}

    for condition in CONDITIONS:
        for seed in AGENT_SEEDS:
            view = project_agent_view(
                episode,
                condition=condition,
                agent_seed=seed,
            )
            assert not (set(_walk_keys(view)) & HIDDEN_AGENT_KEYS)
            assert view["view_hash"] == sha256_json(
                {key: value for key, value in view.items() if key != "view_hash"}
            )
            assert len(view["action_candidates"]) == 4
            assert all(
                set(action) == {"action_id", "label", "config_patch"}
                for action in view["action_candidates"]
            )
            if condition == "NM":
                assert view["memory_context"] == []
            else:
                assert len(view["memory_context"]) == 1
                assert "cell" not in view["memory_context"][0]
                assert condition not in json.dumps(view, sort_keys=True)
            views_by_condition.setdefault(condition, []).append(view)

    assert all(
        len({view["view_hash"] for view in views}) == 1
        for views in views_by_condition.values()
    )

    with pytest.raises(ValueError, match="Unknown Tier-1 condition"):
        project_agent_view(episode, condition="F99", agent_seed=AGENT_SEEDS[0])
    with pytest.raises(ValueError, match="Unknown Tier-1 agent seed"):
        project_agent_view(episode, condition="NM", agent_seed=999)


def test_governance_gold_keeps_result_adoption_and_causal_objects_separate() -> None:
    episodes = build_episodes()
    cold = _action_by_role(
        _by_id(episodes, "tier1::natural-image::governance"), "oracle"
    )["config_patch"]
    exposed_not_actuated = _action_by_role(
        _by_id(episodes, "tier1::medical-image::governance"), "oracle"
    )["config_patch"]
    adopted = _action_by_role(
        _by_id(episodes, "tier1::nlp::governance"), "oracle"
    )["config_patch"]
    causal_not_effective = _action_by_role(
        _by_id(episodes, "tier1::audio::governance"), "oracle"
    )["config_patch"]
    invalid_score = _action_by_role(
        _by_id(episodes, "tier1::tabular::governance"), "oracle"
    )["config_patch"]
    pre_terminal = _action_by_role(
        _by_id(episodes, "tier1::temporal::governance"), "oracle"
    )["config_patch"]

    assert cold == {
        "promote_result": True,
        "publish_adoption": False,
        "publish_causal": False,
    }
    assert exposed_not_actuated == cold
    assert adopted == {
        "promote_result": True,
        "publish_adoption": True,
        "publish_causal": False,
    }
    assert causal_not_effective == {
        "publish_adoption": True,
        "publish_causal": True,
        "claim_effective": False,
    }
    assert invalid_score["promote_result"] is False
    assert invalid_score["distill_diagnostic"] is True
    assert pre_terminal == {
        "defer_writeback": True,
        "promote_result": False,
        "terminal_score_required": True,
    }


def test_validator_rejects_pairing_hash_domain_and_gold_projection_tampering() -> None:
    baseline = build_episodes()

    paired = copy.deepcopy(baseline)
    paired[0]["memory_cells"]["F10"]["recommended_action_id"] = paired[0][
        "neutral_action_id"
    ]
    assert any(
        error.startswith("matched_authority_pair_action:")
        for error in validate_episodes(paired)["errors"]
    )

    domain = copy.deepcopy(baseline)
    domain[0]["memory_cells"]["F11"]["source_domain"] = "forbidden-cross-domain"
    assert any(
        error.startswith("cross_domain:")
        for error in validate_episodes(domain)["errors"]
    )

    target_history = copy.deepcopy(baseline)
    target_history[0]["target_history_refs"] = ["forbidden-target-run"]
    assert any(
        error.startswith("target_history_refs:")
        for error in validate_episodes(target_history)["errors"]
    )


def test_exact_legacy_overlap_audit_binds_old_ids_and_long_text(
    tmp_path: Path,
) -> None:
    episodes = build_episodes()
    clean_root = _write_legacy_fixture(
        tmp_path / "clean-legacy",
        [{"episode_id": "legacy::001", "query_text": "x" * 80}],
    )
    assert audit_legacy_overlap(episodes, clean_root)["passed"] is True

    colliding_root = _write_legacy_fixture(
        tmp_path / "colliding-legacy",
        [
            {
                "episode_id": episodes[0]["episode_id"],
                "query_text": episodes[0]["current_state"],
            }
        ],
    )
    report = audit_legacy_overlap(episodes, colliding_root)
    assert report["passed"] is False
    assert report["episode_id_overlap_count"] == 1
    assert report["exact_long_text_overlap_count"] == 1
    assert len(report["exact_long_text_overlap_hashes"][0]) == 64


def test_repository_superseded_composite_episodes_have_zero_exact_overlap() -> None:
    report = audit_legacy_overlap(build_episodes(), DEFAULT_LEGACY_EPISODE_ROOT)

    assert report["passed"] is True
    assert report["episode_id_overlap_count"] == 0
    assert report["exact_long_text_overlap_count"] == 0
    assert report["legacy_file_count"] > 0
    assert report["legacy_row_count"] > 0


def test_builder_writes_hashed_read_only_packet_and_refuses_reuse(tmp_path: Path) -> None:
    legacy_root = _write_legacy_fixture(
        tmp_path / "legacy",
        [{"episode_id": "legacy::001", "query_text": "legacy " * 20}],
    )
    output = tmp_path / "packet"
    manifest = build(
        output,
        created_at=CREATED_AT,
        legacy_episode_root=legacy_root,
    )
    episode_path = output / "episodes.jsonl"
    manifest_path = output / "manifest.json"
    on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert on_disk == manifest
    assert sum(1 for line in episode_path.read_text().splitlines() if line) == 24
    assert manifest["episode_file_sha256"] == hashlib.sha256(
        episode_path.read_bytes()
    ).hexdigest()
    assert manifest["manifest_hash"] == sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    assert manifest["legacy_overlap_audit"]["passed"] is True
    assert manifest["old_composite_episode_reuse_count"] == 0
    assert manifest["agent_seed_semantics"] == (
        "host_paired_replicate_id_not_provider_rng_seed"
    )
    assert manifest["agent_seed_exposed_to_agent"] is False
    assert manifest["provider_seed_parameter_planned"] is False
    assert stat.S_IMODE(episode_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o444

    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        build(
            output,
            created_at=CREATED_AT,
            legacy_episode_root=legacy_root,
        )


def test_standalone_packet_verifier_checks_all_360_gold_free_views(
    tmp_path: Path,
) -> None:
    legacy_root = _write_legacy_fixture(
        tmp_path / "legacy",
        [{"episode_id": "legacy::001", "query_text": "legacy " * 20}],
    )
    packet = tmp_path / "packet"
    build(packet, created_at=CREATED_AT, legacy_episode_root=legacy_root)

    report = verify_packet(packet, legacy_episode_root=legacy_root)

    assert report["verified"] is True
    assert report["errors"] == []
    assert report["agent_view_count"] == 360
    assert report["expected_unique_agent_view_hash_count"] == 120
    assert report["unique_agent_view_hash_count"] == 120
    assert report["agent_view_gold_leak_count"] == 0
    assert report["agent_view_projection_error_count"] == 0
    assert len(report["verification_hash"]) == 64


def test_packet_verifier_fails_closed_on_payload_or_builder_drift(tmp_path: Path) -> None:
    legacy_root = _write_legacy_fixture(
        tmp_path / "legacy",
        [{"episode_id": "legacy::001", "query_text": "legacy " * 20}],
    )
    packet = tmp_path / "packet"
    build(packet, created_at=CREATED_AT, legacy_episode_root=legacy_root)
    episode_path = packet / "episodes.jsonl"
    episode_path.chmod(0o644)
    episode_path.write_text(
        episode_path.read_text(encoding="utf-8") + "{}\n",
        encoding="utf-8",
    )
    wrong_builder = tmp_path / "wrong_builder.py"
    wrong_builder.write_text("# not the bound builder\n", encoding="utf-8")

    report = verify_packet(
        packet,
        legacy_episode_root=legacy_root,
        builder_source=wrong_builder,
    )

    assert report["verified"] is False
    assert "episode_file_hash" in report["errors"]
    assert "builder_source_hash" in report["errors"]
    assert "packet_file_modes" in report["errors"]
    assert "manifest_validation_binding" in report["errors"]


def test_verification_receipt_is_exclusive_and_read_only(tmp_path: Path) -> None:
    output = tmp_path / "verification.json"
    report = {"schema": "test", "verified": True}

    write_verification_exclusive(output, report)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert stat.S_IMODE(output.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        write_verification_exclusive(output, report)
