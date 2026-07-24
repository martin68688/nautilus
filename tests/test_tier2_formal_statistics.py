from __future__ import annotations

import hashlib
import json
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = ROOT / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from analyze_tier2_formal_statistics import (  # noqa: E402
    INVENTORY_MANIFEST_SCHEMA,
    INVENTORY_SCHEMA,
    INVENTORY_VERIFICATION_SCHEMA,
    MANIFEST_SCHEMA,
    REPORT_SCHEMA,
    SUMMARY_SCHEMA,
    _bootstrap_task_macro,
    _exact_sign_flip_p,
    _holm_adjust,
    _payload_hash,
    build_statistics,
)
from verify_tier2_formal_statistics import verify_statistics  # noqa: E402


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_hashed(path: Path, payload: dict[str, Any], field: str) -> None:
    payload[field] = ""
    payload[field] = _payload_hash(payload, field)
    _write_json(path, payload)


def _score(task: str, system: str, seed: int) -> float:
    seed_offset = {104729: 0.00, 130363: 0.01, 155921: 0.02}[seed]
    if task == "new-york-city-taxi-fare-prediction":
        return 10.0 + seed_offset
    base = {
        "no_memory": 0.60,
        "flat_relevance_memory": 0.63,
        "global_validity_bit": 0.64,
        "authority_only": 0.62,
        "full_decision_admissibility": 0.70,
    }[system]
    return base + seed_offset


def _fixture(tmp_path: Path) -> dict[str, Path]:
    policy_path = tmp_path / "analysis_policy.json"
    shutil.copyfile(
        ROOT
        / "coordination"
        / "decision_admissibility_wp8_tier2_formal_analysis_policy_addendum_20260723_r1.json",
        policy_path,
    )
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    completed_root = tmp_path / "decision-admissibility-wp8-tier2-formal-runs-r10"
    continuation_root = tmp_path / "decision-admissibility-wp8-tier2-formal-runs-r13"
    (completed_root / "blocks").mkdir(parents=True)
    (continuation_root / "blocks").mkdir(parents=True)
    task_design = {row["task_id"]: row for row in policy["design"]["tasks"]}
    inventory_blocks = []
    scored_total = 0
    for block in policy["itt_disposition_matrix"]:
        block_id = block["block_id"]
        root = completed_root if block_id.endswith("-r3") else continuation_root
        block_root = root / "blocks" / block_id
        block_root.mkdir()
        task = task_design[block["task_id"]]
        online = {}
        for system, disposition in block["dispositions"].items():
            if disposition == "scored_selected_result":
                online[system] = {
                    "status": disposition,
                    "terminal_metric_observed": True,
                    "selected_score": _score(block["task_id"], system, block["agent_seed"]),
                    "result_fact_count": 1,
                    "result_fact_derived_from_refs": [],
                }
                scored_total += 1
            else:
                online[system] = {
                    "status": disposition,
                    "terminal_metric_observed": False,
                    "failure_classification": "synthetic_retained_failure",
                    "result_fact_count": 0,
                }
        candidate_count = sum(
            row["status"] == "scored_selected_result" for row in online.values()
        )
        oracle_score = None
        if candidate_count:
            scores = [
                row["selected_score"]
                for row in online.values()
                if row["status"] == "scored_selected_result"
            ]
            oracle_score = max(scores) + 0.01 if task["direction"] == "maximize" else min(scores) - 0.01
        summary: dict[str, Any] = {
            "schema": SUMMARY_SCHEMA,
            "status": "evaluation_complete",
            "block_id": block_id,
            "task_id": block["task_id"],
            "agent_seed": block["agent_seed"],
            "metric": task["native_metric"],
            "maximize": task["direction"] == "maximize",
            "online_conditions": online,
            "online_condition_count": 5,
            "successful_selected_result_count": candidate_count,
            "failed_online_condition_count": 5 - candidate_count,
            "oracle": {
                "best_score": oracle_score,
                "candidate_union_count": candidate_count,
                "scored_candidate_count": candidate_count,
                "normal_result_fact_published": False,
            },
            "system_results_use_preselected_nodes": True,
            "oracle_uses_frozen_candidate_union": True,
            "oracle_publishes_normal_result_fact": False,
            "summary_hash": "",
        }
        _write_hashed(block_root / "EVALUATION_SUMMARY.json", summary, "summary_hash")
        role = "completed_r10" if root == completed_root else "continuation_r13"
        inventory_blocks.append(
            {
                "block_id": block_id,
                "task_id": block["task_id"],
                "agent_seed": block["agent_seed"],
                "root_role": role,
                "source_snapshot_sha256": hashlib.sha256(role.encode()).hexdigest(),
                "staging_gate_hash": hashlib.sha256((role + "gate").encode()).hexdigest(),
                "required_artifact_hashes": {
                    "EVALUATION_SUMMARY.json": _file_hash(
                        block_root / "EVALUATION_SUMMARY.json"
                    )
                },
            }
        )

    inventory_root = tmp_path / "joint-inventory"
    inventory_root.mkdir()
    inventory: dict[str, Any] = {
        "schema": INVENTORY_SCHEMA,
        "status": "passed",
        "formal_roots": {
            "completed_r10": str(completed_root.resolve()),
            "continuation_r13": str(continuation_root.resolve()),
        },
        "blocks": inventory_blocks,
        "totals": {
            "block_count": 9,
            "online_condition_count": 45,
            "successful_selected_result_count": scored_total,
            "failed_online_condition_count": 45 - scored_total,
            "result_fact_count": scored_total,
            "oracle_disposition_count": 9,
        },
        "report_hash": "",
    }
    inventory["report_hash"] = _payload_hash(inventory, "report_hash")
    inventory_path = inventory_root / "joint_inventory.json"
    _write_json(inventory_path, inventory)
    manifest: dict[str, Any] = {
        "schema": INVENTORY_MANIFEST_SCHEMA,
        "status": "passed",
        "joint_inventory_file_sha256": _file_hash(inventory_path),
        "joint_inventory_hash": inventory["report_hash"],
        "manifest_hash": "",
    }
    _write_hashed(inventory_root / "manifest.json", manifest, "manifest_hash")
    verification_path = tmp_path / "joint-inventory-verification.json"
    verification: dict[str, Any] = {
        "schema": INVENTORY_VERIFICATION_SCHEMA,
        "status": "passed",
        "verified": True,
        "errors": [],
        "joint_inventory_hash": inventory["report_hash"],
        "verification_hash": "",
    }
    _write_hashed(verification_path, verification, "verification_hash")
    return {
        "policy": policy_path,
        "completed_root": completed_root,
        "continuation_root": continuation_root,
        "inventory_root": inventory_root,
        "inventory_verification": verification_path,
    }


def _build(tmp_path: Path) -> tuple[dict[str, Path], Path, dict[str, Any]]:
    fixture = _fixture(tmp_path)
    output = tmp_path / "statistics"
    report = build_statistics(
        output_root=output,
        analysis_policy_path=fixture["policy"],
        inventory_root=fixture["inventory_root"],
        inventory_verification_path=fixture["inventory_verification"],
        completed_root=fixture["completed_root"],
        continuation_root=fixture["continuation_root"],
        created_at="2026-07-23T12:00:00Z",
    )
    return fixture, output, report


def test_formal_statistics_retain_failures_and_keep_superiority_closed(
    tmp_path: Path,
) -> None:
    fixture, output, report = _build(tmp_path)
    assert report["schema"] == REPORT_SCHEMA
    assert report["analysis_population"] == {
        "assigned_online_outcomes": 45,
        "scored_selected_results": 22,
        "failed_online_conditions": 23,
        "assigned_oracle_dispositions": 9,
        "imputed_scores": 0,
        "post_assignment_exclusions": 0,
    }
    assert report["completion_by_system"]["full_decision_admissibility"]["completed"] == 4
    assert report["completion_by_system"]["no_memory"]["completed"] == 6
    primary = report["contrasts"]["full_minus_no_memory"]
    assert primary["continuous"]["availability"] == "4/9"
    assert primary["continuous"]["per_task"][
        "new-york-city-taxi-fare-prediction"
    ]["n_scored_pairs"] == 0
    assert report["effect_claim_gate"]["effect_claim_authorized"] is False
    assert "primary_has_at_least_one_scored_pair_in_each_of_all_3_tasks" in report[
        "effect_claim_gate"
    ]["failed_criteria"]
    observations = [
        json.loads(line)
        for line in (output / "formal_observations.jsonl").read_text().splitlines()
    ]
    assert len(observations) == 45
    assert sum(row["score"] is None for row in observations) == 23
    assert all(row["imputed"] is False and row["excluded"] is False for row in observations)
    assert all(not (output / name).stat().st_mode & stat.S_IWUSR for name in (
        "formal_observations.jsonl",
        "oracle_observations.jsonl",
        "paired_contrasts.jsonl",
        "oracle_gaps.jsonl",
        "statistics_report.json",
        "analysis_manifest.json",
    ))
    verification = verify_statistics(
        statistics_root=output,
        analysis_policy_path=fixture["policy"],
        inventory_root=fixture["inventory_root"],
        inventory_verification_path=fixture["inventory_verification"],
        completed_root=fixture["completed_root"],
        continuation_root=fixture["continuation_root"],
    )
    assert verification["verified"] is True
    assert verification["effect_claim_authorized"] is False


def test_formal_statistics_reject_summary_mutation_after_inventory(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    summary = next(fixture["completed_root"].rglob("EVALUATION_SUMMARY.json"))
    summary.write_bytes(summary.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after joint inventory"):
        build_statistics(
            output_root=tmp_path / "mutated-statistics",
            analysis_policy_path=fixture["policy"],
            inventory_root=fixture["inventory_root"],
            inventory_verification_path=fixture["inventory_verification"],
            completed_root=fixture["completed_root"],
            continuation_root=fixture["continuation_root"],
            created_at="2026-07-23T12:00:00Z",
        )


def test_statistics_verifier_detects_self_rehashed_result_mutation(
    tmp_path: Path,
) -> None:
    fixture, output, _report = _build(tmp_path)
    report_path = output / "statistics_report.json"
    manifest_path = output / "analysis_manifest.json"
    report_path.chmod(0o644)
    manifest_path.chmod(0o644)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["analysis_population"]["scored_selected_results"] = 23
    report["report_hash"] = _payload_hash(report, "report_hash")
    _write_json(report_path, report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["statistics_report.json"] = _file_hash(report_path)
    manifest["statistics_report_hash"] = report["report_hash"]
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    _write_json(manifest_path, manifest)
    verification = verify_statistics(
        statistics_root=output,
        analysis_policy_path=fixture["policy"],
        inventory_root=fixture["inventory_root"],
        inventory_verification_path=fixture["inventory_verification"],
        completed_root=fixture["completed_root"],
        continuation_root=fixture["continuation_root"],
    )
    assert verification["verified"] is False
    assert "report_recompute_mismatch" in verification["errors"]


def test_formal_statistics_helpers_are_frozen_and_deterministic() -> None:
    first = _bootstrap_task_macro(
        {"image": [0.1, 0.2], "audio": [0.0, 0.3], "tabular": [0.05, 0.15]},
        tasks=["image", "audio", "tabular"],
        iterations=200,
        seed=20260723,
    )
    second = _bootstrap_task_macro(
        {"image": [0.1, 0.2], "audio": [0.0, 0.3], "tabular": [0.05, 0.15]},
        tasks=["image", "audio", "tabular"],
        iterations=200,
        seed=20260723,
    )
    assert first == second
    assert first["status"] == "estimated"
    assert _exact_sign_flip_p([1.0, 1.0]) == 0.25
    adjusted = _holm_adjust(
        {"a": 0.01, "b": 0.02, "c": 0.5, "d": 1.0},
        ["a", "b", "c", "d"],
    )
    assert adjusted == {"a": 0.04, "b": 0.06, "c": 1.0, "d": 1.0}


def test_formal_statistics_refuse_output_reuse(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        build_statistics(
            output_root=output,
            analysis_policy_path=fixture["policy"],
            inventory_root=fixture["inventory_root"],
            inventory_verification_path=fixture["inventory_verification"],
            completed_root=fixture["completed_root"],
            continuation_root=fixture["continuation_root"],
            created_at="2026-07-23T12:00:00Z",
        )

