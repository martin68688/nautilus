#!/usr/bin/env python3
"""Validate one isolated v137 Dynamic-only release and its workload specs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_leaf_dynamic_retrieval_v137_runtime as builder
import build_leaf_ten_system_gpt_v135_runtime as v135


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    args = parser.parse_args()
    spec = builder.identity(args.mode, args.generation)
    runtime = args.runtime.resolve(strict=True)
    manifests = runtime / spec["manifest_dir"]
    execution = read(manifests / spec["execution_name"])
    budget_payload = read(manifests / "budget.json")
    budget = budget_payload[spec["kind"]]
    systems = read(manifests / "systems.json")
    source_lock = read(manifests / "source_lock.json")

    assert execution["run_count"] == systems["system_count"] == 1
    assert execution["system_ids"] == [builder.SYSTEM_ID]
    assert execution["runs"][0]["logical_run_id"] == spec["logical_run_id"]
    assert execution["bindings"]["source_lock_manifest_hash"] == source_lock[
        "manifest_hash"
    ]
    assert source_lock["overlay_scope"] == [
        path.as_posix() for path in builder.OVERLAY_FILES
    ]
    if args.mode == "smoke":
        assert execution["kind"] == "smoke"
        assert budget["agent_steps"] == 5
        assert execution["formal_result_eligible"] is False
    else:
        assert execution["kind"] == "pilot"
        assert budget["agent_time_limit_seconds"] == 21_600
        assert budget["execution_timeout_seconds"] == 21_600
        assert budget["agent_steps"] >= 2_000_000_000
        assert budget["max_replacement_drafts"] >= 2_000_000_000
        assert execution["formal_result_eligible"] is True
    assert budget["gpu_count"] == budget["parallel_search_num"] == 1
    assert budget["cpu_count"] == 16
    assert budget["memory_gib"] == 64

    experiment_root = runtime / builder.EXPERIMENT
    system_row = systems["systems"][0]
    config_path = (experiment_root / system_row["config_path"]).resolve(strict=True)
    assert config_path == (
        runtime / spec["system_dir"] / "dynamic_hybrid.yaml"
    ).resolve(strict=True)
    config = config_path.read_text(encoding="utf-8")
    assert "extends: ../systems_v135/dynamic_hybrid.yaml" in config
    assert spec["cluster_runtime"] in config
    assert "deepseek" not in config.lower()

    openai_text = (runtime / builder.OVERLAY_FILES[0]).read_text(encoding="utf-8")
    grep_text = (runtime / builder.OVERLAY_FILES[1]).read_text(encoding="utf-8")
    assert "_validate_tool_output(output, func_spec)" in openai_text
    assert 'JUDGE_SCHEMA = "experiment_r_multigranular_retrieval_judge_v2"' in grep_text
    assert 'f"C{index:02d}"' in grep_text
    assert "live_multigranular_grep_search_host_fallback" in grep_text

    lock_map = {row["path"]: row["sha256"] for row in source_lock["files"]}
    assert not any("__pycache__" in relative for relative in lock_map)
    assert not any(relative.endswith(".pyc") for relative in lock_map)
    assert not any(".pytest_cache" in relative for relative in lock_map)
    for relative, expected in lock_map.items():
        assert v135.sha256_file(runtime / relative) == expected, relative
    actual_files = {
        path.relative_to(runtime).as_posix()
        for path in runtime.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    allowed_extras = set(source_lock["control_file_exclusions"])
    assert not (actual_files - set(lock_map) - allowed_extras)
    assert lock_map[builder.OVERLAY_FILES[0].as_posix()] == v135.sha256_file(
        builder.REPO / builder.OVERLAY_FILES[0]
    )
    assert lock_map[builder.OVERLAY_FILES[1].as_posix()] == v135.sha256_file(
        builder.REPO / builder.OVERLAY_FILES[1]
    )

    job_files = sorted(args.jobs.resolve(strict=True).glob("*.yaml"))
    assert len(job_files) == 2
    payloads = [read(path) for path in job_files]
    job = next(row for row in payloads if row["kind"] == "Job")
    stager = next(row for row in payloads if row["kind"] == "Pod")
    assert job["metadata"]["name"] == spec["workload"]
    assert stager["metadata"]["name"] == spec["stager"]
    assert job["metadata"]["labels"] == job["spec"]["template"]["metadata"]["labels"]
    assert job["metadata"]["labels"]["ecepxie.nrp/owner"] == "haoming"
    assert job["spec"]["backoffLimit"] == 0
    assert "activeDeadlineSeconds" not in job["spec"]
    container = job["spec"]["template"]["spec"]["containers"][0]
    assert container["envFrom"] == [
        {"secretRef": {"name": "mlevolve-openai-gpt56sol-v1"}}
    ]
    assert container["resources"]["requests"] == container["resources"]["limits"]
    assert container["resources"]["limits"]["nvidia.com/a100"] == "1"
    rendered_job = json.dumps(job, sort_keys=True).lower()
    assert "deepseek" not in rendered_job
    assert "sleep" not in rendered_job
    assert "--attempt 0" in rendered_job

    expected_hash = hashlib.sha256(
        f"{v135.LLM_BASE_URL}|{v135.LLM_MODEL}|chat-completions".encode()
    ).hexdigest()
    assert expected_hash in rendered_job
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": args.mode,
                "source_locked_files": len(lock_map),
                "agent_steps": budget["agent_steps"],
                "agent_time_limit_seconds": budget["agent_time_limit_seconds"],
                "job": spec["workload"],
                "stager": spec["stager"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
