#!/usr/bin/env python3
"""Focused checks for the four-control v136 bundle-binding repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SYSTEMS = {
    "no_memory",
    "flat_retrieval",
    "sop_only",
    "runforest_only",
}
EXPECTED_MEMORY_HASH = (
    "c46e1ee5e9c2a59330d1f7f44338f5a08e622ebd1715e814c06704d11ee579c6"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    args = parser.parse_args()
    base_runtime = args.base_runtime.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    jobs = args.jobs.resolve(strict=True)
    experiment = runtime / "experiments/end2end_memory_systems_20260804"
    manifests = experiment / "manifests_v136"
    execution = read_json(manifests / "leaf_control_repair_manifest.json")
    budget = read_json(manifests / "budget.json")["pilot"]
    systems = read_json(manifests / "systems.json")
    memory = read_json(manifests / "memory_bundles.json")
    evaluators = read_json(manifests / "evaluators.json")
    source_lock = read_json(manifests / "source_lock.json")

    assert execution["run_count"] == systems["system_count"] == 4
    assert set(execution["system_ids"]) == EXPECTED_SYSTEMS
    assert set(execution["first_parallel_batch"]) == EXPECTED_SYSTEMS
    assert execution["second_parallel_batch"] == []
    assert all("v136" in row["logical_run_id"] for row in execution["runs"])
    assert all(row["bindings"]["memory_bundles_manifest_hash"] == EXPECTED_MEMORY_HASH for row in execution["runs"])

    assert memory["manifest_hash"] == EXPECTED_MEMORY_HASH
    bundle = memory["task_bundles"]["leaf-classification"]
    assert bundle["bundle_version"] == "v2"
    assert bundle["bundle_id"] == "end2end-fourtask-direct-leaf-classification-v2"
    assert bundle["bundle_root"] == "/workspace/experiment-end2end-memory-agent-v13/memory-direct-v2/leaf-classification"
    assert "formal_clause_file_sha256" not in bundle
    assert "v136" in evaluators["formal_releases_root"]

    assert budget["agent_time_limit_seconds"] == 21_600
    assert budget["execution_timeout_seconds"] == 21_600
    assert budget["agent_steps"] >= 2_000_000_000
    assert budget["max_replacement_drafts"] >= 2_000_000_000
    assert budget["gpu_count"] == budget["parallel_search_num"] == 1
    assert budget["cpu_count"] == 16

    system_dir = experiment / "systems_v136"
    assert "../systems_v135/base.yaml" in (system_dir / "base.yaml").read_text(encoding="utf-8")
    for system_id in EXPECTED_SYSTEMS:
        text = (system_dir / f"{system_id}.yaml").read_text(encoding="utf-8")
        assert "extends: base.yaml" in text
        assert f"end2end_memory_system: {system_id}" in text

    assert tree_hashes(runtime / "mlevolve") == tree_hashes(base_runtime / "mlevolve")

    job_paths = sorted(jobs.glob("*.yaml"))
    assert len(job_paths) == 4
    for path in job_paths:
        job = read_json(path)
        assert "v136" in job["metadata"]["name"]
        labels = job["metadata"]["labels"]
        assert labels["experiment"] == "experiment-end2end-memory-agent-v136"
        assert labels["ecepxie.nrp/owner"] == "haoming"
        assert labels["app.kubernetes.io/managed-by"] == "codex-nrp-training"
        assert job["spec"]["backoffLimit"] == 0
        assert "activeDeadlineSeconds" not in job["spec"]
        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["envFrom"] == [{"secretRef": {"name": "mlevolve-openai-gpt56sol-v1"}}]
        assert container["resources"]["requests"] == container["resources"]["limits"]
        assert container["resources"]["requests"]["nvidia.com/a100"] == "1"
        rendered = json.dumps(job, sort_keys=True).lower()
        assert "deepseek" not in rendered
        assert "sleep" not in rendered
        assert "v135" not in rendered

    lock_map = {row["path"]: row["sha256"] for row in source_lock["files"]}
    for relative, expected in lock_map.items():
        assert sha256_file(runtime / relative) == expected, relative

    print(json.dumps({
        "status": "pass",
        "systems": sorted(EXPECTED_SYSTEMS),
        "jobs": len(job_paths),
        "memory_bundle_version": bundle["bundle_version"],
        "mlevolve_tree_unchanged": True,
        "source_locked_files": len(lock_map),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
