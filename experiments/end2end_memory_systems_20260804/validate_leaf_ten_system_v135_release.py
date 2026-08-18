#!/usr/bin/env python3
"""Focused release checks for the Leaf v135 two-batch launch packet."""

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
    "static_hybrid",
    "dynamic_hybrid",
    "reversed_router",
    "gome_style_port",
    "macla_style_port",
    "rcr_router_style_port",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    experiment = runtime / "experiments/end2end_memory_systems_20260804"
    manifests = experiment / "manifests_v135"
    execution = read_json(manifests / "leaf_ten_system_pilot_manifest.json")
    budget = read_json(manifests / "budget.json")["pilot"]
    systems = read_json(manifests / "systems.json")
    source_lock = read_json(manifests / "source_lock.json")

    batch1 = execution["first_parallel_batch"]
    batch2 = execution["second_parallel_batch"]
    assert len(batch1) == len(set(batch1)) == 5
    assert len(batch2) == len(set(batch2)) == 5
    assert set(batch1) | set(batch2) == EXPECTED_SYSTEMS
    assert set(batch1) & set(batch2) == set()
    assert "dynamic_hybrid" in batch1
    assert systems["system_count"] == execution["run_count"] == 10

    assert budget["agent_time_limit_seconds"] == 21_600
    assert budget["execution_timeout_seconds"] == 21_600
    assert budget["agent_steps"] >= 2_000_000_000
    assert budget["max_replacement_drafts"] >= 2_000_000_000
    assert budget["gpu_count"] == budget["parallel_search_num"] == 1
    assert budget["cpu_count"] == 16

    system_dir = experiment / "systems_v135"
    dynamic = (system_dir / "dynamic_hybrid.yaml").read_text(encoding="utf-8")
    assert "../systems_v134/dynamic_hybrid.yaml" in dynamic
    for system_id in EXPECTED_SYSTEMS - {"dynamic_hybrid"}:
        text = (system_dir / f"{system_id}.yaml").read_text(encoding="utf-8")
        assert "extends: base.yaml" in text
        assert f"end2end_memory_system: {system_id}" in text
    base_text = (system_dir / "base.yaml").read_text(encoding="utf-8")
    assert "../systems_v23/base.yaml" in base_text
    assert "gpt-5.6-sol" in base_text and "https://apizh.net/v1" in base_text
    assert "deepseek" not in base_text.lower()

    job_paths = sorted(args.jobs.resolve(strict=True).rglob("*.yaml"))
    assert len(job_paths) == 10
    seen_names = set()
    for path in job_paths:
        job = read_json(path)
        seen_names.add(job["metadata"]["name"])
        labels = job["metadata"]["labels"]
        pod_labels = job["spec"]["template"]["metadata"]["labels"]
        assert labels["ecepxie.nrp/owner"] == "haoming"
        assert labels["app.kubernetes.io/managed-by"] == "codex-nrp-training"
        assert pod_labels == labels
        assert job["spec"]["backoffLimit"] == 0
        assert "activeDeadlineSeconds" not in job["spec"]
        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["envFrom"] == [
            {"secretRef": {"name": "mlevolve-openai-gpt56sol-v1"}}
        ]
        rendered = json.dumps(job, sort_keys=True).lower()
        assert "deepseek" not in rendered
        assert "sleep" not in rendered
        requests = container["resources"]["requests"]
        limits = container["resources"]["limits"]
        assert requests == limits
        assert requests["nvidia.com/a100"] == "1"
        assert requests["cpu"] == "16"
    assert len(seen_names) == 10

    lock_map = {row["path"]: row["sha256"] for row in source_lock["files"]}
    for relative, expected in lock_map.items():
        assert sha256_file(runtime / relative) == expected, relative
    assert sha256_file(runtime / "mlevolve/agents/atomic_actuation.py") == (
        "49850c5c942eb3983e7e87df079c832d8f58660fc64452840a3a78bcd614296d"
    )

    print(
        json.dumps(
            {
                "status": "pass",
                "systems": 10,
                "batch_1": batch1,
                "batch_2": batch2,
                "jobs": len(job_paths),
                "source_locked_files": len(lock_map),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

