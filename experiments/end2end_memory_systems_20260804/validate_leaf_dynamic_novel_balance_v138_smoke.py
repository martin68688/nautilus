#!/usr/bin/env python3
"""Validate the frozen v138 Dynamic Dev-Pod smoke release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_leaf_dynamic_novel_balance_v138_smoke as builder
import build_leaf_ten_system_gpt_v135_runtime as v135


def read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--pod", required=True, type=Path)
    parser.add_argument("--generation", type=int, default=1)
    args = parser.parse_args()
    builder.configure_generation(args.generation)

    runtime = args.runtime.resolve(strict=True)
    manifests = runtime / builder.MANIFEST_DIR
    execution = read(manifests / "leaf_dynamic_smoke_manifest.json")
    budget = read(manifests / "budget.json")["smoke"]
    systems = read(manifests / "systems.json")
    source_lock = read(manifests / "source_lock.json")
    pod = read(args.pod.resolve(strict=True))

    assert execution["run_count"] == systems["system_count"] == 1
    assert execution["system_ids"] == ["dynamic_hybrid"]
    assert execution["bindings"]["source_lock_manifest_hash"] == source_lock[
        "manifest_hash"
    ]
    assert budget["agent_steps"] == 6
    assert budget["gpu_count"] == budget["parallel_search_num"] == 1
    assert budget["cpu_count"] == 16
    assert budget["memory_gib"] == 64

    system_row = systems["systems"][0]
    config = runtime / builder.EXPERIMENT / system_row["config_path"]
    assert config.resolve(strict=True) == (
        runtime / builder.SYSTEM_DIR / "dynamic_hybrid.yaml"
    ).resolve(strict=True)
    config_text = config.read_text(encoding="utf-8")
    assert "extends: ../systems_v137_full/dynamic_hybrid.yaml" in config_text
    assert "ensure_valid_candidate_per_role: true" in config_text
    assert "role_balance_min_valid_candidates: 1" in config_text
    assert builder.CLUSTER_RUNTIME in config_text
    assert "deepseek" not in config_text.lower()

    executor = (runtime / "mlevolve/engine/executor.py").read_text(encoding="utf-8")
    scheduler = (runtime / "mlevolve/engine/agent_search.py").read_text(
        encoding="utf-8"
    )
    assert "_insert_host_preamble_after_future_imports" in executor
    assert 'exc_type="HostSourceInstrumentationError"' in executor
    assert "role_balance_status" in scheduler
    assert "select_role_balance_deficit" in scheduler

    lock_map = {row["path"]: row["sha256"] for row in source_lock["files"]}
    assert source_lock["overlay_scope"] == [
        path.as_posix() for path in builder.OVERLAY_FILES
    ]
    assert not any("__pycache__" in relative for relative in lock_map)
    assert not any(relative.endswith(".pyc") for relative in lock_map)
    for relative, expected in lock_map.items():
        assert v135.sha256_file(runtime / relative) == expected, relative
    actual_files = {
        path.relative_to(runtime).as_posix()
        for path in runtime.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert not (
        actual_files
        - set(lock_map)
        - set(source_lock["control_file_exclusions"])
    )

    assert pod["kind"] == "Pod"
    assert pod["metadata"]["name"] == builder.POD_NAME
    labels = pod["metadata"]["labels"]
    assert labels["ecepxie.nrp/owner"] == "haoming"
    assert labels["experiment"] == builder.EXPERIMENT_LABEL
    container = pod["spec"]["containers"][0]
    assert container["envFrom"] == [
        {"secretRef": {"name": "mlevolve-openai-gpt56sol-v1"}}
    ]
    assert container["resources"]["requests"] == container["resources"]["limits"]
    assert container["resources"]["limits"]["nvidia.com/a100"] == "1"
    assert container["resources"]["limits"]["cpu"] == "16"
    assert container["resources"]["limits"]["memory"] == (
        f"{builder.DEV_MEMORY_GIB}Gi"
    )
    assert len(pod["spec"]["volumes"]) == 2
    assert sum(
        1
        for volume in pod["spec"]["volumes"]
        if "persistentVolumeClaim" in volume
    ) == 1
    rendered = json.dumps(pod, sort_keys=True).lower()
    assert "deepseek" not in rendered
    assert "openai_api_key" not in rendered

    print(
        json.dumps(
            {
                "status": "pass",
                "source_locked_files": len(lock_map),
                "agent_steps": budget["agent_steps"],
                "pod": builder.POD_NAME,
                "role_balance_min_valid_candidates": 1,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
