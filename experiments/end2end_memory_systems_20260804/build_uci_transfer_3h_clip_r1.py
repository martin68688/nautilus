#!/usr/bin/env python3
"""Freeze isolated three-hour v147/v149 runtimes for private CLIProxyAPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ENDPOINT = "http://cliproxyapi-haoming.ecepxie.svc.cluster.local:8317/v1"
SECRET = "cliproxyapi-haoming-client"
MODEL = "gpt-5.6-sol"
TIME_LIMIT_SECONDS = 10800
STEP_CEILING = 2147483647
POD_ACTIVE_DEADLINE_SECONDS = 14400

SHARED_OVERLAYS = (
    "README.md",
    "mlevolve/config/config.yaml",
    "mlevolve/analysis/adoption_verifier_smoke.py",
    "mlevolve/config/__init__.py",
    "mlevolve/run.py",
    "experiments/end2end_memory_systems_20260804/"
    "build_uci_transfer_3h_clip_r1.py",
    "tests/test_nrp_clip_endpoint_3h_runs.py",
)

VARIANTS = {
    "v147": {
        "release_id": "uci100-fixed-topk-transfer-v147-3h-clip-r1",
        "runtime": "/workspace/nautilus-exp-end2end-agent-v147-transfer-3h-clip-r1",
        "data_root": "/workspace/experiment-end2end-uci100-transfer-v147-3h-clip-r1/data/public",
        "output_root": "/workspace/experiment-end2end-uci100-transfer-v147-3h-clip-r1/runs",
        "source_data_root": "/workspace/experiment-end2end-uci100-transfer-v147-r8/data/public",
        "pod_name": "mlevolve-uci100-v147-3h-clip-r1-dev",
        "experiment_label": "experiment-end2end-uci100-v147-3h-clip-r1",
        "config": "experiments/end2end_memory_systems_20260804/"
        "systems_v147_transfer_3h_clip_r1/dynamic_cross_task_transfer.yaml",
        "pod_manifest": "experiments/end2end_memory_systems_20260804/"
        "jobs_uci_transfer_3h_clip_r1/"
        "mlevolve-uci100-v147-3h-clip-r1-dev.yaml",
        "retrieval_mode": "host_same_type_score_free_projection_fixed_topk_v1",
    },
    "v149": {
        "release_id": "uci100-dynamic-transfer-v149-3h-clip-r1",
        "runtime": "/workspace/nautilus-exp-end2end-agent-v149-dynamic-transfer-3h-clip-r1",
        "data_root": "/workspace/experiment-end2end-uci100-dynamic-transfer-v149-3h-clip-r1/data/public",
        "output_root": "/workspace/experiment-end2end-uci100-dynamic-transfer-v149-3h-clip-r1/runs",
        "source_data_root": "/workspace/experiment-end2end-uci100-dynamic-transfer-v149-r3/data/public",
        "pod_name": "mlevolve-uci100-v149-3h-clip-r1-dev",
        "experiment_label": "experiment-end2end-uci100-v149-3h-clip-r1",
        "config": "experiments/end2end_memory_systems_20260804/"
        "systems_v149_dynamic_transfer_3h_clip_r1/dynamic_cross_task_transfer.yaml",
        "pod_manifest": "experiments/end2end_memory_systems_20260804/"
        "jobs_uci_transfer_3h_clip_r1/"
        "mlevolve-uci100-v149-3h-clip-r1-dev.yaml",
        "retrieval_mode": "five_granularity_search_judge_resolver_v2",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _copy_overlay(output: Path, relative: str) -> None:
    source = REPO / relative
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _source_lock(output: Path, *, variant: str, overlays: tuple[str, ...]) -> dict:
    lock_path = output / "RELEASE_SOURCE_LOCK.json"
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink() or path == lock_path:
            continue
        relative = path.relative_to(output).as_posix()
        if (
            relative.endswith((".pyc", ".pyo"))
            or "/__pycache__/" in f"/{relative}/"
            or Path(relative).name.startswith("._")
            or Path(relative).name == ".DS_Store"
        ):
            raise ValueError(f"Forbidden runtime artifact: {relative}")
        rows.append({"path": relative, "sha256": _sha256_file(path)})
    payload = {
        "schema": "mlevolve_end2end_source_lock_v1",
        "release_id": VARIANTS[variant]["release_id"],
        "git_head": _git_head(),
        "complete_runtime_file_hash_lock": True,
        "control_file_exclusions": ["RELEASE_SOURCE_LOCK.json"],
        "overlay_scope": list(overlays),
        "files": rows,
        "manifest_hash": "",
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    payload["manifest_hash"] = hashlib.sha256(canonical).hexdigest()
    _write_json(lock_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    overlays = tuple(
        dict.fromkeys(
            (
                *SHARED_OVERLAYS,
                variant["config"],
                variant["pod_manifest"],
            )
        )
    )
    if args.output_runtime.exists():
        raise FileExistsError(f"Refusing to reuse runtime: {args.output_runtime}")
    shutil.copytree(
        args.base_runtime,
        args.output_runtime,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "RELEASE_SOURCE_LOCK.json",
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".DS_Store",
            "._*",
        ),
    )
    for relative in overlays:
        _copy_overlay(args.output_runtime, relative)
    lock = _source_lock(args.output_runtime, variant=args.variant, overlays=overlays)
    receipt = {
        "schema": "uci100_cross_task_transfer_three_hour_clip_r1",
        "variant": args.variant,
        "release_id": variant["release_id"],
        "git_head": _git_head(),
        "runtime": variant["runtime"],
        "local_runtime": str(args.output_runtime.resolve()),
        "source_lock_hash": lock["manifest_hash"],
        "source_lock_file_count": len(lock["files"]),
        "source_data_root": variant["source_data_root"],
        "data_root": variant["data_root"],
        "output_root": variant["output_root"],
        "pod_name": variant["pod_name"],
        "pod_manifest": variant["pod_manifest"],
        "experiment_label": variant["experiment_label"],
        "retrieval_mode": variant["retrieval_mode"],
        "entrypoint": "mlevolve/run.py",
        "model": MODEL,
        "endpoint": ENDPOINT,
        "credential_secret": SECRET,
        "time_limit_seconds": TIME_LIMIT_SECONDS,
        "step_limit_mode": "time_only_practical_unbounded",
        "step_ceiling": STEP_CEILING,
        "pod_active_deadline_seconds": POD_ACTIVE_DEADLINE_SECONDS,
        "preserve_all_prior_tasks_pods_outputs_checkpoints": True,
    }
    _write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
