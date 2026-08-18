#!/usr/bin/env python3
"""Build the fresh Leaf ten-system v135 GPT runtime and two launch batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
BASE_MANIFESTS = EXPERIMENT / "manifests_v134"
TARGET_MANIFESTS = EXPERIMENT / "manifests_v135"
TARGET_SYSTEMS = EXPERIMENT / "systems_v135"
EXECUTION_MANIFEST_NAME = "leaf_ten_system_pilot_manifest.json"
RUNTIME_IMAGE = (
    "docker.io/haomingwang22/mlevolve@sha256:"
    "fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc"
)
CLUSTER_RUNTIME = "/workspace/nautilus-exp-end2end-agent-v135"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v135/runs"
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v135"
LLM_SECRET = "mlevolve-openai-gpt56sol-v1"
LLM_MODEL = "gpt-5.6-sol"
LLM_BASE_URL = "https://apizh.net/v1"
UNBOUNDED_WITHIN_SIX_HOURS = 2_147_483_647

SYSTEM_IDS = (
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
)
BATCH_1 = (
    "dynamic_hybrid",
    "no_memory",
    "flat_retrieval",
    "sop_only",
    "runforest_only",
)
BATCH_2 = tuple(system for system in SYSTEM_IDS if system not in BATCH_1)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_hashed(path: Path, payload: dict[str, Any], field: str) -> str:
    payload[field] = payload_hash(payload, field)
    write_json(path, payload)
    return str(payload[field])


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source.resolve(strict=True), destination)


def overlay_latest_release_files(output: Path) -> None:
    for source in sorted((REPO / TARGET_SYSTEMS).glob("*.yaml")):
        copy_file(source, output / TARGET_SYSTEMS / source.name)
    for relative in (
        EXPERIMENT / "run_assignment.py",
        EXPERIMENT / "stage_leaf_official_evaluator_v127.py",
        Path("mlevolve/agents/atomic_actuation.py"),
        Path("mlevolve/fixed_holdout/mode.py"),
        Path("mlevolve/llm/gemini.py"),
    ):
        copy_file(REPO / relative, output / relative)


def build_source_lock(
    output: Path, *, source_revision: str, manifests: Path
) -> dict[str, Any]:
    excluded = {
        (manifests / "source_lock.json").as_posix(),
        (manifests / EXECUTION_MANIFEST_NAME).as_posix(),
        "SOURCE_FILES.sha256",
    }
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {
        "schema": "mlevolve_end2end_source_lock_v1",
        "release_id": "end2end-leaf-ten-system-official-gpt56sol-v135",
        "git_head": source_revision,
        "git_head_is_not_sufficient_identity": True,
        "complete_runtime_file_hash_lock": True,
        "control_file_exclusions": sorted(excluded),
        "files": files,
        "manifest_hash": "",
    }


def build_components(output: Path) -> dict[str, str]:
    base = output / BASE_MANIFESTS
    manifests = output / TARGET_MANIFESTS
    manifests.mkdir(parents=True, exist_ok=False)
    for name in (
        "memory_bundles.json",
        "schemas.json",
        "tasks.json",
        "leaf_official_replay_targets.json",
    ):
        shutil.copy2(base / name, manifests / name)

    budget = read_json(base / "budget.json")
    budget["pilot"].update(
        {
            "agent_steps": UNBOUNDED_WITHIN_SIX_HOURS,
            "agent_time_limit_seconds": 21_600,
            "cpu_count": 16,
            "execution_timeout_seconds": 21_600,
            "finalize_reserve_seconds": 900,
            "gpu_count": 1,
            "initial_drafts": 3,
            "max_replacement_drafts": UNBOUNDED_WITHIN_SIX_HOURS,
            "memory_gib": 64,
            "parallel_search_num": 1,
        }
    )
    budget["runtime"].update(
        {
            "solver_model_id": "gpt-5.6-sol-openai-compatible-solver",
            "solver_model_revision": "sha256:"
            + hashlib.sha256(
                f"{LLM_BASE_URL}|{LLM_MODEL}|chat-completions".encode("utf-8")
            ).hexdigest(),
        }
    )
    budget_hash = write_hashed(manifests / "budget.json", budget, "manifest_hash")

    evaluators = read_json(base / "evaluators.json")
    evaluator_root = "/workspace/experiment-end2end-leaf-official-evaluator-v135"
    evaluators["formal_releases_root"] = evaluator_root
    evaluators["tasks"]["leaf-classification"].update(
        {
            "release_root": f"{evaluator_root}/leaf-classification/release",
            "terminal_evaluator_spec": "deferred official Kaggle v1",
        }
    )
    evaluator_hash = write_hashed(
        manifests / "evaluators.json", evaluators, "manifest_hash"
    )

    # The compact v134 runtime retains the executable v23 system configs but
    # intentionally omits the old top-level manifest directory. Read the
    # authoritative labels/descriptions from the repository and hash only the
    # newly overlaid v135 configs into this release.
    old_systems = read_json(REPO / EXPERIMENT / "manifests/systems.json")
    system_rows = {
        str(row["system_id"]): dict(row) for row in old_systems["systems"]
    }
    systems = []
    for system_id in SYSTEM_IDS:
        row = system_rows[system_id]
        config = output / TARGET_SYSTEMS / f"{system_id}.yaml"
        row["config_path"] = f"systems_v135/{system_id}.yaml"
        row["config_sha256"] = sha256_file(config)
        if system_id == "dynamic_hybrid":
            row["label"] = "S5-v135-latest-dynamic"
            row["description"] = (
                "Latest v134 Replay Research dynamic retrieval/controller logic"
            )
        systems.append(row)
    systems_manifest = {
        "schema": "mlevolve_end2end_systems_manifest_v1",
        "experimental_axis": (
            "shared latest v135 runtime and GPT relay; dynamic_hybrid alone uses "
            "the latest Replay Research logic, while nine controls retain v23 logic"
        ),
        "system_count": len(systems),
        "systems": systems,
        "manifest_hash": "",
    }
    systems_hash = write_hashed(
        manifests / "systems.json", systems_manifest, "manifest_hash"
    )

    source_revision = git_head()
    source_lock = build_source_lock(
        output, source_revision=source_revision, manifests=TARGET_MANIFESTS
    )
    source_lock_hash = write_hashed(
        manifests / "source_lock.json", source_lock, "manifest_hash"
    )
    return {
        "budget_manifest_hash": budget_hash,
        "evaluators_manifest_hash": evaluator_hash,
        "memory_bundles_manifest_hash": read_json(
            manifests / "memory_bundles.json"
        )["manifest_hash"],
        "schemas_manifest_hash": read_json(manifests / "schemas.json")[
            "manifest_hash"
        ],
        "source_lock_manifest_hash": source_lock_hash,
        "systems_manifest_hash": systems_hash,
        "tasks_manifest_hash": read_json(manifests / "tasks.json")[
            "manifest_hash"
        ],
    }


def build_execution(bindings: Mapping[str, str]) -> dict[str, Any]:
    runs = []
    launch_order = (*BATCH_1, *BATCH_2)
    for position, system_id in enumerate(launch_order):
        row = {
            "task_id": "leaf-classification",
            "system_id": system_id,
            "seed": 1,
            "logical_run_id": (
                "e2e-pilot-leaf-ten-system-official-gpt56sol-v135__"
                f"leaf-classification__{system_id}__seed-1"
            ),
            "formal_result_eligible": True,
            "exploratory_pilot": True,
            "launch_position": position,
            "task_launch_position": position,
            "bindings": dict(bindings),
            "row_hash": "",
        }
        row["row_hash"] = payload_hash(row, "row_hash")
        runs.append(row)
    manifest = {
        "schema": "mlevolve_end2end_execution_manifest_v1",
        "release_id": "end2end-leaf-ten-system-official-gpt56sol-v135",
        "kind": "pilot",
        "comparison_baseline_release_id": "end2end-agent-v23",
        "seed": 1,
        "task_ids": ["leaf-classification"],
        "system_ids": list(launch_order),
        "run_count": len(runs),
        "first_parallel_batch": list(BATCH_1),
        "second_parallel_batch": list(BATCH_2),
        "launch_order_randomization": (
            "user-directed two batches of five; latest dynamic included in batch 1"
        ),
        "formal_result_eligible": True,
        "exploratory_pilot": True,
        "statistical_significance_claim_allowed": False,
        "bindings": dict(bindings),
        "runs": runs,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = payload_hash(manifest, "manifest_hash")
    return manifest


def field_env(name: str, field_path: str) -> dict[str, Any]:
    return {"name": name, "valueFrom": {"fieldRef": {"fieldPath": field_path}}}


def build_job(
    *, manifest: Mapping[str, Any], index: int, system_id: str, batch: int
) -> dict[str, Any]:
    workload = f"mlevolve-leaf-gpt56sol-v135-{system_id.replace('_', '-')}"
    labels = {
        "app": "mlevolve-end2end",
        "experiment": EXPERIMENT_LABEL,
        "mlevolve.ai/system": system_id,
        "mlevolve.ai/batch": str(batch),
        "ecepxie.nrp/owner": "haoming",
        "app.kubernetes.io/managed-by": "codex-nrp-training",
    }
    resources = {
        "cpu": "16",
        "memory": "64Gi",
        "ephemeral-storage": "64Gi",
        "nvidia.com/a100": "1",
    }
    runner = (
        "set -euo pipefail; "
        "test \"${OPENAI_BASE_URL:-}\" = \"https://apizh.net/v1\"; "
        "test \"${OPENAI_MODEL:-}\" = \"gpt-5.6-sol\"; "
        "test -n \"${OPENAI_API_KEY:-}\"; "
        f"exec /usr/local/bin/python -u {CLUSTER_RUNTIME}/{EXPERIMENT}/run_assignment.py "
        f"--manifest {CLUSTER_RUNTIME}/{TARGET_MANIFESTS}/{EXECUTION_MANIFEST_NAME} "
        f"--index {index} --attempt 0 --output-root {OUTPUT_ROOT}"
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": workload,
            "namespace": "ecepxie",
            "labels": labels,
            "annotations": {
                "mlevolve.ai/launch-gate": "user-authorized",
                "mlevolve.ai/agent-wall-seconds": "21600",
                "mlevolve.ai/search-count-limit": "none-within-wall-clock",
                "mlevolve.ai/pending-time-excluded": "no-job-active-deadline",
                "mlevolve.ai/manifest-sha256": str(manifest["manifest_hash"]),
                "mlevolve.ai/runtime-source-lock-sha256": str(
                    manifest["bindings"]["source_lock_manifest_hash"]
                ),
            },
        },
        "spec": {
            "completions": 1,
            "parallelism": 1,
            "backoffLimit": 0,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 180,
                    "tolerations": [
                        {"key": "nvidia.com/gpu", "operator": "Exists"}
                    ],
                    "containers": [
                        {
                            "name": "end2end-runner",
                            "image": RUNTIME_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc"],
                            "args": [runner],
                            "envFrom": [{"secretRef": {"name": LLM_SECRET}}],
                            "env": [
                                field_env(
                                    "KUBERNETES_JOB_NAME",
                                    "metadata.labels['batch.kubernetes.io/job-name']",
                                ),
                                field_env(
                                    "KUBERNETES_JOB_UID",
                                    "metadata.labels['batch.kubernetes.io/controller-uid']",
                                ),
                                field_env("KUBERNETES_POD_NAME", "metadata.name"),
                                field_env("KUBERNETES_POD_UID", "metadata.uid"),
                                field_env("KUBERNETES_NODE_NAME", "spec.nodeName"),
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                                {
                                    "name": "PYTHONPATH",
                                    "value": f"{CLUSTER_RUNTIME}/mlevolve:{CLUSTER_RUNTIME}",
                                },
                                {
                                    "name": "MLEVOLVE_CONTAINER_IMAGE_REFERENCE",
                                    "value": RUNTIME_IMAGE,
                                },
                                {
                                    "name": "MLEVOLVE_SOLVER_BINDING_ID",
                                    "value": "gpt-5.6-sol-openai-compatible-solver",
                                },
                                {
                                    "name": "MLEVOLVE_SOLVER_MODEL_REVISION",
                                    "value": "sha256:"
                                    + hashlib.sha256(
                                        f"{LLM_BASE_URL}|{LLM_MODEL}|chat-completions".encode(
                                            "utf-8"
                                        )
                                    ).hexdigest(),
                                },
                            ],
                            "resources": {
                                "requests": dict(resources),
                                "limits": dict(resources),
                            },
                            "volumeMounts": [
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "shm", "mountPath": "/dev/shm"},
                            ],
                        }
                    ],
                    "volumes": [
                        {
                            "name": "workspace",
                            "persistentVolumeClaim": {"claimName": "haoming-storage"},
                        },
                        {
                            "name": "shm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"},
                        },
                    ],
                },
            },
        },
    }


def build(
    *, base_runtime: Path, output_runtime: Path, manifests_out: Path, jobs_out: Path
) -> dict[str, Any]:
    for path in (output_runtime, manifests_out, jobs_out):
        if path.exists():
            raise FileExistsError(f"fresh v135 output already exists: {path}")
    shutil.copytree(base_runtime.resolve(strict=True), output_runtime, symlinks=True)
    overlay_latest_release_files(output_runtime)
    bindings = build_components(output_runtime)
    execution = build_execution(bindings)
    runtime_manifests = output_runtime / TARGET_MANIFESTS
    execution_hash = write_hashed(
        runtime_manifests / EXECUTION_MANIFEST_NAME, execution, "manifest_hash"
    )
    shutil.copytree(runtime_manifests, manifests_out)

    jobs_out.mkdir(parents=True, exist_ok=False)
    for index, row in enumerate(execution["runs"]):
        system_id = str(row["system_id"])
        batch = 1 if system_id in BATCH_1 else 2
        job = build_job(
            manifest=execution, index=index, system_id=system_id, batch=batch
        )
        batch_dir = jobs_out / f"batch{batch}"
        write_json(batch_dir / f"{job['metadata']['name']}.yaml", job)

    source_lock = read_json(runtime_manifests / "source_lock.json")
    atomic = output_runtime / "mlevolve/agents/atomic_actuation.py"
    receipt = {
        "schema": "mlevolve_leaf_ten_system_gpt_runtime_build_v1",
        "status": "complete",
        "release_version": 135,
        "git_head": source_lock["git_head"],
        "runtime_root": str(output_runtime),
        "runtime_file_count": sum(1 for path in output_runtime.rglob("*") if path.is_file()),
        "source_lock_hash": source_lock["manifest_hash"],
        "source_lock_file_count": len(source_lock["files"]),
        "execution_manifest_hash": execution_hash,
        "atomic_actuation_sha256": sha256_file(atomic),
        "system_count": len(SYSTEM_IDS),
        "batch_1": list(BATCH_1),
        "batch_2": list(BATCH_2),
        "agent_time_limit_seconds": 21_600,
        "agent_steps_nonbinding_sentinel": UNBOUNDED_WITHIN_SIX_HOURS,
        "llm_model": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "llm_secret_reference": LLM_SECRET,
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--manifests-out", required=True, type=Path)
    parser.add_argument("--jobs-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = build(
        base_runtime=args.base_runtime,
        output_runtime=args.output_runtime,
        manifests_out=args.manifests_out,
        jobs_out=args.jobs_out,
    )
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
