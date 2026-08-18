#!/usr/bin/env python3
"""Build an isolated v137 Dynamic-only smoke or full release.

The v135 Dynamic runtime is the frozen behavioral base.  v137 overlays only
the hardened OpenAI-compatible backend and multi-granular retrieval Judge.
Smoke and full releases deliberately use disjoint runtime, manifest, output,
evaluator, workload, and logical-run identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import build_leaf_ten_system_gpt_v135_runtime as v135


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/end2end_memory_systems_20260804")
BASE_MANIFESTS = EXPERIMENT / "manifests_v135"
BASE_SYSTEMS = EXPERIMENT / "systems_v135"
SYSTEM_ID = "dynamic_hybrid"

OVERLAY_FILES = (
    Path("mlevolve/llm/openai.py"),
    Path("mlevolve/agents/memory/multigranular_grep.py"),
)
TEST_FILES = (
    Path("tests/test_gpt_openai_compatible_config.py"),
    Path("tests/test_multigranular_grep_retrieval.py"),
    Path("tests/test_experiment_r_dynamic_routing.py"),
    Path("tests/test_l3_grep_search_agent.py"),
)


def identity(mode: str) -> dict[str, Any]:
    if mode not in {"smoke", "full"}:
        raise ValueError(f"unsupported mode: {mode}")
    suffix = f"v137-{mode}"
    manifest_suffix = f"v137_{mode}"
    kind = "smoke" if mode == "smoke" else "pilot"
    return {
        "mode": mode,
        "suffix": suffix,
        "kind": kind,
        "manifest_dir": EXPERIMENT / f"manifests_{manifest_suffix}",
        "system_dir": EXPERIMENT / f"systems_{manifest_suffix}",
        "execution_name": f"leaf_dynamic_{mode}_manifest.json",
        "release_id": f"end2end-leaf-dynamic-retrieval-gpt56sol-{suffix}",
        "cluster_runtime": f"/workspace/nautilus-exp-end2end-agent-{suffix}",
        "output_root": f"/workspace/experiment-end2end-memory-agent-{suffix}/runs",
        "evaluator_root": f"/workspace/experiment-end2end-leaf-official-evaluator-{suffix}",
        "experiment_label": f"experiment-end2end-memory-agent-{suffix}",
        "workload": f"mlevolve-leaf-gpt56sol-{suffix}-dynamic-hybrid",
        "stager": f"mlevolve-leaf-gpt56sol-{suffix}-stager",
        "logical_run_id": (
            f"e2e-{mode}-leaf-dynamic-retrieval-official-gpt56sol-{suffix}__"
            "leaf-classification__dynamic_hybrid__seed-1"
        ),
    }


def copy_release_inputs(output: Path) -> None:
    for relative in (*OVERLAY_FILES, *TEST_FILES):
        v135.copy_file(REPO / relative, output / relative)


def write_dynamic_config(output: Path, spec: Mapping[str, Any]) -> Path:
    target = output / spec["system_dir"] / "dynamic_hybrid.yaml"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_text(
        "\n".join(
            [
                "# v137 changes only the OpenAI-compatible and multigranular-retrieval code.",
                "# All Dynamic controller, Replay, validation, and fusion semantics inherit v135.",
                "extends: ../systems_v135/dynamic_hybrid.yaml",
                "",
                "agent:",
                "  draft_role_policy:",
                "    replay_targets_path: "
                f"{spec['cluster_runtime']}/{spec['manifest_dir']}/leaf_official_replay_targets.json",
                "",
                "external_skill_memory:",
                "  transition_evidence_capsules_path: "
                f"{spec['cluster_runtime']}/{EXPERIMENT}/transition_evidence_v122/transition_evidence_capsules.json",
                "",
                "run_identity:",
                "  memory_version: "
                f"leaf_dynamic_retrieval_contract_hardening_gpt56sol_{spec['suffix'].replace('-', '_')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return target


def build_components(output: Path, spec: Mapping[str, Any]) -> dict[str, str]:
    base = output / BASE_MANIFESTS
    manifests = output / spec["manifest_dir"]
    manifests.mkdir(parents=True, exist_ok=False)
    for name in (
        "memory_bundles.json",
        "schemas.json",
        "tasks.json",
        "leaf_official_replay_targets.json",
    ):
        v135.copy_file(base / name, manifests / name)

    budget = v135.read_json(base / "budget.json")
    if spec["mode"] == "smoke":
        budget["smoke"].update(
            {
                "agent_steps": 5,
                "cpu_count": 16,
                "gpu_count": 1,
                "memory_gib": 64,
                "parallel_search_num": 1,
            }
        )
    else:
        budget["pilot"].update(
            {
                "agent_steps": v135.UNBOUNDED_WITHIN_SIX_HOURS,
                "agent_time_limit_seconds": 21_600,
                "cpu_count": 16,
                "execution_timeout_seconds": 21_600,
                "finalize_reserve_seconds": 900,
                "gpu_count": 1,
                "max_replacement_drafts": v135.UNBOUNDED_WITHIN_SIX_HOURS,
                "memory_gib": 64,
                "parallel_search_num": 1,
            }
        )
    budget_hash = v135.write_hashed(
        manifests / "budget.json", budget, "manifest_hash"
    )

    evaluators = v135.read_json(base / "evaluators.json")
    evaluators["formal_releases_root"] = spec["evaluator_root"]
    evaluators["tasks"]["leaf-classification"].update(
        {
            "release_root": f"{spec['evaluator_root']}/leaf-classification/release",
            "terminal_evaluator_spec": "deferred official Kaggle v1",
        }
    )
    evaluators_hash = v135.write_hashed(
        manifests / "evaluators.json", evaluators, "manifest_hash"
    )

    previous = v135.read_json(base / "systems.json")
    source_row = next(
        dict(row)
        for row in previous["systems"]
        if row["system_id"] == SYSTEM_ID
    )
    config = output / spec["system_dir"] / "dynamic_hybrid.yaml"
    source_row.update(
        {
            "config_path": f"{spec['system_dir']}/dynamic_hybrid.yaml",
            "config_sha256": v135.sha256_file(config),
            "label": f"S5-{spec['suffix']}-dynamic-retrieval-contract-hardening",
            "description": (
                "Frozen v135 Dynamic semantics with temperature precedence, local "
                "tool-schema validation, opaque Judge refs, Host stage fit, and "
                "stage-aware fallback over the preserved live Search pool"
            ),
        }
    )
    systems = {
        "schema": "mlevolve_end2end_systems_manifest_v1",
        "experimental_axis": (
            "Dynamic-only v137 contract hardening over the frozen v135 Dynamic runtime"
        ),
        "system_count": 1,
        "systems": [source_row],
        "manifest_hash": "",
    }
    systems_hash = v135.write_hashed(
        manifests / "systems.json", systems, "manifest_hash"
    )

    excluded = {
        (spec["manifest_dir"] / "source_lock.json").as_posix(),
        (spec["manifest_dir"] / spec["execution_name"]).as_posix(),
        "SOURCE_FILES.sha256",
    }
    files = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        files.append({"path": relative, "sha256": v135.sha256_file(path)})
    source_lock = {
        "schema": "mlevolve_end2end_source_lock_v1",
        "release_id": spec["release_id"],
        "git_head": v135.git_head(),
        "git_head_is_not_sufficient_identity": True,
        "complete_runtime_file_hash_lock": True,
        "control_file_exclusions": sorted(excluded),
        "overlay_scope": [path.as_posix() for path in OVERLAY_FILES],
        "files": files,
        "manifest_hash": "",
    }
    source_lock_hash = v135.write_hashed(
        manifests / "source_lock.json", source_lock, "manifest_hash"
    )
    return {
        "budget_manifest_hash": budget_hash,
        "evaluators_manifest_hash": evaluators_hash,
        "memory_bundles_manifest_hash": v135.read_json(
            manifests / "memory_bundles.json"
        )["manifest_hash"],
        "schemas_manifest_hash": v135.read_json(manifests / "schemas.json")[
            "manifest_hash"
        ],
        "source_lock_manifest_hash": source_lock_hash,
        "systems_manifest_hash": systems_hash,
        "tasks_manifest_hash": v135.read_json(manifests / "tasks.json")[
            "manifest_hash"
        ],
    }


def build_execution(
    spec: Mapping[str, Any], bindings: Mapping[str, str]
) -> dict[str, Any]:
    row = {
        "task_id": "leaf-classification",
        "system_id": SYSTEM_ID,
        "seed": 1,
        "logical_run_id": spec["logical_run_id"],
        "formal_result_eligible": spec["mode"] == "full",
        "exploratory_pilot": True,
        "launch_position": 0,
        "task_launch_position": 0,
        "bindings": dict(bindings),
        "row_hash": "",
    }
    row["row_hash"] = v135.payload_hash(row, "row_hash")
    manifest = {
        "schema": "mlevolve_end2end_execution_manifest_v1",
        "release_id": spec["release_id"],
        "kind": spec["kind"],
        "comparison_baseline_release_id": "end2end-leaf-ten-system-official-gpt56sol-v135",
        "seed": 1,
        "task_ids": ["leaf-classification"],
        "system_ids": [SYSTEM_ID],
        "run_count": 1,
        "first_parallel_batch": [SYSTEM_ID],
        "second_parallel_batch": [],
        "launch_order_randomization": "single Dynamic-only condition",
        "formal_result_eligible": spec["mode"] == "full",
        "exploratory_pilot": True,
        "statistical_significance_claim_allowed": False,
        "bindings": dict(bindings),
        "runs": [row],
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = v135.payload_hash(manifest, "manifest_hash")
    return manifest


def field_env(name: str, field_path: str) -> dict[str, Any]:
    return {"name": name, "valueFrom": {"fieldRef": {"fieldPath": field_path}}}


def build_job(
    spec: Mapping[str, Any], manifest: Mapping[str, Any], budget: Mapping[str, Any]
) -> dict[str, Any]:
    labels = {
        "app": "mlevolve-end2end",
        "experiment": spec["experiment_label"],
        "mlevolve.ai/system": SYSTEM_ID,
        "mlevolve.ai/release-mode": spec["mode"],
        "ecepxie.nrp/owner": "haoming",
        "app.kubernetes.io/managed-by": "codex-nrp-training",
    }
    resources = {
        "cpu": str(budget["cpu_count"]),
        "memory": f"{budget['memory_gib']}Gi",
        "ephemeral-storage": "64Gi",
        "nvidia.com/a100": str(budget["gpu_count"]),
    }
    runner = (
        "set -euo pipefail; "
        "test \"${OPENAI_BASE_URL:-}\" = \"https://apizh.net/v1\"; "
        "test \"${OPENAI_MODEL:-}\" = \"gpt-5.6-sol\"; "
        "test -n \"${OPENAI_API_KEY:-}\"; "
        f"exec /usr/local/bin/python -u {spec['cluster_runtime']}/{EXPERIMENT}/run_assignment.py "
        f"--manifest {spec['cluster_runtime']}/{spec['manifest_dir']}/{spec['execution_name']} "
        f"--index 0 --attempt 0 --output-root {spec['output_root']}"
    )
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": spec["workload"],
            "namespace": "ecepxie",
            "labels": labels,
            "annotations": {
                "mlevolve.ai/launch-gate": "user-authorized",
                "mlevolve.ai/release-mode": spec["mode"],
                "mlevolve.ai/agent-wall-seconds": str(
                    budget["agent_time_limit_seconds"]
                ),
                "mlevolve.ai/search-count-limit": (
                    "five-agent-steps" if spec["mode"] == "smoke"
                    else "none-within-wall-clock"
                ),
                "mlevolve.ai/pending-time-excluded": "no-job-active-deadline",
                "mlevolve.ai/manifest-sha256": manifest["manifest_hash"],
                "mlevolve.ai/runtime-source-lock-sha256": manifest["bindings"][
                    "source_lock_manifest_hash"
                ],
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
                    "tolerations": [{"key": "nvidia.com/gpu", "operator": "Exists"}],
                    "containers": [
                        {
                            "name": "end2end-runner",
                            "image": v135.RUNTIME_IMAGE,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc"],
                            "args": [runner],
                            "envFrom": [{"secretRef": {"name": v135.LLM_SECRET}}],
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
                                    "value": (
                                        f"{spec['cluster_runtime']}/mlevolve:"
                                        f"{spec['cluster_runtime']}"
                                    ),
                                },
                                {
                                    "name": "MLEVOLVE_CONTAINER_IMAGE_REFERENCE",
                                    "value": v135.RUNTIME_IMAGE,
                                },
                                {
                                    "name": "MLEVOLVE_SOLVER_BINDING_ID",
                                    "value": "gpt-5.6-sol-openai-compatible-solver",
                                },
                                {
                                    "name": "MLEVOLVE_SOLVER_MODEL_REVISION",
                                    "value": "sha256:"
                                    + hashlib.sha256(
                                        f"{v135.LLM_BASE_URL}|{v135.LLM_MODEL}|chat-completions".encode()
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


def build_stager(spec: Mapping[str, Any]) -> dict[str, Any]:
    labels = {
        "app": "mlevolve-end2end-stager",
        "experiment": f"{spec['experiment_label']}-stager",
        "ecepxie.nrp/owner": "haoming",
        "app.kubernetes.io/managed-by": "codex-nrp-training",
    }
    marker = (
        f"/workspace/experiment-end2end-memory-agent-{spec['suffix']}/"
        "staging/STAGING_COMPLETE"
    )
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": spec["stager"],
            "namespace": "ecepxie",
            "labels": labels,
        },
        "spec": {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 30,
            "containers": [
                {
                    "name": "stager",
                    "image": v135.RUNTIME_IMAGE,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/bash", "-lc"],
                    "args": [
                        f"set -euo pipefail; marker={marker}; "
                        "while [ ! -f \"$marker\" ]; do sleep 2; done"
                    ],
                    "resources": {
                        "requests": {"cpu": "100m", "memory": "256Mi"},
                        "limits": {"cpu": "1", "memory": "1Gi"},
                    },
                    "volumeMounts": [{"name": "workspace", "mountPath": "/workspace"}],
                }
            ],
            "volumes": [
                {
                    "name": "workspace",
                    "persistentVolumeClaim": {"claimName": "haoming-storage"},
                }
            ],
        },
    }


def build(
    *,
    mode: str,
    base_runtime: Path,
    output_runtime: Path,
    manifests_out: Path,
    systems_out: Path,
    jobs_out: Path,
) -> dict[str, Any]:
    spec = identity(mode)
    for path in (output_runtime, manifests_out, systems_out, jobs_out):
        if path.exists():
            raise FileExistsError(f"fresh v137 output already exists: {path}")
    shutil.copytree(base_runtime.resolve(strict=True), output_runtime, symlinks=True)
    copy_release_inputs(output_runtime)
    dynamic = write_dynamic_config(output_runtime, spec)
    bindings = build_components(output_runtime, spec)
    execution = build_execution(spec, bindings)
    runtime_manifests = output_runtime / spec["manifest_dir"]
    execution_hash = v135.write_hashed(
        runtime_manifests / spec["execution_name"], execution, "manifest_hash"
    )
    shutil.copytree(runtime_manifests, manifests_out)
    shutil.copytree(output_runtime / spec["system_dir"], systems_out)

    jobs_out.mkdir(parents=True, exist_ok=False)
    budget_payload = v135.read_json(runtime_manifests / "budget.json")
    budget = budget_payload[spec["kind"]]
    job = build_job(spec, execution, budget)
    stager = build_stager(spec)
    v135.write_json(jobs_out / f"{job['metadata']['name']}.yaml", job)
    v135.write_json(jobs_out / f"{stager['metadata']['name']}.yaml", stager)

    source_lock = v135.read_json(runtime_manifests / "source_lock.json")
    return {
        "schema": "mlevolve_leaf_dynamic_retrieval_v137_runtime_build_v1",
        "status": "complete",
        "mode": mode,
        "release_id": spec["release_id"],
        "git_head": source_lock["git_head"],
        "runtime_root": str(output_runtime),
        "cluster_runtime": spec["cluster_runtime"],
        "output_root": spec["output_root"],
        "evaluator_root": spec["evaluator_root"],
        "experiment_label": spec["experiment_label"],
        "logical_run_id": spec["logical_run_id"],
        "job_name": spec["workload"],
        "stager_name": spec["stager"],
        "source_lock_hash": source_lock["manifest_hash"],
        "source_lock_file_count": len(source_lock["files"]),
        "execution_manifest_hash": execution_hash,
        "openai_sha256": v135.sha256_file(output_runtime / OVERLAY_FILES[0]),
        "multigranular_grep_sha256": v135.sha256_file(
            output_runtime / OVERLAY_FILES[1]
        ),
        "dynamic_config_sha256": v135.sha256_file(dynamic),
        "agent_steps": budget["agent_steps"],
        "agent_time_limit_seconds": budget["agent_time_limit_seconds"],
        "max_replacement_drafts": budget["max_replacement_drafts"],
        "llm_model": v135.LLM_MODEL,
        "llm_base_url": v135.LLM_BASE_URL,
        "llm_secret_reference": v135.LLM_SECRET,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--base-runtime", required=True, type=Path)
    parser.add_argument("--output-runtime", required=True, type=Path)
    parser.add_argument("--manifests-out", required=True, type=Path)
    parser.add_argument("--systems-out", required=True, type=Path)
    parser.add_argument("--jobs-out", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    receipt = build(
        mode=args.mode,
        base_runtime=args.base_runtime,
        output_runtime=args.output_runtime,
        manifests_out=args.manifests_out,
        systems_out=args.systems_out,
        jobs_out=args.jobs_out,
    )
    v135.write_json(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
