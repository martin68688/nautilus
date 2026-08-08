#!/usr/bin/env python3
"""Build the frozen four-condition Spooky v39 launch packet.

The v39 packet keeps the v38 parser/metric-reconciliation source release and
changes only the task/system matrix: Dynamic plus three previously measured
competitor-style controllers.  It is intentionally separate from the
Leaf-only v38 packet and never rewrites existing manifests or run artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any



ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
V35 = ROOT / "manifests_v35"
V38 = ROOT / "manifests_v38"
SYSTEMS_V35 = ROOT / "systems_v35"
SYSTEMS_V38 = ROOT / "systems_v38"
SYSTEMS_V39 = ROOT / "systems_v39"
MANIFESTS = ROOT / "manifests_v39"
JOB_DIR = ROOT / "jobs"

RELEASE_ID = "end2end-spooky-metric-reconcile-router-v39"
CLUSTER_REPO = "/workspace/nautilus-exp-end2end-agent-v39"
CLUSTER_ROOT = f"{CLUSTER_REPO}/experiments/end2end_memory_systems_20260804"
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v39/runs"
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v39"
SYSTEM_IDS = [
    "dynamic_hybrid",
    "macla_style_port",
    "runforest_only",
    "gome_style_port",
]
TASK_ID = "spooky-author-identification"
SPOOKY_GRAPH = REPO / "paper-skills/hyper_memory/run_forest_graph.json"
SPOOKY_INDEX = REPO / "paper-skills/hyper_memory/run_forest_index.npz"
SPOOKY_MEMORY_ROOT = (
    "/workspace/experiment-end2end-memory-agent-v39/"
    "memory-spooky-v1/spooky-author-identification"
)


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: dict[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_system_configs() -> dict[str, str]:
    if not SYSTEMS_V39.exists():
        SYSTEMS_V39.mkdir(parents=True)
        shutil.copy2(SYSTEMS_V38 / "base.yaml", SYSTEMS_V39 / "base.yaml")
        shutil.copy2(SYSTEMS_V38 / "dynamic_hybrid.yaml", SYSTEMS_V39 / "dynamic_hybrid.yaml")
        for system_id in SYSTEM_IDS:
            if system_id == "dynamic_hybrid":
                continue
            shutil.copy2(
                SYSTEMS_V35 / f"{system_id}.yaml",
                SYSTEMS_V39 / f"{system_id}.yaml",
            )
    expected = {"base", *SYSTEM_IDS}
    actual = {path.stem for path in SYSTEMS_V39.glob("*.yaml")}
    if actual != expected:
        raise RuntimeError(f"v39 config set drift: expected {expected}, got {actual}")
    return {
        system_id: sha256_file(SYSTEMS_V39 / f"{system_id}.yaml")
        for system_id in ["base", *SYSTEM_IDS]
    }


def build_spooky_memory() -> dict[str, Any]:
    graph = read_json(SPOOKY_GRAPH)
    meta = graph.get("meta") or {}
    required_meta = {
        "schema": "hyperbolic_run_forest_memory_v1",
        "source_membership_verified": True,
        "leak_audited": True,
        "leak_verified": True,
        "paper_grade": True,
        "positive_admission_enforced": True,
    }
    for key, expected in required_meta.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"Spooky RunForest audit metadata mismatch: {key}")
    eligible = []
    for node in graph.get("nodes") or []:
        audit = node.get("leakage_audit") or {}
        metric = node.get("metric")
        if (
            node.get("type") == "RunNode"
            and node.get("task") == TASK_ID
            and node.get("is_buggy") is False
            and node.get("is_valid") is True
            and isinstance(metric, (int, float))
            and not isinstance(metric, bool)
            and audit.get("status") == "clean"
            and audit.get("memory_disposition") == "positive_eligible"
            and audit.get("paper_grade_eligible") is True
            and audit.get("rank_eligible") is True
        ):
            eligible.append(node)
    if not eligible:
        raise RuntimeError("Spooky RunForest has no clean positive-eligible node")
    best = min(eligible, key=lambda node: float(node["metric"]))
    graph_sha = sha256_file(SPOOKY_GRAPH)
    index_sha = sha256_file(SPOOKY_INDEX)
    bundle = {
        "schema": "memory_bundle_manifest_v1",
        "bundle_id": "end2end-spooky-direct-runforest-v1",
        "bundle_version": "v1",
        "parent_bundle": None,
        "authority_policy_version": "experiment_effectiveness_offline_freeze_v1",
        "certification_level": "legacy_uncertified",
        "source_graph_manifest_schema": meta["schema"],
        "source_archive_sha256": str(meta.get("allowlist_hash") or ""),
        "source_task_ids": sorted(
            {
                str(node.get("task"))
                for node in graph.get("nodes") or []
                if str(node.get("task") or "")
            }
        ),
        "target_task_id": TASK_ID,
        "graph_hashes": {"runforest": graph_sha},
        "index_hashes": {"runforest": index_sha},
        "build_report": "runforest/graph.json",
        "created_at": "2026-08-08T10:30:00Z",
        "artifact_hashes": {
            "runforest/graph.json": graph_sha,
            "runforest/index.npz": index_sha,
        },
        "manifest_sha256": "",
    }
    bundle["manifest_sha256"] = payload_hash(bundle, "manifest_sha256")
    stage = MANIFESTS / "spooky_memory_stage"
    stage.mkdir(parents=True, exist_ok=True)
    write_json(stage / "manifest.json", bundle)
    current = {
        "schema": "memory_bundle_current_v1",
        "bundle_path": "bundles/v1",
        "bundle_id": bundle["bundle_id"],
        "bundle_version": bundle["bundle_version"],
        "manifest_sha256": bundle["manifest_sha256"],
        "parent_bundle": None,
        "published_at": "2026-08-08T10:30:00Z",
        "pointer_sha256": "",
    }
    current["pointer_sha256"] = payload_hash(current, "pointer_sha256")
    write_json(stage / "CURRENT.json", current)
    return {
        "bundle": bundle,
        "current": current,
        "graph_sha256": graph_sha,
        "index_sha256": index_sha,
        "manifest_file_sha256": sha256_file(stage / "manifest.json"),
        "current_file_sha256": sha256_file(stage / "CURRENT.json"),
        "best": best,
        "positive_eligible_count": len(eligible),
    }


def build_components(
    config_hashes: dict[str, str], memory: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    for name in ("budget", "schemas"):
        components[name] = read_json(V35 / f"{name}.json")

    tasks = {
        "schema": "mlevolve_end2end_tasks_manifest_v1",
        "seed": 1,
        "exploratory_pilot": True,
        "task_count": 1,
        "tasks": [
            {
                "task_id": TASK_ID,
                "display_name": "Spooky",
                "terminal_metric": "log_loss",
                "direction": "minimize",
            }
        ],
        "manifest_hash": "",
    }
    tasks["manifest_hash"] = payload_hash(tasks, "manifest_hash")
    components["tasks"] = tasks

    evaluators = read_json(V35 / "evaluators.json")
    evaluators["tasks"] = {
        TASK_ID: {
            "direction": "minimize",
            "metric": "log_loss",
            "release_root": (
                "/workspace/experiment-c-formal-releases-r3/"
                "spooky-author-identification/release"
            ),
            "runtime_spec": "RUNTIME_SPEC.json",
            "terminal_evaluator_spec": (
                "transitively pinned by the aggregate release binding"
            ),
            "terminal_metric": "log_loss",
        }
    }
    evaluators["manifest_hash"] = ""
    evaluators["manifest_hash"] = payload_hash(evaluators, "manifest_hash")
    components["evaluators"] = evaluators

    best = memory["best"]
    memory_bundles = {
        "schema": "mlevolve_end2end_memory_bundle_manifest_v1",
        "verification_mode": "experiment_fast_nonblocking_v1",
        "production_binding_path": "",
        "production_binding_sha256": "",
        "source_graph_manifest_schema": "hyperbolic_run_forest_memory_v1",
        "source_graph_sha256": memory["graph_sha256"],
        "source_index_sha256": memory["index_sha256"],
        "excluded_run_ids": [],
        "same_task_history_policy": (
            "enabled for Spooky; Dynamic pins the direction-aware best "
            "clean positive-eligible record"
        ),
        "task_bundles": {
            TASK_ID: {
                "bundle_id": memory["bundle"]["bundle_id"],
                "bundle_root": SPOOKY_MEMORY_ROOT,
                "bundle_version": memory["bundle"]["bundle_version"],
                "bundle_manifest_sha256": memory["bundle"]["manifest_sha256"],
                "bundle_manifest_file_sha256": memory["manifest_file_sha256"],
                "current_file_sha256": memory["current_file_sha256"],
                "graph_sha256": memory["graph_sha256"],
                "index_sha256": memory["index_sha256"],
                "memory_scope": "full_reviewed_multitask_with_spooky_same_task_history",
                "formal_child_publication": False,
                "same_task_history_enabled": True,
                "same_task_best_node_id": str(best["id"]),
                "protocol_ref": (
                    "stratified-log-loss-classification@1#"
                    "spooky-v39-fixed-holdout"
                ),
                "positive_eligible_count": memory["positive_eligible_count"],
            }
        },
        "manifest_hash": "",
    }
    memory_bundles["manifest_hash"] = payload_hash(
        memory_bundles, "manifest_hash"
    )
    components["memory_bundles"] = memory_bundles

    systems = read_json(V35 / "systems.json")
    systems["systems"] = [
        row for row in systems["systems"] if str(row["system_id"]) in SYSTEM_IDS
    ]
    systems["system_count"] = len(systems["systems"])
    systems["experimental_axis"] = (
        "first-batch system comparison; v38 metric reconciliation source with "
        "Dynamic plus three frozen competitor-style controllers"
    )
    for row in systems["systems"]:
        system_id = str(row["system_id"])
        row["config_path"] = f"systems_v39/{system_id}.yaml"
        row["config_sha256"] = config_hashes[system_id]
    systems["manifest_hash"] = ""
    systems["manifest_hash"] = payload_hash(systems, "manifest_hash")
    components["systems"] = systems

    source_lock = read_json(V38 / "source_lock.json")
    source_lock["release_id"] = RELEASE_ID
    existing = {
        str(row["path"]): str(row["sha256"])
        for row in source_lock.get("files") or []
    }
    for path in sorted(SYSTEMS_V39.glob("*.yaml")):
        existing[
            f"experiments/end2end_memory_systems_20260804/systems_v39/{path.name}"
        ] = sha256_file(path)
    source_lock["files"] = [
        {"path": path, "sha256": digest} for path, digest in sorted(existing.items())
    ]
    source_lock["manifest_hash"] = ""
    source_lock["manifest_hash"] = payload_hash(source_lock, "manifest_hash")
    components["source_lock"] = source_lock
    return components


def build_manifest(components: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bindings = {
        f"{name}_manifest_hash": payload["manifest_hash"]
        for name, payload in components.items()
    }
    runs = []
    for position, system_id in enumerate(SYSTEM_IDS):
        row = {
            "logical_run_id": (
                f"e2e-pilot-spooky-metric-reconcile-router-v39__"
                f"{TASK_ID}__{system_id}__seed-1"
            ),
            "task_id": TASK_ID,
            "system_id": system_id,
            "seed": 1,
            "launch_position": position,
            "task_launch_position": position,
            "formal_result_eligible": True,
            "exploratory_pilot": True,
            "bindings": bindings,
            "row_hash": "",
        }
        row["row_hash"] = payload_hash(row, "row_hash")
        runs.append(row)
    manifest = {
        "schema": "mlevolve_end2end_execution_manifest_v1",
        "release_id": RELEASE_ID,
        "comparison_baseline_release_id": "end2end-agent-v3",
        "kind": "pilot",
        "formal_result_eligible": True,
        "exploratory_pilot": True,
        "seed": 1,
        "statistical_significance_claim_allowed": False,
        "system_ids": SYSTEM_IDS,
        "task_ids": [TASK_ID],
        "run_count": len(runs),
        "launch_order_randomization": (
            "explicit first-batch order locked by v39 packet: "
            "dynamic, MACLA-style, RunForest-only, GOME-style"
        ),
        "bindings": bindings,
        "runs": runs,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = payload_hash(manifest, "manifest_hash")
    return manifest


def build_job(manifest: dict[str, Any]) -> dict[str, Any]:
    name = "mlevolve-e2e-spooky-metric-reconcile-pilot-v39"
    labels = {
        "app": "mlevolve-end2end",
        "experiment": EXPERIMENT_LABEL,
        "mlevolve.ai/workload": name,
        "ecepxie.nrp/owner": "haoming",
        "app.kubernetes.io/managed-by": "codex-nrp-training",
    }
    field_env = lambda name, field_path: {
        "name": name,
        "valueFrom": {"fieldRef": {"fieldPath": field_path}},
    }
    runtime_image = (
        "docker.io/haomingwang22/mlevolve@sha256:"
        "fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc"
    )
    resources = {
        "cpu": "16",
        "memory": "64Gi",
        "ephemeral-storage": "64Gi",
        "nvidia.com/a100": "1",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": "ecepxie",
            "labels": labels,
            "annotations": {
                "mlevolve.ai/launch-gate": "user-authorized",
                "mlevolve.ai/generated-not-submitted": "false",
                "mlevolve.ai/gpu-contract": "nvidia.com/a100=1 x 4 indexed workers",
                "mlevolve.ai/preserve-failed-index-artifacts": "true",
                "mlevolve.ai/per-index-deadline-seconds": "25200",
                "mlevolve.ai/global-deadline-seconds": "25200",
                "mlevolve.ai/manifest-sha256": manifest["manifest_hash"],
            },
        },
        "spec": {
            "completionMode": "Indexed",
            "completions": 4,
            "parallelism": 4,
            "backoffLimitPerIndex": 0,
            "maxFailedIndexes": 4,
            "activeDeadlineSeconds": 25200,
            "ttlSecondsAfterFinished": 86400,
            "template": {
                "metadata": {"labels": labels},
                "spec": {
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 120,
                    "tolerations": [
                        {"key": "nvidia.com/gpu", "operator": "Exists"}
                    ],
                    "containers": [
                        {
                            "name": "end2end-runner",
                            "image": runtime_image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/usr/local/bin/python", "-u"],
                            "args": [
                                f"{CLUSTER_ROOT}/run_assignment.py",
                                "--manifest",
                                f"{CLUSTER_ROOT}/manifests_v39/spooky_pilot_manifest.json",
                                "--output-root",
                                OUTPUT_ROOT,
                            ],
                            "envFrom": [
                                {"secretRef": {"name": "prevalence-audit-deepseek-r1"}}
                            ],
                            "env": [
                                field_env(
                                    "JOB_COMPLETION_INDEX",
                                    "metadata.annotations['batch.kubernetes.io/job-completion-index']",
                                ),
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
                                    "value": f"{CLUSTER_REPO}/mlevolve",
                                },
                                {
                                    "name": "MLEVOLVE_CONTAINER_IMAGE_REFERENCE",
                                    "value": runtime_image,
                                },
                                {
                                    "name": "MLEVOLVE_SOLVER_BINDING_ID",
                                    "value": "deepseek-production-solver",
                                },
                                {
                                    "name": "MLEVOLVE_SOLVER_MODEL_REVISION",
                                    "value": "sha256:6c72890187efc83ef04ac6527c8f22f823708d99c83b7f7b3393dfe27fe4efc6",
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


def build() -> dict[str, Any]:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    hashes = prepare_system_configs()
    memory = build_spooky_memory()
    components = build_components(hashes, memory)
    for name, payload in components.items():
        write_json(MANIFESTS / f"{name}.json", payload)
    manifest = build_manifest(components)
    write_json(MANIFESTS / "spooky_pilot_manifest.json", manifest)
    workload = build_job(manifest)
    job_path = JOB_DIR / "pilot-spooky-metric-reconcile-v39-job.yaml"
    # JSON is a strict subset of YAML and avoids depending on a local YAML
    # package; kubectl accepts this as a YAML/JSON manifest by content.
    job_path.write_text(
        json.dumps(workload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    packet = {
        "schema": "mlevolve_end2end_spooky_v39_launch_packet_v1",
        "status": "generated_not_submitted",
        "release_id": RELEASE_ID,
        "source_root": CLUSTER_REPO,
        "manifest": "manifests_v39/spooky_pilot_manifest.json",
        "manifest_sha256": manifest["manifest_hash"],
        "job": "jobs/pilot-spooky-metric-reconcile-v39-job.yaml",
        "systems": SYSTEM_IDS,
        "task": TASK_ID,
        "indexed_workers": 4,
        "gpu_contract": "4 x nvidia.com/a100, one worker per GPU",
        "checkpoint_policy": "new immutable attempts; preserve every outcome",
        "spooky_memory": {
            "bundle_root": SPOOKY_MEMORY_ROOT,
            "graph_sha256": memory["graph_sha256"],
            "index_sha256": memory["index_sha256"],
            "best_clean_same_task_node_id": str(memory["best"]["id"]),
            "best_clean_same_task_metric": float(memory["best"]["metric"]),
            "positive_eligible_count": memory["positive_eligible_count"],
        },
        "packet_hash": "",
    }
    packet["packet_hash"] = payload_hash(packet, "packet_hash")
    write_json(MANIFESTS / "launch_packet.json", packet)
    return packet


def check() -> None:
    if not MANIFESTS.is_dir():
        raise RuntimeError("v39 manifests are not built")
    for path in MANIFESTS.glob("*.json"):
        payload = read_json(path)
        field = "packet_hash" if path.name == "launch_packet.json" else "manifest_hash"
        if payload_hash(payload, field) != payload.get(field):
            raise RuntimeError(f"Self-hash mismatch: {path}")
    manifest = read_json(MANIFESTS / "spooky_pilot_manifest.json")
    if manifest["task_ids"] != [TASK_ID] or manifest["system_ids"] != SYSTEM_IDS:
        raise RuntimeError("v39 matrix drift")
    if manifest["run_count"] != 4:
        raise RuntimeError("v39 must contain exactly four indexed runs")
    job = read_json(JOB_DIR / "pilot-spooky-metric-reconcile-v39-job.yaml")
    if job["spec"]["completions"] != 4 or job["spec"]["parallelism"] != 4:
        raise RuntimeError("v39 Job is not four-way parallel")
    if job["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/a100"] != "1":
        raise RuntimeError("v39 GPU contract drift")
    stage = MANIFESTS / "spooky_memory_stage"
    bundle = read_json(stage / "manifest.json")
    current = read_json(stage / "CURRENT.json")
    if payload_hash(bundle, "manifest_sha256") != bundle["manifest_sha256"]:
        raise RuntimeError("v39 Spooky memory manifest hash mismatch")
    if payload_hash(current, "pointer_sha256") != current["pointer_sha256"]:
        raise RuntimeError("v39 Spooky CURRENT hash mismatch")
    if sha256_file(SPOOKY_GRAPH) != bundle["artifact_hashes"]["runforest/graph.json"]:
        raise RuntimeError("v39 Spooky graph hash drift")
    if sha256_file(SPOOKY_INDEX) != bundle["artifact_hashes"]["runforest/index.npz"]:
        raise RuntimeError("v39 Spooky index hash drift")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        packet = build()
        print(json.dumps(packet, ensure_ascii=False, indent=2))
