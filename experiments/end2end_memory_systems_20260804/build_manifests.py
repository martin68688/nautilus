#!/usr/bin/env python3
"""Deterministically build and validate the frozen End2End launch packet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CLUSTER_REPO = Path("/workspace/nautilus-exp-end2end")
CLUSTER_ROOT = CLUSTER_REPO / "experiments" / "end2end_memory_systems_20260804"
MANIFESTS = ROOT / "manifests"
SYSTEM_DIR = ROOT / "systems"
JOB_DIR = ROOT / "jobs"
SCHEMA_DIR = ROOT / "schemas"
SEED = 1
RELEASE_ID = "end2end-agent-v1"
OUTPUT_ROOT = "/workspace/experiment-end2end-agent-runs-v1"
SYSTEMS = (
    ("S0", "no_memory", "internal", "Bundle-bound zero Prompt exposure"),
    ("S1", "flat_retrieval", "internal", "Mixed global relevance Top-6"),
    ("S2", "sop_only", "internal", "SOP Top-6"),
    ("S3", "runforest_only", "internal", "RunForest Top-6"),
    ("S4", "static_hybrid", "internal", "3 SOP / 3 RunForest"),
    ("S5", "dynamic_hybrid", "internal", "Draft 4/2, Improve 3/3, Debug 2/4"),
    ("S6", "reversed_router", "internal", "Draft 2/4, Improve 3/3, Debug 4/2"),
    ("C1", "gome_style_port", "competitor_style_port", "Verified Success Memory cards"),
    ("C2", "macla_style_port", "competitor_style_port", "Beta reliability utility"),
    ("C3", "rcr_router_style_port", "competitor_style_port", "Role/stage/recency token routing"),
)
TASKS = (
    ("aerial-cactus-identification", "Aerial", "roc_auc", "maximize"),
    ("leaf-classification", "Leaf", "log_loss", "minimize"),
    ("denoising-dirty-documents", "Denoising", "rmse", "minimize"),
    ("new-york-city-taxi-fare-prediction", "Taxi", "rmse", "minimize"),
)
MEMORY = {
    "aerial-cactus-identification": {
        "bundle_id": "experiment-r-aerial-image-task-heldout-v2",
        "bundle_root": "/workspace/experiment-r-dev-r1/memory/authority-r3/aerial-cactus-identification",
        "bundle_version": "v2",
        "bundle_manifest_sha256": "e997b016052d434791ee16c95f537ccaff28fa433340a3d6473e4dea1d56649b",
        "graph_sha256": "896ecf8bdd27c274f007058fcf364f75246ff8777c21d89dae6fb9594549fba8",
        "index_sha256": "48b207e54185c06826645b79424e5d1ab451e8459ff1244872fec0f461deff13",
        "current_file_sha256": "ef9dea34870582fae850a213123e0104758f3fd6f18edf4a1380fc4eee8b1a69",
        "protocol_ref": "stratified-roc-auc-classification@1#799541fb3a05e1759d76887ae970f061573393a380b185cdfb60ef6f2172a9b1",
    },
    "leaf-classification": {
        "bundle_id": "experiment-r-leaf-tabular-task-heldout-v2",
        "bundle_root": "/workspace/experiment-r-dev-r1/memory/authority-r3/leaf-classification",
        "bundle_version": "v2",
        "bundle_manifest_sha256": "26768ef82cd381ee1bb69ebc0ee8789de99c2b4cd427ac182d0379eb8c0062db",
        "graph_sha256": "32922b7fcea56a2ca69c1055585edf38dfc7feebb30b9ae78637987b8819e86a",
        "index_sha256": "496d858fe075bb58ffd398336112a70a85e32d8c1a0266af2117c497b98cd3b1",
        "current_file_sha256": "2e3cef21239f41eec4e1ceb157a0f5631a6f75082fc5cc686ccc4d5673c4f2a4",
        "protocol_ref": "stratified-log-loss-classification@1#a7601be6346021743e01ab144b38e457109ca3ddc701a4aaae9cd562203d79d5",
    },
    "denoising-dirty-documents": {
        "bundle_id": "experiment-r-denoising-image-task-heldout-v2",
        "bundle_root": "/workspace/experiment-r-dev-r1/memory/authority-r3/denoising-dirty-documents",
        "bundle_version": "v2",
        "bundle_manifest_sha256": "273b6b65cf71e29469694f726d2445ba71d95bd2976d787193124acda03901f5",
        "graph_sha256": "ebfb855162915a2dd61c31f539f1bc9a94a62501dff2e124b6d4fe32f8021e53",
        "index_sha256": "cd938ddc6a0699b7adcf5bf5aa745b966f176098551ca61d4c939fbcbfdc66b9",
        "current_file_sha256": "ae5d62c75a0d7fc44e8df3ced30491ad2d7dd05cd647e43a23cc28b9232685c4",
        "protocol_ref": "deterministic-random-regression@1#ba944032aa890aa2b742030895b670a7780a7698341e249d57917d1cd898695d",
    },
    "new-york-city-taxi-fare-prediction": {
        "bundle_id": "experiment-r-taxi-tabular-task-heldout-v2",
        "bundle_root": "/workspace/experiment-r-dev-r1/memory/authority-r3/new-york-city-taxi-fare-prediction",
        "bundle_version": "v2",
        "bundle_manifest_sha256": "58e7504411432430bf37d1ef0c18bf2c49cad116400084f7693062c8552ff499",
        "graph_sha256": "e500ba241e10a45c1512c7fb0b64e89cec7cb0aa6bb3e5d73144d2b289e24487",
        "index_sha256": "864fab630f1bb512b215671675446af6c344b7bef8df6595cea96ebe2e881726",
        "current_file_sha256": "de8c6ec2204cace0dad06e97bd9d2c4c597a6756bad8362da704f65b434d7864",
        "protocol_ref": "chronological-regression@1#bfc61957b422df5cf09dcb37cffe06aae2ccd2b11db4fee0721b90a2bc6dbf04",
    },
}
HOST = {
    "aerial-cactus-identification": {
        "binding_file_sha256": "43ea83ecebed8977c61eecd81008b9ba43b315f2264d4e93fa69d006d606857d",
        "binding_hash": "522a0da16a012fc64b0a220a9bc0172b9d5c9811c46a50be18bf92160d4288e6",
        "contract_hash": "3313f15076012774921d2bf4de7ebeafc2845e5a452f7ba87701a08382d45df4",
        "data_view_manifest_hash": "a8cc51d255986f1e21c34aa7c13aaad21e6e17eec8506e22e7e0e9cede38caeb",
    },
    "leaf-classification": {
        "binding_file_sha256": "7ac25c692ff86334fb7017fe271747cf72a039ab5979e487c692990dce5537fb",
        "binding_hash": "6932bc4662099620901c57960eb07885559d544c1ccbb7d18e9e754517a75dff",
        "contract_hash": "84c1e5764b64abede9be3e841f2a37d3043c5f1bf18e6190e957a41c84dc4a09",
        "data_view_manifest_hash": "6073c8cd004c84aa048945eeeed408592bebf87581a5c6720b52e3f920daad5d",
    },
    "denoising-dirty-documents": {
        "binding_file_sha256": "2c1e6bcb4c1111819d6a050622a49885fd3da28ae8d0f33ffc798a170bab8868",
        "binding_hash": "83b589f91c8377fa735e2fd5ec01bbe09df8e3d3ed1b39324c9f79acbc32050c",
        "contract_hash": "0e1cbf9f1fda429434375c5a6b68cc4fafd873a140884438a226688c362bbc6e",
        "data_view_manifest_hash": "74e20f50b39534b415f7878f172e29f2d91e467a1daf70d00b683c79ed02e3d9",
    },
    "new-york-city-taxi-fare-prediction": {
        "binding_file_sha256": "d80d19cfd3be964e49bd4076daf8dcbb17ffe73ddf008e762ab25e52c59a0def",
        "binding_hash": "813b67bf410206eff6ffb6bae311147a41f99b6b18363e7e494d8ce347509731",
        "contract_hash": "aee52a4dbc04be10244003ae593f3c0276be3c75eca63331b864ae2f25eb3d1b",
        "data_view_manifest_hash": "3611c0b87c21a5995990a8b7b4604a7b790daf491bc95411d0a768e0382da6a7",
    },
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(payload: dict[str, Any], field: str = "manifest_hash") -> dict[str, Any]:
    payload[field] = payload_hash(payload, field)
    return payload


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def base_config() -> str:
    return """extends: ../../../mlevolve/config/config_prevalence_audit_20260729_host_enforce.yaml

methodology_kb_path: ""
methodology_dynamic: false

agent:
  use_global_memory: false
  search:
    num_gpus: 1
    parallel_search_num: 1
    num_drafts: 3
  draft_role_policy:
    enabled: false
    roles: []
  protocol_repair:
    enabled: false

adoption_tracking:
  enable: true
  enable_analysis: false
  judge_mode: keyword

adoption_verifier:
  enabled: true
  mode: enforce
  model: ""
  temperature: 0.0
  max_tokens: 4096
  max_contracts_per_call: 8
  max_code_chars: 120000
  require_signed_trace: true

prospective_audit:
  enabled: true
  allow_pending_counterfactual: true

external_skill_memory:
  enable: true
  bundle_root: ""
  current_pointer_path: CURRENT.json
  session_overlay_path: ""
  graph_path: ""
  index_path: ""
  text_model_path: ""
  source_name: run_forest_stage_hybrid_memory
  mode: run_forest_stage_hybrid
  scoring_mode: flat_twin
  enable_agentic: false
  top_k: 6
  max_chars: 0
  include_draft: true
  include_improve: true
  include_evolution: true
  include_debug: true
  include_fusion: true
  retrieval_control: stage_hybrid
  visibility_token_budget: 4096
  end2end_memory_system: ""
  end2end_prompt_token_budget: 1536
  end2end_candidate_pool_limit: 12
  excluded_run_ids:
    - "20260701_180146"
    - "20260701_145201"
    - "20260701_145250"
    - "20260516_125444"
    - "20260701_155016"
    - "20260510_025317"
    - "20260701_180038"

evaluation_authority:
  require_bound_bundle: true

run_identity:
  schema: mlevolve_run_identity_v1
  experiment_group: experiment_end2end_memory_systems_pilot_v1
  baseline_reference_group: experiment_end2end_no_memory_v1
  memory_enabled: true
  memory_system: ""
  memory_version: experiment_r_task_heldout_production_v2
  identity_source: end2end_frozen_manifest_v1
"""


def system_config(system_id: str) -> str:
    return f"""extends: base.yaml

external_skill_memory:
  end2end_memory_system: {system_id}

run_identity:
  memory_system: {system_id}
  system_id: {system_id}
"""


def write_system_configs() -> None:
    SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
    (SYSTEM_DIR / "base.yaml").write_text(base_config(), encoding="utf-8")
    for _label, system_id, _kind, _description in SYSTEMS:
        (SYSTEM_DIR / f"{system_id}.yaml").write_text(
            system_config(system_id), encoding="utf-8"
        )


def source_lock() -> dict[str, Any]:
    files = []
    for path in sorted((REPO / "mlevolve").rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(REPO)
        if any(part in {"runs", "data", "__pycache__"} for part in relative.parts):
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".json"}:
            continue
        files.append({"path": relative.as_posix(), "sha256": sha256_file(path)})
    for path in (
        sorted(ROOT.glob("*.py"))
        + sorted(SYSTEM_DIR.glob("*.yaml"))
        + sorted(SCHEMA_DIR.glob("*.json"))
    ):
        relative = path.relative_to(REPO)
        files.append({"path": relative.as_posix(), "sha256": sha256_file(path)})
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    return finalize(
        {
            "schema": "mlevolve_end2end_source_lock_v1",
            "git_head": git_head,
            "git_head_is_not_sufficient_identity": True,
            "complete_runtime_file_hash_lock": True,
            "files": files,
            "manifest_hash": "",
        }
    )


def component_manifests() -> dict[str, dict[str, Any]]:
    systems = []
    for label, system_id, kind, description in SYSTEMS:
        config = SYSTEM_DIR / f"{system_id}.yaml"
        limitation = None
        reference = None
        if system_id == "gome_style_port":
            limitation = "Not the full GOME multi-trace optimizer"
            reference = "https://arxiv.org/abs/2603.01692"
        elif system_id == "macla_style_port":
            limitation = "No full online contrastive refinement"
            reference = "https://arxiv.org/abs/2512.18950"
        elif system_id == "rcr_router_style_port":
            limitation = "Not the original multi-agent QA environment"
            reference = "https://arxiv.org/abs/2508.04903"
        systems.append(
            {
                "label": label,
                "system_id": system_id,
                "kind": kind,
                "description": description,
                "config_path": f"systems/{system_id}.yaml",
                "config_sha256": sha256_file(config),
                "limitation": limitation,
                "primary_reference": reference,
            }
        )
    systems_manifest = finalize(
        {
            "schema": "mlevolve_end2end_systems_manifest_v1",
            "system_count": 10,
            "sole_experimental_axis": "external_skill_memory.end2end_memory_system",
            "systems": systems,
            "manifest_hash": "",
        }
    )
    tasks_manifest = finalize(
        {
            "schema": "mlevolve_end2end_tasks_manifest_v1",
            "task_count": 4,
            "seed": SEED,
            "exploratory_pilot": True,
            "tasks": [
                {
                    "task_id": task_id,
                    "display_name": display,
                    "terminal_metric": metric,
                    "direction": direction,
                }
                for task_id, display, metric, direction in TASKS
            ],
            "manifest_hash": "",
        }
    )
    runtime = {
        "container_image": "docker.io/haomingwang22/mlevolve@sha256:fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc",
        "solver_model_id": "deepseek-production-solver",
        "solver_model_revision": "sha256:6c72890187efc83ef04ac6527c8f22f823708d99c83b7f7b3393dfe27fe4efc6",
        "provider_seed_supported": False,
        "rng_commitment_scope": "Python/NumPy/Torch local RNG state; no provider determinism claim",
        "gpu_type": "NVIDIA A100 family",
        "gpu_product_constraint": None,
        "gpu_resource_key": "nvidia.com/a100",
    }
    budget_manifest = finalize(
        {
            "schema": "mlevolve_end2end_budget_manifest_v1",
            "runtime": runtime,
            "shared_memory": {
                "raw_candidates_per_source": 12,
                "raw_candidate_max": 24,
                "top_k": 6,
                "prompt_token_budget": 1536,
                "token_counter": "whitespace_split_v1",
                "visibility_token_budget": 4096,
            },
            "adoption_verifier": {
                "enabled": True,
                "mode": "enforce",
                "model_source": "agent.feedback.model",
                "temperature": 0.0,
                "max_tokens_per_call": 4096,
                "max_contracts_per_call": 8,
                "max_code_chars": 120000,
                "require_signed_trace": True,
                "runtime_probe": "line_range_executed",
            },
            "pilot": {
                "agent_steps": 80,
                "agent_time_limit_seconds": 21600,
                "execution_timeout_seconds": 18000,
                "finalize_reserve_seconds": 900,
                "initial_drafts": 3,
                "max_replacement_drafts": 5,
                "parallel_search_num": 1,
                "gpu_count": 1,
                "cpu_count": 16,
                "memory_gib": 64,
            },
            "smoke": {
                "agent_steps": 2,
                "agent_time_limit_seconds": 1800,
                "execution_timeout_seconds": 900,
                "finalize_reserve_seconds": 120,
                "initial_drafts": 1,
                "max_replacement_drafts": 0,
                "parallel_search_num": 1,
                "gpu_count": 1,
                "cpu_count": 16,
                "memory_gib": 64,
            },
            "failure_policy": {
                "automatic_job_retry": False,
                "backoff_limit_per_index": 0,
                "preserve_all_attempts": True,
                "explicit_retry_only_for": "infrastructure",
                "score_imputation": False,
            },
            "manifest_hash": "",
        }
    )
    memory_manifest = finalize(
        {
            "schema": "mlevolve_end2end_memory_bundle_manifest_v1",
            "production_binding_path": "/workspace/experiment-r-dev-r1/memory/authority-r3/PRODUCTION_MEMORY_BINDING.json",
            "production_binding_sha256": "47e0d38a2183827f36ab44c9881c542690d154cd735fbfdb9c4de3019bf926f7",
            "task_bundles": MEMORY,
            "excluded_run_ids": [
                "20260701_180146", "20260701_145201", "20260701_145250",
                "20260516_125444", "20260701_155016", "20260510_025317",
                "20260701_180038",
            ],
            "host_bindings_root": "/workspace/experiment-r-dev-r1/host-protocol-formal-r2/bindings",
            "host_runtime_sdk_hash": "1084155c01632ba93c581949cdcf40fc8df372333fdf82a8c6d878a1d75b2375",
            "host_collector_public_key_ed25519": "qb++TPUVPeBugazDY22lXrAEOSFEaxK7uo72cWWXs0w=",
            "host_collector_public_key_sha256": "34a0b39b04a60dc781e9c5699a6771653613b7b582a096dd45f68155cb76f853",
            "host_task_bindings": HOST,
            "manifest_hash": "",
        }
    )
    evaluator_manifest = finalize(
        {
            "schema": "mlevolve_end2end_evaluator_manifest_v1",
            "formal_releases_root": "/workspace/experiment-c-formal-releases-r3",
            "aggregate_binding_path": "/workspace/experiment-c-formal-releases-r3/FORMAL_RELEASE_BINDING.json",
            "aggregate_binding_hash": "668896c08bd1c748ccb5a89220312daccae19ef4ad066db642518f00b9d67e47",
            "reuse_scope": "task data and evaluator assets only; no Exp-C source/config/system manifest",
            "timing": "after Agent exit and candidate-set freeze",
            "failed_terminal_score": None,
            "tasks": {
                task_id: {
                    "metric": metric,
                    "terminal_metric": metric,
                    "direction": direction,
                    "release_root": f"/workspace/experiment-c-formal-releases-r3/{task_id}/release",
                    "runtime_spec": "RUNTIME_SPEC.json",
                    "terminal_evaluator_spec": "transitively pinned by the aggregate release binding",
                }
                for task_id, _display, metric, direction in TASKS
            },
            "manifest_hash": "",
        }
    )
    schema_manifest = finalize(
        {
            "schema": "mlevolve_end2end_schema_manifest_v1",
            "schemas": [
                {
                    "path": f"schemas/{path.name}",
                    "sha256": sha256_file(path),
                }
                for path in sorted(SCHEMA_DIR.glob("*.json"))
            ],
            "manifest_hash": "",
        }
    )
    return {
        "systems": systems_manifest,
        "tasks": tasks_manifest,
        "budget": budget_manifest,
        "memory_bundles": memory_manifest,
        "evaluators": evaluator_manifest,
        "schemas": schema_manifest,
        "source_lock": source_lock(),
    }


def shuffled_system_ids(task_id: str) -> list[str]:
    digest = hashlib.sha256(
        f"{RELEASE_ID}|{task_id}|seed-{SEED}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    values = [system_id for _label, system_id, _kind, _description in SYSTEMS]
    rng.shuffle(values)
    return values


def execution_manifest(
    *, kind: str, components: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if kind == "smoke":
        task_ids = ["leaf-classification"]
        formal = False
        prefix = "e2e-smoke-agent-v1"
    else:
        task_ids = [task_id for task_id, _display, _metric, _direction in TASKS]
        formal = True
        prefix = "e2e-pilot-agent-v1"
    bindings = {
        f"{key}_manifest_hash": value["manifest_hash"]
        for key, value in components.items()
    }
    runs = []
    global_position = 0
    for task_id in task_ids:
        for task_position, system_id in enumerate(shuffled_system_ids(task_id)):
            logical_run_id = f"{prefix}__{task_id}__{system_id}__seed-{SEED}"
            row = {
                "logical_run_id": logical_run_id,
                "task_id": task_id,
                "system_id": system_id,
                "seed": SEED,
                "launch_position": global_position,
                "task_launch_position": task_position,
                "formal_result_eligible": formal,
                "exploratory_pilot": True,
                "bindings": bindings,
                "row_hash": "",
            }
            row["row_hash"] = payload_hash(row, "row_hash")
            runs.append(row)
            global_position += 1
    return finalize(
        {
            "schema": "mlevolve_end2end_execution_manifest_v1",
            "kind": kind,
            "formal_result_eligible": formal,
            "exploratory_pilot": True,
            "seed": SEED,
            "statistical_significance_claim_allowed": False,
            "system_ids": [system_id for _label, system_id, _kind, _desc in SYSTEMS],
            "task_ids": task_ids,
            "run_count": len(runs),
            "launch_order_randomization": "task-local SHA256-seeded deterministic shuffle",
            "bindings": bindings,
            "runs": runs,
            "manifest_hash": "",
        }
    )


def job(
    *, name: str, manifest_name: str, completions: int, task_id: str | None,
    active_deadline: int, parallelism: int, components: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    runtime = components["budget"]["runtime"]
    index_waves = (completions + parallelism - 1) // parallelism
    global_active_deadline = active_deadline * index_waves
    labels = {
        "app": "mlevolve-end2end",
        "experiment": "experiment-end2end-memory-agent-v1",
        "ecepxie.nrp/owner": "haoming",
        "app.kubernetes.io/managed-by": "codex-nrp-training",
    }
    args = [
        str(CLUSTER_ROOT / "run_assignment.py"),
        "--manifest", str(CLUSTER_ROOT / "manifests" / manifest_name),
        "--output-root", OUTPUT_ROOT,
    ]
    if manifest_name == "pilot_manifest.json":
        args.extend(
            [
                "--smoke-gate",
                f"{OUTPUT_ROOT}/SMOKE_GATE.json",
            ]
        )
    if task_id:
        args.extend(["--task", task_id])
    resources = {
        "cpu": "16",
        "memory": "64Gi",
        "ephemeral-storage": "64Gi",
        runtime["gpu_resource_key"]: "1",
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": "ecepxie",
            "labels": labels,
            "annotations": {
                "mlevolve.ai/launch-gate": "explicit-user-authorization-required",
                "mlevolve.ai/generated-not-submitted": "true",
                "mlevolve.ai/gpu-contract": "nvidia.com/a100-family-unpinned",
                "mlevolve.ai/preserve-failed-index-artifacts": "true",
                "mlevolve.ai/per-index-deadline-seconds": str(active_deadline),
                "mlevolve.ai/global-deadline-seconds": str(
                    global_active_deadline
                ),
            },
        },
        "spec": {
            "completionMode": "Indexed",
            "completions": completions,
            "parallelism": parallelism,
            "backoffLimitPerIndex": 0,
            "maxFailedIndexes": completions,
            "activeDeadlineSeconds": global_active_deadline,
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
                            "image": runtime["container_image"],
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/usr/local/bin/python", "-u"],
                            "args": args,
                            "envFrom": [
                                {"secretRef": {"name": "prevalence-audit-deepseek-r1"}}
                            ],
                            "env": [
                                {
                                    "name": "JOB_COMPLETION_INDEX",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.annotations['batch.kubernetes.io/job-completion-index']"
                                        }
                                    },
                                },
                                {
                                    "name": "KUBERNETES_JOB_NAME",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.labels['batch.kubernetes.io/job-name']"
                                        }
                                    },
                                },
                                {
                                    "name": "KUBERNETES_JOB_UID",
                                    "valueFrom": {
                                        "fieldRef": {
                                            "fieldPath": "metadata.labels['batch.kubernetes.io/controller-uid']"
                                        }
                                    },
                                },
                                {
                                    "name": "KUBERNETES_POD_NAME",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "metadata.name"}
                                    },
                                },
                                {
                                    "name": "KUBERNETES_POD_UID",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "metadata.uid"}
                                    },
                                },
                                {
                                    "name": "KUBERNETES_NODE_NAME",
                                    "valueFrom": {
                                        "fieldRef": {"fieldPath": "spec.nodeName"}
                                    },
                                },
                                {"name": "PYTHONUNBUFFERED", "value": "1"},
                                {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
                                {
                                    "name": "PYTHONPATH",
                                    "value": "/workspace/nautilus-exp-end2end/mlevolve",
                                },
                                {"name": "MLEVOLVE_CONTAINER_IMAGE_REFERENCE", "value": runtime["container_image"]},
                                {"name": "MLEVOLVE_SOLVER_BINDING_ID", "value": runtime["solver_model_id"]},
                                {"name": "MLEVOLVE_SOLVER_MODEL_REVISION", "value": runtime["solver_model_revision"]},
                            ],
                            "resources": {"requests": dict(resources), "limits": dict(resources)},
                            "volumeMounts": [
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "collector-key-source", "mountPath": "/run/host-key-source", "readOnly": True},
                                {"name": "collector-key-runtime", "mountPath": "/run/host-key"},
                                {"name": "shm", "mountPath": "/dev/shm"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "workspace", "persistentVolumeClaim": {"claimName": "haoming-storage"}},
                        {"name": "collector-key-source", "secret": {"secretName": "prevalence-audit-collector-r1", "defaultMode": 256}},
                        {"name": "collector-key-runtime", "emptyDir": {}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
                    ],
                },
            },
        },
    }


def build() -> dict[str, Any]:
    write_system_configs()
    components = component_manifests()
    for key, payload in components.items():
        dump_json(MANIFESTS / f"{key}.json", payload)
    smoke = execution_manifest(kind="smoke", components=components)
    pilot = execution_manifest(kind="pilot", components=components)
    dump_json(MANIFESTS / "smoke_manifest.json", smoke)
    dump_json(MANIFESTS / "pilot_manifest.json", pilot)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    smoke_job = job(
        name="mlevolve-e2e-agent-smoke-leaf-v1",
        manifest_name="smoke_manifest.json",
        completions=10,
        task_id=None,
        active_deadline=3600,
        parallelism=1,
        components=components,
    )
    (JOB_DIR / "smoke-leaf-indexed-job.yaml").write_text(
        yaml.safe_dump(smoke_job, sort_keys=False), encoding="utf-8"
    )
    for task_id, display, _metric, _direction in TASKS:
        pilot_job = job(
            name=f"mlevolve-e2e-agent-pilot-{display.lower()}-v1",
            manifest_name="pilot_manifest.json",
            completions=10,
            task_id=task_id,
            active_deadline=25200,
            parallelism=1,
            components=components,
        )
        (JOB_DIR / f"pilot-{display.lower()}-indexed-job.yaml").write_text(
            yaml.safe_dump(pilot_job, sort_keys=False), encoding="utf-8"
        )
    packet = finalize(
        {
            "schema": "mlevolve_end2end_launch_packet_v1",
            "status": "generated_not_submitted",
            "launch_gate": "explicit_user_authorization_required",
            "smoke_gate_output": f"{OUTPUT_ROOT}/SMOKE_GATE.json",
            "pilot_requires_passing_smoke_gate": True,
            "smoke_manifest_hash": smoke["manifest_hash"],
            "pilot_manifest_hash": pilot["manifest_hash"],
            "component_manifest_hashes": {
                key: value["manifest_hash"] for key, value in components.items()
            },
            "jobs": [path.name for path in sorted(JOB_DIR.glob("*.yaml"))],
            "job_count": 5,
            "workload_objects_created": 0,
            "packet_hash": "",
        },
        "packet_hash",
    )
    dump_json(MANIFESTS / "launch_packet.json", packet)
    return packet


def check() -> dict[str, Any]:
    for path in MANIFESTS.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        field = "packet_hash" if path.name == "launch_packet.json" else "manifest_hash"
        if payload_hash(payload, field) != payload.get(field):
            raise ValueError(f"Self-hash mismatch: {path}")
    pilot = json.loads((MANIFESTS / "pilot_manifest.json").read_text())
    smoke = json.loads((MANIFESTS / "smoke_manifest.json").read_text())
    if pilot["run_count"] != 40 or smoke["run_count"] != 10:
        raise ValueError("Frozen matrix cardinality mismatch")
    if len({row["logical_run_id"] for row in pilot["runs"]}) != 40:
        raise ValueError("Pilot logical run IDs are not unique")
    packet = json.loads((MANIFESTS / "launch_packet.json").read_text())
    for path in JOB_DIR.glob("*.yaml"):
        workload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if workload.get("kind") != "Job":
            raise ValueError(f"Generated workload is not a Job: {path}")
    return packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    packet = check() if args.check else build()
    print(json.dumps(packet, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
