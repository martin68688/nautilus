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
CLUSTER_REPO = Path("/workspace/nautilus-exp-end2end-agent-v22")
CLUSTER_ROOT = CLUSTER_REPO / "experiments" / "end2end_memory_systems_20260804"
MANIFESTS = ROOT / "manifests"
SYSTEM_DIR = ROOT / "systems"
JOB_DIR = ROOT / "jobs"
SCHEMA_DIR = ROOT / "schemas"
HOST_BINDINGS_DIR = ROOT / "host_bindings"
CLUSTER_HOST_BINDINGS_DIR = CLUSTER_ROOT / "host_bindings"
SEED = 1
RELEASE_ID = "end2end-agentic-three-role-v22"
BASELINE_RELEASE_ID = "end2end-agent-v3"
RANDOMIZATION_RELEASE_ID = BASELINE_RELEASE_ID
OUTPUT_ROOT = "/workspace/experiment-end2end-memory-agent-v22/runs"
EXPERIMENT_LABEL = "experiment-end2end-memory-agent-v22"
SOLVER_TEMPERATURE = 1.0
SYSTEMS = (
    ("S0", "no_memory", "internal", "Bundle-bound zero Prompt exposure"),
    ("S1", "flat_retrieval", "internal", "Mixed global relevance Top-6"),
    ("S2", "sop_only", "internal", "SOP Top-6"),
    ("S3", "runforest_only", "internal", "RunForest Top-6"),
    ("S4", "static_hybrid", "internal", "3 SOP / 3 RunForest"),
    ("S5", "dynamic_hybrid", "internal", "Draft 5/1, Improve 3/3, Debug 1/5"),
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
        "bundle_id": "end2end-fourtask-direct-aerial-cactus-identification-v2",
        "bundle_root": "/workspace/experiment-end2end-memory-agent-v13/memory-direct-v2/aerial-cactus-identification",
        "bundle_version": "v2",
        "bundle_manifest_sha256": "87b03da6fde381807b2ac58f18ad0404215c0c316d57e3cea556cf39e2a9c812",
        "bundle_manifest_file_sha256": "9ce646336e5408a750ca313c00731cbf1f32e518f6beea3be6365e521caf6129",
        "graph_sha256": "74ce8cd66d4b2b78399ad9a8f703029f33e22a78d2a5e721474dda5065d402fc",
        "index_sha256": "4185ad3c55fce9a5d9af85ccd2059ccb81dad8991c5e98dd6b41cd7a8cb8fced",
        "current_file_sha256": "1dfbbe9bbe752fd7ab68e828901d56935e8b7b7cee78c6fd7cd891334ff1bcba",
        "memory_scope": "full_reviewed_fourtask_with_same_task_history",
        "formal_child_publication": False,
        "same_task_history_enabled": True,
        "same_task_best_node_id": "run::20260725_051618_aerial-cactus-identification-deepseek-v4-full-r1::node::c7d2cb4075a04925a4cae8a98add9119",
        "protocol_ref": "stratified-roc-auc-classification@1#799541fb3a05e1759d76887ae970f061573393a380b185cdfb60ef6f2172a9b1",
    },
    "leaf-classification": {
        "bundle_id": "end2end-fourtask-direct-leaf-classification-v2",
        "bundle_root": "/workspace/experiment-end2end-memory-agent-v13/memory-direct-v2/leaf-classification",
        "bundle_version": "v2",
        "bundle_manifest_sha256": "4e4a3c3dbe541dc025431d9947a35e650093f008e995f108358e9450f4336d69",
        "bundle_manifest_file_sha256": "c65ae7940d96dffdba3428da64a692e34f30298d9c3ef05a1169b782ebc288df",
        "graph_sha256": "74ce8cd66d4b2b78399ad9a8f703029f33e22a78d2a5e721474dda5065d402fc",
        "index_sha256": "4185ad3c55fce9a5d9af85ccd2059ccb81dad8991c5e98dd6b41cd7a8cb8fced",
        "current_file_sha256": "ba458d44121a960493d04fe0b593dfaca341710ad280a8febf89e37f32c4bef0",
        "memory_scope": "full_reviewed_fourtask_with_same_task_history",
        "formal_child_publication": False,
        "same_task_history_enabled": True,
        "same_task_best_node_id": "run::20260717_060628_leaf-classification::node::c9368a59b9324c31afc4813545813045",
        "protocol_ref": "stratified-log-loss-classification@1#a7601be6346021743e01ab144b38e457109ca3ddc701a4aaae9cd562203d79d5",
    },
    "denoising-dirty-documents": {
        "bundle_id": "end2end-fourtask-direct-denoising-dirty-documents-v2",
        "bundle_root": "/workspace/experiment-end2end-memory-agent-v13/memory-direct-v2/denoising-dirty-documents",
        "bundle_version": "v2",
        "bundle_manifest_sha256": "4277332b560bdb3c512df198eff511352dd5c86ba88510a20e443811e2a753d7",
        "bundle_manifest_file_sha256": "d2492775ed150b7cb527155aa2c06dc36c5d152e304560feb8b04793ad2b2502",
        "graph_sha256": "74ce8cd66d4b2b78399ad9a8f703029f33e22a78d2a5e721474dda5065d402fc",
        "index_sha256": "4185ad3c55fce9a5d9af85ccd2059ccb81dad8991c5e98dd6b41cd7a8cb8fced",
        "current_file_sha256": "9c17b6327882485fb07bd33cbec4b36a461878829e81a7aa22bbbaf686be4e42",
        "memory_scope": "full_reviewed_fourtask_with_same_task_history",
        "formal_child_publication": False,
        "same_task_history_enabled": True,
        "same_task_best_node_id": "run::20260725_053032_denoising-dirty-documents-deepseek-v4-full-r2::node::92c40271e8874f249f2a951d595e7452",
        "protocol_ref": "deterministic-random-regression@1#ba944032aa890aa2b742030895b670a7780a7698341e249d57917d1cd898695d",
    },
    "new-york-city-taxi-fare-prediction": {
        "bundle_id": "end2end-fourtask-direct-new-york-city-taxi-fare-prediction-v2",
        "bundle_root": "/workspace/experiment-end2end-memory-agent-v13/memory-direct-v2/new-york-city-taxi-fare-prediction",
        "bundle_version": "v2",
        "bundle_manifest_sha256": "b63abb958d96b0f7a275f0fb17d4c34b6cfb139a8e6451683c4559b6f0c92bcb",
        "bundle_manifest_file_sha256": "d8e2a0f09fe33c7969078006754e6f183ded48aad211023491c307be2bf20e75",
        "graph_sha256": "74ce8cd66d4b2b78399ad9a8f703029f33e22a78d2a5e721474dda5065d402fc",
        "index_sha256": "4185ad3c55fce9a5d9af85ccd2059ccb81dad8991c5e98dd6b41cd7a8cb8fced",
        "current_file_sha256": "7de18ababdf43dd9c81df00eb560ca6b9f7bd83d1a6cb368abc1b19b5cd631e2",
        "memory_scope": "full_reviewed_fourtask_with_same_task_history",
        "formal_child_publication": False,
        "same_task_history_enabled": True,
        "same_task_best_node_id": "run::20260726_022228_new-york-city-taxi-fare-prediction-host-shadow-r7::node::eeb6e2364829449ba6e1ce6c1600fc3d",
        "protocol_ref": "chronological-regression@1#bfc61957b422df5cf09dcb37cffe06aae2ccd2b11db4fee0721b90a2bc6dbf04",
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


def hash_runtime_sdk_tree() -> str:
    sdk_root = REPO / "mlevolve" / "protocol_runtime"
    rows = []
    for path in sorted(sdk_root.rglob("*.py")):
        if (
            "__pycache__" in path.parts
            or path.name.startswith("._")
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        rows.append(
            f"{path.relative_to(sdk_root).as_posix()}\0{sha256_file(path)}"
        )
    if not rows:
        raise ValueError("Host runtime SDK contains no Python sources")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def refresh_host_bindings(
    sdk_hash: str,
) -> dict[str, dict[str, str]]:
    """Rebind frozen task launchers to the exact local Host SDK tree."""

    host: dict[str, dict[str, str]] = {}
    for task_id, _display, _metric, _direction in TASKS:
        path = HOST_BINDINGS_DIR / task_id / "HOST_PROTOCOL_BINDING.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular Host binding for {task_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("task_id") != task_id:
            raise ValueError(f"Host binding task mismatch for {task_id}")
        payload["sdk_hash"] = sdk_hash
        payload["binding_hash"] = payload_hash(payload, "binding_hash")
        dump_json(path, payload)
        host[task_id] = {
            "binding_file_sha256": sha256_file(path),
            "binding_hash": str(payload["binding_hash"]),
            "contract_hash": str(payload["contract_hash"]),
            "data_view_manifest_hash": str(
                payload["data_view_manifest_hash"]
            ),
        }
    return host


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
    return """# End2End is an effectiveness experiment, not a Host-protocol or provenance
# audit. Inherit directly from authority-off so Host gates cannot reactivate.
extends: ../../../mlevolve/config/config_authority_off.yaml

methodology_kb_path: ""
methodology_dynamic: false

fixed_holdout:
  # The fixed split is already published. Never rescan and hash every public
  # asset before an End2End condition.
  preflight_validate_train_view: false

agent:
  check_data_leakage: false
  use_global_memory: false
  code:
    temp: 1.0
  feedback:
    temp: 1.0
  search:
    num_gpus: 1
    parallel_search_num: 1
    num_drafts: 3
  draft_role_policy:
    enabled: false
    roles: []
  protocol_repair:
    enabled: false
  protocol_preflight:
    enabled: false
    agent_semantic_review_enabled: false
    agent_semantic_max_repair_attempts: 0
    agent_semantic_max_review_attempts: 0
    agent_semantic_temperature: 0.0
    agent_semantic_max_tokens: 4096
    agent_controls_protocol_preflight: false
    install_host_candidate_entrypoint: false
    candidate_process_isolation: true

adoption_tracking:
  enable: true
  enable_analysis: false
  judge_mode: keyword

adoption_verifier:
  enabled: false
  mode: shadow
  model: ""
  temperature: 0.0
  max_tokens: 4096
  max_contracts_per_call: 8
  max_code_chars: 120000
  require_signed_trace: false

prospective_audit:
  enabled: false
  allow_pending_counterfactual: false

external_skill_memory:
  enable: true
  bundle_root: ""
  current_pointer_path: CURRENT.json
  verify_bundle_artifacts: false
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
  excluded_run_ids: []

evaluation_authority:
  require_bound_bundle: false
  mode: "off"
  enforce_operations: []
  enforce_generation_stages: []
  enforce_governance_stages: []
  emit_snapshot: false
  runtime_protocol_observer_enabled: false
  protocol_runtime_mode: legacy_ast

run_identity:
  schema: mlevolve_run_identity_v1
  experiment_group: experiment_end2end_memory_systems_pilot_v1
  baseline_reference_group: experiment_end2end_no_memory_v1
  memory_enabled: true
  memory_system: ""
  memory_version: end2end_fourtask_direct_v2
  identity_source: end2end_frozen_manifest_v1
"""


def system_config(system_id: str) -> str:
    if system_id == "dynamic_hybrid":
        return """extends: base.yaml

agent:
  initial_drafts: 3
  search:
    num_drafts: 3
  draft_role_policy:
    enabled: true
    replay_targets_path: ../paper-skills/eval_skill_memory/clean_replay_targets.json
    roles:
      - coldstart_baseline
      - memory_reproduction
      - novel_exploration
external_skill_memory:
  end2end_memory_system: ""
  retrieval_control: layered_strategy
  enable_agentic: true
  recipe_sop_path: ../experiments/end2end_memory_systems_20260804/recipe_distillation_v3/recipe_sops.json
  recipe_sop_file_sha256: e6db95649c20a642738d6ee35df1aa11ff15287e3613221becb393e28d2a9398
  recipe_sop_bundle_sha256: 8cce9dd7ee70897e23e5f1dfda08d056cf6ae77ad63e758bd2bbd05376e88749
  recipe_evidence_path: ../experiments/end2end_memory_systems_20260804/recipe_distillation_v3/evidence_manifest.json
  recipe_evidence_file_sha256: fcb084206cdaa31cfd052c1bce290871b8c075a6376ee698b5c4119636adda04
  recipe_evidence_manifest_sha256: 25f6729ece9b1ead76b0d8501aa6aa4026cb163e3eaa7eb05c755b1d72f6160f
  recipe_implementation_path: ../experiments/end2end_memory_systems_20260804/recipe_distillation_v3/implementation_capsules.json
  experiment_r_enabled: false
  experiment_r_candidate_limit: 12
  experiment_r_top_k: 6
  experiment_r_prompt_token_budget: 1536
  experiment_r_debug_confidence_threshold: 0.50
  experiment_r_agentic_retrieval_enabled: true
  experiment_r_agentic_max_steps: 4
  experiment_r_agentic_per_step_top_k: 8
  experiment_r_agentic_max_observed: 48
  experiment_r_agentic_temperature: 0.0
  experiment_r_agentic_max_tokens: 1200
  experiment_r_l3_agent_match_enabled: false
  experiment_r_l3_agent_match_max_attempts: 2
  experiment_r_l3_agent_match_min_confidence: 0.50
  experiment_r_l3_agent_match_max_tokens: 1800
  experiment_r_memory_transfer_static_gate: false
  experiment_r_memory_transfer_runtime_gate: false
  stage_quotas:
    draft:
      sop_candidates: 5
      sop_gateways: 3
      tree_candidates: 1
    improve:
      sop_candidates: 3
      sop_gateways: 2
      tree_candidates: 3
    debug:
      sop_candidates: 1
      sop_gateways: 1
      tree_candidates: 5
  rrf_weights:
    draft:
      sop: 0.8333333333333334
      tree: 0.16666666666666666
    improve:
      sop: 0.5
      tree: 0.5
    debug:
      sop: 0.16666666666666666
      tree: 0.8333333333333334

run_identity:
  memory_system: dynamic_hybrid
  system_id: dynamic_hybrid
"""
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
        + sorted(ROOT.glob("*.json"))
        + sorted(SYSTEM_DIR.glob("*.yaml"))
        + sorted(SCHEMA_DIR.glob("*.json"))
        + sorted(HOST_BINDINGS_DIR.rglob("*.json"))
        + [
            ROOT / "recipe_distillation_v3" / "recipe_sops.json",
            ROOT / "recipe_distillation_v3" / "evidence_manifest.json",
            ROOT / "recipe_distillation_v3" / "implementation_capsules.json",
            REPO
            / "paper-skills"
            / "eval_skill_memory"
            / "clean_replay_targets.json",
        ]
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


def component_manifests(
    *,
    host_runtime_sdk_hash: str,
    host_task_bindings: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
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
            "experimental_axis": (
                "system_id; dynamic_hybrid is the full Agentic three-role proposed "
                "system while the other nine retain their frozen controller behavior"
            ),
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
            "solver_sampling": {
                "comparison_axis": "memory_system",
                "baseline_release_id": BASELINE_RELEASE_ID,
                "code_temperature": SOLVER_TEMPERATURE,
                "feedback_temperature": SOLVER_TEMPERATURE,
                "adoption_verifier_temperature": 0.0,
                "all_other_frozen_axes_unchanged": True,
            },
            "shared_memory": {
                "raw_candidates_per_source": 12,
                "raw_candidate_max": 24,
                "top_k": 6,
                "prompt_token_budget": 1536,
                "token_counter": "whitespace_split_v1",
                "visibility_token_budget": 4096,
            },
            "l3_agent_matching": {
                "specialized_root_cause_agent_enabled_for_systems": [],
                "manual_router_enabled_for_systems": ["dynamic_hybrid"],
                "decision_stage": "debug",
                "gateway_selector_agent_enabled": True,
                "gateway_selector_temperature": 0.0,
                "failure_signature_min_match": 0.50,
                "task_scope_order": ["exact_task", "same_task_type"],
                "cross_task_type_allowed": False,
                "manual_synonym_table_used": True,
                "agent_input_scope": "hard_gated_clean_candidates_only",
                "agent_cannot_restore_rejected_candidates": True,
                "failure_fallback": "deterministic_manual_router_order",
            },
            "adoption_verifier": {
                "enabled": False,
                "mode": "off",
                "reason": "experiment tracks prompt visibility and code adoption without a blocking verifier",
            },
            "agent_semantic_protocol_review": {
                "enabled_for_systems": [],
                "host_receipt_admission_authority": False,
                "max_repair_attempts": 0,
                "max_review_attempts": 0,
                "unresolved_disposition": "not_applicable",
                "actual_entrypoint_required": False,
                "method_preservation_required": False,
                "agent_controls_protocol_preflight": False,
                "host_dry_run_executed": False,
                "host_runtime_semantic_disposition": "disabled",
            },
            "experiment_validation": {
                "mode": "experiment_fast_nonblocking_v1",
                "pre_run_confirmations": [
                    "human_confirms_local_experiment_intent_summary",
                ],
                "bundle_artifact_traversal": False,
                "public_train_tree_hash_scan": False,
                "host_protocol_preflight": False,
                "host_runtime_protocol": False,
                "receipt_admission": False,
                "source_lock_enforcement": False,
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
                "agent_steps": 3,
                "agent_time_limit_seconds": 3600,
                "execution_timeout_seconds": 900,
                "finalize_reserve_seconds": 120,
                "initial_drafts": 3,
                "max_replacement_drafts": 0,
                "parallel_search_num": 1,
                "gpu_count": 1,
                "cpu_count": 16,
                "memory_gib": 64,
            },
            "debug_smoke": {
                "agent_steps": 8,
                "agent_time_limit_seconds": 5400,
                "execution_timeout_seconds": 1200,
                "finalize_reserve_seconds": 180,
                "initial_drafts": 3,
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
                "condition_level_resume": True,
                "completed_conditions_are_skipped": True,
                "hard_interruption_orphans_finalized_as_infrastructure": True,
                "epoch_level_checkpoint_guaranteed": False,
                "resume_semantics": (
                    "rerun only the interrupted frozen condition from a new "
                    "immutable attempt; arbitrary generated training code may "
                    "not expose epoch checkpoints"
                ),
            },
            "manifest_hash": "",
        }
    )
    memory_manifest = finalize(
        {
            "schema": "mlevolve_end2end_memory_bundle_manifest_v1",
            "production_binding_path": "/workspace/experiment-end2end-memory-agent-v13/memory-direct-v2/MEMORY_BINDING.json",
            "production_binding_sha256": "30854860e2a2cd79f788ddd94cd6225192d2ccb203d5c45b9281628baf2b9b56",
            "verification_mode": "experiment_fast_nonblocking_v1",
            "source_graph_manifest_schema": "fourtask_runforest_graph_manifest_v2",
            "source_graph_sha256": "74ce8cd66d4b2b78399ad9a8f703029f33e22a78d2a5e721474dda5065d402fc",
            "source_index_sha256": "4185ad3c55fce9a5d9af85ccd2059ccb81dad8991c5e98dd6b41cd7a8cb8fced",
            "same_task_history_policy": (
                "enabled for all four tasks; Dynamic pins the direction-aware "
                "best clean positive-eligible record"
            ),
            "task_bundles": MEMORY,
            "excluded_run_ids": [],
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
            "timing": "after Agent exit; candidate-set receipts are observational only",
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
        f"{RANDOMIZATION_RELEASE_ID}|{task_id}|seed-{SEED}".encode()
    ).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    values = [system_id for _label, system_id, _kind, _description in SYSTEMS]
    rng.shuffle(values)
    return values


def execution_manifest(
    *,
    kind: str,
    components: Mapping[str, Mapping[str, Any]],
    system_ids_override: list[str] | None = None,
    task_ids_override: list[str] | None = None,
    prefix_override: str | None = None,
) -> dict[str, Any]:
    if kind == "smoke":
        task_ids = task_ids_override or ["aerial-cactus-identification"]
        system_ids = system_ids_override
        formal = False
        prefix = prefix_override or "e2e-smoke-all-systems-v2"
    else:
        task_ids = [task_id for task_id, _display, _metric, _direction in TASKS]
        system_ids = None
        formal = True
        prefix = "e2e-pilot-agentic-three-role-v22"
    bindings = {
        f"{key}_manifest_hash": value["manifest_hash"]
        for key, value in components.items()
    }
    runs = []
    global_position = 0
    for task_id in task_ids:
        ordered_systems = system_ids or shuffled_system_ids(task_id)
        for task_position, system_id in enumerate(ordered_systems):
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
            "release_id": RELEASE_ID,
            "comparison_baseline_release_id": BASELINE_RELEASE_ID,
            "kind": kind,
            "formal_result_eligible": formal,
            "exploratory_pilot": True,
            "seed": SEED,
            "statistical_significance_claim_allowed": False,
            "system_ids": (
                system_ids
                if system_ids is not None
                else [system_id for _label, system_id, _kind, _desc in SYSTEMS]
            ),
            "task_ids": task_ids,
            "run_count": len(runs),
            "launch_order_randomization": (
                "task-local SHA256-seeded deterministic shuffle; comparison order "
                f"locked to {RANDOMIZATION_RELEASE_ID}"
            ),
            "bindings": bindings,
            "runs": runs,
            "manifest_hash": "",
        }
    )


def job(
    *, name: str, manifest_name: str, completions: int, task_id: str | None,
    active_deadline: int, parallelism: int, components: Mapping[str, Mapping[str, Any]],
    attempt: int = 0, resume: bool = False,
) -> dict[str, Any]:
    runtime = components["budget"]["runtime"]
    index_waves = (completions + parallelism - 1) // parallelism
    global_active_deadline = active_deadline * index_waves
    labels = {
        "app": "mlevolve-end2end",
        "experiment": EXPERIMENT_LABEL,
        "mlevolve.ai/workload": name,
        "ecepxie.nrp/owner": "haoming",
        "app.kubernetes.io/managed-by": "codex-nrp-training",
    }
    args = [
        str(CLUSTER_ROOT / "run_assignment.py"),
        "--manifest", str(CLUSTER_ROOT / "manifests" / manifest_name),
        "--output-root", OUTPUT_ROOT,
    ]
    if attempt:
        args.extend(["--attempt", str(attempt)])
    if resume:
        args.append("--resume")
    if task_id:
        args.extend(["--task", task_id])
    resources = {
        "cpu": "16",
        "memory": "64Gi",
        "ephemeral-storage": "64Gi",
        runtime["gpu_resource_key"]: "1",
    }
    gpu_contract = f"{runtime['gpu_resource_key']}=1"
    product_constraint = runtime.get("gpu_product_constraint")
    if product_constraint:
        gpu_contract += ";product=" + ",".join(product_constraint)
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
                "mlevolve.ai/gpu-contract": gpu_contract,
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
                                    "value": str(CLUSTER_REPO / "mlevolve"),
                                },
                                {"name": "MLEVOLVE_CONTAINER_IMAGE_REFERENCE", "value": runtime["container_image"]},
                                {"name": "MLEVOLVE_SOLVER_BINDING_ID", "value": runtime["solver_model_id"]},
                                {"name": "MLEVOLVE_SOLVER_MODEL_REVISION", "value": runtime["solver_model_revision"]},
                            ],
                            "resources": {"requests": dict(resources), "limits": dict(resources)},
                            "volumeMounts": [
                                {"name": "workspace", "mountPath": "/workspace"},
                                {"name": "shm", "mountPath": "/dev/shm"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "workspace", "persistentVolumeClaim": {"claimName": "haoming-storage"}},
                        {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
                    ],
                },
            },
        },
    }


def build() -> dict[str, Any]:
    write_system_configs()
    components = component_manifests(
        host_runtime_sdk_hash="",
        host_task_bindings={},
    )
    for key, payload in components.items():
        dump_json(MANIFESTS / f"{key}.json", payload)
    # Smoke is completed evidence from release v12. Never rewrite those
    # manifests/jobs when preparing a later formal Pilot release.
    smoke = json.loads((MANIFESTS / "smoke_manifest.json").read_text())
    feasibility_smoke = json.loads(
        (MANIFESTS / "feasibility_smoke_manifest.json").read_text()
    )
    leaf_dynamic_smoke = json.loads(
        (MANIFESTS / "leaf_dynamic_smoke_manifest.json").read_text()
    )
    leaf_controls_smoke = json.loads(
        (MANIFESTS / "leaf_controls_smoke_manifest.json").read_text()
    )
    leaf_recipe_dynamic_smoke = execution_manifest(
        kind="smoke",
        components=components,
        system_ids_override=["dynamic_hybrid"],
        task_ids_override=["leaf-classification"],
        prefix_override="e2e-smoke-leaf-layered-recipe-v4",
    )
    dump_json(
        MANIFESTS / "leaf_recipe_dynamic_smoke_manifest.json",
        leaf_recipe_dynamic_smoke,
    )
    l3_debug_dynamic_smoke = execution_manifest(
        kind="smoke",
        components=components,
        system_ids_override=["dynamic_hybrid"],
        task_ids_override=["aerial-cactus-identification"],
        prefix_override="e2e-smoke-l3-debug-v1",
    )
    dump_json(
        MANIFESTS / "l3_debug_dynamic_smoke_manifest.json",
        l3_debug_dynamic_smoke,
    )
    pilot = execution_manifest(kind="pilot", components=components)
    dump_json(MANIFESTS / "pilot_manifest.json", pilot)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    for stale in JOB_DIR.glob("pilot-*-indexed-job.yaml"):
        stale.unlink()
    pilot_job = job(
        name="mlevolve-e2e-agentic-pilot-all-40-v22",
        manifest_name="pilot_manifest.json",
        completions=40,
        task_id=None,
        active_deadline=25200,
        parallelism=1,
        components=components,
        resume=True,
    )
    (JOB_DIR / "pilot-all-40-indexed-job.yaml").write_text(
        yaml.safe_dump(pilot_job, sort_keys=False), encoding="utf-8"
    )
    leaf_recipe_job = job(
        name="mlevolve-e2e-leaf-layered-recipe-smoke-v28",
        manifest_name="leaf_recipe_dynamic_smoke_manifest.json",
        completions=1,
        task_id=None,
        active_deadline=5400,
        parallelism=1,
        components=components,
        resume=True,
    )
    (JOB_DIR / "smoke-leaf-layered-recipe-job.yaml").write_text(
        yaml.safe_dump(leaf_recipe_job, sort_keys=False), encoding="utf-8"
    )
    generated_jobs = ["pilot-all-40-indexed-job.yaml"]
    packet = finalize(
        {
            "schema": "mlevolve_end2end_launch_packet_v1",
            "status": "generated_not_submitted",
            "launch_gate": "explicit_user_authorization_required",
            "pre_run_confirmation": "one local human-facing intent confirmation only",
            "pilot_requires_passing_smoke_gate": False,
            "smoke_manifest_hash": smoke["manifest_hash"],
            "feasibility_smoke_manifest_hash": feasibility_smoke["manifest_hash"],
            "leaf_dynamic_smoke_manifest_hash": leaf_dynamic_smoke["manifest_hash"],
            "leaf_controls_smoke_manifest_hash": leaf_controls_smoke["manifest_hash"],
            "leaf_recipe_dynamic_smoke_manifest_hash": leaf_recipe_dynamic_smoke[
                "manifest_hash"
            ],
            "l3_debug_dynamic_smoke_manifest_hash": l3_debug_dynamic_smoke[
                "manifest_hash"
            ],
            "pilot_manifest_hash": pilot["manifest_hash"],
            "component_manifest_hashes": {
                key: value["manifest_hash"] for key, value in components.items()
            },
            "jobs": generated_jobs,
            "job_count": len(generated_jobs),
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
    feasibility = json.loads(
        (MANIFESTS / "feasibility_smoke_manifest.json").read_text()
    )
    leaf_dynamic = json.loads(
        (MANIFESTS / "leaf_dynamic_smoke_manifest.json").read_text()
    )
    if (
        pilot["run_count"] != 40
        or smoke["run_count"] != 10
        or feasibility["run_count"] != 1
        or leaf_dynamic["run_count"] != 1
    ):
        raise ValueError("Frozen matrix cardinality mismatch")
    if len({row["logical_run_id"] for row in pilot["runs"]}) != 40:
        raise ValueError("Pilot logical run IDs are not unique")
    packet = json.loads((MANIFESTS / "launch_packet.json").read_text())
    for name in packet["jobs"]:
        path = JOB_DIR / name
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
