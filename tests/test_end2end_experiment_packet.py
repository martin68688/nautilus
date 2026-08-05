from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml
import jsonschema

from protocol_runtime.collector import HostCollectorIdentity


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments" / "end2end_memory_systems_20260804"
MANIFESTS = ROOT / "manifests"
TEST_COLLECTOR_IDENTITY = HostCollectorIdentity.generate()

sys.path.insert(0, str(ROOT))
try:
    import run_assignment
    import validate_smoke_gate
finally:
    sys.path.pop(0)


def _hash(payload: dict, field: str) -> str:
    return hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != field},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_hashed(path: Path, payload: dict, field: str) -> None:
    payload[field] = _hash(payload, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _build_synthetic_gate(output_root: Path) -> dict:
    return validate_smoke_gate.build_smoke_gate(
        output_root=output_root,
        _test_collector_public_key_ed25519=(
            TEST_COLLECTOR_IDENTITY.public_key_ed25519
        ),
    )


def _synthetic_smoke_output(tmp_path: Path) -> Path:
    output_root = tmp_path / "runs"
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    memory = _read(MANIFESTS / "memory_bundles.json")
    for row in smoke["runs"]:
        system_id = row["system_id"]
        attempt_root = output_root / row["logical_run_id"] / "attempt-000"
        journal_path = attempt_root / "agent" / "logs" / "journal.json"
        raw = [
            {
                "candidate_id": f"{system_id}-sop-1",
                "source": "sop",
                "relevance": 0.9,
                "prompt_text": "authorized memory card",
                "source_stage": "draft",
                "source_task_id": row["task_id"],
                "rank": 1,
                "metadata": {"authorized": True},
            },
            {
                "candidate_id": f"{system_id}-run-1",
                "source": "runforest",
                "relevance": 0.8,
                "prompt_text": "authorized execution card",
                "source_stage": "draft",
                "source_task_id": row["task_id"],
                "rank": 1,
                "metadata": {"authorized": True},
            },
        ]
        visible = [] if system_id == "no_memory" else [raw[0]["candidate_id"]]
        selected = [] if system_id == "no_memory" else [raw[0]]
        prompt_candidates = [] if system_id == "no_memory" else [
            {
                "candidate_id": raw[0]["candidate_id"],
                "source": raw[0]["source"],
                "source_stage": raw[0]["source_stage"],
                "source_task_id": raw[0]["source_task_id"],
                "prompt_text": (
                    f"### {raw[0]['candidate_id']} [{raw[0]['source']}]\n"
                    f"{raw[0]['prompt_text']}"
                ),
            }
        ]
        suppressed = [
            {
                "candidate_id": item["candidate_id"],
                "source": item["source"],
                "reason": "not_selected_by_frozen_system_policy",
            }
            for item in (raw if system_id == "no_memory" else raw[1:])
        ]
        trace = {
            "schema": "mlevolve_memory_routing_trace_v1",
            "memory_pack_schema": "mlevolve_end2end_memory_pack_v1",
            "algorithm_version": "end2end_memory_systems_pilot_v1",
            "system_id": system_id,
            "stage_route": {
                "stage": "draft",
                "canonical_stage": "draft",
                "top_k": 6,
                "prompt_token_budget": 1536,
            },
            "target_task_id": row["task_id"],
            "candidate_pool_hash": hashlib.sha256(
                json.dumps(
                    raw,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "candidate_pool_source": "shared_authority_filtered_sop_runforest",
            "raw_pool_observed": True,
            "raw_candidates": raw,
            "selected_candidates": selected,
            "suppressed_candidates": suppressed,
            "final_prompt_candidates": prompt_candidates,
            "final_prompt_candidate_ids": visible,
            "visible_clause_ids": [],
            "prompt_token_count": 0 if system_id == "no_memory" else 5,
            "prompt_truncated": False,
            "visibility_safety_gate": {
                "mode": "enforce",
                "pre_ranking": True,
                "unauthorized_prompt_exposure": 0,
                "unauthorized_activation": 0,
            },
            "unauthorized_prompt_exposure": 0,
            "memory_snapshot_bound_but_not_exposed": system_id == "no_memory",
            "memory_bundle": {
                "bundle_id": memory["task_bundles"][row["task_id"]]["bundle_id"],
                "manifest_sha256": memory["task_bundles"][row["task_id"]][
                    "bundle_manifest_sha256"
                ],
            },
        }
        code = "def train_and_predict():\n    return 1\n"
        memory_candidate_contract_refs = {}
        experience_contract_refs = []
        plan = {}
        adoption_trace = {}
        verdict = {}
        if visible:
            contract_hash = hashlib.sha256(
                f"synthetic-contract:{visible[0]}".encode()
            ).hexdigest()
            contract_id = f"experience_contract::{contract_hash[:24]}"
            memory_candidate_contract_refs = {visible[0]: contract_id}
            experience_contract_refs = [contract_id]
            plan = {
                "schema": "agent_adoption_verification_plan_v1",
                "artifact_id": "candidate-1",
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "verifier_model": "synthetic-agent-verifier",
                "contract_results": [
                    {
                        "contract_id": contract_id,
                        "contract_hash": contract_hash,
                        "clause_id": "synthetic-clause",
                        "sop_id": visible[0],
                        "disposition": "implemented",
                        "reasoning": "synthetic complete-chain fixture",
                        "code_evidence": [
                            {
                                "evidence_id": "evidence-1",
                                "start_line": 1,
                                "end_line": 2,
                                "source_sha256": hashlib.sha256(
                                    code.rstrip("\n").encode()
                                ).hexdigest(),
                                "description": "candidate path",
                                "ordinal": 0,
                            }
                        ],
                        "runtime_probes": [
                            {
                                "probe_id": "probe-1",
                                "kind": "line_range_executed",
                                "start_line": 1,
                                "end_line": 2,
                                "description": "candidate path",
                            }
                        ],
                        "precondition_observations": [],
                        "static_observations": [],
                        "runtime_observations": [],
                    }
                ],
                "plan_hash": "",
            }
            plan["plan_hash"] = _hash(plan, "plan_hash")
            adoption_trace = {
                "schema": "agent_adoption_runtime_trace_v1",
                "artifact_id": "candidate-1",
                "code_sha256": plan["code_sha256"],
                "plan_hash": plan["plan_hash"],
                "raw_trace_sha256": "b" * 64,
                "exit_status": 0,
                "probe_results": [
                    {
                        "probe_id": "probe-1",
                        "kind": "line_range_executed",
                        "executed": True,
                        "executed_lines": [1, 2],
                        "hit_count": 2,
                    }
                ],
                "trace_hash": "",
                "signature_algorithm": "ed25519",
                "public_key_ed25519": (
                    TEST_COLLECTOR_IDENTITY.public_key_ed25519
                ),
                "signature_ed25519": "",
            }
            trace_hash_input = {
                key: value
                for key, value in adoption_trace.items()
                if key not in {"trace_hash", "signature_ed25519"}
            }
            adoption_trace["trace_hash"] = hashlib.sha256(
                json.dumps(
                    trace_hash_input,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            adoption_trace["signature_ed25519"] = (
                TEST_COLLECTOR_IDENTITY.sign_payload(
                    {
                        key: value
                        for key, value in adoption_trace.items()
                        if key != "signature_ed25519"
                    }
                )
            )
            verdict = {
                "schema": "agent_adoption_verdict_v1",
                "artifact_id": "candidate-1",
                "code_sha256": plan["code_sha256"],
                "plan_hash": plan["plan_hash"],
                "trace_hash": adoption_trace["trace_hash"],
                "verifier_model": "synthetic-agent-verifier",
                "contract_results": [
                    {
                        "contract_id": contract_id,
                        "contract_hash": contract_hash,
                        "verdict": "adopted",
                        "reasoning": "synthetic probe executed",
                        "supporting_probe_ids": ["probe-1"],
                        "runtime_evidence_valid": True,
                    }
                ],
                "verdict_hash": "",
            }
            verdict["verdict_hash"] = _hash(verdict, "verdict_hash")
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "id": "candidate-1",
                            "stage": "draft",
                            "code": code,
                            "exec_time": 1.0,
                            "protocol_observation": {
                                "host_full_runtime": {"status": "pass"}
                            },
                            "memory_routing_trace": trace,
                            "memory_candidate_contract_refs": (
                                memory_candidate_contract_refs
                            ),
                            "experience_contract_refs": experience_contract_refs,
                            "adoption_verification_plan": plan,
                            "adoption_runtime_trace": adoption_trace,
                            "adoption_verifier_verdict": verdict,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        terminal_path = attempt_root / "TERMINAL_SCORE_REPORT.json"
        terminal_path.write_text(
            json.dumps({"schema": "synthetic_terminal_report", "score": 0.1}),
            encoding="utf-8",
        )
        measurement = {
            "schema": "mlevolve_end2end_condition_measurement_v1",
            "logical_run_id": row["logical_run_id"],
            "attempt": 0,
            "retry_of": None,
            "manifest_hash": smoke["manifest_hash"],
            "task_id": row["task_id"],
            "system_id": system_id,
            "seed": 1,
            "formal_result_eligible": False,
            "exploratory_pilot": True,
            "status": "scored_terminal_result",
            "failure_class": "none",
            "completed": True,
            "terminal_metric": "log_loss",
            "direction": "minimize",
            "terminal_score": 0.1,
            "selected_candidate_id": "candidate-1",
            "candidate_set_frozen": True,
            "candidate_set_hash": "a" * 64,
            "terminal_report_sha256": hashlib.sha256(
                terminal_path.read_bytes()
            ).hexdigest(),
            "journal_path": str(journal_path),
            "measurement_hash": "",
        }
        _write_hashed(attempt_root / "MEASUREMENT.json", measurement, "measurement_hash")
    return output_root


def test_all_frozen_manifests_have_valid_self_hashes() -> None:
    for path in sorted(MANIFESTS.glob("*.json")):
        payload = _read(path)
        field = "packet_hash" if path.name == "launch_packet.json" else "manifest_hash"
        assert payload[field] == _hash(payload, field), path


def test_pilot_is_exact_cartesian_product_and_smoke_layers_are_frozen() -> None:
    pilot = _read(MANIFESTS / "pilot_manifest.json")
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    feasibility = _read(MANIFESTS / "feasibility_smoke_manifest.json")
    leaf_dynamic = _read(MANIFESTS / "leaf_dynamic_smoke_manifest.json")
    leaf_controls = _read(MANIFESTS / "leaf_controls_smoke_manifest.json")
    systems = pilot["system_ids"]
    tasks = pilot["task_ids"]
    assert len(systems) == 10 and len(set(systems)) == 10
    assert len(tasks) == 4 and len(set(tasks)) == 4
    assert pilot["run_count"] == 40
    assert smoke["run_count"] == 10
    assert smoke["task_ids"] == ["aerial-cactus-identification"]
    assert feasibility["run_count"] == 1
    assert feasibility["task_ids"] == ["aerial-cactus-identification"]
    assert feasibility["system_ids"] == ["dynamic_hybrid"]
    assert leaf_dynamic["run_count"] == 1
    assert leaf_dynamic["task_ids"] == ["leaf-classification"]
    assert leaf_dynamic["system_ids"] == ["dynamic_hybrid"]
    assert leaf_controls["run_count"] == 9
    assert leaf_controls["task_ids"] == ["leaf-classification"]
    assert leaf_controls["system_ids"] == [
        "sop_only",
        "flat_retrieval",
        "no_memory",
        "gome_style_port",
        "static_hybrid",
        "reversed_router",
        "rcr_router_style_port",
        "runforest_only",
        "macla_style_port",
    ]
    assert set(leaf_dynamic["system_ids"] + leaf_controls["system_ids"]) == set(
        systems
    )
    assert smoke["formal_result_eligible"] is False
    assert pilot["formal_result_eligible"] is True
    assert smoke["release_id"] == pilot["release_id"] == (
        "end2end-agentic-three-role-v12"
    )
    assert smoke["comparison_baseline_release_id"] == (
        pilot["comparison_baseline_release_id"]
    )
    assert smoke["comparison_baseline_release_id"] == "end2end-agent-v3"
    assert [row["system_id"] for row in smoke["runs"]] == [
        "no_memory",
        "rcr_router_style_port",
        "reversed_router",
        "macla_style_port",
        "gome_style_port",
        "sop_only",
        "dynamic_hybrid",
        "runforest_only",
        "static_hybrid",
        "flat_retrieval",
    ]
    assert {
        (row["task_id"], row["system_id"], row["seed"])
        for row in pilot["runs"]
    } == {(task, system, 1) for task in tasks for system in systems}
    assert len({row["logical_run_id"] for row in pilot["runs"]}) == 40
    assert all(row["row_hash"] == _hash(row, "row_hash") for row in pilot["runs"])
    assert pilot["exploratory_pilot"] is True
    assert pilot["statistical_significance_claim_allowed"] is False
    schema = _read(ROOT / "schemas" / "execution_manifest.schema.json")
    jsonschema.validate(pilot, schema)
    jsonschema.validate(smoke, schema)


def test_leaf_uses_direct_seed_heldout_base_with_same_task_history() -> None:
    memory = _read(MANIFESTS / "memory_bundles.json")
    leaf = memory["task_bundles"]["leaf-classification"]
    assert memory["production_binding_path"] == (
        "/workspace/experiment-end2end-memory-agent-v12/"
        "memory-direct-v1/MEMORY_BINDING.json"
    )
    assert memory["verification_mode"] == (
        "experiment_fast_nonblocking_v1"
    )
    assert leaf["bundle_id"] == "mlevolve-be034ec-nonspooky-seed-heldout-v1"
    assert leaf["bundle_root"] == (
        "/workspace/experiment-end2end-memory-agent-v12/"
        "memory-direct-v1/leaf-classification"
    )
    assert "20260717_183734_leaf-classification" in memory["excluded_run_ids"]
def test_system_configs_load_against_structured_runtime(monkeypatch) -> None:
    from config import Config, _load_cfg
    from omegaconf import OmegaConf

    systems = _read(MANIFESTS / "systems.json")
    for row in systems["systems"]:
        path = ROOT / row["config_path"]
        cfg = _load_cfg(path, use_cli_args=False)
        cfg.exp_name = "end2end-config-test"
        cfg.exp_id = "leaf-classification"
        cfg.data_dir = "./data"
        cfg.goal = "config validation"
        cfg.desc_file = None
        merged = OmegaConf.merge(OmegaConf.structured(Config), cfg)
        assert merged.evaluation_authority.protocol_runtime_mode == "legacy_ast"
        assert merged.agent.protocol_preflight.enabled is False
        assert merged.agent.protocol_preflight.agent_semantic_review_enabled is False
        assert merged.agent.protocol_preflight.agent_semantic_max_repair_attempts == 0
        assert merged.agent.protocol_preflight.agent_semantic_max_review_attempts == 0
        assert merged.agent.protocol_preflight.agent_controls_protocol_preflight is False
        assert merged.agent.protocol_preflight.install_host_candidate_entrypoint is False
        assert merged.agent.protocol_preflight.candidate_process_isolation is True
        if row["system_id"] == "dynamic_hybrid":
            assert merged.external_skill_memory.end2end_memory_system == ""
            assert merged.external_skill_memory.retrieval_control == "dynamic_hybrid"
            assert merged.external_skill_memory.experiment_r_enabled is True
            assert merged.external_skill_memory.experiment_r_agentic_retrieval_enabled is True
            assert merged.external_skill_memory.experiment_r_memory_transfer_static_gate is False
            assert merged.external_skill_memory.experiment_r_memory_transfer_runtime_gate is False
            assert merged.agent.draft_role_policy.enabled is True
            assert list(merged.agent.draft_role_policy.roles) == [
                "coldstart_baseline", "memory_transfer", "novel_exploration"
            ]
        else:
            assert merged.external_skill_memory.end2end_memory_system == row["system_id"]
            assert merged.agent.draft_role_policy.enabled is False
        assert merged.external_skill_memory.top_k == 6
        assert merged.external_skill_memory.end2end_prompt_token_budget == 1536
        assert merged.external_skill_memory.end2end_candidate_pool_limit == 12
        assert merged.external_skill_memory.enable is True
        assert merged.external_skill_memory.verify_bundle_artifacts is False
        assert merged.fixed_holdout.preflight_validate_train_view is False
        assert merged.agent.search.num_gpus == 1
        assert merged.agent.search.parallel_search_num == 1
        assert merged.agent.code.temp == 1.0
        assert merged.agent.feedback.temp == 1.0
        assert merged.agent.use_global_memory is False
        assert merged.agent.check_data_leakage is False
        assert merged.evaluation_authority.mode == "off"
        assert merged.evaluation_authority.require_bound_bundle is False
        assert merged.evaluation_authority.emit_snapshot is False
        assert merged.evaluation_authority.runtime_protocol_observer_enabled is False
        assert merged.adoption_verifier.enabled is False
        assert merged.adoption_verifier.require_signed_trace is False
        assert merged.prospective_audit.enabled is False


def test_source_lock_covers_and_matches_runtime_files() -> None:
    lock = _read(MANIFESTS / "source_lock.json")
    assert lock["complete_runtime_file_hash_lock"] is True
    paths = {row["path"] for row in lock["files"]}
    assert "mlevolve/agents/memory/end2end_memory_system.py" in paths
    assert "mlevolve/agents/adoption_verifier_agent.py" in paths
    assert "mlevolve/authority/adoption_verification.py" in paths
    assert "mlevolve/protocol_runtime/adoption_trace.py" in paths
    assert "experiments/end2end_memory_systems_20260804/run_assignment.py" in paths
    assert "experiments/end2end_memory_systems_20260804/analyze_results.py" in paths
    assert (
        "experiments/end2end_memory_systems_20260804/smoke_memory_spec.json"
        in paths
    )
    assert "experiments/end2end_memory_systems_20260804/validate_smoke_gate.py" in paths
    assert (
        "experiments/end2end_memory_systems_20260804/schemas/smoke_gate.schema.json"
        in paths
    )
    for row in lock["files"]:
        assert hashlib.sha256((REPO / row["path"]).read_bytes()).hexdigest() == row[
            "sha256"
        ]


def test_generated_jobs_are_finite_owned_indexed_workloads() -> None:
    packet = _read(MANIFESTS / "launch_packet.json")
    jobs = [ROOT / "jobs" / name for name in packet["jobs"]]
    assert packet["job_count"] == len(jobs) == 8
    for path in jobs:
        job = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert job["kind"] == "Job"
        assert job["metadata"]["namespace"] == "ecepxie"
        labels = job["metadata"]["labels"]
        assert labels["ecepxie.nrp/owner"] == "haoming"
        assert labels["app.kubernetes.io/managed-by"] == "codex-nrp-training"
        assert labels["experiment"] == (
            "experiment-end2end-memory-agent-v12"
        )
        assert job["metadata"]["annotations"]["mlevolve.ai/generated-not-submitted"] == "true"
        assert job["metadata"]["annotations"]["mlevolve.ai/gpu-contract"] == (
            "nvidia.com/a100=1"
        )
        spec = job["spec"]
        assert spec["completionMode"] == "Indexed"
        assert spec["backoffLimitPerIndex"] == 0
        assert spec["parallelism"] == 1
        if path.name.startswith("smoke-"):
            assert spec["activeDeadlineSeconds"] == 5400 * spec["completions"]
            assert job["metadata"]["annotations"][
                "mlevolve.ai/per-index-deadline-seconds"
            ] == "5400"
        else:
            assert spec["activeDeadlineSeconds"] == 252000
            assert job["metadata"]["annotations"][
                "mlevolve.ai/per-index-deadline-seconds"
            ] == "25200"
        container = spec["template"]["spec"]["containers"][0]
        assert "affinity" not in spec["template"]["spec"]
        assert container["command"] == ["/usr/local/bin/python", "-u"]
        rendered = json.dumps(container)
        assert "sleep" not in rendered
        assert "tail -f" not in rendered
        requests = container["resources"]["requests"]
        limits = container["resources"]["limits"]
        assert requests == limits
        assert requests["nvidia.com/a100"] == "1"
        assert requests["memory"] == "64Gi"
        assert "nvidia.com/a40" not in requests
        assert "agent.search.num_gpus=1" not in rendered  # runner owns overrides
        env_names = {row["name"] for row in container["env"]}
        assert {
            "JOB_COMPLETION_INDEX",
            "KUBERNETES_JOB_NAME",
            "KUBERNETES_JOB_UID",
            "KUBERNETES_POD_NAME",
            "KUBERNETES_POD_UID",
            "KUBERNETES_NODE_NAME",
        } <= env_names
        env_values = {row["name"]: row.get("value") for row in container["env"]}
        assert env_values["PYTHONPATH"] == (
            "/workspace/nautilus-exp-end2end-agent-v14/mlevolve"
        )
        assert "--smoke-gate" not in container["args"]
        if path.name == "smoke-leaf-dynamic-hybrid-job.yaml":
            assert container["args"][-2:] == ["--attempt", "2"]
        if path.name == "smoke-leaf-controls-job.yaml":
            assert job["metadata"]["name"] == (
                "mlevolve-e2e-leaf-controls-smoke-v15"
            )
            assert container["args"][-1] == "--resume"
        volume_names = {
            row["name"] for row in spec["template"]["spec"]["volumes"]
        }
        assert "collector-key-source" not in volume_names
        assert "collector-key-runtime" not in volume_names


def test_temp05_stager_is_owned_finite_cpu_only_pod() -> None:
    path = REPO / "deploy" / "pod-end2end-agent-stager-temp05-v1.yaml"
    pod = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert pod["kind"] == "Pod"
    assert pod["metadata"]["name"] == "end2end-agent-stager-temp05-v1"
    labels = pod["metadata"]["labels"]
    assert labels["ecepxie.nrp/owner"] == "haoming"
    assert labels["app.kubernetes.io/managed-by"] == "codex-nrp-training"
    assert labels["experiment"] == "experiment-end2end-memory-agent-temp05-v1"
    assert pod["spec"]["restartPolicy"] == "Never"
    assert pod["spec"]["activeDeadlineSeconds"] == 3600
    resources = pod["spec"]["containers"][0]["resources"]
    assert resources["requests"] == resources["limits"]
    assert not any(key.startswith("nvidia.com/") for key in resources["requests"])


def test_launch_packet_records_only_human_intent_confirmation() -> None:
    packet = _read(MANIFESTS / "launch_packet.json")
    assert packet["pilot_requires_passing_smoke_gate"] is False
    assert packet["pre_run_confirmation"] == (
        "one local human-facing intent confirmation only"
    )


def test_end2end_effective_config_has_no_runtime_validation_gates() -> None:
    from config import _load_cfg

    cfg = _load_cfg(ROOT / "systems" / "dynamic_hybrid.yaml", use_cli_args=False)
    assert cfg.evaluation_authority.mode == "off"
    assert cfg.evaluation_authority.enforce_operations == []
    assert cfg.agent.protocol_preflight.enabled is False
    assert cfg.agent.protocol_repair.enabled is False
    assert cfg.agent.check_data_leakage is False
    assert cfg.adoption_verifier.enabled is False
    assert cfg.prospective_audit.enabled is False
    assert cfg.fixed_holdout.preflight_validate_train_view is False


def test_release_overrides_cannot_reactivate_validation_gates() -> None:
    values = run_assignment._fixed_holdout_overrides(
        {
            "additional_overrides": [
                "fixed_holdout.enabled=true",
                "fixed_holdout.bypass_protocol_gates=true",
                "agent.check_data_leakage=true",
                "agent.protocol_repair.enabled=true",
                "prospective_audit.enabled=true",
                "evaluation_authority.mode=enforce",
            ]
        }
    )
    assert values == [
        "fixed_holdout.enabled=true",
        "fixed_holdout.bypass_protocol_gates=true",
    ]


def test_intent_confirmation_is_local_and_does_not_launch_training() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "confirm_experiment_intent.py")],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ready_for_user_confirmation"
    assert payload["launches_training"] is False
    assert payload["experiment"]["runs"] == 40
    assert payload["experiment"]["kind"] == "pilot"
    assert payload["experiment"]["seeds"] == [1]
    assert payload["runtime_checks"] == {
        "host_protocol": False,
        "host_receipts": False,
        "data_tree_hash": False,
        "bundle_artifact_traversal": False,
        "source_lock_gate": False,
        "adoption_gate": False,
        "prospective_audit": False,
    }


def test_leaf_controls_intent_confirmation_covers_only_remaining_nine() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "confirm_experiment_intent.py"),
            "--manifest",
            str(MANIFESTS / "leaf_controls_smoke_manifest.json"),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["launches_training"] is False
    assert payload["experiment"]["kind"] == "smoke"
    assert payload["experiment"]["runs"] == 9
    assert payload["experiment"]["tasks"] == ["leaf-classification"]
    assert payload["dynamic_hybrid"]["included_in_this_manifest"] is False


def test_runner_local_dry_run_makes_no_external_or_agent_calls(tmp_path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'mlevolve'}:{REPO}"
    before = list(tmp_path.iterdir())
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_assignment.py"),
            "--manifest",
            str(MANIFESTS / "smoke_manifest.json"),
            "--index",
            "0",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["agent_calls"] == 0
    assert payload["filesystem_written"] is False
    assert list(tmp_path.iterdir()) == before


def test_pilot_dry_run_does_not_require_smoke_gate_or_write(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_assignment.py"),
            "--manifest",
            str(MANIFESTS / "pilot_manifest.json"),
            "--index",
            "0",
            "--output-root",
            str(tmp_path),
            "--dry-run",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["agent_calls"] == 0
    assert payload["filesystem_written"] is False
    assert list(tmp_path.iterdir()) == []


def test_pilot_normal_execution_does_not_require_smoke_gate(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "run_assignment.py"),
            "--manifest",
            str(MANIFESTS / "pilot_manifest.json"),
            "--index",
            "0",
            "--output-root",
            str(tmp_path),
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Pilot execution requires --smoke-gate" not in completed.stderr
    assert "CURRENT.json" in completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_runner_termination_guard_retains_sigterm_instead_of_exiting() -> None:
    import os
    import signal
    import pytest

    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(run_assignment.RunnerInterrupted) as interrupted:
        with run_assignment.termination_guard():
            os.kill(os.getpid(), signal.SIGTERM)
    assert interrupted.value.signum == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous


def test_resume_starts_missing_and_retries_only_infrastructure(tmp_path) -> None:
    logical_run_id = "leaf-control"
    assert run_assignment.resolve_resume_attempt(tmp_path, logical_run_id) == (
        0,
        None,
    )
    measurement_path = (
        tmp_path / logical_run_id / "attempt-000" / "MEASUREMENT.json"
    )
    measurement_path.parent.mkdir(parents=True)
    measurement_path.write_text(
        json.dumps({"failure_class": "infrastructure", "completed": False}),
        encoding="utf-8",
    )
    assert run_assignment.resolve_resume_attempt(tmp_path, logical_run_id) == (
        1,
        None,
    )
    retained = {"failure_class": "agent", "completed": False}
    measurement_path.write_text(json.dumps(retained), encoding="utf-8")
    attempt, observed = run_assignment.resolve_resume_attempt(
        tmp_path, logical_run_id
    )
    assert attempt == 0
    assert observed == retained


def test_solver_forwards_sigterm_for_child_checkpoint_finalizer() -> None:
    import signal
    import threading

    previous = signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        return_code, forwarded = run_assignment.run_solver_process(
            [
                sys.executable,
                "-c",
                (
                    "import signal,sys,time; "
                    "signal.signal(signal.SIGTERM, lambda *_: sys.exit(143)); "
                    "time.sleep(30)"
                ),
            ],
            cwd=REPO,
            env=os.environ,
            timeout_seconds=10,
        )
    finally:
        timer.cancel()
    assert return_code == 143
    assert forwarded == signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == previous


def test_smoke_gate_accepts_all_ten_systems_and_binds_exact_pilot(tmp_path) -> None:
    output_root = _synthetic_smoke_output(tmp_path)
    gate = _build_synthetic_gate(output_root)
    assert gate["status"] == "pass"
    assert gate["formal_result_eligible"] is False
    assert gate["selected_run_count"] == 10
    assert len(gate["retained_attempts"]) == 10
    assert gate["gate_hash"] == _hash(gate, "gate_hash")
    schema = _read(ROOT / "schemas" / "smoke_gate.schema.json")
    jsonschema.validate(gate, schema)
    gate_path = tmp_path / "SMOKE_GATE.json"
    validate_smoke_gate.write_gate(gate_path, gate)
    pilot = _read(MANIFESTS / "pilot_manifest.json")
    verified = validate_smoke_gate.verify_gate_for_pilot(
        gate_path, pilot_manifest=pilot
    )
    assert verified["gate_hash"] == gate["gate_hash"]


def test_smoke_gate_accepts_retained_infrastructure_retry(tmp_path) -> None:
    output_root = _synthetic_smoke_output(tmp_path)
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    row = smoke["runs"][0]
    condition_root = output_root / row["logical_run_id"]
    attempt_zero = condition_root / "attempt-000"
    attempt_one = condition_root / "attempt-001"
    attempt_zero.rename(attempt_one)
    success_path = attempt_one / "MEASUREMENT.json"
    success = _read(success_path)
    success["attempt"] = 1
    success["retry_of"] = "attempt-000"
    success["journal_path"] = success["journal_path"].replace(
        "attempt-000", "attempt-001"
    )
    _write_hashed(success_path, success, "measurement_hash")
    failure = dict(success)
    failure.update(
        {
            "attempt": 0,
            "retry_of": None,
            "status": "retained_infrastructure_or_timeout_failure",
            "failure_class": "infrastructure",
            "completed": False,
            "terminal_score": None,
            "selected_candidate_id": None,
            "candidate_set_frozen": False,
            "candidate_set_hash": "",
            "terminal_report_sha256": "",
            "journal_path": "",
        }
    )
    _write_hashed(attempt_zero / "MEASUREMENT.json", failure, "measurement_hash")
    gate = _build_synthetic_gate(output_root)
    retained = next(
        item for item in gate["retained_attempts"] if item["system_id"] == row["system_id"]
    )
    selected = next(
        item for item in gate["selected_runs"] if item["system_id"] == row["system_id"]
    )
    assert len(retained["attempts"]) == 2
    assert retained["attempts"][0]["failure_class"] == "infrastructure"
    assert selected["attempt"] == 1


def test_smoke_gate_rejects_missing_system(tmp_path) -> None:
    import pytest

    output_root = _synthetic_smoke_output(tmp_path)
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    row = smoke["runs"][0]
    measurement = (
        output_root / row["logical_run_id"] / "attempt-000" / "MEASUREMENT.json"
    )
    measurement.unlink()
    with pytest.raises(ValueError, match="Missing retained measurement"):
        _build_synthetic_gate(output_root)


def test_smoke_gate_rejects_failed_system_without_valid_retry(tmp_path) -> None:
    import pytest

    output_root = _synthetic_smoke_output(tmp_path)
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    row = smoke["runs"][0]
    measurement_path = (
        output_root / row["logical_run_id"] / "attempt-000" / "MEASUREMENT.json"
    )
    measurement = _read(measurement_path)
    measurement.update(
        {
            "status": "retained_agent_failure",
            "failure_class": "agent",
            "completed": False,
            "terminal_score": None,
            "selected_candidate_id": None,
            "candidate_set_frozen": False,
            "candidate_set_hash": "",
            "terminal_report_sha256": "",
        }
    )
    _write_hashed(measurement_path, measurement, "measurement_hash")
    with pytest.raises(ValueError, match="no complete terminal-scored Smoke attempt"):
        _build_synthetic_gate(output_root)


def test_smoke_gate_rejects_no_memory_prompt_exposure(tmp_path) -> None:
    import pytest

    output_root = _synthetic_smoke_output(tmp_path)
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    row = next(item for item in smoke["runs"] if item["system_id"] == "no_memory")
    measurement = _read(
        output_root / row["logical_run_id"] / "attempt-000" / "MEASUREMENT.json"
    )
    journal_path = Path(measurement["journal_path"])
    journal = _read(journal_path)
    trace = journal["nodes"][0]["memory_routing_trace"]
    trace["selected_candidates"] = [trace["raw_candidates"][0]]
    trace["final_prompt_candidate_ids"] = [
        trace["raw_candidates"][0]["candidate_id"]
    ]
    trace["final_prompt_candidates"] = [
        {
            "candidate_id": trace["raw_candidates"][0]["candidate_id"],
            "source": trace["raw_candidates"][0]["source"],
            "source_stage": trace["raw_candidates"][0]["source_stage"],
            "source_task_id": trace["raw_candidates"][0]["source_task_id"],
            "prompt_text": "unauthorized exposure",
        }
    ]
    trace["suppressed_candidates"] = trace["suppressed_candidates"][1:]
    trace["prompt_token_count"] = 5
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    with pytest.raises(ValueError, match="external memory reached the Prompt"):
        _build_synthetic_gate(output_root)


def test_smoke_gate_rejects_memory_on_without_prompt_activation(tmp_path) -> None:
    import pytest

    output_root = _synthetic_smoke_output(tmp_path)
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    row = next(
        item for item in smoke["runs"] if item["system_id"] == "dynamic_hybrid"
    )
    measurement = _read(
        output_root / row["logical_run_id"] / "attempt-000" / "MEASUREMENT.json"
    )
    journal_path = Path(measurement["journal_path"])
    journal = _read(journal_path)
    trace = journal["nodes"][0]["memory_routing_trace"]
    trace["selected_candidates"] = []
    trace["final_prompt_candidate_ids"] = []
    trace["final_prompt_candidates"] = []
    trace["prompt_token_count"] = 0
    trace["suppressed_candidates"] = [
        {
            "candidate_id": item["candidate_id"],
            "source": item["source"],
            "reason": "not_selected_by_frozen_system_policy",
        }
        for item in trace["raw_candidates"]
    ]
    journal["nodes"][0]["memory_candidate_contract_refs"] = {}
    journal["nodes"][0]["experience_contract_refs"] = []
    journal["nodes"][0]["adoption_verification_plan"] = {}
    journal["nodes"][0]["adoption_runtime_trace"] = {}
    journal["nodes"][0]["adoption_verifier_verdict"] = {}
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    with pytest.raises(ValueError, match="no Prompt-visible route"):
        _build_synthetic_gate(output_root)


def test_smoke_gate_rejects_memory_on_without_candidate_contract_map(tmp_path) -> None:
    import pytest

    output_root = _synthetic_smoke_output(tmp_path)
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    row = next(item for item in smoke["runs"] if item["system_id"] == "runforest_only")
    measurement = _read(
        output_root / row["logical_run_id"] / "attempt-000" / "MEASUREMENT.json"
    )
    journal_path = Path(measurement["journal_path"])
    journal = _read(journal_path)
    journal["nodes"][0]["memory_candidate_contract_refs"] = {}
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(ValueError, match="does not cover every Prompt candidate"):
        _build_synthetic_gate(output_root)


def test_smoke_gate_rejects_tampered_agent_trace_signature(tmp_path) -> None:
    import pytest

    output_root = _synthetic_smoke_output(tmp_path)
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    row = next(item for item in smoke["runs"] if item["system_id"] == "dynamic_hybrid")
    measurement = _read(
        output_root / row["logical_run_id"] / "attempt-000" / "MEASUREMENT.json"
    )
    journal_path = Path(measurement["journal_path"])
    journal = _read(journal_path)
    journal["nodes"][0]["adoption_runtime_trace"]["signature_ed25519"] = (
        "A" * 88
    )
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    with pytest.raises(ValueError, match="signature mismatch"):
        _build_synthetic_gate(output_root)


def test_budget_couples_gpu_search_and_cpu_controls() -> None:
    budget = _read(MANIFESTS / "budget.json")
    assert budget["runtime"]["gpu_type"] == "NVIDIA A100 family"
    assert budget["runtime"]["gpu_resource_key"] == "nvidia.com/a100"
    assert budget["runtime"]["gpu_product_constraint"] is None
    for phase in ("smoke", "pilot"):
        row = budget[phase]
        assert row["gpu_count"] == row["parallel_search_num"] == 1
        assert row["cpu_count"] == 16
        assert row["memory_gib"] == 64
    assert budget["shared_memory"] == {
        "raw_candidate_max": 24,
        "raw_candidates_per_source": 12,
        "prompt_token_budget": 1536,
        "token_counter": "whitespace_split_v1",
        "top_k": 6,
        "visibility_token_budget": 4096,
        }
    assert budget["solver_sampling"] == {
        "comparison_axis": "memory_system",
        "baseline_release_id": "end2end-agent-v3",
        "code_temperature": 1.0,
        "feedback_temperature": 1.0,
        "adoption_verifier_temperature": 0.0,
        "all_other_frozen_axes_unchanged": True,
    }
    assert budget["adoption_verifier"] == {
        "enabled": False,
        "mode": "off",
        "reason": (
            "experiment tracks prompt visibility and code adoption without a "
            "blocking verifier"
        ),
    }
    assert budget["agent_semantic_protocol_review"] == {
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
    }
    assert budget["experiment_validation"] == {
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
    }
    assert budget["failure_policy"]["automatic_job_retry"] is False
    assert budget["failure_policy"]["preserve_all_attempts"] is True


def test_hardware_receipt_records_unpinned_a100_contract(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "NVIDIA A100-SXM4-80GB\n"

    monkeypatch.setenv("KUBERNETES_NODE_NAME", "node-a100.example")
    monkeypatch.setattr(run_assignment.subprocess, "run", lambda *args, **kwargs: Result())
    receipt = run_assignment.capture_hardware_receipt(
        {
            "gpu_resource_key": "nvidia.com/a100",
            "gpu_product_constraint": None,
        }
    )
    assert receipt == {
        "requested_gpu_resource": "nvidia.com/a100",
        "gpu_product_constraint": None,
        "node_name": "node-a100.example",
        "observed_gpu_products": ["NVIDIA A100-SXM4-80GB"],
        "gpu_query_error": "",
    }


def test_runner_selects_hardware_from_global_runtime_not_phase_budget() -> None:
    components = {
        "budget": {
            "runtime": {
                "gpu_resource_key": "nvidia.com/a100",
                "gpu_product_constraint": None,
            },
            "smoke": {
                "gpu_count": 1,
                "cpu_count": 16,
                "memory_gib": 64,
            },
        }
    }
    assert "runtime" not in components["budget"]["smoke"]
    assert run_assignment.frozen_hardware_runtime(components) == {
        "gpu_resource_key": "nvidia.com/a100",
        "gpu_product_constraint": None,
    }


def test_host_artifact_namespace_is_collision_free_and_safe() -> None:
    from config import _resolve_host_artifact_roots

    binding = {"report_root": "/host/reports", "runtime_artifact_root": "/host/runtime"}
    report, runtime = _resolve_host_artifact_roots(
        binding, "end2end-v1/run_id/attempt-000"
    )
    assert report == "/host/reports/runs/end2end-v1/run_id/attempt-000"
    assert runtime == "/host/runtime/runs/end2end-v1/run_id/attempt-000"
    for unsafe in ("/absolute", "../escape", "a/../b", "bad space"):
        import pytest

        with pytest.raises(ValueError, match="unsafe"):
            _resolve_host_artifact_roots(binding, unsafe)


def test_host_protocol_assets_are_not_bound_into_fast_experiment() -> None:
    memory = _read(MANIFESTS / "memory_bundles.json")
    assert "host_runtime_sdk_hash" not in memory
    assert "host_bindings_root" not in memory
    assert "host_task_bindings" not in memory
    assert "host_collector_public_key_ed25519" not in memory


def test_terminal_holdout_keeps_release_public_view_while_host_uses_dataviews() -> None:
    from config import _host_enforce_data_dir

    binding = {"data_view_root": "/host/protected-view"}
    assert _host_enforce_data_dir(
        "/release/fixed/train-view",
        binding,
        terminal_fixed_holdout=True,
    ) == "/release/fixed/train-view"
    assert _host_enforce_data_dir(
        "/legacy/public",
        binding,
        terminal_fixed_holdout=False,
    ) == "/host/protected-view"


def test_evaluator_uses_base_asset_binding_not_exp_c_hardware_control() -> None:
    evaluator = _read(MANIFESTS / "evaluators.json")
    assert (
        evaluator["aggregate_binding_hash"]
        == "668896c08bd1c748ccb5a89220312daccae19ef4ad066db642518f00b9d67e47"
    )
    assert evaluator["reuse_scope"] == (
        "task data and evaluator assets only; no Exp-C source/config/system manifest"
    )


def test_analyzer_orders_terminal_before_noncausal_mechanism(tmp_path) -> None:
    sys.path.insert(0, str(ROOT))
    try:
        import analyze_results
    finally:
        sys.path.pop(0)

    synthetic_root = _synthetic_smoke_output(tmp_path / "synthetic")
    smoke = _read(MANIFESTS / "smoke_manifest.json")
    synthetic_row = next(
        row for row in smoke["runs"] if row["system_id"] == "dynamic_hybrid"
    )
    synthetic_measurement = _read(
        synthetic_root
        / synthetic_row["logical_run_id"]
        / "attempt-000"
        / "MEASUREMENT.json"
    )

    outcomes = [
        {
            "logical_run_id": "no-memory",
            "task_id": "leaf-classification",
            "system_id": "no_memory",
            "completed": True,
            "terminal_score": 0.5,
            "direction": "minimize",
            "status": "scored_terminal_result",
            "time_to_first_valid_seconds": 10.0,
            "allocated_gpu_hours": 0.1,
            "llm_token_usage": None,
            "llm_cost_usd": None,
            "journal_path": str(tmp_path / "missing.json"),
        },
        {
            "logical_run_id": "dynamic",
            "task_id": "leaf-classification",
            "system_id": "dynamic_hybrid",
            "completed": True,
            "terminal_score": 0.6,
            "direction": "minimize",
            "status": "scored_terminal_result",
            "time_to_first_valid_seconds": 12.0,
            "allocated_gpu_hours": 0.2,
            "llm_token_usage": None,
            "llm_cost_usd": None,
            "journal_path": synthetic_measurement["journal_path"],
        },
    ]
    terminal = analyze_results.terminal_summary(
        outcomes, ["leaf-classification"], ["no_memory", "dynamic_hybrid"]
    )
    dynamic = next(row for row in terminal["systems"] if row["system_id"] == "dynamic_hybrid")
    assert terminal["analysis_order"] == 1
    assert dynamic["negative_transfer_rate"] == 1.0
    assert dynamic["cells"][0]["normalized_delta_vs_no_memory"] < 0
    mechanism = analyze_results.mechanism_summary(
        outcomes,
        _test_collector_public_key_ed25519=(
            TEST_COLLECTOR_IDENTITY.public_key_ed25519
        ),
    )
    dynamic_mechanism = next(
        row for row in mechanism["runs"] if row["system_id"] == "dynamic_hybrid"
    )
    assert mechanism["analysis_order"] == 2
    assert mechanism["definitions"]["causal_attribution"] is False
    assert dynamic_mechanism["static_adopted"] == 1
    assert dynamic_mechanism["runtime_activated"] == 1
    assert dynamic_mechanism["adopted"] == 1
    assert dynamic_mechanism["invalid_agent_evidence_routes"] == 0
