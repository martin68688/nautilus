from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

from experiments.prevalence_audit_20260729.run_full_runtime_gate import (
    candidate_source,
)


ROOT = Path(__file__).resolve().parents[1]
JOB_PATH = ROOT / "deploy" / "prevalence-audit-20260729-five-a100.yaml"
VALIDATOR = (
    ROOT / "experiments" / "prevalence_audit_20260729" / "validate_run_packet.py"
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_five_job_design_has_requested_resources_and_profiles():
    payload = yaml.safe_load(JOB_PATH.read_text(encoding="utf-8"))
    jobs = payload["items"]
    assert len(jobs) == 5
    expected = {
        "denoising-dirty-documents": ("64Gi", "natural", "11 29 47"),
        "leaf-classification": ("64Gi", "natural", "11 29 47"),
        "aerial-cactus-identification": ("64Gi", "natural", "11 29 47"),
        "new-york-city-taxi-fare-prediction": ("128Gi", "natural", "11 29 47"),
        "spooky-author-identification": (
            "64Gi",
            "spooky-positive-control",
            "20260729",
        ),
    }
    for job in jobs:
        name = job["metadata"]["name"]
        pod = job["spec"]["template"]["spec"]
        container = pod["containers"][0]
        env = {row["name"]: row["value"] for row in container["env"]}
        task_id = env["TASK_ID"]
        memory, profile, seeds = expected[task_id]
        assert name.startswith("mlev-prevalence-") and "-a100-r" in name
        for side in ("requests", "limits"):
            assert container["resources"][side] == {
                "cpu": "16",
                "memory": memory,
                "nvidia.com/a100": "1",
            }
        assert env["MEMORY_PROFILE"] == profile
        assert env["SEED_LIST"] == seeds
        assert container["image"].count("@sha256:") == 1
        assert sum("persistentVolumeClaim" in volume for volume in pod["volumes"]) == 1
        command = container["args"][0].lower()
        assert "git fetch" not in command
        assert "git pull" not in command
        assert "git clone" not in command
        assert "sleep " not in command
        assert "protocol_runtime.activation verify" in command
        assert "/mlevolve/config/config_prevalence_audit_20260729_host_enforce.yaml" in command
        assert "validate_run_packet.py" in command


def test_formal_prevalence_config_disables_methodology_but_keeps_runforest():
    config = (
        ROOT / "mlevolve/config/config_prevalence_audit_20260729_host_enforce.yaml"
    ).read_text(encoding="utf-8")
    base = (
        ROOT / "mlevolve/config/config_fourtask_graph_v2_all_features_host_shadow.yaml"
    ).read_text(encoding="utf-8")
    assert 'methodology_kb_path: ""' in config
    assert "methodology_dynamic: false" in config
    assert "external_skill_memory:" in base
    assert "enable: true" in base
    builder = (
        ROOT / "deploy/build_prevalence_host_protocol_bundles_20260729.sh"
    ).read_text(encoding="utf-8")
    assert "--timeout-seconds 60" in builder
    assert "--max-epochs" not in builder
    assert "--max-folds" not in builder
    assert "--max-models" not in builder


def test_exact_source_gate_covers_post_freeze_inference_for_both_tasks():
    for task_id in (
        "denoising-dirty-documents",
        "aerial-cactus-identification",
    ):
        tree = ast.parse(candidate_source(task_id))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        for function_name in ("candidate", "main"):
            function = functions[function_name]
            calls = [
                (node.lineno, node.func.attr)
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
            ]
            freeze_lines = [line for line, name in calls if name == "freeze_selection"]
            inference_lines = [line for line, name in calls if name == "inference_scope"]
            assert len(freeze_lines) == 1
            assert len(inference_lines) == 1
            assert freeze_lines[0] < inference_lines[0]


def test_positive_control_packet_gate_accepts_complete_traced_packet(tmp_path: Path):
    control_id = "control::spooky::known-invalid-1"
    receipt = "receipt-1"
    decision = {
        "schema": "mlevolve_prospective_claim_use_decision_v1",
        "run_id": "run-1",
        "task_id": "spooky-author-identification",
        "agent_seed": 20260729,
        "decision_id": "decision-1",
        "decision_stage": "draft",
        "operation": "generate_candidate",
        "protocol_ref": "protocol@1#hash",
        "raw_candidate_ids": [control_id],
        "raw_relevance_scores": [0.99],
        "raw_claim_ids": ["claim-1"],
        "raw_claim_types": ["method_hypothesis"],
        "shadow_authority_decisions": [
            {
                "candidate_id": control_id,
                "claim_id": "claim-1",
                "outcome": "deny",
            }
        ],
        "suppressed_candidate_ids": [control_id],
        "suppression_reasons": {
            control_id: {
                "candidate_id": control_id,
                "claim_id": "claim-1",
                "operation": "generate_candidate",
                "decision_stage": "draft",
                "protocol_ref": "protocol@1#hash",
                "receipt_refs": [receipt],
            }
        },
        "final_prompt_candidate_ids": [],
        "actual_action_hash": _sha("actual-action"),
        "actual_code_hash": _sha("actual-code"),
        "runtime_receipt_refs": [receipt, "host-counterfactual-receipt"],
        "counterfactual_action_hash": _sha("counterfactual-action"),
        "counterfactual_code_hash": _sha("counterfactual-code"),
        "counterfactual_status": "complete",
        "counterfactual_pair_id": "prospective-pair::test",
        "counterfactual_control_hash": _sha("counterfactual-control"),
        "counterfactual_memory_payload_hash": _sha("counterfactual-memory"),
        "counterfactual_prompt_hash": _sha("counterfactual-prompt"),
        "counterfactual_receipt_refs": ["host-counterfactual-receipt"],
    }
    (tmp_path / "prospective_decision_ledger.jsonl").write_text(
        json.dumps(decision) + "\n", encoding="utf-8"
    )
    (tmp_path / "decision_opportunities.jsonl").write_text(
        json.dumps({"decision_id": "decision-1"}) + "\n", encoding="utf-8"
    )
    (tmp_path / "RUN_OUTCOME.json").write_text(
        json.dumps({"status": "complete"}) + "\n", encoding="utf-8"
    )
    manifest = tmp_path / "MEMORY_MANIFEST.json"
    manifest.write_text(
        json.dumps({"controlled_candidate_ids": [control_id]}) + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--run-root",
            str(tmp_path),
            "--task-id",
            "spooky-author-identification",
            "--agent-seed",
            "20260729",
            "--memory-manifest",
            str(manifest),
            "--positive-control",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "ONLINE_PACKET_GATE.json").read_text())
    assert report["raw_logging_coverage"] == 1.0
    assert report["controlled_raw_count"] == 1
    assert report["controlled_prompt_visible_count"] == 0
