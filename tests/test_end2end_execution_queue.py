from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "coordination" / "end2end_execution_queue_v23.json"
EXPERIMENT = REPO / "experiments" / "end2end_memory_systems_20260804"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_common_one_a100_job(job: dict, workload: str) -> list[str]:
    assert job["apiVersion"] == "batch/v1"
    assert job["kind"] == "Job"
    assert job["metadata"]["name"] == workload
    assert job["metadata"]["namespace"] == "ecepxie"
    labels = job["metadata"]["labels"]
    assert labels["ecepxie.nrp/owner"] == "haoming"
    assert labels["app.kubernetes.io/managed-by"] == "codex-nrp-training"
    assert labels["mlevolve.ai/workload"] == workload
    assert job["metadata"]["annotations"]["mlevolve.ai/gpu-contract"] == (
        "nvidia.com/a100=1"
    )
    spec = job["spec"]
    assert spec.get("backoffLimit", spec.get("backoffLimitPerIndex")) == 0
    assert "affinity" not in spec["template"]["spec"]
    container = spec["template"]["spec"]["containers"][0]
    assert container["command"] == ["/usr/local/bin/python", "-u"]
    requests = container["resources"]["requests"]
    assert requests == container["resources"]["limits"]
    assert requests == {
        "cpu": "16",
        "memory": "64Gi",
        "ephemeral-storage": "64Gi",
        "nvidia.com/a100": "1",
    }
    rendered = json.dumps(container)
    assert "sleep" not in rendered
    assert "tail -f" not in rendered
    volumes = spec["template"]["spec"]["volumes"]
    pvc_volumes = [row for row in volumes if "persistentVolumeClaim" in row]
    assert pvc_volumes == [
        {
            "name": "workspace",
            "persistentVolumeClaim": {"claimName": "haoming-storage"},
        }
    ]
    return [str(value) for value in container["args"]]


def test_leaf_pending_queue_is_exactly_bound_to_frozen_resume_manifest() -> None:
    queue = _read(QUEUE)
    assert queue["gpu_resource"] == "nvidia.com/a100"
    assert queue["max_total_gpu_parallelism"] == 4
    assert queue["monitor_interval_minutes"] >= 30
    assert queue["submission_rules"]["never_stop_active_pending_or_running"] is True
    assert queue["submission_rules"]["normal_creating_or_pending_is_not_deleted"] is True
    assert queue["submission_rules"]["exact_workload_preflight_required"] is True
    assert queue["submission_rules"]["seed_1_is_exploratory_only"] is True

    frozen = _read(EXPERIMENT / "manifests_resume_v23" / "pilot_manifest.json")
    expected = {
        "mlevolve-e2e-leaf-rcr-pilot-v23": (17, "rcr_router_style_port"),
        "mlevolve-e2e-leaf-runforest-pilot-v23": (18, "runforest_only"),
        "mlevolve-e2e-leaf-macla-pilot-v23": (19, "macla_style_port"),
    }
    assert [row["workload"] for row in queue["leaf_pending_priority"]] == list(
        expected
    )
    for row in queue["leaf_pending_priority"]:
        workload = row["workload"]
        index, system_id = expected[workload]
        path = REPO / row["manifest"]
        job = yaml.safe_load(path.read_text(encoding="utf-8"))
        args = _assert_common_one_a100_job(job, workload)
        assert job["spec"]["activeDeadlineSeconds"] == 25200
        assert job["metadata"]["labels"]["task"] == "leaf-classification"
        assert args[args.index("--index") + 1] == str(index)
        assert args[args.index("--manifest") + 1].endswith(
            "/manifests_resume_v23/pilot_manifest.json"
        )
        assert args[args.index("--output-root") + 1] == (
            "/workspace/experiment-end2end-memory-agent-v21/runs"
        )
        assert args[-1] == "--resume"
        assert "--resume-source-attempt" not in args
        manifest_row = frozen["runs"][index]
        assert manifest_row["task_id"] == "leaf-classification"
        assert manifest_row["system_id"] == system_id
        assert manifest_row["seed"] == 1
        assert manifest_row["formal_result_eligible"] is True


def test_post_leaf_task_jobs_are_four_way_indexed_a100_blocks() -> None:
    queue = _read(QUEUE)
    assert queue["submission_rules"][
        "submit_task_indexed_job_only_when_no_other_gpu_workload_is_active"
    ] is True
    expected_tasks = [
        "aerial-cactus-identification",
        "denoising-dirty-documents",
        "new-york-city-taxi-fare-prediction",
    ]
    observed = []
    for row in queue["task_jobs_after_leaf"]:
        path = REPO / row["manifest"]
        job = yaml.safe_load(path.read_text(encoding="utf-8"))
        args = _assert_common_one_a100_job(job, row["workload"])
        assert job["spec"]["completionMode"] == "Indexed"
        assert job["spec"]["completions"] == row["completions"] == 10
        assert job["spec"]["parallelism"] == row["parallelism"] == 4
        assert job["spec"]["backoffLimitPerIndex"] == 0
        assert args[-1] == "--resume"
        task_id = job["metadata"]["labels"]["task"]
        observed.append(task_id)
    assert observed == expected_tasks
