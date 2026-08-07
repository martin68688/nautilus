from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
QUEUE = REPO / "coordination" / "end2end_execution_queue_v23.json"
EXPERIMENT = REPO / "experiments" / "end2end_memory_systems_20260804"
SUBMISSION_VALIDATOR_PATH = (
    EXPERIMENT / "analysis_v23" / "validate_queue_submission.py"
)
SUBMISSION_SPEC = importlib.util.spec_from_file_location(
    "validate_queue_submission", SUBMISSION_VALIDATOR_PATH
)
SUBMISSION_VALIDATOR = importlib.util.module_from_spec(SUBMISSION_SPEC)
assert SUBMISSION_SPEC.loader is not None
SUBMISSION_SPEC.loader.exec_module(SUBMISSION_VALIDATOR)
CANCELLATION = (
    EXPERIMENT
    / "infrastructure_attempts"
    / "20260807_v24_fairness_stop.json"
)
FOURWAY_RETRY = (
    EXPERIMENT
    / "infrastructure_attempts"
    / "20260807_leaf_fourway_external_deletion_fresh_retry_v25.json"
)
STATIC_V26_RETRY = (
    EXPERIMENT
    / "infrastructure_attempts"
    / "20260807_leaf_static_preemption_fresh_retry_v26.json"
)


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


def test_leaf_execution_queue_is_exactly_bound_to_frozen_resume_manifest() -> None:
    queue = _read(QUEUE)
    assert queue["gpu_resource"] == "nvidia.com/a100"
    assert queue["max_total_gpu_parallelism"] == 4
    assert queue["monitor_interval_minutes"] >= 30
    observed = datetime.fromisoformat(queue["last_cluster_observation_at"])
    next_monitor = datetime.fromisoformat(queue["next_monitor_not_before"])
    assert (next_monitor - observed).total_seconds() >= 30 * 60
    assert queue["submission_rules"]["never_stop_active_pending_or_running"] is True
    assert queue["submission_rules"][
        "explicit_user_cancellation_is_only_stop_exception"
    ] is True
    assert queue["submission_rules"]["normal_creating_or_pending_is_not_deleted"] is True
    assert queue["submission_rules"][
        "fresh_unstarted_condition_must_not_use_resume"
    ] is True
    assert queue["submission_rules"]["exact_workload_preflight_required"] is True
    assert REPO / queue["submission_rules"][
        "local_submission_validator"
    ] == SUBMISSION_VALIDATOR_PATH
    assert queue["submission_rules"]["seed_1_is_exploratory_only"] is True
    assert queue["submission_rules"][
        "never_retry_after_six_hour_search_budget_exhaustion"
    ] is True
    assert queue["submission_rules"][
        "infrastructure_retry_is_allowed_only_before_search_budget_exhaustion"
    ] is True
    assert queue["submission_rules"][
        "infrastructure_retry_requires_fairness_runner_release"
    ] is True
    runner = EXPERIMENT / "run_assignment.py"
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == queue[
        "submission_rules"
    ]["fairness_runner_sha256"]
    assert queue["submission_rules"][
        "budget_exhausted_partial_is_a_terminal_experimental_outcome"
    ] is True

    frozen = _read(EXPERIMENT / "manifests_resume_v23" / "pilot_manifest.json")
    expected = {
        "mlevolve-e2e-leaf-rcr-pilot-v23": (17, "rcr_router_style_port"),
        "mlevolve-e2e-leaf-runforest-pilot-v23": (18, "runforest_only"),
        "mlevolve-e2e-leaf-macla-pilot-v23": (19, "macla_style_port"),
    }
    scheduled = (
        queue.get("leaf_submitted_in_priority_order", [])
        + queue["leaf_pending_priority"]
    )
    assert [row["workload"] for row in scheduled] == list(expected)
    for row in scheduled:
        workload = row["workload"]
        index, system_id = expected[workload]
        path = REPO / row["manifest"]
        if "manifest_sha256" in row:
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[
                "manifest_sha256"
            ]
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
        if workload == "mlevolve-e2e-leaf-macla-pilot-v23":
            assert "--resume" not in args
            assert job["metadata"]["annotations"][
                "mlevolve.ai/attempt-mode"
            ] == "fresh-only"
        else:
            # These two jobs were already submitted before the fresh-condition
            # fail-closed rule was frozen; both started from absent roots.
            assert args[-1] == "--resume"
        assert "--resume-source-attempt" not in args
        manifest_row = frozen["runs"][index]
        assert manifest_row["task_id"] == "leaf-classification"
        assert manifest_row["system_id"] == system_id
        assert manifest_row["seed"] == 1
        assert manifest_row["formal_result_eligible"] is True


def test_fourway_external_deletion_retry_is_fresh_and_preserves_attempt_zero() -> None:
    queue = _read(QUEUE)
    record = _read(FOURWAY_RETRY)
    frozen = _read(EXPERIMENT / "manifests_resume_v23" / "pilot_manifest.json")
    expected = {
        "mlevolve-e2e-leaf-static-fresh-retry-v25": (
            14,
            "static_hybrid",
            "mlevolve-e2e-leaf-static-pilot-v23",
        ),
        "mlevolve-e2e-leaf-rcr-fresh-retry-v25": (
            17,
            "rcr_router_style_port",
            "mlevolve-e2e-leaf-rcr-pilot-v23",
        ),
        "mlevolve-e2e-leaf-runforest-fresh-retry-v25": (
            18,
            "runforest_only",
            "mlevolve-e2e-leaf-runforest-pilot-v23",
        ),
        "mlevolve-e2e-leaf-macla-fresh-retry-v25": (
            19,
            "macla_style_port",
            "mlevolve-e2e-leaf-macla-pilot-v23",
        ),
    }
    retries = queue["leaf_infrastructure_retries"][:4]
    assert [row["workload"] for row in retries] == list(expected)
    assert queue["active_at_last_observation"] == [
        "mlevolve-e2e-leaf-static-fresh-retry-v26",
        "mlevolve-e2e-leaf-rcr-fresh-retry-v25",
        "mlevolve-e2e-leaf-runforest-fresh-retry-v25",
        "mlevolve-e2e-leaf-macla-fresh-retry-v25",
    ]

    recorded_jobs = {
        row["workload"]: row for row in record["fresh_retry_jobs"]
    }
    assert set(recorded_jobs) == set(expected)
    assert record["policy"]["attempt_000_retained"] is True
    assert record["policy"]["attempt_001_search_resume_enabled"] is False
    assert record["policy"]["attempt_001_agent_time_limit_seconds"] == 21600
    assert record["policy"]["runner_sha256"] == queue["submission_rules"][
        "fairness_runner_sha256"
    ]
    assert len(record["interrupted_attempts"]) == 4
    assert all(row["attempt"] == 0 for row in record["interrupted_attempts"])
    assert all(row["measurement_hash"] for row in record["interrupted_attempts"])

    for row in retries:
        workload = row["workload"]
        index, system_id, source_workload = expected[workload]
        path = REPO / row["manifest"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == row["manifest_sha256"]
        assert digest == recorded_jobs[workload]["manifest_sha256"]
        job = yaml.safe_load(path.read_text(encoding="utf-8"))
        args = _assert_common_one_a100_job(job, workload)
        assert job["spec"]["activeDeadlineSeconds"] == 25200
        assert job["metadata"]["annotations"]["mlevolve.ai/attempt-mode"] == (
            "fresh-retry"
        )
        assert job["metadata"]["annotations"]["mlevolve.ai/retry-of"] == (
            source_workload
        )
        assert job["metadata"]["annotations"]["mlevolve.ai/runner-sha256"] == (
            queue["submission_rules"]["fairness_runner_sha256"]
        )
        assert "--resume" not in args
        assert args[args.index("--attempt") + 1] == "1"
        assert args[args.index("--index") + 1] == str(index)
        assert args[args.index("--manifest") + 1].endswith(
            "/manifests_resume_v23/pilot_manifest.json"
        )
        manifest_row = frozen["runs"][index]
        assert manifest_row["task_id"] == "leaf-classification"
        assert manifest_row["system_id"] == system_id
        assert manifest_row["seed"] == 1


def test_static_preemption_retry_v26_is_fresh_and_preserves_older_attempts() -> None:
    queue = _read(QUEUE)
    record = _read(STATIC_V26_RETRY)
    row = queue["leaf_infrastructure_retries"][-1]
    workload = "mlevolve-e2e-leaf-static-fresh-retry-v26"
    assert row["workload"] == workload
    assert row["source_attempt"] == 1
    assert row["attempt"] == 2
    path = REPO / row["manifest"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == row["manifest_sha256"]
    assert digest == record["fresh_retry"]["manifest_sha256"]

    job = yaml.safe_load(path.read_text(encoding="utf-8"))
    args = _assert_common_one_a100_job(job, workload)
    assert job["metadata"]["annotations"]["mlevolve.ai/attempt-mode"] == (
        "fresh-retry"
    )
    assert job["metadata"]["annotations"]["mlevolve.ai/retry-source-attempt"] == (
        "1"
    )
    assert "--resume" not in args
    assert "--resume-source-attempt" not in args
    assert args[args.index("--attempt") + 1] == "2"
    assert record["interrupted_attempt"]["failure_reason"] == (
        "kubernetes_preemption"
    )
    assert record["interrupted_attempt"]["records_retained"] is True
    assert record["fresh_retry"]["search_resume_enabled"] is False
    assert record["policy"]["attempt_000_retained"] is True
    assert record["policy"]["attempt_001_retained"] is True
    assert record["policy"]["old_workloads_deleted"] is False


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
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row[
            "manifest_sha256"
        ]
        job = yaml.safe_load(path.read_text(encoding="utf-8"))
        args = _assert_common_one_a100_job(job, row["workload"])
        assert job["spec"]["completionMode"] == "Indexed"
        assert job["spec"]["completions"] == row["completions"] == 10
        assert job["spec"]["parallelism"] == row["parallelism"] == 4
        assert job["spec"]["backoffLimitPerIndex"] == 0
        assert row["per_condition_search_budget_seconds"] == 21600
        assert row["global_job_deadline_seconds"] == 90000
        assert job["spec"]["activeDeadlineSeconds"] == row[
            "global_job_deadline_seconds"
        ]
        assert job["metadata"]["annotations"][
            "mlevolve.ai/per-index-deadline-seconds"
        ] == "25200"
        assert job["metadata"]["annotations"][
            "mlevolve.ai/global-deadline-seconds"
        ] == "90000"
        assert "--resume" not in args
        assert job["metadata"]["annotations"][
            "mlevolve.ai/attempt-mode"
        ] == "fresh-only"
        assert job["metadata"]["annotations"][
            "mlevolve.ai/manifest-state"
        ] == "frozen-ready"
        assert "mlevolve.ai/generated-not-submitted" not in job["metadata"][
            "annotations"
        ]
        task_id = job["metadata"]["labels"]["task"]
        observed.append(task_id)
    assert observed == expected_tasks


def test_v24_resume_attempts_are_retained_but_never_resubmitted_or_scored() -> None:
    queue = _read(QUEUE)
    cancellation = _read(CANCELLATION)
    cancelled = {
        "mlevolve-e2e-leaf-dynamic-resume-v24": {
            "measurement_hash": (
                "527a4f85404acd709535a420dad788393f471f13ebb8cdca2e7089dde2900971"
            ),
            "completed_steps": 51,
        },
        "mlevolve-e2e-leaf-flat-resume-v24": {
            "measurement_hash": (
                "8768918b3a519d5f181eed612affc752b8d5db3bf698674006fd2d2b5d299aa9"
            ),
            "completed_steps": 52,
        },
    }

    assert cancellation["policy"] == {
        "six_hour_search_budget_seconds": 21600,
        "never_retry_after_budget_exhaustion": True,
        "budget_exhausted_partial_is_terminal_experimental_outcome": True,
        "preserve_cancelled_attempt_evidence": True,
        "cancelled_attempt_is_formal_result_eligible": False,
    }
    rows = {row["workload"]: row for row in cancellation["cancelled_jobs"]}
    assert set(rows) == set(cancelled)
    for workload, expected in cancelled.items():
        row = rows[workload]
        assert row["measurement_hash"] == expected["measurement_hash"]
        assert row["completed_steps"] == expected["completed_steps"]
        assert row["total_steps"] == 80
        assert row["cumulative_agent_wall_seconds"] < 21600
        assert set(row["evidence_retained"]) >= {
            "MEASUREMENT.json",
            "journal.json",
            "RUN_OUTCOME.json",
        }

    queue_names = {
        *queue["active_at_last_observation"],
        *(row["workload"] for row in queue["leaf_submitted_in_priority_order"]),
        *(row["workload"] for row in queue["leaf_pending_priority"]),
    }
    assert set(cancelled).isdisjoint(queue_names)
    retained = {
        row["workload"]: row
        for row in queue["retained_terminal_at_last_observation"]
        if row["workload"] in cancelled
    }
    assert set(retained) == set(cancelled)
    for workload, expected in cancelled.items():
        assert retained[workload]["status"] == "retained_fairness_stop"
        assert retained[workload]["formal_result_eligible"] is False
        assert retained[workload]["measurement_hash"] == expected["measurement_hash"]


def test_pending_fresh_jobs_pass_lightweight_submission_validation() -> None:
    queue = _read(QUEUE)
    workloads = [
        *(row["workload"] for row in queue["leaf_pending_priority"]),
        *(row["workload"] for row in queue["task_jobs_after_leaf"]),
    ]
    assert workloads == [
        "mlevolve-e2e-aerial-pilot-v23",
        "mlevolve-e2e-denoising-pilot-v23",
        "mlevolve-e2e-taxi-pilot-v23",
    ]
    for workload in workloads:
        report = SUBMISSION_VALIDATOR.validate(QUEUE, workload)
        assert report["workload"] == workload
        assert report["attempt_mode"] == "fresh-only"
        assert report["status"] == "validated"
