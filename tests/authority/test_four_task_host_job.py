from __future__ import annotations

from pathlib import Path

import yaml


MANIFEST = Path("deploy/jobs-fourtask-host-protocol-shadow-a100x1-r3.yaml")


def test_four_task_jobs_bind_host_shadow_without_enforce() -> None:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert payload["kind"] == "List"
    assert len(payload["items"]) == 4
    expected_tasks = {
        "aerial-cactus-identification",
        "denoising-dirty-documents",
        "leaf-classification",
        "new-york-city-taxi-fare-prediction",
    }
    observed = set()
    for job in payload["items"]:
        spec = job["spec"]
        assert spec["backoffLimit"] == 0
        assert spec["activeDeadlineSeconds"] == 28800
        pod = spec["template"]["spec"]
        assert pod["restartPolicy"] == "Never"
        container = pod["containers"][0]
        env = {entry["name"]: entry for entry in container["env"]}
        task = env["TASK_ID"]["value"]
        observed.add(task)
        command = container["args"][0]
        assert "config_authority_host_protocol_shadow.yaml" in command
        assert "config_authority_host_protocol_enforce.yaml" not in command
        assert "protocol_runtime.activation verify" in command
        assert "--expected-image-digest" in command
        assert "--expected-sdk-hash" in command
        assert "exec python -u run.py" in command
        assert "sleep" not in command
        assert "agent.search.num_gpus=1" in command
        assert "agent.search.parallel_search_num=1" in command
        assert "cpu_number=16" in command
        resources = container["resources"]
        assert resources["requests"] == resources["limits"]
        assert resources["limits"] == {
            "cpu": "16",
            "memory": "128Gi",
            "nvidia.com/a100": "1",
        }
        mounts = container["volumeMounts"]
        assert not any(mount["mountPath"] == "/workspace" for mount in mounts)
        assert any(mount["name"] == "host-bundle" and mount["readOnly"] for mount in mounts)
        assert any(mount["name"] == "task-public" and mount["readOnly"] for mount in mounts)
        assert any(mount["name"] == "source-archive" and mount["readOnly"] for mount in mounts)
        assert any(item["name"] == "stage-host-collector-key" for item in pod["initContainers"])
    assert observed == expected_tasks
