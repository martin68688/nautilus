from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments" / "end2end_memory_systems_20260804"
ENDPOINT = "http://cliproxyapi-haoming.ecepxie.svc.cluster.local:8317/v1"
SECRET = "cliproxyapi-haoming-client"


def _pod(name: str) -> dict:
    path = EXP / "jobs_uci_transfer_3h_clip_r1" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_both_three_hour_pods_use_only_private_clip_endpoint_secret():
    for name in (
        "mlevolve-uci100-v147-3h-clip-r1-dev.yaml",
        "mlevolve-uci100-v149-3h-clip-r1-dev.yaml",
    ):
        pod = _pod(name)
        assert pod["metadata"]["labels"]["ecepxie.nrp/owner"] == "haoming"
        assert pod["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "codex-nrp-training"
        assert pod["spec"]["activeDeadlineSeconds"] == 14400
        container = pod["spec"]["containers"][0]
        assert container["envFrom"] == [{"secretRef": {"name": SECRET}}]
        assert {item["name"]: item["value"] for item in container["env"]}[
            "OPENAI_MODEL"
        ] == "gpt-5.6-sol"
        assert container["resources"]["limits"] == container["resources"]["requests"]
        workspace_volumes = [
            volume
            for volume in pod["spec"]["volumes"]
            if "persistentVolumeClaim" in volume
        ]
        assert workspace_volumes == [
            {
                "name": "workspace",
                "persistentVolumeClaim": {"claimName": "haoming-storage"},
            }
        ]
        text = (EXP / "jobs_uci_transfer_3h_clip_r1" / name).read_text(
            encoding="utf-8"
        )
        assert "apizh.net" not in text
        assert "mlevolve-openai-gpt56sol-v1" not in text
        assert "DEEPSEEK_" not in text


def test_both_runs_are_three_hour_and_practically_step_unbounded():
    for relative in (
        "systems_v147_transfer_3h_clip_r1/dynamic_cross_task_transfer.yaml",
        "systems_v149_dynamic_transfer_3h_clip_r1/dynamic_cross_task_transfer.yaml",
    ):
        config = yaml.safe_load((EXP / relative).read_text(encoding="utf-8"))
        assert config["agent"]["time_limit"] == 10800
        assert config["agent"]["steps"] == 2147483647


def test_executable_defaults_and_future_v147_builder_no_longer_use_relay():
    paths = (
        REPO / "mlevolve" / "config" / "config.yaml",
        REPO / "mlevolve" / "analysis" / "adoption_verifier_smoke.py",
        EXP / "build_uci_transfer_v147_smoke.py",
        EXP / "build_uci_transfer_3h_clip_r1.py",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert ENDPOINT in combined
    assert SECRET in combined
    assert "apizh.net" not in combined
    assert "mlevolve-openai-gpt56sol-v1" not in combined
