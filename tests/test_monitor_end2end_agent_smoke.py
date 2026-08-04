from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "deploy" / "monitor_end2end_agent_smoke.py"
SPEC = importlib.util.spec_from_file_location("monitor_end2end_agent_smoke", PATH)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def _pod() -> dict:
    return {
        "metadata": {
            "name": "smoke-0-abcde",
            "labels": {
                "ecepxie.nrp/owner": "haoming",
                "app.kubernetes.io/managed-by": "codex-nrp-training",
                "job-name": "smoke-v2",
            },
            "ownerReferences": [
                {"kind": "Job", "name": "smoke-v2", "uid": "job-uid"}
            ],
        }
    }


def test_index_count_handles_indexed_job_ranges() -> None:
    assert MONITOR.index_count(None) == 0
    assert MONITOR.index_count("0,2-4,9") == 5


def test_pending_recycle_requires_labels_and_exact_controller_uid() -> None:
    pod = _pod()
    assert MONITOR.is_exact_owned_child_pod(
        pod, job_name="smoke-v2", job_uid="job-uid"
    )

    for mutation in (
        lambda row: row["metadata"]["labels"].update(
            {"ecepxie.nrp/owner": "someone-else"}
        ),
        lambda row: row["metadata"]["labels"].update(
            {"app.kubernetes.io/managed-by": "someone-else"}
        ),
        lambda row: row["metadata"]["labels"].update(
            {"job-name": "different-job"}
        ),
        lambda row: row["metadata"]["ownerReferences"][0].update(
            {"uid": "different-uid"}
        ),
    ):
        candidate = _pod()
        mutation(candidate)
        assert not MONITOR.is_exact_owned_child_pod(
            candidate, job_name="smoke-v2", job_uid="job-uid"
        )


def test_kubectl_read_retries_transient_failures(monkeypatch) -> None:
    calls = []
    outcomes = [
        MONITOR.subprocess.CompletedProcess(["kubectl"], 1, "", "timeout"),
        MONITOR.subprocess.CompletedProcess(["kubectl"], 0, "{}", ""),
    ]

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return outcomes.pop(0)

    monkeypatch.setattr(MONITOR.subprocess, "run", fake_run)
    monkeypatch.setattr(MONITOR.time, "sleep", lambda _seconds: None)

    result = MONITOR.kubectl("get", "pods")
    assert result.returncode == 0
    assert len(calls) == 2
    assert calls[0][0][:2] == ["kubectl", "--request-timeout=20s"]
