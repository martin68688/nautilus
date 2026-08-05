#!/usr/bin/env python3
"""Monitor the exact End2End Agent Smoke Job and recycle stale Pending Pods."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


OWNER = "haoming"
MANAGER = "codex-nrp-training"


def kubectl(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    for key in (
        "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
        "https_proxy", "http_proxy", "all_proxy", "no_proxy",
    ):
        env.pop(key, None)
    attempts = 3 if check else 1
    result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(attempts):
        result = subprocess.run(
            ["kubectl", "--request-timeout=20s", *args],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 or not check:
            return result
        if attempt + 1 < attempts:
            print(
                json.dumps(
                    {
                        "event": "kubectl_read_retry",
                        "attempt": attempt + 1,
                        "stderr": result.stderr.strip()[-500:],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(2 ** attempt)
    assert result is not None
    result.check_returncode()
    raise AssertionError("unreachable")


def get_json(*args: str) -> dict[str, Any]:
    return json.loads(kubectl(*args, "-o", "json").stdout)


def index_count(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    total = 0
    for part in text.split(","):
        start, separator, end = part.partition("-")
        total += int(end) - int(start) + 1 if separator else 1
    return total


def is_exact_owned_child_pod(
    pod: dict[str, Any], *, job_name: str, job_uid: str
) -> bool:
    """Require labels plus an exact controller UID before deleting a child Pod."""

    metadata = pod.get("metadata") or {}
    labels = metadata.get("labels") or {}
    owner_refs = metadata.get("ownerReferences") or []
    owned_by_exact_job = any(
        ref.get("kind") == "Job"
        and ref.get("uid") == job_uid
        and ref.get("name") == job_name
        for ref in owner_refs
    )
    return (
        labels.get("ecepxie.nrp/owner") == OWNER
        and labels.get("app.kubernetes.io/managed-by") == MANAGER
        and labels.get("job-name") == job_name
        and owned_by_exact_job
    )


def archive_stale_pending_pod(
    pod: dict[str, Any], *, job: dict[str, Any], artifact_dir: Path
) -> Path:
    """Preserve the scheduler failure before the user-requested Pod deletion."""

    metadata = pod.get("metadata") or {}
    annotations = metadata.get("annotations") or {}
    index = str(annotations.get("batch.kubernetes.io/job-completion-index") or "")
    payload = {
        "schema": "mlevolve_end2end_pending_infrastructure_attempt_v1",
        "failure_class": "infrastructure",
        "reason": "pending_exceeded_user_limit",
        "retry_required": True,
        "job_name": str((job.get("metadata") or {}).get("name") or ""),
        "job_uid": str((job.get("metadata") or {}).get("uid") or ""),
        "pod_name": str(metadata.get("name") or ""),
        "pod_uid": str(metadata.get("uid") or ""),
        "completion_index": index,
        "pod_created_at": str(metadata.get("creationTimestamp") or ""),
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "pod_status": pod.get("status") or {},
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    safe_job = payload["job_name"].replace("/", "_")
    safe_pod = payload["pod_name"].replace("/", "_")
    path = artifact_dir / f"{safe_job}__index-{index}__{safe_pod}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="ecepxie")
    parser.add_argument("--job", required=True)
    parser.add_argument("--pending-seconds", type=int, default=180)
    parser.add_argument("--interval-seconds", type=int, default=15)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(
            "experiments/end2end_memory_systems_20260804/infrastructure_attempts"
        ),
    )
    args = parser.parse_args()

    job = get_json("get", "job", args.job, "-n", args.namespace)
    labels = job["metadata"].get("labels") or {}
    if labels.get("ecepxie.nrp/owner") != OWNER:
        raise RuntimeError("Job lacks the exact positive owner label")
    if labels.get("app.kubernetes.io/managed-by") != MANAGER:
        raise RuntimeError("Job lacks the exact manager label")
    job_uid = str(job["metadata"]["uid"])
    last_snapshot = ""
    recycled: set[str] = set()
    while True:
        job = get_json("get", "job", args.job, "-n", args.namespace)
        pods = get_json(
            "get", "pods", "-n", args.namespace,
            "-l", f"job-name={args.job}",
        ).get("items", [])
        status = job.get("status") or {}
        snapshot = json.dumps(
            {
                "active": int(status.get("active") or 0),
                "succeeded": int(status.get("succeeded") or 0),
                "failed_attempts": int(status.get("failed") or 0),
                "completed_indexes": index_count(status.get("completedIndexes")),
                "failed_indexes": index_count(status.get("failedIndexes")),
                "pods": sorted(
                    [
                        {
                            "name": pod["metadata"]["name"],
                            "index": (pod["metadata"].get("annotations") or {}).get(
                                "batch.kubernetes.io/job-completion-index"
                            ),
                            "phase": (pod.get("status") or {}).get("phase"),
                            "node": (pod.get("spec") or {}).get("nodeName"),
                        }
                        for pod in pods
                    ],
                    key=lambda row: row["name"],
                ),
            },
            sort_keys=True,
        )
        if snapshot != last_snapshot:
            print(snapshot, flush=True)
            last_snapshot = snapshot

        now = datetime.now(timezone.utc)
        for pod in pods:
            metadata = pod.get("metadata") or {}
            if (pod.get("status") or {}).get("phase") != "Pending":
                continue
            created = datetime.fromisoformat(
                str(metadata["creationTimestamp"]).replace("Z", "+00:00")
            )
            if (now - created).total_seconds() < args.pending_seconds:
                continue
            name = str(metadata.get("name") or "")
            if (
                not is_exact_owned_child_pod(
                    pod, job_name=args.job, job_uid=job_uid
                )
                or not name
                or name in recycled
            ):
                continue
            archive_path = archive_stale_pending_pod(
                pod, job=job, artifact_dir=args.artifact_dir
            )
            result = kubectl(
                "delete", "pod", name, "-n", args.namespace,
                "--wait=false", check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Failed to recycle stale Pending Pod {name}: {result.stderr}"
                )
            recycled.add(name)
            print(
                json.dumps(
                    {
                        "event": "stale_pending_index_failed_retry_job_required",
                        "pod": name,
                        "index": (
                            metadata.get("annotations") or {}
                        ).get("batch.kubernetes.io/job-completion-index"),
                        "archive": str(archive_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        conditions = {
            str(row.get("type")): str(row.get("status"))
            for row in status.get("conditions") or []
        }
        if conditions.get("Complete") == "True" or conditions.get("Failed") == "True":
            print(
                json.dumps(
                    {"event": "job_terminal", "conditions": conditions},
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0
        time.sleep(max(5, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
