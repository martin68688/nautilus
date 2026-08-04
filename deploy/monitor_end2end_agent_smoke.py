#!/usr/bin/env python3
"""Monitor the exact End2End Agent Smoke Job and recycle stale Pending Pods."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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
    return subprocess.run(
        ["kubectl", *args],
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", default="ecepxie")
    parser.add_argument("--job", required=True)
    parser.add_argument("--pending-seconds", type=int, default=180)
    parser.add_argument("--interval-seconds", type=int, default=15)
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
                    {"event": "stale_pending_pod_deleted", "pod": name},
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
