#!/usr/bin/env python
"""Validate one frozen End2End Job immediately before cluster preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
DEFAULT_QUEUE = REPO / "coordination" / "end2end_execution_queue_v23.json"


def validate(queue_path: Path, workload: str) -> dict[str, str]:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    rows = [
        *(queue.get("leaf_pending_priority") or []),
        *(queue.get("task_jobs_after_leaf") or []),
    ]
    matches = [row for row in rows if row.get("workload") == workload]
    if len(matches) != 1:
        raise ValueError(f"Workload is not exactly once in the pending queue: {workload}")
    row = matches[0]
    manifest = REPO / str(row["manifest"])
    observed_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if observed_hash != row.get("manifest_sha256"):
        raise ValueError(f"Frozen Job manifest SHA-256 mismatch: {workload}")

    job = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    metadata = job["metadata"]
    labels = metadata["labels"]
    if metadata.get("name") != workload:
        raise ValueError("Job name does not match queued workload")
    if labels.get("mlevolve.ai/workload") != workload:
        raise ValueError("Workload label does not match queued workload")
    if labels.get("ecepxie.nrp/owner") != "haoming":
        raise ValueError("Job is missing positive owner identity")
    if labels.get("app.kubernetes.io/managed-by") != "codex-nrp-training":
        raise ValueError("Job is missing managed-by identity")

    annotations = metadata.get("annotations") or {}
    args = list(job["spec"]["template"]["spec"]["containers"][0]["args"])
    if annotations.get("mlevolve.ai/attempt-mode") == "fresh-only":
        if "--resume" in args or "--resume-source-attempt" in args:
            raise ValueError("Fresh-only workload contains a resume argument")
    return {
        "schema": "mlevolve_end2end_queue_submission_validation_v1",
        "workload": workload,
        "manifest": str(manifest),
        "manifest_sha256": observed_hash,
        "attempt_mode": str(annotations.get("mlevolve.ai/attempt-mode") or ""),
        "status": "validated",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--workload", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.queue, args.workload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
