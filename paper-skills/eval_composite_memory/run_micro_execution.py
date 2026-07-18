#!/usr/bin/env python3
"""Budget-enforcing T4 adapter runner.

The adapter receives one JSON request on stdin and must emit one JSON object on
stdout.  Hidden labels are never placed in that request.  A separate evaluator
may append the metric to the resulting receipt.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from core import CONDITIONS, REPORTS, sha256_file, sha256_value, write_json, write_jsonl


def run_adapter(
    adapter: list[str], request: dict[str, Any], *, timeout_sec: int, output_limit_bytes: int
) -> dict[str, Any]:
    started = time.time()
    try:
        process = subprocess.run(
            adapter,
            input=json.dumps(request),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            env={**os.environ, "RUNFOREST_HIDDEN_HOLDOUT": "unavailable"},
            check=False,
        )
        stdout = process.stdout[:output_limit_bytes]
        stderr = process.stderr[:output_limit_bytes]
        payload = json.loads(stdout) if process.returncode == 0 else {}
        status = "completed" if process.returncode == 0 else "failed"
        reason = "" if process.returncode == 0 else f"adapter_exit_{process.returncode}"
    except subprocess.TimeoutExpired as exc:
        payload, status, reason = {}, "failed", "wall_clock_timeout"
        stdout = str(exc.stdout or "")[:output_limit_bytes]
        stderr = str(exc.stderr or "")[:output_limit_bytes]
    except (json.JSONDecodeError, OSError) as exc:
        payload, status, reason = {}, "failed", f"adapter_protocol_error:{type(exc).__name__}"
        stdout, stderr = "", str(exc)[:output_limit_bytes]
    return {
        "status": status,
        "failure_reason": reason,
        "wall_clock_sec": time.time() - started,
        "adapter_payload": payload,
        "stdout_sha256": sha256_value(stdout),
        "stderr_tail": stderr[-1000:],
    }


def run(
    adapter: list[str], matrix_path: Path, *, timeout_sec: int, output_limit_bytes: int, dry_run: bool
) -> dict[str, Any]:
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    receipts = []
    for entry in matrix["runs"]:
        condition = str(entry["condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition}")
        request = {
            "schema": "runforest_composite_micro_request_v1",
            "task_id": entry["task_id"],
            "condition": condition,
            "seed": int(entry["seed"]),
            "budget": matrix["budget"],
            "train_data_path": entry["train_data_path"],
            "prediction_output_path": entry["prediction_output_path"],
            "hidden_holdout_labels_exposed": False,
            "holdout_isolation_mode": matrix.get("holdout_isolation_mode", "unspecified"),
        }
        result = (
            {"status": "dry_run", "failure_reason": "", "wall_clock_sec": 0.0, "adapter_payload": {}}
            if dry_run
            else run_adapter(adapter, request, timeout_sec=timeout_sec, output_limit_bytes=output_limit_bytes)
        )
        receipts.append(
            {
                "schema": "runforest_composite_micro_receipt_v1",
                **request,
                **result,
                "metric": None,
                "trusted": False,
                "rank_eligible": False,
                "external_evaluation_pending": result["status"] == "completed",
            }
        )
    path = REPORTS / "micro_execution_receipts_v1.jsonl"
    write_jsonl(path, receipts)
    report = {
        "schema": "runforest_composite_micro_report_v1",
        "run_count": len(receipts),
        "completed_count": sum(row["status"] == "completed" for row in receipts),
        "dry_run_count": sum(row["status"] == "dry_run" for row in receipts),
        "hidden_label_exposure_count": 0,
        "downstream_claim_allowed": False,
        "receipt_path": str(path),
        "receipt_sha256": sha256_file(path),
    }
    write_json(REPORTS / "micro_execution_report_v1.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--adapter", nargs="+", default=[])
    parser.add_argument("--timeout-sec", type=int, default=3600)
    parser.add_argument("--output-limit-bytes", type=int, default=1_000_000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.adapter:
        raise SystemExit("--adapter is required unless --dry-run is used")
    print(json.dumps(run(args.adapter, args.matrix, timeout_sec=args.timeout_sec, output_limit_bytes=args.output_limit_bytes, dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
