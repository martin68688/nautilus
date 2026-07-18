#!/usr/bin/env python3
"""Call an external Agent for the frozen R1-R3 replay-repair prompts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from core import EPISODES, MANIFESTS, REPORTS, read_json, read_jsonl, sha256_value, write_jsonl


REPO = Path(__file__).resolve().parents[2]
import sys
if str(REPO / "mlevolve") not in sys.path:
    sys.path.insert(0, str(REPO / "mlevolve"))

from agents.leakage_audit import (  # noqa: E402
    audit_code,
    build_repair_preservation_contract,
    format_repair_preservation_contract,
)


STAGES = ["data_scope", "validation_provenance", "cross_fit", "selection_freeze", "final_holdout"]


def _issue_feedback(code: str) -> list[dict[str, str]]:
    return [
        {
            "issue_code": str(row.get("issue_code") or ""),
            "evidence": str(row.get("evidence") or ""),
            "remediation": str(row.get("remediation") or ""),
        }
        for row in audit_code(code).get("issues", [])
    ]


def _request(case: dict[str, Any], condition: str) -> dict[str, Any]:
    matrix = read_json(MANIFESTS / "replay_matrix_v1.json")
    config = matrix["conditions"][condition]
    request: dict[str, Any] = {
        "schema": "runforest_composite_replay_agent_request_v1",
        "case_id": case["case_id"],
        "condition": condition,
        "task_family": case["task_family"],
        "source_code": case["code"],
        "audit_feedback": _issue_feedback(case["code"]),
        "instruction": (
            "Repair the evaluation protocol without using hidden labels. Return one JSON object with a Python "
            "program in `code`. Do not execute the original defective source."
        ),
        "gold_repair_code_exposed": False,
        "expected_issue_code_exposed": False,
    }
    if config["staged_protocol"]:
        request["required_stage_order"] = STAGES
        request["instruction"] += (
            " Implement the ordered five-stage protocol. Intermediate stages are journal-only and must not "
            "execute, rank, or write positive memory. Return `stage_history` with exactly five objects in the "
            "required order. For each of the first four objects, set `status` to `clean`, `executed` to false, "
            "`rank_eligible` to false, and `positive_memory_write` to false. The final_holdout object must describe "
            "a candidate awaiting isolated execution, not claim that execution occurred. Return "
            "`runtime_provenance` with status `unverified` and holdout_evaluation_count 0; never self-certify it clean."
        )
    if config["preservation_contract_exposed"]:
        contract = build_repair_preservation_contract(case["code"])
        request["preservation_contract"] = format_repair_preservation_contract(contract)
        request["instruction"] += " Preserve every protected model, feature, and training-budget component."
    return request


def run(
    adapter: list[str], *, timeout_sec: int, max_cases: int | None = None,
    conditions: set[str] | None = None, output_path: Path | None = None,
) -> dict[str, Any]:
    cases = read_jsonl(EPISODES / "replay_defects_v1.jsonl")
    if max_cases is not None:
        cases = cases[:max_cases]
    chosen = conditions or {"R1", "R2", "R3"}
    outputs: list[dict[str, Any]] = []
    for case in cases:
        for condition in ("R1", "R2", "R3"):
            if condition not in chosen:
                continue
            request = _request(case, condition)
            started = time.time()
            try:
                process = subprocess.run(
                    adapter,
                    input=json.dumps(request, ensure_ascii=False),
                    text=True,
                    capture_output=True,
                    timeout=timeout_sec,
                    check=False,
                )
                payload = json.loads(process.stdout) if process.returncode == 0 else {}
                status = "completed" if process.returncode == 0 and payload.get("code") else "failed"
                error = "" if status == "completed" else f"adapter_exit_{process.returncode}:{process.stderr[-500:]}"
            except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
                payload, status, error = {}, "failed", f"{type(exc).__name__}:{exc}"
            outputs.append(
                {
                    "schema": "runforest_composite_replay_candidate_v1",
                    "case_id": case["case_id"],
                    "condition": condition,
                    "prompt_sha256": sha256_value(request),
                    "code": str(payload.get("code") or ""),
                    "stage_history": payload.get("stage_history", []),
                    "runtime_provenance": payload.get("runtime_provenance", {}),
                    "model": payload.get("model"),
                    "input_tokens": payload.get("input_tokens"),
                    "output_tokens": payload.get("output_tokens"),
                    "status": status,
                    "error": error,
                    "latency_sec": time.time() - started,
                    "mock": False,
                }
            )
    path = output_path or (REPORTS / "replay_candidates_v1.jsonl")
    write_jsonl(path, outputs)
    return {
        "run_count": len(outputs),
        "completed_count": sum(row["status"] == "completed" for row in outputs),
        "path": str(path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", nargs="+", required=True)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--conditions", nargs="+", choices=["R1", "R2", "R3"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(
        args.adapter,
        timeout_sec=args.timeout_sec,
        max_cases=args.max_cases,
        conditions=set(args.conditions) if args.conditions else None,
        output_path=args.output,
    ), ensure_ascii=False, indent=2))
