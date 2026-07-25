from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from schema import read_json, sha256_file, sha256_json, utc_now, write_json_atomic


SYSTEM_PROMPT = """You distill execution traces into small single-purpose SOP clauses.
Return one JSON object with key sop_containers. Each container has title and clauses.
Each clause must have text, retrieval_text, claim_type_proposal, source_refs,
evidence_refs, applies_when, prevents, and publication_class_proposal.
Allowed claim types are method_hypothesis, debug_repair, audit_finding, and score.
Allowed publication proposals are diagnostic, candidate, and certified.
Every source/evidence ref must be copied exactly from the input trace. Separate
method/debug/audit/score content into different clauses. Never sign a Receipt,
decide ALLOW/DENY, claim protocol_agnostic, delete an audit warning, or upgrade a
score correlation into causality."""
MAX_NETWORK_ATTEMPTS = 3


def request_payload(
    trace: Mapping[str, Any],
    trace_text: str,
    *,
    model: str,
    temperature: float,
) -> dict[str, Any]:
    user_prompt = json.dumps(
        {
            "run_id": trace["run_id"],
            "branch_id": trace["branch_id"],
            "task_id": trace["task_id"],
            "available_refs": trace["refs"],
            "trace": trace_text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    identity = {
        "model": model,
        "temperature": temperature,
        "system_prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "user_prompt_hash": hashlib.sha256(user_prompt.encode()).hexdigest(),
        "trace_sha256": trace["sha256"],
    }
    return {
        "request_id": sha256_json(identity),
        "model": model,
        "temperature": temperature,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "identity": identity,
    }


def parse_response(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])
        value = json.loads(text)
    if not isinstance(value, Mapping):
        raise ValueError("Distiller response must be an object")
    containers = value.get("sop_containers")
    if not isinstance(containers, list):
        raise ValueError("Distiller response missing sop_containers")
    normalized: list[dict[str, Any]] = []
    for container in containers:
        if not isinstance(container, Mapping) or not str(container.get("title") or "").strip():
            raise ValueError("Every SOP container needs a title")
        clauses = container.get("clauses")
        if not isinstance(clauses, list) or not clauses:
            raise ValueError("Every SOP container needs at least one clause")
        normalized.append({"title": str(container["title"]).strip(), "clauses": clauses})
    return {"sop_containers": normalized}


def load_frozen_responses(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if path.suffix == ".jsonl":
        output = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            output[str(row["request_id"])] = row.get("response")
        return output
    payload = read_json(path)
    if isinstance(payload, Mapping) and "responses" in payload:
        payload = payload["responses"]
    if not isinstance(payload, Mapping):
        raise ValueError("Frozen responses must map request_id to response")
    return {str(key): value for key, value in payload.items()}


def call_deepseek(request: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required with --allow-network")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=180,
        # Retries are explicit in distill() so every attempt is auditable.
        max_retries=0,
    )
    response = client.chat.completions.create(
        model=request["model"],
        temperature=request["temperature"],
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": request["system_prompt"]},
            {"role": "user", "content": request["user_prompt"]},
        ],
    )
    usage = getattr(response, "usage", None)
    usage_payload = usage.model_dump() if hasattr(usage, "model_dump") else {}
    return response.choices[0].message.content or "", usage_payload


def _write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_saved_response(
    path: Path,
    request: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    record = read_json(path)
    if record.get("request_id") != request["request_id"]:
        raise ValueError(f"Saved response ID mismatch: {path}")
    if record.get("identity") != request["identity"]:
        raise ValueError(f"Saved response identity mismatch: {path}")
    response = record.get("response")
    parse_response(response)
    usage = record.get("usage") or {}
    attempts = [dict(value) for value in record.get("attempts") or []]
    return response, dict(usage), attempts


def _network_response(
    request: Mapping[str, Any],
    attempt_log_path: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    if attempt_log_path.exists():
        checkpoint = read_json(attempt_log_path)
        if checkpoint.get("identity") != request["identity"]:
            raise ValueError(f"Attempt log identity mismatch: {attempt_log_path}")
        attempts = [dict(value) for value in checkpoint.get("attempts") or []]
    first_attempt = len(attempts) + 1
    for offset in range(MAX_NETWORK_ATTEMPTS):
        attempt_number = first_attempt + offset
        try:
            raw_response, usage = call_deepseek(request)
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "request_status": "error",
                    "error_type": type(error).__name__,
                    "parse_status": "not_attempted",
                }
            )
            write_json_atomic(
                attempt_log_path,
                {
                    "schema": "deepseek_request_attempt_log_v1",
                    "request_id": request["request_id"],
                    "identity": request["identity"],
                    "attempts": attempts,
                    "completed": False,
                },
            )
            continue
        try:
            parsed = parse_response(raw_response)
        except Exception as error:
            attempts.append(
                {
                    "attempt": attempt_number,
                    "request_status": "ok",
                    "usage": usage,
                    "parse_status": "error",
                    "parse_error_type": type(error).__name__,
                    "response": raw_response,
                }
            )
            write_json_atomic(
                attempt_log_path,
                {
                    "schema": "deepseek_request_attempt_log_v1",
                    "request_id": request["request_id"],
                    "identity": request["identity"],
                    "attempts": attempts,
                    "completed": False,
                },
            )
            continue
        attempts.append(
            {
                "attempt": attempt_number,
                "request_status": "ok",
                "usage": usage,
                "parse_status": "ok",
            }
        )
        write_json_atomic(
            attempt_log_path,
            {
                "schema": "deepseek_request_attempt_log_v1",
                "request_id": request["request_id"],
                "identity": request["identity"],
                "attempts": attempts,
                "completed": True,
            },
        )
        return raw_response, usage, parsed, attempts
    raise RuntimeError(
        f"DeepSeek request {request['request_id']} failed after "
        f"{MAX_NETWORK_ATTEMPTS} new attempts"
    )


def distill(
    trace_manifest_path: str | Path,
    trace_root: str | Path,
    output_dir: str | Path,
    *,
    frozen_responses_path: str | Path | None = None,
    model: str = "deepseek-v4-flash",
    temperature: float = 0.0,
    allow_network: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    trace_manifest_path = Path(trace_manifest_path).resolve()
    trace_root = Path(trace_root).resolve()
    output_dir = Path(output_dir).resolve()
    completed_outputs = {
        "proposals.jsonl",
        "distillation_report.json",
        "frozen_responses.json",
    }
    if output_dir.exists() and any(
        (output_dir / name).exists() for name in completed_outputs
    ):
        raise FileExistsError(f"Distillation output is already finalized: {output_dir}")
    if output_dir.exists():
        unexpected = {
            path.name
            for path in output_dir.iterdir()
            if path.name not in {"raw_responses", "attempt_logs"}
        }
        if unexpected:
            raise FileExistsError(
                f"Distillation output contains unexpected partial files: {sorted(unexpected)}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_responses"
    raw_dir.mkdir(exist_ok=True)
    attempt_dir = output_dir / "attempt_logs"
    attempt_dir.mkdir(exist_ok=True)
    manifest = read_json(trace_manifest_path)
    frozen_path = Path(frozen_responses_path).resolve() if frozen_responses_path else None
    frozen = load_frozen_responses(frozen_path)
    proposals: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    request_reports: list[dict[str, Any]] = []
    frozen_output: dict[str, Any] = {}
    cache_misses: list[str] = []
    for trace in manifest.get("traces") or []:
        trace_path = trace_root / str(trace["path"])
        if sha256_file(trace_path) != trace["sha256"]:
            raise ValueError(f"Trace hash mismatch: {trace_path}")
        request = request_payload(
            trace,
            trace_path.read_text(encoding="utf-8"),
            model=model,
            temperature=temperature,
        )
        request_id = request["request_id"]
        saved_path = raw_dir / f"{request_id}.json"
        if saved_path.exists():
            raw_response, usage, attempts = _load_saved_response(
                saved_path,
                request,
            )
            parsed = parse_response(raw_response)
            source = "saved"
        elif request_id in frozen:
            raw_response = frozen[request_id]
            usage = {}
            parsed = parse_response(raw_response)
            attempts = [
                {
                    "attempt": 0,
                    "request_status": "not_sent",
                    "parse_status": "ok",
                    "source": "frozen",
                }
            ]
            source = "frozen"
        elif allow_network:
            raw_response, usage, parsed, attempts = _network_response(
                request,
                attempt_dir / f"{request_id}.json",
            )
            source = "network"
        else:
            cache_misses.append(request_id)
            continue
        if source != "saved":
            raw_record = {
                "schema": "deepseek_frozen_response_v1",
                "request_id": request_id,
                "source": source,
                "model": model,
                "temperature": temperature,
                "identity": request["identity"],
                "response": raw_response,
                "parsed_response": parsed,
                "usage": usage,
                "attempts": attempts,
                "retry_count": max(
                    0,
                    len(
                        [
                            value
                            for value in attempts
                            if value.get("request_status") != "not_sent"
                        ]
                    )
                    - 1,
                ),
                "parse_report": {
                    "status": "ok",
                    "failed_attempt_count": sum(
                        value.get("parse_status") == "error" for value in attempts
                    ),
                },
            }
            write_json_atomic(saved_path, raw_record)
        proposals.append(
            {
                "request_id": request_id,
                "run_id": trace["run_id"],
                "branch_id": trace["branch_id"],
                "task_id": trace["task_id"],
                "trace_sha256": trace["sha256"],
                "response": parsed,
            }
        )
        usages.append(usage)
        request_reports.append(
            {
                "request_id": request_id,
                "source": source,
                "attempt_count": len(
                    [
                        value
                        for value in attempts
                        if value.get("request_status") != "not_sent"
                    ]
                ),
                "retry_count": max(
                    0,
                    len(
                        [
                            value
                            for value in attempts
                            if value.get("request_status") != "not_sent"
                        ]
                    )
                    - 1,
                ),
                "request_failure_count": sum(
                    value.get("request_status") == "error" for value in attempts
                ),
                "parse_failure_count": sum(
                    value.get("parse_status") == "error" for value in attempts
                ),
            }
        )
        frozen_output[request_id] = raw_response
    if cache_misses:
        raise RuntimeError(
            "Frozen response cache is incomplete and network is disabled: "
            f"{cache_misses[:5]} (total={len(cache_misses)})"
        )
    proposals_path = output_dir / "proposals.jsonl"
    _write_jsonl_atomic(proposals_path, proposals)
    frozen_output_path = output_dir / "frozen_responses.json"
    write_json_atomic(
        frozen_output_path,
        {
            "schema": "deepseek_frozen_response_cache_v1",
            "model": model,
            "temperature": temperature,
            "system_prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "responses": frozen_output,
        },
    )
    report = {
        "schema": "sop_clause_distillation_report_v1",
        "created_at": created_at or utc_now(),
        "trace_manifest_hash": sha256_file(trace_manifest_path),
        "frozen_response_source_hash": (
            sha256_file(frozen_path) if frozen_path else ""
        ),
        "frozen_response_artifact": frozen_output_path.name,
        "frozen_response_artifact_hash": sha256_file(frozen_output_path),
        "model": model,
        "temperature": temperature,
        "system_prompt_hash": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "trace_count": len(manifest.get("traces") or []),
        "proposal_count": len(proposals),
        "network_allowed": allow_network,
        "all_responses_frozen_or_saved": len(proposals)
        == len(manifest.get("traces") or []),
        "proposals_sha256": sha256_file(proposals_path),
        "retry_count": sum(row["retry_count"] for row in request_reports),
        "request_failure_count": sum(
            row["request_failure_count"] for row in request_reports
        ),
        "parse_report": {
            "status": "ok",
            "request_count": len(request_reports),
            "parse_failure_count": sum(
                row["parse_failure_count"] for row in request_reports
            ),
        },
        "request_reports": request_reports,
        "usage": usages,
    }
    write_json_atomic(output_dir / "distillation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-responses", type=Path)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--created-at")
    args = parser.parse_args()
    report = distill(
        args.trace_manifest,
        args.trace_root,
        args.output_dir,
        frozen_responses_path=args.frozen_responses,
        model=args.model,
        temperature=args.temperature,
        allow_network=args.allow_network,
        created_at=args.created_at,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
