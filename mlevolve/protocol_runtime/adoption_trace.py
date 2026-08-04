from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from authority.adoption_verification import sha256_payload
from protocol_runtime.events import canonical_json


RAW_TRACE_SCHEMA = "agent_adoption_raw_line_trace_v1"
SEALED_TRACE_SCHEMA = "agent_adoption_runtime_trace_v1"


def render_line_trace_bootstrap(
    plan: Mapping[str, Any],
    *,
    output_path: str | Path,
    nonce: str,
    candidate_start_line: int,
) -> str:
    """Render task-independent Python line tracing for Agent-selected ranges.

    The bootstrap observes only whether candidate source lines execute. It does
    not know task names, model families, APIs, metrics, or memory semantics.
    """

    probes = [
        {
            "probe_id": str(probe["probe_id"]),
            "start_line": int(probe["start_line"]),
            "end_line": int(probe["end_line"]),
        }
        for row in plan.get("contract_results") or []
        for probe in row.get("runtime_probes") or []
    ]
    bootstrap = f"""\n# --- Nautilus generic adoption trace bootstrap (Host generated) ---
import atexit as __nautilus_atexit
import json as __nautilus_json
import os as __nautilus_os
import sys as __nautilus_sys
import threading as __nautilus_threading

__nautilus_adoption_probes = {probes!r}
__nautilus_adoption_hits = {{probe["probe_id"]: {{}} for probe in __nautilus_adoption_probes}}
__nautilus_candidate_start_line = {int(candidate_start_line)}
__nautilus_trace_output = {str(Path(output_path))!r}
__nautilus_trace_nonce = {str(nonce)!r}
__nautilus_candidate_filename = __file__
__nautil_owner_env = "NAUTILUS_ADOPTION_TRACE_OWNER_PID"
__nautilus_owner_pid = __nautilus_os.environ.setdefault(
    __nautil_owner_env, str(__nautilus_os.getpid())
)
__nautilus_trace_enabled = __nautilus_owner_pid == str(__nautilus_os.getpid())

def __nautilus_adoption_tracer(frame, event, arg):
    if event == "line" and frame.f_code.co_filename == __nautilus_candidate_filename:
        source_line = frame.f_lineno - __nautilus_candidate_start_line + 1
        for probe in __nautilus_adoption_probes:
            if probe["start_line"] <= source_line <= probe["end_line"]:
                by_line = __nautilus_adoption_hits[probe["probe_id"]]
                key = str(source_line)
                by_line[key] = by_line.get(key, 0) + 1
    return __nautilus_adoption_tracer

def __nautilus_write_adoption_trace():
    __nautilus_sys.settrace(None)
    payload = {{
        "schema": {RAW_TRACE_SCHEMA!r},
        "artifact_id": {str(plan.get("artifact_id") or "")!r},
        "code_sha256": {str(plan.get("code_sha256") or "")!r},
        "plan_hash": {str(plan.get("plan_hash") or "")!r},
        "nonce": __nautilus_trace_nonce,
        "process_id": __nautilus_os.getpid(),
        "probe_hits": __nautilus_adoption_hits,
    }}
    temporary = __nautilus_trace_output + ".tmp." + str(__nautilus_os.getpid())
    with open(temporary, "x", encoding="utf-8") as handle:
        __nautilus_json.dump(payload, handle, sort_keys=True)
        handle.write("\\n")
        handle.flush()
        __nautilus_os.fsync(handle.fileno())
    __nautilus_os.replace(temporary, __nautilus_trace_output)

if __nautilus_trace_enabled:
    __nautilus_atexit.register(__nautilus_write_adoption_trace)
    __nautilus_sys.settrace(__nautilus_adoption_tracer)
    __nautilus_threading.settrace(__nautilus_adoption_tracer)
    __nautilus_sys._getframe().f_trace = __nautilus_adoption_tracer
# --- End Nautilus generic adoption trace bootstrap ---
"""
    return bootstrap.lstrip("\n")


def bootstrap_for_prefix(
    plan: Mapping[str, Any],
    *,
    output_path: str | Path,
    nonce: str,
    prefix: str,
) -> str:
    """Render a bootstrap whose source-line offset matches its final prefix."""

    provisional = render_line_trace_bootstrap(
        plan,
        output_path=output_path,
        nonce=nonce,
        candidate_start_line=1,
    )
    candidate_start = str(prefix).count("\n") + provisional.count("\n") + 1
    final = render_line_trace_bootstrap(
        plan,
        output_path=output_path,
        nonce=nonce,
        candidate_start_line=candidate_start,
    )
    if final.count("\n") != provisional.count("\n"):
        raise ValueError("Adoption trace bootstrap line count is unstable")
    return final


def _read_raw_trace(path: str | Path) -> dict[str, Any]:
    requested = Path(path)
    if requested.is_symlink() or not requested.is_file():
        raise ValueError("Adoption runtime trace is missing or a symlink")
    payload = json.loads(requested.read_text(encoding="utf-8"))
    if payload.get("schema") != RAW_TRACE_SCHEMA:
        raise ValueError("Adoption runtime trace schema mismatch")
    return payload


def seal_runtime_trace(
    *,
    raw_path: str | Path,
    plan: Mapping[str, Any],
    nonce: str,
    exit_status: int,
    identity: Any | None = None,
) -> dict[str, Any]:
    """Validate child trace bindings and create a Host-owned signed envelope."""

    raw = _read_raw_trace(raw_path)
    for key, expected in (
        ("artifact_id", plan.get("artifact_id")),
        ("code_sha256", plan.get("code_sha256")),
        ("plan_hash", plan.get("plan_hash")),
        ("nonce", str(nonce)),
    ):
        if raw.get(key) != expected:
            raise ValueError(f"Adoption runtime trace binding mismatch: {key}")

    raw_hits = raw.get("probe_hits")
    if not isinstance(raw_hits, Mapping):
        raise ValueError("Adoption runtime trace probe_hits is invalid")
    expected_probes = {
        str(probe["probe_id"]): dict(probe)
        for row in plan.get("contract_results") or []
        for probe in row.get("runtime_probes") or []
    }
    if set(raw_hits) != set(expected_probes):
        raise ValueError("Adoption runtime trace probe set mismatch")

    results: list[dict[str, Any]] = []
    for probe_id in sorted(expected_probes):
        probe = expected_probes[probe_id]
        values = raw_hits.get(probe_id)
        if not isinstance(values, Mapping):
            raise ValueError("Adoption runtime trace probe row is invalid")
        executed_lines: list[int] = []
        hit_count = 0
        for raw_line, raw_count in values.items():
            line = int(raw_line)
            count = int(raw_count)
            if not (probe["start_line"] <= line <= probe["end_line"]):
                raise ValueError("Adoption trace contains an out-of-range line")
            if count <= 0:
                raise ValueError("Adoption trace contains a non-positive hit count")
            executed_lines.append(line)
            hit_count += count
        results.append(
            {
                "probe_id": probe_id,
                "kind": "line_range_executed",
                "executed": bool(executed_lines),
                "executed_lines": sorted(set(executed_lines)),
                "hit_count": hit_count,
            }
        )

    raw_sha256 = hashlib.sha256(Path(raw_path).read_bytes()).hexdigest()
    core = {
        "schema": SEALED_TRACE_SCHEMA,
        "artifact_id": str(plan.get("artifact_id") or ""),
        "code_sha256": str(plan.get("code_sha256") or ""),
        "plan_hash": str(plan.get("plan_hash") or ""),
        "raw_trace_sha256": raw_sha256,
        "exit_status": int(exit_status),
        "probe_results": results,
        "trace_hash": "",
        "signature_algorithm": "none",
        "public_key_ed25519": "",
        "signature_ed25519": "",
    }
    hash_input = {
        key: value
        for key, value in core.items()
        if key not in {"trace_hash", "signature_ed25519"}
    }
    core["trace_hash"] = sha256_payload(hash_input)
    if identity is not None:
        signed_payload = {
            key: value for key, value in core.items() if key != "signature_ed25519"
        }
        core["signature_algorithm"] = "ed25519"
        core["public_key_ed25519"] = str(identity.public_key_ed25519)
        # The algorithm/key fields are part of the final trace hash.
        hash_input = {
            key: value
            for key, value in core.items()
            if key not in {"trace_hash", "signature_ed25519"}
        }
        core["trace_hash"] = sha256_payload(hash_input)
        signed_payload = {
            key: value for key, value in core.items() if key != "signature_ed25519"
        }
        core["signature_ed25519"] = identity.sign_payload(signed_payload)
    return core


def verify_sealed_trace(trace: Mapping[str, Any], identity: Any | None = None) -> None:
    if trace.get("schema") != SEALED_TRACE_SCHEMA:
        raise ValueError("Unsupported sealed adoption trace schema")
    hash_input = {
        key: value
        for key, value in trace.items()
        if key not in {"trace_hash", "signature_ed25519"}
    }
    if trace.get("trace_hash") != sha256_payload(hash_input):
        raise ValueError("Sealed adoption trace hash mismatch")
    if trace.get("signature_algorithm") == "ed25519":
        if identity is None:
            raise ValueError(
                "A Host identity is required to verify this adoption trace"
            )
        signed_payload = {
            key: value for key, value in trace.items() if key != "signature_ed25519"
        }
        identity.verify_payload(
            signed_payload, str(trace.get("signature_ed25519") or "")
        )


__all__ = [
    "RAW_TRACE_SCHEMA",
    "SEALED_TRACE_SCHEMA",
    "bootstrap_for_prefix",
    "render_line_trace_bootstrap",
    "seal_runtime_trace",
    "verify_sealed_trace",
]
