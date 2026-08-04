#!/usr/bin/env python3
"""Validate retained Smoke outcomes and issue the immutable Pilot launch gate.

The gate is deliberately derived from the complete ten-system Smoke evidence.
It is not a user-editable pass flag: every selected measurement, terminal
report, journal, routing trace, retry, and frozen-manifest binding is checked
before a canonical self-hashed ``SMOKE_GATE.json`` can be created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
MANIFESTS = ROOT / "manifests"
ATTEMPT = re.compile(r"attempt-(\d{3})$")
HASH = re.compile(r"[0-9a-f]{64}$")
TRACE_SCHEMA = "mlevolve_memory_routing_trace_v1"
GATE_SCHEMA = "mlevolve_end2end_smoke_gate_v1"
MEASUREMENT_SCHEMA = "mlevolve_end2end_condition_measurement_v1"
NO_MEMORY = "no_memory"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def require_hash(value: object, label: str) -> str:
    text = str(value or "")
    if not HASH.fullmatch(text):
        raise ValueError(f"{label} is not a SHA-256 digest")
    return text


def verify_self_hash(
    payload: Mapping[str, Any], field: str, label: str
) -> None:
    expected = require_hash(payload.get(field), f"{label} {field}")
    if payload_hash(payload, field) != expected:
        raise ValueError(f"{label} self-hash mismatch")


def load_manifest(path: Path, *, kind: str) -> dict[str, Any]:
    payload = read_object(path.resolve(strict=True))
    verify_self_hash(payload, "manifest_hash", path.name)
    if payload.get("schema") != "mlevolve_end2end_execution_manifest_v1":
        raise ValueError(f"Unexpected execution manifest schema: {path}")
    if payload.get("kind") != kind:
        raise ValueError(f"Expected {kind} execution manifest: {path}")
    return payload


def _component_manifest(name: str, expected_hash: object) -> dict[str, Any]:
    path = MANIFESTS / f"{name}.json"
    payload = read_object(path)
    verify_self_hash(payload, "manifest_hash", name)
    if payload["manifest_hash"] != expected_hash:
        raise ValueError(f"{name} component binding mismatch")
    return payload


def _below(root: Path, path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {path}")
    resolved_root = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"{label} escapes retained attempt: {resolved}") from error
    return resolved


def _candidate_ids(rows: object, label: str) -> list[str]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} must be a list")
    values: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} contains a non-object")
        value = str(row.get("candidate_id") or "")
        if not value:
            raise ValueError(f"{label} contains an empty candidate_id")
        values.append(value)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate candidate IDs")
    return values


def validate_trace(
    trace: Mapping[str, Any],
    *,
    system_id: str,
    task_id: str,
    bundle_manifest_sha256: str,
    prompt_token_budget: int,
    top_k: int,
    candidate_limit_per_source: int,
) -> dict[str, Any]:
    """Validate one node-level route and return its activation counters."""

    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"{system_id}: memory routing trace schema mismatch")
    if trace.get("system_id") != system_id:
        raise ValueError(f"{system_id}: routed as {trace.get('system_id')!r}")
    if trace.get("target_task_id") != task_id:
        raise ValueError(f"{system_id}: route targets the wrong task")
    if trace.get("candidate_pool_source") != "shared_authority_filtered_sop_runforest":
        raise ValueError(f"{system_id}: route did not use the shared authorized pool")
    if trace.get("raw_pool_observed") is not True:
        raise ValueError(f"{system_id}: route did not attest raw-pool observation")

    raw = trace.get("raw_candidates")
    raw_ids = _candidate_ids(raw, f"{system_id} raw candidates")
    if hashlib.sha256(canonical_bytes(raw)).hexdigest() != require_hash(
        trace.get("candidate_pool_hash"), f"{system_id} candidate pool hash"
    ):
        raise ValueError(f"{system_id}: candidate pool hash mismatch")
    source_counts = {"sop": 0, "runforest": 0}
    for row in raw:
        source = str(row.get("source") or "")
        if source not in source_counts:
            raise ValueError(f"{system_id}: unexpected candidate source {source!r}")
        source_counts[source] += 1
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("authorized") is not True:
            raise ValueError(f"{system_id}: raw pool contains an unauthorized candidate")
    if any(value > candidate_limit_per_source for value in source_counts.values()):
        raise ValueError(f"{system_id}: raw source pool exceeds the frozen limit")

    selected_ids = _candidate_ids(
        trace.get("selected_candidates"), f"{system_id} selected candidates"
    )
    suppressed_ids = _candidate_ids(
        trace.get("suppressed_candidates"), f"{system_id} suppressed candidates"
    )
    prompt_ids = trace.get("final_prompt_candidate_ids")
    if not isinstance(prompt_ids, list) or any(not str(value) for value in prompt_ids):
        raise ValueError(f"{system_id}: final Prompt candidate IDs are malformed")
    prompt_ids = [str(value) for value in prompt_ids]
    if len(prompt_ids) != len(set(prompt_ids)) or len(prompt_ids) > top_k:
        raise ValueError(f"{system_id}: final Prompt violates frozen Top-{top_k}")
    if not set(selected_ids) <= set(raw_ids):
        raise ValueError(f"{system_id}: selected candidate lies outside the raw pool")
    if len(selected_ids) > top_k:
        raise ValueError(f"{system_id}: selection violates frozen Top-{top_k}")
    if not set(prompt_ids) <= set(selected_ids):
        raise ValueError(f"{system_id}: Prompt candidate lies outside selection")
    if not set(suppressed_ids) <= set(raw_ids):
        raise ValueError(f"{system_id}: suppressed candidate lies outside raw pool")
    if set(suppressed_ids) != set(raw_ids) - set(prompt_ids):
        raise ValueError(f"{system_id}: suppression trace is incomplete")

    prompt_tokens = trace.get("prompt_token_count")
    if (
        not isinstance(prompt_tokens, int)
        or isinstance(prompt_tokens, bool)
        or not 0 <= prompt_tokens <= prompt_token_budget
    ):
        raise ValueError(f"{system_id}: Prompt exceeds frozen token budget")
    if bool(prompt_ids) != bool(prompt_tokens):
        raise ValueError(f"{system_id}: Prompt refs/token accounting disagree")
    route = trace.get("stage_route")
    if not isinstance(route, Mapping):
        raise ValueError(f"{system_id}: missing stage route")
    if route.get("stage") not in {"draft", "improve", "debug"}:
        raise ValueError(f"{system_id}: invalid routed stage")
    if route.get("top_k") != top_k:
        raise ValueError(f"{system_id}: routed Top-k drift")
    if route.get("prompt_token_budget") != prompt_token_budget:
        raise ValueError(f"{system_id}: routed Prompt budget drift")

    safety = trace.get("visibility_safety_gate")
    if not isinstance(safety, Mapping):
        raise ValueError(f"{system_id}: missing visibility safety gate")
    if safety.get("unauthorized_prompt_exposure") != 0:
        raise ValueError(f"{system_id}: unauthorized Prompt exposure observed")
    if safety.get("unauthorized_activation") != 0:
        raise ValueError(f"{system_id}: unauthorized activation observed")
    if trace.get("unauthorized_prompt_exposure") != 0:
        raise ValueError(f"{system_id}: route reports unauthorized Prompt exposure")

    bundle = trace.get("memory_bundle")
    if not isinstance(bundle, Mapping):
        raise ValueError(f"{system_id}: missing Memory Bundle identity")
    if bundle.get("manifest_sha256") != bundle_manifest_sha256:
        raise ValueError(f"{system_id}: Memory Bundle manifest drift")
    if not str(bundle.get("bundle_id") or ""):
        raise ValueError(f"{system_id}: Memory Bundle ID is empty")

    if system_id == NO_MEMORY:
        if selected_ids or prompt_ids or prompt_tokens:
            raise ValueError("no_memory: external memory reached the Prompt")
        if trace.get("memory_snapshot_bound_but_not_exposed") is not True:
            raise ValueError("no_memory: Bundle-bound non-exposure was not attested")
    else:
        if trace.get("memory_snapshot_bound_but_not_exposed") is not False:
            raise ValueError(f"{system_id}: memory-on route marked as non-exposed")

    return {
        "raw_candidate_count": len(raw_ids),
        "prompt_candidate_count": len(prompt_ids),
        "prompt_token_count": prompt_tokens,
        "stage": route["stage"],
    }


def validate_journal(
    path: Path,
    *,
    attempt_root: Path,
    system_id: str,
    task_id: str,
    bundle_manifest_sha256: str,
    selected_candidate_id: str,
    prompt_token_budget: int,
    top_k: int,
    candidate_limit_per_source: int,
) -> dict[str, Any]:
    path = _below(attempt_root, path, f"{system_id} journal")
    journal = read_object(path)
    nodes = journal.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError(f"{system_id}: journal nodes are missing")
    selected_nodes = [
        node
        for node in nodes
        if isinstance(node, Mapping) and node.get("id") == selected_candidate_id
    ]
    if len(selected_nodes) != 1:
        raise ValueError(f"{system_id}: terminal candidate is not unique in journal")
    selected_node = selected_nodes[0]
    if not str(selected_node.get("code") or "").strip():
        raise ValueError(f"{system_id}: terminal candidate code is not retained")
    exec_time = selected_node.get("exec_time")
    if (
        not isinstance(exec_time, (int, float))
        or isinstance(exec_time, bool)
        or not math.isfinite(float(exec_time))
        or float(exec_time) < 0
    ):
        raise ValueError(f"{system_id}: terminal candidate runtime is not retained")
    observation = selected_node.get("protocol_observation")
    full_runtime = (
        observation.get("host_full_runtime")
        if isinstance(observation, Mapping)
        else None
    )
    if not isinstance(full_runtime, Mapping) or full_runtime.get("status") != "pass":
        raise ValueError(f"{system_id}: terminal candidate lacks passing Host runtime")

    summaries = []
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError(f"{system_id}: journal contains a non-object node")
        trace = node.get("memory_routing_trace")
        if not trace:
            continue
        if not isinstance(trace, Mapping):
            raise ValueError(f"{system_id}: node routing trace is not an object")
        summaries.append(
            validate_trace(
                trace,
                system_id=system_id,
                task_id=task_id,
                bundle_manifest_sha256=bundle_manifest_sha256,
                prompt_token_budget=prompt_token_budget,
                top_k=top_k,
                candidate_limit_per_source=candidate_limit_per_source,
            )
        )
    if not summaries:
        raise ValueError(f"{system_id}: no complete memory routing trace in journal")
    raw_routes = sum(item["raw_candidate_count"] > 0 for item in summaries)
    prompt_routes = sum(item["prompt_candidate_count"] > 0 for item in summaries)
    if raw_routes == 0:
        raise ValueError(f"{system_id}: Smoke observed no real raw candidate pool")
    if system_id == NO_MEMORY:
        if prompt_routes:
            raise ValueError("no_memory: Smoke contains Prompt-visible memory")
    elif prompt_routes == 0:
        raise ValueError(f"{system_id}: Smoke contains no Prompt-visible route")
    return {
        "journal_sha256": sha256_file(path),
        "trace_count": len(summaries),
        "raw_pool_route_count": raw_routes,
        "prompt_visible_route_count": prompt_routes,
        "stages_observed": sorted({str(item["stage"]) for item in summaries}),
        "max_prompt_token_count": max(item["prompt_token_count"] for item in summaries),
        "terminal_candidate_code_retained": True,
        "terminal_candidate_host_runtime_pass": True,
    }


def validate_measurement(
    measurement: Mapping[str, Any],
    *,
    path: Path,
    row: Mapping[str, Any],
    smoke_manifest_hash: str,
    attempt: int,
) -> None:
    verify_self_hash(measurement, "measurement_hash", str(path))
    expected = {
        "schema": MEASUREMENT_SCHEMA,
        "logical_run_id": row["logical_run_id"],
        "attempt": attempt,
        "manifest_hash": smoke_manifest_hash,
        "task_id": row["task_id"],
        "system_id": row["system_id"],
        "seed": row["seed"],
        "formal_result_eligible": False,
        "exploratory_pilot": True,
    }
    for key, value in expected.items():
        if measurement.get(key) != value:
            raise ValueError(f"{row['system_id']} attempt {attempt}: {key} mismatch")
    score = measurement.get("terminal_score")
    if score is not None and (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        raise ValueError(f"{row['system_id']} attempt {attempt}: non-finite score")
    if measurement.get("completed") is True:
        if measurement.get("status") != "scored_terminal_result":
            raise ValueError(f"{row['system_id']} attempt {attempt}: not terminal-scored")
        if measurement.get("failure_class") != "none" or score is None:
            raise ValueError(f"{row['system_id']} attempt {attempt}: inconsistent success")
        if measurement.get("candidate_set_frozen") is not True:
            raise ValueError(f"{row['system_id']} attempt {attempt}: candidates not frozen")
        require_hash(
            measurement.get("candidate_set_hash"),
            f"{row['system_id']} attempt {attempt} candidate-set hash",
        )
        if not str(measurement.get("selected_candidate_id") or ""):
            raise ValueError(f"{row['system_id']} attempt {attempt}: no selected candidate")
    else:
        if score is not None:
            raise ValueError(f"{row['system_id']} attempt {attempt}: failed score is not null")
        if measurement.get("failure_class") not in {
            "agent",
            "evaluator",
            "infrastructure",
        }:
            raise ValueError(f"{row['system_id']} attempt {attempt}: invalid failure class")


def resolve_attempts(
    output_root: Path,
    *,
    row: Mapping[str, Any],
    smoke_manifest_hash: str,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    condition_root = output_root / str(row["logical_run_id"])
    if not condition_root.is_dir() or condition_root.is_symlink():
        raise ValueError(f"Missing retained Smoke condition: {row['logical_run_id']}")
    indexed: dict[int, tuple[dict[str, Any], Path]] = {}
    for directory in sorted(condition_root.glob("attempt-*")):
        match = ATTEMPT.fullmatch(directory.name)
        if not match or not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Malformed retained attempt: {directory}")
        attempt = int(match.group(1))
        measurement_path = directory / "MEASUREMENT.json"
        if not measurement_path.is_file() or measurement_path.is_symlink():
            raise ValueError(f"Missing retained measurement: {measurement_path}")
        measurement = read_object(measurement_path)
        validate_measurement(
            measurement,
            path=measurement_path,
            row=row,
            smoke_manifest_hash=smoke_manifest_hash,
            attempt=attempt,
        )
        indexed[attempt] = (measurement, directory)
    if not indexed:
        raise ValueError(f"No retained attempts: {row['logical_run_id']}")
    expected_indices = list(range(max(indexed) + 1))
    if sorted(indexed) != expected_indices:
        raise ValueError(f"Non-contiguous retry chain: {row['logical_run_id']}")
    inventory = []
    selected: tuple[dict[str, Any], Path] | None = None
    for attempt in expected_indices:
        measurement, directory = indexed[attempt]
        expected_retry = f"attempt-{attempt - 1:03d}" if attempt else None
        if measurement.get("retry_of") != expected_retry:
            raise ValueError(f"{row['system_id']} attempt {attempt}: retry_of mismatch")
        if attempt and indexed[attempt - 1][0].get("failure_class") != "infrastructure":
            raise ValueError(
                f"{row['system_id']} attempt {attempt}: retry did not follow infrastructure failure"
            )
        if measurement.get("completed") is True:
            if selected is not None:
                raise ValueError(f"{row['system_id']}: multiple successful attempts")
            selected = (measurement, directory)
        inventory.append(
            {
                "attempt": attempt,
                "completed": measurement.get("completed") is True,
                "failure_class": measurement.get("failure_class"),
                "status": measurement.get("status"),
                "measurement_hash": measurement["measurement_hash"],
            }
        )
    if selected is None:
        raise ValueError(f"{row['system_id']}: no complete terminal-scored Smoke attempt")
    if selected[0]["attempt"] != expected_indices[-1]:
        raise ValueError(f"{row['system_id']}: attempts exist after successful Smoke")
    return selected[0], selected[1], inventory


def build_smoke_gate(
    *,
    output_root: Path,
    smoke_manifest_path: Path = MANIFESTS / "smoke_manifest.json",
    pilot_manifest_path: Path = MANIFESTS / "pilot_manifest.json",
) -> dict[str, Any]:
    smoke = load_manifest(smoke_manifest_path, kind="smoke")
    pilot = load_manifest(pilot_manifest_path, kind="pilot")
    if smoke.get("formal_result_eligible") is not False:
        raise ValueError("Smoke manifest is incorrectly formal-result eligible")
    if smoke.get("exploratory_pilot") is not True:
        raise ValueError("Smoke is not marked exploratory")
    system_ids = list(smoke.get("system_ids") or [])
    rows = list(smoke.get("runs") or [])
    if len(system_ids) != 10 or len(set(system_ids)) != 10 or len(rows) != 10:
        raise ValueError("Smoke must contain exactly ten unique systems/runs")
    if {str(row.get("system_id") or "") for row in rows} != set(system_ids):
        raise ValueError("Smoke is not an exact one-run-per-system matrix")
    if smoke.get("bindings") != pilot.get("bindings"):
        raise ValueError("Smoke and Pilot component bindings differ")

    components = {
        key: _component_manifest(key, value)
        for key, value in (
            (name.removesuffix("_manifest_hash"), digest)
            for name, digest in smoke["bindings"].items()
        )
    }
    budget = components["budget"]
    shared = budget["shared_memory"]
    task_bundles = components["memory_bundles"]["task_bundles"]

    output_root = output_root.resolve(strict=True)
    selected_rows = []
    all_attempts = []
    for row in sorted(rows, key=lambda item: str(item["system_id"])):
        measurement, attempt_root, inventory = resolve_attempts(
            output_root,
            row=row,
            smoke_manifest_hash=smoke["manifest_hash"],
        )
        terminal_path = _below(
            attempt_root,
            attempt_root / "TERMINAL_SCORE_REPORT.json",
            f"{row['system_id']} terminal report",
        )
        if sha256_file(terminal_path) != require_hash(
            measurement.get("terminal_report_sha256"),
            f"{row['system_id']} terminal report hash",
        ):
            raise ValueError(f"{row['system_id']}: terminal report hash mismatch")
        journal_path = Path(str(measurement.get("journal_path") or ""))
        route = validate_journal(
            journal_path,
            attempt_root=attempt_root,
            system_id=str(row["system_id"]),
            task_id=str(row["task_id"]),
            bundle_manifest_sha256=str(
                task_bundles[row["task_id"]]["bundle_manifest_sha256"]
            ),
            selected_candidate_id=str(measurement["selected_candidate_id"]),
            prompt_token_budget=int(shared["prompt_token_budget"]),
            top_k=int(shared["top_k"]),
            candidate_limit_per_source=int(shared["raw_candidates_per_source"]),
        )
        selected_rows.append(
            {
                "logical_run_id": row["logical_run_id"],
                "system_id": row["system_id"],
                "attempt": measurement["attempt"],
                "measurement_hash": measurement["measurement_hash"],
                "terminal_report_sha256": measurement["terminal_report_sha256"],
                **route,
            }
        )
        all_attempts.append(
            {
                "logical_run_id": row["logical_run_id"],
                "system_id": row["system_id"],
                "attempts": inventory,
            }
        )

    gate = {
        "schema": GATE_SCHEMA,
        "status": "pass",
        "formal_result_eligible": False,
        "exploratory_pilot": True,
        "smoke_manifest_hash": smoke["manifest_hash"],
        "pilot_manifest_hash": pilot["manifest_hash"],
        "component_manifest_hashes": dict(smoke["bindings"]),
        "source_lock_manifest_hash": smoke["bindings"][
            "source_lock_manifest_hash"
        ],
        "system_ids": system_ids,
        "selected_run_count": len(selected_rows),
        "selected_runs": selected_rows,
        "retained_attempts": all_attempts,
        "gate_hash": "",
    }
    gate["gate_hash"] = payload_hash(gate, "gate_hash")
    return gate


def verify_gate_for_pilot(
    gate_path: Path,
    *,
    pilot_manifest: Mapping[str, Any],
    smoke_manifest_path: Path = MANIFESTS / "smoke_manifest.json",
) -> dict[str, Any]:
    """Fail closed unless ``gate_path`` authorizes this exact frozen Pilot."""

    path = gate_path.resolve(strict=True)
    if gate_path.is_symlink():
        raise ValueError("Smoke gate may not be a symlink")
    gate = read_object(path)
    verify_self_hash(gate, "gate_hash", "Smoke gate")
    if gate.get("schema") != GATE_SCHEMA or gate.get("status") != "pass":
        raise ValueError("Smoke gate is not a passing End2End gate")
    if gate.get("formal_result_eligible") is not False:
        raise ValueError("Smoke gate incorrectly claims formal eligibility")
    if gate.get("exploratory_pilot") is not True:
        raise ValueError("Smoke gate is not marked exploratory")
    smoke = load_manifest(smoke_manifest_path, kind="smoke")
    expected = {
        "smoke_manifest_hash": smoke["manifest_hash"],
        "pilot_manifest_hash": pilot_manifest["manifest_hash"],
        "component_manifest_hashes": dict(pilot_manifest["bindings"]),
        "source_lock_manifest_hash": pilot_manifest["bindings"][
            "source_lock_manifest_hash"
        ],
        "system_ids": list(smoke["system_ids"]),
        "selected_run_count": 10,
    }
    for key, value in expected.items():
        if gate.get(key) != value:
            raise ValueError(f"Smoke gate {key} binding mismatch")
    selected = gate.get("selected_runs")
    retained = gate.get("retained_attempts")
    if not isinstance(selected, list) or len(selected) != 10:
        raise ValueError("Smoke gate does not contain ten selected runs")
    if not isinstance(retained, list) or len(retained) != 10:
        raise ValueError("Smoke gate does not retain ten attempt chains")
    if {str(row.get("system_id") or "") for row in selected} != set(
        smoke["system_ids"]
    ):
        raise ValueError("Smoke gate selected-system set mismatch")
    return gate


def write_gate(path: Path, gate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(gate, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/experiment-end2end-runs-v1"),
    )
    parser.add_argument("--smoke-manifest", type=Path, default=MANIFESTS / "smoke_manifest.json")
    parser.add_argument("--pilot-manifest", type=Path, default=MANIFESTS / "pilot_manifest.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    gate = build_smoke_gate(
        output_root=args.output_root,
        smoke_manifest_path=args.smoke_manifest,
        pilot_manifest_path=args.pilot_manifest,
    )
    output = args.output or (args.output_root / "SMOKE_GATE.json")
    write_gate(output, gate)
    print(json.dumps(gate, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
