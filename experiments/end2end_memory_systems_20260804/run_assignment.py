#!/usr/bin/env python3
"""Run one frozen End2End assignment and retain every terminal outcome.

This file is the Job PID 1.  It launches MLEvolve as a finite subprocess,
waits for Agent exit and candidate-set freeze, then invokes the release-bound
terminal evaluator.  Kubernetes retries are disabled; an explicit ``attempt``
creates a new immutable directory and never replaces the original failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MLEVOLVE_ROOT = REPO / "mlevolve"
if str(MLEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE_ROOT))
SYSTEM_CONFIGS = ROOT / "systems"
HOST_RESULT_SCHEMA = "mlevolve_experiment_c_host_terminal_result_v1"


class RunnerInterrupted(BaseException):
    """A Pod termination signal caught while durable finalization is possible."""

    def __init__(self, signum: int):
        super().__init__(f"runner received signal {signum}")
        self.signum = int(signum)


@contextmanager
def termination_guard() -> Iterator[None]:
    """Convert SIGTERM/SIGINT into a retained infrastructure measurement.

    Kubernetes normally gives this PID 120 seconds of termination grace.  The
    exception unwinds ``subprocess.run`` (which kills its child), after which
    the ordinary runner finalizer records the immutable failed attempt.
    """

    previous: dict[int, Any] = {}

    def interrupt(signum: int, _frame: object) -> None:
        raise RunnerInterrupted(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def run_solver_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
) -> tuple[int, int | None]:
    """Run MLEvolve while forwarding Pod termination for step checkpointing."""

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        shell=False,
    )
    received_signal: int | None = None
    previous: dict[int, Any] = {}

    def forward(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = int(signum)
        if process.poll() is None:
            process.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, forward)
    try:
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            # Give MLEvolve's SIGTERM finalizer time to persist its latest
            # completed-step journal and interrupted RUN_OUTCOME before kill.
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    return int(return_code), received_signal


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


def capture_hardware_receipt(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Record the observed accelerator against the frozen scheduler contract."""

    products: list[str] = []
    query_error = ""
    execution_mode = str(runtime.get("execution_mode") or "gpu").lower()
    if execution_mode != "cpu_only":
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                products = [
                    line.strip()
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]
            else:
                query_error = f"nvidia-smi exited {result.returncode}"
        except (OSError, subprocess.SubprocessError) as error:
            query_error = f"{type(error).__name__}: {error}"
    return {
        "execution_mode": execution_mode,
        "requested_gpu_resource": str(runtime.get("gpu_resource_key") or ""),
        "gpu_product_constraint": runtime.get("gpu_product_constraint"),
        "node_name": os.environ.get("KUBERNETES_NODE_NAME", ""),
        "observed_gpu_products": products,
        "gpu_query_error": query_error,
    }


def frozen_hardware_runtime(components: Mapping[str, Any]) -> dict[str, Any]:
    """Select hardware identity from the global budget manifest, not a phase budget."""

    runtime = dict(components["budget"]["runtime"])
    execution_mode = str(runtime.get("execution_mode") or "gpu").lower()
    if execution_mode not in {"gpu", "cpu_only"}:
        raise ValueError("Frozen runtime has an invalid execution_mode")
    if execution_mode != "cpu_only" and not str(
        runtime.get("gpu_resource_key") or ""
    ):
        raise ValueError("Frozen runtime is missing gpu_resource_key")
    return runtime


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_self_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    expected = str(payload.get(field) or "")
    if len(expected) != 64 or payload_hash(payload, field) != expected:
        raise ValueError(f"{label} self-hash mismatch")


def resolve_below(root: Path, relative: object, label: str) -> Path:
    raw = Path(str(relative or ""))
    path = raw.resolve(strict=True) if raw.is_absolute() else (root / raw).resolve(strict=True)
    try:
        path.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise ValueError(f"{label} escapes frozen root: {path}") from error
    if path.is_symlink():
        raise ValueError(f"{label} may not be a symlink: {path}")
    return path


def load_frozen_inputs(manifest_path: Path) -> dict[str, Any]:
    """Load experiment bookkeeping without turning hashes into launch gates."""

    manifest = read_object(manifest_path)
    manifests = manifest_path.parent
    component_files = {
        "systems": manifests / "systems.json",
        "tasks": manifests / "tasks.json",
        "budget": manifests / "budget.json",
        "memory_bundles": manifests / "memory_bundles.json",
        "evaluators": manifests / "evaluators.json",
        "schemas": manifests / "schemas.json",
        "source_lock": manifests / "source_lock.json",
    }
    components: dict[str, Any] = {}
    for key, path in component_files.items():
        components[key] = read_object(path)
    manifest["_components"] = components
    return manifest


def select_row(
    manifest: Mapping[str, Any], *, index: int, task_id: str | None
) -> dict[str, Any]:
    rows = list(manifest.get("runs") or [])
    if task_id:
        rows = sorted(
            (row for row in rows if row["task_id"] == task_id),
            key=lambda row: int(row["task_launch_position"]),
        )
    else:
        rows = sorted(rows, key=lambda row: int(row["launch_position"]))
    if not 0 <= index < len(rows):
        raise ValueError(f"Index {index} outside manifest shard of {len(rows)}")
    return dict(rows[index])


def _task_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): dict(row) for row in payload.get("tasks") or []}


def _system_map(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["system_id"]): dict(row) for row in payload.get("systems") or []}


def verify_memory_bundle(task_id: str, frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Open the configured Bundle; do not traverse or certify its artifacts."""

    task = dict((frozen.get("task_bundles") or {}).get(task_id) or {})
    if not str(task.get("bundle_root") or ""):
        raise ValueError(f"No Memory Bundle configured for {task_id}")
    from authority.memory_snapshot import MemorySnapshotLoader

    base = MemorySnapshotLoader(task["bundle_root"]).load_base(
        current_path="CURRENT.json",
        verify_artifacts=False,
    )
    return {"task": task, "base": base}


def verify_evaluator_release(
    task_id: str, frozen: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve evaluator paths without formal-release hash certification."""

    releases_root = Path(str(frozen["formal_releases_root"])).resolve(strict=True)
    release_root = (releases_root / task_id / "release").resolve(strict=True)
    runtime_path = release_root / "RUNTIME_SPEC.json"
    runtime = read_object(runtime_path)
    paths: dict[str, Path] = {}
    for key in (runtime.get("runtime_artifact_sha256") or {}):
        paths[key] = resolve_below(release_root, runtime[key], key)
    for key in ("dataset_dir", "data_dir"):
        paths[key] = resolve_below(release_root, runtime[key], key)
    evaluator_path = paths["terminal_evaluator_spec"]
    evaluator = read_object(evaluator_path)
    return {
        "releases_root": releases_root,
        "release_root": release_root,
        "runtime": runtime,
        "paths": paths,
        "evaluator": evaluator,
        "evaluator_path": evaluator_path,
    }


def locate_runtime_directory(parent: Path, markers: Sequence[str]) -> Path:
    if not parent.is_dir():
        return parent
    candidates = {
        path.parent
        for marker in markers
        for path in parent.rglob(marker)
        if path.is_file() and not path.is_symlink()
    }
    if len(candidates) == 1:
        return next(iter(candidates))
    if len(candidates) > 1:
        raise ValueError(f"Runtime output is ambiguous under {parent}")
    children = [path for path in parent.iterdir() if path.is_dir()]
    return children[0] if len(children) == 1 else parent


def terminal_evaluate(
    *,
    evaluator_spec_path: Path,
    release_root: Path,
    agent_log_root: Path,
    agent_workspace_root: Path,
    output_path: Path,
    task: Mapping[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    spec = read_object(evaluator_spec_path)
    kind = str(spec.get("kind") or "")
    if kind == "fixed_holdout_score_run_v3":
        from fixed_holdout.score_run import score_run

        evaluator_manifest = resolve_below(
            release_root, spec["evaluator_manifest"], "evaluator manifest"
        )
        report = score_run(
            evaluator_manifest,
            agent_workspace_root / "submission",
            output_path,
            journal_path=agent_log_root / "journal.json",
            evaluation_request_path=(
                agent_log_root / "fixed_holdout_evaluation_request.json"
            ),
            finalize_writeback=False,
        )
        score = report.get("selected_score")
        candidate_id = report.get("selected_node_id")
    elif kind == "deferred_official_kaggle_v1":
        request_path = agent_log_root / "official_evaluation_request.json"
        request = read_object(request_path)
        if request.get("request_schema") != "mlevolve_official_evaluation_request_v1":
            raise ValueError("Deferred official evaluator requires a native request")
        verify_self_hash(request, "request_hash", "Native official evaluation request")
        if request.get("status") != "awaiting_external_evaluator":
            raise ValueError("Native official request is not awaiting evaluation")
        if request.get("selection_frozen_before_terminal_evaluation") is not True:
            raise ValueError("Native official candidate was not frozen")
        if request.get("scores_were_visible_during_search") is not False:
            raise ValueError("Native official score was visible during search")
        journal_path = agent_log_root / "journal.json"
        if request.get("journal_sha256") != sha256_file(journal_path):
            raise ValueError("Native official request/journal hash mismatch")
        journal = read_object(journal_path)
        candidate_id = str(request.get("selected_node_id") or "")
        selected_nodes = [
            node
            for node in journal.get("nodes") or []
            if isinstance(node, Mapping) and str(node.get("id") or "") == candidate_id
        ]
        if len(selected_nodes) != 1:
            raise ValueError("Native official request selected an unknown candidate")
        metric = selected_nodes[0].get("metric") or {}
        score = metric.get("value")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
        ):
            raise ValueError("Native official selection lacks a finite internal metric")
        report = {
            "schema": "mlevolve_deferred_official_handoff_v1",
            "task_id": request.get("task_id"),
            "competition": request.get("competition"),
            "selected_candidate_id": candidate_id,
            "selected_submission_sha256": request.get(
                "selected_submission_sha256"
            ),
            "candidate_set_hash": request.get("candidate_set_hash"),
            "internal_search_metric": float(score),
            "internal_metric_disposition": "diagnostic_only",
            "official_score": None,
            "status": "awaiting_official_terminal_score",
            "evaluation_request_sha256": sha256_file(request_path),
            "report_hash": "",
        }
        _write_exclusive(output_path, report, "report_hash")
        return {
            "deferred_official": True,
            "internal_search_metric": float(score),
            "selected_candidate_id": candidate_id,
            "terminal_report_sha256": sha256_file(output_path),
            "report": report,
        }
    elif kind == "host_json_command_v1":
        replacements = {
            "{release_root}": str(release_root),
            "{run_root}": str(agent_log_root.parent),
            "{agent_log_root}": str(agent_log_root),
            "{agent_workspace_root}": str(agent_workspace_root),
            "{output}": str(output_path),
        }
        argv = []
        for raw in spec.get("argv") or []:
            value = str(raw)
            for marker, replacement in replacements.items():
                value = value.replace(marker, replacement)
            argv.append(value)
        if not argv:
            raise ValueError("Host terminal evaluator argv is empty")
        subprocess.run(argv, check=True, timeout=timeout_seconds, shell=False)
        report = read_object(output_path)
        if report.get("schema") != HOST_RESULT_SCHEMA:
            raise ValueError("Host terminal result schema mismatch")
        verify_self_hash(report, "report_hash", "Host terminal result")
        score = report.get("terminal_score")
        candidate_id = report.get("selected_candidate_id")
    else:
        raise ValueError(f"Unsupported terminal evaluator kind: {kind}")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        raise ValueError("Terminal evaluator did not return a finite score")
    if not str(candidate_id or ""):
        raise ValueError("Terminal evaluator did not return a candidate ID")
    return {
        "terminal_score": float(score),
        "selected_candidate_id": str(candidate_id),
        "terminal_report_sha256": sha256_file(output_path),
        "report": report,
    }


def _evaluation_overrides(runtime: Mapping[str, Any]) -> list[str]:
    # Import only frozen data/evaluator surfaces.  A release must never
    # reactivate Agent/Host validation gates for this effectiveness experiment.
    # Native official mode is provider-neutral at runtime: credentials and
    # leaderboard access are deliberately absent from these overrides.
    allowed = ("fixed_holdout.", "official_submission.")
    values = [
        str(value)
        for value in runtime.get("additional_overrides") or []
        if str(value).startswith(allowed)
    ]
    return values


# Backward-compatible name for frozen older packet tests and builders.  New
# releases should call `_evaluation_overrides` because official_submission is
# now an equally valid label-free evaluator surface.
def _fixed_holdout_overrides(runtime: Mapping[str, Any]) -> list[str]:
    return _evaluation_overrides(runtime)


def build_solver_command(
    *,
    row: Mapping[str, Any],
    task: Mapping[str, Any],
    system: Mapping[str, Any],
    budget: Mapping[str, Any],
    bundle: Mapping[str, Any],
    evaluator_release: Mapping[str, Any],
    run_root: Path,
    manifest: Mapping[str, Any],
) -> list[str]:
    config_path = (ROOT / str(system["config_path"])).resolve(strict=True)
    identity = manifest["bindings"]
    overrides = [
        f"exp_id={row['task_id']}",
        f"exp_name={row['logical_run_id']}",
        f"dataset_dir={evaluator_release['paths']['dataset_dir']}",
        f"data_dir={evaluator_release['paths']['data_dir']}",
        f"desc_file={evaluator_release['paths']['description']}",
        f"log_dir={run_root / 'logs'}",
        f"workspace_dir={run_root / 'workspace'}",
        f"agent.seed={row['seed']}",
        f"agent.steps={budget['agent_steps']}",
        f"agent.time_limit={budget['agent_time_limit_seconds']}",
        f"agent.initial_drafts={budget['initial_drafts']}",
        f"agent.search.num_drafts={budget['initial_drafts']}",
        f"agent.search.num_gpus={budget['gpu_count']}",
        f"agent.search.parallel_search_num={budget['parallel_search_num']}",
        f"agent.search.max_replacement_drafts={budget['max_replacement_drafts']}",
        f"cpu_number={budget['cpu_count']}",
        f"exec.timeout={budget['execution_timeout_seconds']}",
        f"finalize_reserve_seconds={budget['finalize_reserve_seconds']}",
        f"external_skill_memory.bundle_root={bundle['task']['bundle_root']}",
        "external_skill_memory.current_pointer_path=CURRENT.json",
        "external_skill_memory.experiment_r_memory_pool_sha256="
        f"{bundle['base'].manifest_sha256}",
        f"evaluation_authority.rollout_id={row['logical_run_id']}",
        f"evaluation_authority.expected_bundle_id={bundle['base'].bundle_id}",
        "evaluation_authority.expected_bundle_manifest_sha256="
        f"{bundle['base'].manifest_sha256}",
        f"run_identity.logical_run_id={row['logical_run_id']}",
        f"run_identity.system_id={row['system_id']}",
        f"run_identity.memory_system={row['system_id']}",
        f"run_identity.experiment_manifest_sha256={manifest['manifest_hash']}",
        f"run_identity.system_manifest_sha256={identity['systems_manifest_hash']}",
        f"run_identity.task_manifest_sha256={identity['tasks_manifest_hash']}",
        f"run_identity.budget_manifest_sha256={identity['budget_manifest_hash']}",
        "run_identity.memory_bundle_binding_sha256="
        f"{identity['memory_bundles_manifest_hash']}",
        "run_identity.memory_current_sha256="
        f"{bundle['task']['current_file_sha256']}",
        f"run_identity.evaluator_manifest_sha256={identity['evaluators_manifest_hash']}",
        *_evaluation_overrides(evaluator_release["runtime"]),
        # Last-write-wins CLI overrides make the non-blocking experiment mode
        # immune to any old release profile that carries formal audit settings.
        "fixed_holdout.preflight_validate_train_view=false",
        "agent.check_data_leakage=false",
        "agent.protocol_repair.enabled=false",
        "agent.protocol_preflight.enabled=false",
        "evaluation_authority.emit_snapshot=false",
        "evaluation_authority.runtime_protocol_observer_enabled=false",
        "adoption_verifier.enabled=false",
        "prospective_audit.enabled=false",
    ]
    return [sys.executable, "-u", str(REPO / "mlevolve" / "run.py"), *overrides]


def _first_valid(log_parent: Path, started_ns: int) -> tuple[float | None, str]:
    paths = list(log_parent.rglob("FIRST_PROTOCOL_VALID_CANDIDATE.json"))
    if len(paths) != 1:
        return None, ""
    payload = read_object(paths[0])
    event_ns = int(payload.get("event_time_ns") or payload.get("created_at_ns") or 0)
    if event_ns < started_ns:
        return None, sha256_file(paths[0])
    return (event_ns - started_ns) / 1_000_000_000.0, sha256_file(paths[0])


def _write_exclusive(path: Path, payload: dict[str, Any], hash_field: str) -> None:
    payload[hash_field] = payload_hash(payload, hash_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")


def resolve_resume_attempt(
    output_root: Path,
    logical_run_id: str,
    *,
    source_attempt: int | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Resume a missing run or a retained infrastructure failure.

    A terminal, Agent, or evaluator outcome is already the result for that
    condition and is returned without running it again.  One resumed Indexed
    Job can therefore cover a mix of never-started and interrupted indices.
    """

    condition_root = output_root / logical_run_id
    attempts: list[tuple[int, Path]] = []
    if condition_root.is_dir():
        for path in condition_root.glob("attempt-*"):
            try:
                attempt = int(path.name.removeprefix("attempt-"))
            except ValueError:
                continue
            if path.is_dir():
                attempts.append((attempt, path))
    if not attempts:
        if source_attempt is not None:
            raise ValueError("Explicit resume source does not exist")
        return 0, None
    attempts.sort(key=lambda item: item[0])
    if source_attempt is not None:
        source_paths = [path for attempt, path in attempts if attempt == source_attempt]
        if not source_paths:
            raise ValueError(
                f"Explicit resume source attempt-{source_attempt:03d} does not exist"
            )
        source_measurement_path = source_paths[0] / "MEASUREMENT.json"
        if not source_measurement_path.is_file():
            raise ValueError("Explicit resume source has no retained MEASUREMENT")
        source_measurement = read_object(source_measurement_path)
        if source_measurement.get("failure_class") != "infrastructure":
            raise ValueError(
                "Explicit resume source must be a retained infrastructure failure"
            )
        for later_attempt, later_path in attempts:
            if later_attempt <= source_attempt:
                continue
            later_measurement_path = later_path / "MEASUREMENT.json"
            if not later_measurement_path.is_file():
                raise ValueError(
                    "An intermediate resume attempt has no retained MEASUREMENT"
                )
            later_measurement = read_object(later_measurement_path)
            search_resume = dict(later_measurement.get("search_resume") or {})
            if (
                search_resume.get("enabled") is not True
                or int(search_resume.get("source_attempt", -1)) != source_attempt
                or later_measurement.get("completed") is True
                or later_measurement.get("terminal_score") is not None
            ):
                raise ValueError(
                    "Explicit resume source may bypass only preserved, unscored "
                    "adapter attempts derived from that source"
                )
        return attempts[-1][0] + 1, None
    attempt, path = max(attempts)
    measurement_path = path / "MEASUREMENT.json"
    if not measurement_path.is_file():
        # A node loss or hard power-off can bypass both the MLEvolve and PID-1
        # SIGTERM finalizers. Preserve the orphaned directory and allocate a
        # fresh immutable attempt; ``run`` writes its infrastructure measurement
        # before launching the replacement condition.
        return attempt + 1, None
    measurement = read_object(measurement_path)
    if measurement.get("failure_class") == "infrastructure":
        return attempt + 1, None
    return attempt, measurement


def recover_orphaned_attempt(
    *,
    attempt_root: Path,
    row: Mapping[str, Any],
    task: Mapping[str, Any],
    budget: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Finalize a hard-interrupted immutable attempt before condition retry."""

    measurement_path = attempt_root / "MEASUREMENT.json"
    if measurement_path.is_file():
        return read_object(measurement_path)
    attempt = int(attempt_root.name.removeprefix("attempt-"))
    launch_path = attempt_root / "LAUNCH_RECEIPT.json"
    launch = read_object(launch_path) if launch_path.is_file() else {}
    started_ns = int(launch.get("started_at_ns") or 0)
    artifact_times = [
        path.stat().st_mtime_ns
        for path in attempt_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ]
    last_artifact_ns = max(artifact_times, default=started_ns)
    observed_wall_seconds = (
        max(0, last_artifact_ns - started_ns) / 1_000_000_000.0
        if started_ns
        else 0.0
    )
    prior_measurement: dict[str, Any] = {}
    if attempt > 0:
        prior_path = (
            attempt_root.parent
            / f"attempt-{attempt - 1:03d}"
            / "MEASUREMENT.json"
        )
        if prior_path.is_file() and not prior_path.is_symlink():
            prior_measurement = read_object(prior_path)
    prior_wall_seconds = float(
        prior_measurement.get(
            "cumulative_agent_wall_seconds",
            prior_measurement.get("agent_wall_seconds", 0.0),
        )
        or 0.0
    )
    prior_gpu_hours = float(
        prior_measurement.get(
            "cumulative_allocated_gpu_hours",
            prior_measurement.get("allocated_gpu_hours", 0.0),
        )
        or 0.0
    )
    prior_ttfv = prior_measurement.get(
        "cumulative_time_to_first_valid_seconds",
        prior_measurement.get("time_to_first_valid_seconds"),
    )
    log_parent = attempt_root / "agent" / "logs"
    log_root = locate_runtime_directory(
        log_parent, ("RUN_OUTCOME.json", "journal.json")
    )
    outcome_path = log_root / "RUN_OUTCOME.json"
    native_request_path = log_root / "official_evaluation_request.json"
    fixed_request_path = log_root / "fixed_holdout_evaluation_request.json"
    if native_request_path.is_file() and fixed_request_path.is_file():
        raise ValueError("Run emitted both native-official and fixed-holdout requests")
    request_path = (
        native_request_path if native_request_path.is_file() else fixed_request_path
    )
    request = read_object(request_path) if request_path.is_file() else {}
    ttfv, first_valid_sha = (
        _first_valid(log_parent, started_ns) if started_ns else (None, "")
    )
    journal_path = log_root / "journal.json"
    measurement = {
        "schema": "mlevolve_end2end_condition_measurement_v1",
        "logical_run_id": str(row["logical_run_id"]),
        "attempt": attempt,
        "retry_of": f"attempt-{attempt - 1:03d}" if attempt > 0 else None,
        "manifest_hash": str(manifest["manifest_hash"]),
        "task_id": str(row["task_id"]),
        "system_id": str(row["system_id"]),
        "seed": int(row["seed"]),
        "formal_result_eligible": bool(row["formal_result_eligible"]),
        "exploratory_pilot": True,
        "status": "retained_infrastructure_hard_interruption",
        "failure_class": "infrastructure",
        "completed": False,
        "terminal_metric": str(task["terminal_metric"]),
        "direction": str(task["direction"]),
        "terminal_score": None,
        "selected_candidate_id": None,
        "solver_exit_code": None,
        "solver_error": (
            "prior Pod exited without a final MEASUREMENT; recovered by --resume"
        ),
        "termination_signal": None,
        "terminal_evaluator_error": "",
        "candidate_set_frozen": bool(
            request.get("selection_frozen_before_terminal_evaluation") is True
            and str(request.get("candidate_set_hash") or "")
        ),
        "candidate_set_hash": str(request.get("candidate_set_hash") or ""),
        "time_to_first_valid_seconds": ttfv,
        "cumulative_time_to_first_valid_seconds": (
            prior_ttfv
            if prior_ttfv is not None
            else (
                prior_wall_seconds + ttfv
                if ttfv is not None
                else None
            )
        ),
        "first_valid_event_sha256": first_valid_sha,
        "agent_wall_seconds": observed_wall_seconds,
        "cumulative_agent_wall_seconds": (
            prior_wall_seconds + observed_wall_seconds
        ),
        "allocated_gpu_hours": (
            observed_wall_seconds / 3600.0 * int(budget["gpu_count"])
        ),
        "cumulative_allocated_gpu_hours": (
            prior_gpu_hours
            + observed_wall_seconds / 3600.0 * int(budget["gpu_count"])
        ),
        "hardware": dict(launch.get("hardware") or {}),
        "llm_token_usage": None,
        "llm_cost_usd": None,
        "cost_null_reason": (
            "provider usage unavailable after hard Pod interruption; GPU time "
            "estimated through the last durable artifact"
        ),
        "terminal_report_sha256": "",
        "agent_outcome_sha256": (
            sha256_file(outcome_path) if outcome_path.is_file() else ""
        ),
        "journal_path": str(journal_path) if journal_path.is_file() else "",
        "recovered_last_artifact_ns": last_artifact_ns or None,
        "measurement_hash": "",
    }
    _write_exclusive(measurement_path, measurement, "measurement_hash")
    return measurement


def build_search_resume_binding(
    attempt_root: Path,
    *,
    expected_total_steps: int,
    prior_agent_wall_seconds: float = 0.0,
) -> dict[str, Any]:
    """Bind a fresh attempt to the last durable completed-node checkpoint."""

    log_root = locate_runtime_directory(
        attempt_root / "agent" / "logs",
        ("RUN_OUTCOME.json", "journal.json"),
    )
    workspace_root = locate_runtime_directory(
        attempt_root / "agent" / "workspace",
        ("submission", "working"),
    )
    journal_path = log_root / "journal.json"
    outcome_path = log_root / "RUN_OUTCOME.json"
    if not journal_path.is_file() or journal_path.is_symlink():
        raise ValueError("Infrastructure retry has no durable Journal checkpoint")
    if not outcome_path.is_file() or outcome_path.is_symlink():
        raise ValueError("Infrastructure retry has no durable RUN_OUTCOME checkpoint")
    if not workspace_root.is_dir() or workspace_root.is_symlink():
        raise ValueError("Infrastructure retry has no durable workspace checkpoint")
    outcome = read_object(outcome_path)
    completed_steps = int(outcome.get("completed_steps") or 0)
    total_steps = int(outcome.get("total_steps") or 0)
    if outcome.get("status") != "partial" or outcome.get("interrupted") is not True:
        raise ValueError("Only an interrupted partial run can resume search state")
    if total_steps != int(expected_total_steps):
        raise ValueError(
            f"Resume total-step mismatch: checkpoint={total_steps}, "
            f"budget={expected_total_steps}"
        )
    journal = read_object(journal_path)
    if len(list(journal.get("nodes") or [])) - 1 != completed_steps:
        raise ValueError("Journal node count does not match RUN_OUTCOME completed_steps")
    if not 0 < completed_steps < total_steps:
        raise ValueError("Resume checkpoint has no remaining search work")
    return {
        "schema": "mlevolve_search_resume_binding_v1",
        "source_attempt_root": str(attempt_root.resolve(strict=True)),
        "source_attempt": int(attempt_root.name.removeprefix("attempt-")),
        "journal_path": str(journal_path.resolve(strict=True)),
        "journal_sha256": sha256_file(journal_path),
        "outcome_path": str(outcome_path.resolve(strict=True)),
        "outcome_sha256": sha256_file(outcome_path),
        "workspace_root": str(workspace_root.resolve(strict=True)),
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "remaining_steps": total_steps - completed_steps,
        "prior_agent_wall_seconds": max(
            0.0, float(prior_agent_wall_seconds)
        ),
    }


def remaining_agent_budget_seconds(
    *, total_seconds: int, prior_agent_wall_seconds: float
) -> int:
    """Return remaining wall budget, rejecting any post-budget resume."""

    total = int(total_seconds)
    prior = float(prior_agent_wall_seconds)
    if total <= 0 or not math.isfinite(prior) or prior < 0.0:
        raise ValueError("Invalid agent wall budget")
    remaining = total - prior
    if remaining <= 0.0:
        raise ValueError(
            "Search wall budget is exhausted; fairness policy forbids resume"
        )
    return max(1, int(math.ceil(remaining)))


def condition_disposition(
    *,
    completed_condition: bool,
    terminal_error: str,
    solver_exit_code: int | None,
    solver_error: str,
    outcome: Mapping[str, Any],
) -> tuple[str, str]:
    """Classify search-budget exhaustion as an Agent outcome."""

    if completed_condition:
        return "scored_terminal_result", "none"
    if terminal_error:
        return "retained_terminal_evaluator_failure", "evaluator"
    if solver_error == "agent_time_limit_exceeded":
        return "retained_agent_budget_exhausted", "agent"
    if solver_exit_code in {124, 125, 137, 143} or solver_error:
        return "retained_infrastructure_or_timeout_failure", "infrastructure"
    if outcome:
        return f"retained_agent_{outcome.get('status', 'failure')}", "agent"
    return "retained_agent_failure_without_outcome", "infrastructure"


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve(strict=True)
    manifest = load_frozen_inputs(manifest_path)
    row = select_row(manifest, index=args.index, task_id=args.task)
    output_root = Path(args.output_root).resolve()
    if args.resume:
        args.attempt, retained = resolve_resume_attempt(
            output_root,
            str(row["logical_run_id"]),
            source_attempt=args.resume_source_attempt,
        )
        if retained is not None:
            return retained
    components = manifest["_components"]
    tasks = _task_map(components["tasks"])
    systems = _system_map(components["systems"])
    task = tasks[row["task_id"]]
    system = systems[row["system_id"]]
    budget_key = "smoke" if manifest["kind"] == "smoke" else "pilot"
    if args.budget_profile == "debug_smoke":
        if row.get("formal_result_eligible") is not False:
            raise ValueError("debug_smoke budget is restricted to exploratory rows")
        budget_key = "debug_smoke"
    budget = dict(components["budget"][budget_key])
    if args.dry_run:
        return {
            "schema": "mlevolve_end2end_assignment_dry_run_v1",
            "status": "experiment_intent_loaded_external_assets_not_opened",
            "logical_run_id": row["logical_run_id"],
            "task_id": row["task_id"],
            "system_id": row["system_id"],
            "agent_calls": 0,
            "filesystem_written": False,
        }

    condition_root = output_root / row["logical_run_id"] / f"attempt-{args.attempt:03d}"
    if condition_root.exists():
        raise ValueError(f"Refusing to replace immutable attempt: {condition_root}")
    prior_measurement: dict[str, Any] | None = None
    search_resume: dict[str, Any] | None = None
    retry_source_attempt: int | None = None
    if args.attempt > 0:
        retry_source_attempt = (
            int(args.resume_source_attempt)
            if args.resume_source_attempt is not None
            else args.attempt - 1
        )
        previous = (
            output_root
            / row["logical_run_id"]
            / f"attempt-{retry_source_attempt:03d}"
            / "MEASUREMENT.json"
        )
        if not previous.is_file():
            if not args.resume:
                raise ValueError(
                    "Explicit retry requires a retained prior infrastructure measurement"
                )
            recover_orphaned_attempt(
                attempt_root=previous.parent,
                row=row,
                task=task,
                budget=budget,
                manifest=manifest,
            )
        prior = read_object(previous)
        prior_measurement = prior
        if prior.get("failure_class") != "infrastructure":
            raise ValueError("Explicit retry is allowed only after an infrastructure failure")
        if args.resume:
            search_resume = build_search_resume_binding(
                previous.parent,
                expected_total_steps=int(budget["agent_steps"]),
                prior_agent_wall_seconds=float(
                    prior.get(
                        "cumulative_agent_wall_seconds",
                        prior.get("agent_wall_seconds", 0.0),
                    )
                    or 0.0
                ),
            )
            remaining_agent_budget_seconds(
                total_seconds=int(budget["agent_time_limit_seconds"]),
                prior_agent_wall_seconds=float(
                    search_resume["prior_agent_wall_seconds"]
                ),
            )

    memory = verify_memory_bundle(row["task_id"], components["memory_bundles"])
    evaluator = verify_evaluator_release(row["task_id"], components["evaluators"])
    condition_root.mkdir(parents=True, exist_ok=False)
    run_root = condition_root / "agent"
    (run_root / "logs").mkdir(parents=True)
    (run_root / "workspace").mkdir(parents=True)
    command = build_solver_command(
        row=row,
        task=task,
        system=system,
        budget=budget,
        bundle=memory,
        evaluator_release=evaluator,
        run_root=run_root,
        manifest=manifest,
    )
    hardware = capture_hardware_receipt(frozen_hardware_runtime(components))
    started_ns = time.time_ns()
    launch = {
        "schema": "mlevolve_end2end_launch_receipt_v1",
        "logical_run_id": row["logical_run_id"],
        "attempt": args.attempt,
        "manifest_hash": manifest["manifest_hash"],
        "task_id": row["task_id"],
        "system_id": row["system_id"],
        "seed": row["seed"],
        "formal_result_eligible": bool(row["formal_result_eligible"]),
        "budget": budget,
        "command_sha256": hashlib.sha256(canonical_bytes(command)).hexdigest(),
        "started_at_ns": started_ns,
        "kubernetes": {
            "job_name": os.environ.get("KUBERNETES_JOB_NAME", ""),
            "job_uid": os.environ.get("KUBERNETES_JOB_UID", ""),
            "pod_name": os.environ.get("KUBERNETES_POD_NAME", ""),
            "pod_uid": os.environ.get("KUBERNETES_POD_UID", ""),
            "node_name": os.environ.get("KUBERNETES_NODE_NAME", ""),
            "completion_index": os.environ.get("JOB_COMPLETION_INDEX", ""),
        },
        "hardware": hardware,
        "validation_mode": "experiment_fast_nonblocking_v1",
        "search_resume": dict(search_resume or {}),
        "receipt_hash": "",
    }
    _write_exclusive(condition_root / "LAUNCH_RECEIPT.json", launch, "receipt_hash")

    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"KUBECONFIG", "GITHUB_TOKEN", "GH_TOKEN", "OPENAI_API_KEY"}
        and not key.startswith(("AWS_", "AZURE_", "KUBERNETES_SERVICE_"))
    }
    env.update(
        {
            "MLEVOLVE_CONFIG": str((ROOT / system["config_path"]).resolve()),
            "MLEVOLVE_CONTAINER_IMAGE_REFERENCE": str(
                components["budget"]["runtime"]["container_image"]
            ),
            "MLEVOLVE_SOLVER_BINDING_ID": str(
                components["budget"]["runtime"]["solver_model_id"]
            ),
            "MLEVOLVE_SOLVER_MODEL_REVISION": str(
                components["budget"]["runtime"]["solver_model_revision"]
            ),
            "MLEVOLVE_CODE_REVISION": str(
                components["source_lock"]["git_head"]
            ),
            "MLEVOLVE_CODE_WORKTREE_SHA256": str(
                components["source_lock"]["manifest_hash"]
            ),
            "MLEVOLVE_CONDITION_STARTED_AT_NS": str(started_ns),
            "MLEVOLVE_HOST_ARTIFACT_NAMESPACE": (
                f"{manifest['release_id']}/{row['logical_run_id']}/"
                f"attempt-{args.attempt:03d}"
            ),
            "MLEVOLVE_FIRST_VALID_EVENT_PATH": str(
                run_root / "logs" / "FIRST_PROTOCOL_VALID_CANDIDATE.json"
            ),
            "PYTHONPATH": f"{REPO / 'mlevolve'}:{REPO}",
            "PYTHONUNBUFFERED": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if search_resume is not None:
        env.update(
            {
                "MLEVOLVE_RESUME_SOURCE_ATTEMPT_ROOT": str(
                    search_resume["source_attempt_root"]
                ),
                "MLEVOLVE_RESUME_JOURNAL_PATH": str(
                    search_resume["journal_path"]
                ),
                "MLEVOLVE_RESUME_JOURNAL_SHA256": str(
                    search_resume["journal_sha256"]
                ),
                "MLEVOLVE_RESUME_OUTCOME_PATH": str(
                    search_resume["outcome_path"]
                ),
                "MLEVOLVE_RESUME_OUTCOME_SHA256": str(
                    search_resume["outcome_sha256"]
                ),
                "MLEVOLVE_RESUME_WORKSPACE_ROOT": str(
                    search_resume["workspace_root"]
                ),
                "MLEVOLVE_RESUME_PRIOR_WALL_SECONDS": str(
                    search_resume["prior_agent_wall_seconds"]
                ),
            }
        )
    solver_exit_code: int | None = None
    solver_error = ""
    termination_signal: int | None = None
    try:
        solver_timeout_seconds = int(budget["agent_time_limit_seconds"])
        if search_resume is not None:
            solver_timeout_seconds = remaining_agent_budget_seconds(
                total_seconds=int(budget["agent_time_limit_seconds"]),
                prior_agent_wall_seconds=float(
                    search_resume["prior_agent_wall_seconds"]
                ),
            )
        solver_exit_code, termination_signal = run_solver_process(
            command,
            cwd=REPO / "mlevolve",
            env=env,
            timeout_seconds=solver_timeout_seconds,
        )
        if termination_signal is not None:
            solver_error = f"runner_forwarded_signal_{termination_signal}"
    except subprocess.TimeoutExpired:
        solver_exit_code = 124
        solver_error = "agent_time_limit_exceeded"
    except Exception as error:
        solver_exit_code = 125
        solver_error = f"{type(error).__name__}: {error}"
    agent_finished_ns = time.time_ns()

    log_root = locate_runtime_directory(
        run_root / "logs", ("RUN_OUTCOME.json", "journal.json")
    )
    workspace_root = locate_runtime_directory(
        run_root / "workspace", ("submission", "working")
    )
    outcome_path = log_root / "RUN_OUTCOME.json"
    outcome = read_object(outcome_path) if outcome_path.is_file() else {}
    native_request_path = log_root / "official_evaluation_request.json"
    fixed_request_path = log_root / "fixed_holdout_evaluation_request.json"
    if native_request_path.is_file() and fixed_request_path.is_file():
        raise ValueError("Run emitted both native-official and fixed-holdout requests")
    request_path = (
        native_request_path if native_request_path.is_file() else fixed_request_path
    )
    request = read_object(request_path) if request_path.is_file() else {}
    candidate_set_frozen = bool(
        request.get("selection_frozen_before_terminal_evaluation") is True
        and str(request.get("candidate_set_hash") or "")
    )
    terminal_score = None
    selected_candidate_id = None
    terminal_report_sha256 = ""
    terminal_error = ""
    completed_condition = False
    awaiting_official_score = False
    internal_search_metric = None
    evaluator_kind = str(read_object(evaluator["evaluator_path"]).get("kind") or "")
    if (
        (
            solver_exit_code == 0
            and outcome.get("status") == "complete"
        )
        or (
            evaluator_kind == "deferred_official_kaggle_v1"
            and request_path == native_request_path
            and outcome.get("status") in {"complete", "partial"}
            and outcome.get("certified_solution_available") is True
        )
    ):
        try:
            with termination_guard():
                terminal = terminal_evaluate(
                    evaluator_spec_path=evaluator["evaluator_path"],
                    release_root=evaluator["release_root"],
                    agent_log_root=log_root,
                    agent_workspace_root=workspace_root,
                    output_path=condition_root / "TERMINAL_SCORE_REPORT.json",
                    task=task,
                    timeout_seconds=int(
                        evaluator["runtime"].get(
                            "terminal_evaluator_timeout_seconds"
                        )
                        or 3600
                    ),
                )
            selected_candidate_id = terminal["selected_candidate_id"]
            terminal_report_sha256 = terminal["terminal_report_sha256"]
            if terminal.get("deferred_official") is True:
                awaiting_official_score = True
                internal_search_metric = terminal["internal_search_metric"]
            else:
                terminal_score = terminal["terminal_score"]
                completed_condition = True
        except RunnerInterrupted as error:
            termination_signal = error.signum
            solver_error = f"runner_received_signal_{error.signum}"
        except Exception as error:
            terminal_error = f"{type(error).__name__}: {error}"

    ttfv, first_valid_sha = _first_valid(run_root / "logs", started_ns)
    wall_seconds = (agent_finished_ns - started_ns) / 1_000_000_000.0
    if awaiting_official_score and not terminal_error:
        status, failure_class = "awaiting_official_terminal_score", "none"
    else:
        status, failure_class = condition_disposition(
            completed_condition=completed_condition,
            terminal_error=terminal_error,
            solver_exit_code=solver_exit_code,
            solver_error=solver_error,
            outcome=outcome,
        )
    prior_wall_seconds = float(
        (prior_measurement or {}).get(
            "cumulative_agent_wall_seconds",
            (prior_measurement or {}).get("agent_wall_seconds", 0.0),
        )
        or 0.0
    )
    prior_gpu_hours = float(
        (prior_measurement or {}).get(
            "cumulative_allocated_gpu_hours",
            (prior_measurement or {}).get("allocated_gpu_hours", 0.0),
        )
        or 0.0
    )
    prior_ttfv = (prior_measurement or {}).get(
        "cumulative_time_to_first_valid_seconds",
        (prior_measurement or {}).get("time_to_first_valid_seconds"),
    )
    resume_receipt_path = log_root / "SEARCH_RESUME_RECEIPT.json"
    measurement = {
        "schema": "mlevolve_end2end_condition_measurement_v1",
        "logical_run_id": row["logical_run_id"],
        "attempt": args.attempt,
        "retry_of": (
            f"attempt-{retry_source_attempt:03d}"
            if retry_source_attempt is not None
            else None
        ),
        "manifest_hash": manifest["manifest_hash"],
        "task_id": row["task_id"],
        "system_id": row["system_id"],
        "seed": row["seed"],
        "formal_result_eligible": bool(row["formal_result_eligible"]),
        "exploratory_pilot": True,
        "status": status,
        "failure_class": failure_class,
        "completed": completed_condition,
        "terminal_metric": task["terminal_metric"],
        "direction": task["direction"],
        "terminal_score": terminal_score,
        "internal_search_metric": internal_search_metric,
        "internal_metric_disposition": (
            "diagnostic_only" if awaiting_official_score else None
        ),
        "selected_candidate_id": selected_candidate_id,
        "solver_exit_code": solver_exit_code,
        "solver_error": solver_error,
        "termination_signal": termination_signal,
        "terminal_evaluator_error": terminal_error,
        "candidate_set_frozen": candidate_set_frozen,
        "candidate_set_hash": str(request.get("candidate_set_hash") or ""),
        "official_evaluation_request_sha256": (
            sha256_file(native_request_path)
            if native_request_path.is_file()
            else ""
        ),
        "time_to_first_valid_seconds": ttfv,
        "cumulative_time_to_first_valid_seconds": (
            prior_ttfv
            if prior_ttfv is not None
            else (
                prior_wall_seconds + ttfv
                if ttfv is not None
                else None
            )
        ),
        "first_valid_event_sha256": first_valid_sha,
        "agent_wall_seconds": wall_seconds,
        "cumulative_agent_wall_seconds": prior_wall_seconds + wall_seconds,
        "allocated_gpu_hours": wall_seconds / 3600.0 * int(budget["gpu_count"]),
        "cumulative_allocated_gpu_hours": (
            prior_gpu_hours
            + wall_seconds / 3600.0 * int(budget["gpu_count"])
        ),
        "hardware": hardware,
        "llm_token_usage": None,
        "llm_cost_usd": None,
        "cost_null_reason": "provider usage is null unless emitted by the frozen runtime",
        "terminal_report_sha256": terminal_report_sha256,
        "agent_outcome_sha256": sha256_file(outcome_path) if outcome_path.is_file() else "",
        "journal_path": str(log_root / "journal.json"),
        "search_resume": {
            "enabled": search_resume is not None,
            "source_attempt": (
                search_resume.get("source_attempt")
                if search_resume is not None
                else None
            ),
            "completed_steps_before_resume": (
                search_resume.get("completed_steps")
                if search_resume is not None
                else 0
            ),
            "receipt_sha256": (
                sha256_file(resume_receipt_path)
                if resume_receipt_path.is_file()
                else ""
            ),
        },
        "measurement_hash": "",
    }
    _write_exclusive(condition_root / "MEASUREMENT.json", measurement, "measurement_hash")
    return measurement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="start missing conditions and retry only retained infrastructure failures",
    )
    parser.add_argument(
        "--resume-source-attempt",
        type=int,
        default=None,
        help=(
            "resume an explicitly retained infrastructure attempt while preserving "
            "intermediate unscored adapter failures"
        ),
    )
    parser.add_argument(
        "--output-root",
        default="/workspace/experiment-end2end-memory-agent-v8/runs",
    )
    parser.add_argument("--smoke-gate", type=Path, default=None)
    parser.add_argument(
        "--budget-profile",
        choices=("auto", "debug_smoke"),
        default="auto",
        help="use the 8-step exploratory Debug profile; formal rows are rejected",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.index is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        if raw is None:
            parser.error("--index or JOB_COMPLETION_INDEX is required")
        args.index = int(raw)
    if args.attempt < 0:
        parser.error("--attempt must be non-negative")
    if args.resume and args.attempt != 0:
        parser.error("--resume and an explicit nonzero --attempt are mutually exclusive")
    if args.resume_source_attempt is not None:
        if not args.resume:
            parser.error("--resume-source-attempt requires --resume")
        if args.resume_source_attempt < 0:
            parser.error("--resume-source-attempt must be non-negative")
    result = run(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if (
        args.dry_run
        or result.get("completed")
        or result.get("status") == "awaiting_official_terminal_score"
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
