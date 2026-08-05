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
MANIFESTS = ROOT / "manifests"
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
    """Record the observed GPU against the frozen scheduler contract."""

    products: list[str] = []
    query_error = ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode == 0:
            products = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        else:
            query_error = f"nvidia-smi exited {result.returncode}"
    except (OSError, subprocess.SubprocessError) as error:
        query_error = f"{type(error).__name__}: {error}"
    return {
        "requested_gpu_resource": str(runtime["gpu_resource_key"]),
        "gpu_product_constraint": runtime.get("gpu_product_constraint"),
        "node_name": os.environ.get("KUBERNETES_NODE_NAME", ""),
        "observed_gpu_products": products,
        "gpu_query_error": query_error,
    }


def frozen_hardware_runtime(components: Mapping[str, Any]) -> dict[str, Any]:
    """Select hardware identity from the global budget manifest, not a phase budget."""

    runtime = dict(components["budget"]["runtime"])
    if not str(runtime.get("gpu_resource_key") or ""):
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
    component_files = {
        "systems": MANIFESTS / "systems.json",
        "tasks": MANIFESTS / "tasks.json",
        "budget": MANIFESTS / "budget.json",
        "memory_bundles": MANIFESTS / "memory_bundles.json",
        "evaluators": MANIFESTS / "evaluators.json",
        "schemas": MANIFESTS / "schemas.json",
        "source_lock": MANIFESTS / "source_lock.json",
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


def _fixed_holdout_overrides(runtime: Mapping[str, Any]) -> list[str]:
    allowed = (
        "fixed_holdout.",
        "agent.check_data_leakage=",
        "agent.protocol_repair.enabled=",
        "prospective_audit.",
    )
    values = [
        str(value)
        for value in runtime.get("additional_overrides") or []
        if str(value).startswith(allowed)
    ]
    return values


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
        *_fixed_holdout_overrides(evaluator_release["runtime"]),
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve(strict=True)
    manifest = load_frozen_inputs(manifest_path)
    row = select_row(manifest, index=args.index, task_id=args.task)
    components = manifest["_components"]
    tasks = _task_map(components["tasks"])
    systems = _system_map(components["systems"])
    task = tasks[row["task_id"]]
    system = systems[row["system_id"]]
    budget_key = "smoke" if manifest["kind"] == "smoke" else "pilot"
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

    output_root = Path(args.output_root).resolve()
    condition_root = output_root / row["logical_run_id"] / f"attempt-{args.attempt:03d}"
    if condition_root.exists():
        raise ValueError(f"Refusing to replace immutable attempt: {condition_root}")
    if args.attempt > 0:
        previous = output_root / row["logical_run_id"] / f"attempt-{args.attempt - 1:03d}" / "MEASUREMENT.json"
        prior = read_object(previous)
        if prior.get("failure_class") != "infrastructure":
            raise ValueError("Explicit retry is allowed only after an infrastructure failure")

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
    solver_exit_code: int | None = None
    solver_error = ""
    termination_signal: int | None = None
    try:
        solver_exit_code, termination_signal = run_solver_process(
            command,
            cwd=REPO / "mlevolve",
            env=env,
            timeout_seconds=int(budget["agent_time_limit_seconds"]),
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
    request_path = log_root / "fixed_holdout_evaluation_request.json"
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
    if (
        solver_exit_code == 0
        and outcome.get("status") == "complete"
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
            terminal_score = terminal["terminal_score"]
            selected_candidate_id = terminal["selected_candidate_id"]
            terminal_report_sha256 = terminal["terminal_report_sha256"]
            completed_condition = True
        except RunnerInterrupted as error:
            termination_signal = error.signum
            solver_error = f"runner_received_signal_{error.signum}"
        except Exception as error:
            terminal_error = f"{type(error).__name__}: {error}"

    ttfv, first_valid_sha = _first_valid(run_root / "logs", started_ns)
    wall_seconds = (agent_finished_ns - started_ns) / 1_000_000_000.0
    if completed_condition:
        status, failure_class = "scored_terminal_result", "none"
    elif terminal_error:
        status, failure_class = "retained_terminal_evaluator_failure", "evaluator"
    elif solver_exit_code in {124, 125, 137, 143} or solver_error:
        status, failure_class = "retained_infrastructure_or_timeout_failure", "infrastructure"
    elif outcome:
        status, failure_class = f"retained_agent_{outcome.get('status', 'failure')}", "agent"
    else:
        status, failure_class = "retained_agent_failure_without_outcome", "infrastructure"
    measurement = {
        "schema": "mlevolve_end2end_condition_measurement_v1",
        "logical_run_id": row["logical_run_id"],
        "attempt": args.attempt,
        "retry_of": (
            f"attempt-{args.attempt - 1:03d}" if args.attempt > 0 else None
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
        "selected_candidate_id": selected_candidate_id,
        "solver_exit_code": solver_exit_code,
        "solver_error": solver_error,
        "termination_signal": termination_signal,
        "terminal_evaluator_error": terminal_error,
        "candidate_set_frozen": candidate_set_frozen,
        "candidate_set_hash": str(request.get("candidate_set_hash") or ""),
        "time_to_first_valid_seconds": ttfv,
        "first_valid_event_sha256": first_valid_sha,
        "agent_wall_seconds": wall_seconds,
        "allocated_gpu_hours": wall_seconds / 3600.0 * int(budget["gpu_count"]),
        "hardware": hardware,
        "llm_token_usage": None,
        "llm_cost_usd": None,
        "cost_null_reason": "provider usage is null unless emitted by the frozen runtime",
        "terminal_report_sha256": terminal_report_sha256,
        "agent_outcome_sha256": sha256_file(outcome_path) if outcome_path.is_file() else "",
        "journal_path": str(log_root / "journal.json"),
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
        "--output-root",
        default="/workspace/experiment-end2end-memory-agent-v8/runs",
    )
    parser.add_argument("--smoke-gate", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.index is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        if raw is None:
            parser.error("--index or JOB_COMPLETION_INDEX is required")
        args.index = int(raw)
    if args.attempt < 0:
        parser.error("--attempt must be non-negative")
    result = run(args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if args.dry_run or result.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
