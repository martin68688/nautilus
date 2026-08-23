"""
Python interpreter for executing code snippets via subprocess.
- Executes code in a separate Python process (avoids CUDA/fork issues).
- Captures stdout/stderr, exceptions and stack traces, execution time limit.
- Supports multiple parallel slots (max_parallel_run) with CPU pinning.
"""

import ast
import logging
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import sys
import threading
import time
import traceback
import subprocess
from dataclasses import dataclass
from multiprocessing import Lock
from pathlib import Path

import humanize
from dataclasses_json import DataClassJsonMixin

from authority.runtime_protocol import (
    OBSERVATION_SCHEMA,
    build_runtime_protocol_plan,
    instrument_runtime_protocol_source,
    parse_runtime_protocol_trace,
    strip_runtime_protocol_markers,
)
from engine.candidate_execution_contract import (
    audit_candidate_code,
    build_candidate_execution_block_receipt,
    candidate_execution_contract_from_cfg,
)
from protocol_runtime.preflight import validate_preflight_admission
from authority.protocol_execution_contract import read_contract_artifact
from authority.protocol_registry import ProtocolRegistry
from protocol_runtime.collector import HostCollectorIdentity
from protocol_runtime.full_runtime import FullRuntimeEvidenceController
from protocol_runtime.adoption_trace import (
    bootstrap_for_prefix as adoption_trace_bootstrap,
    seal_runtime_trace as seal_adoption_runtime_trace,
)
from authority.adoption_verification import verify_plan as verify_adoption_plan
from protocol_runtime.preflight import (
    ProtocolPreflightRunner,
    admission_report_path,
)

logger = logging.getLogger("MLEvolve")


_CANDIDATE_HF_SEED_ROOT_ENV = "MLEVOLVE_CANDIDATE_HF_HUB_SEED"
_CANDIDATE_HF_SEED_REPOSITORIES_ENV = (
    "MLEVOLVE_CANDIDATE_HF_REPOSITORIES"
)
_HF_REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)


def _insert_host_preamble_after_future_imports(
    candidate_code: str,
    host_preamble: str,
) -> str:
    """Compose a runfile without invalidating Python ``__future__`` imports.

    Python permits future imports only after comments, a module docstring, and
    other future imports.  Executor-owned CPU/GPU/runtime setup therefore
    belongs after that header rather than at byte zero of Candidate source.
    The Candidate is parsed before this helper is called, so ``lineno`` and
    ``end_lineno`` are safe deterministic insertion coordinates.
    """

    tree = ast.parse(candidate_code)
    body = list(tree.body)
    insertion_line = 0
    body_index = 0

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insertion_line = int(body[0].end_lineno or body[0].lineno)
        body_index = 1

    while body_index < len(body):
        statement = body[body_index]
        if not (
            isinstance(statement, ast.ImportFrom)
            and statement.module == "__future__"
        ):
            break
        insertion_line = int(statement.end_lineno or statement.lineno)
        body_index += 1

    lines = candidate_code.splitlines(keepends=True)
    prefix = "".join(lines[:insertion_line])
    suffix = "".join(lines[insertion_line:])
    if prefix and not prefix.endswith(("\n", "\r")):
        prefix += "\n"
    if host_preamble and not host_preamble.endswith(("\n", "\r")):
        host_preamble += "\n"
    return prefix + host_preamble + suffix


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hf_repository_cache_name(repository: str) -> str:
    """Return the on-disk Hub name for one explicitly declared repository."""

    repository = str(repository or "").strip()
    if not _HF_REPOSITORY_RE.fullmatch(repository):
        raise ValueError(
            "candidate Hugging Face seed repositories must use owner/name: "
            f"{repository!r}"
        )
    return "models--" + repository.replace("/", "--")


def _validate_hf_seed_repository(repository_dir: Path) -> Path:
    """Reject cache trees whose symlinks escape the declared repository."""

    if repository_dir.is_symlink() or not repository_dir.is_dir():
        raise ValueError(
            "candidate Hugging Face seed repository must be a real directory: "
            f"{repository_dir}"
        )
    resolved_root = repository_dir.resolve(strict=True)
    for path in repository_dir.rglob("*"):
        if not path.is_symlink():
            continue
        try:
            resolved_target = path.resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError(
                f"candidate Hugging Face seed contains a broken symlink: {path}"
            ) from error
        if not resolved_target.is_relative_to(resolved_root):
            raise ValueError(
                "candidate Hugging Face seed symlink escapes its repository: "
                f"{path} -> {resolved_target}"
            )
    return resolved_root


def _seed_candidate_hf_hub_cache(
    *,
    runtime_cache_root: Path,
    destination_hub_cache: Path,
    inherited_env: dict[str, str],
) -> dict | None:
    """Copy allowlisted public Hub repositories into an isolated cache.

    Candidate execution deliberately receives a fresh HOME and HF_HOME.  A
    launch-time pretrained-model preflight therefore cannot make the model
    available to the Candidate unless the Host explicitly bridges the verified
    public cache into that fresh tree.  Only named repositories are copied;
    arbitrary contents of the Host cache never become visible to Candidate
    code.
    """

    raw_seed_root = str(
        inherited_env.get(_CANDIDATE_HF_SEED_ROOT_ENV, "") or ""
    ).strip()
    raw_repositories = str(
        inherited_env.get(_CANDIDATE_HF_SEED_REPOSITORIES_ENV, "") or ""
    ).strip()
    if not raw_seed_root and not raw_repositories:
        return None
    if not raw_seed_root or not raw_repositories:
        raise ValueError(
            "candidate Hugging Face cache seeding requires both "
            f"{_CANDIDATE_HF_SEED_ROOT_ENV} and "
            f"{_CANDIDATE_HF_SEED_REPOSITORIES_ENV}"
        )

    seed_root = Path(raw_seed_root)
    if not seed_root.is_absolute() or seed_root.is_symlink():
        raise ValueError(
            "candidate Hugging Face seed root must be an absolute real directory"
        )
    seed_root = seed_root.resolve(strict=True)
    if not seed_root.is_dir():
        raise ValueError(
            f"candidate Hugging Face seed root is not a directory: {seed_root}"
        )

    repositories = []
    for item in raw_repositories.split(","):
        repository = item.strip()
        if repository and repository not in repositories:
            repositories.append(repository)
    if not repositories:
        raise ValueError("candidate Hugging Face seed repository list is empty")

    destination_hub_cache.mkdir(parents=True, exist_ok=True)
    destination_hub_cache.chmod(0o777)
    seeded = []
    for repository in repositories:
        cache_name = _hf_repository_cache_name(repository)
        source_repository = seed_root / cache_name
        _validate_hf_seed_repository(source_repository)
        destination_repository = destination_hub_cache / cache_name
        shutil.copytree(
            source_repository,
            destination_repository,
            symlinks=True,
            dirs_exist_ok=True,
        )
        _validate_hf_seed_repository(destination_repository)
        for path in (destination_repository, *destination_repository.rglob("*")):
            if path.is_symlink():
                continue
            mode = path.stat().st_mode
            path.chmod(mode | (0o555 if path.is_dir() else 0o444))
        seeded.append(
            {
                "repository": repository,
                "cache_name": cache_name,
                "destination": str(destination_repository),
            }
        )

    receipt = {
        "schema": "mlevolve_candidate_hf_cache_seed_v1",
        "source_hub_cache": str(seed_root),
        "destination_hub_cache": str(destination_hub_cache),
        "repositories": seeded,
    }
    receipt_path = runtime_cache_root / "host_hf_cache_seed_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_path.chmod(0o444)
    logger.info(
        "Seeded Candidate Hugging Face cache with repositories=%s",
        repositories,
    )
    return receipt


def _execution_environment(runtime_cache_root: Path | None = None) -> dict[str, str]:
    """Expose runtime helpers and a writable cache home to isolated runfiles.

    Enforcing Host runs drop the candidate subprocess to an unprivileged UID.
    Inherited root-owned HOME/cache paths would otherwise break legitimate
    libraries such as torch.hub, HuggingFace and matplotlib after Preflight.
    Each candidate receives its own writable cache tree inside its working
    directory; this changes no data-view or leakage permissions.
    """

    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    paths = [path for path in existing.split(os.pathsep) if path]
    if package_root not in paths:
        paths.insert(0, package_root)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONUNBUFFERED"] = "1"
    if runtime_cache_root is not None:
        cache_root = Path(runtime_cache_root).resolve()
        locations = {
            "HOME": cache_root / "home",
            "XDG_CACHE_HOME": cache_root / "xdg-cache",
            "XDG_CONFIG_HOME": cache_root / "xdg-config",
            "TORCH_HOME": cache_root / "torch",
            "HF_HOME": cache_root / "huggingface",
            "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface" / "hub",
            "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
            "MPLCONFIGDIR": cache_root / "matplotlib",
            "NUMBA_CACHE_DIR": cache_root / "numba",
            "KERAS_HOME": cache_root / "keras",
        }
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_root.chmod(0o777)
        for name, path in locations.items():
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o777)
            env[name] = str(path)
        # Current huggingface_hub releases prefer HF_HUB_CACHE while older
        # releases used HUGGINGFACE_HUB_CACHE.  Bind both names to the exact
        # same isolated location so the preflight and Candidate resolve one
        # deterministic cache.
        env["HF_HUB_CACHE"] = env["HUGGINGFACE_HUB_CACHE"]
        _seed_candidate_hf_hub_cache(
            runtime_cache_root=cache_root,
            destination_hub_cache=Path(env["HF_HUB_CACHE"]),
            inherited_env=env,
        )
        # These are Host-only bridge controls, not Candidate inputs.
        env.pop(_CANDIDATE_HF_SEED_ROOT_ENV, None)
        env.pop(_CANDIDATE_HF_SEED_REPOSITORIES_ENV, None)
        env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _runtime_protocol_observer_enabled(cfg) -> bool:
    authority = getattr(cfg, "evaluation_authority", None) if cfg else None
    mode = str(getattr(authority, "mode", "off") or "off").lower()
    enabled = bool(
        getattr(authority, "runtime_protocol_observer_enabled", True)
    )
    runtime_mode = str(
        getattr(authority, "protocol_runtime_mode", "legacy_ast") or "legacy_ast"
    ).lower()
    return (
        enabled
        and mode in {"shadow", "enforce"}
        and runtime_mode == "legacy_ast"
    )


def _protocol_runtime_mode(cfg) -> str:
    authority = getattr(cfg, "evaluation_authority", None) if cfg else None
    return str(
        getattr(authority, "protocol_runtime_mode", "legacy_ast") or "legacy_ast"
    ).lower()


def _candidate_uid_isolation_enabled(cfg) -> bool:
    """Keep OS-level Candidate isolation independent of protocol machinery."""

    runtime_mode = _protocol_runtime_mode(cfg)
    preflight = (
        getattr(getattr(cfg, "agent", None), "protocol_preflight", None)
        if cfg is not None
        else None
    )
    return bool(
        runtime_mode == "host_sdk_enforce"
        or (
            preflight is not None
            and getattr(preflight, "enabled", False)
            and getattr(preflight, "candidate_process_isolation", False)
        )
    )


@dataclass
class ExecutionResult(DataClassJsonMixin):
    """
    Result of executing a code snippet in the interpreter.
    Contains the output, execution time, and exception information.
    """

    term_out: list[str]
    exec_time: float
    exc_type: str | None
    exc_info: dict | None = None
    exc_stack: list[tuple] | None = None
    protocol_observation: dict | None = None
    adoption_trace: dict | None = None



class Interpreter:
    def __init__(
        self,
        working_dir: Path | str,
        timeout: int = 3600,
        agent_file_name: str = "runfile.py",
        max_parallel_run: int = 3,
        cfg=None,
        **kwargs,
    ):
        """
        Executes Python code in subprocess(es). No fork/multiprocessing.Process to avoid CUDA issues.

        Args:
            working_dir: working directory of the agent
            timeout: timeout per code execution (seconds)
            agent_file_name: base name for runfile (actual names are runfile_0.py, ...)
            max_parallel_run: max concurrent execution slots
            cfg: config (start_cpu_id, cpu_number, agent.search.parallel_search_num)
        """
        self.working_dir = Path(working_dir).resolve()
        assert self.working_dir.exists(), f"Working directory {self.working_dir} does not exist"
        self.timeout = timeout
        self.cfg = cfg
        self.candidate_execution_contract = (
            candidate_execution_contract_from_cfg(cfg) if cfg is not None else None
        )
        self.protocol_preflight_config = (
            getattr(getattr(cfg, "agent", None), "protocol_preflight", None)
            if cfg is not None
            else None
        )
        self.protocol_runtime_mode = _protocol_runtime_mode(cfg)
        self._preflight_lock = threading.Lock()
        self._protocol_contract = None
        self._collector_identity = None
        self._preflight_runner = None
        auto_preflight = bool(
            getattr(self.protocol_preflight_config, "enabled", False)
            and getattr(self.protocol_preflight_config, "contract_path", "")
            and getattr(
                self.protocol_preflight_config, "collector_private_key_path", ""
            )
        )
        if auto_preflight:
            self._protocol_contract = read_contract_artifact(
                self.protocol_preflight_config.contract_path
            )
            collector_key_path = Path(
                self.protocol_preflight_config.collector_private_key_path
            ).expanduser()
            self._collector_identity = HostCollectorIdentity.from_private_key_file(
                collector_key_path
            )
            expected_public = str(
                self._protocol_contract.collector_spec.get(
                    "public_key_ed25519", ""
                )
            )
            if self._collector_identity.public_key_ed25519 != expected_public:
                raise ValueError(
                    "Host Collector private key does not match the frozen Contract"
                )
            if bool(
                getattr(
                    self.protocol_preflight_config,
                    "consume_collector_private_key",
                    True,
                )
            ):
                collector_key_path.unlink()
            registry_root = Path(cfg.evaluation_authority.protocol_registry)
            if not registry_root.is_absolute():
                registry_root = Path(__file__).resolve().parents[1] / registry_root
            self._preflight_runner = ProtocolPreflightRunner(
                ProtocolRegistry(registry_root)
            )
            if _candidate_uid_isolation_enabled(cfg) and os.geteuid() != 0:
                raise ValueError(
                    "Candidate UID isolation requires a root Host launcher"
                )
        if self.candidate_execution_contract:
            self.timeout = min(
                self.timeout,
                int(self.candidate_execution_contract["max_execution_seconds"]),
            )
        self.max_parallel_run = (
            cfg.agent.search.parallel_search_num if (cfg and getattr(cfg.agent.search, "parallel_search_num", None)) else max_parallel_run
        )
        self.agent_file_name = [f"runfile_{i}.py" for i in range(self.max_parallel_run)]
        self.current_parallel_run = 0
        self.status_map = [0] * self.max_parallel_run
        self.start_cpu_id = int(cfg.start_cpu_id) if cfg else 0
        self.cpu_number = int(cfg.cpu_number) if cfg else 1
        if self.cpu_number < self.max_parallel_run:
            raise ValueError(
                "The maximum level of parallelism exceeds the number of allocated CPU cores; "
                "ensure that each process has at least one CPU core."
            )
        self.lock = Lock()
        self._procs_lock = threading.Lock()
        self._active_procs: dict[int, subprocess.Popen] = {}
        self._active_candidate_ids: set[str] = set()
        self.deadline_monotonic: float | None = None
        self.slot_deadlines_monotonic: dict[int, float] = {}
        self.finalize_reserve_seconds = 0

    def set_run_deadline(
        self,
        deadline_monotonic: float,
        *,
        finalize_reserve_seconds: int = 0,
    ) -> None:
        if deadline_monotonic <= time.monotonic():
            raise ValueError("Run deadline must be in the future")
        self.deadline_monotonic = float(deadline_monotonic)
        self.finalize_reserve_seconds = max(0, int(finalize_reserve_seconds))

    def set_slot_deadline(
        self,
        slot_id: int,
        deadline_monotonic: float,
    ) -> None:
        slot_id = int(slot_id)
        if not 0 <= slot_id < self.max_parallel_run:
            raise ValueError(
                f"slot deadline is outside allocated slots: {slot_id}"
            )
        if deadline_monotonic <= time.monotonic():
            raise ValueError("Slot deadline must be in the future")
        self.slot_deadlines_monotonic[slot_id] = float(deadline_monotonic)

    def remaining_work_seconds(self, slot_id: int | None = None) -> float | None:
        deadlines = []
        if self.deadline_monotonic is not None:
            deadlines.append(
                self.deadline_monotonic - self.finalize_reserve_seconds
            )
        if slot_id is not None:
            slot_deadline = self.slot_deadlines_monotonic.get(int(slot_id))
            if slot_deadline is not None:
                deadlines.append(slot_deadline)
        if not deadlines:
            return None
        return max(
            0.0,
            min(deadlines)
            - time.monotonic(),
        )

    def effective_timeout_seconds(self, slot_id: int | None = None) -> int:
        remaining = self.remaining_work_seconds(slot_id)
        if remaining is None:
            return int(self.timeout)
        return max(0, min(int(self.timeout), int(remaining)))

    def active_candidate_ids(self) -> list[str]:
        with self._procs_lock:
            return sorted(self._active_candidate_ids)

    def terminate_all_subprocesses(self) -> None:
        """Terminate all active subprocesses (for graceful Ctrl+C exit)."""
        with self._procs_lock:
            procs = list(self._active_procs.items())
            self._active_procs.clear()
        for slot_id, proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
            except Exception as e:
                logger.warning(f"Error terminating subprocess slot {slot_id}: {e}")

    def check_current_status(self):
        """Check current parallel run number."""
        return self.current_parallel_run < self.max_parallel_run

    def isolate_submission_path(self, code: str, _id) -> str:
        """Per-process submission filename to avoid write conflicts."""
        target = f"submission_{_id}.csv"

        code = code.replace("submission/submission.csv", f"submission/{target}")
        code = code.replace("/submission.csv", f"/{target}")

        for quote in ("'", '"'):
            code = code.replace(
                f"to_csv({quote}submission.csv",
                f"to_csv({quote}submission/{target}",
            )

        for quote in ("'", '"'):
            code = code.replace(f"{quote}submission.csv{quote}", f"{quote}{target}{quote}")

        return code
    
    def isolate_model_path(self, code, _id):
        """Replace generic model filenames in code to avoid multi-process conflicts."""
        if '.pth' not in code and '.bin' not in code and '.pt' not in code:
            return code

        modified_code = code

        generic_model_names = [
            "best_model.pth",
            "best_model.bin",
            "best_model.pt",
            "model_best.pth",
            "model_best.bin",
            "model_best.pt",
            "model.pth",
            "model.pt",
            "model.bin",
            "checkpoint.pth",
            "checkpoint.pt",
            "checkpoint.bin",
        ]

        generic_model_names.sort(key=len, reverse=True)

        for filename in generic_model_names:
            if filename not in modified_code:
                continue

            name, ext = filename.rsplit('.', 1)
            new_filename = f"{name}_{_id}.{ext}"

            modified_code = modified_code.replace(f"/{filename}", f"/{new_filename}")
            modified_code = modified_code.replace(f'"{filename}"', f'"{new_filename}"')
            modified_code = modified_code.replace(f"'{filename}'", f"'{new_filename}'")

        return modified_code
    
    def cleanup_session(self, process_id: int = -1) -> None:
        """Clean up resources for the given process slot."""
        pass

    def run(
        self,
        code: str,
        id,
        reset_session=True,
        working_dir: str | None = None,
        adoption_verification_plan: dict | None = None,
        preferred_process_id: int | None = None,
    ):
        """
        Execute the provided Python command in a subprocess and return its output.

        Parameters:
            code: Python code to execute.
            reset_session: Reserved for future use.
            working_dir: Optional per-run working directory.

        Returns:
            ExecutionResult: output, exec_time, exc_type, exc_info, exc_stack.
        """
        return self._run_subprocess(
            code=code,
            id=id,
            working_dir=working_dir,
            adoption_verification_plan=adoption_verification_plan,
            preferred_process_id=preferred_process_id,
        )

    def _run_subprocess(
        self,
        code: str,
        id,
        working_dir: str | None = None,
        adoption_verification_plan: dict | None = None,
        preferred_process_id: int | None = None,
    ):
        """
        Execute code via subprocess (avoids CUDA fork issues).
        Aligned with multiprocessing mode for consistency.
        """
        logger.info("REPL is executing code via subprocess")
        logger.info(f"Current running process: {self.current_parallel_run}")
        effective_timeout = self.effective_timeout_seconds(
            preferred_process_id
        )
        if effective_timeout <= 0:
            return ExecutionResult(
                term_out=[
                    "RunDeadlineExceeded: candidate was not started because "
                    "the finalization reserve has begun.\n"
                ],
                exec_time=0.0,
                exc_type="RunDeadlineExceeded",
                exc_info={
                    "candidate_id": str(id),
                    "finalize_reserve_seconds": self.finalize_reserve_seconds,
                },
                exc_stack=[],
            )
        process_id = None

        with self.lock:
            self.current_parallel_run += 1
            if preferred_process_id is not None:
                preferred_process_id = int(preferred_process_id)
                if not 0 <= preferred_process_id < self.max_parallel_run:
                    self.current_parallel_run -= 1
                    raise ValueError(
                        "preferred_process_id must address an allocated executor slot: "
                        f"slot={preferred_process_id}, slots={self.max_parallel_run}"
                    )
                if self.status_map[preferred_process_id] != 0:
                    self.current_parallel_run -= 1
                    raise RuntimeError(
                        "Dedicated executor slot is already occupied: "
                        f"slot={preferred_process_id}"
                    )
                self.status_map[preferred_process_id] = 1
                process_id = preferred_process_id
                logger.info(
                    "Assigned dedicated process_id: %s", process_id
                )
            else:
                for idx in range(self.max_parallel_run):
                    if self.status_map[idx] == 0:
                        self.status_map[idx] = 1
                        process_id = idx
                        logger.info(f"Assigned process_id: {process_id}")
                        break
                    elif idx == self.max_parallel_run - 1:
                        self.current_parallel_run -= 1
                        logger.info("reach max process parallel number")
                        raise ValueError("reach max process parallel number")

        start_time = time.time()
        candidate_deadline_monotonic = time.monotonic() + effective_timeout
        runfile_path = None
        proc = None
        full_runtime_controller = None
        adoption_trace_path = None
        adoption_trace_nonce = ""
        adoption_trace_evidence = None
        
        try:
            with self._procs_lock:
                self._active_candidate_ids.add(str(id))
            cpu_number_per_session = max(1, int(self.cpu_number / self.max_parallel_run))
            avail_cpus = (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else list(range(os.cpu_count() or 1))
            )
            start = process_id * cpu_number_per_session
            cpu_set = set(avail_cpus[start:start + cpu_number_per_session])
            if not cpu_set:
                cpu_set = set(avail_cpus)
            logger.info(f"has set process_id:{process_id} to use cpu: {cpu_set}")

            # Accelerator allocation. ``num_gpus=0`` is an explicit CPU-only
            # contract: do not perform the multi-GPU modulo and hide any GPU
            # visibility inherited from the parent process. This also lets
            # CPU-only Kubernetes Pods run the same immutable experiment
            # runner without pretending that a GPU was allocated.
            num_gpus = int(
                getattr(self.cfg.agent.search, "num_gpus", 1)
                if self.cfg
                else 1
            )
            if num_gpus < 0:
                raise ValueError("agent.search.num_gpus must be non-negative")
            gpu_id = process_id % num_gpus if num_gpus else None
            if gpu_id is None:
                logger.info(
                    "has set process_id:%s to CPU-only execution", process_id
                )
            else:
                logger.info(
                    "has set process_id:%s to use GPU: %s", process_id, gpu_id
                )

            # decide runfile location and cwd
            run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
            runfile_path = run_wd / self.agent_file_name[process_id]
            run_wd.mkdir(parents=True, exist_ok=True)
            candidate_runtime_cache_root = (
                run_wd
                / "working"
                / "candidate_runtime_cache"
                / hashlib.sha256(str(id).encode("utf-8")).hexdigest()[:24]
            )
            if _candidate_uid_isolation_enabled(self.cfg) and os.geteuid() == 0:
                for writable in (
                    run_wd,
                    run_wd / "submission",
                    run_wd / "working",
                ):
                    writable.mkdir(parents=True, exist_ok=True)
                    writable.chmod(0o777)

            pre_code = (
                "import os\n"
                "if hasattr(os, 'sched_setaffinity'):\n"
                "    os.sched_setaffinity(0, {cpu_set})\n"
            ).format(cpu_set=cpu_set)
            if gpu_id is None:
                pre_code += "os.environ['CUDA_VISIBLE_DEVICES'] = ''\n"
            else:
                pre_code += (
                    "os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'\n"
                ).format(gpu_id=gpu_id)
            source_code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
            preflight_report = None
            if bool(getattr(self.protocol_preflight_config, "enabled", False)):
                agent_controls_preflight = bool(
                    self.protocol_runtime_mode == "host_sdk_shadow"
                    and getattr(
                        self.protocol_preflight_config,
                        "agent_controls_protocol_preflight",
                        False,
                    )
                )
                if agent_controls_preflight:
                    # Dynamic Hybrid delegates semantic protocol review and
                    # narrow repair to the Agent.  Do not execute the Candidate
                    # a second time inside the legacy Host Preflight: the full
                    # subprocess below is the only training launch.  Host SDK
                    # lifecycle events remain attached as observations.
                    preflight_report = {
                        "schema": (
                            "mlevolve_agent_controlled_protocol_preflight_"
                            "observation_v1"
                        ),
                        "status": "agent_controlled",
                        "code_sha256": source_code_sha256,
                        "enforcement_mode": "shadow",
                        "admission_disposition": "agent_review_then_execute",
                        "host_dry_run_executed": False,
                        "candidate_subprocess_started": False,
                    }
                else:
                    try:
                        report_root = Path(
                            self.protocol_preflight_config.report_root
                        ).resolve()
                        if self._preflight_runner is None:
                            preflight_report = validate_preflight_admission(
                                code,
                                report_root=report_root,
                                expected_contract_hash=str(
                                    self.protocol_preflight_config.expected_contract_hash
                                ),
                            )
                        else:
                            report_path = admission_report_path(
                                report_root, source_code_sha256
                            )
                            with self._preflight_lock:
                                if (
                                    report_path.is_file()
                                    and not report_path.is_symlink()
                                ):
                                    preflight_report = json.loads(
                                        report_path.read_text(encoding="utf-8")
                                    )
                                else:
                                    candidate_root = (
                                        report_root
                                        / "candidates"
                                        / source_code_sha256
                                    )
                                    preflight_report = (
                                        self._preflight_runner.run_source(
                                            source=code,
                                            contract=self._protocol_contract,
                                            identity=self._collector_identity,
                                            data_view_manifest_path=(
                                                self.protocol_preflight_config.data_view_manifest_path
                                            ),
                                            output_root=candidate_root,
                                            image_digest=self.protocol_preflight_config.image_digest,
                                            sdk_hash=self.protocol_preflight_config.sdk_hash,
                                            timeout_seconds=min(
                                                60, effective_timeout
                                            ),
                                            candidate_uid=(
                                                int(
                                                    self.protocol_preflight_config.candidate_uid
                                                )
                                                if _candidate_uid_isolation_enabled(
                                                    self.cfg
                                                )
                                                and os.geteuid() == 0
                                                else None
                                            ),
                                        )
                                    )
                        if self.protocol_runtime_mode != "host_sdk_shadow":
                            preflight_report = validate_preflight_admission(
                                code,
                                report_root=report_root,
                                expected_contract_hash=str(
                                    self.protocol_preflight_config.expected_contract_hash
                                ),
                            )
                    except Exception as error:
                        if self.protocol_runtime_mode != "host_sdk_shadow":
                            rejection_report = dict(
                                getattr(error, "report", None)
                                or preflight_report
                                or {}
                            )
                            rejection_report.setdefault("status", "rejected")
                            rejection_report.setdefault(
                                "code_sha256", source_code_sha256
                            )
                            rejection_report["admission_disposition"] = "rejected"
                            rejection_report["admission_error"] = str(error)
                            rejection_details = {
                                "status": rejection_report.get("status"),
                                "violations": list(
                                    rejection_report.get("violations") or []
                                ),
                                "missing_coverage": list(
                                    rejection_report.get("missing_coverage") or []
                                ),
                                "missing_full_runtime_coverage": list(
                                    rejection_report.get(
                                        "missing_full_runtime_coverage"
                                    )
                                    or []
                                ),
                                "missing_receipts": list(
                                    rejection_report.get("missing_receipts") or []
                                ),
                            }
                            return ExecutionResult(
                                term_out=[
                                    "Protocol Preflight rejected full execution before "
                                    f"subprocess launch: {error}\n",
                                    "Protocol Preflight report: "
                                    + json.dumps(rejection_details, sort_keys=True)
                                    + "\n",
                                ],
                                exec_time=time.time() - start_time,
                                exc_type="ProtocolPreflightError",
                                exc_info={
                                    "message": str(error),
                                    "code_sha256": source_code_sha256,
                                    "candidate_subprocess_started": False,
                                    "preflight_report": rejection_report,
                                    **rejection_details,
                                },
                                exc_stack=[],
                                protocol_observation={
                                    "protocol_preflight": rejection_report
                                },
                            )
                        preflight_report = {
                            "status": "shadow_error",
                            "code_sha256": source_code_sha256,
                            "reason": str(error),
                            "enforcement_mode": "shadow",
                        }
                        logger.warning("Host Preflight shadow error: %s", error)
                    else:
                        logger.info(
                            "Protocol Preflight %s observation: status=%s report=%s",
                            self.protocol_runtime_mode,
                            preflight_report.get("status"),
                            preflight_report.get("report_hash"),
                        )
                if agent_controls_preflight:
                    logger.info(
                        "Agent-controlled protocol review selected; Host dry-run "
                        "Preflight skipped for candidate %s",
                        id,
                    )

            # Preflight consumes this candidate's wall-clock allowance. Recheck
            # both the per-candidate deadline and the run finalization reserve
            # immediately before launching the full training subprocess.
            remaining_candidate = int(
                candidate_deadline_monotonic - time.monotonic()
            )
            remaining_run = self.effective_timeout_seconds()
            effective_timeout = min(remaining_candidate, remaining_run)
            if effective_timeout <= 0:
                return ExecutionResult(
                    term_out=[
                        "RunDeadlineExceeded: candidate preflight consumed the "
                        "remaining work budget; full execution was not started.\n"
                    ],
                    exec_time=time.time() - start_time,
                    exc_type="RunDeadlineExceeded",
                    exc_info={
                        "candidate_id": str(id),
                        "candidate_subprocess_started": False,
                        "finalize_reserve_seconds": self.finalize_reserve_seconds,
                    },
                    exc_stack=[],
                    protocol_observation=(
                        {"protocol_preflight": dict(preflight_report)}
                        if preflight_report is not None
                        else None
                    ),
                )
            if self.candidate_execution_contract:
                contract_audit = audit_candidate_code(
                    code,
                    self.candidate_execution_contract,
                )
                audit_root = run_wd / "working"
                audit_root.mkdir(parents=True, exist_ok=True)
                audit_path = audit_root / f"candidate_execution_contract_audit_{id}.json"
                audit_path.write_text(
                    json.dumps(contract_audit, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8",
                )
                audit_path.chmod(audit_path.stat().st_mode & ~0o222)
                if not contract_audit["valid"]:
                    block_receipt = build_candidate_execution_block_receipt(
                        node_id=str(id),
                        contract=self.candidate_execution_contract,
                        audit=contract_audit,
                    )
                    block_path = audit_root / (
                        f"candidate_execution_block_receipt_{id}.json"
                    )
                    block_path.write_text(
                        json.dumps(block_receipt, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    block_path.chmod(block_path.stat().st_mode & ~0o222)
                    message = "; ".join(contract_audit["violations"])
                    return ExecutionResult(
                        term_out=[
                            "Candidate execution contract rejected code before execution: "
                            f"{message}\n",
                            f"Audit receipt: {audit_path}\n",
                        ],
                        exec_time=time.time() - start_time,
                        exc_type="CandidateExecutionContractError",
                        exc_info={
                            "message": message,
                            "audit_path": str(audit_path),
                            "audit_hash": contract_audit["audit_hash"],
                            "block_receipt_path": str(block_path),
                            "block_receipt_hash": block_receipt["receipt_hash"],
                        },
                        exc_stack=[],
                    )
            candidate_code = self.isolate_submission_path(code=code, _id=id)
            candidate_code = self.isolate_model_path(code=candidate_code, _id=id)
            protocol_plan = None
            protocol_nonce = ""
            if _runtime_protocol_observer_enabled(self.cfg):
                protocol_plan = build_runtime_protocol_plan(
                    candidate_code,
                    source_code_sha256=source_code_sha256,
                )
                protocol_nonce = secrets.token_hex(32)
                try:
                    candidate_code = instrument_runtime_protocol_source(
                        candidate_code,
                        protocol_plan,
                        protocol_nonce,
                        filename=str(runfile_path),
                    )
                except Exception as error:
                    logger.warning(
                        "Host runtime protocol instrumentation unavailable: %s",
                        type(error).__name__,
                    )
            full_runtime_evidence = None
            if (
                self.protocol_runtime_mode
                in {"host_sdk_shadow", "host_sdk_enforce"}
                and self._protocol_contract is not None
                and self._collector_identity is not None
            ):
                node_token = hashlib.sha256(str(id).encode("utf-8")).hexdigest()[:24]
                full_runtime_root = (
                    Path(self.protocol_preflight_config.report_root).resolve()
                    / "full_runtime"
                    / source_code_sha256
                    / node_token
                )
                bootstrap_path = (
                    run_wd / "working" / f"host_full_runtime_{node_token}.json"
                )
                try:
                    full_runtime_controller = FullRuntimeEvidenceController(
                        contract=self._protocol_contract,
                        identity=self._collector_identity,
                        data_view_manifest_path=(
                            self.protocol_preflight_config.data_view_manifest_path
                        ),
                        output_root=full_runtime_root,
                        bootstrap_path=bootstrap_path,
                        run_id=f"full-runtime-{node_token}",
                        node_id=str(id),
                        code_sha256=source_code_sha256,
                    ).start()
                    pre_code += (
                        "from protocol_runtime.full_runtime import "
                        "activate_full_runtime_from_bootstrap as "
                        "__mlevolve_activate_full_runtime\n"
                        "__mlevolve_activate_full_runtime("
                        f"{str(bootstrap_path)!r}, "
                        f"runtime_mode={self.protocol_runtime_mode!r})\n"
                    )
                except Exception as error:
                    if full_runtime_controller is not None:
                        full_runtime_controller.stop()
                        full_runtime_controller = None
                    if self.protocol_runtime_mode == "host_sdk_enforce":
                        return ExecutionResult(
                            term_out=[
                                "Host full-runtime bootstrap failed before subprocess "
                                f"launch: {type(error).__name__}: {error}\n"
                            ],
                            exec_time=time.time() - start_time,
                            exc_type="ProtocolRuntimeBootstrapError",
                            exc_info={
                                "message": str(error),
                                "candidate_subprocess_started": False,
                            },
                            exc_stack=[],
                            protocol_observation={
                                "protocol_preflight": dict(preflight_report or {}),
                                "host_full_runtime": {
                                    "status": "bootstrap_error",
                                    "error_type": type(error).__name__,
                                    "error": str(error),
                                },
                            },
                        )
                    logger.warning(
                        "Host full-runtime shadow bootstrap failed: %s", error
                    )
            if adoption_verification_plan:
                try:
                    if self.protocol_runtime_mode == "legacy_ast":
                        raise ValueError(
                            "Agent adoption tracing requires host_sdk_shadow or host_sdk_enforce"
                        )
                    verify_adoption_plan(
                        adoption_verification_plan,
                        artifact_id=str(id),
                        source=code,
                    )
                    adoption_trace_nonce = secrets.token_hex(32)
                    trace_token = hashlib.sha256(
                        f"{id}|{adoption_verification_plan['plan_hash']}".encode("utf-8")
                    ).hexdigest()[:24]
                    adoption_trace_path = (
                        run_wd / "working" / f"adoption_trace_{trace_token}.json"
                    )
                    adoption_trace_path.unlink(missing_ok=True)
                    pre_code += adoption_trace_bootstrap(
                        adoption_verification_plan,
                        output_path=adoption_trace_path,
                        nonce=adoption_trace_nonce,
                        prefix=pre_code,
                    )
                except Exception as error:
                    logger.warning(
                        "Agent adoption runtime tracing unavailable for node %s: %s",
                        id,
                        error,
                    )
                    adoption_trace_evidence = {
                        "schema": "agent_adoption_runtime_trace_error_v1",
                        "artifact_id": str(id),
                        "code_sha256": source_code_sha256,
                        "plan_hash": str(
                            adoption_verification_plan.get("plan_hash") or ""
                        ),
                        "status": "unavailable",
                        "reason": f"{type(error).__name__}: {error}",
                        "probe_results": [],
                        "trace_hash": "",
                    }
            # Validate the Candidate and every Host source transformation
            # before launch.  Candidate syntax remains a normal debuggable
            # Candidate failure; a syntax error introduced only by Host
            # instrumentation is classified separately and must never be fed
            # back to the model as though its algorithm were at fault.
            try:
                compile(candidate_code, str(runfile_path), "exec")
            except SyntaxError as error:
                return ExecutionResult(
                    term_out=[f"Candidate source failed syntax validation: {error}\n"],
                    exec_time=time.time() - start_time,
                    exc_type="CandidateSourceSyntaxError",
                    exc_info={
                        "message": str(error),
                        "candidate_subprocess_started": False,
                        "host_instrumentation_failure": False,
                    },
                    exc_stack=[
                        (
                            self.agent_file_name[process_id],
                            int(error.lineno or 0),
                            "",
                            str(error.text or "").strip(),
                        )
                    ],
                )

            code = _insert_host_preamble_after_future_imports(
                candidate_code,
                pre_code,
            )
            try:
                compile(code, str(runfile_path), "exec")
            except SyntaxError as error:
                return ExecutionResult(
                    term_out=[
                        "Host source composition invalidated Candidate source: "
                        f"{error}\n"
                    ],
                    exec_time=time.time() - start_time,
                    exc_type="HostSourceInstrumentationError",
                    exc_info={
                        "message": str(error),
                        "candidate_subprocess_started": False,
                        "host_instrumentation_failure": True,
                    },
                    exc_stack=[
                        (
                            self.agent_file_name[process_id],
                            int(error.lineno or 0),
                            "",
                            str(error.text or "").strip(),
                        )
                    ],
                )

            with open(runfile_path, "w") as f:
                f.write(code)

            cmd = [sys.executable, str(runfile_path)]
            popen_isolation = {}
            if (
                _candidate_uid_isolation_enabled(self.cfg)
                and os.geteuid() == 0
            ):
                candidate_uid = int(
                    getattr(self.protocol_preflight_config, "candidate_uid", 65534)
                )
                for writable in (
                    run_wd,
                    run_wd / "submission",
                    run_wd / "working",
                ):
                    writable.mkdir(parents=True, exist_ok=True)
                    writable.chmod(0o777)
                popen_isolation = {
                    "user": candidate_uid,
                    "group": candidate_uid,
                    "extra_groups": (),
                }
            proc = subprocess.Popen(
                cmd,
                cwd=str(run_wd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=_execution_environment(
                    candidate_runtime_cache_root
                ),
                **popen_isolation,
            )
            with self._procs_lock:
                self._active_procs[process_id] = proc

            child_in_overtime = False
            exc_type = None
            exc_info = {}
            exc_stack = []
            
            try:
                stdout, stderr = proc.communicate(timeout=effective_timeout)
                exec_time = time.time() - start_time
                
                if proc.returncode != 0:
                    exc_type = "RuntimeError"
                    exc_info = {}
                    exc_stack = []
                    
                    if stderr:
                        stderr_text = stderr
                        exc_patterns = [
                            ("KeyboardInterrupt", "KeyboardInterrupt"),
                            ("TimeoutError", "TimeoutError"),
                            ("CUDA", "RuntimeError"),
                            ("cuda", "RuntimeError"),
                            ("ValueError", "ValueError"),
                            ("TypeError", "TypeError"),
                            ("AttributeError", "AttributeError"),
                            ("KeyError", "KeyError"),
                            ("IndexError", "IndexError"),
                            ("FileNotFoundError", "FileNotFoundError"),
                            ("ImportError", "ImportError"),
                            ("AssertionError", "AssertionError"),
                            ("NameError", "NameError"),
                            ("RuntimeError", "RuntimeError"),
                        ]
                        
                        for pattern, exc_name in exc_patterns:
                            if pattern in stderr_text:
                                exc_type = exc_name
                                break
                        
                        stderr_lines = stderr_text.splitlines()
                        for line in stderr_lines:
                            if 'File "' in line and 'line' in line:
                                try:
                                    file_start = line.find('File "') + 6
                                    file_end = line.find('"', file_start)
                                    if file_end > file_start:
                                        filename = line[file_start:file_end]
                                        line_start = line.find('line ') + 5
                                        line_end = line.find(',', line_start)
                                        if line_end == -1:
                                            line_end = line.find('\n', line_start)
                                        if line_end == -1:
                                            line_end = len(line)
                                        line_num_str = line[line_start:line_end].strip()
                                        if line_num_str.isdigit():
                                            line_num = int(line_num_str)
                                            func_name = ""
                                            if 'in ' in line:
                                                func_start = line.find('in ') + 3
                                                func_end = line.find('\n', func_start)
                                                if func_end == -1:
                                                    func_end = len(line)
                                                func_name = line[func_start:func_end].strip()
                                            
                                            filename_short = filename.replace(str(self.working_dir / self.agent_file_name[process_id]), self.agent_file_name[process_id])
                                            filename_short = os.path.basename(filename_short)
                                            exc_stack.append((filename_short, line_num, func_name, ""))
                                except Exception:
                                    pass
                        
                        for line in reversed(stderr_lines):
                            line = line.strip()
                            if line and not line.startswith("File") and not line.startswith("Traceback"):
                                if ":" in line:
                                    parts = line.split(":", 1)
                                    if len(parts) == 2:
                                        exc_info["message"] = parts[1].strip()
                                    break
            except subprocess.TimeoutExpired:
                logger.warning("Subprocess timeout, sending SIGINT...")
                try:
                    proc.send_signal(signal.SIGINT)
                    stdout, stderr = proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("Subprocess failed to terminate after SIGINT, killing...")
                    proc.kill()
                    stdout, stderr = proc.communicate()
                
                exec_time = time.time() - start_time
                exc_type = "TimeoutError"
                exc_info = {}
                exc_stack = []

            if full_runtime_controller is not None:
                try:
                    execution_succeeded = proc.returncode == 0 and exc_type is None
                    full_runtime_evidence = full_runtime_controller.seal(
                        exit_status=0 if execution_succeeded else 1,
                        executed_path=str(runfile_path),
                        run_hash=_sha256_file(runfile_path),
                    )
                    if (
                        execution_succeeded
                        and full_runtime_evidence.get("status") != "pass"
                        and self.protocol_runtime_mode == "host_sdk_enforce"
                    ):
                        exc_type = "ProtocolRuntimeEvidenceError"
                        exc_info = {
                            "message": "Host full-runtime lifecycle evidence is incomplete",
                            "missing_events": list(
                                full_runtime_evidence.get("missing_events") or []
                            ),
                            "candidate_subprocess_started": True,
                        }
                except Exception as error:
                    full_runtime_evidence = {
                        "status": "collector_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                    if proc.returncode == 0 and self.protocol_runtime_mode == "host_sdk_enforce":
                        exc_type = "ProtocolRuntimeEvidenceError"
                        exc_info = {
                            "message": str(error),
                            "candidate_subprocess_started": True,
                        }
                finally:
                    full_runtime_controller.stop()
                    full_runtime_controller = None

            if adoption_trace_path is not None:
                try:
                    adoption_trace_evidence = seal_adoption_runtime_trace(
                        raw_path=adoption_trace_path,
                        plan=adoption_verification_plan or {},
                        nonce=adoption_trace_nonce,
                        exit_status=(0 if proc.returncode == 0 and exc_type is None else 1),
                        identity=self._collector_identity,
                    )
                    adoption_trace_path.chmod(
                        adoption_trace_path.stat().st_mode & ~0o222
                    )
                except Exception as error:
                    logger.warning(
                        "Failed to seal Agent adoption trace for node %s: %s",
                        id,
                        error,
                    )
                    adoption_trace_evidence = {
                        "schema": "agent_adoption_runtime_trace_error_v1",
                        "artifact_id": str(id),
                        "code_sha256": source_code_sha256,
                        "plan_hash": str(
                            (adoption_verification_plan or {}).get("plan_hash") or ""
                        ),
                        "status": "unavailable",
                        "reason": f"{type(error).__name__}: {error}",
                        "probe_results": [],
                        "trace_hash": "",
                    }
            
            output: list[str] = []
            if stdout:
                output.extend(stdout.splitlines(keepends=True))
            if stderr:
                output.extend(stderr.splitlines(keepends=True))
            if not output:
                output = [""]
            if output and output[-1] and not output[-1].endswith("\n"):
                output.append("\n")

            if exc_type == "TimeoutError":
                output.append(
                    f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(effective_timeout)}"
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(effective_timeout)})."
                )

            protocol_observation = None
            if protocol_plan is not None:
                try:
                    protocol_observation = parse_runtime_protocol_trace(
                        stdout or "",
                        protocol_plan,
                        protocol_nonce,
                        execution_succeeded=(proc.returncode == 0 and exc_type is None),
                    )
                except Exception as error:
                    protocol_observation = {
                        "schema": OBSERVATION_SCHEMA,
                        "status": "blocked",
                        "reason": f"host_protocol_parser_error:{type(error).__name__}",
                        "source_code_sha256": source_code_sha256,
                        "plan_sha256": protocol_plan.get("plan_sha256", ""),
                        "event_hashes": {},
                        "scope_hashes": {},
                        "scope_input_hashes": {},
                        "scope_output_hashes": {},
                        "callable_refs": {},
                    }
                cleaned_stdout = strip_runtime_protocol_markers(stdout or "")
                output = []
                if cleaned_stdout:
                    output.extend(cleaned_stdout.splitlines(keepends=True))
                if stderr:
                    output.extend(stderr.splitlines(keepends=True))
                if not output:
                    output = [""]
                if output and output[-1] and not output[-1].endswith("\n"):
                    output.append("\n")
                if exc_type == "TimeoutError":
                    output.append(
                        f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(effective_timeout)}"
                    )
                else:
                    output.append(
                        f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(effective_timeout)})."
                    )

            if preflight_report is not None:
                if protocol_observation is None:
                    protocol_observation = {}
                attached_preflight = dict(preflight_report)
                if self.protocol_runtime_mode == "host_sdk_shadow":
                    attached_preflight["enforcement_mode"] = "shadow"
                    attached_preflight.setdefault(
                        "admission_disposition", "agent_review_then_execute"
                    )
                protocol_observation["protocol_preflight"] = attached_preflight
            if full_runtime_evidence is not None:
                if protocol_observation is None:
                    protocol_observation = {}
                protocol_observation["host_full_runtime"] = dict(
                    full_runtime_evidence
                )

            return ExecutionResult(
                output,
                exec_time,
                exc_type,
                exc_info,
                exc_stack,
                protocol_observation,
                adoption_trace_evidence,
            )
            
        except Exception as e:
            logger.error(f"Error in _run_subprocess: {e}")
            error_trace = traceback.format_exc()
            logger.error(error_trace)
            
            exec_time = time.time() - start_time if start_time else 0
            return ExecutionResult(
                term_out=[f"Subprocess execution error: {str(e)}", error_trace],
                exec_time=exec_time,
                exc_type="RuntimeError",
                exc_info={"error": str(e)},
                exc_stack=[],
            )
        finally:
            if full_runtime_controller is not None:
                try:
                    full_runtime_controller.stop()
                except Exception as error:
                    logger.warning(
                        "Failed to stop Host full-runtime Collector: %s", error
                    )
            if process_id is not None:
                with self._procs_lock:
                    self._active_procs.pop(process_id, None)
                    self._active_candidate_ids.discard(str(id))
            if proc is not None:
                try:
                    if proc.poll() is None:
                        logger.warning(f"Subprocess {process_id} still running, terminating...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            logger.warning(f"Subprocess {process_id} failed to terminate, killing...")
                            proc.kill()
                            proc.wait()
                except Exception as e:
                    logger.warning(f"Error cleaning up subprocess {process_id}: {e}")
            
            try:
                if runfile_path and runfile_path.exists():
                    os.remove(runfile_path)
            except Exception as e:
                logger.warning(f"Failed to remove runfile after subprocess execution: {e}")
            
            with self.lock:
                if process_id is not None:
                    self.status_map[process_id] = 0
                    self.current_parallel_run -= 1
