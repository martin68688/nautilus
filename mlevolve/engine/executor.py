"""
Python interpreter for executing code snippets via subprocess.
- Executes code in a separate Python process (avoids CUDA/fork issues).
- Captures stdout/stderr, exceptions and stack traces, execution time limit.
- Supports multiple parallel slots (max_parallel_run) with CPU pinning.
"""

import logging
import hashlib
import json
import os
import secrets
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

logger = logging.getLogger("MLEvolve")


def _execution_environment() -> dict[str, str]:
    """Expose mlevolve's internal runtime helpers to isolated runfiles."""
    env = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[1])
    existing = env.get("PYTHONPATH", "")
    paths = [path for path in existing.split(os.pathsep) if path]
    if package_root not in paths:
        paths.insert(0, package_root)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _runtime_protocol_observer_enabled(cfg) -> bool:
    authority = getattr(cfg, "evaluation_authority", None) if cfg else None
    mode = str(getattr(authority, "mode", "off") or "off").lower()
    enabled = bool(
        getattr(authority, "runtime_protocol_observer_enabled", True)
    )
    return enabled and mode in {"shadow", "enforce"}

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

    def run(self, code: str, id, reset_session=True, working_dir: str | None = None):
        """
        Execute the provided Python command in a subprocess and return its output.

        Parameters:
            code: Python code to execute.
            reset_session: Reserved for future use.
            working_dir: Optional per-run working directory.

        Returns:
            ExecutionResult: output, exec_time, exc_type, exc_info, exc_stack.
        """
        return self._run_subprocess(code=code, id=id, working_dir=working_dir)

    def _run_subprocess(self, code: str, id, working_dir: str | None = None):
        """
        Execute code via subprocess (avoids CUDA fork issues).
        Aligned with multiprocessing mode for consistency.
        """
        logger.info("REPL is executing code via subprocess")
        logger.info(f"Current running process: {self.current_parallel_run}")
        process_id = None

        with self.lock:
            self.current_parallel_run += 1
            for idx in range(self.max_parallel_run):
                if self.status_map[idx] == 0:
                    self.status_map[idx] = 1
                    process_id = idx
                    logger.info(f"Assigned process_id: {process_id}")
                    break
                elif idx == self.max_parallel_run - 1:
                    logger.info("reach max process parallel number")
                    raise ValueError("reach max process parallel number")

        start_time = time.time()
        runfile_path = None
        proc = None
        
        try:
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

            # GPU allocation for multi-GPU support
            num_gpus = getattr(self.cfg.agent.search, 'num_gpus', 1) if self.cfg else 1
            gpu_id = process_id % num_gpus
            logger.info(f"has set process_id:{process_id} to use GPU: {gpu_id}")

            # decide runfile location and cwd
            run_wd = Path(working_dir).resolve() if working_dir is not None else self.working_dir
            runfile_path = run_wd / self.agent_file_name[process_id]
            run_wd.mkdir(parents=True, exist_ok=True)

            pre_code = "import os\nif hasattr(os, 'sched_setaffinity'):\n    os.sched_setaffinity(0, {cpu_set})\nos.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'\n".format(cpu_set=cpu_set, gpu_id=gpu_id)
            source_code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
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
            code = pre_code + candidate_code

            with open(runfile_path, "w") as f:
                f.write(code)

            cmd = [sys.executable, str(runfile_path)]
            proc = subprocess.Popen(
                cmd,
                cwd=str(run_wd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=_execution_environment(),
            )
            with self._procs_lock:
                self._active_procs[process_id] = proc

            child_in_overtime = False
            exc_type = None
            exc_info = {}
            exc_stack = []
            
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
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
                    f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                )
            else:
                output.append(
                    f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
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
                        f"Execution time: TimeoutError: Execution exceeded the time limit of {humanize.naturaldelta(self.timeout)}"
                    )
                else:
                    output.append(
                        f"Execution time: {humanize.naturaldelta(exec_time)} seconds (time limit is {humanize.naturaldelta(self.timeout)})."
                    )

            return ExecutionResult(
                output,
                exec_time,
                exc_type,
                exc_info,
                exc_stack,
                protocol_observation,
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
            if process_id is not None:
                with self._procs_lock:
                    self._active_procs.pop(process_id, None)
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
