"""Training-before-terminal Protocol Preflight and bounded repair."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import Enum
import hashlib
import inspect
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import time
from typing import Any, Callable, Mapping

from authority.protocol_execution_contract import ProtocolExecutionContract
from authority.protocol_registry import ProtocolRegistry
from engine.candidate_execution_contract import (
    audit_candidate_code,
    build_candidate_execution_contract,
)

from .closure import dry_evidence_closure
from .collector import HostCollectorIdentity, HostCollectorSidecar
from .collector_bridge import bridge_signed_journal_to_receipts
from .collector_client import CollectorClient
from .data_views import verify_data_view_manifest
from .events import hash_payload, sha256_value
from .session import ProtocolSession
from .views import ProtocolSplit, build_view_handles


PREFLIGHT_REPORT_SCHEMA = "mlevolve_protocol_preflight_report_v1"
PREFLIGHT_REPAIR_RECEIPT_SCHEMA = "mlevolve_preflight_repair_receipt_v1"


class PreflightStatus(str, Enum):
    PASS = "pass"
    PROTOCOL_VIOLATION = "protocol_violation"
    MISSING_EVIDENCE = "missing_evidence"
    UNTRUSTED_EVIDENCE = "untrusted_evidence"
    RUNTIME_FAILURE = "runtime_failure"
    CONTRACT_MISMATCH = "contract_mismatch"
    COLLECTOR_INTERNAL_ERROR = "collector_internal_error"


class PreflightAdmissionError(ValueError):
    """Fail-closed admission error carrying the exact sealed report."""

    def __init__(self, message: str, report: Mapping[str, Any] | None = None):
        self.report = dict(report or {})
        super().__init__(message)


def _admission_failure_message(report: Mapping[str, Any]) -> str:
    details = {
        "status": report.get("status"),
        "violations": list(report.get("violations") or []),
        "missing_coverage": list(report.get("missing_coverage") or []),
        "missing_full_runtime_coverage": list(
            report.get("missing_full_runtime_coverage") or []
        ),
        "missing_receipts": list(report.get("missing_receipts") or []),
    }
    return (
        "Full execution requires Preflight PASS: "
        + json.dumps(details, sort_keys=True, separators=(",", ":"))
    )


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    content = (json.dumps(dict(value), indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _call_name(node: ast.AST) -> str:
    parts = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _direct_function_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[int, str]]:
    """Collect calls executed by one function, excluding nested definitions."""

    calls: list[tuple[int, str]] = []

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            if node is function:
                for statement in node.body:
                    self.visit(statement)

        def visit_ClassDef(self, _node: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, _node: ast.Lambda) -> None:
            return

        def visit_Call(self, node: ast.Call) -> None:
            calls.append((int(getattr(node, "lineno", 0)), _call_name(node.func)))
            self.generic_visit(node)

    Visitor().visit(function)
    return calls


def _expanded_function_calls(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    functions: Mapping[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[tuple[int, str]]:
    """Project local helper lifecycle effects onto their actual call sites."""

    cache: dict[str, list[str]] = {}

    def effects(name: str, stack: tuple[str, ...]) -> list[str]:
        if name in cache:
            return cache[name]
        if name in stack or name not in functions:
            return []
        values: list[str] = []
        for _line, called in _direct_function_calls(functions[name]):
            values.append(called)
            short = called.rsplit(".", 1)[-1]
            values.extend(effects(short, (*stack, name)))
        cache[name] = values
        return values

    expanded: list[tuple[int, str]] = []
    for line, called in _direct_function_calls(function):
        expanded.append((line, called))
        short = called.rsplit(".", 1)[-1]
        expanded.extend((line, effect) for effect in effects(short, (function.name,)))
    return expanded


def _execution_audit_contract(contract: ProtocolExecutionContract) -> dict[str, Any]:
    """Build the leakage-only static audit surface for a Host Protocol run.

    Model design and compute allocation are deliberately not Host validity
    criteria.  Epochs, folds, model count, installed packages, feature passes,
    checkpoints and remote assets are choices of the candidate/runtime.  The
    Host-specific checks below this audit retain only data/evidence isolation
    rules such as terminal-path denial, lifecycle coverage and selection
    freeze ordering.
    """

    budget = contract.execution_budget
    return build_candidate_execution_contract(
        contract_id=contract.contract_id,
        max_execution_seconds=max(1, int(budget.get("timeout_seconds", 60))),
        max_epochs=0,
        max_cv_folds=0,
        max_trainable_models=0,
        allowed_import_roots=(),
        allow_remote_assets=True,
        allow_unverified_local_assets=True,
        allow_dataset_wide_per_sample_precompute=True,
        allow_source_score_inheritance=False,
    )


def static_compatibility_check(
    source: str,
    contract: ProtocolExecutionContract,
    *,
    require_full_runtime_entrypoint: bool | None = None,
) -> dict[str, Any]:
    """P1/P2 static risk and Host SDK coverage check; never mints evidence.

    ``ProtocolPreflightRunner.run`` also supports dependency-light reference
    callables that receive a Host-created ``ProtocolSession`` directly.  Those
    callables exercise the SDK and Collector but are not standalone training
    programs, so they do not need a ``main()``/``current_session()`` entrypoint.
    Exact dynamic source admitted by ``run_source`` still requires the complete
    full-runtime entrypoint.
    """

    feasibility = audit_candidate_code(source, _execution_audit_contract(contract))
    violations = list(feasibility["violations"])
    warnings: list[str] = []
    calls: list[tuple[int, str]] = []
    terminal_markers: set[str] = set()
    unsafe_paths: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = ast.Module(body=[], type_ignores=[])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            calls.append((int(getattr(node, "lineno", 0)), name))
            short = name.rsplit(".", 1)[-1]
            if short in {"symlink", "link"}:
                violations.append(f"symlink_creation:line_{getattr(node, 'lineno', 0)}")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.lower()
            if any(
                marker in lowered
                for marker in (
                    "terminal_holdout",
                    "terminal_labels",
                    "/data/evaluator",
                    "evaluator_view/labels",
                )
            ):
                terminal_markers.add(node.value)
            if ".." in Path(node.value).parts:
                unsafe_paths.add(node.value)
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").startswith("protocol_runtime.collector"):
                violations.append("direct_collector_client_import")
        elif isinstance(node, ast.Import):
            if any(
                alias.name.startswith("protocol_runtime.collector")
                for alias in node.names
            ):
                violations.append("direct_collector_client_import")
    if terminal_markers:
        violations.append("terminal_path_access:" + ",".join(sorted(terminal_markers)))
    if unsafe_paths:
        violations.append("path_traversal:" + ",".join(sorted(unsafe_paths)))
    short_calls = [name.rsplit(".", 1)[-1] for _line, name in calls]
    if "CollectorClient" in short_calls:
        violations.append("direct_collector_client_construction")
    coverage = {
        "split_lineage": "get_split" in short_calls,
        "fit_scope": bool({"fit", "fit_scope", "fit_preprocessor"} & set(short_calls)),
        "prediction_scope": bool({"predict", "prediction_scope"} & set(short_calls)),
        "evaluator": "evaluate_internal" in short_calls,
        "selection_freeze": "freeze_selection" in short_calls,
    }
    missing_coverage = sorted(name for name, present in coverage.items() if not present)
    full_runtime_required = (
        bool(contract.adapter_spec.get("full_runtime_sdk_required", False))
        if require_full_runtime_entrypoint is None
        else bool(require_full_runtime_entrypoint)
    )
    all_functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    main_functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    ]
    main_call_names = {
        name.rsplit(".", 1)[-1]
        for function in main_functions
        for _line, name in _expanded_function_calls(function, all_functions)
    }
    full_runtime_coverage = {
        "current_session": "current_session" in main_call_names,
        "split_lineage": "get_split" in main_call_names,
        "fit_scope": bool(
            {"fit", "fit_scope", "fit_preprocessor"} & main_call_names
        ),
        "prediction_scope": bool(
            {"predict", "prediction_scope"} & main_call_names
        ),
        "evaluator": "evaluate_internal" in main_call_names,
        "selection_freeze": "freeze_selection" in main_call_names,
        "main_guard": any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and any(
                isinstance(value, ast.Constant) and value.value == "__main__"
                for value in node.test.comparators
            )
            for node in tree.body
        ),
    }
    if contract.adapter_spec.get("inference_view_required", False):
        full_runtime_coverage["inference_scope"] = (
            "inference_scope" in main_call_names
        )
    missing_full_runtime_coverage = sorted(
        name for name, present in full_runtime_coverage.items() if not present
    ) if full_runtime_required else []
    for function in [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]:
        function_calls = _expanded_function_calls(function, all_functions)
        freeze_lines = [
            line for line, name in function_calls if name.endswith("freeze_selection")
        ]
        if not freeze_lines:
            continue
        first_freeze = min(freeze_lines)
        inference_lines = [
            line for line, name in function_calls if name.endswith("inference_scope")
        ]
        if any(line < first_freeze for line in inference_lines):
            violations.append(f"pre_selection_inference:{function.name}")
        post_freeze = [
            name
            for line, name in function_calls
            if line > first_freeze
            and name.rsplit(".", 1)[-1]
            in {"fit", "fit_scope", "fit_preprocessor"}
        ]
        if post_freeze:
            violations.append(
                f"post_selection_tuning:{function.name}:" + ",".join(post_freeze)
            )
    if "legacy_ast" in source and "protocol_runtime" not in source:
        warnings.append("legacy_ast_only_is_shadow_diagnostic")
    if violations:
        status = PreflightStatus.PROTOCOL_VIOLATION.value
    elif missing_coverage or missing_full_runtime_coverage:
        status = PreflightStatus.MISSING_EVIDENCE.value
    else:
        status = PreflightStatus.PASS.value
    report = {
        "schema": "mlevolve_static_protocol_preflight_v1",
        "status": status,
        "code_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "contract_hash": contract.contract_hash,
        "violations": sorted(set(violations)),
        "warnings": sorted(set(warnings)),
        "contract_coverage": coverage,
        "missing_coverage": missing_coverage,
        "full_runtime_coverage": full_runtime_coverage,
        "missing_full_runtime_coverage": missing_full_runtime_coverage,
        "trusted_runtime_receipt_created": False,
        "static_report_hash": "",
    }
    report["static_report_hash"] = hash_payload(report, "static_report_hash")
    return report


def preflight_cache_key(
    *, code_sha256: str, contract_hash: str, image_digest: str, sdk_hash: str
) -> str:
    for name, value in {
        "code_sha256": code_sha256,
        "contract_hash": contract_hash,
        "image_digest": image_digest,
        "sdk_hash": sdk_hash,
    }.items():
        if not value:
            raise ValueError(f"Preflight cache key requires {name}")
    return sha256_value(
        {
            "code_sha256": code_sha256,
            "contract_hash": contract_hash,
            "image_digest": image_digest,
            "sdk_hash": sdk_hash,
        }
    )


def admission_report_path(report_root: str | Path, code_sha256: str) -> Path:
    root = Path(report_root).resolve()
    legacy_path = root / f"PREFLIGHT_REPORT.{code_sha256}.json"
    if legacy_path.is_file() or legacy_path.is_symlink():
        return legacy_path
    return root / "candidates" / code_sha256 / "PREFLIGHT_REPORT.json"


def validate_preflight_admission(
    source: str,
    *,
    report_root: str | Path,
    expected_contract_hash: str,
) -> dict[str, Any]:
    code_sha256 = hashlib.sha256(source.encode()).hexdigest()
    path = admission_report_path(report_root, code_sha256)
    if path.is_symlink() or not path.is_file():
        raise PreflightAdmissionError(
            "Full execution requires a matching Preflight report"
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != PREFLIGHT_REPORT_SCHEMA:
        raise PreflightAdmissionError(
            "Preflight admission report schema mismatch", report
        )
    if report.get("report_hash") != hash_payload(report, "report_hash"):
        raise PreflightAdmissionError(
            "Preflight admission report hash mismatch", report
        )
    if report.get("status") != PreflightStatus.PASS.value:
        raise PreflightAdmissionError(_admission_failure_message(report), report)
    if report.get("code_sha256") != code_sha256:
        raise PreflightAdmissionError("Preflight report code hash mismatch", report)
    if report.get("contract_hash") != expected_contract_hash:
        raise PreflightAdmissionError(
            "Preflight report Contract hash mismatch", report
        )
    if report.get("terminal_exposure_count") != 0:
        raise PreflightAdmissionError(
            "Preflight report contains terminal exposure", report
        )
    return report


def _candidate_worker(
    queue,
    candidate: Callable[[ProtocolSession], Any] | None,
    source: str | None,
    contract_payload: dict[str, Any],
    split: ProtocolSplit,
    client_values: dict[str, str],
    candidate_uid: int | None,
) -> None:
    try:
        if candidate_uid is not None:
            if os.geteuid() != 0:
                raise PermissionError(
                    "Host Candidate UID isolation requires a root Host launcher"
                )
            os.setgroups([])
            os.setgid(candidate_uid)
            os.setuid(candidate_uid)
        contract = ProtocolExecutionContract.from_dict(contract_payload)
        session = ProtocolSession(contract, split, CollectorClient(**client_values))
        if source is not None:
            namespace: dict[str, Any] = {
                "__name__": "mlevolve_host_candidate",
                "__file__": "preflight_candidate.py",
            }
            exec(compile(source, "preflight_candidate.py", "exec"), namespace)
            candidate = namespace.get("candidate")
            if not callable(candidate):
                raise TypeError(
                    "Host Protocol source must define callable candidate(session)"
                )
        if candidate is None:
            raise TypeError("Host Protocol Candidate is missing")
        candidate(session)
        queue.put({"status": "ok"})
    except BaseException as error:
        queue.put(
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error)[:2000],
            }
        )


@dataclass
class ProtocolPreflightRunner:
    registry: ProtocolRegistry

    def run(
        self,
        candidate: Callable[[ProtocolSession], Any] | None,
        *,
        source: str,
        contract: ProtocolExecutionContract,
        identity: HostCollectorIdentity,
        data_view_manifest_path: str | Path,
        output_root: str | Path,
        image_digest: str,
        sdk_hash: str,
        timeout_seconds: int = 60,
        candidate_uid: int | None = None,
        _execute_source: bool = False,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        root = Path(output_root).resolve()
        if root.exists() and any(root.iterdir()):
            raise ValueError("Preflight output root must be empty")
        root.mkdir(parents=True, exist_ok=True)
        code_sha256 = hashlib.sha256(source.encode()).hexdigest()
        if not _execute_source:
            try:
                callable_source = inspect.getsource(candidate)
            except (OSError, TypeError):
                callable_source = ""
            if callable_source and hashlib.sha256(callable_source.encode()).hexdigest() != code_sha256:
                raise ValueError("Preflight source does not match the executed candidate callable")
        static = static_compatibility_check(
            source,
            contract,
            require_full_runtime_entrypoint=_execute_source,
        )
        cache_key = preflight_cache_key(
            code_sha256=code_sha256,
            contract_hash=contract.contract_hash,
            image_digest=image_digest,
            sdk_hash=sdk_hash,
        )
        base = {
            "schema": PREFLIGHT_REPORT_SCHEMA,
            "code_sha256": code_sha256,
            "contract_hash": contract.contract_hash,
            "protocol_hash": contract.protocol_ref.canonical_hash,
            "cache_key": cache_key,
            "image_digest": image_digest,
            "sdk_hash": sdk_hash,
            "static_report_hash": static["static_report_hash"],
            "terminal_exposure_count": 0,
            "terminal_score_computed": False,
            "result_fact_created": False,
            "candidate_uid_isolated": candidate_uid is not None,
            "collector_report_hash": "",
            "closure_hash": "",
            "missing_receipts": [],
            "violations": list(static.get("violations") or []),
            "missing_coverage": list(static.get("missing_coverage") or []),
            "missing_full_runtime_coverage": list(
                static.get("missing_full_runtime_coverage") or []
            ),
            "runtime_error": {},
            "report_hash": "",
        }
        _write_exclusive(root / "STATIC_PREFLIGHT_REPORT.json", static)
        if static["status"] != PreflightStatus.PASS.value:
            report = {
                **base,
                "status": static["status"],
                "responsible_component": "candidate_source",
                "repairable": static["status"] == PreflightStatus.MISSING_EVIDENCE.value,
                "missing_receipts": sorted(
                    set(static["missing_coverage"])
                    | {
                        f"full_runtime:{value}"
                        for value in static.get(
                            "missing_full_runtime_coverage", []
                        )
                    }
                ),
            }
            return self._seal_report(root, report, started_at)
        try:
            verify_data_view_manifest(data_view_manifest_path, contract=contract)
        except ValueError as error:
            report = {
                **base,
                "status": PreflightStatus.CONTRACT_MISMATCH.value,
                "responsible_component": "host_data_view",
                "repairable": False,
                "runtime_error": {"error_type": type(error).__name__, "error": str(error)},
            }
            return self._seal_report(root, report, started_at)

        run_id = f"preflight-{cache_key[:16]}"
        node_id = f"preflight-node-{code_sha256[:16]}"
        try:
            sidecar = HostCollectorSidecar(
                root / "collector",
                contract.as_dict(),
                run_id=run_id,
                node_id=node_id,
                code_sha256=code_sha256,
                identity=identity,
            ).start()
        except Exception as error:
            report = {
                **base,
                "status": PreflightStatus.COLLECTOR_INTERNAL_ERROR.value,
                "responsible_component": "host_collector",
                "repairable": False,
                "runtime_error": {
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            }
            return self._seal_report(root, report, started_at)
        runtime_result: dict[str, Any] = {}
        try:
            split = build_view_handles(data_view_manifest_path, contract, sidecar)
            client = sidecar.client()
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            process = context.Process(
                target=_candidate_worker,
                args=(
                    queue,
                    None if _execute_source else candidate,
                    source if _execute_source else None,
                    contract.as_dict(),
                    split,
                    {
                        "socket_path": client.socket_path,
                        "run_id": client.run_id,
                        "node_id": client.node_id,
                        "code_sha256": client.code_sha256,
                        "contract_hash": client.contract_hash,
                    },
                    candidate_uid,
                ),
            )
            process.start()
            process.join(timeout=max(1, min(timeout_seconds, 60)))
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                runtime_result = {
                    "status": "error",
                    "error_type": "TimeoutError",
                    "error": "small dry-run exceeded timeout",
                }
            else:
                try:
                    runtime_result = queue.get(timeout=1)
                except Empty:
                    runtime_result = {
                        "status": "error",
                        "error_type": "CandidateProcessError",
                        "error": f"candidate process exit={process.exitcode}",
                    }
            if not runtime_result:
                runtime_result = {
                    "status": "error",
                    "error_type": "CandidateProcessError",
                    "error": f"candidate process exit={process.exitcode}",
                }
            exit_status = 0 if runtime_result.get("status") == "ok" else 1
            collector_report = sidecar.seal(
                exit_status=exit_status,
                executed_path="preflight_candidate.py",
                run_hash=sha256_value(
                    {"code_sha256": code_sha256, "contract_hash": contract.contract_hash}
                ),
            )
            base["collector_report_hash"] = collector_report["report_hash"]
            if runtime_result.get("status") != "ok":
                status = PreflightStatus.RUNTIME_FAILURE
                responsible = "candidate_runtime"
                missing = collector_report["missing_events"]
            elif collector_report.get("status") != "pass":
                status = PreflightStatus.MISSING_EVIDENCE
                responsible = "candidate_instrumentation"
                missing = collector_report["missing_events"]
            else:
                try:
                    receipts = bridge_signed_journal_to_receipts(
                        sidecar.output_dir, contract=contract
                    )
                except ValueError as error:
                    report = {
                        **base,
                        "status": PreflightStatus.UNTRUSTED_EVIDENCE.value,
                        "responsible_component": "collector_evidence",
                        "repairable": False,
                        "runtime_error": {
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                    }
                    return self._seal_report(root, report, started_at)
                closure = dry_evidence_closure(
                    contract,
                    self.registry,
                    receipts,
                    run_id=run_id,
                    node_id=node_id,
                    code_sha256=code_sha256,
                )
                _write_exclusive(root / "PREFLIGHT_EVIDENCE_CLOSURE.json", closure)
                base["closure_hash"] = closure["closure_hash"]
                decision = closure["authority_decision"]
                if closure["status"] == "pass":
                    status = PreflightStatus.PASS
                    responsible = "none"
                    missing = []
                else:
                    status = PreflightStatus.MISSING_EVIDENCE
                    responsible = "authority_evidence_closure"
                    missing = list(decision.get("missing_obligations") or [])
            report = {
                **base,
                "status": status.value,
                "responsible_component": responsible,
                "repairable": status in {
                    PreflightStatus.MISSING_EVIDENCE,
                    PreflightStatus.RUNTIME_FAILURE,
                },
                "missing_receipts": missing,
                "runtime_error": (
                    runtime_result if runtime_result.get("status") != "ok" else {}
                ),
            }
            return self._seal_report(root, report, started_at)
        except Exception as error:
            report = {
                **base,
                "status": PreflightStatus.COLLECTOR_INTERNAL_ERROR.value,
                "responsible_component": "host_collector",
                "repairable": False,
                "runtime_error": {"error_type": type(error).__name__, "error": str(error)},
            }
            return self._seal_report(root, report, started_at)
        finally:
            sidecar.stop()

    def run_source(
        self,
        *,
        source: str,
        contract: ProtocolExecutionContract,
        identity: HostCollectorIdentity,
        data_view_manifest_path: str | Path,
        output_root: str | Path,
        image_digest: str,
        sdk_hash: str,
        timeout_seconds: int = 60,
        candidate_uid: int | None = None,
    ) -> dict[str, Any]:
        """Preflight the exact dynamic source later admitted to full execution."""

        return self.run(
            None,
            source=source,
            contract=contract,
            identity=identity,
            data_view_manifest_path=data_view_manifest_path,
            output_root=output_root,
            image_digest=image_digest,
            sdk_hash=sdk_hash,
            timeout_seconds=timeout_seconds,
            candidate_uid=candidate_uid,
            _execute_source=True,
        )

    @staticmethod
    def _publish_reports(root: Path, report: Mapping[str, Any]) -> None:
        _write_exclusive(root / "PREFLIGHT_REPORT.json", report)
        admission = admission_report_path(root, str(report["code_sha256"]))
        _write_exclusive(admission, report)

    @classmethod
    def _seal_report(
        cls,
        root: Path,
        report: Mapping[str, Any],
        started_at: float,
    ) -> dict[str, Any]:
        value = dict(report)
        collector_root = root / "collector"
        manifest_path = collector_root / "RUNTIME_EVENT_JOURNAL_MANIFEST.json"
        collector_overhead = 0.0
        receipt_bytes = 0
        if manifest_path.is_file() and not manifest_path.is_symlink():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            collector_overhead = float(
                manifest.get("collector_processing_seconds") or 0.0
            )
            receipt_bytes = sum(
                path.stat().st_size
                for path in collector_root.iterdir()
                if path.is_file() and not path.is_symlink()
            )
        value["preflight_duration_seconds"] = round(
            time.perf_counter() - started_at, 9
        )
        value["collector_overhead_seconds"] = round(collector_overhead, 9)
        value["receipt_bytes"] = receipt_bytes
        value["report_hash"] = hash_payload(value, "report_hash")
        cls._publish_reports(root, value)
        return value


class _MethodNormalizer(ast.NodeTransformer):
    _INSTRUMENTATION_CALLS = {
        "get_split",
        "fit_scope",
        "prediction_scope",
        "evaluate_internal",
        "freeze_selection",
    }

    def visit_Import(self, node: ast.Import):
        names = [alias for alias in node.names if alias.name != "protocol_runtime"]
        return ast.Import(names=names) if names else None

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if (node.module or "").startswith("protocol_runtime"):
            return None
        return self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "candidate":
            return None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if node.name == "candidate":
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        value = node.value
        if isinstance(value, ast.Call) and _call_name(value.func).rsplit(".", 1)[-1] in self._INSTRUMENTATION_CALLS:
            return None
        return self.generic_visit(node)


def method_identity_fingerprint(source: str) -> str:
    tree = ast.parse(source)
    normalized = _MethodNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


def build_bounded_repair_receipt(
    original_source: str,
    repaired_source: str,
    *,
    preflight_status: str,
    repair_kind: str,
    attempt: int,
    max_attempts: int,
    terminal_score_observed: bool = False,
) -> dict[str, Any]:
    if terminal_score_observed:
        raise ValueError("Preflight repair may not observe a terminal score")
    if preflight_status not in {
        PreflightStatus.MISSING_EVIDENCE.value,
        PreflightStatus.RUNTIME_FAILURE.value,
    }:
        raise ValueError("This Preflight status is not repairable")
    if repair_kind not in {"instrumentation", "dependency", "budget_simplification"}:
        raise ValueError("Repair kind is outside the bounded Preflight surface")
    if attempt < 1 or attempt > max_attempts:
        raise ValueError("Preflight repair attempt exceeds its bound")
    original_method = method_identity_fingerprint(original_source)
    repaired_method = method_identity_fingerprint(repaired_source)
    receipt = {
        "schema": PREFLIGHT_REPAIR_RECEIPT_SCHEMA,
        "preflight_status": preflight_status,
        "repair_kind": repair_kind,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "original_code_sha256": hashlib.sha256(original_source.encode()).hexdigest(),
        "repaired_code_sha256": hashlib.sha256(repaired_source.encode()).hexdigest(),
        "original_method_fingerprint": original_method,
        "repaired_method_fingerprint": repaired_method,
        "method_identity_preserved": original_method == repaired_method,
        "terminal_score_observed": False,
        "runtime_receipt_fabricated": False,
        "disposition": "requires_new_preflight",
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = hash_payload(receipt, "receipt_hash")
    return receipt


__all__ = [
    "PREFLIGHT_REPORT_SCHEMA",
    "PREFLIGHT_REPAIR_RECEIPT_SCHEMA",
    "PreflightAdmissionError",
    "PreflightStatus",
    "ProtocolPreflightRunner",
    "admission_report_path",
    "build_bounded_repair_receipt",
    "method_identity_fingerprint",
    "preflight_cache_key",
    "static_compatibility_check",
    "validate_preflight_admission",
]
