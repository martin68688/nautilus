"""Run the frozen WP8 final regression matrix and emit a hash-bound receipt.

The runner is intentionally host-owned.  It snapshots the implementation/test
surface before executing pytest, records every exact argv and JUnit file, then
requires the source snapshot to be byte-identical after the run.  A historical
failed Tier-2 targeted run is retained only as a superseded artifact; it never
contributes to the pass decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


RECEIPT_SCHEMA = "decision_admissibility_wp8_final_test_receipt_v1"
MANIFEST_SCHEMA = "decision_admissibility_wp8_final_test_manifest_v1"
SOURCE_INVENTORY_SCHEMA = "decision_admissibility_source_inventory_row_v1"

SOURCE_DIR_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    "mlevolve": (
        ".py",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".sh",
        ".txt",
    ),
    "paper-skills/memory_bundle": (
        ".py", ".json", ".jsonl", ".yaml", ".yml", ".toml",
        ".npz", ".joblib", ".txt", ".patch",
    ),
    "paper-skills/distillation": (
        ".py", ".json", ".jsonl", ".yaml", ".yml", ".toml",
        ".npz", ".joblib", ".txt", ".patch",
    ),
    "paper-skills/hyper_memory": (
        ".py", ".json", ".jsonl", ".yaml", ".yml", ".toml",
        ".npz", ".joblib", ".txt", ".patch",
    ),
    "paper-skills/eval_composite_memory": (
        ".py", ".json", ".jsonl", ".yaml", ".yml", ".toml",
        ".npz", ".joblib", ".txt", ".patch",
    ),
    "paper-skills/eval_skill_memory": (
        ".py", ".json", ".jsonl", ".yaml", ".yml", ".toml",
        ".npz", ".joblib", ".txt", ".patch",
    ),
    "tests": (
        ".py",
        ".json",
        ".jsonl",
        ".yaml",
        ".yml",
        ".toml",
        ".npz",
        ".joblib",
        ".txt",
    ),
}
SOURCE_EXACT_PATHS = (
    "coordination/decision_admissibility_complete_execution_plan_20260719.md",
    "paper-skills/eval_composite_memory/manifests/claim_gates_v1.yaml",
    "paper-skills/eval_composite_memory/manifests/condition_manifest_v1.yaml",
    (
        "paper-skills/eval_composite_memory/manifests/"
        "replay_heldout_detector_provenance_addendum_v1.json"
    ),
    "paper-skills/eval_composite_memory/manifests/replay_heldout_lock_v1.json",
    "paper-skills/eval_composite_memory/manifests/task_manifest_v1.yaml",
    "paper-skills/eval_skill_memory/clean_replay_targets.json",
    "paper-skills/eval_skill_memory/clean_run_allowlist.json",
    "paper-skills/eval_skill_memory/non_spooky_replay_source_manifest_v1.json",
    "paper-skills/eval_skill_memory/requirements-decision-point-benchmark.txt",
    "paper-skills/eval_skill_memory/run_identity_registry_v1.json",
    "paper-skills/hyper_memory/sop_taxonomy.json",
    "paper-skills/hyper_memory/sop_taxonomy_overrides.json",
    "papers/runforest_iclr2025/evidence/claims.md",
)
SOURCE_EXCLUDED_PREFIXES = (
    "mlevolve/data/",
    "mlevolve/inference/submissions/",
    "mlevolve/runs/",
    "mlevolve/runs_backup/",
    "paper-skills/distillation/distill_branch3_demo/",
    "paper-skills/distillation/graph_build/",
    "paper-skills/hyper_memory/v3_sentence_embedding_probe/",
)
PASSTHROUGH_ENV_KEYS = (
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "SHELL",
    "TMPDIR",
    "TZ",
    "USER",
)
CONTROLLED_ENVIRONMENT = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "mlevolve:paper-skills/memory_bundle",
    "PYTEST_ADDOPTS": "",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
}
BOUND_DISTRIBUTIONS = (
    "joblib",
    "numpy",
    "pandas",
    "pyarrow",
    "pytest",
    "scikit-learn",
    "scipy",
    "torch",
)

PLAN_20_1 = (
    "tests/authority",
    "tests/test_stage_aware_hybrid_memory.py",
    "tests/test_causal_granularity_benchmark_v2.py",
    "tests/test_protocol_repair.py",
    "tests/test_run_forest_memory.py",
)
PLAN_20_1_A = (
    "tests/authority/test_actuation_pipeline.py",
    "tests/authority/test_runtime_protocol_observer.py",
    "tests/authority/test_claim_types.py",
    "tests/authority/test_mlevolve_adapter.py",
    "tests/authority/test_enforce_rollout.py",
)
PLAN_20_2 = (
    "tests/authority/test_stage_ontology.py",
    "tests/authority/test_claim_decomposition.py",
    "tests/authority/test_mixed_value_authority.py",
    "tests/authority/test_trusted_collectors.py",
    "tests/authority/test_receipt_trust_boundary.py",
    "tests/authority/test_sop_visibility_gateway.py",
    "tests/authority/test_mixed_value_sop_visibility.py",
    "tests/authority/test_visibility_pre_prompt.py",
    "tests/authority/test_visibility_projection_bypass.py",
    "tests/authority/test_legacy_sop_visibility.py",
    "tests/authority/test_experience_contract.py",
    "tests/authority/test_actuation_pipeline.py",
    "tests/authority/test_counterfactual_actuation.py",
    "tests/authority/test_result_adoption_causal_writeback.py",
    "tests/authority/test_legacy_promote_not_used.py",
    "tests/authority/test_method_preserving_replay.py",
    "tests/authority/test_method_changing_fake_replay.py",
)
PLAN_20_3 = (
    "tests/test_corpus_manifest.py",
    "tests/test_corpus_split_isolation.py",
    "tests/test_run_forest_bundle_v2.py",
    "tests/test_sop_clause_distillation_schema.py",
    "tests/test_memory_bundle_validation.py",
    "tests/test_memory_snapshot_overlay.py",
    "tests/test_fixed_holdout_terminal_writeback.py",
    "tests/test_positive_result_vs_adopted_distillation.py",
    "tests/test_sleep_time_bundle_publication.py",
    "tests/test_multigeneration_contamination.py",
    "tests/test_decision_admissibility_factorial.py",
)
FROZEN_EXCEPTION_MODULE = "tests/test_composite_memory_benchmark.py"
FROZEN_LOCK_REGRESSION_TEST = (
    "tests.test_composite_memory_benchmark."
    "test_heldout_replay_set_is_independently_authored_and_frozen"
)
FROZEN_LOCK_PATH = (
    "paper-skills/eval_composite_memory/manifests/replay_heldout_lock_v1.json"
)
FROZEN_DETECTOR_PATH = "mlevolve/agents/leakage_audit.py"
SUPERSEDED_JUNIT_SPECS = (
    {
        "path": "tier2_formal_and_new.xml",
        "disposition": "failed",
        "expected_failures": (
            "tests.test_tier2_formal_continuation_amendment::"
            "test_r7_continuation_amendment_verifies",
            "tests.test_tier2_formal_preterminal_recovery_amendment::"
            "test_r6_preterminal_recovery_amendment_verifies",
        ),
    },
    {
        "path": "tier2_formal_and_new_r2.xml",
        "disposition": "passed_replacement",
        "expected_failures": (),
    },
    {
        "path": "baseline_section_20_1_r2.xml",
        "disposition": "closeout_failed",
        "expected_failures": (
            "tests.authority.test_replay_authority_recovery::"
            "test_clean_replay_recovers_only_new_claim_and_publishes_scoped_bundle",
        ),
    },
    {
        "path": "baseline_section_20_1_r3.xml",
        "disposition": "closeout_passed_replacement",
        "expected_failures": (),
    },
)


@dataclass(frozen=True)
class TestSpec:
    name: str
    paths: tuple[str, ...]
    expected_outcome: str = "pass"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        _canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def _write_text_exclusive(path: Path, value: str, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_exclusive(path, _json_text(value))


def _relative_file(repo_root: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(f"Source path escapes repository: {path}") from error
    if path.is_symlink():
        raise ValueError(f"Symlink is forbidden in source snapshot: {path}")
    if not path.is_file():
        raise ValueError(f"Source snapshot entry is not a regular file: {path}")
    return relative.as_posix()


def source_inventory(repo_root: str | Path) -> list[dict[str, Any]]:
    repo_root = Path(repo_root).resolve()
    selected: dict[str, Path] = {}
    for relative_root, suffixes in SOURCE_DIR_SUFFIXES.items():
        root = repo_root / relative_root
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            relative = _relative_file(repo_root, path)
            if (
                "__pycache__" in Path(relative).parts
                or ".pytest_cache" in Path(relative).parts
                or relative.startswith(SOURCE_EXCLUDED_PREFIXES)
            ):
                continue
            selected[relative] = path
    deploy_root = repo_root / "deploy"
    if not deploy_root.is_dir():
        raise FileNotFoundError(deploy_root)
    for path in deploy_root.iterdir():
        if (
            path.is_file()
            and "decision_admissibility" in path.name
            and path.suffix in {".py", ".sh", ".json", ".yaml", ".yml"}
        ):
            selected[_relative_file(repo_root, path)] = path
    for relative in SOURCE_EXACT_PATHS:
        path = repo_root / relative
        selected[_relative_file(repo_root, path)] = path
    return [
        {
            "schema": SOURCE_INVENTORY_SCHEMA,
            "path": relative,
            "size_bytes": selected[relative].stat().st_size,
            "sha256": _sha256_file(selected[relative]),
        }
        for relative in sorted(selected)
    ]


def _inventory_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)


def source_snapshot(repo_root: str | Path) -> dict[str, Any]:
    rows = source_inventory(repo_root)
    payload = _inventory_bytes(rows)
    return {
        "algorithm": "sha256(canonical-jsonl(path,size_bytes,sha256))",
        "row_count": len(rows),
        "inventory_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": rows,
    }


def _testcase_id(case: ET.Element) -> str:
    classname = str(case.attrib.get("classname") or "")
    name = str(case.attrib.get("name") or "")
    if not classname or not name:
        raise ValueError(f"JUnit testcase is missing classname or name: {case.attrib}")
    return f"{classname}::{name}"


def _parse_junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    else:
        raise ValueError(f"Unsupported JUnit root: {root.tag}")
    if not suites:
        raise ValueError(f"JUnit has no testsuite: {path}")
    cases = list(root.iter("testcase"))
    testcase_ids = [_testcase_id(case) for case in cases]
    if len(testcase_ids) != len(set(testcase_ids)):
        raise ValueError(f"JUnit contains duplicate testcase identities: {path}")
    failing_tests = sorted(
        _testcase_id(case)
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    )
    observed = {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }
    declared = {
        name: sum(int(suite.attrib.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if declared != observed:
        raise ValueError(
            f"JUnit declared/observed count mismatch: {path}: "
            f"{declared} != {observed}"
        )
    return {
        **observed,
        "time_seconds": sum(
            float(suite.attrib.get("time", "0") or 0) for suite in suites
        ),
        "failing_tests": failing_tests,
        "testcase_ids": sorted(testcase_ids),
        "testcase_ids_sha256": hashlib.sha256(
            _canonical_bytes(sorted(testcase_ids))
        ).hexdigest(),
    }


def _formal_target_paths(repo_root: Path) -> tuple[str, ...]:
    names = {
        path.relative_to(repo_root).as_posix()
        for path in (repo_root / "tests").glob("test_tier2_formal_*.py")
    }
    names.update(
        {
            "tests/test_wp8_evidence_ledger.py",
            "tests/test_wp8_final_stop_gate.py",
        }
    )
    return tuple(sorted(names))


def test_specs(repo_root: str | Path) -> tuple[TestSpec, ...]:
    repo_root = Path(repo_root).resolve()
    return (
        TestSpec("baseline_section_20_1", PLAN_20_1),
        TestSpec("result_adoption_causal_section_20_1_A", PLAN_20_1_A),
        TestSpec("new_unit_section_20_2", PLAN_20_2),
        TestSpec("integration_section_20_3", PLAN_20_3),
        TestSpec("tier2_formal_and_final", _formal_target_paths(repo_root)),
        TestSpec("full_suite_r2", ("tests",)),
        TestSpec(
            "frozen_composite_lock_regression",
            (FROZEN_EXCEPTION_MODULE,),
        ),
    )


def _command_template(spec: TestSpec, junit_name: str) -> list[str]:
    return [
        "$PYTHON",
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *spec.paths,
        f"--junitxml={{OUTPUT_ROOT}}/{junit_name}",
    ]


def _execute_spec(
    *,
    spec: TestSpec,
    repo_root: Path,
    output_root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    junit_name = f"{spec.name}.xml"
    log_name = f"{spec.name}.log"
    junit_path = output_root / junit_name
    log_path = output_root / log_name
    argv = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        *spec.paths,
        f"--junitxml={junit_path}",
    ]
    started_ns = time.time_ns()
    completed = subprocess.run(
        argv,
        cwd=repo_root,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    ended_ns = time.time_ns()
    _write_text_exclusive(log_path, completed.stdout)
    if not junit_path.is_file():
        raise RuntimeError(f"pytest did not create JUnit XML: {spec.name}")
    os.chmod(junit_path, 0o444)
    junit = _parse_junit(junit_path)
    if spec.expected_outcome == "pass":
        accepted = bool(
            completed.returncode == 0
            and junit["failures"] == 0
            and junit["errors"] == 0
            and junit["skipped"] == 0
            and junit["tests"] > 0
        )
    else:  # reserved for future explicitly documented non-green probes
        accepted = False
    return {
        "name": spec.name,
        "required_green": spec.expected_outcome == "pass",
        "expected_outcome": spec.expected_outcome,
        "accepted": accepted,
        "argv_template": _command_template(spec, junit_name),
        "executed_argv": argv,
        "working_directory": str(repo_root),
        "exit_code": completed.returncode,
        "started_unix_ns": started_ns,
        "ended_unix_ns": ended_ns,
        "duration_seconds": (ended_ns - started_ns) / 1_000_000_000,
        "junit_path": junit_name,
        "junit_sha256": _sha256_file(junit_path),
        "junit": junit,
        "log_path": log_name,
        "log_sha256": _sha256_file(log_path),
    }


def _copy_superseded_artifacts(
    *, repo_root: Path, output_root: Path
) -> list[dict[str, Any]]:
    source_root = repo_root / "coordination" / (
        "decision_admissibility_wp8_final_tests_20260723_r1"
    )
    copied: list[dict[str, Any]] = []
    for specification in SUPERSEDED_JUNIT_SPECS:
        filename = str(specification["path"])
        disposition = str(specification["disposition"])
        expected_failures = list(specification["expected_failures"])
        source = source_root / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output_root / filename
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with source.open("rb") as reader, os.fdopen(descriptor, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        junit = _parse_junit(target)
        if (
            junit["failing_tests"] != expected_failures
            or junit["failures"] + junit["errors"] != len(expected_failures)
            or junit["skipped"] != 0
        ):
            raise ValueError(f"Unexpected superseded JUnit state: {filename}")
        copied.append(
            {
                "path": filename,
                "sha256": _sha256_file(target),
                "disposition": disposition,
                "junit": junit,
            }
        )
    return copied


def _git_value(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _subprocess_environment() -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in PASSTHROUGH_ENV_KEYS
        if key in os.environ
    }
    environment.update(CONTROLLED_ENVIRONMENT)
    return environment


def _environment_receipt(repo_root: Path) -> dict[str, Any]:
    repo_venv = (repo_root / ".venv").resolve()
    python_prefix = Path(sys.prefix).resolve()
    if python_prefix != repo_venv:
        raise RuntimeError(
            f"WP8 final tests require the repository .venv: "
            f"{python_prefix} != {repo_venv}"
        )
    subprocess_environment = _subprocess_environment()
    distribution_versions = {
        name: importlib.metadata.version(name) for name in BOUND_DISTRIBUTIONS
    }
    return {
        "python_executable": sys.executable,
        "python_executable_sha256": _sha256_file(Path(sys.executable)),
        "python_prefix": str(python_prefix),
        "repo_venv": str(repo_venv),
        "repo_venv_active": True,
        "pytest_cache_disabled": True,
        "bound_distribution_versions": distribution_versions,
        "bound_distribution_versions_sha256": hashlib.sha256(
            _canonical_bytes(distribution_versions)
        ).hexdigest(),
        "subprocess_environment": subprocess_environment,
        "subprocess_environment_sha256": hashlib.sha256(
            _canonical_bytes(subprocess_environment)
        ).hexdigest(),
    }


def build_test_receipt(
    *, repo_root: str | Path, output_root: str | Path, created_at: str
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    raw_output_root = Path(output_root)
    if raw_output_root.is_symlink():
        raise FileExistsError(f"Refusing symlink output root: {raw_output_root}")
    output_root = raw_output_root.resolve()
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to reuse output root: {output_root}") from error

    environment_receipt = _environment_receipt(repo_root)
    before = source_snapshot(repo_root)
    inventory_path = output_root / "source_inventory.jsonl"
    _write_text_exclusive(
        inventory_path,
        _inventory_bytes(before["rows"]).decode("utf-8"),
    )
    environment = dict(environment_receipt["subprocess_environment"])
    test_runs = [
        _execute_spec(
            spec=spec,
            repo_root=repo_root,
            output_root=output_root,
            environment=environment,
        )
        for spec in test_specs(repo_root)
    ]
    after = source_snapshot(repo_root)
    superseded = _copy_superseded_artifacts(
        repo_root=repo_root, output_root=output_root
    )
    frozen_bindings = {
        relative: _sha256_file(repo_root / relative)
        for relative in (FROZEN_LOCK_PATH, FROZEN_DETECTOR_PATH)
    }
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "created_at": created_at,
        "repo_identity": {
            "branch": _git_value(repo_root, "branch", "--show-current"),
            "head": _git_value(repo_root, "rev-parse", "HEAD"),
            "dirty_worktree_expected": True,
        },
        "environment": environment_receipt,
        "source_snapshot": {
            key: value for key, value in before.items() if key != "rows"
        },
        "source_inventory_path": "source_inventory.jsonl",
        "source_inventory_file_sha256": _sha256_file(inventory_path),
        "source_unchanged_during_tests": before == after,
        "test_runs": test_runs,
        "required_green_run_names": [
            spec.name
            for spec in test_specs(repo_root)
            if spec.expected_outcome == "pass"
        ],
        "full_suite_run_name": "full_suite_r2",
        "historical_frozen_exception_resolution": {
            "run_name": "frozen_composite_lock_regression",
            "module": FROZEN_EXCEPTION_MODULE,
            "historically_locked_test": FROZEN_LOCK_REGRESSION_TEST,
            "lock_and_detector_sha256": frozen_bindings,
            "current_module_required_green": True,
            "excluded_from_full_suite": False,
        },
        "superseded_chain": {
            "relations": [
                "tier2_formal_and_new.xml superseded_by tier2_formal_and_new_r2.xml",
                "baseline_section_20_1_r2.xml superseded_by baseline_section_20_1_r3.xml",
            ],
            "artifacts": superseded,
            "superseded_failure_not_used_for_gate": True,
        },
        "runner_source_sha256": _sha256_file(Path(__file__).resolve()),
        "receipt_hash": "",
    }
    required = [run for run in test_runs if run["required_green"]]
    receipt["status"] = (
        "passed"
        if before == after
        and all(run["accepted"] for run in required)
        else "failed"
    )
    receipt["receipt_hash"] = _payload_hash(receipt, "receipt_hash")
    receipt_path = output_root / "test_receipt.json"
    _write_json_exclusive(receipt_path, receipt)
    files = {
        path.relative_to(output_root).as_posix(): _sha256_file(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
    }
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": receipt["status"],
        "receipt_hash": receipt["receipt_hash"],
        "files": files,
        "runner_source_sha256": receipt["runner_source_sha256"],
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    _write_json_exclusive(output_root / "manifest.json", manifest)
    directory_descriptor = os.open(output_root, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    os.chmod(output_root, 0o555)
    return receipt


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_inventory(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Source inventory must contain JSON objects")
    return rows


def _safe_receipt_child(root: Path, relative: object) -> Path:
    text = str(relative or "")
    pure = PurePosixPath(text)
    if (
        not text
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"Unsafe receipt path: {text!r}")
    if len(pure.parts) != 1:
        raise ValueError(f"Receipt artifacts must be flat: {text!r}")
    path = root.joinpath(*pure.parts)
    if path.is_symlink():
        raise ValueError(f"Symlink is forbidden in receipt: {text!r}")
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError(f"Receipt path escapes root: {text!r}") from error
    return path


def verify_test_receipt(
    *, receipt_root: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    raw_receipt_root = Path(receipt_root)
    errors: list[str] = []
    if raw_receipt_root.is_symlink():
        errors.append("receipt_root_symlink")
    receipt_root = raw_receipt_root.resolve()
    repo_root = Path(repo_root).resolve()
    receipt_path = receipt_root / "test_receipt.json"
    manifest_path = receipt_root / "manifest.json"
    inventory_path = receipt_root / "source_inventory.jsonl"
    if not receipt_root.is_dir():
        errors.append("receipt_root_missing")
    elif receipt_root.stat().st_mode & 0o222:
        errors.append("receipt_root_writable")
    try:
        receipt = _read_object(receipt_path)
        manifest = _read_object(manifest_path)
        observed_rows = _read_inventory(inventory_path)
    except Exception as error:
        receipt = {}
        manifest = {}
        observed_rows = []
        errors.append(f"receipt_read:{type(error).__name__}")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        errors.append("receipt_schema")
    if receipt.get("status") != "passed":
        errors.append("receipt_status")
    if receipt.get("receipt_hash") != _payload_hash(receipt, "receipt_hash"):
        errors.append("receipt_hash")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("manifest_schema")
    if manifest.get("status") != "passed":
        errors.append("manifest_status")
    if manifest.get("manifest_hash") != _payload_hash(manifest, "manifest_hash"):
        errors.append("manifest_hash")
    if manifest_path.is_file() and manifest_path.stat().st_mode & 0o222:
        errors.append("manifest_writable")
    if manifest.get("receipt_hash") != receipt.get("receipt_hash"):
        errors.append("manifest_receipt_binding")
    if manifest.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve()):
        errors.append("manifest_runner_source_hash")
    declared_files = manifest.get("files") or {}
    observed_entries = list(receipt_root.iterdir()) if receipt_root.is_dir() else []
    for path in observed_entries:
        if path.is_symlink() or not path.is_file():
            errors.append(f"receipt_non_regular_entry:{path.name}")
    observed_files = {
        path.name
        for path in observed_entries
        if path.is_file() and not path.is_symlink() and path.name != "manifest.json"
    }
    if set(declared_files) != observed_files:
        errors.append("manifest_file_set")
    for relative, expected in declared_files.items():
        try:
            path = _safe_receipt_child(receipt_root, relative)
        except ValueError:
            errors.append(f"manifest_path:{relative}")
            continue
        if not path.is_file() or _sha256_file(path) != expected:
            errors.append(f"manifest_file:{relative}")
        elif path.stat().st_mode & 0o222:
            errors.append(f"manifest_file_writable:{relative}")
    if "manifest.json" in (manifest.get("files") or {}):
        errors.append("manifest_self_reference")
    current_rows = source_inventory(repo_root)
    current_inventory_bytes = _inventory_bytes(current_rows)
    observed_inventory_bytes = _inventory_bytes(observed_rows)
    snapshot = receipt.get("source_snapshot") or {}
    if observed_rows != current_rows:
        errors.append("stale_source_snapshot")
    if snapshot.get("row_count") != len(observed_rows):
        errors.append("source_row_count")
    if snapshot.get("inventory_sha256") != hashlib.sha256(
        observed_inventory_bytes
    ).hexdigest():
        errors.append("source_inventory_hash")
    if not inventory_path.is_file() or receipt.get(
        "source_inventory_file_sha256"
    ) != _sha256_file(inventory_path):
        errors.append("source_inventory_file_hash")
    if current_inventory_bytes != observed_inventory_bytes:
        errors.append("source_inventory_recompute")
    if receipt.get("source_unchanged_during_tests") is not True:
        errors.append("source_changed_during_tests")

    expected_repo_identity = {
        "branch": _git_value(repo_root, "branch", "--show-current"),
        "head": _git_value(repo_root, "rev-parse", "HEAD"),
        "dirty_worktree_expected": True,
    }
    if receipt.get("repo_identity") != expected_repo_identity:
        errors.append("repo_identity")
    try:
        expected_environment = _environment_receipt(repo_root)
    except Exception as error:
        expected_environment = None
        errors.append(f"execution_environment_runtime:{type(error).__name__}")
    if receipt.get("environment") != expected_environment:
        errors.append("execution_environment")

    expected_specs = {spec.name: spec for spec in test_specs(repo_root)}
    runs = receipt.get("test_runs") or []
    run_by_name = {str(run.get("name") or ""): run for run in runs}
    if set(run_by_name) != set(expected_specs) or len(runs) != len(run_by_name):
        errors.append("test_run_set")
    expected_green = sorted(
        spec.name for spec in expected_specs.values() if spec.expected_outcome == "pass"
    )
    if sorted(receipt.get("required_green_run_names") or []) != expected_green:
        errors.append("required_green_set")
    if receipt.get("full_suite_run_name") != "full_suite_r2":
        errors.append("full_suite_missing")
    observed_junit_paths: set[str] = set()
    observed_log_paths: set[str] = set()
    for name, spec in expected_specs.items():
        run = run_by_name.get(name) or {}
        expected_junit_name = f"{name}.xml"
        expected_log_name = f"{name}.log"
        if run.get("junit_path") != expected_junit_name:
            errors.append(f"junit_path:{name}")
        if run.get("log_path") != expected_log_name:
            errors.append(f"log_path:{name}")
        if str(run.get("junit_path") or "") in observed_junit_paths:
            errors.append(f"duplicate_junit_path:{name}")
        if str(run.get("log_path") or "") in observed_log_paths:
            errors.append(f"duplicate_log_path:{name}")
        observed_junit_paths.add(str(run.get("junit_path") or ""))
        observed_log_paths.add(str(run.get("log_path") or ""))
        try:
            junit_path = _safe_receipt_child(receipt_root, run.get("junit_path"))
            log_path = _safe_receipt_child(receipt_root, run.get("log_path"))
        except ValueError:
            errors.append(f"run_path:{name}")
            continue
        expected_template = _command_template(spec, f"{name}.xml")
        if run.get("argv_template") != expected_template:
            errors.append(f"command:{name}")
        expected_argv = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *spec.paths,
            f"--junitxml={receipt_root / f'{name}.xml'}",
        ]
        if run.get("executed_argv") != expected_argv:
            errors.append(f"executed_command:{name}")
        if run.get("working_directory") != str(repo_root):
            errors.append(f"working_directory:{name}")
        if run.get("required_green") is not True:
            errors.append(f"required_green:{name}")
        if run.get("expected_outcome") != spec.expected_outcome:
            errors.append(f"expected_outcome:{name}")
        if not junit_path.is_file() or run.get("junit_sha256") != _sha256_file(
            junit_path
        ):
            errors.append(f"junit_hash:{name}")
            junit = {}
        else:
            try:
                junit = _parse_junit(junit_path)
            except Exception as error:
                junit = {}
                errors.append(f"junit_parse:{name}:{type(error).__name__}")
        if run.get("junit") != junit:
            errors.append(f"junit_recompute:{name}")
        if not log_path.is_file() or run.get("log_sha256") != _sha256_file(log_path):
            errors.append(f"log_hash:{name}")
        if spec.expected_outcome == "pass":
            recomputed_accepted = bool(
                run.get("exit_code") == 0
                and (junit or {}).get("tests", 0) > 0
                and (junit or {}).get("failures") == 0
                and (junit or {}).get("errors") == 0
                and (junit or {}).get("skipped") == 0
            )
            if run.get("accepted") is not recomputed_accepted or not recomputed_accepted:
                errors.append(f"required_test_failure:{name}")
            if name == "frozen_composite_lock_regression" and (
                (junit or {}).get("tests") != 19
                or (junit or {}).get("failures") != 0
                or (junit or {}).get("errors") != 0
                or (junit or {}).get("skipped") != 0
            ):
                errors.append("frozen_exception_resolution_drift")
        else:
            errors.append(f"unsupported_non_green_probe:{name}")

    full_suite_ids = set(
        (run_by_name.get("full_suite_r2") or {}).get("junit", {}).get(
            "testcase_ids", []
        )
    )
    if len(full_suite_ids) < 735:
        errors.append("full_suite_floor")
    for name in expected_green:
        if name == "full_suite_r2":
            continue
        scoped_ids = set(
            (run_by_name.get(name) or {}).get("junit", {}).get("testcase_ids", [])
        )
        if not scoped_ids or not scoped_ids <= full_suite_ids:
            errors.append(f"full_suite_scope_coverage:{name}")

    frozen = receipt.get("historical_frozen_exception_resolution") or {}
    if (
        frozen.get("current_module_required_green") is not True
        or frozen.get("excluded_from_full_suite") is not False
        or frozen.get("run_name") != "frozen_composite_lock_regression"
    ):
        errors.append("frozen_exception_resolution_boundary")
    for relative in (FROZEN_LOCK_PATH, FROZEN_DETECTOR_PATH):
        expected = (frozen.get("lock_and_detector_sha256") or {}).get(relative)
        if expected != _sha256_file(repo_root / relative):
            errors.append(f"frozen_binding:{relative}")
    superseded = receipt.get("superseded_chain") or {}
    artifacts = superseded.get("artifacts") or []
    if superseded.get("superseded_failure_not_used_for_gate") is not True:
        errors.append("superseded_boundary")
    if superseded.get("relations") != [
        "tier2_formal_and_new.xml superseded_by tier2_formal_and_new_r2.xml",
        "baseline_section_20_1_r2.xml superseded_by baseline_section_20_1_r3.xml",
    ]:
        errors.append("superseded_relations")
    expected_superseded = {
        str(specification["path"]): specification
        for specification in SUPERSEDED_JUNIT_SPECS
    }
    if (
        {str(row.get("path") or "") for row in artifacts}
        != set(expected_superseded)
        or len(artifacts) != len(expected_superseded)
    ):
        errors.append("superseded_chain")
    for row in artifacts:
        specification = expected_superseded.get(str(row.get("path") or ""))
        if specification is None:
            continue
        if row.get("disposition") != specification["disposition"]:
            errors.append(f"superseded_disposition:{row.get('path')}")
        try:
            path = _safe_receipt_child(receipt_root, row.get("path"))
        except ValueError:
            errors.append(f"superseded_path:{row.get('path')}")
            continue
        if not path.is_file() or row.get("sha256") != _sha256_file(path):
            errors.append(f"superseded_hash:{row.get('path')}")
        else:
            observed = _parse_junit(path)
            if row.get("junit") != observed:
                errors.append(f"superseded_junit:{row.get('path')}")
            if observed.get("failing_tests") != list(
                specification["expected_failures"]
            ):
                errors.append(f"superseded_failure_identity:{row.get('path')}")
    if receipt.get("runner_source_sha256") != _sha256_file(Path(__file__).resolve()):
        errors.append("runner_source_hash")
    return {
        "verified": not errors,
        "errors": sorted(set(errors)),
        "receipt_hash": receipt.get("receipt_hash", ""),
        "manifest_hash": manifest.get("manifest_hash", ""),
        "source_inventory_sha256": snapshot.get("inventory_sha256", ""),
        "required_green_run_count": len(expected_green),
        "required_green_test_count": sum(
            int((run_by_name.get(name) or {}).get("junit", {}).get("tests", 0))
            for name in expected_green
        ),
        "full_suite_test_count": int(
            (run_by_name.get("full_suite_r2") or {}).get("junit", {}).get("tests", 0)
        ),
        "full_suite_testcase_ids_sha256": (
            (run_by_name.get("full_suite_r2") or {})
            .get("junit", {})
            .get("testcase_ids_sha256", "")
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = build_test_receipt(
        repo_root=args.repo_root,
        output_root=args.output_root,
        created_at=args.created_at,
    )
    print(_json_text(receipt), end="")
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()


__all__ = [
    "FROZEN_EXCEPTION_MODULE",
    "FROZEN_LOCK_REGRESSION_TEST",
    "MANIFEST_SCHEMA",
    "RECEIPT_SCHEMA",
    "build_test_receipt",
    "source_inventory",
    "source_snapshot",
    "test_specs",
    "verify_test_receipt",
]
