from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from run_wp8_final_tests import (  # noqa: E402
    FROZEN_DETECTOR_PATH,
    FROZEN_LOCK_PATH,
    MANIFEST_SCHEMA,
    RECEIPT_SCHEMA,
    SUPERSEDED_JUNIT_SPECS,
    _command_template,
    _environment_receipt,
    _git_value,
    _inventory_bytes,
    _parse_junit,
    _payload_hash,
    _sha256_file,
    build_test_receipt,
    source_snapshot,
    test_specs as _test_specs,
    verify_test_receipt,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _minimal_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for directory in (
        "mlevolve",
        "paper-skills/memory_bundle",
        "paper-skills/distillation",
        "paper-skills/hyper_memory",
        "paper-skills/eval_composite_memory/manifests",
        "paper-skills/eval_skill_memory",
        "tests",
        "deploy",
        "coordination",
        "papers/runforest_iclr2025/evidence",
    ):
        (repo / directory).mkdir(parents=True, exist_ok=True)
    files = {
        "mlevolve/module.py": "VALUE = 1\n",
        FROZEN_DETECTOR_PATH: "DETECTOR = 1\n",
        FROZEN_LOCK_PATH: "{}\n",
        (
            "paper-skills/eval_composite_memory/manifests/"
            "replay_heldout_detector_provenance_addendum_v1.json"
        ): "{}\n",
        "paper-skills/eval_composite_memory/manifests/claim_gates_v1.yaml": "{}\n",
        "paper-skills/eval_composite_memory/manifests/condition_manifest_v1.yaml": "{}\n",
        "paper-skills/eval_composite_memory/manifests/task_manifest_v1.yaml": "{}\n",
        "paper-skills/memory_bundle/helper.py": "VALUE = 1\n",
        "paper-skills/distillation/helper.py": "VALUE = 1\n",
        "paper-skills/hyper_memory/helper.py": "VALUE = 1\n",
        "paper-skills/hyper_memory/sop_taxonomy.json": "{}\n",
        "paper-skills/hyper_memory/sop_taxonomy_overrides.json": "{}\n",
        "paper-skills/eval_composite_memory/helper.py": "VALUE = 1\n",
        "paper-skills/eval_composite_memory/manifests/fixture.json": "{}\n",
        "paper-skills/eval_skill_memory/helper.py": "VALUE = 1\n",
        "paper-skills/eval_skill_memory/clean_replay_targets.json": "{}\n",
        "paper-skills/eval_skill_memory/clean_run_allowlist.json": "{}\n",
        "paper-skills/eval_skill_memory/non_spooky_replay_source_manifest_v1.json": "{}\n",
        "paper-skills/eval_skill_memory/requirements-decision-point-benchmark.txt": "requirements\n",
        "paper-skills/eval_skill_memory/run_identity_registry_v1.json": "{}\n",
        "paper-skills/eval_skill_memory/fixture.jsonl": "{}\n",
        "paper-skills/hyper_memory/fixture.npz": "fixture\n",
        "tests/test_wp8_evidence_ledger.py": "def test_x(): pass\n",
        "tests/test_wp8_final_stop_gate.py": "def test_y(): pass\n",
        "deploy/run_decision_admissibility_example.sh": "#!/bin/sh\n",
        "coordination/decision_admissibility_complete_execution_plan_20260719.md": "plan\n",
        "papers/runforest_iclr2025/evidence/claims.md": "claims\n",
    }
    for relative, text in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (repo / ".venv").symlink_to(Path(sys.prefix), target_is_directory=True)
    return repo


def _junit(
    *,
    tests: int,
    failed_ids: tuple[str, ...] = (),
    testcase_ids: tuple[str, ...] = (),
) -> str:
    if len(failed_ids) > tests or len(testcase_ids) > tests:
        raise ValueError("More failures than tests")
    explicit_ids = list(testcase_ids)
    for failed_id in failed_ids:
        if failed_id not in explicit_ids:
            explicit_ids.append(failed_id)
    cases = ""
    for index in range(tests):
        if index < len(explicit_ids):
            testcase_id = explicit_ids[index]
            classname, name = testcase_id.split("::", 1)
        else:
            testcase_id = f"tests.sample::test_{index}"
            classname, name = testcase_id.split("::", 1)
        if testcase_id in failed_ids:
            failure = "<failure message='preserved'>preserved</failure>"
        else:
            failure = ""
        cases += (
            f"<testcase classname='{classname}' name='{name}'>{failure}</testcase>"
        )
    return (
        "<?xml version='1.0' encoding='utf-8'?>"
        f"<testsuites><testsuite name='pytest' tests='{tests}' "
        f"failures='{len(failed_ids)}' "
        f"errors='0' skipped='0' time='0'>{cases}</testsuite></testsuites>"
    )


def _fake_receipt(
    repo: Path,
    output: Path,
    *,
    additional_full_ids: tuple[str, ...] = (),
) -> None:
    repo = repo.resolve()
    output = output.resolve()
    output.mkdir()
    snapshot = source_snapshot(repo)
    inventory = output / "source_inventory.jsonl"
    inventory.write_bytes(_inventory_bytes(snapshot["rows"]))
    specs = _test_specs(repo)
    scoped_ids = {
        spec.name: (
            tuple(f"tests.frozen::test_{index}" for index in range(19))
            if spec.name == "frozen_composite_lock_regression"
            else (f"tests.{spec.name}::test_scope",)
        )
        for spec in specs
        if spec.name != "full_suite_r2"
    }
    full_ids = tuple(
        sorted(
            {
                testcase_id
                for ids in scoped_ids.values()
                for testcase_id in ids
            }
            | set(additional_full_ids)
        )
    )
    runs = []
    for spec in specs:
        if spec.name == "full_suite_r2":
            tests = 735
            testcase_ids = full_ids
        else:
            testcase_ids = scoped_ids[spec.name]
            tests = len(testcase_ids)
        junit = output / f"{spec.name}.xml"
        log = output / f"{spec.name}.log"
        junit.write_text(
            _junit(tests=tests, testcase_ids=testcase_ids), encoding="utf-8"
        )
        log.write_text("host-owned test execution\n", encoding="utf-8")
        runs.append(
            {
                "name": spec.name,
                "required_green": True,
                "expected_outcome": "pass",
                "accepted": True,
                "argv_template": _command_template(spec, junit.name),
                "executed_argv": [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    *spec.paths,
                    f"--junitxml={junit}",
                ],
                "working_directory": str(repo),
                "exit_code": 0,
                "started_unix_ns": 1,
                "ended_unix_ns": 2,
                "duration_seconds": 1e-9,
                "junit_path": junit.name,
                "junit_sha256": _sha256_file(junit),
                "junit": _parse_junit(junit),
                "log_path": log.name,
                "log_sha256": _sha256_file(log),
            }
        )
    superseded = []
    for specification in SUPERSEDED_JUNIT_SPECS:
        name = str(specification["path"])
        failed_ids = tuple(specification["expected_failures"])
        disposition = str(specification["disposition"])
        path = output / name
        path.write_text(
            _junit(tests=92, failed_ids=failed_ids), encoding="utf-8"
        )
        superseded.append(
            {
                "path": name,
                "sha256": _sha256_file(path),
                "disposition": disposition,
                "junit": _parse_junit(path),
            }
        )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed",
        "repo_identity": {
            "branch": _git_value(repo, "branch", "--show-current"),
            "head": _git_value(repo, "rev-parse", "HEAD"),
            "dirty_worktree_expected": True,
        },
        "environment": _environment_receipt(repo),
        "source_snapshot": {
            key: value for key, value in snapshot.items() if key != "rows"
        },
        "source_inventory_path": inventory.name,
        "source_inventory_file_sha256": _sha256_file(inventory),
        "source_unchanged_during_tests": True,
        "test_runs": runs,
        "required_green_run_names": [row["name"] for row in runs],
        "full_suite_run_name": "full_suite_r2",
        "historical_frozen_exception_resolution": {
            "run_name": "frozen_composite_lock_regression",
            "current_module_required_green": True,
            "excluded_from_full_suite": False,
            "lock_and_detector_sha256": {
                relative: _sha256_file(repo / relative)
                for relative in (FROZEN_LOCK_PATH, FROZEN_DETECTOR_PATH)
            },
        },
        "superseded_chain": {
            "relations": [
                "tier2_formal_and_new.xml superseded_by tier2_formal_and_new_r2.xml",
                "baseline_section_20_1_r2.xml superseded_by baseline_section_20_1_r3.xml",
            ],
            "superseded_failure_not_used_for_gate": True,
            "artifacts": superseded,
        },
        "runner_source_sha256": _sha256_file(
            TOOLS / "run_wp8_final_tests.py"
        ),
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = _payload_hash(receipt, "receipt_hash")
    _write_json(output / "test_receipt.json", receipt)
    files = {
        path.relative_to(output).as_posix(): _sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "passed",
        "receipt_hash": receipt["receipt_hash"],
        "files": files,
        "runner_source_sha256": receipt["runner_source_sha256"],
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    _write_json(output / "manifest.json", manifest)
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    output.chmod(0o555)


def test_host_receipt_binds_commands_junit_exit_and_current_source(
    tmp_path: Path,
) -> None:
    repo = _minimal_repo(tmp_path)
    output = tmp_path / "receipt"
    _fake_receipt(repo, output)
    result = verify_test_receipt(receipt_root=output, repo_root=repo)
    assert result["verified"] is True
    assert result["full_suite_test_count"] == 735

    (repo / "mlevolve/module.py").write_text("VALUE = 2\n", encoding="utf-8")
    stale = verify_test_receipt(receipt_root=output, repo_root=repo)
    assert stale["verified"] is False
    assert "stale_source_snapshot" in stale["errors"]


def test_host_receipt_rejects_unbound_file_and_output_reuse(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    output = tmp_path / "receipt"
    _fake_receipt(repo, output)
    output.chmod(0o755)
    (output / "unbound.txt").write_text("extra\n", encoding="utf-8")
    output.chmod(0o555)
    result = verify_test_receipt(receipt_root=output, repo_root=repo)
    assert result["verified"] is False
    assert "manifest_file_set" in result["errors"]

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        build_test_receipt(
            repo_root=repo,
            output_root=existing,
            created_at="2026-07-24T00:00:00Z",
        )


def test_host_receipt_rejects_self_rehashed_executed_command_laundering(
    tmp_path: Path,
) -> None:
    repo = _minimal_repo(tmp_path)
    output = tmp_path / "receipt"
    _fake_receipt(repo, output)
    receipt_path = output / "test_receipt.json"
    manifest_path = output / "manifest.json"
    receipt_path.chmod(0o644)
    manifest_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["test_runs"][0]["executed_argv"] = ["python", "forged.py"]
    receipt["receipt_hash"] = _payload_hash(receipt, "receipt_hash")
    _write_json(receipt_path, receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["receipt_hash"] = receipt["receipt_hash"]
    manifest["files"]["test_receipt.json"] = _sha256_file(receipt_path)
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    _write_json(manifest_path, manifest)
    receipt_path.chmod(0o444)
    manifest_path.chmod(0o444)

    result = verify_test_receipt(receipt_root=output, repo_root=repo)
    assert result["verified"] is False
    assert "executed_command:baseline_section_20_1" in result["errors"]


def test_host_receipt_rejects_self_rehashed_skipped_scope(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    output = tmp_path / "receipt"
    _fake_receipt(repo, output)
    receipt_path = output / "test_receipt.json"
    manifest_path = output / "manifest.json"
    junit_path = output / "baseline_section_20_1.xml"
    for path in (receipt_path, manifest_path, junit_path):
        path.chmod(0o644)
    junit_path.write_text(
        "<?xml version='1.0' encoding='utf-8'?>"
        "<testsuites><testsuite name='pytest' tests='1' failures='0' "
        "errors='0' skipped='1' time='0'>"
        "<testcase classname='tests.baseline_section_20_1' "
        "name='test_scope'><skipped/></testcase>"
        "</testsuite></testsuites>",
        encoding="utf-8",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    run = next(
        row
        for row in receipt["test_runs"]
        if row["name"] == "baseline_section_20_1"
    )
    run["junit_sha256"] = _sha256_file(junit_path)
    run["junit"] = _parse_junit(junit_path)
    run["accepted"] = True
    receipt["receipt_hash"] = _payload_hash(receipt, "receipt_hash")
    _write_json(receipt_path, receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["receipt_hash"] = receipt["receipt_hash"]
    manifest["files"]["baseline_section_20_1.xml"] = _sha256_file(junit_path)
    manifest["files"]["test_receipt.json"] = _sha256_file(receipt_path)
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    _write_json(manifest_path, manifest)
    for path in (receipt_path, manifest_path, junit_path):
        path.chmod(0o444)

    result = verify_test_receipt(receipt_root=output, repo_root=repo)
    assert result["verified"] is False
    assert "required_test_failure:baseline_section_20_1" in result["errors"]


def test_source_snapshot_excludes_runtime_and_user_assets(tmp_path: Path) -> None:
    repo = _minimal_repo(tmp_path)
    for relative in (
        "mlevolve/runs/user/solution.py",
        "mlevolve/runs_backup/user/solution.py",
        "mlevolve/data/private.json",
        "mlevolve/inference/submissions/result.py",
        "paper-skills/distillation/distill_branch3_demo/final_patch.json",
        "paper-skills/distillation/graph_build/graph.json",
        "paper-skills/eval_composite_memory/artifacts/snapshot.json",
        "paper-skills/eval_composite_memory/reports/benchmark.json",
        "paper-skills/eval_skill_memory/artifacts/certified.json",
        "paper-skills/eval_skill_memory/reports/evaluation.json",
        "paper-skills/hyper_memory/run_forest_graph.json",
        "paper-skills/hyper_memory/run_forest_index.npz",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("USER_ASSET = True\n", encoding="utf-8")
    paths = {row["path"] for row in source_snapshot(repo)["rows"]}
    assert {
        "paper-skills/eval_composite_memory/manifests/replay_heldout_lock_v1.json",
        (
            "paper-skills/eval_composite_memory/manifests/"
            "replay_heldout_detector_provenance_addendum_v1.json"
        ),
        "paper-skills/eval_skill_memory/helper.py",
        "paper-skills/eval_skill_memory/clean_run_allowlist.json",
        "paper-skills/hyper_memory/helper.py",
        "paper-skills/hyper_memory/sop_taxonomy.json",
        "paper-skills/eval_composite_memory/manifests/fixture.json",
        "paper-skills/eval_composite_memory/artifacts/snapshot.json",
        "paper-skills/eval_composite_memory/reports/benchmark.json",
        "paper-skills/eval_skill_memory/fixture.jsonl",
        "paper-skills/eval_skill_memory/artifacts/certified.json",
        "paper-skills/eval_skill_memory/reports/evaluation.json",
        "paper-skills/hyper_memory/fixture.npz",
        "paper-skills/hyper_memory/run_forest_graph.json",
        "paper-skills/hyper_memory/run_forest_index.npz",
    } <= paths
    assert not any(
        path.startswith(
            (
                "mlevolve/runs/",
                "mlevolve/runs_backup/",
                "mlevolve/data/",
                "mlevolve/inference/submissions/",
                "paper-skills/distillation/distill_branch3_demo/",
                "paper-skills/distillation/graph_build/",
            )
        )
        for path in paths
    )
