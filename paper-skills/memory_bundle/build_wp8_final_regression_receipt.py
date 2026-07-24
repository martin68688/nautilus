"""Build the immutable WP8 final-regression receipt from saved JUnit evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable, Mapping


RECEIPT_SCHEMA = "decision_admissibility_wp8_final_regression_receipt_v1"
NEGATIVE_JUNIT = "tier2_formal_and_new.xml"
FIXED_JUNIT = "tier2_formal_and_new_r2.xml"
UPGRADE_RECEIPT = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "historical_verifier_upgrade_20260723_r1.json"
)
UPGRADE_SIDECAR = UPGRADE_RECEIPT.replace(".json", ".sha256")
EXPECTED_HISTORICAL_FAILURES = (
    "tests.test_tier2_formal_continuation_amendment::"
    "test_r7_continuation_amendment_verifies",
    "tests.test_tier2_formal_preterminal_recovery_amendment::"
    "test_r6_preterminal_recovery_amendment_verifies",
)
ADDITIONAL_REPAIRED_JUNITS = {
    "baseline_section_20_1_r2.xml": {
        "fixed_junit": "baseline_section_20_1_r3.xml",
        "expected_failed_testcase_ids": (
            "tests.authority.test_replay_authority_recovery::"
            "test_clean_replay_recovers_only_new_claim_and_publishes_scoped_bundle",
        ),
        "root_cause": (
            "The new semantic-provenance loader trigger treated a synthetic "
            "Clean Replay bundle containing only corpus/manifest.json as a "
            "modern corpus_manifest_v1 bundle."
        ),
        "fix": (
            "Require semantic provenance for corpus_manifest_v1 or explicit "
            "split/audit markers; keep resigned source/heldout overlap denied."
        ),
    }
}
REQUIRED_PASSING_SCOPES = {
    "baseline_section_20_1.xml": 408,
    "result_adoption_causal_section_20_1_A.xml": 45,
    "new_unit_section_20_2.xml": 68,
    "integration_section_20_3.xml": 57,
    FIXED_JUNIT: 92,
}
SOURCE_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".sh"}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def payload_hash(payload: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _sidecar_matches(path: Path) -> bool:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        return False
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    return bool(
        len(fields) >= 2
        and fields[0] == sha256_file(path)
        and fields[-1] == path.name
    )


def _testcase_id(node: ET.Element) -> str:
    classname = str(node.attrib.get("classname") or "")
    name = str(node.attrib.get("name") or "")
    if not classname or not name:
        raise ValueError("JUnit testcase is missing classname or name")
    return f"{classname}::{name}"


def parse_junit(path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    raw_path = Path(path)
    if raw_path.is_symlink():
        raise ValueError(f"JUnit symlink is forbidden: {raw_path}")
    path = raw_path.resolve()
    repo_root = Path(repo_root).resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JUnit must be a regular file: {path}")
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
    ids = [_testcase_id(case) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"JUnit contains duplicate testcase identities: {path}")
    failures = sorted(
        _testcase_id(case)
        for case in cases
        if case.find("failure") is not None
    )
    errors = sorted(
        _testcase_id(case) for case in cases if case.find("error") is not None
    )
    skipped = sorted(
        _testcase_id(case) for case in cases if case.find("skipped") is not None
    )
    declared = {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    observed = {
        "tests": len(cases),
        "failures": len(failures),
        "errors": len(errors),
        "skipped": len(skipped),
    }
    if declared != observed:
        raise ValueError(
            f"JUnit declared/observed count mismatch: {path}: {declared} != {observed}"
        )
    return {
        "path": (
            str(path.relative_to(repo_root))
            if path.is_relative_to(repo_root)
            else str(path)
        ),
        "file_sha256": sha256_file(path),
        **observed,
        "duration_seconds": sum(
            float(suite.attrib.get("time", 0.0) or 0.0) for suite in suites
        ),
        "testcase_ids_hash": hashlib.sha256(canonical_bytes(sorted(ids))).hexdigest(),
        "failed_testcase_ids": failures,
        "error_testcase_ids": errors,
        "skipped_testcase_ids": skipped,
        "_testcase_ids": sorted(ids),
    }


def _iter_source_files(repo_root: Path) -> Iterable[tuple[str, Path]]:
    roots = (
        ("mlevolve", repo_root / "mlevolve"),
        ("memory_bundle", repo_root / "paper-skills" / "memory_bundle"),
        ("tests", repo_root / "tests"),
    )
    for label, root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.suffix not in SOURCE_SUFFIXES:
                continue
            relative = path.relative_to(repo_root)
            if relative.parts[:2] == ("mlevolve", "runs"):
                continue
            yield label, path
    deploy = repo_root / "deploy"
    for path in sorted(deploy.glob("*decision_admissibility*")):
        if path.is_file() and not path.is_symlink() and path.suffix in SOURCE_SUFFIXES:
            yield "deploy_decision_admissibility", path
    plan = repo_root / "coordination" / (
        "decision_admissibility_complete_execution_plan_20260719.md"
    )
    if plan.is_file() and not plan.is_symlink():
        yield "execution_plan", plan


def source_inventory(repo_root: str | Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    entries: list[dict[str, str]] = []
    by_root: dict[str, list[dict[str, str]]] = {}
    for label, path in _iter_source_files(repo_root):
        row = {
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(path),
        }
        entries.append(row)
        by_root.setdefault(label, []).append(row)
    if not entries:
        raise ValueError("Final source inventory is empty")
    return {
        "algorithm": "sorted_relative_path_and_byte_sha256_canonical_json_v1",
        "file_count": len(entries),
        "inventory_hash": hashlib.sha256(canonical_bytes(entries)).hexdigest(),
        "root_summaries": {
            label: {
                "file_count": len(rows),
                "inventory_hash": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
            }
            for label, rows in sorted(by_root.items())
        },
    }


def _git_value(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def compute_receipt(
    *,
    repo_root: str | Path,
    test_root: str | Path,
    final_suite_filename: str,
    created_at: str,
    branch: str | None = None,
    head: str | None = None,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    raw_test_root = Path(test_root)
    if raw_test_root.is_symlink():
        raise ValueError(f"Final-test root symlink is forbidden: {raw_test_root}")
    test_root = raw_test_root.resolve()
    if test_root.is_symlink() or not test_root.is_dir():
        raise ValueError(f"Invalid final-test root: {test_root}")
    if Path(final_suite_filename).name != final_suite_filename:
        raise ValueError("final_suite_filename must be a basename")
    junit_paths = sorted(test_root.glob("*.xml"))
    if not junit_paths:
        raise ValueError("Final-test root contains no JUnit XML")
    runs: dict[str, dict[str, Any]] = {
        path.name: parse_junit(path, repo_root=repo_root) for path in junit_paths
    }
    required_names = {
        *REQUIRED_PASSING_SCOPES,
        NEGATIVE_JUNIT,
        final_suite_filename,
        *ADDITIONAL_REPAIRED_JUNITS,
        *(
            str(value["fixed_junit"])
            for value in ADDITIONAL_REPAIRED_JUNITS.values()
        ),
    }
    missing = sorted(required_names - set(runs))
    if missing:
        raise ValueError(f"Missing required JUnit scopes: {missing}")
    negative = runs[NEGATIVE_JUNIT]
    fixed = runs[FIXED_JUNIT]
    if tuple(negative["failed_testcase_ids"]) != EXPECTED_HISTORICAL_FAILURES:
        raise ValueError("Historical JUnit failure identity drift")
    if negative["failures"] != 2 or negative["errors"] or negative["skipped"]:
        raise ValueError("Historical JUnit disposition drift")
    negative_ids = set(negative["_testcase_ids"])
    fixed_ids = set(fixed["_testcase_ids"])
    if not negative_ids <= fixed_ids:
        raise ValueError("Revision-aware fixed JUnit dropped historical tests")
    if not set(EXPECTED_HISTORICAL_FAILURES) <= fixed_ids:
        raise ValueError("Fixed JUnit does not rerun both historical failures")

    closeout_repairs = []
    for failed_name, specification in sorted(ADDITIONAL_REPAIRED_JUNITS.items()):
        failed_run = runs[failed_name]
        fixed_name = str(specification["fixed_junit"])
        fixed_run = runs[fixed_name]
        expected_failures = tuple(specification["expected_failed_testcase_ids"])
        if tuple(failed_run["failed_testcase_ids"]) != expected_failures:
            raise ValueError(f"Closeout failure identity drift: {failed_name}")
        if (
            failed_run["failures"] != len(expected_failures)
            or failed_run["errors"]
            or failed_run["skipped"]
        ):
            raise ValueError(f"Closeout failed JUnit disposition drift: {failed_name}")
        if fixed_run["failures"] or fixed_run["errors"] or fixed_run["skipped"]:
            raise ValueError(f"Closeout fixed JUnit is not clean: {fixed_name}")
        if not set(failed_run["_testcase_ids"]) <= set(fixed_run["_testcase_ids"]):
            raise ValueError(f"Closeout fix dropped tests: {failed_name}")
        closeout_repairs.append(
            {
                "failed_junit": failed_name,
                "fixed_junit": fixed_name,
                "failed_testcase_ids": list(expected_failures),
                "failed_junit_preserved": True,
                "root_cause": specification["root_cause"],
                "fix": specification["fix"],
            }
        )

    for name, minimum in REQUIRED_PASSING_SCOPES.items():
        run = runs[name]
        if run["tests"] < minimum:
            raise ValueError(f"JUnit scope shrank below floor: {name}")
        if run["failures"] or run["errors"] or run["skipped"]:
            raise ValueError(f"Required JUnit scope is not clean: {name}")
    final_suite = runs[final_suite_filename]
    if final_suite["tests"] < 735:
        raise ValueError("Final full suite is below the pre-closeout 735-test floor")
    if final_suite["failures"] or final_suite["errors"] or final_suite["skipped"]:
        raise ValueError("Final full suite is not clean")
    final_ids = set(final_suite["_testcase_ids"])
    for name in REQUIRED_PASSING_SCOPES:
        if not set(runs[name]["_testcase_ids"]) <= final_ids:
            raise ValueError(f"Final full suite does not cover scope: {name}")
    allowed_negative_names = {NEGATIVE_JUNIT, *ADDITIONAL_REPAIRED_JUNITS}
    for name, run in runs.items():
        if name in allowed_negative_names:
            continue
        if run["failures"] or run["errors"] or run["skipped"]:
            raise ValueError(f"Unexpected non-clean JUnit: {name}")

    upgrade_path = repo_root / UPGRADE_RECEIPT
    upgrade = _read_object(upgrade_path)
    if (
        upgrade.get("schema")
        != "decision_admissibility_wp8_tier2_formal_historical_verifier_upgrade_v1"
        or upgrade.get("status") != "revision_aware_verifier_upgrade_frozen"
        or upgrade.get("receipt_hash") != payload_hash(upgrade, "receipt_hash")
        or not _sidecar_matches(upgrade_path)
        or upgrade.get("targeted_regression", {}).get("failed") != 0
    ):
        raise ValueError("Historical verifier upgrade receipt is invalid")
    helper = upgrade.get("revision_chain_helper") or {}
    helper_path = repo_root / str(helper.get("path") or "")
    if not helper_path.is_file() or helper.get("file_sha256") != sha256_file(helper_path):
        raise ValueError("Revision-chain helper binding mismatch")

    public_runs = []
    for name in sorted(runs):
        public_runs.append(
            {key: value for key, value in runs[name].items() if key != "_testcase_ids"}
        )
    branch = branch if branch is not None else _git_value(repo_root, "branch", "--show-current")
    head = head if head is not None else _git_value(repo_root, "rev-parse", "HEAD")
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed_with_preserved_historical_failure",
        "created_at": str(created_at),
        "branch": str(branch),
        "head": str(head),
        "dirty_worktree_preserved": True,
        "test_root": (
            str(test_root.relative_to(repo_root))
            if test_root.is_relative_to(repo_root)
            else str(test_root)
        ),
        "final_suite_filename": final_suite_filename,
        "junit_runs": public_runs,
        "historical_failure_repair": {
            "initial_junit": NEGATIVE_JUNIT,
            "fixed_junit": FIXED_JUNIT,
            "initial_failed_testcase_ids": list(EXPECTED_HISTORICAL_FAILURES),
            "initial_failure_preserved": True,
            "historical_artifacts_mutated": False,
            "fix_class": "strict_revision_aware_verifier_chain",
            "upgrade_receipt": {
                "path": UPGRADE_RECEIPT,
                "file_sha256": sha256_file(upgrade_path),
                "receipt_hash": upgrade["receipt_hash"],
            },
            "upgrade_sidecar": {
                "path": UPGRADE_SIDECAR,
                "file_sha256": sha256_file(repo_root / UPGRADE_SIDECAR),
            },
            "closeout_regressions": closeout_repairs,
        },
        "source_inventory": source_inventory(repo_root),
        "all_required_passing_scopes_clean": True,
        "final_full_suite_clean": True,
        "unexpected_failure_count": 0,
        "final_regression_passed": True,
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = payload_hash(receipt, "receipt_hash")
    return receipt


def build_receipt(*, output_path: str | Path, **kwargs: Any) -> dict[str, Any]:
    raw_output_path = Path(output_path)
    if raw_output_path.is_symlink():
        raise FileExistsError(
            f"Refusing symlink final regression output: {raw_output_path}"
        )
    output_path = raw_output_path.resolve()
    sidecar = output_path.with_suffix(".sha256")
    if output_path.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to reuse final regression output: {output_path}")
    receipt = compute_receipt(**kwargs)
    _write_text_exclusive(
        output_path,
        json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )
    _write_text_exclusive(sidecar, f"{sha256_file(output_path)}  {output_path.name}\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--final-suite", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = build_receipt(
        output_path=args.output,
        repo_root=args.repo_root,
        test_root=args.test_root,
        final_suite_filename=args.final_suite,
        created_at=args.created_at,
    )
    print(json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
