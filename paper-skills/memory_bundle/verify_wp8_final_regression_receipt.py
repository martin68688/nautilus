"""Independently verify and recompute the WP8 final-regression receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from build_wp8_final_regression_receipt import (
    RECEIPT_SCHEMA,
    compute_receipt,
    payload_hash,
    sha256_file,
)


VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_final_regression_receipt_verification_v1"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_receipt(
    *,
    receipt_path: str | Path,
    repo_root: str | Path,
    test_root: str | Path,
) -> dict[str, Any]:
    raw_receipt_path = Path(receipt_path)
    receipt_path = raw_receipt_path.resolve()
    repo_root = Path(repo_root).resolve()
    errors: list[str] = []
    if raw_receipt_path.is_symlink():
        errors.append("receipt_symlink")
    if receipt_path.is_file() and receipt_path.stat().st_mode & 0o222:
        errors.append("receipt_writable")
    try:
        receipt = _read_object(receipt_path)
    except Exception as error:
        receipt = {}
        errors.append(f"receipt_read:{type(error).__name__}")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        errors.append("receipt_schema")
    if receipt.get("status") != "passed_with_preserved_historical_failure":
        errors.append("receipt_status")
    if receipt.get("receipt_hash") != payload_hash(receipt, "receipt_hash"):
        errors.append("receipt_hash")
    if receipt.get("final_regression_passed") is not True:
        errors.append("regression_gate")
    if receipt.get("unexpected_failure_count") != 0:
        errors.append("unexpected_failures")
    if receipt.get("final_full_suite_clean") is not True:
        errors.append("full_suite_gate")
    repair = receipt.get("historical_failure_repair") or {}
    if (
        repair.get("initial_failure_preserved") is not True
        or repair.get("historical_artifacts_mutated") is not False
    ):
        errors.append("historical_failure_boundary")
    sidecar = receipt_path.with_suffix(".sha256")
    if sidecar.is_symlink():
        errors.append("receipt_sidecar_symlink")
    if sidecar.is_file() and sidecar.stat().st_mode & 0o222:
        errors.append("receipt_sidecar_writable")
    try:
        fields = sidecar.read_text(encoding="utf-8").strip().split()
        if not (
            len(fields) >= 2
            and fields[0] == sha256_file(receipt_path)
            and fields[-1] == receipt_path.name
        ):
            errors.append("receipt_sidecar")
    except OSError as error:
        errors.append(f"receipt_sidecar:{type(error).__name__}")
    try:
        expected = compute_receipt(
            repo_root=repo_root,
            test_root=test_root,
            final_suite_filename=str(receipt.get("final_suite_filename") or ""),
            created_at=str(receipt.get("created_at") or ""),
            branch=str(receipt.get("branch") or ""),
            head=str(receipt.get("head") or ""),
        )
    except Exception as error:
        expected = None
        errors.append(f"receipt_recompute:{type(error).__name__}")
    if expected is not None and expected != receipt:
        errors.append("receipt_recompute_mismatch")
    try:
        receipt_display_path = str(receipt_path.relative_to(repo_root))
    except ValueError:
        receipt_display_path = str(receipt_path)
    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed" if not errors else "failed",
        "verified": not errors,
        "errors": sorted(set(errors)),
        "receipt_path": receipt_display_path,
        "receipt_file_sha256": sha256_file(receipt_path) if receipt_path.is_file() else "",
        "receipt_hash": receipt.get("receipt_hash", ""),
        "final_suite_filename": receipt.get("final_suite_filename", ""),
        "final_suite_test_count": next(
            (
                row.get("tests", 0)
                for row in receipt.get("junit_runs") or []
                if Path(str(row.get("path") or "")).name
                == receipt.get("final_suite_filename")
            ),
            0,
        ),
        "source_inventory_hash": (receipt.get("source_inventory") or {}).get(
            "inventory_hash", ""
        ),
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    verification["verification_hash"] = payload_hash(
        verification, "verification_hash"
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_receipt(
        receipt_path=args.receipt,
        repo_root=args.repo_root,
        test_root=args.test_root,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
