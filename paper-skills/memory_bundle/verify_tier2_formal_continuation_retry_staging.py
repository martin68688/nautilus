#!/usr/bin/env python3
"""Verify fresh-root staging after the result-blind r13 control-packaging failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


from verify_tier2_formal_continuation_staging import (  # noqa: E402
    _seal,
    verify_continuation_staging,
)
from verify_tier2_formal_staging_retry_amendment import (  # noqa: E402
    verify_staging_retry_amendment,
)


SCHEMA = "decision_admissibility_wp8_tier2_formal_continuation_retry_stop_gate_v1"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def verify_continuation_retry_staging(
    staging_root: Path,
    *,
    repo_root: Path,
    retry_amendment_path: Path,
    retry_verification_path: Path,
    seal_on_success: bool = True,
) -> dict[str, Any]:
    staging_root = staging_root.resolve()
    repo_root = repo_root.resolve()
    retry_amendment_path = retry_amendment_path.resolve()
    retry_verification_path = retry_verification_path.resolve()

    base = verify_continuation_staging(
        staging_root,
        repo_root=repo_root,
        seal_on_success=False,
    )
    content = _read(staging_root / "STAGING_CONTENT_MANIFEST.json")
    amendment = _read(retry_amendment_path)
    frozen = _read(retry_verification_path)
    live = verify_staging_retry_amendment(
        retry_amendment_path,
        repo_root=repo_root,
    )
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: object) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(name)

    check(
        "base_gate_passed",
        base.get("status") == "passed"
        and base.get("formal_training_authorized") is True
        and base.get("errors") == []
        and base.get("gate_hash") == _payload_hash(base, "gate_hash"),
    )
    for name, passed in (base.get("checks") or {}).items():
        check(f"base:{name}", passed is True)

    check("retry_amendment_live_verified", live.get("verified") is True)
    check(
        "retry_amendment_live_hash",
        live.get("verification_hash") == _payload_hash(live, "verification_hash"),
    )
    check(
        "retry_verification_frozen",
        frozen.get("verified") is True
        and frozen.get("errors") == []
        and frozen.get("verification_hash")
        == _payload_hash(frozen, "verification_hash")
        == live.get("verification_hash")
        and frozen.get("amendment_file_sha256") == _file_sha256(retry_amendment_path),
    )

    retry = amendment.get("retry_overrides") or {}
    check(
        "staging_root_binding",
        content.get("staging_root", str(staging_root)) == str(staging_root),
    )
    check("staging_root_exact", str(staging_root) == retry.get("staging_root"))
    check("source_root_exact", content.get("source_root") == retry.get("source_root"))
    check("output_root_exact", content.get("output_root") == retry.get("output_root"))
    check(
        "source_identity_preserved",
        content.get("source_snapshot_sha256")
        == (amendment.get("control_packaging_correction") or {}).get(
            "required_source_sha256"
        ),
    )
    check(
        "execution_revision_preserved",
        content.get("formal_execution_revision")
        == retry.get("formal_execution_revision")
        == "r4",
    )
    check(
        "scientific_preregistration_preserved",
        content.get("effective_preregistration_id")
        == (amendment.get("parent_continuation") or {}).get("preregistration_id"),
    )
    check(
        "remaining_design_preserved",
        content.get("remaining_block_count") == 5
        and content.get("remaining_online_condition_count") == 25
        and content.get("remaining_oracle_count") == 5,
    )
    check(
        "still_result_blind",
        content.get("terminal_score_values_inspected") is False
        and content.get("terminal_metric_observed_for_remaining_blocks") is False,
    )

    correction = amendment.get("control_packaging_correction") or {}
    for relative in correction.get("added_control_paths") or []:
        check(f"added_control_present:{relative}", (repo_root / relative).is_file())
    implementation = amendment.get("implementation_files") or {}
    for relative, expected in implementation.items():
        source = repo_root / relative
        check(f"retry_control_exists:{relative}", source.is_file())
        if source.is_file():
            check(
                f"retry_control_hash:{relative}",
                _file_sha256(source) == expected,
            )

    combined_errors = sorted(set(errors))
    gate: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if not combined_errors else "failed",
        "formal_training_authorized": not combined_errors,
        "authorized_block_count": 5 if not combined_errors else 0,
        "authorized_online_condition_count": 25 if not combined_errors else 0,
        "authorized_oracle_count": 5 if not combined_errors else 0,
        "completed_blocks_authorized_to_rerun": False,
        "terminal_score_values_inspected": False,
        "terminal_metric_observed_for_remaining_blocks": False,
        "effect_claim_authorized": False,
        "staging_content_manifest_hash": content.get("manifest_hash", ""),
        "continuation_amendment_hash": base.get("continuation_amendment_hash", ""),
        "continuation_verification_hash": base.get(
            "continuation_verification_hash", ""
        ),
        "staging_retry_amendment_hash": amendment.get("amendment_hash", ""),
        "staging_retry_verification_hash": frozen.get("verification_hash", ""),
        "failed_attempt_diagnostic_hash": (amendment.get("failed_attempt") or {}).get(
            "diagnostic_hash", ""
        ),
        "completed_freeze_hash": base.get("completed_freeze_hash", ""),
        "source_snapshot_sha256": content.get("source_snapshot_sha256", ""),
        "parent_source_snapshot_sha256": content.get(
            "parent_source_snapshot_sha256", ""
        ),
        "source_diff_paths": list(base.get("source_diff_paths") or []),
        "underlying_continuation_gate_hash": base.get("gate_hash", ""),
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": combined_errors,
        "evidence": {
            "source_diff_paths": list(base.get("source_diff_paths") or []),
            "base_gate_hash": base.get("gate_hash", ""),
            "retry_amendment_file_sha256": _file_sha256(retry_amendment_path),
            "retry_verification_file_sha256": _file_sha256(retry_verification_path),
        },
        "verifier_source_sha256": _file_sha256(Path(__file__).resolve()),
        "gate_hash": "",
    }
    gate["gate_hash"] = _payload_hash(gate, "gate_hash")
    if not combined_errors and seal_on_success:
        _seal(staging_root)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--retry-amendment", type=Path, required=True)
    parser.add_argument("--retry-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-seal", action="store_true")
    args = parser.parse_args()
    gate = verify_continuation_retry_staging(
        args.staging_root,
        repo_root=args.repo_root,
        retry_amendment_path=args.retry_amendment,
        retry_verification_path=args.retry_verification,
        seal_on_success=not args.no_seal,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(0 if gate["formal_training_authorized"] else 1)


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "verify_continuation_retry_staging"]
