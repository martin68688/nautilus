"""Verify and independently recompute the WP8 Evidence Ledger."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from build_wp8_evidence_ledger import (
    LEDGER_SCHEMA,
    MANIFEST_SCHEMA,
    _json_text,
    _payload_hash,
    _sha256_file,
    compute_evidence_ledger,
    render_claims_addendum,
)


VERIFICATION_SCHEMA = "decision_admissibility_wp8_evidence_ledger_verification_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(_json_text(payload))
        handle.flush()
        os.fsync(handle.fileno())


def verify_evidence_ledger(
    *,
    ledger_root: str | Path,
    repo_root: str | Path,
    analysis_policy_path: str | Path,
    joint_inventory_root: str | Path,
    statistics_root: str | Path,
) -> dict[str, Any]:
    ledger_root = Path(ledger_root).resolve()
    ledger_path = ledger_root / "evidence_ledger.json"
    addendum_path = ledger_root / "claims_addendum.md"
    manifest_path = ledger_root / "manifest.json"
    errors: list[str] = []
    try:
        ledger = _read_object(ledger_path)
        manifest = _read_object(manifest_path)
        addendum = addendum_path.read_text(encoding="utf-8")
    except Exception as error:
        ledger = {}
        manifest = {}
        addendum = ""
        errors.append(f"ledger_read:{type(error).__name__}")
    if ledger.get("schema") != LEDGER_SCHEMA or ledger.get("status") != "complete":
        errors.append("ledger_schema_or_status")
    if ledger.get("ledger_hash") != _payload_hash(ledger, "ledger_hash"):
        errors.append("ledger_hash")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != "complete":
        errors.append("manifest_schema_or_status")
    if manifest.get("manifest_hash") != _payload_hash(manifest, "manifest_hash"):
        errors.append("manifest_hash")
    for filename, path in (
        ("evidence_ledger.json", ledger_path),
        ("claims_addendum.md", addendum_path),
    ):
        if not path.is_file() or (manifest.get("files") or {}).get(filename) != _sha256_file(path):
            errors.append(f"file_hash:{filename}")
    if manifest.get("ledger_hash") != ledger.get("ledger_hash"):
        errors.append("manifest_ledger_binding")
    builder_path = Path(__file__).resolve().with_name("build_wp8_evidence_ledger.py")
    builder_sha = _sha256_file(builder_path)
    if manifest.get("builder_source_sha256") != builder_sha:
        errors.append("builder_source_hash")
    claims = ledger.get("claims") or []
    required_ids = {
        "WP8-C1-FORMAL-EXECUTION",
        "WP8-C2-RESULT-WRITEBACK",
        "WP8-C3-FULL-SUPERIORITY",
        "WP8-C4-CONDITIONAL-UTILITY",
        "WP8-C5-NO-IMPUTATION",
        "WP8-C6-EXPERIENCE-CAUSALITY",
        "WP8-C7-PRIOR-KILL-GATES",
    }
    if {row.get("claim_id") for row in claims} != required_ids:
        errors.append("claim_ids")
    for row in claims:
        claim_id = str(row.get("claim_id") or "")
        if row.get("claim_hash") != _payload_hash(row, "claim_hash"):
            errors.append(f"claim_hash:{claim_id}")
        for field in (
            "group",
            "statement",
            "status",
            "condition",
            "sample_unit",
            "metrics",
            "artifact_bindings",
            "claim_gate",
            "interpretation",
        ):
            if field not in row:
                errors.append(f"claim_field:{claim_id}:{field}")
        for binding in row.get("artifact_bindings") or []:
            path = Path(repo_root).resolve() / str(binding.get("path") or "")
            if not path.is_file() or binding.get("file_sha256") != _sha256_file(path):
                errors.append(f"claim_artifact:{claim_id}:{binding.get('path')}")
    status_by_id = {row.get("claim_id"): row.get("status") for row in claims}
    if status_by_id.get("WP8-C3-FULL-SUPERIORITY") != "rejected":
        errors.append("superiority_status")
    if status_by_id.get("WP8-C4-CONDITIONAL-UTILITY") != "diagnostic":
        errors.append("conditional_status")
    if status_by_id.get("WP8-C6-EXPERIENCE-CAUSALITY") != "pending":
        errors.append("causal_status")
    if ledger.get("headline_effect_claim_authorized") is not False:
        errors.append("headline_claim_gate")
    paper_binding = ledger.get("paper_claims_binding") or {}
    paper_path = Path(repo_root).resolve() / str(paper_binding.get("path") or "")
    if not paper_path.is_file() or paper_binding.get("file_sha256") != _sha256_file(
        paper_path
    ):
        errors.append("paper_claims_binding")

    try:
        expected = compute_evidence_ledger(
            repo_root=repo_root,
            analysis_policy_path=analysis_policy_path,
            joint_inventory_root=joint_inventory_root,
            statistics_root=statistics_root,
            created_at=str(ledger.get("created_at") or ""),
        )
    except Exception as error:
        expected = None
        errors.append(f"ledger_recompute:{type(error).__name__}")
    if expected is not None:
        if expected != ledger:
            errors.append("ledger_recompute_mismatch")
        if render_claims_addendum(expected) != addendum:
            errors.append("addendum_recompute_mismatch")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed" if not errors else "failed",
        "verified": not errors,
        "errors": sorted(set(errors)),
        "claim_count": len(claims),
        "supported_claim_count": sum(row.get("status") == "supported" for row in claims),
        "diagnostic_claim_count": sum(row.get("status") == "diagnostic" for row in claims),
        "rejected_claim_count": sum(row.get("status") == "rejected" for row in claims),
        "pending_claim_count": sum(row.get("status") == "pending" for row in claims),
        "headline_effect_claim_authorized": ledger.get("headline_effect_claim_authorized"),
        "ledger_hash": ledger.get("ledger_hash", ""),
        "manifest_hash": manifest.get("manifest_hash", ""),
        "builder_source_sha256": builder_sha,
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    verification["verification_hash"] = _payload_hash(verification, "verification_hash")
    return verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--analysis-policy", required=True, type=Path)
    parser.add_argument("--joint-inventory-root", required=True, type=Path)
    parser.add_argument("--statistics-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = verify_evidence_ledger(
        ledger_root=args.ledger_root,
        repo_root=args.repo_root,
        analysis_policy_path=args.analysis_policy,
        joint_inventory_root=args.joint_inventory_root,
        statistics_root=args.statistics_root,
    )
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), result)
    print(_json_text(result), end="")
    if not result["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
