from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from verify_tier2_formal_preregistration import verify_preregistration


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "decision_admissibility_wp8_tier2_formal_preregistration_amendment_v1"
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_preregistration_amendment_verification_v1"
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def verify_amendment(
    amendment_path: str | Path,
    *,
    repo_root: str | Path = ROOT,
) -> dict[str, Any]:
    path = Path(amendment_path).resolve()
    repo_root = Path(repo_root).resolve()
    payload = _read(path)
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: Any) -> None:
        checks[name] = bool(condition)
        if not condition:
            errors.append(name)

    check("schema", payload.get("schema") == SCHEMA)
    check(
        "status_pending_staging",
        payload.get("status") == "design_frozen_pending_staging_hash_manifest",
    )
    check(
        "new_preregistration_id",
        payload.get("preregistration_id")
        == "wp8-tier2-formal-3protocol-6system-r2",
    )
    check(
        "amendment_hash",
        payload.get("amendment_hash")
        == _payload_hash(payload, "amendment_hash"),
    )

    parent = payload.get("parent_preregistration") or {}
    parent_path = repo_root / str(parent.get("path") or "")
    check("parent_exists", parent_path.is_file())
    check(
        "parent_hash",
        parent_path.is_file()
        and parent.get("file_sha256") == _sha256_file(parent_path),
    )
    parent_report = (
        verify_preregistration(parent_path, repo_root=repo_root)
        if parent_path.is_file()
        else {"verified": False, "preregistration_id": ""}
    )
    check("parent_verifies", parent_report.get("verified") is True)
    check(
        "parent_id",
        parent.get("preregistration_id")
        == parent_report.get("preregistration_id"),
    )

    evidence = payload.get("pretraining_evidence") or {}
    loaded_evidence: dict[str, dict[str, Any]] = {}
    for name in ("r1_failure", "r2_feasibility"):
        row = evidence.get(name) or {}
        evidence_path = repo_root / str(row.get("path") or "")
        check(f"{name}_exists", evidence_path.is_file())
        check(
            f"{name}_file_hash",
            evidence_path.is_file()
            and row.get("file_sha256") == _sha256_file(evidence_path),
        )
        report = _read(evidence_path) if evidence_path.is_file() else {}
        loaded_evidence[name] = report
        check(
            f"{name}_internal_hash",
            report.get("report_hash") == _payload_hash(report, "report_hash"),
        )
        check(
            f"{name}_row_report_hash",
            row.get("report_hash") == report.get("report_hash"),
        )
        for key in (
            "checks_passed",
            "fold0_record_count",
            "fold0_group_count",
            "holdout_group_count",
            "train_record_count",
            "holdout_record_count",
        ):
            check(f"{name}_bound_field:{key}", row.get(key) == report.get(key))

    r1 = loaded_evidence.get("r1_failure") or {}
    r2 = loaded_evidence.get("r2_feasibility") or {}
    check("r1_failed", r1.get("checks_passed") is False)
    check("r1_missing_classes", r1.get("missing_holdout_class_ids") == [3, 16])
    check("r1_preserved", r1.get("split_revision") == "r1")
    check("r2_passed", r2.get("checks_passed") is True)
    check("r2_train_coverage", r2.get("missing_train_class_ids") == [])
    check("r2_holdout_coverage", r2.get("missing_holdout_class_ids") == [])
    check(
        "r2_selected_groups_bound",
        (evidence.get("r2_feasibility") or {}).get(
            "selected_group_ids_sha256"
        )
        == r2.get("selected_group_ids_sha256"),
    )

    scope = payload.get("scope") or {}
    check(
        "inherit_parent_except_overrides",
        scope.get("inherit_all_parent_fields_except_explicit_overrides") is True,
    )
    check("only_birds_changed", scope.get("changed_tasks") == ["mlsp-2013-birds"])
    check(
        "experimental_design_unchanged",
        scope.get(
            "systems_seeds_condition_order_budgets_oracle_and_statistics_changed"
        )
        is False,
    )
    check(
        "no_training_before_revision",
        scope.get("formal_training_observed_before_revision") is False,
    )
    check(
        "no_metric_before_revision",
        scope.get("terminal_metric_observed_before_revision") is False,
    )

    overrides = payload.get("overrides") or {}
    task_overrides = overrides.get("tasks_by_id") or {}
    check("exact_task_override", set(task_overrides) == {"mlsp-2013-birds"})
    builder = (
        (task_overrides.get("mlsp-2013-birds") or {}).get("holdout_builder")
        or {}
    )
    check(
        "r2_split_version",
        builder.get("split_version") == "wp8-tier2-formal-mlsp-grouped-v2",
    )
    check(
        "r2_strategy",
        builder.get("strategy") == "grouped_multilabel_stratified",
    )
    check("phase_a_frozen", "sha256" in str(builder.get("phase_a") or ""))
    check("phase_b_frozen", "sha256" in str(builder.get("phase_b") or ""))
    required = set(builder.get("required_checks") or [])
    check(
        "two_sided_coverage_required",
        "all_19_species_have_at_least_one_positive_in_train_and_holdout"
        in required,
    )
    check(
        "group_overlap_zero_required",
        "train_and_holdout_group_overlap_is_zero" in required,
    )
    check(
        "fold1_exclusion_required",
        "fold1_records_and_labels_absent_from_both_views" in required,
    )
    interpretation = payload.get("interpretation") or {}
    check(
        "r1_not_used",
        interpretation.get("r1_status")
        == "retained_failed_design_not_used_for_training",
    )
    check(
        "r2_pending_staging",
        interpretation.get("r2_status")
        == "effective_design_pending_verified_staging",
    )
    check(
        "formal_training_still_forbidden",
        interpretation.get("formal_training_authorized") is False,
    )
    check(
        "effect_claim_still_forbidden",
        interpretation.get("effect_claim_authorized") is False,
    )

    report = {
        "schema": VERIFICATION_SCHEMA,
        "preregistration_id": payload.get("preregistration_id", ""),
        "amendment_file_sha256": _sha256_file(path),
        "parent_verification_hash": parent_report.get("verification_hash", ""),
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": sorted(set(errors)),
        "verified": not errors,
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    report["verification_hash"] = _payload_hash(
        report, "verification_hash"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_amendment(args.amendment, repo_root=args.repo_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    raise SystemExit(0 if report["verified"] else 1)


if __name__ == "__main__":
    main()
