from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
MEMORY_BUNDLE = Path(__file__).resolve().parent
MLEVOLVE = REPO / "mlevolve"
for path in (MEMORY_BUNDLE, MLEVOLVE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_decision_admissibility_factorial import (  # noqa: E402
    ATTACKS,
    PROTOCOL_IDS,
    REPORT_SCHEMA,
    VARIANTS,
    run_factorial,
)
from schema import sha256_json, write_json_atomic  # noqa: E402


VERIFICATION_SCHEMA = "decision_admissibility_tier0_verification_v1"
TRACE_FIELDS = {
    "requested_claim_refs",
    "requested_operations",
    "requested_protocol_ref",
    "requested_generation_stages",
    "requested_governance_stages",
    "satisfied_paths",
    "missing_obligations",
    "blocking_receipts",
    "visible_clause_ids",
    "suppressed_clause_refs",
    "warning_clause_ids",
    "prompt_contains_forbidden_text",
    "rendered_prompt_sha256",
    "lineage_scope_widened",
    "lineage_trace_sha256",
    "bundle_version",
    "split_id",
    "policy_version",
    "pre_prompt_visibility_enforced",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_factorial_report(
    report_path: str | Path,
    *,
    source_root: str | Path = REPO,
    exact_replay: bool = True,
) -> dict[str, Any]:
    report_path = Path(report_path).resolve()
    source_root = Path(source_root).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, code: str) -> None:
        if not condition:
            errors.append(code)

    require(report.get("schema") == REPORT_SCHEMA, "schema")
    expected_report_hash = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    require(report.get("report_hash") == expected_report_hash, "report_hash")
    source_hashes = report.get("implementation_source_hashes") or {}
    require(isinstance(source_hashes, dict) and bool(source_hashes), "source_hashes")
    for relative, expected in sorted(source_hashes.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(f"unsafe_source_path:{relative}")
            continue
        path = source_root / relative_path
        require(path.is_file(), f"missing_source:{relative}")
        if path.is_file():
            require(
                _sha256_file(path) == str(expected),
                f"source_hash:{relative}",
            )
    runner_relative = "paper-skills/memory_bundle/run_decision_admissibility_factorial.py"
    require(
        report.get("runner_source_sha256") == source_hashes.get(runner_relative),
        "runner_source_binding",
    )

    registry_relative = Path(str(report.get("protocol_registry") or ""))
    if (
        not str(registry_relative)
        or registry_relative.is_absolute()
        or ".." in registry_relative.parts
    ):
        errors.append("protocol_registry_path")
        registry_path = source_root / "mlevolve" / "config" / "protocols"
    else:
        registry_path = source_root / registry_relative
    protocol_hashes = report.get("protocol_file_hashes") or {}
    for protocol_id in PROTOCOL_IDS:
        filename = f"{protocol_id}-v1.json"
        path = registry_path / filename
        require(path.is_file(), f"missing_protocol:{filename}")
        if path.is_file():
            require(
                _sha256_file(path) == str(protocol_hashes.get(filename) or ""),
                f"protocol_hash:{filename}",
            )

    cases = report.get("cases") or []
    require(isinstance(cases, list), "cases_type")
    expected_keys = {
        (protocol_id, attack, variant)
        for protocol_id in PROTOCOL_IDS
        for attack in ATTACKS
        for variant in VARIANTS
    }
    observed_keys = {
        (
            str(case.get("protocol_ref") or "").split("@", 1)[0],
            str(case.get("attack") or ""),
            str(case.get("variant") or ""),
        )
        for case in cases
        if isinstance(case, dict)
    }
    require(len(cases) == len(expected_keys) == 63, "case_count")
    require(observed_keys == expected_keys, "matrix_keys")
    require(len(observed_keys) == len(cases), "duplicate_cases")
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or "missing")
        trace = case.get("trace_contract") or {}
        require(set(trace) == TRACE_FIELDS, f"trace_fields:{case_id}")
        require(bool(trace.get("requested_claim_refs")), f"trace_claim:{case_id}")
        require(bool(trace.get("requested_operations")), f"trace_operation:{case_id}")
        require(
            trace.get("requested_protocol_ref") == case.get("protocol_ref"),
            f"trace_protocol:{case_id}",
        )
        require(
            trace.get("pre_prompt_visibility_enforced") is True,
            f"trace_visibility:{case_id}",
        )
        require(
            trace.get("prompt_contains_forbidden_text") is False,
            f"trace_prompt:{case_id}",
        )
        require(case.get("passed") is True, f"case_failed:{case_id}")

    invalid_cases = [
        case for case in cases if case.get("invalid_attack_present") is True
    ]
    invalid_activation_count = sum(
        int(case.get("invalid_activation_count") or 0) for case in cases
    )
    valid_opportunities = sum(
        int(case.get("valid_knowledge_opportunity_count") or 0)
        for case in cases
    )
    valid_retained = sum(
        int(case.get("valid_knowledge_retained_count") or 0) for case in cases
    )
    prompt_exposure = sum(
        int(case.get("unauthorized_prompt_exposure") or 0) for case in cases
    )
    require(len(invalid_cases) == report.get("invalid_attack_episode_count") == 42, "invalid_denominator")
    require(invalid_activation_count == report.get("invalid_activation_count") == 0, "invalid_activation")
    require(report.get("invalid_influence_rate") == 0.0, "iir")
    require(valid_opportunities == report.get("valid_knowledge_opportunity_count") == 57, "valid_denominator")
    require(valid_retained == report.get("valid_knowledge_retained_count") == 57, "valid_retained")
    require(report.get("valid_knowledge_retention") == 1.0, "vkr")
    require(prompt_exposure == report.get("unauthorized_prompt_exposure_count") == 0, "prompt_exposure")
    require(report.get("matrix_complete") is True, "matrix_complete")
    require(report.get("all_cases_passed") is True, "all_cases_passed")
    require(report.get("failed_case_ids") == [], "failed_case_ids")

    replay_exact_match = None
    replay_report_hash = ""
    if exact_replay and registry_path.is_dir():
        replay = run_factorial(
            registry_path,
            created_at=str(report.get("created_at") or ""),
        )
        replay_report_hash = replay["report_hash"]
        replay_exact_match = replay == report
        require(replay_exact_match, "exact_replay")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "valid": not errors,
        "errors": sorted(set(errors)),
        "report_artifact": "tier0_factorial_report",
        "report_file_sha256": _sha256_file(report_path),
        "report_hash": str(report.get("report_hash") or ""),
        "case_count": len(cases),
        "matrix_complete": observed_keys == expected_keys,
        "exact_replay_requested": bool(exact_replay),
        "exact_replay_match": replay_exact_match,
        "exact_replay_report_hash": replay_report_hash,
        "invalid_attack_episode_count": len(invalid_cases),
        "invalid_activation_count": invalid_activation_count,
        "valid_knowledge_opportunity_count": valid_opportunities,
        "valid_knowledge_retained_count": valid_retained,
        "unauthorized_prompt_exposure_count": prompt_exposure,
        "verification_hash": "",
    }
    verification["verification_hash"] = sha256_json(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_hash"
        }
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently verify and exactly replay a WP8 Tier-0 report."
    )
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=REPO)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-exact-replay", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite verification: {args.output}")
    verification = verify_factorial_report(
        args.report,
        source_root=args.source_root,
        exact_replay=not args.no_exact_replay,
    )
    write_json_atomic(args.output, verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False, indent=2))
    if not verification["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
