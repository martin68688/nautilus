from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
MLEVOLVE_ROOT = ROOT / "mlevolve"
if str(MLEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE_ROOT))

from authority.protocol_registry import ProtocolRegistry, canonical_json  # noqa: E402
from engine.candidate_execution_contract import (  # noqa: E402
    build_candidate_execution_contract,
)


PREREGISTRATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_preregistration_v1"
)
VERIFICATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_preregistration_verification_v1"
)
ONLINE_SYSTEMS = (
    "no_memory",
    "flat_relevance_memory",
    "global_validity_bit",
    "authority_only",
    "full_decision_admissibility",
)
TASK_CONTRACTS = {
    "aerial-cactus-identification": {
        "domain": "image",
        "memory_split": "task-heldout",
        "protocol_id": "random-classification",
        "metric": "macro_f1",
        "direction": "maximize",
        "strategy": "stratified_random",
    },
    "mlsp-2013-birds": {
        "domain": "audio",
        "memory_split": "seed-heldout",
        "protocol_id": "grouped-classification",
        "metric": "macro_f1",
        "direction": "maximize",
        "strategy": "grouped",
    },
    "new-york-city-taxi-fare-prediction": {
        "domain": "tabular",
        "memory_split": "seed-heldout",
        "protocol_id": "chronological-regression",
        "metric": "rmse",
        "direction": "minimize",
        "strategy": "chronological",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(value)).encode("utf-8")).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _protocol_id(protocol_ref: str) -> str:
    return str(protocol_ref).split("@", 1)[0]


def _check_schedule(
    payload: Mapping[str, Any],
    *,
    task_ids: set[str],
    seeds: set[int],
) -> dict[str, bool]:
    design = payload.get("condition_order_design") or {}
    blocks = design.get("blocks") or []
    expected_pairs = {(task_id, seed) for task_id in task_ids for seed in seeds}
    observed_pairs: list[tuple[str, int]] = []
    orders: list[tuple[str, ...]] = []
    for block in blocks:
        try:
            observed_pairs.append(
                (str(block["task_id"]), int(block["agent_seed"]))
            )
        except (KeyError, TypeError, ValueError):
            observed_pairs.append(("", -1))
        orders.append(tuple(str(value) for value in block.get("order") or []))
    position_counts = [Counter(order[index] for order in orders if len(order) == 5) for index in range(5)]
    balanced = bool(
        len(position_counts) == 5
        and all(
            set(counter) == set(ONLINE_SYSTEMS)
            and max(counter.values()) - min(counter.values()) <= 1
            for counter in position_counts
        )
    )
    return {
        "schedule_has_all_task_seed_blocks": (
            len(observed_pairs) == len(expected_pairs)
            and set(observed_pairs) == expected_pairs
            and len(set(observed_pairs)) == len(observed_pairs)
        ),
        "every_order_is_online_system_permutation": all(
            len(order) == len(ONLINE_SYSTEMS)
            and set(order) == set(ONLINE_SYSTEMS)
            for order in orders
        ),
        "orders_are_unique": len(set(orders)) == len(orders),
        "position_balance_within_one": balanced,
        "oracle_is_posthoc": design.get("oracle_position")
        == "host-side only after all five online candidate sets are frozen",
    }


def verify_preregistration(
    preregistration_path: str | Path,
    *,
    repo_root: str | Path = ROOT,
) -> dict[str, Any]:
    path = Path(preregistration_path).resolve()
    repo_root = Path(repo_root).resolve()
    errors: list[str] = []
    checks: dict[str, bool] = {}

    try:
        payload = _read_object(path)
    except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as error:
        payload = {}
        errors.append(f"preregistration_read:{type(error).__name__}")

    def check(name: str, condition: Any) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(name)

    check("schema", payload.get("schema") == PREREGISTRATION_SCHEMA)
    check(
        "status_pending_staging",
        payload.get("status") == "design_frozen_pending_staging_hash_manifest",
    )
    freeze = payload.get("freeze_sequence") or {}
    check(
        "training_forbidden_before_staging_manifest",
        freeze.get("formal_training_before_phase_2") == "forbidden",
    )
    check(
        "revision_requires_new_id",
        "new preregistration_id" in str(freeze.get("revision_rule") or ""),
    )

    authoritative = payload.get("authoritative_inputs") or {}
    for name, relative in (
        ("execution_plan", "coordination/decision_admissibility_complete_execution_plan_20260719.md"),
        ("formal_inventory", "coordination/decision_admissibility_wp8_tier2_formal_inventory_20260722_r1.json"),
    ):
        row = authoritative.get(name) or {}
        source = repo_root / relative
        check(f"{name}_path", row.get("path") == relative)
        check(
            f"{name}_hash",
            source.is_file() and row.get("sha256") == _sha256_file(source),
        )

    registry = ProtocolRegistry(repo_root / "mlevolve/config/protocols")
    protocol_rows = authoritative.get("protocol_files") or {}
    expected_protocol_files = {
        "random-classification-v1.json": "random-classification@1",
        "grouped-classification-v1.json": "grouped-classification@1",
        "chronological-regression-v1.json": "chronological-regression@1",
    }
    check(
        "exact_protocol_file_set",
        set(protocol_rows) == set(expected_protocol_files),
    )
    for filename, key in expected_protocol_files.items():
        row = protocol_rows.get(filename) or {}
        source = repo_root / "mlevolve/config/protocols" / filename
        try:
            expected_ref = registry.resolve(key).ref().key()
        except Exception:
            expected_ref = ""
        check(
            f"protocol_file_hash:{filename}",
            source.is_file() and row.get("file_sha256") == _sha256_file(source),
        )
        check(
            f"protocol_ref:{filename}",
            row.get("protocol_ref") == expected_ref,
        )

    raw_seeds = payload.get("agent_seeds") or []
    try:
        seeds = {int(value) for value in raw_seeds}
    except (TypeError, ValueError):
        seeds = set()
    check("three_unique_agent_seeds", len(raw_seeds) == len(seeds) == 3)
    check("formal_seeds_not_wp4_source_seeds", not seeds.intersection(range(42, 48)))
    seed_policy = payload.get("seed_policy") or {}
    check("best_seed_selection_forbidden", seed_policy.get("best_seed_selection") is False)
    check("all_seeds_declared", seed_policy.get("all_declared_seeds_run") is True)

    task_rows = payload.get("tasks") or []
    task_by_id = {
        str(row.get("task_id") or ""): row
        for row in task_rows
        if isinstance(row, Mapping)
    }
    check("exact_task_set", set(task_by_id) == set(TASK_CONTRACTS))
    for task_id, expected in TASK_CONTRACTS.items():
        row = task_by_id.get(task_id) or {}
        metric = row.get("terminal_metric") or {}
        holdout = row.get("holdout_builder") or {}
        check(f"task_domain:{task_id}", row.get("domain") == expected["domain"])
        check(
            f"task_memory_split:{task_id}",
            row.get("memory_split") == expected["memory_split"],
        )
        check(
            f"task_protocol:{task_id}",
            _protocol_id(str(row.get("protocol_ref") or ""))
            == expected["protocol_id"],
        )
        check(f"task_metric:{task_id}", metric.get("name") == expected["metric"])
        check(
            f"task_metric_direction:{task_id}",
            metric.get("direction") == expected["direction"],
        )
        check(
            f"task_split_strategy:{task_id}",
            holdout.get("strategy") == expected["strategy"],
        )
        check(
            f"physical_label_isolation:{task_id}",
            "terminal_labels_absent_from_train_view"
            in set(holdout.get("required_checks") or []),
        )

        contract = row.get("candidate_contract") or {}
        shared = payload.get("shared_candidate_contract") or {}
        try:
            rebuilt = build_candidate_execution_contract(
                contract_id=str(contract.get("contract_id") or ""),
                max_execution_seconds=int(shared.get("max_execution_seconds", 0)),
                max_epochs=int(shared.get("max_epochs", 0)),
                max_cv_folds=int(shared.get("max_cv_folds", 0)),
                max_trainable_models=int(shared.get("max_trainable_models", 0)),
                allowed_import_roots=list(contract.get("allowed_import_roots") or []),
                allow_remote_assets=bool(shared.get("allow_remote_assets")),
                allow_unverified_local_assets=bool(
                    shared.get("allow_unverified_local_assets")
                ),
                allow_dataset_wide_per_sample_precompute=bool(
                    shared.get("allow_dataset_wide_per_sample_precompute")
                ),
                allow_source_score_inheritance=bool(
                    shared.get("allow_source_score_inheritance")
                ),
            )
        except Exception:
            rebuilt = {}
        check(
            f"candidate_contract_hash:{task_id}",
            contract.get("contract_hash") == rebuilt.get("contract_hash"),
        )

    check(
        "at_least_three_protocol_families",
        len({_protocol_id(str(row.get("protocol_ref") or "")) for row in task_rows})
        >= 3,
    )
    check(
        "both_seed_and_task_heldout",
        {str(row.get("memory_split") or "") for row in task_rows}
        == {"seed-heldout", "task-heldout"},
    )

    systems = payload.get("online_systems") or []
    system_ids = tuple(str(row.get("system_id") or "") for row in systems)
    check("exact_online_systems", set(system_ids) == set(ONLINE_SYSTEMS) and len(system_ids) == 5)
    check("oracle_declared", (payload.get("oracle") or {}).get("system_id") == "oracle")
    check("oracle_not_agent_visible", (payload.get("oracle") or {}).get("agent_visible") is False)
    check("oracle_no_cross_seed_selection", (payload.get("oracle") or {}).get("cross_seed_selection") is False)

    for name, passed in _check_schedule(
        payload,
        task_ids=set(task_by_id),
        seeds=seeds,
    ).items():
        check(name, passed)

    bundle = payload.get("memory_bundle_contract") or {}
    check("same_bundle_within_block", bundle.get("same_bundle_within_task_seed_block") is True)
    check("no_memory_binds_bundle", bundle.get("no_memory_still_binds_bundle_identity") is True)
    check("task_history_exposure_zero", bundle.get("task_heldout_target_history_exposure") == 0)
    check("seed_overlap_zero", bundle.get("seed_heldout_group_overlap") == 0)
    check(
        "raw_wp4_not_direct_enforce_memory",
        "not mounted directly" in str(bundle.get("formal_child_bundle_requirement") or ""),
    )
    check(
        "source_score_inheritance_forbidden",
        (payload.get("shared_candidate_contract") or {}).get(
            "allow_source_score_inheritance"
        )
        is False,
    )
    check(
        "same_domain_hard_rule",
        "Cross-domain and unknown-source method content is forbidden"
        in str(bundle.get("hard_domain_rule") or ""),
    )

    selection = payload.get("selection_and_terminal_evaluation") or {}
    check(
        "selected_node_frozen_before_labels",
        "Before any terminal labels are mounted"
        in str(selection.get("system_selected_node") or ""),
    )
    check(
        "system_metric_not_terminal_best",
        "must not replace that node with the terminal-best candidate"
        in str(selection.get("system_metric") or ""),
    )
    check(
        "oracle_only_terminal_best",
        "solely to construct" in str(selection.get("oracle_metric") or ""),
    )
    check(
        "result_adoption_causal_separate",
        "Their absence never blocks an independent clean Result Fact"
        in str(selection.get("adoption_and_causal") or ""),
    )

    isolation = payload.get("pod_and_label_isolation") or {}
    check("devpod_only", isolation.get("execution_kind") == "devpod_only")
    check("kubernetes_job_forbidden", isolation.get("kubernetes_job_forbidden") is True)
    check("cpu_only_evaluator", isolation.get("evaluator_is_cpu_only") is True)
    check(
        "gpu_deleted_before_evaluator",
        "verify NotFound" in str(isolation.get("ordering") or ""),
    )

    failure = payload.get("failure_and_retry_policy") or {}
    check("failed_roots_not_reused", failure.get("failed_source_or_output_root_reuse") is False)
    check("old_failed_roots_preserved", failure.get("old_failed_roots_preserved") is True)
    check("model_failures_are_outcomes", failure.get("model_or_candidate_failure_counts_as_outcome") is True)
    check(
        "no_post_metric_exclusions",
        "No condition, seed or task is excluded after observing a terminal metric"
        in str(failure.get("exclusion_rule") or ""),
    )

    analysis = payload.get("analysis_plan") or {}
    check("forty_five_online_observations", analysis.get("online_system_observation_count") == 45)
    check("nine_oracle_observations", analysis.get("oracle_observation_count") == 9)
    check("holm_correction", str(analysis.get("multiple_comparison_correction") or "").startswith("Holm"))
    check(
        "raw_cross_metric_average_forbidden",
        "Never average raw macro-F1 and RMSE"
        in str(analysis.get("cross_task_aggregation") or ""),
    )

    claims = set(payload.get("claims_not_authorized_by_preregistration") or [])
    check(
        "superiority_claim_not_authorized",
        "Full Decision Admissibility improves any target metric." in claims,
    )
    check("wp8_not_complete", "WP8 Stop Gate has passed." in claims)

    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "preregistration_id": payload.get("preregistration_id", ""),
        "preregistration_file_sha256": _sha256_file(path) if path.is_file() else "",
        "task_ids": sorted(task_by_id),
        "protocol_ids": sorted(
            {_protocol_id(str(row.get("protocol_ref") or "")) for row in task_rows}
        ),
        "agent_seeds": sorted(seeds),
        "online_system_ids": sorted(system_ids),
        "check_count": len(checks),
        "passed_check_count": sum(value is True for value in checks.values()),
        "checks": dict(sorted(checks.items())),
        "verified": not errors,
        "errors": sorted(set(errors)),
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    report["verification_hash"] = _sha256_json(
        {key: value for key, value in report.items() if key != "verification_hash"}
    )
    return report


def write_verification_exclusive(
    output_path: str | Path,
    report: Mapping[str, Any],
) -> None:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                copy.deepcopy(dict(report)),
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
            )
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_preregistration(
        args.preregistration,
        repo_root=args.repo_root,
    )
    if args.output is not None:
        write_verification_exclusive(args.output, report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
