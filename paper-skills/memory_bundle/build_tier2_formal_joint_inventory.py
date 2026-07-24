"""Build a result-blind, hash-only inventory of formal WP8 Tier-2 blocks.

This module deliberately does not deserialize any JSON below either formal
output root.  The block trees are treated as opaque byte collections.  All
semantic counts come from the separately frozen completed-block inventory and
score-blind structure audits supplied by the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence


ANALYSIS_SPEC_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_joint_inventory_analysis_spec_v1"
)
COMPLETED_FREEZE_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_completed_blocks_freeze_v1"
)
STRUCTURE_AUDIT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_block_structure_audit_v1"
)
COMPLETED_GATE_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_staging_stop_gate_v1"
)
CONTINUATION_GATE_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_continuation_r5_stop_gate_v1"
)
RECOVERY_GATE_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_recovery_stop_gate_v1"
)
RECOVERY_DIAGNOSTIC_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_preterminal_finalizer_diagnostic_v1"
)
REPORT_SCHEMA = "decision_admissibility_wp8_tier2_formal_joint_inventory_v1"
MANIFEST_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_joint_inventory_manifest_v1"
)

COMPLETED_ROOT_NAME = "decision-admissibility-wp8-tier2-formal-runs-r10"
CONTINUATION_ROOT_NAME = "decision-admissibility-wp8-tier2-formal-runs-r13"
ROOT_ROLE_COMPLETED = "completed_r10"
ROOT_ROLE_CONTINUATION = "continuation_r13"

CONDITIONS = (
    "no_memory",
    "flat_relevance_memory",
    "global_validity_bit",
    "authority_only",
    "full_decision_admissibility",
)

EXPECTED_BLOCKS: dict[str, dict[str, Any]] = {
    "wp8-tier2-formal-aerial-seed-104729-r3": {
        "task_id": "aerial-cactus-identification",
        "agent_seed": 104729,
        "root_role": ROOT_ROLE_COMPLETED,
    },
    "wp8-tier2-formal-aerial-seed-130363-r3": {
        "task_id": "aerial-cactus-identification",
        "agent_seed": 130363,
        "root_role": ROOT_ROLE_COMPLETED,
    },
    "wp8-tier2-formal-aerial-seed-155921-r3": {
        "task_id": "aerial-cactus-identification",
        "agent_seed": 155921,
        "root_role": ROOT_ROLE_COMPLETED,
    },
    "wp8-tier2-formal-birds-seed-104729-r3": {
        "task_id": "mlsp-2013-birds",
        "agent_seed": 104729,
        "root_role": ROOT_ROLE_COMPLETED,
    },
    "wp8-tier2-formal-birds-seed-130363-r5": {
        "task_id": "mlsp-2013-birds",
        "agent_seed": 130363,
        "root_role": ROOT_ROLE_CONTINUATION,
    },
    "wp8-tier2-formal-birds-seed-155921-r5": {
        "task_id": "mlsp-2013-birds",
        "agent_seed": 155921,
        "root_role": ROOT_ROLE_CONTINUATION,
    },
    "wp8-tier2-formal-taxi-seed-104729-r5": {
        "task_id": "new-york-city-taxi-fare-prediction",
        "agent_seed": 104729,
        "root_role": ROOT_ROLE_CONTINUATION,
    },
    "wp8-tier2-formal-taxi-seed-130363-r5": {
        "task_id": "new-york-city-taxi-fare-prediction",
        "agent_seed": 130363,
        "root_role": ROOT_ROLE_CONTINUATION,
    },
    "wp8-tier2-formal-taxi-seed-155921-r5": {
        "task_id": "new-york-city-taxi-fare-prediction",
        "agent_seed": 155921,
        "root_role": ROOT_ROLE_CONTINUATION,
    },
}

COMPLETED_BLOCK_IDS = frozenset(
    block_id
    for block_id, row in EXPECTED_BLOCKS.items()
    if row["root_role"] == ROOT_ROLE_COMPLETED
)
CONTINUATION_BLOCK_IDS = frozenset(EXPECTED_BLOCKS) - COMPLETED_BLOCK_IDS

# The original r10 staging created all nine r3 directories before execution.
# Five were never started and must remain byte-empty; their formal outcomes
# live only under the fresh r13/r5 continuation root.
COMPLETED_PRECREATED_EMPTY_BLOCK_IDS = frozenset(
    {
        "wp8-tier2-formal-birds-seed-130363-r3",
        "wp8-tier2-formal-birds-seed-155921-r3",
        "wp8-tier2-formal-taxi-seed-104729-r3",
        "wp8-tier2-formal-taxi-seed-130363-r3",
        "wp8-tier2-formal-taxi-seed-155921-r3",
    }
)

REQUIRED_BLOCK_ARTIFACTS = (
    "BLOCK_CONTRACT.json",
    "TRAINING_MANIFEST.json",
    "TRAINING_POD_DELETION_ATTESTATION.json",
    "EVALUATOR_POD_CREATION_ATTESTATION.json",
    "EVALUATOR_ISOLATION.json",
    "EVALUATION_SUMMARY.json",
    "HOST_ORACLE.json",
    "BLOCK_EVALUATION_FILE_MANIFEST.json",
    "EVALUATOR_POD_DELETION_ATTESTATION.json",
    "EVALUATION_COMPLETE",
)

AUDIT_REQUIRED_BINDINGS = frozenset(
    {
        "TRAINING_MANIFEST.json",
        "TRAINING_POD_DELETION_ATTESTATION.json",
        "EVALUATION_SUMMARY.json",
        "BLOCK_EVALUATION_FILE_MANIFEST.json",
        "EVALUATOR_POD_DELETION_ATTESTATION.json",
    }
)

FORBIDDEN_ROOT_MARKERS = (
    "decision-admissibility-wp8-tier2-formal-runs-r8",
    "decision-admissibility-wp8-tier2-formal-runs-r12",
)
FORBIDDEN_BLOCK_SUFFIX = "-r4"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload), sort_keys=True, ensure_ascii=False, indent=2
    ) + "\n"


def _is_hex_digest(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _require_hex(value: object, label: str) -> str:
    if not _is_hex_digest(value):
        raise ValueError(f"Invalid SHA-256 binding: {label}")
    return str(value)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    _assert_no_numeric_score_values(value, label=str(path))
    return value


def _is_score_value_key(key: str) -> bool:
    normalized = key.strip().lower()
    if normalized in {
        "score",
        "selected_score",
        "best_score",
        "terminal_score",
        "source_score",
        "oracle_score",
    }:
        return True
    if normalized.endswith("_score") and not normalized.endswith(
        ("_score_hash", "_score_sha256")
    ):
        return True
    if normalized in {"score_value", "terminal_score_value"}:
        return True
    return False


def _assert_no_numeric_score_values(value: Any, *, label: str) -> None:
    """Reject semantic inputs/outputs containing a numeric score value.

    Counts, hashes and boolean declarations such as ``score_values_inspected``
    remain legal.  Formal block files are never passed to this function because
    they are never parsed at all.
    """

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                if _is_score_value_key(key) and isinstance(child, (int, float)) and not isinstance(child, bool):
                    raise ValueError(f"Numeric score value is forbidden: {label}:{path}.{key}")
                if key.strip().lower() in {"selected_score", "best_score"}:
                    raise ValueError(f"Terminal score field is forbidden: {label}:{path}.{key}")
                walk(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")

    walk(value, "$")


def _assert_no_nonoutcome_references(value: Any, *, label: str) -> None:
    """Reject failed-root or r4 block references on the formal outcome surface."""

    def walk(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                walk(child, f"{path}.{raw_key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
        elif isinstance(item, str):
            if any(marker in item for marker in FORBIDDEN_ROOT_MARKERS):
                raise ValueError(
                    f"Failed formal root is forbidden: {label}:{path}"
                )
            if item.startswith("wp8-tier2-formal-") and item.endswith(
                FORBIDDEN_BLOCK_SUFFIX
            ):
                raise ValueError(
                    f"Non-outcome r4 block is forbidden: {label}:{path}"
                )

    walk(value, "$")


def _validate_internal_hash(
    payload: Mapping[str, Any], field: str, *, label: str
) -> str:
    declared = _require_hex(payload.get(field), f"{label}:{field}")
    if declared != _payload_hash(payload, field):
        raise ValueError(f"Internal hash mismatch: {label}:{field}")
    return declared


def _binding(path: Path, internal_hash: str, internal_hash_field: str) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "file_sha256": _sha256_file(path),
        "internal_hash_field": internal_hash_field,
        "internal_hash": internal_hash,
    }


def _validate_analysis_spec(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    payload = _read_object(path)
    if payload.get("schema") != ANALYSIS_SPEC_SCHEMA:
        raise ValueError("Joint-inventory analysis spec schema mismatch")
    if payload.get("status") != "result_blind_frozen":
        raise ValueError("Joint-inventory analysis spec is not frozen")
    if payload.get("score_values_may_be_inspected") is not False:
        raise ValueError("Analysis spec permits terminal score inspection")
    if payload.get("expected_online_condition_count") != 45:
        raise ValueError("Analysis spec online-condition count drift")
    if payload.get("expected_oracle_disposition_count") != 9:
        raise ValueError("Analysis spec Oracle count drift")
    _assert_no_nonoutcome_references(payload, label=str(path))
    expected_rows = payload.get("expected_blocks") or []
    if not isinstance(expected_rows, list):
        raise ValueError("Analysis spec expected_blocks must be a list")
    observed: dict[str, dict[str, Any]] = {}
    for row in expected_rows:
        if not isinstance(row, Mapping):
            raise ValueError("Analysis spec block entry is not an object")
        block_id = str(row.get("block_id") or "")
        if not block_id or block_id in observed:
            raise ValueError("Analysis spec block IDs are empty or duplicated")
        observed[block_id] = {
            "task_id": str(row.get("task_id") or ""),
            "agent_seed": int(row.get("agent_seed", -1)),
            "root_role": str(row.get("root_role") or ""),
        }
    if observed != EXPECTED_BLOCKS:
        raise ValueError("Analysis spec does not declare the exact nine-block set")
    spec_hash = _validate_internal_hash(payload, "spec_hash", label=str(path))
    return payload, _binding(path, spec_hash, "spec_hash")


def _validate_completed_freeze(
    path: Path,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
    payload = _read_object(path)
    if payload.get("schema") != COMPLETED_FREEZE_SCHEMA:
        raise ValueError("Completed-block freeze schema mismatch")
    if payload.get("status") != "four_completed_blocks_frozen_before_continuation_staging":
        raise ValueError("Completed-block freeze status mismatch")
    if payload.get("score_values_included") is not False or payload.get(
        "score_values_inspected"
    ) is not False:
        raise ValueError("Completed-block freeze is not result-blind")
    inventory_hash = _validate_internal_hash(
        payload, "inventory_hash", label=str(path)
    )
    if payload.get("completed_block_count") != 4:
        raise ValueError("Completed-block freeze must contain four blocks")
    if payload.get("completed_online_condition_count") != 20:
        raise ValueError("Completed-block freeze condition count mismatch")
    if payload.get("completed_oracle_count") != 4:
        raise ValueError("Completed-block freeze Oracle count mismatch")
    blocks = payload.get("blocks") or {}
    if not isinstance(blocks, Mapping) or set(map(str, blocks)) != COMPLETED_BLOCK_IDS:
        raise ValueError("Completed-block freeze has the wrong block set")
    rows: dict[str, dict[str, Any]] = {}
    successful_total = 0
    failed_total = 0
    result_total = 0
    for block_id in sorted(COMPLETED_BLOCK_IDS):
        raw = blocks[block_id]
        if not isinstance(raw, Mapping):
            raise ValueError(f"Completed freeze block is not an object: {block_id}")
        expected = EXPECTED_BLOCKS[block_id]
        successful = int(raw.get("successful_selected_result_count", -1))
        failed = int(raw.get("failed_online_condition_count", -1))
        result_facts = int(raw.get("result_fact_count", -1))
        if raw.get("task_id") != expected["task_id"] or int(
            raw.get("agent_seed", -1)
        ) != expected["agent_seed"]:
            raise ValueError(f"Completed freeze metadata mismatch: {block_id}")
        if successful + failed != 5 or result_facts != successful:
            raise ValueError(f"Completed freeze structure mismatch: {block_id}")
        if raw.get("may_rerun") is not False:
            raise ValueError(f"Completed block is incorrectly rerunnable: {block_id}")
        for field in (
            "evaluation_summary_hash",
            "evaluation_summary_file_sha256",
            "evaluator_pod_deletion_attestation_hash",
            "evaluator_pod_deletion_file_sha256",
        ):
            _require_hex(raw.get(field), f"{block_id}:{field}")
        optional_manifest = raw.get("block_evaluation_file_manifest_sha256")
        if optional_manifest is not None:
            _require_hex(optional_manifest, f"{block_id}:block_evaluation_file_manifest_sha256")
        rows[block_id] = {
            "task_id": expected["task_id"],
            "agent_seed": expected["agent_seed"],
            "online_condition_count": 5,
            "successful_selected_result_count": successful,
            "failed_online_condition_count": failed,
            "result_fact_count": result_facts,
            "oracle_disposition_count": 1,
            "condition_structure_source": "completed_freeze",
            "condition_structure": [],
            "semantic_source_hash": inventory_hash,
        }
        successful_total += successful
        failed_total += failed
        result_total += result_facts
    if successful_total != payload.get("successful_selected_result_count"):
        raise ValueError("Completed freeze successful-result aggregate mismatch")
    if failed_total != payload.get("failed_online_condition_count"):
        raise ValueError("Completed freeze failure aggregate mismatch")
    if result_total != payload.get("result_fact_count"):
        raise ValueError("Completed freeze Result Fact aggregate mismatch")
    return payload, _binding(path, inventory_hash, "inventory_hash"), rows


def _validate_structure_audit(
    path: Path,
) -> tuple[str, dict[str, Any], dict[str, str], dict[str, Any]]:
    payload = _read_object(path)
    if payload.get("schema") != STRUCTURE_AUDIT_SCHEMA:
        raise ValueError(f"Structure-audit schema mismatch: {path}")
    if payload.get("status") != "passed":
        raise ValueError(f"Structure audit did not pass: {path}")
    if payload.get("score_values_included") is not False or payload.get(
        "score_values_inspected"
    ) is not False:
        raise ValueError(f"Structure audit is not result-blind: {path}")
    audit_hash = _validate_internal_hash(payload, "audit_hash", label=str(path))
    block_id = str(payload.get("block_id") or "")
    expected = EXPECTED_BLOCKS.get(block_id)
    if expected is None or block_id.endswith(FORBIDDEN_BLOCK_SUFFIX):
        raise ValueError(f"Structure audit refers to a non-outcome block: {block_id}")
    if payload.get("task_id") != expected["task_id"] or int(
        payload.get("agent_seed", -1)
    ) != expected["agent_seed"]:
        raise ValueError(f"Structure-audit metadata mismatch: {block_id}")
    successful = int(payload.get("successful_selected_result_count", -1))
    failed = int(payload.get("failed_online_condition_count", -1))
    result_facts = int(payload.get("result_fact_count", -1))
    if payload.get("online_condition_count") != 5 or successful + failed != 5:
        raise ValueError(f"Structure-audit condition count mismatch: {block_id}")
    if result_facts != successful:
        raise ValueError(f"Structure-audit Result Fact mismatch: {block_id}")

    raw_conditions = payload.get("condition_structure") or {}
    if not isinstance(raw_conditions, Mapping) or set(map(str, raw_conditions)) != set(CONDITIONS):
        raise ValueError(f"Structure audit lacks the exact condition set: {block_id}")
    condition_rows: list[dict[str, Any]] = []
    observed_successful = 0
    observed_failed = 0
    for condition in CONDITIONS:
        raw = raw_conditions[condition]
        if not isinstance(raw, Mapping):
            raise ValueError(f"Condition structure is not an object: {block_id}:{condition}")
        status_value = str(raw.get("status") or "")
        result_count = int(raw.get("result_fact_count", -1))
        row: dict[str, Any] = {
            "condition": condition,
            "status": status_value,
            "result_fact_count": result_count,
        }
        if status_value == "scored_selected_result":
            if result_count != 1 or raw.get("result_fact_derived_from_refs", []) != []:
                raise ValueError(f"Successful condition Result Fact mismatch: {block_id}:{condition}")
            if raw.get("terminal_value_omitted") is not True:
                raise ValueError(f"Successful condition is not blinded: {block_id}:{condition}")
            row["terminal_value_omitted"] = True
            observed_successful += 1
        elif status_value in {"pre_terminal_failure", "selected_candidate_rejected"}:
            if result_count != 0 or raw.get("terminal_metric_observed") is not False:
                raise ValueError(f"Failed condition has terminal evidence: {block_id}:{condition}")
            classification = str(raw.get("failure_classification") or "")
            if not classification:
                raise ValueError(f"Failed condition lacks classification: {block_id}:{condition}")
            row["failure_classification"] = classification
            row["terminal_metric_observed"] = False
            observed_failed += 1
        else:
            raise ValueError(f"Unknown condition status: {block_id}:{condition}:{status_value}")
        condition_rows.append(row)
    if observed_successful != successful or observed_failed != failed:
        raise ValueError(f"Condition detail/aggregate mismatch: {block_id}")

    invariants = payload.get("structural_invariants") or {}
    for name in (
        "all_five_conditions_retained",
        "successful_plus_failed_equals_five",
        "successful_conditions_have_exactly_one_independent_result_fact",
        "target_history_not_used",
        "terminal_values_remain_blinded",
    ):
        if invariants.get(name) is not True:
            raise ValueError(f"Structure-audit invariant failed: {block_id}:{name}")
    # The recovered r3 Birds audit predates the r5 wording change and freezes
    # the equivalent shorter key.  Condition rows above are still checked
    # directly for both terminal_metric_observed=False and result_fact_count=0;
    # accepting this alias therefore does not weaken the invariant.
    failure_invariant = invariants.get(
        "failed_conditions_have_no_terminal_result_or_result_fact"
    )
    if failure_invariant is None:
        failure_invariant = invariants.get("failed_conditions_have_no_result_fact")
    if failure_invariant is not True:
        raise ValueError(
            "Structure-audit invariant failed: "
            f"{block_id}:failed_conditions_have_no_terminal_result_or_result_fact"
        )
    lifecycle = payload.get("lifecycle")
    if isinstance(lifecycle, Mapping):
        for name in (
            "training_pod_not_found",
            "evaluator_created_after_training_not_found_attestation",
            "evaluator_pod_not_found",
        ):
            if lifecycle.get(name) is not True:
                raise ValueError(
                    f"Structure-audit lifecycle failed: {block_id}:{name}"
                )
    else:
        # The recovered r3 Birds block has an additional recovery Pod and
        # freezes the stronger ordering proof under recovery_structure.
        recovery = payload.get("recovery_structure") or {}
        required_true = (
            "recovery_pod_not_found",
            "training_pod_not_found",
            "evaluator_pod_not_found",
            "evaluator_created_after_both_required_not_found_attestations",
        )
        required_false = (
            "agent_reexecuted",
            "candidate_code_reexecuted",
            "full_condition_reexecuted",
            "terminal_metric_observed_during_recovery",
        )
        if recovery.get("recovery_status") != (
            "deterministic_preterminal_finalizer_recovered"
        ):
            raise ValueError(
                f"Structure-audit recovery lifecycle failed: {block_id}:status"
            )
        for name in required_true:
            if recovery.get(name) is not True:
                raise ValueError(
                    f"Structure-audit recovery lifecycle failed: {block_id}:{name}"
                )
        for name in required_false:
            if recovery.get(name) is not False:
                raise ValueError(
                    f"Structure-audit recovery lifecycle failed: {block_id}:{name}"
                )
    oracle = payload.get("oracle_structure") or {}
    if oracle.get("normal_result_fact_published") is not False:
        raise ValueError(f"Oracle published a normal Result Fact: {block_id}")
    if "score_values_omitted" in oracle and oracle.get("score_values_omitted") is not True:
        raise ValueError(f"Oracle values are not omitted: {block_id}")

    bindings = payload.get("artifact_bindings") or {}
    if not isinstance(bindings, Mapping) or not AUDIT_REQUIRED_BINDINGS <= set(bindings):
        raise ValueError(f"Structure audit lacks required artifact bindings: {block_id}")
    for filename, raw in bindings.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"Invalid artifact binding: {block_id}:{filename}")
        _require_hex(raw.get("file_sha256"), f"{block_id}:{filename}:file_sha256")
        _require_hex(raw.get("internal_hash"), f"{block_id}:{filename}:internal_hash")

    row = {
        "task_id": expected["task_id"],
        "agent_seed": expected["agent_seed"],
        "online_condition_count": 5,
        "successful_selected_result_count": successful,
        "failed_online_condition_count": failed,
        "result_fact_count": result_facts,
        "oracle_disposition_count": 1,
        "condition_structure_source": "structure_audit",
        "condition_structure": condition_rows,
        "semantic_source_hash": audit_hash,
    }
    return block_id, payload, _binding(path, audit_hash, "audit_hash"), row


def _validate_gate(
    path: Path,
    *,
    schema: str,
    score_flag: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = _read_object(path)
    if payload.get("schema") != schema or payload.get("status") != "passed":
        raise ValueError(f"Formal staging/recovery Gate did not pass: {path}")
    if score_flag in payload and payload.get(score_flag) is not False:
        raise ValueError(f"Formal Gate is not result-blind: {path}")
    gate_hash = _validate_internal_hash(payload, "gate_hash", label=str(path))
    return payload, _binding(path, gate_hash, "gate_hash")


def _validate_recovery_diagnostic(
    path: Path,
    *,
    completed_source_sha256: str,
    completed_gate_hash: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    payload = _read_object(path)
    if payload.get("schema") != RECOVERY_DIAGNOSTIC_SCHEMA:
        raise ValueError("Recovery diagnostic schema mismatch")
    if payload.get("block_id") != "wp8-tier2-formal-birds-seed-104729-r3":
        raise ValueError("Recovery diagnostic block mismatch")
    preserved = payload.get("preserved_output") or {}
    if preserved.get("terminal_score_file_count") != 0 or preserved.get(
        "terminal_metric_observed"
    ) is not False or preserved.get("terminal_score_values_inspected") is not False:
        raise ValueError("Recovery diagnostic is not result-blind")
    frozen = payload.get("frozen_execution") or {}
    if frozen.get("source_snapshot_sha256") != completed_source_sha256:
        raise ValueError("Recovery diagnostic source snapshot mismatch")
    if frozen.get("staging_gate_hash") != completed_gate_hash:
        raise ValueError("Recovery diagnostic staging Gate mismatch")
    diagnostic_hash = _validate_internal_hash(
        payload, "diagnostic_hash", label=str(path)
    )
    return payload, _binding(path, diagnostic_hash, "diagnostic_hash")


def _hash_block_tree(
    block_root: Path,
    *,
    required_artifacts: Sequence[str] = REQUIRED_BLOCK_ARTIFACTS,
) -> tuple[dict[str, str], str]:
    """Hash a formal block without parsing files or following symlinks."""

    if not block_root.is_dir() or block_root.is_symlink():
        raise ValueError(f"Formal block root is absent or unsafe: {block_root}")
    files: dict[str, str] = {}
    entries: list[Path] = []
    for current, directory_names, file_names in os.walk(
        block_root, followlinks=False
    ):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        # os.walk does not descend into directory symlinks when followlinks is
        # false, but they still need their readlink target hash-bound.
        entries.extend(
            current_path / name
            for name in directory_names
            if (current_path / name).is_symlink()
        )
        entries.extend(current_path / name for name in file_names)
    for path in sorted(entries, key=lambda item: item.as_posix()):
        relative = path.relative_to(block_root).as_posix()
        if path.is_symlink():
            target = os.readlink(path)
            files[relative] = hashlib.sha256(
                b"formal-symlink-v1\0" + os.fsencode(target)
            ).hexdigest()
            continue
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"Non-regular formal artifact is forbidden: {path}")
        files[relative] = _sha256_file(path)
    missing = sorted(set(required_artifacts) - set(files))
    if missing:
        raise ValueError(f"Formal block lacks required artifacts: {block_root.name}:{missing}")
    tree_hash = hashlib.sha256(_canonical_bytes(files)).hexdigest()
    return files, tree_hash


def _block_directories(
    root: Path,
    expected: frozenset[str],
    *,
    allowed_empty: frozenset[str] = frozenset(),
) -> dict[str, Path]:
    root = root.resolve()
    blocks_root = root / "blocks"
    if not blocks_root.is_dir() or blocks_root.is_symlink():
        raise ValueError(f"Formal root lacks a safe blocks directory: {root}")
    entries: dict[str, Path] = {}
    for path in sorted(blocks_root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Unexpected formal blocks entry: {path}")
        entries[path.name] = path
    if set(entries) != expected | allowed_empty:
        raise ValueError(
            f"Formal root block set mismatch: {root.name}:"
            f"expected={sorted(expected | allowed_empty)}:actual={sorted(entries)}"
        )
    if any(block_id.endswith(FORBIDDEN_BLOCK_SUFFIX) for block_id in entries):
        raise ValueError("A non-outcome r4 block entered the formal roots")
    for block_id in sorted(allowed_empty):
        if any(entries[block_id].iterdir()):
            raise ValueError(
                f"Precreated non-outcome block root is not empty: {block_id}"
            )
    return {block_id: entries[block_id] for block_id in expected}


def _validate_root_identity(root: Path, expected_name: str) -> Path:
    resolved = root.resolve()
    if resolved.name != expected_name:
        raise ValueError(
            f"Formal root identity mismatch: expected={expected_name}:actual={resolved.name}"
        )
    text = resolved.as_posix()
    if any(marker in text for marker in FORBIDDEN_ROOT_MARKERS):
        raise ValueError("A failed r8/r12 root was supplied as a formal result root")
    return resolved


def _crosscheck_artifact_bindings(
    *,
    block_id: str,
    files: Mapping[str, str],
    completed_freeze: Mapping[str, Any],
    structure_audit: Mapping[str, Any] | None,
) -> None:
    if block_id in COMPLETED_BLOCK_IDS:
        frozen = (completed_freeze.get("blocks") or {})[block_id]
        comparisons = {
            "EVALUATION_SUMMARY.json": frozen.get(
                "evaluation_summary_file_sha256"
            ),
            "EVALUATOR_POD_DELETION_ATTESTATION.json": frozen.get(
                "evaluator_pod_deletion_file_sha256"
            ),
            "BLOCK_EVALUATION_FILE_MANIFEST.json": frozen.get(
                "block_evaluation_file_manifest_sha256"
            ),
        }
        for filename, expected in comparisons.items():
            if expected is None:
                if filename in files:
                    raise ValueError(
                        "Completed-freeze declared artifact absence but file "
                        f"exists: {block_id}:{filename}"
                    )
                continue
            if files.get(filename) != expected:
                raise ValueError(
                    f"Completed-freeze file binding mismatch: {block_id}:{filename}"
                )
    if structure_audit is not None:
        for relative, raw in (structure_audit.get("artifact_bindings") or {}).items():
            expected = str(raw.get("file_sha256") or "")
            if files.get(str(relative)) != expected:
                raise ValueError(
                    f"Structure-audit file binding mismatch: {block_id}:{relative}"
                )


def compute_joint_inventory(
    *,
    completed_root: str | Path,
    continuation_root: str | Path,
    analysis_spec_path: str | Path,
    completed_freeze_path: str | Path,
    completed_staging_gate_path: str | Path,
    continuation_staging_gate_path: str | Path,
    recovery_gate_path: str | Path,
    recovery_diagnostic_path: str | Path,
    structure_audit_paths: Sequence[str | Path],
    created_at: str,
) -> dict[str, Any]:
    completed_root = _validate_root_identity(Path(completed_root), COMPLETED_ROOT_NAME)
    continuation_root = _validate_root_identity(
        Path(continuation_root), CONTINUATION_ROOT_NAME
    )
    if completed_root == continuation_root:
        raise ValueError("Completed and continuation formal roots must differ")

    analysis_spec_path = Path(analysis_spec_path).resolve()
    completed_freeze_path = Path(completed_freeze_path).resolve()
    completed_staging_gate_path = Path(completed_staging_gate_path).resolve()
    continuation_staging_gate_path = Path(
        continuation_staging_gate_path
    ).resolve()
    recovery_gate_path = Path(recovery_gate_path).resolve()
    recovery_diagnostic_path = Path(recovery_diagnostic_path).resolve()
    audit_paths = [Path(path).resolve() for path in structure_audit_paths]
    if len(set(audit_paths)) != len(audit_paths):
        raise ValueError("Structure-audit paths are duplicated")

    _analysis_spec, spec_binding = _validate_analysis_spec(analysis_spec_path)
    completed_freeze, freeze_binding, semantic_rows = _validate_completed_freeze(
        completed_freeze_path
    )
    completed_gate, completed_gate_binding = _validate_gate(
        completed_staging_gate_path,
        schema=COMPLETED_GATE_SCHEMA,
        score_flag="terminal_metric_observed",
    )
    continuation_gate, continuation_gate_binding = _validate_gate(
        continuation_staging_gate_path,
        schema=CONTINUATION_GATE_SCHEMA,
        score_flag="terminal_score_values_inspected",
    )
    recovery_gate, recovery_gate_binding = _validate_gate(
        recovery_gate_path,
        schema=RECOVERY_GATE_SCHEMA,
        score_flag="terminal_score_values_inspected",
    )

    completed_source = _require_hex(
        ((completed_gate.get("evidence") or {}).get("source_verification") or {}).get(
            "source_sha256"
        ),
        "completed staging Gate source_sha256",
    )
    continuation_source = _require_hex(
        continuation_gate.get("source_snapshot_sha256"),
        "continuation staging Gate source_snapshot_sha256",
    )
    completed_gate_hash = str(completed_gate["gate_hash"])
    continuation_gate_hash = str(continuation_gate["gate_hash"])
    _recovery_diagnostic, recovery_diagnostic_binding = (
        _validate_recovery_diagnostic(
            recovery_diagnostic_path,
            completed_source_sha256=completed_source,
            completed_gate_hash=completed_gate_hash,
        )
    )
    if recovery_gate.get("source_snapshot_sha256") not in {None, completed_source}:
        raise ValueError("Recovery Gate source snapshot mismatch")

    audit_payloads: dict[str, dict[str, Any]] = {}
    audit_bindings: list[dict[str, str]] = []
    for path in audit_paths:
        block_id, payload, binding, row = _validate_structure_audit(path)
        if block_id in audit_payloads:
            raise ValueError(f"Duplicate structure audit for block: {block_id}")
        audit_payloads[block_id] = payload
        audit_bindings.append({"block_id": block_id, **binding})
        if block_id in COMPLETED_BLOCK_IDS:
            frozen = semantic_rows[block_id]
            for field in (
                "successful_selected_result_count",
                "failed_online_condition_count",
                "result_fact_count",
            ):
                if frozen[field] != row[field]:
                    raise ValueError(
                        f"Completed freeze/audit mismatch: {block_id}:{field}"
                    )
        semantic_rows[block_id] = row
    missing_continuation_audits = sorted(CONTINUATION_BLOCK_IDS - set(audit_payloads))
    if missing_continuation_audits:
        raise ValueError(
            f"Continuation blocks lack structure audits: {missing_continuation_audits}"
        )

    completed_dirs = _block_directories(
        completed_root,
        COMPLETED_BLOCK_IDS,
        allowed_empty=COMPLETED_PRECREATED_EMPTY_BLOCK_IDS,
    )
    continuation_dirs = _block_directories(
        continuation_root, CONTINUATION_BLOCK_IDS
    )
    all_dirs = {**completed_dirs, **continuation_dirs}
    block_rows: list[dict[str, Any]] = []
    for block_id in sorted(EXPECTED_BLOCKS):
        expected = EXPECTED_BLOCKS[block_id]
        legacy_manifest_absence = False
        required_artifacts = list(REQUIRED_BLOCK_ARTIFACTS)
        if expected["root_role"] == ROOT_ROLE_COMPLETED:
            frozen_block = (completed_freeze.get("blocks") or {})[block_id]
            legacy_manifest_absence = frozen_block.get(
                "block_evaluation_file_manifest_sha256"
            ) is None
            if legacy_manifest_absence:
                required_artifacts.remove("BLOCK_EVALUATION_FILE_MANIFEST.json")
        files, tree_hash = _hash_block_tree(
            all_dirs[block_id], required_artifacts=required_artifacts
        )
        _crosscheck_artifact_bindings(
            block_id=block_id,
            files=files,
            completed_freeze=completed_freeze,
            structure_audit=audit_payloads.get(block_id),
        )
        semantics = semantic_rows[block_id]
        source_sha256 = (
            completed_source
            if expected["root_role"] == ROOT_ROLE_COMPLETED
            else continuation_source
        )
        staging_gate_hash = (
            completed_gate_hash
            if expected["root_role"] == ROOT_ROLE_COMPLETED
            else continuation_gate_hash
        )
        required_hashes = {
            filename: files.get(filename) for filename in REQUIRED_BLOCK_ARTIFACTS
        }
        block_rows.append(
            {
                "block_id": block_id,
                "task_id": expected["task_id"],
                "agent_seed": expected["agent_seed"],
                "root_role": expected["root_role"],
                "source_snapshot_sha256": source_sha256,
                "staging_gate_hash": staging_gate_hash,
                "block_file_count": len(files),
                "block_tree_sha256": tree_hash,
                "required_artifact_hashes": required_hashes,
                **semantics,
                "oracle_disposition": "host_only_frozen_union_no_normal_result_fact",
                "training_deletion_attestation_hashed": True,
                "evaluator_deletion_attestation_hashed": True,
                "block_evaluation_file_manifest_hashed": (
                    "BLOCK_EVALUATION_FILE_MANIFEST.json" in files
                ),
                "block_evaluation_file_manifest_legacy_absence_frozen": (
                    legacy_manifest_absence
                ),
            }
        )

    successful_total = sum(
        int(row["successful_selected_result_count"]) for row in block_rows
    )
    failed_total = sum(int(row["failed_online_condition_count"]) for row in block_rows)
    result_fact_total = sum(int(row["result_fact_count"]) for row in block_rows)
    condition_total = sum(int(row["online_condition_count"]) for row in block_rows)
    oracle_total = sum(int(row["oracle_disposition_count"]) for row in block_rows)
    source_by_block = {
        row["block_id"]: {
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "staging_gate_hash": row["staging_gate_hash"],
        }
        for row in block_rows
    }
    source_binding_hash = hashlib.sha256(
        _canonical_bytes(source_by_block)
    ).hexdigest()
    input_bindings: dict[str, Any] = {
        "analysis_spec": spec_binding,
        "completed_blocks_freeze": freeze_binding,
        "completed_staging_gate": completed_gate_binding,
        "continuation_staging_gate": continuation_gate_binding,
        "recovery_gate": recovery_gate_binding,
        "recovery_diagnostic": recovery_diagnostic_binding,
        "structure_audits": sorted(audit_bindings, key=lambda row: row["block_id"]),
    }
    invariants = {
        "exact_expected_block_set": {row["block_id"] for row in block_rows}
        == set(EXPECTED_BLOCKS),
        "exact_nine_blocks": len(block_rows) == 9,
        "exact_45_online_conditions": condition_total == 45,
        "successful_plus_failed_equals_45": successful_total + failed_total == 45,
        "result_fact_count_equals_successful_count": result_fact_total
        == successful_total,
        "exact_nine_oracle_dispositions": oracle_total == 9,
        "all_required_lifecycle_artifacts_hashed": all(
            row["training_deletion_attestation_hashed"]
            and row["evaluator_deletion_attestation_hashed"]
            and (
                row["block_evaluation_file_manifest_hashed"]
                or row[
                    "block_evaluation_file_manifest_legacy_absence_frozen"
                ]
            )
            for row in block_rows
        ),
        "legacy_manifest_absence_matches_completed_freeze": all(
            row["block_evaluation_file_manifest_hashed"]
            != row["block_evaluation_file_manifest_legacy_absence_frozen"]
            for row in block_rows
        ),
        "all_semantic_sources_result_blind": True,
        "formal_block_files_hashed_without_json_parsing": True,
        "non_outcome_formal_roots_absent": True,
        "precreated_unstarted_r10_block_dirs_verified_empty": True,
        "mixed_source_hashes_bound_per_block": len(
            {row["source_snapshot_sha256"] for row in block_rows}
        )
        == 2,
    }
    if not all(invariants.values()):
        raise ValueError(f"Joint-inventory invariant failure: {invariants}")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "passed",
        "created_at": str(created_at),
        "score_policy": "hash_only",
        "score_values_included": False,
        "score_values_inspected": False,
        "formal_block_json_parsed": False,
        "score_bearing_artifacts_parsed": False,
        "parsed_structural_input_classes": [
            "analysis_spec",
            "completed_blocks_freeze",
            "staging_and_recovery_gates",
            "recovery_diagnostic",
            "score_blind_structure_audits",
        ],
        "block_tree_hash_algorithm": (
            "sha256(canonical_json(relative_posix_path_to_domain_separated_"
            "regular_file_or_readlink_sha256));symlinks_are_never_followed"
        ),
        "formal_roots": {
            ROOT_ROLE_COMPLETED: str(completed_root),
            ROOT_ROLE_CONTINUATION: str(continuation_root),
        },
        "excluded_precreated_empty_block_dirs": [
            {
                "block_id": block_id,
                "root_role": ROOT_ROLE_COMPLETED,
                "disposition": "precreated_never_started_empty_not_an_outcome",
            }
            for block_id in sorted(COMPLETED_PRECREATED_EMPTY_BLOCK_IDS)
        ],
        "input_bindings": input_bindings,
        "source_bindings": {
            ROOT_ROLE_COMPLETED: {
                "source_snapshot_sha256": completed_source,
                "staging_gate_hash": completed_gate_hash,
                "recovery_gate_hash": recovery_gate["gate_hash"],
            },
            ROOT_ROLE_CONTINUATION: {
                "source_snapshot_sha256": continuation_source,
                "staging_gate_hash": continuation_gate_hash,
            },
            "source_binding_hash": source_binding_hash,
        },
        "blocks": block_rows,
        "totals": {
            "block_count": len(block_rows),
            "online_condition_count": condition_total,
            "successful_selected_result_count": successful_total,
            "failed_online_condition_count": failed_total,
            "result_fact_count": result_fact_total,
            "oracle_disposition_count": oracle_total,
            "legacy_block_evaluation_file_manifest_absence_count": sum(
                row[
                    "block_evaluation_file_manifest_legacy_absence_frozen"
                ]
                for row in block_rows
            ),
        },
        "invariants": invariants,
        "effect_claim_authorized": False,
        "next_authorized_phase": "terminal_score_unblinding_and_frozen_statistics",
        "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
        "report_hash": "",
    }
    _assert_no_numeric_score_values(report, label="joint_inventory_report")
    _assert_no_nonoutcome_references(report, label="joint_inventory_report")
    report["report_hash"] = _payload_hash(report, "report_hash")
    return report


def compute_manifest(
    report: Mapping[str, Any], *, inventory_file_sha256: str
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "status": "passed",
        "joint_inventory_file": "joint_inventory.json",
        "joint_inventory_file_sha256": _require_hex(
            inventory_file_sha256, "joint inventory file"
        ),
        "joint_inventory_hash": report["report_hash"],
        "input_binding_hash": hashlib.sha256(
            _canonical_bytes(report["input_bindings"])
        ).hexdigest(),
        "source_binding_hash": report["source_bindings"]["source_binding_hash"],
        "block_tree_binding_hash": hashlib.sha256(
            _canonical_bytes(
                {
                    row["block_id"]: row["block_tree_sha256"]
                    for row in report["blocks"]
                }
            )
        ).hexdigest(),
        "score_policy": "hash_only",
        "score_values_included": False,
        "score_values_inspected": False,
        "score_bearing_artifacts_parsed": False,
        "builder_source_sha256": report["builder_source_sha256"],
        "manifest_hash": "",
    }
    _assert_no_numeric_score_values(manifest, label="joint_inventory_manifest")
    manifest["manifest_hash"] = _payload_hash(manifest, "manifest_hash")
    return manifest


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_exclusive(path, _json_text(payload))


def build_joint_inventory(
    *,
    output_root: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse joint-inventory root: {output_root}")
    report = compute_joint_inventory(**kwargs)
    output_root.mkdir(parents=True, exist_ok=False)
    inventory_path = output_root / "joint_inventory.json"
    _write_json_exclusive(inventory_path, report)
    manifest = compute_manifest(
        report, inventory_file_sha256=_sha256_file(inventory_path)
    )
    _write_json_exclusive(output_root / "manifest.json", manifest)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-root", required=True, type=Path)
    parser.add_argument("--continuation-root", required=True, type=Path)
    parser.add_argument("--analysis-spec", required=True, type=Path)
    parser.add_argument("--completed-freeze", required=True, type=Path)
    parser.add_argument("--completed-staging-gate", required=True, type=Path)
    parser.add_argument("--continuation-staging-gate", required=True, type=Path)
    parser.add_argument("--recovery-gate", required=True, type=Path)
    parser.add_argument("--recovery-diagnostic", required=True, type=Path)
    parser.add_argument(
        "--structure-audit", action="append", default=[], type=Path
    )
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build_joint_inventory(
        output_root=args.output_root,
        completed_root=args.completed_root,
        continuation_root=args.continuation_root,
        analysis_spec_path=args.analysis_spec,
        completed_freeze_path=args.completed_freeze,
        completed_staging_gate_path=args.completed_staging_gate,
        continuation_staging_gate_path=args.continuation_staging_gate,
        recovery_gate_path=args.recovery_gate,
        recovery_diagnostic_path=args.recovery_diagnostic,
        structure_audit_paths=args.structure_audit,
        created_at=args.created_at,
    )
    print(_json_text(report), end="")


if __name__ == "__main__":
    main()


__all__ = [
    "ANALYSIS_SPEC_SCHEMA",
    "COMPLETED_FREEZE_SCHEMA",
    "COMPLETED_GATE_SCHEMA",
    "CONTINUATION_GATE_SCHEMA",
    "RECOVERY_DIAGNOSTIC_SCHEMA",
    "RECOVERY_GATE_SCHEMA",
    "STRUCTURE_AUDIT_SCHEMA",
    "REPORT_SCHEMA",
    "MANIFEST_SCHEMA",
    "EXPECTED_BLOCKS",
    "COMPLETED_PRECREATED_EMPTY_BLOCK_IDS",
    "REQUIRED_BLOCK_ARTIFACTS",
    "build_joint_inventory",
    "compute_joint_inventory",
    "compute_manifest",
]
