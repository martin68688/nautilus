"""Runtime contracts and isolation receipts for formal Tier-2 blocks.

The training and evaluator processes run in different Pods with deliberately
different mounts.  This module keeps their small, hash-bound handoff formats in
one place so launchers, finalizers, staging verification, and tests agree on
the exact trust boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from authority.protocol_registry import canonical_json
from authority.runtime_protocol import (
    OBSERVATION_SCHEMA,
    PROTOCOL_EVIDENCE_LEVEL,
    verify_persisted_runtime_protocol_observation,
)
from fixed_holdout.common import sha256_file, write_json


SOURCE_SCHEMA = "decision_admissibility_wp8_tier2_source_snapshot_v2"
BLOCK_TEMPLATE_SCHEMA = "decision_admissibility_wp8_tier2_formal_block_template_v1"
STAGING_CONTENT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_staging_content_manifest_v1"
)
CONTINUATION_STAGING_CONTENT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_continuation_staging_content_v1"
)
FORMAL_STAGING_CONTENT_SCHEMAS = frozenset(
    {STAGING_CONTENT_SCHEMA, CONTINUATION_STAGING_CONTENT_SCHEMA}
)
CONDITION_RECEIPT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_condition_runtime_receipt_v1"
)
TRAINING_ISOLATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_training_isolation_v1"
)
EVALUATOR_ISOLATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_evaluator_isolation_v1"
)
EVALUATOR_CREATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_evaluator_creation_attestation_v1"
)
SELECTED_RUNTIME_PROTOCOL_EVIDENCE_SCHEMA = (
    "decision_admissibility_wp8_tier2_selected_runtime_protocol_evidence_v1"
)
SELECTED_RUNTIME_PROTOCOL_DENIAL_SCHEMA = (
    "decision_admissibility_wp8_tier2_selected_runtime_protocol_denial_v1"
)

_RUNTIME_PROTOCOL_KINDS = (
    "split_lineage",
    "fit_scope",
    "prediction_scope",
    "evaluator",
    "selection_freeze",
)


def payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def read_object(path: str | Path) -> dict[str, Any]:
    target = Path(path).resolve()
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {target}")
    return value


def write_hashed_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    hash_field: str,
) -> dict[str, Any]:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    value = dict(payload)
    value[hash_field] = payload_hash(value, hash_field)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, value)
    return value


def _sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _validated_runtime_mapping(
    observation: Mapping[str, Any],
    field: str,
    *,
    allow_empty_kinds: set[str] | None = None,
) -> dict[str, list[Any]]:
    raw = observation.get(field)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Runtime protocol observation lacks {field}")
    allowed_empty = allow_empty_kinds or set()
    result: dict[str, list[Any]] = {}
    for kind in _RUNTIME_PROTOCOL_KINDS:
        values = raw.get(kind)
        if not isinstance(values, list):
            raise ValueError(f"Runtime protocol observation has invalid {field}:{kind}")
        if not values and kind not in allowed_empty:
            raise ValueError(f"Runtime protocol observation has empty {field}:{kind}")
        result[kind] = list(values)
    return result


def build_selected_runtime_protocol_evidence(
    node: Mapping[str, Any],
    *,
    protocol_ref: str,
) -> dict[str, Any]:
    """Freeze the selected node's persisted host runtime observation.

    The process-local nonce registry intentionally cannot survive the training
    Pod.  This record therefore does not recreate that registry.  It binds the
    already attested observation, selected code, and all five runtime scope
    classes into the immutable training manifest that the host later seals by
    deleting the training Pod.
    """

    node_id = str(node.get("id") or "")
    code = str(node.get("code") or "")
    if not node_id or not code:
        raise ValueError("Selected node lacks an ID or code snapshot")
    observation = node.get("protocol_observation")
    if not isinstance(observation, dict) or not (
        verify_persisted_runtime_protocol_observation(observation)
    ):
        raise ValueError(
            "Selected node lacks a valid persisted runtime protocol observation"
        )
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError("Selected runtime protocol observation schema mismatch")
    if observation.get("evidence_level") != PROTOCOL_EVIDENCE_LEVEL:
        raise ValueError("Selected runtime protocol evidence level mismatch")
    if observation.get("source_code_sha256") != code_sha256:
        raise ValueError("Selected runtime protocol observation/code mismatch")
    if observation.get("code_snapshot_frozen_before_execution") is not True:
        raise ValueError("Selected code was not frozen before runtime observation")

    event_hashes = _validated_runtime_mapping(observation, "event_hashes")
    scope_hashes = _validated_runtime_mapping(observation, "scope_hashes")
    scope_inputs = _validated_runtime_mapping(observation, "scope_input_hashes")
    scope_outputs = _validated_runtime_mapping(
        observation,
        "scope_output_hashes",
        allow_empty_kinds={"fit_scope"},
    )
    callable_refs = _validated_runtime_mapping(observation, "callable_refs")
    for field, mapping in (
        ("event_hashes", event_hashes),
        ("scope_hashes", scope_hashes),
        ("scope_input_hashes", scope_inputs),
        ("scope_output_hashes", scope_outputs),
    ):
        for kind, values in mapping.items():
            if any(not _is_sha256(value) for value in values):
                raise ValueError(
                    f"Selected runtime protocol {field}:{kind} contains "
                    "a non-SHA256 value"
                )
    for kind, values in callable_refs.items():
        if any(
            not isinstance(value, Mapping)
            or not str(value.get("module") or "")
            or not str(value.get("qualname") or "")
            for value in values
        ):
            raise ValueError(
                f"Selected runtime protocol callable_refs:{kind} is invalid"
            )

    evidence: dict[str, Any] = {
        "schema": SELECTED_RUNTIME_PROTOCOL_EVIDENCE_SCHEMA,
        "node_id": node_id,
        "protocol_ref": str(protocol_ref),
        "source_code_sha256": code_sha256,
        "observation_sha256": _sha256_canonical(observation),
        "attestation_sha256": str(observation["attestation_sha256"]),
        "plan_sha256": str(observation["plan_sha256"]),
        "trace_sha256": str(observation["trace_sha256"]),
        "event_hashes_sha256": _sha256_canonical(event_hashes),
        "scope_hashes_sha256": _sha256_canonical(scope_hashes),
        "scope_input_hashes_sha256": _sha256_canonical(scope_inputs),
        "scope_output_hashes_sha256": _sha256_canonical(scope_outputs),
        "callable_refs_sha256": _sha256_canonical(callable_refs),
        "fit_scope_hashes_sha256": _sha256_canonical(scope_hashes["fit_scope"]),
        "split_scope_hashes_sha256": _sha256_canonical(scope_hashes["split_lineage"]),
        "prediction_scope_hashes_sha256": _sha256_canonical(
            scope_hashes["prediction_scope"]
        ),
        "runtime_kind_counts": {
            kind: {
                "event": len(event_hashes[kind]),
                "scope": len(scope_hashes[kind]),
                "input": len(scope_inputs[kind]),
                "output": len(scope_outputs[kind]),
                "callable": len(callable_refs[kind]),
            }
            for kind in _RUNTIME_PROTOCOL_KINDS
        },
        "persisted_observation_integrity_verified": True,
        "code_snapshot_frozen_before_execution": True,
        "evidence_level": PROTOCOL_EVIDENCE_LEVEL,
        "evidence_hash": "",
    }
    evidence["evidence_hash"] = payload_hash(evidence, "evidence_hash")
    return evidence


def build_selected_runtime_protocol_denial(
    node: Mapping[str, Any],
    *,
    protocol_ref: str,
) -> dict[str, Any]:
    """Bind a host-produced blocked observation as a condition outcome.

    A blocked observation is not positive protocol evidence and can never
    authorize terminal scoring or Result publication.  It is nevertheless a
    legitimate, non-retryable Authority outcome when the host observer bound
    the selected code and its attempted instrumentation plan.  Malformed,
    tampered, or unbound observations still raise so artifact-integrity
    failures remain block-fatal.
    """

    node_id = str(node.get("id") or "")
    code = str(node.get("code") or "")
    if not node_id or not code:
        raise ValueError("Selected node lacks an ID or code snapshot")
    observation = node.get("protocol_observation")
    if not isinstance(observation, Mapping):
        raise ValueError("Selected node lacks a runtime protocol observation")
    if observation.get("schema") != OBSERVATION_SCHEMA:
        raise ValueError("Blocked runtime protocol observation schema mismatch")
    if observation.get("status") != "blocked":
        raise ValueError("Runtime protocol observation is not a blocked outcome")
    reason = str(observation.get("reason") or "")
    if not reason:
        raise ValueError("Blocked runtime protocol observation lacks a reason")
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if observation.get("source_code_sha256") != code_sha256:
        raise ValueError("Blocked runtime protocol observation/code mismatch")
    for field in (
        "source_code_sha256",
        "executed_source_sha256",
        "plan_sha256",
    ):
        if not _is_sha256(observation.get(field)):
            raise ValueError(
                f"Blocked runtime protocol observation lacks valid {field}"
            )
    for field in (
        "event_hashes",
        "scope_hashes",
        "scope_input_hashes",
        "scope_output_hashes",
        "callable_refs",
    ):
        value = observation.get(field)
        if not isinstance(value, Mapping) or value:
            raise ValueError(
                f"Blocked runtime protocol observation has invalid {field}"
            )
    denial: dict[str, Any] = {
        "schema": SELECTED_RUNTIME_PROTOCOL_DENIAL_SCHEMA,
        "node_id": node_id,
        "protocol_ref": str(protocol_ref),
        "source_code_sha256": code_sha256,
        "executed_source_sha256": observation["executed_source_sha256"],
        "plan_sha256": observation["plan_sha256"],
        "observation_status": "blocked",
        "observation_reason": reason,
        "observation_sha256": _sha256_canonical(dict(observation)),
        "terminal_scoring_authorized": False,
        "result_publication_authorized": False,
        "retry_as_infrastructure_authorized": False,
        "denial_hash": "",
    }
    denial["denial_hash"] = payload_hash(denial, "denial_hash")
    return denial


def validate_selected_runtime_protocol_evidence(
    evidence: Mapping[str, Any],
    node: Mapping[str, Any],
    *,
    protocol_ref: str,
) -> dict[str, Any]:
    expected = build_selected_runtime_protocol_evidence(node, protocol_ref=protocol_ref)
    observed = dict(evidence)
    if observed != expected:
        raise ValueError("Selected runtime protocol evidence binding mismatch")
    return expected


def _file_inventory(root: Path, *, excluded: set[str]) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    }


def verify_source_snapshot(
    root: str | Path,
    *,
    expected_source_sha256: str,
    expected_manifest_file_sha256: str,
) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest_path = root / "WP8_TIER2_SOURCE_MANIFEST.json"
    if sha256_file(manifest_path) != expected_manifest_file_sha256:
        raise ValueError("Formal source manifest file hash mismatch")
    manifest = read_object(manifest_path)
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Formal source manifest schema mismatch")
    if manifest.get("source_sha256") != payload_hash(manifest, "source_sha256"):
        raise ValueError("Formal source manifest internal hash mismatch")
    if manifest.get("source_sha256") != expected_source_sha256:
        raise ValueError("Formal source snapshot binding mismatch")
    actual = _file_inventory(root, excluded={"WP8_TIER2_SOURCE_MANIFEST.json"})
    if actual != manifest.get("file_hashes"):
        raise ValueError("Formal source file inventory mismatch")
    writable = [
        relative for relative in actual if (root / relative).stat().st_mode & 0o222
    ]
    if writable:
        raise ValueError(f"Formal source contains writable files: {writable[:5]}")
    return {
        "schema": "decision_admissibility_wp8_tier2_source_verification_v2",
        "source_sha256": manifest["source_sha256"],
        "manifest_file_sha256": expected_manifest_file_sha256,
        "base_commit": manifest.get("base_commit"),
        "file_count": len(actual),
        "file_inventory_matches": True,
        "writable_file_count": 0,
        "verified": True,
    }


def validate_block_template(value: Mapping[str, Any]) -> dict[str, Any]:
    template = dict(value)
    if template.get("schema") != BLOCK_TEMPLATE_SCHEMA:
        raise ValueError("Formal block template schema mismatch")
    if template.get("template_hash") != payload_hash(template, "template_hash"):
        raise ValueError("Formal block template hash mismatch")
    for field in (
        "block_id",
        "task_id",
        "target_task_family",
        "target_domain",
        "protocol_ref",
        "split_id",
        "metric",
        "source_snapshot_sha256",
        "source_manifest_file_sha256",
        "train_manifest_sha256",
        "evaluator_manifest_sha256",
        "bundle_id",
        "bundle_manifest_sha256",
        "bundle_current_file_sha256",
        "formal_clause_id",
        "condition_order",
        "candidate_execution_contract",
        "expected_training_pod_name",
        "expected_training_pod_namespace",
        "expected_evaluator_pod_name",
        "expected_evaluator_pod_namespace",
        "container_image_digest",
        "output_root_id",
    ):
        if not template.get(field):
            raise ValueError(f"Formal block template lacks {field}")
    if not isinstance(template.get("maximize"), bool):
        raise ValueError("Formal block template metric direction is invalid")
    return template


def build_runtime_block_contract(
    template: Mapping[str, Any],
    *,
    staging_content_hash: str,
    staging_gate_hash: str,
    pod_name: str,
    pod_namespace: str,
    pod_uid: str,
) -> dict[str, Any]:
    from fixed_holdout.formal_block_training import (
        CONTRACT_SCHEMA,
        POD_IDENTITY_SCHEMA,
    )

    checked = validate_block_template(template)
    if pod_name != checked["expected_training_pod_name"]:
        raise ValueError("Runtime training Pod name does not match the template")
    if pod_namespace != checked["expected_training_pod_namespace"]:
        raise ValueError("Runtime training Pod namespace does not match the template")
    if not pod_uid:
        raise ValueError("Runtime training Pod UID is empty")
    value = {
        key: item
        for key, item in checked.items()
        if key
        not in {
            "template_hash",
            "expected_training_pod_name",
            "expected_training_pod_namespace",
            "expected_evaluator_pod_name",
            "expected_evaluator_pod_namespace",
        }
    }
    value.update(
        {
            "schema": CONTRACT_SCHEMA,
            "block_template_hash": checked["template_hash"],
            "staging_manifest_hash": staging_content_hash,
            "staging_gate_hash": staging_gate_hash,
            "training_pod_identity": {
                "schema": POD_IDENTITY_SCHEMA,
                "execution_kind": "devpod",
                "namespace": pod_namespace,
                "pod_name": pod_name,
                "pod_uid": pod_uid,
            },
            "contract_hash": "",
        }
    )
    value["contract_hash"] = payload_hash(value, "contract_hash")
    return value


def mount_points() -> set[str]:
    points: set[str] = set()
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        prefix = line.split(" - ", 1)[0].split()
        if len(prefix) >= 5:
            points.add(prefix[4])
    return points


def validate_evaluator_isolation_receipt(
    path: str | Path,
    *,
    block_id: str,
    training_manifest_hash: str,
    deletion_attestation_hash: str,
    evaluator_manifest_sha256: str,
    train_manifest_sha256: str,
    source_snapshot_sha256: str,
    container_image_digest: str,
) -> dict[str, Any]:
    payload = read_object(path)
    if payload.get("schema") != EVALUATOR_ISOLATION_SCHEMA:
        raise ValueError("Formal evaluator isolation schema mismatch")
    if payload.get("receipt_hash") != payload_hash(payload, "receipt_hash"):
        raise ValueError("Formal evaluator isolation hash mismatch")
    expected = {
        "block_id": block_id,
        "training_manifest_hash": training_manifest_hash,
        "training_pod_deletion_attestation_hash": deletion_attestation_hash,
        "evaluator_manifest_sha256": evaluator_manifest_sha256,
        "train_manifest_sha256": train_manifest_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "container_image_digest": container_image_digest,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Formal evaluator isolation binding mismatch: {field}")
    required_true = (
        "cpu_only",
        "memory_bundle_absent",
        "solver_secret_absent",
        "solver_environment_absent",
        "whole_workspace_absent",
        "source_read_only",
        "train_view_read_only",
        "evaluator_view_read_only",
        "created_after_training_pod_not_found",
    )
    for field in required_true:
        if payload.get(field) is not True:
            raise ValueError(f"Formal evaluator isolation check failed: {field}")
    return payload


def environment_has_solver_secret() -> bool:
    names = {
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
    }
    return any(bool(os.environ.get(name)) for name in names)


__all__ = [
    "BLOCK_TEMPLATE_SCHEMA",
    "CONTINUATION_STAGING_CONTENT_SCHEMA",
    "CONDITION_RECEIPT_SCHEMA",
    "EVALUATOR_CREATION_SCHEMA",
    "EVALUATOR_ISOLATION_SCHEMA",
    "FORMAL_STAGING_CONTENT_SCHEMAS",
    "SELECTED_RUNTIME_PROTOCOL_DENIAL_SCHEMA",
    "SELECTED_RUNTIME_PROTOCOL_EVIDENCE_SCHEMA",
    "SOURCE_SCHEMA",
    "STAGING_CONTENT_SCHEMA",
    "TRAINING_ISOLATION_SCHEMA",
    "build_selected_runtime_protocol_denial",
    "build_selected_runtime_protocol_evidence",
    "build_runtime_block_contract",
    "environment_has_solver_secret",
    "mount_points",
    "payload_hash",
    "read_object",
    "validate_block_template",
    "validate_evaluator_isolation_receipt",
    "validate_selected_runtime_protocol_evidence",
    "verify_source_snapshot",
    "write_hashed_json",
]
