"""Freeze and audit one five-condition formal Tier-2 training block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from authority.ledger import AuthorityLedger
from authority.memory_snapshot import ImmutableBaseBundle
from authority.protocol_registry import ProtocolRegistry
from engine.candidate_execution_contract import (
    AUDIT_SCHEMA,
    CONTRACT_SCHEMA as CANDIDATE_CONTRACT_SCHEMA,
    audit_candidate_code,
    valid_candidate_execution_audit,
    valid_candidate_execution_block_receipt,
)
from fixed_holdout.common import sha256_file, write_json
from fixed_holdout.formal_runtime import (
    CONDITION_RECEIPT_SCHEMA,
    TRAINING_ISOLATION_SCHEMA,
    build_selected_runtime_protocol_denial,
    build_selected_runtime_protocol_evidence,
)


SCHEMA = "decision_admissibility_wp8_tier2_formal_training_manifest_v1"
CONTRACT_SCHEMA = "decision_admissibility_wp8_tier2_formal_block_contract_v1"
POD_IDENTITY_SCHEMA = "decision_admissibility_wp8_tier2_training_pod_identity_v1"
CONDITIONS = (
    "no_memory",
    "flat_relevance_memory",
    "global_validity_bit",
    "authority_only",
    "full_decision_admissibility",
)
RETRIEVAL_CONTROLS = {
    "no_memory": "no_memory",
    "flat_relevance_memory": "flat_relevance_memory",
    "global_validity_bit": "global_validity_bit",
    "authority_only": "authority_only",
    "full_decision_admissibility": "full_decision_admissibility",
}


def _hash_payload(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _only_directory(root: Path) -> Path | None:
    if not root.is_dir():
        return None
    rows = sorted(path for path in root.iterdir() if path.is_dir())
    return rows[0] if len(rows) == 1 else None


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Formal artifact escapes block output: {path}") from error


def _protocol_key(payload: Mapping[str, Any]) -> str:
    return (
        f"{payload.get('protocol_id', '')}@{payload.get('version', '')}#"
        f"{payload.get('canonical_hash', '')}"
    )


def _resolve_frozen_protocol(protocol_key: str):
    try:
        identifier, declared_hash = str(protocol_key).rsplit("#", 1)
    except ValueError as error:
        raise ValueError("Formal block protocol key is malformed") from error
    registry = ProtocolRegistry(
        Path(__file__).resolve().parents[1] / "config" / "protocols"
    )
    spec = registry.resolve(identifier)
    if spec.canonical_hash != declared_hash or spec.ref().key() != protocol_key:
        raise ValueError("Formal block protocol hash is not the frozen registry value")
    return spec


def _validate_training_pod_identity(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("Formal block contract lacks training Pod identity")
    identity = {str(key): str(item) for key, item in value.items()}
    if identity.get("schema") != POD_IDENTITY_SCHEMA:
        raise ValueError("Formal training Pod identity schema mismatch")
    if identity.get("execution_kind") != "devpod":
        raise ValueError("Formal execution must use a devpod")
    if identity.get("namespace") != "ecepxie":
        raise ValueError("Formal training Pod namespace mismatch")
    for field in ("pod_name", "pod_uid"):
        if not identity.get(field):
            raise ValueError(f"Formal training Pod identity lacks {field}")
    if identity["pod_name"] == "jupyter-a10-d48dfd589-pqfkb":
        raise ValueError("The user-owned Jupyter Pod is outside formal scope")
    return identity


def _validate_writeback_descriptor(
    descriptor: object,
    *,
    run_dir: Path,
    ledger_path: Path,
    contract: Mapping[str, Any],
    bundle: ImmutableBaseBundle,
) -> list[Path]:
    if not isinstance(descriptor, Mapping):
        raise ValueError("Formal condition lacks an Authority writeback descriptor")
    if descriptor.get("schema") != "fixed_holdout_authority_writeback_descriptor_v1":
        raise ValueError("Authority writeback descriptor schema mismatch")
    if descriptor.get("status") != "ready":
        raise ValueError("Authority writeback descriptor is not ready")
    if descriptor.get("descriptor_hash") != _hash_payload(
        descriptor, "descriptor_hash"
    ):
        raise ValueError("Authority writeback descriptor hash mismatch")
    if descriptor.get("task_id") != contract.get("task_id"):
        raise ValueError("Authority writeback descriptor task mismatch")
    if _protocol_key(descriptor.get("active_protocol") or {}) != contract.get(
        "protocol_ref"
    ):
        raise ValueError("Authority writeback descriptor protocol mismatch")
    if descriptor.get("bundle_id") != bundle.bundle_id or descriptor.get(
        "bundle_manifest_sha256"
    ) != bundle.manifest_sha256:
        raise ValueError("Authority writeback descriptor Bundle mismatch")
    described_ledger = Path(
        str(descriptor.get("authority_ledger_path") or "")
    ).resolve()
    if described_ledger != ledger_path.resolve():
        raise ValueError("Authority writeback descriptor ledger mismatch")
    tracked: list[Path] = []
    for field in ("authority_snapshot_path",):
        path = Path(str(descriptor.get(field) or "")).resolve()
        _relative_to(path, run_dir)
        if not path.is_file():
            raise ValueError(f"Authority writeback descriptor lacks {field}")
        tracked.append(path)
    snapshot_path = tracked[0]
    if sha256_file(snapshot_path) != descriptor.get("authority_snapshot_sha256"):
        raise ValueError("Authority snapshot changed before formal freeze")
    overlay_path = Path(
        str(descriptor.get("session_overlay_path") or "")
    ).resolve()
    _relative_to(overlay_path, run_dir)
    if not overlay_path.is_dir() or not descriptor.get("session_overlay_id"):
        raise ValueError("Authority writeback descriptor overlay is incomplete")
    if len(str(descriptor.get("session_overlay_manifest_sha256") or "")) != 64:
        raise ValueError("Authority writeback descriptor overlay hash is missing")
    return tracked


def _validate_training_isolation(
    path: Path,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema") != TRAINING_ISOLATION_SCHEMA:
        raise ValueError("Formal training isolation schema mismatch")
    if payload.get("receipt_hash") != _hash_payload(payload, "receipt_hash"):
        raise ValueError("Formal training isolation hash mismatch")
    expected = {
        "block_id": contract["block_id"],
        "runtime_contract_hash": contract["contract_hash"],
        "training_pod_identity": contract["training_pod_identity"],
        "source_snapshot_sha256": contract["source_snapshot_sha256"],
        "source_manifest_file_sha256": contract["source_manifest_file_sha256"],
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "bundle_id": contract["bundle_id"],
        "bundle_manifest_sha256": contract["bundle_manifest_sha256"],
        "bundle_current_file_sha256": contract["bundle_current_file_sha256"],
        "container_image_digest": contract["container_image_digest"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Formal training isolation binding mismatch: {field}")
    for field in (
        "gpu_visible",
        "source_read_only",
        "train_view_read_only",
        "bundle_read_only",
        "solver_secret_single_file_mount",
        "whole_workspace_absent",
        "evaluator_view_absent",
        "terminal_labels_absent",
        "all_candidate_import_roots_importable",
    ):
        if payload.get(field) is not True:
            raise ValueError(f"Formal training isolation check failed: {field}")
    if payload.get("gpu_count") != 1 or payload.get("cpu_count") != 8:
        raise ValueError("Formal training resource isolation mismatch")
    return payload


def _validate_condition_runtime_receipt(
    path: Path,
    *,
    contract: Mapping[str, Any],
    condition: str,
    position: int,
    exit_code: int,
    condition_root: Path,
) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema") != CONDITION_RECEIPT_SCHEMA:
        raise ValueError("Formal condition runtime receipt schema mismatch")
    if payload.get("receipt_hash") != _hash_payload(payload, "receipt_hash"):
        raise ValueError("Formal condition runtime receipt hash mismatch")
    expected = {
        "block_id": contract["block_id"],
        "runtime_contract_hash": contract["contract_hash"],
        "condition": condition,
        "position": position,
        "retrieval_control": RETRIEVAL_CONTROLS[condition],
        "memory_enabled": condition != "no_memory",
        "memory_system": condition,
        "task_id": contract["task_id"],
        "agent_seed": contract["agent_seed"],
        "run_exit_code": exit_code,
        "steps": contract["steps_per_condition"],
        "initial_drafts": contract["initial_drafts_per_condition"],
        "agent_time_limit_seconds": contract["agent_time_limit_seconds"],
        "launcher_timeout_seconds": contract["condition_launcher_timeout_seconds"],
        "candidate_execution_contract_hash": contract[
            "candidate_execution_contract_hash"
        ],
        "training_pod_identity": contract["training_pod_identity"],
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Formal condition runtime binding mismatch: {field}")
    if payload.get("only_experimental_variable") != (
        "external_skill_memory.retrieval_control"
    ):
        raise ValueError("Formal condition changed an undeclared experimental axis")
    if payload.get("terminal_metric_observed") is not False:
        raise ValueError("Formal condition observed a terminal metric")
    if payload.get("pre_evaluator_score_file_count") != 0:
        raise ValueError("Formal condition contains a pre-evaluator score file")
    stdout = condition_root / "run_stdout.log"
    expected_stdout = sha256_file(stdout) if stdout.is_file() else ""
    if payload.get("run_stdout_sha256") != expected_stdout:
        raise ValueError("Formal condition stdout binding mismatch")
    return payload


def _candidate_inventory(submission_dir: Path) -> list[dict[str, str]]:
    rows = []
    for path in sorted(submission_dir.glob("submission_*.csv")):
        rows.append(
            {
                "node_id": path.stem.removeprefix("submission_"),
                "submission": path.name,
                "submission_sha256": sha256_file(path),
            }
        )
    return rows


def _failure_receipt(
    condition_root: Path,
    *,
    condition: str,
    exit_code: int,
    reason: str,
) -> dict[str, Any]:
    stdout = condition_root / "run_stdout.log"
    payload: dict[str, Any] = {
        "schema": "decision_admissibility_wp8_tier2_condition_failure_v1",
        "condition": condition,
        "exit_code": exit_code,
        "terminal_metric_observed": False,
        "failure_classification": "pre_terminal_unclassified",
        "reason": reason,
        "run_stdout_sha256": sha256_file(stdout) if stdout.is_file() else "",
        "receipt_hash": "",
    }
    payload["receipt_hash"] = _hash_payload(payload, "receipt_hash")
    path = condition_root / "CONDITION_FAILURE.json"
    if path.exists():
        if _read(path) != payload:
            raise ValueError("Existing condition failure Receipt changed")
        return payload
    write_json(path, payload)
    return payload


def _runtime_protocol_denial_receipt(
    condition_root: Path,
    *,
    condition: str,
    selected_node_id: str,
    request: Mapping[str, Any],
    denial: Mapping[str, Any],
) -> dict[str, Any]:
    stdout = condition_root / "run_stdout.log"
    payload: dict[str, Any] = {
        "schema": "decision_admissibility_wp8_tier2_condition_failure_v1",
        "condition": condition,
        "exit_code": 0,
        "terminal_metric_observed": False,
        "failure_classification": "authority_denial",
        "reason": "selected_runtime_protocol_observation_blocked",
        "selected_node_id": selected_node_id,
        "evaluation_request_hash": str(request.get("request_hash") or ""),
        "candidate_set_hash": str(request.get("candidate_set_hash") or ""),
        "runtime_protocol_denial": dict(denial),
        "candidate_reexecution_authorized": False,
        "retry_as_infrastructure_authorized": False,
        "run_stdout_sha256": sha256_file(stdout) if stdout.is_file() else "",
        "receipt_hash": "",
    }
    payload["receipt_hash"] = _hash_payload(payload, "receipt_hash")
    path = condition_root / "CONDITION_FAILURE.json"
    if path.exists():
        if _read(path) != payload:
            raise ValueError(
                "Existing runtime-protocol denial Receipt changed"
            )
        return payload
    write_json(path, payload)
    return payload


def _exposure_audit(
    events: list[dict[str, Any]],
    clauses: Mapping[str, Mapping[str, Any]],
    *,
    condition: str,
    task_id: str,
    target_domain: str,
    formal_clause_id: str,
) -> dict[str, Any]:
    exposures = [
        dict(event.get("payload") or {})
        for event in events
        if event.get("event_type") == "experience_exposed"
    ]
    invalid: list[dict[str, Any]] = []
    formal_count = 0
    for index, exposure in enumerate(exposures):
        reasons = []
        clause_id = str(exposure.get("clause_id") or "")
        clause = clauses.get(clause_id)
        if exposure.get("schema") != "experience_exposure_event_v2":
            reasons.append("exposure_schema_not_v2")
        if clause is None:
            reasons.append("clause_absent_from_bundle")
        target = exposure.get("target_scope") or {}
        if target.get("task_id") != task_id:
            reasons.append("target_task_mismatch")
        if target.get("domain") != target_domain:
            reasons.append("target_domain_mismatch")
        domains = set(map(str, exposure.get("source_domains") or []))
        if domains != {target_domain}:
            reasons.append("source_domain_mismatch")
        declared_domains = set(map(str, (clause or {}).get("source_domains") or []))
        if declared_domains and domains != declared_domains:
            reasons.append("source_domain_not_clause_bound")
        if exposure.get("transfer_scope") != "same_domain":
            reasons.append("transfer_scope_mismatch")
        declared_transfer = str((clause or {}).get("transfer_scope") or "")
        if declared_transfer and exposure.get("transfer_scope") != declared_transfer:
            reasons.append("transfer_scope_not_clause_bound")
        if len(str(exposure.get("prompt_sha256") or "")) != 64:
            reasons.append("prompt_hash_missing")
        if clause_id == formal_clause_id:
            formal_count += 1
        if reasons:
            invalid.append(
                {
                    "index": index,
                    "clause_id": clause_id,
                    "reasons": sorted(set(reasons)),
                }
            )
    if condition == "no_memory" and exposures:
        invalid.append(
            {
                "index": -1,
                "clause_id": "",
                "reasons": ["no_memory_exposed_experience"],
            }
        )
    if condition == "full_decision_admissibility":
        if not exposures:
            invalid.append(
                {
                    "index": -1,
                    "clause_id": formal_clause_id,
                    "reasons": ["formal_method_never_exposed"],
                }
            )
        unexpected = sorted(
            {
                str(row.get("clause_id") or "")
                for row in exposures
                if str(row.get("clause_id") or "") != formal_clause_id
            }
        )
        if unexpected:
            invalid.append(
                {
                    "index": -1,
                    "clause_id": "",
                    "reasons": [f"full_unexpected_clause:{value}" for value in unexpected],
                }
            )
    report = {
        "schema": "decision_admissibility_wp8_tier2_formal_exposure_audit_v1",
        "condition": condition,
        "exposure_event_count": len(exposures),
        "formal_method_exposure_count": formal_count,
        "invalid_exposure_count": len(invalid),
        "invalid_exposures": invalid,
        "valid": not invalid,
        "report_hash": "",
    }
    report["report_hash"] = _hash_payload(report, "report_hash")
    return report


def finalize_training_block(
    output_root: str | Path,
    block_contract_path: str | Path,
    bundle_root: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    contract_path = Path(block_contract_path).resolve()
    bundle_root = Path(bundle_root).resolve()
    output_path = output_root / "TRAINING_MANIFEST.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    contract = _read(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Formal block contract schema mismatch")
    if contract.get("contract_hash") != _hash_payload(contract, "contract_hash"):
        raise ValueError("Formal block contract hash mismatch")
    training_pod_identity = _validate_training_pod_identity(
        contract.get("training_pod_identity")
    )
    for field in (
        "block_id",
        "task_id",
        "target_task_family",
        "target_domain",
        "protocol_ref",
        "split_id",
        "metric",
        "train_manifest_sha256",
        "evaluator_manifest_sha256",
        "staging_manifest_hash",
        "staging_gate_hash",
        "block_template_hash",
        "source_snapshot_sha256",
        "source_manifest_file_sha256",
        "container_image_digest",
        "bundle_current_file_sha256",
        "bundle_manifest_file_sha256",
        "agent_time_limit_seconds",
        "condition_launcher_timeout_seconds",
    ):
        if not contract.get(field):
            raise ValueError(f"Formal block contract lacks {field}")
    if not isinstance(contract.get("maximize"), bool):
        raise ValueError("Formal block contract metric direction is invalid")
    active_protocol = _resolve_frozen_protocol(str(contract["protocol_ref"]))
    protocol_payload_enforcement = bool(
        active_protocol.promotion_policy.get("enforce_protocol_payloads")
        is True
    )
    order = list(map(str, contract.get("condition_order") or []))
    if len(order) != len(CONDITIONS) or set(order) != set(CONDITIONS):
        raise ValueError("Formal block condition order is not the frozen permutation")
    candidate_contract = contract.get("candidate_execution_contract") or {}
    if candidate_contract.get("schema") != CANDIDATE_CONTRACT_SCHEMA:
        raise ValueError("Candidate execution contract schema mismatch")
    if candidate_contract.get("contract_hash") != _hash_payload(
        candidate_contract, "contract_hash"
    ):
        raise ValueError("Candidate execution contract hash mismatch")
    if candidate_contract.get("enabled") is not True:
        raise ValueError("Candidate execution contract is not enabled")
    if candidate_contract.get("allow_source_score_inheritance") is not False:
        raise ValueError("Candidate execution contract permits source scores")
    if candidate_contract.get("contract_hash") != contract.get(
        "candidate_execution_contract_hash"
    ):
        raise ValueError("Candidate execution contract binding mismatch")

    pointer = _read(bundle_root / "CURRENT.json")
    if pointer.get("schema") != "memory_bundle_current_v1":
        raise ValueError("Formal Bundle CURRENT schema mismatch")
    if pointer.get("pointer_sha256") != _hash_payload(pointer, "pointer_sha256"):
        raise ValueError("Formal Bundle CURRENT hash mismatch")
    relative_bundle = Path(str(pointer.get("bundle_path") or ""))
    if relative_bundle.is_absolute() or ".." in relative_bundle.parts:
        raise ValueError("Formal Bundle CURRENT path is unsafe")
    bundle = ImmutableBaseBundle.load(
        bundle_root / relative_bundle, verify_artifacts=True
    )
    if bundle.bundle_id != contract.get("bundle_id"):
        raise ValueError("Formal block Bundle ID mismatch")
    if bundle.manifest_sha256 != contract.get("bundle_manifest_sha256"):
        raise ValueError("Formal block Bundle manifest mismatch")
    if pointer.get("bundle_id") != bundle.bundle_id or pointer.get(
        "manifest_sha256"
    ) != bundle.manifest_sha256:
        raise ValueError("Formal Bundle CURRENT binding mismatch")
    if sha256_file(bundle_root / "CURRENT.json") != contract.get(
        "bundle_current_file_sha256"
    ):
        raise ValueError("Formal Bundle CURRENT file hash mismatch")
    if sha256_file(bundle.path / "manifest.json") != contract.get(
        "bundle_manifest_file_sha256"
    ):
        raise ValueError("Formal Bundle manifest file hash mismatch")
    clauses = bundle.read_jsonl("sop/clauses.jsonl")
    clause_by_id = {str(row["clause_id"]): row for row in clauses}
    formal_clause_id = str(contract.get("formal_clause_id") or "")
    if formal_clause_id not in clause_by_id:
        raise ValueError("Formal method Clause is absent from the bound Bundle")
    formal_clause = clause_by_id[formal_clause_id]
    if formal_clause.get("claim_types") != ["method_hypothesis"]:
        raise ValueError("Formal Clause is not a pure method hypothesis")
    if formal_clause.get("protocol_scope") != [contract["protocol_ref"]]:
        raise ValueError("Formal Clause protocol binding mismatch")
    if formal_clause.get("source_domains") != [contract["target_domain"]]:
        raise ValueError("Formal Clause domain binding mismatch")
    if formal_clause.get("transfer_scope") != "same_domain":
        raise ValueError("Formal Clause is not same-domain transfer")
    if formal_clause.get("permitted_operations") != ["generate_candidate"]:
        raise ValueError("Formal Clause operation scope is too broad")
    if (formal_clause.get("contract_spec") or {}).get(
        "source_score_inheritance"
    ) is not False:
        raise ValueError("Formal Clause permits source-score inheritance")

    training_isolation_path = output_root / "TRAINING_ISOLATION.json"
    if not training_isolation_path.is_file():
        raise ValueError("Formal block lacks training isolation evidence")
    training_isolation = _validate_training_isolation(
        training_isolation_path, contract=contract
    )

    condition_rows: dict[str, dict[str, Any]] = {}
    for position, condition in enumerate(order):
        root = output_root / "conditions" / condition
        exit_path = root / "RUN_EXIT_CODE"
        if not exit_path.is_file():
            raise ValueError(f"Condition has no exit code: {condition}")
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
        runtime_receipt_path = root / "CONDITION_RUNTIME_RECEIPT.json"
        if not runtime_receipt_path.is_file():
            raise ValueError(f"Condition lacks runtime receipt: {condition}")
        runtime_receipt = _validate_condition_runtime_receipt(
            runtime_receipt_path,
            contract=contract,
            condition=condition,
            position=position,
            exit_code=exit_code,
            condition_root=root,
        )
        run_dir = _only_directory(root / "runs")
        workspace_dir = _only_directory(root / "workspace")
        base_row: dict[str, Any] = {
            "condition": condition,
            "position": position,
            "retrieval_control": RETRIEVAL_CONTROLS[condition],
            "run_exit_code": exit_code,
            "terminal_metric_observed": False,
            "status": "",
        }
        if exit_code != 0 or run_dir is None or workspace_dir is None:
            reason = (
                "run_process_nonzero"
                if exit_code != 0
                else "run_directory_cardinality_invalid"
            )
            failure = _failure_receipt(
                root,
                condition=condition,
                exit_code=exit_code,
                reason=reason,
            )
            condition_rows[condition] = {
                **base_row,
                "status": "pre_terminal_failure",
                "failure_receipt_path": _relative_to(
                    root / "CONDITION_FAILURE.json", output_root
                ),
                "failure_receipt_hash": failure["receipt_hash"],
                "failure_receipt_sha256": sha256_file(
                    root / "CONDITION_FAILURE.json"
                ),
                "condition_runtime_receipt_path": _relative_to(
                    runtime_receipt_path, output_root
                ),
                "condition_runtime_receipt_hash": runtime_receipt[
                    "receipt_hash"
                ],
                "condition_runtime_receipt_sha256": sha256_file(
                    runtime_receipt_path
                ),
            }
            runtime_receipt_path.chmod(
                runtime_receipt_path.stat().st_mode & ~0o222
            )
            (root / "CONDITION_FAILURE.json").chmod(
                (root / "CONDITION_FAILURE.json").stat().st_mode & ~0o222
            )
            continue

        request_path = run_dir / "fixed_holdout_evaluation_request.json"
        journal_path = run_dir / "journal.json"
        ledger_path = run_dir / "authority_events.jsonl"
        rollout_path = run_dir / "authority_rollout_report.json"
        required = (request_path, journal_path, ledger_path, rollout_path)
        if not all(path.is_file() for path in required):
            missing = [path.name for path in required if not path.is_file()]
            raise ValueError(f"Successful condition is missing artifacts: {condition}:{missing}")
        request = _read(request_path)
        if request.get("request_schema") != "fixed_holdout_evaluation_request_v3":
            raise ValueError("Formal condition did not freeze a V3 evaluation request")
        if request.get("request_hash") != _hash_payload(request, "request_hash"):
            raise ValueError("Formal evaluation request hash mismatch")
        if request.get("scores_were_visible_during_search") is not False:
            raise ValueError("Terminal score was visible during formal search")
        if request.get("selection_frozen_before_terminal_evaluation") is not True:
            raise ValueError("Formal system selection was not frozen")
        if request.get("selection_policy") != "terminal_only":
            raise ValueError("Formal selection policy is not terminal-only")
        selection_basis = request.get("selection_basis") or {}
        if selection_basis.get("type") != "solver_internal_search_metric":
            raise ValueError("Formal selection did not use the frozen search metric")
        if selection_basis.get("metric_disposition") != "search_only":
            raise ValueError("Formal selection metric is not search-only")
        if selection_basis.get("terminal_metric_observed") is not False:
            raise ValueError("Formal selection basis observed a terminal metric")
        if selection_basis.get("formal_rank_claim_authorized") is not False:
            raise ValueError("Search-only selection was promoted to a formal rank claim")
        if selection_basis.get("source_score_inherited") is not False:
            raise ValueError("Formal selection inherited a source score")
        for key in ("task_id", "split_id", "metric", "maximize"):
            if request.get(key) != contract.get(key):
                raise ValueError(f"Formal evaluation request/contract mismatch: {key}")
        if request.get("train_manifest_sha256") != contract.get(
            "train_manifest_sha256"
        ):
            raise ValueError("Formal train-view manifest binding mismatch")
        if Path(str(request.get("journal_path") or "")).resolve() != request_path.with_name(
            "journal.json"
        ).resolve():
            raise ValueError("Formal evaluation request journal path mismatch")

        journal = _read(journal_path)
        nodes = [
            row
            for row in journal.get("nodes") or []
            if isinstance(row, dict) and row.get("stage") != "root"
        ]
        expected_steps = int(contract["steps_per_condition"])
        if len(nodes) != expected_steps:
            raise ValueError(
                f"Successful condition has {len(nodes)} nodes, expected {expected_steps}"
            )
        node_by_id = {str(row["id"]): row for row in nodes}
        if len(node_by_id) != len(nodes):
            raise ValueError("Formal journal contains duplicate node IDs")
        if any(
            (row.get("role_contract") or {}).get(
                "candidate_execution_contract"
            )
            != candidate_contract
            for row in nodes
        ):
            raise ValueError("Candidate contract role binding drift")

        audit_paths = sorted(
            workspace_dir.glob("working/candidate_execution_contract_audit_*.json")
        )
        block_paths = sorted(
            workspace_dir.glob("working/candidate_execution_block_receipt_*.json")
        )
        if len(audit_paths) != len(nodes):
            raise ValueError("Candidate execution audit count mismatch")
        block_by_node = {
            path.stem.removeprefix("candidate_execution_block_receipt_"): path
            for path in block_paths
        }
        admitted: set[str] = set()
        denied: set[str] = set()
        for audit_path in audit_paths:
            audit = _read(audit_path)
            node_id = audit_path.stem.removeprefix(
                "candidate_execution_contract_audit_"
            )
            if node_id not in node_by_id or not valid_candidate_execution_audit(audit):
                raise ValueError("Invalid candidate execution audit")
            if audit.get("schema") != AUDIT_SCHEMA:
                raise ValueError("Candidate execution audit schema mismatch")
            if audit.get("contract_hash") != candidate_contract.get("contract_hash"):
                raise ValueError("Candidate execution audit contract mismatch")
            if audit.get("code_sha256") != hashlib.sha256(
                str(node_by_id[node_id].get("code") or "").encode("utf-8")
            ).hexdigest():
                raise ValueError("Candidate execution audit code mismatch")
            recomputed = audit_candidate_code(
                str(node_by_id[node_id].get("code") or ""), candidate_contract
            )
            if audit != recomputed:
                raise ValueError("Candidate execution audit was not host-reproducible")
            if audit.get("valid") is True:
                admitted.add(node_id)
                if node_id in block_by_node:
                    raise ValueError("Admitted node has a block Receipt")
            else:
                denied.add(node_id)
                block_path = block_by_node.get(node_id)
                if block_path is None:
                    raise ValueError("Denied node lacks a block Receipt")
                block = _read(block_path)
                if not valid_candidate_execution_block_receipt(block):
                    raise ValueError("Invalid candidate execution block Receipt")
                if block.get("audit_hash") != audit.get("audit_hash"):
                    raise ValueError("Candidate block/audit binding mismatch")
        if admitted | denied != set(node_by_id) or admitted & denied:
            raise ValueError("Candidate admission partition is invalid")
        if set(block_by_node) != denied:
            raise ValueError("Candidate block Receipt set mismatch")

        submission_dir = Path(str(request.get("submission_dir") or "")).resolve()
        _relative_to(submission_dir, root)
        inventory = _candidate_inventory(submission_dir)
        if inventory != request.get("candidate_inventory"):
            raise ValueError("Frozen candidate inventory changed")
        if request.get("candidate_set_hash") != _hash_payload(
            {"candidate_inventory": inventory}, "unused"
        ):
            raise ValueError("Frozen candidate-set hash mismatch")
        submitted = {row["node_id"] for row in inventory}
        if not submitted or not submitted <= admitted or submitted & denied:
            raise ValueError("Submitted candidates are not exactly admitted artifacts")
        selected_node_id = str(request.get("selected_node_id") or "")
        if selected_node_id not in submitted or selected_node_id not in node_by_id:
            raise ValueError("Preselected node is outside the admitted candidate set")
        selected_node = node_by_id[selected_node_id]
        if (
            selected_node.get("is_buggy") is True
            or selected_node.get("is_valid") is not True
            or not isinstance(selected_node.get("exec_time"), (int, float))
            or not math.isfinite(float(selected_node.get("exec_time") or 0.0))
            or float(selected_node.get("exec_time") or 0.0) <= 0.0
        ):
            raise ValueError("Preselected node lacks clean execution evidence")
        selected_runtime_protocol_evidence: dict[str, Any] = {}
        selected_runtime_protocol_denial: dict[str, Any] = {}
        if protocol_payload_enforcement:
            observation = selected_node.get("protocol_observation")
            if isinstance(observation, Mapping) and (
                observation.get("status") == "blocked"
            ):
                selected_runtime_protocol_denial = (
                    build_selected_runtime_protocol_denial(
                        selected_node,
                        protocol_ref=str(contract["protocol_ref"]),
                    )
                )
            else:
                selected_runtime_protocol_evidence = (
                    build_selected_runtime_protocol_evidence(
                        selected_node,
                        protocol_ref=str(contract["protocol_ref"]),
                    )
                )
        if request.get("journal_sha256") != sha256_file(journal_path):
            raise ValueError("Frozen journal hash mismatch")
        if list(run_dir.glob("fixed_holdout_scores*.json")):
            raise ValueError("Terminal score file exists before CPU evaluation")

        ledger = AuthorityLedger(ledger_path)
        if not ledger.verify():
            raise ValueError("Authority Ledger hash chain is invalid")
        events = ledger.read()
        exposure = _exposure_audit(
            events,
            clause_by_id,
            condition=condition,
            task_id=str(contract["task_id"]),
            target_domain=str(contract["target_domain"]),
            formal_clause_id=formal_clause_id,
        )
        if exposure.get("valid") is not True:
            raise ValueError(f"Formal exposure audit failed: {condition}")
        exposure_path = root / "EXPOSURE_AUDIT.json"
        write_json(exposure_path, exposure)

        descriptor_paths = _validate_writeback_descriptor(
            request.get("authority_writeback"),
            run_dir=run_dir,
            ledger_path=ledger_path,
            contract=contract,
            bundle=bundle,
        )

        failure_path: Path | None = None
        failure: dict[str, Any] = {}
        if selected_runtime_protocol_denial:
            failure_path = root / "CONDITION_FAILURE.json"
            failure = _runtime_protocol_denial_receipt(
                root,
                condition=condition,
                selected_node_id=selected_node_id,
                request=request,
                denial=selected_runtime_protocol_denial,
            )

        tracked_paths = [
            request_path,
            journal_path,
            ledger_path,
            rollout_path,
            exposure_path,
            runtime_receipt_path,
            *descriptor_paths,
            *audit_paths,
            *block_paths,
            *sorted(submission_dir.glob("submission_*.csv")),
        ]
        if failure_path is not None:
            tracked_paths.append(failure_path)
        file_hashes = {
            _relative_to(path, output_root): sha256_file(path)
            for path in tracked_paths
        }
        # The CPU evaluator may append terminal Result evidence only for a
        # protocol-legal selected node.  A blocked selected observation is a
        # retained Authority-denial outcome and its ledger is frozen too.
        for path in tracked_paths:
            if (
                path.resolve() != ledger_path.resolve()
                or selected_runtime_protocol_denial
            ):
                path.chmod(path.stat().st_mode & ~0o222)
        runtime_failed_admitted = {
            node_id
            for node_id in admitted
            if node_by_id[node_id].get("is_buggy") is True
        }
        common_row = {
            **base_row,
            "run_dir": str(run_dir),
            "journal_path": str(journal_path),
            "evaluation_request_path": str(request_path),
            "submission_dir": str(submission_dir),
            "selected_node_id": selected_node_id,
            "selection_basis": request.get("selection_basis") or {},
            "candidate_set_hash": request["candidate_set_hash"],
            "candidate_inventory": inventory,
            "candidate_execution_audit_count": len(audit_paths),
            "candidate_execution_admitted_node_ids": sorted(admitted),
            "candidate_execution_denied_node_ids": sorted(denied),
            "candidate_execution_submitted_node_ids": sorted(submitted),
            "candidate_execution_runtime_failed_admitted_node_ids": sorted(
                runtime_failed_admitted
            ),
            "selected_runtime_protocol_evidence": (
                selected_runtime_protocol_evidence
            ),
            "experience_exposure_count": exposure["exposure_event_count"],
            "formal_method_exposure_count": exposure[
                "formal_method_exposure_count"
            ],
            "exposure_audit_path": str(exposure_path),
            "authority_ledger_valid": True,
            "condition_runtime_receipt_hash": runtime_receipt["receipt_hash"],
            "condition_runtime_receipt_path": _relative_to(
                runtime_receipt_path, output_root
            ),
            "condition_runtime_receipt_sha256": sha256_file(
                runtime_receipt_path
            ),
            "file_hashes": dict(sorted(file_hashes.items())),
        }
        if selected_runtime_protocol_denial:
            condition_rows[condition] = {
                **common_row,
                "status": "pre_terminal_failure",
                "failure_classification": "authority_denial",
                "failure_receipt_path": _relative_to(
                    failure_path, output_root
                ),
                "failure_receipt_hash": failure["receipt_hash"],
                "failure_receipt_sha256": sha256_file(failure_path),
                "selected_runtime_protocol_denial": (
                    selected_runtime_protocol_denial
                ),
                "terminal_scoring_authorized": False,
                "candidate_reexecution_authorized": False,
            }
            continue
        condition_rows[condition] = {
            **common_row,
            "status": "training_complete_unscored",
        }

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "training_complete_unscored",
        "block_id": contract["block_id"],
        "task_id": contract["task_id"],
        "target_task_family": contract["target_task_family"],
        "target_domain": contract["target_domain"],
        "protocol_ref": contract["protocol_ref"],
        "split_id": contract["split_id"],
        "metric": contract["metric"],
        "maximize": contract["maximize"],
        "agent_seed": contract["agent_seed"],
        "condition_order": order,
        "conditions": condition_rows,
        "successful_condition_count": sum(
            row.get("status") == "training_complete_unscored"
            for row in condition_rows.values()
        ),
        "failed_condition_count": sum(
            row.get("status") == "pre_terminal_failure"
            for row in condition_rows.values()
        ),
        "steps_per_condition": contract["steps_per_condition"],
        "initial_drafts_per_condition": contract[
            "initial_drafts_per_condition"
        ],
        "repair_steps_budget_per_condition": (
            int(contract["steps_per_condition"])
            - int(contract["initial_drafts_per_condition"])
        ),
        "candidate_execution_contract": candidate_contract,
        "same_candidate_execution_contract": True,
        "same_source_snapshot": True,
        "same_bundle_binding": True,
        "terminal_scores_visible_during_search": False,
        "system_selection_frozen_before_terminal_evaluation": True,
        "legacy_static_coldstart_enabled": False,
        "protocol_payload_enforcement": protocol_payload_enforcement,
        "formal_tier2_evidence": False,
        "block_contract_hash": contract["contract_hash"],
        "block_contract_sha256": sha256_file(contract_path),
        "staging_manifest_hash": contract["staging_manifest_hash"],
        "staging_gate_hash": contract["staging_gate_hash"],
        "block_template_hash": contract["block_template_hash"],
        "source_snapshot_sha256": contract["source_snapshot_sha256"],
        "source_manifest_file_sha256": contract["source_manifest_file_sha256"],
        "container_image_digest": contract["container_image_digest"],
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "evaluator_manifest_sha256": contract["evaluator_manifest_sha256"],
        "training_pod_identity": training_pod_identity,
        "training_isolation_receipt_hash": training_isolation["receipt_hash"],
        "training_isolation_receipt_sha256": sha256_file(
            training_isolation_path
        ),
        "bundle_id": bundle.bundle_id,
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "manifest_hash": "",
    }
    if set(condition_rows) != set(CONDITIONS):
        raise ValueError("Formal training manifest is missing conditions")
    manifest["manifest_hash"] = _hash_payload(manifest, "manifest_hash")
    write_json(output_path, manifest)
    for path in (output_path, contract_path, training_isolation_path):
        path.chmod(path.stat().st_mode & ~0o222)
    bundle.assert_unchanged()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--block-contract", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_training_block(
        args.output_root,
        args.block_contract,
        args.bundle_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
