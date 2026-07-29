"""Host-owned terminal Result Fact writeback for fixed-holdout runs."""

from __future__ import annotations

import hashlib
import json
import dataclasses
import os
from pathlib import Path
from typing import Any, Mapping

from authority.authority_engine import AuthorityEngine
from authority.bundle_authority import restore_engine_snapshot
from authority.collectors import (
    CodeExecutionCollector,
    EvaluatorIntegrityCollector,
    FitScopeCollector,
    MethodIdentityCollector,
    PredictionScopeCollector,
    SelectionFreezeCollector,
    SplitLineageCollector,
    TrustedCollectorHost,
)
from authority.evidence_graph import EvidencePath
from authority.ledger import AuthorityLedger
from authority.memory_snapshot import SessionOverlay, write_json_atomic
from authority.models import (
    AuthorityDecision,
    AuthorityRequest,
    AuthorityScope,
    Claim,
    ClaimType,
    DecisionOutcome,
    DecisionStage,
    Operation,
    ProtocolRef,
    ProtocolSpec,
    Receipt,
    TaskContext,
)
from authority.protocol_registry import ProtocolRegistry, canonical_json
from authority.stage_ontology import GovernanceStage, resolve_stage_axes
from fixed_holdout.common import read_manifest, sha256_file
from fixed_holdout.formal_runtime import (
    validate_selected_runtime_protocol_evidence,
)


STATUS_SCHEMA = "fixed_holdout_terminal_writeback_status_v1"
RESULT_EVENT_SCHEMA = "fixed_holdout_result_fact_v1"
FORMAL_TRAINING_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_training_manifest_v1"
)
TERMINAL_PROTOCOL_EVIDENCE_SCHEMA = (
    "fixed_holdout_terminal_protocol_evidence_v1"
)

_FORMAL_SPLIT_STRATEGY_MAP = {
    "stratified_random": "stratified_random",
    "grouped_multilabel_stratified": "grouped",
    "chronological_deterministic_sha256_sample": "chronological",
}


class TerminalWritebackError(RuntimeError):
    pass


def _hash_payload(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(canonical_json(unsigned).encode("utf-8")).hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _protocol_ref(payload: Mapping[str, Any]) -> ProtocolRef:
    return ProtocolRef(
        protocol_id=str(payload["protocol_id"]),
        version=str(payload["version"]),
        canonical_hash=str(payload["canonical_hash"]),
    )


def _validate_prefix(
    ledger: AuthorityLedger,
    *,
    expected_count: int,
    expected_last_hash: str,
) -> list[dict[str, Any]]:
    events = ledger.read()
    if not ledger.verify(events):
        raise ValueError("Authority ledger hash chain is invalid")
    if len(events) < expected_count:
        raise ValueError("Authority ledger is shorter than the sealed prefix")
    prefix = events[:expected_count]
    observed_last = str(prefix[-1]["event_hash"]) if prefix else ""
    if observed_last != expected_last_hash:
        raise ValueError("Authority ledger sealed prefix mismatch")
    return events


def _decision_from_payload(payload: Mapping[str, Any]) -> AuthorityDecision:
    values = dict(payload)
    values["outcome"] = DecisionOutcome(str(values["outcome"]))
    scope = values.get("permitted_scope")
    values["permitted_scope"] = (
        AuthorityScope(**scope) if isinstance(scope, dict) else None
    )
    return AuthorityDecision(**values)


def _prior_decision(
    events: list[dict[str, Any]],
    *,
    claim_id: str,
    artifact_id: str,
) -> AuthorityDecision | None:
    for event in reversed(events):
        if event.get("event_type") != "authority_decision":
            continue
        payload = event.get("payload") or {}
        request = payload.get("request") or {}
        if (
            str(request.get("claim_id") or "") == claim_id
            and str(request.get("artifact_id") or "") == artifact_id
            and str(request.get("operation") or "")
            == Operation.PROMOTE_RESULT.value
        ):
            decision = payload.get("decision")
            if isinstance(decision, dict):
                return _decision_from_payload(decision)
    return None


def _terminal_result_fields(
    report: Mapping[str, Any],
) -> tuple[str, float, str, str]:
    """Return the frozen system result, never the post-hoc Oracle choice."""

    schema = str(report.get("report_schema") or "")
    if schema == "fixed_holdout_terminal_score_report_v3":
        return (
            str(report.get("selected_node_id") or ""),
            float(report["selected_score"]),
            str(report.get("selected_submission_sha256") or ""),
            str(report.get("selected_submission") or ""),
        )
    if schema == "fixed_holdout_terminal_score_report_v2":
        return (
            str(report.get("best_node_id") or ""),
            float(report["best_score"]),
            str(report.get("best_submission_sha256") or ""),
            "",
        )
    raise ValueError("Unsupported fixed-holdout terminal score report")


def _validated_formal_receipt(
    manifest: Mapping[str, Any],
    name: str,
    *,
    schema: str,
    protocol_ref: ProtocolRef,
) -> dict[str, Any]:
    receipt = manifest.get(name)
    if not isinstance(receipt, dict):
        raise ValueError(f"Formal manifest lacks {name}")
    if receipt.get("schema") != schema:
        raise ValueError(f"Formal {name} schema mismatch")
    if receipt.get("receipt_hash") != _hash_payload(
        receipt, "receipt_hash"
    ):
        raise ValueError(f"Formal {name} hash mismatch")
    expected = {
        "task_id": manifest.get("task_id"),
        "split_id": manifest.get("split_id"),
        "protocol_ref": protocol_ref.key(),
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(f"Formal {name} binding mismatch: {field}")
    return dict(receipt)


def _load_formal_runtime_evidence(
    training_manifest_path: Path,
    condition: str,
    *,
    request_path: Path,
    request: Mapping[str, Any],
    report: Mapping[str, Any],
    node: Mapping[str, Any],
    protocol_ref: ProtocolRef,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = _read_object(
        training_manifest_path, label="formal training manifest"
    )
    if training.get("schema") != FORMAL_TRAINING_SCHEMA:
        raise ValueError("Formal training manifest schema mismatch")
    if training.get("manifest_hash") != _hash_payload(
        training, "manifest_hash"
    ):
        raise ValueError("Formal training manifest hash mismatch")
    if training.get("status") != "training_complete_unscored":
        raise ValueError("Formal training manifest is not pre-evaluator frozen")
    if training.get("protocol_payload_enforcement") is not True:
        raise ValueError("Formal training did not enforce ProtocolSpec payloads")
    expected = {
        "task_id": report.get("task_id"),
        "protocol_ref": protocol_ref.key(),
        "split_id": report.get("split_id"),
        "metric": report.get("metric"),
        "maximize": report.get("maximize"),
    }
    for field, value in expected.items():
        if training.get(field) != value:
            raise ValueError(
                f"Formal training/terminal protocol mismatch: {field}"
            )
    conditions = training.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError("Formal training manifest lacks condition rows")
    row = conditions.get(condition)
    if not isinstance(row, dict):
        raise ValueError("Formal training manifest lacks the active condition")
    if row.get("status") != "training_complete_unscored":
        raise ValueError("Formal condition is not ready for terminal writeback")
    if row.get("selected_node_id") != node.get("id"):
        raise ValueError("Formal selected runtime evidence targets another node")
    if Path(str(row.get("evaluation_request_path") or "")).resolve() != (
        request_path
    ):
        raise ValueError("Formal training request path binding mismatch")
    if Path(str(row.get("journal_path") or "")).resolve() != Path(
        str(request.get("journal_path") or "")
    ).resolve():
        raise ValueError("Formal training journal path binding mismatch")
    evidence = row.get("selected_runtime_protocol_evidence")
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError("Formal selected node lacks runtime protocol evidence")
    verified = validate_selected_runtime_protocol_evidence(
        evidence,
        node,
        protocol_ref=protocol_ref.key(),
    )
    binding = {
        "condition": str(condition),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "training_manifest_hash": str(training["manifest_hash"]),
        "runtime_evidence_hash": str(verified["evidence_hash"]),
        "runtime_observation_sha256": str(
            verified["observation_sha256"]
        ),
    }
    return verified, binding


def _formal_terminal_protocol_evidence(
    *,
    protocol_spec: ProtocolSpec,
    protocol_ref: ProtocolRef,
    report: Mapping[str, Any],
    train_manifest: Mapping[str, Any],
    evaluator_manifest: Mapping[str, Any],
    train_manifest_path: Path,
    evaluator_manifest_path: Path,
    runtime_evidence: Mapping[str, Any],
    training_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if train_manifest.get("protocol_ref") != protocol_ref.key() or (
        evaluator_manifest.get("protocol_ref") != protocol_ref.key()
    ):
        raise ValueError("Fixed-holdout manifest protocol binding mismatch")
    for name in (
        "split_receipt",
        "fit_scope_receipt",
        "metric_spec_receipt",
    ):
        if train_manifest.get(name) != evaluator_manifest.get(name):
            raise ValueError(f"Train/evaluator formal Receipt mismatch: {name}")

    split_receipt = _validated_formal_receipt(
        evaluator_manifest,
        "split_receipt",
        schema="formal_split_lineage_receipt_v1",
        protocol_ref=protocol_ref,
    )
    fit_receipt = _validated_formal_receipt(
        evaluator_manifest,
        "fit_scope_receipt",
        schema="formal_fit_scope_receipt_v1",
        protocol_ref=protocol_ref,
    )
    metric_receipt = _validated_formal_receipt(
        evaluator_manifest,
        "metric_spec_receipt",
        schema="formal_metric_spec_receipt_v1",
        protocol_ref=protocol_ref,
    )

    formal_strategy = str(split_receipt.get("strategy") or "")
    canonical_strategy = _FORMAL_SPLIT_STRATEGY_MAP.get(formal_strategy)
    expected_strategy = str(
        protocol_spec.data_split_policy.get("strategy") or ""
    )
    if canonical_strategy is None or canonical_strategy != expected_strategy:
        raise ValueError(
            "Formal split strategy is not a verified ProtocolSpec mapping"
        )
    if split_receipt.get("terminal_labels_absent_from_train_view") is not True:
        raise ValueError("Formal split Receipt does not prove label isolation")
    split_flags: dict[str, Any] = {"split_strategy": canonical_strategy}
    if canonical_strategy == "stratified_random":
        if (
            split_receipt.get("stratification_verified") is not True
            or int(split_receipt.get("overlap_count", -1)) != 0
        ):
            raise ValueError("Formal stratified split Receipt is incomplete")
        split_flags["stratification_verified"] = True
    elif canonical_strategy == "grouped":
        if (
            int(split_receipt.get("group_overlap_count", -1)) != 0
            or int(split_receipt.get("record_overlap_count", -1)) != 0
        ):
            raise ValueError("Formal grouped split Receipt contains overlap")
        split_flags["group_overlap_count"] = 0
    elif canonical_strategy == "chronological":
        if (
            int(split_receipt.get("future_to_past_count", -1)) != 0
            or int(
                split_receipt.get("train_holdout_key_overlap_count", -1)
            )
            != 0
            or not str(split_receipt.get("max_train_pickup_datetime") or "")
            < str(split_receipt.get("min_holdout_pickup_datetime") or "")
        ):
            raise ValueError(
                "Formal chronological split Receipt is not strictly ordered"
            )
        split_flags.update(
            {
                "future_to_past_count": 0,
                "chronological_order_verified": True,
            }
        )

    if (
        fit_receipt.get("fit_scope") != "train_view_only"
        or fit_receipt.get("verified") is not True
        or int(fit_receipt.get("holdout_fit_count", -1)) != 0
        or (fit_receipt.get("fit_scope_hashes") or {}).get(
            "train_view_input"
        )
        != evaluator_manifest.get("public_tree_sha256")
    ):
        raise ValueError("Formal terminal fit-scope Receipt is incomplete")
    internal_fit_scope = str(
        protocol_spec.preprocessing_policy.get("fit_scope") or ""
    )
    if not internal_fit_scope:
        raise ValueError("ProtocolSpec lacks an internal preprocessing fit scope")

    direction = "maximize" if bool(report["maximize"]) else "minimize"
    metric_name = str(report.get("metric") or "")
    if (
        metric_receipt.get("metric") != metric_name
        or metric_receipt.get("direction") != direction
        or metric_receipt.get("terminal_only") is not True
        or metric_receipt.get("verified") is not True
        or metric_receipt.get("evaluator_module_sha256")
        != sha256_file(Path(__file__).with_name("evaluate.py"))
        or protocol_spec.metric_spec.get("name") != metric_name
        or protocol_spec.metric_spec.get("direction") != direction
    ):
        raise ValueError("Formal terminal metric Receipt/ProtocolSpec mismatch")

    evidence: dict[str, Any] = {
        "schema": TERMINAL_PROTOCOL_EVIDENCE_SCHEMA,
        "condition": str(training_binding["condition"]),
        "protocol_ref": protocol_ref.key(),
        "formal_split_strategy": formal_strategy,
        "canonical_split_strategy": canonical_strategy,
        "internal_fit_scope": internal_fit_scope,
        "terminal_fit_scope": "train_view_only",
        "metric_name": metric_name,
        "metric_direction": direction,
        "training_manifest_sha256": str(
            training_binding["training_manifest_sha256"]
        ),
        "training_manifest_hash": str(
            training_binding["training_manifest_hash"]
        ),
        "runtime_evidence_hash": str(
            training_binding["runtime_evidence_hash"]
        ),
        "runtime_observation_sha256": str(
            training_binding["runtime_observation_sha256"]
        ),
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "evaluator_manifest_sha256": sha256_file(
            evaluator_manifest_path
        ),
        "split_receipt_hash": str(split_receipt["receipt_hash"]),
        "fit_scope_receipt_hash": str(fit_receipt["receipt_hash"]),
        "metric_spec_receipt_hash": str(metric_receipt["receipt_hash"]),
        "internal_runtime_verified": bool(runtime_evidence),
        "terminal_manifest_verified": True,
        "receipt_semantics_separated": True,
        "evidence_join_sha256": "",
    }
    evidence["evidence_join_sha256"] = _hash_payload(
        evidence, "evidence_join_sha256"
    )
    return evidence, split_flags


def _trusted_terminal_receipts(
    *,
    host: TrustedCollectorHost,
    protocol_ref: ProtocolRef,
    protocol_spec: ProtocolSpec,
    run_id: str,
    node: Mapping[str, Any],
    report: Mapping[str, Any],
    train_manifest: Mapping[str, Any],
    evaluator_manifest: Mapping[str, Any],
    journal_hash: str,
    train_manifest_path: Path,
    evaluator_manifest_path: Path,
    runtime_evidence: Mapping[str, Any],
    training_binding: Mapping[str, Any],
) -> tuple[list[Receipt], str]:
    artifact_id = str(node["id"])
    code = str(node.get("code") or "")
    if not code:
        raise ValueError("Selected fixed-holdout node has no code")
    if node.get("exec_time") is None:
        raise ValueError("Selected fixed-holdout node lacks execution evidence")
    if node.get("is_buggy") is True or node.get("is_valid") is False:
        raise ValueError("Selected fixed-holdout node is not a valid execution")
    code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest()
    declared_method = str(node.get("method_fingerprint") or "")
    method_hash = declared_method if len(declared_method) == 64 else code_hash
    _, _, selected_submission_hash, _ = _terminal_result_fields(report)
    run_hash = hashlib.sha256(
        canonical_json(
            {
                "run_id": run_id,
                "artifact_id": artifact_id,
                "journal_sha256": journal_hash,
                "code_sha256": code_hash,
                "submission_sha256": selected_submission_hash,
            }
        ).encode("utf-8")
    ).hexdigest()
    direction = "maximize" if bool(report["maximize"]) else "minimize"
    terminal_protocol_evidence: dict[str, Any] = {}
    split_flags: dict[str, Any] = {}
    if protocol_spec.promotion_policy.get("enforce_protocol_payloads") is True:
        if not runtime_evidence or not training_binding:
            raise ValueError(
                "Enforced ProtocolSpec terminal writeback lacks formal runtime evidence"
            )
        terminal_protocol_evidence, split_flags = (
            _formal_terminal_protocol_evidence(
                protocol_spec=protocol_spec,
                protocol_ref=protocol_ref,
                report=report,
                train_manifest=train_manifest,
                evaluator_manifest=evaluator_manifest,
                train_manifest_path=train_manifest_path,
                evaluator_manifest_path=evaluator_manifest_path,
                runtime_evidence=runtime_evidence,
                training_binding=training_binding,
            )
        )
    terminal_evidence_payload = (
        {"terminal_protocol_evidence": terminal_protocol_evidence}
        if terminal_protocol_evidence
        else {}
    )
    evaluator_code_hash = sha256_file(Path(__file__).with_name("evaluate.py"))
    evaluator_hash = hashlib.sha256(
        canonical_json(
            {
                "evaluator_code_sha256": evaluator_code_hash,
                "evaluator_manifest_sha256": sha256_file(
                    evaluator_manifest_path
                ),
                "metric": report["metric"],
                "direction": direction,
                "terminal_protocol_evidence_sha256": (
                    terminal_protocol_evidence.get("evidence_join_sha256", "")
                ),
            }
        ).encode("utf-8")
    ).hexdigest()
    inputs_hash = hashlib.sha256(
        canonical_json(
            {
                "submission_sha256": selected_submission_hash,
                "labels_sha256": evaluator_manifest["labels_sha256"],
                "holdout_id_sha256": evaluator_manifest[
                    "holdout_id_sha256"
                ],
                "score_report_hash": report["report_hash"],
                "terminal_protocol_evidence_sha256": (
                    terminal_protocol_evidence.get("evidence_join_sha256", "")
                ),
            }
        ).encode("utf-8")
    ).hexdigest()
    partition_hashes = {
        "public_train_view": evaluator_manifest["public_tree_sha256"],
        "fixed_holdout_ids": evaluator_manifest["holdout_id_sha256"],
    }
    fit_scope_hashes = {
        "label_isolated_train_view": evaluator_manifest[
            "public_tree_sha256"
        ]
    }
    prediction_scope_hashes = {
        "fixed_holdout_submission": selected_submission_hash
    }
    fit_scope_payload: dict[str, Any] = {}
    if terminal_protocol_evidence:
        partition_hashes.update(
            {
                "formal_split_receipt": terminal_protocol_evidence[
                    "split_receipt_hash"
                ],
                "internal_runtime_split_scope": runtime_evidence[
                    "split_scope_hashes_sha256"
                ],
            }
        )
        fit_scope_hashes.update(
            {
                "formal_terminal_fit_scope_receipt": (
                    terminal_protocol_evidence["fit_scope_receipt_hash"]
                ),
                "internal_runtime_fit_scope": runtime_evidence[
                    "fit_scope_hashes_sha256"
                ],
            }
        )
        prediction_scope_hashes["internal_runtime_prediction_scope"] = (
            runtime_evidence["prediction_scope_hashes_sha256"]
        )
        fit_scope_payload = {
            "fit_scope": terminal_protocol_evidence[
                "internal_fit_scope"
            ],
            "terminal_fit_scope": terminal_protocol_evidence[
                "terminal_fit_scope"
            ],
        }
    specifications = (
        (
            MethodIdentityCollector,
            {
                "method_fingerprint": method_hash,
                "code_sha256": code_hash,
            },
        ),
        (
            CodeExecutionCollector,
            {
                "exit_status": 0,
                "executed_path": artifact_id,
                "run_hash": run_hash,
            },
        ),
        (
            SplitLineageCollector,
            {
                "partition_hashes": partition_hashes,
                "overlap_count": 0,
                **split_flags,
                **terminal_evidence_payload,
            },
        ),
        (
            FitScopeCollector,
            {
                "fit_scope_hashes": fit_scope_hashes,
                "holdout_fit_count": 0,
                **fit_scope_payload,
                **terminal_evidence_payload,
            },
        ),
        (
            PredictionScopeCollector,
            {
                "prediction_scope_hashes": prediction_scope_hashes,
                "forbidden_overlap_count": 0,
                **terminal_evidence_payload,
            },
        ),
        (
            EvaluatorIntegrityCollector,
            {
                "evaluator_hash": evaluator_hash,
                "inputs_hash": inputs_hash,
                "metric_name": str(report["metric"]),
                "metric_direction": direction,
                "tampered": False,
                **terminal_evidence_payload,
            },
        ),
        (
            SelectionFreezeCollector,
            {
                "candidate_set_hash": report["candidate_set_hash"],
                "frozen_before_holdout": True,
                **terminal_evidence_payload,
            },
        ),
    )
    receipts = [
        host.collect(
            collector,
            artifact_id=artifact_id,
            run_id=run_id,
            protocol_ref=protocol_ref,
            source="host.fixed_holdout_terminal_scorer",
            payload=payload,
        )
        for collector, payload in specifications
    ]
    return receipts, code_hash


def _complete_status(payload: dict[str, Any]) -> dict[str, Any]:
    value = {
        "schema": STATUS_SCHEMA,
        "status": "complete",
        **payload,
        "status_hash": "",
    }
    value["status_hash"] = _hash_payload(value, "status_hash")
    return value


def record_terminal_writeback_failure(
    status_path: str | Path,
    error: Exception,
    *,
    request_path: str | Path | None = None,
    score_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist a hash-bound, explicit failure for host terminal closure."""

    reason = str(error)
    lowered = reason.lower()
    if "contract" in lowered or "protocolref" in lowered:
        reason_code = "contract_mismatch"
        component = "terminal_contract_join"
    elif any(
        marker in lowered
        for marker in ("collector", "payload:", "receipt", "schema", "manifest")
    ):
        reason_code = "collector_internal_error"
        component = "terminal_host_collector"
    else:
        reason_code = "terminal_writeback_error"
        component = "terminal_writeback"
    incomplete = {
        "schema": STATUS_SCHEMA,
        "status": "writeback_incomplete",
        "error_type": type(error).__name__,
        "reason": reason,
        "reason_codes": [reason_code],
        "responsible_component": component,
        "repairable": False,
        "authority_disposition": "quarantine",
        "request_path": str(Path(request_path).resolve()) if request_path else "",
        "score_report_path": (
            str(Path(score_report_path).resolve()) if score_report_path else ""
        ),
        "status_hash": "",
    }
    incomplete["status_hash"] = _hash_payload(incomplete, "status_hash")
    write_json_atomic(Path(status_path).resolve(), incomplete)
    return incomplete


def _seal_authority_snapshot(
    canonical_path: Path,
    snapshot: Mapping[str, Any],
) -> dict[str, str]:
    """Persist one immutable content-addressed terminal Authority snapshot."""

    write_json_atomic(canonical_path, snapshot)
    payload = canonical_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    directory = canonical_path.parent / "authority_snapshots"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"authority_snapshot.{digest}.json"
    try:
        with target.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if target.read_bytes() != payload:
            raise ValueError("Content-addressed terminal snapshot collision")
    return {
        "schema": "authority_snapshot_pointer_v1",
        "path": str(target.resolve()),
        "sha256": digest,
    }


def finalize_result_writeback(
    evaluation_request_path: str | Path,
    score_report_path: str | Path,
    evaluator_manifest_path: str | Path,
    *,
    status_path: str | Path | None = None,
    formal_training_manifest_path: str | Path | None = None,
    formal_condition: str | None = None,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Write exactly one terminal Result Fact after hidden-label scoring."""

    request_path = Path(evaluation_request_path).resolve()
    report_path = Path(score_report_path).resolve()
    manifest_path = Path(evaluator_manifest_path).resolve()
    formal_training_path = (
        Path(formal_training_manifest_path).resolve()
        if formal_training_manifest_path is not None
        else None
    )
    status_target = (
        Path(status_path).resolve()
        if status_path is not None
        else report_path.with_name("fixed_holdout_writeback_status.json")
    )
    try:
        request = _read_object(request_path, label="evaluation request")
        report = _read_object(report_path, label="terminal score report")
        request_schema = str(request.get("request_schema") or "")
        report_schema = str(report.get("report_schema") or "")
        if request_schema not in {
            "fixed_holdout_evaluation_request_v2",
            "fixed_holdout_evaluation_request_v3",
        }:
            raise ValueError("Unsupported fixed-holdout evaluation request")
        if request.get("status") != "awaiting_external_evaluator":
            raise ValueError("Fixed-holdout request is not awaiting evaluation")
        if request.get("request_hash") != _hash_payload(
            request, "request_hash"
        ):
            raise ValueError("Fixed-holdout evaluation request hash mismatch")
        expected_report_schema = {
            "fixed_holdout_evaluation_request_v2": (
                "fixed_holdout_terminal_score_report_v2"
            ),
            "fixed_holdout_evaluation_request_v3": (
                "fixed_holdout_terminal_score_report_v3"
            ),
        }[request_schema]
        if report_schema != expected_report_schema:
            raise ValueError("Unsupported fixed-holdout terminal score report")
        if report.get("report_hash") != _hash_payload(report, "report_hash"):
            raise ValueError("Fixed-holdout score report hash mismatch")
        if report.get("terminal_score_sealed") is not True:
            raise ValueError("Fixed-holdout terminal score is not sealed")
        if report.get("candidate_set_frozen_before_scoring") is not True:
            raise ValueError("Fixed-holdout candidate set was not frozen")
        if report.get("scores_were_visible_during_search") is not False:
            raise ValueError("Fixed-holdout scores were visible during search")
        if request.get("scores_were_visible_during_search") is not False:
            raise ValueError("Evaluation request exposes fixed-holdout scores")
        if request.get("selection_policy") != "terminal_only":
            raise ValueError("Evaluation request is not terminal-only")
        if report.get("selection_policy") != "terminal_only":
            raise ValueError("Score report is not terminal-only")
        if request_schema == "fixed_holdout_evaluation_request_v3":
            if request.get("selection_frozen_before_terminal_evaluation") is not True:
                raise ValueError("System selection was not frozen before evaluation")
            if report.get("selection_frozen_before_terminal_evaluation") is not True:
                raise ValueError("Score report lacks the pre-evaluator selection freeze")
            if report.get("system_selection_used_terminal_labels") is not False:
                raise ValueError("System selection used terminal labels")
            if report.get("oracle_selection_is_host_only") is not True:
                raise ValueError("Terminal-best selection is not isolated to the Oracle")
            if report.get("evaluation_request_hash") != request.get(
                "request_hash"
            ):
                raise ValueError("Score report is not bound to the evaluation request")
            if report.get("evaluation_request_sha256") != sha256_file(
                request_path
            ):
                raise ValueError("Evaluation request file changed after scoring")
            for key in (
                "candidate_set_hash",
                "selected_node_id",
                "selected_submission",
                "selected_submission_sha256",
                "selection_basis",
            ):
                if request.get(key) != report.get(key):
                    raise ValueError(f"Selection freeze request/report mismatch: {key}")

        evaluator_manifest = read_manifest(
            manifest_path, expected_role="evaluator_view"
        )
        train_manifest_path = (
            manifest_path.parent.parent
            / "train_view"
            / "fixed_holdout_manifest.json"
        )
        train_manifest = read_manifest(
            train_manifest_path, expected_role="train_view"
        )
        for key in (
            "task_id",
            "split_id",
            "public_tree_sha256",
            "holdout_id_sha256",
            "selection_policy",
        ):
            if evaluator_manifest.get(key) != train_manifest.get(key):
                raise ValueError(f"Fixed-holdout manifest mismatch: {key}")
        if train_manifest.get("hidden_labels_present") is not False:
            raise ValueError("Training view contains hidden labels")
        if sha256_file(manifest_path) != report.get(
            "evaluator_manifest_sha256"
        ):
            raise ValueError("Evaluator manifest hash mismatch")
        if sha256_file(train_manifest_path) != report.get(
            "train_manifest_sha256"
        ):
            raise ValueError("Train manifest hash mismatch in score report")
        if sha256_file(train_manifest_path) != request.get(
            "train_manifest_sha256"
        ):
            raise ValueError("Train manifest hash mismatch in request")
        for key in ("task_id", "split_id", "metric", "maximize"):
            if report.get(key) != request.get(key):
                raise ValueError(f"Request/report mismatch: {key}")

        journal_path = Path(str(request["journal_path"])).resolve()
        journal_hash = sha256_file(journal_path)
        if journal_hash not in {
            str(request.get("journal_sha256") or ""),
            str(report.get("journal_sha256") or ""),
        } or (
            journal_hash != str(request.get("journal_sha256") or "")
            or journal_hash != str(report.get("journal_sha256") or "")
        ):
            raise ValueError("Journal changed after terminal handoff")
        journal = _read_object(journal_path, label="journal")
        nodes = {
            str(node.get("id")): node
            for node in journal.get("nodes") or []
            if isinstance(node, dict) and node.get("id")
        }
        (
            artifact_id,
            selected_score,
            selected_submission_sha256,
            selected_submission_name,
        ) = _terminal_result_fields(report)
        if not artifact_id or artifact_id not in nodes:
            raise ValueError("Pre-evaluator selection references an unknown node")
        selected_rows = [
            row
            for row in report.get("results") or []
            if row.get("node_id") == artifact_id
        ]
        if len(selected_rows) != 1 or selected_rows[0].get("status") != "scored":
            raise ValueError("Selected fixed-holdout node is not uniquely scored")
        if selected_rows[0].get("submission_sha256") != selected_submission_sha256:
            raise ValueError("Selected submission hash mismatch")
        if selected_submission_name and selected_rows[0].get(
            "submission"
        ) != selected_submission_name:
            raise ValueError("Selected submission name mismatch")
        submission_path = (
            Path(str(request["submission_dir"]))
            / str(selected_rows[0]["submission"])
        ).resolve()
        if sha256_file(submission_path) != selected_submission_sha256:
            raise ValueError("Selected submission changed after scoring")

        descriptor = request.get("authority_writeback") or {}
        if descriptor.get("schema") != (
            "fixed_holdout_authority_writeback_descriptor_v1"
        ):
            raise ValueError("Authority writeback descriptor schema mismatch")
        if descriptor.get("status") != "ready":
            raise ValueError("Authority writeback descriptor is incomplete")
        if descriptor.get("descriptor_hash") != _hash_payload(
            descriptor, "descriptor_hash"
        ):
            raise ValueError("Authority writeback descriptor hash mismatch")
        if descriptor.get("task_id") != report.get("task_id"):
            raise ValueError("Authority descriptor task mismatch")

        snapshot_path = Path(
            str(descriptor["authority_snapshot_path"])
        ).resolve()
        if sha256_file(snapshot_path) != descriptor.get(
            "authority_snapshot_sha256"
        ):
            raise ValueError("Authority snapshot changed after handoff")
        registry = ProtocolRegistry(descriptor["protocol_registry_path"])
        protocol_ref = _protocol_ref(descriptor["active_protocol"])
        protocol_spec = registry.resolve(protocol_ref)
        runtime_evidence: dict[str, Any] = {}
        training_binding: dict[str, Any] = {}
        protocol_payloads_enforced = (
            protocol_spec.promotion_policy.get("enforce_protocol_payloads")
            is True
        )
        if protocol_payloads_enforced:
            if formal_training_path is None or not formal_condition:
                raise ValueError(
                    "Enforced ProtocolSpec writeback requires the frozen formal "
                    "training manifest and condition"
                )
            runtime_evidence, training_binding = (
                _load_formal_runtime_evidence(
                    formal_training_path,
                    str(formal_condition),
                    request_path=request_path,
                    request=request,
                    report=report,
                    node=nodes[artifact_id],
                    protocol_ref=protocol_ref,
                )
            )
        elif (formal_training_path is None) != (not formal_condition):
            raise ValueError(
                "Formal training manifest and condition must be supplied together"
            )
        ledger = AuthorityLedger(descriptor["authority_ledger_path"])
        ledger_events = _validate_prefix(
            ledger,
            expected_count=int(descriptor["authority_ledger_event_count"]),
            expected_last_hash=str(
                descriptor["authority_ledger_last_event_hash"]
            ),
        )
        engine = AuthorityEngine(
            registry,
            ledger=ledger,
            policy_version=str(descriptor["policy_version"]),
        )
        restore_report = restore_engine_snapshot(
            engine,
            _read_object(snapshot_path, label="authority snapshot"),
        )
        overlay = SessionOverlay(
            descriptor["session_overlay_path"],
            overlay_id=descriptor["session_overlay_id"],
        )
        frozen_overlay_manifest_hash = str(
            descriptor["session_overlay_manifest_sha256"]
        )
        initial_count = int(descriptor["authority_ledger_event_count"])
        idempotency_key = hashlib.sha256(
            canonical_json(
                {
                    "run_id": descriptor["run_id"],
                    "artifact_id": artifact_id,
                    "protocol_hash": protocol_ref.canonical_hash,
                    "publication_class": "result_fact",
                    "terminal_report_hash": report["report_hash"],
                }
            ).encode("utf-8")
        ).hexdigest()
        overlay_events = overlay.events()
        existing = [
            event
            for event in overlay_events
            if event.event_type == "memory_claim"
            and event.payload.get("idempotency_key") == idempotency_key
        ]
        if existing:
            if len(existing) != 1:
                raise ValueError("Duplicate terminal Result Fact events detected")
            status = _complete_status(
                {
                    "completion": "already_finalized",
                    "artifact_id": artifact_id,
                    "idempotency_key": idempotency_key,
                    "overlay_event_id": existing[0].event_id,
                    "overlay_event_hash": existing[0].event_hash,
                    "terminal_report_hash": report["report_hash"],
                }
            )
            write_json_atomic(status_target, status)
            return status
        if (
            overlay.manifest.get("manifest_sha256")
            != frozen_overlay_manifest_hash
        ):
            raise ValueError("Session Overlay changed before terminal writeback")

        node = nodes[artifact_id]
        host = TrustedCollectorHost(
            f"fixed-holdout:{descriptor['run_id']}",
            collector_version=str(descriptor["collector_version"]),
        )
        receipts, code_hash = _trusted_terminal_receipts(
            host=host,
            protocol_ref=protocol_ref,
            protocol_spec=protocol_spec,
            run_id=str(descriptor["run_id"]),
            node=node,
            report=report,
            train_manifest=train_manifest,
            evaluator_manifest=evaluator_manifest,
            journal_hash=journal_hash,
            train_manifest_path=train_manifest_path,
            evaluator_manifest_path=manifest_path,
            runtime_evidence=runtime_evidence,
            training_binding=training_binding,
        )
        claim_id = (
            f"node:{artifact_id}:terminal-result-score:"
            f"{report['report_hash'][:24]}"
        )
        claim = Claim(
            claim_id=claim_id,
            claim_type=ClaimType.SCORE,
            subject_artifact_id=artifact_id,
            task_scope={"task_id": str(report["task_id"])},
            method_fingerprint=code_hash,
            protocol_ref=protocol_ref,
            statement=(
                f"Terminal fixed-holdout {report['metric']}="
                f"{selected_score} for artifact {artifact_id}."
            ),
            source_artifact_refs=[artifact_id],
            evidence_refs=[report["report_hash"]],
            boundary={
                "terminal_result": True,
                "terminal_report_hash": report["report_hash"],
                "split_id": report["split_id"],
                "metric": report["metric"],
                "score": selected_score,
                "code_sha256": code_hash,
            },
        )
        engine.graph.add_claim(claim)
        for receipt in receipts:
            engine.graph.add_receipt(receipt)
        path = EvidencePath(
            path_id=(
                f"path:{claim_id}:{protocol_ref.canonical_hash[:12]}"
            ),
            claim_id=claim_id,
            receipt_ids=[receipt.receipt_id for receipt in receipts],
        )
        engine.graph.add_path(path)
        prepared_payload = {
            "idempotency_key": idempotency_key,
            "artifact_id": artifact_id,
            "claim_id": claim_id,
            "receipt_ids": [receipt.receipt_id for receipt in receipts],
            "path_id": path.path_id,
            "terminal_report_hash": report["report_hash"],
        }
        if not any(
            event.get("event_type") == "terminal_result_writeback_prepared"
            and (event.get("payload") or {}).get("idempotency_key")
            == idempotency_key
            for event in ledger_events[initial_count:]
        ):
            ledger.append("terminal_result_writeback_prepared", prepared_payload)
            for receipt in receipts:
                ledger.append(
                    "terminal_result_receipt_written",
                    dataclasses.asdict(receipt),
                )
            ledger.append(
                "terminal_result_claim_created", dataclasses.asdict(claim)
            )
            ledger.append(
                "terminal_result_path_created", dataclasses.asdict(path)
            )

        decision = _prior_decision(
            ledger.read(), claim_id=claim_id, artifact_id=artifact_id
        )
        if decision is None:
            axes = resolve_stage_axes(runtime_stage=node.get("stage"))
            decision = engine.authorize(
                AuthorityRequest(
                    artifact_id=artifact_id,
                    claim_id=claim_id,
                    operation=Operation.PROMOTE_RESULT,
                    decision_stage=DecisionStage.MEMORY_WRITEBACK,
                    active_protocol=protocol_ref,
                    task_context=TaskContext(task_id=str(report["task_id"])),
                    requesting_component="fixed_holdout.terminal_scorer",
                    generation_stage=axes.generation_stage,
                    governance_stage=GovernanceStage.MEMORY_WRITEBACK,
                )
            )
        else:
            engine.decisions[decision.decision_id] = decision
        if not decision.allowed:
            raise ValueError(
                "Terminal Result Fact Authority denied: "
                + ",".join(decision.missing_obligations)
            )

        terminal_snapshot = engine.snapshot()
        terminal_snapshot["restore_report"] = restore_report
        terminal_snapshot["terminal_writeback_idempotency_key"] = (
            idempotency_key
        )
        authority_snapshot_pointer = _seal_authority_snapshot(
            snapshot_path.with_name("authority_terminal_snapshot.json"),
            terminal_snapshot,
        )

        event = overlay.append(
            "memory_claim",
            {
                "schema": RESULT_EVENT_SCHEMA,
                "idempotency_key": idempotency_key,
                "artifact_id": artifact_id,
                "task_id": report["task_id"],
                "protocol_ref": protocol_ref.key(),
                "authority_policy_version": descriptor["policy_version"],
                "claim_refs": [claim_id],
                "claim_types": [ClaimType.SCORE.value],
                "receipt_refs": [receipt.receipt_id for receipt in receipts],
                "authority_decision_refs": [decision.decision_id],
                "authority_snapshot_pointer": authority_snapshot_pointer,
                "artifact_pointer": {
                    "journal_path": str(journal_path),
                    "journal_sha256": journal_hash,
                    "node_id": artifact_id,
                },
                "code_sha256": code_hash,
                "metric": report["metric"],
                "score": selected_score,
                "maximize": report["maximize"],
                "terminal_report_hash": report["report_hash"],
                "submission_sha256": selected_submission_sha256,
                "exposure_report_refs": list(
                    node.get("actuation_report_refs") or []
                ),
                "verified_adoption_report_refs": [],
                "derived_from_refs": [],
                "adoption_status": "not_published",
                "permitted_operations": list(
                    decision.permitted_scope.operations
                    if decision.permitted_scope
                    else []
                ),
                "audited": True,
                "publication_class": "result_fact",
            },
        )
        ledger.append(
            "terminal_result_writeback_committed",
            {
                **prepared_payload,
                "decision_id": decision.decision_id,
                "overlay_event_id": event.event_id,
                "overlay_event_hash": event.event_hash,
            },
        )
        status = _complete_status(
            {
                "completion": "finalized",
                "artifact_id": artifact_id,
                "claim_id": claim_id,
                "decision_id": decision.decision_id,
                "idempotency_key": idempotency_key,
                "overlay_event_id": event.event_id,
                "overlay_event_hash": event.event_hash,
                "terminal_report_hash": report["report_hash"],
            }
        )
        write_json_atomic(status_target, status)
        return status
    except Exception as error:
        incomplete = record_terminal_writeback_failure(
            status_target,
            error,
            request_path=request_path,
            score_report_path=report_path,
        )
        if raise_on_error:
            raise TerminalWritebackError(str(error)) from error
        return incomplete


__all__ = [
    "RESULT_EVENT_SCHEMA",
    "STATUS_SCHEMA",
    "TerminalWritebackError",
    "finalize_result_writeback",
    "record_terminal_writeback_failure",
]
