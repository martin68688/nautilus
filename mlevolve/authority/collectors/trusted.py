from __future__ import annotations

import math
from typing import Any

from ..models import ClaimType, ReceiptType
from ..runtime_protocol import PROTOCOL_EVIDENCE_LEVEL, PROTOCOL_EVIDENCE_SCHEMA
from .base import (
    TrustedReceiptCollector,
    UntrustedObservationError,
    require_nonempty,
    require_sha256,
)


_METHOD_CLAIMS = (
    ClaimType.EXECUTED,
    ClaimType.SCORE,
    ClaimType.METHOD_HYPOTHESIS,
    ClaimType.DEBUG_REPAIR,
    ClaimType.EXPERIENCE_ADOPTION,
    ClaimType.PAIRWISE_SUPERIORITY,
    ClaimType.CAUSAL_ATTRIBUTION,
    ClaimType.GENERALIZATION,
)
_SCORE_CLAIMS = (
    ClaimType.SCORE,
    ClaimType.PAIRWISE_SUPERIORITY,
    ClaimType.CAUSAL_ATTRIBUTION,
    ClaimType.GENERALIZATION,
)
_POSITIVE_ACTUATION_CLAIMS = (
    ClaimType.SCORE,
    ClaimType.METHOD_HYPOTHESIS,
    ClaimType.DEBUG_REPAIR,
    ClaimType.EXPERIENCE_ADOPTION,
    ClaimType.PAIRWISE_SUPERIORITY,
    ClaimType.CAUSAL_ATTRIBUTION,
    ClaimType.GENERALIZATION,
)

_TERMINAL_PROTOCOL_EVIDENCE_SCHEMA = (
    "fixed_holdout_terminal_protocol_evidence_v1"
)


def _validated_execution_contract(payload: dict[str, Any]) -> dict[str, str]:
    value = payload.get("execution_contract_hash")
    if value in (None, ""):
        return {}
    return {"execution_contract_hash": require_sha256(payload, "execution_contract_hash")}


def _validated_hash_mapping(payload: dict[str, Any], key: str) -> dict[str, str]:
    values = require_nonempty(payload, key)
    if not isinstance(values, dict):
        raise UntrustedObservationError(f"{key} must be a mapping")
    return {
        str(name): require_sha256({"value": value}, "value")
        for name, value in sorted(values.items())
    }


def _validated_protocol_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    evidence = payload.get("protocol_evidence")
    if evidence in (None, {}):
        return {}
    if not isinstance(evidence, dict):
        raise UntrustedObservationError("protocol_evidence must be a mapping")
    if evidence.get("schema") != PROTOCOL_EVIDENCE_SCHEMA:
        raise UntrustedObservationError("protocol_evidence schema mismatch")
    if evidence.get("evidence_level") != PROTOCOL_EVIDENCE_LEVEL:
        raise UntrustedObservationError("protocol_evidence level is not trusted")
    return {
        "protocol_evidence": {
            "schema": evidence["schema"],
            "evidence_level": evidence["evidence_level"],
            **{
                key: require_sha256(evidence, key)
                for key in (
                    "source_code_sha256",
                    "executed_source_sha256",
                    "plan_sha256",
                    "trace_sha256",
                    "attestation_sha256",
                    "static_audit_sha256",
                    "scope_binding_sha256",
                )
            },
        }
    }


def _validated_terminal_protocol_evidence(
    payload: dict[str, Any],
) -> dict[str, Any]:
    evidence = payload.get("terminal_protocol_evidence")
    if evidence in (None, {}):
        return {}
    if not isinstance(evidence, dict):
        raise UntrustedObservationError(
            "terminal_protocol_evidence must be a mapping"
        )
    if evidence.get("schema") != _TERMINAL_PROTOCOL_EVIDENCE_SCHEMA:
        raise UntrustedObservationError(
            "terminal protocol evidence schema mismatch"
        )
    for field in (
        "internal_runtime_verified",
        "terminal_manifest_verified",
        "receipt_semantics_separated",
    ):
        if evidence.get(field) is not True:
            raise UntrustedObservationError(
                f"terminal protocol evidence lacks {field}"
            )
    hashes = {
        field: require_sha256(evidence, field)
        for field in (
            "training_manifest_sha256",
            "training_manifest_hash",
            "runtime_evidence_hash",
            "runtime_observation_sha256",
            "train_manifest_sha256",
            "evaluator_manifest_sha256",
            "split_receipt_hash",
            "fit_scope_receipt_hash",
            "metric_spec_receipt_hash",
            "evidence_join_sha256",
        )
    }
    canonical_strategy = str(
        require_nonempty(evidence, "canonical_split_strategy")
    )
    if canonical_strategy not in {
        "stratified_random",
        "grouped",
        "chronological",
        "deterministic_random",
    }:
        raise UntrustedObservationError(
            "terminal protocol evidence has an unknown canonical split strategy"
        )
    terminal_fit_scope = str(
        require_nonempty(evidence, "terminal_fit_scope")
    )
    if terminal_fit_scope != "train_view_only":
        raise UntrustedObservationError(
            "terminal fit scope must remain train_view_only"
        )
    metric_direction = str(
        require_nonempty(evidence, "metric_direction")
    )
    if metric_direction not in {"maximize", "minimize"}:
        raise UntrustedObservationError(
            "terminal protocol metric direction is invalid"
        )
    return {
        "terminal_protocol_evidence": {
            "schema": evidence["schema"],
            "condition": str(require_nonempty(evidence, "condition")),
            "protocol_ref": str(
                require_nonempty(evidence, "protocol_ref")
            ),
            "formal_split_strategy": str(
                require_nonempty(evidence, "formal_split_strategy")
            ),
            "canonical_split_strategy": canonical_strategy,
            "internal_fit_scope": str(
                require_nonempty(evidence, "internal_fit_scope")
            ),
            "terminal_fit_scope": terminal_fit_scope,
            "metric_name": str(
                require_nonempty(evidence, "metric_name")
            ),
            "metric_direction": metric_direction,
            "internal_runtime_verified": True,
            "terminal_manifest_verified": True,
            "receipt_semantics_separated": True,
            **hashes,
        }
    }


class CodeExecutionCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.CODE_EXECUTION
    collector_id = "host.code_execution"
    supports_claim_types = _METHOD_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        exit_status = int(payload.get("exit_status", -1))
        executed_path = str(require_nonempty(payload, "executed_path"))
        run_hash = require_sha256(payload, "run_hash")
        if exit_status != 0:
            raise UntrustedObservationError("code execution did not exit successfully")
        return {
            "exit_status": exit_status,
            "executed_path": executed_path,
            "run_hash": run_hash,
            "execution_verified": True,
            **_validated_execution_contract(payload),
        }


class MethodIdentityCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.METHOD_IDENTITY
    collector_id = "host.method_identity"
    supports_claim_types = _METHOD_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "method_fingerprint": require_sha256(payload, "method_fingerprint"),
            "code_sha256": require_sha256(payload, "code_sha256"),
            "method_identity_verified": True,
            **_validated_execution_contract(payload),
        }


class SplitLineageCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.SPLIT_LINEAGE
    collector_id = "host.split_lineage"
    supports_claim_types = _SCORE_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        partition_hashes = _validated_hash_mapping(payload, "partition_hashes")
        if int(payload.get("overlap_count", -1)) != 0:
            raise UntrustedObservationError("split lineage contains forbidden overlap")
        output = {
            "partition_hashes": partition_hashes,
            "overlap_count": 0,
            "partition_lineage_verified": True,
            **_validated_protocol_evidence(payload),
            **_validated_terminal_protocol_evidence(payload),
            **_validated_execution_contract(payload),
        }
        strategy = str(payload.get("split_strategy") or "")
        if strategy:
            if strategy not in {
                "stratified_random",
                "grouped",
                "chronological",
                "deterministic_random",
            }:
                raise UntrustedObservationError("unknown split strategy")
            output["split_strategy"] = strategy
        terminal_evidence = output.get("terminal_protocol_evidence") or {}
        if terminal_evidence and strategy != terminal_evidence.get(
            "canonical_split_strategy"
        ):
            raise UntrustedObservationError(
                "split strategy does not match terminal protocol evidence"
            )
        if strategy == "stratified_random":
            if payload.get("stratification_verified") is not True:
                raise UntrustedObservationError("stratified split was not verified")
            output["stratification_verified"] = True
        if strategy == "grouped":
            if int(payload.get("group_overlap_count", -1)) != 0:
                raise UntrustedObservationError(
                    "split lineage contains forbidden group overlap"
                )
            output["group_overlap_count"] = 0
        if strategy == "chronological":
            if int(payload.get("future_to_past_count", -1)) != 0:
                raise UntrustedObservationError(
                    "chronological split contains future-to-past leakage"
                )
            if payload.get("chronological_order_verified") is not True:
                raise UntrustedObservationError(
                    "chronological partition order was not verified"
                )
            output["future_to_past_count"] = 0
            output["chronological_order_verified"] = True
        if strategy == "deterministic_random":
            if payload.get("deterministic_partition_verified") is not True:
                raise UntrustedObservationError(
                    "deterministic random partition was not verified"
                )
            output["deterministic_partition_verified"] = True
        return output


class FitScopeCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.FIT_SCOPE
    collector_id = "host.fit_scope"
    supports_claim_types = _SCORE_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        hashes = _validated_hash_mapping(payload, "fit_scope_hashes")
        if int(payload.get("holdout_fit_count", -1)) != 0:
            raise UntrustedObservationError("a learned component was fit on holdout data")
        output = {
            "fit_scope_hashes": hashes,
            "holdout_fit_count": 0,
            "fit_scope_verified": True,
            **_validated_protocol_evidence(payload),
            **_validated_terminal_protocol_evidence(payload),
            **_validated_execution_contract(payload),
        }
        fit_scope = str(payload.get("fit_scope") or "")
        if fit_scope:
            output["fit_scope"] = fit_scope
        terminal_evidence = output.get("terminal_protocol_evidence") or {}
        if terminal_evidence:
            terminal_fit_scope = str(
                payload.get("terminal_fit_scope") or ""
            )
            if fit_scope != terminal_evidence.get("internal_fit_scope"):
                raise UntrustedObservationError(
                    "internal fit scope does not match terminal protocol evidence"
                )
            if terminal_fit_scope != terminal_evidence.get(
                "terminal_fit_scope"
            ):
                raise UntrustedObservationError(
                    "terminal fit scope evidence was relabeled or omitted"
                )
            output["terminal_fit_scope"] = terminal_fit_scope
        return output


class PredictionScopeCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.PREDICTION_SCOPE
    collector_id = "host.prediction_scope"
    supports_claim_types = _SCORE_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        hashes = _validated_hash_mapping(payload, "prediction_scope_hashes")
        if int(payload.get("forbidden_overlap_count", -1)) != 0:
            raise UntrustedObservationError("prediction scope contains forbidden overlap")
        return {
            "prediction_scope_hashes": hashes,
            "forbidden_overlap_count": 0,
            "prediction_scope_verified": True,
            **_validated_protocol_evidence(payload),
            **_validated_terminal_protocol_evidence(payload),
            **_validated_execution_contract(payload),
        }


class EvaluatorIntegrityCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.EVALUATOR
    collector_id = "host.evaluator_integrity"
    supports_claim_types = _SCORE_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        direction = str(require_nonempty(payload, "metric_direction"))
        if direction not in {"maximize", "minimize"}:
            raise UntrustedObservationError("metric_direction must be maximize or minimize")
        if bool(payload.get("tampered", True)):
            raise UntrustedObservationError("host observed evaluator tampering")
        output = {
            "evaluator_hash": require_sha256(payload, "evaluator_hash"),
            "inputs_hash": require_sha256(payload, "inputs_hash"),
            "metric_direction": direction,
            "tampered": False,
            "evaluator_integrity_verified": True,
            **_validated_protocol_evidence(payload),
            **_validated_terminal_protocol_evidence(payload),
            **_validated_execution_contract(payload),
        }
        metric_name = str(payload.get("metric_name") or "")
        if metric_name:
            output["metric_name"] = metric_name
        if payload.get("metric_value") is not None:
            metric_value = float(payload["metric_value"])
            if not math.isfinite(metric_value):
                raise UntrustedObservationError("metric_value must be finite")
            output["metric_value"] = metric_value
        terminal_evidence = output.get("terminal_protocol_evidence") or {}
        if terminal_evidence:
            if metric_name != terminal_evidence.get("metric_name"):
                raise UntrustedObservationError(
                    "metric name does not match terminal protocol evidence"
                )
            if direction != terminal_evidence.get("metric_direction"):
                raise UntrustedObservationError(
                    "metric direction does not match terminal protocol evidence"
                )
        return output


class SelectionFreezeCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.SELECTION_FREEZE
    collector_id = "host.selection_freeze"
    supports_claim_types = _SCORE_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("frozen_before_holdout") is not True:
            raise UntrustedObservationError("selection was not frozen before holdout")
        return {
            "candidate_set_hash": require_sha256(payload, "candidate_set_hash"),
            "frozen_before_holdout": True,
            "selection_freeze_verified": True,
            **_validated_protocol_evidence(payload),
            **_validated_terminal_protocol_evidence(payload),
            **_validated_execution_contract(payload),
        }


class SeedAggregationCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.SEED_AGGREGATION
    collector_id = "host.seed_aggregation"
    supports_claim_types = (ClaimType.PAIRWISE_SUPERIORITY, ClaimType.GENERALIZATION)

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        seeds = list(require_nonempty(payload, "declared_seeds"))
        if len({str(seed) for seed in seeds}) < 2:
            raise UntrustedObservationError("seed aggregation requires at least two seeds")
        if payload.get("best_seed_selection") is not False:
            raise UntrustedObservationError("best-seed selection is not a trusted aggregation")
        if payload.get("paired") is not True:
            raise UntrustedObservationError("pairwise seed aggregation must be paired")
        if payload.get("preregistered") is not True:
            raise UntrustedObservationError("seed aggregation must be preregistered")
        return {
            "declared_seeds": seeds,
            "all_results_hash": require_sha256(payload, "all_results_hash"),
            "aggregation": str(require_nonempty(payload, "aggregation")),
            "paired": True,
            "preregistered": True,
            "best_seed_selection": False,
            "seed_aggregation_verified": True,
        }


class ReplicationCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.REPLICATION
    collector_id = "host.replication"
    supports_claim_types = (ClaimType.PAIRWISE_SUPERIORITY, ClaimType.GENERALIZATION)

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("equal_data_budget") is not True:
            raise UntrustedObservationError("replication data budgets are not equal")
        if payload.get("equal_compute_budget") is not True:
            raise UntrustedObservationError("replication compute budgets are not equal")
        if payload.get("paired_bootstrap_ci_lower_gt_zero") is not True:
            raise UntrustedObservationError("replication superiority interval is not positive")
        return {
            "replication_id": str(require_nonempty(payload, "replication_id")),
            "task_family": str(require_nonempty(payload, "task_family")),
            "result_hash": require_sha256(payload, "result_hash"),
            "equal_data_budget": True,
            "equal_compute_budget": True,
            "paired_bootstrap_ci_lower_gt_zero": True,
            "replication_verified": True,
        }


class StaticActuationCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.STATIC_ACTUATION
    collector_id = "host.static_actuation"
    supports_claim_types = _POSITIVE_ACTUATION_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        checks = require_nonempty(payload, "checks")
        if not isinstance(checks, dict) or not all(value is True for value in checks.values()):
            raise UntrustedObservationError("static actuation contract checks did not all pass")
        return {
            "contract_hash": require_sha256(payload, "contract_hash"),
            "checks": dict(sorted(checks.items())),
            "static_actuation_verified": True,
        }


class RuntimeActuationCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.RUNTIME_ACTUATION
    collector_id = "host.runtime_actuation"
    supports_claim_types = _POSITIVE_ACTUATION_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_hashes = list(require_nonempty(payload, "event_hashes"))
        if payload.get("target_path_executed") is not True:
            raise UntrustedObservationError("target actuation path did not execute")
        validated_hashes = [
            require_sha256({"event_hash": value}, "event_hash")
            for value in event_hashes
        ]
        output = {
            "event_hashes": validated_hashes,
            "target_path_executed": True,
            "runtime_actuation_verified": True,
        }
        contract_hash = str(payload.get("contract_hash") or "")
        observations_hash = str(payload.get("observations_hash") or "")
        if contract_hash:
            output["contract_hash"] = require_sha256(
                {"contract_hash": contract_hash}, "contract_hash"
            )
        if observations_hash:
            output["observations_hash"] = require_sha256(
                {"observations_hash": observations_hash}, "observations_hash"
            )
        return output


class AdoptionPublicationCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.ADOPTION_PUBLICATION
    collector_id = "host.adoption_publication"
    supports_claim_types = (ClaimType.CAUSAL_ATTRIBUTION,)

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "edge_id": str(require_nonempty(payload, "edge_id")),
            "edge_hash": require_sha256(payload, "edge_hash"),
            "contract_hash": require_sha256(payload, "contract_hash"),
            "adoption_decision_id": str(
                require_nonempty(payload, "adoption_decision_id")
            ),
            "adoption_publication_verified": True,
        }


class CounterfactualCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.COUNTERFACTUAL_ACTUATION
    collector_id = "host.counterfactual"
    supports_claim_types = _POSITIVE_ACTUATION_CLAIMS

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory_on_hash = require_sha256(payload, "memory_on_action_hash")
        memory_off_hash = require_sha256(payload, "memory_off_action_hash")
        on_code_hash = str(payload.get("memory_on_code_hash") or "")
        off_code_hash = str(payload.get("memory_off_code_hash") or "")
        if on_code_hash:
            on_code_hash = require_sha256(
                {"memory_on_code_hash": on_code_hash}, "memory_on_code_hash"
            )
        if off_code_hash:
            off_code_hash = require_sha256(
                {"memory_off_code_hash": off_code_hash}, "memory_off_code_hash"
            )
        if payload.get("action_or_code_changed") is not True:
            raise UntrustedObservationError("counterfactual did not change action or code")
        if memory_on_hash == memory_off_hash and (
            not on_code_hash or not off_code_hash or on_code_hash == off_code_hash
        ):
            raise UntrustedObservationError("counterfactual action hashes are identical")
        protocol_legal = payload.get("protocol_legal") is True
        effective = payload.get("effective") is True
        outcome_delta = payload.get("outcome_delta")
        metric_direction = str(payload.get("metric_direction") or "")
        if effective:
            if not protocol_legal:
                raise UntrustedObservationError(
                    "effective counterfactual outcome is not protocol legal"
                )
            if metric_direction not in {"maximize", "minimize"}:
                raise UntrustedObservationError(
                    "effective counterfactual has no metric direction"
                )
            if not isinstance(outcome_delta, (int, float)) or float(outcome_delta) <= 0:
                raise UntrustedObservationError(
                    "effective counterfactual has no positive outcome delta"
                )
        output = {
            "pair_id": str(require_nonempty(payload, "pair_id")),
            "memory_on_action_hash": memory_on_hash,
            "memory_off_action_hash": memory_off_hash,
            "action_or_code_changed": True,
            "protocol_legal": protocol_legal,
            "effective": effective,
            "counterfactual_verified": True,
        }
        control_hash = str(payload.get("control_hash") or "")
        if control_hash:
            output["control_hash"] = require_sha256(
                {"control_hash": control_hash}, "control_hash"
            )
        if on_code_hash:
            output["memory_on_code_hash"] = on_code_hash
        if off_code_hash:
            output["memory_off_code_hash"] = off_code_hash
        if outcome_delta is not None:
            output["outcome_delta"] = float(outcome_delta)
        if metric_direction:
            output["metric_direction"] = metric_direction
        contract_hash = str(payload.get("contract_hash") or "")
        if contract_hash:
            output["contract_hash"] = require_sha256(
                {"contract_hash": contract_hash}, "contract_hash"
            )
        return output


class CounterfactualObservationCollector(TrustedReceiptCollector):
    """Attest an observer-only pair, including a confirmed no-change pair.

    Unlike ``CounterfactualCollector`` this receipt does not claim positive
    actuation or efficacy.  It only proves that the Host captured both frozen
    generation arms and that the counterfactual arm was never executed.
    """

    receipt_type = ReceiptType.COUNTERFACTUAL_OBSERVATION
    collector_id = "host.counterfactual_observation"
    supports_claim_types = ()

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        on_action = require_sha256(payload, "memory_on_action_hash")
        off_action = require_sha256(payload, "memory_off_action_hash")
        on_code = require_sha256(payload, "memory_on_code_hash")
        off_code = require_sha256(payload, "memory_off_code_hash")
        changed = payload.get("action_or_code_changed") is True
        observed_changed = on_action != off_action or on_code != off_code
        if changed != observed_changed:
            raise UntrustedObservationError(
                "counterfactual change flag does not match the observed hashes"
            )
        if payload.get("never_submitted_to_executor") is not True:
            raise UntrustedObservationError(
                "prospective counterfactual must remain observer-only"
            )
        return {
            "pair_id": str(require_nonempty(payload, "pair_id")),
            "control_hash": require_sha256(payload, "control_hash"),
            "memory_payload_hash": require_sha256(payload, "memory_payload_hash"),
            "memory_on_action_hash": on_action,
            "memory_off_action_hash": off_action,
            "memory_on_code_hash": on_code,
            "memory_off_code_hash": off_code,
            "action_or_code_changed": changed,
            "never_submitted_to_executor": True,
            "counterfactual_observed": True,
        }


class DerivationCollector(TrustedReceiptCollector):
    receipt_type = ReceiptType.DERIVATION
    collector_id = "host.derivation"

    def validated_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        parents = [str(value) for value in require_nonempty(payload, "parent_claim_refs")]
        if payload.get("scope_widened") is not False:
            raise UntrustedObservationError("derived artifact widened parent authority scope")
        return {
            "parent_claim_refs": parents,
            "mapping_hash": require_sha256(payload, "mapping_hash"),
            "scope_widened": False,
            "derivation_verified": True,
        }
