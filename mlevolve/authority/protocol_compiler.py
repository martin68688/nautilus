from __future__ import annotations

from dataclasses import dataclass, field

from .models import AuthorityRequest, Claim, ClaimType, Operation, ReceiptType
from .protocol_registry import ProtocolRegistry
from .protocol_execution_contract import (
    ProtocolExecutionContract,
    compile_protocol_execution_contract,
    protocol_payload_requirements,
)


@dataclass
class EvidenceObligations:
    required_receipts: set[ReceiptType] = field(default_factory=set)
    minimum_counts: dict[ReceiptType, int] = field(default_factory=dict)
    require_protocol_compatibility: bool = True
    require_clean_ancestry: bool = True
    require_positive_effect: bool = False
    required_payload_flags: dict[ReceiptType, dict[str, object]] = field(default_factory=dict)
    distinct_payload_values: dict[ReceiptType, tuple[str, int]] = field(default_factory=dict)
    require_trusted_receipts: bool = False


class ProtocolCompiler:
    HIGH_RISK = {
        Operation.RANK,
        Operation.SELECT,
        Operation.PROMOTE_RESULT,
        Operation.PUBLISH_ADOPTION,
        Operation.PUBLISH_CAUSAL,
        Operation.PROMOTE,
        Operation.CODE_SEED,
        Operation.DISTILL_DIAGNOSTIC,
        Operation.DISTILL_CANDIDATE,
        Operation.DISTILL_POSITIVE_RESULT,
        Operation.DISTILL_POSITIVE_ADOPTED,
        Operation.DISTILL_POSITIVE,
        Operation.DISTILL,
        Operation.DERIVED_PUBLICATION,
    }

    def __init__(self, registry: ProtocolRegistry):
        self.registry = registry

    def compile_execution_contract(
        self,
        active_protocol,
        *,
        task_context,
        train_view_ref: str,
        validation_view_ref: str,
        terminal_view_ref: str,
        execution_budget,
        allowed_import_roots=(
            "PIL",
            "cv2",
            "lightgbm",
            "numpy",
            "pandas",
            "scipy",
            "sklearn",
            "timm",
            "torch",
            "torchvision",
            "transformers",
            "xgboost",
        ),
        collector_spec=None,
        adapter_spec=None,
    ) -> ProtocolExecutionContract:
        """Compile one deterministic Host-owned contract for a task execution."""

        spec = self.registry.resolve(active_protocol)
        return compile_protocol_execution_contract(
            spec,
            task_id=task_context.task_id,
            task_family=task_context.task_family,
            train_view_ref=train_view_ref,
            validation_view_ref=validation_view_ref,
            terminal_view_ref=terminal_view_ref,
            execution_budget=execution_budget,
            allowed_import_roots=tuple(allowed_import_roots),
            collector_spec=collector_spec,
            adapter_spec=adapter_spec,
        )

    @staticmethod
    def claim_operation_compatible(claim_type: ClaimType, operation: Operation) -> bool:
        allowed = {
            Operation.INSPECT: set(ClaimType),
            Operation.DEBUG_HYPOTHESIS: {ClaimType.DEBUG_REPAIR, ClaimType.AUDIT_FINDING},
            Operation.GENERATE_CANDIDATE: {ClaimType.METHOD_HYPOTHESIS, ClaimType.DEBUG_REPAIR},
            Operation.RANK: {ClaimType.SCORE, ClaimType.PAIRWISE_SUPERIORITY, ClaimType.GENERALIZATION},
            Operation.SELECT: {ClaimType.SCORE, ClaimType.PAIRWISE_SUPERIORITY, ClaimType.GENERALIZATION},
            Operation.PROMOTE_RESULT: {
                ClaimType.EXECUTED,
                ClaimType.SCORE,
                ClaimType.PAIRWISE_SUPERIORITY,
                ClaimType.GENERALIZATION,
            },
            Operation.PUBLISH_ADOPTION: {ClaimType.EXPERIENCE_ADOPTION},
            Operation.PUBLISH_CAUSAL: {ClaimType.CAUSAL_ATTRIBUTION},
            Operation.PROMOTE: {
                ClaimType.SCORE,
                ClaimType.PAIRWISE_SUPERIORITY,
                ClaimType.CAUSAL_ATTRIBUTION,
                ClaimType.GENERALIZATION,
            },
            Operation.CODE_SEED: {ClaimType.METHOD_HYPOTHESIS, ClaimType.DEBUG_REPAIR},
            Operation.REPAIR_SEED: {ClaimType.DEBUG_REPAIR, ClaimType.METHOD_HYPOTHESIS},
            Operation.DISTILL_DIAGNOSTIC: {ClaimType.AUDIT_FINDING, ClaimType.DEBUG_REPAIR},
            Operation.DISTILL_CANDIDATE: {ClaimType.METHOD_HYPOTHESIS, ClaimType.DEBUG_REPAIR},
            Operation.DISTILL_POSITIVE_RESULT: {
                ClaimType.SCORE,
                ClaimType.PAIRWISE_SUPERIORITY,
                ClaimType.GENERALIZATION,
            },
            Operation.DISTILL_POSITIVE_ADOPTED: {
                ClaimType.EXPERIENCE_ADOPTION,
                ClaimType.CAUSAL_ATTRIBUTION,
            },
            Operation.DISTILL_POSITIVE: {
                ClaimType.METHOD_HYPOTHESIS,
                ClaimType.SCORE,
                ClaimType.PAIRWISE_SUPERIORITY,
                ClaimType.CAUSAL_ATTRIBUTION,
                ClaimType.GENERALIZATION,
            },
            Operation.DERIVED_PUBLICATION: set(ClaimType),
        }
        return claim_type in allowed.get(operation, set())

    def compile(self, claim: Claim, request: AuthorityRequest) -> EvidenceObligations:
        if request.operation in {
            Operation.INSPECT,
            Operation.DEBUG_HYPOTHESIS,
        }:
            return EvidenceObligations(require_protocol_compatibility=False, require_clean_ancestry=False)
        if request.operation == Operation.DISTILL_DIAGNOSTIC:
            return EvidenceObligations(
                require_protocol_compatibility=False,
                require_clean_ancestry=False,
                require_trusted_receipts=True,
            )
        if request.operation == Operation.REPAIR_SEED:
            return EvidenceObligations(
                required_receipts={ReceiptType.METHOD_IDENTITY},
                require_protocol_compatibility=False,
                require_clean_ancestry=False,
                require_trusted_receipts=True,
            )
        required = {ReceiptType.METHOD_IDENTITY}
        payload_flags: dict[ReceiptType, dict[str, object]] = {}
        distinct_values: dict[ReceiptType, tuple[str, int]] = {}
        if claim.claim_type in {
            ClaimType.EXECUTED,
            ClaimType.SCORE,
            ClaimType.METHOD_HYPOTHESIS,
            ClaimType.DEBUG_REPAIR,
            ClaimType.EXPERIENCE_ADOPTION,
            ClaimType.PAIRWISE_SUPERIORITY,
            ClaimType.CAUSAL_ATTRIBUTION,
            ClaimType.GENERALIZATION,
        }:
            required.add(ReceiptType.CODE_EXECUTION)
        if claim.claim_type in {ClaimType.SCORE, ClaimType.PAIRWISE_SUPERIORITY, ClaimType.GENERALIZATION}:
            required.update(
                {
                    ReceiptType.SPLIT_LINEAGE,
                    ReceiptType.FIT_SCOPE,
                    ReceiptType.PREDICTION_SCOPE,
                    ReceiptType.EVALUATOR,
                    ReceiptType.SELECTION_FREEZE,
                }
            )
            protocol = self.registry.resolve(request.active_protocol)
            if (
                protocol.promotion_policy.get("enforce_protocol_payloads")
                is True
            ):
                for receipt_name, flags in protocol_payload_requirements(
                    protocol
                ).items():
                    payload_flags[ReceiptType(receipt_name)] = flags
        minimum: dict[ReceiptType, int] = {}
        if claim.claim_type == ClaimType.PAIRWISE_SUPERIORITY:
            required.update({ReceiptType.SEED_AGGREGATION, ReceiptType.REPLICATION})
            protocol = self.registry.resolve(request.active_protocol)
            minimum[ReceiptType.REPLICATION] = int(protocol.seed_policy.get("pairwise_min_seeds", 3))
            payload_flags[ReceiptType.SEED_AGGREGATION] = {
                "paired": True,
                "preregistered": True,
                "best_seed_selection": False,
            }
            payload_flags[ReceiptType.REPLICATION] = {
                "equal_data_budget": True,
                "equal_compute_budget": True,
                "paired_bootstrap_ci_lower_gt_zero": True,
            }
        if claim.claim_type == ClaimType.GENERALIZATION:
            required.add(ReceiptType.REPLICATION)
            minimum[ReceiptType.REPLICATION] = 2
            distinct_values[ReceiptType.REPLICATION] = ("task_family", 2)
            payload_flags[ReceiptType.REPLICATION] = {
                "equal_data_budget": True,
                "equal_compute_budget": True,
            }
        # Result-fact writeback is intentionally independent of experience
        # actuation.  Actuation receipts authorize the *edge* from an injected
        # experience to the result, not whether the executed result exists.
        # Legacy PROMOTE retains its conservative L3 behavior during migration.
        if request.operation == Operation.DISTILL_CANDIDATE:
            required.add(ReceiptType.STATIC_ACTUATION)
        if request.operation in {
            Operation.DISTILL_POSITIVE_ADOPTED,
            Operation.PUBLISH_ADOPTION,
            Operation.PUBLISH_CAUSAL,
            Operation.PROMOTE,
        }:
            required.update(
                {ReceiptType.STATIC_ACTUATION, ReceiptType.RUNTIME_ACTUATION}
            )

        required_actuation_level = int(
            (claim.boundary or {}).get(
                "required_actuation_level",
                4 if claim.claim_type == ClaimType.CAUSAL_ATTRIBUTION else 0,
            )
            or 0
        )
        if (
            request.operation == Operation.PUBLISH_CAUSAL
            or claim.claim_type == ClaimType.CAUSAL_ATTRIBUTION
            or required_actuation_level >= 4
        ):
            required.update(
                {
                    ReceiptType.STATIC_ACTUATION,
                    ReceiptType.RUNTIME_ACTUATION,
                    ReceiptType.COUNTERFACTUAL_ACTUATION,
                }
            )
            if request.operation == Operation.PUBLISH_CAUSAL:
                required.add(ReceiptType.ADOPTION_PUBLICATION)
            payload_flags[ReceiptType.COUNTERFACTUAL_ACTUATION] = {
                "action_or_code_changed": True,
            }
        if required_actuation_level >= 5:
            payload_flags.setdefault(
                ReceiptType.COUNTERFACTUAL_ACTUATION, {}
            )["effective"] = True
        contract_hash = str(
            (claim.boundary or {}).get("experience_contract_hash") or ""
        )
        if len(contract_hash) == 64:
            for receipt_type in (
                ReceiptType.STATIC_ACTUATION,
                ReceiptType.RUNTIME_ACTUATION,
                ReceiptType.COUNTERFACTUAL_ACTUATION,
            ):
                if receipt_type in required:
                    payload_flags.setdefault(receipt_type, {})[
                        "contract_hash"
                    ] = contract_hash
        if request.operation == Operation.DERIVED_PUBLICATION:
            required.add(ReceiptType.DERIVATION)
        if request.operation in {
            Operation.DISTILL_POSITIVE_RESULT,
            Operation.DISTILL_POSITIVE_ADOPTED,
        }:
            required.add(ReceiptType.DERIVATION)
        return EvidenceObligations(
            required_receipts=required,
            minimum_counts=minimum,
            require_protocol_compatibility=request.operation in self.HIGH_RISK,
            require_clean_ancestry=request.operation in self.HIGH_RISK,
            require_positive_effect=required_actuation_level >= 5,
            required_payload_flags=payload_flags,
            distinct_payload_values=distinct_values,
            require_trusted_receipts=request.operation in self.HIGH_RISK,
        )
