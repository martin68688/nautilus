from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .models import Claim, Receipt, ReceiptType
from .protocol_compiler import EvidenceObligations


@dataclass
class EvidencePath:
    path_id: str
    claim_id: str
    receipt_ids: list[str]
    required_parent_claims: list[str] = field(default_factory=list)


@dataclass
class PathEvaluation:
    satisfied_paths: list[str]
    missing_obligations: list[str]
    blocking_receipts: list[str]

    @property
    def satisfied(self) -> bool:
        return bool(self.satisfied_paths)


class EvidenceGraph:
    """AND within a path, OR across independently complete paths."""

    def __init__(self):
        self.claims: dict[str, Claim] = {}
        self.receipts: dict[str, Receipt] = {}
        self.paths: dict[str, EvidencePath] = {}
        self.claim_paths: dict[str, list[str]] = defaultdict(list)

    def add_claim(self, claim: Claim) -> None:
        existing = self.claims.get(claim.claim_id)
        if existing and existing != claim:
            raise ValueError(f"Claim is immutable: {claim.claim_id}")
        self.claims[claim.claim_id] = claim

    def add_receipt(self, receipt: Receipt) -> None:
        existing = self.receipts.get(receipt.receipt_id)
        if existing and existing.payload_hash != receipt.payload_hash:
            raise ValueError(f"Receipt is immutable: {receipt.receipt_id}")
        self.receipts[receipt.receipt_id] = receipt

    def add_path(self, path: EvidencePath) -> None:
        if path.claim_id not in self.claims:
            raise KeyError(f"Unknown claim for evidence path: {path.claim_id}")
        self.paths[path.path_id] = path
        if path.path_id not in self.claim_paths[path.claim_id]:
            self.claim_paths[path.claim_id].append(path.path_id)

    def _has_ancestry_cycle(self, claim_id: str, visiting: set[str] | None = None) -> bool:
        visiting = set() if visiting is None else visiting
        if claim_id in visiting:
            return True
        claim = self.claims.get(claim_id)
        if not claim:
            return False
        visiting.add(claim_id)
        cyclic = any(self._has_ancestry_cycle(parent, visiting) for parent in claim.parent_claims)
        visiting.remove(claim_id)
        return cyclic

    def evaluate(self, claim_id: str, obligations: EvidenceObligations) -> PathEvaluation:
        claim = self.claims.get(claim_id)
        if claim is None:
            return PathEvaluation([], ["claim_exists"], [])
        if self._has_ancestry_cycle(claim_id):
            return PathEvaluation([], ["acyclic_claim_ancestry"], [])
        missing_by_path: list[set[str]] = []
        blocking: set[str] = set()
        satisfied: list[str] = []
        for path_id in self.claim_paths.get(claim_id, []):
            path = self.paths[path_id]
            receipts = [self.receipts[rid] for rid in path.receipt_ids if rid in self.receipts]
            claim_type = claim.claim_type.value
            applicable = [
                receipt
                for receipt in receipts
                if not receipt.supports_claim_types or claim_type in receipt.supports_claim_types
            ]
            positive = [
                receipt
                for receipt in applicable
                if not receipt.payload.get("contradicts", False)
            ]
            blockers = [
                receipt.receipt_id
                for receipt in receipts
                if receipt.payload.get("contradicts", False)
                and (
                    not receipt.blocks_claim_types
                    or claim_type in receipt.blocks_claim_types
                )
            ]
            blocking.update(blockers)
            trusted_positive = [
                receipt for receipt in positive if receipt.trust_status == "trusted_host"
            ]
            eligible = trusted_positive if obligations.require_trusted_receipts else positive
            types = {r.receipt_type for r in eligible}
            missing = {f"receipt:{item.value}" for item in obligations.required_receipts - types}
            if obligations.require_trusted_receipts:
                if not trusted_positive:
                    missing.add("trusted_receipt:any")
                observed_untrusted_types = {
                    receipt.receipt_type for receipt in positive
                    if receipt.trust_status != "trusted_host"
                }
                for receipt_type in obligations.required_receipts & observed_untrusted_types - types:
                    missing.discard(f"receipt:{receipt_type.value}")
                    missing.add(f"trusted_receipt:{receipt_type.value}")
            for receipt_type, count in obligations.minimum_counts.items():
                observed = sum(1 for r in eligible if r.receipt_type == receipt_type)
                if observed < count:
                    missing.add(f"count:{receipt_type.value}>={count}")
            for receipt_type, flags in obligations.required_payload_flags.items():
                candidates = [r for r in eligible if r.receipt_type == receipt_type]
                if not candidates or not any(
                    all(
                        candidate.payload.get(key) == expected
                        for key, expected in flags.items()
                    )
                    for candidate in candidates
                ):
                    missing.add(f"payload:{receipt_type.value}")
            if obligations.require_positive_effect:
                effective_receipts = [
                    receipt
                    for receipt in eligible
                    if receipt.receipt_type == ReceiptType.COUNTERFACTUAL_ACTUATION
                    and receipt.payload.get("effective") is True
                    and receipt.payload.get("protocol_legal") is True
                    and isinstance(receipt.payload.get("outcome_delta"), (int, float))
                    and float(receipt.payload["outcome_delta"]) > 0
                ]
                if not effective_receipts:
                    missing.add("positive_effect:protocol_legal_counterfactual")
            for receipt_type, (key, count) in obligations.distinct_payload_values.items():
                values = {
                    receipt.payload.get(key)
                    for receipt in eligible
                    if receipt.receipt_type == receipt_type and receipt.payload.get(key) is not None
                }
                if len(values) < count:
                    missing.add(f"distinct:{receipt_type.value}.{key}>={count}")
            if obligations.require_clean_ancestry:
                for parent in path.required_parent_claims:
                    if parent not in self.claims:
                        missing.add(f"authorized_parent:{parent}")
                        continue
                    parent_evaluation = self.evaluate(parent, obligations)
                    if not parent_evaluation.satisfied:
                        missing.add(f"authorized_parent:{parent}")
            if not missing and not blockers:
                satisfied.append(path_id)
            missing_by_path.append(missing)
        if satisfied:
            # Evidence paths are alternatives: a complete clean path is enough
            # even when another independent path is contaminated. Blockers on
            # the rejected path remain in its receipts; they do not poison the
            # clean support path or become falsely described as satisfied.
            return PathEvaluation(satisfied, [], [])
        if not missing_by_path:
            return PathEvaluation([], ["complete_evidence_path"], sorted(blocking))
        # Do not union partial paths: report the smallest incomplete path for diagnostics.
        best_missing = min(missing_by_path, key=lambda item: (len(item), sorted(item)))
        return PathEvaluation([], sorted(best_missing), sorted(blocking))
