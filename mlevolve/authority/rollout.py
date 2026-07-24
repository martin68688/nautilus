from __future__ import annotations

import collections
import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .ledger import AuthorityLedger
from .memory_snapshot import (
    CURRENT_POINTER_SCHEMA,
    ImmutableBaseBundle,
    _payload_hash,
    _read_json,
    _safe_relative_path,
    file_lock,
    make_current_pointer,
    sha256_json,
    write_json_atomic,
)
from .models import AuthorityDecision, DecisionOutcome, Operation, canonical_operation
from .policy import is_high_risk


ROLLOUT_VERSION_SCHEMA = "authority_rollout_version_set_v1"
SHADOW_RECORD_SCHEMA = "authority_shadow_decision_record_v1"
CANARY_REPORT_SCHEMA = "authority_canary_report_v1"
CANARY_ORACLE_PACKET_SCHEMA = "authority_canary_oracle_packet_v1"
CANARY_ORACLE_REVIEW_REPORT_SCHEMA = "authority_canary_oracle_review_report_v1"
ROLLBACK_REPORT_SCHEMA = "memory_bundle_rollback_report_v1"
SHADOW_REVIEW_PACKET_SCHEMA = "authority_shadow_review_packet_v1"
SHADOW_REVIEW_REPORT_SCHEMA = "authority_shadow_review_report_v1"
SHADOW_REVIEW_DISPOSITIONS = frozenset(
    {
        "expected_safety_block",
        "confirmed_legacy_false_allow",
        "confirmed_legacy_false_denial",
        "confirmed_authority_false_denial",
        "requires_fix",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: str) -> bool:
    value = str(value).lower()
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


@dataclass(frozen=True)
class RolloutVersionSet:
    rollout_id: str
    policy_version: str
    protocol_ref: str
    collector_version: str
    bundle_id: str = "none"
    bundle_manifest_sha256: str = ""
    version_hash: str = ""
    schema: str = ROLLOUT_VERSION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "rollout_id",
            "policy_version",
            "protocol_ref",
            "collector_version",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"Rollout version field is required: {field_name}")
        if self.bundle_id == "none" and self.bundle_manifest_sha256:
            raise ValueError("Bundle-free rollout cannot declare a manifest digest")
        if self.bundle_id != "none" and not _is_sha256(
            self.bundle_manifest_sha256
        ):
            raise ValueError("Rollout bundle manifest must be a SHA-256 digest")
        payload = self.as_dict()
        payload.pop("version_hash", None)
        expected = _sha256_json(payload)
        if self.version_hash and self.version_hash != expected:
            raise ValueError("Rollout version-set hash mismatch")
        object.__setattr__(self, "version_hash", expected)

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass
class ShadowDecisionRecord:
    decision_id: str
    artifact_id: str
    claim_id: str
    operation: str
    generation_stage: str
    governance_stage: str
    legacy_allowed: bool
    authority_allowed: bool
    effective_allowed: bool
    enforced: bool
    taxonomy: str
    reason_class: str
    policy_version: str
    rollout_version_hash: str
    missing_obligations: tuple[str, ...] = ()
    blocking_receipts: tuple[str, ...] = ()
    recorded_at: str = field(default_factory=_utc_now)
    record_hash: str = ""
    schema: str = SHADOW_RECORD_SCHEMA

    def finalize(self) -> "ShadowDecisionRecord":
        payload = self.as_dict()
        payload.pop("record_hash", None)
        self.record_hash = _sha256_json(payload)
        return self

    def verify(self) -> None:
        payload = self.as_dict()
        payload.pop("record_hash", None)
        if self.schema != SHADOW_RECORD_SCHEMA or self.record_hash != _sha256_json(payload):
            raise ValueError("Shadow decision record hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ShadowDecisionRecord":
        values = copy.deepcopy(dict(payload))
        values["missing_obligations"] = tuple(
            str(value) for value in values.get("missing_obligations") or []
        )
        values["blocking_receipts"] = tuple(
            str(value) for value in values.get("blocking_receipts") or []
        )
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(
                f"Shadow decision record has unknown fields: {sorted(unknown)}"
            )
        record = cls(**values)
        record.verify()
        return record


def _reason_class(decision: AuthorityDecision) -> str:
    missing = set(decision.missing_obligations)
    if any(value.startswith("authority_internal_error:") for value in missing):
        return "authority_internal_error"
    if "active_protocol_compatibility" in missing:
        return "protocol_mismatch"
    if any(value.startswith("claim_") or value == "claim_exists" for value in missing):
        return "claim_or_scope_mismatch"
    if decision.blocking_receipts:
        return "contradictory_receipt"
    if any(value.startswith(("receipt:", "trusted_receipt:", "payload:", "count:")) for value in missing):
        return "missing_or_untrusted_evidence"
    if "claim_operation_compatibility" in missing:
        return "operation_incompatible"
    return "complete" if decision.allowed else "other_denial"


def _taxonomy(legacy_allowed: bool, authority_allowed: bool, decision: AuthorityDecision) -> str:
    if _reason_class(decision) == "authority_internal_error":
        return (
            "internal_error_high_risk_block"
            if is_high_risk(decision.operation)
            else "internal_error_low_risk_abstain"
        )
    if legacy_allowed and authority_allowed:
        return "agreement_allow"
    if not legacy_allowed and not authority_allowed:
        return "agreement_deny"
    if legacy_allowed and not authority_allowed:
        return "legacy_allow_authority_deny"
    return "legacy_deny_authority_allow"


class AuthorityRolloutController:
    """Freeze versions, compare legacy/full decisions, and stage enforcement."""

    def __init__(
        self,
        *,
        mode: str,
        versions: RolloutVersionSet,
        ledger: AuthorityLedger | None = None,
        enforce_operations: Iterable[str] = (),
        enforce_generation_stages: Iterable[str] = (),
        enforce_governance_stages: Iterable[str] = (),
    ) -> None:
        self.mode = str(mode).lower()
        if self.mode not in {"off", "shadow", "enforce"}:
            raise ValueError(f"Unsupported rollout mode: {mode}")
        self.versions = versions
        self.ledger = ledger
        self.enforce_operations = {
            canonical_operation(value).value for value in enforce_operations
        }
        self.enforce_generation_stages = {
            str(value) for value in enforce_generation_stages if str(value)
        }
        self.enforce_governance_stages = {
            str(value) for value in enforce_governance_stages if str(value)
        }
        self._records: dict[str, ShadowDecisionRecord] = {}
        self._frozen = False

    def bind_bundle(self, *, bundle_id: str, manifest_sha256: str) -> None:
        if self._frozen or self._records:
            raise RuntimeError("Rollout versions are frozen after the first decision")
        bundle_id = str(bundle_id).strip()
        manifest_sha256 = str(manifest_sha256).lower()
        if not bundle_id or bundle_id == "none":
            raise ValueError("A loaded Base Bundle must have a non-none bundle ID")
        if not _is_sha256(manifest_sha256):
            raise ValueError("Loaded Base Bundle manifest must be a SHA-256 digest")
        if self.versions.bundle_id not in {"none", str(bundle_id)}:
            raise ValueError("Configured rollout bundle ID does not match CURRENT")
        if (
            self.versions.bundle_manifest_sha256
            and self.versions.bundle_manifest_sha256 != str(manifest_sha256)
        ):
            raise ValueError("Configured rollout bundle manifest does not match CURRENT")
        self.versions = replace(
            self.versions,
            bundle_id=str(bundle_id),
            bundle_manifest_sha256=str(manifest_sha256),
            version_hash="",
        )

    def freeze(self) -> None:
        if self._frozen:
            return
        self._frozen = True
        if self.ledger is not None:
            self.ledger.append("authority_rollout_versions_frozen", self.versions.as_dict())

    @property
    def frozen(self) -> bool:
        return self._frozen

    def should_enforce(self, decision: AuthorityDecision) -> bool:
        if self.mode != "enforce":
            return False
        operation = canonical_operation(decision.operation).value
        if self.enforce_operations and operation not in self.enforce_operations:
            return False
        if (
            self.enforce_generation_stages
            and decision.generation_stage not in self.enforce_generation_stages
        ):
            return False
        if (
            self.enforce_governance_stages
            and decision.governance_stage not in self.enforce_governance_stages
        ):
            return False
        return True

    def record(
        self,
        decision: AuthorityDecision,
        *,
        legacy_allowed: bool,
        effective_allowed: bool,
        enforced: bool,
    ) -> ShadowDecisionRecord:
        current = self._records.get(decision.decision_id)
        if current is not None:
            # A decision ID denotes one host decision episode. Repeated helper
            # probes must not append a second comparison or rewrite the first.
            return current
        if decision.policy_version != self.versions.policy_version:
            raise ValueError("Decision policy does not match frozen rollout policy")
        self.freeze()
        record = ShadowDecisionRecord(
            decision_id=decision.decision_id,
            artifact_id=decision.artifact_id,
            claim_id=decision.claim_id,
            operation=decision.operation,
            generation_stage=decision.generation_stage,
            governance_stage=decision.governance_stage,
            legacy_allowed=bool(legacy_allowed),
            authority_allowed=decision.allowed,
            effective_allowed=bool(effective_allowed),
            enforced=bool(enforced),
            taxonomy=_taxonomy(bool(legacy_allowed), decision.allowed, decision),
            reason_class=_reason_class(decision),
            policy_version=decision.policy_version,
            rollout_version_hash=self.versions.version_hash,
            missing_obligations=tuple(sorted(decision.missing_obligations)),
            blocking_receipts=tuple(sorted(decision.blocking_receipts)),
        ).finalize()
        self._records[decision.decision_id] = record
        if self.ledger is not None:
            self.ledger.append("authority_shadow_comparison", record.as_dict())
        return record

    def records(self) -> list[ShadowDecisionRecord]:
        return [copy.deepcopy(self._records[key]) for key in sorted(self._records)]

    def report(
        self,
        *,
        review_dispositions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        records = self.records()
        dispositions = copy.deepcopy(dict(review_dispositions or {}))
        unknown = set(dispositions) - {record.decision_id for record in records}
        if unknown:
            raise ValueError(f"Review dispositions reference unknown decisions: {sorted(unknown)}")
        for decision_id, review in dispositions.items():
            if not str(review.get("reviewer") or "").strip():
                raise ValueError(f"Shadow review lacks reviewer: {decision_id}")
            if str(review.get("disposition") or "") not in SHADOW_REVIEW_DISPOSITIONS:
                raise ValueError(f"Unsupported shadow review disposition: {decision_id}")
        taxonomy_counts = collections.Counter(record.taxonomy for record in records)
        disagreement_ids = sorted(
            record.decision_id
            for record in records
            if record.legacy_allowed != record.authority_allowed
        )
        payload = {
            "schema": "authority_shadow_disagreement_report_v1",
            "rollout_versions": self.versions.as_dict(),
            "mode": self.mode,
            "record_count": len(records),
            "enforced_record_count": sum(record.enforced for record in records),
            "effective_allow_count": sum(record.effective_allowed for record in records),
            "legacy_allow_count": sum(record.legacy_allowed for record in records),
            "authority_allow_count": sum(record.authority_allowed for record in records),
            "taxonomy_counts": dict(sorted(taxonomy_counts.items())),
            "disagreement_decision_ids": disagreement_ids,
            "disagreement_count": len(disagreement_ids),
            "reviewed_disagreement_count": sum(
                decision_id in dispositions for decision_id in disagreement_ids
            ),
            "review_dispositions": dispositions,
            "records_sha256": _sha256_json([record.as_dict() for record in records]),
            "report_hash": "",
        }
        payload["report_hash"] = _sha256_json(
            {key: value for key, value in payload.items() if key != "report_hash"}
        )
        return payload


@dataclass(frozen=True)
class CanaryThresholds:
    minimum_decisions: int = 1
    max_unauthorized_authority_allows: int = 0
    max_false_denial_rate: float = 0.05

    def __post_init__(self) -> None:
        if self.minimum_decisions <= 0:
            raise ValueError("Canary minimum_decisions must be positive")
        if self.max_unauthorized_authority_allows < 0:
            raise ValueError("Canary unauthorized-allow threshold cannot be negative")
        if not 0.0 <= self.max_false_denial_rate <= 1.0:
            raise ValueError("Canary false-denial threshold must be in [0, 1]")


def evaluate_canary(
    records: Iterable[ShadowDecisionRecord],
    *,
    oracle_should_allow: Mapping[str, bool],
    thresholds: CanaryThresholds,
) -> dict[str, Any]:
    observed = list(records)
    for record in observed:
        record.verify()
    if len({record.decision_id for record in observed}) != len(observed):
        raise ValueError("Canary records contain duplicate decision IDs")
    values = [record for record in observed if record.enforced]
    rollout_hashes = {record.rollout_version_hash for record in values}
    if len(rollout_hashes) > 1:
        raise ValueError("Canary records span multiple rollout version sets")
    record_ids = {record.decision_id for record in values}
    if set(oracle_should_allow) != record_ids:
        raise ValueError("Canary oracle must label every decision with no extras")
    valid = [record for record in values if oracle_should_allow[record.decision_id]]
    invalid = [record for record in values if not oracle_should_allow[record.decision_id]]
    unauthorized = [record for record in invalid if record.effective_allowed]
    false_denials = [record for record in valid if not record.effective_allowed]
    policy_unauthorized = [record for record in invalid if record.authority_allowed]
    policy_false_denials = [record for record in valid if not record.authority_allowed]
    false_denial_rate = len(false_denials) / len(valid) if valid else 0.0
    legacy_invalid_allows = [record for record in invalid if record.legacy_allowed]
    authority_invalid_allows = [record for record in invalid if record.authority_allowed]
    iir_legacy = len(legacy_invalid_allows) / len(invalid) if invalid else 0.0
    iir_authority = len(authority_invalid_allows) / len(invalid) if invalid else 0.0
    vkr_authority = (
        sum(record.authority_allowed for record in valid) / len(valid) if valid else 1.0
    )
    iir_effective = (
        sum(record.effective_allowed for record in invalid) / len(invalid)
        if invalid
        else 0.0
    )
    vkr_effective = (
        sum(record.effective_allowed for record in valid) / len(valid)
        if valid
        else 1.0
    )
    passed = bool(
        len(values) >= thresholds.minimum_decisions
        and len(unauthorized) <= thresholds.max_unauthorized_authority_allows
        and false_denial_rate <= thresholds.max_false_denial_rate
    )
    payload = {
        "schema": CANARY_REPORT_SCHEMA,
        "passed": passed,
        "observed_record_count": len(observed),
        "excluded_unenforced_decision_ids": sorted(
            record.decision_id for record in observed if not record.enforced
        ),
        "decision_count": len(values),
        "valid_decision_count": len(valid),
        "invalid_decision_count": len(invalid),
        "unauthorized_authority_allow_count": len(unauthorized),
        "authority_false_denial_count": len(false_denials),
        "authority_false_denial_rate": false_denial_rate,
        "unauthorized_effective_allow_count": len(unauthorized),
        "effective_false_denial_count": len(false_denials),
        "effective_false_denial_rate": false_denial_rate,
        "policy_unauthorized_allow_count": len(policy_unauthorized),
        "policy_false_denial_count": len(policy_false_denials),
        "legacy_iir": iir_legacy,
        "authority_iir": iir_authority,
        "authority_vkr": vkr_authority,
        "effective_iir": iir_effective,
        "effective_vkr": vkr_effective,
        "gate_metric_basis": "effective_allowed_on_enforced_records",
        "thresholds": _jsonable(thresholds),
        "decision_record_hashes": sorted(record.record_hash for record in values),
        "rollout_version_hash": next(iter(rollout_hashes), ""),
        "report_hash": "",
    }
    payload["report_hash"] = _sha256_json(
        {key: value for key, value in payload.items() if key != "report_hash"}
    )
    return payload


def build_canary_oracle_packet(
    records: Iterable[ShadowDecisionRecord],
) -> dict[str, Any]:
    """Build a hash-bound packet for an independent canary oracle reviewer."""

    observed = list(records)
    for record in observed:
        record.verify()
    enforced = sorted(
        (record for record in observed if record.enforced),
        key=lambda record: record.decision_id,
    )
    if not enforced:
        raise ValueError("Canary oracle packet requires enforced decisions")
    rollout_hashes = {record.rollout_version_hash for record in enforced}
    if len(rollout_hashes) != 1:
        raise ValueError("Canary oracle packet spans rollout version sets")
    evidence = {
        "schema": CANARY_ORACLE_PACKET_SCHEMA,
        "rollout_version_hash": next(iter(rollout_hashes)),
        "enforced_record_hashes": [record.record_hash for record in enforced],
    }
    return {
        **evidence,
        "decision_count": len(enforced),
        "reviewed_records": [
            {
                "record": record.as_dict(),
                "review": {
                    "reviewer": "",
                    "oracle_should_allow": None,
                    "notes": "",
                },
            }
            for record in enforced
        ],
        "evidence_hash": _sha256_json(evidence),
    }


def verify_canary_oracle_packet(
    packet: Mapping[str, Any],
    records: Iterable[ShadowDecisionRecord],
) -> dict[str, Any]:
    """Verify independent labels and return an exact oracle plus review report."""

    payload = copy.deepcopy(dict(packet))
    if payload.get("schema") != CANARY_ORACLE_PACKET_SCHEMA:
        raise ValueError("Unsupported canary oracle packet schema")
    observed = list(records)
    for record in observed:
        record.verify()
    enforced = sorted(
        (record for record in observed if record.enforced),
        key=lambda record: record.decision_id,
    )
    if not enforced:
        raise ValueError("Canary oracle packet requires enforced decisions")
    rollout_hashes = {record.rollout_version_hash for record in enforced}
    if len(rollout_hashes) != 1:
        raise ValueError("Canary oracle packet spans rollout version sets")
    rollout_version_hash = next(iter(rollout_hashes))
    expected_hashes = [record.record_hash for record in enforced]
    evidence = {
        "schema": CANARY_ORACLE_PACKET_SCHEMA,
        "rollout_version_hash": rollout_version_hash,
        "enforced_record_hashes": expected_hashes,
    }
    if payload.get("rollout_version_hash") != rollout_version_hash:
        raise ValueError("Canary oracle rollout version mismatch")
    if payload.get("enforced_record_hashes") != expected_hashes:
        raise ValueError("Canary oracle population does not match the ledger")
    if payload.get("evidence_hash") != _sha256_json(evidence):
        raise ValueError("Canary oracle evidence hash mismatch")
    rows = payload.get("reviewed_records")
    if not isinstance(rows, list) or len(rows) != len(enforced):
        raise ValueError("Canary oracle packet decision count mismatch")
    if int(payload.get("decision_count", -1)) != len(enforced):
        raise ValueError("Canary oracle packet decision count mismatch")

    current = {record.decision_id: record for record in enforced}
    reviewed_hashes: list[str] = []
    reviews: list[dict[str, Any]] = []
    oracle: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("record"), Mapping):
            raise ValueError("Canary oracle row is malformed")
        embedded = ShadowDecisionRecord.from_dict(row["record"])
        ledger_record = current.get(embedded.decision_id)
        if ledger_record is None or ledger_record.record_hash != embedded.record_hash:
            raise ValueError("Canary oracle record does not match the ledger")
        if embedded.record_hash in reviewed_hashes:
            raise ValueError("Canary oracle packet contains a duplicate record")
        reviewed_hashes.append(embedded.record_hash)
        review = row.get("review")
        if not isinstance(review, Mapping):
            raise ValueError(f"Canary oracle review is missing: {embedded.decision_id}")
        reviewer = str(review.get("reviewer") or "").strip()
        should_allow = review.get("oracle_should_allow")
        if not reviewer:
            raise ValueError(f"Canary oracle lacks reviewer: {embedded.decision_id}")
        if type(should_allow) is not bool:
            raise ValueError(
                f"Canary oracle label is not boolean: {embedded.decision_id}"
            )
        oracle[embedded.decision_id] = should_allow
        reviews.append(
            {
                "decision_id": embedded.decision_id,
                "reviewer": reviewer,
                "oracle_should_allow": should_allow,
                "notes": str(review.get("notes") or ""),
            }
        )
    if reviewed_hashes != expected_hashes:
        raise ValueError("Canary oracle review order/hash mismatch")
    reviews_sha256 = _sha256_json(reviews)
    report = {
        "schema": CANARY_ORACLE_REVIEW_REPORT_SCHEMA,
        "verified": True,
        "decision_count": len(enforced),
        "allow_count": sum(oracle.values()),
        "deny_count": len(oracle) - sum(oracle.values()),
        "reviewers": sorted({review["reviewer"] for review in reviews}),
        "reviewed_decision_ids": sorted(oracle),
        "rollout_version_hash": rollout_version_hash,
        "evidence_hash": payload["evidence_hash"],
        "reviews_sha256": reviews_sha256,
        "report_hash": "",
    }
    report["report_hash"] = _sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return {
        "oracle_should_allow": dict(sorted(oracle.items())),
        "review_report": report,
    }


def load_shadow_records_from_ledger(
    ledger_path: str | Path,
) -> list[ShadowDecisionRecord]:
    path = Path(ledger_path)
    if not path.is_file():
        raise FileNotFoundError(f"Authority ledger does not exist: {path}")
    ledger = AuthorityLedger(path)
    if not ledger.verify():
        raise ValueError("Authority ledger hash chain is invalid")
    version_hashes: set[str] = set()
    records: dict[str, ShadowDecisionRecord] = {}
    for event in ledger.read():
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload")
        if not isinstance(payload, Mapping):
            continue
        if event_type == "authority_rollout_versions_frozen":
            versions = RolloutVersionSet(**dict(payload))
            version_hashes.add(versions.version_hash)
        elif event_type == "authority_shadow_comparison":
            record = ShadowDecisionRecord.from_dict(payload)
            current = records.get(record.decision_id)
            if current is not None and current.record_hash != record.record_hash:
                raise ValueError(
                    "Authority ledger contains conflicting shadow comparisons: "
                    f"{record.decision_id}"
                )
            records[record.decision_id] = record
    unknown_versions = {
        record.rollout_version_hash for record in records.values()
    } - version_hashes
    if unknown_versions:
        raise ValueError(
            "Shadow comparisons lack a verified frozen version event: "
            f"{sorted(unknown_versions)}"
        )
    return [records[key] for key in sorted(records)]


def build_shadow_review_packet(
    records: Iterable[ShadowDecisionRecord],
    *,
    max_records: int = 50,
) -> dict[str, Any]:
    if max_records <= 0:
        raise ValueError("Shadow review max_records must be positive")
    values = list(records)
    for record in values:
        record.verify()
    population = [
        record
        for record in values
        if record.legacy_allowed != record.authority_allowed
        or record.reason_class == "authority_internal_error"
    ]
    grouped: dict[tuple[str, str], list[ShadowDecisionRecord]] = {}
    for record in sorted(
        population,
        key=lambda item: (
            item.taxonomy,
            item.reason_class,
            item.decision_id,
        ),
    ):
        grouped.setdefault((record.taxonomy, record.reason_class), []).append(record)
    sampled: list[ShadowDecisionRecord] = []
    while len(sampled) < min(max_records, len(population)):
        advanced = False
        for key in sorted(grouped):
            if grouped[key] and len(sampled) < max_records:
                sampled.append(grouped[key].pop(0))
                advanced = True
        if not advanced:
            break
    evidence = {
        "schema": SHADOW_REVIEW_PACKET_SCHEMA,
        "sampling_method": "deterministic_round_robin_by_taxonomy_and_reason",
        "population_record_hashes": sorted(
            record.record_hash for record in population
        ),
        "sample_record_hashes": [record.record_hash for record in sampled],
    }
    return {
        **evidence,
        "population_count": len(population),
        "sample_count": len(sampled),
        "allowed_dispositions": sorted(SHADOW_REVIEW_DISPOSITIONS),
        "sampled_records": [
            {
                "record": record.as_dict(),
                "review": {
                    "reviewer": "",
                    "disposition": "",
                    "notes": "",
                },
            }
            for record in sampled
        ],
        "evidence_hash": _sha256_json(evidence),
    }


def verify_shadow_review_packet(
    packet: Mapping[str, Any],
    records: Iterable[ShadowDecisionRecord],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(packet))
    if payload.get("schema") != SHADOW_REVIEW_PACKET_SCHEMA:
        raise ValueError("Unsupported shadow review packet schema")
    current = {record.decision_id: record for record in records}
    for record in current.values():
        record.verify()
    population = [
        record
        for record in current.values()
        if record.legacy_allowed != record.authority_allowed
        or record.reason_class == "authority_internal_error"
    ]
    expected_population_hashes = sorted(
        record.record_hash for record in population
    )
    if payload.get("population_record_hashes") != expected_population_hashes:
        raise ValueError("Shadow review population does not match the ledger")
    rows = payload.get("sampled_records")
    if not isinstance(rows, list):
        raise ValueError("Shadow review packet lacks sampled_records")
    sampled_hashes: list[str] = []
    dispositions: collections.Counter[str] = collections.Counter()
    reviewed_ids: list[str] = []
    verified_reviews: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("record"), Mapping):
            raise ValueError("Shadow review row is malformed")
        embedded = ShadowDecisionRecord.from_dict(row["record"])
        ledger_record = current.get(embedded.decision_id)
        if ledger_record is None or ledger_record.record_hash != embedded.record_hash:
            raise ValueError("Shadow review record does not match the ledger")
        if embedded.record_hash in sampled_hashes:
            raise ValueError("Shadow review packet contains a duplicate record")
        sampled_hashes.append(embedded.record_hash)
        review = row.get("review")
        if not isinstance(review, Mapping):
            raise ValueError(f"Shadow review is missing: {embedded.decision_id}")
        reviewer = str(review.get("reviewer") or "").strip()
        disposition = str(review.get("disposition") or "").strip()
        if not reviewer:
            raise ValueError(f"Shadow review lacks reviewer: {embedded.decision_id}")
        if disposition not in SHADOW_REVIEW_DISPOSITIONS:
            raise ValueError(
                f"Unsupported shadow review disposition: {embedded.decision_id}"
            )
        dispositions[disposition] += 1
        reviewed_ids.append(embedded.decision_id)
        verified_reviews.append(
            {
                "decision_id": embedded.decision_id,
                "reviewer": reviewer,
                "disposition": disposition,
                "notes": str(review.get("notes") or ""),
            }
        )
    if payload.get("sample_record_hashes") != sampled_hashes:
        raise ValueError("Shadow review sample order/hash does not match evidence")
    evidence = {
        "schema": SHADOW_REVIEW_PACKET_SCHEMA,
        "sampling_method": payload.get("sampling_method"),
        "population_record_hashes": payload.get("population_record_hashes"),
        "sample_record_hashes": payload.get("sample_record_hashes"),
    }
    if payload.get("evidence_hash") != _sha256_json(evidence):
        raise ValueError("Shadow review evidence hash mismatch")
    if int(payload.get("population_count", -1)) != len(population):
        raise ValueError("Shadow review population count mismatch")
    if int(payload.get("sample_count", -1)) != len(rows):
        raise ValueError("Shadow review sample count mismatch")
    report = {
        "schema": SHADOW_REVIEW_REPORT_SCHEMA,
        "verified": True,
        "population_count": len(population),
        "reviewed_sample_count": len(rows),
        "sample_coverage": len(rows) / len(population) if population else 1.0,
        "reviewed_decision_ids": sorted(reviewed_ids),
        "reviewers": sorted(
            {review["reviewer"] for review in verified_reviews}
        ),
        "disposition_counts": dict(sorted(dispositions.items())),
        "reviews_sha256": _sha256_json(verified_reviews),
        "evidence_hash": payload["evidence_hash"],
        "report_hash": "",
    }
    report["report_hash"] = _sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


class BundleRollbackController:
    """Atomically repoint CURRENT to a verified prior Bundle without deletion."""

    def __init__(
        self,
        bundle_root: str | Path,
        *,
        ledger: AuthorityLedger,
    ) -> None:
        self.bundle_root = Path(bundle_root).resolve()
        self.current_path = self.bundle_root / "CURRENT.json"
        self.lock_path = self.bundle_root / ".publication.lock"
        self.ledger = ledger

    def rollback(
        self,
        *,
        target_bundle_path: str | Path,
        expected_current_manifest_sha256: str,
    ) -> dict[str, Any]:
        with file_lock(self.lock_path):
            if not self.ledger.verify():
                raise ValueError("Refusing rollback with an invalid Authority ledger")
            current = _read_json(self.current_path)
            if current.get("schema") != CURRENT_POINTER_SCHEMA or current.get(
                "pointer_sha256"
            ) != _payload_hash(current, "pointer_sha256"):
                raise ValueError("CURRENT pointer is invalid")
            if current.get("manifest_sha256") != str(expected_current_manifest_sha256):
                raise ValueError("CURRENT changed before rollback")
            raw_target = Path(target_bundle_path)
            if raw_target.is_absolute():
                try:
                    relative = raw_target.resolve().relative_to(self.bundle_root)
                except ValueError as error:
                    raise ValueError("Rollback target is outside bundle root") from error
            else:
                relative = raw_target
            target_path = _safe_relative_path(
                self.bundle_root, str(relative), label="rollback target"
            )
            target = ImmutableBaseBundle.load(target_path, verify_artifacts=True)
            if target.bundle_id == str(current.get("bundle_id") or ""):
                raise ValueError("Rollback target is already CURRENT")
            prepared = {
                "schema": ROLLBACK_REPORT_SCHEMA,
                "from_bundle_id": str(current.get("bundle_id") or ""),
                "from_manifest_sha256": str(current.get("manifest_sha256") or ""),
                "to_bundle_id": target.bundle_id,
                "to_manifest_sha256": target.manifest_sha256,
                "target_bundle_path": str(target.path.relative_to(self.bundle_root)),
                "prepared_at": _utc_now(),
            }
            self.ledger.append("bundle_rollback_prepared", prepared)
            pointer = make_current_pointer(
                bundle_path=str(target.path.relative_to(self.bundle_root)),
                manifest=target.manifest,
                parent_bundle=str(current.get("bundle_id") or ""),
            )
            write_json_atomic(self.current_path, pointer)
            report = {
                **prepared,
                "current_pointer_sha256": pointer["pointer_sha256"],
                "committed_at": _utc_now(),
                "report_hash": "",
            }
            report["report_hash"] = sha256_json(
                {key: value for key, value in report.items() if key != "report_hash"}
            )
            self.ledger.append("bundle_rollback_committed", report)
            return report


__all__ = [
    "AuthorityRolloutController",
    "BundleRollbackController",
    "CANARY_REPORT_SCHEMA",
    "CANARY_ORACLE_PACKET_SCHEMA",
    "CANARY_ORACLE_REVIEW_REPORT_SCHEMA",
    "CanaryThresholds",
    "ROLLBACK_REPORT_SCHEMA",
    "ROLLOUT_VERSION_SCHEMA",
    "RolloutVersionSet",
    "SHADOW_RECORD_SCHEMA",
    "SHADOW_REVIEW_DISPOSITIONS",
    "SHADOW_REVIEW_PACKET_SCHEMA",
    "SHADOW_REVIEW_REPORT_SCHEMA",
    "ShadowDecisionRecord",
    "evaluate_canary",
    "build_canary_oracle_packet",
    "build_shadow_review_packet",
    "load_shadow_records_from_ledger",
    "verify_shadow_review_packet",
    "verify_canary_oracle_packet",
]
