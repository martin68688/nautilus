from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .bundle_publisher import PublicationReport, SleepTimePipeline, SleepTimePublisher
from .clean_replay import ReplayRegistration
from .evidence_graph import EvidenceGraph
from .memory_snapshot import (
    SessionOverlay,
    sha256_file,
    sha256_json,
    verify_bundle_directory,
    write_json_atomic,
)
from .domain_scope import normalize_transfer_scope
from .models import ClaimType, Operation
from .protocol_registry import ProtocolRegistry
from .replay_clause_publication import (
    ReplayClausePublication,
    publish_replay_clauses,
)
from .replay_certifier import ReplayIdentity, ReplayVerificationReport


CERTIFIED_REPLAY_REPORT_SCHEMA = "certified_replay_bundle_report_v1"


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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        output.append(value)
    return output


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = "".join(
        json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _merge_rows(
    existing: Iterable[Mapping[str, Any]],
    additions: Iterable[Mapping[str, Any]],
    *,
    key: str,
) -> list[dict[str, Any]]:
    output = {str(row[key]): copy.deepcopy(dict(row)) for row in existing}
    for value in additions:
        row = copy.deepcopy(dict(value))
        row_id = str(row[key])
        current = output.get(row_id)
        if current is not None and current != row:
            raise ValueError(f"Certified publication would mutate immutable {key}: {row_id}")
        output[row_id] = row
    return [output[row_id] for row_id in sorted(output)]


@dataclass(frozen=True)
class CertifiedPublicationResult:
    publication: PublicationReport
    certification_report: dict[str, Any]


class CertifiedBundlePublisher:
    """Publish only replay-scoped certifications; never bless the whole history."""

    def __init__(
        self,
        bundle_root: str | Path,
        graph: EvidenceGraph,
        registry: ProtocolRegistry,
    ):
        self.bundle_root = Path(bundle_root).resolve()
        self.graph = graph
        self.registry = registry

    def _validate_material(
        self,
        registrations: Iterable[ReplayRegistration],
        verifications: Mapping[str, ReplayVerificationReport],
    ) -> tuple[list[ReplayRegistration], dict[str, ReplayVerificationReport]]:
        values = list(registrations)
        if not values:
            raise ValueError("Certified publication requires a clean replay registration")
        reports = dict(verifications)
        seen_claims: set[str] = set()
        for registration in values:
            registration.verify()
            if registration.identity == ReplayIdentity.REQUIRE_HUMAN_REVIEW:
                raise ValueError("Human-review replay cannot enter a certified Bundle")
            if registration.old_claim_mutated or registration.authority_recovered_for_old_claim:
                raise ValueError("Certified publication cannot rewrite historical Claim authority")
            if not registration.replay_claim_id or registration.replay_claim_id == registration.original_claim_id:
                raise ValueError("Replay certification must use a new Claim ID")
            if registration.replay_claim_id in seen_claims:
                raise ValueError("Duplicate replay registration Claim")
            seen_claims.add(registration.replay_claim_id)
            report = reports.get(registration.verification_report_hash)
            if report is None:
                raise ValueError("Replay registration lacks its verification report")
            report.verify()
            if report.report_hash != registration.verification_report_hash:
                raise ValueError("Replay registration/report hash mismatch")
            if report.identity != registration.identity:
                raise ValueError("Replay registration/report identity mismatch")
            claim = self.graph.claims.get(registration.replay_claim_id)
            path = self.graph.paths.get(registration.path_id)
            if claim is None or path is None or path.claim_id != claim.claim_id:
                raise ValueError("Replay registration is absent from the EvidenceGraph")
            self.registry.resolve(claim.protocol_ref)
            if set(path.receipt_ids) != set(registration.receipt_ids):
                raise ValueError("Replay registration path/Receipt mismatch")
            for receipt_id in registration.receipt_ids:
                receipt = self.graph.receipts.get(receipt_id)
                if receipt is None or receipt.trust_status != "trusted_host":
                    raise ValueError("Certified replay path contains an untrusted Receipt")
                if receipt.artifact_id != registration.replay_artifact_id:
                    raise ValueError("Certified replay path reuses a historical Receipt")
        return values, reports

    def publish(
        self,
        *,
        new_version: str,
        overlay: SessionOverlay,
        registrations: Iterable[ReplayRegistration],
        verifications: Mapping[str, ReplayVerificationReport],
        expected_parent_manifest_sha256: str,
        bundle_id: str | None = None,
        replay_clause_publications: Iterable[ReplayClausePublication] = (),
        formal_bundle_validator: Callable[
            [Path], Mapping[str, Any]
        ]
        | None = None,
    ) -> CertifiedPublicationResult:
        registrations, reports = self._validate_material(registrations, verifications)
        replay_clause_publications = list(replay_clause_publications)
        registrations_by_hash = {
            value.registration_hash: value for value in registrations
        }
        replay_claim_ids = {value.replay_claim_id for value in registrations}
        seen_clause_ids: set[str] = set()
        seen_sop_ids: set[str] = set()
        for publication in replay_clause_publications:
            clause = publication.clause
            registration = registrations_by_hash.get(publication.registration_hash)
            if registration is None:
                raise ValueError("Replay clause lacks its replay registration")
            claim = self.graph.claims.get(registration.replay_claim_id)
            original = self.graph.claims.get(registration.original_claim_id)
            if claim is None or original is None:
                raise ValueError("Replay clause Claim lineage is unavailable")
            if clause.clause_id in seen_clause_ids or clause.sop_id in seen_sop_ids:
                raise ValueError("Replay clause publication IDs must be unique")
            seen_clause_ids.add(clause.clause_id)
            seen_sop_ids.add(clause.sop_id)
            if set(clause.claim_refs) != {registration.replay_claim_id}:
                raise ValueError("Replay clause must reference only the new replay Claim")
            if set(clause.claim_types) != {claim.claim_type.value}:
                raise ValueError("Replay clause Claim type mismatch")
            if set(clause.source_artifact_refs) != {original.subject_artifact_id}:
                raise ValueError("Replay clause must preserve only predecessor method lineage")
            if set(clause.receipt_refs) != set(registration.receipt_ids):
                raise ValueError("Replay clause must bind every trusted replay Receipt")
            if set(clause.protocol_scope) != {claim.protocol_ref.key()}:
                raise ValueError("Replay clause must use the replay Claim ProtocolRef")
            if clause.publication_class != "certified":
                raise ValueError("Replay clause publication must be certified")
            if Operation.GENERATE_CANDIDATE.value not in clause.permitted_operations:
                raise ValueError("Replay method clause must permit candidate generation")
            if clause.protocol_agnostic:
                raise ValueError("Replay clause cannot be protocol-agnostic")
            if normalize_transfer_scope(clause.transfer_scope) != clause.transfer_scope:
                raise ValueError("Replay clause transfer scope is missing or noncanonical")
            if not clause.source_task_ids or not clause.source_domains:
                raise ValueError("Replay clause requires explicit source task/domain lineage")
            if publication.verification_report_hash != registration.verification_report_hash:
                raise ValueError("Replay clause/verifier binding mismatch")
            contract = clause.contract_spec
            required_contract = {
                "replay_artifact_id": registration.replay_artifact_id,
                "registration_hash": registration.registration_hash,
                "verification_report_hash": registration.verification_report_hash,
                "predecessor_claim_id": registration.original_claim_id,
                "predecessor_clause_id": publication.source_clause_id,
                "historical_metric_used_as_evidence": False,
            }
            if any(contract.get(key) != value for key, value in required_contract.items()):
                raise ValueError("Replay clause contract is not bound to certification material")
        if replay_clause_publications and {
            publication.clause.claim_refs[0]
            for publication in replay_clause_publications
        } != replay_claim_ids:
            raise ValueError("Every published replay Claim requires exactly one replay clause")
        certification_report: dict[str, Any] = {}
        replay_clause_report: dict[str, Any] = {}
        resolved_bundle_id = bundle_id or ""

        def audit(_context):
            return {
                "status": "passed",
                "registration_count": len(registrations),
                "old_claim_mutation_count": 0,
                "historical_authority_recovery_count": 0,
            }

        def claim_decomposition(_context):
            return {
                "status": "passed",
                "new_claim_ids": sorted(item.replay_claim_id for item in registrations),
            }

        def distillation(_context):
            return {
                "status": "passed",
                "mode": (
                    "replay_scoped_clause_publication"
                    if replay_clause_publications
                    else "authority_only_replay_certification"
                ),
                "blanket_clause_upgrade": False,
            }

        def build(context):
            parent = context["parent_bundle"]
            candidate = context["candidate_dir"]
            if str(parent.manifest.get("certification_level") or "") not in {
                "raw_audited",
                "certified",
            }:
                raise ValueError("Clean Replay requires a raw-audited or certified parent Bundle")
            nonlocal resolved_bundle_id
            resolved_bundle_id = bundle_id or f"{parent.bundle_id}::certified::{new_version}"
            shutil.copytree(parent.path, candidate, dirs_exist_ok=True)
            manifest_path = candidate / "manifest.json"
            sums_path = candidate / "SHA256SUMS"
            stale_validation_path = (
                candidate / "reports" / "validation_report.json"
            )
            manifest = _read_json(manifest_path)
            manifest_path.unlink()
            if sums_path.exists():
                sums_path.unlink()
            # This report certifies the parent and was intentionally written
            # after its manifest/SHA256SUMS.  It must not be copied forward as
            # if it certified the new replay-scoped child.
            if stale_validation_path.exists():
                stale_validation_path.unlink()

            authority = candidate / "authority"
            authority.mkdir(exist_ok=True)
            parent_claim_rows = _read_jsonl(authority / "claims.jsonl")
            parent_claim_ids = {str(row.get("claim_id") or "") for row in parent_claim_rows}
            missing_predecessors = sorted(
                {
                    item.original_claim_id
                    for item in registrations
                    if item.original_claim_id not in parent_claim_ids
                }
            )
            if missing_predecessors:
                raise ValueError(
                    f"Certified replay parent lacks predecessor Claims: {missing_predecessors}"
                )
            new_claims = [self.graph.claims[item.replay_claim_id] for item in registrations]
            receipt_ids = sorted(
                {receipt_id for item in registrations for receipt_id in item.receipt_ids}
            )
            new_receipts = [self.graph.receipts[receipt_id] for receipt_id in receipt_ids]
            new_paths = [self.graph.paths[item.path_id] for item in registrations]
            _write_jsonl_atomic(
                authority / "claims.jsonl",
                _merge_rows(
                    parent_claim_rows,
                    [_jsonable(value) for value in new_claims],
                    key="claim_id",
                ),
            )
            merged_receipts = _merge_rows(
                _read_jsonl(authority / "receipts.jsonl"),
                [_jsonable(value) for value in new_receipts],
                key="receipt_id",
            )
            _write_jsonl_atomic(authority / "receipts.jsonl", merged_receipts)
            _write_jsonl_atomic(
                authority / "replay_receipts.jsonl",
                _merge_rows(
                    _read_jsonl(authority / "replay_receipts.jsonl"),
                    [_jsonable(value) for value in new_receipts],
                    key="receipt_id",
                ),
            )
            _write_jsonl_atomic(
                authority / "replay_paths.jsonl",
                _merge_rows(
                    _read_jsonl(authority / "replay_paths.jsonl"),
                    [_jsonable(value) for value in new_paths],
                    key="path_id",
                ),
            )
            _write_jsonl_atomic(
                authority / "replay_verifications.jsonl",
                _merge_rows(
                    _read_jsonl(authority / "replay_verifications.jsonl"),
                    [report.as_dict() for report in reports.values()],
                    key="report_hash",
                ),
            )
            _write_jsonl_atomic(
                authority / "replay_registrations.jsonl",
                _merge_rows(
                    _read_jsonl(authority / "replay_registrations.jsonl"),
                    [item.as_dict() for item in registrations],
                    key="registration_hash",
                ),
            )

            replay_clause_report.clear()
            replay_clause_report.update(
                publish_replay_clauses(
                    candidate,
                    replay_clause_publications,
                    bundle_id=resolved_bundle_id,
                    parent_bundle_id=parent.bundle_id,
                )
            )
            if replay_clause_publications:
                reports_dir = candidate / "reports"
                reports_dir.mkdir(exist_ok=True)
                write_json_atomic(
                    reports_dir / "replay_clause_publication_report.json",
                    replay_clause_report,
                )

            replay_claim_ids = {item.replay_claim_id for item in registrations}
            original_claim_ids = {item.original_claim_id for item in registrations}
            unreplayed_score_claims = sorted(
                claim.claim_id
                for claim in self.graph.claims.values()
                if claim.claim_type == ClaimType.SCORE
                and claim.claim_id not in replay_claim_ids
                and claim.claim_id not in original_claim_ids
            )
            report_payload = {
                "schema": CERTIFIED_REPLAY_REPORT_SCHEMA,
                "parent_bundle_id": parent.bundle_id,
                "parent_manifest_sha256": parent.manifest_sha256,
                "bundle_version": str(new_version),
                "registration_hashes": sorted(item.registration_hash for item in registrations),
                "verification_report_hashes": sorted(reports),
                "replay_claim_ids": sorted(replay_claim_ids),
                "predecessor_claim_ids": sorted(original_claim_ids),
                "method_preserved_count": sum(
                    item.identity == ReplayIdentity.METHOD_PRESERVED for item in registrations
                ),
                "successor_method_count": sum(
                    item.identity == ReplayIdentity.SUCCESSOR_METHOD for item in registrations
                ),
                "old_claim_mutation_count": 0,
                "historical_authority_recovery_count": 0,
                "blanket_clause_upgrade": False,
                "replay_clause_ids": sorted(
                    publication.clause.clause_id
                    for publication in replay_clause_publications
                ),
                "replay_sop_ids": sorted(
                    publication.clause.sop_id
                    for publication in replay_clause_publications
                ),
                "predecessor_clause_ids": sorted(
                    publication.source_clause_id
                    for publication in replay_clause_publications
                ),
                "replay_clause_publication_report_hash": replay_clause_report.get(
                    "report_hash", ""
                ),
                "unreplayed_score_claim_ids": unreplayed_score_claims,
                "report_hash": "",
            }
            report_payload["report_hash"] = sha256_json(
                {key: value for key, value in report_payload.items() if key != "report_hash"}
            )
            reports_dir = candidate / "reports"
            reports_dir.mkdir(exist_ok=True)
            write_json_atomic(
                reports_dir / "replay_certification_report.json", report_payload
            )
            certification_report.clear()
            certification_report.update(copy.deepcopy(report_payload))

            registry_dir = candidate / "protocol_registry"
            registry_dir.mkdir(exist_ok=True)
            for claim in new_claims:
                spec = self.registry.resolve(claim.protocol_ref)
                filename = f"{spec.protocol_id}-{spec.version}.json"
                if "/" in filename or filename.startswith("."):
                    raise ValueError("Unsafe ProtocolSpec filename")
                target = registry_dir / filename
                if target.exists():
                    current_spec = _read_json(target)
                    declared_hash = str(current_spec.get("canonical_hash") or "")
                    if declared_hash and declared_hash != spec.canonical_hash:
                        raise ValueError("Published ProtocolSpec version is immutable")
                write_json_atomic(target, spec.as_dict())
            registry_hashes = {
                path.relative_to(registry_dir).as_posix(): sha256_file(path)
                for path in sorted(registry_dir.rglob("*"))
                if path.is_file()
            }
            protocol_registry_hash = sha256_json(registry_hashes)

            build_report_path = candidate / str(manifest.get("build_report") or "")
            if build_report_path.is_file():
                build_report = _read_json(build_report_path)
                if build_report.get("schema") == "memory_bundle_build_report_v1":
                    build_report.update(
                        {
                            "bundle_id": bundle_id
                            or resolved_bundle_id,
                            "bundle_version": str(new_version),
                            "parent_bundle": parent.bundle_id,
                            "certification_level": "certified",
                            "replay_certification_report": "reports/replay_certification_report.json",
                            "replay_registration_count": len(registrations),
                            "blanket_clause_upgrade": False,
                        }
                    )
                    write_json_atomic(build_report_path, build_report)

            artifact_hashes: dict[str, str] = {}
            for path in sorted(candidate.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(candidate).as_posix()
                if relative in {"manifest.json", "SHA256SUMS"}:
                    continue
                artifact_hashes[relative] = sha256_file(path)
            lineage_inputs = {
                "clauses": artifact_hashes.get("sop/clauses.jsonl"),
                "containers": artifact_hashes.get("sop/containers.json"),
                "derivations": artifact_hashes.get("authority/derivations.jsonl"),
                "sop_graph": artifact_hashes.get("sop/graph.json"),
            }
            lineage_hash = (
                sha256_json(lineage_inputs)
                if all(lineage_inputs.values())
                else str(manifest.get("lineage_hash") or "")
            )
            manifest.update(
                {
                    "bundle_id": resolved_bundle_id,
                    "bundle_version": str(new_version),
                    "parent_bundle": parent.bundle_id,
                    "certification_level": "certified",
                    "protocol_registry_hash": protocol_registry_hash,
                    "graph_hashes": {
                        "runforest": artifact_hashes["runforest/graph.json"]
                    },
                    "index_hashes": {
                        "runforest": artifact_hashes["runforest/index.npz"],
                        "clauses": artifact_hashes[
                            "runforest/clause_index.npz"
                        ],
                    },
                    "lineage_hash": lineage_hash,
                    "artifact_hashes": artifact_hashes,
                    "manifest_sha256": "",
                }
            )
            manifest["manifest_sha256"] = sha256_json(
                {key: value for key, value in manifest.items() if key != "manifest_sha256"}
            )
            write_json_atomic(manifest_path, manifest)
            checksum_files = [
                path
                for path in sorted(candidate.rglob("*"))
                if path.is_file()
                and path.relative_to(candidate).as_posix() != "SHA256SUMS"
            ]
            encoded = "".join(
                f"{sha256_file(path)}  {path.relative_to(candidate).as_posix()}\n"
                for path in checksum_files
            ).encode("utf-8")
            descriptor, temporary = tempfile.mkstemp(
                dir=candidate, prefix=".SHA256SUMS.", suffix=".tmp"
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, sums_path)
            except Exception:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise

        def derivation_validation(context):
            parent = context["parent_bundle"]
            candidate = context["candidate_dir"]
            unchanged = {}
            for relative in (
                "sop/clauses.jsonl",
                "sop/containers.json",
                "sop/graph.json",
                "runforest/graph.json",
                "runforest/index.npz",
            ):
                source = parent.path / relative
                target = candidate / relative
                if source.is_file() and target.is_file():
                    unchanged[relative] = sha256_file(source) == sha256_file(target)
            valid = (
                replay_clause_report.get("status") == "passed"
                and replay_clause_report.get("old_semantic_rows_mutated") is False
                if replay_clause_publications
                else all(unchanged.values())
            )
            return {
                "status": "passed" if valid else "failed",
                "unchanged_parent_semantic_artifacts": unchanged,
                "old_claim_mutation_count": 0,
                "replay_clause_ids": replay_clause_report.get("clause_ids", []),
                "old_semantic_rows_mutated": replay_clause_report.get(
                    "old_semantic_rows_mutated", False
                ),
            }

        def visibility_validation(_context):
            return {
                "status": "passed",
                "blanket_clause_upgrade": False,
                "unreplayed_history_authority_changed": False,
                "replay_clause_ids": replay_clause_report.get("clause_ids", []),
            }

        def bundle_validation(context):
            manifest = verify_bundle_directory(
                context["candidate_dir"], verify_artifacts=True, allow_staging=True
            )
            formal_validation = (
                dict(formal_bundle_validator(context["candidate_dir"]))
                if formal_bundle_validator is not None
                else {"valid": True, "status": "not_requested"}
            )
            valid = bool(
                manifest.get("certification_level") == "certified"
                and manifest.get("parent_bundle") == context["parent_bundle"].bundle_id
                and certification_report.get("old_claim_mutation_count") == 0
                and certification_report.get("blanket_clause_upgrade") is False
                and formal_validation.get("valid") is True
            )
            return {
                "valid": valid,
                "manifest_sha256": manifest.get("manifest_sha256"),
                "formal_validation": formal_validation,
            }

        pipeline = SleepTimePipeline(
            audit=audit,
            claim_decomposition=claim_decomposition,
            distillation=distillation,
            build_candidate=build,
            derivation_validation=derivation_validation,
            visibility_validation=visibility_validation,
            bundle_validation=bundle_validation,
        )
        publication = SleepTimePublisher(self.bundle_root).publish(
            new_version=str(new_version),
            overlay=overlay,
            pipeline=pipeline,
            expected_parent_manifest_sha256=str(expected_parent_manifest_sha256),
        )
        return CertifiedPublicationResult(
            publication=publication,
            certification_report=copy.deepcopy(certification_report),
        )


__all__ = [
    "CERTIFIED_REPLAY_REPORT_SCHEMA",
    "CertifiedBundlePublisher",
    "CertifiedPublicationResult",
]
