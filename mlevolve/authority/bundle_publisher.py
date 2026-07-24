from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .memory_snapshot import (
    CURRENT_POINTER_SCHEMA,
    ImmutableBaseBundle,
    SessionOverlay,
    _fsync_directory,
    _payload_hash,
    _read_json,
    _safe_relative_path,
    file_lock,
    make_current_pointer,
    sha256_file,
    sha256_json,
    verify_bundle_directory,
    write_json_atomic,
)


PUBLICATION_EVENT_SCHEMA = "bundle_publication_event_v1"
PUBLICATION_REPORT_SCHEMA = "sleep_time_publication_report_v1"
REQUIRED_PIPELINE_REPORTS = (
    "audit",
    "claim_decomposition",
    "distillation",
    "derivation_validation",
    "visibility_validation",
    "bundle_validation",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PublicationConflictError(RuntimeError):
    pass


class PublicationValidationError(RuntimeError):
    pass


def _stage_passed(report: Any) -> bool:
    if report is True:
        return True
    if not isinstance(report, Mapping):
        return False
    if report.get("valid") is True or report.get("passed") is True:
        return True
    return str(report.get("status") or "").lower() in {
        "ok",
        "pass",
        "passed",
        "valid",
        "complete",
        "completed",
    }


def _validate_authority_snapshot_binding(
    payload: Mapping[str, Any],
    *,
    event_kind: str,
) -> dict[str, Any]:
    pointer = payload.get("authority_snapshot_pointer")
    if not isinstance(pointer, Mapping) or pointer.get("schema") != (
        "authority_snapshot_pointer_v1"
    ):
        raise PublicationValidationError(
            f"{event_kind} lacks an immutable Authority snapshot pointer"
        )
    path = Path(str(pointer.get("path") or "")).resolve()
    digest = str(pointer.get("sha256") or "")
    if not path.is_file() or len(digest) != 64 or sha256_file(path) != digest:
        raise PublicationValidationError(
            f"{event_kind} Authority snapshot hash mismatch"
        )
    snapshot = _read_json(path)
    if not isinstance(snapshot, Mapping):
        raise PublicationValidationError(
            f"{event_kind} Authority snapshot is not an object"
        )
    sections = {
        name: snapshot.get(name)
        for name in ("claims", "receipts", "paths", "decisions")
    }
    if not all(isinstance(value, Mapping) for value in sections.values()):
        raise PublicationValidationError(
            f"{event_kind} Authority snapshot graph is incomplete"
        )
    claims = sections["claims"]
    receipts = sections["receipts"]
    decisions = sections["decisions"]
    artifact_id = str(payload.get("artifact_id") or payload.get(
        "target_artifact_id"
    ) or "")
    claim_refs = [str(value) for value in payload.get("claim_refs") or []]
    if event_kind == "Result Fact":
        if not claim_refs:
            raise PublicationValidationError("Result Fact has no Claim refs")
        for claim_ref in claim_refs:
            row = claims.get(claim_ref)
            if not isinstance(row, Mapping) or str(row.get("claim_id")) != (
                claim_ref
            ) or str(row.get("subject_artifact_id")) != artifact_id:
                raise PublicationValidationError(
                    "Result Fact Claim does not resolve in its Authority snapshot"
                )
        declared_types = {
            str(value) for value in payload.get("claim_types") or []
        }
        resolved_types = {
            str(claims[claim_ref].get("claim_type") or "")
            for claim_ref in claim_refs
        }
        if declared_types != resolved_types:
            raise PublicationValidationError(
                "Result Fact Claim types do not match its Authority snapshot"
            )
    else:
        edge_claim_ref = str(payload.get("edge_claim_ref") or "")
        edge_claim = claims.get(edge_claim_ref)
        expected_type = (
            "causal_attribution"
            if payload.get("kind") == "causal"
            else "experience_adoption"
        )
        if (
            not edge_claim_ref
            or not isinstance(edge_claim, Mapping)
            or str(edge_claim.get("claim_id")) != edge_claim_ref
            or str(edge_claim.get("claim_type")) != expected_type
            or str(edge_claim.get("subject_artifact_id")) != artifact_id
        ):
            raise PublicationValidationError(
                f"{event_kind} Claim does not resolve in its Authority snapshot"
            )
    decision_refs = [
        str(value) for value in payload.get("authority_decision_refs") or []
    ]
    if len(decision_refs) != 1:
        raise PublicationValidationError(
            f"{event_kind} must bind exactly one Authority decision"
        )
    decision = decisions.get(decision_refs[0])
    if (
        not isinstance(decision, Mapping)
        or str(decision.get("decision_id")) != decision_refs[0]
        or str(decision.get("artifact_id")) != artifact_id
        or str(decision.get("operation"))
        != str(
            payload.get("operation")
            or (
                "promote_result"
                if event_kind == "Result Fact"
                else ""
            )
        )
        or str(decision.get("outcome"))
        not in {"allow", "allow_with_warning"}
    ):
        raise PublicationValidationError(
            f"{event_kind} Authority decision binding is invalid"
        )
    if event_kind != "Result Fact" and str(decision.get("claim_id")) != str(
        payload.get("edge_claim_ref") or ""
    ):
        raise PublicationValidationError(
            f"{event_kind} decision does not authorize its edge Claim"
        )
    receipt_refs = {
        str(value) for value in payload.get("receipt_refs") or []
    }
    if event_kind != "Result Fact":
        receipt_refs.update(
            str(value)
            for field in (
                "static_receipt_refs",
                "runtime_receipt_refs",
                "counterfactual_receipt_refs",
            )
            for value in payload.get(field) or []
        )
    if not receipt_refs or any(
        not isinstance(receipts.get(receipt_ref), Mapping)
        or str(receipts[receipt_ref].get("receipt_id")) != receipt_ref
        for receipt_ref in receipt_refs
    ):
        raise PublicationValidationError(
            f"{event_kind} Receipt refs do not resolve in its Authority snapshot"
        )
    if str(snapshot.get("policy_version") or "") != str(
        payload.get("authority_policy_version") or ""
    ):
        raise PublicationValidationError(
            f"{event_kind} Authority policy snapshot mismatch"
        )
    return dict(snapshot)


def classify_writeback_events(overlay_snapshot: str | Path) -> dict[str, Any]:
    """Validate and partition Result Facts, Adoption Edges, and Causal Edges."""

    root = Path(overlay_snapshot).resolve()
    manifest = _read_json(root / "overlay_manifest.json")
    if manifest.get("manifest_sha256") != _payload_hash(
        manifest, "manifest_sha256"
    ):
        raise PublicationValidationError("Overlay manifest hash mismatch")
    events_path = root / "events.jsonl"
    events = []
    parent_hash = ""
    for line_number, line in enumerate(
        events_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise PublicationValidationError(
                f"Overlay event line {line_number} is invalid"
            ) from error
        event = SessionOverlay._event_from_dict(raw)
        if (
            event.sequence != len(events) + 1
            or event.parent_event_hash != parent_hash
        ):
            raise PublicationValidationError("Overlay event chain is broken")
        events.append(event)
        parent_hash = event.event_hash
    if (
        manifest.get("event_count") != len(events)
        or manifest.get("last_event_hash") != parent_hash
        or manifest.get("events_sha256") != sha256_file(events_path)
    ):
        raise PublicationValidationError("Overlay manifest does not bind events")
    result_facts = []
    adoption_edges = []
    causal_edges = []
    other_events = []
    seen_keys: set[str] = set()
    adoption_keys: set[tuple[str, str]] = set()
    for event in events:
        payload = event.payload
        if (
            event.event_type == "memory_claim"
            and payload.get("publication_class") == "result_fact"
        ):
            key = str(payload.get("idempotency_key") or "")
            if not key or key in seen_keys:
                raise PublicationValidationError(
                    "Result Fact has a missing/duplicate idempotency key"
                )
            if payload.get("derived_from_refs") != []:
                raise PublicationValidationError(
                    "Result Fact cannot contain derivation lineage"
                )
            _validate_authority_snapshot_binding(
                payload,
                event_kind="Result Fact",
            )
            code_hash = str(payload.get("code_sha256") or "")
            pointer = payload.get("artifact_pointer") or {}
            if (
                len(code_hash) != 64
                or pointer.get("node_id") != payload.get("artifact_id")
                or not pointer.get("journal_path")
            ):
                raise PublicationValidationError(
                    "Result Fact does not resolve to an immutable code artifact"
                )
            journal_path = Path(str(pointer["journal_path"])).resolve()
            if not journal_path.is_file():
                raise PublicationValidationError(
                    "Result Fact Journal artifact is missing"
                )
            declared_journal_hash = str(
                pointer.get("journal_sha256") or ""
            )
            if declared_journal_hash and (
                sha256_file(journal_path) != declared_journal_hash
            ):
                raise PublicationValidationError(
                    "Result Fact Journal hash mismatch"
                )
            try:
                journal = json.loads(journal_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise PublicationValidationError(
                    "Result Fact Journal is unreadable"
                ) from error
            matching_nodes = [
                node
                for node in journal.get("nodes") or []
                if isinstance(node, dict)
                and str(node.get("id") or "")
                == str(payload.get("artifact_id") or "")
            ]
            if len(matching_nodes) != 1 or hashlib.sha256(
                str(matching_nodes[0].get("code") or "").encode("utf-8")
            ).hexdigest() != code_hash:
                raise PublicationValidationError(
                    "Result Fact does not resolve to the declared node code"
                )
            seen_keys.add(key)
            result_facts.append(event.as_dict())
            continue
        if event.event_type == "memory_derivation_edge":
            kind = str(payload.get("kind") or "")
            key = str(payload.get("idempotency_key") or "")
            if kind not in {"adoption", "causal"}:
                raise PublicationValidationError(
                    "Unknown memory derivation edge kind"
                )
            if not key or key in seen_keys:
                raise PublicationValidationError(
                    "Memory derivation edge has a missing/duplicate key"
                )
            identity_payload = {
                name: value
                for name, value in payload.items()
                if name not in {"edge_id", "edge_hash"}
            }
            edge_hash = sha256_json(identity_payload)
            if (
                payload.get("schema") != "experience_derivation_edge_v1"
                or payload.get("edge_hash") != edge_hash
                or payload.get("edge_id")
                != f"experience_edge::{edge_hash[:24]}"
                or not payload.get("source_claim_refs")
                or not payload.get("target_artifact_id")
                or not payload.get("authority_decision_refs")
                or not payload.get("static_receipt_refs")
                or not payload.get("runtime_receipt_refs")
            ):
                raise PublicationValidationError(
                    "Memory derivation edge lacks authorized lineage evidence"
                )
            _validate_authority_snapshot_binding(
                payload,
                event_kind=(
                    "Causal Edge" if kind == "causal" else "Adoption Edge"
                ),
            )
            edge_key = (
                str(payload["target_artifact_id"]),
                str(payload["contract_hash"]),
            )
            if kind == "adoption":
                adoption_keys.add(edge_key)
                adoption_edges.append(event.as_dict())
            else:
                if edge_key not in adoption_keys:
                    raise PublicationValidationError(
                        "Causal Edge has no preceding authorized Adoption Edge"
                    )
                if not payload.get("counterfactual_receipt_refs"):
                    raise PublicationValidationError(
                        "Causal Edge lacks counterfactual evidence"
                    )
                causal_edges.append(event.as_dict())
            seen_keys.add(key)
            continue
        other_events.append(event.as_dict())
    report = {
        "schema": "writeback_event_inventory_v1",
        "result_fact_count": len(result_facts),
        "adoption_edge_count": len(adoption_edges),
        "causal_edge_count": len(causal_edges),
        "other_event_count": len(other_events),
        "result_facts": result_facts,
        "adoption_edges": adoption_edges,
        "causal_edges": causal_edges,
        "other_events": other_events,
        "inventory_hash": "",
    }
    report["inventory_hash"] = _payload_hash(report, "inventory_hash")
    return report


def _validate_positive_distillation_report(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    typed_event_count = int(plan.get("result_fact_count") or 0) + int(
        plan.get("adoption_edge_count") or 0
    ) + int(plan.get("causal_edge_count") or 0)
    if typed_event_count == 0:
        return
    if str(report.get("plan_hash") or report.get("writeback_plan_hash") or "") != str(
        plan.get("plan_hash") or ""
    ):
        raise PublicationValidationError(
            "Sleep-time distillation did not bind the typed writeback plan"
        )
    expected_result = int(plan.get("positive_result_candidate_count") or 0)
    expected_adopted = int(plan.get("positive_adopted_candidate_count") or 0)
    if (
        int(report.get("positive_result_count") or 0) != expected_result
        or int(report.get("positive_adopted_count") or 0) != expected_adopted
    ):
        raise PublicationValidationError(
            "Sleep-time distillation did not separately consume Result/Adoption inputs"
        )
    consumed = {
        str(value)
        for value in (
            report.get("consumed_event_ids")
            or report.get("source_event_ids")
            or []
        )
    }
    if consumed != {
        str(value) for value in plan.get("consumed_event_ids") or []
    }:
        raise PublicationValidationError(
            "Sleep-time distillation did not account for every typed event"
        )


@dataclass
class SleepTimePipeline:
    """Ordered callbacks for the plan-mandated sleep-time production stages."""

    audit: Callable[[dict[str, Any]], Mapping[str, Any]]
    claim_decomposition: Callable[[dict[str, Any]], Mapping[str, Any]]
    distillation: Callable[[dict[str, Any]], Mapping[str, Any]]
    build_candidate: Callable[[dict[str, Any]], Any]
    derivation_validation: Callable[[dict[str, Any]], Mapping[str, Any]]
    visibility_validation: Callable[[dict[str, Any]], Mapping[str, Any]]
    bundle_validation: Callable[[dict[str, Any]], Mapping[str, Any]]

    def __call__(
        self,
        parent_bundle: ImmutableBaseBundle,
        overlay_snapshot: Path,
        candidate_dir: Path,
    ) -> dict[str, Any]:
        writeback_inventory = classify_writeback_events(overlay_snapshot)
        from .writeback_distillation import build_positive_writeback_plan

        positive_writeback_plan = build_positive_writeback_plan(
            writeback_inventory
        )
        context: dict[str, Any] = {
            "parent_bundle": parent_bundle,
            "parent_bundle_path": parent_bundle.path,
            "overlay_snapshot": overlay_snapshot,
            "candidate_dir": candidate_dir,
            "reports": {},
            "writeback_inventory": writeback_inventory,
            "positive_writeback_plan": positive_writeback_plan,
        }
        for name, callback in (
            ("audit", self.audit),
            ("claim_decomposition", self.claim_decomposition),
            ("distillation", self.distillation),
        ):
            report = dict(callback(context))
            context["reports"][name] = report
            if not _stage_passed(report):
                raise PublicationValidationError(
                    f"Sleep-time stage failed before build: {name}"
                )
            if name == "distillation":
                _validate_positive_distillation_report(
                    report,
                    positive_writeback_plan,
                )
        self.build_candidate(context)
        for name, callback in (
            ("derivation_validation", self.derivation_validation),
            ("visibility_validation", self.visibility_validation),
            ("bundle_validation", self.bundle_validation),
        ):
            report = dict(callback(context))
            context["reports"][name] = report
            if not _stage_passed(report):
                raise PublicationValidationError(
                    f"Sleep-time validation failed: {name}"
                )
        return copy.deepcopy(context["reports"])


@dataclass
class PublicationReport:
    publication_id: str
    parent_bundle_id: str
    parent_manifest_sha256: str
    bundle_id: str
    bundle_version: str
    bundle_manifest_sha256: str
    overlay_manifest_sha256: str
    overlay_events_sha256: str
    bundle_path: str
    pipeline_reports: dict[str, Any]
    current_pointer_sha256: str
    published_at: str = field(default_factory=_utc_now)
    report_hash: str = ""
    schema: str = PUBLICATION_REPORT_SCHEMA

    def finalize(self) -> PublicationReport:
        payload = dataclasses.asdict(self)
        payload.pop("report_hash", None)
        self.report_hash = sha256_json(payload)
        return self

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class SleepTimePublisher:
    """Crash-safe immutable Bundle publication with atomic CURRENT swap."""

    def __init__(self, bundle_root: str | Path) -> None:
        self.bundle_root = Path(bundle_root).resolve()
        self.bundle_root.mkdir(parents=True, exist_ok=True)
        self.bundles_dir = self.bundle_root / "bundles"
        self.bundles_dir.mkdir(exist_ok=True)
        self.current_path = self.bundle_root / "CURRENT.json"
        self.lock_path = self.bundle_root / ".publication.lock"
        self.ledger_path = self.bundle_root / "publication_ledger.jsonl"

    def _read_current(self) -> dict[str, Any]:
        pointer = _read_json(self.current_path)
        if not isinstance(pointer, dict):
            raise ValueError("CURRENT pointer must be an object")
        if pointer.get("schema") != CURRENT_POINTER_SCHEMA:
            raise ValueError("Unsupported CURRENT pointer schema")
        if pointer.get("pointer_sha256") != _payload_hash(
            pointer, "pointer_sha256"
        ):
            raise ValueError("CURRENT pointer hash mismatch")
        return pointer

    def _parent_bundle(self, pointer: Mapping[str, Any]) -> ImmutableBaseBundle:
        parent_path = _safe_relative_path(
            self.bundle_root,
            str(pointer.get("bundle_path") or ""),
            label="CURRENT bundle",
        )
        parent = ImmutableBaseBundle.load(parent_path, verify_artifacts=True)
        if pointer.get("manifest_sha256") != parent.manifest_sha256:
            raise ValueError("CURRENT does not bind its Base Bundle")
        return parent

    def _ledger_events(self) -> list[dict[str, Any]]:
        if not self.ledger_path.exists():
            return []
        output: list[dict[str, Any]] = []
        parent = ""
        for line_number, line in enumerate(
            self.ledger_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"Publication ledger line {line_number} is not an object")
            if raw.get("schema") != PUBLICATION_EVENT_SCHEMA:
                raise ValueError("Unsupported publication ledger schema")
            if raw.get("sequence") != len(output) + 1:
                raise ValueError("Publication ledger sequence is broken")
            if raw.get("parent_event_hash") != parent:
                raise ValueError("Publication ledger hash chain is broken")
            if raw.get("event_hash") != _payload_hash(raw, "event_hash"):
                raise ValueError("Publication ledger event hash mismatch")
            output.append(raw)
            parent = str(raw["event_hash"])
        return output

    def _append_ledger(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        events = self._ledger_events()
        event = {
            "schema": PUBLICATION_EVENT_SCHEMA,
            "sequence": len(events) + 1,
            "event_type": str(event_type),
            "payload": copy.deepcopy(dict(payload)),
            "created_at": _utc_now(),
            "parent_event_hash": events[-1]["event_hash"] if events else "",
            "event_hash": "",
        }
        event["event_hash"] = _payload_hash(event, "event_hash")
        encoded = (
            json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with self.ledger_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.bundle_root)
        return event

    @staticmethod
    def _validate_pipeline_reports(reports: Mapping[str, Any]) -> None:
        missing = [name for name in REQUIRED_PIPELINE_REPORTS if name not in reports]
        failed = [
            name
            for name in REQUIRED_PIPELINE_REPORTS
            if name in reports and not _stage_passed(reports[name])
        ]
        if missing or failed:
            raise PublicationValidationError(
                f"Incomplete sleep-time pipeline reports: missing={missing} failed={failed}"
            )

    @staticmethod
    def _fault(
        fault_injector: Callable[[str], Any] | None,
        point: str,
    ) -> None:
        if fault_injector is not None:
            fault_injector(point)

    def publish(
        self,
        *,
        new_version: str,
        overlay: SessionOverlay,
        pipeline: Callable[[ImmutableBaseBundle, Path, Path], Mapping[str, Any]],
        expected_parent_manifest_sha256: str | None = None,
        fault_injector: Callable[[str], Any] | None = None,
    ) -> PublicationReport:
        version = str(new_version).strip()
        if not version or "/" in version or version.startswith("."):
            raise ValueError("Unsafe bundle version")
        with file_lock(self.lock_path):
            # Validate the append-only audit boundary before doing any work.
            # A corrupt ledger must fail closed before a candidate Bundle is
            # frozen, built, or made visible on disk.
            self._ledger_events()
            pointer = self._read_current()
            parent = self._parent_bundle(pointer)
            parent.assert_unchanged()
            if (
                expected_parent_manifest_sha256 is not None
                and parent.manifest_sha256 != expected_parent_manifest_sha256
            ):
                raise PublicationConflictError(
                    "CURRENT changed since this publication was prepared"
                )
            target = self.bundles_dir / version
            if target.exists():
                raise FileExistsError(f"Bundle version already exists: {target}")
            attempt_id = uuid.uuid4().hex
            overlay_input = self.bundle_root / f".inputs-{version}-{attempt_id}"
            staging = self.bundle_root / f".staging-{version}-{attempt_id}"
            failed = self.bundle_root / f".failed-{version}-{attempt_id}"
            overlay.freeze_to(overlay_input)
            self._fault(fault_injector, "after_overlay_freeze")
            staging.mkdir()
            try:
                reports = dict(pipeline(parent, overlay_input, staging))
                self._validate_pipeline_reports(reports)
                self._fault(fault_injector, "after_pipeline")
                manifest = verify_bundle_directory(
                    staging, verify_artifacts=True, allow_staging=True
                )
                if str(manifest.get("bundle_version")) != version:
                    raise PublicationValidationError(
                        "Candidate bundle version does not match requested version"
                    )
                if manifest.get("parent_bundle") != parent.bundle_id:
                    raise PublicationValidationError(
                        "Candidate bundle does not bind the current parent bundle"
                    )
                self._fault(fault_injector, "after_validation")
                os.replace(staging, target)
                _fsync_directory(self.bundles_dir)
                self._fault(fault_injector, "after_bundle_publish")
                parent.assert_unchanged()
                current = make_current_pointer(
                    bundle_path=str(target.relative_to(self.bundle_root)),
                    manifest=manifest,
                    parent_bundle=parent.bundle_id,
                )
                prepared = {
                    "publication_id": f"publication::{attempt_id}",
                    "parent_bundle_id": parent.bundle_id,
                    "parent_manifest_sha256": parent.manifest_sha256,
                    "bundle_id": manifest["bundle_id"],
                    "bundle_version": version,
                    "bundle_manifest_sha256": manifest["manifest_sha256"],
                    "overlay_manifest_sha256": _read_json(
                        overlay_input / "overlay_manifest.json"
                    )["manifest_sha256"],
                    "overlay_events_sha256": _read_json(
                        overlay_input / "overlay_manifest.json"
                    )["events_sha256"],
                    "bundle_path": str(target.relative_to(self.bundle_root)),
                    "pipeline_reports": reports,
                    "current_pointer_sha256": current["pointer_sha256"],
                }
                self._append_ledger("publication_prepared", prepared)
                self._fault(fault_injector, "before_current_swap")
                write_json_atomic(self.current_path, current)
                self._fault(fault_injector, "after_current_swap")
                report = PublicationReport(**prepared).finalize()
                # This event is audit metadata after the atomic commit. A
                # failure here must not roll back a valid CURRENT pointer.
                try:
                    self._append_ledger("publication_committed", report.as_dict())
                except Exception:
                    pass
                shutil.rmtree(overlay_input)
                return report
            except Exception:
                if staging.exists():
                    if failed.exists():
                        raise RuntimeError(f"Failed publication path collision: {failed}")
                    os.replace(staging, failed)
                    if overlay_input.exists():
                        os.replace(overlay_input, failed / "overlay_snapshot")
                raise


__all__ = [
    "PUBLICATION_EVENT_SCHEMA",
    "PUBLICATION_REPORT_SCHEMA",
    "PublicationConflictError",
    "PublicationReport",
    "PublicationValidationError",
    "REQUIRED_PIPELINE_REPORTS",
    "SleepTimePipeline",
    "SleepTimePublisher",
    "classify_writeback_events",
]
