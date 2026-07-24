from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from authority.bundle_publisher import SleepTimePipeline, verify_bundle_directory
from authority.memory_snapshot import sha256_file, sha256_json, write_json_atomic
from authority.models import SOPClauseV1
from authority.protocol_registry import ProtocolRegistry
from authority.replay_clause_publication import (
    publish_replay_clauses,
)
from authority.writeback_distillation import materialize_positive_writeback
from bind_positive_writeback import _proposals
from bind_sop_clauses import validate_positive_clause_payload


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _merge_rows(
    existing: list[dict[str, Any]],
    additions: list[Mapping[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    rows = {str(row[key]): copy.deepcopy(row) for row in existing}
    for value in additions:
        row = copy.deepcopy(dict(value))
        row_key = str(row[key])
        if row_key in rows and rows[row_key] != row:
            raise ValueError(f"Positive writeback would mutate immutable {key}: {row_key}")
        rows[row_key] = row
    return [rows[row_key] for row_key in sorted(rows)]


@dataclass(frozen=True)
class _PositiveClausePublication:
    clause: SOPClauseV1
    title: str
    source_claim_id: str
    materialization_hash: str

    @property
    def source_clause_id(self) -> str:
        return self.source_claim_id

    @property
    def registration_hash(self) -> str:
        return self.materialization_hash

    @property
    def verification_report_hash(self) -> str:
        return self.materialization_hash

    def clause_row(self) -> dict[str, Any]:
        row = dataclasses.asdict(self.clause)
        row.update(
            {
                "publication_origin": "typed_writeback_distillation_v1",
                "positive_writeback_materialization_hash": self.materialization_hash,
            }
        )
        return row

    def container_row(self) -> dict[str, Any]:
        return {
            "sop_id": self.clause.sop_id,
            "title": self.title,
            "task_id": self.clause.source_task_ids[0]
            if self.clause.source_task_ids
            else "",
            "clause_ids": [self.clause.clause_id],
            "source_run_ids": list(self.clause.source_run_ids),
            "source_task_ids": list(self.clause.source_task_ids),
            "source_task_families": list(self.clause.source_task_families),
            "source_domains": list(self.clause.source_domains),
            "transfer_scopes": [self.clause.transfer_scope],
            "domain_scope_complete": bool(
                self.clause.source_task_ids and self.clause.source_domains
            ),
            "publication_origin": "typed_writeback_distillation_v1",
        }


def _material_rows(material) -> dict[str, list[dict[str, Any]]]:
    snapshot = material.authority_snapshot
    return {
        "claims": [
            row for _key, row in sorted((snapshot.get("claims") or {}).items())
        ],
        "receipts": [
            row
            for _key, row in sorted((snapshot.get("receipts") or {}).items())
        ],
        "paths": [
            row for _key, row in sorted((snapshot.get("paths") or {}).items())
        ],
        "decisions": [
            row
            for _key, row in sorted((snapshot.get("decisions") or {}).items())
        ],
    }


def make_positive_writeback_pipeline(
    *,
    new_version: str,
    proposals_path: str | Path,
    protocol_registry_path: str | Path,
    policy_version: str = "authority_v1",
    collector_version: str = "1",
) -> SleepTimePipeline:
    """Create the production sleep-time pipeline for typed writeback material."""

    proposals_path = Path(proposals_path).resolve()
    registry_path = Path(protocol_registry_path).resolve()
    proposals = _proposals(proposals_path)
    registry = ProtocolRegistry(registry_path)

    def audit(context):
        inventory = context["writeback_inventory"]
        plan = context["positive_writeback_plan"]
        return {
            "status": "passed",
            "inventory_hash": inventory["inventory_hash"],
            "result_fact_count": inventory["result_fact_count"],
            "adoption_edge_count": inventory["adoption_edge_count"],
            "causal_edge_count": inventory["causal_edge_count"],
            "plan_hash": plan["plan_hash"],
        }

    def claim_decomposition(context):
        plan = context["positive_writeback_plan"]
        return {
            "status": "passed",
            "plan_hash": plan["plan_hash"],
            "source_claim_refs": sorted(
                str(item["source_claim_ref"]) for item in plan["items"]
            ),
            "quarantined": list(plan.get("quarantined") or []),
        }

    def distillation(context):
        plan = context["positive_writeback_plan"]
        expected = {str(item["candidate_id"]) for item in plan["items"]}
        if set(proposals) != expected:
            raise ValueError(
                "Positive writeback proposals must exactly cover the typed plan"
            )
        material = materialize_positive_writeback(
            plan,
            proposals,
            registry=registry,
            policy_version=policy_version,
            collector_version=collector_version,
        )
        context["positive_writeback_material"] = material
        report = dict(material.report)
        report["plan_hash"] = plan["plan_hash"]
        report["consumed_event_ids"] = list(plan["consumed_event_ids"])
        report["status"] = "passed"
        return report

    def build(context):
        parent = context["parent_bundle_path"]
        candidate = context["candidate_dir"]
        material = context["positive_writeback_material"]
        shutil.copytree(parent, candidate, dirs_exist_ok=True)
        manifest_path = candidate / "manifest.json"
        sums_path = candidate / "SHA256SUMS"
        if sums_path.exists():
            sums_path.unlink()
        manifest = _read_json(manifest_path)
        manifest_path.unlink()
        material_root = candidate / "positive_writeback"
        material_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            material_root / "plan.json", material.plan
        )
        write_json_atomic(
            material_root / "materialization_report.json", material.report
        )
        write_json_atomic(
            material_root / "authority_snapshot.json",
            material.authority_snapshot,
        )
        _write_jsonl(material_root / "derivations.jsonl", list(material.derivations))
        write_json_atomic(
            material_root / "inventory.json",
            context["writeback_inventory"],
        )

        authority_root = candidate / "authority"
        authority_root.mkdir(parents=True, exist_ok=True)
        rows = _material_rows(material)
        for name, values in rows.items():
            path = authority_root / f"{name}.jsonl"
            _write_jsonl(path, _merge_rows(_read_jsonl(path), values, "claim_id" if name == "claims" else "receipt_id" if name == "receipts" else "path_id" if name == "paths" else "decision_id"))
        derivations_path = authority_root / "derivations.jsonl"
        _write_jsonl(
            derivations_path,
            _merge_rows(
                _read_jsonl(derivations_path),
                list(material.derivations),
                "derivation_id",
            ),
        )

        publications = []
        for clause_row in material.clauses:
            validate_positive_clause_payload(clause_row)
            fields = {
                key: value
                for key, value in clause_row.items()
                if key in {field.name for field in dataclasses.fields(SOPClauseV1)}
            }
            publications.append(
                _PositiveClausePublication(
                    clause=SOPClauseV1(**fields),
                    title=str(fields["text"]),
                    source_claim_id=str(
                        fields["contract_spec"]["source_claim_id"]
                    ),
                    materialization_hash=material.report[
                        "materialization_hash"
                    ],
                )
            )
        if publications:
            publish_replay_clauses(
                candidate,
                publications,
                bundle_id=f"{manifest['bundle_id']}::{new_version}",
                parent_bundle_id=str(manifest["bundle_id"]),
            )
            # The shared graph/index writer is intentionally reused, but its
            # metadata must not call typed writeback a Clean Replay.
            for relative in (
                "runforest/graph.json",
                "runforest/build_report.json",
                "reports/build_report.json",
            ):
                path = candidate / relative
                if not path.is_file():
                    continue
                payload = _read_json(path)
                for key in ("certified_replay_clause_ids",):
                    payload.pop(key, None)
                payload["positive_writeback_clause_ids"] = sorted(
                    clause.clause_id for clause in (publication.clause for publication in publications)
                )
                payload["positive_writeback_materialization_hash"] = material.report[
                    "materialization_hash"
                ]
                write_json_atomic(path, payload)

        manifest["bundle_id"] = f"{manifest['bundle_id']}::{new_version}"
        manifest["bundle_version"] = str(new_version)
        manifest["parent_bundle"] = str(context["parent_bundle"].bundle_id)
        manifest["positive_writeback_materialization_hash"] = material.report[
            "materialization_hash"
        ]
        manifest["positive_writeback_counts"] = {
            "result": material.report["positive_result_count"],
            "adopted": material.report["positive_adopted_count"],
        }
        artifact_hashes = {
            path.relative_to(candidate).as_posix(): sha256_file(path)
            for path in sorted(candidate.rglob("*"))
            if path.is_file()
        }
        manifest["artifact_hashes"] = artifact_hashes
        manifest["graph_hashes"] = {
            "runforest": artifact_hashes.get("runforest/graph.json", "")
        }
        manifest["index_hashes"] = {
            "runforest": artifact_hashes.get("runforest/index.npz", ""),
            "clauses": artifact_hashes.get("runforest/clause_index.npz", ""),
        }
        manifest["lineage_hash"] = sha256_json(
            {
                "claims": artifact_hashes.get("authority/claims.jsonl", ""),
                "derivations": artifact_hashes.get(
                    "authority/derivations.jsonl", ""
                ),
                "clauses": artifact_hashes.get("sop/clauses.jsonl", ""),
                "writeback": material.report["materialization_hash"],
            }
        )
        manifest["manifest_sha256"] = sha256_json(
            {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
        )
        write_json_atomic(manifest_path, manifest)

    def derivation_validation(context):
        material = context["positive_writeback_material"]
        valid = all(
            row.get("scope_widened") is False for row in material.derivations
        )
        return {
            "status": "passed" if valid else "failed",
            "materialization_hash": material.report["materialization_hash"],
            "derivation_count": len(material.derivations),
        }

    def visibility_validation(context):
        material = context["positive_writeback_material"]
        valid = all(
            row.get("publication_class")
            in {"positive_result", "positive_adopted"}
            and tuple(row.get("claim_types") or []) == ("method_hypothesis",)
            for row in material.clauses
        )
        return {
            "status": "passed" if valid else "failed",
            "positive_clause_count": len(material.clauses),
            "cross_domain_inference": False,
        }

    def bundle_validation(context):
        manifest = verify_bundle_directory(
            context["candidate_dir"], verify_artifacts=True, allow_staging=True
        )
        return {
            "status": "passed",
            "valid": True,
            "manifest_sha256": manifest["manifest_sha256"],
            "positive_writeback_materialization_hash": manifest.get(
                "positive_writeback_materialization_hash"
            ),
        }

    return SleepTimePipeline(
        audit=audit,
        claim_decomposition=claim_decomposition,
        distillation=distillation,
        build_candidate=build,
        derivation_validation=derivation_validation,
        visibility_validation=visibility_validation,
        bundle_validation=bundle_validation,
    )


__all__ = ["make_positive_writeback_pipeline"]
