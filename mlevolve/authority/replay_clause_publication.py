from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .domain_scope import SAME_DOMAIN, normalize_transfer_scope
from .memory_snapshot import sha256_file, sha256_json, write_json_atomic
from .models import SOPClauseV1


REPLAY_CLAUSE_PUBLICATION_SCHEMA = "replay_clause_publication_report_v1"
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")


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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = "".join(
        json.dumps(_jsonable(row), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
        for row in rows
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
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
    rows = {str(row[key]): copy.deepcopy(dict(row)) for row in existing}
    for value in additions:
        row = copy.deepcopy(dict(value))
        row_id = str(row[key])
        if row_id in rows:
            raise ValueError(f"Replay publication ID already exists: {row_id}")
        rows[row_id] = row
    return [rows[row_id] for row_id in sorted(rows)]


def _append_unique_rows(
    existing: Iterable[Mapping[str, Any]], additions: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    output = [copy.deepcopy(dict(row)) for row in existing]
    seen = {json.dumps(row, sort_keys=True, separators=(",", ":")) for row in output}
    for value in additions:
        row = copy.deepcopy(dict(value))
        key = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            output.append(row)
            seen.add(key)
    return output


def _hash_embedding(text: str, dims: int = 64) -> np.ndarray:
    vector = np.zeros(dims, dtype=np.float32)
    for token in _TOKEN_RE.findall(str(text).lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _coordinate(node_id: str, node_type: str) -> np.ndarray:
    radii = {"SOP": 0.52, "SOPClause": 0.62}
    digest = hashlib.sha256(str(node_id).encode()).digest()
    angle = 2 * math.pi * int.from_bytes(digest[:8], "big") / 2**64
    radius = radii[node_type]
    return np.asarray(
        [radius * math.cos(angle), radius * math.sin(angle)], dtype=np.float32
    )


@dataclass(frozen=True)
class ReplayClausePublication:
    clause: SOPClauseV1
    title: str
    source_clause_id: str
    registration_hash: str
    verification_report_hash: str

    def clause_row(self) -> dict[str, Any]:
        row = _jsonable(self.clause)
        row.update(
            {
                "admissible_target_domains": (
                    sorted(set(self.clause.source_domains))
                    if normalize_transfer_scope(self.clause.transfer_scope)
                    == SAME_DOMAIN
                    else []
                ),
                "publication_origin": "clean_replay_certification_v1",
                "predecessor_clause_id": self.source_clause_id,
                "replay_registration_hash": self.registration_hash,
                "replay_verification_report_hash": self.verification_report_hash,
            }
        )
        return row

    def container_row(self) -> dict[str, Any]:
        return {
            "sop_id": self.clause.sop_id,
            "title": str(self.title),
            "task_id": self.clause.source_task_ids[0],
            "clause_ids": [self.clause.clause_id],
            "source_run_ids": list(self.clause.source_run_ids),
            "source_task_ids": list(self.clause.source_task_ids),
            "source_task_families": list(self.clause.source_task_families),
            "source_domains": list(self.clause.source_domains),
            "transfer_scopes": [self.clause.transfer_scope],
            "domain_scope_complete": True,
            "publication_origin": "clean_replay_certification_v1",
        }


def publish_replay_clauses(
    candidate: str | Path,
    publications: Iterable[ReplayClausePublication],
    *,
    bundle_id: str,
    parent_bundle_id: str,
) -> dict[str, Any]:
    candidate = Path(candidate)
    publications = list(publications)
    if not publications:
        return {
            "schema": REPLAY_CLAUSE_PUBLICATION_SCHEMA,
            "status": "passed",
            "clause_count": 0,
            "clause_ids": [],
            "old_semantic_rows_mutated": False,
            "report_hash": "",
        }

    clause_rows = [value.clause_row() for value in publications]
    container_rows = [value.container_row() for value in publications]
    clause_ids = [str(row["clause_id"]) for row in clause_rows]
    sop_ids = [str(row["sop_id"]) for row in container_rows]
    if len(clause_ids) != len(set(clause_ids)) or len(sop_ids) != len(set(sop_ids)):
        raise ValueError("Replay clause/SOP IDs must be unique")

    clauses_path = candidate / "sop" / "clauses.jsonl"
    containers_path = candidate / "sop" / "containers.json"
    sop_graph_path = candidate / "sop" / "graph.json"
    runforest_graph_path = candidate / "runforest" / "graph.json"
    metadata_path = candidate / "visibility" / "clause_metadata.jsonl"
    masks_path = (
        candidate
        / "visibility"
        / "precompiled_masks"
        / "declared_scope_masks.json"
    )
    parent_clauses = _read_jsonl(clauses_path)
    parent_containers_payload = _read_json(containers_path)
    parent_containers = list(parent_containers_payload.get("containers") or [])
    parent_sop_graph = _read_json(sop_graph_path)
    parent_runforest = _read_json(runforest_graph_path)
    parent_nodes = {
        str(node["id"]): copy.deepcopy(node)
        for node in parent_runforest.get("nodes") or []
    }

    _write_jsonl_atomic(
        clauses_path,
        _merge_rows(parent_clauses, clause_rows, key="clause_id"),
    )
    write_json_atomic(
        containers_path,
        {
            **{
                key: copy.deepcopy(value)
                for key, value in parent_containers_payload.items()
                if key != "containers"
            },
            "containers": _merge_rows(
                parent_containers, container_rows, key="sop_id"
            ),
        },
    )

    contains_edges = [
        {"src": row["sop_id"], "dst": row["clause_id"], "kind": "contains_clause"}
        for row in clause_rows
    ]
    derived_edges = [
        {
            "src": row["source_artifact_refs"][0],
            "dst": row["clause_id"],
            "kind": "derived_into_clause",
        }
        for row in clause_rows
    ]
    write_json_atomic(
        sop_graph_path,
        {
            **{
                key: copy.deepcopy(value)
                for key, value in parent_sop_graph.items()
                if key not in {"containers", "clauses", "edges"}
            },
            "containers": sorted(
                set(map(str, parent_sop_graph.get("containers") or []))
                | set(sop_ids)
            ),
            "clauses": sorted(
                set(map(str, parent_sop_graph.get("clauses") or []))
                | set(clause_ids)
            ),
            "edges": _append_unique_rows(
                parent_sop_graph.get("edges") or [],
                [*contains_edges, *derived_edges],
            ),
        },
    )

    sop_nodes = [
        {
            **row,
            "id": row["sop_id"],
            "type": "SOP",
            "task": row["task_id"],
            "task_families": list(row["source_domains"]),
        }
        for row in container_rows
    ]
    clause_nodes = [
        {**row, "id": row["clause_id"], "type": "SOPClause"}
        for row in clause_rows
    ]
    if (set(parent_nodes) & set(sop_ids)) or (set(parent_nodes) & set(clause_ids)):
        raise ValueError("Replay clause publication collides with RunForest nodes")
    runforest = copy.deepcopy(parent_runforest)
    runforest["nodes"] = [
        *list(runforest.get("nodes") or []),
        *sop_nodes,
        *clause_nodes,
    ]
    runforest["edges"] = _append_unique_rows(
        runforest.get("edges") or [], [*contains_edges, *derived_edges]
    )
    meta = dict(runforest.get("meta") or {})
    meta.update(
        {
            "bundle_id": bundle_id,
            "parent_bundle_id": parent_bundle_id,
            "certification_level": "certified",
            "certified_replay_clause_ids": sorted(clause_ids),
        }
    )
    runforest["meta"] = meta
    write_json_atomic(runforest_graph_path, runforest)

    visibility_rows = [
        {
            key: copy.deepcopy(row.get(key))
            for key in (
                "clause_id",
                "sop_id",
                "claim_refs",
                "source_artifact_refs",
                "source_transition_refs",
                "source_run_ids",
                "source_task_ids",
                "source_task_families",
                "source_domains",
                "transfer_scope",
                "admissible_target_domains",
                "protocol_scope",
                "task_scope",
                "permitted_operations",
                "permitted_generation_stages",
                "permitted_governance_stages",
                "publication_class",
                "legacy_status",
                "receipt_refs",
            )
        }
        for row in clause_rows
    ]
    _write_jsonl_atomic(
        metadata_path,
        _merge_rows(_read_jsonl(metadata_path), visibility_rows, key="clause_id"),
    )
    masks = _read_json(masks_path)
    mask_values = {
        str(key): list(map(str, values))
        for key, values in (masks.get("masks") or {}).items()
    }
    for row in clause_rows:
        for protocol in row.get("protocol_scope") or ["unscoped"]:
            for operation in row.get("permitted_operations") or []:
                for generation_stage in row.get("permitted_generation_stages") or []:
                    for governance_stage in row.get("permitted_governance_stages") or []:
                        key = "|".join(
                            [
                                str(protocol),
                                str(operation),
                                str(generation_stage),
                                str(governance_stage),
                            ]
                        )
                        mask_values.setdefault(key, []).append(row["clause_id"])
    write_json_atomic(
        masks_path,
        {
            **{key: value for key, value in masks.items() if key != "masks"},
            "masks": {
                key: sorted(set(values))
                for key, values in sorted(mask_values.items())
            },
        },
    )

    index_path = candidate / "runforest" / "index.npz"
    clause_index_path = candidate / "runforest" / "clause_index.npz"
    index = np.load(index_path, allow_pickle=True)
    new_nodes = [*sop_nodes, *clause_nodes]
    new_node_ids = np.asarray([row["id"] for row in new_nodes], dtype=object)
    new_node_types = np.asarray([row["type"] for row in new_nodes], dtype=object)
    new_coordinates = np.stack(
        [_coordinate(row["id"], row["type"]) for row in new_nodes]
    )
    new_euclidean = np.stack(
        [
            _hash_embedding(str(row.get("text") or row.get("title") or ""))
            for row in new_nodes
        ]
    )
    np.savez_compressed(
        index_path,
        node_ids=np.concatenate([index["node_ids"], new_node_ids]),
        node_types=np.concatenate([index["node_types"], new_node_types]),
        poincare=np.concatenate([index["poincare"], new_coordinates]),
        flat_twin=np.concatenate([index["flat_twin"], new_coordinates.copy()]),
        euclidean=np.concatenate([index["euclidean"], new_euclidean]),
    )
    clause_index = np.load(clause_index_path, allow_pickle=True)
    np.savez_compressed(
        clause_index_path,
        clause_ids=np.concatenate(
            [clause_index["clause_ids"], np.asarray(clause_ids, dtype=object)]
        ),
        embeddings=np.concatenate(
            [
                clause_index["embeddings"],
                np.stack(
                    [
                        _hash_embedding(str(row.get("retrieval_text") or ""))
                        for row in clause_rows
                    ]
                ),
            ]
        ),
    )

    runforest_report_path = candidate / "runforest" / "build_report.json"
    runforest_report = _read_json(runforest_report_path)
    node_type_counts = dict(runforest_report.get("node_type_counts") or {})
    node_type_counts["SOP"] = int(node_type_counts.get("SOP", 0)) + len(sop_nodes)
    node_type_counts["SOPClause"] = int(
        node_type_counts.get("SOPClause", 0)
    ) + len(clause_nodes)
    edge_kind_counts = dict(runforest_report.get("edge_kind_counts") or {})
    edge_kind_counts["contains_clause"] = int(
        edge_kind_counts.get("contains_clause", 0)
    ) + len(contains_edges)
    edge_kind_counts["derived_into_clause"] = int(
        edge_kind_counts.get("derived_into_clause", 0)
    ) + len(derived_edges)
    runforest_report.update(
        {
            "bundle_id": bundle_id,
            "certification_level": "certified",
            "node_count": len(runforest["nodes"]),
            "edge_count": len(runforest["edges"]),
            "node_type_counts": dict(sorted(node_type_counts.items())),
            "edge_kind_counts": dict(sorted(edge_kind_counts.items())),
            "included_clause_count": len(parent_clauses) + len(clause_rows),
            "certified_replay_clause_ids": sorted(clause_ids),
            "graph_sha256": sha256_file(runforest_graph_path),
            "index_sha256": sha256_file(index_path),
            "clause_index_sha256": sha256_file(clause_index_path),
            "declared_scope_masks_sha256": sha256_file(masks_path),
        }
    )
    write_json_atomic(runforest_report_path, runforest_report)

    build_report_path = candidate / "reports" / "build_report.json"
    build_report = _read_json(build_report_path)
    build_report.update(
        {
            "bundle_id": bundle_id,
            "certification_level": "certified",
            "parent_bundle": parent_bundle_id,
            "clause_count": len(parent_clauses) + len(clause_rows),
            "container_count": len(parent_containers) + len(container_rows),
            "certified_replay_clause_ids": sorted(clause_ids),
            "blanket_clause_upgrade": False,
        }
    )
    write_json_atomic(build_report_path, build_report)

    current_clauses = {
        str(row["clause_id"]): row for row in _read_jsonl(clauses_path)
    }
    current_containers = {
        str(row["sop_id"]): row
        for row in _read_json(containers_path).get("containers") or []
    }
    old_semantic_rows_mutated = any(
        current_clauses.get(str(row["clause_id"])) != row for row in parent_clauses
    ) or any(
        current_containers.get(str(row["sop_id"])) != row
        for row in parent_containers
    ) or any(
        next(
            (
                node
                for node in runforest.get("nodes") or []
                if str(node.get("id")) == node_id
            ),
            None,
        )
        != node
        for node_id, node in parent_nodes.items()
    )
    report = {
        "schema": REPLAY_CLAUSE_PUBLICATION_SCHEMA,
        "status": "passed" if not old_semantic_rows_mutated else "failed",
        "parent_bundle_id": parent_bundle_id,
        "bundle_id": bundle_id,
        "clause_count": len(clause_rows),
        "clause_ids": sorted(clause_ids),
        "sop_ids": sorted(sop_ids),
        "source_clause_ids": sorted(value.source_clause_id for value in publications),
        "old_semantic_rows_mutated": old_semantic_rows_mutated,
        "runforest_graph_sha256": sha256_file(runforest_graph_path),
        "runforest_index_sha256": sha256_file(index_path),
        "clause_index_sha256": sha256_file(clause_index_path),
        "visibility_masks_sha256": sha256_file(masks_path),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


__all__ = [
    "REPLAY_CLAUSE_PUBLICATION_SCHEMA",
    "ReplayClausePublication",
    "publish_replay_clauses",
]
