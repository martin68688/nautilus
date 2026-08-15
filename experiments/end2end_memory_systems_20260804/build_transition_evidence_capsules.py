#!/usr/bin/env python3
"""Materialize strict-clean Debug/Improve parent-child programs for Resolver use.

The first-stage Search/Grep/Judge path intentionally sees only compact graph
metadata.  This builder creates a separate, content-addressed artifact that a
deterministic Evidence Resolver may open *after* the Judge has selected a
candidate.  Full source never participates in retrieval or ranking.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


SCHEMA = "mlevolve_transition_evidence_capsules_v2"
TASK_ID = "leaf-classification"
EVIDENCE_CLASSES = {
    "strict_debug_observed",
    "official_observed",
    "strict_internal_observed",
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _payload_hash(value: Mapping[str, Any], excluded: str) -> str:
    payload = {key: item for key, item in value.items() if key != excluded}
    return _sha256_bytes(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _node_metric(node: Mapping[str, Any]) -> float | None:
    metric = node.get("metric")
    return _finite(metric.get("value") if isinstance(metric, Mapping) else metric)


def _strict_clean(node: Mapping[str, Any], *, task_id: str) -> bool:
    audit = node.get("leakage_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    return bool(
        node.get("type") == "RunNode"
        and str(node.get("task") or "") == task_id
        and node.get("is_buggy") is False
        and node.get("is_valid") is True
        and _finite(node.get("metric")) is not None
        and node.get("audit_status") == "clean"
        and node.get("metric_disposition") == "rank_eligible"
        and node.get("memory_disposition") == "positive_eligible"
        and node.get("paper_grade_eligible") is True
        and audit.get("status") == "clean"
        and audit.get("rank_eligible") is True
        and audit.get("memory_disposition") == "positive_eligible"
        and audit.get("paper_grade_eligible") is True
        and node.get("quarantined") is not True
        and node.get("protocol_biased") is not True
    )


def _official_observed(node: Mapping[str, Any]) -> bool:
    audit = node.get("leakage_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    provenance = str(node.get("metric_provenance") or "").lower()
    return bool(
        audit.get("official_score_review") == "pass"
        or audit.get("official_ledger_sha256")
        or "official_kaggle" in provenance
        or "sealed_fixed_holdout_terminal" in provenance
    )


def _attempt_identity(relative_path: str) -> tuple[str, str]:
    parts = PurePosixPath(relative_path).parts
    try:
        runs_index = parts.index("runs")
    except ValueError as exc:
        raise ValueError(f"Journal path is outside a run: {relative_path}") from exc
    if len(parts) <= runs_index + 2:
        raise ValueError(f"Journal path has no logical run/attempt: {relative_path}")
    return str(parts[runs_index + 1]), str(parts[runs_index + 2])


def _journal_entries(
    inventory: Mapping[str, Any], source_root: Path
) -> list[dict[str, Any]]:
    entries = []
    for raw in inventory.get("files") or []:
        if not isinstance(raw, Mapping):
            continue
        relative = str(raw.get("path") or "")
        if PurePosixPath(relative).name != "journal.json":
            continue
        logical_run_id, attempt = _attempt_identity(relative)
        path = (source_root / relative).resolve(strict=True)
        observed_size = path.stat().st_size
        expected_size = int(raw.get("size_bytes") or -1)
        if observed_size != expected_size:
            raise ValueError(
                f"Frozen journal size mismatch: {relative}: "
                f"expected={expected_size} observed={observed_size}"
            )
        observed_sha = _sha256_file(path)
        expected_sha = str(raw.get("sha256") or "")
        if observed_sha != expected_sha:
            raise ValueError(
                f"Frozen journal SHA-256 mismatch: {relative}: "
                f"expected={expected_sha} observed={observed_sha}"
            )
        entries.append(
            {
                "relative_path": relative,
                "path": path,
                "sha256": observed_sha,
                "logical_run_id": logical_run_id,
                "attempt": attempt,
                "base_run_id": f"{logical_run_id}::{attempt}",
            }
        )
    if not entries:
        raise ValueError("Source inventory contains no journal.json entries")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["base_run_id"]].append(entry)
    for base_run_id, rows in grouped.items():
        unique_by_sha = {str(row["sha256"]): row for row in rows}
        unique_rows = sorted(
            unique_by_sha.values(), key=lambda row: str(row["relative_path"])
        )
        canonical = unique_rows[-1]
        for row in unique_rows:
            row["staged_run_id"] = (
                base_run_id
                if row is canonical
                else f"{base_run_id}::source-{str(row['sha256'])[:8]}"
            )
        duplicate_rows = [row for row in rows if row not in unique_rows]
        for row in duplicate_rows:
            source = unique_by_sha[str(row["sha256"])]
            row["staged_run_id"] = source["staged_run_id"]
    return sorted(entries, key=lambda row: str(row["relative_path"]))


def _pair_key(kind: str, before_sha: str, after_sha: str) -> str:
    return _sha256_text(f"{kind}\0{before_sha}\0{after_sha}")


def _representative_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    class_rank = {
        "official_observed": 0,
        "strict_internal_observed": 1,
        "strict_debug_observed": 2,
    }
    return (
        class_rank[str(row["evidence_class"])],
        -float(row.get("metric_improvement") or 0.0),
        str(row["transition_id"]),
    )


def _atomic_alias_specs(
    graph_nodes: Mapping[str, Mapping[str, Any]], *, task_id: str
) -> list[dict[str, Any]]:
    """Validate formal atomic Debug claims and expose their exact source IDs."""

    output = []
    for atomic_id, atomic in sorted(graph_nodes.items()):
        if not atomic_id.startswith("atomic-transition::"):
            continue
        claim = atomic.get("atomic_repair_claim")
        claim = claim if isinstance(claim, Mapping) else {}
        verification = claim.get("verification")
        verification = verification if isinstance(verification, Mapping) else {}
        taint = claim.get("taint")
        taint = taint if isinstance(taint, Mapping) else {}
        source_program = taint.get("source_program")
        source_program = (
            source_program if isinstance(source_program, Mapping) else {}
        )
        visibility = claim.get("operation_visibility")
        visibility = visibility if isinstance(visibility, Mapping) else {}
        source_transition_id = str(claim.get("source_transition_id") or "")
        source = graph_nodes.get(source_transition_id)
        if not isinstance(source, Mapping) or source.get("type") != "Transition":
            raise ValueError(f"Atomic claim source Transition is missing: {atomic_id}")
        source_parent_id = str(source.get("parent_node_id") or "")
        source_child_id = str(source.get("child_node_id") or "")
        parent = graph_nodes.get(source_parent_id)
        child = graph_nodes.get(source_child_id)
        allowed = {str(value) for value in visibility.get("allowed_operations") or []}
        forbidden = {
            str(value) for value in visibility.get("forbidden_operations") or []
        }
        required_structural_verification = {
            "claim_scope_independently_audited",
            "observed_child_execution_success",
            "observed_parent_failure",
            "repair_action_bound_to_transition",
        }
        if (
            atomic.get("type") != "Transition"
            or str(atomic.get("task") or "") != task_id
            or str(atomic.get("outcome") or "") != "debug_fixed"
            or atomic.get("quarantined") is True
            or atomic.get("protocol_biased") is True
            or str(claim.get("schema") or "") != "mlevolve_atomic_memory_claim_v1"
            or str(claim.get("claim_status") or "") != "authorized_debug_only"
            or str(claim.get("task_id") or "") != task_id
            or str(claim.get("outcome") or "") != "debug_fixed"
            or not {"debug_hypothesis", "debug_repair"}.issubset(allowed)
            or not {"improve_method_selection", "metric_ranking"}.issubset(
                forbidden
            )
            or any(
                verification.get(key) is not True
                for key in required_structural_verification
            )
            or not isinstance(parent, Mapping)
            or not isinstance(child, Mapping)
            or str(claim.get("source_parent_node_id") or "") != source_parent_id
            or str(claim.get("source_child_node_id") or "") != source_child_id
            or str(atomic.get("parent_node_id") or "") != source_parent_id
            or str(atomic.get("child_node_id") or "") != source_child_id
            or str(source.get("outcome") or "") != "debug_fixed"
            or str(verification.get("before_code_sha256") or "")
            != str(parent.get("code_sha256") or "")
            or str(verification.get("after_code_sha256") or "")
            != str(child.get("code_sha256") or "")
        ):
            raise ValueError(f"Atomic claim authorization is malformed: {atomic_id}")
        if (
            str(taint.get("claim") or "") != "clean"
            or str(taint.get("code") or "") != "clean"
            or str(source_program.get("status") or "") != "clean"
            or source_program.get("rank_eligible") is not True
            or str(source_program.get("memory_disposition") or "")
            != "positive_eligible"
            or verification.get("full_program_clean") is not True
        ):
            continue
        output.append(
            {
                "atomic_transition_id": atomic_id,
                "claim_id": str(claim.get("id") or ""),
                "claim_status": str(claim.get("claim_status") or ""),
                "source_transition_id": source_transition_id,
                "source_parent_node_id": source_parent_id,
                "source_child_node_id": source_child_id,
                "before_code_sha256": str(
                    verification.get("before_code_sha256") or ""
                ),
                "after_code_sha256": str(
                    verification.get("after_code_sha256") or ""
                ),
                "outcome": "debug_fixed",
            }
        )
    return output


def _candidate_alias_rows(
    specs: Iterable[Mapping[str, Any]], materialized_transition_ids: set[str]
) -> list[dict[str, Any]]:
    output = []
    for spec in specs:
        atomic_id = str(spec["atomic_transition_id"])
        repair_id = atomic_id.replace("atomic-transition::", "repair-claim::", 1)
        for alias_kind, candidate_id in (
            ("atomic_transition", atomic_id),
            ("repair_claim", repair_id),
        ):
            output.append(
                {
                    "candidate_id": candidate_id,
                    "alias_kind": alias_kind,
                    "atomic_transition_id": atomic_id,
                    "claim_id": str(spec["claim_id"]),
                    "claim_status": str(spec["claim_status"]),
                    "source_transition_id": str(spec["source_transition_id"]),
                    "outcome": str(spec["outcome"]),
                    "before_code_sha256": str(spec["before_code_sha256"]),
                    "after_code_sha256": str(spec["after_code_sha256"]),
                    "materialized": (
                        str(spec["source_transition_id"])
                        in materialized_transition_ids
                    ),
                }
            )
    return sorted(output, key=lambda row: str(row["candidate_id"]))


def _pair_rows(
    transition_bindings: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in transition_bindings:
        row = dict(raw)
        by_pair[str(row["pair_key"])].append(row)
    pairs = []
    for pair_key, aliases in sorted(by_pair.items()):
        aliases = sorted(aliases, key=_representative_key)
        representative = aliases[0]
        pairs.append(
            {
                "pair_key": pair_key,
                "outcome": representative["outcome"],
                "evidence_class": representative["evidence_class"],
                "representative_transition_id": representative["transition_id"],
                "alias_transition_ids": [row["transition_id"] for row in aliases],
                "before_code_sha256": representative["before_code_sha256"],
                "after_code_sha256": representative["after_code_sha256"],
            }
        )
    return pairs


def build(
    *,
    graph_path: Path,
    source_inventory_path: Path,
    source_root: Path,
    task_id: str = TASK_ID,
    expected_debug_unique_pairs: int | None = None,
    expected_improve_unique_pairs: int | None = None,
) -> dict[str, Any]:
    graph_path = graph_path.resolve(strict=True)
    source_inventory_path = source_inventory_path.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    graph = _read_object(graph_path)
    inventory = _read_object(source_inventory_path)
    graph_nodes = {
        str(row.get("id") or ""): row
        for row in graph.get("nodes") or []
        if isinstance(row, Mapping) and row.get("id")
    }
    atomic_alias_specs = _atomic_alias_specs(graph_nodes, task_id=task_id)
    atomic_aliases_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in atomic_alias_specs:
        atomic_aliases_by_source[str(spec["source_transition_id"])].append(spec)
    journals = _journal_entries(inventory, source_root)

    code_blobs: dict[str, dict[str, Any]] = {}
    node_bindings: dict[str, dict[str, Any]] = {}
    transition_bindings: list[dict[str, Any]] = []
    observed_parent_child_pairs = 0
    graph_matched_parent_child_pairs = 0

    def bind_node(
        node_id: str,
        code: str,
        *,
        source_journal: str,
        source_journal_sha256: str,
        source_raw_node_id: str,
    ) -> str:
        code_sha = _sha256_text(code)
        graph_node = graph_nodes.get(node_id)
        if not isinstance(graph_node, Mapping) or graph_node.get("type") != "RunNode":
            raise ValueError(f"Resolver endpoint is missing from RunForest: {node_id}")
        graph_sha = str(graph_node.get("code_sha256") or "")
        if graph_sha != code_sha:
            raise ValueError(
                f"Resolver endpoint code identity mismatch: {node_id}: "
                f"graph={graph_sha} journal={code_sha}"
            )
        existing = node_bindings.get(node_id)
        current = {
            "node_id": node_id,
            "code_sha256": code_sha,
            "source_journal": source_journal,
            "source_journal_sha256": source_journal_sha256,
            "source_raw_node_id": source_raw_node_id,
        }
        if existing is not None and existing["code_sha256"] != code_sha:
            raise ValueError(f"Conflicting resolver endpoint code: {node_id}")
        node_bindings.setdefault(node_id, current)
        code_blobs.setdefault(
            code_sha,
            {
                "code_sha256": code_sha,
                "code": code,
            },
        )
        return code_sha

    for journal in journals:
        payload = _read_object(Path(journal["path"]))
        raw_nodes = {
            str(row.get("id") or ""): row
            for row in payload.get("nodes") or []
            if isinstance(row, Mapping) and row.get("id")
        }
        for child_raw_id, parent_raw_id in sorted(
            (payload.get("node2parent") or {}).items()
        ):
            child_raw_id = str(child_raw_id)
            parent_raw_id = str(parent_raw_id)
            child_raw = raw_nodes.get(child_raw_id)
            parent_raw = raw_nodes.get(parent_raw_id)
            if child_raw is None or parent_raw is None:
                continue
            before_code = str(parent_raw.get("code") or "")
            after_code = str(child_raw.get("code") or "")
            if not before_code.strip() or not after_code.strip():
                continue
            observed_parent_child_pairs += 1
            run_id = str(journal["staged_run_id"])
            transition_id = (
                f"run::{run_id}::transition::"
                f"{parent_raw_id[:12]}::{child_raw_id[:12]}"
            )
            transition = graph_nodes.get(transition_id)
            if not isinstance(transition, Mapping) or transition.get("type") != "Transition":
                raise ValueError(
                    f"Frozen journal transition is absent from active graph: {transition_id}"
                )
            graph_matched_parent_child_pairs += 1
            outcome = str(transition.get("outcome") or "")
            if outcome not in {"debug_fixed", "metric_improved"}:
                continue
            child_id = str(transition.get("child_node_id") or "")
            parent_id = str(transition.get("parent_node_id") or "")
            child = graph_nodes.get(child_id, {})
            authorized_atomic = bool(atomic_aliases_by_source.get(transition_id))
            if not _strict_clean(child, task_id=task_id) and not authorized_atomic:
                continue
            evidence_class = (
                "strict_debug_observed"
                if outcome == "debug_fixed"
                else "official_observed"
                if _official_observed(child)
                else "strict_internal_observed"
            )
            before_sha = bind_node(
                parent_id,
                before_code,
                source_journal=str(journal["relative_path"]),
                source_journal_sha256=str(journal["sha256"]),
                source_raw_node_id=parent_raw_id,
            )
            after_sha = bind_node(
                child_id,
                after_code,
                source_journal=str(journal["relative_path"]),
                source_journal_sha256=str(journal["sha256"]),
                source_raw_node_id=child_raw_id,
            )
            transition_bindings.append(
                {
                    "transition_id": transition_id,
                    "task_id": task_id,
                    "outcome": outcome,
                    "evidence_class": evidence_class,
                    "pair_key": _pair_key(outcome, before_sha, after_sha),
                    "parent_node_id": parent_id,
                    "child_node_id": child_id,
                    "before_code_sha256": before_sha,
                    "after_code_sha256": after_sha,
                    "parent_metric": transition.get("parent_metric"),
                    "child_metric": transition.get("child_metric"),
                    "metric_improvement": transition.get("metric_improvement"),
                    "metric_provenance": child.get("metric_provenance"),
                    "stage_pair": transition.get("stage_pair"),
                    "source_journal": str(journal["relative_path"]),
                    "source_journal_sha256": str(journal["sha256"]),
                }
            )

    pairs = _pair_rows(transition_bindings)
    materialized_transition_ids = {
        str(row["transition_id"]) for row in transition_bindings
    }
    candidate_aliases = _candidate_alias_rows(
        atomic_alias_specs, materialized_transition_ids
    )

    debug_pairs = sum(row["outcome"] == "debug_fixed" for row in pairs)
    improve_pairs = sum(row["outcome"] == "metric_improved" for row in pairs)
    if (
        expected_debug_unique_pairs is not None
        and debug_pairs != expected_debug_unique_pairs
    ):
        raise ValueError(
            "Strict-clean Debug unique-pair coverage mismatch: "
            f"expected={expected_debug_unique_pairs} observed={debug_pairs}"
        )
    if (
        expected_improve_unique_pairs is not None
        and improve_pairs != expected_improve_unique_pairs
    ):
        raise ValueError(
            "Strict-clean Improve unique-pair coverage mismatch: "
            f"expected={expected_improve_unique_pairs} observed={improve_pairs}"
        )

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "task_id": task_id,
        "graph_file_sha256": _sha256_file(graph_path),
        "source_inventory_file_sha256": _sha256_file(source_inventory_path),
        "source_inventory_sha256": inventory.get("inventory_sha256"),
        "journal_count": len(journals),
        "journal_sha256s": sorted(str(row["sha256"]) for row in journals),
        "observed_parent_child_pair_count": observed_parent_child_pairs,
        "graph_matched_parent_child_pair_count": graph_matched_parent_child_pairs,
        "transition_count": len(transition_bindings),
        "unique_pair_count": len(pairs),
        "debug_transition_count": sum(
            row["outcome"] == "debug_fixed" for row in transition_bindings
        ),
        "debug_unique_pair_count": debug_pairs,
        "improve_transition_count": sum(
            row["outcome"] == "metric_improved" for row in transition_bindings
        ),
        "improve_unique_pair_count": improve_pairs,
        "candidate_alias_count": len(candidate_aliases),
        "materialized_candidate_alias_count": sum(
            bool(row["materialized"]) for row in candidate_aliases
        ),
        "node_binding_count": len(node_bindings),
        "unique_code_count": len(code_blobs),
        "code_bytes": sum(len(str(row["code"])) for row in code_blobs.values()),
        "evidence_class_counts": {
            evidence_class: sum(
                row["evidence_class"] == evidence_class
                for row in transition_bindings
            )
            for evidence_class in sorted(EVIDENCE_CLASSES)
        },
        "code_blobs": [code_blobs[key] for key in sorted(code_blobs)],
        "nodes": [node_bindings[key] for key in sorted(node_bindings)],
        "transitions": sorted(
            transition_bindings, key=lambda row: str(row["transition_id"])
        ),
        "pairs": pairs,
        "candidate_aliases": candidate_aliases,
    }
    result["capsule_sha256"] = _payload_hash(result, "capsule_sha256")
    return result


def augment_atomic_aliases(
    *,
    base_capsule_path: Path,
    graph_path: Path,
    atomic_journal_paths: Iterable[Path],
    task_id: str = TASK_ID,
) -> dict[str, Any]:
    """Add formal atomic aliases and recover their frozen source programs.

    A journal may contribute code only when the exact parent/child program
    hashes match a clean, independently audited formal atomic claim.  Journals
    outside the base capsule inventory are recorded as explicit extensions;
    they cannot introduce a new claim or a semantically matched substitute.
    """

    base_capsule_path = base_capsule_path.resolve(strict=True)
    graph_path = graph_path.resolve(strict=True)
    result = copy.deepcopy(_read_object(base_capsule_path))
    if result.get("schema") != SCHEMA:
        raise ValueError("Unsupported base transition evidence capsule schema")
    if str(result.get("capsule_sha256") or "") != _payload_hash(
        result, "capsule_sha256"
    ):
        raise ValueError("Base transition evidence capsule payload hash mismatch")
    if str(result.get("graph_file_sha256") or "") != _sha256_file(graph_path):
        raise ValueError("Base transition evidence capsule graph mismatch")

    graph = _read_object(graph_path)
    graph_nodes = {
        str(row.get("id") or ""): row
        for row in graph.get("nodes") or []
        if isinstance(row, Mapping) and row.get("id")
    }
    specs = _atomic_alias_specs(graph_nodes, task_id=task_id)
    specs_by_source = {
        str(row["source_transition_id"]): row for row in specs
    }
    allowed_journal_sha256s = {
        str(value) for value in result.get("journal_sha256s") or []
    }
    code_blobs = {
        str(row["code_sha256"]): dict(row)
        for row in result.get("code_blobs") or []
    }
    node_bindings = {
        str(row["node_id"]): dict(row) for row in result.get("nodes") or []
    }
    transitions = {
        str(row["transition_id"]): dict(row)
        for row in result.get("transitions") or []
    }
    recovered_transition_ids: list[str] = []
    inspected_journals: list[dict[str, Any]] = []

    def bind_node(
        node_id: str,
        code: str,
        *,
        journal_path: Path,
        journal_sha256: str,
        source_raw_node_id: str,
    ) -> str:
        code_sha = _sha256_text(code)
        graph_node = graph_nodes.get(node_id)
        if not isinstance(graph_node, Mapping) or graph_node.get("type") != "RunNode":
            raise ValueError(f"Atomic alias endpoint is missing: {node_id}")
        if code_sha != str(graph_node.get("code_sha256") or ""):
            raise ValueError(f"Atomic alias endpoint code mismatch: {node_id}")
        existing = node_bindings.get(node_id)
        if existing is not None and str(existing.get("code_sha256") or "") != code_sha:
            raise ValueError(f"Conflicting atomic alias endpoint code: {node_id}")
        node_bindings.setdefault(
            node_id,
            {
                "node_id": node_id,
                "code_sha256": code_sha,
                "source_journal": str(journal_path),
                "source_journal_sha256": journal_sha256,
                "source_raw_node_id": source_raw_node_id,
            },
        )
        code_blobs.setdefault(code_sha, {"code_sha256": code_sha, "code": code})
        return code_sha

    unresolved_sources = set(specs_by_source).difference(transitions)
    for raw_path in atomic_journal_paths:
        path = Path(raw_path).resolve(strict=True)
        journal_sha = _sha256_file(path)
        inventory_status = (
            "base_frozen_inventory"
            if journal_sha in allowed_journal_sha256s
            else "formal_atomic_claim_extension"
        )
        payload = _read_object(path)
        raw_nodes = {
            str(row.get("id") or ""): row
            for row in payload.get("nodes") or []
            if isinstance(row, Mapping) and row.get("id")
        }
        matched = []
        for transition_id in sorted(unresolved_sources):
            spec = specs_by_source[transition_id]
            parent_graph = graph_nodes[str(spec["source_parent_node_id"])]
            child_graph = graph_nodes[str(spec["source_child_node_id"])]
            parent_raw_id = str(
                parent_graph.get("original_node_id")
                or str(spec["source_parent_node_id"]).rsplit("::node::", 1)[-1]
            )
            child_raw_id = str(
                child_graph.get("original_node_id")
                or str(spec["source_child_node_id"]).rsplit("::node::", 1)[-1]
            )
            parent_raw = raw_nodes.get(parent_raw_id)
            child_raw = raw_nodes.get(child_raw_id)
            if not isinstance(parent_raw, Mapping) or not isinstance(
                child_raw, Mapping
            ):
                continue
            before_code = str(parent_raw.get("code") or "")
            after_code = str(child_raw.get("code") or "")
            if (
                not before_code.strip()
                or not after_code.strip()
                or _sha256_text(before_code) != str(spec["before_code_sha256"])
                or _sha256_text(after_code) != str(spec["after_code_sha256"])
            ):
                raise ValueError(
                    f"Atomic alias journal/code mismatch: {transition_id}"
                )
            transition = graph_nodes[transition_id]
            parent_id = str(spec["source_parent_node_id"])
            child_id = str(spec["source_child_node_id"])
            before_sha = bind_node(
                parent_id,
                before_code,
                journal_path=path,
                journal_sha256=journal_sha,
                source_raw_node_id=parent_raw_id,
            )
            after_sha = bind_node(
                child_id,
                after_code,
                journal_path=path,
                journal_sha256=journal_sha,
                source_raw_node_id=child_raw_id,
            )
            child = graph_nodes[child_id]
            transitions[transition_id] = {
                "transition_id": transition_id,
                "task_id": task_id,
                "outcome": "debug_fixed",
                "evidence_class": "strict_debug_observed",
                "pair_key": _pair_key("debug_fixed", before_sha, after_sha),
                "parent_node_id": parent_id,
                "child_node_id": child_id,
                "before_code_sha256": before_sha,
                "after_code_sha256": after_sha,
                "parent_metric": transition.get("parent_metric"),
                "child_metric": transition.get("child_metric"),
                "metric_improvement": transition.get("metric_improvement"),
                "metric_provenance": child.get("metric_provenance"),
                "stage_pair": transition.get("stage_pair"),
                "source_journal": str(path),
                "source_journal_sha256": journal_sha,
            }
            matched.append(transition_id)
            recovered_transition_ids.append(transition_id)
        unresolved_sources.difference_update(matched)
        inspected_journals.append(
            {
                "path": str(path),
                "sha256": journal_sha,
                "inventory_status": inventory_status,
                "recovered_transition_ids": matched,
            }
        )

    transition_rows = sorted(
        transitions.values(), key=lambda row: str(row["transition_id"])
    )
    pairs = _pair_rows(transition_rows)
    candidate_aliases = _candidate_alias_rows(specs, set(transitions))
    debug_pairs = sum(row["outcome"] == "debug_fixed" for row in pairs)
    improve_pairs = sum(row["outcome"] == "metric_improved" for row in pairs)
    result.update(
        {
            "transition_count": len(transition_rows),
            "unique_pair_count": len(pairs),
            "debug_transition_count": sum(
                row["outcome"] == "debug_fixed" for row in transition_rows
            ),
            "debug_unique_pair_count": debug_pairs,
            "improve_transition_count": sum(
                row["outcome"] == "metric_improved" for row in transition_rows
            ),
            "improve_unique_pair_count": improve_pairs,
            "candidate_alias_count": len(candidate_aliases),
            "materialized_candidate_alias_count": sum(
                bool(row["materialized"]) for row in candidate_aliases
            ),
            "node_binding_count": len(node_bindings),
            "unique_code_count": len(code_blobs),
            "code_bytes": sum(len(str(row["code"])) for row in code_blobs.values()),
            "evidence_class_counts": {
                evidence_class: sum(
                    row["evidence_class"] == evidence_class
                    for row in transition_rows
                )
                for evidence_class in sorted(EVIDENCE_CLASSES)
            },
            "code_blobs": [code_blobs[key] for key in sorted(code_blobs)],
            "nodes": [node_bindings[key] for key in sorted(node_bindings)],
            "transitions": transition_rows,
            "pairs": pairs,
            "candidate_aliases": candidate_aliases,
            "atomic_alias_extension": {
                "schema": "mlevolve_atomic_alias_extension_v1",
                "base_capsule_file_sha256": _sha256_file(base_capsule_path),
                "base_capsule_sha256": str(result.get("capsule_sha256") or ""),
                "formal_atomic_claim_count": len(specs),
                "recovered_transition_ids": sorted(set(recovered_transition_ids)),
                "extension_journal_sha256s": sorted(
                    {
                        str(row["sha256"])
                        for row in inspected_journals
                        if row["inventory_status"]
                        == "formal_atomic_claim_extension"
                        and row["recovered_transition_ids"]
                    }
                ),
                "unmaterialized_source_transition_ids": sorted(
                    set(specs_by_source).difference(transitions)
                ),
                "inspected_journals": inspected_journals,
            },
        }
    )
    result["capsule_sha256"] = _payload_hash(result, "capsule_sha256")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--base-capsule", type=Path)
    parser.add_argument("--atomic-journal", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-id", default=TASK_ID)
    parser.add_argument("--expected-debug-unique-pairs", type=int)
    parser.add_argument("--expected-improve-unique-pairs", type=int)
    args = parser.parse_args()
    if args.base_capsule:
        if args.source_inventory or args.source_root:
            parser.error(
                "--base-capsule cannot be combined with --source-inventory/--source-root"
            )
        payload = augment_atomic_aliases(
            base_capsule_path=args.base_capsule,
            graph_path=args.graph,
            atomic_journal_paths=args.atomic_journal,
            task_id=str(args.task_id),
        )
    else:
        if not args.source_inventory or not args.source_root:
            parser.error(
                "fresh builds require --source-inventory and --source-root"
            )
        if args.atomic_journal:
            parser.error("--atomic-journal requires --base-capsule")
        payload = build(
            graph_path=args.graph,
            source_inventory_path=args.source_inventory,
            source_root=args.source_root,
            task_id=str(args.task_id),
            expected_debug_unique_pairs=args.expected_debug_unique_pairs,
            expected_improve_unique_pairs=args.expected_improve_unique_pairs,
        )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable capsule: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "capsule_sha256",
                    "journal_count",
                    "transition_count",
                    "unique_pair_count",
                    "debug_transition_count",
                    "debug_unique_pair_count",
                    "improve_transition_count",
                    "improve_unique_pair_count",
                    "candidate_alias_count",
                    "materialized_candidate_alias_count",
                    "unique_code_count",
                    "code_bytes",
                )
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
