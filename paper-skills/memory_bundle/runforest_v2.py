from __future__ import annotations

import collections
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from authority.domain_scope import (
    DOMAIN_GENERAL,
    SAME_DOMAIN,
    canonical_domain,
    normalize_transfer_scope,
    transfer_is_compatible,
)
from bind_sop_clauses import read_jsonl, write_jsonl
from build_corpus_manifest import journal_nodes
from schema import (
    CorpusManifestV1,
    SplitManifestV1,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")


def _audit_sidecars(audit_dir: Path) -> dict[str, dict[str, Any]]:
    index = read_json(audit_dir / "index.json")
    return {
        str(artifact_id): read_json(audit_dir / str(filename))
        for artifact_id, filename in (index.get("entries") or {}).items()
    }


def _node_id(node: Mapping[str, Any], index: int) -> str:
    return str(node.get("id") or node.get("node_id") or index)


def run_node_ref(run_id: str, node_id: str) -> str:
    return f"run::{run_id}::node::{node_id}"


def transition_ref(run_id: str, parent_id: str, child_id: str) -> str:
    suffix = hashlib.sha256(f"{run_id}\0{parent_id}\0{child_id}".encode()).hexdigest()[
        :16
    ]
    return f"run::{run_id}::transition::{suffix}"


def source_run_id(reference: str) -> str | None:
    match = re.match(r"^run::(.+?)::(?:node|transition)::", str(reference))
    return match.group(1) if match else None


def _metric(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _maximize(node: Mapping[str, Any]) -> bool | None:
    metric = node.get("metric")
    if isinstance(metric, Mapping) and metric.get("maximize") is not None:
        return bool(metric["maximize"])
    return None


def _improvement(child: Mapping[str, Any], parent: Mapping[str, Any]) -> float | None:
    child_metric = _metric(child.get("metric"))
    parent_metric = _metric(parent.get("metric"))
    if child_metric is None or parent_metric is None:
        return None
    maximize = _maximize(child)
    return (
        child_metric - parent_metric
        if maximize is not False
        else parent_metric - child_metric
    )


def transition_outcome(
    child: Mapping[str, Any], parent: Mapping[str, Any]
) -> str:
    """Classify an observed journal edge without inventing outcome evidence."""

    if (
        str(parent.get("stage") or "") == "root"
        and child.get("is_buggy") is False
        and child.get("is_valid") is True
    ):
        return "initial_valid"
    if parent.get("is_buggy") is True and child.get("is_buggy") is False:
        return "debug_fixed"
    if child.get("is_buggy") is True:
        return "buggy"
    improvement = _improvement(child, parent)
    if improvement is None:
        return "unknown"
    if improvement > 1e-12:
        return "metric_improved"
    if improvement < -1e-12:
        return "metric_worsened"
    return "metric_flat"


def journal_parent_links(
    journal: Mapping[str, Any],
    indexed_nodes: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, str]:
    """Resolve the serialized search tree from modern and legacy journals.

    Production MLEvolve journals store topology in the top-level
    ``node2parent`` map; small legacy fixtures sometimes inline ``parent_id``.
    When both are present they must agree.
    """

    nodes = {str(node_id): node for node_id, node in indexed_nodes}
    parents: dict[str, str] = {}
    raw_topology = journal.get("node2parent") or {}
    if not isinstance(raw_topology, Mapping):
        raise ValueError("Journal node2parent must be an object")
    for child, parent in raw_topology.items():
        child_id = str(child)
        parent_id = str(parent)
        if child_id not in nodes or parent_id not in nodes:
            raise ValueError(
                f"Journal topology references an unknown node: {parent_id}->{child_id}"
            )
        if child_id == parent_id:
            raise ValueError(f"Journal topology contains a self edge: {child_id}")
        parents[child_id] = parent_id
    for child_id, node in nodes.items():
        inline = str(node.get("parent_id") or "")
        if not inline:
            continue
        if inline not in nodes or inline == child_id:
            raise ValueError(
                f"Journal inline parent is invalid: {inline}->{child_id}"
            )
        if child_id in parents and parents[child_id] != inline:
            raise ValueError(
                f"Journal topology disagrees for {child_id}: "
                f"{parents[child_id]} != {inline}"
            )
        parents[child_id] = inline

    # A cycle would make both causal transition semantics and depth undefined.
    for node_id in nodes:
        seen: set[str] = set()
        current = node_id
        while current in parents:
            if current in seen:
                raise ValueError(f"Journal topology contains a cycle at {current}")
            seen.add(current)
            current = parents[current]
    return parents


def journal_local_best_links(
    journal: Mapping[str, Any],
    indexed_nodes: Iterable[tuple[str, Mapping[str, Any]]],
) -> dict[str, str]:
    nodes = {str(node_id) for node_id, _node in indexed_nodes}
    raw = journal.get("node2best_local_node") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Journal node2best_local_node must be an object")
    links: dict[str, str] = {}
    for node_id, best_id in raw.items():
        node_text = str(node_id)
        best_text = str(best_id)
        if node_text not in nodes or best_text not in nodes:
            raise ValueError(
                "Journal local-best topology references an unknown node: "
                f"{node_text}->{best_text}"
            )
        links[node_text] = best_text
    return links


def audit_projection(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the production leakage disposition from an immutable sidecar."""

    status = str(sidecar.get("status") or "audit_unavailable")
    dispositions = {
        "clean": ("positive_eligible", "accept", "allow", "normal"),
        "blocked": ("quarantine", "reject", "block", "blocked"),
        "protocol_biased": (
            "negative_only",
            "protocol_biased",
            "allow_diagnostic",
            "repair_only",
        ),
        "warning": (
            "negative_only",
            "unverified",
            "allow_diagnostic",
            "repair_only",
        ),
        "audit_unavailable": (
            "negative_only",
            "unverified",
            "allow_diagnostic",
            "provisional",
        ),
    }
    if status not in dispositions:
        raise ValueError(f"Unsupported audit sidecar status: {status}")
    memory, metric, execution, search = dispositions[status]
    clean = status == "clean"
    return {
        "schema": sidecar.get("detector_schema"),
        "detector_version": sidecar.get("detector_version"),
        "status": status,
        "hard_block": status == "blocked",
        "paper_grade_eligible": clean,
        "metric_disposition": metric,
        "memory_disposition": memory,
        "execution_disposition": execution,
        "search_disposition": search,
        "rank_eligible": clean,
        "repair_required": not clean,
        "issues": sidecar.get("issues") or [],
        "legacy_receipt_level": sidecar.get("legacy_receipt_level"),
        "audit_sidecar_sha256": sidecar.get("sidecar_sha256"),
    }


def _short(value: Any, limit: int = 1200) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit]} …"


def _node_text(node: Mapping[str, Any]) -> str:
    terminal = node.get("_term_out")
    if isinstance(terminal, list):
        terminal = "".join(str(value) for value in terminal)
    return "\n".join(
        text
        for text in (
            _short(node.get("plan")),
            _short(node.get("code_summary")),
            _short(node.get("analysis")),
            _short(node.get("exc_info")),
            _short(terminal),
        )
        if text
    )


def _hash_embedding(text: str, dims: int = 64) -> np.ndarray:
    vector = np.zeros(dims, dtype=np.float32)
    for token in TOKEN_RE.findall(text.lower()):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _coordinates(node_ids: list[str], node_types: list[str]) -> np.ndarray:
    coordinates = np.zeros((len(node_ids), 2), dtype=np.float32)
    radii = {
        "Run": 0.08,
        "RunNode": 0.30,
        "Transition": 0.45,
        "Evidence": 0.62,
        "FailurePattern": 0.68,
        "SOP": 0.52,
        "SOPClause": 0.62,
    }
    for index, (node_id, node_type) in enumerate(zip(node_ids, node_types)):
        digest = hashlib.sha256(node_id.encode()).digest()
        angle = 2 * math.pi * int.from_bytes(digest[:8], "big") / 2**64
        radius = radii.get(node_type, 0.55)
        coordinates[index] = [radius * math.cos(angle), radius * math.sin(angle)]
    return coordinates


def _decision_allows(
    clause: Mapping[str, Any], decisions: Mapping[str, Mapping[str, Any]]
) -> bool:
    refs = [str(value) for value in clause.get("authority_decision_refs") or []]
    if not refs or str(clause.get("publication_class")) != "certified":
        return False
    for decision_ref in refs:
        decision = decisions.get(decision_ref)
        if not decision:
            continue
        if str(decision.get("outcome")) not in {"allow", "allow_with_warning"}:
            continue
        scope = decision.get("permitted_scope") or {}
        if not set(clause.get("claim_refs") or []) & {
            str(decision.get("claim_id") or "")
        }:
            continue
        if not set(clause.get("permitted_operations") or []) & set(
            scope.get("operations") or []
        ):
            continue
        return True
    return False


def build_runforest_v2(
    corpus_manifest_path: str | Path,
    audit_dir: str | Path,
    clauses_path: str | Path,
    containers_path: str | Path,
    split_manifest_path: str | Path,
    out_dir: str | Path,
    *,
    bundle_id: str,
    authority_decisions_path: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    corpus_manifest_path = Path(corpus_manifest_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    clauses_path = Path(clauses_path).resolve()
    containers_path = Path(containers_path).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"RunForest v2 output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    visibility_dir = out_dir / "visibility"
    visibility_dir.mkdir()
    precompiled_mask_dir = visibility_dir / "precompiled_masks"
    precompiled_mask_dir.mkdir()
    manifest = CorpusManifestV1.from_dict(read_json(corpus_manifest_path))
    split = SplitManifestV1.from_dict(read_json(split_manifest_path))
    if split.corpus_manifest_hash != manifest.manifest_sha256:
        raise ValueError("Split manifest does not bind corpus manifest")
    source_runs = set(split.source_run_ids)
    heldout_runs = set(split.heldout_run_ids)
    if source_runs & heldout_runs:
        raise ValueError("Source/heldout run overlap")
    entries = {run.run_id: run for run in manifest.runs}
    missing_runs = source_runs - set(entries)
    if missing_runs:
        raise ValueError(f"Unknown source runs: {sorted(missing_runs)}")
    missing_heldout_runs = heldout_runs - set(entries)
    if missing_heldout_runs:
        raise ValueError(f"Unknown held-out runs: {sorted(missing_heldout_runs)}")
    if any(entries[run_id].status != "complete" for run_id in source_runs):
        raise ValueError("Split source includes non-complete run")
    if any("spooky" in entries[run_id].canonical_task_id for run_id in source_runs):
        raise ValueError("Spooky run cannot enter a formal bundle")
    source_task_ids = {str(entries[run_id].canonical_task_id) for run_id in source_runs}
    source_task_families = {
        str(entries[run_id].task_family or "").strip() for run_id in source_runs
    }
    source_task_families.discard("")
    source_domains = {canonical_domain(family) for family in source_task_families}
    source_domains.discard("")
    heldout_task_ids = {
        str(entries[run_id].canonical_task_id)
        for run_id in heldout_runs
        if run_id in entries
    }
    heldout_task_families = {
        str(entries[run_id].task_family or "").strip()
        for run_id in heldout_runs
        if run_id in entries
    }
    heldout_task_families.discard("")
    heldout_domains = {canonical_domain(family) for family in heldout_task_families}
    heldout_domains.discard("")
    same_domain_split = split.split_kind == "same-domain-task-heldout"
    target_task_id = str(split.allocation.get("target_task_id") or "")
    target_task_family = str(split.allocation.get("target_task_family") or "")
    target_domain = canonical_domain(
        split.allocation.get("target_domain") or target_task_family
    )
    if same_domain_split:
        if not target_task_id or split.heldout_task_ids != [target_task_id]:
            raise ValueError(
                "Same-domain split must bind exactly one held-out target task"
            )
        if target_task_id in source_task_ids:
            raise ValueError("Same-domain target task leaked into source tasks")
        if not target_domain or source_domains != {target_domain}:
            raise ValueError(
                "Same-domain split source domains do not match target domain"
            )
        if heldout_runs:
            if heldout_domains != {target_domain}:
                raise ValueError(
                    "Same-domain split held-out domain does not match target domain"
                )
        elif (
            heldout_domains
            or split.allocation.get("transfer_design")
            != "same_domain_different_task_target_history_absent"
        ):
            # A genuinely new task has no historical held-out run from which
            # to infer a domain.  In that stricter design the target domain is
            # bound directly by the split allocation, while the source view
            # above must still be exactly same-domain and target-history-free.
            raise ValueError(
                "Same-domain split without held-out runs lacks an explicit "
                "target-history-absent binding"
            )
    sidecars = _audit_sidecars(audit_dir)
    raw_clauses = read_jsonl(clauses_path)
    container_payload = read_json(containers_path)
    raw_containers = [
        dict(value) for value in container_payload.get("containers") or []
    ]
    decisions = {
        str(row["decision_id"]): row
        for row in (
            read_jsonl(authority_decisions_path) if authority_decisions_path else []
        )
    }
    selected_clauses: list[dict[str, Any]] = []
    excluded_clauses: list[dict[str, Any]] = []
    # Task IDs overlap by design in a seed-heldout split.  Task-scope
    # exclusion is only valid for task-heldout experiments; seed-heldout
    # isolation is enforced by source run/seed membership below.
    heldout_tasks = (
        set(split.heldout_task_ids)
        if split.split_kind in {"task-heldout", "same-domain-task-heldout"}
        else set()
    )
    for clause in raw_clauses:
        refs = [
            *list(clause.get("source_artifact_refs") or []),
            *list(clause.get("source_transition_refs") or []),
        ]
        parsed_ref_runs = [source_run_id(str(ref)) for ref in refs]
        referenced_runs = {run_id for run_id in parsed_ref_runs if run_id}
        task_scope = (
            dict(clause.get("task_scope"))
            if isinstance(clause.get("task_scope"), Mapping)
            else {}
        )
        declared_task_ids = {
            str(value)
            for value in (task_scope.get("task_ids") or [task_scope.get("task_id")])
            if value not in {None, ""}
        }
        referenced_task_ids = {
            str(entries[run_id].canonical_task_id)
            for run_id in referenced_runs
            if run_id in entries
        }
        referenced_task_families = {
            str(entries[run_id].task_family or "").strip()
            for run_id in referenced_runs
            if run_id in entries
        }
        referenced_task_families.discard("")
        referenced_domains = {
            canonical_domain(family) for family in referenced_task_families
        }
        referenced_domains.discard("")
        raw_transfer_scope = str(clause.get("transfer_scope") or "")
        transfer_scope = normalize_transfer_scope(raw_transfer_scope) or SAME_DOMAIN
        reason = ""
        if not refs:
            reason = "missing_source_refs"
        elif any(run_id is None for run_id in parsed_ref_runs):
            reason = "unparseable_source_ref"
        elif referenced_runs - set(entries):
            reason = "unknown_source_run"
        elif not referenced_runs <= source_runs:
            reason = "source_ref_outside_split"
        elif declared_task_ids & heldout_tasks:
            reason = "heldout_task_scope"
        elif declared_task_ids and declared_task_ids != referenced_task_ids:
            reason = "declared_source_task_scope_mismatch"
        elif not referenced_task_ids:
            reason = "missing_source_task_lineage"
        elif not referenced_task_families or not referenced_domains:
            reason = "missing_source_domain_lineage"
        elif raw_transfer_scope and not normalize_transfer_scope(raw_transfer_scope):
            reason = "invalid_transfer_scope"
        elif (
            same_domain_split
            and transfer_scope == SAME_DOMAIN
            and not transfer_is_compatible(
                referenced_domains,
                target_domain,
                transfer_scope,
            )
        ):
            reason = "cross_domain_clause_source"
        if reason:
            excluded_clauses.append(
                {"clause_id": clause.get("clause_id"), "reason": reason}
            )
        else:
            value = dict(clause)
            value["task_scope"] = task_scope
            value["source_run_ids"] = sorted(referenced_runs)
            value["source_task_ids"] = sorted(referenced_task_ids)
            value["source_task_families"] = sorted(referenced_task_families)
            value["source_domains"] = sorted(referenced_domains)
            value["transfer_scope"] = transfer_scope
            value["admissible_target_domains"] = (
                [DOMAIN_GENERAL]
                if transfer_scope == DOMAIN_GENERAL
                else sorted(referenced_domains)
            )
            selected_clauses.append(value)
    selected_clause_ids = {row["clause_id"] for row in selected_clauses}
    selected_clauses_by_id = {str(row["clause_id"]): row for row in selected_clauses}
    selected_containers: list[dict[str, Any]] = []
    clause_container: dict[str, str] = {}
    for container in raw_containers:
        clause_ids = [
            value
            for value in container.get("clause_ids") or []
            if value in selected_clause_ids
        ]
        if not clause_ids:
            continue
        value = dict(container)
        value["clause_ids"] = sorted(clause_ids)
        member_clauses = [selected_clauses_by_id[item] for item in clause_ids]
        value["source_run_ids"] = sorted(
            {
                source_id
                for row in member_clauses
                for source_id in row.get("source_run_ids") or []
            }
        )
        value["source_task_ids"] = sorted(
            {
                source_id
                for row in member_clauses
                for source_id in row.get("source_task_ids") or []
            }
        )
        value["source_task_families"] = sorted(
            {
                family
                for row in member_clauses
                for family in row.get("source_task_families") or []
            }
        )
        value["source_domains"] = sorted(
            {
                domain
                for row in member_clauses
                for domain in row.get("source_domains") or []
            }
        )
        value["transfer_scopes"] = sorted(
            {str(row.get("transfer_scope") or "") for row in member_clauses}
        )
        value["domain_scope_complete"] = bool(
            value["source_run_ids"]
            and value["source_task_ids"]
            and value["source_task_families"]
            and value["source_domains"]
            and all(
                normalize_transfer_scope(scope) for scope in value["transfer_scopes"]
            )
        )
        selected_containers.append(value)
        for clause_id in clause_ids:
            if clause_id in clause_container:
                raise ValueError(f"Clause belongs to multiple containers: {clause_id}")
            clause_container[clause_id] = str(value["sop_id"])
    orphan_clauses = selected_clause_ids - set(clause_container)
    if orphan_clauses:
        raise ValueError(f"Clauses have no container: {sorted(orphan_clauses)}")

    graph_nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    node_refs: set[str] = set()
    transition_refs: set[str] = set()
    source_root = Path(manifest.source_root)
    audited_code_nodes = 0
    for run_id in sorted(source_runs):
        entry = entries[run_id]
        run_ref = f"run::{run_id}"
        graph_nodes.append(
            {
                "id": run_ref,
                "type": "Run",
                "run_id": run_id,
                "task": entry.canonical_task_id,
                "task_family": entry.task_family,
                "task_domain": canonical_domain(entry.task_family),
                "seed": entry.seed,
                "source_artifact_hashes": entry.artifact_hashes,
            }
        )
        journal_path = source_root / str(entry.journal_path)
        if sha256_file(journal_path) != entry.artifact_hashes.get("journal"):
            raise ValueError(f"Journal hash drift: {run_id}")
        journal_payload = read_json(journal_path)
        journal = journal_nodes(journal_payload)
        raw_to_ref: dict[str, str] = {}
        indexed_nodes: list[tuple[str, dict[str, Any]]] = []
        for index, raw_node in enumerate(journal):
            raw_id = _node_id(raw_node, index)
            if raw_id in raw_to_ref:
                raise ValueError(f"Duplicate raw journal node id: {raw_id}")
            reference = run_node_ref(run_id, raw_id)
            raw_to_ref[raw_id] = reference
            indexed_nodes.append((raw_id, raw_node))
        indexed_node_by_id = {raw_id: node for raw_id, node in indexed_nodes}
        parent_links = journal_parent_links(journal_payload, indexed_nodes)
        local_best_links = journal_local_best_links(journal_payload, indexed_nodes)
        children: dict[str, list[str]] = collections.defaultdict(list)
        for child_id, parent_id in parent_links.items():
            children[parent_id].append(child_id)
        for values in children.values():
            values.sort(
                key=lambda node_id: (
                    int(indexed_node_by_id[node_id].get("step") or 0),
                    node_id,
                )
            )
        for raw_id, raw_node in indexed_nodes:
            reference = raw_to_ref[raw_id]
            node_refs.add(reference)
            code = str(raw_node.get("code") or "")
            sidecar = sidecars.get(reference)
            if code.strip() and sidecar is None:
                raise ValueError(f"Missing audit sidecar: {reference}")
            if sidecar is not None:
                code_hash = hashlib.sha256(code.encode()).hexdigest()
                if code_hash != sidecar.get("code_sha256"):
                    raise ValueError(f"Audit sidecar code hash mismatch: {reference}")
                audited_code_nodes += 1
            node = {
                "id": reference,
                "type": "RunNode",
                "run_id": run_id,
                "task": entry.canonical_task_id,
                "task_family": entry.task_family,
                "task_domain": canonical_domain(entry.task_family),
                "seed": entry.seed,
                "raw_node_id": raw_id,
                "branch_id": raw_node.get("branch_id"),
                "step": raw_node.get("step"),
                "stage": raw_node.get("stage"),
                "parent_id": (
                    raw_to_ref[parent_links[raw_id]]
                    if raw_id in parent_links
                    else None
                ),
                "local_best_node_id": (
                    raw_to_ref[local_best_links[raw_id]]
                    if raw_id in local_best_links
                    else None
                ),
                "is_leaf": not bool(children.get(raw_id)),
                "num_children": len(children.get(raw_id, [])),
                "is_buggy": raw_node.get("is_buggy"),
                "is_valid": raw_node.get("is_valid"),
                "metric": _metric(raw_node.get("metric")),
                "metric_maximize": _maximize(raw_node),
                "metric_improvement": (
                    _improvement(
                        raw_node,
                        next(
                            node
                            for node_id, node in indexed_nodes
                            if node_id == parent_links[raw_id]
                        ),
                    )
                    if raw_id in parent_links
                    else None
                ),
                "code_sha256": (
                    hashlib.sha256(code.encode()).hexdigest() if code.strip() else ""
                ),
                "audit_sidecar_ref": (
                    str(sidecar.get("sidecar_sha256")) if sidecar else ""
                ),
                "leakage_audit": audit_projection(sidecar) if sidecar else {},
                "audit_status": sidecar.get("status") if sidecar else None,
                "paper_grade_eligible": (
                    audit_projection(sidecar)["paper_grade_eligible"]
                    if sidecar
                    else False
                ),
                "memory_disposition": (
                    audit_projection(sidecar)["memory_disposition"]
                    if sidecar
                    else "negative_only"
                ),
                "plan": _short(raw_node.get("plan")),
                "code_summary": _short(raw_node.get("code_summary")),
                "analysis": _short(raw_node.get("analysis")),
                "terminal_excerpt": _short(
                    "".join(
                        str(value)
                        for value in (
                            raw_node.get("_term_out")
                            if isinstance(raw_node.get("_term_out"), list)
                            else [raw_node.get("_term_out") or ""]
                        )
                    )
                ),
                "text": _node_text(raw_node),
            }
            graph_nodes.append(node)
            edges.append({"src": run_ref, "dst": reference, "kind": "contains_node"})
            if sidecar:
                evidence_id = f"evidence::{hashlib.sha256(f'{reference}:audit'.encode()).hexdigest()[:20]}"
                graph_nodes.append(
                    {
                        "id": evidence_id,
                        "type": "Evidence",
                        "artifact_id": reference,
                        "evidence_kind": "legacy_static_audit_sidecar",
                        "audit_status": sidecar.get("status"),
                        "audit_sidecar_ref": sidecar.get("sidecar_sha256"),
                        "text": "Static sidecar; not an original runtime Receipt.",
                    }
                )
                edges.append(
                    {"src": reference, "dst": evidence_id, "kind": "supported_by"}
                )
            parent_raw = parent_links.get(raw_id, "")
            if parent_raw:
                parent_ref = raw_to_ref[parent_raw]
                edges.append({"src": parent_ref, "dst": reference, "kind": "parent_of"})
                parent_node = next(
                    value for value_id, value in indexed_nodes if value_id == parent_raw
                )
                transition_id = transition_ref(run_id, parent_raw, raw_id)
                transition_refs.add(transition_id)
                transition = {
                    "id": transition_id,
                    "type": "Transition",
                    "run_id": run_id,
                    "task": entry.canonical_task_id,
                    "task_family": entry.task_family,
                    "task_domain": canonical_domain(entry.task_family),
                    "parent_node_id": parent_ref,
                    "child_node_id": reference,
                    "parent_original_node_id": parent_raw,
                    "child_original_node_id": raw_id,
                    "branch_id": raw_node.get("branch_id"),
                    "stage_pair": f"{parent_node.get('stage')}->{raw_node.get('stage')}",
                    "parent_metric": _metric(parent_node.get("metric")),
                    "child_metric": _metric(raw_node.get("metric")),
                    "metric_delta": (
                        None
                        if _metric(parent_node.get("metric")) is None
                        or _metric(raw_node.get("metric")) is None
                        else _metric(raw_node.get("metric"))
                        - _metric(parent_node.get("metric"))
                    ),
                    "metric_improvement": _improvement(raw_node, parent_node),
                    "outcome": transition_outcome(raw_node, parent_node),
                    "parent_buggy": parent_node.get("is_buggy"),
                    "child_buggy": raw_node.get("is_buggy"),
                    "text": _short(
                        "\n".join(
                            value
                            for value in (
                                _node_text(parent_node),
                                _node_text(raw_node),
                            )
                            if value
                        )
                    ),
                }
                graph_nodes.append(transition)
                edges.extend(
                    [
                        {
                            "src": parent_ref,
                            "dst": transition_id,
                            "kind": "has_transition",
                        },
                        {
                            "src": transition_id,
                            "dst": reference,
                            "kind": "transition_to",
                        },
                    ]
                )
            if sidecar:
                for issue_index, issue in enumerate(sidecar.get("issues") or []):
                    failure_id = f"failure::{hashlib.sha256(f'{reference}:{issue_index}'.encode()).hexdigest()[:20]}"
                    graph_nodes.append(
                        {
                            "id": failure_id,
                            "type": "FailurePattern",
                            "issue_code": issue.get("issue_code"),
                            "category": issue.get("category"),
                            "text": _short(
                                issue.get("evidence") or issue.get("reason")
                            ),
                            "audit_sidecar_ref": sidecar.get("sidecar_sha256"),
                        }
                    )
                    edges.append(
                        {
                            "src": reference,
                            "dst": failure_id,
                            "kind": "has_failure_pattern",
                        }
                    )

    for container in sorted(selected_containers, key=lambda row: row["sop_id"]):
        graph_nodes.append(
            {
                "id": str(container["sop_id"]),
                "type": "SOP",
                "sop_id": str(container["sop_id"]),
                "title": container.get("title"),
                "task": container.get("task_id"),
                "clause_ids": container.get("clause_ids"),
                "source_run_ids": container.get("source_run_ids") or [],
                "source_task_ids": container.get("source_task_ids") or [],
                "source_task_families": container.get("source_task_families") or [],
                "source_domains": container.get("source_domains") or [],
                "task_families": container.get("source_domains") or [],
                "transfer_scopes": container.get("transfer_scopes") or [],
                "domain_scope_complete": container.get("domain_scope_complete") is True,
            }
        )
    navigation_pairs: set[tuple[str, str]] = set()
    authorized_pairs: set[tuple[str, str]] = set()
    navigation_edges: dict[tuple[str, str], dict[str, Any]] = {}
    authorized_edges: dict[tuple[str, str], dict[str, Any]] = {}
    visibility_rows: list[dict[str, Any]] = []
    for clause in sorted(selected_clauses, key=lambda row: row["clause_id"]):
        container_id = clause_container[clause["clause_id"]]
        value = dict(clause)
        value["id"] = clause["clause_id"]
        value["type"] = "SOPClause"
        value["sop_id"] = container_id
        graph_nodes.append(value)
        edges.append(
            {"src": container_id, "dst": clause["clause_id"], "kind": "contains_clause"}
        )
        for source_ref in clause.get("source_artifact_refs") or []:
            if source_ref not in node_refs:
                raise ValueError(f"Unresolved clause artifact ref: {source_ref}")
            edges.append(
                {
                    "src": source_ref,
                    "dst": clause["clause_id"],
                    "kind": "derived_into_clause",
                }
            )
        authorized = _decision_allows(clause, decisions)
        for source_ref in clause.get("source_transition_refs") or []:
            if source_ref not in transition_refs:
                raise ValueError(f"Unresolved clause transition ref: {source_ref}")
            edges.append(
                {
                    "src": source_ref,
                    "dst": clause["clause_id"],
                    "kind": "derived_into_clause",
                }
            )
            pair = (source_ref, container_id)
            if pair not in navigation_pairs:
                navigation_pairs.add(pair)
                navigation_edges[pair] = {
                    "src": source_ref,
                    "dst": container_id,
                    "kind": "navigation_attached_to",
                    "authority_outcome": "runtime_clause_authority_required",
                    "clause_ids": [],
                    "quality": "direct_clause_transition_lineage",
                    "score": 1.0,
                }
            navigation_edges[pair]["clause_ids"] = sorted(
                {
                    *navigation_edges[pair]["clause_ids"],
                    str(clause["clause_id"]),
                }
            )
            if authorized and pair not in authorized_pairs:
                authorized_pairs.add(pair)
                authorized_edges[pair] = {
                    "src": source_ref,
                    "dst": container_id,
                    "kind": "authorized_distills_to",
                    "authority_outcome": "allow",
                    "authority_decision_refs": [],
                    "clause_ids": [],
                    "quality": "direct_clause_transition_lineage",
                    "score": 1.0,
                }
            if authorized:
                authorized_edges[pair]["clause_ids"] = sorted(
                    {
                        *authorized_edges[pair]["clause_ids"],
                        str(clause["clause_id"]),
                    }
                )
                authorized_edges[pair]["authority_decision_refs"] = sorted(
                    {
                        *authorized_edges[pair]["authority_decision_refs"],
                        *map(str, clause.get("authority_decision_refs") or []),
                    }
                )
        visibility_rows.append(
            {
                "clause_id": clause["clause_id"],
                "sop_id": container_id,
                "claim_refs": clause.get("claim_refs") or [],
                "source_artifact_refs": clause.get("source_artifact_refs") or [],
                "source_transition_refs": clause.get("source_transition_refs") or [],
                "source_run_ids": clause.get("source_run_ids") or [],
                "source_task_ids": clause.get("source_task_ids") or [],
                "source_task_families": clause.get("source_task_families") or [],
                "source_domains": clause.get("source_domains") or [],
                "transfer_scope": clause.get("transfer_scope"),
                "admissible_target_domains": clause.get("admissible_target_domains")
                or [],
                "protocol_scope": clause.get("protocol_scope") or [],
                "task_scope": clause.get("task_scope") or {},
                "permitted_operations": clause.get("permitted_operations") or [],
                "permitted_generation_stages": clause.get("permitted_generation_stages")
                or [],
                "permitted_governance_stages": clause.get("permitted_governance_stages")
                or [],
                "publication_class": clause.get("publication_class"),
                "legacy_status": clause.get("legacy_status"),
            }
        )

    edges.extend(navigation_edges[pair] for pair in sorted(navigation_edges))
    edges.extend(authorized_edges[pair] for pair in sorted(authorized_edges))

    ids = [str(node["id"]) for node in graph_nodes]
    if len(ids) != len(set(ids)):
        duplicates = [
            item for item, count in collections.Counter(ids).items() if count > 1
        ]
        raise ValueError(f"Duplicate RunForest node IDs: {duplicates[:10]}")
    node_types = [str(node["type"]) for node in graph_nodes]
    texts = [str(node.get("text") or node.get("title") or "") for node in graph_nodes]
    poincare = _coordinates(ids, node_types)
    euclidean = (
        np.stack([_hash_embedding(text) for text in texts])
        if texts
        else np.zeros((0, 64), dtype=np.float32)
    )
    clause_ids = [str(row["clause_id"]) for row in selected_clauses]
    clause_embeddings = (
        np.stack(
            [
                _hash_embedding(str(row.get("retrieval_text") or ""))
                for row in selected_clauses
            ]
        )
        if selected_clauses
        else np.zeros((0, 64), dtype=np.float32)
    )
    graph = {
        "meta": {
            "schema": "hyperbolic_run_forest_memory_v2",
            "bundle_id": bundle_id,
            "created_at": created_at or utc_now(),
            "corpus_id": manifest.corpus_id,
            "corpus_manifest_hash": manifest.manifest_sha256,
            "split_id": split.split_id,
            "split_manifest_hash": split.manifest_sha256,
            "source_run_count": len(source_runs),
            "heldout_run_count": len(heldout_runs),
            "source_task_ids": sorted(source_task_ids),
            "source_task_families": sorted(source_task_families),
            "source_domains": sorted(source_domains),
            "heldout_task_ids": sorted(heldout_task_ids),
            "heldout_task_families": sorted(heldout_task_families),
            "heldout_domains": sorted(heldout_domains),
            "transfer_design": split.allocation.get("transfer_design"),
            "target_task_id": target_task_id,
            "target_task_family": target_task_family,
            "target_domain": target_domain,
            "domain_scope_required": True,
            "domain_scope_policy": "explicit_domain_general_else_same_domain_v1",
            "certification_level": "raw_audited",
            "legacy_artifact_overwritten": False,
        },
        "nodes": graph_nodes,
        "edges": edges,
    }
    graph_path = out_dir / "graph.json"
    index_path = out_dir / "index.npz"
    clause_index_path = out_dir / "clause_index.npz"
    write_json_atomic(graph_path, graph)
    np.savez_compressed(
        index_path,
        node_ids=np.asarray(ids, dtype=object),
        node_types=np.asarray(node_types, dtype=object),
        poincare=poincare,
        flat_twin=poincare.copy(),
        euclidean=euclidean,
    )
    np.savez_compressed(
        clause_index_path,
        clause_ids=np.asarray(clause_ids, dtype=object),
        embeddings=clause_embeddings,
    )
    write_jsonl(visibility_dir / "clause_metadata.jsonl", visibility_rows)
    declared_masks: dict[str, list[str]] = collections.defaultdict(list)
    for row in visibility_rows:
        for protocol in row["protocol_scope"] or ["unscoped"]:
            for operation in row["permitted_operations"]:
                for generation_stage in row["permitted_generation_stages"]:
                    for governance_stage in row["permitted_governance_stages"]:
                        key = "|".join(
                            [
                                str(protocol),
                                str(operation),
                                str(generation_stage),
                                str(governance_stage),
                            ]
                        )
                        declared_masks[key].append(row["clause_id"])
    write_json_atomic(
        precompiled_mask_dir / "declared_scope_masks.json",
        {
            "schema": "declared_scope_visibility_masks_v1",
            "semantics": (
                "Declared-scope prefilter only; runtime Authority ALLOW is still required."
            ),
            "masks": {
                key: sorted(set(values))
                for key, values in sorted(declared_masks.items())
            },
        },
    )
    report = {
        "schema": "run_forest_builder_report_v2",
        "bundle_id": bundle_id,
        "corpus_manifest_hash": manifest.manifest_sha256,
        "split_manifest_hash": split.manifest_sha256,
        "source_run_count": len(source_runs),
        "heldout_run_count": len(heldout_runs),
        "source_task_ids": sorted(source_task_ids),
        "source_task_families": sorted(source_task_families),
        "source_domains": sorted(source_domains),
        "heldout_task_ids": sorted(heldout_task_ids),
        "heldout_task_families": sorted(heldout_task_families),
        "heldout_domains": sorted(heldout_domains),
        "transfer_design": split.allocation.get("transfer_design"),
        "target_task_id": target_task_id,
        "target_task_family": target_task_family,
        "target_domain": target_domain,
        "same_domain_task_heldout": same_domain_split,
        "spooky_source_run_count": sum(
            "spooky" in entries[run_id].canonical_task_id for run_id in source_runs
        ),
        "node_count": len(graph_nodes),
        "edge_count": len(edges),
        "node_type_counts": dict(sorted(collections.Counter(node_types).items())),
        "edge_kind_counts": dict(
            sorted(collections.Counter(edge["kind"] for edge in edges).items())
        ),
        "expected_audited_code_node_count": sum(
            entries[run_id].code_node_count for run_id in source_runs
        ),
        "audited_code_node_count": audited_code_nodes,
        "transition_count": sum(
            node.get("type") == "Transition" for node in graph_nodes
        ),
        "source_runs_with_transitions": sorted(
            {
                str(node.get("run_id") or "")
                for node in graph_nodes
                if node.get("type") == "Transition" and node.get("run_id")
            }
        ),
        "rank_eligible_run_node_count": sum(
            node.get("type") == "RunNode"
            and (node.get("leakage_audit") or {}).get("rank_eligible") is True
            for node in graph_nodes
        ),
        "all_code_nodes_have_sidecars": audited_code_nodes
        == sum(entries[run_id].code_node_count for run_id in source_runs),
        "input_clause_count": len(raw_clauses),
        "included_clause_count": len(selected_clauses),
        "excluded_clauses": excluded_clauses,
        "excluded_clause_reason_counts": dict(
            sorted(
                collections.Counter(row["reason"] for row in excluded_clauses).items()
            )
        ),
        "task_scope_exclusion_enabled": split.split_kind
        in {"task-heldout", "same-domain-task-heldout"},
        "all_included_clauses_have_domain_lineage": all(
            row.get("source_run_ids")
            and row.get("source_task_ids")
            and row.get("source_task_families")
            and row.get("source_domains")
            and normalize_transfer_scope(row.get("transfer_scope"))
            for row in selected_clauses
        ),
        "cross_domain_included_clause_ids": sorted(
            str(row["clause_id"])
            for row in selected_clauses
            if same_domain_split
            and normalize_transfer_scope(row.get("transfer_scope")) == SAME_DOMAIN
            and not transfer_is_compatible(
                row.get("source_domains") or [],
                target_domain,
                row.get("transfer_scope"),
            )
        ),
        "domain_general_clause_ids": sorted(
            str(row["clause_id"])
            for row in selected_clauses
            if normalize_transfer_scope(row.get("transfer_scope")) == DOMAIN_GENERAL
        ),
        "all_clause_sources_resolve": True,
        "navigation_edge_count": len(navigation_pairs),
        "authorized_edge_count": len(authorized_pairs),
        "heldout_run_refs_in_graph": sorted(
            heldout_runs
            & {
                str(node.get("run_id"))
                for node in graph_nodes
                if node.get("run_id") is not None
            }
        ),
        "graph_sha256": sha256_file(graph_path),
        "index_sha256": sha256_file(index_path),
        "clause_index_sha256": sha256_file(clause_index_path),
        "declared_scope_masks_sha256": sha256_file(
            precompiled_mask_dir / "declared_scope_masks.json"
        ),
        "legacy_artifact_overwritten": False,
    }
    write_json_atomic(out_dir / "build_report.json", report)
    return report
