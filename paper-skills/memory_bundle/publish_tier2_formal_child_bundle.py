from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from authority.adapters.mlevolve.node_adapter import method_fingerprint
from authority.adapters.mlevolve.retrieval_gate import authorize_clause_for_visibility
from authority.authority_engine import AuthorityEngine
from authority.bundle_authority import load_snapshot_authority
from authority.clean_replay import verify_trusted_receipt_integrity
from authority.collectors import (
    CodeExecutionCollector,
    MethodIdentityCollector,
    TrustedCollectorHost,
)
from authority.evidence_graph import EvidencePath
from authority.domain_scope import canonical_domain
from authority.memory_snapshot import (
    ImmutableBaseBundle,
    MemorySnapshotLoader,
    make_current_pointer,
    sha256_json as runtime_sha256_json,
    write_json_atomic as runtime_write_json_atomic,
)
from authority.models import (
    Claim,
    ClaimType,
    GenerationStage,
    GovernanceStage,
    Operation,
    ProtocolRef,
    Receipt,
    ReceiptType,
    SOPClauseV1,
    TaskContext,
    VisibilityRequest,
)
from authority.protocol_registry import ProtocolRegistry, canonical_json

from bind_sop_clauses import read_jsonl, write_jsonl
from build_corpus_manifest import journal_nodes
from build_memory_bundle import build_bundle
from method_claim_purity import require_method_claim_semantic_purity
from runforest_v2 import (
    build_runforest_v2,
    journal_parent_links,
    transition_outcome,
    transition_ref,
)
from schema import (
    CorpusManifestV1,
    SplitManifestV1,
    payload_hash,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    write_json_atomic,
)
from validate_memory_bundle import validate_bundle


PUBLICATION_SCHEMA = "tier2_formal_child_bundle_publication_v1"
VERIFICATION_SCHEMA = "tier2_formal_child_bundle_verification_v1"
DERIVATION_SCHEMA = "tier2_formal_method_publication_derivation_v1"
PROVISIONAL_METHOD_ONLY_PROTOCOL_ISSUES = frozenset({"TEMPORAL_SPLIT_LEAKAGE"})
FORMAL_SUPPORT_OUTCOMES = frozenset(
    {"initial_valid", "debug_fixed", "metric_improved"}
)
FORMAL_GENERATION_STAGES = (
    GenerationStage.DRAFT.value,
    GenerationStage.MODEL_DESIGN.value,
    GenerationStage.IMPROVE.value,
    GenerationStage.EVOLUTION.value,
    GenerationStage.FUSION.value,
)
FORMAL_METHOD_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")
FORMAL_METHOD_GLUE_TOKENS = frozenset(
    {
        "and",
        "are",
        "for",
        "from",
        "into",
        "the",
        "then",
        "these",
        "this",
        "through",
        "use",
        "uses",
        "using",
        "with",
    }
)
CORE_ARTIFACT_NAMES = {
    "journal_path": "journal.json",
    "config_path": "config.yaml",
    "filtered_journal_path": "filtered_journal.json",
    "best_solution_path": "best_solution.py",
}
LEGACY_PARENT_LINEAGE_ERROR_PREFIXES = frozenset(
    {
        "runforest_domain_scope_not_required",
        "runforest_source_task_lineage_mismatch",
        "runforest_source_family_lineage_mismatch",
        "runforest_source_domain_lineage_mismatch",
        "container_domain_scope_incomplete",
        "clause_source_run_lineage",
        "clause_source_task_lineage",
        "clause_source_family_lineage",
        "clause_source_domain_lineage",
        "clause_missing_transfer_scope",
    }
)


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}{sha256_json(payload)[:24]}"


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _claim_protocol_hash(row: Mapping[str, Any]) -> str:
    """Read modern object or legacy string ProtocolRefs without widening."""

    value = row.get("protocol_ref")
    if isinstance(value, Mapping):
        return str(value.get("canonical_hash") or "")
    _prefix, separator, digest = str(value or "").partition("#")
    return digest if separator else ""


def _parent_validation_disposition(
    parent: ImmutableBaseBundle,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    errors = [str(value) for value in report.get("errors") or []]
    prefixes = collections.Counter(value.split(":", 1)[0] for value in errors)
    if report.get("valid") is True and not errors:
        mode = "strict_current_schema"
        accepted = True
        reasons: list[str] = []
    else:
        unexpected = sorted(set(prefixes) - LEGACY_PARENT_LINEAGE_ERROR_PREFIXES)
        invariant_failures = []
        if parent.manifest.get("certification_level") != "raw_audited":
            invariant_failures.append("parent_not_raw_audited")
        if report.get("heldout_reference_count") != 0:
            invariant_failures.append("heldout_reference_count_nonzero")
        if report.get("spooky_node_count") != 0:
            invariant_failures.append("spooky_node_count_nonzero")
        if report.get("all_clause_sources_resolve") is not True:
            invariant_failures.append("clause_sources_unresolved")
        if report.get("source_run_count", 0) <= 0:
            invariant_failures.append("empty_source_population")
        if report.get("code_node_count") != report.get("sidecar_count"):
            invariant_failures.append("code_sidecar_count_mismatch")
        reasons = [
            *[f"unexpected_error_prefix:{value}" for value in unexpected],
            *invariant_failures,
        ]
        accepted = bool(errors) and not reasons
        mode = "legacy_lineage_migration" if accepted else "rejected"
    disposition: dict[str, Any] = {
        "schema": "tier2_formal_parent_validation_disposition_v1",
        "parent_bundle_id": parent.bundle_id,
        "parent_manifest_sha256": parent.manifest_sha256,
        "parent_certification_level": str(
            parent.manifest.get("certification_level") or ""
        ),
        "mode": mode,
        "accepted": accepted,
        "validator_valid": report.get("valid") is True,
        "validator_error_count": len(errors),
        "validator_error_prefix_counts": dict(sorted(prefixes.items())),
        "allowed_legacy_error_prefixes": sorted(LEGACY_PARENT_LINEAGE_ERROR_PREFIXES),
        "reasons": sorted(reasons),
        "disposition_hash": "",
    }
    disposition["disposition_hash"] = payload_hash(disposition, "disposition_hash")
    return disposition


def _receipt_from_row(row: Mapping[str, Any]) -> Receipt:
    values = copy.deepcopy(dict(row))
    values["receipt_type"] = ReceiptType(str(values["receipt_type"]))
    return Receipt(**values)


def _claim_row(claim: Claim) -> dict[str, Any]:
    return _jsonable(claim)


def _receipt_row(receipt: Receipt) -> dict[str, Any]:
    return _jsonable(receipt)


def _path_row(path: EvidencePath) -> dict[str, Any]:
    return _jsonable(path)


def _append_unique(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    identity_key: str,
) -> None:
    existing = read_jsonl(path) if path.exists() else []
    by_id = {str(row.get(identity_key) or ""): dict(row) for row in existing}
    for raw in rows:
        row = _jsonable(raw)
        identity = str(row.get(identity_key) or "")
        if not identity:
            raise ValueError(f"Authority row has no {identity_key}")
        if identity in by_id and by_id[identity] != row:
            raise ValueError(f"Authority {identity_key} is immutable: {identity}")
        by_id[identity] = row
    write_jsonl(path, [by_id[key] for key in sorted(by_id)])


def _copy_declared_authority(parent: ImmutableBaseBundle, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    declared = parent.manifest.get("artifact_hashes") or {}
    for relative in sorted(declared):
        if not str(relative).startswith("authority/"):
            continue
        source = parent.path / str(relative)
        if not source.is_file() or sha256_file(source) != str(declared[relative]):
            raise ValueError(f"Parent Authority artifact drift: {relative}")
        target = destination / Path(str(relative)).relative_to("authority")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in (
        "claims.jsonl",
        "receipts.jsonl",
        "decisions.jsonl",
        "derivations.jsonl",
        "paths.jsonl",
        "replay_paths.jsonl",
        "replay_receipts.jsonl",
    ):
        path = destination / name
        if not path.exists():
            path.write_text("", encoding="utf-8")


def _find_clause(parent: ImmutableBaseBundle, clause_id: str) -> dict[str, Any]:
    matches = [
        row
        for row in parent.read_jsonl("sop/clauses.jsonl")
        if str(row.get("clause_id") or "") == clause_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one source clause {clause_id}, got {len(matches)}")
    return matches[0]


def _find_node(journal: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    matches = [
        node
        for node in journal_nodes(journal)
        if str(node.get("id") or node.get("node_id") or "") == node_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one source journal node {node_id}, got {len(matches)}"
        )
    return matches[0]


def _method_projection(
    source_clause: Mapping[str, Any], method_text: str | None
) -> tuple[str, dict[str, Any]]:
    source_text = " ".join(
        str(source_clause.get(key) or "") for key in ("text", "retrieval_text")
    ).strip()
    projected = str(
        method_text
        if method_text is not None
        else source_clause.get("retrieval_text") or ""
    ).strip()
    purity = require_method_claim_semantic_purity(projected)
    source_tokens = {
        token.lower()
        for token in FORMAL_METHOD_TOKEN_RE.findall(source_text)
        if token.lower() not in FORMAL_METHOD_GLUE_TOKENS
    }
    projected_tokens = {
        token.lower()
        for token in FORMAL_METHOD_TOKEN_RE.findall(projected)
        if token.lower() not in FORMAL_METHOD_GLUE_TOKENS
    }
    novel_tokens = sorted(projected_tokens - source_tokens)
    if novel_tokens:
        raise ValueError(
            "Formal method projection introduces unsupported source tokens: "
            f"{novel_tokens}"
        )
    projection = {
        "schema": "formal_method_lexical_projection_v1",
        "source_clause_id": str(source_clause.get("clause_id") or ""),
        "source_text_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "projected_text_sha256": purity["text_sha256"],
        "projected_content_tokens": sorted(projected_tokens),
        "novel_content_tokens": [],
        "projection_hash": "",
    }
    projection["projection_hash"] = payload_hash(projection, "projection_hash")
    return projected, {**purity, "lexical_projection": projection}


def _supporting_transition_evidence(
    parent: ImmutableBaseBundle,
    *,
    source_run_id: str,
    source_node_id: str,
) -> dict[str, Any]:
    journal_relative = f"raw_journals/{source_run_id}/journal.json"
    if journal_relative not in (parent.manifest.get("artifact_hashes") or {}):
        raise ValueError("Formal method source journal is not manifest-bound")
    journal = parent.read_json(journal_relative)
    indexed = [
        (
            str(node.get("id") or node.get("node_id") or index),
            node,
        )
        for index, node in enumerate(journal_nodes(journal))
    ]
    nodes = {node_id: node for node_id, node in indexed}
    if source_node_id not in nodes:
        raise ValueError("Formal method source node is absent from its journal")
    parents = journal_parent_links(journal, indexed)
    candidate_pairs: list[tuple[str, str]] = []
    if source_node_id in parents:
        candidate_pairs.append((parents[source_node_id], source_node_id))
    candidate_pairs.extend(
        sorted(
            (
                (parent_id, child_id)
                for child_id, parent_id in parents.items()
                if parent_id == source_node_id
            ),
            key=lambda pair: (
                int(nodes[pair[1]].get("step") or 0),
                pair[1],
            ),
        )
    )
    audit_index = parent.read_json("audit_sidecars/index.json").get("entries") or {}
    rejected: list[dict[str, str]] = []
    for parent_id, child_id in candidate_pairs:
        child = nodes[child_id]
        artifact_id = f"run::{source_run_id}::node::{child_id}"
        sidecar_filename = audit_index.get(artifact_id)
        sidecar = (
            parent.read_json(f"audit_sidecars/{sidecar_filename}")
            if sidecar_filename
            else {}
        )
        outcome = transition_outcome(child, nodes[parent_id])
        checks = {
            "audit_clean": sidecar.get("status") == "clean",
            "valid": child.get("is_valid") is True,
            "not_buggy": child.get("is_buggy") is False,
            "positive_or_initial_outcome": outcome in FORMAL_SUPPORT_OUTCOMES,
        }
        if all(checks.values()):
            identifier = transition_ref(source_run_id, parent_id, child_id)
            evidence = {
                "transition_ref": identifier,
                "parent_node_id": parent_id,
                "child_node_id": child_id,
                "outcome": outcome,
                "metric_improvement": None,
                "child_audit_sidecar_sha256": sidecar.get("sidecar_sha256"),
                "checks": checks,
                "evidence_hash": "",
            }
            if outcome == "metric_improved":
                def scalar_metric(node: Mapping[str, Any]) -> float | None:
                    metric = node.get("metric")
                    maximize = True
                    if isinstance(metric, Mapping):
                        maximize = bool(metric.get("maximize", True))
                        metric = metric.get("value")
                    if isinstance(metric, bool) or not isinstance(metric, (int, float)):
                        return None
                    value = float(metric)
                    if not math.isfinite(value):
                        return None
                    return value if maximize else -value

                child_metric = scalar_metric(child)
                parent_metric = scalar_metric(nodes[parent_id])
                if child_metric is None or parent_metric is None:
                    rejected.append(
                        {
                            "transition": f"{parent_id}->{child_id}",
                            "reason": "metric_value_missing_or_invalid",
                        }
                    )
                    continue
                evidence["metric_improvement"] = child_metric - parent_metric
            evidence["evidence_hash"] = payload_hash(evidence, "evidence_hash")
            return evidence
        rejected.append(
            {
                "transition": f"{parent_id}->{child_id}",
                "reason": ",".join(key for key, passed in checks.items() if not passed),
            }
        )
    raise ValueError(
        "Formal method lacks a clean positive/initial supporting transition: "
        f"{rejected}"
    )


def _protocol_ref(
    parent: ImmutableBaseBundle,
    protocol_file: Path,
    destination: Path,
) -> ProtocolRef:
    destination.mkdir(parents=True, exist_ok=False)
    for relative in sorted(parent.manifest.get("artifact_hashes") or {}):
        if not str(relative).startswith("protocol_registry/"):
            continue
        source = parent.path / str(relative)
        target = destination / Path(str(relative)).relative_to("protocol_registry")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    copied = destination / protocol_file.name
    shutil.copy2(protocol_file, copied)
    registry = ProtocolRegistry(destination)
    payload = read_json(copied)
    return registry.get(str(payload["protocol_id"]), str(payload["version"])).ref()


def _filtered_corpus(
    parent_corpus: CorpusManifestV1,
    *,
    retained_run_ids: set[str],
    materialized_source_root: Path,
    materialized_source_run_ids: set[str],
    parent_manifest_sha256: str,
    target_task_id: str,
    target_domain: str,
    created_at: str,
) -> CorpusManifestV1:
    entries = [
        copy.deepcopy(run)
        for run in parent_corpus.runs
        if run.run_id in retained_run_ids
    ]
    if {entry.run_id for entry in entries} != retained_run_ids:
        raise ValueError("Domain child corpus is missing retained split runs")
    if not materialized_source_run_ids <= retained_run_ids:
        raise ValueError("Materialized source runs are outside the child corpus")
    for entry in entries:
        if entry.run_id in materialized_source_run_ids:
            for attribute, filename in CORE_ARTIFACT_NAMES.items():
                value = getattr(entry, attribute)
                setattr(
                    entry,
                    attribute,
                    (f"{entry.run_id}/logs/{filename}" if value else None),
                )
            entry.source_relpath = entry.run_id
        else:
            for attribute in CORE_ARTIFACT_NAMES:
                setattr(entry, attribute, None)
            entry.source_relpath = ""
            entry.warnings = sorted(
                set(
                    [
                        *entry.warnings,
                        "heldout_core_artifacts_intentionally_not_materialized",
                    ]
                )
            )
    actual_snapshot = {
        "derivation": "formal_domain_child_subset_v1",
        "parent_corpus_manifest_sha256": parent_corpus.manifest_sha256,
        "parent_bundle_manifest_sha256": parent_manifest_sha256,
        "target_task_id": target_task_id,
        "target_domain": target_domain,
        "run_count": len(entries),
        "complete_run_count": sum(run.status == "complete" for run in entries),
        "code_node_count": sum(run.code_node_count for run in entries),
        "metric_node_count": sum(run.metric_node_count for run in entries),
        "task_ids": sorted({run.canonical_task_id for run in entries}),
        "task_families": sorted({run.task_family for run in entries}),
        "materialized_source_run_count": len(materialized_source_run_ids),
        "heldout_core_artifacts_materialized": False,
    }
    return CorpusManifestV1(
        corpus_id=(
            f"{parent_corpus.corpus_id}-formal-{target_domain}-{target_task_id}"
        ),
        created_at=created_at,
        source_repo=parent_corpus.source_repo,
        source_commit=parent_corpus.source_commit,
        source_root=str(materialized_source_root.resolve()),
        exclusion_rules=[
            *copy.deepcopy(parent_corpus.exclusion_rules),
            {
                "kind": "formal_domain_child_filter",
                "target_task_id": target_task_id,
                "target_domain": target_domain,
                "retained_run_ids_sha256": sha256_json(sorted(retained_run_ids)),
            },
        ],
        runs=entries,
        expected_snapshot=copy.deepcopy(actual_snapshot),
        actual_snapshot=actual_snapshot,
        split_manifests=[],
    ).finalize()


def _materialize_parent_source_view(
    parent: ImmutableBaseBundle,
    parent_corpus: CorpusManifestV1,
    *,
    source_run_ids: set[str],
    destination: Path,
) -> dict[str, Any]:
    """Rebase child construction onto immutable parent Bundle artifacts.

    Historical corpus roots may no longer exist on a later devpod.  A formal
    child must therefore be reproducible from the parent Bundle alone and must
    never silently fall back to a mutable external source path.
    """

    destination.mkdir(parents=True, exist_ok=False)
    entries = {entry.run_id: entry for entry in parent_corpus.runs}
    missing = source_run_ids - set(entries)
    if missing:
        raise ValueError(f"Parent source view has unknown runs: {sorted(missing)}")
    declared = parent.manifest.get("artifact_hashes") or {}
    copied: dict[str, str] = {}
    for run_id in sorted(source_run_ids):
        entry = entries[run_id]
        run_root = destination / run_id / "logs"
        run_root.mkdir(parents=True)
        for attribute, filename in CORE_ARTIFACT_NAMES.items():
            if not getattr(entry, attribute):
                continue
            relative = f"raw_journals/{run_id}/{filename}"
            source = parent.path / relative
            if relative not in declared:
                raise ValueError(
                    f"Parent source artifact is not manifest-bound: {relative}"
                )
            if not source.is_file() or sha256_file(source) != str(declared[relative]):
                raise ValueError(f"Parent source artifact drift: {relative}")
            expected = entry.artifact_hashes.get(attribute.replace("_path", ""))
            if expected and expected != sha256_file(source):
                raise ValueError(
                    f"Parent corpus/source artifact hash mismatch: {relative}"
                )
            target = run_root / filename
            shutil.copy2(source, target)
            copied[target.relative_to(destination).as_posix()] = sha256_file(target)
    return {
        "schema": "tier2_formal_parent_source_view_v1",
        "parent_bundle_id": parent.bundle_id,
        "parent_manifest_sha256": parent.manifest_sha256,
        "source_run_ids": sorted(source_run_ids),
        "artifact_hashes": dict(sorted(copied.items())),
        "source_view_hash": sha256_json(dict(sorted(copied.items()))),
    }


def _child_split(
    parent_split: SplitManifestV1,
    corpus: CorpusManifestV1,
    *,
    target_task_id: str,
    target_task_family: str,
    target_domain: str,
    split_mode: str,
    agent_seeds: tuple[int, ...],
    created_at: str,
) -> SplitManifestV1:
    entries = {run.run_id: run for run in corpus.runs}
    if split_mode == "same-domain-task-heldout":
        # ``corpus`` is the already filtered, immutable same-domain source
        # view.  Derive the child split from that view instead of inheriting
        # the parent's unrelated domains or requiring the parent to have
        # materialized target-task history.  The target is represented only
        # by its held-out task identity; no target run or target seed is
        # admitted to the Base Bundle.
        source_runs = [
            run_id for run_id in parent_split.source_run_ids if run_id in entries
        ]
        heldout_runs = []
        source_tasks = sorted(
            {entries[run_id].canonical_task_id for run_id in source_runs}
        )
        heldout_tasks = [target_task_id]
        if not source_runs:
            raise ValueError("Task-heldout child requires same-domain source runs")
        if target_task_id in source_tasks:
            raise ValueError("Task-heldout child contains target-task history")
        source_domains = {
            canonical_domain(entries[run_id].task_family) for run_id in source_runs
        }
        source_domains.discard("")
        if source_domains != {target_domain}:
            raise ValueError("Task-heldout child source domain mismatch")
    elif split_mode == "same-domain-seed-heldout":
        source_runs = [
            run_id
            for run_id in parent_split.source_run_ids
            if entries.get(run_id) is not None
            and canonical_domain(entries[run_id].task_family) == target_domain
        ]
        heldout_runs = [
            run_id
            for run_id in parent_split.heldout_run_ids
            if entries.get(run_id) is not None
            and canonical_domain(entries[run_id].task_family) == target_domain
        ]
        source_tasks = sorted(
            {entries[run_id].canonical_task_id for run_id in source_runs}
        )
        heldout_tasks = sorted(
            {entries[run_id].canonical_task_id for run_id in heldout_runs}
        )
        if not source_runs or target_task_id not in source_tasks:
            raise ValueError(
                "Same-domain seed-heldout child requires target-task source history"
            )
        if not heldout_runs:
            raise ValueError(
                "Same-domain seed-heldout child requires held-out domain runs"
            )
    elif split_mode == "seed-heldout":
        source_runs = [
            run_id
            for run_id in parent_split.source_run_ids
            if entries.get(run_id) is not None
            and entries[run_id].canonical_task_id == target_task_id
        ]
        heldout_runs = [
            run_id
            for run_id in parent_split.heldout_run_ids
            if entries.get(run_id) is not None
            and entries[run_id].canonical_task_id == target_task_id
        ]
        source_tasks = [target_task_id]
        heldout_tasks = [target_task_id]
        if not source_runs or not heldout_runs:
            raise ValueError("Seed-heldout child requires source and heldout runs")
    else:
        raise ValueError(
            "Formal child split_mode must be seed-heldout, "
            "same-domain-seed-heldout, or same-domain-task-heldout"
        )
    source_seed_groups = [
        f"{entries[run_id].canonical_task_id}::{entries[run_id].seed}"
        for run_id in source_runs
    ]
    heldout_seed_groups = [
        f"{entries[run_id].canonical_task_id}::{entries[run_id].seed}"
        for run_id in heldout_runs
    ]
    if set(source_runs) & set(heldout_runs):
        raise ValueError("Formal child source/heldout runs overlap")
    if split_mode in {"seed-heldout", "same-domain-seed-heldout"} and set(source_seed_groups) & set(
        heldout_seed_groups
    ):
        raise ValueError("Formal child source/heldout seed groups overlap")
    source_seed_values = {str(entries[run_id].seed) for run_id in source_runs}
    if source_seed_values & {str(value) for value in agent_seeds}:
        raise ValueError("Formal agent seed appears in source memory")
    retained = set(source_runs) | set(heldout_runs)
    excluded = sorted(
        (
            set(parent_split.source_run_ids)
            | set(parent_split.heldout_run_ids)
            | set(parent_split.excluded_run_ids)
        )
        - retained
    )
    split_id = f"wp8-tier2-formal-{target_domain}-{target_task_id}-{split_mode}-v1"
    return SplitManifestV1(
        split_id=split_id,
        split_kind=split_mode,
        split_version="wp8-tier2-formal-child-v1",
        corpus_manifest_hash=corpus.manifest_sha256,
        created_at=created_at,
        source_run_ids=sorted(source_runs),
        heldout_run_ids=sorted(heldout_runs),
        source_task_ids=sorted(source_tasks),
        heldout_task_ids=sorted(heldout_tasks),
        source_seed_groups=sorted(set(source_seed_groups)),
        heldout_seed_groups=sorted(set(heldout_seed_groups)),
        excluded_run_ids=excluded,
        allocation={
            "derivation": "formal_domain_child_split_v1",
            "parent_split_id": parent_split.split_id,
            "parent_split_manifest_sha256": parent_split.manifest_sha256,
            "target_task_id": target_task_id,
            "target_task_family": target_task_family,
            "target_domain": target_domain,
            "transfer_design": (
                "same_domain_different_task_target_history_absent"
                if split_mode == "same-domain-task-heldout"
                else "same_domain_mixed_task_different_seed"
                if split_mode == "same-domain-seed-heldout"
                else "same_task_different_seed"
            ),
            "formal_agent_seeds": list(agent_seeds),
        },
        validation={
            "run_overlap": [],
            "run_overlap_count": 0,
            "seed_group_overlap": [],
            "seed_group_overlap_count": 0,
            "cross_domain_source_run_count": 0,
            "target_history_in_source_count": (
                0
                if split_mode == "same-domain-task-heldout"
                else sum(
                    entries[run_id].canonical_task_id == target_task_id
                    for run_id in source_runs
                )
            ),
            "formal_agent_seed_overlap_count": 0,
        },
    ).finalize()


def _task_heldout_source_run_ids(
    parent_split: SplitManifestV1,
    parent_entries: Mapping[str, Any],
    *,
    target_task_id: str,
    target_domain: str,
    agent_seeds: Iterable[int],
) -> set[str]:
    """Select same-domain, cross-task memory without formal-seed overlap."""

    formal_seeds = {str(int(value)) for value in agent_seeds}
    return {
        run_id
        for run_id in parent_split.source_run_ids
        if parent_entries.get(run_id) is not None
        and canonical_domain(parent_entries[run_id].task_family) == target_domain
        and parent_entries[run_id].canonical_task_id != target_task_id
        and str(parent_entries[run_id].seed) not in formal_seeds
    }


def _same_domain_seed_source_run_ids(
    parent_split: SplitManifestV1,
    parent_entries: Mapping[str, Any],
    *,
    target_domain: str,
    agent_seeds: Iterable[int],
) -> set[str]:
    """Select all same-domain memory-side runs except the frozen Agent seeds."""

    formal_seeds = {str(int(value)) for value in agent_seeds}
    return {
        run_id
        for run_id in parent_split.source_run_ids
        if parent_entries.get(run_id) is not None
        and canonical_domain(parent_entries[run_id].task_family) == target_domain
        and str(parent_entries[run_id].seed) not in formal_seeds
    }


def _certified_source_evidence(
    parent: ImmutableBaseBundle,
    source_clause: Mapping[str, Any],
) -> dict[str, Any]:
    if source_clause.get("publication_class") != "certified":
        raise ValueError(
            "Certified formal publication requires a certified parent clause"
        )
    claim_refs = [str(value) for value in source_clause.get("claim_refs") or []]
    if len(claim_refs) != 1:
        raise ValueError("Certified parent clause must bind exactly one Claim")
    claim_id = claim_refs[0]
    claims = {
        str(row["claim_id"]): row for row in parent.read_jsonl("authority/claims.jsonl")
    }
    claim = claims.get(claim_id)
    if claim is None or claim.get("claim_type") != ClaimType.METHOD_HYPOTHESIS.value:
        raise ValueError("Certified parent method Claim is unavailable")
    receipts = {
        str(row["receipt_id"]): row
        for row in parent.read_jsonl("authority/receipts.jsonl")
    }
    declared = {str(value) for value in source_clause.get("receipt_refs") or []}
    selected = [receipts[value] for value in sorted(declared) if value in receipts]
    if len(selected) != len(declared) or not selected:
        raise ValueError("Certified parent clause has unresolved Receipts")
    typed: dict[ReceiptType, list[Receipt]] = {}
    for row in selected:
        receipt = _receipt_from_row(row)
        verify_trusted_receipt_integrity(receipt)
        if receipt.artifact_id != claim.get("subject_artifact_id"):
            raise ValueError("Certified parent Receipt artifact mismatch")
        typed.setdefault(receipt.receipt_type, []).append(receipt)
    if not {ReceiptType.METHOD_IDENTITY, ReceiptType.CODE_EXECUTION} <= set(typed):
        raise ValueError("Certified parent path lacks method/execution evidence")
    path_rows = []
    for relative in ("authority/paths.jsonl", "authority/replay_paths.jsonl"):
        if relative in (parent.manifest.get("artifact_hashes") or {}):
            path_rows.extend(parent.read_jsonl(relative))
    matching_paths = [
        row
        for row in path_rows
        if row.get("claim_id") == claim_id
        and set(map(str, row.get("receipt_ids") or [])) == declared
    ]
    if not matching_paths:
        raise ValueError("Certified parent method has no exact EvidencePath")
    method_receipt = typed[ReceiptType.METHOD_IDENTITY][0]
    execution_receipt = typed[ReceiptType.CODE_EXECUTION][0]
    return {
        "source_claim_id": claim_id,
        "subject_artifact_id": str(claim["subject_artifact_id"]),
        "method_fingerprint": str(claim.get("method_fingerprint") or ""),
        "code_sha256": str(method_receipt.payload.get("code_sha256") or ""),
        "run_hash": str(execution_receipt.payload.get("run_hash") or ""),
        "executed_path": str(execution_receipt.payload.get("executed_path") or ""),
        "source_receipt_ids": sorted(declared),
        "source_path_ids": sorted(str(row["path_id"]) for row in matching_paths),
        "source_audit_status": "certified_clean_replay",
        "source_audit_issue_codes": [],
        "source_journal_sha256": "",
        "source_sidecar_sha256": "",
        "source_execution_evidence_hash": runtime_sha256_json(
            {
                "claim_id": claim_id,
                "receipt_ids": sorted(declared),
                "path_ids": sorted(str(row["path_id"]) for row in matching_paths),
            }
        ),
    }


def _provisional_source_evidence(
    parent: ImmutableBaseBundle,
    *,
    source_run_id: str,
    source_node_id: str,
    allowed_protocol_issue_codes: set[str],
) -> dict[str, Any]:
    artifact_id = f"run::{source_run_id}::node::{source_node_id}"
    split = parent.read_json("splits/active.json")
    if source_run_id not in set(split.get("source_run_ids") or []):
        raise ValueError(
            "Provisional source run is outside the memory side of the split"
        )
    journal_relative = f"raw_journals/{source_run_id}/journal.json"
    journal_path = parent.path / journal_relative
    if journal_relative not in (parent.manifest.get("artifact_hashes") or {}):
        raise ValueError("Provisional source journal is not manifest-bound")
    journal = parent.read_json(journal_relative)
    node = _find_node(journal, source_node_id)
    code = str(node.get("code") or "")
    if not code.strip():
        raise ValueError("Provisional source node has no code")
    code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    graph = parent.read_json("runforest/graph.json")
    graph_nodes = {
        str(value.get("id") or ""): value for value in graph.get("nodes") or []
    }
    graph_node = graph_nodes.get(artifact_id)
    if graph_node is None or graph_node.get("code_sha256") != code_sha256:
        raise ValueError("Provisional source code does not match RunForest")
    audit_index = parent.read_json("audit_sidecars/index.json")
    filename = (audit_index.get("entries") or {}).get(artifact_id)
    if not filename:
        raise ValueError("Provisional source node has no audit sidecar")
    sidecar = parent.read_json(f"audit_sidecars/{filename}")
    if sidecar.get("sidecar_sha256") != payload_hash(sidecar, "sidecar_sha256"):
        raise ValueError(
            "Provisional source audit sidecar has an invalid internal hash"
        )
    if sidecar.get("artifact_id") != artifact_id:
        raise ValueError("Provisional source audit sidecar artifact mismatch")
    if sidecar.get("run_id") != source_run_id:
        raise ValueError("Provisional source audit sidecar run mismatch")
    if sidecar.get("code_sha256") != code_sha256:
        raise ValueError("Provisional source code does not match audit sidecar")
    if sidecar.get("source_journal_sha256") != sha256_file(journal_path):
        raise ValueError("Provisional audit sidecar does not bind the source journal")
    issue_codes = sorted(
        {
            str(issue.get("issue_code") or "")
            for issue in sidecar.get("issues") or []
            if str(issue.get("issue_code") or "")
        }
    )
    issue_set = set(issue_codes)
    unsupported_exceptions = sorted(
        allowed_protocol_issue_codes - PROVISIONAL_METHOD_ONLY_PROTOCOL_ISSUES
    )
    if unsupported_exceptions:
        raise ValueError(
            "Provisional method requested unsupported audit exceptions: "
            f"{unsupported_exceptions}"
        )
    unused_exceptions = sorted(allowed_protocol_issue_codes - issue_set)
    if unused_exceptions:
        raise ValueError(
            "Provisional method requested audit exceptions absent from the "
            f"source sidecar: {unused_exceptions}"
        )
    method_fatal = sorted(issue_set - allowed_protocol_issue_codes)
    if method_fatal:
        raise ValueError(
            f"Provisional method has non-protocol audit blockers: {method_fatal}"
        )
    expected_status = "blocked" if issue_codes else "clean"
    if str(sidecar.get("status") or "") != expected_status:
        raise ValueError(
            "Provisional source audit status is inconsistent with its issue set"
        )
    terminal_output = "".join(
        str(value)
        for value in (
            node.get("_term_out")
            if isinstance(node.get("_term_out"), list)
            else [node.get("_term_out") or ""]
        )
    )
    execution_checks = {
        "is_valid": node.get("is_valid") is True,
        "not_buggy": node.get("is_buggy") is False,
        "no_exception_type": node.get("exc_type") in {None, ""},
        "positive_exec_time": isinstance(node.get("exec_time"), (int, float))
        and not isinstance(node.get("exec_time"), bool)
        and math.isfinite(float(node["exec_time"]))
        and float(node["exec_time"]) > 0,
        "terminal_execution_signal": "Execution time:" in terminal_output,
        "submission_signal": "Submission saved" in terminal_output,
    }
    if not all(execution_checks.values()):
        raise ValueError(
            f"Provisional source lacks execution evidence: {execution_checks}"
        )
    execution_evidence = {
        "artifact_id": artifact_id,
        "journal_sha256": sha256_file(journal_path),
        "node_id": source_node_id,
        "code_sha256": code_sha256,
        "terminal_output_sha256": hashlib.sha256(
            terminal_output.encode("utf-8")
        ).hexdigest(),
        "exec_time": float(node["exec_time"]),
        "checks": execution_checks,
    }
    return {
        "source_claim_id": "",
        "subject_artifact_id": artifact_id,
        "method_fingerprint": method_fingerprint(code),
        "code_sha256": code_sha256,
        "run_hash": runtime_sha256_json(execution_evidence),
        "executed_path": artifact_id,
        "source_receipt_ids": [],
        "source_path_ids": [],
        "source_audit_status": str(sidecar.get("status") or ""),
        "source_audit_issue_codes": issue_codes,
        "source_journal_sha256": sha256_file(journal_path),
        "source_sidecar_sha256": str(sidecar.get("sidecar_sha256") or ""),
        "source_execution_evidence_hash": runtime_sha256_json(execution_evidence),
    }


def _formal_method_records(
    *,
    parent: ImmutableBaseBundle,
    source_clause: Mapping[str, Any],
    source_run_id: str,
    source_node_id: str,
    source_task_id: str,
    source_task_family: str,
    target_task_id: str,
    target_task_family: str,
    target_domain: str,
    protocol_ref: ProtocolRef,
    publication_class: str,
    allowed_protocol_issue_codes: set[str],
    method_text: str | None = None,
    collector_host: TrustedCollectorHost | None = None,
    receipt_cache: dict[tuple[str, str, str, str, str], Receipt] | None = None,
) -> dict[str, Any]:
    if publication_class == "certified":
        source = _certified_source_evidence(parent, source_clause)
    elif publication_class == "provisional":
        source = _provisional_source_evidence(
            parent,
            source_run_id=source_run_id,
            source_node_id=source_node_id,
            allowed_protocol_issue_codes=allowed_protocol_issue_codes,
        )
        source["source_claim_id"] = str((source_clause.get("claim_refs") or [""])[0])
    else:
        raise ValueError("Formal method publication must be certified or provisional")
    for key in ("method_fingerprint", "code_sha256", "run_hash"):
        if not _is_sha256(source[key]):
            raise ValueError(f"Formal method source has invalid {key}")
    method_text, purity = _method_projection(source_clause, method_text)
    support = _supporting_transition_evidence(
        parent,
        source_run_id=source_run_id,
        source_node_id=source_node_id,
    )
    identity = {
        "parent_bundle_id": parent.bundle_id,
        "parent_manifest_sha256": parent.manifest_sha256,
        "source_clause_id": str(source_clause["clause_id"]),
        "source_run_id": source_run_id,
        "source_node_id": source_node_id,
        "target_task_id": target_task_id,
        "target_domain": target_domain,
        "protocol_ref": protocol_ref.key(),
        "publication_class": publication_class,
        "method_text_sha256": purity["text_sha256"],
        "supporting_transition_ref": support["transition_ref"],
    }
    claim_id = _stable_id("claim::formal-method::", identity)
    generation_clause_id = _stable_id(
        "clause::formal-method-generate::", {**identity, "scope": "generate"}
    )
    debug_clause_id = _stable_id(
        "clause::formal-method-debug::", {**identity, "scope": "debug"}
    )
    path_id = _stable_id("path::formal-method::", identity)
    generation_derivation_id = _stable_id(
        "derivation::formal-method-generate::", {**identity, "scope": "generate"}
    )
    debug_derivation_id = _stable_id(
        "derivation::formal-method-debug::", {**identity, "scope": "debug"}
    )
    sop_id = _stable_id("sop::formal-method::", identity)
    host = collector_host or TrustedCollectorHost(
        f"wp8-tier2-formal-child::{target_task_id}", collector_version="1"
    )
    cache = receipt_cache if receipt_cache is not None else {}

    def cached_receipt(
        collector_type: type[MethodIdentityCollector] | type[CodeExecutionCollector],
        *,
        source_name: str,
        payload: dict[str, Any],
    ) -> Receipt:
        cache_key = (
            collector_type.receipt_type.value,
            str(source["subject_artifact_id"]),
            source_run_id,
            protocol_ref.key(),
            runtime_sha256_json(payload),
        )
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
        receipt = host.collect(
            collector_type,
            artifact_id=source["subject_artifact_id"],
            run_id=source_run_id,
            protocol_ref=protocol_ref,
            source=source_name,
            payload=payload,
        )
        cache[cache_key] = receipt
        return receipt

    method_receipt = cached_receipt(
        MethodIdentityCollector,
        source_name="host.formal_child.method_identity",
        payload={
            "method_fingerprint": source["method_fingerprint"],
            "code_sha256": source["code_sha256"],
        },
    )
    execution_receipt = cached_receipt(
        CodeExecutionCollector,
        source_name="host.formal_child.code_execution",
        payload={
            "exit_status": 0,
            "executed_path": source["executed_path"],
            "run_hash": source["run_hash"],
        },
    )
    receipt_ids = [method_receipt.receipt_id, execution_receipt.receipt_id]
    claim = Claim(
        claim_id=claim_id,
        claim_type=ClaimType.METHOD_HYPOTHESIS,
        subject_artifact_id=source["subject_artifact_id"],
        task_scope={
            "task_ids": [source_task_id],
            "task_families": [
                target_task_family
                if source_task_id == target_task_id
                else source_task_family
            ],
        },
        method_fingerprint=source["method_fingerprint"],
        protocol_ref=protocol_ref,
        statement=method_text,
        parent_claims=(
            [source["source_claim_id"]] if source["source_claim_id"] else []
        ),
        source_artifact_refs=[
            f"run::{source_run_id}::node::{source_node_id}",
            f"bundle::{parent.bundle_id}::{parent.manifest_sha256}",
        ],
        evidence_refs=[
            source["source_execution_evidence_hash"],
            purity["report_hash"],
        ],
        boundary={
            "formal_method_publication": True,
            "publication_class": publication_class,
            "same_domain_only": True,
            "source_score_inheritance": False,
            "historical_metric_used_as_evidence": False,
            "permitted_operations": [
                Operation.GENERATE_CANDIDATE.value,
                Operation.DEBUG_HYPOTHESIS.value,
            ],
            "source_audit_status": source["source_audit_status"],
            "source_audit_issue_codes": source["source_audit_issue_codes"],
            "allowed_protocol_issue_codes": sorted(allowed_protocol_issue_codes),
            "method_semantic_purity_report_hash": purity["report_hash"],
            "method_lexical_projection_hash": purity["lexical_projection"][
                "projection_hash"
            ],
            "supporting_transition_evidence_hash": support["evidence_hash"],
            "parent_bundle_manifest_sha256": parent.manifest_sha256,
        },
        legacy_status=f"formal_{publication_class}_method_v1",
    )
    source_artifact = f"run::{source_run_id}::node::{source_node_id}"
    contract_spec = {
            "source_score_inheritance": False,
            "historical_metric_used_as_evidence": False,
            "source_clause_id": str(source_clause["clause_id"]),
            "source_claim_id": source["source_claim_id"],
            "source_execution_evidence_hash": source["source_execution_evidence_hash"],
            "source_journal_sha256": source["source_journal_sha256"],
            "source_sidecar_sha256": source["source_sidecar_sha256"],
            "source_receipt_ids": source["source_receipt_ids"],
            "source_path_ids": source["source_path_ids"],
            "method_semantic_purity_report_hash": purity["report_hash"],
            "method_lexical_projection": purity["lexical_projection"],
            "supporting_transition": support,
            "parent_bundle_id": parent.bundle_id,
            "parent_manifest_sha256": parent.manifest_sha256,
    }

    def clause_record(
        *,
        clause_id: str,
        operation: Operation,
        generation_stages: Iterable[str],
        derivation_id: str,
        scope: str,
    ) -> dict[str, Any]:
        return {
            "clause_id": clause_id,
            "sop_id": sop_id,
            "text": method_text,
            "retrieval_text": method_text,
            "claim_refs": [claim_id],
            "claim_types": [ClaimType.METHOD_HYPOTHESIS.value],
            "source_artifact_refs": [source_artifact],
            "source_transition_refs": [support["transition_ref"]],
            "source_run_ids": [source_run_id],
            "source_task_ids": [source_task_id],
            "source_task_families": [source_task_family],
            "source_domains": [target_domain],
            "transfer_scope": "same_domain",
            "admissible_target_domains": [target_domain],
            "protocol_scope": [protocol_ref.key()],
            "task_scope": {
                "task_ids": [source_task_id],
                "task_families": [
                    target_task_family
                    if source_task_id == target_task_id
                    else source_task_family
                ],
            },
            "permitted_operations": [operation.value],
            "permitted_generation_stages": list(generation_stages),
            "permitted_governance_stages": [GovernanceStage.RETRIEVAL.value],
            "publication_class": publication_class,
            "authority_decision_refs": [],
            "receipt_refs": receipt_ids,
            "derivation_refs": [derivation_id],
            "protocol_agnostic": False,
            "legacy_status": f"formal_{publication_class}_method_v2",
            "publication_origin": "tier2_formal_child_publisher_v2",
            "formal_scope": scope,
            "applies_when": [
                f"{scope} for {target_task_id} under the frozen formal protocol"
            ],
            "prevents": [
                "cross-domain method transfer",
                "source score inheritance",
            ],
            "contract_spec": copy.deepcopy(contract_spec),
        }

    generation_clause = clause_record(
        clause_id=generation_clause_id,
        operation=Operation.GENERATE_CANDIDATE,
        generation_stages=FORMAL_GENERATION_STAGES,
        derivation_id=generation_derivation_id,
        scope="candidate generation",
    )
    debug_clause = clause_record(
        clause_id=debug_clause_id,
        operation=Operation.DEBUG_HYPOTHESIS,
        generation_stages=(GenerationStage.DEBUG.value,),
        derivation_id=debug_derivation_id,
        scope="debug hypothesis generation",
    )
    clauses = [generation_clause, debug_clause]
    container = {
        "sop_id": sop_id,
        "title": f"Formal {publication_class} method for {target_task_id}",
        "task_id": source_task_id,
        "clause_ids": [generation_clause_id, debug_clause_id],
        "source_run_ids": [source_run_id],
        "source_task_ids": [source_task_id],
        "source_task_families": [source_task_family],
        "source_domains": [target_domain],
        "transfer_scopes": ["same_domain"],
        "domain_scope_complete": True,
    }
    derivations = []
    for clause, derivation_id in (
        (generation_clause, generation_derivation_id),
        (debug_clause, debug_derivation_id),
    ):
        derivation = {
            "schema": DERIVATION_SCHEMA,
            "derivation_id": derivation_id,
            "clause_id": clause["clause_id"],
            "claim_id": claim_id,
            "parent_claim_id": source["source_claim_id"],
            "parent_clause_id": str(source_clause["clause_id"]),
            "parent_bundle_id": parent.bundle_id,
            "parent_manifest_sha256": parent.manifest_sha256,
            "source_run_id": source_run_id,
            "source_node_id": source_node_id,
            "source_transition_ref": support["transition_ref"],
            "source_journal_sha256": source["source_journal_sha256"],
            "source_sidecar_sha256": source["source_sidecar_sha256"],
            "source_audit_status": source["source_audit_status"],
            "source_audit_issue_codes": source["source_audit_issue_codes"],
            "allowed_protocol_issue_codes": sorted(allowed_protocol_issue_codes),
            "method_semantic_purity_report": purity,
            "source_score_inheritance": False,
            "historical_metric_used_as_evidence": False,
            "receipt_ids": receipt_ids,
            "path_id": path_id,
            "derivation_hash": "",
        }
        derivation["derivation_hash"] = payload_hash(
            derivation, "derivation_hash"
        )
        derivations.append(derivation)
    return {
        "claim": claim,
        "receipts": [method_receipt, execution_receipt],
        "path": EvidencePath(
            path_id=path_id,
            claim_id=claim_id,
            receipt_ids=receipt_ids,
        ),
        "clause": generation_clause,
        "debug_clause": debug_clause,
        "clauses": clauses,
        "container": container,
        "derivation": derivations[0],
        "derivations": derivations,
        "source": source,
        "purity": purity,
        "support": support,
    }


def verify_formal_child_bundle(
    publication_root: str | Path,
    *,
    expected_parent_bundle_id: str,
    expected_parent_manifest_sha256: str,
    target_task_id: str,
    target_task_family: str,
    target_domain: str,
    formal_clause_id: str,
    formal_claim_id: str,
    protocol_ref: ProtocolRef,
    split_mode: str,
    agent_seeds: tuple[int, ...],
    formal_methods: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    publication_root = Path(publication_root).resolve()
    errors: list[str] = []

    def require(condition: object, code: str) -> None:
        if not bool(condition):
            errors.append(code)

    try:
        loader = MemorySnapshotLoader(publication_root)
        base = loader.load_base(verify_artifacts=True)
        generic = validate_bundle(base.path)
        require(generic.get("valid") is True, "generic_bundle_validation")
        require(
            base.manifest.get("parent_bundle") == expected_parent_bundle_id,
            "parent_bundle",
        )
        report_path = publication_root / "reports" / "publication_report.json"
        publication = read_json(report_path)
        require(
            publication.get("parent_manifest_sha256")
            == expected_parent_manifest_sha256,
            "parent_manifest",
        )
        split = base.read_json("splits/active.json")
        require(split.get("split_kind") == split_mode, "split_mode")
        require(
            not set(split.get("source_run_ids") or [])
            & set(split.get("heldout_run_ids") or []),
            "run_overlap",
        )
        require(
            not set(split.get("source_seed_groups") or [])
            & set(split.get("heldout_seed_groups") or []),
            "seed_overlap",
        )
        source_seeds = {
            str(value).rsplit("::", 1)[-1]
            for value in split.get("source_seed_groups") or []
        }
        require(
            not source_seeds & {str(value) for value in agent_seeds},
            "formal_agent_seed_overlap",
        )
        if split_mode == "same-domain-task-heldout":
            require(
                target_task_id not in set(split.get("source_task_ids") or []),
                "target_history",
            )
            require(split.get("heldout_task_ids") == [target_task_id], "heldout_target")
        elif split_mode == "same-domain-seed-heldout":
            require(
                target_task_id in set(split.get("source_task_ids") or []),
                "target_history_missing",
            )
            require(
                int((split.get("validation") or {}).get(
                    "target_history_in_source_count", 0
                )) > 0,
                "target_history_count",
            )
        graph = base.read_json("runforest/graph.json")
        run_nodes = [
            row
            for row in graph.get("nodes") or []
            if row.get("type") in {"Run", "RunNode"}
        ]
        require(
            all(
                str(row.get("task_domain") or "") == target_domain for row in run_nodes
            ),
            "runforest_domain",
        )
        clauses = base.read_jsonl("sop/clauses.jsonl")
        by_id = {str(row["clause_id"]): row for row in clauses}
        method_rows = [dict(row) for row in formal_methods]
        if not method_rows:
            method_rows = [
                {
                    "formal_clause_ids": [formal_clause_id],
                    "formal_claim_id": formal_claim_id,
                    "formal_sop_id": str(
                        (by_id.get(formal_clause_id) or {}).get("sop_id") or ""
                    ),
                }
            ]
        expected_formal_clause_ids = {
            str(clause_id)
            for method in method_rows
            for clause_id in method.get("formal_clause_ids") or []
        }
        expected_formal_claim_ids = {
            str(method.get("formal_claim_id") or "") for method in method_rows
        }
        expected_formal_sop_ids = {
            str(method.get("formal_sop_id") or "") for method in method_rows
        }
        require(
            all(clause_id in by_id for clause_id in expected_formal_clause_ids),
            "formal_clauses_present",
        )
        graph_nodes = {
            str(row.get("id") or ""): row for row in graph.get("nodes") or []
        }
        graph_edges = [dict(row) for row in graph.get("edges") or []]
        for method in method_rows:
            method_claim_id = str(method.get("formal_claim_id") or "")
            method_sop_id = str(method.get("formal_sop_id") or "")
            method_clause_ids = [
                str(value) for value in method.get("formal_clause_ids") or []
            ]
            method_clauses = [by_id.get(value) or {} for value in method_clause_ids]
            generate = [
                row
                for row in method_clauses
                if row.get("permitted_operations")
                == [Operation.GENERATE_CANDIDATE.value]
            ]
            debug = [
                row
                for row in method_clauses
                if row.get("permitted_operations")
                == [Operation.DEBUG_HYPOTHESIS.value]
            ]
            require(len(generate) == 1, f"formal_generate_clause:{method_sop_id}")
            require(len(debug) == 1, f"formal_debug_clause:{method_sop_id}")
            for clause in method_clauses:
                require(
                    clause.get("claim_refs") == [method_claim_id],
                    f"formal_claim_binding:{clause.get('clause_id')}",
                )
                require(
                    clause.get("protocol_scope") == [protocol_ref.key()],
                    f"formal_protocol_binding:{clause.get('clause_id')}",
                )
                require(
                    clause.get("source_domains") == [target_domain],
                    f"formal_domain_binding:{clause.get('clause_id')}",
                )
                require(
                    clause.get("transfer_scope") == "same_domain",
                    f"formal_transfer_scope:{clause.get('clause_id')}",
                )
                require(
                    len(clause.get("source_transition_refs") or []) == 1,
                    f"formal_transition_binding:{clause.get('clause_id')}",
                )
                require(
                    (clause.get("contract_spec") or {}).get(
                        "source_score_inheritance"
                    )
                    is False,
                    f"source_score_inheritance:{clause.get('clause_id')}",
                )
            if generate:
                require(
                    tuple(generate[0].get("permitted_generation_stages") or ())
                    == FORMAL_GENERATION_STAGES,
                    f"formal_generation_stages:{method_sop_id}",
                )
            if debug:
                require(
                    debug[0].get("permitted_generation_stages")
                    == [GenerationStage.DEBUG.value],
                    f"formal_debug_stage:{method_sop_id}",
                )
            support_ref = str(method.get("supporting_transition_ref") or "")
            require(
                graph_nodes.get(support_ref, {}).get("type") == "Transition",
                f"formal_transition_materialized:{method_sop_id}",
            )
            require(
                graph_nodes.get(support_ref, {}).get("outcome")
                in FORMAL_SUPPORT_OUTCOMES,
                f"formal_transition_outcome:{method_sop_id}",
            )
            require(
                any(
                    edge.get("src") == support_ref
                    and edge.get("dst") == method_sop_id
                    and edge.get("kind") == "navigation_attached_to"
                    and bool(
                        set(map(str, edge.get("clause_ids") or []))
                        & set(method_clause_ids)
                    )
                    for edge in graph_edges
                ),
                f"formal_clause_scoped_edge:{method_sop_id}",
            )
        formal_protocol_clauses = [
            row
            for row in clauses
            if protocol_ref.key() in set(row.get("protocol_scope") or [])
        ]
        require(
            {str(row["clause_id"]) for row in formal_protocol_clauses}
            == expected_formal_clause_ids,
            "formal_protocol_clause_universe",
        )
        require(len(clauses) > 1, "flat_raw_clause_population")
        require(
            all(
                set(row.get("source_domains") or []) in ({target_domain}, set())
                for row in clauses
            ),
            "clause_domain_universe",
        )
        authority_claims = base.read_jsonl("authority/claims.jsonl")
        formal_claims = [
            row
            for row in authority_claims
            if _claim_protocol_hash(row) == protocol_ref.canonical_hash
        ]
        require(
            {str(row.get("claim_id") or "") for row in formal_claims}
            == expected_formal_claim_ids
            and all(
                row.get("claim_type") == "method_hypothesis"
                for row in formal_claims
            ),
            "formal_claim_universe",
        )
        require(
            all(
                (row.get("boundary") or {}).get("source_score_inheritance")
                is False
                for row in formal_claims
            ),
            "formal_claim_score_boundary",
        )
        snapshot = loader.load(
            session_overlay_path=publication_root / "verification-overlay",
            active_protocol_ref=protocol_ref.key(),
            authority_policy_version="authority_v1",
            verify_artifacts=True,
        )
        registry = ProtocolRegistry(base.path / "protocol_registry")
        engine = AuthorityEngine(registry, policy_version="authority_v1")
        authority_load = load_snapshot_authority(engine, snapshot)
        require(
            expected_formal_claim_ids <= set(authority_load["claim_ids"]),
            "formal_claims_loaded",
        )
        clause_fields = {field.name for field in dataclasses.fields(SOPClauseV1)}
        for operation, stage in (
            *(
                (Operation.GENERATE_CANDIDATE, GenerationStage(value))
                for value in FORMAL_GENERATION_STAGES
            ),
            (Operation.DEBUG_HYPOTHESIS, GenerationStage.DEBUG),
        ):
            scoped = [
                row
                for row in formal_protocol_clauses
                if row.get("permitted_operations") == [operation.value]
                and stage.value in set(row.get("permitted_generation_stages") or [])
            ]
            expected_ids = {str(row["clause_id"]) for row in scoped}
            require(
                len(scoped) == len(method_rows),
                f"formal_scope_population:{operation.value}:{stage.value}",
            )
            for formal in scoped:
                clause_value = SOPClauseV1(
                    **{
                        key: value
                        for key, value in formal.items()
                        if key in clause_fields
                    }
                )
                request = VisibilityRequest(
                    operation=operation,
                    generation_stage=stage,
                    governance_stage=GovernanceStage.RETRIEVAL,
                    active_protocol=protocol_ref,
                    task_context=TaskContext(
                        task_id=target_task_id,
                        task_family=target_task_family,
                    ),
                    memory_bundle_version=base.bundle_version,
                    token_budget=4096,
                    requesting_component="formal_child_verifier",
                )
                decision = authorize_clause_for_visibility(
                    clause_value,
                    request,
                    authority_engine=engine,
                )
                require(
                    decision.allowed is True,
                    f"formal_allow:{operation.value}:{stage.value}:"
                    f"{formal.get('clause_id')}",
                )
            visible = snapshot.base_clauses(
                operation,
                task_id=target_task_id,
                task_family=target_task_family,
                generation_stage=stage.value,
                governance_stage=GovernanceStage.RETRIEVAL.value,
            )
            require(
                {str(row["clause_id"]) for row in visible} == expected_ids,
                f"full_visible_universe:{operation.value}:{stage.value}",
            )
        formal = by_id.get(formal_clause_id) or {}
        clause_value = SOPClauseV1(
            **{key: value for key, value in formal.items() if key in clause_fields}
        )
        rank_request = VisibilityRequest(
            operation=Operation.RANK,
            generation_stage=GenerationStage.IMPROVE,
            governance_stage=GovernanceStage.BRANCH_SELECTION,
            active_protocol=protocol_ref,
            task_context=TaskContext(
                task_id=target_task_id,
                task_family=target_task_family,
            ),
            memory_bundle_version=base.bundle_version,
            token_budget=4096,
            requesting_component="formal_child_verifier",
        )
        rank_decision = authorize_clause_for_visibility(
            clause_value,
            rank_request,
            authority_engine=engine,
        )
        require(rank_decision.allowed is False, "source_score_rank_denial")
        snapshot.assert_unchanged()
        result: dict[str, Any] = {
            "schema": VERIFICATION_SCHEMA,
            "valid": not errors,
            "errors": sorted(set(errors)),
            "bundle_id": base.bundle_id,
            "bundle_version": base.bundle_version,
            "bundle_manifest_sha256": base.manifest_sha256,
            "current_pointer_sha256": read_json(publication_root / "CURRENT.json").get(
                "pointer_sha256"
            ),
            "parent_bundle_id": expected_parent_bundle_id,
            "parent_manifest_sha256": expected_parent_manifest_sha256,
            "target_task_id": target_task_id,
            "target_domain": target_domain,
            "formal_clause_id": formal_clause_id,
            "formal_claim_id": formal_claim_id,
            "formal_protocol_ref": protocol_ref.key(),
            "raw_clause_count": len(clauses) - len(formal_protocol_clauses),
            "formal_protocol_clause_count": len(formal_protocol_clauses),
            "source_run_count": len(split.get("source_run_ids") or []),
            "heldout_run_count": len(split.get("heldout_run_ids") or []),
            "generic_validation": generic,
            "authority_load_report_hash": authority_load["report_hash"],
            "report_hash": "",
        }
    except Exception as error:
        result = {
            "schema": VERIFICATION_SCHEMA,
            "valid": False,
            "errors": sorted(
                set([*errors, f"exception:{type(error).__name__}:{error}"])
            ),
            "target_task_id": target_task_id,
            "target_domain": target_domain,
            "formal_clause_id": formal_clause_id,
            "formal_claim_id": formal_claim_id,
            "formal_protocol_ref": protocol_ref.key(),
            "report_hash": "",
        }
    result["report_hash"] = payload_hash(result, "report_hash")
    return result


def publish_formal_child_bundle(
    *,
    parent_bundle: str | Path,
    expected_parent_manifest_sha256: str,
    publication_root: str | Path,
    bundle_id: str,
    bundle_version: str,
    target_task_id: str,
    target_task_family: str,
    target_domain: str,
    split_mode: str,
    source_clause_id: str,
    source_run_id: str,
    source_node_id: str,
    protocol_file: str | Path,
    publication_class: str,
    agent_seeds: tuple[int, ...] = (104729, 130363, 155921),
    allowed_protocol_issue_codes: Iterable[str] = (),
    additional_source_methods: Iterable[Mapping[str, Any]] = (),
    created_at: str | None = None,
) -> dict[str, Any]:
    parent = ImmutableBaseBundle.load(parent_bundle, verify_artifacts=True)
    if parent.manifest_sha256 != expected_parent_manifest_sha256:
        raise ValueError("Formal child parent manifest mismatch")
    parent_validation = validate_bundle(parent.path)
    parent_validation_disposition = _parent_validation_disposition(
        parent, parent_validation
    )
    if parent_validation_disposition.get("accepted") is not True:
        raise ValueError(
            "Formal child parent Bundle validation was rejected: "
            f"{parent_validation_disposition.get('reasons') or []}"
        )
    publication_root = Path(publication_root).resolve()
    if publication_root.exists():
        raise FileExistsError(
            f"Formal child publication root already exists: {publication_root}"
        )
    publication_root.mkdir(parents=True)
    reports_dir = publication_root / "reports"
    reports_dir.mkdir()
    created_at = created_at or utc_now()
    try:
        inputs = publication_root / "build_inputs"
        inputs.mkdir()
        protocol_ref = _protocol_ref(
            parent,
            Path(protocol_file).resolve(),
            inputs / "protocol_registry",
        )
        parent_corpus = CorpusManifestV1.from_dict(
            parent.read_json("corpus/manifest.json")
        )
        parent_split = SplitManifestV1.from_dict(parent.read_json("splits/active.json"))
        parent_entries = {run.run_id: run for run in parent_corpus.runs}
        method_specs = [
            {
                "source_clause_id": source_clause_id,
                "source_run_id": source_run_id,
                "source_node_id": source_node_id,
                "method_text": None,
            },
            *[dict(value) for value in additional_source_methods],
        ]
        if not method_specs:
            raise ValueError("Formal child publication requires source methods")
        for spec in method_specs:
            missing = {
                key
                for key in ("source_clause_id", "source_run_id", "source_node_id")
                if not str(spec.get(key) or "")
            }
            if missing:
                raise ValueError(f"Formal source method is incomplete: {sorted(missing)}")
            selected_run = str(spec["source_run_id"])
            if selected_run not in parent_entries:
                raise ValueError("Selected source run is absent from parent corpus")
            selected_family = parent_entries[selected_run].task_family
            if canonical_domain(selected_family) != target_domain:
                raise ValueError(
                    "Selected source method is outside the formal target domain"
                )
            if selected_run not in parent_split.source_run_ids:
                raise ValueError("Selected source run is outside parent memory split")
        if split_mode == "same-domain-task-heldout":
            # Build a true task-heldout view from any reviewed parent split:
            # retain only same-domain memory-side runs from *other* tasks.
            # This supports genuinely new targets (for example Spooky when
            # the reviewed corpus intentionally excludes Spooky) without
            # inventing held-out runs or widening task-scoped Authority.
            materialized_source_run_ids = _task_heldout_source_run_ids(
                parent_split,
                parent_entries,
                target_task_id=target_task_id,
                target_domain=target_domain,
                agent_seeds=agent_seeds,
            )
            if not materialized_source_run_ids:
                raise ValueError(
                    "Task-heldout child has no same-domain non-target source runs"
                )
            missing_selected_runs = {
                str(spec["source_run_id"])
                for spec in method_specs
                if str(spec["source_run_id"]) not in materialized_source_run_ids
            }
            if missing_selected_runs:
                raise ValueError(
                    "Selected source method is not in the task-heldout source view: "
                    f"{sorted(missing_selected_runs)}"
                )
            retained = set(materialized_source_run_ids)
        elif split_mode == "same-domain-seed-heldout":
            materialized_source_run_ids = _same_domain_seed_source_run_ids(
                parent_split,
                parent_entries,
                target_domain=target_domain,
                agent_seeds=agent_seeds,
            )
            if not any(
                parent_entries[run_id].canonical_task_id == target_task_id
                for run_id in materialized_source_run_ids
            ):
                raise ValueError(
                    "Same-domain seed-heldout child has no target-task history"
                )
            missing_selected_runs = {
                str(spec["source_run_id"])
                for spec in method_specs
                if str(spec["source_run_id"])
                not in materialized_source_run_ids
            }
            if missing_selected_runs:
                raise ValueError(
                    "Selected source method is outside the same-domain source view: "
                    f"{sorted(missing_selected_runs)}"
                )
            retained = {
                run_id
                for run_id in (
                    *parent_split.source_run_ids,
                    *parent_split.heldout_run_ids,
                )
                if parent_entries.get(run_id) is not None
                and canonical_domain(parent_entries[run_id].task_family)
                == target_domain
            }
        else:
            retained = {
                run_id
                for run_id in (
                    *parent_split.source_run_ids,
                    *parent_split.heldout_run_ids,
                )
                if parent_entries.get(run_id) is not None
                and parent_entries[run_id].canonical_task_id == target_task_id
            }
            materialized_source_run_ids = {
                run_id
                for run_id in parent_split.source_run_ids
                if parent_entries.get(run_id) is not None
                and parent_entries[run_id].canonical_task_id == target_task_id
            }
        source_view_root = inputs / "parent_source_view"
        source_view_report = _materialize_parent_source_view(
            parent,
            parent_corpus,
            source_run_ids=materialized_source_run_ids,
            destination=source_view_root,
        )
        write_json_atomic(
            inputs / "parent_source_view_report.json",
            source_view_report,
        )
        corpus = _filtered_corpus(
            parent_corpus,
            retained_run_ids=retained,
            materialized_source_root=source_view_root,
            materialized_source_run_ids=materialized_source_run_ids,
            parent_manifest_sha256=parent.manifest_sha256,
            target_task_id=target_task_id,
            target_domain=target_domain,
            created_at=created_at,
        )
        corpus_path = inputs / "corpus_manifest.json"
        write_json_atomic(corpus_path, corpus.as_dict())
        split = _child_split(
            parent_split,
            corpus,
            target_task_id=target_task_id,
            target_task_family=target_task_family,
            target_domain=target_domain,
            split_mode=split_mode,
            agent_seeds=agent_seeds,
            created_at=created_at,
        )
        split_path = inputs / "split_manifest.json"
        write_json_atomic(split_path, split.as_dict())
        drift_review = {
            "schema": "corpus_drift_review_v1",
            "reviewed": True,
            "reviewed_by": "host.formal_domain_child_filter_v1",
            "reviewed_at": created_at,
            "corpus_manifest_hash": corpus.manifest_sha256,
            "actual_snapshot_hash": sha256_json(corpus.actual_snapshot),
            "excluded_runs_reviewed": True,
            "disposition": "derive_domain_subset_from_reviewed_immutable_parent",
            "parent_corpus_manifest_sha256": parent_corpus.manifest_sha256,
            "parent_bundle_manifest_sha256": parent.manifest_sha256,
        }
        drift_path = inputs / "corpus_drift_review.json"
        write_json_atomic(drift_path, drift_review)

        record_sets = []
        collector_host = TrustedCollectorHost(
            f"wp8-tier2-formal-child::{target_task_id}", collector_version="1"
        )
        receipt_cache: dict[tuple[str, str, str, str, str], Receipt] = {}
        for spec in method_specs:
            selected_clause_id = str(spec["source_clause_id"])
            selected_run_id = str(spec["source_run_id"])
            selected_node_id = str(spec["source_node_id"])
            source_clause = _find_clause(parent, selected_clause_id)
            if set(map(str, source_clause.get("claim_types") or [])) != {
                ClaimType.METHOD_HYPOTHESIS.value
            }:
                raise ValueError(
                    "Selected formal source clause is not a pure METHOD_HYPOTHESIS"
                )
            expected_source_ref = (
                f"run::{selected_run_id}::node::{selected_node_id}"
            )
            if expected_source_ref not in set(
                source_clause.get("source_artifact_refs") or []
            ):
                raise ValueError(
                    "Selected source clause does not bind the selected node"
                )
            selected_entry = parent_entries[selected_run_id]
            record_sets.append(
                _formal_method_records(
                    parent=parent,
                    source_clause=source_clause,
                    source_run_id=selected_run_id,
                    source_node_id=selected_node_id,
                    source_task_id=selected_entry.canonical_task_id,
                    source_task_family=selected_entry.task_family,
                    target_task_id=target_task_id,
                    target_task_family=target_task_family,
                    target_domain=target_domain,
                    protocol_ref=protocol_ref,
                    publication_class=publication_class,
                    allowed_protocol_issue_codes=set(
                        map(str, allowed_protocol_issue_codes)
                    ),
                    method_text=(
                        str(spec["method_text"])
                        if spec.get("method_text") is not None
                        else None
                    ),
                    collector_host=collector_host,
                    receipt_cache=receipt_cache,
                )
            )
        records = record_sets[0]
        formal_clause_rows = [
            clause for record in record_sets for clause in record["clauses"]
        ]
        formal_clause_ids = [str(row["clause_id"]) for row in formal_clause_rows]
        if len(formal_clause_ids) != len(set(formal_clause_ids)):
            raise ValueError("Formal child methods produced duplicate Clause IDs")
        formal_sop_ids = [str(record["container"]["sop_id"]) for record in record_sets]
        if len(formal_sop_ids) != len(set(formal_sop_ids)):
            raise ValueError("Formal child methods produced duplicate SOP IDs")

        clauses = parent.read_jsonl("sop/clauses.jsonl")
        parent_clause_ids = {str(row["clause_id"]) for row in clauses}
        if parent_clause_ids & set(formal_clause_ids):
            raise ValueError("Formal child Clause ID already exists in parent")
        clauses.extend(formal_clause_rows)
        clauses_path = inputs / "clauses.jsonl"
        write_jsonl(
            clauses_path,
            sorted(clauses, key=lambda row: str(row["clause_id"])),
        )
        containers = parent.read_json("sop/containers.json")
        container_rows = [dict(row) for row in containers.get("containers") or []]
        container_rows.extend(record["container"] for record in record_sets)
        containers_path = inputs / "containers.json"
        write_json_atomic(
            containers_path,
            {
                "schema": "bundle_sop_containers_v1",
                "containers": sorted(
                    container_rows, key=lambda row: str(row["sop_id"])
                ),
            },
        )

        authority_dir = inputs / "authority"
        _copy_declared_authority(parent, authority_dir)
        _append_unique(
            authority_dir / "claims.jsonl",
            [_claim_row(record["claim"]) for record in record_sets],
            identity_key="claim_id",
        )
        _append_unique(
            authority_dir / "receipts.jsonl",
            [
                _receipt_row(receipt)
                for record in record_sets
                for receipt in record["receipts"]
            ],
            identity_key="receipt_id",
        )
        _append_unique(
            authority_dir / "paths.jsonl",
            [_path_row(record["path"]) for record in record_sets],
            identity_key="path_id",
        )
        _append_unique(
            authority_dir / "derivations.jsonl",
            [
                derivation
                for record in record_sets
                for derivation in record["derivations"]
            ],
            identity_key="derivation_id",
        )

        runforest_dir = inputs / "runforest"
        runforest_report = build_runforest_v2(
            corpus_path,
            parent.path / "audit_sidecars",
            clauses_path,
            containers_path,
            split_path,
            runforest_dir,
            bundle_id=bundle_id,
            authority_decisions_path=authority_dir / "decisions.jsonl",
            created_at=created_at,
        )
        if set(runforest_report.get("source_domains") or []) != {target_domain}:
            raise ValueError("Formal child RunForest is not domain-bound")
        bundle_path = publication_root / "bundles" / bundle_version
        build_result = build_bundle(
            corpus_path,
            drift_path,
            split_path,
            parent.path / "audit_sidecars",
            runforest_dir,
            inputs / "protocol_registry",
            bundle_path,
            bundle_id=bundle_id,
            bundle_version=bundle_version,
            authority_policy_version=str(
                parent.manifest.get("authority_policy_version") or "authority_v1"
            ),
            detector_version=str(parent.manifest.get("detector_version") or ""),
            deepseek_model=str(parent.manifest.get("deepseek_model") or ""),
            deepseek_prompt_hash=str(parent.manifest.get("deepseek_prompt_hash") or ""),
            authority_dir=authority_dir,
            parent_bundle=parent.bundle_id,
            certification_level=f"formal_domain_{publication_class}",
            created_at=created_at,
        )
        child = ImmutableBaseBundle.load(bundle_path, verify_artifacts=True)
        pointer = make_current_pointer(
            bundle_path=bundle_path.relative_to(publication_root).as_posix(),
            manifest=child.manifest,
            parent_bundle=parent.bundle_id,
            published_at=created_at,
        )
        runtime_write_json_atomic(publication_root / "CURRENT.json", pointer)
        formal_methods = []
        for spec, record in zip(method_specs, record_sets):
            selected_run_id = str(spec["source_run_id"])
            formal_methods.append(
                {
                    "source_clause_id": str(spec["source_clause_id"]),
                    "source_run_id": selected_run_id,
                    "source_node_id": str(spec["source_node_id"]),
                    "source_task_id": parent_entries[
                        selected_run_id
                    ].canonical_task_id,
                    "formal_sop_id": str(record["container"]["sop_id"]),
                    "formal_clause_ids": [
                        str(clause["clause_id"])
                        for clause in record["clauses"]
                    ],
                    "formal_claim_id": str(record["claim"].claim_id),
                    "formal_path_id": str(record["path"].path_id),
                    "formal_receipt_ids": sorted(
                        receipt.receipt_id for receipt in record["receipts"]
                    ),
                    "supporting_transition_ref": str(
                        record["support"]["transition_ref"]
                    ),
                    "supporting_transition_outcome": str(
                        record["support"]["outcome"]
                    ),
                    "method_semantic_purity_report_hash": str(
                        record["purity"]["report_hash"]
                    ),
                }
            )
        all_formal_receipt_ids = sorted(
            {
                receipt_id
                for method in formal_methods
                for receipt_id in method["formal_receipt_ids"]
            }
        )
        publication_report = {
            "schema": PUBLICATION_SCHEMA,
            "bundle_id": child.bundle_id,
            "bundle_version": child.bundle_version,
            "bundle_path": str(child.path),
            "bundle_manifest_sha256": child.manifest_sha256,
            "current_pointer_sha256": pointer["pointer_sha256"],
            "parent_bundle_id": parent.bundle_id,
            "parent_manifest_sha256": parent.manifest_sha256,
            "parent_manifest_file_sha256": parent.manifest_file_sha256,
            "parent_validation_report_hash": sha256_json(parent_validation),
            "parent_validation_disposition": parent_validation_disposition,
            "target_task_id": target_task_id,
            "target_task_family": target_task_family,
            "target_domain": target_domain,
            "split_mode": split_mode,
            "split_id": split.split_id,
            "split_manifest_sha256": split.manifest_sha256,
            "corpus_manifest_sha256": corpus.manifest_sha256,
            "parent_source_view_hash": source_view_report["source_view_hash"],
            "source_clause_id": source_clause_id,
            "source_run_id": source_run_id,
            "source_node_id": source_node_id,
            "source_task_id": formal_methods[0]["source_task_id"],
            "source_task_ids": sorted(
                {str(method["source_task_id"]) for method in formal_methods}
            ),
            "formal_clause_id": records["clause"]["clause_id"],
            "formal_debug_clause_id": records["debug_clause"]["clause_id"],
            "formal_claim_id": records["claim"].claim_id,
            "formal_path_id": records["path"].path_id,
            "formal_receipt_ids": all_formal_receipt_ids,
            "formal_method_count": len(formal_methods),
            "formal_methods": formal_methods,
            "formal_protocol_ref": protocol_ref.key(),
            "publication_class": publication_class,
            "source_score_inheritance": False,
            "historical_metric_used_as_evidence": False,
            "allowed_protocol_issue_codes": sorted(
                set(map(str, allowed_protocol_issue_codes))
            ),
            "method_semantic_purity_report_hash": records["purity"]["report_hash"],
            "build_result": build_result,
            "runforest_report_hash": sha256_json(runforest_report),
            "created_at": created_at,
            "report_hash": "",
        }
        publication_report["report_hash"] = payload_hash(
            publication_report, "report_hash"
        )
        write_json_atomic(reports_dir / "publication_report.json", publication_report)
        verification = verify_formal_child_bundle(
            publication_root,
            expected_parent_bundle_id=parent.bundle_id,
            expected_parent_manifest_sha256=parent.manifest_sha256,
            target_task_id=target_task_id,
            target_task_family=target_task_family,
            target_domain=target_domain,
            formal_clause_id=records["clause"]["clause_id"],
            formal_claim_id=records["claim"].claim_id,
            formal_methods=formal_methods,
            protocol_ref=protocol_ref,
            split_mode=split_mode,
            agent_seeds=agent_seeds,
        )
        write_json_atomic(reports_dir / "verification_report.json", verification)
        if verification.get("valid") is not True:
            raise ValueError(
                f"Formal child Bundle verification failed: {verification['errors']}"
            )
        parent.assert_unchanged()
        # The formal publication is a sealed experiment input, not a mutable
        # sleep-time channel.  Freeze the Bundle, pointer, reports and build
        # provenance together; the later staging manifest also binds their
        # hashes and training mounts the root read-only.
        for path in sorted(publication_root.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        publication_root.chmod(0o555)
        return {
            "publication": publication_report,
            "verification": verification,
        }
    except BaseException as error:
        failure = {
            "schema": "tier2_formal_child_bundle_publication_failure_v1",
            "parent_bundle_id": parent.bundle_id,
            "parent_manifest_sha256": parent.manifest_sha256,
            "target_task_id": target_task_id,
            "target_domain": target_domain,
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_at": utc_now(),
            "failure_hash": "",
        }
        failure["failure_hash"] = payload_hash(failure, "failure_hash")
        write_json_atomic(reports_dir / "publication_failure.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-bundle", type=Path, required=True)
    parser.add_argument("--expected-parent-manifest-sha256", required=True)
    parser.add_argument("--publication-root", type=Path, required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--target-task-id", required=True)
    parser.add_argument("--target-task-family", required=True)
    parser.add_argument(
        "--target-domain",
        choices=["image", "audio", "tabular", "nlp"],
        required=True,
    )
    parser.add_argument(
        "--split-mode",
        choices=[
            "same-domain-task-heldout",
            "same-domain-seed-heldout",
            "seed-heldout",
        ],
        required=True,
    )
    parser.add_argument("--source-clause-id", required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-node-id", required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument(
        "--publication-class",
        choices=["certified", "provisional"],
        required=True,
    )
    parser.add_argument("--agent-seed", type=int, action="append")
    parser.add_argument("--allow-source-audit-issue-code", action="append")
    parser.add_argument("--created-at")
    args = parser.parse_args()
    result = publish_formal_child_bundle(
        parent_bundle=args.parent_bundle,
        expected_parent_manifest_sha256=args.expected_parent_manifest_sha256,
        publication_root=args.publication_root,
        bundle_id=args.bundle_id,
        bundle_version=args.bundle_version,
        target_task_id=args.target_task_id,
        target_task_family=args.target_task_family,
        target_domain=args.target_domain,
        split_mode=args.split_mode,
        source_clause_id=args.source_clause_id,
        source_run_id=args.source_run_id,
        source_node_id=args.source_node_id,
        protocol_file=args.protocol_file,
        publication_class=args.publication_class,
        agent_seeds=tuple(args.agent_seed or (104729, 130363, 155921)),
        allowed_protocol_issue_codes=tuple(args.allow_source_audit_issue_code or ()),
        created_at=args.created_at,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "PUBLICATION_SCHEMA",
    "VERIFICATION_SCHEMA",
    "publish_formal_child_bundle",
    "verify_formal_child_bundle",
]
