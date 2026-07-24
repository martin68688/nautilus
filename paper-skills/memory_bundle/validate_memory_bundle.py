from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from authority.domain_scope import (
    DOMAIN_GENERAL,
    SAME_DOMAIN,
    canonical_domain,
    normalize_transfer_scope,
    transfer_is_compatible,
)
from bind_sop_clauses import read_jsonl
from schema import (
    CorpusManifestV1,
    MemoryBundleManifestV1,
    SplitManifestV1,
    payload_hash,
    read_json,
    sha256_file,
    sha256_json,
    write_json_atomic,
)


RAW_ARTIFACTS = {
    "journal": ("journal_path", "journal.json", True),
    "config": ("config_path", "config.yaml", True),
    "filtered_journal": ("filtered_journal_path", "filtered_journal.json", False),
    "best_solution": ("best_solution_path", "best_solution.py", False),
}
REQUIRED_DECLARED_ARTIFACTS = {
    "audit_sidecars/index.json",
    "authority/claims.jsonl",
    "authority/decisions.jsonl",
    "authority/derivations.jsonl",
    "authority/receipts.jsonl",
    "corpus/drift_review.json",
    "corpus/manifest.json",
    "reports/build_report.json",
    "runforest/build_report.json",
    "runforest/clause_index.npz",
    "runforest/graph.json",
    "runforest/index.npz",
    "sop/clauses.jsonl",
    "sop/containers.json",
    "sop/graph.json",
    "splits/active.json",
    "visibility/clause_metadata.jsonl",
    "visibility/precompiled_masks/declared_scope_masks.json",
}


def _source_run_id(reference: str) -> str | None:
    match = re.match(r"^run::(.+?)::(?:node|transition)::", str(reference))
    return match.group(1) if match else None


def validate_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir).resolve()
    manifest = MemoryBundleManifestV1.from_dict(
        read_json(bundle_dir / "manifest.json")
    )
    errors: list[str] = []
    warnings: list[str] = []
    for relative in sorted(
        REQUIRED_DECLARED_ARTIFACTS - set(manifest.artifact_hashes)
    ):
        errors.append(f"artifact_not_declared:{relative}")
    for relative, expected in sorted(manifest.artifact_hashes.items()):
        path = (bundle_dir / relative).resolve()
        try:
            path.relative_to(bundle_dir)
        except ValueError:
            errors.append(f"artifact_path_escape:{relative}")
            continue
        if not path.is_file():
            errors.append(f"missing_artifact:{relative}")
        elif sha256_file(path) != expected:
            errors.append(f"artifact_hash_mismatch:{relative}")
    corpus = CorpusManifestV1.from_dict(
        read_json(bundle_dir / "corpus" / "manifest.json")
    )
    split = SplitManifestV1.from_dict(
        read_json(bundle_dir / "splits" / "active.json")
    )
    if manifest.corpus_manifest_hash != corpus.manifest_sha256:
        errors.append("bundle_corpus_manifest_binding")
    drift_review = read_json(bundle_dir / "corpus" / "drift_review.json")
    if drift_review.get("schema") != "corpus_drift_review_v1":
        errors.append("missing_corpus_drift_review")
    if drift_review.get("reviewed") is not True:
        errors.append("corpus_drift_not_reviewed")
    if drift_review.get("corpus_manifest_hash") != corpus.manifest_sha256:
        errors.append("drift_review_manifest_binding")
    if drift_review.get("actual_snapshot_hash") != sha256_json(
        corpus.actual_snapshot
    ):
        errors.append("drift_review_snapshot_binding")
    if drift_review.get("excluded_runs_reviewed") is not True:
        errors.append("drift_review_exclusions")
    if split.corpus_manifest_hash != corpus.manifest_sha256:
        errors.append("split_corpus_manifest_binding")
    if manifest.split_id != split.split_id:
        errors.append("bundle_split_id_binding")
    source_runs = set(split.source_run_ids)
    heldout_runs = set(split.heldout_run_ids)
    if source_runs & heldout_runs:
        errors.append("split_run_overlap")
    if split.split_kind in {
        "task-heldout",
        "same-domain-task-heldout",
    } and set(split.source_task_ids) & set(split.heldout_task_ids):
        errors.append("task_heldout_task_overlap")
    if split.split_kind == "seed-heldout" and set(
        split.source_seed_groups
    ) & set(split.heldout_seed_groups):
        errors.append("seed_heldout_group_overlap")
    entries = {run.run_id: run for run in corpus.runs}
    source_task_ids = {
        entries[run_id].canonical_task_id
        for run_id in source_runs
        if run_id in entries
    }
    source_task_families = {
        str(entries[run_id].task_family or "").strip()
        for run_id in source_runs
        if run_id in entries
    }
    source_task_families.discard("")
    source_domains = {
        canonical_domain(value) for value in source_task_families
    }
    source_domains.discard("")
    target_task_id = str(split.allocation.get("target_task_id") or "")
    target_domain = canonical_domain(
        split.allocation.get("target_domain")
        or split.allocation.get("target_task_family")
    )
    if split.split_kind == "same-domain-task-heldout":
        if split.heldout_task_ids != [target_task_id] or not target_task_id:
            errors.append("same_domain_target_binding")
        if target_task_id in source_task_ids:
            errors.append("same_domain_target_in_source")
        if not target_domain or source_domains != {target_domain}:
            errors.append("same_domain_source_domain_mismatch")
        if split.validation.get("cross_domain_source_run_count") != 0:
            errors.append("same_domain_cross_domain_source_runs")

    graph = read_json(bundle_dir / "runforest" / "graph.json")
    if manifest.graph_hashes.get("runforest") != sha256_file(
        bundle_dir / "runforest" / "graph.json"
    ):
        errors.append("bundle_graph_hash_binding")
    nodes = [dict(node) for node in graph.get("nodes") or []]
    graph_meta = graph.get("meta") or {}
    if graph_meta.get("domain_scope_required") is not True:
        errors.append("runforest_domain_scope_not_required")
    if set(graph_meta.get("source_task_ids") or []) != source_task_ids:
        errors.append("runforest_source_task_lineage_mismatch")
    if set(graph_meta.get("source_task_families") or []) != source_task_families:
        errors.append("runforest_source_family_lineage_mismatch")
    if set(graph_meta.get("source_domains") or []) != source_domains:
        errors.append("runforest_source_domain_lineage_mismatch")
    if split.split_kind == "same-domain-task-heldout":
        if graph_meta.get("target_task_id") != target_task_id:
            errors.append("runforest_target_task_mismatch")
        if canonical_domain(graph_meta.get("target_domain")) != target_domain:
            errors.append("runforest_target_domain_mismatch")
    node_by_id = {str(node["id"]): node for node in nodes}
    if len(node_by_id) != len(nodes):
        errors.append("duplicate_graph_node_ids")
    graph_run_ids = {
        str(node.get("run_id"))
        for node in nodes
        if node.get("run_id") is not None
    }
    foreign_runs = graph_run_ids - source_runs
    if foreign_runs:
        errors.append(f"graph_refs_outside_source_split:{sorted(foreign_runs)}")
    if graph_run_ids & heldout_runs:
        errors.append("heldout_run_in_graph")
    spooky = [
        node["id"]
        for node in nodes
        if "spooky" in str(node.get("task") or "").lower()
    ]
    if spooky:
        errors.append(f"spooky_nodes:{len(spooky)}")

    clauses = read_jsonl(bundle_dir / "sop" / "clauses.jsonl")
    runforest_report = read_json(bundle_dir / "runforest" / "build_report.json")
    if runforest_report.get("included_clause_count") != len(clauses):
        errors.append("runforest_report_clause_count_mismatch")
    excluded_reasons = {
        str(row.get("reason"))
        for row in runforest_report.get("excluded_clauses") or []
    }
    if split.split_kind == "seed-heldout" and "heldout_task_scope" in excluded_reasons:
        errors.append("seed_heldout_task_scope_exclusion")
    if (
        source_runs
        and int(runforest_report.get("input_clause_count") or 0) > 0
        and len(clauses) == 0
    ):
        errors.append("nonempty_source_bundle_has_zero_clauses")
    containers = read_json(bundle_dir / "sop" / "containers.json").get(
        "containers"
    ) or []
    clause_ids = {str(clause["clause_id"]) for clause in clauses}
    memberships: dict[str, int] = {clause_id: 0 for clause_id in clause_ids}
    for container in containers:
        for clause_id in container.get("clause_ids") or []:
            if clause_id not in clause_ids:
                errors.append(f"container_missing_clause:{clause_id}")
            else:
                memberships[clause_id] += 1
    invalid_memberships = {
        clause_id: count for clause_id, count in memberships.items() if count != 1
    }
    if invalid_memberships:
        errors.append(f"clause_container_membership:{invalid_memberships}")
    clauses_by_id = {
        str(clause["clause_id"]): clause for clause in clauses
    }
    for container in containers:
        member_clauses = [
            clauses_by_id[clause_id]
            for clause_id in container.get("clause_ids") or []
            if clause_id in clauses_by_id
        ]
        expected_container_runs = {
            value
            for clause in member_clauses
            for value in clause.get("source_run_ids") or []
        }
        expected_container_tasks = {
            value
            for clause in member_clauses
            for value in clause.get("source_task_ids") or []
        }
        expected_container_families = {
            value
            for clause in member_clauses
            for value in clause.get("source_task_families") or []
        }
        expected_container_domains = {
            value
            for clause in member_clauses
            for value in clause.get("source_domains") or []
        }
        expected_transfer_scopes = {
            normalize_transfer_scope(clause.get("transfer_scope"))
            for clause in member_clauses
        }
        expected_transfer_scopes.discard("")
        if set(container.get("source_run_ids") or []) != expected_container_runs:
            errors.append(f"container_source_run_lineage:{container['sop_id']}")
        if set(container.get("source_task_ids") or []) != expected_container_tasks:
            errors.append(f"container_source_task_lineage:{container['sop_id']}")
        if set(container.get("source_task_families") or []) != expected_container_families:
            errors.append(f"container_source_family_lineage:{container['sop_id']}")
        if set(container.get("source_domains") or []) != expected_container_domains:
            errors.append(f"container_source_domain_lineage:{container['sop_id']}")
        if set(container.get("transfer_scopes") or []) != expected_transfer_scopes:
            errors.append(f"container_transfer_scope_lineage:{container['sop_id']}")
        if container.get("domain_scope_complete") is not True:
            errors.append(f"container_domain_scope_incomplete:{container['sop_id']}")
    for clause in clauses:
        refs = [
            *list(clause.get("source_artifact_refs") or []),
            *list(clause.get("source_transition_refs") or []),
        ]
        if not refs:
            errors.append(f"clause_missing_source:{clause['clause_id']}")
            continue
        missing_refs = [reference for reference in refs if reference not in node_by_id]
        if missing_refs:
            errors.append(
                f"clause_unresolved_refs:{clause['clause_id']}:{missing_refs}"
            )
        ref_runs = {
            run_id for reference in refs if (run_id := _source_run_id(reference))
        }
        if len(ref_runs) == 0 or any(
            _source_run_id(reference) is None for reference in refs
        ):
            errors.append(f"clause_unparseable_source_refs:{clause['clause_id']}")
        if not ref_runs <= source_runs:
            errors.append(f"clause_source_outside_split:{clause['clause_id']}")
        expected_task_ids = {
            entries[run_id].canonical_task_id
            for run_id in ref_runs
            if run_id in entries
        }
        expected_families = {
            str(entries[run_id].task_family or "").strip()
            for run_id in ref_runs
            if run_id in entries
        }
        expected_families.discard("")
        expected_domains = {
            canonical_domain(value) for value in expected_families
        }
        expected_domains.discard("")
        if set(clause.get("source_run_ids") or []) != ref_runs:
            errors.append(f"clause_source_run_lineage:{clause['clause_id']}")
        if set(clause.get("source_task_ids") or []) != expected_task_ids:
            errors.append(f"clause_source_task_lineage:{clause['clause_id']}")
        if set(clause.get("source_task_families") or []) != expected_families:
            errors.append(f"clause_source_family_lineage:{clause['clause_id']}")
        if set(clause.get("source_domains") or []) != expected_domains:
            errors.append(f"clause_source_domain_lineage:{clause['clause_id']}")
        transfer_scope = normalize_transfer_scope(
            clause.get("transfer_scope")
        )
        if not transfer_scope:
            errors.append(f"clause_missing_transfer_scope:{clause['clause_id']}")
        if (
            split.split_kind == "same-domain-task-heldout"
            and transfer_scope == SAME_DOMAIN
            and not transfer_is_compatible(
                expected_domains,
                target_domain,
                transfer_scope,
            )
        ):
            errors.append(f"clause_cross_domain_transfer:{clause['clause_id']}")
        if transfer_scope == DOMAIN_GENERAL and clause.get(
            "admissible_target_domains"
        ) != [DOMAIN_GENERAL]:
            errors.append(f"clause_domain_general_marker:{clause['clause_id']}")

    authority_dir = bundle_dir / "authority"
    claims = read_jsonl(authority_dir / "claims.jsonl")
    receipts = read_jsonl(authority_dir / "receipts.jsonl")
    decisions = read_jsonl(authority_dir / "decisions.jsonl")
    derivations = read_jsonl(authority_dir / "derivations.jsonl")
    claim_ids = {str(row.get("claim_id")) for row in claims}
    receipt_ids = {str(row.get("receipt_id")) for row in receipts}
    decision_ids = {str(row.get("decision_id")) for row in decisions}
    derivation_ids = {str(row.get("derivation_id")) for row in derivations}
    requested_claim_ids = {
        str(value) for clause in clauses for value in clause.get("claim_refs") or []
    }
    requested_receipt_ids = {
        str(value) for clause in clauses for value in clause.get("receipt_refs") or []
    }
    requested_decision_ids = {
        str(value)
        for clause in clauses
        for value in clause.get("authority_decision_refs") or []
    }
    requested_derivation_ids = {
        str(value) for clause in clauses for value in clause.get("derivation_refs") or []
    }
    if requested_claim_ids - claim_ids:
        errors.append("clause_unresolved_claim_refs")
    if requested_receipt_ids - receipt_ids:
        errors.append("clause_unresolved_receipt_refs")
    if requested_decision_ids - decision_ids:
        errors.append("clause_unresolved_decision_refs")
    if requested_derivation_ids - derivation_ids:
        errors.append("clause_unresolved_derivation_refs")
    if claim_ids - requested_claim_ids:
        errors.append("authority_claims_outside_clause_set")

    audit_index = read_json(bundle_dir / "audit_sidecars" / "index.json")
    sidecar_entries = audit_index.get("entries") or {}
    code_nodes = [node for node in nodes if node.get("type") == "RunNode" and node.get("code_sha256")]
    missing_sidecars = [node["id"] for node in code_nodes if node["id"] not in sidecar_entries]
    if missing_sidecars:
        errors.append(f"code_nodes_missing_sidecars:{len(missing_sidecars)}")
    for artifact_id, filename in sidecar_entries.items():
        sidecar_path = bundle_dir / "audit_sidecars" / str(filename)
        if not sidecar_path.is_file():
            errors.append(f"missing_sidecar_file:{artifact_id}")
            continue
        sidecar = read_json(sidecar_path)
        if sidecar.get("artifact_id") != artifact_id:
            errors.append(f"sidecar_artifact_mismatch:{artifact_id}")
        if sidecar.get("run_id") not in source_runs:
            errors.append(f"sidecar_outside_split:{artifact_id}")
        if sidecar.get("sidecar_sha256") != payload_hash(
            sidecar, "sidecar_sha256"
        ):
            errors.append(f"sidecar_hash_mismatch:{artifact_id}")
        graph_node = node_by_id.get(artifact_id)
        if graph_node is None or graph_node.get("type") != "RunNode":
            errors.append(f"sidecar_without_graph_code_node:{artifact_id}")
        elif graph_node.get("code_sha256") != sidecar.get("code_sha256"):
            errors.append(f"sidecar_graph_code_hash_mismatch:{artifact_id}")
    extra_sidecars = set(sidecar_entries) - {str(node["id"]) for node in code_nodes}
    if extra_sidecars:
        errors.append(f"sidecars_outside_code_node_set:{len(extra_sidecars)}")

    index = np.load(bundle_dir / "runforest" / "index.npz", allow_pickle=True)
    indexed_node_ids = {str(value) for value in index["node_ids"].tolist()}
    if indexed_node_ids != set(node_by_id):
        errors.append("runforest_index_node_mismatch")
    if manifest.index_hashes.get("runforest") != sha256_file(
        bundle_dir / "runforest" / "index.npz"
    ):
        errors.append("bundle_runforest_index_hash_binding")
    clause_index = np.load(
        bundle_dir / "runforest" / "clause_index.npz", allow_pickle=True
    )
    indexed_clause_ids = {str(value) for value in clause_index["clause_ids"].tolist()}
    if indexed_clause_ids != clause_ids:
        errors.append("clause_index_id_mismatch")
    if manifest.index_hashes.get("clauses") != sha256_file(
        bundle_dir / "runforest" / "clause_index.npz"
    ):
        errors.append("bundle_clause_index_hash_binding")
    visibility_rows = read_jsonl(
        bundle_dir / "visibility" / "clause_metadata.jsonl"
    )
    if {str(row["clause_id"]) for row in visibility_rows} != clause_ids:
        errors.append("visibility_metadata_clause_mismatch")
    masks_path = (
        bundle_dir
        / "visibility"
        / "precompiled_masks"
        / "declared_scope_masks.json"
    )
    if not masks_path.is_file():
        errors.append("missing_declared_scope_masks")
    else:
        masks = read_json(masks_path).get("masks") or {}
        mask_clause_ids = {
            str(clause_id)
            for values in masks.values()
            for clause_id in values
        }
        if not mask_clause_ids <= clause_ids:
            errors.append("visibility_mask_unknown_clause")

    raw_root = bundle_dir / "raw_journals"
    raw_run_dirs = {path.name for path in raw_root.iterdir() if path.is_dir()}
    if raw_run_dirs != source_runs:
        errors.append(
            f"raw_journal_run_set_mismatch:expected={len(source_runs)} actual={len(raw_run_dirs)}"
        )
    corpus_entries = {run.run_id: run for run in corpus.runs}
    for run_id in sorted(source_runs):
        entry = corpus_entries.get(run_id)
        if entry is None or entry.status != "complete":
            errors.append(f"raw_source_not_complete:{run_id}")
            continue
        run_dir = raw_root / run_id
        for artifact_name, (attribute, filename, required) in RAW_ARTIFACTS.items():
            source_relative = getattr(entry, attribute)
            destination = run_dir / filename
            expected = entry.artifact_hashes.get(artifact_name)
            if source_relative is None:
                if destination.exists():
                    errors.append(f"unexpected_raw_artifact:{run_id}/{filename}")
                elif required:
                    errors.append(f"required_raw_artifact_unavailable:{run_id}/{filename}")
                continue
            if not destination.is_file():
                errors.append(f"missing_raw_artifact:{run_id}/{filename}")
            elif not expected or sha256_file(destination) != expected:
                errors.append(f"raw_artifact_hash_mismatch:{run_id}/{filename}")

    registry_hashes = {
        path.relative_to(bundle_dir / "protocol_registry").as_posix(): sha256_file(path)
        for path in sorted((bundle_dir / "protocol_registry").rglob("*"))
        if path.is_file()
    }
    if manifest.protocol_registry_hash != sha256_json(registry_hashes):
        errors.append("protocol_registry_hash_binding")
    expected_lineage_hash = sha256_json(
        {
            "clauses": sha256_file(bundle_dir / "sop" / "clauses.jsonl"),
            "containers": sha256_file(bundle_dir / "sop" / "containers.json"),
            "derivations": sha256_file(
                bundle_dir / "authority" / "derivations.jsonl"
            ),
            "sop_graph": sha256_file(bundle_dir / "sop" / "graph.json"),
        }
    )
    if manifest.lineage_hash != expected_lineage_hash:
        errors.append("lineage_hash_binding")
    build_report = read_json(bundle_dir / manifest.build_report)
    if build_report.get("bundle_id") != manifest.bundle_id:
        errors.append("build_report_bundle_binding")
    if build_report.get("split_id") != split.split_id:
        errors.append("build_report_split_binding")
    if build_report.get("all_code_nodes_have_sidecars") is not True:
        errors.append("build_report_sidecar_incomplete")
    if build_report.get("all_clause_sources_resolve") is not True:
        errors.append("build_report_lineage_incomplete")
    if build_report.get("secret_scan_passed") is not True:
        errors.append("build_report_secret_scan")
    if build_report.get("corpus_drift_reviewed") is not True:
        errors.append("build_report_drift_review")
    if build_report.get("published_atomically") is not True:
        errors.append("build_report_atomic_publication")
    sums_path = bundle_dir / "SHA256SUMS"
    if sums_path.exists():
        summed_paths: set[str] = set()
        for line in sums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            if relative in summed_paths:
                errors.append(f"sha256sums_duplicate:{relative}")
                continue
            summed_paths.add(relative)
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                errors.append(f"sha256sums_path_escape:{relative}")
                continue
            path = bundle_dir / relative_path
            if not path.is_file() or sha256_file(path) != expected:
                errors.append(f"sha256sums_mismatch:{relative}")
        expected_summed_paths = {
            path.relative_to(bundle_dir).as_posix()
            for path in bundle_dir.rglob("*")
            if path.is_file()
            and path.name != "SHA256SUMS"
            and path.relative_to(bundle_dir).as_posix()
            != "reports/validation_report.json"
        }
        if summed_paths != expected_summed_paths:
            errors.append("sha256sums_file_set_mismatch")
    else:
        errors.append("missing_sha256sums")
    report = {
        "schema": "memory_bundle_validation_report_v1",
        "bundle_id": manifest.bundle_id,
        "bundle_version": manifest.bundle_version,
        "split_id": split.split_id,
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "source_run_count": len(source_runs),
        "heldout_run_count": len(heldout_runs),
        "graph_run_count": len(graph_run_ids),
        "code_node_count": len(code_nodes),
        "sidecar_count": len(sidecar_entries),
        "clause_count": len(clauses),
        "container_count": len(containers),
        "spooky_node_count": len(spooky),
        "heldout_reference_count": len(graph_run_ids & heldout_runs),
        "all_clause_sources_resolve": not any(
            error.startswith(("clause_missing_source", "clause_unresolved_refs"))
            for error in errors
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_bundle(args.bundle)
    if args.report:
        write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
