from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from bind_sop_clauses import read_jsonl, write_jsonl
from schema import (
    CorpusManifestV1,
    MemoryBundleManifestV1,
    SplitManifestV1,
    canonical_json,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    write_json_atomic,
)
from validate_memory_bundle import validate_bundle


SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*"
        r"[\"']?(?!\$\{oc\.env:)[A-Za-z0-9_./+-]{20,}"
    ),
]
CORE_NAMES = {
    "journal_path": "journal.json",
    "config_path": "config.yaml",
    "filtered_journal_path": "filtered_journal.json",
    "best_solution_path": "best_solution.py",
}


def _secret_scan(path: Path) -> None:
    if path.suffix.lower() not in {".json", ".jsonl", ".yaml", ".yml", ".py", ".txt", ".md"}:
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ValueError(f"Secret-like material detected in source artifact: {path.name}")


def _directory_hash(directory: Path) -> tuple[str, dict[str, str]]:
    hashes = {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }
    return sha256_json(hashes), hashes


def _copy_file(source: Path, destination: Path, *, secret_scan: bool = False) -> None:
    if secret_scan:
        _secret_scan(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _filter_authority(
    authority_dir: Path | None,
    destination: Path,
    clauses: list[dict[str, Any]],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    claim_ids = {
        str(value) for clause in clauses for value in clause.get("claim_refs") or []
    }
    decision_ids = {
        str(value)
        for clause in clauses
        for value in clause.get("authority_decision_refs") or []
    }
    receipt_ids = {
        str(value) for clause in clauses for value in clause.get("receipt_refs") or []
    }
    clause_ids = {str(clause["clause_id"]) for clause in clauses}
    specifications = {
        "claims.jsonl": ("claim_id", claim_ids),
        "receipts.jsonl": ("receipt_id", receipt_ids),
        "decisions.jsonl": ("decision_id", decision_ids),
        "derivations.jsonl": ("clause_id", clause_ids),
        "replay_receipts.jsonl": ("receipt_id", receipt_ids),
    }
    for filename, (key, allowed) in specifications.items():
        source = authority_dir / filename if authority_dir else None
        rows = read_jsonl(source) if source is not None and source.exists() else []
        selected = [row for row in rows if str(row.get(key) or "") in allowed]
        write_jsonl(destination / filename, selected)
    # Evidence paths are first-class authority records.  Earlier raw WP4
    # bundles had no trusted paths, so omitting them was harmless there; a
    # formal/provisional child Bundle must retain the exact Claim->Receipt set
    # or the live gateway correctly fails closed.
    for filename in ("paths.jsonl", "replay_paths.jsonl"):
        source = authority_dir / filename if authority_dir else None
        rows = read_jsonl(source) if source is not None and source.exists() else []
        selected = [
            row
            for row in rows
            if str(row.get("claim_id") or "") in claim_ids
            and set(map(str, row.get("receipt_ids") or [])) <= receipt_ids
        ]
        write_jsonl(destination / filename, selected)


def _bundle_sop_from_graph(graph: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clauses: list[dict[str, Any]] = []
    containers: list[dict[str, Any]] = []
    for node in graph.get("nodes") or []:
        if node.get("type") == "SOPClause":
            value = dict(node)
            value.pop("id", None)
            value.pop("type", None)
            clauses.append(value)
        elif node.get("type") == "SOP":
            containers.append(
                {
                    "sop_id": node["sop_id"],
                    "title": node.get("title"),
                    "task_id": node.get("task"),
                    "clause_ids": list(node.get("clause_ids") or []),
                    "source_run_ids": list(
                        node.get("source_run_ids") or []
                    ),
                    "source_task_ids": list(
                        node.get("source_task_ids") or []
                    ),
                    "source_task_families": list(
                        node.get("source_task_families") or []
                    ),
                    "source_domains": list(
                        node.get("source_domains") or []
                    ),
                    "transfer_scopes": list(
                        node.get("transfer_scopes") or []
                    ),
                    "domain_scope_complete": node.get(
                        "domain_scope_complete"
                    )
                    is True,
                }
            )
    clauses.sort(key=lambda row: row["clause_id"])
    containers.sort(key=lambda row: row["sop_id"])
    return clauses, containers


def create_tar_zst(bundle_dir: str | Path, archive_path: str | Path) -> dict[str, str]:
    try:
        import zstandard
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("zstandard is required to create .tar.zst archives") from error
    bundle_dir = Path(bundle_dir).resolve()
    archive_path = Path(archive_path).resolve()
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    staging = archive_path.parent / (
        f".{archive_path.name}.staging-{uuid.uuid4().hex}"
    )

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        return info

    try:
        with staging.open("wb") as raw:
            compressor = zstandard.ZstdCompressor(level=10)
            with compressor.stream_writer(raw, closefd=False) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as archive:
                    archive.add(
                        bundle_dir,
                        arcname=bundle_dir.name,
                        recursive=True,
                        filter=normalize,
                    )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(staging, archive_path)
    except Exception:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
        raise
    return {"archive": str(archive_path), "sha256": sha256_file(archive_path)}


def build_bundle(
    corpus_manifest_path: str | Path,
    drift_review_path: str | Path,
    split_manifest_path: str | Path,
    audit_dir: str | Path,
    runforest_dir: str | Path,
    protocol_registry_dir: str | Path,
    output_dir: str | Path,
    *,
    bundle_id: str,
    bundle_version: str,
    authority_policy_version: str,
    detector_version: str,
    deepseek_model: str,
    deepseek_prompt_hash: str,
    authority_dir: str | Path | None = None,
    splits_dir: str | Path | None = None,
    parent_bundle: str | None = None,
    certification_level: str = "raw_audited",
    created_at: str | None = None,
) -> dict[str, Any]:
    corpus_manifest_path = Path(corpus_manifest_path).resolve()
    drift_review_path = Path(drift_review_path).resolve()
    split_manifest_path = Path(split_manifest_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    runforest_dir = Path(runforest_dir).resolve()
    protocol_registry_dir = Path(protocol_registry_dir).resolve()
    output_dir = Path(output_dir).resolve()
    authority_dir = Path(authority_dir).resolve() if authority_dir else None
    splits_dir = Path(splits_dir).resolve() if splits_dir else None
    if output_dir.exists():
        raise FileExistsError(f"Published bundle path already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        corpus = CorpusManifestV1.from_dict(read_json(corpus_manifest_path))
        drift_review = read_json(drift_review_path)
        if drift_review.get("schema") != "corpus_drift_review_v1":
            raise ValueError("Missing corpus_drift_review_v1 approval")
        if drift_review.get("reviewed") is not True:
            raise ValueError("Corpus drift review is not approved")
        if drift_review.get("corpus_manifest_hash") != corpus.manifest_sha256:
            raise ValueError("Corpus drift review does not bind the manifest")
        if drift_review.get("actual_snapshot_hash") != sha256_json(
            corpus.actual_snapshot
        ):
            raise ValueError("Corpus drift review does not bind the actual snapshot")
        if drift_review.get("excluded_runs_reviewed") is not True:
            raise ValueError("Corpus exclusions have not been reviewed")
        if not str(drift_review.get("reviewed_by") or "").strip():
            raise ValueError("Corpus drift review has no reviewer")
        split = SplitManifestV1.from_dict(read_json(split_manifest_path))
        if split.corpus_manifest_hash != corpus.manifest_sha256:
            raise ValueError("Split manifest does not bind corpus manifest")
        source_runs = set(split.source_run_ids)
        entries = {run.run_id: run for run in corpus.runs}
        if source_runs - set(entries):
            raise ValueError("Split contains unknown source runs")
        source_root = Path(corpus.source_root)

        _copy_file(corpus_manifest_path, staging / "corpus" / "manifest.json")
        _copy_file(drift_review_path, staging / "corpus" / "drift_review.json")
        (staging / "raw_journals").mkdir()
        for run_id in sorted(source_runs):
            run = entries[run_id]
            if run.status != "complete":
                raise ValueError(f"Non-complete source run: {run_id}")
            destination = staging / "raw_journals" / run_id
            destination.mkdir()
            for attribute, output_name in CORE_NAMES.items():
                relative = getattr(run, attribute)
                if not relative:
                    continue
                source = source_root / relative
                expected = run.artifact_hashes.get(
                    attribute.replace("_path", "")
                )
                if expected and sha256_file(source) != expected:
                    raise ValueError(f"Source artifact drift: {run_id}/{output_name}")
                _copy_file(source, destination / output_name, secret_scan=True)

        source_audit_index = read_json(audit_dir / "index.json")
        selected_sidecars: dict[str, str] = {}
        audit_destination = staging / "audit_sidecars"
        audit_destination.mkdir()
        for artifact_id, filename in sorted(
            (source_audit_index.get("entries") or {}).items()
        ):
            sidecar = read_json(audit_dir / str(filename))
            if str(sidecar.get("run_id")) not in source_runs:
                continue
            _copy_file(audit_dir / str(filename), audit_destination / str(filename))
            selected_sidecars[str(artifact_id)] = str(filename)
        write_json_atomic(
            audit_destination / "index.json",
            {
                "schema": "audit_sidecar_index_v1",
                "corpus_manifest_hash": corpus.manifest_sha256,
                "detector_version": detector_version,
                "entries": selected_sidecars,
            },
        )

        runforest_destination = staging / "runforest"
        runforest_destination.mkdir()
        for filename in ("graph.json", "index.npz", "clause_index.npz", "build_report.json"):
            _copy_file(runforest_dir / filename, runforest_destination / filename)
        graph = read_json(runforest_dir / "graph.json")
        clauses, containers = _bundle_sop_from_graph(graph)
        sop_destination = staging / "sop"
        sop_destination.mkdir()
        write_jsonl(sop_destination / "clauses.jsonl", clauses)
        write_json_atomic(
            sop_destination / "containers.json",
            {"schema": "bundle_sop_containers_v1", "containers": containers},
        )
        write_json_atomic(
            sop_destination / "graph.json",
            {
                "schema": "bundle_sop_graph_v1",
                "containers": [container["sop_id"] for container in containers],
                "clauses": [clause["clause_id"] for clause in clauses],
                "edges": [
                    edge
                    for edge in graph.get("edges") or []
                    if edge.get("kind")
                    in {
                        "contains_clause",
                        "derived_into_clause",
                        "navigation_attached_to",
                        "authorized_distills_to",
                    }
                ],
            },
        )
        write_json_atomic(
            sop_destination / "taxonomy.json",
            {
                "schema": "sop_clause_taxonomy_v1",
                "publication_class_counts": dict(
                    sorted(
                        collections.Counter(
                            str(clause.get("publication_class")) for clause in clauses
                        ).items()
                    )
                ),
            },
        )
        visibility_destination = staging / "visibility"
        visibility_destination.mkdir()
        _copy_file(
            runforest_dir / "visibility" / "clause_metadata.jsonl",
            visibility_destination / "clause_metadata.jsonl",
        )
        mask_destination = visibility_destination / "precompiled_masks"
        mask_destination.mkdir()
        for path in sorted(
            (runforest_dir / "visibility" / "precompiled_masks").glob("*.json")
        ):
            _copy_file(path, mask_destination / path.name)
        _filter_authority(authority_dir, staging / "authority", clauses)

        splits_destination = staging / "splits"
        splits_destination.mkdir()
        _copy_file(split_manifest_path, splits_destination / "active.json")
        if splits_dir:
            for path in sorted(splits_dir.glob("*.json")):
                _copy_file(path, splits_destination / path.name)

        registry_destination = staging / "protocol_registry"
        registry_destination.mkdir()
        for path in sorted(protocol_registry_dir.glob("*.json")):
            _copy_file(path, registry_destination / path.name)
        protocol_registry_hash, _registry_hashes = _directory_hash(
            registry_destination
        )

        reports = staging / "reports"
        reports.mkdir()
        runforest_report = read_json(runforest_dir / "build_report.json")
        build_report = {
            "schema": "memory_bundle_build_report_v1",
            "bundle_id": bundle_id,
            "bundle_version": bundle_version,
            "split_id": split.split_id,
            "source_run_count": len(source_runs),
            "heldout_run_count": len(split.heldout_run_ids),
            "source_task_ids": runforest_report.get("source_task_ids"),
            "source_task_families": runforest_report.get(
                "source_task_families"
            ),
            "source_domains": runforest_report.get("source_domains"),
            "heldout_task_ids": runforest_report.get("heldout_task_ids"),
            "heldout_task_families": runforest_report.get(
                "heldout_task_families"
            ),
            "heldout_domains": runforest_report.get("heldout_domains"),
            "transfer_design": runforest_report.get("transfer_design"),
            "target_task_id": runforest_report.get("target_task_id"),
            "target_domain": runforest_report.get("target_domain"),
            "raw_journal_run_count": len(list((staging / "raw_journals").iterdir())),
            "sidecar_count": len(selected_sidecars),
            "clause_count": len(clauses),
            "container_count": len(containers),
            "spooky_source_run_count": runforest_report.get(
                "spooky_source_run_count"
            ),
            "all_code_nodes_have_sidecars": runforest_report.get(
                "all_code_nodes_have_sidecars"
            ),
            "all_clause_sources_resolve": runforest_report.get(
                "all_clause_sources_resolve"
            ),
            "all_included_clauses_have_domain_lineage": runforest_report.get(
                "all_included_clauses_have_domain_lineage"
            ),
            "cross_domain_included_clause_ids": runforest_report.get(
                "cross_domain_included_clause_ids"
            ),
            "heldout_run_refs_in_graph": runforest_report.get(
                "heldout_run_refs_in_graph"
            ),
            "legacy_artifact_overwritten": False,
            "secret_scan_passed": True,
            "corpus_drift_reviewed": True,
            "corpus_drift_review_disposition": drift_review.get("disposition"),
            "published_atomically": True,
        }
        write_json_atomic(reports / "build_report.json", build_report)

        artifact_hashes = {
            path.relative_to(staging).as_posix(): sha256_file(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        lineage_hash = sha256_json(
            {
                "clauses": artifact_hashes["sop/clauses.jsonl"],
                "containers": artifact_hashes["sop/containers.json"],
                "derivations": artifact_hashes["authority/derivations.jsonl"],
                "sop_graph": artifact_hashes["sop/graph.json"],
            }
        )
        bundle_manifest = MemoryBundleManifestV1(
            bundle_id=bundle_id,
            bundle_version=bundle_version,
            parent_bundle=parent_bundle,
            corpus_manifest_hash=corpus.manifest_sha256,
            protocol_registry_hash=protocol_registry_hash,
            authority_policy_version=authority_policy_version,
            detector_version=detector_version,
            deepseek_model=deepseek_model,
            deepseek_prompt_hash=deepseek_prompt_hash,
            graph_hashes={"runforest": artifact_hashes["runforest/graph.json"]},
            index_hashes={
                "runforest": artifact_hashes["runforest/index.npz"],
                "clauses": artifact_hashes["runforest/clause_index.npz"],
            },
            lineage_hash=lineage_hash,
            split_id=split.split_id,
            certification_level=certification_level,
            build_report="reports/build_report.json",
            created_at=created_at or utc_now(),
            artifact_hashes=artifact_hashes,
        ).finalize()
        write_json_atomic(staging / "manifest.json", bundle_manifest.as_dict())
        checksum_files = [
            path
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS"
        ]
        with (staging / "SHA256SUMS").open("w", encoding="utf-8") as handle:
            for path in checksum_files:
                handle.write(
                    f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        validation = validate_bundle(staging)
        write_json_atomic(reports / "validation_report.json", validation)
        if not validation["valid"]:
            raise ValueError(f"Bundle validation failed: {validation['errors']}")
        os.replace(staging, output_dir)
        return {
            "schema": "memory_bundle_publication_result_v1",
            "bundle_path": str(output_dir),
            "bundle_id": bundle_id,
            "bundle_version": bundle_version,
            "split_id": split.split_id,
            "manifest_sha256": bundle_manifest.manifest_sha256,
            "validation": validation,
            "published_atomically": True,
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--drift-review", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--runforest-dir", type=Path, required=True)
    parser.add_argument("--authority-dir", type=Path)
    parser.add_argument("--protocol-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--parent-bundle")
    parser.add_argument("--authority-policy-version", default="authority_v1")
    parser.add_argument("--detector-version", required=True)
    parser.add_argument("--deepseek-model", required=True)
    parser.add_argument("--deepseek-prompt-hash", required=True)
    parser.add_argument("--certification-level", default="raw_audited")
    parser.add_argument("--created-at")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    result = build_bundle(
        args.corpus_manifest,
        args.drift_review,
        args.split_manifest,
        args.audit_dir,
        args.runforest_dir,
        args.protocol_registry,
        args.output_dir,
        bundle_id=args.bundle_id,
        bundle_version=args.bundle_version,
        authority_policy_version=args.authority_policy_version,
        detector_version=args.detector_version,
        deepseek_model=args.deepseek_model,
        deepseek_prompt_hash=args.deepseek_prompt_hash,
        authority_dir=args.authority_dir,
        splits_dir=args.splits_dir,
        parent_bundle=args.parent_bundle,
        certification_level=args.certification_level,
        created_at=args.created_at,
    )
    if args.archive:
        result["archive"] = create_tar_zst(args.output_dir, args.archive)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
