from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.leakage_audit import (  # noqa: E402
    AUDIT_SCHEMA,
    DETECTOR_VERSION,
    audit_code,
    code_sha256,
)
from schema import (  # noqa: E402
    AuditSidecarV1,
    CorpusManifestV1,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)
from build_corpus_manifest import journal_nodes  # noqa: E402


def stable_artifact_id(run_id: str, node_id: str) -> str:
    return f"run::{run_id}::node::{node_id}"


def sidecar_filename(artifact_id: str) -> str:
    return f"{hashlib.sha256(artifact_id.encode()).hexdigest()}.json"


def audit_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    active_protocol_ref: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    manifest = CorpusManifestV1.from_dict(read_json(manifest_path))
    source_root = Path(manifest.source_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Audit output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    created = generated_at or utc_now()
    index: dict[str, str] = {}
    status_counts: collections.Counter[str] = collections.Counter()
    issue_counts: collections.Counter[str] = collections.Counter()
    audited_runs: list[str] = []
    expected_code_nodes = 0
    sidecars: list[AuditSidecarV1] = []
    for run in manifest.runs:
        if run.status != "complete":
            continue
        expected_code_nodes += run.code_node_count
        if not run.journal_path:
            raise ValueError(f"Complete run has no journal path: {run.run_id}")
        journal_path = source_root / run.journal_path
        expected_hash = run.artifact_hashes.get("journal")
        actual_hash = sha256_file(journal_path)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"Journal hash drift for run {run.run_id}")
        payload = read_json(journal_path)
        nodes = journal_nodes(payload)
        run_code_count = 0
        for index_in_journal, node in enumerate(nodes):
            code = str(node.get("code") or "")
            if not code.strip():
                continue
            run_code_count += 1
            node_id = str(node.get("id") or node.get("node_id") or index_in_journal)
            artifact_id = stable_artifact_id(run.run_id, node_id)
            audit = audit_code(code)
            issues = [
                dict(issue)
                for issue in audit.get("issues") or []
                if isinstance(issue, Mapping)
            ]
            status = str(audit.get("status") or "unavailable")
            sidecar = AuditSidecarV1(
                artifact_id=artifact_id,
                run_id=run.run_id,
                node_id=node_id,
                code_sha256=code_sha256(code),
                detector_schema=str(audit.get("schema") or AUDIT_SCHEMA),
                detector_version=str(
                    audit.get("detector_version") or DETECTOR_VERSION
                ),
                active_protocol_ref=active_protocol_ref,
                status=status,
                issues=issues,
                legacy_receipt_level="legacy_static_only",
                generated_at=created,
                source_journal_sha256=actual_hash,
            ).finalize()
            filename = sidecar_filename(artifact_id)
            write_json_atomic(output_dir / filename, sidecar.as_dict())
            index[artifact_id] = filename
            sidecars.append(sidecar)
            status_counts[status] += 1
            for issue in issues:
                issue_counts[str(issue.get("issue_code") or "unknown")] += 1
        if run_code_count != run.code_node_count:
            raise ValueError(
                f"Code-node count drift for {run.run_id}: "
                f"manifest={run.code_node_count} observed={run_code_count}"
            )
        audited_runs.append(run.run_id)
    if len(sidecars) != expected_code_nodes:
        raise ValueError(
            f"Sidecar completeness failure: expected={expected_code_nodes} "
            f"actual={len(sidecars)}"
        )
    write_json_atomic(
        output_dir / "index.json",
        {
            "schema": "audit_sidecar_index_v1",
            "corpus_manifest_hash": manifest.manifest_sha256,
            "detector_version": DETECTOR_VERSION,
            "entries": dict(sorted(index.items())),
        },
    )
    return {
        "schema": "corpus_audit_report_v1",
        "corpus_id": manifest.corpus_id,
        "corpus_manifest_hash": manifest.manifest_sha256,
        "active_protocol_ref": active_protocol_ref,
        "detector_schema": AUDIT_SCHEMA,
        "detector_version": DETECTOR_VERSION,
        "complete_run_count": len(audited_runs),
        "audited_run_ids": sorted(audited_runs),
        "expected_code_node_count": expected_code_nodes,
        "sidecar_count": len(sidecars),
        "status_counts": dict(sorted(status_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "all_code_nodes_have_sidecars": len(sidecars) == expected_code_nodes,
        "source_journals_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create immutable static-audit sidecars without modifying journals."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-registry", type=Path)
    parser.add_argument("--default-protocol", default="mlevolve-default@1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args()
    report = audit_manifest(
        args.manifest,
        args.output_dir,
        active_protocol_ref=args.default_protocol,
        generated_at=args.generated_at,
    )
    write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
