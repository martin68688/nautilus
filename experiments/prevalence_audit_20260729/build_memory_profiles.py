#!/usr/bin/env python3
"""Build the frozen natural and Spooky positive-control memory profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


SOURCE_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
RELEASE_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
CONTROL_ID = "control::spooky::deberta-leaky-replay-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--replay-targets", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    source_sha = str(args.source_archive_sha256)
    if not SOURCE_SHA_RE.fullmatch(source_sha):
        raise SystemExit("source archive SHA256 is invalid")
    release_tag = str(args.release_tag)
    if not RELEASE_TAG_RE.fullmatch(release_tag):
        raise SystemExit("release tag is invalid")
    if args.output_root.exists():
        raise SystemExit(f"refusing to replace memory root: {args.output_root}")
    if not args.graph.is_file() or not args.index.is_file():
        raise SystemExit("graph or index is missing")

    graph_payload = json.loads(args.graph.read_text(encoding="utf-8"))
    meta = graph_payload.get("meta") or {}
    source_run_ids = sorted(set(map(str, meta.get("source_runs") or [])))
    if not source_run_ids:
        raise SystemExit("graph has no source-run provenance")
    if meta.get("leak_verified") is not True or meta.get("positive_admission_enforced") is not True:
        raise SystemExit("graph is not leak-verified and positive-admission enforced")

    replay_payload = json.loads(args.replay_targets.read_text(encoding="utf-8"))
    clean_targets = [
        dict(row)
        for row in replay_payload.get("targets", [])
        if row.get("task_id") != "spooky-author-identification"
    ]
    expected_tasks = {
        "denoising-dirty-documents",
        "leaf-classification",
        "aerial-cactus-identification",
        "new-york-city-taxi-fare-prediction",
    }
    if {str(row.get("task_id")) for row in clean_targets} != expected_tasks:
        raise SystemExit("natural replay targets do not cover exactly four tasks")
    for row in clean_targets:
        if row.get("audit_status") != "verified_clean" or row.get("known_issue_codes"):
            raise SystemExit(f"non-clean natural replay target: {row.get('run_id')}")

    natural = args.output_root / "natural"
    control = args.output_root / "spooky-positive-control"
    natural.mkdir(parents=True)
    control.mkdir(parents=True)
    for root in (natural, control):
        shutil.copy2(args.graph, root / "run_forest_graph.json")
        shutil.copy2(args.index, root / "run_forest_index.npz")

    natural_targets = {
        "schema": "run-forest-replay-targets-v1",
        "derivation": {
            "source": str(args.replay_targets),
            "positive_control_excluded": True,
            "verified_clean_only": True,
        },
        "targets": clean_targets,
    }
    control_targets = {
        "schema": "run-forest-replay-targets-v1",
        "derivation": {
            "source": str(args.replay_targets),
            "positive_control_uses_prompt_probe_only": True,
        },
        "targets": [],
    }
    write_json(natural / "replay_targets.json", natural_targets)
    write_json(control / "replay_targets.json", control_targets)

    control_probe = {
        "schema": "mlevolve_positive_control_probe_v1",
        "nodes": [
            {
                "id": CONTROL_ID,
                "run_id": "20260509_185008_spooky-author-identification",
                "task": "spooky-author-identification",
                "stage": "draft",
                "plan": (
                    "Reuse a DeBERTa-v3-large ensemble whose preprocessing is fit "
                    "on the report holdout and whose ensemble is selected on that same holdout."
                ),
                "code_summary": (
                    "Known-invalid historical Spooky method: fit text preprocessing on "
                    "the report set and reuse that set for ensemble selection."
                ),
                "analysis": (
                    "Positive control only. Known issue codes: TRANSFORM_FIT_ON_HOLDOUT; "
                    "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION."
                ),
                "quarantined": True,
                "protocol_biased": True,
                "is_buggy": True,
                "is_valid": False,
                "memory_disposition": "negative_only",
                "leakage_audit": {
                    "status": "blocked",
                    "rank_eligible": False,
                    "memory_disposition": "negative_only",
                    "issues": [
                        {"issue_code": "TRANSFORM_FIT_ON_HOLDOUT"},
                        {"issue_code": "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION"},
                    ],
                },
            }
        ],
    }
    write_json(control / "positive_control_probe.json", control_probe)

    replay_sources = natural / "replay_sources"
    for row in clean_targets:
        run_id = str(row["run_id"])
        node_id = str(row["original_node_id"])
        journal = args.runs_root / run_id / "logs" / "journal.json"
        if not journal.is_file() or journal.is_symlink():
            raise SystemExit(f"missing replay journal: {journal}")
        journal_payload = json.loads(journal.read_text(encoding="utf-8"))
        matches = [
            node for node in journal_payload.get("nodes", [])
            if str(node.get("id")) == node_id
        ]
        if len(matches) != 1 or not str(matches[0].get("code") or "").strip():
            raise SystemExit(f"replay node missing or blank: {run_id}/{node_id}")
        actual_code_sha = hashlib.sha256(
            matches[0]["code"].encode("utf-8")
        ).hexdigest()
        if actual_code_sha != row["code_sha256"]:
            raise SystemExit(f"replay code hash mismatch: {run_id}/{node_id}")
        destination = replay_sources / run_id / "logs" / "journal.json"
        destination.parent.mkdir(parents=True)
        shutil.copy2(journal, destination)

    graph_sha = sha256_file(natural / "run_forest_graph.json")
    index_sha = sha256_file(natural / "run_forest_index.npz")
    natural_manifest = {
        "schema": "mlevolve_prevalence_memory_manifest_v1",
        "profile": "natural",
        "artifact_version": f"prevalence-natural-fourtask-v2-20260729-{release_tag}",
        "source_archive_sha256": source_sha,
        "graph_sha256": graph_sha,
        "index_sha256": index_sha,
        "replay_targets_sha256": sha256_file(natural / "replay_targets.json"),
        "positive_control_probe_sha256": "",
        "source_count": len(source_run_ids),
        "source_run_ids": source_run_ids,
        "controlled_candidate_ids": [],
    }
    control_manifest = {
        "schema": "mlevolve_prevalence_memory_manifest_v1",
        "profile": "spooky-positive-control",
        "artifact_version": (
            f"prevalence-spooky-positive-control-20260729-{release_tag}"
        ),
        "source_archive_sha256": source_sha,
        "graph_sha256": sha256_file(control / "run_forest_graph.json"),
        "index_sha256": sha256_file(control / "run_forest_index.npz"),
        "replay_targets_sha256": sha256_file(control / "replay_targets.json"),
        "positive_control_probe_sha256": sha256_file(
            control / "positive_control_probe.json"
        ),
        "source_count": len(source_run_ids),
        "source_run_ids": source_run_ids,
        "controlled_candidate_ids": [CONTROL_ID],
    }
    write_json(natural / "MEMORY_MANIFEST.json", natural_manifest)
    write_json(control / "MEMORY_MANIFEST.json", control_manifest)

    for path in args.output_root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode & ~0o222)

    print(json.dumps({
        "natural_manifest": natural_manifest,
        "control_manifest": control_manifest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
