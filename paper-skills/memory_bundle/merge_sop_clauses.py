from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from bind_sop_clauses import read_jsonl, write_jsonl
from schema import read_json, sha256_file, sha256_json, utc_now, write_json_atomic


STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "use",
    "when",
    "with",
}


def tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token not in STOPWORDS
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def cluster_containers(
    containers: list[dict[str, Any]], threshold: float
) -> list[list[dict[str, Any]]]:
    clusters: list[tuple[set[str], list[dict[str, Any]]]] = []
    for container in sorted(containers, key=lambda row: row["sop_id"]):
        signature = tokens(container.get("title", ""))
        task_id = str(container.get("task_id") or "")
        for representative, members in clusters:
            if str(members[0].get("task_id") or "") != task_id:
                continue
            if jaccard(signature, representative) >= threshold:
                members.append(container)
                representative.update(signature)
                break
        else:
            clusters.append((set(signature), [container]))
    return [members for _signature, members in clusters]


def merge(
    clauses_path: str | Path,
    containers_path: str | Path,
    output_dir: str | Path,
    *,
    threshold: float = 0.6,
    created_at: str | None = None,
) -> dict[str, Any]:
    clauses_path = Path(clauses_path).resolve()
    containers_path = Path(containers_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Merge output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    clauses = read_jsonl(clauses_path)
    clauses_by_id = {row["clause_id"]: row for row in clauses}
    payload = read_json(containers_path)
    containers = [dict(row) for row in payload.get("containers") or []]
    referenced = {
        clause_id
        for container in containers
        for clause_id in container.get("clause_ids") or []
    }
    missing = referenced - set(clauses_by_id)
    if missing:
        raise ValueError(f"Containers reference missing clauses: {sorted(missing)}")
    clusters = cluster_containers(containers, threshold)
    merged: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for members in clusters:
        member_ids = sorted(row["sop_id"] for row in members)
        merged_id = (
            member_ids[0]
            if len(member_ids) == 1
            else f"sop-merged::{sha256_json(member_ids)[:24]}"
        )
        titles = sorted(
            {str(row.get("title") or "") for row in members},
            key=lambda value: (len(value), value),
        )
        clause_ids = sorted(
            {
                clause_id
                for row in members
                for clause_id in row.get("clause_ids") or []
            }
        )
        merged.append(
            {
                "sop_id": merged_id,
                "title": titles[0] if titles else "",
                "task_id": members[0].get("task_id"),
                "clause_ids": clause_ids,
                "member_sop_ids": member_ids,
            }
        )
        for member_id in member_ids:
            lineage.append(
                {
                    "source_sop_id": member_id,
                    "merged_sop_id": merged_id,
                    "operation": "container_merge_only",
                    "clause_authority_changed": False,
                }
            )
    shutil.copyfile(clauses_path, output_dir / "clauses.jsonl")
    write_json_atomic(
        output_dir / "containers.json",
        {"schema": "merged_sop_containers_v1", "containers": merged},
    )
    write_jsonl(output_dir / "container_lineage.jsonl", lineage)
    output_clauses_hash = sha256_file(output_dir / "clauses.jsonl")
    if output_clauses_hash != sha256_file(clauses_path):
        raise ValueError("Container merge changed clause payloads")
    report = {
        "schema": "sop_container_merge_report_v1",
        "created_at": created_at or utc_now(),
        "threshold": threshold,
        "container_count_before": len(containers),
        "container_count_after": len(merged),
        "merged_cluster_count": sum(len(cluster) > 1 for cluster in clusters),
        "clause_count": len(clauses),
        "clause_payload_sha256_before": sha256_file(clauses_path),
        "clause_payload_sha256_after": output_clauses_hash,
        "clause_authority_changed": False,
    }
    write_json_atomic(output_dir / "merge_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clauses", type=Path, required=True)
    parser.add_argument("--containers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    report = merge(
        args.clauses,
        args.containers,
        args.output_dir,
        threshold=args.threshold,
        created_at=args.created_at,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
