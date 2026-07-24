"""Extract split-scoped, globally referenced branch traces from a corpus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[2]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
if str(MEMORY_BUNDLE) not in sys.path:
    sys.path.insert(0, str(MEMORY_BUNDLE))

from build_corpus_manifest import journal_nodes  # noqa: E402
from schema import (  # noqa: E402
    CorpusManifestV1,
    SplitManifestV1,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    write_json_atomic,
)


def metric_value(node: Mapping[str, Any]) -> Any:
    value = node.get("metric")
    return value.get("value") if isinstance(value, Mapping) else value


def _short(value: Any, limit: int = 800) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit]} …"


def _node_id(node: Mapping[str, Any], index: int) -> str:
    return str(node.get("id") or node.get("node_id") or index)


def node_ref(run_id: str, node_id: str) -> str:
    return f"run::{run_id}::node::{node_id}"


def transition_ref(run_id: str, parent_id: str, child_id: str) -> str:
    payload = f"{run_id}\0{parent_id}\0{child_id}"
    suffix = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"run::{run_id}::transition::{suffix}"


def load_audit_index(audit_dir: Path | None) -> dict[str, dict[str, Any]]:
    if audit_dir is None:
        return {}
    index_path = audit_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Audit sidecar index not found: {index_path}")
    index = read_json(index_path)
    output: dict[str, dict[str, Any]] = {}
    for artifact_id, filename in (index.get("entries") or {}).items():
        output[str(artifact_id)] = read_json(audit_dir / str(filename))
    return output


def render_branch(
    run_id: str,
    branch_id: str,
    nodes: list[dict[str, Any]],
    *,
    task: str,
    audit_sidecars: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    nodes = sorted(
        nodes,
        key=lambda node: (
            int(node.get("step") or 0),
            str(node.get("id") or node.get("node_id") or ""),
        ),
    )
    lines = [
        "# Manifest-Scoped Branch Trace",
        "",
        f"**Task**: {task}",
        f"**Run**: {run_id}",
        f"**Branch**: {branch_id}",
        "",
        "---",
    ]
    refs: list[dict[str, Any]] = []
    previous_id = ""
    for index, node in enumerate(nodes):
        identifier = _node_id(node, index)
        artifact_ref = node_ref(run_id, identifier)
        sidecar = audit_sidecars.get(artifact_ref, {})
        current_transition = (
            transition_ref(run_id, previous_id, identifier)
            if previous_id
            else ""
        )
        lines.extend(
            [
                "",
                f"## Turn {index + 1}",
                f"- node_ref: `{artifact_ref}`",
                f"- transition_ref: `{current_transition or 'root'}`",
                f"- stage: `{node.get('stage')}`",
                f"- buggy: `{bool(node.get('is_buggy'))}`",
                f"- metric: `{metric_value(node)}`",
                f"- audit_status: `{sidecar.get('status', 'unavailable')}`",
                f"- audit_issue_refs: `{json.dumps([issue.get('issue_code') for issue in sidecar.get('issues') or []])}`",
                f"- plan: {_short(node.get('plan'))}",
                f"- code_summary: {_short(node.get('code_summary'))}",
                f"- observation: {_short(node.get('analysis'))}",
                f"- failure: {_short(node.get('exc_info') or node.get('_term_out'))}",
            ]
        )
        refs.append(
            {
                "node_ref": artifact_ref,
                "transition_ref": current_transition or None,
                "node_id": identifier,
                "step": node.get("step"),
                "stage": node.get("stage"),
                "audit_status": sidecar.get("status", "unavailable"),
                "audit_sidecar_sha256": sidecar.get("sidecar_sha256", ""),
            }
        )
        previous_id = identifier
    return "\n".join(lines) + "\n", {
        "run_id": run_id,
        "branch_id": branch_id,
        "task_id": task,
        "refs": refs,
    }


def extract_branches(
    corpus_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    split_manifest_path: str | Path | None = None,
    audit_dir: str | Path | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    manifest_path = Path(corpus_manifest_path).resolve()
    manifest = CorpusManifestV1.from_dict(read_json(manifest_path))
    source_root = Path(manifest.source_root).resolve()
    split = None
    if split_manifest_path is not None:
        split = SplitManifestV1.from_dict(read_json(split_manifest_path))
        if split.corpus_manifest_hash != manifest.manifest_sha256:
            raise ValueError("Split manifest does not bind the corpus manifest")
        selected_run_ids = set(split.source_run_ids)
    else:
        selected_run_ids = {
            run.run_id for run in manifest.runs if run.status == "complete"
        }
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Trace output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    sidecars = load_audit_index(Path(audit_dir).resolve() if audit_dir else None)
    traces: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {
        "corpus_manifest": sha256_file(manifest_path),
    }
    if split_manifest_path is not None:
        input_hashes["split_manifest"] = sha256_file(split_manifest_path)
    if audit_dir is not None:
        input_hashes["audit_sidecar_index"] = sha256_file(
            Path(audit_dir) / "index.json"
        )
    selected = [run for run in manifest.runs if run.run_id in selected_run_ids]
    missing = selected_run_ids - {run.run_id for run in selected}
    if missing:
        raise ValueError(f"Split references unknown runs: {sorted(missing)}")
    for run in sorted(selected, key=lambda item: item.run_id):
        if run.status != "complete" or not run.journal_path:
            raise ValueError(f"Selected source run is not complete: {run.run_id}")
        journal_path = source_root / run.journal_path
        if sha256_file(journal_path) != run.artifact_hashes.get("journal"):
            raise ValueError(f"Journal drift for run {run.run_id}")
        nodes = journal_nodes(read_json(journal_path))
        branches: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            branch_id = str(node.get("branch_id") or "root")
            branches.setdefault(branch_id, []).append(node)
        run_dir = output_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        for branch_id in sorted(branches):
            text, trace = render_branch(
                run.run_id,
                branch_id,
                branches[branch_id],
                task=run.canonical_task_id,
                audit_sidecars=sidecars,
            )
            safe_branch = hashlib.sha256(branch_id.encode()).hexdigest()[:12]
            path = run_dir / f"branch-{safe_branch}.md"
            path.write_text(text, encoding="utf-8")
            trace["path"] = path.relative_to(output_dir).as_posix()
            trace["sha256"] = sha256_file(path)
            traces.append(trace)
    trace_manifest = {
        "schema": "branch_trace_manifest_v1",
        "created_at": created_at or utc_now(),
        "corpus_id": manifest.corpus_id,
        "corpus_manifest_hash": manifest.manifest_sha256,
        "split_id": split.split_id if split else "all-complete",
        "input_hashes": input_hashes,
        "run_count": len(selected),
        "trace_count": len(traces),
        "traces": traces,
    }
    trace_manifest["manifest_sha256"] = sha256_json(trace_manifest)
    write_json_atomic(output_dir / "trace_manifest.json", trace_manifest)
    return trace_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    report = extract_branches(
        args.corpus_manifest,
        args.out_dir,
        split_manifest_path=args.split_manifest,
        audit_dir=args.audit_dir,
        created_at=args.created_at,
    )
    print(
        json.dumps(
            {
                "run_count": report["run_count"],
                "trace_count": report["trace_count"],
                "trace_manifest": str(args.out_dir / "trace_manifest.json"),
                "manifest_sha256": report["manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
