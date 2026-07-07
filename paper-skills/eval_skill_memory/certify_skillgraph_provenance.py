"""Attach clean-run provenance back onto a compact SkillGraph-C artifact.

The current SkillGraph-C compact graph intentionally dropped non-paper fields
such as source_branches/evidence_turns. Hyperbolic memory needs those fields for
paper-grade provenance. This script joins the compact graph with the normalized
node file that still carries evidence, verifies every source run against a
machine-readable allowlist, and writes a certified graph for the hyperbolic
builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = REPO / "paper-skills" / "distillation" / "graph_build" / "graph_skillgraph_c_trace_prereq.json"
DEFAULT_SOURCE_NODES = REPO / "paper-skills" / "distillation" / "graph_build" / "merged_nodes_general_normalized.json"
DEFAULT_ALLOWLIST = REPO / "paper-skills" / "eval_skill_memory" / "clean_run_allowlist.json"
DEFAULT_OUTPUT = REPO / "paper-skills" / "eval_skill_memory" / "artifacts" / "graph_skillgraph_c_trace_prereq_certified.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _entries(allowlist: Any) -> list[dict[str, Any]]:
    if isinstance(allowlist, list):
        return allowlist
    if isinstance(allowlist, dict):
        return list(allowlist.get("entries", []))
    return []


def allowlist_hash(allowlist: Any) -> str:
    payload = json.dumps(allowlist, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def allowed_run_ids(allowlist: Any) -> set[str]:
    return {
        str(entry.get("run_id"))
        for entry in _entries(allowlist)
        if entry.get("allowed") is True and str(entry.get("run_id", "")).strip()
    }


def source_runs_for_node(node: dict[str, Any]) -> set[str]:
    runs = set()
    for item in node.get("source_branches", []) or []:
        if isinstance(item, (list, tuple)) and item:
            runs.add(str(item[0]))
        elif isinstance(item, str) and item:
            runs.add(item.split(":", 1)[0])
    return runs


def certify(
    *,
    graph_path: Path,
    source_nodes_path: Path,
    allowlist_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    graph = _load_json(graph_path)
    source_data = _load_json(source_nodes_path)
    allowlist = _load_json(allowlist_path)

    graph_nodes = graph.get("nodes", [])
    source_nodes = source_data.get("nodes", source_data if isinstance(source_data, list) else [])
    by_id = {str(n.get("id")): n for n in source_nodes if n.get("id")}
    allowed = allowed_run_ids(allowlist)
    if not allowed:
        raise ValueError(f"allowlist has no allowed runs: {allowlist_path}")

    missing_source_nodes: list[str] = []
    missing_source_evidence: list[str] = []
    disallowed_sources: list[dict[str, Any]] = []
    used_runs: set[str] = set()
    certified_nodes: list[dict[str, Any]] = []

    for node in graph_nodes:
        nid = str(node.get("id"))
        src = by_id.get(nid)
        if not src:
            missing_source_nodes.append(nid)
            continue
        source_branches = src.get("source_branches", []) or []
        evidence_turns = src.get("evidence_turns", []) or []
        if not source_branches and not evidence_turns:
            missing_source_evidence.append(nid)
            continue
        node_runs = source_runs_for_node(src)
        bad = sorted(node_runs - allowed)
        if bad:
            disallowed_sources.append({"node_id": nid, "run_ids": bad})
        used_runs.update(node_runs)

        enriched = dict(node)
        enriched["source_branches"] = source_branches
        enriched["evidence_turns"] = evidence_turns
        for optional in ("absorbed_node_ids", "absorbed_titles", "source_task_types", "general_normalization_action"):
            if optional in src:
                enriched[optional] = src[optional]
        certified_nodes.append(enriched)

    if missing_source_nodes or missing_source_evidence or disallowed_sources:
        raise ValueError(
            "Cannot certify graph provenance: "
            f"missing_source_nodes={missing_source_nodes[:10]} "
            f"missing_source_evidence={missing_source_evidence[:10]} "
            f"disallowed_sources={disallowed_sources[:10]}"
        )

    meta = dict(graph.get("meta", {}) or {})
    meta.update(
        {
            "source_runs": sorted(used_runs),
            "allowlist": sorted(allowed),
            "allowlist_hash": allowlist_hash(allowlist),
            "allowlist_path": str(allowlist_path),
            "leak_verified": True,
            "provenance_certifier": "certify_skillgraph_provenance.py",
            "provenance_source_nodes": str(source_nodes_path),
            "paper_grade_provenance": True,
        }
    )
    certified = {"meta": meta, "nodes": certified_nodes, "edges": graph.get("edges", [])}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certified, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "status": "clean_certified",
        "output": str(output_path),
        "nodes": len(certified_nodes),
        "edges": len(certified["edges"]),
        "source_runs": sorted(used_runs),
        "allowlist_hash": meta["allowlist_hash"],
        "nodes_with_source_evidence": len(certified_nodes),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify SkillGraph-C provenance for hyperbolic memory.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--source-nodes", type=Path, default=DEFAULT_SOURCE_NODES)
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = certify(
        graph_path=args.graph,
        source_nodes_path=args.source_nodes,
        allowlist_path=args.allowlist,
        output_path=args.output,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
