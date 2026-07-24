from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.clean_replay import build_replay_queue  # noqa: E402
from authority.memory_snapshot import ImmutableBaseBundle, sha256_file, sha256_json  # noqa: E402
from authority.replay_certifier import fingerprint_method  # noqa: E402
from build_corpus_manifest import journal_nodes  # noqa: E402
from schema import write_json_atomic  # noqa: E402


REPAIRABLE_PROTOCOL_TERMS = (
    "SPLIT",
    "FIT_SCOPE",
    "FIT_ON_HOLDOUT",
    "PREPROCESS",
    "EVALUATOR",
    "METRIC_DIRECTION",
    "HOLDOUT_REUSE",
    "SELECTION_FREEZE",
    "OOF",
    "CROSS_FIT",
    "TEMPORAL",
    "GROUP_OVERLAP",
    "PROVENANCE",
)
METHOD_FATAL_TERMS = (
    "TEST_LABEL",
    "TARGET_LEAK",
    "LABEL_FEATURE",
    "LABEL_DERIVED",
    "MODEL_SELECTION_ON_HOLDOUT",
    "REPORT_SET_REUSED_FOR_ENSEMBLE_SELECTION",
    "FORBIDDEN_DEPENDENCY",
)


def _node_id(node: Mapping[str, Any], index: int) -> str:
    return str(node.get("id") or node.get("node_id") or index)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        output.append(value)
    return output


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite replay artifact: {path}")
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _metric_value(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("value")
    if value is None:
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _historical_metric_delta(
    parent: Mapping[str, Any] | None,
    child: Mapping[str, Any],
) -> float | None:
    """Return a direction-normalized priority hint, never replay evidence."""

    if parent is None:
        return None
    parent_metric = _metric_value(parent.get("metric"))
    child_metric = _metric_value(child.get("metric"))
    if parent_metric is None or child_metric is None:
        return None
    maximize = child.get("metric_maximize")
    if maximize is None and isinstance(child.get("metric"), Mapping):
        maximize = child["metric"].get("maximize")
    return (
        child_metric - parent_metric
        if maximize is not False
        else parent_metric - child_metric
    )


def _fatal_issues(status: str, issue_codes: list[str]) -> list[str]:
    output = [
        code
        for code in issue_codes
        if any(term in code.upper() for term in METHOD_FATAL_TERMS)
    ]
    if str(status) in {"audit_unavailable", "unavailable"}:
        output.append("audit_unavailable")
    if str(status) == "blocked":
        output.extend(
            code
            for code in issue_codes
            if code not in output
            and not any(term in code.upper() for term in REPAIRABLE_PROTOCOL_TERMS)
        )
    return sorted(set(output))


def extract_candidates(bundle_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = ImmutableBaseBundle.load(bundle_dir, verify_artifacts=True)
    graph = base.read_json("runforest/graph.json")
    nodes = {str(node["id"]): dict(node) for node in graph.get("nodes") or []}
    claims = {
        str(row.get("claim_id") or ""): row
        for row in _jsonl(base.path / "authority" / "claims.jsonl")
        if row.get("claim_id")
    }
    transitions_by_child = {
        str(node.get("child_node_id")): dict(node)
        for node in nodes.values()
        if node.get("type") == "Transition" and node.get("child_node_id")
    }
    run_node_ids = {
        (str(node.get("run_id") or ""), str(node.get("raw_node_id") or "")): node_id
        for node_id, node in nodes.items()
        if node.get("type") == "RunNode"
        and node.get("run_id")
        and node.get("raw_node_id")
    }
    hypotheses: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for node in nodes.values():
        if node.get("type") != "SOPClause":
            continue
        claim_types = {str(value) for value in node.get("claim_types") or []}
        if not claim_types & {"method_hypothesis", "debug_repair"}:
            continue
        text = str(node.get("text") or "").strip()
        if not text:
            continue
        clause_id = str(node.get("clause_id") or node.get("id") or "")
        for reference in node.get("source_artifact_refs") or []:
            for claim_ref in sorted(
                {str(value) for value in node.get("claim_refs") or [] if value}
            ):
                claim = claims.get(claim_ref)
                if not claim:
                    continue
                if str(claim.get("claim_type") or "") not in {
                    "method_hypothesis",
                    "debug_repair",
                }:
                    continue
                if str(claim.get("subject_artifact_id") or "") != str(reference):
                    continue
                hypotheses[str(reference)].append(
                    {
                        "text": text,
                        "claim_id": claim_ref,
                        "clause_id": clause_id,
                    }
                )

    run_code: dict[tuple[str, str], str] = {}
    run_parent_raw_ids: dict[tuple[str, str], str] = {}
    run_ids = sorted(
        {
            str(node.get("run_id") or "")
            for node in nodes.values()
            if node.get("type") == "RunNode" and node.get("run_id")
        }
    )
    for run_id in run_ids:
        journal = base.read_json(f"raw_journals/{run_id}/journal.json")
        raw_nodes = list(journal_nodes(journal))
        node2parent = journal.get("node2parent") or {}
        if not isinstance(node2parent, Mapping):
            raise ValueError(f"Journal node2parent is not a mapping: {run_id}")
        for index, raw in enumerate(raw_nodes):
            raw_id = _node_id(raw, index)
            code = str(raw.get("code") or "")
            if code:
                run_code[(run_id, raw_id)] = code
            parent_raw_id = str(
                node2parent.get(raw_id)
                or raw.get("parent_id")
                or raw.get("parent_node_id")
                or ""
            )
            if parent_raw_id:
                run_parent_raw_ids[(run_id, raw_id)] = parent_raw_id

    candidates: list[dict[str, Any]] = []
    code_hash_mismatches: list[str] = []
    lineage_source_counts: collections.Counter[str] = collections.Counter()
    missing_parent_refs: list[str] = []
    for node_id, node in sorted(nodes.items()):
        if node.get("type") != "RunNode" or not str(node.get("code_sha256") or ""):
            continue
        transition = transitions_by_child.get(node_id)
        run_id = str(node.get("run_id") or "")
        raw_node_id = str(node.get("raw_node_id") or "")
        parent_id = str((transition or {}).get("parent_node_id") or "")
        lineage_source = "graph_transition" if parent_id else ""
        if not parent_id:
            parent_raw_id = run_parent_raw_ids.get((run_id, raw_node_id), "")
            parent_id = run_node_ids.get((run_id, parent_raw_id), "")
            if parent_id:
                lineage_source = "raw_journal_node2parent"
        if parent_id:
            lineage_source_counts[lineage_source] += 1
        else:
            missing_parent_refs.append(node_id)
        transition_id = str((transition or {}).get("id") or "")
        code = run_code.get((run_id, raw_node_id), "")
        actual_code_hash = hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""
        if actual_code_hash != str(node.get("code_sha256") or ""):
            code_hash_mismatches.append(node_id)
        audit = node.get("leakage_audit") or {}
        audit_status = str(audit.get("status") or "audit_unavailable")
        issue_codes = sorted(
            {
                str(issue.get("issue_code") or issue.get("category") or "unknown_issue")
                for issue in audit.get("issues") or []
                if isinstance(issue, Mapping)
            }
        )
        fatal = _fatal_issues(audit_status, issue_codes)
        method_bindings = sorted(
            hypotheses.get(node_id, []),
            key=lambda item: (
                item["text"], item["claim_id"], item["clause_id"]
            ),
        )
        method_binding = method_bindings[0] if method_bindings else {}
        method_hypothesis = str(method_binding.get("text") or "")
        original_claim_id = str(method_binding.get("claim_id") or "")
        source_clause_id = str(method_binding.get("clause_id") or "")
        try:
            fingerprint = fingerprint_method(code) if code else None
            families = tuple(fingerprint.model_families) if fingerprint else ()
            method_family = "+".join(families) or "unclassified"
        except SyntaxError:
            method_family = "unparseable"
            fatal.append("source_code_syntax_error")
        source_artifact_id = f"run::{node.get('run_id')}"
        source_refs = sorted(
            {
                value
                for value in (
                    source_artifact_id,
                    parent_id,
                    node_id,
                    transition_id,
                    original_claim_id,
                    source_clause_id,
                )
                if value
            }
        )
        candidate_payload = {
            "bundle_manifest_sha256": base.manifest_sha256,
            "node_id": node_id,
            "transition_id": transition_id,
            "parent_node_id": parent_id,
            "lineage_source": lineage_source,
            "code_sha256": actual_code_hash,
        }
        historical_metric_delta = (transition or {}).get("metric_improvement")
        if historical_metric_delta is None:
            historical_metric_delta = _historical_metric_delta(
                nodes.get(parent_id), node
            )
        candidates.append(
            {
                "candidate_id": f"replay_candidate::{sha256_json(candidate_payload)[:24]}",
                "task_id": str(node.get("task") or ""),
                "source_artifact_id": source_artifact_id,
                "parent_artifact_id": parent_id,
                "child_artifact_id": node_id,
                "original_claim_id": original_claim_id,
                "source_clause_id": source_clause_id,
                "code_sha256": actual_code_hash,
                "method_hypothesis": method_hypothesis,
                "method_family": method_family,
                "audit_status": (
                    "candidate_replay" if audit_status == "blocked" and not fatal else audit_status
                ),
                "source_refs": source_refs,
                "historical_metric_delta": historical_metric_delta,
                "method_fatal_issues": sorted(set(fatal)),
                "protocol_issue_codes": issue_codes,
            }
        )
    report = {
        "schema": "clean_replay_candidate_extraction_report_v1",
        "bundle_id": base.bundle_id,
        "bundle_manifest_sha256": base.manifest_sha256,
        "candidate_count": len(candidates),
        "task_count": len({row["task_id"] for row in candidates}),
        "code_hash_mismatch_count": len(code_hash_mismatches),
        "code_hash_mismatch_refs": code_hash_mismatches,
        "lineage_source_counts": dict(sorted(lineage_source_counts.items())),
        "missing_parent_count": len(missing_parent_refs),
        "missing_parent_refs": sorted(missing_parent_refs),
        "with_method_hypothesis_count": sum(bool(row["method_hypothesis"]) for row in candidates),
        "with_bound_original_claim_count": sum(
            bool(row["original_claim_id"] and row["source_clause_id"])
            for row in candidates
        ),
        "method_fatal_count": sum(bool(row["method_fatal_issues"]) for row in candidates),
        "historical_metric_used_as_evidence": False,
        "candidate_rows_sha256": sha256_json(candidates),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return candidates, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and select a deterministic Clean Replay queue from a verified WP4 Bundle."
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--queue-manifest", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--max-per-task", type=int, default=3)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    for path in (args.candidates, args.queue, args.queue_manifest, args.report):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite replay artifact: {path}")
    candidates, report = extract_candidates(args.bundle)
    if report["code_hash_mismatch_count"]:
        raise ValueError("Bundle replay extraction found a code hash mismatch")
    queue = build_replay_queue(
        candidates,
        max_per_task=args.max_per_task,
        created_at=args.created_at,
    )
    _write_jsonl(args.candidates, candidates)
    queue.write(args.queue, args.queue_manifest)
    report.update(
        {
            "selected_count": len(queue.entries),
            "rejected_count": len(queue.rejected),
            "queue_manifest_sha256": queue.manifest_sha256,
        }
    )
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    write_json_atomic(args.report, report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
