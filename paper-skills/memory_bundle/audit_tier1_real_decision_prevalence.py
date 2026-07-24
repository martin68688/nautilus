from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from schema import sha256_json


REPORT_SCHEMA = "decision_admissibility_real_prevalence_report_v1"
RECEIPT_SCHEMA = "decision_admissibility_real_prevalence_receipt_v1"
TOP_K = 5
GATE_THRESHOLDS = {
    "minimum_eligible_decisions": 200,
    "minimum_covered_domains": 3,
    "minimum_covered_stages": 3,
    "minimum_top5_any_mismatch_wilson_lower_95": 0.10,
    "minimum_top1_mismatch_wilson_lower_95": 0.02,
}
STAGE_OPERATIONS = {
    "draft": {"generate_candidate"},
    "model_design": {"generate_candidate"},
    "improve": {"generate_candidate"},
    "debug": {"repair_seed", "debug_hypothesis"},
    "evolution": {"generate_candidate"},
    "fusion": {"generate_candidate"},
}
TRACE_FIELDS = (
    "node_ref",
    "transition_ref",
    "stage",
    "buggy",
    "metric",
    "audit_status",
    "audit_issue_refs",
    "plan",
    "code_summary",
    "observation",
    "failure",
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_text_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(dict(value), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_text_exclusive(
        path,
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
    )


def _extract_trace_field(block: str, field: str) -> str:
    alternatives = "|".join(re.escape(value) for value in TRACE_FIELDS)
    match = re.search(
        rf"(?ms)^- {re.escape(field)}: (.*?)(?=^- (?:{alternatives}): |\Z)",
        block,
    )
    return " ".join((match.group(1) if match else "").split())


def load_real_decisions(
    trace_manifest_path: str | Path,
    trace_root: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifest_path = Path(trace_manifest_path).resolve()
    root = Path(trace_root).resolve()
    manifest = _read_json(manifest_path)
    by_node: dict[str, dict[str, Any]] = {}
    trace_hashes: dict[str, str] = {}
    for trace in manifest.get("traces") or []:
        path = root / str(trace["path"])
        observed_hash = _sha256_file(path)
        if observed_hash != trace["sha256"]:
            raise ValueError(f"Trace hash mismatch: {path}")
        trace_hashes[str(trace["path"])] = observed_hash
        blocks = re.split(r"(?m)^## Turn \d+\s*$", path.read_text(encoding="utf-8"))[1:]
        parsed_by_ref = {
            _extract_trace_field(block, "node_ref").strip("`"): block
            for block in blocks
        }
        for ref in trace.get("refs") or []:
            node_ref = str(ref["node_ref"])
            block = parsed_by_ref.get(node_ref, "")
            plan = _extract_trace_field(block, "plan")
            code_summary = _extract_trace_field(block, "code_summary")
            query = " ".join(value for value in (plan, code_summary) if value).strip()
            record = {
                "node_ref": node_ref,
                "run_id": trace["run_id"],
                "task_id": trace["task_id"],
                "stage": str(ref.get("stage") or "").lower(),
                "audit_status": str(ref.get("audit_status") or ""),
                "query_text": query,
                "query_text_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
                "query_character_count": len(query),
            }
            existing = by_node.get(node_ref)
            if existing is None or len(query) > len(existing["query_text"]):
                by_node[node_ref] = record
            elif (
                existing["run_id"] != record["run_id"]
                or existing["task_id"] != record["task_id"]
                or existing["stage"] != record["stage"]
            ):
                raise ValueError(f"Conflicting real decision record: {node_ref}")
    return sorted(by_node.values(), key=lambda row: row["node_ref"]), trace_hashes


def _task_domains(corpus_manifest: Mapping[str, Any]) -> dict[str, str]:
    domains: dict[str, str] = {}
    for run in corpus_manifest.get("runs") or []:
        if run.get("status") != "complete":
            continue
        task = str(run.get("canonical_task_id") or run.get("task_id") or "")
        family = str(run.get("task_family") or "")
        if not task or not family or family == "unknown":
            continue
        prior = domains.get(task)
        if prior is not None and prior != family:
            raise ValueError(f"Conflicting task family for {task}: {prior} vs {family}")
        domains[task] = family
    return domains


def _source_run_ids(clause: Mapping[str, Any]) -> set[str]:
    output = set()
    for ref in clause.get("source_artifact_refs") or []:
        match = re.match(r"^run::([^:]+)::", str(ref))
        if match:
            output.add(match.group(1))
    return output


def _clause_task_ids(clause: Mapping[str, Any]) -> set[str]:
    scope = clause.get("task_scope") or {}
    values = scope.get("task_ids") or []
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values if str(value)}


def _clause_text(clause: Mapping[str, Any]) -> str:
    applies = clause.get("applies_when") or []
    if isinstance(applies, str):
        applies = [applies]
    return " ".join(
        [
            str(clause.get("text") or ""),
            str(clause.get("retrieval_text") or ""),
            *(str(value) for value in applies),
        ]
    ).strip()


def _protocol_compatible(clause: Mapping[str, Any], active_protocol_ref: str) -> bool:
    return bool(
        clause.get("protocol_agnostic") is True
        or active_protocol_ref in set(clause.get("protocol_scope") or [])
    )


def _operation_authorized(clause: Mapping[str, Any], stage: str) -> bool:
    requested = STAGE_OPERATIONS.get(stage) or set()
    permitted = set(clause.get("permitted_operations") or [])
    return bool(requested & permitted)


def _wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    any_mismatch = sum(row["top_k_any_mismatch"] for row in rows)
    top1_mismatch = sum(row["top1_mismatch"] for row in rows)
    stage_mismatch = sum(row["top_k_stage_mismatch"] for row in rows)
    authority_invalid = sum(row["top_k_authority_invalid"] for row in rows)
    valid_available = sum(row["top_k_full_valid_available"] for row in rows)
    return {
        "decision_count": total,
        "top_k_any_mismatch_count": any_mismatch,
        "top_k_any_mismatch_rate": _ratio(any_mismatch, total),
        "top_k_any_mismatch_wilson_lower_95": _wilson_lower(any_mismatch, total),
        "top1_mismatch_count": top1_mismatch,
        "top1_mismatch_rate": _ratio(top1_mismatch, total),
        "top1_mismatch_wilson_lower_95": _wilson_lower(top1_mismatch, total),
        "top_k_stage_mismatch_count": stage_mismatch,
        "top_k_stage_mismatch_rate": _ratio(stage_mismatch, total),
        "top_k_authority_invalid_count": authority_invalid,
        "top_k_authority_invalid_rate": _ratio(authority_invalid, total),
        "top_k_full_valid_available_count": valid_available,
        "top_k_full_valid_available_rate": _ratio(valid_available, total),
    }


def audit_prevalence(
    work_root: str | Path,
    output_root: str | Path,
    *,
    created_at: str,
    top_k: int = TOP_K,
    gate_thresholds: Mapping[str, Any] = GATE_THRESHOLDS,
) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    work_root = Path(work_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to reuse prevalence root: {output_root}")
    trace_manifest_path = work_root / "traces" / "trace_manifest.json"
    clauses_path = work_root / "binder" / "clauses.jsonl"
    corpus_manifest_path = work_root / "corpus_manifest.json"
    audit_report_path = work_root / "audit_report.json"
    trace_manifest = _read_json(trace_manifest_path)
    corpus_manifest = _read_json(corpus_manifest_path)
    audit_report = _read_json(audit_report_path)
    clauses = _read_jsonl(clauses_path)
    all_run_nodes, trace_hashes = load_real_decisions(
        trace_manifest_path,
        work_root / "traces",
    )
    non_code_root_count = sum(row["stage"] == "root" for row in all_run_nodes)
    decisions = [row for row in all_run_nodes if row["stage"] != "root"]
    if len(decisions) != int(audit_report["expected_code_node_count"]):
        raise ValueError(
            f"Real decision count mismatch: {len(decisions)} vs "
            f"{audit_report['expected_code_node_count']}"
        )
    domains = _task_domains(corpus_manifest)
    active_protocol_ref = str(audit_report["active_protocol_ref"])
    clause_texts = [_clause_text(clause) for clause in clauses]
    query_texts = [row["query_text"] for row in decisions]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_./+-]*\b",
    )
    matrix = vectorizer.fit_transform([*clause_texts, *query_texts])
    clause_matrix = matrix[: len(clauses)]
    query_matrix = matrix[len(clauses) :]

    clause_metadata: list[dict[str, Any]] = []
    for index, clause in enumerate(clauses):
        task_ids = _clause_task_ids(clause)
        task_domains = {domains.get(task, "") for task in task_ids} - {""}
        clause_metadata.append(
            {
                "index": index,
                "clause_id": str(clause["clause_id"]),
                "task_ids": task_ids,
                "task_domains": task_domains,
                "source_run_ids": _source_run_ids(clause),
            }
        )

    receipts: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for decision_index, decision in enumerate(decisions):
        stage = decision["stage"]
        target_task = decision["task_id"]
        target_domain = domains.get(target_task, "")
        if stage not in STAGE_OPERATIONS:
            exclusions["unsupported_stage"] += 1
            continue
        if not decision["query_text"]:
            exclusions["empty_predecision_context"] += 1
            continue
        if not target_domain:
            exclusions["unknown_target_domain"] += 1
            continue
        candidates = [
            meta
            for meta in clause_metadata
            if meta["task_ids"]
            and target_task not in meta["task_ids"]
            and meta["task_domains"] == {target_domain}
            and decision["run_id"] not in meta["source_run_ids"]
        ]
        if not candidates:
            exclusions["no_same_domain_different_task_candidates"] += 1
            continue
        indices = [meta["index"] for meta in candidates]
        similarities = (
            clause_matrix[indices] @ query_matrix[decision_index].T
        ).toarray().ravel()
        ranked = sorted(
            (
                (float(score), meta)
                for score, meta in zip(similarities, candidates)
                if score > 0.0
            ),
            key=lambda pair: (-pair[0], pair[1]["clause_id"]),
        )[:top_k]
        if not ranked:
            exclusions["no_nonzero_relevance_candidates"] += 1
            continue
        top_candidates = []
        for rank, (score, meta) in enumerate(ranked, start=1):
            clause = clauses[meta["index"]]
            stage_match = stage in set(clause.get("permitted_generation_stages") or [])
            operation_authorized = _operation_authorized(clause, stage)
            protocol_compatible = _protocol_compatible(clause, active_protocol_ref)
            authority_valid = operation_authorized and protocol_compatible
            mismatch = not (stage_match and authority_valid)
            top_candidates.append(
                {
                    "rank": rank,
                    "clause_id": meta["clause_id"],
                    "relevance_score": round(score, 12),
                    "source_task_ids": sorted(meta["task_ids"]),
                    "source_run_count": len(meta["source_run_ids"]),
                    "same_domain": meta["task_domains"] == {target_domain},
                    "different_task": target_task not in meta["task_ids"],
                    "current_run_excluded": decision["run_id"] not in meta["source_run_ids"],
                    "stage_match": stage_match,
                    "operation_authorized": operation_authorized,
                    "protocol_compatible": protocol_compatible,
                    "authority_valid": authority_valid,
                    "mismatch": mismatch,
                    "publication_class": clause.get("publication_class"),
                    "claim_types": list(clause.get("claim_types") or []),
                    "permitted_generation_stages": list(
                        clause.get("permitted_generation_stages") or []
                    ),
                    "permitted_operations": list(
                        clause.get("permitted_operations") or []
                    ),
                }
            )
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "node_ref": decision["node_ref"],
            "run_id": decision["run_id"],
            "target_task_id": target_task,
            "target_domain": target_domain,
            "stage": stage,
            "audit_status": decision["audit_status"],
            "query_text_sha256": decision["query_text_sha256"],
            "query_character_count": decision["query_character_count"],
            "candidate_pool_count": len(candidates),
            "nonzero_candidate_count": int(
                sum(bool(score > 0) for score in similarities)
            ),
            "top_k": top_candidates,
            "top_k_any_mismatch": any(row["mismatch"] for row in top_candidates),
            "top1_mismatch": top_candidates[0]["mismatch"],
            "top_k_stage_mismatch": any(
                not row["stage_match"] for row in top_candidates
            ),
            "top_k_authority_invalid": any(
                not row["authority_valid"] for row in top_candidates
            ),
            "top_k_full_valid_available": any(
                row["stage_match"] and row["authority_valid"]
                for row in top_candidates
            ),
            "target_history_exposure_count": sum(
                target_task in row["source_task_ids"] for row in top_candidates
            ),
            "cross_domain_exposure_count": sum(
                not row["same_domain"] for row in top_candidates
            ),
            "receipt_hash": "",
        }
        receipt["receipt_hash"] = sha256_json(
            {key: value for key, value in receipt.items() if key != "receipt_hash"}
        )
        receipts.append(receipt)

    receipts.sort(key=lambda row: row["node_ref"])
    overall = _summarize_rows(receipts)
    by_stage = {
        key: _summarize_rows([row for row in receipts if row["stage"] == key])
        for key in sorted({row["stage"] for row in receipts})
    }
    by_domain = {
        key: _summarize_rows(
            [row for row in receipts if row["target_domain"] == key]
        )
        for key in sorted({row["target_domain"] for row in receipts})
    }
    thresholds = dict(gate_thresholds)
    gate_checks = {
        "eligible_decisions": len(receipts)
        >= int(thresholds["minimum_eligible_decisions"]),
        "covered_domains": len(by_domain)
        >= int(thresholds["minimum_covered_domains"]),
        "covered_stages": len(by_stage)
        >= int(thresholds["minimum_covered_stages"]),
        "top5_any_mismatch_prevalence": bool(
            overall["top_k_any_mismatch_wilson_lower_95"] is not None
            and overall["top_k_any_mismatch_wilson_lower_95"]
            >= float(thresholds["minimum_top5_any_mismatch_wilson_lower_95"])
        ),
        "top1_mismatch_prevalence": bool(
            overall["top1_mismatch_wilson_lower_95"] is not None
            and overall["top1_mismatch_wilson_lower_95"]
            >= float(thresholds["minimum_top1_mismatch_wilson_lower_95"])
        ),
    }
    gate_passed = all(gate_checks.values())

    output_root.mkdir(parents=True)
    receipts_path = output_root / "retrieval_receipts.jsonl"
    _write_jsonl_exclusive(receipts_path, receipts)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at": str(created_at),
        "work_root_name": work_root.name,
        "corpus_manifest_hash": corpus_manifest["manifest_sha256"],
        "trace_manifest_hash": trace_manifest["manifest_sha256"],
        "trace_file_hashes_hash": sha256_json(trace_hashes),
        "clauses_file_sha256": _sha256_file(clauses_path),
        "audit_report_sha256": _sha256_file(audit_report_path),
        "retrieval_model": {
            "kind": "sklearn_tfidf",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "norm": "l2",
            "top_k": top_k,
            "query_surface": "predecision_plan_plus_code_summary",
        },
        "transfer_scope": {
            "same_domain_required": True,
            "different_task_required": True,
            "current_run_excluded": True,
            "target_history_exposure_count": sum(
                row["target_history_exposure_count"] for row in receipts
            ),
            "cross_domain_exposure_count": sum(
                row["cross_domain_exposure_count"] for row in receipts
            ),
        },
        "real_run_node_count": len(all_run_nodes),
        "non_code_root_count": non_code_root_count,
        "real_code_node_count": len(decisions),
        "eligible_decision_count": len(receipts),
        "exclusion_counts": dict(sorted(exclusions.items())),
        "covered_task_count": len({row["target_task_id"] for row in receipts}),
        "covered_domain_count": len(by_domain),
        "covered_stage_count": len(by_stage),
        "overall": overall,
        "by_stage": by_stage,
        "by_domain": by_domain,
        "candidate_exposure_count": sum(len(row["top_k"]) for row in receipts),
        "retrieval_receipts_file": receipts_path.name,
        "retrieval_receipts_file_sha256": _sha256_file(receipts_path),
        "gate_1": {
            "name": "problem_prevalence",
            "thresholds_fixed_before_audit": True,
            "thresholds": thresholds,
            "checks": gate_checks,
            "passed": gate_passed,
            "status": "pass" if gate_passed else "fail",
        },
        "causal_or_downstream_performance_claimed": False,
        "limitations": [
            "This is a retrospective lexical retrieval probe over real run nodes, not an online intervention.",
            "TF-IDF prevalence does not establish downstream metric improvement.",
            "Nodes without a same-domain different-task nonzero candidate are excluded with explicit reasons.",
        ],
        "auditor_source_sha256": _sha256_file(Path(__file__).resolve()),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    _write_json_exclusive(output_root / "prevalence_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit real same-domain decision mismatch prevalence for WP8 Gate 1."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    report = audit_prevalence(
        args.work_root,
        args.output_root,
        created_at=args.created_at,
        top_k=args.top_k,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["gate_1"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
