from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from schema import sha256_file, sha256_json, utc_now, write_json_atomic


CLAIM_TYPES = {
    "method_hypothesis",
    "debug_repair",
    "audit_finding",
    "score",
}
PUBLICATION_CLASSES = {"diagnostic", "candidate", "certified"}
POSITIVE_PUBLICATION_CLASSES = {"positive_result", "positive_adopted"}
PUBLICATION_ORDER = {"diagnostic": 0, "candidate": 1, "certified": 2}
SCORE_PATTERN = re.compile(
    r"(?i)\b(score|metric|auc|f1|macro[-_ ]?f1|rmse|mse|accuracy|log[-_ ]?loss)"
    r"\s*(?:of|was|is|=|:)?\s*-?\d+(?:\.\d+)?%?"
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    output = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row must be an object: {path}")
            output.append(dict(value))
    return output


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sanitize_retrieval_text(text: str, claim_type: str) -> str:
    text = str(text or "").strip()
    if claim_type == "score":
        return text
    return re.sub(r"\s+", " ", SCORE_PATTERN.sub("[score redacted]", text)).strip()


def operation_scope(claim_type: str, compiled_class: str) -> list[str]:
    if claim_type == "audit_finding":
        return ["inspect", "debug_hypothesis", "distill_diagnostic"]
    if claim_type == "debug_repair":
        operations = [
            "inspect",
            "debug_hypothesis",
            "repair_seed",
            "distill_diagnostic",
        ]
        if compiled_class in {"candidate", "certified"}:
            operations.append("distill_candidate")
        return operations
    if claim_type == "method_hypothesis":
        if compiled_class == "diagnostic":
            return ["inspect"]
        return ["inspect", "generate_candidate", "distill_candidate"]
    return ["inspect"]


def generation_scope(claim_type: str, compiled_class: str) -> list[str]:
    if claim_type in {"audit_finding", "debug_repair"}:
        return ["debug"]
    if claim_type == "method_hypothesis" and compiled_class != "diagnostic":
        return ["draft", "model_design", "improve", "evolution", "fusion"]
    return []


def publication_class(claim_type: str) -> str:
    if claim_type in {"audit_finding", "debug_repair", "score"}:
        return "diagnostic"
    return "candidate"


def compile_publication_class(claim_type: str, proposal: str) -> str:
    proposal = str(proposal or "").strip().lower()
    if proposal not in PUBLICATION_CLASSES:
        raise ValueError(f"invalid_publication_class:{proposal}")
    authority_ceiling = publication_class(claim_type)
    if PUBLICATION_ORDER[proposal] <= PUBLICATION_ORDER[authority_ceiling]:
        return proposal
    return authority_ceiling


def validate_positive_clause_payload(clause: Mapping[str, Any]) -> None:
    """Validate the extra lineage contract for typed positive SOP clauses.

    The legacy DeepSeek binder intentionally cannot mint these classes.  The
    host-owned writeback binder calls this validator after Authority has
    created the derived Method Claim and its trusted evidence path.
    """

    publication = str(clause.get("publication_class") or "")
    if publication not in POSITIVE_PUBLICATION_CLASSES:
        raise ValueError(f"not_a_typed_positive_clause:{publication}")
    claim_types = {str(value) for value in clause.get("claim_types") or []}
    if claim_types != {"method_hypothesis"}:
        raise ValueError("typed positive SOP must reference a Method Claim")
    for field in (
        "claim_refs",
        "source_artifact_refs",
        "protocol_scope",
        "authority_decision_refs",
        "receipt_refs",
        "derivation_refs",
    ):
        if not clause.get(field):
            raise ValueError(f"typed positive SOP missing {field}")
    contract = clause.get("contract_spec")
    if not isinstance(contract, Mapping) or contract.get("scope_widened") is not False:
        raise ValueError("typed positive SOP lacks a non-widening contract")
    if publication == "positive_adopted" and not str(
        contract.get("source_claim_id") or ""
    ):
        raise ValueError("Positive Adopted SOP lacks its source Adoption Claim")
    if publication == "positive_result" and contract.get(
        "positive_distillation_kind"
    ) != "result":
        raise ValueError("Positive Result SOP kind mismatch")
    if publication == "positive_adopted" and contract.get(
        "positive_distillation_kind"
    ) != "adopted":
        raise ValueError("Positive Adopted SOP kind mismatch")


def _string_list(
    value: Any,
    *,
    field: str,
    required: bool = False,
    allow_scalar: bool = False,
) -> list[str]:
    if value is None:
        output: list[str] = []
    elif isinstance(value, str):
        if not allow_scalar:
            raise ValueError(f"{field}_not_list")
        output = [value.strip()] if value.strip() else []
    elif isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{field}_contains_non_string")
        output = [item.strip() for item in value if item.strip()]
    else:
        raise ValueError(f"{field}_not_list")
    if required and not output:
        raise ValueError(f"missing_{field}")
    return output


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}::{sha256_json(payload)[:24]}"


def _trace_lookup(trace_manifest: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    output = {}
    for trace in trace_manifest.get("traces") or []:
        key = (str(trace["run_id"]), str(trace["branch_id"]))
        if key in output:
            raise ValueError(f"Duplicate trace key: {key}")
        output[key] = dict(trace)
    return output


def bind(
    proposals_path: str | Path,
    trace_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    active_protocol_ref: str,
    created_at: str | None = None,
) -> dict[str, Any]:
    proposals_path = Path(proposals_path).resolve()
    trace_manifest_path = Path(trace_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Binder output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals = read_jsonl(proposals_path)
    trace_manifest = json.loads(trace_manifest_path.read_text(encoding="utf-8"))
    traces = _trace_lookup(trace_manifest)
    clauses: list[dict[str, Any]] = []
    containers: list[dict[str, Any]] = []
    claims: dict[str, dict[str, Any]] = {}
    derivations: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    schema_normalizations: collections.Counter[str] = collections.Counter()
    seen_clause_ids: set[str] = set()
    for proposal in proposals:
        key = (str(proposal.get("run_id")), str(proposal.get("branch_id")))
        trace = traces.get(key)
        if trace is None:
            quarantine.append(
                {"reason": "missing_trace", "proposal": proposal}
            )
            continue
        available_node_refs = {
            str(row["node_ref"]) for row in trace.get("refs") or []
        }
        available_transition_refs = {
            str(row["transition_ref"])
            for row in trace.get("refs") or []
            if row.get("transition_ref")
        }
        available_refs = available_node_refs | available_transition_refs
        response = proposal.get("response") or {}
        for container_index, raw_container in enumerate(
            response.get("sop_containers") or []
        ):
            title = str(raw_container.get("title") or "").strip()
            container_payload = {
                "request_id": proposal.get("request_id"),
                "title": title,
                "task_id": proposal.get("task_id"),
                "container_index": container_index,
            }
            container_id = _stable_id("sop", container_payload)
            container_clause_ids: list[str] = []
            for clause_index, raw_clause in enumerate(
                raw_container.get("clauses") or []
            ):
                try:
                    if not isinstance(raw_clause, Mapping):
                        raise ValueError("clause_not_object")
                    text = str(raw_clause.get("text") or "").strip()
                    if not text:
                        raise ValueError("missing_clause_text")
                    claim_type = str(
                        raw_clause.get("claim_type_proposal") or ""
                    ).strip().lower()
                    if claim_type not in CLAIM_TYPES:
                        raise ValueError(f"invalid_claim_type:{claim_type}")
                    source_refs = set(
                        _string_list(
                            raw_clause.get("source_refs"),
                            field="source_refs",
                            required=True,
                        )
                    )
                    evidence_refs = set(
                        _string_list(
                            raw_clause.get("evidence_refs"),
                            field="evidence_refs",
                        )
                    )
                    if not source_refs:
                        raise ValueError("missing_source_refs")
                    unknown = (source_refs | evidence_refs) - available_refs
                    if unknown:
                        raise ValueError(f"unknown_refs:{sorted(unknown)}")
                    if evidence_refs - available_node_refs:
                        raise ValueError("evidence_ref_not_node")
                    claim_payload = {
                        "claim_type": claim_type,
                        "statement": text,
                        "source_refs": sorted(source_refs),
                        "task_id": proposal.get("task_id"),
                        "protocol_ref": active_protocol_ref,
                    }
                    claim_id = _stable_id("claim", claim_payload)
                    claim = {
                        "claim_id": claim_id,
                        "claim_type": claim_type,
                        "subject_artifact_id": sorted(source_refs)[0],
                        "task_scope": {"task_id": proposal.get("task_id")},
                        "method_fingerprint": "legacy-unavailable",
                        "protocol_ref": active_protocol_ref,
                        "statement": text,
                        "parent_claims": [],
                        "source_artifact_refs": sorted(source_refs),
                        "evidence_refs": sorted(evidence_refs),
                        "boundary": {"legacy_static_only": True},
                        "legacy_status": "legacy_static_only",
                    }
                    existing = claims.get(claim_id)
                    if existing is not None and existing != claim:
                        raise ValueError("claim_id_collision")
                    claims[claim_id] = claim
                    retrieval_text = sanitize_retrieval_text(
                        str(raw_clause.get("retrieval_text") or text),
                        claim_type,
                    )
                    proposal_class = str(
                        raw_clause.get("publication_class_proposal") or ""
                    ).strip().lower()
                    compiled_class = compile_publication_class(
                        claim_type,
                        proposal_class,
                    )
                    applies_raw = raw_clause.get("applies_when")
                    prevents_raw = raw_clause.get("prevents")
                    if isinstance(applies_raw, str):
                        schema_normalizations["applies_when_scalar_to_list"] += 1
                    if isinstance(prevents_raw, str):
                        schema_normalizations["prevents_scalar_to_list"] += 1
                    elif prevents_raw is None:
                        schema_normalizations["prevents_null_or_missing_to_empty"] += 1
                    applies_when = _string_list(
                        applies_raw,
                        field="applies_when",
                        required=True,
                        allow_scalar=True,
                    )
                    prevents = _string_list(
                        prevents_raw,
                        field="prevents",
                        allow_scalar=True,
                    )
                    clause_payload = {
                        "sop_id": container_id,
                        "text": text,
                        "retrieval_text": retrieval_text,
                        "claim_ref": claim_id,
                        "source_refs": sorted(source_refs),
                        "protocol_ref": active_protocol_ref,
                    }
                    clause_id = _stable_id("clause", clause_payload)
                    if clause_id in seen_clause_ids:
                        raise ValueError("duplicate_clause")
                    seen_clause_ids.add(clause_id)
                    derivation_id = (
                        f"distillation::{proposal.get('request_id')}::"
                        f"{container_index}::{clause_index}"
                    )
                    source_artifacts = sorted(source_refs & available_node_refs)
                    source_transitions = sorted(
                        source_refs & available_transition_refs
                    )
                    clause = {
                        "clause_id": clause_id,
                        "sop_id": container_id,
                        "text": text,
                        "retrieval_text": retrieval_text,
                        "claim_refs": [claim_id],
                        "claim_types": [claim_type],
                        "source_artifact_refs": source_artifacts,
                        "source_transition_refs": source_transitions,
                        "protocol_scope": [active_protocol_ref],
                        "task_scope": {"task_ids": [proposal.get("task_id")]},
                        "permitted_operations": operation_scope(
                            claim_type,
                            compiled_class,
                        ),
                        "permitted_generation_stages": generation_scope(
                            claim_type,
                            compiled_class,
                        ),
                        "permitted_governance_stages": ["retrieval"],
                        "publication_class_proposal": proposal_class,
                        "publication_class": compiled_class,
                        "authority_decision_refs": [],
                        "receipt_refs": [],
                        "derivation_refs": [derivation_id],
                        "protocol_agnostic": False,
                        "legacy_status": "native_v1",
                        "applies_when": applies_when,
                        "prevents": prevents,
                    }
                    clauses.append(clause)
                    container_clause_ids.append(clause_id)
                    derivations.append(
                        {
                            "derivation_id": derivation_id,
                            "request_id": proposal.get("request_id"),
                            "clause_id": clause_id,
                            "parent_refs": sorted(source_refs),
                            "publication_class_proposal": proposal_class,
                            "publication_class_compiled": compiled_class,
                            "scope_widened": PUBLICATION_ORDER[compiled_class]
                            > PUBLICATION_ORDER[proposal_class],
                            "transform": "deepseek_proposal_then_deterministic_bind",
                        }
                    )
                except Exception as error:
                    quarantine.append(
                        {
                            "reason": str(error),
                            "request_id": proposal.get("request_id"),
                            "container_index": container_index,
                            "clause_index": clause_index,
                            "raw_clause": raw_clause,
                        }
                    )
            containers.append(
                {
                    "sop_id": container_id,
                    "title": title,
                    "task_id": proposal.get("task_id"),
                    "clause_ids": container_clause_ids,
                    "request_id": proposal.get("request_id"),
                }
            )
    clauses.sort(key=lambda row: row["clause_id"])
    containers.sort(key=lambda row: row["sop_id"])
    derivations.sort(key=lambda row: row["derivation_id"])
    write_jsonl(output_dir / "clauses.jsonl", clauses)
    write_json_atomic(
        output_dir / "containers.json",
        {"schema": "sop_containers_v1", "containers": containers},
    )
    write_jsonl(
        output_dir / "claims.jsonl",
        sorted(claims.values(), key=lambda row: row["claim_id"]),
    )
    write_jsonl(output_dir / "derivations.jsonl", derivations)
    write_jsonl(output_dir / "quarantine.jsonl", quarantine)
    report = {
        "schema": "sop_clause_binder_report_v1",
        "created_at": created_at or utc_now(),
        "proposals_sha256": sha256_file(proposals_path),
        "trace_manifest_sha256": sha256_file(trace_manifest_path),
        "active_protocol_ref": active_protocol_ref,
        "container_count": len(containers),
        "clause_count": len(clauses),
        "claim_count": len(claims),
        "derivation_count": len(derivations),
        "quarantine_count": len(quarantine),
        "all_clause_sources_resolve": all(
            clause["source_artifact_refs"] or clause["source_transition_refs"]
            for clause in clauses
        ),
        "scope_widened_count": sum(
            bool(row["scope_widened"]) for row in derivations
        ),
        "schema_normalization_counts": dict(sorted(schema_normalizations.items())),
        "publication_class_proposal_counts": dict(
            sorted(
                collections.Counter(
                    clause["publication_class_proposal"] for clause in clauses
                ).items()
            )
        ),
        "compiled_publication_class_counts": dict(
            sorted(
                collections.Counter(
                    clause["publication_class"] for clause in clauses
                ).items()
            )
        ),
        "publication_class_override_count": sum(
            clause["publication_class_proposal"] != clause["publication_class"]
            for clause in clauses
        ),
        "publication_class_upgrade_count": sum(
            PUBLICATION_ORDER[clause["publication_class"]]
            > PUBLICATION_ORDER[clause["publication_class_proposal"]]
            for clause in clauses
        ),
        "publication_class_downgrade_count": sum(
            PUBLICATION_ORDER[clause["publication_class"]]
            < PUBLICATION_ORDER[clause["publication_class_proposal"]]
            for clause in clauses
        ),
        "clauses_sha256": sha256_file(output_dir / "clauses.jsonl"),
        "claims_sha256": sha256_file(output_dir / "claims.jsonl"),
        "derivations_sha256": sha256_file(output_dir / "derivations.jsonl"),
    }
    write_json_atomic(output_dir / "binder_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--trace-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-protocol", required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    report = bind(
        args.proposals,
        args.trace_manifest,
        args.output_dir,
        active_protocol_ref=args.active_protocol,
        created_at=args.created_at,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
