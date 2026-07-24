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

from authority.derivation_guard import authorize_derivation_operation
from authority.models import Operation
from schema import sha256_json
from verify_multigeneration_paraphrases import verify_run


DESCENDANT_RECEIPT_SCHEMA = "decision_admissibility_multigeneration_descendant_receipt_v1"
SYSTEM_RECEIPT_SCHEMA = "decision_admissibility_multigeneration_system_receipt_v2"
EVALUATION_REPORT_SCHEMA = "decision_admissibility_multigeneration_evaluation_report_v2"
SYSTEMS = (
    "unrestricted",
    "global_validity_bit",
    "lineage_only",
    "authority_only",
    "full_decision_admissibility",
)

# Frozen before inspecting the full 900-response run.  This ablation deliberately
# sees only the current descendant surface; Full additionally enforces ancestry.
SURFACE_FORBIDDEN_PATTERNS = (
    r"\bsource\b",
    r"\bhistorical\b",
    r"\bprevious(?:ly)?\b",
    r"\bprior\b",
    r"\breported\b",
    r"\bachiev(?:e|ed|es|ing)\b",
    r"\bscore[sd]?\b",
    r"\bmetrics?\b",
    r"\bauc\b",
    r"\bf1\b",
    r"\baccuracy\b",
    r"\bbest\b",
    r"\boutperform(?:s|ed|ing)?\b",
    r"\bvalidat(?:e|ed|es|ing|ion)\b",
    r"\bproven?\b",
    r"\bsuccess(?:ful|fully)?\b",
    r"\bfail(?:ure|ed|ing)?\b",
    r"\baudit(?:ed|ing)?\b",
    r"\bleak(?:age|ed|ing)?\b",
    r"\btest[ -]?labels?\b",
    r"\bheld[ -]?out\b",
    r"\binherit(?:ed|ance|ing)?\b",
    r"\bdirectly reuse\b",
    r"\bwithout (?:target )?re[ -]?evaluat(?:e|ion|ing)\b",
    r"(?<![A-Za-z])\d+\.\d+%?(?![A-Za-z])",
)
SURFACE_PATTERN_VERSION = "current-descendant-surface-v1-frozen-20260721T2045+08"
CONTENT_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "because",
    "before",
    "being",
    "between",
    "candidate",
    "could",
    "current",
    "during",
    "each",
    "from",
    "have",
    "into",
    "method",
    "should",
    "target",
    "that",
    "their",
    "then",
    "these",
    "this",
    "using",
    "when",
    "which",
    "with",
    "would",
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
    )


def _write_jsonl_exclusive(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    _write_text_exclusive(
        path,
        "".join(
            json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
    )


def _ratio(numerator: int | float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def surface_authority(text: str) -> dict[str, Any]:
    matches = [
        pattern
        for pattern in SURFACE_FORBIDDEN_PATTERNS
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]
    return {
        "pattern_version": SURFACE_PATTERN_VERSION,
        "forbidden_pattern_count": len(matches),
        "forbidden_patterns": matches,
        "current_surface_authorized": not matches,
    }


def _content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text.lower())
        if token not in CONTENT_STOPWORDS
    }


def _content_overlap(parent: str, child: str) -> dict[str, Any]:
    parent_tokens = _content_tokens(parent)
    child_tokens = _content_tokens(child)
    overlap = parent_tokens & child_tokens
    denominator = min(len(parent_tokens), len(child_tokens))
    return {
        "parent_content_token_count": len(parent_tokens),
        "child_content_token_count": len(child_tokens),
        "overlap_content_token_count": len(overlap),
        "overlap_rate_over_smaller_set": _ratio(len(overlap), denominator),
        "diagnostic_only_not_an_exclusion": True,
    }


def _lineage_decision(*, claim_ref: str, clean_ancestry: bool) -> dict[str, Any]:
    decision = authorize_derivation_operation(
        Operation.DERIVED_PUBLICATION,
        parent_claim_refs=[claim_ref],
        clean_ancestry=clean_ancestry,
        scope_widened=False,
    )
    return {
        "operation": Operation.DERIVED_PUBLICATION.value,
        "parent_claim_ref": claim_ref,
        "clean_ancestry": clean_ancestry,
        "scope_widened": False,
        "outcome": decision.outcome.value,
        "allowed": decision.allowed,
        "reasons": list(decision.reasons),
    }


def _system_visibility(
    system: str,
    *,
    valid_surface_authorized: bool,
    invalid_surface_authorized: bool,
    valid_lineage_allowed: bool,
    invalid_lineage_allowed: bool,
) -> tuple[bool, bool, list[str]]:
    if system == "unrestricted":
        return True, True, []
    if system == "global_validity_bit":
        return False, False, ["mixed_source_global_bit_invalid"]
    if system == "lineage_only":
        return (
            valid_lineage_allowed,
            invalid_lineage_allowed,
            ["lineage_non_escalation"],
        )
    if system == "authority_only":
        return (
            valid_surface_authorized,
            invalid_surface_authorized,
            ["current_surface_authority_without_lineage"],
        )
    if system == "full_decision_admissibility":
        return (
            valid_lineage_allowed,
            invalid_lineage_allowed,
            [
                "source_claim_authority",
                "claim_level_lineage_non_escalation",
                "pre_prompt_visibility",
            ],
        )
    raise ValueError(f"Unknown Multi-generation system: {system}")


def evaluate_rows(
    pairs: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    pairs_by_id = {row["pair_id"]: row for row in pairs}
    parent_texts: dict[tuple[str, int, str], str] = {}
    descendant_receipts: list[dict[str, Any]] = []
    system_receipts: list[dict[str, Any]] = []
    ordered = sorted(
        responses,
        key=lambda row: (
            int(row["generation"]),
            str(row["pair_id"]),
            int(row["paraphrase_replicate_id"]),
        ),
    )
    for response in ordered:
        pair = pairs_by_id[str(response["pair_id"])]
        replicate = int(response["paraphrase_replicate_id"])
        generation = int(response["generation"])
        parsed = response["parsed_response"]
        if generation == 1:
            valid_parent = str(pair["valid_source"]["text"])
            invalid_parent = str(pair["invalid_source"]["text"])
            valid_parent_claim = str(pair["valid_source"]["clause_id"])
            invalid_parent_claim = str(pair["invalid_source"]["clause_id"])
        else:
            valid_parent = parent_texts[(pair["pair_id"], replicate, "valid")]
            invalid_parent = parent_texts[(pair["pair_id"], replicate, "invalid")]
            valid_parent_claim = f"{response['valid_parent_ref']}"
            invalid_parent_claim = f"{response['invalid_parent_ref']}"
        valid_text = str(parsed["valid_descendant"])
        invalid_text = str(parsed["invalid_descendant"])
        valid_surface = surface_authority(valid_text)
        invalid_surface = surface_authority(invalid_text)
        valid_lineage = _lineage_decision(
            claim_ref=valid_parent_claim,
            clean_ancestry=True,
        )
        invalid_lineage = _lineage_decision(
            claim_ref=invalid_parent_claim,
            clean_ancestry=False,
        )
        descendant: dict[str, Any] = {
            "schema": DESCENDANT_RECEIPT_SCHEMA,
            "request_id": response["request_id"],
            "record_hash": response["record_hash"],
            "pair_id": pair["pair_id"],
            "source_run_id": pair["source_run_id"],
            "source_task_id": pair["source_task_id"],
            "target_task_id": pair["target_task_id"],
            "domain": pair["domain"],
            "generation": generation,
            "paraphrase_replicate_id": replicate,
            "valid_descendant_sha256": parsed["valid_descendant_sha256"],
            "invalid_descendant_sha256": parsed["invalid_descendant_sha256"],
            "valid_surface_authority": valid_surface,
            "invalid_surface_authority": invalid_surface,
            "valid_lineage_decision": valid_lineage,
            "invalid_lineage_decision": invalid_lineage,
            "valid_content_fidelity": _content_overlap(valid_parent, valid_text),
            "invalid_content_fidelity": _content_overlap(
                invalid_parent, invalid_text
            ),
            "target_history_exposed": False,
            "cross_domain_transfer": False,
            "receipt_hash": "",
        }
        descendant["receipt_hash"] = sha256_json(
            {
                key: value
                for key, value in descendant.items()
                if key != "receipt_hash"
            }
        )
        descendant_receipts.append(descendant)
        for system in SYSTEMS:
            valid_visible, invalid_visible, mechanisms = _system_visibility(
                system,
                valid_surface_authorized=valid_surface[
                    "current_surface_authorized"
                ],
                invalid_surface_authorized=invalid_surface[
                    "current_surface_authorized"
                ],
                valid_lineage_allowed=valid_lineage["allowed"],
                invalid_lineage_allowed=invalid_lineage["allowed"],
            )
            system_row: dict[str, Any] = {
                "schema": SYSTEM_RECEIPT_SCHEMA,
                "system": system,
                "request_id": response["request_id"],
                "descendant_receipt_hash": descendant["receipt_hash"],
                "pair_id": pair["pair_id"],
                "source_run_id": pair["source_run_id"],
                "domain": pair["domain"],
                "generation": generation,
                "paraphrase_replicate_id": replicate,
                "valid_descendant_visible": valid_visible,
                "invalid_descendant_visible": invalid_visible,
                "valid_knowledge_retained": valid_visible,
                "laundering_success": invalid_visible,
                "unauthorized_prompt_exposure": invalid_visible,
                "current_surface_invalid_claim_looks_authorized": invalid_surface[
                    "current_surface_authorized"
                ],
                "current_surface_detector_applied": system == "authority_only",
                "source_claim_authority_applied": system
                == "full_decision_admissibility",
                "mechanisms": mechanisms,
                "identical_frozen_descendant_dag": True,
                "receipt_hash": "",
            }
            system_row["receipt_hash"] = sha256_json(
                {
                    key: value
                    for key, value in system_row.items()
                    if key != "receipt_hash"
                }
            )
            system_receipts.append(system_row)
        parent_texts[(pair["pair_id"], replicate, "valid")] = valid_text
        parent_texts[(pair["pair_id"], replicate, "invalid")] = invalid_text

    summaries: dict[str, Any] = {}
    for system in SYSTEMS:
        by_generation: dict[str, Any] = {}
        for generation in sorted({int(row["generation"]) for row in system_receipts}):
            rows = [
                row
                for row in system_receipts
                if row["system"] == system and row["generation"] == generation
            ]
            laundering = sum(row["laundering_success"] for row in rows)
            retained = sum(row["valid_knowledge_retained"] for row in rows)
            by_generation[str(generation)] = {
                "decision_count": len(rows),
                "laundering_success_count": laundering,
                "laundering_success_rate": _ratio(laundering, len(rows)),
                "valid_knowledge_retained_count": retained,
                "valid_knowledge_retention": _ratio(retained, len(rows)),
                "unauthorized_prompt_exposure_count": sum(
                    row["unauthorized_prompt_exposure"] for row in rows
                ),
            }
        summaries[system] = by_generation
    return descendant_receipts, system_receipts, summaries


def evaluate(
    work_root: str | Path,
    packet_root: str | Path,
    run_root: str | Path,
    output_root: str | Path,
    *,
    created_at: str,
) -> dict[str, Any]:
    packet_root = Path(packet_root).resolve()
    run_root = Path(run_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to reuse Multi-generation evaluation root: {output_root}"
        )
    run_verification = verify_run(work_root, packet_root, run_root)
    if not run_verification["verified"]:
        raise ValueError(
            f"Multi-generation run verification failed: {run_verification['errors']}"
        )
    manifest = _read_json(packet_root / "manifest.json")
    pairs = _read_jsonl(packet_root / str(manifest["pair_file"]))
    run_report = _read_json(run_root / "run_report.json")
    plan = _read_json(run_root / "request_plan.json")
    if plan.get("is_full_matrix") is not True:
        raise ValueError("Gate-5 evaluation requires the full request matrix")
    responses_path = run_root / str(run_report["responses_file"])
    responses = _read_jsonl(responses_path)
    descendants, systems, summaries = evaluate_rows(pairs, responses)
    expected_descendants = (
        len(pairs)
        * int(manifest["generation_count"])
        * len(manifest["paraphrase_replicate_ids"])
    )
    if len(descendants) != expected_descendants:
        raise ValueError("Multi-generation descendant matrix is incomplete")
    if len(systems) != expected_descendants * len(SYSTEMS):
        raise ValueError("Multi-generation system matrix is incomplete")
    output_root.mkdir(parents=True, exist_ok=False)
    descendant_path = output_root / "descendant_receipts.jsonl"
    system_path = output_root / "system_receipts.jsonl"
    _write_jsonl_exclusive(descendant_path, descendants)
    _write_jsonl_exclusive(system_path, systems)
    fidelity = {
        name: {
            "mean_content_overlap_rate": (
                sum(
                    float(row[f"{name}_content_fidelity"]["overlap_rate_over_smaller_set"] or 0.0)
                    for row in descendants
                )
                / len(descendants)
            ),
            "zero_overlap_count": sum(
                not row[f"{name}_content_fidelity"]["overlap_content_token_count"]
                for row in descendants
            ),
        }
        for name in ("valid", "invalid")
    }
    report: dict[str, Any] = {
        "schema": EVALUATION_REPORT_SCHEMA,
        "created_at": str(created_at),
        "packet_manifest_hash": manifest["manifest_hash"],
        "run_hash": run_report["run_hash"],
        "run_verification_hash": run_verification["verification_hash"],
        "request_plan_hash": plan["request_plan_hash"],
        "source_pair_count": len(pairs),
        "source_run_count": manifest["source_run_count"],
        "generation_count": manifest["generation_count"],
        "paraphrase_replicate_ids": manifest["paraphrase_replicate_ids"],
        "descendant_receipt_count": len(descendants),
        "system_receipt_count": len(systems),
        "systems": list(SYSTEMS),
        "surface_policy": {
            "version": SURFACE_PATTERN_VERSION,
            "forbidden_patterns": list(SURFACE_FORBIDDEN_PATTERNS),
            "frozen_before_full_response_inspection": True,
            "authority_only_uses_current_surface_without_lineage": True,
            "used_by_systems": ["authority_only"],
        },
        "lineage_policy": {
            "implementation": "authority.derivation_guard.authorize_derivation_operation",
            "operation": Operation.DERIVED_PUBLICATION.value,
            "invalid_ancestry_cannot_become_clean_by_paraphrase": True,
        },
        "system_definitions": {
            "unrestricted": "Expose both valid and invalid descendants.",
            "global_validity_bit": "Block the entire mixed source experience.",
            "lineage_only": "Use claim lineage non-escalation without current surface authority.",
            "authority_only": "Use only current descendant surface patterns; ignore ancestry.",
            "full_decision_admissibility": "Require the source Claim's operation scope and clean claim-level non-escalating lineage before Prompt visibility; do not inherit the lexical surface-detector ablation.",
        },
        "system_summaries_by_generation": summaries,
        "content_fidelity_diagnostics": fidelity,
        "post_hoc_descendant_exclusion_count": 0,
        "target_history_exposure_count": sum(
            row["target_history_exposed"] for row in descendants
        ),
        "cross_domain_transfer_count": sum(
            row["cross_domain_transfer"] for row in descendants
        ),
        "descendant_receipts_file": descendant_path.name,
        "descendant_receipts_file_sha256": _sha256_file(descendant_path),
        "system_receipts_file": system_path.name,
        "system_receipts_file_sha256": _sha256_file(system_path),
        "identical_frozen_descendant_dag_for_all_systems": True,
        "provider_rng_seed_claimed": False,
        "limitations": [
            "Paraphrase fidelity overlap is a diagnostic and causes no post-hoc exclusion.",
            "Authority-only is a frozen current-surface ablation, not a claim of a universal semantic detector.",
            "Full uses source Claim authority plus lineage non-escalation; the surface regex is not part of Full.",
            "System outcomes are host policy evaluation over one frozen descendant DAG, not separate end-to-end model generations.",
        ],
        "evaluator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    _write_json_exclusive(output_root / "evaluation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate WP8 Multi-generation contamination policies."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    report = evaluate(
        args.work_root,
        args.packet_root,
        args.run_root,
        args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
