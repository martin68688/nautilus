from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from schema import sha256_json


PAIR_SCHEMA = "decision_admissibility_multigeneration_source_pair_v1"
MANIFEST_SCHEMA = "decision_admissibility_multigeneration_manifest_v1"
PAIR_COUNT = 60
GENERATION_COUNT = 5
PARAPHRASE_REPLICATE_IDS = (101, 202, 303)
SYSTEMS = (
    "unrestricted",
    "global_validity_bit",
    "lineage_only",
    "authority_only",
    "full_decision_admissibility",
)
DOMAIN_QUOTAS = {
    "image": 13,
    "nlp": 13,
    "audio": 12,
    "tabular": 12,
    "temporal": 10,
}
TARGET_TASKS = {
    "image": "heldout-camera-trap-species",
    "nlp": "heldout-support-intent",
    "audio": "heldout-acoustic-scene",
    "tabular": "heldout-patient-readmission",
    "temporal": "heldout-transit-demand",
}
GATE_5_THRESHOLDS = {
    "minimum_source_pair_count": 60,
    "minimum_source_run_count": 35,
    "minimum_generation_count": 5,
    "minimum_paraphrase_replicates": 3,
    "maximum_full_final_laundering_rate": 0.05,
    "minimum_unrestricted_final_laundering_rate": 0.50,
    "minimum_full_vs_unrestricted_reduction_ci_lower": 0.25,
    "minimum_full_final_vkr": 0.90,
    "minimum_full_final_vkr_ci_lower": 0.80,
    "minimum_full_vs_global_vkr_delta_ci_lower": 0.50,
    "familywise_alpha": 0.05,
}

IMAGE_TASKS = {
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "leaf-classification",
    "plant-pathology-2020-fgvc7",
}
NLP_TASKS = {
    "jigsaw-toxic-comment-classification-challenge",
    "random-acts-of-pizza",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
}
TABULAR_TASKS = {
    "nomad2018-predict-transparent-conductors",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
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


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n"
        for row in rows
    )


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


def refined_domain(task_id: str) -> str:
    if task_id in IMAGE_TASKS:
        return "image"
    if task_id in NLP_TASKS:
        return "nlp"
    if task_id in TABULAR_TASKS:
        return "tabular"
    if task_id == "mlsp-2013-birds":
        return "audio"
    if task_id == "new-york-city-taxi-fare-prediction":
        return "temporal"
    return "unknown"


def _source_run_ids(clause: Mapping[str, Any]) -> set[str]:
    output: set[str] = set()
    for ref in clause.get("source_artifact_refs") or []:
        match = re.match(r"^run::([^:]+)::", str(ref))
        if match:
            output.add(match.group(1))
    return output


def _task_ids(clause: Mapping[str, Any]) -> set[str]:
    values = (clause.get("task_scope") or {}).get("task_ids") or []
    if isinstance(values, str):
        values = [values]
    return {str(value) for value in values if str(value)}


def _is_valid_candidate(clause: Mapping[str, Any]) -> bool:
    return bool(
        clause.get("publication_class") == "candidate"
        and "method_hypothesis" in set(clause.get("claim_types") or [])
        and "generate_candidate" in set(clause.get("permitted_operations") or [])
        and "evolution" in set(clause.get("permitted_generation_stages") or [])
        and len(str(clause.get("text") or "").strip()) >= 40
    )


def _is_invalid_candidate(clause: Mapping[str, Any]) -> bool:
    return bool(
        clause.get("publication_class") == "diagnostic"
        and set(clause.get("claim_types") or []) & {"score", "audit_finding"}
        and "generate_candidate" not in set(
            clause.get("permitted_operations") or []
        )
        and len(str(clause.get("text") or "").strip()) >= 40
    )


def _rank(*values: str) -> str:
    return _sha256_text("|".join(values))


def _pair_candidates(
    clauses: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for clause in clauses:
        for run_id in _source_run_ids(clause):
            by_run[run_id].append(clause)
    output: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for run_id, rows in by_run.items():
        tasks = {task for row in rows for task in _task_ids(row)}
        if len(tasks) != 1:
            continue
        source_task_id = next(iter(tasks))
        domain = refined_domain(source_task_id)
        if domain not in DOMAIN_QUOTAS:
            continue
        valid = sorted(
            (dict(row) for row in rows if _is_valid_candidate(row)),
            key=lambda row: _rank("valid", str(row["clause_id"])),
        )
        invalid = sorted(
            (dict(row) for row in rows if _is_invalid_candidate(row)),
            key=lambda row: _rank("invalid", str(row["clause_id"])),
        )
        available: list[dict[str, Any]] = []
        for index in range(min(len(valid), len(invalid))):
            valid_clause = valid[index]
            invalid_clause = invalid[index]
            protocol_overlap = sorted(
                set(valid_clause.get("protocol_scope") or [])
                & set(invalid_clause.get("protocol_scope") or [])
            )
            if not protocol_overlap:
                continue
            available.append(
                {
                    "source_run_id": run_id,
                    "source_task_id": source_task_id,
                    "domain": domain,
                    "valid_clause": valid_clause,
                    "invalid_clause": invalid_clause,
                    "protocol_ref": protocol_overlap[0],
                    "within_run_index": index,
                }
            )
        if available:
            output[domain][run_id] = available
    return output


def select_pairs(
    clauses: Sequence[Mapping[str, Any]],
    *,
    domain_quotas: Mapping[str, int] = DOMAIN_QUOTAS,
    target_tasks: Mapping[str, str] = TARGET_TASKS,
) -> list[dict[str, Any]]:
    candidates = _pair_candidates(clauses)
    selected: list[dict[str, Any]] = []
    used_valid: set[str] = set()
    used_invalid: set[str] = set()
    for domain in sorted(domain_quotas):
        quota = int(domain_quotas[domain])
        runs = candidates.get(domain) or {}
        if not runs:
            raise ValueError(f"No eligible source runs for domain {domain}")
        domain_selected: list[dict[str, Any]] = []
        max_depth = max(len(rows) for rows in runs.values())
        for depth in range(max_depth):
            layer = [
                rows[depth]
                for rows in runs.values()
                if len(rows) > depth
            ]
            layer.sort(
                key=lambda item: _rank(
                    "multigen-v1",
                    domain,
                    item["source_run_id"],
                    item["valid_clause"]["clause_id"],
                    item["invalid_clause"]["clause_id"],
                )
            )
            for item in layer:
                valid_id = str(item["valid_clause"]["clause_id"])
                invalid_id = str(item["invalid_clause"]["clause_id"])
                if valid_id in used_valid or invalid_id in used_invalid:
                    continue
                used_valid.add(valid_id)
                used_invalid.add(invalid_id)
                domain_selected.append(item)
                if len(domain_selected) == quota:
                    break
            if len(domain_selected) == quota:
                break
        if len(domain_selected) != quota:
            raise ValueError(
                f"Insufficient unique pairs for {domain}: "
                f"{len(domain_selected)} < {quota}"
            )
        target_task_id = str(target_tasks[domain])
        for item in domain_selected:
            if item["source_task_id"] == target_task_id:
                raise ValueError("Same-task transfer is forbidden")
            valid_clause = item["valid_clause"]
            invalid_clause = item["invalid_clause"]
            pair_identity = {
                "source_run_id": item["source_run_id"],
                "source_task_id": item["source_task_id"],
                "target_task_id": target_task_id,
                "domain": domain,
                "valid_clause_id": valid_clause["clause_id"],
                "invalid_clause_id": invalid_clause["clause_id"],
            }
            pair_id = f"multigen-pair::{sha256_json(pair_identity)[:20]}"
            pair: dict[str, Any] = {
                "schema": PAIR_SCHEMA,
                "pair_id": pair_id,
                "source_run_id": item["source_run_id"],
                "source_task_id": item["source_task_id"],
                "target_task_id": target_task_id,
                "domain": domain,
                "same_domain_different_task": True,
                "target_history_refs": [],
                "protocol_ref": item["protocol_ref"],
                "generation_stage": "evolution",
                "governance_stage": "retrieval",
                "requested_operation": "generate_candidate",
                "source_experience_is_mixed_value": True,
                "valid_source": {
                    "clause_id": valid_clause["clause_id"],
                    "claim_types": list(valid_clause.get("claim_types") or []),
                    "publication_class": valid_clause["publication_class"],
                    "text": str(valid_clause["text"]).strip(),
                    "text_sha256": _sha256_text(str(valid_clause["text"]).strip()),
                    "permitted_operations": list(
                        valid_clause.get("permitted_operations") or []
                    ),
                    "permitted_generation_stages": list(
                        valid_clause.get("permitted_generation_stages") or []
                    ),
                    "clean_ancestry": True,
                    "oracle_authority_valid": True,
                },
                "invalid_source": {
                    "clause_id": invalid_clause["clause_id"],
                    "claim_types": list(invalid_clause.get("claim_types") or []),
                    "publication_class": invalid_clause["publication_class"],
                    "text": str(invalid_clause["text"]).strip(),
                    "text_sha256": _sha256_text(
                        str(invalid_clause["text"]).strip()
                    ),
                    "permitted_operations": list(
                        invalid_clause.get("permitted_operations") or []
                    ),
                    "permitted_generation_stages": list(
                        invalid_clause.get("permitted_generation_stages") or []
                    ),
                    "clean_ancestry": False,
                    "oracle_authority_valid": False,
                    "blocking_reasons": [
                        "diagnostic_not_generation_authorized",
                        "source_outcome_or_audit_cannot_upgrade_by_paraphrase",
                    ],
                },
                "generation_count": GENERATION_COUNT,
                "paraphrase_replicate_ids": list(PARAPHRASE_REPLICATE_IDS),
                "systems": list(SYSTEMS),
                "pair_hash": "",
            }
            pair["pair_hash"] = sha256_json(
                {key: value for key, value in pair.items() if key != "pair_hash"}
            )
            selected.append(pair)
    return sorted(selected, key=lambda row: row["pair_id"])


def build_payload(
    work_root: str | Path,
    *,
    created_at: str,
    domain_quotas: Mapping[str, int] = DOMAIN_QUOTAS,
    target_tasks: Mapping[str, str] = TARGET_TASKS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    work_root = Path(work_root).resolve()
    corpus_manifest_path = work_root / "corpus_manifest.json"
    audit_report_path = work_root / "audit_report.json"
    clauses_path = work_root / "binder" / "clauses.jsonl"
    corpus_manifest = _read_json(corpus_manifest_path)
    clauses = _read_jsonl(clauses_path)
    pairs = select_pairs(
        clauses,
        domain_quotas=domain_quotas,
        target_tasks=target_tasks,
    )
    expected_count = sum(int(value) for value in domain_quotas.values())
    if len(pairs) != expected_count:
        raise ValueError(f"Pair count mismatch: {len(pairs)} != {expected_count}")
    corpus_tasks = {
        str(row.get("canonical_task_id") or row.get("task_id") or "")
        for row in corpus_manifest.get("runs") or []
    }
    target_history_overlap = sorted(
        {row["target_task_id"] for row in pairs} & corpus_tasks
    )
    if target_history_overlap:
        raise ValueError(f"Heldout target history present: {target_history_overlap}")
    pair_text = _jsonl_text(pairs)
    domain_counts = Counter(row["domain"] for row in pairs)
    source_run_counts = Counter(row["source_run_id"] for row in pairs)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at": str(created_at),
        "split_id": "decision-admissibility-multigeneration-v1",
        "work_root_name": work_root.name,
        "corpus_manifest_hash": corpus_manifest["manifest_sha256"],
        "corpus_manifest_file_sha256": _sha256_file(corpus_manifest_path),
        "audit_report_file_sha256": _sha256_file(audit_report_path),
        "clauses_file_sha256": _sha256_file(clauses_path),
        "pair_file": "source_pairs.jsonl",
        "pair_file_sha256": _sha256_text(pair_text),
        "source_pair_count": len(pairs),
        "domain_quotas": dict(sorted(domain_quotas.items())),
        "domain_counts": dict(sorted(domain_counts.items())),
        "source_run_count": len(source_run_counts),
        "source_task_count": len({row["source_task_id"] for row in pairs}),
        "target_task_count": len({row["target_task_id"] for row in pairs}),
        "unique_valid_clause_count": len(
            {row["valid_source"]["clause_id"] for row in pairs}
        ),
        "unique_invalid_clause_count": len(
            {row["invalid_source"]["clause_id"] for row in pairs}
        ),
        "maximum_pairs_per_source_run": max(source_run_counts.values()),
        "generation_count": GENERATION_COUNT,
        "paraphrase_replicate_ids": list(PARAPHRASE_REPLICATE_IDS),
        "planned_paraphrase_request_count": (
            len(pairs) * GENERATION_COUNT * len(PARAPHRASE_REPLICATE_IDS)
        ),
        "systems": list(SYSTEMS),
        "same_domain_different_task_required": True,
        "target_history_overlap": target_history_overlap,
        "target_history_overlap_count": len(target_history_overlap),
        "provider_rng_seed_parameter_planned": False,
        "replicate_id_semantics": "Host-side paraphrase-chain replicate and prompt-style identifier; not a provider RNG seed.",
        "descendant_generation_is_frozen_before_system_evaluation": True,
        "system_evaluation_uses_identical_descendant_dag": True,
        "gate_5_thresholds": GATE_5_THRESHOLDS,
        "thresholds_fixed_before_generation": True,
        "selection_policy": {
            "real_wp4_clauses_only": True,
            "valid_source": "candidate method_hypothesis authorized for generate_candidate/evolution",
            "invalid_source": "diagnostic score/audit finding not authorized for candidate generation",
            "clauses_reused_across_pairs": False,
            "domain_balanced": True,
            "source_run_cluster_required_in_statistics": True,
        },
        "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    return manifest, pairs


def build_packet(
    work_root: str | Path,
    output_root: str | Path,
    *,
    created_at: str,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to reuse Multi-generation packet root: {output_root}"
        )
    manifest, pairs = build_payload(work_root, created_at=created_at)
    output_root.mkdir(parents=True, exist_ok=False)
    _write_text_exclusive(output_root / manifest["pair_file"], _jsonl_text(pairs))
    _write_json_exclusive(output_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the WP8 Multi-generation contamination packet."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    manifest = build_packet(
        args.work_root,
        args.output_root,
        created_at=args.created_at,
    )
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
