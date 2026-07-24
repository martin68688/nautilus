from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from build_multigeneration_contamination_packet import (
    DOMAIN_QUOTAS,
    GATE_5_THRESHOLDS,
    GENERATION_COUNT,
    MANIFEST_SCHEMA,
    PAIR_COUNT,
    PAIR_SCHEMA,
    PARAPHRASE_REPLICATE_IDS,
    SYSTEMS,
    build_payload,
)
from schema import sha256_json


VERIFICATION_SCHEMA = "decision_admissibility_multigeneration_packet_verification_v1"


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


def _valid_hash(payload: Mapping[str, Any], field: str) -> bool:
    return payload.get(field) == sha256_json(
        {key: value for key, value in payload.items() if key != field}
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(dict(payload), sort_keys=True, ensure_ascii=False, indent=2)
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_packet(
    work_root: str | Path,
    packet_root: str | Path,
) -> dict[str, Any]:
    packet_root = Path(packet_root).resolve()
    manifest = _read_json(packet_root / "manifest.json")
    pair_path = packet_root / str(manifest.get("pair_file") or "")
    pairs = _read_jsonl(pair_path)
    errors: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("manifest_schema")
    if not _valid_hash(manifest, "manifest_hash"):
        errors.append("manifest_hash")
    builder_path = Path(__file__).resolve().with_name(
        "build_multigeneration_contamination_packet.py"
    )
    if manifest.get("builder_source_sha256") != _sha256_file(builder_path):
        errors.append("builder_source_hash")
    if manifest.get("pair_file_sha256") != _sha256_file(pair_path):
        errors.append("pair_file_hash")
    if len(pairs) != manifest.get("source_pair_count"):
        errors.append("source_pair_count")
    if len(pairs) != PAIR_COUNT:
        errors.append("source_pair_floor")
    if len({row.get("pair_id") for row in pairs}) != len(pairs):
        errors.append("duplicate_pair_id")
    if len({row.get("valid_source", {}).get("clause_id") for row in pairs}) != len(
        pairs
    ):
        errors.append("valid_clause_reuse")
    if len(
        {row.get("invalid_source", {}).get("clause_id") for row in pairs}
    ) != len(pairs):
        errors.append("invalid_clause_reuse")
    if Counter(row.get("domain") for row in pairs) != Counter(DOMAIN_QUOTAS):
        errors.append("domain_balance")
    for row in pairs:
        pair_id = str(row.get("pair_id") or "")
        if row.get("schema") != PAIR_SCHEMA:
            errors.append(f"pair_schema:{pair_id}")
        if not _valid_hash(row, "pair_hash"):
            errors.append(f"pair_hash:{pair_id}")
        if row.get("same_domain_different_task") is not True:
            errors.append(f"transfer_scope:{pair_id}")
        if row.get("source_task_id") == row.get("target_task_id"):
            errors.append(f"same_task:{pair_id}")
        if row.get("target_history_refs") != []:
            errors.append(f"target_history:{pair_id}")
        valid = row.get("valid_source") or {}
        invalid = row.get("invalid_source") or {}
        if valid.get("oracle_authority_valid") is not True:
            errors.append(f"valid_authority:{pair_id}")
        if invalid.get("oracle_authority_valid") is not False:
            errors.append(f"invalid_authority:{pair_id}")
        if valid.get("clean_ancestry") is not True:
            errors.append(f"valid_lineage:{pair_id}")
        if invalid.get("clean_ancestry") is not False:
            errors.append(f"invalid_lineage:{pair_id}")
        if "generate_candidate" not in set(valid.get("permitted_operations") or []):
            errors.append(f"valid_generation_scope:{pair_id}")
        if "generate_candidate" in set(
            invalid.get("permitted_operations") or []
        ):
            errors.append(f"invalid_generation_scope:{pair_id}")
        for name, source in (("valid", valid), ("invalid", invalid)):
            text = str(source.get("text") or "")
            if source.get("text_sha256") != hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest():
                errors.append(f"{name}_text_hash:{pair_id}")
    if manifest.get("domain_counts") != DOMAIN_QUOTAS:
        errors.append("manifest_domain_counts")
    if manifest.get("generation_count") != GENERATION_COUNT:
        errors.append("generation_count")
    if manifest.get("paraphrase_replicate_ids") != list(
        PARAPHRASE_REPLICATE_IDS
    ):
        errors.append("paraphrase_replicates")
    if manifest.get("systems") != list(SYSTEMS):
        errors.append("systems")
    if manifest.get("planned_paraphrase_request_count") != (
        PAIR_COUNT * GENERATION_COUNT * len(PARAPHRASE_REPLICATE_IDS)
    ):
        errors.append("request_count")
    if manifest.get("target_history_overlap_count") != 0:
        errors.append("target_history_overlap")
    if manifest.get("gate_5_thresholds") != GATE_5_THRESHOLDS:
        errors.append("gate_thresholds")
    if manifest.get("thresholds_fixed_before_generation") is not True:
        errors.append("threshold_preregistration")
    if manifest.get("provider_rng_seed_parameter_planned") is not False:
        errors.append("provider_seed_boundary")
    if manifest.get("descendant_generation_is_frozen_before_system_evaluation") is not True:
        errors.append("generation_freeze")
    try:
        expected_manifest, expected_pairs = build_payload(
            work_root,
            created_at=str(manifest.get("created_at") or ""),
        )
    except Exception as error:
        errors.append(f"packet_recompute_exception:{type(error).__name__}")
        expected_manifest = None
        expected_pairs = None
    if expected_manifest is not None and expected_manifest != manifest:
        errors.append("manifest_recompute")
    if expected_pairs is not None and expected_pairs != pairs:
        errors.append("pairs_recompute")

    verification: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "packet_root_name": packet_root.name,
        "source_pair_count": len(pairs),
        "source_run_count": len({row["source_run_id"] for row in pairs}),
        "domain_counts": dict(sorted(Counter(row["domain"] for row in pairs).items())),
        "generation_count": manifest.get("generation_count"),
        "paraphrase_replicate_count": len(
            manifest.get("paraphrase_replicate_ids") or []
        ),
        "planned_request_count": manifest.get("planned_paraphrase_request_count"),
        "target_history_overlap_count": manifest.get(
            "target_history_overlap_count"
        ),
        "verified": not errors,
        "errors": sorted(set(errors)),
        "manifest_hash": manifest.get("manifest_hash", ""),
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    verification["verification_hash"] = sha256_json(
        {
            key: value
            for key, value in verification.items()
            if key != "verification_hash"
        }
    )
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the WP8 Multi-generation contamination packet."
    )
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verification = verify_packet(args.work_root, args.packet_root)
    if args.output is not None:
        _write_json_exclusive(args.output.resolve(), verification)
    print(json.dumps(verification, sort_keys=True, ensure_ascii=False, indent=2))
    if not verification["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
