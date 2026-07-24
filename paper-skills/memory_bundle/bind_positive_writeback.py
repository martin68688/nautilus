from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.bundle_publisher import classify_writeback_events  # noqa: E402
from authority.memory_snapshot import (  # noqa: E402
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from authority.protocol_registry import ProtocolRegistry  # noqa: E402
from authority.writeback_distillation import (  # noqa: E402
    build_positive_writeback_plan,
    materialize_positive_writeback,
)
from bind_sop_clauses import validate_positive_clause_payload  # noqa: E402


BINDER_REPORT_SCHEMA = "positive_writeback_binder_report_v1"


def _write_jsonl_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _proposals(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("proposals") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("Positive proposals must be a list")
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Positive proposal row must be an object")
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id or candidate_id in output:
            raise ValueError("Positive proposal candidate ID is missing/duplicate")
        output[candidate_id] = dict(row)
    return output


def bind(
    overlay_snapshot: str | Path,
    proposals_path: str | Path,
    protocol_registry_path: str | Path,
    output_dir: str | Path,
    *,
    policy_version: str,
    collector_version: str = "1",
) -> dict[str, Any]:
    overlay_snapshot = Path(overlay_snapshot).resolve()
    proposals_path = Path(proposals_path).resolve()
    protocol_registry_path = Path(protocol_registry_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Positive writeback binder output is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory = classify_writeback_events(overlay_snapshot)
    plan = build_positive_writeback_plan(inventory)
    proposal_map = _proposals(proposals_path)
    expected_candidates = {
        str(row["candidate_id"]) for row in plan.get("items") or []
    }
    if set(proposal_map) != expected_candidates:
        raise ValueError(
            "Positive proposal IDs do not exactly match the typed writeback plan"
        )
    material = materialize_positive_writeback(
        plan,
        proposal_map,
        registry=ProtocolRegistry(protocol_registry_path),
        policy_version=str(policy_version),
        collector_version=str(collector_version),
    )
    for clause in material.clauses:
        validate_positive_clause_payload(clause)
    write_json_atomic(output_dir / "inventory.json", inventory)
    write_json_atomic(output_dir / "plan.json", material.plan)
    write_json_atomic(
        output_dir / "containers.json",
        {
            "schema": "positive_writeback_sop_containers_v1",
            "containers": list(material.containers),
        },
    )
    _write_jsonl_atomic(output_dir / "clauses.jsonl", material.clauses)
    _write_jsonl_atomic(
        output_dir / "claims.jsonl",
        [
            row
            for _key, row in sorted(
                (material.authority_snapshot.get("claims") or {}).items()
            )
        ],
    )
    _write_jsonl_atomic(
        output_dir / "receipts.jsonl",
        [
            row
            for _key, row in sorted(
                (material.authority_snapshot.get("receipts") or {}).items()
            )
        ],
    )
    _write_jsonl_atomic(
        output_dir / "paths.jsonl",
        [
            row
            for _key, row in sorted(
                (material.authority_snapshot.get("paths") or {}).items()
            )
        ],
    )
    _write_jsonl_atomic(
        output_dir / "decisions.jsonl",
        [
            row
            for _key, row in sorted(
                (material.authority_snapshot.get("decisions") or {}).items()
            )
        ],
    )
    _write_jsonl_atomic(output_dir / "derivations.jsonl", material.derivations)
    write_json_atomic(
        output_dir / "materialization_report.json",
        material.report,
    )
    artifact_names = (
        "inventory.json",
        "plan.json",
        "containers.json",
        "clauses.jsonl",
        "claims.jsonl",
        "receipts.jsonl",
        "paths.jsonl",
        "decisions.jsonl",
        "derivations.jsonl",
        "materialization_report.json",
    )
    report = {
        "schema": BINDER_REPORT_SCHEMA,
        "status": "passed",
        "overlay_manifest_sha256": sha256_file(
            overlay_snapshot / "overlay_manifest.json"
        ),
        "proposals_sha256": sha256_file(proposals_path),
        "inventory_hash": inventory["inventory_hash"],
        "plan_hash": plan["plan_hash"],
        "consumed_event_ids": list(plan.get("consumed_event_ids") or []),
        "materialization_hash": material.report["materialization_hash"],
        "positive_result_count": material.report["positive_result_count"],
        "positive_adopted_count": material.report[
            "positive_adopted_count"
        ],
        "quarantined_input_count": material.report[
            "quarantined_input_count"
        ],
        "artifact_hashes": {
            name: sha256_file(output_dir / name) for name in artifact_names
        },
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    write_json_atomic(output_dir / "binder_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bind Result/Adoption/Causal Overlay events into separately "
            "authorized Positive Result/Positive Adopted SOP material."
        )
    )
    parser.add_argument("--overlay-snapshot", required=True, type=Path)
    parser.add_argument("--proposals", required=True, type=Path)
    parser.add_argument("--protocol-registry", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--policy-version", default="authority_v1")
    parser.add_argument("--collector-version", default="1")
    args = parser.parse_args()
    report = bind(
        args.overlay_snapshot,
        args.proposals,
        args.protocol_registry,
        args.output_dir,
        policy_version=args.policy_version,
        collector_version=args.collector_version,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
