#!/usr/bin/env python3
"""Fail-closed online packet gate for Experiment A.

This validates collection completeness only. Oracle Claim-use labels, blind
dual annotation, adjudication, IIR, VKR, and final statistical analysis remain
post-run operations and must not be inferred by this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


SCHEMA = "mlevolve_prospective_claim_use_decision_v1"
GATE_SCHEMA = "mlevolve_prevalence_online_packet_gate_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FIELDS = {
    "run_id",
    "task_id",
    "agent_seed",
    "decision_id",
    "decision_stage",
    "operation",
    "protocol_ref",
    "raw_candidate_ids",
    "raw_relevance_scores",
    "raw_claim_ids",
    "raw_claim_types",
    "shadow_authority_decisions",
    "suppressed_candidate_ids",
    "suppression_reasons",
    "final_prompt_candidate_ids",
    "actual_action_hash",
    "actual_code_hash",
    "runtime_receipt_refs",
    "counterfactual_action_hash",
    "counterfactual_code_hash",
    "counterfactual_status",
}


class PacketError(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise PacketError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise PacketError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _one(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise PacketError(f"expected exactly one {name} below {root}, found {len(matches)}")
    return matches[0]


def _list(row: dict[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    if not isinstance(value, list):
        raise PacketError(f"{row.get('decision_id')}: {key} must be a list")
    return value


def _reason_for(reasons: Any, candidate_id: str) -> dict[str, Any] | None:
    if isinstance(reasons, dict):
        value = reasons.get(candidate_id)
        return value if isinstance(value, dict) else None
    if isinstance(reasons, list):
        for value in reasons:
            if isinstance(value, dict) and str(value.get("candidate_id")) == candidate_id:
                return value
    return None


def _validate_decision(
    row: dict[str, Any],
    *,
    task_id: str,
    seed: int,
    allow_pending_counterfactual: bool,
) -> None:
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        raise PacketError(f"{row.get('decision_id')}: missing fields {missing}")
    if row.get("schema") != SCHEMA:
        raise PacketError(f"{row.get('decision_id')}: schema mismatch")
    if str(row["task_id"]) != task_id or int(row["agent_seed"]) != seed:
        raise PacketError(f"{row.get('decision_id')}: task/seed mismatch")
    for key in ("run_id", "decision_id", "decision_stage", "operation", "protocol_ref"):
        if not str(row.get(key) or "").strip():
            raise PacketError(f"{row.get('decision_id')}: empty {key}")

    raw_ids = [str(value) for value in _list(row, "raw_candidate_ids")]
    raw_scores = _list(row, "raw_relevance_scores")
    raw_claim_ids = [str(value) for value in _list(row, "raw_claim_ids")]
    raw_claim_types = [str(value) for value in _list(row, "raw_claim_types")]
    if len({len(raw_ids), len(raw_scores), len(raw_claim_ids), len(raw_claim_types)}) != 1:
        raise PacketError(f"{row['decision_id']}: raw Claim-use arrays are not aligned")
    if any(not value for value in raw_ids + raw_claim_ids + raw_claim_types):
        raise PacketError(f"{row['decision_id']}: blank raw Claim-use identifier")
    if any(not isinstance(value, (int, float)) for value in raw_scores):
        raise PacketError(f"{row['decision_id']}: non-numeric relevance score")

    shadow = _list(row, "shadow_authority_decisions")
    shadow_pairs = {
        (str(value.get("candidate_id")), str(value.get("claim_id")))
        for value in shadow
        if isinstance(value, dict)
    }
    raw_pairs = set(zip(raw_ids, raw_claim_ids))
    if not raw_pairs.issubset(shadow_pairs):
        raise PacketError(f"{row['decision_id']}: missing per-Claim-use shadow decision")

    suppressed = [str(value) for value in _list(row, "suppressed_candidate_ids")]
    final = [str(value) for value in _list(row, "final_prompt_candidate_ids")]
    if set(suppressed) & set(final):
        raise PacketError(f"{row['decision_id']}: suppressed candidate is Prompt-visible")
    if not set(suppressed).issubset(set(raw_ids)) or not set(final).issubset(set(raw_ids)):
        raise PacketError(f"{row['decision_id']}: post-gate candidate absent from raw set")

    reasons = row["suppression_reasons"]
    for candidate_id in suppressed:
        reason = _reason_for(reasons, candidate_id)
        if reason is None:
            raise PacketError(f"{row['decision_id']}: no trace for suppression {candidate_id}")
        for key in ("claim_id", "operation", "decision_stage", "protocol_ref"):
            if not str(reason.get(key) or "").strip():
                raise PacketError(f"{row['decision_id']}: suppression {candidate_id} lacks {key}")
        receipts = reason.get("receipt_refs")
        if not isinstance(receipts, list) or not receipts:
            raise PacketError(f"{row['decision_id']}: suppression {candidate_id} lacks receipt")

    _list(row, "runtime_receipt_refs")
    for key in ("actual_action_hash", "actual_code_hash"):
        if not SHA256_RE.fullmatch(str(row[key])):
            raise PacketError(f"{row['decision_id']}: {key} is not SHA-256")
    counterfactual_status = str(row.get("counterfactual_status") or "")
    if counterfactual_status == "pending":
        if not allow_pending_counterfactual:
            raise PacketError(f"{row['decision_id']}: counterfactual remains pending")
        if row["counterfactual_action_hash"] or row["counterfactual_code_hash"]:
            raise PacketError(f"{row['decision_id']}: pending counterfactual has fabricated hash")
    elif counterfactual_status in {"identity", "complete"}:
        for key in ("counterfactual_action_hash", "counterfactual_code_hash"):
            if not SHA256_RE.fullmatch(str(row[key])):
                raise PacketError(f"{row['decision_id']}: {key} is not SHA-256")
        if counterfactual_status == "complete":
            for key in (
                "counterfactual_control_hash",
                "counterfactual_memory_payload_hash",
                "counterfactual_prompt_hash",
            ):
                if not SHA256_RE.fullmatch(str(row.get(key) or "")):
                    raise PacketError(
                        f"{row['decision_id']}: {key} is not SHA-256"
                    )
            if not str(row.get("counterfactual_pair_id") or "").strip():
                raise PacketError(
                    f"{row['decision_id']}: complete counterfactual lacks pair_id"
                )
            receipt_refs = row.get("counterfactual_receipt_refs")
            if not isinstance(receipt_refs, list) or not receipt_refs:
                raise PacketError(
                    f"{row['decision_id']}: complete counterfactual lacks Host receipt"
                )
            if not set(map(str, receipt_refs)).issubset(
                set(map(str, row["runtime_receipt_refs"]))
            ):
                raise PacketError(
                    f"{row['decision_id']}: counterfactual Host receipt is unbound"
                )
    else:
        raise PacketError(f"{row['decision_id']}: invalid counterfactual_status")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve()
    ledger_path = _one(root, "prospective_decision_ledger.jsonl")
    opportunities_path = _one(root, "decision_opportunities.jsonl")
    outcome_path = _one(root, "RUN_OUTCOME.json")
    rows = _read_jsonl(ledger_path)
    opportunities = _read_jsonl(opportunities_path)
    if not rows or not opportunities:
        raise PacketError("decision ledger and opportunity ledger must be non-empty")

    decision_ids: set[str] = set()
    for row in rows:
        _validate_decision(
            row,
            task_id=args.task_id,
            seed=args.agent_seed,
            allow_pending_counterfactual=args.allow_pending_counterfactual,
        )
        decision_id = str(row["decision_id"])
        if decision_id in decision_ids:
            raise PacketError(f"duplicate decision_id {decision_id}")
        decision_ids.add(decision_id)

    opportunity_ids = {str(row.get("decision_id") or "") for row in opportunities}
    if "" in opportunity_ids or len(opportunity_ids) != len(opportunities):
        raise PacketError("opportunity ledger contains blank or duplicate decision_id")
    covered = len(decision_ids & opportunity_ids)
    coverage = covered / len(opportunity_ids)
    if coverage < 0.99:
        raise PacketError(f"raw candidate logging coverage {coverage:.6f} is below 0.99")

    manifest = json.loads(args.memory_manifest.read_text(encoding="utf-8"))
    controlled_ids = {str(value) for value in manifest.get("controlled_candidate_ids", [])}
    controlled_raw: set[str] = set()
    controlled_final: set[str] = set()
    controlled_suppressed: set[str] = set()
    for row in rows:
        controlled_raw.update(controlled_ids & set(map(str, row["raw_candidate_ids"])))
        controlled_final.update(controlled_ids & set(map(str, row["final_prompt_candidate_ids"])))
        controlled_suppressed.update(controlled_ids & set(map(str, row["suppressed_candidate_ids"])))
    if args.positive_control:
        if not controlled_ids or not controlled_raw:
            raise PacketError("positive control was not proposed in any raw candidate set")
        if controlled_final:
            raise PacketError(f"known-invalid Prompt exposure is nonzero: {sorted(controlled_final)}")
        if not controlled_raw.issubset(controlled_suppressed):
            raise PacketError("not every proposed known-invalid control was suppressed")
    elif controlled_ids:
        raise PacketError("natural memory manifest contains controlled candidates")

    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    if outcome.get("status") not in {"complete", "partial"}:
        raise PacketError(f"unsupported run outcome status: {outcome.get('status')!r}")

    report = {
        "schema": GATE_SCHEMA,
        "status": "verified",
        "task_id": args.task_id,
        "agent_seed": args.agent_seed,
        "decision_count": len(rows),
        "opportunity_count": len(opportunities),
        "raw_logging_coverage": coverage,
        "positive_control": bool(args.positive_control),
        "controlled_raw_count": len(controlled_raw),
        "controlled_prompt_visible_count": len(controlled_final),
        "pending_counterfactual_count": sum(
            row.get("counterfactual_status") == "pending" for row in rows
        ),
        "note": "Oracle labels, adjudication, IIR and VKR are post-run requirements.",
    }
    (root / "ONLINE_PACKET_GATE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--agent-seed", type=int, required=True)
    parser.add_argument("--memory-manifest", type=Path, required=True)
    parser.add_argument("--positive-control", action="store_true")
    parser.add_argument("--allow-pending-counterfactual", action="store_true")
    args = parser.parse_args()
    try:
        report = validate(args)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"prevalence packet gate failed: {error}") from error
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
