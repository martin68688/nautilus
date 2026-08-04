from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from .protocol_registry import canonical_json


VERIFICATION_PLAN_SCHEMA = "agent_adoption_verification_plan_v1"
VERIFICATION_VERDICT_SCHEMA = "agent_adoption_verdict_v1"
STATIC_DISPOSITIONS = {
    "implemented",
    "partially_implemented",
    "not_implemented",
    "uncertain",
}
FINAL_VERDICTS = {
    "adopted",
    "partially_adopted",
    "rejected",
    "uncertain",
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_payload(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(value)).encode("utf-8")).hexdigest()


def _line_count(source: str) -> int:
    return max(1, len(str(source).splitlines()))


def _span_hash(source: str, start_line: int, end_line: int) -> str:
    lines = str(source).splitlines()
    selected = "\n".join(lines[start_line - 1 : end_line])
    return sha256_text(selected)


def _contract_map(contracts: Iterable[Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in contracts:
        payload = value.as_dict() if hasattr(value, "as_dict") else dict(value)
        contract_id = str(payload.get("contract_id") or "")
        contract_hash = str(payload.get("contract_hash") or "")
        if not contract_id or len(contract_hash) != 64:
            raise ValueError("Verifier input contains an unbound ExperienceContract")
        if contract_id in output:
            raise ValueError(f"Duplicate verifier contract: {contract_id}")
        output[contract_id] = payload
    return output


def _normalized_observations(
    raw: Any,
    *,
    allowed: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in seen or name not in allowed:
            continue
        # The verifier may identify evidence for a predicate, but it may not
        # rewrite the immutable value expected by the ExperienceContract.
        if item.get("value") != allowed[name]:
            continue
        seen.add(name)
        output.append(
            {
                "name": name,
                "value": allowed[name],
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    return output


def build_verification_plan(
    *,
    artifact_id: str,
    source: str,
    contracts: Iterable[Any],
    response: Mapping[str, Any],
    verifier_model: str,
) -> dict[str, Any]:
    """Validate and bind one Agent-authored static verification plan.

    The Agent supplies semantic judgments and probe locations. The Host owns
    every identity field and silently turns missing/malformed contract rows
    into ``uncertain`` rows, so omission can never create adoption credit.
    """

    by_contract = _contract_map(contracts)
    source = str(source)
    source_lines = _line_count(source)
    rows = response.get("contract_results")
    rows = rows if isinstance(rows, list) else []
    supplied: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        contract_id = str(row.get("contract_id") or "")
        if contract_id in by_contract and contract_id not in supplied:
            supplied[contract_id] = row

    normalized_rows: list[dict[str, Any]] = []
    probe_ids: set[str] = set()
    for contract_id in sorted(by_contract):
        contract = by_contract[contract_id]
        raw = supplied.get(contract_id, {})
        disposition = str(raw.get("disposition") or "uncertain").lower()
        if disposition not in STATIC_DISPOSITIONS:
            disposition = "uncertain"

        evidence: list[dict[str, Any]] = []
        for index, item in enumerate(
            raw.get("code_evidence")
            if isinstance(raw.get("code_evidence"), list)
            else []
        ):
            if not isinstance(item, Mapping):
                continue
            try:
                start_line = int(item.get("start_line"))
                end_line = int(item.get("end_line"))
            except (TypeError, ValueError):
                continue
            if not (1 <= start_line <= end_line <= source_lines):
                continue
            evidence_id = f"code_evidence::{sha256_text(f'{contract_id}|{start_line}|{end_line}')[:24]}"
            evidence.append(
                {
                    "evidence_id": evidence_id,
                    "start_line": start_line,
                    "end_line": end_line,
                    "source_sha256": _span_hash(source, start_line, end_line),
                    "description": str(item.get("description") or "").strip(),
                    "ordinal": index,
                }
            )

        probes: list[dict[str, Any]] = []
        for index, item in enumerate(
            raw.get("runtime_probes")
            if isinstance(raw.get("runtime_probes"), list)
            else []
        ):
            if not isinstance(item, Mapping):
                continue
            try:
                start_line = int(item.get("start_line"))
                end_line = int(item.get("end_line"))
            except (TypeError, ValueError):
                continue
            if not (1 <= start_line <= end_line <= source_lines):
                continue
            requested_id = str(item.get("probe_id") or "").strip()
            probe_id = requested_id or (
                f"probe::{sha256_text(f'{contract_id}|{start_line}|{end_line}|{index}')[:24]}"
            )
            if probe_id in probe_ids:
                continue
            probe_ids.add(probe_id)
            probes.append(
                {
                    "probe_id": probe_id,
                    "kind": "line_range_executed",
                    "start_line": start_line,
                    "end_line": end_line,
                    "description": str(item.get("description") or "").strip(),
                }
            )

        precondition_expected = {
            str(item.get("name")): item.get("expected")
            for item in contract.get("preconditions") or []
            if isinstance(item, Mapping) and item.get("name")
        }
        static_expected = {
            str(item.get("name")): item.get("expected")
            for field in ("must_preserve", "must_change", "must_not_use")
            for item in contract.get(field) or []
            if isinstance(item, Mapping) and item.get("name")
        }
        runtime_expected = {
            str(item.get("name")): item.get("expected")
            for item in contract.get("expected_runtime_observations") or []
            if isinstance(item, Mapping) and item.get("name")
        }
        positive = disposition in {"implemented", "partially_implemented"}
        if positive and (not evidence or not probes):
            # A positive semantic statement without inspectable source and an
            # executable probe is not a verification plan.
            disposition = "uncertain"

        proposed_static = _normalized_observations(
            raw.get("static_observations"),
            allowed={**precondition_expected, **static_expected},
        )
        normalized_rows.append(
            {
                "contract_id": contract_id,
                "contract_hash": contract["contract_hash"],
                "clause_id": str(contract.get("clause_id") or ""),
                "sop_id": str(contract.get("sop_id") or ""),
                "disposition": disposition,
                "reasoning": str(raw.get("reasoning") or "").strip(),
                "code_evidence": evidence,
                "runtime_probes": probes,
                "precondition_observations": [
                    item
                    for item in proposed_static
                    if item["name"] in precondition_expected
                ],
                "static_observations": [
                    item for item in proposed_static if item["name"] in static_expected
                ],
                "runtime_observations": _normalized_observations(
                    raw.get("runtime_observations"), allowed=runtime_expected
                ),
            }
        )

    payload = {
        "schema": VERIFICATION_PLAN_SCHEMA,
        "artifact_id": str(artifact_id),
        "code_sha256": sha256_text(source),
        "verifier_model": str(verifier_model),
        "contract_results": normalized_rows,
        "plan_hash": "",
    }
    payload["plan_hash"] = sha256_payload(
        {key: value for key, value in payload.items() if key != "plan_hash"}
    )
    return payload


def verify_plan(plan: Mapping[str, Any], *, artifact_id: str, source: str) -> None:
    if plan.get("schema") != VERIFICATION_PLAN_SCHEMA:
        raise ValueError("Unsupported adoption verification plan schema")
    if plan.get("artifact_id") != str(artifact_id):
        raise ValueError("Adoption verification plan artifact mismatch")
    if plan.get("code_sha256") != sha256_text(source):
        raise ValueError("Adoption verification plan code mismatch")
    expected = sha256_payload(
        {key: value for key, value in plan.items() if key != "plan_hash"}
    )
    if plan.get("plan_hash") != expected:
        raise ValueError("Adoption verification plan hash mismatch")


def plan_row(plan: Mapping[str, Any], contract_id: str) -> dict[str, Any] | None:
    for row in plan.get("contract_results") or []:
        if isinstance(row, Mapping) and row.get("contract_id") == str(contract_id):
            return dict(row)
    return None


def build_final_verdict(
    *,
    artifact_id: str,
    plan: Mapping[str, Any],
    trace: Mapping[str, Any],
    response: Mapping[str, Any],
    verifier_model: str,
) -> dict[str, Any]:
    """Bind the Agent's final decision to actually executed probe IDs."""

    rows = {
        str(row.get("contract_id")): dict(row)
        for row in plan.get("contract_results") or []
        if isinstance(row, Mapping) and row.get("contract_id")
    }
    trace_rows = {
        str(row.get("probe_id")): dict(row)
        for row in trace.get("probe_results") or []
        if isinstance(row, Mapping) and row.get("probe_id")
    }
    raw_rows = response.get("contract_results")
    raw_rows = raw_rows if isinstance(raw_rows, list) else []
    supplied = {
        str(row.get("contract_id")): row
        for row in raw_rows
        if isinstance(row, Mapping) and row.get("contract_id") in rows
    }

    output_rows: list[dict[str, Any]] = []
    for contract_id in sorted(rows):
        plan_value = rows[contract_id]
        raw = supplied.get(contract_id, {})
        verdict = str(raw.get("verdict") or "uncertain").lower()
        if verdict not in FINAL_VERDICTS:
            verdict = "uncertain"
        requested_refs = [
            str(value) for value in raw.get("supporting_probe_ids") or [] if str(value)
        ]
        allowed_probe_ids = {
            str(item.get("probe_id"))
            for item in plan_value.get("runtime_probes") or []
            if isinstance(item, Mapping) and item.get("probe_id")
        }
        supporting = sorted(
            {
                probe_id
                for probe_id in requested_refs
                if probe_id in allowed_probe_ids
                and trace_rows.get(probe_id, {}).get("executed") is True
            }
        )
        positive = verdict in {"adopted", "partially_adopted"}
        static_positive = plan_value.get("disposition") in {
            "implemented",
            "partially_implemented",
        }
        if positive and (not static_positive or not supporting):
            verdict = "uncertain"
        output_rows.append(
            {
                "contract_id": contract_id,
                "contract_hash": plan_value.get("contract_hash"),
                "verdict": verdict,
                "reasoning": str(raw.get("reasoning") or "").strip(),
                "supporting_probe_ids": supporting,
                "runtime_evidence_valid": bool(
                    supporting
                    and trace.get("plan_hash") == plan.get("plan_hash")
                    and trace.get("code_sha256") == plan.get("code_sha256")
                ),
            }
        )

    payload = {
        "schema": VERIFICATION_VERDICT_SCHEMA,
        "artifact_id": str(artifact_id),
        "code_sha256": plan.get("code_sha256"),
        "plan_hash": plan.get("plan_hash"),
        "trace_hash": trace.get("trace_hash"),
        "verifier_model": str(verifier_model),
        "contract_results": output_rows,
        "verdict_hash": "",
    }
    payload["verdict_hash"] = sha256_payload(
        {key: value for key, value in payload.items() if key != "verdict_hash"}
    )
    return payload


def verdict_row(verdict: Mapping[str, Any], contract_id: str) -> dict[str, Any] | None:
    for row in verdict.get("contract_results") or []:
        if isinstance(row, Mapping) and row.get("contract_id") == str(contract_id):
            return dict(row)
    return None


__all__ = [
    "FINAL_VERDICTS",
    "STATIC_DISPOSITIONS",
    "VERIFICATION_PLAN_SCHEMA",
    "VERIFICATION_VERDICT_SCHEMA",
    "build_final_verdict",
    "build_verification_plan",
    "plan_row",
    "sha256_payload",
    "sha256_text",
    "verdict_row",
    "verify_plan",
]
