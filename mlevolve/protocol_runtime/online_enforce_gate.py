"""Verifier for the immutable online Host-SDK enforce smoke gate.

The verifier is intentionally host-side and deterministic.  A GPU smoke Job
produces a sealed JSON evidence packet; this module checks that packet before a
formal multi-task Job is allowed to start.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SMOKE_SCHEMA = "mlevolve_online_enforce_smoke_v1"
GATE_REPORT_SCHEMA = "mlevolve_online_enforce_release_gate_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SmokeGateError(ValueError):
    """Raised when online smoke evidence is absent or inconsistent."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash_payload(value: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(
        _canonical({key: item for key, item in value.items() if key != field}).encode(
            "utf-8"
        )
    ).hexdigest()


def _is_hash(value: Any) -> bool:
    return bool(SHA256_RE.fullmatch(str(value or "")))


def _case_decisions(case: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = case.get("shadow_authority_decisions") or []
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, list):
        raise SmokeGateError(f"Case {case.get('case_id')} decisions are invalid")
    return [dict(item) for item in raw if isinstance(item, Mapping)]


def _check_evidence_files(
    case: Mapping[str, Any], *, evidence_root: Path | None
) -> None:
    entries = case.get("evidence_files") or []
    if not isinstance(entries, list):
        raise SmokeGateError(f"Case {case.get('case_id')} evidence_files is invalid")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SmokeGateError("Smoke evidence file entry is invalid")
        relative = str(entry.get("path") or "")
        expected = str(entry.get("sha256") or "")
        if not relative or not _is_hash(expected):
            raise SmokeGateError("Smoke evidence file lacks a valid path/hash")
        if evidence_root is None:
            continue
        path = (evidence_root / relative).resolve()
        if path.is_symlink() or not path.is_file():
            raise SmokeGateError(f"Smoke evidence file is missing or symlinked: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise SmokeGateError(f"Smoke evidence file changed: {relative}")


def _check_traceability(case: Mapping[str, Any]) -> None:
    raw_ids = {str(value) for value in case.get("raw_candidate_ids") or [] if value}
    if not raw_ids:
        raise SmokeGateError(f"Case {case.get('case_id')} has no raw candidate set")
    raw_claim_ids = {
        str(value) for value in case.get("raw_claim_ids") or [] if value
    }
    if not raw_claim_ids:
        raise SmokeGateError(f"Case {case.get('case_id')} has no raw Claim-use set")
    decisions = _case_decisions(case)
    observed_claim_ids = {
        str(value.get("claim_id") or "") for value in decisions
    }
    missing_observations = sorted(raw_claim_ids - observed_claim_ids)
    if missing_observations:
        raise SmokeGateError(
            f"Case {case.get('case_id')} lacks raw shadow decisions for: "
            f"{missing_observations}"
        )
    for decision in decisions:
        if not str(decision.get("outcome") or ""):
            raise SmokeGateError("Raw shadow Authority observation lacks outcome")
        if not str(
            decision.get("decision_ref") or decision.get("decision_id") or ""
        ):
            raise SmokeGateError("Raw shadow Authority observation lacks decision ID")
    suppressed = {str(value) for value in case.get("suppressed_candidate_ids") or [] if value}
    reasons = case.get("suppression_reasons") or {}
    if not isinstance(reasons, Mapping):
        raise SmokeGateError("suppression_reasons must be an object")
    for candidate_id in suppressed:
        if candidate_id not in raw_ids:
            raise SmokeGateError(f"Suppressed candidate is not in raw set: {candidate_id}")
        reason = reasons.get(candidate_id)
        if not isinstance(reason, Mapping):
            raise SmokeGateError(f"Suppression lacks structured trace: {candidate_id}")
        for key in ("claim_id", "operation", "decision_stage", "protocol_ref"):
            if not str(reason.get(key) or ""):
                raise SmokeGateError(
                    f"Suppression {candidate_id} lacks {key}/Claim-Operation-Stage-Protocol trace"
                )
        if not reason.get("receipt_refs"):
            raise SmokeGateError(f"Suppression {candidate_id} lacks Receipt trace")


def verify_online_enforce_smoke(
    smoke_path: str | Path,
    *,
    evidence_root: str | Path | None = None,
    required_freeze_hash: str = "",
    required_image_digest: str = "",
) -> dict[str, Any]:
    path = Path(smoke_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise SmokeGateError("Smoke manifest must be a regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SMOKE_SCHEMA:
        raise SmokeGateError("Online smoke schema mismatch")
    if payload.get("manifest_hash") != _hash_payload(payload, "manifest_hash"):
        raise SmokeGateError("Online smoke manifest hash mismatch")
    runtime = payload.get("runtime")
    if not isinstance(runtime, Mapping):
        raise SmokeGateError("Online smoke runtime binding is missing")
    required_runtime = {
        "authority_mode": "enforce",
        "protocol_runtime_mode": "host_sdk_enforce",
        "execution_environment": "online_gpu",
        "gpu_probe_passed": True,
    }
    for key, expected in required_runtime.items():
        if runtime.get(key) != expected:
            raise SmokeGateError(f"Online smoke runtime binding failed: {key}")
    freeze_hash = str(payload.get("freeze_manifest_hash") or "")
    if not _is_hash(freeze_hash):
        raise SmokeGateError("Online smoke is not bound to a freeze manifest")
    if required_freeze_hash and freeze_hash != required_freeze_hash:
        raise SmokeGateError("Online smoke freeze hash does not match formal freeze")
    image = str(payload.get("container_image_digest") or "")
    if "@sha256:" not in image or not re.search(r"@sha256:[0-9a-f]{64}$", image):
        raise SmokeGateError("Online smoke image is not digest-pinned")
    if required_image_digest and image != required_image_digest:
        raise SmokeGateError("Online smoke image digest does not match formal freeze")

    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SmokeGateError("Online smoke has no cases")
    ids: set[str] = set()
    for case_value in cases:
        if not isinstance(case_value, Mapping):
            raise SmokeGateError("Online smoke case is invalid")
        case = dict(case_value)
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in ids:
            raise SmokeGateError("Online smoke case IDs must be unique")
        ids.add(case_id)
        _check_traceability(case)
        _check_evidence_files(
            case,
            evidence_root=Path(evidence_root).resolve() if evidence_root else None,
        )
        if not str(case.get("decision_stage") or "") or not str(case.get("operation") or ""):
            raise SmokeGateError(f"Case {case_id} lacks decision stage/operation")

    by_class = {str(case.get("case_class")): case for case in cases}
    legal = [case for case in cases if case.get("expected_legal") is True]
    invalid = [case for case in cases if case.get("expected_legal") is False]
    if not any(str(case.get("task_id")) == "denoising-dirty-documents" for case in legal):
        raise SmokeGateError("Legal Denoising case is required")
    if not any(str(case.get("task_id")) == "leaf-classification" for case in legal):
        raise SmokeGateError("Legal Leaf case is required")
    if not any(str(case.get("task_id")) == "spooky-author-identification" for case in invalid):
        raise SmokeGateError("Known-invalid Spooky positive control is required")
    if "mixed_value" not in by_class:
        raise SmokeGateError("Mixed-value Claim case is required")

    legal_denials = 0
    invalid_prompt_exposure = 0
    legal_evidence_failures = 0
    invalid_preflight_allows = 0
    mixed = by_class["mixed_value"]
    for case in cases:
        decisions = _case_decisions(case)
        host_evidence = case.get("host_evidence") or {}
        if not isinstance(host_evidence, Mapping):
            raise SmokeGateError(f"Case {case.get('case_id')} host evidence is invalid")
        if case.get("expected_legal") is True:
            suppressed_claims = {
                str(reason.get("claim_id"))
                for reason in (case.get("suppression_reasons") or {}).values()
                if isinstance(reason, Mapping) and reason.get("claim_id")
            }
            legal_denials += sum(
                str(row.get("outcome", "")).lower() in {"deny", "require_replay", "block"}
                and str(row.get("claim_id") or "") not in suppressed_claims
                for row in decisions
            )
            if case.get("case_class") == "mixed_value":
                legal_evidence_failures += int(
                    host_evidence.get("status") != "pass"
                    or not _is_hash(host_evidence.get("closure_hash"))
                    or host_evidence.get("terminal_exposure_count") != 0
                )
            else:
                legal_evidence_failures += int(
                    host_evidence.get("status") != "pass"
                    or not _is_hash(host_evidence.get("closure_hash"))
                    or not host_evidence.get("runtime_receipt_refs")
                    or host_evidence.get("terminal_exposure_count") != 0
                )
        else:
            invalid_preflight_allows += int(host_evidence.get("status") == "pass")
            invalid_prompt_exposure += len(
                set(case.get("prompt_visible_invalid_candidate_ids") or [])
            )
    mixed_raw = {str(value) for value in mixed.get("raw_candidate_ids") or []}
    mixed_suppressed = {str(value) for value in mixed.get("suppressed_candidate_ids") or []}
    mixed_retained = {str(value) for value in mixed.get("final_prompt_candidate_ids") or []}
    if not mixed_raw & mixed_suppressed or not mixed_raw - mixed_suppressed:
        raise SmokeGateError("Mixed-value case must contain both retained and suppressed Claims")
    if mixed_suppressed & mixed_retained:
        raise SmokeGateError("Mixed-value invalid Claim reached the Prompt")
    if not (mixed_raw - mixed_suppressed) & mixed_retained:
        raise SmokeGateError("Mixed-value legal Claim was not retained")

    checks = {
        "host_sdk_enforce": True,
        "legal_false_denial_zero": legal_denials == 0,
        "legal_host_evidence_closure_complete": legal_evidence_failures == 0,
        "known_invalid_preflight_allow_zero": invalid_preflight_allows == 0,
        "known_invalid_prompt_exposure_zero": invalid_prompt_exposure == 0,
        "mixed_value_retains_legal_and_suppresses_invalid": bool(
            mixed_raw - mixed_suppressed and mixed_suppressed
        ),
        "all_cases_traceable": True,
    }
    report: dict[str, Any] = {
        "schema": GATE_REPORT_SCHEMA,
        "status": "passed" if all(checks.values()) else "blocked",
        "smoke_manifest": str(path),
        "smoke_manifest_hash": payload["manifest_hash"],
        "freeze_manifest_hash": freeze_hash,
        "container_image_digest": image,
        "case_count": len(cases),
        "legal_case_count": len(legal),
        "invalid_case_count": len(invalid),
        "checks": checks,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "report_hash": "",
    }
    report["report_hash"] = _hash_payload(report, "report_hash")
    if report["status"] != "passed":
        raise SmokeGateError("Online enforce release gate is blocked")
    return report


def write_smoke_manifest(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    value.setdefault("schema", SMOKE_SCHEMA)
    value.setdefault(
        "created_at_utc",
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    value["manifest_hash"] = _hash_payload(value, "manifest_hash")
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise SmokeGateError(f"Refusing to replace immutable smoke manifest: {target}") from error
    with os.fdopen(fd, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return value


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--freeze-hash", default="")
    parser.add_argument("--image-digest", default="")
    args = parser.parse_args()
    report = verify_online_enforce_smoke(
        args.smoke,
        evidence_root=args.evidence_root,
        required_freeze_hash=args.freeze_hash,
        required_image_digest=args.image_digest,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "GATE_REPORT_SCHEMA",
    "SMOKE_SCHEMA",
    "SmokeGateError",
    "verify_online_enforce_smoke",
    "write_smoke_manifest",
]
