"""Revision-aware source binding for immutable formal preregistration amendments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


FILES = {
    revision: f"coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_{revision}.json"
    for revision in ("r6", "r7", "r8", "r9")
}
R9_VERIFICATION = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "preregistration_verification_20260723_r9.json"
)
VERIFIER_UPGRADE_RECEIPT = (
    "coordination/decision_admissibility_wp8_tier2_formal_"
    "historical_verifier_upgrade_20260723_r1.json"
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sidecar_matches(path: Path) -> bool:
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        return False
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    return len(fields) >= 2 and fields[0] == _file_hash(path) and fields[-1] == path.name


def r9_binds_current_source(
    repo_root: str | Path,
    relative: str,
    *,
    official_amendment_path: str | Path,
    ancestor_revision: str,
) -> bool:
    """Return true only for a source superseded through the official r9 chain."""

    repo = Path(repo_root).resolve()
    if ancestor_revision not in {"r6", "r7"}:
        return False
    official = (repo / FILES[ancestor_revision]).resolve()
    if Path(official_amendment_path).resolve() != official:
        return False
    try:
        paths = {revision: (repo / value).resolve() for revision, value in FILES.items()}
        if not all(path.is_file() for path in paths.values()):
            return False
        payloads = {revision: _read(path) for revision, path in paths.items()}
        if not all(
            payload.get("amendment_hash") == _payload_hash(payload, "amendment_hash")
            for payload in payloads.values()
        ):
            return False
        if not _sidecar_matches(paths["r9"]):
            return False
        r7_parent = payloads["r7"].get("parent_preregistration") or {}
        r8_parent = payloads["r8"].get("parent_continuation") or {}
        r9_parent = payloads["r9"].get("parent_amendment") or {}
        if r7_parent.get("path") != FILES["r6"] or r7_parent.get(
            "file_sha256"
        ) != _file_hash(paths["r6"]):
            return False
        if (
            r8_parent.get("path") != FILES["r7"]
            or r8_parent.get("file_sha256") != _file_hash(paths["r7"])
            or r8_parent.get("amendment_hash") != payloads["r7"]["amendment_hash"]
        ):
            return False
        if (
            r9_parent.get("path") != FILES["r8"]
            or r9_parent.get("file_sha256") != _file_hash(paths["r8"])
            or r9_parent.get("amendment_hash") != payloads["r8"]["amendment_hash"]
        ):
            return False
        if payloads["r9"].get("status") != (
            "result_blind_precontract_recovery_and_r5_retry_frozen"
        ):
            return False
        verification_path = (repo / R9_VERIFICATION).resolve()
        verification = _read(verification_path)
        if (
            not verification_path.is_file()
            or not _sidecar_matches(verification_path)
            or verification.get("verified") is not True
            or verification.get("errors") != []
            or verification.get("verification_hash")
            != _payload_hash(verification, "verification_hash")
        ):
            return False
        source_path = (repo / relative).resolve()
        if not source_path.is_file():
            return False
        current_hash = _file_hash(source_path)
        if (payloads["r9"].get("implementation_files") or {}).get(
            relative
        ) == current_hash:
            return True
        receipt_path = (repo / VERIFIER_UPGRADE_RECEIPT).resolve()
        receipt = _read(receipt_path)
        if (
            not _sidecar_matches(receipt_path)
            or receipt.get("schema")
            != "decision_admissibility_wp8_tier2_formal_historical_verifier_upgrade_v1"
            or receipt.get("status") != "revision_aware_verifier_upgrade_frozen"
            or receipt.get("receipt_hash") != _payload_hash(receipt, "receipt_hash")
        ):
            return False
        upgrade = (receipt.get("upgrades") or {}).get(relative) or {}
        ancestor_payload = payloads[ancestor_revision]
        correction_field = (
            "implementation_correction"
            if ancestor_revision == "r6"
            else "implementation_correction"
        )
        historical_hash = (ancestor_payload.get(correction_field) or {}).get(relative)
        helper_binding = receipt.get("revision_chain_helper") or {}
        helper_path = Path(__file__).resolve()
        return (
            upgrade.get("ancestor_revision") == ancestor_revision
            and upgrade.get("historical_file_sha256") == historical_hash
            and upgrade.get("upgraded_file_sha256") == current_hash
            and helper_binding.get("path")
            == "paper-skills/memory_bundle/tier2_formal_revision_chain.py"
            and helper_binding.get("file_sha256") == _file_hash(helper_path)
            and receipt.get("targeted_regression", {}).get("failed") == 0
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


__all__ = ["r9_binds_current_source"]
