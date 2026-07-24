from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from tier2_formal_revision_chain import (  # noqa: E402
    FILES,
    R9_VERIFICATION,
    VERIFIER_UPGRADE_RECEIPT,
    r9_binds_current_source,
)


SOURCE = (
    "paper-skills/memory_bundle/"
    "verify_tier2_formal_preterminal_recovery_amendment.py"
)


def _copy(root: Path, relative: str) -> None:
    source = ROOT / relative
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _mirror(tmp_path: Path) -> Path:
    for relative in FILES.values():
        _copy(tmp_path, relative)
    _copy(tmp_path, FILES["r9"].replace(".json", ".sha256"))
    _copy(tmp_path, R9_VERIFICATION)
    _copy(tmp_path, R9_VERIFICATION.replace(".json", ".sha256"))
    _copy(tmp_path, VERIFIER_UPGRADE_RECEIPT)
    _copy(tmp_path, VERIFIER_UPGRADE_RECEIPT.replace(".json", ".sha256"))
    _copy(tmp_path, SOURCE)
    return tmp_path


def test_official_revision_chain_and_upgrade_receipt_bind_current_verifier(
    tmp_path: Path,
) -> None:
    repo = _mirror(tmp_path)
    official = repo / FILES["r6"]
    assert r9_binds_current_source(
        repo,
        SOURCE,
        official_amendment_path=official,
        ancestor_revision="r6",
    ) is True

    receipt = repo / VERIFIER_UPGRADE_RECEIPT
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    assert r9_binds_current_source(
        repo,
        SOURCE,
        official_amendment_path=official,
        ancestor_revision="r6",
    ) is False


def test_temporary_amendment_cannot_use_official_revision_fallback(
    tmp_path: Path,
) -> None:
    repo = _mirror(tmp_path)
    temporary = repo / "temporary-amendment.json"
    shutil.copyfile(repo / FILES["r6"], temporary)
    assert r9_binds_current_source(
        repo,
        SOURCE,
        official_amendment_path=temporary,
        ancestor_revision="r6",
    ) is False

