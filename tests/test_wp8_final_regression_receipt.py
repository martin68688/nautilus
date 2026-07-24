from __future__ import annotations

import json
import shutil
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_wp8_final_regression_receipt import (  # noqa: E402
    build_receipt,
    parse_junit,
    payload_hash,
    sha256_file,
)
from verify_wp8_final_regression_receipt import verify_receipt  # noqa: E402


TEST_ROOT = ROOT / "coordination" / (
    "decision_admissibility_wp8_final_tests_20260723_r1"
)


def _curated_test_root(tmp_path: Path) -> Path:
    output = tmp_path / "curated-tests"
    output.mkdir()
    for name in (
        "baseline_section_20_1.xml",
        "result_adoption_causal_section_20_1_A.xml",
        "new_unit_section_20_2.xml",
        "integration_section_20_3.xml",
        "tier2_formal_and_new.xml",
        "tier2_formal_and_new_r2.xml",
        "baseline_section_20_1_r2.xml",
        "baseline_section_20_1_r3.xml",
        "full_suite_r1.xml",
    ):
        shutil.copyfile(TEST_ROOT / name, output / name)
    return output


def _kwargs(tmp_path: Path) -> dict[str, object]:
    return {
        "repo_root": ROOT,
        "test_root": _curated_test_root(tmp_path),
        "final_suite_filename": "full_suite_r1.xml",
        "created_at": "2026-07-24T00:00:00Z",
        "branch": "codex/test",
        "head": "a" * 40,
    }


def test_final_regression_receipt_preserves_failure_then_binds_clean_fix(
    tmp_path: Path,
) -> None:
    output = tmp_path / "final-regression.json"
    kwargs = _kwargs(tmp_path)
    receipt = build_receipt(output_path=output, **kwargs)

    assert receipt["final_regression_passed"] is True
    assert receipt["unexpected_failure_count"] == 0
    assert receipt["historical_failure_repair"]["initial_failure_preserved"] is True
    assert len(
        receipt["historical_failure_repair"]["initial_failed_testcase_ids"]
    ) == 2
    assert receipt["source_inventory"]["file_count"] > 300
    assert not output.stat().st_mode & stat.S_IWUSR
    assert not output.with_suffix(".sha256").stat().st_mode & stat.S_IWUSR

    verification = verify_receipt(
        receipt_path=output,
        repo_root=ROOT,
        test_root=kwargs["test_root"],
    )
    assert verification["verified"] is True
    assert verification["errors"] == []


def test_final_regression_verifier_rejects_self_rehashed_authorization_laundering(
    tmp_path: Path,
) -> None:
    output = tmp_path / "final-regression.json"
    kwargs = _kwargs(tmp_path)
    build_receipt(output_path=output, **kwargs)
    output.chmod(0o644)
    output.with_suffix(".sha256").chmod(0o644)
    receipt = json.loads(output.read_text(encoding="utf-8"))
    receipt["source_inventory"]["inventory_hash"] = "f" * 64
    receipt["receipt_hash"] = payload_hash(receipt, "receipt_hash")
    output.write_text(
        json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output.with_suffix(".sha256").write_text(
        f"{sha256_file(output)}  {output.name}\n", encoding="utf-8"
    )

    verification = verify_receipt(
        receipt_path=output,
        repo_root=ROOT,
        test_root=kwargs["test_root"],
    )
    assert verification["verified"] is False
    assert "receipt_recompute_mismatch" in verification["errors"]


def test_junit_parser_rejects_declared_count_laundering(tmp_path: Path) -> None:
    junit = tmp_path / "bad.xml"
    junit.write_text(
        "<testsuite tests='2' failures='0' errors='0' skipped='0'>"
        "<testcase classname='tests.sample' name='test_one'/>"
        "</testsuite>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="declared/observed"):
        parse_junit(junit, repo_root=tmp_path)


def test_final_regression_receipt_refuses_output_reuse(tmp_path: Path) -> None:
    output = tmp_path / "existing.json"
    output.write_text("user asset\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        build_receipt(output_path=output, **_kwargs(tmp_path))
