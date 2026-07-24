from __future__ import annotations

import json
import shutil
import stat
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_wp8_final_regression_receipt import (  # noqa: E402
    build_receipt,
    payload_hash,
    sha256_file,
)
import build_wp8_final_stop_gate as stop_gate_builder  # noqa: E402
from build_wp8_final_stop_gate import (  # noqa: E402
    _render_markdown,
    build_stop_gate,
    gate_state,
    formal_integrity_checks,
)
from verify_wp8_final_stop_gate import verify_stop_gate  # noqa: E402
from tests.test_wp8_final_test_runner import _fake_receipt  # noqa: E402


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


def _build_regression(tmp_path: Path) -> tuple[Path, Path]:
    package = tmp_path / "final-regression-package"
    package.mkdir()
    receipt = package / "final-regression.json"
    test_root = _curated_test_root(tmp_path)
    build_receipt(
        output_path=receipt,
        repo_root=ROOT,
        test_root=test_root,
        final_suite_filename="full_suite_r1.xml",
        created_at="2026-07-24T00:00:00Z",
        branch="codex/test",
        head="a" * 40,
    )
    package.chmod(0o555)
    return receipt, test_root


def _host_receipt_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    root = tmp_path / "host-test-receipt"
    root.mkdir()
    required_ids = list(stop_gate_builder.REQUIRED_FULL_SUITE_TESTS)
    testcase_ids = required_ids + [
        f"tests.placeholder::test_{index}"
        for index in range(735 - len(required_ids))
    ]
    suites = ET.Element("testsuites")
    suite = ET.SubElement(
        suites,
        "testsuite",
        {
            "name": "pytest",
            "tests": str(len(testcase_ids)),
            "failures": "0",
            "errors": "0",
            "skipped": "0",
            "time": "0",
        },
    )
    for testcase_id in testcase_ids:
        classname, name = testcase_id.split("::", 1)
        ET.SubElement(suite, "testcase", {"classname": classname, "name": name})
    ET.ElementTree(suites).write(
        root / "full_suite_r2.xml", encoding="utf-8", xml_declaration=True
    )
    receipt = {
        "full_suite_run_name": "full_suite_r2",
        "test_runs": [
            {
                "name": "full_suite_r2",
                "junit_path": "full_suite_r2.xml",
            }
        ],
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = payload_hash(receipt, "receipt_hash")
    manifest = {"manifest_hash": ""}
    manifest["manifest_hash"] = payload_hash(manifest, "manifest_hash")
    (root / "test_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        stop_gate_builder,
        "verify_host_test_receipt",
        lambda **_: {
            "verified": True,
            "errors": [],
            "receipt_hash": receipt["receipt_hash"],
            "manifest_hash": manifest["manifest_hash"],
            "source_inventory_sha256": "s" * 64,
            "full_suite_test_count": 735,
            "full_suite_testcase_ids_sha256": "t" * 64,
        },
    )
    return root


def _build_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict]:
    receipt, test_root = _build_regression(tmp_path)
    host_root = _host_receipt_root(tmp_path, monkeypatch)
    gate = tmp_path / "final-gate"
    report = build_stop_gate(
        output_root=gate,
        repo_root=ROOT,
        final_regression_receipt_path=receipt,
        final_test_root=test_root,
        host_test_receipt_root=host_root,
        created_at="2026-07-24T01:00:00Z",
    )
    return gate, receipt, test_root, host_root, report


def test_final_stop_gate_separates_engineering_completion_from_effect_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, receipt, test_root, host_root, report = _build_gate(tmp_path, monkeypatch)

    assert report["wp8_engineering_complete"] is True
    assert report["wp8_stop_gate_passed"] is True
    assert report["effect_claim_authorized"] is False
    assert report["next_authorized_phase"] == "Independent Claude audit"
    assert report["independent_claude_audit_required"] is True
    assert report["goal_completion_authorized"] is False
    assert report["formal_population"]["assigned_online_outcomes"] == 45
    assert report["formal_population"]["imputed_scores"] == 0
    assert report["claim_status"]["WP8-C3-FULL-SUPERIORITY"] == "rejected"
    assert all(report["prerequisite_checks"].values())
    assert all(report["kill_gates"].values())
    assert all(
        value
        for group in report["acceptance_checks"].values()
        for value in group.values()
    )
    for filename in ("stop_gate_report.json", "stop_gate_report.md", "manifest.json"):
        assert not (gate / filename).stat().st_mode & stat.S_IWUSR

    verification = verify_stop_gate(
        stop_gate_root=gate,
        repo_root=ROOT,
        final_regression_receipt_path=receipt,
        final_test_root=test_root,
        host_test_receipt_root=host_root,
    )
    assert verification["verified"] is True
    assert verification["errors"] == []


@pytest.mark.parametrize("category", ["prerequisite", "kill_gate", "acceptance"])
def test_any_required_gate_failure_revokes_next_phase_and_never_opens_effect(
    category: str,
) -> None:
    prerequisites = {"p1": True, "p2": True}
    kill_gates = {"g1": True, "g2": True}
    acceptance = {"code": {"a1": True}, "paper": {"a2": True}}
    if category == "prerequisite":
        prerequisites["p2"] = False
    elif category == "kill_gate":
        kill_gates["g2"] = False
    else:
        acceptance["paper"]["a2"] = False

    state = gate_state(
        prerequisite_checks=prerequisites,
        kill_gates=kill_gates,
        acceptance_checks=acceptance,
    )
    assert state["wp8_engineering_complete"] is False
    assert state["wp8_stop_gate_passed"] is False
    assert state["next_authorized_phase"] is None
    assert state["independent_claude_audit_required"] is False
    assert state["effect_claim_authorized"] is False
    assert state["goal_completion_authorized"] is False


def test_final_stop_gate_verifier_rejects_self_rehashed_effect_laundering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, receipt, test_root, host_root, _report = _build_gate(tmp_path, monkeypatch)
    report_path = gate / "stop_gate_report.json"
    markdown_path = gate / "stop_gate_report.md"
    manifest_path = gate / "manifest.json"
    for path in (report_path, markdown_path, manifest_path):
        path.chmod(0o644)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["effect_claim_authorized"] = True
    report["report_hash"] = payload_hash(report, "report_hash")
    report_path.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["stop_gate_report.json"] = sha256_file(report_path)
    manifest["files"]["stop_gate_report.md"] = sha256_file(markdown_path)
    manifest["report_hash"] = report["report_hash"]
    manifest["manifest_hash"] = payload_hash(manifest, "manifest_hash")
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    verification = verify_stop_gate(
        stop_gate_root=gate,
        repo_root=ROOT,
        final_regression_receipt_path=receipt,
        final_test_root=test_root,
        host_test_receipt_root=host_root,
    )
    assert verification["verified"] is False
    assert "effect_claim_boundary" in verification["errors"]
    assert "gate_recompute_mismatch" in verification["errors"]


def test_final_stop_gate_refuses_existing_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, test_root = _build_regression(tmp_path)
    host_root = _host_receipt_root(tmp_path, monkeypatch)
    output = tmp_path / "user-owned"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        build_stop_gate(
            output_root=output,
            repo_root=ROOT,
            final_regression_receipt_path=receipt,
            final_test_root=test_root,
            host_test_receipt_root=host_root,
            created_at="2026-07-24T01:00:00Z",
        )


def test_final_stop_gate_integrates_real_host_receipt_verifier_and_tamper(
    tmp_path: Path,
) -> None:
    receipt, test_root = _build_regression(tmp_path)
    host_root = tmp_path / "real-host-test-receipt"
    _fake_receipt(
        ROOT,
        host_root,
        additional_full_ids=tuple(stop_gate_builder.REQUIRED_FULL_SUITE_TESTS),
    )
    gate = tmp_path / "real-host-final-gate"
    report = build_stop_gate(
        output_root=gate,
        repo_root=ROOT,
        final_regression_receipt_path=receipt,
        final_test_root=test_root,
        host_test_receipt_root=host_root,
        created_at="2026-07-24T02:00:00Z",
    )
    assert report["wp8_stop_gate_passed"] is True
    assert report["host_test_receipt_verification"]["verified"] is True

    clean = verify_stop_gate(
        stop_gate_root=gate,
        repo_root=ROOT,
        final_regression_receipt_path=receipt,
        final_test_root=test_root,
        host_test_receipt_root=host_root,
    )
    assert clean["verified"] is True

    junit = host_root / "full_suite_r2.xml"
    junit.chmod(0o644)
    with junit.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    junit.chmod(0o444)
    tampered = verify_stop_gate(
        stop_gate_root=gate,
        repo_root=ROOT,
        final_regression_receipt_path=receipt,
        final_test_root=test_root,
        host_test_receipt_root=host_root,
    )
    assert tampered["verified"] is False
    assert "gate_recompute_mismatch" in tampered["errors"]


def _formal_payloads() -> tuple[dict, dict, dict]:
    inventory = {
        "totals": {
            "block_count": 9,
            "online_condition_count": 45,
            "oracle_disposition_count": 9,
            "successful_selected_result_count": 22,
            "failed_online_condition_count": 23,
            "result_fact_count": 22,
        }
    }
    statistics = {
        "analysis_population": {
            "assigned_online_outcomes": 45,
            "assigned_oracle_dispositions": 9,
            "scored_selected_results": 22,
            "failed_online_conditions": 23,
            "imputed_scores": 0,
            "post_assignment_exclusions": 0,
        },
        "effect_claim_gate": {"effect_claim_authorized": False},
    }
    ledger = {
        "headline_effect_claim_authorized": False,
        "claims": [
            {
                "claim_id": "WP8-C2-RESULT-WRITEBACK",
                "status": "supported",
                "metrics": {
                    "result_facts": 22,
                    "fixed_holdout_orphans": 0,
                    "failed_conditions_with_result_fact": 0,
                },
            },
            {
                "claim_id": "WP8-C3-FULL-SUPERIORITY",
                "status": "rejected",
            },
            {
                "claim_id": "WP8-C4-CONDITIONAL-UTILITY",
                "status": "diagnostic",
                "claim_gate": {"superiority_authorized": False},
            },
            {
                "claim_id": "WP8-C5-NO-IMPUTATION",
                "status": "supported",
                "metrics": {"imputed": 0, "post_assignment_exclusions": 0},
            },
            {
                "claim_id": "WP8-C6-EXPERIENCE-CAUSALITY",
                "status": "pending",
                "claim_gate": {"satisfied": False},
                "metrics": {"required_minimum_actuation_level": "L4"},
            },
        ],
    }
    return inventory, statistics, ledger


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            lambda i, s, l: s["analysis_population"].__setitem__(
                "assigned_online_outcomes", 44
            ),
            "exact_9_blocks_45_online_9_oracle",
        ),
        (
            lambda i, s, l: s["analysis_population"].__setitem__(
                "imputed_scores", 1
            ),
            "no_imputation_and_no_post_assignment_exclusion",
        ),
        (
            lambda i, s, l: l["claims"][0]["metrics"].__setitem__(
                "fixed_holdout_orphans", 1
            ),
            "result_fact_closure_has_zero_orphans",
        ),
        (
            lambda i, s, l: l["claims"][1].__setitem__(
                "status", "supported"
            ),
            "full_superiority_rejection_preserved",
        ),
        (
            lambda i, s, l: l["claims"][4].__setitem__(
                "status", "supported"
            ),
            "experience_causality_remains_pending_without_l4",
        ),
    ],
)
def test_formal_integrity_rejects_count_imputation_or_claim_laundering(
    mutation,
    failed_check: str,
) -> None:
    inventory, statistics, ledger = _formal_payloads()
    mutation(inventory, statistics, ledger)
    checks = formal_integrity_checks(
        inventory=inventory,
        statistics=statistics,
        ledger=ledger,
    )
    assert checks[failed_check] is False
