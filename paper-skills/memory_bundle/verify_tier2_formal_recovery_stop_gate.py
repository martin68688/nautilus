#!/usr/bin/env python3
"""Verify and authorize the result-blind r6 recovery devpods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from verify_tier2_formal_preterminal_recovery_amendment import (  # noqa: E402
    verify_preterminal_recovery_amendment,
)


SCHEMA = "decision_admissibility_wp8_tier2_formal_recovery_stop_gate_v1"
SOURCE_SCHEMA = "decision_admissibility_wp8_tier2_formal_recovery_source_v1"
IMAGE_DIGEST = (
    "docker.io/haomingwang22/mlevolve@sha256:"
    "fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc"
)
BLOCK_ID = "wp8-tier2-formal-birds-seed-104729-r3"
RECOVERY_POD = "da-wp8-f-birds-s104729-recovery-cpu-r1"
EVALUATOR_POD = "da-wp8-f-birds-s104729-cpu-r3"
PROTECTED_POD = "jupyter-a10-d48dfd589-pqfkb"
TERMINAL_FILENAMES = {
    "all_candidate_terminal_scores.json",
    "fixed_holdout_scores.json",
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _tree_hash(inventory: Mapping[str, str]) -> str:
    return hashlib.sha256(_canonical(dict(inventory))).hexdigest()


def _pod(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected Pod document: {path}")
    return value


def _mounts(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    container = (document.get("spec", {}).get("containers") or [{}])[0]
    return {
        str(row.get("mountPath") or ""): dict(row)
        for row in container.get("volumeMounts") or []
    }


def _env(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    container = (document.get("spec", {}).get("containers") or [{}])[0]
    return {str(row.get("name") or ""): dict(row) for row in container.get("env") or []}


def _resources(document: Mapping[str, Any]) -> dict[str, Any]:
    container = (document.get("spec", {}).get("containers") or [{}])[0]
    return dict(container.get("resources") or {})


def verify_recovery_stop_gate(
    *,
    recovery_root: Path,
    output_root: Path,
    amendment_path: Path,
    amendment_verification_path: Path,
    diagnostic_path: Path,
    source_manifest_path: Path,
    original_staging_gate_path: Path,
    original_staging_content_path: Path,
    recovery_pod_path: Path,
    evaluator_pod_path: Path,
) -> dict[str, Any]:
    recovery_root = recovery_root.resolve()
    output_root = output_root.resolve()
    paths = [
        amendment_path,
        amendment_verification_path,
        diagnostic_path,
        source_manifest_path,
        original_staging_gate_path,
        original_staging_content_path,
        recovery_pod_path,
        evaluator_pod_path,
    ]
    for path in paths:
        if not path.resolve().is_file():
            raise ValueError(f"Recovery Stop Gate input is absent: {path}")
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: object) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(name)

    amendment = _read(amendment_path)
    amendment_report = verify_preterminal_recovery_amendment(
        amendment_path, repo_root=recovery_root
    )
    frozen_report = _read(amendment_verification_path)
    check("amendment_verified_live", amendment_report.get("verified") is True)
    check("amendment_live_errors_empty", amendment_report.get("errors") == [])
    check("amendment_verified_frozen", frozen_report.get("verified") is True)
    check("amendment_frozen_errors_empty", frozen_report.get("errors") == [])
    check(
        "amendment_verification_agrees",
        frozen_report.get("amendment_file_sha256")
        == amendment_report.get("amendment_file_sha256")
        == _sha256_file(amendment_path),
    )

    source_manifest = _read(source_manifest_path)
    check("source_schema", source_manifest.get("schema") == SOURCE_SCHEMA)
    check(
        "source_manifest_hash",
        source_manifest.get("manifest_hash")
        == _payload_hash(source_manifest, "manifest_hash"),
    )
    check(
        "source_amendment_binding",
        source_manifest.get("amendment_hash") == amendment.get("amendment_hash"),
    )
    source_files = source_manifest.get("file_hashes") or {}
    changed = []
    for relative, expected in source_files.items():
        path = recovery_root / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            changed.append(str(relative))
    check("source_file_count", len(source_files) == source_manifest.get("file_count"))
    check("source_files_unchanged", not changed)
    check("source_read_only", not bool(recovery_root.stat().st_mode & stat.S_IWUSR))
    check(
        "source_contains_no_training_data",
        source_manifest.get("contains_training_data") is False,
    )
    check(
        "source_contains_no_terminal_labels",
        source_manifest.get("contains_terminal_labels") is False,
    )
    check(
        "source_contains_no_solver_secret",
        source_manifest.get("contains_solver_secret") is False,
    )
    check(
        "source_forbids_reexecution",
        source_manifest.get("candidate_or_agent_reexecution_authorized") is False,
    )

    diagnostic = _read(diagnostic_path)
    trigger = amendment.get("triggering_failure") or {}
    check(
        "diagnostic_hash",
        diagnostic.get("diagnostic_hash")
        == _payload_hash(diagnostic, "diagnostic_hash"),
    )
    check(
        "diagnostic_block",
        diagnostic.get("block_id") == trigger.get("block_id") == BLOCK_ID,
    )
    preserved = diagnostic.get("preserved_output") or {}
    inventory = _tree_inventory(output_root)
    check(
        "output_file_count",
        len(inventory) == preserved.get("file_count_before_recovery"),
    )
    check(
        "output_tree_hash",
        _tree_hash(inventory) == preserved.get("tree_sha256_before_recovery"),
    )
    check(
        "output_launcher_failed",
        (output_root / "STATE").read_text(encoding="utf-8").strip()
        == "training_launcher_failed",
    )
    check(
        "output_launcher_nonzero",
        int(
            (output_root / "TRAINING_LAUNCHER_EXIT_CODE")
            .read_text(encoding="utf-8")
            .strip()
        )
        != 0,
    )
    check(
        "output_no_training_manifest",
        not (output_root / "TRAINING_MANIFEST.json").exists(),
    )
    check(
        "output_no_training_complete", not (output_root / "TRAINING_COMPLETE").exists()
    )
    check(
        "output_no_evaluation_summary",
        not (output_root / "EVALUATION_SUMMARY.json").exists(),
    )
    terminal_paths = [
        path for path in output_root.rglob("*.json") if path.name in TERMINAL_FILENAMES
    ]
    check("output_terminal_artifact_count_zero", not terminal_paths)
    check(
        "terminal_metric_unobserved", preserved.get("terminal_metric_observed") is False
    )
    check(
        "terminal_scores_uninspected",
        preserved.get("terminal_score_values_inspected") is False,
    )
    lifecycle = diagnostic.get("kubernetes_lifecycle") or {}
    check("training_pod_not_found", lifecycle.get("training_pod_not_found") is True)
    check("evaluator_pod_not_found", lifecycle.get("evaluator_pod_not_found") is True)
    check(
        "not_found_probe_hashes",
        all(
            len(str(lifecycle.get(field) or "")) == 64
            for field in (
                "training_not_found_probe_sha256",
                "evaluator_not_found_probe_sha256",
            )
        ),
    )

    original_gate = _read(original_staging_gate_path)
    original_content = _read(original_staging_content_path)
    check(
        "original_gate_hash",
        original_gate.get("gate_hash") == _payload_hash(original_gate, "gate_hash"),
    )
    check(
        "original_gate_passed",
        original_gate.get("status") == "passed"
        and original_gate.get("formal_training_authorized") is True,
    )
    check(
        "original_content_hash",
        original_content.get("manifest_hash")
        == _payload_hash(original_content, "manifest_hash"),
    )
    check(
        "original_gate_content_binding",
        original_gate.get("staging_content_manifest_hash")
        == original_content.get("manifest_hash"),
    )
    entry = (original_content.get("blocks_by_id") or {}).get(BLOCK_ID) or {}
    check(
        "original_block_bound",
        entry.get("block_id") == BLOCK_ID
        and entry.get("output_root") == preserved.get("path"),
    )
    check(
        "original_source_bound",
        original_content.get("source_snapshot_sha256")
        == diagnostic.get("frozen_execution", {}).get("source_snapshot_sha256"),
    )

    recovery_pod = _pod(recovery_pod_path)
    evaluator_pod = _pod(evaluator_pod_path)
    for label, document, expected_name in (
        ("recovery", recovery_pod, RECOVERY_POD),
        ("evaluator", evaluator_pod, EVALUATOR_POD),
    ):
        metadata = document.get("metadata") or {}
        spec = document.get("spec") or {}
        container = (spec.get("containers") or [{}])[0]
        resources = _resources(document)
        limits = resources.get("limits") or {}
        requests = resources.get("requests") or {}
        check(f"{label}_kind_pod", document.get("kind") == "Pod")
        check(f"{label}_name", metadata.get("name") == expected_name)
        check(f"{label}_not_protected", metadata.get("name") != PROTECTED_POD)
        check(f"{label}_namespace", metadata.get("namespace") == "ecepxie")
        check(
            f"{label}_devpod",
            (metadata.get("labels") or {}).get("execution-kind") == "devpod",
        )
        check(f"{label}_restart_never", spec.get("restartPolicy") == "Never")
        check(f"{label}_image", container.get("image") == IMAGE_DIGEST)
        check(
            f"{label}_cpu_only",
            not any(str(key).startswith("nvidia.com/") for key in {*limits, *requests}),
        )
        check(f"{label}_resource_symmetry", limits == requests)
        check(
            f"{label}_sleep_devpod",
            "sleep infinity" in "\n".join(map(str, container.get("args") or [])),
        )

    recovery_mounts = _mounts(recovery_pod)
    check(
        "recovery_mount_set",
        set(recovery_mounts)
        == {"/opt/nautilus", "/recovery", "/memory", "/output", "/work"},
    )
    for path in ("/opt/nautilus", "/recovery", "/memory"):
        check(
            f"recovery_read_only:{path}",
            recovery_mounts.get(path, {}).get("readOnly") is True,
        )
    check(
        "recovery_output_writable",
        recovery_mounts.get("/output", {}).get("readOnly") is not True,
    )
    check(
        "recovery_source_subpath",
        recovery_mounts.get("/recovery", {}).get("subPath") == recovery_root.name,
    )
    check(
        "recovery_output_subpath",
        str(recovery_mounts.get("/output", {}).get("subPath") or "").endswith(
            f"blocks/{BLOCK_ID}"
        ),
    )
    check(
        "recovery_no_terminal_mount",
        "/fixed/evaluator_view" not in recovery_mounts
        and "/task" not in recovery_mounts,
    )
    check("recovery_no_secret_mount", "/secrets/mlevolve.env" not in recovery_mounts)

    evaluator_mounts = _mounts(evaluator_pod)
    check(
        "evaluator_mount_set",
        set(evaluator_mounts)
        == {
            "/opt/nautilus",
            "/recovery",
            "/fixed/train_view",
            "/fixed/evaluator_view",
            "/contract",
            "/output",
            "/work",
        },
    )
    for path in (
        "/opt/nautilus",
        "/recovery",
        "/fixed/train_view",
        "/fixed/evaluator_view",
        "/contract",
    ):
        check(
            f"evaluator_read_only:{path}",
            evaluator_mounts.get(path, {}).get("readOnly") is True,
        )
    check("evaluator_no_memory", "/memory" not in evaluator_mounts)
    check("evaluator_no_secret", "/secrets/mlevolve.env" not in evaluator_mounts)
    check(
        "evaluator_recovery_subpath",
        evaluator_mounts.get("/recovery", {}).get("subPath") == recovery_root.name,
    )
    evaluator_env = _env(evaluator_pod)
    check(
        "evaluator_overlay_env",
        evaluator_env.get("WP8_FORMAL_FINALIZER_SOURCE_ROOT", {}).get("value")
        == "/recovery",
    )

    required_overlay = set(
        (amendment.get("implementation_correction") or {}).get("recovery_overlay_paths")
        or []
    )
    check("overlay_paths_declared", bool(required_overlay))
    check(
        "overlay_paths_manifest_bound",
        required_overlay <= set(source_manifest.get("overlay_paths") or []),
    )
    check(
        "recovery_pod_yaml_bound",
        _sha256_file(recovery_pod_path)
        == (amendment.get("recovery_pod_specs") or {}).get("recovery_pod_yaml_sha256"),
    )
    check(
        "evaluator_pod_yaml_bound",
        _sha256_file(evaluator_pod_path)
        == (amendment.get("recovery_pod_specs") or {}).get("evaluator_pod_yaml_sha256"),
    )

    gate: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if not errors else "failed",
        "block_id": BLOCK_ID,
        "recovery_authorized": not errors,
        "evaluator_authorized_only_after_recovery_pod_not_found": not errors,
        "terminal_metric_observed": False,
        "terminal_score_values_inspected": False,
        "candidate_or_agent_reexecution_authorized": False,
        "amendment_hash": amendment.get("amendment_hash", ""),
        "amendment_verification_hash": frozen_report.get("verification_hash", ""),
        "diagnostic_hash": diagnostic.get("diagnostic_hash", ""),
        "recovery_source_manifest_hash": source_manifest.get("manifest_hash", ""),
        "original_staging_gate_hash": original_gate.get("gate_hash", ""),
        "original_staging_content_hash": original_content.get("manifest_hash", ""),
        "pre_recovery_output_tree_sha256": _tree_hash(inventory),
        "recovery_pod_yaml_sha256": _sha256_file(recovery_pod_path),
        "evaluator_pod_yaml_sha256": _sha256_file(evaluator_pod_path),
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": sorted(set(errors)),
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "gate_hash": "",
    }
    gate["gate_hash"] = _payload_hash(gate, "gate_hash")
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--amendment-verification", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--original-staging-gate", type=Path, required=True)
    parser.add_argument("--original-staging-content", type=Path, required=True)
    parser.add_argument("--recovery-pod", type=Path, required=True)
    parser.add_argument("--evaluator-pod", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = verify_recovery_stop_gate(
        recovery_root=args.recovery_root,
        output_root=args.output_root,
        amendment_path=args.amendment,
        amendment_verification_path=args.amendment_verification,
        diagnostic_path=args.diagnostic,
        source_manifest_path=args.source_manifest,
        original_staging_gate_path=args.original_staging_gate,
        original_staging_content_path=args.original_staging_content,
        recovery_pod_path=args.recovery_pod,
        evaluator_pod_path=args.evaluator_pod,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(gate, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(gate, indent=2, sort_keys=True))
    raise SystemExit(0 if gate["recovery_authorized"] else 1)


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "verify_recovery_stop_gate"]
