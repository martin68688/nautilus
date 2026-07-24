#!/usr/bin/env python3
"""Independently verify the five-block formal continuation staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
MLEVOLVE = ROOT / "mlevolve"
if str(MLEVOLVE) not in os.sys.path:
    os.sys.path.insert(0, str(MLEVOLVE))

from authority.memory_snapshot import ImmutableBaseBundle  # noqa: E402
from fixed_holdout.common import sha256_file  # noqa: E402
from fixed_holdout.formal_runtime import (  # noqa: E402
    payload_hash,
    read_object,
    verify_source_snapshot,
)
from build_tier2_formal_continuation_staging import (  # noqa: E402
    BUILD_SCHEMA,
    CONTENT_SCHEMA,
    REVISION,
    _verify_completed_blocks,
)
from build_tier2_formal_staging import (  # noqa: E402
    IMAGE_DIGEST,
    TRAINING_GPU_RESOURCE_KEY,
    _tree_inventory_hash,
)
from verify_tier2_formal_continuation_amendment import (  # noqa: E402
    verify_continuation_amendment,
)


SCHEMA = "decision_admissibility_wp8_tier2_formal_continuation_stop_gate_v1"
PROTECTED_POD = "jupyter-a10-d48dfd589-pqfkb"
EXPECTED_PAIRS = [
    ("mlsp-2013-birds", 130363),
    ("mlsp-2013-birds", 155921),
    ("new-york-city-taxi-fare-prediction", 104729),
    ("new-york-city-taxi-fare-prediction", 130363),
    ("new-york-city-taxi-fare-prediction", 155921),
]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_digest(payload: Mapping[str, Any], field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned)).hexdigest()


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


def _resources(document: Mapping[str, Any]) -> dict[str, Any]:
    container = (document.get("spec", {}).get("containers") or [{}])[0]
    return dict(container.get("resources") or {})


def _source_diff(
    parent_manifest: Mapping[str, Any],
    current_manifest: Mapping[str, Any],
) -> list[str]:
    before = dict(parent_manifest.get("file_hashes") or {})
    after = dict(current_manifest.get("file_hashes") or {})
    return sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )


def _seal(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink() or not path.exists():
            continue
        path.chmod(path.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    root.chmod(root.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def verify_continuation_staging(
    staging_root: Path,
    *,
    repo_root: Path = ROOT,
    seal_on_success: bool = True,
) -> dict[str, Any]:
    staging_root = staging_root.resolve()
    repo_root = repo_root.resolve()
    content_path = staging_root / "STAGING_CONTENT_MANIFEST.json"
    build_path = staging_root / "STAGING_BUILD_REPORT.json"
    content = read_object(content_path)
    build = read_object(build_path)
    checks: dict[str, bool] = {}
    errors: list[str] = []
    evidence: dict[str, Any] = {}

    def check(name: str, condition: object) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(name)

    check("content_schema", content.get("schema") == CONTENT_SCHEMA)
    check(
        "content_hash",
        content.get("manifest_hash") == payload_hash(content, "manifest_hash"),
    )
    check("build_schema", build.get("schema") == BUILD_SCHEMA)
    check("build_hash", build.get("build_hash") == payload_hash(build, "build_hash"))
    check(
        "build_content_binding",
        build.get("staging_content_manifest_hash") == content.get("manifest_hash")
        and build.get("staging_content_manifest_sha256") == sha256_file(content_path),
    )
    check("revision", content.get("formal_execution_revision") == REVISION == "r4")
    check("remaining_block_count", content.get("remaining_block_count") == 5)
    check(
        "remaining_online_count",
        content.get("remaining_online_condition_count") == 25,
    )
    check("remaining_oracle_count", content.get("remaining_oracle_count") == 5)
    check("combined_online_count", content.get("combined_online_condition_count") == 45)
    check("combined_oracle_count", content.get("combined_oracle_count") == 9)
    check("formal_not_started", content.get("formal_training_started") is False)
    check(
        "remaining_terminal_unobserved",
        content.get("terminal_metric_observed_for_remaining_blocks") is False,
    )
    check(
        "terminal_values_uninspected",
        content.get("terminal_score_values_inspected") is False,
    )

    source_report = verify_source_snapshot(
        content["source_root"],
        expected_source_sha256=content["source_snapshot_sha256"],
        expected_manifest_file_sha256=content["source_manifest_file_sha256"],
    )
    check("source_verified", source_report.get("verified") is True)
    current_source_manifest = read_object(
        Path(content["source_root"]) / "WP8_TIER2_SOURCE_MANIFEST.json"
    )
    parent_content_path = Path(
        "/workspace/decision-admissibility-wp8-tier2-formal-staging-r12/"
        "STAGING_CONTENT_MANIFEST.json"
    )
    parent_content = read_object(parent_content_path)
    parent_source_manifest = read_object(
        Path(parent_content["source_root"]) / "WP8_TIER2_SOURCE_MANIFEST.json"
    )
    check(
        "parent_content_hash",
        parent_content.get("manifest_hash")
        == payload_hash(parent_content, "manifest_hash")
        == content.get("parent_staging_content_hash"),
    )
    changed_paths = _source_diff(parent_source_manifest, current_source_manifest)
    amendment_path = (
        staging_root
        / "preregistration"
        / ("decision_admissibility_wp8_tier2_formal_preregistration_20260723_r7.json")
    )
    amendment = read_object(amendment_path)
    allowed_changes = sorted(
        amendment.get("implementation_correction", {}).get(
            "allowed_runtime_source_changes"
        )
        or []
    )
    check("source_diff_exact", changed_paths == allowed_changes)
    check(
        "source_diff_excludes_agent_generation",
        not any(
            path.startswith("mlevolve/agents/")
            or path.startswith("mlevolve/engine/agent_search")
            for path in changed_paths
        ),
    )
    evidence["source_diff_paths"] = changed_paths
    evidence["source_verification_hash"] = source_report.get("report_hash", "")

    amendment_live = verify_continuation_amendment(amendment_path, repo_root=repo_root)
    frozen_pointer = content.get("continuation_amendment_verification") or {}
    frozen_path = Path(str(frozen_pointer.get("path") or ""))
    frozen = read_object(frozen_path)
    check("amendment_live_verified", amendment_live.get("verified") is True)
    check("amendment_frozen_verified", frozen.get("verified") is True)
    check(
        "amendment_verification_bound",
        sha256_file(frozen_path) == frozen_pointer.get("sha256")
        and frozen.get("verification_hash")
        == frozen_pointer.get("verification_hash")
        == amendment_live.get("verification_hash"),
    )
    check(
        "effective_preregistration",
        content.get("effective_preregistration_id")
        == amendment.get("preregistration_id"),
    )

    freeze_pointer = content.get("completed_blocks_freeze") or {}
    freeze_path = Path(str(freeze_pointer.get("path") or ""))
    freeze = read_object(freeze_path)
    check(
        "completed_freeze_bound",
        sha256_file(freeze_path) == freeze_pointer.get("sha256")
        and freeze.get("inventory_hash")
        == _payload_digest(freeze, "inventory_hash")
        == freeze_pointer.get("inventory_hash"),
    )
    completed_root = Path("/workspace/decision-admissibility-wp8-tier2-formal-runs-r10")
    verified_completed = _verify_completed_blocks(freeze, completed_root)
    check(
        "completed_records_exact", verified_completed == content.get("completed_blocks")
    )
    check("completed_block_count", len(verified_completed) == 4)
    check(
        "completed_blocks_not_in_new_output",
        not any(
            (Path(content["output_root"]) / "blocks" / block_id).exists()
            for block_id in verified_completed
        ),
    )

    pairs = sorted(
        (str(row["task_id"]), int(row["agent_seed"]))
        for row in (content.get("blocks_by_id") or {}).values()
    )
    check("remaining_pairs_exact", pairs == sorted(EXPECTED_PAIRS))
    blocks = content.get("blocks_by_id") or {}
    check("five_block_records", len(blocks) == 5)
    output_root = Path(content["output_root"])
    check("output_root_exists", output_root.is_dir())
    check(
        "output_block_set",
        {path.name for path in (output_root / "blocks").iterdir() if path.is_dir()}
        == set(blocks),
    )
    for block_id, row in blocks.items():
        block_output = Path(row["output_root"])
        check(
            f"output_empty:{block_id}",
            block_output.is_dir()
            and not any(block_output.iterdir())
            and row.get("output_root_initially_empty") is True,
        )
        template_path = staging_root / "blocks" / block_id / "BLOCK_TEMPLATE.json"
        template = read_object(template_path)
        check(
            f"template_hash:{block_id}",
            template.get("template_hash")
            == payload_hash(template, "template_hash")
            == row.get("block_template_hash")
            and sha256_file(template_path) == row.get("block_template_sha256"),
        )
        check(
            f"template_revision:{block_id}",
            block_id.endswith("-r4")
            and str(template.get("expected_training_pod_name") or "").endswith("-r4")
            and str(template.get("expected_evaluator_pod_name") or "").endswith("-r4"),
        )
        copied_content = (
            staging_root / "blocks" / block_id / "STAGING_CONTENT_MANIFEST.json"
        )
        check(
            f"content_copy:{block_id}",
            sha256_file(copied_content) == sha256_file(content_path),
        )

    for task_id, record in content.get("task_records", {}).items():
        data_root = Path(record["data_root"])
        count, inventory = _tree_inventory_hash(data_root)
        check(f"data_count:{task_id}", count == record["data_file_count"])
        check(
            f"data_inventory:{task_id}",
            inventory == record["data_inventory_sha256"],
        )
        bundle_root = Path(record["bundle_root"])
        current = read_object(bundle_root / "CURRENT.json")
        bundle = ImmutableBaseBundle.load(
            bundle_root / str(current["bundle_path"]), verify_artifacts=True
        )
        check(
            f"bundle_identity:{task_id}",
            bundle.bundle_id == record["bundle_id"]
            and bundle.manifest_sha256 == record["bundle_manifest_sha256"]
            and sha256_file(bundle_root / "CURRENT.json")
            == record["bundle_current_file_sha256"],
        )

    for relative, expected in (content.get("runtime_source_files") or {}).items():
        check(
            f"runtime_hash:{relative}",
            sha256_file(Path(content["source_root"]) / relative) == expected,
        )
    for relative, expected in (content.get("control_source_files") or {}).items():
        check(
            f"control_hash:{relative}",
            sha256_file(repo_root / relative) == expected,
        )

    pod_rows = build.get("pod_yamls") or {}
    check("pod_yaml_count", len(pod_rows) == 11)
    for key, row in pod_rows.items():
        pod_path = Path(row["path"])
        check(f"pod_hash:{key}", sha256_file(pod_path) == row["sha256"])
        document = _pod(pod_path)
        metadata = document.get("metadata") or {}
        spec = document.get("spec") or {}
        container = (spec.get("containers") or [{}])[0]
        mounts = _mounts(document)
        resources = _resources(document)
        limits = resources.get("limits") or {}
        requests = resources.get("requests") or {}
        check(f"pod_kind:{key}", document.get("kind") == "Pod")
        check(f"pod_namespace:{key}", metadata.get("namespace") == "ecepxie")
        check(f"pod_not_protected:{key}", metadata.get("name") != PROTECTED_POD)
        check(
            f"pod_devpod:{key}",
            (metadata.get("labels") or {}).get("execution-kind") == "devpod",
        )
        check(f"pod_restart:{key}", spec.get("restartPolicy") == "Never")
        check(f"pod_image:{key}", container.get("image") == IMAGE_DIGEST)
        check(f"pod_resources:{key}", limits == requests)
        check(
            f"pod_source_read_only:{key}",
            (mounts.get("/opt/nautilus") or {}).get("readOnly") is True,
        )
        if key.endswith(":training"):
            check(
                f"training_gpu:{key}",
                limits.get(TRAINING_GPU_RESOURCE_KEY) == "1",
            )
            check(
                f"training_mounts:{key}",
                set(mounts)
                == {
                    "/opt/nautilus",
                    "/task",
                    "/memory",
                    "/contract",
                    "/output",
                    "/secrets/mlevolve.env",
                    "/work",
                    "/cache",
                },
            )
            check(
                f"training_no_terminal:{key}",
                "/fixed/evaluator_view" not in mounts,
            )
        elif key.endswith(":evaluator"):
            check(
                f"evaluator_cpu:{key}",
                not any(str(name).startswith("nvidia.com/") for name in limits),
            )
            check(
                f"evaluator_mounts:{key}",
                set(mounts)
                == {
                    "/opt/nautilus",
                    "/fixed/train_view",
                    "/fixed/evaluator_view",
                    "/contract",
                    "/output",
                    "/work",
                },
            )
            check(f"evaluator_no_memory:{key}", "/memory" not in mounts)
            check(
                f"evaluator_no_secret:{key}",
                "/secrets/mlevolve.env" not in mounts,
            )
        else:
            check(
                "controller_name", metadata.get("name") == "da-wp8-f-controller-cpu-r4"
            )
            check(
                "controller_mounts",
                set(mounts)
                == {"/opt/nautilus", "/formal/staging", "/formal/outputs", "/work"},
            )

    gate: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if not errors else "failed",
        "formal_training_authorized": not errors,
        "authorized_block_count": 5 if not errors else 0,
        "authorized_online_condition_count": 25 if not errors else 0,
        "authorized_oracle_count": 5 if not errors else 0,
        "completed_blocks_authorized_to_rerun": False,
        "terminal_score_values_inspected": False,
        "terminal_metric_observed_for_remaining_blocks": False,
        "effect_claim_authorized": False,
        "staging_content_manifest_hash": content.get("manifest_hash", ""),
        "continuation_amendment_hash": amendment.get("amendment_hash", ""),
        "continuation_verification_hash": frozen.get("verification_hash", ""),
        "completed_freeze_hash": freeze.get("inventory_hash", ""),
        "source_snapshot_sha256": content.get("source_snapshot_sha256", ""),
        "parent_source_snapshot_sha256": content.get(
            "parent_source_snapshot_sha256", ""
        ),
        "source_diff_paths": changed_paths,
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": sorted(set(errors)),
        "evidence": evidence,
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "gate_hash": "",
    }
    gate["gate_hash"] = _payload_digest(gate, "gate_hash")
    if not errors and seal_on_success:
        _seal(staging_root)
    return gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-seal", action="store_true")
    args = parser.parse_args()
    gate = verify_continuation_staging(
        args.staging_root,
        repo_root=args.repo_root,
        seal_on_success=not args.no_seal,
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
    raise SystemExit(0 if gate["formal_training_authorized"] else 1)


if __name__ == "__main__":
    main()


__all__ = ["SCHEMA", "verify_continuation_staging"]
