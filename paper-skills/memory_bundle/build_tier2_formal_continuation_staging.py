#!/usr/bin/env python3
"""Build immutable staging for the five remaining formal Tier-2 blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping
import uuid

import yaml


ROOT = Path(__file__).resolve().parents[2]
MLEVOLVE = ROOT / "mlevolve"
if str(MLEVOLVE) not in os.sys.path:
    os.sys.path.insert(0, str(MLEVOLVE))

from authority.memory_snapshot import ImmutableBaseBundle  # noqa: E402
from fixed_holdout.common import sha256_file  # noqa: E402
from fixed_holdout.formal_runtime import (  # noqa: E402
    BLOCK_TEMPLATE_SCHEMA,
    SOURCE_SCHEMA,
    payload_hash,
    read_object,
    verify_source_snapshot,
)
from build_tier2_formal_staging import (  # noqa: E402
    IMAGE_DIGEST,
    TASK_ALIASES,
    TRAINING_GPU_RESOURCE_KEY,
    _pod_base,
    _pvc_mount,
    _render_evaluator_pod,
    _render_training_pod,
    _tree_inventory_hash,
)
from verify_tier2_formal_continuation_amendment import (  # noqa: E402
    verify_continuation_amendment,
)


CONTENT_SCHEMA = (
    "decision_admissibility_wp8_tier2_formal_continuation_staging_content_v1"
)
BUILD_SCHEMA = "decision_admissibility_wp8_tier2_formal_continuation_staging_build_v1"
REVISION = "r4"
CONTROLLER_NAME = "da-wp8-f-controller-cpu-r4"
CONTROLLER_YAML = "formal-controller-cpu-r4.yaml"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _render_controller(
    *,
    source_root: Path,
    staging_root: Path,
    output_root: Path,
    content_hash: str,
) -> dict[str, Any]:
    pod = _pod_base(
        CONTROLLER_NAME,
        role="formal-controller",
        image=IMAGE_DIGEST,
    )
    pod["spec"]["activeDeadlineSeconds"] = 259200
    container = pod["spec"]["containers"][0]
    container["env"].append(
        {"name": "WP8_FORMAL_STAGING_CONTENT_HASH", "value": content_hash}
    )
    container["resources"] = {
        "requests": {"cpu": "1", "memory": "2Gi"},
        "limits": {"cpu": "1", "memory": "2Gi"},
    }
    container["volumeMounts"] = [
        _pvc_mount("/opt/nautilus", source_root, read_only=True),
        _pvc_mount("/formal/staging", staging_root, read_only=True),
        _pvc_mount("/formal/outputs", output_root, read_only=False),
        {"name": "work", "mountPath": "/work"},
    ]
    return pod


def _verify_completed_blocks(
    freeze: Mapping[str, Any], completed_output_root: Path
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for block_id, row_value in (freeze.get("blocks") or {}).items():
        row = dict(row_value or {})
        root = completed_output_root / "blocks" / str(block_id)
        summary_path = root / "EVALUATION_SUMMARY.json"
        deletion_path = root / "EVALUATOR_POD_DELETION_ATTESTATION.json"
        summary = read_object(summary_path)
        deletion = read_object(deletion_path)
        if (
            sha256_file(summary_path) != row.get("evaluation_summary_file_sha256")
            or summary.get("summary_hash") != payload_hash(summary, "summary_hash")
            or summary.get("summary_hash") != row.get("evaluation_summary_hash")
        ):
            raise ValueError(f"Completed summary changed: {block_id}")
        result_facts = sum(
            int(value.get("result_fact_count", 0))
            for value in (summary.get("online_conditions") or {}).values()
        )
        if (
            summary.get("successful_selected_result_count")
            != row.get("successful_selected_result_count")
            or summary.get("failed_online_condition_count")
            != row.get("failed_online_condition_count")
            or result_facts != row.get("result_fact_count")
        ):
            raise ValueError(f"Completed structural counts changed: {block_id}")
        if (
            sha256_file(deletion_path) != row.get("evaluator_pod_deletion_file_sha256")
            or deletion.get("attestation_hash")
            != payload_hash(deletion, "attestation_hash")
            or deletion.get("attestation_hash")
            != row.get("evaluator_pod_deletion_attestation_hash")
            or deletion.get("not_found_verified") is not True
        ):
            raise ValueError(f"Completed evaluator deletion changed: {block_id}")
        manifest_sha = row.get("block_evaluation_file_manifest_sha256")
        manifest_path = root / "BLOCK_EVALUATION_FILE_MANIFEST.json"
        if manifest_sha is not None and (
            not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha
        ):
            raise ValueError(f"Completed file manifest changed: {block_id}")
        if row.get("may_rerun") is not False:
            raise ValueError(f"Completed block is marked retryable: {block_id}")
        verified[str(block_id)] = {
            "evaluation_summary_hash": summary["summary_hash"],
            "evaluation_summary_file_sha256": sha256_file(summary_path),
            "successful_selected_result_count": summary[
                "successful_selected_result_count"
            ],
            "failed_online_condition_count": summary["failed_online_condition_count"],
            "result_fact_count": result_facts,
            "evaluator_pod_deletion_attestation_hash": deletion["attestation_hash"],
            "evaluator_pod_deletion_file_sha256": sha256_file(deletion_path),
            "may_rerun": False,
        }
    if len(verified) != 4:
        raise ValueError("Continuation requires exactly four completed blocks")
    return verified


def build_continuation_staging(
    *,
    source_root: Path,
    artifact_root: Path,
    parent_staging_root: Path,
    parent_gate_path: Path,
    completed_output_root: Path,
    staging_root: Path,
    output_root: Path,
    preregistration_paths: tuple[Path, ...],
    continuation_amendment_path: Path,
    continuation_verification_path: Path,
    completed_freeze_path: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    artifact_root = artifact_root.resolve()
    parent_staging_root = parent_staging_root.resolve()
    parent_gate_path = parent_gate_path.resolve()
    completed_output_root = completed_output_root.resolve()
    staging_root = staging_root.resolve()
    output_root = output_root.resolve()
    for target in (staging_root, output_root):
        if target.exists():
            raise FileExistsError(target)
    source_manifest_path = source_root / "WP8_TIER2_SOURCE_MANIFEST.json"
    source_manifest = read_object(source_manifest_path)
    if source_manifest.get("schema") != SOURCE_SCHEMA or source_manifest.get(
        "source_sha256"
    ) != payload_hash(source_manifest, "source_sha256"):
        raise ValueError("Continuation source snapshot is invalid")
    source_report = verify_source_snapshot(
        source_root,
        expected_source_sha256=source_manifest["source_sha256"],
        expected_manifest_file_sha256=sha256_file(source_manifest_path),
    )
    if source_report.get("verified") is not True:
        raise ValueError("Continuation source verification failed")

    amendment = read_object(continuation_amendment_path)
    amendment_report = verify_continuation_amendment(
        continuation_amendment_path, repo_root=ROOT
    )
    frozen_verification = read_object(continuation_verification_path)
    if (
        amendment_report.get("verified") is not True
        or frozen_verification.get("verified") is not True
        or frozen_verification.get("errors") != []
        or frozen_verification.get("amendment_file_sha256")
        != sha256_file(continuation_amendment_path)
    ):
        raise ValueError("Continuation amendment verification failed")
    completed_freeze = read_object(completed_freeze_path)
    completed_ref = amendment.get("completed_blocks_freeze") or {}
    if (
        completed_freeze.get("inventory_hash")
        != payload_hash(completed_freeze, "inventory_hash")
        or completed_freeze.get("inventory_hash") != completed_ref.get("inventory_hash")
        or sha256_file(completed_freeze_path) != completed_ref.get("file_sha256")
    ):
        raise ValueError("Completed block freeze binding mismatch")
    completed_records = _verify_completed_blocks(
        completed_freeze, completed_output_root
    )

    parent_content_path = parent_staging_root / "STAGING_CONTENT_MANIFEST.json"
    parent_content = read_object(parent_content_path)
    parent_gate = read_object(parent_gate_path)
    if (
        parent_content.get("manifest_hash")
        != payload_hash(parent_content, "manifest_hash")
        or parent_gate.get("gate_hash") != payload_hash(parent_gate, "gate_hash")
        or parent_gate.get("status") != "passed"
        or parent_gate.get("staging_content_manifest_hash")
        != parent_content.get("manifest_hash")
    ):
        raise ValueError("Parent r10 staging binding mismatch")

    prereg = [read_object(path) for path in preregistration_paths]
    design = prereg[0]
    if len(prereg) != 7 or prereg[-1].get("preregistration_id") != amendment.get(
        "preregistration_id"
    ):
        raise ValueError("Continuation preregistration chain mismatch")
    order_by_pair = {
        (str(row["task_id"]), int(row["agent_seed"])): list(row["order"])
        for row in design["condition_order_design"]["blocks"]
    }
    remaining = [
        (str(row["task_id"]), int(row["agent_seed"]))
        for row in amendment["continuation_design"]["remaining_blocks"]
    ]
    if remaining != [
        ("mlsp-2013-birds", 130363),
        ("mlsp-2013-birds", 155921),
        ("new-york-city-taxi-fare-prediction", 104729),
        ("new-york-city-taxi-fare-prediction", 130363),
        ("new-york-city-taxi-fare-prediction", 155921),
    ]:
        raise ValueError("Continuation remaining block set changed")

    for task_id in {task for task, _ in remaining}:
        record = parent_content["task_records"][task_id]
        data_root = Path(record["data_root"])
        count, inventory = _tree_inventory_hash(data_root)
        if (
            count != record["data_file_count"]
            or inventory != record["data_inventory_sha256"]
        ):
            raise ValueError(f"Continuation holdout changed: {task_id}")
        bundle_root = Path(record["bundle_root"])
        current = read_object(bundle_root / "CURRENT.json")
        bundle = ImmutableBaseBundle.load(
            bundle_root / str(current["bundle_path"]), verify_artifacts=True
        )
        if (
            bundle.bundle_id != record["bundle_id"]
            or bundle.manifest_sha256 != record["bundle_manifest_sha256"]
            or sha256_file(bundle_root / "CURRENT.json")
            != record["bundle_current_file_sha256"]
        ):
            raise ValueError(f"Continuation Bundle changed: {task_id}")

    staging_tmp = staging_root.with_name(
        f".{staging_root.name}.staging-{uuid.uuid4().hex}"
    )
    output_tmp = output_root.with_name(
        f".{output_root.name}.staging-{uuid.uuid4().hex}"
    )
    staging_tmp.mkdir(parents=True)
    output_tmp.mkdir(parents=True)
    try:
        prereg_dir = staging_tmp / "preregistration"
        prereg_dir.mkdir()
        prereg_files: dict[str, dict[str, str]] = {}
        for path in preregistration_paths:
            target = prereg_dir / path.name
            shutil.copy2(path, target)
            prereg_files[path.name] = {
                "path": str(staging_root / "preregistration" / path.name),
                "sha256": sha256_file(target),
            }
        verification_target = (
            staging_tmp / "reports" / continuation_verification_path.name
        )
        freeze_target = staging_tmp / "reports" / completed_freeze_path.name
        verification_target.parent.mkdir()
        shutil.copy2(continuation_verification_path, verification_target)
        shutil.copy2(completed_freeze_path, freeze_target)

        blocks: dict[str, dict[str, Any]] = {}
        templates: dict[str, dict[str, Any]] = {}
        for task_id, seed in remaining:
            alias = TASK_ALIASES[task_id]
            block_id = f"wp8-tier2-formal-{alias}-seed-{seed}-{REVISION}"
            block_output_tmp = output_tmp / "blocks" / block_id
            block_output_tmp.mkdir(parents=True)
            block_output = output_root / "blocks" / block_id
            record = parent_content["task_records"][task_id]
            template: dict[str, Any] = {
                "schema": BLOCK_TEMPLATE_SCHEMA,
                "block_id": block_id,
                "task_id": task_id,
                "target_task_family": record["target_task_family"],
                "target_domain": record["target_domain"],
                "protocol_ref": record["protocol_ref"],
                "split_id": record["split_id"],
                "metric": record["metric"],
                "maximize": record["maximize"],
                "agent_seed": seed,
                "condition_order": order_by_pair[(task_id, seed)],
                "steps_per_condition": design["search_budget"]["total_steps"],
                "initial_drafts_per_condition": design["search_budget"][
                    "initial_drafts"
                ],
                "agent_time_limit_seconds": design["search_budget"][
                    "agent_time_limit_seconds"
                ],
                "condition_launcher_timeout_seconds": design["search_budget"][
                    "condition_launcher_timeout_seconds"
                ],
                "candidate_execution_contract": record["candidate_execution_contract"],
                "candidate_execution_contract_hash": record[
                    "candidate_execution_contract"
                ]["contract_hash"],
                "source_snapshot_sha256": source_manifest["source_sha256"],
                "source_manifest_file_sha256": sha256_file(source_manifest_path),
                "train_manifest_sha256": record["train_manifest_sha256"],
                "evaluator_manifest_sha256": record["evaluator_manifest_sha256"],
                "bundle_id": record["bundle_id"],
                "bundle_manifest_sha256": record["bundle_manifest_sha256"],
                "bundle_manifest_file_sha256": record["bundle_manifest_file_sha256"],
                "bundle_current_file_sha256": record["bundle_current_file_sha256"],
                "formal_clause_id": record["formal_clause_id"],
                "formal_claim_id": record["formal_claim_id"],
                "container_image_digest": IMAGE_DIGEST,
                "expected_training_pod_name": (
                    f"da-wp8-f-{alias}-s{seed}-gpu-{REVISION}"
                ),
                "expected_training_pod_namespace": "ecepxie",
                "expected_evaluator_pod_name": (
                    f"da-wp8-f-{alias}-s{seed}-cpu-{REVISION}"
                ),
                "expected_evaluator_pod_namespace": "ecepxie",
                "output_root_id": f"{output_root.name}/blocks/{block_id}",
                "template_hash": "",
            }
            template["template_hash"] = payload_hash(template, "template_hash")
            contract_dir = staging_tmp / "blocks" / block_id
            contract_dir.mkdir(parents=True)
            _write_exclusive(contract_dir / "BLOCK_TEMPLATE.json", template)
            templates[block_id] = template
            blocks[block_id] = {
                "block_id": block_id,
                "task_id": task_id,
                "agent_seed": seed,
                "condition_order": order_by_pair[(task_id, seed)],
                "block_template_hash": template["template_hash"],
                "block_template_sha256": sha256_file(
                    contract_dir / "BLOCK_TEMPLATE.json"
                ),
                "contract_root": str(staging_root / "blocks" / block_id),
                "output_root": str(block_output),
                "output_root_initially_empty": True,
                "training_pod_name": template["expected_training_pod_name"],
                "evaluator_pod_name": template["expected_evaluator_pod_name"],
            }

        runtime_paths = (
            "deploy/run_decision_admissibility_wp8_tier2_formal_training_devpod.sh",
            "deploy/run_decision_admissibility_wp8_tier2_formal_evaluator_devpod.sh",
            "mlevolve/fixed_holdout/formal_runtime.py",
            "mlevolve/fixed_holdout/formal_host_receipts.py",
            "mlevolve/fixed_holdout/formal_block_training.py",
            "mlevolve/fixed_holdout/formal_block_evaluate.py",
            "mlevolve/fixed_holdout/formal_prepare.py",
            "mlevolve/fixed_holdout/score_run.py",
            "mlevolve/fixed_holdout/writeback.py",
            "mlevolve/authority/collectors/trusted.py",
            "mlevolve/authority/protocol_compiler.py",
            "mlevolve/authority/runtime_protocol.py",
        )
        runtime_files = {
            relative: sha256_file(source_root / relative) for relative in runtime_paths
        }
        control_paths = (
            "deploy/run_decision_admissibility_wp8_tier2_formal_block.sh",
            "deploy/run_decision_admissibility_wp8_tier2_formal_continuation_staging_pipeline.sh",
            "deploy/stage_decision_admissibility_wp8_tier2_formal_continuation.sh",
            "paper-skills/memory_bundle/build_tier2_formal_continuation_staging.py",
            "paper-skills/memory_bundle/verify_tier2_formal_continuation_amendment.py",
            "paper-skills/memory_bundle/verify_tier2_formal_continuation_staging.py",
        )
        control_files = {
            relative: sha256_file(ROOT / relative) for relative in control_paths
        }
        content: dict[str, Any] = {
            "schema": CONTENT_SCHEMA,
            "status": "continuation_content_frozen_pending_independent_stop_gate",
            "design_preregistration_id": design["preregistration_id"],
            "effective_preregistration_id": amendment["preregistration_id"],
            "preregistration_files": prereg_files,
            "continuation_amendment_verification": {
                "path": str(staging_root / "reports" / verification_target.name),
                "sha256": sha256_file(verification_target),
                "verification_hash": frozen_verification["verification_hash"],
            },
            "completed_blocks_freeze": {
                "path": str(staging_root / "reports" / freeze_target.name),
                "sha256": sha256_file(freeze_target),
                "inventory_hash": completed_freeze["inventory_hash"],
            },
            "completed_blocks": completed_records,
            "completed_block_count": 4,
            "remaining_block_count": 5,
            "remaining_online_condition_count": 25,
            "remaining_oracle_count": 5,
            "combined_online_condition_count": 45,
            "combined_oracle_count": 9,
            "formal_execution_revision": REVISION,
            "training_gpu_resource_key": TRAINING_GPU_RESOURCE_KEY,
            "source_root": str(source_root),
            "source_snapshot_sha256": source_manifest["source_sha256"],
            "source_manifest_file_sha256": sha256_file(source_manifest_path),
            "base_commit": source_manifest["base_commit"],
            "container_image_digest": IMAGE_DIGEST,
            "artifact_root": str(artifact_root),
            "parent_staging_content_hash": parent_content["manifest_hash"],
            "parent_staging_gate_hash": parent_gate["gate_hash"],
            "parent_source_snapshot_sha256": parent_content["source_snapshot_sha256"],
            "task_records": {
                task_id: parent_content["task_records"][task_id]
                for task_id in sorted({task for task, _ in remaining})
            },
            "output_root": str(output_root),
            "output_roots_initially_empty": True,
            "runtime_source_files": runtime_files,
            "control_source_files": control_files,
            "blocks_by_id": blocks,
            "formal_training_started": False,
            "terminal_score_values_inspected": False,
            "terminal_metric_observed_for_remaining_blocks": False,
            "manifest_hash": "",
        }
        content["manifest_hash"] = payload_hash(content, "manifest_hash")
        _write_exclusive(staging_tmp / "STAGING_CONTENT_MANIFEST.json", content)
        for block_id in blocks:
            shutil.copy2(
                staging_tmp / "STAGING_CONTENT_MANIFEST.json",
                staging_tmp / "blocks" / block_id / "STAGING_CONTENT_MANIFEST.json",
            )

        pod_dir = staging_tmp / "pods"
        pod_dir.mkdir()
        pod_hashes: dict[str, dict[str, str]] = {}
        for block_id, block in blocks.items():
            template = templates[block_id]
            task_id = block["task_id"]
            record = parent_content["task_records"][task_id]
            documents = {
                "training": _render_training_pod(
                    template,
                    source_root=source_root,
                    data_root=Path(record["data_root"]),
                    bundle_root=Path(record["bundle_root"]),
                    contract_root=staging_root / "blocks" / block_id,
                    output_root=output_root / "blocks" / block_id,
                    content_hash=content["manifest_hash"],
                ),
                "evaluator": _render_evaluator_pod(
                    template,
                    source_root=source_root,
                    data_root=Path(record["data_root"]),
                    contract_root=staging_root / "blocks" / block_id,
                    output_root=output_root / "blocks" / block_id,
                    content_hash=content["manifest_hash"],
                ),
            }
            for role, document in documents.items():
                pod_path = pod_dir / f"{block_id}-{role}.yaml"
                pod_path.write_text(
                    yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
                )
                pod_path.chmod(0o444)
                pod_hashes[f"{block_id}:{role}"] = {
                    "path": str(staging_root / "pods" / pod_path.name),
                    "sha256": sha256_file(pod_path),
                }
        controller_path = pod_dir / CONTROLLER_YAML
        controller_path.write_text(
            yaml.safe_dump(
                _render_controller(
                    source_root=source_root,
                    staging_root=staging_root,
                    output_root=output_root,
                    content_hash=content["manifest_hash"],
                ),
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        controller_path.chmod(0o444)
        pod_hashes["formal-controller"] = {
            "path": str(staging_root / "pods" / controller_path.name),
            "sha256": sha256_file(controller_path),
        }

        build: dict[str, Any] = {
            "schema": BUILD_SCHEMA,
            "status": "built_pending_independent_stop_gate",
            "staging_root": str(staging_root),
            "output_root": str(output_root),
            "staging_content_manifest_hash": content["manifest_hash"],
            "staging_content_manifest_sha256": sha256_file(
                staging_tmp / "STAGING_CONTENT_MANIFEST.json"
            ),
            "block_count": len(blocks),
            "pod_yaml_count": len(pod_hashes),
            "pod_yamls": pod_hashes,
            "builder_source_sha256": sha256_file(Path(__file__).resolve()),
            "formal_training_started": False,
            "terminal_score_values_inspected": False,
            "terminal_metric_observed_for_remaining_blocks": False,
            "build_hash": "",
        }
        build["build_hash"] = payload_hash(build, "build_hash")
        _write_exclusive(staging_tmp / "STAGING_BUILD_REPORT.json", build)
        os.replace(output_tmp, output_root)
        os.replace(staging_tmp, staging_root)
        return read_object(staging_root / "STAGING_BUILD_REPORT.json")
    except BaseException:
        shutil.rmtree(staging_tmp, ignore_errors=True)
        shutil.rmtree(output_tmp, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--parent-staging-root", type=Path, required=True)
    parser.add_argument("--parent-gate", type=Path, required=True)
    parser.add_argument("--completed-output-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, action="append", required=True)
    parser.add_argument("--continuation-amendment", type=Path, required=True)
    parser.add_argument("--continuation-verification", type=Path, required=True)
    parser.add_argument("--completed-freeze", type=Path, required=True)
    args = parser.parse_args()
    result = build_continuation_staging(
        source_root=args.source_root,
        artifact_root=args.artifact_root,
        parent_staging_root=args.parent_staging_root,
        parent_gate_path=args.parent_gate,
        completed_output_root=args.completed_output_root,
        staging_root=args.staging_root,
        output_root=args.output_root,
        preregistration_paths=tuple(args.preregistration),
        continuation_amendment_path=args.continuation_amendment,
        continuation_verification_path=args.continuation_verification,
        completed_freeze_path=args.completed_freeze,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "BUILD_SCHEMA",
    "CONTENT_SCHEMA",
    "REVISION",
    "build_continuation_staging",
]
