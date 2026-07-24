#!/usr/bin/env python3
"""Build the immutable, result-blind WP8 Tier-2 formal staging packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
MLEVOLVE = ROOT / "mlevolve"
if str(MLEVOLVE) not in os.sys.path:
    os.sys.path.insert(0, str(MLEVOLVE))

from authority.memory_snapshot import ImmutableBaseBundle  # noqa: E402
from engine.candidate_execution_contract import (  # noqa: E402
    build_candidate_execution_contract,
)
from fixed_holdout.common import sha256_file  # noqa: E402
from fixed_holdout.formal_runtime import (  # noqa: E402
    BLOCK_TEMPLATE_SCHEMA,
    SOURCE_SCHEMA,
    STAGING_CONTENT_SCHEMA,
    payload_hash,
    read_object,
)
from verify_tier2_formal_postfailure_amendment import (  # noqa: E402
    verify_postfailure_amendment,
)


BUILD_SCHEMA = "decision_admissibility_wp8_tier2_formal_staging_build_v1"
IMAGE_DIGEST = (
    "docker.io/haomingwang22/mlevolve@sha256:"
    "fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc"
)
TASK_ALIASES = {
    "aerial-cactus-identification": "aerial",
    "mlsp-2013-birds": "birds",
    "new-york-city-taxi-fare-prediction": "taxi",
}
EXCLUDED_GPU_NODES = (
    "k8s-chase-ci-07.calit2.optiputer.net",
    "ucm-fiona01.ucmerced.edu",
)
FORMAL_EXECUTION_REVISION = "r3"
FORMAL_CONTROLLER_NAME = "da-wp8-f-controller-cpu-r3"
FORMAL_CONTROLLER_YAML = "formal-controller-cpu-r3.yaml"
TRAINING_GPU_RESOURCE_KEY = "nvidia.com/rtxa6000"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _tree_inventory_hash(root: Path) -> tuple[int, str]:
    rows = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return len(rows), hashlib.sha256(_canonical(rows).encode("utf-8")).hexdigest()


def _workspace_subpath(path: Path) -> str:
    resolved = path.resolve()
    workspace = Path("/workspace")
    try:
        relative = resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError(f"Formal PVC path is outside /workspace: {resolved}") from error
    return relative.as_posix()


def _pod_base(name: str, *, role: str, image: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": "ecepxie",
            "labels": {
                "app": "decision-admissibility-wp8-tier2",
                "work-package": "wp8",
                "experiment-tier": "tier2-formal",
                "execution-kind": "devpod",
                "role": role,
            },
        },
        "spec": {
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 30,
            "containers": [
                {
                    "name": role,
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": ["/bin/bash", "-lc"],
                    "args": ["set -euo pipefail\nexec sleep infinity\n"],
                    "env": [
                        {
                            "name": "POD_NAME",
                            "valueFrom": {
                                "fieldRef": {"fieldPath": "metadata.name"}
                            },
                        },
                        {
                            "name": "POD_NAMESPACE",
                            "valueFrom": {
                                "fieldRef": {"fieldPath": "metadata.namespace"}
                            },
                        },
                        {
                            "name": "POD_UID",
                            "valueFrom": {
                                "fieldRef": {"fieldPath": "metadata.uid"}
                            },
                        },
                    ],
                }
            ],
            "volumes": [
                {
                    "name": "workspace",
                    "persistentVolumeClaim": {"claimName": "haoming-storage"},
                },
                {"name": "work", "emptyDir": {}},
            ],
        },
    }


def _pvc_mount(
    mount_path: str,
    source: Path,
    *,
    read_only: bool,
) -> dict[str, Any]:
    return {
        "name": "workspace",
        "mountPath": mount_path,
        "subPath": _workspace_subpath(source),
        "readOnly": read_only,
    }


def _render_training_pod(
    template: Mapping[str, Any],
    *,
    source_root: Path,
    data_root: Path,
    bundle_root: Path,
    contract_root: Path,
    output_root: Path,
    content_hash: str,
) -> dict[str, Any]:
    pod = _pod_base(
        str(template["expected_training_pod_name"]),
        role="formal-training",
        image=str(template["container_image_digest"]),
    )
    pod["spec"]["activeDeadlineSeconds"] = 28800
    pod["spec"]["affinity"] = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/hostname",
                                "operator": "NotIn",
                                "values": list(EXCLUDED_GPU_NODES),
                            }
                        ]
                    }
                ]
            }
        }
    }
    container = pod["spec"]["containers"][0]
    container["env"].append(
        {"name": "WP8_FORMAL_STAGING_CONTENT_HASH", "value": content_hash}
    )
    container["resources"] = {
        "requests": {
            "cpu": "8",
            "memory": "32Gi",
            TRAINING_GPU_RESOURCE_KEY: "1",
        },
        "limits": {
            "cpu": "8",
            "memory": "32Gi",
            TRAINING_GPU_RESOURCE_KEY: "1",
        },
    }
    container["volumeMounts"] = [
        _pvc_mount("/opt/nautilus", source_root, read_only=True),
        _pvc_mount("/task", data_root / "train_view", read_only=True),
        _pvc_mount("/memory", bundle_root, read_only=True),
        _pvc_mount("/contract", contract_root, read_only=True),
        _pvc_mount("/output", output_root, read_only=False),
        _pvc_mount(
            "/secrets/mlevolve.env",
            Path("/workspace/nautilus/mlevolve/.env"),
            read_only=True,
        ),
        {"name": "work", "mountPath": "/work"},
        {"name": "cache", "mountPath": "/cache"},
    ]
    pod["spec"]["volumes"].append({"name": "cache", "emptyDir": {}})
    return pod


def _render_evaluator_pod(
    template: Mapping[str, Any],
    *,
    source_root: Path,
    data_root: Path,
    contract_root: Path,
    output_root: Path,
    content_hash: str,
) -> dict[str, Any]:
    pod = _pod_base(
        str(template["expected_evaluator_pod_name"]),
        role="formal-evaluator",
        image=str(template["container_image_digest"]),
    )
    pod["spec"]["activeDeadlineSeconds"] = 7200
    container = pod["spec"]["containers"][0]
    container["env"].append(
        {"name": "WP8_FORMAL_STAGING_CONTENT_HASH", "value": content_hash}
    )
    container["resources"] = {
        "requests": {"cpu": "4", "memory": "8Gi"},
        "limits": {"cpu": "4", "memory": "8Gi"},
    }
    container["volumeMounts"] = [
        _pvc_mount("/opt/nautilus", source_root, read_only=True),
        _pvc_mount("/fixed/train_view", data_root / "train_view", read_only=True),
        _pvc_mount(
            "/fixed/evaluator_view", data_root / "evaluator_view", read_only=True
        ),
        _pvc_mount("/contract", contract_root, read_only=True),
        _pvc_mount("/output", output_root, read_only=False),
        {"name": "work", "mountPath": "/work"},
    ]
    return pod


def _render_controller_pod(
    *,
    source_root: Path,
    staging_root: Path,
    output_root: Path,
    image_digest: str,
    content_hash: str,
) -> dict[str, Any]:
    pod = _pod_base(
        FORMAL_CONTROLLER_NAME,
        role="formal-controller",
        image=image_digest,
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


def _task_artifact_roots(artifact_root: Path) -> dict[str, dict[str, Path]]:
    rows: dict[str, dict[str, Path]] = {}
    for task_id in TASK_ALIASES:
        data_candidates = sorted(
            path for path in (artifact_root / "data" / task_id).iterdir() if path.is_dir()
        )
        if len(data_candidates) != 1:
            raise ValueError(f"Formal holdout cardinality mismatch: {task_id}")
        bundle_root = (
            artifact_root / "memory_bundles" / task_id / "formal-child-r3"
        )
        if not bundle_root.is_dir():
            raise ValueError(f"Formal child Bundle is missing: {task_id}")
        rows[task_id] = {"data": data_candidates[0], "bundle": bundle_root}
    return rows


def build_staging(
    *,
    source_root: Path,
    artifact_root: Path,
    staging_root: Path,
    output_root: Path,
    preregistration_r1: Path,
    preregistration_r2: Path,
    preregistration_r3: Path,
    preregistration_r4: Path,
    preregistration_r5: Path,
    superseded_evidence: tuple[Path, ...] = (),
    failed_formal_evidence: tuple[Path, ...] = (),
    image_digest: str = IMAGE_DIGEST,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    artifact_root = artifact_root.resolve()
    staging_root = staging_root.resolve()
    output_root = output_root.resolve()
    for target in (staging_root, output_root):
        if target.exists():
            raise FileExistsError(target)
    staging_tmp = staging_root.with_name(
        f".{staging_root.name}.staging-{uuid.uuid4().hex}"
    )
    output_tmp = output_root.with_name(f".{output_root.name}.staging-{uuid.uuid4().hex}")
    staging_tmp.mkdir(parents=True)
    output_tmp.mkdir(parents=True)
    try:
        source_manifest_path = source_root / "WP8_TIER2_SOURCE_MANIFEST.json"
        source_manifest = read_object(source_manifest_path)
        if source_manifest.get("schema") != SOURCE_SCHEMA:
            raise ValueError("Formal source snapshot schema mismatch")
        if source_manifest.get("source_sha256") != payload_hash(
            source_manifest, "source_sha256"
        ):
            raise ValueError("Formal source snapshot hash mismatch")
        prereg_paths = [
            preregistration_r1.resolve(),
            preregistration_r2.resolve(),
            preregistration_r3.resolve(),
            preregistration_r4.resolve(),
            preregistration_r5.resolve(),
        ]
        prereg = [read_object(path) for path in prereg_paths]
        design = prereg[0]
        if design.get("schema") != "decision_admissibility_wp8_tier2_formal_preregistration_v1":
            raise ValueError("Formal r1 preregistration schema mismatch")
        postfailure_report = verify_postfailure_amendment(
            preregistration_r5, repo_root=ROOT
        )
        if postfailure_report.get("verified") is not True:
            raise ValueError(
                "Formal r5 post-failure amendment verification failed: "
                + ",".join(postfailure_report.get("errors") or [])
            )
        tasks = {str(row["task_id"]): row for row in design["tasks"]}
        shared = design["shared_candidate_contract"]
        artifacts = _task_artifact_roots(artifact_root)

        copied_prereg: dict[str, dict[str, str]] = {}
        prereg_dir = staging_tmp / "preregistration"
        prereg_dir.mkdir()
        for path in prereg_paths:
            target = prereg_dir / path.name
            shutil.copy2(path, target)
            copied_prereg[path.name] = {
                "path": str(staging_root / "preregistration" / path.name),
                "sha256": sha256_file(target),
            }
        superseded_records: list[dict[str, Any]] = []
        seen_superseded_paths: set[Path] = set()
        for evidence_path in superseded_evidence:
            evidence_path = evidence_path.resolve()
            if evidence_path in seen_superseded_paths:
                raise ValueError(f"Duplicate superseded evidence: {evidence_path}")
            seen_superseded_paths.add(evidence_path)
            payload = read_object(evidence_path)
            if payload.get("terminal_metric_observed") is not False:
                raise ValueError("Superseded formal evidence observed a terminal metric")
            internal_hash = str(
                payload.get("report_hash")
                or payload.get("attestation_hash")
                or ""
            )
            hash_field = (
                "report_hash" if payload.get("report_hash") else "attestation_hash"
            )
            if not internal_hash or internal_hash != payload_hash(payload, hash_field):
                raise ValueError(f"Superseded evidence hash mismatch: {evidence_path}")
            superseded_records.append(
                {
                    "path": str(evidence_path),
                    "schema": payload.get("schema"),
                    "file_sha256": sha256_file(evidence_path),
                    "internal_hash": internal_hash,
                    "terminal_metric_observed": False,
                    "reuse_for_formal_execution": False,
                }
            )

        failed_records: list[dict[str, Any]] = []
        failed_dir = staging_tmp / "failed_formal_evidence"
        failed_dir.mkdir()
        declared_failure = prereg[-1].get("failed_formal_attempt") or {}
        for evidence_path in failed_formal_evidence:
            evidence_path = evidence_path.resolve()
            payload = read_object(evidence_path)
            if payload.get("schema") != (
                "decision_admissibility_wp8_tier2_formal_failure_diagnostic_v1"
            ):
                raise ValueError("Unknown failed formal evidence schema")
            if payload.get("diagnostic_hash") != payload_hash(
                payload, "diagnostic_hash"
            ):
                raise ValueError("Failed formal evidence hash mismatch")
            if (
                payload.get("terminal_metric_observed") is not True
                or payload.get("pre_metric_abort") is not False
                or payload.get("score_values_inspected_during_recovery")
                is not False
                or payload.get("r8_reuse_permitted") is not False
            ):
                raise ValueError("Failed formal evidence disposition mismatch")
            if (
                sha256_file(evidence_path)
                != declared_failure.get("diagnostic_file_sha256")
                or payload.get("diagnostic_hash")
                != declared_failure.get("diagnostic_hash")
                or payload.get("block_id") != declared_failure.get("block_id")
            ):
                raise ValueError("r5 failed formal evidence binding mismatch")
            target = failed_dir / evidence_path.name
            shutil.copy2(evidence_path, target)
            failed_records.append(
                {
                    "path": str(
                        staging_root
                        / "failed_formal_evidence"
                        / evidence_path.name
                    ),
                    "schema": payload["schema"],
                    "file_sha256": sha256_file(target),
                    "diagnostic_hash": payload["diagnostic_hash"],
                    "block_id": payload["block_id"],
                    "output_tree_sha256": payload["preserved_output"][
                        "tree_sha256"
                    ],
                    "terminal_metric_observed": True,
                    "score_values_inspected": False,
                    "reuse_for_formal_execution": False,
                    "included_in_effect_analysis": False,
                }
            )
        if len(failed_records) != 1:
            raise ValueError("Exactly one r8 failed formal diagnostic is required")

        reports_dir = staging_tmp / "reports"
        reports_dir.mkdir(exist_ok=True)
        postfailure_report_path = (
            reports_dir / "preregistration-r5-postfailure-verification.json"
        )
        _write_exclusive(postfailure_report_path, postfailure_report)

        task_records: dict[str, dict[str, Any]] = {}
        task_contracts: dict[str, dict[str, Any]] = {}
        for task_id, spec in tasks.items():
            data_root = artifacts[task_id]["data"]
            bundle_root = artifacts[task_id]["bundle"]
            split = read_object(data_root / "split_manifest.json")
            train_path = data_root / "train_view" / "fixed_holdout_manifest.json"
            evaluator_path = (
                data_root / "evaluator_view" / "fixed_holdout_manifest.json"
            )
            train = read_object(train_path)
            evaluator = read_object(evaluator_path)
            current_path = bundle_root / "CURRENT.json"
            current = read_object(current_path)
            bundle_path = bundle_root / str(current["bundle_path"])
            bundle = ImmutableBaseBundle.load(bundle_path, verify_artifacts=True)
            publication_path = bundle_root / "reports" / "publication_report.json"
            publication = read_object(publication_path)
            for key in ("task_id", "split_id", "protocol_ref", "metric", "maximize"):
                expected = task_id if key == "task_id" else split.get(key)
                if train.get(key) != expected or evaluator.get(key) != expected:
                    raise ValueError(f"Formal holdout binding mismatch: {task_id}:{key}")
            candidate_spec = spec["candidate_contract"]
            candidate = build_candidate_execution_contract(
                contract_id=candidate_spec["contract_id"],
                max_execution_seconds=shared["max_execution_seconds"],
                max_epochs=shared["max_epochs"],
                max_cv_folds=shared["max_cv_folds"],
                max_trainable_models=shared["max_trainable_models"],
                allowed_import_roots=candidate_spec["allowed_import_roots"],
                allow_remote_assets=shared["allow_remote_assets"],
                allow_unverified_local_assets=shared[
                    "allow_unverified_local_assets"
                ],
                allow_dataset_wide_per_sample_precompute=shared[
                    "allow_dataset_wide_per_sample_precompute"
                ],
                allow_source_score_inheritance=shared[
                    "allow_source_score_inheritance"
                ],
            )
            if candidate["contract_hash"] != candidate_spec["contract_hash"]:
                raise ValueError(f"Candidate contract drift: {task_id}")
            task_contracts[task_id] = candidate
            prior_holdout_path = (
                artifact_root
                / "reports"
                / f"holdout-{TASK_ALIASES[task_id]}-verification.json"
            )
            prior_holdout = read_object(prior_holdout_path)
            if (
                prior_holdout.get("valid") is not True
                or prior_holdout.get("report_hash")
                != payload_hash(prior_holdout, "report_hash")
                or Path(str(prior_holdout.get("root") or "")).resolve()
                != data_root.resolve()
            ):
                raise ValueError(f"Prior independent holdout report is invalid: {task_id}")
            data_count = int(prior_holdout["file_count"])
            data_inventory = str(prior_holdout["file_inventory_sha256"])
            task_records[task_id] = {
                "task_id": task_id,
                "target_task_family": publication["target_task_family"],
                "target_domain": publication["target_domain"],
                "data_root": str(data_root),
                "data_file_count": data_count,
                "data_inventory_sha256": data_inventory,
                "prior_holdout_verification_sha256": sha256_file(
                    prior_holdout_path
                ),
                "prior_holdout_verification_hash": prior_holdout[
                    "report_hash"
                ],
                "split_id": split["split_id"],
                "split_manifest_hash": split["manifest_hash"],
                "split_manifest_file_sha256": sha256_file(
                    data_root / "split_manifest.json"
                ),
                "protocol_ref": split["protocol_ref"],
                "metric": split["metric"],
                "maximize": split["maximize"],
                "train_manifest_sha256": sha256_file(train_path),
                "evaluator_manifest_sha256": sha256_file(evaluator_path),
                "bundle_root": str(bundle_root),
                "bundle_id": bundle.bundle_id,
                "bundle_manifest_sha256": bundle.manifest_sha256,
                "bundle_manifest_file_sha256": sha256_file(
                    bundle.path / "manifest.json"
                ),
                "bundle_current_file_sha256": sha256_file(current_path),
                "bundle_current_pointer_sha256": current["pointer_sha256"],
                "formal_clause_id": publication["formal_clause_id"],
                "formal_claim_id": publication["formal_claim_id"],
                "publication_report_sha256": sha256_file(publication_path),
                "publication_report_hash": publication["report_hash"],
                "candidate_execution_contract": candidate,
            }

        blocks_by_id: dict[str, dict[str, Any]] = {}
        templates: dict[str, dict[str, Any]] = {}
        for declared in design["condition_order_design"]["blocks"]:
            task_id = str(declared["task_id"])
            seed = int(declared["agent_seed"])
            alias = TASK_ALIASES[task_id]
            block_id = (
                f"wp8-tier2-formal-{alias}-seed-{seed}-"
                f"{FORMAL_EXECUTION_REVISION}"
            )
            block_output_staging = output_tmp / "blocks" / block_id
            block_output_staging.mkdir(parents=True)
            block_output = output_root / "blocks" / block_id
            training_pod = (
                f"da-wp8-f-{alias}-s{seed}-gpu-"
                f"{FORMAL_EXECUTION_REVISION}"
            )
            evaluator_pod = (
                f"da-wp8-f-{alias}-s{seed}-cpu-"
                f"{FORMAL_EXECUTION_REVISION}"
            )
            record = task_records[task_id]
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
                "condition_order": list(declared["order"]),
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
                "candidate_execution_contract": task_contracts[task_id],
                "candidate_execution_contract_hash": task_contracts[task_id][
                    "contract_hash"
                ],
                "source_snapshot_sha256": source_manifest["source_sha256"],
                "source_manifest_file_sha256": sha256_file(source_manifest_path),
                "train_manifest_sha256": record["train_manifest_sha256"],
                "evaluator_manifest_sha256": record[
                    "evaluator_manifest_sha256"
                ],
                "bundle_id": record["bundle_id"],
                "bundle_manifest_sha256": record["bundle_manifest_sha256"],
                "bundle_manifest_file_sha256": record[
                    "bundle_manifest_file_sha256"
                ],
                "bundle_current_file_sha256": record[
                    "bundle_current_file_sha256"
                ],
                "formal_clause_id": record["formal_clause_id"],
                "formal_claim_id": record["formal_claim_id"],
                "container_image_digest": image_digest,
                "expected_training_pod_name": training_pod,
                "expected_training_pod_namespace": "ecepxie",
                "expected_evaluator_pod_name": evaluator_pod,
                "expected_evaluator_pod_namespace": "ecepxie",
                "output_root_id": f"{output_root.name}/blocks/{block_id}",
                "template_hash": "",
            }
            template["template_hash"] = payload_hash(template, "template_hash")
            contract_dir = staging_tmp / "blocks" / block_id
            contract_dir.mkdir(parents=True)
            _write_exclusive(contract_dir / "BLOCK_TEMPLATE.json", template)
            templates[block_id] = template
            blocks_by_id[block_id] = {
                "block_id": block_id,
                "task_id": task_id,
                "agent_seed": seed,
                "condition_order": list(declared["order"]),
                "block_template_hash": template["template_hash"],
                "block_template_sha256": sha256_file(
                    contract_dir / "BLOCK_TEMPLATE.json"
                ),
                "contract_root": str(staging_root / "blocks" / block_id),
                "output_root": str(block_output),
                "output_root_initially_empty": True,
                "training_pod_name": training_pod,
                "evaluator_pod_name": evaluator_pod,
            }

        runtime_files = {
            relative: sha256_file(source_root / relative)
            for relative in (
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
        }
        control_files = {
            relative: sha256_file(ROOT / relative)
            for relative in (
                "deploy/stage_decision_admissibility_wp8_tier2_formal.sh",
                "deploy/run_decision_admissibility_wp8_tier2_formal_block.sh",
                "deploy/run_decision_admissibility_wp8_tier2_formal_staging_pipeline.sh",
                "paper-skills/memory_bundle/build_tier2_formal_staging.py",
                "paper-skills/memory_bundle/verify_tier2_formal_staging.py",
                "paper-skills/memory_bundle/verify_tier2_formal_postfailure_amendment.py",
            )
        }
        content: dict[str, Any] = {
            "schema": STAGING_CONTENT_SCHEMA,
            "status": "content_frozen_pending_independent_stop_gate",
            "design_preregistration_id": design["preregistration_id"],
            "effective_preregistration_id": prereg[-1]["preregistration_id"],
            "preregistration_files": copied_prereg,
            "postfailure_amendment_verification": {
                "path": str(
                    staging_root
                    / "reports"
                    / postfailure_report_path.name
                ),
                "sha256": sha256_file(postfailure_report_path),
                "verification_hash": postfailure_report[
                    "verification_hash"
                ],
            },
            "superseded_preterminal_attempts": superseded_records,
            "failed_formal_attempts": failed_records,
            "formal_execution_revision": FORMAL_EXECUTION_REVISION,
            "training_gpu_resource_key": TRAINING_GPU_RESOURCE_KEY,
            "excluded_gpu_nodes": list(EXCLUDED_GPU_NODES),
            "source_root": str(source_root),
            "source_snapshot_sha256": source_manifest["source_sha256"],
            "source_manifest_file_sha256": sha256_file(source_manifest_path),
            "base_commit": source_manifest["base_commit"],
            "container_image_digest": image_digest,
            "artifact_root": str(artifact_root),
            "task_records": task_records,
            "output_root": str(output_root),
            "output_roots_initially_empty": True,
            "runtime_source_files": runtime_files,
            "control_source_files": control_files,
            "blocks_by_id": blocks_by_id,
            "online_condition_count": 45,
            "oracle_count": 9,
            "formal_training_started": False,
            "terminal_metric_observed": False,
            "manifest_hash": "",
        }
        content["manifest_hash"] = payload_hash(content, "manifest_hash")
        _write_exclusive(staging_tmp / "STAGING_CONTENT_MANIFEST.json", content)
        for block_id in blocks_by_id:
            shutil.copy2(
                staging_tmp / "STAGING_CONTENT_MANIFEST.json",
                staging_tmp / "blocks" / block_id / "STAGING_CONTENT_MANIFEST.json",
            )

        pod_hashes: dict[str, dict[str, str]] = {}
        pod_dir = staging_tmp / "pods"
        pod_dir.mkdir()
        for block_id, block in blocks_by_id.items():
            template = templates[block_id]
            task_id = block["task_id"]
            data_root = artifacts[task_id]["data"]
            bundle_root = artifacts[task_id]["bundle"]
            contract_root = staging_root / "blocks" / block_id
            block_output = output_root / "blocks" / block_id
            documents = {
                "training": _render_training_pod(
                    template,
                    source_root=source_root,
                    data_root=data_root,
                    bundle_root=bundle_root,
                    contract_root=contract_root,
                    output_root=block_output,
                    content_hash=content["manifest_hash"],
                ),
                "evaluator": _render_evaluator_pod(
                    template,
                    source_root=source_root,
                    data_root=data_root,
                    contract_root=contract_root,
                    output_root=block_output,
                    content_hash=content["manifest_hash"],
                ),
            }
            for role, document in documents.items():
                path = pod_dir / f"{block_id}-{role}.yaml"
                path.write_text(
                    yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
                )
                path.chmod(0o444)
                pod_hashes[f"{block_id}:{role}"] = {
                    "path": str(staging_root / "pods" / path.name),
                    "sha256": sha256_file(path),
                }
        controller_path = pod_dir / FORMAL_CONTROLLER_YAML
        controller_path.write_text(
            yaml.safe_dump(
                _render_controller_pod(
                    source_root=source_root,
                    staging_root=staging_root,
                    output_root=output_root,
                    image_digest=image_digest,
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
            "block_count": len(blocks_by_id),
            "pod_yaml_count": len(pod_hashes),
            "pod_yamls": pod_hashes,
            "builder_source_sha256": sha256_file(Path(__file__).resolve()),
            "formal_training_started": False,
            "terminal_metric_observed": False,
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
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preregistration-r1", type=Path, required=True)
    parser.add_argument("--preregistration-r2", type=Path, required=True)
    parser.add_argument("--preregistration-r3", type=Path, required=True)
    parser.add_argument("--preregistration-r4", type=Path, required=True)
    parser.add_argument("--preregistration-r5", type=Path, required=True)
    parser.add_argument("--superseded-evidence", type=Path, action="append")
    parser.add_argument("--failed-formal-evidence", type=Path, action="append")
    parser.add_argument("--image-digest", default=IMAGE_DIGEST)
    args = parser.parse_args()
    report = build_staging(
        source_root=args.source_root,
        artifact_root=args.artifact_root,
        staging_root=args.staging_root,
        output_root=args.output_root,
        preregistration_r1=args.preregistration_r1,
        preregistration_r2=args.preregistration_r2,
        preregistration_r3=args.preregistration_r3,
        preregistration_r4=args.preregistration_r4,
        preregistration_r5=args.preregistration_r5,
        superseded_evidence=tuple(args.superseded_evidence or ()),
        failed_formal_evidence=tuple(args.failed_formal_evidence or ()),
        image_digest=args.image_digest,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
