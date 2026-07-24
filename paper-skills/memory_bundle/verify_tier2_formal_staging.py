#!/usr/bin/env python3
"""Independently verify the WP8 Tier-2 staging Stop Gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
MLEVOLVE = ROOT / "mlevolve"
if str(MLEVOLVE) not in os.sys.path:
    os.sys.path.insert(0, str(MLEVOLVE))

from engine.candidate_execution_contract import (  # noqa: E402
    build_candidate_execution_contract,
)
from fixed_holdout.common import sha256_file  # noqa: E402
from fixed_holdout.formal_runtime import (  # noqa: E402
    BLOCK_TEMPLATE_SCHEMA,
    STAGING_CONTENT_SCHEMA,
    payload_hash,
    read_object,
    validate_block_template,
    verify_source_snapshot,
)
from fixed_holdout.formal_verify import verify_formal_holdout  # noqa: E402
from build_tier2_formal_staging import (  # noqa: E402
    EXCLUDED_GPU_NODES,
    FORMAL_CONTROLLER_NAME,
    FORMAL_CONTROLLER_YAML,
    FORMAL_EXECUTION_REVISION,
    TRAINING_GPU_RESOURCE_KEY,
)
from verify_tier2_formal_child_bundle import (  # noqa: E402
    verify_formal_child_publication,
)
from verify_tier2_formal_claim_authority_amendment import (  # noqa: E402
    verify_claim_authority_amendment,
)
from verify_tier2_formal_preregistration import (  # noqa: E402
    verify_preregistration,
)
from verify_tier2_formal_preregistration_amendment import (  # noqa: E402
    verify_amendment,
)
from verify_tier2_formal_postfailure_amendment import (  # noqa: E402
    verify_postfailure_amendment,
)


SCHEMA = "decision_admissibility_wp8_tier2_formal_staging_stop_gate_v1"
ONLINE_SYSTEMS = {
    "no_memory",
    "flat_relevance_memory",
    "global_validity_bit",
    "authority_only",
    "full_decision_admissibility",
}
USER_POD = "jupyter-a10-d48dfd589-pqfkb"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _tree_inventory_hash(root: Path) -> tuple[int, str]:
    files = [path for path in sorted(root.rglob("*")) if path.is_file()]
    def hash_one(path: Path) -> tuple[str, str]:
        return path.relative_to(root).as_posix(), sha256_file(path)

    with ThreadPoolExecutor(max_workers=min(32, max(1, len(files)))) as pool:
        rows = dict(pool.map(hash_one, files))
    return len(rows), hashlib.sha256(_canonical(rows).encode("utf-8")).hexdigest()


def _mounts(document: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["mountPath"]): dict(row)
        for row in document["spec"]["containers"][0]["volumeMounts"]
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_pod(
    document: Mapping[str, Any],
    *,
    role: str,
    template: Mapping[str, Any],
    content_hash: str,
) -> list[str]:
    errors: list[str] = []

    def require(value: object, code: str) -> None:
        if not bool(value):
            errors.append(code)

    expected_name = template[
        "expected_training_pod_name"
        if role == "training"
        else "expected_evaluator_pod_name"
    ]
    require(document.get("kind") == "Pod", "kind_is_pod")
    require(document.get("apiVersion") == "v1", "pod_api_version")
    require((document.get("metadata") or {}).get("name") == expected_name, "pod_name")
    require((document.get("metadata") or {}).get("namespace") == "ecepxie", "namespace")
    require(expected_name != USER_POD, "user_pod_out_of_scope")
    spec = document.get("spec") or {}
    require(spec.get("restartPolicy") == "Never", "restart_policy")
    containers = spec.get("containers") or []
    require(len(containers) == 1, "single_container")
    if len(containers) != 1:
        return errors
    container = containers[0]
    require(container.get("image") == template["container_image_digest"], "image_digest")
    require("sleep infinity" in "".join(container.get("args") or []), "devpod_lifecycle")
    env = {
        str(row.get("name")): row.get("value")
        for row in container.get("env") or []
        if "value" in row
    }
    require(
        env.get("WP8_FORMAL_STAGING_CONTENT_HASH") == content_hash,
        "content_hash_env",
    )
    mounts = _mounts(document)
    require("/workspace" not in mounts, "whole_workspace_absent")
    volumes = spec.get("volumes") or []
    pvc = [row for row in volumes if "persistentVolumeClaim" in row]
    require(
        pvc
        == [
            {
                "name": "workspace",
                "persistentVolumeClaim": {"claimName": "haoming-storage"},
            }
        ],
        "single_workspace_pvc",
    )
    resources = container.get("resources") or {}
    requests = resources.get("requests") or {}
    limits = resources.get("limits") or {}
    if role == "training":
        require(
            spec.get("affinity")
            == {
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
            },
            "known_bad_gpu_nodes_excluded",
        )
        require(
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
            "training_mount_universe",
        )
        require("/fixed/evaluator_view" not in mounts, "evaluator_view_absent")
        require(
            requests.get(TRAINING_GPU_RESOURCE_KEY)
            == limits.get(TRAINING_GPU_RESOURCE_KEY)
            == "1",
            "one_gpu",
        )
        require(requests.get("cpu") == limits.get("cpu") == "8", "eight_cpu")
        for path in ("/opt/nautilus", "/task", "/memory", "/contract", "/secrets/mlevolve.env"):
            require((mounts.get(path) or {}).get("readOnly") is True, f"read_only:{path}")
    else:
        require(
            set(mounts)
            == {
                "/opt/nautilus",
                "/fixed/train_view",
                "/fixed/evaluator_view",
                "/contract",
                "/output",
                "/work",
            },
            "evaluator_mount_universe",
        )
        require("/memory" not in mounts, "memory_absent")
        require("/secrets/mlevolve.env" not in mounts, "solver_secret_absent")
        require(
            not any(str(key).startswith("nvidia.com/") for key in limits),
            "cpu_only",
        )
        require(requests.get("cpu") == limits.get("cpu") == "4", "four_cpu")
        for path in ("/opt/nautilus", "/fixed/train_view", "/fixed/evaluator_view", "/contract"):
            require((mounts.get(path) or {}).get("readOnly") is True, f"read_only:{path}")
    return errors


def verify_staging(
    staging_root: str | Path,
    *,
    repo_root: str | Path = ROOT,
    seal_on_success: bool = True,
) -> dict[str, Any]:
    staging_root = Path(staging_root).resolve()
    repo_root = Path(repo_root).resolve()
    errors: list[str] = []
    checks: dict[str, bool] = {}
    evidence: dict[str, Any] = {}

    def check(code: str, value: object) -> None:
        passed = bool(value)
        checks[code] = passed
        if not passed:
            errors.append(code)

    try:
        content_path = staging_root / "STAGING_CONTENT_MANIFEST.json"
        build_path = staging_root / "STAGING_BUILD_REPORT.json"
        content = read_object(content_path)
        build = read_object(build_path)
        check("content_schema", content.get("schema") == STAGING_CONTENT_SCHEMA)
        check(
            "content_hash",
            content.get("manifest_hash") == payload_hash(content, "manifest_hash"),
        )
        check(
            "build_hash",
            build.get("build_hash") == payload_hash(build, "build_hash"),
        )
        check(
            "build_content_binding",
            build.get("staging_content_manifest_hash") == content.get("manifest_hash")
            and build.get("staging_content_manifest_sha256")
            == sha256_file(content_path),
        )
        check("formal_not_started", content.get("formal_training_started") is False)
        check("terminal_metric_unobserved", content.get("terminal_metric_observed") is False)
        check("declared_online_count", content.get("online_condition_count") == 45)
        check("declared_oracle_count", content.get("oracle_count") == 9)
        check(
            "known_bad_gpu_nodes_declared",
            content.get("excluded_gpu_nodes") == list(EXCLUDED_GPU_NODES),
        )
        check(
            "training_gpu_resource_key",
            content.get("training_gpu_resource_key")
            == TRAINING_GPU_RESOURCE_KEY,
        )
        superseded = content.get("superseded_preterminal_attempts") or []
        check("superseded_attempts_bound", len(superseded) == 10)
        check(
            "superseded_attempt_paths_unique",
            len({str(row.get("path") or "") for row in superseded})
            == len(superseded),
        )
        superseded_payloads: list[dict[str, Any]] = []
        for index, row in enumerate(superseded):
            path = Path(row["path"])
            payload = read_object(path)
            superseded_payloads.append(payload)
            hash_field = "report_hash" if payload.get("report_hash") else "attestation_hash"
            check(
                f"superseded_file:{index}",
                sha256_file(path) == row["file_sha256"]
                and payload_hash(payload, hash_field) == row["internal_hash"]
                and payload.get("terminal_metric_observed") is False
                and row.get("reuse_for_formal_execution") is False,
            )
        superseded_classifications = {
            str(payload.get("classification") or "")
            for payload in superseded_payloads
        }
        check(
            "superseded_attempt_classifications",
            {
                "source_and_evaluator_runtime_compatibility_failure",
                "kubernetes_node_not_ready_taint_manager_eviction",
                "search_only_candidate_ordering_misrouted_through_formal_authority_rank",
                "gpu_node_nvml_driver_library_version_mismatch",
                "kubectl_exec_websocket_close_1006_during_control_overlay_transfer",
                "unexpected_training_devpod_loss",
                "kubectl_warning_stderr_contaminated_jsonpath_phase_stdout",
            }
            <= superseded_classifications,
        )
        check(
            "r2_gpu_scheduling_abort_bound",
            any(
                payload.get("block_id")
                == "wp8-tier2-formal-aerial-seed-104729-r2"
                and payload.get("classification")
                == "training_container_failed_before_ready"
                and payload.get("formal_training_started") is False
                and payload.get("terminal_metric_observed") is False
                and payload.get("candidate_submission_count") == 0
                and payload.get("evaluator_pod_created") is False
                and payload.get("reuse_for_formal_execution") is False
                for payload in superseded_payloads
            ),
        )
        failed_formal = content.get("failed_formal_attempts") or []
        check("one_postmetric_failed_formal_attempt", len(failed_formal) == 1)
        if len(failed_formal) == 1:
            failed_row = failed_formal[0]
            failed_path = Path(str(failed_row.get("path") or ""))
            failed_payload = read_object(failed_path)
            check(
                "postmetric_failure_file_bound",
                sha256_file(failed_path) == failed_row.get("file_sha256")
                and failed_payload.get("diagnostic_hash")
                == payload_hash(failed_payload, "diagnostic_hash")
                == failed_row.get("diagnostic_hash"),
            )
            check(
                "postmetric_failure_disposition",
                failed_payload.get("terminal_metric_observed") is True
                and failed_payload.get("pre_metric_abort") is False
                and failed_payload.get(
                    "score_values_inspected_during_recovery"
                )
                is False
                and failed_row.get("score_values_inspected") is False
                and failed_row.get("reuse_for_formal_execution") is False
                and failed_row.get("included_in_effect_analysis") is False,
            )
            check(
                "postmetric_failure_tree_bound",
                failed_payload.get("preserved_output", {}).get("tree_sha256")
                == failed_row.get("output_tree_sha256"),
            )
            evidence["failed_formal_diagnostic_hash"] = failed_payload.get(
                "diagnostic_hash"
            )

        source_report = verify_source_snapshot(
            content["source_root"],
            expected_source_sha256=content["source_snapshot_sha256"],
            expected_manifest_file_sha256=content["source_manifest_file_sha256"],
        )
        check("source_snapshot_verified", source_report["verified"] is True)
        check("base_commit", content.get("base_commit") == "b47dab63b7861f3ea0871094d6dd07b77e6b81a4")
        evidence["source_verification"] = source_report

        prereg_dir = staging_root / "preregistration"
        r1 = prereg_dir / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r1.json"
        r2 = prereg_dir / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r2.json"
        r3 = prereg_dir / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r3.json"
        r4 = prereg_dir / "decision_admissibility_wp8_tier2_formal_preregistration_20260722_r4.json"
        r5 = prereg_dir / "decision_admissibility_wp8_tier2_formal_preregistration_20260723_r5.json"
        for path in (r1, r2, r3, r4, r5):
            expected = content["preregistration_files"][path.name]["sha256"]
            check(f"prereg_file:{path.name}", sha256_file(path) == expected)
        r1_report = verify_preregistration(r1, repo_root=repo_root)
        r2_report = verify_amendment(r2, repo_root=repo_root)
        r3_static = verify_claim_authority_amendment(
            r3, repo_root=repo_root, require_source_bundle=False
        )
        design = read_object(r1)
        parent_source = Path(
            design["memory_bundle_contract"]["seed_memory_parent"]["path"]
        )
        r3_source = verify_claim_authority_amendment(
            r3,
            repo_root=repo_root,
            source_bundle=parent_source,
            require_source_bundle=True,
        )
        r4_source = verify_claim_authority_amendment(
            r4,
            repo_root=repo_root,
            source_bundle=parent_source,
            require_source_bundle=True,
        )
        r5_report = verify_postfailure_amendment(r5, repo_root=repo_root)
        check("r1_prereg_verified", r1_report["verified"] is True)
        check("r2_amendment_verified", r2_report["verified"] is True)
        check("r3_static_verified", r3_static["verified"] is True)
        check(
            "r3_failed_design_preserved",
            r3_source["verified"] is False
            and r3_source["errors"] == ["source_task_family_bound"],
        )
        check("r4_source_verified", r4_source["verified"] is True)
        check("r5_postfailure_verified", r5_report["verified"] is True)
        check(
            "r5_effective_preregistration",
            content.get("effective_preregistration_id")
            == "wp8-tier2-formal-3protocol-6system-r5-postfailure",
        )
        postfailure_pointer = content.get(
            "postfailure_amendment_verification"
        ) or {}
        postfailure_path = Path(str(postfailure_pointer.get("path") or ""))
        frozen_r5_report = read_object(postfailure_path)
        check(
            "r5_verification_report_bound",
            sha256_file(postfailure_path)
            == postfailure_pointer.get("sha256")
            and frozen_r5_report.get("verification_hash")
            == postfailure_pointer.get("verification_hash")
            == r5_report.get("verification_hash"),
        )
        check(
            "primary_contrast_full_vs_no_memory",
            design["analysis_plan"]["primary_online_contrast"]
            == "full_decision_admissibility minus no_memory, paired within task and agent_seed",
        )
        check(
            "no_target_answer_or_score_inheritance",
            design["memory_bundle_contract"]["task_heldout_target_history_exposure"] == 0
            and design["shared_candidate_contract"]["allow_source_score_inheritance"] is False,
        )
        evidence["preregistration"] = {
            "r1": r1_report["verification_hash"],
            "r2": r2_report["verification_hash"],
            "r3_static": r3_static["verification_hash"],
            "r3_failed_source": r3_source["verification_hash"],
            "r4": r4_source["verification_hash"],
            "r5": r5_report["verification_hash"],
        }

        task_specs = {str(row["task_id"]): row for row in design["tasks"]}
        holdout_reports: dict[str, Any] = {}
        bundle_reports: dict[str, Any] = {}
        for task_id, record in content["task_records"].items():
            data_root = Path(record["data_root"])
            count, inventory = _tree_inventory_hash(data_root)
            check(f"data_count:{task_id}", count == record["data_file_count"])
            check(
                f"data_inventory:{task_id}",
                inventory == record["data_inventory_sha256"],
            )
            holdout = verify_formal_holdout(data_root, verify_source_artifacts=True)
            check(f"holdout_verified:{task_id}", holdout["valid"] is True)
            check(
                f"holdout_train_hash:{task_id}",
                holdout.get("train_manifest_sha256")
                == record["train_manifest_sha256"],
            )
            check(
                f"holdout_evaluator_hash:{task_id}",
                holdout.get("evaluator_manifest_sha256")
                == record["evaluator_manifest_sha256"],
            )
            holdout_reports[task_id] = holdout

            publication_root = Path(record["bundle_root"])
            publication = read_object(
                publication_root / "reports" / "publication_report.json"
            )
            child = verify_formal_child_publication(
                publication_root,
                expected_parent_bundle_id=publication["parent_bundle_id"],
                expected_parent_manifest_sha256=publication[
                    "parent_manifest_sha256"
                ],
                target_task_id=publication["target_task_id"],
                target_task_family=publication["target_task_family"],
                target_domain=publication["target_domain"],
                split_mode=publication["split_mode"],
                source_clause_id=publication["source_clause_id"],
                source_run_id=publication["source_run_id"],
                source_node_id=publication["source_node_id"],
                publication_class=publication["publication_class"],
                protocol_ref=publication["formal_protocol_ref"],
                expected_parent_validation_mode=publication[
                    "parent_validation_disposition"
                ]["mode"],
                allowed_protocol_issue_codes=publication[
                    "allowed_protocol_issue_codes"
                ],
                agent_seeds=design["agent_seeds"],
            )
            check(f"child_bundle_verified:{task_id}", child["verified"] is True)
            check(
                f"child_bundle_identity:{task_id}",
                publication["bundle_id"] == record["bundle_id"]
                and publication["bundle_manifest_sha256"]
                == record["bundle_manifest_sha256"],
            )
            bundle_reports[task_id] = child

            shared = design["shared_candidate_contract"]
            declared = task_specs[task_id]["candidate_contract"]
            rebuilt = build_candidate_execution_contract(
                contract_id=declared["contract_id"],
                max_execution_seconds=shared["max_execution_seconds"],
                max_epochs=shared["max_epochs"],
                max_cv_folds=shared["max_cv_folds"],
                max_trainable_models=shared["max_trainable_models"],
                allowed_import_roots=declared["allowed_import_roots"],
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
            check(
                f"candidate_contract:{task_id}",
                rebuilt == record["candidate_execution_contract"],
            )
        evidence["holdout_verification_hashes"] = {
            task: report["report_hash"] for task, report in holdout_reports.items()
        }
        evidence["bundle_verification_hashes"] = {
            task: report["verification_hash"] for task, report in bundle_reports.items()
        }

        declared_blocks = {
            (str(row["task_id"]), int(row["agent_seed"])): list(row["order"])
            for row in design["condition_order_design"]["blocks"]
        }
        blocks = content.get("blocks_by_id") or {}
        check("nine_blocks", len(blocks) == 9)
        check(
            "formal_execution_revision",
            content.get("formal_execution_revision")
            == FORMAL_EXECUTION_REVISION,
        )
        positions = {system: [] for system in ONLINE_SYSTEMS}
        pod_hashes: dict[str, str] = {}
        for block_id, block in blocks.items():
            check(
                f"block_revision:{block_id}",
                block_id.endswith(f"-{FORMAL_EXECUTION_REVISION}")
                and str(block.get("training_pod_name") or "").endswith(
                    f"-gpu-{FORMAL_EXECUTION_REVISION}"
                )
                and str(block.get("evaluator_pod_name") or "").endswith(
                    f"-cpu-{FORMAL_EXECUTION_REVISION}"
                ),
            )
            contract_root = Path(block["contract_root"])
            template_path = contract_root / "BLOCK_TEMPLATE.json"
            copied_content = contract_root / "STAGING_CONTENT_MANIFEST.json"
            template = validate_block_template(read_object(template_path))
            check(f"template_schema:{block_id}", template["schema"] == BLOCK_TEMPLATE_SCHEMA)
            check(
                f"template_hash:{block_id}",
                template["template_hash"] == block["block_template_hash"]
                and sha256_file(template_path) == block["block_template_sha256"],
            )
            check(
                f"template_order:{block_id}",
                template["condition_order"]
                == declared_blocks[(template["task_id"], int(template["agent_seed"]))],
            )
            check(
                f"condition_universe:{block_id}",
                set(template["condition_order"]) == ONLINE_SYSTEMS,
            )
            for index, system in enumerate(template["condition_order"]):
                positions[system].append(index)
            check(
                f"copied_content:{block_id}",
                sha256_file(copied_content) == sha256_file(content_path),
            )
            output = Path(block["output_root"])
            check(f"output_exists:{block_id}", output.is_dir())
            check(f"output_empty:{block_id}", not any(output.iterdir()))
            for role in ("training", "evaluator"):
                yaml_path = staging_root / "pods" / f"{block_id}-{role}.yaml"
                document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                pod_errors = _verify_pod(
                    document,
                    role=role,
                    template=template,
                    content_hash=content["manifest_hash"],
                )
                check(f"pod_contract:{block_id}:{role}", not pod_errors)
                if pod_errors:
                    evidence[f"pod_errors:{block_id}:{role}"] = pod_errors
                expected = build["pod_yamls"][f"{block_id}:{role}"]
                check(
                    f"pod_hash:{block_id}:{role}",
                    sha256_file(yaml_path) == expected["sha256"],
                )
                pod_hashes[f"{block_id}:{role}"] = sha256_file(yaml_path)
        check(
            "counterbalanced_positions",
            all(
                max(values.count(index) for index in range(5))
                - min(values.count(index) for index in range(5))
                <= 1
                for values in positions.values()
            ),
        )
        evidence["pod_yaml_hashes"] = pod_hashes

        controller_path = staging_root / "pods" / FORMAL_CONTROLLER_YAML
        controller = yaml.safe_load(controller_path.read_text(encoding="utf-8"))
        controller_mounts = _mounts(controller)
        controller_container = controller["spec"]["containers"][0]
        controller_resources = controller_container["resources"]
        check("controller_is_pod", controller.get("kind") == "Pod")
        check(
            "controller_identity",
            controller["metadata"]["name"] == FORMAL_CONTROLLER_NAME
            and controller["metadata"]["name"] != USER_POD,
        )
        check(
            "controller_mount_universe",
            set(controller_mounts)
            == {"/opt/nautilus", "/formal/staging", "/formal/outputs", "/work"},
        )
        check("controller_no_workspace", "/workspace" not in controller_mounts)
        check("controller_no_secret", "/secrets/mlevolve.env" not in controller_mounts)
        check("controller_no_memory", "/memory" not in controller_mounts)
        check(
            "controller_source_read_only",
            controller_mounts["/opt/nautilus"].get("readOnly") is True,
        )
        check(
            "controller_cpu_only",
            controller_resources["requests"]
            == controller_resources["limits"]
            == {"cpu": "1", "memory": "2Gi"},
        )
        check(
            "controller_hash",
            sha256_file(controller_path)
            == build["pod_yamls"]["formal-controller"]["sha256"],
        )
        pod_hashes["formal-controller"] = sha256_file(controller_path)

        runtime_hashes = {
            relative: sha256_file(Path(content["source_root"]) / relative)
            for relative in content["runtime_source_files"]
        }
        check(
            "runtime_sources_bound",
            runtime_hashes == content["runtime_source_files"],
        )
        control_hashes = {
            relative: sha256_file(repo_root / relative)
            for relative in content["control_source_files"]
        }
        check(
            "control_sources_bound",
            control_hashes == content["control_source_files"],
        )
        check(
            "devpod_only_no_job_yaml",
            all(
                yaml.safe_load(path.read_text()).get("kind") == "Pod"
                for path in (staging_root / "pods").glob("*.yaml")
            ),
        )
    except Exception as error:
        errors.append(f"exception:{type(error).__name__}:{error}")

    verified = not errors
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed" if verified else "failed",
        "staging_root": str(staging_root),
        "staging_content_manifest_hash": (
            content.get("manifest_hash", "") if "content" in locals() else ""
        ),
        "staging_content_manifest_sha256": (
            sha256_file(content_path) if "content_path" in locals() and content_path.is_file() else ""
        ),
        "check_count": len(checks),
        "passed_check_count": sum(checks.values()),
        "checks": dict(sorted(checks.items())),
        "errors": sorted(set(errors)),
        "evidence": evidence,
        "formal_training_authorized": verified,
        "terminal_metric_observed": False,
        "verifier_source_sha256": sha256_file(Path(__file__).resolve()),
        "gate_hash": "",
    }
    report["gate_hash"] = payload_hash(report, "gate_hash")
    if verified and seal_on_success:
        for path in sorted(staging_root.rglob("*"), reverse=True):
            path.chmod(path.stat().st_mode & ~0o222)
        staging_root.chmod(staging_root.stat().st_mode & ~0o222)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-seal", action="store_true")
    args = parser.parse_args()
    report = verify_staging(
        args.staging_root,
        repo_root=args.repo_root,
        seal_on_success=not args.no_seal,
    )
    _write_exclusive(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["formal_training_authorized"] else 1)


if __name__ == "__main__":
    main()
