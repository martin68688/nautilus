from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from authority.adapters.mlevolve.runtime import MLEvolveAuthorityAdapter
from authority.memory_snapshot import MemorySnapshotLoader
from engine.candidate_execution_contract import (
    audit_candidate_code,
    build_candidate_execution_contract,
)
from engine.executor import Interpreter
from fixed_holdout.common import sha256_file, write_json
from fixed_holdout.formal_block_evaluate import (
    DELETION_ATTESTATION_SCHEMA,
    evaluate_formal_block,
)
from fixed_holdout.formal_block_training import (
    CONDITIONS,
    CONTRACT_SCHEMA,
    POD_IDENTITY_SCHEMA,
    finalize_training_block,
)
from fixed_holdout.formal_host_receipts import (
    EVALUATOR_FAILURE_SCHEMA,
    INFRASTRUCTURE_ABORT_SCHEMA,
    PRECONTRACT_ABORT_SCHEMA,
    PRELAUNCH_ABORT_SCHEMA,
    write_evaluator_failure,
    write_training_infrastructure_abort,
    write_training_precontract_abort,
    write_training_prelaunch_abort,
)
from fixed_holdout.formal_runtime import (
    CONDITION_RECEIPT_SCHEMA,
    EVALUATOR_ISOLATION_SCHEMA,
    TRAINING_ISOLATION_SCHEMA,
)
from fixed_holdout.handoff import write_evaluation_request
from tests.authority.test_mlevolve_adapter import fake_agent
from tests.test_fixed_holdout import _prepared
from tests.test_memory_snapshot_overlay import build_tiny_bundle, write_current


def _payload_hash(payload: dict, field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _formal_fixture(
    tmp_path: Path,
    *,
    invalid_submission_condition: str | None = None,
    enforced_protocol: bool = False,
    blocked_protocol_condition: str | None = None,
) -> dict:
    if enforced_protocol:
        from fixed_holdout.formal_prepare import build_aerial_holdout
        from tests.test_tier2_formal_holdout_builders import _aerial_source

        split_root = build_aerial_holdout(
            _aerial_source(tmp_path), tmp_path / "formal-aerial-holdout"
        )
        target_domain = "image"
        target_task_family = "image_classification"
    else:
        split_root = _prepared(tmp_path)
        target_domain = "nlp"
        target_task_family = "text_classification"
    train_manifest_path = split_root / "train_view" / "fixed_holdout_manifest.json"
    evaluator_manifest_path = (
        split_root / "evaluator_view" / "fixed_holdout_manifest.json"
    )
    probe_agent = fake_agent(tmp_path / "protocol-probe")
    if enforced_protocol:
        probe_agent.cfg.evaluation_authority.active_protocol_id = (
            "random-classification"
        )
        probe_agent.cfg.evaluation_authority.active_protocol_version = "1"
    probe = MLEvolveAuthorityAdapter(probe_agent)
    protocol_key = probe.active_protocol.key()
    for manifest_path in (train_manifest_path, evaluator_manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["protocol_ref"] = protocol_key
        if enforced_protocol:
            manifest["metric"] = "macro_f1"
            manifest["maximize"] = True
        write_json(manifest_path, manifest)
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    memory_root = tmp_path / "memory"
    bundle, bundle_manifest = build_tiny_bundle(memory_root)
    clause_path = bundle / "sop" / "clauses.jsonl"
    clauses = [json.loads(line) for line in clause_path.read_text().splitlines()]
    clauses[0].update(
        {
            "claim_types": ["method_hypothesis"],
            "protocol_scope": [protocol_key],
            "source_domains": [target_domain],
            "transfer_scope": "same_domain",
            "permitted_operations": ["generate_candidate"],
            "contract_spec": {"source_score_inheritance": False},
        }
    )
    clause_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in clauses),
        encoding="utf-8",
    )
    bundle_manifest["artifact_hashes"]["sop/clauses.jsonl"] = sha256_file(clause_path)
    bundle_manifest["manifest_sha256"] = _payload_hash(
        bundle_manifest, "manifest_sha256"
    )
    write_json(bundle / "manifest.json", bundle_manifest)
    write_current(memory_root, bundle, bundle_manifest)
    candidate_contract = build_candidate_execution_contract(
        contract_id="synthetic-formal-v1",
        max_execution_seconds=60,
        max_epochs=1,
        max_cv_folds=1,
        max_trainable_models=1,
        allowed_import_roots=(
            ["numpy", "sklearn"] if enforced_protocol else ["pandas"]
        ),
        allow_remote_assets=False,
        allow_unverified_local_assets=False,
        allow_dataset_wide_per_sample_precompute=True,
        allow_source_score_inheritance=False,
    )
    output_root = tmp_path / "formal-output"
    output_root.mkdir()
    order = list(CONDITIONS)
    runtime_code = "print('synthetic formal candidate')\n"
    runtime_observation: dict = {}
    blocked_runtime_code = "print('blocked protocol candidate')\n"
    blocked_runtime_observation: dict = {}
    runtime_exec_time = 1.0
    if enforced_protocol:
        runtime_code = """
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
X = np.arange(80, dtype=float).reshape(40, 2)
y = np.array([0, 1] * 20)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)
model = LogisticRegression().fit(X_train, y_train)
pred = model.predict_proba(X_valid)[:, 1]
print(roc_auc_score(y_valid, pred))
"""
        runtime_root = tmp_path / "formal-runtime-observer"
        runtime_root.mkdir()
        runtime_cfg = SimpleNamespace(
            start_cpu_id=0,
            cpu_number=1,
            agent=SimpleNamespace(
                search=SimpleNamespace(parallel_search_num=1, num_gpus=1)
            ),
            evaluation_authority=SimpleNamespace(
                mode="enforce", runtime_protocol_observer_enabled=True
            ),
        )
        runtime_result = Interpreter(runtime_root, timeout=30, cfg=runtime_cfg).run(
            runtime_code, "formal-selected"
        )
        assert runtime_result.exc_type is None
        runtime_observation = dict(runtime_result.protocol_observation or {})
        runtime_exec_time = runtime_result.exec_time
        blocked_result = Interpreter(runtime_root, timeout=30, cfg=runtime_cfg).run(
            blocked_runtime_code, "formal-blocked-selected"
        )
        assert blocked_result.exc_type is None
        blocked_runtime_observation = dict(blocked_result.protocol_observation or {})
        assert blocked_runtime_observation.get("status") == "blocked"
    for position, condition in enumerate(order):
        condition_root = output_root / "conditions" / condition
        run_dir = condition_root / "runs" / f"run-{position}"
        workspace_dir = condition_root / "workspace" / f"workspace-{position}"
        submission_dir = workspace_dir / "submission"
        working_dir = workspace_dir / "working"
        submission_dir.mkdir(parents=True)
        working_dir.mkdir(parents=True)

        agent = fake_agent(run_dir, mode="enforce")
        if enforced_protocol:
            agent.cfg.evaluation_authority.active_protocol_id = "random-classification"
            agent.cfg.evaluation_authority.active_protocol_version = "1"
        agent.cfg.exp_id = train_manifest["task_id"]
        agent.cfg.exp_name = f"formal-{condition}"
        agent.cfg.workspace_dir = workspace_dir
        agent.cfg.fixed_holdout = type(
            "FixedHoldoutConfig",
            (),
            {
                "enabled": True,
                "evaluation_mode": "terminal_only",
                "bypass_protocol_gates": True,
                "internal_metric_disposition": "search_only",
                "train_manifest_path": str(train_manifest_path),
            },
        )()
        adapter = MLEvolveAuthorityAdapter(agent)
        snapshot = MemorySnapshotLoader(memory_root).load(
            session_overlay_path=run_dir / "session_overlay",
            active_protocol_ref=adapter.active_protocol.key(),
            authority_policy_version=adapter.engine.policy_version,
        )
        adapter.configure_memory_snapshot(snapshot)
        agent.evaluation_authority = adapter
        if condition == "full_decision_admissibility":
            adapter.ledger.append(
                "experience_exposed",
                {
                    "schema": "experience_exposure_event_v2",
                    "clause_id": "clause-a",
                    "target_scope": {
                        "task_id": train_manifest["task_id"],
                        "domain": target_domain,
                    },
                    "source_domains": [target_domain],
                    "transfer_scope": "same_domain",
                    "prompt_sha256": "a" * 64,
                },
            )

        node_id = f"node-{position}"
        code = runtime_code if enforced_protocol else f"print({condition!r})\n"
        node_protocol_observation = runtime_observation
        if condition == blocked_protocol_condition:
            code = blocked_runtime_code
            node_protocol_observation = blocked_runtime_observation
        journal_path = run_dir / "journal.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            journal_path,
            {
                "nodes": [
                    {
                        "id": node_id,
                        "stage": "draft",
                        "draft_role": "coldstart_baseline",
                        "code": code,
                        "exec_time": runtime_exec_time,
                        "is_buggy": False,
                        "is_valid": True,
                        "metric": {
                            "value": float(position),
                            "maximize": bool(enforced_protocol),
                        },
                        "protocol_observation": node_protocol_observation,
                        "actuation_report_refs": [],
                        "role_contract": {
                            "candidate_execution_contract": candidate_contract
                        },
                    }
                ]
            },
        )
        if enforced_protocol:
            terminal_labels = pd.read_csv(split_root / "evaluator_view" / "labels.csv")
            submission = {
                "id": terminal_labels["id"].astype(str).tolist(),
                "has_cactus": [
                    0.8 if index % 2 else 0.2 for index in range(len(terminal_labels))
                ],
            }
        else:
            submission = {
                "id": ["holdout-1", "holdout-2"],
                "EAP": [0.8, 0.1],
                "HPL": [0.1, 0.8],
                "MWS": [0.1, 0.1],
            }
        if condition == invalid_submission_condition:
            submission.pop("MWS")
        pd.DataFrame(submission).to_csv(
            submission_dir / f"submission_{node_id}.csv", index=False
        )
        audit = audit_candidate_code(code, candidate_contract)
        assert audit["valid"] is True
        write_json(
            working_dir / f"candidate_execution_contract_audit_{node_id}.json",
            audit,
        )
        write_json(run_dir / "authority_rollout_report.json", adapter.rollout_report())
        request_path = write_evaluation_request(
            agent.cfg,
            journal_path,
            authority=adapter,
            selected_node_id=node_id,
        )
        assert request_path is not None
        (condition_root / "run_stdout.log").write_text(
            f"synthetic {condition}\n", encoding="utf-8"
        )
        (condition_root / "RUN_EXIT_CODE").write_text("0\n", encoding="utf-8")

    protocol_ref = json.loads(
        (
            output_root
            / "conditions"
            / order[0]
            / "runs"
            / "run-0"
            / "fixed_holdout_evaluation_request.json"
        ).read_text(encoding="utf-8")
    )["authority_writeback"]["active_protocol"]
    observed_protocol_key = (
        f"{protocol_ref['protocol_id']}@{protocol_ref['version']}#"
        f"{protocol_ref['canonical_hash']}"
    )
    assert observed_protocol_key == protocol_key
    pod_identity = {
        "schema": POD_IDENTITY_SCHEMA,
        "execution_kind": "devpod",
        "namespace": "ecepxie",
        "pod_name": "synthetic-formal-gpu-pod",
        "pod_uid": "synthetic-pod-uid",
    }
    block_contract = {
        "schema": CONTRACT_SCHEMA,
        "block_id": "synthetic-task-seed-block",
        "task_id": train_manifest["task_id"],
        "target_task_family": target_task_family,
        "target_domain": target_domain,
        "protocol_ref": observed_protocol_key,
        "split_id": train_manifest["split_id"],
        "metric": train_manifest["metric"],
        "maximize": train_manifest["maximize"],
        "agent_seed": 104729,
        "condition_order": order,
        "steps_per_condition": 1,
        "initial_drafts_per_condition": 1,
        "candidate_execution_contract": candidate_contract,
        "candidate_execution_contract_hash": candidate_contract["contract_hash"],
        "bundle_id": bundle_manifest["bundle_id"],
        "bundle_manifest_sha256": bundle_manifest["manifest_sha256"],
        "bundle_current_file_sha256": sha256_file(memory_root / "CURRENT.json"),
        "bundle_manifest_file_sha256": sha256_file(bundle / "manifest.json"),
        "formal_clause_id": "clause-a",
        "staging_manifest_hash": "f" * 64,
        "staging_gate_hash": "d" * 64,
        "block_template_hash": "c" * 64,
        "source_snapshot_sha256": "e" * 64,
        "source_manifest_file_sha256": "b" * 64,
        "container_image_digest": (
            "docker.io/haomingwang22/mlevolve@sha256:" + "a" * 64
        ),
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "evaluator_manifest_sha256": sha256_file(evaluator_manifest_path),
        "training_pod_identity": pod_identity,
        "agent_time_limit_seconds": 60,
        "condition_launcher_timeout_seconds": 90,
        "contract_hash": "",
    }
    block_contract["contract_hash"] = _payload_hash(block_contract, "contract_hash")
    block_contract_path = tmp_path / "BLOCK_CONTRACT.json"
    write_json(block_contract_path, block_contract)
    isolation = {
        "schema": TRAINING_ISOLATION_SCHEMA,
        "block_id": block_contract["block_id"],
        "runtime_contract_hash": block_contract["contract_hash"],
        "training_pod_identity": pod_identity,
        "source_snapshot_sha256": block_contract["source_snapshot_sha256"],
        "source_manifest_file_sha256": block_contract["source_manifest_file_sha256"],
        "train_manifest_sha256": block_contract["train_manifest_sha256"],
        "bundle_id": block_contract["bundle_id"],
        "bundle_manifest_sha256": block_contract["bundle_manifest_sha256"],
        "bundle_current_file_sha256": block_contract["bundle_current_file_sha256"],
        "container_image_digest": block_contract["container_image_digest"],
        "gpu_visible": True,
        "gpu_count": 1,
        "cpu_count": 8,
        "source_read_only": True,
        "train_view_read_only": True,
        "bundle_read_only": True,
        "solver_secret_single_file_mount": True,
        "whole_workspace_absent": True,
        "evaluator_view_absent": True,
        "terminal_labels_absent": True,
        "all_candidate_import_roots_importable": True,
        "receipt_hash": "",
    }
    isolation["receipt_hash"] = _payload_hash(isolation, "receipt_hash")
    write_json(output_root / "TRAINING_ISOLATION.json", isolation)
    for position, condition in enumerate(order):
        condition_root = output_root / "conditions" / condition
        receipt = {
            "schema": CONDITION_RECEIPT_SCHEMA,
            "block_id": block_contract["block_id"],
            "runtime_contract_hash": block_contract["contract_hash"],
            "condition": condition,
            "position": position,
            "retrieval_control": condition,
            "memory_enabled": condition != "no_memory",
            "memory_system": condition,
            "task_id": block_contract["task_id"],
            "agent_seed": block_contract["agent_seed"],
            "run_exit_code": 0,
            "steps": block_contract["steps_per_condition"],
            "initial_drafts": block_contract["initial_drafts_per_condition"],
            "agent_time_limit_seconds": block_contract["agent_time_limit_seconds"],
            "launcher_timeout_seconds": block_contract[
                "condition_launcher_timeout_seconds"
            ],
            "candidate_execution_contract_hash": candidate_contract["contract_hash"],
            "training_pod_identity": pod_identity,
            "only_experimental_variable": ("external_skill_memory.retrieval_control"),
            "terminal_metric_observed": False,
            "pre_evaluator_score_file_count": 0,
            "run_stdout_sha256": sha256_file(condition_root / "run_stdout.log"),
            "receipt_hash": "",
        }
        receipt["receipt_hash"] = _payload_hash(receipt, "receipt_hash")
        write_json(condition_root / "CONDITION_RUNTIME_RECEIPT.json", receipt)
    return {
        "output_root": output_root,
        "memory_root": memory_root,
        "block_contract_path": block_contract_path,
        "evaluator_manifest_path": evaluator_manifest_path,
        "pod_identity": pod_identity,
    }


def _deletion_attestation(fixture: dict, training: dict) -> Path:
    payload = {
        "schema": DELETION_ATTESTATION_SCHEMA,
        "block_id": training["block_id"],
        "training_manifest_hash": training["manifest_hash"],
        "training_pod_identity": fixture["pod_identity"],
        "delete_requested": True,
        "not_found_verified": True,
        "kubernetes_reason": "NotFound",
        "verified_by": "host_launcher",
        "terminal_metric_observed_before_not_found": False,
        "evaluator_create_allowed_after_verification": True,
        "not_found_verified_at": "2026-07-22T12:00:00Z",
        "attestation_hash": "",
    }
    payload["attestation_hash"] = _payload_hash(payload, "attestation_hash")
    path = fixture["output_root"] / "TRAINING_POD_DELETION_ATTESTATION.json"
    write_json(path, payload)
    return path


def _evaluator_isolation(fixture: dict, training: dict, deletion: Path) -> Path:
    deletion_payload = json.loads(deletion.read_text(encoding="utf-8"))
    payload = {
        "schema": EVALUATOR_ISOLATION_SCHEMA,
        "block_id": training["block_id"],
        "training_manifest_hash": training["manifest_hash"],
        "training_pod_deletion_attestation_hash": deletion_payload["attestation_hash"],
        "evaluator_manifest_sha256": training["evaluator_manifest_sha256"],
        "train_manifest_sha256": training["train_manifest_sha256"],
        "source_snapshot_sha256": training["source_snapshot_sha256"],
        "container_image_digest": training["container_image_digest"],
        "evaluator_pod_identity": {
            "execution_kind": "devpod",
            "namespace": "ecepxie",
            "pod_name": "synthetic-formal-evaluator",
            "pod_uid": "synthetic-evaluator-uid",
        },
        "cpu_only": True,
        "memory_bundle_absent": True,
        "solver_secret_absent": True,
        "solver_environment_absent": True,
        "whole_workspace_absent": True,
        "source_read_only": True,
        "train_view_read_only": True,
        "evaluator_view_read_only": True,
        "created_after_training_pod_not_found": True,
        "receipt_hash": "",
    }
    payload["receipt_hash"] = _payload_hash(payload, "receipt_hash")
    path = fixture["output_root"] / "EVALUATOR_ISOLATION.json"
    write_json(path, payload)
    return path


def test_evaluator_failure_attestation_preserves_post_metric_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "failed-formal-block"
    run_root = root / "conditions" / "no_memory" / "runs" / "run-1"
    overlay_root = run_root / "session_overlay"
    overlay_root.mkdir(parents=True)
    training = {
        "schema": ("decision_admissibility_wp8_tier2_formal_training_manifest_v1"),
        "status": "training_complete_unscored",
        "block_id": "failed-block",
        "task_id": "target-task",
        "agent_seed": 104729,
        "staging_gate_hash": "a" * 64,
        "manifest_hash": "",
    }
    training["manifest_hash"] = _payload_hash(training, "manifest_hash")
    write_json(root / "TRAINING_MANIFEST.json", training)
    identity = {
        "execution_kind": "devpod",
        "namespace": "ecepxie",
        "pod_name": "failed-evaluator",
        "pod_uid": "failed-evaluator-uid",
    }
    creation = {
        "schema": (
            "decision_admissibility_wp8_tier2_evaluator_creation_attestation_v1"
        ),
        "block_id": training["block_id"],
        "evaluator_pod_identity": identity,
        "staging_gate_hash": training["staging_gate_hash"],
        "attestation_hash": "",
    }
    creation["attestation_hash"] = _payload_hash(creation, "attestation_hash")
    write_json(root / "EVALUATOR_POD_CREATION_ATTESTATION.json", creation)
    (root / "EVALUATOR_LAUNCHER_EXIT_CODE").write_text("1\n", encoding="utf-8")
    (root / "EVALUATOR_LAUNCHER.log").write_text(
        "terminal authority failure\n", encoding="utf-8"
    )
    (root / "STATE").write_text("evaluator_failed\n", encoding="utf-8")
    write_json(run_root / "fixed_holdout_scores.json", {"sealed": True})
    status = {
        "schema": "fixed_holdout_terminal_writeback_status_v1",
        "status": "writeback_incomplete",
        "error_type": "ValueError",
        "reason": "Terminal Result Fact Authority denied: payload:fit_scope",
        "status_hash": "",
    }
    status["status_hash"] = _payload_hash(status, "status_hash")
    write_json(run_root / "fixed_holdout_writeback_status.json", status)
    (overlay_root / "events.jsonl").write_text("", encoding="utf-8")

    report = write_evaluator_failure(
        root,
        namespace="ecepxie",
        pod_name=identity["pod_name"],
        pod_uid=identity["pod_uid"],
        failure_detected_at="2026-07-22T18:10:00Z",
        delete_requested_at="2026-07-22T18:10:01Z",
        not_found_verified_at="2026-07-22T18:10:05Z",
        not_found_probe_sha256="b" * 64,
        staging_gate_hash=training["staging_gate_hash"],
    )

    assert report["schema"] == EVALUATOR_FAILURE_SCHEMA
    assert report["classification"] == "authority_denial"
    assert report["terminal_metric_observed"] is True
    assert report["pre_metric_abort"] is False
    assert report["normal_result_fact_count"] == 0
    assert report["not_found_verified"] is True
    assert report["reuse_for_formal_execution"] is False
    assert report["retry_requires_post_failure_preregistration_and_new_roots"] is True
    assert len(report["partial_terminal_artifact_hashes"]) == 1


def test_enforced_training_manifest_freezes_selected_runtime_protocol_evidence(
    tmp_path: Path,
) -> None:
    fixture = _formal_fixture(tmp_path, enforced_protocol=True)

    training = finalize_training_block(
        fixture["output_root"],
        fixture["block_contract_path"],
        fixture["memory_root"],
    )

    assert training["protocol_payload_enforcement"] is True
    for condition in CONDITIONS:
        evidence = training["conditions"][condition][
            "selected_runtime_protocol_evidence"
        ]
        assert evidence["schema"] == (
            "decision_admissibility_wp8_tier2_selected_runtime_protocol_evidence_v1"
        )
        assert evidence["protocol_ref"] == training["protocol_ref"]
        assert evidence["persisted_observation_integrity_verified"] is True
        assert evidence["code_snapshot_frozen_before_execution"] is True
        assert len(evidence["evidence_hash"]) == 64


def test_blocked_selected_runtime_protocol_is_retained_condition_failure(
    tmp_path: Path,
) -> None:
    denied_condition = "full_decision_admissibility"
    fixture = _formal_fixture(
        tmp_path,
        enforced_protocol=True,
        blocked_protocol_condition=denied_condition,
    )

    training = finalize_training_block(
        fixture["output_root"],
        fixture["block_contract_path"],
        fixture["memory_root"],
    )

    assert training["successful_condition_count"] == 4
    assert training["failed_condition_count"] == 1
    denied = training["conditions"][denied_condition]
    assert denied["status"] == "pre_terminal_failure"
    assert denied["failure_classification"] == "authority_denial"
    assert denied["terminal_scoring_authorized"] is False
    assert denied["candidate_reexecution_authorized"] is False
    evidence = denied["selected_runtime_protocol_denial"]
    assert evidence["observation_status"] == "blocked"
    assert evidence["observation_reason"].startswith("missing_protocol_event_plan:")
    assert evidence["retry_as_infrastructure_authorized"] is False
    assert len(evidence["denial_hash"]) == 64

    deletion = _deletion_attestation(fixture, training)
    isolation = _evaluator_isolation(fixture, training, deletion)
    summary = evaluate_formal_block(
        fixture["output_root"],
        fixture["evaluator_manifest_path"],
        deletion,
        isolation,
    )

    assert summary["successful_selected_result_count"] == 4
    assert summary["failed_online_condition_count"] == 1
    assert summary["oracle"]["candidate_union_count"] == 4
    outcome = summary["online_conditions"][denied_condition]
    assert outcome["status"] == "pre_terminal_failure"
    assert outcome["failure_classification"] == "authority_denial"
    assert outcome["terminal_metric_observed"] is False
    assert outcome["result_fact_count"] == 0
    assert outcome["authority_ledger_valid_after_rejection"] is True


def test_tampered_blocked_runtime_protocol_remains_block_fatal(
    tmp_path: Path,
) -> None:
    denied_condition = "full_decision_admissibility"
    fixture = _formal_fixture(
        tmp_path,
        enforced_protocol=True,
        blocked_protocol_condition=denied_condition,
    )
    run_dir = next(
        (fixture["output_root"] / "conditions" / denied_condition / "runs").iterdir()
    )
    journal_path = run_dir / "journal.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    journal["nodes"][0]["protocol_observation"]["source_code_sha256"] = "0" * 64
    write_json(journal_path, journal)
    request_path = run_dir / "fixed_holdout_evaluation_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["journal_sha256"] = sha256_file(journal_path)
    request["request_hash"] = _payload_hash(request, "request_hash")
    write_json(request_path, request)

    with pytest.raises(
        ValueError,
        match="Blocked runtime protocol observation/code mismatch",
    ):
        finalize_training_block(
            fixture["output_root"],
            fixture["block_contract_path"],
            fixture["memory_root"],
        )
    assert not (fixture["output_root"] / "TRAINING_MANIFEST.json").exists()


def test_training_pod_loss_writes_hash_bound_abort_and_seals_partial_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "partial-block"
    condition = root / "conditions" / "no_memory"
    submission = condition / "workspace" / "run" / "submission"
    events = condition / "runs" / "run" / "session_overlay"
    submission.mkdir(parents=True)
    events.mkdir(parents=True)
    (submission / "submission_node-a.csv").write_text(
        "id,prediction\nrow-1,0.5\n",
        encoding="utf-8",
    )
    (events / "events.jsonl").write_text("", encoding="utf-8")
    (condition / "input").symlink_to("/task", target_is_directory=True)
    identity = {
        "schema": POD_IDENTITY_SCHEMA,
        "execution_kind": "devpod",
        "namespace": "ecepxie",
        "pod_name": "lost-training-pod",
        "pod_uid": "lost-training-uid",
    }
    contract = {
        "block_id": "formal-partial-block",
        "task_id": "synthetic-task",
        "agent_seed": 104729,
        "training_pod_identity": identity,
        "source_snapshot_sha256": "a" * 64,
        "staging_manifest_hash": "b" * 64,
        "staging_gate_hash": "c" * 64,
        "container_image_digest": "image@sha256:" + "d" * 64,
        "contract_hash": "",
    }
    contract["contract_hash"] = _payload_hash(contract, "contract_hash")
    write_json(root / "BLOCK_CONTRACT.json", contract)

    report = write_training_infrastructure_abort(
        root,
        namespace="ecepxie",
        pod_name="lost-training-pod",
        pod_uid="lost-training-uid",
        detected_at="2026-07-22T14:29:35Z",
        not_found_verified_at="2026-07-22T14:29:36Z",
        not_found_probe_sha256="e" * 64,
        evaluator_not_found_probe_sha256="f" * 64,
        event_snapshot_sha256="1" * 64,
        event_reasons=["TaintManagerEviction", "NodeNotReady"],
        failure_phase="NotFound",
        pod_status_snapshot_sha256="",
        staging_gate_hash="c" * 64,
    )

    assert report["schema"] == INFRASTRUCTURE_ABORT_SCHEMA
    assert report["classification"] == (
        "kubernetes_node_not_ready_taint_manager_eviction"
    )
    assert report["candidate_submission_count"] == 1
    assert report["terminal_metric_observed"] is False
    assert report["formal_effect_observation"] is False
    assert report["report_hash"] == _payload_hash(report, "report_hash")
    assert not (root.stat().st_mode & stat.S_IWUSR)
    assert all(
        not (path.stat().st_mode & stat.S_IWUSR)
        for path in root.rglob("*")
        if path.exists() and not path.is_symlink()
    )


def test_training_start_error_seals_empty_output_before_formal_start(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "formal-output"
    block_id = "formal-prelaunch-block"
    (output_root / "blocks" / block_id).mkdir(parents=True)
    contract_root = tmp_path / "contract"
    contract_root.mkdir()
    template = {
        "schema": "decision_admissibility_wp8_tier2_formal_block_template_v1",
        "block_id": block_id,
        "task_id": "synthetic-task",
        "target_task_family": "synthetic",
        "target_domain": "tabular",
        "protocol_ref": "mlevolve-default@1#" + "a" * 64,
        "split_id": "synthetic-split",
        "metric": "rmse",
        "maximize": False,
        "agent_seed": 104729,
        "condition_order": list(CONDITIONS),
        "source_snapshot_sha256": "b" * 64,
        "source_manifest_file_sha256": "c" * 64,
        "train_manifest_sha256": "d" * 64,
        "evaluator_manifest_sha256": "e" * 64,
        "bundle_id": "bundle-v1",
        "bundle_manifest_sha256": "f" * 64,
        "bundle_current_file_sha256": "1" * 64,
        "formal_clause_id": "clause-a",
        "candidate_execution_contract": {"contract_hash": "2" * 64},
        "expected_training_pod_name": "failed-prelaunch-pod",
        "expected_training_pod_namespace": "ecepxie",
        "expected_evaluator_pod_name": "never-created-evaluator",
        "expected_evaluator_pod_namespace": "ecepxie",
        "container_image_digest": "image@sha256:" + "3" * 64,
        "output_root_id": "formal-output/blocks/formal-prelaunch-block",
        "template_hash": "",
    }
    template["template_hash"] = _payload_hash(template, "template_hash")
    write_json(contract_root / "BLOCK_TEMPLATE.json", template)

    report = write_training_prelaunch_abort(
        output_root,
        contract_root,
        namespace="ecepxie",
        pod_name="failed-prelaunch-pod",
        pod_uid="failed-prelaunch-uid",
        detected_at="2026-07-22T15:16:46Z",
        event_snapshot_sha256="4" * 64,
        event_reasons=["Scheduled", "Failed"],
        pod_status_snapshot_sha256="5" * 64,
        scheduled_node="broken-gpu-node",
        container_start_reason="StartError",
        container_start_exit_code=128,
        failure_message="failed to initialize NVML: Driver/library version mismatch",
        not_found_verified_at="2026-07-22T15:17:00Z",
        not_found_probe_sha256="6" * 64,
        evaluator_not_found_probe_sha256="7" * 64,
        staging_content_manifest_hash="8" * 64,
        staging_gate_hash="9" * 64,
    )

    assert report["schema"] == PRELAUNCH_ABORT_SCHEMA
    assert report["classification"] == ("gpu_node_nvml_driver_library_version_mismatch")
    assert report["formal_training_started"] is False
    assert report["block_output_empty_at_abort"] is True
    assert report["report_hash"] == _payload_hash(report, "report_hash")
    assert not (output_root.stat().st_mode & stat.S_IWUSR)


def test_training_schema_error_seals_precontract_output(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "formal-precontract-output"
    output_root.mkdir()
    contract_root = tmp_path / "precontract-contract"
    contract_root.mkdir()
    block_id = "formal-precontract-block"
    template = {
        "schema": "decision_admissibility_wp8_tier2_formal_block_template_v1",
        "block_id": block_id,
        "task_id": "synthetic-task",
        "target_task_family": "synthetic",
        "target_domain": "tabular",
        "protocol_ref": "mlevolve-default@1#" + "a" * 64,
        "split_id": "synthetic-split",
        "metric": "rmse",
        "maximize": False,
        "agent_seed": 130363,
        "condition_order": list(CONDITIONS),
        "source_snapshot_sha256": "b" * 64,
        "source_manifest_file_sha256": "c" * 64,
        "train_manifest_sha256": "d" * 64,
        "evaluator_manifest_sha256": "e" * 64,
        "bundle_id": "bundle-v1",
        "bundle_manifest_sha256": "f" * 64,
        "bundle_current_file_sha256": "1" * 64,
        "formal_clause_id": "clause-a",
        "candidate_execution_contract": {"contract_hash": "2" * 64},
        "expected_training_pod_name": "failed-precontract-pod",
        "expected_training_pod_namespace": "ecepxie",
        "expected_evaluator_pod_name": "never-created-evaluator",
        "expected_evaluator_pod_namespace": "ecepxie",
        "container_image_digest": "image@sha256:" + "3" * 64,
        "output_root_id": "formal-output/blocks/formal-precontract-block",
        "template_hash": "",
    }
    template["template_hash"] = _payload_hash(template, "template_hash")
    write_json(contract_root / "BLOCK_TEMPLATE.json", template)
    (output_root / "STATE").write_text("training_launcher_failed\n")
    (output_root / "TRAINING_LAUNCHER_EXIT_CODE").write_text("1\n")
    (output_root / "TRAINING_STARTED_AT").write_text("2026-07-23T03:56:33Z\n")
    (output_root / "TRAINING_LAUNCHER.log").write_text(
        "Traceback (most recent call last):\n"
        "AssertionError: decision_admissibility_wp8_tier2_formal_"
        "continuation_staging_content_v1\n"
    )

    report = write_training_precontract_abort(
        output_root,
        contract_root,
        namespace="ecepxie",
        pod_name="failed-precontract-pod",
        pod_uid="failed-precontract-uid",
        detected_at="2026-07-23T03:58:00Z",
        event_snapshot_sha256="4" * 64,
        event_reasons=["Scheduled", "Started", "Killing"],
        pod_status_snapshot_sha256="",
        scheduled_node="gpu-node",
        not_found_verified_at="2026-07-23T03:58:10Z",
        not_found_probe_sha256="5" * 64,
        evaluator_not_found_probe_sha256="6" * 64,
        staging_content_manifest_hash="7" * 64,
        staging_gate_hash="8" * 64,
    )

    assert report["schema"] == PRECONTRACT_ABORT_SCHEMA
    assert report["classification"] == "staging_schema_compatibility_failure"
    assert report["runtime_block_contract_written"] is False
    assert report["agent_generation_started"] is False
    assert report["terminal_metric_observed"] is False
    assert report["report_hash"] == _payload_hash(report, "report_hash")
    assert not (output_root.stat().st_mode & stat.S_IWUSR)


def test_five_condition_training_and_cpu_finalizers_close_result_facts(
    tmp_path: Path,
) -> None:
    fixture = _formal_fixture(tmp_path)
    training = finalize_training_block(
        fixture["output_root"],
        fixture["block_contract_path"],
        fixture["memory_root"],
    )
    assert training["successful_condition_count"] == 5
    assert training["failed_condition_count"] == 0
    assert training["conditions"]["no_memory"]["experience_exposure_count"] == 0
    assert (
        training["conditions"]["full_decision_admissibility"][
            "formal_method_exposure_count"
        ]
        == 1
    )
    for row in training["conditions"].values():
        ledger = Path(row["run_dir"]) / "authority_events.jsonl"
        assert ledger.stat().st_mode & stat.S_IWUSR

    deletion = _deletion_attestation(fixture, training)
    isolation = _evaluator_isolation(fixture, training, deletion)
    summary = evaluate_formal_block(
        fixture["output_root"],
        fixture["evaluator_manifest_path"],
        deletion,
        isolation,
    )

    assert summary["successful_selected_result_count"] == 5
    assert summary["failed_online_condition_count"] == 0
    assert summary["oracle"]["candidate_union_count"] == 5
    assert summary["oracle"]["scored_candidate_count"] == 5
    assert summary["oracle"]["normal_result_fact_published"] is False
    assert summary["training_pod_absent_before_evaluation"] is True
    for row in summary["online_conditions"].values():
        assert row["result_fact_count"] == 1
        assert row["result_fact_derived_from_refs"] == []
        assert row["authority_ledger_valid_after_writeback"] is True


def test_enforced_five_condition_finalizer_closes_protocol_payloads_end_to_end(
    tmp_path: Path,
) -> None:
    fixture = _formal_fixture(tmp_path, enforced_protocol=True)
    training = finalize_training_block(
        fixture["output_root"],
        fixture["block_contract_path"],
        fixture["memory_root"],
    )
    deletion = _deletion_attestation(fixture, training)
    isolation = _evaluator_isolation(fixture, training, deletion)

    summary = evaluate_formal_block(
        fixture["output_root"],
        fixture["evaluator_manifest_path"],
        deletion,
        isolation,
    )

    assert summary["successful_selected_result_count"] == 5
    assert summary["failed_online_condition_count"] == 0
    for condition, row in summary["online_conditions"].items():
        assert row["status"] == "scored_selected_result", condition
        assert row["result_fact_count"] == 1
        assert row["authority_ledger_valid_after_writeback"] is True
        run_dir = Path(training["conditions"][condition]["run_dir"])
        status = json.loads(
            (run_dir / "fixed_holdout_writeback_status.json").read_text(
                encoding="utf-8"
            )
        )
        assert status["status"] == "complete"


def test_cpu_finalizer_rejects_self_asserted_or_unbound_pod_deletion(
    tmp_path: Path,
) -> None:
    fixture = _formal_fixture(tmp_path)
    training = finalize_training_block(
        fixture["output_root"],
        fixture["block_contract_path"],
        fixture["memory_root"],
    )
    deletion = _deletion_attestation(fixture, training)
    payload = json.loads(deletion.read_text(encoding="utf-8"))
    payload["verified_by"] = "training_pod"
    payload["attestation_hash"] = _payload_hash(payload, "attestation_hash")
    write_json(deletion, payload)

    with pytest.raises(ValueError, match="host launcher"):
        evaluate_formal_block(
            fixture["output_root"],
            fixture["evaluator_manifest_path"],
            deletion,
            _evaluator_isolation(fixture, training, deletion),
        )
    assert not (fixture["output_root"] / "EVALUATION_SUMMARY.json").exists()


def test_rejected_preselected_submission_is_an_outcome_without_result_fact(
    tmp_path: Path,
) -> None:
    rejected_condition = "global_validity_bit"
    fixture = _formal_fixture(tmp_path, invalid_submission_condition=rejected_condition)
    training = finalize_training_block(
        fixture["output_root"],
        fixture["block_contract_path"],
        fixture["memory_root"],
    )
    deletion = _deletion_attestation(fixture, training)
    isolation = _evaluator_isolation(fixture, training, deletion)

    summary = evaluate_formal_block(
        fixture["output_root"],
        fixture["evaluator_manifest_path"],
        deletion,
        isolation,
    )

    assert summary["successful_selected_result_count"] == 4
    assert summary["failed_online_condition_count"] == 1
    assert summary["oracle"]["candidate_union_count"] == 5
    assert summary["oracle"]["scored_candidate_count"] == 4
    rejected = summary["online_conditions"][rejected_condition]
    assert rejected["status"] == "selected_candidate_rejected"
    assert rejected["terminal_metric_observed"] is False
    assert rejected["result_fact_count"] == 0
    assert rejected["authority_ledger_valid_after_rejection"] is True


def test_training_finalizer_recomputes_host_candidate_audits(
    tmp_path: Path,
) -> None:
    fixture = _formal_fixture(tmp_path)
    condition = "authority_only"
    audit_path = next(
        (fixture["output_root"] / "conditions" / condition / "workspace").glob(
            "*/working/candidate_execution_contract_audit_*.json"
        )
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["checks"]["deadline_host_enforced"] = False
    audit["audit_hash"] = _payload_hash(audit, "audit_hash")
    write_json(audit_path, audit)

    with pytest.raises(ValueError, match="host-reproducible"):
        finalize_training_block(
            fixture["output_root"],
            fixture["block_contract_path"],
            fixture["memory_root"],
        )
    assert not (fixture["output_root"] / "TRAINING_MANIFEST.json").exists()
