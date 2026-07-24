"""CPU-only terminal evaluation for one frozen formal Tier-2 block."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from authority.ledger import AuthorityLedger
from authority.memory_snapshot import SessionOverlay
from fixed_holdout.common import read_manifest, sha256_file, write_json
from fixed_holdout.evaluate import evaluate_submission
from fixed_holdout.formal_runtime import validate_evaluator_isolation_receipt
from fixed_holdout.score_run import score_run


SCHEMA = "decision_admissibility_wp8_tier2_formal_evaluation_v1"
DELETION_ATTESTATION_SCHEMA = (
    "decision_admissibility_wp8_tier2_training_pod_deletion_attestation_v1"
)


def _hash_payload(payload: Mapping[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _inventory(submission_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "node_id": path.stem.removeprefix("submission_"),
            "submission": path.name,
            "submission_sha256": sha256_file(path),
        }
        for path in sorted(submission_dir.glob("submission_*.csv"))
    ]


def _verify_result_fact(
    request: Mapping[str, Any],
    selected_node_id: str,
) -> dict[str, Any]:
    descriptor = request.get("authority_writeback") or {}
    if descriptor.get("status") == "writeback_incomplete":
        raise ValueError("Successful formal condition has no Authority descriptor")
    ledger_path = Path(str(descriptor.get("authority_ledger_path") or ""))
    ledger = AuthorityLedger(ledger_path)
    if not ledger.verify():
        raise ValueError("Authority Ledger is invalid after terminal writeback")
    overlay = SessionOverlay(
        descriptor["session_overlay_path"],
        overlay_id=descriptor["session_overlay_id"],
    )
    results = [
        event
        for event in overlay.events()
        if event.event_type == "memory_claim"
        and event.payload.get("publication_class") == "result_fact"
    ]
    if len(results) != 1:
        raise ValueError("Successful formal condition lacks exactly one Result Fact")
    payload = dict(results[0].payload)
    if payload.get("artifact_id") != selected_node_id:
        raise ValueError("Result Fact is not bound to the preselected node")
    if payload.get("derived_from_refs") != []:
        raise ValueError("Independent Result Fact incorrectly claims derivation")
    return {
        "result_fact_count": 1,
        "result_fact_artifact_id": payload["artifact_id"],
        "result_fact_derived_from_refs": [],
        "authority_ledger_valid_after_writeback": True,
        "authority_ledger_sha256_after_writeback": sha256_file(ledger_path),
        "session_overlay_manifest_sha256_after_writeback": overlay.manifest[
            "manifest_sha256"
        ],
        "result_fact_event_id": results[0].event_id,
        "result_fact_event_hash": results[0].event_hash,
    }


def _verify_no_result_fact(request: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = request.get("authority_writeback") or {}
    ledger_path = Path(str(descriptor.get("authority_ledger_path") or ""))
    ledger = AuthorityLedger(ledger_path)
    if not ledger.verify():
        raise ValueError("Authority Ledger is invalid after rejected selection")
    overlay = SessionOverlay(
        descriptor["session_overlay_path"],
        overlay_id=descriptor["session_overlay_id"],
    )
    results = [
        event
        for event in overlay.events()
        if event.event_type == "memory_claim"
        and event.payload.get("publication_class") == "result_fact"
    ]
    if results:
        raise ValueError("Rejected selected candidate published a Result Fact")
    return {
        "result_fact_count": 0,
        "authority_ledger_valid_after_rejection": True,
        "authority_ledger_sha256_after_rejection": sha256_file(ledger_path),
        "session_overlay_manifest_sha256_after_rejection": overlay.manifest[
            "manifest_sha256"
        ],
    }


def _verify_deletion_attestation(
    path: Path,
    training: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _read(path)
    if payload.get("schema") != DELETION_ATTESTATION_SCHEMA:
        raise ValueError("Training Pod deletion attestation schema mismatch")
    if payload.get("attestation_hash") != _hash_payload(
        payload, "attestation_hash"
    ):
        raise ValueError("Training Pod deletion attestation hash mismatch")
    if payload.get("block_id") != training.get("block_id"):
        raise ValueError("Training Pod deletion attestation block mismatch")
    if payload.get("training_manifest_hash") != training.get("manifest_hash"):
        raise ValueError("Training Pod deletion attestation manifest mismatch")
    if payload.get("training_pod_identity") != training.get(
        "training_pod_identity"
    ):
        raise ValueError("Training Pod deletion attestation identity mismatch")
    if payload.get("delete_requested") is not True:
        raise ValueError("Training Pod deletion was not requested")
    if payload.get("not_found_verified") is not True:
        raise ValueError("Training Pod absence was not verified")
    if payload.get("kubernetes_reason") != "NotFound":
        raise ValueError("Training Pod deletion did not end in NotFound")
    if payload.get("verified_by") != "host_launcher":
        raise ValueError("Training Pod deletion was not verified by the host launcher")
    if payload.get("terminal_metric_observed_before_not_found") is not False:
        raise ValueError("Terminal metric was observed before training Pod deletion")
    if payload.get("evaluator_create_allowed_after_verification") is not True:
        raise ValueError("Evaluator ordering was not fail-closed")
    if not str(payload.get("not_found_verified_at") or ""):
        raise ValueError("Training Pod deletion attestation lacks verification time")
    return payload


def evaluate_formal_block(
    output_root: str | Path,
    evaluator_manifest_path: str | Path,
    deletion_attestation_path: str | Path,
    evaluator_isolation_path: str | Path,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    evaluator_manifest_path = Path(evaluator_manifest_path).resolve()
    summary_path = output_root / "EVALUATION_SUMMARY.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    training_path = output_root / "TRAINING_MANIFEST.json"
    training = _read(training_path)
    if training.get("schema") != "decision_admissibility_wp8_tier2_formal_training_manifest_v1":
        raise ValueError("Formal training manifest schema mismatch")
    if training.get("manifest_hash") != _hash_payload(training, "manifest_hash"):
        raise ValueError("Formal training manifest hash mismatch")
    if training.get("status") != "training_complete_unscored":
        raise ValueError("Formal block is not ready for terminal evaluation")
    deletion_attestation_path = Path(deletion_attestation_path).resolve()
    deletion_attestation = _verify_deletion_attestation(
        deletion_attestation_path, training
    )
    evaluator_manifest = read_manifest(
        evaluator_manifest_path, expected_role="evaluator_view"
    )
    if sha256_file(evaluator_manifest_path) != training.get(
        "evaluator_manifest_sha256"
    ):
        raise ValueError("Formal evaluator manifest hash mismatch")
    train_manifest_path = (
        evaluator_manifest_path.parent.parent
        / "train_view"
        / "fixed_holdout_manifest.json"
    )
    train_manifest = read_manifest(train_manifest_path, expected_role="train_view")
    if sha256_file(train_manifest_path) != training.get("train_manifest_sha256"):
        raise ValueError("Formal train manifest hash mismatch")
    for key in ("task_id", "split_id", "metric", "maximize"):
        if evaluator_manifest.get(key) != training.get(key):
            raise ValueError(f"Formal evaluator/training mismatch: {key}")
        if train_manifest.get(key) != training.get(key):
            raise ValueError(f"Formal train/training mismatch: {key}")
    if evaluator_manifest.get("protocol_ref") != training.get("protocol_ref"):
        raise ValueError("Formal evaluator protocol mismatch")
    if train_manifest.get("hidden_labels_present") is not False:
        raise ValueError("Formal train view contains terminal labels")
    evaluator_isolation_path = Path(evaluator_isolation_path).resolve()
    evaluator_isolation = validate_evaluator_isolation_receipt(
        evaluator_isolation_path,
        block_id=str(training["block_id"]),
        training_manifest_hash=str(training["manifest_hash"]),
        deletion_attestation_hash=str(deletion_attestation["attestation_hash"]),
        evaluator_manifest_sha256=str(training["evaluator_manifest_sha256"]),
        train_manifest_sha256=str(training["train_manifest_sha256"]),
        source_snapshot_sha256=str(training["source_snapshot_sha256"]),
        container_image_digest=str(training["container_image_digest"]),
    )
    order = list(map(str, training.get("condition_order") or []))
    conditions = training.get("conditions") or {}
    if set(order) != set(conditions) or len(order) != 5:
        raise ValueError("Formal evaluation condition universe mismatch")

    union_inventory: list[dict[str, Any]] = []
    for position, condition in enumerate(order):
        row = conditions[condition]
        if row.get("status") != "training_complete_unscored":
            continue
        for candidate in row.get("candidate_inventory") or []:
            union_inventory.append(
                {
                    "condition": condition,
                    "condition_position": position,
                    "node_id": candidate["node_id"],
                    "submission": candidate["submission"],
                    "submission_sha256": candidate["submission_sha256"],
                    "candidate_set_hash": row["candidate_set_hash"],
                }
            )
    union_inventory.sort(
        key=lambda row: (
            int(row["condition_position"]),
            str(row["node_id"]),
            str(row["submission_sha256"]),
        )
    )
    union_hash = _hash_payload(
        {"candidate_union_inventory": union_inventory}, "unused"
    )

    online: dict[str, dict[str, Any]] = {}
    oracle_candidates: list[dict[str, Any]] = []
    for position, condition in enumerate(order):
        train_row = conditions[condition]
        if train_row.get("status") != "training_complete_unscored":
            for relative, expected in (train_row.get("file_hashes") or {}).items():
                path = output_root / str(relative)
                if not path.is_file() or sha256_file(path) != expected:
                    raise ValueError(
                        "Frozen failed-condition artifact changed before "
                        f"evaluation: {condition}:{relative}"
                    )
            runtime_path = output_root / str(
                train_row.get("condition_runtime_receipt_path") or ""
            )
            failure_path = output_root / str(
                train_row.get("failure_receipt_path") or ""
            )
            if (
                not runtime_path.is_file()
                or sha256_file(runtime_path)
                != train_row.get("condition_runtime_receipt_sha256")
            ):
                raise ValueError("Failed condition runtime receipt changed")
            if (
                not failure_path.is_file()
                or sha256_file(failure_path)
                != train_row.get("failure_receipt_sha256")
            ):
                raise ValueError("Failed condition failure receipt changed")
            runtime_receipt = _read(runtime_path)
            failure_receipt = _read(failure_path)
            if runtime_receipt.get("receipt_hash") != _hash_payload(
                runtime_receipt, "receipt_hash"
            ):
                raise ValueError("Failed condition runtime receipt hash mismatch")
            if failure_receipt.get("receipt_hash") != _hash_payload(
                failure_receipt, "receipt_hash"
            ):
                raise ValueError("Failed condition failure receipt hash mismatch")
            failure_row = {
                "condition": condition,
                "position": position,
                "status": "pre_terminal_failure",
                "terminal_metric_observed": False,
                "failure_receipt_hash": train_row.get("failure_receipt_hash", ""),
            }
            request_value = str(train_row.get("evaluation_request_path") or "")
            if request_value:
                request_path = Path(request_value)
                request = _read(request_path)
                failure_row.update(_verify_no_result_fact(request))
                failure_row["failure_classification"] = str(
                    train_row.get("failure_classification") or ""
                )
            online[condition] = failure_row
            continue

        for relative, expected in (train_row.get("file_hashes") or {}).items():
            path = output_root / str(relative)
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(
                    f"Frozen training artifact changed before evaluation: {condition}:{relative}"
                )
        request_path = Path(train_row["evaluation_request_path"])
        journal_path = Path(train_row["journal_path"])
        submission_dir = Path(train_row["submission_dir"])
        request = _read(request_path)
        if request.get("request_hash") != _hash_payload(request, "request_hash"):
            raise ValueError("Evaluation request hash mismatch")
        if request.get("candidate_inventory") != _inventory(submission_dir):
            raise ValueError("Candidate set changed before terminal evaluation")
        if request.get("candidate_set_hash") != train_row.get("candidate_set_hash"):
            raise ValueError("Candidate-set binding changed before evaluation")
        if request.get("journal_sha256") != sha256_file(journal_path):
            raise ValueError("Journal changed before terminal evaluation")

        candidate_results = []
        by_node: dict[str, dict[str, Any]] = {}
        for candidate in request.get("candidate_inventory") or []:
            submission_path = submission_dir / candidate["submission"]
            try:
                score = evaluate_submission(
                    evaluator_manifest_path, submission_path
                )
                item = {
                    **candidate,
                    "status": "scored",
                    "score": score["score"],
                }
                oracle_candidates.append(
                    {
                        "condition": condition,
                        "condition_position": position,
                        "node_id": candidate["node_id"],
                        "submission_sha256": candidate["submission_sha256"],
                        "score": score["score"],
                    }
                )
            except Exception as error:
                item = {
                    **candidate,
                    "status": "rejected",
                    "reason_type": type(error).__name__,
                    "reason": str(error),
                }
            candidate_results.append(item)
            by_node[str(candidate["node_id"])] = item
        if _inventory(submission_dir) != request.get("candidate_inventory"):
            raise ValueError("Candidate set changed during independent scoring")
        candidate_report = {
            "schema": "decision_admissibility_wp8_tier2_all_candidate_score_report_v1",
            "condition": condition,
            "candidate_set_hash": train_row["candidate_set_hash"],
            "candidate_inventory": request["candidate_inventory"],
            "results": candidate_results,
            "scores_visible_during_search": False,
            "host_only_oracle_eligible": True,
            "report_hash": "",
        }
        candidate_report["report_hash"] = _hash_payload(
            candidate_report, "report_hash"
        )
        run_dir = Path(train_row["run_dir"])
        candidate_report_path = run_dir / "all_candidate_terminal_scores.json"
        if candidate_report_path.exists():
            raise FileExistsError(candidate_report_path)
        write_json(candidate_report_path, candidate_report)

        selected_node_id = str(train_row["selected_node_id"])
        selected = by_node.get(selected_node_id) or {}
        if selected.get("status") != "scored":
            failure = {
                "schema": "decision_admissibility_wp8_tier2_selected_candidate_failure_v1",
                "condition": condition,
                "selected_node_id": selected_node_id,
                "selected_status": selected.get("status", "missing"),
                "selected_reason": selected.get("reason", ""),
                "terminal_metric_observed": False,
                "oracle_candidates_may_still_be_scored": True,
                "failure_hash": "",
            }
            failure["failure_hash"] = _hash_payload(failure, "failure_hash")
            failure_path = run_dir / "selected_candidate_failure.json"
            write_json(failure_path, failure)
            online[condition] = {
                "condition": condition,
                "position": position,
                "status": "selected_candidate_rejected",
                "selected_node_id": selected_node_id,
                "terminal_metric_observed": False,
                "candidate_report_hash": candidate_report["report_hash"],
                "failure_hash": failure["failure_hash"],
                **_verify_no_result_fact(request),
            }
            continue

        score_path = run_dir / "fixed_holdout_scores.json"
        if score_path.exists():
            raise FileExistsError(score_path)
        report = score_run(
            evaluator_manifest_path,
            submission_dir,
            score_path,
            journal_path=journal_path,
            evaluation_request_path=request_path,
            finalize_writeback=True,
            formal_training_manifest_path=training_path,
            formal_condition=condition,
        )
        if report.get("report_schema") != "fixed_holdout_terminal_score_report_v3":
            raise ValueError("Formal evaluator did not produce a V3 score report")
        if report.get("selected_node_id") != selected_node_id:
            raise ValueError("System result does not use the preselected node")
        if not math.isclose(
            float(report["selected_score"]),
            float(selected["score"]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("Independent/system selected scores disagree")
        if report.get("system_selection_used_terminal_labels") is not False:
            raise ValueError("Formal system selection used terminal labels")
        if report.get("oracle_selection_is_host_only") is not True:
            raise ValueError("Per-system Oracle was not host-only")
        status_path = run_dir / "fixed_holdout_writeback_status.json"
        status = _read(status_path)
        if status.get("status") != "complete":
            raise ValueError("Successful formal Result Fact writeback is incomplete")
        result_fact = _verify_result_fact(request, selected_node_id)
        online[condition] = {
            "condition": condition,
            "position": position,
            "status": "scored_selected_result",
            "metric": report["metric"],
            "maximize": report["maximize"],
            "selected_node_id": selected_node_id,
            "selected_score": report["selected_score"],
            "selected_submission_sha256": report[
                "selected_submission_sha256"
            ],
            "candidate_set_hash": report["candidate_set_hash"],
            "candidate_report_hash": candidate_report["report_hash"],
            "score_report_hash": report["report_hash"],
            "score_report_sha256": sha256_file(score_path),
            "writeback_status_hash": status["status_hash"],
            "terminal_metric_observed": True,
            "system_selection_used_terminal_labels": False,
            **result_fact,
        }

    maximize = bool(training["maximize"])
    oracle_candidates.sort(
        key=lambda row: (
            -float(row["score"]) if maximize else float(row["score"]),
            int(row["condition_position"]),
            str(row["node_id"]),
            str(row["submission_sha256"]),
        )
    )
    oracle = {
        "schema": "decision_admissibility_wp8_tier2_host_oracle_v1",
        "agent_visible": False,
        "feeds_back_into_search": False,
        "cross_seed_selection": False,
        "candidate_union_hash": union_hash,
        "candidate_union_count": len(union_inventory),
        "scored_candidate_count": len(oracle_candidates),
        "best_condition": (
            oracle_candidates[0]["condition"] if oracle_candidates else None
        ),
        "best_node_id": (
            oracle_candidates[0]["node_id"] if oracle_candidates else None
        ),
        "best_submission_sha256": (
            oracle_candidates[0]["submission_sha256"]
            if oracle_candidates
            else None
        ),
        "best_score": (
            oracle_candidates[0]["score"] if oracle_candidates else None
        ),
        "normal_result_fact_published": False,
        "oracle_hash": "",
    }
    oracle["oracle_hash"] = _hash_payload(oracle, "oracle_hash")
    write_json(output_root / "HOST_ORACLE.json", oracle)

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "evaluation_complete",
        "block_id": training["block_id"],
        "task_id": training["task_id"],
        "agent_seed": training["agent_seed"],
        "protocol_ref": training["protocol_ref"],
        "metric": training["metric"],
        "maximize": training["maximize"],
        "condition_order": order,
        "online_conditions": online,
        "online_condition_count": 5,
        "successful_selected_result_count": sum(
            row.get("status") == "scored_selected_result"
            for row in online.values()
        ),
        "failed_online_condition_count": sum(
            row.get("status") != "scored_selected_result"
            for row in online.values()
        ),
        "oracle": oracle,
        "host_owned_terminal_evaluator": True,
        "training_pod_absent_before_evaluation": True,
        "training_pod_deletion_attestation_sha256": sha256_file(
            deletion_attestation_path
        ),
        "training_pod_deletion_attestation_hash": deletion_attestation[
            "attestation_hash"
        ],
        "training_pod_identity": deletion_attestation[
            "training_pod_identity"
        ],
        "evaluator_pod_identity": evaluator_isolation["evaluator_pod_identity"],
        "evaluator_isolation_receipt_hash": evaluator_isolation["receipt_hash"],
        "evaluator_isolation_receipt_sha256": sha256_file(
            evaluator_isolation_path
        ),
        "evaluator_cpu_only": True,
        "evaluator_solver_secret_absent": True,
        "evaluator_memory_bundle_absent": True,
        "scores_used_for_further_search": False,
        "system_results_use_preselected_nodes": True,
        "oracle_uses_frozen_candidate_union": True,
        "oracle_publishes_normal_result_fact": False,
        "effect_claim_authorized": False,
        "formal_tier2_evidence": True,
        "training_manifest_hash": training["manifest_hash"],
        "evaluator_manifest_sha256": sha256_file(evaluator_manifest_path),
        "summary_hash": "",
    }
    summary["summary_hash"] = _hash_payload(summary, "summary_hash")
    write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluator-manifest", type=Path, required=True)
    parser.add_argument(
        "--training-pod-deletion-attestation", type=Path, required=True
    )
    parser.add_argument("--evaluator-isolation", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_formal_block(
        args.output_root,
        args.evaluator_manifest,
        args.training_pod_deletion_attestation,
        args.evaluator_isolation,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
