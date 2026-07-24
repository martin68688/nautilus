"""Score every node submission after a run without feeding scores back."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from fixed_holdout.common import read_manifest, sha256_file, write_json
from fixed_holdout.evaluate import evaluate_submission


SUBMISSION_RE = re.compile(r"^submission_(.+)\.csv$")


def _payload_hash(payload: dict, field: str = "report_hash") -> str:
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


def _journal_nodes(journal_path: Path | None) -> dict[str, dict]:
    if journal_path is None:
        return {}
    journal = json.loads(Path(journal_path).read_text(encoding="utf-8"))
    return {
        str(node.get("id")): node
        for node in journal.get("nodes", [])
        if node.get("id")
    }


def _score_run(
    manifest_path: Path,
    submission_dir: Path,
    output_path: Path,
    *,
    journal_path: Path | None = None,
    evaluation_request_path: Path | None = None,
    finalize_writeback: bool = False,
    formal_training_manifest_path: Path | None = None,
    formal_condition: str | None = None,
) -> dict:
    manifest_path = Path(manifest_path).resolve()
    submission_dir = Path(submission_dir).resolve()
    output_path = Path(output_path).resolve()
    manifest = read_manifest(manifest_path, expected_role="evaluator_view")
    evaluator_manifest_hash = sha256_file(manifest_path)
    train_manifest_path = (
        manifest_path.parent.parent
        / "train_view"
        / "fixed_holdout_manifest.json"
    )
    train_manifest = read_manifest(
        train_manifest_path, expected_role="train_view"
    )
    train_manifest_hash = sha256_file(train_manifest_path)
    for key in (
        "task_id",
        "split_id",
        "public_tree_sha256",
        "holdout_id_sha256",
        "selection_policy",
    ):
        if train_manifest.get(key) != manifest.get(key):
            raise ValueError(
                f"Train/evaluator fixed-holdout manifest mismatch: {key}"
            )
    if train_manifest.get("hidden_labels_present") is not False:
        raise ValueError("Training view contains hidden labels")
    evaluation_request = None
    evaluation_request_hash = ""
    evaluation_request_file_hash = ""
    request_schema = ""
    if evaluation_request_path is not None:
        evaluation_request_path = Path(evaluation_request_path).resolve()
        evaluation_request_file_hash = sha256_file(evaluation_request_path)
        evaluation_request = json.loads(
            evaluation_request_path.read_text(encoding="utf-8")
        )
        if not isinstance(evaluation_request, dict):
            raise ValueError("Fixed-holdout evaluation request must be an object")
        request_schema = str(evaluation_request.get("request_schema") or "")
        if request_schema not in {
            "fixed_holdout_evaluation_request_v2",
            "fixed_holdout_evaluation_request_v3",
        }:
            raise ValueError("Unsupported fixed-holdout evaluation request")
        if evaluation_request.get("request_hash") != _payload_hash(
            evaluation_request, "request_hash"
        ):
            raise ValueError("Fixed-holdout evaluation request hash mismatch")
        evaluation_request_hash = str(evaluation_request["request_hash"])
    submission_paths = sorted(submission_dir.glob("submission_*.csv"))
    candidate_inventory = []
    for submission_path in submission_paths:
        match = SUBMISSION_RE.match(submission_path.name)
        if match:
            candidate_inventory.append(
                {
                    "node_id": match.group(1),
                    "submission": submission_path.name,
                    "submission_sha256": sha256_file(submission_path),
                }
            )
    candidate_set_hash = _payload_hash(
        {"candidate_inventory": candidate_inventory},
        field="unused",
    )
    if request_schema == "fixed_holdout_evaluation_request_v3":
        if evaluation_request.get("status") != "awaiting_external_evaluator":
            raise ValueError("Fixed-holdout request is not awaiting evaluation")
        if evaluation_request.get("scores_were_visible_during_search") is not False:
            raise ValueError("Evaluation request exposes fixed-holdout scores")
        if evaluation_request.get("selection_policy") != "terminal_only":
            raise ValueError("Evaluation request is not terminal-only")
        if (
            evaluation_request.get(
                "selection_frozen_before_terminal_evaluation"
            )
            is not True
        ):
            raise ValueError("System selection was not frozen before evaluation")
        if evaluation_request.get("candidate_inventory") != candidate_inventory:
            raise ValueError(
                "Candidate inventory changed after the pre-evaluator selection freeze"
            )
        if evaluation_request.get("candidate_set_hash") != candidate_set_hash:
            raise ValueError(
                "Candidate-set hash changed after the pre-evaluator selection freeze"
            )
        if Path(
            str(evaluation_request.get("submission_dir") or "")
        ).resolve() != submission_dir:
            raise ValueError("Evaluation request submission directory mismatch")
        if evaluation_request.get("train_manifest_sha256") != train_manifest_hash:
            raise ValueError("Train manifest changed after terminal handoff")
        if journal_path is None:
            raise ValueError("V3 fixed-holdout scoring requires the frozen journal")
        journal_path = Path(journal_path).resolve()
        if Path(
            str(evaluation_request.get("journal_path") or "")
        ).resolve() != journal_path:
            raise ValueError("Evaluation request journal path mismatch")
        if evaluation_request.get("journal_sha256") != sha256_file(journal_path):
            raise ValueError("Journal changed after terminal handoff")
        for key in ("task_id", "split_id", "metric", "maximize"):
            if evaluation_request.get(key) != manifest.get(key):
                raise ValueError(f"Evaluation request/manifest mismatch: {key}")
        selected_node_id = str(
            evaluation_request.get("selected_node_id") or ""
        )
        selected_inventory = [
            row
            for row in candidate_inventory
            if row["node_id"] == selected_node_id
        ]
        if len(selected_inventory) != 1:
            raise ValueError("Pre-evaluator selection is not in the frozen candidate set")
        if (
            selected_inventory[0]["submission"]
            != evaluation_request.get("selected_submission")
            or selected_inventory[0]["submission_sha256"]
            != evaluation_request.get("selected_submission_sha256")
        ):
            raise ValueError("Selected submission changed after selection freeze")
    nodes = _journal_nodes(journal_path)
    if request_schema == "fixed_holdout_evaluation_request_v3":
        if selected_node_id not in nodes:
            raise ValueError("Pre-evaluator selection references an unknown node")
    results = []
    for submission_path in submission_paths:
        match = SUBMISSION_RE.match(submission_path.name)
        if not match:
            continue
        node_id = match.group(1)
        try:
            score = evaluate_submission(manifest_path, submission_path)
            item = {
                "node_id": node_id,
                "submission": submission_path.name,
                "status": "scored",
                "score": score["score"],
                "submission_sha256": score["submission_sha256"],
            }
        except Exception as exc:
            item = {
                "node_id": node_id,
                "submission": submission_path.name,
                "status": "rejected",
                "reason": str(exc),
            }
        if node_id in nodes:
            item["stage"] = nodes[node_id].get("stage")
            item["draft_role"] = nodes[node_id].get("draft_role")
            metric = nodes[node_id].get("metric") or {}
            item["internal_metric"] = metric.get("value")
            item["internal_metric_disposition"] = "search_only"
        results.append(item)

    scored = [item for item in results if item["status"] == "scored"]
    frozen_hashes = {
        item["node_id"]: item["submission_sha256"]
        for item in candidate_inventory
    }
    if any(
        item.get("status") == "scored"
        and item.get("submission_sha256")
        != frozen_hashes.get(str(item.get("node_id")))
        for item in results
    ):
        raise ValueError("A fixed-holdout submission changed during scoring")
    post_score_inventory = []
    for submission_path in sorted(submission_dir.glob("submission_*.csv")):
        match = SUBMISSION_RE.match(submission_path.name)
        if match:
            post_score_inventory.append(
                {
                    "node_id": match.group(1),
                    "submission": submission_path.name,
                    "submission_sha256": sha256_file(submission_path),
                }
            )
    if post_score_inventory != candidate_inventory:
        raise ValueError("Candidate inventory changed during terminal scoring")
    if sha256_file(manifest_path) != evaluator_manifest_hash:
        raise ValueError("Evaluator manifest changed during terminal scoring")
    if sha256_file(train_manifest_path) != train_manifest_hash:
        raise ValueError("Train manifest changed during terminal scoring")
    if request_schema == "fixed_holdout_evaluation_request_v3":
        if sha256_file(evaluation_request_path) != evaluation_request_file_hash:
            raise ValueError("Evaluation request changed during terminal scoring")
        if sha256_file(journal_path) != evaluation_request.get("journal_sha256"):
            raise ValueError("Journal changed during terminal scoring")
    scored.sort(key=lambda item: item["score"], reverse=bool(manifest["maximize"]))
    common_report = {
        "schema": manifest["schema"],
        "task_id": manifest["task_id"],
        "split_id": manifest["split_id"],
        "metric": manifest["metric"],
        "maximize": manifest["maximize"],
        "selection_policy": "terminal_only",
        "scores_were_visible_during_search": False,
        "candidate_set_frozen_before_scoring": True,
        "candidate_set_hash": candidate_set_hash,
        "candidate_inventory": candidate_inventory,
        "evaluator_manifest_sha256": evaluator_manifest_hash,
        "train_manifest_sha256": train_manifest_hash,
        "journal_sha256": (
            sha256_file(Path(journal_path)) if journal_path is not None else ""
        ),
        "results": results,
        "terminal_score_sealed": True,
        "report_hash": "",
    }
    if request_schema == "fixed_holdout_evaluation_request_v3":
        for key in ("task_id", "split_id", "metric", "maximize"):
            if evaluation_request.get(key) != common_report.get(key):
                raise ValueError(f"Evaluation request/manifest mismatch: {key}")
        selected_node_id = str(
            evaluation_request.get("selected_node_id") or ""
        )
        selected_rows = [
            row for row in results if str(row.get("node_id")) == selected_node_id
        ]
        if len(selected_rows) != 1 or selected_rows[0].get("status") != "scored":
            raise ValueError(
                "The pre-evaluator selected node is not uniquely and validly scored"
            )
        selected = selected_rows[0]
        if selected.get("submission") != evaluation_request.get(
            "selected_submission"
        ) or selected.get("submission_sha256") != evaluation_request.get(
            "selected_submission_sha256"
        ):
            raise ValueError("Selected submission changed after selection freeze")
        report = {
            **common_report,
            "report_schema": "fixed_holdout_terminal_score_report_v3",
            "evaluation_request_hash": evaluation_request_hash,
            "evaluation_request_sha256": evaluation_request_file_hash,
            "selection_frozen_before_terminal_evaluation": True,
            "selection_basis": evaluation_request.get("selection_basis") or {},
            "selected_node_id": selected_node_id,
            "selected_score": selected["score"],
            "selected_submission": selected["submission"],
            "selected_submission_sha256": selected["submission_sha256"],
            "oracle_best_node_id": scored[0]["node_id"] if scored else None,
            "oracle_best_score": scored[0]["score"] if scored else None,
            "oracle_best_submission_sha256": (
                scored[0]["submission_sha256"] if scored else None
            ),
            "system_selection_used_terminal_labels": False,
            "oracle_selection_is_host_only": True,
        }
    else:
        report = {
            **common_report,
            "report_schema": "fixed_holdout_terminal_score_report_v2",
            "best_node_id": scored[0]["node_id"] if scored else None,
            "best_score": scored[0]["score"] if scored else None,
            "best_submission_sha256": (
                scored[0]["submission_sha256"] if scored else None
            ),
        }
    report["report_hash"] = _payload_hash(report)
    write_json(Path(output_path), report)

    selected_solution_id = (
        report.get("selected_node_id")
        if report.get("report_schema") == "fixed_holdout_terminal_score_report_v3"
        else report.get("best_node_id")
    )
    if selected_solution_id in nodes:
        selected_code = nodes[selected_solution_id].get("code")
        if selected_code:
            filename = (
                "selected_solution_fixed_holdout.py"
                if report.get("report_schema")
                == "fixed_holdout_terminal_score_report_v3"
                else "best_solution_fixed_holdout.py"
            )
            Path(output_path).with_name(filename).write_text(
                str(selected_code), encoding="utf-8"
            )
    if finalize_writeback:
        if evaluation_request_path is None:
            raise ValueError(
                "Terminal writeback requires an evaluation request"
            )
        from fixed_holdout.writeback import finalize_result_writeback

        finalize_result_writeback(
            evaluation_request_path,
            output_path,
            manifest_path,
            formal_training_manifest_path=formal_training_manifest_path,
            formal_condition=formal_condition,
        )
    return report


def score_run(
    manifest_path: Path,
    submission_dir: Path,
    output_path: Path,
    *,
    journal_path: Path | None = None,
    evaluation_request_path: Path | None = None,
    finalize_writeback: bool = False,
    formal_training_manifest_path: Path | None = None,
    formal_condition: str | None = None,
) -> dict:
    """Score a frozen run and make terminal-closure failures explicit."""

    try:
        return _score_run(
            manifest_path,
            submission_dir,
            output_path,
            journal_path=journal_path,
            evaluation_request_path=evaluation_request_path,
            finalize_writeback=finalize_writeback,
            formal_training_manifest_path=formal_training_manifest_path,
            formal_condition=formal_condition,
        )
    except Exception as error:
        from fixed_holdout.writeback import TerminalWritebackError

        if isinstance(error, TerminalWritebackError):
            raise
        if finalize_writeback:
            from fixed_holdout.writeback import record_terminal_writeback_failure

            status_path = Path(output_path).resolve().with_name(
                "fixed_holdout_writeback_status.json"
            )
            record_terminal_writeback_failure(
                status_path,
                error,
                request_path=evaluation_request_path,
                score_report_path=output_path,
            )
            raise TerminalWritebackError(str(error)) from error
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--evaluation-request", type=Path)
    parser.add_argument("--finalize-writeback", action="store_true")
    parser.add_argument("--formal-training-manifest", type=Path)
    parser.add_argument("--formal-condition")
    args = parser.parse_args()
    report = score_run(
        args.manifest,
        args.submission_dir,
        args.output,
        journal_path=args.journal,
        evaluation_request_path=args.evaluation_request,
        finalize_writeback=args.finalize_writeback,
        formal_training_manifest_path=args.formal_training_manifest,
        formal_condition=args.formal_condition,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
