"""Terminal-only Kaggle scoring for a preselected frozen submission.

The search process must finish and freeze its selected submission before this
module is invoked.  Official scores are never returned to the search loop.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fixed_holdout.common import sha256_file
from official_submission import (
    RECEIPT_SCHEMA as NATIVE_RECEIPT_SCHEMA,
    REQUEST_SCHEMA as NATIVE_REQUEST_SCHEMA,
    validate_submission as validate_official_submission,
)


SPEC_SCHEMA = "mlevolve_kaggle_terminal_evaluator_v1"
REPORT_SCHEMA = "mlevolve_kaggle_terminal_score_report_v1"
MEASUREMENT_SCHEMA = "mlevolve_official_measurement_v1"


def _payload_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _verify_hash(payload: Mapping[str, Any], field: str, label: str) -> None:
    if payload.get(field) != _payload_hash(payload, field):
        raise ValueError(f"{label} hash mismatch")


def write_evaluator_spec(
    output_path: Path,
    *,
    task_id: str,
    competition: str,
    metric: str,
    maximize: bool,
    sample_submission: Path,
    id_column: str,
    prediction_kind: str,
    score_field_preference: Sequence[str] = ("privateScore", "publicScore"),
    probability_row_sum_tolerance: float = 1e-4,
    poll_seconds: float = 15,
    poll_timeout_seconds: float = 1800,
) -> dict[str, Any]:
    """Write one immutable task-specific Kaggle evaluator specification."""

    sample_submission = Path(sample_submission).resolve(strict=True)
    if prediction_kind not in {
        "numeric",
        "probability",
        "multiclass_probability",
    }:
        raise ValueError(f"Unsupported prediction_kind: {prediction_kind}")
    for field in score_field_preference:
        if field not in {"privateScore", "publicScore"}:
            raise ValueError(f"Unsupported Kaggle score field: {field}")
    spec: dict[str, Any] = {
        "schema": SPEC_SCHEMA,
        "task_id": str(task_id),
        "competition": str(competition),
        "metric": str(metric),
        "maximize": bool(maximize),
        "sample_submission": str(sample_submission),
        "sample_submission_sha256": sha256_file(sample_submission),
        "id_column": str(id_column),
        "prediction_kind": prediction_kind,
        "score_field_preference": list(score_field_preference),
        "probability_row_sum_tolerance": float(probability_row_sum_tolerance),
        "poll_seconds": float(poll_seconds),
        "poll_timeout_seconds": float(poll_timeout_seconds),
        "credentials_embedded": False,
        "spec_hash": "",
    }
    if not all(
        (spec["task_id"], spec["competition"], spec["metric"], spec["id_column"])
    ):
        raise ValueError("Kaggle evaluator spec has an empty required field")
    spec["spec_hash"] = _payload_hash(spec, "spec_hash")
    output_path = Path(output_path).resolve()
    if output_path.exists():
        existing = _read_object(output_path, "Kaggle evaluator spec")
        _verify_hash(existing, "spec_hash", "Kaggle evaluator spec")
        if existing != spec:
            raise ValueError(
                "Existing Kaggle evaluator spec conflicts with requested spec"
            )
        return existing
    _write_exclusive(output_path, spec)
    return spec


def _candidate_inventory(submission_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(Path(submission_dir).glob("submission_*.csv")):
        rows.append(
            {
                "node_id": path.stem.removeprefix("submission_"),
                "submission": path.name,
                "submission_sha256": sha256_file(path),
            }
        )
    return rows


def _default_command_runner(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout


def _submission_rows(
    kaggle_cli: Path,
    competition: str,
    command_runner: Callable[[Sequence[str]], str],
) -> list[dict[str, str]]:
    output = command_runner(
        [
            str(kaggle_cli),
            "competitions",
            "submissions",
            "-c",
            competition,
            "--csv",
        ]
    )
    return list(csv.DictReader(output.splitlines()))


def _is_complete(status: object) -> bool:
    return str(status or "").upper().endswith("COMPLETE")


def _is_error(status: object) -> bool:
    return str(status or "").upper().endswith("ERROR")


def _finite_score(value: object) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _matching_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    description: str,
    filename: str,
) -> dict[str, Any] | None:
    matches = [
        dict(row)
        for row in rows
        if row.get("description") == description and row.get("fileName") == filename
    ]
    if not matches:
        return None
    complete = [row for row in matches if _is_complete(row.get("status"))]
    candidates = complete or matches
    score_pairs = {
        (row.get("publicScore"), row.get("privateScore")) for row in complete
    }
    if len(score_pairs) > 1:
        raise ValueError("Kaggle returned conflicting scores for one frozen submission")
    return candidates[-1]


def _select_official_score(
    row: Mapping[str, Any], preference: Sequence[str]
) -> tuple[float, str]:
    for field in preference:
        if field not in {"privateScore", "publicScore"}:
            raise ValueError(f"Unsupported Kaggle score field: {field}")
        score = _finite_score(row.get(field))
        if score is not None:
            return score, field
    raise ValueError("Kaggle completed without an available official score")


def _validated_request(
    request_path: Path,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, Path, dict[str, str], float | None]:
    request_path = Path(request_path).resolve(strict=True)
    request = _read_object(request_path, "fixed-holdout evaluation request")
    request_schema = request.get("request_schema")
    if request_schema not in {
        "fixed_holdout_evaluation_request_v3",
        NATIVE_REQUEST_SCHEMA,
    }:
        raise ValueError(
            "Official scoring requires a supported frozen evaluation request"
        )
    _verify_hash(request, "request_hash", "Evaluation request")
    expected = {
        "task_id": spec.get("task_id"),
        "metric": spec.get("metric"),
        "maximize": spec.get("maximize"),
        "selection_policy": "terminal_only",
        "scores_were_visible_during_search": False,
        "selection_frozen_before_terminal_evaluation": True,
        "status": "awaiting_external_evaluator",
    }
    for field, value in expected.items():
        if request.get(field) != value:
            raise ValueError(f"Official request mismatch: {field}")
    if request_schema == NATIVE_REQUEST_SCHEMA:
        native_expected = {
            "provider": "kaggle",
            "competition": spec.get("competition"),
            "official_test_inference_during_candidate_execution": True,
            "selected_model_retrained_after_search": False,
        }
        for field, value in native_expected.items():
            if request.get(field) != value:
                raise ValueError(f"Native official request mismatch: {field}")
        sample_path = Path(str(spec.get("sample_submission") or "")).resolve(
            strict=True
        )
        if request.get("sample_submission_sha256") != sha256_file(sample_path):
            raise ValueError("Native official request/sample submission mismatch")
    submission_dir = Path(str(request.get("submission_dir") or "")).resolve(strict=True)
    inventory = _candidate_inventory(submission_dir)
    if request.get("candidate_inventory") != inventory:
        raise ValueError("Candidate inventory changed after terminal selection freeze")
    if request.get("candidate_set_hash") != _payload_hash(
        {"candidate_inventory": inventory}, "unused"
    ):
        raise ValueError("Candidate-set hash changed after terminal selection freeze")
    selected_node_id = str(request.get("selected_node_id") or "")
    selected = [row for row in inventory if row["node_id"] == selected_node_id]
    if len(selected) != 1:
        raise ValueError(
            "Preselected node is not uniquely present in the candidate set"
        )
    selected_row = selected[0]
    if selected_row["submission"] != request.get("selected_submission") or selected_row[
        "submission_sha256"
    ] != request.get("selected_submission_sha256"):
        raise ValueError(
            "Preselected submission changed after terminal selection freeze"
        )
    journal_path = Path(str(request.get("journal_path") or "")).resolve(strict=True)
    if sha256_file(journal_path) != request.get("journal_sha256"):
        raise ValueError("Journal changed after terminal selection freeze")
    journal = _read_object(journal_path, "journal")
    nodes = {
        str(node.get("id")): node
        for node in journal.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    if selected_node_id not in nodes:
        raise ValueError("Preselected node is absent from the frozen journal")
    if request_schema == NATIVE_REQUEST_SCHEMA:
        receipt = nodes[selected_node_id].get("official_submission_receipt") or {}
        if receipt.get("schema") != NATIVE_RECEIPT_SCHEMA:
            raise ValueError(
                "Native selected node lacks an official submission receipt"
            )
        _verify_hash(receipt, "receipt_hash", "Native official submission receipt")
        receipt_expected = {
            "node_id": selected_node_id,
            "submission_sha256": selected_row["submission_sha256"],
            "candidate_code_sha256": request.get("selected_candidate_code_sha256"),
            "receipt_hash": request.get("selected_submission_receipt_hash"),
            "sample_submission_sha256": request.get("sample_submission_sha256"),
            "row_count": request.get("official_test_row_count"),
            "official_test_id_sha256": request.get("official_test_id_sha256"),
            "generated_during_candidate_execution": True,
            "requires_post_search_retraining": False,
            "official_score_visible": False,
        }
        for field, value in receipt_expected.items():
            if receipt.get(field) != value:
                raise ValueError(f"Native official receipt mismatch: {field}")
    metric = nodes[selected_node_id].get("metric") or {}
    internal_metric = _finite_score(metric.get("value"))
    return (
        request,
        submission_dir / selected_row["submission"],
        journal_path,
        selected_row,
        internal_metric,
    )


def score_kaggle_terminal(
    spec_path: Path,
    request_path: Path,
    output_path: Path,
    *,
    work_dir: Path,
    kaggle_cli: Path = Path("kaggle"),
    poll_seconds: float | None = None,
    poll_timeout_seconds: float | None = None,
    command_runner: Callable[[Sequence[str]], str] = _default_command_runner,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Submit and score exactly the system-preselected terminal submission."""

    spec_path = Path(spec_path).resolve(strict=True)
    spec = _read_object(spec_path, "Kaggle evaluator spec")
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("Unsupported Kaggle terminal evaluator spec")
    _verify_hash(spec, "spec_hash", "Kaggle evaluator spec")
    output_path = Path(output_path).resolve()
    if output_path.exists():
        report = _read_object(output_path, "Kaggle terminal report")
        if report.get("schema") != REPORT_SCHEMA:
            raise ValueError("Existing output is not a Kaggle terminal report")
        _verify_hash(report, "report_hash", "Kaggle terminal report")
        if report.get("evaluator_spec_sha256") != sha256_file(spec_path):
            raise ValueError("Existing official report uses another evaluator spec")
        if report.get("evaluation_request_sha256") != sha256_file(Path(request_path)):
            raise ValueError("Existing official report uses another evaluation request")
        request, selected_path, _, selected, _ = _validated_request(request_path, spec)
        if (
            report.get("evaluation_request_hash") != request.get("request_hash")
            or report.get("selected_node_id") != request.get("selected_node_id")
            or report.get("selected_submission_sha256")
            != selected.get("submission_sha256")
            or sha256_file(selected_path) != selected.get("submission_sha256")
        ):
            raise ValueError("Existing official report no longer matches frozen inputs")
        return report

    (
        request,
        selected_path,
        journal_path,
        selected,
        internal_metric,
    ) = _validated_request(request_path, spec)
    sample_submission = Path(str(spec.get("sample_submission") or "")).resolve(
        strict=True
    )
    pinned_sample_hash = str(spec.get("sample_submission_sha256") or "")
    if pinned_sample_hash and pinned_sample_hash != sha256_file(sample_submission):
        raise ValueError("Kaggle evaluator sample submission hash mismatch")
    prediction_kind = str(spec.get("prediction_kind") or "numeric")
    if prediction_kind == "probability" and bool(
        spec.get("normalize_probabilities", False)
    ):
        prediction_kind = "multiclass_probability"
    validate_official_submission(
        selected_path,
        sample_submission,
        id_column=str(spec.get("id_column") or "id"),
        prediction_kind=prediction_kind,
        probability_row_sum_tolerance=float(
            spec.get("probability_row_sum_tolerance", 1e-4)
        ),
    )
    submission_sha256 = sha256_file(selected_path)
    competition = str(spec.get("competition") or "")
    if not competition:
        raise ValueError("Kaggle evaluator spec lacks competition")
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    filename = f"mlevolve_terminal_{submission_sha256[:16]}.csv"
    staged_submission = work_dir / filename
    if staged_submission.exists():
        if sha256_file(staged_submission) != submission_sha256:
            raise ValueError("Staged Kaggle submission hash mismatch")
    else:
        shutil.copy2(selected_path, staged_submission)
    description = f"mlevolve-terminal-v1 sha256={submission_sha256}"
    rows = _submission_rows(kaggle_cli, competition, command_runner)
    row = _matching_row(rows, description=description, filename=filename)
    if row is None:
        command_runner(
            [
                str(kaggle_cli),
                "competitions",
                "submit",
                "-c",
                competition,
                "-f",
                str(staged_submission),
                "-m",
                description,
            ]
        )
    timeout = float(
        poll_timeout_seconds
        if poll_timeout_seconds is not None
        else spec.get("poll_timeout_seconds", 1800)
    )
    interval = float(
        poll_seconds if poll_seconds is not None else spec.get("poll_seconds", 15)
    )
    deadline = time.monotonic() + timeout
    while row is None or not _is_complete(row.get("status")):
        if row is not None and _is_error(row.get("status")):
            raise RuntimeError("Kaggle rejected the frozen official submission")
        if time.monotonic() >= deadline:
            raise TimeoutError("Kaggle official scoring did not complete in time")
        sleep(interval)
        rows = _submission_rows(kaggle_cli, competition, command_runner)
        row = _matching_row(rows, description=description, filename=filename)
    score, score_field = _select_official_score(
        row,
        list(spec.get("score_field_preference") or ["privateScore", "publicScore"]),
    )
    public_score = _finite_score(row.get("publicScore"))
    private_score = _finite_score(row.get("privateScore"))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "provider": "kaggle",
        "competition": competition,
        "task_id": request["task_id"],
        "metric": request["metric"],
        "maximize": request["maximize"],
        "selection_policy": "terminal_only",
        "evaluation_request_schema": request.get("request_schema"),
        "scores_were_visible_during_search": False,
        "selection_frozen_before_terminal_evaluation": True,
        "system_selection_used_official_score": False,
        "official_test_inference_during_candidate_execution": bool(
            request.get("official_test_inference_during_candidate_execution", False)
        ),
        "selected_model_retrained_after_search": bool(
            request.get("selected_model_retrained_after_search", False)
        ),
        "candidate_set_hash": request["candidate_set_hash"],
        "evaluation_request_hash": request["request_hash"],
        "evaluation_request_sha256": sha256_file(Path(request_path)),
        "journal_sha256": sha256_file(journal_path),
        "evaluator_spec_sha256": sha256_file(spec_path),
        "selected_node_id": request["selected_node_id"],
        "selected_submission": request["selected_submission"],
        "selected_submission_sha256": submission_sha256,
        "internal_search_metric": internal_metric,
        "internal_metric_disposition": "search_only_diagnostic",
        "official_public_score": public_score,
        "official_private_score": private_score,
        "selected_score": score,
        "selected_score_source": score_field,
        "official_submission": {
            "file_name": str(row.get("fileName") or filename),
            "description": description,
            "reference": str(row.get("ref") or ""),
            "status": str(row.get("status") or ""),
        },
        "terminal_score_sealed": True,
        "memory_admission": {
            "official_score_required": True,
            "official_score_sufficient": False,
            "status": "pending_safety_and_leakage_audit",
        },
        "report_hash": "",
    }
    report["report_hash"] = _payload_hash(report, "report_hash")
    _write_exclusive(output_path, report)
    return report


def write_official_measurement(
    base_measurement_path: Path,
    report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write an immutable overlay; never rewrite the original measurement."""

    base_measurement_path = Path(base_measurement_path).resolve(strict=True)
    report_path = Path(report_path).resolve(strict=True)
    base = _read_object(base_measurement_path, "base measurement")
    report = _read_object(report_path, "Kaggle terminal report")
    _verify_hash(base, "measurement_hash", "Base measurement")
    _verify_hash(report, "report_hash", "Kaggle terminal report")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("Official measurement requires a Kaggle terminal report")
    if report.get("task_id") != base.get("task_id"):
        raise ValueError("Official report/base measurement task mismatch")
    selected = str(base.get("selected_candidate_id") or "")
    if not selected or selected != report.get("selected_node_id"):
        raise ValueError("Official report selected another candidate")
    base_candidate_set_hash = str(base.get("candidate_set_hash") or "")
    if base_candidate_set_hash and base_candidate_set_hash != report.get(
        "candidate_set_hash"
    ):
        raise ValueError("Official report/base measurement candidate set mismatch")
    official_score_value = _finite_score(report.get("selected_score"))
    if official_score_value is None:
        raise ValueError("Official report lacks a finite selected score")
    official_score = official_score_value
    internal_score = (
        base.get("internal_search_metric")
        if base.get("internal_search_metric") is not None
        else base.get("terminal_score")
    )
    internal_value = _finite_score(internal_score)
    overlay: dict[str, Any] = {
        "schema": MEASUREMENT_SCHEMA,
        "logical_run_id": base.get("logical_run_id"),
        "attempt": base.get("attempt"),
        "task_id": base.get("task_id"),
        "system_id": base.get("system_id"),
        "seed": base.get("seed"),
        "base_completed_before_official_score": base.get("completed") is True,
        "base_measurement_sha256": sha256_file(base_measurement_path),
        "base_measurement_hash": base.get("measurement_hash"),
        "official_report_sha256": sha256_file(report_path),
        "official_report_hash": report.get("report_hash"),
        "internal_terminal_score": internal_value,
        "internal_metric_disposition": "diagnostic_only",
        "official_score": official_score,
        "official_public_score": report.get("official_public_score"),
        "official_private_score": report.get("official_private_score"),
        "official_metric": report.get("metric"),
        "official_score_source": report.get("selected_score_source"),
        "primary_score": official_score,
        "primary_score_authority": "official_kaggle_terminal",
        "selected_candidate_id": report.get("selected_node_id"),
        "candidate_set_hash": report.get("candidate_set_hash"),
        "official_test_inference_during_candidate_execution": report.get(
            "official_test_inference_during_candidate_execution"
        ),
        "selected_model_retrained_after_search": report.get(
            "selected_model_retrained_after_search"
        ),
        "score_gap_official_minus_internal": (
            official_score - internal_value if internal_value is not None else None
        ),
        "memory_admission": report.get("memory_admission"),
        "official_measurement_hash": "",
    }
    overlay["official_measurement_hash"] = _payload_hash(
        overlay, "official_measurement_hash"
    )
    output_path = Path(output_path).resolve()
    if output_path.exists():
        existing = _read_object(output_path, "official measurement")
        _verify_hash(existing, "official_measurement_hash", "Official measurement")
        if existing != overlay:
            raise ValueError("Existing official measurement conflicts with new overlay")
        return existing
    _write_exclusive(output_path, overlay)
    return overlay


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--spec", type=Path, required=True)
    score.add_argument("--request", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--work-dir", type=Path, required=True)
    score.add_argument("--kaggle-cli", type=Path, default=Path("kaggle"))
    measurement = subparsers.add_parser("measurement")
    measurement.add_argument("--base", type=Path, required=True)
    measurement.add_argument("--report", type=Path, required=True)
    measurement.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--spec", type=Path, required=True)
    finalize.add_argument("--request", type=Path, required=True)
    finalize.add_argument("--base", type=Path, required=True)
    finalize.add_argument("--report", type=Path, required=True)
    finalize.add_argument("--measurement", type=Path, required=True)
    finalize.add_argument("--work-dir", type=Path, required=True)
    finalize.add_argument("--kaggle-cli", type=Path, default=Path("kaggle"))
    spec_parser = subparsers.add_parser("spec")
    spec_parser.add_argument("--task-id", required=True)
    spec_parser.add_argument("--competition", required=True)
    spec_parser.add_argument("--metric", required=True)
    direction = spec_parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--maximize", action="store_true")
    direction.add_argument("--minimize", action="store_true")
    spec_parser.add_argument("--sample-submission", type=Path, required=True)
    spec_parser.add_argument("--id-column", required=True)
    spec_parser.add_argument(
        "--prediction-kind",
        choices=("numeric", "probability", "multiclass_probability"),
        required=True,
    )
    spec_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "score":
        result = score_kaggle_terminal(
            args.spec,
            args.request,
            args.output,
            work_dir=args.work_dir,
            kaggle_cli=args.kaggle_cli,
        )
    elif args.command == "measurement":
        result = write_official_measurement(args.base, args.report, args.output)
    elif args.command == "finalize":
        report = score_kaggle_terminal(
            args.spec,
            args.request,
            args.report,
            work_dir=args.work_dir,
            kaggle_cli=args.kaggle_cli,
        )
        result = write_official_measurement(
            args.base,
            args.report,
            args.measurement,
        )
        result = {"report": report, "measurement": result}
    else:
        result = write_evaluator_spec(
            args.output,
            task_id=args.task_id,
            competition=args.competition,
            metric=args.metric,
            maximize=args.maximize,
            sample_submission=args.sample_submission,
            id_column=args.id_column,
            prediction_kind=args.prediction_kind,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
