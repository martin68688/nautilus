"""Native official-test submission contract for mlevolve candidates.

The contract is provider-neutral.  Candidate code trains once, computes its
internal search metric, and writes predictions for the task's complete frozen
official test set during that same execution.  The Host validates those
predictions against the task's sample submission and freezes the selected
candidate before any external score is visible.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

from fixed_holdout.common import sha256_file


REQUEST_SCHEMA = "mlevolve_official_evaluation_request_v1"
RECEIPT_SCHEMA = "mlevolve_official_submission_receipt_v1"


def _section(cfg: Any) -> Any:
    return getattr(cfg, "official_submission", None)


def enabled(cfg: Any) -> bool:
    return bool(getattr(_section(cfg), "enabled", False))


def validate_runtime_config(cfg: Any) -> dict[str, Any] | None:
    """Fail before search if the native official-test contract is ambiguous."""

    if not enabled(cfg):
        return None
    if bool(getattr(getattr(cfg, "fixed_holdout", None), "enabled", False)):
        raise ValueError("official_submission and fixed_holdout are mutually exclusive")
    section = _section(cfg)
    provider = str(getattr(section, "provider", "kaggle") or "kaggle")
    competition = str(getattr(section, "competition", "") or "")
    metric = str(getattr(section, "metric", "") or "")
    if provider != "kaggle" or not competition or not metric:
        raise ValueError(
            "official_submission requires provider=kaggle, competition, and metric"
        )
    sample = _resolve_sample_submission(cfg)
    with sample.open(newline="", encoding="utf-8-sig") as handle:
        columns = list(csv.DictReader(handle).fieldnames or [])
    id_column = str(
        getattr(section, "id_column", "") or (columns[0] if columns else "")
    )
    if not columns or id_column not in columns or len(columns) < 2:
        raise ValueError("Frozen official sample submission has an invalid schema")
    prediction_columns = [column for column in columns if column != id_column]
    kind = _prediction_kind(section, prediction_columns)
    if kind not in {"numeric", "probability", "multiclass_probability"}:
        raise ValueError(f"Unsupported official prediction kind: {kind}")
    _submission_dir(cfg)
    return {
        "sample_submission": str(sample),
        "sample_submission_sha256": sha256_file(sample),
        "id_column": id_column,
        "prediction_columns": prediction_columns,
        "prediction_kind": kind,
    }


def _canonical_hash(payload: Mapping[str, Any], field: str) -> str:
    unsigned = {key: value for key, value in payload.items() if key != field}
    return hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _resolve_sample_submission(cfg: Any) -> Path:
    section = _section(cfg)
    raw = str(getattr(section, "sample_submission_path", "") or "")
    roots = [
        Path(str(getattr(cfg, "data_dir", "") or "")).expanduser(),
        Path(str(getattr(cfg, "workspace_dir", "") or "")).expanduser() / "input",
    ]
    if raw:
        path = Path(raw).expanduser()
        candidates = [path] if path.is_absolute() else [root / path for root in roots]
        candidates.append(path)
        existing = {
            candidate.resolve() for candidate in candidates if candidate.is_file()
        }
    else:
        existing = {
            candidate.resolve()
            for root in roots
            if root.is_dir()
            for candidate in root.rglob("*sample*submission*.csv")
            if candidate.is_file()
        }
    if len(existing) != 1:
        raise ValueError(
            "official_submission requires exactly one frozen sample submission; "
            f"found {sorted(str(path) for path in existing)}"
        )
    return next(iter(existing))


def _submission_dir(cfg: Any) -> Path:
    subdir = str(
        getattr(_section(cfg), "submission_subdir", "submission") or "submission"
    )
    path = Path(subdir)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "official_submission.submission_subdir must be a safe relative path"
        )
    return Path(cfg.workspace_dir).resolve() / path


def _prediction_kind(section: Any, prediction_columns: list[str]) -> str:
    configured = str(getattr(section, "prediction_kind", "auto") or "auto")
    if configured != "auto":
        return configured
    metric = str(getattr(section, "metric", "") or "").lower()
    if len(prediction_columns) > 1:
        return "multiclass_probability"
    if any(token in metric for token in ("auc", "log_loss", "logloss")):
        return "probability"
    return "numeric"


def validate_submission(
    submission_path: Path,
    sample_submission_path: Path,
    *,
    id_column: str = "",
    prediction_kind: str = "auto",
    probability_row_sum_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Stream-validate one submission against a task-specific frozen sample."""

    submission_path = Path(submission_path)
    if submission_path.is_symlink():
        raise ValueError("Official submission may not be a symlink")
    submission_path = submission_path.resolve(strict=True)
    sample_submission_path = Path(sample_submission_path).resolve(strict=True)
    id_digest = hashlib.sha256()
    row_count = 0
    probability_min: float | None = None
    probability_max: float | None = None
    with sample_submission_path.open(
        newline="", encoding="utf-8-sig"
    ) as sample_handle, submission_path.open(
        newline="", encoding="utf-8-sig"
    ) as prediction_handle:
        sample_reader = csv.DictReader(sample_handle)
        prediction_reader = csv.DictReader(prediction_handle)
        sample_columns = list(sample_reader.fieldnames or [])
        submitted_columns = list(prediction_reader.fieldnames or [])
        if not sample_columns or submitted_columns != sample_columns:
            raise ValueError(
                "Official submission columns/order differ from sample submission"
            )
        resolved_id = id_column or sample_columns[0]
        if resolved_id not in sample_columns:
            raise ValueError(f"Official submission lacks ID column {resolved_id!r}")
        prediction_columns = [
            column for column in sample_columns if column != resolved_id
        ]
        if not prediction_columns:
            raise ValueError("Official sample submission has no prediction columns")
        kind = prediction_kind
        if kind == "auto":
            kind = (
                "multiclass_probability" if len(prediction_columns) > 1 else "numeric"
            )
        if kind not in {"numeric", "probability", "multiclass_probability"}:
            raise ValueError(f"Unsupported official prediction kind: {kind}")
        sentinel = object()
        for sample_row, prediction_row in itertools.zip_longest(
            sample_reader, prediction_reader, fillvalue=sentinel
        ):
            if sample_row is sentinel or prediction_row is sentinel:
                raise ValueError(
                    "Official submission row count differs from sample submission"
                )
            sample_id = str(sample_row[resolved_id])
            prediction_id = str(prediction_row[resolved_id])
            if prediction_id != sample_id:
                raise ValueError(
                    "Official submission IDs or order differ from sample submission"
                )
            id_digest.update(sample_id.encode("utf-8"))
            id_digest.update(b"\n")
            values: list[float] = []
            for column in prediction_columns:
                try:
                    value = float(prediction_row[column])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Official submission contains a nonnumeric prediction in {column!r}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        "Official submission contains a non-finite prediction"
                    )
                values.append(value)
                probability_min = (
                    value if probability_min is None else min(probability_min, value)
                )
                probability_max = (
                    value if probability_max is None else max(probability_max, value)
                )
            if kind in {"probability", "multiclass_probability"} and any(
                value < 0.0 or value > 1.0 for value in values
            ):
                raise ValueError("Official probability submission is outside [0, 1]")
            if kind == "multiclass_probability" and not math.isclose(
                sum(values),
                1.0,
                rel_tol=0.0,
                abs_tol=float(probability_row_sum_tolerance),
            ):
                raise ValueError(
                    "Official multiclass probability row does not sum to one"
                )
            row_count += 1
    if row_count == 0:
        raise ValueError("Official submission contains no prediction rows")
    return {
        "row_count": row_count,
        "columns": sample_columns,
        "id_column": resolved_id,
        "prediction_columns": prediction_columns,
        "prediction_kind": kind,
        "official_test_id_sha256": id_digest.hexdigest(),
        "sample_submission_sha256": sha256_file(sample_submission_path),
        "submission_sha256": sha256_file(submission_path),
        "probability_min": probability_min if kind != "numeric" else None,
        "probability_max": probability_max if kind != "numeric" else None,
    }


def validate_candidate_submission(cfg: Any, node: Any) -> dict[str, Any] | None:
    """Validate and journal-bind the native official output of one candidate."""

    if not enabled(cfg):
        return None
    section = _section(cfg)
    sample_path = _resolve_sample_submission(cfg)
    submission_path = _submission_dir(cfg) / f"submission_{node.id}.csv"
    with sample_path.open(newline="", encoding="utf-8-sig") as handle:
        columns = list(csv.DictReader(handle).fieldnames or [])
    id_column = str(
        getattr(section, "id_column", "") or (columns[0] if columns else "")
    )
    prediction_columns = [column for column in columns if column != id_column]
    details = validate_submission(
        submission_path,
        sample_path,
        id_column=id_column,
        prediction_kind=_prediction_kind(section, prediction_columns),
        probability_row_sum_tolerance=float(
            getattr(section, "probability_row_sum_tolerance", 1e-4)
        ),
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "task_id": str(getattr(cfg, "exp_id", "") or ""),
        "competition": str(getattr(section, "competition", "") or ""),
        "node_id": str(node.id),
        "candidate_code_sha256": hashlib.sha256(
            str(node.code).encode("utf-8")
        ).hexdigest(),
        "submission_path": str(submission_path.resolve()),
        **details,
        "generated_during_candidate_execution": True,
        "requires_post_search_retraining": False,
        "official_score_visible": False,
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = _canonical_hash(receipt, "receipt_hash")
    receipt_root = Path(cfg.workspace_dir).resolve() / "official_submission_receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_root / f"receipt_{node.id}.json"
    content = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if receipt_path.exists():
        if receipt_path.read_text(encoding="utf-8") != content:
            raise ValueError(
                "Official candidate receipt conflicts with an existing receipt"
            )
    else:
        receipt_path.write_text(content, encoding="utf-8")
        receipt_path.chmod(receipt_path.stat().st_mode & ~0o222)
    return receipt


def _candidate_inventory(submission_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "node_id": path.stem.removeprefix("submission_"),
            "submission": path.name,
            "submission_sha256": sha256_file(path),
        }
        for path in sorted(submission_dir.glob("submission_*.csv"))
        if path.is_file() and not path.is_symlink()
    ]


def write_evaluation_request(
    cfg: Any,
    journal_path: Path,
    *,
    selected_node_id: str,
    selection_basis: Mapping[str, Any],
) -> Path | None:
    """Freeze a natively generated official submission after search stops."""

    if not enabled(cfg):
        return None
    if bool(getattr(getattr(cfg, "fixed_holdout", None), "enabled", False)):
        raise ValueError(
            "Native official_submission mode and fixed_holdout cannot share one "
            "candidate submission directory"
        )
    section = _section(cfg)
    journal_path = Path(journal_path).resolve(strict=True)
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    nodes = {
        str(node.get("id")): node
        for node in journal.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    selected_node_id = str(selected_node_id or "")
    if selected_node_id not in nodes:
        raise ValueError("Official terminal selection references an unknown node")
    node = nodes[selected_node_id]
    receipt = dict(node.get("official_submission_receipt") or {})
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("Selected node lacks its native official submission receipt")
    if receipt.get("receipt_hash") != _canonical_hash(receipt, "receipt_hash"):
        raise ValueError("Selected node official submission receipt hash mismatch")
    submission_dir = _submission_dir(cfg)
    inventory = _candidate_inventory(submission_dir)
    selected = [row for row in inventory if row["node_id"] == selected_node_id]
    if len(selected) != 1 or selected[0]["submission_sha256"] != receipt.get(
        "submission_sha256"
    ):
        raise ValueError("Selected official submission is absent or changed")
    sample_path = _resolve_sample_submission(cfg)
    frozen_basis = dict(selection_basis)
    required_basis = {
        "type": "solver_internal_search_metric",
        "metric_disposition": "search_only",
        "official_score_observed": False,
        "selection_frozen_before_official_evaluation": True,
    }
    for key, value in required_basis.items():
        if key in frozen_basis and frozen_basis[key] != value:
            raise ValueError(f"Incompatible official selection basis: {key}")
        frozen_basis[key] = value
    candidate_set_hash = _canonical_hash({"candidate_inventory": inventory}, "unused")
    payload: dict[str, Any] = {
        "request_schema": REQUEST_SCHEMA,
        "provider": str(getattr(section, "provider", "kaggle") or "kaggle"),
        "task_id": str(getattr(cfg, "exp_id", "") or ""),
        "competition": str(getattr(section, "competition", "") or ""),
        "metric": str(getattr(section, "metric", "") or ""),
        "maximize": bool(getattr(section, "maximize", False)),
        "selection_policy": "terminal_only",
        "scores_were_visible_during_search": False,
        "selection_frozen_before_terminal_evaluation": True,
        "official_test_inference_during_candidate_execution": True,
        "selected_model_retrained_after_search": False,
        "journal_path": str(journal_path),
        "journal_sha256": sha256_file(journal_path),
        "submission_dir": str(submission_dir),
        "candidate_inventory": inventory,
        "candidate_set_hash": candidate_set_hash,
        "selected_node_id": selected_node_id,
        "selected_submission": selected[0]["submission"],
        "selected_submission_sha256": selected[0]["submission_sha256"],
        "selected_candidate_code_sha256": receipt["candidate_code_sha256"],
        "selected_submission_receipt_hash": receipt["receipt_hash"],
        "sample_submission_path": str(sample_path),
        "sample_submission_sha256": sha256_file(sample_path),
        "official_test_row_count": receipt["row_count"],
        "official_test_id_sha256": receipt["official_test_id_sha256"],
        "selection_basis": frozen_basis,
        "status": "awaiting_external_evaluator",
        "request_hash": "",
    }
    if not payload["competition"] or not payload["metric"]:
        raise ValueError("Native official submission requires competition and metric")
    payload["request_hash"] = _canonical_hash(payload, "request_hash")
    output_path = Path(cfg.log_dir).resolve() / "official_evaluation_request.json"
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != content:
            raise ValueError(
                "Refusing to replace immutable official evaluation request"
            )
    else:
        output_path.write_text(content, encoding="utf-8")
        output_path.chmod(output_path.stat().st_mode & ~0o222)
    return output_path


__all__ = [
    "RECEIPT_SCHEMA",
    "REQUEST_SCHEMA",
    "enabled",
    "validate_runtime_config",
    "validate_candidate_submission",
    "validate_submission",
    "write_evaluation_request",
]
