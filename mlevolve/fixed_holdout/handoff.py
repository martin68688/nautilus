"""Write the label-free request consumed by a terminal evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fixed_holdout.common import read_manifest, sha256_file, write_json
from fixed_holdout.mode import (
    INTERNAL_METRIC_DISPOSITION,
    enabled,
    search_only_candidate_selection,
    train_manifest_path,
)


def write_evaluation_request(
    cfg: Any,
    journal_path: Path,
    *,
    authority: Any | None = None,
    selected_node_id: str | None = None,
    selection_basis: dict[str, Any] | None = None,
) -> Path | None:
    if not enabled(cfg):
        return None
    if not search_only_candidate_selection(cfg):
        raise ValueError(
            "Fixed-holdout handoff requires terminal-only search-only candidate selection"
        )
    manifest = read_manifest(train_manifest_path(cfg), expected_role="train_view")
    output_path = Path(cfg.log_dir) / "fixed_holdout_evaluation_request.json"
    journal_path = Path(journal_path).resolve()
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    nodes = {
        str(node.get("id")): node
        for node in journal.get("nodes") or []
        if isinstance(node, dict) and node.get("id")
    }
    selected_node_id = str(selected_node_id or "")
    if not selected_node_id or selected_node_id not in nodes:
        raise ValueError(
            "Fixed-holdout handoff requires a known selected_node_id before "
            "terminal evaluation"
        )
    submission_dir = (Path(cfg.workspace_dir) / "submission").resolve()
    candidate_inventory = []
    for submission_path in sorted(submission_dir.glob("submission_*.csv")):
        node_id = submission_path.stem.removeprefix("submission_")
        candidate_inventory.append(
            {
                "node_id": node_id,
                "submission": submission_path.name,
                "submission_sha256": sha256_file(submission_path),
            }
        )
    selected_rows = [
        row for row in candidate_inventory if row["node_id"] == selected_node_id
    ]
    if len(selected_rows) != 1:
        raise ValueError(
            "Selected fixed-holdout node must have exactly one frozen submission"
        )
    selected = selected_rows[0]
    candidate_set_hash = hashlib.sha256(
        json.dumps(
            {"candidate_inventory": candidate_inventory},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    node = nodes[selected_node_id]
    metric = node.get("metric") or {}
    frozen_selection_basis = dict(
        selection_basis
        or {
            "type": "solver_internal_search_metric",
            "metric_value": metric.get("value"),
            "metric_maximize": metric.get("maximize"),
            "stage": node.get("stage"),
            "draft_role": node.get("draft_role"),
        }
    )
    required_basis = {
        "type": "solver_internal_search_metric",
        "metric_disposition": INTERNAL_METRIC_DISPOSITION,
        "terminal_metric_observed": False,
        "formal_rank_claim_authorized": False,
        "source_score_inherited": False,
    }
    for key, value in required_basis.items():
        if key in frozen_selection_basis and frozen_selection_basis[key] != value:
            raise ValueError(f"Incompatible fixed-holdout selection basis: {key}")
        frozen_selection_basis[key] = value
    descriptor = (
        authority.terminal_writeback_descriptor()
        if authority is not None
        and callable(getattr(authority, "terminal_writeback_descriptor", None))
        else {
            "schema": "fixed_holdout_authority_writeback_descriptor_v1",
            "status": "writeback_incomplete",
            "reason": "authority_descriptor_unavailable",
        }
    )
    payload = {
        "schema": manifest["schema"],
        "request_schema": "fixed_holdout_evaluation_request_v3",
        "task_id": manifest["task_id"],
        "split_id": manifest["split_id"],
        "metric": manifest["metric"],
        "maximize": manifest["maximize"],
        "selection_policy": "terminal_only",
        "scores_were_visible_during_search": False,
        "journal_path": str(journal_path),
        "journal_sha256": sha256_file(journal_path),
        "submission_dir": str(submission_dir),
        "candidate_inventory": candidate_inventory,
        "candidate_set_hash": candidate_set_hash,
        "selected_node_id": selected_node_id,
        "selected_submission": selected["submission"],
        "selected_submission_sha256": selected["submission_sha256"],
        "selection_basis": frozen_selection_basis,
        "selection_frozen_before_terminal_evaluation": True,
        "train_manifest_sha256": sha256_file(train_manifest_path(cfg)),
        "authority_writeback": descriptor,
        "status": "awaiting_external_evaluator",
        "request_hash": "",
    }
    payload["request_hash"] = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in payload.items()
                if key != "request_hash"
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    write_json(output_path, payload)
    return output_path
