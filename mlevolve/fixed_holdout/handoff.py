"""Write the label-free request consumed by a terminal evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fixed_holdout.common import read_manifest, write_json
from fixed_holdout.mode import enabled, train_manifest_path


def write_evaluation_request(cfg: Any, journal_path: Path) -> Path | None:
    if not enabled(cfg):
        return None
    manifest = read_manifest(train_manifest_path(cfg), expected_role="train_view")
    output_path = Path(cfg.log_dir) / "fixed_holdout_evaluation_request.json"
    write_json(
        output_path,
        {
            "schema": manifest["schema"],
            "task_id": manifest["task_id"],
            "split_id": manifest["split_id"],
            "metric": manifest["metric"],
            "maximize": manifest["maximize"],
            "selection_policy": "terminal_only",
            "scores_were_visible_during_search": False,
            "journal_path": str(Path(journal_path)),
            "submission_dir": str(Path(cfg.workspace_dir) / "submission"),
            "status": "awaiting_external_evaluator",
        },
    )
    return output_path
