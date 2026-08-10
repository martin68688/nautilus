#!/usr/bin/env python3
"""Finalize one End2End condition with its frozen official Kaggle score.

Run this as a CPU-only Indexed Job after the corresponding training condition
has reached ``awaiting_official_terminal_score``.  It never executes candidate
code and therefore never retrains or reruns official-test inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MLEVOLVE_ROOT = REPO / "mlevolve"
if str(MLEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE_ROOT))

from fixed_holdout.kaggle_terminal import (  # noqa: E402
    score_kaggle_terminal,
    write_official_measurement,
)
from run_assignment import (  # noqa: E402
    load_frozen_inputs,
    read_object,
    select_row,
    verify_evaluator_release,
)


def _attempts(condition_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in condition_root.glob("attempt-*")
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: int(path.name.removeprefix("attempt-")),
    )


def run(
    *,
    manifest_path: Path,
    index: int,
    task_id: str | None,
    output_root: Path,
    kaggle_cli: Path,
) -> dict[str, Any]:
    manifest = load_frozen_inputs(Path(manifest_path).resolve(strict=True))
    row = select_row(manifest, index=index, task_id=task_id)
    condition_root = Path(output_root).resolve() / str(row["logical_run_id"])
    attempts = _attempts(condition_root)
    if not attempts:
        raise ValueError(
            f"Official scorer found no retained condition: {condition_root}"
        )
    attempt_root = attempts[-1]
    base_path = attempt_root / "MEASUREMENT.json"
    base = read_object(base_path)
    official_path = attempt_root / "OFFICIAL_MEASUREMENT.json"
    report_path = attempt_root / "OFFICIAL_SCORE_REPORT.json"
    if official_path.is_file():
        # The measurement writer revalidates both immutable inputs and refuses
        # conflicting replay, so this is a safe idempotent fast path.
        result = write_official_measurement(base_path, report_path, official_path)
        return {"status": "already_scored", "measurement": result}
    if base.get("status") != "awaiting_official_terminal_score":
        raise ValueError(
            "Condition is not ready for official scoring: " f"{base.get('status')!r}"
        )
    if (
        base.get("failure_class") != "none"
        or base.get("candidate_set_frozen") is not True
    ):
        raise ValueError("Condition lacks a clean frozen official candidate set")
    journal_path = Path(str(base.get("journal_path") or "")).resolve(strict=True)
    request_path = journal_path.with_name("official_evaluation_request.json")
    if not request_path.is_file() or request_path.is_symlink():
        raise ValueError("Condition lacks its immutable official evaluation request")
    evaluator = verify_evaluator_release(
        str(row["task_id"]), manifest["_components"]["evaluators"]
    )
    spec_path = evaluator["paths"].get("official_kaggle_evaluator_spec")
    if spec_path is None:
        raise ValueError("Frozen task release lacks official_kaggle_evaluator_spec")
    report = score_kaggle_terminal(
        spec_path,
        request_path,
        report_path,
        work_dir=attempt_root / "official-kaggle",
        kaggle_cli=kaggle_cli,
    )
    measurement = write_official_measurement(
        base_path,
        report_path,
        official_path,
    )
    return {
        "status": "scored_official_terminal_result",
        "logical_run_id": row["logical_run_id"],
        "selected_candidate_id": report["selected_node_id"],
        "official_score": report["selected_score"],
        "official_score_source": report["selected_score_source"],
        "measurement": measurement,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--task", default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--kaggle-cli", type=Path, default=Path("kaggle"))
    args = parser.parse_args()
    if args.index is None:
        raw = os.environ.get("JOB_COMPLETION_INDEX")
        if raw is None:
            parser.error("--index or JOB_COMPLETION_INDEX is required")
        args.index = int(raw)
    result = run(
        manifest_path=args.manifest,
        index=args.index,
        task_id=args.task,
        output_root=args.output_root,
        kaggle_cli=args.kaggle_cli,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
