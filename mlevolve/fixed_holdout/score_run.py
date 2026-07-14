"""Score every node submission after a run without feeding scores back."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from fixed_holdout.common import read_manifest, write_json
from fixed_holdout.evaluate import evaluate_submission


SUBMISSION_RE = re.compile(r"^submission_(.+)\.csv$")


def _journal_nodes(journal_path: Path | None) -> dict[str, dict]:
    if journal_path is None:
        return {}
    journal = json.loads(Path(journal_path).read_text(encoding="utf-8"))
    return {
        str(node.get("id")): node
        for node in journal.get("nodes", [])
        if node.get("id")
    }


def score_run(
    manifest_path: Path,
    submission_dir: Path,
    output_path: Path,
    *,
    journal_path: Path | None = None,
) -> dict:
    manifest = read_manifest(Path(manifest_path), expected_role="evaluator_view")
    nodes = _journal_nodes(journal_path)
    results = []
    for submission_path in sorted(Path(submission_dir).glob("submission_*.csv")):
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
    scored.sort(key=lambda item: item["score"], reverse=bool(manifest["maximize"]))
    report = {
        "schema": manifest["schema"],
        "task_id": manifest["task_id"],
        "split_id": manifest["split_id"],
        "metric": manifest["metric"],
        "maximize": manifest["maximize"],
        "selection_policy": "terminal_only",
        "scores_were_visible_during_search": False,
        "best_node_id": scored[0]["node_id"] if scored else None,
        "best_score": scored[0]["score"] if scored else None,
        "results": results,
    }
    write_json(Path(output_path), report)

    if scored and scored[0]["node_id"] in nodes:
        best_code = nodes[scored[0]["node_id"]].get("code")
        if best_code:
            Path(output_path).with_name("best_solution_fixed_holdout.py").write_text(
                str(best_code),
                encoding="utf-8",
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path)
    args = parser.parse_args()
    report = score_run(
        args.manifest,
        args.submission_dir,
        args.output,
        journal_path=args.journal,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
