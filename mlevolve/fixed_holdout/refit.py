"""Host refit: retrain the selected program on every labelled era.

During the search the embargo and selection eras are withheld (their labels
are Host-only), so the frozen program's model never saw the most recent data.
After the search, this runner executes the selected node's code unchanged in
a fresh directory whose ``./input`` is ``refit_view/input``: the same public
files, except ``train.parquet`` also contains those eras with labels.  The
lockbox stays excluded.  The refit submission is then scored like any other
frozen submission (``fixed_holdout.evaluate``), never fed back to a search.

    python -m fixed_holdout.refit --split <split_root> --run <run_dir> \\
        --output <run_dir>/refit
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fixed_holdout.common import read_manifest, sha256_file, tree_sha256, write_json


def _selected_node(run_dir: Path, node_id: str | None) -> tuple[str, str]:
    journal = json.loads((run_dir / "logs" / "journal.json").read_text(encoding="utf-8"))
    if node_id is None:
        request = json.loads(
            (run_dir / "logs" / "fixed_holdout_evaluation_request.json").read_text(encoding="utf-8")
        )
        node_id = request["selected_node_id"]
    for node in journal["nodes"]:
        if node["id"] == node_id or node["id"].startswith(node_id):
            if not node.get("code"):
                break
            return node["id"], node["code"]
    raise ValueError(f"Node {node_id} with code not found in journal")


def refit_node(
    split_root: Path,
    run_dir: Path,
    output_dir: Path,
    *,
    node_id: str | None = None,
    python: str = sys.executable,
    timeout: int = 7200,
) -> dict:
    split_root, run_dir, output_dir = (Path(p).resolve() for p in (split_root, run_dir, output_dir))
    manifest = read_manifest(split_root / "refit_view" / "fixed_holdout_manifest.json", "refit_view")
    refit_input = split_root / "refit_view" / manifest["input_subdir"]
    if tree_sha256(refit_input) != manifest["refit_tree_sha256"]:
        raise ValueError("Refit view does not match its manifest")
    full_id, code = _selected_node(run_dir, node_id)

    work = output_dir / full_id
    if work.exists():
        shutil.rmtree(work)
    (work / "submission").mkdir(parents=True)
    (work / "working").mkdir()
    (work / "input").symlink_to(refit_input, target_is_directory=True)
    runfile = work / "refit_runfile.py"
    runfile.write_text(code, encoding="utf-8")

    started = time.time()
    with (work / "refit.log").open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            [python, runfile.name], cwd=work, stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    submissions = sorted(
        (p for p in (work / "submission").glob("*.csv") if not p.name.startswith("selection_predictions")),
        key=lambda p: p.stat().st_mtime,
    )
    if proc.returncode != 0 or not submissions:
        raise RuntimeError(
            f"Refit of {full_id} failed (exit={proc.returncode}, submissions={len(submissions)}); "
            f"see {work / 'refit.log'}"
        )
    target = output_dir / f"submission_{full_id}.csv"
    shutil.copy2(submissions[-1], target)
    receipt = {
        "schema": "mlevolve_fixed_holdout_refit_v1",
        "node_id": full_id,
        "split_id": manifest["split_id"],
        "refit_tree_sha256": manifest["refit_tree_sha256"],
        "code_sha256": sha256_file(runfile),
        "submission": target.name,
        "submission_sha256": sha256_file(target),
        "seconds": round(time.time() - started, 1),
    }
    write_json(output_dir / f"refit_receipt_{full_id}.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--node", help="Node id (default: the frozen selected node)")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    receipt = refit_node(args.split, args.run, args.output, node_id=args.node,
                         python=args.python, timeout=args.timeout)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
