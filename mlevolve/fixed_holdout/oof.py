"""Host out-of-fold check: re-run a node's program on earlier periods.

The selection set is one block of recent eras, so its score is noisy and
reflects a single market regime. Out-of-fold (OOF) folds add more blocks from
the train view: fold k holds out ``fold_eras`` consecutive eras near the end of
the train view and trains only on eras that end ``embargo_eras`` before them
(walk-forward, so no later data and no overlapping targets reach the model).
Each fold is a view like ``train_view``: ``input/`` with a shorter
``train.parquet`` and a ``test.parquet`` of the held-out eras without labels.
Their labels and benchmark predictions stay with the Host (``labels.csv``).

During the search the Host runs the node's code unchanged in each fold view,
scores the fold predictions like the selection rows, and the node's score
becomes the mean over blocks (each fold and the selection set count once).
``window_means`` then lists one mean per block, so the worst window is the
worst period.

    python -m fixed_holdout.oof --split <split_root> \\
        --benchmarks validation_benchmark_models.parquet --folds 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fixed_holdout.common import read_manifest, sha256_file, write_json

LABEL = "target"
BENCHMARK = "benchmark"
OOF_NOTE = (
    "\n\nThe Host also re-runs your program unchanged on earlier periods: there "
    "`./input/train.parquet` holds only eras up to a gap and `./input/test.parquet` "
    "holds the later eras to predict (and `sample_submission.csv` lists their ids). "
    "Do not assume particular eras, dates or row counts; each run must finish within "
    "the time limit."
)


class OofRunError(RuntimeError):
    """A fold re-run did not produce a usable submission."""


def fold_eras(eras: list[str], folds: int, fold_eras: int, embargo_eras: int) -> list[dict]:
    """Walk-forward blocks at the end of ``eras``: most recent fold first."""
    eras = sorted(eras)
    plan = []
    end = len(eras)
    for fold in range(1, folds + 1):
        start = end - fold_eras
        train_end = start - embargo_eras
        if start < 0 or train_end <= 0:
            raise ValueError(f"Not enough eras for {folds} folds of {fold_eras} with embargo {embargo_eras}")
        plan.append({"fold": fold, "train": eras[:train_end], "test": eras[start:end]})
        end = start
    return plan


def _describe(text: str, train: list[str], test: list[str]) -> str:
    text = re.sub(r"Eras \d{4}-\d\d-\d\d to \d{4}-\d\d-\d\d\.", f"Eras {train[0]} to {train[-1]}.", text, count=1)
    return re.sub(
        r"(`test\.parquet`: unlabelled rows to predict, )eras .*?(?=\n  \(|\n- )",
        lambda m: f"{m.group(1)}eras {test[0]} to {test[-1]}",
        text, count=1, flags=re.S,
    )


def build_folds(
    split_root: Path,
    benchmarks_path: Path,
    *,
    folds: int = 2,
    fold_eras_count: int = 52,
    embargo_eras: int = 13,
) -> list[Path]:
    """Write ``<split>/oof_view/fold<k>/`` (input/, labels.csv, manifest)."""
    split_root = Path(split_root).resolve()
    source = split_root / "train_view" / "input"
    selection = read_manifest(split_root / "selection_view" / "fixed_holdout_manifest.json", "selection_view")
    train_file = pq.ParquetFile(source / "train.parquet")
    test_columns = pq.ParquetFile(source / "test.parquet").schema_arrow.names
    eras = pd.read_parquet(source / "train.parquet", columns=["era"])["era"].astype(str).unique().tolist()
    bench = pd.read_parquet(benchmarks_path)
    model_column = [c for c in bench.columns if c not in ("numerai_ticker", "date", "era")][0]
    bench = bench.rename(columns={"date": "era", model_column: BENCHMARK})
    bench["era"] = bench["era"].astype(str)
    bench = bench[["numerai_ticker", "era", BENCHMARK]]
    description = (source / "description.md").read_text(encoding="utf-8")

    out_root = split_root / "oof_view"
    written = []
    for plan in fold_eras(eras, folds, fold_eras_count, embargo_eras):
        fold_dir = out_root / f"fold{plan['fold']}"
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
        inp = fold_dir / "input"
        inp.mkdir(parents=True)
        pq.write_table(
            pq.read_table(source / "train.parquet", filters=[("era", "in", plan["train"])]),
            inp / "train.parquet",
        )
        held = pq.read_table(
            source / "train.parquet", filters=[("era", "in", plan["test"])],
            columns=list(dict.fromkeys(test_columns + [LABEL])),
        ).to_pandas()
        held = held.sort_values(["era", "id"], kind="stable").reset_index(drop=True)
        held[test_columns].to_parquet(inp / "test.parquet", index=False)
        held[["id"]].assign(prediction=0.5).to_csv(inp / "sample_submission.csv", index=False)
        shutil.copy2(source / "features.json", inp / "features.json")
        if (source / "neutralizer").exists():
            (inp / "neutralizer").symlink_to(source / "neutralizer", target_is_directory=True)
        (inp / "description.md").write_text(_describe(description, plan["train"], plan["test"]), encoding="utf-8")

        labels = held[["id", "numerai_ticker", "era", LABEL]].merge(bench, on=["numerai_ticker", "era"], how="left")
        if labels[BENCHMARK].isna().any():
            raise ValueError(f"fold{plan['fold']}: {int(labels[BENCHMARK].isna().sum())} rows lack a benchmark")
        labels = labels[["id", "era", LABEL, BENCHMARK]]
        labels.to_csv(fold_dir / "labels.csv", index=False)
        write_json(fold_dir / "fixed_holdout_manifest.json", {
            "schema": "mlevolve_fixed_holdout_v1",
            "role": "selection_view",
            "task_id": selection["task_id"],
            "split_id": selection["split_id"],
            "selection_policy": "host_oof_fold",
            "metric": "numerai_era_corr",
            "id_column": "id",
            "group_column": "era",
            "label_columns": [LABEL],
            "prediction_columns": ["prediction"],
            "benchmark_column": BENCHMARK,
            "predictions_source": "submission",
            "labels_file": "labels.csv",
            "labels_sha256": sha256_file(fold_dir / "labels.csv"),
            "row_count": int(len(labels)),
            "fold": plan["fold"],
            "train_eras": [plan["train"][0], plan["train"][-1], len(plan["train"])],
            "test_eras": [plan["test"][0], plan["test"][-1], len(plan["test"])],
            "embargo_eras": embargo_eras,
        })
        written.append(fold_dir)
    return written


def fold_dirs(root: Path) -> list[Path]:
    dirs = sorted(Path(root).glob("fold*"), key=lambda p: int(p.name[4:]))
    if not dirs:
        raise ValueError(f"No OOF folds under {root}")
    return dirs


def run_fold(code: str, fold_dir: Path, work: Path, *, python: str, timeout: int) -> Path:
    """Run ``code`` with ``./input`` = the fold's input; return its submission."""
    if work.exists():
        shutil.rmtree(work)
    (work / "submission").mkdir(parents=True)
    (work / "working").mkdir()
    (work / "input").symlink_to((fold_dir / "input").resolve(), target_is_directory=True)
    (work / "oof_runfile.py").write_text(code, encoding="utf-8")
    try:
        with (work / "oof.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(
                [python, "oof_runfile.py"], cwd=work, stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
    except subprocess.TimeoutExpired:
        raise OofRunError(f"{fold_dir.name}: timed out after {timeout} s") from None
    submissions = sorted(
        (p for p in (work / "submission").glob("*.csv") if not p.name.startswith("selection_predictions")),
        key=lambda p: p.stat().st_mtime,
    )
    if proc.returncode != 0 or not submissions:
        tail = (work / "oof.log").read_text(encoding="utf-8", errors="replace")[-1500:]
        raise OofRunError(
            f"{fold_dir.name}: exit {proc.returncode}, {len(submissions)} submission file(s). Log tail:\n{tail}"
        )
    return submissions[-1]


def add_folds(agent, node, result: dict, score_fn, logger) -> dict:
    """Run every fold for ``node`` and fold the scores into ``result`` (in place)."""
    cfg = agent.cfg.oof
    selection_block = float(np.mean(list(result["per_era_score"].values())))
    blocks, per_era, details = [], {}, []
    started = time.time()
    for fold_dir in fold_dirs(Path(cfg.root)):
        work = Path(agent.cfg.workspace_dir) / "oof" / node.id / fold_dir.name
        submission = run_fold(node.code, fold_dir, work, python=str(getattr(cfg, "python", "") or sys.executable),
                              timeout=int(cfg.timeout))
        scored = score_fn(fold_dir / "fixed_holdout_manifest.json", submission)
        series = scored["per_era_score"]
        blocks.append(float(np.mean(list(series.values()))))
        per_era.update(series)
        details.append({
            "fold": fold_dir.name, "score": blocks[-1], "corr_mean": scored["corr_mean"],
            "bmc_mean": scored.get("bmc_mean"), "benchmark_similarity": scored.get("benchmark_similarity"),
            "n_eras": len(series),
        })
    blocks.append(selection_block)
    per_era.update(result["per_era_score"])
    result["oof"] = {"folds": details, "selection_score": result["score"], "seconds": round(time.time() - started, 1)}
    result["per_era_score"] = dict(sorted(per_era.items()))
    result["window_means"] = blocks
    result["window_std"] = float(np.std(blocks, ddof=1)) if len(blocks) > 1 else 0.0
    result["score"] = float(np.mean(blocks))
    logger.info(
        "Node %s OOF blocks=%s (folds then selection) -> score %.5f in %.0f s",
        node.id, [round(b, 4) for b in blocks], result["score"], result["oof"]["seconds"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--benchmarks", type=Path, required=True,
                        help="Host-only benchmark predictions (v6 validation_benchmark_models.parquet)")
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--fold-eras", type=int, default=52)
    parser.add_argument("--embargo-eras", type=int, default=13)
    args = parser.parse_args()
    written = build_folds(args.split, args.benchmarks, folds=args.folds,
                          fold_eras_count=args.fold_eras, embargo_eras=args.embargo_eras)
    for fold_dir in written:
        manifest = json.loads((fold_dir / "fixed_holdout_manifest.json").read_text())
        print(fold_dir, "train", manifest["train_eras"], "test", manifest["test_eras"], "rows", manifest["row_count"])


if __name__ == "__main__":
    main()
