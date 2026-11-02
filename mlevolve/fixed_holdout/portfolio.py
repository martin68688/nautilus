"""Host delivery: an equal-weight portfolio of dissimilar nodes.

After the search, nodes clearing a CORR floor (a fraction of the benchmark's
CORR on the selection eras) are collapsed so that near-identical predictions
count once.  Of those, the subset with the lowest mean pairwise similarity is
kept: past the floor the choice ignores scores, so nothing can be fitted to
them.  If no node clears the floor, the best-scoring working node is delivered
alone.  Each member is refitted and the refit submissions are averaged in
per-era Gaussian rank.  The result is a frozen submission like any other
(``fixed_holdout.evaluate``); it is never fed back to a search.

    python -m fixed_holdout.portfolio --split <split_root> --run <run_dir> \\
        --output <run_dir>/refit
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fixed_holdout.common import read_manifest, sha256_file, write_json
from fixed_holdout.evaluate import _gaussian_rank, numerai_era_scores
from fixed_holdout.refit import refit_node
from fixed_holdout.selection import predictions_path


def _usable_nodes(run_dir: Path) -> list[dict]:
    journal = json.loads((run_dir / "logs" / "journal.json").read_text(encoding="utf-8"))
    return [
        node
        for node in journal["nodes"]
        if not node.get("is_buggy") and "corr_mean" in (node.get("host_selection") or {})
    ]


def _similarity_matrix(ranks: np.ndarray, eras: np.ndarray) -> np.ndarray:
    """Mean per-era correlation between the columns of ``ranks``."""
    total = np.zeros((ranks.shape[1], ranks.shape[1]))
    unique = np.unique(eras)
    for era in unique:
        total += np.corrcoef(ranks[eras == era], rowvar=False)
    return total / len(unique)


def choose_members(scores: list[float], similarity: np.ndarray, size: int, copy_threshold: float) -> list[int]:
    """Indices of the least similar subset, the best of each group of copies kept."""
    kept: list[int] = []
    for index in np.argsort(scores)[::-1]:
        if all(similarity[index, other] < copy_threshold for other in kept):
            kept.append(int(index))
    if len(kept) <= size:
        return kept
    return list(
        min(
            combinations(sorted(kept), size),
            key=lambda subset: np.mean([similarity[a, b] for a, b in combinations(subset, 2)]),
        )
    )


def select_portfolio(
    split_root: Path,
    run_dir: Path,
    *,
    size: int = 5,
    floor_ratio: float = 0.75,
    copy_threshold: float = 0.999,
) -> dict:
    """Choose the members from selection rows only and write ``logs/portfolio.json``."""
    split_root, run_dir = Path(split_root).resolve(), Path(run_dir).resolve()
    manifest_path = split_root / "selection_view" / "fixed_holdout_manifest.json"
    manifest = read_manifest(manifest_path, expected_role="selection_view")
    if not manifest.get("benchmark_column"):
        raise ValueError("Portfolio floor needs Host benchmark predictions")
    labels = pd.read_csv(manifest_path.parent / manifest["labels_file"])
    label_column = manifest["label_columns"][0]
    eras = labels[manifest["group_column"]].astype(str).to_numpy()
    targets = labels[[label_column]].to_numpy(dtype=float)
    benchmark = labels[[manifest["benchmark_column"]]].to_numpy(dtype=float)
    benchmark_corr = numerai_era_scores(targets, benchmark, eras)["corr_mean"]
    floor = floor_ratio * benchmark_corr

    nodes = _usable_nodes(run_dir)
    if not nodes:
        raise ValueError("No scored node to deliver")
    members = [n for n in nodes if n["host_selection"]["corr_mean"] >= floor]
    fallback = not members
    if fallback:
        chosen = [max(nodes, key=lambda n: n["host_selection"]["score"])]
    else:
        id_column, prediction_column = manifest["id_column"], manifest["prediction_columns"][0]
        ranks = np.empty((len(labels), len(members)), dtype=np.float32)
        order = pd.Series(np.arange(len(labels)), index=labels[id_column])
        for column, node in enumerate(members):
            path = predictions_path(run_dir / "workspace", node["id"], manifest_path)
            frame = pd.read_csv(path, usecols=[id_column, prediction_column]).set_index(id_column)
            values = frame[prediction_column].reindex(order.index).to_numpy(dtype=float)
            for era in np.unique(eras):
                ranks[eras == era, column] = _gaussian_rank(values[eras == era])
        similarity = _similarity_matrix(ranks, eras)
        scores = [n["host_selection"]["score"] for n in members]
        chosen = [members[i] for i in choose_members(scores, similarity, size, copy_threshold)]
    result = {
        "schema": "mlevolve_fixed_holdout_portfolio_v1",
        "split_id": manifest["split_id"],
        "benchmark_corr": benchmark_corr,
        "corr_floor": floor,
        "fallback_single_node": fallback,
        "members": [
            {
                "node_id": n["id"],
                "score": n["host_selection"]["score"],
                "corr_mean": n["host_selection"]["corr_mean"],
                "bmc_mean": n["host_selection"].get("bmc_mean"),
                "benchmark_similarity": n["host_selection"].get("benchmark_similarity"),
            }
            for n in chosen
        ],
    }
    write_json(run_dir / "logs" / "portfolio.json", result)
    return result


def combine_submissions(paths: list[Path], eras: pd.Series, output: Path) -> None:
    """Average per-era Gaussian ranks; rows keep the order of the first file."""
    frames = [pd.read_csv(path) for path in paths]
    id_column, prediction_column = frames[0].columns[:2]
    era = frames[0][id_column].map(eras)
    if era.isna().any():
        raise ValueError("Submission has ids outside test.parquet")
    total = np.zeros(len(frames[0]))
    for frame in frames:
        if not frame[id_column].equals(frames[0][id_column]):
            raise ValueError("Member submissions have different rows")
        ranked = frame[prediction_column].groupby(era.to_numpy()).transform(
            lambda values: _gaussian_rank(values.to_numpy())
        )
        total += ranked.to_numpy()
    pd.DataFrame({id_column: frames[0][id_column], prediction_column: total / len(frames)}).to_csv(
        output, index=False
    )


def deliver(split_root: Path, run_dir: Path, output_dir: Path, *, python: str, timeout: int, **choice) -> dict:
    """Choose the members, refit them and write ``submission_portfolio.csv``.

    A member whose refit fails is left out and recorded; finished refits are reused.
    """
    split_root, run_dir, output_dir = (Path(p).resolve() for p in (split_root, run_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    portfolio = select_portfolio(split_root, run_dir, **choice)
    submissions, failed = [], {}
    for member in portfolio["members"]:
        node_id = member["node_id"]
        target = output_dir / f"submission_{node_id}.csv"
        if not (output_dir / f"refit_receipt_{node_id}.json").exists() or not target.exists():
            try:
                refit_node(split_root, run_dir, output_dir, node_id=node_id, python=python, timeout=timeout)
            except Exception as error:
                failed[node_id] = str(error)
                continue
        submissions.append(target)
    if not submissions:
        raise RuntimeError(f"Every portfolio refit failed: {failed}")
    manifest = read_manifest(split_root / "refit_view" / "fixed_holdout_manifest.json", "refit_view")
    test = pd.read_parquet(split_root / "refit_view" / manifest["input_subdir"] / "test.parquet", columns=["id", "era"])
    output = output_dir / "submission_portfolio.csv"
    combine_submissions(submissions, test.set_index("id")["era"].astype(str), output)
    receipt = {
        **portfolio,
        "refit_failed": failed,
        "delivered_nodes": [path.stem.removeprefix("submission_") for path in submissions],
        "submission": output.name,
        "submission_sha256": sha256_file(output),
    }
    write_json(output_dir / "portfolio_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--floor-ratio", type=float, default=0.75)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    receipt = deliver(
        args.split, args.run, args.output, python=args.python, timeout=args.timeout,
        size=args.size, floor_ratio=args.floor_ratio,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
