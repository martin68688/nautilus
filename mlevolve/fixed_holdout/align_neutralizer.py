"""Publish the Numerai neutralizer as arrays in the row order of each public file.

The official files hold 7 million rows keyed by ticker and date, so joining them
costs more memory than most programs have left after loading the features.  The
values do not change: each input folder gets ``neutralizer_train.npy`` and
``neutralizer_test.npy`` under the same names, so a program moves from the train
view to the refit view unedited.  The parquet files move to ``neutralizer_source``
beside the views, and the digest printed here is then passed to the evaluator's
copy with ``strip_benchmark --public-tree-sha256``.

    python -m fixed_holdout.align_neutralizer --split <split_root>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from fixed_holdout.strip_benchmark import _replace_text, strip_benchmark

VIEWS = ("train_view", "refit_view")
KEY_COLUMNS = ("numerai_ticker", "date")
OFFICIAL_TEXT = re.compile(
    r"- `neutralizer/\*\.parquet`:.*?join them on `numerai_ticker` and `date == era`\.", re.S
)
ALIGNED_TEXT = """- `neutralizer_train.npy`, `neutralizer_test.npy`: Numerai's per-stock risk
  exposures, float16 arrays of shape (rows, 300) in the row order of `train.parquet`
  and `test.parquet`. `neutralizer_columns.json` names the 300 columns. They contain
  no labels. Rows with no exposures are NaN. Open them with
  `np.load(path, mmap_mode="r")` to read only the rows you need."""


def _row_index(path: Path) -> pd.MultiIndex:
    table = pq.read_table(path, columns=["numerai_ticker", "era"])
    return pd.MultiIndex.from_arrays(
        [table["numerai_ticker"].to_pandas(), table["era"].to_pandas()]
    )


def _value_columns(parts: list[Path]) -> list[str]:
    columns = [c for c in pq.read_schema(parts[0]).names if c.startswith("neutralizer_column_")]
    for part in parts[1:]:
        other = [c for c in pq.read_schema(part).names if c.startswith("neutralizer_column_")]
        if set(other) != set(columns):
            raise ValueError(f"{part.name} has different neutralizer columns")
    return columns


def align_view(input_dir: Path, parts: list[Path]) -> dict[str, float]:
    """Write the aligned arrays and return the covered share of each file's rows."""
    columns = _value_columns(parts)
    targets = {}
    for name in ("train", "test"):
        index = _row_index(input_dir / f"{name}.parquet")
        values = np.full((len(index), len(columns)), np.nan, dtype=np.float16)
        targets[name] = (index, values, np.zeros(len(index), dtype=bool))
    for part in parts:
        for batch in pq.ParquetFile(part).iter_batches(
            batch_size=200_000, columns=[*KEY_COLUMNS, *columns]
        ):
            frame = batch.to_pandas()
            keys = pd.MultiIndex.from_arrays([frame[c] for c in KEY_COLUMNS])
            exposures = frame[columns].to_numpy(dtype=np.float32)
            for index, values, seen in targets.values():
                rows = index.get_indexer(keys)
                found = rows >= 0
                values[rows[found]] = exposures[found]
                seen[rows[found]] = True
    for name, (_, values, _) in targets.items():
        temporary = input_dir / f".neutralizer_{name}.tmp.npy"
        np.save(temporary, values)
        os.replace(temporary, input_dir / f"neutralizer_{name}.npy")
    _replace_text(input_dir / "neutralizer_columns.json", json.dumps(columns) + "\n")
    return {name: float(seen.mean()) for name, (_, _, seen) in targets.items()}


def align_description(text: str) -> str:
    if ALIGNED_TEXT in text:
        return text
    text, count = OFFICIAL_TEXT.subn(lambda _: ALIGNED_TEXT, text)
    if not count:
        raise ValueError("Description does not describe the neutralizer files")
    return text


def align_split(split_root: Path) -> tuple[str, dict[str, dict[str, float]]]:
    """Return the new public tree digest and the coverage of each view."""
    split_root = Path(split_root).resolve()
    source_root = split_root / "neutralizer_source"
    coverage = {}
    for view in VIEWS:
        input_dir = split_root / view / "input"
        if not input_dir.exists():
            continue
        source = source_root / view
        official = input_dir / "neutralizer"
        if official.exists():
            source_root.mkdir(exist_ok=True)
            shutil.move(official, source)
        coverage[view] = align_view(input_dir, sorted(source.glob("*.parquet")))
        description = input_dir / "description.md"
        _replace_text(description, align_description(description.read_text(encoding="utf-8")))
    return strip_benchmark(split_root), coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    digest, coverage = align_split(args.split)
    print(json.dumps({"public_tree_sha256": digest, "coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
