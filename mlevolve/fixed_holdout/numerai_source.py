"""Build the Numerai v6 source dataset consumed by `fixed_holdout.prepare`.

This runs on the Host before a search, never inside a candidate.  It turns the
official Numerai `train.parquet` and `validation.parquet` into the MLE-bench
style layout that the fixed-holdout pipeline expects:

    <dataset_root>/numerai-v6/prepared/public/
        train.parquet          labelled public eras (features, era, targets)
        test.parquet           lockbox eras, features only
        sample_submission.csv  id,prediction in lockbox row order
        features.json          feature names actually included
        description.md         task statement shown to the agent
    <dataset_root>/numerai-v6/prepared/private/test.csv   id,target,era

With ``selection_eras > 0`` the last public eras become a Host-held selection
set: ``public/selection.parquet`` (features only) and
``private/selection.csv`` (labels, see ``fixed_holdout.selection``).

The lockbox is the last `lockbox_eras` eras whose main target has matured.
The `embargo_eras` eras just before it are dropped from the public data, so
no public 60-day target overlaps the lockbox return window.

Typical use:

    cd mlevolve
    python -m fixed_holdout.numerai_source --dataset-root /data/numerai \\
        --cache-dir /data/numerai/raw --feature-set small
    python -m fixed_holdout.prepare --dataset-root /data/numerai \\
        --output-root /data/numerai/fixed --task numerai-v6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fixed_holdout.common import sha256_file, write_json


TASK_ID = "numerai-v6"
DATA_VERSION = "v6.0"
MAIN_TARGET = "target"
TICKER_COLUMN = "numerai_ticker"
DATE_COLUMN = "date"


def split_lockbox(
    eras: list[str], *, lockbox_eras: int, embargo_eras: int
) -> tuple[list[str], list[str]]:
    """Return (public_eras, lockbox_eras) from chronologically sortable eras."""
    ordered = sorted(set(eras))
    if lockbox_eras <= 0 or embargo_eras < 0:
        raise ValueError("lockbox_eras must be positive and embargo_eras non-negative")
    if len(ordered) <= lockbox_eras + embargo_eras:
        raise ValueError(
            f"Only {len(ordered)} eras; need more than {lockbox_eras + embargo_eras}"
        )
    lockbox = ordered[-lockbox_eras:]
    public = ordered[: len(ordered) - lockbox_eras - embargo_eras]
    return public, lockbox


def _download(cache_dir: Path) -> tuple[Path, Path, Path]:
    from numerapi import NumerAPI

    cache_dir.mkdir(parents=True, exist_ok=True)
    napi = NumerAPI()
    paths = []
    for name in ("features.json", "train.parquet", "validation.parquet"):
        destination = cache_dir / name
        if not destination.exists():
            napi.download_dataset(f"{DATA_VERSION}/{name}", str(destination))
        paths.append(destination)
    return tuple(paths)  # type: ignore[return-value]


def _description(
    *,
    feature_set: str,
    n_features: int,
    targets: list[str],
    public_eras: list[str],
    lockbox: list[str],
    embargo_eras: int,
    train_era_stride: int,
    selection: list[str] | None = None,
) -> str:
    gap_rows = -(-embargo_eras // max(1, train_era_stride))
    gap_text = (
        f"at least {embargo_eras} weeks between the last training era and the first "
        f"validation era (= {gap_rows} consecutive eras of this file, because only every "
        f"{train_era_stride}th weekly era is kept)"
        if train_era_stride > 1
        else f"at least {embargo_eras} eras (weeks) between the last training era and the "
        "first validation era"
    )
    selection_files = (
        f"""
- `selection.parquet`: unlabelled rows for eras {selection[0]} to {selection[-1]}
  (every weekly era). Same columns as `test.parquet`. The Host keeps their
  labels and uses them to rank your solution."""
        if selection
        else ""
    )
    selection_output = (
        """
Also predict every row of `selection.parquet` with the same final model and
write `./submission/selection_predictions.csv` with columns `id,prediction`.
The Host scores these predictions with the metric above; that Host score, not
the printed validation score, is what ranks solutions during the search."""
        if selection
        else ""
    )
    return f"""# Numerai v6 stock-ranking task

Predict, for every stock in every week ("era"), how well it will perform over
the next 60 business days relative to the other stocks in the same era.

## Files
- `train.parquet`: labelled rows. Columns: `id`, `era` (a weekly date string,
  chronological when sorted), {n_features} features (`feature_*`, Numerai
  "{feature_set}" set; integer bins 0-4 plus a few floats, `feature_country` is
  a string), the main label `target` and auxiliary labels
  {", ".join(f"`{t}`" for t in targets if t != MAIN_TARGET) or "(none)"}.
  Eras {public_eras[0]} to {public_eras[-1]}{
  f", every {train_era_stride}th era kept" if train_era_stride > 1 else ""}.
- `test.parquet`: unlabelled rows to predict, eras {lockbox[0]} to
  {lockbox[-1]}. Same columns as train minus every label.
- `sample_submission.csv`: `id,prediction` in the exact required row order.{selection_files}

Labels take values in {{0, 0.25, 0.5, 0.75, 1}}. Auxiliary labels may be NaN.

## Metric (higher is better)
The score is the mean over eras of Numerai's per-era correlation:

```python
import numpy as np
from scipy.stats import norm

def numerai_corr(pred, target):
    ranks = pred.rank(method="first").to_numpy()
    g = norm.ppf((ranks - 0.5) / len(ranks))
    p = np.sign(g) * np.abs(g) ** 1.5
    t = target.to_numpy() - target.mean()
    t = np.sign(t) * np.abs(t) ** 1.5
    return np.corrcoef(p, t)[0, 1]

score = df.groupby("era").apply(lambda d: numerai_corr(d["prediction"], d["target"])).mean()
```

Only the ranking of predictions inside each era matters. Typical good models
score 0.01-0.03; a score above 0.05 almost always means leakage.

## Validation rules
- Never split rows randomly. Hold out the **latest** eras of `train.parquet`
  as validation and leave a gap of {gap_text}, because each label looks
  60 business days ahead and neighbouring eras share future returns.
- Compute the validation score with the metric above and print exactly
  `Final Validation Score: <value>`.

## Required output
Fit the final model, predict every row of `test.parquet`, and write
`./submission/submission.csv` with columns `id,prediction`, rows in the order of
`sample_submission.csv`, finite numeric predictions.{selection_output}

## Practical notes
- The data is large; read only the columns you need with
  `pd.read_parquet(path, columns=[...])` and keep integer features as int8.
- Gradient-boosted trees (LightGBM) are a strong baseline. A reference
  configuration is n_estimators=2000, learning_rate=0.01, max_depth=5,
  num_leaves=31, colsample_bytree=0.1.
"""


def build_source(
    train_path: Path,
    validation_path: Path,
    features_path: Path,
    dataset_root: Path,
    *,
    feature_set: str = "small",
    lockbox_eras: int = 52,
    embargo_eras: int = 13,
    train_era_stride: int = 1,
    extra_targets: tuple[str, ...] = ("target_jupiter_20",),
    selection_eras: int = 0,
) -> Path:
    """Write the source layout and return `<dataset_root>/<task>/prepared`."""
    metadata = json.loads(Path(features_path).read_text(encoding="utf-8"))
    features = list(metadata["feature_sets"][feature_set])
    targets = [MAIN_TARGET, *[t for t in extra_targets if t != MAIN_TARGET]]
    unknown = sorted(set(targets) - set(metadata["targets"]))
    if unknown:
        raise ValueError(f"Unknown Numerai targets: {unknown}")
    columns = [TICKER_COLUMN, DATE_COLUMN, *features, *targets]

    frames = [pd.read_parquet(path, columns=columns) for path in (train_path, validation_path)]
    data = pd.concat(frames, ignore_index=True)
    data = data.dropna(subset=[MAIN_TARGET])
    data = data.rename(columns={DATE_COLUMN: "era"})
    data["era"] = data["era"].astype(str)
    data.insert(0, "id", data[TICKER_COLUMN].astype(str) + "|" + data["era"])
    data = data.drop(columns=[TICKER_COLUMN])
    if data["id"].duplicated().any():
        raise ValueError("Numerai source has duplicate (ticker, era) rows")

    public_eras, lockbox = split_lockbox(
        data["era"].tolist(), lockbox_eras=lockbox_eras, embargo_eras=embargo_eras
    )
    labelled_eras = list(public_eras)  # every labelled era before the lockbox embargo
    selection: list[str] = []
    if selection_eras > 0:
        # [train][embargo][selection][embargo][lockbox]: the selection eras keep
        # every week; their labels stay with the Host.
        public_eras, selection = split_lockbox(
            public_eras, lockbox_eras=selection_eras, embargo_eras=embargo_eras
        )
    if train_era_stride > 1:
        # Keep the most recent public era so the embargo gap stays as configured.
        public_eras = public_eras[::-1][::train_era_stride][::-1]
        labelled_eras = labelled_eras[::-1][::train_era_stride][::-1]

    train = data[data["era"].isin(public_eras)].sort_values(["era", "id"])
    selection_rows = data[data["era"].isin(selection)].sort_values(["era", "id"])
    test = data[data["era"].isin(lockbox)].sort_values(["era", "id"])

    prepared = Path(dataset_root) / TASK_ID / "prepared"
    public = prepared / "public"
    private = prepared / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    train.to_parquet(public / "train.parquet", index=False)
    test.drop(columns=targets).to_parquet(public / "test.parquet", index=False)
    pd.DataFrame({"id": test["id"], "prediction": 0.5}).to_csv(
        public / "sample_submission.csv", index=False
    )
    if selection:
        selection_rows.drop(columns=targets).to_parquet(public / "selection.parquet", index=False)
        selection_rows[["id", MAIN_TARGET, "era"]].to_csv(private / "selection.csv", index=False)
        # Host-only refit training data: train + embargo + selection eras with labels,
        # used to retrain the selected program after the search (fixed_holdout.refit).
        refit = prepared / "refit"
        refit.mkdir(parents=True, exist_ok=True)
        data[data["era"].isin(labelled_eras)].sort_values(["era", "id"]).to_parquet(
            refit / "train.parquet", index=False
        )
    write_json(public / "features.json", {"feature_set": feature_set, "features": features})
    (public / "description.md").write_text(
        _description(
            feature_set=feature_set,
            n_features=len(features),
            targets=targets,
            public_eras=public_eras,
            lockbox=lockbox,
            embargo_eras=embargo_eras,
            train_era_stride=train_era_stride,
            selection=selection or None,
        ),
        encoding="utf-8",
    )
    test[["id", MAIN_TARGET, "era"]].to_csv(private / "test.csv", index=False)
    write_json(
        prepared / "numerai_source_manifest.json",
        {
            "task_id": TASK_ID,
            "data_version": DATA_VERSION,
            "feature_set": feature_set,
            "targets": targets,
            "lockbox_eras": lockbox_eras,
            "embargo_eras": embargo_eras,
            "train_era_stride": train_era_stride,
            "public_era_range": [public_eras[0], public_eras[-1]],
            "public_era_count": len(public_eras),
            "lockbox_era_range": [lockbox[0], lockbox[-1]],
            "selection_era_range": [selection[0], selection[-1]] if selection else None,
            "selection_rows": int(len(selection_rows)),
            "refit_era_range": [labelled_eras[0], labelled_eras[-1]] if selection else None,
            "train_rows": int(len(train)),
            "lockbox_rows": int(len(test)),
            "source_sha256": {
                "train.parquet": sha256_file(Path(train_path)),
                "validation.parquet": sha256_file(Path(validation_path)),
                "features.json": sha256_file(Path(features_path)),
            },
        },
    )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, help="Download Numerai files here")
    parser.add_argument("--train", type=Path, help="Existing v6 train.parquet")
    parser.add_argument("--validation", type=Path, help="Existing v6 validation.parquet")
    parser.add_argument("--features", type=Path, help="Existing v6 features.json")
    parser.add_argument("--feature-set", default="small")
    parser.add_argument("--lockbox-eras", type=int, default=52)
    parser.add_argument("--embargo-eras", type=int, default=13)
    parser.add_argument("--train-era-stride", type=int, default=1)
    parser.add_argument("--extra-target", action="append", default=None)
    parser.add_argument("--selection-eras", type=int, default=0,
                        help="Host-held selection eras taken from the end of the public data")
    args = parser.parse_args()
    if args.train and args.validation and args.features:
        features_path, train_path, validation_path = args.features, args.train, args.validation
    elif args.cache_dir:
        features_path, train_path, validation_path = _download(args.cache_dir)
    else:
        parser.error("Pass --cache-dir, or all of --train, --validation and --features")
    prepared = build_source(
        train_path,
        validation_path,
        features_path,
        args.dataset_root,
        feature_set=args.feature_set,
        lockbox_eras=args.lockbox_eras,
        embargo_eras=args.embargo_eras,
        train_era_stride=args.train_era_stride,
        extra_targets=tuple(args.extra_target or ("target_jupiter_20",)),
        selection_eras=args.selection_eras,
    )
    print(prepared)


if __name__ == "__main__":
    main()
