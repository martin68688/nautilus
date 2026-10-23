import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mlevolve"))

from fixed_holdout.common import read_manifest  # noqa: E402
from fixed_holdout.evaluate import _numerai_corr, evaluate_submission  # noqa: E402
from fixed_holdout.numerai_source import build_source, split_lockbox  # noqa: E402
from fixed_holdout.prepare import prepare_task  # noqa: E402
from fixed_holdout.validation import validate_submission  # noqa: E402


FEATURES = ["feature_a", "feature_b", "feature_c"]


def _reference_numerai_corr(pred: pd.Series, target: pd.Series) -> float:
    ranked = (pred.rank(method="first") - 0.5) / len(pred)
    gauss = norm.ppf(ranked)
    p = np.sign(gauss) * np.abs(gauss) ** 1.5
    t = target - target.mean()
    t = np.sign(t) * np.abs(t) ** 1.5
    return float(np.corrcoef(p, t)[0, 1])


def _write_raw_numerai(tmp_path: Path, n_eras: int = 40, n_stocks: int = 30) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(0)
    eras = pd.date_range("2018-01-05", periods=n_eras, freq="7D").strftime("%Y-%m-%d")
    rows = []
    for era in eras:
        signal = rng.normal(size=n_stocks)
        target = pd.Series(signal + rng.normal(scale=2.0, size=n_stocks)).rank(pct=True)
        binned = np.clip(np.floor(target * 5) / 4, 0, 1)
        for i in range(n_stocks):
            rows.append(
                {
                    "numerai_ticker": f"T{i:03d} US",
                    "date": era,
                    "feature_a": int(np.clip(signal[i] + 2, 0, 4)),
                    "feature_b": int(rng.integers(0, 5)),
                    "feature_c": int(rng.integers(0, 5)),
                    "target": float(binned[i]),
                    "target_jupiter_20": float(binned[i]),
                }
            )
    frame = pd.DataFrame(rows)
    # The newest eras have immature (NaN) main targets, as in real validation data.
    frame.loc[frame["date"] >= eras[-2], "target"] = np.nan
    raw = tmp_path / "raw"
    raw.mkdir()
    split_era = eras[n_eras // 2]
    frame[frame["date"] < split_era].to_parquet(raw / "train.parquet", index=False)
    frame[frame["date"] >= split_era].to_parquet(raw / "validation.parquet", index=False)
    (raw / "features.json").write_text(
        json.dumps(
            {
                "feature_sets": {"small": FEATURES},
                "targets": ["target", "target_jupiter_20", "target_jupiter_60"],
            }
        ),
        encoding="utf-8",
    )
    return raw / "train.parquet", raw / "validation.parquet", raw / "features.json"


def test_split_lockbox_leaves_an_embargo_gap():
    eras = [f"2020-01-{day:02d}" for day in range(1, 21)]
    public, lockbox = split_lockbox(eras[::-1], lockbox_eras=5, embargo_eras=3)
    assert lockbox == eras[-5:]
    assert public == eras[:12]
    assert set(public).isdisjoint(lockbox)
    with pytest.raises(ValueError):
        split_lockbox(eras, lockbox_eras=15, embargo_eras=5)


def test_numerai_corr_matches_reference_transform():
    rng = np.random.default_rng(1)
    pred = pd.Series(rng.normal(size=500))
    target = pd.Series(rng.choice([0, 0.25, 0.5, 0.75, 1.0], size=500))
    assert _numerai_corr(pred.to_numpy(), target.to_numpy()) == pytest.approx(
        _reference_numerai_corr(pred, target), abs=1e-12
    )


def test_numerai_source_and_lockbox_are_label_isolated(tmp_path):
    train_path, validation_path, features_path = _write_raw_numerai(tmp_path)
    prepared = build_source(
        train_path,
        validation_path,
        features_path,
        tmp_path / "datasets",
        lockbox_eras=6,
        embargo_eras=3,
    )
    source = json.loads((prepared / "numerai_source_manifest.json").read_text())
    public_train = pd.read_parquet(prepared / "public" / "train.parquet")
    public_test = pd.read_parquet(prepared / "public" / "test.parquet")
    # Immature eras are dropped before choosing the lockbox.
    assert source["lockbox_era_range"][1] < "2018-10-05"
    assert public_train["era"].max() < public_test["era"].min()
    gap = sorted(set(pd.read_parquet(train_path)["date"]) | set(pd.read_parquet(validation_path)["date"]))
    assert gap.index(public_test["era"].min()) - gap.index(public_train["era"].max()) == 4
    assert not {"target", "target_jupiter_20"} & set(public_test.columns)
    assert public_test["era"].nunique() == 6

    split_root = prepare_task(tmp_path / "datasets", "numerai-v6", tmp_path / "fixed", copy_mode="copy")
    train_manifest = read_manifest(split_root / "train_view" / "fixed_holdout_manifest.json", "train_view")
    evaluator_manifest_path = split_root / "evaluator_view" / "fixed_holdout_manifest.json"
    labels = pd.read_csv(split_root / "evaluator_view" / "labels.csv")
    assert train_manifest["maximize"] is True
    assert train_manifest["group_column"] == "era"
    assert list(labels.columns) == ["id", "target", "era"]
    assert not (split_root / "train_view" / "input" / "labels.csv").exists()

    sample = split_root / "train_view" / "input" / "sample_submission.csv"
    assert validate_submission(split_root / "train_view" / "fixed_holdout_manifest.json", sample) == (True, "valid")

    informed = public_test[["id"]].copy()
    informed["prediction"] = public_test["feature_a"].to_numpy() + 1e-3 * np.arange(len(public_test))
    submission = tmp_path / "submission_node.csv"
    informed.to_csv(submission, index=False)
    result = evaluate_submission(evaluator_manifest_path, submission)
    details = result["metric_details"]
    assert result["metric"] == "numerai_era_corr"
    assert details["n_eras"] == 6
    assert result["score"] == pytest.approx(details["corr_mean"])
    assert result["score"] > 0.1

    merged = public_test[["id", "era"]].merge(labels[["id", "target"]], on="id")
    expected = np.mean(
        [
            _reference_numerai_corr(
                informed.set_index("id").loc[group["id"], "prediction"].reset_index(drop=True),
                group["target"].reset_index(drop=True),
            )
            for _, group in merged.groupby("era")
        ]
    )
    assert result["score"] == pytest.approx(expected, abs=1e-12)
