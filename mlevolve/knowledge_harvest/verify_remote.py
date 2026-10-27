"""Remote side of the knowledge check (standalone; copied to the compute host).

  python verify_remote.py prepare  --train train.parquet --cache DIR
  python verify_remote.py baseline --cache DIR
  python verify_remote.py run      --cache DIR --variant v.py --name ID --timeout SEC

Held-out set: the last 156 eras of the training data, after a 16-era purge;
training rows: every 4th earlier era. A variant only receives the context
below: features, eras and training targets. Validation targets stay in a file
the variant process never opens; scoring happens in this parent process.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

VAL_ERAS, PURGE, EVERY = 156, 16, 4
BASELINE_PARAMS = dict(n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=31, colsample_bytree=0.1)
N_JOBS = 24


def gauss_rank(pred, era):
    import pandas as pd
    from scipy.stats import norm
    s = pd.Series(np.asarray(pred, dtype=float))
    g = s.groupby(np.asarray(era))
    return np.asarray(norm.ppf(((g.rank(method="average") - 0.5) / g.transform("size")).to_numpy()))


def neutralize(pred, feats, era, proportion=1.0):
    pred = gauss_rank(pred, era)
    out = np.empty_like(pred)
    era = np.asarray(era)
    for e in np.unique(era):
        m = era == e
        f = np.nan_to_num(feats[m] - np.nanmean(feats[m], axis=0))
        beta = np.linalg.lstsq(f, pred[m], rcond=None)[0]
        r = pred[m] - proportion * f @ beta
        out[m] = r / (r.std() or 1.0)
    return out


def fit_lgbm(params, X, y, **fit_kwargs):
    import lightgbm as lgb
    m = ~np.isnan(y)
    model = lgb.LGBMRegressor(n_jobs=N_JOBS, verbose=-1, **params)
    model.fit(X[m], y[m], **fit_kwargs)
    return model


def numerai_corr(pred, target, era):
    from scipy.stats import norm
    out = {}
    era = np.asarray(era)
    for e in np.unique(era):
        m = (era == e) & ~np.isnan(target)
        p = np.argsort(np.argsort(pred[m], kind="stable"), kind="stable") + 1.0
        p = norm.ppf((p - 0.5) / m.sum())
        p = np.sign(p) * np.abs(p) ** 1.5
        t = target[m] - target[m].mean()
        t = np.sign(t) * np.abs(t) ** 1.5
        out[str(e)] = float(np.corrcoef(p, t)[0, 1])
    return out


def compare(scores: dict, base: dict, lags: int = 12) -> dict:
    eras = sorted(base)
    d = np.array([scores[e] - base[e] for e in eras])
    x = d - d.mean()
    v = x @ x / len(x)
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (x[k:] @ x[:-k]) / len(x)
    mean = np.array([scores[e] for e in eras])
    return {"corr": float(mean.mean()), "sharpe": float(mean.mean() / mean.std(ddof=1)),
            "diff": float(d.mean()), "t": float(d.mean() / np.sqrt(v / len(d))),
            "win_rate": float((d > 0).mean()), "eras": len(eras)}


# ---------------------------------------------------------------- prepare
def prepare(train: Path, cache: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    cache.mkdir(parents=True, exist_ok=True)
    schema = pq.ParquetFile(train).schema_arrow
    feats = [f.name for f in schema if f.name.startswith("feature_")
             and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))]
    targets = [f.name for f in schema if f.name.startswith("target")]
    era = pq.read_table(train, columns=["era"]).column("era").to_numpy().astype(str)
    eras = np.array(sorted(set(era)))
    val_eras = set(eras[-VAL_ERAS:])
    train_eras = set(eras[: len(eras) - VAL_ERAS - PURGE][::EVERY])
    tr, va = np.isin(era, list(train_eras)), np.isin(era, list(val_eras))
    table = pq.read_table(train, columns=feats + targets)
    for name, mask in (("train", tr), ("val", va)):
        X = np.lib.format.open_memmap(cache / f"X_{name}.npy", mode="w+", dtype=np.float32, shape=(int(mask.sum()), len(feats)))
        idx = np.flatnonzero(mask)
        for j, f in enumerate(feats):
            X[:, j] = table.column(f).to_numpy(zero_copy_only=False)[idx]
        X.flush()
        np.save(cache / f"era_{name}.npy", era[mask])
    np.savez(cache / "targets_train.npz", **{t: table.column(t).to_numpy(zero_copy_only=False)[tr].astype(np.float32) for t in targets})
    # Held-out labels: read only by the scoring parent, never by a variant.
    scoring = cache.parent / (cache.name + "_scoring")
    scoring.mkdir(exist_ok=True)
    np.save(scoring / "heldout_target.npy", table.column("target").to_numpy(zero_copy_only=False)[va].astype(np.float32))
    sets = {}
    fjson = train.parent / "features.json"
    if fjson.exists():
        raw = json.loads(fjson.read_text())
        sets = {k: [c for c in v if c in feats] for k, v in (raw.get("feature_sets") or {}).items()}
    (cache / "meta.json").write_text(json.dumps({"features": feats, "targets": targets, "feature_sets": sets,
                                                 "train_eras": len(train_eras), "val_eras": len(val_eras)}))
    print(json.dumps({"train_rows": int(tr.sum()), "val_rows": int(va.sum()), "features": len(feats)}))


def context(cache: Path) -> SimpleNamespace:
    meta = json.loads((cache / "meta.json").read_text())
    t = np.load(cache / "targets_train.npz")
    return SimpleNamespace(
        X_train=np.load(cache / "X_train.npy", mmap_mode="r"), X_val=np.load(cache / "X_val.npy", mmap_mode="r"),
        train_era=np.load(cache / "era_train.npy"), val_era=np.load(cache / "era_val.npy"),
        targets={k: t[k] for k in t.files}, main_target="target", feature_names=meta["features"],
        feature_sets=meta["feature_sets"], baseline_params=dict(BASELINE_PARAMS), n_jobs=N_JOBS,
        baseline_val_pred=np.load(cache / "pred_baseline.npy") if (cache / "pred_baseline.npy").exists() else None,
        fit_lgbm=fit_lgbm, gauss_rank=gauss_rank, neutralize=neutralize,
    )


# ---------------------------------------------------------------- run
def _child(cache: str, variant: str, out: str) -> None:
    namespace: dict = {"np": np}
    exec(compile(Path(variant).read_text(), variant, "exec"), namespace)
    pred = np.asarray(namespace["variant"](context(Path(cache))), dtype=float).reshape(-1)
    np.save(out, pred)


def run(cache: Path, variant: Path, name: str, timeout: int) -> dict:
    out = cache / f"pred_{name}.npy"
    t0 = time.time()
    ctx = mp.get_context("spawn")
    errfile = cache / f"err_{name}.txt"
    proc = ctx.Process(target=_guarded, args=(str(cache), str(variant), str(out), str(errfile)))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.kill()
        proc.join()
        return {"status": "timeout", "seconds": round(time.time() - t0)}
    if proc.exitcode != 0 or not out.exists():
        err = errfile.read_text()[-3000:] if errfile.exists() else f"exit code {proc.exitcode}"
        return {"status": "error", "error": err, "seconds": round(time.time() - t0)}
    pred = np.load(out)
    held = np.load(cache.parent / (cache.name + "_scoring") / "heldout_target.npy")
    era = np.load(cache / "era_val.npy")
    if pred.shape != held.shape or not np.isfinite(pred).all():
        return {"status": "error", "error": f"prediction shape {pred.shape} (want {held.shape}) or non-finite values"}
    base = json.loads((cache / "baseline_scores.json").read_text())
    return {"status": "ok", "seconds": round(time.time() - t0), **compare(numerai_corr(pred, held, era), base)}


def _guarded(cache, variant, out, errfile):
    try:
        _child(cache, variant, out)
    except Exception:
        Path(errfile).write_text(traceback.format_exc())
        sys.exit(1)


def baseline(cache: Path) -> dict:
    ctx = context(cache)
    model = fit_lgbm(BASELINE_PARAMS, ctx.X_train, ctx.targets["target"])
    pred = model.predict(ctx.X_val)
    np.save(cache / "pred_baseline.npy", pred)
    scores = numerai_corr(pred, np.load(cache.parent / (cache.name + "_scoring") / "heldout_target.npy"), ctx.val_era)
    (cache / "baseline_scores.json").write_text(json.dumps(scores))
    vals = np.array(list(scores.values()))
    return {"corr": float(vals.mean()), "sharpe": float(vals.mean() / vals.std(ddof=1)), "eras": len(vals)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "baseline", "run"])
    ap.add_argument("--train", type=Path)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--variant", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare(a.train, a.cache)
    elif a.cmd == "baseline":
        print(json.dumps(baseline(a.cache)))
    else:
        print(json.dumps(run(a.cache, a.variant, a.name, a.timeout)))
