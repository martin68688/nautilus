"""Evaluate Agentic Poincare vs same-coordinate Flat-Twin results.

Input JSON shape:
[
  {
    "query_id": "...",
    "poincare": {"rare_recall_at_5": 0.4, "condition_precision": 0.8},
    "flat_twin": {"rare_recall_at_5": 0.2, "condition_precision": 0.8}
  }
]

The geometry claim passes only if:
  1. Rare Recall@5 improves by at least 5 percentage points,
  2. paired bootstrap p < 0.05,
  3. Condition Precision does not decrease.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """One-sided paired bootstrap for mean(left - right) > 0."""
    if left.shape != right.shape:
        raise ValueError("paired bootstrap arrays must have the same shape")
    if left.ndim != 1 or left.size == 0:
        raise ValueError("paired bootstrap needs a non-empty 1D array")
    diffs = left.astype(float) - right.astype(float)
    observed = float(np.mean(diffs))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    samples = diffs[idx].mean(axis=1)
    p_value = float((np.sum(samples <= 0.0) + 1) / (n_resamples + 1))
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
    return {
        "observed_mean_diff": observed,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_value": p_value,
        "n_pairs": int(diffs.size),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def evaluate(results: list[dict[str, Any]], *, n_resamples: int = 10_000, seed: int = 42) -> dict[str, Any]:
    if not results:
        raise ValueError("empty results")
    poincare_rr = np.asarray([float(x["poincare"]["rare_recall_at_5"]) for x in results], dtype=float)
    flat_rr = np.asarray([float(x["flat_twin"]["rare_recall_at_5"]) for x in results], dtype=float)
    poincare_cp = np.asarray([float(x["poincare"]["condition_precision"]) for x in results], dtype=float)
    flat_cp = np.asarray([float(x["flat_twin"]["condition_precision"]) for x in results], dtype=float)

    rr_bootstrap = paired_bootstrap(poincare_rr, flat_rr, n_resamples=n_resamples, seed=seed)
    rr_diff = rr_bootstrap["observed_mean_diff"]
    cp_diff = float(np.mean(poincare_cp - flat_cp))
    passed = rr_diff >= 0.05 and rr_bootstrap["p_value"] < 0.05 and cp_diff >= 0.0
    return {
        "status": "hyperbolic_geometry_claim_passed" if passed else "hyperbolic_geometry_claim_not_supported",
        "passed": passed,
        "thresholds": {
            "rare_recall_at_5_min_diff": 0.05,
            "paired_bootstrap_p_max": 0.05,
            "condition_precision_min_diff": 0.0,
        },
        "rare_recall_at_5": rr_bootstrap,
        "condition_precision": {
            "poincare_mean": float(np.mean(poincare_cp)),
            "flat_twin_mean": float(np.mean(flat_cp)),
            "mean_diff": cp_diff,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Poincare vs same-coordinate Flat-Twin ablation.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--n-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = json.loads(args.results.read_text(encoding="utf-8"))
    report = evaluate(results, n_resamples=args.n_resamples, seed=args.seed)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
