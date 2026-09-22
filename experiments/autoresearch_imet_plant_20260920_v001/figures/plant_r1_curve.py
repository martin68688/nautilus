#!/usr/bin/env python3
"""Plant R1 Host-score curve from the persisted, current-manifest ledger.

Core claim: the high Plant R1 score is a Host-recomputed search macro-F1
from a valid run, and it is reached by a small number of certified
candidate attempts rather than by a tune-score-only proxy.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent / "plant"
RUN = TASK_ROOT / "runs" / "plant-r1-20260920-repair-b1"
OUT = HERE / "plant_r1_score_curve"
CURRENT_MANIFEST = "a2209c362fae1f7ea84ecda99d977ba377984fa84d263e1e82b125cd6a5055ca"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "STHeiti", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "legend.frameon": False,
    }
)


def load_rows() -> tuple[list[dict], dict, dict]:
    launch = json.loads((RUN / "LAUNCH.json").read_text())
    state = json.loads((RUN / "STATE.json").read_text())
    rows: list[dict] = []
    for index, row in enumerate(state["history"]):
        rows.append(
            {
                "attempt": index,
                "attempt_id": row["attempt"],
                "status": row.get("status", ""),
                "elapsed_min": float(row.get("elapsed_seconds", 0.0)) / 60.0,
                "search_macro_f1": row.get("search_score"),
                "tune_macro_f1": row.get("tune_score"),
                "host_recomputed": bool(row.get("host_recomputed", False)),
                "reload_verified": bool(row.get("reload_verified", False)),
                "decision": row.get("decision", ""),
                "source_sha256": row.get("source_sha256", ""),
            }
        )
    return rows, launch, state


def write_source(rows: list[dict], launch: dict, state: dict) -> None:
    fields = [
        "attempt",
        "attempt_id",
        "status",
        "elapsed_min",
        "search_macro_f1",
        "tune_macro_f1",
        "host_recomputed",
        "reload_verified",
        "decision",
        "source_sha256",
    ]
    with (OUT.with_name(OUT.name + "_data.csv")).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "run_id": launch["run_id"],
        "data_manifest_sha256": launch["data_manifest_sha256"],
        "expected_current_manifest_sha256": CURRENT_MANIFEST,
        "manifest_matches": launch["data_manifest_sha256"] == CURRENT_MANIFEST,
        "metric": launch["metric"],
        "budget_seconds": launch["budget_seconds"],
        "best_search_macro_f1": state["best"],
        "best_attempt": state["best_attempt"],
        "attempt_count": len(rows),
        "valid_host_recomputed_count": sum(
            r["status"] == "valid" and r["host_recomputed"] and r["reload_verified"]
            for r in rows
        ),
        "failed_count": sum(r["status"] != "valid" for r in rows),
    }
    (OUT.with_name(OUT.name + "_audit.json")).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\\n"
    )


def draw(rows: list[dict], launch: dict, state: dict) -> None:
    valid = [r for r in rows if r["status"] == "valid" and r["host_recomputed"]]
    failed = [r for r in rows if r["status"] != "valid"]
    xs = np.asarray([r["elapsed_min"] for r in valid], dtype=float)
    search = np.asarray([r["search_macro_f1"] for r in valid], dtype=float)
    tune = np.asarray([r["tune_macro_f1"] for r in valid], dtype=float)
    best = np.maximum.accumulate(search)
    end_min = float(launch["budget_seconds"]) / 60.0

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=False)
    ax.plot(
        xs,
        search,
        color="#277D83",
        alpha=0.30,
        linewidth=1.3,
        zorder=1,
    )
    ax.scatter(
        xs,
        search,
        s=48,
        facecolors="white",
        edgecolors="#277D83",
        linewidths=1.4,
        label="Host search macro-F1",
        zorder=4,
    )
    ax.step(
        np.r_[xs, end_min],
        np.r_[best, best[-1]],
        where="post",
        color="#0F5E63",
        linewidth=2.7,
        label="best-so-far",
        zorder=3,
    )
    ax.plot(
        xs,
        tune,
        color="#BD8541",
        linestyle="--",
        linewidth=1.6,
        marker="s",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.1,
        label="tune macro-F1 (selection only)",
        zorder=2,
    )
    if failed:
        fx = [r["elapsed_min"] for r in failed]
        fy = [max(float(np.nanmax(search)) + 0.006, 0.97)] * len(failed)
        ax.scatter(
            fx,
            fy,
            marker="x",
            s=54,
            linewidths=1.5,
            color="#7B858C",
            label="failed candidate",
            zorder=5,
        )

    ax.scatter(
        [end_min],
        [best[-1]],
        marker="*",
        s=145,
        color="#0F5E63",
        edgecolors="white",
        linewidths=0.8,
        zorder=6,
    )
    ax.annotate(
        f"best {best[-1]:.5f}",
        xy=(end_min, best[-1]),
        xytext=(-7, 12),
        textcoords="offset points",
        ha="right",
        color="#0F5E63",
        fontsize=10,
        fontweight="bold",
    )
    ax.set(
        xlim=(0, end_min + 2),
        ylim=(0.90, 0.985),
        xlabel="Elapsed time (min)",
        ylabel="Macro-F1 (higher is better)",
    )
    ax.set_title("Plant / R1 Host score curve", loc="left", fontsize=16, fontweight="bold", pad=12)
    ax.text(
        0.0,
        1.01,
        "Current-manifest run · 7 valid Host-recomputed candidates · 1 failed candidate",
        transform=ax.transAxes,
        color="#5F6B74",
        fontsize=9,
        va="bottom",
    )
    ax.grid(axis="y", color="#E5E9EC", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.text(
        0.0,
        -0.18,
        "Search score is recomputed by the Host on the held-out search split (482 images); "
        "tune score is used only for candidate selection. × marks the invalid candidate.",
        transform=ax.transAxes,
        fontsize=8,
        color="#5F6B74",
        va="top",
    )
    fig.subplots_adjust(left=0.11, right=0.97, bottom=0.25, top=0.84)
    fig.savefig(str(OUT) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(OUT) + ".svg", bbox_inches="tight")
    fig.savefig(str(OUT) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows, launch, state = load_rows()
    assert launch["data_manifest_sha256"] == CURRENT_MANIFEST
    assert all(r["status"] != "valid" or r["host_recomputed"] for r in rows)
    write_source(rows, launch, state)
    draw(rows, launch, state)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
