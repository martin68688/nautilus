#!/usr/bin/env python3
"""Plot the best Host-verified iMet score from each completed round."""

from pathlib import Path
import csv
import json

import matplotlib as mpl
import matplotlib.pyplot as plt


OUT = Path(__file__).resolve().parent
STEM = OUT / "imet_three_rounds_best"

ROUNDS = ["R1\n无记忆", "R2\n读取 R1", "R3\n读取 R1+R2"]
SCORES = [0.3891708697355064, 0.44095033341835804, 0.4585922833606213]
COLORS = ["#3B82A6", "#C45D7A", "#159A8C"]

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#AFC0CC",
        "xtick.color": "#526779",
        "ytick.color": "#526779",
        "text.color": "#243746",
        "axes.labelcolor": "#243746",
    }
)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = list(range(len(SCORES)))
    ax.plot(x, SCORES, color="#7B8FA0", linewidth=2.2, zorder=2)
    for xi, yi, color in zip(x, SCORES, COLORS):
        ax.scatter(
            xi,
            yi,
            s=125,
            facecolor="white",
            edgecolor=color,
            linewidth=2.6,
            zorder=3,
        )
        ax.text(
            xi,
            yi + 0.007,
            f"{yi:.5f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color=color,
            fontweight="semibold",
        )

    ax.set_title("iMet / 三轮最佳 Host micro-F1", loc="left", fontsize=19, pad=18)
    ax.set_ylabel("Host micro-F1 ↑", fontsize=11)
    ax.set_xticks(x, ROUNDS)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.36, 0.48)
    ax.set_yticks([0.36, 0.39, 0.42, 0.45, 0.48])
    ax.grid(axis="y", color="#DCE6ED", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", width=0.8)

    fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with STEM.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["round", "memory", "best_host_micro_f1"])
        writer.writerow(["R1", "none", SCORES[0]])
        writer.writerow(["R2", "R1", SCORES[1]])
        writer.writerow(["R3", "R1+R2", SCORES[2]])

    audit = {
        "figure_claim": "The best Host-verified iMet micro-F1 increased in each cumulative-memory round.",
        "archetype": "single-panel quantitative trend",
        "metric": "Host micro-F1",
        "round_scores": dict(zip(["R1", "R2", "R3"], SCORES)),
        "r1_source": "imet-r1-20260920-a1/RUN_OUTCOME.json",
        "r2_source": "imet-r2-20260920-b1/RUN_OUTCOME.json",
        "r3_source": "orchestrator.log terminal receipt at 2026-09-21T09:39:57Z",
        "scope": "One best terminal score per round; this is not a within-round attempt trajectory.",
        "excluded": "No preflight, failed-probe, or Plant records were included.",
    }
    STEM.with_name(STEM.name + "_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
