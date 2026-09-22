#!/usr/bin/env python3
"""iMet three-round figure in the requested trajectory-style layout.

R1/R2 use their persisted attempt states.  R3 is intentionally represented by
the terminal receipt only because its intermediate candidate state was not
returned to the Mac; no R3 intermediate points are invented.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
STEM = OUT / "imet_three_rounds_style"

BLUE = "#3E83B3"
PINK = "#C85C7D"
TEAL = "#159B8F"
SLATE = "#6F879A"
GRID = "#DCE6ED"
TEXT = "#40576A"


def load_state(name: str) -> dict:
    return json.loads((ROOT / "imet" / "runs" / name / "STATE.json").read_text())


def r3_terminal() -> tuple[float, float]:
    log = (ROOT / "orchestrator.log").read_text()
    match = re.findall(
        r"imet terminal imet-r3-20260920-a1 best=([0-9.]+) elapsed=([0-9.]+)", log
    )
    if not match:
        raise RuntimeError("R3 terminal receipt not found")
    score, elapsed = match[-1]
    return float(elapsed) / 60.0, float(score)


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
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "text.color": "#263A4A",
        "axes.labelcolor": "#263A4A",
    }
)


def plot_round(ax, state: dict, color: str, marker: str, label: str, linestyle: str = "-") -> dict:
    history = state["history"]
    valid = [h for h in history if h.get("status") == "valid" and h.get("search_score") is not None]
    for item in valid:
        start = float(item["started_elapsed"]) / 60.0
        end = float(item["elapsed_seconds"]) / 60.0
        score = float(item["search_score"])
        ax.plot([start, end], [score, score], color=color, linewidth=2.8, linestyle=linestyle, zorder=2)
        ax.plot([end, end], [score, score - 0.000001], color=color, linewidth=3.0, zorder=3)
        ax.scatter(end, score, s=125, marker=marker, facecolor="white", edgecolor=color, linewidth=2.4, zorder=4)

    return {
        "valid": valid,
        "all": history,
        "best": float(state["best"]),
        "elapsed": max(float(h.get("elapsed_seconds", 0.0)) for h in history) / 60.0,
        "label": label,
    }


def annotate_valid_nodes(ax, info: dict, color: str, round_name: str) -> None:
    """Show the numeric score beside every recorded valid candidate node."""
    label_box = dict(facecolor="white", edgecolor="none", pad=1.0)
    valid = info["valid"]
    # The round-best value is already called out at the right edge; annotate
    # the other scored nodes here so every distinct score remains visible.
    best_idx = max(range(len(valid)), key=lambda i: float(valid[i]["search_score"])) if valid else -1
    for idx, item in enumerate(valid):
        if idx == best_idx:
            continue
        x = float(item["elapsed_seconds"]) / 60.0
        score = float(item["search_score"])
        # Stagger labels around nearby R2 nodes so both scores remain legible.
        if round_name == "R2" and idx == 0:
            dx, dy, ha = 1.0, 0.006, "left"
        elif round_name == "R2" and idx == 1:
            dx, dy, ha = -1.0, -0.009, "right"
        else:
            dx, dy, ha = -1.0, 0.006, "right"
        ax.text(
            x + dx,
            score + dy,
            f"{score:.5f}",
            color=color,
            fontsize=9.5,
            ha=ha,
            va="center",
            bbox=label_box,
            zorder=11,
        )


def main() -> None:
    r1 = load_state("imet-r1-20260920-a1")
    r2 = load_state("imet-r2-20260920-b1")
    r3_minutes, r3_best = r3_terminal()

    fig = plt.figure(figsize=(12.0, 7.0), facecolor="white")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.79, bottom=0.085)
    grid = fig.add_gridspec(3, 1, height_ratios=[5.4, 1.15, 2.05], hspace=0.28)
    ax = fig.add_subplot(grid[0])
    ax_t = fig.add_subplot(grid[1], sharex=ax)
    ax_b = fig.add_subplot(grid[2])

    r1_info = plot_round(ax, r1, BLUE, "o", "R1（无记忆）")
    r2_info = plot_round(ax, r2, PINK, "s", "R2（R1 记忆，最新重跑）")

    # R3 has only a terminal receipt on the Mac.  Plot that receipt as one point.
    ax.scatter(r3_minutes, r3_best, s=140, marker="D", facecolor="white", edgecolor=TEAL, linewidth=2.5, zorder=5)
    annotate_valid_nodes(ax, r1_info, BLUE, "R1")
    annotate_valid_nodes(ax, r2_info, PINK, "R2")

    # Dashed continuation highlights that the R3 intermediate trajectory is not available.
    ax.plot([r3_minutes - 12, r3_minutes], [r3_best, r3_best], color=TEAL, linewidth=2.5, linestyle=(0, (5, 4)), alpha=0.75)

    fig.text(0.105, 0.956, "iMet / 三轮外层 search micro-F1", ha="left", va="top", fontsize=23, color="#263A4A")
    fig.text(0.105, 0.913, "同一固定 search 划分 · Host micro-F1 越高越好", ha="left", va="top", color="#758DA0", fontsize=13)
    fig.text(0.985, 0.942, "3 轮 / R3 仅回传终态收据", ha="right", va="top", color="#758DA0", fontsize=12)

    # Legend matches the reference figure and explicitly separates unverified points.
    handles = [
        plt.Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white", markeredgewidth=2, linewidth=2.6, label="R1（无记忆） · 1 个认证"),
        plt.Line2D([0], [0], color=PINK, marker="s", markerfacecolor="white", markeredgewidth=2, linewidth=2.6, label="R2（R1 记忆，最新重跑） · 2 个认证"),
        plt.Line2D([0], [0], color=TEAL, marker="D", markerfacecolor="white", markeredgewidth=2, linewidth=2.6, label="R3（累计记忆） · 1 个终态收据"),
        plt.Line2D([0], [0], color="#73899A", marker="x", linestyle="None", markersize=8, markeredgewidth=2, label="未认证 / 失败候选"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=4, frameon=False, fontsize=10)

    ax.set_ylabel("Host micro-F1 ↑", fontsize=12)
    ax.set_ylim(0.35, 0.48)
    ax.set_xlim(0, 122)
    ax.set_yticks([0.36, 0.39, 0.42, 0.45, 0.48])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False, length=0)

    # Annotate only the three round bests, as in the supplied style.
    label_box = dict(facecolor="white", edgecolor="none", pad=1.5)
    ax.text(121.2, r1_info["best"] + 0.004, f"R1  {r1_info['best']:.5f}", color=BLUE, fontsize=12, ha="right", bbox=label_box, zorder=10)
    ax.text(121.2, r2_info["best"] + 0.004, f"R2  {r2_info['best']:.5f}", color=PINK, fontsize=12, ha="right", bbox=label_box, zorder=10)
    ax.text(121.2, r3_best + 0.004, f"R3  {r3_best:.5f}", color=TEAL, fontsize=12, ha="right", bbox=label_box, zorder=10)

    # Candidate timeline strip.
    ax_t.set_ylim(-0.5, 2.5)
    ax_t.set_yticks([2, 1, 0], ["R1 · 无记忆", "R2 · 仅 R1 记忆", "R3 · 累计记忆"])
    ax_t.tick_params(axis="y", length=0, labelsize=10)
    ax_t.set_xticks([0, 20, 40, 60, 80, 100, 120])
    ax_t.set_xticklabels(["0", "20", "40", "60", "80", "100", "120"])
    ax_t.tick_params(axis="x", labelbottom=True, length=0, pad=6, labelsize=10)
    ax_t.grid(axis="x", color="#EDF2F5", linewidth=0.8)
    ax_t.set_axisbelow(True)
    for y, color in [(2, BLUE), (1, PINK), (0, TEAL)]:
        ax_t.plot([0, 122], [y, y], color=color, alpha=0.22, linewidth=1.4)

    def add_attempts(info: dict, y: float, color: str, marker: str) -> None:
        for item in info["all"]:
            x = float(item.get("elapsed_seconds", 0.0)) / 60.0
            if item.get("status") == "valid" and item.get("search_score") is not None:
                ax_t.scatter(x, y, s=75, marker=marker, facecolor="white", edgecolor=color, linewidth=2, zorder=3)
            else:
                ax_t.scatter(x, y, s=65, marker="x", color="#6E8798", linewidth=2, zorder=3)

    add_attempts(r1_info, 2, BLUE, "o")
    add_attempts(r2_info, 1, PINK, "s")
    ax_t.scatter(r3_minutes, 0, s=80, marker="D", facecolor="white", edgecolor=TEAL, linewidth=2, zorder=3)

    ax_b.axis("off")
    summaries = [
        (0.00, "R1（无记忆）", r1_info["best"], f"{r1_info['elapsed']:.1f} min · {len(r1_info['valid'])}/{len(r1_info['all'])} 认证", BLUE),
        (0.34, "R2（R1 记忆，最新重跑）", r2_info["best"], f"{r2_info['elapsed']:.1f} min · {len(r2_info['valid'])}/{len(r2_info['all'])} 认证", PINK),
        (0.69, "R3（累计记忆）", r3_best, f"{r3_minutes:.1f} min · 终态收据", TEAL),
    ]
    for x, title, score, detail, color in summaries:
        ax_b.text(x, 0.76, title, transform=ax_b.transAxes, color="#758DA0", fontsize=11, ha="left")
        ax_b.text(x, 0.30, f"{score:.5f}", transform=ax_b.transAxes, color=color, fontsize=25, ha="left")
        ax_b.text(x + 0.19, 0.33, detail, transform=ax_b.transAxes, color="#758DA0", fontsize=10, ha="left")
    ax_b.text(0.0, 0.02, "阶梯线 = 已记录的候选结果；空心点 = Host 认证；× = 未认证或生成失败；R3 中间候选未回传，未做推断。", transform=ax_b.transAxes, color="#758DA0", fontsize=9.5, ha="left")

    ax_t.set_xlabel("距离各轮启动的时间 / 分钟（初始化、规划、训练与评分均计入）", color=TEXT, fontsize=11, labelpad=8)

    fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (STEM.with_name(STEM.name + "_audit.json")).write_text(
        json.dumps(
            {
                "figure_claim": "iMet best Host micro-F1 by cumulative-memory round",
                "r1_best": r1_info["best"],
                "r2_best": r2_info["best"],
                "r3_best": r3_best,
                "r3_representation": "terminal receipt only; intermediate trajectory unavailable locally",
                "sources": [
                    "imet/runs/imet-r1-20260920-a1/STATE.json",
                    "imet/runs/imet-r2-20260920-b1/STATE.json",
                    "orchestrator.log R3 terminal receipt",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
