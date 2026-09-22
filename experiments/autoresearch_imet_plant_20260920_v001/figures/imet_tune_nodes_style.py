#!/usr/bin/env python3
"""iMet internal-tune node figure.

This figure uses every persisted per-epoch internal trajectory point from R1/R2.
Candidate-final tune scores are shown as larger markers.  R3 is marked as having
no returned tune trajectory; its Host terminal score is kept in the summary only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
STEM = OUT / "imet_tune_nodes_style"

BLUE = "#3E83B3"
PINK = "#C85C7D"
TEAL = "#159B8F"
SLATE = "#6F879A"
GRID = "#DCE6ED"
TEXT = "#40576A"

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


def metadata(run: str, attempt: str) -> dict | None:
    path = ROOT / "imet" / "runs" / run / attempt / "worker" / "artifacts" / "metadata.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else None


def host_score(run: str) -> float | None:
    path = ROOT / "imet" / "runs" / run / "STATE.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    return float(state["best"]) if state.get("best") is not None else None


def r3_host_terminal() -> float:
    log = (ROOT / "orchestrator.log").read_text()
    matches = re.findall(r"imet terminal imet-r3-20260920-a1 best=([0-9.]+)", log)
    if not matches:
        raise RuntimeError("R3 terminal receipt not found")
    return float(matches[-1])


def trajectory_values(meta: dict) -> list[float]:
    values: list[float] = []
    for row in meta.get("trajectory") or []:
        if not isinstance(row, dict):
            continue
        value = None
        # R1 records a validation trajectory; R2 records tune micro-F1.
        for key in ("tune_micro_f1", "val_micro_f1_at_0.15", "val_micro_f1_at_0.20"):
            if row.get(key) is not None:
                value = float(row[key])
                break
        if value is not None:
            values.append(value)
    return values


def candidate(run: str, attempt: str, label: str, marker: str) -> dict | None:
    meta = metadata(run, attempt)
    if meta is None:
        return None
    values = trajectory_values(meta)
    if not values or meta.get("tune_micro_f1") is None:
        return None
    return {
        "label": label,
        "marker": marker,
        "trajectory": values,
        "final_tune": float(meta["tune_micro_f1"]),
    }


def main() -> None:
    r1_candidates = [
        candidate("imet-r1-20260920-a1", "attempt-001", "R1-001", "o"),
        candidate("imet-r1-20260920-a1", "attempt-002", "R1-002（Host失败）", "o"),
    ]
    r1_candidates = [c for c in r1_candidates if c is not None]
    r2_candidates = [
        candidate("imet-r2-20260920-b1", "attempt-000", "R2-000", "s"),
        candidate("imet-r2-20260920-b1", "attempt-001", "R2-001（Host淘汰）", "s"),
    ]
    r2_candidates = [c for c in r2_candidates if c is not None]
    r3_host = r3_host_terminal()

    fig = plt.figure(figsize=(12.0, 8.2), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.80, bottom=0.08)
    grid = fig.add_gridspec(3, 1, height_ratios=[5.1, 1.2, 2.15], hspace=0.30)
    ax = fig.add_subplot(grid[0])
    ax_t = fig.add_subplot(grid[1], sharex=ax)
    ax_b = fig.add_subplot(grid[2])

    fig.text(0.10, 0.958, "iMet / 内部 tune 节点轨迹", ha="left", va="top", fontsize=23, color="#263A4A")
    fig.text(
        0.10,
        0.916,
        "所有已回传的候选内 epoch 节点 · 大标记为候选最终 tune 分数",
        ha="left",
        va="top",
        color="#758DA0",
        fontsize=13,
    )
    fig.text(0.985, 0.942, "R3 tune 明细未回传", ha="right", va="top", color="#758DA0", fontsize=12)

    colors = {"R1": BLUE, "R2": PINK}
    markers = {"R1": "o", "R2": "s"}
    handles = [
        plt.Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white", markeredgewidth=2, linewidth=2.6, label="R1 · 12 个内部节点"),
        plt.Line2D([0], [0], color=PINK, marker="s", markerfacecolor="white", markeredgewidth=2, linewidth=2.6, label="R2 · 18 个内部节点"),
        plt.Line2D([0], [0], color="#263A4A", marker="*", markerfacecolor="white", markeredgewidth=1.3, linewidth=0, markersize=10, label="候选最终 tune 分数"),
        plt.Line2D([0], [0], color=TEAL, marker="D", markerfacecolor="white", markeredgewidth=2, linewidth=0, markersize=8, label="R3 · 无 tune 轨迹回传"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=4, frameon=False, fontsize=10)

    all_endpoints: dict[str, list[tuple[float, float, str]]] = {"R1": [], "R2": []}
    x_end = {"R1": 0.0, "R2": 0.0}
    for round_name, candidates in (("R1", r1_candidates), ("R2", r2_candidates)):
        color = colors[round_name]
        marker = markers[round_name]
        cursor = 1.0
        for cidx, c in enumerate(candidates):
            xs = list(range(int(cursor), int(cursor) + len(c["trajectory"])))
            ys = c["trajectory"]
            ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.86, zorder=2)
            ax.scatter(xs, ys, s=62, marker=marker, facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
            final_x = xs[-1] + 0.38
            ax.scatter(final_x, c["final_tune"], s=155, marker="*", facecolor="white", edgecolor=color, linewidth=1.8, zorder=6)
            ax.text(final_x + 0.25, c["final_tune"], f"{c['final_tune']:.5f}", color=color, fontsize=9.5, va="center", ha="left", bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=10)
            all_endpoints[round_name].append((final_x, c["final_tune"], c["label"]))
            ax.axvline(final_x + 0.62, color="#EDF2F5", linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
            cursor = final_x + 1.2
        x_end[round_name] = max(x_end[round_name], cursor)

    xmax = max(x_end.values()) + 0.8
    ax.set_xlim(0.5, xmax)
    ax.set_ylim(0.13, 0.47)
    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    ax.set_ylabel("内部 tune / validation micro-F1 ↑", fontsize=12)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    ax.text(xmax - 0.15, 0.462, f"R3 Host 终态 {r3_host:.5f}（非 tune）", color=TEAL, fontsize=10.5, ha="right", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    # Candidate-node strip: every epoch point is retained, with final tune stars.
    ax_t.set_ylim(-0.5, 2.5)
    ax_t.set_yticks([2, 1, 0], ["R1 · tune 节点", "R2 · tune 节点", "R3 · tune 未回传"])
    ax_t.tick_params(axis="y", length=0, labelsize=10)
    ticks = list(range(1, int(xmax) + 1))
    ax_t.set_xticks(ticks)
    ax_t.set_xticklabels([str(i) for i in ticks])
    ax_t.tick_params(axis="x", labelbottom=True, length=0, pad=5, labelsize=9)
    ax_t.grid(axis="x", color="#EDF2F5", linewidth=0.8)
    ax_t.set_axisbelow(True)
    for y, color in [(2, BLUE), (1, PINK), (0, TEAL)]:
        ax_t.plot([0.5, xmax], [y, y], color=color, alpha=0.22, linewidth=1.4)
    for round_y, round_name, candidates in [(2, "R1", r1_candidates), (1, "R2", r2_candidates)]:
        cursor = 1.0
        for c in candidates:
            xs = list(range(int(cursor), int(cursor) + len(c["trajectory"])))
            ax_t.scatter(xs, [round_y] * len(xs), s=55, marker=markers[round_name], facecolor="white", edgecolor=colors[round_name], linewidth=1.5, zorder=3)
            final_x = xs[-1] + 0.38
            ax_t.scatter(final_x, round_y, s=100, marker="*", facecolor="white", edgecolor=colors[round_name], linewidth=1.4, zorder=4)
            cursor = final_x + 1.2
    ax_t.text(xmax - 0.15, 0, "R3 仅有 Host 终态，无 tune 节点文件", ha="right", va="center", color=TEAL, fontsize=9.5)
    ax_t.set_xlabel("内部 tune 节点顺序（虚线分隔不同候选）", color=TEXT, fontsize=11, labelpad=8)

    ax_b.axis("off")
    summary = [
        (0.00, "R1", "12 个 epoch 节点", "候选 tune 终点 0.38823 / 0.30211", f"Host best {host_score('imet-r1-20260920-a1'):.5f}", BLUE),
        (0.34, "R2", "18 个 epoch 节点", "候选 tune 终点 0.43860 / 0.39477", f"Host best {host_score('imet-r2-20260920-b1'):.5f}", PINK),
        (0.69, "R3", "0 个 tune 节点回传", "只有终态收据", f"Host terminal {r3_host:.5f}", TEAL),
    ]
    for x, title, detail, detail2, detail3, color in summary:
        ax_b.text(x, 0.82, title, transform=ax_b.transAxes, color="#758DA0", fontsize=11, ha="left")
        ax_b.text(x, 0.48, detail, transform=ax_b.transAxes, color=color, fontsize=17, ha="left")
        ax_b.text(x, 0.25, detail2, transform=ax_b.transAxes, color="#758DA0", fontsize=10, ha="left")
        ax_b.text(x, 0.08, detail3, transform=ax_b.transAxes, color="#758DA0", fontsize=10, ha="left")
    ax_b.text(0.0, -0.06, "说明：R1 epoch 曲线来自候选的 validation micro-F1@0.15，R2 epoch 曲线来自候选的 tune micro-F1；大星号是候选最终 tune 分数。Host 分数使用隐藏 search 集，单独列出。", transform=ax_b.transAxes, color="#758DA0", fontsize=9.2, ha="left")

    fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    (STEM.with_name(STEM.name + "_audit.json")).write_text(
        json.dumps(
            {
                "figure_claim": "All persisted internal tune or validation trajectory nodes for iMet R1/R2",
                "r1_epoch_nodes": sum(len(c["trajectory"]) for c in r1_candidates),
                "r2_epoch_nodes": sum(len(c["trajectory"]) for c in r2_candidates),
                "r3_tune_trajectory": "not returned locally",
                "host_scores_separate": {
                    "r1": host_score("imet-r1-20260920-a1"),
                    "r2": host_score("imet-r2-20260920-b1"),
                    "r3": r3_host,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
