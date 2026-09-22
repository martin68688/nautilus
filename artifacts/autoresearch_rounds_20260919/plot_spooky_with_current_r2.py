from pathlib import Path
import csv
import json
from datetime import datetime, timezone

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


O = Path(__file__).resolve().parent
rows = list(csv.DictReader((O / "tune_observations.csv").open()))
current = json.loads((O / "current_r2_snapshot" / "STATE.json").read_text())
launch = json.loads((O / "current_r2_snapshot" / "LAUNCH.json").read_text())

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "Arial", "DejaVu Sans"],
        "font.size": 14,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#BCD0DC",
        "axes.labelcolor": "#344C59",
        "text.color": "#344C59",
        "xtick.color": "#8096AA",
        "ytick.color": "#8096AA",
        "axes.unicode_minus": False,
        "legend.frameon": False,
    }
)

colors = ["#377FAE", "#C86281", "#10988F", "#7A64B5"]
markers = ["o", "^", "D", "s"]
labels = [
    "R1  无记忆",
    "R2  读取 R1（历史）",
    "R3  读取 R1 + R2",
    "R2  当前重跑",
]

historical = []
for rnd in ["R1", "R2", "R3"]:
    rr = [r for r in rows if r["task"] == "Spooky" and r["round"] == rnd]
    y = np.array([float(r["value"]) for r in rr], dtype=float)
    x = np.array([float(r["progress_percent"]) for r in rr], dtype=float)
    assert np.isfinite(y).all() and (y >= 0).all()
    historical.append((x, y))

# The live rerun uses only explicit tune_loss values. Failed candidates and
# host search scores are kept out of this tune curve; the latter is reported
# separately in the footer.
live = [h for h in current["history"] if h.get("tune_loss") is not None]
live_y = np.array([float(h["tune_loss"]) for h in live], dtype=float)
live_x = np.linspace(0, 100, len(live)) if len(live) else np.array([])
assert len(live) > 0 and np.isfinite(live_y).all() and (live_y >= 0).all()

host_scores = [
    float(h["search_loss"])
    for h in current["history"]
    if h.get("search_loss") is not None
]
live_best_host = min(host_scores) if host_scores else float("nan")
live_best_tune = float(live_y.min())
last_event = current["history"][-1]
terminal_provider_stop = bool(last_event.get("provider_stop") or last_event.get("http_status"))
capture = "2026-09-20 00:54 UTC（最终轮询）"
elapsed = float(last_event.get("elapsed_seconds", 0.0))

fig = plt.figure(figsize=(20, 12.5), facecolor="white")
ax = fig.add_axes([0.08, 0.29, 0.76, 0.45])
fig.text(0.07, 0.94, "Spooky / 内部调参轨迹 · R1 / R2 / R3 + 当前 R2", fontsize=29)

groups = historical + [(live_x, live_y)]
for i, (x, y) in enumerate(groups):
    pos = 0.07 + i * 0.225
    line_style = "--" if i == 3 else "-"
    fig.add_artist(
        Line2D(
            [pos, pos + 0.026],
            [0.803, 0.803],
            transform=fig.transFigure,
            color=colors[i],
            lw=3,
            linestyle=line_style,
        )
    )
    fig.text(pos + 0.033, 0.797, labels[i], fontsize=16, color=colors[i])
    scatter_size = 22 if len(y) > 100 else 40
    ax.scatter(
        x,
        y,
        s=scatter_size,
        color=colors[i],
        marker=markers[i],
        alpha=0.30 if i < 3 else 0.85,
        zorder=3,
    )
    best = np.minimum.accumulate(y)
    ax.step(x, best, where="post", lw=2.5, color=colors[i], linestyle=line_style, zorder=4)
    last = int(np.where(y == best[-1])[0][0])
    ax.scatter(
        [x[last]],
        [best[-1]],
        s=170,
        marker="*",
        color=colors[i],
        edgecolor="white",
        linewidth=0.7,
        zorder=6,
    )

ax.set_xlim(-1.5, 101)
ax.set_ylim(0.30, 1.0)
ax.grid(axis="y", color="#E8EFF5", lw=1)
ax.set_axisbelow(True)
ax.set_xticks([0, 25, 50, 75, 100])
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g}%"))
ax.set_xlabel("本轮观测进度（首次 = 0%，最后一次 = 100%；当前 R2 为已评分候选序列）", labelpad=12, fontsize=15)
ax.set_ylabel("内部 tune log-loss ↓", fontsize=15)

vals = np.array([min(y) for _, y in groups])
span = ax.get_ylim()[1] - ax.get_ylim()[0]
positions = vals.copy()
order = np.argsort(vals)
for a, b in zip(order, order[1:]):
    positions[b] = max(positions[b], positions[a] + span * 0.052)
for i, v in enumerate(vals):
    suffix = "*" if i == 3 else ""
    ax.annotate(
        f"{labels[i].split('  ')[0]}  {v:.5f}{suffix}",
        xy=(100, v),
        xytext=(103, positions[i]),
        color=colors[i],
        fontsize=14,
        va="center",
        annotation_clip=False,
        arrowprops={"arrowstyle": "-", "color": colors[i], "lw": 1},
    )

ins = fig.add_axes([0.49, 0.465, 0.31, 0.205], facecolor="white")
ins.set_title("后期低损失区间 · 放大", fontsize=12, loc="left", color="#8096AA")
for i, (x, y) in enumerate(groups):
    ins.scatter(
        x,
        y,
        s=9 if i < 3 else 28,
        color=colors[i],
        marker=markers[i],
        alpha=0.23 if i < 3 else 0.85,
    )
    ins.step(
        x,
        np.minimum.accumulate(y),
        where="post",
        color=colors[i],
        lw=1.7,
        linestyle="--" if i == 3 else "-",
    )
ins.set_ylim(0.325, 0.365)
ins.set_xlim(-1, 101)
ins.set_xticks([0, 50, 100])
ins.set_xticklabels(["0%", "50%", "100%"])
ins.tick_params(labelsize=10)
ins.yaxis.set_major_locator(MaxNLocator(4))
ins.grid(axis="y", color="#E8EFF5")
ins.spines[["top", "right"]].set_visible(True)

footer = [
    ("R1 Host", 0.35028213509959194, colors[0]),
    ("R2 历史 Host", 0.3511051102229948, colors[1]),
    ("R3 Host", 0.34911176412385214, colors[2]),
    ("R2 重跑 Host", live_best_host, colors[3]),
]
for i, (title, value, color) in enumerate(footer):
    pos = 0.07 + i * 0.225
    fig.text(pos, 0.165, title, fontsize=13, color="#8096AA")
    fig.text(pos, 0.12, f"{value:.5f}", fontsize=25, color=color)

stem = O / "spooky_rounds_with_current_r2_clean"
fig.savefig(stem.with_suffix(".png"), dpi=300)
fig.savefig(stem.with_suffix(".svg"))
fig.savefig(stem.with_suffix(".pdf"))
plt.close(fig)

from PIL import Image

im = Image.open(stem.with_suffix(".png"))
im.resize((2000, 1250), Image.Resampling.LANCZOS).save(stem.with_name(stem.name + "_preview").with_suffix(".png"))

summary = {
    "captured_utc": capture,
    "current_r2_attempts_in_state": len(current["history"]),
    "current_r2_scored_candidates": len(live),
    "current_r2_tune_best": live_best_tune,
    "current_r2_host_best": live_best_host,
    "current_r2_search_scores": host_scores,
    "current_r2_best_attempt": current["best_attempt"],
    "current_r2_run_best_search": current["best"],
    "current_r2_terminal": terminal_provider_stop,
    "current_r2_stop_reason": "HTTP 403 Forbidden" if terminal_provider_stop else None,
    "source_snapshot": "current_r2_snapshot/STATE.json",
}
(O / "SPOOKY_CURRENT_R2_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
