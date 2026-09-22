#!/usr/bin/env python3
"""Reconstruct iMet trajectories while separating scored and unscored nodes.

Every persisted node remains in the source CSV and candidate strip.  The main
trajectory panel only connects candidate-local nodes whose candidate produced
a reload-verified Host score.  Candidates that could not refresh a Host score
are represented independently in the lower strip, so a failed attempt can
never look like a valid improvement on the main line.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
STEM = OUT / "imet_complete_data"

BLUE = "#3E83B3"
PINK = "#C85C7D"
TEAL = "#159B8F"
SLATE = "#6F879A"
GRID = "#DCE6ED"
TEXT = "#40576A"
PALE = "#EDF2F5"

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


RUNS = {
    "R1": ROOT / "imet" / "runs" / "imet-r1-20260920-a1",
    "R2": ROOT / "imet" / "runs" / "imet-r2-20260920-b1",
    # R3 was present on the AWS EBS run root but had not been returned to the
    # Mac when the first chart was drawn.  The recovery root is intentionally
    # separate from the live run roots and is read-only for this plotter.
    "R3": ROOT / "imet" / "remote-recovery" / "r3-scored-artifacts" / "imet-r3-20260920-a1",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata(run_path: Path, attempt: str) -> dict | None:
    for filename in ("metadata.json", "meta.json"):
        path = run_path / attempt / "worker" / "artifacts" / filename
        if path.exists():
            return read_json(path)
    return None


def result(run_path: Path, attempt: str) -> dict:
    path = run_path / attempt / "RESULT.json"
    return read_json(path) if path.exists() else {}


def r3_terminal() -> tuple[float, float]:
    recovered = RUNS["R3"] / "STATE.json"
    outcome = RUNS["R3"] / "RUN_OUTCOME.json"
    if recovered.exists() and outcome.exists():
        state_doc = read_json(recovered)
        outcome_doc = read_json(outcome)
        end = max((float(h.get("elapsed_seconds", 0.0)) for h in state_doc.get("history", [])), default=0.0)
        return end / 60.0, float(outcome_doc["best_search"])
    log = (ROOT / "orchestrator.log").read_text(encoding="utf-8")
    matches = re.findall(
        r"imet terminal imet-r3-20260920-a1 best=([0-9.]+) elapsed=([0-9.]+)", log
    )
    if not matches:
        raise RuntimeError("R3 terminal receipt not found")
    score, elapsed = matches[-1]
    return float(elapsed) / 60.0, float(score)


def trajectory_row(round_name: str, run: str, attempt: str, index: int, row: dict) -> dict:
    if round_name == "R1":
        metric_name = "validation micro-F1@0.15"
        metric = row.get("val_micro_f1_at_0.15")
        loss = row.get("train_bce")
        loss_name = "train BCE"
    else:
        metric_name = "tune micro-F1"
        metric = row.get("tune_micro_f1")
        loss = row.get("train_loss")
        loss_name = "train loss"
    return {
        "round": round_name,
        "run": run,
        "attempt": attempt,
        "epoch": row.get("epoch", index + 1),
        "metric_name": metric_name,
        "metric": metric,
        "loss_name": loss_name,
        "loss": loss,
        "tune_micro_f1": None,
        "host_search_score": None,
        "status": "internal_epoch",
    }


def collect() -> tuple[list[dict], list[dict], dict]:
    rows: list[dict] = []
    candidates: list[dict] = []
    for round_name, run_path in RUNS.items():
        if not run_path.exists():
            continue
        for attempt_dir in sorted(run_path.glob("attempt-*")):
            attempt = attempt_dir.name
            res = result(run_path, attempt)
            meta = metadata(run_path, attempt)
            status = res.get("status", "unknown")
            if meta is not None:
                trajectory = meta.get("trajectory") or []
                for idx, row in enumerate(trajectory):
                    if isinstance(row, dict):
                        rows.append(trajectory_row(round_name, run_path.name, attempt, idx, row))
                tune = meta.get("tune_micro_f1")
                if tune is not None:
                    rows.append(
                        {
                            "round": round_name,
                            "run": run_path.name,
                            "attempt": attempt,
                            "epoch": "final",
                            "metric_name": "candidate final tune micro-F1",
                            "metric": float(tune),
                            "loss_name": "",
                            "loss": None,
                            "tune_micro_f1": float(tune),
                            "host_search_score": res.get("search_score"),
                            "status": "candidate_final_tune",
                        }
                    )
                    candidates.append(
                        {
                            "round": round_name,
                            "run": run_path.name,
                            "attempt": attempt,
                            "status": status,
                            "started_elapsed_min": float(res.get("started_elapsed", 0.0)) / 60.0,
                            "ended_elapsed_min": float(res.get("elapsed_seconds", 0.0)) / 60.0,
                            "duration_min": float(res.get("seconds", 0.0)) / 60.0,
                            "tune_micro_f1": float(tune),
                            "host_search_score": res.get("search_score"),
                            "host_recomputed": res.get("host_recomputed", False),
                            "reload_verified": res.get("reload_verified", False),
                            "trajectory_count": len(trajectory),
                        }
                    )
            else:
                # Preserve every attempted candidate in the source table even
                # when it has no returned score or epoch metadata.
                candidates.append(
                    {
                        "round": round_name,
                        "run": run_path.name,
                        "attempt": attempt,
                        "status": status,
                        "started_elapsed_min": float(res.get("started_elapsed", 0.0)) / 60.0,
                        "ended_elapsed_min": float(res.get("elapsed_seconds", 0.0)) / 60.0,
                        "duration_min": float(res.get("seconds", 0.0)) / 60.0,
                        "tune_micro_f1": None,
                        "host_search_score": res.get("search_score"),
                        "host_recomputed": res.get("host_recomputed", False),
                        "reload_verified": res.get("reload_verified", False),
                        "trajectory_count": 0,
                    }
                )

    for c in candidates:
        if c["host_search_score"] is not None:
            rows.append(
                {
                    "round": c["round"],
                    "run": c["run"],
                    "attempt": c["attempt"],
                    "epoch": "Host",
                    "metric_name": "Host search micro-F1",
                    "metric": float(c["host_search_score"]),
                    "loss_name": "",
                    "loss": None,
                    "tune_micro_f1": c["tune_micro_f1"],
                    "host_search_score": float(c["host_search_score"]),
                    "status": "host_validated",
                }
            )

    round_order = {"R1": 0, "R2": 1, "R3": 2}
    def epoch_order(value: object) -> tuple[int, float]:
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (1, 0.0)
    rows.sort(key=lambda x: (round_order.get(x["round"], 99), str(x["attempt"]), epoch_order(x["epoch"])))
    r3_min, r3_score = r3_terminal()
    meta = {
        "r3_elapsed_min": r3_min,
        "r3_host_score": r3_score,
        "r3_detail_status": "candidate-level scored artifacts recovered from AWS EBS; large binary prediction/model caches remain on AWS",
    }
    return rows, candidates, meta


def write_source(rows: list[dict], candidates: list[dict], meta: dict) -> None:
    node_path = STEM.with_name(STEM.name + "_nodes.csv")
    fields = [
        "round", "run", "attempt", "epoch", "metric_name", "metric",
        "loss_name", "loss", "tune_micro_f1", "host_search_score", "status",
    ]
    with node_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    cand_path = STEM.with_name(STEM.name + "_candidates.csv")
    cfields = list(candidates[0].keys()) if candidates else []
    with cand_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cfields)
        w.writeheader()
        w.writerows(candidates)
    audit = {
        "figure_contract": {
            "core_conclusion": "The main trajectories include only reload-verified Host-scored candidates; unscored or failed candidates stay in the independent candidate strip and source CSV.",
            "archetype": "asymmetric mixed-modality quantitative composite",
            "backend": "python/matplotlib",
            "hero_evidence": "all returned internal epoch nodes",
            "validation_evidence": "Host search scores and terminal R3 receipt",
            "review_risk": "R3 scoring artifacts are recovered; large binary prediction/model caches remain on AWS and are not needed for this figure",
        },
        "node_count": len(rows),
        "candidate_count": len(candidates),
        "main_panel_policy": "only candidates with host_search_score and host_recomputed and reload_verified",
        "main_panel_candidates": [
            f"{c['round']}-{c['attempt'][-3:]}"
            for c in candidates
            if c["host_search_score"] is not None and c["host_recomputed"] and c["reload_verified"]
        ],
        "independent_unscored_candidates": [
            f"{c['round']}-{c['attempt'][-3:]}"
            for c in candidates
            if not (c["host_search_score"] is not None and c["host_recomputed"] and c["reload_verified"])
        ],
        "r1_internal_nodes": sum(1 for x in rows if x["round"] == "R1" and x["status"] == "internal_epoch"),
        "r2_internal_nodes": sum(1 for x in rows if x["round"] == "R2" and x["status"] == "internal_epoch"),
        "r3": meta,
        "sources": [
            "R1/R2 STATE.json, RESULT.json and worker/artifacts/metadata.json",
            "AWS-recovered R3 STATE.json, RESULT.json, HOST_RESULT.json and worker/artifacts/meta.json",
        ],
    }
    (STEM.with_name(STEM.name + "_audit.json")).write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_internal_panel(ax, rows: list[dict], candidates: list[dict]) -> None:
    colors = {"R1": BLUE, "R2": PINK, "R3": TEAL}
    markers = {"R1": "o", "R2": "s", "R3": "D"}
    x_cursor = {"R1": 1.0, "R2": 1.0, "R3": 1.0}
    endpoint_text: list[tuple[float, float, str, str]] = []
    for round_name in ("R1", "R2", "R3"):
        color = colors[round_name]
        marker = markers[round_name]
        # Reserve x-space for every candidate so the main panel and the
        # independent strip share the same candidate order.  Only a candidate
        # with a persisted, reload-verified Host score gets a line here.
        run_candidates = [c for c in candidates if c["round"] == round_name and c["trajectory_count"]]
        valid_blocks: list[list[tuple[float, float]]] = []
        current_block: list[tuple[float, float]] = []
        for c in run_candidates:
            local_rows = [
                r for r in rows
                if r["round"] == round_name and r["run"] == c["run"]
                and r["attempt"] == c["attempt"] and r["status"] == "internal_epoch"
            ]
            ys = [float(r["metric"]) for r in local_rows]
            xs = list(range(int(x_cursor[round_name]), int(x_cursor[round_name]) + len(ys)))
            scored = (
                c["host_search_score"] is not None
                and c["host_recomputed"]
                and c["reload_verified"]
            )
            if scored:
                # Consecutive scored candidates in one round form one
                # staircase, matching the Leaf reference.  A failed/unscored
                # candidate closes the current block and is never bridged.
                current_block.extend(zip(xs, ys))
                ax.scatter(xs, ys, s=62, marker=marker, facecolor="white", edgecolor=color, linewidth=1.8, zorder=4)
            if scored and c["tune_micro_f1"] is not None:
                final_x = xs[-1] + 0.38
                final_y = float(c["tune_micro_f1"])
                ax.scatter(final_x, final_y, s=155, marker="*", facecolor="white", edgecolor=color, linewidth=1.8, zorder=6)
                endpoint_text.append((final_x, final_y, f"{round_name}-{c['attempt'][-3:]} {final_y:.5f}", color))
            elif current_block:
                valid_blocks.append(current_block)
                current_block = []
            # Failed/unscored candidates deliberately leave a blank block on
            # the main panel; their marker appears only in the lower strip.
            x_cursor[round_name] = (xs[-1] + 1.55) if xs else x_cursor[round_name]
            ax.axvline(x_cursor[round_name] - 0.62, color=PALE, linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
        if current_block:
            valid_blocks.append(current_block)

        # Match the Leaf reference: hold each node's score until the next
        # node, then jump vertically.  The small extension after the last
        # node makes the terminal plateau visible beside the final-tune star.
        for block in valid_blocks:
            step_x = [x for x, _ in block]
            step_y = [y for _, y in block]
            step_x.append(step_x[-1] + 0.9)
            step_y.append(step_y[-1])
            ax.step(step_x, step_y, where="post", color=color, linewidth=2.6, alpha=0.86, zorder=2)

    for x, y, text, color in endpoint_text:
        dy = 0.004 if y > 0.36 else -0.004
        ax.text(x + 0.20, y + dy, text, color=color, fontsize=9.5, va="center", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=10)

    xmax = max(x_cursor.values()) + 0.8
    ax.set_xlim(0.5, xmax)
    ax.set_ylim(0.13, 0.47)
    ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
    ax.set_ylabel("候选内 tune / validation micro-F1 ↑", fontsize=11)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelbottom=False, length=0)
    return xmax


def add_status_strip(ax, xmax: float, candidates: list[dict], r3_min: float) -> None:
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([2, 1, 0], ["R1", "R2", "R3"])
    ax.tick_params(axis="y", length=0, labelsize=10)
    ticks = list(range(1, int(xmax) + 1))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(i) for i in ticks])
    ax.tick_params(axis="x", labelbottom=True, length=0, pad=5, labelsize=9)
    ax.grid(axis="x", color=PALE, linewidth=0.8)
    ax.set_axisbelow(True)
    for y, color in [(2, BLUE), (1, PINK), (0, TEAL)]:
        ax.plot([0.5, xmax], [y, y], color=color, alpha=0.22, linewidth=1.4)
    for round_name in ("R1", "R2", "R3"):
        # Each lane follows the same within-round node coordinate as the
        # upper panel; lanes are independent so R1/R2/R3 remain comparable.
        pos = 1.0
        round_candidates = [c for c in candidates if c["round"] == round_name]
        y = {"R1": 2, "R2": 1, "R3": 0}[round_name]
        color = {"R1": BLUE, "R2": PINK, "R3": TEAL}[round_name]
        marker = {"R1": "o", "R2": "s", "R3": "D"}[round_name]
        for c in round_candidates:
            n = max(1, int(c["trajectory_count"]))
            x = pos + n - 0.5
            if c["host_search_score"] is not None:
                ax.scatter(x, y, s=75, marker=marker, facecolor="white", edgecolor=color, linewidth=2, zorder=3)
            else:
                ax.scatter(x, y, s=65, marker="x", color=SLATE, linewidth=2, zorder=3)
            pos += n + 1.55
    ax.set_xlabel("候选节点顺序（虚线分隔不同候选；× = 无可用评分）", color=TEXT, fontsize=10.5, labelpad=8)


def add_host_panel(ax, candidates: list[dict], r3_min: float, r3_score: float) -> None:
    colors = {"R1": BLUE, "R2": PINK, "R3": TEAL}
    markers = {"R1": "o", "R2": "s", "R3": "D"}
    for round_name in ("R1", "R2", "R3"):
        color = colors[round_name]
        marker = markers[round_name]
        valid = [c for c in candidates if c["round"] == round_name and c["host_search_score"] is not None]
        invalid = [c for c in candidates if c["round"] == round_name and c["host_search_score"] is None]
        for c in valid:
            x = c["ended_elapsed_min"]
            y = float(c["host_search_score"])
            ax.scatter(x, y, s=115, marker=marker, facecolor="white", edgecolor=color, linewidth=2.2, zorder=5)
            ax.text(x + 1.0, y + (0.003 if round_name == "R1" else -0.004), f"{y:.5f}", color=color,
                    fontsize=9.5, ha="left", va="center", bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=10)
        for c in invalid:
            x = c["ended_elapsed_min"]
            ax.scatter(x, 0.352, s=75, marker="x", color=SLATE, linewidth=2, zorder=4)
    if not any(c["round"] == "R3" and c["host_search_score"] is not None for c in candidates):
        ax.scatter(r3_min, r3_score, s=135, marker="D", facecolor="white", edgecolor=TEAL, linewidth=2.4, zorder=6)
        ax.text(r3_min - 1.0, r3_score + 0.004, f"{r3_score:.5f}", color=TEAL, fontsize=10, ha="right", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=10)
    ax.axvline(r3_min, color=TEAL, linewidth=0.8, linestyle=(0, (3, 4)), alpha=0.55, zorder=0)
    ax.text(r3_min - 1.0, 0.477, f"R3 轮终点 {r3_min:.1f} min", color=TEAL, fontsize=9.2,
            ha="right", va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1.0), zorder=10)
    ax.set_xlim(0, max(122, r3_min + 3))
    ax.set_ylim(0.35, 0.48)
    ax.set_yticks([0.36, 0.39, 0.42, 0.45, 0.48])
    ax.set_ylabel("Host search micro-F1 ↑", fontsize=11)
    ax.set_xlabel("距离各轮启动的真实运行时间 / 分钟", fontsize=10.5, labelpad=7)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main() -> None:
    rows, candidates, meta = collect()
    write_source(rows, candidates, meta)
    r3_min = meta["r3_elapsed_min"]
    r3_score = meta["r3_host_score"]

    fig = plt.figure(figsize=(12.0, 9.2), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.985, top=0.79, bottom=0.08)
    grid = fig.add_gridspec(3, 1, height_ratios=[4.8, 1.05, 2.1], hspace=0.31)
    ax_internal = fig.add_subplot(grid[0])
    ax_strip = fig.add_subplot(grid[1], sharex=ax_internal)
    ax_host = fig.add_subplot(grid[2])

    fig.text(0.10, 0.958, "iMet / 全部可回收节点与 Host 结果", ha="left", va="top", fontsize=23, color="#263A4A")
    fig.text(0.10, 0.918, "同一轮的有效节点合并为直角阶梯线；无成绩或失败候选独立放在下方", ha="left", va="top", color="#758DA0", fontsize=13)
    fig.text(0.985, 0.942, "R3 两个候选的评分数据已回收", ha="right", va="top", color="#758DA0", fontsize=12)

    handles = [
        Line2D([0], [0], color=BLUE, marker="o", markerfacecolor="white", markeredgewidth=2, linewidth=2.6, label="R1 · 12 个内部节点（1 个可刷新候选）"),
        Line2D([0], [0], color=PINK, marker="s", markerfacecolor="white", markeredgewidth=2, linewidth=2.6, label="R2 · 18 个内部节点（2 个可刷新候选）"),
        Line2D([0], [0], color=TEAL, marker="D", markerfacecolor="white", markeredgewidth=2, linewidth=2.6, label="R3 · 20 个内部节点（1 个可刷新候选）"),
        Line2D([0], [0], color="#263A4A", marker="*", markerfacecolor="white", markeredgewidth=1.3, linewidth=0, markersize=10, label="候选最终 tune 分数"),
        Line2D([0], [0], color=TEAL, marker="D", markerfacecolor="white", markeredgewidth=2, linewidth=0, markersize=8, label="Host search 分数"),
        Line2D([0], [0], color=SLATE, marker="x", linestyle="None", markersize=8, markeredgewidth=2, label="无评分 / 失败候选"),
    ]
    ax_internal.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=6, frameon=False, fontsize=9.0)
    xmax = add_internal_panel(ax_internal, rows, candidates)
    add_status_strip(ax_strip, xmax, candidates, r3_min)
    add_host_panel(ax_host, candidates, r3_min, r3_score)

    # A compact summary preserves the distinction between internal and Host metrics.
    fig.text(0.10, 0.016, "内部节点：R1 使用 validation micro-F1@0.15，R2/R3 使用 tune micro-F1；主线只保留 reload+Host 验证通过的候选，大星号为其最终 tune。无成绩候选只在下方以 × 独立显示。", color="#758DA0", fontsize=9.3, ha="left")

    fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
