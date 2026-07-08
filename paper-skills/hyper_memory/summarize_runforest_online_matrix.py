#!/usr/bin/env python3
"""Summarize an online Run-Forest memory matrix run."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def iter_nodes(run_dir: Path) -> list[dict[str, Any]]:
    for name in ["filtered_journal.json", "journal.json"]:
        data = load_json(run_dir / "logs" / name)
        if not data:
            continue
        if isinstance(data, dict) and isinstance(data.get("nodes"), list):
            return [n for n in data["nodes"] if isinstance(n, dict)]
        if isinstance(data, list):
            return [n for n in data if isinstance(n, dict)]
    return []


def metric_value(node: dict[str, Any]) -> tuple[float | None, bool | None]:
    metric = node.get("metric")
    if isinstance(metric, dict):
        value = metric.get("value")
        maximize = metric.get("maximize")
    else:
        value = metric
        maximize = None
    try:
        value_f = float(value)
    except Exception:
        return None, maximize if isinstance(maximize, bool) else None
    if not math.isfinite(value_f):
        return None, maximize if isinstance(maximize, bool) else None
    return value_f, maximize if isinstance(maximize, bool) else None


def best_metric(run_dir: Path) -> dict[str, Any]:
    nodes = iter_nodes(run_dir)
    values = []
    maximize = None
    for node in nodes:
        value, node_maximize = metric_value(node)
        if value is None:
            continue
        if maximize is None and node_maximize is not None:
            maximize = node_maximize
        values.append((value, node.get("id"), node.get("stage"), node.get("is_buggy")))
    if not values:
        return {"value": None, "maximize": maximize, "node_count": len(nodes)}
    maximize = True if maximize is None else maximize
    best = max(values, key=lambda x: x[0]) if maximize else min(values, key=lambda x: x[0])
    return {
        "value": best[0],
        "maximize": maximize,
        "node_id": best[1],
        "stage": best[2],
        "is_buggy": best[3],
        "node_count": len(nodes),
        "metric_count": len(values),
    }


def adoption_summary(run_dir: Path) -> dict[str, Any]:
    report = load_json(run_dir / "logs" / "adoption_report.json")
    if not isinstance(report, dict):
        return {"available": False}
    summary = report.get("summary", {})
    return {
        "available": True,
        "overall": summary.get("overall_adoption_rate"),
        "external": summary.get("external_memory", {}),
        "by_source": summary.get("by_source", {}),
    }


def navigation_summary(run_dir: Path) -> dict[str, Any]:
    log_files = sorted((run_dir / "logs").glob("*.log"))
    text = "\n".join(p.read_text(encoding="utf-8", errors="ignore")[-500000:] for p in log_files[:5])
    if not text:
        for p in sorted(run_dir.glob("**/*.log"))[:10]:
            text += "\n" + p.read_text(encoding="utf-8", errors="ignore")[-200000:]
    return {
        "runforest_log_lines": len(re.findall(r"\\[RunForestMemory\\]", text)),
        "llm_fallbacks": len(re.findall(r"LLM navigator failed", text)),
        "draft_strategy": len(re.findall(r"draft_successful_branches", text)),
        "improve_strategy": len(re.findall(r"improve_local_best_lineage", text)),
        "debug_strategy": len(re.findall(r"debug_failure_recovery", text)),
    }


def task_from_config(run_dir: Path) -> str:
    text = (run_dir / "logs" / "config.yaml").read_text(encoding="utf-8", errors="ignore") if (run_dir / "logs" / "config.yaml").exists() else ""
    match = re.search(r"^exp_id:\\s*['\"]?([^'\"\\n]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def historical_baselines(runs_dir: Path, task: str, exclude_tag: str) -> list[dict[str, Any]]:
    rows = []
    for run_dir in sorted(runs_dir.glob(f"*{task}*")):
        if exclude_tag and exclude_tag in run_dir.name:
            continue
        cfg = run_dir / "logs" / "config.yaml"
        if not cfg.exists():
            continue
        cfg_text = cfg.read_text(encoding="utf-8", errors="ignore")
        if "run_forest_agentic_memory" in cfg_text or "mode: run_forest" in cfg_text:
            continue
        if task_from_config(run_dir) not in {"", task}:
            continue
        metric = best_metric(run_dir)
        rows.append({
            "run_dir": str(run_dir),
            "best_metric": metric,
            "adoption": adoption_summary(run_dir),
        })
    metric_rows = [r for r in rows if r["best_metric"].get("value") is not None]
    if not metric_rows:
        return rows[:5]
    maximize = metric_rows[0]["best_metric"].get("maximize")
    metric_rows.sort(key=lambda r: r["best_metric"]["value"], reverse=bool(maximize))
    return metric_rows[:5]


def latest_manifest(runs_dir: Path) -> Path:
    manifests = sorted(runs_dir.glob("*_matrix/runforest_online_manifest.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not manifests:
        raise FileNotFoundError("No runforest_online_manifest.jsonl found")
    return manifests[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    manifest = Path(args.manifest).resolve() if args.manifest else latest_manifest(runs_dir)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    tag = rows[0].get("tag", manifest.parent.name) if rows else manifest.parent.name
    results = []
    for row in rows:
        run_dir = Path(row.get("run_dir") or "")
        current = {
            "manifest_row": row,
            "task": row.get("task"),
            "run_dir": str(run_dir) if run_dir else "",
            "best_metric": best_metric(run_dir) if run_dir.exists() else {"value": None},
            "adoption": adoption_summary(run_dir) if run_dir.exists() else {"available": False},
            "navigation": navigation_summary(run_dir) if run_dir.exists() else {},
            "historical_baselines": historical_baselines(runs_dir, row.get("task", ""), tag),
        }
        results.append(current)

    report = {
        "schema": "runforest_online_matrix_summary_v1",
        "tag": tag,
        "manifest": str(manifest),
        "results": results,
    }
    out_json = Path(args.out_json) if args.out_json else manifest.parent / "runforest_online_summary.json"
    out_md = Path(args.out_md) if args.out_md else manifest.parent / "runforest_online_summary.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Run-Forest Online Memory Matrix Summary", "", f"- tag: `{tag}`", f"- manifest: `{manifest}`", ""]
    for item in results:
        metric = item["best_metric"]
        adoption = item["adoption"]
        nav = item["navigation"]
        lines.extend([
            f"## {item['task']}",
            f"- status: `{item['manifest_row'].get('status')}` returncode={item['manifest_row'].get('returncode')}",
            f"- run dir: `{item['run_dir']}`",
            f"- best metric: `{metric.get('value')}` maximize={metric.get('maximize')} node_count={metric.get('node_count')}",
            f"- adoption available: `{adoption.get('available')}` external={adoption.get('external')}",
            f"- navigation: {nav}",
            "- historical baselines:",
        ])
        for base in item["historical_baselines"]:
            lines.append(f"  - `{base['run_dir']}` best={base['best_metric'].get('value')} maximize={base['best_metric'].get('maximize')}")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
