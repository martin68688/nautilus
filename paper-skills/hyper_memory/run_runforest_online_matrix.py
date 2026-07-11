#!/usr/bin/env python3
"""Run a multi-task online pilot for Agentic Run-Forest memory.

The script is intended for the NRP Job. It runs the memory-enabled condition
across several existing Kaggle-style tasks and records a machine-readable
manifest so the monitor/summarizer can compare against historical runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


TASKS = [
    "spooky-author-identification",
    "aerial-cactus-identification",
    "leaf-classification",
    "new-york-city-taxi-fare-prediction",
]

CONDITIONS = {
    "no_memory": [
        "external_skill_memory.enable=False",
        "agent.draft_role_policy.enabled=False",
    ],
    "sop_only": [
        "external_skill_memory.enable=True",
        "external_skill_memory.mode=run_forest_stage_hybrid",
        "external_skill_memory.source_name=run_forest_stage_hybrid_memory",
        "external_skill_memory.scoring_mode=poincare",
        "external_skill_memory.retrieval_control=sop_only",
    ],
    "tree_only": [
        "external_skill_memory.enable=True",
        "external_skill_memory.mode=run_forest_stage_hybrid",
        "external_skill_memory.source_name=run_forest_stage_hybrid_memory",
        "external_skill_memory.scoring_mode=poincare",
        "external_skill_memory.retrieval_control=tree_only",
    ],
    "naive_concat": [
        "external_skill_memory.enable=True",
        "external_skill_memory.mode=run_forest_stage_hybrid",
        "external_skill_memory.source_name=run_forest_stage_hybrid_memory",
        "external_skill_memory.scoring_mode=poincare",
        "external_skill_memory.retrieval_control=naive_concat",
    ],
    "stage_hybrid": [
        "external_skill_memory.enable=True",
        "external_skill_memory.mode=run_forest_stage_hybrid",
        "external_skill_memory.source_name=run_forest_stage_hybrid_memory",
        "external_skill_memory.scoring_mode=poincare",
        "external_skill_memory.retrieval_control=stage_hybrid",
    ],
    "layered_strategy": [
        "external_skill_memory.enable=True",
        "external_skill_memory.mode=run_forest_stage_hybrid",
        "external_skill_memory.source_name=run_forest_stage_hybrid_memory",
        "external_skill_memory.scoring_mode=poincare",
        "external_skill_memory.retrieval_control=layered_strategy",
    ],
    "flat_twin_hybrid": [
        "external_skill_memory.enable=True",
        "external_skill_memory.mode=run_forest_stage_hybrid",
        "external_skill_memory.source_name=run_forest_stage_hybrid_memory",
        "external_skill_memory.scoring_mode=flat_twin",
        "external_skill_memory.retrieval_control=stage_hybrid",
    ],
    "independent_euclidean": [
        "external_skill_memory.enable=True",
        "external_skill_memory.mode=run_forest_stage_hybrid",
        "external_skill_memory.source_name=run_forest_stage_hybrid_memory",
        "external_skill_memory.scoring_mode=euclidean",
        "external_skill_memory.retrieval_control=stage_hybrid",
    ],
}


def stream_command(cmd: list[str], log_path: Path, cwd: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return proc.wait()


def find_newest_run(runs_dir: Path, marker: str, start_time: float) -> str:
    candidates = []
    for path in runs_dir.glob(f"*{marker}*"):
        if path.is_dir() and path.stat().st_mtime >= start_time - 5:
            candidates.append(path)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--tag", default=os.environ.get("RUNFOREST_ONLINE_TAG") or datetime.utcnow().strftime("runforest_online_%Y%m%d_%H%M%S"))
    parser.add_argument("--num-gpus", type=int, default=int(os.environ.get("RUNFOREST_NUM_GPUS", "4")))
    parser.add_argument("--cpu-number", type=int, default=int(os.environ.get("RUNFOREST_CPU_NUMBER", "12")))
    parser.add_argument("--steps", type=int, default=int(os.environ["RUNFOREST_STEPS"]) if os.environ.get("RUNFOREST_STEPS") else None)
    parser.add_argument("--config-path", default="./config/config_run_forest_stage_hybrid.yaml")
    parser.add_argument("--conditions", default="layered_strategy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", default=os.environ.get("RUNFOREST_RUNS_DIR", ""))
    args = parser.parse_args()

    mlevolve_dir = Path.cwd().resolve()
    runs_dir = Path(args.runs_dir).resolve() if args.runs_dir else mlevolve_dir / "runs"
    matrix_dir = runs_dir / f"{args.tag}_matrix"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = matrix_dir / "runforest_online_manifest.jsonl"

    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    condition_names = [name.strip() for name in args.conditions.split(",") if name.strip()]
    unknown_conditions = sorted(set(condition_names) - set(CONDITIONS))
    if unknown_conditions:
        raise SystemExit(f"Unknown conditions: {unknown_conditions}; choices={sorted(CONDITIONS)}")
    print(f"[matrix] tag={args.tag}")
    print(f"[matrix] tasks={task_names}")
    print(f"[matrix] conditions={condition_names}")
    print(f"[matrix] num_gpus={args.num_gpus} cpu_number={args.cpu_number} steps={args.steps or 'config-default'}")
    print(f"[matrix] manifest={manifest_path}")

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for task in task_names:
            for condition in condition_names:
                data_dir = mlevolve_dir / "data" / task / "prepared" / "public"
                desc_file = data_dir / "description.md"
                marker = f"{args.tag}_{task}_{condition}"
                row = {
                    "tag": args.tag,
                    "task": task,
                    "condition": condition,
                    "marker": marker,
                    "start_unix": time.time(),
                    "data_dir": str(data_dir),
                    "desc_file": str(desc_file),
                    "config_path": args.config_path,
                }
                if not desc_file.exists():
                    row.update({"status": "skipped", "reason": "missing description.md"})
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    manifest.flush()
                    print(f"[matrix] SKIP {task}: missing {desc_file}")
                    continue

                cmd = [
                    sys.executable,
                    "run.py",
                    f"exp_id={task}",
                    "dataset_dir=./data",
                    f"data_dir=./data/{task}/prepared/public",
                    f"desc_file=./data/{task}/prepared/public/description.md",
                    f"exp_name={marker}",
                    f"agent.search.num_gpus={args.num_gpus}",
                    f"agent.search.parallel_search_num={args.num_gpus}",
                    f"cpu_number={args.cpu_number}",
                    f"log_dir={runs_dir}",
                    f"workspace_dir={runs_dir}",
                    "external_skill_memory.enable_agentic=True",
                    "adoption_tracking.enable=True",
                    "adoption_tracking.enable_analysis=True",
                    "adoption_tracking.judge_mode=llm-all",
                    "coldstart.use_coldstart=True",
                ]
                cmd.extend(CONDITIONS[condition])
                if args.steps is not None:
                    cmd.append(f"agent.steps={args.steps}")

                log_path = matrix_dir / f"{task}__{condition}.log"
                print(f"[matrix] START {task}/{condition}; log={log_path}")
                if args.dry_run:
                    row.update({"status": "planned", "command": cmd, "log_path": str(log_path)})
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                    manifest.flush()
                    continue
                env = os.environ.copy()
                env["MLEVOLVE_CONFIG"] = args.config_path
                rc = stream_command(cmd, log_path, mlevolve_dir, env)
                row["end_unix"] = time.time()
                row["returncode"] = rc
                row["status"] = "completed" if rc == 0 else "failed"
                row["run_dir"] = find_newest_run(runs_dir, marker, row["start_unix"])
                row["log_path"] = str(log_path)
                manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                manifest.flush()
                print(f"[matrix] END {task}/{condition}; status={row['status']} rc={rc} run_dir={row['run_dir']}")

    print(f"[matrix] DONE manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
