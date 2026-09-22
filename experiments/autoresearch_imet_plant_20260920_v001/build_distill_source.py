#!/usr/bin/env python3
"""Adapt the common autoresearch run ledger to the distillation skill contract.

The experiment harness intentionally keeps a compact JSON ledger rather than the
older MLEvolve directory layout.  This adapter preserves every attempt, score,
failure and candidate checksum in a fresh, immutable source tree; it does not
copy weights into the memory pack.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-root", required=True)
    ap.add_argument("--run", nargs="+", required=True,
                    help="one run, or multiple authorized runs to combine into one cumulative source")
    ap.add_argument("--task", choices=["autoresearch-plant", "autoresearch-imet"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.task_root).resolve()
    run_names = list(args.run)
    runs = [root / "runs" / name for name in run_names]
    for run in runs:
        if not run.exists():
            raise FileNotFoundError(run)
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    launches = [json.loads((run / "LAUNCH.json").read_text()) for run in runs]
    launch = dict(launches[0])
    if len(runs) > 1:
        launch["run_id"] = "+".join(run_names)
        launch["source_rounds"] = [x.get("round") for x in launches]
        launch["round"] = max(int(x.get("round", 1)) for x in launches)
    states = [json.loads((run / "STATE.json").read_text()) for run in runs]
    launch["task_id"] = args.task

    combined_name = launch["run_id"]
    source_run = out / "runs" / combined_name
    logs = source_run / "logs"
    checkpoints = source_run / "workspace" / "working" / "budget_checkpoints"
    code_dir = logs / "candidate_checkpoints"
    journal = []
    prior_score = None
    all_attempts = []
    global_step = 0
    for run_name, run, state in zip(run_names, runs, states):
      for local_step, row in enumerate(state.get("history", []), 1):
        global_step += 1
        attempt_name = row["attempt"]
        attempt_dir = run / f"attempt-{int(attempt_name.rsplit('-', 1)[1]):03d}"
        # run_worker uses a run directory and attempt-XYZ; the attempt id is
        # stable even if the human-facing attempt string contains a suffix.
        if not attempt_dir.exists():
            # fallback: locate by the numeric suffix
            matches = sorted(run.glob(f"attempt-{int(attempt_name.rsplit('-', 1)[1]):03d}"))
            if not matches:
                raise FileNotFoundError(attempt_dir)
            attempt_dir = matches[0]
        node = f"node-{global_step:03d}"
        code = attempt_dir / "candidate.py"
        if code.exists():
            target_code = code_dir / f"{node}.py"
            target_code.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(code, target_code)
            code_hash = sha(code)
        else:
            code_hash = None
        valid = row.get("status") == "valid" and row.get("search_score") is not None
        search_score = row.get("search_score")
        tune_score = row.get("tune_score")
        metric = None if not valid else -float(search_score)
        context = {
            "task": args.task,
            "phase": "host_scored_candidate" if valid else "failed_candidate",
            "architecture": row.get("change", "unscored candidate"),
            "recipe": "same autoresearch single-agent image-search harness",
        }
        parameters = {
            "attempt": attempt_name,
            "budget_seconds": row.get("seconds"),
            "host_metric_role": launch.get("metric"),
            "reported_tune_score": tune_score,
        }
        cp = checkpoints / node / "inner_tuning"
        trial_id = f"{node}:trial"
        trial = {
            "trial_id": trial_id,
            "node_id": node,
            "code_sha256": code_hash,
            "action": row.get("change", "candidate execution"),
            "parameters": parameters,
            "status": "ok" if valid else "failed",
            "tune_loss": None if tune_score is None else -float(tune_score),
            "incumbent_before": None if prior_score is None else -float(prior_score),
            "improved": bool(valid and (prior_score is None or float(search_score) > float(prior_score))),
            "elapsed_seconds": row.get("seconds"),
            "parent_trial_id": None,
            "error_type": None if valid else row.get("reason", "candidate_stage_failed"),
            "error": None if valid else row.get("reason", "candidate_stage_failed"),
            "metadata": {"host_search_score": search_score, "task": args.task},
        }
        write(cp / "SESSION.json", {
            "context": context,
            "parameters": parameters,
            "initial": {"task": args.task, "round": launch.get("round")},
            "mutable": ["candidate architecture", "training schedule", "preprocessing"],
        })
        write(cp / f"trial-{global_step:03d}-outcome.json", trial)
        if valid:
            write(cp.parent / "VERIFIED_RESULT.json", {
                "host_recomputed": True,
                "checkpoint_validated": True,
                "metric": metric,
                "micro_f1": float(search_score) if args.task.endswith("imet") else None,
                "receipt": {
                    "tune_logloss": None if tune_score is None else -float(tune_score),
                    "metadata": {"inner_trial_id": trial_id, "parameters": parameters},
                },
            })
        journal.append({
            "id": node,
            "step": global_step,
            "metric": {"value": metric},
            "context": context,
            "exec_time": row.get("seconds"),
            "exc_type": None if valid else row.get("reason", "candidate_stage_failed"),
            "analysis": row.get("change", "") if valid else row.get("reason", "candidate_stage_failed"),
        })
        all_attempts.append((node, row, valid, search_score, tune_score))
        if valid and (prior_score is None or float(search_score) > float(prior_score)):
            prior_score = float(search_score)

    write(out / "control" / "LAUNCH.json", launch)
    write(logs / "journal.json", {"nodes": journal})
    write(out / "exported_memory" / "inner_memory.json", {"cards": []})
    write(out / "ADAPTER_RECEIPT.json", {
        "method": "autoresearch_compact_ledger_to_distillation_contract",
        "source_run": args.run,
        "attempts": len(all_attempts),
        "host_scored_attempts": sum(v for _, _, v, _, _ in all_attempts),
        "weights_copied": False,
        "source_code_copied_for_hash_audit": True,
    })
    print(json.dumps({"out": str(out), "attempts": len(all_attempts), "scored": sum(v for _, _, v, _, _ in all_attempts)}))


if __name__ == "__main__":
    main()
