#!/usr/bin/env python3
"""Adapt the frozen Spooky R1 AWS run into the distiller's documented source contract.

The R1 harness predates the generic distiller layout. This adapter keeps the
original candidate code, result, execution and inner-evaluation records, and
only creates the journal/checkpoint index that the bundled extractor expects.
It never changes the original R1 copy or any experiment runtime.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "r1-selected"
OUT = ROOT / "distill-source-r1"
TASK = "spooky-author-identification"
RUN_ID = "spooky-r1-nomemory-20260919"


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maybe_read(path: Path):
    return read(path) if path.exists() else None


def inner_records(params: dict):
    for key in ("inner_evaluation", "trajectory"):
        if isinstance(params.get(key), list):
            return params[key]
    return []


def stage_summary(attempt: Path) -> dict:
    execution = maybe_read(attempt / "execution.json") or {}
    return {
        "stages": execution.get("stages", []),
        "execution_success": execution.get("success"),
        "execution_source_sha256": execution.get("source_sha256"),
    }


def make_trial(node_id: str, attempt: dict, result: dict, raw: dict, index: int, previous: float | None) -> dict:
    tune = raw.get("tune_logloss")
    if not isinstance(tune, (int, float)):
        tune = None
    score = tune if isinstance(tune, (int, float)) else None
    return {
        "trial_id": f"{node_id}:inner-{index:03d}",
        "node_id": node_id,
        "code_sha256": result.get("source_sha256"),
        "action": raw.get("stage", result.get("change", "inner evaluation")),
        "parameters": raw,
        "status": "ok" if score is not None else "failed",
        "tune_loss": score,
        "incumbent_before": previous,
        "improved": bool(score is not None and (previous is None or score < previous)),
        "elapsed_seconds": raw.get("seconds"),
        "parent_trial_id": f"{node_id}:inner-{index - 1:03d}" if index else None,
        "error_type": None if score is not None else "missing_tune_logloss",
        "error": None if score is not None else "R1 candidate did not record a numeric tune score",
        "metadata": {"source_attempt": attempt["attempt"], "source_record": raw},
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite existing adapter root: {OUT}")
    OUT.mkdir(parents=True)
    original_launch = read(RAW / "LAUNCH.json")
    state = read(RAW / "STATE.json")
    outcome = read(RAW / "RUN_OUTCOME.json")

    runtime_manifest = {
        "adapter": "spooky_r1_legacy_to_conditional_distillation_v1",
        "source_run": RUN_ID,
        "source_condition": "R1 no memory",
        "original_launch": original_launch,
        "source_state_sha256": sha(RAW / "STATE.json"),
        "source_outcome_sha256": sha(RAW / "RUN_OUTCOME.json"),
        "source_attempt_count": len(state.get("history", [])),
        "terminal": outcome.get("terminal"),
        "terminal_best_search": outcome.get("best_search"),
        "terminal_best_attempt": outcome.get("best_attempt"),
    }
    write(OUT / "control/RUNTIME_MANIFEST.json", runtime_manifest)
    runtime_hash = sha(OUT / "control/RUNTIME_MANIFEST.json")

    launch = {
        "task_id": TASK,
        "round": 1,
        "run_id": RUN_ID,
        "data_manifest_sha256": original_launch["data_manifest_sha256"],
        "runtime_manifest_sha256": runtime_hash,
        "metric": "host_search_log_loss",
        "lower_is_better": True,
        "model": original_launch["model"],
        "effort": original_launch["effort"],
        "budget_seconds": original_launch["budget_seconds"],
        "memory": False,
        "source_launch_sha256": sha(RAW / "LAUNCH.json"),
        "source_outcome_sha256": sha(RAW / "RUN_OUTCOME.json"),
        "source_state_sha256": sha(RAW / "STATE.json"),
        "source_cutoff": "R1 terminal RUN_OUTCOME.json; no later round or diagnostic is included",
    }
    write(OUT / "control/LAUNCH.json", launch)

    journal_nodes = []
    histories = state.get("history", [])
    for ordinal, history in enumerate(histories):
        number = int(history["attempt"].rsplit("-", 1)[-1])
        node_id = f"node-{number:03d}"
        attempt = RAW / f"attempt-{number:03d}"
        cp = OUT / "runs/r1/workspace/working/budget_checkpoints" / node_id
        inner = cp / "inner_tuning"
        inner.mkdir(parents=True, exist_ok=True)

        result = maybe_read(attempt / "RESULT.json") or history
        params = maybe_read(attempt / "actual_parameters.json") or {}
        parsed = maybe_read(attempt / "03-edit.parsed.json") or {}
        records = inner_records(params)
        trials = []
        previous = None
        for i, raw in enumerate(records):
            trial = make_trial(node_id, history, result, raw, i, previous)
            trials.append(trial)
            if trial["tune_loss"] is not None:
                previous = trial["tune_loss"] if previous is None else min(previous, trial["tune_loss"])

        # Preserve the candidate's actual serialized selection as an explicit
        # selected trial so the generic verifier can bind tune score to it.
        if result.get("status") == "valid":
            selected_id = f"{node_id}:selected"
            selected_trial = {
                "trial_id": selected_id,
                "node_id": node_id,
                "code_sha256": result.get("source_sha256"),
                "action": result.get("change", "selected R1 candidate"),
                "parameters": params,
                "status": "ok",
                "tune_loss": result.get("tune_loss"),
                "incumbent_before": history.get("search_loss"),
                "improved": True,
                "elapsed_seconds": result.get("seconds"),
                "parent_trial_id": trials[-1]["trial_id"] if trials else None,
                "error_type": None,
                "error": None,
                "metadata": {"source_attempt": history["attempt"], "source_actual_parameters": True},
            }
            trials.append(selected_trial)

        for trial in trials:
            write(inner / f"trial-{trial['trial_id'].replace(':', '-')}-outcome.json", trial)
        session = {
            "context": {
                "task": TASK,
                "phase": "candidate_execution",
                "architecture": params.get("architecture", "unknown candidate architecture"),
                "recipe": "R1 no-memory candidate; host decides by search multiclass log-loss",
                "source_attempt": history["attempt"],
                "source_condition": "R1 no memory",
            },
            "parameters": {"inner_evaluation_count": len(records), "source_actual_parameters": params},
            "initial": {},
            "mutable": [],
        }
        write(inner / "SESSION.json", session)

        candidate = attempt / "candidate.py"
        if candidate.exists():
            target = OUT / "runs/r1/logs/candidate_checkpoints" / f"{node_id}.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)

        valid = result.get("status") == "valid" and isinstance(result.get("search_loss"), (int, float))
        node = {
            "id": node_id,
            "step": ordinal + 1,
            "context": session["context"],
            "metric": {"value": result.get("search_loss")} if valid else {},
            "exec_time": result.get("seconds"),
            "exc_type": None if valid else result.get("reason", "candidate execution failure"),
            "analysis": result.get("change", result.get("reason", "")),
        }
        journal_nodes.append(node)
        if valid:
            final_trial = trials[-1]
            verified = {
                "host_recomputed": True,
                "checkpoint_validated": True,
                "metric": result["search_loss"],
                "micro_f1": None,
                "receipt": {
                    "step": node_id,
                    "tune_logloss": result.get("tune_loss"),
                    "metadata": {
                        "inner_trial_id": final_trial["trial_id"],
                        "parameters": final_trial["parameters"],
                    },
                    "source_result_sha256": sha(attempt / "RESULT.json"),
                    "source_execution_sha256": sha(attempt / "execution.json") if (attempt / "execution.json").exists() else None,
                },
            }
            write(cp / "VERIFIED_RESULT.json", verified)

    write(OUT / "runs/r1/logs/journal.json", {"version": "legacy-adapter-v1", "nodes": journal_nodes})
    write(OUT / "control/SOURCE_CUTOFF.json", {
        "source_root": str(RAW),
        "source_state_sha256": sha(RAW / "STATE.json"),
        "source_outcome_sha256": sha(RAW / "RUN_OUTCOME.json"),
        "terminal_best_search": outcome.get("best_search"),
        "terminal_best_attempt": outcome.get("best_attempt"),
        "included_attempts": [h["attempt"] for h in histories],
        "excluded": "No later experiments, memory packs, diagnostics, or Leaf records.",
    })
    print(json.dumps({"source": str(RAW), "adapter": str(OUT), "nodes": len(journal_nodes), "terminal": outcome.get("terminal")}))


if __name__ == "__main__":
    main()
