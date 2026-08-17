#!/usr/bin/env python3
"""Reproduce frozen Leaf candidates on the full official Kaggle train/test split.

Each Indexed-Job completion owns one immutable candidate attempt.  The exact
historical source is preserved, then an appended recorder saves prediction
variants that the historical script kept only in memory (for example v22
NN-only versus its submitted blend).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


DATASET_ROOT = Path(
    "/workspace/experiment-end2end-kaggle-leaf-validation-v1/"
    "dataset-v1-official-20260810"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/workspace/experiment-end2end-kaggle-leaf-validation-v1/reproductions-v1"
)

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "dynamic_v22_nn_and_blend",
        "system_id": "dynamic_hybrid",
        "historical_metric": 0.005085691845970755,
        "historical_metric_note": "NN-only OOF; historical submission used blend",
        "journal": (
            "/workspace/experiment-end2end-memory-agent-v22/runs/"
            "e2e-pilot-agentic-three-role-v22__leaf-classification__dynamic_hybrid__seed-1/"
            "attempt-000/agent/logs/20260806_185700_"
            "e2e-pilot-agentic-three-role-v22__leaf-classification__dynamic_hybrid__seed-1/"
            "journal.json"
        ),
        "node_id": "36537a9cab674391b848515dfdeca00b",
        "array_variants": {
            "nn_only": "test_preds",
            "historical_blend": "test_preds_final",
        },
        "memory_disposition_before_official_score": "quarantine_pending_official_score",
    },
    {
        "candidate_id": "macla_v21_best",
        "system_id": "macla_style_port",
        "historical_metric": 0.002827361857504395,
        "historical_metric_note": "reported full OOF",
        "journal": (
            "/workspace/experiment-end2end-memory-agent-v21/runs/"
            "e2e-pilot-agentic-three-role-v21__leaf-classification__macla_style_port__seed-1/"
            "attempt-001/agent/logs/20260807_055819_"
            "e2e-pilot-agentic-three-role-v21__leaf-classification__macla_style_port__seed-1/"
            "journal.json"
        ),
        "node_id": "2e81a77222c7401c9c2585c5e6323905",
        "array_variants": {"macla_exact": "test_pred_final"},
        "memory_disposition_before_official_score": "quarantine_pending_official_score",
    },
    {
        "candidate_id": "runforest_v21_best",
        "system_id": "runforest_only",
        "historical_metric": 0.0048590419143516046,
        "historical_metric_note": (
            "reported OOF; historical submission applied an additional average temperature"
        ),
        "journal": (
            "/workspace/experiment-end2end-memory-agent-v21/runs/"
            "e2e-pilot-agentic-three-role-v21__leaf-classification__runforest_only__seed-1/"
            "attempt-001/agent/logs/20260807_055845_"
            "e2e-pilot-agentic-three-role-v21__leaf-classification__runforest_only__seed-1/"
            "journal.json"
        ),
        "node_id": "4d2afeda635a44b4adb1c7c97d7a45d6",
        "array_variants": {
            "single_temperature_fold_average": "test_preds_sum / len(folds)",
            "historical_double_temperature": "test_probs",
        },
        "memory_disposition_before_official_score": "quarantine_pending_official_score",
    },
    {
        "candidate_id": "gome_v21_best",
        "system_id": "gome_style_port",
        "historical_metric": 0.007668362295007117,
        "historical_metric_note": "reported full OOF",
        "journal": (
            "/workspace/experiment-end2end-memory-agent-v21/runs/"
            "e2e-pilot-agentic-three-role-v21__leaf-classification__gome_style_port__seed-1/"
            "attempt-000/agent/logs/20260806_220452_"
            "e2e-pilot-agentic-three-role-v21__leaf-classification__gome_style_port__seed-1/"
            "journal.json"
        ),
        "node_id": "05bb9616e28748d8b7aad1e7389e7ac9",
        "array_variants": {"gome_exact": "test_probs_norm"},
        "memory_disposition_before_official_score": "quarantine_pending_official_score",
    },
    {
        "candidate_id": "dynamic_v37_submission_aligned",
        "system_id": "dynamic_hybrid",
        "historical_metric": 0.04659931150982626,
        "historical_metric_note": "full OOF submission-aligned internal",
        "journal": (
            "/workspace/experiment-end2end-memory-agent-v37/runs/"
            "e2e-smoke-leaf-causal-sparse-router-v37__leaf-classification__dynamic_hybrid__seed-1/"
            "attempt-000/agent/logs/20260808_062820_"
            "e2e-smoke-leaf-causal-sparse-router-v37__leaf-classification__dynamic_hybrid__seed-1/"
            "journal.json"
        ),
        "node_id": "7dd024170f3740b29b5c7e0f4a67bcdb",
        "array_variants": {"v37_exact": "test_preds"},
        "memory_disposition_before_official_score": "eligible_internal_control",
    },
    {
        "candidate_id": "no_memory_v21_stacking_diagnostic",
        "system_id": "no_memory",
        "historical_metric": 0.000541202108586477,
        "historical_metric_note": "known meta-learner training-set self-evaluation",
        "journal": (
            "/workspace/experiment-end2end-memory-agent-v21/runs/"
            "e2e-pilot-agentic-three-role-v21__leaf-classification__no_memory__seed-1/"
            "attempt-000/agent/logs/20260806_154751_"
            "e2e-pilot-agentic-three-role-v21__leaf-classification__no_memory__seed-1/"
            "journal.json"
        ),
        "node_id": "54cdcd23fd5f4b4bb20eed4c542beb42",
        "array_variants": {},
        "memory_disposition_before_official_score": "permanent_quarantine_validation_leakage",
    },
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def source_code(journal_path: Path, node_id: str) -> str:
    journal = read_json(journal_path)
    rows = [row for row in journal.get("nodes") or [] if row.get("id") == node_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one node {node_id}, found {len(rows)}")
    code = rows[0].get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Node {node_id} has no executable code")
    return code


def recorder_source(candidate: dict[str, Any]) -> str:
    variants = json.dumps(candidate["array_variants"], sort_keys=True)
    return f'''\n\n# ---- appended official-reproduction recorder (source above is unchanged) ----
import json as _rec_json
from pathlib import Path as _RecPath
import numpy as _rec_np
import pandas as _rec_pd
import shutil as _rec_shutil

_rec_root = _RecPath("./official_variants")
_rec_root.mkdir(parents=True, exist_ok=True)
_rec_sample = _rec_pd.read_csv("./input/sample_submission.csv")
_rec_default = _RecPath("./submission/submission.csv")
if _rec_default.is_file():
    _rec_shutil.copy2(_rec_default, _rec_root / "as_written_raw.csv")

def _rec_classes():
    value = globals().get("class_names")
    if value is None and globals().get("le") is not None:
        value = globals()["le"].classes_
    if value is None:
        value = _rec_sample.columns[1:].to_numpy()
    return [str(item) for item in list(value)]

def _rec_save(name, expression):
    array = eval(expression, globals())
    array = _rec_np.asarray(array, dtype=float)
    if array.shape != (len(_rec_sample), len(_rec_sample.columns) - 1):
        raise ValueError(f"variant {{name}} has unexpected shape {{array.shape}}")
    frame = _rec_pd.DataFrame(array, columns=_rec_classes())
    frame.insert(0, "id", _rec_sample["id"].to_numpy())
    frame = frame[_rec_sample.columns]
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    values = _rec_np.clip(values, 1e-15, 1.0)
    values = values / values.sum(axis=1, keepdims=True)
    frame.iloc[:, 1:] = values
    frame.to_csv(_rec_root / f"{{name}}.csv", index=False)

for _rec_name, _rec_expr in {variants}.items():
    _rec_save(_rec_name, _rec_expr)
print("OFFICIAL_VARIANT_RECORDER_COMPLETE")
'''


def run_streamed(
    command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path, timeout: int
) -> int:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert process.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                    raise TimeoutError(f"candidate exceeded {timeout}s")
                for key, _ in selector.select(timeout=1.0):
                    line = key.fileobj.readline()
                    if line:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            return int(process.returncode or 0)
        finally:
            selector.close()


def canonicalize_submission(
    source: Path, destination: Path, sample: pd.DataFrame
) -> dict[str, Any]:
    frame = pd.read_csv(source)
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"submission columns differ: {source}")
    if len(frame) != len(sample):
        raise ValueError(f"submission row count differs: {source}")
    if frame["id"].duplicated().any() or set(frame["id"]) != set(sample["id"]):
        raise ValueError(f"submission IDs differ: {source}")
    frame = frame.set_index("id").loc[sample["id"]].reset_index()
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite probabilities: {source}")
    if values.min() < 0.0 or values.max() > 1.0:
        raise ValueError(f"out-of-range probabilities: {source}")
    row_sums = values.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"non-positive row sums: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "canonical_path": str(destination),
        "canonical_sha256": sha256_file(destination),
        "rows": len(frame),
        "columns": len(frame.columns),
        "probability_min": float(values.min()),
        "probability_max": float(values.max()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
    }


def static_audit(code: str) -> dict[str, Any]:
    lower = code.lower().replace(" ", "")
    return {
        "has_full_fold_loop": "forfold" in lower and "oof" in lower,
        "has_submission_write": "submission.csv" in lower,
        "has_temperature_search": "temperature" in lower and "log_loss" in lower,
        "has_meta_model": "logisticregression" in lower or "stack" in lower,
        "possible_meta_training_self_eval": (
            "logisticregression" in lower
            and ".fit(" in lower
            and "predict_proba" in lower
        ),
        "code_sha256": sha256_bytes(code.encode("utf-8")),
    }


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    value = dict(receipt)
    value["receipt_sha256"] = ""
    value["receipt_sha256"] = sha256_bytes(
        json.dumps(
            {key: item for key, item in value.items() if key != "receipt_sha256"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    parser.add_argument("--attempt", default="attempt-000")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args()
    index = args.index
    if index is None:
        index = int(os.environ["JOB_COMPLETION_INDEX"])
    if not 0 <= index < len(CANDIDATES):
        raise IndexError(index)
    candidate = dict(CANDIDATES[index])

    dataset_receipt = read_json(DATASET_ROOT / "DATASET_RECEIPT.json")
    if dataset_receipt.get("status") != "pass":
        raise RuntimeError("official dataset receipt is not passing")
    sample_path = DATASET_ROOT / "input" / "sample_submission.csv"
    sample = pd.read_csv(sample_path)
    if sample.shape != (594, 100):
        raise RuntimeError(f"unexpected official sample shape: {sample.shape}")

    attempt_root = args.output_root / candidate["candidate_id"] / args.attempt
    attempt_root.parent.mkdir(parents=True, exist_ok=True)
    attempt_root.mkdir()
    source_dir = attempt_root / "source_snapshot"
    work_dir = attempt_root / "work"
    source_dir.mkdir()
    work_dir.mkdir()
    (work_dir / "input").symlink_to(DATASET_ROOT / "input", target_is_directory=True)
    (work_dir / "submission").mkdir()
    (work_dir / "working").mkdir()

    journal_path = Path(candidate["journal"])
    code = source_code(journal_path, candidate["node_id"])
    source_path = source_dir / "solution.py"
    source_path.write_text(code, encoding="utf-8")
    instrumented_path = source_dir / "solution_instrumented.py"
    instrumented_path.write_text(code + recorder_source(candidate), encoding="utf-8")

    receipt: dict[str, Any] = {
        "schema": "kaggle_leaf_official_reproduction_receipt_v1",
        "candidate_index": index,
        "candidate_id": candidate["candidate_id"],
        "system_id": candidate["system_id"],
        "historical_metric": candidate["historical_metric"],
        "historical_metric_note": candidate["historical_metric_note"],
        "memory_disposition_before_official_score": candidate[
            "memory_disposition_before_official_score"
        ],
        "journal_path": str(journal_path),
        "journal_sha256": sha256_file(journal_path),
        "node_id": candidate["node_id"],
        "source_code_sha256": sha256_file(source_path),
        "instrumented_code_sha256": sha256_file(instrumented_path),
        "static_audit": static_audit(code),
        "dataset_receipt_sha256": dataset_receipt.get("receipt_sha256"),
        "sample_submission_sha256": sha256_file(sample_path),
        "timeout_seconds": args.timeout_seconds,
        "started_at_unix": time.time(),
        "status": "running",
        "variants": {},
    }
    receipt_path = attempt_root / "REPRODUCTION_RECEIPT.json"
    write_receipt(receipt_path, receipt)
    print(
        f"OFFICIAL_REPRO_START index={index} candidate={candidate['candidate_id']} "
        f"node={candidate['node_id']} source_sha256={receipt['source_code_sha256']}",
        flush=True,
    )

    env = dict(os.environ)
    env.setdefault("HF_HOME", "/workspace/huggingface")
    env.setdefault("HUGGINGFACE_HUB_CACHE", "/workspace/huggingface/hub")
    env.setdefault("TRANSFORMERS_CACHE", "/workspace/huggingface/hub")
    env.setdefault("TORCH_HOME", "/workspace/huggingface/torch")
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    exit_code: int | None = None
    error: str | None = None
    try:
        exit_code = run_streamed(
            [sys.executable, "-u", str(instrumented_path)],
            cwd=work_dir,
            env=env,
            log_path=attempt_root / "execution.log",
            timeout=args.timeout_seconds,
        )
        if exit_code != 0:
            raise RuntimeError(f"candidate process exited with {exit_code}")
        raw_variants = work_dir / "official_variants"
        for source in sorted(raw_variants.glob("*.csv")):
            variant = source.stem
            destination = attempt_root / "submissions" / f"{variant}.csv"
            receipt["variants"][variant] = canonicalize_submission(
                source, destination, sample
            )
        if not receipt["variants"]:
            raise RuntimeError("candidate produced no official submission variants")
        receipt["status"] = "pass"
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        receipt["status"] = "failed"
        receipt["error"] = error
    receipt["exit_code"] = exit_code
    receipt["runtime_seconds"] = time.monotonic() - started
    receipt["finished_at_unix"] = time.time()
    write_receipt(receipt_path, receipt)
    print(
        f"OFFICIAL_REPRO_END candidate={candidate['candidate_id']} "
        f"status={receipt['status']} runtime_seconds={receipt['runtime_seconds']:.1f} "
        f"variants={sorted(receipt['variants'])} error={error}",
        flush=True,
    )
    return 0 if receipt["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
