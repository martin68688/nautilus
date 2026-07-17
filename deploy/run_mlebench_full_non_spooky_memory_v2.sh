#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/mlebench-memory-v2-20260717
SRC="$ROOT/mlevolve"
DATA_ROOT="$ROOT/mlebench-data"
SUITE_ROOT="$ROOT/full-suite"
RUNS_ROOT="$SUITE_ROOT/tasks"
HF_ROOT=/workspace/cache/huggingface
SOURCE_MANIFEST="$ROOT/SOURCE_MANIFEST.json"
GPU_COUNT="${MLEVOLVE_GPU_COUNT:-2}"
PARALLEL_SEARCH_NUM="${MLEVOLVE_PARALLEL_SEARCH_NUM:-$GPU_COUNT}"
CPU_COUNT="${MLEVOLVE_CPU_COUNT:-8}"

for value in "$GPU_COUNT" "$PARALLEL_SEARCH_NUM" "$CPU_COUNT"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    printf 'resource values must be positive integers: gpu=%s parallel=%s cpu=%s\n' \
      "$GPU_COUNT" "$PARALLEL_SEARCH_NUM" "$CPU_COUNT" >&2
    exit 2
  }
done
[[ "$PARALLEL_SEARCH_NUM" == "$GPU_COUNT" ]] || {
  printf 'parallel_search_num must equal GPU count: gpu=%s parallel=%s\n' \
    "$GPU_COUNT" "$PARALLEL_SEARCH_NUM" >&2
  exit 2
}
export MLEVOLVE_GPU_COUNT="$GPU_COUNT"
export MLEVOLVE_PARALLEL_SEARCH_NUM="$PARALLEL_SEARCH_NUM"
export MLEVOLVE_CPU_COUNT="$CPU_COUNT"

tasks=(
  leaf-classification
  aerial-cactus-identification
  mlsp-2013-birds
  denoising-dirty-documents
  new-york-city-taxi-fare-prediction
)

test -s "$SOURCE_MANIFEST"
test -s "$ROOT/paper-skills/hyper_memory/run_forest_graph.json"
test -s "$ROOT/paper-skills/hyper_memory/run_forest_index.npz"
mkdir -p "$DATA_ROOT" "$RUNS_ROOT" "$HF_ROOT"
ln -sfn /workspace/nautilus/mlevolve/data/leaf-classification "$DATA_ROOT/leaf-classification"
ln -sfn /workspace/nautilus/mlevolve/data/aerial-cactus-identification "$DATA_ROOT/aerial-cactus-identification"
ln -sfn /workspace/nautilus/mlevolve/data/mlsp-2013-bird "$DATA_ROOT/mlsp-2013-birds"
ln -sfn /workspace/nautilus/mlevolve/data/denoising-dirty-documents "$DATA_ROOT/denoising-dirty-documents"
ln -sfn /workspace/nautilus/mlevolve/data/new-york-city-taxi-fare-prediction "$DATA_ROOT/new-york-city-taxi-fare-prediction"

cd "$SRC"
set -a
. ./.env
set +a
export DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL
export MLEVOLVE_CONFIG="$SRC/config/config_run_forest_stage_hybrid.yaml"
export MLEVOLVE_CODE_REVISION
export MLEVOLVE_CODE_WORKTREE_SHA256
MLEVOLVE_CODE_REVISION="$(python -c 'import json; print(json.load(open("../SOURCE_MANIFEST.json"))["base_commit"])')"
MLEVOLVE_CODE_WORKTREE_SHA256="$(python -c 'import json; print(json.load(open("../SOURCE_MANIFEST.json"))["source_sha256"])')"
export HF_HOME="$HF_ROOT"
export HUGGINGFACE_HUB_CACHE="$HF_ROOT/hub"
export TRANSFORMERS_CACHE="$HF_ROOT/transformers"
export DATASET_DIR="$DATA_ROOT"
export PYTHONUNBUFFERED=1

test -n "${DEEPSEEK_API_KEY:-}"
test -n "${DEEPSEEK_BASE_URL:-}"
test -n "${DEEPSEEK_MODEL:-}"

python - <<'PY'
import collections, hashlib, json, os
from pathlib import Path
from config import _load_cfg

root = Path("/workspace/mlebench-memory-v2-20260717")
graph = root / "paper-skills/hyper_memory/run_forest_graph.json"
index = root / "paper-skills/hyper_memory/run_forest_index.npz"
graph_data = json.loads(graph.read_text())
meta = graph_data["meta"]
cfg = _load_cfg(root / "mlevolve/config/config_run_forest_stage_hybrid.yaml", use_cli_args=False)
assert cfg.external_skill_memory.enable is True
assert cfg.external_skill_memory.mode == "run_forest_stage_hybrid"
assert cfg.external_skill_memory.scoring_mode == "flat_twin"
assert cfg.external_skill_memory.retrieval_control == "layered_strategy"
assert cfg.run_identity.memory_enabled is True
assert cfg.run_identity.memory_system == "run_forest_stage_hybrid"
assert cfg.run_identity.memory_version == "stage_hybrid_v2"
assert len(meta["source_runs"]) == 29

task_clean_runs = collections.defaultdict(set)
for node in graph_data["nodes"]:
    audit = node.get("leakage_audit") if isinstance(node.get("leakage_audit"), dict) else {}
    if node.get("type") != "RunNode" or audit.get("status") != "clean" or audit.get("rank_eligible") is not True:
        continue
    task_clean_runs[str(node.get("task") or "")].add(str(node.get("run_id") or ""))
minimum_task_local_runs = {
    "leaf-classification": 8,
    "aerial-cactus-identification": 2,
    "denoising-dirty-documents": 1,
    "new-york-city-taxi-fare-prediction": 1,
}
for task, minimum in minimum_task_local_runs.items():
    assert len(task_clean_runs[task]) >= minimum, (task, sorted(task_clean_runs[task]))
assert not task_clean_runs["mlsp-2013-birds"]

record = {
    "schema": "mlevolve_suite_identity_v1",
    "experiment_group": "stage_hybrid_v2_all_clean_history",
    "baseline_reference_group": "baseline_no_external_memory",
    "memory_enabled": True,
    "memory_system": "run_forest_stage_hybrid",
    "memory_version": "stage_hybrid_v2",
    "memory_snapshot_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
    "memory_index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
    "memory_source_count": len(meta["source_runs"]),
    "memory_source_runs": meta["source_runs"],
    "task_local_clean_runs": {
        task: sorted(task_clean_runs[task])
        for task in [
            "leaf-classification",
            "aerial-cactus-identification",
            "mlsp-2013-birds",
            "denoising-dirty-documents",
            "new-york-city-taxi-fare-prediction",
        ]
    },
    "birds_memory_mode": "stage_hybrid_v2_clean_cross_task_memory_transfer",
    "code_revision": os.environ["MLEVOLVE_CODE_REVISION"],
    "code_worktree_sha256": os.environ["MLEVOLVE_CODE_WORKTREE_SHA256"],
    "resources": {
        "gpu_count": int(os.environ["MLEVOLVE_GPU_COUNT"]),
        "parallel_search_num": int(os.environ["MLEVOLVE_PARALLEL_SEARCH_NUM"]),
        "cpu_count": int(os.environ["MLEVOLVE_CPU_COUNT"]),
    },
    "tasks": [
        "leaf-classification", "aerial-cactus-identification", "mlsp-2013-birds",
        "denoising-dirty-documents", "new-york-city-taxi-fare-prediction",
    ],
}
(root / "full-suite/SUITE_IDENTITY.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: record[key] for key in ("experiment_group", "memory_version", "memory_source_count", "memory_snapshot_sha256")}))
PY

for task in "${tasks[@]}"; do
  task_root="$RUNS_ROOT/$task"
  mkdir -p "$task_root/runs"
  if [[ -f "$task_root/SUCCESS" ]]; then
    printf '%s already passed with memory-v2 identity; skipping\n' "$task"
    continue
  fi

  test -s "$DATA_ROOT/$task/prepared/public/description.md"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$task_root/STARTED_AT"
  printf 'running\n' > "$task_root/STATE"
  printf '%s\n' "$task" > "$SUITE_ROOT/CURRENT_TASK"
  printf 'starting %s: stage_hybrid_v2, all-clean-history, %s GPU(s), %s worker(s), %s CPUs\n' \
    "$task" "$GPU_COUNT" "$PARALLEL_SEARCH_NUM" "$CPU_COUNT"

  role_override=()
  if [[ "$task" == "mlsp-2013-birds" ]]; then
    role_override+=("agent.draft_role_policy.roles=[coldstart_baseline,memory_transfer,novel_exploration]")
  fi

  set +e
  timeout --foreground --signal=TERM --kill-after=30s 22200s \
    python -u run.py \
      exp_id="$task" \
      exp_name="memory-v2-$task" \
      dataset_dir="$DATA_ROOT" \
      data_dir="$DATA_ROOT/$task/prepared/public" \
      desc_file="$DATA_ROOT/$task/prepared/public/description.md" \
      log_dir="$task_root/runs" \
      workspace_dir="$task_root/runs" \
      agent.search.num_gpus="$GPU_COUNT" \
      agent.search.parallel_search_num="$PARALLEL_SEARCH_NUM" \
      cpu_number="$CPU_COUNT" \
      "${role_override[@]}" \
      2>&1 | tee "$task_root/launcher.log"
  run_rc=${PIPESTATUS[0]}
  set -e

  printf '%s\n' "$run_rc" > "$task_root/EXIT_CODE"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$task_root/FINISHED_AT"
  if [[ "$run_rc" -ne 0 ]]; then
    printf 'failed\n' > "$task_root/STATE"
    printf '%s\n' "$task" > "$SUITE_ROOT/FAILED_TASK"
    printf 'task %s failed with exit code %s; stopping supervisor for in-pod debugging\n' "$task" "$run_rc"
    exit "$run_rc"
  fi

  latest_run="$(find "$task_root/runs" -mindepth 1 -maxdepth 1 -type d -name '*_memory-v2-*' | sort | tail -n 1)"
  python - "$latest_run/logs/run_identity.json" <<'PY'
import json, sys
p = sys.argv[1]
identity = json.load(open(p))
assert identity["experiment_group"] == "stage_hybrid_v2_all_clean_history"
assert identity["memory_enabled"] is True
assert identity["memory_system"] == "run_forest_stage_hybrid"
assert identity["memory_version"] == "stage_hybrid_v2"
assert identity["memory_source_count"] > 0
assert len(identity["memory_snapshot_sha256"]) == 64
print(json.dumps({"identity_verified": p, "memory_source_count": identity["memory_source_count"]}))
PY
  touch "$task_root/SUCCESS"
  printf 'success\n' > "$task_root/STATE"
done

rm -f "$SUITE_ROOT/CURRENT_TASK" "$SUITE_ROOT/FAILED_TASK"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$SUITE_ROOT/ALL_SUCCESS"
printf 'all non-spooky MLEBench tasks passed with stage_hybrid_v2 memory\n'
