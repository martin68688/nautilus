#!/usr/bin/env bash
set -euo pipefail

# This script is intentionally Pod-oriented.  It creates one immutable,
# target-specific positive-transfer Bundle on the PVC and never submits a Job.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC="${WP7_SOURCE_ROOT:-/work/decision-admissibility-wp7-aerial-r11-source}"
PVC_ROOT="${WP7_PVC_ROOT:-/workspace}"
WP4_ROOT="${WP7_WP4_ROOT:-$PVC_ROOT/decision-admissibility-wp4-20260719-r1/corrected-r2}"
CORPUS_SOURCE="${WP7_CORPUS_SOURCE:-$PVC_ROOT/mlevolve-original-runs/be034ec/3h120}"
OUTPUT_ROOT="${WP7_DOMAIN_BUNDLE_ROOT:-$PVC_ROOT/decision-admissibility-wp7-image-domain-bundle-r1}"
CREATED_AT="${WP7_DOMAIN_BUNDLE_CREATED_AT:-2026-07-19T21:30:00Z}"
SPLIT_VERSION="be034ec-nonspooky-image-aerial-v1"
BUNDLE_ID="mlevolve-be034ec-image-aerial-task-heldout-v1"
TARGET_TASK="aerial-cactus-identification"
TARGET_FAMILY="General Image"
TOOLS="$SRC/paper-skills/memory_bundle"
STAGING="$PVC_ROOT/.decision-admissibility-wp7-image-domain-bundle-r1-staging-$(date +%s)-$RANDOM"

SOURCE_TASKS=(
  aptos2019-blindness-detection
  dog-breed-identification
  dogs-vs-cats-redux-kernels-edition
  leaf-classification
  plant-pathology-2020-fgvc7
)

test ! -e "$OUTPUT_ROOT"
test ! -e "$STAGING"
test -d "$CORPUS_SOURCE"
test -s "$WP4_ROOT/corpus_manifest.json"
test -s "$WP4_ROOT/source_before.json"
test -s "$WP4_ROOT/audit_report.json"
test -s "$WP4_ROOT/audit_sidecars/index.json"
test -s "$WP4_ROOT/reviewed-inputs/corpus_drift_review.json"
test -s "$WP4_ROOT/reviewed-inputs/merged/clauses.jsonl"
test -s "$WP4_ROOT/reviewed-inputs/merged/containers.json"
test -d "$WP4_ROOT/reviewed-inputs/binder"
test -d "$SRC/mlevolve/config/protocols"

if [[ -e /corpus ]]; then
  test "$(readlink -f /corpus)" = "$(readlink -f "$CORPUS_SOURCE")"
else
  ln -s "$CORPUS_SOURCE" /corpus
fi

mkdir -p "$STAGING/splits"
printf 'building\n' > "$STAGING/STATE"
exec > >(tee -a "$STAGING/build.log") 2>&1

input_hashes="$STAGING/INPUT_SHA256SUMS"
sha256sum \
  "$WP4_ROOT/corpus_manifest.json" \
  "$WP4_ROOT/source_before.json" \
  "$WP4_ROOT/audit_report.json" \
  "$WP4_ROOT/audit_sidecars/index.json" \
  "$WP4_ROOT/reviewed-inputs/corpus_drift_review.json" \
  "$WP4_ROOT/reviewed-inputs/merged/clauses.jsonl" \
  "$WP4_ROOT/reviewed-inputs/merged/containers.json" \
  > "$input_hashes"

export PYTHONPATH="$TOOLS:$SRC/mlevolve${PYTHONPATH:+:$PYTHONPATH}"

split_args=(
  --manifest "$WP4_ROOT/corpus_manifest.json"
  --output-dir "$STAGING/splits"
  --split-version "$SPLIT_VERSION"
  --created-at "$CREATED_AT"
  --same-domain-target-task "$TARGET_TASK"
  --same-domain-target-family "$TARGET_FAMILY"
)
for task in "${SOURCE_TASKS[@]}"; do
  split_args+=(--same-domain-source-task "$task")
done
python "$TOOLS/build_split_manifests.py" "${split_args[@]}" \
  | tee "$STAGING/SPLIT_BUILD_OUTPUT.json"

SPLIT="$STAGING/splits/same-domain-task-heldout.json"
RUNFOREST="$STAGING/runforest-image-task-heldout"
BUNDLE="$STAGING/bundle-image-task-heldout"
VALIDATION="$STAGING/image_task_heldout_validation_report.json"

python "$SRC/paper-skills/hyper_memory/build_run_forest_memory.py" \
  --corpus-manifest "$WP4_ROOT/corpus_manifest.json" \
  --audit-dir "$WP4_ROOT/audit_sidecars" \
  --sop-clauses "$WP4_ROOT/reviewed-inputs/merged/clauses.jsonl" \
  --sop-containers "$WP4_ROOT/reviewed-inputs/merged/containers.json" \
  --split-manifest "$SPLIT" \
  --bundle-id "$BUNDLE_ID" \
  --out-dir "$RUNFOREST" \
  | tee "$STAGING/RUNFOREST_BUILD_OUTPUT.json"

DEEPSEEK_MODEL="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$WP4_ROOT/reviewed-inputs/distillation/distillation_report.json")"
DEEPSEEK_PROMPT_HASH="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["system_prompt_hash"])' "$WP4_ROOT/reviewed-inputs/distillation/distillation_report.json")"
DETECTOR_VERSION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["detector_version"])' "$WP4_ROOT/audit_report.json")"

python "$TOOLS/build_memory_bundle.py" \
  --corpus-manifest "$WP4_ROOT/corpus_manifest.json" \
  --drift-review "$WP4_ROOT/reviewed-inputs/corpus_drift_review.json" \
  --split-manifest "$SPLIT" \
  --splits-dir "$STAGING/splits" \
  --audit-dir "$WP4_ROOT/audit_sidecars" \
  --runforest-dir "$RUNFOREST" \
  --authority-dir "$WP4_ROOT/reviewed-inputs/binder" \
  --protocol-registry "$SRC/mlevolve/config/protocols" \
  --output-dir "$BUNDLE" \
  --bundle-id "$BUNDLE_ID" \
  --bundle-version v1 \
  --authority-policy-version authority_v1 \
  --detector-version "$DETECTOR_VERSION" \
  --deepseek-model "$DEEPSEEK_MODEL" \
  --deepseek-prompt-hash "$DEEPSEEK_PROMPT_HASH" \
  --certification-level raw_audited \
  | tee "$STAGING/BUNDLE_BUILD_OUTPUT.json"

python "$TOOLS/validate_memory_bundle.py" \
  --bundle "$BUNDLE" \
  --report "$VALIDATION" \
  | tee "$STAGING/BUNDLE_VALIDATION_OUTPUT.json"

python "$TOOLS/source_fingerprint.py" \
  --root /corpus \
  --compare "$WP4_ROOT/source_before.json" \
  --output "$STAGING/source_after_bundle_build.json"

python - "$SPLIT" "$RUNFOREST" "$BUNDLE" "$VALIDATION" \
  "$input_hashes" "$WP4_ROOT" "$STAGING" "$BUNDLE_ID" \
  "${SOURCE_TASKS[@]}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

split_path = Path(sys.argv[1])
runforest = Path(sys.argv[2])
bundle = Path(sys.argv[3])
validation_path = Path(sys.argv[4])
input_hashes_path = Path(sys.argv[5])
wp4_root = Path(sys.argv[6])
staging = Path(sys.argv[7])
expected_bundle_id = sys.argv[8]
expected_source_tasks = set(sys.argv[9:])

split = json.loads(split_path.read_text())
graph = json.loads((runforest / "graph.json").read_text())
report = json.loads((runforest / "build_report.json").read_text())
manifest = json.loads((bundle / "manifest.json").read_text())
validation = json.loads(validation_path.read_text())
source_after = json.loads((staging / "source_after_bundle_build.json").read_text())
source_before = json.loads((wp4_root / "source_before.json").read_text())

assert split["split_kind"] == "same-domain-task-heldout"
assert set(split["source_task_ids"]) == expected_source_tasks
assert split["heldout_task_ids"] == ["aerial-cactus-identification"]
assert split["validation"]["cross_domain_source_run_count"] == 0
assert split["validation"]["all_sources_have_target_domain"] is True
assert report["source_domains"] == ["image"]
assert report["heldout_domains"] == ["image"]
assert report["target_domain"] == "image"
assert report["all_included_clauses_have_domain_lineage"] is True
assert report["cross_domain_included_clause_ids"] == []
assert report["heldout_run_refs_in_graph"] == []
assert validation["valid"] is True and validation["errors"] == []
assert manifest["bundle_id"] == expected_bundle_id
assert all(
    source_after[key] == source_before[key]
    for key in ("file_count", "paths_sha256", "stat_fingerprint_sha256")
)

nodes = graph["nodes"]
run_tasks = {
    node.get("task") for node in nodes if node.get("type") == "Run"
}
assert run_tasks == expected_source_tasks
assert "aerial-cactus-identification" not in run_tasks
clauses = [node for node in nodes if node.get("type") == "SOPClause"]
assert clauses
for clause in clauses:
    assert set(clause["source_task_ids"]) <= expected_source_tasks
    assert clause["source_task_ids"]
    assert clause["source_task_families"] == ["General Image"]
    assert clause["source_domains"] == ["image"]
    assert clause["transfer_scope"] in {"same_domain", "domain_general"}

subprocess.run(
    ["sha256sum", "--check", str(input_hashes_path)],
    cwd=wp4_root,
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

unsigned_manifest = dict(manifest)
manifest_sha = unsigned_manifest["manifest_sha256"]
summary = {
    "schema": "decision_admissibility_same_domain_bundle_build_v1",
    "bundle_id": manifest["bundle_id"],
    "bundle_manifest_sha256": manifest_sha,
    "split_id": split["split_id"],
    "split_manifest_sha256": split["manifest_sha256"],
    "transfer_design": split["allocation"]["transfer_design"],
    "target_task_id": split["allocation"]["target_task_id"],
    "target_task_family": split["allocation"]["target_task_family"],
    "target_domain": split["allocation"]["target_domain"],
    "source_task_ids": sorted(expected_source_tasks),
    "source_domains": report["source_domains"],
    "source_run_count": report["source_run_count"],
    "heldout_run_count": report["heldout_run_count"],
    "included_clause_count": report["included_clause_count"],
    "domain_general_clause_count": len(report["domain_general_clause_ids"]),
    "cross_domain_included_clause_count": len(
        report["cross_domain_included_clause_ids"]
    ),
    "bundle_validation_sha256": sha256(validation_path),
    "runforest_report_sha256": sha256(runforest / "build_report.json"),
    "graph_sha256": sha256(runforest / "graph.json"),
    "source_snapshot_unchanged": True,
    "wp4_inputs_unchanged": True,
}
(staging / "BUILD_SUMMARY.json").write_text(
    json.dumps(summary, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY

find "$STAGING" -type f ! -name FINAL_SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$STAGING/FINAL_SHA256SUMS"
printf 'complete\n' > "$STAGING/STATE"
mv "$STAGING" "$OUTPUT_ROOT"
printf 'same_domain_bundle_complete=%s\n' "$OUTPUT_ROOT"
