#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/work/nautilus}"
CORPUS_ROOT="${CORPUS_ROOT:-/corpus}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/output}"
INPUT_ROOT="${INPUT_ROOT:-$OUTPUT_ROOT/reviewed-inputs}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
test -n "$PYTHON_BIN"
test -x "$PYTHON_BIN"
TOOLS="$CODE_ROOT/paper-skills/memory_bundle"

test -f "$OUTPUT_ROOT/corpus_manifest.json"
test -f "$INPUT_ROOT/corpus_drift_review.json"
test -f "$OUTPUT_ROOT/source_before.json"
test -f "$INPUT_ROOT/merged/clauses.jsonl"
test -f "$INPUT_ROOT/merged/containers.json"
test -d "$INPUT_ROOT/binder"
test -f "$INPUT_ROOT/distillation/distillation_report.json"

export PYTHONPATH="$CODE_ROOT/.wp4-python-deps:$TOOLS:$CODE_ROOT/mlevolve${PYTHONPATH:+:$PYTHONPATH}"

DEEPSEEK_MODEL="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["model"])' "$INPUT_ROOT/distillation/distillation_report.json")"
DEEPSEEK_PROMPT_HASH="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["system_prompt_hash"])' "$INPUT_ROOT/distillation/distillation_report.json")"
DETECTOR_VERSION="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["detector_version"])' "$OUTPUT_ROOT/audit_report.json")"

for split in full seed-heldout task-heldout; do
  runforest="$OUTPUT_ROOT/runforest-$split"
  bundle="$OUTPUT_ROOT/bundle-$split"
  archive="$OUTPUT_ROOT/bundle-$split.tar.zst"
  validation_report="$OUTPUT_ROOT/${split}_validation_report.json"

  if [[ ! -e "$runforest" ]]; then
    runforest_stage="$OUTPUT_ROOT/.runforest-$split-staging-$(date +%s)-$RANDOM"
    "$PYTHON_BIN" "$CODE_ROOT/paper-skills/hyper_memory/build_run_forest_memory.py" \
      --corpus-manifest "$OUTPUT_ROOT/corpus_manifest.json" \
      --audit-dir "$OUTPUT_ROOT/audit_sidecars" \
      --sop-clauses "$INPUT_ROOT/merged/clauses.jsonl" \
      --sop-containers "$INPUT_ROOT/merged/containers.json" \
      --split-manifest "$OUTPUT_ROOT/splits/$split.json" \
      --bundle-id "mlevolve-be034ec-nonspooky-$split-v1" \
      --out-dir "$runforest_stage"
    mv "$runforest_stage" "$runforest"
  fi
  test -f "$runforest/graph.json"
  test -f "$runforest/index.npz"
  test -f "$runforest/clause_index.npz"
  test -f "$runforest/build_report.json"
  test -f "$runforest/visibility/clause_metadata.jsonl"
  test -f "$runforest/visibility/precompiled_masks/declared_scope_masks.json"

  if [[ ! -e "$bundle" ]]; then
    "$PYTHON_BIN" "$TOOLS/build_memory_bundle.py" \
      --corpus-manifest "$OUTPUT_ROOT/corpus_manifest.json" \
      --drift-review "$INPUT_ROOT/corpus_drift_review.json" \
      --split-manifest "$OUTPUT_ROOT/splits/$split.json" \
      --splits-dir "$OUTPUT_ROOT/splits" \
      --audit-dir "$OUTPUT_ROOT/audit_sidecars" \
      --runforest-dir "$runforest" \
      --authority-dir "$INPUT_ROOT/binder" \
      --protocol-registry "$CODE_ROOT/mlevolve/config/protocols" \
      --output-dir "$bundle" \
      --bundle-id "mlevolve-be034ec-nonspooky-$split-v1" \
      --bundle-version v1 \
      --authority-policy-version authority_v1 \
      --detector-version "$DETECTOR_VERSION" \
      --deepseek-model "$DEEPSEEK_MODEL" \
      --deepseek-prompt-hash "$DEEPSEEK_PROMPT_HASH" \
      --certification-level raw_audited
  fi

  if [[ ! -e "$archive" ]]; then
    "$PYTHON_BIN" -c \
      'import json,sys; from build_memory_bundle import create_tar_zst; print(json.dumps(create_tar_zst(sys.argv[1], sys.argv[2]), indent=2))' \
      "$bundle" "$archive"
  fi

  if [[ ! -e "$validation_report" ]]; then
    "$PYTHON_BIN" "$TOOLS/validate_memory_bundle.py" \
      --bundle "$bundle" \
      --report "$validation_report"
  else
    "$PYTHON_BIN" -c \
      'import json,sys; report=json.load(open(sys.argv[1])); assert report.get("valid") is True' \
      "$validation_report"
  fi
done

"$PYTHON_BIN" "$TOOLS/source_fingerprint.py" \
  --root "$CORPUS_ROOT" \
  --compare "$OUTPUT_ROOT/source_before.json" \
  --output "$OUTPUT_ROOT/source_after_bundle_build.json"

find "$OUTPUT_ROOT" -type f ! -name WP4_FINAL_SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUTPUT_ROOT/WP4_FINAL_SHA256SUMS"

echo "WP4 full/seed/task raw-audited bundles validated without source stat drift."
