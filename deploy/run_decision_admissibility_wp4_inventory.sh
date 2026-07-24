#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/work/nautilus}"
CORPUS_ROOT="${CORPUS_ROOT:-/corpus}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/output}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
test -n "$PYTHON_BIN"
test -x "$PYTHON_BIN"
TOOLS="$CODE_ROOT/paper-skills/memory_bundle"
EXPECTED="$CODE_ROOT/coordination/decision_admissibility_wp4_expected_snapshot_20260719.json"

test -d "$CORPUS_ROOT"
test -d "$OUTPUT_ROOT"
test -f "$EXPECTED"
test ! -e "$OUTPUT_ROOT/corpus_manifest.json"
test ! -e "$OUTPUT_ROOT/audit_sidecars"
test ! -e "$OUTPUT_ROOT/splits"
test ! -e "$OUTPUT_ROOT/traces"

export PYTHONPATH="$CODE_ROOT/mlevolve${PYTHONPATH:+:$PYTHONPATH}"
ACTIVE_PROTOCOL="$($PYTHON_BIN -c 'from authority.protocol_registry import ProtocolRegistry; import sys; r=ProtocolRegistry(sys.argv[1]); print(r.get("mlevolve-default", "1").ref().key())' "$CODE_ROOT/mlevolve/config/protocols")"

"$PYTHON_BIN" "$TOOLS/source_fingerprint.py" \
  --root "$CORPUS_ROOT" \
  --output "$OUTPUT_ROOT/source_before.json"

"$PYTHON_BIN" "$TOOLS/build_corpus_manifest.py" \
  --runs-root "$CORPUS_ROOT" \
  --source-repo third_party/MLEvolve \
  --source-commit be034ec81d58e96ca333abb7bda155726aaa3668 \
  --exclude-task spooky-author-identification \
  --expected-snapshot "$EXPECTED" \
  --task-tags "$CODE_ROOT/mlevolve/engine/coldstart/competition_tag_classified.json" \
  --output "$OUTPUT_ROOT/corpus_manifest.json" \
  --report "$OUTPUT_ROOT/corpus_inventory_report.json"

"$PYTHON_BIN" "$TOOLS/audit_corpus.py" \
  --manifest "$OUTPUT_ROOT/corpus_manifest.json" \
  --protocol-registry "$CODE_ROOT/mlevolve/config/protocols" \
  --default-protocol "$ACTIVE_PROTOCOL" \
  --output-dir "$OUTPUT_ROOT/audit_sidecars" \
  --report "$OUTPUT_ROOT/audit_report.json"

"$PYTHON_BIN" "$TOOLS/build_split_manifests.py" \
  --manifest "$OUTPUT_ROOT/corpus_manifest.json" \
  --output-dir "$OUTPUT_ROOT/splits" \
  --split-version be034ec-nonspooky-v1

"$PYTHON_BIN" "$CODE_ROOT/paper-skills/distillation/extract_branches.py" \
  --corpus-manifest "$OUTPUT_ROOT/corpus_manifest.json" \
  --split-manifest "$OUTPUT_ROOT/splits/full.json" \
  --audit-dir "$OUTPUT_ROOT/audit_sidecars" \
  --out-dir "$OUTPUT_ROOT/traces"

"$PYTHON_BIN" "$TOOLS/source_fingerprint.py" \
  --root "$CORPUS_ROOT" \
  --compare "$OUTPUT_ROOT/source_before.json" \
  --output "$OUTPUT_ROOT/source_after.json"

find "$OUTPUT_ROOT" -type f ! -name WP4_INVENTORY_SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$OUTPUT_ROOT/WP4_INVENTORY_SHA256SUMS"

echo "WP4 inventory/audit/split/trace stage completed without source stat drift."
