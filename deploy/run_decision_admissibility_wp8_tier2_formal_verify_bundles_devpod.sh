#!/usr/bin/env bash
set -euo pipefail

STAGING_ROOT="${WP8_TIER2_FORMAL_STAGING_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r2}"
SOURCE_ROOT="${WP8_TIER2_FORMAL_SOURCE_ROOT:-/tmp/formal-publisher-src-r2}"
PUBLICATION_REVISION="${WP8_TIER2_FORMAL_PUBLICATION_REVISION:-r2}"
VERIFICATION_REVISION="${WP8_TIER2_FORMAL_VERIFICATION_REVISION:-r1}"
REPORTS="$STAGING_ROOT/reports"
VERIFIER="$SOURCE_ROOT/paper-skills/memory_bundle/verify_tier2_formal_child_bundle.py"

ROOT_AERIAL="$STAGING_ROOT/memory_bundles/aerial-cactus-identification/formal-child-$PUBLICATION_REVISION"
ROOT_BIRDS="$STAGING_ROOT/memory_bundles/mlsp-2013-birds/formal-child-$PUBLICATION_REVISION"
ROOT_TAXI="$STAGING_ROOT/memory_bundles/new-york-city-taxi-fare-prediction/formal-child-$PUBLICATION_REVISION"

PREFIX="formal-child-independent-verification-$PUBLICATION_REVISION-$VERIFICATION_REVISION"
STARTED="$REPORTS/$PREFIX.started"
FINISHED="$REPORTS/$PREFIX.finished"
EXIT_FILE="$REPORTS/$PREFIX.exit"
OUT_AERIAL="$REPORTS/$PREFIX-aerial.json"
OUT_BIRDS="$REPORTS/$PREFIX-birds.json"
OUT_TAXI="$REPORTS/$PREFIX-taxi.json"

test -f "$VERIFIER"
test "$(cat "$REPORTS/formal-child-publication-$PUBLICATION_REVISION.exit")" = "0"
for root in "$ROOT_AERIAL" "$ROOT_BIRDS" "$ROOT_TAXI"; do
  test -d "$root"
done
for output in \
  "$STARTED" "$FINISHED" "$EXIT_FILE" \
  "$OUT_AERIAL" "$OUT_BIRDS" "$OUT_TAXI"; do
  test ! -e "$output"
done

finish() {
  rc=$?
  printf '%s\n' "$rc" > "$EXIT_FILE"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$FINISHED"
}
trap finish EXIT
date -u +%Y-%m-%dT%H:%M:%SZ > "$STARTED"

export PYTHONPATH="$SOURCE_ROOT/mlevolve:$SOURCE_ROOT/paper-skills/memory_bundle"

python "$VERIFIER" \
  --publication-root "$ROOT_AERIAL" \
  --expected-parent-bundle-id mlevolve-be034ec-image-aerial-task-heldout-certified-replay-method-only-v7 \
  --expected-parent-manifest-sha256 782bbae91c19d5a2e92aa8d7aca9bf6949824ca50505121373a784e6018881f3 \
  --target-task-id aerial-cactus-identification \
  --target-task-family image_binary_classification \
  --target-domain image \
  --split-mode same-domain-task-heldout \
  --source-clause-id 'clause::certified-replay::bb07a6dee42a0b3063dba514' \
  --source-run-id 20260717_183734_leaf-classification \
  --source-node-id 96e81f07834447928c22df30ce2e5413 \
  --publication-class certified \
  --protocol-ref 'random-classification@1#ecf870583bf524f66b11ea6b1e33829351c7c761c1a842d22338a95bd976c9cc' \
  --agent-seed 104729 \
  --agent-seed 130363 \
  --agent-seed 155921 \
  --output "$OUT_AERIAL"

python "$VERIFIER" \
  --publication-root "$ROOT_BIRDS" \
  --expected-parent-bundle-id mlevolve-be034ec-nonspooky-seed-heldout-v1 \
  --expected-parent-manifest-sha256 8d612b60a83d1469dbb05caad228be791280ebcd68027d0141ccdae1840ae7d8 \
  --target-task-id mlsp-2013-birds \
  --target-task-family audio_multilabel_classification \
  --target-domain audio \
  --split-mode seed-heldout \
  --source-clause-id 'clause::06c4a3d0455368ac50513172' \
  --source-run-id 20260716_070743_mlsp-2013-birds \
  --source-node-id 91ffa7572b3d4774bc33d0c000e418e4 \
  --publication-class provisional \
  --protocol-ref 'grouped-classification@1#901703060d3f2dd756cc29339a645583410a1d69ac4f2b6ccd53f8631b411382' \
  --expected-parent-validation-mode legacy_lineage_migration \
  --allow-source-audit-issue-code TEMPORAL_SPLIT_LEAKAGE \
  --agent-seed 104729 \
  --agent-seed 130363 \
  --agent-seed 155921 \
  --output "$OUT_BIRDS"

python "$VERIFIER" \
  --publication-root "$ROOT_TAXI" \
  --expected-parent-bundle-id mlevolve-be034ec-nonspooky-seed-heldout-v1 \
  --expected-parent-manifest-sha256 8d612b60a83d1469dbb05caad228be791280ebcd68027d0141ccdae1840ae7d8 \
  --target-task-id new-york-city-taxi-fare-prediction \
  --target-task-family tabular_regression \
  --target-domain tabular \
  --split-mode seed-heldout \
  --source-clause-id 'clause::641451816c260f793c6cc4b8' \
  --source-run-id 20260716_173558_new-york-city-taxi-fare-prediction \
  --source-node-id 896f942ce32946658c08a46c7591acac \
  --publication-class provisional \
  --protocol-ref 'chronological-regression@1#bfc61957b422df5cf09dcb37cffe06aae2ccd2b11db4fee0721b90a2bc6dbf04' \
  --expected-parent-validation-mode legacy_lineage_migration \
  --agent-seed 104729 \
  --agent-seed 130363 \
  --agent-seed 155921 \
  --output "$OUT_TAXI"
