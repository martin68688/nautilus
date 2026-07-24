#!/usr/bin/env bash
set -euo pipefail

STAGING_ROOT="${WP8_TIER2_FORMAL_STAGING_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r2}"
SOURCE_ROOT="${WP8_TIER2_FORMAL_SOURCE_ROOT:-/tmp/formal-publisher-src-r1}"
PUBLICATION_REVISION="${WP8_TIER2_FORMAL_PUBLICATION_REVISION:-r1}"
REPORTS="$STAGING_ROOT/reports"
PUBLISHER="$SOURCE_ROOT/paper-skills/memory_bundle/publish_tier2_formal_child_bundle.py"
PROTOCOLS="$SOURCE_ROOT/mlevolve/config/protocols"
CREATED_AT="${WP8_TIER2_FORMAL_BUNDLE_CREATED_AT:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"

PARENT_AERIAL="/workspace/decision-admissibility-wp8-method-only-certified-r3/bundles/certified-image-task-heldout-method-only-v7"
PARENT_SEED="/workspace/decision-admissibility-wp4-20260719-r1/corrected-r2/bundle-seed-heldout"
ROOT_AERIAL="$STAGING_ROOT/memory_bundles/aerial-cactus-identification/formal-child-$PUBLICATION_REVISION"
ROOT_BIRDS="$STAGING_ROOT/memory_bundles/mlsp-2013-birds/formal-child-$PUBLICATION_REVISION"
ROOT_TAXI="$STAGING_ROOT/memory_bundles/new-york-city-taxi-fare-prediction/formal-child-$PUBLICATION_REVISION"

STARTED="$REPORTS/formal-child-publication-$PUBLICATION_REVISION.started"
FINISHED="$REPORTS/formal-child-publication-$PUBLICATION_REVISION.finished"
EXIT_FILE="$REPORTS/formal-child-publication-$PUBLICATION_REVISION.exit"

mkdir -p "$REPORTS"
test ! -e "$STARTED"
test ! -e "$FINISHED"
test ! -e "$EXIT_FILE"
for root in "$ROOT_AERIAL" "$ROOT_BIRDS" "$ROOT_TAXI"; do
  test ! -e "$root"
done
for source in \
  "$PUBLISHER" \
  "$SOURCE_ROOT/paper-skills/memory_bundle/build_memory_bundle.py" \
  "$PROTOCOLS/random-classification-v1.json" \
  "$PROTOCOLS/grouped-classification-v1.json" \
  "$PROTOCOLS/chronological-regression-v1.json"; do
  test -f "$source"
done
test "$(cat "$REPORTS/holdout-verification.exit")" = "0"
test "$(cat "$REPORTS/preregistration-r4-claim-authority-verification-r1.exit")" = "0"
python -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["verified"] is True and p["source_evidence_verified"] is True' \
  "$REPORTS/preregistration-r4-claim-authority-verification-r1.json"

finish() {
  rc=$?
  printf '%s\n' "$rc" > "$EXIT_FILE"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$FINISHED"
}
trap finish EXIT
date -u +%Y-%m-%dT%H:%M:%SZ > "$STARTED"

export PYTHONPATH="$SOURCE_ROOT/mlevolve:$SOURCE_ROOT/paper-skills/memory_bundle"

python "$PUBLISHER" \
  --parent-bundle "$PARENT_AERIAL" \
  --expected-parent-manifest-sha256 782bbae91c19d5a2e92aa8d7aca9bf6949824ca50505121373a784e6018881f3 \
  --publication-root "$ROOT_AERIAL" \
  --bundle-id decision-admissibility-wp8-tier2-formal-aerial-image-v1 \
  --bundle-version v1 \
  --target-task-id aerial-cactus-identification \
  --target-task-family image_binary_classification \
  --target-domain image \
  --split-mode same-domain-task-heldout \
  --source-clause-id 'clause::certified-replay::bb07a6dee42a0b3063dba514' \
  --source-run-id 20260717_183734_leaf-classification \
  --source-node-id 96e81f07834447928c22df30ce2e5413 \
  --protocol-file "$PROTOCOLS/random-classification-v1.json" \
  --publication-class certified \
  --agent-seed 104729 \
  --agent-seed 130363 \
  --agent-seed 155921 \
  --created-at "$CREATED_AT"

python "$PUBLISHER" \
  --parent-bundle "$PARENT_SEED" \
  --expected-parent-manifest-sha256 8d612b60a83d1469dbb05caad228be791280ebcd68027d0141ccdae1840ae7d8 \
  --publication-root "$ROOT_BIRDS" \
  --bundle-id decision-admissibility-wp8-tier2-formal-birds-audio-v1 \
  --bundle-version v1 \
  --target-task-id mlsp-2013-birds \
  --target-task-family audio_multilabel_classification \
  --target-domain audio \
  --split-mode seed-heldout \
  --source-clause-id 'clause::06c4a3d0455368ac50513172' \
  --source-run-id 20260716_070743_mlsp-2013-birds \
  --source-node-id 91ffa7572b3d4774bc33d0c000e418e4 \
  --protocol-file "$PROTOCOLS/grouped-classification-v1.json" \
  --publication-class provisional \
  --allow-source-audit-issue-code TEMPORAL_SPLIT_LEAKAGE \
  --agent-seed 104729 \
  --agent-seed 130363 \
  --agent-seed 155921 \
  --created-at "$CREATED_AT"

python "$PUBLISHER" \
  --parent-bundle "$PARENT_SEED" \
  --expected-parent-manifest-sha256 8d612b60a83d1469dbb05caad228be791280ebcd68027d0141ccdae1840ae7d8 \
  --publication-root "$ROOT_TAXI" \
  --bundle-id decision-admissibility-wp8-tier2-formal-taxi-tabular-v1 \
  --bundle-version v1 \
  --target-task-id new-york-city-taxi-fare-prediction \
  --target-task-family tabular_regression \
  --target-domain tabular \
  --split-mode seed-heldout \
  --source-clause-id 'clause::641451816c260f793c6cc4b8' \
  --source-run-id 20260716_173558_new-york-city-taxi-fare-prediction \
  --source-node-id 896f942ce32946658c08a46c7591acac \
  --protocol-file "$PROTOCOLS/chronological-regression-v1.json" \
  --publication-class provisional \
  --agent-seed 104729 \
  --agent-seed 130363 \
  --agent-seed 155921 \
  --created-at "$CREATED_AT"
