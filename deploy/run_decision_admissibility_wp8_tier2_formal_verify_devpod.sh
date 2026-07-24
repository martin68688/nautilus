#!/usr/bin/env bash
set -euo pipefail

ROOT="${WP8_TIER2_FORMAL_STAGING_ROOT:-/workspace/decision-admissibility-wp8-tier2-formal-staging-r2}"
SOURCE_ROOT="${WP8_TIER2_FORMAL_SOURCE_ROOT:-/tmp/formal-src-r8}"
REPORTS="$ROOT/reports"

mkdir -p "$REPORTS"
test ! -e "$REPORTS/holdout-verification.exit"
test ! -e "$REPORTS/holdout-verification.finished"

date -u +"%Y-%m-%dT%H:%M:%SZ" > "$REPORTS/holdout-verification.started"
export PYTHONPATH="$SOURCE_ROOT"

set +e
python -m fixed_holdout.formal_verify \
  --root "$ROOT/data/aerial-cactus-identification/wp8-tier2-formal-aerial-stratified-v1-b41166e9d3c2" \
  --output "$REPORTS/holdout-aerial-verification.json" \
  > "$REPORTS/holdout-aerial-verification.log" 2>&1
rc=$?

if [[ "$rc" -eq 0 ]]; then
  python -m fixed_holdout.formal_verify \
    --root "$ROOT/data/mlsp-2013-birds/wp8-tier2-formal-mlsp-grouped-v2-d37a8cc74b35" \
    --output "$REPORTS/holdout-birds-verification.json" \
    > "$REPORTS/holdout-birds-verification.log" 2>&1
  rc=$?
fi

if [[ "$rc" -eq 0 ]]; then
  python -m fixed_holdout.formal_verify \
    --root "$ROOT/data/new-york-city-taxi-fare-prediction/wp8-tier2-formal-nyc-chronological-v1-20497ff0e594" \
    --output "$REPORTS/holdout-taxi-verification.json" \
    > "$REPORTS/holdout-taxi-verification.log" 2>&1
  rc=$?
fi
set -e

printf '%s\n' "$rc" > "$REPORTS/holdout-verification.exit"
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$REPORTS/holdout-verification.finished"
exit "$rc"
