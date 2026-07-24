#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="${CONTROL_ROOT:?required}"
SOURCE_ROOT="${SOURCE_ROOT:?required}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:?required}"
PARENT_STAGING_ROOT="${PARENT_STAGING_ROOT:?required}"
PARENT_GATE="${PARENT_GATE:?required}"
COMPLETED_OUTPUT_ROOT="${COMPLETED_OUTPUT_ROOT:?required}"
STAGING_ROOT="${STAGING_ROOT:?required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?required}"
GATE_ROOT="${GATE_ROOT:?required}"
PIPELINE_ROOT="${PIPELINE_ROOT:?required}"
RECOVERY_RECEIPT="${RECOVERY_RECEIPT:?required}"

test -d "$PIPELINE_ROOT"
test -z "$(find "$PIPELINE_ROOT" -mindepth 1 -print -quit)"
exec > "$PIPELINE_ROOT/PIPELINE.log" 2>&1
printf 'running\n' > "$PIPELINE_ROOT/STATE"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$PIPELINE_ROOT/STARTED_AT"

finish() {
  local rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$PIPELINE_ROOT/EXIT_CODE"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$PIPELINE_ROOT/FINISHED_AT"
  if [[ "$rc" -eq 0 ]]; then
    printf 'complete\n' > "$PIPELINE_ROOT/STATE"
  else
    printf 'failed\n' > "$PIPELINE_ROOT/STATE"
  fi
  exit "$rc"
}
trap finish EXIT

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CONTROL_ROOT/mlevolve:$CONTROL_ROOT:$CONTROL_ROOT/paper-skills/memory_bundle"

python "$CONTROL_ROOT/paper-skills/memory_bundle/build_tier2_formal_continuation_r5_staging.py" \
  --source-root "$SOURCE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --parent-staging-root "$PARENT_STAGING_ROOT" \
  --parent-gate "$PARENT_GATE" \
  --completed-output-root "$COMPLETED_OUTPUT_ROOT" \
  --staging-root "$STAGING_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --preregistration "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r1.json" \
  --preregistration "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r2.json" \
  --preregistration "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r3.json" \
  --preregistration "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r4.json" \
  --preregistration "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r5.json" \
  --preregistration "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r6.json" \
  --preregistration "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r7.json" \
  --continuation-amendment "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r7.json" \
  --continuation-verification "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260723_r7.json" \
  --completed-freeze "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_completed_blocks_freeze_20260723_r1.json" \
  --r9-amendment "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r9.json" \
  --r9-verification "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260723_r9.json" \
  --recovery-receipt "$RECOVERY_RECEIPT"

mkdir "$GATE_ROOT"
python "$CONTROL_ROOT/paper-skills/memory_bundle/verify_tier2_formal_continuation_r5_staging.py" \
  --staging-root "$STAGING_ROOT" \
  --repo-root "$CONTROL_ROOT" \
  --r9-amendment "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r9.json" \
  --r9-verification "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260723_r9.json" \
  --r7-verification "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_verification_20260723_r7.json" \
  --recovery-receipt "$RECOVERY_RECEIPT" \
  --output "$GATE_ROOT/STAGING_STOP_GATE.json"
chmod -R a-w "$GATE_ROOT"
