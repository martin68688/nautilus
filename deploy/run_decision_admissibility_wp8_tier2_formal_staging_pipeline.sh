#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="${WP8_FORMAL_CONTROL_ROOT:?required}"
SOURCE_ROOT="${WP8_FORMAL_SOURCE_ROOT:?required}"
ARTIFACT_ROOT="${WP8_FORMAL_ARTIFACT_ROOT:?required}"
STAGING_ROOT="${WP8_FORMAL_STAGING_ROOT:?required}"
OUTPUT_ROOT="${WP8_FORMAL_OUTPUT_ROOT:?required}"
GATE_ROOT="${WP8_FORMAL_GATE_ROOT:?required}"
PIPELINE_ROOT="${WP8_FORMAL_PIPELINE_ROOT:?required}"
SUPERSEDED_STAGING_ABORT="${WP8_FORMAL_SUPERSEDED_STAGING_ABORT:?required}"
SUPERSEDED_COMPATIBILITY_ABORT="${WP8_FORMAL_SUPERSEDED_COMPATIBILITY_ABORT:?required}"
SUPERSEDED_NODE_EVICTION_ABORT="${WP8_FORMAL_SUPERSEDED_NODE_EVICTION_ABORT:?required}"
SUPERSEDED_RANKING_DIAGNOSTIC="${WP8_FORMAL_SUPERSEDED_RANKING_DIAGNOSTIC:?required}"
SUPERSEDED_PRELAUNCH_ABORT="${WP8_FORMAL_SUPERSEDED_PRELAUNCH_ABORT:?required}"
SUPERSEDED_STAGING_TRANSFER_ABORT="${WP8_FORMAL_SUPERSEDED_STAGING_TRANSFER_ABORT:?required}"
SUPERSEDED_SECOND_PRELAUNCH_ABORT="${WP8_FORMAL_SUPERSEDED_SECOND_PRELAUNCH_ABORT:?required}"
SUPERSEDED_PHASE_POLLUTION_ABORT="${WP8_FORMAL_SUPERSEDED_PHASE_POLLUTION_ABORT:?required}"
SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC="${WP8_FORMAL_SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC:?required}"
SUPERSEDED_R2_SCHEDULING_ABORT="${WP8_FORMAL_SUPERSEDED_R2_SCHEDULING_ABORT:?required}"

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
export PYTHONPATH="$CONTROL_ROOT/mlevolve:$CONTROL_ROOT"

python "$CONTROL_ROOT/paper-skills/memory_bundle/build_tier2_formal_staging.py" \
  --source-root "$SOURCE_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" \
  --staging-root "$STAGING_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --preregistration-r1 "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r1.json" \
  --preregistration-r2 "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r2.json" \
  --preregistration-r3 "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r3.json" \
  --preregistration-r4 "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260722_r4.json" \
  --preregistration-r5 "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_preregistration_20260723_r5.json" \
  --superseded-evidence "$SUPERSEDED_STAGING_ABORT" \
  --superseded-evidence "$SUPERSEDED_COMPATIBILITY_ABORT" \
  --superseded-evidence "$SUPERSEDED_NODE_EVICTION_ABORT" \
  --superseded-evidence "$SUPERSEDED_RANKING_DIAGNOSTIC" \
  --superseded-evidence "$SUPERSEDED_PRELAUNCH_ABORT" \
  --superseded-evidence "$SUPERSEDED_STAGING_TRANSFER_ABORT" \
  --superseded-evidence "$SUPERSEDED_SECOND_PRELAUNCH_ABORT" \
  --superseded-evidence "$SUPERSEDED_PHASE_POLLUTION_ABORT" \
  --superseded-evidence "$SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC" \
  --superseded-evidence "$SUPERSEDED_R2_SCHEDULING_ABORT" \
  --failed-formal-evidence "$CONTROL_ROOT/coordination/decision_admissibility_wp8_tier2_formal_r8_authority_failure_diagnostic_20260723.json"

test ! -e "$GATE_ROOT"
mkdir -p "$GATE_ROOT"
python "$CONTROL_ROOT/paper-skills/memory_bundle/verify_tier2_formal_staging.py" \
  --staging-root "$STAGING_ROOT" \
  --repo-root "$CONTROL_ROOT" \
  --output "$GATE_ROOT/STAGING_STOP_GATE.json"
chmod -R a-w "$GATE_ROOT"
