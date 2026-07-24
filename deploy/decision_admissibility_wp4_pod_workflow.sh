#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="ecepxie"
POD="decision-admissibility-corpus-inventory-r1"
YAML="$REPO/deploy/pod-decision-admissibility-corpus-inventory-r1.yaml"
LOCAL_ROOT="${WP4_LOCAL_ROOT:-$REPO/coordination/decision_admissibility_wp4_real_work_20260719_r2_verified}"
POD_OUTPUT_ROOT="${WP4_POD_OUTPUT_ROOT:-/output/corrected-r2}"

case "$POD_OUTPUT_ROOT" in
  /output|/output/*) ;;
  *)
    echo "Refusing Pod output path outside /output: $POD_OUTPUT_ROOT" >&2
    exit 2
    ;;
esac

k() {
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u NO_PROXY \
      -u https_proxy -u http_proxy -u all_proxy -u no_proxy \
      kubectl "$@"
}

require_authorized() {
  if [[ "${WP4_POD_AUTHORIZED:-}" != "YES" ]]; then
    echo "Refusing cluster mutation: set WP4_POD_AUTHORIZED=YES only after explicit user authorization." >&2
    exit 2
  fi
}

sync_code() {
  require_authorized
  k exec -n "$NAMESPACE" "$POD" -- mkdir -p /work/nautilus
  (
    cd "$REPO"
    export COPYFILE_DISABLE=1
    tar \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='._*' \
      -cf - \
      paper-skills/memory_bundle \
      paper-skills/distillation/extract_branches.py \
      paper-skills/hyper_memory/build_run_forest_memory.py \
      paper-skills/hyper_memory/build_sop_taxonomy.py \
      mlevolve/agents/leakage_audit.py \
      mlevolve/authority \
      mlevolve/config/protocols \
      mlevolve/engine/coldstart/competition_tag_classified.json \
      coordination/decision_admissibility_wp4_expected_snapshot_20260719.json \
      deploy/run_decision_admissibility_wp4_inventory.sh \
      deploy/run_decision_admissibility_wp4_bundles.sh
  ) | k exec -i -n "$NAMESPACE" "$POD" -- tar -xf - -C /work/nautilus
  k exec -n "$NAMESPACE" "$POD" -- \
    find /work/nautilus -type f -name '._*' -delete
}

bind_local() {
  test -f "$LOCAL_ROOT/traces/trace_manifest.json"
  test -f "$LOCAL_ROOT/distillation/proposals.jsonl"
  test ! -e "$LOCAL_ROOT/binder"
  test ! -e "$LOCAL_ROOT/merged"
  export PYTHONPATH="$REPO/mlevolve${PYTHONPATH:+:$PYTHONPATH}"
  ACTIVE_PROTOCOL="$($REPO/.venv/bin/python -c 'from authority.protocol_registry import ProtocolRegistry; import sys; r=ProtocolRegistry(sys.argv[1]); print(r.get("mlevolve-default", "1").ref().key())' "$REPO/mlevolve/config/protocols")"
  "$REPO/.venv/bin/python" "$REPO/paper-skills/memory_bundle/bind_sop_clauses.py" \
    --proposals "$LOCAL_ROOT/distillation/proposals.jsonl" \
    --trace-manifest "$LOCAL_ROOT/traces/trace_manifest.json" \
    --output-dir "$LOCAL_ROOT/binder" \
    --active-protocol "$ACTIVE_PROTOCOL"
  "$REPO/.venv/bin/python" "$REPO/paper-skills/memory_bundle/merge_sop_clauses.py" \
    --clauses "$LOCAL_ROOT/binder/clauses.jsonl" \
    --containers "$LOCAL_ROOT/binder/containers.json" \
    --output-dir "$LOCAL_ROOT/merged"
}

case "${1:-}" in
  status)
    k get pod -n "$NAMESPACE" "$POD" -o wide
    ;;
  bundle-status)
    k exec -n "$NAMESPACE" "$POD" -- bash -c \
      'for path in "$1"/runforest-* "$1"/bundle-* "$1"/*_validation_report.json; do [[ -e "$path" ]] || continue; if [[ -d "$path" ]]; then printf "%s\t" "${path##*/}"; find "$path" -type f | wc -l; else printf "%s\tfile\n" "${path##*/}"; fi; done; if command -v zstd >/dev/null; then echo zstd_cli=YES; else echo zstd_cli=NO; fi; if python3 -c "import zstandard" 2>/dev/null; then echo python_zstandard=YES; else echo python_zstandard=NO; fi' \
      _ "$POD_OUTPUT_ROOT"
    ;;
  create)
    require_authorized
    k create -f "$YAML"
    k wait -n "$NAMESPACE" --for=condition=Ready "pod/$POD" --timeout=300s
    sync_code
    ;;
  sync-code)
    sync_code
    ;;
  prepare-archive-runtime)
    require_authorized
    k exec -n "$NAMESPACE" "$POD" -- bash -lc '
      set -euo pipefail
      target=/work/nautilus/.wp4-python-deps
      version=0.23.0
      if [[ ! -d "$target" ]]; then
        stage="${target}.staging-${RANDOM}"
        test ! -e "$stage"
        python3 -m pip install \
          --disable-pip-version-check \
          --no-deps \
          --target "$stage" \
          "zstandard==$version"
        PYTHONPATH="$stage" python3 -c \
          "import zstandard; assert zstandard.__version__ == \"0.23.0\""
        mv "$stage" "$target"
      fi
      PYTHONPATH="$target" python3 -c \
        "import zstandard; assert zstandard.__version__ == \"0.23.0\"; print(\"zstandard_runtime=0.23.0\")"
    '
    ;;
  inventory)
    require_authorized
    if [[ "$POD_OUTPUT_ROOT" != "/output" ]]; then
      k exec -n "$NAMESPACE" "$POD" -- mkdir "$POD_OUTPUT_ROOT"
    fi
    k exec -n "$NAMESPACE" "$POD" -- \
      env OUTPUT_ROOT="$POD_OUTPUT_ROOT" \
      /work/nautilus/deploy/run_decision_admissibility_wp4_inventory.sh
    ;;
  fetch-inventory)
    mkdir -p "$LOCAL_ROOT"
    for name in \
      corpus_manifest.json \
      corpus_inventory_report.json \
      audit_report.json \
      source_before.json \
      source_after.json \
      WP4_INVENTORY_SHA256SUMS; do
      k cp "$NAMESPACE/$POD:$POD_OUTPUT_ROOT/$name" "$LOCAL_ROOT/$name"
    done
    k cp "$NAMESPACE/$POD:$POD_OUTPUT_ROOT/traces" "$LOCAL_ROOT/traces"
    k cp "$NAMESPACE/$POD:$POD_OUTPUT_ROOT/splits" "$LOCAL_ROOT/splits"
    ;;
  distill-local)
    test -n "${DEEPSEEK_API_KEY:-}"
    test -f "$LOCAL_ROOT/traces/trace_manifest.json"
    if [[ -e "$LOCAL_ROOT/distillation" ]]; then
      test -d "$LOCAL_ROOT/distillation"
    fi
    test ! -e "$LOCAL_ROOT/binder"
    test ! -e "$LOCAL_ROOT/merged"
    export PYTHONPATH="$REPO/mlevolve${PYTHONPATH:+:$PYTHONPATH}"
    ACTIVE_PROTOCOL="$($REPO/.venv/bin/python -c 'from authority.protocol_registry import ProtocolRegistry; import sys; r=ProtocolRegistry(sys.argv[1]); print(r.get("mlevolve-default", "1").ref().key())' "$REPO/mlevolve/config/protocols")"
    "$REPO/.venv/bin/python" "$REPO/paper-skills/memory_bundle/distill_sop_clauses.py" \
      --trace-manifest "$LOCAL_ROOT/traces/trace_manifest.json" \
      --trace-root "$LOCAL_ROOT/traces" \
      --output-dir "$LOCAL_ROOT/distillation" \
      --model "${DEEPSEEK_MODEL:-deepseek-chat}" \
      --temperature 0 \
      --allow-network
    bind_local
    ;;
  bind-local)
    bind_local
    ;;
  push-reviewed-inputs)
    require_authorized
    test -f "$LOCAL_ROOT/corpus_drift_review.json"
    test -d "$LOCAL_ROOT/distillation"
    test -d "$LOCAL_ROOT/binder"
    test -d "$LOCAL_ROOT/merged"
    if [[ -e "$LOCAL_ROOT/reviewed_inputs_SHA256SUMS" ]]; then
      (
        cd "$LOCAL_ROOT"
        shasum -a 256 -c reviewed_inputs_SHA256SUMS >/dev/null
      )
    else
      (
        cd "$LOCAL_ROOT"
        find corpus_drift_review.json distillation binder merged -type f -print0 \
          | LC_ALL=C sort -z \
          | xargs -0 shasum -a 256
      ) > "$LOCAL_ROOT/reviewed_inputs_SHA256SUMS"
    fi
    REMOTE_FINAL="$POD_OUTPUT_ROOT/reviewed-inputs"
    REMOTE_STAGE="$POD_OUTPUT_ROOT/.reviewed-inputs-staging-$(date +%s)-$RANDOM"
    k exec -n "$NAMESPACE" "$POD" -- test ! -e "$REMOTE_FINAL"
    k exec -n "$NAMESPACE" "$POD" -- mkdir "$REMOTE_STAGE"
    (
      cd "$LOCAL_ROOT"
      export COPYFILE_DISABLE=1
      tar --exclude='._*' -czf - \
        corpus_drift_review.json \
        distillation \
        binder \
        merged \
        reviewed_inputs_SHA256SUMS
    ) | k exec -i -n "$NAMESPACE" "$POD" -- \
      tar -xzf - -C "$REMOTE_STAGE"
    k exec -n "$NAMESPACE" "$POD" -- bash -c \
      'cd "$1" && sha256sum -c reviewed_inputs_SHA256SUMS > reviewed_inputs_verification.log' \
      _ "$REMOTE_STAGE"
    k exec -n "$NAMESPACE" "$POD" -- \
      mv "$REMOTE_STAGE" "$REMOTE_FINAL"
    k exec -n "$NAMESPACE" "$POD" -- bash -c \
      'shopt -s nullglob; stale=("$1"/.reviewed-inputs-staging-*); if ((${#stale[@]})); then mkdir -p "$2"; for path in "${stale[@]}"; do mv "$path" "$2/"; done; fi' \
      _ "$POD_OUTPUT_ROOT" "/output/decision-admissibility-wp4-transfer-attempts"
    k exec -n "$NAMESPACE" "$POD" -- tail -n 1 \
      "$REMOTE_FINAL/reviewed_inputs_verification.log"
    ;;
  bundles)
    require_authorized
    k exec -n "$NAMESPACE" "$POD" -- \
      env OUTPUT_ROOT="$POD_OUTPUT_ROOT" \
      /work/nautilus/deploy/run_decision_admissibility_wp4_bundles.sh
    ;;
  quarantine-invalid-seed-prefixed)
    require_authorized
    k exec -n "$NAMESPACE" "$POD" -- bash -c '
      set -euo pipefail
      root="$1"
      destination=/output/decision-admissibility-wp4-invalid-seed-pre-fix-v1
      test ! -e "$destination"
      test -d "$root/runforest-seed-heldout"
      test -d "$root/bundle-seed-heldout"
      test -f "$root/bundle-seed-heldout.tar.zst"
      test -f "$root/seed-heldout_validation_report.json"
      python3 -c '\''import json,sys; r=json.load(open(sys.argv[1])); assert r["input_clause_count"] > 0; assert r["included_clause_count"] == 0; assert "heldout_task_scope" in {x["reason"] for x in r["excluded_clauses"]}'\'' "$root/runforest-seed-heldout/build_report.json"
      mkdir "$destination"
      mv \
        "$root/runforest-seed-heldout" \
        "$root/bundle-seed-heldout" \
        "$root/bundle-seed-heldout.tar.zst" \
        "$root/seed-heldout_validation_report.json" \
        "$destination/"
      echo "invalid_seed_prefixed_artifacts=$destination"
    ' _ "$POD_OUTPUT_ROOT"
    ;;
  fetch-final-reports)
    mkdir -p "$LOCAL_ROOT/final_reports"
    for name in \
      full_validation_report.json \
      seed-heldout_validation_report.json \
      task-heldout_validation_report.json \
      WP4_FINAL_SHA256SUMS \
      source_after_bundle_build.json; do
      k cp "$NAMESPACE/$POD:$POD_OUTPUT_ROOT/$name" "$LOCAL_ROOT/final_reports/$name"
    done
    ;;
  fetch-bundle-evidence)
    mkdir -p "$LOCAL_ROOT/final_reports"
    for split in full seed-heldout task-heldout; do
      for specification in \
        "bundle-$split/manifest.json:${split}_bundle_manifest.json" \
        "bundle-$split/reports/build_report.json:${split}_bundle_build_report.json" \
        "runforest-$split/build_report.json:${split}_runforest_build_report.json"; do
        remote="${specification%%:*}"
        local_name="${specification##*:}"
        if [[ -e "$LOCAL_ROOT/final_reports/$local_name" ]]; then
          if "$REPO/.venv/bin/python" -c \
              'import json,sys; json.load(open(sys.argv[1]))' \
              "$LOCAL_ROOT/final_reports/$local_name"; then
            continue
          fi
          mv "$LOCAL_ROOT/final_reports/$local_name" \
            "$LOCAL_ROOT/final_reports/$local_name.partial-eof-$(date +%s)"
        fi
        k cp "$NAMESPACE/$POD:$POD_OUTPUT_ROOT/$remote" \
          "$LOCAL_ROOT/final_reports/$local_name"
      done
    done
    ;;
  delete)
    require_authorized
    k delete pod -n "$NAMESPACE" "$POD" --wait=true
    ;;
  *)
    echo "usage: $0 {status|bundle-status|create|sync-code|prepare-archive-runtime|inventory|fetch-inventory|distill-local|bind-local|push-reviewed-inputs|bundles|quarantine-invalid-seed-prefixed|fetch-final-reports|fetch-bundle-evidence|delete}" >&2
    exit 2
    ;;
esac
