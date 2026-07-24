#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD="${1:-decision-admissibility-wp8-bundle-cpu-r1}"
NAMESPACE="${WP8_TIER2_NAMESPACE:-ecepxie}"
EXPECTED_HEAD="${WP8_TIER2_EXPECTED_HEAD:-b47dab63b7861f3ea0871094d6dd07b77e6b81a4}"
REMOTE_ROOT="${WP8_TIER2_SOURCE_ROOT:-/workspace/decision-admissibility-wp8-tier2-canary-r10-source}"
OUTPUT_ROOT="${WP8_TIER2_OUTPUT_ROOT:-/workspace/decision-admissibility-wp8-tier2-canary-r10-output}"
PARENT_SOURCE_SHA256="${WP8_TIER2_PARENT_SOURCE_SHA256:-a92c5e25778fad43265b42cfd4334cfc8f1c379df8dc36d5b43e378d573aedf0}"

cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{u}')" = "$EXPECTED_HEAD"

kubectl_clean() {
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u NO_PROXY \
      -u https_proxy -u http_proxy -u all_proxy -u no_proxy \
      kubectl "$@"
}

test "$(kubectl_clean get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}')" = "Running"
kubectl_clean exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "test ! -e '$REMOTE_ROOT' && test ! -e '$OUTPUT_ROOT' && mkdir -p '$REMOTE_ROOT' '$OUTPUT_ROOT'"

# Start from the pushed checkpoint, excluding historical runs, data, inference
# exports, and generated caches. The current WP implementation is then copied
# through an explicit source-only overlay list; no user output asset is staged.
git archive --format=tar.gz "$EXPECTED_HEAD" -- \
  mlevolve/__init__.py mlevolve/run.py \
  mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
  mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils \
  tests \
  paper-skills/eval_skill_memory/clean_run_allowlist.json \
  paper-skills/eval_skill_memory/run_identity_registry_v1.json \
  ':(exclude)**/__pycache__' | \
  kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
    tar -xzf - -C "$REMOTE_ROOT"

overlay_list="$(mktemp)"
trap 'rm -f "$overlay_list"' EXIT
{
  git diff HEAD --name-only --diff-filter=ACMRTUXB -- \
    mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
    mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils \
    mlevolve/run.py \
    tests/authority/test_domain_transfer_scope.py \
    tests/test_stage_aware_hybrid_memory.py
  git ls-files --others --exclude-standard -- \
    mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
    mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils \
    tests/__init__.py tests/authority/__init__.py \
    tests/authority/clean_replay_helpers.py \
    tests/authority/test_candidate_execution_contract.py \
    tests/authority/test_domain_transfer_scope.py \
    tests/authority/test_tier2_canary_launcher_static.py \
    tests/test_stage_aware_hybrid_memory.py
  printf '%s\n' \
    deploy/run_decision_admissibility_wp8_tier2_train_devpod.sh \
    deploy/run_decision_admissibility_wp8_tier2_evaluator_devpod.sh \
    deploy/stage_decision_admissibility_wp8_tier2_canary.sh \
    deploy/devpod-decision-admissibility-wp8-tier2-train-a100x1-r1.yaml \
    deploy/devpod-decision-admissibility-wp8-tier2-evaluator-cpu-r1.yaml
} | LC_ALL=C sort -u > "$overlay_list"

test -s "$overlay_list"
while IFS= read -r path; do
  test -f "$path"
done < "$overlay_list"

COPYFILE_DISABLE=1 tar --no-xattrs -czf - -T "$overlay_list" | \
  kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
    tar -xzf - -C "$REMOTE_ROOT"
kubectl_clean cp "$overlay_list" \
  "$NAMESPACE/$POD:$REMOTE_ROOT/WP8_TIER2_OVERLAY_PATHS.txt"

kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
  python - "$REMOTE_ROOT" "$EXPECTED_HEAD" "$PARENT_SOURCE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
base_commit = sys.argv[2]
parent_source_sha256 = sys.argv[3]
excluded = {"WP8_TIER2_SOURCE_MANIFEST.json"}
files = {}
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name not in excluded:
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema": "decision_admissibility_wp8_tier2_source_snapshot_v1",
    "base_commit": base_commit,
    "parent_source_sha256": parent_source_sha256,
    "purpose": "wp8_tier2_no_memory_vs_full_canary_r10_contract_bound_runtime_repairs",
    "overlay_paths": [
        line.strip()
        for line in (root / "WP8_TIER2_OVERLAY_PATHS.txt").read_text().splitlines()
        if line.strip()
    ],
    "file_count": len(files),
    "file_hashes": files,
    "excluded_user_assets": [
        "mlevolve/data",
        "mlevolve/runs",
        "mlevolve/inference",
        "outputs",
        "coordination",
    ],
    "source_sha256": "",
}
unsigned = {key: value for key, value in payload.items() if key != "source_sha256"}
payload["source_sha256"] = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
(root / "WP8_TIER2_SOURCE_MANIFEST.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "base_commit": base_commit,
    "file_count": len(files),
    "overlay_count": len(payload["overlay_paths"]),
    "source_sha256": payload["source_sha256"],
}, sort_keys=True))
PY

kubectl_clean exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "chmod -R a-w '$REMOTE_ROOT' && chmod 0770 '$OUTPUT_ROOT' && \
   test -s '$REMOTE_ROOT/WP8_TIER2_SOURCE_MANIFEST.json' && \
   test -z \"\$(find '$REMOTE_ROOT' -type d -name __pycache__ -print -quit)\" && \
   test -z \"\$(find '$REMOTE_ROOT' -perm /222 -print -quit)\" && \
   test -z \"\$(find '$OUTPUT_ROOT' -mindepth 1 -print -quit)\""
