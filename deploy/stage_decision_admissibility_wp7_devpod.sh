#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD="${1:-decision-admissibility-wp7-shadow-a100x1-r2}"
NAMESPACE="${WP7_NAMESPACE:-ecepxie}"
EXPECTED_HEAD="b47dab63b7861f3ea0871094d6dd07b77e6b81a4"
REMOTE_ROOT="${WP7_REMOTE_ROOT:-/work/decision-admissibility-wp7-aerial-r15b-canary-source}"
PARENT_SOURCE_SHA256="${WP7_PARENT_SOURCE_SHA256:-}"
SOURCE_REMEDIATION_REASON="${WP7_SOURCE_REMEDIATION_REASON:-}"
CHANGED_PATHS_FROM_PARENT="${WP7_CHANGED_PATHS_FROM_PARENT:-}"

cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD"
test "$(git rev-parse '@{u}')" = "$EXPECTED_HEAD"

kubectl_clean() {
  env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u NO_PROXY \
      -u https_proxy -u http_proxy -u all_proxy -u no_proxy \
      kubectl "$@"
}

status="$(kubectl_clean get pod "$POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}')"
test "$status" = "Running"
kubectl_clean exec "$POD" -n "$NAMESPACE" -- \
  bash -lc "test ! -e '$REMOTE_ROOT' && mkdir -p '$REMOTE_ROOT'"

# Stage only the source/config/test surface needed by the WP7 shadow run.  The
# baseline commit also tracks historical runs, datasets, inference exports, and
# unrelated prebuilt graph/index artifacts.  The frozen legacy RunForest
# graph/index pair remains in scope because the authority migration regression
# suite consumes it as a required fixture.
# Compress the archive before crossing kubectl exec.  A compact source mode is
# available for corrected canaries: it excludes historical runs/data and large
# hyper-memory artifacts while retaining every runtime package and test used by
# the synthetic/enforced canary.  This keeps the transport below the kubelet
# exec EOF failure threshold and never changes the source manifest semantics.
if [[ "${WP7_COMPACT_STAGE:-false}" == "true" ]]; then
  git archive --format=tar.gz "$EXPECTED_HEAD" -- \
      mlevolve/__init__.py mlevolve/run.py \
      mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
      mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils \
      tests \
      paper-skills/eval_skill_memory/clean_run_allowlist.json \
      paper-skills/eval_skill_memory/run_identity_registry_v1.json \
      paper-skills/hyper_memory/run_forest_graph.json \
      paper-skills/hyper_memory/run_forest_index.npz \
      ':(exclude)**/__pycache__' | \
    kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
      tar -xzf - -C "$REMOTE_ROOT"
else
  git archive --format=tar.gz "$EXPECTED_HEAD" -- \
      mlevolve tests paper-skills/distillation paper-skills/hyper_memory \
      paper-skills/eval_skill_memory/clean_run_allowlist.json \
      paper-skills/eval_skill_memory/run_identity_registry_v1.json \
      ':(exclude)mlevolve/runs' ':(exclude)mlevolve/data' \
      ':(exclude)mlevolve/inference' ':(exclude)**/__pycache__' \
      ':(glob,exclude)paper-skills/distillation/**/*.json' \
      ':(glob,exclude)paper-skills/distillation/**/*.jsonl' \
      ':(glob,exclude)paper-skills/distillation/**/*.npz' \
      ':(glob,exclude)paper-skills/distillation/**/*.joblib' \
      ':(glob,exclude)paper-skills/hyper_memory/**/*.jsonl' \
      ':(glob,exclude)paper-skills/hyper_memory/**/*.joblib' \
      ':(exclude)paper-skills/hyper_memory/hyper_graph.json' \
      ':(exclude)paper-skills/hyper_memory/hyper_index.npz' | \
    kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
      tar -xzf - -C "$REMOTE_ROOT"
fi

kubectl_clean exec "$POD" -n "$NAMESPACE" -- bash -lc \
  "test ! -e '$REMOTE_ROOT/mlevolve/runs' && \
   test ! -e '$REMOTE_ROOT/mlevolve/data' && \
   test ! -e '$REMOTE_ROOT/mlevolve/inference' && \
   test -s '$REMOTE_ROOT/paper-skills/eval_skill_memory/clean_run_allowlist.json' && \
   test -s '$REMOTE_ROOT/paper-skills/eval_skill_memory/run_identity_registry_v1.json' && \
   test -s '$REMOTE_ROOT/paper-skills/hyper_memory/run_forest_graph.json' && \
   test -s '$REMOTE_ROOT/paper-skills/hyper_memory/run_forest_index.npz'"

overlay_list="$(mktemp)"
trap 'rm -f "$overlay_list"' EXIT
{
  if [[ "${WP7_COMPACT_STAGE:-false}" == "true" ]]; then
    git diff HEAD --name-only --diff-filter=ACMRTUXB -- \
      mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
      mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils \
      mlevolve/run.py \
      paper-skills/memory_bundle tests
    git ls-files --others --exclude-standard -- \
      mlevolve/agents mlevolve/analysis mlevolve/authority mlevolve/config \
      mlevolve/engine mlevolve/fixed_holdout mlevolve/llm mlevolve/utils \
      paper-skills/memory_bundle tests
  else
    git diff HEAD --name-only --diff-filter=ACMRTUXB -- \
      mlevolve/agents mlevolve/authority mlevolve/config mlevolve/engine \
      mlevolve/fixed_holdout paper-skills/distillation paper-skills/hyper_memory \
      paper-skills/memory_bundle tests
    git ls-files --others --exclude-standard -- \
      mlevolve/agents mlevolve/authority mlevolve/config mlevolve/engine \
      mlevolve/fixed_holdout paper-skills/distillation paper-skills/hyper_memory \
      paper-skills/memory_bundle tests
  fi
  printf '%s\n' \
    deploy/build_decision_admissibility_wp7_same_domain_bundle_devpod.sh \
    deploy/run_decision_admissibility_wp7_canary_devpod.sh \
    deploy/run_decision_admissibility_wp7_shadow_devpod.sh \
    deploy/stage_decision_admissibility_wp7_devpod.sh \
    deploy/devpod-decision-admissibility-wp7-canary-a100x1-r7.yaml \
    deploy/devpod-decision-admissibility-wp8-bundle-cpu-r1.yaml \
    deploy/run_decision_admissibility_wp8_tier0_devpod.sh
} | LC_ALL=C sort -u > "$overlay_list"

test -s "$overlay_list"
while IFS= read -r path; do
  test -f "$path"
done < "$overlay_list"

COPYFILE_DISABLE=1 tar --no-xattrs -czf - -T "$overlay_list" | \
  kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
    tar -xzf - -C "$REMOTE_ROOT"
kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
  bash -lc "cat > '$REMOTE_ROOT/WP7_OVERLAY_PATHS.txt'" < "$overlay_list"

kubectl_clean exec -i "$POD" -n "$NAMESPACE" -- \
  python - "$REMOTE_ROOT" "$EXPECTED_HEAD" "$PARENT_SOURCE_SHA256" \
    "$SOURCE_REMEDIATION_REASON" "$CHANGED_PATHS_FROM_PARENT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
base_commit = sys.argv[2]
parent_source_sha256 = sys.argv[3].strip()
remediation_reason = sys.argv[4].strip()
changed_paths_from_parent = [
    value.strip()
    for value in sys.argv[5].split(":")
    if value.strip()
]
excluded_names = {"WP7_SOURCE_MANIFEST.json"}
files = {}
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name in excluded_names:
        continue
    relative = str(path.relative_to(root))
    files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
payload = {
    "schema": "decision_admissibility_wp7_source_snapshot_v1",
    "base_commit": base_commit,
    "overlay_paths": [
        line.strip()
        for line in (root / "WP7_OVERLAY_PATHS.txt").read_text().splitlines()
        if line.strip()
    ],
    "file_count": len(files),
    "file_hashes": files,
}
if parent_source_sha256:
    if len(parent_source_sha256) != 64:
        raise ValueError("Parent source SHA256 must be a 64-character digest")
    if not remediation_reason:
        raise ValueError("A parent source requires a remediation reason")
    missing_changed_paths = sorted(set(changed_paths_from_parent) - set(files))
    if missing_changed_paths:
        raise ValueError(
            f"Changed paths are absent from the source snapshot: {missing_changed_paths}"
        )
    payload.update(
        {
            "parent_source_sha256": parent_source_sha256,
            "changed_paths_from_parent": sorted(set(changed_paths_from_parent)),
            "remediation_reason": remediation_reason,
        }
    )
payload["source_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
(root / "WP7_SOURCE_MANIFEST.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "base_commit": base_commit,
    "file_count": payload["file_count"],
    "overlay_count": len(payload["overlay_paths"]),
    "source_sha256": payload["source_sha256"],
}, sort_keys=True))
PY

kubectl_clean exec "$POD" -n "$NAMESPACE" -- \
  bash -lc "chmod -R a-w '$REMOTE_ROOT' && \
   test -s '$REMOTE_ROOT/WP7_SOURCE_MANIFEST.json' && \
   test -s '$REMOTE_ROOT/deploy/run_decision_admissibility_wp7_shadow_devpod.sh' && \
   test -s '$REMOTE_ROOT/deploy/run_decision_admissibility_wp7_canary_devpod.sh' && \
   test -s '$REMOTE_ROOT/deploy/build_decision_admissibility_wp7_same_domain_bundle_devpod.sh' && \
   test -z \"\$(find '$REMOTE_ROOT' -type d -name __pycache__ -print -quit)\" && \
   test -z \"\$(find '$REMOTE_ROOT' -perm /222 -print -quit)\""
