#!/usr/bin/env bash
set -euo pipefail

# Package the exact local worktree used by the five-Job prevalence release.
# The online packet validator is part of the runtime contract, so both the
# solver tree and the experiment-specific validator/specs must be archived.
repo_root="${1:-/workspace/nautilus}"
output_root="${2:-/workspace/prevalence-audit-20260729-source-staging}"
archive="${output_root}/mlevolve-runtime.tar.gz"

test -d "${repo_root}/mlevolve"
test -s "${repo_root}/experiments/prevalence_audit_20260729/validate_run_packet.py"
test ! -e "${output_root}"
mkdir -p "${output_root}"

COPYFILE_DISABLE=1 tar \
  --no-xattrs \
  --no-acls \
  --exclude='mlevolve/.env' \
  --exclude='mlevolve/data' \
  --exclude='mlevolve/runs' \
  --exclude='._*' \
  --exclude='.DS_Store' \
  --exclude='*/__pycache__' \
  --exclude='*/.pytest_cache' \
  --exclude='*.pyc' \
  -czf "${archive}" \
  -C "${repo_root}" \
  mlevolve \
  experiments/prevalence_audit_20260729

sha256sum "${archive}" > "${archive}.sha256"
test -n "$(tar -tzf "${archive}" | grep -Fx 'mlevolve/config/config_prevalence_audit_20260729_host_enforce.yaml')"
test -n "$(tar -tzf "${archive}" | grep -Fx 'experiments/prevalence_audit_20260729/validate_run_packet.py')"
chmod a-w "${archive}" "${archive}.sha256"
