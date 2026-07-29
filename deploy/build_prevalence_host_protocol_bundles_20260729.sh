#!/usr/bin/env bash
set -euo pipefail

release_tag="${RELEASE_TAG:-r3}"
root="${RELEASE_ROOT:-/workspace/prevalence-audit-20260729-${release_tag}}"
source_archive="${root}/source/mlevolve-runtime.tar.gz"
source_sha_file="${source_archive}.sha256"
image_digest="${MLEVOLVE_IMAGE_DIGEST:-sha256:fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc}"
source_root="${HOST_SOURCE_ROOT:-/tmp/prevalence-host-source-${release_tag}}"
binding_root="${root}/bindings"
staging_root="${root}/.host-staging"
secret_root="${root}/.secrets"

test -s "${source_archive}"
source_sha="$(awk 'NF {print $1; exit}' "${source_sha_file}")"
test "$(sha256sum "${source_archive}" | awk '{print $1}')" = "${source_sha}"
test ! -e "${binding_root}"
test ! -e "${staging_root}"
test ! -e "${secret_root}"
mkdir -p "${binding_root}" "${staging_root}" "${secret_root}" "${source_root}"
tar -xzf "${source_archive}" -C "${source_root}"
if [[ -n "${COLLECTOR_PRIVATE_KEY_SOURCE:-}" ]]; then
  test -s "${COLLECTOR_PRIVATE_KEY_SOURCE}"
  install -m 0400 "${COLLECTOR_PRIVATE_KEY_SOURCE}" \
    "${secret_root}/collector.ed25519"
fi

tasks=(
  denoising-dirty-documents
  leaf-classification
  aerial-cactus-identification
  new-york-city-taxi-fare-prediction
  spooky-author-identification
)

for task in "${tasks[@]}"; do
  public_root="${root}/data/${task}/public"
  description="${public_root}/description.md"
  test -s "${description}"
  PYTHONPATH="${source_root}/mlevolve" \
    python -m protocol_runtime.activation build-task \
      --task-id "${task}" \
      --public-root "${public_root}" \
      --staging-root "${staging_root}/${task}" \
      --output-root "${binding_root}/${task}" \
      --description "${description}" \
      --registry-root "${source_root}/mlevolve/config/protocols" \
      --image-digest "${image_digest}" \
      --sdk-root "${source_root}/mlevolve/protocol_runtime" \
      --collector-private-key-output "${secret_root}/collector.ed25519" \
      --split-id "prevalence-audit-20260729-${release_tag}-${task}" \
      --report-root "${binding_root}/${task}/reports" \
      --runtime-artifact-root "${binding_root}/${task}/runtime" \
      --timeout-seconds 60 \
      --validation-fraction 0.2 \
      --seed 20260729
done

test -s "${secret_root}/collector.ed25519"
chmod 0400 "${secret_root}/collector.ed25519"
find "${binding_root}" -type f -not -path '*/reports/*' -not -path '*/runtime/*' -exec chmod a-w {} +
find "${staging_root}" -type f -exec chmod a-w {} +
printf '%s\n' "${source_sha}" > "${binding_root}/RUNTIME_SOURCE_SHA256"
chmod a-w "${binding_root}/RUNTIME_SOURCE_SHA256"
echo "prevalence ${release_tag} Host bundles built under ${binding_root}"
