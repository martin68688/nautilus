#!/usr/bin/env bash
set -euo pipefail

# Build an isolated r3 release without modifying the frozen r2 tree.  This
# script deliberately stops before the final freeze: the exact-source A40
# enforce smoke must be copied into freeze/ first.
release_tag="${RELEASE_TAG:-r3}"
source_tree="${SOURCE_TREE:-/tmp/prevalence-source-${release_tag}}"
previous_root="${PREVIOUS_RELEASE_ROOT:-/workspace/prevalence-audit-20260729-r2}"
release_root="${RELEASE_ROOT:-/workspace/prevalence-audit-20260729-${release_tag}}"
source_staging="${SOURCE_STAGING:-/tmp/prevalence-audit-source-${release_tag}}"
host_source_root="${HOST_SOURCE_ROOT:-/tmp/prevalence-host-source-${release_tag}}"
collector_key_source="${COLLECTOR_PRIVATE_KEY_SOURCE:-/run/host-key-source/collector.ed25519}"
formal_jobs_source="${FORMAL_JOBS_SOURCE:-${source_tree}/deploy/prevalence-audit-20260729-five-a100.yaml}"
seed_matrix_source="${SEED_MATRIX_SOURCE:-${source_tree}/experiments/prevalence_audit_20260729/seed_matrix.json}"

test -d "${source_tree}/mlevolve"
test -s "${source_tree}/deploy/package_prevalence_audit_source_20260729.sh"
test -s "${source_tree}/deploy/build_prevalence_host_protocol_bundles_20260729.sh"
test -s "${formal_jobs_source}"
test -s "${seed_matrix_source}"
test -s "${previous_root}/freeze/FREEZE_MANIFEST.json"
test -s "${collector_key_source}"
test ! -e "${release_root}"
test ! -e "${source_staging}"
test ! -e "${host_source_root}"

mkdir -p "${release_root}/source" "${release_root}/freeze" "${release_root}/outputs"
cp -al "${previous_root}/data" "${release_root}/data"
cp -a "${previous_root}/source/dependencies" "${release_root}/source/dependencies"

bash "${source_tree}/deploy/package_prevalence_audit_source_20260729.sh" \
  "${source_tree}" "${source_staging}"
mv "${source_staging}/mlevolve-runtime.tar.gz" "${release_root}/source/"
mv "${source_staging}/mlevolve-runtime.tar.gz.sha256" "${release_root}/source/"
rmdir "${source_staging}"

RELEASE_ROOT="${release_root}" \
RELEASE_TAG="${release_tag}" \
HOST_SOURCE_ROOT="${host_source_root}" \
COLLECTOR_PRIVATE_KEY_SOURCE="${collector_key_source}" \
  bash "${source_tree}/deploy/build_prevalence_host_protocol_bundles_20260729.sh"

source_sha="$(awk 'NF {print $1; exit}' "${release_root}/source/mlevolve-runtime.tar.gz.sha256")"
PYTHONPATH="${source_tree}/mlevolve" python \
  "${source_tree}/experiments/prevalence_audit_20260729/build_memory_profiles.py" \
  --source-archive-sha256 "${source_sha}" \
  --release-tag "${release_tag}" \
  --graph "${previous_root}/memory/natural/run_forest_graph.json" \
  --index "${previous_root}/memory/natural/run_forest_index.npz" \
  --replay-targets "${previous_root}/memory/natural/replay_targets.json" \
  --runs-root "${previous_root}/memory/natural/replay_sources" \
  --output-root "${release_root}/memory"

cp "${seed_matrix_source}" \
  "${release_root}/freeze/seed_matrix.json"
cp "${source_tree}/experiments/prevalence_audit_20260729/freeze_spec.template.json" \
  "${release_root}/freeze/FREEZE_SPEC.template.json"
cp "${formal_jobs_source}" \
  "${release_root}/freeze/formal_jobs.yaml"

python - "${release_root}" "${source_sha}" "${previous_root}" "${source_tree}" "${collector_key_source}" "${release_tag}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source_sha = sys.argv[2]
previous = pathlib.Path(sys.argv[3])
release_tag = sys.argv[6]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

tasks = [
    "denoising-dirty-documents",
    "leaf-classification",
    "aerial-cactus-identification",
    "new-york-city-taxi-fare-prediction",
    "spooky-author-identification",
]
contracts = {}
for task in tasks:
    path = root / "bindings" / task / "contract" / "PROTOCOL_EXECUTION_CONTRACT.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    metric = payload["evaluator_spec"]["metric"]
    contracts[task] = {
        "contract_hash": payload["contract_hash"],
        "protocol_ref": payload["protocol_ref"],
        "metric": metric["name"],
        "direction": metric["direction"],
    }
expected = {
    "denoising-dirty-documents": ("rmse", "minimize"),
    "leaf-classification": ("log_loss", "minimize"),
    "aerial-cactus-identification": ("roc_auc", "maximize"),
    "new-york-city-taxi-fare-prediction": ("rmse", "minimize"),
    "spooky-author-identification": ("log_loss", "minimize"),
}
for task, pair in expected.items():
    assert (contracts[task]["metric"], contracts[task]["direction"]) == pair
evaluator_identity = hashlib.sha256(
    json.dumps(contracts, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
evaluator = {
    "schema": "mlevolve_frozen_evaluator_binding_v1",
    "id": f"frozen-terminal-evaluators-five-task-{release_tag}",
    "revision": f"sha256:{evaluator_identity}",
    "terminal_source": "host_terminal",
    "best_seed_selection": False,
    "contracts": contracts,
}
(root / "freeze" / "evaluator_binding.json").write_text(
    json.dumps(evaluator, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

model = json.loads((previous / "freeze" / "model_binding.json").read_text(encoding="utf-8"))
(root / "freeze" / "model_binding.json").write_text(
    json.dumps(model, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

config_paths = [
    "mlevolve/config/config_prevalence_audit_20260729_host_enforce.yaml",
    "mlevolve/config/config_fourtask_graph_v2_all_features_host_shadow.yaml",
    "mlevolve/config/config_authority_host_protocol_shadow.yaml",
    "mlevolve/config/config_authority_shadow.yaml",
    "mlevolve/config/config.yaml",
]
source_tree = pathlib.Path(sys.argv[4])
collector_key_source = pathlib.Path(sys.argv[5])
config_chain = [
    {"path": value, "sha256": sha256(source_tree / value)} for value in config_paths
]
config_revision = hashlib.sha256(
    json.dumps(config_chain, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
config = {
    "schema": "mlevolve_frozen_runtime_config_binding_v1",
    "id": f"prevalence-audit-host-enforce-{release_tag}",
    "revision": f"sha256:{config_revision}",
    "source_archive_sha256": source_sha,
    "effective_mode": "host_sdk_enforce",
    "methodology_enabled": False,
    "methodology_kb_path": "",
    "prospective_audit_enabled": True,
    "allow_pending_counterfactual": False,
    "config_chain": config_chain,
}
(root / "freeze" / "config_binding.json").write_text(
    json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

state = {
    "schema": "mlevolve_prevalence_release_staging_v1",
    "status": "awaiting_exact_source_online_smoke",
    "release_root": str(root),
    "source_archive_sha256": source_sha,
    "collector_private_key_sha256": sha256(root / ".secrets" / "collector.ed25519"),
    "immutable_secret_collector_private_key_sha256": sha256(collector_key_source),
    "methodology_enabled": False,
    "tasks": contracts,
}
assert state["collector_private_key_sha256"] == state["immutable_secret_collector_private_key_sha256"]
(root / "freeze" / "STAGING_STATE.json").write_text(
    json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

find "${release_root}/source" "${release_root}/data" "${release_root}/memory" \
  "${release_root}/bindings" -type f \
  -not -path '*/reports/*' -not -path '*/runtime/*' -exec chmod a-w {} +
rm -rf "${release_root}/.host-staging" "${release_root}/.secrets"

echo "${release_tag} release staged at ${release_root}; final freeze awaits exact-source A40 smoke"
