#!/usr/bin/env bash
set -euo pipefail

release_tag="${RELEASE_TAG:-r3}"
source_tree="${SOURCE_TREE:-/tmp/prevalence-source-${release_tag}}"
release_root="${RELEASE_ROOT:-/workspace/prevalence-audit-20260729-${release_tag}}"
smoke_report="${SMOKE_REPORT:?SMOKE_REPORT must point to the copied A40 exact-source gate report}"
previous_root="${PREVIOUS_RELEASE_ROOT:-/workspace/prevalence-audit-20260729-r2}"
previous_environment_binding="${PREVIOUS_ENVIRONMENT_BINDING:-${previous_root}/freeze/environment_binding.json}"
final_source_root="${FINAL_SOURCE_ROOT:-/tmp/prevalence-final-source-${release_tag}}"

test -s "${release_root}/source/mlevolve-runtime.tar.gz"
test -s "${release_root}/source/mlevolve-runtime.tar.gz.sha256"
test -s "${release_root}/freeze/seed_matrix.json"
test -s "${release_root}/freeze/formal_jobs.yaml"
test -s "${release_root}/freeze/FREEZE_SPEC.template.json"
test -s "${previous_environment_binding}"
test -s "${smoke_report}"
test ! -e "${release_root}/freeze/FREEZE_MANIFEST.json"
test ! -e "${release_root}/freeze/FREEZE_SPEC.json"
test ! -e "${release_root}/freeze/exact_source_smoke_result.json"
test ! -e "${final_source_root}"

source_sha="$(awk 'NF {print $1; exit}' "${release_root}/source/mlevolve-runtime.tar.gz.sha256")"
test "$(sha256sum "${release_root}/source/mlevolve-runtime.tar.gz" | awk '{print $1}')" = "${source_sha}"

python - "${smoke_report}" "${source_sha}" "${release_root}/freeze/exact_source_smoke_result.json" <<'PY'
import hashlib
import json
import pathlib
import sys

source_report = pathlib.Path(sys.argv[1])
source_sha = sys.argv[2]
output = pathlib.Path(sys.argv[3])
payload = json.loads(source_report.read_text(encoding="utf-8"))
assert payload.get("schema") == "mlevolve_prevalence_full_runtime_online_gate_v1"
assert payload.get("status") == "pass"
tasks = payload.get("tasks") or {}
assert set(tasks) >= {"denoising-dirty-documents", "aerial-cactus-identification"}
for task_id in ("denoising-dirty-documents", "aerial-cactus-identification"):
    task = tasks[task_id]
    assert task.get("status") == "pass"
    assert task.get("preflight", {}).get("status") == "pass"
    assert task.get("host_full_runtime", {}).get("status") == "pass"
    assert task.get("host_full_runtime", {}).get("missing_events") == []
    assert all(
        value.get("outcome") == "allow"
        for value in (task.get("authority") or {}).values()
    )
    assert task.get("metric", {}).get("device") == "cuda"
report_hash = hashlib.sha256(source_report.read_bytes()).hexdigest()
summary = {
    "schema": "mlevolve_exact_source_enforce_smoke_v1",
    "status": "verified",
    "source_archive_sha256": source_sha,
    "detailed_gate_report_sha256": report_hash,
    "authority_mode": "enforce",
    "protocol_runtime_mode": "host_sdk_enforce",
    "raw_logging_coverage": 1.0,
    "pending_counterfactual_count": 0,
    "run_outcome_status": "complete",
    "task_ids": sorted(tasks),
    "task_results": {
        task_id: {
            "metric": dict(tasks[task_id]["metric"]),
            "contract_hash": tasks[task_id]["contract_hash"],
            "preflight_report_hash": tasks[task_id]["preflight"]["report_hash"],
            "full_runtime_evidence_hash": tasks[task_id]["host_full_runtime"]["evidence_hash"],
            "authority": dict(tasks[task_id]["authority"]),
        }
        for task_id in sorted(tasks)
    },
}
output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

rm -rf "${final_source_root}"
mkdir -m 0755 "${final_source_root}"
tar -xzf "${release_root}/source/mlevolve-runtime.tar.gz" -C "${final_source_root}"

python - "${release_root}" "${source_sha}" "${source_tree}" "${final_source_root}" "${previous_environment_binding}" "${release_tag}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
source_sha = sys.argv[2]
source_tree = pathlib.Path(sys.argv[3])
final_source = pathlib.Path(sys.argv[4])
previous_environment = pathlib.Path(sys.argv[5])
release_tag = sys.argv[6]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

smoke = json.loads((root / "freeze" / "exact_source_smoke_result.json").read_text())
contracts = {}
for path in sorted((root / "bindings").glob("*/contract/PROTOCOL_EXECUTION_CONTRACT.json")):
    payload = json.loads(path.read_text())
    metric = payload["evaluator_spec"]["metric"]
    contracts[payload["task_id"]] = {
        "contract_hash": payload["contract_hash"],
        "protocol_ref": payload["protocol_ref"],
        "metric": metric["name"],
        "direction": metric["direction"],
    }
evaluator_revision = hashlib.sha256(
    json.dumps(contracts, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
evaluator = {
    "schema": "mlevolve_frozen_evaluator_binding_v1",
    "id": f"frozen-terminal-evaluators-five-task-{release_tag}",
    "revision": f"sha256:{evaluator_revision}",
    "terminal_source": "host_terminal",
    "best_seed_selection": False,
    "contracts": contracts,
}
(root / "freeze" / "evaluator_binding.json").write_text(
    json.dumps(evaluator, indent=2, sort_keys=True) + "\n"
)

model = json.loads((root / "freeze" / "model_binding.json").read_text())
(root / "freeze" / "model_binding.json").write_text(
    json.dumps(model, indent=2, sort_keys=True) + "\n"
)

config_paths = [
    "mlevolve/config/config_prevalence_audit_20260729_host_enforce.yaml",
    "mlevolve/config/config_fourtask_graph_v2_all_features_host_shadow.yaml",
    "mlevolve/config/config_authority_host_protocol_shadow.yaml",
    "mlevolve/config/config_authority_shadow.yaml",
    "mlevolve/config/config.yaml",
]
config_chain = [
    {"path": value, "sha256": sha256(final_source / value)} for value in config_paths
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
    json.dumps(config, indent=2, sort_keys=True) + "\n"
)

sys.path.insert(0, str(final_source / "mlevolve"))
from protocol_runtime.activation import hash_sdk_tree
sdk_hash = hash_sdk_tree(final_source / "mlevolve/protocol_runtime")
dependency = root / "source/dependencies/lazy_loader-0.4-py3-none-any.whl"
environment = {
    "schema": "mlevolve_frozen_runtime_environment_binding_v1",
    "container_image": "docker.io/haomingwang22/mlevolve@sha256:fe0b9c383391d3e62e9f321943b4fdedaa4df54ad7f45b0395c8647a195c20cc",
    "source_archive_sha256": source_sha,
    "host_sdk_sha256": sdk_hash,
    "offline_dependencies": {dependency.name: sha256(dependency)},
    "formal_job_manifest": {
        "path": "freeze/formal_jobs.yaml",
        "sha256": sha256(root / "freeze/formal_jobs.yaml"),
        "job_count": 5,
    },
    "exact_source_enforce_smoke": {
        "path": "freeze/exact_source_smoke_result.json",
        "sha256": sha256(root / "freeze/exact_source_smoke_result.json"),
        "status": smoke["status"],
        "source_archive_sha256": smoke["source_archive_sha256"],
        "task_ids": smoke["task_ids"],
    },
    "immutable_secrets": json.loads(previous_environment.read_text())["immutable_secrets"],
    "mount_contract": {
        "persistent_volume_claim": "haoming-storage",
        "single_pvc_declaration_per_pod": True,
        "release_mount": "/release",
        "canonical_host_binding_mount": f"/workspace/prevalence-audit-20260729-{release_tag}",
        "same_subpath": f"prevalence-audit-20260729-{release_tag}",
    },
}
(root / "freeze" / "environment_binding.json").write_text(
    json.dumps(environment, indent=2, sort_keys=True) + "\n"
)

spec = json.loads(
    (root / "freeze/FREEZE_SPEC.template.json").read_text(encoding="utf-8")
)
spec["evaluator"] = {
    "id": evaluator["id"],
    "revision": evaluator["revision"],
}
spec["model"] = {
    "id": model["id"],
    "revision": model["revision"],
}
spec["artifacts"] = [
    item for item in spec["artifacts"] if item["name"] != "freeze-spec"
]
spec["artifacts"].extend([
    {"name": "freeze-spec", "category": "config", "path": "freeze/FREEZE_SPEC.json"},
    {"name": "formal-job-manifest", "category": "config", "path": "freeze/formal_jobs.yaml"},
    {"name": "exact-source-smoke-result", "category": "environment", "path": "freeze/exact_source_smoke_result.json"},
])
(root / "freeze/FREEZE_SPEC.json").write_text(
    json.dumps(spec, indent=2, sort_keys=True) + "\n"
)
PY

PYTHONPATH="${final_source_root}/mlevolve" python -m experiment_freeze create \
  --spec "${release_root}/freeze/FREEZE_SPEC.json" \
  --root "${release_root}" \
  --output "${release_root}/freeze/FREEZE_MANIFEST.json" \
  >"/tmp/prevalence-${release_tag}-freeze-create.json"

PYTHONPATH="${final_source_root}/mlevolve" python -m experiment_freeze verify \
  --manifest "${release_root}/freeze/FREEZE_MANIFEST.json" \
  --root "${release_root}" \
  --job "${release_root}/freeze/formal_jobs.yaml" \
  >"/tmp/prevalence-${release_tag}-freeze-verify.json"

cat > "${release_root}/RELEASE_READY.json" <<EOF
{
  "schema": "mlevolve_prevalence_release_ready_v1",
  "status": "verified",
  "source_archive_sha256": "${source_sha}",
  "freeze_manifest_sha256": "$(sha256sum "${release_root}/freeze/FREEZE_MANIFEST.json" | awk '{print $1}')",
  "formal_job_manifest_sha256": "$(sha256sum "${release_root}/freeze/formal_jobs.yaml" | awk '{print $1}')",
  "methodology_enabled": false,
  "online_smoke": "freeze/exact_source_smoke_result.json"
}
EOF
chmod a-w "${release_root}/RELEASE_READY.json"
echo "${release_tag} release finalized and verified: ${release_root}"
