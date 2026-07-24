#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC="${WP8_TIER2_SOURCE_ROOT:-/opt/nautilus}"
OUTPUT="${WP8_TIER2_OUTPUT_ROOT:-/output}"
EXPECTED_SOURCE_SHA256="${WP8_TIER2_EXPECTED_SOURCE_SHA256:?WP8_TIER2_EXPECTED_SOURCE_SHA256 is required}"
EXPECTED_HEAD="${WP8_TIER2_EXPECTED_HEAD:-b47dab63b7861f3ea0871094d6dd07b77e6b81a4}"
EXPECTED_CURRENT_SHA256="${WP8_TIER2_EXPECTED_CURRENT_SHA256:-4a303d88b6d81fbe5d01c9666f4751cb69a229baa248fc7d667816287f619f48}"
EXPECTED_BUNDLE_ID="${WP8_TIER2_EXPECTED_BUNDLE_ID:-mlevolve-be034ec-image-aerial-task-heldout-certified-replay-method-only-v7}"
EXPECTED_BUNDLE_MANIFEST_SHA256="${WP8_TIER2_EXPECTED_BUNDLE_MANIFEST_SHA256:-782bbae91c19d5a2e92aa8d7aca9bf6949824ca50505121373a784e6018881f3}"
EXPECTED_BUNDLE_MANIFEST_FILE_SHA256="${WP8_TIER2_EXPECTED_BUNDLE_MANIFEST_FILE_SHA256:-d92ac6f8bb833a8f714c708f84808eae6ef825dd2911e833b925be2eebd85cce}"
EXPECTED_TRAIN_MANIFEST_SHA256="${WP8_TIER2_EXPECTED_TRAIN_MANIFEST_SHA256:-a7968c2bd3ee254f6e70d6b360bc9362aafc4449b5985078d508ecb3b8de64ea}"
TASK="aerial-cactus-identification"
SEED=314159
INITIAL_DRAFTS=3
STEPS=6
CPU_COUNT=1
CANDIDATE_EXECUTION_CONTRACT_ID="wp8-tier2-canary-paired-feasibility-v1"
CANDIDATE_MAX_EXECUTION_SECONDS=600
CANDIDATE_MAX_EPOCHS=8
CANDIDATE_MAX_CV_FOLDS=1
CANDIDATE_MAX_TRAINABLE_MODELS=1
CANDIDATE_ALLOWED_IMPORT_ROOTS="numpy,pandas,sklearn,torch,torchvision,cv2,PIL,xgboost,lightgbm"
EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256="96dcbf44b2ae5ff706620c14807f9d5f3063a07324af70d7b876230cc1b48ee3"
export CANDIDATE_EXECUTION_CONTRACT_ID CANDIDATE_MAX_EXECUTION_SECONDS
export CANDIDATE_MAX_EPOCHS CANDIDATE_MAX_CV_FOLDS
export CANDIDATE_MAX_TRAINABLE_MODELS CANDIDATE_ALLOWED_IMPORT_ROOTS
export EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256

test -d "$SRC"
test -d /task/input
test -f /task/fixed_holdout_manifest.json
test -f /memory/CURRENT.json
test -s /secrets/mlevolve.env
test -z "$(find "$OUTPUT" -mindepth 1 -print -quit)"
test ! -e /fixed/evaluator_view
test ! -e /task/evaluator_view
test -z "$(find /task -name labels.csv -print -quit)"
test -z "$(find /memory -name labels.csv -print -quit)"

exec > >(tee -a "$OUTPUT/training_launcher.log") 2>&1
printf 'running\n' > "$OUTPUT/STATE"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT/TRAINING_STARTED_AT"

finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$OUTPUT/TRAINING_LAUNCHER_EXIT_CODE"
  if [[ "$rc" -ne 0 ]]; then
    printf 'training_failed\n' > "$OUTPUT/STATE"
  fi
  exit "$rc"
}
term() {
  trap - TERM
  printf 'TERM\n' > "$OUTPUT/TRAINING_SIGNAL"
  exit 143
}
trap finish EXIT
trap term TERM

export PYTHONPATH="$SRC:$SRC/mlevolve"
export MLEVOLVE_CONFIG="$SRC/mlevolve/config/config_authority_canary_enforce.yaml"
export MLEVOLVE_CODE_REVISION="$EXPECTED_HEAD"
export MLEVOLVE_CODE_WORKTREE_SHA256="$EXPECTED_SOURCE_SHA256"
export HF_HOME=/cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME=/cache/torch
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" /work/runtime

verify_source_snapshot() {
  local destination="$1"
  python - "$SRC" "$EXPECTED_HEAD" "$EXPECTED_SOURCE_SHA256" "$destination" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected_head, expected_source, output = sys.argv[2], sys.argv[3], Path(sys.argv[4])
manifest = json.loads((root / "WP8_TIER2_SOURCE_MANIFEST.json").read_text())
files = {}
writable = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "WP8_TIER2_SOURCE_MANIFEST.json":
        continue
    relative = str(path.relative_to(root))
    files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_mode & 0o222:
        writable.append(relative)
unsigned = {key: value for key, value in manifest.items() if key != "source_sha256"}
actual_source = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
report = {
    "schema": "decision_admissibility_wp8_tier2_source_verification_v1",
    "base_commit": manifest.get("base_commit"),
    "source_sha256": manifest.get("source_sha256"),
    "actual_source_sha256": actual_source,
    "file_inventory_matches": files == manifest.get("file_hashes"),
    "writable_paths": writable,
    "verified": bool(
        manifest.get("base_commit") == expected_head
        and manifest.get("source_sha256") == expected_source == actual_source
        and files == manifest.get("file_hashes")
        and not writable
    ),
}
output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
assert report["verified"], report
PY
}

verify_source_snapshot "$OUTPUT/SOURCE_PREFLIGHT.json"

python - "$OUTPUT" "$EXPECTED_CURRENT_SHA256" "$EXPECTED_BUNDLE_ID" \
  "$EXPECTED_BUNDLE_MANIFEST_SHA256" "$EXPECTED_BUNDLE_MANIFEST_FILE_SHA256" \
  "$EXPECTED_TRAIN_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from authority.memory_snapshot import ImmutableBaseBundle
from fixed_holdout.validation import validate_train_view

output = Path(sys.argv[1])
expected_current, expected_id, expected_bundle, expected_bundle_file, expected_train = sys.argv[2:]
current_path = Path("/memory/CURRENT.json")
assert hashlib.sha256(current_path.read_bytes()).hexdigest() == expected_current
pointer = json.loads(current_path.read_text())
bundle_path = Path("/memory") / pointer["bundle_path"]
manifest_path = bundle_path / "manifest.json"
bundle = ImmutableBaseBundle.load(bundle_path, verify_artifacts=True)
assert bundle.bundle_id == expected_id
assert pointer["bundle_id"] == expected_id
assert pointer["manifest_sha256"] == expected_bundle
assert bundle.manifest_sha256 == expected_bundle
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected_bundle_file
train_manifest_path = Path("/task/fixed_holdout_manifest.json")
assert hashlib.sha256(train_manifest_path.read_bytes()).hexdigest() == expected_train
train_manifest = validate_train_view(train_manifest_path, Path("/task/input"))
assert train_manifest["task_id"] == "aerial-cactus-identification"
assert train_manifest["metric"] == "binary_roc_auc"
assert train_manifest["selection_policy"] == "terminal_only"
assert train_manifest["hidden_labels_present"] is False

mountinfo = Path("/proc/self/mountinfo").read_text().splitlines()
mount_points = sorted({line.split(" - ", 1)[0].split()[4] for line in mountinfo})
required = {"/opt/nautilus", "/task", "/memory", "/output", "/secrets/mlevolve.env"}
forbidden = {"/workspace", "/fixed/evaluator_view"}
assert required <= set(mount_points), (required, mount_points)
assert not forbidden & set(mount_points), (forbidden, mount_points)
report = {
    "schema": "decision_admissibility_wp8_tier2_training_isolation_v1",
    "training_role": "label_isolated",
    "required_mounts_present": True,
    "whole_workspace_mounted": False,
    "evaluator_view_mounted": False,
    "labels_csv_visible": False,
    "source_read_only": True,
    "train_view_read_only": True,
    "bundle_read_only": True,
    "current_pointer_sha256": expected_current,
    "bundle_id": bundle.bundle_id,
    "bundle_manifest_sha256": bundle.manifest_sha256,
    "bundle_manifest_file_sha256": expected_bundle_file,
    "train_manifest_sha256": expected_train,
    "split_id": train_manifest["split_id"],
    "metric": train_manifest["metric"],
    "mount_points": mount_points,
}
(output / "TRAINING_ISOLATION.json").write_text(
    json.dumps(report, sort_keys=True, indent=2) + "\n"
)
PY

cd "$SRC"
python -m pytest -q -p no:cacheprovider \
  tests/authority/test_candidate_execution_contract.py \
  tests/authority/test_domain_transfer_scope.py \
  tests/authority/test_tier2_canary_launcher_static.py \
  tests/test_stage_aware_hybrid_memory.py \
  -k 'candidate_execution_contract or cross_task_provisional_method or no_memory_binds_bundle' \
  | tee "$OUTPUT/TIER2_TARGETED_PREFLIGHT.log"

python - "$OUTPUT" "$CANDIDATE_ALLOWED_IMPORT_ROOTS" \
  "$EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256" <<'PY'
import importlib
import json
import os
import sys
from pathlib import Path

from engine.candidate_execution_contract import build_candidate_execution_contract

output, roots, expected_contract_hash = (
    Path(sys.argv[1]),
    sys.argv[2].split(","),
    sys.argv[3],
)
imports = {}
for root in roots:
    try:
        module = importlib.import_module(root)
    except Exception as error:
        imports[root] = {
            "importable": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    else:
        imports[root] = {
            "importable": True,
            "version": str(getattr(module, "__version__", "unknown")),
        }
assert all(row["importable"] for row in imports.values()), imports
report = {
    "schema": "decision_admissibility_wp8_tier2_candidate_environment_v1",
    "allowed_import_roots": roots,
    "imports": imports,
    "all_allowed_import_roots_importable": True,
}
(output / "CANDIDATE_EXECUTION_ENVIRONMENT.json").write_text(
    json.dumps(report, sort_keys=True, indent=2) + "\n"
)
contract = build_candidate_execution_contract(
    contract_id=os.environ["CANDIDATE_EXECUTION_CONTRACT_ID"],
    max_execution_seconds=int(os.environ["CANDIDATE_MAX_EXECUTION_SECONDS"]),
    max_epochs=int(os.environ["CANDIDATE_MAX_EPOCHS"]),
    max_cv_folds=int(os.environ["CANDIDATE_MAX_CV_FOLDS"]),
    max_trainable_models=int(os.environ["CANDIDATE_MAX_TRAINABLE_MODELS"]),
    allowed_import_roots=roots,
    allow_remote_assets=False,
    allow_unverified_local_assets=False,
    allow_dataset_wide_per_sample_precompute=False,
    allow_source_score_inheritance=False,
)
assert contract["contract_hash"] == expected_contract_hash, contract
(output / "CANDIDATE_EXECUTION_CONTRACT.json").write_text(
    json.dumps(contract, sort_keys=True, indent=2) + "\n"
)
PY

nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader \
  | tee "$OUTPUT/GPU_IDENTITY.txt"

set -a
. /secrets/mlevolve.env
set +a
export DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL
test -n "${DEEPSEEK_API_KEY:-}"
test -n "${DEEPSEEK_BASE_URL:-}"
test -n "${DEEPSEEK_MODEL:-}"
python - "$OUTPUT" <<'PY'
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

base = os.environ["DEEPSEEK_BASE_URL"]
host = (urlparse(base).hostname or "").lower()
assert host not in {"", "localhost", "127.0.0.1"}
payload = {
    "schema": "decision_admissibility_wp8_tier2_provider_attestation_v1",
    "provider": "deepseek",
    "model": os.environ["DEEPSEEK_MODEL"],
    "base_url_sha256": hashlib.sha256(base.encode()).hexdigest(),
    "api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
    "non_local_endpoint": True,
    "real_provider_required": True,
}
(Path(os.sys.argv[1]) / "PROVIDER_ATTESTATION.json").write_text(
    json.dumps(payload, sort_keys=True, indent=2) + "\n"
)
PY

run_condition() {
  local condition="$1"
  local retrieval_control="$2"
  local memory_enabled="$3"
  local memory_system="$4"
  local root="$OUTPUT/$condition"
  test ! -e "$root"
  mkdir -p "$root/runs" "$root/workspace"
  printf 'running\n' > "$root/STATE"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$root/STARTED_AT"
  cd "$SRC/mlevolve"
  set +e
  timeout --foreground --signal=TERM --kill-after=30s 4800s \
    python -u run.py \
      exp_id="$TASK" \
      exp_name="wp8-tier2-canary-$condition" \
      dataset_dir=/task/input \
      data_dir=/task/input \
      desc_file=/task/input/description.md \
      log_dir="$root/runs" \
      workspace_dir="$root/workspace" \
      fixed_holdout.enabled=true \
      fixed_holdout.evaluation_mode=terminal_only \
      fixed_holdout.train_manifest_path=/task/fixed_holdout_manifest.json \
      fixed_holdout.bypass_protocol_gates=true \
      fixed_holdout.internal_metric_disposition=search_only \
      coldstart.use_coldstart=false \
      agent.check_data_leakage=false \
      agent.protocol_repair.enabled=false \
      evaluation_authority.mode=enforce \
      evaluation_authority.active_protocol_id=mlevolve-default \
      evaluation_authority.active_protocol_version=2 \
      evaluation_authority.rollout_id="wp8-tier2-canary-$condition-r10" \
      evaluation_authority.collector_version=2 \
      evaluation_authority.require_bound_bundle=true \
      evaluation_authority.expected_bundle_id="$EXPECTED_BUNDLE_ID" \
      evaluation_authority.expected_bundle_manifest_sha256="$EXPECTED_BUNDLE_MANIFEST_SHA256" \
      evaluation_authority.enforce_operations='[generate_candidate,debug_hypothesis,promote_result,publish_adoption,publish_causal,code_seed,distill_diagnostic,distill_candidate,distill_positive_result,distill_positive_adopted,derived_publication]' \
      external_skill_memory.bundle_root=/memory \
      external_skill_memory.current_pointer_path=CURRENT.json \
      external_skill_memory.retrieval_control="$retrieval_control" \
      agent.draft_role_policy.roles='[coldstart_baseline,memory_transfer,novel_exploration]' \
      agent.candidate_execution_contract.enabled=true \
      agent.candidate_execution_contract.contract_id="$CANDIDATE_EXECUTION_CONTRACT_ID" \
      agent.candidate_execution_contract.max_execution_seconds="$CANDIDATE_MAX_EXECUTION_SECONDS" \
      agent.candidate_execution_contract.max_epochs="$CANDIDATE_MAX_EPOCHS" \
      agent.candidate_execution_contract.max_cv_folds="$CANDIDATE_MAX_CV_FOLDS" \
      agent.candidate_execution_contract.max_trainable_models="$CANDIDATE_MAX_TRAINABLE_MODELS" \
      agent.candidate_execution_contract.allowed_import_roots='[numpy,pandas,sklearn,torch,torchvision,cv2,PIL,xgboost,lightgbm]' \
      agent.candidate_execution_contract.allow_remote_assets=false \
      agent.candidate_execution_contract.allow_unverified_local_assets=false \
      agent.candidate_execution_contract.allow_dataset_wide_per_sample_precompute=false \
      agent.candidate_execution_contract.allow_source_score_inheritance=false \
      run_identity.experiment_group="wp8_tier2_canary_${condition}_r10" \
      run_identity.baseline_reference_group=wp8_tier2_canary_nm_r10 \
      run_identity.memory_enabled="$memory_enabled" \
      run_identity.memory_system="$memory_system" \
      run_identity.memory_version=method_only_task_heldout_v7 \
      agent.initial_drafts="$INITIAL_DRAFTS" \
      agent.steps="$STEPS" \
      agent.time_limit=4200 \
      agent.seed="$SEED" \
      agent.search.num_gpus=1 \
      agent.search.parallel_search_num=1 \
      cpu_number="$CPU_COUNT" \
      exec.timeout="$CANDIDATE_MAX_EXECUTION_SECONDS" \
      2>&1 | tee "$root/run_stdout.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$root/EXIT_CODE"
  date -u +'%Y-%m-%dT%H:%M:%SZ' > "$root/FINISHED_AT"
  if [[ "$rc" -ne 0 ]]; then
    printf 'failed\n' > "$root/STATE"
    return "$rc"
  fi
  printf 'training_complete_unscored\n' > "$root/STATE"
}

# Fixed order is acceptable only for this infrastructure canary. Formal Tier-2
# must counterbalance condition order and use at least three agent seeds.
run_condition nm no_memory false no_memory
run_condition full stage_hybrid true full_decision_admissibility

python - "$OUTPUT" "$EXPECTED_BUNDLE_ID" "$EXPECTED_BUNDLE_MANIFEST_SHA256" \
  "$SEED" "$STEPS" "$INITIAL_DRAFTS" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from authority.domain_scope import audit_same_domain_task_heldout_exposures
from authority.ledger import AuthorityLedger
from authority.memory_snapshot import ImmutableBaseBundle
from engine.candidate_execution_contract import (
    build_candidate_execution_contract,
    valid_candidate_execution_audit,
    valid_candidate_execution_block_receipt,
)

root = Path(sys.argv[1]).resolve()
expected_bundle_id, expected_bundle_sha = sys.argv[2], sys.argv[3]
seed, steps, initial_drafts = map(int, sys.argv[4:7])
assert initial_drafts == 3
assert steps > initial_drafts
execution_contract = build_candidate_execution_contract(
    contract_id=os.environ["CANDIDATE_EXECUTION_CONTRACT_ID"],
    max_execution_seconds=int(os.environ["CANDIDATE_MAX_EXECUTION_SECONDS"]),
    max_epochs=int(os.environ["CANDIDATE_MAX_EPOCHS"]),
    max_cv_folds=int(os.environ["CANDIDATE_MAX_CV_FOLDS"]),
    max_trainable_models=int(os.environ["CANDIDATE_MAX_TRAINABLE_MODELS"]),
    allowed_import_roots=os.environ["CANDIDATE_ALLOWED_IMPORT_ROOTS"].split(","),
    allow_remote_assets=False,
    allow_unverified_local_assets=False,
    allow_dataset_wide_per_sample_precompute=False,
    allow_source_score_inheritance=False,
)
assert execution_contract["contract_hash"] == os.environ[
    "EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256"
]
assert execution_contract == json.loads(
    (root / "CANDIDATE_EXECUTION_CONTRACT.json").read_text()
)
pointer = json.loads(Path("/memory/CURRENT.json").read_text())
bundle = ImmutableBaseBundle.load(Path("/memory") / pointer["bundle_path"], verify_artifacts=True)
assert bundle.bundle_id == expected_bundle_id
assert bundle.manifest_sha256 == expected_bundle_sha
clauses = bundle.read_jsonl("sop/clauses.jsonl")
eligible = [
    row for row in clauses
    if row.get("publication_class") == "certified"
    and set(row.get("claim_types") or []) == {"method_hypothesis"}
    and "image" in set(row.get("source_domains") or [])
    and row.get("transfer_scope") == "same_domain"
]
assert len(eligible) == 1, len(eligible)
certified = eligible[0]

conditions = {}
for name, control in (("nm", "no_memory"), ("full", "stage_hybrid")):
    condition_root = root / name
    run_dirs = sorted(
        path for path in (condition_root / "runs").iterdir() if path.is_dir()
    )
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]
    request_path = run_dir / "fixed_holdout_evaluation_request.json"
    journal_path = run_dir / "journal.json"
    ledger_path = run_dir / "authority_events.jsonl"
    rollout_path = run_dir / "authority_rollout_report.json"
    for path in (request_path, journal_path, ledger_path, rollout_path):
        assert path.is_file(), path
    request = json.loads(request_path.read_text())
    journal = json.loads(journal_path.read_text())
    candidate_nodes = [
        node
        for node in journal.get("nodes") or []
        if isinstance(node, dict) and node.get("stage") != "root"
    ]
    assert len(candidate_nodes) == steps, (name, len(candidate_nodes), steps)
    assert all(
        (node.get("role_contract") or {}).get("candidate_execution_contract")
        == execution_contract
        for node in candidate_nodes
    ), name
    candidate_by_id = {str(node["id"]): node for node in candidate_nodes}
    audit_paths = sorted(
        (condition_root / "workspace").glob(
            "*/working/candidate_execution_contract_audit_*.json"
        )
    )
    assert len(audit_paths) == len(candidate_nodes), (name, audit_paths)
    block_paths = sorted(
        (condition_root / "workspace").glob(
            "*/working/candidate_execution_block_receipt_*.json"
        )
    )
    block_by_node = {
        path.stem.removeprefix("candidate_execution_block_receipt_"): path
        for path in block_paths
    }
    audit_receipts = []
    admitted_node_ids = set()
    denied_node_ids = set()
    for audit_path_item in audit_paths:
        audit = json.loads(audit_path_item.read_text())
        node_id = audit_path_item.stem.removeprefix(
            "candidate_execution_contract_audit_"
        )
        assert node_id in candidate_by_id, (name, node_id)
        assert valid_candidate_execution_audit(audit), (name, audit_path_item)
        assert audit.get("contract_hash") == execution_contract["contract_hash"]
        assert audit.get("code_sha256") == hashlib.sha256(
            str(candidate_by_id[node_id].get("code") or "").encode()
        ).hexdigest()
        if audit.get("valid") is True:
            admitted_node_ids.add(node_id)
            assert node_id not in block_by_node, (name, node_id)
        else:
            denied_node_ids.add(node_id)
            node = candidate_by_id[node_id]
            block_path = block_by_node.get(node_id)
            assert block_path is not None and block_path.is_file(), (name, node_id)
            block = json.loads(block_path.read_text())
            assert valid_candidate_execution_block_receipt(block), (name, block)
            assert block.get("node_id") == node_id
            assert block.get("contract_hash") == execution_contract["contract_hash"]
            assert block.get("audit_hash") == audit.get("audit_hash")
            assert block.get("code_sha256") == audit.get("code_sha256")
            assert node.get("exc_type") == "CandidateExecutionContractError"
            assert node.get("is_buggy") is True
            assert (node.get("metric") or {}).get("value") is None
            assert float(node.get("exec_time") or 0.0) < 30.0
            exc_info = node.get("exc_info") or {}
            assert Path(exc_info.get("block_receipt_path", "")) == block_path
            assert exc_info.get("block_receipt_hash") == block.get("receipt_hash")
        audit_receipts.append(audit)
    assert admitted_node_ids | denied_node_ids == set(candidate_by_id)
    assert set(block_by_node) == denied_node_ids
    submission_dir = Path(request["submission_dir"])
    assert submission_dir.is_dir() and str(submission_dir).startswith(str(root))
    submissions = sorted(submission_dir.glob("submission_*.csv"))
    assert submissions, (name, submission_dir)
    submitted_node_ids = {
        path.stem.removeprefix("submission_") for path in submissions
    }
    assert submitted_node_ids <= admitted_node_ids, (
        name,
        submitted_node_ids,
        admitted_node_ids,
    )
    assert not submitted_node_ids & denied_node_ids
    runtime_failed_admitted_node_ids = {
        node_id
        for node_id in admitted_node_ids
        if candidate_by_id[node_id].get("is_buggy") is True
    }
    assert not list(run_dir.glob("fixed_holdout_scores*.json"))
    ledger = AuthorityLedger(ledger_path)
    assert ledger.verify()
    events = ledger.read()
    exposures = [
        event["payload"]
        for event in events
        if event.get("event_type") == "experience_exposed"
    ]
    if name == "nm":
        assert exposures == []
        exposure_audit = {
            "schema": "decision_admissibility_wp8_tier2_no_memory_exposure_audit_v1",
            "valid": True,
            "exposure_event_count": 0,
            "invalid_exposure_count": 0,
            "certified_method_exposure_count": 0,
        }
    else:
        exposure_audit = audit_same_domain_task_heldout_exposures(
            exposures,
            clauses,
            target_task_id="aerial-cactus-identification",
            target_domain="image",
            certified_clause_id=certified["clause_id"],
            certified_source_task_id=(certified.get("source_task_ids") or [""])[0],
        )
        assert exposure_audit["valid"] is True, exposure_audit
        assert exposure_audit["invalid_exposure_count"] == 0
        assert exposure_audit["certified_method_exposure_count"] > 0
    audit_path = condition_root / "EXPOSURE_AUDIT.json"
    audit_path.write_text(json.dumps(exposure_audit, sort_keys=True, indent=2) + "\n")
    file_hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            request_path,
            journal_path,
            ledger_path,
            rollout_path,
            audit_path,
            *audit_paths,
            *block_paths,
            *submissions,
        ]
    }
    conditions[name] = {
        "condition": name,
        "retrieval_control": control,
        "run_dir": str(run_dir),
        "journal_path": str(journal_path),
        "evaluation_request_path": str(request_path),
        "submission_dir": str(submission_dir),
        "submission_count": len(submissions),
        "experience_exposure_count": len(exposures),
        "exposure_audit_path": str(audit_path),
        "pre_evaluator_score_file_count": 0,
        "candidate_execution_contract_hash": execution_contract["contract_hash"],
        "candidate_execution_contract_role_binding_valid": True,
        "candidate_execution_audit_paths": [str(path) for path in audit_paths],
        "candidate_execution_audit_count": len(audit_receipts),
        "candidate_execution_audits_integrity_valid": True,
        "candidate_execution_block_receipt_paths": [
            str(path) for path in block_paths
        ],
        "candidate_execution_block_receipt_count": len(block_paths),
        "candidate_execution_admitted_count": len(admitted_node_ids),
        "candidate_execution_denied_count": len(denied_node_ids),
        "candidate_execution_denials_enforced": True,
        "candidate_execution_runtime_failed_admitted_count": len(
            runtime_failed_admitted_node_ids
        ),
        "candidate_execution_submitted_admitted_count": len(
            submitted_node_ids
        ),
        "candidate_execution_admitted_node_ids": sorted(admitted_node_ids),
        "candidate_execution_denied_node_ids": sorted(denied_node_ids),
        "candidate_execution_submitted_node_ids": sorted(submitted_node_ids),
        "max_candidate_execution_seconds_observed": max(
            float(node.get("exec_time") or 0.0) for node in candidate_nodes
        ),
        "file_hashes": file_hashes,
    }
    for path in [request_path, journal_path, *submissions]:
        path.chmod(path.stat().st_mode & ~0o222)

manifest = {
    "schema": "decision_admissibility_wp8_tier2_training_manifest_v1",
    "status": "training_complete_unscored",
    "task_id": "aerial-cactus-identification",
    "protocol_ref": "mlevolve-default@2#4e54d9e6e3c44af8d92f578ef25b4be489b602e62ccc2ac88fa2113768f7eff2",
    "fixed_holdout_metric": "binary_roc_auc",
    "conditions": conditions,
    "condition_order": ["nm", "full"],
    "same_host_seed": seed,
    "steps_per_condition": steps,
    "initial_drafts_per_condition": initial_drafts,
    "repair_steps_budget_per_condition": steps - initial_drafts,
    "same_source_snapshot": True,
    "same_bundle_binding": True,
    "candidate_execution_contract": execution_contract,
    "same_candidate_execution_contract": True,
    "candidate_execution_contract_host_enforced": True,
    "legacy_static_coldstart_enabled": False,
    "condition_difference_limited_to_external_memory_retrieval": True,
    "terminal_scores_visible_during_search": False,
    "effect_claim_authorized": False,
    "formal_tier2_evidence": False,
    "formal_tier2_requirements_remaining": [
        "counterbalanced_condition_order",
        "at_least_three_agent_seeds",
        "at_least_three_protocol_families",
        "full_baseline_matrix",
    ],
    "manifest_hash": "",
}
unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
manifest["manifest_hash"] = hashlib.sha256(
    json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
(root / "TRAINING_MANIFEST.json").write_text(
    json.dumps(manifest, sort_keys=True, indent=2) + "\n"
)
bundle.assert_unchanged()
PY

# Recompute the immutable source inventory after all model calls and executions;
# copying the preflight report would not prove post-run integrity.
verify_source_snapshot "$OUTPUT/SOURCE_POSTRUN.json"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUTPUT/TRAINING_FINISHED_AT"
printf 'training_complete_unscored\n' > "$OUTPUT/STATE"
touch "$OUTPUT/TRAINING_COMPLETE"
printf '0\n' > "$OUTPUT/TRAINING_LAUNCHER_EXIT_CODE"
trap - EXIT
