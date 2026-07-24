from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TRAIN_YAML = (
    ROOT
    / "deploy"
    / "devpod-decision-admissibility-wp8-tier2-train-a100x1-r1.yaml"
)
EVALUATOR_YAML = (
    ROOT
    / "deploy"
    / "devpod-decision-admissibility-wp8-tier2-evaluator-cpu-r1.yaml"
)
TRAIN_RUNNER = (
    ROOT / "deploy" / "run_decision_admissibility_wp8_tier2_train_devpod.sh"
)
EVALUATOR_RUNNER = (
    ROOT / "deploy" / "run_decision_admissibility_wp8_tier2_evaluator_devpod.sh"
)
STAGER = ROOT / "deploy" / "stage_decision_admissibility_wp8_tier2_canary.sh"


def _embedded_python_blocks(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    index = 0
    while index < len(lines):
        if "<<'PY'" not in lines[index]:
            index += 1
            continue
        start = index + 1
        end = start
        while end < len(lines) and lines[end] != "PY":
            end += 1
        assert end < len(lines), f"unterminated heredoc in {path}:{start}"
        blocks.append("\n".join(lines[start:end]) + "\n")
        index = end + 1
    return blocks


def _mounts(document: dict) -> dict[str, dict]:
    return {
        row["mountPath"]: row
        for row in document["spec"]["containers"][0]["volumeMounts"]
    }


def _assert_single_workspace_pvc(document: dict) -> None:
    pvc_volumes = [
        row for row in document["spec"]["volumes"]
        if "persistentVolumeClaim" in row
    ]
    assert pvc_volumes == [
        {
            "name": "workspace",
            "persistentVolumeClaim": {"claimName": "haoming-storage"},
        }
    ]
    pvc_mounts = [
        row for row in document["spec"]["containers"][0]["volumeMounts"]
        if row["name"] == "workspace"
    ]
    assert pvc_mounts
    assert all(row.get("subPath") for row in pvc_mounts)


def test_tier2_canary_uses_only_devpods_and_scripts_are_syntactically_valid() -> None:
    for path in (TRAIN_YAML, EVALUATOR_YAML):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["kind"] == "Pod"
        assert document["spec"]["restartPolicy"] == "Never"
        assert "sleep infinity" in document["spec"]["containers"][0]["args"][0]
        assert "job" not in document["metadata"]["name"]
    for path in (TRAIN_RUNNER, EVALUATOR_RUNNER, STAGER):
        subprocess.run(["bash", "-n", str(path)], check=True)
        blocks = _embedded_python_blocks(path)
        assert blocks
        for index, source in enumerate(blocks):
            ast.parse(source, filename=f"{path}:heredoc:{index}")


def test_training_pod_has_label_isolated_subpath_mounts_only() -> None:
    document = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))
    _assert_single_workspace_pvc(document)
    mounts = _mounts(document)
    assert "/workspace" not in mounts
    assert "/fixed/evaluator_view" not in mounts
    assert set(mounts) == {
        "/opt/nautilus",
        "/task",
        "/memory",
        "/output",
        "/secrets/mlevolve.env",
        "/work",
        "/cache",
    }
    assert mounts["/opt/nautilus"]["readOnly"] is True
    assert mounts["/task"]["readOnly"] is True
    assert mounts["/memory"]["readOnly"] is True
    assert mounts["/secrets/mlevolve.env"]["readOnly"] is True
    assert mounts["/task"]["subPath"].endswith("/train_view")
    assert "evaluator_view" not in mounts["/task"]["subPath"]
    assert mounts["/secrets/mlevolve.env"]["subPath"] == "nautilus/mlevolve/.env"
    resources = document["spec"]["containers"][0]["resources"]
    assert resources["requests"]["nvidia.com/a100"] == resources["limits"]["nvidia.com/a100"] == "1"
    assert resources["requests"]["cpu"] == resources["limits"]["cpu"] == "1"
    assert resources["requests"]["memory"] == resources["limits"]["memory"] == "8Gi"
    expressions = document["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    hostname = next(
        row for row in expressions if row["key"] == "kubernetes.io/hostname"
    )
    assert hostname == {
        "key": "kubernetes.io/hostname",
        "operator": "NotIn",
        "values": ["gp-argo.usd.edu"],
    }


def test_evaluator_is_cpu_only_and_cannot_read_solver_secret_or_bundle() -> None:
    document = yaml.safe_load(EVALUATOR_YAML.read_text(encoding="utf-8"))
    _assert_single_workspace_pvc(document)
    mounts = _mounts(document)
    assert set(mounts) == {
        "/opt/nautilus",
        "/fixed/train_view",
        "/fixed/evaluator_view",
        "/output",
        "/work",
    }
    assert all("nvidia.com" not in key for key in document["spec"]["containers"][0]["resources"]["limits"])
    resources = document["spec"]["containers"][0]["resources"]
    assert resources["requests"] == resources["limits"] == {
        "cpu": "1",
        "memory": "2G",
    }
    assert "/memory" not in mounts
    assert "/secrets/mlevolve.env" not in mounts
    assert mounts["/fixed/train_view"]["readOnly"] is True
    assert mounts["/fixed/evaluator_view"]["readOnly"] is True


def test_canary_runners_freeze_search_before_host_terminal_scoring() -> None:
    training = TRAIN_RUNNER.read_text(encoding="utf-8")
    evaluator = EVALUATOR_RUNNER.read_text(encoding="utf-8")
    assert "EXPECTED_BUNDLE_MANIFEST_SHA256" in training
    assert "EXPECTED_BUNDLE_MANIFEST_FILE_SHA256" in training
    assert 'pointer["manifest_sha256"] == expected_bundle' in training
    assert "hashlib.sha256(manifest_path.read_bytes()).hexdigest() == expected_bundle_file" in training
    assert (
        'evaluation_authority.expected_bundle_manifest_sha256="$EXPECTED_BUNDLE_MANIFEST_SHA256"'
        in training
    )
    assert 'evaluation_authority.expected_bundle_manifest_sha256="$EXPECTED_BUNDLE_MANIFEST_FILE_SHA256"' not in training
    assert "run_condition nm no_memory" in training
    assert "run_condition full stage_hybrid" in training
    assert "fixed_holdout.enabled=true" in training
    assert "coldstart.use_coldstart=false" in training
    assert 'CANDIDATE_EXECUTION_CONTRACT_ID="wp8-tier2-canary-paired-feasibility-v1"' in training
    assert "CANDIDATE_MAX_EXECUTION_SECONDS=600" in training
    assert "CANDIDATE_MAX_EPOCHS=8" in training
    assert "CANDIDATE_MAX_CV_FOLDS=1" in training
    assert "CANDIDATE_MAX_TRAINABLE_MODELS=1" in training
    assert "INITIAL_DRAFTS=3" in training
    assert "STEPS=6" in training
    assert "agent.initial_drafts=\"$INITIAL_DRAFTS\"" in training
    assert "agent.time_limit=4200" in training
    assert "timeout --foreground --signal=TERM --kill-after=30s 4800s" in training
    assert '"initial_drafts_per_condition": initial_drafts' in training
    assert '"repair_steps_budget_per_condition": steps - initial_drafts' in training
    assert (
        'EXPECTED_CANDIDATE_EXECUTION_CONTRACT_SHA256="96dcbf44b2ae5ff706620c14807f9d5f3063a07324af70d7b876230cc1b48ee3"'
        in training
    )
    assert "CANDIDATE_EXECUTION_CONTRACT.json" in training
    assert "agent.candidate_execution_contract.enabled=true" in training
    assert "agent.candidate_execution_contract.allow_remote_assets=false" in training
    assert "agent.candidate_execution_contract.allow_unverified_local_assets=false" in training
    assert (
        "agent.candidate_execution_contract.allow_dataset_wide_per_sample_precompute=false"
        in training
    )
    assert "agent.candidate_execution_contract.allow_source_score_inheritance=false" in training
    assert 'exec.timeout="$CANDIDATE_MAX_EXECUTION_SECONDS"' in training
    assert '"same_candidate_execution_contract": True' in training
    assert '"candidate_execution_contract_host_enforced": True' in training
    assert "valid_candidate_execution_audit" in training
    assert "valid_candidate_execution_block_receipt" in training
    assert 'node.get("exc_type") == "CandidateExecutionContractError"' in training
    assert '"candidate_execution_audits_integrity_valid": True' in training
    assert '"candidate_execution_denials_enforced": True' in training
    assert 'assert audit.get("valid") is True' not in training
    assert "candidate_execution_submitted_node_ids" in training
    assert '"legacy_static_coldstart_enabled": False' in training
    assert '"condition_difference_limited_to_external_memory_retrieval": True' in training
    assert 'CPU_COUNT=1' in training
    assert 'cpu_number="$CPU_COUNT"' in training
    assert "terminal_scores_visible_during_search" in training
    assert "pre_evaluator_score_file_count" in training
    assert "effect_claim_authorized" in training
    assert 'verify_source_snapshot "$OUTPUT/SOURCE_PREFLIGHT.json"' in training
    assert 'verify_source_snapshot "$OUTPUT/SOURCE_POSTRUN.json"' in training
    assert 'cp "$OUTPUT/SOURCE_PREFLIGHT.json"' not in training
    assert "TRAINING_COMPLETE" in training
    assert "TRAINING_POD_DELETION_ATTESTATION.json" in evaluator
    assert "training_pod_absent_before_evaluation" in evaluator
    assert "--finalize-writeback" in evaluator
    assert '{"finalized", "already_finalized"}' in evaluator
    assert "score_condition nm" in evaluator
    assert "score_condition full" in evaluator
    assert "full_superiority_claim_authorized" in evaluator
    assert "candidate_execution_block_receipt_paths" in evaluator
    assert "candidate_execution_denied_node_ids" in evaluator
    assert evaluator.index("score_condition nm") < evaluator.index(
        "CANARY_EVALUATION_SUMMARY.json"
    )


def test_source_stager_excludes_user_data_and_output_assets() -> None:
    text = STAGER.read_text(encoding="utf-8")
    assert "mlevolve/runs" not in text.split("git archive", 1)[1].split("overlay_list=", 1)[0]
    assert "mlevolve/data" not in text.split("git archive", 1)[1].split("overlay_list=", 1)[0]
    assert "git add" not in text
    assert "git commit" not in text
    assert "git push" not in text
    assert "test ! -e '$REMOTE_ROOT'" in text
    assert "test ! -e '$OUTPUT_ROOT'" in text
    assert "decision-admissibility-wp8-tier2-canary-r10-source" in text
    assert "decision-admissibility-wp8-tier2-canary-r10-output" in text
    assert "tests/authority/test_candidate_execution_contract.py" in text
    assert "tests/authority/test_tier2_canary_launcher_static.py" in text


def test_r10_pods_bind_only_the_new_source_and_output_roots() -> None:
    train = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))
    evaluator = yaml.safe_load(EVALUATOR_YAML.read_text(encoding="utf-8"))

    assert train["metadata"]["name"].endswith("-r10")
    assert evaluator["metadata"]["name"].endswith("-r10")
    for document in (train, evaluator):
        mounts = _mounts(document)
        assert mounts["/opt/nautilus"]["subPath"].endswith("canary-r10-source")
        assert mounts["/output"]["subPath"].endswith("canary-r10-output")
