from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "paper-skills" / "memory_bundle"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_tier2_formal_staging import (  # noqa: E402
    EXCLUDED_GPU_NODES,
    IMAGE_DIGEST,
    TRAINING_GPU_RESOURCE_KEY,
    _render_evaluator_pod,
    _render_controller_pod,
    _render_training_pod,
)
from fixed_holdout.formal_runtime import (  # noqa: E402
    CONTINUATION_STAGING_CONTENT_SCHEMA,
    FORMAL_STAGING_CONTENT_SCHEMAS,
    STAGING_CONTENT_SCHEMA,
)


TRAINING = (
    ROOT / "deploy" / "run_decision_admissibility_wp8_tier2_formal_training_devpod.sh"
)
EVALUATOR = (
    ROOT / "deploy" / "run_decision_admissibility_wp8_tier2_formal_evaluator_devpod.sh"
)
STAGER = ROOT / "deploy" / "stage_decision_admissibility_wp8_tier2_formal.sh"
HOST_LAUNCHER = ROOT / "deploy" / "run_decision_admissibility_wp8_tier2_formal_block.sh"
STAGING_PIPELINE = (
    ROOT / "deploy" / "run_decision_admissibility_wp8_tier2_formal_staging_pipeline.sh"
)


def _embedded_python(path: Path) -> list[str]:
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
        assert end < len(lines), (path, start)
        blocks.append("\n".join(lines[start:end]) + "\n")
        index = end + 1
    return blocks


def _template() -> dict:
    return {
        "expected_training_pod_name": "formal-gpu",
        "expected_evaluator_pod_name": "formal-cpu",
        "container_image_digest": IMAGE_DIGEST,
    }


def _mounts(document: dict) -> dict[str, dict]:
    return {
        row["mountPath"]: row
        for row in document["spec"]["containers"][0]["volumeMounts"]
    }


def test_formal_scripts_and_embedded_python_are_syntactically_valid() -> None:
    for path in (TRAINING, EVALUATOR, STAGER, STAGING_PIPELINE, HOST_LAUNCHER):
        subprocess.run(["bash", "-n", str(path)], check=True)
    for path in (TRAINING, EVALUATOR, STAGER, HOST_LAUNCHER):
        blocks = _embedded_python(path)
        assert blocks
        for index, source in enumerate(blocks):
            ast.parse(source, filename=f"{path}:heredoc:{index}")


def test_generated_training_pod_is_one_gpu_label_isolated_devpod() -> None:
    document = _render_training_pod(
        _template(),
        source_root=Path("/workspace/formal-source"),
        data_root=Path("/workspace/formal-data"),
        bundle_root=Path("/workspace/formal-bundle"),
        contract_root=Path("/workspace/formal-contract"),
        output_root=Path("/workspace/formal-output"),
        content_hash="a" * 64,
    )
    assert document["kind"] == "Pod"
    assert document["spec"]["restartPolicy"] == "Never"
    assert "sleep infinity" in document["spec"]["containers"][0]["args"][0]
    mounts = _mounts(document)
    assert set(mounts) == {
        "/opt/nautilus",
        "/task",
        "/memory",
        "/contract",
        "/output",
        "/secrets/mlevolve.env",
        "/work",
        "/cache",
    }
    assert "/workspace" not in mounts
    assert "/fixed/evaluator_view" not in mounts
    for path in (
        "/opt/nautilus",
        "/task",
        "/memory",
        "/contract",
        "/secrets/mlevolve.env",
    ):
        assert mounts[path]["readOnly"] is True
    resources = document["spec"]["containers"][0]["resources"]
    assert (
        resources["requests"]
        == resources["limits"]
        == {
            "cpu": "8",
            "memory": "32Gi",
            TRAINING_GPU_RESOURCE_KEY: "1",
        }
    )
    assert TRAINING_GPU_RESOURCE_KEY == "nvidia.com/rtxa6000"
    expression = document["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"][0]
    assert expression == {
        "key": "kubernetes.io/hostname",
        "operator": "NotIn",
        "values": list(EXCLUDED_GPU_NODES),
    }


def test_generated_evaluator_pod_is_cpu_only_without_memory_or_secret() -> None:
    document = _render_evaluator_pod(
        _template(),
        source_root=Path("/workspace/formal-source"),
        data_root=Path("/workspace/formal-data"),
        contract_root=Path("/workspace/formal-contract"),
        output_root=Path("/workspace/formal-output"),
        content_hash="a" * 64,
    )
    assert document["kind"] == "Pod"
    mounts = _mounts(document)
    assert set(mounts) == {
        "/opt/nautilus",
        "/fixed/train_view",
        "/fixed/evaluator_view",
        "/contract",
        "/output",
        "/work",
    }
    assert "/workspace" not in mounts
    assert "/memory" not in mounts
    assert "/secrets/mlevolve.env" not in mounts
    resources = document["spec"]["containers"][0]["resources"]
    assert (
        resources["requests"]
        == resources["limits"]
        == {
            "cpu": "4",
            "memory": "8Gi",
        }
    )
    assert not any(key.startswith("nvidia.com/") for key in resources["limits"])


def test_generated_controller_has_only_staging_and_output_mounts() -> None:
    document = _render_controller_pod(
        source_root=Path("/workspace/formal-source"),
        staging_root=Path("/workspace/formal-staging"),
        output_root=Path("/workspace/formal-output"),
        image_digest=IMAGE_DIGEST,
        content_hash="a" * 64,
    )
    assert document["kind"] == "Pod"
    mounts = _mounts(document)
    assert set(mounts) == {
        "/opt/nautilus",
        "/formal/staging",
        "/formal/outputs",
        "/work",
    }
    assert mounts["/opt/nautilus"]["readOnly"] is True
    assert "/workspace" not in mounts
    assert "/memory" not in mounts
    assert "/secrets/mlevolve.env" not in mounts
    resources = document["spec"]["containers"][0]["resources"]
    assert (
        resources["requests"]
        == resources["limits"]
        == {
            "cpu": "1",
            "memory": "2Gi",
        }
    )


def test_formal_training_runner_changes_only_declared_retrieval_control() -> None:
    text = TRAINING.read_text(encoding="utf-8")
    assert 'external_skill_memory.retrieval_control="$condition"' in text
    assert 'run_identity.memory_system="$condition"' in text
    assert "only_experimental_variable" in text
    assert "external_skill_memory.retrieval_control" in text
    assert 'agent.initial_drafts="$INITIAL_DRAFTS"' in text
    assert 'agent.steps="$STEPS"' in text
    assert 'agent.time_limit="$TIME_LIMIT"' in text
    assert "agent.search.num_gpus=1" in text
    assert "agent.search.parallel_search_num=1" in text
    assert "cpu_number=8" in text
    assert "allow_source_score_inheritance=false" in text
    assert "allow_dataset_wide_per_sample_precompute=true" in text
    assert "fixed_holdout.evaluation_mode=terminal_only" in text
    assert "coldstart.use_coldstart=false" in text
    assert "TRAINING_POD_DELETION_ATTESTATION" not in text
    assert "kubectl" not in text
    assert "FORMAL_STAGING_CONTENT_SCHEMAS" in text


def test_formal_runtime_accepts_original_and_continuation_staging_schemas() -> None:
    assert FORMAL_STAGING_CONTENT_SCHEMAS == {
        STAGING_CONTENT_SCHEMA,
        CONTINUATION_STAGING_CONTENT_SCHEMA,
    }


def test_formal_evaluator_requires_host_deletion_and_isolation_receipts() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")
    assert "TRAINING_POD_DELETION_ATTESTATION.json" in text
    assert "EVALUATOR_POD_CREATION_ATTESTATION.json" in text
    assert "EVALUATOR_ISOLATION.json" in text
    assert "--training-pod-deletion-attestation" in text
    assert "--evaluator-isolation" in text
    assert "environment_has_solver_secret" in text
    assert "effect_claim_authorized" in text
    assert "DEEPSEEK_API_KEY" not in text
    assert "/memory" in text
    assert "test ! -e /memory" in text


def test_formal_evaluator_supports_hash_bound_finalizer_overlay() -> None:
    text = EVALUATOR.read_text(encoding="utf-8")

    assert 'FINALIZER_SRC="${WP8_FORMAL_FINALIZER_SOURCE_ROOT:-$SRC}"' in text
    assert 'test -d "$FINALIZER_SRC/mlevolve"' in text
    assert (
        'PYTHONPATH="$FINALIZER_SRC/mlevolve:$FINALIZER_SRC:'
        '$SRC/mlevolve:$SRC"' in text
    )


def test_formal_stager_never_stages_user_assets_or_launches_gpu() -> None:
    text = STAGER.read_text(encoding="utf-8")
    archive = text.split("git archive", 1)[1].split("runtime_list=", 1)[0]
    assert "mlevolve/data" not in archive
    assert "mlevolve/runs" not in archive
    assert "outputs" not in archive
    assert "kubectl create" not in text
    assert "kubectl apply" not in text
    assert "kind: Job" not in text
    assert "git add" not in text
    assert "git commit" not in text
    assert "git push" not in text
    assert "jupyter-a10-d48dfd589-pqfkb" in text


def test_formal_retry_uses_fresh_roots_and_binds_superseded_attempts() -> None:
    stager = STAGER.read_text(encoding="utf-8")
    pipeline = STAGING_PIPELINE.read_text(encoding="utf-8")
    launcher = HOST_LAUNCHER.read_text(encoding="utf-8")
    for revision in (
        "formal-source-r10",
        "formal-control-r10",
        "formal-staging-r12",
        "formal-runs-r10",
        "formal-staging-r12-stop-gate-r1",
        "formal-staging-r12-pipeline-r1",
    ):
        assert revision in stager
    assert "formal_staging_stop_gate_20260723_r10" in stager
    assert "formal_staging_stop_gate_20260723_r10" in launcher
    assert "formal-stager-cpu-r11" in stager
    assert (
        "decision_admissibility_wp8_tier2_formal_preregistration_20260723_r5.json"
        in pipeline
    )
    assert (
        "decision_admissibility_wp8_tier2_formal_r8_authority_failure_diagnostic_20260723.json"
        in pipeline
    )
    assert "WP8_FORMAL_SUPERSEDED_STAGING_ABORT" in stager
    assert "WP8_FORMAL_SUPERSEDED_COMPATIBILITY_ABORT" in stager
    assert "WP8_FORMAL_SUPERSEDED_NODE_EVICTION_ABORT" in stager
    assert "WP8_FORMAL_SUPERSEDED_RANKING_DIAGNOSTIC" in stager
    assert "WP8_FORMAL_SUPERSEDED_PRELAUNCH_ABORT" in stager
    assert "WP8_FORMAL_SUPERSEDED_STAGING_TRANSFER_ABORT" in stager
    assert "WP8_FORMAL_SUPERSEDED_SECOND_PRELAUNCH_ABORT" in stager
    assert "WP8_FORMAL_SUPERSEDED_PHASE_POLLUTION_ABORT" in stager
    assert "WP8_FORMAL_SUPERSEDED_PHASE_POLLUTION_DIAGNOSTIC" in stager
    assert "WP8_FORMAL_SUPERSEDED_R2_SCHEDULING_ABORT" in stager
    assert "kubectl_retry" in stager
    assert "upload_list_archive" in stager
    assert pipeline.count("--superseded-evidence") == 10
    assert pipeline.count("--failed-formal-evidence") == 1


def test_host_launcher_enforces_gate_and_gpu_delete_before_evaluator() -> None:
    text = HOST_LAUNCHER.read_text(encoding="utf-8")
    assert 'gate["formal_training_authorized"] is True' in text
    assert "STAGING_STOP_GATE.json" in text
    assert "jupyter-a10-d48dfd589-pqfkb" in text
    assert "kind: Job" not in text
    assert "kubectl_clean create -f -" in text
    assert "training-deletion" in text
    assert "evaluator-creation" in text
    assert "evaluator-deletion" in text
    assert "evaluator-failure" in text
    assert "da-wp8-f-controller-cpu-r3" in text
    assert "formal-controller-cpu-r3.yaml" in text
    assert "training-infrastructure-abort" in text
    assert "training-prelaunch-abort" in text
    assert "training-precontract-abort" in text
    assert "record_training_launcher_failure" in text
    assert "FORMAL_BLOCK_INFRASTRUCTURE_ABORT.json" not in text
    assert "event-snapshot-sha256" in text
    assert "involvedObject.uid=$CREATED_POD_UID" in text
    assert "involvedObject.uid=$TRAINING_UID" in text
    assert "POLL_FAILURE_PHASE" in text
    assert "NotFound|Failed|Succeeded" in text
    assert '2>"$error_file"' in text
    assert "verify_delete_not_found" in text
    assert "EVAL_FAILURE_NOT_FOUND_SHA" in text
    assert "the failed root was hash-sealed" in text
    assert text.index('delete pod "$TRAINING_POD"') < text.index(
        'create_remote_pod evaluator "$EVALUATOR_POD"'
    )
    assert text.index("training-deletion") < text.index(
        'create_remote_pod evaluator "$EVALUATOR_POD"'
    )
