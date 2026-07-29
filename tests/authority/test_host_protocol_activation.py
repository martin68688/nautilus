from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import _load_cfg, prep_cfg
from protocol_runtime.activation import (
    build_host_protocol_bundle,
    hash_sdk_tree,
    load_host_protocol_binding,
)


IMAGE = "sha256:" + "1" * 64
TASK = "activation-cactus"


def _bundle(tmp_path: Path) -> tuple[Path, Path]:
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"sample-{label}-{index}",
                    "label": label,
                    "x": float(index),
                }
            )
            + "\n"
            for label in (0, 1)
            for index in range(6)
        ),
        encoding="utf-8",
    )
    description = tmp_path / "description.md"
    description.write_text("Host activation fixture.\n", encoding="utf-8")
    private_key = tmp_path / "host-secrets" / "collector.ed25519"
    root = tmp_path / "bundle"
    build_host_protocol_bundle(
        output_root=root,
        records_path=records,
        description_path=description,
        registry_root="mlevolve/config/protocols",
        protocol_ref="random-classification@1",
        task_id=TASK,
        task_family="image",
        image_digest=IMAGE,
        sdk_root="mlevolve/protocol_runtime",
        collector_private_key_output=private_key,
        split_id="activation-split-v1",
        execution_budget={
            "max_epochs": 1,
            "max_folds": 1,
            "max_models": 2,
            "timeout_seconds": 30,
        },
    )
    return root / "HOST_PROTOCOL_BINDING.json", private_key


def test_binding_verifies_task_image_sdk_and_hash_drift(tmp_path: Path) -> None:
    binding_path, _private_key = _bundle(tmp_path)
    sdk_hash = hash_sdk_tree("mlevolve/protocol_runtime")
    loaded = load_host_protocol_binding(
        binding_path,
        expected_task_id=TASK,
        expected_image_digest=IMAGE,
        expected_sdk_hash=sdk_hash,
    )
    assert loaded["terminal_exposure_count"] == 0
    contract = json.loads(Path(loaded["contract_path"]).read_text(encoding="utf-8"))
    assert contract["execution_budget"] == {"timeout_seconds": 30}
    assert contract["allowed_import_roots"] == []

    with pytest.raises(ValueError, match="task mismatch"):
        load_host_protocol_binding(binding_path, expected_task_id="wrong-task")
    with pytest.raises(ValueError, match="image digest mismatch"):
        load_host_protocol_binding(
            binding_path,
            expected_image_digest="sha256:" + "2" * 64,
        )
    with pytest.raises(ValueError, match="SDK hash mismatch"):
        load_host_protocol_binding(binding_path, expected_sdk_hash="3" * 64)

    payload = json.loads(binding_path.read_text(encoding="utf-8"))
    manifest = Path(payload["data_view_manifest_path"])
    manifest.chmod(0o644)
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="DataView manifest file hash mismatch"):
        load_host_protocol_binding(binding_path)


def test_private_key_cannot_be_written_inside_candidate_bundle(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        "".join(
            json.dumps({"sample_id": f"s-{i}", "label": i % 2}) + "\n"
            for i in range(8)
        ),
        encoding="utf-8",
    )
    description = tmp_path / "description.md"
    description.write_text("fixture", encoding="utf-8")
    root = tmp_path / "bundle"
    with pytest.raises(ValueError, match="outside the Candidate bundle"):
        build_host_protocol_bundle(
            output_root=root,
            records_path=records,
            description_path=description,
            registry_root="mlevolve/config/protocols",
            protocol_ref="random-classification@1",
            task_id=TASK,
            task_family="image",
            image_digest=IMAGE,
            sdk_root="mlevolve/protocol_runtime",
            collector_private_key_output=root / "collector.ed25519",
            split_id="split",
            execution_budget={"timeout_seconds": 30},
        )


def test_binding_verifier_uses_manifest_nondefault_label_key(tmp_path: Path) -> None:
    records = tmp_path / "spooky-records.jsonl"
    records.write_text(
        "".join(
            json.dumps({
                "sample_id": f"sample-{author}-{index}",
                "text": f"text {index}",
                "author": author,
            }) + "\n"
            for author in ("EAP", "HPL", "MWS")
            for index in range(4)
        ),
        encoding="utf-8",
    )
    description = tmp_path / "description.md"
    description.write_text("Spooky activation fixture.\n", encoding="utf-8")
    private_key = tmp_path / "host-secrets" / "collector.ed25519"
    root = tmp_path / "spooky-bundle"
    build_host_protocol_bundle(
        output_root=root,
        records_path=records,
        description_path=description,
        registry_root="mlevolve/config/protocols",
        protocol_ref="stratified-log-loss-classification@1",
        task_id="spooky-author-identification",
        task_family="text",
        image_digest=IMAGE,
        sdk_root="mlevolve/protocol_runtime",
        collector_private_key_output=private_key,
        split_id="spooky-activation-v1",
        execution_budget={"timeout_seconds": 30},
        label_key="author",
    )
    loaded = load_host_protocol_binding(
        root / "HOST_PROTOCOL_BINDING.json",
        expected_task_id="spooky-author-identification",
        expected_image_digest=IMAGE,
        expected_sdk_hash=hash_sdk_tree("mlevolve/protocol_runtime"),
    )
    assert loaded["terminal_exposure_count"] == 0


def test_shadow_profile_is_bound_and_legacy_enforcement_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binding_path, private_key = _bundle(tmp_path)
    monkeypatch.setenv("MLEVOLVE_HOST_PROTOCOL_BINDING", str(binding_path))
    monkeypatch.setenv("MLEVOLVE_HOST_COLLECTOR_KEY_FILE", str(private_key))
    monkeypatch.setenv("MLEVOLVE_RUNTIME_IMAGE_DIGEST", IMAGE)
    cfg = _load_cfg(
        Path("mlevolve/config/config_authority_host_protocol_shadow.yaml"),
        use_cli_args=False,
    )
    cfg.exp_id = TASK
    candidate_data = tmp_path / "candidate-public"
    candidate_data.mkdir()
    cfg.data_dir = str(candidate_data)
    cfg.log_dir = str(tmp_path / "runs")
    cfg.workspace_dir = str(tmp_path / "runs")
    bound = prep_cfg(cfg)
    assert bound.evaluation_authority.protocol_runtime_mode == "host_sdk_shadow"
    assert bound.agent.protocol_repair.enabled is False
    assert bound.agent.protocol_preflight.legacy_ast_mode == "shadow"
    assert bound.agent.protocol_preflight.expected_contract_hash
    assert bound.data_dir == candidate_data.resolve()
