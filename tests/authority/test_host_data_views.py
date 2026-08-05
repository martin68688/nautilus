from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from authority.protocol_execution_contract import compile_protocol_execution_contract
from authority.protocol_registry import ProtocolRegistry
from protocol_runtime.data_views import (
    DataViewManifest,
    build_evaluator_launch_contract,
    build_training_mount_contract,
    materialize_data_views,
    verify_data_view_manifest,
)


REGISTRY = ProtocolRegistry("mlevolve/config/protocols")
BUDGET = {"max_epochs": 2, "max_folds": 1, "timeout_seconds": 60}


def _contract(protocol: str, task: str, family: str):
    return compile_protocol_execution_contract(
        REGISTRY.resolve(protocol),
        task_id=task,
        task_family=family,
        train_view_ref=f"view://{task}/train",
        validation_view_ref=f"view://{task}/internal-validation",
        terminal_view_ref=f"evaluator-only://{task}/terminal",
        execution_budget=BUDGET,
    )


def _cactus():
    return [
        {"sample_id": f"c-{label}-{index}", "label": label, "x": index}
        for label in (0, 1)
        for index in range(6)
    ]


def _birds():
    return [
        {
            "sample_id": f"b-{group}-{index}",
            "group_id": group,
            "label": [index % 2, (index + 1) % 2],
        }
        for group in ("recording-a", "recording-b", "recording-c", "recording-d")
        for index in range(3)
    ]


def _taxi():
    return [
        {"sample_id": f"t-{index}", "timestamp": index, "fare": index * 1.5}
        for index in range(12)
    ]


def _denoising():
    return [{"sample_id": f"d-{index}"} for index in range(12)]


def _attestation(manifest, *, not_found: bool = True):
    value = {
        "schema": "mlevolve_training_pod_deletion_attestation_v2",
        "not_found_verified": not_found,
        "kubernetes_reason": "NotFound" if not_found else "Running",
        "contract_hash": manifest.contract_hash,
        "data_view_manifest_hash": manifest.manifest_hash,
        "verified_by": "host_launcher",
        "terminal_metric_observed_before_not_found": False,
        "preterminal_closure_report_hash": _closure(manifest)["report_hash"],
        "attestation_hash": "",
    }
    value["attestation_hash"] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "attestation_hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return value


def _closure(manifest, *, status: str = "pass"):
    value = {
        "schema": "mlevolve_preterminal_evidence_closure_v1",
        "status": status,
        "contract_hash": manifest.contract_hash,
        "data_view_manifest_hash": manifest.manifest_hash,
        "terminal_exposure_count": 0,
        "terminal_score_observed": False,
        "evaluator_launch_authorized": status == "pass",
        "report_hash": "",
    }
    value["report_hash"] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in value.items() if key != "report_hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return value


@pytest.mark.parametrize(
    ("protocol", "task", "family", "records"),
    [
        ("random-classification@1", "cactus", "image", _cactus()),
        ("grouped-classification@1", "birds", "audio", _birds()),
        ("chronological-regression@1", "taxi", "tabular", _taxi()),
        (
            "deterministic-random-regression@1",
            "denoising",
            "image",
            _denoising(),
        ),
    ],
)
def test_three_host_materializers_are_hash_bound_and_isolated(
    tmp_path: Path, protocol: str, task: str, family: str, records
) -> None:
    contract = _contract(protocol, task, family)
    manifest, path = materialize_data_views(
        records, tmp_path / task, contract, split_id=f"{task}-split-1", seed="7"
    )
    report = verify_data_view_manifest(path, contract=contract)
    assert report["status"] == "pass"
    assert report["terminal_exposure_count"] == 0
    assert manifest.sample_overlap_count == 0
    assert manifest.group_overlap_count == 0
    assert manifest.future_to_past_count == 0
    assert manifest.terminal_view_mounted_in_training is False
    for relative in ("train_view/data.jsonl", "internal_validation_view/data.jsonl"):
        assert os.stat(tmp_path / task / relative).st_mode & 0o222 == 0


def test_unlabeled_inference_view_is_hash_bound_disjoint_and_read_only(
    tmp_path: Path,
) -> None:
    contract = _contract("random-classification@1", "cactus", "image")
    inference = [
        {"sample_id": f"test-{index}", "x": float(index)}
        for index in range(3)
    ]
    manifest, path = materialize_data_views(
        _cactus(),
        tmp_path / "views-with-inference",
        contract,
        inference_records=inference,
        inference_view_ref="view://cactus/test/inference",
        split_id="cactus-inference-split",
    )

    assert set(manifest.views) == {
        "train",
        "internal_validation",
        "inference",
    }
    assert manifest.views["inference"]["sample_count"] == 3
    assert verify_data_view_manifest(path, contract=contract)["status"] == "pass"
    inference_path = tmp_path / "views-with-inference/inference_view/data.jsonl"
    materialized_ids = [
        json.loads(line)["sample_id"]
        for line in inference_path.read_text(encoding="utf-8").splitlines()
    ]
    assert materialized_ids == [row["sample_id"] for row in inference]
    assert os.stat(inference_path).st_mode & 0o222 == 0
    mount = json.loads(
        (tmp_path / "views-with-inference/TRAINING_MOUNT_CONTRACT.json").read_text()
    )
    assert [row["role"] for row in mount["mounts"]] == [
        "train",
        "internal_validation",
        "inference",
        "manifest",
    ]


def test_grouped_and_chronological_semantics(tmp_path: Path) -> None:
    birds_contract = _contract("grouped-classification@1", "birds", "audio")
    birds, birds_path = materialize_data_views(
        _birds(), tmp_path / "birds", birds_contract, split_id="birds-split"
    )
    assert birds.views["train"]["group_id_sha256"]
    verify_data_view_manifest(birds_path, contract=birds_contract)

    taxi_contract = _contract("chronological-regression@1", "taxi", "tabular")
    taxi, taxi_path = materialize_data_views(
        _taxi(), tmp_path / "taxi", taxi_contract, split_id="taxi-split"
    )
    assert taxi.views["train"]["time_max"] < taxi.views["internal_validation"]["time_min"]
    verify_data_view_manifest(taxi_path, contract=taxi_contract)


def test_chronological_inference_view_is_not_required_to_be_future_dated(
    tmp_path: Path,
) -> None:
    contract = _contract("chronological-regression@1", "taxi", "tabular")
    manifest, path = materialize_data_views(
        _taxi(),
        tmp_path / "taxi-with-inference",
        contract,
        # A real test set may overlap the training time range; it is still
        # unlabeled and disjoint by sample_id, so it must not be treated as a
        # second validation split.
        inference_records=[
            {"sample_id": "test-row", "timestamp": 0, "x": 99.0},
        ],
        inference_view_ref="view://taxi/test/inference",
        split_id="taxi-inference-split",
    )
    assert manifest.views["inference"]["time_min"] is None
    assert verify_data_view_manifest(path, contract=contract)["status"] == "pass"


def test_manifest_and_data_tamper_fail_closed(tmp_path: Path) -> None:
    contract = _contract("random-classification@1", "cactus", "image")
    manifest, path = materialize_data_views(
        _cactus(), tmp_path / "views", contract, split_id="cactus-split"
    )
    payload = manifest.as_dict()
    payload["sample_overlap_count"] = 1
    with pytest.raises(ValueError, match="hash mismatch"):
        DataViewManifest.from_dict(payload)
    data = tmp_path / "views" / "train_view" / "data.jsonl"
    data.chmod(0o644)
    data.write_text(data.read_text() + "{}\n")
    with pytest.raises(ValueError, match="data view hash mismatch"):
        verify_data_view_manifest(path, contract=contract)


def test_host_assets_are_copied_hash_bound_and_source_independent(
    tmp_path: Path,
) -> None:
    contract = _contract("random-classification@1", "images", "image")
    source_root = tmp_path / "raw-assets"
    source_root.mkdir()
    records = []
    for label in (0, 1):
        for index in range(4):
            source = source_root / f"{label}-{index}.bin"
            source.write_bytes(f"pixels-{label}-{index}".encode())
            records.append(
                {
                    "sample_id": f"image-{label}-{index}",
                    "label": label,
                    "_host_assets": {"image": str(source)},
                }
            )
    manifest, manifest_path = materialize_data_views(
        records,
        tmp_path / "views",
        contract,
        split_id="images-split",
    )
    assert sum(int(view["asset_count"]) for view in manifest.views.values()) == 8
    for role in ("train", "internal_validation"):
        rows = [
            json.loads(line)
            for line in (
                tmp_path / "views" / f"{role}_view" / "data.jsonl"
            ).read_text().splitlines()
        ]
        assert all(Path(row["assets"]["image"]).is_file() for row in rows)
        assert all("_host_assets" not in row for row in rows)

    for source in source_root.iterdir():
        source.unlink()
    assert verify_data_view_manifest(manifest_path, contract=contract)["status"] == "pass"

    copied_asset = next((tmp_path / "views" / "train_view" / "assets").rglob("*.bin"))
    copied_asset.chmod(0o644)
    copied_asset.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        verify_data_view_manifest(manifest_path, contract=contract)


def test_runtime_manifest_verification_reuses_frozen_asset_attestation(
    tmp_path: Path,
) -> None:
    """Runtime checks stay cheap while the freeze-time check remains strict."""

    contract = _contract("random-classification@1", "images", "image")
    source_root = tmp_path / "raw-assets"
    source_root.mkdir()
    records = []
    for label in (0, 1):
        for index in range(4):
            source = source_root / f"{label}-{index}.bin"
            source.write_bytes(f"pixels-{label}-{index}".encode())
            records.append(
                {
                    "sample_id": f"image-{label}-{index}",
                    "label": label,
                    "_host_assets": {"image": str(source)},
                }
            )
    _manifest, manifest_path = materialize_data_views(
        records,
        tmp_path / "views",
        contract,
        split_id="images-runtime-attestation",
    )

    copied_asset = next((tmp_path / "views" / "train_view" / "assets").rglob("*.bin"))
    copied_asset.chmod(0o644)
    copied_asset.write_bytes(b"tampered-after-freeze")

    # Loading an already frozen binding checks the immutable manifests and
    # split invariants, without rereading every asset byte.
    assert verify_data_view_manifest(
        manifest_path,
        contract=contract,
        verify_asset_contents=False,
    )["status"] == "pass"
    # A new freeze/publication still performs the expensive content audit.
    with pytest.raises(ValueError, match="asset hash mismatch"):
        verify_data_view_manifest(manifest_path, contract=contract)


def test_path_traversal_and_symlink_fail_closed(tmp_path: Path) -> None:
    contract = _contract("random-classification@1", "cactus", "image")
    manifest, path = materialize_data_views(
        _cactus(), tmp_path / "views", contract, split_id="cactus-split"
    )
    payload = manifest.as_dict()
    payload["views"]["train"]["relative_path"] = "../../outside.jsonl"
    payload["manifest_hash"] = ""
    payload["manifest_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "manifest_hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    path.chmod(0o644)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe data-view relative path"):
        verify_data_view_manifest(path, contract=contract)

    other_root = tmp_path / "symlink-views"
    _manifest, other_path = materialize_data_views(
        _cactus(), other_root, contract, split_id="cactus-split-2"
    )
    original = other_root / "train_view" / "data.jsonl"
    backup = tmp_path / "outside.jsonl"
    backup.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(backup)
    with pytest.raises(ValueError, match="symlink"):
        verify_data_view_manifest(other_path, contract=contract)


def test_training_mount_allowlist_rejects_terminal_or_writable_mount(tmp_path: Path) -> None:
    contract = _contract("random-classification@1", "cactus", "image")
    manifest, _path = materialize_data_views(
        _cactus(), tmp_path / "views", contract, split_id="cactus-split"
    )
    mount_path = tmp_path / "views" / "TRAINING_MOUNT_CONTRACT.json"
    mount = json.loads(mount_path.read_text())
    assert mount["terminal_mount_count"] == 0
    terminal = tmp_path / "views" / "terminal_labels"
    terminal.mkdir()
    with pytest.raises(ValueError, match="exactly the Host-owned allowlist"):
        build_training_mount_contract(
            manifest,
            tmp_path / "views",
            extra_mounts=(
                {
                    "role": "terminal",
                    "source": str(terminal),
                    "target": "/data/terminal_holdout",
                    "read_only": True,
                },
            ),
        )
    broken = copy.deepcopy(mount)
    broken["mounts"][0]["read_only"] = False
    broken["mount_contract_hash"] = hashlib.sha256(
        json.dumps(
            {key: item for key, item in broken.items() if key != "mount_contract_hash"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="writable"):
        build_evaluator_launch_contract(
            manifest,
            broken,
            _attestation(manifest),
            _closure(manifest),
        )


def test_evaluator_launch_requires_hash_bound_training_not_found(tmp_path: Path) -> None:
    contract = _contract("chronological-regression@1", "taxi", "tabular")
    manifest, _path = materialize_data_views(
        _taxi(), tmp_path / "views", contract, split_id="taxi-split"
    )
    mount = json.loads(
        (tmp_path / "views" / "TRAINING_MOUNT_CONTRACT.json").read_text()
    )
    attestation = _attestation(manifest, not_found=False)
    with pytest.raises(ValueError, match="NotFound"):
        build_evaluator_launch_contract(
            manifest, mount, attestation, _closure(manifest)
        )
    attestation = _attestation(manifest, not_found=True)
    with pytest.raises(ValueError, match="Closure PASS"):
        build_evaluator_launch_contract(
            manifest, mount, attestation, _closure(manifest, status="blocked")
        )
    launch = build_evaluator_launch_contract(
        manifest, mount, attestation, _closure(manifest)
    )
    assert launch["training_not_found_verified"] is True
    assert launch["evaluation_system"] == "existing_fixed_holdout_terminal_evaluator"
    forged = dict(attestation)
    forged["attestation_hash"] = "f" * 64
    with pytest.raises(ValueError, match="hash-bound"):
        build_evaluator_launch_contract(
            manifest, mount, forged, _closure(manifest)
        )


def test_duplicate_samples_and_bad_group_or_time_are_rejected(tmp_path: Path) -> None:
    cactus = _cactus()
    cactus[1]["sample_id"] = cactus[0]["sample_id"]
    with pytest.raises(ValueError, match="unique"):
        materialize_data_views(
            cactus,
            tmp_path / "cactus",
            _contract("random-classification@1", "cactus", "image"),
            split_id="bad",
        )
    with pytest.raises(ValueError, match="two distinct groups"):
        materialize_data_views(
            [{"sample_id": "a", "group_id": "g"}, {"sample_id": "b", "group_id": "g"}],
            tmp_path / "birds",
            _contract("grouped-classification@1", "birds", "audio"),
            split_id="bad",
        )
    with pytest.raises(ValueError, match="distinct times"):
        materialize_data_views(
            [{"sample_id": "a", "timestamp": 1}, {"sample_id": "b", "timestamp": 1}],
            tmp_path / "taxi",
            _contract("chronological-regression@1", "taxi", "tabular"),
            split_id="bad",
        )
