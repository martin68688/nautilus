from __future__ import annotations

import json
from types import SimpleNamespace

from agents.prompts.impl_guideline import get_impl_guideline
from engine.candidate_execution_contract import (
    audit_candidate_code,
    build_candidate_execution_contract,
    candidate_execution_contract_from_cfg,
    valid_candidate_execution_audit,
    valid_candidate_execution_block_receipt,
)
from engine.executor import Interpreter


def _contract() -> dict:
    return build_candidate_execution_contract(
        contract_id="wp8-tier2-canary-paired-feasibility-v1",
        max_execution_seconds=600,
        max_epochs=8,
        max_cv_folds=1,
        max_trainable_models=1,
        allowed_import_roots=[
            "numpy",
            "pandas",
            "sklearn",
            "scipy",
            "joblib",
            "torch",
            "torchvision",
            "cv2",
            "PIL",
            "xgboost",
            "lightgbm",
        ],
        allow_remote_assets=False,
        allow_unverified_local_assets=False,
        allow_dataset_wide_per_sample_precompute=False,
        allow_source_score_inheritance=False,
    )


def _cfg(contract: dict) -> SimpleNamespace:
    raw = {
        key: value
        for key, value in contract.items()
        if key not in {"schema", "contract_hash"}
    }
    return SimpleNamespace(
        start_cpu_id=0,
        cpu_number=1,
        exec=SimpleNamespace(timeout=600),
        evaluation_authority=SimpleNamespace(
            mode="off",
            runtime_protocol_observer_enabled=False,
        ),
        agent=SimpleNamespace(
            candidate_execution_contract=SimpleNamespace(**raw),
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
        ),
    )


def test_candidate_execution_contract_accepts_bounded_local_pipeline() -> None:
    code = """
import os
import numpy as np
import torch
from torchvision.models import resnet18

NUM_EPOCHS = 8
model = resnet18(weights=None)
for epoch in range(1, NUM_EPOCHS + 1):
    pass
torch.save(model.state_dict(), './working/best_model.pth')
torch.load('./working/best_model.pth')
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is True
    assert audit["violations"] == []
    assert valid_candidate_execution_audit(audit)


def test_candidate_execution_contract_rejects_unavailable_or_remote_assets() -> None:
    code = """
import skimage
import torch
from torchvision.models import resnet18
model = resnet18(weights='DEFAULT')
torch.hub.load('owner/repository', 'model')
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is False
    assert any("unauthorized_import_roots:skimage" in row for row in audit["violations"])
    assert any("remote_asset_refs:" in row for row in audit["violations"])
    assert valid_candidate_execution_audit(audit)


def test_candidate_execution_contract_rejects_epoch_and_cv_expansion() -> None:
    code = """
from sklearn.model_selection import StratifiedKFold
MAX_EPOCHS = 50
splitter = StratifiedKFold(n_splits=5)
for epoch in range(MAX_EPOCHS):
    pass
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is False
    assert any("epoch_cap:" in row for row in audit["violations"])
    assert any("cv_fold_cap:" in row for row in audit["violations"])


def test_candidate_execution_contract_rejects_ensembles_and_handcrafted_passes() -> None:
    code = """
import torch

def extract_features(image):
    return image

cnn_model = object()
xgb_model = object()
features = [extract_features(image) for image in images]
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is False
    assert any("trainable_model_cap:" in row for row in audit["violations"])
    assert any(
        "dataset_wide_per_sample_precompute:" in row
        for row in audit["violations"]
    )


def test_candidate_execution_contract_counts_chained_device_model_constructors() -> None:
    code = """
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    pass

cnn_model = TinyNet().to('cuda')
aux_model = TinyNet().cuda()
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is False
    assert len(audit["observed_model_constructor_sites"]) == 2
    assert any("trainable_model_cap:" in row for row in audit["violations"])


def test_candidate_execution_contract_rejects_unverified_literal_checkpoint() -> None:
    audit = audit_candidate_code(
        "import torch\nstate = torch.load('/models/source-task-success.pth')\n",
        _contract(),
    )

    assert audit["valid"] is False
    assert any("unverified_local_loads:" in row for row in audit["violations"])


def test_candidate_execution_contract_is_resolved_from_host_config() -> None:
    contract = _contract()
    resolved = candidate_execution_contract_from_cfg(_cfg(contract))

    assert resolved == contract
    assert resolved["allow_source_score_inheritance"] is False


def test_interpreter_fails_closed_before_running_rejected_code(tmp_path) -> None:
    marker = tmp_path / "must_not_exist"
    interpreter = Interpreter(tmp_path, timeout=900, cfg=_cfg(_contract()))

    result = interpreter.run(
        f"import skimage\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n",
        id="blocked-node",
        working_dir=str(tmp_path),
    )

    assert result.exc_type == "CandidateExecutionContractError"
    assert not marker.exists()
    receipt = tmp_path / "working" / "candidate_execution_contract_audit_blocked-node.json"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["valid"] is False
    assert valid_candidate_execution_audit(payload)
    block_path = (
        tmp_path / "working" / "candidate_execution_block_receipt_blocked-node.json"
    )
    block = json.loads(block_path.read_text(encoding="utf-8"))
    assert valid_candidate_execution_block_receipt(block)
    assert block["audit_hash"] == payload["audit_hash"]
    assert result.exc_info["block_receipt_path"] == str(block_path)
    assert interpreter.timeout == 600


def test_prompt_uses_actual_deadline_and_records_paired_contract() -> None:
    contract = _contract()
    guideline = get_impl_guideline(
        tot_time_remaining=2400,
        steps_remaining=3,
        exec_timeout=600,
        candidate_execution_contract=contract,
    )["Implementation guideline"]
    text = "\n".join(guideline)

    assert "within 10 minutes" in text
    assert "within 9 hours" not in text
    assert contract["contract_id"] in text
    assert contract["contract_hash"] in text
    assert "identical for No-Memory and memory-enabled conditions" in text
    assert "never copy or inherit its source-task score" in text
