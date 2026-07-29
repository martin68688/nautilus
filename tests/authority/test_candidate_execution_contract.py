from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from agents.prompts.impl_guideline import (
    _install_host_candidate_source,
    get_impl_guideline,
)
from engine.candidate_execution_contract import (
    audit_candidate_code,
    build_candidate_execution_contract,
    candidate_execution_contract_from_cfg,
    valid_candidate_execution_audit,
    valid_candidate_execution_block_receipt,
)
from engine.executor import Interpreter, _execution_environment


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


def test_isolated_candidate_environment_uses_writable_per_candidate_caches(
    tmp_path,
) -> None:
    cache_root = tmp_path / "candidate-cache"
    env = _execution_environment(cache_root)

    for name in (
        "HOME",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "TORCH_HOME",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "MPLCONFIGDIR",
        "NUMBA_CACHE_DIR",
        "KERAS_HOME",
    ):
        path = Path(env[name])
        assert path.is_dir()
        assert path.is_relative_to(cache_root)
        assert path.stat().st_mode & 0o777 == 0o777
    assert env["TOKENIZERS_PARALLELISM"] == "false"


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


def test_candidate_execution_contract_does_not_restrict_imports_or_remote_assets() -> None:
    code = """
import skimage
import torch
from torchvision.models import resnet18
model = resnet18(weights='DEFAULT')
torch.hub.load('owner/repository', 'model')
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is True
    assert not any("unauthorized_import_roots:" in row for row in audit["violations"])
    assert not any("remote_asset_refs:" in row for row in audit["violations"])
    assert valid_candidate_execution_audit(audit)


def test_candidate_execution_contract_does_not_restrict_epoch_or_cv_method() -> None:
    code = """
from sklearn.model_selection import StratifiedKFold
MAX_EPOCHS = 50
splitter = StratifiedKFold(n_splits=5)
for epoch in range(MAX_EPOCHS):
    pass
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is True
    assert not any("epoch_cap:" in row for row in audit["violations"])
    assert not any("cv_fold_cap:" in row for row in audit["violations"])


def test_zero_epoch_cap_preserves_historical_uncapped_semantics() -> None:
    contract = build_candidate_execution_contract(
        contract_id="historical-no-uniform-epoch-cap",
        max_execution_seconds=600,
        max_epochs=0,
        max_cv_folds=1,
        max_trainable_models=1,
        allowed_import_roots=["torch"],
        allow_remote_assets=False,
        allow_unverified_local_assets=False,
        allow_dataset_wide_per_sample_precompute=True,
        allow_source_score_inheritance=False,
    )
    audit = audit_candidate_code(
        "MAX_EPOCHS = 500\nfor epoch in range(MAX_EPOCHS):\n    pass\n",
        contract,
    )
    assert audit["valid"] is True
    assert not any(row.startswith("epoch_cap:") for row in audit["violations"])


def test_candidate_execution_contract_does_not_restrict_ensembles_or_feature_passes() -> None:
    code = """
import torch

def extract_features(image):
    return image

cnn_model = object()
xgb_model = object()
features = [extract_features(image) for image in images]
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is True
    assert not any("trainable_model_cap:" in row for row in audit["violations"])
    assert not any(
        "dataset_wide_per_sample_precompute:" in row for row in audit["violations"]
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

    assert audit["valid"] is True
    assert len(audit["observed_model_constructor_sites"]) == 2
    assert not any("trainable_model_cap:" in row for row in audit["violations"])


def test_feature_helper_assigned_to_model_named_variable_is_not_a_constructor() -> None:
    code = """
def add_pca_features(frame):
    return frame

pca_model = add_pca_features(training_frame)
"""
    audit = audit_candidate_code(code, _contract())

    assert audit["valid"] is True
    assert audit["observed_model_constructor_sites"] == []


def test_real_classifier_constructor_still_counts_after_helper_fix() -> None:
    code = """
from sklearn.linear_model import LogisticRegression
leaf_model = LogisticRegression()
aux_classifier = LogisticRegression()
"""
    audit = audit_candidate_code(code, _contract())

    assert len(audit["observed_model_constructor_sites"]) == 2
    assert not any("trainable_model_cap:" in row for row in audit["violations"])


def test_code_review_deterministic_audit_preserves_cv_method_choice() -> None:
    from agents.code_review_agent import _deterministic_contract_audit

    agent = SimpleNamespace(cfg=_cfg(_contract()))
    audit = _deterministic_contract_audit(
        agent,
        "from sklearn.model_selection import StratifiedKFold\n"
        "splitter = StratifiedKFold(n_splits=5)\n",
    )

    assert audit is not None
    assert not any("cv_fold_cap:" in row for row in audit["violations"])


def test_candidate_execution_contract_does_not_restrict_literal_checkpoint_method() -> None:
    audit = audit_candidate_code(
        "import torch\nstate = torch.load('/models/source-task-success.pth')\n",
        _contract(),
    )

    assert audit["valid"] is True
    assert not any("unverified_local_loads:" in row for row in audit["violations"])


def test_candidate_execution_contract_is_resolved_from_host_config() -> None:
    contract = _contract()
    resolved = candidate_execution_contract_from_cfg(_cfg(contract))

    assert resolved == contract
    assert resolved["allow_source_score_inheritance"] is False


def test_interpreter_fails_closed_before_running_rejected_code(tmp_path) -> None:
    marker = tmp_path / "must_not_exist"
    interpreter = Interpreter(tmp_path, timeout=900, cfg=_cfg(_contract()))

    result = interpreter.run(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef broken(\n",
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
    assert "never copy or inherit a source-task score" in text.lower()


def test_host_prompt_exposes_frozen_sdk_entrypoint_without_method_limits() -> None:
    host_contract = {
        "task_id": "denoising-dirty-documents",
        "task_family": "image",
        "contract_id": "pec-host-test",
        "contract_hash": "a" * 64,
        "label_key": "target",
        "metric_name": "rmse",
        "allowed_import_roots": [],
        "execution_budget": {"timeout_seconds": 60},
    }
    guideline = get_impl_guideline(
        tot_time_remaining=600,
        steps_remaining=1,
        exec_timeout=60,
        host_protocol_contract=host_contract,
    )["Implementation guideline"]
    text = "\n".join(guideline)

    assert host_contract["contract_id"] in text
    assert host_contract["contract_hash"] in text
    assert "def candidate(session)" in text
    for call in (
        "get_split",
        "fit_scope",
        "prediction_scope",
        "evaluate_internal",
        "freeze_selection",
    ):
        assert call in text
    assert "Host imposes no limit on epochs, CV folds" in text
    assert "ensembles, model family, feature engineering" in text
    assert "package allowlist" in text
    assert "No remote assets" not in text
    assert "folds≤" not in text
    assert "trainable models≤" not in text
    assert "row['assets']['noisy']" in text
    assert "direct absolute read-only PNG path string" in text
    assert "different heights and widths" in text
    assert "do not `np.stack` whole images" in text


def test_leaf_host_prompt_exposes_direct_feature_and_inference_schema() -> None:
    host_contract = {
        "task_id": "leaf-classification",
        "task_family": "tabular",
        "contract_id": "pec-leaf-host-test",
        "contract_hash": "b" * 64,
        "label_key": "label",
        "metric_name": "log_loss",
        "allowed_import_roots": [],
        "execution_budget": {"timeout_seconds": 60},
        "inference_view_required": True,
    }
    text = "\n".join(
        get_impl_guideline(
            tot_time_remaining=600,
            steps_remaining=1,
            exec_timeout=60,
            host_protocol_contract=host_contract,
        )["Implementation guideline"]
    )

    assert "`margin1`…`margin64`" in text
    assert "numeric suffix has no underscore" in text
    assert "`pd.DataFrame(train_rows)`" in text
    assert "`margin_1`" not in text
    assert "no nested or JSON-encoded `margin_features`" in text
    assert "views.inference" in text
    assert "session.inference_scope" in text


def test_non_host_prompt_preserves_normal_development_guidance() -> None:
    guideline = get_impl_guideline(
        tot_time_remaining=600,
        steps_remaining=1,
        exec_timeout=60,
    )["Implementation guideline"]
    text = "\n".join(guideline)
    assert "torch.hub.load(), HuggingFace, etc. available" in text
    assert "HOST PROTOCOL SDK ENTRYPOINT" not in text


def test_host_entrypoint_is_restored_after_reviewer_removes_or_duplicates_it() -> None:
    frozen = "def candidate(session):\n    session.get_split()\n"
    reviewed = '''import numpy as np

def candidate(session):
    raise RuntimeError("reviewer changed the Host lifecycle")

def main():
    print(np.ones(1))

def candidate(session):
    return None

if __name__ == "__main__":
    main()
'''
    installed = _install_host_candidate_source(reviewed, frozen)

    assert installed.count("def candidate(session):") == 1
    assert 'raise RuntimeError("reviewer changed the Host lifecycle")' not in installed
    assert "return None" not in installed
    assert installed.index("def candidate(session):") < installed.index(
        'if __name__ == "__main__":'
    )
    assert "def main():" in installed
