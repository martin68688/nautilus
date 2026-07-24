"""Implementation guideline."""

import time

import humanize

from engine.candidate_execution_contract import (
    candidate_execution_contract_from_cfg,
)


def get_candidate_execution_contract_from_agent(agent):
    """Return the canonical host-owned contract, or an empty mapping."""

    return candidate_execution_contract_from_cfg(agent.cfg) or {}


def get_impl_guideline_from_agent(agent):
    """Build implementation guideline from agent config."""
    tot_time_remaining = agent.acfg.time_limit - (time.time() - agent.start_time)
    exec_timeout = int(min(agent.cfg.exec.timeout, tot_time_remaining))
    return get_impl_guideline(
        tot_time_remaining=tot_time_remaining,
        steps_remaining=agent.acfg.steps - agent.current_step,
        exec_timeout=exec_timeout,
        expose_prediction=getattr(agent.acfg, "expose_prediction", False),
        k_fold_validation=getattr(agent.acfg, "k_fold_validation", 0),
        pretrain_model_dir=getattr(agent.cfg, "pretrain_model_dir", ""),
        candidate_execution_contract=get_candidate_execution_contract_from_agent(
            agent
        ),
    )


def _format_time(time_in_sec):
    """Format seconds for display."""
    return f"{int(time_in_sec) // 3600}h {(int(time_in_sec) % 3600) // 60}m {int(time_in_sec) % 60}s"


def get_impl_guideline(
    tot_time_remaining: float,
    steps_remaining: int,
    exec_timeout: int,
    expose_prediction: bool = False,
    k_fold_validation: int = 0,
    pretrain_model_dir: str = "",
    candidate_execution_contract: dict | None = None,
) -> dict:
    """Build implementation guideline from time and config."""
    impl_guideline = [
        f"**Resource Budget**: Time left ≈ {_format_time(tot_time_remaining)} | Steps left = {steps_remaining} | Max execution time per run = {humanize.naturaldelta(exec_timeout)}",
        "",
        f"**Hard deadline:** Code execution MUST complete within {humanize.naturaldelta(exec_timeout)}. This is the actual per-candidate host limit; any conflicting generic runtime guidance is void.",
        "🎯 **CRITICAL REQUIREMENTS** (Non-Negotiable):",
        "",
        "**1. Model Inference for ALL Predictions**",
        "• EVERY prediction (validation & test) MUST come from trained model's forward pass",
        "• Process: Load data → Preprocess → model.predict()/model.forward() → Save predictions",
        "• ❌ FORBIDDEN: Constants, placeholders, dummy values, empty arrays, statistics, random numbers",
        "• ❌ FORBIDDEN: Fake/mock metric functions (must use real sklearn.metrics or correct manual implementation)",
        "• Why: Shortcuts create fake high validation scores but fail on test (CRITICAL SYSTEM FAILURE)",
        "",
        "**2. Generate submission.csv**",
        "• Path: `./submission/submission.csv` (NOT ./working/submission.csv)",
        "• Content: Model predictions on ALL test samples",
        "• Format: Follow task description exactly",
        "",
        "**3. Print Validation Metric**",
        "• MUST print: `print(f'Final Validation Score: {score}')`",
        "• Score MUST be computed on hold-out validation set using proper metric formula",
        "• CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "",
        "📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`",
        "",
        f"📦 **Packages & Internet**: numpy, pandas, sklearn, torch, transformers, timm, xgboost, lightgbm (all pre-installed). torch.hub.load(), HuggingFace, etc. available during development."
        + (f" Offline models at `{pretrain_model_dir}`" if pretrain_model_dir else ""),
        "",
        "⚠️ **API Compatibility**: LightGBM/XGBoost: ❌ `fit(..., early_stopping_rounds=...)` → ✅ LightGBM: `fit(..., callbacks=[lgb.early_stopping(...)])` ✅ XGBoost: `XGBClassifier(early_stopping_rounds=...)`",
        "• AdamW: ❌ `from transformers import AdamW` (deprecated) → ✅ `from torch.optim import AdamW`",
        "",
        "🚫 **Execution Guidelines**:",
        "• NO tqdm (not installed), NO verbose=1",
        "• Print only 1 line per epoch (minimize logging)",
        "• Use DataLoader with num_workers>=2 for speed",
        "",
        "⚠️  **Self-Check Before Finalizing**:",
        "□ Did predictions pass through model's learned weights during inference? (If NO → INVALID)",
        "□ Did I generate submission.csv in correct path with ALL test predictions?",
        "□ Did I print validation metric as the last line?",
        "□ Did I use the COMPLETE training dataset (not a tiny subset)?",
    ]
    contract = dict(candidate_execution_contract or {})
    if contract:
        allowed = ", ".join(contract["allowed_import_roots"])
        impl_guideline.extend(
            [
                "",
                "🧪 **PAIRED CANDIDATE EXECUTION CONTRACT (HOST-ENFORCED)**:",
                f"• Contract ID/hash: `{contract['contract_id']}` / `{contract['contract_hash']}`.",
                "• This feasibility boundary is identical for No-Memory and memory-enabled conditions. Memory content cannot relax it.",
                f"• Produce a complete model-inference submission within {contract['max_execution_seconds']} seconds; the host terminates longer candidates.",
                f"• Train at most {contract['max_epochs']} epochs, use at most {contract['max_cv_folds']} validation fold, and instantiate at most {contract['max_trainable_models']} trainable model. No CV ensemble or model ensemble.",
                f"• Imports are restricted to the Python standard library plus these verified roots: {allowed}.",
                "• Do not use `skimage` or another package outside that allowlist, even if a transitive dependency appears likely to exist.",
                "• Do not download or resolve remote repositories, pretrained weights, HuggingFace assets, or torch.hub assets. Use locally constructible architectures with `weights=None`/`pretrained=False`.",
                "• Do not load a pre-existing local checkpoint or repository. A checkpoint created by this candidate under `./working` may be reloaded by the same candidate.",
                "• Do not precompute handcrafted features with a Python per-image pass over the full dataset. Use batched DataLoader/tensor operations or vectorized tabular operations.",
                "• If a retrieved method is too expensive, implement a budget-compatible instance of its method hypothesis; never copy or inherit its source-task score, rank, or success conclusion.",
                "• The host statically audits imports, remote assets, literal local loads, fold count, and epoch count before execution; violations fail closed.",
            ]
        )
    if expose_prediction:
        impl_guideline.append(
            "The implementation should include a predict() function, "
            "allowing users to seamlessly reuse the code to make predictions on new data. "
            "The prediction function should be well-documented, especially the function signature."
        )

    if k_fold_validation > 1:
        impl_guideline.append(
            f"The evaluation should be based on {k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
        )

    return {"Implementation guideline": impl_guideline}
