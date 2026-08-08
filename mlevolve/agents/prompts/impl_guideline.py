"""Implementation guideline."""

import ast
import json
from pathlib import Path
import time

import humanize

from engine.candidate_execution_contract import (
    candidate_execution_contract_from_cfg,
)


def get_candidate_execution_contract_from_agent(agent):
    """Return the canonical host-owned contract, or an empty mapping."""

    return candidate_execution_contract_from_cfg(agent.cfg) or {}


def host_protocol_preflight_enabled(agent) -> bool:
    preflight = getattr(getattr(agent, "acfg", None), "protocol_preflight", None)
    return bool(preflight is not None and getattr(preflight, "enabled", False))


def submission_aligned_metric_required(agent) -> bool:
    identity = getattr(getattr(agent, "cfg", None), "run_identity", None)
    return bool(
        getattr(identity, "require_submission_aligned_internal_metric", False)
    )


def get_host_protocol_contract_from_agent(agent) -> dict:
    """Return the frozen Host SDK surface that generated code must implement."""

    if not host_protocol_preflight_enabled(agent):
        return {}
    preflight = agent.acfg.protocol_preflight
    contract = json.loads(
        Path(str(preflight.contract_path))
        .resolve(strict=True)
        .read_text(encoding="utf-8")
    )
    manifest = json.loads(
        Path(str(preflight.data_view_manifest_path))
        .resolve(strict=True)
        .read_text(encoding="utf-8")
    )
    task_id = str(contract["task_id"])
    fallback_labels = {
        "aerial-cactus-identification": "label",
        "denoising-dirty-documents": "target",
        "leaf-classification": "label",
        "new-york-city-taxi-fare-prediction": "fare",
        "spooky-author-identification": "author",
    }
    label_key = str(
        (manifest.get("strategy_verification") or {}).get("label_key")
        or fallback_labels.get(task_id)
        or "label"
    )
    metric_name = str(
        ((contract.get("evaluator_spec") or {}).get("metric") or {}).get("name")
        or ""
    )
    return {
        "task_id": task_id,
        "task_family": str(contract["task_family"]),
        "contract_id": str(contract["contract_id"]),
        "contract_hash": str(contract["contract_hash"]),
        "label_key": label_key,
        "metric_name": metric_name,
        "allowed_import_roots": list(contract["allowed_import_roots"]),
        "execution_budget": dict(contract["execution_budget"]),
        "inference_view_required": bool(
            (contract.get("adapter_spec") or {}).get(
                "inference_view_required", False
            )
        ),
    }


def _host_candidate_source(contract: dict) -> str:
    label_key = contract["label_key"]
    metric_name = contract["metric_name"]
    if contract["task_id"] == "denoising-dirty-documents":
        body = (
            "    with session.fit_scope(component=\"denoising_model\", data_view=views.train) as train_rows:\n"
            "        if not train_rows:\n"
            "            raise ValueError(\"Host training view is empty\")\n"
            "    with session.prediction_scope(component=\"denoising_model\", data_view=views.validation) as validation_rows:\n"
            "        predictions = [row[\"assets\"][\"noisy\"] for row in validation_rows]\n"
        )
    elif metric_name in {"log_loss", "roc_auc"}:
        prediction_line = (
            "        predictions = [0.5 for _ in validation_rows]\n"
            if metric_name == "roc_auc"
            else "        predictions = [[probability] * len(classes) for _ in validation_rows]\n"
        )
        body = (
            "    with session.fit_scope(component=\"classification_model\", data_view=views.train) as train_rows:\n"
            f"        classes = sorted({{row[{label_key!r}] for row in train_rows}}, key=str)\n"
            "        if len(classes) < 2:\n"
            "            raise ValueError(\"Host classification view requires at least two classes\")\n"
            "    with session.prediction_scope(component=\"classification_model\", data_view=views.validation) as validation_rows:\n"
            "        probability = 1.0 / len(classes)\n"
            + prediction_line
        )
    else:
        body = (
            "    with session.fit_scope(component=\"regression_model\", data_view=views.train) as train_rows:\n"
            f"        fitted_value = sum(float(row[{label_key!r}]) for row in train_rows) / len(train_rows)\n"
            "    with session.prediction_scope(component=\"regression_model\", data_view=views.validation) as validation_rows:\n"
            "        predictions = [fitted_value for _ in validation_rows]\n"
        )
    inference_source = ""
    if contract.get("inference_view_required", False):
        inference_source = (
            "\n    with session.inference_scope(component=\"final_submission\", data_view=views.inference) as inference_rows:\n"
            "        _ = len(inference_rows)"
        )
    return (
        "def candidate(session):\n"
        "    views = session.get_split()\n"
        + body
        + f"    session.evaluate_internal(views.validation, predictions, label_key={label_key!r})\n"
        "    session.freeze_selection(\"host_protocol_dry_run\", based_on=views.validation, artifact_hash=\"0\" * 64)"
        + inference_source
    )


def _host_full_runtime_validation_source(contract: dict) -> str:
    """Return a deterministic two-path source used only by Host smoke tests."""

    candidate_source = _host_candidate_source(contract)
    candidate_body = candidate_source.splitlines()[1:]
    # ``candidate_body`` already carries one function-body indentation level.
    # Reuse it verbatim under ``main``; adding another level produces invalid
    # source (``unexpected indent``) and makes the full-runtime smoke fail
    # before it can exercise any lifecycle receipts.
    main_body = "\n".join(candidate_body)
    return (
        candidate_source
        + "\n\nfrom protocol_runtime import current_session\n\n"
        + "def main():\n"
        + "    session = current_session()\n"
        + main_body
        + "\n\nif __name__ == \"__main__\":\n"
        + "    main()\n"
    )


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
        return False
    test = node.test
    return bool(
        isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _install_host_candidate_source(code: str, candidate_source: str) -> str:
    """Replace every top-level Candidate entrypoint with the frozen Host source."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    lines = code.splitlines(keepends=True)
    removal: set[int] = set()
    main_guard_line: int | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "candidate":
            start = min(
                [node.lineno]
                + [decorator.lineno for decorator in node.decorator_list]
            )
            end = int(node.end_lineno or node.lineno)
            removal.update(range(start - 1, end))
        elif main_guard_line is None and _is_main_guard(node):
            main_guard_line = node.lineno - 1

    installed = candidate_source.rstrip() + "\n\n"
    output: list[str] = []
    inserted = False
    for index, line in enumerate(lines):
        if not inserted and main_guard_line == index:
            output.append(installed)
            inserted = True
        if index not in removal:
            output.append(line)
    if not inserted:
        if output and not output[-1].endswith("\n"):
            output[-1] += "\n"
        if output and output[-1].strip():
            output.append("\n")
        output.append(candidate_source.rstrip() + "\n")
    return "".join(output)


def enforce_host_candidate_entrypoint(agent, code: str) -> str:
    """Install the exact Contract-bound entrypoint after all LLM review edits."""

    if not host_protocol_preflight_enabled(agent):
        return code
    if not bool(
        getattr(
            agent.acfg.protocol_preflight,
            "install_host_candidate_entrypoint",
            True,
        )
    ):
        return code
    contract = get_host_protocol_contract_from_agent(agent)
    return _install_host_candidate_source(code, _host_candidate_source(contract))


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
        host_protocol_contract=get_host_protocol_contract_from_agent(agent),
        require_submission_aligned_metric=submission_aligned_metric_required(agent),
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
    host_protocol_contract: dict | None = None,
    require_submission_aligned_metric: bool = False,
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
        (
            "• MUST make the very last line: `print(f'Final Submission-Aligned Validation Score: {score} | variant={submission_variant}')`"
            if require_submission_aligned_metric
            else "• MUST print: `print(f'Final Validation Score: {score}')`"
        ),
        "• Score MUST be computed on hold-out validation set using the proper metric formula",
        *(
            [
                "• SUBMISSION ALIGNMENT IS REQUIRED: `score` must evaluate the exact model/ensemble, preprocessing, post-processing and blend weights used for `submission.csv`.",
                "• If you compare NN-only, tree-only and blend variants, print them as diagnostics, select one variant, then use that same selected variant for both validation scoring and test submission.",
                "• Set `submission_variant` to a short stable name for the actually submitted variant. A component-only best score paired with a different submitted blend is invalid.",
            ]
            if require_submission_aligned_metric
            else []
        ),
        "• CRITICAL CONSISTENCY REQUIREMENT: Ensure that validation and test inference use IDENTICAL processing logic. Any differences in how validation and test data are handled (such as post-processing, reconstruction, or formatting) can cause large performance gaps between validation and test sets. Maintain consistency across all data processing steps for both validation and test phases.",
        "",
        "📁 **Directories**: Input data in `./input/`, submission in `./submission/`, temp files in `./working/`",
        "",
        (
            "📦 **Packages & Internet**: Host does not prescribe a model family, package allowlist, or asset strategy; actual runtime availability still applies."
            if host_protocol_contract
            else f"📦 **Packages & Internet**: numpy, pandas, sklearn, torch, transformers, timm, xgboost, lightgbm (all pre-installed). torch.hub.load(), HuggingFace, etc. available during development."
            + (f" Offline models at `{pretrain_model_dir}`" if pretrain_model_dir else "")
        ),
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
        (
            "□ Does the final submission-aligned metric use the exact prediction variant written to submission.csv?"
            if require_submission_aligned_metric
            else "□ Did I print validation metric as the last line?"
        ),
        "□ Did I use the COMPLETE training dataset (not a tiny subset)?",
    ]
    contract = dict(candidate_execution_contract or {})
    if contract:
        impl_guideline.extend(
            [
                "",
                "🧪 **PAIRED CANDIDATE EXECUTION CONTRACT (HOST-ENFORCED)**:",
                f"• Contract ID/hash: `{contract['contract_id']}` / `{contract['contract_hash']}`.",
                "• This feasibility boundary is identical for No-Memory and memory-enabled conditions. Memory content cannot relax it.",
                f"• Produce a complete model-inference submission within {contract['max_execution_seconds']} seconds; the host terminates longer candidates.",
                "• The Host does not restrict epochs, CV folds, models, ensembles, feature engineering, imports, checkpoints, or asset strategy. Legacy max/import/asset fields in this contract are compatibility metadata and are not enforcement inputs.",
                "• Never copy or inherit a source-task score, rank, or success conclusion. A current score must come from the current trusted evaluator.",
            ]
        )
    host_contract = dict(host_protocol_contract or {})
    if host_contract:
        budget = host_contract["execution_budget"]
        candidate_source = _host_candidate_source(host_contract)
        data_layout = [
            "• Real training input is the Host-owned `train_rows` / `validation_rows` yielded by the full-runtime scopes below; these correspond to `./input/train_view/data.jsonl` and `./input/internal_validation_view/data.jsonl`. Do not reopen, rediscover, or resplit `/task/public` outside the session scopes.",
            "• Full training is evidence-bearing, not an unobserved sibling of preflight. In `main()`, import `current_session` from `protocol_runtime`, call `session = current_session()` and `views = session.get_split()`, perform the actual model fit entirely inside `with session.fit_scope(..., data_view=views.train) as train_rows:`, and perform the actual internal-validation inference entirely inside `with session.prediction_scope(..., data_view=views.validation) as validation_rows:`.",
            f"• Compute the reported validation score only as `score = session.evaluate_internal(views.validation, predictions, label_key={host_contract['label_key']!r})`. Print that returned Host score, then call `session.freeze_selection(...)` on the real selected model artifact before submission/test inference. After selection freeze, final prediction is allowed, but fitting, validation evaluation, hyperparameter changes and model replacement are forbidden.",
        ]
        if host_contract.get("inference_view_required", False):
            data_layout.append(
                "• Unlabeled test/submission rows are available only as `views.inference` (alias `views.test`). Never read `./input/test.csv` or `/task/public` directly. After `session.freeze_selection(...)`, run final test prediction inside `with session.inference_scope(component=\"final_submission\", data_view=views.inference) as inference_rows:` and build `submission.csv` from those rows in their Host order."
            )
        if host_contract["task_id"] == "denoising-dirty-documents":
            data_layout.extend(
                [
                    "• Denoising row schema is exactly `{'sample_id': str, 'assets': {'noisy': str, 'target': str}}`. Each asset value is already a direct absolute read-only PNG path string. Use `row['assets']['noisy']` / `row['assets']['target']`; never call `.get('file')` on it and never join it to an `assets/` directory.",
                    "• Denoising images have different heights and widths. Preserve each image's original shape; do not `np.stack` whole images. If batching patches/crops, keep per-image shape metadata and reconstruct every output at its own original size.",
                    "• For Host RMSE, `predictions` must be a list containing exactly one reconstructed image array (or image path) per `validation_rows` item, in the same order. Do not pass `(sample_id, prediction, target)` tuples and do not supply targets to the evaluator; Host loads trusted targets from the validation DataView.",
                ]
            )
        elif host_contract["task_id"] == "leaf-classification":
            data_layout.extend(
                [
                    "• Leaf rows use direct scalar columns: `id`, `sample_id`, `margin1`…`margin64`, `shape1`…`shape64`, and `texture1`…`texture64`; the numeric suffix has no underscore. Training/validation rows additionally contain direct string field `label`. Construct the frame directly as `pd.DataFrame(train_rows)` (and likewise for validation/inference rows).",
                    "• There are no nested or JSON-encoded `margin_features`, `shape_features`, or `texture_features` fields. Do not call `json.loads` to reconstruct the 192 features and do not rename them to zero-based column names.",
                    "• `views.inference` has the same 192 direct feature columns and IDs but no `label`. Generate exactly one 99-class probability row per inference row, preserving Host order and the label encoder's class-column order.",
                ]
            )
        else:
            data_layout.append(
                f"• The supervised target is the direct row field `{host_contract['label_key']}` (not a nested metadata object)."
            )
        impl_guideline.extend(
            [
                "",
                "🔐 **HOST PROTOCOL SDK ENTRYPOINT (MANDATORY; SUPERSEDES CONFLICTING MODEL/MEMORY GUIDANCE)**:",
                f"• Frozen Contract: `{host_contract['contract_id']}` / `{host_contract['contract_hash']}`.",
                "• Method choice is Agent-controlled: Host imposes no limit on epochs, CV folds, trainable-model count, ensembles, model family, feature engineering, installed packages, checkpoints, or asset strategy.",
                f"• Host Preflight has an operational timeout of {budget['timeout_seconds']}s, while full execution is bounded separately by the Executor/job deadline. Timing out is an execution outcome, not evidence that the method or retrieved experience is invalid.",
                "• Host enforcement is leakage-only: use only the bound train/validation/inference views, keep fitting and tuning before selection freeze, never access validation/test labels outside the trusted evaluator, and never forge Collector evidence.",
                "• Put the complete training/submission workflow inside `def main():` and invoke it only with `if __name__ == \"__main__\": main()`; do not train or touch files at module import time. Host full-runtime pre-code activates the Contract-bound session before `main()` starts.",
                *data_layout,
                "• Lifecycle order is strict and one-way: perform real validation prediction, call `score = session.evaluate_internal(...)` exactly once, then call `session.freeze_selection(...)`; never call `evaluate_internal`, fit, tune, or replace the model after selection freeze.",
                "• Prefer `session.freeze_selection(\"./working/model_artifact.pt\", based_on=views.validation)` so Host hashes the real regular checkpoint. If using `artifact_hash=`, pass only a real 64-character lowercase SHA-256 (or an existing regular checkpoint path); never use Python `hash(...)`, `repr(...)`, or a custom `model_hash_...` string.",
                "• In the same code block, copy the following top-level `candidate(session)` function exactly. It is only the bounded preflight dry run. `main()` must independently execute the same five SDK lifecycle operations around the real model training and real validation predictions as required above.",
                "```python\n" + candidate_source + "\n```",
                "• Before answering, verify the final source calls `get_split`, `fit_scope`, `prediction_scope`, `evaluate_internal`, and `freeze_selection`"
                + (
                    ", plus `inference_scope` for final submission inference,"
                    if host_contract.get("inference_view_required", False)
                    else ""
                )
                + " and obeys every leakage boundary above.",
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
