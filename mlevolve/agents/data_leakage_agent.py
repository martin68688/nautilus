"""Data Leakage Agent: LLM-based data leakage check for node code."""

import json
import logging
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from utils.response import wrap_code

logger = logging.getLogger("MLEvolve")

DATA_LEAKAGE_CHECK_SPEC = FunctionSpec(
    name="check_data_leakage",
    json_schema={
        "type": "object",
        "properties": {
            "has_leakage": {
                "type": "boolean",
                "description": (
                    "true if there are signs of data leakage that could lead to unrealistically high validation metrics, "
                    "false otherwise. Common data leakage patterns include:\n"
                    "1. Using test/validation data during training (e.g., fitting scaler/encoder on full dataset)\n"
                    "2. Incorrect train/validation split causing temporal or group leakage\n"
                    "3. Feature engineering using global statistics from validation/test set\n"
                    "4. Data augmentation duplicating validation samples into training\n"
                    "5. Using target/future information not available at prediction time\n"
                    "6. Cross-fold/OOF embedding leakage: when using KFold/StratifiedKFold OOF, extracting "
                    "embeddings or meta-features for the val/holdout set with a SINGLE fold model (e.g. "
                    "last_fold_model / best_fold) whose training fold CONTAINS those holdout rows — the "
                    "embedding then carries supervised label info for rows it should not, leaking into any "
                    "downstream model fed those embeddings (e.g. XGBoost/stacker). Each row's embedding MUST "
                    "come from a fold model that did NOT see that row (proper OOF, same rule as OOF preds). "
                    "Watch for code that reuses one fold's model to extract train+val+test embeddings uniformly.\n"
                    "ALSO FLAG (not classic leakage but inflates val metric): ensemble weight optimization / "
                    "model selection done on the SAME val set used to report the final metric (select+score "
                    "on the same set) -> over-optimistic val, generalizes worse on test.\n"
                    "IMPORTANT: Consider task complexity. Simple tasks (e.g., clear binary classification) "
                    "can legitimately achieve near-perfect scores without leakage."
                ),
            },
            "leakage_reason": {
                "type": "string",
                "description": (
                    "Provide a detailed explanation:\n"
                    "- If has_leakage=true: Describe the specific code pattern causing leakage (e.g., "
                    "'Line 45 fits StandardScaler on entire dataset including validation data before splitting')\n"
                    "- If has_leakage=false: Explain why the high metric is reasonable (e.g., "
                    "'Task is simple binary image classification with clear visual patterns, 0.99+ accuracy is achievable')"
                ),
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": (
                    "Confidence level:\n"
                    "- high: Clear evidence of leakage in code (e.g., explicit use of validation data in training)\n"
                    "- medium: Suspicious patterns that likely cause leakage (e.g., unclear split logic)\n"
                    "- low: Task might be simple or code is unclear, uncertain about leakage"
                ),
            },
            "classification": {
                "type": "string",
                "enum": [
                    "hard_leakage",
                    "transductive_contamination",
                    "selection_bias",
                    "warning",
                    "clean",
                ],
                "description": (
                    "Classify the finding precisely. hard_leakage exposes targets, future information, or row identity; "
                    "transductive_contamination fits learned preprocessing on validation/test inputs; selection_bias "
                    "reuses a tuning set for the reported metric; warning is suspicious but unproven; clean has no issue."
                ),
            },
        },
        "required": ["has_leakage", "leakage_reason", "confidence", "classification"],
    },
    description="Detect data leakage issues that lead to unrealistically high validation metrics.",
)


PROTOCOL_SPLIT_REVIEW_SPEC = FunctionSpec(
    name="review_evaluation_protocol",
    json_schema={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["clean", "violation", "uncertain"],
                "description": "clean only when the full data flow proves a trustworthy evaluation protocol",
            },
            "classification": {
                "type": "string",
                "enum": [
                    "clean", "hard_leakage", "transductive_contamination",
                    "selection_bias", "prediction_label_misalignment", "audit_unavailable",
                ],
            },
            "reason": {
                "type": "string",
                "description": "Concrete data-flow explanation with the source variables and failing operation",
            },
            "prediction_source": {
                "type": "string",
                "description": "The actual dataset and index path used to produce the final reported predictions",
            },
            "label_source": {
                "type": "string",
                "description": "The actual dataset and index path used for final metric labels",
            },
            "required_fix": {
                "type": "string",
                "description": "A narrow protocol-only correction that preserves the complete model design",
            },
        },
        "required": [
            "status", "classification", "reason", "prediction_source",
            "label_source", "required_fix",
        ],
    },
    description="Trace data splits and verify final prediction/label provenance before GPU execution.",
)


def _protocol_review_consistency_issue(result: dict) -> str | None:
    """Reject structured decisions that contradict their own remediation."""
    status = str(result.get("status") or "uncertain").strip().lower()
    classification = str(result.get("classification") or "audit_unavailable").strip().lower()
    required_fix = str(result.get("required_fix") or "").strip().lower()
    reason = str(result.get("reason") or "").strip().lower()

    if status == "clean" and classification != "clean":
        return "status=clean requires classification=clean"
    if status == "violation" and classification in {"clean", "audit_unavailable"}:
        return "status=violation requires a concrete violation classification"

    no_fix_markers = (
        "no fix needed",
        "no correction needed",
        "none needed",
        "protocol is clean",
    )
    clean_conclusion_markers = (
        "status should be clean",
        "protocol is actually clean",
        "protocol appears clean",
    )
    if status == "violation" and (
        any(marker in required_fix for marker in no_fix_markers)
        or any(marker in reason for marker in clean_conclusion_markers)
    ):
        return "violation decision says the protocol is clean or needs no fix"
    return None


def run_pre_execution_protocol_review(agent, node: SearchNode, transaction: dict) -> dict:
    """Fail-closed semantic review of the complete final protocol program."""
    try:
        introduction = (
            "You are the final data-protocol gate before an expensive GPU run. Trace the actual "
            "data flow in the complete Python program. Do not trust variable names, comments, "
            "ProtocolProvenanceGuard calls, or claimed sample IDs by themselves; verify what arrays "
            "are actually read, transformed, predicted, selected, and scored. Focus on split integrity.\n\n"
            "You must prove all of the following:\n"
            "1. outer_train and outer_holdout are disjoint and fixed before learned fitting;\n"
            "2. every OOF row is predicted by models/preprocessors that did not fit or early-stop on it;\n"
            "3. model/ensemble selection uses complete outer_train OOF only and is frozen before holdout access;\n"
            "4. final reported predictions are computed from the actual outer_holdout feature rows;\n"
            "5. final metric labels are the same outer_holdout rows in exactly the same order;\n"
            "6. external test/submission predictions are separate and are never sliced, renamed, or "
            "paired with outer_holdout labels as a substitute for holdout predictions.\n\n"
            "A guard call such as record_prediction(... outer_holdout_ids ...) is not proof if the "
            "prediction array came from test data or another partition. If any source-to-metric path "
            "cannot be proven, return status=uncertain. clean requires positive evidence, not absence "
            "of an obvious suspicious keyword. Preserve all model components and training settings in "
            "the required fix."
        )
        prompt = {
            "Introduction": introduction,
            "Task description": agent.task_desc,
            # Prompt compilation recursively formats mappings but protocol metadata
            # also contains booleans and numbers. Render it once as stable text.
            "Protocol plan": json.dumps(
                transaction.get("protocol_plan", {}), sort_keys=True, indent=2
            ),
            "Frozen preservation contract": json.dumps(
                transaction.get("preservation_contract", {}), sort_keys=True, indent=2
            ),
            "Complete candidate program": wrap_code(node.code),
        }
        result = {}
        consistency_issue = None
        for review_attempt in range(1, 4):
            review_prompt = dict(prompt)
            if result:
                review_prompt["Previous self-contradictory review"] = json.dumps(
                    result, sort_keys=True, indent=2
                )
                review_prompt["Correction required"] = (
                    f"The previous structured decision was invalid: {consistency_issue}. "
                    "Re-read the program and return one internally consistent decision. If the "
                    "protocol is clean, use status=clean, classification=clean, and required_fix=none. "
                    "If it is a violation, identify one concrete failing data-flow operation and a "
                    "real correction. Do not call clean behavior a violation."
                )
            response = cast(
                dict,
                query(
                    system_message=review_prompt,
                    user_message=None,
                    func_spec=PROTOCOL_SPLIT_REVIEW_SPEC,
                    model=agent.acfg.feedback.model,
                    temperature=0.0,
                    cfg=agent.cfg,
                ),
            )
            result = {
                "status": str(response.get("status") or "uncertain"),
                "classification": str(response.get("classification") or "audit_unavailable"),
                "reason": str(response.get("reason") or "semantic protocol review returned no reason"),
                "prediction_source": str(response.get("prediction_source") or "unproven"),
                "label_source": str(response.get("label_source") or "unproven"),
                "required_fix": str(response.get("required_fix") or "prove prediction/label alignment"),
                "review_attempts": review_attempt,
            }
            consistency_issue = _protocol_review_consistency_issue(result)
            if consistency_issue is None:
                break
            logger.warning(
                "Self-contradictory protocol review for node %s (attempt %d/3): %s",
                node.id,
                review_attempt,
                consistency_issue,
            )
        if consistency_issue is not None:
            result = {
                "status": "uncertain",
                "classification": "audit_unavailable",
                "reason": (
                    "semantic protocol reviewer returned three self-contradictory decisions: "
                    f"{consistency_issue}"
                ),
                "prediction_source": "unproven",
                "label_source": "unproven",
                "required_fix": (
                    "Do not modify the candidate from this contradictory review. Retry semantic "
                    "protocol review before execution."
                ),
                "review_attempts": 3,
            }
        logger.warning(
            "Pre-execution protocol review for node %s: status=%s classification=%s attempts=%s",
            node.id,
            result["status"],
            result["classification"],
            result.get("review_attempts"),
        )
        return result
    except Exception as exc:
        logger.error("Pre-execution protocol review failed for node %s: %s", node.id, exc)
        return {
            "status": "uncertain",
            "classification": "audit_unavailable",
            "reason": f"semantic protocol review unavailable: {type(exc).__name__}: {exc}",
            "prediction_source": "unproven",
            "label_source": "unproven",
            "required_fix": "Retry the semantic protocol review; do not execute while provenance is unproven.",
        }



def run(agent, node: SearchNode) -> dict:
    try:
        introduction = (
            "You are an expert machine learning engineer specializing in detecting data leakage issues. "
            "You need to analyze the following code to determine if it has data leakage problems that "
            "could lead to unrealistically high validation metrics but poor test performance.\n\n"
            "Common data leakage patterns:\n"
            "1. Using test/validation data during training (e.g., fitting transformers on full dataset)\n"
            "2. Incorrect train/validation split (e.g., temporal leakage in time-series, group leakage)\n"
            "3. Feature engineering using global statistics that include validation/test data\n"
            "4. Data augmentation that duplicates validation samples into training set\n"
            "5. Using target information that wouldn't be available at prediction time\n"
            "6. Cross-fold/OOF embedding leakage: when doing KFold/StratifiedKFold OOF, extracting embeddings "
            "(or any meta-features) for the val/holdout set with a SINGLE fold model (e.g. last_fold_model, "
            "best_fold) whose training fold CONTAINS those holdout rows. The embedding then carries supervised "
            "label info for rows it should not, and leaks into any downstream model fed those embeddings (e.g. "
            "XGBoost/stacker trained on them) -> val metric artificially low. Each row's embedding MUST come "
            "from a fold model that did NOT see that row (proper OOF, same rule as OOF predictions). Watch for "
            "code that reuses one fold's model to extract train+val+test embeddings uniformly.\n"
            "ALSO FLAG: ensemble weight optimization / model selection on the SAME val set used to report the "
            "final metric (select+score same set) -> over-optimistic val, not classic leakage but inflates the "
            "reported number vs test.\n"
            "Classify that case as selection_bias, not hard_leakage. Ordinary early stopping on a validation "
            "set is allowed when a separate untouched holdout is used for the final reported result. Do not "
            "reject sound augmentation, frozen feature extraction, or model capacity merely because a metric "
            "looks unusually strong; identify a concrete data-flow violation.\n"
            "Note: Some tasks are genuinely simple and achieving perfect or near-perfect scores is reasonable. "
            "For example, binary classification with clear visual patterns (like cactus detection) can legitimately "
            "achieve 0.99-1.0 accuracy. Consider the task complexity before declaring leakage."
        )

        prompt = {
            "Introduction": introduction,
            "Task description": agent.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
            "Validation metric": f"{node.metric.value:.4f} (maximize={agent.metric_maximize})",
        }

        response = cast(
            dict,
            query(
                system_message=prompt,
                user_message=None,
                func_spec=DATA_LEAKAGE_CHECK_SPEC,
                model=agent.acfg.feedback.model,
                temperature=agent.acfg.feedback.temp,
                cfg=agent.cfg
            ),
        )

        has_leakage = response["has_leakage"]
        confidence = response["confidence"]
        reason = response["leakage_reason"]
        classification = str(response.get("classification") or ("hard_leakage" if has_leakage else "clean"))

        logger.info(
            f"Data leakage check for node {node.id}: "
            f"has_leakage={has_leakage}, confidence={confidence}"
        )
        logger.info(f"Reason: {reason}")

        return {
            "has_leakage": has_leakage,
            "reason": reason,
            "confidence": confidence,
            "classification": classification,
        }
    except Exception as e:
        logger.error(f"Data leakage check failed for node {node.id}: {e}")
        return {
            "has_leakage": False,
            "reason": f"Leakage check failed due to error: {str(e)}",
            "confidence": "low",
            "classification": "audit_unavailable",
        }
