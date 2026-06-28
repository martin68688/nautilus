"""Data Leakage Agent: LLM-based data leakage check for node code."""

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
                    "embedding then carries supervised label info for rows it shouldn't, leaking into any "
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
        },
        "required": ["has_leakage", "leakage_reason", "confidence"],
    },
    description="Detect data leakage issues that lead to unrealistically high validation metrics.",
)



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

        logger.info(
            f"Data leakage check for node {node.id}: "
            f"has_leakage={has_leakage}, confidence={confidence}"
        )
        logger.info(f"Reason: {reason}")

        return {
            "has_leakage": has_leakage,
            "reason": reason,
            "confidence": confidence,
        }
    except Exception as e:
        logger.error(f"Data leakage check failed for node {node.id}: {e}")
        return {
            "has_leakage": False,
            "reason": f"Leakage check failed due to error: {str(e)}",
            "confidence": "low",
        }
