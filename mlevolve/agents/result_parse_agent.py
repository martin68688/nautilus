import logging
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from engine.executor import ExecutionResult
from utils.metric import MetricValue, WorstMetricValue
from utils.response import wrap_code
from engine.validation import call_validate, _validate_submission_with_retry, validate_submission_content_quality
from agents import data_leakage_agent, leakage_audit
from agents.triggers import should_check_data_leakage

logger = logging.getLogger("MLEvolve")

metric_direction_func_spec = FunctionSpec(
    name="determine_metric_direction",
    json_schema={
        "type": "object",
        "properties": {
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE, RMSE, MAE, loss, error rate), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy, F1 score, AUC, precision, recall, Jaccard score, IoU).",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this metric direction is chosen based on the task's evaluation metric description.",
            },
        },
        "required": [
            "lower_is_better",
            "reasoning",
        ],
    },
    description="Determine whether the evaluation metric should be minimized or maximized based on the task description.",
)


def determine_metric_direction(agent) -> None:
    logger.info("=" * 80)
    logger.info("Starting pre-determination of metric optimization direction...")
    logger.info("=" * 80)

    prompt = f"""You are analyzing a machine learning competition task. Your task is to determine whether the evaluation metric should be minimized or maximized.

    **IMPORTANT: Focus on the EVALUATION section in the task description, which specifies the metric used to score submissions.**

    Task Description:
    {agent.task_desc}

    Based on the evaluation metric mentioned in the task description, determine:
    - If the metric should be MINIMIZED (lower is better), set lower_is_better to TRUE.
    Examples: MSE, RMSE, MAE, Cross-Entropy Loss, Log Loss, Error Rate
    - If the metric should be MAXIMIZED (higher is better), set lower_is_better to FALSE.
    Examples: Accuracy, F1 Score, AUC-ROC, Precision, Recall, Jaccard Score, IoU, mAP

    **Pay special attention to:**
    1. The "Evaluation" or "Metric" section in the task description
    2. Common metric conventions (e.g., accuracy is always maximized, MSE is always minimized)
    3. Whether the metric measures error/loss (minimize) or performance/quality (maximize)

    Provide clear reasoning based on the evaluation metric specified in the task.
    """

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if attempt == 1:
                logger.info(f"Attempt {attempt}/{max_retries} to determine metric direction...")
            else:
                logger.info(f"Retry attempt {attempt}/{max_retries} to determine metric direction...")
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=metric_direction_func_spec,
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    cfg=agent.cfg
                ),
            )

            lower_is_better = response["lower_is_better"]
            agent.metric_maximize = not lower_is_better
            reasoning = response.get("reasoning", "")
            agent.metric_maximize_reasoning = reasoning

            logger.info("=" * 80)
            logger.info("Pre-determination completed successfully:")
            logger.info(f"  - lower_is_better = {lower_is_better}")
            logger.info(f"  - maximize = {agent.metric_maximize}")
            logger.info(f"  - Reasoning: {reasoning}")
            logger.info("=" * 80)
            logger.info(f"All subsequent nodes MUST use maximize={agent.metric_maximize}, otherwise they will be marked as buggy")
            logger.info("=" * 80)
            return

        except Exception as e:
            logger.warning(f"Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                logger.info("Retrying in a moment...")
                time.sleep(1)
            else:
                logger.error("=" * 80)
                logger.error(f"All {max_retries} attempts failed. Last error: {e}")
                logger.error("Using default value maximize=True (assuming higher is better)")
                logger.error("=" * 80)
                agent.metric_maximize = True
                agent.metric_maximize_reasoning = "Default: assuming higher is better (most common case)"


def get_review_func_spec(use_memory: bool) -> FunctionSpec:
    properties = {
        "is_bug": {
            "type": "boolean",
            "description": "true if the output log shows that the execution failed or has some bug, otherwise false. "
                           "Focus only on actual execution errors, exceptions, or crashes.",
        },
        "summary": {
            "type": "string",
            "description": "Provide a concise summary (2-3 sentences) of the execution outcome. "
                           "If successful, describe the key empirical results. "
                           "If failed, describe the error encountered. "
                           "Focus on observations only — do not include suggestions for improvement.",
        },
        "metric": {
            "type": "number",
            "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
        },
        "lower_is_better": {
            "type": "boolean",
            "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
        },
    }
    required = ["is_bug", "summary", "metric", "lower_is_better"]
    if use_memory:
        properties["code_summary"] = {
            "type": "string",
            "description": "Write a summary including the methods used in each stage of the code, such as data preprocessing, feature engineering, model architecture, etc.",
        }
        required.append("code_summary")
    return FunctionSpec(
        name="submit_review",
        json_schema={"type": "object", "properties": properties, "required": required},
        description="Submit a review evaluating the output of the training script.",
    )


def _build_introduction(agent) -> str:
    use_memory = getattr(agent.acfg, "use_global_memory", False)
    intro = (
        "You are a Kaggle grandmaster attending a competition. "
        "You have written code to solve this task and now need to evaluate the output of the code execution. "
        "You should determine if there were any bugs as well as report the empirical findings.\n\n"
        "You MUST respond with a JSON object containing ALL of the following fields:\n"
        "- \"is_bug\": (boolean) true if execution failed or has bugs, false otherwise. Must be a JSON boolean (true/false), NOT a string.\n"
        "- \"summary\": (string) A concise 2-3 sentence summary of the execution outcome.\n"
        "- \"metric\": (number or null) The validation metric value as a raw JSON number (e.g. 0.9995), NOT a string. If failed, use null.\n"
        "- \"lower_is_better\": (boolean) true if the metric should be minimized, false if maximized. Must be a JSON boolean (true/false), NOT a string.\n"
    )
    if use_memory:
        intro += (
            "- \"code_summary\": (string) A concise method summary of the code, covering key parts such as "
            "data preprocessing, feature engineering, model architecture/training, and validation strategy.\n"
        )
    intro += "\nDo NOT omit any field."
    return intro
    


def _check_submission_file(agent, node: SearchNode) -> bool:
    correct_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    if not correct_path.exists():
        wrong_path = agent.cfg.workspace_dir / f"submission_{node.id}.csv"
        if wrong_path.exists():
            correct_path.parent.mkdir(parents=True, exist_ok=True)
            wrong_path.rename(correct_path)
            logger.warning(f" {wrong_path} are moved to {correct_path}")

    return correct_path.exists()


def _save_code_summary(agent, node: SearchNode, response: dict):
    use_memory = getattr(agent.acfg, "use_global_memory", False)
    if not use_memory:
        node.code_summary = None
        return
    if "code_summary" in response and response["code_summary"]:
        node.code_summary = response["code_summary"]
        logger.info(f"Saved code summary for node {node.id}")
    else:
        logger.warning(f"Node {node.id} missing code_summary in response")
        node.code_summary = None


def _determine_buggy(node: SearchNode, response: dict, has_csv_submission: bool):
    failure_reasons = []
    if response["is_bug"]:
        failure_reasons.append("execution error detected")
    if node.exc_type is not None:
        failure_reasons.append(f"exception raised: {node.exc_type}")
    if response["metric"] is None:
        failure_reasons.append("no metric value reported")
    if not has_csv_submission:
        failure_reasons.append("submission file not found")

    node.is_buggy = len(failure_reasons) > 0
    if node.is_buggy:
        logger.warning(f"Node {node.id} marked as buggy: {'; '.join(failure_reasons)}")


def _validate_format_with_retry(agent, node: SearchNode):
    exp_id = agent.cfg.exp_name.split("_")[2]
    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    status, res = _validate_submission_with_retry(
        exp_id=exp_id,
        submission_path=submission_path,
        cfg=agent.cfg,
        max_attempts=2,
        sample_path=None,
    )

    if status:
        if not res['is_valid']:
            logger.warning(f"[validate] node {node.id}: invalid after retry attempts.")
            node.is_valid = False
            node.is_buggy = True
            node._term_out.append(f"\n{res['result']}")
            node.analysis = f"FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\nDetails:\n{res['result']}"
        else:
            _check_content_quality(agent, node, submission_path)
    else:
        logger.error(f"An unexpected error occurred: {res}, skip this stage.")
        logger.info(f"Node {node.id} format validation passed. Now checking content quality...")
        content_valid, content_error = validate_submission_content_quality(
                submission_path=submission_path,
                sample_path=None,
                constant_threshold=0.95,
            )

        if not content_valid:
            _mark_content_quality_failure(node, content_error)
        else:
            logger.info(f"[validate] node {node.id}: valid")
            node.is_valid = True


def _validate_format_simple(agent, node: SearchNode):
    exp_id = agent.cfg.exp_name.split("_")[2]
    submission_path = agent.cfg.workspace_dir / "submission" / f"submission_{node.id}.csv"

    status, res = call_validate(exp_id=exp_id, submission_path=submission_path)
    if status:
        if not res['is_valid']:
            logger.warning(f"[validate] node {node.id}: invalid.")
            node.is_valid = False
            node.is_buggy = True
            node._term_out.append(f"\n{res['result']}")
            node.analysis = f"FORMAT_ERROR: Execution succeeded but submission file failed format validation.\n\nDetails:\n{res['result']}"
        else:
            _check_content_quality(agent, node, submission_path)
    else:
        logger.error(f"An unexpected error occurred: {res}, skip this stage.")


def _check_content_quality(agent, node: SearchNode, submission_path):
    logger.info(f"Node {node.id} format validation passed. Now checking content quality...")
    content_valid, content_error = validate_submission_content_quality(
            submission_path=submission_path,
            sample_path=None,
            constant_threshold=0.95,
        )

    if not content_valid:
        _mark_content_quality_failure(node, content_error)
    else:
        logger.info(f"✅ Node {node.id} passed both format and content quality checks.")
        node.is_valid = True


def _mark_content_quality_failure(node: SearchNode, content_error):
    logger.warning(f"Node {node.id} is marked as buggy due to content quality check failure.")
    node.is_valid = False
    node.is_buggy = True
    error_message = (
        "Submission format is correct, but content quality check FAILED:\n\n"
        f"{content_error}\n\n"
        "🚨 CRITICAL: All predictions must come from actual model inference.\n"
        "You must:\n"
        "1. Load each test sample\n"
        "2. Preprocess it with the same transformations as training\n"
        "3. Run model.predict() / model.forward() on the sample\n"
        "4. Use the model's output as the prediction\n\n"
        "Filling submissions with constants, placeholders, or dummy values is STRICTLY FORBIDDEN."
    )
    node._term_out.append(f"\n{error_message}")
    node.analysis = f"CONTENT_QUALITY_ERROR: This previous solution runs without bugs and has correct format, but failed content quality check.\n\nDetails:\n{content_error}"


def _validate_metric_direction(agent, node: SearchNode, response: dict):
    returned_maximize = not response["lower_is_better"]
    if agent.metric_maximize is not None and returned_maximize != agent.metric_maximize:
        logger.error("=" * 80)
        logger.error(f"METRIC DIRECTION MISMATCH for Node {node.id}!")
        logger.error(f"  - Returned lower_is_better = {response['lower_is_better']} (maximize={returned_maximize})")
        logger.error(f"  - Pre-determined maximize = {agent.metric_maximize}")
        logger.error(f"  - Marking this node as BUGGY, will NOT update top candidates")
        logger.error("=" * 80)
        node.is_buggy = True
        node.metric = WorstMetricValue()
        node.analysis = (
            f"{node.analysis}\n\n[ERROR] Metric direction mismatch detected:\n"
            f"- Returned lower_is_better={response['lower_is_better']} (maximize={returned_maximize})\n"
            f"- Expected maximize={agent.metric_maximize}\n"
            f"- Pre-determination reasoning: {agent.metric_maximize_reasoning or 'N/A'}\n"
            f"This node is marked as buggy and will not be considered for best/top candidates."
        )
    else:
        logger.info(f"Node {node.id} metric direction validated: maximize={agent.metric_maximize}")
        node.metric = MetricValue(
            response["metric"], maximize=agent.metric_maximize
        )


def _persist_leakage_audit(agent, node: SearchNode) -> None:
    try:
        leakage_audit.persist_audit(agent, node)
    except Exception as exc:
        logger.warning("Failed to persist leakage audit for node %s: %s", node.id, exc)
    if agent.global_memory and node.leakage_audit.get("memory_disposition") != "positive_eligible":
        try:
            agent.global_memory.save_leakage_audit(node)
        except Exception as exc:
            logger.warning("Failed to save negative leakage memory for node %s: %s", node.id, exc)


def run_pre_execution_leakage_audit(agent, node: SearchNode) -> bool:
    """Audit reviewed code before GPU execution. Return True when execution is blocked."""
    if not getattr(agent.acfg, "check_data_leakage", False):
        return False
    audit = leakage_audit.audit_code(node.code)
    layer = getattr(agent, "external_skill_memory", None)
    if layer is not None and hasattr(layer, "structural_failure_patterns"):
        patterns = layer.structural_failure_patterns(node.code)
        fresh_issue_codes = {
            str(item.get("issue_code"))
            for item in audit.get("issues", [])
            if item.get("issue_code")
        }
        patterns = [
            pattern for pattern in patterns
            if str(pattern.get("issue_code")) in fresh_issue_codes
        ]
        if patterns:
            audit = leakage_audit.merge_audits(
                node.code,
                audit,
                leakage_audit.failure_pattern_audit(node.code, patterns),
            )
    node.leakage_audit = audit
    node.audit_repair_required = audit.get("status") != "clean"
    if audit.get("status") == "clean" and node.leakage_repair_context:
        node.resolved_issue_codes = [
            str(item.get("issue_code"))
            for item in node.leakage_repair_context.get("issues", [])
            if item.get("issue_code")
        ]
    leakage_audit.persist_audit(agent, node)
    if not audit.get("hard_block"):
        if audit.get("status") != "clean":
            logger.warning(
                "Node %s preflight leakage audit: status=%s issues=%s",
                node.id,
                audit.get("status"),
                [item.get("issue_code") for item in audit.get("issues", [])],
            )
        return False

    audit_text = leakage_audit.format_audit(audit, heading="PRE-EXECUTION LEAKAGE AUDIT")
    node.is_buggy = True
    node.is_valid = False
    node.metric = WorstMetricValue()
    node.analysis = audit_text
    node._term_out = [audit_text]
    node.replay_status = "blocked_by_leakage_audit" if node.replay_source else node.replay_status
    _persist_leakage_audit(agent, node)
    logger.error("Node %s blocked before execution by deterministic leakage audit", node.id)
    return True


def _check_data_leakage(agent, node: SearchNode, response: dict):
    if not (agent.acfg.check_data_leakage and should_check_data_leakage(agent, node)):
        return

    logger.warning(
        f"Node {node.id} running data leakage check (metric={node.metric.value})"
    )

    static_audit = node.leakage_audit or leakage_audit.audit_code(node.code)
    leakage_result = data_leakage_agent.run(agent, node)
    for _ in range(2):
        if str(leakage_result.get("classification")) != "audit_unavailable":
            break
        leakage_result = data_leakage_agent.run(agent, node)
    llm_audit = leakage_audit.llm_result_to_audit(node.code, leakage_result)
    merged_audit = leakage_audit.merge_audits(node.code, static_audit, llm_audit)
    merged_audit["observed_metric"] = response.get("metric")
    node.leakage_audit = merged_audit
    node.audit_repair_required = merged_audit.get("status") != "clean"
    if merged_audit.get("status") == "clean" and node.leakage_repair_context:
        node.resolved_issue_codes = [
            str(item.get("issue_code"))
            for item in node.leakage_repair_context.get("issues", [])
            if item.get("issue_code")
        ]

    if merged_audit.get("hard_block"):
        logger.error(
            "Node %s detected blocking leakage. Marking as buggy and resetting metric. issues=%s",
            node.id,
            [item.get("issue_code") for item in merged_audit.get("issues", [])],
        )
        node.is_buggy = True
        node.is_valid = False
        node.metric = WorstMetricValue()
        node.replay_status = "blocked_by_leakage_audit" if node.replay_source else node.replay_status
    else:
        logger.info(
            "Node %s leakage audit completed: status=%s metric=%s memory=%s",
            node.id,
            merged_audit.get("status"),
            merged_audit.get("metric_disposition"),
            merged_audit.get("memory_disposition"),
        )

    if merged_audit.get("status") != "clean":
        audit_text = leakage_audit.format_audit(merged_audit)
        node.analysis = f"{node.analysis or ''}\n\n{audit_text}".strip()
    _persist_leakage_audit(agent, node)


def _save_to_global_memory(agent, node: SearchNode):
    audit = node.leakage_audit or {}
    if getattr(agent.acfg, "check_data_leakage", False):
        positive_eligible = (
            bool(audit)
            and audit.get("status") == "clean"
            and audit.get("memory_disposition") == "positive_eligible"
        )
    else:
        positive_eligible = not audit or audit.get("memory_disposition") == "positive_eligible"
    if agent.global_memory and positive_eligible and not node.is_buggy and node.metric and node.metric.value is not None:
        try:
            parent_node = node.parent
            agent.global_memory.save_node(node, parent_node)
        except Exception as e:
            logger.warning(f"[AgentSearch] Failed to save node {node.id} to global memory: {e}")


def run(agent, node: SearchNode, exec_result: ExecutionResult) -> SearchNode:
    max_retries = 3
    for retry_idx in range(max_retries):
        try:
            logger.info(f"Agent is parsing execution results for node {node.id}")

            node.absorb_exec_result(exec_result)

            introduction = _build_introduction(agent)
            prompt = {
                "Introduction": introduction,
                "Implementation": wrap_code(node.code),
                "Execution output": wrap_code(node.term_out, lang=""),
            }

            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=get_review_func_spec(getattr(agent.acfg, "use_global_memory", False)),
                    model=agent.acfg.feedback.model,
                    temperature=agent.acfg.feedback.temp,
                    cfg=agent.cfg
                ),
            )

            # Gemini structured output may omit required fields; fill defaults
            response.setdefault("is_bug", True)
            response.setdefault("summary", "No summary returned by model.")
            response.setdefault("metric", None)
            response.setdefault("lower_is_better",
                                not agent.metric_maximize if agent.metric_maximize is not None else False)

            metric_val = response.get("metric")
            if not isinstance(metric_val, (int, float)):
                try:
                    response["metric"] = float(metric_val)
                except (TypeError, ValueError):
                    response["metric"] = None

            for bool_field in ("is_bug", "lower_is_better"):
                v = response.get(bool_field)
                if isinstance(v, str):
                    response[bool_field] = v.strip().lower() not in ("false", "0", "no", "")

            has_csv_submission = _check_submission_file(agent, node)

            node.analysis = response["summary"]
            _save_code_summary(agent, node, response)
            _determine_buggy(node, response, has_csv_submission)

            if not node.is_buggy:
                _validate_format_with_retry(agent, node)

            if node.is_buggy:
                node.metric = WorstMetricValue()
            else:
                _validate_metric_direction(agent, node, response)
                _check_data_leakage(agent, node, response)

            status = "FAIL" if node.is_buggy else "PASS"
            metric_val = node.metric.value if node.metric else None
            logger.info(f"[parse] node {node.id}: {status} | metric={metric_val}")

            _save_to_global_memory(agent, node)

            return node
        except Exception as e:
            logger.warning(f"[parse] tool call failed: {e}")
            continue

    logger.error(f"All {max_retries} parse attempts failed for node {node.id}, marking as buggy")
    node.is_buggy = True
    node.metric = WorstMetricValue()
    node.analysis = "Execution result parsing failed after multiple attempts."
    return node
