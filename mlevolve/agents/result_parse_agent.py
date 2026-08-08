import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import cast

from llm import FunctionSpec, query
from engine.search_node import SearchNode
from engine.executor import ExecutionResult
from utils.metric import MetricValue, WorstMetricValue
from utils.response import wrap_code
from engine.validation import call_validate, _validate_submission_with_retry, validate_submission_content_quality
from agents import data_leakage_agent, leakage_audit, protocol_repair
from agents.result_log_facts import (
    extract_high_confidence_metric as _extract_high_confidence_metric,
    result_parser_conflict as _result_parser_conflict,
    result_parser_facts as _result_parser_facts,
    result_parser_output_view as _result_parser_output_view,
)
from agents.triggers import should_check_data_leakage
from fixed_holdout.mode import bypass_protocol_gates, enabled, train_manifest_path
from fixed_holdout.validation import validate_submission as validate_fixed_submission
from authority.adapters.mlevolve.receipt_bridge import trusted_runtime_metric

logger = logging.getLogger("MLEvolve")


def _emit_first_protocol_valid_candidate(node: SearchNode) -> None:
    """Durably timestamp the first runnable, submission-valid candidate.

    The End2End runner supplies one attempt-scoped path.  ``x`` mode makes the
    event first-writer-wins even if a future configuration executes candidates
    concurrently.  This is measurement only: inability to write the optional
    event must never change candidate validity.
    """

    raw_path = os.environ.get("MLEVOLVE_FIRST_VALID_EVENT_PATH", "").strip()
    if not raw_path:
        return
    path = Path(raw_path)
    event_time_ns = time.time_ns()
    try:
        started_ns = int(os.environ.get("MLEVOLVE_CONDITION_STARTED_AT_NS", "0"))
    except ValueError:
        started_ns = 0
    payload = {
        "schema": "mlevolve_first_protocol_valid_candidate_v1",
        "node_id": str(node.id),
        "event_time_ns": event_time_ns,
        "condition_started_at_ns": started_ns,
        "validation": "label_free_fixed_holdout_submission",
        "event_hash": "",
    }
    payload["event_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key != "event_hash"},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, ensure_ascii=False, indent=2)
            handle.write("\n")
    except FileExistsError:
        return
    except OSError as error:
        logger.warning(
            "Unable to record first protocol-valid candidate %s: %s",
            node.id,
            error,
        )


def _legacy_ast_mode(agent) -> str:
    preflight = getattr(getattr(agent, "acfg", None), "protocol_preflight", None)
    if not bool(getattr(preflight, "enabled", False)):
        return "enforce"
    return str(getattr(preflight, "legacy_ast_mode", "shadow") or "shadow")


def _shadow_only_audit(audit: dict) -> dict:
    """Preserve a legacy observation without allowing it to affect the run."""

    observed = copy.deepcopy(audit)
    shadow = copy.deepcopy(audit)
    shadow.update(
        {
            "status": "clean",
            "hard_block": False,
            "paper_grade_eligible": True,
            "metric_disposition": "accept",
            "memory_disposition": "positive_eligible",
            "execution_disposition": "allow",
            "search_disposition": "normal",
            "rank_eligible": True,
            "repair_required": False,
            "enforcement_mode": "shadow",
            "legacy_shadow_observation": observed,
        }
    )
    return shadow

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


def _method_identity_audit(agent, node: SearchNode) -> dict:
    """Fail closed when a protocol-only repair changes the learned method."""
    context = node.leakage_repair_context or {}
    transaction = node.protocol_repair or {}
    source_id = str(
        transaction.get("source_node_id")
        or context.get("source_node_id")
        or ""
    )
    journal = getattr(agent, "journal", None)
    source = None
    if journal is not None:
        source = next(
            (
                candidate
                for candidate in getattr(journal, "nodes", [])
                if candidate.id == source_id
            ),
            None,
        )
        ancestor = node.parent
        while source is None and ancestor is not None:
            if ancestor.id == source_id or not source_id:
                source = ancestor
                break
            ancestor = ancestor.parent
    issue = None
    identity_value = "source_unavailable"
    if source is None and journal is None:
        # Lightweight unit/integration stubs do not own a Journal. Their
        # existing preservation-contract audit remains authoritative.
        identity_value = "delegated_to_preservation_contract"
    elif source is None:
        issue = {
            "issue_code": "PROTOCOL_REPAIR_METHOD_IDENTITY_UNAVAILABLE",
            "category": "repair_integrity",
            "severity": "critical",
            "line": 0,
            "evidence": f"Frozen source node {source_id or '<missing>'} is unavailable for method comparison.",
            "remediation": "Restore the frozen source node before continuing protocol repair.",
            "execution_disposition": "block",
            "detector": "authority_method_identity_v1",
        }
    repair_surface = None
    verification_hash = ""
    if source is not None:
        from authority.replay_certifier import (
            ProtocolRepairSurface,
            ReplayIdentity,
            verify_protocol_only_patch,
        )

        try:
            authority = getattr(agent, "evaluation_authority", None)
            active_spec = getattr(authority, "active_protocol_spec", None)
            if active_spec is None:
                # Lightweight callers may not own the runtime adapter. Resolve
                # the same immutable config-selected spec rather than treating
                # protocol-repair workflow stages as code-change kinds.
                from authority.adapters.mlevolve.protocol_adapter import build_registry

                _, active_spec = build_registry(agent.cfg)
            repair_surface = ProtocolRepairSurface.from_protocol_spec(active_spec)
            verification = verify_protocol_only_patch(
                source.code,
                node.code,
                repair_surface,
                source_artifact_id=source.id,
                replay_artifact_id=node.id,
            )
            verification_hash = verification.report_hash
            identity_value = verification.identity.value
            if verification.identity != ReplayIdentity.METHOD_PRESERVED:
                issue = {
                    "issue_code": "PROTOCOL_REPAIR_METHOD_CHANGED",
                    "category": "repair_integrity",
                    "severity": "critical",
                    "line": 0,
                    "evidence": (
                        f"Protocol-only repair identity={verification.identity.value}; "
                        f"source_node_id={source.id}. Model family, feature logic, "
                        "training objective, inference pipeline, or protected "
                        "hyperparameters changed."
                    ),
                    "remediation": (
                        "Restore the frozen method and change only "
                        "evaluation-protocol code."
                    ),
                    "execution_disposition": "block",
                    "detector": "authority_method_identity_v1",
                }
        except Exception as error:
            # A malformed or unavailable protocol surface is an auditable
            # fail-closed result, never an exception that escapes the search
            # worker and disappears from Authority taxonomy.
            identity_value = "protocol_surface_resolution_failed"
            issue = {
                "issue_code": "PROTOCOL_REPAIR_SURFACE_INVALID",
                "category": "repair_integrity",
                "severity": "critical",
                "line": 0,
                "evidence": f"{type(error).__name__}: {error}",
                "remediation": (
                    "Restore the active ProtocolSpec and its declared clean-replay "
                    "allowed-change surface before continuing protocol repair."
                ),
                "execution_disposition": "block",
                "detector": "authority_method_identity_v1",
            }
    return {
        "detector_status": "complete",
        "method_identity": identity_value,
        "repair_surface": repair_surface.as_dict() if repair_surface else None,
        "replay_verification_hash": verification_hash,
        "issues": [issue] if issue else [],
    }


def run_pre_execution_leakage_audit(agent, node: SearchNode) -> bool:
    """Audit reviewed code before GPU execution. Return True when execution is blocked."""
    if bypass_protocol_gates(agent.cfg):
        logger.info(
            "Node %s uses terminal fixed-holdout evaluation; internal protocol gates are bypassed",
            node.id,
        )
        return False
    migration_seed = bool(
        node.replay_source
        and node.replay_source.get("requires_full_runtime_migration") is True
        and node.replay_source.get("execution_seed_only") is True
    )
    if migration_seed:
        migration = dict(node.replay_source.get("full_runtime_migration") or {})
        digest = leakage_audit.code_sha256(node.code)
        node.leakage_audit = {
            "schema": "mlevolve_leakage_audit_v2",
            "detector_version": "host_full_runtime_migration_v1",
            "detector_status": "complete",
            "status": "blocked",
            "hard_block": False,
            "code_sha256": digest,
            "issues": [
                {
                    "issue_code": "LEGACY_FULL_RUNTIME_LIFECYCLE_MISSING",
                    "category": "runtime_contract_migration",
                    "severity": "high",
                    "line": 0,
                    "evidence": (
                        "Historical method code lacks required Host full-runtime calls: "
                        + ", ".join(
                            migration.get("missing_full_runtime_coverage") or []
                        )
                    ),
                    "remediation": (
                        "Generate a method-preserving derived child and collect fresh "
                        "Preflight/full-runtime receipts."
                    ),
                    "execution_disposition": "block",
                    "detector": "host_full_runtime_migration_v1",
                }
            ],
            "execution_disposition": "block",
            "search_disposition": "repair_only",
            "memory_disposition": "diagnostic_only",
            "metric_disposition": "reject",
            "rank_eligible": False,
            "paper_grade_eligible": False,
            "repair_required": True,
            "pre_execution_gate_reason": "legacy_full_runtime_migration_seed",
        }
        node.audit_repair_required = True
        node.is_buggy = True
        node.is_valid = False
        node.metric = WorstMetricValue()
        node.analysis = (
            "Historical replay is a legal method seed but cannot execute under "
            "the current Host full-runtime Contract; a derived child is required."
        )
        node._term_out = [node.analysis]
        node.replay_status = "blocked_legacy_full_runtime_seed"
        _persist_leakage_audit(agent, node)
        logger.warning(
            "Node %s blocked before execution as a legacy full-runtime migration seed",
            node.id,
        )
        return True
    verifier_cfg = getattr(agent.cfg, "adoption_verifier", None)
    verifier_enforced = bool(
        verifier_cfg is not None
        and getattr(verifier_cfg, "enabled", False)
        and str(getattr(verifier_cfg, "mode", "shadow") or "shadow").lower()
        == "enforce"
    )
    if (
        not verifier_enforced
        and node.draft_role == "novel_exploration"
        and node.selected_strategy
    ):
        from agents.memory.stage_aware_hybrid_memory import strategy_alignment_for_code

        node.strategy_alignment = strategy_alignment_for_code(node.selected_strategy, node.code)
        if node.strategy_alignment.get("status") != "verified":
            logger.warning(
                "Node %s does not fully implement selected method_family=%s; status=%s. "
                "Execution may continue, but certified ranking is disabled.",
                node.id,
                node.strategy_alignment.get("method_family"),
                node.strategy_alignment.get("status"),
            )
    if not getattr(agent.acfg, "check_data_leakage", False):
        return False
    static_audit = leakage_audit.audit_code(node.code)
    preservation_audit = None
    audit = static_audit
    if node.leakage_repair_context:
        preservation_audit = leakage_audit.audit_repair_preservation(
            node.code,
            node.leakage_repair_context.get("preservation_contract", {}),
        )
        preservation_audit = leakage_audit.merge_audits(
            node.code,
            preservation_audit,
            _method_identity_audit(agent, node),
        )
        audit = leakage_audit.merge_audits(
            node.code,
            audit,
            preservation_audit,
        )
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

    if _legacy_ast_mode(agent) == "shadow":
        node.leakage_audit = _shadow_only_audit(audit)
        node.audit_repair_required = False
        # A historical replay may carry an old enforcing repair transaction.
        # Host mode re-admits the immutable code through its own Contract and
        # Receipt, so the legacy transaction is diagnostic-only as well.
        node.protocol_repair = {}
        node.leakage_repair_context = {}
        _persist_leakage_audit(agent, node)
        observed = node.leakage_audit["legacy_shadow_observation"]
        logger.warning(
            "Node %s legacy pre-execution audit is shadow-only: observed_status=%s issues=%s",
            node.id,
            observed.get("status"),
            [item.get("issue_code") for item in observed.get("issues", [])],
        )
        return False

    # Protocol repair is a transaction, not a normal debug retry.  Every
    # intermediate stage is journaled and requeued but cannot consume GPU.
    # The final stage must satisfy static leakage, preservation, and its own
    # protocol contract before execution is allowed.
    protocol_tx = node.protocol_repair or {}
    if protocol_repair.is_active(protocol_tx):
        stage = protocol_repair.current_stage(protocol_tx)
        stage_audit = protocol_repair.audit_stage(node.code, protocol_tx)
        scope_gate = protocol_repair.stage_scope_gate(static_audit, protocol_tx)
        preservation_clean = (
            preservation_audit is None
            or preservation_audit.get("status") == "clean"
        )
        final_stage = stage == "final_holdout"
        stage_passed = bool(
            stage_audit.get("status") == "clean"
            and scope_gate.get("status") == "clean"
            and preservation_clean
        )
        semantic_protocol_review = None
        if final_stage and stage_passed and audit.get("status") == "clean":
            semantic_protocol_review = data_leakage_agent.run_pre_execution_protocol_review(
                agent, node, protocol_tx
            )
            if semantic_protocol_review.get("status") != "clean":
                stage_passed = False
                stage_audit = dict(stage_audit)
                stage_audit["status"] = "blocked"
                stage_audit["issues"] = list(stage_audit.get("issues") or []) + [{
                    "issue_code": "PROTOCOL_FINAL_SEMANTIC_SPLIT_NOT_CLEAN",
                    "category": "protocol_repair_stage",
                    "severity": "critical",
                    "line": 0,
                    "evidence": (
                        f"{semantic_protocol_review.get('reason')} "
                        f"prediction_source={semantic_protocol_review.get('prediction_source')} "
                        f"label_source={semantic_protocol_review.get('label_source')}"
                    ),
                    "remediation": semantic_protocol_review.get("required_fix"),
                    "execution_disposition": "block",
                    "detector": "protocol_semantic_agent_v1",
                }]
        if final_stage and audit.get("status") != "clean":
            stage_passed = False
            stage_audit = dict(stage_audit)
            stage_audit["status"] = "blocked"
            stage_audit["issues"] = list(stage_audit.get("issues") or []) + [{
                "issue_code": "PROTOCOL_FINAL_STATIC_AUDIT_NOT_CLEAN",
                "category": "protocol_repair_stage",
                "severity": "critical",
                "line": 0,
                "evidence": f"Final protocol stage still has leakage/preservation status={audit.get('status')}",
                "remediation": "Resolve every static leakage and preservation issue before final execution.",
                "execution_disposition": "block",
                "detector": "protocol_stage_v1",
            }]
        feedback_issues = list(stage_audit.get("issues") or [])
        blocking_codes = set(scope_gate.get("blocking_issue_codes") or [])
        feedback_issues.extend(
            issue
            for issue in static_audit.get("issues", [])
            if issue.get("issue_code") in blocking_codes
        )
        if not preservation_clean and preservation_audit is not None:
            feedback_issues.extend(preservation_audit.get("issues") or [])
        stage_result = {
            **stage_audit,
            "status": "clean" if stage_passed else "blocked",
            "issues": feedback_issues,
        }
        node.protocol_repair = protocol_repair.apply_stage_result(
            protocol_tx,
            stage_result,
            node.id,
        )
        audit["protocol_stage_audit"] = stage_audit
        audit["protocol_scope_gate"] = scope_gate
        audit["protocol_preservation_clean"] = preservation_clean
        audit["protocol_semantic_review"] = semantic_protocol_review
        audit["protocol_transaction_id"] = protocol_tx.get("transaction_id")
        audit["protocol_stage"] = stage

        if final_stage and stage_passed:
            audit["runtime_protocol_required"] = bool(
                protocol_tx.get("require_runtime_provenance", True)
            )
            node.leakage_audit = audit
            node.audit_repair_required = False
            node.replay_status = "staged_protocol_repair_clean_pending_execution"
            leakage_audit.persist_audit(agent, node)
            return False

        if stage_passed:
            audit["status"] = "protocol_stage_complete"
            audit["completed_protocol_stage"] = stage
            audit["next_protocol_stage"] = protocol_repair.current_stage(node.protocol_repair)
        else:
            audit = leakage_audit.merge_audits(node.code, audit, stage_audit)
            audit["protocol_stage_audit"] = stage_audit
            audit["protocol_transaction_id"] = protocol_tx.get("transaction_id")
            audit["protocol_stage"] = stage
            audit["protocol_scope_gate"] = scope_gate
            audit["protocol_preservation_clean"] = preservation_clean
        audit["execution_disposition"] = "block"
        audit["search_disposition"] = "repair_only"
        audit["memory_disposition"] = "negative_only"
        audit["metric_disposition"] = "reject"
        audit["rank_eligible"] = False
        audit["paper_grade_eligible"] = False
        audit["repair_required"] = node.protocol_repair.get("state") != "exhausted"
        node.leakage_audit = audit
        node.audit_repair_required = audit["repair_required"]
        if node.protocol_repair.get("state") == "exhausted":
            node.is_terminal = True
        node.is_buggy = True
        node.is_valid = False
        node.metric = WorstMetricValue()
        node.analysis = (
            f"STAGED PROTOCOL REPAIR: stage={stage} "
            f"status={'passed' if stage_passed else 'failed'} "
            f"next={protocol_repair.current_stage(node.protocol_repair)}"
        )
        node._term_out = [node.analysis]
        node.replay_status = (
            "staged_protocol_repair_exhausted"
            if node.protocol_repair.get("state") == "exhausted"
            else "staged_protocol_repair_stage_complete"
            if stage_passed
            else "staged_protocol_repair_stage_blocked"
        )
        _persist_leakage_audit(agent, node)
        logger.warning(
            "Node %s protocol stage %s %s before GPU execution",
            node.id, stage, "passed" if stage_passed else "failed",
        )
        return True

    replay_repair_child = bool(
        node.replay_source
        and (
            node.replay_source.get("requires_repair") is True
            or node.replay_source.get("requires_full_runtime_migration") is True
        )
        and node.leakage_repair_context
    )
    repair_seed_only = bool(
        node.replay_source
        and node.replay_source.get("requires_repair") is True
        and node.replay_source.get(
            "repair_seed_only",
            not bool(node.leakage_repair_context),
        ) is True
    )
    if audit.get("status") == "clean" and node.leakage_repair_context:
        node.resolved_issue_codes = [
            str(item.get("issue_code"))
            for item in node.leakage_repair_context.get("issues", [])
            if item.get("issue_code")
        ]
        if replay_repair_child:
            node.replay_status = (
                "derived_full_runtime_candidate_clean_pending_execution"
                if node.replay_source.get("requires_full_runtime_migration") is True
                else "mandatory_audit_repair_clean_pending_execution"
            )
    leakage_audit.persist_audit(agent, node)
    if audit.get("status") == "clean" and not repair_seed_only:
        return False

    audit["execution_disposition"] = "block"
    audit["search_disposition"] = "repair_only"
    if not audit.get("hard_block"):
        audit["memory_disposition"] = "negative_only"
    audit["metric_disposition"] = "reject"
    audit["rank_eligible"] = False
    audit["paper_grade_eligible"] = False
    audit["repair_required"] = True
    audit["pre_execution_gate_reason"] = (
        "immutable_repair_seed" if repair_seed_only else "audit_status_not_clean"
    )
    if repair_seed_only:
        audit["repair_seed_execution_blocked"] = True

    audit_text = leakage_audit.format_audit(audit, heading="PRE-EXECUTION LEAKAGE AUDIT")
    node.is_buggy = True
    node.is_valid = False
    node.metric = WorstMetricValue()
    node.analysis = audit_text
    node._term_out = [audit_text]
    if repair_seed_only:
        node.replay_status = "blocked_exact_source_repair_seed"
    elif replay_repair_child:
        node.replay_status = "mandatory_audit_repair_blocked"
    elif node.replay_source:
        node.replay_status = "blocked_by_leakage_audit"
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
    if (node.protocol_repair or {}).get("state") == "ready_for_execution":
        runtime_result = protocol_repair.runtime_provenance_audit(
            "".join(node._term_out or []),
            node.protocol_repair,
        )
        runtime_audit = protocol_repair.runtime_result_as_audit(node.code, runtime_result)
        merged_audit = leakage_audit.merge_audits(node.code, merged_audit, runtime_audit)
        merged_audit["runtime_protocol_provenance"] = runtime_result
        if runtime_result.get("status") == "clean" and merged_audit.get("status") == "clean":
            node.protocol_repair["state"] = "completed"
            node.protocol_repair["runtime_provenance"] = runtime_result
            node.replay_status = "staged_protocol_repair_executed_clean"
        else:
            node.protocol_repair = protocol_repair.rollback_final_runtime_failure(
                node.protocol_repair,
                node.id,
                str(runtime_result.get("reason") or "runtime provenance failed"),
            )
            node.replay_status = "staged_protocol_repair_runtime_blocked"
    merged_audit["observed_metric"] = response.get("metric")
    if _legacy_ast_mode(agent) == "shadow":
        node.leakage_audit = _shadow_only_audit(merged_audit)
        node.audit_repair_required = False
        _persist_leakage_audit(agent, node)
        logger.warning(
            "Node %s legacy post-execution leakage review is shadow-only: observed_status=%s",
            node.id,
            merged_audit.get("status"),
        )
        return
    node.leakage_audit = merged_audit
    node.audit_repair_required = merged_audit.get("status") != "clean"
    if merged_audit.get("status") == "clean" and node.leakage_repair_context:
        node.resolved_issue_codes = [
            str(item.get("issue_code"))
            for item in node.leakage_repair_context.get("issues", [])
            if item.get("issue_code")
        ]
        if (
            node.protocol_repair
            and node.protocol_repair.get("state") == "completed"
        ):
            node.replay_status = "staged_protocol_repair_executed_clean"
        elif (
            node.replay_source
            and (
                node.replay_source.get("requires_repair") is True
                or node.replay_source.get("requires_full_runtime_migration") is True
            )
            and node.leakage_repair_context
        ):
            node.replay_status = (
                "derived_full_runtime_candidate_executed_clean"
                if node.replay_source.get("requires_full_runtime_migration") is True
                else "mandatory_audit_repair_executed_clean"
            )

    if merged_audit.get("hard_block"):
        logger.error(
            "Node %s detected blocking leakage. Marking as buggy and resetting metric. issues=%s",
            node.id,
            [item.get("issue_code") for item in merged_audit.get("issues", [])],
        )
        node.is_buggy = True
        node.is_valid = False
        node.metric = WorstMetricValue()
        if node.protocol_repair:
            node.replay_status = "staged_protocol_repair_runtime_blocked"
        elif (
            node.replay_source
            and node.replay_source.get("requires_repair") is True
            and node.leakage_repair_context
        ):
            node.replay_status = "mandatory_audit_repair_blocked"
        elif node.replay_source:
            node.replay_status = "blocked_by_leakage_audit"
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
    if enabled(agent.cfg):
        logger.info(
            "Node %s is not written to positive memory before terminal fixed-holdout scoring",
            node.id,
        )
        return
    audit = node.leakage_audit or {}
    if getattr(agent.acfg, "check_data_leakage", False):
        positive_eligible = (
            bool(audit)
            and audit.get("status") == "clean"
            and audit.get("memory_disposition") == "positive_eligible"
        )
    else:
        positive_eligible = not audit or audit.get("memory_disposition") == "positive_eligible"
    legacy_allowed = bool(
        positive_eligible
        and not node.is_buggy
        and node.metric
        and node.metric.value is not None
    )
    from authority.adapters.mlevolve.memory_gate import authorize_positive_memory_write
    authority_allowed = authorize_positive_memory_write(
        agent,
        node,
        legacy_allowed=legacy_allowed,
        component="agents.result_parse_agent._save_to_global_memory",
    )
    if authority_allowed:
        if agent.global_memory:
            try:
                parent_node = node.parent
                agent.global_memory.save_node(node, parent_node)
            except Exception as e:
                logger.warning(f"[AgentSearch] Failed to save node {node.id} to global memory: {e}")
        try:
            adapter = getattr(agent, "evaluation_authority", None)
            append_overlay = getattr(
                adapter, "append_authorized_memory_overlay", None
            )
            if callable(append_overlay):
                append_overlay(node)
        except Exception as e:
            logger.warning(
                "[AgentSearch] Failed to append node %s to Session Overlay: %s",
                node.id,
                e,
            )


def run(agent, node: SearchNode, exec_result: ExecutionResult) -> SearchNode:
    max_retries = 3
    for retry_idx in range(max_retries):
        try:
            logger.info(f"Agent is parsing execution results for node {node.id}")

            node.absorb_exec_result(exec_result)

            signed_metric = None
            authority = getattr(agent, "evaluation_authority", None)
            active_protocol = getattr(authority, "active_protocol", None)
            if active_protocol is not None:
                signed_metric = trusted_runtime_metric(
                    node,
                    active_protocol,
                    task_id=str(getattr(agent.cfg, "exp_id", "") or ""),
                )

            has_csv_submission = _check_submission_file(agent, node)
            parser_view = _result_parser_output_view(node)
            parser_facts = _result_parser_facts(
                node, has_csv_submission, parser_view
            )
            introduction = (
                _build_introduction(agent)
                + "\n\nObjective executor facts are supplied separately. Treat a normal "
                "process exit and an existing submission as authoritative facts. Do not "
                "claim timeout, interruption, or missing completion merely because the "
                "displayed log was compacted."
            )
            prompt = {
                "Introduction": introduction,
                "Implementation": wrap_code(node.code),
                "Objective executor facts": json.dumps(
                    parser_facts, ensure_ascii=False, sort_keys=True
                ),
                "Execution output": wrap_code(parser_view, lang=""),
            }

            def query_result_review(extra_instruction: str = "") -> dict:
                review_prompt = dict(prompt)
                if extra_instruction:
                    review_prompt["Reconciliation required"] = extra_instruction
                return cast(
                    dict,
                    query(
                        system_message=review_prompt,
                        user_message=None,
                        func_spec=get_review_func_spec(
                            getattr(agent.acfg, "use_global_memory", False)
                        ),
                        model=agent.acfg.feedback.model,
                        temperature=agent.acfg.feedback.temp,
                        cfg=agent.cfg,
                    ),
                )

            def normalize_response(value: dict) -> dict:
                value.setdefault("is_bug", True)
                value.setdefault("summary", "No summary returned by model.")
                value.setdefault("metric", None)
                value.setdefault(
                    "lower_is_better",
                    not agent.metric_maximize
                    if agent.metric_maximize is not None
                    else False,
                )
                for bool_field in ("is_bug", "lower_is_better"):
                    raw_value = value.get(bool_field)
                    if isinstance(raw_value, str):
                        value[bool_field] = raw_value.strip().lower() not in (
                            "false",
                            "0",
                            "no",
                            "",
                        )
                return value

            response = normalize_response(query_result_review())
            initial_conflict = _result_parser_conflict(response, parser_facts)
            parser_calls = 1
            if initial_conflict:
                parser_calls += 1
                response = normalize_response(
                    query_result_review(
                        "The previous review conflicted with objective executor facts: "
                        f"{initial_conflict}. Re-read the salient full-log lines and return "
                        "the actual completed result."
                    )
                )

            remaining_conflict = _result_parser_conflict(response, parser_facts)
            fallback_metric_used = False
            false_failure_overridden = False
            submission_alignment_required = bool(
                getattr(
                    getattr(agent.cfg, "run_identity", None),
                    "require_submission_aligned_internal_metric",
                    False,
                )
            )
            if (
                response.get("metric") is None
                and parser_facts.get("high_confidence_self_reported_metric")
                is not None
                and parser_facts.get("process_exited_normally")
                and parser_facts.get("submission_file_exists")
            ):
                response["metric"] = parser_facts[
                    "high_confidence_self_reported_metric"
                ]
                fallback_metric_used = True
            remaining_conflict = _result_parser_conflict(response, parser_facts)
            if remaining_conflict == "agent_failure_claim_conflicts_with_clean_process_exit":
                response["is_bug"] = False
                false_failure_overridden = True
                remaining_conflict = _result_parser_conflict(
                    response, parser_facts
                )
            aligned_metric = parser_facts.get("submission_aligned_metric")
            legacy_exact_replay = bool(
                node.stage == "draft"
                and node.draft_role == "memory_reproduction"
                and node.replay_source
            )
            if (
                aligned_metric is not None
                and parser_facts.get("process_exited_normally")
                and parser_facts.get("submission_file_exists")
            ):
                response["metric"] = float(aligned_metric)
                submission_alignment_status = "verified_marker"
            elif submission_alignment_required and not legacy_exact_replay:
                response["metric"] = None
                response["is_bug"] = True
                response["summary"] = (
                    str(response.get("summary") or "")
                    + " SUBMISSION_METRIC_ALIGNMENT_MISSING: the run did not emit "
                    "the required submission-aligned metric and variant marker."
                ).strip()
                submission_alignment_status = "required_marker_missing"
            elif submission_alignment_required:
                submission_alignment_status = "legacy_exact_replay_unverified"
            else:
                submission_alignment_status = "not_required"
            observation = getattr(node, "protocol_observation", None)
            if not isinstance(observation, dict):
                observation = {}
                node.protocol_observation = observation
            observation["result_parser_reconciliation"] = {
                "schema": "mlevolve_result_parser_reconciliation_v1",
                "facts": parser_facts,
                "agent_calls": parser_calls,
                "initial_conflict": initial_conflict,
                "remaining_conflict": remaining_conflict,
                "fallback_metric_used": fallback_metric_used,
                "false_failure_overridden": false_failure_overridden,
            }
            observation["submission_metric_alignment"] = {
                "schema": "mlevolve_submission_metric_alignment_v1",
                "required": submission_alignment_required,
                "status": submission_alignment_status,
                "metric": aligned_metric,
                "submission_variant": str(
                    parser_facts.get("submission_variant") or ""
                ),
                "marker_line": str(
                    parser_facts.get("submission_aligned_metric_line") or ""
                ),
            }

            if signed_metric is not None:
                llm_metric = response.get("metric")
                host_metric = float(signed_metric["metric_value"])
                if (
                    not isinstance(llm_metric, (int, float))
                    or not math.isclose(
                        float(llm_metric),
                        host_metric,
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
                ):
                    logger.warning(
                        "Signed Host evaluator metric overrides result-parser "
                        "value for node %s: parser=%r host=%s",
                        node.id,
                        llm_metric,
                        host_metric,
                    )
                response["metric"] = host_metric
                response["is_bug"] = False
                direction = str(
                    signed_metric.get("metric_direction") or ""
                ).lower()
                if direction in {"minimize", "maximize"}:
                    response["lower_is_better"] = direction == "minimize"
                observation = getattr(node, "protocol_observation", None)
                if not isinstance(observation, dict):
                    observation = {}
                    node.protocol_observation = observation
                observation["trusted_evaluator_metric"] = {
                    "metric_name": signed_metric.get("metric_name"),
                    "metric_value": host_metric,
                    "metric_direction": direction,
                    "source": "signed_host_runtime_receipt",
                }

            metric_val = response.get("metric")
            if not isinstance(metric_val, (int, float)):
                try:
                    response["metric"] = float(metric_val)
                except (TypeError, ValueError):
                    response["metric"] = None

            node.analysis = response["summary"]
            _save_code_summary(agent, node, response)
            _determine_buggy(node, response, has_csv_submission)

            if (
                node.is_buggy
                and (node.protocol_repair or {}).get("state") == "ready_for_execution"
            ):
                exc_message = ""
                if isinstance(node.exc_info, dict):
                    exc_message = str(node.exc_info.get("message") or "")
                elif node.exc_info:
                    exc_message = str(node.exc_info)
                traceback_tail = "".join(node._term_out or []).splitlines()[-8:]
                runtime_reason = "; ".join(
                    part for part in [
                        f"{node.exc_type}: {exc_message}" if node.exc_type else exc_message,
                        " | ".join(traceback_tail),
                    ] if part
                ) or "final protocol program failed before clean runtime provenance"
                node.protocol_repair = protocol_repair.rollback_final_runtime_failure(
                    node.protocol_repair,
                    node.id,
                    runtime_reason,
                )
                node.audit_repair_required = node.protocol_repair.get("state") != "exhausted"
                node.replay_status = (
                    "staged_protocol_repair_exhausted"
                    if node.protocol_repair.get("state") == "exhausted"
                    else "staged_protocol_repair_runtime_blocked"
                )

            if not node.is_buggy:
                if enabled(agent.cfg):
                    submission_path = (
                        agent.cfg.workspace_dir
                        / "submission"
                        / f"submission_{node.id}.csv"
                    )
                    is_valid, reason = validate_fixed_submission(
                        train_manifest_path(agent.cfg),
                        submission_path,
                    )
                    if is_valid:
                        node.is_valid = True
                        logger.info(
                            "Node %s passed label-free fixed-holdout submission validation",
                            node.id,
                        )
                    else:
                        node.is_valid = False
                        node.is_buggy = True
                        node.analysis = (
                            f"FIXED_HOLDOUT_FORMAT_ERROR: {reason}"
                        )
                        node._term_out.append(f"\n{node.analysis}")
                else:
                    _validate_format_with_retry(agent, node)

            if node.is_buggy:
                node.metric = WorstMetricValue()
            else:
                _validate_metric_direction(agent, node, response)
                _check_data_leakage(agent, node, response)

            status = "FAIL" if node.is_buggy else "PASS"
            metric_val = node.metric.value if node.metric else None
            logger.info(f"[parse] node {node.id}: {status} | metric={metric_val}")

            adapter = getattr(agent, "evaluation_authority", None)
            finalize_actuation = getattr(
                adapter, "finalize_production_actuation", None
            )
            if callable(finalize_actuation):
                try:
                    finalize_actuation(node)
                except Exception as error:
                    # Actuation/writeback is fail-closed and must not turn a
                    # completed candidate into fabricated provenance.
                    logger.warning(
                        "Production actuation finalization failed closed for node %s: %s",
                        node.id,
                        type(error).__name__,
                    )

            if node.is_valid is True and not node.is_buggy:
                _emit_first_protocol_valid_candidate(node)

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
