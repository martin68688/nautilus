"""Code Review Agent: LLM-based code review and fix for node code."""

import ast
import logging
import time
from typing import cast

from llm import FunctionSpec, query
from engine.candidate_execution_contract import (
    audit_candidate_code,
    candidate_execution_contract_from_cfg,
)
from engine.search_node import SearchNode
from authority.protocol_execution_contract import read_contract_artifact
from protocol_runtime.preflight import static_compatibility_check
from agents.prompts.validation_template_prompts import get_code_review_prompt
from agents.prompts import (
    get_impl_guideline_from_agent,
    get_internet_clarification,
    host_protocol_preflight_enabled,
)

from agents.coder.diff_coder import SearchReplacePatcher

logger = logging.getLogger("MLEvolve")

_METRIC_NAME_PARTS = {
    "accuracy", "auc", "f1", "logloss", "loss", "mae", "metric", "mse",
    "ndcg", "reward", "rmse", "score",
}


def _metric_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            name = child.id.lower()
        elif isinstance(child, ast.Attribute):
            name = child.attr.lower()
        else:
            continue
        normalized = name.replace("-", "_")
        if any(part in normalized for part in _METRIC_NAME_PARTS):
            names.add(normalized)
    return names


def _metric_comparison(node: ast.AST) -> bool:
    """Return whether a condition compares two distinct metric values."""

    return any(
        isinstance(child, ast.Compare) and len(_metric_names(child)) >= 2
        for child in ast.walk(node)
    )


def _contains_termination(statements: list[ast.stmt]) -> bool:
    for statement in statements:
        for child in ast.walk(statement):
            if isinstance(child, ast.Raise):
                return True
            if isinstance(child, ast.Call):
                target = child.func
                if isinstance(target, ast.Name) and target.id in {"exit", "quit"}:
                    return True
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "exit"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "sys"
                ):
                    return True
    return False


def _metric_falsification_fallback_audit(code: str) -> dict:
    """Reject turning an empirically worse variant into a runtime failure.

    Candidate improvements are hypotheses.  A worse validation metric should
    select the already-computed baseline predictions, not abort before writing
    a submission.  This audit intentionally targets only comparisons between
    two named metric values so ordinary shape/finite/range assertions remain
    valid program invariants.
    """

    violations: list[str] = []
    try:
        tree = ast.parse(str(code or ""))
    except SyntaxError:
        return {
            "schema": "mlevolve_metric_falsification_fallback_audit_v1",
            "valid": True,
            "violations": [],
        }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert) and _metric_comparison(node.test):
            violations.append(
                f"metric_improvement_assert:line={node.lineno}: empirical non-improvement "
                "must select baseline predictions and continue to submission"
            )
        elif (
            isinstance(node, ast.If)
            and _metric_comparison(node.test)
            and _contains_termination([*node.body, *node.orelse])
        ):
            violations.append(
                f"metric_gated_termination:line={node.lineno}: empirical non-improvement "
                "must select baseline predictions and continue to submission"
            )
    return {
        "schema": "mlevolve_metric_falsification_fallback_audit_v1",
        "valid": not violations,
        "violations": violations,
    }

CODE_REVIEW_SPEC = FunctionSpec(
    name="submit_code_review",
    json_schema={
        "type": "object",
        "properties": {
            "needs_revision": {
                "type": "boolean",
                "description": (
                    "true if the code has issues that must be fixed "
                    "(metric mismatch, data leakage, or missing packages), "
                    "false if the code is correct."
                )
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "CONCISE explanation in EXACTLY 2-4 sentences. Explain: "
                    "(1) what issues were found, (2) why they matter, (3) what will be fixed. "
                    "DO NOT write detailed analysis or step-by-step checks - keep it brief."
                )
            },
            "revised_code": {
                # A clean review is contractually represented by JSON null or
                # by omitting this optional field.  The local OpenAI-compatible
                # schema validator must accept the same representation that
                # the prompt explicitly requires.
                "type": ["string", "null"],
                "description": (
                    "ONLY if needs_revision=true: Provide targeted fixes using SEARCH/REPLACE diff format.\n\n"
                    "**REQUIRED FORMAT** (use this for each fix):\n"
                    "<<<<<<< SEARCH\n"
                    "[exact code to find - copy verbatim with exact indentation]\n"
                    "=======\n"
                    "[corrected code]\n"
                    ">>>>>>> REPLACE\n\n"
                    "**CRITICAL**: \n"
                    "- SEARCH block must match original code EXACTLY (character-by-character, including all spaces/tabs)\n"
                    "- Only include the specific buggy lines that need fixing\n"
                    "- Can provide multiple SEARCH/REPLACE blocks for different bugs\n"
                    "- Do NOT output complete code - only diff blocks\n"
                    "- Do NOT wrap output in markdown code fences (``` or ```python) - output raw diff only\n\n"
                    "If needs_revision=false: MUST be null (DO NOT output code)."
                )
            }
        },
        "required": ["needs_revision", "reasoning"]
    },
    description="Submit code review for search node solution."
)


def _deterministic_contract_audit(agent, code: str) -> dict | None:
    """Run the same machine-checkable Host feasibility audit before review."""

    try:
        report: dict | None = None
        contract = candidate_execution_contract_from_cfg(agent.cfg)
        if contract is not None:
            report = audit_candidate_code(code, contract)
        preflight = getattr(
            getattr(agent, "acfg", None), "protocol_preflight", None
        )
        contract_path = str(getattr(preflight, "contract_path", "") or "")
        if (
            report is None
            and preflight is not None
            and getattr(preflight, "enabled", False)
            and contract_path
        ):
            static_report = static_compatibility_check(
                code, read_contract_artifact(contract_path)
            )
            issues = list(static_report.get("violations") or [])
            issues.extend(
                "missing_full_runtime_coverage:" + str(value)
                for value in static_report.get(
                    "missing_full_runtime_coverage", []
                )
            )
            report = {
                "valid": not bool(issues),
                "violations": issues,
                "code_sha256": static_report.get("code_sha256"),
                "contract_hash": static_report.get("contract_hash"),
            }
        if report is None:
            report = {
                "schema": "mlevolve_code_review_host_audit_v1",
                "valid": True,
                "violations": [],
            }
        fallback = _metric_falsification_fallback_audit(code)
        report = dict(report)
        report["violations"] = [
            *list(report.get("violations") or []),
            *list(fallback.get("violations") or []),
        ]
        report["metric_falsification_fallback_audit"] = fallback
        report["valid"] = not bool(report["violations"])
        return report
    except (AttributeError, TypeError, ValueError) as error:
        logger.warning("Deterministic Host Contract review unavailable: %s", error)
        return None


def run(agent, node: SearchNode) -> str:
    logger.debug(f"[review] node {node.id}")

    prompt = get_code_review_prompt(
        task_desc=agent.task_desc,
        code=node.code,
    )
    if "Instructions" not in prompt:
        prompt["Instructions"] = {}
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    deterministic_audit = _deterministic_contract_audit(agent, node.code)
    if deterministic_audit and deterministic_audit.get("violations"):
        prompt["Instructions"][
            "DETERMINISTIC HOST CONTRACT AUDIT - HIGHEST PRIORITY"
        ] = [
            "The Host static checker has already rejected this exact code. Set needs_revision=true and repair every listed violation; do not approve unchanged code.",
            "Exact violations: "
            + "; ".join(deterministic_audit["violations"]),
            "Preserve the method, model family, metric, folds, ensembles, epochs, and training design. Repair only the listed leakage or Host-lifecycle violations.",
            "Do not evade the checker through variable renaming; change the actual violating operation.",
            "For metric_improvement_assert or metric_gated_termination, preserve the baseline validation/test predictions. Select the proposed variant only when it is genuinely better under the metric direction; otherwise select the baseline predictions, report the baseline score/variant, and still write the complete submission. Do not merely delete the assertion while continuing with a worse variant.",
        ]
    if not host_protocol_preflight_enabled(agent):
        internet_clarification = get_internet_clarification(
            getattr(agent.cfg, "pretrain_model_dir", "")
        )
        prompt["Instructions"]["Implementation guideline"].extend(
            internet_clarification
        )

    use_diff_for_review = agent.acfg.use_diff_mode
    max_retries = 3

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logger.info(f"Code review retry attempt {attempt + 1}/{max_retries} for node {node.id}")
                time.sleep(5)

            review_response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=CODE_REVIEW_SPEC,
                    model=agent.acfg.code.model,
                    temperature=agent.acfg.code.temp,
                    cfg=agent.cfg
                ),
            )

            needs_revision = review_response.get("needs_revision", False)
            reasoning = review_response.get("reasoning", "")
            revised_code = review_response.get("revised_code")
            logger.info(f"Code review for node {node.id}: needs_revision={needs_revision}")
            logger.info(f"Reasoning: {reasoning}", extra={"verbose": True})

            if needs_revision:
                if revised_code and revised_code.strip():
                    if use_diff_for_review and (
                        "<<<<<<< SEARCH" in revised_code or "< SEARCH" in revised_code
                        ):
                        try:
                            logger.info("Code review returned diff format, applying patch")
                            patcher = SearchReplacePatcher()
                            patched_code, count = patcher.apply_patch(
                                revised_code, node.code, strict=False
                            )
                            if count > 0 and patched_code and patched_code != node.code:
                                post_audit = _deterministic_contract_audit(
                                    agent, patched_code
                                )
                                if not (
                                    post_audit
                                    and post_audit.get("violations")
                                ):
                                    logger.info(
                                        f"Successfully applied {count} review patch(es)"
                                    )
                                    return patched_code.strip()
                                logger.warning(
                                    "Reviewed code still violates deterministic Host audit: %s",
                                    "; ".join(post_audit["violations"]),
                                )
                                prompt["Instructions"][
                                    "DETERMINISTIC HOST CONTRACT AUDIT - HIGHEST PRIORITY"
                                ] = [
                                    "The previous review patch still failed the Host static checker. Return a complete replacement diff against the original code in this prompt.",
                                    "Exact remaining violations: "
                                    + "; ".join(post_audit["violations"]),
                                    "Metric non-improvement must select baseline validation/test predictions and continue through submission writing.",
                                ]
                                continue
                            logger.warning(
                                f"Diff patch failed (count={count}), keeping original code to avoid writing raw diff to runfile"
                            )
                            if deterministic_audit and deterministic_audit.get(
                                "violations"
                            ):
                                continue
                            return node.code
                        except Exception as e:
                            logger.warning(
                                f"Failed to apply diff patch in code review: {e}, keeping original code to avoid writing raw diff to runfile"
                            )
                            if deterministic_audit and deterministic_audit.get(
                                "violations"
                            ):
                                continue
                            return node.code
                    else:
                        # Full code revision (original behavior)
                        if not use_diff_for_review:
                            post_audit = _deterministic_contract_audit(
                                agent, revised_code
                            )
                            if post_audit and post_audit.get("violations"):
                                logger.warning(
                                    "Full review revision still violates deterministic Host audit: %s",
                                    "; ".join(post_audit["violations"]),
                                )
                                continue
                            logger.info("Using revised code from reviewer")
                            return revised_code.strip()

                if attempt < max_retries - 1:
                    logger.warning(f"Code review violation: needs_revision=True but revised_code is empty/None - Will retry ({attempt + 1}/{max_retries})")
                    logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                    continue
                logger.error(f"Code review violation: needs_revision=True but revised_code is empty/None - Max retries reached, returning original code")
                logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                if deterministic_audit and deterministic_audit.get("violations"):
                    raise ValueError(
                        "Code review could not repair deterministic Host violations: "
                        + "; ".join(deterministic_audit["violations"])
                    )
                return node.code

            if revised_code is not None and revised_code.strip():
                logger.warning(
                    "Code review warning: needs_revision=False but revised_code was provided. "
                    "Ignoring revised_code and using original code."
                )
            post_audit = _deterministic_contract_audit(agent, node.code)
            if post_audit and post_audit.get("violations"):
                if attempt < max_retries - 1:
                    logger.warning(
                        "Code reviewer approved code rejected by Host audit; retrying: %s",
                        "; ".join(post_audit["violations"]),
                    )
                    continue
                raise ValueError(
                    "Code reviewer repeatedly approved deterministic Host violations: "
                    + "; ".join(post_audit["violations"])
                )
            logger.info("Code approved, using original code")
            return node.code

        except Exception as e:
            error_msg = f"Code review failed with exception: {e}"
            if attempt < max_retries - 1:
                logger.warning(f"{error_msg} - Will retry (attempt {attempt + 1}/{max_retries})")
                continue
            if deterministic_audit and deterministic_audit.get("violations"):
                raise ValueError(
                    "Code review failed while deterministic Host violations remain: "
                    + "; ".join(deterministic_audit["violations"])
                ) from e
            logger.error(f"{error_msg} - Max retries reached, returning original code")
            return node.code

    if deterministic_audit and deterministic_audit.get("violations"):
        raise ValueError(
            "Code review exhausted retries while deterministic Host violations remain: "
            + "; ".join(deterministic_audit["violations"])
        )
    logger.error("Code review: Unexpected exit from retry loop, returning original code")
    return node.code
