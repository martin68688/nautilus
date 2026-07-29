"""Code Review Agent: LLM-based code review and fix for node code."""

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
                "type": "string",
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
        contract = candidate_execution_contract_from_cfg(agent.cfg)
        if contract is not None:
            return audit_candidate_code(code, contract)
        preflight = getattr(getattr(agent, "acfg", None), "protocol_preflight", None)
        contract_path = str(getattr(preflight, "contract_path", "") or "")
        if preflight is not None and getattr(preflight, "enabled", False) and contract_path:
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
            return {
                "valid": not bool(issues),
                "violations": issues,
                "code_sha256": static_report.get("code_sha256"),
                "contract_hash": static_report.get("contract_hash"),
            }
        return None
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
                                logger.info(f"Successfully applied {count} review patch(es)")
                                return patched_code.strip()
                            logger.warning(
                                f"Diff patch failed (count={count}), keeping original code to avoid writing raw diff to runfile"
                            )
                            return node.code
                        except Exception as e:
                            logger.warning(
                                f"Failed to apply diff patch in code review: {e}, keeping original code to avoid writing raw diff to runfile"
                            )
                            return node.code
                    else:
                        # Full code revision (original behavior)
                        if use_diff_for_review:
                            return node.code
                        else:
                            logger.info("Using revised code from reviewer")
                            return revised_code.strip()

                if attempt < max_retries - 1:
                    logger.warning(f"Code review violation: needs_revision=True but revised_code is empty/None - Will retry ({attempt + 1}/{max_retries})")
                    logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                    continue
                logger.error(f"Code review violation: needs_revision=True but revised_code is empty/None - Max retries reached, returning original code")
                logger.info(f"Reasoning detail: {reasoning}", extra={"verbose": True})
                return node.code

            if revised_code is not None and revised_code.strip():
                logger.warning(
                    "Code review warning: needs_revision=False but revised_code was provided. "
                    "Ignoring revised_code and using original code."
                )
            logger.info("Code approved, using original code")
            return node.code

        except Exception as e:
            error_msg = f"Code review failed with exception: {e}"
            if attempt < max_retries - 1:
                logger.warning(f"{error_msg} - Will retry (attempt {attempt + 1}/{max_retries})")
                continue
            logger.error(f"{error_msg} - Max retries reached, returning original code")
            return node.code

    logger.error("Code review: Unexpected exit from retry loop, returning original code")
    return node.code
