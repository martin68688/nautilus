"""Improve Agent: generate improved plan/code from a successful parent node (diff or full mode)."""

import copy
import logging
import time
from typing import Any

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from utils.response import wrap_code
from agents.triggers import get_patience_counter, register_node
from agents.leakage_audit import (
    build_repair_preservation_contract,
    format_audit,
    format_repair_preservation_contract,
)
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    MODEL_ARCHITECTURE_SAFETY,
    prompt_leakage_prevention,
    prompt_resp_fmt,
    get_internet_clarification,
    get_impl_guideline_from_agent,
    host_protocol_preflight_enabled,
)
from agents.planner import run_planner, generate_initial_plan, refine_plan_to_json, build_planner_task, build_planner_suffix, build_chat_prompt_for_model
from agents.coder import plan_and_code_query
from agents.coder.diff_coder import diff_generate_and_apply
from agents.memory.external_skill_memory import fetch_external_skill_memory, external_memory_section_title, external_memory_section_intro
from agents.memory_strategy_agent import payload_sha256, run_memory_strategy_shadow
from agents.strategy_actuation import (
    MemoryStrategyActuationRejected,
    active_strategy_enabled,
    active_strategy_required,
    run_active_strategy_actuation,
)

logger = logging.getLogger("MLEvolve")


def run(agent, parent_node: SearchNode) -> SearchNode:
    memory_layer = getattr(agent, "external_skill_memory", None)
    atomic_memory_actuation = bool(
        getattr(memory_layer, "experiment_r_atomic_actuation_enabled", False)
    )
    strategy_active = active_strategy_enabled(agent, "improve")
    improvement_standards = (
        "🎯 As a Grandmaster, make MEANINGFUL improvements that boost leaderboard performance.\n\n"
        "**Acceptable**: Advanced architectures, ensemble techniques, feature engineering, hyperparameter optimization, improved pipelines.\n"
        "**NOT Acceptable**: Cosmetic changes, minor tweaks without justification, breaking functionality.\n\n"
    )

    introduction = (
        improvement_standards +
        "You are provided with a previously developed solution below and should improve it "
        "in order to further increase the (test time) performance. "
        "For this you should first outline a brief plan in natural language for how the solution can be improved and "
        "then implement this improvement in Python based on the provided previous solution."
    )

    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Memory": parent_node.fetch_child_memory(include_code=False),
        "Instructions": {},
    }
    prompt["Previous solution"] = {
        "Code": wrap_code(parent_node.code),
    }
    if parent_node.draft_role:
        inherited = [
            f"This node originated from the `{parent_node.draft_role}` Draft branch.",
            "That role selected only the initial Draft origin. It does not restrict "
            "the Dynamic Router, memory visibility, or the method changes available now.",
        ]
        prompt["Instructions"]["Draft origin metadata"] = inherited
    if parent_node.leakage_audit and parent_node.leakage_audit.get("status") != "clean":
        repair_contract = {
            "LEAKAGE REPAIR CONTRACT - HIGHEST PRIORITY": [
                "This branch is repair-only. Fix every audited data-flow/evaluation issue before any optimization or novelty work.",
                "Preserve useful model and ensemble components; change split, fitting, selection, and reporting boundaries as required.",
                "Do not change model architecture, checkpoint identity, feature families, ensemble membership, loss, or optimizer. This is not an optimization pass.",
                format_repair_preservation_contract(
                    parent_node.leakage_repair_context.get("preservation_contract", {})
                    or build_repair_preservation_contract(parent_node.code)
                ),
                "The replacement code receives a fresh audit. The parent audit is evidence, not the child's verdict.",
                format_audit(parent_node.leakage_audit),
            ]
        }
        prompt["Instructions"] = repair_contract | prompt["Instructions"]

    success_patience, total_patience, branch_best_score = get_patience_counter(agent, parent_node)
    use_magnitude_prompt = (success_patience >= 2) or (total_patience >= 5)

    if use_magnitude_prompt:
        trigger_reason = []
        if success_patience >= 2:
            trigger_reason.append(f"success_patience={success_patience}>=2")
        if total_patience >= 5:
            trigger_reason.append(f"total_patience={total_patience}>=5")
        logger.warning(f"🔥 PLATEAU DETECTED! Triggered by: {' AND '.join(trigger_reason)}, using Magnitude-Based prompt")
        if branch_best_score is None:
            best_score_str = "N/A (no successful nodes yet)"
        else:
            best_score_str = f"{branch_best_score:.4f}"
        prompt["Instructions"] |= {
            "🔥 Improvement Strategy: Magnitude-Based Reasoning": [
                "",
                "⚠️ **CRITICAL: The current approach has hit a plateau.**",
                "",
                "Do NOT just tweak parameters unless we are very close to the target.",
                "Classify your thoughts into these 3 Tiers based on the **Magnitude of Change**:",
                "",
                "**Tier 1: Optimization (The \"How\")**",
                "- Definition: Keep the model architecture and data fixed. Only change *how* we train.",
                "- Scope: Hyperparameters, Learning Rate Schedulers, Random Seeds, Post-processing thresholds.",
                "- *When to use: We are fine-tuning a working solution.*",
                "",
                "**Tier 2: Representation & Components (The \"What\")**",
                "- Definition: Change specific modules of the pipeline, but keep the overall paradigm.",
                "- Scope:",
                "    - **Data**: New feature engineering, different augmentations, input normalization.",
                "    - **Model**: Swapping the backbone (e.g., larger model), changing the loss function, adding regularization layers (Dropout/BN).",
                "- *When to use: The current model is underfitting or overfitting.*",
                "",
                "**Tier 3: Systemic Paradigm Shift (The \"Architecture\")**",
                "- Definition: Fundamentally change the approach. The old code structure might need a rewrite.",
                "- Scope:",
                "    - **Paradigm**: Switching from GBDT to Neural Net (or vice versa), Single Model -> Ensemble.",
                "    - **Objective**: Changing from Regression to Classification (binning), Multi-task learning.",
                "    - **Data Flow**: Pseudo-labeling, Self-supervised pre-training.",
                "- *When to use: The current approach has hit a hard ceiling (plateau).*",
                "",
                "**Current Status**:",
                f"- Best Score: {best_score_str}",
                f"- Successful nodes without improvement: {success_patience}",
                f"- Total nodes since best: {total_patience} (including failed attempts)",
                "",
                "**⚠️ CRITICAL INSTRUCTION**:",
                f"The branch has stagnated (success_patience={success_patience}, total_patience={total_patience}).",
                "You MUST propose a **Tier 2 or Tier 3** change to break the plateau.",
                "Do NOT propose Tier 1 (hyperparameter tuning). The current approach needs a more fundamental change.",
                "",
                "You can refer to the expert technique suggestions above, which are distilled from the kaggle award-winning solutions.",
                "",
                "After deciding your Tier 2/3 strategy, briefly describe:",
                "- Which Tier you're using and why",
                "- What specific components will change",
                "- Why this addresses the root cause of the plateau",
            ],
        }
    else:
        prompt["Instructions"] |= {
            "🔬 Critical: Scientific Approach to Optimization": [
                "",
                "⚠️ **MANDATORY FORMAT REQUIREMENT**",
                "You MUST structure your plan using the following EXACT format:",
                "",
                "CHANGES (list ALL modifications, one or multiple):",
                "",
                "Change #1: [Category: Data Augmentation / Model Architecture / Loss Function / Optimization / Regularization / Training Strategy]",
                "- What: [Describe the SPECIFIC technical modification you will make]",
                "- Why: [Explain why THIS TASK needs this specific change]",
                "",
                "Change #2 (if applicable): [Category]",
                "- What: [Describe the SPECIFIC technical modification]",
                "- Why: [Explain why THIS TASK needs this specific change]",
                "",
                "[Add more changes if needed, but keep them focused and related]",
                "",
                "---",
                "",
                "WHY current solution limited:",
                "- Root cause: [Specific analysis, not just 'low performance']",
                "- Evidence: [Data/observation that supports your diagnosis]",
                "",
                "HOW these changes address it:",
                "- Mechanism: [Theoretical justification of WHY this should work]",
                "- Expected improvement: [Concrete prediction]",
                "- Synergy (if multiple changes): [How changes work together, if applicable]",
                "",
                "KEEP UNCHANGED (must explicitly list):",
                "- Random seed: [specify value, e.g., 42]",
                "- Data split: [must be identical to parent]",
                "- [List other key components that remain unchanged]",
                "",
                "⚠️ Plans that do not follow this structure will be considered invalid.",
                "",
                "---",
                "",
                "**Guidelines on Number of Changes**:",
                "",
                "- **Single change (Recommended for most cases)**: Best for establishing clear causality",
                "  Example: \"Add [specific augmentation technique]\" → easy to attribute performance change",
                "",
                "- **Multiple related changes (Acceptable)**: When changes naturally work together",
                "  Example: \"Change model architecture + adjust corresponding hyperparameters\"",
                "  (architecture changes often require optimizer/lr adjustments)",
                "",
                "- **Fusion scenario (Acceptable)**: Combining proven improvements from Memory",
                "  Example: \"Integrate [technique from Attempt #X] + [technique from Attempt #Y]\"",
                "  (both already validated separately in Memory)",
                "",
                "⚠️ **Key Principle**: Whether single or multiple changes, you MUST:",
                "1. Clearly list each specific change",
                "2. Explain the rationale for each",
                "3. Specify what stays the same for proper baseline comparison",
                "",
                "---",
                "",
                "**Explanation of Requirements**:",
                "",
                "1. **WHY is the current solution limited?**",
                "   - Not just 'performance is low' - what is the ROOT CAUSE?",
                "   - What EVIDENCE supports your diagnosis?",
                "",
                "2. **HOW will your changes address this root cause?**",
                "   - Not just 'try method X' - explain the MECHANISM",
                "   - Why should this work? What is the theoretical justification?",
                "",
                "3. **WHAT will you change, and what will stay the same?**",
                "   - List ALL changes explicitly (even if multiple)",
                "   - Keep other things identical for proper baseline comparison",
                "   - This enables understanding WHAT led to performance changes",
                "",
                "---",
                "",
                "⚠️ This structured format enables proper performance tracking and knowledge accumulation.",
                "Random trial-and-error without clear documentation is not acceptable.",
                "Others will learn from your reasoning and can replicate your improvements.",
            ],
        }

    prompt["Instructions"] |= {
        "Solution improvement guidelines": [
            "- Propose a single, specific, actionable improvement (atomic change for controlled experiment).\n",
            "- Your improvement must be distinctly different from existing attempts in the Memory section.\n",
            "",
            "⚠️ **IMPORTANT: Depth of Improvement**",
            "Consider TWO types of improvements (both are valid, but think about which is more appropriate):",
            "",
            "**Type A: Architectural Deepening (Often More Powerful)**",
            "- ADD components to existing model (e.g., add relevant mechanisms for this task)",
            "- MODIFY internal structure (e.g., enhance feature extraction for task characteristics)",
            "- DESIGN task-specific modules based on domain knowledge and data patterns",
            "Example: 'Keep current backbone, but ADD [mechanism] to address [specific task challenge]'",
            "",
            "**Type B: Model/Method Replacement (Simpler but may miss potential)**",
            "- REPLACE entire model/algorithm with a different approach",
            "- This is valid when current architecture is fundamentally unsuitable for this task",
            "- But ask yourself: Could I improve the current approach by adding/modifying instead of replacing?",
            "",
            "- Your plan should be concise but comprehensive: Must address WHY/HOW/WHAT (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.\n",
            "- Don't suggest to do EDA.\n",
        ],
    }

    if atomic_memory_actuation:
        prompt["Instructions"]["DYNAMIC MEMORY ATOMIC ACTUATION CONTRACT"] = [
            "Retrieved memories are alternatives, not a shopping list. Choose one primary causal hypothesis for this child.",
            "Modify one primary component; a second component is allowed only when it is an interface dependency of the same hypothesis.",
            "Do not combine a new backbone, a new tree model, a new fusion architecture, and new feature families in one child.",
            "When the parent is already competitive, preserve its model family, split, prediction variant, and working feature pipeline by default.",
            "Do not introduce a heavier pretrained backbone, extra cross-validation loop, or broad ensemble when it cannot finish comfortably within the remaining search budget.",
            "State the one selected memory ID or memory hypothesis you are actuating and explicitly list the other retrieved alternatives you are declining.",
        ]

    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= prompt_leakage_prevention()
    prompt["Instructions"] |= MODEL_ARCHITECTURE_SAFETY
    if not host_protocol_preflight_enabled(agent):
        internet_clarification = get_internet_clarification(
            getattr(agent.cfg, "pretrain_model_dir", "")
        )
        prompt["Instructions"]["Implementation guideline"].extend(
            internet_clarification
        )
    prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY

    output = wrap_code(parent_node.term_out, lang="")

    if not agent.acfg.use_diff_mode:
        prompt["Instructions"] |= prompt_resp_fmt()

    external_skill_text, external_skill_ref_ids, external_skill_source = fetch_external_skill_memory(
        agent,
        "improve",
        run_memory=prompt.get("Memory", ""),
        parent_plan=parent_node.plan or "",
        parent_code_summary=getattr(parent_node, "code_summary", "") or "",
        execution_output=parent_node.term_out or "",
        draft_role=getattr(parent_node, "draft_role", None),
        strategy_context={
            "selected_strategy": getattr(parent_node, "selected_strategy", {}) or {},
            "task_profile": getattr(parent_node, "task_profile", {}) or {},
        },
    )
    if external_skill_text:
        prompt["External Skill Memory"] = external_skill_text

    # Strategy receives the wider Router pack. Shadow mode remains a strict
    # side channel; active mode actuates only through the separate bounded
    # Atomic Planner/Coder transaction.
    strategy_prompt_snapshot = copy.deepcopy(prompt)
    strategy_prompt_sha256 = payload_sha256(strategy_prompt_snapshot)
    router_pack = (
        memory_layer.current_navigation_pack()
        if memory_layer is not None
        and callable(getattr(memory_layer, "current_navigation_pack", None))
        else {}
    )
    parent_node.add_expected_child_count()
    active_actuation_trace: dict[str, Any] = {}
    if strategy_active:
        active_actuation_trace = run_active_strategy_actuation(
            agent,
            parent_node,
            stage="improve",
            router_pack=router_pack,
            branch_best_metric=branch_best_score,
            production_prompt_sha256=strategy_prompt_sha256,
        )
        memory_strategy_trace = dict(active_actuation_trace.get("strategy") or {})
    else:
        memory_strategy_trace = run_memory_strategy_shadow(
            agent,
            parent_node,
            stage="improve",
            router_pack=router_pack,
            branch_best_metric=branch_best_score,
            production_prompt_sha256=strategy_prompt_sha256,
        )
    strategy_prompt_sha256_after = payload_sha256(prompt)
    if strategy_prompt_sha256_after != strategy_prompt_sha256:
        logger.error(
            "Memory Strategy shadow attempted to mutate the production Improve prompt; restoring snapshot"
        )
        prompt = strategy_prompt_snapshot
        memory_strategy_trace["status"] = "noninterference_violation"
        memory_strategy_trace["production_prompt_modified"] = True
    memory_strategy_trace["production_prompt_sha256_after"] = payload_sha256(prompt)
    memory_strategy_trace["noninterference_verified"] = bool(
        payload_sha256(prompt) == strategy_prompt_sha256
    )

    instructions = "\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    memory_section = ""
    if prompt.get("Memory", "").strip():
        memory_section = f"\n# Memory\nBelow is a record of previous improvement attempts and their outcomes:\n {prompt['Memory']}\n"

    external_skill_section = ""
    if prompt.get("External Skill Memory", "").strip():
        section_title = external_memory_section_title(external_skill_source)
        section_intro = external_memory_section_intro(external_skill_source, "proposing this improvement")
        external_skill_section = (
            f"\n# {section_title}\n"
            f"{section_intro}\n"
            f"{prompt['External Skill Memory']}\n"
        )

    user_prompt = f"\n# Task description\n{prompt['Task description']}{memory_section}{external_skill_section}\n{instructions}"
    assistant_prefix = f"Let me approach this systematically.\nFirst, I'll review the dataset:\n{agent.data_preview}\nThe current solution uses the following code:\n{prompt['Previous solution']['Code']}\nIts output was:\n{output}\nBuilding on this, I'll develop an improved approach."
    prompt_complete = build_chat_prompt_for_model(agent.acfg.code.model, introduction, user_prompt, assistant_prefix)

    if strategy_active and active_actuation_trace.get("status") == "accepted":
        plan = str(active_actuation_trace.get("plan_text") or "")
        code = str(active_actuation_trace.get("candidate_code") or "")
        prompt_complete = dict(active_actuation_trace.get("prompt_record") or {})
        logger.info(
            "Required Memory Strategy actuation accepted for Improve node %s",
            parent_node.id,
        )
    elif strategy_active and active_strategy_required(agent):
        parent_node.memory_strategy_trace = memory_strategy_trace
        parent_node.atomic_actuation_trace = dict(
            active_actuation_trace.get("atomic") or {}
        )
        if not isinstance(parent_node.protocol_observation, dict):
            parent_node.protocol_observation = {}
        parent_node.protocol_observation[
            "memory_strategy_active_rejection"
        ] = {
            "schema": "mlevolve_memory_strategy_active_rejection_v1",
            "stage": "improve",
            "reason": str(active_actuation_trace.get("reason") or "rejected"),
        }
        parent_node.is_terminal = True
        parent_node.continue_improve = False
        raise MemoryStrategyActuationRejected(active_actuation_trace)
    elif agent.acfg.use_diff_mode:
        try:
            logger.info(f"Using diff improve for node {parent_node.id}")
            plan, code = _diff_improve(agent, prompt, agent.data_preview, parent_node)
        except Exception as e:
            logger.warning(f"Diff improve failed: {e}, falling back to full rewrite")
            plan, code = plan_and_code_query(agent, prompt_complete)
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)

    from_topk = getattr(parent_node, '_topk_triggered', False)

    new_node = SearchNode(plan=plan, code=code, parent=parent_node, stage="improve",
                        local_best_node=parent_node.local_best_node, from_topk=from_topk)
    new_node.memory_strategy_trace = memory_strategy_trace
    if active_actuation_trace:
        new_node.atomic_actuation_trace = dict(
            active_actuation_trace.get("atomic") or {}
        )
        new_node.plan_diff_verdict = dict(
            active_actuation_trace.get("plan_diff_verdict") or {}
        )
    register_node(agent, new_node, prompt_complete, parent_node=parent_node)

    from agents.adoption import log_adoption
    log_adoption(new_node, agent, external_skill_source, external_skill_ref_ids, "improve")
    if active_actuation_trace.get("status") == "accepted":
        log_adoption(
            new_node,
            agent,
            "memory_strategy",
            list(active_actuation_trace.get("source_memory_ids") or []),
            "improve",
            adoption_mode="strategy_atomic_actuation",
        )
    if new_node.leakage_repair_context:
        log_adoption(
            new_node,
            agent,
            "leakage_failure_memory",
            [new_node.leakage_repair_context.get("source_node_id")],
            "improve",
            adoption_mode="mandatory_audit_repair",
        )

    if hasattr(parent_node, '_topk_triggered'):
        parent_node._topk_triggered = False

    logger.info(f"[improve] {parent_node.id} → node {new_node.id}")
    return new_node


# ============ Diff improve pipeline ============

_IMPROVE_STAGE_INTRO = (
    "Based on the task requirements, data characteristics, and execution results, carefully analyze "
    "the current solution to identify improvement opportunities that will enhance the final test set "
    "performance. Then select which component(s) to modify and provide detailed, actionable modification plans."
)
_IMPROVE_EXTRA_GUIDELINE = (
    "1. **Analyze the task description and data type carefully** before proposing enhancements. "
    "Your improvements must be based on the current task."
)
_IMPROVE_PLANNER_TASK = build_planner_task(_IMPROVE_STAGE_INTRO, _IMPROVE_EXTRA_GUIDELINE)

_IMPROVE_DIFF_INTRODUCTION = (
    "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
    "solution and a detailed improvement plan. Your task is to implement the improvement plan to enhance "
    "the solution's test set performance."
)


_IMPROVE_SUFFIX_EXTRA = (
    "Building on the current solution, I'll develop an improved approach "
    "that addresses identified limitations while preserving what works well."
)


def _diff_improve(agent, prompt_base, data_preview, parent_node):
    context = {
        "stage": "improve",
        "memory": prompt_base["Memory"],
        "previous_code": parent_node.code,
        "previous_code_summary": parent_node.code_summary if hasattr(parent_node, 'code_summary') and parent_node.code_summary else None,
        "execution_output": parent_node.term_out if hasattr(parent_node, 'term_out') else "",
        "parent_node": parent_node,
        "draft_role": parent_node.draft_role or "",
        "role_contract": parent_node.role_contract or {},
    }

    use_memory = (
        getattr(agent.acfg, 'use_global_memory', False)
        and agent.global_memory is not None
        and len(agent.global_memory.records) > 0
    )

    if use_memory:
        logger.info("[DiffImprove] Using two-stage planning with memory")
        initial_plan = generate_initial_plan(agent, prompt_base, data_preview, context)
        planning_result = refine_plan_to_json(agent, initial_plan, prompt_base, data_preview, context)
    else:
        logger.info("[DiffImprove] Using direct planner (memory disabled or empty)")
        planning_result = run_planner(
            agent_instance=agent,
            prompt_base=prompt_base,
            data_preview=data_preview,
            context=context,
            your_task_section=_IMPROVE_PLANNER_TASK,
            assistant_suffix=build_planner_suffix(prompt_base, data_preview, context, extra_text=_IMPROVE_SUFFIX_EXTRA),
            stage_name="ImprovePlanning",
        )

    modules = planning_result.get('module', [])
    plans = planning_result.get('plan', {})

    if not planning_result.get("parse_success", False):
        raise RuntimeError("Planner returned empty result after retries, triggering outer fallback")

    if not modules and plans:
        modules = list(plans.keys())
        planning_result['module'] = modules

    memory_layer = getattr(agent, "external_skill_memory", None)
    atomic_memory_actuation = bool(
        getattr(memory_layer, "experiment_r_atomic_actuation_enabled", False)
    )
    if atomic_memory_actuation:
        module_cap = int(
            getattr(memory_layer, "experiment_r_improve_max_modules", 2)
        )
        modules = list(planning_result.get("module") or [])
        if len(modules) > module_cap:
            kept = modules[:module_cap]
            planning_result["module"] = kept
            planning_result["plan"] = {
                key: value
                for key, value in plans.items()
                if key in set(kept)
            }
            planning_result["atomicity_enforced"] = {
                "status": "trimmed_to_module_cap",
                "module_cap": module_cap,
                "declined_modules": modules[module_cap:],
            }
            logger.warning(
                "[DiffImprove] Atomic actuation trimmed modules from %s to %s",
                modules,
                kept,
            )

    extra_user_sections = ""
    if prompt_base.get("External Skill Memory", "").strip():
        extra_user_sections = (
            "# External Skill Memory\n"
            "Use retrieved candidates only when their execution evidence matches; obey risk warnings and do not treat SOP-only references as proven recipes:\n"
            f"{prompt_base['External Skill Memory']}\n"
        )

    return diff_generate_and_apply(
        agent_instance=agent,
        planning_result=planning_result,
        parent_code=parent_node.code,
        data_preview=data_preview,
        execution_output=context["execution_output"],
        introduction=_IMPROVE_DIFF_INTRODUCTION,
        extra_user_sections=extra_user_sections,
        max_total_patches=(
            int(getattr(memory_layer, "experiment_r_improve_max_patches", 6))
            if atomic_memory_actuation
            else None
        ),
    )
