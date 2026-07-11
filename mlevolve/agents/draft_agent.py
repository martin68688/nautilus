"""Draft Agent: initial plan and code draft."""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from llm import compile_prompt_to_md
from engine.search_node import SearchNode
from agents.coder import plan_and_code_query, stepwise_plan_and_code_query
from agents.triggers import register_node
from agents.memory.external_skill_memory import fetch_external_skill_memory, external_memory_section_title, external_memory_section_intro
from agents.prompts import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    MODEL_ARCHITECTURE_SAFETY,
    prompt_leakage_prevention,
    prompt_resp_fmt,
    get_prompt_environment,
    get_impl_guideline_from_agent,
)
from agents.planner import build_chat_prompt_for_model

logger = logging.getLogger("MLEvolve")


def run(
    agent,
    init_solution_path: Optional[str] = None,
    draft_role: Optional[str] = None,
) -> SearchNode:
    """Generate initial draft. If init_solution_path is provided and readable, use file content directly."""
    draft_role = agent.claim_draft_role(draft_role)

    if draft_role == "memory_reproduction":
        from agents.adoption import log_adoption
        from agents.memory.run_forest_replay import load_exact_replay

        replay = load_exact_replay(agent)
        agent.virtual_root.add_expected_child_count()
        new_node = SearchNode(
            plan=replay["plan"],
            code=replay["code"],
            parent=agent.virtual_root,
            stage="draft",
            local_best_node=agent.virtual_root,
            draft_role=draft_role,
            role_contract=replay["role_contract"],
            source_ref_ids=replay["source_ref_ids"],
            replay_source=replay["replay_source"],
            replay_status=replay["replay_status"],
            skip_code_review=True,
        )
        register_node(agent, new_node, replay["role_contract"], new_branch=True)
        log_adoption(
            new_node,
            agent,
            "run_forest_agentic_memory",
            replay["source_ref_ids"],
            "draft",
            adoption_mode=replay["adoption_mode"],
        )
        logger.info(
            "[draft] → node %s (branch=%s role=%s source=%s sha256=%s)",
            new_node.id,
            new_node.branch_id,
            draft_role,
            replay["replay_source"]["graph_node_id"],
            replay["replay_source"]["code_sha256"],
        )
        return new_node

    if init_solution_path:
        try:
            code = Path(init_solution_path).read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read init_solution from {init_solution_path}: {e}, falling back to LLM generation")
            init_solution_path = None
        else:
            plan = "User-provided init solution."
            agent.virtual_root.add_expected_child_count()
            new_node = SearchNode(
                plan=plan,
                code=code,
                parent=agent.virtual_root,
                stage="draft",
                local_best_node=agent.virtual_root,
                draft_role=draft_role,
                role_contract={"role": draft_role, "requirement": "Use the user-provided source code."},
                skip_code_review=True,
            )
            register_node(agent, new_node, "User-provided init solution (no LLM).", new_branch=True)
            logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id}) [init_solution]")
            return new_node

    professional_identity = (
        "🏆 You are a Kaggle Grandmaster - a top-tier ML expert competing to WIN.\n\n"
        "**Your Standards**:\n"
        "✓ Design complete ML pipelines (data → model → training → inference)\n"
        "✓ Implement real models that LEARN from data (not baseline scripts with constants)\n"
        "✓ Generate predictions through ACTUAL MODEL INFERENCE on each sample\n"
        "✓ Compete for TOP performance, not trivial baselines\n\n"
        "Your solution will be evaluated on a real leaderboard. Treat this with professionalism.\n\n"
    )

    introduction = (
        professional_identity +
        "Now, let's begin the competition. "
        "You need to come up with an excellent and creative plan for a competitive solution "
        "and then implement this solution in Python with the quality expected of a Kaggle Grandmaster. "
        "We will now provide a description of the task."
    )
    prompt: Any = {
        "Introduction": introduction,
        "Task description": agent.task_desc,
        "Memory": "" if draft_role == "coldstart_baseline" else agent.virtual_root.fetch_child_memory(),
        "Instructions": {},
    }
    prompt["Instructions"] |= prompt_resp_fmt()

    prompt["Instructions"] |= {
        "🔬 Critical: Scientific Approach to Design": [
            "",
            "Before designing your solution, you must answer three fundamental questions:",
            "",
            "1. **WHAT makes this task unique?**",
            "   - Not generic observations like 'it's a classification task'",
            "   - What SPECIFIC patterns, challenges, or domain characteristics?",

            "",
            "2. **WHY is your approach suitable for this task?**",
            "   - Not just 'this model is good' - explain the MATCH between approach and task",
            "   - What properties of your method address the task characteristics?",

            "",
            "3. **HOW will you validate your hypothesis?**",
            "   - What outcome would confirm your approach is right?",
            "   - What outcome would suggest you need to reconsider?",

            "",
            "---",
            "",
            "⚠️ This is not a template to fill - this is how scientists think.",
            "Blindly applying standard methods without understanding WHY is not acceptable.",
            "",
            "Your plan should naturally reflect this reasoning process.",
        ],
    }

    prompt["Instructions"] |= {
        "Solution sketch guideline": [
            "- Your plan should be concise but comprehensive: Must address WHAT/WHY/HOW (2-4 sentences each). Avoid verbosity - every sentence should add new insight. Natural length: around 8-12 sentences for a complete reasoning process.\n",
            "- Propose an evaluation metric that is reasonable for this task.\n",
            "- Don't suggest to do EDA.\n",
            "- The data is already prepared in `./input` directory. No need to unzip files.\n",
        ],
        "Coding & Execution Guidelines (CRITICAL)": [
            "- **NO PROGRESS BARS**: You MUST NOT use `tqdm`. Assume `tqdm` is not installed. Use standard Python loops only. Do not use `verbose=1`.",
            "- **MINIMAL LOGGING**: Print ONLY 1 line per epoch (e.g. loss/accuracy). Do NOT print batch-level logs.",
            "- **FINAL OUTPUT**: The VERY LAST line of execution MUST be `print(f'Final Validation Score: {score}')`. This is required for the score parser."
        ]
    }

    if draft_role == "coldstart_baseline":
        role_contract = {
            "role": draft_role,
            "requirement": (
                "Build a runnable baseline from only the first applicable original cold-start model template. "
                "Preserve its model name, checkpoint path, and loading API; adapt only task data, labels, "
                "training, validation, and submission code. Do not use RunForest memory."
            ),
            "primary_model": getattr(agent, "coldstart_primary_model_name", ""),
        }
        prompt["Instructions"]["Draft role contract (MANDATORY)"] = [role_contract["requirement"]]
    elif draft_role == "novel_exploration":
        role_contract = {
            "role": draft_role,
            "requirement": (
                "Explore a materially different hypothesis from the cold-start baseline, exact memory replay, "
                "and previous attempts. Novelty applies to this branch only. Complex pipelines and ensembles "
                "are allowed when justified by the task and resource budget."
            ),
        }
        prompt["Instructions"]["Draft role contract (MANDATORY)"] = [
            role_contract["requirement"],
            "Minor hyperparameter-only variations do not satisfy this role.",
        ]
    else:
        role_contract = {
            "role": draft_role,
            "requirement": "Design a competitive runnable solution without assuming another branch covers required components.",
        }
        prompt["Instructions"]["Draft role contract"] = [role_contract["requirement"]]
    prompt["Instructions"] |= get_impl_guideline_from_agent(agent)
    prompt["Instructions"] |= prompt_leakage_prevention()

    coldstart_description = (
        getattr(agent, "coldstart_primary_description", agent.coldstart_description)
        if draft_role == "coldstart_baseline"
        else agent.coldstart_description
    )
    if agent.use_coldstart and (coldstart_description != "None model"):
        coldstart_guideline = [
            f"""
            **Pretrained Model Strategy**:

            • **Option A [RECOMMENDED]**: {coldstart_description}
              → SOTA models with proven performance. Use for end-to-end fine-tuning OR as frozen feature extractors.

            • **Option B**: Alternative pretrained models if better suited to task characteristics.

            • **Option C**: Train from scratch / non-DL methods (only when pretraining provides no advantage).

            **CRITICAL: When using any recommended pretrained model (Option A), you MUST copy the Code template EXACTLY as provided — including model variant names, file paths, and checkpoint filenames. Only the listed weights are available locally; other variants will fail to load.**

            **Key Techniques**:
            1. **Feature Extractor Pattern**: If dataset is small or domain mismatch exists → Freeze backbone + train only final layers (or feed to XGBoost/SVM).

            2. **Mixed Precision (MANDATORY for pretrained models)**: Use `torch.cuda.amp` (autocast + GradScaler) to save memory. DO NOT manually convert to .half().

            3. **Avoid Timeouts**: #1 cause is slow data loading, NOT GPU model.
               • Use DataLoader with num_workers>=2, pin_memory=True (NOT raw for loops)
               • For large datasets + heavy backbones: Extract & cache features to disk (.npy/.h5)
            """
        ]
    else:
        coldstart_guideline = [""]

    prompt["Instructions"]["Implementation guideline"].extend(coldstart_guideline)
    prompt["Instructions"] |= get_prompt_environment()
    prompt["Instructions"] |= ROBUSTNESS_GENERALIZATION_STRATEGY
    prompt["Instructions"] |= MODEL_ARCHITECTURE_SAFETY

    if draft_role == "coldstart_baseline":
        external_skill_text, external_skill_ref_ids, external_skill_source = "", [], "run_forest_agentic_memory"
    else:
        external_skill_text, external_skill_ref_ids, external_skill_source = fetch_external_skill_memory(
            agent,
            "draft",
            run_memory=prompt.get("Memory", ""),
            data_preview=agent.data_preview or "",
            coldstart=getattr(agent, "coldstart_description", ""),
            draft_role=draft_role,
        )
    if external_skill_text:
        prompt["External Skill Memory"] = external_skill_text
    coldstart_external_text = getattr(agent, "coldstart_external_memory_text", "")
    if draft_role != "coldstart_baseline" and coldstart_external_text and str(coldstart_external_text).strip():
        existing_external = prompt.get("External Skill Memory", "")
        prompt["External Skill Memory"] = (
            f"{coldstart_external_text.strip()}\n\n"
            f"---\n## Run-Forest Runtime Draft Navigation\n"
            f"{existing_external.strip()}"
        ).strip() if existing_external.strip() else coldstart_external_text.strip()

    instructions = f"\n# Instructions\n\n"
    instructions += compile_prompt_to_md(prompt["Instructions"], 2)

    memory_section = ""
    if prompt.get("Memory", "").strip():
        memory_section = f"\n# Memory\nBelow is a record of previous solution attempts and their outcomes:\n {prompt['Memory']}\n"

    external_skill_section = ""
    if prompt.get("External Skill Memory", "").strip():
        section_title = external_memory_section_title(external_skill_source)
        section_intro = external_memory_section_intro(external_skill_source, "designing this node")
        external_skill_section = (
            f"\n# {section_title}\n"
            f"{section_intro}\n"
            f"{prompt['External Skill Memory']}\n"
        )

    user_prompt = f"\n# Task description\n{prompt['Task description']}{memory_section}{external_skill_section}\n{instructions}"
    assistant_prefix = f"Let me approach this systematically.\nFirst, I'll examine the dataset:\n{agent.data_preview}"
    prompt_complete = build_chat_prompt_for_model(
        agent.acfg.code.model, introduction, user_prompt, assistant_prefix
    )
    agent.virtual_root.add_expected_child_count()

    if agent.use_stepwise_generation:
        plan, code = stepwise_plan_and_code_query(
            agent_instance=agent,
            prompt_base=prompt,
            data_preview=agent.data_preview,
            context={
                "stage": "draft",
                "memory": prompt.get("Memory", ""),
                "draft_role": draft_role,
                "role_contract": role_contract,
            },
        )
    else:
        plan, code = plan_and_code_query(agent, prompt_complete)
    new_node = SearchNode(
        plan=plan,
        code=code,
        parent=agent.virtual_root,
        stage="draft",
        local_best_node=agent.virtual_root,
        draft_role=draft_role,
        role_contract=role_contract,
        source_ref_ids=list(external_skill_ref_ids),
    )
    register_node(agent, new_node, prompt_complete, new_branch=True)

    from agents.adoption import log_adoption
    log_adoption(new_node, agent, "methodology", getattr(agent, "methodology_ref_ids", []), "draft")
    if draft_role != "coldstart_baseline":
        log_adoption(
            new_node,
            agent,
            getattr(agent, "coldstart_external_source", "") or external_skill_source,
            getattr(agent, "coldstart_external_ref_ids", []),
            "coldstart",
        )
    log_adoption(new_node, agent, external_skill_source, external_skill_ref_ids, "draft")

    logger.info(f"[draft] → node {new_node.id} (branch={new_node.branch_id} role={draft_role})")
    return new_node
