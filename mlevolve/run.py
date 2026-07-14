import atexit
import logging
import os
import sys
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from engine.agent_search import AgentSearch as Agent
from engine.executor import Interpreter
from engine.search_node import Journal
from omegaconf import OmegaConf
from rich.status import Status
from config import load_task_desc, prep_agent_workspace, save_run, load_cfg
from utils.visualization import journal_to_string_tree
from utils.seed import set_global_seed
from engine.coldstart import build_guidance_description
from engine.run_control import (
    focused_protocol_status,
    focused_protocol_success_error,
    should_continue_focused_search,
)
from utils.logging_config import setup_logging
import torch



def run():
    cfg = load_cfg()
    if cfg.torch_hub_dir:
        torch.hub.set_dir(cfg.torch_hub_dir)
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    logger.info(f'Starting run "{cfg.exp_name}"')

    task_desc = load_task_desc(cfg)

    if cfg.coldstart.use_coldstart:
        logger.info("Loading guidance from knowledge base")
        cfg.coldstart.description = build_guidance_description(cfg, task_desc=task_desc)
        logger.info(f"Guidance description: {cfg.coldstart.description}")

    with Status("Preparing agent workspace (copying and extracting files) ..."):
        prep_agent_workspace(cfg)

    global_step = 0

    def cleanup():
        if global_step == 0:
            shutil.rmtree(cfg.workspace_dir)

    atexit.register(cleanup)

    journal = Journal()
    agent = Agent(
        task_desc=task_desc,
        cfg=cfg,
        journal=journal,
    )

    interpreter = Interpreter(
        cfg.workspace_dir, **OmegaConf.to_container(cfg.exec), cfg=cfg  # type: ignore
    )

    global_step = len(journal)
    status = Status("[green]Generating code...")

    def exec_callback(*args, **kwargs):
        status.update("[magenta]Executing code...")
        res = interpreter.run(*args, **kwargs)
        status.update("[green]Generating code...")
        return res

    def step_task(node=None, focused=False):
        if node:
            logger.info(f"[step_task] Processing node: {node.id}")
        else:
            logger.info(f"[step_task] Processing virtual root node.")
        return agent.step(
            exec_callback=exec_callback,
            node=node,
            mandatory_repair_role=dev_execution_role if focused else None,
            excluded_mandatory_repair_role=(
                dev_execution_role if dev_execution_role and not focused else None
            ),
        )

    max_workers = interpreter.max_parallel_run
    total_steps = cfg.agent.steps
    initial_draft_count = cfg.agent.initial_drafts
    strict_draft_roles = bool(getattr(getattr(cfg.agent, "draft_role_policy", None), "enabled", False))
    logger.info(f"🚀 ThreadPool max_workers set to: {max_workers} (matching interpreter capacity)")
    logger.info(f"🎯 Initial draft count: {initial_draft_count} (will be executed sequentially for diversity)")

    lock = threading.Lock()
    completed = 0

    dev_execution_role = os.environ.get("RUNFOREST_DEV_EXECUTION_ROLE", "").strip()
    draft_indices = list(range(min(initial_draft_count, total_steps)))
    if dev_execution_role:
        configured_roles = [
            agent.configured_draft_role(index)
            for index in range(initial_draft_count)
        ]
        if dev_execution_role not in configured_roles:
            raise RuntimeError(
                "RUNFOREST_DEV_EXECUTION_ROLE must name one configured Draft role; "
                f"role={dev_execution_role!r}, configured={configured_roles}"
            )
        target_index = configured_roles.index(dev_execution_role)
        agent._draft_generation_count = target_index
        draft_indices = [target_index]
        logger.warning(
            "DEV role filter active: generating and executing only role=%s (slot=%s)",
            dev_execution_role,
            target_index,
        )

    pending_draft_nodes = []

    def protocol_focus_status():
        return focused_protocol_status(agent.journal.nodes, dev_execution_role)
    if initial_draft_count > 0 and total_steps > 0:
        logger.info(f"📝 Phase 1: Sequential draft generation (code only, {initial_draft_count} drafts)")

        def step_task_generate_only(draft_idx):
            logger.info(f"[step_task_generate_only] Generating draft from virtual root")
            return agent.step(
                exec_callback=exec_callback,
                node=None,
                execute_immediately=False,
                draft_role=agent.configured_draft_role(draft_idx),
            )

        for draft_idx in draft_indices:
            try:
                logger.info(f"🔨 Generating draft {draft_idx + 1}/{min(initial_draft_count, total_steps)} (code only)")
                cur_node = step_task_generate_only(draft_idx)
                pending_draft_nodes.append(cur_node)
                logger.info(f"✅ Draft {draft_idx + 1} code generated: node.id={cur_node.id}, added to virtual_root.children")

            except Exception as e:
                logger.exception(f"❌ Exception during draft {draft_idx + 1} generation: {e}")
                if strict_draft_roles:
                    raise RuntimeError(
                        f"Fixed three-role Draft generation failed at slot {draft_idx}; refusing a partial root set"
                    ) from e

        logger.info(f"✅ Phase 1 complete: {len(pending_draft_nodes)} draft codes generated")

        if dev_execution_role:
            matching_nodes = [
                node for node in pending_draft_nodes
                if getattr(node, "draft_role", None) == dev_execution_role
            ]
            if len(matching_nodes) != 1:
                raise RuntimeError(
                    "RUNFOREST_DEV_EXECUTION_ROLE must match exactly one generated Draft; "
                    f"role={dev_execution_role!r}, matches={len(matching_nodes)}"
                )
            skipped_roles = [
                getattr(node, "draft_role", None)
                for node in pending_draft_nodes
                if node not in matching_nodes
            ]
            pending_draft_nodes = matching_nodes
            logger.warning(
                "DEV role filter active: executing only role=%s; skipped=%s",
                dev_execution_role,
                skipped_roles,
            )

    if pending_draft_nodes or completed < total_steps:
        logger.info(f"🚀 Phase 2: Pipelined parallel execution")
        logger.info(f"   - Pending draft executions: {len(pending_draft_nodes)}")
        logger.info(f"   - Remaining steps: {total_steps - completed}")

        def execute_draft_node(node):
            try:
                executed_node = agent.execute_deferred_node(node, exec_callback)
                logger.info(f"✅ Draft node {executed_node.id} executed: metric={executed_node.metric.value}")
                return executed_node
            except Exception as e:
                logger.exception(f"❌ Exception during draft node {node.id} execution: {e}")
                return None

        executor = ThreadPoolExecutor(max_workers=max_workers)
        interrupted = False
        try:
            futures = set()
            focus_futures = set()

            def submit_future(callable_, *args, focused=False):
                future = executor.submit(callable_, *args)
                futures.add(future)
                if focused:
                    focus_futures.add(future)
                return future

            for i, node in enumerate(pending_draft_nodes):
                submit_future(
                    execute_draft_node,
                    node,
                    focused=bool(dev_execution_role and node.draft_role == dev_execution_role),
                )
                logger.info(f"📤 Submitted draft execution: {node.id}")
                if i < len(pending_draft_nodes) - 1:
                    time.sleep(10)
                    logger.info(f"⏱️  Waiting 10s before next draft to stagger initialization...")

            initial_step_tasks = min(max_workers, total_steps - completed) - len(pending_draft_nodes)
            if initial_step_tasks > 0:
                for _ in range(initial_step_tasks):
                    submit_future(step_task)
                    logger.info(f"📤 Submitted initial step_task to fill thread pool")

            while (
                should_continue_focused_search(
                    completed_steps=completed,
                    total_steps=total_steps,
                    status=protocol_focus_status(),
                    focus_in_flight=bool(focus_futures),
                )
                if dev_execution_role
                else completed < total_steps
            ):
                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=1.0)

                if not done:
                    continue  # timeout, no completed futures, retry (allows SIGINT handling)

                # Process the focused lane first so its newly queued repair is
                # claimed before an ordinary worker asks for more work.
                for fut in sorted(done, key=lambda item: item not in focus_futures):
                    futures.remove(fut)
                    was_focused = fut in focus_futures
                    focus_futures.discard(fut)
                    try:
                        cur_node = fut.result()
                        if cur_node:
                            logger.info(f"✅ Task completed: node_id={cur_node.id}, step={cur_node.step}, is_buggy={cur_node.is_buggy}, metric={cur_node.metric.value if cur_node.metric else 'N/A'}")
                        else:
                            logger.warning(f"⚠️  Task returned None (execution failed)")
                    except Exception as e:
                        logger.exception(f"❌ Exception during task execution: {e}")
                        cur_node = None

                    with lock:
                        save_run(cfg, journal)
                        completed = len(journal) - 1  # Exclude virtual node
                        focus_status = protocol_focus_status()
                        if dev_execution_role and focus_status.completed:
                            logger.warning(
                                "DEV role filter completed its protocol transaction; stopping search"
                            )
                        if completed >= total_steps or focus_status.completed:
                            logger.info(journal_to_string_tree(journal))

                    within_shared_budget = completed + len(futures) < total_steps
                    continue_focused_replay = bool(
                        dev_execution_role
                        and focus_status.active
                        and was_focused
                        and not focus_futures
                    )
                    focus_has_finished = bool(
                        dev_execution_role and focus_status.seen and not focus_status.active
                    )
                    if not focus_has_finished and (within_shared_budget or continue_focused_replay):
                        next_is_focused = bool(
                            dev_execution_role
                            and (was_focused or getattr(cur_node, "draft_role", None) == dev_execution_role)
                        )
                        submit_future(
                            step_task,
                            cur_node,
                            next_is_focused,
                            focused=next_is_focused,
                        )
                        logger.info(f"📤 Submitted next task based on node {cur_node.id if cur_node else 'None'}")
                    logger.info(f"📊 Progress: {completed}/{total_steps} steps completed, {len(futures)} tasks running")
        except KeyboardInterrupt:
            interrupted = True
            logger.info("KeyboardInterrupt received, terminating subprocesses and shutting down...")
            interpreter.terminate_all_subprocesses()
            executor.shutdown(wait=False, cancel_futures=True) if sys.version_info >= (3, 9) else executor.shutdown(wait=False)
            raise
        finally:
            if not interrupted:
                executor.shutdown(wait=True)

        if dev_execution_role:
            focus_status = protocol_focus_status()
            focus_error = focused_protocol_success_error(focus_status)
            if focus_error:
                raise RuntimeError(
                    f"Focused replay role {dev_execution_role!r} did not complete cleanly: {focus_error}"
                )
    else:
        logger.info(f"✅ All steps completed in Phase 1 (total_steps={total_steps} <= initial_draft_count={initial_draft_count})")

    interpreter.cleanup_session(-1)

    try:
        from fixed_holdout.handoff import write_evaluation_request

        evaluation_request = write_evaluation_request(
            cfg,
            cfg.log_dir / "journal.json",
        )
        if evaluation_request is not None:
            logger.info(
                "Fixed-holdout search finished; external evaluation request: %s",
                evaluation_request,
            )
    except Exception as e:
        logger.error("Failed to write fixed-holdout evaluation request: %s", e)
        raise

    # Adoption tracking: post-run analysis (side-channel, never affects the run itself).
    # Only runs if adoption_tracking.enable + enable_analysis; failure is non-fatal.
    try:
        if getattr(cfg, "adoption_tracking", None) and cfg.adoption_tracking.enable and cfg.adoption_tracking.enable_analysis:
            from analysis.adoption_tracker import run_adoption_analysis
            run_adoption_analysis(cfg, journal)
    except Exception as e:
        logger.warning(f"[adoption_tracker] analysis skipped: {e}")


if __name__ == "__main__":    
    run()
